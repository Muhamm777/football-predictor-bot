from typing import Any, Dict, List
import time
import os
import logging
from scrapers.registry import register
from storage.db import get_today_fixtures, upsert_fixtures
from utils.normalizer import normalize_team, normalize_league
from datetime import datetime, timezone

# We reuse the free package's simple ML engine
try:
    from free.prediction_engine import FootballPredictionEngine  # type: ignore
except Exception:  # pragma: no cover
    FootballPredictionEngine = None  # type: ignore

_engine = None
_model_paths = [
    "football_model.pkl",
    "free/football_model.pkl",
    "./free/football_model.pkl",
]

# Simple TTL cache to avoid recomputing within short intervals
_cache_ts: float = 0.0
_cache_data: List[Dict[str, Any]] = []
_CACHE_TTL_SEC = int(os.environ.get("FREE_ENGINE_CACHE_TTL", "300") or "300")
_MAX_ITEMS = int(os.environ.get("FREE_ENGINE_MAX", "60") or "60")
_LIVE_FALLBACK = (os.environ.get("FREE_ENGINE_LIVE_FALLBACK", "1") or "1") not in {"0","false","False"}
_TEMP = float(os.environ.get("FREE_ENGINE_TEMP", "1.0") or "1.0")
_EPS = float(os.environ.get("FREE_ENGINE_EPS", "1e-6") or "1e-6")
_cache_key: set[tuple[str,str]] = set()
_PERSIST_FALLBACK = (os.environ.get("FREE_ENGINE_PERSIST_FALLBACK", "1") or "1") not in {"0","false","False"}
_last_stats: Dict[str, Any] = {}


def _get_engine():
    global _engine
    # If engine exists but is untrained, try to load model now
    if _engine is not None:
        try:
            if not getattr(_engine, "is_trained", False):
                for p in _model_paths:
                    try:
                        import os
                        if os.path.exists(p):
                            _engine.load_model(p)
                            try:
                                setattr(_engine, "is_trained", True)
                            except Exception:
                                pass
                            return _engine
                    except Exception:
                        continue
        except Exception:
            pass
        return _engine
    if FootballPredictionEngine is None:
        return None
    try:
        eng = FootballPredictionEngine()
        # try load a pre-trained model if available
        for p in _model_paths:
            try:
                import os
                if os.path.exists(p):
                    eng.load_model(p)
                    try:
                        setattr(eng, "is_trained", True)
                    except Exception:
                        pass
                    _engine = eng
                    return _engine
            except Exception:
                continue
        # If no model available, leave untrained (provider will noop)
        _engine = eng
        return _engine
    except Exception:
        return None


async def odds_or_prob() -> List[Dict[str, Any]]:
    # Return cached data if fresh and fixtures unchanged
    now = time.time()
    eng = _get_engine()
    if eng is None:
        return []
    # Ensure model is loaded if not yet trained (e.g., engine was cached before training)
    if not getattr(eng, "is_trained", False):
        for p in _model_paths:
            try:
                import os
                if os.path.exists(p):
                    eng.load_model(p)
                    try:
                        setattr(eng, "is_trained", True)
                    except Exception:
                        pass
                    break
            except Exception:
                continue
    try:
        fixtures = get_today_fixtures() or []
    except Exception:
        fixtures = []
    # Fallback to live fixtures scraping if DB is empty
    source = "db"
    if not fixtures and _LIVE_FALLBACK:
        try:
            from scrapers.soccer365 import fetch_fixtures as _fetch_live  # type: ignore
            try:
                live = await _fetch_live()
            except Exception:
                live = []
            if isinstance(live, list) and live:
                fixtures = live
                source = "live"
        except Exception:
            pass
    # Build filtered list of valid fixtures first
    def _ok(name: str) -> bool:
        if not name:
            return False
        # normalize basic artifacts
        name = name.replace("\u2014", "-").replace("\u2013", "-").replace("—", "-")
        name = " ".join(name.split())
        low = name.lower().strip()
        if low in {"vs", "v", "-", "—", "tbd", "unknown"}:
            return False
        # relax: allow names with at least 2 visible chars and not only punctuation
        return sum(1 for ch in name if ch.isalnum()) >= 2
    filtered: List[Dict[str, Any]] = []
    seen = set()
    db_total = len(fixtures or [])
    skipped_invalid = 0
    for f in fixtures:
        home = normalize_team((f.get("home") or "").strip())
        away = normalize_team((f.get("away") or "").strip())
        if _ok(home) and _ok(away):
            key = (home.lower(), away.lower())
            if key in seen:
                continue
            seen.add(key)
            filtered.append({
                "home": home,
                "away": away,
                "competition": normalize_league((f.get("league") or f.get("competition") or "").strip() or "General")
            })
        else:
            skipped_invalid += 1
    # If nothing valid, try live fetch as fallback
    live_total = 0
    if not filtered and _LIVE_FALLBACK:
        try:
            from scrapers.soccer365 import fetch_fixtures as _fetch_live  # type: ignore
            live = await _fetch_live()
            live_total = len(live or [])
            for f in live or []:
                home = (f.get("home") or "").strip()
                away = (f.get("away") or "").strip()
                if _ok(home) and _ok(away):
                    key = (home.lower(), away.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    filtered.append({
                        "home": normalize_team(home),
                        "away": normalize_team(away),
                        "competition": normalize_league((f.get("league") or f.get("competition") or "").strip() or "General")
                    })
            if live_total > 0:
                source = "live"
            if _PERSIST_FALLBACK and (live or []):
                try:
                    to_save = []
                    for f in live or []:
                        h = (f.get("home") or "").strip()
                        a = (f.get("away") or "").strip()
                        if not (_ok(h) and _ok(a)):
                            continue
                        to_save.append({
                            "date": (f.get("date") or ""),
                            "league": normalize_league((f.get("league") or "")),
                            "home": normalize_team(h),
                            "away": normalize_team(a),
                        })
                    if to_save:
                        upsert_fixtures(to_save)
                except Exception:
                    pass
        except Exception:
            pass
    # Cache key based on fixture pairs
    new_cache_key = {(x["home"].lower(), x["away"].lower()) for x in filtered}
    global _cache_data, _cache_ts, _cache_key
    if _cache_data and (now - _cache_ts) < _CACHE_TTL_SEC and new_cache_key == _cache_key:
        return _cache_data
    out: List[Dict[str, Any]] = []
    count = 0
    skipped_zero = 0
    skipped_fail = 0
    for f in filtered:
        try:
            pred = eng.predict_match(f["home"], f["away"], f["competition"])
            probs_src = pred.get("probabilities") or {}
            h = float(probs_src.get("HOME_WIN", 0.0) or 0.0)
            d = float(probs_src.get("DRAW", 0.0) or 0.0)
            a = float(probs_src.get("AWAY_WIN", 0.0) or 0.0)
            s = h + d + a
            if s <= 0:
                try:
                    logging.debug("free_engine: zero-sum probabilities skipped %s vs %s", f["home"], f["away"])
                except Exception:
                    pass
                skipped_zero += 1
                continue
            # normalize, apply epsilon floor and optional temperature scaling
            h, d, a = h / s, d / s, a / s
            h = max(_EPS, min(1.0 - 2*_EPS, h))
            d = max(_EPS, min(1.0 - 2*_EPS, d))
            a = max(_EPS, min(1.0 - 2*_EPS, a))
            if _TEMP and _TEMP != 1.0:
                t = max(0.1, min(10.0, _TEMP))
                ph = (h + _EPS) ** (1.0 / t)
                pd = (d + _EPS) ** (1.0 / t)
                pa = (a + _EPS) ** (1.0 / t)
                st = ph + pd + pa
                h, d, a = ph / st, pd / st, pa / st
            probs = {"home": h, "draw": d, "away": a}
            out.append({"home": f["home"], "away": f["away"], "league": f.get("competition", "General"), "probs": probs})
            count += 1
            if count >= _MAX_ITEMS:
                break
        except Exception:
            try:
                logging.debug("free_engine: prediction failed for %s vs %s", f.get("home"), f.get("away"))
            except Exception:
                pass
            skipped_fail += 1
            continue
    # Save to cache
    _cache_ts = time.time()
    _cache_data = out
    _cache_key = new_cache_key
    # diagnostics
    try:
        _last_stats.update({
            "db_total": db_total,
            "filtered_valid": len(filtered),
            "live_total": live_total,
            "final_count": len(out),
            "skipped_invalid": skipped_invalid,
            "skipped_zero": skipped_zero,
            "skipped_fail": skipped_fail,
            "source": source,
            "cache_time_iso": datetime.now(timezone.utc).isoformat(),
            "cache_age_sec": int(max(0, now - _cache_ts)),
        })
    except Exception:
        pass
    return _cache_data


# Register on import
try:
    register(
        name="free_engine",
        role="external_model",
        fetch={"odds_or_prob": odds_or_prob},
        enabled=True,
        notes="Free ML prediction engine probabilities (requires pre-trained model)",
    )
except Exception:
    pass
