from typing import Any, Dict, List
import os
import time
from scrapers.registry import register

# Reuse free_engine probabilities and convert to synthetic bookmaker odds
try:
    from scrapers import free_engine as free_engine_provider  # type: ignore
except Exception:  # pragma: no cover
    free_engine_provider = None  # type: ignore

_MARGIN = float(os.environ.get("FREE_SYNTH_ODDS_MARGIN", "0.06") or "0.06")
_TTL = int(os.environ.get("FREE_SYNTH_ODDS_TTL", "180") or "180")
_cache_ts: float = 0.0
_cache_data: List[Dict[str, Any]] = []


def _to_odds(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(p or 0.0)))
    return 1.0 / p


def _apply_margin(h: float, d: float, a: float) -> tuple[float, float, float]:
    # Scale probabilities to sum to (1 + margin), then convert to odds
    margin = max(0.0, min(0.25, _MARGIN))
    scale = 1.0 + margin
    ph, pd, pa = h * scale, d * scale, a * scale
    s = ph + pd + pa
    if s <= 0:
        # fallback to uniform with margin
        ph = pd = pa = scale / 3.0
        s = ph + pd + pa
    ph, pd, pa = ph / s * scale, pd / s * scale, pa / s * scale
    return _to_odds(ph), _to_odds(pd), _to_odds(pa)


async def odds_or_prob() -> List[Dict[str, Any]]:
    global _cache_ts, _cache_data
    now = time.time()
    if _cache_data and (now - _cache_ts) < _TTL:
        return _cache_data
    out: List[Dict[str, Any]] = []
    if free_engine_provider is None:
        return out
    try:
        probs_arr = await free_engine_provider.odds_or_prob()
    except Exception:
        probs_arr = []
    for item in probs_arr or []:
        try:
            home = (item.get("home") or "").strip()
            away = (item.get("away") or "").strip()
            pr = item.get("probs") or {}
            h = float(pr.get("home", 0.0) or 0.0)
            d = float(pr.get("draw", 0.0) or 0.0)
            a = float(pr.get("away", 0.0) or 0.0)
            s = h + d + a
            if not home or not away or s <= 0:
                continue
            h, d, a = h / s, d / s, a / s
            oh, od, oa = _apply_margin(h, d, a)
            out.append({
                "home": home,
                "away": away,
                "probs": {"home": h, "draw": d, "away": a},
                "odds": {"home": oh, "draw": od, "away": oa},
            })
        except Exception:
            continue
    _cache_ts = time.time()
    _cache_data = out
    return _cache_data


try:
    register(
        name="free_synth_odds",
        role="odds",
        fetch={"odds_or_prob": odds_or_prob},
        enabled=True,
        notes="Synthetic odds derived from free_engine probabilities with margin",
    )
except Exception:
    pass
