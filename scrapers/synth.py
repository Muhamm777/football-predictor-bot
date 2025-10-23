from typing import Any, Dict, List
from scrapers.registry import register
from storage.db import get_today_fixtures
from scrapers.sstats import fetch_team_stats
from predictor.poisson import match_probabilities

# Lightweight synthetic probabilities provider
# Uses today's fixtures + simple team form stats to produce 1X2 probs

async def odds_or_prob() -> List[Dict[str, Any]]:
    try:
        fixtures = get_today_fixtures()
    except Exception:
        fixtures = []
    # Build simple form map once
    form_map: Dict[str, Dict[str, Any]] = {}
    try:
        stats = await fetch_team_stats()
        for s in stats or []:
            team = (s.get("team") or "").strip()
            if not team:
                continue
            form_map[team.lower()] = {
                "gf": s.get("gf"),
                "ga": s.get("ga"),
                "home_wdl": s.get("home_wdl"),
                "away_wdl": s.get("away_wdl"),
            }
    except Exception:
        pass
    out: List[Dict[str, Any]] = []
    for f in fixtures or []:
        home = (f.get("home") or "").strip()
        away = (f.get("away") or "").strip()
        if not home or not away:
            continue
        stats_payload = {
            "home": form_map.get(home.lower(), {}),
            "away": form_map.get(away.lower(), {}),
        }
        try:
            probs = match_probabilities(stats_payload)
            # Validate and normalize
            if not isinstance(probs, dict):
                continue
            h = float(probs.get("home", 0.0) or 0.0)
            d = float(probs.get("draw", 0.0) or 0.0)
            a = float(probs.get("away", 0.0) or 0.0)
            s = h + d + a
            if s <= 0:
                continue
            probs = {"home": h / s, "draw": d / s, "away": a / s}
            out.append({"home": home, "away": away, "probs": probs})
        except Exception:
            continue
    return out

# Register on import
try:
    register(
        name="synth",
        role="external_model",
        fetch={"odds_or_prob": odds_or_prob},
        enabled=True,
        notes="Synthetic 1X2 probabilities from Poisson + simple form"
    )
except Exception:
    pass
