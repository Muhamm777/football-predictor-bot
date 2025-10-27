import httpx
from typing import Any, Dict, List
from config import REQUESTS_TIMEOUT, USER_AGENT
from storage.db import get_today_fixtures
from scrapers.registry import register

HEADERS = {"User-Agent": USER_AGENT}

async def fetch_comments() -> List[Dict[str, Any]]:
    # Placeholder implementation (no external scraping yet):
    # Build simple sentiment for today's fixture pairs so the pipeline can
    # show "Сентимент" reason in Mini App even without real comments.
    try:
        fixtures = get_today_fixtures()
    except Exception:
        fixtures = []
    out: List[Dict[str, Any]] = []
    for f in fixtures:
        home = f.get("home")
        away = f.get("away")
        if not home or not away:
            continue
        out.append({
            "home": home,
            "away": away,
            # Slightly positive placeholder sentiment
            "sentiment": 0.3,
        })
    return out


async def odds_or_prob() -> List[Dict[str, Any]]:
    """Map sentiment to 1X2-like probabilities (auxiliary signal).
    sentiment in [-1..1] -> adjust base probabilities slightly towards HOME/AWAY.
    """
    comments = await fetch_comments()
    out: List[Dict[str, Any]] = []
    for c in comments:
        try:
            home = (c.get("home") or "").strip()
            away = (c.get("away") or "").strip()
            s = float(c.get("sentiment") or 0.0)
            if not (home and away):
                continue
            # Base neutral distribution
            ph, pd, pa = 0.40, 0.24, 0.36
            # Shift by sentiment: positive -> home, negative -> away
            delta = max(-0.08, min(0.08, s * 0.08))
            ph = max(0.05, min(0.85, ph + delta))
            pa = max(0.05, min(0.85, pa - delta))
            # Normalize with fixed draw weight window
            pd = max(0.18, min(0.32, pd))
            ssum = ph + pd + pa
            ph, pd, pa = ph/ssum, pd/ssum, pa/ssum
            out.append({
                "home": home,
                "away": away,
                "probs": {"home": ph, "draw": pd, "away": pa},
            })
        except Exception:
            continue
    return out


try:
    register(
        name="vprognoze",
        role="sentiment_model",
        fetch={"comments": fetch_comments, "odds_or_prob": odds_or_prob},
        enabled=True,
        notes="Maps sentiment to 1X2 probabilities (auxiliary).",
    )
except Exception:
    pass
