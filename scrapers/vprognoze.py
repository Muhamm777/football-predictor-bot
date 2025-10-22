import httpx
from typing import Any, Dict, List
from config import REQUESTS_TIMEOUT, USER_AGENT
from storage.db import get_today_fixtures

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
