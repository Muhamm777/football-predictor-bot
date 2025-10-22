import httpx
from typing import Any, Dict, List
from config import REQUESTS_TIMEOUT, USER_AGENT
from storage.db import get_today_fixtures

HEADERS = {"User-Agent": USER_AGENT}

async def fetch_team_stats() -> List[Dict[str, Any]]:
    # Placeholder implementation (no external scraping yet):
    # Build simple form summaries for teams present in today's fixtures so that
    # the pipeline can display "Форма: дом ..., выезд ..." reasons in Mini App.
    try:
        fixtures = get_today_fixtures()
    except Exception:
        fixtures = []
    teams = set()
    for f in fixtures:
        if f.get("home"):
            teams.add(f["home"])
        if f.get("away"):
            teams.add(f["away"])
    out: List[Dict[str, Any]] = []
    for t in teams:
        out.append({
            "team": t,
            # Simple deterministic placeholders to avoid randomness
            "home_wdl": "3-1-1",
            "away_wdl": "2-2-1",
            "gf": 7,
            "ga": 4,
        })
    return out
