import httpx
from typing import Any, Dict, List
from config import REQUESTS_TIMEOUT, USER_AGENT
from utils.network import new_client, fetch_url
from bs4 import BeautifulSoup
from datetime import datetime
from zoneinfo import ZoneInfo
from config import TIMEZONE
from scrapers.registry import register

HEADERS = {"User-Agent": USER_AGENT}

async def fetch_fixtures() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    url = "https://soccer365.ru/"
    try:
        async with new_client() as client:
            resp = await fetch_url(url, client=client, headers=HEADERS)
            if resp.status_code != 200:
                return []
            html = resp.text
    except Exception:
        return []

    try:
        soup = BeautifulSoup(html, "lxml")
        blocks = soup.select(".game_block, .game, .live_game")
        # Determine today's date in configured timezone
        date_str = datetime.now(ZoneInfo(TIMEZONE)).date().isoformat()
        for b in blocks:
            t = (b.select_one(".status, .time, .game_time") or {}).get_text(" ", strip=True) if hasattr(b.select_one(".status, .time, .game_time"), 'get_text') else ""
            lg = (b.select_one(".tournament, .league, .league_title a") or {}).get_text(" ", strip=True) if hasattr(b.select_one(".tournament, .league, .league_title a"), 'get_text') else ""
            h = (b.select_one(".team.home, .ht, .team1") or {}).get_text(" ", strip=True) if hasattr(b.select_one(".team.home, .ht, .team1"), 'get_text') else ""
            a = (b.select_one(".team.away, .at, .team2") or {}).get_text(" ", strip=True) if hasattr(b.select_one(".team.away, .at, .team2"), 'get_text') else ""
            if h and a:
                items.append({
                    "date": date_str,
                    "time": t,
                    "league": lg,
                    "home": h,
                    "away": a,
                })
        return items
    except Exception:
        return []

async def fetch_stats() -> List[Dict[str, Any]]:
    # TODO: implement real parsing
    return []

# Self-register as a fixtures provider so scheduler can merge via registry
try:
    register(
        name="soccer365",
        role="fixtures",
        fetch={"fixtures": fetch_fixtures},
        enabled=True,
        notes="Basic fixtures parser (home/away/league/time)"
    )
except Exception:
    # registry is optional; ignore if not available at import time
    pass
