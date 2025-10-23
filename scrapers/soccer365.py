import httpx
from typing import Any, Dict, List
from config import REQUESTS_TIMEOUT, USER_AGENT
from utils.network import new_client, fetch_url
from bs4 import BeautifulSoup
from datetime import datetime
from zoneinfo import ZoneInfo
from config import TIMEZONE
from scrapers.registry import register
import asyncio

HEADERS = {"User-Agent": USER_AGENT}

async def _parse_fixtures_from_html(html: str, default_date: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    soup = BeautifulSoup(html, "lxml")
    blocks = soup.select(".game_block, .game, .live_game")
    for b in blocks:
        try:
            tnode = b.select_one(".status, .time, .game_time")
            lnode = b.select_one(".tournament, .league, .league_title a")
            hnode = b.select_one(".team.home, .ht, .team1")
            anode = b.select_one(".team.away, .at, .team2")
            t = tnode.get_text(" ", strip=True) if tnode else ""
            lg = lnode.get_text(" ", strip=True) if lnode else ""
            h = hnode.get_text(" ", strip=True) if hnode else ""
            a = anode.get_text(" ", strip=True) if anode else ""
            if h and a:
                out.append({
                    "date": default_date,
                    "time": t,
                    "league": lg,
                    "home": h,
                    "away": a,
                })
        except Exception:
            continue
    return out

async def fetch_fixtures() -> List[Dict[str, Any]]:
    # Deepen coverage: crawl homepage and linked league pages (first level) with throttling
    items: List[Dict[str, Any]] = []
    url = "https://soccer365.ru/"
    date_str = datetime.now(ZoneInfo(TIMEZONE)).date().isoformat()
    try:
        async with new_client() as client:
            # Home page
            try:
                resp = await fetch_url(url, client=client, headers=HEADERS)
                if resp.status_code == 200:
                    items.extend(await _parse_fixtures_from_html(resp.text, date_str))
                else:
                    return []
            except Exception:
                return []
            # Try to follow league links present on homepage (limited to 25 to stay polite)
            try:
                soup = BeautifulSoup(resp.text, "lxml")
                links = []
                for a in soup.select(".league_title a[href], .tournament a[href], a.league[href]"):
                    href = a.get("href") or ""
                    if href and href.startswith("/"):
                        links.append("https://soccer365.ru" + href)
                seen = set()
                for lk in links[:25]:
                    if lk in seen:
                        continue
                    seen.add(lk)
                    await asyncio.sleep(0.4)
                    try:
                        r2 = await fetch_url(lk, client=client, headers=HEADERS)
                        if r2.status_code == 200:
                            items.extend(await _parse_fixtures_from_html(r2.text, date_str))
                    except Exception:
                        continue
            except Exception:
                pass
    except Exception:
        return []
    return items

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
