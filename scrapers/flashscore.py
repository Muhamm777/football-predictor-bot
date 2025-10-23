from typing import Any, Dict, List
import os
import time
import asyncio
import httpx
from bs4 import BeautifulSoup
from config import USER_AGENT
from utils.network import new_client, fetch_url
from scrapers.registry import register

HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
CACHE_TTL = int(os.environ.get("FLASH_CACHE_TTL_SEC", "600") or "600")
_cache_ts: float = 0.0
_cache_data: List[Dict[str, Any]] = []

def _now() -> float:
    return time.time()

def _parse_list_html(html: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    soup = BeautifulSoup(html, "lxml")
    for m in soup.select("div.event__match")[:120]:
        try:
            hnode = m.select_one("div.event__participant--home")
            anode = m.select_one("div.event__participant--away")
            home = (hnode.get_text(" ", strip=True) if hnode else "").strip()
            away = (anode.get_text(" ", strip=True) if anode else "").strip()
            if not home or not away:
                continue
            # Odds (if present in listing; sometimes absent without JS)
            def _flt(x):
                try:
                    return float(x)
                except Exception:
                    return None
            oh = m.select_one(".oddsCell__odd--home")
            od = m.select_one(".oddsCell__odd--draw")
            oa = m.select_one(".oddsCell__odd--away")
            h = _flt(oh.get_text(strip=True)) if oh else None
            d = _flt(od.get_text(strip=True)) if od else None
            a = _flt(oa.get_text(strip=True)) if oa else None
            rec = {"home": home, "away": away}
            if h and d and a:
                rec.update({"h": h, "d": d, "a": a})
            out.append(rec)
        except Exception:
            continue
    return out

async def fetch_odds() -> List[Dict[str, Any]]:
    global _cache_ts, _cache_data
    # Serve from cache if fresh
    if _cache_data and (_now() - _cache_ts) < CACHE_TTL:
        return _cache_data
    urls = [
        "https://www.flashscore.com/football/",
        "https://www.flashscore.com.ua/football/",
    ]
    collected: List[Dict[str, Any]] = []
    try:
        async with new_client() as client:
            for i, url in enumerate(urls):
                try:
                    resp = await fetch_url(url, client=client, headers=HEADERS)
                    if resp.status_code == 200 and resp.text:
                        collected.extend(_parse_list_html(resp.text))
                except Exception:
                    continue
                await asyncio.sleep(0.4 + 0.2*(i%2))
    except Exception:
        return []
    # Deduplicate by (home,away)
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in collected:
        key = (r.get("home",""), r.get("away",""))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    _cache_data = out
    _cache_ts = _now()
    return out

# Also register as an odds_or_prob provider for the registry-based merge
try:
    register(
        name="flashscore",
        role="odds",
        fetch={"odds_or_prob": fetch_odds},
        enabled=True,
        notes="Flashscore 1X2 odds via httpx parser (best-effort; may be partial without JS)"
    )
except Exception:
    pass
