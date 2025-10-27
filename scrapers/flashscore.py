from typing import Any, Dict, List
import os
import time
import asyncio
import logging
import httpx
from bs4 import BeautifulSoup
from config import USER_AGENT
from utils.network import new_client, fetch_url
from utils.normalizer import normalize_team
from scrapers.registry import register

HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
CACHE_TTL = int(os.environ.get("FLASH_CACHE_TTL_SEC", "600") or "600")
_cache_ts: float = 0.0
_cache_data: List[Dict[str, Any]] = []
_daily_keys: set[str] = set()
_daily_date: str | None = None

def _now() -> float:
    return time.time()

def _parse_list_html(html: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    soup = BeautifulSoup(html, "lxml")
    for m in soup.select("div.event__match")[:120]:
        try:
            hnode = m.select_one("div.event__participant--home")
            anode = m.select_one("div.event__participant--away")
            home = normalize_team((hnode.get_text(" ", strip=True) if hnode else "").strip())
            away = normalize_team((anode.get_text(" ", strip=True) if anode else "").strip())
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
        "https://www.flashscore.com/en/football/",
        "https://www.flashscore.com.ua/football/",
        # League pages to increase coverage
        "https://www.flashscore.com/football/england/premier-league/",
        "https://www.flashscore.com/football/england/championship/",
        "https://www.flashscore.com/football/spain/laliga/",
        "https://www.flashscore.com/football/italy/serie-a/",
        "https://www.flashscore.com/football/italy/serie-b/",
        "https://www.flashscore.com/football/germany/bundesliga/",
        "https://www.flashscore.com/football/france/ligue-1/",
        "https://www.flashscore.com/football/portugal/primeira-liga/",
        "https://www.flashscore.com/football/netherlands/eredivisie/",
        "https://www.flashscore.com/football/netherlands/eerste-divisie/",
        "https://www.flashscore.com/football/belgium/jupiler-pro-league/",
        "https://www.flashscore.com/football/turkey/super-lig/",
    ]
    collected: List[Dict[str, Any]] = []
    try:
        async with new_client() as client:
            for i, url in enumerate(urls):
                backoff = 0.5
                for attempt in range(3):
                    try:
                        resp = await fetch_url(url, client=client, headers=HEADERS)
                        if resp.status_code == 200 and resp.text:
                            collected.extend(_parse_list_html(resp.text))
                            break
                    except Exception:
                        pass
                    await asyncio.sleep(backoff)
                    backoff *= 2
                await asyncio.sleep(0.4 + 0.2*(i%2))
    except Exception:
        return []
    # Deduplicate by (home,away) and persist keys for the current UTC day
    seen = set()
    out: List[Dict[str, Any]] = []
    try:
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).date().isoformat()
    except Exception:
        today = None
    for r in collected:
        key = (r.get("home",""), r.get("away",""))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        try:
            hk = f"{(r.get('home','') or '').strip().lower()}|{(r.get('away','') or '').strip().lower()}"
            global _daily_date, _daily_keys
            if _daily_date != today:
                _daily_date = today
                _daily_keys = set()
            if hk:
                _daily_keys.add(hk)
        except Exception:
            pass
    _cache_data = out
    _cache_ts = _now()
    try:
        logging.debug("flashscore: collected odds entries=%d", len(out))
    except Exception:
        pass
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
