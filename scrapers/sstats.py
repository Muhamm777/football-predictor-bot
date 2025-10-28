import httpx
from typing import Any, Dict, List
from config import REQUESTS_TIMEOUT, USER_AGENT
from storage.db import get_today_fixtures
import os
import asyncio
from datetime import datetime, timezone
from scrapers.registry import register
import logging

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

async def fetch_fixtures() -> List[Dict[str, Any]]:
    base = os.getenv("SSTATS_API_BASE", "https://api.sstats.net").rstrip("/")
    key = os.getenv("SSTATS_API_KEY") or os.getenv("SSTATS_APIKEY") or os.getenv("SSTATS_KEY") or ""
    url = f"{base}/games/list"
    out: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=REQUESTS_TIMEOUT, headers=HEADERS) as client:
        try:
            retries = int(os.getenv("SSTATS_RETRIES", "2") or "2")
            items: Any = []
            today = datetime.now(timezone.utc).date()
            params_chain = []
            q1 = {"upcoming": "true", "limit": 100}
            q2 = {"Date": today.isoformat(), "limit": 100}
            q3 = {"From": today.isoformat(), "To": (today).isoformat(), "limit": 200}
            if key:
                q1["apikey"] = key
                q2["apikey"] = key
                q3["apikey"] = key
            params_chain = [q1, q2, q3]
            for qi in params_chain:
                ok = False
                for _ in range(max(1, retries)):
                    r = await client.get(url, params=qi)
                    if r.status_code == 200:
                        try:
                            data = r.json()
                            items = data if isinstance(data, list) else data.get("items") or data
                        except Exception:
                            items = []
                        if items:
                            ok = True
                            break
                    await asyncio.sleep(0.3)
                logging.info("sstats fixtures query done: params=%s status=%s items=%s", list(qi.keys()), getattr(r, "status_code", None), 0 if not items else (len(items) if isinstance(items, list) else 1))
                if ok:
                    break
            for g in items or []:
                home = (g.get("homeTeam") or {}).get("name")
                away = (g.get("awayTeam") or {}).get("name")
                if not (home and away):
                    continue
                league = ((g.get("season") or {}).get("league") or {}).get("name") or ""
                kickoff = g.get("date")
                out.append({
                    "league": league,
                    "time": kickoff or "",
                    "home": home,
                    "away": away,
                })
        except Exception:
            return []
    return out

def _extract_odds_1x2(odds_list: Any) -> Dict[str, float] | None:
    try:
        for m in odds_list or []:
            if m.get("marketId") == 1:
                arr = m.get("odds")
                if not isinstance(arr, list):
                    return None
                vals = { (o.get("name") or "").lower(): float(o.get("value")) for o in arr if o is not None and o.get("value") is not None }
                h = vals.get("home")
                d = vals.get("draw")
                a = vals.get("away")
                if h and d and a:
                    return {"home": h, "draw": d, "away": a}
    except Exception:
        return None
    return None

def _odds_to_probs(odds: Dict[str, float]) -> Dict[str, float]:
    inv = {k: 1.0 / v for k, v in odds.items() if v and v > 0}
    s = sum(inv.values()) or 1.0
    return {k: v / s for k, v in inv.items()}

async def fetch_odds_or_probabilities() -> List[Dict[str, Any]]:
    base = os.getenv("SSTATS_API_BASE", "https://api.sstats.net").rstrip("/")
    key = os.getenv("SSTATS_API_KEY") or os.getenv("SSTATS_APIKEY") or os.getenv("SSTATS_KEY") or ""
    url = f"{base}/games/list"
    out: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=REQUESTS_TIMEOUT, headers=HEADERS) as client:
        try:
            retries = int(os.getenv("SSTATS_RETRIES", "2") or "2")
            items: Any = []
            today = datetime.now(timezone.utc).date()
            params_chain = []
            q1 = {"upcoming": "true", "limit": 100}
            q2 = {"Date": today.isoformat(), "limit": 100}
            q3 = {"From": today.isoformat(), "To": (today).isoformat(), "limit": 200}
            if key:
                q1["apikey"] = key
                q2["apikey"] = key
                q3["apikey"] = key
            params_chain = [q1, q2, q3]
            for qi in params_chain:
                ok = False
                for _ in range(max(1, retries)):
                    r = await client.get(url, params=qi)
                    if r.status_code == 200:
                        try:
                            data = r.json()
                            items = data if isinstance(data, list) else data.get("items") or data
                        except Exception:
                            items = []
                        if items:
                            ok = True
                            break
                    await asyncio.sleep(0.3)
                logging.info("sstats odds query done: params=%s status=%s items=%s", list(qi.keys()), getattr(r, "status_code", None), 0 if not items else (len(items) if isinstance(items, list) else 1))
                if ok:
                    break
            for g in items or []:
                home = (g.get("homeTeam") or {}).get("name")
                away = (g.get("awayTeam") or {}).get("name")
                if not (home and away):
                    continue
                odds = _extract_odds_1x2(g.get("odds"))
                if not odds:
                    continue
                probs = _odds_to_probs(odds)
                out.append({
                    "home": home,
                    "away": away,
                    "probs": probs,
                    "odds": odds,
                })
        except Exception:
            return []
    return out

register(
    name="sstats",
    role="fixtures_odds",
    fetch={"fixtures": fetch_fixtures, "odds_or_prob": fetch_odds_or_probabilities},
    enabled=True,
    notes="",
)
