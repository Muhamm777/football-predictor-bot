import os
from typing import Any, Dict, List
import asyncio
import httpx
from datetime import datetime, timedelta, timezone
from scrapers.registry import register
from config import REQUESTS_TIMEOUT, USER_AGENT

HEADERS_BASE = {"User-Agent": USER_AGENT}
API_URL = "https://api.football-data.org/v4"

async def fetch_fixtures(date_from: str | None = None, date_to: str | None = None) -> List[Dict[str, Any]]:
    """Fetch fixtures from Football-Data.org within a date window (UTC).
    Requires FOOTBALL_DATA_API_KEY in environment.
    Returns items: {"league","home","away","kickoff","time_iso"}
    """
    token = os.environ.get("FOOTBALL_DATA_API_KEY", "").strip()
    if not token:
        return []
    now = datetime.now(timezone.utc)
    if not date_from:
        date_from = now.date().isoformat()
    if not date_to:
        date_to = (now + timedelta(days=1)).date().isoformat()
    url = f"{API_URL}/matches?dateFrom={date_from}&dateTo={date_to}"
    headers = {**HEADERS_BASE, "X-Auth-Token": token}
    out: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=REQUESTS_TIMEOUT, headers=headers) as client:
        try:
            r = await client.get(url)
            if r.status_code != 200:
                await asyncio.sleep(0.2)
                return []
            data = r.json() or {}
            for m in (data.get("matches") or []):
                try:
                    comp = (m.get("competition") or {}).get("name") or ""
                    ht = (m.get("homeTeam") or {}).get("name") or ""
                    at = (m.get("awayTeam") or {}).get("name") or ""
                    ko = m.get("utcDate") or ""
                    if ht and at:
                        out.append({
                            "league": comp,
                            "home": ht,
                            "away": at,
                            "kickoff": ko,
                            "time_iso": ko,
                        })
                except Exception:
                    continue
        except Exception:
            return []
    return out

try:
    register(
        name="football_data",
        role="fixtures",
        fetch={"fixtures": fetch_fixtures},
        enabled=True,
        notes="Fixtures via Football-Data.org API (requires FOOTBALL_DATA_API_KEY)",
    )
except Exception:
    pass
