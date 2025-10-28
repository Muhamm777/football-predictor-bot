import httpx
from typing import Any, Dict, List
import os
import asyncio
from config import REQUESTS_TIMEOUT, USER_AGENT
from scrapers.registry import register

HEADERS = {"User-Agent": USER_AGENT}

async def fetch_fixtures() -> List[Dict[str, Any]]:
    return []

async def fetch_odds_or_probabilities() -> List[Dict[str, Any]]:
    url = os.getenv("STATAREA_URL", "https://www.statarea.com/predictions")
    timeout = float(os.getenv("STATAREA_TIMEOUT_SEC", "15") or "15")
    retries = int(os.getenv("STATAREA_RETRIES", "2") or "2")
    out: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=REQUESTS_TIMEOUT, headers=HEADERS) as client:
        for _ in range(max(1, retries)):
            try:
                r = await client.get(url)
                if r.status_code != 200:
                    await asyncio.sleep(0.3)
                    continue
                # HTML parsing omitted in MVP
                break
            except Exception:
                await asyncio.sleep(0.3)
                continue
    return out

register(
    name="statarea",
    role="fixtures_odds",
    fetch={"fixtures": fetch_fixtures, "odds_or_prob": fetch_odds_or_probabilities},
    enabled=os.getenv("STATAREA_ENABLED", "true").lower() in {"1","true","yes","on"},
    notes="MVP statarea provider (HTML probe)",
)
