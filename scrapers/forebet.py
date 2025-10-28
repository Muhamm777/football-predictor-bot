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
    url = os.getenv("FOREBET_URL", "https://www.forebet.com/en/football-tips")
    timeout = float(os.getenv("FOREBET_TIMEOUT_SEC", "15") or "15")
    retries = int(os.getenv("FOREBET_RETRIES", "2") or "2")
    out: List[Dict[str, Any]] = []
    # Lightweight HTML probe with graceful failure
    async with httpx.AsyncClient(timeout=REQUESTS_TIMEOUT, headers=HEADERS) as client:
        for _ in range(max(1, retries)):
            try:
                r = await client.get(url)
                if r.status_code != 200:
                    await asyncio.sleep(0.3)
                    continue
                html = r.text
                # Minimal heuristic: look for patterns "Home - Away" within tips table is complex; skip for MVP
                # Keep provider registered even if no items parsed
                break
            except Exception:
                await asyncio.sleep(0.3)
                continue
    return out

register(
    name="forebet",
    role="fixtures_odds",
    fetch={"fixtures": fetch_fixtures, "odds_or_prob": fetch_odds_or_probabilities},
    enabled=os.getenv("FOREBET_ENABLED", "true").lower() in {"1","true","yes","on"},
    notes="MVP forebet provider (HTML probe)",
)
