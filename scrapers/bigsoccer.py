import httpx
from typing import Any, Dict, List
from config import REQUESTS_TIMEOUT, USER_AGENT
from scrapers.registry import register

HEADERS = {"User-Agent": USER_AGENT}

async def fetch_comments() -> List[Dict[str, Any]]:
    # TODO: implement discovery: forum categories/threads; aggregate sentiment per match
    # Expected item example: {"home": str, "away": str, "sentiment": float}
    return []

register(
    name="bigsoccer",
    role="sentiment",
    fetch={"comments": fetch_comments},
    enabled=False,
    notes="MVP scaffold; will aggregate forum sentiment per match.",
)
