import httpx
from typing import Any, Dict, List
from config import REQUESTS_TIMEOUT, USER_AGENT
from scrapers.registry import register

HEADERS = {"User-Agent": USER_AGENT}

async def fetch_predictions() -> List[Dict[str, Any]]:
    # TODO: implement discovery and extraction of predictions with confidence
    # Expected item example: {"home": str, "away": str, "league": str, "time": str, "confidence": float}
    return []

register(
    name="eaglepredict",
    role="external_model",
    fetch={"predictions": fetch_predictions},
    enabled=False,
    notes="MVP scaffold; will extract per-match confidence.",
)
