import asyncio
import time
from typing import Optional, Dict, Any
import httpx
from config import (
    REQUESTS_TIMEOUT,
    USER_AGENT,
    PROXY_URL,
    RETRIES,
    BACKOFF_BASE,
    RATE_LIMIT_RPS,
)

_DEFAULT_HEADERS = {"User-Agent": USER_AGENT}

class RateLimiter:
    def __init__(self, rps: float):
        self.min_interval = 1.0 / max(0.001, rps)
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            delta = now - self._last
            wait_for = self.min_interval - delta
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last = time.monotonic()

_limiter = RateLimiter(RATE_LIMIT_RPS)

def _client_kwargs() -> Dict[str, Any]:
    kw: Dict[str, Any] = {
        "timeout": httpx.Timeout(REQUESTS_TIMEOUT),
        "headers": dict(_DEFAULT_HEADERS),
        "follow_redirects": True,
        "http2": True,
        "verify": True,
    }
    if PROXY_URL:
        kw["proxies"] = PROXY_URL
    return kw

def new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(**_client_kwargs())

async def fetch_url(url: str, *, client: Optional[httpx.AsyncClient] = None, headers: Optional[Dict[str, str]] = None) -> httpx.Response:
    await _limiter.wait()
    owns = False
    if client is None:
        client = new_client()
        owns = True
    try:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, RETRIES + 1):
            try:
                resp = await client.get(url, headers=headers)
                if resp.status_code >= 500:
                    raise httpx.HTTPError(f"Server error {resp.status_code}")
                return resp
            except (httpx.HTTPError, httpx.ConnectError, httpx.ReadTimeout) as e:
                last_exc = e
                backoff = (BACKOFF_BASE ** attempt) + (0.05 * attempt)
                await asyncio.sleep(backoff)
        assert last_exc is not None
        raise last_exc
    finally:
        if owns:
            await client.aclose()
