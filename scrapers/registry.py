from typing import Callable, Dict, Any

# Lightweight registry to plug scrapers and control simple flags/limits
# This is intentionally minimal; advanced rate limiting and robots handling
# can be added later without changing the interface.

Scraper = Dict[str, Any]

_REGISTRY: Dict[str, Scraper] = {}


def register(name: str, role: str, fetch: Callable[..., Any], enabled: bool = True, notes: str = "") -> None:
    _REGISTRY[name] = {
        "name": name,
        "role": role,  # e.g., 'fixtures', 'odds', 'sentiment', 'external_model'
        "fetch": fetch,
        "enabled": enabled,
        "notes": notes,
    }


def all_enabled() -> Dict[str, Scraper]:
    return {k: v for k, v in _REGISTRY.items() if v.get("enabled")}


def get(name: str) -> Scraper | None:
    return _REGISTRY.get(name)
