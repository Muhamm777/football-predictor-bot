from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseScraper(ABC):
    name: str = "base"

    @abstractmethod
    async def fetch_fixtures(self) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    async def fetch_odds(self) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    async def fetch_comments(self) -> List[Dict[str, Any]]:
        ...
