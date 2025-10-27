import httpx
from typing import Any, Dict, List
from config import REQUESTS_TIMEOUT, USER_AGENT
from scrapers.registry import register
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # bs4 might be missing in current env; keep scraper disabled by default
    BeautifulSoup = None  # type: ignore
import asyncio
from datetime import datetime, timezone
from pathlib import Path

HEADERS = {"User-Agent": USER_AGENT}
# Ensure cache directory is absolute (uvicorn may change CWD)
BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = (BASE_DIR / "cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

async def fetch_fixtures() -> List[Dict[str, Any]]:
    """Extract fixtures from 'Today's Football Predictions' page.
    Structure (from snapshot):
      div#today-div .match-section
        .match-title: "<League>" and <b.float-right>HH:MM</b>
        .match-content: contains two <b> tags for teams around 'vs'
    Returns items like: {"league","time","home","away"}
    """
    seeds = [
        "https://www.matchoutlook.com/todays-football-predictions/",
    ]
    out: List[Dict[str, Any]] = []
    if BeautifulSoup is None:
        return []
    async with httpx.AsyncClient(timeout=REQUESTS_TIMEOUT, headers=HEADERS) as client:
        for url in seeds:
            try:
                r = await client.get(url)
                if r.status_code != 200:
                    await asyncio.sleep(0.2)
                    continue
                soup = BeautifulSoup(r.text, "lxml")
                # Save latest HTML snapshot for selector refinement
                try:
                    (CACHE_DIR / "matchoutlook-football.html").write_text(r.text, encoding="utf-8", errors="ignore")
                except Exception:
                    pass
                # Parse today's matches
                today = soup.select_one("#today-div") or soup
                for card in today.select(".match-section"):
                    # League and time
                    league = ""
                    when = ""
                    ttl = card.select_one(".match-title")
                    if ttl:
                        # League name is text of match-title without the trailing <b.float-right>
                        try:
                            time_node = ttl.select_one(".float-right")
                            if time_node:
                                t = time_node.get_text(strip=True)
                                if t:
                                    # store HH:MM as-is (no date provided)
                                    when = t
                        except Exception:
                            pass
                        try:
                            # Get text before time node
                            league = (ttl.get_text(" ", strip=True) or "").replace(when, "").strip()
                        except Exception:
                            league = ttl.get_text(" ", strip=True) if ttl else ""
                    # Teams
                    home = ""
                    away = ""
                    mc = card.select_one(".match-content")
                    if mc:
                        bolds = [b.get_text(strip=True) for b in mc.find_all("b")]
                        # filter out placeholders like 'vs' or empty
                        names = [t for t in bolds if t and t.lower() not in ("vs", "v", "-")]
                        if len(names) >= 2:
                            home = names[0]
                            away = names[1]
                    if home and away:
                        out.append({
                            "league": league,
                            "time": when,
                            "home": home,
                            "away": away,
                        })
                await asyncio.sleep(0.3)
            except Exception:
                await asyncio.sleep(0.2)
                continue
    return out

async def fetch_odds_or_probabilities() -> List[Dict[str, Any]]:
    """Extract 1X2-like signal from 'Today's Football Predictions' page.
    Page exposes 'Best Bet' text (e.g., 'Home win', 'Away win', '1X', 'Over 1.5').
    We map supported values into pseudo-probabilities to aid the ensemble when odds are absent.
    Returns items like: {"home","away","probs": {"home":float,"draw":float,"away":float}}
    Unsupported markets (e.g., totals) are skipped.
    """
    seeds = [
        "https://www.matchoutlook.com/todays-football-predictions/",
    ]
    out: List[Dict[str, Any]] = []
    if BeautifulSoup is None:
        return []
    async with httpx.AsyncClient(timeout=REQUESTS_TIMEOUT, headers=HEADERS) as client:
        for url in seeds:
            try:
                r = await client.get(url)
                if r.status_code != 200:
                    await asyncio.sleep(0.2)
                    continue
                soup = BeautifulSoup(r.text, "lxml")
                today = soup.select_one("#today-div") or soup
                for card in today.select(".match-section"):
                    home = away = ""
                    mc = card.select_one(".match-content")
                    if mc:
                        bolds = [b.get_text(strip=True) for b in mc.find_all("b")]
                        names = [t for t in bolds if t and t.lower() not in ("vs", "v", "-")]
                        if len(names) >= 2:
                            home = names[0]
                            away = names[1]
                    if not (home and away):
                        continue
                    # Best Bet label
                    best = None
                    bet_node = mc.select_one(".our-bet b") if mc else None
                    if bet_node:
                        best = bet_node.get_text(strip=True).lower()
                    if not best:
                        continue
                    # Map to pseudo-probabilities
                    probs = None
                    if best in ("home win", "1"):
                        probs = {"home": 0.62, "draw": 0.22, "away": 0.16}
                    elif best in ("away win", "2"):
                        probs = {"home": 0.16, "draw": 0.22, "away": 0.62}
                    elif best in ("1x", "home win or draw"):
                        probs = {"home": 0.45, "draw": 0.35, "away": 0.20}
                    elif best in ("x2", "away win or draw"):
                        probs = {"home": 0.20, "draw": 0.35, "away": 0.45}
                    elif best in ("x", "draw"):
                        probs = {"home": 0.28, "draw": 0.44, "away": 0.28}
                    else:
                        # Skip non-1X2 markets like Over/Under, BTTS, etc.
                        probs = None
                    if probs:
                        out.append({"home": home, "away": away, "probs": probs})
                await asyncio.sleep(0.3)
            except Exception:
                await asyncio.sleep(0.2)
                continue
    return out

# Register in registry (disabled by default until extractor is ready)
register(
    name="matchoutlook",
    role="fixtures_odds",
    fetch={"fixtures": fetch_fixtures, "odds_or_prob": fetch_odds_or_probabilities},
    enabled=True,
    notes="MVP scaffold; implement sitemap/seed crawling and extraction.",
)
