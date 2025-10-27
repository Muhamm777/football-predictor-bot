import httpx
from typing import Any, Dict, List
from config import REQUESTS_TIMEOUT, USER_AGENT
from scrapers.registry import register
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # bs4 may be missing; keep scraper resilient
    BeautifulSoup = None  # type: ignore
import asyncio

HEADERS = {"User-Agent": USER_AGENT}

async def fetch_predictions() -> List[Dict[str, Any]]:
    """Extract per-match prediction with confidence (best-effort).
    Returns items like: {"home","away","league","time","confidence"}
    """
    if BeautifulSoup is None:
        return []
    seeds = [
        # Placeholder seed pages; adjust selectors as site evolves
        "https://www.eaglepredict.com/predictions",
    ]
    out: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=REQUESTS_TIMEOUT, headers=HEADERS) as client:
        for url in seeds:
            try:
                r = await client.get(url)
                if r.status_code != 200:
                    await asyncio.sleep(0.2)
                    continue
                soup = BeautifulSoup(r.text, "lxml")
                # Generic table/card parsing (best-effort)
                rows = soup.select(".match-row, table tr, .prediction-card")
                for el in rows:
                    try:
                        text = el.get_text(" ", strip=True).lower()
                        if not text:
                            continue
                        # naive splits for teams
                        home = away = ""
                        league = ""
                        when = ""
                        # Try to find team names in specific subnodes first
                        hnode = el.select_one(".team-home, .home, .team1")
                        anode = el.select_one(".team-away, .away, .team2")
                        if hnode and anode:
                            home = hnode.get_text(strip=True)
                            away = anode.get_text(strip=True)
                        # Confidence (as % or score)
                        conf = None
                        cnode = el.select_one(".confidence, .prob, .percent")
                        if cnode:
                            raw = cnode.get_text(strip=True).replace("%", "")
                            try:
                                conf = float(raw) / 100.0 if float(raw) > 1 else float(raw)
                            except Exception:
                                conf = None
                        if not (home and away and conf is not None):
                            continue
                        out.append({
                            "league": league,
                            "time": when,
                            "home": home,
                            "away": away,
                            "confidence": max(0.0, min(1.0, float(conf))),
                        })
                    except Exception:
                        continue
                await asyncio.sleep(0.2)
            except Exception:
                await asyncio.sleep(0.2)
                continue
    return out


async def odds_or_prob() -> List[Dict[str, Any]]:
    """Map confidence to 1X2-like probabilities as a weak auxiliary signal.
    If confidence refers to a favored team, prefer HOME by default; without a side, skip.
    Returns: [{"home","away","probs": {"home","draw","away"}}]
    """
    preds = await fetch_predictions()
    out: List[Dict[str, Any]] = []
    for p in preds:
        try:
            home = (p.get("home") or "").strip()
            away = (p.get("away") or "").strip()
            conf = float(p.get("confidence") or 0)
            if not (home and away and conf > 0):
                continue
            # Simple mapping: favor HOME with given confidence, reasonable draw mass
            # conf in [0.5..0.7] -> home ~ conf, draw ~ 0.22, away ~ 1 - home - draw
            conf = max(0.5, min(0.7, conf))
            ph = conf
            pd = 0.22
            pa = max(0.05, 1.0 - ph - pd)
            s = ph + pd + pa
            ph, pd, pa = ph/s, pd/s, pa/s
            out.append({
                "home": home,
                "away": away,
                "probs": {"home": ph, "draw": pd, "away": pa},
            })
        except Exception:
            continue
    return out


register(
    name="eaglepredict",
    role="external_model",
    fetch={"predictions": fetch_predictions, "odds_or_prob": odds_or_prob},
    enabled=True,
    notes="Maps site confidence into 1X2-like probabilities (auxiliary).",
)
