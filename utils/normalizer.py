from typing import Tuple
import re
import unicodedata
import json
import os

# Minimal, extendable normalizer for teams/leagues
# Maps common aliases to canonical names
_TEAM_MAP = {
    "man city": "Manchester City",
    "mancity": "Manchester City",
    "man utd": "Manchester United",
    "manchester utd": "Manchester United",
    "manchester u": "Manchester United",
    "man united": "Manchester United",
    "fc barcelona": "Barcelona",
    "barca": "Barcelona",
    "barça": "Barcelona",
    "real madrid cf": "Real Madrid",
    "real m": "Real Madrid",
    "r. madrid": "Real Madrid",
    "bayern": "Bayern Munich",
    "bayern munchen": "Bayern Munich",
    "bayern münchen": "Bayern Munich",
    "borussia dortmund": "Borussia Dortmund",
    "b. dortmund": "Borussia Dortmund",
    "psg": "Paris Saint-Germain",
    "paris sg": "Paris Saint-Germain",
    "paris st germain": "Paris Saint-Germain",
    "internazionale": "Inter",
    "inter milan": "Inter",
    "ac milan": "Milan",
    "athletic bilbao": "Athletic Club",
    "athletic club bilbao": "Athletic Club",
    "real sociedad de futbol": "Real Sociedad",
    "newcastle u": "Newcastle United",
    "newcastle utd": "Newcastle United",
    "spurs": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "sporting cp": "Sporting",
    "sporting lisbon": "Sporting",
}
_LEAGUE_MAP = {
    "epl": "England Premier League",
    "premier league": "England Premier League",
    "english premier league": "England Premier League",
    "la liga": "Spain La Liga",
    "laliga": "Spain La Liga",
    "bundesliga": "Germany Bundesliga",
    "serie a": "Italy Serie A",
    "ligue 1": "France Ligue 1",
    "primeira liga": "Portugal Primeira Liga",
}

_ws = re.compile(r"\s+")
_suffixes = [
    " fc", " cf", " fk", " sc", " ac", " u19", " u20", " u21", " u23", " ii", " b", " youth"
]


def _strip_diacritics(s: str) -> str:
    if not s:
        return ""
    nfkd_form = unicodedata.normalize('NFKD', s)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def _canon(s: str) -> str:
    s = s or ""
    s = _strip_diacritics(s)
    s = s.strip()
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = _ws.sub(" ", s)
    low = s.lower()
    for suf in _suffixes:
        if low.endswith(suf):
            low = low[: -len(suf)]
            break
    return low.strip().title()


def normalize_team(name: str) -> str:
    n = _canon(name)
    low = n.lower()
    return _TEAM_EXTRA.get(low) or _TEAM_MAP.get(low, n)


def normalize_league(name: str) -> str:
    n = _canon(name)
    low = n.lower()
    return _LEAGUE_EXTRA.get(low) or _LEAGUE_MAP.get(low, n)


def normalize_pair(home: str, away: str) -> Tuple[str, str]:
    return normalize_team(home), normalize_team(away)


def reload_aliases(path: str | None = None) -> bool:
    global _TEAM_EXTRA, _LEAGUE_EXTRA
    p = path or _ALIASES_PATH
    try:
        if not os.path.exists(p):
            _TEAM_EXTRA, _LEAGUE_EXTRA = {}, {}
            return False
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        _TEAM_EXTRA = {str(k).lower(): str(v) for k, v in (data.get("teams") or {}).items()}
        _LEAGUE_EXTRA = {str(k).lower(): str(v) for k, v in (data.get("leagues") or {}).items()}
        return True
    except Exception:
        _TEAM_EXTRA, _LEAGUE_EXTRA = {}, {}
        return False


# Load on import (best-effort)
try:
    reload_aliases()
except Exception:
    pass
