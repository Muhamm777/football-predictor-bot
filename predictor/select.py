from typing import List, Dict, Any
from config import MAX_DAILY_PICKS, POPULAR_LEAGUES

# Select up to MAX_DAILY_PICKS predictions with the highest confidence
# Each item: { 'fixture_id': str, 'league': str, 'probs': {'home': float,'draw': float,'away': float}, 'meta': {...} }

def compute_confidence(item: Dict[str, Any]) -> float:
    p = item.get('probs', {})
    best = max(p.get('home', 0.0), p.get('draw', 0.0), p.get('away', 0.0))
    # Penalize ambiguous outcomes lightly
    margin = best - sorted([p.get('home',0.0), p.get('draw',0.0), p.get('away',0.0)])[-2]
    return 0.7 * best + 0.3 * max(0.0, margin)


def filter_leagues(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        lg = (it.get('league') or '').strip()
        if lg in POPULAR_LEAGUES:
            out.append(it)
    return out


def top_picks(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Filter by leagues first (exclude rare/minor by default)
    items = filter_leagues(items)
    # Score by confidence
    scored = [(compute_confidence(it), it) for it in items]
    scored.sort(key=lambda x: x[0], reverse=True)
    # Cap selection
    selected = [it for _, it in scored[:MAX_DAILY_PICKS]]
    return selected
