from typing import Dict, Any
from predictor.poisson import match_probabilities

# Combines poisson baseline with market odds + sentiment signals (placeholders)

def combined_prediction(features: Dict[str, Any]) -> Dict[str, float]:
    base = match_probabilities(features.get("stats", {}))

    # Market odds to implied probabilities if provided: odds = {h, d, a}
    market_probs = None
    odds = features.get("odds") or {}
    try:
        h = float(odds.get("h")) if odds.get("h") else None
        d = float(odds.get("d")) if odds.get("d") else None
        a = float(odds.get("a")) if odds.get("a") else None
        if h and d and a and h > 1e-6 and d > 1e-6 and a > 1e-6:
            ih, idr, ia = 1.0 / h, 1.0 / d, 1.0 / a
            s = ih + idr + ia
            if s > 0:
                market_probs = {"home": ih / s, "draw": idr / s, "away": ia / s}
    except Exception:
        market_probs = None

    # Optional sentiment signal (0..1), default 0
    sentiment_signal = float(features.get("sentiment_signal", 0.0) or 0.0)

    # Blend weights: prioritize base model, use market as anchor if available
    w_base = 0.6
    w_mkt = 0.4 if market_probs else 0.0

    ph = base["home"] * w_base + (market_probs["home"] if market_probs else 0.0) * w_mkt
    pd = base["draw"] * w_base + (market_probs["draw"] if market_probs else 0.0) * w_mkt
    pa = base["away"] * w_base + (market_probs["away"] if market_probs else 0.0) * w_mkt

    # Minor adjustment by sentiment (push to home if positive, to away if negative)
    ph = max(0.0, ph + 0.05 * sentiment_signal)
    pa = max(0.0, pa - 0.05 * sentiment_signal)

    s = ph + pd + pa
    if s <= 1e-9:
        return {"home": 0.34, "draw": 0.32, "away": 0.34}
    return {"home": ph / s, "draw": pd / s, "away": pa / s}
