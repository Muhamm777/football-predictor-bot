from __future__ import annotations
from typing import Dict, Any, List
import math

# Minimal, deterministic feature builder for tabular models
# Input: features dict assembled in scheduler/jobs.py for a match
# Output: ordered float vector and feature names

def build_feature_vector(feat: Dict[str, Any]) -> tuple[List[float], List[str]]:
    x: List[float] = []
    cols: List[str] = []

    # Odds-derived features (if present)
    odds = feat.get("odds") or {}
    h = _to_float(odds.get("h"))
    d = _to_float(odds.get("d"))
    a = _to_float(odds.get("a"))
    if h and d and a and h > 0 and d > 0 and a > 0:
        ih, idr, ia = 1.0 / h, 1.0 / d, 1.0 / a
        s = ih + idr + ia
        if s > 0:
            p_h, p_d, p_a = ih / s, idr / s, ia / s
        else:
            p_h = p_d = p_a = 1.0 / 3.0
        # raw odds
        x += [h, d, a]
        cols += ["odds_h", "odds_d", "odds_a"]
        # implied probs
        x += [p_h, p_d, p_a]
        cols += ["prob_h", "prob_d", "prob_a"]
        # log-odds
        x += [math.log(h), math.log(d), math.log(a)]
        cols += ["log_odds_h", "log_odds_d", "log_odds_a"]
        # bookmaker margin
        margin = (ih + idr + ia) - 1.0
        x += [margin]
        cols += ["bk_margin"]
    else:
        # pad when odds missing
        x += [0.0, 0.0, 0.0, 1/3, 1/3, 1/3, 0.0, 0.0, 0.0, 0.0]
        cols += [
            "odds_h","odds_d","odds_a",
            "prob_h","prob_d","prob_a",
            "log_odds_h","log_odds_d","log_odds_a",
            "bk_margin",
        ]

    # Basic team form features if present
    stats = feat.get("stats") or {}
    home_form = stats.get("home_form") or {}
    away_form = stats.get("away_form") or {}

    def _form_triplet(s: Any) -> tuple[float,float,float]:
        # expected like "3-1-1" -> W,D,L
        if isinstance(s, str) and "-" in s:
            try:
                w,d,l = s.split("-", 2)
                return float(w), float(d), float(l)
            except Exception:
                return 0.0,0.0,0.0
        return 0.0,0.0,0.0

    hf_w, hf_d, hf_l = _form_triplet(home_form.get("home_wdl")) if isinstance(home_form, dict) else (0.0,0.0,0.0)
    af_w, af_d, af_l = _form_triplet(away_form.get("away_wdl")) if isinstance(away_form, dict) else (0.0,0.0,0.0)
    x += [hf_w, hf_d, hf_l, af_w, af_d, af_l]
    cols += ["hf_w","hf_d","hf_l","af_w","af_d","af_l"]

    # Sentiment signal
    s = float(feat.get("sentiment_signal", 0.0) or 0.0)
    x += [s]
    cols += ["sentiment"]

    return x, cols


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None
