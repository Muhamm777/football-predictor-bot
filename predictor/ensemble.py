from typing import Dict, Any
import os
from predictor.poisson import match_probabilities
from predictor.feature_builder import build_feature_vector

_MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
_MODEL_A_PATH = os.path.abspath(os.path.join(_MODEL_DIR, "modelA.joblib"))
_CALIB_A_PATH = os.path.abspath(os.path.join(_MODEL_DIR, "calibA.joblib"))
_MODEL_B_PATH = os.path.abspath(os.path.join(_MODEL_DIR, "modelB.joblib"))
_CALIB_B_PATH = os.path.abspath(os.path.join(_MODEL_DIR, "calibB.joblib"))
_MODEL_C_PATH = os.path.abspath(os.path.join(_MODEL_DIR, "modelC.joblib"))
_CALIB_C_PATH = os.path.abspath(os.path.join(_MODEL_DIR, "calibC.joblib"))
_MODEL_D_PATH = os.path.abspath(os.path.join(_MODEL_DIR, "modelD.joblib"))
_CALIB_D_PATH = os.path.abspath(os.path.join(_MODEL_DIR, "calibD.joblib"))
_MODEL_E_PATH = os.path.abspath(os.path.join(_MODEL_DIR, "modelE.joblib"))
_CALIB_E_PATH = os.path.abspath(os.path.join(_MODEL_DIR, "calibE.joblib"))
_META_PATH = os.path.abspath(os.path.join(_MODEL_DIR, "meta.joblib"))

_loaded = False
_modelA = None
_calibA = None
_modelB = None
_calibB = None
_modelC = None
_calibC = None
_modelD = None
_calibD = None
_modelE = None
_calibE = None
_meta = None

def _try_load_models() -> None:
    global _loaded, _modelA, _calibA, _modelB, _calibB, _modelC, _calibC, _modelD, _calibD, _modelE, _calibE, _meta
    if _loaded:
        return
    try:
        from joblib import load  # lazy import
        if os.path.exists(_MODEL_A_PATH):
            _modelA = load(_MODEL_A_PATH)
        if os.path.exists(_CALIB_A_PATH):
            _calibA = load(_CALIB_A_PATH)
        if os.path.exists(_MODEL_B_PATH):
            _modelB = load(_MODEL_B_PATH)
        if os.path.exists(_CALIB_B_PATH):
            _calibB = load(_CALIB_B_PATH)
        if os.path.exists(_MODEL_C_PATH):
            _modelC = load(_MODEL_C_PATH)
        if os.path.exists(_CALIB_C_PATH):
            _calibC = load(_CALIB_C_PATH)
        if os.path.exists(_MODEL_D_PATH):
            _modelD = load(_MODEL_D_PATH)
        if os.path.exists(_CALIB_D_PATH):
            _calibD = load(_CALIB_D_PATH)
        if os.path.exists(_MODEL_E_PATH):
            _modelE = load(_MODEL_E_PATH)
        if os.path.exists(_CALIB_E_PATH):
            _calibE = load(_CALIB_E_PATH)
        if os.path.exists(_META_PATH):
            _meta = load(_META_PATH)
    except Exception:
        _modelA = _calibA = _modelB = _calibB = _modelC = _calibC = _modelD = _calibD = _modelE = _calibE = _meta = None
    finally:
        _loaded = True


def _apply_models(feat: Dict[str, Any]) -> Dict[str, float] | None:
    """Return probabilities dict via ML ensemble if models are available; else None."""
    _try_load_models()
    if not (_modelA or _modelB or _modelC or _modelD or _modelE or _meta):
        return None
    x, _ = build_feature_vector(feat)
    import numpy as np
    X = np.array([x], dtype=float)

    def _proba(model, calib):
        if model is None:
            return None
        try:
            p = model.predict_proba(X)
            if calib is not None:
                p = calib.predict_proba(X)
            # Ensure order [home, draw, away]; assume model trained with classes [0,1,2]
            if p.shape[1] == 3:
                return {"home": float(p[0,0]), "draw": float(p[0,1]), "away": float(p[0,2])}
        except Exception:
            return None
        return None

    pa = _proba(_modelA, _calibA)
    pb = _proba(_modelB, _calibB)
    pc = _proba(_modelC, _calibC)
    pd = _proba(_modelD, _calibD)
    pe = _proba(_modelE, _calibE)
    if not pa and not pb and not pc and not pd and not pe and not _meta:
        return None
    # If meta model exists, build meta features and use it
    if _meta is not None:
        # meta features: concat probs from A/B/C (missing -> uniform) + core odds features
        def vec(pdct):
            if not pdct:
                return [1/3, 1/3, 1/3]
            return [pdct.get("home",1/3), pdct.get("draw",1/3), pdct.get("away",1/3)]
        mvec_full = vec(pa) + vec(pb) + vec(pc) + vec(pd) + vec(pe)
        # Align vector length to meta expectations if needed
        try:
            nfeat = getattr(_meta, 'n_features_in_', None)
        except Exception:
            nfeat = None
        mvec = mvec_full if (nfeat is None or nfeat >= len(mvec_full)) else mvec_full[:nfeat]
        MX = np.array([mvec], dtype=float)
        try:
            mp = _meta.predict_proba(MX)
            if mp.shape[1] == 3:
                return {"home": float(mp[0,0]), "draw": float(mp[0,1]), "away": float(mp[0,2])}
        except Exception:
            pass
    # Weighted blend with gating based on confidence/entropy/disagreement
    def conf(p):
        if not p:
            return 0.0
        return max(p.get("home",0.0), p.get("draw",0.0), p.get("away",0.0))
    def entropy(p):
        import math
        if not p:
            return 10.0
        vals = [p.get("home",0.0), p.get("draw",0.0), p.get("away",0.0)]
        s = 0.0
        for v in vals:
            if v > 1e-12:
                s -= v * math.log(v + 1e-12)
        return s
    def avg_disagreement(ps: list[dict|None]):
        import statistics
        arr = []
        base = None
        for q in ps:
            if q:
                base = q
                break
        if not base:
            return 0.0
        for q in ps:
            if not q:
                continue
            arr.append(abs(q.get("home",0)-base.get("home",0)) + abs(q.get("draw",0)-base.get("draw",0)) + abs(q.get("away",0)-base.get("away",0)))
        try:
            return statistics.mean(arr) if arr else 0.0
        except Exception:
            return 0.0

    cA, cB, cC, cD, cE = conf(pa), conf(pb), conf(pc), conf(pd), conf(pe)
    eA, eB, eC, eD, eE = entropy(pa), entropy(pb), entropy(pc), entropy(pd), entropy(pe)
    disag = avg_disagreement([pa, pb, pc, pd, pe])

    # base weights
    wA = 0.32 if pa else 0.0
    wB = 0.18 if pb else 0.0
    wC = 0.20 if pc else 0.0
    wD = 0.22 if pd else 0.0
    wE = 0.08 if pe else 0.0  # small stabilizer weight

    # gating: if low confidence or high entropy for A/B, shift weight to C/D
    low_conf_thresh = 0.45
    high_ent_thresh = 0.98  # ~uniform entropy for 3 classes ~1.098, take slightly below
    if pa and (cA < low_conf_thresh or eA > high_ent_thresh):
        if pc:
            wC += 0.05
        if pd:
            wD += 0.05
        if pe:
            wE += 0.02
        wA = max(0.15, wA - 0.10)
    if pb and (cB < low_conf_thresh or eB > high_ent_thresh):
        if pc:
            wC += 0.05
        if pd:
            wD += 0.05
        if pe:
            wE += 0.02
        wB = max(0.10, wB - 0.10)

    # if models disagree strongly, trust Teacher/Power more
    if disag > 0.6:
        if pc:
            wC += 0.05
        if pd:
            wD += 0.10
        if pe:
            wE += 0.03

    # normalize weights if any present
    tw = wA + wB + wC + wD + wE
    if tw > 1e-9:
        wA, wB, wC, wD, wE = wA/tw, wB/tw, wC/tw, wD/tw, wE/tw
    if pa and not (pb or pc or pd or pe):
        wA = 1.0
    if pb and not (pa or pc or pd or pe):
        wB = 1.0
    if pc and not (pa or pb or pd or pe):
        wC = 1.0
    if pd and not (pa or pb or pc or pe):
        wD = 1.0
    if pe and not (pa or pb or pc or pd):
        wE = 1.0
    h = (pa.get("home",0.0) if pa else 0.0) * wA + (pb.get("home",0.0) if pb else 0.0) * wB + (pc.get("home",0.0) if pc else 0.0) * wC + (pd.get("home",0.0) if pd else 0.0) * wD + (pe.get("home",0.0) if pe else 0.0) * wE
    d = (pa.get("draw",0.0) if pa else 0.0) * wA + (pb.get("draw",0.0) if pb else 0.0) * wB + (pc.get("draw",0.0) if pc else 0.0) * wC + (pd.get("draw",0.0) if pd else 0.0) * wD + (pe.get("draw",0.0) if pe else 0.0) * wE
    a = (pa.get("away",0.0) if pa else 0.0) * wA + (pb.get("away",0.0) if pb else 0.0) * wB + (pc.get("away",0.0) if pc else 0.0) * wC + (pd.get("away",0.0) if pd else 0.0) * wD + (pe.get("away",0.0) if pe else 0.0) * wE
    s = h + d + a
    if s <= 1e-9:
        return None
    return {"home": h/s, "draw": d/s, "away": a/s}


def combined_prediction(features: Dict[str, Any]) -> Dict[str, float]:
    # Try ML ensemble first
    ml = _apply_models(features)
    if ml:
        return ml

    # Fallback: poisson + market odds + sentiment
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

    sentiment_signal = float(features.get("sentiment_signal", 0.0) or 0.0)
    w_base = 0.6
    w_mkt = 0.4 if market_probs else 0.0

    ph = base["home"] * w_base + (market_probs["home"] if market_probs else 0.0) * w_mkt
    pd = base["draw"] * w_base + (market_probs["draw"] if market_probs else 0.0) * w_mkt
    pa = base["away"] * w_base + (market_probs["away"] if market_probs else 0.0) * w_mkt

    ph = max(0.0, ph + 0.05 * sentiment_signal)
    pa = max(0.0, pa - 0.05 * sentiment_signal)

    s = ph + pd + pa
    if s <= 1e-9:
        return {"home": 0.34, "draw": 0.32, "away": 0.34}
    return {"home": ph / s, "draw": pd / s, "away": pa / s}
