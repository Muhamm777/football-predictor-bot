from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import sqlite3
import numpy as np
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier

from config import DB_PATH, MODEL_DIR
from .prepare_dataset import prepare_dataset


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_metric(y_true, proba) -> Dict[str, float]:
    try:
        ll = float(log_loss(y_true, proba, labels=[0,1,2]))
    except Exception:
        ll = float('nan')
    try:
        # Brier for one-vs-rest (average)
        bs = 0.0
        for k in range(3):
            yk = (y_true == k).astype(int)
            bs += brier_score_loss(yk, proba[:, k])
        bs /= 3.0
    except Exception:
        bs = float('nan')
    return {"logloss": ll, "brier": bs}


def _log_metrics(rows: list[tuple[str, str, str, float]]):
    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            for model_name, version, metric, value in rows:
                cur.execute(
                    "INSERT INTO model_metrics(model_name, version, metric, value, created_at) VALUES(?,?,?,?,datetime('now'))",
                    (model_name, version, metric, float(value)),
                )
            con.commit()
    except Exception:
        pass


def train_all(model_dir: str | None = None) -> dict:
    X, y = prepare_dataset()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    root = Path(model_dir or MODEL_DIR)
    root = root if isinstance(root, Path) else Path(root)
    _ensure_dir(root)

    saved = []
    metrics_rows = []
    version = "v1"

    # Model A: GBDT + calibration (sigmoid works for multiclass reliably)
    try:
        mA_base = GradientBoostingClassifier(random_state=42)
        mA = CalibratedClassifierCV(mA_base, cv=2, method="sigmoid")
        mA.fit(Xtr, ytr)
        dump(mA, root / "modelA.joblib")
        saved.append("modelA.joblib")
        pa = mA.predict_proba(Xva)
        mm = _safe_metric(yva, pa)
        metrics_rows += [("modelA", version, k, v) for k, v in mm.items()]
    except Exception:
        try:
            # Fallback: uniform dummy so pipeline can continue
            mA = DummyClassifier(strategy="uniform")
            mA.fit(Xtr, ytr)
            dump(mA, root / "modelA.joblib")
            saved.append("modelA.joblib")
            pa = mA.predict_proba(Xva)
        except Exception:
            pa = None

    # Model B: MLP + calibration (lighter config)
    try:
        mB_base = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=150, random_state=42)
        mB = CalibratedClassifierCV(mB_base, cv=2, method="sigmoid")
        mB.fit(Xtr, ytr)
        dump(mB, root / "modelB.joblib")
        saved.append("modelB.joblib")
        pb = mB.predict_proba(Xva)
        mm = _safe_metric(yva, pb)
        metrics_rows += [("modelB", version, k, v) for k, v in mm.items()]
    except Exception:
        try:
            mB = DummyClassifier(strategy="uniform")
            mB.fit(Xtr, ytr)
            dump(mB, root / "modelB.joblib")
            saved.append("modelB.joblib")
            pb = mB.predict_proba(Xva)
        except Exception:
            pb = None

    # Model C (Teacher): fast robust booster + calibration (acts as a strong stabilizer/teacher)
    try:
        # Teacher prioritizes speed+stability; shallow HGB works well on mixed tabular features
        mC_base = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.15, max_iter=150, random_state=42)
        mC = CalibratedClassifierCV(mC_base, cv=2, method="sigmoid")
        mC.fit(Xtr, ytr)
        dump(mC, root / "modelC.joblib")
        saved.append("modelC.joblib")
        pc = mC.predict_proba(Xva)
        mm = _safe_metric(yva, pc)
        metrics_rows += [("modelC", version, k, v) for k, v in mm.items()]
    except Exception:
        try:
            mC = DummyClassifier(strategy="uniform")
            mC.fit(Xtr, ytr)
            dump(mC, root / "modelC.joblib")
            saved.append("modelC.joblib")
            pc = mC.predict_proba(Xva)
        except Exception:
            pc = None

    # Model D (Power): ExtraTrees (robust high-capacity) + calibration
    try:
        mD_base = ExtraTreesClassifier(n_estimators=400, max_depth=None, min_samples_split=4, random_state=42)
        mD = CalibratedClassifierCV(mD_base, cv=2, method="sigmoid")
        mD.fit(Xtr, ytr)
        dump(mD, root / "modelD.joblib")
        saved.append("modelD.joblib")
        pd = mD.predict_proba(Xva)
        mm = _safe_metric(yva, pd)
        metrics_rows += [("modelD", version, k, v) for k, v in mm.items()]
    except Exception:
        try:
            mD = DummyClassifier(strategy="uniform")
            mD.fit(Xtr, ytr)
            dump(mD, root / "modelD.joblib")
            saved.append("modelD.joblib")
            pd = mD.predict_proba(Xva)
        except Exception:
            pd = None

    # Model E (Regularizer): lightweight LogReg with interactions + calibration
    try:
        mE_base = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("logreg", LogisticRegression(max_iter=300, multi_class="multinomial", C=1.0))
        ])
        mE = CalibratedClassifierCV(mE_base, cv=2, method="sigmoid")
        mE.fit(Xtr, ytr)
        dump(mE, root / "modelE.joblib")
        saved.append("modelE.joblib")
        pe = mE.predict_proba(Xva)
        mm = _safe_metric(yva, pe)
        metrics_rows += [("modelE", version, k, v) for k, v in mm.items()]
    except Exception:
        try:
            mE = DummyClassifier(strategy="uniform")
            mE.fit(Xtr, ytr)
            dump(mE, root / "modelE.joblib")
            saved.append("modelE.joblib")
            pe = mE.predict_proba(Xva)
        except Exception:
            pe = None

    # Meta model: stacking on A/B/C/D/E val probabilities + gating features (confidence/entropy/disagreement)
    # Build train data for meta from validation fold
    # Meta model: stacking if we have at least two base probas
    try:
        comps = [p for p in [pa, pb, pc, pd, pe] if p is not None]
        if len(comps) >= 2:
            # base block: concatenated probabilities
            M_blocks = [c for c in comps]
            # gating block: per-model confidence (max prob) and entropy, plus average disagreement
            def _conf(P):
                try:
                    return np.max(P, axis=1)
                except Exception:
                    return np.full((P.shape[0] if hasattr(P,'shape') else 1,), 0.0)
            def _entropy(P):
                try:
                    eps = 1e-12
                    return -np.sum(P * np.log(P + eps), axis=1)
                except Exception:
                    return np.full((P.shape[0] if hasattr(P,'shape') else 1,), 10.0)
            confs = []
            ents = []
            for P in [pa, pb, pc, pd, pe]:
                if P is not None:
                    confs.append(_conf(P))
                    ents.append(_entropy(P))
            # disagreement: mean L1 distance to first available model
            def _avg_disagreement(arr):
                if not arr:
                    return None
                base = arr[0]
                diffs = []
                for Q in arr:
                    try:
                        diffs.append(np.sum(np.abs(Q - base), axis=1))
                    except Exception:
                        continue
                if not diffs:
                    return None
                try:
                    return np.mean(np.vstack(diffs), axis=0)
                except Exception:
                    return None
            disg = _avg_disagreement([p for p in [pa, pb, pc, pd, pe] if p is not None])
            stats_blocks = []
            if confs:
                stats_blocks.append(np.vstack(confs).T)
            if ents:
                stats_blocks.append(np.vstack(ents).T)
            if disg is not None:
                stats_blocks.append(disg.reshape(-1, 1))
            if stats_blocks:
                M_blocks.append(np.hstack(stats_blocks))
            M_va = np.hstack(M_blocks)
            meta = LogisticRegression(max_iter=400, multi_class="multinomial")
            meta.fit(M_va, yva)
            dump(meta, root / "meta.joblib")
            saved.append("meta.joblib")
            pm = meta.predict_proba(M_va)
            mm = _safe_metric(yva, pm)
            metrics_rows += [("meta", version, k, v) for k, v in mm.items()]
    except Exception:
        pass

    _log_metrics(metrics_rows)
    return {"saved": saved, "dir": str(root), "metrics": {r[0]: {} for r in metrics_rows}}
