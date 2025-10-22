from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from sklearn.dummy import DummyClassifier
from joblib import dump

# Minimal training stub: creates placeholder models so the ensemble can load.
# Replace with real pipeline later (LightGBM/XGBoost, MLP, Logistic, meta stacking).

CLASSES = np.array([0, 1, 2])  # home, draw, away


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def train_all(model_dir: str | None = None) -> dict:
    root = Path(model_dir or Path(__file__).resolve().parents[1] / "models")
    _ensure_dir(root)

    X = np.zeros((30, 10), dtype=float)
    y = np.random.randint(0, 3, size=(30,))

    # Model A: placeholder
    mA = DummyClassifier(strategy="uniform")
    mA.fit(X, y)
    dump(mA, root / "modelA.joblib")

    # Model B: placeholder
    mB = DummyClassifier(strategy="uniform")
    mB.fit(X, y)
    dump(mB, root / "modelB.joblib")

    # Model C (odds-only anchor): placeholder
    mC = DummyClassifier(strategy="uniform")
    mC.fit(X, y)
    dump(mC, root / "modelC.joblib")

    # Meta-model: placeholder
    meta = DummyClassifier(strategy="uniform")
    meta.fit(X, y)
    dump(meta, root / "meta.joblib")

    return {"saved": ["modelA.joblib", "modelB.joblib", "modelC.joblib", "meta.joblib"], "dir": str(root)}
