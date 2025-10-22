from __future__ import annotations
import csv
import io
import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import urllib.request

# Very small, robust dataset prep:
# - Try to download a single football-data.co.uk CSV (Premier League recent season)
# - Parse minimal columns to build a simple 1X2 label and odds features
# - If anything fails, return a small synthetic dataset so training code can run

FD_URL = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"  # 2023/24 EPL


def _download_csv(url: str) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = resp.read()
            return data.decode("utf-8", errors="ignore")
    except Exception:
        return None


def _parse_fd(csv_text: str) -> Tuple[np.ndarray, np.ndarray]:
    # Use columns: FTR (H/D/A), B365H/B365D/B365A (odds) if available
    f = io.StringIO(csv_text)
    reader = csv.DictReader(f)
    X_list = []
    y_list = []
    for row in reader:
        ftr = row.get("FTR") or row.get("Res") or ""
        if ftr not in ("H", "D", "A"):
            continue
        try:
            oh = float(row.get("B365H") or row.get("AvgH") or 0) or 0.0
            od = float(row.get("B365D") or row.get("AvgD") or 0) or 0.0
            oa = float(row.get("B365A") or row.get("AvgA") or 0) or 0.0
        except Exception:
            oh = od = oa = 0.0
        # implied probabilities (with normalization)
        ih = (1/oh) if oh>0 else 0.0
        idr = (1/od) if od>0 else 0.0
        ia = (1/oa) if oa>0 else 0.0
        s = ih+idr+ia
        if s>0:
            p_h, p_d, p_a = ih/s, idr/s, ia/s
        else:
            p_h = p_d = p_a = 1/3
        X_list.append([oh, od, oa, p_h, p_d, p_a])
        y_list.append({"H":0, "D":1, "A":2}[ftr])
    if not X_list:
        raise ValueError("empty parsed dataset")
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)
    return X, y


def prepare_dataset() -> Tuple[np.ndarray, np.ndarray]:
    # Try remote
    txt = _download_csv(FD_URL)
    if txt:
        try:
            return _parse_fd(txt)
        except Exception:
            pass
    # Fallback: synthetic small dataset
    rng = np.random.RandomState(42)
    X = rng.rand(200, 6)
    y = rng.randint(0, 3, size=200)
    return X, y
