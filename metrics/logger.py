import os
import csv
from datetime import datetime, timezone
from typing import Dict, Any

BASE_DIR = os.path.join("storage", "metrics")
os.makedirs(BASE_DIR, exist_ok=True)


def log_metrics(name: str, data: Dict[str, Any]) -> None:
    """Append metrics row to a daily CSV file under storage/metrics/.
    name: logical metric stream name (e.g., 'scheduler')
    data: flat dict of values; timestamp will be added automatically.
    """
    ts = datetime.now(timezone.utc).isoformat()
    fname = os.path.join(BASE_DIR, f"{name}-{datetime.now(timezone.utc).date().isoformat()}.csv")
    # Ensure stable header order: timestamp first, then sorted keys
    keys = ["ts"] + sorted(data.keys())
    # If file does not exist, write header
    write_header = not os.path.exists(fname)
    with open(fname, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            w.writeheader()
        row = {k: "" for k in keys}
        row["ts"] = ts
        for k, v in data.items():
            row[k] = v
        w.writerow(row)
