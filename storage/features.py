import os
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from config import DB_PATH

FEATURES_SCHEMA = [
    "CREATE TABLE IF NOT EXISTS features (match_id TEXT PRIMARY KEY, data TEXT, ts TEXT)",
]

def _ensure_features_table():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        for stmt in FEATURES_SCHEMA:
            cur.execute(stmt)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_features_ts ON features(ts)")
        con.commit()

_ensure_features_table()

def save_features(match_id: str, data: Dict[str, Any]) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    blob = json.dumps(data, ensure_ascii=False)
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO features(match_id, data, ts) VALUES(?,?,?)",
            (match_id, blob, ts),
        )
        con.commit()


def get_features(match_id: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        row = cur.execute(
            "SELECT data FROM features WHERE match_id=?",
            (match_id,),
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["data"])
        except Exception:
            return None
