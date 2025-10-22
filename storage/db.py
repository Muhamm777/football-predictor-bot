import os
import sqlite3
from typing import Iterable
from config import DB_PATH
from datetime import datetime, timezone

SCHEMA = [
    "CREATE TABLE IF NOT EXISTS fixtures (id TEXT PRIMARY KEY, league TEXT, date TEXT, home TEXT, away TEXT)",
    "CREATE TABLE IF NOT EXISTS odds (fixture_id TEXT, book TEXT, h REAL, d REAL, a REAL, ts TEXT)",
    "CREATE TABLE IF NOT EXISTS team_stats (team TEXT, league TEXT, metric TEXT, value REAL, ts TEXT)",
    "CREATE TABLE IF NOT EXISTS comments (id TEXT PRIMARY KEY, source TEXT, fixture_id TEXT, text TEXT, sentiment REAL, ts TEXT)",
    "CREATE TABLE IF NOT EXISTS predictions (fixture_id TEXT PRIMARY KEY, home REAL, draw REAL, away REAL, confidence REAL, ts TEXT)",
    "CREATE TABLE IF NOT EXISTS prepared_picks (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, text TEXT, category TEXT, ts TEXT)",
    # ML registry: store model versions and metrics
    "CREATE TABLE IF NOT EXISTS model_registry (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, version TEXT, path TEXT, status TEXT, created_at TEXT)",
    "CREATE TABLE IF NOT EXISTS model_metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT, version TEXT, metric TEXT, value REAL, created_at TEXT)",
]

def ensure_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        for stmt in SCHEMA:
            cur.execute(stmt)
        # Indexes for performance
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fixtures_league ON fixtures(league)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_odds_fixture ON odds(fixture_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_team_stats_team ON team_stats(team)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_prepared_picks_ts ON prepared_picks(ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_model_registry_name ON model_registry(name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_model_metrics_name ON model_metrics(model_name)")
        con.commit()

def save_prepared_picks(picks: Iterable[dict]):
    """Save prepared picks. Each pick: {title, text, category, ts(optional)}"""
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        for p in picks:
            ts = p.get("ts") or datetime.now(timezone.utc).isoformat()
            cur.execute(
                "INSERT INTO prepared_picks(title, text, category, ts) VALUES (?, ?, ?, ?)",
                (p.get("title", ""), p.get("text", ""), p.get("category", ""), ts),
            )
        con.commit()

def get_prepared_picks_for_today(limit: int = 5) -> list[dict]:
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        # Filter by UTC date (YYYY-MM-DD) portion of ts
        today = datetime.now(timezone.utc).date().isoformat()
        rows = cur.execute(
            "SELECT id, title, text, category, ts FROM prepared_picks "
            "WHERE substr(ts,1,10)=? AND (category IS NULL OR category <> 'demo') "
            "ORDER BY id DESC LIMIT ?",
            (today, limit),
        ).fetchall()
        return [dict(r) for r in rows]

# Basic helpers for fixtures and odds
def make_fixture_id(date: str, home: str, away: str) -> str:
    return f"{date}|{home.strip()}|{away.strip()}".lower()

def upsert_fixtures(fixtures: Iterable[dict]):
    """fixtures: [{date: 'YYYY-MM-DD', league, home, away}]"""
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        for f in fixtures:
            fid = f.get("id") or make_fixture_id(f.get("date", ""), f.get("home", ""), f.get("away", ""))
            cur.execute(
                "INSERT OR REPLACE INTO fixtures(id, league, date, home, away) VALUES(?,?,?,?,?)",
                (fid, f.get("league", ""), f.get("date", ""), f.get("home", ""), f.get("away", "")),
            )
        con.commit()

def save_odds(entries: Iterable[dict]):
    """entries: [{fixture_id, book, h, d, a, ts(optional)}]"""
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        for e in entries:
            ts = e.get("ts") or datetime.now(timezone.utc).isoformat()
            cur.execute(
                "INSERT INTO odds(fixture_id, book, h, d, a, ts) VALUES(?,?,?,?,?,?)",
                (e.get("fixture_id", ""), e.get("book", ""), e.get("h"), e.get("d"), e.get("a"), ts),
            )
        con.commit()

def get_today_fixtures() -> list[dict]:
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        today = datetime.now(timezone.utc).date().isoformat()
        rows = cur.execute(
            "SELECT * FROM fixtures WHERE date=? ORDER BY league, time(date)",
            (today,),
        ).fetchall()
        return [dict(r) for r in rows]
