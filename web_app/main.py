from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Dict
from storage.db import get_prepared_picks_for_today, save_prepared_picks
from storage.db import get_today_fixtures
from pathlib import Path
import asyncio
from config import API_TOKEN
from fastapi import HTTPException, BackgroundTasks
from scheduler.jobs import job_update_all
from scheduler.jobs import start_scheduler
from scheduler.jobs import job_deep_crawl
import sqlite3
from config import DB_PATH
from storage.db import ensure_db
from storage.db import upsert_fixtures
from ml.train_all import train_all
import os
from glob import glob
from joblib import dump
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
import logging
import difflib
import json
from web_app.telegram_client import send_message
import httpx
from scrapers.registry import all_enabled
from scrapers import free_engine as free_engine_provider
from scrapers import free_synth_odds as free_synth_provider
from free.prediction_engine import FootballPredictionEngine
from free.analysis_factors import PredictionFactors
from free.prediction_visualization import create_prediction_factors_diagram, create_algorithm_flow
import scrapers  # ensure registry providers are imported
from datetime import datetime, timezone

app = FastAPI(title="Football Predictor Bot - Web")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Use absolute paths (uvicorn reloader can change CWD)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.on_event("startup")
async def _startup_init_db():
    # Ensure SQLite schema exists before serving requests
    ensure_db()
    try:
        # Start APScheduler (scrape/predict jobs) on service startup
        await start_scheduler()
    except Exception:
        pass

@app.get("/health", response_class=HTMLResponse)
async def health() -> str:
    return "OK"

@app.get("/api/picks")
async def api_picks(limit: int = 10, include_info: bool = True, include_demo: bool = True):
    picks: List[Dict] = get_prepared_picks_for_today(limit=limit)
    if picks and (include_info and include_demo):
        return picks
    # Fallback: include info/demo if function filtered them out, or nothing for today
    try:
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            where = []
            params = []
            # only today
            where.append("substr(ts,1,10)=date('now')")
            if not include_info:
                where.append("category<> 'info'")
            if not include_demo:
                where.append("category<> 'demo'")
            w = (" WHERE " + " AND ".join(where)) if where else ""
            rows = cur.execute(f"SELECT id,title,text,category,ts FROM prepared_picks{w} ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            if rows:
                return [dict(r) for r in rows]
    except Exception:
        pass
    return picks

@app.get("/api/picks_any")
async def api_picks_any(limit: int = 10):
    try:
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            rows = cur.execute("SELECT id,title,text,category,ts FROM prepared_picks ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []

@app.post("/api/rebuild")
async def api_rebuild(request: Request, token: str = ""):
    # Accept token from query, header, or form for robustness
    if not token:
        token = request.headers.get("X-Api-Token", "")
    if not token:
        try:
            form = await request.form()
            token = form.get("token", "")
        except Exception:
            token = token or ""
    if not API_TOKEN or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Fire-and-forget rebuild (scrape -> predict -> save)
    asyncio.create_task(job_update_all())
    return {"status": "started"}

@app.post("/api/deep_crawl")
async def api_deep_crawl(request: Request, token: str = ""):
    # Manual trigger for nightly deep crawl scaffold
    if not token:
        token = request.headers.get("X-Api-Token", "")
    if not token:
        try:
            form = await request.form()
            token = form.get("token", "")
        except Exception:
            token = token or ""
    if not API_TOKEN or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    asyncio.create_task(job_deep_crawl())
    return {"status": "started"}

@app.post("/api/publish_now")
async def api_publish_now(request: Request, token: str = "", chat_id: str = ""):
    # Secure admin endpoint to publish current picks to Telegram
    if not token:
        token = request.headers.get("X-Api-Token", "")
    if not API_TOKEN or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not chat_id:
        chat_id = os.environ.get("PUBLISH_CHAT_ID", "")
    if not chat_id:
        raise HTTPException(status_code=400, detail="chat_id required")
    picks: List[Dict] = get_prepared_picks_for_today(limit=10)
    if not picks:
        # Try fallback to any recent picks
        try:
            with sqlite3.connect(DB_PATH) as con:
                con.row_factory = sqlite3.Row
                cur = con.cursor()
                rows = cur.execute("SELECT title,text FROM prepared_picks ORDER BY id DESC LIMIT 10").fetchall()
                picks = [dict(r) for r in rows]
        except Exception:
            picks = []
    if not picks:
        return {"ok": False, "detail": "no picks available"}
    # Split and send
    sent = 0
    chunks: list[str] = []
    for i in range(0, len(picks), 5):
        part = picks[i:i+5]
        lines = []
        for p in part:
            title = p.get("title", "")
            text = p.get("text", "")
            lines.append(f"• {title}\n{text}")
        chunks.append("Подготовленные прогнозы:\n" + "\n\n".join(lines))
    for msg in chunks:
        resp = await send_message(chat_id, msg)
        if (resp or {}).get("ok"):
            sent += 1
    return {"ok": True, "chunks": len(chunks), "sent": sent}

@app.get("/api/debug")
async def api_debug(limit: int = 10):
    today = get_prepared_picks_for_today(limit=limit)
    total = 0
    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            row = cur.execute("SELECT COUNT(*) FROM prepared_picks").fetchone()
            total = int(row[0]) if row else 0
    except Exception as e:
        return {"error": str(e)}
    return {"today_count": len(today), "total_count": total, "today": today}

@app.post("/api/clear_demo_today")
async def api_clear_demo_today(request: Request, token: str = ""):
    # Accept token from header, query, or form
    if not token:
        token = request.headers.get("X-Api-Token", "")
    if not token:
        try:
            form = await request.form()
            token = form.get("token", "")
        except Exception:
            token = token or ""
    if not API_TOKEN or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            cur.execute("DELETE FROM prepared_picks WHERE substr(ts,1,10)=? AND category='demo'", (today,))
            con.commit()
        return {"deleted": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/build_picks")
async def api_build_picks(token: str = "", limit: int = 10):
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized (use token query)")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Collect probabilities and odds
    try:
        probs_arr = await free_engine_provider.odds_or_prob()
    except Exception:
        probs_arr = []
    # Fallback: if free_engine is empty, merge auxiliary providers' probabilities
    if not probs_arr:
        try:
            reg = all_enabled() or {}
            seen_keys = set()
            def k(h: str, a: str) -> str:
                return f"{(h or '').strip().lower()}|{(a or '').strip().lower()}"
            aux: list[dict] = []
            for name, s in reg.items():
                fn = s.get("fetch", {}).get("odds_or_prob")
                if not fn:
                    continue
                # Skip known pure-odds providers
                if name in ("flashscore", "free_synth_odds"):
                    continue
                try:
                    arr = await fn()
                except Exception:
                    arr = []
                for it in (arr or []):
                    if not isinstance(it, dict):
                        continue
                    pr = it.get("probs") or {}
                    if not (pr.get("home") or pr.get("draw") or pr.get("away")):
                        continue
                    h = it.get("home") or ""
                    a = it.get("away") or ""
                    key = k(h, a)
                    if not h or not a or key in seen_keys:
                        continue
                    aux.append({"home": h, "away": a, "probs": pr, "league": it.get("league")})
                    seen_keys.add(key)
            if aux:
                probs_arr = aux
        except Exception:
            pass
    # Aggregate odds from registry providers, prioritize real odds over synthetic
    odds_arr: list[dict] = []
    prov_counts: dict[str, int] = {}
    priority = {"flashscore": 2, "free_synth_odds": 1}
    try:
        reg = all_enabled() or {}
        for name, s in reg.items():
            fn = s.get("fetch", {}).get("odds_or_prob")
            if not fn:
                continue
            try:
                arr = await fn()
                prov_counts[name] = len(arr or [])
                for o in (arr or []):
                    o = dict(o)
                    o["__provider"] = name
                    odds_arr.append(o)
            except Exception:
                continue
    except Exception:
        pass
    # Index by (home,away)
    def k(h: str, a: str) -> str:
        return f"{h.strip().lower()}|{a.strip().lower()}"
    probs_map = {k(x.get("home",""), x.get("away","")): x for x in (probs_arr or [])}
    # Choose best odds per pair by provider priority
    odds_map: dict[str, dict] = {}
    for x in (odds_arr or []):
        key = k(x.get("home",""), x.get("away",""))
        prov = x.get("__provider") or ""
        score = priority.get(str(prov), 0)
        cur = odds_map.get(key)
        # Accept only entries that contain odds (h/d/a) or odds dict
        has_triplet = (x.get("h") and x.get("d") and x.get("a")) is not None
        if not has_triplet:
            od = x.get("odds") or {}
            has_triplet = (od.get("home") and od.get("draw") and od.get("away")) is not None
        if not has_triplet:
            continue
        if cur is None or score > cur.get("__score", -1):
            rec = {}
            if x.get("h") is not None:
                rec = {"home": x.get("h"), "draw": x.get("d"), "away": x.get("a")}
            else:
                od = x.get("odds") or {}
                rec = {"home": od.get("home"), "draw": od.get("draw"), "away": od.get("away")}
            odds_map[key] = {"odds": rec, "__provider": prov, "__score": score}
    # Log unmatched pairs into sqlite for diagnostics (today only)
    try:
        missing = [kk for kk in probs_map.keys() if kk not in odds_map]
        if missing:
            with sqlite3.connect(DB_PATH) as con:
                cur = con.cursor()
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS unmatched_pairs (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, home TEXT, away TEXT, created_at TEXT)"
                )
                from datetime import datetime, timezone
                today = datetime.now(timezone.utc).date().isoformat()
                for kk in missing[:100]:
                    h,a = kk.split("|",1)
                    try:
                        cur.execute(
                            "INSERT INTO unmatched_pairs(date, home, away, created_at) VALUES(?,?,?,datetime('now'))",
                            (today, h, a)
                        )
                    except Exception:
                        continue
                con.commit()
    except Exception:
        pass
    candidates: list[dict] = []
    odds_keys = list(odds_map.keys())
    for key, pv in probs_map.items():
        try:
            ov = odds_map.get(key)
            if not ov:
                # Try fuzzy match on normalized key "home|away"
                try:
                    match = difflib.get_close_matches(key, odds_keys, n=1, cutoff=0.92)
                except Exception:
                    match = []
                if match:
                    ov = odds_map.get(match[0])
                else:
                    continue
            home = (pv.get("home") or "").strip()
            away = (pv.get("away") or "").strip()
            pr = pv.get("probs") or {}
            od = (ov.get("odds") or {})
            ph, pd, pa = float(pr.get("home",0) or 0), float(pr.get("draw",0) or 0), float(pr.get("away",0) or 0)
            oh, odr, oa = float(od.get("home",0) or 0), float(od.get("draw",0) or 0), float(od.get("away",0) or 0)
            # EV per outcome: odds*prob - 1
            ev_h = oh*ph - 1.0 if oh>0 else -1
            ev_d = odr*pd - 1.0 if odr>0 else -1
            ev_a = oa*pa - 1.0 if oa>0 else -1
            best_ev = max(ev_h, ev_d, ev_a)
            if best_ev <= -0.05:
                continue
            pick_side = "home" if best_ev==ev_h else ("draw" if best_ev==ev_d else "away")
            pick_odds = oh if pick_side=="home" else (odr if pick_side=="draw" else oa)
            pick_prob = ph if pick_side=="home" else (pd if pick_side=="draw" else pa)
            candidates.append({
                "home": home,
                "away": away,
                "edge": best_ev,
                "side": pick_side,
                "odds": pick_odds,
                "prob": pick_prob,
                "league": (pv.get("league") or ov.get("league") or "General"),
                "provider": (ov.get("__provider") or "unknown"),
            })
        except Exception:
            continue
    # Rank and pick top-N
    candidates.sort(key=lambda x: (x.get("edge") or 0.0), reverse=True)
    top = candidates[:max(1, min(int(limit or 10), 20))]
    # Fallback: if still empty, build by max probability using synthetic odds
    if not top:
        fallback: list[dict] = []
        # reuse already fetched arrays
        for pv in (probs_arr or [])[:max(1, min(int(limit or 10), 20))]:
            try:
                home = (pv.get("home") or "").strip()
                away = (pv.get("away") or "").strip()
                if not home or not away:
                    continue
                pr = pv.get("probs") or {}
                ph, pd, pa = float(pr.get("home",0) or 0), float(pr.get("draw",0) or 0), float(pr.get("away",0) or 0)
                s = ph+pd+pa
                if s <= 0:
                    continue
                ph, pd, pa = ph/s, pd/s, pa/s
                side = "home"
                psel = ph
                if pd >= psel:
                    side = "draw"; psel = pd
                if pa >= psel:
                    side = "away"; psel = pa
                # synthetic odds with small margin (5%) for display
                margin = 0.05
                fair = max(1e-6, min(1.0-1e-6, psel))
                odds = (1.0/fair) * (1.0 + margin)
                fallback.append({
                    "home": home,
                    "away": away,
                    "edge": 0.0,
                    "side": side,
                    "odds": odds,
                    "prob": psel,
                    "league": (pv.get("league") or "General"),
                })
            except Exception:
                continue
        top = fallback
    # Save to prepared_picks for daily_top5
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    picks = []
    for c in top:
        title = f"{c['league']} — {c['home']} vs {c['away']}"
        prov = c.get('provider') or 'unknown'
        text = (
            f"Ставка: {c['side'].upper()}\n"
            f"Коэф.: {c['odds']:.2f}  Вероятность: {c['prob']:.2f}  EV: {c['edge']:.2f}\n"
            f"Источник: free_engine + odds[{prov}]\n"
        )
        picks.append({"title": title, "text": text, "category": "auto", "ts": now})
    if picks:
        save_prepared_picks(picks)
    return {"inserted": len(picks), "total_candidates": len(candidates)}

@app.get("/api/odds_sources")
async def api_odds_sources(token: str = ""):
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized (use token query)")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    out = {"providers": [], "total": 0, "samples": []}
    try:
        reg = all_enabled() or {}
        total = 0
        samples = []
        for name, s in reg.items():
            fn = s.get("fetch", {}).get("odds_or_prob")
            if not fn:
                continue
            try:
                arr = await fn()
                cnt = len(arr or [])
                total += cnt
                out["providers"].append({"name": name, "count": cnt})
                samples.extend([{"provider": name, **(arr[i] if isinstance(arr[i], dict) else {})} for i in range(min(2, cnt))])
            except Exception:
                out["providers"].append({"name": name, "count": 0, "error": True})
        out["total"] = total
        out["samples"] = samples[:10]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return out

# ---------------- Top-7 predictions (stable cards) ----------------
def _ensure_top7_table():
    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS top7_picks (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT,
                  league TEXT,
                  home TEXT,
                  away TEXT,
                  side TEXT,
                  prob REAL,
                  odds REAL,
                  ev REAL,
                  provider TEXT,
                  meta TEXT,
                  ts TEXT
                )
                """
            )
            # best-effort add meta if table existed before
            try:
                cur.execute("ALTER TABLE top7_picks ADD COLUMN meta TEXT")
            except Exception:
                pass
            con.commit()
    except Exception:
        pass


@app.post("/api/top7_build")
async def api_top7_build(token: str = "", limit: int = 7):
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized (use token query)")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    _ensure_top7_table()
    # Collect probabilities and odds similar to build_picks
    try:
        probs_arr = await free_engine_provider.odds_or_prob()
    except Exception:
        probs_arr = []
    # Aggregate odds from registry
    odds_arr: list[dict] = []
    priority = {"flashscore": 2, "free_synth_odds": 1}
    fixtures_times: dict[str, str] = {}
    try:
        reg = all_enabled() or {}
        for name, s in reg.items():
            fn = s.get("fetch", {}).get("odds_or_prob")
            if not fn:
                continue
            try:
                arr = await fn()
                for o in (arr or []):
                    o = dict(o)
                    o["__provider"] = name
                    odds_arr.append(o)
            except Exception:
                continue
        # Try to collect kickoff times from fixtures providers if exposed
        for name, s in reg.items():
            fxf = s.get("fetch", {}).get("fixtures")
            if not fxf:
                continue
            try:
                arrfx = await fxf()
            except Exception:
                arrfx = []
            for f in (arrfx or []):
                h = (str(f.get("home") or "")).strip()
                a = (str(f.get("away") or "")).strip()
                if not h or not a:
                    continue
                keyfa = f"{h.lower()}|{a.lower()}"
                ko = f.get("kickoff") or f.get("time_iso") or f.get("time") or ""
                if ko and keyfa not in fixtures_times:
                    fixtures_times[keyfa] = str(ko)
    except Exception:
        pass
    def k(h: str, a: str) -> str:
        return f"{h.strip().lower()}|{a.strip().lower()}"
    probs_map = {k(x.get("home",""), x.get("away","")): x for x in (probs_arr or [])}
    odds_map: dict[str, dict] = {}
    for x in (odds_arr or []):
        key = k(x.get("home",""), x.get("away",""))
        prov = x.get("__provider") or ""
        score = priority.get(str(prov), 0)
        cur = odds_map.get(key)
        has_triplet = (x.get("h") and x.get("d") and x.get("a")) is not None
        if not has_triplet:
            od = x.get("odds") or {}
            has_triplet = (od.get("home") and od.get("draw") and od.get("away")) is not None
        if not has_triplet:
            continue
        if cur is None or score > cur.get("__score", -1):
            rec = {}
            if x.get("h") is not None:
                rec = {"home": x.get("h"), "draw": x.get("d"), "away": x.get("a")}
            else:
                od = x.get("odds") or {}
                rec = {"home": od.get("home"), "draw": od.get("draw"), "away": od.get("away")}
            odds_map[key] = {"odds": rec, "__provider": prov, "__score": score}
    # Build candidate list prioritizing probability, then EV
    candidates: list[dict] = []
    odds_keys = list(odds_map.keys())
    for key, pv in probs_map.items():
        try:
            ov = odds_map.get(key)
            if not ov:
                # fuzzy try
                try:
                    match = difflib.get_close_matches(key, odds_keys, n=1, cutoff=0.92)
                except Exception:
                    match = []
                if match:
                    ov = odds_map.get(match[0])
                else:
                    continue
            pr = pv.get("probs") or {}
            ph, pd, pa = float(pr.get("home",0) or 0), float(pr.get("draw",0) or 0), float(pr.get("away",0) or 0)
            od = (ov.get("odds") or {})
            oh, odr, oa = float(od.get("home",0) or 0), float(od.get("draw",0) or 0), float(od.get("away",0) or 0)
            ev_h = oh*ph - 1.0 if oh>0 else -1
            ev_d = odr*pd - 1.0 if odr>0 else -1
            ev_a = oa*pa - 1.0 if oa>0 else -1
            # choose by highest probability first
            best_prob = ph
            side = "home"; odds = oh; ev = ev_h
            if pd > best_prob:
                best_prob = pd; side = "draw"; odds = odr; ev = ev_d
            if pa > best_prob:
                best_prob = pa; side = "away"; odds = oa; ev = ev_a
            if best_prob <= 0:
                continue
            # kickoff (if available from fixtures)
            kk_pair = f"{(pv.get('home') or '').strip().lower()}|{(pv.get('away') or '').strip().lower()}"
            ko_val = fixtures_times.get(kk_pair)
            meta = {
                "probs": {"home": ph, "draw": pd, "away": pa},
                "odds": {"home": oh, "draw": odr, "away": oa},
                "evs": {"home": ev_h, "draw": ev_d, "away": ev_a},
                "picked": side,
                "provider": ov.get("__provider") or "unknown",
                "kickoff": ko_val,
            }
            candidates.append({
                "league": (pv.get("league") or "General"),
                "home": (pv.get("home") or "").strip(),
                "away": (pv.get("away") or "").strip(),
                "side": side,
                "prob": best_prob,
                "odds": odds,
                "ev": ev,
                "provider": ov.get("__provider") or "unknown",
                "meta": meta,
            })
        except Exception:
            continue
    # Optional: time window filtering for upcoming matches only
    try:
        hours_ahead = float(os.environ.get("TOP7_HOURS_AHEAD", "6") or 6)
    except Exception:
        hours_ahead = 6.0
    if hours_ahead > 0 and fixtures_times:
        from datetime import datetime as _dt
        from datetime import timezone as _tz
        now_utc = _dt.now(_tz.utc)
        filt2: list[dict] = []
        for c in candidates:
            kk = f"{(c.get('home') or '').lower()}|{(c.get('away') or '').lower()}"
            ko = fixtures_times.get(kk)
            if not ko:
                # keep when no info
                filt2.append(c)
                continue
            try:
                # try ISO first
                ko_dt = _dt.fromisoformat(ko)
            except Exception:
                # try HH:MM assume today
                try:
                    hh, mm = str(ko).split(":", 1)
                    co = now_utc.replace(hour=int(hh), minute=int(mm.split()[0]), second=0, microsecond=0)
                    ko_dt = co
                except Exception:
                    ko_dt = None
            if not ko_dt:
                filt2.append(c)
                continue
            delta_h = (ko_dt - now_utc).total_seconds() / 3600.0
            if delta_h >= 0 and delta_h <= hours_ahead:
                filt2.append(c)
        candidates = filt2
    # Rank: probability desc, then EV desc
    candidates.sort(key=lambda x: (x.get("prob") or 0.0, x.get("ev") or -1.0), reverse=True)
    # Apply thresholds and diversity limits
    try:
        min_prob = float(os.environ.get("TOP7_MIN_PROB", "0.55") or 0.55)
    except Exception:
        min_prob = 0.55
    try:
        min_ev = float(os.environ.get("TOP7_MIN_EV", "-0.02") or -0.02)
    except Exception:
        min_ev = -0.02
    try:
        max_per_league = int(os.environ.get("TOP7_MAX_PER_LEAGUE", "2") or 2)
    except Exception:
        max_per_league = 2
    filtered = [c for c in candidates if (c.get("prob") or 0.0) >= min_prob and (c.get("ev") or -1.0) >= min_ev]
    cap = max(1, min(int(limit or 7), 7))
    # Stability margins
    try:
        stab_prob = float(os.environ.get("TOP7_STABILITY_PROB", "0.02") or 0.02)
    except Exception:
        stab_prob = 0.02
    try:
        stab_ev = float(os.environ.get("TOP7_STABILITY_EV", "0.02") or 0.02)
    except Exception:
        stab_ev = 0.02
    # Load today's existing selection to keep stability
    existing: list[dict] = []
    try:
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            today = datetime.now(timezone.utc).date().isoformat()
            rows = cur.execute(
                "SELECT league, home, away, side, prob, odds, ev, provider, meta FROM top7_picks WHERE date=?",
                (today,)
            ).fetchall()
            for r in rows:
                d = dict(r)
                try:
                    if isinstance(d.get("meta"), str) and d["meta"]:
                        d["meta"] = json.loads(d["meta"])
                except Exception:
                    pass
                existing.append(d)
    except Exception:
        existing = []
    # Helper maps
    def key_of(c: dict) -> str:
        return f"{(c.get('home') or '').lower()}|{(c.get('away') or '').lower()}|{c.get('side') or ''}"
    cand_by_key = {key_of(c): c for c in filtered}
    # Start with stable set: keep existing that are still valid and under league caps
    top: list[dict] = []
    league_cnt: dict[str, int] = {}
    for e in sorted(existing, key=lambda x: (x.get("prob") or 0.0, x.get("ev") or -1.0), reverse=True):
        k = key_of(e)
        c = cand_by_key.get(k)
        if not c:
            continue
        lg = (c.get("league") or e.get("league") or "General")
        if league_cnt.get(lg, 0) >= max_per_league:
            continue
        top.append(c)
        league_cnt[lg] = league_cnt.get(lg, 0) + 1
        if len(top) >= cap:
            break
    # Try to improve selection with candidates that beat worst by stability margins
    def better(a: dict, b: dict) -> bool:
        return (a.get("prob", 0) >= (b.get("prob", 0) + stab_prob)) or (a.get("ev", -1) >= (b.get("ev", -1) + stab_ev))
    def worst_index(lst: list[dict]) -> int:
        if not lst:
            return -1
        w = min(range(len(lst)), key=lambda i: ((lst[i].get("prob") or 0.0), (lst[i].get("ev") or -1.0)))
        return w
    for c in filtered:
        if c in top:
            continue
        lg = c.get("league") or "General"
        if len(top) < cap:
            if league_cnt.get(lg, 0) < max_per_league:
                top.append(c)
                league_cnt[lg] = league_cnt.get(lg, 0) + 1
            continue
        wi = worst_index(top)
        if wi >= 0 and better(c, top[wi]):
            # check league cap if we replace
            if league_cnt.get(lg, 0) < max_per_league or (top[wi].get("league") == lg):
                out_lg = top[wi].get("league") or "General"
                league_cnt[out_lg] = max(0, (league_cnt.get(out_lg, 1) - 1))
                top[wi] = c
                league_cnt[lg] = league_cnt.get(lg, 0) + 1
    # If still empty, fallback to best by prob regardless
    if not top:
        top = candidates[:cap]
    # Persist replacing today's top7
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            cur.execute("DELETE FROM top7_picks WHERE date=?", (today,))
            now = datetime.now(timezone.utc).isoformat()
            for c in top:
                cur.execute(
                    "INSERT INTO top7_picks(date, league, home, away, side, prob, odds, ev, provider, meta, ts) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (today, c["league"], c["home"], c["away"], c["side"], float(c["prob"] or 0), float(c["odds"] or 0), float(c["ev"] or -1), c["provider"], json.dumps(c.get("meta") or {}), now)
                )
            con.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"inserted": len(top), "total_candidates": len(candidates)}


@app.post("/api/top7_refresh")
async def api_top7_refresh(token: str = ""):
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized (use token query)")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Simple trigger to rebuild Top-7 now
    try:
        await api_top7_build(token=token, limit=7)
        return {"status": "refreshed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/top7")
async def api_top7(token: str = "", hours_ahead: float = 0.0):
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized (use token query)")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    _ensure_top7_table()
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            rows = cur.execute(
                "SELECT league, home, away, side, prob, odds, ev, provider, meta, ts FROM top7_picks WHERE date=? ORDER BY prob DESC, ev DESC LIMIT 7",
                (today,)
            ).fetchall()
            items = []
            for r in rows:
                d = dict(r)
                try:
                    if isinstance(d.get("meta"), str) and d["meta"]:
                        d["meta"] = json.loads(d["meta"])
                except Exception:
                    pass
                items.append(d)
            # Optional filter by hours_ahead on read
            try:
                ha = float(hours_ahead or 0)
            except Exception:
                ha = 0.0
            if ha > 0:
                from datetime import datetime as _dt
                from datetime import timezone as _tz
                now_utc = _dt.now(_tz.utc)
                filt: list[dict] = []
                for d in items:
                    ko = None
                    try:
                        ko = (d.get("meta") or {}).get("kickoff")
                    except Exception:
                        ko = None
                    if not ko:
                        filt.append(d)
                        continue
                    try:
                        ko_dt = _dt.fromisoformat(ko)
                    except Exception:
                        ko_dt = None
                    if not ko_dt:
                        filt.append(d)
                        continue
                    delta_h = (ko_dt - now_utc).total_seconds()/3600.0
                    if delta_h >= 0 and delta_h <= ha:
                        filt.append(d)
                items = filt
            return {"date": today, "count": len(items), "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/unmatched_today")
async def api_unmatched_today(token: str = "", limit: int = 100):
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized (use token query)")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            # Ensure table exists
            cur.execute(
                "CREATE TABLE IF NOT EXISTS unmatched_pairs (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, home TEXT, away TEXT, created_at TEXT)"
            )
            from datetime import datetime, timezone
            today = datetime.now(timezone.utc).date().isoformat()
            rows = cur.execute(
                "SELECT id, date, home, away, created_at FROM unmatched_pairs WHERE date=? ORDER BY id DESC LIMIT ?",
                (today, int(max(1, min(limit or 100, 1000))))
            ).fetchall()
            return {"date": today, "count": len(rows), "items": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/match_diagnostics")
async def api_match_diagnostics(token: str = ""):
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized (use token query)")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    def k(h: str, a: str) -> str:
        return f"{(h or '').strip().lower()}|{(a or '').strip().lower()}"
    out = {"probs": 0, "odds": {}, "overlap": {}, "missing_odds_for_probs": 0, "samples_missing": []}
    try:
        # Collect probability keys from free_engine
        try:
            probs_arr = await free_engine_provider.odds_or_prob()
        except Exception:
            probs_arr = []
        pkeys = {k(x.get("home",""), x.get("away","")) for x in (probs_arr or [])}
        out["probs"] = len(pkeys)
        # Collect odds keys per provider
        reg = all_enabled() or {}
        odds_sets = {}
        for name, s in reg.items():
            fn = s.get("fetch", {}).get("odds_or_prob")
            if not fn:
                continue
            try:
                arr = await fn()
            except Exception:
                arr = []
            keys = set()
            for o in (arr or []):
                if isinstance(o, dict):
                    keys.add(k(o.get("home",""), o.get("away","")))
            odds_sets[name] = keys
        # Build overlaps and missing
        out["odds"] = {name: len(keys) for name, keys in odds_sets.items()}
        out["overlap"] = {name: len(pkeys & keys) for name, keys in odds_sets.items()}
        missing = []
        for kk in sorted(pkeys):
            if all(kk not in keys for keys in odds_sets.values()):
                missing.append(kk)
        out["missing_odds_for_probs"] = len(missing)
        out["samples_missing"] = missing[:10]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return out

@app.get("/api/metrics")
async def api_metrics(token: str = ""):
    # Simple metrics dump (secured via query header like other endpoints)
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized (use token query)")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    out = {"registry": [], "metrics": [], "free_engine": {}, "summary_today": {}}
    try:
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            out["registry"] = [dict(r) for r in cur.execute("SELECT name, version, status, created_at FROM model_registry ORDER BY id DESC LIMIT 50").fetchall()]
            out["metrics"] = [dict(r) for r in cur.execute("SELECT model_name, version, metric, value, created_at FROM model_metrics ORDER BY id DESC LIMIT 100").fetchall()]
            # Latest metrics for free_engine
            latest = {}
            for m in ("accuracy","logloss","brier"):
                row = cur.execute(
                    "SELECT value, created_at, version FROM model_metrics WHERE model_name=? AND metric=? ORDER BY id DESC LIMIT 1",
                    ("free_engine", m),
                ).fetchone()
                if row:
                    latest[m] = {"value": row[0], "created_at": row[1], "version": row[2]}
            # Active model info
            act = cur.execute(
                "SELECT name, version, status, created_at FROM model_registry WHERE name=? AND status='active' ORDER BY id DESC LIMIT 1",
                ("free_engine",)
            ).fetchone()
            out["free_engine"] = {
                "active": dict(act) if act else None,
                "latest_metrics": latest,
            }
            # Summary for today picks
            try:
                today = datetime.now(timezone.utc).date().isoformat()
                rows = cur.execute("SELECT category, COUNT(*) as cnt FROM prepared_picks WHERE substr(ts,1,10)=? GROUP BY category", (today,)).fetchall()
                total = sum(int(r[1]) for r in rows)
                by_cat = {r[0]: int(r[1]) for r in rows}
                # Parse avg EV and avg probability from text fields
                avg_ev = None
                avg_prob = None
                try:
                    import re as _re
                    vals_ev = []
                    vals_p = []
                    for r in cur.execute("SELECT text FROM prepared_picks WHERE substr(ts,1,10)=?", (today,)).fetchall():
                        txt = (r[0] or "")
                        m_ev = _re.search(r"EV:\s*([-+]?\d+(?:[\.,]\d+)?)", txt)
                        m_p = _re.search(r"Вероятность:\s*([-+]?\d+(?:[\.,]\d+)?)", txt)
                        if m_ev:
                            try:
                                vals_ev.append(float(m_ev.group(1).replace(',', '.')))
                            except Exception:
                                pass
                        if m_p:
                            try:
                                vals_p.append(float(m_p.group(1).replace(',', '.')))
                            except Exception:
                                pass
                    if vals_ev:
                        avg_ev = sum(vals_ev)/len(vals_ev)
                    if vals_p:
                        avg_prob = sum(vals_p)/len(vals_p)
                except Exception:
                    avg_ev = None
                    avg_prob = None
                out["summary_today"] = {"date": today, "total": total, "by_category": by_cat, "avg_ev": avg_ev, "avg_prob": avg_prob}
            except Exception:
                out["summary_today"] = {}
    except Exception as e:
        return {"error": str(e)}
    return out

@app.get("/api/free_status")
async def api_free_status(token: str = ""):
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized (use token query)")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        arr = await free_engine_provider.odds_or_prob()
        out = {
            "cache_ttl": getattr(free_engine_provider, "_CACHE_TTL_SEC", None),
            "max_items": getattr(free_engine_provider, "_MAX_ITEMS", None),
            "live_fallback": getattr(free_engine_provider, "_LIVE_FALLBACK", None),
            "temp": getattr(free_engine_provider, "_TEMP", None),
            "eps": getattr(free_engine_provider, "_EPS", None),
            "count": len(arr or []),
            "samples": (arr or [])[:5],
            "last_stats": getattr(free_engine_provider, "_last_stats", {}),
        }
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/free_clear")
async def api_free_clear(token: str = ""):
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized (use token query)")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        setattr(free_engine_provider, "_cache_ts", 0.0)
        setattr(free_engine_provider, "_cache_data", [])
        setattr(free_engine_provider, "_cache_key", set())
        return {"cleared": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scrape_status")
async def api_scrape_status(token: str = ""):
    # Lightweight diagnostics: check which scrapers are enabled and what they return right now
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized (use token query)")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    prov = list((all_enabled() or {}).keys())
    fixtures_count = 0
    fixtures_samples: list[dict] = []
    odds_count = 0
    odds_samples: list[dict] = []
    try:
        reg = all_enabled() or {}
        # Gather fixtures from any provider that exposes fixtures
        for name, s in reg.items():
            fx = s.get("fetch", {}).get("fixtures")
            if fx:
                try:
                    arr = await asyncio.wait_for(fx(), timeout=8.0)
                except Exception:
                    arr = []
                fixtures_count += len(arr or [])
                for it in (arr or [])[:3]:
                    fixtures_samples.append({
                        "provider": name,
                        "home": it.get("home"),
                        "away": it.get("away"),
                        "league": it.get("league"),
                        "time": it.get("time"),
                    })
        # Gather odds/probabilities from providers exposing odds_or_prob
        for name, s in reg.items():
            fn = s.get("fetch", {}).get("odds_or_prob")
            if fn:
                try:
                    arr = await asyncio.wait_for(fn(), timeout=8.0)
                except Exception:
                    arr = []
                odds_count += len(arr or [])
                for it in (arr or [])[:3]:
                    odds_samples.append({
                        "provider": name,
                        "home": it.get("home"),
                        "away": it.get("away"),
                        # show either odds or probs brief
                        "h": it.get("h"),
                        "d": it.get("d"),
                        "a": it.get("a"),
                        "probs": it.get("probs"),
                    })
    except Exception:
        pass
    return {
        "providers": prov,
        "fixtures_count": fixtures_count,
        "fixtures_samples": fixtures_samples,
        "odds_or_prob_count": odds_count,
        "odds_or_prob_samples": odds_samples,
    }

@app.post("/api/ai_suggest")
async def api_ai_suggest(token: str = "", top_n: int = 5):
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized (use token)")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    recs: list[dict] = []
    try:
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            rows = cur.execute(
                "SELECT model_name, version, metric, value, created_at FROM model_metrics ORDER BY id DESC LIMIT 200"
            ).fetchall()
            metrics = [dict(r) for r in rows]
    except Exception:
        metrics = []
    env = {
        "MIN_CONF": os.environ.get("MIN_CONF", ""),
        "EV_MIN": os.environ.get("EV_MIN", ""),
        "MAX_PICKS": os.environ.get("MAX_PICKS", ""),
        "MODEL_DIR": os.environ.get("MODEL_DIR", ""),
        "DB_PATH": DB_PATH,
    }
    def _avg(vals):
        xs = [float(v) for v in vals if v is not None]
        return sum(xs)/len(xs) if xs else None
    by_model = {}
    for m in metrics:
        k = m.get("model_name")
        by_model.setdefault(k, {}).setdefault(m.get("metric"), []).append(m.get("value"))
    for name, mm in by_model.items():
        avg_ll = _avg(mm.get("logloss", []))
        avg_br = _avg(mm.get("brier", []))
        if avg_ll is not None and avg_ll > 1.05:
            recs.append({"type": "hpo", "target": name, "suggest": "increase_capacity_or_regularize", "detail": {"avg_logloss": avg_ll}})
        if avg_br is not None and avg_br > 0.22:
            recs.append({"type": "calibration", "target": name, "suggest": "recalibrate_platt_or_isotonic", "detail": {"avg_brier": avg_br}})
    try:
        mc = float(env.get("MIN_CONF") or 0)
    except Exception:
        mc = 0.0
    try:
        ev = float(env.get("EV_MIN") or 0)
    except Exception:
        ev = 0.0
    if mc > 0.5:
        recs.append({"type": "threshold", "target": "MIN_CONF", "suggest": "lower_to_0.48_0.50"})
    if ev > 0.0:
        recs.append({"type": "threshold", "target": "EV_MIN", "suggest": "lower_to_-0.01_0.00"})
    if not recs:
        recs.append({"type": "general", "suggest": "run_train_start_and_rebuild", "detail": env})
    return {"suggestions": recs[:max(1, min(top_n, 20))], "env": env, "metrics_seen": len(metrics)}

@app.post("/api/train")
async def api_train(request: Request, background_tasks: BackgroundTasks, token: str = ""):
    # Placeholder async trigger; full training pipeline will be added in ml/
    if not token:
        token = request.headers.get("X-Api-Token", "")
    if not token:
        try:
            form = await request.form()
            token = form.get("token", "")
        except Exception:
            token = token or ""
    if not API_TOKEN or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Fire-and-forget: run training in background to avoid Render 502 on long calls
    try:
        ensure_db()
    except Exception:
        pass
    def _run():
        try:
            train_all()
        except Exception:
            pass
    background_tasks.add_task(_run)
    return {"status": "started"}

@app.get("/api/train_start")
async def api_train_start(request: Request, background_tasks: BackgroundTasks, token: str = ""):
    # GET-friendly trigger (some hosts restrict POST)
    if not token:
        token = request.headers.get("X-Api-Token", "")
    if not API_TOKEN or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    def _run():
        try:
            train_all()
        except Exception:
            pass
    background_tasks.add_task(_run)
    return {"status": "started"}

@app.get("/api/train_status")
async def api_train_status(token: str = ""):
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized (use token query)")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Check saved models and latest metrics
    model_dir = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
    files = sorted([os.path.basename(p) for p in glob(os.path.join(model_dir, "*.joblib"))])
    metrics = []
    try:
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            metrics = [dict(r) for r in cur.execute("SELECT model_name, version, metric, value, created_at FROM model_metrics ORDER BY id DESC LIMIT 20").fetchall()]
    except Exception:
        pass
    return {"models": files, "metrics": metrics}

@app.get("/api/train_fast")
async def api_train_fast(token: str = ""):
    # Ultra-fast seeding of models (dummy classifiers) to unblock ensemble
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized (use token query)")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    model_dir = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(model_dir, exist_ok=True)
    X = np.zeros((30, 6), dtype=float)
    y = np.array([0,1,2] * 10, dtype=int)
    names = ["modelA.joblib","modelB.joblib","modelC.joblib","meta.joblib"]
    for n in names:
        try:
            m = DummyClassifier(strategy="uniform")
            m.fit(X, y)
            dump(m, os.path.join(model_dir, n))
        except Exception:
            pass
    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            for nm in ["modelA","modelB","modelC","meta"]:
                cur.execute(
                    "INSERT INTO model_metrics(model_name, version, metric, value, created_at) VALUES(?,?,?,?,datetime('now'))",
                    (nm, "v0", "seed", 0.0),
                )
            con.commit()
    except Exception:
        pass
    return {"status": "ok", "seeded": names}

@app.get("/api/daily_top5")
async def api_daily_top5(limit: int = 5, explain: bool = False):
    # Open endpoint for mini-app: top-N picks for today based on EV and probability with simple league diversity
    try:
        allp = get_prepared_picks_for_today(limit=200) or []
    except Exception:
        allp = []
    def _score(p: dict) -> float:
        try:
            ev = float(p.get("edge") or 0.0)
        except Exception:
            ev = 0.0
        try:
            pr = float(p.get("prob") or 0.0)
        except Exception:
            pr = 0.0
        return ev + max(0.0, pr - 0.5)
    candidates = [p for p in allp if (p.get("category") not in {"info","demo"})]
    candidates.sort(key=_score, reverse=True)
    out: list[dict] = []
    per_league: dict[str,int] = {}
    for p in candidates:
        if len(out) >= max(1, min(limit, 10)):
            break
        lg = (p.get("league") or "").strip()
        if per_league.get(lg, 0) >= 2:
            continue
        out.append(p)
        per_league[lg] = per_league.get(lg, 0) + 1
    resp = {"date": datetime.now(timezone.utc).date().isoformat(), "count": len(out), "picks": out}
    if explain:
        try:
            pf = PredictionFactors()
            resp["legend"] = {
                "algorithm": pf.get_prediction_algorithm(),
            }
        except Exception:
            pass
    return resp

# --- Telegram Webhook (no aiogram) ---
@app.post("/tg/webhook")
async def tg_webhook(request: Request):
    secret = os.environ.get("TG_SECRET", "")
    got = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if secret and got != secret:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        update = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Bad JSON")
    msg = (update or {}).get("message") or (update or {}).get("edited_message")
    if not msg:
        return {"ok": True}
    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    text = (msg.get("text") or "").strip()
    if not chat_id:
        return {"ok": True}
    # Process in background to acknowledge telegram immediately
    async def _process():
        if text.lower().startswith("/start"):
            resp = await send_message(chat_id, "Привет! Я присылаю прогнозы 1X2. Нажмите /picks чтобы получить на сегодня.")
            try:
                if not (resp or {}).get("ok"):
                    logging.warning("tg send_message /start failed: %s", resp)
            except Exception:
                pass
            return
        if text.lower().startswith("/picks"):
            picks = get_prepared_picks_for_today(limit=10)
            if not picks:
                # Try any picks (including info/demo) if today is empty
                try:
                    with sqlite3.connect(DB_PATH) as con:
                        con.row_factory = sqlite3.Row
                        cur = con.cursor()
                        rows = cur.execute("SELECT id,title,text,category,ts FROM prepared_picks ORDER BY id DESC LIMIT 10").fetchall()
                        picks = [dict(r) for r in rows]
                except Exception:
                    picks = []
                if not picks:
                    resp = await send_message(chat_id, "Пока нет уверенных подборов. Попробуйте позже.")
                    try:
                        if not (resp or {}).get("ok"):
                            logging.warning("tg send_message /picks empty failed: %s", resp)
                    except Exception:
                        pass
                    return
            # Format concise list
            lines = []
            for p in picks[:10]:
                title = p.get("title") or ""
                percent = p.get("percent")
                cat = p.get("category")
                line = f"• {title}"
                if percent:
                    line += f" — {percent}%"
                if cat:
                    line += f" [{cat}]"
                lines.append(line)
            resp = await send_message(chat_id, "\n".join(lines))
            try:
                if not (resp or {}).get("ok"):
                    logging.warning("tg send_message /picks list failed: %s", resp)
            except Exception:
                pass
            return
        resp = await send_message(chat_id, "Команды: /picks — текущие прогнозы")
        try:
            if not (resp or {}).get("ok"):
                logging.warning("tg send_message default failed: %s", resp)
        except Exception:
            pass
    try:
        asyncio.create_task(_process())
    except Exception:
        pass
    return {"ok": True}

@app.post("/tg/ping")
async def tg_ping(request: Request, chat_id: str = "", text: str = "ping", token: str = ""):
    # Diagnostic: send a message to check BOT_TOKEN delivery; secured by API_TOKEN
    if not token:
        token = request.headers.get("X-Api-Token", "")
    if not API_TOKEN or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not chat_id:
        # try env publish chat
        chat_id = os.environ.get("PUBLISH_CHAT_ID", "")
    if not chat_id:
        raise HTTPException(status_code=400, detail="chat_id required")
    try:
        resp = await send_message(chat_id, text)
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/promote")
async def api_promote(request: Request, token: str = "", name: str = "", version: str = ""):
    if not token:
        token = request.headers.get("X-Api-Token", "")
    if not token:
        try:
            form = await request.form()
            token = form.get("token", "")
            name = name or form.get("name", "")
            version = version or form.get("version", "")
        except Exception:
            token = token or ""
    if not API_TOKEN or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not name or not version:
        raise HTTPException(status_code=400, detail="name and version required")
    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            # demote others
            cur.execute("UPDATE model_registry SET status='inactive' WHERE name=?", (name,))
            # promote target (insert if missing)
            cur.execute(
                "INSERT INTO model_registry(name, version, path, status, created_at) VALUES(?,?,?,?,datetime('now'))",
                (name, version, '', 'active')
            )
            con.commit()
        return {"promoted": {"name": name, "version": version}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rollback")
async def api_rollback(request: Request, token: str = "", name: str = ""):
    if not token:
        token = request.headers.get("X-Api-Token", "")
    if not token:
        try:
            form = await request.form()
            token = form.get("token", "")
            name = name or form.get("name", "")
        except Exception:
            token = token or ""
    if not API_TOKEN or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            # set last inactive as active
            row = cur.execute(
                "SELECT version FROM model_registry WHERE name=? AND status='inactive' ORDER BY id DESC LIMIT 1",
                (name,)
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="no previous version")
            prev_ver = row[0]
            cur.execute("UPDATE model_registry SET status='inactive' WHERE name=?", (name,))
            cur.execute(
                "INSERT INTO model_registry(name, version, path, status, created_at) VALUES(?,?,?,?,datetime('now'))",
                (name, prev_ver, '', 'active')
            )
            con.commit()
        return {"rolled_back": {"name": name, "version": prev_ver}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/seed")
async def api_seed(request: Request, token: str = "", n: int = 5):
    # Accept token from query, header, or form
    if not token:
        token = request.headers.get("X-Api-Token", "")
    if not token:
        try:
            form = await request.form()
            token = form.get("token", "")
            if "n" in form and not n:
                try:
                    n = int(form.get("n"))
                except Exception:
                    pass
        except Exception:
            token = token or ""
    if not API_TOKEN or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    picks = []
    for i in range(max(1, min(n, 10))):
        title = f"Демо-прогноз #{i+1}"
        text = (
            f"Оценка проходимости: {65 + i*3}%\n"
            f"Рекомендация: Победа\n"
            f"- Демо данные (ручная инициализация)\n"
            f"- Время генерации: {now}"
        )
        picks.append({"title": title, "text": text, "category": "demo", "ts": now})
    save_prepared_picks(picks)
    return {"inserted": len(picks)}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    picks: List[Dict] = get_prepared_picks_for_today(limit=10)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Актуальные прогнозы на сегодня",
            "picks": picks,
        },
    )

@app.get("/tgapp", response_class=HTMLResponse)
async def tgapp(request: Request):
    picks: List[Dict] = get_prepared_picks_for_today(limit=10)
    return templates.TemplateResponse(
        "tgapp.html",
        {
            "request": request,
            "title": "Прогнозы — Mini App",
            "picks": picks,
        },
    )

@app.get("/api/explain_factors")
async def api_explain_factors():
    try:
        pf = PredictionFactors()
        return {
            "factors": pf.factors,
            "algorithm": pf.get_prediction_algorithm(),
            "diagram": create_prediction_factors_diagram(),
            "flow": create_algorithm_flow(),
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/free_train")
async def api_free_train(request: Request, token: str = ""):
    if not token:
        token = request.headers.get("X-Api-Token", "")
    if not API_TOKEN or token != API_TOKEN:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Unauthorized")
    import os
    os.makedirs("free", exist_ok=True)
    import numpy as np
    np.random.seed(42)
    n_matches = 1000
    teams = ['Arsenal','Chelsea','Liverpool','Manchester City','Manchester United','Tottenham','Barcelona','Real Madrid','Atletico Madrid','Bayern Munich']
    data = {
        'home_team': np.random.choice(teams, n_matches),
        'away_team': np.random.choice(teams, n_matches),
        'competition': np.random.choice(['Premier League','La Liga','Bundesliga'], n_matches),
        'match_date': pd.date_range('2023-01-01', periods=n_matches, freq='D')
    }
    res = []
    for i in range(n_matches):
        if data['home_team'][i] == data['away_team'][i]:
            res.append(np.random.choice(['HOME_WIN','DRAW','AWAY_WIN'], p=[0.4,0.3,0.3]))
        else:
            res.append(np.random.choice(['HOME_WIN','DRAW','AWAY_WIN'], p=[0.5,0.25,0.25]))
    data['result'] = res
    df = pd.DataFrame(data)
    # Augment with today's fixtures to cover real team names in label encoders
    try:
        fixtures = get_today_fixtures() or []
    except Exception:
        fixtures = []
    if fixtures:
        import random
        random.seed(42)
        aug = []
        for f in fixtures:
            h = (f.get("home") or "").strip()
            a = (f.get("away") or "").strip()
            if not h or not a:
                continue
            comp = (f.get("league") or f.get("competition") or "General").strip() or "General"
            r = random.choices(["HOME_WIN","DRAW","AWAY_WIN"], weights=[0.5,0.25,0.25], k=1)[0]
            aug.append({"home_team": h, "away_team": a, "competition": comp, "match_date": pd.Timestamp("2024-01-01"), "result": r})
        if aug:
            df = pd.concat([df, pd.DataFrame(aug)], ignore_index=True)
    eng = FootballPredictionEngine()
    metrics = eng.train_model(df)
    try:
        eng.save_model('free/football_model.pkl')
    except Exception:
        pass
    try:
        eng.save_model('football_model.pkl')
    except Exception:
        pass
    # Persist metrics to DB
    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            name = "free_engine"
            ver = "v0"
            for k in ["accuracy","logloss","brier"]:
                v = metrics.get(k)
                if v is None:
                    continue
                try:
                    cur.execute(
                        "INSERT INTO model_metrics(model_name, version, metric, value, created_at) VALUES(?,?,?,?,datetime('now'))",
                        (name, ver, k, float(v)),
                    )
                except Exception:
                    continue
            # Update registry status (demote others)
            try:
                cur.execute("UPDATE model_registry SET status='inactive' WHERE name=?", (name,))
            except Exception:
                pass
            cur.execute(
                "INSERT INTO model_registry(name, version, path, status, created_at) VALUES(?,?,?,?,datetime('now'))",
                (name, ver, 'free/football_model.pkl', 'active')
            )
            con.commit()
    except Exception:
        pass
    return {"trained": True, "metrics": metrics}

@app.post("/api/free_predict")
async def api_free_predict(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    home = (body.get("home") or body.get("home_team") or "").strip()
    away = (body.get("away") or body.get("away_team") or "").strip()
    competition = (body.get("competition") or "Premier League").strip()
    if not home or not away:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="home and away required")
    import os
    eng = FootballPredictionEngine()
    loaded = False
    for p in ["free/football_model.pkl","football_model.pkl"]:
        try:
            if os.path.exists(p):
                eng.load_model(p)
                loaded = True
                break
        except Exception:
            continue
    if not loaded:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="model not available; train via /api/free_train")
    pred = eng.predict_match(home, away, competition)
    return pred

@app.post("/api/fixtures_seed")
async def api_fixtures_seed(token: str = ""):
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized (use token query)")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    fixtures = [
        {"date": today, "league": "England Premier League", "home": "Arsenal", "away": "Chelsea"},
        {"date": today, "league": "Spain La Liga", "home": "Barcelona", "away": "Real Madrid"},
        {"date": today, "league": "Germany Bundesliga", "home": "Bayern Munich", "away": "Borussia Dortmund"},
    ]
    try:
        upsert_fixtures(fixtures)
        return {"inserted": len(fixtures)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
