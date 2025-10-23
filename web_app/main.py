from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Dict
from storage.db import get_prepared_picks_for_today, save_prepared_picks
from pathlib import Path
import asyncio
from config import API_TOKEN
from fastapi import HTTPException, BackgroundTasks
from scheduler.jobs import job_update_all
from scheduler.jobs import start_scheduler
import sqlite3
from config import DB_PATH
from storage.db import ensure_db
from ml.train_all import train_all
import os
from glob import glob
from joblib import dump
import numpy as np
from sklearn.dummy import DummyClassifier
import logging
from web_app.telegram_client import send_message

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

@app.get("/api/metrics")
async def api_metrics(token: str = ""):
    # Simple metrics dump (secured via query header like other endpoints)
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized (use token query)")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    out = {"registry": [], "metrics": []}
    try:
        with sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            out["registry"] = [dict(r) for r in cur.execute("SELECT name, version, status, created_at FROM model_registry ORDER BY id DESC LIMIT 50").fetchall()]
            out["metrics"] = [dict(r) for r in cur.execute("SELECT model_name, version, metric, value, created_at FROM model_metrics ORDER BY id DESC LIMIT 100").fetchall()]
    except Exception as e:
        return {"error": str(e)}
    return out

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
