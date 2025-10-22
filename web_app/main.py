from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Dict
from storage.db import get_prepared_picks_for_today, save_prepared_picks
from pathlib import Path
import asyncio
from config import API_TOKEN
from fastapi import HTTPException
from scheduler.jobs import job_update_all
import sqlite3
from config import DB_PATH
from storage.db import ensure_db
from ml.train_all import train_all

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

@app.get("/health", response_class=HTMLResponse)
async def health() -> str:
    return "OK"

@app.get("/api/picks")
async def api_picks(limit: int = 10):
    picks: List[Dict] = get_prepared_picks_for_today(limit=limit)
    return picks

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

@app.post("/api/train")
async def api_train(request: Request, token: str = ""):
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
    # For now: run minimal stub to persist placeholder models, then return
    try:
        ensure_db()
    except Exception:
        pass
    try:
        res = train_all()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"train failed: {e}")
    return {"status": "ok", "result": res}

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
