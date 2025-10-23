from __future__ import annotations
import os
from typing import Optional
import httpx

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"


async def send_message(chat_id: int | str, text: str, parse_mode: Optional[str] = "HTML"):  
    if not BOT_TOKEN:
        return {"ok": False, "error": "BOT_TOKEN not set"}
    data = {"chat_id": chat_id, "text": text}
    if parse_mode:
        data["parse_mode"] = parse_mode
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(f"{TELEGRAM_API}/sendMessage", data=data)
        try:
            return r.json()
        except Exception:
            return {"ok": False, "status_code": r.status_code}
