import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from config import BOT_TOKEN, MAX_DAILY_PICKS, WEB_BASE_URL, API_TOKEN
from storage.db import ensure_db
from scheduler.jobs import start_scheduler
from scrapers.soccer365 import fetch_fixtures
from storage.db import get_prepared_picks_for_today
import httpx

logging.basicConfig(level=logging.INFO)

# In-memory override for WEB_BASE_URL (useful for quick tunnel swaps)
_current_web_base_url = WEB_BASE_URL

def get_web_base_url() -> str:
    return _current_web_base_url

async def cmd_start(message: types.Message):
    kb = types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="Открыть Mini App", web_app=types.WebAppInfo(url=f"{get_web_base_url()}/tgapp"))],
    ])
    await message.answer("Привет! Я бот футбольных прогнозов. Доступные команды: /help, /today, /predict, /picks, /mini.\nНажмите кнопку ниже, чтобы открыть Mini App.", reply_markup=kb)

async def cmd_help(message: types.Message):
    await message.answer("Список команд:\n/start — приветствие\n/help — список команд\n/today — матчи на сегодня\n/predict — сгенерировать прогнозы (заглушка)\n/picks — получить 10 подготовленных прогнозов\n/mini — открыть Mini App")

async def cmd_today(message: types.Message):
    # Fetch today fixtures from soccer365
    fixtures = await fetch_fixtures()
    if not fixtures:
        await message.answer("Матчи на сегодня не найдены или источник временно недоступен.")
        return
    lines = []
    for it in fixtures[:15]:
        t = it.get("time", "")
        lg = it.get("league", "")
        h = it.get("home", "")
        a = it.get("away", "")
        line = f"{t} | {lg} — {h} vs {a}".strip()
        lines.append(line)
    text = "Матчи на сегодня (до 15):\n" + "\n".join(lines)
    await message.answer(text)

async def cmd_predict(message: types.Message):
    # TODO: parse match selection and run predictor
    await message.answer(f"Отправлю до {MAX_DAILY_PICKS} прогнозов с наибольшей уверенностью. Функция в разработке.")

async def cmd_picks(message: types.Message):
    # Optional args: /picks [limit]
    parts = (message.text or "").split()
    limit = 5
    if len(parts) > 1:
        try:
            limit = max(1, min(30, int(parts[1])))
        except Exception:
            limit = 5
    picks = get_prepared_picks_for_today(limit=limit)
    if not picks:
        await message.answer("Готовых прогнозов на сегодня нет. Ночные расчёты могли не выполниться.")
        return
    lines = []
    for p in picks:
        title = p.get("title", "")
        text = p.get("text", "")
        lines.append(f"• {title}\n{text}")
    await message.answer(f"Подготовленные прогнозы ({len(picks)}):\n" + "\n\n".join(lines))

async def cmd_mini(message: types.Message):
    kb = types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="Открыть Mini App", web_app=types.WebAppInfo(url=f"{get_web_base_url()}/tgapp"))]
    ])
    await message.answer("Откройте Mini App по кнопке ниже.", reply_markup=kb)

async def cmd_web(message: types.Message):
    kb = types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="Открыть Mini App", web_app=types.WebAppInfo(url=f"{get_web_base_url()}/tgapp"))]
    ])
    await message.answer(f"Текущий адрес Mini App:\n{get_web_base_url()}", reply_markup=kb)

async def cmd_rebuild(message: types.Message):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{get_web_base_url()}/api/rebuild", headers={"X-Api-Token": API_TOKEN})
        if r.status_code == 200:
            await message.answer("Пересчёт запущен.")
        else:
            await message.answer(f"Пересчёт не запущен: {r.status_code} {r.text}")
    except Exception as e:
        await message.answer(f"Ошибка пересчёта: {e}")

async def cmd_seed(message: types.Message):
    # Optional args: /seed [n]
    parts = (message.text or "").split()
    n = 6
    if len(parts) > 1:
        try:
            n = max(1, min(30, int(parts[1])))
        except Exception:
            n = 6
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{get_web_base_url()}/api/seed",
                headers={"X-Api-Token": API_TOKEN},
                data={"n": str(n)}
            )
        if r.status_code == 200:
            await message.answer(f"Засев демо завершён (n={n}). Откройте Mini App и нажмите ‘Обновить’.")
        else:
            await message.answer(f"Засев не выполнен: {r.status_code} {r.text}")
    except Exception as e:
        await message.answer(f"Ошибка засева: {e}")

async def cmd_seturl(message: types.Message):
    global _current_web_base_url
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Использование: /seturl https://example.com")
        return
    new_url = parts[1].strip()
    if not (new_url.startswith("http://") or new_url.startswith("https://")):
        await message.answer("URL должен начинаться с http:// или https://")
        return
    _current_web_base_url = new_url.rstrip('/')
    kb = types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="Открыть Mini App", web_app=types.WebAppInfo(url=f"{get_web_base_url()}/tgapp"))]
    ])
    try:
        await message.bot.set_chat_menu_button(
            menu_button=types.MenuButtonWebApp(
                text="Открыть прогнозы",
                web_app=types.WebAppInfo(url=f"{get_web_base_url()}/tgapp"),
            )
        )
    except Exception:
        logging.exception("Не удалось обновить кнопку меню WebApp после /seturl")
    await message.answer(f"WEB_BASE_URL обновлён на:\n{get_web_base_url()}", reply_markup=kb)

async def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is not set. Put it in .env or environment.")

    ensure_db()

    bot = Bot(BOT_TOKEN)
    dp = Dispatcher()

    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_help, Command("help"))
    dp.message.register(cmd_today, Command("today"))
    dp.message.register(cmd_predict, Command("predict"))
    dp.message.register(cmd_picks, Command("picks"))
    dp.message.register(cmd_mini, Command("mini"))
    dp.message.register(cmd_web, Command("web"))
    dp.message.register(cmd_rebuild, Command("rebuild"))
    dp.message.register(cmd_seed, Command("seed"))
    dp.message.register(cmd_seturl, Command("seturl"))

    # Установим кнопку меню (сбоку/внизу рядом с полем ввода) для открытия Mini App по умолчанию
    try:
        await bot.set_chat_menu_button(
            menu_button=types.MenuButtonWebApp(
                text="Открыть прогнозы",
                web_app=types.WebAppInfo(url=f"{get_web_base_url()}/tgapp"),
            )
        )
    except Exception:
        logging.exception("Не удалось установить кнопку меню WebApp")

    await start_scheduler()

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
