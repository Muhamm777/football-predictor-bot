import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from config import UPDATE_CRON, TIMEZONE, POPULAR_LEAGUES, PUBLISH_CHAT_ID, BOT_TOKEN
from storage.db import save_prepared_picks, get_prepared_picks_for_today, upsert_fixtures, make_fixture_id
from storage.features import save_features
from scrapers.soccer365 import fetch_fixtures
from scrapers.flashscore import fetch_odds
from scrapers.sstats import fetch_team_stats
from scrapers.vprognoze import fetch_comments
from datetime import datetime, timezone, timedelta
from predictor.ensemble import combined_prediction
from aiogram import Bot
from time import perf_counter
from metrics.logger import log_metrics
from scrapers.registry import all_enabled

# Placeholder jobs: fetch data and rebuild predictions 3x daily

scheduler = AsyncIOScheduler(timezone=TIMEZONE)

async def job_update_all():
    logging.info("Nightly job: scraping -> analysis -> predictions (light placeholder)")
    t0 = perf_counter()
    fixtures_count = 0
    candidates_count = 0
    try:
        fixtures = await fetch_fixtures()
        # Filter to popular leagues if provided (football-only expectation)
        if POPULAR_LEAGUES:
            fixtures = [f for f in fixtures if (f.get('league') or '') in POPULAR_LEAGUES]
        fixtures_count = len(fixtures or [])
        # Try to augment fixtures from enabled registry scrapers (if any provide fixtures)
        try:
            for name, s in (all_enabled() or {}).items():
                fx = s.get('fetch', {}).get('fixtures')
                if fx:
                    try:
                        extra = await fx()
                        if POPULAR_LEAGUES:
                            extra = [f for f in (extra or []) if (f.get('league') or '') in POPULAR_LEAGUES]
                        if extra:
                            fixtures.extend(extra)
                    except Exception:
                        logging.warning("Registry fixtures fetch failed for %s", name)
            fixtures_count = len(fixtures or [])
        except Exception:
            logging.warning("Registry fixtures merge failed")
        # Нормализация и словарь соответствий названий команд/лиг
        ALIAS_MAP = {
            # Команды (пополняется по мере встречаемости)
            "manchester united": ["man utd", "manchester utd", "man united"],
            "manchester city": ["man city", "manchester c."],
            "real madrid": ["r. madrid", "real m.", "real-madrid"],
            "barcelona": ["fc barcelona", "barca", "barça"],
            "bayern munich": ["bayern", "bayern münchen", "bayern m."],
            # Лиги
            "premier league": ["england premier league", "epl"],
            "la liga": ["laliga", "spain la liga"],
        }

        def _basic_norm(x: str) -> str:
            x = (x or "").lower().strip()
            for ch in ["'", "`", "´", "’", ".", ",", "-", "_", "(", ")", "/", "\\"]:
                x = x.replace(ch, " ")
            for suf in [" fc", " cf", " fk", " sc", " ac", " u19", " u21", " u23", " ii", " b", " youth"]:
                if x.endswith(suf):
                    x = x[: -len(suf)]
            while "  " in x:
                x = x.replace("  ", " ")
            return x.strip()

        def norm(s: str) -> str:
            b = _basic_norm(s)
            # Прямое совпадение ключа
            if b in ALIAS_MAP:
                return b
            # Совпадение по алиасам
            for canon, aliases in ALIAS_MAP.items():
                if b == canon:
                    return canon
                if b in ( _basic_norm(a) for a in aliases ):
                    return canon
            return b

        # Подтянем котировки Flashscore (если недоступно — продолжим без них)
        odds_map = {}
        try:
            odds_list = await fetch_odds()
            for o in odds_list or []:
                key = f"{norm(o.get('home',''))}|{norm(o.get('away',''))}"
                odds_map[key] = {"h": o.get("h"), "d": o.get("d"), "a": o.get("a")}
        except Exception:
            logging.warning("Flashscore odds fetch failed; continuing with base signals")
        # Merge odds/probabilities from registry scrapers (if present)
        try:
            for name, s in (all_enabled() or {}).items():
                fn = s.get('fetch', {}).get('odds_or_prob')
                if fn:
                    try:
                        arr = await fn()
                        for o in arr or []:
                            key = f"{norm(o.get('home',''))}|{norm(o.get('away',''))}"
                            # Allow both odds triplet and probability triplet
                            if 'h' in o or 'd' in o or 'a' in o:
                                odds_map[key] = {"h": o.get("h"), "d": o.get("d"), "a": o.get("a")}
                            elif 'home' in o.get('probs', {}) or 'draw' in o.get('probs', {}) or 'away' in o.get('probs', {}):
                                odds_map[key] = {"h": o.get('probs', {}).get('home'), "d": o.get('probs', {}).get('draw'), "a": o.get('probs', {}).get('away')}
                    except Exception:
                        logging.warning("Registry odds fetch failed for %s", name)
        except Exception:
            logging.warning("Registry odds merge failed")

        # Форма команд (sstats) — агрегируем по имени команды
        form_map = {}
        try:
            stats = await fetch_team_stats()
            for s in stats or []:
                t = norm(s.get('team',''))
                if not t:
                    continue
                form_map[t] = {
                    "home_wdl": s.get("home_wdl"),
                    "away_wdl": s.get("away_wdl"),
                    "gf": s.get("gf"),
                    "ga": s.get("ga"),
                }
        except Exception:
            logging.warning("Sstats fetch failed; continuing without form")

        # Комментарии/сентимент (vprognoze) — ключ по паре
        sent_map = {}
        try:
            comms = await fetch_comments()
            for c in comms or []:
                key = f"{norm(c.get('home',''))}|{norm(c.get('away',''))}"
                sent = c.get("sentiment", 0.0)
                sent_map[key] = float(sent) if isinstance(sent, (int, float)) else 0.0
        except Exception:
            logging.warning("Vprognoze fetch failed; continuing without sentiment")
        # Persist fixtures for today
        try:
            upsert_fixtures(fixtures)
        except Exception:
            logging.warning("Failed to persist fixtures; continuing")
        picks = []
        upcoming = []
        # If no fixtures found, create demo picks and save immediately
        if not fixtures:
            now = datetime.now(timezone.utc).isoformat()
            for i in range(3):
                title = f"Демо-прогноз #{i+1}"
                text = (
                    f"Оценка проходимости: {60 + i*5}%\n"
                    f"Рекомендация: Победа\n"
                    f"- Демо данные (источники недоступны сейчас)\n"
                    f"- Рынок 1X2: недоступно\n"
                    f"- Форма: дом 3-1-1, выезд 2-2-1\n"
                    f"- Время генерации: {now}"
                )
                picks.append({"title": title, "text": text, "category": "demo", "ts": now})
            if picks:
                save_prepared_picks(picks)
                logging.info("Prepared demo picks saved: %d", len(picks))
            return
        categories = [
            ("win", "Победа"),
            ("total", "Тотал"),
            ("corners", "Угловые"),
            ("both_to_score", "Обе забьют"),
            ("handicap", "Фора"),
        ]

        candidates = []
        for idx, tup in enumerate(upcoming[:30] if upcoming else fixtures[:30]):
            base = tup[1] if isinstance(tup, tuple) else tup
            cat_key, cat_title = categories[idx % len(categories)]
            title = f"{cat_title}: {base.get('home','')} vs {base.get('away','')}"
            # Compute probability using current ensemble with odds if available
            key = f"{norm(base.get('home',''))}|{norm(base.get('away',''))}"
            features = {"stats": {}}
            if key in odds_map:
                features["odds"] = odds_map[key]
            # Добавим сигналы формы (если есть)
            h_form = form_map.get(norm(base.get('home','')))
            a_form = form_map.get(norm(base.get('away','')))
            if h_form or a_form:
                features["stats"].update({
                    "home_form": h_form,
                    "away_form": a_form,
                })
            # Добавим сентимент (если есть)
            if key in sent_map:
                features["sentiment_signal"] = sent_map[key]
            probs = combined_prediction(features)
            best_prob = max(probs.get("home",0.0), probs.get("draw",0.0), probs.get("away",0.0))
            percent = round(best_prob * 100.0, 1)
            # Сформируем краткие причины (черновик)
            reasons = []
            lg = base.get('league','')
            if lg:
                reasons.append(f"Лига: {lg}")
            tm = base.get('time','')
            if tm:
                reasons.append(f"Время матча: {tm}")
            if key in odds_map:
                o = odds_map[key]
                reasons.append(f"Рынок 1X2: {o.get('h')} / {o.get('d')} / {o.get('a')}")
            else:
                reasons.append("Без учёта рынка (нет котировок)")
            if h_form or a_form:
                hf = h_form.get('home_wdl') if h_form else None
                af = a_form.get('away_wdl') if a_form else None
                segs = []
                if hf:
                    segs.append(f"дом {hf}")
                if af:
                    segs.append(f"выезд {af}")
                if segs:
                    reasons.append("Форма: " + ", ".join(segs))
            if key in sent_map:
                s = sent_map[key]
                if s > 0.2:
                    reasons.append("Сентимент: позитивный")
                elif s < -0.2:
                    reasons.append("Сентимент: негативный")
            text = (
                f"Оценка проходимости: {percent}%\n"
                f"Рекомендация: {cat_title}\n"
                + ("\n".join([f"- {r}" for r in reasons]))
            )
            candidates.append({
                "title": title,
                "text": text,
                "category": cat_key,
                "ts": datetime.now(timezone.utc).isoformat(),
                "prob": best_prob,
            })
            # Persist per-match features into FeatureStore so other modules can reuse
            try:
                # Build a stable match_id (date|home|away) using UTC date
                dt_raw = base.get('time') or ''
                dt_obj = None
                try:
                    dt_obj = datetime.fromisoformat(dt_raw)
                except Exception:
                    dt_obj = None
                match_date = (dt_obj.date().isoformat() if isinstance(dt_obj, datetime) else datetime.now(timezone.utc).date().isoformat())
                match_id = make_fixture_id(match_date, base.get('home',''), base.get('away',''))
                save_features(match_id, {
                    "league": base.get('league',''),
                    "time": dt_raw,
                    "normalized_key": key,
                    "odds": features.get('odds'),
                    "stats": features.get('stats'),
                    "sentiment_signal": features.get('sentiment_signal'),
                    "probs": probs,
                    "best_prob": best_prob,
                })
            except Exception:
                logging.warning("FeatureStore save failed for match %s vs %s", base.get('home',''), base.get('away',''))
        # Keep only those strictly above 0.93 and take the TOP-5 among them (no fallback below threshold)
        candidates.sort(key=lambda x: x.get("prob", 0.0), reverse=True)
        candidates_count = len(candidates)
        selected = [c for c in candidates if c.get("prob", 0.0) > 0.93][:5]
        # Strip prob from final payload
        for c in selected:
            c.pop("prob", None)
        picks.extend(selected)
        if picks:
            save_prepared_picks(picks)
            logging.info("Prepared picks saved: %d", len(picks))
        else:
            logging.info("No fixtures fetched; prepared picks not updated")
    except Exception as e:
        logging.exception("Nightly job failed: %s", e)
    finally:
        try:
            dt = perf_counter() - t0
            log_metrics('scheduler', {
                'fixtures_count': fixtures_count,
                'candidates_count': candidates_count,
                'picked_count': len(picks or []),
                'duration_s': round(dt, 3),
            })
        except Exception:
            pass

async def job_publish():
    if not PUBLISH_CHAT_ID or not BOT_TOKEN:
        logging.info("Publish skipped: PUBLISH_CHAT_ID or BOT_TOKEN not configured")
        return
    picks = get_prepared_picks_for_today(limit=10)
    if not picks:
        logging.info("Publish skipped: no prepared picks for today")
        return
    bot = Bot(BOT_TOKEN)
    try:
        chunks = []
        # Split into chunks of 5 to avoid overly long messages
        for i in range(0, len(picks), 5):
            part = picks[i:i+5]
            lines = []
            for p in part:
                title = p.get("title", "")
                text = p.get("text", "")
                lines.append(f"• {title}\n{text}")
            chunks.append("Подготовленные прогнозы:\n" + "\n\n".join(lines))
        for msg in chunks:
            await bot.send_message(chat_id=PUBLISH_CHAT_ID, text=msg)
        logging.info("Published %d picks to chat %s", len(picks), PUBLISH_CHAT_ID)
    except Exception:
        logging.exception("Publish job failed")
    finally:
        await bot.session.close()

async def start_scheduler():
    # Use cron expression from config (default corresponds to hourly if misconfigured)
    try:
        trigger = CronTrigger.from_crontab(UPDATE_CRON, timezone=TIMEZONE)
    except Exception:
        # Fallback to hourly
        trigger = CronTrigger(minute=0, timezone=TIMEZONE)
    scheduler.add_job(job_update_all, trigger=trigger)
    scheduler.start()
