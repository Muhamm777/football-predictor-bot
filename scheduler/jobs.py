import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from config import UPDATE_CRON, TIMEZONE, POPULAR_LEAGUES, PUBLISH_CHAT_ID, BOT_TOKEN
from config import EV_MIN, MIN_CONF, MAX_PICKS
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
import scrapers  # ensure registry scrapers self-register on import

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
            # Skip garbage fixtures without proper team names
            if not base.get('home') or not base.get('away'):
                continue
            if str(base.get('home')).strip().lower() in {"vs","-",""} or str(base.get('away')).strip().lower() in {"vs","-",""}:
                continue
            probs = combined_prediction(features)
            best_prob = max(probs.get("home",0.0), probs.get("draw",0.0), probs.get("away",0.0))
            # Compute market implied probabilities for EV if odds present
            mkt = None
            if key in odds_map:
                o = odds_map[key]
                try:
                    ih = (1.0/float(o.get('h'))) if o.get('h') else None
                    idr = (1.0/float(o.get('d'))) if o.get('d') else None
                    ia = (1.0/float(o.get('a'))) if o.get('a') else None
                    if ih and idr and ia:
                        s = ih+idr+ia
                        if s>0:
                            mkt = {"home": ih/s, "draw": idr/s, "away": ia/s}
                except Exception:
                    mkt = None
            # Choose outcome with max model prob and compute EV edge vs market
            outcome = max(("home","draw","away"), key=lambda k: probs.get(k,0.0))
            ev = None
            try:
                if mkt is not None:
                    ev = probs.get(outcome,0.0) - mkt.get(outcome,0.0)
            except Exception:
                ev = None
            percent = round(best_prob * 100.0, 1)
            # Сформируем краткие причины (фокус на 1X2)
            reasons = []
            lg = base.get('league','')
            if lg:
                reasons.append(f"Лига: {lg}")
            tm = base.get('time','')
            if tm:
                reasons.append(f"Время матча: {tm}")
            if mkt is not None and key in odds_map:
                o = odds_map[key]
                reasons.append(f"Рынок 1X2: {round(mkt['home']*100,1)}% / {round(mkt['draw']*100,1)}% / {round(mkt['away']*100,1)}% (коэф: {o.get('h','?')} / {o.get('d','?')} / {o.get('a','?')})")
            # Determine category text for UI
            pick_map = {"home": "П1", "draw": "Х", "away": "П2"}
            cat_key = "win"
            cat_title = "Победа"
            title = f"{cat_title}: {base.get('home','')} vs {base.get('away','')}"
            # Make explicit recommendation line with model vs market
            rec_line = None
            try:
                model_pct = round(probs.get(outcome,0.0)*100, 1)
                if mkt is not None:
                    market_pct = round(mkt.get(outcome,0.0)*100, 1)
                    ev_pct = round((probs.get(outcome,0.0) - mkt.get(outcome,0.0))*100, 1)
                    rec_line = f"Выбор: {pick_map[outcome]} (модель {model_pct}%, рынок {market_pct}%, EV +{ev_pct}%)"
                else:
                    rec_line = f"Выбор: {pick_map[outcome]} (модель {model_pct}%)"
            except Exception:
                rec_line = None
            lines = []
            if rec_line:
                lines.append(rec_line)
            lines.append(f"Оценка проходимости (макс из 1/Х/2): {percent}%")
            lines.extend([f"- {r}" for r in reasons])
            text = "\n".join(lines)
            candidates.append({
                "title": title,
                "text": text,
                "category": cat_key,
                "ts": datetime.now(timezone.utc).isoformat(),
                "prob": best_prob,
                "edge": ev,
                "outcome": outcome,
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
        # Rank by probability (desc)
        candidates.sort(key=lambda x: x.get("prob", 0.0), reverse=True)
        candidates_count = len(candidates)
        # Primary selection by EV and confidence thresholds
        def _pass_ev_conf(c: dict) -> bool:
            p = float(c.get("prob", 0.0) or 0.0)
            e = c.get("edge", None)
            if p < MIN_CONF:
                return False
            if e is None:
                # If нет рынка, пропускаем в первичный список только очень уверенные
                return p >= max(MIN_CONF, 0.70)
            return (e >= EV_MIN)

        selected = [c for c in candidates if _pass_ev_conf(c)][:MAX_PICKS]
        # Fallback 1: relax EV to EV_MIN*0.5, keep MIN_CONF, top up to MAX_PICKS
        if len(selected) < MAX_PICKS:
            relaxed = []
            for c in candidates:
                if c in selected:
                    continue
                p = float(c.get("prob",0.0) or 0.0)
                e = c.get("edge", None)
                if p >= MIN_CONF and ((e is not None and e >= (EV_MIN*0.5)) or (e is None and p >= max(MIN_CONF, 0.65))):
                    relaxed.append(c)
            for f in relaxed:
                if len(selected) >= MAX_PICKS:
                    break
                selected.append(f)
        # Fallback 2: fill with top by prob but only если prob>=0.45 (up to MAX_PICKS)
        if len(selected) < MAX_PICKS and candidates:
            for f in candidates:
                if len(selected) >= MAX_PICKS:
                    break
                if f in selected:
                    continue
                if float(f.get("prob",0.0) or 0.0) >= 0.45:
                    selected.append(f)
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
