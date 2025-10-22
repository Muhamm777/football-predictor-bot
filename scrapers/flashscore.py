from typing import Any, Dict, List
from playwright.async_api import async_playwright
import asyncio

async def fetch_odds() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(channel="msedge", args=["--headless=new"]) 
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto("https://www.flashscore.com/football/", wait_until="domcontentloaded")
            try:
                btn = await page.query_selector("button#onetrust-accept-btn-handler")
                if btn:
                    await btn.click()
            except Exception:
                pass
            await asyncio.sleep(1.0)
            matches = await page.query_selector_all("div.event__match")
            for m in matches[:50]:
                try:
                    home_el = await m.query_selector("div.event__participant--home")
                    away_el = await m.query_selector("div.event__participant--away")
                    if not home_el or not away_el:
                        continue
                    home = (await home_el.inner_text()).strip()
                    away = (await away_el.inner_text()).strip()
                    h = d = a = None
                    odds_btn = await m.query_selector("a.event__more")
                    if odds_btn:
                        await odds_btn.click()
                        await asyncio.sleep(0.4)
                        oh = await page.query_selector(".oddsCell__odd--home")
                        od = await page.query_selector(".oddsCell__odd--draw")
                        oa = await page.query_selector(".oddsCell__odd--away")
                        if oh and od and oa:
                            try:
                                h = float((await oh.inner_text()).strip())
                                d = float((await od.inner_text()).strip())
                                a = float((await oa.inner_text()).strip())
                            except Exception:
                                h = d = a = None
                    if home and away and h and d and a:
                        out.append({"home": home, "away": away, "h": h, "d": d, "a": a})
                except Exception:
                    continue
            await browser.close()
    except Exception:
        return []
    return out

async def fetch_odds() -> List[Dict[str, Any]]:
    # TODO: implement Playwright-based scraping for flashscore odds
    data: List[Dict[str, Any]] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="msedge", args=["--headless=new"])
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto("https://www.flashscore.com.ua/")
        # TODO: navigate to odds pages, extract data, respect delays
        await browser.close()
    return data
