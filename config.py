import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the project root regardless of current working directory
_DOTENV_PATH = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=_DOTENV_PATH)

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
UPDATE_CRON = os.getenv("UPDATE_CRON", "0 1 * * *")
MAX_DAILY_PICKS = int(os.getenv("MAX_DAILY_PICKS", "10"))
DB_PATH = os.getenv("DB_PATH", "storage/data.db")
CACHE_DIR = os.getenv("CACHE_DIR", "cache/")
USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
REQUESTS_TIMEOUT = float(os.getenv("REQUESTS_TIMEOUT", "20"))
RETRIES = int(os.getenv("RETRIES", "3"))
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "0.8"))
RATE_LIMIT_RPS = float(os.getenv("RATE_LIMIT_RPS", "1.0"))

# Proxy (HTTP(S) or SOCKS). Example: http://user:pass@host:port or socks5://user:pass@host:port
PROXY_URL = os.getenv("PROXY_URL", "")

# Popular leagues (used to exclude rare/minor by default). Edit to your preference.
POPULAR_LEAGUES = set([
    "Premier League",
    "LaLiga",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
    "Eredivisie",
    "Primeira Liga",
    "Championship",
    "Scottish Premiership",
    "UEFA Champions League",
    "UEFA Europa League",
    "UEFA Conference League",
    "MLS",
    "Brasileirao",
    "Argentine Primera",
    "EFL Cup",
])
TIMEZONE = os.getenv("TIMEZONE", "Europe/Moscow")
WEB_BASE_URL = os.getenv("WEB_BASE_URL", "http://127.0.0.1:8000")
API_TOKEN = os.getenv("API_TOKEN", "")
PUBLISH_CRON = os.getenv("PUBLISH_CRON", "0 9 * * *")
PUBLISH_CHAT_ID = os.getenv("PUBLISH_CHAT_ID", "")
