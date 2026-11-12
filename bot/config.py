import os
from pathlib import Path

from dotenv import load_dotenv


ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_bot_token() -> str:
    # Уже заданные переменные окружения имеют приоритет над файлом .env.
    load_dotenv(ENV_FILE, override=False)
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise ValueError("Укажите BOT_TOKEN в файле .env или в переменных окружения.")
    return token


def receipt_db_path() -> Path:
    """Stable path across deployments; persistent volume recommended in containers."""
    load_dotenv(ENV_FILE, override=False)
    configured = os.getenv("RECEIPT_DB_PATH", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".local/share/split-check-bot/receipts.sqlite3"
