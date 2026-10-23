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
