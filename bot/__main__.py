import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage, SimpleEventIsolation
from aiogram.utils.token import TokenValidationError

from bot.config import load_bot_token
from bot.handlers import router
from bot.services.receipt_recognition import log_ocr_runtime


async def main() -> None:
    try:
        token = load_bot_token()
    except ValueError as error:
        raise SystemExit(str(error)) from None

    await asyncio.to_thread(log_ocr_runtime)

    try:
        bot = Bot(token=token)
    except TokenValidationError:
        raise SystemExit("Некорректный формат BOT_TOKEN. Проверьте токен от @BotFather.") from None

    dispatcher = Dispatcher(storage=MemoryStorage(), events_isolation=SimpleEventIsolation())
    dispatcher.include_router(router)
    # Long polling получает новые сообщения; aiogram закрывает сессию при остановке.
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
