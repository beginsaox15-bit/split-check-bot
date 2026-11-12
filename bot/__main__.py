import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage, SimpleEventIsolation
from aiogram.utils.token import TokenValidationError

from bot.config import load_bot_token, receipt_db_path
from bot.receipt_store import ReceiptStore, StoreError
from bot.receipt_persistence import install_persistence
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

    store = ReceiptStore(receipt_db_path())
    try:
        await store.initialize()
    except StoreError:
        # Continue serving /start; operations fail safely until storage recovers.
        logging.getLogger(__name__).warning("Receipt storage unavailable at startup")
    install_persistence(router, store)
    dispatcher = Dispatcher(storage=MemoryStorage(), events_isolation=SimpleEventIsolation())
    dispatcher.include_router(router)
    # Long polling получает новые сообщения; aiogram закрывает сессию при остановке.
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
