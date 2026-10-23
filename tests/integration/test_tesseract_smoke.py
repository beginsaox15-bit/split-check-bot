"""Отдельная проверка реального локального OCR; сеть и Telegram не нужны."""

import os
import shutil
import unittest
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytesseract
from PIL import Image, ImageDraw, ImageFont
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Update

from bot.handlers import ReceiptInput, router
from bot.services.receipt_models import RecognizedReceipt


class TesseractSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_russian_and_english_receipt(self) -> None:
        if not shutil.which("tesseract"):
            self.skipTest("System Tesseract is not installed")
        if not {"rus", "eng"}.issubset(pytesseract.get_languages(config="")):
            self.skipTest("Tesseract rus+eng language data is not installed")
        font_path = Path(os.getenv("TESSERACT_SMOKE_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
        if not font_path.is_file():
            self.skipTest("Set TESSERACT_SMOKE_FONT to a font with Cyrillic support")
        font = ImageFont.truetype(str(font_path), 40)
        with Image.new("RGB", (1200, 360), "white") as image:
            ImageDraw.Draw(image).multiline_text(
                (40, 40), "Пицца 1800.00\nCoffee 200.00\nИТОГО 2000.00",
                font=font, fill="black", spacing=24,
            )
            with BytesIO() as buffer:
                image.save(buffer, format="PNG")
                image_data = buffer.getvalue()
        # Настоящий photo Update проходит через dispatcher, фильтры и handler.
        # Подменяется только Telegram; OCR и parser работают по-настоящему.
        async def download(file_id, *, destination, timeout):
            destination.write(image_data)
            return destination

        storage = MemoryStorage()
        dispatcher = Dispatcher(storage=storage)
        dispatcher.include_router(router)
        async with Bot(token="123456:TEST_TOKEN") as bot:
            state = FSMContext(storage=storage, key=StorageKey(bot_id=bot.id, chat_id=1, user_id=1))
            update = Update.model_validate({"update_id": 1, "message": {
                "message_id": 1, "date": 1,
                "chat": {"id": 1, "type": "private"},
                "from": {"id": 1, "is_bot": False, "first_name": "Smoke"},
                "photo": [
                    {"file_id": "small-image", "file_unique_id": "small", "width": 90, "height": 27},
                    {"file_id": "synthetic-smoke-image", "file_unique_id": "large",
                     "width": 1200, "height": 360, "file_size": len(image_data)},
                ],
            }})
            try:
                await state.set_state(ReceiptInput.photo)
                with patch.dict(os.environ, {"OPENAI_API_KEY": "", "RECEIPT_OCR_BACKEND": ""}), \
                        patch.object(bot, "download", new_callable=AsyncMock, side_effect=download) as fetch, \
                        patch.object(bot.session, "make_request", new_callable=AsyncMock) as request:
                    await dispatcher.feed_update(bot, update)
                fetch.assert_awaited_once()
                self.assertEqual(fetch.await_args.args[0], "synthetic-smoke-image")
                data = await state.get_data()
                self.assertEqual(await state.get_state(), ReceiptInput.review.state)
                result = RecognizedReceipt.model_validate(data["recognition"])
                replies = [call.args[1].text for call in request.await_args_list
                           if isinstance(call.args[1], SendMessage)]
            finally:
                await storage.close()
        self.assertEqual([item.name for item in result.items], ["Пицца", "Coffee"])
        self.assertEqual(result.currency, "RUB")
        self.assertEqual(result.items_total, Decimal("2000.00"))
        self.assertFalse(result.totals_mismatch)
        self.assertIn("ИТОГО", result.raw_text)
        self.assertIn("Итого по найденным позициям: 2000 ₽", replies[-1])
        self.assertIn("Проверьте позиции", replies[-1])
