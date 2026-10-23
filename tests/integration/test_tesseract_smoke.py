"""Отдельная проверка реального локального OCR; сеть и Telegram не нужны."""

import os
import shutil
import unittest
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytesseract
from PIL import Image, ImageDraw, ImageFont
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage

from bot.handlers import ReceiptInput, recognize_last_photo
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
        # Подменяется только Telegram: handler -> facade -> Pillow -> Tesseract
        # -> существующий parser -> состояние проверки работают по-настоящему.
        async def download(file_id, *, destination, timeout):
            destination.write(image_data)
            return destination

        message = SimpleNamespace(
            bot=SimpleNamespace(download=AsyncMock(side_effect=download)),
            answer=AsyncMock(),
        )
        storage = MemoryStorage()
        state = FSMContext(storage=storage, key=StorageKey(bot_id=1, chat_id=1, user_id=1))
        try:
            await state.set_data({"receipt_photo": {"file_id": "synthetic-smoke-image"}})
            await recognize_last_photo(message, state)
            data = await state.get_data()
            self.assertEqual(await state.get_state(), ReceiptInput.review.state)
            result = RecognizedReceipt.model_validate(data["recognition"])
        finally:
            await storage.close()
        self.assertEqual([item.name for item in result.items], ["Пицца", "Coffee"])
        self.assertEqual(result.items_total, Decimal("2000.00"))
        self.assertFalse(result.totals_mismatch)
        self.assertIn("ИТОГО", result.raw_text)
        self.assertIn("Итого по найденным позициям: 2000", message.answer.await_args.args[0])
        self.assertIn("Проверьте позиции", message.answer.await_args.args[0])
