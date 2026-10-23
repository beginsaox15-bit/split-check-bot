"""Настоящий локальный OCR длинного чека -> handler -> проверка сумм и гостей."""

import shutil
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytesseract
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage

from bot.handlers import ReceiptInput, recognize_last_photo
from bot.services.receipt_models import RecognizedReceipt


class GuestReceiptSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_guest_image_through_photo_recognition_handler(self):
        if not shutil.which("tesseract") or not {"rus", "eng"}.issubset(pytesseract.get_languages(config="")):
            self.skipTest("Tesseract rus+eng required")
        image = (Path(__file__).parents[1] / "fixtures/receipts/long_restaurant_guest_receipt.png").read_bytes()
        replies = []

        async def download(file_id, *, destination, timeout):
            destination.write(image)
            return destination

        async def answer(text, **kwargs):
            replies.append(text)

        state = FSMContext(storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=1, user_id=1))
        message = SimpleNamespace(bot=SimpleNamespace(download=download), answer=answer)
        try:
            await state.set_data({"receipt_photo": {"file_id": "synthetic-guest-receipt"}})
            await recognize_last_photo(message, state)
            self.assertEqual(await state.get_state(), ReceiptInput.review.state)
            result = RecognizedReceipt.model_validate((await state.get_data())["recognition"])
        finally:
            await state.storage.close()
        self.assertEqual(len(result.items), 14)
        self.assertEqual(result.items_total, Decimal("41902.50"))
        self.assertFalse(result.totals_mismatch)
        self.assertFalse(result.guest_subtotals_mismatch)
        self.assertFalse(result.payment_mismatch)
        self.assertIn("Полная сумма: 41902.5 ₽", replies[-1])
        self.assertIn("Предоплата: 7000 ₽", replies[-1])
        self.assertIn("Итого к оплате: 34902.5 ₽", replies[-1])
