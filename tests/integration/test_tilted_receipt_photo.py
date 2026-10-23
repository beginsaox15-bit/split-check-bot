"""Opt-in: настоящая фотография Whole Foods из публичного doc_scanner sample."""

import os
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytesseract
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage

from bot.handlers import recognize_last_photo


class TiltedReceiptPhotoTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_tilted_photo_through_handler(self) -> None:
        filename = os.getenv("RECEIPT_SMOKE_PHOTO")
        if not filename:
            self.skipTest("Set RECEIPT_SMOKE_PHOTO to the documented Whole Foods photograph")
        if not shutil.which("tesseract") or not {"rus", "eng"}.issubset(pytesseract.get_languages(config="")):
            self.skipTest("Tesseract rus+eng required")
        photo = Path(filename).read_bytes()
        replies = []

        async def download(file_id, *, destination, timeout):
            destination.write(photo)
            return destination

        async def answer(text, **kwargs):
            replies.append(text)

        message = SimpleNamespace(bot=SimpleNamespace(download=download), answer=answer)
        storage = MemoryStorage()
        state = FSMContext(storage=storage, key=StorageKey(bot_id=1, chat_id=1, user_id=1))
        try:
            await state.set_data({"receipt_photo": {"file_id": "public-tilted-photograph"}})
            with self.assertLogs("bot.services.tesseract_receipt_recognition", level="INFO") as logs:
                await recognize_last_photo(message, state)
            data = await state.get_data()
            text = data.get("recognition", {}).get("raw_text") or data.get("raw_text", "")
        finally:
            await storage.close()
        self.assertGreaterEqual(len(text.splitlines()), 10)
        self.assertGreaterEqual(text.upper().count("BACON"), 4)
        self.assertIn("FLOUR ALMOND", text.upper())
        self.assertIn("101.33", text)
        self.assertIn("crop_perspective=True", "\n".join(logs.output))
        self.assertIn("OCR selected:", "\n".join(logs.output))
        self.assertNotIn("psm=", "\n".join(replies))
