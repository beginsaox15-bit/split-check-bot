"""Real Tesseract crop through photo handler, with a controlled primary OCR error.

The damaged first-pass text is injected; the crop and second-pass pixels are real.
This is not a claimed improvement on the user's unavailable original photograph.
"""

import os
import shutil
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import pytesseract
from PIL import Image, ImageDraw, ImageFont
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage

from bot.handlers import recognize_last_photo, ReceiptInput
from bot.services.receipt_models import RecognizedReceipt
from bot.services.receipt_ocr_layout import OCRLine
from bot.services.tesseract_receipt_recognition import make_candidate


class TargetedLineOCRIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"RECEIPT_OCR_BACKEND": "tesseract"}))
        self.enterContext(patch("bot.services.openrouter_receipt_recognition._post",
                                side_effect=AssertionError("Local smoke must not call OpenRouter")))

    async def test_real_crop_fixes_injected_damaged_price_through_photo_handler(self):
        if not shutil.which("tesseract") or not {"rus", "eng"}.issubset(pytesseract.get_languages(config="")):
            self.skipTest("Tesseract rus+eng required")
        text = ["ЧЕК ТЕСТОВЫЙ", "Зап: Зап Cron 18", "Гость 2", "Кофе 1 350.00", "Чай 1 400.00",
                "ИНЖИР 1 950.00", "Салат 1 650.00", "Суп 1 700.00", "Предоплата 7000.00", "Итого к оплате 34902.50"]
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        rows = []
        with Image.new("L", (600, 1700), 255) as source, BytesIO() as buffer:
            draw = ImageDraw.Draw(source)
            for i, value in enumerate(text):
                y = 30 + i * 70
                draw.text((20, y), value, font=font, fill=0)
                _, t, _, b = draw.textbbox((20, y), value, font=font)
                observed = value.replace("950.00", "9350.08") if i == 5 else value
                conf = 40 if i == 5 else 95
                rows.append(OCRLine(observed, t, b, conf, 4, 4, 20, raw_confidence=conf, numeric_confidence=conf))
            source.save(buffer, format="PNG")
            image = buffer.getvalue()
        candidate = make_candidate(rows, "grayscale", 6)
        candidate.geometry_size = (600, 1700)
        candidate.strategy = "segmented"
        replies = []

        async def download(file_id, *, destination, timeout):
            destination.write(image)

        async def answer(message, **kwargs):
            replies.append((message, kwargs))

        state = FSMContext(storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=1, user_id=1))
        try:
            await state.set_data({"receipt_photo": {"file_id": "safe-targeted-regression"}})
            with patch("bot.services.tesseract_receipt_recognition.extract_text", return_value=candidate):
                await recognize_last_photo(SimpleNamespace(bot=SimpleNamespace(download=download), answer=answer), state)
            self.assertEqual(await state.get_state(), ReceiptInput.review.state)
            result = RecognizedReceipt.model_validate((await state.get_data())["recognition"])
        finally:
            await state.storage.close()
        fig = next(item for item in result.items if item.name.casefold() == "инжир")
        self.assertEqual(fig.total_price, "950")
        self.assertTrue(fig.was_reread)
        self.assertFalse(fig.suspicious)
        self.assertEqual(result.line_review.successfully_reread_items, 1)
        self.assertEqual(result.line_review.removed_service_items, 1)
        self.assertEqual(result.derived_gross_total, "41902.50")
        self.assertIn("9350.08", result.raw_text)
        self.assertIn("950 ₽", "\n".join(reply[0] for reply in replies))
        self.assertTrue(any("Показать распознанный текст" in str(reply[1]) for reply in replies))
        self.assertNotIn("PSM", "\n".join(reply[0] for reply in replies))
