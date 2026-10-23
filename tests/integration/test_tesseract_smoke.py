"""Отдельная проверка реального локального OCR; сеть и Telegram не нужны."""

import os
import shutil
import unittest
from decimal import Decimal
from io import BytesIO
from pathlib import Path

import pytesseract
from PIL import Image, ImageDraw, ImageFont

from bot.services.tesseract_receipt_recognition import recognize_receipt


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
                result = await recognize_receipt(buffer.getvalue())
        self.assertEqual([item.name for item in result.items], ["Пицца", "Coffee"])
        self.assertEqual(result.items_total, Decimal("2000.00"))
        self.assertFalse(result.totals_mismatch)
        self.assertIn("ИТОГО", result.raw_text)
