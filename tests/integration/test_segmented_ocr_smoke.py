"""Реальный Tesseract: длинный русский чек и повторяющиеся товары в overlap."""
import shutil
import os
import unittest
from pathlib import Path
from io import BytesIO
from decimal import Decimal

import pytesseract
from PIL import Image, ImageDraw, ImageFont

from bot.services.tesseract_receipt_recognition import extract_text, choose_strategy
from bot.services.receipt_parser import parse_receipt_lines


class SegmentedOCRSmokeTests(unittest.TestCase):
    def test_real_long_russian_receipt_and_duplicate_goods(self):
        if not shutil.which('tesseract') or not {'rus', 'eng'}.issubset(pytesseract.get_languages(config='')):
            self.skipTest('Tesseract rus+eng required')
        rows = ['ЧЕК (ТЕСТОВЫЙ)']
        for guest in range(1, 4):
            rows += [f'Гость {guest}'] + ['Чай 1 100.00'] * 12 + [f'Итого к оплате Гость {guest}: 1200.00']
        rows += ['Полная сумма 3600.00']
        font_path = Path(os.getenv('TESSERACT_SMOKE_FONT', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
        if not font_path.is_file():
            self.skipTest('Set TESSERACT_SMOKE_FONT to a font with Cyrillic support')
        font = ImageFont.truetype(str(font_path), 26)
        with Image.new('RGB', (600, 80 + 48 * len(rows)), 'white') as source, BytesIO() as buffer:
            draw = ImageDraw.Draw(source)
            for i, row in enumerate(rows):draw.text((20, 20 + 48 * i), row, font=font, fill='black')
            source.save(buffer, format='PNG')
            selected = extract_text(buffer.getvalue())
        self.assertEqual(len(selected.alternatives), 2)
        full, segmented = selected.alternatives
        self.assertEqual(segmented.strategy, 'segmented')
        self.assertGreater(segmented.segment_count, 1)
        for candidate in (full, segmented):
            receipt = parse_receipt_lines(candidate.text.splitlines())
            self.assertEqual(len(receipt.items), 36)
            self.assertEqual(receipt.items_total, Decimal('3600'))
            self.assertEqual([item.guest_id for item in receipt.items], ['1'] * 12 + ['2'] * 12 + ['3'] * 12)
        self.assertEqual(selected.strategy, choose_strategy(full, segmented).strategy)
