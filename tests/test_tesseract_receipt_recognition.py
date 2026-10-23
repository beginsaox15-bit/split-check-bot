import asyncio
import os
import threading
import unittest
from decimal import Decimal
from io import BytesIO
from unittest.mock import patch

import pytesseract
from PIL import Image, ImageStat

from bot.services import tesseract_receipt_recognition as local
from bot.services.receipt_models import (
    RecognitionUnavailable, TesseractLanguagesUnavailable, TesseractNotInstalled,
    UnparsedReceipt, UnreadableReceipt,
)
from bot.services.receipt_parser import parse_receipt_lines


def image_bytes(size=(100, 150), *, orientation=None) -> bytes:
    with BytesIO() as buffer:
        with Image.new("RGB", size, "white") as image:
            if orientation is not None:
                exif = image.getexif()
                exif[274] = orientation
                image.save(buffer, format="JPEG", exif=exif)
            else:
                image.save(buffer, format="PNG")
        return buffer.getvalue()


class ImagePreparationTests(unittest.TestCase):
    def test_exif_rotation_grayscale_and_upscale(self) -> None:
        with local.prepare_image(image_bytes((100, 150), orientation=6)) as image:
            self.assertEqual(image.mode, "L")
            self.assertEqual(image.size, (300, 200))

    def test_large_images_are_bounded(self) -> None:
        with local.prepare_image(image_bytes((2000, 4000))) as image:
            self.assertEqual(image.size, (1800, 3600))

    def test_contrast_increases(self) -> None:
        with Image.new("RGB", (1200, 100), (110, 110, 110)) as source:
            source.paste((160, 160, 160), (600, 0, 1200, 100))
            with BytesIO() as buffer:
                source.save(buffer, format="PNG")
                with local.prepare_image(buffer.getvalue()) as image:
                    self.assertGreater(ImageStat.Stat(image).stddev[0], 25)

    def test_decoded_image_size_limit(self) -> None:
        with self.assertRaises(UnreadableReceipt):
            local.prepare_image(image_bytes((5001, 5000)))


class TesseractBackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # Все unit-тесты подменяют обе точки запуска системного Tesseract.
        self.languages = self.enterContext(patch.object(local.pytesseract, "get_languages", return_value=["rus", "eng"]))
        self.ocr = self.enterContext(patch.object(local.pytesseract, "image_to_string", return_value="Пицца 1800\nИТОГО 1800"))

    async def test_mocked_ocr_preserves_parser_result_and_raw_text(self) -> None:
        raw_text = "1. Бургер 2 × 850,00 = 1700,00\nКоктейль 650.00\nИТОГО 2350,00"
        self.ocr.return_value = raw_text
        with patch.dict(os.environ, {}, clear=True):
            result = await local.recognize_receipt(image_bytes())
        self.assertEqual(result.items_total, Decimal("2350"))
        self.assertEqual(result.items[0].quantity, "2")
        self.assertEqual(result.raw_text, raw_text)
        self.assertFalse(result.totals_mismatch)
        self.assertEqual(self.ocr.call_args.kwargs, {"lang": "rus+eng", "config": "--psm 6", "timeout": 30})

    async def test_missing_tesseract_has_dedicated_error(self) -> None:
        self.languages.side_effect = pytesseract.TesseractNotFoundError()
        with self.assertRaises(TesseractNotInstalled):
            await local.recognize_receipt(image_bytes())
        self.ocr.assert_not_called()

    async def test_missing_languages_prevents_partial_language_recognition(self) -> None:
        for languages in (["eng"], ["rus"], []):
            with self.subTest(languages=languages):
                self.languages.return_value = languages
                with self.assertRaises(TesseractLanguagesUnavailable):
                    await local.recognize_receipt(image_bytes())
        self.ocr.assert_not_called()

    async def test_tesseract_disappearing_after_preflight(self) -> None:
        self.ocr.side_effect = pytesseract.TesseractNotFoundError()
        with self.assertRaises(TesseractNotInstalled):
            await local.recognize_receipt(image_bytes())

    async def test_invalid_empty_and_oversized_files_do_not_run_ocr(self) -> None:
        for data in (b"not-an-image", b"", b"a" * (10 * 1024 * 1024 + 1)):
            with self.subTest(length=len(data)):
                with self.assertRaises(UnreadableReceipt):
                    await local.recognize_receipt(data)
        self.languages.assert_not_called()
        self.ocr.assert_not_called()

    async def test_empty_ocr_text_is_unreadable(self) -> None:
        self.ocr.return_value = " \n\t"
        with self.assertRaises(UnreadableReceipt):
            await local.recognize_receipt(image_bytes())

    async def test_unparsed_text_is_available_for_manual_input(self) -> None:
        raw_text = "ЧЕК\nнеобычная вёрстка\nИТОГО 100"
        self.ocr.return_value = raw_text
        with self.assertRaises(UnparsedReceipt) as caught:
            await local.recognize_receipt(image_bytes())
        self.assertEqual(caught.exception.raw_text, raw_text)
        self.assertNotIn(raw_text, str(caught.exception))

    async def test_ocr_error_and_timeout_are_sanitized(self) -> None:
        for error in (pytesseract.TesseractError(1, "sensitive receipt data"),
                      RuntimeError("sensitive receipt data")):
            with self.subTest(error=type(error).__name__):
                self.ocr.side_effect = error
                with self.assertLogs(local.__name__, level="WARNING") as logs:
                    with self.assertRaises(RecognitionUnavailable) as caught:
                        await local.recognize_receipt(image_bytes())
                self.assertNotIn("sensitive receipt data", str(caught.exception))
                self.assertNotIn("sensitive receipt data", " ".join(logs.output))

    async def test_ocr_does_not_block_event_loop(self) -> None:
        started = threading.Event()
        finished = threading.Event()
        result = parse_receipt_lines(["Пицца 1800"])

        def work(image):
            started.set()
            finished.wait(timeout=2)
            return result

        with patch.object(local, "_recognize_sync", side_effect=work):
            task = asyncio.create_task(local.recognize_receipt(b"image"))
            try:
                async with asyncio.timeout(1):
                    while not started.is_set():
                        await asyncio.sleep(0.001)
                self.assertFalse(task.done())
            finally:
                finished.set()
            self.assertEqual(await task, result)
