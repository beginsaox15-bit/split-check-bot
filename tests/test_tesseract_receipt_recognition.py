import asyncio
import os
import threading
import unittest
from decimal import Decimal
from io import BytesIO
from unittest.mock import patch

import pytesseract
from PIL import Image, ImageDraw
import cv2
import numpy as np

from bot.services.receipt_preprocessing import rectify_receipt, deskew, adaptive_image

from bot.services import tesseract_receipt_recognition as local
from bot.services.receipt_models import (
    LowQualityReceipt, RecognitionUnavailable, TesseractLanguagesUnavailable, TesseractNotInstalled,
    UnparsedReceipt, UnreadableReceipt,
)
from bot.services.receipt_parser import parse_receipt_lines
from bot.services.receipt_recognition import recognize_receipt


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
    def test_perspective_crops_dark_background(self) -> None:
        source = np.full((800, 600), 40, dtype=np.uint8)
        cv2.fillConvexPoly(source, np.int32([[130, 70], [470, 120], [520, 730], [80, 690]]), 235)
        rectified, applied = rectify_receipt(source)
        self.assertTrue(applied)
        self.assertLess(rectified.size, source.size)
        self.assertGreater(float(np.mean(rectified)), 210)

    def test_uncertain_boundary_keeps_entire_image(self) -> None:
        source = np.full((300, 400), 230, dtype=np.uint8)
        result, applied = rectify_receipt(source)
        self.assertFalse(applied)
        np.testing.assert_array_equal(result, source)

    def test_skew_is_corrected_without_clipping_text(self) -> None:
        source = np.full((400, 700), 255, dtype=np.uint8)
        for y in range(70, 340, 40):
            cv2.line(source, (80, y), (610, y), 0, 2)
        tilted = cv2.warpAffine(source, cv2.getRotationMatrix2D((350, 200), 7, 1),
                               (700, 400), borderValue=255)
        corrected, angle = deskew(tilted)
        self.assertAlmostEqual(angle, -7, delta=1)
        self.assertGreater(np.count_nonzero(corrected < 128), np.count_nonzero(source < 128) * .85)

    def test_binary_variant_preserves_thin_strokes(self) -> None:
        with Image.new("L", (200, 200), 240) as source:
            ImageDraw.Draw(source).line((100, 20, 100, 180), fill=60, width=1)
            with adaptive_image(source) as binary:
                values = np.array(binary)
                self.assertEqual(set(np.unique(values)), {0, 255})
                self.assertGreater(np.count_nonzero(values[:, 100] == 0), 150)

    def test_confident_osd_rotation_and_uncertain_orientation(self) -> None:
        with Image.new("L", (100, 200), 255) as source:
            with patch.object(local.pytesseract, "image_to_osd", return_value={"rotate": 90, "orientation_conf": 8}):
                with local.orient_image(source) as rotated:
                    self.assertEqual(rotated.size, (200, 100))
            with patch.object(local.pytesseract, "image_to_osd", return_value={"rotate": 90, "orientation_conf": 1}):
                with local.orient_image(source) as unchanged:
                    self.assertEqual(unchanged.size, source.size)

    def test_exif_rotation_grayscale_and_upscale(self) -> None:
        with local.prepare_image(image_bytes((100, 150), orientation=6)) as image:
            self.assertEqual(image.mode, "L")
            self.assertEqual(image.size, (450, 300))

    def test_large_images_are_bounded(self) -> None:
        with local.prepare_image(image_bytes((2000, 4000))) as image:
            self.assertEqual(image.size, (1800, 3600))

    def test_contrast_increases(self) -> None:
        with Image.new("RGB", (1200, 100), (110, 110, 110)) as source:
            ImageDraw.Draw(source).text((30, 30), "Thin receipt text 12345", fill=(65, 65, 65))
            with BytesIO() as buffer:
                source.save(buffer, format="PNG")
                with local.prepare_image(buffer.getvalue()) as image:
                    low, high = image.getextrema()
                    self.assertGreater(high - low, 100)

    def test_decoded_image_size_limit(self) -> None:
        with self.assertRaises(UnreadableReceipt):
            local.prepare_image(image_bytes((5001, 5000)))


def ocr_data(text, confidence=90):
    data = {key: [] for key in ("text", "conf", "page_num", "block_num", "par_num", "line_num")}
    for line_number, line in enumerate(text.splitlines(), 1):
        for word in line.split():
            for key, value in (("text", word), ("conf", confidence), ("page_num", 1),
                               ("block_num", 1), ("par_num", 1), ("line_num", line_number)):
                data[key].append(value)
    return data


class TesseractBackendTests(unittest.IsolatedAsyncioTestCase):
    def test_sparse_blocks_reassembled_without_inflating_quality(self) -> None:
        data = ocr_data("Пицца 1800\nКофе 200\nИТОГО 2000")
        data.update(top=[10, 11, 50, 51, 90, 91], height=[20] * 6,
                    left=[10, 200, 10, 200, 10, 200], block_num=list(range(6)))
        result = local.candidate_from_data(data, "adaptive", 11)
        normal = local.candidate_from_data(ocr_data("Пицца 1800\nКофе 200\nИТОГО 2000"), "grayscale", 6)
        self.assertEqual(result.lines, 3)
        self.assertEqual(result.text, normal.text)
        self.assertEqual(result.score, normal.score)

    def setUp(self) -> None:
        # Все обращения к системному Tesseract подменены в unit-тестах.
        self.enterContext(patch.object(local.pytesseract, "image_to_osd", return_value={"rotate": 0}))
        self.languages = self.enterContext(patch.object(local.pytesseract, "get_languages", return_value=["rus", "eng"]))
        self.ocr = self.enterContext(patch.object(local.pytesseract, "image_to_data", return_value=ocr_data("Пицца 1800\nКофе 200\nИТОГО 2000")))

    async def test_best_variant_and_psm_selected_by_reliable_text(self) -> None:
        weak = ocr_data("Мусор 777", confidence=20)
        good = ocr_data("Пицца 1800\nКофе 200\nИТОГО 2000")
        self.ocr.side_effect = [weak, weak, weak, weak, good, weak]
        with self.assertLogs(local.logger, level="INFO") as logs:
            result = await local.recognize_receipt(image_bytes())
        self.assertEqual(result.items_total, Decimal("2000"))
        self.assertIn("OCR selected: preprocessing=adaptive psm=4 lines=3", "\n".join(logs.output))
        self.assertNotIn("Пицца", "\n".join(logs.output))

    async def test_one_line_or_low_confidence_never_reaches_parser(self) -> None:
        for text, confidence in (("Бессмысленная строка 999", 95),
                                 ("Пицца 1800\nКофе 200\nИТОГО 2000", 15)):
            with self.subTest(confidence=confidence):
                self.ocr.return_value = ocr_data(text, confidence)
                with patch.object(local, "parse_receipt_lines") as parser:
                    with self.assertRaises(LowQualityReceipt) as caught:
                        await local.recognize_receipt(image_bytes())
                parser.assert_not_called()
                self.assertEqual(caught.exception.raw_text, text)

    async def test_one_configuration_timeout_does_not_discard_good_text(self) -> None:
        good = ocr_data("Пицца 1800\nКофе 200\nИТОГО 2000")
        self.ocr.side_effect = [RuntimeError("timeout"), good, good, good, good, good]
        result = await local.recognize_receipt(image_bytes())
        self.assertEqual(result.items_total, Decimal("2000"))

    async def test_mocked_ocr_preserves_parser_result_and_raw_text(self) -> None:
        raw_text = "1. Бургер 2 × 850,00 = 1700,00\nКоктейль 650.00\nИТОГО 2350,00"
        self.ocr.return_value = ocr_data(raw_text)
        with patch.dict(os.environ, {}, clear=True):
            result = await local.recognize_receipt(image_bytes())
        self.assertEqual(result.items_total, Decimal("2350"))
        self.assertEqual(result.items[0].quantity, "2")
        self.assertEqual(result.raw_text, raw_text)
        self.assertFalse(result.totals_mismatch)
        self.assertEqual(self.ocr.call_count, 6)
        self.assertEqual({call.kwargs["config"] for call in self.ocr.call_args_list},
                         {"--psm 6", "--psm 4", "--psm 11"})
        self.assertTrue(all(call.kwargs["lang"] == "rus+eng" for call in self.ocr.call_args_list))

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
        self.ocr.return_value = ocr_data(" \n\t")
        with self.assertRaises(UnreadableReceipt):
            await local.recognize_receipt(image_bytes())

    async def test_unparsed_text_is_available_for_manual_input(self) -> None:
        raw_text = "КАССОВЫЙ ЧЕК\nнеобычная вёрстка текста\nИТОГО 100"
        self.ocr.return_value = ocr_data(raw_text)
        with self.assertRaises(UnparsedReceipt) as caught:
            await local.recognize_receipt(image_bytes())
        self.assertEqual(caught.exception.raw_text, raw_text)
        self.assertNotIn(raw_text, str(caught.exception))

    async def test_ocr_error_and_timeout_keep_traceback_in_developer_logs(self) -> None:
        for error in (pytesseract.TesseractError(1, "sensitive receipt data"),
                      RuntimeError("sensitive receipt data")):
            with self.subTest(error=type(error).__name__):
                self.ocr.side_effect = error
                with self.assertLogs("bot.services.receipt_recognition", level="ERROR") as logs:
                    with self.assertRaises(RecognitionUnavailable) as caught:
                        await recognize_receipt(image_bytes())
                self.assertNotIn("sensitive receipt data", str(caught.exception))
                self.assertIs(caught.exception.__cause__, error)
                self.assertIn("Traceback (most recent call last)", " ".join(logs.output))
                self.assertIn("sensitive receipt data", " ".join(logs.output))
                self.assertIn("TESSDATA_PREFIX=", " ".join(logs.output))

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
