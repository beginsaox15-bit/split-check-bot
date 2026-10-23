import asyncio
import os
import sys
import unittest
from decimal import Decimal
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from bot.services.receipt_models import RecognitionUnavailable, UnreadableReceipt
from bot.services.receipt_parser import parse_receipt_lines
from bot.services.receipt_recognition import recognize_receipt
from bot.services.paddle_receipt_recognition import prediction_rows


class ReceiptParserTests(unittest.TestCase):
    def test_russian_items_quantity_and_fiscal_lines(self) -> None:
        result = parse_receipt_lines([
            "ООО Ресторан", "ИНН 123456789012", "КАССА № 1", "ЧЕК № 1234",
            "ФН 1234567890123456", "ФД 1234", "ФПД 1234567890", "QR 123456789",
            "23.10.2026 12:34:56", "КАССИР Иванов", "ТЕРМИНАЛ 12345",
            "1. Бургер 2 × 850,00 = 1700,00", "Коктейль 650.00",
            "НДС 20% 391.67", "ИТОГО 2350,00", "БЕЗНАЛИЧНЫМИ 2350,00",
            "СДАЧА 0.00", "СПАСИБО ЗА ПОКУПКУ",
        ])
        self.assertEqual([item.name for item in result.items], ["Бургер", "Коктейль"])
        self.assertEqual(result.items[0].quantity, "2")
        self.assertEqual(result.items[0].unit_price, "850")
        self.assertEqual(result.items[0].total_price, "1700")
        self.assertEqual(result.items_total, Decimal("2350"))
        self.assertFalse(result.totals_mismatch)

    def test_english_items_and_money_separators(self) -> None:
        result = parse_receipt_lines([
            "Coffee 850.00", "Cake 650,00", "SUBTOTAL 1500", "VAT 250",
            "TOTAL 1500.00", "CARD 1500", "CHANGE 0.00",
        ])
        self.assertEqual([item.name for item in result.items], ["Coffee", "Cake"])
        self.assertEqual(result.receipt_total, "1500")

    def test_multiline_items_fractional_quantity_and_separate_total(self) -> None:
        result = parse_receipt_lines([
            "Молоко 3,2%", "2 x 100,10 200,20", "Яблоки", "0,5 x 100,00 = 50,00",
            "ИТОГО", "250,20",
        ])
        self.assertEqual(result.items[0].name, "Молоко 3,2%")
        self.assertEqual(result.items[1].quantity, "0.5")
        self.assertEqual(result.items_total, Decimal("250.20"))

    def test_columns_and_grouped_thousands(self) -> None:
        result = parse_receipt_lines([
            "Пицца 2 900.00 1800.00", "Паста 1\u202f234,50 ₽", "ИТОГО 3 034,50 ₽",
        ])
        self.assertEqual(result.items[0].quantity, "2")
        self.assertEqual(result.items[1].total_price, "1234.5")
        self.assertFalse(result.totals_mismatch)

    def test_product_name_with_digits_is_preserved(self) -> None:
        result = parse_receipt_lines(["7UP 0.5L 150.00", "ИТОГО 150.00"])
        self.assertEqual(result.items[0].name, "7UP 0.5L")

    def test_missing_total_and_exact_decimal_sum(self) -> None:
        result = parse_receipt_lines(["Вода 0.10", "Чай 0.20"])
        self.assertEqual(result.items_total, Decimal("0.30"))
        self.assertIsNone(result.receipt_total)
        self.assertFalse(result.totals_mismatch)

    def test_discount_is_not_item_and_mismatch_remains_visible(self) -> None:
        result = parse_receipt_lines(["Пицца 1000", "СКИДКА 100", "ИТОГО 900"])
        self.assertEqual(len(result.items), 1)
        self.assertTrue(result.totals_mismatch)

    def test_empty_noise_and_service_only_are_unreadable(self) -> None:
        for rows in ([], ["нечитаемый текст"], ["ИНН 1234567890", "ИТОГО 100", "QR 12345"],
                     ["1234567890", "23.10.2026 12:00"], ["Пицца -100"]):
            with self.subTest(rows=rows):
                with self.assertRaises(UnreadableReceipt):
                    parse_receipt_lines(rows)

    def test_prediction_geometry_reassembles_columns_in_reading_order(self) -> None:
        rows = prediction_rows({
            "rec_texts": ["850,00", "Бургер", "ИТОГО", "850,00", "noise"],
            "rec_scores": [0.99, 0.99, 0.99, 0.99, 0.1],
            "rec_boxes": [[300, 10, 360, 30], [10, 12, 100, 32], [10, 50, 80, 70],
                          [300, 50, 360, 70], [0, 0, 10, 10]],
        })
        self.assertEqual(rows, ["Бургер 850,00", "ИТОГО 850,00"])
        self.assertEqual(parse_receipt_lines(rows).items_total, Decimal("850"))

    def test_polygons_and_missing_scores_are_supported(self) -> None:
        rows = prediction_rows({
            "rec_texts": ["Чай", "100.00"],
            "rec_polys": [[[0, 0], [50, 0], [50, 20], [0, 20]],
                          [[100, 0], [160, 0], [160, 20], [100, 20]]],
        })
        self.assertEqual(rows, ["Чай 100.00"])


class PaddleBackendTests(unittest.IsolatedAsyncioTestCase):
    def test_engine_uses_local_model_directories_and_cpu(self) -> None:
        from bot.services import paddle_receipt_recognition as local
        constructor = Mock()
        with patch.object(local, "_engine", None):
            with patch.dict(sys.modules, {"paddleocr": SimpleNamespace(PaddleOCR=constructor)}):
                with patch.dict(os.environ, {"PADDLE_DET_MODEL_DIR": "/local/det",
                                             "PADDLE_REC_MODEL_DIR": "/local/rec"}):
                    self.assertIs(local._get_engine(), constructor.return_value)
                    self.assertIs(local._get_engine(), constructor.return_value)
        constructor.assert_called_once()
        options = constructor.call_args.kwargs
        self.assertEqual(options["text_detection_model_dir"], "/local/det")
        self.assertEqual(options["text_recognition_model_dir"], "/local/rec")
        self.assertEqual(options["text_recognition_model_name"], "cyrillic_PP-OCRv5_mobile_rec")
        self.assertEqual(options["device"], "cpu")

    async def test_default_backend_does_not_need_key_or_cloud(self) -> None:
        result = parse_receipt_lines(["Пицца 1800"])
        with patch.dict(os.environ, {}, clear=True):
            with patch("bot.services.paddle_receipt_recognition.recognize_receipt", new_callable=AsyncMock) as local:
                local.return_value = result
                self.assertEqual(await recognize_receipt(b"image"), result)
                local.assert_awaited_once_with(b"image")

    async def test_local_failure_does_not_fall_back_to_paid_api(self) -> None:
        with patch.dict(os.environ, {"RECEIPT_OCR_BACKEND": "paddleocr", "OPENAI_API_KEY": "unused"}):
            with patch("bot.services.paddle_receipt_recognition.recognize_receipt", new_callable=AsyncMock) as local:
                local.side_effect = RecognitionUnavailable("No model weights")
                with self.assertRaises(RecognitionUnavailable):
                    await recognize_receipt(b"image")
                self.assertEqual(local.await_count, 1)

    async def test_unknown_backend_is_not_silently_replaced(self) -> None:
        with patch.dict(os.environ, {"RECEIPT_OCR_BACKEND": "unknown"}):
            with self.assertRaises(RecognitionUnavailable):
                await recognize_receipt(b"image")

    async def test_real_image_decoder_and_mocked_ocr_engine(self) -> None:
        from PIL import Image
        from bot.services.paddle_receipt_recognition import recognize_receipt as recognize_local
        image = BytesIO()
        Image.new("RGB", (100, 150), "white").save(image, format="PNG")
        engine = Mock()
        engine.predict.return_value = [{"rec_texts": ["Пицца 1800", "ИТОГО 1800"]}]
        with patch("bot.services.paddle_receipt_recognition._get_engine", return_value=engine):
            result = await recognize_local(image.getvalue())
        self.assertEqual(result.items_total, Decimal("1800"))
        pixels = engine.predict.call_args.args[0]
        self.assertEqual(pixels.shape, (150, 100, 3))
        self.assertEqual(str(pixels.dtype), "uint8")

    async def test_invalid_image_is_not_sent_to_ocr(self) -> None:
        from bot.services.paddle_receipt_recognition import recognize_receipt as recognize_local
        with patch("bot.services.paddle_receipt_recognition._get_engine") as engine:
            with self.assertRaises(UnreadableReceipt):
                await recognize_local(b"not-an-image")
            engine.assert_not_called()

    async def test_ocr_does_not_block_event_loop(self) -> None:
        import threading
        from bot.services.paddle_receipt_recognition import recognize_receipt as recognize_local
        finished = threading.Event()
        result = parse_receipt_lines(["Пицца 1800"])

        def work(image):
            finished.wait(timeout=2)
            return result

        with patch("bot.services.paddle_receipt_recognition._recognize_sync", side_effect=work):
            task = asyncio.create_task(recognize_local(b"image"))
            try:
                await asyncio.sleep(0.02)
                self.assertFalse(task.done())
            finally:
                finished.set()
            self.assertEqual(await task, result)
