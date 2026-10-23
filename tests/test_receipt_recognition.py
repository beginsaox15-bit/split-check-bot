import os
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from bot.services.receipt_models import RecognizedReceipt, RecognitionUnavailable
from bot.services.receipt_parser import parse_receipt_lines
from bot.services.receipt_recognition import recognize_receipt


def receipt_payload() -> dict:
    return {
        "readable": True,
        "items": [
            {"name": "Бургер", "quantity": "2", "unit_price": "850.00", "total_price": "1700.00"},
            {"name": "Коктейль", "quantity": "1", "unit_price": None, "total_price": "650.00"},
        ],
        "receipt_total": "2350.00", "currency": "RUB",
    }


class ReceiptModelTests(unittest.TestCase):
    def test_exact_decimal_sum_and_fractional_quantity(self) -> None:
        data = receipt_payload()
        data["items"][0].update(quantity="0.5", unit_price="0.20", total_price="0.10")
        data["items"][1].update(total_price="0.20")
        data["receipt_total"] = "0.30"
        result = RecognizedReceipt.model_validate(data)
        self.assertEqual(result.items_total, Decimal("0.30"))
        self.assertFalse(result.totals_mismatch)

    def test_total_mismatch(self) -> None:
        data = receipt_payload()
        data["receipt_total"] = "2350.01"
        self.assertTrue(RecognizedReceipt.model_validate(data).totals_mismatch)

    def test_money_is_string_and_invalid_values_are_rejected(self) -> None:
        for field, value in (("total_price", 0.1), ("total_price", "-1"),
                             ("total_price", "NaN"), ("quantity", "0"), ("name", " ")):
            with self.subTest(field=field, value=value):
                data = receipt_payload()
                data["items"][0][field] = value
                with self.assertRaises(ValidationError):
                    RecognizedReceipt.model_validate(data)


class BackendSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_backend_without_api_key(self) -> None:
        result = parse_receipt_lines(["Пицца 1800"])
        with patch.dict(os.environ, {}, clear=True):
            with patch("bot.services.receipt_recognition.recognize_local", new_callable=AsyncMock) as local:
                local.return_value = result
                self.assertEqual(await recognize_receipt(b"image"), result)
                local.assert_awaited_once_with(b"image")

    async def test_explicit_and_blank_backend_use_tesseract(self) -> None:
        for backend in ("tesseract", " TESSERACT ", ""):
            with self.subTest(backend=backend):
                with patch.dict(os.environ, {"RECEIPT_OCR_BACKEND": backend}):
                    with patch("bot.services.receipt_recognition.recognize_local", new_callable=AsyncMock) as local:
                        await recognize_receipt(b"image")
                        local.assert_awaited_once_with(b"image")

    async def test_local_failure_does_not_fall_back_to_external_api(self) -> None:
        with patch.dict(os.environ, {"RECEIPT_OCR_BACKEND": "tesseract", "OPENAI_API_KEY": "unused"}):
            with patch("bot.services.receipt_recognition.recognize_local", new_callable=AsyncMock) as local:
                local.side_effect = RecognitionUnavailable("Missing executable")
                with self.assertRaises(RecognitionUnavailable):
                    await recognize_receipt(b"image")
                self.assertEqual(local.await_count, 1)

    async def test_removed_or_unknown_backend_is_not_silently_replaced(self) -> None:
        for backend in ("paddleocr", "openai", "unknown"):
            with self.subTest(backend=backend):
                with patch.dict(os.environ, {"RECEIPT_OCR_BACKEND": backend}):
                    with patch("bot.services.receipt_recognition.recognize_local", new_callable=AsyncMock) as local:
                        with self.assertRaises(RecognitionUnavailable):
                            await recognize_receipt(b"image")
                        local.assert_not_awaited()
