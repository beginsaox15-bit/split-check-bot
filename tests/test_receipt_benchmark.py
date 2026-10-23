"""Быстрые тесты метрик/формата. Tesseract здесь не запускается."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image

from tests.receipt_benchmark import (
    ExpectedItem, ExpectedReceipt, equal_total, load_cases, match_items, run_case, summarize, trace_pipeline,
)
from bot.services import tesseract_receipt_recognition as local
from bot.services.receipt_models import LowQualityReceipt


def item(name, price="850.00", quantity="1"):
    return ExpectedItem(name=name, quantity=quantity, total_price=price)


class BenchmarkMetricTests(unittest.TestCase):
    def test_failed_pipeline_does_not_claim_correct_absent_total_or_currency(self):
        case = ExpectedReceipt(name="case", image="case.png", expected_items=[item("Бургер")],
                               expected_total=None, currency=None)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "case.png"
            image.write_bytes(b"mocked-image")
            with patch("tests.receipt_benchmark.recognize_receipt", new_callable=AsyncMock,
                       side_effect=LowQualityReceipt("weak")):
                row = asyncio.run(run_case(case, image, None))
        self.assertEqual(row["status"], "low_quality")
        self.assertEqual(row["missing"], 1)
        self.assertFalse(row["total_correct"])
        self.assertFalse(row["currency_correct"])

    def test_small_name_errors_and_decimal_formatting_are_tolerated(self):
        metrics = match_items([item("Бургер"), item("Чёрный чай", "200")],
                              [item("бургер!", "850", "1.0"), item("Черный чаи", "200.00")])
        self.assertEqual((metrics["found"], metrics["missing"], metrics["false_positives"]), (2, 0, 0))

    def test_duplicates_need_distinct_detected_rows(self):
        metrics = match_items([item("Молоко", "100"), item("Молоко", "100")], [item("Молоко", "100")])
        self.assertEqual((metrics["found"], metrics["missing"], metrics["false_positives"]), (1, 1, 0))
        metrics = match_items([item("Молоко", "100")], [item("Молоко", "100"), item("Молоко", "100")])
        self.assertEqual((metrics["found"], metrics["missing"], metrics["false_positives"]), (1, 0, 1))

    def test_wrong_quantity_price_and_unrelated_names_are_not_matches(self):
        for actual in (item("Бургер", "800"), item("Бургер", quantity="2"), item("Официант")):
            with self.subTest(actual=actual):
                metrics = match_items([item("Бургер")], [actual])
                self.assertEqual((metrics["found"], metrics["missing"], metrics["false_positives"]), (0, 1, 1))

    def test_summary_weights_items_across_receipts(self):
        def row(found, expected, correct):
            return dict(found=found, expected=expected, missing=expected-found, false_positives=1,
                        total_correct=correct, currency_correct=correct, status="ok", elapsed_seconds=1)
        summary = summarize([row(5, 6, True), row(4, 4, False)])
        self.assertEqual(summary["items_accuracy_percent"], 90)
        self.assertEqual(summary["totals_correct"], 1)
        self.assertEqual(summary["false_positives"], 2)
        self.assertTrue(equal_total("850", "850.00"))
        self.assertFalse(equal_total("850", None))
        self.assertTrue(equal_total(None, None))

    def test_builtin_fixtures_are_valid_without_running_ocr(self):
        root = Path(__file__).parent / "fixtures/receipts"
        for name in ("synthetic_restaurant_01", "synthetic_restaurant_angled_dark", "synthetic_shop_discount_duplicates"):
            case = ExpectedReceipt.model_validate_json((root / "expected" / (name + ".json")).read_text())
            self.assertTrue((root / case.image).is_file())

    def test_invalid_json_duplicate_names_and_outside_paths_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "expected").mkdir()
            path = root / "expected/case.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Invalid expected"):
                load_cases(root)
            payload = {"name": "case", "image": "../outside.png", "expected_items": [item("Бургер").model_dump()],
                       "expected_total": "850", "currency": "RUB"}
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "inside fixtures"):
                load_cases(root)
            payload["image"] = "image.png"
            (root / "image.png").write_bytes(b"placeholder")
            path.write_text(json.dumps(payload))
            (root / "expected/duplicate.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_cases(root)

    def test_trace_preserves_real_calls_and_selected_input(self):
        candidate = SimpleNamespace(preprocessing="adaptive", psm=4)

        def extract(data):
            with Image.new("L", (2, 2), 120) as gray, Image.new("L", (2, 2), 255) as binary:
                local.pytesseract.image_to_data(gray, config="--psm 6")
                local.pytesseract.image_to_data(binary, config="--psm 4")
            return candidate

        with patch.object(local, "extract_text", side_effect=extract) as real_extract, \
                patch.object(local.pytesseract, "image_to_data", return_value={}) as real_data:
            with trace_pipeline(True) as trace:
                result = local.extract_text(b"test")
                self.assertIs(result, candidate)
                self.assertIs(trace["candidate"], candidate)
                self.assertEqual(trace["images"]["adaptive"].getpixel((0, 0)), 255)
            self.assertEqual(real_data.call_count, 2)
            real_extract.assert_called_once_with(b"test")
