"""Only confident source-pixel evidence can replace a damaged accepted item."""

import unittest
from decimal import Decimal
from io import BytesIO
from unittest.mock import patch

from PIL import Image

from bot.services import receipt_line_review as review
from bot.services.receipt_models import ReceiptItem
from bot.services.receipt_parser import parse_receipt_lines
from bot.services.receipt_ocr_layout import OCRLine
from bot.services.tesseract_receipt_recognition import make_candidate, candidate_from_data
from tests.test_tesseract_receipt_recognition import ocr_data


def line(text, y=40, conf=90, numeric=None):
    return OCRLine(text, y, y + 20, conf, 4, 4, 20, raw_confidence=conf,
                   numeric_confidence=conf if numeric is None else numeric)


def fixture(price="9350.08", conf=48):
    rows = [line("Гость 2", 10), line("Кофе 1 350.00", 40), line("Чай 1 400.00", 70),
            line(f"ИНЖИР 1 {price}", 100, conf), line("Салат 1 650.00", 130), line("Суп 1 700.00", 160),
            line("Предоплата 7000.00", 190), line("Итого к оплате 34902.50", 220)]
    candidate = make_candidate(rows, "grayscale", 6)
    candidate.geometry_size = (300, 300)
    receipt = parse_receipt_lines(candidate.text.splitlines())
    with Image.new("L", (300, 300), 255) as image, BytesIO() as out:
        image.save(out, format="PNG")
        return out.getvalue(), candidate, receipt


class SuspiciousLineTests(unittest.TestCase):
    def test_damaged_price_requires_combined_signals(self):
        image, candidate, receipt = fixture(conf=95)
        item = receipt.items[2]
        reasons = review.suspicion_reasons(item, receipt.items, [candidate.positioned_lines[3]])
        self.assertIn("price_outlier", reasons)
        self.assertIn("unusual_cents", reasons)
        self.assertTrue(review.is_suspicious(reasons))

    def test_normal_expensive_item_and_weighted_cents_remain_valid(self):
        _, candidate, receipt = fixture("9350.00", 95)
        reasons = review.suspicion_reasons(receipt.items[2], receipt.items, [candidate.positioned_lines[3]])
        self.assertEqual(reasons, ["price_outlier"])
        self.assertFalse(review.is_suspicious(reasons))
        weighted = ReceiptItem(name="Стейк", quantity="1.258", unit_price="750", total_price="943.50")
        self.assertFalse(review.is_suspicious(review.suspicion_reasons(weighted, [], [line("Стейк 1.258 750 943.50")])))

    def test_service_header_is_removed_without_removing_real_dish(self):
        for name in ("Зап: Зап Cron", "Зап: Зап Сгоп", "Стол", "Зал", "Заказ", "Зал: Зал Стол: 18"):
            self.assertTrue(review.service_name(name), name)
        for name in ("Столичный салат", "Салат", "Заказной торт"):
            self.assertFalse(review.service_name(name))
        image, candidate, receipt = fixture()
        # This corrupted header is actually accepted by the primary parser.
        rows = [line("Зап: Зап Cron 18", 0), *candidate.positioned_lines]
        candidate = make_candidate(rows, "grayscale", 6)
        receipt = parse_receipt_lines(candidate.text.splitlines())
        with patch.object(review, "reread_crop", return_value=[]):
            result = review.review_suspicious_lines(image, candidate, receipt)
        self.assertEqual(result.line_review.removed_service_items, 1)
        self.assertNotIn("Зап: Зап Cron", [item.name for item in result.items])
        self.assertEqual(result.parser_diagnostics.lines[0].category, "ignored_service")
        for decision in result.parser_diagnostics.lines:
            if decision.item_index is not None:
                self.assertLess(decision.item_index, len(result.items))

    def test_better_local_read_replaces_original_but_preserves_raw_ocr(self):
        image, candidate, receipt = fixture()
        replacement = receipt.items[2].model_copy(update={"total_price": "950"})
        with patch.object(review, "reread_crop", return_value=[(replacement, [line("ИНЖИР 1 950.00", conf=96)], 7)]) as reread:
            result = review.review_suspicious_lines(image, candidate, receipt)
        reread.assert_called_once()
        self.assertEqual(result.items[2].total_price, "950")
        self.assertTrue(result.items[2].was_reread)
        self.assertFalse(result.items[2].suspicious)
        self.assertEqual(result.line_review.successfully_reread_items, 1)
        self.assertEqual(Decimal(result.line_review.original_item_sum) - result.items_total, Decimal("8400.08"))
        self.assertIn("9350.08", candidate.text)

    def test_worse_reread_keeps_original_and_warning(self):
        image, candidate, receipt = fixture()
        replacement = receipt.items[2].model_copy(update={"total_price": "950"})
        with patch.object(review, "reread_crop", return_value=[(replacement, [line("ИНЖИР 1 950", conf=35)], 7)]):
            result = review.review_suspicious_lines(image, candidate, receipt)
        self.assertEqual(result.items[2].total_price, "9350.08")
        self.assertTrue(result.items[2].suspicious)
        self.assertEqual(result.line_review.unresolved_suspicious_items, 1)
        self.assertIn("ручной проверки", result.parse_warnings[-1])

    def test_confident_but_still_broken_or_low_numeric_read_never_replaces(self):
        _, candidate, receipt = fixture()
        original = receipt.items[2]
        reasons = review.suspicion_reasons(original, receipt.items, [candidate.positioned_lines[3]])
        for replacement, source in ((original, line("ИНЖИР 1 9350.08", conf=98)),
                                    (original.model_copy(update={"total_price": "950"}), line("ИНЖИР 1 950", conf=98, numeric=30))):
            self.assertIsNone(review.choose_reread(original, reasons, [candidate.positioned_lines[3]],
                                                  [(replacement, [source], 7)], receipt.items))

    def test_already_good_receipt_does_not_decode_or_reread_image(self):
        image, candidate, receipt = fixture("950.00", 95)
        with patch.object(review, "prepare_image") as prepare, patch.object(review, "reread_crop") as reread:
            result = review.review_suspicious_lines(image, candidate, receipt)
        prepare.assert_not_called()
        reread.assert_not_called()
        self.assertEqual(result.line_review.suspicious_items, 0)

    def test_derived_gross_is_diagnostic_only_and_requires_confident_summaries(self):
        image, candidate, receipt = fixture("950.00", 95)
        # User's observed item sum, not a reconstructed product list.
        receipt.items[0].total_price = "35893.16"
        with patch.object(review, "reread_crop", return_value=[]):
            result = review.review_suspicious_lines(image, candidate, receipt)
        self.assertEqual(result.derived_gross_total, "41902.50")
        self.assertIsNone(result.gross_total)
        self.assertIsNone(result.receipt_total)
        self.assertEqual(result.items_total, Decimal("38593.16"))
        self.assertEqual(result.missing_difference, Decimal("3309.34"))
        candidate.positioned_lines[-2].numeric_confidence = 35
        self.assertIsNone(review.derived_gross(receipt, candidate))
        receipt.prepayment_suspect = True
        self.assertIsNone(review.derived_gross(receipt, candidate))

    def test_full_and_segmented_coordinates_crop_same_source_row(self):
        image, candidate, receipt = fixture()
        original = receipt.items[2]
        for scale in (1, 2):
            candidate.coordinate_scale = scale
            source = [line("ИНЖИР 1 9350.08", 100 * scale)]
            source[0].bottom = 120 * scale
            data = ocr_data("ИНЖИР 1 950.00", 96)
            # Crop y0=86, scale4: target y100..120 => y56..136 inside crop.
            data.update(top=[60] * 3, height=[60] * 3, left=[0, 100, 160])
            with Image.new("L", (300, 300), 255) as geometry, patch.object(review.pytesseract, "image_to_data", return_value=data) as ocr:
                results = review.reread_crop(geometry, candidate, source, original, float("inf"), 2)
            self.assertEqual(len(results), 6)
            self.assertEqual(results[0][0].total_price, "950")
            self.assertEqual({call.kwargs["config"] for call in ocr.call_args_list}, {"--psm 7", "--psm 13", "--psm 6"})
            self.assertTrue(all(call.kwargs["lang"] == "rus+eng" for call in ocr.call_args_list))

    def test_neighbour_in_crop_cannot_replace_target(self):
        _, candidate, receipt = fixture()
        data = ocr_data("Кофе 1 350", 98)
        data.update(top=[0] * 3, height=[10] * 3, left=[0, 100, 160])
        with Image.new("L", (300, 300), 255) as geometry, patch.object(review.pytesseract, "image_to_data", return_value=data):
            results = review.reread_crop(geometry, candidate, [candidate.positioned_lines[3]], receipt.items[2], float("inf"), 2)
        self.assertEqual(results, [])

    def test_coordinate_mismatch_or_unavailable_crop_keeps_original(self):
        image, candidate, receipt = fixture()
        candidate.geometry_size = (123, 456)
        with patch.object(review, "reread_crop") as reread:
            result = review.review_suspicious_lines(image, candidate, receipt)
        reread.assert_not_called()
        self.assertTrue(result.items[2].suspicious)
        self.assertEqual(result.items[2].total_price, "9350.08")
        candidate.geometry_size = (300, 300)
        with patch.object(review, "reread_crop", side_effect=OSError("unavailable")):
            result = review.review_suspicious_lines(image, candidate, receipt)
        self.assertEqual(result.line_review.unresolved_suspicious_items, 1)

    def test_raw_confidence_includes_bad_numeric_token(self):
        data = ocr_data("ИНЖИР 1 9350.08", 95)
        data["conf"][-1] = 10
        row = candidate_from_data(data, "grayscale", 6).positioned_lines[0]
        self.assertEqual(row.confidence, 95)
        self.assertLess(row.raw_confidence, 70)
        self.assertEqual(row.numeric_confidence, 10)

    def test_quantity_unit_total_math_conflict_is_suspicious(self):
        item = ReceiptItem(name="Инжир", quantity="2", unit_price="950", total_price="9350.08")
        self.assertIn("arithmetic_conflict", review.suspicion_reasons(item, [], []))

    def test_low_confidence_alone_is_reviewed_without_deleting_item(self):
        item = ReceiptItem(name="Чай", quantity="1", unit_price=None, total_price="300")
        reasons = review.suspicion_reasons(item, [], [line("Чай 1 300", conf=50)])
        self.assertEqual(reasons, ["low_confidence"])
        self.assertTrue(review.is_suspicious(reasons))
