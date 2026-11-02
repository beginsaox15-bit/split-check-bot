"""Regression по обезличенному текущему OCR, без догадок о пропущенных товарах."""

import unittest
from decimal import Decimal
from pathlib import Path

from bot.services.receipt_models import RecognizedReceipt, UnreadableReceipt
from bot.services.receipt_parser import detect_currency, normalize, parse_receipt_lines
from tests.receipt_benchmark import replay_raw_ocr


ROOT = Path(__file__).parent / "fixtures/receipts"
RAW = ROOT / "ocr/long_restaurant_current_redacted.txt"


class LongRestaurantDiagnosticsTests(unittest.TestCase):
    def test_russian_receipt_and_dollar_noise_stay_rub(self):
        base = ["Гость 1", "Паста 1 850.00", "Итого к оплате Гость 1: 850.00", "Итого к оплате 850.00"]
        for noise in ("", "save / 091$", "мусор $ 123", "QR ABC$123USD", "https://example.test/$123", "КАРТА 850 USD"):
            with self.subTest(noise=noise):
                self.assertEqual(detect_currency([noise] + base), "RUB")

    def test_foreign_currency_requires_clear_evidence(self):
        self.assertEqual(detect_currency(["Гость 1", "ИТОГО К ОПЛАТЕ 850.00 USD"]), "USD")
        self.assertEqual(detect_currency(["Coffee 10.00 $", "TOTAL 10.00 $"]), "USD")
        self.assertEqual(detect_currency(["noise 91 $", "TOTAL 10.00 EUR"]), "EUR")
        self.assertEqual(detect_currency(["Гость 1", "Валюта: USD"]), "USD")

    def test_wrong_prepayment_is_quarantined_not_corrected(self):
        rows = (ROOT / "long_restaurant_guest_receipt.txt").read_text().replace("Предоплата 7000.00", "Предоплата 7009.00").splitlines()
        result = parse_receipt_lines(rows)
        self.assertIsNone(result.prepayment)
        self.assertEqual(result.prepayment_ocr, "7009")
        self.assertTrue(result.prepayment_suspect)
        self.assertFalse(result.prepayment_verified)
        self.assertTrue(result.payment_mismatch)
        self.assertEqual(result.calculated_prepayment, Decimal("7000"))
        self.assertEqual(result.comparison_total, Decimal("41902.50"))
        reloaded = RecognizedReceipt.model_validate(result.model_dump())
        self.assertIsNone(reloaded.prepayment)
        self.assertTrue(reloaded.prepayment_suspect)

    def test_supplied_partial_ocr_preserves_unknown_totals(self):
        result = parse_receipt_lines(RAW.read_text().splitlines())
        self.assertEqual(result.currency, "RUB")
        self.assertEqual(len(result.items), 22)
        self.assertEqual(result.items_total, Decimal("27476"))
        self.assertEqual(result.guest_subtotals, {"2": "29682.5"})
        self.assertIsNone(result.gross_total)
        self.assertEqual(result.prepayment, "7009")
        self.assertFalse(result.prepayment_verified)
        self.assertIsNone(result.comparison_total)
        self.assertEqual(result.amount_due, "34902.5")
        self.assertTrue(result.parse_warnings)
        self.assertFalse(any(item.name == "save /" or "|" in item.name for item in result.items))

    def test_diagnostics_partition_every_nonempty_line(self):
        result = parse_receipt_lines(RAW.read_text().splitlines())
        diag = result.parser_diagnostics
        self.assertEqual((diag.ocr_lines, diag.accepted_item_lines, diag.rejected_candidate_lines, diag.ignored_service_lines), (47, 22, 10, 15))
        self.assertEqual(diag.ocr_lines, diag.accepted_item_lines + diag.rejected_candidate_lines + diag.ignored_service_lines)
        decisions = {row.line_number: row for row in diag.lines}
        self.assertEqual(decisions[12].category, "rejected_candidate")
        self.assertEqual(decisions[12].reason, "damaged_columns")
        self.assertEqual(decisions[41].reason, "implausible_quantity")
        self.assertEqual(decisions[1].category, "ignored_service")

    def test_only_spacing_is_repaired_in_three_rows(self):
        result = parse_receipt_lines(RAW.read_text().splitlines())
        by_line = {row.line_number: row for row in result.parser_diagnostics.lines}
        for line_number, total in ((14, "2370"), (32, "900"), (34, "400")):
            item = result.items[by_line[line_number].item_index]
            self.assertEqual(item.total_price, total)
        self.assertEqual(normalize("7009.00"), "7009.00")
        self.assertEqual(normalize("7. 00.00"), "7.00.00")
        self.assertEqual(normalize("1.500"), "1.500")

    def test_multiline_item_diagnostics_and_no_items_case(self):
        result = parse_receipt_lines(["Пицца", "2 x 900.00 = 1800.00", "ИТОГО 1800"])
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.parser_diagnostics.accepted_item_lines, 2)
        with self.assertRaises(UnreadableReceipt) as caught:
            parse_receipt_lines(["Гость 1", "а б в 999.00", "ИТОГО 999"])
        self.assertEqual(caught.exception.parser_diagnostics.rejected_candidate_lines, 1)

    def test_raw_replay_does_not_claim_gold_or_run_image_ocr(self):
        from unittest.mock import patch
        with patch("pytesseract.image_to_data") as ocr:
            result = replay_raw_ocr(RAW)
        ocr.assert_not_called()
        self.assertEqual(result["mode"], "raw_ocr_replay")
        self.assertFalse(result["reference_available"])
        self.assertEqual(result["prepayment_status"], "unverified")
        self.assertEqual(result["parser_diagnostics"]["rejected_candidate_lines"], 10)
