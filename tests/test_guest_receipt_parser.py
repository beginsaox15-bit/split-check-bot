import unittest
from decimal import Decimal
from pathlib import Path

from bot.services.receipt_models import UnreadableReceipt
from bot.services.receipt_parser import parse_receipt_lines


ROOT = Path(__file__).parent / "fixtures/receipts"


class GuestReceiptTests(unittest.TestCase):
    def full_receipt(self):
        return (ROOT / "long_restaurant_guest_receipt.txt").read_text().splitlines()

    def test_three_guests_and_all_payment_totals(self):
        result = parse_receipt_lines(self.full_receipt())
        self.assertEqual(len(result.items), 14)
        self.assertEqual(result.items_total, Decimal("41902.50"))
        self.assertEqual(Decimal(result.gross_total), Decimal("41902.50"))
        self.assertEqual(Decimal(result.prepayment), Decimal("7000.00"))
        self.assertEqual(Decimal(result.amount_due), Decimal("34902.50"))
        self.assertEqual(result.guest_subtotals_total, Decimal("41902.50"))
        self.assertEqual([item.guest_id for item in result.items], ["1"] * 3 + ["2"] * 8 + ["3"] * 3)
        self.assertFalse(result.totals_mismatch)
        self.assertFalse(result.guest_subtotals_mismatch)
        self.assertFalse(result.payment_mismatch)

    def test_services_and_split_subtotals_do_not_become_items(self):
        result = parse_receipt_lines([
            "Гость 1", "Стол 18", "Заказ 10", "Официант", "Оеициант: СКРЫТО",
            "Cron: 18", "Открыт: 01.01.2026 12:00", "Наименование Кол-во Сумма",
            "Чай 1 330.00", "Итого к оплате Гость 1", "330.00", "Полная сумма", "330.00",
            "Предоплата", "100.00", "ИТОГО К ОПЛАТЕ", "230.00",
        ])
        self.assertEqual([item.name for item in result.items], ["Чай"])
        self.assertEqual(result.guest_subtotals, {"1": "330"})
        self.assertEqual(result.receipt_total, "330")

    def test_guest_subtotal_math_and_payment_math_are_independent(self):
        rows = [line.replace("29082.50", "29682.50") for line in self.full_receipt()]
        result = parse_receipt_lines(rows)
        self.assertTrue(result.guest_subtotals_mismatch)
        self.assertFalse(result.totals_mismatch)
        self.assertFalse(result.payment_mismatch)
        rows = [line.replace("34902.50", "34900.00") for line in self.full_receipt()]
        result = parse_receipt_lines(rows)
        self.assertTrue(result.payment_mismatch)
        self.assertFalse(result.totals_mismatch)

    def test_missing_item_compared_against_gross_not_amount_due(self):
        rows = [line for line in self.full_receipt() if not line.startswith("Бургер ")]
        result = parse_receipt_lines(rows)
        self.assertTrue(result.totals_mismatch)
        self.assertEqual(result.comparison_total, Decimal("41902.50"))

    def test_three_columns_and_thousand_separator(self):
        result = parse_receipt_lines(["Гость 1", "Сырная тарелка 2 1 960.00",
                                      "Итого к оплате Гость 1: 1960.00", "Полная сумма 1960.00"])
        self.assertEqual(result.items[0].name, "Сырная тарелка")
        self.assertEqual(result.items[0].quantity, "2")
        self.assertEqual(result.items[0].total_price, "1960")
        self.assertIsNone(result.items[0].unit_price)

    def test_ocr_label_fragments_and_subtotals_without_header(self):
        result = parse_receipt_lines(["Чай 1 330.00", "того к оплате Гость 1: 330.00",
                                      "ость 2", "Суп 1 350.00", "Игого к оплате Гость 2; 350.00",
                                      "Полная сумма 680.00"])
        self.assertEqual([item.guest_id for item in result.items], ["1", "2"])
        self.assertFalse(result.guest_subtotals_mismatch)

    def test_damaged_financial_values_are_not_guessed(self):
        result = parse_receipt_lines(["Гость 1", "Чай 1 330.00", "Полная сумма; ait | 902 SO",
                                      "Предоплата: 7. 00.00", "ИТОГО К ОПЛАТЕ: 34 902,50"])
        self.assertIsNone(result.gross_total)
        self.assertIsNone(result.prepayment)
        self.assertIsNone(result.comparison_total)
        self.assertEqual(result.amount_due, "34902.5")
        self.assertTrue(result.parse_warnings)

    def test_garbage_is_skipped_without_renaming_products(self):
        result = parse_receipt_lines(["Гость 1", "а б в 999.00", ".. / -- 888.00",
                                      "Кoфe 777.00", "Шапка Наименование 999.00", "%%% мусор %%% 900.00",
                                      "Чай 1 330.00", "Полная сумма 330.00"])
        self.assertEqual([item.name for item in result.items], ["Чай"])
        self.assertGreaterEqual(result.skipped_lines, 5)

    def test_repeated_subtotal_not_double_counted_and_conflicts_warn(self):
        result = parse_receipt_lines(["Гость 1", "Чай 1 330.00", "Итого к оплате Гость 1: 330.00",
                                      "Итого к оплате Гость 1: 330.00", "Полная сумма 330.00"])
        self.assertEqual(result.guest_subtotals_total, Decimal("330"))
        result = parse_receipt_lines(["Гость 1", "Чай 1 330.00", "Итого к оплате Гость 1: 330.00",
                                      "Итого к оплате Гость 1: 999.00", "Полная сумма 330.00"])
        self.assertTrue(result.parse_warnings)

    def test_prepayment_plus_due_can_be_used_when_gross_not_printed(self):
        result = parse_receipt_lines(["Гость 1", "Чай 1 330.00", "Предоплата 100", "К оплате 230"])
        self.assertEqual(result.comparison_total, Decimal("330"))
        self.assertFalse(result.totals_mismatch)

    def test_service_only_receipt_has_no_goods(self):
        with self.assertRaises(UnreadableReceipt):
            parse_receipt_lines(["Гость 1", "Стол 18", "Заказ 123", "Полная сумма 100", "Итого к оплате 100"])
