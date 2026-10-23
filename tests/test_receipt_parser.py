import unittest
from decimal import Decimal

from bot.services.receipt_models import UnreadableReceipt
from bot.services.receipt_parser import parse_receipt_lines


class ReceiptParserTests(unittest.TestCase):
    def test_explicit_ruble_markers_override_dollar_noise(self) -> None:
        for marker in ("₽", "руб", "руб.", "RUB", "rub"):
            for noise_first in (True, False):
                with self.subTest(marker=marker, noise_first=noise_first):
                    rows = [f"Пицца 650 {marker}", f"ИТОГО 650 {marker}"]
                    # Служебная строка не должна переключать валюту всего чека.
                    noise = ["КАРТА 650 $"]
                    result = parse_receipt_lines(noise + rows if noise_first else rows + noise)
                    self.assertEqual(result.currency, "RUB")
                    self.assertEqual(result.items_total, Decimal("650"))
                    self.assertEqual(result.receipt_total, "650")

    def test_russian_fiscal_markers_take_priority(self) -> None:
        for marker in ("ИНН 1234567890", "ККТ 123456", "ККМ 123456", "ФН 1234567890123456",
                       "ФД 1234", "ФПД 1234567890", "ОФД", "ФФД 1.2"):
            with self.subTest(marker=marker):
                result = parse_receipt_lines([marker, "Пицца 650", "ИТОГО 650", "КАРТА 650 $"])
                self.assertEqual(result.currency, "RUB")

    def test_russian_receipt_without_currency_uses_rubles(self) -> None:
        result = parse_receipt_lines([
            "Коктейль 650", "Бургер 850", "Чай 330", "Салат 570", "Пицца 1250",
            "Паста 850", "ИТОГО 4500",
        ])
        self.assertEqual(result.currency, "RUB")
        self.assertEqual(result.items_total, Decimal("4500"))
        self.assertEqual(result.receipt_total, "4500")
        self.assertFalse(result.totals_mismatch)

    def test_dollars_require_explicit_currency_evidence(self) -> None:
        for marker in ("$", "USD"):
            with self.subTest(marker=marker):
                result = parse_receipt_lines([f"Coffee 12.50 {marker}", f"TOTAL 12.50 {marker}"])
                self.assertEqual(result.currency, "USD")
                self.assertEqual(result.items_total, Decimal("12.50"))
                self.assertFalse(result.totals_mismatch)

    def test_euros_remain_euros(self) -> None:
        for marker in ("EUR", "€"):
            with self.subTest(marker=marker):
                result = parse_receipt_lines([f"Coffee 12.50 {marker}", f"TOTAL 12.50 {marker}"])
                self.assertEqual(result.currency, "EUR")
                self.assertEqual(result.items_total, Decimal("12.50"))
                self.assertFalse(result.totals_mismatch)

    def test_qr_urls_and_stray_dollar_are_not_currency_evidence(self) -> None:
        for noise in ("QR ABC$123", "https://example.test/USD", "www.example.test/$123", "$", "Код A$Z"):
            with self.subTest(noise=noise):
                result = parse_receipt_lines(["Пицца 650", "ИТОГО 650", noise])
                self.assertEqual(result.currency, "RUB")
                self.assertEqual(result.items_total, Decimal("650"))

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

