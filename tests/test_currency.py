import unittest
from copy import deepcopy
from unittest.mock import AsyncMock

from bot.currency import receipt_currency, rubles_confirmed, symbol
from bot.expense_split import snapshot
from bot.allocation_handlers import currency as allocation_currency
from bot.receipt_editor import currency as editor_currency
from bot.receipt_view import show_recognition
from bot.services.receipt_models import RecognizedReceipt


class CurrencyTests(unittest.IsolatedAsyncioTestCase):
    def receipt(self, currency=None, raw_text=''):
        return RecognizedReceipt(readable=True,items=[{'name':'Тестовая позиция','quantity':'1','unit_price':None,'total_price':'100.00'}],
                                 receipt_total='100.00',currency=currency,raw_text=raw_text)

    async def test_explicit_ruble_evidence_displays_without_changing_ocr_facts(self):
        for text in ('Итого 100 ₽','Total 100.00 RUB','ИНН 1234567890\nИтого 100 руб.', '100 российских рублей'):
            with self.subTest(text=text):
                receipt = self.receipt(raw_text=text)
                before = deepcopy(receipt.model_dump())
                message = AsyncMock()
                await show_recognition(message,receipt)
                shown = ''.join(call.args[0] for call in message.answer.await_args_list)
                self.assertIn('100 ₽',shown)
                self.assertNotIn('Валюта не определена',shown)
                self.assertIn('поле OCR не изменено',shown)
                self.assertEqual(editor_currency(receipt),'₽')
                plan = snapshot(receipt.model_dump(),user_id=1,chat_id=1)
                self.assertEqual(allocation_currency(plan),'₽')
                self.assertIsNone(plan.currency)
                self.assertEqual(receipt.model_dump(),before)

    async def test_cyrillic_unknown_and_conflicting_evidence_do_not_imply_rub(self):
        for text in ('','Сыр и чай 100.00','Гость 1\nПолная сумма 100.00','ИНН 1234567890',
                     '100 рублей','100 белорусских рублей','₽ USD','₽ 100 KZT','₽ 100 PLN',
                     '₽ Currency: ABC','Total $100.00','100 ₸'):
            with self.subTest(text=text):
                receipt = self.receipt(raw_text=text)
                self.assertFalse(rubles_confirmed(text))
                self.assertIsNone(receipt_currency(receipt))
                self.assertEqual(editor_currency(receipt),'')
                self.assertIn('валюта не определена',allocation_currency(snapshot(receipt.model_dump(),user_id=1,chat_id=1)))
                message = AsyncMock()
                await show_recognition(message,receipt)
                shown = ''.join(call.args[0] for call in message.answer.await_args_list)
                self.assertIn('Валюта не определена',shown)
                self.assertNotIn(' ₽',shown)

    async def test_explicit_currency_always_wins_and_other_currencies_survive(self):
        for code in ('RUB','USD','EUR','KZT','BYN','GBP'):
            receipt = self.receipt(currency=code,raw_text='Итого 100 ₽')
            self.assertEqual(receipt_currency(receipt),code)
            self.assertEqual(editor_currency(receipt),'₽' if code=='RUB' else code)
            self.assertEqual(symbol(code,ruble_evidence=True),'₽' if code=='RUB' else code)
