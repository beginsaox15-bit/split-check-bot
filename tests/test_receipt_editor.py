import unittest
from decimal import Decimal

from bot.receipt_editor import decimal_input, total_price, changed_item
from bot.services.receipt_models import ReceiptItem


class EditorMoneyTests(unittest.TestCase):
    def test_fractional_quantity_and_kopeck_rounding(self):
        self.assertEqual(decimal_input('0,25',quantity=True),Decimal('.25'))
        self.assertEqual(total_price('.25','.10'),'0.03')
        self.assertEqual(total_price('3','.10'),'0.30')
        self.assertEqual(total_price('1.000001','850.50'),'850.50')

    def test_decimal_bounds_and_no_exponential_or_float_input(self):
        for value in ['NaN','Infinity','1e2','-1','0','1.0000001','1000000000']:
            self.assertIsNone(decimal_input(value,quantity=True))
        for value in ['1.001','1000000000000','0','-1','1,2.3']:
            self.assertIsNone(decimal_input(value))
        self.assertEqual(decimal_input('850,50'),Decimal('850.50'))

    def test_total_overflow_is_rejected(self):
        with self.assertRaises(ValueError):
            total_price('999999999','999999999999.99')

    def test_unknown_unit_price_is_never_inferred(self):
        original=ReceiptItem(name='Суп',quantity='2',unit_price=None,total_price='1000',guest_id='3',suspicious=True)
        with self.assertRaises(ValueError):
            changed_item(original,quantity='3')
        changed=changed_item(original,name='Другой суп')
        self.assertIsNone(changed.unit_price)
        self.assertEqual(changed.total_price,'1000')
        self.assertEqual(changed.guest_id,'3')
        self.assertTrue(changed.suspicious)
        self.assertEqual(original.name,'Суп')
