import unittest
from decimal import Decimal, localcontext

from bot.expense_split import Allocation, add_participant, assign_item, complete, participant_totals, snapshot
from bot.settlements import (balances, confirm_settlement, expense_shares, paid_by, set_payment,
                             set_tips, settlement_issues, tip_shares, transfers)


class TipTests(unittest.TestCase):
    def plan(self, amounts=('3.00', '3.00', '3.00')):
        plan = snapshot({'items': [{'name': 'Тестовая позиция', 'quantity': '1', 'total_price': value}
                                  for value in amounts], 'gross_total': str(sum(map(Decimal, amounts))),
                         'prepayment': '1.00', 'amount_due': str(sum(map(Decimal, amounts))-1),
                         'currency': 'RUB'}, user_id=42, chat_id=42)
        for index in range(3):
            plan = add_participant(plan, f'Участник {index+1}')
        for index, item in enumerate(plan.items):
            plan = assign_item(plan, item.id, [f'p{index%3}'])
        return complete(plan)

    def assert_balanced(self, plan):
        net = balances(plan)
        self.assertEqual(sum(net.values()), 0)
        for move in transfers(plan):
            net[move.sender_id] += Decimal(move.amount)
            net[move.recipient_id] -= Decimal(move.amount)
        self.assertTrue(all(value == 0 for value in net.values()))

    def test_tips_equal_shares_without_changing_restaurant_facts(self):
        original = set_payment(self.plan(), 'p1', '9')
        tipped = set_tips(original, '0.90', 'p2')
        self.assertEqual(tip_shares(tipped), dict.fromkeys(('p0','p1','p2'), Decimal('.30')))
        for key in ('items','assignments','gross_total','prepayment','amount_due','payments','currency'):
            self.assertEqual(tipped.model_dump()[key], original.model_dump()[key])
        self.assertEqual(tipped.paid_total, Decimal('9'))
        self.assertEqual(tipped.expenses_total, Decimal('9.90'))
        self.assert_balanced(tipped)

    def test_example_14240_plus_900(self):
        plan = set_tips(set_payment(self.plan(('5130','4810','4300')), 'p1', '14240'), '900', 'p2')
        self.assertEqual(expense_shares(plan), {'p0': Decimal('5430'), 'p1': Decimal('5110'), 'p2': Decimal('4600')})
        self.assertEqual(balances(plan), {'p0': Decimal('-5430'), 'p1': Decimal('9130'), 'p2': Decimal('-3700')})
        self.assertEqual([(m.sender_id,m.recipient_id,m.amount) for m in transfers(plan)],
                         [('p0','p1','5430.00'), ('p2','p1','3700.00')])
        self.assertEqual(paid_by(plan, 'p1'), Decimal('14240'))
        self.assertEqual(paid_by(plan, 'p2'), Decimal('900'))
        self.assertEqual(plan.expenses_total, Decimal('15140'))
        self.assertTrue(confirm_settlement(plan).settlement_confirmed)
        self.assert_balanced(plan)

    def test_kopecks_follow_participant_order_even_if_payer_is_last(self):
        original = set_payment(self.plan(), 'p0', '9')
        for cents in range(1,31):
            plan = set_tips(original, str(Decimal(cents)/100), 'p2')
            shares = tip_shares(plan)
            self.assertEqual(sum(shares.values()), Decimal(cents)/100)
            self.assertEqual(list(shares), ['p0','p1','p2'])
            self.assertGreaterEqual(shares['p0'],shares['p1'])
            self.assertGreaterEqual(shares['p1'],shares['p2'])
            self.assertLessEqual(shares['p0']-shares['p2'], Decimal('.01'))
            self.assert_balanced(plan)

    def test_change_amount_and_payer_invalidate_confirmation_without_accumulation(self):
        plan = confirm_settlement(set_tips(set_payment(self.plan(), 'p1', '9'), '1', 'p0'))
        changed = set_tips(plan, '2', 'p2')
        self.assertFalse(changed.settlement_confirmed)
        self.assertEqual(changed.tip_amount, '2.00')
        self.assertEqual(paid_by(changed, 'p0'), 0)
        self.assertEqual(paid_by(changed, 'p2'), Decimal('2'))
        self.assertEqual(set_tips(changed, '2', 'p2'), changed)
        again = set_tips(confirm_settlement(changed), '2', 'p0')
        self.assertFalse(again.settlement_confirmed)
        self.assert_balanced(again)

    def test_zero_removes_tips_and_payer(self):
        original = set_payment(self.plan(), 'p0', '9')
        changed = set_tips(set_tips(original, '1', 'p2'), '0', None)
        self.assertEqual(changed.tip_amount, '0.00')
        self.assertIsNone(changed.tip_payer_id)
        self.assertEqual(balances(changed), balances(original))
        self.assertEqual(transfers(changed), transfers(original))

    def test_missing_payer_and_double_counting_block_final_transfers(self):
        base = set_payment(self.plan(), 'p0', '9')
        pending = Allocation.model_validate({**base.model_dump(), 'tip_amount': '1.00'})
        self.assertIn('tips_unpaid',settlement_issues(pending))
        for action in (transfers, confirm_settlement):
            with self.assertRaises(ValueError): action(pending)
        with self.assertRaises(ValueError):
            Allocation.model_validate({**pending.model_dump(), 'settlement_confirmed': True})
        double = set_tips(set_payment(base, 'p0', '10'), '1', 'p1')
        self.assertEqual(settlement_issues(double), ['payments_mismatch'])
        with self.assertRaises(ValueError): transfers(double)
        mismatch = Allocation.model_validate({**set_tips(base,'1','p1').model_dump(), 'gross_total': '10'})
        self.assertIn('receipt_mismatch',settlement_issues(mismatch))
        with self.assertRaises(ValueError): confirm_settlement(mismatch)

    def test_invalid_tips_and_unknown_payer(self):
        plan = self.plan()
        for value in ('-1','0.001','NaN','1e2','','1,2.3','1000000000000000',True,0.5):
            with self.subTest(value=value), self.assertRaises(ValueError): set_tips(plan,value,'p0')
        for payer in (None,'p99'):
            with self.assertRaises(ValueError): set_tips(plan,'1',payer)
        for fields in ({'tip_amount':'-1'}, {'tip_amount':'0.001'}, {'tip_amount':0.5},
                       {'tip_payer_id':'p99'}, {'tip_amount':'0','tip_payer_id':'p0'}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                Allocation.model_validate({**plan.model_dump(),**fields})
        unfinished = Allocation.model_validate({**plan.model_dump(),'completed':False})
        with self.assertRaises(ValueError): set_tips(unfinished,'1','p0')

    def test_legacy_data_and_low_precision_context(self):
        plan = self.plan()
        legacy = Allocation.model_validate(plan.model_dump(exclude={'tip_amount','tip_payer_id','ruble_evidence'}))
        self.assertEqual(legacy.tip_amount,'0.00')
        self.assertIsNone(legacy.tip_payer_id)
        with localcontext() as context:
            context.prec = 2
            tipped = set_tips(set_payment(plan,'p0','9'),'999999999999999.99','p2')
            shares, net, moves = expense_shares(tipped), balances(tipped), transfers(tipped)
            self.assertEqual(tipped.expenses_total, Decimal('1000000000000008.99'))
        self.assertEqual(sum(shares.values()), tipped.expenses_total)
        self.assertEqual(sum(net.values()),0)
        self.assertTrue(moves)
        self.assert_balanced(tipped)
