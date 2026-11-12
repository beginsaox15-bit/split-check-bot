import unittest
from decimal import Decimal, localcontext

from bot.expense_split import Allocation, add_participant, assign_item, complete, participant_totals, snapshot
from bot.settlements import balances, confirm_settlement, parse_payment, set_payment, settlement_issues, transfers


class SettlementTests(unittest.TestCase):
    def plan(self, amount='9.00', count=1, gross='9.00', prepayment=None):
        plan = snapshot({'items': [{'name': 'Тестовая позиция', 'quantity': '1', 'total_price': amount}
                                   for _ in range(count)], 'gross_total': gross, 'prepayment': prepayment,
                         'currency': 'RUB'}, user_id=1, chat_id=1)
        for index in range(3):
            plan = add_participant(plan, f'Участник {index+1}')
        for item in plan.items:
            plan = assign_item(plan, item.id, ['p0', 'p1', 'p2'])
        return complete(plan)

    def assert_reconciles(self, plan):
        net = balances(plan)
        moves = transfers(plan)
        for move in moves:
            value = Decimal(move.amount)
            self.assertGreater(value, 0)
            self.assertEqual(value, value.quantize(Decimal('.01')))
            self.assertNotEqual(move.sender_id, move.recipient_id)
            net[move.sender_id] += value
            net[move.recipient_id] -= value
        self.assertTrue(all(value == 0 for value in net.values()))
        self.assertLessEqual(len(moves), len(plan.participants)-1)

    def test_single_payer_and_balance_direction(self):
        plan = set_payment(self.plan(), 'p0', '9')
        self.assertEqual(balances(plan), {'p0': Decimal('6'), 'p1': Decimal('-3'), 'p2': Decimal('-3')})
        self.assertEqual([(t.sender_id, t.recipient_id, t.amount) for t in transfers(plan)],
                         [('p1', 'p0', '3.00'), ('p2', 'p0', '3.00')])
        self.assertTrue(confirm_settlement(plan).settlement_confirmed)

    def test_multiple_payers_and_exact_reconciliation(self):
        plan = set_payment(set_payment(self.plan(), 'p0', '5'), 'p1', '4')
        self.assertEqual(balances(plan), {'p0': Decimal('2'), 'p1': Decimal('1'), 'p2': Decimal('-3')})
        self.assert_reconciles(plan)
        self.assertEqual(transfers(plan), transfers(plan))

    def test_zero_balances_need_no_transfers(self):
        plan = self.plan()
        for person in plan.participants:
            plan = set_payment(plan, person.id, '3')
        self.assertEqual(transfers(plan), [])
        self.assertTrue(confirm_settlement(plan).settlement_confirmed)

    def test_prepaid_amount_is_only_counted_when_explicitly_entered(self):
        plan = self.plan(amount='10.00', gross='10.00', prepayment='4.00')
        self.assertEqual(plan.paid_total, 0)
        plan = set_payment(plan, 'p1', '6')
        self.assertEqual(plan.paid_total, Decimal('6'))
        self.assertIn('payments_mismatch', settlement_issues(plan))
        plan = set_payment(plan, 'p0', '4')
        plan = set_payment(plan, 'p0', '4')
        self.assertEqual(plan.paid_total, Decimal('10'))
        self.assertEqual(plan.prepayment, '4.00')
        self.assert_reconciles(plan)
        self.assertTrue(confirm_settlement(plan).settlement_confirmed)

    def test_rounding_and_duplicate_items_use_existing_shares(self):
        plan = self.plan(amount='0.05', count=2, gross='0.10')
        self.assertEqual(participant_totals(plan), {'p0': Decimal('.04'), 'p1': Decimal('.04'), 'p2': Decimal('.02')})
        plan = set_payment(set_payment(plan, 'p0', '0.06'), 'p1', '0.04')
        self.assertEqual([(t.sender_id, t.recipient_id, t.amount) for t in transfers(plan)], [('p2', 'p0', '0.02')])

    def test_underpayment_overpayment_and_receipt_mismatch_block_confirmation(self):
        for paid in ('8.99', '9.01'):
            plan = set_payment(self.plan(), 'p0', paid)
            self.assertIn('payments_mismatch', settlement_issues(plan))
            with self.assertRaises(ValueError):
                transfers(plan)
            with self.assertRaises(ValueError):
                confirm_settlement(plan)
        plan = set_payment(self.plan(gross='10.00'), 'p0', '9')
        self.assertEqual(settlement_issues(plan), ['receipt_mismatch'])
        with self.assertRaises(ValueError):
            confirm_settlement(plan)

    def test_edit_replaces_and_zero_clears_payment_and_confirmation(self):
        original = self.plan()
        plan = confirm_settlement(set_payment(original, 'p0', '9'))
        changed = set_payment(plan, 'p0', '8')
        self.assertFalse(changed.settlement_confirmed)
        self.assertEqual(changed.payments['p0'], '8.00')
        self.assertEqual(set_payment(changed, 'p0', '0').paid_total, 0)
        self.assertEqual(original.payments, {})
        self.assertTrue(plan.settlement_confirmed)

    def test_reallocation_invalidates_confirmation_but_keeps_actual_payments(self):
        plan = confirm_settlement(set_payment(self.plan(), 'p0', '9'))
        plan = Allocation.model_validate({**plan.model_dump(), 'completed': False, 'settlement_confirmed': False})
        plan = complete(assign_item(plan, 'i0', ['p1']))
        self.assertEqual(plan.payments, {'p0': '9.00'})
        self.assertFalse(plan.settlement_confirmed)
        self.assertEqual(balances(plan)['p1'], Decimal('-9'))
        self.assert_reconciles(plan)

    def test_invalid_amounts_and_unknown_payers(self):
        plan = self.plan()
        for value in ('-1', '0.001', '1e2', 'NaN', 'Infinity', '', '1,2.3', '1000000000000000', 0.5, True):
            with self.subTest(value=value):
                self.assertIsNone(parse_payment(value))
                with self.assertRaises(ValueError):
                    set_payment(plan, 'p0', value)
        self.assertEqual(parse_payment(' 001,50 '), '1.50')
        with self.assertRaises(ValueError):
            set_payment(plan, 'unknown', '9')

    def test_schema_rejects_forged_confirmation_and_unknown_payer(self):
        plan = self.plan()
        for updates in ({'settlement_confirmed': True}, {'payments': {'unknown': '9.00'}},
                        {'payments': {'p0': 9.0}}, {'payments': {'p0': '0.001'}}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                Allocation.model_validate({**plan.model_dump(), **updates})
        unfinished = Allocation.model_validate({**plan.model_dump(), 'completed': False})
        with self.assertRaises(ValueError):
            set_payment(unfinished, 'p0', '9')
        self.assertIn('allocation_incomplete', settlement_issues(unfinished))

    def test_old_saved_allocation_defaults_to_no_payments(self):
        data = self.plan().model_dump(exclude={'payments', 'settlement_confirmed'})
        plan = Allocation.model_validate(data)
        self.assertEqual(plan.payments, {})
        self.assertFalse(plan.settlement_confirmed)

    def test_large_amount_and_low_decimal_context(self):
        plan = self.plan(amount='999999999999.99', count=200, gross=None)
        total = format(plan.items_total, '.2f')
        with localcontext() as context:
            context.prec = 2
            plan = set_payment(plan, 'p0', total)
            net = balances(plan)
            moves = transfers(plan)
        self.assertEqual(sum(net.values()), 0)
        self.assertEqual(sum(Decimal(t.amount) for t in moves), net['p0'])
        self.assert_reconciles(plan)

    def test_many_small_payment_combinations_conserve_money(self):
        for cents in range(1, 26):
            for first in range(cents+1):
                second = (cents-first)//2
                plan = self.plan(amount=format(Decimal(cents)/100, '.2f'), gross=None)
                for person, value in zip(plan.participants, (first, second, cents-first-second)):
                    plan = set_payment(plan, person.id, str(Decimal(value)/100))
                self.assert_reconciles(plan)
