import unittest
from decimal import Decimal, localcontext

from bot.expense_split import (Allocation, add_participant, assign_item, complete,
                               participant_totals, snapshot, split_amount)


class ExpenseSplitTests(unittest.TestCase):
    def plan(self, amount='0.05', count=2, total=None):
        return snapshot({'items': [{'name': 'Тестовая позиция', 'quantity': '1',
                                    'total_price': amount, 'guest_id': str(i+1)} for i in range(count)],
                         'gross_total': total, 'currency': 'RUB'}, user_id=1, chat_id=1)

    def people(self, plan, count=3):
        for i in range(count):
            plan = add_participant(plan, f'Участник {i+1}')
        return plan

    def test_kopecks_remainder_and_zero_shares(self):
        self.assertEqual(split_amount('0.05', ['a', 'b', 'c']),
                         {'a': Decimal('.02'), 'b': Decimal('.02'), 'c': Decimal('.01')})
        self.assertEqual(split_amount('0.01', ['a', 'b', 'c']),
                         {'a': Decimal('.01'), 'b': Decimal('0'), 'c': Decimal('0')})
        self.assertEqual(split_amount('0.00', ['a']), {'a': Decimal(0)})
        self.assertEqual(split_amount('100', ['a']), {'a': Decimal(100)})

    def test_every_split_preserves_exact_sum(self):
        for cents in range(201):
            for count in range(1, 31):
                with self.subTest(cents=cents, count=count):
                    shares = split_amount(str(Decimal(cents)/100), [str(i) for i in range(count)])
                    self.assertEqual(sum(shares.values()), Decimal(cents)/100)
                    self.assertLessEqual(max(shares.values())-min(shares.values()), Decimal('.01'))

    def test_low_decimal_context_does_not_change_money(self):
        with localcontext() as context:
            context.prec = 2
            shares = split_amount('999999999999.99', ['a', 'b', 'c'])
        self.assertEqual(sum(shares.values()), Decimal('999999999999.99'))

    def test_duplicate_items_and_guest_metadata_remain_distinct(self):
        plan = self.people(self.plan())
        original = plan.model_dump()
        first, second = plan.items
        self.assertNotEqual(first.id, second.id)
        self.assertEqual([i.guest_id for i in plan.items], ['1', '2'])
        plan = assign_item(plan, first.id, ['p2', 'p1', 'p0'])
        plan = assign_item(plan, second.id, ['p2'])
        self.assertEqual(participant_totals(plan),
                         {'p0': Decimal('.02'), 'p1': Decimal('.02'), 'p2': Decimal('.06')})
        self.assertEqual([i.model_dump() for i in plan.items], original['items'])
        self.assertTrue(complete(plan).completed)

    def test_receipt_mismatch_is_not_used_to_balance_items(self):
        plan = self.people(self.plan(amount='1.00', total='3.00'), 1)
        for item in plan.items:
            plan = assign_item(plan, item.id, ['p0'])
        self.assertTrue(plan.mismatch)
        self.assertEqual(plan.items_total, Decimal('2.00'))
        self.assertEqual(participant_totals(plan)['p0'], Decimal('2.00'))
        self.assertEqual(complete(plan).gross_total, '3.00')

    def test_empty_assignments_block_completion_and_can_be_cleared(self):
        plan = self.people(self.plan(), 1)
        with self.assertRaises(ValueError):
            complete(plan)
        plan = assign_item(plan, 'i0', ['p0'])
        self.assertEqual([i.id for i in plan.unassigned], ['i1'])
        plan = assign_item(plan, 'i0', [])
        self.assertEqual(len(plan.unassigned), 2)

    def test_unknown_ids_and_duplicate_owners_are_rejected(self):
        plan = self.people(self.plan(), 1)
        for item, people in [('i99', ['p0']), ('i0', ['p99']), ('i0', ['p0', 'p0'])]:
            with self.subTest(item=item, people=people), self.assertRaises(ValueError):
                assign_item(plan, item, people)
        self.assertEqual(plan.assignments, {})

    def test_completed_plan_cannot_be_changed(self):
        plan = self.people(self.plan(count=1), 1)
        plan = complete(assign_item(plan, 'i0', ['p0']))
        with self.assertRaises(ValueError):
            assign_item(plan, 'i0', [])
        with self.assertRaises(ValueError):
            add_participant(plan, 'Новый участник')

    def test_names_and_participant_limit(self):
        plan = add_participant(self.plan(), '  Участник  ')
        self.assertEqual(plan.participants[0].name, 'Участник')
        for name in ('', '  ', 'участник', 'x'*61, 'a\nb', 'a\x00b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                add_participant(plan, name)
        plan = self.people(self.plan(), 30)
        with self.assertRaises(ValueError):
            add_participant(plan, 'Ещё один')

    def test_manual_and_recognized_snapshots_preserve_facts(self):
        manual = snapshot({'items': [{'name': 'Тест', 'price': '.10', 'quantity': 3}],
                           'receipt_total': '0.30', 'currency': 'RUB'}, user_id=1, chat_id=2)
        self.assertEqual(manual.items[0].amount, '0.30')
        data = {'items': [{'name': 'Тест', 'quantity': '0.5', 'unit_price': '99.00',
                           'total_price': '1.00', 'guest_id': '3'}],
                'gross_total': '9.00', 'receipt_total': '8.00', 'prepayment': '2.00',
                'amount_due': '7.00', 'currency': None}
        plan = snapshot(data, user_id=1, chat_id=2)
        self.assertEqual(plan.items[0].amount, '1.00')
        self.assertEqual(plan.items[0].guest_id, '3')
        self.assertEqual(plan.gross_total, '9.00')
        self.assertEqual(plan.prepayment, '2.00')
        self.assertIsNone(plan.currency)
        self.assertEqual(Allocation.model_validate_json(plan.model_dump_json()), plan)

    def test_sub_kopeck_manual_amount_is_rejected_without_rounding(self):
        with self.assertRaises(ValueError):
            snapshot({'items': [{'name': 'Тест', 'price': '0.005', 'quantity': 1}]}, user_id=1, chat_id=1)
        for value in ('-1', 'NaN', 'Infinity', '0.001'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                split_amount(value, ['a'])

    def test_binary_floats_are_never_accepted_as_money(self):
        for value in (0.5, 1.0, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                split_amount(value, ['a'])
        with self.assertRaises(ValueError):
            snapshot({'items': [{'name': 'Тест', 'quantity': '1', 'total_price': 0.5}]}, user_id=1, chat_id=1)

    def test_partial_totals_and_unassigned_amount_reconcile(self):
        plan = self.people(self.plan(amount='99.99', count=3))
        plan = assign_item(plan, 'i0', ['p0', 'p1', 'p2'])
        plan = assign_item(plan, 'i1', ['p0'])
        remaining = sum((Decimal(item.amount) for item in plan.unassigned), Decimal(0))
        self.assertEqual(sum(participant_totals(plan).values()) + remaining, plan.items_total)
