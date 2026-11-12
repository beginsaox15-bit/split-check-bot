import unittest
from copy import deepcopy

from bot.expense_split import add_participant, assign_item, complete, snapshot
from bot.payment_input import accepts_message, new_session
from bot.receipt_states import ExpenseAllocation


class PlainPaymentInputTests(unittest.TestCase):
    def setUp(self):
        self.plan = snapshot({'items': [{'name': 'Тест', 'quantity': '1', 'total_price': '1.00'}]},
                             user_id=42, chat_id=42)
        self.plan = add_participant(add_participant(self.plan, 'Участник 1'), 'Участник 2')
        self.plan = complete(assign_item(self.plan, 'i0', ['p0', 'p1']))
        self.data = {'allocation_token': 'receipt-token', 'allocation_revision': 7,
                     'payment_person': 'p0', 'payment_last_message_id': 50,
                     'payment_last_update_id': 90}
        self.data['payment_input'] = new_session(self.data, 'p0', 100, 20)

    def accepts(self, data=None, plan=None, **kwargs):
        values = {'state_name': ExpenseAllocation.payment_amount.state,
                  'user_id': 42, 'chat_id': 42, 'message_id': 51, 'update_id': 101}
        values.update(kwargs)
        return accepts_message(data or self.data, plan or self.plan, **values)

    def test_new_plain_message_is_accepted_in_active_session(self):
        self.assertTrue(self.accepts())
        self.assertEqual(self.data['payment_input']['message_floor'], 50)

    def test_owner_chat_and_mode_are_required(self):
        for kwargs in ({'user_id': 43}, {'chat_id': 43}, {'state_name': ExpenseAllocation.payments.state},
                       {'state_name': None}, {'is_reply': True}):
            with self.subTest(kwargs=kwargs):
                self.assertFalse(self.accepts(**kwargs))

    def test_duplicate_message_and_queued_old_update_are_rejected(self):
        for kwargs in ({'message_id': 50}, {'message_id': 49}, {'update_id': 100}, {'update_id': 99},
                       {'update_id': None}, {'message_id': True}):
            with self.subTest(kwargs=kwargs):
                self.assertFalse(self.accepts(**kwargs))

    def test_receipt_revision_and_selected_payer_must_match(self):
        for key, value in (('allocation_token', 'other-receipt'), ('allocation_revision', 8),
                           ('payment_person', 'p1'), ('payment_input', None)):
            data = deepcopy(self.data)
            data[key] = value
            with self.subTest(key=key):
                self.assertFalse(self.accepts(data=data))
        data = deepcopy(self.data)
        data['payment_person'] = data['payment_input']['person_id'] = 'p99'
        self.assertFalse(self.accepts(data=data))

    def test_missing_or_malformed_session_metadata_fails_closed(self):
        for key in self.data['payment_input']:
            data = deepcopy(self.data)
            del data['payment_input'][key]
            with self.subTest(key=key):
                self.assertFalse(self.accepts(data=data))
        for key in ('payment_last_message_id', 'payment_last_update_id'):
            data = deepcopy(self.data)
            data[key] = 'invalid'
            self.assertFalse(self.accepts(data=data))

    def test_seen_ids_remain_a_barrier_after_ui_rollback(self):
        data = deepcopy(self.data)
        data.update(payment_last_message_id=51, payment_last_update_id=101)
        self.assertFalse(self.accepts(data=data))
        self.assertTrue(self.accepts(data=data, message_id=52, update_id=102))

    def test_incomplete_allocation_is_not_a_payment_session(self):
        plan = self.plan.model_copy(update={'completed': False})
        self.assertFalse(self.accepts(plan=plan))

    def test_session_requires_update_context(self):
        with self.assertRaises(ValueError):
            new_session(self.data, 'p0', None, 50)

    def test_tip_and_restaurant_input_cannot_consume_each_others_messages(self):
        data = deepcopy(self.data)
        data['payment_person'] = None
        data['payment_input'] = new_session(data, None, 100, 50, kind='tip')
        self.assertTrue(self.accepts(data=data, state_name=ExpenseAllocation.tip_amount.state))
        self.assertFalse(self.accepts(data=data))
        self.assertFalse(self.accepts(state_name=ExpenseAllocation.tip_amount.state))
        self.assertFalse(self.accepts(data=data,state_name=ExpenseAllocation.tip_payer.state))
        data['payment_person'] = data['payment_input']['person_id'] = 'p0'
        self.assertFalse(self.accepts(data=data,state_name=ExpenseAllocation.tip_amount.state))

    def test_discount_input_is_bound_to_its_own_money_session(self):
        data = deepcopy(self.data)
        data['payment_person'] = None
        data['payment_input'] = new_session(data, None, 100, 50, kind='discount')
        self.assertTrue(self.accepts(data=data,state_name=ExpenseAllocation.discount_amount.state))
        for state in (ExpenseAllocation.payment_amount.state,ExpenseAllocation.tip_amount.state):
            self.assertFalse(self.accepts(data=data,state_name=state))
        self.assertFalse(self.accepts(state_name=ExpenseAllocation.discount_amount.state))
        data['payment_person'] = data['payment_input']['person_id'] = 'p0'
        self.assertFalse(self.accepts(data=data,state_name=ExpenseAllocation.discount_amount.state))
