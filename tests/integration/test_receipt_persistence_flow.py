"""Real dispatcher + staged FSM + temporary SQLite, with no network or OCR."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import EditMessageText, SendMessage

from tests.integration import test_expense_allocation_flow as flow
from bot.handlers import router
from bot.receipt_persistence import install_persistence
from bot.receipt_states import ExpenseAllocation
from bot.receipt_store import ReceiptStore, StoreUnavailable


class ReceiptPersistenceFlowTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        flow.ExpenseAllocationFlowTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        flow.ExpenseAllocationFlowTests.tearDownClass.__func__(cls)

    async def asyncSetUp(self):
        await flow.ExpenseAllocationFlowTests.asyncSetUp(self)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ReceiptStore(Path(self.tmp.name) / 'receipts.sqlite3')
        await self.store.initialize()
        self.middleware = install_persistence(router, self.store)

    async def asyncTearDown(self):
        router.message.outer_middleware.unregister(self.middleware)
        router.callback_query.outer_middleware.unregister(self.middleware)
        await flow.ExpenseAllocationFlowTests.asyncTearDown(self)

    async def stored(self):
        data = await self.context().get_data()
        return await self.store.get(data['saved_receipt_id'], user_id=42, chat_id=42)

    async def history(self):
        await self.message('/start')
        return await self.callback('my_receipts')

    def history_code(self, receipt_id=None):
        return next(b.callback_data for row in self.menu.inline_keyboard for b in row
                    if b.callback_data.startswith('rc:') and ':open:' in b.callback_data
                    and (receipt_id is None or b.callback_data.endswith(':'+receipt_id)))

    async def restart_fsm(self):
        await self.dispatcher.storage.close()
        self.dispatcher.fsm.storage = MemoryStorage()
        # New repository object, same on-disk database (no in-memory cache).
        self.store = ReceiptStore(self.store.path)
        await self.store.initialize()
        self.middleware.store = self.store

    async def test_partial_allocation_survives_restart_and_new_screen_rejects_old_buttons(self):
        await self.manual(count=2, price='10')
        await self.click('begin')
        await self.people(2)
        await self.click('items')
        await self.assign(0, [0, 1])
        old = self.code('select', 1)
        before = await self.stored()
        old_data = await self.context().get_data()
        old_anchor = self.menu_message_id
        await self.restart_fsm()
        await self.history()
        methods = await self.callback(self.history_code(before.id))
        self.assertEqual(len([m for m in methods if isinstance(m, SendMessage)]), 1)
        self.assertNotEqual(self.menu_message_id, old_anchor)
        after = await self.stored()
        self.assertEqual(after.snapshot, before.snapshot)
        data = await self.context().get_data()
        self.assertNotEqual(data['allocation_token'], old_data['allocation_token'])
        self.assertNotIn('payment_input', data)
        self.assertEqual(len((await self.plan()).unassigned), 1)
        self.assertEqual(await self.context().get_state(), ExpenseAllocation.overview.state)
        unchanged = await self.context().get_data()
        for stale in (old, 'cancel_receipt', 'confirm_recognition', 'finish_receipt', 'edit_recognition'):
            await self.callback(stale)
            self.assertEqual(await self.context().get_data(), unchanged)
        await self.click('items')
        await self.assign(1, [1])
        self.assertFalse((await self.plan()).unassigned)

    async def test_discounts_tips_payments_and_confirmation_survive_restart(self):
        await self.payment_ready(count=1, price='100', people=2)
        await self.click('discount')
        await self.message('20')
        await self.enter_payment(0, '80')
        await self.click('tips')
        await self.message('10.01')
        await self.click('tip_who', 1)
        await self.click('pay_confirm')
        saved = await self.stored()
        self.assertEqual(saved.status, 'completed')
        self.assertEqual(saved.state, 'payments')
        self.assertEqual(saved.snapshot.allocation.discount_amount, '20.00')
        self.assertEqual(saved.snapshot.allocation.tip_amount, '10.01')
        await self.restart_fsm()
        await self.history()
        methods = await self.callback(self.history_code(saved.id))
        self.assertIn('Расчёт переводов подтверждён', self.text(methods))
        self.assertEqual((await self.plan()), saved.snapshot.allocation)
        await self.enter_payment(0, '79')
        self.assertEqual((await self.stored()).status, 'draft')

    async def test_two_receipts_start_and_new_receipt_do_not_mix(self):
        await self.payment_ready(price='10', people=1)
        await self.enter_payment(0, '10')
        first = await self.stored()
        await self.manual(count=1, price='20')
        second = await self.stored()
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(second.snapshot.allocation.participants, [])
        self.assertEqual(second.snapshot.allocation.payments, {})
        await self.history()
        self.assertEqual(len(await self.store.list(user_id=42, chat_id=42)), 2)
        await self.callback(self.history_code(first.id))
        self.assertEqual(await self.plan(), first.snapshot.allocation)
        await self.history()
        await self.callback(self.history_code(second.id))
        self.assertEqual(await self.plan(), second.snapshot.allocation)

    async def test_foreign_id_callback_and_copied_fsm_are_rejected(self):
        await self.manual(count=1)
        own = await self.stored()
        await self.history()
        code = self.history_code(own.id)
        original = await self.context().get_data()
        for user, chat in ((43, 42), (42, 43)):
            await self.context(user, chat).set_data(original)
            methods = await self.callback(code, user=user, chat=chat)
            self.assertIn('чужое', self.text(methods))
            self.assertFalse(any(isinstance(m, SendMessage) for m in methods))
        # Valid history token with a forged ID still performs scoped SQL lookup.
        other = await self.store.save(own.snapshot.model_copy(deep=True).model_copy(update={
            'allocation': own.snapshot.allocation.model_copy(update={'owner_user_id': 43})}))
        forged = code.rsplit(':', 1)[0]+':'+other.id
        methods = await self.callback(forged)
        self.assertIn('недоступен', self.text(methods))
        self.assertEqual(await self.context().get_data(), original)

    async def test_telegram_edit_failure_does_not_save_assignment_or_payment(self):
        await self.payment_ready(price='10', people=2)
        await self.enter_payment(0, '10')
        await self.click('pay_confirm')
        before = await self.stored()
        await self.click('paid')
        self.edit_error = TelegramBadRequest(method=EditMessageText(chat_id=42, message_id=1, text='x'),
                                             message='Bad Request: failed')
        await self.message('9')
        self.assertEqual(await self.stored(), before)
        self.assertEqual(await self.plan(), before.snapshot.allocation)
        self.edit_error = None
        await self.click('payments')
        await self.click('split_back')
        await self.click('reopen')
        await self.click('items')
        await self.click('select')
        await self.click('toggle', 0)
        before = await self.stored()
        data = await self.context().get_data()
        self.edit_error = TelegramBadRequest(method=EditMessageText(chat_id=42, message_id=1, text='x'),
                                             message='Bad Request: failed')
        await self.click('apply')
        self.assertEqual(await self.stored(), before)
        self.assertEqual(await self.context().get_data(), data)

    async def test_database_failure_compensates_visible_edit_and_preserves_snapshot(self):
        await self.payment_ready(price='10', people=2)
        await self.enter_payment(0, '10')
        await self.click('pay_confirm')
        before = await self.stored()
        await self.click('paid')
        data = await self.context().get_data()
        with patch.object(self.store, 'save', side_effect=StoreUnavailable('offline')):
            methods = await self.message('9')
        edits = [m for m in methods if isinstance(m, EditMessageText)]
        self.assertEqual(len(edits), 2)  # Applied UI then compensating previous UI, same message.
        self.assertEqual(edits[0].message_id, edits[1].message_id)
        self.assertEqual(edits[-1].text, data['allocation_screen_text'])
        self.assertEqual(await self.stored(), before)
        self.assertEqual(await self.plan(), before.snapshot.allocation)
        self.assertEqual((await self.context().get_data())['allocation_revision'], data['allocation_revision'])
        # Input remains active for a fresh message; no double application.
        await self.message('9')
        self.assertEqual((await self.stored()).snapshot.allocation.payments, {'p0': '9.00'})

    async def test_repeated_updates_do_not_duplicate_receipts_or_versions(self):
        await self.manual(count=1)
        before = await self.stored()
        await self.callback('finish_receipt')
        self.assertEqual(await self.stored(), before)
        await self.click('begin')
        await self.click('add')
        payload = {'message': {'message_id': 1000, 'date': 1,
            'from': {'id': 42, 'is_bot': False, 'first_name': 'Test'},
            'chat': {'id': 42, 'type': 'private'}, 'text': 'Участник'}}
        await self.send(payload)
        after = await self.stored()
        await self.send(payload)
        self.assertEqual(await self.stored(), after)
        self.assertEqual(len(await self.store.list(user_id=42, chat_id=42)), 1)

    async def test_restoration_rejects_corrupt_record(self):
        await self.manual(count=1)
        before = await self.stored()
        await self.history()
        code = self.history_code(before.id)
        data = await self.context().get_data()
        with sqlite3.connect(self.store.path) as db:
            db.execute('UPDATE receipts SET payload=?', ('INVALID PRIVATE DATA',))
        methods = await self.callback(code)
        self.assertIn('повреждён', self.text(methods))
        self.assertEqual(await self.context().get_data(), data)

    async def test_first_confirmation_replay_after_lost_fsm_publication_is_idempotent(self):
        await self.message('/start')
        await self.callback('split_check')
        await self.callback('manual_entry')
        await self.message('Тестовая позиция')
        await self.message('10')
        await self.message('1')
        original, previous = await self.context().get_data(), await self.context().get_state()
        payload = self.callback_payload('finish_receipt')
        payload['update_id'] = 9000
        await self.send(payload)
        saved = await self.stored()
        # Simulate the tiny crash window after durable COMMIT, before FSM publication.
        await self.context().set_data(original)
        await self.context().set_state(previous)
        await self.send(payload)
        self.assertEqual(await self.stored(), saved)
        self.assertEqual(len(await self.store.list(user_id=42, chat_id=42)), 1)

    async def test_database_failure_during_confirmation_leaves_no_receipt(self):
        await self.message('/start')
        await self.callback('split_check')
        await self.callback('manual_entry')
        await self.message('Тестовая позиция')
        await self.message('10')
        await self.message('1')
        original = await self.context().get_data()
        with patch.object(self.store, 'save', side_effect=StoreUnavailable('offline')):
            methods = await self.callback('finish_receipt')
        self.assertEqual(await self.context().get_data(), original)
        self.assertEqual(await self.store.list(user_id=42, chat_id=42), [])
        self.assertIn('Чек не сохранён', next(m.text for m in methods if isinstance(m, EditMessageText)))
        await self.callback('finish_receipt')
        self.assertEqual(len(await self.store.list(user_id=42, chat_id=42)), 1)

    async def test_failed_compensation_invalidates_buttons_and_money_input(self):
        await self.payment_ready(price='10', people=1)
        await self.enter_payment(0, '10')
        before = await self.stored()
        await self.click('paid')
        data = await self.context().get_data()
        async def reject(*args, **kwargs):
            from bot.allocation_screen import ScreenEditError
            raise ScreenEditError('offline')
        with patch.object(self.store, 'save', side_effect=StoreUnavailable('offline')), \
                patch('bot.receipt_persistence.render', side_effect=reject):
            await self.message('9')
        after = await self.context().get_data()
        self.assertNotEqual(after['allocation_token'], data['allocation_token'])
        self.assertIsNone(after['payment_input'])
        self.assertIsNone(after['allocation_screen_id'])
        self.assertEqual(await self.context().get_state(), ExpenseAllocation.done.state)
        self.assertEqual(await self.stored(), before)
        await self.message('9')
        self.assertEqual(await self.stored(), before)
        await self.message('/allocation')
        await self.click('payments')
        self.assertEqual(await self.plan(), before.snapshot.allocation)

    async def test_history_navigation_edits_one_screen_and_rejects_old_history(self):
        await self.manual(count=1)
        one = await self.stored()
        for _ in range(9):
            await self.store.save(one.snapshot)
        methods = await self.history()
        anchor = self.menu_message_id
        self.assertEqual(len([m for m in methods if isinstance(m, EditMessageText)]), 1)
        old_open = self.history_code()
        next_page = next(b.callback_data for row in self.menu.inline_keyboard for b in row if ':page:1' in b.callback_data)
        self.assert_single_edit(await self.callback(next_page), anchor)
        data = await self.context().get_data()
        await self.callback(old_open)
        self.assertEqual(await self.context().get_data(), data)

    async def test_old_name_and_money_messages_after_restore_cannot_apply(self):
        await self.manual(count=1)
        saved = await self.stored()
        await self.restart_fsm()
        await self.history()
        await self.callback(self.history_code(saved.id))
        await self.click('add')
        await self.message('Старое имя', update_id=1)
        self.assertEqual((await self.plan()).participants, [])
        await self.message('Новый участник')
        await self.click('items')
        await self.assign(0, [0])
        await self.click('overview')
        await self.click('finish')
        await self.click('payments')
        await self.click('paid')
        await self.message('0.05', update_id=1)
        self.assertEqual((await self.plan()).payments, {})

    async def test_stale_database_version_rolls_back_edit_instead_of_overwriting(self):
        await self.payment_ready(price='10', people=1)
        before = await self.stored()
        from bot.settlements import set_payment
        newer = before.snapshot.model_copy(deep=True)
        newer.allocation = set_payment(newer.allocation, 'p0', '8')
        changed = await self.store.save(newer, receipt_id=before.id, expected_version=before.version)
        await self.click('paid')
        methods = await self.message('9')
        self.assertIn('другой сессии', self.text(methods))
        self.assertEqual(await self.stored(), changed)
        self.assertEqual((await self.plan()).payments, {})
        await self.history()
        await self.callback(self.history_code(changed.id))
        self.assertEqual((await self.plan()).payments, {'p0': '8.00'})

    async def test_navigation_and_unsaved_tip_draft_do_not_write_business_snapshot(self):
        await self.payment_ready(price='10', people=1)
        saved = await self.stored()
        await self.click('tips')
        await self.message('2.01')
        self.assertEqual(await self.stored(), saved)
        await self.click('payments')
        self.assertEqual(await self.stored(), saved)

    async def test_history_database_error_preserves_active_session_without_edit(self):
        await self.manual(count=1)
        before = await self.context().get_data()
        saved = await self.stored()
        with patch.object(self.store, 'list', side_effect=StoreUnavailable('offline')):
            methods = await self.callback('my_receipts')
        self.assertFalse(any(isinstance(m, (SendMessage, EditMessageText)) for m in methods))
        self.assertEqual(await self.context().get_data(), before)
        self.assertEqual(await self.stored(), saved)

    async def test_restore_send_error_preserves_previous_fsm_and_db(self):
        await self.manual(count=1)
        saved = await self.stored()
        await self.history()
        before = await self.context().get_data()
        with patch('bot.allocation_handlers.render', side_effect=TelegramNetworkError(
                method=SendMessage(chat_id=42, text='x'), message='offline')):
            await self.callback(self.history_code(saved.id))
        self.assertEqual((await self.context().get_data())['history_token'], before['history_token'])
        self.assertEqual((await self.store.get(saved.id, user_id=42, chat_id=42)), saved)


# Reuse only transport/scenario helpers, without inheriting the original test cases.
for _name in ('context', 'send', 'callback_payload', 'callback', 'text', 'message', 'code', 'click',
              'plan', 'manual', 'people', 'assign', 'payment_ready', 'enter_payment', 'assert_single_edit'):
    setattr(ReceiptPersistenceFlowTests, _name, getattr(flow.ExpenseAllocationFlowTests, _name))
