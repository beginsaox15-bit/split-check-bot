"""Real dispatcher/FSM integration; Telegram transport and OCR are stubs."""
import asyncio
import itertools
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import SimpleEventIsolation
from aiogram.methods import AnswerCallbackQuery, EditMessageText, SendMessage
from aiogram.types import ForceReply, InlineKeyboardMarkup, Message, Update

from bot.handlers import router
from bot.expense_split import Allocation, participant_totals
from bot.receipt_states import ExpenseAllocation, ManualReceipt, ReceiptInput
from bot.services.receipt_models import RecognizedReceipt
from bot.settlements import balances, transfers


class ExpenseAllocationFlowTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.dispatcher = Dispatcher(events_isolation=SimpleEventIsolation())
        cls.dispatcher.include_router(router)

    @classmethod
    def tearDownClass(cls):
        # aiogram has no public detach API. Release the shared application router
        # so the existing native OCR dispatcher smoke test can attach it next.
        cls.dispatcher.sub_routers.remove(router)
        router._parent_router = None

    def context(self, user=42, chat=42):
        return FSMContext(storage=self.dispatcher.storage,
                          key=StorageKey(bot_id=123456, chat_id=chat, user_id=user))

    async def asyncSetUp(self):
        # Each IsolatedAsyncioTestCase owns a different loop; concurrent tests
        # must not reuse asyncio locks bound by an earlier test's loop.
        await self.dispatcher.fsm.events_isolation.close()
        self.dispatcher.fsm.events_isolation = SimpleEventIsolation()
        self.ids = itertools.count(100)
        self.menu = None
        self.menu_message_id = 1
        self.edit_error = None
        for user, chat in ((42, 42), (43, 42), (42, 43)):
            await self.context(user, chat).clear()
        self.recognize = self.enterContext(patch('bot.handlers.recognize_receipt', new_callable=AsyncMock))
        self.remote = self.enterContext(patch('bot.services.openrouter_receipt_recognition._post',
                                             side_effect=AssertionError('AI APIs prohibited')))
        self.download = self.enterContext(patch.object(Bot, 'download', new_callable=AsyncMock))

    async def asyncTearDown(self):
        self.remote.assert_not_called()

    async def send(self, payload):
        update = Update.model_validate({'update_id': next(self.ids), **payload})
        async with Bot(token='123456:TEST_TOKEN') as bot:
            response_ids = {}
            async def response(current_bot, method, **kwargs):
                if isinstance(method, (SendMessage, EditMessageText)):
                    if isinstance(method, EditMessageText) and self.edit_error:
                        if 'message is not modified' in self.edit_error.message.lower():
                            response_ids[id(method)] = method.message_id
                        raise self.edit_error
                    message_id = next(self.ids) if isinstance(method, SendMessage) else method.message_id
                    response_ids[id(method)] = message_id
                    return Message.model_validate({'message_id': message_id, 'date': 1,
                        'chat': {'id': method.chat_id, 'type': 'private'}, 'text': method.text,
                        'from': {'id': 123456, 'is_bot': True, 'first_name': 'Bot'}}, context={'bot': current_bot})
                return True
            with patch.object(bot.session, 'make_request', new_callable=AsyncMock, side_effect=response) as request:
                await self.dispatcher.feed_update(bot, update)
        methods = [call.args[1] for call in request.await_args_list]
        menus = [m for m in methods if isinstance(m, (SendMessage, EditMessageText))
                 and isinstance(m.reply_markup, InlineKeyboardMarkup) and id(m) in response_ids]
        if menus:
            self.menu = menus[-1].reply_markup
            self.menu_message_id = response_ids[id(menus[-1])]
        return methods

    def callback_payload(self, data, user=42, chat=42, with_message=True):
        query = {'id': str(next(self.ids)), 'chat_instance': 'test', 'data': data,
                 'from': {'id': user, 'is_bot': False, 'first_name': 'Test'}}
        if with_message:
            query['message'] = {'message_id': self.menu_message_id, 'date': 1, 'chat': {'id': chat, 'type': 'private'}, 'text': 'Menu'}
        else:
            query['inline_message_id'] = 'test'
        return {'callback_query': query}

    async def callback(self, data, **kwargs):
        return await self.send(self.callback_payload(data, **kwargs))

    def text(self, methods):
        return next(method.text for method in reversed(methods) if getattr(method, 'text', None) is not None)

    async def message(self, text, message_id=None, reply_to=None, update_id=None):
        payload = {'message': {'message_id': message_id or next(self.ids), 'date': 1,
                                           'from': {'id': 42, 'is_bot': False, 'first_name': 'Test'},
                                           'chat': {'id': 42, 'type': 'private'}, 'text': text}}
        if reply_to is not None:
            payload['message']['reply_to_message'] = {'message_id': reply_to, 'date': 1,
                                                     'chat': {'id': 42, 'type': 'private'}, 'text': 'Payment prompt'}
        if update_id is not None:
            payload['update_id'] = update_id
        return await self.send(payload)

    def code(self, action, index=0):
        return next(b.callback_data for row in self.menu.inline_keyboard for b in row
                    if b.callback_data.startswith('sp:') and b.callback_data.split(':')[3:] == [action, str(index)])

    async def click(self, action, index=0):
        return await self.callback(self.code(action, index))

    async def plan(self):
        return Allocation.model_validate((await self.context().get_data())['allocation'])

    async def manual(self, count=2, price='0.05'):
        await self.message('/start')
        await self.callback('split_check')
        await self.callback('manual_entry')
        for i in range(count):
            if i:
                await self.callback('add_item')
            await self.message('Тестовая позиция')
            await self.message(price)
            await self.message('1')
        return await self.callback('finish_receipt')

    async def people(self, count=3):
        for index in range(count):
            await self.click('add')
            await self.message(f'Участник {index+1}')

    async def assign(self, index, people):
        await self.click('select', index)
        for person in people:
            await self.click('toggle', person)
        return await self.click('apply')

    async def test_manual_duplicates_shared_kopecks_and_completion(self):
        await self.manual()
        self.assertEqual(await self.context().get_state(), ReceiptInput.confirmed.state)
        await self.click('begin')
        methods = await self.click('finish')
        self.assertIn('нераспределённые', self.text(methods))
        await self.people()
        await self.click('items')
        await self.assign(0, [2, 0, 1])
        await self.assign(1, [2, 0, 1])
        plan = await self.plan()
        self.assertEqual(len(plan.items), 2)
        self.assertEqual(participant_totals(plan),
                         {'p0': Decimal('.04'), 'p1': Decimal('.04'), 'p2': Decimal('.02')})
        await self.click('overview')
        old = self.code('finish')
        methods = await self.callback(old)
        self.assertEqual(await self.context().get_state(), ExpenseAllocation.done.state)
        self.assertIn('Распределение завершено', self.text(methods))
        self.assertIn('Нераспределённые позиции: 0', self.text(methods))
        before = await self.context().get_data()
        methods = await self.callback(old)
        self.assertEqual(await self.context().get_data(), before)
        self.assertEqual(len(methods), 1)
        self.assertIn('устарело', methods[0].text)

    async def test_recognized_editor_confirmation_and_mismatch(self):
        self.recognize.return_value = RecognizedReceipt.model_validate({
            'readable': True, 'items': [{'name': 'Тестовая позиция', 'quantity': '1', 'unit_price': '0.10',
                                       'total_price': '0.10', 'guest_id': '1'}],
            'receipt_total': '0.50', 'gross_total': '0.50', 'prepayment': '0.20',
            'amount_due': '0.30', 'currency': 'USD'})
        # Use valid monetary strings at the existing schema boundary.
        await self.callback('split_check')
        await self.send({'message': {'message_id': next(self.ids), 'date': 1,
                                    'from': {'id': 42, 'is_bot': False, 'first_name': 'Test'},
                                    'chat': {'id': 42, 'type': 'private'},
                                    'photo': [{'file_id': 'stub', 'file_unique_id': 'stub', 'width': 100, 'height': 100}]}})
        await self.callback('edit_recognition')
        async def editor(action, index=0):
            value = next(b.callback_data for row in self.menu.inline_keyboard for b in row
                         if b.callback_data.startswith('re:') and b.callback_data.split(':')[-2:] == [str(index), action])
            return await self.callback(value)
        await editor('select')
        await editor('price')
        await self.message('0.20')
        await editor('back')
        await editor('review')
        await self.callback('confirm_recognition')
        self.assertEqual((await self.plan()).items[0].amount, '0.20')
        self.assertEqual((await self.plan()).items[0].guest_id, '1')
        await self.click('begin')
        await self.people(1)
        await self.click('items')
        await self.assign(0, [0])
        await self.click('overview')
        methods = await self.click('finish')
        text = self.text(methods)
        self.assertIn('Сумма позиций не совпадает', text)
        self.assertIn('0.20 USD', text)
        self.assertIn('Полная сумма чека: 0.50 USD', text)
        self.assertIn('Предоплата по чеку: 0.20 USD', text)
        self.assertEqual((await self.plan()).gross_total, '0.50')
        self.assertTrue((await self.plan()).completed)

    async def test_stale_foreign_inline_and_forged_callbacks_do_not_mutate(self):
        await self.manual(count=1)
        old_entry = self.code('begin')
        await self.click('begin')
        await self.people(1)
        await self.click('items')
        valid = self.code('select')
        before = await self.context().get_data()
        for code, kwargs in [(old_entry, {}), (valid, {'user': 43}), (valid, {'chat': 43}),
                             (valid, {'with_message': False}), (valid.rsplit(':', 1)[0]+':999', {})]:
            methods = await self.callback(code, **kwargs)
            self.assertEqual(len(methods), 1)
            self.assertIsInstance(methods[0], AnswerCallbackQuery)
            self.assertEqual(await self.context().get_data(), before)
        # Even if a token is copied into another FSM, the explicit owner/chat
        # checks reject the callback without exposing or mutating the receipt.
        for user, chat in ((43, 42), (42, 43)):
            foreign = self.context(user, chat)
            await foreign.set_state(await self.context().get_state())
            await foreign.set_data(before)
            methods = await self.callback(valid, user=user, chat=chat)
            self.assertIn('другого', self.text(methods))
            self.assertEqual(await foreign.get_data(), before)
        await self.manual(count=1)
        before = await self.context().get_data()
        await self.callback(valid)
        self.assertEqual(await self.context().get_data(), before)

    async def test_simultaneous_duplicate_click_is_applied_once(self):
        await self.manual(count=1)
        await self.click('begin')
        await self.people(1)
        await self.click('items')
        await self.click('select')
        code = self.code('toggle')
        await asyncio.gather(self.callback(code), self.callback(code))
        self.assertEqual((await self.context().get_data())['allocation_selection'], ['p0'])

    async def test_clear_reopen_and_cancel_selection(self):
        await self.manual(count=1)
        await self.click('begin')
        await self.people(1)
        await self.click('items')
        await self.assign(0, [0])
        await self.click('overview')
        await self.click('finish')
        await self.click('reopen')
        await self.click('items')
        await self.click('select')
        await self.click('toggle')
        await self.click('items')
        self.assertEqual((await self.plan()).assignments['i0'], ['p0'])
        await self.click('select')
        await self.click('clear')
        await self.click('overview')
        methods = await self.click('finish')
        self.assertIn('нераспределённые', self.text(methods))
        self.assertFalse((await self.plan()).completed)

    async def test_name_validation_duplicate_messages_and_cancel(self):
        await self.manual(count=1)
        await self.click('begin')
        await self.click('add')
        for text in (' ', 'x'*61, 'a\nb'):
            await self.message(text)
            self.assertEqual(await self.context().get_state(), ExpenseAllocation.name.state)
        await self.message('Участник', message_id=999)
        await self.click('add')
        await self.message('Участник', message_id=999)
        self.assertEqual(len((await self.plan()).participants), 1)
        await self.message('участник')
        self.assertEqual(await self.context().get_state(), ExpenseAllocation.name.state)
        await self.click('overview')
        await self.click('back')
        self.assertEqual(await self.context().get_state(), ReceiptInput.confirmed.state)
        await self.click('begin')
        self.assertEqual(len((await self.plan()).participants), 1)
        await self.message('/start')
        self.assertEqual(await self.context().get_data(), {})

    async def test_item_pagination_and_unassigned_summary(self):
        await self.manual(count=11)
        await self.click('begin')
        await self.people(1)
        await self.click('items')
        self.assertTrue(all(len(b.callback_data.encode()) <= 64 for row in self.menu.inline_keyboard for b in row))
        await self.click('items', 1)
        await self.assign(10, [0])
        self.assertEqual(list((await self.plan()).assignments), ['i10'])
        methods = await self.click('overview')
        self.assertIn('Нераспределённые позиции: 10', self.text(methods))

    async def test_sub_kopeck_manual_input_is_not_rounded_silently(self):
        methods = await self.manual(count=1, price='0.005')
        self.assertIn('целых копейках', self.text(methods))
        self.assertEqual(await self.context().get_state(), ManualReceipt.review.state)
        self.assertEqual((await self.context().get_data())['items'][0]['price'], '0.005')

    async def test_photo_during_allocation_does_not_trigger_ocr_or_reset_receipt(self):
        await self.manual(count=1)
        await self.click('begin')
        before = await self.context().get_data()
        methods = await self.send({'message': {'message_id': next(self.ids), 'date': 1,
            'from': {'id': 42, 'is_bot': False, 'first_name': 'Test'},
            'chat': {'id': 42, 'type': 'private'},
            'photo': [{'file_id': 'stub', 'file_unique_id': 'stub', 'width': 100, 'height': 100}]}})
        self.assertIn('распределение расходов', self.text(methods))
        self.assertEqual(await self.context().get_data(), before)
        self.recognize.assert_not_awaited()

    async def payment_ready(self, count=1, price='9', people=3):
        await self.manual(count=count, price=price)
        await self.click('begin')
        await self.people(people)
        await self.click('items')
        for index in range(count):
            await self.assign(index, list(range(people)))
        await self.click('overview')
        await self.click('finish')
        return await self.click('payments')

    async def enter_payment(self, index, value):
        await self.click('paid', index)
        return await self.message(value)

    async def test_one_payer_and_confirmed_transfers(self):
        methods = await self.payment_ready()
        self.assertIn('не указано', self.text(methods))
        methods = await self.click('pay_confirm')
        self.assertIn('не подтверждён', self.text(methods))
        methods = await self.enter_payment(0, '9,00')
        self.assertIn('Предварительные переводы:', self.text(methods))
        self.assertIn('Участник 2 → Участник 1: 3.00 ₽', self.text(methods))
        old = self.code('pay_confirm')
        methods = await self.callback(old)
        self.assertIn('Расчёт переводов подтверждён ✅', self.text(methods))
        self.assertTrue((await self.plan()).settlement_confirmed)
        before = await self.context().get_data()
        await self.callback(old)
        self.assertEqual(await self.context().get_data(), before)

    async def test_multiple_payers_prepaid_amount_is_explicit(self):
        await self.payment_ready(price='10')
        plan = await self.plan()
        plan = Allocation.model_validate({**plan.model_dump(), 'prepayment': '4.00', 'amount_due': '6.00'})
        await self.context().update_data(allocation=plan.model_dump(mode='json'))
        methods = await self.enter_payment(1, '6')
        self.assertEqual((await self.plan()).paid_total, Decimal('6'))
        self.assertIn('автоматически не добавляется', self.text(methods))
        self.assertIn('Сумма оплат не совпадает', self.text(methods))
        self.assertNotIn(' → ', self.text(methods))
        methods = await self.enter_payment(0, '4')
        self.assertEqual((await self.plan()).paid_total, Decimal('10'))
        self.assertIn('Предварительные переводы', self.text(methods))
        await self.click('pay_confirm')
        self.assertTrue((await self.plan()).settlement_confirmed)

    async def test_payment_edit_replaces_value_and_invalidates_confirmation(self):
        await self.payment_ready()
        await self.enter_payment(0, '9')
        await self.click('pay_confirm')
        methods = await self.enter_payment(0, '8.99')
        self.assertEqual((await self.plan()).payments, {'p0': '8.99'})
        self.assertFalse((await self.plan()).settlement_confirmed)
        self.assertIn('Сумма оплат не совпадает', self.text(methods))
        self.assertNotIn(' → ', self.text(methods))
        await self.enter_payment(0, '0')
        self.assertEqual((await self.plan()).paid_total, 0)
        await self.enter_payment(1, '9')
        await self.click('pay_confirm')
        self.assertTrue((await self.plan()).settlement_confirmed)

    async def test_receipt_mismatch_blocks_final_debts_even_with_balanced_payments(self):
        await self.payment_ready()
        plan = await self.plan()
        plan = Allocation.model_validate({**plan.model_dump(), 'gross_total': '10.00'})
        await self.context().update_data(allocation=plan.model_dump(mode='json'))
        methods = await self.enter_payment(0, '9')
        self.assertIn('Сумма позиций не совпадает', self.text(methods))
        self.assertNotIn(' → ', self.text(methods))
        await self.click('pay_confirm')
        self.assertFalse((await self.plan()).settlement_confirmed)

    async def test_old_replies_duplicates_and_invalid_payment_input(self):
        await self.payment_ready()
        await self.click('paid')
        original_session = (await self.context().get_data())['payment_input']
        methods = await self.message('0.001')
        self.assert_single_edit(methods, self.menu_message_id)
        self.assertIn('Нужна неотрицательная', self.text(methods))
        self.assertEqual((await self.plan()).payments, {})
        fresh_session = (await self.context().get_data())['payment_input']
        self.assertNotEqual(original_session['revision'], fresh_session['revision'])
        await self.message('9', reply_to=1)  # Old ForceReply replies are not ordinary input.
        self.assertEqual((await self.plan()).payments, {})
        await self.message('9', message_id=999)
        self.ids = itertools.count(1000)
        await self.click('paid', 1)
        before = await self.context().get_data()
        await self.message('9', message_id=999)
        self.assertEqual(await self.context().get_data(), before)
        await asyncio.gather(self.message('0', message_id=2000), self.message('0', message_id=2000))
        self.assertEqual((await self.plan()).payments, {'p0': '9.00', 'p1': '0.00'})

    async def test_payment_callbacks_are_scoped_and_forged_indices_rejected(self):
        await self.payment_ready()
        valid = self.code('paid')
        before = await self.context().get_data()
        for user, chat in ((43, 42), (42, 43)):
            await self.context(user, chat).set_state(ExpenseAllocation.payments)
            await self.context(user, chat).set_data(before)
            methods = await self.callback(valid, user=user, chat=chat)
            self.assertIn('другого', self.text(methods))
            self.assertEqual(await self.context(user, chat).get_data(), before)
        await self.callback(valid.rsplit(':', 1)[0]+':999')
        self.assertEqual(await self.context().get_data(), before)
        await self.click('paid')
        after = await self.context().get_data()
        await self.callback(valid)
        self.assertEqual(await self.context().get_data(), after)

    async def test_reallocation_keeps_payments_and_recalculates_balances(self):
        await self.payment_ready()
        await self.enter_payment(0, '9')
        await self.click('pay_confirm')
        await self.click('split_back')
        await self.click('reopen')
        self.assertFalse((await self.plan()).settlement_confirmed)
        await self.click('items')
        await self.click('select')
        await self.click('clear')
        await self.assign(0, [1])
        await self.click('overview')
        await self.click('finish')
        methods = await self.click('payments')
        self.assertEqual((await self.plan()).payments, {'p0': '9.00'})
        self.assertEqual(balances(await self.plan())['p1'], Decimal('-9'))
        self.assertIn('Участник 2 → Участник 1: 9.00 ₽', self.text(methods))
        self.assertFalse((await self.plan()).settlement_confirmed)

    async def test_kopeck_payments_and_zero_debts(self):
        await self.payment_ready(count=2, price='0.05')
        for index, value in enumerate(('0.04', '0.04', '0.02')):
            methods = await self.enter_payment(index, value)
        self.assertIn('Переводы не нужны', self.text(methods))
        await self.click('pay_confirm')
        self.assertTrue((await self.plan()).settlement_confirmed)

    async def test_cancel_payment_edit_preserves_confirmed_amounts(self):
        await self.payment_ready()
        await self.enter_payment(0, '9')
        await self.click('pay_confirm')
        before = (await self.plan()).model_dump()
        await self.click('paid', 0)
        await self.click('payments')
        self.assertEqual((await self.plan()).model_dump(), before)
        await self.message('8')
        self.assertEqual((await self.plan()).model_dump(), before)

    async def test_foreign_payment_text_cannot_mutate_copied_context(self):
        await self.payment_ready()
        await self.click('paid')
        before = await self.context().get_data()
        for user, chat in ((43, 42), (42, 43)):
            context = self.context(user, chat)
            await context.set_state(ExpenseAllocation.payment_amount)
            await context.set_data(before)
            await self.send({'message': {'message_id': next(self.ids), 'date': 1,
                'from': {'id': user, 'is_bot': False, 'first_name': 'Test'},
                'chat': {'id': chat, 'type': 'private'}, 'text': '9'}})
            self.assertEqual(await context.get_data(), before)
        self.assertEqual(await self.context().get_data(), before)

    def assert_single_edit(self, methods, message_id):
        self.assertFalse(any(isinstance(method, SendMessage) for method in methods))
        edits = [method for method in methods if isinstance(method, EditMessageText)]
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0].message_id, message_id)
        return edits[0]

    async def test_toggles_save_back_and_adjacent_dishes_edit_one_message(self):
        await self.manual(count=2)
        anchor = self.menu_message_id
        self.assert_single_edit(await self.click('begin'), anchor)
        await self.people(2)
        self.assert_single_edit(await self.click('items'), anchor)
        self.assert_single_edit(await self.click('select', 0), anchor)
        edit = self.assert_single_edit(await self.click('toggle', 0), anchor)
        self.assertTrue(edit.reply_markup.inline_keyboard[0][0].text.startswith('☑️'))
        edit = self.assert_single_edit(await self.click('toggle', 0), anchor)
        self.assertTrue(edit.reply_markup.inline_keyboard[0][0].text.startswith('⬜️'))
        await self.click('toggle', 0)
        self.assert_single_edit(await self.click('dish', 1), anchor)
        self.assertEqual((await self.plan()).assignments['i0'], ['p0'])
        edit = self.assert_single_edit(await self.click('dish', 0), anchor)
        self.assertIn('#1 из 2', edit.text)
        self.assertIn('× 1', edit.text)
        self.assertIn('0.05 ₽', edit.text)
        self.assertTrue(edit.reply_markup.inline_keyboard[0][0].text.startswith('☑️'))
        code = self.code('apply')
        from bot import allocation_handlers
        with patch.object(allocation_handlers, 'assign_item', wraps=allocation_handlers.assign_item) as assign:
            self.assert_single_edit(await self.callback(code), anchor)
            methods = await self.callback(code)
            assign.assert_called_once()
        self.assertFalse(any(isinstance(method, (SendMessage, EditMessageText)) for method in methods))
        await self.click('select', 0)
        await self.click('toggle', 0)
        self.assert_single_edit(await self.click('items'), anchor)
        self.assertEqual((await self.plan()).assignments['i0'], ['p0'])

    async def test_edit_failure_rolls_back_save_and_allows_retry(self):
        await self.manual(count=1)
        await self.click('begin')
        await self.people(1)
        await self.click('items')
        await self.click('select')
        await self.click('toggle')
        before = await self.context().get_data()
        code = self.code('apply')
        self.edit_error = TelegramBadRequest(method=EditMessageText(chat_id=42, message_id=self.menu_message_id, text='Test'),
                                             message='Bad Request: message to edit not found')
        methods = await self.callback(code)
        self.assertFalse(any(isinstance(method, SendMessage) for method in methods))
        self.assertIn('не сохранено', self.text(methods))
        self.assertEqual(await self.context().get_data(), before)
        self.assertEqual(await self.context().get_state(), ExpenseAllocation.item.state)
        self.edit_error = None
        await self.callback(code)
        self.assertEqual((await self.plan()).assignments, {'i0': ['p0']})

    async def test_not_modified_is_acknowledged_without_new_message(self):
        await self.manual(count=1)
        await self.click('begin')
        await self.people(1)
        await self.click('items')
        await self.click('select')
        self.edit_error = TelegramBadRequest(method=EditMessageText(chat_id=42, message_id=self.menu_message_id, text='Test'),
                                             message='Bad Request: message is not modified')
        methods = await self.click('toggle')
        self.assertFalse(any(isinstance(method, SendMessage) for method in methods))
        self.assertEqual((await self.context().get_data())['allocation_selection'], ['p0'])
        self.assertEqual(sum(isinstance(method, AnswerCallbackQuery) for method in methods), 1)

    async def test_explicit_screen_recovery_preserves_saved_allocation(self):
        await self.manual(count=1)
        await self.click('begin')
        await self.people(1)
        await self.click('items')
        await self.assign(0, [0])
        before = (await self.plan()).model_dump()
        old_anchor = self.menu_message_id
        old_code = self.code('select')
        methods = await self.message('/allocation')
        self.assertEqual(sum(isinstance(method, SendMessage) for method in methods), 1)
        self.assertNotEqual(self.menu_message_id, old_anchor)
        self.assertEqual((await self.plan()).model_dump(), before)
        await self.callback(old_code)
        self.assertEqual((await self.plan()).model_dump(), before)
        payload = self.callback_payload(self.code('items'))
        payload['callback_query']['message']['message_id'] = old_anchor
        snapshot = await self.context().get_data()
        methods = await self.send(payload)
        self.assertEqual(await self.context().get_data(), snapshot)
        self.assertEqual(len(methods), 1)

    async def test_long_text_paginates_inside_same_message_without_losing_names(self):
        await self.manual(count=11)
        await self.click('begin')
        await self.people(1)
        plan = await self.plan()
        data = plan.model_dump()
        for index, item in enumerate(data['items']):
            item['name'] = f'Блюдо {index+1} ' + '🧾'*190
        await self.context().update_data(allocation=data)
        anchor = self.menu_message_id
        self.assert_single_edit(await self.click('overview'), anchor)
        stored = await self.context().get_data()
        self.assertIn('Блюдо 11', stored['allocation_screen_text'])
        self.assert_single_edit(await self.click('screen_page', 1), anchor)
        self.assertEqual((await self.context().get_data())['allocation_screen_page'], 1)
        self.assert_single_edit(await self.click('items'), anchor)

    async def test_text_input_edit_failure_restores_previous_payment(self):
        await self.payment_ready()
        await self.enter_payment(0, '9')
        await self.click('pay_confirm')
        await self.click('paid', 0)
        before = await self.context().get_data()
        self.edit_error = TelegramBadRequest(method=EditMessageText(chat_id=42, message_id=self.menu_message_id, text='Test'),
                                             message='Bad Request: message cannot be edited')
        methods = await self.message('8')
        self.assertIn('не сохранено', self.text(methods))
        after = await self.context().get_data()
        self.assertGreater(after['payment_last_message_id'], before['payment_last_message_id'])
        self.assertGreater(after['payment_last_update_id'], before['payment_last_update_id'])
        for key in before.keys() - {'payment_last_message_id', 'payment_last_update_id'}:
            self.assertEqual(after[key], before[key])
        self.assertEqual(await self.context().get_state(), ExpenseAllocation.payment_amount.state)
        self.assertTrue((await self.plan()).settlement_confirmed)
        self.edit_error = None
        self.assertEqual(await self.message('8', message_id=after['payment_last_message_id']), [])
        self.assertEqual(await self.context().get_data(), after)
        self.assert_single_edit(await self.message('8'), self.menu_message_id)
        self.assertEqual((await self.plan()).payments['p0'], '8.00')

    async def test_name_input_returns_to_same_anchor(self):
        await self.manual(count=1)
        await self.click('begin')
        anchor = self.menu_message_id
        self.assert_single_edit(await self.click('add'), anchor)
        self.assert_single_edit(await self.message('Участник'), anchor)
        self.assertEqual((await self.plan()).participants[0].name, 'Участник')

    async def test_payment_plain_text_edits_same_screen_without_force_reply(self):
        await self.payment_ready()
        anchor = self.menu_message_id
        methods = await self.click('paid')
        self.assert_single_edit(methods, anchor)
        self.assertFalse(any(isinstance(getattr(method, 'reply_markup', None), ForceReply) for method in methods))
        self.assertIn('обычным текстовым', self.text(methods))
        self.assertIsNotNone((await self.context().get_data())['payment_input'])
        self.assert_single_edit(await self.message('9'), anchor)
        self.assertEqual((await self.plan()).payments['p0'], '9.00')
        self.assertEqual(await self.context().get_state(), ExpenseAllocation.payments.state)
        self.assertIsNone((await self.context().get_data())['payment_input'])

    async def test_plain_amount_outside_input_and_queued_old_update_are_ignored(self):
        await self.payment_ready()
        before = await self.context().get_data()
        await self.message('9')
        self.assertEqual(await self.context().get_data(), before)
        await self.click('paid')
        before = await self.context().get_data()
        started = before['payment_input']['started_update_id']
        self.assertEqual(await self.message('9', update_id=started-1), [])
        self.assertEqual(await self.context().get_data(), before)
        self.assert_single_edit(await self.message('9'), self.menu_message_id)
        self.assertEqual((await self.plan()).payments, {'p0': '9.00'})

    async def test_payment_text_requires_current_receipt_revision_and_payer(self):
        await self.payment_ready()
        await self.click('paid')
        before = await self.context().get_data()
        for changes in ({'allocation_token': 'other-receipt'}, {'allocation_revision': -1},
                        {'payment_person': 'p1'}, {'payment_input': None}):
            with self.subTest(changes=changes):
                await self.context().set_data({**before, **changes})
                invalid = await self.context().get_data()
                self.assertEqual(await self.message('9'), [])
                self.assertEqual(await self.context().get_data(), invalid)
        await self.context().set_data(before)
        await self.message('9')
        self.assertEqual((await self.plan()).payments, {'p0': '9.00'})

    async def test_open_and_cancel_payment_edit_failures_restore_session(self):
        await self.payment_ready()
        before = await self.context().get_data()
        old_state = await self.context().get_state()
        paid = self.code('paid')
        self.edit_error = TelegramBadRequest(
            method=EditMessageText(chat_id=42, message_id=self.menu_message_id, text='Test'),
            message='Bad Request: message cannot be edited')
        await self.callback(paid)
        self.assertEqual(await self.context().get_data(), before)
        self.assertEqual(await self.context().get_state(), old_state)
        self.edit_error = None
        await self.callback(paid)
        before = await self.context().get_data()
        cancel = self.code('payments')
        self.edit_error = TelegramBadRequest(
            method=EditMessageText(chat_id=42, message_id=self.menu_message_id, text='Test'),
            message='Bad Request: message cannot be edited')
        await self.callback(cancel)
        self.assertEqual(await self.context().get_data(), before)
        self.assertEqual(await self.context().get_state(), ExpenseAllocation.payment_amount.state)
        self.edit_error = None
        self.assert_single_edit(await self.callback(cancel), self.menu_message_id)
        cancelled = await self.context().get_data()
        await self.message('9')
        self.assertEqual(await self.context().get_data(), cancelled)

    async def test_invalid_amount_edit_failure_keeps_input_retryable(self):
        await self.payment_ready()
        await self.click('paid')
        before = await self.context().get_data()
        self.edit_error = TelegramBadRequest(
            method=EditMessageText(chat_id=42, message_id=self.menu_message_id, text='Test'),
            message='Bad Request: message cannot be edited')
        await self.message('0.001')
        after = await self.context().get_data()
        self.assertEqual(after['payment_input'], before['payment_input'])
        self.assertEqual(after['allocation'], before['allocation'])
        self.assertEqual(await self.context().get_state(), ExpenseAllocation.payment_amount.state)
        self.edit_error = None
        self.assert_single_edit(await self.message('9'), self.menu_message_id)
        self.assertEqual((await self.plan()).payments, {'p0': '9.00'})

    async def enter_tips(self, value, payer=0):
        self.assert_single_edit(await self.click('tips'), self.menu_message_id)
        methods = await self.message(value)
        self.assert_single_edit(methods, self.menu_message_id)
        if Decimal(value.replace(',', '.')) > 0:
            self.assertEqual(await self.context().get_state(), ExpenseAllocation.tip_payer.state)
            methods = await self.click('tip_who', payer)
            self.assert_single_edit(methods, self.menu_message_id)
        return methods

    async def test_tips_and_restaurant_payers_example(self):
        await self.message('/start')
        await self.callback('split_check')
        await self.callback('manual_entry')
        for index, price in enumerate(('5130', '4810', '4300')):
            if index:
                await self.callback('add_item')
            await self.message('Тестовая позиция')
            await self.message(price)
            await self.message('1')
        await self.callback('finish_receipt')
        await self.click('begin')
        for name in ('Никита', 'Андрей', 'Артём'):
            await self.click('add')
            await self.message(name)
        await self.click('items')
        for i in range(3):
            await self.assign(i,[i])
        await self.click('overview')
        await self.click('finish')
        await self.click('payments')
        await self.enter_payment(1,'14240')
        before = await self.plan()
        methods = await self.enter_tips('900',2)
        text = self.text(methods)
        self.assertIn('Никита → Андрей: 5430.00 ₽',text)
        self.assertIn('Артём → Андрей: 3700.00 ₽',text)
        self.assertIn('Общие расходы: 15140.00 ₽',text)
        self.assertIn('Ресторанные оплаты: 14240.00 ₽',text)
        self.assertIn('Чаевые оплатил(а): Артём',text)
        plan = await self.plan()
        self.assertEqual(plan.payments,{'p1':'14240.00'})
        self.assertEqual(balances(plan),{'p0':Decimal('-5430'),'p1':Decimal('9130'),'p2':Decimal('-3700')})
        for field in ('items','gross_total','prepayment','amount_due','assignments'):
            self.assertEqual(plan.model_dump()[field], before.model_dump()[field])
        self.assert_single_edit(await self.click('pay_confirm'),self.menu_message_id)
        self.assertTrue((await self.plan()).settlement_confirmed)

    async def test_tip_amount_payer_changes_and_zero_edit_the_same_screen(self):
        await self.payment_ready()
        await self.enter_payment(0,'9')
        await self.enter_tips('0,05',2)
        self.assertEqual((await self.plan()).tip_amount,'0.05')
        from bot.settlements import tip_shares
        self.assertEqual(tip_shares(await self.plan()),{'p0':Decimal('.02'),'p1':Decimal('.02'),'p2':Decimal('.01')})
        await self.click('pay_confirm')
        before = await self.plan()
        self.assert_single_edit(await self.click('tip_payer_edit'),self.menu_message_id)
        old = self.code('tip_who',1)
        self.assert_single_edit(await self.callback(old),self.menu_message_id)
        self.assertFalse((await self.plan()).settlement_confirmed)
        self.assertEqual((await self.plan()).tip_payer_id,'p1')
        self.assertEqual((await self.plan()).tip_amount,before.tip_amount)
        snapshot = await self.context().get_data()
        await self.callback(old)
        self.assertEqual(await self.context().get_data(),snapshot)
        await self.enter_tips('0.06',0)
        self.assertEqual((await self.plan()).tip_amount,'0.06')
        self.assertEqual((await self.plan()).tip_payer_id,'p0')
        await self.click('pay_confirm')
        self.assertTrue((await self.plan()).settlement_confirmed)
        await self.enter_tips('0')
        plan = await self.plan()
        self.assertEqual(plan.tip_amount,'0.00')
        self.assertIsNone(plan.tip_payer_id)
        self.assertFalse(plan.settlement_confirmed)
        self.assertEqual(plan.payments,{'p0':'9.00'})

    async def test_tip_drafts_cancel_without_changing_confirmed_amount_or_payer(self):
        await self.payment_ready()
        await self.enter_payment(0,'9')
        await self.enter_tips('1',2)
        await self.click('pay_confirm')
        before = (await self.plan()).model_dump()
        for typed in (None,'2'):
            await self.click('tips')
            if typed:
                self.assert_single_edit(await self.message(typed),self.menu_message_id)
            self.assertEqual((await self.plan()).model_dump(),before)
            self.assert_single_edit(await self.click('payments'),self.menu_message_id)
            self.assertEqual((await self.plan()).model_dump(),before)
            self.assertIsNone((await self.context().get_data())['tip_draft_amount'])
            await self.message('3')
            self.assertEqual((await self.plan()).model_dump(),before)
        await self.click('tip_payer_edit')
        await self.click('payments')
        self.assertEqual((await self.plan()).model_dump(),before)

    async def test_invalid_tip_amounts_retry_plain_text_without_force_reply(self):
        await self.payment_ready()
        await self.click('tips')
        original = (await self.plan()).model_dump()
        for value in ('-1','0.001','1e2','NaN','1000000000000000','abc'):
            methods = await self.message(value)
            edit = self.assert_single_edit(methods,self.menu_message_id)
            self.assertNotIsInstance(edit.reply_markup,ForceReply)
            self.assertIn('Нужна неотрицательная',edit.text)
            self.assertEqual(await self.context().get_state(),ExpenseAllocation.tip_amount.state)
            self.assertEqual((await self.plan()).model_dump(),original)
        self.assert_single_edit(await self.message('1,01'),self.menu_message_id)
        self.assert_single_edit(await self.click('tip_who',1),self.menu_message_id)
        self.assertEqual((await self.plan()).tip_amount,'1.01')

    async def test_tip_messages_require_owner_chat_current_session_and_fresh_ids(self):
        await self.payment_ready()
        await self.click('tips')
        before = await self.context().get_data()
        for user,chat in ((43,42),(42,43)):
            other = self.context(user,chat)
            await other.set_data(before)
            await other.set_state(ExpenseAllocation.tip_amount)
            await self.send({'message': {'message_id':next(self.ids),'date':1,
                'from':{'id':user,'is_bot':False,'first_name':'Test'},
                'chat':{'id':chat,'type':'private'},'text':'1'}})
            self.assertEqual(await other.get_data(),before)
        for changes in ({'allocation_token':'old-receipt'},{'allocation_revision':-1},
                        {'payment_person':'p0'},{'payment_input':None}):
            await self.context().set_data({**before,**changes})
            invalid = await self.context().get_data()
            await self.message('1')
            self.assertEqual(await self.context().get_data(),invalid)
        await self.context().set_data(before)
        started = before['payment_input']['started_update_id']
        self.assertEqual(await self.message('1',update_id=started-1),[])
        self.assertEqual(await self.message('1',reply_to=1),[])
        self.assertEqual(await self.context().get_data(),before)
        message_id = next(self.ids)
        await self.message('1',message_id=message_id)
        payer_state = await self.context().get_data()
        await self.message('2')  # Text cannot change the draft during payer selection.
        self.assertEqual(await self.context().get_data(),payer_state)
        await self.click('tip_who',0)
        await self.click('tips')
        fresh = await self.context().get_data()
        await self.message('1',message_id=message_id)
        self.assertEqual(await self.context().get_data(),fresh)
        await self.click('payments')
        await self.click('paid')
        restaurant = await self.context().get_data()
        await self.message('1',message_id=message_id)
        self.assertEqual(await self.context().get_data(),restaurant)

    async def test_tip_callbacks_reject_foreign_stale_forged_and_missing_drafts(self):
        await self.payment_ready()
        old = self.code('tips')
        await self.callback(old)
        await self.message('1')
        before = await self.context().get_data()
        self.assertEqual(await self.context().get_state(),ExpenseAllocation.tip_payer.state)
        await self.callback(old)
        self.assertEqual(await self.context().get_data(),before)
        who = self.code('tip_who')
        await self.callback(who.replace(':tip_who:', ':pay_confirm:'))
        self.assertEqual(await self.context().get_data(),before)
        self.assertFalse((await self.plan()).settlement_confirmed)
        for user,chat in ((43,42),(42,43)):
            await self.context(user,chat).set_data(before)
            await self.context(user,chat).set_state(ExpenseAllocation.tip_payer)
            await self.callback(who,user=user,chat=chat)
            self.assertEqual(await self.context(user,chat).get_data(),before)
        await self.callback(who.rsplit(':',1)[0]+':999')
        self.assertEqual(await self.context().get_data(),before)
        for changes in ({'tip_draft_token':'old-receipt'},{'tip_draft_amount':None},{'tip_draft_amount':'0'}):
            await self.context().set_data({**before,**changes})
            invalid = await self.context().get_data()
            await self.callback(who)
            self.assertEqual(await self.context().get_data(),invalid)
        await self.context().set_data(before)
        await asyncio.gather(self.callback(who),self.callback(who))
        self.assertEqual((await self.plan()).tip_amount,'1.00')
        self.assertEqual((await self.plan()).tip_payer_id,'p0')

    async def test_tip_edit_failures_roll_back_open_amount_payer_and_cancel(self):
        await self.payment_ready()
        await self.enter_payment(0,'9')
        await self.enter_tips('1',2)
        await self.click('pay_confirm')
        error = TelegramBadRequest(method=EditMessageText(chat_id=42,message_id=self.menu_message_id,text='Test'),
                                 message='Bad Request: message cannot be edited')
        before = await self.context().get_data()
        self.edit_error = error
        await self.click('tips')
        self.assertEqual(await self.context().get_data(),before)
        self.assertEqual(await self.context().get_state(),ExpenseAllocation.payments.state)
        self.edit_error = None
        await self.click('tips')
        before = await self.context().get_data()
        self.edit_error = error
        message_id = next(self.ids)
        await self.message('2',message_id=message_id)
        after = await self.context().get_data()
        for key in before.keys()-{'payment_last_message_id','payment_last_update_id'}:
            self.assertEqual(after[key],before[key])
        self.assertEqual(await self.context().get_state(),ExpenseAllocation.tip_amount.state)
        self.edit_error = None
        await self.message('2',message_id=message_id)
        self.assertEqual(await self.context().get_data(),after)
        await self.message('2')
        before = await self.context().get_data()
        who = self.code('tip_who',1)
        cancel = self.code('payments')
        self.edit_error = error
        for callback in (who,cancel):
            await self.callback(callback)
            self.assertEqual(await self.context().get_data(),before)
            self.assertEqual(await self.context().get_state(),ExpenseAllocation.tip_payer.state)
            self.assertTrue((await self.plan()).settlement_confirmed)
        self.edit_error = None
        self.assert_single_edit(await self.callback(who),self.menu_message_id)
        self.assertEqual((await self.plan()).tip_amount,'2.00')
        self.assertEqual((await self.plan()).tip_payer_id,'p1')
        self.assertFalse((await self.plan()).settlement_confirmed)

    async def test_zero_tips_edit_failure_restores_saved_amount_and_confirmation(self):
        await self.payment_ready()
        await self.enter_payment(0,'9')
        await self.enter_tips('1',2)
        await self.click('pay_confirm')
        await self.click('tips')
        before = (await self.plan()).model_dump()
        self.edit_error = TelegramBadRequest(method=EditMessageText(chat_id=42,message_id=self.menu_message_id,text='Test'),
                                             message='Bad Request: message cannot be edited')
        await self.message('0')
        self.assertEqual((await self.plan()).model_dump(),before)
        self.assertEqual(await self.context().get_state(),ExpenseAllocation.tip_amount.state)
        self.edit_error = None
        await self.message('0')
        self.assertEqual((await self.plan()).tip_amount,'0.00')
        self.assertIsNone((await self.plan()).tip_payer_id)

    async def test_zero_tips_refreshes_confirmed_balances_and_transfers_in_same_message(self):
        await self.payment_ready()
        await self.enter_payment(0, '9')
        await self.enter_tips('3', 2)
        methods = await self.click('pay_confirm')
        self.assertTrue((await self.plan()).settlement_confirmed)
        self.assertIn('Чаевые: 3.00 ₽', self.text(methods))
        self.assertIn('Чаевые оплатил(а): Участник 3', self.text(methods))
        self.assertEqual(balances(await self.plan()),
                         {'p0': Decimal('5'), 'p1': Decimal('-4'), 'p2': Decimal('-1')})
        anchor = self.menu_message_id
        self.assert_single_edit(await self.click('tips'), anchor)
        methods = await self.message('0')
        edit = self.assert_single_edit(methods, anchor)
        plan = await self.plan()
        self.assertEqual(plan.tip_amount, '0.00')
        self.assertIsNone(plan.tip_payer_id)
        self.assertFalse(plan.settlement_confirmed)
        self.assertEqual(plan.payments, {'p0': '9.00'})
        self.assertEqual(balances(plan),
                         {'p0': Decimal('6'), 'p1': Decimal('-3'), 'p2': Decimal('-3')})
        self.assertEqual([(t.sender_id, t.recipient_id, t.amount) for t in transfers(plan)],
                         [('p1', 'p0', '3.00'), ('p2', 'p0', '3.00')])
        self.assertIn('Чаевые: 0.00 ₽', edit.text)
        self.assertNotIn('Чаевые оплатил(а):', edit.text)
        self.assertIn('Общие расходы: 9.00 ₽', edit.text)
        self.assertIn('Баланс: 6.00 ₽', edit.text)
        self.assertEqual(edit.text.count('Баланс: -3.00 ₽'), 2)
        self.assertIn('Предварительные переводы:', edit.text)
        self.assertNotIn('Расчёт переводов подтверждён ✅', edit.text)
        self.assertIn('Участник 2 → Участник 1: 3.00 ₽', edit.text)
        self.assertIn('Участник 3 → Участник 1: 3.00 ₽', edit.text)
        self.assertEqual(await self.context().get_state(), ExpenseAllocation.payments.state)
        self.assertIsNone((await self.context().get_data())['payment_input'])
        self.assertEqual((await self.context().get_data())['allocation_screen_text'], edit.text)
        actions = {button.callback_data.split(':')[3]
                   for row in edit.reply_markup.inline_keyboard for button in row}
        self.assertIn('pay_confirm', actions)
        self.assertIn('tips', actions)
        self.assertNotIn('tip_payer_edit', actions)

    async def test_tip_payer_pages_and_reallocation_keep_saved_tips(self):
        await self.payment_ready(people=11)
        await self.click('tips')
        await self.message('0.01')
        stale = self.code('tip_who',0)
        self.assert_single_edit(await self.click('tips_page',1),self.menu_message_id)
        before = await self.context().get_data()
        await self.callback(stale)
        self.assertEqual(await self.context().get_data(),before)
        await self.click('tip_who',10)
        await self.click('split_back')
        await self.click('reopen')
        plan = await self.plan()
        self.assertEqual(plan.tip_amount,'0.01')
        self.assertEqual(plan.tip_payer_id,'p10')
        self.assertFalse(plan.settlement_confirmed)

    async def test_missing_tip_payer_and_restaurant_double_counting_block_confirmation(self):
        await self.payment_ready()
        await self.enter_payment(0,'9')
        plan = await self.plan()
        await self.context().update_data(allocation={**plan.model_dump(),'tip_amount':'1.00'})
        await self.click('split_back')
        methods = await self.click('payments')
        self.assertIn('Не выбран участник',self.text(methods))
        self.assertNotIn(' → ',self.text(methods))
        await self.click('pay_confirm')
        self.assertFalse((await self.plan()).settlement_confirmed)
        await self.click('tip_payer_edit')
        await self.click('tip_who',1)
        await self.enter_payment(0,'10')
        methods = await self.click('pay_confirm')
        self.assertIn('не подтверждён',self.text(methods))
        self.assertFalse((await self.plan()).settlement_confirmed)

    async def test_confirmed_ruble_display_survives_recognition_confirmation(self):
        for raw,currency,expected in (('Итого 100.00 ₽',None,'₽'),('Чай 100.00',None,'(валюта не определена)'),
                                      ('Total 100.00 ₽','USD','USD')):
            with self.subTest(raw=raw,currency=currency):
                await self.message('/start')
                self.recognize.return_value = RecognizedReceipt(readable=True,
                    items=[{'name':'Тестовая позиция','quantity':'1','unit_price':None,'total_price':'100.00'}],
                    receipt_total='100.00',currency=currency,raw_text=raw)
                await self.callback('split_check')
                await self.send({'message':{'message_id':next(self.ids),'date':1,
                    'from':{'id':42,'is_bot':False,'first_name':'Test'},'chat':{'id':42,'type':'private'},
                    'photo':[{'file_id':'stub','file_unique_id':'stub','width':100,'height':100}]}})
                await self.callback('confirm_recognition')
                data = await self.context().get_data()
                self.assertEqual(data['currency'],currency)
                self.assertEqual((await self.plan()).currency,currency)
                await self.click('begin')
                methods = await self.click('overview')
                self.assertIn('100.00 '+expected,self.text(methods))
