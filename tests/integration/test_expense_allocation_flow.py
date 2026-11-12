"""Real dispatcher/FSM integration; Telegram transport and OCR are stubs."""
import asyncio
import itertools
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from aiogram import Bot, Dispatcher
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import SimpleEventIsolation
from aiogram.methods import AnswerCallbackQuery, SendMessage
from aiogram.types import InlineKeyboardMarkup, Message, Update

from bot.handlers import router
from bot.expense_split import Allocation, participant_totals
from bot.receipt_states import ExpenseAllocation, ManualReceipt, ReceiptInput
from bot.services.receipt_models import RecognizedReceipt
from bot.settlements import balances


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
            async def response(current_bot, method, **kwargs):
                if isinstance(method, SendMessage):
                    return Message.model_validate({'message_id': next(self.ids), 'date': 1,
                        'chat': {'id': method.chat_id, 'type': 'private'}, 'text': method.text,
                        'from': {'id': 123456, 'is_bot': True, 'first_name': 'Bot'}}, context={'bot': current_bot})
                return True
            with patch.object(bot.session, 'make_request', new_callable=AsyncMock, side_effect=response) as request:
                await self.dispatcher.feed_update(bot, update)
        methods = [call.args[1] for call in request.await_args_list]
        menus = [m for m in methods if isinstance(m, SendMessage) and isinstance(m.reply_markup, InlineKeyboardMarkup)]
        if menus:
            self.menu = menus[-1].reply_markup
        return methods

    def callback_payload(self, data, user=42, chat=42, with_message=True):
        query = {'id': str(next(self.ids)), 'chat_instance': 'test', 'data': data,
                 'from': {'id': user, 'is_bot': False, 'first_name': 'Test'}}
        if with_message:
            query['message'] = {'message_id': 1, 'date': 1, 'chat': {'id': chat, 'type': 'private'}, 'text': 'Menu'}
        else:
            query['inline_message_id'] = 'test'
        return {'callback_query': query}

    async def callback(self, data, **kwargs):
        return await self.send(self.callback_payload(data, **kwargs))

    async def message(self, text, message_id=None, reply_to=None):
        payload = {'message': {'message_id': message_id or next(self.ids), 'date': 1,
                                           'from': {'id': 42, 'is_bot': False, 'first_name': 'Test'},
                                           'chat': {'id': 42, 'type': 'private'}, 'text': text}}
        if reply_to is not None:
            payload['message']['reply_to_message'] = {'message_id': reply_to, 'date': 1,
                                                     'chat': {'id': 42, 'type': 'private'}, 'text': 'Payment prompt'}
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
        self.assertIn('нераспределённые', methods[-1].text)
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
        self.assertIn('Распределение завершено', methods[-1].text)
        self.assertIn('Нераспределённые позиции: 0', methods[-1].text)
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
        text = methods[-1].text
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
            self.assertIn('другого', methods[-1].text)
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
        self.assertIn('нераспределённые', methods[-1].text)
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
        self.assertIn('Нераспределённые позиции: 10', methods[-1].text)

    async def test_sub_kopeck_manual_input_is_not_rounded_silently(self):
        methods = await self.manual(count=1, price='0.005')
        self.assertIn('целых копейках', methods[-1].text)
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
        self.assertIn('распределение расходов', methods[-1].text)
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
        prompt = (await self.context().get_data())['payment_prompt_id']
        return await self.message(value, reply_to=prompt)

    async def test_one_payer_and_confirmed_transfers(self):
        methods = await self.payment_ready()
        self.assertIn('не указано', methods[-1].text)
        methods = await self.click('pay_confirm')
        self.assertIn('не подтверждён', methods[-1].text)
        methods = await self.enter_payment(0, '9,00')
        self.assertIn('Предварительные переводы:', methods[-1].text)
        self.assertIn('Участник 2 → Участник 1: 3.00 ₽', methods[-1].text)
        old = self.code('pay_confirm')
        methods = await self.callback(old)
        self.assertIn('Расчёт переводов подтверждён ✅', methods[-1].text)
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
        self.assertIn('автоматически не добавляется', methods[-1].text)
        self.assertIn('Сумма оплат не совпадает', methods[-1].text)
        self.assertNotIn(' → ', methods[-1].text)
        methods = await self.enter_payment(0, '4')
        self.assertEqual((await self.plan()).paid_total, Decimal('10'))
        self.assertIn('Предварительные переводы', methods[-1].text)
        await self.click('pay_confirm')
        self.assertTrue((await self.plan()).settlement_confirmed)

    async def test_payment_edit_replaces_value_and_invalidates_confirmation(self):
        await self.payment_ready()
        await self.enter_payment(0, '9')
        await self.click('pay_confirm')
        methods = await self.enter_payment(0, '8.99')
        self.assertEqual((await self.plan()).payments, {'p0': '8.99'})
        self.assertFalse((await self.plan()).settlement_confirmed)
        self.assertIn('Сумма оплат не совпадает', methods[-1].text)
        self.assertNotIn(' → ', methods[-1].text)
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
        self.assertIn('Сумма позиций не совпадает', methods[-1].text)
        self.assertNotIn(' → ', methods[-1].text)
        await self.click('pay_confirm')
        self.assertFalse((await self.plan()).settlement_confirmed)

    async def test_old_replies_duplicates_and_invalid_payment_input(self):
        await self.payment_ready()
        await self.click('paid')
        old_prompt = (await self.context().get_data())['payment_prompt_id']
        await self.message('0.001', reply_to=old_prompt)
        self.assertEqual((await self.plan()).payments, {})
        new_prompt = (await self.context().get_data())['payment_prompt_id']
        self.assertNotEqual(old_prompt, new_prompt)
        await self.message('9', reply_to=old_prompt)
        await self.message('9')
        self.assertEqual((await self.plan()).payments, {})
        await self.message('9', message_id=999, reply_to=new_prompt)
        await self.click('paid', 1)
        await self.message('9', message_id=999, reply_to=new_prompt)
        self.assertEqual((await self.plan()).payments, {'p0': '9.00'})
        await self.click('payments')
        await self.click('paid', 1)
        prompt = (await self.context().get_data())['payment_prompt_id']
        await asyncio.gather(self.message('0', message_id=1001, reply_to=prompt),
                             self.message('0', message_id=1001, reply_to=prompt))
        self.assertEqual((await self.plan()).payments, {'p0': '9.00', 'p1': '0.00'})

    async def test_payment_callbacks_are_scoped_and_forged_indices_rejected(self):
        await self.payment_ready()
        valid = self.code('paid')
        before = await self.context().get_data()
        for user, chat in ((43, 42), (42, 43)):
            await self.context(user, chat).set_state(ExpenseAllocation.payments)
            await self.context(user, chat).set_data(before)
            methods = await self.callback(valid, user=user, chat=chat)
            self.assertIn('другого', methods[-1].text)
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
        self.assertIn('Участник 2 → Участник 1: 9.00 ₽', methods[-1].text)
        self.assertFalse((await self.plan()).settlement_confirmed)

    async def test_kopeck_payments_and_zero_debts(self):
        await self.payment_ready(count=2, price='0.05')
        for index, value in enumerate(('0.04', '0.04', '0.02')):
            methods = await self.enter_payment(index, value)
        self.assertIn('Переводы не нужны', methods[-1].text)
        await self.click('pay_confirm')
        self.assertTrue((await self.plan()).settlement_confirmed)

    async def test_cancel_payment_edit_preserves_confirmed_amounts(self):
        await self.payment_ready()
        await self.enter_payment(0, '9')
        await self.click('pay_confirm')
        before = (await self.plan()).model_dump()
        await self.click('paid', 0)
        prompt = (await self.context().get_data())['payment_prompt_id']
        await self.click('payments')
        self.assertEqual((await self.plan()).model_dump(), before)
        await self.message('8', reply_to=prompt)
        self.assertEqual((await self.plan()).model_dump(), before)

    async def test_foreign_payment_reply_cannot_mutate_copied_context(self):
        await self.payment_ready()
        await self.click('paid')
        before = await self.context().get_data()
        for user, chat in ((43, 42), (42, 43)):
            context = self.context(user, chat)
            await context.set_state(ExpenseAllocation.payment_amount)
            await context.set_data(before)
            await self.send({'message': {'message_id': next(self.ids), 'date': 1,
                'from': {'id': user, 'is_bot': False, 'first_name': 'Test'},
                'chat': {'id': chat, 'type': 'private'}, 'text': '9',
                'reply_to_message': {'message_id': before['payment_prompt_id'], 'date': 1,
                                     'chat': {'id': chat, 'type': 'private'}, 'text': 'Prompt'}}})
            self.assertEqual(await context.get_data(), before)
        self.assertEqual(await self.context().get_data(), before)
