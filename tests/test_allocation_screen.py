import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramNetworkError
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import EditMessageText
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from bot.allocation_screen import ScreenEditError, TEXT_LIMIT, render, text_pages


class ScreenTextTests(unittest.TestCase):
    def test_long_text_and_emoji_are_preserved_with_utf16_limit(self):
        text = ('Тестовое блюдо 🧾 × 2 — 0.05 ₽\n' * 400) + 'Конец'
        pages = text_pages(text)
        self.assertEqual(''.join(pages), text)
        self.assertGreater(len(pages), 1)
        self.assertTrue(all(len(page.encode('utf-16-le'))//2 <= TEXT_LIMIT for page in pages))


class AllocationScreenTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = FSMContext(storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=42, user_id=42))
        await self.state.set_data({'allocation_token': 'testtoken', 'allocation_revision': 7,
                                   'allocation_screen_id': 55, 'allocation_screen_chat_id': 42})
        self.message = SimpleNamespace(chat=SimpleNamespace(id=42), bot=AsyncMock(),
                                       answer=AsyncMock(return_value=SimpleNamespace(message_id=99)))
        self.markup = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text='Сохранить', callback_data='sp:testtoken:6:apply:0')]])
        self.method = EditMessageText(chat_id=42, message_id=55, text='Test')

    async def test_normal_render_only_edits_the_anchor(self):
        await render(self.message, self.state, 'Тест', self.markup)
        self.message.answer.assert_not_awaited()
        self.message.bot.edit_message_text.assert_awaited_once()
        args = self.message.bot.edit_message_text.await_args.kwargs
        self.assertEqual(args['message_id'], 55)
        self.assertEqual(args['chat_id'], 42)
        self.assertIsNone(args['parse_mode'])
        self.assertEqual(args['reply_markup'].inline_keyboard[0][0].callback_data, 'sp:testtoken:7:apply:0')

    async def test_message_not_modified_is_success_without_duplicate_message(self):
        self.message.bot.edit_message_text.side_effect = TelegramBadRequest(
            method=self.method, message='Bad Request: message is not modified')
        await render(self.message, self.state, 'Тест', self.markup)
        self.assertEqual((await self.state.get_data())['allocation_screen_text'], 'Тест')
        self.message.answer.assert_not_awaited()

    async def test_edit_errors_do_not_fall_back_to_send_message_or_log_data(self):
        for error in (TelegramBadRequest, TelegramForbiddenError, TelegramNetworkError):
            self.message.bot.edit_message_text.side_effect = error(method=self.method, message='PRIVATE_USER_DATA')
            before = await self.state.get_data()
            with self.subTest(error=error), self.assertLogs('bot.allocation_screen', level='WARNING') as captured:
                with self.assertRaises(ScreenEditError):
                    await render(self.message, self.state, 'Тест', self.markup)
            self.assertNotIn('PRIVATE_USER_DATA', '\n'.join(captured.output))
            self.assertEqual(await self.state.get_data(), before)
            self.message.answer.assert_not_awaited()

    async def test_pagination_reuses_id_and_refreshes_all_action_versions(self):
        await render(self.message, self.state, '🧾'*4000, self.markup, 1)
        args = self.message.bot.edit_message_text.await_args.kwargs
        self.assertEqual(args['message_id'], 55)
        self.assertIn('Страница 2/3', args['text'])
        callbacks = [b.callback_data for row in args['reply_markup'].inline_keyboard for b in row]
        self.assertTrue(all(value.startswith('sp:testtoken:7:') for value in callbacks))
        self.assertIn('sp:testtoken:7:screen_page:0', callbacks)
        self.assertIn('sp:testtoken:7:screen_page:2', callbacks)
        self.message.answer.assert_not_awaited()

    async def test_explicit_new_anchor_and_wrong_chat_guard(self):
        await self.state.update_data(allocation_screen_id=None)
        await render(self.message, self.state, 'Новый экран', self.markup)
        self.assertEqual((await self.state.get_data())['allocation_screen_id'], 99)
        self.message.answer.assert_awaited_once()
        self.message.bot.edit_message_text.assert_not_awaited()
        await self.state.update_data(allocation_screen_chat_id=99)
        with self.assertRaises(ScreenEditError):
            await render(self.message, self.state, 'Другой чат', self.markup)
        self.message.answer.assert_awaited_once()
