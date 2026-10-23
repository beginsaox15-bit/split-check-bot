import unittest
from unittest.mock import AsyncMock, patch

from aiogram import Bot, Dispatcher
from aiogram.methods import AnswerCallbackQuery, SendMessage
from aiogram.types import Update

from bot.handlers import router


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dispatcher = Dispatcher()
        cls.dispatcher.include_router(router)

    async def dispatch(self, payload: dict) -> list:
        # Обновления проходят через настоящий диспетчер и фильтры без сети.
        update = Update.model_validate({"update_id": 1, **payload})
        async with Bot(token="123456:TEST_TOKEN") as bot:
            with patch.object(bot.session, "make_request", new_callable=AsyncMock) as request:
                await self.dispatcher.feed_update(bot, update)
        return [call.args[1] for call in request.await_args_list]

    def callback(self, data: str, *, with_message: bool = True) -> dict:
        query = {
            "id": "callback-1",
            "from": {"id": 42, "is_bot": False, "first_name": "Друг"},
            "chat_instance": "test-chat",
            "data": data,
        }
        if with_message:
            query["message"] = {
                "message_id": 1,
                "date": 1,
                "chat": {"id": 42, "type": "private"},
                "text": "Меню",
            }
        else:
            query["inline_message_id"] = "inline-1"
        return {"callback_query": query}

    def assert_buttons(self, method: SendMessage, expected: list) -> None:
        self.assertIsNotNone(method.reply_markup)
        self.assertEqual(
            [[(button.text, button.callback_data) for button in row]
             for row in method.reply_markup.inline_keyboard],
            expected,
        )

    async def test_start_sends_welcome_and_menu(self) -> None:
        methods = await self.dispatch({"message": {
            "message_id": 1,
            "date": 1,
            "chat": {"id": 42, "type": "private"},
            "from": {"id": 42, "is_bot": False, "first_name": "Друг"},
            "text": "/start",
            "entities": [{"type": "bot_command", "offset": 0, "length": 6}],
        }})
        self.assertEqual(len(methods), 1)
        method = methods[0]
        self.assertIsInstance(method, SendMessage)
        self.assertEqual(method.chat_id, 42)
        self.assertEqual(
            method.text,
            "Привет! 👋\nЯ помогу разделить общий чек между друзьями.\n\nЧто хотите сделать?",
        )
        self.assert_buttons(method, [
            [("🧾 Разделить чек", "split_check")],
            [("👥 Мои компании", "my_groups")],
            [("ℹ️ Как это работает", "how_it_works")],
        ])

    async def test_callbacks_acknowledge_and_send_expected_message(self) -> None:
        cases = [
            ("split_check", "Отправьте фотографию чека или нажмите кнопку ниже, чтобы ввести позиции вручную."),
            ("my_groups", "Сохранённых компаний пока нет."),
            ("how_it_works", "1. Загружаете чек.\n2. Добавляете участников.\n"
             "3. Указываете, кто что заказывал.\n4. Бот считает, кто кому и сколько должен."),
            ("manual_entry", "Ручной ввод позиций пока недоступен."),
        ]
        for data, text in cases:
            with self.subTest(callback=data):
                methods = await self.dispatch(self.callback(data))
                self.assertEqual(len(methods), 2)
                acknowledgement, message = methods
                self.assertIsInstance(acknowledgement, AnswerCallbackQuery)
                self.assertEqual(acknowledgement.callback_query_id, "callback-1")
                self.assertIsInstance(message, SendMessage)
                self.assertEqual(message.chat_id, 42)
                self.assertEqual(message.text, text)
                if data == "split_check":
                    self.assert_buttons(message, [[("✍️ Ввести вручную", "manual_entry")]])
                else:
                    self.assertIsNone(message.reply_markup)

    async def test_callbacks_without_chat_message_are_acknowledged(self) -> None:
        for data in ("split_check", "my_groups", "how_it_works", "manual_entry"):
            with self.subTest(callback=data):
                methods = await self.dispatch(self.callback(data, with_message=False))
                self.assertEqual(len(methods), 1)
                self.assertIsInstance(methods[0], AnswerCallbackQuery)

    async def test_unknown_callback_is_ignored(self) -> None:
        self.assertEqual(await self.dispatch(self.callback("unknown")), [])
