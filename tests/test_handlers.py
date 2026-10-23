import unittest
from unittest.mock import AsyncMock, patch

from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import Update

from bot.handlers import router


class StartTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_sends_welcome_message(self) -> None:
        # Синтетическое обновление проходит через настоящий диспетчер без сети.
        update = Update.model_validate(
            {
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "date": 0,
                    "chat": {"id": 42, "type": "private"},
                    "from": {"id": 42, "is_bot": False, "first_name": "Друг"},
                    "text": "/start",
                    "entities": [{"type": "bot_command", "offset": 0, "length": 6}],
                },
            }
        )
        dispatcher = Dispatcher()
        dispatcher.include_router(router)
        async with Bot(token="123456:TEST_TOKEN") as bot:
            with patch.object(bot.session, "make_request", new_callable=AsyncMock) as request:
                await dispatcher.feed_update(bot, update)

        request.assert_awaited_once()
        method = request.call_args.args[1]
        self.assertIsInstance(method, SendMessage)
        self.assertEqual(method.chat_id, 42)
        self.assertEqual(
            method.text,
            "Привет! 👋\n"
            "Я помогу разделить общий чек между друзьями.\n\n"
            "Скоро здесь можно будет загрузить чек, добавить участников и автоматически "
            "посчитать, кто сколько должен.",
        )
