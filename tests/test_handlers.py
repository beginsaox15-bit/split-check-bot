import unittest
from unittest.mock import AsyncMock, patch

from aiogram import Bot, Dispatcher
from aiogram.methods import AnswerCallbackQuery, SendMessage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.types import Update

from bot.handlers import ManualReceipt, ReceiptInput, router


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dispatcher = Dispatcher()
        cls.dispatcher.include_router(router)

    def context(self, user_id: int = 42) -> FSMContext:
        return FSMContext(
            storage=self.dispatcher.storage,
            key=StorageKey(bot_id=123456, chat_id=user_id, user_id=user_id),
        )

    async def asyncSetUp(self) -> None:
        await self.context().clear()
        await self.context(43).clear()

    def message(self, text: str | None, user_id: int = 42) -> dict:
        message = {
            "message_id": 2, "date": 1,
            "chat": {"id": user_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "first_name": "Друг"},
        }
        if text is not None:
            message["text"] = text
        return {"message": message}

    def photo(self, user_id: int = 42) -> dict:
        payload = self.message(None, user_id)
        payload["message"]["photo"] = [
            {"file_id": "small-photo", "file_unique_id": "small-unique",
             "width": 90, "height": 120, "file_size": 1000},
            {"file_id": "large-photo", "file_unique_id": "large-unique",
             "width": 960, "height": 1280, "file_size": 150000},
        ]
        return payload

    async def dispatch(self, payload: dict) -> list:
        # Обновления проходят через настоящий диспетчер и фильтры без сети.
        update = Update.model_validate({"update_id": 1, **payload})
        async with Bot(token="123456:TEST_TOKEN") as bot:
            with patch.object(bot.session, "make_request", new_callable=AsyncMock) as request:
                await self.dispatcher.feed_update(bot, update)
        return [call.args[1] for call in request.await_args_list]

    def callback(self, data: str, *, with_message: bool = True, user_id: int = 42) -> dict:
        query = {
            "id": "callback-1",
            "from": {"id": user_id, "is_bot": False, "first_name": "Друг"},
            "chat_instance": "test-chat",
            "data": data,
        }
        if with_message:
            query["message"] = {
                "message_id": 1,
                "date": 1,
                "chat": {"id": user_id, "type": "private"},
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
            ("manual_entry", "Введите название позиции."),
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

    async def begin_receipt(self, user_id: int = 42) -> None:
        await self.dispatch(self.callback("split_check", user_id=user_id))
        methods = await self.dispatch(self.callback("manual_entry", user_id=user_id))
        self.assertEqual(methods[-1].text, "Введите название позиции.")
        self.assertEqual(await self.context(user_id).get_state(), ManualReceipt.name.state)

    async def enter_item(self, name: str, price: str, quantity: str, user_id: int = 42) -> list:
        methods = await self.dispatch(self.message(name, user_id))
        self.assertEqual(methods[-1].text, "Введите цену позиции в рублях.")
        methods = await self.dispatch(self.message(price, user_id))
        self.assertEqual(methods[-1].text, "Введите количество.")
        return await self.dispatch(self.message(quantity, user_id))

    async def test_manual_receipt_add_another_and_finish(self) -> None:
        await self.begin_receipt()
        methods = await self.enter_item("Пицца", "900", "2")
        self.assertEqual(methods[-1].text,
                         "Добавлено ✅\n\nПицца × 2 — 1800 ₽\n\nТекущий чек:\n\n"
                         "1. Пицца × 2 — 1800 ₽\n\nИтого: 1800 ₽")
        self.assert_buttons(methods[-1], [
            [("➕ Добавить позицию", "add_item")],
            [("✅ Закончить ввод", "finish_receipt")],
            [("❌ Отменить", "cancel_receipt")],
        ])
        await self.dispatch(self.callback("add_item"))
        await self.enter_item("Сок", "0,10", "3")
        await self.dispatch(self.callback("add_item"))
        methods = await self.enter_item("Вода", "99.90", "1")
        self.assertIn("Итого: 1900.2 ₽", methods[-1].text)
        methods = await self.dispatch(self.callback("finish_receipt"))
        self.assertEqual(methods[-1].text,
                         "Итоговый чек:\n\n1. Пицца × 2 — 1800 ₽\n"
                         "2. Сок × 3 — 0.3 ₽\n3. Вода × 1 — 99.9 ₽\n\nИтого: 1900.2 ₽")
        self.assertIsNotNone(methods[-1].reply_markup)
        self.assertIsNone(await self.context().get_state())
        self.assertEqual(await self.context().get_data(), {})

    async def test_invalid_inputs_keep_current_step(self) -> None:
        await self.begin_receipt()
        for invalid in (None, "", "  "):
            methods = await self.dispatch(self.message(invalid))
            self.assertEqual(methods[-1].text, "Введите название позиции текстом.")
            self.assertEqual(await self.context().get_state(), ManualReceipt.name.state)
        await self.dispatch(self.message("Пицца"))
        for invalid in (None, "", "0", "-1", "abc", "NaN", "Infinity", "1e2", "1,2.3"):
            with self.subTest(price=invalid):
                methods = await self.dispatch(self.message(invalid))
                self.assertIn("Введите цену больше 0", methods[-1].text)
                self.assertEqual(await self.context().get_state(), ManualReceipt.price.state)
        await self.dispatch(self.message(" 12,50 "))
        for invalid in (None, "", "0", "-1", "1.5", "2,0", "abc"):
            with self.subTest(quantity=invalid):
                methods = await self.dispatch(self.message(invalid))
                self.assertIn("целое число больше 0", methods[-1].text)
                self.assertEqual(await self.context().get_state(), ManualReceipt.quantity.state)
                self.assertEqual((await self.context().get_data())["items"], [])
        methods = await self.dispatch(self.message(" 2 "))
        self.assertIn("Итого: 25 ₽", methods[-1].text)
        # Повторное сообщение с количеством не добавляет позицию повторно.
        self.assertEqual(await self.dispatch(self.message("2")), [])
        self.assertEqual(len((await self.context().get_data())["items"]), 1)

    async def test_cancel_clears_draft_and_items_at_every_step(self) -> None:
        for step in ("name", "price", "quantity", "review"):
            with self.subTest(step=step):
                await self.begin_receipt()
                if step != "name":
                    await self.dispatch(self.message("Пицца"))
                if step in ("quantity", "review"):
                    await self.dispatch(self.message("900"))
                if step == "review":
                    await self.dispatch(self.message("2"))
                methods = await self.dispatch(self.callback("cancel_receipt"))
                self.assertEqual(methods[-1].text, "Ввод отменён. Текущий чек очищен.")
                self.assertIsNotNone(methods[-1].reply_markup)
                self.assertEqual(await self.context().get_data(), {})
                self.assertIsNone(await self.context().get_state())

    async def test_users_have_separate_receipts(self) -> None:
        await self.begin_receipt(42)
        await self.begin_receipt(43)
        await self.enter_item("Пицца", "900", "2", 42)
        methods = await self.enter_item("Сок", "100", "1", 43)
        self.assertNotIn("Пицца", methods[-1].text)
        self.assertIn("Итого: 100 ₽", methods[-1].text)
        await self.dispatch(self.callback("cancel_receipt", user_id=42))
        self.assertEqual(await self.context(43).get_state(), ManualReceipt.review.state)
        self.assertEqual((await self.context(43).get_data())["items"][0]["name"], "Сок")

    async def test_stale_callbacks_do_not_change_receipt(self) -> None:
        for data in ("add_item", "finish_receipt"):
            methods = await self.dispatch(self.callback(data))
            self.assertEqual(len(methods), 1)
            self.assertIsInstance(methods[0], AnswerCallbackQuery)
            self.assertEqual(await self.context().get_data(), {})
        await self.begin_receipt()
        await self.dispatch(self.message("Пицца"))
        before = await self.context().get_data()
        for data in ("split_check", "manual_entry", "add_item", "finish_receipt"):
            await self.dispatch(self.callback(data))
            self.assertEqual(await self.context().get_state(), ManualReceipt.price.state)
            self.assertEqual(await self.context().get_data(), before)

    async def test_start_discards_active_receipt(self) -> None:
        await self.begin_receipt()
        await self.enter_item("Пицца", "900", "2")
        payload = self.message("/start")
        payload["message"]["entities"] = [{"type": "bot_command", "offset": 0, "length": 6}]
        methods = await self.dispatch(payload)
        self.assertIn("Что хотите сделать?", methods[-1].text)
        self.assertIsNone(await self.context().get_state())
        self.assertEqual(await self.context().get_data(), {})

    async def test_long_receipt_is_sent_in_parts_with_buttons_at_end(self) -> None:
        await self.begin_receipt()
        methods = await self.enter_item("Я" * 3000, "1", "1")
        self.assertGreater(len(methods), 1)
        self.assertTrue(all(len(method.text) <= 4000 for method in methods))
        self.assertTrue(all(method.reply_markup is None for method in methods[:-1]))
        self.assertIsNotNone(methods[-1].reply_markup)

    async def test_receipt_photo_in_split_scenario(self) -> None:
        await self.dispatch(self.callback("split_check"))
        methods = await self.dispatch(self.photo())
        # Только ответ пользователю: getFile и скачивание не вызываются.
        self.assertEqual(len(methods), 1)
        self.assertIsInstance(methods[0], SendMessage)
        self.assertEqual(methods[0].chat_id, 42)
        self.assertEqual(methods[0].text,
                         "Фото чека получено ✅\n\n"
                         "На следующем этапе я смогу распознать позиции автоматически.\n\n"
                         "Пока можете ввести позиции вручную.")
        self.assert_buttons(methods[0], [
            [("✍️ Ввести вручную", "manual_entry")],
            [("❌ Отменить", "cancel_receipt")],
        ])
        self.assertEqual(await self.context().get_state(), ReceiptInput.photo.state)
        self.assertEqual(await self.context().get_data(), {"receipt_photo": {
            "file_id": "large-photo", "file_unique_id": "large-unique",
            "width": 960, "height": 1280, "file_size": 150000,
        }})
        self.assertEqual(await self.context(43).get_data(), {})

    async def test_photo_outside_scenario_offers_to_start(self) -> None:
        methods = await self.dispatch(self.photo())
        self.assertEqual(len(methods), 1)
        self.assertIn("нажмите «🧾 Разделить чек»", methods[0].text)
        self.assert_buttons(methods[0], [
            [("🧾 Разделить чек", "split_check")],
            [("👥 Мои компании", "my_groups")],
            [("ℹ️ Как это работает", "how_it_works")],
        ])
        self.assertIsNone(await self.context().get_state())
        self.assertEqual(await self.context().get_data(), {})
        # Кнопка из подсказки действительно запускает приём фото.
        await self.dispatch(self.callback("split_check"))
        methods = await self.dispatch(self.photo())
        self.assertIn("Фото чека получено ✅", methods[-1].text)

    async def test_photo_to_manual_entry_and_finish(self) -> None:
        await self.dispatch(self.callback("split_check"))
        await self.dispatch(self.photo())
        methods = await self.dispatch(self.callback("manual_entry"))
        self.assertEqual(methods[-1].text, "Введите название позиции.")
        self.assertEqual(await self.context().get_state(), ManualReceipt.name.state)
        self.assertEqual((await self.context().get_data())["items"], [])
        methods = await self.enter_item("Пицца", "900", "2")
        self.assertIn("1. Пицца × 2 — 1800 ₽", methods[-1].text)
        methods = await self.dispatch(self.callback("finish_receipt"))
        self.assertIn("Итоговый чек:", methods[-1].text)
        self.assertIn("Итого: 1800 ₽", methods[-1].text)
        self.assertEqual(await self.context().get_data(), {})

    async def test_cancel_after_photo_clears_scenario(self) -> None:
        await self.dispatch(self.callback("split_check"))
        await self.dispatch(self.photo())
        methods = await self.dispatch(self.callback("cancel_receipt"))
        self.assertIsInstance(methods[0], AnswerCallbackQuery)
        self.assertEqual(methods[-1].text, "Ввод отменён. Текущий чек очищен.")
        self.assertIsNotNone(methods[-1].reply_markup)
        self.assertIsNone(await self.context().get_state())
        self.assertEqual(await self.context().get_data(), {})
        methods = await self.dispatch(self.photo())
        self.assertIn("нажмите «🧾 Разделить чек»", methods[-1].text)

    async def test_photo_does_not_disrupt_manual_steps_or_items(self) -> None:
        await self.begin_receipt()
        for text in ("Пицца", "900", "2", None):
            before_state = await self.context().get_state()
            before_data = await self.context().get_data()
            methods = await self.dispatch(self.photo())
            self.assertIn("Сейчас открыт ручной ввод чека.", methods[-1].text)
            self.assertEqual(await self.context().get_state(), before_state)
            self.assertEqual(await self.context().get_data(), before_data)
            if text is not None:
                await self.dispatch(self.message(text))
        self.assertEqual(len((await self.context().get_data())["items"]), 1)

    async def test_photo_scenario_belongs_to_its_user(self) -> None:
        await self.dispatch(self.callback("split_check", user_id=42))
        methods = await self.dispatch(self.photo(user_id=43))
        self.assertIn("нажмите «🧾 Разделить чек»", methods[-1].text)
        self.assertEqual(await self.context(43).get_data(), {})
        self.assertEqual(await self.context(42).get_state(), ReceiptInput.photo.state)
