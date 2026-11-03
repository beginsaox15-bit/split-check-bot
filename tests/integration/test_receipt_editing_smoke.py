"""Original local OCR and editing FSM; only Telegram transport is stubbed."""
import os
import shutil
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytesseract
from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import SendMessage
from aiogram.types import CallbackQuery, Message, User

from bot import handlers, receipt_editor
from bot.receipt_states import ReceiptInput


class ReceiptEditingSmokeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"RECEIPT_OCR_BACKEND": "tesseract"}))
        self.enterContext(patch("bot.services.openrouter_receipt_recognition._post",
                                side_effect=AssertionError("Local smoke must not call OpenRouter")))

    async def test_native_ocr_edit_delete_add_finish_and_confirm(self):
        if not shutil.which('tesseract') or not {'rus','eng'}.issubset(pytesseract.get_languages(config='')):
            self.skipTest('Tesseract rus+eng required')
        image=(Path(__file__).parents[1]/'fixtures/receipts/synthetic_restaurant_01.png').read_bytes()
        storage=MemoryStorage()
        state=FSMContext(storage=storage,key=StorageKey(bot_id=123456,chat_id=1,user_id=1))
        async with Bot(token='123456:TEST_TOKEN') as bot:
            message=Message.model_validate({'message_id':1,'date':1,'chat':{'id':1,'type':'private'},
                                            'from':{'id':1,'is_bot':False,'first_name':'Smoke'}},context={'bot':bot})
            def callback(data):
                return CallbackQuery(id='editing-smoke',from_user=User(id=1,is_bot=False,first_name='Smoke'),
                                     chat_instance='smoke',data=data,message=message).as_(bot)
            async def download(file_id,*,destination,timeout):
                destination.write(image)
            with patch.object(bot,'download',new_callable=AsyncMock,side_effect=download), \
                 patch.object(bot.session,'make_request',new_callable=AsyncMock) as request:
                async def click(action,index=None):
                    menu=next(call.args[1].reply_markup for call in reversed(request.await_args_list)
                              if isinstance(call.args[1],SendMessage) and call.args[1].reply_markup is not None)
                    data=next(b.callback_data for row in menu.inline_keyboard for b in row
                              if b.callback_data.startswith('re:') and b.callback_data.endswith(':'+action)
                              and (index is None or int(b.callback_data.split(':')[-2])==index))
                    await receipt_editor.editor_callback(callback(data),state)
                try:
                    await state.set_state(ReceiptInput.photo)
                    await state.set_data({'receipt_photo':{'file_id':'safe-editing-smoke'}})
                    await handlers.recognize_last_photo(message,state)
                    original=(await state.get_data())['recognition']
                    self.assertEqual(len(original['items']),6)
                    self.assertEqual(sum(Decimal(x['total_price']) for x in original['items']),Decimal('4500'))
                    await handlers.edit_recognition(callback('edit_recognition'),state)
                    await click('select',0)
                    await click('price')
                    await receipt_editor.editor_text(message.model_copy(update={'text':'900'}),state)
                    await click('back')
                    await click('select',1)
                    await click('delete')
                    await click('yes')
                    await click('add')
                    await handlers.item_name(message.model_copy(update={'text':'Картофель'}),state)
                    await handlers.item_price(message.model_copy(update={'text':'350'}),state)
                    await handlers.item_quantity(message.model_copy(update={'text':'2'}),state)
                    await click('finish')
                    revised=(await state.get_data())['recognition']
                    self.assertEqual(revised['raw_text'],original['raw_text'])
                    self.assertEqual(revised['receipt_total'],original['receipt_total'])
                    self.assertEqual(sum(Decimal(x['total_price']) for x in revised['items']),Decimal('4600'))
                    await handlers.confirm_recognition(callback('confirm_recognition'),state)
                    data=await state.get_data()
                    self.assertEqual(await state.get_state(),ReceiptInput.confirmed.state)
                    self.assertEqual(len(data['items']),6)
                    self.assertEqual(data['items'][-1]['name'],'Картофель')
                    self.assertNotIn('Коктейль',[x['name'] for x in data['items']])
                    self.assertEqual(data['currency'],'RUB')
                finally:
                    await storage.close()
