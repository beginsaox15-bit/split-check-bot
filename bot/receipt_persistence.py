"""Publish staged FSM changes only after Telegram and durable snapshot succeed."""
import logging
from copy import deepcopy
import secrets
from uuid import NAMESPACE_URL, uuid5

from aiogram import BaseMiddleware
from aiogram.exceptions import TelegramAPIError
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message

from bot.allocation_screen import ScreenEditError, render
from bot.receipt_states import ExpenseAllocation
from bot.receipt_store import ReceiptSnapshot, StoreError, VersionConflict, ReceiptLimit

logger = logging.getLogger(__name__)


async def compensate(message, staged, original, before, previous_state):
    """Undo a visible edit after a failed DB commit; never commit new business data."""
    attempted = await staged.get_data()
    recovery = deepcopy(before)
    # Input IDs stay consumed, even if the amount itself was rolled back.
    for key in ('payment_last_message_id', 'payment_last_update_id'):
        if key in attempted:
            recovery[key] = max(attempted[key], recovery.get(key, -1))
    await original.set_data(recovery)
    await original.set_state(previous_state)
    target = attempted.get('allocation_screen_id')
    if target is None:
        return
    if before.get('allocation_screen_id') == target and before.get('allocation_screen_text'):
        try:
            await render(message, original, before['allocation_screen_text'],
                         InlineKeyboardMarkup.model_validate(before['allocation_screen_markup']),
                         before.get('allocation_screen_page', 0))
            return
        except (ScreenEditError, TelegramAPIError):
            pass
    else:
        # A newly-created confirmation/recovery screen has no previous UI.
        try:
            await message.bot.edit_message_text(chat_id=message.chat.id, message_id=target,
                                                text='Чек не сохранён. Повторите действие.', reply_markup=None)
        except TelegramAPIError:
            pass
    # If Telegram also rejects compensation, old visible buttons must be inert.
    if 'allocation' in recovery:
        recovery.update(allocation_token=secrets.token_hex(8), allocation_revision=0,
                        allocation_screen_id=None, allocation_screen_chat_id=None,
                        payment_input=None, payment_person=None, tip_draft_amount=None,
                        tip_draft_token=None, allocation_selection=[], allocation_item=None)
        await original.set_data(recovery)
        await original.set_state(ExpenseAllocation.done if recovery['allocation']['completed']
                                 else ExpenseAllocation.overview)


class ReceiptPersistence(BaseMiddleware):
    def __init__(self, store):
        self.store = store

    async def __call__(self, handler, event, data):
        original = data.get('state')
        message = event.message if isinstance(event, CallbackQuery) else event
        if original is None or not isinstance(message, Message):
            return await handler(event, data)
        before, previous_state = deepcopy(await original.get_data()), await original.get_state()
        # Dispatcher event isolation locks the real key. All handler/UI rollbacks
        # operate on this copy, never publishing a half-applied business state.
        staged = FSMContext(storage=MemoryStorage(), key=original.key)
        await staged.set_data(deepcopy(before))
        await staged.set_state(previous_state)
        data['state'], data['receipt_store'] = staged, self.store
        try:
            result = await handler(event, data)
        except (TelegramAPIError, ScreenEditError):
            await compensate(message, staged, original, before, previous_state)
            logger.warning('Receipt UI request failed; staged changes discarded')
            try:
                await message.answer('Не удалось открыть экран. Данные не изменены. Повторите действие.')
            except TelegramAPIError:
                pass
            return None
        after = await staged.get_data()
        changed = (after.get('allocation') is not None and
                   (after.get('allocation') != before.get('allocation')
                    or after.get('saved_receipt_id') != before.get('saved_receipt_id')))
        if changed:
            try:
                snap = ReceiptSnapshot.from_fsm(after)
                # Explicit owner/chat check, independent of handler/FSM guards.
                if (snap.allocation.owner_user_id != event.from_user.id
                        or snap.allocation.owner_chat_id != message.chat.id):
                    raise StoreError('Receipt ownership mismatch')
                receipt_id = after.get('saved_receipt_id')
                if receipt_id is None:
                    update = data['event_update']
                    # The same first-confirmation update has one durable identity,
                    # even if delivery is retried after losing the FSM publication.
                    receipt_id = uuid5(NAMESPACE_URL, f'split-check-bot:{message.bot.id}:'
                                       f'{snap.allocation.owner_user_id}:{message.chat.id}:{update.update_id}').hex
                saved = await self.store.save(snap, receipt_id=receipt_id,
                                              expected_version=after.get('saved_receipt_version', 0))
            except (StoreError, ValueError, TypeError, KeyError) as error:
                await compensate(message, staged, original, before, previous_state)
                logger.warning('Receipt persistence failed: %s', type(error).__name__)
                if isinstance(error, VersionConflict):
                    text = 'Чек изменён в другой сессии. Откройте актуальную версию через «📂 Мои чеки». Изменение не сохранено.'
                elif isinstance(error, ReceiptLimit):
                    text = 'Достигнут лимит хранения (100 чеков или 256 КиБ на чек). Изменение не сохранено; история не удалялась.'
                else:
                    text = 'Хранилище временно недоступно или данные не прошли проверку. Изменение не сохранено. Повторите действие.'
                try:
                    await message.answer(text)
                except TelegramAPIError:
                    logger.warning('Receipt persistence error notice could not be delivered')
                return None
            await staged.update_data(saved_receipt_id=saved.id, saved_receipt_version=saved.version)
        await original.set_data(await staged.get_data())
        await original.set_state(await staged.get_state())
        return result


def install_persistence(router, store):
    middleware = ReceiptPersistence(store)
    router.message.outer_middleware(middleware)
    router.callback_query.outer_middleware(middleware)
    return middleware
