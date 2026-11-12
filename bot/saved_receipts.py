"""Owner/chat-scoped history. Only business snapshots survive a restart."""
import secrets
from copy import deepcopy

from aiogram import F, Router
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot import allocation_handlers
from bot.allocation_screen import render
from bot.receipt_store import StoreError, ReceiptNotFound, InvalidRecord

router = Router(name='saved_receipts')
PAGE_SIZE = 8


def history_button(text, data, action, value):
    return InlineKeyboardButton(text=text, callback_data=f"rc:{data['history_token']}:{data['history_revision']}:{action}:{value}")


async def show_history(callback, state, store, page=0):
    records = await store.list(user_id=callback.from_user.id, chat_id=callback.message.chat.id,
                               offset=page*PAGE_SIZE, limit=PAGE_SIZE+1)
    data = await state.get_data()
    revision = data.get('history_revision', 0) + 1
    await state.update_data(history_revision=revision)
    data = await state.get_data()
    text = ('📂 Мои чеки\nДанные хранятся локально на сервере бота, не на телефоне. '
            'Показаны чеки из этого чата. Открытие создаёт новую безопасную сессию.')
    rows = []
    for record in records[:PAGE_SIZE]:
        if record is None:
            text += '\n⚠️ Повреждённая запись недоступна; остальные чеки можно открыть.'
            continue
        plan = record.snapshot.allocation
        status = 'завершённый расчёт' if record.status == 'completed' else 'черновик'
        label = (f'{record.created_at[:10]} · {allocation_handlers.amount(plan.expenses_total, plan)} · '
                 f'{len(plan.participants)} уч. · {status}')
        rows.append([history_button(label, data, 'open', record.id)])
    if not records:
        text += '\nСохранённых чеков пока нет.'
    nav = []
    if page:
        nav.append(history_button('⬅️', data, 'page', page-1))
    if len(records) > PAGE_SIZE:
        nav.append(history_button('➡️', data, 'page', page+1))
    if nav:
        rows.append(nav)
    rows.append([history_button('↩️ Главное меню', data, 'home', 0)])
    await render(callback.message, state, text, InlineKeyboardMarkup(inline_keyboard=rows))


@router.callback_query(F.data == 'my_receipts')
async def my_receipts(callback: CallbackQuery, state, receipt_store=None):
    if not isinstance(callback.message, Message):
        await callback.answer('Откройте меню в чате.')
        return
    if receipt_store is None:
        await callback.answer('Хранилище недоступно.', show_alert=True)
        return
    # Drop volatile inputs and tokens, preserving saved snapshots in the DB.
    # The persistence middleware publishes this only if the UI succeeds.
    before, previous_state = deepcopy(await state.get_data()), await state.get_state()
    await state.clear()
    await state.update_data(history_owner=callback.from_user.id, history_chat=callback.message.chat.id,
                            history_token=secrets.token_hex(4), history_revision=0,
                            history_message_id=callback.message.message_id,
                            allocation_token=secrets.token_hex(8), allocation_revision=0,
                            allocation_screen_id=callback.message.message_id,
                            allocation_screen_chat_id=callback.message.chat.id)
    try:
        await show_history(callback, state, receipt_store)
    except StoreError:
        # The query failed before rendering: keep the active calculation intact.
        await state.set_data(before)
        await state.set_state(previous_state)
        await callback.answer('Хранилище временно недоступно. Попробуйте позже.', show_alert=True)
        return
    await callback.answer()


@router.callback_query(F.data.startswith('rc:'))
async def receipt_callback(callback: CallbackQuery, state, receipt_store=None):
    data = await state.get_data()
    parts = (callback.data or '').split(':')
    if (receipt_store is None or not isinstance(callback.message, Message) or len(parts) != 5
            or parts[1] != data.get('history_token') or parts[2] != str(data.get('history_revision'))
            or callback.from_user.id != data.get('history_owner')
            or callback.message.chat.id != data.get('history_chat')
            or callback.message.message_id != data.get('history_message_id')):
        await callback.answer('Это чужое или устаревшее меню. Откройте «📂 Мои чеки» заново.')
        return
    action, value = parts[3:]
    try:
        if action == 'page':
            if not value.isdecimal() or not 0 <= int(value) < 13:
                await callback.answer('Недоступная страница.')
                return
            await show_history(callback, state, receipt_store, int(value))
        elif action == 'open':
            record = await receipt_store.get(value, user_id=callback.from_user.id, chat_id=callback.message.chat.id)
            # No old token, revision, selection, message ID or input session is read.
            await state.set_data({**record.snapshot.fsm_data(), 'saved_receipt_id': record.id,
                                  'saved_receipt_version': record.version,
                                  'allocation_token': secrets.token_hex(8), 'allocation_revision': 0,
                                  'allocation_screen_id': None, 'allocation_screen_chat_id': None})
            plan = record.snapshot.allocation
            if plan.completed:
                await allocation_handlers.show_payments(callback.message, state, plan)
            else:
                await allocation_handlers.show_overview(callback.message, state, plan)
        elif action == 'home':
            from bot.handlers import START_MENU, START_TEXT
            await render(callback.message, state, START_TEXT, START_MENU)
            await state.clear()
        else:
            await callback.answer('Недоступное действие.')
            return
    except (ReceiptNotFound, InvalidRecord):
        await callback.answer('Чек недоступен или повреждён.', show_alert=True)
        return
    except StoreError:
        await callback.answer('Хранилище временно недоступно. Попробуйте позже.', show_alert=True)
        return
    await callback.answer()
