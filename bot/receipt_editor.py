"""Editing of the current recognized receipt in per-user FSM storage.

Only items change; OCR text, totals and diagnostics remain the original facts.
Callback revisions prevent old indices from editing a shifted duplicate item.
"""
import re
import secrets
from decimal import Decimal, ROUND_HALF_UP, localcontext

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot.receipt import money
from bot.currency import receipt_currency, symbol
from bot.receipt_states import ManualReceipt, ReceiptInput, ReceiptEditor
from bot.receipt_view import show_recognition, suspicious
from bot.services.receipt_models import RecognizedReceipt, ReceiptItem

router = Router(name='receipt_editor')
PAGE_SIZE = 20
INPUT_STATES = {ReceiptEditor.name.state, ReceiptEditor.quantity.state, ReceiptEditor.price.state}


def decimal_input(text, *, quantity=False):
    value = (text or '').strip().replace(',', '.')
    fraction = 6 if quantity else 2
    if not re.fullmatch(rf'[0-9]{{1,12}}(?:\.[0-9]{{1,{fraction}}})?', value):
        return None
    number = Decimal(value)
    maximum = Decimal('1000000000' if quantity else '1000000000000')
    return number if 0 < number < maximum else None


def total_price(quantity, unit_price):
    with localcontext() as context:
        context.prec = 40
        value = (Decimal(quantity) * Decimal(unit_price)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    if value >= Decimal('1000000000000'):
        raise ValueError('Item total is too large')
    return format(value, '.2f')


def changed_item(item, **changes):
    payload = item.model_dump()
    payload.update(changes)
    if 'quantity' in changes or 'unit_price' in changes:
        if payload['unit_price'] is None:
            raise ValueError('A user-supplied unit price is required')
        payload['total_price'] = total_price(payload['quantity'], payload['unit_price'])
    return ReceiptItem.model_validate(payload)


def currency(receipt):
    return symbol(receipt_currency(receipt))


def receipt_summary(receipt):
    text = f'Итого по позициям: {money(receipt.items_total)} {currency(receipt)}'.rstrip()
    if receipt.totals_mismatch:
        text += '\n⚠️ Сумма позиций не совпадает с итогом чека.'
    return text


async def current_receipt(state):
    data = await state.get_data()
    value = data.get('recognition')
    return RecognizedReceipt.model_validate(value) if value is not None else None


async def save_items(state, receipt, items):
    # model_copy intentionally preserves immutable OCR facts and original raw text.
    changed = receipt.model_copy(update={'items': items})
    await state.update_data(recognition=changed.model_dump(mode='json'))
    return changed


async def menu_revision(state):
    data = await state.get_data()
    token = data.get('receipt_edit_token') or secrets.token_hex(3)
    revision = data.get('receipt_edit_revision', 0) + 1
    await state.update_data(receipt_edit_token=token, receipt_edit_revision=revision)
    def button(label, action, index=0):
        return InlineKeyboardButton(text=label, callback_data=f're:{token}:{revision}:{index}:{action}')
    return button


async def show_items(message, state, page=0):
    receipt = await current_receipt(state)
    if receipt is None:
        return
    pages = max(1, (len(receipt.items) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(max(0, page), pages - 1)
    button = await menu_revision(state)
    keyboard = []
    for i in range(page * PAGE_SIZE, min((page + 1) * PAGE_SIZE, len(receipt.items))):
        item = receipt.items[i]
        label = ('⚠️ ' if suspicious(item) else '') + f'{i + 1}. {item.name[:40]}'
        label += (f' (гость {item.guest_id})' if item.guest_id else '')
        label += f' ×{money(Decimal(item.quantity))} — {money(Decimal(item.total_price))} {currency(receipt)}'
        keyboard.append([button(label.rstrip(), 'select', i)])
    nav = []
    if page > 0:
        nav.append(button('←', 'page', page - 1))
    if page + 1 < pages:
        nav.append(button('→', 'page', page + 1))
    if nav:
        keyboard.append(nav)
    keyboard += [[button('➕ Добавить позицию', 'add')],
                 [button('✅ Закончить редактирование', 'finish')], [button('↩️ Назад', 'review')]]
    await state.update_data(receipt_edit_page=page, pending_quantity=None)
    await state.set_state(ReceiptEditor.items)
    text = 'Выберите позицию, которую нужно исправить.' if receipt.items else 'В чеке пока нет позиций. Добавьте пропущенные.'
    if pages > 1:
        text += f'\nСтраница {page + 1}/{pages}.'
    text += '\n\n' + receipt_summary(receipt)
    if any(suspicious(item) for item in receipt.items):
        text += '\n⚠️ Отмеченные позиции требуют проверки.'
    await message.answer(text, reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard))


async def show_item(message, state, index):
    receipt = await current_receipt(state)
    if receipt is None or not 0 <= index < len(receipt.items):
        await show_items(message, state)
        return
    item = receipt.items[index]
    button = await menu_revision(state)
    unit = (f'{money(Decimal(item.unit_price))} {currency(receipt)}'.rstrip() if item.unit_price is not None else 'не указана')
    text = f'Что изменить?\n\n{item.name}\nКоличество: {money(Decimal(item.quantity))}\nЦена за единицу: {unit}\nСумма: {money(Decimal(item.total_price))} {currency(receipt)}'
    if suspicious(item):
        text += '\n⚠️ Проверьте эту позицию.'
    text += '\n\n' + receipt_summary(receipt)
    await state.update_data(receipt_edit_index=index, pending_quantity=None)
    await state.set_state(ReceiptEditor.item)
    keyboard = [[button(label, action, index)] for label, action in [
        ('📝 Название', 'name'), ('🔢 Количество', 'quantity'), ('💰 Цена', 'price'),
        ('🗑 Удалить', 'delete'), ('↩️ Назад', 'back')]]
    await message.answer(text, reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard))


async def prompt_field(message, state, index, field, prefix=''):
    prompts = {'name': 'Введите новое название позиции.', 'quantity': 'Введите новое количество.',
               'price': 'Введите цену за одну единицу.'}
    button = await menu_revision(state)
    await state.update_data(receipt_edit_index=index)
    await state.set_state(getattr(ReceiptEditor, field))
    await message.answer(prefix + prompts[field], reply_markup=InlineKeyboardMarkup(
        inline_keyboard=[[button('↩️ Назад', 'cancel_input', index)]]))


async def begin_add(message, state):
    receipt = await current_receipt(state)
    if receipt is None:
        return
    if len(receipt.items) >= 200:
        await message.answer('В чеке уже 200 позиций. Сначала удалите лишние.')
        return
    button = await menu_revision(state)
    await state.update_data(manual_target='recognition', name=None, price=None, pending_quantity=None)
    await state.set_state(ManualReceipt.name)
    await message.answer('Введите название позиции.', reply_markup=InlineKeyboardMarkup(
        inline_keyboard=[[button('↩️ Назад', 'cancel_add')]]))


@router.callback_query(F.data == 'add_recognition_item')
async def add_recognized(callback: CallbackQuery, state: FSMContext):
    if await state.get_state() != ReceiptInput.review.state or await current_receipt(state) is None:
        await callback.answer('Нет открытого распознанного чека.')
        return
    await callback.answer()
    if isinstance(callback.message, Message):
        await begin_add(callback.message, state)


@router.callback_query(F.data.startswith('re:'))
async def editor_callback(callback: CallbackQuery, state: FSMContext):
    parts = (callback.data or '').split(':')
    data = await state.get_data()
    if (len(parts) != 5 or parts[1] != data.get('receipt_edit_token')
            or parts[2] != str(data.get('receipt_edit_revision')) or not re.fullmatch(r'[0-9]{1,3}', parts[3])
            or not data.get('recognition')):
        await callback.answer('Меню устарело. Откройте редактирование ещё раз.')
        return
    index, action = int(parts[3]), parts[4]
    current = await state.get_state()
    allowed = {
        'page': {ReceiptEditor.items.state}, 'select': {ReceiptEditor.items.state},
        'add': {ReceiptEditor.items.state}, 'finish': {ReceiptEditor.items.state}, 'review': {ReceiptEditor.items.state},
        'name': {ReceiptEditor.item.state}, 'quantity': {ReceiptEditor.item.state}, 'price': {ReceiptEditor.item.state},
        'delete': {ReceiptEditor.item.state}, 'back': {ReceiptEditor.item.state},
        'yes': {ReceiptEditor.delete.state}, 'no': {ReceiptEditor.delete.state}, 'cancel_input': INPUT_STATES,
        'cancel_add': {ManualReceipt.name.state, ManualReceipt.price.state, ManualReceipt.quantity.state},
    }
    if current not in allowed.get(action, set()):
        await callback.answer('Это действие сейчас недоступно.')
        return
    await callback.answer()
    if not isinstance(callback.message, Message):
        return
    message = callback.message
    if action == 'page':
        await show_items(message, state, index)
    elif action == 'select':
        await show_item(message, state, index)
    elif action == 'add':
        await begin_add(message, state)
    elif action in {'finish', 'review'}:
        await state.update_data(pending_quantity=None)
        await state.set_state(ReceiptInput.review)
        await show_recognition(message, await current_receipt(state), edited=True)
    elif action in {'name', 'quantity', 'price'}:
        await state.update_data(pending_quantity=None)
        await prompt_field(message, state, index, action)
    elif action == 'back':
        await show_items(message, state, index // PAGE_SIZE)
    elif action == 'cancel_input' or action == 'no':
        await show_item(message, state, index)
    elif action == 'cancel_add':
        await state.update_data(manual_target=None, name=None, price=None)
        await show_items(message, state, data.get('receipt_edit_page', 0))
    else:
        receipt = await current_receipt(state)
        if not 0 <= index < len(receipt.items):
            await show_items(message, state)
            return
        if action == 'delete':
            button = await menu_revision(state)
            await state.set_state(ReceiptEditor.delete)
            await message.answer(f'Удалить позицию “{receipt.items[index].name}”?', reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[[button('🗑 Да, удалить', 'yes', index)], [button('↩️ Нет', 'no', index)]]))
        elif action == 'yes':
            items = list(receipt.items)
            del items[index]
            await save_items(state, receipt, items)
            await show_items(message, state, index // PAGE_SIZE)


@router.message(ReceiptEditor.name)
@router.message(ReceiptEditor.quantity)
@router.message(ReceiptEditor.price)
async def editor_text(message: Message, state: FSMContext):
    data = await state.get_data()
    receipt = await current_receipt(state)
    index = data.get('receipt_edit_index', -1)
    if receipt is None or not 0 <= index < len(receipt.items):
        await message.answer('Откройте список позиций заново.')
        return
    field = (await state.get_state()).rsplit(':', 1)[-1]
    item = receipt.items[index]
    try:
        if field == 'name':
            name = ' '.join((message.text or '').split())
            if not name or len(name) > 200:
                await message.answer('Введите непустое название, не длиннее 200 символов.')
                return
            changed = changed_item(item, name=name)
        else:
            value = decimal_input(message.text, quantity=field == 'quantity')
            if value is None:
                await message.answer('Введите положительное количество (до 6 знаков после запятой).' if field == 'quantity'
                                     else 'Введите цену больше 0, например 850 или 850,50 (не более двух знаков после запятой).')
                return
            if field == 'quantity' and item.unit_price is None:
                await state.update_data(pending_quantity=format(value, 'f'))
                await prompt_field(message, state, index, 'price', 'Цена за единицу не указана. Чтобы пересчитать сумму, введите её.\n')
                return
            changes = {'quantity': format(value, 'f')} if field == 'quantity' else {'unit_price': format(value, 'f')}
            if field == 'price' and data.get('pending_quantity') is not None:
                changes['quantity'] = data['pending_quantity']
            changed = changed_item(item, **changes)
    except ValueError:
        await message.answer('Сумма позиции слишком велика. Уменьшите цену или количество.')
        return
    items = list(receipt.items)
    items[index] = changed
    await save_items(state, receipt, items)
    await show_item(message, state, index)
