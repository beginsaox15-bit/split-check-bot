"""Named participants and versioned allocation menus in the existing user FSM."""
import secrets
from decimal import Decimal

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot.expense_split import Allocation, add_participant, assign_item, complete, participant_totals, snapshot
from bot.receipt_states import ExpenseAllocation, ReceiptInput
from bot.receipt_view import answer_in_parts

router = Router(name='expense_allocation')
PAGE_SIZE = 10
ALLOCATION_STATES = {state.state for state in ExpenseAllocation.__all_states__}


def currency(plan):
    return '₽' if plan.currency == 'RUB' else plan.currency or '(валюта не определена)'


def amount(value, plan):
    return f'{Decimal(value):.2f} {currency(plan)}'


async def save(state, plan):
    await state.update_data(allocation=plan.model_dump(mode='json'))


async def buttons(state):
    data = await state.get_data()
    revision = data.get('allocation_revision', 0) + 1
    await state.update_data(allocation_revision=revision)
    token = data['allocation_token']
    def button(label, action, index=0):
        return InlineKeyboardButton(text=label, callback_data=f'sp:{token}:{revision}:{action}:{index}')
    return button


async def entry_menu(state):
    button = await buttons(state)
    return InlineKeyboardMarkup(inline_keyboard=[[button('👥 Распределить расходы', 'begin')]])


async def prepare(state, *, user_id, chat_id):
    plan = snapshot(await state.get_data(), user_id=user_id, chat_id=chat_id)
    await save(state, plan)
    await state.update_data(allocation_token=secrets.token_hex(4), allocation_revision=0,
                            allocation_seen_messages=[], allocation_selection=[], allocation_item=None)
    return await entry_menu(state)


def overview_text(plan):
    totals = participant_totals(plan)
    text = 'Распределение завершено ✅' if plan.completed else 'Предварительное распределение'
    text += '\n\nУчастники:'
    for index, person in enumerate(plan.participants, 1):
        text += f'\n{index}. {person.name} — {amount(totals[person.id], plan)}'
    if not plan.participants:
        text += '\nПока никого нет. Добавьте участников по одному.'
    text += f'\n\nСумма позиций: {amount(plan.items_total, plan)}'
    if plan.gross_total is not None:
        text += f'\nПолная сумма чека: {amount(plan.gross_total, plan)}'
    if plan.mismatch:
        text += '\n⚠️ Сумма позиций не совпадает с общей суммой чека. '
        text += 'Доли рассчитаны по позициям; проверьте чек. Суммы не исправлялись.'
    if plan.prepayment is not None:
        text += f'\nПредоплата по чеку: {amount(plan.prepayment, plan)}'
    if plan.amount_due is not None:
        text += f'\nОстаток к оплате по чеку: {amount(plan.amount_due, plan)}'
    if plan.prepayment is not None or plan.amount_due is not None:
        text += '\nПредоплата и остаток отдельно между участниками пока не распределяются.'
    text += f'\n\nНераспределённые позиции: {len(plan.unassigned)}'
    missing = {item.id for item in plan.unassigned}
    for index, item in enumerate(plan.items, 1):
        if item.id in missing:
            text += f'\n#{index}. {item.name} × {item.quantity} — {amount(item.amount, plan)}'
    text += '\n\nОбщие позиции делятся поровну. Остаток копеек получают первые выбранные участники в порядке добавления.'
    return text


async def show_overview(message, state, plan):
    await state.set_state(ExpenseAllocation.done if plan.completed else ExpenseAllocation.overview)
    button = await buttons(state)
    if plan.completed:
        rows = [[button('✏️ Изменить распределение', 'reopen')]]
    else:
        rows = [[button('➕ Добавить участника', 'add')],
                [button('🧾 Назначить позиции', 'items')],
                [button('🔄 Предварительный итог', 'overview')],
                [button('✅ Завершить распределение', 'finish')],
                [button('↩️ К чеку', 'back')]]
    await answer_in_parts(message, overview_text(plan), InlineKeyboardMarkup(inline_keyboard=rows))


async def show_items(message, state, plan, page=0):
    page = max(0, min(page, (len(plan.items)-1)//PAGE_SIZE))
    await state.set_state(ExpenseAllocation.items)
    await state.update_data(allocation_page=page, allocation_item=None, allocation_selection=[])
    button = await buttons(state)
    text = f'Назначьте участников каждой позиции. Страница {page+1}.'
    rows = []
    for index in range(page*PAGE_SIZE, min((page+1)*PAGE_SIZE, len(plan.items))):
        item = plan.items[index]
        owners = [p.name for p in plan.participants if p.id in plan.assignments.get(item.id, [])]
        text += f'\n\n#{index+1}. {item.name} × {item.quantity} — {amount(item.amount, plan)}'
        if item.guest_id is not None:
            text += f'\nГость в исходном чеке: {item.guest_id}'
        text += '\nУчастники: ' + (', '.join(owners) or 'не назначены')
        rows.append([button(f'#{index+1} {item.name[:35]}', 'select', index)])
    navigation = []
    if page:
        navigation.append(button('⬅️', 'items', page-1))
    if (page+1)*PAGE_SIZE < len(plan.items):
        navigation.append(button('➡️', 'items', page+1))
    if navigation:
        rows.append(navigation)
    rows.append([button('↩️ Предварительный итог', 'overview')])
    await answer_in_parts(message, text, InlineKeyboardMarkup(inline_keyboard=rows))


async def show_selection(message, state, plan, index, selected):
    await state.set_state(ExpenseAllocation.item)
    await state.update_data(allocation_item=index, allocation_selection=selected)
    button = await buttons(state)
    item = plan.items[index]
    text = f'#{index+1}. {item.name} — {amount(item.amount, plan)}\nВыберите одного или нескольких участников.'
    rows = [[button(('☑️ ' if p.id in selected else '⬜️ ') + p.name, 'toggle', i)]
            for i, p in enumerate(plan.participants)]
    rows += [[button('✅ Сохранить назначение', 'apply')],
             [button('🧹 Снять назначение', 'clear')],
             [button('↩️ Назад без сохранения', 'items', index//PAGE_SIZE)]]
    await answer_in_parts(message, text, InlineKeyboardMarkup(inline_keyboard=rows))


@router.callback_query(F.data.startswith('sp:'))
async def allocation_callback(callback: CallbackQuery, state: FSMContext):
    parts = (callback.data or '').split(':')
    data = await state.get_data()
    current = await state.get_state()
    if (len(parts) != 5 or not isinstance(callback.message, Message)
            or parts[1] != data.get('allocation_token')
            or parts[2] != str(data.get('allocation_revision'))
            or current not in ALLOCATION_STATES | {ReceiptInput.confirmed.state}):
        await callback.answer('Меню устарело. Откройте распределение заново.')
        return
    try:
        plan = Allocation.model_validate(data.get('allocation'))
        index = int(parts[4])
    except (ValueError, TypeError):
        await callback.answer('Не удалось открыть распределение. Начните с /start.')
        return
    if (plan.owner_user_id != callback.from_user.id
            or plan.owner_chat_id != callback.message.chat.id):
        await callback.answer('Это меню другого участника или чата.')
        return
    action = parts[3]
    allowed = {
        'begin': {ReceiptInput.confirmed.state},
        'add': {ExpenseAllocation.overview.state},
        'items': {ExpenseAllocation.overview.state, ExpenseAllocation.items.state, ExpenseAllocation.item.state},
        'select': {ExpenseAllocation.items.state},
        'toggle': {ExpenseAllocation.item.state}, 'apply': {ExpenseAllocation.item.state},
        'clear': {ExpenseAllocation.item.state}, 'finish': {ExpenseAllocation.overview.state},
        'reopen': {ExpenseAllocation.done.state},
        'overview': ALLOCATION_STATES - {ExpenseAllocation.done.state},
        'back': {ExpenseAllocation.overview.state},
    }
    if current not in allowed.get(action, set()) or index < 0:
        await callback.answer('Действие сейчас недоступно.')
        return
    if action == 'finish' and plan.unassigned:
        await callback.answer(f'Остались нераспределённые позиции: {len(plan.unassigned)}.', show_alert=True)
        return
    if action in {'items', 'select'} and not plan.participants:
        await callback.answer('Сначала добавьте хотя бы одного участника.')
        return
    if action == 'select' and index >= len(plan.items) or action == 'toggle' and index >= len(plan.participants):
        await callback.answer('Позиция или участник не найдены.')
        return
    await callback.answer()
    message = callback.message
    if action in {'begin', 'overview'}:
        await show_overview(message, state, plan)
    elif action == 'add':
        if len(plan.participants) >= 30:
            await message.answer('В одном чеке можно добавить до 30 участников.')
            return
        await state.set_state(ExpenseAllocation.name)
        button = await buttons(state)
        await message.answer('Введите имя участника (до 60 символов).',
                             reply_markup=InlineKeyboardMarkup(inline_keyboard=[[button('↩️ Назад', 'overview')]]))
    elif action == 'items':
        await show_items(message, state, plan, index)
    elif action == 'select':
        item = plan.items[index]
        await show_selection(message, state, plan, index, list(plan.assignments.get(item.id, [])))
    elif action in {'toggle', 'apply', 'clear'}:
        item_index = data.get('allocation_item')
        if not isinstance(item_index, int) or not 0 <= item_index < len(plan.items):
            await show_items(message, state, plan)
            return
        selected = list(data.get('allocation_selection', []))
        if action == 'toggle':
            person = plan.participants[index].id
            selected.remove(person) if person in selected else selected.append(person)
            await show_selection(message, state, plan, item_index, selected)
        else:
            plan = assign_item(plan, plan.items[item_index].id, [] if action == 'clear' else selected)
            await save(state, plan)
            await show_items(message, state, plan, item_index//PAGE_SIZE)
    elif action == 'finish':
        plan = complete(plan)
        await save(state, plan)
        await show_overview(message, state, plan)
    elif action == 'reopen':
        plan = Allocation.model_validate({**plan.model_dump(), 'completed': False})
        await save(state, plan)
        await show_overview(message, state, plan)
    elif action == 'back':
        await state.set_state(ReceiptInput.confirmed)
        await message.answer('Чек сохранён. Распределение можно продолжить.', reply_markup=await entry_menu(state))


@router.message(ExpenseAllocation.name)
async def participant_name(message: Message, state: FSMContext):
    data = await state.get_data()
    plan = Allocation.model_validate(data['allocation'])
    if (message.from_user is None or message.from_user.id != plan.owner_user_id
            or message.chat.id != plan.owner_chat_id):
        return
    if message.message_id in data.get('allocation_seen_messages', []):
        return
    try:
        plan = add_participant(plan, message.text or '')
    except ValueError:
        await message.answer('Введите непустое уникальное имя до 60 символов, без переносов строк (не более 30 участников).')
        return
    await save(state, plan)
    await state.update_data(allocation_seen_messages=[*data.get('allocation_seen_messages', []), message.message_id])
    await show_overview(message, state, plan)
