"""Named participants and versioned allocation menus in the existing user FSM."""
import secrets
from copy import deepcopy
from decimal import Decimal

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update

from bot.expense_split import Allocation, add_participant, assign_item, complete, participant_totals, snapshot
from bot.receipt_states import ExpenseAllocation, ReceiptInput
from bot.allocation_screen import ScreenEditError, render
from bot.settlements import (balances, confirm_settlement, expense_shares, paid_by, parse_payment,
                             set_payment, set_tips, settlement_issues, tip_shares, transfers)
from bot.payment_input import accepts_message, new_session
from bot.currency import symbol

router = Router(name='expense_allocation')
PAGE_SIZE = 10
ALLOCATION_STATES = {state.state for state in ExpenseAllocation.__all_states__}


def currency(plan):
    return symbol(plan.currency, ruble_evidence=plan.ruble_evidence) or '(валюта не определена)'


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
                            allocation_seen_messages=[], allocation_selection=[], allocation_item=None,
                            allocation_screen_id=None, allocation_screen_chat_id=None)
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
        text += '\nПосле распределения укажите фактические оплаты, включая предоплату, по каждому плательщику.'
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
        rows = [[button('💳 Оплаты и переводы', 'payments')],
                [button('✏️ Изменить распределение', 'reopen')]]
    else:
        rows = [[button('➕ Добавить участника', 'add')],
                [button('🧾 Назначить позиции', 'items')],
                [button('🔄 Предварительный итог', 'overview')],
                [button('✅ Завершить распределение', 'finish')],
                [button('↩️ К чеку', 'back')]]
    await render(message, state, overview_text(plan), InlineKeyboardMarkup(inline_keyboard=rows))


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
    await render(message, state, text, InlineKeyboardMarkup(inline_keyboard=rows))


async def show_selection(message, state, plan, index, selected):
    await state.set_state(ExpenseAllocation.item)
    await state.update_data(allocation_item=index, allocation_selection=selected)
    button = await buttons(state)
    item = plan.items[index]
    text = f'#{index+1} из {len(plan.items)}. {item.name} × {item.quantity} — {amount(item.amount, plan)}\nВыберите одного или нескольких участников.'
    if item.guest_id is not None:
        text += f'\nГость в исходном чеке: {item.guest_id}'
    chosen = ', '.join(p.name for p in plan.participants if p.id in selected) or 'никого'
    text += f'\nВыбрано: {chosen}\nПереход к соседнему блюду сохраняет выбор. «Назад без сохранения» отменяет изменения.'
    rows = [[button(('☑️ ' if p.id in selected else '⬜️ ') + p.name, 'toggle', i)]
            for i, p in enumerate(plan.participants)]
    navigation = []
    if index:
        navigation.append(button('⬅️ Предыдущее блюдо', 'dish', index-1))
    if index+1 < len(plan.items):
        navigation.append(button('Следующее блюдо ➡️', 'dish', index+1))
    if navigation:
        rows.append(navigation)
    rows += [[button('✅ Сохранить назначение', 'apply')],
             [button('🧹 Снять назначение', 'clear')],
             [button('↩️ Назад без сохранения', 'items', index//PAGE_SIZE)]]
    await render(message, state, text, InlineKeyboardMarkup(inline_keyboard=rows))


def payments_text(plan):
    restaurant = participant_totals(plan)
    tips = tip_shares(plan)
    owed = expense_shares(plan)
    net = balances(plan)
    text = 'Расчёт переводов подтверждён ✅' if plan.settlement_confirmed else 'Оплаты и предварительный расчёт'
    text += '\nБаланс = все оплаты (ресторан + чаевые) минус общая доля. Плюс — к получению, минус — к переводу.'
    for person in plan.participants:
        paid = plan.payments.get(person.id, '0.00')
        missing = ' (не указано, считается 0)' if person.id not in plan.payments else ''
        tip_paid = plan.tip_amount if person.id == plan.tip_payer_id else '0'
        text += (f'\n\n{person.name}\nРесторан: доля {amount(restaurant[person.id], plan)}; оплачено {amount(paid, plan)}{missing}'
                 f'\nЧаевые: доля {amount(tips[person.id], plan)}; оплачено {amount(tip_paid, plan)}'
                 f'\nВсего по расходам: {amount(owed[person.id], plan)}; оплачено {amount(paid_by(plan, person.id), plan)}'
                 f'\nБаланс: {amount(net[person.id], plan)}')
    text += f'\n\nРесторанный счёт (сумма позиций): {amount(plan.items_total, plan)}'
    text += f'\nРесторанные оплаты: {amount(plan.paid_total, plan)}'
    text += f'\nЧаевые: {amount(plan.tip_amount, plan)}'
    if Decimal(plan.tip_amount) > 0:
        payer = next((p.name for p in plan.participants if p.id == plan.tip_payer_id), 'не выбран')
        text += f'\nЧаевые оплатил(а): {payer}'
    text += f'\nОбщие расходы: {amount(plan.expenses_total, plan)}'
    if plan.gross_total is not None:
        text += f'\nПолная сумма чека: {amount(plan.gross_total, plan)}'
    if plan.prepayment is not None:
        text += f'\nПредоплата по чеку: {amount(plan.prepayment, plan)}'
    text += ('\nПредоплата автоматически не добавляется. Включите её ровно один раз в общую '
             'ресторанную оплату внёсшего её участника. Чаевые в ресторанные оплаты не включайте. '
             'Переводы между участниками в эти оплаты не включайте.')
    issues = settlement_issues(plan)
    if 'allocation_incomplete' in issues:
        text += '\n⚠️ Сначала завершите распределение всех позиций.'
    if 'receipt_mismatch' in issues:
        text += '\n⚠️ Сумма позиций не совпадает с общей суммой чека. Проверьте исходный чек.'
    if 'payments_mismatch' in issues:
        text += '\n⚠️ Сумма оплат не совпадает с суммой распределённых позиций. Исправьте оплаты.'
    if 'tips_unpaid' in issues:
        text += '\n⚠️ Не выбран участник, оплативший чаевые.'
    if issues:
        text += '\nРасчёт переводов не подтверждён. Окончательный список недоступен до устранения расхождений.'
    else:
        names = {person.id: person.name for person in plan.participants}
        moves = transfers(plan)
        text += '\n\n' + ('Переводы:' if plan.settlement_confirmed else 'Предварительные переводы:')
        for move in moves:
            text += f'\n{names[move.sender_id]} → {names[move.recipient_id]}: {amount(move.amount, plan)}'
        if not moves:
            text += '\nПереводы не нужны: каждый оплатил свою долю.'
        text += '\nЭто расчёт; переводы участники выполняют самостоятельно.'
    return text


async def show_payments(message, state, plan, page=0):
    page = max(0, min(page, (len(plan.participants)-1)//PAGE_SIZE))
    await state.set_state(ExpenseAllocation.payments)
    await state.update_data(payment_page=page, payment_person=None, payment_input=None,
                            tip_draft_amount=None, tip_draft_token=None)
    button = await buttons(state)
    rows = [[button('✏️ Ресторан: ' + person.name[:24] + ' — ' + plan.payments.get(person.id, '0.00'), 'paid', index)]
            for index, person in enumerate(plan.participants) if page*PAGE_SIZE <= index < (page+1)*PAGE_SIZE]
    navigation = []
    if page:
        navigation.append(button('⬅️', 'payments', page-1))
    if (page+1)*PAGE_SIZE < len(plan.participants):
        navigation.append(button('➡️', 'payments', page+1))
    if navigation:
        rows.append(navigation)
    rows.append([button('💰 Изменить чаевые' if Decimal(plan.tip_amount) else '💰 Добавить чаевые', 'tips')])
    if Decimal(plan.tip_amount) > 0:
        rows.append([button('✏️ Кто оплатил чаевые', 'tip_payer_edit')])
    if not plan.settlement_confirmed:
        rows.append([button('✅ Подтвердить расчёт переводов', 'pay_confirm')])
    rows.append([button('↩️ К распределению', 'split_back')])
    await render(message, state, payments_text(plan), InlineKeyboardMarkup(inline_keyboard=rows))


async def show_payment_input(message, state, plan, index, update_id, *, error=False):
    person = plan.participants[index]
    await state.set_state(ExpenseAllocation.payment_amount)
    await state.update_data(payment_person=person.id, payment_input=None)
    button = await buttons(state)
    text = ('Нужна неотрицательная сумма с максимум двумя знаками после запятой, без округления.\n\n'
            if error else '')
    text += (f'{person.name}: сколько всего оплачено в ресторан? Сейчас: {amount(plan.payments.get(person.id, "0"), plan)}. '
             'Отправьте сумму обычным текстовым сообщением, например 1500,50; 0 снимает оплату. '
             'Не используйте «Ответить». Включите внесённую этим участником предоплату. Чаевые сюда не включайте. '
             'Новое значение заменит прежнее и сбросит подтверждение расчёта.')
    await render(message, state, text, InlineKeyboardMarkup(
        inline_keyboard=[[button('↩️ Назад без сохранения', 'payments', index//PAGE_SIZE)]]))
    data = await state.get_data()
    await state.update_data(payment_input=new_session(data, person.id, update_id, message.message_id))


async def show_tip_input(message, state, plan, update_id, *, error=False):
    await state.set_state(ExpenseAllocation.tip_amount)
    await state.update_data(payment_person=None, payment_input=None, tip_draft_amount=None, tip_draft_token=None)
    button = await buttons(state)
    text = ('Нужна неотрицательная сумма с максимум двумя знаками после запятой, без округления.\n\n'
            if error else '')
    text += (f'Чаевые сейчас: {amount(plan.tip_amount, plan)}. Отправьте общую сумму чаевых обычным текстовым сообщением. '
             '0 удаляет чаевые. Они делятся поровну между всеми участниками; остаток копеек — по порядку добавления. '
             'Затем выберите одного участника, оплатившего чаевые. До выбора плательщика изменения не сохраняются. '
             'Ресторанные оплаты и исходный чек не меняются.')
    await render(message, state, text, InlineKeyboardMarkup(inline_keyboard=[[
        button('↩️ Назад без сохранения', 'payments')]]))
    data = await state.get_data()
    await state.update_data(payment_input=new_session(data, None, update_id, message.message_id, kind='tip'))


async def show_tip_payers(message, state, plan, draft, page=0):
    page = max(0, min(page, (len(plan.participants)-1)//PAGE_SIZE))
    data = await state.get_data()
    await state.set_state(ExpenseAllocation.tip_payer)
    await state.update_data(payment_input=None, payment_person=None, tip_draft_amount=draft,
                            tip_draft_token=data['allocation_token'])
    button = await buttons(state)
    text = (f'Кто полностью оплатил чаевые {amount(draft, plan)}? Выберите одного участника. '
            'Выбор сохранит сумму и плательщика. В ресторанные оплаты чаевые повторно не добавляйте.')
    rows = [[button(('☑️ ' if p.id == plan.tip_payer_id else '') + p.name, 'tip_who', i)]
            for i, p in enumerate(plan.participants) if page*PAGE_SIZE <= i < (page+1)*PAGE_SIZE]
    navigation = []
    if page:
        navigation.append(button('⬅️', 'tips_page', page-1))
    if (page+1)*PAGE_SIZE < len(plan.participants):
        navigation.append(button('➡️', 'tips_page', page+1))
    if navigation:
        rows.append(navigation)
    rows.append([button('↩️ Назад без сохранения', 'payments')])
    await render(message, state, text, InlineKeyboardMarkup(inline_keyboard=rows))


async def _allocation_callback(callback: CallbackQuery, state: FSMContext, event_update: Update | None = None):
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
        'dish': {ExpenseAllocation.item.state},
        'screen_page': ALLOCATION_STATES | {ReceiptInput.confirmed.state},
        'clear': {ExpenseAllocation.item.state}, 'finish': {ExpenseAllocation.overview.state},
        'reopen': {ExpenseAllocation.done.state},
        'overview': {ExpenseAllocation.overview.state, ExpenseAllocation.name.state,
                     ExpenseAllocation.items.state, ExpenseAllocation.item.state},
        'back': {ExpenseAllocation.overview.state},
        'payments': {ExpenseAllocation.done.state, ExpenseAllocation.payments.state, ExpenseAllocation.payment_amount.state,
                     ExpenseAllocation.tip_amount.state, ExpenseAllocation.tip_payer.state},
        'paid': {ExpenseAllocation.payments.state}, 'pay_confirm': {ExpenseAllocation.payments.state},
        'tips': {ExpenseAllocation.payments.state}, 'tip_payer_edit': {ExpenseAllocation.payments.state},
        'tips_page': {ExpenseAllocation.tip_payer.state}, 'tip_who': {ExpenseAllocation.tip_payer.state},
        'split_back': {ExpenseAllocation.payments.state},
    }
    if current not in allowed.get(action, set()) or index < 0:
        await callback.answer('Действие сейчас недоступно.')
        return
    if action == 'screen_page' and (not isinstance(data.get('allocation_screen_text'), str)
                                    or not isinstance(data.get('allocation_screen_markup'), dict)):
        await callback.answer('Откройте экран заново через /allocation.')
        return
    if action == 'finish' and plan.unassigned:
        await callback.answer(f'Остались нераспределённые позиции: {len(plan.unassigned)}.', show_alert=True)
        return
    if action in {'payments', 'paid', 'pay_confirm', 'tips', 'tip_payer_edit', 'tips_page', 'tip_who'} and not plan.completed:
        await callback.answer('Сначала завершите распределение позиций.')
        return
    if action in {'paid', 'tips'} and event_update is None:
        await callback.answer('Откройте ввод оплаты заново.')
        return
    if action == 'pay_confirm' and settlement_issues(plan):
        await callback.answer('Расчёт не подтверждён: проверьте ресторанные оплаты, позиции, общий итог и плательщика чаевых.', show_alert=True)
        return
    if action == 'tip_payer_edit' and Decimal(plan.tip_amount) <= 0:
        await callback.answer('Сначала укажите сумму чаевых.')
        return
    if action in {'tips_page', 'tip_who'} and (
            data.get('tip_draft_token') != data.get('allocation_token')
            or parse_payment(data.get('tip_draft_amount')) is None
            or Decimal(data['tip_draft_amount']) <= 0):
        await callback.answer('Ввод чаевых устарел. Откройте его заново.')
        return
    if action in {'items', 'select'} and not plan.participants:
        await callback.answer('Сначала добавьте хотя бы одного участника.')
        return
    if (action == 'select' and index >= len(plan.items)
            or action == 'dish' and index >= len(plan.items)
            or action in {'toggle', 'paid', 'tip_who'} and index >= len(plan.participants)):
        await callback.answer('Позиция или участник не найдены.')
        return
    if data.get('allocation_screen_id') is not None and (
            data['allocation_screen_id'] != callback.message.message_id
            or data.get('allocation_screen_chat_id') != callback.message.chat.id):
        await callback.answer('Это старый экран. Используйте текущий или /allocation.')
        return
    if data.get('allocation_screen_id') is None:
        await state.update_data(allocation_screen_id=callback.message.message_id,
                                allocation_screen_chat_id=callback.message.chat.id)
    message = callback.message
    if action in {'begin', 'overview'}:
        await show_overview(message, state, plan)
    elif action == 'add':
        if len(plan.participants) >= 30:
            await callback.answer('В одном чеке можно добавить до 30 участников.')
            return
        await state.set_state(ExpenseAllocation.name)
        button = await buttons(state)
        await render(message, state, 'Введите имя участника (до 60 символов).',
                     InlineKeyboardMarkup(inline_keyboard=[[button('↩️ Назад', 'overview')]]))
    elif action == 'items':
        await show_items(message, state, plan, index)
    elif action == 'select':
        item = plan.items[index]
        await show_selection(message, state, plan, index, list(plan.assignments.get(item.id, [])))
    elif action in {'toggle', 'apply', 'clear', 'dish'}:
        item_index = data.get('allocation_item')
        if not isinstance(item_index, int) or not 0 <= item_index < len(plan.items):
            await show_items(message, state, plan)
            return True
        selected = list(data.get('allocation_selection', []))
        if action == 'toggle':
            person = plan.participants[index].id
            selected.remove(person) if person in selected else selected.append(person)
            await show_selection(message, state, plan, item_index, selected)
        else:
            if action == 'dish' and abs(index-item_index) != 1:
                await callback.answer('Выберите соседнее блюдо.')
                return
            plan = assign_item(plan, plan.items[item_index].id, [] if action == 'clear' else selected)
            await save(state, plan)
            if action == 'dish':
                await show_selection(message, state, plan, index, list(plan.assignments.get(plan.items[index].id, [])))
            else:
                await show_items(message, state, plan, item_index//PAGE_SIZE)
    elif action == 'finish':
        plan = complete(plan)
        await save(state, plan)
        await show_overview(message, state, plan)
    elif action == 'reopen':
        plan = Allocation.model_validate({**plan.model_dump(), 'completed': False, 'settlement_confirmed': False})
        await save(state, plan)
        await show_overview(message, state, plan)
    elif action == 'back':
        await state.set_state(ReceiptInput.confirmed)
        await render(message, state, 'Чек сохранён. Распределение можно продолжить.', await entry_menu(state))
    elif action == 'payments':
        await show_payments(message, state, plan, index)
    elif action == 'paid':
        await show_payment_input(message, state, plan, index, event_update.update_id)
    elif action == 'tips':
        await show_tip_input(message, state, plan, event_update.update_id)
    elif action == 'tip_payer_edit':
        await show_tip_payers(message, state, plan, plan.tip_amount)
    elif action == 'tips_page':
        await show_tip_payers(message, state, plan, data['tip_draft_amount'], index)
    elif action == 'tip_who':
        plan = set_tips(plan, data['tip_draft_amount'], plan.participants[index].id)
        await save(state, plan)
        await show_payments(message, state, plan, data.get('payment_page', 0))
    elif action == 'pay_confirm':
        plan = confirm_settlement(plan)
        await save(state, plan)
        await show_payments(message, state, plan, data.get('payment_page', 0))
    elif action == 'split_back':
        await show_overview(message, state, plan)
    elif action == 'screen_page':
        await buttons(state)
        await render(message, state, data['allocation_screen_text'],
                     InlineKeyboardMarkup.model_validate(data['allocation_screen_markup']), index)
    return True


async def _participant_name(message: Message, state: FSMContext):
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


async def _payment_amount(message: Message, state: FSMContext, event_update: Update):
    data = await state.get_data()
    plan = Allocation.model_validate(data['allocation'])
    if (message.from_user is None or message.from_user.id != plan.owner_user_id
            or message.chat.id != plan.owner_chat_id):
        return
    person_id = data.get('payment_person')
    if not plan.completed or person_id not in {person.id for person in plan.participants}:
        await show_overview(message, state, plan)
        return
    index = next(i for i, person in enumerate(plan.participants) if person.id == person_id)
    try:
        plan = set_payment(plan, person_id, message.text or '')
    except ValueError:
        await show_payment_input(message, state, plan, index, event_update.update_id, error=True)
        return
    await save(state, plan)
    await show_payments(message, state, plan, index//PAGE_SIZE)


async def _tip_amount(message: Message, state: FSMContext, event_update: Update):
    data = await state.get_data()
    plan = Allocation.model_validate(data['allocation'])
    value = parse_payment(message.text or '')
    if value is None:
        await show_tip_input(message, state, plan, event_update.update_id, error=True)
    elif Decimal(value) == 0:
        await save(state, set_tips(plan, value, None))
        await show_payments(message, state, plan, data.get('payment_page', 0))
    else:
        await show_tip_payers(message, state, plan, value)


@router.callback_query(F.data.startswith('sp:'))
async def allocation_callback(callback: CallbackQuery, state: FSMContext, event_update: Update | None = None):
    before = deepcopy(await state.get_data())
    previous_state = await state.get_state()
    try:
        changed = await _allocation_callback(callback, state, event_update)
    except ScreenEditError:
        # Keep the old menu/revision and allocation together if Telegram rejects
        # the edit. The same callback may safely be retried; nothing was saved twice.
        await state.set_data(before)
        await state.set_state(previous_state)
        await callback.answer('Не удалось обновить экран. Изменение не сохранено. '
                              'Повторите нажатие или откройте /allocation.', show_alert=True)
        return
    if changed:
        await callback.answer()


async def text_transaction(message, state, handler, **kwargs):
    before = deepcopy(await state.get_data())
    previous_state = await state.get_state()
    try:
        await handler(message, state, **kwargs)
    except ScreenEditError:
        await state.set_data(before)
        await state.set_state(previous_state)
        await message.answer('Не удалось обновить экран. Изменение не сохранено. '
                             'Повторите ввод или откройте /allocation.')


@router.message(ExpenseAllocation.name)
async def participant_name(message: Message, state: FSMContext):
    await text_transaction(message, state, _participant_name)


@router.message(ExpenseAllocation.payment_amount)
@router.message(ExpenseAllocation.tip_amount)
async def payment_amount(message: Message, state: FSMContext, event_update: Update | None = None):
    data = await state.get_data()
    try:
        plan = Allocation.model_validate(data.get('allocation'))
    except (ValueError, TypeError):
        return
    if message.from_user is None or event_update is None or not accepts_message(
            data, plan, await state.get_state(), user_id=message.from_user.id, chat_id=message.chat.id,
            message_id=message.message_id, update_id=event_update.update_id,
            is_reply=message.reply_to_message is not None):
        return
    # Consume IDs before the UI transaction. A duplicate delivery must not apply
    # an amount later, even if Telegram rejects an edit and the business state rolls back.
    await state.update_data(payment_last_message_id=message.message_id,
                            payment_last_update_id=event_update.update_id)
    handler = _tip_amount if await state.get_state() == ExpenseAllocation.tip_amount.state else _payment_amount
    await text_transaction(message, state, handler, event_update=event_update)


async def recover_screen(message: Message, state: FSMContext):
    data = await state.get_data()
    try:
        plan = Allocation.model_validate(data.get('allocation'))
    except (ValueError, TypeError):
        await message.answer('Сначала подтвердите чек для распределения.')
        return
    if (message.from_user is None or message.from_user.id != plan.owner_user_id
            or message.chat.id != plan.owner_chat_id):
        return
    await state.update_data(allocation_screen_id=None, allocation_screen_chat_id=None,
                            allocation_item=None, allocation_selection=[], payment_person=None, payment_input=None,
                            tip_draft_amount=None, tip_draft_token=None)
    await show_overview(message, state, plan)
