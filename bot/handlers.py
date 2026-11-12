import logging
from io import BytesIO

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot.receipt import parse_price, parse_quantity, receipt_lines
from bot.receipt_states import ManualReceipt, ReceiptInput, ReceiptEditor
from bot.receipt_view import RECOGNITION_MENU, answer_in_parts, show_recognition
from bot import receipt_editor, allocation_handlers, saved_receipts
from bot.allocation_screen import render
from bot.currency import receipt_currency
from bot.services.receipt_models import (
    LowQualityReceipt, RecognizedReceipt, ReceiptItem, TesseractLanguagesUnavailable,
    TesseractNotInstalled, UnparsedReceipt, UnreadableReceipt,
)
from bot.services.receipt_recognition import recognize_receipt


router = Router()
logger = logging.getLogger(__name__)


START_TEXT = (
    "Привет! 👋\n"
    "Я помогу разделить общий чек между друзьями.\n\n"
    "Что хотите сделать?"
)

START_MENU = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="🧾 Разделить чек", callback_data="split_check")],
    [InlineKeyboardButton(text="📂 Мои чеки", callback_data="my_receipts")],
    [InlineKeyboardButton(text="👥 Мои компании", callback_data="my_groups")],
    [InlineKeyboardButton(text="ℹ️ Как это работает", callback_data="how_it_works")],
])

MANUAL_MENU = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="✍️ Ввести вручную", callback_data="manual_entry")],
])

RECOGNITION_RETRY_MENU = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="📷 Попробовать ещё раз", callback_data="retry_photo")],
    [InlineKeyboardButton(text="✍️ Ввести вручную", callback_data="manual_entry")],
    [InlineKeyboardButton(text="❌ Отменить", callback_data="cancel_receipt")],
])

UNREADABLE_TEXT = (
    "Чек удалось прочитать не полностью 😕\n"
    "Попробуйте сфотографировать его ближе, ровнее и без теней."
)

RAW_TEXT_MENU = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="✍️ Ввести вручную", callback_data="manual_entry")],
    [InlineKeyboardButton(text="🔄 Распознать заново", callback_data="repeat_recognition")],
    [InlineKeyboardButton(text="📄 Показать распознанный текст", callback_data="show_recognized_text")],
    [InlineKeyboardButton(text="❌ Отменить", callback_data="cancel_receipt")],
])

RECEIPT_MENU = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="➕ Добавить позицию", callback_data="add_item")],
    [InlineKeyboardButton(text="✅ Закончить ввод", callback_data="finish_receipt")],
    [InlineKeyboardButton(text="❌ Отменить", callback_data="cancel_receipt")],
])


async def show_receipt(message: Message, items: list[dict], *, finished: bool = False,
                       finished_menu: InlineKeyboardMarkup | None = None, state: FSMContext | None = None) -> None:
    lines, total = receipt_lines(items)
    heading = "Итоговый чек:" if finished else f"Добавлено ✅\n\n{lines[-1]}\n\nТекущий чек:"
    text = heading + "\n\n" + "\n".join(
        f"{index}. {line}" for index, line in enumerate(lines, 1)
    ) + f"\n\nИтого: {total} ₽"
    if finished and state is not None:
        await render(message, state, text, finished_menu or START_MENU)
    else:
        await answer_in_parts(message, text, (finished_menu or START_MENU) if finished else RECEIPT_MENU)


async def recognize_last_photo(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    photo = data.get("receipt_photo")
    if not photo:
        await message.answer("Отправьте фотографию чека ещё раз.", reply_markup=RECOGNITION_RETRY_MENU)
        return
    # Повторное распознавание инвалидирует предыдущий неподтверждённый результат.
    await state.set_data({"receipt_photo": photo})
    await state.set_state(ReceiptInput.photo)
    await message.answer("Чек получен 🧾\nАнализирую позиции…")

    async def on_fallback() -> None:
        await message.answer("Основное распознавание временно недоступно, пробую резервный способ…")
    try:
        with BytesIO() as image:
            await message.bot.download(photo["file_id"], destination=image, timeout=30)
            result = await recognize_receipt(image.getvalue(), on_fallback=on_fallback)
    except (TesseractNotInstalled, TesseractLanguagesUnavailable) as error:
        logger.exception("Local OCR setup missing: %s", type(error).__name__)
        await message.answer(
            "Локальное распознавание пока не настроено: недоступен Tesseract или языки русского и английского. "
            "Пока можете ввести позиции вручную.",
            reply_markup=RECOGNITION_RETRY_MENU,
        )
        return
    except LowQualityReceipt as error:
        await state.update_data(raw_text=error.raw_text)
        await message.answer(UNREADABLE_TEXT, reply_markup=RAW_TEXT_MENU)
        return
    except UnparsedReceipt as error:
        await state.update_data(raw_text=error.raw_text)
        await answer_in_parts(
            message,
            "Текст распознан, но выделить позиции автоматически не удалось. "
            "Введите позиции вручную или попробуйте распознать заново.\n\n"
            "Распознанный текст:\n" + error.raw_text,
            RAW_TEXT_MENU,
        )
        return
    except UnreadableReceipt:
        logger.info("Receipt recognition returned no reliable items")
        await message.answer(UNREADABLE_TEXT, reply_markup=RECOGNITION_RETRY_MENU)
        return
    except Exception:
        # Developer log сохраняет причину и traceback; Telegram получает общий текст.
        logger.exception("Receipt photo processing failed")
        await message.answer(
            "Сервис распознавания временно недоступен. Попробуйте позже или введите позиции вручную.",
            reply_markup=RECOGNITION_RETRY_MENU,
        )
        return
    await state.update_data(recognition=result.model_dump(mode="json"))
    await state.set_state(ReceiptInput.review)
    await show_recognition(message, result)


@router.message(CommandStart())
async def start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(START_TEXT, reply_markup=START_MENU)


@router.message(Command('allocation'))
async def recover_allocation(message: Message, state: FSMContext) -> None:
    await allocation_handlers.recover_screen(message, state)


@router.callback_query(F.data == "split_check")
async def split_check(callback: CallbackQuery, state: FSMContext) -> None:
    current_state = await state.get_state()
    if current_state not in (None, ReceiptInput.photo.state, ReceiptInput.confirmed.state):
        await callback.answer("Чек уже открыт. Продолжите ввод или отмените его.")
        return
    await callback.answer()
    if isinstance(callback.message, Message):
        if current_state == ReceiptInput.confirmed.state:
            await state.clear()
        await state.set_state(ReceiptInput.photo)
        await callback.message.answer(
            "Отправьте фотографию чека или нажмите кнопку ниже, чтобы ввести позиции вручную.",
            reply_markup=MANUAL_MENU,
        )


@router.callback_query(F.data == "my_groups")
async def my_groups(callback: CallbackQuery) -> None:
    await callback.answer()
    if isinstance(callback.message, Message):
        await callback.message.answer("Сохранённых компаний пока нет.")


@router.callback_query(F.data == "how_it_works")
async def how_it_works(callback: CallbackQuery) -> None:
    await callback.answer()
    if isinstance(callback.message, Message):
        await callback.message.answer(
            "1. Загружаете чек.\n"
            "2. Добавляете участников.\n"
            "3. Указываете, кто что заказывал.\n"
            "4. Бот считает, кто кому и сколько должен."
        )


@router.callback_query(F.data == "manual_entry")
async def manual_entry(callback: CallbackQuery, state: FSMContext) -> None:
    if await state.get_state() not in (None, ReceiptInput.photo.state, ReceiptInput.review.state):
        await callback.answer("Чек уже открыт. Продолжите ввод или отмените его.")
        return
    if (await state.get_data()).get("recognition") is not None:
        await callback.answer("Чтобы дополнить текущий чек, нажмите «➕ Добавить позицию».")
        return
    await callback.answer()
    if isinstance(callback.message, Message):
        await state.set_data({"items": []})
        await state.set_state(ManualReceipt.name)
        await callback.message.answer("Введите название позиции.")


# До обработчиков шагов ручного ввода: фото не должно стать названием позиции
# или сбросить уже набранный чек.
@router.message(F.photo)
async def receipt_photo(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    if current_state in (ReceiptInput.photo.state, ReceiptInput.review.state):
        # Telegram передаёт варианты одного фото; последний — наибольший.
        # В FSM остаётся только ссылка: при повторе фото снова скачивается в память.
        photo = message.photo[-1]
        await state.update_data(receipt_photo={
            "file_id": photo.file_id,
            "file_unique_id": photo.file_unique_id,
            "width": photo.width,
            "height": photo.height,
            "file_size": photo.file_size,
        })
        await recognize_last_photo(message, state)
        return
    if current_state in {
        ManualReceipt.name.state, ManualReceipt.price.state,
        ManualReceipt.quantity.state, ManualReceipt.review.state,
    }:
        prompts = {
            ManualReceipt.name.state: "Введите название позиции.",
            ManualReceipt.price.state: "Введите цену позиции в рублях.",
            ManualReceipt.quantity.state: "Введите количество.",
            ManualReceipt.review.state: "Добавьте следующую позицию или закончите ввод.",
        }
        await message.answer(
            "Сейчас открыт ручной ввод чека. " + prompts[current_state],
            reply_markup=RECEIPT_MENU if current_state == ManualReceipt.review.state else None,
        )
        return
    if current_state in {ReceiptEditor.items.state, ReceiptEditor.item.state, ReceiptEditor.name.state,
                         ReceiptEditor.quantity.state, ReceiptEditor.price.state, ReceiptEditor.delete.state}:
        await message.answer("Сейчас открыт редактор чека. Закончите исправления или отмените чек.")
        return
    if current_state in allocation_handlers.ALLOCATION_STATES:
        await message.answer("Сейчас открыто распределение расходов. Используйте его кнопки или /start для нового чека.")
        return
    await message.answer(
        "Чтобы добавить фото чека, нажмите «🧾 Разделить чек» и отправьте фотографию ещё раз.",
        reply_markup=START_MENU,
    )


@router.callback_query(F.data == "confirm_recognition")
async def confirm_recognition(callback: CallbackQuery, state: FSMContext) -> None:
    data = await state.get_data()
    if await state.get_state() != ReceiptInput.review.state or not data.get("recognition"):
        await callback.answer("Нет распознанного чека для подтверждения.")
        return
    if not data["recognition"]["items"]:
        await callback.answer("Добавьте хотя бы одну позицию перед подтверждением.")
        return
    await callback.answer()
    if isinstance(callback.message, Message):
        result = RecognizedReceipt.model_validate(data["recognition"])
        await state.set_data({
            "items": [item.model_dump(mode="json", exclude_defaults=True) for item in result.items],
            "receipt_total": result.receipt_total,
            "currency": result.currency,
            "ruble_evidence": result.currency is None and receipt_currency(result) == 'RUB',
            "gross_total": result.gross_total,
            "prepayment": result.prepayment,
            "prepayment_ocr": result.prepayment_ocr,
            "prepayment_suspect": result.prepayment_suspect,
            "prepayment_verified": result.prepayment_verified,
            "amount_due": result.amount_due,
            "guest_subtotals": result.guest_subtotals,
            "derived_gross_total": result.derived_gross_total,
        })
        await state.set_state(ReceiptInput.confirmed)
        menu = await allocation_handlers.prepare(state, user_id=callback.from_user.id,
                                                 chat_id=callback.message.chat.id)
        await render(callback.message, state, "Чек подтверждён ✅", menu)


@router.callback_query(F.data == "edit_recognition")
async def edit_recognition(callback: CallbackQuery, state: FSMContext) -> None:
    if await state.get_state() != ReceiptInput.review.state or not (await state.get_data()).get("recognition"):
        await callback.answer("Нет открытого распознанного чека.")
        return
    await callback.answer()
    if isinstance(callback.message, Message):
        await receipt_editor.show_items(callback.message, state)


@router.callback_query(F.data == "repeat_recognition")
async def repeat_recognition(callback: CallbackQuery, state: FSMContext) -> None:
    if await state.get_state() not in (ReceiptInput.photo.state, ReceiptInput.review.state):
        await callback.answer("Последняя фотография больше недоступна. Начните новый чек.")
        return
    await callback.answer()
    if isinstance(callback.message, Message):
        await recognize_last_photo(callback.message, state)


@router.callback_query(F.data == "show_recognized_text")
async def show_recognized_text(callback: CallbackQuery, state: FSMContext) -> None:
    current_state = await state.get_state()
    data = await state.get_data()
    raw_text = data.get("recognition", {}).get("raw_text") or data.get("raw_text")
    if current_state not in (ReceiptInput.photo.state, ReceiptInput.review.state) or not raw_text:
        await callback.answer("Нет распознанного текста. Отправьте новую фотографию.")
        return
    await callback.answer()
    if isinstance(callback.message, Message):
        menu = RECOGNITION_MENU if current_state == ReceiptInput.review.state else RAW_TEXT_MENU
        await answer_in_parts(callback.message, "Распознанный текст:\n" + raw_text, menu)


@router.callback_query(F.data == "retry_photo")
async def retry_photo(callback: CallbackQuery, state: FSMContext) -> None:
    if await state.get_state() not in (ReceiptInput.photo.state, ReceiptInput.review.state):
        await callback.answer("Начните разделение нового чека.")
        return
    await callback.answer()
    if isinstance(callback.message, Message):
        await state.clear()
        await state.set_state(ReceiptInput.photo)
        await callback.message.answer("Отправьте фотографию чека ещё раз.", reply_markup=RECOGNITION_RETRY_MENU)


@router.message(ManualReceipt.name)
async def item_name(message: Message, state: FSMContext) -> None:
    name = (message.text or "").strip()
    editing = (await state.get_data()).get("manual_target") == "recognition"
    if not name or (editing and len(name) > 200):
        await message.answer("Введите название позиции текстом.")
        return
    await state.update_data(name=name)
    await state.set_state(ManualReceipt.price)
    await message.answer("Введите цену за одну единицу." if editing else "Введите цену позиции в рублях.")


@router.message(ManualReceipt.price)
async def item_price(message: Message, state: FSMContext) -> None:
    editing = (await state.get_data()).get("manual_target") == "recognition"
    price = receipt_editor.decimal_input(message.text) if editing else parse_price(message.text)
    if price is None:
        await message.answer("Введите цену больше 0: целое число или число с точкой или запятой.")
        return
    await state.update_data(price=str(price))
    await state.set_state(ManualReceipt.quantity)
    await message.answer("Введите количество.")


@router.message(ManualReceipt.quantity)
async def item_quantity(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    if data.get("manual_target") == "recognition":
        quantity = receipt_editor.decimal_input(message.text, quantity=True)
        if quantity is None:
            await message.answer("Введите положительное количество (до 6 знаков после запятой).")
            return
        result = await receipt_editor.current_receipt(state)
        if result is None:
            await message.answer("Начните новый чек.")
            return
        try:
            item = ReceiptItem(name=data["name"], quantity=format(quantity, "f"), unit_price=data["price"],
                               total_price=receipt_editor.total_price(quantity, data["price"]))
            if len(result.items) >= 200:
                raise ValueError("Too many items")
            await receipt_editor.save_items(state, result, [*result.items, item])
        except ValueError:
            await message.answer("Не удалось добавить позицию: проверьте количество и сумму (максимум 200 позиций).")
            return
        await state.update_data(manual_target=None, name=None, price=None)
        await receipt_editor.show_items(message, state, data.get("receipt_edit_page", 0))
        return
    quantity = parse_quantity(message.text)
    if quantity is None:
        await message.answer("Введите количество: целое число больше 0.")
        return
    items = [*data["items"], {"name": data["name"], "price": data["price"], "quantity": quantity}]
    await state.set_data({"items": items})
    await state.set_state(ManualReceipt.review)
    await show_receipt(message, items)


@router.callback_query(F.data == "add_item")
async def add_item(callback: CallbackQuery, state: FSMContext) -> None:
    if await state.get_state() != ManualReceipt.review.state:
        await callback.answer("Нет чека, готового к добавлению позиции.")
        return
    await callback.answer()
    if isinstance(callback.message, Message):
        await state.set_state(ManualReceipt.name)
        await callback.message.answer("Введите название позиции.")


@router.callback_query(F.data == "finish_receipt")
async def finish_receipt(callback: CallbackQuery, state: FSMContext) -> None:
    if await state.get_state() != ManualReceipt.review.state:
        await callback.answer("Нет чека, готового к завершению.")
        return
    await callback.answer()
    if isinstance(callback.message, Message):
        data = await state.get_data()
        try:
            # Retain the exact manual lines for allocation instead of clearing them.
            _, total = receipt_lines(data['items'])
            await state.update_data(currency='RUB', receipt_total=total)
            menu = await allocation_handlers.prepare(state, user_id=callback.from_user.id,
                                                     chat_id=callback.message.chat.id)
        except (ValueError, ArithmeticError):
            await callback.message.answer('Не удалось перейти к делёжке: нужны от 1 до 200 позиций '
                                          'с суммами в целых копейках, не более 999999999999.99 за позицию. '
                                          'Чек сохранён; проверьте ввод или начните заново через /start.')
            return
        await state.set_state(ReceiptInput.confirmed)
        await show_receipt(callback.message, data["items"], finished=True, finished_menu=menu, state=state)


@router.callback_query(F.data == "cancel_receipt")
async def cancel_receipt(callback: CallbackQuery, state: FSMContext) -> None:
    input_states = {s.state for group in (ManualReceipt, ReceiptEditor) for s in group.__all_states__}
    if await state.get_state() not in input_states | {ReceiptInput.photo.state, ReceiptInput.review.state}:
        await callback.answer('Меню устарело. Используйте текущий экран или /start.')
        return
    await callback.answer()
    if isinstance(callback.message, Message):
        await state.clear()
        await callback.message.answer("Ввод отменён. Текущий чек очищен.", reply_markup=START_MENU)


router.include_router(saved_receipts.router)
router.include_router(receipt_editor.router)
router.include_router(allocation_handlers.router)
