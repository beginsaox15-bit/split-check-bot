import logging
from decimal import Decimal
from io import BytesIO

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot.receipt import money, parse_price, parse_quantity, receipt_lines
from bot.services.receipt_models import (
    LowQualityReceipt, RecognizedReceipt, TesseractLanguagesUnavailable,
    TesseractNotInstalled, UnparsedReceipt, UnreadableReceipt,
)
from bot.services.receipt_recognition import recognize_receipt


router = Router()
logger = logging.getLogger(__name__)


class ManualReceipt(StatesGroup):
    name = State()
    price = State()
    quantity = State()
    review = State()


class ReceiptInput(StatesGroup):
    photo = State()
    review = State()
    confirmed = State()


START_TEXT = (
    "Привет! 👋\n"
    "Я помогу разделить общий чек между друзьями.\n\n"
    "Что хотите сделать?"
)

START_MENU = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="🧾 Разделить чек", callback_data="split_check")],
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

RECOGNITION_MENU = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="✅ Всё верно", callback_data="confirm_recognition")],
    [InlineKeyboardButton(text="✏️ Исправить", callback_data="edit_recognition")],
    [InlineKeyboardButton(text="🔄 Распознать заново", callback_data="repeat_recognition")],
    [InlineKeyboardButton(text="📄 Показать распознанный текст", callback_data="show_recognized_text")],
    [InlineKeyboardButton(text="❌ Отменить", callback_data="cancel_receipt")],
])

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


async def show_receipt(message: Message, items: list[dict], *, finished: bool = False) -> None:
    lines, total = receipt_lines(items)
    heading = "Итоговый чек:" if finished else f"Добавлено ✅\n\n{lines[-1]}\n\nТекущий чек:"
    text = heading + "\n\n" + "\n".join(
        f"{index}. {line}" for index, line in enumerate(lines, 1)
    ) + f"\n\nИтого: {total} ₽"
    await answer_in_parts(message, text, START_MENU if finished else RECEIPT_MENU)


async def answer_in_parts(message: Message, text: str, menu: InlineKeyboardMarkup) -> None:
    # Длинный чек отправляем частями; кнопки находятся под последней частью.
    chunks = []
    # 2000 символов укладываются в лимит Telegram и при наличии emoji.
    while len(text) > 2000:
        split_at = text.rfind("\n", 0, 2000)
        if split_at <= 0:
            split_at = 2000
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    chunks.append(text)
    for chunk in chunks[:-1]:
        await message.answer(chunk)
    await message.answer(chunks[-1], reply_markup=menu)


async def recognize_last_photo(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    photo = data.get("receipt_photo")
    if not photo:
        await message.answer("Отправьте фотографию чека ещё раз.", reply_markup=RECOGNITION_RETRY_MENU)
        return
    # Повторное распознавание инвалидирует предыдущий неподтверждённый результат.
    await state.set_data({"receipt_photo": photo})
    await state.set_state(ReceiptInput.photo)
    await message.answer("Чек получен 🧾\nРаспознаю текст…")
    try:
        with BytesIO() as image:
            await message.bot.download(photo["file_id"], destination=image, timeout=30)
            result = await recognize_receipt(image.getvalue())
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
    currency = "₽" if result.currency == "RUB" else (result.currency or "")
    lines = [
        f"{index}. {item.name}"
        + (f" × {money(Decimal(item.quantity))}" if Decimal(item.quantity) != 1 else "")
        + f" — {money(Decimal(item.total_price))} {currency}".rstrip()
        for index, item in enumerate(result.items, 1)
    ]
    text = "Я распознал чек 🧾\n\n" + "\n".join(lines)
    text += f"\n\nИтого по найденным позициям: {money(result.items_total)} {currency}".rstrip()
    if result.receipt_total is not None:
        total = f"{money(Decimal(result.receipt_total))} {currency}".rstrip()
        text += f"\nИтого в чеке: {total}"
    if result.currency is None:
        text += "\nВалюта не определена."
    if result.totals_mismatch:
        text += "\n\n⚠️ OCR мог распознать чек неточно. Сумма позиций не совпадает с итогом чека."
    text += "\n\nПроверьте позиции перед подтверждением."
    await answer_in_parts(message, text, RECOGNITION_MENU)


@router.message(CommandStart())
async def start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(START_TEXT, reply_markup=START_MENU)


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
    await callback.answer()
    if isinstance(callback.message, Message):
        result = RecognizedReceipt.model_validate(data["recognition"])
        await state.set_data({
            "items": [item.model_dump(mode="json") for item in result.items],
            "receipt_total": result.receipt_total,
            "currency": result.currency,
        })
        await state.set_state(ReceiptInput.confirmed)
        await callback.message.answer("Чек подтверждён ✅", reply_markup=START_MENU)


@router.callback_query(F.data == "edit_recognition")
async def edit_recognition(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    if isinstance(callback.message, Message) and await state.get_state() == ReceiptInput.review.state:
        await callback.message.answer("Редактирование распознанных позиций будет добавлено следующим этапом.")


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
    if not name:
        await message.answer("Введите название позиции текстом.")
        return
    await state.update_data(name=name)
    await state.set_state(ManualReceipt.price)
    await message.answer("Введите цену позиции в рублях.")


@router.message(ManualReceipt.price)
async def item_price(message: Message, state: FSMContext) -> None:
    price = parse_price(message.text)
    if price is None:
        await message.answer("Введите цену больше 0: целое число или число с точкой или запятой.")
        return
    await state.update_data(price=str(price))
    await state.set_state(ManualReceipt.quantity)
    await message.answer("Введите количество.")


@router.message(ManualReceipt.quantity)
async def item_quantity(message: Message, state: FSMContext) -> None:
    quantity = parse_quantity(message.text)
    if quantity is None:
        await message.answer("Введите количество: целое число больше 0.")
        return
    data = await state.get_data()
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
        await show_receipt(callback.message, data["items"], finished=True)
        await state.clear()


@router.callback_query(F.data == "cancel_receipt")
async def cancel_receipt(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    if isinstance(callback.message, Message):
        await state.clear()
        await callback.message.answer("Ввод отменён. Текущий чек очищен.", reply_markup=START_MENU)
