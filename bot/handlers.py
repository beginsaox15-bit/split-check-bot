from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot.receipt import parse_price, parse_quantity, receipt_lines


router = Router()


class ManualReceipt(StatesGroup):
    name = State()
    price = State()
    quantity = State()
    review = State()


class ReceiptInput(StatesGroup):
    photo = State()


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

PHOTO_MENU = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="✍️ Ввести вручную", callback_data="manual_entry")],
    [InlineKeyboardButton(text="❌ Отменить", callback_data="cancel_receipt")],
])

PHOTO_RECEIVED_TEXT = (
    "Фото чека получено ✅\n\n"
    "На следующем этапе я смогу распознать позиции автоматически.\n\n"
    "Пока можете ввести позиции вручную."
)

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
    await message.answer(chunks[-1], reply_markup=START_MENU if finished else RECEIPT_MENU)


@router.message(CommandStart())
async def start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(START_TEXT, reply_markup=START_MENU)


@router.callback_query(F.data == "split_check")
async def split_check(callback: CallbackQuery, state: FSMContext) -> None:
    if await state.get_state() not in (None, ReceiptInput.photo.state):
        await callback.answer("Чек уже открыт. Продолжите ввод или отмените его.")
        return
    await callback.answer()
    if isinstance(callback.message, Message):
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
    if await state.get_state() not in (None, ReceiptInput.photo.state):
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
    if current_state == ReceiptInput.photo.state:
        # Telegram передаёт варианты одного фото; последний — наибольший.
        # Для будущего OCR достаточно ссылки Telegram и метаданных в памяти FSM.
        photo = message.photo[-1]
        await state.update_data(receipt_photo={
            "file_id": photo.file_id,
            "file_unique_id": photo.file_unique_id,
            "width": photo.width,
            "height": photo.height,
            "file_size": photo.file_size,
        })
        await message.answer(PHOTO_RECEIVED_TEXT, reply_markup=PHOTO_MENU)
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
