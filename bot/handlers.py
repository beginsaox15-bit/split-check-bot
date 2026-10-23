from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message


router = Router()

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


@router.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(START_TEXT, reply_markup=START_MENU)


@router.callback_query(F.data == "split_check")
async def split_check(callback: CallbackQuery) -> None:
    await callback.answer()
    if isinstance(callback.message, Message):
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
async def manual_entry(callback: CallbackQuery) -> None:
    await callback.answer()
    if isinstance(callback.message, Message):
        await callback.message.answer("Ручной ввод позиций пока недоступен.")
