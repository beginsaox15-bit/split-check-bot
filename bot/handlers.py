from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message


router = Router()

START_TEXT = (
    "Привет! 👋\n"
    "Я помогу разделить общий чек между друзьями.\n\n"
    "Скоро здесь можно будет загрузить чек, добавить участников и автоматически "
    "посчитать, кто сколько должен."
)


@router.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(START_TEXT)
