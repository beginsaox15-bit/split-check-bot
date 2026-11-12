"""One editable allocation message; text-entry prompts remain separate."""
import logging

from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

logger = logging.getLogger(__name__)
TEXT_LIMIT = 3500


class ScreenEditError(Exception):
    pass


def text_pages(text):
    """Keep all text, including long names and emoji, within Telegram's limit."""
    pages, current, units = [], [], 0
    for char in text:
        width = 2 if ord(char) > 0xFFFF else 1
        if units + width > TEXT_LIMIT:
            pages.append(''.join(current))
            current, units = [], 0
        current.append(char)
        units += width
    pages.append(''.join(current))
    return pages


async def render(message, state, text, markup, page=0):
    data = await state.get_data()
    pages = text_pages(text)
    page = max(0, min(page, len(pages)-1))
    token, revision = data['allocation_token'], data['allocation_revision']
    # Refresh every action when navigating text pages; stale menu versions stay invalid.
    rows = []
    for row in markup.inline_keyboard:
        buttons = []
        for button in row:
            callback = button.callback_data
            if callback and callback.startswith('sp:'):
                parts = callback.split(':')
                callback = f'sp:{token}:{revision}:{parts[3]}:{parts[4]}'
            buttons.append(button.model_copy(update={'callback_data': callback}))
        rows.append(buttons)
    base = InlineKeyboardMarkup(inline_keyboard=rows)
    navigation = []
    if page:
        navigation.append(InlineKeyboardButton(text='⬅️ Текст', callback_data=f'sp:{token}:{revision}:screen_page:{page-1}'))
    if page+1 < len(pages):
        navigation.append(InlineKeyboardButton(text='Текст ➡️', callback_data=f'sp:{token}:{revision}:screen_page:{page+1}'))
    shown = InlineKeyboardMarkup(inline_keyboard=[*rows, *([navigation] if navigation else [])])
    body = pages[page] + (f'\n\nСтраница {page+1}/{len(pages)}' if len(pages) > 1 else '')
    target = data.get('allocation_screen_id')
    if target is None:
        # Only explicit recovery or a text-input migration can create an anchor.
        sent = await message.answer(body, reply_markup=shown, parse_mode=None)
        target = sent.message_id
    else:
        if data.get('allocation_screen_chat_id') != message.chat.id:
            raise ScreenEditError('Screen belongs to another chat')
        try:
            await message.bot.edit_message_text(chat_id=message.chat.id, message_id=target,
                                                text=body, reply_markup=shown, parse_mode=None)
        except TelegramBadRequest as error:
            if 'message is not modified' not in error.message.lower():
                logger.warning('Allocation screen edit failed: %s', type(error).__name__)
                raise ScreenEditError('Screen could not be edited') from None
        except TelegramAPIError as error:
            logger.warning('Allocation screen edit failed: %s', type(error).__name__)
            raise ScreenEditError('Screen could not be edited') from None
    await state.update_data(allocation_screen_id=target, allocation_screen_chat_id=message.chat.id,
                            allocation_screen_text=text, allocation_screen_markup=base.model_dump(mode='json'),
                            allocation_screen_page=page)
