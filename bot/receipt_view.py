"""Shared receipt display. Financial OCR facts are never changed by rendering."""
from decimal import Decimal

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message
from bot.receipt import money
from bot.services.receipt_models import RecognizedReceipt
from bot.currency import receipt_currency, symbol


RECOGNITION_MENU = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="✅ Всё верно", callback_data="confirm_recognition")],
    [InlineKeyboardButton(text="✏️ Исправить", callback_data="edit_recognition")],
    [InlineKeyboardButton(text="➕ Добавить позицию", callback_data="add_recognition_item")],
    [InlineKeyboardButton(text="🔄 Распознать заново", callback_data="repeat_recognition")],
    [InlineKeyboardButton(text="📄 Показать распознанный текст", callback_data="show_recognized_text")],
    [InlineKeyboardButton(text="❌ Отменить", callback_data="cancel_receipt")],
])


def suspicious(item) -> bool:
    return item.suspicious or 'low_confidence' in item.suspicion_reasons


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


async def show_recognition(message: Message, result: RecognizedReceipt, *, edited: bool = False) -> None:
    shown_currency = receipt_currency(result)
    currency = symbol(shown_currency)
    lines = [
        f"{index}. {item.name}"
        + (f" (гость {item.guest_id})" if item.guest_id else "")
        + (f" × {money(Decimal(item.quantity))}" if Decimal(item.quantity) != 1 else "")
        + f" — {money(Decimal(item.total_price))} {currency}".rstrip()
        + (" ⚠️ проверьте" if suspicious(item) else "")
        for index, item in enumerate(result.items, 1)
    ]
    text = ("Текущий чек 🧾\n\n" if edited else "Я распознал чек 🧾\n\n") + "\n".join(lines)
    label = "Итого по позициям" if edited else "Итого по найденным позициям"
    text += f"\n\n{label}: {money(result.items_total)} {currency}".rstrip()
    if any(value is not None for value in (result.gross_total, result.prepayment, result.amount_due)):
        for label, value in (("Полная сумма", result.gross_total), ("Предоплата", result.prepayment),
                             ("Итого к оплате", result.amount_due)):
            if value is not None:
                text += f"\n{label}: {money(Decimal(value))} {currency}".rstrip()
                if label == "Предоплата" and not result.prepayment_verified:
                    text += " (не проверена по итогам)"
    elif result.receipt_total is not None:
        total = f"{money(Decimal(result.receipt_total))} {currency}".rstrip()
        text += f"\nИтого в чеке: {total}"
    if shown_currency is None:
        text += "\nВалюта не определена."
    elif result.currency is None:
        text += "\nВалюта: RUB (по обозначению на чеке; поле OCR не изменено)."
    if result.gross_total is None and result.derived_gross_total is not None:
        text += f"\nПолная сумма (вычислена из предоплаты и остатка): {money(Decimal(result.derived_gross_total))} {currency}"
        if result.missing_difference:
            text += f"\nРазница с суммой найденных позиций: {money(result.missing_difference)} {currency}. Проверьте полноту чека."
    if result.totals_mismatch:
        text += "\n\n⚠️ OCR мог распознать чек неточно. Сумма позиций не совпадает с итогом чека."
    if result.guest_subtotals:
        text += f"\nСумма итогов гостей: {money(result.guest_subtotals_total)} {currency}".rstrip()
    if result.guest_subtotals_mismatch:
        text += "\n⚠️ Сумма итогов гостей не совпадает с полной суммой чека."
    if result.payment_mismatch:
        text += "\n⚠️ Полная сумма минус предоплата не совпадает с суммой к оплате."
    if result.prepayment_suspect:
        text += f"\n⚠️ Предоплата прочитана как {money(Decimal(result.prepayment_ocr))} {currency}; требует проверки и не используется в расчётах."
        if result.calculated_prepayment is not None:
            text += f"\nРазница полной суммы и остатка: {money(result.calculated_prepayment)} {currency}. Это не подтверждённая предоплата."
    has_suspicious = any(suspicious(item) for item in result.items)
    if result.skipped_lines or has_suspicious or result.totals_mismatch or (len(result.items) < 3 and result.comparison_total is None):
        text += "\n\nЧек распознан не полностью. "
        text += ("Я отметил сомнительные позиции значком ⚠️ — проверьте их или добавьте пропущенные."
                 if has_suspicious else "Проверьте позиции или добавьте пропущенные.")
    if result.skipped_lines:
        text += f"\n⚠️ Сомнительные строки пропущены: {result.skipped_lines}. Проверьте чек."
    for warning in result.parse_warnings:
        text += "\n⚠️ " + warning
    text += "\n\nПроверьте позиции перед подтверждением."
    await answer_in_parts(message, text, RECOGNITION_MENU)
