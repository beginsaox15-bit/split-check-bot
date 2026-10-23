"""Осторожные эвристики для строк OCR. Результат всегда требует проверки."""

import re
from decimal import Decimal, localcontext

from pydantic import ValidationError

from bot.services.receipt_models import RecognizedReceipt, ReceiptItem, UnreadableReceipt


AMOUNT = r"(?:[0-9]{1,3}(?: [0-9]{3})+|[0-9]+)(?:[.,][0-9]{1,2})?"
TAIL_AMOUNT = re.compile(rf"(?<![\d.,])(?P<amount>{AMOUNT})\s*$")
MULTIPLIER = re.compile(
    rf"(?P<quantity>[0-9]+(?:[.,][0-9]{{1,6}})?)\s*[xх×*]\s*"
    rf"(?P<price>{AMOUNT})(?:\s*(?:=|—)\s*|\s+)?(?P<total>{AMOUNT})?$", re.I,
)
COLUMNS = re.compile(
    r"^(?P<name>.*\S)\s+(?P<quantity>[0-9]+(?:[.,][0-9]{1,6})?)"
    r"\s+(?P<price>[0-9]+(?:[.,][0-9]{1,2})?)\s+(?P<total>[0-9]+(?:[.,][0-9]{1,2})?)$"
)
TOTAL = re.compile(r"^(?:ИТОГ(?:О)?|ВСЕГО(?: К ОПЛАТЕ)?|К ОПЛАТЕ|GRAND TOTAL|TOTAL(?: AMOUNT)?|AMOUNT DUE)\b", re.I)
SERVICE = re.compile(
    r"^(?:ИНН|КПП|ОГРН|QR|ККТ|ККМ|КАССА|КАССИР|КАССОВЫЙ|ЧЕК|ФН|ФД|ФПД|ФП|РН|ЗН|"
    r"СМЕНА|ТЕРМИНАЛ|ОПЕРАТОР|ПРИХОД|ВОЗВРАТ|СДАЧА|НАЛИЧН\w*|БЕЗНАЛ\w*|ОПЛАТА|"
    r"БАНК\w*|КАРТА|НДС|СКИДКА|СЕРВИСНЫЙ СБОР|ОБСЛУЖИВАНИЕ|ФИСКАЛ\w*|ДАТА|ВРЕМЯ|"
    r"АДРЕС|ТЕЛ\w*|СПАСИБО|ПОДЫТОГ|ООО|АО|ИП|НАИМЕНОВАНИЕ|КОЛИЧЕСТВО|ЦЕНА|СУММА|"
    r"VAT|TAX|CASH|CARD|PAYMENT|CHANGE|SUBTOTAL|DISCOUNT|SERVICE CHARGE|RECEIPT|"
    r"ORDER|CASHIER|TERMINAL|DATE|TIME|THANK YOU|QTY|ITEM|DESCRIPTION)\b", re.I,
)
DATE_TIME = re.compile(r"\b(?:[0-3]?\d[./-][01]?\d[./-](?:20)?\d{2}|20\d{2}-\d{2}-\d{2}|\d{1,2}:\d{2}(?::\d{2})?)\b")
CURRENCY_SUFFIX = re.compile(r"\s*(?:₽|руб\.?|р\.|RUB|USD|EUR|€|\$)\s*$", re.I)
RUBLE_MARKER = re.compile(r"₽|\b(?:руб|RUB)\b", re.I)
RUSSIAN_FISCAL = re.compile(r"\b(?:ИНН|ККТ|ККМ|ФН|ФД|ФПД|ОФД|ФФД)\b", re.I)
DOLLAR_MARKER = re.compile(r"\bUSD\b|\$\s*\d|\d\s*\$", re.I)
EURO_MARKER = re.compile(r"\bEUR\b|€\s*\d|\d\s*€", re.I)


def detect_currency(lines: list[str]) -> str:
    normalized = [normalize(line) for line in lines]
    # Признаки рублёвого чека важнее случайного $ в OCR/фискальных данных;
    # результат не должен зависеть от порядка строк.
    if any(RUBLE_MARKER.search(line) or RUSSIAN_FISCAL.search(line) for line in normalized):
        return "RUB"
    evidence = [line for line in normalized
                if not re.search(r"^QR\b|://|www\.", line, re.I)]
    for currency, marker in (("USD", DOLLAR_MARKER), ("EUR", EURO_MARKER)):
        if any(marker.search(line) for line in evidence):
            return currency
    # Существующий безопасный fallback для русскоязычных рублёвых чеков.
    return "RUB"


def normalize(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").replace("\u202f", " ").split())


def decimal_value(text: str) -> Decimal:
    return Decimal(text.replace(" ", "").replace(",", "."))


def decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def make_item(name: str, quantity: Decimal, unit: Decimal | None, total: Decimal) -> ReceiptItem | None:
    name = re.sub(r"^\d{1,3}[.)]\s*", "", name).strip(" .-=:")
    if not name or not re.search(r"[A-Za-zА-Яа-яЁё]", name) or SERVICE.match(name):
        return None
    if quantity <= 0 or total <= 0:
        return None
    try:
        return ReceiptItem(
            name=name, quantity=decimal_text(quantity),
            unit_price=decimal_text(unit) if unit is not None else None,
            total_price=decimal_text(total),
        )
    except ValidationError:
        return None


def parse_receipt_lines(lines: list[str]) -> RecognizedReceipt:
    items = []
    receipt_total = None
    pending_name = None
    awaiting_total = False
    currency = detect_currency(lines)
    for raw in lines:
        line = normalize(raw)
        if not line:
            continue
        line = CURRENCY_SUFFIX.sub("", line).strip()
        if TOTAL.match(line):
            tail = TAIL_AMOUNT.search(line)
            receipt_total = decimal_text(decimal_value(tail["amount"])) if tail else None
            awaiting_total = tail is None
            pending_name = None
            continue
        if awaiting_total:
            awaiting_total = False
            if re.fullmatch(AMOUNT, line):
                receipt_total = decimal_text(decimal_value(line))
                continue
        if SERVICE.match(line) or DATE_TIME.search(line) or "://" in line or "www." in line.lower():
            pending_name = None
            continue
        item = None
        match = MULTIPLIER.search(line)
        if match:
            name = line[:match.start()].strip() or pending_name or ""
            quantity, unit = decimal_value(match["quantity"]), decimal_value(match["price"])
            with localcontext() as context:
                context.prec = 40
                total = decimal_value(match["total"]) if match["total"] else quantity * unit
            item = make_item(name, quantity, unit, total)
        else:
            columns = COLUMNS.fullmatch(line)
            if columns:
                item = make_item(columns["name"], decimal_value(columns["quantity"]),
                                 decimal_value(columns["price"]), decimal_value(columns["total"]))
            else:
                tail = TAIL_AMOUNT.search(line)
                if tail:
                    if line[:tail.start()].rstrip().endswith(("-", "−")):
                        pending_name = None
                        continue
                    name = line[:tail.start()].strip(" =—:") or pending_name or ""
                    total = decimal_value(tail["amount"])
                    item = make_item(name, Decimal(1), None, total)
        if item:
            items.append(item)
            pending_name = None
        elif re.search(r"[A-Za-zА-Яа-яЁё]", line):
            # Распространённый формат: название на строке перед ценой/количеством.
            pending_name = line
        else:
            pending_name = None
    if not items or len(items) > 200:
        raise UnreadableReceipt("No reliable purchase lines found")
    return RecognizedReceipt(readable=True, items=items, receipt_total=receipt_total, currency=currency)
