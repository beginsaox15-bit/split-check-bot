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
GUEST_TOTAL = re.compile(r"^(?:и?того|игого|подытог)(?:\s+к\s+оплате)?\s+г?ость\s*[:№#]?\s*(?P<guest>\d{1,3})\b(?P<value>.*)$", re.I)
GUEST = re.compile(r"^г?ость\s*[:№#]?\s*(?P<guest>\d{1,3})\b(?P<value>.*)$", re.I)
SUMMARY_LABELS = (
    ("gross_total", re.compile(r"^(?:полная\s+сумма|gross total)\b(?P<value>.*)$", re.I)),
    ("prepayment", re.compile(r"^(?:предоплата|аванс|prepayment)\b(?P<value>.*)$", re.I)),
    ("amount_due", re.compile(r"^(?:(?:и?того|игого)\s+к\s+оплате|к\s+оплате|amount due)\b(?P<value>.*)$", re.I)),
)
RESTAURANT_SERVICE = re.compile(r"^(?:гость|ость|стол|cron|стоп|заказ|официант|оеициант|гостевой|ресторан|открыт|закрыт|сч[её]т|кол[\s.-]*во|кой[\s.-]*во|оплатить|чаевые|отзыв)\b", re.I)
THREE_COLUMNS = re.compile(rf"^(?P<name>.*?\S)\s+(?P<quantity>[0-9]+(?:[.,][0-9]{{1,6}})?)\s+(?P<total>{AMOUNT})$")
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
    if not plausible_name(name):
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


def plausible_name(name: str) -> bool:
    if not name or SERVICE.match(name) or RESTAURANT_SERVICE.match(name):
        return False
    if any(pattern.match(name) for _, pattern in SUMMARY_LABELS) or TOTAL.match(name):
        return False
    if re.search(r"\b(?:наименование|официант|оеициант|предоплата|итого|кол[\s.-]*во|гость\s*\d|стол\s*\d)\b", name, re.I):
        return False
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", name)
    if sum(map(len, tokens)) < len(name.replace(" ", "")) * .55:
        return False
    words = [token for token in tokens if re.search(r"[A-Za-zА-Яа-яЁё]", token)]
    solid = [token for token in words if len(token) >= 3 and len(re.findall(r"[A-Za-zА-Яа-яЁё]", token)) >= 2]
    if not solid or len(solid) < len(words) / 2:
        return False
    # Не склеиваем обрывки разных алфавитов внутри слова и числовые колонки.
    if any(re.search(r"[A-Za-z]", word) and re.search(r"[А-Яа-яЁё]", word) for word in words):
        return False
    return not re.search(r"\s\d+(?:[.,]\d+)?\s*$", name)


def parse_receipt_lines(lines: list[str]) -> RecognizedReceipt:
    items, guest_subtotals, totals, warnings = [], {}, {}, []
    pending_name = None
    pending_summary = None
    current_guest = None
    guest_block_start = 0
    skipped = 0
    prepayment_seen = False
    currency = detect_currency(lines)
    restaurant = any(GUEST.match(normalize(line)) or GUEST_TOTAL.match(normalize(line)) for line in lines)

    def warn(message):
        if message not in warnings:
            warnings.append(message)

    def set_summary(kind, guest, value):
        target, key = (guest_subtotals, guest) if kind == "guest" else (totals, kind)
        amount = decimal_text(decimal_value(value))
        if len(amount.split(".")[0]) > 12:
            warn("Не все итоговые суммы удалось прочитать. Проверьте чек.")
            return
        if key in target and target[key] != amount:
            warn("Повторные итоговые значения различаются. Проверьте чек.")
        else:
            target[key] = amount

    def summary_amount(value):
        value = value.strip(" :;=—")
        return value if re.fullmatch(AMOUNT, value) else None

    for raw in lines:
        line = CURRENCY_SUFFIX.sub("", normalize(raw)).strip()
        if not line:
            continue
        if pending_summary:
            kind, guest = pending_summary
            pending_summary = None
            value = summary_amount(line)
            if value is not None:
                set_summary(kind, guest, value)
                continue
            warn("Не все итоговые суммы удалось прочитать. Проверьте чек.")
        guest_total, guest_header = GUEST_TOTAL.match(line), GUEST.match(line)
        if guest_total:
            kind, guest, rest = "guest", guest_total["guest"], guest_total["value"]
            if current_guest is not None and current_guest != guest:
                warn("Номера гостей в заголовке и подытоге различаются. Проверьте чек.")
            else:
                for item in items[guest_block_start:]:
                    item.guest_id = guest
            current_guest = None
            guest_block_start = len(items)
        elif guest_header:
            current_guest = guest_header["guest"]
            guest_block_start = len(items)
            pending_name = None
            # Некоторые чеки печатают подытог прямо в строке заголовка гостя.
            rest = re.sub(r"^[ :;—-]*(?:итог(?:о)?[ :;]*)?", "", guest_header["value"], flags=re.I)
            if rest and summary_amount(rest) is not None:
                set_summary("guest", current_guest, rest)
            continue
        else:
            summary = next(((kind, pattern.match(line)) for kind, pattern in SUMMARY_LABELS if pattern.match(line)), None)
            if summary:
                kind, match = summary
                guest, rest = None, match["value"]
            elif TOTAL.match(line):
                kind, guest, rest = "receipt_total", None, line[TOTAL.match(line).end():]
            else:
                kind = None
        if kind is not None:
            pending_name = None
            if kind == "prepayment":
                prepayment_seen = True
            value = summary_amount(rest)
            if value is not None:
                set_summary(kind, guest, value)
            elif not rest.strip(" :;=—"):
                pending_summary = (kind, guest)
            else:
                warn("Не все итоговые суммы удалось прочитать. Проверьте чек.")
            continue
        if (SERVICE.match(line) or RESTAURANT_SERVICE.match(line) or DATE_TIME.search(line)
                or "://" in line or "www." in line.lower()):
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
            three = THREE_COLUMNS.fullmatch(line) if restaurant else None
            if columns and (not three or decimal_value(columns["quantity"]) * decimal_value(columns["price"]) == decimal_value(columns["total"])):
                item = make_item(columns["name"], decimal_value(columns["quantity"]),
                                 decimal_value(columns["price"]), decimal_value(columns["total"]))
            elif three:
                item = make_item(three["name"], decimal_value(three["quantity"]), None, decimal_value(three["total"]))
            else:
                tail = TAIL_AMOUNT.search(line)
                if tail:
                    if line[:tail.start()].rstrip().endswith(("-", "−")):
                        skipped += 1
                        pending_name = None
                        continue
                    name = line[:tail.start()].strip(" =—:") or pending_name or ""
                    total = decimal_value(tail["amount"])
                    item = make_item(name, Decimal(1), None, total)
        if item:
            item.guest_id = current_guest
            items.append(item)
            pending_name = None
        elif plausible_name(line) and not re.search(r"\d+[.,]\d{2}\s*$", line):
            # Перенос названия разрешён только на непосредственно следующую строку.
            if pending_name is not None:
                skipped += 1
            pending_name = line
        else:
            skipped += 1
            pending_name = None
    if pending_summary:
        warn("Не все итоговые суммы удалось прочитать. Проверьте чек.")
    if not items or len(items) > 200:
        raise UnreadableReceipt("No reliable purchase lines found")
    receipt_total = totals.get("gross_total")
    if receipt_total is None and not prepayment_seen:
        receipt_total = totals.get("receipt_total", totals.get("amount_due"))
    return RecognizedReceipt(
        readable=True, items=items, receipt_total=receipt_total, currency=currency,
        gross_total=totals.get("gross_total"), prepayment=totals.get("prepayment"),
        amount_due=totals.get("amount_due"), guest_subtotals=guest_subtotals,
        skipped_lines=skipped, parse_warnings=warnings,
    )
