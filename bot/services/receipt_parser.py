"""Осторожные эвристики для строк OCR. Результат всегда требует проверки."""

import re
from decimal import Decimal, localcontext

from pydantic import ValidationError

from bot.services.receipt_models import ParserDiagnostics, ParserLineDecision, RecognizedReceipt, ReceiptItem, UnreadableReceipt


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
    ("gross_total", re.compile(r"^(?:полная\s+сум(?:ма|ми|и)|gross total)\b(?P<value>.*)$", re.I)),
    ("prepayment", re.compile(r"^(?:предоплата|аванс|prepayment)\b(?P<value>.*)$", re.I)),
    ("amount_due", re.compile(r"^(?:(?:и?того|игого)\s+к\s+оплате|к\s+оплате|amount due)\b(?P<value>.*)$", re.I)),
)
RESTAURANT_SERVICE = re.compile(r"^(?:гость|ость|стол|cron|стоп|заказ|официант|оеициант|гостевой|ресторан|открыт|закрыт|сч[её]т|кол[\s.-]*во|кой[\s.-]*во|оплатить|чаевые|отзыв)\b", re.I)
THREE_COLUMNS = re.compile(rf"^(?P<name>.*?\S)\s+(?P<quantity>[0-9]+(?:[.,][0-9]{{1,6}})?)\s+(?P<total>{AMOUNT})$")
RUBLE_MARKER = re.compile(r"₽|\b(?:руб|RUB)\b", re.I)
RUSSIAN_FISCAL = re.compile(r"\b(?:ИНН|ККТ|ККМ|ФН|ФД|ФПД|ОФД|ФФД)\b", re.I)


def detect_currency(lines: list[str]) -> str:
    normalized = [normalize(line) for line in lines]
    # Признаки рублёвого чека важнее случайного $ в OCR/фискальных данных;
    # результат не должен зависеть от порядка строк.
    if any(RUBLE_MARKER.search(line) or RUSSIAN_FISCAL.search(line) for line in normalized):
        return "RUB"
    russian = (any(GUEST.match(line) or GUEST_TOTAL.match(line) for line in normalized)
               or sum(len(re.findall(r"[А-Яа-яЁё]", line)) for line in normalized)
               > sum(len(re.findall(r"[A-Za-z]", line)) for line in normalized))
    # Уверенное обозначение: поле валюты либо цена в валидной строке товара/итога.
    # Изолированный $ и символ в мусоре/служебных данных ничего не доказывают.
    evidence = []
    for line in normalized:
        if re.search(r"^QR\b|://|www\.|^save\s*/", line, re.I):
            continue
        field = re.fullmatch(r"(?:валюта|currency)\s*[:=]\s*(USD|EUR|\$|€)", line, re.I)
        if field:
            evidence.append((3, "USD" if field[1].upper() in ("USD", "$") else "EUR"))
            continue
        match = re.fullmatch(rf"(?P<label>.*?)\s+(?P<amount>{AMOUNT})\s*(?P<currency>USD|EUR|\$|€)", line, re.I)
        if match is None:
            match = re.fullmatch(rf"(?P<label>.*?)\s+(?P<currency>USD|EUR|\$|€)\s*(?P<amount>{AMOUNT})", line, re.I)
        if not match:
            continue
        label, marker = match["label"].strip(" :;=—"), match["currency"].upper()
        total_line = bool(TOTAL.fullmatch(label) or any(
            (m := pattern.match(label)) and not m["value"].strip(" :;=—") for _, pattern in SUMMARY_LABELS))
        if not total_line and (SERVICE.match(label) or RESTAURANT_SERVICE.match(label) or not plausible_name(label)):
            continue
        if marker in ("USD", "$"):
            if russian and (marker == "$" or not total_line):
                continue
            currency = "USD"
        else:
            currency = "EUR"
        evidence.append((2 if total_line and marker in ("USD", "EUR") else 1 if total_line else 0, currency))
    if evidence:
        return max(evidence, key=lambda item: item[0])[1]
    # Существующий безопасный fallback для русскоязычных рублёвых чеков.
    return "RUB"


def normalize(text: str) -> str:
    text = " ".join(text.replace("\u00a0", " ").replace("\u202f", " ").split())
    # Меняем только пробелы вокруг уже существующего десятичного разделителя.
    return re.sub(r"(?<=\d)\s*([.,])\s*(?=\d{2}(?!\d))", r"\1", text)


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
    return "|" not in name and not re.search(r"\s\d+(?:[.,]\d+)?[^\w%]*$", name)


def rejection_reason(line: str) -> str:
    if "|" in line:
        return "damaged_columns"
    if any(re.search(r"[A-Za-z]", word) and re.search(r"[А-Яа-яЁё]", word)
           for word in re.findall(r"\w+", line)):
        return "mixed_script_name"
    return "implausible_name_quantity_or_price"


def parse_receipt_lines(lines: list[str]) -> RecognizedReceipt:
    items, guest_subtotals, totals, warnings = [], {}, {}, []
    pending_name = None
    pending_summary = None
    current_guest = None
    guest_block_start = 0
    skipped = 0
    prepayment_seen = False
    decisions = {}
    pending_name_line = None
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

    for line_number, raw in enumerate(lines, 1):
        line = CURRENCY_SUFFIX.sub("", normalize(raw)).strip()
        if normalize(raw):
            decisions[line_number] = ParserLineDecision(line_number=line_number, category="rejected_candidate", reason="unparsed_or_incomplete")

        def decide(category, reason, item_index=None):
            if line_number in decisions:
                decisions[line_number] = ParserLineDecision(line_number=line_number, category=category, reason=reason, item_index=item_index)

        if not line:
            decide("ignored_service", "currency_only")
            continue
        if pending_summary:
            kind, guest = pending_summary
            pending_summary = None
            value = summary_amount(line)
            if value is not None:
                set_summary(kind, guest, value)
                decide("ignored_service", "summary_value")
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
            decide("ignored_service", "guest_header")
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
            decide("ignored_service", "guest_subtotal" if kind == "guest" else kind)
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
                or "://" in line or "www." in line.lower() or re.match(r"^save\s*/", line, re.I)
                or re.search(r"\bкол[ .-]*во\b", line, re.I)):
            decide("ignored_service", "service_date_or_url")
            pending_name = None
            continue
        item = None
        used_pending = False
        match = MULTIPLIER.search(line)
        if match:
            used_pending = not line[:match.start()].strip() and bool(pending_name)
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
                    used_pending = not line[:tail.start()].strip(" =—:") and bool(pending_name)
                    total = decimal_value(tail["amount"])
                    item = make_item(name, Decimal(1), None, total)
        if item and restaurant and Decimal(item.quantity) > 1000:
            item = None
            decide("rejected_candidate", "implausible_quantity")
        if item:
            decide("accepted_item", "parsed_item", len(items))
            if used_pending and pending_name_line in decisions:
                decisions[pending_name_line] = ParserLineDecision(line_number=pending_name_line, category="accepted_item", reason="multiline_item_name", item_index=len(items))
            item.guest_id = current_guest
            items.append(item)
            pending_name = None
        elif plausible_name(line) and not re.search(r"\d+[.,]\d{2}\s*$", line):
            # Перенос названия разрешён только на непосредственно следующую строку.
            if pending_name is not None:
                skipped += 1
            pending_name = line
            pending_name_line = line_number
        else:
            skipped += 1
            pending_name = None
            if decisions[line_number].reason != "implausible_quantity":
                decide("rejected_candidate", rejection_reason(line))
    if pending_summary:
        warn("Не все итоговые суммы удалось прочитать. Проверьте чек.")
    diagnostics = ParserDiagnostics(
        ocr_lines=len(decisions), lines=list(decisions.values()),
        accepted_item_lines=sum(row.category == "accepted_item" for row in decisions.values()),
        rejected_candidate_lines=sum(row.category == "rejected_candidate" for row in decisions.values()),
        ignored_service_lines=sum(row.category == "ignored_service" for row in decisions.values()),
    )
    if not items or len(items) > 200:
        error = UnreadableReceipt("No reliable purchase lines found")
        error.parser_diagnostics = diagnostics
        raise error
    receipt_total = totals.get("gross_total")
    if receipt_total is None and not prepayment_seen:
        receipt_total = totals.get("receipt_total", totals.get("amount_due"))
    return RecognizedReceipt(
        readable=True, items=items, receipt_total=receipt_total, currency=currency,
        gross_total=totals.get("gross_total"), prepayment=totals.get("prepayment"),
        amount_due=totals.get("amount_due"), guest_subtotals=guest_subtotals,
        skipped_lines=skipped, parse_warnings=warnings, parser_diagnostics=diagnostics,
    )
