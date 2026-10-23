"""Временные позиции чека: проверка ввода и точные суммы без float."""

import re
from decimal import Decimal, localcontext


def parse_price(text: str | None) -> Decimal | None:
    value = (text or "").strip().replace(",", ".")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
        return None
    price = Decimal(value)
    return price if price > 0 else None


def parse_quantity(text: str | None) -> int | None:
    value = (text or "").strip()
    if not re.fullmatch(r"[0-9]+", value):
        return None
    quantity = int(value)
    return quantity if quantity > 0 else None


def money(value: Decimal) -> str:
    formatted = format(value, "f")
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def receipt_lines(items: list[dict]) -> tuple[list[str], str]:
    # Точность учитывает длину ввода, количество и сложение всех позиций.
    precision = sum(
        len(format(Decimal(item["price"]), "f")) + len(str(item["quantity"]))
        for item in items
    ) + 1
    with localcontext() as context:
        context.prec = max(28, precision)
        total = Decimal(0)
        lines = []
        for item in items:
            subtotal = Decimal(item["price"]) * item["quantity"]
            total += subtotal
            lines.append(f'{item["name"]} × {item["quantity"]} — {money(subtotal)} ₽')
        return lines, money(total)
