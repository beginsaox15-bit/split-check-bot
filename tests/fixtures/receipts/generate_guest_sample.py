"""Обезличенный длинный ресторанный чек: три гостя и предоплата."""

import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
GROUPS = [
    ("1", "11150.00", [("Бургер", "2", "850.00", "1700.00"),
                       ("Стейк", "3", "2500.00", "7500.00"), ("Салат", "1", "1950.00", "1950.00")]),
    ("2", "29082.50", [("Морепродукты", "2", "6500.00", "13000.00"),
                       ("Паста", "3", "1250.00", "3750.00"), ("Суп", "2", "1100.00", "2200.00"),
                       ("Десерт", "4", "850.00", "3400.00"), ("Лимонад", "1", "1670.00", "1670.00"),
                       ("Чай", "2", "330.00", "660.00"), ("Кофе", "3", "570.00", "1710.00"),
                       ("Ассорти", "1", "2692.50", "2692.50")]),
    ("3", "1670.00", [("Кофе", "1", "570.00", "570.00"), ("Чай", "1", "330.00", "330.00"),
                      ("Десерт", "1", "770.00", "770.00")]),
]


def main():
    lines = ["ЧЕК (ТЕСТОВЫЙ)", "Стол 18", "Заказ 123", "Официант", "01.01.2026 19:30"]
    items = []
    for guest, subtotal, dishes in GROUPS:
        lines += [f"Гость {guest}", "Наименование Кол-во Сумма"]
        for name, quantity, unit, total in dishes:
            lines.append(f"{name} {quantity} {unit} {total}")
            items.append({"name": name, "quantity": quantity, "total_price": total})
        lines.append(f"Итого к оплате Гость {guest} {subtotal}")
    lines += ["Полная сумма 41902.50 RUB", "Предоплата 7000.00 RUB", "ИТОГО К ОПЛАТЕ 34902.50 RUB"]
    (ROOT / "long_restaurant_guest_receipt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    font = ImageFont.truetype(os.getenv("TESSERACT_SMOKE_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), 32)
    with Image.new("RGB", (1200, 80 + 52 * len(lines)), "white") as image:
        ImageDraw.Draw(image).multiline_text((40, 30), "\n".join(lines), font=font, fill="black", spacing=20)
        image.save(ROOT / "long_restaurant_guest_receipt.png")
    (ROOT / "expected/long_restaurant_guest_receipt.json").write_text(json.dumps({
        "name": "long_restaurant_guest_receipt", "image": "long_restaurant_guest_receipt.png",
        "tags": ["long_restaurant_guest_receipt", "long", "restaurant", "guests", "prepayment"],
        "expected_items": items, "expected_total": "41902.50", "currency": "RUB",
        "expected_gross_total": "41902.50", "expected_prepayment": "7000.00",
        "expected_amount_due": "34902.50",
        "expected_guest_subtotals": {guest: total for guest, total, _ in GROUPS},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
