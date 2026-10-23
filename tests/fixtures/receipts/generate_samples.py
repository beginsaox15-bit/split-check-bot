"""Воспроизводимые синтетические чеки. Никаких реальных персональных данных."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent


def render(lines):
    font = ImageFont.truetype(os.getenv("TESSERACT_SMOKE_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), 36)
    image = Image.new("RGB", (1100, 90 + 62 * len(lines)), "white")
    ImageDraw.Draw(image).multiline_text((45, 35), "\n".join(lines), fill="black", font=font, spacing=24)
    return image


def expected(name, tags, items, total):
    data = {"name": name, "image": name + ".png", "tags": tags,
            "expected_items": [{"name": title, "quantity": "1", "total_price": price} for title, price in items],
            "expected_total": total, "currency": "RUB"}
    (ROOT / "expected" / (name + ".json")).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    (ROOT / "expected").mkdir(exist_ok=True)
    items = [("Бургер", "850.00"), ("Коктейль", "650.00"), ("Чай", "330.00"),
             ("Салат", "570.00"), ("Пицца", "1250.00"), ("Паста", "850.00")]
    lines = ["ЧЕК (ТЕСТОВЫЙ)"] + [f"{title} {price}" for title, price in items] + ["ИТОГО 4500.00 RUB"]
    with render(lines) as image:
        image.save(ROOT / "synthetic_restaurant_01.png")
        expected("synthetic_restaurant_01", ["straight", "restaurant"], items, "4500.00")
        pixels = np.array(image)
        h, w = pixels.shape[:2]
        transform = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]),
            np.float32([[150, 150], [w + 60, 80], [w + 140, h + 140], [80, h + 70]]),
        )
        tilted = cv2.warpPerspective(pixels, transform, (w + 240, h + 230), borderValue=(45, 45, 45))
        shade = np.linspace(.55, .83, tilted.shape[1])[None, :, None]
        dark = np.clip(tilted * shade, 0, 255).astype(np.uint8)
        with Image.fromarray(dark) as dark_image:
            dark_image.save(ROOT / "synthetic_restaurant_angled_dark.png")
        expected("synthetic_restaurant_angled_dark", ["angled", "dark", "restaurant"], items, "4500.00")
    shop_items = [("Молоко", "100.00"), ("Молоко", "100.00"), ("Хлеб", "60.00")]
    with render(["ЧЕК (ТЕСТОВЫЙ)"] + [f"{name} {price}" for name, price in shop_items]
                + ["СКИДКА 10.00", "ИТОГО 250.00 RUB"]) as image:
        image.save(ROOT / "synthetic_shop_discount_duplicates.png")
    expected("synthetic_shop_discount_duplicates", ["straight", "shop", "discount", "duplicates"], shop_items, "250.00")


if __name__ == "__main__":
    main()
