"""Safe synthetic header regression, not a photograph of the user's receipt."""

import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent


def main():
    lines = (ROOT / "long_restaurant_guest_receipt.txt").read_text().splitlines()
    lines[1] = "Зап: Зап Cron 18"
    font = ImageFont.truetype(os.getenv("TESSERACT_SMOKE_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), 24)
    with Image.new("RGB", (720, 60 * len(lines) + 80), "white") as image:
        draw = ImageDraw.Draw(image)
        for index, text in enumerate(lines):
            draw.text((20, 30 + 60 * index), text, font=font, fill="black")
        image.save(ROOT / "long_restaurant_review.png")
    expected = json.loads((ROOT / "expected/long_restaurant_guest_receipt.json").read_text())
    expected.update(name="long_restaurant_review", image="long_restaurant_review.png")
    expected["tags"].append("targeted_review")
    (ROOT / "expected/long_restaurant_review.json").write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
