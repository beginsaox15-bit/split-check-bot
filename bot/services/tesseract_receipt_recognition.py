"""Tesseract rus+eng: лёгкая подготовка Pillow и существующий receipt parser."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pytesseract
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

from bot.services.receipt_models import (
    RecognizedReceipt, RecognitionUnavailable, TesseractLanguagesUnavailable,
    TesseractNotInstalled, UnparsedReceipt, UnreadableReceipt,
)
from bot.services.receipt_parser import parse_receipt_lines


_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="receipt-ocr")
OCR_LANGUAGES = "rus+eng"


def prepare_image(image: bytes) -> Image.Image:
    """Учитывает EXIF, осветляет фон, повышает контраст и умеренно масштабирует."""
    try:
        with Image.open(BytesIO(image)) as source:
            if source.width * source.height > 25_000_000:
                raise UnreadableReceipt("Image dimensions exceed limit")
            upright = ImageOps.exif_transpose(source)
            with upright:
                grayscale = ImageOps.grayscale(upright)
            with grayscale:
                contrasted = ImageOps.autocontrast(grayscale)
            with contrasted:
                prepared = ImageEnhance.Contrast(contrasted).enhance(1.4)
        # Не более 2x; длинная сторона ограничена даже у очень узких чеков.
        scale = min(2.0, max(1.0, 1200 / prepared.width), 3600 / max(prepared.size))
        if scale != 1:
            with prepared:
                return prepared.resize(
                    (max(1, round(prepared.width * scale)), max(1, round(prepared.height * scale))),
                    Image.Resampling.LANCZOS,
                )
        return prepared
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise UnreadableReceipt("Cannot decode image") from error


def _recognize_sync(image: bytes) -> RecognizedReceipt:
    with prepare_image(image) as prepared:
        try:
            # Tesseract может молча работать с одним языком: проверяем оба явно.
            languages = set(pytesseract.get_languages(config=""))
            if not {"rus", "eng"}.issubset(languages):
                raise TesseractLanguagesUnavailable("Install tesseract-ocr-rus and tesseract-ocr-eng")
            raw_text = pytesseract.image_to_string(
                prepared, lang=OCR_LANGUAGES, config="--psm 6", timeout=30,
            ).strip()
        except (pytesseract.TesseractNotFoundError, FileNotFoundError) as error:
            raise TesseractNotInstalled("Install Tesseract using scripts/install_tesseract.sh") from error
        except (pytesseract.TesseractError, RuntimeError, OSError) as error:
            # Сохраняем исходную причину для traceback в receipt_recognition.
            raise RecognitionUnavailable(type(error).__name__) from error
    if not raw_text:
        raise UnreadableReceipt("No recognized text")
    try:
        result = parse_receipt_lines(raw_text.splitlines())
    except UnreadableReceipt:
        raise UnparsedReceipt(raw_text) from None
    return result.model_copy(update={"raw_text": raw_text})


async def recognize_receipt(image: bytes) -> RecognizedReceipt:
    if not image or len(image) > 10 * 1024 * 1024:
        raise UnreadableReceipt("Empty or oversized image")
    return await asyncio.get_running_loop().run_in_executor(_executor, _recognize_sync, image)
