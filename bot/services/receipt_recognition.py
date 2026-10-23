"""Локальный Tesseract — единственный backend; внешние API не используются."""

import logging
import os
import shutil
import sys

import pytesseract

from bot.services.receipt_models import (
    RecognizedReceipt, RecognitionUnavailable, TesseractLanguagesUnavailable, UnreadableReceipt,
)
from bot.services.tesseract_receipt_recognition import recognize_receipt as recognize_local


logger = logging.getLogger(__name__)


def log_ocr_runtime() -> None:
    """Диагностика в процессе бота; отсутствие OCR не мешает ручному вводу."""
    logger.info(
        "OCR runtime: python=%s backend=%s pytesseract=%s executable=%s TESSDATA_PREFIX=%s",
        sys.executable, recognition_backend(), pytesseract.__version__,
        shutil.which(pytesseract.pytesseract.tesseract_cmd), os.getenv("TESSDATA_PREFIX"),
    )
    try:
        version = pytesseract.get_tesseract_version()
        languages = pytesseract.get_languages(config="")
        logger.info("OCR runtime: tesseract=%s languages=%s", version, languages)
        if not {"rus", "eng"}.issubset(languages):
            raise TesseractLanguagesUnavailable("Tesseract requires both rus and eng")
    except Exception:
        logger.exception("OCR runtime check failed; manual input remains available")


def recognition_backend() -> str:
    return os.getenv("RECEIPT_OCR_BACKEND", "tesseract").strip().lower() or "tesseract"


async def recognize_receipt(image: bytes) -> RecognizedReceipt:
    backend = recognition_backend()
    try:
        if backend != "tesseract":
            raise RecognitionUnavailable("Set RECEIPT_OCR_BACKEND=tesseract")
        return await recognize_local(image)
    except UnreadableReceipt:
        # Пустое фото/текст и ручной ввод после ошибки parser — штатные ветки.
        raise
    except Exception:
        # Полный traceback, включая исходный stderr Tesseract, только для разработчика.
        # Байты фото, OCR-текст и окружение целиком в лог не передаём.
        logger.exception(
            "Receipt recognition failed: backend=%s executable=%s TESSDATA_PREFIX=%s",
            backend, shutil.which(pytesseract.pytesseract.tesseract_cmd), os.getenv("TESSDATA_PREFIX"),
        )
        raise
