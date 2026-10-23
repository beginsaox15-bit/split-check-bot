"""Локальный Tesseract — единственный backend; внешние API не используются."""

import os

from bot.services.receipt_models import (
    RecognizedReceipt, RecognitionUnavailable,
)
from bot.services.tesseract_receipt_recognition import recognize_receipt as recognize_local


def recognition_backend() -> str:
    return os.getenv("RECEIPT_OCR_BACKEND", "tesseract").strip().lower() or "tesseract"


async def recognize_receipt(image: bytes) -> RecognizedReceipt:
    if recognition_backend() != "tesseract":
        raise RecognitionUnavailable("Set RECEIPT_OCR_BACKEND=tesseract")
    return await recognize_local(image)
