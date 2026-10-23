"""Выбор backend: локальный PaddleOCR по умолчанию, без облачного fallback."""

import os

from bot.services.receipt_models import (
    MissingAPIKey, RecognizedReceipt, RecognitionUnavailable, UnreadableReceipt,
)


def recognition_backend() -> str:
    return os.getenv("RECEIPT_OCR_BACKEND", "paddleocr").strip().lower() or "paddleocr"


async def recognize_receipt(image: bytes) -> RecognizedReceipt:
    backend = recognition_backend()
    if backend == "paddleocr":
        from bot.services.paddle_receipt_recognition import recognize_receipt as recognize
    elif backend == "openai":
        try:
            from bot.services.openai_receipt_recognition import recognize_receipt as recognize
        except ImportError:
            raise RecognitionUnavailable("Optional OpenAI dependencies are not installed") from None
    else:
        raise RecognitionUnavailable("Unknown OCR backend")
    return await recognize(image)
