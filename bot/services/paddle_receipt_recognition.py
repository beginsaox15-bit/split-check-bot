"""PaddleOCR CPU: один переиспользуемый движок в отдельном рабочем потоке."""

import asyncio
import logging
import os
import ssl
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from bot.services.receipt_models import RecognizedReceipt, RecognitionUnavailable, UnreadableReceipt
from bot.services.receipt_parser import normalize, parse_receipt_lines


logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="receipt-ocr")
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        cache = Path(os.getenv("OCR_CACHE_DIR") or Path(__file__).resolve().parents[2] / "work" / "ocr-cache")
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache / "paddlex"))
        os.environ.setdefault("PADDLE_HOME", str(cache / "paddle"))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache))
        os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
        os.environ.setdefault("MODELSCOPE_CACHE", str(cache / "modelscope"))
        cafile = ssl.get_default_verify_paths().cafile
        if cafile:
            os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
        os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
        # Локальные модели не должны требовать сетевого healthcheck.
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        options = {
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "cyrillic_PP-OCRv5_mobile_rec",
            "device": "cpu", "cpu_threads": 2, "enable_mkldnn": False,
            "use_doc_orientation_classify": False, "use_doc_unwarping": False,
            "use_textline_orientation": False, "text_rec_score_thresh": 0.5,
        }
        for variable, option in (("PADDLE_DET_MODEL_DIR", "text_detection_model_dir"),
                                 ("PADDLE_REC_MODEL_DIR", "text_recognition_model_dir")):
            directory = os.getenv(variable, "").strip()
            if directory:
                options[option] = directory
        _engine = PaddleOCR(**options)
    return _engine


def prediction_rows(prediction) -> list[str]:
    """Собирает соседние ячейки одного ряда по координатам, а не по индексу OCR."""
    texts = prediction.get("rec_texts", [])
    scores = prediction.get("rec_scores")
    boxes = prediction.get("rec_boxes")
    if boxes is None:
        polygons = prediction.get("rec_polys")
        if polygons is not None:
            boxes = [[min(p[0] for p in poly), min(p[1] for p in poly),
                      max(p[0] for p in poly), max(p[1] for p in poly)] for poly in polygons]
    if boxes is None:
        return [normalize(text) for index, text in enumerate(texts)
                if scores is None or float(scores[index]) >= 0.5]
    fragments = []
    for index, text in enumerate(texts):
        if scores is not None and float(scores[index]) < 0.5:
            continue
        text = normalize(text)
        if not text:
            continue
        x1, y1, x2, y2 = map(float, boxes[index])
        fragments.append((text, x1, (y1 + y2) / 2, max(1, y2 - y1)))
    fragments.sort(key=lambda fragment: (fragment[2], fragment[1]))
    rows = []
    for fragment in fragments:
        if rows and abs(fragment[2] - rows[-1][0][2]) <= min(fragment[3], rows[-1][0][3]) * 0.5:
            rows[-1].append(fragment)
        else:
            rows.append([fragment])
    return [" ".join(fragment[0] for fragment in sorted(row, key=lambda fragment: fragment[1])) for row in rows]


def _recognize_sync(image: bytes) -> RecognizedReceipt:
    try:
        import numpy as np
        from PIL import Image, ImageOps, UnidentifiedImageError
        try:
            with Image.open(BytesIO(image)) as source:
                # Ограничиваем и декодированное изображение, не только размер файла.
                if source.width * source.height > 25_000_000:
                    raise UnreadableReceipt("Image dimensions exceed limit")
                upright = ImageOps.exif_transpose(source).convert("RGB")
                upright.thumbnail((2400, 2400))
                pixels = np.asarray(upright)[:, :, ::-1].copy()  # PaddleOCR принимает BGR.
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise UnreadableReceipt("Cannot decode image") from None
        engine = _get_engine()
        rows = []
        for prediction in engine.predict(pixels):
            rows.extend(prediction_rows(prediction))
        return parse_receipt_lines(rows)
    except UnreadableReceipt:
        raise
    except Exception as error:
        # Не выводим в лог фото, распознанный текст или содержимое исключения.
        logger.warning("PaddleOCR unavailable: type=%s", type(error).__name__)
        raise RecognitionUnavailable(type(error).__name__) from None


async def recognize_receipt(image: bytes) -> RecognizedReceipt:
    if not image or len(image) > 10 * 1024 * 1024:
        raise UnreadableReceipt("Empty or oversized image")
    return await asyncio.get_running_loop().run_in_executor(_executor, _recognize_sync, image)


if __name__ == "__main__":
    # Явная подготовка весов при развёртывании, без фотографий и платных API.
    try:
        _get_engine()
    except Exception as error:
        raise SystemExit(
            f"PaddleOCR preparation failed ({type(error).__name__}). "
            "Check model files or allow the model download host."
        ) from None
    print("PaddleOCR CPU models are ready.")
