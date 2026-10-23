"""Локальный rus+eng OCR: варианты preprocessing и PSM, проверка качества текста."""

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytesseract
from PIL import Image

from bot.services.receipt_models import (
    LowQualityReceipt, RecognizedReceipt, RecognitionUnavailable, TesseractLanguagesUnavailable,
    TesseractNotInstalled, UnparsedReceipt, UnreadableReceipt,
)
from bot.services.receipt_parser import parse_receipt_lines
from bot.services.receipt_preprocessing import adaptive_image, prepare_image


logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="receipt-ocr")
OCR_LANGUAGES = "rus+eng"
PSM_MODES = (6, 4, 11)
OCR_BUDGET_SECONDS = 45


@dataclass
class OCRCandidate:
    text: str
    preprocessing: str
    psm: int
    score: float
    reliable: bool
    lines: int


def word_rows(data: dict, psm: int) -> list[list[int]]:
    indices = [i for i, word in enumerate(data["text"]) if word.strip()]
    if psm != 11 or not {"top", "height", "left"}.issubset(data):
        groups = {}
        for i in indices:
            key = tuple(data[name][i] for name in ("page_num", "block_num", "par_num", "line_num"))
            groups.setdefault(key, []).append(i)
        return list(groups.values())
    # PSM 11 создаёт отдельные блоки для слов одной строки. Собираем их по
    # координатам, чтобы не выбирать его только из-за раздутого числа блоков.
    rows = []
    for i in sorted(indices, key=lambda i: data["top"][i] + data["height"][i] / 2):
        center = data["top"][i] + data["height"][i] / 2
        height = max(1, data["height"][i])
        row = next((row for row in reversed(rows)
                    if abs(row[0] - center) <= .45 * max(row[1], height)), None)
        if row is None:
            rows.append([center, height, [i]])
        else:
            row[2].append(i)
    return [sorted(row[2], key=lambda i: data["left"][i]) for row in rows]


def candidate_from_data(data: dict, preprocessing: str, psm: int) -> OCRCandidate:
    rows = word_rows(data, psm)
    line_for_word = {i: row_number for row_number, row in enumerate(rows) for i in row}
    good_lines = set()
    confidences = []
    good_chars = 0
    amounts = 0
    words = 0
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        key = line_for_word[i]
        words += 1
        confidence = float(data["conf"][i])
        meaningful = re.findall(r"[A-Za-zА-Яа-яЁё0-9]", word)
        plausible = len(meaningful) >= 2 and len(meaningful) / len(word) >= .5
        if plausible and confidence >= 45:
            confidences.append(confidence)
            good_chars += min(len(meaningful), 24)
            good_lines.add(key)
            amounts += bool(re.fullmatch(r"\d+[.,]\d{2}", word))
    text = "\n".join(" ".join(data["text"][i].strip() for i in row) for row in rows)
    mean_confidence = sum(confidences) / max(1, len(confidences))
    reliable = (len(good_lines) >= 3 and len(confidences) >= 6 and good_chars >= 20
                and len(confidences) / max(1, words) >= .45)
    score = good_chars * mean_confidence / 100 + 5 * amounts
    return OCRCandidate(text, preprocessing, psm, score, reliable, len(rows))


def orient_image(prepared: Image.Image) -> Image.Image:
    try:
        orientation = pytesseract.image_to_osd(prepared, output_type=pytesseract.Output.DICT, timeout=5)
        rotation = int(orientation.get("rotate", 0))
        if rotation in (90, 180, 270) and float(orientation.get("orientation_conf", 0)) >= 5:
            logger.info("OCR orientation: rotation=%s", rotation)
            return prepared.rotate(-rotation, expand=True, fillcolor=255)
    except (pytesseract.TesseractError, RuntimeError, OSError):
        logger.debug("OCR orientation unavailable; preserving EXIF orientation")
    return prepared.copy()


def extract_text(image: bytes) -> OCRCandidate:
    with prepare_image(image) as prepared:
        languages = set(pytesseract.get_languages(config=""))
        if not {"rus", "eng"}.issubset(languages):
            raise TesseractLanguagesUnavailable("Install tesseract-ocr-rus and tesseract-ocr-eng")
        logger.info("OCR preprocessing: %s", prepared.info.get("preprocessing"))
        with orient_image(prepared) as oriented, adaptive_image(oriented) as binary:
            candidates = []
            last_error = None
            deadline = time.monotonic() + OCR_BUDGET_SECONDS
            for name, variant in (("grayscale", oriented), ("adaptive", binary)):
                for psm in PSM_MODES:
                    remaining = deadline - time.monotonic()
                    if remaining < 1:
                        break
                    try:
                        data = pytesseract.image_to_data(
                            variant, lang=OCR_LANGUAGES, config=f"--psm {psm}",
                            output_type=pytesseract.Output.DICT, timeout=min(12, remaining),
                        )
                        candidate = candidate_from_data(data, name, psm)
                        candidates.append(candidate)
                        logger.info("OCR candidate: preprocessing=%s psm=%s lines=%s score=%.1f reliable=%s",
                                    name, psm, candidate.lines, candidate.score, candidate.reliable)
                    except (pytesseract.TesseractNotFoundError, FileNotFoundError):
                        raise
                    except (pytesseract.TesseractError, RuntimeError) as error:
                        last_error = error
                        logger.info("OCR candidate failed: preprocessing=%s psm=%s type=%s", name, psm, type(error).__name__)
            if not candidates:
                raise RecognitionUnavailable("No OCR configuration completed") from last_error
            best = max(candidates, key=lambda candidate: (candidate.reliable, candidate.score))
            logger.info("OCR selected: preprocessing=%s psm=%s lines=%s score=%.1f reliable=%s",
                        best.preprocessing, best.psm, best.lines, best.score, best.reliable)
            return best


def _recognize_sync(image: bytes) -> RecognizedReceipt:
    try:
        best = extract_text(image)
    except (pytesseract.TesseractNotFoundError, FileNotFoundError) as error:
        raise TesseractNotInstalled("Install Tesseract using scripts/install_tesseract.sh") from error
    except OSError as error:
        raise RecognitionUnavailable(type(error).__name__) from error
    if not best.reliable:
        raise LowQualityReceipt(best.text)
    try:
        result = parse_receipt_lines(best.text.splitlines())
    except UnreadableReceipt:
        raise UnparsedReceipt(best.text) from None
    return result.model_copy(update={"raw_text": best.text})


async def recognize_receipt(image: bytes) -> RecognizedReceipt:
    if not image or len(image) > 10 * 1024 * 1024:
        raise UnreadableReceipt("Empty or oversized image")
    return await asyncio.get_running_loop().run_in_executor(_executor, _recognize_sync, image)
