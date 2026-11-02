"""Локальный rus+eng OCR: варианты preprocessing и PSM, проверка качества текста."""

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from collections import Counter
from math import floor

import pytesseract
from PIL import Image

from bot.services.receipt_models import (
    LowQualityReceipt, RecognizedReceipt, RecognitionUnavailable, TesseractLanguagesUnavailable,
    TesseractNotInstalled, UnparsedReceipt, UnreadableReceipt,
)
from bot.services.receipt_parser import parse_receipt_lines
from bot.services.receipt_preprocessing import adaptive_image, prepare_image, enhance_image
from bot.services.receipt_ocr_layout import OCRLine, OVERLAP, is_long_receipt, segment_bounds, merge_segments, assess_lines, estimate_glyph_height


logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="receipt-ocr")
OCR_LANGUAGES = "rus+eng"
PSM_MODES = (6, 4, 11)
OCR_BUDGET_SECONDS = 45
SEGMENT_BUDGET_SECONDS = 60


@dataclass
class OCRCandidate:
    text: str
    preprocessing: str
    psm: int
    score: float
    reliable: bool
    lines: int
    positioned_lines: list[OCRLine] = field(default_factory=list)
    strategy: str = "full"
    segment_count: int = 1
    overlap: float = 0
    psms: list[int] = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    elapsed_seconds: float = 0
    segment_details: list[dict] = field(default_factory=list)
    alternatives: list = field(default_factory=list)
    complete: bool = True
    orientation_rotation: int = 0
    geometry_size: tuple[int, int] | None = None
    coordinate_scale: float = 1


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


def make_candidate(lines, preprocessing, psm):
    quality = assess_lines(lines)
    return OCRCandidate("\n".join(line.text for line in lines), preprocessing, psm,
                        quality["score"], quality["reliable"], len(lines),
                        positioned_lines=lines, psms=[psm], quality=quality)


def candidate_from_data(data: dict, preprocessing: str, psm: int) -> OCRCandidate:
    lines = []
    for row_number, row in enumerate(word_rows(data, psm)):
        tokens, good, chars, confidences, numeric = [], [], 0, [], []
        for i in row:
            word = data["text"][i].strip()
            tokens.append(word)
            confidence = float(data["conf"][i])
            confidences.append(max(0, confidence))
            if re.search(r"\d", word):
                numeric.append(max(0, confidence))
            meaningful = re.findall(r"[A-Za-zА-Яа-яЁё0-9]", word)
            if len(meaningful) >= 2 and len(meaningful) / max(1, len(word)) >= .5 and confidence >= 45:
                good.append(confidence)
                chars += min(len(meaningful), 24)
        top = min(data["top"][i] for i in row) if "top" in data else row_number * 30
        bottom = max(data["top"][i] + data["height"][i] for i in row) if "height" in data else top + 20
        lines.append(OCRLine(" ".join(tokens), top, bottom, sum(good) / max(1, len(good)), len(good), len(tokens), chars,
                             raw_confidence=sum(confidences) / len(confidences),
                             numeric_confidence=min(numeric) if numeric else None))
    return make_candidate(lines, preprocessing, psm)


def orient_image(prepared: Image.Image) -> Image.Image:
    try:
        orientation = pytesseract.image_to_osd(prepared, output_type=pytesseract.Output.DICT, timeout=5)
        rotation = int(orientation.get("rotate", 0))
        if rotation in (90, 180, 270) and float(orientation.get("orientation_conf", 0)) >= 5:
            logger.info("OCR orientation: rotation=%s", rotation)
            result = prepared.rotate(-rotation, expand=True, fillcolor=255)
            result.info["orientation_rotation"] = rotation
            return result
    except (pytesseract.TesseractError, RuntimeError, OSError):
        logger.debug("OCR orientation unavailable; preserving EXIF orientation")
    result = prepared.copy()
    result.info["orientation_rotation"] = 0
    return result


def recognize_variants(prepared, deadline, *, scope="full", modes=PSM_MODES):
    candidates, last_error = [], None
    with adaptive_image(prepared) as binary:
        for name, variant in (("grayscale", prepared), ("adaptive", binary)):
            variant.info.update(ocr_scope=scope, ocr_preprocessing=name)
            for psm in modes:
                remaining = deadline - time.monotonic()
                if remaining < 1:
                    break
                try:
                    data = pytesseract.image_to_data(variant, lang=OCR_LANGUAGES, config=f"--psm {psm}",
                                                    output_type=pytesseract.Output.DICT, timeout=min(12, remaining))
                    candidate = candidate_from_data(data, name, psm)
                    candidates.append(candidate)
                    logger.info("OCR candidate: scope=%s preprocessing=%s psm=%s lines=%s score=%.1f reliable=%s", scope, name, psm, candidate.lines, candidate.score, candidate.reliable)
                except (pytesseract.TesseractNotFoundError, FileNotFoundError):
                    raise
                except (pytesseract.TesseractError, RuntimeError) as error:
                    last_error = error
                    logger.info("OCR candidate failed: scope=%s preprocessing=%s psm=%s type=%s", scope, name, psm, type(error).__name__)
    if not candidates:
        raise RecognitionUnavailable("No OCR configuration completed") from last_error
    return max(candidates, key=lambda item: (item.reliable, item.score) if scope == "full" else (True, item.score))


def choose_strategy(full, segmented):
    # При равенстве остаётся полный проход; неполная обработка не выигрывает.
    # Дробная добавка confidence не оправдывает переключение при тех же строках.
    if segmented is not None and segmented.complete and (not full.complete or (segmented.reliable, floor(segmented.score)) > (full.reliable, floor(full.score))):
        return segmented
    return full


def recognize_segments(geometry, deadline, modes):
    bounds = segment_bounds(geometry, force=True)
    glyph = estimate_glyph_height(geometry)
    max_scale = min(3.0, max(2.0, 40 / glyph)) if glyph else 2.0
    parts, details, psms, variants = [], [], [], []
    start_time = time.monotonic()
    for index, (top, bottom) in enumerate(bounds):
        remaining = deadline - time.monotonic()
        if remaining < 1:
            return None
        segment_deadline = time.monotonic() + remaining / (len(bounds) - index)
        with geometry.crop((0, top, geometry.width, bottom)) as crop, enhance_image(crop, max_scale=max_scale) as prepared:
            try:
                best = recognize_variants(prepared, segment_deadline, scope=f"segment_{index}", modes=modes)
            except RecognitionUnavailable:
                # Нельзя выдавать пропущенный участок за полный сегментный результат.
                logger.info("Segment OCR incomplete: index=%s", index)
                return None
            scale = prepared.height / (bottom - top)
            lines = [replace(line, top=top + line.top / scale, bottom=top + line.bottom / scale,
                             segment=index, margin=min(line.top, prepared.height - line.bottom) / scale)
                     for line in best.positioned_lines]
            parts.append((top, bottom, lines))
            psms.append(best.psm)
            variants.append(best.preprocessing)
            details.append(dict(index=index, top=top, bottom=bottom, psm=best.psm, preprocessing=best.preprocessing, lines=best.lines))
    merged = merge_segments(parts)
    candidate = make_candidate(merged, variants[0] if len(set(variants)) == 1 else "mixed", Counter(psms).most_common(1)[0][0])
    return replace(candidate, strategy="segmented", segment_count=len(bounds), overlap=OVERLAP if len(bounds) > 1 else 0,
                   psms=psms, segment_details=details, elapsed_seconds=time.monotonic() - start_time)


def extract_text(image: bytes, *, strategy="auto") -> OCRCandidate:
    if strategy not in ("auto", "full", "segmented", "compare"):
        raise ValueError("Unknown OCR strategy")
    start = time.monotonic()
    with prepare_image(image, geometry_only=True) as geometry, enhance_image(geometry) as prepared:
        languages = set(pytesseract.get_languages(config=""))
        if not {"rus", "eng"}.issubset(languages):
            raise TesseractLanguagesUnavailable("Install tesseract-ocr-rus and tesseract-ocr-eng")
        logger.info("OCR preprocessing: %s", prepared.info.get("preprocessing"))
        with orient_image(prepared) as oriented:
            t = time.monotonic()
            try:
                full = recognize_variants(oriented, t + OCR_BUDGET_SECONDS)
            except RecognitionUnavailable:
                if strategy == "full" or not (is_long_receipt(oriented.size) or strategy in ("compare", "segmented")):
                    raise
                logger.info("Full OCR unavailable; trying smaller segments")
                full = replace(make_candidate([], "grayscale", 6), complete=False)
            full.elapsed_seconds = time.monotonic() - t
            rotation = oriented.info.get("orientation_rotation", 0)
            oriented_height = oriented.height
        segmented = None
        with geometry.rotate(-rotation, expand=True, fillcolor=255) as upright:
            full.geometry_size = upright.size
            full.coordinate_scale = oriented_height / upright.height
            full.orientation_rotation = rotation
            if strategy in ("segmented", "compare") or (strategy == "auto" and is_long_receipt(upright.size)):
                modes = tuple(dict.fromkeys((full.psm, *PSM_MODES)))
                segmented = recognize_segments(upright, time.monotonic() + SEGMENT_BUDGET_SECONDS, modes)
                if segmented is not None:
                    segmented.geometry_size = upright.size
                    segmented.orientation_rotation = rotation
        selected = segmented if strategy == "segmented" and segmented is not None else choose_strategy(full, segmented)
        if not selected.complete:
            raise RecognitionUnavailable("Neither OCR strategy completed")
        alternatives = [full] + ([segmented] if segmented is not None else [])
        selected = replace(selected, alternatives=alternatives, elapsed_seconds=time.monotonic() - start)
        logger.info("OCR selected: preprocessing=%s psm=%s lines=%s score=%.1f reliable=%s strategy=%s segments=%s overlap=%.2f elapsed=%.2f",
                    selected.preprocessing, selected.psm, selected.lines, selected.score, selected.reliable, selected.strategy,
                    selected.segment_count, selected.overlap, time.monotonic() - start)
        return selected


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
    except UnreadableReceipt as error:
        raise UnparsedReceipt(best.text, getattr(error, "parser_diagnostics", None)) from None
    from bot.services.receipt_line_review import review_suspicious_lines
    result = review_suspicious_lines(image, best, result)
    if not result.items:
        raise UnparsedReceipt(best.text, result.parser_diagnostics)
    logger.info("OCR result: strategy=%s segments=%s overlap=%.2f psms=%s ocr_lines=%s candidate_item_lines=%s accepted_items=%s rejected_candidates=%s elapsed=%.2f",
                best.strategy, best.segment_count, best.overlap, best.psms, best.lines,
                best.quality.get("candidate_item_lines", 0), len(result.items),
                result.parser_diagnostics.rejected_candidate_lines if result.parser_diagnostics else None, best.elapsed_seconds)
    return result.model_copy(update={"raw_text": best.text})


async def recognize_receipt(image: bytes) -> RecognizedReceipt:
    if not image or len(image) > 10 * 1024 * 1024:
        raise UnreadableReceipt("Empty or oversized image")
    return await asyncio.get_running_loop().run_in_executor(_executor, _recognize_sync, image)
