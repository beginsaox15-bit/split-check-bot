"""Геометрия сегментов, координатная дедупликация и простая оценка текста OCR."""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import cv2
import numpy as np
from PIL import Image

LONG_ASPECT_RATIO = 2.5
OVERLAP = .12
MAX_SEGMENTS = 12


@dataclass
class OCRLine:
    text: str
    top: float
    bottom: float
    confidence: float
    good_words: int
    words: int
    good_chars: int
    segment: int = 0
    margin: float = 0
    raw_confidence: float | None = None
    numeric_confidence: float | None = None


def is_long_receipt(size: tuple[int, int]) -> bool:
    return size[1] / max(1, size[0]) >= LONG_ASPECT_RATIO


def estimate_glyph_height(image: Image.Image) -> float:
    ratio = min(1.0, 2400 / max(image.size))
    small = cv2.resize(np.array(image), None, fx=ratio, fy=ratio)
    _, ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    heights = [row[cv2.CC_STAT_HEIGHT] / ratio for row in stats[1:]
               if 3 <= row[cv2.CC_STAT_HEIGHT] <= small.shape[1] * .25
               and 2 <= row[cv2.CC_STAT_WIDTH] < small.shape[1] * .4 and row[cv2.CC_STAT_AREA] >= 5]
    return float(np.median(heights)) if heights else 0.0


def segment_bounds(image: Image.Image, *, force=False) -> list[tuple[int, int]]:
    width, height = image.size
    glyph = estimate_glyph_height(image)
    target = glyph * 50 if glyph else width * 1.8
    target = min(width * 2.6, max(width * 1.3, target))
    target = max(target, height / (MAX_SEGMENTS * (1 - OVERLAP) + OVERLAP))
    chunk = min(height, max(1, round(target)))
    if force and chunk == height and height > width:
        chunk = max(1, round(height * .6))
    if chunk >= height:
        return [(0, height)]
    count = min(MAX_SEGMENTS, max(2, int(np.ceil(height / chunk))))
    # Подстраиваем высоту под число блоков: фактический overlap близок к 12%,
    # а не раздувается до половины блока при почти целочисленном числе сегментов.
    chunk = min(height, int(np.ceil(height / (count * (1 - OVERLAP) + OVERLAP))))
    return [(round(i * (height - chunk) / (count - 1)), round(i * (height - chunk) / (count - 1)) + chunk)
            for i in range(count)]


def line_similarity(left: str, right: str) -> bool:
    normalize = lambda text: ' '.join(re.findall(r'[\w.,]+', text.casefold().replace('ё', 'е')))
    # Разные цифры не объединяем даже у почти одинаковых названий.
    numbers = lambda text: re.findall(r'\d+(?:[.,]\d+)?', text.replace(',', '.'))
    return numbers(left) == numbers(right) and SequenceMatcher(None, normalize(left), normalize(right)).ratio() >= .88


def merge_segments(parts: list[tuple[int, int, list[OCRLine]]]) -> list[OCRLine]:
    merged = []
    for segment, (start, end, lines) in enumerate(parts):
        overlap_end = parts[segment - 1][1] if segment else start
        for line in lines:
            # Отрезанная строка принадлежит соседнему сегменту с запасом overlap.
            tolerance = max(2, (line.bottom - line.top) * .2)
            neighbors = []
            if segment > 0 and line.top <= start + tolerance:
                neighbors += [(other, segment - 1) for other in parts[segment - 1][2]]
            if segment + 1 < len(parts) and line.bottom >= end - tolerance:
                neighbors += [(other, segment + 1) for other in parts[segment + 1][2]]
            # Если сосед не прочитал строку, сохраняем единственное наблюдение.
            # Удалять граничную строку без замены означало бы терять OCR-текст.
            if any(min(line.bottom, other.bottom) - max(line.top, other.top)
                   > .35 * min(line.bottom - line.top, other.bottom - other.top)
                   and (index == 0 or other.top > parts[index][0] + tolerance)
                   and (index == len(parts) - 1 or other.bottom < parts[index][1] - tolerance)
                   for other, index in neighbors):
                continue
            duplicate = None
            if segment and line.top < overlap_end:
                for index in range(len(merged) - 1, -1, -1):
                    old = merged[index]
                    if old.bottom < start:
                        continue
                    center = (line.top + line.bottom) / 2
                    old_center = (old.top + old.bottom) / 2
                    close = abs(center - old_center) <= .5 * max(line.bottom - line.top, old.bottom - old.top)
                    if close and line_similarity(old.text, line.text):
                        duplicate = index
                        break
            if duplicate is None:
                merged.append(line)
            elif (line.margin, line.confidence) > (merged[duplicate].margin, merged[duplicate].confidence):
                merged[duplicate] = line
        # Только порядок физических строк; слова из разных строк не склеиваются.
        merged.sort(key=lambda line: (line.top, line.bottom))
    return merged


PRICE = re.compile(r'\d+(?:[.,]\d{1,2})?\s*(?:RUB|USD|EUR|₽|€|\$)?\s*$', re.I)
SUMMARY = re.compile(r'^(?:и?того|игого|полная\s+сум(?:ма|ми|и)|предоплата|к\s+оплате|total|grand total|amount due)\b', re.I)
SERVICE = re.compile(r'^(?:гость|ость|стол|заказ|официант|наименование|кол[.-]?во|чек|касс|инн|фн|фд|фпд|ндс|скидка|карта|save|qr|receipt|tax|card|cash|qty|date|time)\b', re.I)


def assess_lines(lines: list[OCRLine]) -> dict:
    meaningful = [line for line in lines if line.good_words and line.confidence >= 45]
    prices = [line for line in meaningful if PRICE.search(line.text)]
    summaries = [line for line in meaningful if SUMMARY.match(line.text)]
    candidates = [line for line in prices if not SUMMARY.match(line.text) and not SERVICE.match(line.text)
                  and len(re.findall(r'[А-Яа-яЁёA-Za-z]', line.text)) >= 3]
    garbage = len(lines) - len(meaningful)
    good_words, words = sum(line.good_words for line in lines), sum(line.words for line in lines)
    reliable = len(meaningful) >= 3 and good_words >= 6 and sum(line.good_chars for line in lines) >= 20 and good_words / max(1, words) >= .45
    confidence = sum(line.confidence for line in meaningful) / max(1, len(meaningful))
    return dict(score=8 * len(candidates) + 3 * len(prices) + len(meaningful) + 4 * len(summaries) - 4 * garbage + confidence / 100,
                reliable=reliable, meaningful_lines=len(meaningful), price_lines=len(prices),
                candidate_item_lines=len(candidates), garbage_lines=garbage, summary_lines=len(summaries))
