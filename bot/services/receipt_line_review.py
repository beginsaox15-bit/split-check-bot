"""Bounded local re-OCR of doubtful accepted lines. Never reconstruct missing goods.

Coordinates refer to the same rectified/deskewed original pixels as the main OCR.
The original text remains untouched. Totals never influence a replacement price.
"""

import logging
import re
import time
from collections import Counter
from decimal import Decimal
from difflib import SequenceMatcher
from statistics import median

import pytesseract

from bot.services.receipt_models import LineReviewDiagnostics, UnreadableReceipt
from bot.services.receipt_parser import (
    AMOUNT, CURRENCY_SUFFIX, SUMMARY_LABELS, decimal_value, normalize, parse_receipt_lines,
)
from bot.services.receipt_preprocessing import prepare_image, enhance_image, adaptive_image

logger = logging.getLogger(__name__)
REVIEW_BUDGET_SECONDS = 20
MAX_REREAD_ITEMS = 5
REREAD_PSMS = (7, 13, 6)
# Anchored header grammar: no matching "Салат", "Столичный салат", etc.
SERVICE_HEADER = re.compile(
    r"^(?:(?:зап|зал|стол|заказ|[cс][rг][oо][nп])\b[\s:;№#—\-]*"
    r"(?:\d+(?:[.,]0{1,2})?[\s:;№#—\-]*)?){1,5}$", re.I)


def service_name(name):
    return bool(SERVICE_HEADER.fullmatch(name.strip()))


def confidence(lines):
    values = [line.raw_confidence if line.raw_confidence is not None else line.confidence for line in lines]
    return sum(values) / len(values) if values else None


def item_source_lines(receipt, candidate, index):
    if receipt.parser_diagnostics is None:
        return []
    return [candidate.positioned_lines[row.line_number - 1]
            for row in receipt.parser_diagnostics.lines
            if row.category == "accepted_item" and row.item_index == index
            and 0 < row.line_number <= len(candidate.positioned_lines)]


def suspicion_reasons(item, neighbours, lines):
    reasons = []
    if service_name(item.name):
        reasons.append("service_header")
    total = Decimal(item.total_price)
    price = total / Decimal(item.quantity)
    peers = [Decimal(other.total_price) / Decimal(other.quantity) for other in neighbours
             if other is not item and other.guest_id == item.guest_id]
    # An expensive item alone is never enough. Require at least three neighbours.
    if len(peers) >= 3:
        typical = median(peers)
        if price > typical * 4 and price - typical > 500:
            reasons.append("price_outlier")
    consistent = item.unit_price is not None and abs(Decimal(item.unit_price) * Decimal(item.quantity) - total) <= Decimal("0.02")
    if item.unit_price is not None and not consistent:
        reasons.append("arithmetic_conflict")
    # Fractional quantities/discounts can legitimately produce arbitrary cents.
    if total % 1 not in (Decimal(0), Decimal("0.50")) and Decimal(item.quantity) % 1 == 0 and not consistent:
        reasons.append("unusual_cents")
    conf = confidence(lines)
    numeric = [line.numeric_confidence for line in lines if line.numeric_confidence is not None]
    if conf is not None and (conf < 55 or (numeric and min(numeric) < 45)):
        reasons.append("low_confidence")
    if conf is not None and conf < 30:
        reasons.append("very_low_confidence")
    text = " ".join(line.text for line in lines)
    if text and (len(re.findall(r"[^\w\s.,₽$€×=()\-]", text)) >= 3
                 or re.search(r"[а-яё][a-z]|[a-z][а-яё]", item.name, re.I)):
        reasons.append("damaged_structure")
    return reasons


def is_suspicious(reasons):
    return bool({"service_header", "arithmetic_conflict", "low_confidence", "very_low_confidence"} & set(reasons)) or len(reasons) >= 2


def _parse_reread(lines, original):
    try:
        receipt = parse_receipt_lines([f"Гость {original.guest_id or '0'}", *(line.text for line in lines)])
    except UnreadableReceipt:
        return None
    if len(receipt.items) != 1:
        return None
    item = receipt.items[0].model_copy(update={"guest_id": original.guest_id})
    if service_name(item.name):
        return None
    similarity = SequenceMatcher(None, re.sub(r"\W", "", original.name.casefold()),
                                 re.sub(r"\W", "", item.name.casefold())).ratio()
    return item if similarity >= .78 else None


def reread_crop(geometry, candidate, source, original, deadline, index):
    """Read only the source row band; a clearer neighbouring item cannot replace it."""
    from bot.services.tesseract_receipt_recognition import candidate_from_data, OCR_LANGUAGES

    scale = candidate.coordinate_scale
    top, bottom = min(line.top for line in source) / scale, max(line.bottom for line in source) / scale
    if not 0 <= top < bottom <= geometry.height:
        return []
    height = bottom - top
    y0, y1 = max(0, int(top - height * .7)), min(geometry.height, int(bottom + height * .7 + 1))
    results = []
    with geometry.crop((0, y0, geometry.width, y1)) as crop, enhance_image(crop, requested_scale=4) as prepared, adaptive_image(prepared) as binary:
        factor = prepared.height / (y1 - y0)
        for variant_name, variant in (("grayscale", prepared), ("adaptive", binary)):
            variant.info.update(ocr_scope=f"reread_{index}", ocr_preprocessing=variant_name)
            for psm in REREAD_PSMS:
                remaining = deadline - time.monotonic()
                if remaining < .5:
                    return results
                try:
                    data = pytesseract.image_to_data(variant, lang=OCR_LANGUAGES, config=f"--psm {psm}",
                                                    output_type=pytesseract.Output.DICT, timeout=min(3, remaining))
                    observed = candidate_from_data(data, variant_name, psm)
                    target = [line for line in observed.positioned_lines
                              if top - height * .15 <= y0 + (line.top + line.bottom) / (2 * factor) <= bottom + height * .15]
                    item = _parse_reread(target, original)
                    if item is not None:
                        results.append((item, target, psm))
                except (pytesseract.TesseractError, RuntimeError, OSError):
                    logger.info("Targeted OCR failed: item=%s psm=%s", index, psm)
    return results


def choose_reread(original, reasons, source, results, neighbours):
    original_conf = confidence(source)
    if original_conf is None:
        return None
    # Agreement must involve different PSMs, not two threshold variants of one PSM.
    votes = {}
    for item, lines, psm in results:
        key = (re.sub(r"\W", "", item.name.casefold()), Decimal(item.quantity),
               Decimal(item.total_price), Decimal(item.unit_price) if item.unit_price else None)
        votes.setdefault(key, set()).add(psm)
    accepted = []
    for item, lines, psm in results:
        new_reasons = suspicion_reasons(item, neighbours, lines)
        new_conf = confidence(lines)
        numeric = [line.numeric_confidence for line in lines if line.numeric_confidence is not None]
        key = (re.sub(r"\W", "", item.name.casefold()), Decimal(item.quantity),
               Decimal(item.total_price), Decimal(item.unit_price) if item.unit_price else None)
        if new_conf is None or new_conf < 80 or (numeric and min(numeric) < 70):
            continue
        if is_suspicious(new_reasons) or len(new_reasons) >= len(reasons):
            continue
        if new_conf >= original_conf + 12 or (new_conf >= original_conf + 3 and len(votes[key]) >= 2):
            accepted.append((new_conf, item, lines, psm))
    return max(accepted, key=lambda row: row[0]) if accepted else None


def derived_gross(receipt, candidate):
    """Diagnostic arithmetic from two confident OCR facts, never a read gross total."""
    if receipt.prepayment is None or receipt.amount_due is None or receipt.prepayment_suspect:
        return None
    # OCR 7009 without reliable supporting pixels must not turn into a fake gross.
    summary_patterns = dict(SUMMARY_LABELS)
    for field, value in (("prepayment", receipt.prepayment), ("amount_due", receipt.amount_due)):
        matched = False
        for line in candidate.positioned_lines:
            summary = summary_patterns[field].match(CURRENCY_SUFFIX.sub("", normalize(line.text)).strip())
            if summary is None:
                continue
            conf = confidence([line])
            if conf is None or conf < 80 or line.numeric_confidence is None or line.numeric_confidence < 80:
                continue
            # Reuse the parser's strict summary grammar; reject extra row fragments.
            amount = summary["value"].strip(" :;=—")
            matched = bool(re.fullmatch(AMOUNT, amount)) and decimal_value(amount) == Decimal(value)
            if matched:
                break
        if not matched:
            return None
    total = Decimal(receipt.prepayment) + Decimal(receipt.amount_due)
    return format(total, ".2f") if total <= Decimal("999999999999.99") else None


def review_suspicious_lines(image, candidate, receipt):
    started = time.monotonic()
    original_sum = str(receipt.items_total)
    original_items = [item.model_copy(deep=True) for item in receipt.items]
    suspects = []
    for index, item in enumerate(receipt.items):
        source = item_source_lines(receipt, candidate, index)
        reasons = suspicion_reasons(item, receipt.items[max(0, index - 4):index + 5], source)
        if is_suspicious(reasons):
            suspects.append((index, reasons, source))
    removed, corrected, details = set(), {}, []
    # Service grammar is stronger evidence than a second reading of an obvious header.
    for index, reasons, source in suspects:
        if "service_header" in reasons:
            removed.add(index)
            details.append(dict(item_index=index, reasons=reasons, status="removed_service"))
        else:
            corrected[index] = receipt.items[index].model_copy(update={"suspicious": True, "suspicion_reasons": reasons})
            details.append(dict(item_index=index, reasons=reasons, status="unresolved", original=receipt.items[index].model_dump(),
                                original_confidence=confidence(source)))
    pending = [(i, reasons, source) for i, reasons, source in suspects if i not in removed and source]
    pending.sort(key=lambda row: ("arithmetic_conflict" in row[1], len(row[1])), reverse=True)
    if pending and candidate.geometry_size is not None:
        try:
            with prepare_image(image, geometry_only=True) as geometry, geometry.rotate(-candidate.orientation_rotation, expand=True, fillcolor=255) as upright:
                # Refuse coordinate drift instead of sampling the wrong row.
                if upright.size == candidate.geometry_size:
                    deadline = started + REVIEW_BUDGET_SECONDS
                    for order, (index, reasons, source) in enumerate(pending[:MAX_REREAD_ITEMS]):
                        remaining_count = min(MAX_REREAD_ITEMS, len(pending)) - order
                        per_line = time.monotonic() + max(0, deadline - time.monotonic()) / remaining_count
                        results = reread_crop(upright, candidate, source, receipt.items[index], per_line, index)
                        neighbours = receipt.items[max(0, index - 4):index + 5]
                        selected = choose_reread(receipt.items[index], reasons, source, results, neighbours)
                        detail = next(row for row in details if row["item_index"] == index)
                        detail["observations"] = [dict(item=item.model_dump(), confidence=confidence(lines), psm=psm)
                                                  for item, lines, psm in results]
                        if selected is not None:
                            conf, item, lines, psm = selected
                            corrected[index] = item.model_copy(update={"was_reread": True})
                            detail.update(status="reread", corrected=item.model_dump(), corrected_confidence=conf, psm=psm)
        except (OSError, UnreadableReceipt, pytesseract.TesseractNotFoundError):
            # Optional second pass must never discard the primary OCR result.
            logger.info("Targeted OCR unavailable; preserving suspicious originals")
    items, remap = [], {}
    for index, item in enumerate(receipt.items):
        if index not in removed:
            remap[index] = len(items)
            items.append(corrected.get(index, item))
    diagnostics = receipt.parser_diagnostics.model_copy(deep=True) if receipt.parser_diagnostics else None
    if diagnostics is not None:
        for row in diagnostics.lines:
            if row.item_index in removed:
                row.category, row.reason, row.item_index = "ignored_service", "targeted_review_service_header", None
            elif row.item_index is not None:
                row.item_index = remap[row.item_index]
        counts = Counter(row.category for row in diagnostics.lines)
        diagnostics.accepted_item_lines = counts["accepted_item"]
        diagnostics.ignored_service_lines = counts["ignored_service"]
    result = receipt.model_copy(update={"items": items, "parser_diagnostics": diagnostics,
                                       "derived_gross_total": derived_gross(receipt, candidate)})
    unresolved = sum(item.suspicious for item in items)
    warnings = list(receipt.parse_warnings)
    if unresolved:
        warnings.append(f"Сомнительные позиции требуют ручной проверки: {unresolved}. Значения не угаданы.")
    if removed:
        warnings.append(f"Служебные строки исключены из товаров: {len(removed)}.")
    review = LineReviewDiagnostics(suspicious_items=len(suspects), successfully_reread_items=sum(item.was_reread for item in items),
                                  unresolved_suspicious_items=unresolved, removed_service_items=len(removed),
                                  original_item_sum=original_sum, corrected_item_sum=str(result.items_total),
                                  extra_seconds=round(time.monotonic() - started, 3), lines=details)
    # Original items support exact benchmark before/after, including quantities and duplicates.
    for detail in review.lines:
        detail.setdefault("original", original_items[detail["item_index"]].model_dump())
    logger.info("Targeted OCR: suspicious=%s reread=%s unresolved=%s removed_service=%s original_sum=%s corrected_sum=%s extra_seconds=%.3f",
                review.suspicious_items, review.successfully_reread_items, unresolved, len(removed), original_sum, result.items_total, review.extra_seconds)
    return result.model_copy(update={"line_review": review, "parse_warnings": warnings})
