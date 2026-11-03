"""Локальный benchmark. Запуск: python -m tests.receipt_benchmark [--artifacts]."""

import argparse
import asyncio
import json
import re
import sys
import time
import unicodedata
from contextlib import contextmanager
from decimal import Decimal
from difflib import SequenceMatcher
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

from bot.services import tesseract_receipt_recognition as local
from bot.services.receipt_models import LowQualityReceipt, UnparsedReceipt, UnreadableReceipt
from bot.services.tesseract_receipt_recognition import recognize_receipt
from bot.services.receipt_parser import parse_receipt_lines


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = ROOT / "tests/fixtures/receipts"
DEFAULT_ARTIFACTS = ROOT / "artifacts/receipt_benchmark"
NUMBER = r"^(0|[1-9][0-9]*)(\.[0-9]+)?$"


class ExpectedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    quantity: str = Field(pattern=NUMBER)
    total_price: str = Field(pattern=NUMBER)

    @field_validator("name")
    @classmethod
    def nonblank_name(cls, value):
        if not value.strip():
            raise ValueError("Item name must not be blank")
        return value.strip()

    @field_validator("quantity")
    @classmethod
    def positive_quantity(cls, value):
        if Decimal(value) <= 0:
            raise ValueError("Quantity must be positive")
        return value


class ExpectedReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[\w-]+$")
    image: str
    tags: list[str] = Field(default_factory=list)
    expected_items: list[ExpectedItem] = Field(min_length=1)
    expected_total: str | None = Field(pattern=NUMBER)
    currency: str | None = Field(pattern=r"^[A-Z]{3}$")
    expected_gross_total: str | None = Field(default=None, pattern=NUMBER)
    expected_prepayment: str | None = Field(default=None, pattern=NUMBER)
    expected_amount_due: str | None = Field(default=None, pattern=NUMBER)
    expected_guest_subtotals: dict[str, str] | None = None

    @field_validator("expected_guest_subtotals")
    @classmethod
    def valid_guest_subtotals(cls, value):
        if value is not None and any(not re.fullmatch(NUMBER, amount) for amount in value.values()):
            raise ValueError("Guest subtotals must be decimal strings")
        return value


def normalized_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    return " ".join(re.findall(r"[a-zа-я0-9]+", value))


def match_items(expected: list[ExpectedItem], actual: list) -> dict:
    """Одно совпадение расходует одну строку; дубликаты не теряются."""
    candidates = []
    for ei, wanted in enumerate(expected):
        for ai, found in enumerate(actual):
            if (Decimal(wanted.quantity) != Decimal(found.quantity)
                    or Decimal(wanted.total_price) != Decimal(found.total_price)):
                continue
            similarity = SequenceMatcher(None, normalized_name(wanted.name), normalized_name(found.name)).ratio()
            if similarity >= .80:
                candidates.append((similarity, ei, ai))
    used_expected, used_actual = set(), set()
    for _, ei, ai in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
        if ei not in used_expected and ai not in used_actual:
            used_expected.add(ei)
            used_actual.add(ai)
    return {
        "found": len(used_expected), "expected": len(expected),
        "missing": len(expected) - len(used_expected),
        "false_positives": len(actual) - len(used_actual),
        "missing_names": [item.name for i, item in enumerate(expected) if i not in used_expected],
        "false_positive_names": [item.name for i, item in enumerate(actual) if i not in used_actual],
    }


def load_cases(fixtures: Path) -> list[tuple[ExpectedReceipt, Path]]:
    root = fixtures.resolve()
    paths = sorted((root / "expected").rglob("*.json"))
    if not paths:
        raise ValueError(f"No expected JSON files in {root / 'expected'}")
    cases, names = [], set()
    for path in paths:
        try:
            case = ExpectedReceipt.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            raise ValueError(f"Invalid expected JSON: {path}") from error
        if case.name in names:
            raise ValueError(f"Duplicate case name: {case.name}")
        names.add(case.name)
        image = (root / case.image).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise ValueError(f"Image must exist inside fixtures: {case.image}")
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}:
            raise ValueError(f"Unsupported image: {case.image}")
        cases.append((case, image))
    return cases


@contextmanager
def trace_pipeline(save_images: bool, strategy="auto"):
    """Собирает диагностику, всегда вызывая настоящие функции текущего pipeline.

    Обёртки живут только в отдельном CLI-процессе, не в работающем боте.
    Сохраняется точное изображение, переданное выбранному вызову Tesseract.
    """
    trace = {"candidate": None, "images": {}, "parser_diagnostics": None, "original_receipt": None}
    original_extract, original_data = local.extract_text, local.pytesseract.image_to_data
    original_parser = local.parse_receipt_lines
    variant_ids = {}

    def extract(image):
        candidate = original_extract(image) if strategy == "auto" else original_extract(image, strategy=strategy)
        trace["candidate"] = candidate
        return candidate

    def image_to_data(image, **kwargs):
        if save_images:
            scope = image.info.get("ocr_scope", "full")
            variant = image.info.get("ocr_preprocessing")
            if variant is None:
                if id(image) not in variant_ids:
                    variant_ids[id(image)] = "grayscale" if not variant_ids else "adaptive"
                variant = variant_ids[id(image)]
            name = variant if scope == "full" else scope + "/" + variant
            if name not in trace["images"]:
                trace["images"][name] = image.copy()
        return original_data(image, **kwargs)

    def parse(lines):
        try:
            result = original_parser(lines)
            trace["parser_diagnostics"] = result.parser_diagnostics
            trace["original_receipt"] = result.model_copy(deep=True)
            return result
        except UnreadableReceipt as error:
            trace["parser_diagnostics"] = getattr(error, "parser_diagnostics", None)
            raise

    try:
        with patch.object(local, "extract_text", side_effect=extract), \
                patch.object(local.pytesseract, "image_to_data", side_effect=image_to_data), \
                patch.object(local, "parse_receipt_lines", side_effect=parse):
            yield trace
    finally:
        for image in trace["images"].values():
            image.close()


def equal_total(expected: str | None, actual: str | None) -> bool:
    if expected is None or actual is None:
        return expected is actual
    return Decimal(expected) == Decimal(actual)


async def run_case(case: ExpectedReceipt, image: Path, artifacts: Path | None, strategy="auto") -> dict:
    start = time.perf_counter()
    result, status, error_type = None, "ok", None
    with trace_pipeline(artifacts is not None, strategy) as trace:
        try:
            result = await recognize_receipt(image.read_bytes())
        except LowQualityReceipt:
            status = "low_quality"
        except UnparsedReceipt:
            status = "unparsed"
        except UnreadableReceipt:
            status = "unreadable"
        except Exception as error:
            status, error_type = "error", type(error).__name__
        elapsed = time.perf_counter() - start
        candidate = trace["candidate"]
        actual_total = result.receipt_total if result is not None else None
        actual_currency = result.currency if result is not None else None
        metrics = match_items(case.expected_items, result.items if result is not None else [])
        row = {
            "name": case.name, "file": case.image, "tags": case.tags, "status": status,
            "lines": candidate.lines if candidate else 0, **metrics,
            "total_correct": result is not None and equal_total(case.expected_total, actual_total),
            "expected_total": case.expected_total, "actual_total": actual_total,
            "currency_correct": result is not None and case.currency == actual_currency,
            "expected_currency": case.currency, "actual_currency": actual_currency,
            "strategy": candidate.strategy if candidate else None,
            "segment_count": candidate.segment_count if candidate else 0,
            "overlap": candidate.overlap if candidate else 0,
            "psms": candidate.psms if candidate else [],
            "candidate_item_lines": candidate.quality.get("candidate_item_lines", 0) if candidate else 0,
            "psm": candidate.psm if candidate else None,
            "preprocessing": candidate.preprocessing if candidate else None,
            "elapsed_seconds": round(elapsed, 3), "error": error_type,
            "items_total": str(result.items_total) if result is not None else None,
            "prepayment_ocr": result.prepayment_ocr if result is not None else None,
            "prepayment_status": ("suspect" if result.prepayment_suspect else "verified" if result.prepayment_verified
                                  else "unverified" if result.prepayment is not None else "not_found") if result is not None else "not_found",
        }
        for name in ("gross_total", "prepayment", "amount_due"):
            actual = getattr(result, name) if result is not None else None
            row[name] = actual
            if "expected_" + name in case.model_fields_set:
                row[name + "_correct"] = result is not None and equal_total(getattr(case, "expected_" + name), actual)
        row["guest_subtotals"] = result.guest_subtotals if result is not None else {}
        if case.expected_guest_subtotals is not None:
            row["guest_subtotals_correct"] = result is not None and ({key: Decimal(value) for key, value in row["guest_subtotals"].items()}
                                              == {key: Decimal(value) for key, value in case.expected_guest_subtotals.items()})
        diagnostics = result.parser_diagnostics if result is not None else trace["parser_diagnostics"]
        row["parser_diagnostics"] = diagnostics.model_dump() if diagnostics is not None else None
        row["accepted_items_count"] = len(result.items) if result is not None else 0
        row["line_review"] = result.line_review.model_dump() if result is not None and result.line_review else None
        row["derived_gross_total"] = result.derived_gross_total if result is not None else None
        row["difference_to_gross"] = str(result.missing_difference) if result is not None and result.missing_difference is not None else None
        row["difference_reference"] = ("gross_ocr" if result.comparison_total is not None else "derived_gross") if result is not None and result.diagnostic_gross_total is not None else None
        original = trace["original_receipt"]
        if original is not None and result is not None:
            before_metrics = match_items(case.expected_items, original.items)
            reference = result.diagnostic_gross_total
            row["before_targeted_review"] = {
                "accepted_items": len(original.items), "suspicious_items": row["line_review"]["suspicious_items"] if row["line_review"] else 0,
                "item_sum": str(original.items_total), "false_positives": before_metrics["false_positives"],
                "found": before_metrics["found"], "rejected_candidates": original.parser_diagnostics.rejected_candidate_lines,
                "difference_to_gross": str(reference - original.items_total) if reference is not None else None,
                "processing_seconds": round(elapsed - result.line_review.extra_seconds, 3) if result.line_review else round(elapsed, 3),
            }
            row["after_targeted_review"] = {
                "accepted_items": len(result.items), "suspicious_items": sum(item.suspicious for item in result.items),
                "item_sum": str(result.items_total), "false_positives": metrics["false_positives"], "found": metrics["found"],
                "rejected_candidates": diagnostics.rejected_candidate_lines,
                "difference_to_gross": row["difference_to_gross"], "processing_seconds": round(elapsed, 3),
            }
        row["strategies"] = []
        for alternative in candidate.alternatives if candidate else []:
            parsed = None
            rejected = None
            try:
                if alternative.reliable:
                    parsed = parse_receipt_lines(alternative.text.splitlines())
                    rejected = parsed.parser_diagnostics.rejected_candidate_lines
            except UnreadableReceipt as error:
                diag = getattr(error, "parser_diagnostics", None)
                rejected = diag.rejected_candidate_lines if diag else None
            metrics_alt = match_items(case.expected_items, parsed.items if parsed else [])
            row["strategies"].append({
                "strategy": alternative.strategy, "segments": alternative.segment_count, "overlap": alternative.overlap,
                "complete": alternative.complete,
                "status": "ocr_unavailable" if not alternative.complete else "low_quality" if not alternative.reliable else "parsed" if parsed else "unparsed",
                "psms": alternative.psms, "lines": alternative.lines, "quality": alternative.quality,
                "accepted_items": len(parsed.items) if parsed else 0, "rejected_candidates": rejected,
                "found": metrics_alt["found"], "false_positives": metrics_alt["false_positives"],
                "item_sum": str(parsed.items_total) if parsed else None, "time": round(alternative.elapsed_seconds, 3),
            })
        if artifacts is not None:
            directory = artifacts / case.name
            directory.mkdir(parents=True, exist_ok=True)
            segment_dir = directory / "processed_segments"
            if segment_dir.exists():
                for old in segment_dir.glob("[0-9][0-9].png"):
                    old.unlink()
                if (segment_dir / "segments.json").exists():
                    (segment_dir / "segments.json").unlink()
            (directory / "ocr.txt").write_text(candidate.text if candidate else "", encoding="utf-8")
            (directory / "parser_lines.json").write_text(json.dumps(row["parser_diagnostics"], ensure_ascii=False, indent=2), encoding="utf-8")
            (directory / "line_review.json").write_text(json.dumps(row["line_review"], ensure_ascii=False, indent=2), encoding="utf-8")
            reread_dir = directory / "reread"
            if reread_dir.exists():
                for old in reread_dir.glob("*.png"):
                    old.unlink()
            for name, crop in trace["images"].items():
                if name.startswith("reread_"):
                    reread_dir.mkdir(exist_ok=True)
                    crop.save(reread_dir / (name.replace("/", "_") + ".png"))
            selected = trace["images"].get(candidate.preprocessing) if candidate and candidate.strategy == "full" else None
            if candidate and candidate.strategy == "segmented":
                segment_dir = directory / "processed_segments"
                segment_dir.mkdir(exist_ok=True)
                for detail in candidate.segment_details:
                    selected_segment = trace["images"].get(f"segment_{detail['index']}/{detail['preprocessing']}")
                    if selected_segment is not None:
                        selected_segment.save(segment_dir / f"{detail['index']:02d}.png")
                (segment_dir / "segments.json").write_text(json.dumps(candidate.segment_details, indent=2))
            if selected is not None:
                selected.save(directory / "processed.png")
            elif (directory / "processed.png").exists():
                # Не оставляем изображение успешного прошлого запуска при ошибке.
                (directory / "processed.png").unlink()
        return row


def summarize(rows: list[dict]) -> dict:
    found, expected = sum(row["found"] for row in rows), sum(row["expected"] for row in rows)
    return {
        "receipts": len(rows), "found_items": found, "expected_items": expected,
        "items_accuracy_percent": round(100 * found / expected, 1) if expected else 0,
        "missing_items": sum(row["missing"] for row in rows),
        "false_positives": sum(row["false_positives"] for row in rows),
        "totals_correct": sum(row["total_correct"] for row in rows),
        "currencies_correct": sum(row["currency_correct"] for row in rows),
        "errors": sum(row["status"] == "error" for row in rows),
        "elapsed_seconds": round(sum(row["elapsed_seconds"] for row in rows), 3),
    }


def print_case(row):
    print(f"\n{row['name']} ({row['file']}):")
    print(f"  lines: {row['lines']} | items: {row['found']}/{row['expected']} | missing: {row['missing']}")
    print(f"  false positives: {row['false_positives']}")
    diagnostics = row.get("parser_diagnostics")
    if diagnostics:
        print(f"  OCR lines: {diagnostics['ocr_lines']} | accepted item lines: {diagnostics['accepted_item_lines']} | rejected candidate lines: {diagnostics['rejected_candidate_lines']} | ignored service lines: {diagnostics['ignored_service_lines']}")
    if "items_total" in row:
        print(f"  items total: {row['items_total']}")
    review = row.get("line_review")
    if review:
        print(f"  targeted review: suspicious {review['suspicious_items']} | re-read {review['successfully_reread_items']} | unresolved {review['unresolved_suspicious_items']} | removed service {review['removed_service_items']} | extra {review['extra_seconds']:.2f}s")
        print(f"  original sum: {review['original_item_sum']} | corrected sum: {review['corrected_item_sum']} | difference to {row['difference_reference']}: {row['difference_to_gross']}")
        if row.get("derived_gross_total") is not None:
            print(f"  derived gross (calculated, not OCR): {row['derived_gross_total']}")
        for stage in ("before_targeted_review", "after_targeted_review"):
            if stage in row:
                m = row[stage]
                print(f"  {stage}: accepted {m['accepted_items']} | suspicious {m['suspicious_items']} | sum {m['item_sum']} | difference {m['difference_to_gross']} | false positives {m['false_positives']} | time {m['processing_seconds']:.2f}s")
    total_status = "OK" if row["total_correct"] else f"FAIL (expected {row['expected_total']}, got {row['actual_total']})"
    currency_status = "OK" if row["currency_correct"] else f"FAIL (expected {row['expected_currency']})"
    print(f"  total: {total_status} | currency: {row['actual_currency'] or '-'} {currency_status}")
    print(f"  PSM: {row['psm']} | preprocessing: {row['preprocessing']} | time: {row['elapsed_seconds']:.2f}s | status: {row['status']}")
    for name in ("gross_total", "prepayment", "amount_due", "guest_subtotals"):
        if name + "_correct" in row:
            print(f"  {name}: {row[name]} {'OK' if row[name + '_correct'] else 'FAIL'}")
    if "strategy" in row:
        print(f"  strategy: {row['strategy']} | segments: {row['segment_count']} | overlap: {row['overlap']:.0%} | PSMs: {row['psms']} | candidate item lines: {row['candidate_item_lines']}")
    for alternative in row.get("strategies", []):
        print(f"  {alternative['strategy']}: accepted {alternative['accepted_items']}, rejected {alternative['rejected_candidates']}, false positives {alternative['false_positives']}, items {alternative['found']}/{row['expected']}, sum {alternative['item_sum']}, time {alternative['time']:.2f}s")
    if row["error"]:
        print(f"  error: {row['error']}")


async def benchmark(cases, artifacts, strategy="auto"):
    print("Receipt benchmark", flush=True)
    rows = []
    for case, image in cases:
        row = await run_case(case, image, artifacts, strategy)
        rows.append(row)
        print_case(row)
        sys.stdout.flush()
    overall = summarize(rows)
    print(f"\nOverall:\n  items accuracy: {overall['items_accuracy_percent']:.1f}% ({overall['found_items']}/{overall['expected_items']})")
    print(f"  totals correct: {overall['totals_correct']}/{len(rows)}")
    print(f"  currency correct: {overall['currencies_correct']}/{len(rows)}")
    print(f"  false positives: {overall['false_positives']} | errors: {overall['errors']} | time: {overall['elapsed_seconds']:.2f}s")
    return {"receipts": rows, "overall": overall}


def replay_raw_ocr(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    start = time.perf_counter()
    try:
        result = parse_receipt_lines(text.splitlines())
    except UnreadableReceipt as error:
        diagnostics = getattr(error, "parser_diagnostics", None)
        return {"mode": "raw_ocr_replay", "status": "unparsed", "parser_diagnostics": diagnostics.model_dump() if diagnostics else None}
    return {
        "mode": "raw_ocr_replay", "status": "ok", "reference_available": False,
        "accepted_items_count": len(result.items), "items_total": str(result.items_total),
        "guest_subtotals": result.guest_subtotals, "guest_totals_sum": str(result.guest_subtotals_total),
        "gross_total": result.gross_total, "prepayment": result.prepayment, "amount_due": result.amount_due,
        "prepayment_ocr": result.prepayment_ocr,
        "prepayment_status": "suspect" if result.prepayment_suspect else "verified" if result.prepayment_verified else "unverified" if result.prepayment is not None else "not_found",
        "calculated_prepayment": str(result.calculated_prepayment) if result.calculated_prepayment is not None else None,
        "comparison_total": str(result.comparison_total) if result.comparison_total is not None else None,
        "currency": result.currency, "parser_diagnostics": result.parser_diagnostics.model_dump(),
        "accepted_items": [item.model_dump() for item in result.items],
        "parse_warnings": result.parse_warnings,
        "elapsed_seconds": round(time.perf_counter() - start, 4),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--case", action="append", help="Case name; repeat to select several")
    parser.add_argument("--artifacts", nargs="?", type=Path, const=DEFAULT_ARTIFACTS,
                        help="Save raw OCR, selected processed image and report.json")
    parser.add_argument("--json", type=Path, help="Save a machine-readable report for before/after comparison")
    parser.add_argument("--raw-ocr", type=Path, help="Replay existing OCR through parser only; no image OCR or item gold is claimed")
    parser.add_argument("--strategy", choices=("auto", "full", "segmented", "compare"), default="auto",
                        help="Force a strategy or compare both even for a shorter fixture")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env", override=False)
    if args.raw_ocr:
        if args.case or args.artifacts:
            parser.error("--raw-ocr uses --json for its report; do not combine with --case/--artifacts")
        report = replay_raw_ocr(args.raw_ocr)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    try:
        cases = load_cases(args.fixtures)
        if args.case:
            missing = set(args.case) - {case.name for case, _ in cases}
            if missing:
                raise ValueError(f"Unknown cases: {', '.join(sorted(missing))}")
            cases = [(case, image) for case, image in cases if case.name in args.case]
    except ValueError as error:
        parser.error(str(error))
    report = asyncio.run(benchmark(cases, args.artifacts, args.strategy))
    outputs = [args.json] if args.json else []
    if args.artifacts is not None:
        outputs.append(args.artifacts / "report.json")
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if report["overall"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
