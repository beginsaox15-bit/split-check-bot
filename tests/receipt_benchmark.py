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
from bot.services.receipt_recognition import recognize_receipt


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
def trace_pipeline(save_images: bool):
    """Собирает диагностику, всегда вызывая настоящие функции текущего pipeline.

    Обёртки живут только в отдельном CLI-процессе, не в работающем боте.
    Сохраняется точное изображение, переданное выбранному вызову Tesseract.
    """
    trace = {"candidate": None, "images": {}}
    original_extract, original_data = local.extract_text, local.pytesseract.image_to_data
    variant_ids = {}

    def extract(image):
        candidate = original_extract(image)
        trace["candidate"] = candidate
        return candidate

    def image_to_data(image, **kwargs):
        if save_images:
            if id(image) not in variant_ids:
                variant_ids[id(image)] = "grayscale" if not variant_ids else "adaptive"
            name = variant_ids[id(image)]
            if name not in trace["images"]:
                trace["images"][name] = image.copy()
        return original_data(image, **kwargs)

    try:
        with patch.object(local, "extract_text", side_effect=extract), \
                patch.object(local.pytesseract, "image_to_data", side_effect=image_to_data):
            yield trace
    finally:
        for image in trace["images"].values():
            image.close()


def equal_total(expected: str | None, actual: str | None) -> bool:
    if expected is None or actual is None:
        return expected is actual
    return Decimal(expected) == Decimal(actual)


async def run_case(case: ExpectedReceipt, image: Path, artifacts: Path | None) -> dict:
    start = time.perf_counter()
    result, status, error_type = None, "ok", None
    with trace_pipeline(artifacts is not None) as trace:
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
            "psm": candidate.psm if candidate else None,
            "preprocessing": candidate.preprocessing if candidate else None,
            "elapsed_seconds": round(elapsed, 3), "error": error_type,
        }
        if artifacts is not None:
            directory = artifacts / case.name
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "ocr.txt").write_text(candidate.text if candidate else "", encoding="utf-8")
            selected = trace["images"].get(candidate.preprocessing) if candidate else None
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
    total_status = "OK" if row["total_correct"] else f"FAIL (expected {row['expected_total']}, got {row['actual_total']})"
    currency_status = "OK" if row["currency_correct"] else f"FAIL (expected {row['expected_currency']})"
    print(f"  total: {total_status} | currency: {row['actual_currency'] or '-'} {currency_status}")
    print(f"  PSM: {row['psm']} | preprocessing: {row['preprocessing']} | time: {row['elapsed_seconds']:.2f}s | status: {row['status']}")
    if row["error"]:
        print(f"  error: {row['error']}")


async def benchmark(cases, artifacts):
    print("Receipt benchmark", flush=True)
    rows = []
    for case, image in cases:
        row = await run_case(case, image, artifacts)
        rows.append(row)
        print_case(row)
        sys.stdout.flush()
    overall = summarize(rows)
    print(f"\nOverall:\n  items accuracy: {overall['items_accuracy_percent']:.1f}% ({overall['found_items']}/{overall['expected_items']})")
    print(f"  totals correct: {overall['totals_correct']}/{len(rows)}")
    print(f"  currency correct: {overall['currencies_correct']}/{len(rows)}")
    print(f"  false positives: {overall['false_positives']} | errors: {overall['errors']} | time: {overall['elapsed_seconds']:.2f}s")
    return {"receipts": rows, "overall": overall}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--case", action="append", help="Case name; repeat to select several")
    parser.add_argument("--artifacts", nargs="?", type=Path, const=DEFAULT_ARTIFACTS,
                        help="Save raw OCR, selected processed image and report.json")
    parser.add_argument("--json", type=Path, help="Save a machine-readable report for before/after comparison")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env", override=False)
    try:
        cases = load_cases(args.fixtures)
        if args.case:
            missing = set(args.case) - {case.name for case, _ in cases}
            if missing:
                raise ValueError(f"Unknown cases: {', '.join(sorted(missing))}")
            cases = [(case, image) for case, image in cases if case.name in args.case]
    except ValueError as error:
        parser.error(str(error))
    report = asyncio.run(benchmark(cases, args.artifacts))
    outputs = [args.json] if args.json else []
    if args.artifacts is not None:
        outputs.append(args.artifacts / "report.json")
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if report["overall"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
