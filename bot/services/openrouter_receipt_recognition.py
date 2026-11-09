"""Original image → free OpenRouter vision → existing receipt models. No OCR."""

import asyncio
import base64
from decimal import Decimal, DecimalException, InvalidOperation, localcontext
from io import BytesIO
import json
import logging
import os
import re
from time import perf_counter

import aiohttp
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from bot.services.receipt_models import (
    Money, Quantity, ReceiptItem, RecognizedReceipt, RecognitionUnavailable, UnreadableReceipt,
)


DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_SECONDS = 180
CONNECT_TIMEOUT_SECONDS = 15
MAX_RESPONSE_BYTES = 1_000_000
logger = logging.getLogger(__name__)


def _diagnostic(stage: str, **fields) -> None:
    # Only explicit receipt fields reach this helper, never payload/headers/body.
    message = json.dumps(fields, ensure_ascii=False, default=str)
    for variable in ("OPENROUTER_API_KEY", "BOT_TOKEN"):
        secret = os.getenv(variable, "").strip()
        if secret:
            message = message.replace(json.dumps(secret, ensure_ascii=False)[1:-1], "[REDACTED]")
    message = re.sub(r"data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+", "[REDACTED_IMAGE]", message)
    logger.info("OpenRouter receipt %s %s", stage, message)


def _raw_diagnostics(raw) -> None:
    if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
        return
    total, invalid, missing = Decimal(0), 0, 0
    for index, item in enumerate(raw["items"]):
        # Aggregate only; item names and arbitrary upstream strings are sensitive.
        value = item.get("total_price") if isinstance(item, dict) else None
        if value is None:
            missing += 1
            continue
        try:
            amount = Decimal(str(value))
            if not amount.is_finite():
                raise InvalidOperation
            with localcontext() as context:
                context.prec = 32
                total += amount
        except (DecimalException, ValueError, TypeError):
            invalid += 1
            _diagnostic("raw_item_error", index=index, reason="invalid Decimal")
    totals = {}
    for key in ("gross_total", "prepayment", "amount_due", "currency"):
        value = raw.get(key)
        pattern = r"[A-Z]{3}" if key == "currency" else r"(0|[1-9][0-9]{0,11})(\.[0-9]{1,2})?"
        totals[key] = value if value is None or (
            isinstance(value, str) and re.fullmatch(pattern, value)) else "[invalid]"
    _diagnostic("raw_summary", item_count=len(raw["items"]), items_total=str(total),
                sum_complete=not (invalid or missing), invalid_total_prices=invalid,
                missing_total_prices=missing, **totals)


def _completion_diagnostics(body: dict) -> None:
    choices = body.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    choice = choice if isinstance(choice, dict) else {}
    error = choice.get("error")
    if error is not None:
        # Error messages/metadata may echo names, phone numbers or request data.
        code = error.get("code") if isinstance(error, dict) else None
        error = {"code": code if isinstance(code, int) else None, "present": True}
    usage = body.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    counts = {key: value for key, value in usage.items()
              if key in ("prompt_tokens", "completion_tokens", "reasoning_tokens")
              and isinstance(value, (int, Decimal))}
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), (int, Decimal)):
        counts["completion_tokens_details"] = {"reasoning_tokens": details["reasoning_tokens"]}
    finish = choice.get("finish_reason")
    _diagnostic("completion", choice={"finish_reason": finish if finish in ("stop", "length", "tool_calls", "content_filter", "error", None) else "unknown",
                                      "error": error}, usage=counts)


class OpenRouterUnavailable(RecognitionUnavailable):
    """Only safe fixed messages: never propagate upstream bodies or credentials."""


class VisionItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str | None = Field(max_length=200)
    quantity: Quantity | None
    unit_price: Money | None
    total_price: Money | None
    guest: str | None = Field(max_length=30)
    uncertain: StrictBool


class VisionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    items: list[VisionItem] = Field(max_length=200)
    gross_total: Money | None
    prepayment: Money | None
    amount_due: Money | None
    currency: str | None = Field(pattern=r"^[A-Z]{3}$")
    # Optional literal image evidence supports currency inference without OCR.
    currency_evidence: str | None = Field(default=None, max_length=500)


def receipt_response_format() -> dict:
    schema = VisionReceipt.model_json_schema()

    def strict(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                strict(value)
        elif isinstance(node, list):
            for value in node:
                strict(value)

    strict(schema)
    return {"type": "json_schema", "json_schema": {
        "name": "vision_receipt", "strict": True, "schema": schema}}


def receipt_model() -> str:
    model = os.getenv("OPENROUTER_RECEIPT_MODEL", "").strip() or DEFAULT_MODEL
    if model != DEFAULT_MODEL:
        raise OpenRouterUnavailable("OpenRouter receipt model must be " + DEFAULT_MODEL)
    return model


def image_mime(image: bytes) -> str:
    if not image or len(image) > 10 * 1024 * 1024:
        raise UnreadableReceipt("Image size exceeds limit or image is empty")
    try:
        with Image.open(BytesIO(image)) as source:
            if source.width * source.height > 25_000_000:
                raise UnreadableReceipt("Image dimensions exceed limit")
            mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(source.format)
            if mime is None:
                raise UnreadableReceipt("Unsupported image format")
            source.verify()
            return mime
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise UnreadableReceipt("Invalid image") from None


def normalized_currency(data: VisionReceipt) -> str | None:
    if data.currency is not None:
        return data.currency
    evidence = data.currency_evidence or ""
    names = " ".join(item.name or "" for item in data.items)
    # A Cyrillic product name alone does not distinguish RUB from KZT/BYN/UAH.
    russian = bool(re.search(r"[а-яё]", names, re.I))
    russian_format = any(value is not None and re.fullmatch(r"\d+\.\d{2}", value)
                         for value in (data.gross_total, data.amount_due))
    rubles = bool(re.search(r"₽|\bRUB\b|\bруб(?:\.|ля|лей)?\b", evidence, re.I))
    russian_fiscal = bool(re.search(r"\bИНН\s*[:№]?\s*(?:\d{12}|\d{10})\b", evidence, re.I))
    foreign = bool(re.search(r"\b(?:USD|EUR|KZT|BYN|UAH|тенге|грн)\b|[$€₸₴]", evidence, re.I))
    if not foreign and (rubles or (russian and russian_format and russian_fiscal)):
        return "RUB"
    return None


def parse_receipt(content: str) -> RecognizedReceipt:
    if not isinstance(content, str):
        raise OpenRouterUnavailable("OpenRouter returned no structured receipt")
    content = content.strip()
    if content.startswith("```json\n") and content.endswith("```"):
        content = content[len("```json\n"):-3].strip()
    try:
        # Pydantic enforces decimal strings. JSON numbers are never converted to floats.
        raw = json.loads(content, parse_float=Decimal)
        _raw_diagnostics(raw)
        data = VisionReceipt.model_validate(raw)
        items, skipped = [], 0
        for index, item in enumerate(data.items):
            if (not item.name or not item.name.strip() or item.quantity is None
                    or Decimal(item.quantity) <= 0 or item.total_price is None):
                skipped += 1
                skip_reasons = []
                if not item.name or not item.name.strip():
                    skip_reasons.append("empty name")
                if item.quantity is None or Decimal(item.quantity) <= 0:
                    skip_reasons.append("missing/zero quantity")
                if item.total_price is None:
                    skip_reasons.append("missing total_price")
                _diagnostic("rejected_item", index=index, reasons=skip_reasons)
                continue
            reasons = ["ai_uncertain"] if item.uncertain else []
            if item.unit_price is not None and abs(Decimal(item.unit_price) * Decimal(item.quantity)
                                                   - Decimal(item.total_price)) > Decimal("0.02"):
                reasons.append("arithmetic_conflict")
            items.append(ReceiptItem(name=item.name, quantity=item.quantity, unit_price=item.unit_price,
                                     total_price=item.total_price, guest_id=item.guest,
                                     suspicious=bool(reasons), suspicion_reasons=reasons))
        if not items:
            raise OpenRouterUnavailable("OpenRouter returned no usable receipt items")
        result = RecognizedReceipt(readable=True, items=items, receipt_total=data.gross_total,
                                 gross_total=data.gross_total, prepayment=data.prepayment,
                                 amount_due=data.amount_due, currency=normalized_currency(data),
                                 skipped_lines=skipped)
        _diagnostic("converted_summary", item_count=len(result.items),
                    items_total=str(result.items_total), gross_total=result.gross_total,
                    prepayment=result.prepayment, amount_due=result.amount_due,
                    currency=result.currency, skipped_lines=result.skipped_lines)
        return result
    except json.JSONDecodeError as error:
        _diagnostic("json_error", error_type="JSONDecodeError", line=error.lineno, column=error.colno)
        raise OpenRouterUnavailable("OpenRouter returned invalid receipt JSON") from None
    except ValidationError as error:
        # Validation still aborts the receipt, including errors during conversion.
        for entry in error.errors(include_input=False, include_context=False, include_url=False):
            allowed = set(VisionReceipt.model_fields) | set(VisionItem.model_fields)
            path = [part if isinstance(part, int) or part in allowed else "[extra]"
                    for part in entry["loc"]]
            _diagnostic("validation_error", field_path=path,
                        error_type=entry["type"], message=entry["msg"])
        raise OpenRouterUnavailable("OpenRouter returned invalid receipt JSON") from None
    except (ValueError, TypeError):
        raise OpenRouterUnavailable("OpenRouter returned invalid receipt JSON") from None


async def _post(session: aiohttp.ClientSession, key: str, payload: dict) -> tuple[int, dict | None]:
    started = perf_counter()
    status = None
    try:
        async with session.post(API_URL, headers={"Authorization": "Bearer " + key},
                                json=payload, allow_redirects=False) as response:
            status = response.status
            if response.status != 200:
                return response.status, None
            chunks, size = [], 0
            async for chunk in response.content.iter_chunked(65536):
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise OpenRouterUnavailable("OpenRouter response exceeds limit")
                chunks.append(chunk)
            try:
                return response.status, json.loads(b"".join(chunks), parse_float=Decimal)
            except (ValueError, UnicodeDecodeError):
                raise OpenRouterUnavailable("OpenRouter returned invalid API JSON") from None
    finally:
        _diagnostic("http", model=payload.get("model"), http_status=status,
                    elapsed_seconds=round(perf_counter() - started, 6))


async def recognize_receipt(image: bytes) -> RecognizedReceipt:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise OpenRouterUnavailable("OPENROUTER_API_KEY is not configured")
    model = receipt_model()
    mime = await asyncio.to_thread(image_mime, image)
    payload = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": RECEIPT_PROMPT},
        {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + base64.b64encode(image).decode()}},
    ]}], "provider": {"allow_fallbacks": False}, "temperature": 0, "max_tokens": 12000,
        "reasoning": {"enabled": False}, "response_format": receipt_response_format()}
    try:
        # Deadline includes both attempts and backoff; cancellation remains cancellable.
        async with asyncio.timeout(TIMEOUT_SECONDS):
            async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(
                    total=TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS,
                    sock_connect=CONNECT_TIMEOUT_SECONDS, sock_read=TIMEOUT_SECONDS)) as session:
                for attempt in range(2):
                    try:
                        status, body = await _post(session, key, payload)
                    except TimeoutError:
                        # aiohttp socket/connect timeouts also inherit ClientConnectionError.
                        # Never spend another model request after any timeout.
                        raise
                    except aiohttp.ClientConnectionError:
                        if attempt == 0:
                            await asyncio.sleep(1)
                            continue
                        raise OpenRouterUnavailable("OpenRouter transport unavailable") from None
                    if 500 <= status < 600 and attempt == 0:
                        await asyncio.sleep(1)
                        continue
                    if status != 200:
                        raise OpenRouterUnavailable(f"OpenRouter HTTP {status}")
                    if isinstance(body, dict):
                        _completion_diagnostics(body)
                    if not isinstance(body, dict) or body.get("error"):
                        raise OpenRouterUnavailable("OpenRouter API error")
                    # The route is explicitly free; reject unexpected model substitutions.
                    if body.get("model") != model:
                        raise OpenRouterUnavailable("OpenRouter returned an unexpected model")
                    try:
                        choice = body["choices"][0]
                        if choice.get("error") or choice.get("finish_reason") != "stop":
                            raise OpenRouterUnavailable("OpenRouter returned incomplete receipt")
                        content = choice["message"]["content"]
                    except (KeyError, IndexError, TypeError, AttributeError):
                        raise OpenRouterUnavailable("OpenRouter returned malformed completion") from None
                    return parse_receipt(content)
    except TimeoutError:
        raise OpenRouterUnavailable("OpenRouter timed out") from None
    except aiohttp.ClientError:
        raise OpenRouterUnavailable("OpenRouter transport unavailable") from None


RECEIPT_PROMPT = """Проанализируй только приложенное оригинальное изображение ресторанного чека. Верни только JSON, без Markdown и объяснений.
Выпиши все реальные товарные позиции по порядку, сохраняя повторяющиеся строки отдельными позициями и относя каждую к напечатанному гостю. Количество считывай из колонки количества; сумму строки — из колонки суммы. Дробные количества сохраняй точно. Перенос названия на следующую строку не является новой позицией. unit_price укажи только если она явно напечатана; не вычисляй и не угадывай отсутствующую цену.
Не считай товарами строки Гость N, Стол, Зал, Официант, Заказ, дату, время, QR, ИНН, фискальные данные, заголовки, итоги гостей, платёжные и другие служебные строки.
Ничего не угадывай. Если поле невозможно уверенно прочитать, верни null и/или uncertain=true. Не создавай отсутствующие позиции для совпадения суммы. Не подменяй прочитанные значения вычисленными.
Формат: {"items":[{"name":"название или null","quantity":"десятичная строка или null","unit_price":"десятичная строка или null","total_price":"десятичная строка или null","guest":"номер гостя или null","uncertain":false}],"gross_total":"десятичная строка или null","prepayment":"десятичная строка или null","amount_due":"десятичная строка или null","currency":"код валюты или null"}.
Используй настоящий JSON null для неизвестных значений. Все числовые поля возвращай десятичными строками, без разделителей тысяч. gross_total — полная сумма, prepayment — предоплата, amount_due — остаток к оплате, каждое поле считай с изображения отдельно.
Дополнительно верни optional поле currency_evidence: короткие буквальные фрагменты с изображения, подтверждающие валюту или российский формат чека (например знак рубля или строку ИНН с номером). Если таких фрагментов нет, верни null. Не придумывай признаки валюты."""
