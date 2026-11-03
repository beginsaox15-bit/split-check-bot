"""Original image → free OpenRouter vision → existing receipt models. No OCR."""

import asyncio
import base64
from decimal import Decimal
from io import BytesIO
import json
import os
import re

import aiohttp
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from bot.services.receipt_models import (
    Money, Quantity, ReceiptItem, RecognizedReceipt, RecognitionUnavailable, UnreadableReceipt,
)


DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_SECONDS = 120
MAX_RESPONSE_BYTES = 1_000_000


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


def receipt_model() -> str:
    model = os.getenv("OPENROUTER_RECEIPT_MODEL", "").strip() or DEFAULT_MODEL
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+:free", model):
        raise OpenRouterUnavailable("OpenRouter receipt model must explicitly end in :free")
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
        data = VisionReceipt.model_validate(json.loads(content, parse_float=Decimal))
        items, skipped = [], 0
        for item in data.items:
            if (not item.name or not item.name.strip() or item.quantity is None
                    or Decimal(item.quantity) <= 0 or item.total_price is None):
                skipped += 1
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
        return RecognizedReceipt(readable=True, items=items, receipt_total=data.gross_total,
                                 gross_total=data.gross_total, prepayment=data.prepayment,
                                 amount_due=data.amount_due, currency=normalized_currency(data),
                                 skipped_lines=skipped)
    except (ValueError, TypeError, ValidationError):
        raise OpenRouterUnavailable("OpenRouter returned invalid receipt JSON") from None


async def _post(session: aiohttp.ClientSession, key: str, payload: dict) -> tuple[int, dict | None]:
    async with session.post(API_URL, headers={"Authorization": "Bearer " + key},
                            json=payload, allow_redirects=False) as response:
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


async def recognize_receipt(image: bytes) -> RecognizedReceipt:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise OpenRouterUnavailable("OPENROUTER_API_KEY is not configured")
    model = receipt_model()
    mime = await asyncio.to_thread(image_mime, image)
    payload = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": RECEIPT_PROMPT},
        {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + base64.b64encode(image).decode()}},
    ]}], "provider": {"allow_fallbacks": False}, "temperature": 0, "max_tokens": 12000}
    try:
        # Deadline includes both attempts and backoff; cancellation remains cancellable.
        async with asyncio.timeout(TIMEOUT_SECONDS):
            async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(
                    total=TIMEOUT_SECONDS, connect=15)) as session:
                for attempt in range(2):
                    try:
                        status, body = await _post(session, key, payload)
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
