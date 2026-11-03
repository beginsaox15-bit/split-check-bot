import asyncio
import base64
from decimal import Decimal
from io import BytesIO
import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from PIL import Image

from bot.services import openrouter_receipt_recognition as ai
from bot.services import receipt_recognition as flow
from bot.services.receipt_models import UnreadableReceipt


def image_bytes():
    with Image.new("RGB", (32, 32), "white") as image, BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def payload(count=2):
    total = Decimal("0.10") * count
    return {
        "items": [{"name": "Чай", "quantity": "0.250", "unit_price": "0.40",
                   "total_price": "0.10", "guest": str(1 + i // 10), "uncertain": False}
                  for i in range(count)],
        "gross_total": str(total), "prepayment": "0.10",
        "amount_due": str(total - Decimal("0.10")), "currency": None,
        "currency_evidence": "ИНН 1234567890",
    }


def completion(data=None):
    return {"model": ai.DEFAULT_MODEL, "choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps(data if data is not None else payload(), ensure_ascii=False)}}]}


class ConversionTests(unittest.TestCase):
    def parse(self, data):
        return ai.parse_receipt(json.dumps(data, ensure_ascii=False))

    def test_32_items_exact_decimals_duplicates_and_guests(self):
        result = self.parse(payload(32))
        self.assertEqual(len(result.items), 32)
        self.assertEqual(result.items_total, Decimal("3.20"))
        self.assertEqual(result.items[0].quantity, "0.250")
        self.assertEqual(result.items[0].unit_price, "0.40")
        self.assertEqual([i.guest_id for i in result.items], [str(1+i//10) for i in range(32)])
        self.assertEqual([i.name for i in result.items], ["Чай"] * 32)
        self.assertEqual(result.currency, "RUB")
        self.assertEqual(result.prepayment, "0.10")
        self.assertEqual(result.amount_due, "3.10")
        self.assertTrue(result.prepayment_verified)
        self.assertFalse(result.totals_mismatch)
        self.assertEqual(result.raw_text, "")
        self.assertIsInstance(result.model_dump()["items"][0]["total_price"], str)

    def test_explicit_currency_preserved_even_with_russian_evidence(self):
        for currency in ("USD", "EUR", "RUB", "KZT"):
            data = payload()
            data["currency"] = currency
            self.assertEqual(self.parse(data).currency, currency)

    def test_currency_unknown_without_evidence_or_with_foreign_evidence(self):
        for evidence in (None, "", "ИНН 1234567890 USD", "Total $10.00"):
            data = payload()
            data["currency_evidence"] = evidence
            self.assertIsNone(self.parse(data).currency)

    def test_ruble_symbol_is_explicit_currency_evidence(self):
        data = payload()
        data["currency_evidence"] = "Итого 0.20 ₽"
        self.assertEqual(self.parse(data).currency, "RUB")

    def test_uncertain_marks_suspicious_and_names_are_not_translated(self):
        data = payload()
        data["items"][0].update(name="Open Bar", uncertain=True, unit_price=None)
        item = self.parse(data).items[0]
        self.assertEqual(item.name, "Open Bar")
        self.assertTrue(item.suspicious)
        self.assertIn("ai_uncertain", item.suspicion_reasons)
        self.assertIsNone(item.unit_price)

    def test_null_required_item_fields_skip_without_invention(self):
        for field in ("name", "quantity", "total_price"):
            data = payload()
            data["items"][0][field] = None
            result = self.parse(data)
            self.assertEqual(len(result.items), 1)
            self.assertEqual(result.skipped_lines, 1)
            self.assertEqual(result.items_total, Decimal("0.10"))
            self.assertTrue(result.totals_mismatch)

    def test_invalid_json_and_invalid_numeric_values_rejected(self):
        for content in ("not JSON", "{}", "[]", '{"items":NaN}'):
            with self.subTest(content=content), self.assertRaises(ai.OpenRouterUnavailable):
                ai.parse_receipt(content)
        for value in (0.1, "NaN", "Infinity", "-1", "0.001", "1e2"):
            data = payload()
            data["items"][0]["total_price"] = value
            with self.subTest(value=value), self.assertRaises(ai.OpenRouterUnavailable):
                self.parse(data)

    def test_empty_or_all_unknown_items_trigger_local_fallback(self):
        for count in (0, 1):
            data = payload(count)
            data["amount_due"] = None
            for item in data["items"]:
                item["quantity"] = None
            with self.assertRaises(ai.OpenRouterUnavailable):
                self.parse(data)

    def test_inconsistent_prepayment_uses_existing_quarantine(self):
        data = payload()
        data["prepayment"] = "0.15"
        result = self.parse(data)
        self.assertIsNone(result.prepayment)
        self.assertEqual(result.prepayment_ocr, "0.15")
        self.assertTrue(result.prepayment_suspect)

    def test_arithmetic_conflict_flagged_without_rewriting_money(self):
        data = payload()
        data["items"][0]["unit_price"] = "10.00"
        item = self.parse(data).items[0]
        self.assertEqual(item.total_price, "0.10")
        self.assertTrue(item.suspicious)
        self.assertIn("arithmetic_conflict", item.suspicion_reasons)


class TransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"OPENROUTER_API_KEY": "unit-test-secret",
                                                "OPENROUTER_RECEIPT_MODEL": ""}))
        self.post = self.enterContext(patch.object(ai, "_post", new_callable=AsyncMock))
        self.post.return_value = (200, completion())
        self.sleep = self.enterContext(patch.object(ai.asyncio, "sleep", new_callable=AsyncMock))

    async def test_direct_request_contains_original_bytes_and_only_free_model(self):
        image = image_bytes()
        result = await ai.recognize_receipt(image)
        self.assertEqual(result.currency, "RUB")
        self.post.assert_awaited_once()
        session, key, request = self.post.await_args.args
        self.assertTrue(session.trust_env)
        self.assertEqual(key, "unit-test-secret")
        self.assertEqual(request["model"], "nvidia/nemotron-3.5-super-vl:free")
        self.assertFalse(request["provider"]["allow_fallbacks"])
        self.assertNotIn("models", request)
        url = request["messages"][0]["content"][1]["image_url"]["url"]
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), image)
        self.assertIn("data:image/png;base64,", url)

    async def test_missing_key_never_calls_api(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            with self.assertRaises(ai.OpenRouterUnavailable):
                await ai.recognize_receipt(image_bytes())
        self.post.assert_not_awaited()

    async def test_paid_model_is_rejected_without_modification(self):
        with patch.dict(os.environ, {"OPENROUTER_RECEIPT_MODEL": ai.DEFAULT_MODEL.removesuffix(":free")}):
            with self.assertRaises(ai.OpenRouterUnavailable):
                await ai.recognize_receipt(image_bytes())
        self.post.assert_not_awaited()

    async def test_configured_free_model_preserves_suffix(self):
        with patch.dict(os.environ, {"OPENROUTER_RECEIPT_MODEL": ai.DEFAULT_MODEL}):
            await ai.recognize_receipt(image_bytes())
        self.assertEqual(self.post.await_args.args[2]["model"], ai.DEFAULT_MODEL)

    async def test_timeout_has_no_retry_and_safe_error(self):
        self.post.side_effect = TimeoutError("unit-test-secret data:image/png;base64,PRIVATE")
        with self.assertRaisesRegex(ai.OpenRouterUnavailable, "timed out") as caught:
            await ai.recognize_receipt(image_bytes())
        self.assertNotIn("unit-test-secret", str(caught.exception))
        self.assertEqual(self.post.await_count, 1)

    async def test_overall_deadline_interrupts_stalled_response(self):
        async def stall(*args):
            await asyncio.Event().wait()

        self.post.side_effect = stall
        with patch.object(ai, 'TIMEOUT_SECONDS', 0.01):
            with self.assertRaisesRegex(ai.OpenRouterUnavailable, 'timed out'):
                await ai.recognize_receipt(image_bytes())
        self.assertEqual(self.post.await_count, 1)

    async def test_5xx_retry_then_success_and_retry_exhaustion(self):
        self.post.side_effect = [(503, None), (200, completion())]
        await ai.recognize_receipt(image_bytes())
        self.assertEqual(self.post.await_count, 2)
        self.sleep.assert_awaited_once_with(1)
        self.post.reset_mock()
        self.post.side_effect = [(502, None), (503, None)]
        with self.assertRaises(ai.OpenRouterUnavailable):
            await ai.recognize_receipt(image_bytes())
        self.assertEqual(self.post.await_count, 2)

    async def test_transport_retries_at_most_once(self):
        self.post.side_effect = aiohttp.ClientConnectionError("secret must not propagate")
        with self.assertRaisesRegex(ai.OpenRouterUnavailable, "transport unavailable"):
            await ai.recognize_receipt(image_bytes())
        self.assertEqual(self.post.await_count, 2)

    async def test_429_401_402_404_no_provider_or_paid_fallback(self):
        for status in (429, 401, 402, 404):
            self.post.reset_mock()
            self.post.side_effect = None
            self.post.return_value = (status, None)
            with self.subTest(status=status), self.assertRaises(ai.OpenRouterUnavailable):
                await ai.recognize_receipt(image_bytes())
            self.assertEqual(self.post.await_count, 1)

    async def test_invalid_json_truncated_error_or_model_substitution_no_retry(self):
        invalid = completion()
        invalid["choices"][0]["message"]["content"] = "not JSON"
        truncated = completion()
        truncated["choices"][0]["finish_reason"] = "length"
        substituted = completion()
        substituted["model"] = ai.DEFAULT_MODEL.removesuffix(":free")
        for body in (invalid, truncated, substituted, {"error": {"message": "private"}}, [], {}):
            self.post.reset_mock()
            self.post.return_value = (200, body)
            with self.subTest(body=body), self.assertRaises(ai.OpenRouterUnavailable):
                await ai.recognize_receipt(image_bytes())
            self.assertEqual(self.post.await_count, 1)

    async def test_invalid_image_is_not_uploaded(self):
        with self.assertRaises(UnreadableReceipt):
            await ai.recognize_receipt(b"not an image")
        self.post.assert_not_awaited()

    async def test_cancellation_not_swallowed(self):
        self.post.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await ai.recognize_receipt(image_bytes())
        self.assertEqual(self.post.await_count, 1)

    async def test_http_boundary_headers_proxy_redirects_and_bounded_json(self):
        async def chunks(size):
            yield json.dumps(completion()).encode()
        response = MagicMock(status=200)
        response.content.iter_chunked = chunks
        context = AsyncMock()
        context.__aenter__.return_value = response
        session = MagicMock()
        session.post.return_value = context
        status, body = await self.post_original(session, "private", {"model": ai.DEFAULT_MODEL})
        self.assertEqual(status, 200)
        self.assertEqual(body["model"], ai.DEFAULT_MODEL)
        args, kwargs = session.post.call_args
        self.assertEqual(args, (ai.API_URL,))
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer private")
        self.assertFalse(kwargs["allow_redirects"])

    post_original = staticmethod(ai._post)


class FallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_json_timeout_5xx_and_rate_limit_reach_tesseract(self):
        result = ai.parse_receipt(json.dumps(payload()))
        invalid = completion()
        invalid['choices'][0]['message']['content'] = 'invalid JSON'
        for outcome in (TimeoutError(), (503, None), (429, None), (200, invalid)):
            with self.subTest(outcome=outcome), \
                    patch.dict(os.environ, {'RECEIPT_OCR_BACKEND': 'openrouter',
                                            'OPENROUTER_API_KEY': 'unit-test-secret',
                                            'OPENROUTER_RECEIPT_MODEL': ai.DEFAULT_MODEL}), \
                    patch.object(ai, '_post', new_callable=AsyncMock) as post, \
                    patch.object(ai.asyncio, 'sleep', new_callable=AsyncMock), \
                    patch.object(flow, 'recognize_local', new_callable=AsyncMock, return_value=result) as local:
                if isinstance(outcome, Exception):
                    post.side_effect = outcome
                else:
                    post.return_value = outcome
                notify = AsyncMock()
                self.assertEqual(await flow.recognize_receipt(image_bytes(), on_fallback=notify), result)
                local.assert_awaited_once()
                notify.assert_awaited_once_with()

    async def test_failure_notifies_once_and_passes_original_to_local(self):
        result = ai.parse_receipt(json.dumps(payload()))
        with patch.dict(os.environ, {"RECEIPT_OCR_BACKEND": "openrouter"}), \
                patch.object(flow, "recognize_openrouter", new_callable=AsyncMock) as primary, \
                patch.object(flow, "recognize_local", new_callable=AsyncMock) as local:
            primary.side_effect = ai.OpenRouterUnavailable("OpenRouter HTTP 429")
            local.return_value = result
            notify = AsyncMock()
            self.assertEqual(await flow.recognize_receipt(b"original", on_fallback=notify), result)
            primary.assert_awaited_once_with(b"original")
            local.assert_awaited_once_with(b"original")
            notify.assert_awaited_once_with()

    async def test_success_does_not_run_tesseract_or_notify(self):
        result = ai.parse_receipt(json.dumps(payload()))
        with patch.dict(os.environ, {"RECEIPT_OCR_BACKEND": "openrouter"}), \
                patch.object(flow, "recognize_openrouter", new_callable=AsyncMock, return_value=result), \
                patch.object(flow, "recognize_local", new_callable=AsyncMock) as local:
            notify = AsyncMock()
            self.assertEqual(await flow.recognize_receipt(b"original", on_fallback=notify), result)
            local.assert_not_awaited()
            notify.assert_not_awaited()

    async def test_actual_missing_key_falls_back_without_http(self):
        with patch.dict(os.environ, {"RECEIPT_OCR_BACKEND": "openrouter", "OPENROUTER_API_KEY": ""}), \
                patch.object(ai, "_post", new_callable=AsyncMock) as post, \
                patch.object(flow, "recognize_local", new_callable=AsyncMock) as local:
            await flow.recognize_receipt(b"original")
            post.assert_not_awaited()
            local.assert_awaited_once_with(b"original")
