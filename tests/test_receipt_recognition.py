import base64
import json
import os
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

try:
    import httpx
    from openai import APIConnectionError, AsyncOpenAI
    from bot.services.openai_receipt_recognition import recognize_receipt
except ImportError:
    AsyncOpenAI = None
from pydantic import ValidationError

from bot.services.receipt_models import (
    MissingAPIKey, RecognizedReceipt, RecognitionUnavailable, UnreadableReceipt,
)


def receipt_payload() -> dict:
    return {
        "readable": True,
        "items": [
            {"name": "Бургер", "quantity": "2", "unit_price": "850.00", "total_price": "1700.00"},
            {"name": "Коктейль", "quantity": "1", "unit_price": None, "total_price": "650.00"},
        ],
        "receipt_total": "2350.00", "currency": "RUB",
    }


@unittest.skipIf(AsyncOpenAI is None, "Optional OpenAI dependencies are not installed")
class RecognitionServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.parse = AsyncMock(return_value=SimpleNamespace(
            status="completed", output_parsed=RecognizedReceipt.model_validate(receipt_payload()),
        ))
        self.client = SimpleNamespace(responses=SimpleNamespace(parse=self.parse))

    async def test_structured_image_request_multiple_items_and_quantity(self) -> None:
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4.1-mini"}):
            result = await recognize_receipt(b"image-bytes", client=self.client)
        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.items[0].quantity, "2")
        self.assertIsNone(result.items[1].unit_price)
        self.assertEqual(result.items_total, Decimal("2350.00"))
        self.assertFalse(result.totals_mismatch)
        kwargs = self.parse.await_args.kwargs
        self.assertIs(kwargs["text_format"], RecognizedReceipt)
        self.assertFalse(kwargs["store"])
        self.assertEqual(kwargs["model"], "gpt-4.1-mini")
        image = kwargs["input"][0]["content"][1]
        self.assertEqual(image["type"], "input_image")
        self.assertEqual(base64.b64decode(image["image_url"].split(",", 1)[1]), b"image-bytes")

    async def test_exact_decimal_sum_and_fractional_quantity(self) -> None:
        data = receipt_payload()
        data["items"][0].update(quantity="0.5", unit_price="0.20", total_price="0.10")
        data["items"][1].update(total_price="0.20")
        data["receipt_total"] = "0.30"
        self.parse.return_value.output_parsed = RecognizedReceipt.model_validate(data)
        result = await recognize_receipt(b"image", client=self.client)
        self.assertEqual(result.items_total, Decimal("0.30"))
        self.assertFalse(result.totals_mismatch)

    async def test_total_mismatch(self) -> None:
        data = receipt_payload()
        data["receipt_total"] = "2350.01"
        self.parse.return_value.output_parsed = RecognizedReceipt.model_validate(data)
        result = await recognize_receipt(b"image", client=self.client)
        self.assertTrue(result.totals_mismatch)

    async def test_empty_and_unreadable_receipts_are_rejected(self) -> None:
        for readable, items in ((True, []), (False, receipt_payload()["items"])):
            with self.subTest(readable=readable):
                data = receipt_payload()
                data.update(readable=readable, items=items)
                self.parse.return_value.output_parsed = RecognizedReceipt.model_validate(data)
                with self.assertRaises(UnreadableReceipt):
                    await recognize_receipt(b"image", client=self.client)

    async def test_refusal_and_incomplete_response(self) -> None:
        for response in (
            SimpleNamespace(status="completed", output_parsed=None),
            SimpleNamespace(status="incomplete", output_parsed=RecognizedReceipt.model_validate(receipt_payload())),
        ):
            self.parse.return_value = response
            with self.assertRaises(UnreadableReceipt):
                await recognize_receipt(b"image", client=self.client)

    async def test_api_error_is_sanitized_and_logged(self) -> None:
        secret = "sensitive-request-content"
        self.parse.side_effect = APIConnectionError(
            message=secret, request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        )
        with self.assertLogs("bot.services.openai_receipt_recognition", level="WARNING") as logs:
            with self.assertRaises(RecognitionUnavailable) as error:
                await recognize_receipt(b"image", client=self.client)
        self.assertNotIn(secret, str(error.exception))
        self.assertNotIn(secret, " ".join(logs.output))
        self.assertIn("APIConnectionError", " ".join(logs.output))

    async def test_missing_key_does_not_construct_client_or_request_api(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("bot.services.openai_receipt_recognition.AsyncOpenAI") as client:
                with self.assertRaises(MissingAPIKey):
                    await recognize_receipt(b"image")
        client.assert_not_called()

    async def test_invalid_structured_output_is_handled(self) -> None:
        try:
            RecognizedReceipt.model_validate({})
        except ValidationError as error:
            self.parse.side_effect = error
        with self.assertRaises(UnreadableReceipt):
            await recognize_receipt(b"image", client=self.client)

    async def test_sdk_contract_uses_responses_json_schema_without_network(self) -> None:
        # Настоящий SDK работает с локальным MockTransport: платных запросов нет.
        requests = []

        def transport(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "resp_offline", "object": "response", "created_at": 1,
                "status": "completed", "model": "gpt-4.1-mini",
                "output": [{"type": "message", "id": "msg_offline", "role": "assistant",
                            "status": "completed", "content": [{"type": "output_text",
                            "text": json.dumps(receipt_payload()), "annotations": []}]}],
            })

        async with AsyncOpenAI(
            api_key="offline-test-only", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
        ) as client:
            result = await recognize_receipt(b"image", client=client)
        self.assertEqual(result.items_total, Decimal("2350.00"))
        self.assertEqual(len(requests), 1)
        schema = requests[0]["text"]["format"]
        self.assertEqual(schema["type"], "json_schema")
        self.assertTrue(schema["strict"])
        self.assertFalse(requests[0]["store"])

    def test_money_is_string_and_invalid_values_are_rejected(self) -> None:
        for field, value in (("total_price", 0.1), ("total_price", "-1"),
                             ("total_price", "NaN"), ("quantity", "0"), ("name", " ")):
            with self.subTest(field=field, value=value):
                data = receipt_payload()
                data["items"][0][field] = value
                with self.assertRaises(ValidationError):
                    RecognizedReceipt.model_validate(data)
