"""Общие модели и ошибки локального и опционального облачного OCR."""

from decimal import Decimal, localcontext
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Строки в JSON не допускают потери точности денежных значений через float.
Money = Annotated[str, Field(pattern=r"^(0|[1-9][0-9]{0,11})(\.[0-9]{1,2})?$")]
Quantity = Annotated[str, Field(pattern=r"^(0|[1-9][0-9]{0,8})(\.[0-9]{1,6})?$")]


class ReceiptItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    quantity: Quantity = Field(description="Положительное количество десятичной строкой, например '2' или '0.5'.")
    unit_price: Money | None
    total_price: Money

    @field_validator("name")
    @classmethod
    def nonblank_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Empty item name")
        return value.strip()

    @field_validator("quantity")
    @classmethod
    def positive_quantity(cls, value: str) -> str:
        if Decimal(value) <= 0:
            raise ValueError("Quantity must be positive")
        return value


class RecognizedReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    readable: bool = Field(description="False, если изображение нечитаемо, не является чеком или позиции ненадёжны.")
    items: list[ReceiptItem] = Field(max_length=200)
    receipt_total: Money | None
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] | None

    @property
    def items_total(self) -> Decimal:
        # Значения ограничены схемой; 32 знака достаточно для суммы 200 строк.
        with localcontext() as context:
            context.prec = 32
            return sum((Decimal(item.total_price) for item in self.items), Decimal(0))

    @property
    def totals_mismatch(self) -> bool:
        return self.receipt_total is not None and self.items_total != Decimal(self.receipt_total)


class RecognitionError(Exception):
    """Безопасная для обработки в Telegram ошибка без содержимого API-запроса."""


class MissingAPIKey(RecognitionError):
    pass


class UnreadableReceipt(RecognitionError):
    pass


class RecognitionUnavailable(RecognitionError):
    pass


