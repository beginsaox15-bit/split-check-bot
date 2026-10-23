"""Модели и ошибки локального OCR. Деньги сериализуются строками, считаются Decimal."""

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
    guest_id: str | None = None

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
    raw_text: str = ""
    gross_total: Money | None = None
    prepayment: Money | None = None
    amount_due: Money | None = None
    guest_subtotals: dict[str, Money] = Field(default_factory=dict)
    skipped_lines: int = Field(default=0, ge=0)
    parse_warnings: list[str] = Field(default_factory=list)

    @property
    def comparison_total(self) -> Decimal | None:
        if self.gross_total is not None:
            return Decimal(self.gross_total)
        if self.prepayment is not None and self.amount_due is not None:
            return Decimal(self.prepayment) + Decimal(self.amount_due)
        return Decimal(self.receipt_total) if self.receipt_total is not None else None

    @property
    def guest_subtotals_total(self) -> Decimal:
        with localcontext() as context:
            context.prec = 32
            return sum((Decimal(value) for value in self.guest_subtotals.values()), Decimal(0))

    @property
    def guest_subtotals_mismatch(self) -> bool:
        return bool(self.guest_subtotals) and self.comparison_total is not None and self.guest_subtotals_total != self.comparison_total

    @property
    def payment_mismatch(self) -> bool:
        if self.gross_total is None or self.prepayment is None or self.amount_due is None:
            return False
        return Decimal(self.gross_total) - Decimal(self.prepayment) != Decimal(self.amount_due)

    @property
    def items_total(self) -> Decimal:
        # Значения ограничены схемой; 32 знака достаточно для суммы 200 строк.
        with localcontext() as context:
            context.prec = 32
            return sum((Decimal(item.total_price) for item in self.items), Decimal(0))

    @property
    def totals_mismatch(self) -> bool:
        return self.comparison_total is not None and self.items_total != self.comparison_total


class RecognitionError(Exception):
    """Ошибка распознавания без содержимого фотографии в сообщении."""


class UnreadableReceipt(RecognitionError):
    pass


class RecognitionUnavailable(RecognitionError):
    pass


class TesseractNotInstalled(RecognitionUnavailable):
    pass


class TesseractLanguagesUnavailable(RecognitionUnavailable):
    pass


class UnparsedReceipt(UnreadableReceipt):
    """Текст прочитан, но parser не смог выделить надёжные позиции."""

    def __init__(self, raw_text: str):
        super().__init__("No reliable purchase lines found")
        self.raw_text = raw_text


class LowQualityReceipt(UnreadableReceipt):
    """Слишком мало уверенно распознанного текста; parser не запускался."""

    def __init__(self, raw_text: str):
        super().__init__("Insufficient reliable OCR text")
        self.raw_text = raw_text


