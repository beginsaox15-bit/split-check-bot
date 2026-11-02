"""Модели и ошибки локального OCR. Деньги сериализуются строками, считаются Decimal."""

from decimal import Decimal, localcontext
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    suspicious: bool = False
    suspicion_reasons: list[str] = Field(default_factory=list)
    was_reread: bool = False

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


class ParserLineDecision(BaseModel):
    line_number: int
    category: Literal["accepted_item", "rejected_candidate", "ignored_service"]
    reason: str
    item_index: int | None = None


class ParserDiagnostics(BaseModel):
    ocr_lines: int = 0
    accepted_item_lines: int = 0
    rejected_candidate_lines: int = 0
    ignored_service_lines: int = 0
    lines: list[ParserLineDecision] = Field(default_factory=list)


class LineReviewDiagnostics(BaseModel):
    suspicious_items: int = 0
    successfully_reread_items: int = 0
    unresolved_suspicious_items: int = 0
    removed_service_items: int = 0
    original_item_sum: str
    corrected_item_sum: str
    extra_seconds: float = 0
    # Только локальные диагностические данные; не выводятся в Telegram.
    lines: list[dict] = Field(default_factory=list)


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
    parser_diagnostics: ParserDiagnostics | None = None
    prepayment_ocr: Money | None = None
    prepayment_suspect: bool = False
    prepayment_verified: bool = False
    line_review: LineReviewDiagnostics | None = None
    # Вычисляется только при уверенно прочитанных предоплате и остатке.
    # Не заменяет ни одно поле, прочитанное с чека.
    derived_gross_total: Money | None = None

    @property
    def diagnostic_gross_total(self) -> Decimal | None:
        return self.comparison_total if self.comparison_total is not None else (
            Decimal(self.derived_gross_total) if self.derived_gross_total is not None else None)

    @property
    def missing_difference(self) -> Decimal | None:
        total = self.diagnostic_gross_total
        return total - self.items_total if total is not None else None

    @model_validator(mode="after")
    def check_prepayment(self):
        self.prepayment_verified = False
        if self.gross_total is not None and self.amount_due is not None and self.prepayment is not None:
            if Decimal(self.gross_total) - Decimal(self.prepayment) != Decimal(self.amount_due):
                self.prepayment_ocr = self.prepayment
                self.prepayment = None
                self.prepayment_suspect = True
            else:
                self.prepayment_verified = True
        return self

    @property
    def calculated_prepayment(self) -> Decimal | None:
        if self.gross_total is None or self.amount_due is None:
            return None
        value = Decimal(self.gross_total) - Decimal(self.amount_due)
        return value if value >= 0 else None

    @property
    def comparison_total(self) -> Decimal | None:
        if self.gross_total is not None:
            return Decimal(self.gross_total)
        # Непроверенная OCR-предоплата не служит основанием для вычисления gross.
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
        if self.prepayment_suspect:
            return True
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

    def __init__(self, raw_text: str, parser_diagnostics: ParserDiagnostics | None = None):
        super().__init__("No reliable purchase lines found")
        self.raw_text = raw_text
        self.parser_diagnostics = parser_diagnostics


class LowQualityReceipt(UnreadableReceipt):
    """Слишком мало уверенно распознанного текста; parser не запускался."""

    def __init__(self, raw_text: str):
        super().__init__("Insufficient reliable OCR text")
        self.raw_text = raw_text


