"""Per-receipt allocations. Receipt amounts are facts, never balancing targets."""
import unicodedata
from decimal import Decimal, localcontext
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bot.services.receipt_models import Money
from bot.currency import rubles_confirmed

# A single payer may cover up to 200 maximum-sized item lines.
PaymentAmount = Annotated[str, Field(pattern=r'^(0|[1-9][0-9]{0,14})(\.[0-9]{1,2})?$')]


def decimal_amount(value):
    if isinstance(value, bool) or not isinstance(value, (str, Decimal, int)):
        raise ValueError('Money must not use float')
    amount = Decimal(value)
    if not amount.is_finite():
        raise ValueError('Money must be finite')
    return amount


class SplitItem(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str
    name: str
    quantity: str
    amount: Money
    guest_id: str | None = None


class Participant(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str
    name: str = Field(min_length=1, max_length=60)


class Allocation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    owner_user_id: int
    owner_chat_id: int
    items: list[SplitItem] = Field(min_length=1, max_length=200)
    participants: list[Participant] = Field(default_factory=list, max_length=30)
    assignments: dict[str, list[str]] = Field(default_factory=dict)
    gross_total: Money | None = None
    receipt_total: Money | None = None
    prepayment: Money | None = None
    prepayment_suspect: bool = False
    amount_due: Money | None = None
    currency: str | None = None
    # A display hint derived from explicit text markers, separate from OCR facts.
    ruble_evidence: bool = False
    completed: bool = False
    payments: dict[str, PaymentAmount] = Field(default_factory=dict)
    tip_amount: PaymentAmount = '0.00'
    tip_payer_id: str | None = None
    discount_amount: PaymentAmount = '0.00'
    settlement_confirmed: bool = False

    @model_validator(mode='after')
    def valid_assignments(self):
        items = {item.id for item in self.items}
        people = {person.id for person in self.participants}
        if len(items) != len(self.items) or len(people) != len(self.participants):
            raise ValueError('Duplicate IDs')
        for item, owners in self.assignments.items():
            if item not in items or len(set(owners)) != len(owners) or not set(owners) <= people:
                raise ValueError('Invalid assignment')
        if self.completed and self.unassigned:
            raise ValueError('Unassigned items remain')
        if not set(self.payments) <= people:
            raise ValueError('Unknown payer')
        if self.tip_payer_id is not None and self.tip_payer_id not in people:
            raise ValueError('Unknown tip payer')
        if Decimal(self.tip_amount) == 0 and self.tip_payer_id is not None:
            raise ValueError('Zero tips cannot have a payer')
        if Decimal(self.tip_amount) > 0 and not people:
            raise ValueError('Tips require participants')
        if Decimal(self.discount_amount) > self.items_total:
            raise ValueError('Discount exceeds item total')
        if self.settlement_confirmed and (not self.completed or self.unassigned
                                          or self.mismatch or self.paid_total != self.restaurant_total
                                          or Decimal(self.tip_amount) > 0 and self.tip_payer_id is None):
            raise ValueError('Settlement is not reconciled')
        return self

    @property
    def unassigned(self):
        return [item for item in self.items if not self.assignments.get(item.id)]

    @property
    def items_total(self):
        with localcontext() as context:
            context.prec = 32
            return sum((Decimal(item.amount) for item in self.items), Decimal(0))

    @property
    def mismatch(self):
        return bool(self.receipt_total_issues)

    @property
    def receipt_total_issues(self):
        # Legacy zero-discount reconciliation remains unchanged. A stored positive
        # discount is an explicit user decision, never inferred from OCR totals.
        before = self.items_total
        if Decimal(self.discount_amount) == 0:
            return ['gross_mismatch'] if self.gross_total is not None and before != Decimal(self.gross_total) else []
        after = self.restaurant_total
        issues = []
        for value in (self.gross_total, self.receipt_total):
            if value is not None and Decimal(value) not in {before, after}:
                issues.append('unexplained_total')
                break
        # A user-entered discount may explain distinct totals in this explicit
        # direction: gross is before discount, receipt total is after discount.
        # Other unequal pairs remain ambiguous; never infer a missing discount.
        if (self.gross_total is not None and self.receipt_total is not None
                and Decimal(self.gross_total) != Decimal(self.receipt_total)
                and not (Decimal(self.gross_total) == before and Decimal(self.receipt_total) == after)):
            issues.append('conflicting_totals')
        if self.prepayment_suspect:
            issues.append('suspect_prepayment')
        prepaid = Decimal(self.prepayment) if self.prepayment is not None else None
        due = Decimal(self.amount_due) if self.amount_due is not None else None
        if prepaid is not None and prepaid > after:
            issues.append('prepayment_exceeds_restaurant')
        if due is not None:
            with localcontext() as context:
                context.prec = 32
                if prepaid is None and due != after:
                    issues.append('ambiguous_due_without_prepayment')
                elif prepaid is not None and prepaid + due != after:
                    issues.append('payment_totals_mismatch')
        return issues

    @property
    def restaurant_total(self):
        with localcontext() as context:
            context.prec = 32
            return self.items_total - Decimal(self.discount_amount)

    @property
    def paid_total(self):
        with localcontext() as context:
            context.prec = 32
            return sum((Decimal(value) for value in self.payments.values()), Decimal(0))

    @property
    def expenses_total(self):
        with localcontext() as context:
            context.prec = 32
            return self.restaurant_total + Decimal(self.tip_amount)


def snapshot(data: dict, *, user_id: int, chat_id: int) -> Allocation:
    items = []
    for index, source in enumerate(data.get('items', [])):
        if 'total_price' in source:
            amount = decimal_amount(source['total_price'])
        else:
            # The existing manual input accepts arbitrary precision; do not round it
            # silently when converting to amounts that must be split in kopecks.
            with localcontext() as context:
                context.prec = max(40, len(str(source['price'])) + len(str(source['quantity'])) + 4)
                amount = decimal_amount(source['price']) * decimal_amount(source['quantity'])
        with localcontext() as context:
            context.prec = max(32, len(amount.as_tuple().digits) + 4)
            if amount != amount.quantize(Decimal('.01')):
                raise ValueError('Item amount must be an exact number of kopecks')
        items.append(SplitItem(id=f'i{index}', name=source['name'], quantity=str(source['quantity']),
                               amount=format(amount, '.2f'), guest_id=source.get('guest_id')))
    return Allocation(owner_user_id=user_id, owner_chat_id=chat_id, items=items,
                      gross_total=data.get('gross_total') or data.get('receipt_total'),
                      receipt_total=data.get('receipt_total'),
                      prepayment=data.get('prepayment'), amount_due=data.get('amount_due'),
                      prepayment_suspect=data.get('prepayment_suspect', False),
                      currency=data.get('currency'),
                      ruble_evidence=data.get('ruble_evidence', False) or rubles_confirmed(data.get('raw_text', '')))


def add_participant(plan: Allocation, name: str) -> Allocation:
    name = name.strip()
    if (plan.completed or not name or len(name) > 60
            or any(unicodedata.category(char).startswith('C') for char in name)):
        raise ValueError('Invalid participant name')
    if any(person.name.casefold() == name.casefold() for person in plan.participants):
        raise ValueError('Participant already exists')
    data = plan.model_dump()
    data['settlement_confirmed'] = False
    data['participants'].append({'id': f'p{len(plan.participants)}', 'name': name})
    return Allocation.model_validate(data)


def assign_item(plan: Allocation, item_id: str, people: list[str]) -> Allocation:
    if plan.completed or item_id not in {item.id for item in plan.items}:
        raise ValueError('Invalid item')
    data = plan.model_dump()
    data['settlement_confirmed'] = False
    data['assignments'][item_id] = list(people)
    return Allocation.model_validate(data)


def split_amount(amount: str, people: list[str]) -> dict[str, Decimal]:
    if not people or len(set(people)) != len(people):
        raise ValueError('Choose distinct participants')
    value = decimal_amount(amount)
    if not value.is_finite() or value < 0:
        raise ValueError('Amount must contain whole kopecks')
    with localcontext() as context:
        context.prec = max(32, len(value.as_tuple().digits) + 4)
        if value != value.quantize(Decimal('.01')):
            raise ValueError('Amount must contain whole kopecks')
        cents = int(value * 100)
        share, remainder = divmod(cents, len(people))
        return {person: Decimal(share + (index < remainder)) / 100
                for index, person in enumerate(people)}


def participant_totals(plan: Allocation) -> dict[str, Decimal]:
    totals = {person.id: Decimal(0) for person in plan.participants}
    with localcontext() as context:
        context.prec = 32
        for item in plan.items:
            # Insertion order is the stable tiebreaker, independent of click order.
            owners = [person.id for person in plan.participants
                      if person.id in plan.assignments.get(item.id, [])]
            if owners:
                for person, amount in split_amount(item.amount, owners).items():
                    totals[person] += amount
    return totals


def discount_shares(plan: Allocation) -> dict[str, Decimal]:
    """Largest remainders on pre-discount restaurant shares, in whole kopecks."""
    result = {person.id: Decimal(0) for person in plan.participants}
    if Decimal(plan.discount_amount) == 0:
        return result
    if plan.unassigned:
        raise ValueError('Complete all assignments before splitting a discount')
    with localcontext() as context:
        context.prec = 32
        shares = participant_totals(plan)
        total_cents = int(plan.items_total * 100)
        discount_cents = int(Decimal(plan.discount_amount) * 100)
        # Integer products/divmod preserve fractional remainders exactly, even
        # when monetary amounts approach their maximum supported size.
        parts = [divmod(discount_cents * int(shares[p.id]*100), total_cents) for p in plan.participants]
        cents = [quotient for quotient, _ in parts]
        remainder = discount_cents - sum(cents)
        order = sorted(range(len(parts)), key=lambda i: (-parts[i][1], i))
        for index in order[:remainder]:
            cents[index] += 1
        return {person.id: Decimal(cents[index])/100 for index, person in enumerate(plan.participants)}


def restaurant_shares(plan: Allocation) -> dict[str, Decimal]:
    before, discounts = participant_totals(plan), discount_shares(plan)
    with localcontext() as context:
        context.prec = 32
        return {person.id: before[person.id] - discounts[person.id] for person in plan.participants}


def complete(plan: Allocation) -> Allocation:
    if plan.unassigned:
        raise ValueError('Unassigned items remain')
    return Allocation.model_validate({**plan.model_dump(), 'completed': True})
