"""Per-receipt allocations. Receipt amounts are facts, never balancing targets."""
import unicodedata
from decimal import Decimal, localcontext

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bot.services.receipt_models import Money


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
    prepayment: Money | None = None
    amount_due: Money | None = None
    currency: str | None = None
    completed: bool = False

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
        return self.gross_total is not None and self.items_total != Decimal(self.gross_total)


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
                      prepayment=data.get('prepayment'), amount_due=data.get('amount_due'),
                      currency=data.get('currency'))


def add_participant(plan: Allocation, name: str) -> Allocation:
    name = name.strip()
    if (plan.completed or not name or len(name) > 60
            or any(unicodedata.category(char).startswith('C') for char in name)):
        raise ValueError('Invalid participant name')
    if any(person.name.casefold() == name.casefold() for person in plan.participants):
        raise ValueError('Participant already exists')
    data = plan.model_dump()
    data['participants'].append({'id': f'p{len(plan.participants)}', 'name': name})
    return Allocation.model_validate(data)


def assign_item(plan: Allocation, item_id: str, people: list[str]) -> Allocation:
    if plan.completed or item_id not in {item.id for item in plan.items}:
        raise ValueError('Invalid item')
    data = plan.model_dump()
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


def complete(plan: Allocation) -> Allocation:
    if plan.unassigned:
        raise ValueError('Unassigned items remain')
    return Allocation.model_validate({**plan.model_dump(), 'completed': True})
