"""Explicit restaurant payments and deterministic, kopeck-exact reimbursements."""
import re
from decimal import Decimal, localcontext

from pydantic import BaseModel, ConfigDict

from bot.expense_split import Allocation, PaymentAmount, participant_totals


class Transfer(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    sender_id: str
    recipient_id: str
    amount: PaymentAmount


def parse_payment(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().replace(',', '.')
    if not re.fullmatch(r'[0-9]{1,15}(?:\.[0-9]{1,2})?', value):
        return None
    return format(Decimal(value), '.2f')


def set_payment(plan: Allocation, person_id: str, value: str) -> Allocation:
    amount = parse_payment(value)
    if (not plan.completed or amount is None
            or person_id not in {person.id for person in plan.participants}):
        raise ValueError('Invalid payment')
    data = plan.model_dump()
    # Payments are absolute totals, not append-only increments. Retries cannot add money.
    data['payments'][person_id] = amount
    data['settlement_confirmed'] = False
    return Allocation.model_validate(data)


def balances(plan: Allocation) -> dict[str, Decimal]:
    owed = participant_totals(plan)
    with localcontext() as context:
        context.prec = 32
        return {person.id: Decimal(plan.payments.get(person.id, '0')) - owed[person.id]
                for person in plan.participants}


def settlement_issues(plan: Allocation) -> list[str]:
    issues = []
    if not plan.completed or plan.unassigned:
        issues.append('allocation_incomplete')
    if plan.mismatch:
        issues.append('receipt_mismatch')
    if plan.paid_total != plan.items_total:
        issues.append('payments_mismatch')
    return issues


def transfers(plan: Allocation) -> list[Transfer]:
    if settlement_issues(plan):
        raise ValueError('Settlement is not reconciled')
    with localcontext() as context:
        context.prec = 32
        values = balances(plan)
        debtors = [[person.id, int(-values[person.id] * 100)]
                   for person in plan.participants if values[person.id] < 0]
        creditors = [[person.id, int(values[person.id] * 100)]
                     for person in plan.participants if values[person.id] > 0]
        result = []
        creditor = 0
        for sender, remaining in debtors:
            while remaining:
                recipient, available = creditors[creditor]
                cents = min(remaining, available)
                result.append(Transfer(sender_id=sender, recipient_id=recipient,
                                       amount=format(Decimal(cents)/100, '.2f')))
                remaining -= cents
                creditors[creditor][1] -= cents
                if creditors[creditor][1] == 0:
                    creditor += 1
        return result


def confirm_settlement(plan: Allocation) -> Allocation:
    if settlement_issues(plan):
        raise ValueError('Settlement is not reconciled')
    return Allocation.model_validate({**plan.model_dump(), 'settlement_confirmed': True})
