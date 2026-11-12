"""Explicit restaurant payments and deterministic, kopeck-exact reimbursements."""
import re
from decimal import Decimal, localcontext

from pydantic import BaseModel, ConfigDict

from bot.expense_split import Allocation, PaymentAmount, participant_totals, split_amount


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


def set_tips(plan: Allocation, value: str, payer_id: str | None) -> Allocation:
    amount = parse_payment(value)
    if (not plan.completed or amount is None or not plan.participants
            or Decimal(amount) > 0 and payer_id not in {p.id for p in plan.participants}):
        raise ValueError('Invalid tips or payer')
    return Allocation.model_validate({**plan.model_dump(), 'tip_amount': amount,
        'tip_payer_id': payer_id if Decimal(amount) > 0 else None, 'settlement_confirmed': False})


def tip_shares(plan: Allocation) -> dict[str, Decimal]:
    return split_amount(plan.tip_amount, [p.id for p in plan.participants]) if plan.participants else {}


def expense_shares(plan: Allocation) -> dict[str, Decimal]:
    owed = participant_totals(plan)
    tips = tip_shares(plan)
    with localcontext() as context:
        context.prec = 32
        return {p.id: owed[p.id] + tips[p.id] for p in plan.participants}


def paid_by(plan: Allocation, person_id: str) -> Decimal:
    with localcontext() as context:
        context.prec = 32
        return Decimal(plan.payments.get(person_id, '0')) + (
            Decimal(plan.tip_amount) if plan.tip_payer_id == person_id else Decimal(0))


def balances(plan: Allocation) -> dict[str, Decimal]:
    owed = expense_shares(plan)
    with localcontext() as context:
        context.prec = 32
        return {person.id: paid_by(plan, person.id) - owed[person.id]
                for person in plan.participants}


def settlement_issues(plan: Allocation) -> list[str]:
    issues = []
    if not plan.completed or plan.unassigned:
        issues.append('allocation_incomplete')
    if plan.mismatch:
        issues.append('receipt_mismatch')
    if plan.paid_total != plan.items_total:
        issues.append('payments_mismatch')
    if Decimal(plan.tip_amount) > 0 and plan.tip_payer_id is None:
        issues.append('tips_unpaid')
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
