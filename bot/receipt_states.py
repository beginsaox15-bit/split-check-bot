"""FSM states shared by manual input and recognized-receipt editing."""
from aiogram.fsm.state import State, StatesGroup


class ManualReceipt(StatesGroup):
    name = State()
    price = State()
    quantity = State()
    review = State()


class ReceiptInput(StatesGroup):
    photo = State()
    review = State()
    confirmed = State()


class ReceiptEditor(StatesGroup):
    items = State()
    item = State()
    name = State()
    quantity = State()
    price = State()
    delete = State()


class ExpenseAllocation(StatesGroup):
    overview = State()
    name = State()
    items = State()
    item = State()
    done = State()
    payments = State()
    payment_amount = State()
    tip_amount = State()
    tip_payer = State()
    discount_amount = State()
