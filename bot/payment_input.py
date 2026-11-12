"""Guards for ordinary-text payment input, scoped to one active FSM session."""
from bot.receipt_states import ExpenseAllocation


def new_session(data, person_id, update_id, message_floor, *, kind='payment'):
    if type(update_id) is not int or type(message_floor) is not int:
        raise ValueError('Payment input requires Telegram update/message IDs')
    if kind not in {'payment', 'tip', 'discount'}:
        raise ValueError('Unknown money input kind')
    return {'token': data['allocation_token'], 'revision': data['allocation_revision'], 'kind': kind,
            'person_id': person_id, 'started_update_id': update_id,
            'message_floor': max(message_floor, data.get('payment_last_message_id', 0))}


def accepts_message(data, plan, state_name, *, user_id, chat_id, message_id, update_id, is_reply=False):
    session = data.get('payment_input')
    kind = {ExpenseAllocation.payment_amount.state: 'payment', ExpenseAllocation.tip_amount.state: 'tip',
            ExpenseAllocation.discount_amount.state: 'discount'}.get(state_name)
    if (kind is None or not isinstance(session, dict) or session.get('kind') != kind
            or not plan.completed or is_reply
            or user_id != plan.owner_user_id or chat_id != plan.owner_chat_id):
        return False
    if (session.get('token') != data.get('allocation_token')
            or not isinstance(session.get('token'), str) or not session['token']
            or type(session.get('revision')) is not int
            or session.get('revision') != data.get('allocation_revision')
            or session.get('person_id') != data.get('payment_person')
            or kind == 'payment' and session.get('person_id') not in {person.id for person in plan.participants}
            or kind in {'tip', 'discount'} and session.get('person_id') is not None):
        return False
    values = (message_id, update_id, session.get('started_update_id'), session.get('message_floor'),
              data.get('payment_last_message_id', 0), data.get('payment_last_update_id', -1))
    if any(type(value) is not int for value in values):
        return False
    return (message_id > max(session['message_floor'], data.get('payment_last_message_id', 0))
            and update_id > max(session['started_update_id'], data.get('payment_last_update_id', -1)))
