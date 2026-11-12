"""Display-only currency evidence; never rewrite recognized currency facts."""
import re


def rubles_confirmed(text: str) -> bool:
    if not isinstance(text, str):
        return False
    # Mixed/foreign currency evidence requires review rather than a RUB guess.
    if re.search(r'\b(?:USD|EUR|KZT|BYN|UAH|GBP|CHF|CNY|JPY|PLN|TRY|GEL|AMD|AZN|KGS|UZS|тенге|грн)\b|[$€₸₴£¥₺₾]|белорус', text, re.I):
        return False
    if re.search(r'(?:валюта|currency)\s*[:=]\s*(?!RUB\b)[A-Z]{3}\b', text, re.I):
        return False
    if re.search(r'₽|\bRUB\b', text, re.I):
        return True
    # "Рубли" alone also occurs on Belarusian receipts.
    return bool(re.search(r'\bруб(?:\.|ля|лей)?\b', text, re.I) and (
        re.search(r'российск', text, re.I)
        or re.search(r'\bИНН\s*[:№]?\s*(?:\d{12}|\d{10})\b', text, re.I)))


def receipt_currency(receipt):
    return receipt.currency or ('RUB' if rubles_confirmed(receipt.raw_text) else None)


def symbol(code, *, ruble_evidence=False):
    code = code or ('RUB' if ruble_evidence else None)
    return '₽' if code == 'RUB' else code or ''
