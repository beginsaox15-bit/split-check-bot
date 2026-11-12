"""Offline JSON-to-JSON receipt evaluation, using only the standard library.

Reports deliberately contain indices and field statuses instead of source names,
guest metadata, raw OCR text or images. No bot code, OCR engine or API is used.
"""
import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal, localcontext
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sys
import unicodedata

MAX_ITEMS = 200
MAX_FILE_BYTES = 2 * 1024 * 1024
NUMBER = re.compile(r'-?[0-9]{1,30}(?:\.[0-9]{1,12})?\Z')
TOTAL_FIELDS = ('gross_total', 'prepayment', 'amount_due', 'receipt_total')
ITEM_FIELDS = ('name', 'quantity', 'total_price')


class InputError(ValueError):
    """Only fixed schema paths/codes may be exposed, never input values/paths."""


@dataclass(frozen=True)
class Item:
    name: str | None
    quantity: Decimal | None
    total_price: Decimal | None
    errors: dict[str, str]


def normalized_name(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().replace('ё', 'е').split())


def decimal_value(value):
    # JSON numbers are loaded directly as Decimal; Python float/bool are rejected.
    if type(value) not in (str, int, Decimal):
        return None
    if isinstance(value, Decimal) and (not value.is_finite() or value.adjusted() >= 30
                                     or not -12 <= value.as_tuple().exponent <= 30):
        return None
    if type(value) is int and abs(value) >= 10**30:
        return None
    text = format(value, 'f') if isinstance(value, Decimal) else str(value)
    if not NUMBER.fullmatch(text):
        return None
    return Decimal(text)


def money_text(value):
    return format(value, 'f') if value is not None else None


def parse_items(data, side):
    aliases = ('items', 'expected_items') if side == 'expected' else ('items',)
    present = [key for key in aliases if key in data]
    if len(present) != 1 or not isinstance(data[present[0]], list):
        raise InputError(f'{side}.items: array_required_or_ambiguous')
    rows = data[present[0]]
    if len(rows) > MAX_ITEMS:
        raise InputError(f'{side}.items: too_many_items')
    items = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise InputError(f'{side}.items[{index}]: object_required')
        errors = {}
        name = row.get('name')
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            errors['name'] = 'missing' if name is None else 'invalid'
            name = None
        else:
            name = normalized_name(name)
        values = {}
        for field in ('quantity', 'total_price'):
            raw = row.get(field)
            values[field] = decimal_value(raw)
            if values[field] is None:
                errors[field] = 'missing' if raw is None else 'invalid'
        if side == 'expected' and errors:
            field = next(field for field in ITEM_FIELDS if field in errors)
            raise InputError(f'expected.items[{index}].{field}: {errors[field]}')
        items.append(Item(name, values['quantity'], values['total_price'], errors))
    return items


def total_value(data, field, side):
    alias = 'expected_total' if field == 'receipt_total' else 'expected_' + field
    keys = [key for key in ((field, alias) if side == 'expected' else (field,)) if key in data]
    if len(keys) > 1:
        raise InputError(f'{side}.{field}: ambiguous_fields')
    raw = data[keys[0]] if keys else None
    value = decimal_value(raw)
    status = 'missing' if raw is None else 'invalid' if value is None else 'valid'
    if side == 'expected' and status == 'invalid':
        raise InputError(f'expected.{field}: invalid')
    return value, status


def name_similarity(first, second):
    if first is None or second is None:
        return 0
    if first == second:
        return 1000
    # Integer score, including its 800 threshold; no float conversions.
    matching = sum(block.size for block in SequenceMatcher(None, first, second, autojunk=False).get_matching_blocks())
    return 2000 * matching // (len(first) + len(second))


def field_statuses(expected, actual):
    return {field: actual.errors.get(field) or (
        'match' if getattr(expected, field) == getattr(actual, field) else 'mismatch') for field in ITEM_FIELDS}


def best_pairs(row_count, column_count, weights):
    """Maximum-weight one-to-one assignment (Hungarian), with unmatched dummies.

    Stable index iteration breaks ties. Forbidden/unused edges have weight zero;
    every eligible partial match has positive weight. O(n^2 * (n + m)).
    """
    if not row_count or not column_count or not weights:
        return []
    columns = column_count + row_count
    u, v, owners, ways = [0]*(row_count+1), [0]*(columns+1), [0]*(columns+1), [0]*(columns+1)
    infinity = (max(weights.values()) + 1) * (row_count + columns + 1)
    for row in range(1, row_count+1):
        owners[0], column = row, 0
        minimum, visited = [infinity]*(columns+1), [False]*(columns+1)
        while True:
            visited[column] = True
            current_row, delta, next_column = owners[column], infinity, 0
            for candidate in range(1, columns+1):
                if visited[candidate]:
                    continue
                cost = -weights.get((current_row-1, candidate-1), 0) - u[current_row] - v[candidate]
                if cost < minimum[candidate]:
                    minimum[candidate], ways[candidate] = cost, column
                if minimum[candidate] < delta:
                    delta, next_column = minimum[candidate], candidate
            for candidate in range(columns+1):
                if visited[candidate]:
                    u[owners[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if owners[column] == 0:
                break
        while column:
            previous = ways[column]
            owners[column] = owners[previous]
            column = previous
    return sorted((owners[column]-1, column-1) for column in range(1, column_count+1)
                  if owners[column] and (owners[column]-1, column-1) in weights)


def match_items(expected, actual):
    available = defaultdict(deque)
    for index, item in enumerate(actual):
        if not item.errors:
            available[(item.name, item.quantity, item.total_price)].append(index)
    full = []
    for index, item in enumerate(expected):
        candidates = available[(item.name, item.quantity, item.total_price)]
        if candidates:
            full.append((index, candidates.popleft(), 'exact_fields'))
    used_expected, used_actual = {p[0] for p in full}, {p[1] for p in full}
    er = [i for i in range(len(expected)) if i not in used_expected]
    ar = [i for i in range(len(actual)) if i not in used_actual]
    signatures_expected, signatures_actual = defaultdict(int), defaultdict(int)
    for indices, items, counts in ((er, expected, signatures_expected), (ar, actual, signatures_actual)):
        for index in indices:
            item = items[index]
            if item.quantity is not None and item.total_price is not None:
                counts[(item.quantity, item.total_price)] += 1
    n = max(len(er), len(ar), 1)
    # Lexicographic objectives: same-name identities, equal fields, pair count,
    # name similarity. Each tier outweighs ALL lower tiers across the assignment.
    pair_unit = n*1000 + 1
    field_unit = n*(pair_unit+1000) + 1
    identity_unit = n*(3*field_unit+pair_unit+1000) + 1
    weights, bases, similarities = {}, {}, {}
    for row, ei in enumerate(er):
        wanted = expected[ei]
        for column, ai in enumerate(ar):
            found = actual[ai]
            key = (wanted.name, found.name)
            if key not in similarities:
                similarities[key] = name_similarity(*key)
            similarity = similarities[key]
            same_name = wanted.name == found.name
            numbers_equal = wanted.quantity == found.quantity and wanted.total_price == found.total_price
            signature = (wanted.quantity, wanted.total_price)
            unique_numbers = (numbers_equal and signatures_expected[signature] == 1
                              and signatures_actual[signature] == 1)
            if not (same_name or similarity >= 800 or unique_numbers):
                continue
            equal = sum(value == 'match' for value in field_statuses(wanted, found).values())
            weights[(row, column)] = same_name*identity_unit + equal*field_unit + pair_unit + similarity
            bases[(row, column)] = 'same_name' if same_name else 'similar_name' if similarity >= 800 else 'unique_quantity_and_price'
    partial = [(er[row], ar[column], bases[(row, column)])
               for row, column in best_pairs(len(er), len(ar), weights)]
    used_expected.update(p[0] for p in partial)
    used_actual.update(p[1] for p in partial)
    records = [{'expected_index': ei+1, 'actual_index': ai+1, 'classification': category,
                'pairing_basis': basis, 'fields': field_statuses(expected[ei], actual[ai])}
               for category, pairs in (('full', full), ('partial', partial)) for ei, ai, basis in pairs]
    return (sorted(records, key=lambda row: row['expected_index']),
            [i+1 for i in range(len(expected)) if i not in used_expected],
            [i+1 for i in range(len(actual)) if i not in used_actual])


def evaluate(expected, actual):
    """Compare caller-provided JSON objects without mutating or echoing them."""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        raise InputError('root: object_required')
    expected_items, actual_items = parse_items(expected, 'expected'), parse_items(actual, 'actual')
    records, missing, extra = match_items(expected_items, actual_items)
    totals = {}
    with localcontext() as context:
        context.prec = 60  # Bounded decimals, up to 200 rows; never round inputs.
        expected_sum = sum((i.total_price for i in expected_items), Decimal(0))
        actual_known = sum((i.total_price for i in actual_items if i.total_price is not None), Decimal(0))
        unknown = sum(i.total_price is None for i in actual_items)
        actual_sum = None if unknown else actual_known
        for field in TOTAL_FIELDS:
            wanted, wanted_status = total_value(expected, field, 'expected')
            found, found_status = total_value(actual, field, 'actual')
            status = ('not_provided' if wanted_status == 'missing' else found_status
                      if found_status != 'valid' else 'match' if wanted == found else 'mismatch')
            totals[field] = {'status': status, 'actual_status': found_status,
                             'expected': money_text(wanted), 'actual': money_text(found),
                             'actual_minus_expected': money_text(found-wanted) if wanted is not None and found is not None else None}
        expected_gross = total_value(expected, 'gross_total', 'expected')[0]
        actual_gross = total_value(actual, 'gross_total', 'actual')[0]
        sums = {'expected': money_text(expected_sum), 'actual': money_text(actual_sum),
                'actual_known': money_text(actual_known), 'actual_unknown_prices': unknown,
                'actual_minus_expected': money_text(actual_sum-expected_sum) if actual_sum is not None else None,
                'expected_gross_minus_items': money_text(expected_gross-expected_sum) if expected_gross is not None else None,
                'actual_gross_minus_items': money_text(actual_gross-actual_sum) if actual_gross is not None and actual_sum is not None else None}
    full = sum(row['classification'] == 'full' for row in records)
    partial = len(records)-full
    return {'report_version': 1,
            'rules': {'full_match': 'NFKC/casefold/ё→е/whitespace-normalized name AND exact Decimal quantity AND line total',
                      'partial_match': 'same name; similarity >=800/1000; or unique equal quantity+total among remaining rows',
                      'assignment': 'full first; global one-to-one partial assignment; stable index ties',
                      'delta_direction': 'actual minus expected', 'source_text_included': False},
            'summary': {'expected_count': len(expected_items), 'actual_count': len(actual_items),
                        'full_matches': full, 'partial_matches': partial, 'missing_count': len(missing), 'extra_count': len(extra),
                        'matched_fields': {field: sum(row['fields'][field] == 'match' for row in records) for field in ITEM_FIELDS}},
            'matches': records, 'missing_expected_indices': missing, 'extra_actual_indices': extra,
            'actual_field_errors': [{'actual_index': i+1, 'field': field, 'status': status}
                                    for i, item in enumerate(actual_items) for field, status in item.errors.items()],
            'item_totals': sums, 'totals': totals,
            'has_differences': bool(partial or missing or extra or any(v['status'] in {'mismatch','missing','invalid'} for v in totals.values()))}


def load_json(path, side):
    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise InputError(f'{side}: duplicate_json_key')
            result[key] = value
        return result

    def nonfinite(_value):
        raise InputError(f'{side}: nonfinite_json_number')

    try:
        with Path(path).open('rb') as stream:
            data = stream.read(MAX_FILE_BYTES+1)
        if len(data) > MAX_FILE_BYTES:
            raise InputError(f'{side}: file_too_large')
        return json.loads(data, parse_float=Decimal, parse_int=Decimal,
                          parse_constant=nonfinite, object_pairs_hook=pairs)
    except (OSError, ValueError, UnicodeError, RecursionError) as error:
        if isinstance(error, InputError):
            raise
        raise InputError(f'{side}: unreadable_or_invalid_json') from None


def text_report(report):
    s = report['summary']
    lines = [f"Позиций: эталон {s['expected_count']}, OCR {s['actual_count']}",
             f"Полностью совпало: {s['full_matches']}/{s['expected_count']}",
             f"Частично сопоставлено: {s['partial_matches']}",
             'Пропущены (номера эталона): ' + (', '.join(map(str, report['missing_expected_indices'])) or 'нет'),
             'Лишние (номера OCR): ' + (', '.join(map(str, report['extra_actual_indices'])) or 'нет')]
    for row in report['matches']:
        if row['classification'] == 'partial':
            fields = ', '.join(f'{field}={status}' for field, status in row['fields'].items() if status != 'match')
            lines.append(f"Эталон #{row['expected_index']} → OCR #{row['actual_index']}: {fields}; основание: {row['pairing_basis']}")
    for row in report['actual_field_errors']:
        lines.append(f"OCR #{row['actual_index']}: {row['field']}={row['status']}")
    sums = report['item_totals']
    lines += [f"Сумма позиций: эталон {sums['expected']}; OCR {sums['actual'] or 'неполная'}",
              f"Разница суммы позиций (OCR − эталон): {sums['actual_minus_expected'] if sums['actual_minus_expected'] is not None else 'не вычисляется'}",
              f"Известная часть суммы OCR: {sums['actual_known']}; неизвестных сумм: {sums['actual_unknown_prices']}",
              f"Gross − сумма позиций: эталон={sums['expected_gross_minus_items']}; OCR={sums['actual_gross_minus_items']}"]
    for field, row in report['totals'].items():
        lines.append(f"{field}: {row['status']}; эталон={row['expected']}; OCR={row['actual']}; разница={row['actual_minus_expected']}")
    lines.append('Частичное сопоставление — диагностическая гипотеза, а не полное совпадение.')
    lines.append('Названия и приватные поля не включены; номера позиций начинаются с 1.')
    return '\n'.join(lines)


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, 'Ошибка: неверные параметры CLI. Используйте --help.\n')


def main(argv=None):
    parser = SafeArgumentParser(prog='python -m tools.evaluate_receipt', description='Офлайн-сравнение двух JSON чеков без OCR и API.')
    parser.add_argument('--expected', required=True)
    parser.add_argument('--actual', required=True)
    parser.add_argument('--format', choices=('text', 'json'), default='text')
    parser.add_argument('--fail-on-difference', action='store_true')
    args = parser.parse_args(argv)
    try:
        report = evaluate(load_json(args.expected, 'expected'), load_json(args.actual, 'actual'))
    except InputError as error:
        print(f'Ошибка: {error}', file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.format == 'json' else text_report(report))
    return 1 if args.fail_on_difference and report['has_differences'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
