"""Pure evaluation tests: no OCR, Telegram, image loading or external APIs."""
from copy import deepcopy
from decimal import Decimal, localcontext
from itertools import permutations, product
import json
from pathlib import Path
import tempfile
import unittest

from tools.evaluate_receipt import InputError, best_pairs, decimal_value, evaluate, load_json, text_report

FIXTURES = Path(__file__).parent/'fixtures/evaluation'


def item(name='Синтетический суп', quantity='1', total='100.00', **extra):
    return {'name': name, 'quantity': quantity, 'total_price': total, **extra}


def receipt(*items, **totals):
    return {'items': list(items), **totals}


class EvaluationTests(unittest.TestCase):
    def test_synthetic_cases(self):
        counts = {'short': (2,0,0,0), 'long': (29,1,2,1), 'duplicates': (3,1,0,0),
                  'wrong_quantity': (0,1,0,0), 'wrong_price': (0,1,0,0),
                  'zero_positions': (2,0,0,1), 'prepayment': (1,0,0,0), 'reordered': (3,0,0,0)}
        for name, wanted in counts.items():
            with self.subTest(case=name):
                report = evaluate(load_json(FIXTURES/(name+'_expected.json'),'expected'),
                                  load_json(FIXTURES/(name+'_actual.json'),'actual'))
                s = report['summary']
                self.assertEqual(tuple(s[k] for k in ('full_matches','partial_matches','missing_count','extra_count')),wanted)
                pairs = report['matches']
                self.assertEqual(len({p['expected_index'] for p in pairs}),len(pairs))
                self.assertEqual(len({p['actual_index'] for p in pairs}),len(pairs))

    def test_exact_decimal_values_and_normalization_never_need_float(self):
        report = evaluate(receipt(item('Чёрный чай','0.250','0.10'),item('Сок','1','0.20')),
                          receipt(item('  ЧЕРНЫЙ   чай ','0.25','0.100'),item('сок',Decimal('1.0'),Decimal('.2'))))
        self.assertEqual(report['summary']['full_matches'],2)
        self.assertEqual(Decimal(report['item_totals']['actual']),Decimal('.30'))
        self.assertEqual(Decimal(report['item_totals']['actual_minus_expected']),0)

    def test_full_name_equality_is_not_fuzzy_partial_name_errors_are_visible(self):
        report = evaluate(receipt(item('Черный чай')),receipt(item('Черный чаи')))
        self.assertEqual(report['summary']['full_matches'],0)
        self.assertEqual(report['summary']['partial_matches'],1)
        self.assertEqual(report['matches'][0]['fields'],{'name':'mismatch','quantity':'match','total_price':'match'})

    def test_same_name_with_wrong_quantity_or_price_is_not_full(self):
        for actual,field in ((item(quantity='2'),'quantity'),(item(total='99.00'),'total_price')):
            with self.subTest(field=field):
                report=evaluate(receipt(item()),receipt(actual))
                self.assertEqual(report['summary']['full_matches'],0)
                self.assertEqual(report['matches'][0]['fields'][field],'mismatch')

    def test_duplicate_full_matches_are_reserved_before_partial_pairs(self):
        expected=receipt(item(),item(quantity='2'),item())
        actual=receipt(item(quantity='3'),item(),item(quantity='2'))
        report=evaluate(expected,actual)
        self.assertEqual(report['summary']['full_matches'],2)
        self.assertEqual(report['summary']['partial_matches'],1)
        self.assertEqual([r['actual_index'] for r in report['matches']],[2,3,1])

    def test_duplicate_rows_cannot_be_reused_and_order_is_irrelevant(self):
        expected=receipt(item(),item(),item('Синтетический чай',total='50'))
        for order in permutations(expected['items']):
            self.assertEqual(evaluate(expected,receipt(*order))['summary']['full_matches'],3)
        missing=evaluate(expected,receipt(item()))
        self.assertEqual(missing['missing_expected_indices'],[2,3])
        extra=evaluate(receipt(item()),receipt(item(),item()))
        self.assertEqual(extra['extra_actual_indices'],[2])

    def test_partial_assignment_is_global_rather_than_greedy(self):
        expected=receipt(item(quantity='1',total='10'),item(quantity='2',total='20'))
        actual=receipt(item(quantity='1',total='20'),item(quantity='3',total='10'))
        report=evaluate(expected,actual)
        self.assertEqual([r['actual_index'] for r in report['matches']],[2,1])
        self.assertEqual(report['summary']['matched_fields'],{'name':2,'quantity':0,'total_price':2})
        self.assertEqual(report,evaluate(expected,actual))

    def test_unique_numeric_signature_is_only_a_partial_hypothesis(self):
        report=evaluate(receipt(item('Пицца')),receipt(item('Компот')))
        row=report['matches'][0]
        self.assertEqual(row['classification'],'partial')
        self.assertEqual(row['pairing_basis'],'unique_quantity_and_price')
        self.assertEqual(row['fields']['name'],'mismatch')

    def test_ambiguous_numeric_coincidences_do_not_establish_identity(self):
        report=evaluate(receipt(item('Чай'),item('Рыба')),receipt(item('Лимонад'),item('Пирог')))
        self.assertEqual(report['summary']['partial_matches'],0)
        self.assertEqual(report['missing_expected_indices'],[1,2])
        self.assertEqual(report['extra_actual_indices'],[1,2])

    def test_zero_fields_and_empty_receipts_are_preserved(self):
        report=evaluate(receipt(item(quantity='0',total='0')),receipt(item(quantity='0',total='0.00')))
        self.assertEqual(report['summary']['full_matches'],1)
        self.assertEqual(Decimal(report['item_totals']['actual']),0)
        both=evaluate(receipt(),receipt())
        self.assertEqual(both['summary']['expected_count'],0)
        self.assertFalse(both['has_differences'])
        self.assertEqual(evaluate(receipt(item()),receipt())['summary']['missing_count'],1)

    def test_missing_invalid_fields_remain_in_counts_and_sums_are_not_fabricated(self):
        for value in (None,'NaN','PRIVATE_INVALID',True,0.5):
            with self.subTest(value=value):
                actual=item(quantity=None,total=value,unit_price='100.00')
                report=evaluate(receipt(item()),receipt(actual))
                self.assertEqual(report['summary']['actual_count'],1)
                self.assertEqual(report['summary']['full_matches'],0)
                self.assertEqual(report['summary']['partial_matches'],1)
                self.assertIsNone(report['item_totals']['actual'])
                self.assertIsNone(report['item_totals']['actual_minus_expected'])
                self.assertEqual(Decimal(report['item_totals']['actual_known']),0)
                self.assertEqual(report['item_totals']['actual_unknown_prices'],1)
                self.assertNotIn('PRIVATE_INVALID',json.dumps(report))

    def test_non_cent_values_are_compared_without_rounding_or_repair(self):
        report=evaluate(receipt(item(total='0.00')),receipt(item(total='0.005')))
        self.assertEqual(report['summary']['full_matches'],0)
        self.assertEqual(report['item_totals']['actual_minus_expected'],'0.005')

    def test_line_sums_and_gross_remain_separate_even_when_gross_is_correct(self):
        report=evaluate(receipt(item(total='100'),gross_total='100'),
                        receipt(item(total='99.995'),gross_total='100'))
        self.assertEqual(report['totals']['gross_total']['status'],'match')
        self.assertEqual(report['item_totals']['actual_minus_expected'],'-0.005')
        self.assertEqual(report['item_totals']['actual_gross_minus_items'],'0.005')

    def test_prepayment_due_and_absent_optional_totals_are_not_derived(self):
        expected=receipt(item(),gross_total='100',prepayment='20',amount_due='80')
        actual=receipt(item(),gross_total='100',prepayment='25',amount_due='75')
        totals=evaluate(expected,actual)['totals']
        self.assertEqual(totals['gross_total']['status'],'match')
        self.assertEqual(totals['prepayment']['actual_minus_expected'],'5')
        self.assertEqual(totals['amount_due']['actual_minus_expected'],'-5')
        incomplete=evaluate(expected,receipt(item(),gross_total='100',amount_due='80'))
        self.assertEqual(incomplete['totals']['prepayment']['status'],'missing')
        self.assertIsNone(incomplete['totals']['prepayment']['actual'])
        absent=evaluate(receipt(item()),receipt(item()))
        self.assertTrue(all(row['status']=='not_provided' for row in absent['totals'].values()))

    def test_existing_expected_fixture_aliases_and_distinct_receipt_total(self):
        expected={'expected_items':[item()], 'expected_gross_total':'100','expected_total':'90',
                  'expected_prepayment':'10','expected_amount_due':'90'}
        actual=receipt(item(),gross_total='100',receipt_total='100',prepayment='10',amount_due='90')
        report=evaluate(expected,actual)
        self.assertEqual(report['totals']['gross_total']['status'],'match')
        self.assertEqual(report['totals']['receipt_total']['status'],'mismatch')
        with self.assertRaises(InputError):evaluate({**expected,'items':[item()]},actual)
        with self.assertRaises(InputError):evaluate({**expected,'gross_total':'100'},actual)

    def test_reports_never_echo_source_text_or_private_metadata(self):
        private='PRIVATE_NAME_PHONE_ADDRESS_IMAGE_TOKEN'
        expected=receipt(item(private,guest=private),guest_subtotals={private:'100'},phone=private,raw_text=private)
        actual=receipt(item(private+' changed',guest_id=private),address=private,image=private,currency_evidence=private)
        before=deepcopy((expected,actual))
        report=evaluate(expected,actual)
        self.assertNotIn(private,json.dumps(report))
        self.assertNotIn(private,text_report(report))
        self.assertEqual((expected,actual),before)
        self.assertEqual(report['summary']['partial_matches'],1)

    def test_bounded_input_and_safe_validation_errors(self):
        for expected in ({},[],receipt({'name':'PRIVATE','quantity':'NaN','total_price':'1'}),
                         receipt(*([item()]*201)),receipt(item(),prepayment='PRIVATE')):
            with self.subTest(expected_type=type(expected)), self.assertRaises(InputError) as caught:
                evaluate(expected,receipt())
            self.assertNotIn('PRIVATE',str(caught.exception))
        with self.assertRaises(InputError):evaluate(receipt(),{'items':['PRIVATE']})
        for value in (Decimal('1e9999999'),Decimal('1e-9999999'),Decimal('Infinity'),10**10000):
            self.assertIsNone(decimal_value(value))

    def test_large_values_are_exact_even_under_low_decimal_context(self):
        expected=receipt(*[item(total='99999999999999999999999999999.999999999999')]*200)
        with localcontext() as context:
            context.prec=2
            report=evaluate(expected,deepcopy(expected))
        self.assertEqual(report['item_totals']['actual_minus_expected'],'0.000000000000')
        self.assertEqual(report['summary']['full_matches'],200)

    def test_json_numbers_are_decimals_and_invalid_files_are_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'PRIVATE_INPUT.json'
            path.write_text('{"items":[{"name":"Тест","quantity":0.25,"total_price":0.1}]}')
            data=load_json(path,'actual')
            self.assertIsInstance(data['items'][0]['total_price'],Decimal)
            self.assertEqual(data['items'][0]['total_price'],Decimal('.1'))
            for invalid in ('{"items":[],"items":[]}','{"items":NaN}','PRIVATE_BROKEN_JSON'):
                path.write_text(invalid)
                with self.assertRaises(InputError) as caught:load_json(path,'actual')
                self.assertNotIn('PRIVATE',str(caught.exception))

    def test_assignment_algorithm_matches_brute_force_on_small_graphs(self):
        # Exhaustive oracle, independent of receipt matching and the Hungarian code.
        for values in product(range(3),repeat=4):
            weights={(r,c):values[r*2+c] for r in range(2) for c in range(2) if values[r*2+c]}
            optimum=0
            for picks in product((-1,0,1),repeat=2):
                used=[c for c in picks if c>=0]
                if len(set(used))!=len(used):continue
                optimum=max(optimum,sum(weights.get((r,c),0) for r,c in enumerate(picks)))
            pairs=best_pairs(2,2,weights)
            self.assertEqual(sum(weights[pair] for pair in pairs),optimum)
