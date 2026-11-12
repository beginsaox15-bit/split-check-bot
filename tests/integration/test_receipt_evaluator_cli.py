"""End-to-end offline CLI tests with the standard library only (-S)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT/'tests/fixtures/evaluation'


class ReceiptEvaluatorCLITests(unittest.TestCase):
    def run_cli(self, expected, actual, *options):
        environment = {**os.environ, 'BOT_TOKEN':'PRIVATE_FAKE_TOKEN', 'OPENROUTER_API_KEY':'PRIVATE_FAKE_KEY'}
        environment.pop('PYTHONPATH',None)
        return subprocess.run([sys.executable,'-S','-m','tools.evaluate_receipt',
                               '--expected',str(expected),'--actual',str(actual),*options],
                              cwd=ROOT,env=environment,capture_output=True,text=True,timeout=20)

    def test_short_receipt_json_without_site_packages_or_bot_dependencies(self):
        result = self.run_cli(FIXTURES/'short_expected.json',FIXTURES/'short_actual.json','--format','json')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stderr,'')
        report=json.loads(result.stdout)
        self.assertEqual(report['summary']['full_matches'],2)
        self.assertFalse(report['has_differences'])
        self.assertNotIn('Синтетический',result.stdout)

    def test_text_report_names_error_fields_and_strict_exit_code(self):
        result=self.run_cli(FIXTURES/'wrong_quantity_expected.json',FIXTURES/'wrong_quantity_actual.json','--fail-on-difference')
        self.assertEqual(result.returncode,1)
        self.assertIn('Полностью совпало: 0/1',result.stdout)
        self.assertIn('Частично сопоставлено: 1',result.stdout)
        self.assertIn('quantity=mismatch',result.stdout)
        self.assertIn('Эталон #1 → OCR #1',result.stdout)
        self.assertNotIn('Синтетическая пицца',result.stdout)

    def test_default_reports_quality_errors_without_failing_execution(self):
        result=self.run_cli(FIXTURES/'long_expected.json',FIXTURES/'long_actual.json','--format','json')
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads(result.stdout)
        self.assertEqual(report['summary']['full_matches'],29)
        self.assertEqual(report['summary']['partial_matches'],1)
        self.assertEqual(report['missing_expected_indices'],[6,27])
        self.assertEqual(len(report['extra_actual_indices']),1)

    def test_cli_never_echoes_private_fields_paths_or_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            expected=root/'PRIVATE_PERSON_expected.json'
            actual=root/'PRIVATE_ADDRESS_actual.json'
            private='PRIVATE_NAME_PHONE_ADDRESS_IMAGE'
            data={'items':[{'name':private,'quantity':'1','total_price':'100'}],
                  'guest':private,'phone':private,'address':private,'image':'data:image/jpeg;base64,PRIVATE_IMAGE'}
            expected.write_text(json.dumps(data))
            data['items'][0]['quantity']='PRIVATE_BAD_QUANTITY'
            actual.write_text(json.dumps(data))
            before=(expected.read_bytes(),actual.read_bytes())
            result=self.run_cli(expected,actual,'--format','json')
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertNotIn('PRIVATE_',result.stdout+result.stderr)
            self.assertEqual((expected.read_bytes(),actual.read_bytes()),before)
            actual.write_text('PRIVATE_BROKEN_JSON')
            result=self.run_cli(expected,actual)
            self.assertEqual(result.returncode,2)
            self.assertEqual(result.stdout,'')
            self.assertNotIn('PRIVATE_',result.stderr)
            result=self.run_cli(root/'PRIVATE_MISSING.json',actual)
            self.assertEqual(result.returncode,2)
            self.assertNotIn('PRIVATE_',result.stderr)

    def test_cli_rejects_ambiguous_json_and_nonfinite_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            actual=Path(directory)/'actual.json'
            for content in ('{"items":[],"items":[]}','{"items":NaN}','[]'):
                actual.write_text(content)
                result=self.run_cli(FIXTURES/'short_expected.json',actual)
                self.assertEqual(result.returncode,2)
                self.assertNotIn('Traceback',result.stderr)

    def test_cli_does_not_import_network_or_bot_code(self):
        script = '''
import sys, runpy
class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'bot','socket','requests','aiohttp','pytesseract','rapidocr'}:
            raise RuntimeError('Forbidden dependency imported')
sys.meta_path.insert(0,Guard())
sys.argv=['evaluate_receipt','--expected','tests/fixtures/evaluation/short_expected.json',
          '--actual','tests/fixtures/evaluation/short_actual.json','--format','json']
runpy.run_module('tools.evaluate_receipt',run_name='__main__')
'''
        result=subprocess.run([sys.executable,'-S','-c',script],cwd=ROOT,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['summary']['full_matches'],2)
