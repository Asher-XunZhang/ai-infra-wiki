"""Compare the browser solver with the reviewed Python dependency model."""
import copy
import json
import random
import subprocess
import unittest
from pathlib import Path

from build_pp_scenarios import SCENARIOS
from build_pp_quick_data import derive_quick
from pp_timing_model import build_model, timing_parameters
from test_pp_timing_model import check_source_contracts

ROOT = Path(__file__).resolve().parents[1]
NODE = r"""
const fs = require('fs'), vm = require('vm');
const page = './pages/sglang/pd-prefill-pp-loop/';
const context = {window:{}};
vm.runInNewContext(fs.readFileSync(page+'timing-template.js','utf8'), context);
const engine = require(page+'timing-engine.js');
const cases = JSON.parse(fs.readFileSync(0,'utf8'));
process.stdout.write(JSON.stringify(cases.map(c => {
  try {return {packet:engine.buildPacket(context.window.PP_TIMING_TEMPLATE,c.timing,c.horizon)};}
  catch(e) {return {error:e.message};}
})));
"""


def browser(cases):
    result = subprocess.run(['node', '-e', NODE], cwd=ROOT, input=json.dumps(cases),
                            capture_output=True, encoding='utf-8', check=True, timeout=120)
    return json.loads(result.stdout)


class CustomTiming(unittest.TestCase):
    def compare(self, actual, expected, path='root'):
        if isinstance(expected, dict):
            # Python quick links use integer batch IDs; JSON uses string keys.
            expected = {str(k): v for k, v in expected.items()}
            self.assertEqual(set(actual), set(expected), path)
            for key in expected:
                self.compare(actual[key], expected[key], path+'.'+key)
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected), path)
            for i, (a, e) in enumerate(zip(actual, expected)):
                self.compare(a, e, path+f'[{i}]')
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            # Independent languages may round an exact decimal half differently.
            self.assertAlmostEqual(actual, expected, delta=.00101, msg=path)
        else:
            self.assertEqual(actual, expected, path)

    def check_packet(self, packet, expected):
        check_source_contracts(packet['model'])
        self.compare(packet['model'], expected)
        quick = derive_quick(expected, 'test')
        del quick['sourceModelSha256']
        self.compare(packet['quick'], quick)

    def test_ten_presets_match_python(self):
        cases = [dict(timing=timing_parameters(s['config']), horizon=s['config'].get('horizon',22)) for s in SCENARIOS]
        for scene, result in zip(SCENARIOS, browser(cases)):
            with self.subTest(scene=scene['id']):
                self.assertNotIn('error', result)
                self.check_packet(result['packet'], build_model(scene['config']))

    def test_custom_matrices_phases_and_zero_durations(self):
        mixed = timing_parameters()
        mixed['gpu'][1][2] = 20
        mixed['h2d'][1][0] = 30
        mixed['kv'][2][4] = 40
        mixed['cpu'].update(A=.5, C=2, F=3, H=1.4, I=.7)
        mixed.update(proxy=2.1, output=1.3, control=.7, d2h=1.2)
        zero = timing_parameters()
        for kind in ['gpu','h2d','kv']:
            zero[kind] = [[0 for _ in row] for row in zero[kind]]
        zero.update(proxy=0, output=0, control=0, d2h=0)
        cases = [dict(timing=t, horizon=64) for t in [mixed, zero]]
        for case, result in zip(cases, browser(cases)):
            self.assertNotIn('error', result)
            self.check_packet(result['packet'], build_model(case))

    def test_adaptive_horizon_and_invalid_inputs(self):
        slow = timing_parameters({'kv_scale':4})
        invalid = copy.deepcopy(slow); invalid['gpu'][0][0] = -1
        invalid_cpu = copy.deepcopy(slow); invalid_cpu['cpu']['A'] = 0
        results = browser([dict(timing=slow), dict(timing=invalid), dict(timing=invalid_cpu), dict(timing=slow,horizon=513)])
        self.check_packet(results[0]['packet'], build_model(dict(timing=slow,horizon=64)))
        for result in results[1:]:
            self.assertIn('error', result)

    def test_varied_custom_parameters(self):
        rng = random.Random(279339)
        cases = []
        for _ in range(4):
            timing = timing_parameters()
            for kind in ['gpu','h2d','kv']:
                timing[kind] = [[round(rng.uniform(.2,40),3) for _ in row] for row in timing[kind]]
            timing['cpu'] = {p:round(rng.uniform(.2,3),3) for p in 'ABCDEFGHI'}
            timing.update({p:round(rng.uniform(0,3),3) for p in ['proxy','output','control','d2h']})
            cases.append(dict(timing=timing,horizon=128))
        for case, result in zip(cases,browser(cases)):
            self.assertNotIn('error',result)
            self.check_packet(result['packet'],build_model(case))

    def test_too_many_retirement_rounds_return_error(self):
        timing = timing_parameters({'cpu_scale':.1})
        timing['kv'] = [[100]*5 for _ in range(3)]
        timing['control'] = 0
        result = browser([dict(timing=timing)])[0]
        self.assertIn('512',result['error'])


if __name__ == '__main__':
    unittest.main()
