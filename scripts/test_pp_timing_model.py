"""Source-derived regression checks; no SGLang/GPU runtime is imported.

Contracts: scheduler_pp_mixin.py:991,1062,335; cache_controller.py:923;
l2_transfer.py:86; unified_radix_cache.py:2957 at 72d5c5bb73.
These test necessary ordering, not hardware timing or full stream simulation.
"""
import copy
import unittest

from pp_timing_model import build_model
from build_pp_scenarios import SCENARIOS
from build_pp_quick_data import derive_quick


def check_source_contracts(data):
    graph = data['graph']

    def requires(target, producer):
        # Require an actual causal edge/path, not an accidental time ordering.
        pending = list(graph[target]['deps'])
        seen = set()
        while pending:
            key = pending.pop()
            if key in seen:
                continue
            seen.add(key)
            pending.extend(graph[key]['deps'])
        assert producer in seen, (target, 'missing source dependency', producer)
        assert graph[target]['start'] + .002 >= graph[producer]['end'], (target, producer)

    for loop in data['loops']:
        r, n = loop['r'], loop['n']
        prefix = f'{r}:{n}:'
        # The normal UnifiedRadixCache path combines write/load counts once.
        assert prefix + 'l2_counts' in graph
        assert prefix + 'l2_write' not in graph and prefix + 'l2_load' not in graph
        if loop['current'] in (2, 4):
            requires(prefix + 'h2d', f'{r}:{n-1}:gpu')
        if r == 2 and loop['current'] and loop['old']:
            requires(prefix + 'copy', prefix + 'gpu')
            requires(prefix + 'result', prefix + 'copy')
        if loop['current'] and r:
            requires(prefix + 'gpu', f'{r-1}:{n}:proxy_message')
        if loop['old']:
            requires(prefix + 'send_kv', f'{r}:{n-2}:gpu')

    for row in data['releases']:
        cleanup = graph[f"{row['r']}:{row['n']}:release"]
        for batch in row['batches']:
            kv = graph[f"{row['r']}:{batch+4}:kv"]
            assert cleanup['start'] + .002 >= kv['end']


class TimingSourceContracts(unittest.TestCase):
    def test_all_scenarios(self):
        for scene in SCENARIOS:
            with self.subTest(scene=scene['id']):
                data = build_model(scene['config'])
                check_source_contracts(data)
                quick = derive_quick(data, 'source-contract-check')
                self.assertEqual(len(quick['batches']), 5)

    def test_missing_d2h_stream_edge_is_rejected(self):
        data = copy.deepcopy(build_model())
        data['graph']['2:5:copy']['deps'].remove('2:5:gpu')
        with self.assertRaisesRegex(AssertionError, 'missing source dependency'):
            check_source_contracts(data)

    def test_missing_h2d_start_event_is_rejected(self):
        data = copy.deepcopy(build_model())
        data['graph']['0:4:h2d']['deps'].remove('0:3:gpu')
        with self.assertRaisesRegex(AssertionError, 'missing source dependency'):
            check_source_contracts(data)


if __name__ == '__main__':
    unittest.main()
