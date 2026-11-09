"""Execute selected fixed-source gates with CPU doubles; no GPU/runtime claims."""
import __future__
import ast
import copy
import functools
import os
import subprocess
import unittest
from enum import IntEnum
from types import SimpleNamespace as NS

REVISION = '279339f113b79af84f27fd3ac92d0a13bd3f4cbd'
SOURCE = os.environ.get('SGLANG_SOURCE_DIR')
if not SOURCE:
    raise SystemExit('SGLANG_SOURCE_DIR is required; fixed-source checks not executed')

@functools.lru_cache(None)
def tree(path):
    return ast.parse(subprocess.check_output(['git', '-C', SOURCE, 'show',
        f'{REVISION}:python/sglang/srt/{path}'], text=True))

def function(path, name):
    return next(n for n in ast.walk(tree(path)) if isinstance(n, ast.FunctionDef) and n.name == name)

def load(path, name, env):
    node = copy.deepcopy(function(path, name)); node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec',
        flags=__future__.annotations.compiler_flag), env)
    return env[name]

def calls(node, attr):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and
        ((isinstance(n.func, ast.Attribute) and n.func.attr == attr) or
         (isinstance(n.func, ast.Name) and n.func.id == attr))]

class Poll(IntEnum):
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4

class HandoffSource(unittest.TestCase):
    def test_zero_optimistic_attempts_keeps_bootstrapping_request(self):
        req = NS(prefill_attempt_count=0, is_retracted=False,
                 time_stats=NS(set_wait_queue_entry_time=lambda: None))
        queue = NS(queue=[req], pp_size=1, ensure_metadata_buffer=lambda req: True,
                   scheduler=NS(attn_cp_cpu_group=None, attn_tp_cpu_group=None,
                                processed_tokens_counter=0))
        req.disagg_kv_sender = NS()
        args = NS(optimistic_prefill_attempts=0)
        pop = load('disaggregation/prefill.py', 'pop_bootstrapped', {
            'KVPoll': Poll, 'get_disagg': lambda: args,
            'poll_and_all_reduce_attn_cp_tp_group': lambda *a: [Poll.Bootstrapping]})
        self.assertEqual(pop(queue), [])
        self.assertEqual(queue.queue, [req])
        args.optimistic_prefill_attempts = 1
        self.assertEqual(pop(queue), [req])
        self.assertEqual(queue.queue, [])

    def test_preallocation_precedes_address_publication(self):
        node = function('disaggregation/decode.py', 'pop_preallocated')
        self.assertLess(calls(node, '_pre_alloc')[0].lineno, calls(node, 'send_metadata')[0].lineno)

    def test_nonoverlap_chunk_is_enqueued_before_next_selection(self):
        node = function('disaggregation/prefill.py', 'get_next_disagg_prefill_batch_to_run')
        self.assertLess(calls(node, 'process_prefill_chunk')[0].lineno, calls(node, 'get_new_batch_prefill')[0].lineno)
        node = function('disaggregation/prefill.py', 'process_prefill_chunk')
        self.assertTrue(calls(node, 'send_kv_chunk'))

    def test_sender_enqueues_and_worker_waits_for_transfer(self):
        node = function('disaggregation/mooncake/conn.py', 'add_transfer_request')
        self.assertEqual(len(calls(node, 'put')), 1)
        self.assertFalse(calls(node, 'batch_transfer_sync'))
        node = function('disaggregation/mooncake/conn.py', '_transfer_data')
        self.assertTrue(calls(node, 'batch_transfer_sync'))
        node = function('disaggregation/mooncake/conn.py', 'transfer_worker')
        self.assertLess(calls(node, 'send_aux')[0].lineno, calls(node, 'conclude_transfer')[0].lineno)

    def test_metadata_gate_holds_zero_room(self):
        class Rooms:
            def __init__(self, values): self.values = values
            def __getitem__(self, key): return NS(item=lambda: self.values[key[0]])
        polls = [4, 4, 4, 3]
        reqs = [NS(req=NS(), metadata_buffer_index=i) for i in range(4)]
        gate = load('disaggregation/utils.py', '_apply_metadata_gate',
                    {'KVPoll': Poll, '_is_fake_transfer': lambda req: False})
        gate(polls, reqs, NS(bootstrap_room=Rooms([0, 42, 99, 0])))
        self.assertEqual(polls, [3, 4, 4, 3])  # mismatch is checked by commit, not this visibility gate

    def test_success_notification_not_undoing_failure(self):
        node = function('disaggregation/common/conn.py', 'conclude_transfer')
        self.assertLess(calls(node, 'update_status')[0].lineno, calls(node, 'send_kv_status_message')[0].lineno)

    def test_prebuilt_has_no_prompt_model_forward(self):
        node = function('disaggregation/decode_schedule_batch_mixin.py', 'prepare_for_prebuilt')
        self.assertFalse(calls(node, 'forward'))
        node = function('disaggregation/decode_schedule_batch_mixin.py', 'process_prebuilt')
        self.assertTrue(calls(node, 'stash'))
        node = function('disaggregation/decode.py', 'get_next_disagg_decode_batch_to_run')
        self.assertLess(calls(node, 'process_batch_result_prebuilt')[0].lineno, calls(node, 'update_running_batch')[0].lineno)

    def test_sender_success_waits_for_outstanding_chunks(self):
        mgr = NS(check_status=lambda room: Poll.Success, _staging_outstanding={42: 1})
        sender = NS(conclude_state=None, kv_mgr=mgr, bootstrap_room=42,
                    trace_ctx=NS(trace_req_finish=lambda: None))
        cls = next(n for n in tree('disaggregation/mooncake/conn.py').body if isinstance(n, ast.ClassDef) and n.name == 'MooncakeKVSender')
        node = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'poll'))
        env = {'KVPoll': Poll}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'poll', 'exec', flags=__future__.annotations.compiler_flag), env)
        self.assertEqual(env['poll'](sender), Poll.Transferring)
        mgr._staging_outstanding[42] = 0
        self.assertEqual(env['poll'](sender), Poll.Success)

if __name__ == '__main__':
    unittest.main()
