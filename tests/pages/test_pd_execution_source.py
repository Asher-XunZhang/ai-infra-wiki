"""Pinned source ordering contracts for the two PP lessons; no GPU execution."""
import ast
import os
import subprocess
import unittest
from functools import lru_cache

SOURCE = os.environ.get('SGLANG_SOURCE_DIR')
if not SOURCE:
    raise SystemExit('SGLANG_SOURCE_DIR required; source checks not executed')
REVISIONS = ['279339f113b79af84f27fd3ac92d0a13bd3f4cbd', '882577451e764a515df2a386a055012e8f075a16']

@lru_cache(None)
def source(rev, path):
    return subprocess.check_output(['git', '-C', SOURCE, 'show', f'{rev}:python/sglang/srt/{path}'], text=True)

def function(rev, path, name):
    tree = ast.parse(source(rev, path))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)

def calls(node, name):
    return sorted(n.lineno for n in ast.walk(node) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == name)

class PPSource(unittest.TestCase):
    def test_local_chunk_send_before_selection_and_proxy_after_launch(self):
        for rev in REVISIONS:
            with self.subTest(revision=rev):
                n = function(rev, 'managers/scheduler_pp_mixin.py', 'event_loop_pp_disagg_prefill')
                self.assertLess(calls(n, 'process_prefill_chunk')[0], calls(n, 'get_new_batch_prefill')[0])
                self.assertLess(calls(n, '_pp_recv_proxy_tensors')[0], calls(n, '_pp_launch_batch')[0])
                self.assertLess(calls(n, '_pp_launch_batch')[0], calls(n, 'wait_event')[-1])
                self.assertLess(calls(n, 'synchronize')[0], calls(n, '_pp_process_batch_result')[0])

    def test_submit_is_queued_and_worker_drains_before_success_observation(self):
        for rev in REVISIONS:
            with self.subTest(revision=rev):
                path = 'disaggregation/mooncake/conn.py'
                n = function(rev, path, 'add_transfer_request')
                self.assertTrue(calls(n, 'put'))
                self.assertFalse(calls(n, 'batch_transfer_sync'))
                self.assertIn('session_port_sum % len(self.transfer_queues)', ast.unparse(n))
                worker = function(rev, path, 'transfer_worker')
                self.assertTrue(calls(worker, 'get'))
                self.assertLess(calls(worker, 'send_aux')[0], calls(worker, 'conclude_transfer')[0])
                self.assertIn('_staging_outstanding[kv_chunk.room] -= 1', ast.unparse(worker))

    def test_release_is_terminal_intersection_followed_by_local_recheck(self):
        for rev in REVISIONS:
            with self.subTest(revision=rev):
                n = function(rev, 'managers/scheduler_pp_mixin.py', '_pp_pd_get_prefill_transferred_ids')
                text = ast.unparse(n)
                self.assertIn('KVPoll.Success', text)
                self.assertIn('KVPoll.Failed', text)
                self.assertTrue(any(isinstance(x, ast.BinOp) and isinstance(x.op, ast.BitAnd) for x in ast.walk(n)))
                n = function(rev, 'disaggregation/prefill.py', 'process_disagg_prefill_inflight_queue')
                text = ast.unparse(n)
                self.assertLess(text.index('rids_to_check'), text.index('KVPoll.Success'))
                self.assertIn('undone_reqs', text)
                self.assertTrue(calls(n, 'clear'))

if __name__ == '__main__':
    unittest.main()
