"""Source contracts behind request-state teaching traces; no engine/GPU execution."""
import ast
import os
import subprocess
import unittest
from functools import lru_cache

REVISION = '279339f113b79af84f27fd3ac92d0a13bd3f4cbd'
ROOT = os.environ.get('SGLANG_SOURCE_DIR')
if not ROOT:
    raise SystemExit('SGLANG_SOURCE_DIR required; source audit not executed')


@lru_cache(None)
def function(path, name):
    source = subprocess.check_output(['git', '-C', ROOT, 'show', f'{REVISION}:python/sglang/srt/{path}'], text=True)
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)


def call_lines(node, name):
    return sorted(n.lineno for n in ast.walk(node) if isinstance(n, ast.Call) and
                  ((isinstance(n.func, ast.Attribute) and n.func.attr == name) or
                   (isinstance(n.func, ast.Name) and n.func.id == name)))


class RequestContracts(unittest.TestCase):
    def test_snapshot_keeps_req_identity_but_copies_membership(self):
        code = ast.unparse(function('managers/schedule_batch.py', 'copy'))
        self.assertIn('reqs=self.reqs[:]', code)
        self.assertNotIn('deepcopy', code)
        plan = ast.unparse(function('managers/scheduler.py', 'get_next_batch_to_run'))
        self.assertIn('running_batch = last_batch', plan)
        self.assertIn('running_batch.merge_batch(last_batch)', plan)

    def test_queue_pop_precedes_wait_inside_processor(self):
        loop = function('managers/scheduler.py', 'event_loop_overlap')
        pop = next(n for n in ast.walk(loop) if isinstance(n, ast.FunctionDef) and n.name == 'pop_and_process')
        self.assertLess(call_lines(pop, 'popleft')[0], call_lines(pop, 'process_batch_result')[0])
        processor = function('managers/scheduler_components/batch_result_processor.py', 'process_batch_result_decode')
        self.assertLess(call_lines(processor, 'synchronize')[0], call_lines(processor, 'update_finish_state')[0])
        self.assertIn('req.finished() or req.is_retracted', ast.unparse(processor))

    def test_chunk_continuation_is_not_waiting_reenqueue(self):
        node = function('managers/schedule_policy.py', 'add_chunked_req')
        code = ast.unparse(node)
        self.assertIn('return req if truncated else None', code)
        self.assertNotIn('waiting_queue.append', code)
        planner = function('managers/scheduler.py', 'get_next_batch_to_run')
        self.assertLess(call_lines(planner, 'stash_chunked_request')[-1], call_lines(planner, 'filter_batch')[0])

    def test_retract_releases_before_reset_and_requeues(self):
        release = function('managers/schedule_batch.py', 'release_req')
        self.assertLess(call_lines(release, 'release_kv_cache')[0], call_lines(release, 'reset_for_retract')[0])
        self.assertIn('is_insert=False', ast.unparse(release))
        reset = function('managers/schedule_batch.py', 'reset_for_retract')
        self.assertIn('self.is_retracted = True', ast.unparse(reset))
        clear = [n for n in ast.walk(reset) if isinstance(n, ast.Assign) and any(ast.unparse(t) == 'self.output_ids' for t in n.targets)]
        self.assertTrue(clear)
        self.assertTrue(all(any(isinstance(n, ast.If) and 'self.input_embeds is not None' == ast.unparse(n.test) and c in ast.walk(n) for n in ast.walk(reset)) for c in clear))
        self.assertIn('_add_request_to_queue(req, is_retracted=True)', ast.unparse(function('managers/scheduler.py', 'update_running_batch')))

    def test_running_abort_uses_pending_finish_and_filter_does_not_free_kv(self):
        abort = ast.unparse(function('managers/scheduler.py', 'abort_request'))
        self.assertIn('self.waiting_queue.pop(i)', abort)
        self.assertIn('req.to_finish = FINISH_ABORT()', abort)
        self.assertNotIn('req.finished_reason =', abort)
        self.assertIn('self.finished_reason = self.to_finish', ast.unparse(function('managers/schedule_batch.py', 'update_finish_state')))
        filt = ast.unparse(function('managers/schedule_batch.py', 'filter_batch'))
        self.assertIn('not self.reqs[i].finished()', filt)
        self.assertNotIn('release_kv_cache', filt)


if __name__ == '__main__':
    unittest.main()
