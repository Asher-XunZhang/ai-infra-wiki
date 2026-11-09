"""Execute pinned scheduler loop bodies with recording CPU stubs; no GPU claims."""
import ast
import collections
import json
import subprocess
import unittest
from types import SimpleNamespace as NS
from test_parallelism_source import load, source, REVISION, SOURCE


class EndWindow(Exception):
    pass


class LoopSource(unittest.TestCase):
    def trace(self, overlap):
        records = []
        batches = [NS(name=f'B{i}', copy=lambda i=i: NS(name=f'B{i}')) for i in range(3 if overlap else 2)]
        count = 0
        sched = NS(gracefully_exit=False, _engine_paused=False, running_batch=None, last_batch=None,
                   is_generation=True, enable_unified_memory=False)
        def ingest():
            nonlocal count
            if count == len(batches) + 2:
                raise EndWindow()
            records.append(('ingest', count));count += 1
        def plan(**kwargs):
            batch = batches[count-1] if count <= len(batches) else None
            records.append(('plan', batch.name if batch else None))
            return NS(batch_to_run=batch, running_batch=batch)
        sched.ingest_requests = ingest
        sched.get_next_batch_to_run = plan
        sched.run_batch = lambda b: records.append(('launch',b.name)) or b.name+'-result'
        sched.process_batch_result = lambda b,r: records.append(('process',b.name))
        sched._apply_war_barrier = lambda: records.append(('WAR',None))
        sched.is_disable_overlap_for_batch = lambda *a,**k: False
        sched.launch_batch_sample_if_needed = lambda *a: None
        sched.on_idle = lambda: records.append(('idle',None))
        env={'deque':collections.deque,'envs':NS(SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=NS(get=lambda:False))}
        with self.assertRaises(EndWindow):
            load('managers/scheduler.py','event_loop_overlap' if overlap else 'event_loop_normal',env)(sched)
        return records

    def test_normal_consumes_before_next_plan(self):
        t=self.trace(False)
        self.assertLess(t.index(('process','B0')),t.index(('plan','B1')))
        self.assertLess(t.index(('process','B1')),t.index(('idle',None)))

    def test_overlap_submits_before_consuming_and_drains_without_new_batch(self):
        t=self.trace(True)
        self.assertLess(t.index(('launch','B1')),t.index(('process','B0')))
        self.assertLess(t.index(('launch','B2')),t.index(('process','B1')))
        self.assertLess(t.index(('plan',None)),t.index(('process','B2')))
        self.assertLess(t.index(('process','B2')),t.index(('idle',None)))

    def test_war_fine_event_and_coarse_stream(self):
        for fine in [True,False]:
            record=[]
            runner=NS(shared_read_done_event='read-done' if fine else None)
            s=NS(_war_barrier_enabled=True,model_worker=NS(last_shared_read_runner=runner),forward_stream='fwd',
                 schedule_stream=NS(wait_event=lambda e:record.append(('event',e)),wait_stream=lambda e:record.append(('stream',e))))
            env={'envs':NS(SGLANG_FORCE_COARSE_WAR_BARRIER=NS(get=lambda:False))}
            load('managers/scheduler.py','_apply_war_barrier',env)(s)
            self.assertEqual(record,[('event','read-done') if fine else ('stream','fwd')])
            self.assertIsNone(runner.shared_read_done_event)

    def test_source_links_resolve_to_named_function(self):
        model=json.loads(subprocess.check_output(['node','-e',
            "console.log(JSON.stringify(require('./pages/sglang/loop-journey/model.js').sources))"],text=True))
        for key,(path,line,name) in model.items():
            functions=[n for n in ast.walk(source(path)) if isinstance(n,ast.FunctionDef) and n.name==name]
            self.assertTrue(functions,f'{key}: missing {name}')
            self.assertTrue(any(n.lineno-2<=line<=n.end_lineno for n in functions),f'{key}: anchor outside {name}: {line}')

    def test_finished_overshoot_is_skipped_after_copy_sync(self):
        fn=next(n for n in ast.walk(source('managers/scheduler_components/batch_result_processor.py'))
                if isinstance(n,ast.FunctionDef) and n.name=='process_batch_result_decode')
        text=ast.unparse(fn)
        self.assertLess(text.index('result.copy_done.synchronize()'),text.index('req.finished() or req.is_retracted'))
        self.assertIn('continue',text[text.index('req.finished() or req.is_retracted'):][:300])

if __name__=='__main__':unittest.main()
