"""Execute pinned metric functions with CPU substitutes, not an engine benchmark."""
import ast
import copy
import os
import subprocess
import unittest
from functools import lru_cache
from types import SimpleNamespace as NS

SOURCE = os.environ.get('SGLANG_SOURCE_DIR')
if not SOURCE:
    raise SystemExit('SGLANG_SOURCE_DIR required; fixed-source execution NOT_EXECUTED')
REV = '279339f113b79af84f27fd3ac92d0a13bd3f4cbd'
STATS = 'observability/req_time_stats.py'
MODE = NS(PREFILL='P', DECODE='D')
STAGE = NS(**{k:k for k in ['PREFILL_FORWARD','PREFILL_BOOTSTRAP','DECODE_TRANSFERRED','REQUEST_PROCESS','PREFILL_CHUNKED_FORWARD']})

@lru_cache(None)
def source(path):
    return subprocess.check_output(['git','-C',SOURCE,'show',REV+':python/sglang/srt/'+path],text=True)

def function(path, name, **env):
    node = copy.deepcopy(next(n for n in ast.walk(ast.parse(source(path))) if isinstance(n,ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    module = ast.parse('from __future__ import annotations\n')
    module.body.append(node)
    ast.fix_missing_locations(module)
    namespace = {'RequestStage':STAGE,'DisaggregationMode':MODE,**env}
    exec(compile(module,path,'exec'),namespace)
    return namespace[name]

class PerformanceSource(unittest.TestCase):
    def test_chunking_preserves_forward_origin(self):
        seen=[]
        req=NS(forward_entry_time=0,last_forward_entry_time=0,prefill_finished_time=0,last_chunked_prefill_finish_time=0,
               enable_metrics=False,trace_ctx=NS(tracing_enable=False),observe_per_stage_req_latency=lambda *a:seen.append(a),trace_slice=lambda *a:None)
        forward=function(STATS,'set_forward_entry_time')
        forward(req,ts=.210)
        function(STATS,'set_last_chunked_prefill_finish_time')(req,ts=.280)
        forward(req,ts=.285)
        self.assertEqual(req.last_forward_entry_time,.210)
        function(STATS,'set_prefill_finished_time')(req,ts=.350)
        self.assertEqual(seen[-1][0],'PREFILL_FORWARD')
        self.assertAlmostEqual(seen[-1][1],.140)
        function(STATS,'set_retract_time')(req,ts=.400)
        forward(req,ts=.450)
        self.assertEqual(req.last_forward_entry_time,.450)

    def test_decode_transfer_ends_at_wait_queue_commit(self):
        seen=[]
        req=NS(wait_queue_entry_time=0,enable_metrics=True,trace_ctx=NS(tracing_enable=False),disagg_mode=MODE.DECODE,
               decode_transfer_queue_entry_time=.180,observe_per_stage_req_latency=lambda *a:seen.append(a),trace_slice=lambda *a:None)
        function(STATS,'set_wait_queue_entry_time')(req,ts=.410)
        self.assertEqual(seen[0][0],'DECODE_TRANSFERRED')
        self.assertAlmostEqual(seen[0][1],.230)
        self.assertEqual(req.wait_queue_entry_time,.410)

    def test_sender_override_fallback_and_missing_bytes(self):
        seen=[]
        req=NS(completion_time=.426,prefill_transfer_queue_entry_time=.355,enable_metrics=True,
               prefill_bootstrap_queue_entry_time=0,bootstrap_done_time=0,wait_queue_entry_time=0,
               metrics_collector=NS(observe_kv_transfer_metrics=lambda **kw:seen.append(kw)))
        metric=NS(transfer_total_bytes=64*1024*1024,transfer_latency_s=None)
        fn=function(STATS,'compute_and_observe_kv_transfer_metrics')
        result=fn(req,metric)
        self.assertAlmostEqual(result['latency_ms'],71)
        self.assertEqual(result['total_mb'],64)
        self.assertAlmostEqual(result['speed_gb_s'],64/1024/.071)
        metric.transfer_latency_s=.048
        self.assertEqual(fn(req,metric)['latency_ms'],48)
        metric.transfer_total_bytes=None
        self.assertIsNone(fn(req,metric))
        self.assertEqual(len(seen),2)

    def test_prefill_skips_ttft_but_still_counts_finished_output(self):
        fn=function('managers/tokenizer_manager.py','collect_metrics')
        for mode in [MODE.PREFILL, MODE.DECODE]:
            calls=[]
            collector=NS(labels={'engine_type':mode},observe_time_to_first_token=lambda *a,**kw:calls.append(('ttft',a)),
                         observe_inter_token_latency=lambda *a:calls.append(('itl',a)),observe_one_finished_request=lambda *a,**kw:calls.append(('finished',a)))
            manager=NS(metrics_collector=collector,enable_priority_scheduling=False,disaggregation_mode=mode,_request_has_grammar=lambda o:False)
            state=NS(obj=NS(stream=True),ttft_observed=False,last_completion_tokens=0,finished=True,
                     time_stats=NS(get_first_token_latency=lambda:.333,get_interval=lambda:0,set_last_time=lambda:None,get_e2e_latency=lambda:.453))
            n=1 if mode==MODE.PREFILL else 4
            fn(manager,state,NS(completion_tokens=[n],prompt_tokens=[20],cached_tokens=[0]),0)
            self.assertEqual(any(c[0]=='ttft' for c in calls),mode==MODE.DECODE)
            self.assertEqual(calls[-1][0],'finished')
            self.assertEqual(calls[-1][1][2],n)

    def test_itl_histogram_weights_each_new_token(self):
        samples=[]
        # Use the public collector-injection branch; the pinned method repeats observe N times.
        collector=NS(_histogram_cls=object(),histogram_inter_token_latency=NS(labels=lambda **kw:NS(observe=samples.append)))
        function('observability/metrics_collector.py','observe_inter_token_latency')(collector,{},.120,3)
        self.assertEqual(samples,[.04,.04,.04])

if __name__=='__main__':
    unittest.main()
