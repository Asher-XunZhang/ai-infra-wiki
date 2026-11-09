"""Execute pinned functions with CPU recording substitutes; no SGLang/GPU import."""
import ast
import copy
import json
import os
import subprocess
import unittest
from functools import lru_cache
from types import SimpleNamespace as NS

SOURCE = os.environ.get('SGLANG_SOURCE_DIR')
if not SOURCE:
    raise SystemExit('SGLANG_SOURCE_DIR required; fixed-source execution NOT_EXECUTED')
REV = '279339f113b79af84f27fd3ac92d0a13bd3f4cbd'
POLL = NS(WaitingForInput=2, Bootstrapping=1, Failed=0, Success=4, Transferring=3)

@lru_cache(None)
def source(path):
    return subprocess.check_output(['git', '-C', SOURCE, 'show', REV + ':python/sglang/srt/' + path], text=True)

def node(path, name):
    return next(n for n in ast.walk(ast.parse(source(path))) if isinstance(n, ast.FunctionDef) and n.name == name)

def function(path, name, **env):
    n = copy.deepcopy(node(path, name)); n.decorator_list = []
    module = ast.parse('from __future__ import annotations\n')
    module.body.append(n); ast.fix_missing_locations(module)
    namespace = {'KVPoll': POLL, **env}
    exec(compile(module, path, 'exec'), namespace)
    return namespace[name]

PP = 'managers/scheduler_pp_mixin.py'
CACHE = 'mem_cache/unified_radix_cache.py'
PREFILL = 'disaggregation/prefill.py'

class ConsensusSource(unittest.TestCase):
    def test_bootstrap_real_intersection_union_and_abort_routing(self):
        class Abort: pass
        route = function(PP, '_route_aborts_to_bad')
        merge = function(PP, '_pp_pd_get_bootstrapped_ids', FINISH_ABORT=Abort)
        local = NS(pp_group=NS(is_first_rank=False),
                   disagg_prefill_bootstrap_queue=NS(queue=[NS(rid='R1', finished_reason=Abort())]),
                   get_rids=lambda *a: (['R0', 'R1'], []),
                   _pp_recv_pyobj_from_prev_stage=lambda: [['R0', 'R1', 'R2'], ['R2']],
                   _route_aborts_to_bad=route)
        good, bad = merge(local)
        self.assertEqual(good, ['R0']); self.assertEqual(set(bad), {'R1', 'R2'})

    def test_metadata_failure_filters_actual_returned_bootstrap_list(self):
        mapping = function('disaggregation/utils.py', 'poll_and_all_reduce_pp')
        pop = function(PREFILL, 'pop_bootstrapped', poll_and_all_reduce_pp=mapping,
                       should_force_retry=lambda req: False)
        queue = NS(pp_size=3, queue=[NS(rid=id, time_stats=NS(set_wait_queue_entry_time=lambda: None)) for id in ['R0','R1']],
                   scheduler=NS(processed_tokens_counter=0), finalize_bootstrap=lambda req: req.rid == 'R0')
        queue.pop_bootstrapped = lambda **kw: pop(queue, **kw)
        scheduler = NS(disagg_prefill_bootstrap_queue=queue, waiting_queue=[])
        processed = function(PP, 'process_bootstrapped_queue')(scheduler, [['R0','R1'], []])
        self.assertEqual(processed, [['R0'], []]); self.assertEqual([r.rid for r in queue.queue], ['R1'])
        sent = []
        scheduler.pp_group = NS(is_last_rank=False)
        scheduler._pp_send_pyobj_to_next_stage = lambda data, **kw: sent.append((data, kw)) or []
        function(PP, '_pp_pd_send_consensus_bootstrapped_ids')(scheduler, [], 0, processed, [['R0','R1'],[]])
        self.assertEqual(sent, [([['R0'],[]], {'async_send':True})])
        self.assertFalse(function(PREFILL, 'finalize_bootstrap')(NS(ensure_metadata_buffer=lambda r:False), NS(pending_bootstrap=True)))

    def test_terminal_intersection_accepts_failed_and_repolls_locally(self):
        seen = []
        def get(*args):
            seen.append(args[-1]); return ['R0','R1']
        s = NS(pp_group=NS(is_first_rank=False), disagg_prefill_inflight_queue=[],
               get_rids=get, _pp_recv_pyobj_from_prev_stage=lambda: ['R1','R2'])
        self.assertEqual(function(PP, '_pp_pd_get_prefill_transferred_ids')(s), ['R1'])
        self.assertEqual(seen[0], [POLL.Success, POLL.Failed])
        req = NS(rid='R1', disagg_kv_sender=object())
        s = NS(disagg_prefill_inflight_queue=[req],attn_cp_cpu_group=None,attn_tp_cpu_group=None,
               ps=NS(pp_rank=1),output_streamer=NS(stream_output=lambda *args:None))
        release = function(PREFILL,'process_disagg_prefill_inflight_queue',
                           poll_and_all_reduce_attn_cp_tp_group=lambda *a:[POLL.Transferring],
                           logger=NS(warning_once=lambda *a:None))
        self.assertEqual(release(s, ['R1']), []);self.assertEqual(s.disagg_prefill_inflight_queue, [req])

    def test_ack_counts_contiguous_prefix_and_only_pp0_reduces(self):
        count = function(CACHE, '_count_ready_acks')
        acks = [NS(finish_event=NS(query=lambda v=v:v)) for v in [True,False,True]]
        self.assertEqual(count(None, acks),1)
        for rank, expected in [(0,['min','pp']), (1,['pp']), (2,['pp'])]:
            log=[]; s=NS(pp_rank=rank,_all_reduce_attn_groups=lambda *a:log.append('min'),_pp_sync=lambda *a:log.append('pp'))
            function(CACHE,'_all_reduce')(s, object(), 'MIN');self.assertEqual(log,expected)

    def test_hicache_isend_precedes_local_event_and_unlock(self):
        log=[]
        class Queue(list):
            def pop(self,i):log.append('pop');return super().pop(i)
        class Tensor:
            def clone(self):log.append('clone');return Tensor()
        dist=NS(recv=lambda *a,**k:log.append('recv'),isend=lambda *a,**k:log.append('isend') or 'work')
        s=NS(pp_rank=1,pp_size=3,pp_group='cpu',work_list=[],cache_controller=NS(ack_load_queue=Queue([
            NS(finish_event=NS(synchronize=lambda:log.append('event')),node_ids=[7])])),buffer_pipeline=None,
            ongoing_load_back={7:('node','device-lock','host-lock')},dec_lock_ref=lambda *a:log.append('unlock-device'),
            dec_host_lock_ref=lambda *a:log.append('unlock-host'),tree_core=NS(finish_load_back=lambda *a:log.append('finish')),metrics_collector=None)
        function(CACHE,'_pp_sync',torch=NS(distributed=dist),P2PTag=NS(HIRADIX_PP_SYNC=9))(s,Tensor())
        function(CACHE,'loading_check')(s,1)
        self.assertEqual(log,['recv','clone','isend','pop','event','unlock-device','unlock-host','finish'])
        self.assertEqual(s.work_list,['work']);self.assertEqual(s.ongoing_load_back,{})

    def test_loop_order_and_all_animation_source_anchors(self):
        n=node(PP,'event_loop_pp_disagg_prefill')
        calls=lambda name:sorted(x.lineno for x in ast.walk(n) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr==name)
        self.assertLess(calls('get_new_batch_prefill')[0], calls('process_bootstrapped_queue')[0])
        code=ast.unparse(n)
        self.assertIn('consensus_bootstrapped_rids = next_consensus_bootstrapped_rids',code)
        cache=ast.unparse(node(CACHE,'check_hicache_events'))
        self.assertLess(cache.index('self._drain_async_work()'),cache.index('self._sync_hicache_ready_counts()'))
        self.assertLess(cache.index('self.writing_check('),cache.index('self.loading_check('))
        data=json.loads(subprocess.check_output(['node','-e',"const m=require('./pages/sglang/pp-consensus/model.js'); console.log(JSON.stringify({sources:m.sources,files:m.files}));"],text=True))
        for key,(file,line) in data['sources'].items():
            lines=source(data['files'][file]).splitlines();self.assertTrue(lines[line-1].strip(),key)
            if key!='loop':self.assertIn('def ',lines[line-1],key)

if __name__=='__main__':unittest.main()
