"""Execute fixed-source host functions with recording stubs; no SGLang/GPU import."""
import ast
import copy
import os
import subprocess
import unittest
from functools import lru_cache
from types import SimpleNamespace as NS

SOURCE = os.environ.get("SGLANG_SOURCE_DIR")
if not SOURCE:
    raise SystemExit("SGLANG_SOURCE_DIR required; fixed-source checks NOT_EXECUTED")
REV = "279339f113b79af84f27fd3ac92d0a13bd3f4cbd"
RESULT = "managers/scheduler_components/batch_result_processor.py"
SPEC = "speculative/spec_utils.py"

@lru_cache(None)
def source(path):
    return subprocess.check_output(["git", "-C", SOURCE, "show", REV + ":python/sglang/srt/" + path], text=True)

def node(path, name):
    return next(n for n in ast.walk(ast.parse(source(path))) if isinstance(n, ast.FunctionDef) and n.name == name)

def function(path, name, **env):
    n = copy.deepcopy(node(path, name)); n.decorator_list = []
    module = ast.parse("from __future__ import annotations\n"); module.body.append(n)
    ast.fix_missing_locations(module)
    exec(compile(module, path, "exec"), env)
    return env[name]

class Vec(list):
    is_cpu = True
    @property
    def shape(self): return (len(self),)
    def tolist(self): return list(self)

class Source(unittest.TestCase):
    def test_actual_dfs_restores_matcher_and_skips_illegal_node(self):
        state = [0]; log = []; masks = [[-1] for _ in range(5)]
        allowed = {(0,): [1, 2], (0, 1): [3], (0, 1, 3): [5], (0, 2): [3]}
        def accept(token):
            self.assertIn(token, allowed[tuple(state)])
            state.append(token); log.append(("accept", token))
        def rollback(k):
            log.append(("rollback", state[-1])); del state[-k:]
        def fill(rows, i):
            rows[i][0] = sum(1 << t for t in allowed[tuple(state)])
        grammar = NS(accept_token=accept, rollback=rollback, fill_vocab_mask=fill, is_terminated=lambda: False)
        function(SPEC, "traverse_tree")(Vec([1,3,4,-1,-1]), Vec([-1,2,-1,-1,-1]), Vec([0,1,2,3,4]), grammar, masks, vocab_size=6)
        self.assertEqual(log, [("accept",1),("accept",3),("rollback",3),("rollback",1),("accept",2),("rollback",2)])
        self.assertEqual(state, [0]); self.assertEqual(masks, [[6],[8],[8],[32],[-1]])

    def test_barrier_then_tree_resolution_then_nonblocking_mask_copy(self):
        log = []
        class Mask:
            def to(self, device, **kw):
                log.append(("copy", device, kw)); return "gpu-mask"
        info = NS(vocab_size=6, grammar_mask="stale")
        def generate(*args): log.append("dfs"); return Mask(), "grammar"
        build = function(SPEC, "build_grammar_vocab_mask", generate_token_bitmask=generate,
                         GrammarMask=lambda *args: args)
        result = build(reqs=[], tree=NS(resolve=lambda: log.append("resolve") or (1,2,3)),
                       sampling_info=info, device="cuda", barrier=lambda: log.append("barrier"))
        self.assertEqual(log, ["barrier","resolve","dfs",("copy","cuda",{"non_blocking":True})])
        self.assertIsNone(info.grammar_mask); self.assertEqual(result, ("grammar","gpu-mask"))

    def test_cache_miss_returns_future_and_hit_creates_fresh_matcher(self):
        ctx = object()
        def grammar(matcher, size, ctx, stop, key, stats):
            return NS(matcher=matcher, ctx=ctx, accepted_tokens=[], maybe_init_reasoning=lambda r: None)
        clone = function("constrained/xgrammar_backend.py", "copy",
                         GrammarMatcher=lambda ctx, **kw: NS(ctx=ctx, history=[]),
                         MAX_ROLLBACK_TOKENS=32, XGrammarGrammar=grammar)
        original = NS(ctx=ctx, override_stop_tokens=None, vocab_size=6, key_string="regex", grammar_stats=None)
        original.copy = lambda: clone(original)
        submitted = []; future = object()
        backend = NS(cache={}, executor=NS(submit=lambda *args: submitted.append(args) or future), _init_value_dispatch=object())
        get = function("constrained/base_grammar_backend.py", "get_cached_or_future_value")
        self.assertEqual(get(backend, ("regex","yes|no"), False), (future,False)); self.assertEqual(len(submitted),1)
        backend.cache[("regex","yes|no")] = original
        r1, hit = get(backend, ("regex","yes|no"), False)
        r1.accepted_tokens.append(1); r1.matcher.history.append(1)
        r2, _ = get(backend, ("regex","yes|no"), False)
        self.assertTrue(hit); self.assertIs(r1.ctx, r2.ctx); self.assertIsNot(r1.matcher, r2.matcher)
        self.assertEqual(r2.accepted_tokens, []); self.assertEqual(r2.matcher.history, [])

    def test_formal_advance_is_idempotent_truncates_and_settles_per_request_stride(self):
        log = []
        accept = function(RESULT, "_accept_grammar_tokens", logger=NS(error=lambda s: log.append("error")), FINISH_ABORT=lambda: "abort")
        advance = function(RESULT, "advance_grammar_fsm", _get_speculative_output_stride=function(RESULT, "_get_speculative_output_stride"))
        worker = NS(model_worker=NS(on_verify_complete_cpu=lambda *a, **kw: None))
        worker._accept_grammar_tokens = lambda req, tokens: accept(worker, req, tokens)
        worker.advance_grammar_fsm = lambda result, batch: advance(worker, result, batch)
        histories = [[],[]]
        reqs = []
        for i in range(2):
            reqs.append(NS(rid=str(i), grammar=NS(
                accept_token=lambda token,i=i: histories[i].append(token),
                is_terminated=lambda i=i: 9 in histories[i]),
                is_retracted=False, finished=lambda:False, kv=NS(kv_committed_len=8),
                output_ids=[], spec_verify_ct=0, spec_num_correct_drafts=0,
                update_spec_correct_drafts_histogram=lambda n:None))
        result = NS(grammar_advanced=False, copy_done=NS(synchronize=lambda:log.append("copy_done")),
                    next_token_ids=Vec([1,9,2,99,3,4,99,99]), accept_lens=Vec([3,2]),
                    speculative_output_stride=4, num_non_draft_tokens_per_req=1,
                    block_accept_lens=None, cap_lens=None)
        batch = NS(reqs=reqs, has_grammar=True, forward_mode=NS(is_decode=lambda:True))
        worker.advance_grammar_fsm(result,batch); worker.advance_grammar_fsm(result,batch)
        self.assertEqual(histories, [[1,9],[3,4]]); self.assertEqual(log, ["copy_done"])
        self.assertEqual(result.grammar_retained_tokens, histories)
        resolve = function(RESULT,"_resolve_spec_v2_tokens",_get_speculative_output_stride=function(RESULT,"_get_speculative_output_stride"))
        self.assertEqual(resolve(worker,result,batch), [[1,9],[3,4]])
        self.assertEqual([r.kv.kv_committed_len for r in reqs], [10,10])
        self.assertEqual([r.output_ids for r in reqs], [[],[]], "output appended by the caller, not resolver")
        self.assertEqual(histories, [[1,9],[3,4]], "no second grammar advance")
        self.assertEqual(result.num_correct_drafts_per_req_cpu, [2,1])

    def test_invalid_formal_token_aborts(self):
        def invalid(token): raise ValueError("illegal")
        req = NS(rid="R", grammar=NS(accept_token=invalid))
        accept = function(RESULT,"_accept_grammar_tokens",logger=NS(error=lambda s:None),FINISH_ABORT=lambda:"abort")
        self.assertEqual(accept(None,req,[1,2]),[]); self.assertEqual(req.to_finish,"abort")

    def test_source_launch_order_and_reserved_allocation(self):
        verify = ast.get_source_segment(source("speculative/eagle_worker_common.py"), node("speculative/eagle_worker_common.py","run_eagle_verify"))
        names = ["GrammarTree.from_device(", "target_worker.forward_batch_generation(", "build_grammar_vocab_mask(", "eagle_sample("]
        positions = [verify.index(n) for n in names]; self.assertEqual(positions,sorted(positions))
        worker = ast.get_source_segment(source("speculative/eagle_worker_v2.py"),node("speculative/eagle_worker_v2.py","forward_batch_generation"))
        decode = worker[worker.index("self.activate_step_by_batch"):]
        positions=[decode.index(n) for n in ["self.draft_worker.draft(batch)","self.verify(batch","on_publish(batch_output.new_seq_lens)","self.draft_worker._draft_extend_for_decode("]]
        self.assertEqual(positions,sorted(positions))
        alloc = function("mem_cache/allocation_sizing.py","page_aligned_decode_alloc_lens")
        req = NS(kv=NS(kv_allocated_len=20, kv_committed_len=8))
        self.assertEqual(alloc([req], reserve=5, page_size=1), ([20],[20],0))
        req.kv.kv_committed_len = 18
        self.assertEqual(alloc([req], reserve=5, page_size=1), ([20],[23],3))

if __name__ == "__main__": unittest.main()
