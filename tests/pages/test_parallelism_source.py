"""Pinned-source CPU call-order checks; no model, CUDA or distributed runtime.

SGLANG_SOURCE_DIR=/path/to/sglang python3 -B tests/pages/test_parallelism_source.py
Original function bodies are executed with recording I/O. Device duration and
concurrency remain illustrative, and are tested separately as teaching invariants.
"""
import __future__
import ast
import collections
import contextlib
import copy
import functools
import os
import subprocess
import unittest
from types import SimpleNamespace as NS

REVISION = "279339f113b79af84f27fd3ac92d0a13bd3f4cbd"
SOURCE = os.environ.get("SGLANG_SOURCE_DIR")
if not SOURCE:
    raise SystemExit("SGLANG_SOURCE_DIR is required; source checks were not run")
PP = "managers/scheduler_pp_mixin.py"


@functools.lru_cache(None)
def source(path):
    return ast.parse(subprocess.check_output(
        ["git", "-C", SOURCE, "show", f"{REVISION}:python/sglang/srt/{path}"], text=True))


def load(path, name, env=None):
    node = copy.deepcopy(next(n for n in ast.walk(source(path))
                             if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    namespace = dict(env or {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


class WindowEnd(Exception):
    pass


class ParallelSource(unittest.TestCase):
    def loop_trace(self, depth):
        trace = []
        cur, old = NS(), NS()
        scheduler = NS(ps=NS(pp_size=2), pp_group=NS(is_last_rank=False),
                       pp_loop_size=2, running_mbs=[None, None], last_mbs=[None, None],
                       mbs=[None, old], mb_metadata=[None, None], last_rank_comm_queue=[],
                       send_req_work=[], send_proxy_work=[], pp_outputs=None)
        def ingest():
            if "ingest" in trace:
                raise WindowEnd()
            trace.append("ingest")
            return [cur]
        scheduler.init_pp_loop_state = lambda: None
        scheduler.ingest_requests = ingest
        scheduler._pp_commit_comm_work = lambda w: None
        scheduler._pp_send_pyobj_to_next_stage = lambda *a, **kw: trace.append("send requests") or []
        scheduler.get_next_batch_to_run = lambda **kw: NS(running_batch=None, batch_to_run=cur)
        scheduler._pp_recv_proxy_tensors = lambda: trace.append("recv proxy")
        scheduler._pp_launch_batch = lambda *a: (trace.append("launch") or NS(pp_hidden_states_proxy_tensors=NS(tensors={})), "launch-event")
        scheduler._pp_commit_send_output_work_and_preprocess_output_tensors = lambda *a: (
            trace.append("outputs") or "outputs", "old result", NS(synchronize=lambda: trace.append("D2H sync")))
        scheduler._pp_process_batch_result = lambda *a: trace.append("process old")
        scheduler.device_module = NS(current_stream=lambda: NS(wait_event=lambda e: trace.append("wait launch event")))
        scheduler._pp_send_dict_to_next_stage = lambda *a, **kw: trace.append("send proxy") or []
        env = {"torch": NS(profiler=NS(record_function=lambda s: contextlib.nullcontext())),
               "get_parallel": lambda: NS(pp_async_batch_depth=depth)}
        with self.assertRaises(WindowEnd):
            load(PP, "event_loop_pp", env)(scheduler)
        return trace

    def test_depth_zero_launches_before_old_result_processing(self):
        trace = self.loop_trace(0)
        self.assertLess(trace.index("launch"), trace.index("outputs"))
        self.assertLess(trace.index("outputs"), trace.index("D2H sync"))
        self.assertLess(trace.index("D2H sync"), trace.index("process old"))
        self.assertLess(trace.index("process old"), trace.index("wait launch event"))
        self.assertLess(trace.index("wait launch event"), trace.index("send proxy"))

    def test_nonzero_depth_has_a_different_order(self):
        trace = self.loop_trace(1)
        self.assertLess(trace.index("outputs"), trace.index("launch"))
        self.assertLess(trace.index("launch"), trace.index("process old"))

    def test_forward_stream_depends_on_schedule_stream(self):
        trace = []
        event = NS(record=lambda stream: trace.append("record forward completion"))
        sched = NS(forward_stream_ctx=contextlib.nullcontext(), schedule_stream="schedule",
                   forward_stream=NS(wait_stream=lambda s: trace.append("forward waits " + s)),
                   device_module=NS(Event=lambda: event, current_stream=lambda: "forward"),
                   pp_group=NS(is_last_rank=False), run_batch=lambda *a: trace.append("run_batch") or NS(can_run_cuda_graph=False))
        env = {"torch": NS(profiler=NS(record_function=lambda s: contextlib.nullcontext())),
               "set_time_batch": lambda *a, **kw: None, "PPBatchMetadata": lambda **kw: NS(**kw)}
        load(PP, "_pp_launch_batch", env)(sched, 0, NS(reqs=[]), None, [None], [])
        self.assertEqual(trace, ["forward waits schedule", "run_batch", "record forward completion"])

    def test_last_rank_copy_follows_output_send_stream_dependency(self):
        trace = []
        batch = NS(forward_mode=NS(is_prebuilt=lambda: False))
        group = NS(is_last_rank=True)
        env = {"_is_npu": False, "is_xpu": lambda: False, "_pp_can_skip_output_comm": lambda b: False,
               "PPProxyTensors": lambda tensors: NS(tensors=tensors),
               "torch": NS(profiler=NS(record_function=lambda s: contextlib.nullcontext()))}
        sched = NS(pp_group=group, ps=NS(pp_size=2, pp_rank=1),
                   device_module=NS(current_stream=lambda: NS(wait_event=lambda e: trace.append("wait current GPU output")),
                                    Event=lambda: NS(record=lambda s: trace.append("record copy event"))),
                   copy_stream_ctx=contextlib.nullcontext(), schedule_stream="schedule",
                   copy_stream=NS(wait_stream=lambda s: trace.append("copy waits schedule")))
        sched._pp_send_dict_to_next_stage = lambda *a, **kw: trace.append("send output") or []
        sched._pp_recv_dict_from_prev_stage = lambda: trace.append("recv old output") or {}
        sched._pp_prep_batch_result = lambda *a: trace.append("prepare CPU output") or NS(logits_output=None)
        sender = load(PP, "_pp_send_output_to_next_stage", env)
        sched._pp_send_output_to_next_stage = lambda *a: sender(sched, *a)
        queue = collections.deque([("gpu event", NS(tensors={}))])
        load(PP, "_pp_send_recv_and_preprocess_output_tensors", env)(sched, 0, 1, [batch, batch], [None, None], queue, None)
        self.assertEqual(trace, ["wait current GPU output", "send output", "recv old output",
                                 "copy waits schedule", "prepare CPU output", "record copy event"])

    def test_round_robin_sends_each_request_without_a_compute_barrier(self):
        calls = []
        s = NS(_active_workers=[0, 1], round_robin_counter=0, status=[True, True], workers=[0, 1],
               maybe_external_dp_rank_routing=lambda req: False)
        route = load("managers/data_parallel_controller.py", "round_robin_scheduler",
                     {"sock_send": lambda w, r: calls.append((w, r)), "logger": NS(debug=lambda *a: None)})
        for req in ["R1", "R2", "R3"]:
            route(s, req)
        self.assertEqual(calls, [(0, "R1"), (1, "R2"), (0, "R3")])
        calls.clear();s.status[0] = False
        route(s, "R4");self.assertEqual(calls, [(1, "R4")])

    def test_ep_dispatch_compute_combine_order(self):
        trace = []
        s = NS(deprecate_flag=False,
               dispatcher=NS(dispatch=lambda **kw: trace.append("dispatch") or "expert rows",
                             combine=lambda **kw: trace.append("combine") or "token rows"),
               run_moe_core=lambda x: trace.append("compute") or "expert output")
        result = load("layers/moe/ep_moe/layer.py", "forward_impl")(s, "hidden", "topk")
        self.assertEqual(trace, ["dispatch", "compute", "combine"])
        self.assertEqual(result, "token rows")

    def test_sp_plain_column_gathers_before_gemm(self):
        trace = []
        linear = NS(quant_method=NS(apply=lambda *a: trace.append("GEMM") or "output"))
        env = {"sp_num_tokens": lambda: 5, "sp_fused_matmul_eligible": lambda _: False,
               "sp_exit_gather": lambda *a, **kw: trace.append("gather") or "all rows"}
        load("layers/layernorm_sp.py", "column_parallel_g_matmul", env)(linear, "local rows", None)
        self.assertEqual(trace, ["gather", "GEMM"])

    def test_dcp_lse_correction_precedes_reduce_scatter(self):
        fn = next(n for n in ast.walk(source("layers/dcp/comm.py"))
                  if isinstance(n, ast.FunctionDef) and n.name == "cp_lse_ag_out_rs_mla")
        calls = sorted((n.lineno, ast.unparse(n.func)) for n in ast.walk(fn) if isinstance(n, ast.Call))
        order = [name for _, name in calls]
        self.assertLess(order.index("_ag_lse"), order.index("correct_attn_out"))
        self.assertLess(order.index("correct_attn_out"), order.index("cp_group.reduce_scatter_along_dim"))


if __name__ == "__main__":
    unittest.main()
