"""Execute selected functions FROM the pinned Git objects, with recording I/O.

SGLANG_SOURCE_DIR=/path/to/sglang python3 -B tests/pages/test_communication_source.py
Standard library only. These are source-contract checks, not GPU/RDMA tests.
No upstream module import, checkout mutation, network or model execution.
"""
import __future__
import ast
import collections
import contextlib
import copy
import enum
import functools
import logging
import os
import subprocess
import unittest
from http import HTTPStatus
from types import SimpleNamespace as NS

REVISION = "279339f113b79af84f27fd3ac92d0a13bd3f4cbd"
SOURCE = os.environ.get("SGLANG_SOURCE_DIR")
if not SOURCE:
    raise SystemExit("SGLANG_SOURCE_DIR is required; no source behavior checked")


@functools.lru_cache(None)
def tree(path):
    code = subprocess.check_output(
        ["git", "-C", SOURCE, "show", f"{REVISION}:python/sglang/srt/{path}"],
        text=True,
    )
    return ast.parse(code, filename=path)


def load(path, name, env=None, owner=None):
    """Compile the original body, preserving line numbers; replace only I/O globals."""
    scope = tree(path)
    if owner:
        scope = next(n for n in ast.walk(scope) if isinstance(n, ast.ClassDef) and n.name == owner)
    node = copy.deepcopy(next(n for n in ast.walk(scope)
                              if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name))
    node.decorator_list = []
    namespace = dict(env or {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


PARALLEL = "distributed/parallel_state.py"
PP = "managers/scheduler_pp_mixin.py"
COMMON = "disaggregation/common/conn.py"
MOONCAKE = "disaggregation/mooncake/conn.py"
DECODE = "disaggregation/decode.py"
UTILS = "disaggregation/utils.py"
HICACHE = "disaggregation/decode_hicache_mixin.py"
KVPoll = load("disaggregation/base/conn.py", "KVPoll")
Restore = load(HICACHE, "HiCacheRestoreResult", {"Enum": enum.Enum})
LOG = logging.getLogger("communication-source-contract")
LOG.addHandler(logging.NullHandler())
LOG.propagate = False


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Cells(dict):
    def __getitem__(self, key):
        return Scalar(super().__getitem__(key))


class Tensor:
    def __init__(self, size=(2, 4), dtype="float16", device="cuda", ptr=1000):
        self.shape, self.dtype = size, dtype
        self.device = NS(type=device)
        self.is_cpu, self.ptr = device == "cpu", ptr

    def size(self):
        return self.shape

    def numel(self):
        import math
        return math.prod(self.shape)

    def data_ptr(self):
        return self.ptr

    def narrow(self, dim, offset, count):
        assert dim == 0
        out = Tensor((count,), self.dtype, ptr=self.ptr + offset * 2)
        out.device = self.device
        return out


class SourceContracts(unittest.TestCase):
    def test_pp_async_sender_posts_all_work_before_wait(self):
        trace = []
        metadata = collections.namedtuple("TensorMetadata", "device dtype size")
        def isend(tensor, rank, **kw):
            trace.append(("isend", kw["group"]))
            return NS(wait=lambda: trace.append(("wait", kw["group"])))
        torch = NS(Tensor=Tensor, uint8="uint8", long="long",
                   frombuffer=lambda data, dtype: Tensor((len(data),), dtype, "cpu"),
                   tensor=lambda value, dtype, device: Tensor((len(value),), dtype, device),
                   distributed=NS(isend=isend, send=lambda *a, **k: self.fail("sync send")))
        env = {"torch": torch, "pickle": NS(dumps=lambda obj: b"metadata-bytes"), "TensorMetadata": metadata,
               "P2PWork": lambda work, payload: NS(work=work, payload=payload)}
        env["_split_tensor_dict"] = load(PARALLEL, "_split_tensor_dict", env)
        group = NS(world_size=2, rank_in_group=0, ranks=[0, 1],
                   cpu_group="cpu", device_group="device")
        send_object = load(PARALLEL, "send_object", env)
        group.send_object = lambda obj, **kw: send_object(group, obj, **kw)
        h, r = Tensor(), Tensor()
        work = load(PARALLEL, "send_tensor_dict", env)(
            group, {"hidden_states": h, "residual": r, "__msg_type__": "proxy"}, async_send=True)
        self.assertEqual(trace, [("isend", "cpu")] * 2 + [("isend", "device")] * 2)
        self.assertIs(work[2].payload, h)
        self.assertIs(work[3].payload, r)
        load(PP, "_pp_commit_comm_work")(None, work)
        self.assertEqual(trace[4:], [("wait", "cpu")] * 2 + [("wait", "device")] * 2)
        self.assertEqual(work, [])

    def test_kv_scheduler_enqueues_and_plain_worker_batches_layers(self):
        chunks, transfers = [], []
        mgr = NS(disaggregation_mode="prefill", request_status={42: KVPoll.WaitingForInput},
                 check_status=lambda room: KVPoll.WaitingForInput,
                 transfer_infos={42: {"D:9000": object()}},
                 transfer_queues=[NS(put=chunks.append)])
        env = {"DisaggregationMode": NS(PREFILL="prefill"), "KVPoll": KVPoll,
               "TransferKVChunk": lambda **kw: NS(**kw), "TraceNullContext": lambda: None}
        load(MOONCAKE, "add_transfer_request", env)(mgr, 42, [7, 8], slice(0, 2), True, aux_index=2)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].room, 42)
        worker = NS(is_mla_backend=False, is_hybrid_mla_backend=False,
                    enable_custom_mem_pool=False, max_transfer_batch_indices=0,
                    get_mha_kv_ptrs_with_pp=lambda *a: ([1000], [2000], [3000], [4000], 1),
                    _transfer_data=lambda session, blocks: transfers.append((session, blocks)) or 0)
        env = {"group_concurrent_contiguous": lambda *a: ([[7, 8]], [[3, 4]])}
        executor = NS(submit=lambda *a: self.fail("plain batch is not per-layer thread submission"))
        result = load(MOONCAKE, "_send_kvcache_generic", env)(
            worker, "D:9000", [1000, 2000], [3000, 4000], [16, 16], NS(size=2), [3, 4], executor)
        self.assertEqual(result, 0)
        self.assertEqual(transfers, [("D:9000", [(1112, 3048, 32), (2112, 4048, 32)])])

    def test_ipc_bytes_and_sender_state_are_separate(self):
        calls = []
        obj = NS(rid="R1", input_ids=[101, 102])
        sock = NS(send=lambda b, **kw: calls.append((b, kw)), recv=lambda **kw: b"encoded")
        env = {"_USE_PICKLE_IPC": False, "msgpack_encode": lambda x: b"encoded",
               "msgpack_decode": lambda b: NS(rid="R1", input_ids=[101, 102])}
        load("managers/io_struct.py", "sock_send", env)(sock, obj)
        decoded = load("managers/io_struct.py", "sock_recv", env)(sock)
        self.assertIsNot(decoded, obj)
        self.assertEqual(calls, [(b"encoded", {"flags": 0})])
        state = NS(dispatched=False)
        load("managers/tokenizer_manager.py", "_mark_state_dispatched")(
            NS(rid_to_state={"R1": state}), "R1")
        self.assertTrue(state.dispatched)
        self.assertFalse(hasattr(decoded, "dispatched"))

    def test_pp_allocation_wait_order_and_typed_inbox(self):
        metadata = collections.namedtuple("TensorMetadata", "device dtype size")
        trace = []
        def empty(size, dtype, device):
            trace.append(("allocate", size))
            return Tensor(size, dtype, device)
        def receive(tensor, **kw):
            trace.append(("irecv", kw["group"]))
            return NS(wait=lambda: trace.append(("wait", tensor.shape)))
        torch = NS(Tensor=Tensor, empty=empty,
                   distributed=NS(is_initialized=lambda: True, irecv=receive))
        env = {"torch": torch, "TensorMetadata": metadata}
        payload = {"hidden_states": Tensor(), "residual": Tensor(), "__msg_type__": "proxy"}
        meta, tensors = load(PARALLEL, "_split_tensor_dict", env)(payload)
        self.assertIs(tensors[0], payload["hidden_states"])
        self.assertEqual(meta[-1], ("__msg_type__", "proxy"))
        group = NS(world_size=2, rank_in_group=1, ranks=[4, 5], device_group="device",
                   cpu_group="cpu", recv_object=lambda **kw: meta)
        out = load(PARALLEL, "recv_tensor_dict", env)(group)
        self.assertEqual(out["__msg_type__"], "proxy")
        self.assertEqual(trace, [("allocate", (2, 4)), ("irecv", "device"), ("wait", (2, 4))] * 2)
        messages = iter([{"__msg_type__": "output"}, out])
        receiver = NS(_pp_tensor_dict_inbox=collections.defaultdict(collections.deque),
                      pp_group=NS(recv_tensor_dict=lambda **kw: next(messages)))
        receive_typed = load(PP, "_pp_recv_typed_dict", {"logger": LOG})
        self.assertIs(receive_typed(receiver, "proxy"), out)
        self.assertEqual(receive_typed(receiver, "output"), {"__msg_type__": "output"})
        work = [NS(work=NS(wait=lambda: trace.append("send-wait")), payload=payload)]
        load(PP, "_pp_commit_comm_work")(None, work)
        self.assertEqual(work, [])
        self.assertEqual(trace[-1], "send-wait")

    def test_nccl_pointer_aliases_stream_and_equal_split(self):
        calls = []
        class Nccl:
            def __getattr__(self, name):
                return lambda *args: calls.append((name, args))
        env = {"ReduceOp": NS(SUM="sum"), "buffer_type": lambda p: p,
               "cudaStream_t": lambda p: p,
               "ncclDataTypeEnum": NS(from_torch=lambda x: x),
               "ncclRedOpTypeEnum": NS(from_torch=lambda x: x)}
        a, b = Tensor((4,)), Tensor((4,), ptr=2000)
        b.device = a.device
        comm = NS(disabled=False, device=a.device, nccl=Nccl(), comm="group", world_size=2,
                  _resolve_stream=lambda: NS(cuda_stream=37))
        path = "distributed/device_communicators/pynccl.py"
        load(path, "all_reduce", env)(comm, a)
        self.assertEqual(calls.pop(), ("ncclAllReduce", (1000, 1000, 4, "float16", "sum", "group", 37)))
        b.shape = (8,)  # two ranks, four input elements each
        load(path, "all_gather", env)(comm, b, a)
        b.shape = (4,)
        self.assertEqual(calls.pop(), ("ncclAllGather", (1000, 2000, 4, "float16", "group", 37)))
        exchange = load(path, "all_to_all_single", env)
        exchange(comm, b, a)
        self.assertEqual([name for name, _ in calls],
                         ["ncclGroupStart", "ncclSend", "ncclRecv", "ncclSend", "ncclRecv", "ncclGroupEnd"])
        self.assertEqual([args[0] for _, args in calls[1:-1]], [1000, 2000, 1004, 2004])
        self.assertEqual([args[3] for _, args in calls[1:-1]], [0, 0, 1, 1])
        a.shape = b.shape = (3,)
        with self.assertRaises(AssertionError):
            exchange(comm, b, a)

    def test_kv_pointer_plan_engine_arguments_and_aux_index(self):
        env = {"dst_device_data_ptrs": None, "prefill_kv_blocks": [[7, 8], [12]],
               "dst_kv_blocks": [[3, 4], [9]]}
        plan = load(MOONCAKE, "set_transfer_blocks", env)(1000, 2000, 16)
        self.assertEqual(plan, [(1112, 2048, 32), (1192, 2144, 16)])
        calls = []
        engine = NS(batch_transfer_sync=lambda *args: calls.append(args) or 0)
        transfer = load(MOONCAKE, "_transfer_data")
        transfer(NS(engine=engine), "D-session", plan)
        self.assertEqual(calls, [("D-session", [1112, 1192], [2048, 2144], [32, 16])])
        transfer(NS(engine=engine), "D-session", [])
        self.assertEqual(len(calls), 1)
        sender = NS(enable_custom_mem_pool=False,
                    kv_args=NS(aux_data_ptrs=[3000, 4000], aux_item_lens=[8, 16]),
                    _transfer_data=lambda session, blocks: calls.append((session, blocks)) or 0)
        env = {"envs": NS(SGLANG_MOONCAKE_SEND_AUX_TCP=NS(get=lambda: False))}
        load(MOONCAKE, "send_aux", env)(sender, NS(mooncake_session_id="D-session", dst_aux_index=3), 2, [5000, 6000])
        self.assertEqual(calls[-1], ("D-session", [(3016, 5024, 8), (4032, 6048, 16)]))

    def test_success_deduplicates_and_does_not_resurrect_cleared_room(self):
        state = {42: KVPoll.Transferring}
        tracker = collections.defaultdict(set)
        mgr = NS(request_status=state, prefill_response_tracker=tracker,
                 required_prefill_response_num_table={42: 2}, enable_staging=False,
                 update_status=lambda room, status: state.__setitem__(room, status))
        apply = load(COMMON, "apply_prefill_status", {"KVPoll": KVPoll, "logger": LOG})
        for rank in [0, 0]:
            apply(mgr, bootstrap_room=42, status=KVPoll.Success, prefill_rank=rank)
            self.assertEqual(state[42], KVPoll.Transferring)
        apply(mgr, bootstrap_room=42, status=KVPoll.Success, prefill_rank=1)
        self.assertEqual(state[42], KVPoll.Success)
        state.clear()
        apply(mgr, bootstrap_room=42, status=KVPoll.Success, prefill_rank=2)
        self.assertEqual(state, {})
        self.assertEqual(tracker[42], {0, 1})

    def test_sender_failure_cannot_be_overwritten_by_late_success(self):
        emitted = []
        mgr = NS(request_status={42: KVPoll.Failed}, failure_records={},
                 failure_lock=contextlib.nullcontext(),
                 update_status=lambda room, status: mgr.request_status.__setitem__(room, status),
                 send_kv_status_message=lambda **kw: emitted.append(kw))
        conclude = load(COMMON, "conclude_transfer", {"KVPoll": KVPoll})
        self.assertEqual(conclude(mgr, bootstrap_room=42, status=KVPoll.Success, targets=[]), KVPoll.Failed)
        self.assertEqual(emitted[-1]["status"], KVPoll.Failed)
        mgr.request_status.clear()
        self.assertIsNone(conclude(mgr, bootstrap_room=42, status=KVPoll.Success, targets=[]))
        self.assertEqual(len(emitted), 1)

    def test_hicache_and_metadata_gate_precede_min(self):
        calls = []
        env = {"KVPoll": KVPoll, "HiCacheRestoreResult": Restore, "_is_fake_transfer": lambda r: False}
        env["HiCacheRestoreGatedKVReceiver"] = load(HICACHE, "HiCacheRestoreGatedKVReceiver", env)
        env["_apply_metadata_gate"] = load(UTILS, "_apply_metadata_gate", env)
        def reduce_polls(polls, group):
            calls.append((list(polls), group))
            return [min(polls[0], KVPoll.Success)]
        env.update(_all_reduce_polls=reduce_polls, _poll_with_failure_injection=lambda ps: [p.poll() for p in ps])
        env["poll_and_all_reduce"] = load(UTILS, "poll_and_all_reduce", env)
        poll = load(DECODE, "_poll_with_metadata_gate", env)
        req = NS(req=NS(), metadata_buffer_index=3, hicache_restore_status=Restore.PENDING,
                 kv_receiver=NS(poll=lambda: KVPoll.Success))
        rooms = Cells({(3, 0): 42})
        queue = NS(queue=[req], gloo_group="D-group", scheduler=NS(enable_decode_hicache=True),
                   metadata_buffers=NS(bootstrap_room=rooms))
        self.assertEqual(poll(queue), [KVPoll.Transferring])
        self.assertEqual(calls[-1], ([KVPoll.Transferring], "D-group"))
        req.hicache_restore_status = Restore.READY
        rooms[3, 0] = 0
        self.assertEqual(poll(queue), [KVPoll.Transferring])
        rooms[3, 0] = 99  # identity is NOT checked by this gate
        self.assertEqual(poll(queue), [KVPoll.Success])

    def test_poll_collective_is_cpu_uint8_min(self):
        calls = []
        tensor = NS(tolist=lambda: [KVPoll.Transferring])
        env = {"torch": NS(uint8="uint8", tensor=lambda values, **kw: calls.append((values, kw)) or tensor),
               "dist": NS(ReduceOp=NS(MIN="MIN"), all_reduce=lambda t, **kw: calls.append(kw))}
        out = load(UTILS, "_all_reduce_polls", env)([KVPoll.Success], "D-group")
        self.assertEqual(calls, [([4], {"dtype": "uint8", "device": "cpu"}), {"op": "MIN", "group": "D-group"}])
        self.assertEqual(out, [3])

    def test_commit_room_identity_before_output_token(self):
        for actual in [0, 99, 42]:
            aborted, cleared = [], []
            values = [[Scalar(320)], [Scalar(0)] * 7] + [None] * 11 + [[Scalar(actual)]]
            req = NS(bootstrap_room=42, rid="R1", output_ids=[], return_logprob=False,
                     return_sampling_mask=False, time_stats=NS(set_wait_queue_entry_time=lambda: None))
            dr = NS(req=req, metadata_buffer_index=3, is_rebootstrap=False,
                    kv_receiver=NS(clear=lambda: cleared.append(True)))
            queue = NS(scheduler=NS(kv_checksum_computer=None,
                                   batch_result_processor=NS(_maybe_update_reasoning_tokens=lambda *a: None)),
                       metadata_buffers=NS(get_buf=lambda idx: values),
                       _commit_hicache_local_restore_to_req=lambda dr: None,
                       spec_algorithm=NS(is_none=lambda: True))
            env = {"logger": LOG, "HTTPStatus": HTTPStatus, "_is_fake_transfer": lambda r: False,
                   "prepare_abort": lambda *a, **kw: aborted.append(True)}
            load(DECODE, "_commit_transfer_to_req", env)(queue, dr)
            self.assertEqual(req.output_ids, [320] if actual == 42 else [])
            self.assertEqual(bool(aborted), actual != 42)
            self.assertEqual(cleared, [True])

    def test_abort_ack_requires_drained_source_and_distinct_senders(self):
        calls = []
        mgr = NS(_staging_outstanding={42: 1}, _deferred_ack_targets={42: ("D", 9)},
                 _send_abort_ack=lambda *a: calls.append(a))
        ack = load(COMMON, "_maybe_ack_drained_abort")
        ack(mgr, 42)
        self.assertEqual(calls, [])
        mgr._staging_outstanding[42] = 0
        ack(mgr, 42)
        ack(mgr, 42)
        self.assertEqual(calls, [("D", 9, 42)])
        receiver = NS(_deferred_abort_ack_tracker={42: set()})
        note = load(COMMON, "note_abort_ack")
        safe = load(COMMON, "is_abort_release_safe")
        note(receiver, 42, 0)
        note(receiver, 42, 0)
        self.assertFalse(safe(receiver, 42, 2))
        note(receiver, 42, 1)
        self.assertTrue(safe(receiver, 42, 2))

    def test_release_holds_before_deadline_but_timeout_also_releases(self):
        for drained, now, expected in [(False, 9, False), (False, 10, True), (True, 9, True)]:
            released = []
            receiver = NS(kv_mgr=NS(is_abort_release_safe=lambda room, required: drained))
            dr = NS(req=NS(bootstrap_room=42), kv_receiver=receiver)
            queue = NS(_deferred_releases=[(dr, 10, 3, 2)], deferred_kv_release_timeout=10,
                       _do_release=lambda *args: released.append(args))
            env = {"time": NS(monotonic=lambda: now), "logger": LOG}
            load(DECODE, "resolve_deferred_releases", env)(queue)
            self.assertEqual(bool(released), expected)
            self.assertEqual(bool(queue._deferred_releases), not expected)

    def test_do_release_returns_resources_and_resets_metadata(self):
        calls, rooms = [], {3: 42}
        mgr = NS(clear_deferred_abort_state=lambda room: calls.append(("clear-ack", room)))
        dr = NS(req=NS(bootstrap_room=42), kv_receiver=NS(kv_mgr=mgr, clear=lambda: calls.append("clear-receiver")))
        queue = NS(enable_staging=False, tree_cache="cache", metadata_buffers=NS(bootstrap_room=rooms),
                   req_to_metadata_buffer_idx_allocator=NS(free=lambda idx: calls.append(("free-index", idx))))
        env = {"release_kv_cache": lambda req, cache, **kw: calls.append(("release-kv", kw))}
        load(DECODE, "_do_release", env)(queue, dr, 3)
        self.assertEqual(calls, [("release-kv", {"is_insert": False}), ("free-index", 3), ("clear-ack", 42), "clear-receiver"])
        self.assertEqual(rooms[3], 0)
        self.assertIsNone(dr.kv_receiver)


if __name__ == "__main__":
    print(f"Fixed source {REVISION}; recording CPU substitutes only, no GPU/RDMA", flush=True)
    unittest.main(verbosity=2)
