"""Reviewed local SGLang baseline for the PP teaching pages (2026-09-17)."""

SOURCE_COMMIT = "279339f113b79af84f27fd3ac92d0a13bd3f4cbd"
SOURCE_SHORT = SOURCE_COMMIT[:10]

SOURCE_FILES = {'pp': 'managers/scheduler_pp_mixin.py', 'scheduler': 'managers/scheduler.py', 'prefill': 'disaggregation/prefill.py', 'policy': 'managers/schedule_policy.py', 'cache': 'mem_cache/unified_radix_cache.py', 'controller': 'managers/cache_controller.py'}

# Exact anchor text reviewed against the pinned local checkout.
SOURCE_ANCHORS = {
    'cache:3073': 'def _sync_hicache_ready_counts(',
    'cache:3187': 'def loading_check(self, finish_count: Optional[int] = None) -> None:',
    'cache:321': 'def _drain_async_work(self):',
    'controller:64': 'def wait(self, layer_index: int):',
    'controller:941': 'def start_loading(self) -> int:',
    'controller:960': 'completion = self.l2_transfer_engine.submit_host_to_device(',
    'policy:1208': 'if req.needs_host_load_back():',
    'policy:1347': 'def _commit_prefill_admission(',
    'pp:1072': 'def _do_recv():',
    'pp:1084': 'with self.copy_stream_ctx:',
    'pp:1221': 'def _pp_launch_batch(',
    'pp:1229': 'with torch.profiler.record_function("run_batch"):',
    'pp:221': 'while True:',
    'pp:223': 'for mb_id in range(self.pp_loop_size):',
    'pp:272': 'self._pp_commit_comm_work(self.send_proxy_work)',
    'pp:305': 'next_consensus_bootstrapped_rids = (',
    'pp:311': 'self._pp_commit_comm_work(send_consensus_bootstrapped_work)',
    'pp:312': 'if tmbs[next_mb_id] is not None:',
    'pp:314': 'self._pp_commit_comm_work(send_release_work)',
    'pp:317': 'd2h_event.synchronize()',
    'pp:326': 'if not self.pp_group.is_last_rank:',
    'pp:327': 'self.send_req_work = self._pp_send_pyobj_to_next_stage(',
    'pp:331': 'bootstrapped_rids, async_send=True',
    'pp:334': 'transferred_rids, async_send=True',
    'pp:336': 'if cur_batch:',
    'pp:340': 'self.send_proxy_work = self._pp_send_dict_to_next_stage(',
    'pp:346': 'self.pp_outputs = next_pp_outputs',
    'pp:593': 'def _pp_pd_get_bootstrapped_ids(self: Scheduler):',
    'pp:635': 'def _pp_pd_get_prefill_transferred_ids(self: Scheduler):',
    'pp:660': 'def _pp_pd_send_consensus_bootstrapped_ids(',
    'pp:683': 'def _pp_pd_send_consensus_release_ids(',
    'pp:710': 'def _pp_commit_send_output_work_and_preprocess_output_tensors(',
    'pp:858': 'def _pp_recv_proxy_tensors(self: Scheduler) -> Optional[PPProxyTensors]:',
    'pp:983': 'def _pp_send_output_to_next_stage(',
    'prefill:1210': 'def process_prefill_chunk(',
    'prefill:1502': 'req.disagg_kv_sender.send(',
    'prefill:800': 'for i, (req, next_token_id) in enumerate(',
    'prefill:893': 'if not req.pending_bootstrap:',
    'prefill:972': 'def process_disagg_prefill_inflight_queue(',
    'scheduler:2064': 'def ingest_requests(self) -> List:',
    'scheduler:3959': 'req.init_next_round_input(self.tree_cache)',
    'scheduler:4058': 'new_batch.prepare_for_extend()',
}
