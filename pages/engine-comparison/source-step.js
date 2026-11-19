/* Source annotations for the selected teaching step. The arrow means a
 * transformation, lookup, or gate here, and carries no transport pulse. */
window.AtlasSourceStep = (() => {
  const h = React.createElement;
  const select = (pair, en) => Array.isArray(pair) ? pair[en ? 1 : 0] : pair;
  const SG = 'https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/';
  const anchors = {
    launch: SG + 'entrypoints/engine.py',
    tokenizer: SG + 'managers/tokenizer_manager.py',
    prefill: SG + 'managers/scheduler.py#L4052',
    forward: SG + 'managers/scheduler.py#L4614',
    finishTime: SG + 'managers/tokenizer_manager.py#L2750',
    commit: SG + 'managers/scheduler.py#L5074',
    abort: SG + 'managers/scheduler.py#L5675',
    retract: SG + 'managers/schedule_batch.py#L3451',
    admission: SG + 'managers/schedule_policy.py',
    decode: SG + 'managers/schedule_batch.py#L3736',
    output: SG + 'managers/scheduler_components/output_streamer.py#L445',
    transfer: SG + 'disaggregation/decode.py#L2918',
    hierarchy: SG + 'managers/cache_controller.py#L1078',
  };
  const detail = (anchor, symbol, input, zh, en, result) => ({ anchor, symbol, input, action: [zh, en], result });

  const steps = {
    process: [
      {
        sg: detail('tokenizer', 'TokenizerManager', 'HTTP → R1', '编码文本并附上请求参数', 'Encode the prompt and attach request settings', 'input_ids + sampling_params'),
        vl: detail('vlAsync', 'AsyncLLM', 'HTTP → R1', '前端准备核心请求', 'Prepare a core request in the frontend', 'EngineCoreRequest'),
      },
      {
        sg: detail('prefill', 'get_new_batch_prefill', 'waiting_queue', '检查准入后组织本轮工作', 'Admit requests and build the work for this step', 'NextBatchPlan.batch_to_run'),
        vl: detail('vlSched', 'Scheduler.schedule', 'waiting / running', '分配 token 工作量与 KV 块', 'Assign token work and KV blocks', 'SchedulerOutput'),
      },
      {
        sg: detail('forward', 'Scheduler.run_batch', 'ScheduleBatch', '把本轮模型输入交给本 rank 的执行侧', 'Pass this step’s model inputs to the local rank', ['本 rank 写入 K/V', 'Rank-local K/V written']),
        vl: detail('v2_execute', 'V2.execute_model', 'SchedulerOutput', '准备输入和 KV 写入位置，再做前向', 'Prepare inputs and KV slots, then run forward', 'ExecuteModelState'),
      },
      {
        sg: detail('output', 'OutputStreamer.accept', ['采样 token ID', 'Sampled token IDs'], '按流式条件收集本次输出', 'Collect output according to streaming rules', ['送 Detokenizer 处理', 'Send to Detokenizer']),
        vl: detail('vlAsync', 'AsyncLLM.output_handler', 'EngineCoreOutputs', '交给前端输出处理和反分词', 'Hand outputs to frontend processing and detokenization', ['流式文本', 'Streamed text']),
      },
    ],
    wire: [
      {
        sg: detail('sgIpc', 'ipc_channels.py', 'TokenizerManager · PUSH', '转交序列化的内部请求', 'Deliver the serialized internal request', 'Scheduler · PULL'),
        vl: detail('vlCore', 'EngineCoreProc', 'ROUTER → DEALER', '用类型帧区分新增请求和控制消息', 'Use a type frame to distinguish requests and control messages', 'ADD / ABORT / UTILITY'),
      },
      {
        sg: detail('output', 'OutputStreamer.accept', ['Scheduler 的 token ID', 'Scheduler token IDs'], '中间经过 Detokenizer', 'Pass through the separate detokenizer', ['TokenizerManager 的文本结果', 'Text at TokenizerManager']),
        vl: detail('vlCore', 'EngineCoreOutputs', 'Core · PUSH', '送回独立的输出 socket', 'Return results on a separate output socket', 'Frontend · PULL'),
      },
      {
        sg: detail('sgReceiver', 'request_receiver.py', ['入口 rank 的请求输入', 'Request input on the receiving rank'], '先同步请求，随后各 rank 执行本地部分', 'Synchronize inputs before each rank runs its part', ['各 rank 的调度输入', 'Scheduling inputs on each rank']),
        vl: detail('v2_mp', 'WorkerProc.handle_output', 'AsyncOutput.copy_event', '等复制结果可供 CPU 读取', 'Wait for the copied result to become CPU-readable', 'Worker response MQ'),
      },
      {
        sg: detail('abort', 'Scheduler.abort_request', 'AbortReq.rid', '按请求 ID 执行取消路径', 'Apply cancellation by request ID', ['对应请求的状态更新', 'State change for that request']),
        vl: detail('vlCore', 'EngineCoreProc', 'ABORT · request IDs', '把控制消息送入核心的取消处理', 'Dispatch the control message to core cancellation', 'Scheduler'),
      },
    ],
    prefix: [
      {
        sg: detail('sgRadix', 'UnifiedRadixCache.match_prefix', ['token 序列', 'Token sequence'], '沿页对齐的路径查找共享前缀', 'Match the prefix along page-aligned paths', ['已复用的 KV 位置', 'Reusable KV locations']),
        vl: detail('vlBlockPool', 'BlockPool', ['带父前缀的完整块 hash', 'Full-block hash including the parent prefix'], '查找已登记的缓存块', 'Look up registered cached blocks', ['物理块编号', 'Physical block IDs']),
      },
      {
        sg: detail('sgRadix', 'UnifiedRadixCache.match_prefix', ['前 10 个 token 相同', 'First 10 tokens match'], '按 4 个 token 一页对齐', 'Align to four tokens per page', ['复用 8 / 12 个 token', 'Reuse 8 / 12 tokens']),
        vl: detail('vlBlockPool', 'BlockPool', ['第 3 块后两项不同', 'Last two tokens of block 3 differ'], '命中两块完整 hash', 'Find two complete block hashes', ['复用 8 / 12 个 token', 'Reuse 8 / 12 tokens']),
      },
      {
        sg: detail('sgKvCommon', 'KV 分配 · KV allocation', ['共享页 + 待补算尾部', 'Shared pages + uncached tail'], '只为本例未命中的尾部补算并取得新位置', 'Compute the uncached tail and allocate its new locations', ['新尾部写入独立页', 'New tail written to a separate page']),
        vl: detail('v2_blocks', 'BlockTables.compute_slot_mappings', 'position 8 → block ID 31', '用块内偏移定位 KV', 'Apply the in-block offset to locate KV', 'KV slot 124'),
      },
      {
        sg: detail('decode', 'ScheduleBatch.prepare_for_decode', ['当前 Decode 请求', 'Current decode request'], '为新位置准备可写的 KV 槽位', 'Prepare a writable KV slot for the new position', ['历史 KV 继续保留', 'Existing KV stays available']),
        vl: detail('v2_blocks', 'BlockTables.compute_slot_mappings', 'position 12 → block ID 32', '本例追加一整块的首个位置', 'Append at the first position of a new block', 'KV slot 128'),
      },
    ],
    batch: [
      {
        sg: detail('prefill', 'get_new_batch_prefill', 'waiting_queue + running_batch', '检查本轮可进入的 Prefill 请求', 'Check which prefills can enter this step', ['本轮候选', 'Candidates for this step']),
        vl: detail('vlSched', 'Scheduler.schedule', 'running + waiting', '确定每个请求在本轮推进的工作量', 'Assign the work each request advances this step', 'num_scheduled_tokens'),
      },
      {
        sg: detail('admission', 'PrefillAdder', ['候选请求 + token／KV 余量', 'Candidate + free token/KV budget'], '满足容量条件才加入本轮', 'Admit only when capacity constraints are met', ['可入批的 Prefill 或 chunk', 'Admitted prefill or chunk']),
        vl: detail('vlSched', 'Scheduler.schedule', 'token_budget', 'Decode 与 Prefill 共同消耗本轮额度', 'Decode and prefill draw from the same step budget', 'SchedulerOutput.num_scheduled_tokens'),
      },
      {
        sg: detail('forward', 'Scheduler.run_batch', 'ScheduleBatch', '提交本轮 forward，结果留待后续消费', 'Submit this step’s forward; consume results later', ['本轮 forward 结果', 'Forward result for this step']),
        vl: detail('v2_sample', 'V2.sample_tokens', 'ExecuteModelState', '在本轮 forward 之后读取它并采样', 'Read this step’s state after forward and sample', ['采样 token ID', 'Sampled token IDs']),
      },
      {
        sg: detail('commit', 'process_batch_result', ['可消费的批次结果', 'Consumable batch result'], '更新请求输出和完成状态', 'Update request outputs and completion state', 'Req.output_ids'),
        vl: detail('v2_async', 'AsyncOutput / copy_event', ['GPU 已采样的结果', 'GPU sampling result'], 'D2H 复制之后，CPU 才读取这份结果', 'Read this result on the CPU after D2H copy', 'ModelRunnerOutput'),
      },
    ],
    crosscut: [
      {
        sg: detail('sgKvCommon', 'release_kv_cache', ['请求完成', 'Request finishes'], '结算缓存并解除该请求的保护', 'Settle cached entries and drop this request’s protection', ['仍在共享的页继续保留', 'Pages still shared remain protected']),
        vl: detail('v2_free', 'Scheduler._free_request', ['请求完成', 'Request finishes'], '归还请求的块引用，Connector 可延迟释放', 'Return block references; a connector may delay release', ['可复用的完整块仍保留 hash', 'Reusable complete blocks can retain their hashes']),
      },
      {
        sg: detail('retract', 'ScheduleBatch.retract_decode', ['本轮 Decode 需要更多 KV', 'This decode step needs more KV'], '必要时回撤请求并重估预留比例', 'Retract requests when needed and re-estimate the reserve ratio', ['释放后回等待，恢复时补算', 'Requeue after release; recompute on resumption']),
        vl: detail('vlSched', 'Scheduler preemption', 'PREEMPTED', '清理分配并重置已计算进度', 'Release allocations and reset computed progress', 'num_computed_tokens = 0'),
      },
      {
        sg: detail('abort', 'Scheduler.abort_request', 'AbortReq', '按当前所在队列走对应的取消和收尾路径', 'Cancel and clean up according to the request’s current queue', ['停止后续执行与输出', 'Stop subsequent work and output']),
        vl: detail('v2_cleanup', 'V2.finish_requests', 'finished_req_ids', '执行侧消费完成集合后清理本地请求槽位', 'Clean up worker-local request slots after consuming finished IDs', ['请求槽位可再分配', 'Request slot becomes reusable']),
      },
      {
        sg: detail('sgGraph', 'Breakable CUDA Graph backend', ['当前形状与已捕获图', 'Current shape and captured graphs'], '兼容时使用已捕获的执行路径', 'Use a captured execution path when compatible', ['图重放或可用的回退路径', 'Replay or the available fallback']),
        vl: detail('v2_graph', 'CudaGraphManager.dispatch', ['batch descriptor', 'Batch descriptor'], '查找兼容的已捕获候选', 'Find a compatible captured candidate', 'FULL / PIECEWISE / NONE'),
      },
    ],
    disagg: [
      {
        sg: detail('launch', 'Engine', ['已选端点', 'Selected endpoint'], '由该实例承担配置的 Prefill 或 Decode 职责', 'Run the configured prefill or decode role on the selected instance', ['进入该实例的调度路径', 'Enter that instance’s scheduling path']),
        vl: detail('vlCore', 'EngineCore', ['外部路由已选实例', 'Instance selected by external routing'], '所选实例内的 EngineCore 接收请求', 'The selected EngineCore accepts the request', 'EngineCoreRequest'),
      },
      {
        sg: detail('transfer', 'Decode transfer queue', 'KVPoll.Success', '接收侧完成检查后移出传输队列', 'Leave the transfer queue after the receiver completes its checks', ['交给 Decode 侧后续处理', 'Pass to subsequent Decode handling']),
        vl: detail('v2_ready', 'Scheduler remote-KV gate', 'finished_recving', '收到就绪结果后，再参与本轮预算判断', 'Check this step’s budget after a successful receive', ['可候选调度', 'Eligible for scheduling']),
      },
      {
        sg: detail('hierarchy', 'HiCache prefetch', ['下层缓存中的前缀', 'Prefix in a lower cache tier'], '由缓存控制器发起预取', 'Start prefetch through the cache controller', ['等待恢复到可用位置', 'Wait for the prefix to become usable']),
        vl: detail('v2_connector', 'ActiveKVConnector', ['保存或加载的元数据', 'Save or load metadata'], '由所选 Connector 执行后端操作', 'Let the selected connector perform the backend operation', ['按后端报告加载状态', 'Backend-specific load status']),
      },
      {
        sg: detail('launch', 'Engine', ['实例角色与部署配置', 'Instance roles and deployment configuration'], '启动对应角色所需的参与进程', 'Launch the processes required by each configured role', ['分角色的服务端点', 'Serving endpoints by role']),
        vl: detail('v2_choice', 'VllmConfig.use_v2_model_runner', 'external_launcher + PP > 1', '自动选择时检查执行器与 PP 组合', 'Check the executor and PP combination during automatic selection', ['该组合回退到 V1 Runner', 'This combination falls back to the V1 runner']),
      },
    ],
    metrics: [
      {
        sg: detail('sgTM', 'TokenizerManager', ['首批反分词输出', 'First detokenized output'], '前端处理首批输出时记录结束时刻', 'Record the endpoint when the frontend handles the first output', 'first_token_time − created_time'),
        vl: detail('vlTiming', 'IterationStats.update_from_output', ['AsyncLLM 取出首批输出', 'AsyncLLM takes the first output batch'], '使用该批的前端时间戳', 'Use the frontend timestamp for this batch', 'iteration_timestamp − arrival_time'),
      },
      {
        sg: detail('finishTime', 'TokenizerManager', ['前端处理终态输出', 'Frontend handles terminal output'], '用请求创建时间计算完整区间', 'Measure the interval since request creation', 'finished_time − created_time'),
        vl: detail('vlTiming', 'IterationStats.update_from_finished_request', ['最终输出批次', 'Final output batch'], '终态统计使用最终批次的时间戳', 'Use the final batch timestamp for completion statistics', 'iteration_timestamp − arrival_time'),
      },
      {
        sg: detail('sgTiming', 'set_forward_entry_time', 'wait_queue_entry_time', '首次进入 forward 时记录这段等待', 'Record this wait at the first forward entry', 'forward_entry_time − wait_queue_entry_time'),
        vl: detail('vlTiming', 'IterationStats.update_from_events', ['QUEUED → 首次 SCHEDULED', 'QUEUED → first SCHEDULED'], '保存首次调度的时间戳，完成时上报', 'Keep the first schedule timestamp and report on completion', 'scheduled_ts − queued_ts'),
      },
      {
        sg: detail('sgCollector', 'observe_inter_token_latency', 'interval + num_new_tokens', '按新增 token 数均分间隔并记样本', 'Divide the interval among the new tokens and record samples', ''),
        vl: detail('vlTiming', 'IterationStats.update_from_output', 'engine_core_timestamp − last_token_ts', '普通 Decode 的一次核心输出记一个样本', 'Record one sample per ordinary decode core output', ''),
      },
    ],
  };

  function Annotation({ chapter, step, side, en, tokens, ownership, tp, sources }) {
    let annotation = steps[chapter][step][side];
    if (chapter === 'process' && step === 2 && side === 'vl' && ownership && tp === 1) {
      annotation = detail('vlUni', 'UniProcExecutor', 'SchedulerOutput',
        '在 EngineCore 进程内调用 Worker', 'Call Worker inside the EngineCore process', 'Worker → ModelRunner');
    }
    const { anchor, symbol, input, action, result } = annotation;
    const href = anchors[anchor] || sources.find(source => source.key === 'sources.' + anchor).url;
    const samples = chapter === 'metrics' && step === 3 ? window.AtlasBoundaries.metricModel(side, step, tokens) : null;
    const output = samples ? `${samples.count} × ${samples.samples[0]} ms` : select(result, en);
    return h('section', { id: 'ia-source-' + side, className: 'ia-source-step',
      'aria-label': en ? 'Source for this step' : '当前步骤源码' },
      h('div', { className: 'ia-source-step-head' },
        h('span', null, en ? 'In source' : '源码落点'),
        h('a', { href, target: '_blank', rel: 'noreferrer' }, symbol, h('span', { 'aria-hidden': true }, ' ↗'))),
      h('div', { className: 'ia-source-step-flow' },
        h('span', { className: 'ia-source-step-input' }, select(input, en)),
        h('span', { className: 'ia-source-step-arrow', 'aria-hidden': true }, '→'),
        h('strong', { className: 'ia-source-step-result' }, output)),
      h('p', { className: 'ia-source-step-change' }, select(action, en)));
  }
  return { Annotation };
})();
