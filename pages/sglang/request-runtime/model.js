/* Teaching snapshots for one ordinary request, not an SGLang execution simulator. */
(function (root) {
  'use strict';
  const revision = '279339f113b79af84f27fd3ac92d0a13bd3f4cbd';
  const sources = {
    startup: ['entrypoints/engine.py', 1051, '_launch_subprocesses'],
    input: ['managers/tokenizer_manager.py', 797, 'generate_request'],
    dispatch: ['managers/tokenizer_manager.py', 1592, '_send_one_request'],
    receive: ['managers/scheduler.py', 2740, 'handle_generate_request'],
    queue: ['managers/scheduler.py', 3233, '_add_request_to_queue'],
    loop: ['managers/scheduler.py', 1907, 'event_loop_normal'],
    worker: ['managers/tp_worker.py', 593, 'forward_batch_generation'],
    output: ['managers/detokenizer_manager.py', 443, 'handle_batch_token_id_out'],
    front: ['managers/tokenizer_manager.py', 2255, '_handle_batch_output'],
    abortfront: ['managers/tokenizer_manager.py', 3281, '_handle_abort_req'],
    cancel: ['managers/tokenizer_manager.py', 2024, 'abort_request'],
    abort: ['managers/scheduler.py', 5257, 'abort_request'],
    result: ['managers/scheduler_components/batch_result_processor.py', 920, 'process_batch_result_decode'],
    prefillresult: ['managers/scheduler_components/batch_result_processor.py', 257, 'process_batch_result_prefill'],
    release: ['mem_cache/common.py', 254, 'release_kv_cache'],
    cache: ['mem_cache/radix_cache.py', 479, 'cache_finished_req'],
  };
  function sourceURL(key) {
    const [file, line] = sources[key];
    return `https://github.com/sgl-project/sglang/blob/${revision}/python/sglang/srt/${file}#L${line}`;
  }
  function build({scenario = 'normal', outputs = 3, cache = true} = {}) {
    if (!['normal', 'queued-abort', 'running-abort', 'invalid'].includes(scenario)) throw new Error('Unknown scenario');
    if (!Number.isInteger(outputs) || outputs < 1 || outputs > 5) throw new Error('outputs must be 1..5');
    // Cancellation examples have a fixed interruption point after the first output.
    if (scenario !== 'normal') outputs = 3;
    const frames = [];
    let state = {front: '尚未登记', scheduler: '尚无 Req', batch: '无', waiting: false,
      slot: false, kv: 0, cached: 0, sampled: 0, visible: 0, dispatched: false,
      abortSent: false, toFinish: false, finished: false, frontFinished: false, forwards: 0};
    function add(id, title, actor, payload, explanation, source, changes = {}) {
      state = {...state, ...changes};
      frames.push({...state, id, title, actor, payload, explanation, source});
    }
    add('input', '接入：登记前端状态', 'front', 'GenerateReqInput · rid=R1',
      '前端为 R1 建立 ReqState，记住返回连接和结果等待者。此时 Scheduler 还没有 Req，GPU 也没有计算。', 'input', {front: 'ReqState · 等待预处理'});
    if (scenario === 'invalid') {
      add('invalid', '输入验证失败', 'front', '异常 → 清理未派发的 ReqState',
        '例如输入长度不合法；generate_request 的失败清理区分尚未派发和已经派发。本分支没有进入 Scheduler，无需假想一个 GPU 取消过程。', 'input', {front: '未派发状态已清理', frontFinished: true});
      return frames;
    }
    add('tokenize', '分词：文本变成 ID', 'front', 'TokenizedGenerateReqInput · p1 p2 p3 p4',
      '用 4 个输入位置说明机制，p1…p4 不是实际分词结果。消息里是 ID 和采样配置，还不是模型执行 tensor 或 KV。', 'input', {front: 'ReqState · 输入已准备'});
    add('dispatch', '发送：跨过 IPC 边界', 'ipc', 'TokenizerManager → Scheduler · tokenized request',
      '前端派发后记下 dispatched。这个字段不构成后端已接纳或 GPU 已开始的回执；下一步才画 Scheduler 的接收处理。', 'dispatch', {front: 'ReqState · 等待结果', dispatched: true});
    add('queued', '排队：建立调度侧 Req', 'scheduler', 'Req(R1) → waiting_queue',
      '同一个 rid 关联两个进程内的不同对象。普通路径通过验证后进入等待队列；进入队列还没有获得本例的执行槽位与 KV。', 'queue', {scheduler: 'Req · 等待 Prefill', waiting: true});
    if (scenario === 'queued-abort') {
      add('cancel-sent', '前端发出取消', 'ipc', 'AbortReq(R1) → Scheduler',
        'abort_sent 只记录取消已派发。消息尚未被后端处理时，队列里的 R1 仍然存在。', 'cancel', {abortSent: true});
      add('abort-queued', '后端移除等待请求', 'scheduler', 'waiting_queue 移除 R1 → 取消回告',
        '本例是从未执行的普通请求：直接移出队列，并回告前端清理状态。PD Decode 等已经预分配资源的等待队列不能照搬此图。', 'abort', {scheduler: '已移出等待队列', waiting: false});
      add('abort-front', '前端收到取消回告', 'front', '取消回告 → 前端状态收尾',
        '后端移除与前端收尾是两个边界。本例从头到尾没有一次模型 forward，也没有分配 KV。连接若已断开，不代表客户端还能收到回告。', 'abortfront', {front: '取消完成 · 状态已清理', frontFinished: true});
      return frames;
    }
    add('prefill', 'Prefill：读取 4 个输入位置', 'worker', 'ScheduleBatch → ForwardBatch → y1',
      'Scheduler 准入并准备 batch，Worker/ModelRunner 执行模型和采样。写入 p1…p4 的 KV，采样 y1；y1 尚未作为输入，因而还没有 y1 的 KV。', 'worker',
      {scheduler: 'Req · 结果待处理', waiting: false, batch: 'B0 · EXTEND · p1…p4', slot: true, kv: 4, sampled: 1, forwards: 1});
    if (outputs > 1) {
      add('first-output', '输出回程：首段文字可见', 'output', 'BatchTokenIDOutput → BatchStrOutput → 前端 → 客户端',
        '反分词把 ID 变成文本，前端按 rid 关联响应。本教学序列把回程完整走完后再进入 Decode；真实流式输出与后续计算可能交错，也不保证一 token 一个网络块。', 'output', {scheduler: 'Req · 继续 Decode', visible: 1});
    }
    if (scenario === 'running-abort') {
      add('cancel-sent', '运行中：前端发出取消', 'ipc', 'AbortReq(R1) · GPU 资源仍被持有',
        '取消是控制消息，不是从远处直接删除 GPU tensor。R1 的请求槽位和 KV 此刻仍然有效。', 'cancel', {abortSent: true});
      add('abort-pending', '后端登记待结束标记', 'scheduler', 'Req.to_finish = FINISH_ABORT',
        '普通在途请求设置 to_finish，后续结果处理消费它。不要把它等同于 finished_reason 已记录，更不能在这里把 KV 全部抹掉。', 'abort', {scheduler: 'Req · 待处理取消', toFinish: true});
      add('abort-result', '经过执行与结果收尾边界', 'worker', 'Decode 结果 → update_finish_state',
        '本普通非重叠示例再经过一轮 Decode，结果处理消费待结束标记。它不是抢占 GPU 的演示，也不规定其他执行模式还能多算几轮。额外采样不代表客户端必须看见额外文字。', 'result',
        {scheduler: 'Req · finished_reason=abort', batch: 'B1 · DECODE · y1', kv: 5, sampled: 2, forwards: 2, toFinish: false, finished: true});
    } else {
      for (let k = 2; k <= outputs; k++) {
        add(`decode-${k}`, `Decode：输入 y${k - 1}，预测 y${k}`, 'worker', `B${k - 1} · DECODE · y${k - 1} → y${k}`,
          `同一个 Req 进入新一轮执行视图。现在有 ${4 + k - 1} 个位置的 KV、${k} 个采样输出；最后的 y${k} 还没有经过下一轮前向。`, 'loop',
          {scheduler: 'Req · 结果待处理', batch: `B${k - 1} · DECODE · y${k - 1}`, kv: 4 + k - 1, sampled: k, forwards: k});
        if (k < outputs) add(`output-${k}`, `回传第 ${k} 段输出`, 'output', '反分词 → 前端按 rid 返回',
          '请求身份持续存在，执行 batch 按轮次变化。这里按可读性画成分步，不承诺真实回程耗时或消息粒度。', 'output', {scheduler: 'Req · 继续 Decode', visible: k});
      }
      add('finish', '后端记录结束原因', 'scheduler', `达到 ${outputs} 个输出的教学上限`,
        '普通示例用输出长度结束；实际还可能命中停止 token/字符串。Req.finished() 检查 finished_reason；这个值不是客户端收包证明。', outputs === 1 ? 'prefillresult' : 'result', {scheduler: 'Req · finished_reason=length', finished: true});
    }
    add('release', '资源交接：请求行与缓存分开', 'memory', 'release_kv_cache → cache_finished_req → 释放请求行',
      cache ? '示例采用 page_size=1、启用可插入的普通 Radix Cache，无其他持有者。有效 KV 交给前缀缓存，请求槽位释放；缓存仍占内存，之后才可能被驱逐。' : '示例关闭前缀缓存，无共享前缀和额外持有者。请求的有效 KV 归还分配器，请求槽位也释放。释放并不意味着归零擦写显存。', 'release',
      {scheduler: 'Req · 已完成资源收尾', slot: false, batch: '无后续执行', cached: cache ? state.kv : 0, kv: 0});
    add('final', '回程完成：前端结束等待', 'front', '结束输出 → ReqState.finished → 移除 rid 状态',
      scenario === 'normal' ? '前端处理带结束原因的输出，清理状态并通知响应等待者。此图假设连接正常，客户端随后得到最终响应；前端收尾本身不是网络 ACK。' : '图中保留取消前已经送出的首段文字；额外执行结果的具体返回文本不在此控制流模型中模拟。前端处理取消结束后收尾；断开的客户端不会因此重新收到输出。', 'front',
      {front: '已结束 · 状态已清理', frontFinished: true, visible: scenario === 'normal' ? outputs : state.visible});
    return frames;
  }
  const api = {build, revision, sources, sourceURL};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.RequestRuntimeModel = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
