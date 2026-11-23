(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.ServingLabModel = api;
})(typeof globalThis === 'object' ? globalThis : this, function () {
  'use strict';
  const revision = '72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a';
  const GiB = 2 ** 30;
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  const progress = (t, a, b) => b === a ? Number(t >= b) : clamp((t - a) / (b - a), 0, 1);
  const event = (id, lane, start, end, title, detail, source) =>
    ({ id, lane, start, end, title, detail, source });

  function memoryBudget(options = {}) {
    const o = Object.assign({ baseline: 40, available: 20, fraction: .875, heads: 4,
      layers: 32, keyDim: 128, valueDim: 128, bytes: 2, page: 128,
      tokenLimit: 100000, requests: 12, context: 2048, reserve: 512,
      activationPeak: 2, graphFuture: 1, temporaryPeak: 1 }, options);
    for (const k of ['baseline', 'heads', 'layers', 'keyDim', 'valueDim', 'bytes', 'page']) {
      if (!finite(o[k]) || o[k] <= 0) throw new Error(k + ' 必须为正数');
    }
    if (!finite(o.available) || o.available < 0 || o.available > o.baseline ||
        !finite(o.fraction) || o.fraction < 0 || o.fraction > 1) throw new Error('无效的内存预算');
    for (const k of ['requests', 'context', 'reserve', 'tokenLimit']) {
      if (!Number.isSafeInteger(o[k]) || o[k] < 0) throw new Error(k + ' 必须为非负整数');
    }
    for (const k of ['activationPeak', 'graphFuture', 'temporaryPeak']) {
      if (!finite(o[k]) || o[k] < 0) throw new Error(k + ' 必须为非负数');
    }
    const resident = o.baseline - o.available;
    const slack = o.baseline * (1 - o.fraction);
    const rawBudget = o.available - slack;
    const tokenBytes = o.heads * (o.keyDim + o.valueDim) * o.layers * o.bytes;
    const profiledTokens = Math.floor(Math.floor(Math.max(0, rawBudget) * GiB / tokenBytes) / o.page) * o.page;
    const limited = o.tokenLimit > 0 ? Math.min(profiledTokens, o.tokenLimit) : profiledTokens;
    const capacity = Math.floor(limited / o.page) * o.page;
    const requestedPositions = o.context + o.reserve;
    const perRequestPages = Math.ceil(requestedPositions / o.page);
    const perRequest = perRequestPages * o.page;
    const paddingPerRequest = perRequest - requestedPositions;
    const demand = o.requests * perRequest;
    const executionNeed = o.activationPeak + o.graphFuture + o.temporaryPeak;
    return { options: o, resident, slack, rawBudget, tokenBytes, profiledTokens,
      capacity, pages: capacity / o.page, payloadGiB: capacity * tokenBytes / GiB,
      unusedBudgetGiB: Math.max(0, rawBudget - capacity * tokenBytes / GiB),
      rawTokens: Math.floor(Math.max(0, rawBudget) * GiB / tokenBytes),
      requestedPositions, perRequestPages, paddingPerRequest,
      demandPages: o.requests * perRequestPages,
      usefulPositions: o.requests * requestedPositions,
      paddingTotal: o.requests * paddingPerRequest,
      pageBytes: tokenBytes * o.page,
      userCapBinding: o.tokenLimit > 0 && o.tokenLimit < profiledTokens,
      alignmentLoss: limited - capacity,
      perRequest, demand, fits: rawBudget > 0 && demand <= capacity,
      maxRequests: perRequest > 0 ? Math.floor(capacity / perRequest) : 0,
      deficitTokens: Math.max(0, demand - capacity), executionNeed,
      executionFits: rawBudget > 0 && executionNeed <= slack };
  }

  function pageGrowth(lengths = [7, 8, 3], pageSize = 4) {
    if (!Number.isInteger(pageSize) || pageSize < 1 || lengths.some(n => !Number.isInteger(n) || n < 1)) {
      throw new Error('页大小与已有长度必须为正整数');
    }
    return lengths.map((n, i) => ({ id: 'R' + (i + 1), before: n, after: n + 1,
      oldPages: Math.ceil(n / pageSize), newPages: Math.ceil((n + 1) / pageSize),
      allocated: (Math.ceil((n + 1) / pageSize) - Math.ceil(n / pageSize)) * pageSize }));
  }

  function hicache(scenario = 'l3') {
    const scenarios = ['l3', 'host', 'device-shortage', 'cancel', 'miss'];
    if (!scenarios.includes(scenario)) throw new Error('未知的回载场景');
    let es = [];
    if (scenario !== 'host') {
      es.push(event('exists', 'control', 0, 2, '查询 L3 连续前缀',
        'exists 只返回存在范围，还没有向主机读入 KV。', 'prefetch_from_storage'));
      if (scenario === 'miss') {
        es.push(event('miss', 'control', 2, 3, '未命中，保留设备原前缀',
          '不分配本次 L2 目标，也不提交 H2D；请求按剩余输入计算。', '_drain_storage_control_queues_impl'));
        es.push(event('recompute', 'forward', 3, 9, '剩余输入进入 Prefill',
          '这是重新计算路径的教学占位，不预测真实重算耗时。', 'init_load_back'));
        return { scenario, events: es, duration: 10 };
      }
      es.push(event('host-alloc', 'control', 2, 3, '为读取分配 L2 目标',
        '查询结果经调度侧核对阈值与容量，分配主机槽位后才能交给 IO。', '_drain_storage_control_queues_impl'));
      es.push(event('storage-read', 'storage', 3, 7, 'L3 → L2 后台读取',
        '目标缓冲区由后台 IO 持有，尚未交接的部分不能提前复用。', '_drain_storage_control_queues_impl'));
      if (scenario === 'cancel') {
        es.push(event('abort', 'control', 5, 5.5, 'R1 取消，标记终止',
          '本例尚无已交接前缀。请求退出不代表后台已经停止使用 L2 目标。', 'release_aborted_request'));
        es.push(event('final-ack', 'control', 7, 8, 'L3 最终 ACK 与目标归还',
          '后台收尾后交回剩余所有权，将尚未采用的目标放入释放队列；本例不再把结果交给已取消的 R1。', '_drain_storage_control_queues_impl'));
        es.push(event('release-target', 'control', 8, 9, '释放队列实际归还 L2 目标', '从释放队列取出本次目标，再调用主机池 free；最终 ACK 已消费与池容量已归还分开。', '_drain_storage_control_queues_impl'));
        return { scenario, events: es, duration: 10 };
      }
      es.push(event('host-adopt', 'control', 7, 8, 'L3 交接 ACK：采用到主机树',
        '只完成 L3 → L2；有 host 副本仍不等于 GPU 可读。', '_handle_prefetch_result'));
    } else {
      es.push(event('host-match', 'control', 0, 1, '匹配 L2 已有副本',
        '本例已有合法主机前缀，不再查询或读取 L3。', 'init_load_back'));
    }
    const a = scenario === 'host' ? 1 : 8;
    es.push(event('device-alloc', 'control', a, a + 1,
      scenario === 'device-shortage' ? 'L1 容量不足，回载未提交' : '分配 L1 地址并建立保护',
      scenario === 'device-shortage' ? '淘汰后仍不足，解除本次临时保护；不把 host hit 计成可用设备前缀。' :
        '先保护 host anchor 与设备路径，再分配 L1；成功入队后返回索引，尚未完成复制。', 'load_back'));
    if (scenario === 'device-shortage') {
      es.push(event('reschedule', 'control', a + 1, a + 2, '保留原前缀，等待重新调度',
        '仍须通过普通准入。本图停在重新调度，不在容量不足时额外画出一次 GPU 计算。', 'init_load_back'));
      return { scenario, events: es, duration: a + 3 };
    }
    const s = a + 1;
    es.push({ ...event('wait-layer-0', 'forward', s, s + 3, '等第 0 层复制 event',
      'Forward stream 尚不能读取本层；H2D stream 此时继续复制。', 'wait_until'), wait: true });
    for (let layer = 0; layer < 3; layer++) {
      es.push(event('copy-' + layer, 'h2d', s + layer * 3, s + (layer + 1) * 3,
        '复制第 ' + layer + ' 层 KV', 'H2D stream 写入本层目标，结束时记录对应层 event。', 'start_loading'));
      es.push(event('attention-' + layer, 'forward', s + 3 + layer * 4, s + 7 + layer * 4,
        '第 ' + layer + ' 层读取', 'Forward stream 先等本层复制 event；无需先等整批 ACK 被 CPU 消费。', 'wait_until'));
    }
    es.push(event('load-ack', 'control', s + 10, s + 11, '消费 H2D 整批完成 ACK',
      '解除本次主机与设备的传输保护。请求仍持有前缀；主机副本也不在此处自动删除。', 'loading_check'));
    return { scenario, events: es, duration: s + 16 };
  }

  function hicacheState(plan, t) {
    const by = id => plan.events.find(e => e.id === id);
    const done = id => !!by(id) && t >= by(id).end;
    const active = id => !!by(id) && t >= by(id).start && t < by(id).end;
    const canceled = done('abort');
    const hostAllocated = plan.scenario === 'host' || done('host-alloc');
    const hostValid = plan.scenario === 'host' || done('host-adopt');
    const deviceAllocated = done('device-alloc') && plan.scenario !== 'device-shortage';
    const ack = done('load-ack');
    const loadPreparing = active('device-alloc');
    const hostTransferPinned = loadPreparing || deviceAllocated && !ack;
    const allLayersReady = [0, 1, 2].every(i => done('copy-' + i));
    return { canceled, hostAllocated, hostValid, deviceAllocated, ack,
      storageChecked: done('exists'),
      storageHitKnown: done('exists') && plan.scenario !== 'miss',
      storageReadDone: done('storage-read'),
      storageAck: done('host-adopt') || done('final-ack'),
      allLayersReady, ackPending: allLayersReady && !ack,
      loadPreparing, hostTransferPinned,
      hostReleased: done('release-target'),
      releaseQueued: done('final-ack') && !done('release-target'),
      ioOwns: hostAllocated && plan.scenario !== 'host' && !done('host-adopt') && !done('final-ack'),
      hostPinned: hostTransferPinned,
      devicePathPinned: hostTransferPinned,
      deviceTransferPinned: deviceAllocated && !ack,
      requestPinned: deviceAllocated && !canceled,
      readyLayers: [0, 1, 2].filter(i => done('copy-' + i)),
      readLayers: [0, 1, 2].filter(i => done('attention-' + i)),
      activeLayers: [0, 1, 2].filter(i => active('attention-' + i)),
      storageProgress: by('storage-read') ? progress(t, by('storage-read').start, by('storage-read').end) : 0,
      current: plan.events.filter(e => t >= e.start && t < e.end),
      done: plan.events.filter(e => t >= e.end).map(e => e.id) };
  }

  function deployment(scenario = 'normal') {
    if (!['normal', 'probe', 'oom', 'cancel'].includes(scenario)) throw new Error('未知的部署场景');
    const es = [
      event('identity', 'client', 0, 1, '固定环境与模型身份', '保存源码 commit、权重/tokenizer revision、设备和依赖清单。', 'cli'),
      event('weights', 'gpu', 1, 4, '模型装载', 'HTTP 端口与模型准备各有进度，此阶段不把探活成功当作请求成功。', 'launch'),
      event('kv', 'gpu', 4, 6, '计算 KV 预算并建立池', '权重之外还需要 KV、执行临时空间与相应 buffers。', '_profile_available_bytes')
    ];
    if (scenario === 'oom') {
      es.push(event('boot-failed', 'http', 6, 7, '初始化失败，尚未 ready',
        '保存最早异常和设备账本；调整配置后是一次新启动，不在原请求上伪造继续。', '_profile_available_bytes'));
      return { scenario, events: es, duration: 8 };
    }
    if (scenario === 'probe') es.push(event('early-probe', 'client', 2, 3,
      '过早发送探测', '此时探测失败不能单独证明最终启动会失败；核对进程、日志与后续就绪。', 'http'));
    es.push(event('warmup', 'gpu', 6, 8, '执行准备与预热', '本课最小配置关闭 Graph 和 Overlap，先减少路径变化。', 'launch'));
    es.push(event('ready', 'http', 8, 9, '检查身份与就绪', '/model_info 核对服务；健康探测仍不能替代一次生成。', 'http'));
    es.push(event('post', 'client', 9, 10, '发送 R1 到 /generate', '保存原始请求、HTTP 状态和完整响应或 SSE 事件。', 'http'));
    es.push(event('submit-prefill', 'scheduler', 10, 10.2, '准备本轮输入并提交', '本例张量提交和设备执行分开；提交返回不代表采样完成。', 'http'));
    es.push(event('prefill', 'gpu', 10.2, 12, 'R1 处理输入并采样 t1', 't1 是教学身份；本页不预设真实模型会输出什么文字。', 'http'));
    es.push(event('text', 'http', 12, 13, '首个非空文本回程', '记录客户端观察时刻；设备采样与客户端可见不是同一事件。', 'http'));
    es.push(event('submit-decode', 'scheduler', 12, 12.1, '提交后续 Decode', '与首个结果的 CPU 回程可重叠，不添加整条回程完成的全局屏障。', 'http'));
    es.push(event('decode', 'gpu', 12.1, 14, '后续 Decode 在途', '取消通知到来时，在途设备工作仍有独立完成条件。', 'http'));
    if (scenario === 'cancel') {
      es.push(event('abort', 'client', 13, 13.5, '取消 R1', '客户端停止等待不能直接释放仍被执行读取的资源。', 'abort'));
      es.push(event('cancel-ack', 'client', 13.5, 13.7, '取消通知返回', '通知已被接收，不等于设备完成或资源已经归还。', 'abort'));
      es.push(event('retire', 'http', 14.5, 15, '核对取消收尾', '观察取消路径与资源退役，已经写出的文件保留实际结束原因。', 'abort'));
    } else {
      es.push(event('finish', 'http', 14.5, 15, '核对完成与计数', '检查 finish_reason、实际 prompt/completion token 数和服务端错误。', 'http'));
    }
    es.push(event('terminal', 'http', 14, 14.5, scenario === 'cancel' ? '核对取消流尾与结束原因' : '核对完整响应与结束原因',
      '后续输出回程在本例合并；核对 finish_reason、计数和完整流尾。首个文本不代表响应完整，流尾也不单独证明 GPU 资源已归还。', scenario === 'cancel' ? 'abort' : 'http'));
    // Teaching intervals resolve the handoffs inside a phase; not measured
    // packets or kernel timing. Both projections use these exact intervals.
    const transfers = [
      { id: 'post-http', path: 'post', start: 9, end: 9.3, payload: 'R1 HTTP 请求', kind: 'data' },
      { id: 'post-ipc', path: 'ipc', start: 9.6, end: 10, payload: 'R1 token IDs / 元数据', kind: 'data' },
      { id: 'submit-prefill', path: 'submit', start: 10, end: 10.2, payload: '本轮输入张量', kind: 'data' },
      { id: 'submit-decode', path: 'submit', start: 12, end: 12.1, payload: 'Decode 输入张量', kind: 'data' },
      { id: 'return-id', path: 'back', start: 12, end: 12.15, payload: 't1 输出 ID', kind: 'data' },
      { id: 'return-detok', path: 'detok', start: 12.15, end: 12.35, payload: 't1 批次输出 ID', kind: 'data' },
      { id: 'return-text', path: 'text', start: 12.55, end: 12.75, payload: 't1 文本', kind: 'data' },
      { id: 'return-visible', path: 'visible', start: 12.75, end: 13, payload: 't1 客户端文本', kind: 'data' }
    ];
    es.push(event('tokenize', 'http', 9.3, 9.6, '准备 token IDs', 'HTTP 前端进程准备本次输入，再经本机 IPC 交给 Scheduler。', 'http'));
    es.push(event('detokenize', 'detok', 12.35, 12.55, '输出 ID 转文本', 'CPU 处理和下一轮设备 Decode 可以同时推进。', 'http'));
    if (scenario === 'cancel') transfers.push(
      { id: 'abort-http', path: 'post', start: 13, end: 13.25, payload: 'R1 取消控制', kind: 'control' },
      { id: 'abort-ipc', path: 'ipc', start: 13.25, end: 13.5, payload: 'R1 取消控制', kind: 'control' });
    es.push(...transfers.filter(e => e.id.startsWith('return-')).map(e => event(e.id, 'return', e.start, e.end, e.payload, '结果交接区间独立于 CPU 处理与后续设备计算；脉冲不决定完成。', 'http')));
    return { scenario, events: es, transfers, duration: 16 };
  }

  function percentile(values, p = .95) {
    const a = values.filter(finite).slice().sort((x, y) => x - y);
    if (!a.length) return null;
    const i = (a.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i);
    return a[lo] + (a[hi] - a[lo]) * (i - lo);
  }
  function metrics(rows, duration, sla = { ttft: 6, tpot: 1.5 }) {
    if (!Array.isArray(rows) || !finite(duration) || duration <= 0) throw new Error('需要请求数组和正的统一观察窗口');
    const good = rows.filter(r => r.success === true);
    const ttfts = good.filter(r => finite(r.first) && finite(r.start) && r.first >= r.start).map(r => r.first - r.start);
    const tpots = good.filter(r => r.out > 1 && finite(r.first) && finite(r.end) && r.end >= r.first)
      .map(r => (r.end - r.first) / (r.out - 1));
    const passing = good.filter(r => finite(r.first) && finite(r.start) && finite(r.out) &&
      r.first - r.start <= sla.ttft && (r.out === 1 ||
        (r.out > 1 && finite(r.end) && (r.end - r.first) / (r.out - 1) <= sla.tpot)));
    return { requests: rows.length, completed: good.length, failed: rows.length - good.length,
      ttftSamples: ttfts.length, tpotSamples: tpots.length,
      p95TTFT: percentile(ttfts), p95TPOT: percentile(tpots),
      requestThroughput: good.length / duration, goodput: passing.length / duration,
      goodputCount: passing.length,
      outputThroughput: good.reduce((a, r) => a + (finite(r.out) ? r.out : 0), 0) / duration,
      knownOutputs: good.filter(r => finite(r.out)).length,
      unknownEnd: good.filter(r => !finite(r.end)).length };
  }

  function workload(options = {}) {
    const o = Object.assign({ interval: 1, concurrency: 2, failure: false, warm: false }, options);
    const permits = Array(Math.max(1, Math.min(8, Math.floor(o.concurrency)))).fill(0);
    let serverFree = 0;
    const rows = [];
    [2, 2, 6, 2, 4, 2].forEach((prefill, i) => {
      const scheduled = i * o.interval;
      const idx = permits.indexOf(Math.min(...permits));
      const start = Math.max(scheduled, permits[idx]);
      const work = Math.max(start, serverFree);
      const prefillWork = o.warm && i > 0 ? Math.max(1, prefill - 1) : prefill;
      const first = work + prefillWork;
      const end = first + 2.5;
      const success = !(o.failure && i === 2);
      rows.push({ id: 'R' + (i + 1), scheduled, permit: start, start, work, deviceFirst: first, first: success ? first : null,
        end, out: success ? 3 : 0, input: 8, success,
        error: success ? '' : '教学故障：保留相同工作区间，只改变成功样本集合' });
      permits[idx] = end;
      serverFree = end;
    });
    const duration = Math.max(...rows.map(r => r.end)) + .5;
    return { rows, duration, unit: 'u', evidence: 'synthetic', options: o };
  }

  function union(intervals) {
    const sorted = intervals.filter(x => finite(x[0]) && finite(x[1]) && x[1] > x[0]).slice().sort((a, b) => a[0] - b[0]);
    const out = [];
    for (const [s, e] of sorted) {
      const prev = out[out.length - 1];
      if (prev && s <= prev[1]) prev[1] = Math.max(prev[1], e);
      else out.push([s, e]);
    }
    return out;
  }
  const unionDuration = intervals => union(intervals).reduce((n, x) => n + x[1] - x[0], 0);
  function parseTrace(raw) {
    const input = Array.isArray(raw) ? raw : raw && raw.traceEvents;
    if (!Array.isArray(input)) throw new Error('需要 Chrome Trace 数组或含 traceEvents 的 JSON');
    const processNames = {}, threadNames = {}, spans = [], stacks = new Map();
    let dropped = 0, unsupported = 0, unmatched = 0;
    for (const e of input) {
      if (!e || typeof e !== 'object') continue;
      if (e.ph !== 'M') continue;
      if (e.name === 'process_name') processNames[String(e.pid)] = e.args && e.args.name;
      if (e.name === 'thread_name') threadNames[e.pid + ':' + e.tid] = e.args && e.args.name;
    }
    function add(e, dur) {
      if (!finite(e.ts) || !finite(dur) || dur < 0) { dropped++; return; }
      if (dur === 0) return;
      const args = e.args || {}, cat = String(e.cat || '');
      // A GPU-looking process name may also contain capture/CPU ranges. Only
      // recognized device categories contribute to the device interval union.
      const gpu = /(^|,)(kernel|gpu_memcpy|gpu_memset|memcpy|memset)(,|$)/i.test(cat);
      const lane = String(e.pid) + ':' + String(e.tid);
      spans.push({ name: String(e.name || '(未命名)'), start: e.ts, end: e.ts + dur,
        duration: dur, pid: e.pid, tid: e.tid, lane, cat, gpu, args,
        cpu: /^(cpu_op|user_annotation|Operator|Runtime|cuda_runtime|cuda_driver)$/i.test(cat),
        laneName: (processNames[String(e.pid)] || 'PID ' + e.pid) + ' / ' +
          (threadNames[lane] || 'TID ' + e.tid),
        correlation: args.correlation ?? args['correlation id'] ?? args['Correlation ID'] ?? null });
    }
    const ordered = input.filter(e => e && typeof e === 'object').slice()
      .sort((a, b) => (finite(a.ts) ? a.ts : 0) - (finite(b.ts) ? b.ts : 0));
    for (const e of ordered) {
      if (e.ph === 'M') continue;
      if (e.ph === 'X') { add(e, e.dur); continue; }
      if (e.ph === 'B') {
        const key = e.pid + ':' + e.tid;
        if (!stacks.has(key)) stacks.set(key, []);
        stacks.get(key).push(e);
      } else if (e.ph === 'E') {
        const a = stacks.get(e.pid + ':' + e.tid);
        const b = a && a.pop();
        if (b && finite(e.ts)) add(b, e.ts - b.ts); else unmatched++;
      } else unsupported++;
    }
    for (const a of stacks.values()) unmatched += a.length;
    if (!spans.length) throw new Error('没有可读的完整区间事件（支持 X 与同线程 B/E）');
    spans.sort((a, b) => a.start - b.start);
    const origin = spans[0].start, end = spans.reduce((n, e) => Math.max(n, e.end), origin);
    spans.forEach((e, i) => { e.id = i; e.start -= origin; e.end -= origin; });
    return { spans, duration: end - origin, origin, processNames, threadNames,
      diagnostics: { dropped, unsupported, unmatched }, unit: 'µs',
      metadata: Array.isArray(raw) ? {} : { deviceProperties: raw.deviceProperties || [], schemaVersion: raw.schemaVersion } };
  }
  function traceStats(trace, start = 0, end = trace.duration) {
    if (!finite(start) || !finite(end) || end <= start) throw new Error('观察窗口必须为正');
    const spans = trace.spans.filter(e => e.end > start && e.start < end);
    const gpu = spans.filter(e => e.gpu);
    const clip = e => [Math.max(start, e.start), Math.min(end, e.end)];
    const active = unionDuration(gpu.map(clip));
    const groups = new Map();
    for (const e of gpu) {
      const g = groups.get(e.name) || { name: e.name, total: 0, count: 0 };
      const c = clip(e); g.total += c[1] - c[0]; g.count++; groups.set(e.name, g);
    }
    return { spans, gpuCount: gpu.length, observedGpuActive: active,
      observedGpuEmpty: end - start - active, window: end - start,
      summedGpuDuration: gpu.reduce((n, e) => { const c = clip(e); return n + c[1] - c[0]; }, 0),
      top: [...groups.values()].sort((a, b) => b.total - a.total),
      lanes: [...new Set(spans.map(e => e.lane))] };
  }
  function largestGpuGap(trace) {
    const intervals = union(trace.spans.filter(e => e.gpu).map(e => [e.start, e.end]));
    if (!intervals.length) return null;
    let previous = 0, gap = [0, 0];
    for (const [start, end] of [...intervals, [trace.duration, trace.duration]]) {
      if (start - previous > gap[1] - gap[0]) gap = [previous, start];
      previous = Math.max(previous, end);
    }
    return gap[1] > gap[0] ? gap : null;
  }
  function traceFocusWindow(trace, start, end, choices = [5, 10, 25, 50, 100]) {
    if (!(trace.duration > 0) || !finite(start) || !finite(end) || end < start) throw new Error('无效的事件窗口');
    start = clamp(start, 0, trace.duration); end = clamp(end, start, trace.duration);
    const needed = Math.max((end - start) * 1.2, trace.duration * .02);
    const percent = choices.find(p => trace.duration * p / 100 >= needed) || 100;
    const span = trace.duration * percent / 100;
    const left = clamp(start - (span - (end - start)) / 2, 0, trace.duration - span);
    return { percent, startPercent: percent === 100 ? 0 : left / (trace.duration - span) * 100,
      start: left, end: left + span };
  }
  function launchCandidates(trace, gpuEvent) {
    if (gpuEvent.correlation === null) return [];
    return trace.spans.filter(e => /^(cuda_runtime|cuda_driver|Runtime)$/i.test(e.cat) &&
      String(e.correlation) === String(gpuEvent.correlation) && e.start <= gpuEvent.start);
  }
  return { revision, GiB, clamp, progress, memoryBudget, pageGrowth, hicache, hicacheState,
    deployment, percentile, metrics, workload, union, unionDuration, parseTrace, traceStats, launchCandidates, largestGpuGap, traceFocusWindow };
});
