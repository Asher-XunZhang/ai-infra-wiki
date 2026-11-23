(function () {
  'use strict';
  const root = document.querySelector('[data-serving-lab]');
  if (!root) return;
  const M = window.ServingLabModel, E = window.ServingLabEvidence;
  const kind = root.dataset.servingLab;
  const $ = s => root.querySelector(s);
  const svg = $('#sl-svg'), seek = $('[name=seek]'), error = $('#sl-error');
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (n, digits = 2) => Number.isFinite(n) ? n.toLocaleString('en-US', { maximumFractionDigits: digits }) : '无有效样本';
  const palette = { green: 'var(--accent)', blue: 'var(--sl-blue)', gold: 'var(--sl-gold)', purple: 'var(--sl-purple)', red: 'var(--sl-red)', muted: 'var(--muted)' };
  let t = 0, duration = 1, running = false, raf = 0, last = null, plan = null, selected = null;
  let imported = null, trace = null, traceLabel = '', traceSelected = null, loadSequence = 0, traceGeometry = '';
  let depth = null, drawingPrefix = 'sl', restorePending = true, restoring = false;
  let initialURL = new URL(location.href);
  const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
  const markupCache = new WeakMap();
  function putHTML(el, value) {
    if (markupCache.get(el) !== value) { el.innerHTML = value; markupCache.set(el, value); }
  }
  const numbers = new Set(['available', 'fraction', 'heads', 'bytes', 'page', 'tokenLimit', 'requests',
    'context', 'reserve', 'activationPeak', 'graphFuture', 'temporaryPeak',
    'interval', 'concurrency', 'slaTTFT', 'slaTPOT', 'windowSpan', 'windowStart']);
  function options() {
    const out = {};
    for (const el of root.querySelectorAll('[name]')) {
      if (el.name === 'speed' || el.name === 'seek') continue;
      out[el.name] = el.type === 'checkbox' ? el.checked : numbers.has(el.name) ? Number(el.value) : el.value;
    }
    return out;
  }
  function tx(x, y, text, cls = '', extra = '') {
    return '<text x="' + x + '" y="' + y + '" class="' + cls + '" ' + extra + '>' + esc(text) + '</text>';
  }
  function rect(x, y, w, h, fill, extra = '') {
    return '<rect x="' + x + '" y="' + y + '" width="' + Math.max(0, w) + '" height="' + h +
      '" rx="5" fill="' + fill + '" ' + extra + '/>';
  }
  function defs() {
    return '<defs>' + Object.keys(palette).map(k => '<marker id="sl-arrow-' + k +
      '" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
      '<path d="M0 0L10 5L0 10Z" fill="' + palette[k] + '"/></marker>').join('') +
      '<pattern id="sl-hatch" width="8" height="8" patternUnits="userSpaceOnUse"><path d="M-2 2L2 -2M0 8L8 0M6 10L10 6" stroke="var(--muted)" opacity=".4"/></pattern></defs>';
  }
  function arrow(points, color = 'green', dashed = false) {
    return '<path d="' + points.map((p, i) => (i ? 'L' : 'M') + p.join(' ')).join(' ') +
      '" fill="none" stroke="' + palette[color] + '" stroke-width="2" marker-end="url(#' + drawingPrefix + '-arrow-' +
      color + ')"' + (dashed ? ' stroke-dasharray="5 4"' : '') + '/>';
  }
  function flow(points, event, label, color = 'purple', type = 'data') {
    const active = !!event && t >= event.start && t < event.end;
    const phase = active && !reduced() ? -(t - event.start) * 30 : 0;
    const path = points.map((p, i) => (i ? 'L' : 'M') + p.join(' ')).join(' ');
    return '<g data-flow="' + esc(event ? event.id : label) + '" data-flow-kind="' + type +
      '" data-active="' + active + '" data-time="' + t + '"><title>' + esc(label + (active ? ' · 传递中' : ' · 当前未传递')) + '</title>' +
      '<path d="' + path + '" fill="none" stroke="' + palette[color] + '" stroke-width="' + (active ? 3 : 1.8) +
      '" marker-end="url(#' + drawingPrefix + '-arrow-' + color + ')" opacity="' + (active ? 1 : .45) + '"' +
      (active && !reduced() ? ' stroke-dasharray="7 6" stroke-dashoffset="' + phase.toFixed(3) + '"' : type === 'control' ? ' stroke-dasharray="3 5"' : '') + '/></g>';
  }
  function begin(title, description, height) {
    drawingPrefix = 'sl';
    svg.setAttribute('viewBox', '0 0 1000 ' + height);
    return '<title id="sl-svg-title">' + esc(title) + '</title><desc id="sl-svg-desc">' +
      esc(description) + '</desc>' + defs();
  }
  function result(label, value) {
    return '<div class="sl-result"><small>' + esc(label) + '</small><strong>' + esc(value) + '</strong></div>';
  }
  function source(key) {
    const a = $('#sl-event-source'), e = E.sources[key];
    a.href = e ? e.url : '#evidence';
    a.textContent = e ? e.symbol + ' · 固定源码 ' + E.revision.slice(0, 10) : '查看固定源码与证据范围';
  }
  function stage(title, detail, key) {
    putHTML($('#sl-stage'), '<h3>' + esc(title) + '</h3><p>' + esc(detail) + '</p>');
    $('#sl-stage').setAttribute('aria-live', running ? 'off' : 'polite');
    source(key);
  }
  function tracks(events, lanes, top, time, end) {
    const x0 = 200, w = 760, row = 43, x = v => x0 + w * v / end;
    let s = '';
    for (let i = 0; i <= 4; i++) {
      const xx = x0 + w * i / 4;
      s += '<line class="sl-grid" x1="' + xx + '" y1="' + (top - 8) + '" x2="' + xx + '" y2="' +
        (top + row * lanes.length) + '"/>' + tx(xx, top - 16, fmt(end * i / 4, 1) + ' u', 'sl-small', 'text-anchor="middle"');
    }
    lanes.forEach((lane, i) => {
      const yy = top + row * i;
      s += tx(24, yy + 22, lane[1], 'sl-small');
      for (const e of events.filter(e => e.lane === lane[0])) {
        const width = Math.max(2, x(e.end) - x(e.start)), color = palette[lane[2] || 'green'];
        const active = time >= e.start && time < e.end;
        const completed = time >= e.end;
        s += '<g class="sl-event" data-event="' + esc(e.id) + '" tabindex="0" role="button" aria-label="' + esc(e.title) +
          '"><title>' + esc(e.title + '：' + e.detail) + '</title>' +
          rect(x(e.start), yy + 4, width, 25, 'none', 'stroke="' + color + '" opacity=".6"' + (e.wait ? ' stroke-dasharray="4 3"' : '')) +
          rect(x(e.start), yy + 4, width * M.progress(time, e.start, e.end), 25, e.wait ? 'url(#' + drawingPrefix + '-hatch)' : color,
            'opacity="' + (active ? '.35' : completed ? '.22' : '0') + '"') +
          (active || selected === e.id ? rect(x(e.start), yy + 4, width, 25, 'none', 'stroke="' + color + '" stroke-width="2.5"') : '') +
          (width > 75 ? tx(x(e.start) + 5, yy + 21, e.title.slice(0, Math.max(3, Math.floor(width / 14) - 1)) +
            (e.title.length * 14 > width ? '…' : ''), 'sl-small') : '') + '</g>';
      }
    });
    s += '<line class="sl-cursor" x1="' + x(time) + '" x2="' + x(time) + '" y1="' + (top - 8) +
      '" y2="' + (top + row * lanes.length - 6) + '"/>';
    return s;
  }
  function currentEvent() {
    return plan.events.find(e => e.id === selected) ||
      plan.events.find(e => t >= e.start && t < e.end) ||
      plan.events.filter(e => e.end <= t).sort((a, b) => a.end - b.end).slice(-1)[0] || plan.events[0];
  }
  function renderMemory() {
    const b = M.memoryBudget(options()), o = b.options;
    svg.innerHTML = begin('字节预算、KV 槽位与请求页需求', '四段讲解使用同一组参数：字节预算、单位成本、池容量和请求页需求。不是分配或执行时间线。', 710) +
      window.ServingMemoryView.moduleFigure(b, t, { tx, rect, arrow, palette });
    $('#sl-equation').textContent = '加载前可用基线 ' + fmt(o.baseline) + ' GiB；本 rank ' + o.layers + ' 层，K/V head_dim 各 ' + o.keyDim + '。每页 ' + o.page + ' 槽位；每条后续规划 ' + o.reserve + ' 个位置。';
    putHTML($('#sl-results'), '<div class="sl-results">' +
      result('字节预算可支持（页对齐后）', fmt(b.profiledTokens, 0) + ' 槽位') +
      result('最终池容量', fmt(b.pages, 0) + ' 页 / ' + fmt(b.capacity, 0) + ' 槽位') +
      result('每条请求的规划页', fmt(b.perRequestPages, 0) + ' 页') +
      result('整批规划页需求', fmt(b.demandPages, 0) + ' 页') +
      result('每条尾页空位', fmt(b.paddingPerRequest, 0) + ' 槽位') +
      result('执行预留检查', b.rawBudget <= 0 ? '先修正启动预算' : b.executionFits ? '假设未超出' : '新增峰值超出') + '</div>');
    const e = currentEvent();
    const detail = e.id === 'budget' ? '当前可用 ' + fmt(o.available) + ' GiB 已扣除驻留对象；减去运行预留 ' + fmt(b.slack) + ' GiB 后，KV 预算为 ' + fmt(b.rawBudget) + ' GiB。' :
      e.id === 'cost' ? '同一 token 位置在本 rank 所计各层的 K/V 合起来占 ' + fmt(b.tokenBytes / 1024) + ' KiB。用字节预算除以这个成本，才能得到槽位数。' :
      e.id === 'capacity' ? '字节预算按页可支持 ' + fmt(b.profiledTokens, 0) + ' 槽位；再受用户上限约束并向下对齐，最终得到 ' + fmt(b.pages, 0) + ' 页。' :
        '每条的 ' + fmt(b.requestedPositions, 0) + ' 个位置向上取整为 ' + fmt(b.perRequestPages, 0) + ' 页；' + o.requests + ' 条共需 ' + fmt(b.demandPages, 0) + ' 页。' + window.ServingMemoryView.verdict(b);
    stage(e.title, detail, e.source);
  }
  function renderHicache() {
    const st = M.hicacheState(plan, t), c = { M, plan, t, duration, tx, rect, flow, tracks, palette };
    const view = window.ServingHiCacheView.figure(c, true);
    svg.innerHTML = begin('同一前缀的 HiCache 回载与释放条件', 'L3 查询、L2 读入与采用、L1 分配与逐层可读、H2D ACK 分开；主图、执行轨道和脉冲共用时钟。', view.height) + view.body;
    putHTML($('#sl-hicache-gates'), window.ServingHiCacheView.gates(plan, st));
    const e = currentEvent(), summary = window.ServingHiCacheView.summary(plan, st);
    const active = st.current.map(e => e.title).join('；');
    stage(summary[0], summary[1] + (active ? ' 当前' + (st.current.length > 1 ? '同时推进：' : '事件：') + active + '。' : ''), selected ? e.source : window.ServingHiCacheView.sourceKey(st));
    putHTML($('#sl-results'), '<div class="sl-results">' +
      result('L3 读取交接 ACK', window.ServingHiCacheView.storageAckLabel(plan, st)) +
      result('已可读层 / 已读完层', st.readyLayers.length + ' / ' + st.readLayers.length) +
      result('H2D 整批完成 ACK', st.ack ? '已消费' : st.ackPending ? '复制完，尚未消费' : st.deviceAllocated ? '复制未全部完成' : '尚未提交 H2D') +
      result('R1 设备前缀引用', st.requestPinned ? '仍持有' : st.canceled ? '本次未建立' : '尚未建立') + '</div>');
    for (const button of root.querySelectorAll('[data-hicache-moment]')) {
      const moment = hicacheMoment(button.dataset.hicacheMoment);
      button.disabled = moment === null;
      button.title = moment === null ? '当前场景没有这一阶段' : '暂停到 ' + fmt(moment) + ' u';
    }
  }
  function hicacheMoment(name) {
    const end = id => plan.events.find(e => e.id === id)?.end;
    if (name === 'hit') return end('exists') ?? end('host-match') ?? null;
    if (name === 'layer') return end('copy-0') === undefined ? null : end('copy-0') + .5;
    if (name === 'copied') return end('copy-2') === undefined ? null : end('copy-2') + .5;
    if (name === 'ack') return end('load-ack') === undefined ? null : end('load-ack') + .5;
    if (name === 'end') return duration;
    return null;
  }
  function renderDeployment() {
    const done = id => plan.events.some(e => e.id === id && t >= e.end);
    const e = currentEvent();
    const nodes = {
      client: [35, 75, 165, 95, '实验客户端', '请求 / 观察证据'],
      http: [285, 75, 235, 95, 'HTTP + TokenizerManager', '服务端前端进程'],
      scheduler: [700, 75, 260, 95, 'Scheduler', '本例独立调度进程'],
      gpu: [285, 275, 235, 100, '单张 GPU', done('weights') ? '权重已加载；执行依阶段准备' : '尚在装载 / 初始化'],
      detok: [700, 275, 260, 100, 'DetokenizerManager', '本例独立反分词进程']
    };
    let s = begin('从启动到完整请求', '服务端前端、Scheduler、Detokenizer 进程与单 GPU 分别呈现，客户端与设备工作共享时钟。', 865);
    s += rect(255, 38, 725, 365, 'none', 'stroke="var(--border)" stroke-dasharray="7 5"') +
      tx(275, 29, '服务端：本例 Linux 单 GPU；跨进程消息与设备提交分别标注', 'sl-small');
    Object.values(nodes).forEach(n => {
      s += rect(n[0], n[1], n[2], n[3], 'var(--soft)', 'stroke="var(--border)"') +
        tx(n[0] + 12, n[1] + 31, n[4], 'sl-strong') +
        tx(n[0] + 12, n[1] + 59, n[5], 'sl-small');
    });
    const post = [[200, 105], [285, 105]], ipc = [[520, 105], [700, 105]];
    const submit = [[735, 170], [735, 211], [400, 211], [400, 275]];
    const back = [[520, 320], [610, 320], [610, 144], [700, 144]];
    const detok = [[830, 170], [830, 275]], text = [[700, 346], [550, 346], [550, 160], [520, 160]];
    const visible = [[285, 149], [240, 149], [240, 209], [115, 209], [115, 170]];
    const paths = { post, ipc, submit, back, detok, text, visible };
    const colors = { post: 'blue', ipc: 'blue', submit: 'green', back: 'green', detok: 'blue', text: 'purple', visible: 'purple' };
    for (const [path, points] of Object.entries(paths)) {
      const active = (plan.transfers || []).filter(e => e.path === path && t >= e.start && t < e.end);
      const fallback = (plan.transfers || []).find(e => e.path === path);
      s += flow(points, active[0] || (fallback ? { ...fallback, start: Infinity, end: Infinity } : null),
        active[0]?.payload || fallback?.payload || path, active[0]?.kind === 'control' ? 'gold' : colors[path], active[0]?.kind || 'data');
    }
    s += tx(240, 91, t >= 13 && t < 13.25 && plan.scenario === 'cancel' ? '取消控制' : 'HTTP 请求', 'sl-small', 'text-anchor="middle"');
    s += tx(605, 91, t >= 13.25 && t < 13.5 && plan.scenario === 'cancel' ? '取消控制 / IPC' : 'token IDs / 本机 IPC', 'sl-small', 'text-anchor="middle"');
    s += tx(450, 201, '本轮张量 / GPU 提交', 'sl-small');
    s += tx(622, 244, '输出 ID / 回读', 'sl-small');
    s += tx(844, 225, '输出 ID / IPC', 'sl-small');
    s += tx(565, 360, '文本 / IPC 回程', 'sl-small');
    s += tx(45, 230, 'HTTP 文本', 'sl-small');
    const activeGPU = plan.events.filter(x => x.lane === 'gpu' && ['prefill','decode'].includes(x.id) && t >= x.start && t < x.end);
    if (activeGPU.length) s += rect(279, 269, 247, 112, 'none', 'stroke="var(--accent)" stroke-width="3"');
    s += tx(40, 263, done('boot-failed') ? '初始化失败，未 ready' : done('ready') ? '身份 / 就绪已检查' : '尚未完成就绪检查',
      'sl-small', done('boot-failed') ? 'style="fill:var(--sl-red)"' : '');
    s += tx(40, 293, done('finish') ? 'R1：已核对完成证据' : done('retire') ? 'R1：沿取消路径收尾' :
      t >= 13 && plan.scenario === 'cancel' ? 'R1：取消通知已发出' : t >= 9 ? 'R1：请求在途' : 'R1：尚未发送', 'sl-small');
    const laneFor = e => ['weights', 'kv', 'warmup'].includes(e.id) ? 'init' :
      ['ready', 'finish', 'retire', 'terminal', 'boot-failed'].includes(e.id) ? 'client' : e.lane;
    const execution = plan.events.filter(e => e.id !== 'text').map(e => ({ ...e, lane: laneFor(e) }));
    s += tracks(execution, [['init', '初始化阶段（含等待）', 'muted'], ['client', '客户端请求 / 检查', 'gold'],
      ['http', 'HTTP 前端进程', 'blue'], ['scheduler', 'Scheduler 提交', 'green'],
      ['detok', 'Detokenizer 处理', 'purple'], ['gpu', 'GPU 执行示例', 'green'], ['return', '结果交接活动', 'purple']], 465, t, duration);
    s += tx(40, 810, '回程交接 12–13 u 与下一轮 Decode 重叠；初始化条是阶段范围，不表示 GPU 一直忙。', 'sl-small');
    s += tx(40, 838, '完整箭头标明载荷；短划线持续脉冲仅表示该交接活动，不计包数或传输百分比。', 'sl-small');
    svg.innerHTML = s;
    const active = plan.events.filter(e => t >= e.start && t < e.end).map(e => e.title);
    const lesson = window.ServingPractice.deployment(plan, t);
    putHTML($('#sl-practice-reading'), lesson.html);
    stage(lesson.title, lesson.detail + (active.length ? ' 当前事件：' + active.join('；') + '。' : ''), e.source);
    const commands = {
      identity: 'git rev-parse HEAD\npython3 -m pip freeze\nnvidia-smi',
      weights: 'sglang serve --model-type llm --model-path ../model-weights/study-model --served-model-name study-model --host 127.0.0.1 --port 30000 --tp-size 1 --pp-size 1 --dp-size 1 --disable-overlap-schedule --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled',
      kv: '# 保存启动日志，核对内存基线、KV 容量与初始化异常。\n# 继续在显存账本课核对对应单位与采样位置。',
      ready: 'curl --fail --silent --show-error http://127.0.0.1:30000/model_info',
      post: 'python3 -B scripts/serving_labs.py smoke --url http://127.0.0.1:30000 --output runs/study-smoke-01',
      terminal: 'python3 -m json.tool runs/study-' + (plan.scenario === 'cancel' ? 'cancel' : 'smoke') + '-01/run.json\n# 对照 request-*.observations.json、原始响应和同 rid 的服务日志。',
      finish: 'python3 -m json.tool runs/study-smoke-01/run.json',
      retire: '# 对照取消通知、流尾与服务日志。采集器不会自动核实 GPU 资源归还。',
      abort: 'python3 -B scripts/serving_labs.py cancel --url http://127.0.0.1:30000 --output runs/study-cancel-01 --max-new-tokens 64'
    };
    $('#sl-command').textContent = commands[e.id] || (e.id === 'boot-failed' ?
      '# 保留最早异常；调整配置后重新启动，不把此实例当成已 ready。' :
      '# 本步骤通过启动日志、run.json、原始响应和 SSE 核对。\n# 本网页不连接或启动你的推理服务。');
  }
  function benchmarkData() { return imported || M.workload(options()); }
  function renderBenchmark() {
    const d = benchmarkData(), o = options(), ms = d.unit === 's' ? 's' : 'u';
    const stats = M.metrics(d.rows, d.duration, { ttft: o.slaTTFT, tpot: o.slaTPOT });
    const x0 = 180, width = 780, x = v => x0 + v / d.duration * width;
    const rows = d.rows.slice(0, 36), height = 100 + rows.length * 49;
    let s = begin('请求的到达、许可与观测', '条带分别表示客户端等待、服务等待与工作；导入实测时只画实际具备证据的区间。', height);
    for (let i = 0; i <= 4; i++) {
      const xx = x(d.duration * i / 4);
      s += '<line class="sl-grid" x1="' + xx + '" y1="40" x2="' + xx + '" y2="' + (height - 30) + '"/>' +
        tx(xx, 28, fmt(d.duration * i / 4, 2) + ' ' + ms, 'sl-small', 'text-anchor="middle"');
    }
    rows.forEach((r, i) => {
      const y = 56 + i * 49;
      if (i === (Number(selected) || 0)) s += rect(6, y - 2, 4, 43, palette.green);
      s += tx(24, y + 19, String(r.id).slice(0, 18), i === (Number(selected) || 0) ? 'sl-small sl-strong' : 'sl-small');
      const permit = Number.isFinite(r.permit) ? r.permit : r.start;
      const segs = [[r.scheduled, permit, 'muted', '客户端许可等待'], [permit, r.start, 'purple', '客户端发送前准备']];
      if (Number.isFinite(r.work)) {
        segs.push([r.start, r.work, 'gold', '服务等待'], [r.work, r.deviceFirst, 'green', '输入处理'],
          [r.deviceFirst, r.end, r.success ? 'blue' : 'red', r.success ? '续写与收尾' : '失败前消耗']);
      } else segs.push([r.start, r.end, r.success ? 'blue' : 'red', 'adapter 观测区间；服务内部未拆分']);
      segs.forEach(([a, b, c, label]) => {
        if (!Number.isFinite(a) || !Number.isFinite(b) || b <= a) return;
        const w = x(b) - x(a);
        s += '<g class="sl-event" data-row="' + i + '" tabindex="0" role="button" aria-label="' + esc(r.id + ' ' + label) + '"><title>' +
          esc(r.id + '：' + label + ' [' + fmt(a) + ', ' + fmt(b) + '] ' + ms) + '</title>' +
          rect(x(a), y, w, 26, 'none', 'stroke="' + palette[c] + '"') +
          rect(x(a), y, w * M.progress(t, a, b), 26, palette[c], 'opacity=".35"') + '</g>';
      });
      if (Number.isFinite(r.first)) {
        s += '<line x1="' + x(r.first) + '" x2="' + x(r.first) + '" y1="' + (y - 3) + '" y2="' + (y + 30) + '" stroke="var(--fg)" stroke-width="2"/>';
      }
      s += tx(24, y + 39, r.success ? '最终成功' : '最终失败', 'sl-small', r.success ? '' : 'style="fill:var(--sl-red)"');
    });
    s += '<line class="sl-cursor" x1="' + x(t) + '" x2="' + x(t) + '" y1="40" y2="' + (height - 30) + '"/>';
    svg.innerHTML = s;
    const row = d.rows[Number(selected) || 0] || d.rows[0];
    const picker = $('[name=requestFocus]');
    putHTML(picker, d.rows.map((r, i) => '<option value="' + i + '">' + esc(r.id) + '</option>').join(''));
    picker.value = String(Number(selected) || 0);
    putHTML($('#sl-practice-reading'), window.ServingPractice.benchmark(M, d, row, o));
    stage(row.id + '：逐请求账本', '计划到达 ' + fmt(row.scheduled) + '，获得许可 ' + fmt(row.permit) + '，adapter 开始 ' + fmt(row.start) +
      '，首个非空输出 ' + fmt(row.first) + '，结束 ' + fmt(row.end) + ' ' + ms +
      '。TTFT 从 adapter 开始计量，前面的许可等待另记。' + (row.error ? ' 错误：' + row.error : ''), 'benchmark');
    $('#sl-record-label').textContent = imported ? '当前：导入的原始采集记录；单位 s。' +
      (d.evidence === 'measured' ? '采集器声明为实测，仍需核对版本与环境证据。' : '文件未声明实测身份。') :
      '当前：合成教学排程，单位 u；一个串行服务位，非 SGLang 执行模拟或压测结果。';
    $('#sl-lab-boundary').textContent = imported ?
      '横轴使用采集器的客户端相对秒。只绘制记录中具备证据的区间，不从客户端时刻补造服务端内部阶段。' :
      'u 是任意教学时间单位；边框为完整排程，填色为回看进度，左侧色条标出固定查看请求。教学窗口含最后请求结束后的 0.5 u 留白。并发许可和服务位只用于解释等待，不预测 SGLang 性能。';
    putHTML($('#sl-results'), '<p class="sl-control-note">以下统计使用完整记录和统一窗口，与回放游标位置无关。P95 使用线性插值。</p><div class="sl-results">' +
      result('成功 / 全部请求', stats.completed + ' / ' + stats.requests) +
      result('失败请求', stats.failed) + result('TTFT P95（' + stats.ttftSamples + ' 样本）', fmt(stats.p95TTFT) + ' ' + ms) +
      result('TPOT P95（' + stats.tpotSamples + ' 样本）', fmt(stats.p95TPOT) + ' ' + ms) +
      result('成功吞吐', fmt(stats.requestThroughput, 3) + ' req/' + ms) +
      result('达标吞吐（' + stats.goodputCount + ' 条）', fmt(stats.goodput, 3) + ' req/' + ms) +
      result('成功输出吞吐', fmt(stats.outputThroughput, 3) + ' token/' + ms) +
      result('统一窗口', fmt(d.duration, 3) + ' ' + ms) + '</div>' +
      (d.rows.length > 36 ? '<p class="sl-note">时间图仅显示前 36 条，统计仍覆盖全部 ' + d.rows.length + ' 条请求。</p>' : '') +
      '<details class="sl-reading"><summary>逐请求计时与样本</summary><div class="sl-table-wrap"><table><thead><tr><th>请求</th><th>许可前等待</th><th>TTFT</th><th>结束</th><th>输出 token</th><th>结果</th></tr></thead><tbody>' +
      d.rows.slice(0, 200).map(r => '<tr><th>' + esc(r.id) + '</th><td>' + fmt((Number.isFinite(r.permit) ? r.permit : r.start) - r.scheduled) +
        '</td><td>' + fmt(Number.isFinite(r.first) ? r.first - r.start : null) + '</td><td>' + fmt(r.end) +
        '</td><td>' + fmt(r.out, 0) + '</td><td>' + (r.success ? '成功' : '失败') + '</td></tr>').join('') +
      '</tbody></table></div></details>');
  }

  function traceWindow() {
    const o = options(), span = trace.duration * o.windowSpan / 100;
    const start = (trace.duration - span) * o.windowStart / 100;
    return { start, end: start + span };
  }
  function currentTraceEvent(stats) {
    return trace.spans.find(e => e.id === traceSelected) || stats.spans.find(e => e.gpu && t >= e.start && t < e.end) ||
      stats.spans.find(e => t >= e.start && t < e.end) || stats.spans.find(e => e.gpu) || stats.spans[0];
  }
  function renderTrace() {
    if (!trace) return;
    const w = traceWindow(), stats = M.traceStats(trace, w.start, w.end);
    const filtered = stats.spans.filter(e => options().traceFilter !== 'gpu' || e.gpu);
    // Preserve both CPU launches and device events rather than filling the
    // display limit with one long, deeply nested CPU lane.
    const sample = (a, count) => a.length <= count ? a : Array.from({ length: count }, (_, i) => a[Math.floor(i * a.length / count)]);
    const gpu = sample(filtered.filter(e => e.gpu), 650);
    const cpu = sample(filtered.filter(e => !e.gpu && /^(cuda_runtime|cuda_driver|Runtime)$/i.test(e.cat)), 350);
    const other = sample(filtered.filter(e => !e.gpu && !/^(cuda_runtime|cuda_driver|Runtime)$/i.test(e.cat)), 150);
    const shown = [...gpu, ...cpu, ...other].sort((a, b) => a.start - b.start);
    const forced = trace.spans.find(e => e.id === traceSelected);
    const linked = forced?.gpu ? M.launchCandidates(trace, forced).slice(0, 6) : [];
    for (const e of [forced, ...linked].filter(Boolean)) if (e.end > w.start && e.start < w.end && !shown.includes(e)) shown.push(e);
    shown.sort((a, b) => a.start - b.start);
    const lanes = [...new Set([...[forced, ...linked].filter(Boolean).map(e => e.lane), ...shown.map(e => e.lane)])].slice(0, 16), x0 = 210, width = 750;
    const x = n => x0 + (n - w.start) / (w.end - w.start) * width;
    let s = begin('实测 Trace 时间线', '每条事件保持原始相对时间差，按 pid/tid 分行；显示上限不影响完整窗口的统计。', 95 + lanes.length * 76);
    for (let i = 0; i <= 4; i++) {
      const xx = x(w.start + (w.end - w.start) * i / 4);
      s += '<line class="sl-grid" x1="' + xx + '" x2="' + xx + '" y1="44" y2="' + (60 + lanes.length * 76) + '"/>' +
        tx(xx, 28, fmt((w.start + (w.end - w.start) * i / 4) / 1000, 2) + ' ms', 'sl-small', 'text-anchor="middle"');
    }
    const selectedEvent = trace.spans.find(e => e.id === traceSelected);
    const correlated = selectedEvent && selectedEvent.gpu ? M.launchCandidates(trace, selectedEvent).map(e => e.id) : [];
    lanes.forEach((lane, i) => {
      const row = shown.filter(e => e.lane === lane), yy = 58 + i * 76;
      if (!row.length) return;
      const label = row[0].laneName;
      s += tx(18, yy + 18, 'PID ' + String(row[0].pid).slice(0, 14), 'sl-small') +
        tx(18, yy + 36, 'TID ' + String(row[0].tid).slice(0, 14), 'sl-small') +
        tx(18, yy + 54, (row[0].gpu ? 'GPU · ' : '主机/其他 · ') + label.slice(0, 12), 'sl-small');
      // Coverage is computed from all events in this lane. Sampling individual
      // clickable rectangles must never create a false empty interval.
      for (const [a, b] of M.union(filtered.filter(e => e.lane === lane)
        .map(e => [Math.max(w.start, e.start), Math.min(w.end, e.end)]))) {
        s += rect(x(a), yy - 1, Math.max(.5, x(b) - x(a)), 55,
          row[0].gpu ? palette.green : palette.blue, 'opacity=".1"');
      }
      const ends = [w.start, w.start, w.start];
      row.forEach(e => {
        let level = ends.findIndex(n => n <= e.start);
        if (level < 0) level = 2;
        ends[level] = e.end;
        const a = Math.max(w.start, e.start), b = Math.min(w.end, e.end);
        const ww = Math.max(1, x(b) - x(a));
        const color = e.gpu ? (/memcpy|memset/i.test(e.cat) ? 'purple' : 'green') : 'blue';
        const focused = traceSelected === e.id || correlated.includes(e.id);
        s += '<g class="sl-event" data-trace="' + e.id + '" tabindex="0" role="button" aria-label="' +
          esc(e.name + '，' + fmt(e.duration) + ' 微秒') + '"><title>' + esc(e.name) + '</title>' +
          rect(x(a), yy + level * 18, ww, 17, palette[color], 'opacity="' + (focused ? '.9' : '.4') +
            '" stroke="' + (focused ? 'var(--fg)' : palette[color]) + '" stroke-width="' + (focused ? 2 : 1) + '"')
          + '</g>';
      });
    });
    s += '<line class="sl-cursor" data-trace-cursor x1="' + x(w.start) + '" x2="' + x(w.start) + '" y1="44" y2="' + (60 + lanes.length * 76) + '"/>';
    const geometry = [w.start, w.end, traceSelected, options().traceFilter].join(':');
    if (geometry !== traceGeometry) { svg.innerHTML = s; traceGeometry = geometry; }
    const cursor = $('[data-trace-cursor]');
    cursor.setAttribute('x1', x(t)); cursor.setAttribute('x2', x(t)); cursor.style.display = t >= w.start && t <= w.end ? '' : 'none';
    $('#sl-window-label').textContent = fmt(w.start / 1000) + '–' + fmt(w.end / 1000) + ' ms';
    const current = currentTraceEvent(stats);
    if (current) {
      const launches = current.gpu ? M.launchCandidates(trace, current) : [];
      const args = Object.entries(current.args).filter(([k]) => /stream|device|correlation|external|Input Dims|grid|block/i.test(k));
      putHTML($('#sl-stage'), '<h3 class="sl-trace-name">' + esc(current.name) + '</h3><div class="sl-inspector"><dl>' +
        '<dt>原始起点</dt><dd>' + esc(fmt(trace.origin + current.start, 3)) + ' µs</dd>' +
        '<dt>相对范围 / 时长</dt><dd>' + esc(fmt(current.start, 3) + '–' + fmt(current.end, 3) + ' / ' + fmt(current.duration, 3)) + ' µs</dd>' +
        '<dt>类别 / pid / tid</dt><dd>' + esc(current.cat + ' / ' + current.pid + ' / ' + current.tid) + '</dd>' +
        args.map(([k, v]) => '<dt>' + esc(k) + '</dt><dd>' + esc(JSON.stringify(v)) + '</dd>').join('') +
        '</dl></div><p>' + (current.gpu ? (launches.length === 1 ? '按 correlation 找到一条较早的 CPU runtime 候选；仍须核对进程与采集上下文。' :
          launches.length > 1 ? '同一 correlation 对应多条 CPU 候选，不能在此认定唯一提交调用。' :
            '片段中没有唯一可关联的 CPU runtime，不能补造提交事件。') :
          current.cpu ? '这是主机侧区间。它的持续时间不等于 GPU 执行时间。' :
            '本事件未按已知 GPU kernel/拷贝类别归类；请结合原始类别和 pid/tid 核对，不擅自计入 GPU 活动并集。') + '</p>' +
        '<button type="button" data-trace="' + current.id + '">聚焦当前事件</button> ' + launches.slice(0, 6).map(e => '<button type="button" data-trace="' + e.id + '">' + esc(e.name + ' / PID ' + e.pid) + '</button>').join(' '));
      $('#sl-stage').setAttribute('aria-live', running ? 'off' : 'polite');
      putHTML($('#sl-practice-reading'), window.ServingPractice.trace(M, trace, current, w));
      source('profiler');
    } else {
      stage('当前窗口没有可识别的完整区间', '扩大或平移窗口查看其他范围；本窗口缺少区间证据，不能据此认定设备空闲。', 'profiler');
      putHTML($('#sl-practice-reading'), '<p class="pl-reading">当前窗口没有可检查的事件，原文件仍保留。窗口统计不补造缺失活动或提交关系。</p>');
    }
    putHTML($('#sl-results'), '<div class="sl-results">' +
      result('当前窗口', fmt(stats.window / 1000) + ' ms') +
      result('GPU 活动并集', fmt(stats.observedGpuActive / 1000) + ' ms') +
      result('GPU 事件累计时长', fmt(stats.summedGpuDuration / 1000) + ' ms') +
      result('未记录 GPU 活动', fmt(stats.observedGpuEmpty / 1000) + ' ms') + '</div>' +
      '<p class="sl-note">统计按窗口内全部已识别事件计算；图中最多显示 650 条 GPU、350 条 runtime 和 150 条其他 CPU 事件，最多 16 个 pid/tid。显示 ' +
      shown.filter(e => lanes.includes(e.lane)).length + ' / ' + filtered.length +
      ' 条。GPU 统计只识别 kernel、Memcpy/Memset 与 gpu_memcpy/gpu_memset 类别，不从进程名字猜测。浅色底带是该显示 lane 的全部区间并集，单个事件在整段中均匀抽样；缩小窗口可看更多细节。重叠事件分行不推断调用层级。未记录 GPU 活动不等于设备空闲。</p>' +
      '<p class="sl-note">表内长名称最多显示两行；选中后在上方查看完整名称与字段。</p><div class="sl-table-wrap"><table class="sl-tops"><thead><tr><th>窗口内 GPU 事件</th><th>累计时长</th><th>次数</th></tr></thead><tbody>' +
      stats.top.slice(0, 6).map(a => {
        const e = stats.spans.find(e => e.gpu && e.name === a.name);
        return '<tr><td><button type="button" data-trace="' + e.id + '">' + esc(a.name) + '</button></td><td class="sl-nowrap">' +
          fmt(a.total / 1000) + ' ms</td><td>' + a.count + '</td></tr>';
      }).join('') + '</tbody></table></div><p class="sl-note">未纳入区间：格式无效 ' +
      trace.diagnostics.dropped + '；未匹配 B/E ' + trace.diagnostics.unmatched +
      '；非区间事件 ' + trace.diagnostics.unsupported + '（含瞬时、flow、counter 等，未用于区间统计）。</p>');
    $('#sl-record-label').textContent = traceLabel;
  }

  function updateDepth() {
    if (!depth || !plan || kind === 'trace' && !trace) return;
    const e = kind === 'trace' ? currentTraceEvent(M.traceStats(trace, traceWindow().start, traceWindow().end)) :
      kind === 'benchmark' ? benchmarkData().rows[Number(selected) || 0] : currentEvent();
    let rows = [['当前时刻', kind === 'trace' ? fmt(t / 1000, 3) + ' ms' : fmt(t, 3) + (kind === 'benchmark' && imported ? ' s' : ' u')]], explanation = '';
    if (kind === 'memory') {
      const b = M.memoryBudget(options());
      rows.push(['KV 原始预算', fmt(b.rawBudget) + ' GiB'], ['每槽位载荷', fmt(b.tokenBytes) + ' bytes'],
        ['页对齐后容量', fmt(b.capacity) + ' 槽位'], ['当前请求需求', fmt(b.demandPages) + ' 页 / ' + fmt(b.demand) + ' 槽位'], ['每条页内空位', fmt(b.paddingPerRequest) + ' 槽位'], ['执行峰值假设', fmt(b.executionNeed) + ' GiB']);
      explanation = '当前可用量已扣除驻留权重，不重复减权重。字节到槽位按本 rank 的 KV heads、K/V 维度、层数与元素字节计算；图中未计所有 buffers。';
    } else if (kind === 'hicache') {
      const st = M.hicacheState(plan, t);
      rows.push(['R1 可读层 / 已读层', st.readyLayers.join(', ') + ' / ' + st.readLayers.join(', ')],
        ['后台 IO 持有目标', String(st.ioOwns)], ['释放队列待归还', String(st.releaseQueued)], ['L2 目标实际已归还', String(st.hostReleased)], ['回载主机源保护', String(st.hostTransferPinned)], ['设备目标传输保护', String(st.deviceTransferPinned)],
        ['L3 交接 ACK 已消费', String(st.storageAck)], ['H2D 三层复制完成', String(st.allLayersReady)], ['H2D ACK 已消费', String(st.ack)], ['请求前缀引用', String(st.requestPinned)]);
      explanation = window.ServingHiCacheView.summary(plan, st)[1] + (selected ? ' 选中事件：' + e.title + '。' : '');
    } else if (kind === 'deployment') {
      const active = (plan.transfers || []).filter(e => t >= e.start && t < e.end);
      rows.push(['R1 场景', plan.scenario], ['活动交接', active.map(e => e.payload).join('；') || '无'],
        ['当前执行', plan.events.filter(e => t >= e.start && t < e.end).map(e => e.title).join('；') || '无'],
        ['源事件区间', e.start + '–' + e.end + ' u']);
      explanation = e.detail + ' 交接区间是教学拆解；主机放置、进程与载荷关系沿固定源码核对。';
    } else if (kind === 'benchmark') {
      const unit = imported ? ' s' : ' u';
      rows.push(['请求身份', e.id], ['计划 / 许可 / adapter 开始', [e.scheduled, e.permit, e.start].map(n => fmt(n)).join(' / ') + unit],
        ['首个输出 / 结束', fmt(e.first) + ' / ' + fmt(e.end) + unit], ['成功 / 输出 token', String(e.success) + ' / ' + e.out],
        ['TPOT', e.success && e.out > 1 && Number.isFinite(e.first) ? fmt((e.end - e.first) / (e.out - 1)) + unit : '没有有效样本']);
      explanation = 'TTFT = first − start；TPOT = (end − first)/(out − 1)。许可前等待另记；统计使用完整记录与统一窗口，展开视图不改变样本集合。';
    } else if (e) {
      rows.push(['事件身份', e.id], ['原始 ts / dur', fmt(trace.origin + e.start, 3) + ' / ' + fmt(e.duration, 3) + ' µs'],
        ['cat / pid / tid', e.cat + ' / ' + e.pid + ' / ' + e.tid], ['correlation', e.correlation ?? '未记录'],
        ...Object.entries(e.args).filter(([k]) => /stream|device|external|grid|block/i.test(k)).map(([k, v]) => [k, JSON.stringify(v)]));
      explanation = '事件字段保留原记录。correlation 只用于候选关联；缺少跨事件与采集上下文时不声明唯一依赖，不给真实区间叠加教学传输脉冲。';
    }
    drawingPrefix = 'sl-overview';
    const v = window.ServingLabViews({ M, kind, t, plan, duration, tx, rect, flow, tracks, palette, fmt,
      options: options(), selectedRow: Number(selected) || 0, data: kind === 'benchmark' ? benchmarkData() : null, trace, window: trace ? traceWindow() : null });
    const body = v.html ? v.body : '<div class="sl-overview-canvas" tabindex="0" role="region" aria-label="可横向查看的机制总览">' +
      '<svg viewBox="0 0 1000 ' + v.height + '" role="img" aria-label="同一时刻的机制总览">' + defs().replaceAll('id="sl-', 'id="sl-overview-') + v.body + '</svg></div>';
    drawingPrefix = 'sl';
    const key = kind === 'trace' ? 'profiler' : kind === 'benchmark' ? 'benchmark' : kind === 'hicache' && !selected ? window.ServingHiCacheView.sourceKey(M.hicacheState(plan, t)) : e.source;
    depth.update({ time: t, body, rows, event: kind === 'trace' ? e?.name || '当前窗口无区间' : kind === 'benchmark' ? e.id + '：计时与样本' : kind === 'hicache' && !selected ? window.ServingHiCacheView.summary(plan, M.hicacheState(plan, t))[0] : e.title,
      explanation, href: E.sources[key]?.url || '', sourceLabel: '固定源码 ' + E.revision.slice(0, 10) });
  }
  let lastRemembered = -Infinity;
  function remember(force = false) {
    if (restoring || restorePending) return;
    const now = performance.now();
    if (running && !force && now - lastRemembered < 1000) return;
    lastRemembered = now;
    const u = new URL(location.href);
    for (const el of root.querySelectorAll('[name]')) {
      if (el.type === 'file' || ['seek', 'speed'].includes(el.name)) continue;
      u.searchParams.set('lab-' + el.name, el.type === 'checkbox' ? String(el.checked) : el.value);
    }
    u.searchParams.set('t', String(Number(t.toFixed(5))));
    u.searchParams.set('speed', $('[name=speed]').value);
    u.searchParams.set('record', imported || traceLabel.startsWith('当前：导入') ? 'external' : 'example');
    if (traceSelected !== null) u.searchParams.set('focus', traceSelected);
    else if (selected !== null) u.searchParams.set('focus', selected);
    else u.searchParams.delete('focus');
    history.replaceState(history.state, '', u);
  }
  function restoreControls(url) {
    for (const el of root.querySelectorAll('[name]')) {
      const key = el.name === 'speed' ? 'speed' : 'lab-' + el.name;
      if (!url.searchParams.has(key) || el.type === 'file' || el.name === 'seek') continue;
      const value = url.searchParams.get(key);
      if (el.type === 'checkbox') el.checked = value === 'true';
      else if (el.tagName === 'SELECT' ? [...el.options].some(o => o.value === value) : Number.isFinite(+value) && +value >= +el.min && +value <= +el.max) el.value = value;
    }
    $('[data-speed]').textContent = $('[name=speed]').value + '×';
  }
  function restoreTime(url) {
    if (url.searchParams.get('record') === 'external') {
      t = 0; selected = null; traceSelected = null;
      error.textContent = '导入文件不保存在链接中；请重新选择原文件。当前显示默认示例，未恢复导入记录的进度。';
    } else {
      t = M.clamp(Number(url.searchParams.get('t')) || 0, 0, duration);
      const focus = url.searchParams.get('focus');
      if (kind === 'trace') traceSelected = focus !== null && trace.spans.some(e => e.id === +focus) ? +focus : null;
      else if (kind === 'benchmark') selected = focus !== null && benchmarkData().rows[+focus] ? focus : null;
      else selected = plan.events.some(e => e.id === focus) ? focus : null;
    }
  }
  function render() {
    root.dataset.time = String(t); root.dataset.playing = String(running);
    seek.max = String(duration); seek.value = String(t);
    $('[data-clock]').textContent = kind === 'trace' ? fmt(t / 1000, 3) + ' ms' : fmt(t, 2) + ' / ' + fmt(duration, 1) + (kind === 'benchmark' && imported ? ' s' : ' u');
    $('[data-action=play]').textContent = running ? '暂停' : t >= duration ? '重播' : '播放';
    $('[data-action=play]').setAttribute('aria-pressed', String(running));
    if (kind === 'memory') renderMemory();
    else if (kind === 'hicache') renderHicache();
    else if (kind === 'deployment') renderDeployment();
    else if (kind === 'benchmark') renderBenchmark();
    else renderTrace();
    const activity = $('#sl-activity');
    if (activity) {
      const active = (kind === 'hicache' ? plan.events.filter(e => e.id === 'storage-read' || e.id.startsWith('copy-')) : plan.transfers || [])
        .filter(e => t >= e.start && t < e.end).map(e => e.payload || e.title);
      activity.textContent = (reduced() ? '静态高亮表示活动。' : '短划线持续脉冲表示路径活动，不计包数。') +
        '传递中：' + (active.join('；') || '无；等待、本地处理和引用不算传输') + '。';
    }
    updateDepth();
    remember();
  }
  function pause() { running = false; last = null; cancelAnimationFrame(raf); }
  function frame(now) {
    if (!running) return;
    const dt = last === null ? 0 : Math.min(.1, (now - last) / 1000);
    last = now;
    const rate = kind === 'trace' ? duration / 20 : kind === 'benchmark' && imported ? duration / 20 : 1;
    t = Math.min(duration, t + dt * Number($('[name=speed]').value) * rate);
    if (t >= duration) pause();
    render();
    if (running) raf = requestAnimationFrame(frame);
  }
  function rebuild() {
    pause(); t = 0; selected = null; error.textContent = '';
    if (kind === 'memory') {
      plan = { duration: 8, events: [
        { id: 'budget', start: 0, end: 2, title: '先分开当前可用量和运行预留', detail: '当前可用量已经扣除了仍驻留的权重等对象，不再重复扣权重。', source: '_profile_available_bytes' },
        { id: 'cost', start: 2, end: 4, title: '把字节预算换成本 rank 的槽位', detail: '数 KV heads、K/V 维度、层和每元素字节；用户上限之后再按页取整。', source: 'pool_cost' },
        { id: 'capacity', start: 4, end: 6, title: '池容量为什么向下取整？', source: 'pool_cost' },
        { id: 'requests', start: 6, end: 8, title: '请求页需求为什么向上取整？', source: 'pool_cost' }
      ] };
    } else if (kind === 'hicache') plan = M.hicache(options().scenario);
    else if (kind === 'deployment') plan = M.deployment(options().scenario);
    else if (kind === 'benchmark') plan = { duration: benchmarkData().duration, events: [] };
    else if (trace) plan = { duration: trace.duration, events: [] };
    duration = plan ? plan.duration : 1;
    if (restorePending) { restoreTime(initialURL); restorePending = false; }
    render();
  }
  function boundaries() {
    if (kind === 'trace' && trace) return [...new Set([0, ...trace.spans.filter(e => e.gpu).slice(0, 1500).map(e => e.start), duration])].sort((a, b) => a - b);
    if (kind === 'benchmark') return [...new Set([0, ...benchmarkData().rows.flatMap(r => [r.start, r.first, r.end]).filter(Number.isFinite), duration])].sort((a, b) => a - b);
    return [...new Set([0, ...plan.events.flatMap(e => [e.start, e.end]), duration])].sort((a, b) => a - b);
  }
  async function bytesJSON(file) {
    const limit = 50 * 1024 * 1024;
    if (file.size > limit) throw new Error('文件超过 50 MiB 上限，请先裁剪目标采集窗口。');
    let data = new Uint8Array(await file.arrayBuffer());
    if (data[0] === 0x1f && data[1] === 0x8b) {
      if (!window.DecompressionStream) throw new Error('浏览器不能解压 gzip，请先在本地解压成 JSON。');
      const reader = new Blob([data]).stream().pipeThrough(new DecompressionStream('gzip')).getReader();
      const chunks = []; let size = 0;
      try {
        while (true) {
          const part = await reader.read(); if (part.done) break;
          size += part.value.length;
          if (size > limit) { await reader.cancel(); throw new Error('解压内容超过 50 MiB 上限。'); }
          chunks.push(part.value);
        }
      } finally { reader.releaseLock(); }
      data = new Uint8Array(size); let offset = 0;
      for (const b of chunks) { data.set(b, offset); offset += b.length; }
    }
    return JSON.parse(new TextDecoder().decode(data));
  }
  async function setTrace(raw, label) {
    const parsed = M.parseTrace(raw);
    trace = parsed; traceLabel = label; traceSelected = null; traceGeometry = '';
    $('[name=windowStart]').value = '0'; $('[name=windowSpan]').value = '100';
    if (restorePending) restoreControls(initialURL);
    rebuild();
  }
  async function loadSample() {
    const seq = ++loadSequence; pause();
    try {
      const response = await fetch('samples/resnet50-step6.json');
      if (!response.ok) throw new Error('实测样本读取失败：HTTP ' + response.status);
      const raw = await response.json();
      if (seq !== loadSequence) return;
      await setTrace(raw, '当前：PyTorch/Kineto 历史 ResNet-50 训练实测片段（非 SGLang 推理），原记录相对时间，单位 ms。');
    } catch (e) { error.textContent = e.message; }
  }
  root.addEventListener('input', e => {
    if (e.target.name === 'speed') { $('[data-speed]').textContent = e.target.value + '×'; remember(true); return; }
    if (e.target.name === 'seek') { pause(); t = M.clamp(Number(e.target.value), 0, duration); if (kind !== 'benchmark') selected = null; render(); }
    if (e.target.name === 'windowStart') { pause(); render(); }
  });
  root.addEventListener('change', e => {
    if (!e.target.name || ['speed', 'seek'].includes(e.target.name)) return;
    if (kind === 'benchmark' && e.target.name === 'requestFocus') { selected = e.target.value; render(); return; }
    if (kind === 'benchmark' && ['slaTTFT', 'slaTPOT'].includes(e.target.name)) { render(); return; }
    if (kind === 'trace' && ['traceFilter', 'windowStart', 'windowSpan'].includes(e.target.name)) {
      pause(); render(); return;
    }
    rebuild();
  });
  function focusTrace(event, fit = true) {
    traceSelected = event.id; t = event.start;
    const linked = event.gpu ? M.launchCandidates(trace, event).slice(0, 6) : [];
    const events = [event, ...linked];
    if (!event.gpu || linked.length) $('[name=traceFilter]').value = 'all';
    if (fit) applyTraceWindow(Math.min(...events.map(e => e.start)), Math.max(...events.map(e => e.end)));
  }
  function applyTraceWindow(start, end) {
    const w = M.traceFocusWindow(trace, start, end);
    $('[name=windowSpan]').value = String(w.percent);
    $('[name=windowStart]').value = String(w.startPercent);
  }
  function chooseEvent(target) {
    if (target.dataset.trace !== undefined && trace) {
      pause(); traceSelected = Number(target.dataset.trace);
      const e = trace.spans.find(e => e.id === traceSelected);
      if (e) focusTrace(e);
      render(); return;
    }
    if (target.dataset.row !== undefined) {
      pause(); selected = target.dataset.row; t = benchmarkData().rows[Number(selected)].start; render(); return;
    }
    if (target.dataset.event) {
      pause(); selected = target.dataset.event;
      const e = plan.events.find(e => e.id === selected);
      if (e) t = e.start;
      render();
    }
  }
  root.addEventListener('keydown', e => {
    const target = e.target.closest('[data-event],[data-row],[data-trace]');
    if (target && ['Enter', ' '].includes(e.key)) {
      e.preventDefault();
      const key = target.dataset.trace !== undefined ? 'trace' : target.dataset.row !== undefined ? 'row' : 'event';
      const value = target.dataset[key]; chooseEvent(target);
      const next = root.querySelector('[data-' + key + '="' + CSS.escape(value) + '"]');
      if (next) next.focus({ preventScroll: true });
    }
  });
  root.addEventListener('click', async e => {
    const moment = e.target.closest('[data-hicache-moment]');
    if (moment && kind === 'hicache') {
      const target = hicacheMoment(moment.dataset.hicacheMoment);
      if (target !== null) { pause(); selected = null; t = M.clamp(target, 0, duration); render(); }
      return;
    }
    const preset = e.target.closest('[data-memory-preset]');
    if (preset && kind === 'memory') {
      const type = preset.dataset.memoryPreset;
      if (type === 'uncap') $('[name=tokenLimit]').value = '0';
      else {
        for (const el of root.querySelectorAll('.sl-controls select')) {
          const def = [...el.options].find(o => o.defaultSelected) || el.options[0];
          el.value = def.value;
        }
        if (type === 'boundary') $('[name=context]').value = '2049';
      }
      rebuild(); return;
    }
    const target = e.target.closest('[data-action],[data-event],[data-row],[data-trace],[data-copy-command]');
    if (!target) return;
    if (target.hasAttribute('data-copy-command')) {
      try { await navigator.clipboard.writeText($('#sl-command').textContent); $('#sl-copy-status').textContent = ' 已复制'; }
      catch (_) { $('#sl-copy-status').textContent = ' 可在上方选中命令复制'; }
      return;
    }
    const action = target.dataset.action;
    if (!action) return chooseEvent(target);
    if (action === 'play') {
      if (kind === 'trace' && !trace) return;
      if (running) pause();
      else { if (t >= duration) t = 0; if (kind !== 'benchmark') selected = null; running = true; last = null; raf = requestAnimationFrame(frame); }
      render();
    } else if (action === 'reset') { pause(); t = 0; if (kind !== 'benchmark') selected = null; render(); }
    else if (action === 'next' || action === 'prev') {
      pause(); if (kind !== 'benchmark') selected = null;
      const a = boundaries(); t = action === 'next' ? a.find(n => n > t + 1e-7) ?? duration :
        a.filter(n => n < t - 1e-7).slice(-1)[0] ?? 0;
      render();
    } else if (action === 'example') {
      imported = null;
      for (const name of ['interval', 'concurrency', 'failure', 'warm']) $('[name=' + name + ']').disabled = false;
      rebuild();
    } else if (action === 'trace-example') await loadSample();
    else if (action === 'trace-pair' && trace) {
      pause();
      const event = trace.spans.filter(e => e.gpu && M.launchCandidates(trace, e).length === 1).sort((a, b) => b.duration - a.duration)[0];
      if (!event) { error.textContent = '当前文件没有可唯一配对的 GPU/CPU 候选；不能补造提交关系。'; return; }
      error.textContent = ''; focusTrace(event); depth.set('modules'); render();
    } else if (action === 'trace-gap' && trace) {
      pause();
      const gap = M.largestGpuGap(trace);
      if (!gap) { error.textContent = '未找到可定位的 GPU 记录间隔；若没有识别到 GPU 事件，请先核对采集活动和类别。'; return; }
      error.textContent = ''; traceSelected = null; t = gap[0]; applyTraceWindow(gap[0], gap[1]); render();
      stage('整段记录中的最大未记录 GPU 活动区间', fmt(gap[0] / 1000) + '–' + fmt(gap[1] / 1000) +
        ' ms；当前窗口覆盖完整间隔。这里只定位记录空档，仍须核对 CPU 提交、依赖与采集覆盖。', 'profiler');
    }
  });
  if (kind === 'benchmark') $('#sl-benchmark-file').addEventListener('change', async e => {
    const file = e.target.files[0]; if (!file) return;
    pause();
    try {
      const raw = await bytesJSON(file);
      if (raw.schema !== 'ai-infra-serving-lab/v1' || raw.unit !== 's' || !Array.isArray(raw.rows) ||
        !Number.isFinite(raw.duration) || raw.duration <= 0 || !raw.rows.length) throw new Error('需要随课采集器的 run.json；不补造摘要缺失的时刻。');
      if (new Set(raw.rows.map(r => r.id)).size !== raw.rows.length) throw new Error('请求 ID 重复，不能把重复记录当作独立样本。');
      if (raw.rows.some(r => !Number.isFinite(r.scheduled) || !Number.isFinite(r.permit) ||
        typeof r.id !== 'string' || !r.id.length ||
        r.permit < r.scheduled || r.start < r.permit || !Number.isFinite(r.start) ||
        !Number.isFinite(r.end) || r.scheduled < 0 || r.start < r.scheduled || r.end < r.start ||
        r.end > raw.duration + .01 || typeof r.success !== 'boolean' ||
        (r.first !== null && (!Number.isFinite(r.first) || r.first < r.start || r.first > r.end)) ||
        !Number.isInteger(r.out) || r.out < 0 || (r.success && r.out === 0))) throw new Error('记录中的计时、计数或成功字段不完整，请核对原始采集文件。');
      imported = raw;
      for (const name of ['interval', 'concurrency', 'failure', 'warm']) $('[name=' + name + ']').disabled = true;
      rebuild();
    } catch (e) { error.textContent = '导入失败，仍显示先前记录：' + e.message; }
  });
  if (kind === 'trace') $('#sl-trace-file').addEventListener('change', async e => {
    const file = e.target.files[0]; if (!file) return;
    const seq = ++loadSequence; pause();
    try {
      const raw = await bytesJSON(file);
      if (seq !== loadSequence) return;
      await setTrace(raw, '当前：导入 ' + file.name + '；采集身份、时钟和缺失范围需按原记录核对。单位 ms。');
    } catch (e) { error.textContent = '导入失败，仍显示先前记录：' + e.message; }
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) { pause(); render(); } });
  window.addEventListener('pagehide', pause);
  restoreControls(initialURL);
  depth = window.LearningDepth.mount({ host: root, before: '.sl-canvas', module: ['.sl-canvas'],
    onChange: () => { if (plan) render(); }, persist: () => remember(true) });
  if (kind === 'hicache') {
    const gates = document.createElement('div'); gates.id = 'sl-hicache-gates'; depth.bar.before(gates);
  }
  if (['memory', 'hicache', 'deployment', 'benchmark', 'trace'].includes(kind)) {
    const transport = $('.sl-transport'), clock = $('.sl-seek');
    depth.bar.before(transport, clock);
    if (kind === 'hicache') depth.bar.before($('#sl-hicache-gates'));
  }
  if (['hicache', 'deployment'].includes(kind)) {
    const activity = document.createElement('p'); activity.id = 'sl-activity'; activity.className = 'sl-control-note';
    depth.bar.before(activity);
  }
  window.addEventListener('popstate', () => {
    initialURL = new URL(location.href); restoring = true;
    restoreControls(initialURL); pause();
    if (kind === 'benchmark' && initialURL.searchParams.get('record') !== 'external') {
      imported = null;
      for (const name of ['interval', 'concurrency', 'failure', 'warm']) $('[name=' + name + ']').disabled = false;
    }
    rebuild(); restoreTime(initialURL); render(); restoring = false;
  });
  matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change', () => render());
  root.dataset.ready = 'true';
  if (kind === 'trace') loadSample(); else rebuild();
})();
