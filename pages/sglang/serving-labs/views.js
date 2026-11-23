/* Projections only: all views receive the player's event model and clock. */
(function () {
  'use strict';
  window.ServingLabViews = function (c) {
    const { M, kind, t, plan, duration, tx, rect, flow, tracks, palette: p, fmt, options: o } = c;
    let body = '', height = 420;
    const box = (x, y, w, h, title, value, color = 'green') => rect(x, y, w, h, 'var(--soft)',
      'stroke="' + p[color] + '"') + tx(x + 16, y + 29, title, 'sl-strong') + tx(x + 16, y + 58, value, 'sl-small');
    const done = id => plan.events.some(e => e.id === id && t >= e.end);
    if (kind === 'memory') {
      return { body: window.ServingMemoryView.overview(M.memoryBudget(o), t), html: true };
    } else if (kind === 'hicache') {
      return window.ServingHiCacheView.figure(c, false);
    } else if (kind === 'deployment') {
      body += tx(36, 32, '同一 R1：ready、设备采样、可见文本和完整结束分别确认', 'sl-strong');
      const gpu = plan.events.find(e => e.lane === 'gpu' && ['prefill', 'decode'].includes(e.id) && t >= e.start && t < e.end);
      body += box(36, 69, 216, 142, '实验客户端', done('finish') ? '已核对完整响应' : done('retire') ? '教学收尾结束；另查后台' : done('terminal') ? '流尾已核对；继续核对日志' :
        plan.scenario === 'cancel' && t >= 13 ? '取消已通知；仍在核对收尾' : t >= 13 ? '已观察 t1 文本' : '等待可观察证据', 'blue');
      body += box(386, 69, 266, 142, '服务主机 · CPU 进程', done('boot-failed') ? '初始化失败，未 ready' : done('ready') ? '身份与就绪已检查' : '初始化 / 未 ready', 'purple');
      body += tx(402, 166, 'HTTP / 调度 / 反分词', 'sl-small') + tx(402, 190, '展开模块过程查看本机 IPC', 'sl-small');
      body += box(786, 69, 178, 142, '单张 GPU', gpu ? gpu.id === 'prefill' ? '输入处理 / 采样' : 'Decode 在途' : t >= 14 && !done('boot-failed') ? '本例执行已完成' : '尚无本轮计算');
      const paths = { post: [[252, 106], [386, 106]], visible: [[386, 184], [252, 184]],
        submit: [[652, 106], [786, 106]], back: [[786, 184], [652, 184]] };
      for (const [path, points] of Object.entries(paths)) {
        const events = (plan.transfers || []).filter(e => e.path === path), active = events.find(e => t >= e.start && t < e.end);
        body += flow(points, active || (events[0] ? { ...events[0], start: Infinity, end: Infinity } : null),
          active?.payload || events[0]?.payload || path, active?.kind === 'control' ? 'gold' : path === 'visible' ? 'purple' : ['submit', 'back'].includes(path) ? 'green' : 'blue', active?.kind || 'data');
      }
      body += tx(318, 93, t >= 13 && t < 13.25 && plan.scenario === 'cancel' ? '取消控制' : 'HTTP 请求', 'sl-small', 'text-anchor="middle"') +
        tx(318, 171, 'HTTP 文本', 'sl-small', 'text-anchor="middle"') + tx(719, 93, '输入张量', 'sl-small', 'text-anchor="middle"') + tx(719, 171, '输出 ID', 'sl-small', 'text-anchor="middle"');
      body += tx(36, 248, plan.scenario === 'cancel' && t >= 13 && t < 15 ?
        t < 14 ? '取消已通知；设备仍在途，资源不能立即释放。' : '本例设备执行已结束；取消收尾仍需单独核对，尚未确认完整结束。' :
        t >= 12 && t < 13 ? 't1 正在回程；下一轮 Decode 无需等待整条文本回程结束。' : '初始化与回程条表示阶段范围，包含处理和等待，不能据此计算 GPU 利用率。', 'sl-small');
      const es = plan.events.filter(e => !['tokenize', 'detokenize'].includes(e.id) && !e.id.startsWith('return-')).map(e => ({ ...e,
        lane: ['weights', 'kv', 'warmup'].includes(e.id) ? 'init' : e.lane === 'scheduler' ? 'gpu-submit' : e.lane }));
      body += tracks(es, [['init', '初始化阶段（含等待）', 'muted'], ['client', '客户端检查 / 控制', 'gold'],
        ['http', '就绪 / 回程 / 收尾', 'purple'], ['gpu-submit', 'CPU 提交', 'blue'], ['gpu', 'GPU 计算示例', 'green']], 308, t, duration);
      body += tx(36, 552, '单机单 GPU 教学放置；u 非实测耗时。模块展开不改变 R1、区间、计数或结束条件。', 'sl-small');
      height = 577;
    } else if (kind === 'benchmark') {
      const d = c.data, unit = d.unit === 's' ? 's' : 'u', x = v => 200 + v / d.duration * 750;
      const rows = d.rows.slice(0, 36); height = 110 + rows.length * 48;
      body += tx(28, 27, '虚框：许可等待；实框：adapter 观测范围（含内部处理和等待）；竖线：首个非空输出', 'sl-small');
      for (let i = 0; i <= 4; i++) body += tx(x(d.duration * i / 4), 55, fmt(d.duration * i / 4) + ' ' + unit, 'sl-small', 'text-anchor="middle"');
      rows.forEach((r, i) => {
        const y = 77 + i * 48, permit = Number.isFinite(r.permit) ? r.permit : r.start;
        if (i === c.selectedRow) body += rect(8, y - 2, 4, 29, p.green);
        body += tx(26, y + 19, String(r.id).slice(0, 16), i === c.selectedRow ? 'sl-small sl-strong' : 'sl-small');
        if (permit > r.scheduled) body += rect(x(r.scheduled), y, x(permit) - x(r.scheduled), 24, 'none', 'stroke="var(--muted)" stroke-dasharray="4 3"');
        body += rect(x(r.start), y, x(r.end) - x(r.start), 24, 'none', 'stroke="' + (r.success ? p.blue : p.red) + '"') +
          rect(x(r.start), y + 19, (x(r.end) - x(r.start)) * M.progress(t, r.start, r.end), 5, r.success ? p.blue : p.red);
        if (Number.isFinite(r.first)) body += '<path d="M' + x(r.first) + ' ' + (y - 3) + 'v30" stroke="var(--fg)" stroke-width="2"/>';
      });
      body += '<path class="sl-cursor" d="M' + x(t) + ' 65V' + (height - 15) + '"/>';
    } else if (kind === 'trace' && c.trace) {
      const tr = c.trace, w = c.window, stats = M.traceStats(tr, w.start, w.end), x = v => 220 + (v - w.start) / (w.end - w.start) * 730;
      body += tx(28, 30, '原记录同一时间轴：覆盖条只表示已记录区间，不代表设备利用率', 'sl-strong');
      for (let i = 0; i <= 4; i++) body += tx(x(w.start + (w.end - w.start) * i / 4), 69,
        fmt((w.start + (w.end - w.start) * i / 4) / 1000) + ' ms', 'sl-small', 'text-anchor="middle"');
      const lanes = [['GPU 事件并集', e => e.gpu, 'green'], ['CPU runtime 记录', e => e.cpu && /^(cuda_runtime|cuda_driver|Runtime)$/i.test(e.cat), 'blue'],
        ['其他已记录区间', e => !e.gpu && !/^(cuda_runtime|cuda_driver|Runtime)$/i.test(e.cat), 'purple']];
      lanes.filter((_, i) => o.traceFilter !== 'gpu' || i === 0).forEach(([label, match, color], i) => {
        const y = 99 + i * 66; body += tx(28, y + 24, label, 'sl-small') + rect(220, y, 730, 35, 'var(--soft)');
        for (const [a, b] of M.union(stats.spans.filter(match).map(e => [Math.max(w.start, e.start), Math.min(w.end, e.end)])))
          body += rect(x(a), y, Math.max(1, x(b) - x(a)), 35, p[color], 'opacity=".45"');
      });
      if (t >= w.start && t <= w.end) body += '<path class="sl-cursor" d="M' + x(t) + ' 85V284"/>';
      body += tx(28, 325, 'GPU 未记录区间 ' + fmt(stats.observedGpuEmpty / 1000) + ' ms；先查 CPU 提交、依赖与采集范围。', 'sl-strong') +
        tx(28, 361, '展开模块过程按 pid/tid 找事件；源码注解查看原始字段与 correlation 候选。', 'sl-small') +
        tx(28, 392, '没有可靠传输起止或配对证据的路径不画脉冲；事件覆盖保留原始测量时长。', 'sl-small');
    }
    return { body, height };
  };
})();
