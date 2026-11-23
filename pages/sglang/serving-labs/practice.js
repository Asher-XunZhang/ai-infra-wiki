/* Beginner reading aids: derived evidence, never a second execution clock. */
(function () {
  'use strict';
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const f = n => Number.isFinite(n) ? n.toLocaleString('en-US', { maximumFractionDigits: 3 }) : '未记录';
  const facts = rows => '<dl class="pl-facts">' + rows.map(([k, v]) => '<div><dt>' + esc(k) + '</dt><dd>' + esc(v) + '</dd></div>').join('') + '</dl>';
  function deployment(plan, t) {
    const done = id => plan.events.some(e => e.id === id && t >= e.end);
    const canceled = plan.scenario === 'cancel' && t >= 13;
    const failed = done('boot-failed');
    let title = '先固定身份，再等初始化完成', detail = '端口、模型信息和一次完整生成分别核对；网页没有连接真实服务。';
    if (failed) { title = '初始化失败，没有进入生成'; detail = '保留最早异常和显存账本；本例没有 ready、R1 或生成结果。调整配置后重新启动。'; }
    else if (plan.scenario === 'probe' && t >= 2 && t < 3) { title = '过早探测与模型装载同时发生'; detail = '这次探测还不能证明启动最终失败；继续核对进程与初始化日志，不能把它当作一次生成。'; }
    else if (canceled) {
      title = done('retire') ? '取消路径的教学收尾已结束，服务仍在运行' : done('terminal') ? '流尾已核对，仍要检查后台收尾' : done('cancel-ack') ? '取消通知已返回，设备可能仍在途' : '正在通知取消，设备工作不会被抹掉';
      detail = '取消通知响应、流尾/结束原因、后台资源观测是三份证据。采集器保存前两项；GPU 退役仍需服务日志或资源观测，不能从通知或时间游标推定。';
    } else if (done('finish')) { title = '完整请求证据已核对，服务并未退出'; detail = '本例假定已核对结束原因、实际 token 数和日志。最小成功只证明此通路；退出服务是之后的独立操作。'; }
    else if (done('terminal')) { title = '已收到完整响应，核对计数与日志'; detail = 'finish_reason 和完整流尾用于判断响应是否结束；再将同一 rid、实际输入/输出计数与服务日志对应。'; }
    else if (done('text')) { title = '看见首个文本，完整响应尚未确认'; detail = '首个非空文本证明结果已到客户端；还要等结束记录并核对业务状态。'; }
    else if (t >= 12) { title = '设备已采样 t1，文本仍在回程'; detail = '输出 ID 经反分词和回程后才成为客户端文字；下一轮 Decode 可以与这条回程重叠。'; }
    else if (done('ready')) { title = '身份/就绪已核对，正在验证生成'; detail = '模型信息用于核对服务身份；一次真实 /generate 的完整响应才能验证生成通路。'; }
    return { title, detail, html: facts([
      ['初始化', failed ? '失败，未 ready' : done('ready') ? '本例已 ready' : '进行中'],
      ['首个非空文本', done('text') ? '客户端已观察' : '尚未观察'],
      ['完整响应', done('terminal') ? canceled ? '本例取消结束' : '本例已核对流尾' : '尚未核对'],
      ['取消 / 后台', canceled ? done('cancel-ack') ? '通知已返回；另查后台证据' : '通知在途，不能释放推断' : '此时未取消'],
      ['服务进程', failed ? '按失败日志处理' : '本图不执行服务退出']
    ]) };
  }
  function benchmark(M, data, row, o) {
    const unit = data.unit === 's' ? 's' : 'u';
    const stats = M.metrics(data.rows, data.duration, { ttft: o.slaTTFT, tpot: o.slaTPOT });
    const ttft = Number.isFinite(row.first) ? row.first - row.start : null;
    const tpot = row.success && row.out > 1 && Number.isFinite(row.first) ? (row.end - row.first) / (row.out - 1) : null;
    const samples = data.rows.filter(r => r.success && Number.isFinite(r.first)).map(r => r.first - r.start).sort((a, b) => a - b);
    const idx = (samples.length - 1) * .95, lo = Math.floor(idx), hi = Math.ceil(idx);
    return '<section class="pl-reading" data-benchmark-reading><h3>固定查看 ' + esc(row.id) + '：先找秒表从哪里开始</h3>' + facts([
      ['计划到达', f(row.scheduled) + ' ' + unit], ['拿到客户端许可', f(row.permit) + ' ' + unit],
      ['adapter 开始计时', f(row.start) + ' ' + unit], ['首个非空输出', f(row.first) + ' ' + unit], ['请求收尾', f(row.end) + ' ' + unit]
    ]) + '<p>许可前等待 = ' + f(row.permit) + ' − ' + f(row.scheduled) + ' = <b>' + f(row.permit - row.scheduled) + ' ' + unit + '</b>。它发生在 adapter 开始前，不计入本页 TTFT。</p>' +
      '<p>TTFT = ' + f(row.first) + ' − ' + f(row.start) + ' = <b>' + f(ttft) + ' ' + unit + '</b>；从计划到达到首输出则为 <b>' + f(Number.isFinite(row.first) ? row.first - row.scheduled : null) + ' ' + unit + '</b>。两把秒表的起点不同。</p>' +
      '<p>TPOT = (结束 − 首输出) ÷ (输出 token 数 − 1)：<b>' + f(tpot) + ' ' + unit + '/token</b>。' + (row.out <= 1 || !row.success || !Number.isFinite(row.first) ? '本请求没有可用于成功 TPOT 的样本，不填 0。' : '它是平均后续输出开销，不是每个 token 的 ITL；结束观测也包含协议收尾。') + '</p>' +
      '<p>本请求最终<b>' + (row.success ? '成功' : '失败') + '</b>。固定查看的对象不随游标更换；整批统计始终使用完整记录。</p></section>' +
      '<section class="pl-reading" data-benchmark-samples><h3>P95 与达标吞吐，分别怎么算？</h3>' +
      '<p>成功 TTFT 样本（' + samples.length + ' 个，已排序）：<b>' + (samples.map(f).join('、') || '无') + '</b> ' + unit + '。</p>' +
      (samples.length ? '<p>线性插值位置 (' + samples.length + ' − 1) × 0.95 = ' + f(idx) + '（从 0 编号）；在 ' + f(samples[lo]) + ' 与 ' + f(samples[hi]) + ' 之间插值，P95 = <b>' + f(stats.p95TTFT) + ' ' + unit + '</b>。</p>' : '<p>没有成功延迟样本，P95 未定义。</p>') +
      '<p>成功吞吐 = ' + stats.completed + ' ÷ ' + f(data.duration) + ' = <b>' + f(stats.requestThroughput) + ' req/' + unit + '</b>；同时满足 TTFT ≤ ' + f(o.slaTTFT) + '、TPOT ≤ ' + f(o.slaTPOT) + ' 的成功请求有 ' + stats.goodputCount + ' 条，达标吞吐 = <b>' + f(stats.goodput) + ' req/' + unit + '</b>。</p>' +
      '<p>失败不进入成功延迟分布，但其耗时留在整轮窗口。慢请求失败可能让 P95 降低，并不能证明服务变快。只输出 1 token 的成功请求没有 TPOT，按本页约定仅检查 TTFT。</p></section>';
  }
  function trace(M, trace, event, window) {
    if (!event) return '';
    const candidates = event.gpu ? M.launchCandidates(trace, event) : [];
    return '<section class="pl-reading" data-trace-reading><h3>先认对象，再收窄原因</h3>' + facts([
      ['当前对象', event.gpu ? 'GPU 执行/拷贝区间' : event.cpu ? '主机侧区间' : '其他已记录区间'],
      ['pid / tid', event.pid + ' / ' + event.tid], ['本事件时长', f(event.duration / 1000) + ' ms'],
      ['事件与窗口', event.end > window.start && event.start < window.end ? '相交，可在图中查看' : '事件在窗口外，可重新聚焦'],
      ['CPU 提交候选', event.gpu ? candidates.length + ' 条；仍须核对上下文' : '此事件本身不是 GPU 执行'],
      ['当前窗口', f(window.start / 1000) + '–' + f(window.end / 1000) + ' ms']
    ]) + '<p>已知：这个文件记录了上述区间和字段。尚不能据此确定：请求 TTFT、跨机器先后、某段空档的根因或整机利用率。</p>' +
      '<p>聚焦事件或关联候选会调整显示窗口；统计随窗口重算，不代表性能发生变化。只描边可核对的候选，不补造传输箭头或脉冲。</p></section>';
  }
  window.ServingPractice = { deployment, benchmark, trace };
})();
