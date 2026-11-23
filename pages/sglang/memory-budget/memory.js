/* Memory accounting is a calculation, not a simulated allocation timeline. */
(function () {
  'use strict';
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const n = value => value.toLocaleString('en-US', { maximumFractionDigits: 3 });
  const pagePhrase = b => b.options.page + ' 槽位/页';
  function verdict(b) {
    if (b.rawBudget <= 0) return '启动预算不足：运行预留还差 ' + n(-b.rawBudget) + ' GiB，不能建立正的 KV 预算。';
    if (!b.fits) return '此账本不能接纳：需求 ' + n(b.demandPages) + ' 页超过池容量 ' + n(b.pages) + ' 页，还差 ' + n(b.demandPages - b.pages) + ' 页。';
    return '页数账本可容纳：' + n(b.demandPages) + ' 页 ≤ ' + n(b.pages) + ' 页，规划后余 ' + n(b.pages - b.demandPages) + ' 页。';
  }
  function capNote(b) {
    if (b.rawBudget <= 0) return '先修正字节预算；此时不建立 KV 池，槽位和页容量均为 0。';
    if (b.userCapBinding) return '当前受用户上限 ' + n(b.options.tokenLimit) + ' 槽位限制。继续增加字节预算，也不会突破这个上限。';
    return '当前受字节预算限制；用户上限' + (b.options.tokenLimit ? '未进一步缩小容量。' : '已取消。');
  }
  function pageCells(b) {
    const count = b.perRequestPages;
    const indices = count <= 24 ? Array.from({ length: count }, (_, i) => i) :
      [...Array.from({ length: 8 }, (_, i) => i), -1, ...Array.from({ length: 8 }, (_, i) => count - 8 + i)];
    return indices.map(i => {
      if (i < 0) return '<span class="ml-page-gap">…另 ' + n(count - 16) + ' 页…</span>';
      const start = i * b.options.page;
      const old = Math.max(0, Math.min(b.options.page, b.options.context - start));
      const future = Math.max(0, Math.min(b.options.page, b.requestedPositions - start) - old);
      const title = '第 ' + (i + 1) + ' 页：已有 KV ' + old + '，后续规划 ' + future + '，尾页空位 ' + (b.options.page - old - future);
      return '<span class="ml-page" role="img" aria-label="' + title + '" title="' + title + '">' +
        '<i class="ml-old" style="width:' + old / b.options.page * 100 + '%"></i><i class="ml-future" style="width:' + future / b.options.page * 100 + '%"></i></span>';
    }).join('');
  }
  function overview(b, t) {
    const o = b.options, phase = Math.min(3, Math.floor(t / 2));
    const step = (i, title, body) => '<section class="ml-step' + (i.includes(phase) ? ' ml-current' : '') + '" data-memory-step="' + i.join(',') + '"><h3>' + title + '</h3>' + body + '</section>';
    const amount = (value, unit) => '<strong>' + n(value) + '</strong> <span class="ml-unit">' + unit + '</span>';
    const segments = [[b.resident, 'ml-resident', '已驻留（含权重）'], [Math.min(b.slack, o.available), 'ml-slack', '运行预留可覆盖部分'],
      [b.payloadGiB, 'ml-pool', '本 KV 池载荷近似'], [b.unusedBudgetGiB, 'ml-unused', '未纳入本 KV 池的预算']];
    const maxPages = Math.max(1, b.pages, b.demandPages);
    const bars = '<div class="ml-compare" aria-label="按同一页单位比较池容量与请求需求">' +
      [['池容量', b.pages, 'ml-pool'], ['这批请求的规划需求', b.demandPages, b.fits ? 'ml-pool' : 'ml-shortage']].map(([label, value, cls]) =>
        '<div><span>' + label + '：<b>' + n(value) + ' 页</b></span><div class="ml-meter"><i class="' + cls + '" style="width:' + value / maxPages * 100 + '%"></i></div></div>').join('') + '</div>';
    return '<div class="sl-memory-overview" data-memory-overview data-time="' + t + '">' +
      '<p class="ml-key">本例同一 rank：<b>1 槽位</b>保存一个 token 位置在所计各层的 K/V；<b>1 页 = ' + n(o.page) + ' 槽位</b>。槽位数与请求条数要分别计算。</p>' +
      step([0], '① 先确定能交给 KV 的字节预算',
        '<p class="ml-equation">' + amount(o.available, 'GiB 当前可用') + ' − ' + amount(b.slack, 'GiB 运行预留') + ' = ' + amount(b.rawBudget, 'GiB KV 预算') + '</p>' +
        '<p class="ml-help">运行预留 = 加载前可用基线 ' + n(o.baseline) + ' GiB × (1 − ' + o.fraction + ')。当前可用量已扣除驻留对象，权重不再重复扣除。</p>' +
        '<div class="ml-budget-bar" role="img" aria-label="字节预算分解">' + segments.map(([value, cls, label]) => '<span class="' + cls + '" style="width:' + value / o.baseline * 100 + '%" title="' + label + ' ' + n(value) + ' GiB"></span>').join('') + '</div>' +
        '<div class="ml-budget-key">' + segments.map(([value, cls, label]) => '<span><i class="' + cls + '"></i>' + label + ' ' + n(value) + ' GiB</span>').join('') + '</div>') +
      step([1, 2], '② 用每槽位成本换算，再按整页确定池容量',
        '<p class="ml-equation">' + amount(Math.max(0, b.rawBudget), 'GiB') + ' ÷ ' + amount(b.tokenBytes / 1024, 'KiB/槽位') + ' → ' + amount(b.profiledTokens, '槽位（已按页向下取整）') + '</p>' +
        '<p class="ml-help">GiB 与 KiB 换成同一单位再相除。<a href="#formula">展开每槽位成本与单位换算 →</a></p>' +
        '<p class="ml-equation">取较小值：' + n(b.profiledTokens) + ' 与 ' + (o.tokenLimit ? n(o.tokenLimit) + '（用户上限）' : '不另设上限') + '；再向下取整 → <strong>' + n(b.pages) + ' 页 = ' + n(b.capacity) + ' 槽位</strong></p>' +
        '<p class="ml-cap-note" data-memory-cap>' + capNote(b) + (b.alignmentLoss ? ' 最后舍去不足一页的 ' + n(b.alignmentLoss) + ' 个槽位。' : '') + '</p>') +
      step([3], '③ 请求按页向上取整，再与池容量比较',
        '<p class="ml-equation">每条：已有 KV ' + n(o.context) + ' + 后续规划 ' + n(o.reserve) + ' = <strong>' + n(b.requestedPositions) + ' 个位置</strong></p>' +
        '<p class="ml-equation">' + n(b.requestedPositions) + ' ÷ ' + pagePhrase(b) + '，向上取整 → <strong>' + n(b.perRequestPages) + ' 页/条</strong>（' + n(b.perRequest) + ' 槽位/条）</p>' +
        '<div class="ml-page-cells" aria-label="一条请求的规划页布局">' + pageCells(b) + '</div>' +
        '<p class="ml-help ml-page-key"><i class="ml-old"></i>已有 KV　<i class="ml-future"></i>后续规划　<i class="ml-empty"></i>尾页空位。每格表示一整页，颜色表示页内比例。</p>' +
        '<p data-memory-padding>每条尾页空位 <b>' + n(b.paddingPerRequest) + '</b> 槽位，属于该请求的页，不能挪给其他请求。</p>' +
        '<p class="ml-equation">' + n(o.requests) + ' 条 × ' + n(b.perRequestPages) + ' 页/条 = <strong>' + n(b.demandPages) + ' 页</strong>（' + n(b.demand) + ' 槽位）</p>' + bars +
        '<p class="ml-verdict' + (!b.fits ? ' ml-failed' : '') + '" data-memory-verdict>' + verdict(b) + '</p>') +
      '<p class="ml-boundary">这里按每条的“已有 KV＋后续长度”做保守规划，不代表引擎已经申请了这些页，也不是最大服务并发。已占页时只需另外申请增量页，见<a href="#pages">独立 P=4 例子</a>。</p>' +
      '<p class="ml-boundary' + (!b.executionFits ? ' ml-failed' : '') + '">执行峰值假设 ' + n(b.executionNeed) + ' GiB / 运行预留 ' + n(b.slack) + ' GiB：' +
        (b.rawBudget <= 0 ? '先修正启动预算。' : b.executionFits ? '本假设未超出预留，仍需实测。' : '新增峰值超出预留；槽位够也不能证明可运行。') + '</p></div>';
  }
  function moduleFigure(b, t, h) {
    const { tx, rect, arrow, palette: p } = h, o = b.options;
    const row = (y, idx, title) => rect(30, y, 940, 117, 'var(--soft)', 'stroke="' + (Math.min(3, Math.floor(t / 2)) === idx ? p.green : 'var(--border)') + '" stroke-width="2"') + tx(48, y + 28, title, 'sl-strong');
    let s = row(18, 0, '① 字节：当前可用 − 运行预留');
    s += tx(48, 78, n(o.available) + ' GiB − [' + n(o.baseline) + ' GiB × (1 − ' + o.fraction + ')] = ' + n(b.rawBudget) + ' GiB', 'sl-strong');
    s += tx(48, 108, '已驻留（含权重）' + n(b.resident) + ' GiB 已反映在当前可用量中，不重复扣除。', 'sl-small');
    s += arrow([[500, 139], [500, 159]], 'blue');
    s += row(166, 1, '② 单位：一个槽位保存本 rank 一个位置的各层 K/V');
    s += tx(48, 225, o.heads + ' KV heads × (' + o.keyDim + ' + ' + o.valueDim + ') × ' + o.layers + ' 层 × ' + o.bytes + ' bytes = ' + n(b.tokenBytes) + ' bytes/槽位', 'sl-strong');
    s += tx(48, 255, n(Math.max(0, b.rawBudget)) + ' × 2³⁰ bytes ÷ ' + n(b.tokenBytes) + ' bytes/槽位 = ' + n(b.rawTokens) + ' 槽位（按字节换算）', 'sl-small');
    s += arrow([[500, 287], [500, 307]], 'blue');
    s += row(314, 2, '③ 供给：用户上限与页对齐决定最终 KV 池');
    s += tx(48, 373, 'floor(' + (o.tokenLimit ? 'min(' + n(b.profiledTokens) + ', ' + n(o.tokenLimit) + ')' : n(b.profiledTokens)) + ' ÷ ' + o.page + ') × ' + o.page + ' = ' + n(b.capacity) + ' 槽位', 'sl-strong');
    s += tx(48, 403, '最终 ' + n(b.pages) + ' 页；每页 ' + n(b.pageBytes / 1024 ** 2) + ' MiB 载荷。' + (b.userCapBinding ? '当前用户上限生效，更多字节不会自动扩大池。' : '当前按字节预算确定容量。'), 'sl-small');
    s += arrow([[500, 435], [500, 455]], 'green');
    s += row(462, 3, '④ 需求：每条向上取整，再乘请求条数');
    s += tx(48, 521, 'ceil((' + n(o.context) + ' + ' + n(o.reserve) + ') ÷ ' + o.page + ') = ' + n(b.perRequestPages) + ' 页/条；' + o.requests + ' 条 × ' + n(b.perRequestPages) + ' = ' + n(b.demandPages) + ' 页', 'sl-strong');
    s += tx(48, 551, '有效＋后续规划 ' + n(b.usefulPositions) + ' 位置 ＋ 尾页空位 ' + n(b.paddingTotal) + ' 槽位 = 总规划 ' + n(b.demand) + ' 槽位', 'sl-small');
    s += tx(32, 620, verdict(b), 'sl-strong', !b.fits ? 'style="fill:var(--sl-red)"' : '');
    s += tx(32, 654, '容量向下取整，需求向上取整；比较时统一用“页”。每页 ' + pagePhrase(b) + '。', 'sl-small');
    s += tx(32, 683, '这是规划账本；没有模拟真实分配、前缀共享、逐轮准入或 GPU 执行。', 'sl-small');
    return s;
  }
  window.ServingMemoryView = { overview, moduleFigure, verdict };
})();
