/* V15: two heads, independent projections, concat and output projection.
 *
 * The board is a progress-driven mechanism animation, not a set of finished
 * cards that are revealed one by one. Every visible number is tied to the
 * operation that produced it: x fans out, each head projects its own Query,
 * scores and softmax run per head, w*V contributions build that head's result,
 * four coordinate tokens move unchanged into four ordered concat slots, and
 * only Wo then mixes those four coordinates into two outputs.
 *
 * This is a self-contained 2-head / 2-dim / width-2 teaching example. It does
 * not replace the accepted single-head Q/K/V, weights or mask scenes, and it
 * makes no accuracy or performance claim. The special Wo equals an average of
 * matching coordinates only because it is deliberately chosen that way.
 */
(function(root, factory) {
 'use strict';
 const api = factory();
 if (typeof module === 'object' && module.exports) module.exports = api;
 root.AttentionMultihead = api;
 if (typeof document === 'undefined') return;
 if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => api.mountAll(), { once: true });
 else api.mountAll();
})(typeof globalThis === 'object' ? globalThis : this, function() {
 'use strict';
 const SQRT2 = Math.SQRT2;
 const X = [1, 1];
 const K = [[1, 0], [0, 1]];
 const V = [[2, 0], [0, 2]];
 const WQ = [
  [[1, 0], [0, 0]],
  [[0, 0], [0, 1]]
 ];
 const WO = [
  [0.5, 0],
  [0, 0.5],
  [0.5, 0],
  [0, 0.5]
 ];
 const POSITIONS = ['p1', 'p2'];
 const STEP_META = [
  { label: '输入', title: '同一个 x 进入两个头', duration: 850, hold: 480,
    action: 'x=[1,1] 是投影前输入；它现在分叉到两个头，但 q0、q1 还没有产生。' },
  { label: '投影', title: '同一 x，两份不同的 Query', duration: 1150, hold: 520,
    action: '两个头各自乘 Wq；投影互相独立，动画依次展示只为便于观察。x 相同、投影不同，所以 q0=[1,0]、q1=[0,1]。' },
  { label: '打分', title: '每个 q 与同一组 K 比较', duration: 1000, hold: 520,
    action: '两头各算自己的分数 s=q·K/√2；头 0 偏向 p1，头 1 偏向 p2。' },
  { label: '归一化', title: '各自 softmax，得到两套权重', duration: 1150, hold: 540,
    action: 'e^s 分别归一化：头 0 权重约为 [66.98%,33.02%]，头 1 约为 [33.02%,66.98%]。' },
  { label: '加权', title: 'w·V 汇成各头自己的结果', duration: 1450, hold: 600,
    action: '每个头分别把两个位置的 w×V 逐维相加，得到 h0 与 h1；两个结果不共享。' },
  { label: 'concat', title: '四个坐标原样进入四个槽', duration: 1550, hold: 680,
    action: '按 [h0_0,h0_1,h1_0,h1_1] 排成 4 维；槽位只改变位置，不改数值，也不相加。' },
  { label: 'Wo', title: '只有 Wo 把 4 维组合成 2 维', duration: 1550, hold: 0,
    action: 'Wo 读取四个坐标，投回 y0、y1。本例 Wo 特选，不能推广成固定平均。' }
 ];
 const HEAD_META = [
  { name: '头 0', color: '0', q: 'q0', h: 'h0' },
  { name: '头 1', color: '1', q: 'q1', h: 'h1' }
 ];
 // Motion is serialized inside each stage: at most one source→destination
 // actor is in flight.  A short gap between legs keeps tokens from touching,
 // while the windows still finish before the stage ends so the settled state
 // is visible before the next step.  p is the eased stage progress.
 const MOTION_WINDOWS = {
  proj: [[0, 0.46], [0.50, 0.96]],
  concat: [[0, 0.21], [0.24, 0.45], [0.48, 0.69], [0.72, 0.93]],
  wo: [[0, 0.21], [0.24, 0.45], [0.48, 0.69], [0.72, 0.93]]
 };

 const dot = (a, b) => a.reduce((sum, x, i) => sum + x * (b[i] || 0), 0);
 const sum = a => a.reduce((total, x) => total + x, 0);
 const matVec = (x, w) => w[0].map((_, j) => x.reduce((total, v, i) => total + v * w[i][j], 0));
 const clamp = n => Math.max(0, Math.min(1, n));
 const lerp = (a, b, t) => a + (b - a) * t;
 const ease = t => t * t * (3 - 2 * t);
 const legAt = (stageP, windows, i) => {
  const [start, end] = windows[i];
  return {
   start, end,
   local: clamp((stageP - start) / (end - start)),
   started: stageP > start,
   arrived: stageP >= end - 1e-9
  };
 };
 const routeD = (from, to, laneY) => `M${from.x.toFixed(1)} ${from.y.toFixed(1)}`
  + ` C${from.x.toFixed(1)} ${laneY.toFixed(1)} ${to.x.toFixed(1)} ${laneY.toFixed(1)} ${to.x.toFixed(1)} ${to.y.toFixed(1)}`;
 const routePoint = (from, to, laneY, t) => {
  const u = 1 - t;
  return {
   x: u * u * u * from.x + 3 * u * u * t * from.x + 3 * u * t * t * to.x + t * t * t * to.x,
   y: u * u * u * from.y + 3 * u * u * t * laneY + 3 * u * t * t * laneY + t * t * t * to.y
  };
 };
 const fmt = (v, digits = 2) => '[' + v.map(x => Number(x.toFixed(digits))).join(', ') + ']';
 const fmt4 = v => v.map(x => x.toFixed(4)).join(', ');
 const pct = w => (w * 100).toFixed(1) + '%';
 const matInline = w => '[' + w.map(row => '[' + row.join(',') + ']').join(',') + ']';
 const setText = (el, value) => { if (el && el.textContent !== value) el.textContent = value; };

 function example() {
  const heads = WQ.map((wq, h) => {
   const q = matVec(X, wq);
   const scores = K.map(k => dot(q, k) / SQRT2);
   const exps = scores.map(x => Math.exp(x));
   const denominator = sum(exps);
   const weights = exps.map(x => x / denominator);
   const contributions = V.map((v, i) => v.map(x => weights[i] * x));
   const result = [0, 1].map(j => sum(contributions.map(c => c[j])));
   return { index: h, wq, q, k: K, v: V, scores, exps, denominator, weights, contributions, result };
  });
  const concat = heads.flatMap(head => head.result);
  const output = matVec(concat, WO);
  return {
   kind: 'attention-multihead', x: X, k: K, v: V, wq: WQ, wo: WO, heads, concat, output,
   width: 2, headDim: 2, headCount: 2, positions: POSITIONS,
   isPerformanceClaim: false, isAccuracyClaim: false,
   woIsSpecialAverage: true
  };
 }

 function headCard(head) {
  const index = head.index;
  const rows = POSITIONS.map((position, i) => `<div class="mh-pos" data-pos="${index}-${i}">
   <span class="mh-pos-name">${position}</span>
   <span class="mh-pos-k">K <code>${fmt(head.k[i])}</code></span>
   <span class="mh-score" data-score="${index}-${i}" data-ready="false">s ?</span>
   <span class="mh-weight" data-weight="${index}-${i}" data-ready="false"><i data-wfill="${index}-${i}"></i><b>w ?</b></span>
   <span class="mh-contrib" data-contrib="${index}-${i}" data-ready="false">w·V = ?</span>
  </div>`).join('');
  return `<article class="mh-head" data-head-card="${index}">
   <header class="mh-head-head">
    <span class="mh-dot" aria-hidden="true"></span>
    <span class="mh-head-name">${HEAD_META[index].name}</span>
    <span class="mh-head-role">Query ${HEAD_META[index].q}</span>
    <code class="mh-wq">Wq${index} ${matInline(head.wq)}</code>
   </header>
   <div class="mh-qrow"><span class="mh-qchip" data-q="${index}" data-ready="false"><small>${HEAD_META[index].q}</small><b>待投影</b></span><span class="mh-qhint">x · Wq${index}</span></div>
   <div class="mh-poslist">${rows}</div>
   <div class="mh-result" data-result="${index}" data-ready="false">
    <div class="mh-result-label"><span>${HEAD_META[index].h} = Σ w·V</span><small data-result-note>等待 w×V</small></div>
    <div class="mh-coords">
     <span class="mh-coord" data-coord-src="${index}-0" data-ready="false"><small>${HEAD_META[index].h}_0</small><b data-coord-val="${index}-0">?</b><i class="mh-coord-fill" data-coord-fill="${index}-0"></i></span>
     <span class="mh-coord" data-coord-src="${index}-1" data-ready="false"><small>${HEAD_META[index].h}_1</small><b data-coord-val="${index}-1">?</b><i class="mh-coord-fill" data-coord-fill="${index}-1"></i></span>
    </div>
   </div>
  </article>`;
 }

 function shell(data) {
  const slots = data.concat.map((value, i) => {
   const head = i < 2 ? 0 : 1;
   return `<span class="mh-slot" data-slot="${i}" data-head="${head}" data-ready="false"><small>${i + 1} · h${head}_${i % 2}</small><b>待放入</b></span>`;
  }).join('');
  const woRows = [0, 1].map(row => `<tr><th>y${row}</th>${WO.map((w, col) => `<td data-wo-cell="${row}-${col}" class="${w[row] ? 'is-live' : 'is-zero'}">${w[row] ? '.5' : '0'}</td>`).join('')}</tr>`).join('');
  return `<div class="mh-toolbar">
   <div class="mh-controls" role="group" aria-label="多头过程控制">
    <button type="button" data-mh-action="prev">上一步</button>
    <button type="button" data-mh-action="play" aria-pressed="false"><span data-play-label>播放</span></button>
    <button type="button" data-mh-action="next">下一步</button>
    <button type="button" data-mh-action="reset">重置</button>
   </div>
   <output class="mh-step-out" data-mh-step-out aria-live="polite">步骤 1 / 7 · 输入</output>
  </div>
  <p class="mh-action" data-mh-caption>${STEP_META[0].action}</p>
  <div class="mh-workspace">
   <div class="mh-stage" data-mh-stage aria-label="多头注意力过程图">
    <div class="mh-source">
     <span class="mh-source-kicker">投影前输入</span>
     <span class="mh-x-node" data-mh-x><small>x</small><span>1</span><span>1</span></span>
     <span class="mh-source-note">两个头共用同一个 x</span>
    </div>
    <svg class="mh-fan" viewBox="0 0 100 22" preserveAspectRatio="none" aria-hidden="true">
     <path data-fan="0" pathLength="100" d="M50 0 C50 11 24 9 24 22"/>
     <path data-fan="1" pathLength="100" d="M50 0 C50 11 76 9 76 22"/>
    </svg>
    <div class="mh-heads">${data.heads.map(headCard).join('')}</div>
    <div class="mh-merge">
     <div class="mh-concat">
      <div class="mh-merge-head"><span>concat · 4 维</span><small>按头序原样排入</small></div>
      <div class="mh-slots">${slots}</div>
     </div>
     <div class="mh-merge-arrow" aria-hidden="true">→</div>
     <div class="mh-wo">
      <div class="mh-merge-head mh-wo-head"><span>Wo · 系数按输出列出</span><small>4 输入 → 2 输出 · Wo 转置视图</small></div>
      <table class="mh-wotable" aria-label="Wo 转置视图：每一行是一个输出，列出四个输入坐标的系数"><thead><tr><th></th><th class="h0">h0_0</th><th class="h0">h0_1</th><th class="h1">h1_0</th><th class="h1">h1_1</th></tr></thead><tbody>${woRows}</tbody></table>
      <div class="mh-outs">
       <div class="mh-out" data-out="0" data-ready="false"><span>y0</span><b data-out-val="0">?</b><small data-out-sum>待 Wo</small></div>
       <div class="mh-out" data-out="1" data-ready="false"><span>y1</span><b data-out-val="1">?</b><small data-out-sum>待 Wo</small></div>
      </div>
     </div>
    </div>
    <div class="mh-flight" data-mh-flight aria-hidden="true">
     <svg class="mh-route-svg" data-mh-route-svg focusable="false">
      <g data-mh-route hidden>
       <path data-mh-route-path></path>
       <circle data-mh-route-start r="3"></circle>
       <circle data-mh-route-end r="4"></circle>
      </g>
     </svg>
    </div>
   </div>
   <aside class="mh-detail" aria-label="当前步骤说明">
    <div class="mh-detail-top"><span data-detail-kicker>步骤 1 / 7 · 输入</span><span data-detail-state>已完成</span></div>
    <h3 data-detail-title>同一个 x 进入两个头</h3>
    <div class="mh-progress" aria-hidden="true"><i data-detail-progress></i></div>
    <div class="mh-detail-body" data-detail-body></div>
    <p class="mh-detail-foot">比较头 0/1 会暂停在当前过程点；concat 与 Wo 仍保留来源标签。</p>
   </aside>
  </div>
  <div class="mh-underbar">
   <div class="mh-focus" role="group" aria-label="比较两个头">
    <span>比较</span>
    <button type="button" data-mh-focus="all" aria-pressed="true">并排</button>
    <button type="button" data-mh-focus="0" aria-pressed="false"><i class="mh-dot" aria-hidden="true"></i>头 0</button>
    <button type="button" data-mh-focus="1" aria-pressed="false"><i class="mh-dot" aria-hidden="true"></i>头 1</button>
    <small>点击比较会暂停</small>
   </div>
   <a class="mh-return" href="#lesson-animation" data-mh-return>回到单头 Query ↑</a>
  </div>
  <p class="mh-boundary">独立教学算例：2 头、每头 2 维、模型宽度 2。为了让差异集中在 Wq，本例两头共用同一组 K/V；真实模型各头通常有自己的 K/V 投影。教学 Wo 是特选矩阵，只因这组数值恰好等于两头同坐标平均；一般 output projection 可混合全部 4 个坐标，不是固定平均。数值不声称多头必然提升正确率或性能，也不冒充同一次完整层计算。</p>
  <details class="mh-math">
   <summary>完整矩阵与逐项数值（展开核对，不代替上面的过程）</summary>
   <div class="mh-math-body">
    <p class="mh-mono">x=${fmt(data.x)}　K=[[1,0],[0,1]]　V=[[2,0],[0,2]]</p>
    <p class="mh-mono">Wq0=[[1,0],[0,0]]　Wq1=[[0,0],[0,1]]</p>
    <p class="mh-mono">Wo=[[.5,0],[0,.5],[.5,0],[0,.5]]</p>
    <div class="mh-math-scroll"><table class="mh-math-table"><thead><tr><th>头</th><th>q</th><th>s</th><th>e^s</th><th>w</th><th>h</th></tr></thead><tbody>
     ${data.heads.map(head => `<tr><th>h${head.index}</th><td>${fmt(head.q)}</td><td>${fmt(head.scores)}</td><td>${fmt(head.exps, 3)}</td><td>[${head.weights.map(w => w.toFixed(8)).join(', ')}]</td><td>[${fmt4(head.result)}]</td></tr>`).join('')}
    </tbody></table></div>
    <p class="mh-mono">concat=[${fmt4(data.concat)}]　y=[${fmt4(data.output)}]</p>
    <p class="mh-note">这是独立教学数值；没有运行模型服务、GPU 或性能实验。</p>
   </div>
  </details>
  <span class="sr-only" data-mh-announce aria-live="polite"></span>
 </div>`;
 }

 function renderDetail(step, state, data) {
  const h0 = data.heads[0], h1 = data.heads[1];
  const pending = '…';
  switch (step) {
   case 0:
    return `<p>先固定投影前输入：</p><div class="mh-eq"><code>x = [1, 1]</code></div><p class="mh-note">x 不是 Query。它同时进入两个头，下一步才分别乘 Wq0、Wq1；K/V 也还没有参与。</p>`;
   case 1:
    return `<p>同一个 x，两个不同的投影：</p>
     <div class="mh-eq"><code>q0 = x·Wq0 = ${state.qArrived[0] ? fmt(h0.q) : pending}</code></div>
     <div class="mh-eq"><code>q1 = x·Wq1 = ${state.qArrived[1] ? fmt(h1.q) : pending}</code></div>
     <p class="mh-note">q0、q1 才是各头的 Query；x 是它们共同的投影前输入。两个副本依次飞出只为便于观察，不表示头 0 是头 1 的前置条件；两头投影彼此独立，可以并行计算，这里不讨论性能收益。</p>`;
   case 2:
    return `<p>两头各自与同一组 K 比较，除以 √2：</p>
     <div class="mh-eq"><code>头 0：s = q0·K/√2 = ${state.scoreReady[0].every(Boolean) ? fmt(h0.scores) : pending}</code></div>
     <div class="mh-eq"><code>头 1：s = q1·K/√2 = ${state.scoreReady[1].every(Boolean) ? fmt(h1.scores) : pending}</code></div>
     <p class="mh-note">s 还是分数，不是权重。下一段才做 e^s 与归一化。</p>`;
   case 3:
    const w0 = state.weightReady[0].every(Boolean) ? `[${h0.weights.map(w => pct(w)).join(', ')}]` : pending;
    const w1 = state.weightReady[1].every(Boolean) ? `[${h1.weights.map(w => pct(w)).join(', ')}]` : pending;
    return `<p>每个头独立 softmax：</p>
     <div class="mh-eq"><code>头 0：e^s=${fmt(h0.exps, 3)}，Σ=${h0.denominator.toFixed(3)}<br>w = ${w0}</code></div>
     <div class="mh-eq"><code>头 1：e^s=${fmt(h1.exps, 3)}，Σ=${h1.denominator.toFixed(3)}<br>w = ${w1}</code></div>
     <p class="mh-note">${state.weightReady[0].every(Boolean) && state.weightReady[1].every(Boolean) ? '两套权重都已形成，它们分别属于各自的头。' : '归一化进行中；权重未完成前不显示为最终结果。'}</p>`;
   case 4:
    return `<p>每个头逐位置乘 V，再逐维相加：</p>
     <div class="mh-eq"><code>头 0：${state.contribReady[0][0] ? `${h0.weights[0].toFixed(4)}×${fmt(h0.v[0])} = ${fmt(h0.contributions[0], 4)}` : pending}<br>${state.contribReady[0][1] ? `${h0.weights[1].toFixed(4)}×${fmt(h0.v[1])} = ${fmt(h0.contributions[1], 4)}` : pending}<br>h0 = ${state.resultReady[0] ? `[${fmt4(h0.result)}]` : pending}</code></div>
     <div class="mh-eq"><code>头 1：${state.contribReady[1][0] ? `${h1.weights[0].toFixed(4)}×${fmt(h1.v[0])} = ${fmt(h1.contributions[0], 4)}` : pending}<br>${state.contribReady[1][1] ? `${h1.weights[1].toFixed(4)}×${fmt(h1.v[1])} = ${fmt(h1.contributions[1], 4)}` : pending}<br>h1 = ${state.resultReady[1] ? `[${fmt4(h1.result)}]` : pending}</code></div>
     <p class="mh-note">两个头各自形成结果；这里还没有 concat，也没有 Wo。</p>`;
   case 5:
    const slotStatus = i => state.slotArrived[i] ? '已放入' : state.concat[i].local > 0 ? '放入中' : '待放入';
    return `<p>四个坐标按头序进入四个槽：</p>
     <ol class="mh-slot-list">
      ${data.concat.map((value, i) => `<li data-head="${i < 2 ? 0 : 1}"><span>槽 ${i + 1}</span><code>h${i < 2 ? 0 : 1}_${i % 2} = ${value.toFixed(4)}</code><b>${slotStatus(i)}</b></li>`).join('')}
     </ol>
     <p class="mh-note">只改变位置，不改写数值，也不做加法。当前 ${state.slotCount}/4 个槽已放入；未放入的槽会区分“待放入”和“放入中”。</p>`;
   case 6:
    return `<p>Wo 只读取指定坐标并组合成两个输出：</p>
     <div class="mh-eq"><code>y0 = .5·h0_0 + 0·h0_1 + .5·h1_0 + 0·h1_1 = ${state.outReady[0] ? data.output[0].toFixed(2) : pending}</code></div>
     <div class="mh-eq"><code>y1 = 0·h0_0 + .5·h0_1 + 0·h1_0 + .5·h1_1 = ${state.outReady[1] ? data.output[1].toFixed(2) : pending}</code></div>
     <p class="mh-wo-paths">${[0, 1, 2, 3].map(i => `<span data-head="${i < 2 ? 0 : 1}" data-arrived="${state.woArrived[i]}">${i < 2 ? 'h0' : 'h1'}_${i % 2} × .5 → y${i % 2}</span>`).join('')}</p>
     <p class="mh-note">这组 Wo 是特选：恰好等于两头同坐标平均。一般 output projection 是学出来的 4×2 矩阵，可混合全部 4 个坐标，不能说成固定平均。</p>`;
   default: return '';
  }
 }

 function mount(root) {
  if (!root || root.dataset.mhMounted === 'true') return;
  root.dataset.mhMounted = 'true';
  const data = example();
  root.innerHTML = shell(data);
  const $ = selector => root.querySelector(selector);
  const $$ = selector => Array.from(root.querySelectorAll(selector));
  const stage = $('[data-mh-stage]');
  const fan0 = $('[data-fan="0"]'), fan1 = $('[data-fan="1"]');
  const qChips = [0, 1].map(i => $(`[data-q="${i}"]`));
  const scores = $$('[data-score]');
  const weights = $$('[data-weight]');
  const contribs = $$('[data-contrib]');
  const coordVals = $$('[data-coord-val]');
  const coordFills = $$('[data-coord-fill]');
  const coordSrcs = $$('[data-coord-src]');
  const slots = $$('[data-slot]');
  const woCells = $$('[data-wo-cell]');
  const outs = [0, 1].map(i => $(`[data-out="${i}"]`));
  const outVals = $$('[data-out-val]');
  const outSums = $$('[data-out-sum]');
  const playBtn = $('[data-mh-action="play"]');
  const playLabel = $('[data-play-label]');
  const stepOut = $('[data-mh-step-out]');
  const caption = $('[data-mh-caption]');
  const announceEl = $('[data-mh-announce]');
  const detailKicker = $('[data-detail-kicker]');
  const detailState = $('[data-detail-state]');
  const detailTitle = $('[data-detail-title]');
  const detailBody = $('[data-detail-body]');
  const detailProgress = $('[data-detail-progress]');
  const focusButtons = $$('[data-mh-focus]');
  const flightLayer = $('[data-mh-flight]');
  const routeSvg = $('[data-mh-route-svg]');
  const routeGroup = $('[data-mh-route]');
  const routePath = $('[data-mh-route-path]');
  const routeStart = $('[data-mh-route-start]');
  const routeEnd = $('[data-mh-route-end]');
  const reduceQuery = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
  let reduced = !!(reduceQuery && reduceQuery.matches);
  let step = 0, progress = 1, mode = 'idle';
  let raf = 0, lastTs = 0, holdUntil = 0;
  let focus = 'all', lastDetailKey = '', lastAnnounce = '';

  const flyTokens = {};
  ['proj-0', 'proj-1', 'concat-0', 'concat-1', 'concat-2', 'concat-3', 'wo-0', 'wo-1', 'wo-2', 'wo-3'].forEach(key => {
   const token = document.createElement('span');
   token.className = 'mh-fly';
   token.dataset.fly = key;
   const index = Number(key.split('-')[1]);
   token.dataset.head = key.startsWith('concat') || key.startsWith('wo') ? String(index < 2 ? 0 : 1) : String(index);
   token.hidden = true;
   flightLayer.appendChild(token);
   flyTokens[key] = token;
  });
  const positions = { valid: false, stage: null, source: null, q: [], coords: [], slots: [], outs: [] };

  function stageP(k) { return step > k ? 1 : step < k ? 0 : clamp(progress); }
  function eased(k) { return ease(stageP(k)); }

  // One readiness model feeds both the main board and the detail panel.  Every
  // "ready" bit is the same predicate the moving actor uses to finish its leg,
  // so a value cannot appear in one place before the token actually arrives.
  function computeState() {
   const proj = [0, 1].map(i => legAt(eased(1), MOTION_WINDOWS.proj, i));
   const qArrived = proj.map(leg => leg.arrived);

   const scoreP = eased(2);
   const scoreLocal = [0, 1].map(i => clamp((scoreP - i * 0.12) / 0.76));
   const scoreReady = [0, 1].map(() => scoreLocal.map(local => local >= 0.95));

   const weightP = eased(3);
   const weightLocal = [0, 1].map(i => clamp((weightP - i * 0.14) / 0.72));
   const weightReady = [0, 1].map(() => weightLocal.map(local => local >= 0.95));

   const valueP = eased(4);
   const sumP = clamp((valueP - 0.7) / 0.28);
   const contribLocal = [0, 1].map(i => clamp((valueP - (i ? 0.34 : 0.04)) / 0.3));
   const contribReady = [0, 1].map(() => contribLocal.map(local => local >= 0.95));
   const resultReady = [0, 1].map(() => sumP >= 0.98);

   const concat = [0, 1, 2, 3].map(i => legAt(eased(5), MOTION_WINDOWS.concat, i));
   const slotArrived = concat.map(leg => leg.arrived);
   const slotCount = slotArrived.filter(Boolean).length;

   const wo = [0, 1, 2, 3].map(i => legAt(eased(6), MOTION_WINDOWS.wo, i));
   const woArrived = wo.map(leg => leg.arrived);
   const outReady = [0, 1].map(row => [0, 1, 2, 3].every(i => WO[i][row] === 0 || woArrived[i]));
   const woOutCounts = [0, 1].map(row => [0, 1, 2, 3].filter(i => WO[i][row] !== 0 && woArrived[i]).length);

   const bits = list => list.map(value => value ? 1 : 0).join('');
   const key = [
    step,
    bits(qArrived),
    scoreReady.map(bits).join(','),
    weightReady.map(bits).join(','),
    contribReady.map(bits).join(','),
    bits(resultReady),
    bits(slotArrived),
    bits(concat.map(leg => leg.started)),
    bits(woArrived),
    bits(outReady)
   ].join('|');

   return {
    proj, qArrived, scoreLocal, scoreReady, weightLocal, weightReady,
    contribLocal, contribReady, resultReady, sumP,
    concat, slotArrived, slotCount,
    wo, woArrived, woOutCounts, outReady,
    key
   };
  }

  function announce(message) {
   if (message === lastAnnounce) return;
   lastAnnounce = message;
   setText(announceEl, message);
  }
  function syncPlay() {
   const running = mode === 'play' || mode === 'step';
   const label = running ? '暂停' : (step === 6 && progress >= 1 ? '重播' : '播放');
   setText(playLabel, label);
   playBtn.setAttribute('aria-pressed', String(running));
   root.dataset.playing = String(running);
   root.dataset.mode = mode;
  }
  function updateCaption() {
   setText(caption, STEP_META[step].action);
   setText(stepOut, `步骤 ${step + 1} / 7 · ${STEP_META[step].label}`);
   setText(detailKicker, `步骤 ${step + 1} / 7 · ${STEP_META[step].label}`);
   setText(detailTitle, STEP_META[step].title);
  }
  function updateDetail(state) {
   // Re-render on every discrete arrival/readiness change, not on a rounded
   // progress bucket.  Motion can change the count between any two buckets.
   if (state.key === lastDetailKey) return;
   lastDetailKey = state.key;
   detailBody.innerHTML = renderDetail(step, state, data);
   root.dataset.detailStep = String(step + 1);
  }
  function updateDetailState() {
   const state = mode === 'play' ? '播放中' : mode === 'step' ? '单步中' : progress >= 0.999 ? '已完成' : '已暂停';
   setText(detailState, state);
  }

  function measure() {
   if (!stage) return;
   const base = stage.getBoundingClientRect();
   if (!base.width || !base.height) { positions.valid = false; return; }
   const box = el => {
    const r = el.getBoundingClientRect();
    return {
     x: r.left - base.left + r.width / 2,
     y: r.top - base.top + r.height / 2,
     top: r.top - base.top, bottom: r.bottom - base.top,
     left: r.left - base.left, right: r.right - base.left,
     w: r.width, h: r.height
    };
   };
   positions.stage = { w: base.width, h: base.height };
   positions.source = box($('[data-mh-x]'));
   positions.q = qChips.map(box);
   positions.coords = coordSrcs.map(box);
   positions.slots = slots.map(box);
   positions.outs = outs.map(box);
   if (routeSvg) {
    routeSvg.setAttribute('viewBox', `0 0 ${base.width} ${base.height}`);
    routeSvg.setAttribute('width', String(base.width));
    routeSvg.setAttribute('height', String(base.height));
   }
   positions.valid = true;
  }
  function scheduleMeasure() {
   requestAnimationFrame(() => { measure(); renderFlight(); });
  }

  function setFly(key, visible, x, y, opacity, content) {
   const token = flyTokens[key];
   if (!token) return;
   if (!visible) { token.hidden = true; return; }
   if (token.dataset.content !== content) { token.innerHTML = content; token.dataset.content = content; }
   token.hidden = false;
   token.style.opacity = opacity.toFixed(3);
   token.style.transform = `translate3d(${x.toFixed(1)}px,${y.toFixed(1)}px,0) translate(-50%,-50%)`;
  }
  function showRoute(spec) {
   if (!routeGroup || !routePath || !spec) return;
   routeGroup.removeAttribute('hidden');
   routeGroup.dataset.head = String(spec.head);
   routePath.setAttribute('d', spec.d);
   [routeStart, routeEnd].forEach((circle, i) => {
    if (!circle) return;
    const point = i ? spec.to : spec.from;
    circle.setAttribute('cx', point.x.toFixed(1));
    circle.setAttribute('cy', point.y.toFixed(1));
   });
  }
  function hideRoute() {
   if (routeGroup) routeGroup.setAttribute('hidden', '');
  }

  function renderFlight() {
   if (!positions.valid) {
    Object.values(flyTokens).forEach(token => { token.hidden = true; });
    hideRoute();
    return;
   }
   const state = computeState();
   let activeRoute = null;
   const showLeg = (key, leg, from, to, laneY, head, content) => {
    if (!(leg.local > 0 && leg.local < 1)) { flyTokens[key].hidden = true; return; }
    const point = routePoint(from, to, laneY, ease(leg.local));
    const opacity = Math.min(1, leg.local * 8, (1 - leg.local) * 8);
    setFly(key, true, point.x, point.y, opacity, content);
    activeRoute = { from, to, laneY, head, d: routeD(from, to, laneY) };
   };

   [0, 1].forEach(i => {
    const leg = state.proj[i];
    const from = positions.source, to = positions.q[i];
    const laneY = Math.max(6, Math.min(from.y, to.y) - 12);
    const content = leg.local < 0.45
     ? `<small>x → ${HEAD_META[i].q}</small><b>[1,1]</b>`
     : `<small>${HEAD_META[i].q}</small><b>${leg.local < 0.9 ? '…' : fmt(data.heads[i].q)}</b>`;
    showLeg('proj-' + i, leg, from, to, laneY, i, content);
   });

   [0, 1, 2, 3].forEach(i => {
    const leg = state.concat[i];
    const from = positions.coords[i];
    const to = { x: positions.slots[i].x, y: positions.slots[i].top - 18 };
    const laneY = Math.max(from.y, to.y) + 12;
    const content = `<small>h${i < 2 ? 0 : 1}_${i % 2} → 槽 ${i + 1}</small><b>${data.concat[i].toFixed(4)}</b>`;
    showLeg('concat-' + i, leg, from, to, laneY, i < 2 ? 0 : 1, content);
   });

   [0, 1, 2, 3].forEach(i => {
    const leg = state.wo[i];
    const row = i % 2;
    const from = positions.slots[i];
    const out = positions.outs[row];
    // The Wo actor travels in the stage's bottom lane and docks just below the
    // output box, so neither the coefficient table nor the target value is
    // covered at the moment of arrival.
    const to = { x: out.x, y: out.bottom + 18 };
    const laneY = Math.min(positions.stage.h - 24, Math.max(from.y, to.y) + 20);
    const content = `<small>h${i < 2 ? 0 : 1}_${i % 2} → y${row}</small><b>×.5</b>`;
    showLeg('wo-' + i, leg, from, to, laneY, i < 2 ? 0 : 1, content);
   });

   if (activeRoute) showRoute(activeRoute);
   else hideRoute();
  }

  function render() {
   const p = clamp(progress);
   const fanP = eased(0);
   if (fan0) fan0.style.strokeDashoffset = String(100 - 100 * fanP);
   if (fan1) fan1.style.strokeDashoffset = String(100 - 100 * fanP);
   root.dataset.step = String(step);
   root.dataset.progress = p.toFixed(3);
   root.dataset.mhFocus = focus;
   root.style.setProperty('--mh-progress', p.toFixed(3));
   const state = computeState();

   qChips.forEach((chip, i) => {
    const ready = state.qArrived[i];
    const local = state.proj[i].local;
    setText(chip.querySelector('b'), ready ? fmt(data.heads[i].q) : local > 0 ? '投影中…' : '待投影');
    chip.dataset.ready = String(ready);
    chip.classList.toggle('is-busy', local > 0 && !ready);
    chip.classList.toggle('is-receiving', local > 0 && !ready);
   });
   root.dataset.q0 = state.qArrived[0] ? fmt(data.heads[0].q) : '';
   root.dataset.q1 = state.qArrived[1] ? fmt(data.heads[1].q) : '';

   scores.forEach(el => {
    const [h, i] = el.dataset.score.split('-').map(Number);
    const local = state.scoreLocal[i];
    const ready = state.scoreReady[h][i];
    setText(el, ready ? 's ' + data.heads[h].scores[i].toFixed(2) : local > 0.04 ? 's …' : 's ?');
    el.dataset.ready = String(ready);
    el.classList.toggle('is-busy', local > 0.04 && !ready);
   });

   weights.forEach(el => {
    const [h, i] = el.dataset.weight.split('-').map(Number);
    const local = state.weightLocal[i];
    const ready = state.weightReady[h][i];
    el.querySelector('i').style.width = (local * data.heads[h].weights[i] * 100).toFixed(2) + '%';
    setText(el.querySelector('b'), ready ? pct(data.heads[h].weights[i]) : local > 0.04 ? '…' : 'w ?');
    el.dataset.ready = String(ready);
    el.classList.toggle('is-busy', local > 0.04 && !ready);
   });

   data.heads.forEach(head => {
    const resultReady = state.resultReady[head.index];
    contribs.filter(el => el.dataset.contrib.startsWith(head.index + '-')).forEach((el, i) => {
     const ready = state.contribReady[head.index][i];
     setText(el, ready ? `w·V = ${fmt(head.contributions[i], 4)}` : `w·V = ${state.sumP > 0.02 ? '…' : '?'}`);
     el.dataset.ready = String(ready);
    });
    const note = root.querySelector(`[data-result="${head.index}"] [data-result-note]`);
    setText(note, resultReady ? '逐维相加完成' : state.sumP > 0.02 ? '逐维相加中' : '等待 w×V');
    const result = root.querySelector(`[data-result="${head.index}"]`);
    result.dataset.ready = String(resultReady);
    head.result.forEach((value, j) => {
     const key = `${head.index}-${j}`;
     const slotIndex = head.index * 2 + j;
     const coord = root.querySelector(`[data-coord-src="${key}"]`);
     const moved = step > 5 || (step === 5 && state.concat[slotIndex].local > 0.02);
     coord.dataset.value = value.toFixed(4);
     if (moved) {
      setText(coord.querySelector('b'), `→ ${slotIndex + 1}`);
      coord.dataset.moved = 'true';
      coord.dataset.slotIndex = String(slotIndex + 1);
      coord.title = `${HEAD_META[head.index].h}_${j} = ${value.toFixed(4)} → 槽 ${slotIndex + 1}`;
     } else {
      setText(coord.querySelector('b'), resultReady ? value.toFixed(4) : state.sumP > 0.02 ? '…' : '?');
      coord.dataset.moved = 'false';
      coord.dataset.slotIndex = '';
      coord.removeAttribute('title');
     }
     coord.dataset.ready = String(resultReady);
     const fill = coordFills.find(el => el.dataset.coordFill === key);
     if (fill) fill.style.width = (resultReady || state.sumP > 0 ? Math.min(1, state.sumP) * (value / 1.4) * 100 : 0).toFixed(1) + '%';
    });
   });
   root.dataset.h0 = state.resultReady[0] ? fmt(data.heads[0].result) : '';
   root.dataset.h1 = state.resultReady[1] ? fmt(data.heads[1].result) : '';

   slots.forEach((slot, i) => {
    const ready = state.slotArrived[i];
    const local = state.concat[i].local;
    const label = ready ? data.concat[i].toFixed(4) : local > 0 ? '放入中' : '待放入';
    setText(slot.querySelector('b'), label);
    slot.dataset.ready = String(ready);
    slot.classList.toggle('is-busy', !ready && local > 0);
    slot.classList.toggle('is-receiving', !ready && local > 0);
   });
   root.dataset.concatCount = String(state.slotCount);

   woCells.forEach(cell => {
    const [row, col] = cell.dataset.woCell.split('-').map(Number);
    const live = WO[col][row] !== 0;
    cell.classList.toggle('is-active', live && state.wo[col].local >= 0.8);
   });
   outs.forEach((out, row) => {
    const ready = state.outReady[row];
    const sources = [0, 1, 2, 3].filter(i => WO[i][row] !== 0);
    const arrived = state.woOutCounts[row];
    const incoming = sources.some(i => {
     const local = state.wo[i].local;
     return local > 0 && local < 1;
    });
    setText(outVals[row], ready ? data.output[row].toFixed(2) : '?');
    setText(outSums[row], ready ? '由 4 维合并' : (arrived > 0 || incoming) ? `已接入 ${arrived}/${sources.length}` : '待 Wo');
    out.dataset.ready = String(ready);
    out.classList.toggle('is-busy', !ready && (arrived > 0 || incoming));
    out.classList.toggle('is-receiving', !ready && (arrived > 0 || incoming));
   });
   root.dataset.output = state.outReady[0] && state.outReady[1] ? `[${data.output.map(v => v.toFixed(2)).join(',')}]` : '';

   detailProgress.style.width = (stageP(step) * 100).toFixed(1) + '%';
   updateDetail(state);
   updateDetailState();
   syncPlay();
   renderFlight();
  }

  function scheduleLoop() {
   if (!raf) { lastTs = 0; raf = requestAnimationFrame(tick); }
  }
  function tick(now) {
   raf = 0;
   if (mode === 'idle') { syncPlay(); return; }
   if (!lastTs) lastTs = now;
   const dt = Math.min(64, now - lastTs);
   lastTs = now;
   if (progress < 1) {
    const duration = reduced ? 1 : STEP_META[step].duration;
    progress = Math.min(1, progress + dt / duration);
    render();
   }
   if (progress >= 1) {
    if (mode === 'step') { mode = 'idle'; syncPlay(); render(); return; }
    if (step >= 6) { mode = 'idle'; syncPlay(); render(); return; }
    if (!holdUntil) holdUntil = now + (reduced ? 1350 : STEP_META[step].hold);
    if (now >= holdUntil) {
     step += 1;
     progress = 0;
     holdUntil = 0;
     lastTs = 0;
     updateCaption();
     announce(`${step + 1}/7 ${STEP_META[step].title}。${STEP_META[step].action}`);
     render();
     scheduleMeasure();
    }
   }
   raf = requestAnimationFrame(tick);
  }
  function enterStep(next, p, nextMode) {
   step = Math.max(0, Math.min(6, next));
   progress = clamp(p);
   mode = nextMode;
   holdUntil = 0;
   lastTs = 0;
   lastDetailKey = '';
   root.dataset.step = String(step);
   updateCaption();
   render();
   scheduleMeasure();
   if (mode !== 'idle') scheduleLoop();
  }
  function pause() {
   const wasRunning = mode !== 'idle';
   mode = 'idle';
   holdUntil = 0;
   lastTs = 0;
   if (raf) { cancelAnimationFrame(raf); raf = 0; }
   render();
   if (wasRunning) announce(`已暂停在第 ${step + 1} 步，过程点 ${Math.round(progress * 100)}%。`);
  }
  function play() {
   if (mode === 'play' || mode === 'step') { pause(); return; }
   if (step === 6 && progress >= 1) { step = 0; progress = 0; }
   else if (progress >= 1) { step = Math.min(6, step + 1); progress = 0; }
   mode = 'play';
   holdUntil = 0;
   lastTs = 0;
   updateCaption();
   announce(`开始播放：第 ${step + 1} 步，${STEP_META[step].title}。`);
   render();
   scheduleMeasure();
   scheduleLoop();
  }
  function next() {
   if (mode !== 'idle') { mode = 'idle'; if (raf) { cancelAnimationFrame(raf); raf = 0; } }
   if (step >= 6) { pause(); return; }
   enterStep(step + 1, 0, 'step');
   announce(`第 ${step + 1} 步，${STEP_META[step].title}。${STEP_META[step].action}`);
  }
  function prev() {
   if (mode !== 'idle') { mode = 'idle'; if (raf) { cancelAnimationFrame(raf); raf = 0; } }
   step = Math.max(0, step - 1);
   progress = 1;
   holdUntil = 0;
   lastDetailKey = '';
   updateCaption();
   render();
   scheduleMeasure();
   announce(`回到第 ${step + 1} 步，${STEP_META[step].title}。`);
  }
  function reset() {
   enterStep(0, 0, 'step');
   announce('重置到输入步骤，重新分叉。');
  }
  function setFocus(nextFocus) {
   pause();
   focus = nextFocus;
   focusButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.mhFocus === focus)));
   root.dataset.mhFocus = focus;
   announce(focus === 'all' ? '并排显示两个头，已暂停在当前过程点。' : `突出${focus === '0' ? '头 0' : '头 1'}，已暂停在当前过程点，便于比较。`);
   render();
  }

  root.addEventListener('click', event => {
   const action = event.target.closest('[data-mh-action]')?.dataset.mhAction;
   // root itself carries data-mh-focus for the CSS view state, so only the
   // actual comparison buttons may change focus/pause the player.
   const focusButton = event.target.closest('button[data-mh-focus]');
   const back = event.target.closest('[data-mh-return]');
   if (back) { pause(); return; }
   if (action === 'prev') prev();
   else if (action === 'next') next();
   else if (action === 'reset') reset();
   else if (action === 'play') play();
   if (focusButton) setFocus(focusButton.dataset.mhFocus);
  });
  root.addEventListener('keydown', event => {
   if (event.target.matches('button, a, input, select, summary')) return;
   if (event.key === 'ArrowRight') { event.preventDefault(); next(); }
   else if (event.key === 'ArrowLeft') { event.preventDefault(); prev(); }
   else if (event.key === ' ') { event.preventDefault(); play(); }
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  if (reduceQuery) {
   const onMotionChange = event => { reduced = event.matches; render(); };
   if (reduceQuery.addEventListener) reduceQuery.addEventListener('change', onMotionChange);
   else if (reduceQuery.addListener) reduceQuery.addListener(onMotionChange);
  }
  window.addEventListener('resize', scheduleMeasure, { passive: true });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(scheduleMeasure).catch(() => {});

  updateCaption();
  render();
  scheduleMeasure();
  announce(`步骤 1/7 ${STEP_META[0].title}。${STEP_META[0].action}`);
 }

 function mountAll() {
  document.querySelectorAll('[data-multihead]').forEach(mount);
 }

 return { example, STEP_META, mount, mountAll };
});
