/* Attention variants: one interaction surface for six different design moves.
 *
 * This file is deliberately data-driven. Each variant owns a small teaching
 * model and a mount function; the tab shell only knows the registry. Adding
 * a family should not require changing the other panels or the shared player.
 *
 * All values are independent teaching settings. They are not model weights,
 * GPU measurements, quality claims, or traces from a serving engine.
 */
(function(root, factory) {
 'use strict';
 const api = factory();
 if (typeof module === 'object' && module.exports) module.exports = api;
 root.AttentionVariants = api;
 if (typeof document === 'undefined') return;
 if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => api.mountAll(), { once: true });
 else api.mountAll();
})(typeof globalThis === 'object' ? globalThis : this, function() {
 'use strict';

 const SQRT2 = Math.SQRT2;
 const clamp = (n, min, max) => Math.max(min, Math.min(max, n));
 const sum = values => values.reduce((total, value) => total + value, 0);
 const dot = (a, b) => a.reduce((total, value, i) => total + value * (b[i] || 0), 0);
 // Matrix times column vector: every MLA teaching quantity below is a column
 // vector, so Mv is a row-by-coordinate dot product (not the row-vector xW form).
 const applyMatrix = (matrix, vector) => matrix.map(row => dot(row, vector));
 const transpose = matrix => matrix[0].map((_, j) => matrix.map(row => row[j]));
 const softmax = scores => {
  const max = Math.max(...scores);
  const exps = scores.map(score => Math.exp(score - max));
  const total = sum(exps);
  return exps.map(value => value / total);
 };
 const range = (start, end) => {
  const result = [];
  for (let value = start; value <= end; value += 1) result.push(value);
  return result;
 };
 const fmt = (value, digits = 2) => {
  const rounded = Number(value.toFixed(digits));
  return Object.is(rounded, -0) ? '0' : String(rounded);
 };
 const arr = (values, digits = 2) => '[' + values.map(value => fmt(value, digits)).join(', ') + ']';
 const pct = (value, digits = 1) => (value * 100).toFixed(digits) + '%';
 const list = values => values.length ? values.join(', ') : '无';
 const escapeHtml = text => String(text).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));

 /* ------------------------------------------------------------------ */
 /* Teaching models                                                     */
 /* ------------------------------------------------------------------ */

 const HEAD_Q = [[1, 0], [0, 1], [1, 1], [2, -1]];
 const HEAD_K = [[1, 0], [0, 1]];
 const HEAD_V = [[2, 0], [0, 2]];
 const HEAD_GROUPS = {
  mha: [[0], [1], [2], [3]],
  gqa: [[0, 1], [2, 3]],
  mqa: [[0, 1, 2, 3]]
 };
 const HEAD_MODE_LABEL = { mha: 'MHA', gqa: 'GQA', mqa: 'MQA' };

 function headsExample(mode = 'mha') {
  const key = Object.hasOwn(HEAD_GROUPS, mode) ? mode : 'mha';
  const groups = HEAD_GROUPS[key];
  const groupOf = [];
  groups.forEach((group, groupIndex) => group.forEach(head => { groupOf[head] = groupIndex; }));
  const heads = HEAD_Q.map((q, index) => {
   const scores = HEAD_K.map(k => dot(q, k) / SQRT2);
   const weights = softmax(scores);
   const contributions = HEAD_V.map((v, position) => v.map(value => weights[position] * value));
   const output = [0, 1].map(coordinate => sum(contributions.map(contribution => contribution[coordinate])));
   return {
    index, label: `Q${index}`, q, k: HEAD_K, v: HEAD_V,
    scores, weights, contributions, output, group: groupOf[index]
   };
  });
  return {
   kind: 'attention-heads', mode: key, modeLabel: HEAD_MODE_LABEL[key],
   qHeads: HEAD_Q.length, kvGroups: groups.length, groups, groupOf,
   heads, k: HEAD_K, v: HEAD_V,
   qProjectionsIndependent: true,
   kvValuesAreTeachingPlaceholders: true,
   isPerformanceClaim: false, isAccuracyClaim: false
  };
 }

 const MLA_Q = [1, 0];
 const MLA_C = [[1, 2], [3, 1]];
 const MLA_UK = [[1, 0], [0, 2]];
 const MLA_UV = [[1, 0], [1, 1]];

 function mlaExample() {
  // c_i, U_K c_i, U_K^T q, and U_V c_i are all column-vector operations.
  const k = MLA_C.map(c => applyMatrix(MLA_UK, c));
  const v = MLA_C.map(c => applyMatrix(MLA_UV, c));
  const qAbs = applyMatrix(transpose(MLA_UK), MLA_Q);
  const scoresExpanded = k.map(key => dot(MLA_Q, key) / SQRT2);
  const scoresAbsorbed = MLA_C.map(c => dot(qAbs, c) / SQRT2);
  const weights = softmax(scoresExpanded);
  const cMix = [0, 1].map(coordinate => sum(MLA_C.map((c, position) => weights[position] * c[coordinate])));
  const output = applyMatrix(MLA_UV, cMix);
  const outputExpanded = [0, 1].map(coordinate => sum(v.map((value, position) => weights[position] * value[coordinate])));
  return {
   kind: 'mla', q: MLA_Q, c: MLA_C, uk: MLA_UK, uv: MLA_UV,
   k, v, qAbs, scoresExpanded, scoresAbsorbed, weights, cMix, output, outputExpanded,
   scoresEqual: scoresExpanded.every((score, i) => Math.abs(score - scoresAbsorbed[i]) < 1e-12),
   outputEqual: output.every((value, i) => Math.abs(value - outputExpanded[i]) < 1e-12),
   columnVectorConvention: true,
   positionKeyCount: 2, positionCacheSeparate: true, ropeAbsorbed: false,
   isPerformanceClaim: false
  };
 }

 const SWA_MAX_T = 8;
 const SWA_MAX_W = 4;
 const SWA_DEFAULT_T = 6;
 const SWA_DEFAULT_W = 3;

 function swaExample(t = SWA_DEFAULT_T, w = SWA_DEFAULT_W) {
  const current = clamp(Math.round(Number(t) || 1), 1, SWA_MAX_T);
  const width = clamp(Math.round(Number(w) || 1), 1, SWA_MAX_W);
  const start = Math.max(1, current - width + 1);
  const visible = range(start, current);
  const outside = range(1, start - 1);
  const future = range(current + 1, SWA_MAX_T);
  return {
   kind: 'swa', t: current, w: width, start, visible, outside, future,
   count: visible.length, includesCurrent: visible.includes(current),
   maxT: SWA_MAX_T, maxW: SWA_MAX_W,
   isPerformanceClaim: false
  };
 }

 const DSA_INDEX_Q = [1, 0];
 const DSA_INDEX_KEYS = [[1, 0], [3, 0], [-1, 0], [2, 0]];
 const DSA_MAIN_LOGITS = [0, Math.log(3), 0.5, 0];
 const DSA_VALUES = [[1, 1], [2, 0], [0, 2], [0, 4]];

 function dsaExample(k = 2) {
  const topK = clamp(Math.round(Number(k) || 1), 1, 4);
  const indexScores = DSA_INDEX_KEYS.map(key => Math.max(0, dot(DSA_INDEX_Q, key)));
  const ranking = indexScores
   .map((score, index) => ({ score, index }))
   .sort((a, b) => b.score - a.score || a.index - b.index)
   .map(item => item.index);
  const selected = ranking.slice(0, topK).slice().sort((a, b) => a - b);
  const selectedLogits = selected.map(index => DSA_MAIN_LOGITS[index]);
  const weights = softmax(selectedLogits);
  const output = [0, 1].map(coordinate => sum(selected.map((index, slot) => weights[slot] * DSA_VALUES[index][coordinate])));
  return {
   kind: 'dsa', k: topK, indexQ: DSA_INDEX_Q, indexKeys: DSA_INDEX_KEYS,
   indexScores, ranking, rankingPositions: ranking.map(index => index + 1),
   selected, selectedPositions: selected.map(index => index + 1),
   selectedLogits, mainLogits: DSA_MAIN_LOGITS, values: DSA_VALUES,
   weights, output,
   indexScoresAreFinalWeights: false, deletesUnselected: false, indexCacheSeparate: true,
   isPerformanceClaim: false
  };
 }

 /* ------------------------------------------------------------------ */
 /* NSA teaching model: compression / selection / window + gates        */
 /* ------------------------------------------------------------------ */

 const NSA_VISIBLE = 8;
 const NSA_BLOCK_SIZE = 2;
 const NSA_STRIDE = 2;
 const NSA_BLOCK_COUNT = 4;
 const NSA_WINDOW = 2;
 const NSA_HEAD_BLOCK_WEIGHTS = [
  [0.4, 0.3, 0.2, 0.1],
  [0.1, 0.1, 0.7, 0.1]
 ];
 // Fine-grained selection and window logits are separate teaching inputs.
 // They must not be read as the block-selection scores or as branch outputs.
 const NSA_FINE_LOGITS = [0.2, 0.5, 0.8, 0.3, 1.2, 0.4, 0.9, 0.6];
 const NSA_WINDOW_LOGITS = [0.7, 1.0];
 const NSA_DEFAULT_GATES = [0.5, 0.75, 0.25];
 const NSA_BRANCH_OUTPUTS = { compressed: [2, 0], selected: [0, 4], window: [2, 2] };
 // Summaries are deliberately unlike a block mean. They stand for a learned
 // MLP output with intra-block positions; this page does not run that MLP.
 const NSA_SUMMARIES = [[1.0, 0.5], [0.2, 1.4], [1.8, 0.9], [0.6, 1.7]];
 const NSA_TOKEN_K = [[1, 0], [0, 1], [2, 1], [1, 2], [1, 1], [0, 2], [1, -1], [2, 1]];
 const NSA_TOKEN_V = [[1, 0], [0, 1], [2, 1], [1, 2], [3, 0], [1, 1], [0, 2], [2, 2]];
 const NSA_TOKENS = NSA_TOKEN_K.map((k, index) => ({
  position: index + 1,
  block: Math.floor(index / NSA_BLOCK_SIZE),
  k,
  v: NSA_TOKEN_V[index]
 }));
 const NSA_BLOCKS = range(0, NSA_BLOCK_COUNT - 1).map(index => ({
  index,
  label: `B${index + 1}`,
  tokenPositions: [index * NSA_BLOCK_SIZE + 1, (index + 1) * NSA_BLOCK_SIZE],
  tokens: NSA_TOKENS.slice(index * NSA_BLOCK_SIZE, (index + 1) * NSA_BLOCK_SIZE),
  summaryLabel: `c${index + 1}`,
  summary: NSA_SUMMARIES[index]
 }));

 function nsaBlockSelection(topN = 1) {
  const count = clamp(Math.round(Number(topN) || 1), 1, NSA_BLOCK_COUNT);
  const groupScores = NSA_HEAD_BLOCK_WEIGHTS.reduce(
   (scores, row) => scores.map((value, index) => value + row[index]),
   new Array(NSA_BLOCK_COUNT).fill(0)
  ).map(value => Number(value.toFixed(10)));
  const ranking = groupScores
   .map((score, index) => ({ score, index }))
   .sort((a, b) => b.score - a.score || a.index - b.index)
   .map(item => item.index);
  const selectedByRank = ranking.slice(0, count);
  const selectedBlocks = selectedByRank.slice().sort((a, b) => a - b);
  const selectedPositions = selectedBlocks.flatMap(index => NSA_BLOCKS[index].tokenPositions);
  return {
   topN: count, ranking, selectedByRank, selectedBlocks, selectedPositions,
   groupScores, groupScoreSum: Number(sum(groupScores).toFixed(10)),
   groupScoresAreProbabilities: false
  };
 }

 function nsaGateMerge(gates = NSA_DEFAULT_GATES, branchOutputs = NSA_BRANCH_OUTPUTS) {
  const values = NSA_DEFAULT_GATES.map((fallback, index) => {
   const raw = Array.isArray(gates) ? gates[index] : undefined;
   const numeric = Number(raw);
   return clamp(Number.isFinite(numeric) ? numeric : fallback, 0, 1);
  });
  const outputs = {
   compressed: (branchOutputs && branchOutputs.compressed ? branchOutputs.compressed : NSA_BRANCH_OUTPUTS.compressed).slice(0, 2).map(Number),
   selected: (branchOutputs && branchOutputs.selected ? branchOutputs.selected : NSA_BRANCH_OUTPUTS.selected).slice(0, 2).map(Number),
   window: (branchOutputs && branchOutputs.window ? branchOutputs.window : NSA_BRANCH_OUTPUTS.window).slice(0, 2).map(Number)
  };
  const output = [0, 1].map(coordinate => (
   values[0] * outputs.compressed[coordinate]
   + values[1] * outputs.selected[coordinate]
   + values[2] * outputs.window[coordinate]
  ));
  return {
   gates: values, gateSum: sum(values), branchOutputs: outputs, output,
   normalized: false, independentSigmoid: true
  };
 }

 function nsaExample(options) {
  const settings = options || {};
  const selection = nsaBlockSelection(settings.topN);
  const gate = nsaGateMerge(settings.gates, settings.branchOutputs);
  const selectedSet = new Set(selection.selectedPositions);
  const selectedTokens = selection.selectedPositions.map(position => NSA_TOKENS[position - 1]);
  const fineLogits = selection.selectedPositions.map(position => NSA_FINE_LOGITS[position - 1]);
  const fineWeights = softmax(fineLogits);
  const windowPositions = range(NSA_VISIBLE - NSA_WINDOW + 1, NSA_VISIBLE);
  const windowWeights = softmax(NSA_WINDOW_LOGITS);
  const headBlockWeights = NSA_HEAD_BLOCK_WEIGHTS.map(row => row.slice());
  const headWeightSums = headBlockWeights.map(row => Number(sum(row).toFixed(10)));
  const blocks = NSA_BLOCKS.map(block => ({
   ...block,
   selected: block.tokenPositions.every(position => selectedSet.has(position))
  }));
  return {
   kind: 'nsa', topN: selection.topN,
   visibleTokens: NSA_VISIBLE, blockSize: NSA_BLOCK_SIZE, selectionBlockSize: NSA_BLOCK_SIZE,
   stride: NSA_STRIDE, blockCount: NSA_BLOCK_COUNT, completeBlocks: NSA_BLOCK_COUNT,
   blocks, tokens: NSA_TOKENS, summaries: NSA_SUMMARIES,
   headBlockWeights, headWeightSums,
   groupBlockWeights: selection.groupScores,
   groupWeightSum: selection.groupScoreSum,
   groupScoresAreProbabilities: false,
   ranking: selection.ranking,
   selectedByRank: selection.selectedByRank,
   selectedBlocks: selection.selectedBlocks,
   selectedPositions: selection.selectedPositions,
   selectedTokens,
   selection: {
    positions: selection.selectedPositions, tokens: selectedTokens,
    logits: fineLogits, weights: fineWeights, softmaxSum: sum(fineWeights)
   },
   window: {
    w: NSA_WINDOW, positions: windowPositions,
    tokens: windowPositions.map(position => NSA_TOKENS[position - 1]),
    logits: NSA_WINDOW_LOGITS.slice(), weights: windowWeights, softmaxSum: sum(windowWeights)
   },
   gates: gate.gates, gateSum: gate.gateSum, gatesAreNormalized: false,
   branchOutputs: gate.branchOutputs, output: gate.output,
   branchOutputsDerivedFromBlockWeights: false,
   compressor: {
    type: 'learnable-mlp', usesIntraBlockPosition: true, fixedAverage: false,
    summaryValuesAreTeachingInputs: true
   },
   branchesUseIndependentKV: true,
   branchesHaveIndependentSoftmax: true,
   isPerformanceClaim: false, isAccuracyClaim: false
  };
 }

 /* ------------------------------------------------------------------ */
 /* Shared controls                                                     */
 /* ------------------------------------------------------------------ */

 const LINEAR_S0 = [[2, 0], [0, 4]];
 const LINEAR_WRITE = { k: [1, 0], v: [6, 2], q: [1, 1] };
 const LINEAR_BETA = 0.5;
 const LINEAR_MODE_LABEL = { add: '简单累加', delta: 'DeltaNet', gdn: 'GDN', kda: 'KDA' };

 function scaleRows(matrix, factors) {
  return matrix.map((row, i) => row.map(value => value * factors[i]));
 }

 function addMatrices(a, b) {
  return a.map((row, i) => row.map((value, j) => value + b[i][j]));
 }

 function outer(a, b) {
  return a.map(value => b.map(other => value * other));
 }

 function linearStep(state, write, mode, decay, beta) {
  const sbar = scaleRows(state, decay);
  const prediction = applyMatrix(transpose(sbar), write.k);
  const error = write.v.map((value, i) => value - prediction[i]);
  const usesDelta = mode !== 'add';
  const update = usesDelta
    ? outer(write.k, error).map(row => row.map(value => value * beta))
    : outer(write.k, write.v);
  const snew = addMatrices(sbar, update);
  const output = applyMatrix(transpose(snew), write.q);
  return { stateBefore: state, decay, sbar, prediction, error, update, state: snew, output, usesDelta };
 }

 function linearStateExample(mode = 'gdn', options) {
  const settings = options || {};
  const key = Object.hasOwn(LINEAR_MODE_LABEL, mode) ? mode : 'gdn';
  const alpha = clamp(Number.isFinite(Number(settings.alpha)) ? Number(settings.alpha) : 0.5, 0, 1);
  const alpha1 = clamp(Number.isFinite(Number(settings.alpha1)) ? Number(settings.alpha1) : 0.5, 0, 1);
  const alpha2 = clamp(Number.isFinite(Number(settings.alpha2)) ? Number(settings.alpha2) : 0.25, 0, 1);
  const decay = key === 'kda' ? [alpha1, alpha2] : key === 'gdn' ? [alpha, alpha] : [1, 1];
  const beta = key === 'add' ? 1 : key === 'delta' ? 1 : LINEAR_BETA;
  const first = linearStep(LINEAR_S0, LINEAR_WRITE, key, decay, beta);
  const second = linearStep(first.state, LINEAR_WRITE, key, decay, beta);
  return {
   kind: 'linear-state', mode: key, modeLabel: LINEAR_MODE_LABEL[key],
   s0: LINEAR_S0, write: LINEAR_WRITE, beta, decay,
   alpha, alpha1, alpha2, decayMode: key === 'kda' ? 'channel' : key === 'gdn' ? 'scalar' : 'none',
   updates: [first, second], states: [LINEAR_S0, first.state, second.state],
   outputs: [first.output, second.output],
   usesErrorCorrection: key !== 'add', sameWriteRepeated: true,
   isPerformanceClaim: false
  };
 }

 /* ------------------------------------------------------------------ */
 /* Shared controls                                                     */
 /* ------------------------------------------------------------------ */

 function optionButtons(attribute, items, active, label) {
  return `<div class="av-seg" role="group" aria-label="${escapeHtml(label)}">${items.map(item => `<button type="button" ${attribute}="${item.id}" aria-pressed="${item.id === active}">${escapeHtml(item.label)}${item.note ? `<small>${escapeHtml(item.note)}</small>` : ''}</button>`).join('')}</div>`;
 }

 // These panels are still diagrams, not players. They keep the same lifecycle
 // contract as the tab shell so a future panel can add motion without changing it.
 function idleInstance() {
  return { pause() {}, reset() {} };
 }

 /* ------------------------------------------------------------------ */
 /* Panel: MHA / GQA / MQA — only the K/V group count changes           */
 /* ------------------------------------------------------------------ */

 function mountHeads(panel, announce) {
  let mode = 'mha';
  let selected = null;
  panel.innerHTML = `<div class="av-panel-head">
    <p class="av-purpose"><span>改动对象</span>每个 Q 头共享几套历史 K/V。</p>
    <p class="av-fixed"><span>保持不变</span>4 个 Q 头、各自的 q、softmax 与输出。</p>
   </div>
   <div class="av-controls"><span class="av-control-label">共享方式</span>${optionButtons('data-av-mode', [
    { id: 'mha', label: 'MHA', note: '4 组' },
    { id: 'gqa', label: 'GQA', note: '2 组' },
    { id: 'mqa', label: 'MQA', note: '1 组' }
   ], 'mha', '选择 K/V 共享方式')}</div>
   <div class="av-stage">
    <div class="av-heads-map" data-av-map></div>
   </div>
   <p class="av-readout" data-av-readout aria-live="polite"></p>
   <p class="av-note">共享的是同一套历史 K/V 被多个 Q 引用，不是把 Q 头合并，也不是复制一份 K/V 后宣称省内存。同组 Q 的权重与输出仍各自计算；本图不比较模型质量。</p>`;

  const map = panel.querySelector('[data-av-map]');
  const readout = panel.querySelector('[data-av-readout]');

  function setPressed(next) {
   mode = next;
   selected = null;
   panel.querySelectorAll('[data-av-mode]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.avMode === mode));
   });
  }

  function render() {
   const data = headsExample(mode);
   const qButtons = data.heads.map((head, index) => `<button type="button" class="av-q" data-av-q="${index}" data-group="${head.group}" style="grid-row:${index + 2}" aria-pressed="${selected === index}" aria-label="Q${index}，独立 q，读取 KV${head.group}">
     <b>Q${index}</b><small>独立 q</small>
    </button>`).join('');
   const wires = data.heads.map((head, index) => `<span class="av-wire" data-group="${head.group}" style="grid-row:${index + 2}" aria-hidden="true"></span>`).join('');
   const kvCards = data.groups.map((group, groupIndex) => `<div class="av-kv" data-group="${groupIndex}" style="grid-row:${group[0] + 2} / span ${group.length}">
     <div class="av-kv-head"><b>KV${groupIndex}</b><span>${group.length > 1 ? '共享' : '独占'}</span></div>
     <small>${group.map(head => 'Q' + head).join(' · ')} 读取</small>
    </div>`).join('');
   map.innerHTML = `<div class="av-heads-label av-heads-label-q">Q 头 · ${data.qHeads} 个</div>
    <div class="av-heads-label av-heads-label-kv">历史 K/V · <b>${data.kvGroups} 组</b></div>
    ${qButtons}${wires}${kvCards}`;

   if (selected === null) {
    const relation = data.kvGroups === data.qHeads
     ? '每个 Q 头各读一套。'
     : data.kvGroups === 1
      ? '4 个 Q 头都读同一套。'
      : 'Q0、Q1 读 KV0；Q2、Q3 读 KV1。';
    readout.textContent = `${data.modeLabel}：Q 头仍是 ${data.qHeads} 个；历史 K/V 分成 ${data.kvGroups} 组，${relation} 同组 Q 仍各自用 q 打分、各自输出。`;
   } else {
    const head = data.heads[selected];
    const peers = data.groups[head.group].filter(index => index !== selected).map(index => 'Q' + index);
    readout.textContent = `Q${selected} 用自己的 q 读 KV${head.group}；${peers.length ? peers.join('、') + ' 也读这套 K/V，但 q、softmax 和输出各自计算。' : '这一组只有它一个 Q 头。'}`;
   }
  }

  panel.addEventListener('click', event => {
   const modeButton = event.target.closest('[data-av-mode]');
   if (modeButton && panel.contains(modeButton) && modeButton.dataset.avMode !== mode) {
    setPressed(modeButton.dataset.avMode);
    render();
    announce(`已切换到 ${headsExample(mode).modeLabel}：Q 头数量不变，只改历史 K/V 的共享组数。`);
    return;
   }
   const qButton = event.target.closest('[data-av-q]');
   if (!qButton || !panel.contains(qButton)) return;
   selected = Number(qButton.dataset.avQ);
   render();
   announce(readout.textContent);
  });

  render();
  return idleInstance();
 }

 /* ------------------------------------------------------------------ */
 /* Panel: MLA — per-position latent plus a separate position key       */
 /* ------------------------------------------------------------------ */

 function mountMLA(panel) {
  let history = 4;
  panel.innerHTML = `<div class="av-panel-head">
    <p class="av-purpose"><span>改动对象</span>每个位置缓存什么。</p>
    <p class="av-fixed"><span>仍然如此</span>缓存按位置增长；不是固定形状的状态。</p>
   </div>
   <div class="av-controls">
    <label class="av-slider"><span>历史位置</span><input type="range" min="2" max="6" step="1" value="4" data-av-history aria-label="历史位置数"><output data-av-history-out>4</output></label>
   </div>
   <div class="av-stage">
    <div class="av-mla-cache" data-av-cache></div>
    <p class="av-grow">历史每增加一个位置，就多一列缓存。</p>
    <div class="av-mla-flow">
     <div class="av-flow-node is-latent"><b>latent c</b><span>内容 K/V 可由 c 恢复；计算可合并到查询和输出路径。</span></div>
     <div class="av-flow-node is-pos"><b>位置 key kᴿ</b><span>单独保留，参与位置打分；不并进同一份 latent。</span></div>
    </div>
   </div>
   <p class="av-readout" data-av-readout aria-live="polite"></p>
   <p class="av-note">MLA 改的是每个位置的缓存表示，不是把 Q 头合并。缓存仍随历史长度增长，只是每位置放的东西更窄；本图省略下投影生成、norm 和真实头数。</p>`;

  const cache = panel.querySelector('[data-av-cache]');
  const readout = panel.querySelector('[data-av-readout]');
  const input = panel.querySelector('[data-av-history]');
  const output = panel.querySelector('[data-av-history-out]');

  function render() {
   const positions = range(1, history);
   cache.style.setProperty('--cols', history);
   cache.innerHTML = positions.map(position => `<div class="av-mla-col">
     <span class="av-mla-pos">p${position}</span>
     <span class="av-mla-cell is-latent"><b>c${position}</b><small>latent</small></span>
     <span class="av-mla-cell is-pos"><b>kᴿ${position}</b><small>位置 key</small></span>
    </div>`).join('');
   output.textContent = history;
   readout.textContent = `历史 ${history} 个位置 = ${history} 列缓存。每列只有 latent 与位置 key；内容 K/V 读时恢复或吸收，位置 key 单独参与打分。`;
  }

  input.addEventListener('input', () => {
   history = clamp(Number(input.value) || 4, 2, 6);
   render();
  });
  render();
  return idleInstance();
 }

 /* ------------------------------------------------------------------ */
 /* Panel: SWA — a window that moves with the query                     */
 /* ------------------------------------------------------------------ */

 function mountSWA(panel, announce) {
  let t = 6;
  let w = 3;
  panel.innerHTML = `<div class="av-panel-head">
    <p class="av-purpose"><span>改动对象</span>当前层能直接读哪些位置。</p>
    <p class="av-fixed"><span>保持不变</span>窗口外的历史仍可能在缓存里，不等于删除。</p>
   </div>
   <div class="av-controls"><span class="av-control-label">窗口宽度 W</span>${optionButtons('data-av-w', [
    { id: '1', label: '1' }, { id: '2', label: '2' }, { id: '3', label: '3' }, { id: '4', label: '4' }
   ], '3', '选择窗口宽度')}</div>
   <div class="av-stage">
    <p class="av-instruction">点一个位置，把 query 放过去。</p>
    <div class="av-swa-grid" data-av-grid></div>
    <div class="av-legend"><span class="is-current">当前 query</span><span class="is-visible">窗口内可读</span><span class="is-outside">窗口外仍保留</span><span class="is-future">未来不可见</span></div>
   </div>
   <p class="av-readout" data-av-readout aria-live="polite"></p>
   <p class="av-note">窗口外不进入本次 softmax，不等于被删除或立即回收。更早的信息能否被用到，要看缓存策略和后续层；本图不测量速度或质量。</p>`;

  const grid = panel.querySelector('[data-av-grid]');
  const readout = panel.querySelector('[data-av-readout]');

  function render() {
   const data = swaExample(t, w);
   grid.innerHTML = range(1, data.maxT).map(position => {
    const state = position === data.t ? 'current' : data.visible.includes(position) ? 'visible' : position < data.start ? 'outside' : 'future';
    const hint = { current: '当前 query，本层可读', visible: '窗口内，本层可读', outside: '窗口外，仍保留在缓存', future: '未来位置，不可见' }[state];
    const mark = { current: 'Q', visible: '读', outside: '存', future: '—' }[state];
    return `<button type="button" class="av-swa-token is-${state}" style="grid-row:1;grid-column:${position}" data-av-pos="${position}" aria-pressed="${position === data.t}" aria-label="p${position}：${hint}。点击把 query 放到这里"><span>p${position}</span><small>${mark}</small></button>`;
   }).join('');
   const brackets = [];
   if (data.start > 1) brackets.push(`<span class="av-bracket is-outside" style="grid-row:2;grid-column:1 / ${data.start}">${data.start > 2 ? '仍在缓存' : '存'}</span>`);
   brackets.push(`<span class="av-bracket is-read" style="grid-row:2;grid-column:${data.start} / ${data.t + 1}">${data.count > 1 ? '本层读取' : '读'}</span>`);
   if (data.t < data.maxT) brackets.push(`<span class="av-bracket is-future" style="grid-row:2;grid-column:${data.t + 1} / ${data.maxT + 1}">${data.maxT - data.t > 1 ? '尚不可见' : '未来'}</span>`);
   grid.insertAdjacentHTML('beforeend', brackets.join(''));

   const kept = data.start > 1 ? `p1–p${data.start - 1} 在窗口外，但仍留在缓存里。` : '';
   const future = data.t < data.maxT ? `p${data.t + 1}–p${data.maxT} 是未来位置，本次不可见。` : '没有未来位置。';
   readout.textContent = `query 在 p${data.t}，窗口 W=${data.w}：本层只读 p${data.start}–p${data.t}（${data.count} 个位置）。${kept}${future}`;
  }

  panel.addEventListener('click', event => {
   const widthButton = event.target.closest('[data-av-w]');
   if (widthButton && panel.contains(widthButton)) {
    const next = clamp(Number(widthButton.dataset.avW), 1, 4);
    if (next === w) return;
    w = next;
    panel.querySelectorAll('[data-av-w]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.avW) === w)));
    render();
    announce(`窗口宽度改为 ${w}。${readout.textContent}`);
    return;
   }
   const token = event.target.closest('[data-av-pos]');
   if (!token || !panel.contains(token)) return;
   t = Number(token.dataset.avPos);
   render();
   announce(readout.textContent);
  });

  render();
  return idleInstance();
 }

 /* ------------------------------------------------------------------ */
 /* Panel: DSA — choose positions, then read only those main entries    */
 /* ------------------------------------------------------------------ */

 function mountDSA(panel, announce) {
  let topK = 2;
  panel.innerHTML = `<div class="av-panel-head">
    <p class="av-purpose"><span>改动对象</span>主注意力本次读取哪些位置。</p>
    <p class="av-fixed"><span>保持不变</span>未选中的主缓存条目不删除。</p>
   </div>
   <div class="av-controls"><span class="av-control-label">选几个位置</span>${optionButtons('data-av-topk', [
    { id: '1', label: 'Top-1' }, { id: '2', label: 'Top-2' }, { id: '3', label: 'Top-3' }
   ], '2', '选择 Top-K')}</div>
   <div class="av-stage">
    <div class="av-dsa-label"><b>① index cache</b><span>轻量索引键，只用来选位置</span></div>
    <div class="av-dsa-index" data-av-index></div>
    <div class="av-dsa-gate"><span class="av-dsa-gate-note">indexer 选位置</span><div class="av-dsa-gate-cells" data-av-gate></div></div>
    <div class="av-dsa-label"><b>② main cache</b><span>完整历史；可采用 MLA latent</span></div>
    <div class="av-dsa-main" data-av-main></div>
   </div>
   <p class="av-readout" data-av-readout aria-live="polite"></p>
   <p class="av-note">索引分数不是最终 attention 权重；Top-K 也不是固定滑动窗口，更不是缓存合并。主缓存只读选中条目，未选中条目仍保留，供后续 query 读取。</p>`;

  const indexEl = panel.querySelector('[data-av-index]');
  const gateEl = panel.querySelector('[data-av-gate]');
  const mainEl = panel.querySelector('[data-av-main]');
  const readout = panel.querySelector('[data-av-readout]');

  function render() {
   const data = dsaExample(topK);
   const selected = new Set(data.selected);
   const maxScore = Math.max(1, ...data.indexScores);
   indexEl.innerHTML = data.indexScores.map((score, index) => {
    const isSelected = selected.has(index);
    const rank = data.ranking.indexOf(index) + 1;
    return `<div class="av-dsa-item${isSelected ? ' is-selected' : ''}">
      <div class="av-dsa-item-head"><b>p${index + 1}</b><small>#${rank}</small></div>
      <span class="av-score" aria-hidden="true"><i style="--score:${(score / maxScore * 100).toFixed(1)}%"></i></span>
      <small>${isSelected ? '选中' : '候选'}</small>
     </div>`;
   }).join('');
   gateEl.innerHTML = data.indexScores.map((_, index) => `<span class="${selected.has(index) ? 'is-on' : ''}" aria-hidden="true">${selected.has(index) ? '↓' : ''}</span>`).join('');
   mainEl.innerHTML = data.indexScores.map((_, index) => `<div class="av-dsa-item av-dsa-main-item${selected.has(index) ? ' is-selected' : ' is-kept'}">
     <b>p${index + 1}</b><small>${selected.has(index) ? '本次读取' : '保留，不删除'}</small>
    </div>`).join('');
   readout.textContent = `Top-K=${data.k}：indexer 在 4 个可见位置中选中 p${data.selectedPositions.join('、p')}；主注意力只读这些主缓存条目。其余位置没有被选中，但仍在主缓存里；索引分只负责选位，不充当最终权重。`;
  }

  panel.querySelectorAll('[data-av-topk]').forEach(button => {
   button.addEventListener('click', () => {
    const next = clamp(Number(button.dataset.avTopk), 1, 3);
    if (next === topK) return;
    topK = next;
    panel.querySelectorAll('[data-av-topk]').forEach(item => item.setAttribute('aria-pressed', String(Number(item.dataset.avTopk) === topK)));
    render();
    announce(readout.textContent);
   });
  });
  render();
  return idleInstance();
 }

 /* ------------------------------------------------------------------ */
 /* Panel: NSA — three independent reads, then independent gates        */
 /* ------------------------------------------------------------------ */

 function mountNSA(panel, announce) {
  let topN = 1;
  panel.innerHTML = `<div class="av-panel-head">
    <p class="av-purpose"><span>改动对象</span>历史按三种粒度并行读取，再门控汇合。</p>
    <p class="av-fixed"><span>保持不变</span>三路各有自己的 K/V 读取与 softmax；门不归一化。</p>
   </div>
   <div class="av-controls"><span class="av-control-label">选块路</span>${optionButtons('data-av-topn', [
    { id: '1', label: 'Top-1' }, { id: '2', label: 'Top-2' }
   ], '1', '选择块数量')}</div>
   <div class="av-stage">
    <div class="av-nsa-board" data-av-nsa></div>
    <div class="av-nsa-merge"><span class="av-nsa-dot" aria-hidden="true"></span><b>g₁·压缩 + g₂·选择 + g₃·窗口</b><small>三路独立门控，不归一化</small></div>
   </div>
   <p class="av-readout" data-av-readout aria-live="polite"></p>
   <p class="av-note">三路不是同一套 attention 的三种视图：压缩路读块摘要，选择路读选中块的原 token，窗口路读最近原 token。门是逐路独立系数，和不必为 1，不能事后重新 softmax。</p>`;

  const board = panel.querySelector('[data-av-nsa]');
  const readout = panel.querySelector('[data-av-readout]');

  function blocksFor(branch, data) {
   const selected = new Set(data.selectedBlocks);
   return data.blocks.map((block, index) => {
    const label = `<span class="av-nsa-block-label">B${block.index + 1}</span>`;
    if (branch === 'compressed') {
     return `<div class="av-nsa-block is-summary-block">${label}<div class="av-nsa-cells"><span class="av-nsa-cell is-summary">c${block.index + 1}</span></div></div>`;
    }
    if (branch === 'selected') {
     const on = selected.has(index);
     return `<div class="av-nsa-block${on ? ' is-selected' : ' is-skipped'}">${label}<div class="av-nsa-cells">${block.tokens.map(token => `<span class="av-nsa-cell">p${token.position}</span>`).join('')}</div></div>`;
    }
    const on = index === data.blocks.length - 1;
    return `<div class="av-nsa-block${on ? ' is-selected' : ' is-skipped'}">${label}<div class="av-nsa-cells">${block.tokens.map(token => `<span class="av-nsa-cell">p${token.position}</span>`).join('')}</div></div>`;
   }).join('');
  }

  function row(branch, name, sub, gate, inner) {
   return `<div class="av-nsa-row" data-branch="${branch}">
     <div class="av-nsa-name"><b>${name}</b><small>${sub}</small></div>
     <div class="av-nsa-track">${inner}</div>
     <div class="av-nsa-gate" aria-label="${name}路的独立门"><i aria-hidden="true" style="--g:${gate}"></i><span>g</span></div>
    </div>`;
  }

  function render() {
   const data = nsaExample({ topN });
   board.innerHTML = [
    row('compressed', '压缩', '读块摘要', 0.5, blocksFor('compressed', data)),
    row('selected', '选择', '选整块，读原 token', 0.75, blocksFor('selected', data)),
    row('window', '窗口', '读最近 token', 0.25, blocksFor('window', data))
   ].join('');
   const selectedText = data.selectedBlocks.map(index => 'B' + (index + 1)).join('、');
   readout.textContent = `选块路取 Top-${data.topN}：${selectedText}。压缩路仍读 4 个块摘要，窗口路仍读 p7–p8；三路各自 softmax，门不因 Top-N 重新归一化。`;
  }

  panel.querySelectorAll('[data-av-topn]').forEach(button => {
   button.addEventListener('click', () => {
    const next = clamp(Number(button.dataset.avTopn), 1, 2);
    if (next === topN) return;
    topN = next;
    panel.querySelectorAll('[data-av-topn]').forEach(item => item.setAttribute('aria-pressed', String(Number(item.dataset.avTopn) === topN)));
    render();
    announce(readout.textContent);
   });
  });
  render();
  return idleInstance();
 }

 /* ------------------------------------------------------------------ */
 /* Panel: GDN / KDA — one state, two decay shapes                      */
 /* ------------------------------------------------------------------ */

 function mountLinearState(panel) {
  const stateCard = (mode, name, note, decays) => `<article class="av-state-card" data-mode="${mode}">
    <header><b>${name}</b><span>${note}</span></header>
    <div class="av-state" role="img" aria-label="${name}：固定形状状态，${mode === 'gdn' ? '整张状态使用同一个衰减 α' : '每个 key 通道使用自己的衰减 α₁、α₂'}">
     <span class="av-state-rowlabel">key 1</span><span class="av-state-cell">S₁₁</span><span class="av-state-cell">S₁₂</span><span class="av-state-decay">${decays[0]}</span>
     <span class="av-state-rowlabel">key 2</span><span class="av-state-cell">S₂₁</span><span class="av-state-cell">S₂₂</span><span class="av-state-decay">${decays[1]}</span>
    </div>
    <p>${mode === 'gdn' ? '所有 key 通道一起退：同一个 α 缩放整张状态。' : '每个 key 通道各自退：第 1 行 α₁、第 2 行 α₂，可以不同。'}</p>
   </article>`;

  panel.innerHTML = `<div class="av-panel-head">
    <p class="av-purpose"><span>改动对象</span>历史放在哪里：不再保存逐 token K/V 列表。</p>
    <p class="av-fixed"><span>保持不变</span>每来一个 token 仍要更新一次，形状固定但计算不消失。</p>
   </div>
   <div class="av-stage">
    <div class="av-linear-flow">
     <div class="av-flow-step"><b>旧状态 S</b><small>固定形状</small></div>
     <span class="av-flow-arrow" aria-hidden="true">→</span>
     <div class="av-flow-step"><b>先衰减</b><small>GDN 整体 / KDA 按通道</small></div>
     <span class="av-flow-arrow" aria-hidden="true">→</span>
     <div class="av-flow-step"><b>用 k 预测、与 v 比较</b><small>新信息决定修正量</small></div>
     <span class="av-flow-arrow" aria-hidden="true">→</span>
     <div class="av-flow-step"><b>按误差写回</b><small>状态被纠正，不是只累加</small></div>
     <span class="av-flow-arrow" aria-hidden="true">→</span>
     <div class="av-flow-step"><b>q 读出</b><small>输出当前表示</small></div>
    </div>
    <div class="av-state-compare">
     ${stateCard('gdn', 'GDN', '整体衰减', ['×α', '×α'])}
     ${stateCard('kda', 'KDA', '按通道衰减', ['×α₁', '×α₂'])}
    </div>
   </div>
   <p class="av-readout">两种更新都先把旧状态衰减，再用当前 k、v 的预测误差纠正写回，最后由 q 读出。差别只在衰减的形状：GDN 整张一个 α，KDA 每个 key 通道一个 α。</p>
   <p class="av-note">固定形状只描述单层、单头、当前状态；处理 L 个位置仍要执行 L 次更新。混合模型还可能有卷积窗口、检查点和投机临时状态；本图不展示矩阵数值或训练过程。</p>`;

  return idleInstance();
 }

 /* ------------------------------------------------------------------ */
 /* Registry and tab shell                                              */
 /* ------------------------------------------------------------------ */

 const VARIANTS = [
  {
   id: 'heads', kicker: 'KV 共享', label: 'MHA / GQA / MQA',
   purpose: '4 个 Q 头不变，只改每组 Q 共享几套历史 K/V。',
   mount: mountHeads
  },
  {
   id: 'mla', kicker: 'Latent', label: 'MLA',
   purpose: '每个位置存 latent 与位置 key，读时再恢复内容 K/V。',
   mount: mountMLA
  },
  {
   id: 'swa', kicker: '窗口', label: 'SWA',
   purpose: '当前层只读最近 W 个位置，更早历史仍可保留。',
   mount: mountSWA
  },
  {
   id: 'dsa', kicker: '索引', label: 'DSA',
   purpose: '先选位置，再只读选中的主缓存条目。',
   mount: mountDSA
  },
  {
   id: 'nsa', kicker: '三路', label: 'NSA',
   purpose: '摘要、选中块原 token、近期窗口三路独立读取，再门控汇合。',
   mount: mountNSA
  },
  {
   id: 'linear', kicker: '递推', label: 'GDN / KDA',
   purpose: '历史压进固定形状状态，衰减后按新信息纠正写回。',
   mount: mountLinearState
  }
 ];

 function buildShell(root) {
  const tabs = VARIANTS.map((variant, index) => `<button type="button" role="tab" id="av-tab-${variant.id}" aria-controls="av-panel-${variant.id}" aria-selected="${index === 0}" tabindex="${index === 0 ? 0 : -1}" data-av-tab="${variant.id}"><span>${escapeHtml(variant.kicker)}</span><small>${escapeHtml(variant.label)}</small></button>`).join('');
  const panels = VARIANTS.map((variant, index) => `<section role="tabpanel" id="av-panel-${variant.id}" aria-labelledby="av-tab-${variant.id}" data-av-panel="${variant.id}" ${index === 0 ? '' : 'hidden'}></section>`).join('');
  root.innerHTML = `<div class="av-tabs" role="tablist" aria-label="选择 Attention 变体">${tabs}</div><div class="av-panels">${panels}</div><p class="av-announce av-sr" data-av-announce aria-live="polite"></p>`;
 }

 function mount(root) {
  if (!root || root.dataset.avMounted === 'true') return;
  root.dataset.avMounted = 'true';
  buildShell(root);
  const announceEl = root.querySelector('[data-av-announce]');
  const announce = message => {
   if (!announceEl) return;
   announceEl.textContent = '';
   announceEl.textContent = message;
  };
  const instances = {};
  VARIANTS.forEach(variant => {
   const panel = root.querySelector(`[data-av-panel="${variant.id}"]`);
   instances[variant.id] = variant.mount(panel, announce);
  });
  const tabs = Array.from(root.querySelectorAll('[data-av-tab]'));
  let active = 'heads';

  function activate(id, options) {
   const settings = options || {};
   if (!instances[id]) return;
   if (id === active) {
    if (settings.focus) root.querySelector(`[data-av-tab="${id}"]`).focus();
    return;
   }
   active = id;
   tabs.forEach(tab => {
    const selected = tab.dataset.avTab === id;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
   });
   VARIANTS.forEach(variant => {
    const panel = root.querySelector(`[data-av-panel="${variant.id}"]`);
    if (panel) panel.hidden = variant.id !== id;
   });
   if (settings.focus) root.querySelector(`[data-av-tab="${id}"]`).focus();
   if (!settings.initial) {
    const variant = VARIANTS.find(item => item.id === id);
    announce(`已切换到 ${variant.label}：${variant.purpose}`);
   }
  }

  root.addEventListener('click', event => {
   const tab = event.target.closest('[data-av-tab]');
   if (tab && root.contains(tab)) activate(tab.dataset.avTab, { focus: false });
  });
  root.addEventListener('keydown', event => {
   const tab = event.target.closest('[data-av-tab]');
   if (!tab) return;
   const index = tabs.indexOf(tab);
   let nextIndex = null;
   if (event.key === 'ArrowRight') nextIndex = (index + 1) % tabs.length;
   else if (event.key === 'ArrowLeft') nextIndex = (index - 1 + tabs.length) % tabs.length;
   else if (event.key === 'Home') nextIndex = 0;
   else if (event.key === 'End') nextIndex = tabs.length - 1;
   if (nextIndex === null) return;
   event.preventDefault();
   activate(tabs[nextIndex].dataset.avTab, { focus: true });
  });
  activate('heads', { initial: true });
 }

 function mountAll() {
  document.querySelectorAll('[data-attention-variants]').forEach(mount);
 }

 return {
  headsExample, mlaExample, swaExample, dsaExample, nsaExample, nsaBlockSelection, nsaGateMerge,
  linearStateExample, linearStep,
  VARIANTS: VARIANTS.map(variant => ({ id: variant.id, kicker: variant.kicker, label: variant.label, purpose: variant.purpose })),
  mount, mountAll
 };
});
