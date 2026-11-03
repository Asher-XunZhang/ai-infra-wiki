/* Attention variants: verified teaching models and black comparison diagrams. */
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

  const C={cyan:'#62d8e9',green:'#7fe0c3',purple:'#bda9f7',orange:'#f2b45c',line:'#4b7184',dim:'#a6bac6'};
  const sources={gqa:['GQA','https://arxiv.org/abs/2305.13245v3'],mla:['MLA','https://arxiv.org/html/2405.04434v5'],swa:['SWA','https://arxiv.org/abs/2310.06825'],dsa:['DSA','https://arxiv.org/html/2512.02556v1#S2.SS1'],nsa:['NSA','https://arxiv.org/html/2502.11089v2#S3'],qsa:['QSA','https://arxiv.org/html/2608.30320v1#S2.SS1.SSS2'],csa:['CSA / HCA','https://arxiv.org/html/2606.19348v1#S2.SS3'],gdn:['GDN','https://arxiv.org/html/2412.06464v3#S3.SS1'],kda:['KDA','https://arxiv.org/html/2510.26692v1#S3'],csa2:['CSA2','https://arxiv.org/html/2609.19969v1#S2.SS3.SSS1'],flash:['FlashAttention','https://arxiv.org/abs/2205.14135'],paged:['PagedAttention','https://arxiv.org/abs/2309.06180']};
  const boards=[
    {id:'heads',tab:'头共享',question:'Q 头相同，谁共用一套 K/V？',context:'MHA / GQA / MQA',note:'连线表示共享关系；每个 Q 仍独立查询。',refs:['gqa'],panels:[['MHA','一头一套','4 套 K/V 历史','heads',4],['GQA','组内共享','2 套 K/V 历史','heads',2],['MQA','所有头共享','1 套 K/V 历史','heads',1]]},
    {id:'latent',tab:'MLA 压表示',question:'同一个历史位置，保存什么？',context:'逐位置 KV / MLA',note:'MLA 保留逐位置历史；投影可吸收到计算中，不一定展开完整 K/V。',refs:['mla'],panels:[['普通 K/V','按头保存 K 与 V','一行对应一个历史位置','latent',false],['MLA','latent ＋ 独立的位置 key','位置数不变，内容表示更紧凑','latent',true]]},
    {id:'visibility',tab:'全量 / 窗口 / 选位',question:'当前 Q 到底读取哪些位置？',context:'亮格 = 本次读取；暗格 = 本次不读',note:'DSA 每次按内容选位置；未选条目仍保留。SWA 定义读取范围，缓存释放由实现决定。',refs:['swa','dsa'],panels:[['Full Attention','读取全部因果可见位置','每个可见位置都参与','visibility','full'],['SWA','只看最近窗口','按距离截取近期位置','visibility','window'],['DSA','轻量索引器先选 Top-K','按内容挑选，再读主 KV','visibility','sparse']]},
    {id:'nsa',tab:'NSA 三路',question:'NSA：粗看全局、细读重点、保留近期',context:'三条分支同时提供信息',note:'压缩分支的重要性用于选块；三路分别归一化，再用各自的门汇合。',merge:'压缩路输出 ＋ 选择路输出 ＋ 窗口路输出 → 门控汇合',refs:['nsa'],panels:[['压缩路','从块摘要看全局','保留粗粒度全局信息','nsa','summary'],['选择路','选中块展开为原 token','补上重点内容的细节','nsa','select'],['窗口路','直接读取最近的 token','保留近期的细粒度信息','nsa','window']]},
    {id:'compression',tab:'QSA / CSA / HCA',question:'压缩的是“找位置”，还是“实际读取的内容”？',context:'示意粒度，不表示固定压缩率',note:'QSA 选块后读原位置；CSA/HCA 直接读压缩条目，另有近期窗口。',refs:['qsa','csa'],panels:[['QSA','先压索引，再展开选中块','主 KV 仍对应原 token','compression','qsa'],['CSA','压缩主 KV，再选一部分','读取选中的压缩条目','compression','csa'],['HCA','更强压缩，遍历压缩历史','读取全部可见压缩条目','compression','hca']]},
    {id:'state',tab:'GDN / KDA 状态',question:'历史是一排记录，还是不断更新的一份记忆？',context:'GDN / KDA 展开递推核',note:'GDN 每头整体衰减，KDA 按 key 通道衰减；二者都按预测误差纠正状态。',refs:['gdn','kda'],panels:[['逐位置 Attention','新 token 到来，追加一条 KV','Q 对历史记录逐项读取','state','kv'],['GDN','同一个门控制本头旧状态','衰减 → 纠正 → 从状态读出','state','gdn'],['KDA','不同通道有不同遗忘程度','衰减 → 纠正 → 从状态读出','state','kda']]},
    {id:'reuse',tab:'CSA2 跨层复用',question:'下一层要重建记录、重选位置，还是直接沿用？',context:'Full / Reindex / Reuse：层的静态模式',note:'三种模式都用本层 Q 和本层 SWA 重新计算输出；Reuse 沿用位置，不沿用答案。',refs:['csa2'],panels:[['Full','生成全局 KV 并选位置','建立记录，也建立选择','reuse','full'],['Reindex','复用全局 KV，重新选位置','同一批记录，换一组位置','reuse','reindex'],['Reuse','复用全局 KV 与已选位置','同一组位置，用新的 Q 读取','reuse','reuse']]},
    {id:'execution',tab:'执行与存储',question:'规则可以不变，计算和存放方式也能改变',context:'工程实现维度 · 可与前面机制组合',note:'FlashAttention 重排精确注意力的计算与搬运；PagedAttention 组织 KV 的物理存储。',refs:['flash','paged'],panels:[['常规实现','中间分数矩阵写回显存','显示完整中间矩阵的实现','execution','standard'],['FlashAttention','分块加载，在片上累积结果','不落盘完整分数矩阵','execution','flash'],['PagedAttention','逻辑连续，物理分块存放','按块表找到需要的 KV','execution','paged']]}
  ];
  const NS='http://www.w3.org/2000/svg';
  function el(tag,attrs={},parent){const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(parent)parent.append(n);return n}
  function text(s,x,y,value,cls='',anchor='middle',maxWidth){
    const n=el('text',{x,y,class:cls,'text-anchor':anchor,'dominant-baseline':'middle'},s);
    const width=maxWidth||Math.max(24,s.viewBox.baseVal.width-16);
    n.textContent=value;
    if(n.getComputedTextLength()>width){
      const lines=[];let part='';
      for(const ch of Array.from(value)){n.textContent=part+ch;if(part&&n.getComputedTextLength()>width){lines.push(part);part=ch;}else part+=ch;}
      if(part)lines.push(part);n.replaceChildren();
      lines.forEach((v,i)=>{const t=el('tspan',{x,y:y+(i-(lines.length-1)/2)*15},n);t.textContent=v;});
    }
    return n;
  }
  function box(s,x,y,w,h,label,cls='node'){el('rect',{x,y,width:w,height:h,rx:6,class:cls},s);if(label)text(s,x+w/2,y+h/2,label,'','middle',w-10);return {x,y,w,h,cx:x+w/2,cy:y+h/2}}
  function line(s,x1,y1,x2,y2,color=C.line,dashed=false,arrow=true){const d=`M${x1},${y1} L${x2},${y2}`;el('path',{d,stroke:color,'stroke-width':1.5,fill:'none',...(dashed?{'stroke-dasharray':'4 4'}:{})},s);if(arrow){const a=Math.atan2(y2-y1,x2-x1),l=5;el('path',{d:`M${x2-l*Math.cos(a-.5)},${y2-l*Math.sin(a-.5)} L${x2},${y2} L${x2-l*Math.cos(a+.5)},${y2-l*Math.sin(a+.5)}`,stroke:color,'stroke-width':1.5,fill:'none'},s)}}
  function curve(s,x1,y1,x2,y2,color=C.line){el('path',{d:`M${x1} ${y1} C${x1} ${(y1+y2)/2},${x2} ${(y1+y2)/2},${x2} ${y2}`,fill:'none',stroke:color,'stroke-width':1.5},s);line(s,x2,y2-5,x2,y2,color)}
  function row(s,w,y,count,chosen,kind='cache',labels){const gap=5,sw=(w-16-gap*(count-1))/count;return Array.from({length:count},(_,i)=>{const active=chosen==null||chosen.includes(i);return box(s,8+i*(sw+gap),y,sw,31,labels?labels[i]:String(i+1),active?kind:'inactive')})}
  function down(s,w,from,to,color=C.line){line(s,w/2,from,w/2,to,color)}
  function lower(s,w,label,y=272){box(s,w*.17,y,w*.66,34,label,'node')}
  function heads(s,w,n){
    const gap=8,qw=(w-16-gap*3)/4;
    text(s,w/2,10,'4 个独立 Q 头','label');
    const qs=Array.from({length:4},(_,i)=>box(s,8+i*(qw+gap),30,qw,32,'Q'+i));
    const kw=(w-16-(n-1)*8)/n;
    const kvs=Array.from({length:n},(_,i)=>({x:8+i*(kw+8),y:128,w:kw,h:103,cx:8+i*(kw+8)+kw/2}));
    qs.forEach((q,i)=>curve(s,q.cx,62,kvs[Math.floor(i*n/4)].cx,128,[C.cyan,C.purple,C.green,C.orange][i]));
    kvs.forEach((k,i)=>{box(s,k.x,k.y,k.w,k.h,'','cache');text(s,k.cx,148,'KV'+i);const tw=(k.w-14)/3;for(let j=0;j<3;j++)el('rect',{x:k.x+5+j*(tw+2),y:178,width:Math.max(3,tw-1),height:17,rx:2,fill:'#2c6c78'},s);text(s,k.cx,216,'历史','label')});
    text(s,w/2,267,n===4?'每个 Q 读自己的 KV':n===2?'两个 Q 共用一套 KV':'四个 Q 共用一套 KV','emphasis');
  }
  function latent(s,w,mla){
    text(s,w/2,12,'因果可见历史','label');
    for(let i=0;i<3;i++){
      const y=36+i*59;text(s,9,y+18,'p'+(i+1),'label','start');
      if(mla){const cw=Math.max(58,(w-65)*.4),pw=Math.max(52,(w-65)*.25);box(s,44,y,cw,36,'latent','compressed');box(s,54+cw,y,pw,36,'位置 key');}
      else{const cw=(w-62)/2;box(s,44,y,cw,36,'各头 K','cache');box(s,50+cw,y,cw,36,'各头 V','cache');}
    }
    down(s,w,192,223,mla?C.purple:C.cyan);
    lower(s,w,mla?'通过投影 / 吸收使用':'直接使用各头 K/V',228);
    text(s,w/2,294,'每个历史位置仍有自己的记录','label');
  }
  function visibility(s,w,mode){
    const sel=mode==='full'?[0,1,2,3,4,5,6,7]:mode==='window'?[5,6,7]:[1,4,7];
    box(s,w/2-38,8,76,34,'当前 Q');
    const y=mode==='sparse'?154:154;
    if(mode==='sparse'){down(s,w,42,66);box(s,w/2-62,66,124,34,'索引器：选位置');text(s,w/2,231,'选择 ≠ 最终权重','label');}
    const cells=row(s,w,y,8,sel,'selected');
    const start=mode==='sparse'?100:42;
    sel.forEach(i=>curve(s,w/2,start,cells[i].cx,y,C.cyan));
    text(s,w/2,211,'因果可见位置 1 → 8','label');
    if(mode==='window'){el('path',{d:`M${cells[5].x},194 V201 H${cells[7].x+cells[7].w} V194`,stroke:C.orange,fill:'none'},s)}
    lower(s,w,'读取选中 KV → 输出',252);
  }
  function nsa(s,w,mode){
    text(s,w/2,10,'同一段因果可见历史','label');row(s,w,29,8,null,'cache');
    down(s,w,60,89,mode==='summary'?C.purple:mode==='window'?C.orange:C.cyan);
    if(mode==='summary'){
      box(s,w*.17,89,w*.66,33,'压缩成块摘要','compressed');down(s,w,122,153,C.purple);row(s,w,153,4,null,'compressed',['摘1','摘2','摘3','摘4']);
    }else if(mode==='select'){
      box(s,w*.1,89,w*.8,33,'按压缩路重要性选块');down(s,w,122,153,C.cyan);row(s,w,153,8,[4,5],'selected');
    }else{
      box(s,w*.17,89,w*.66,33,'保留最近窗口','local');down(s,w,122,153,C.orange);row(s,w,153,8,[6,7],'local');
    }
    down(s,w,184,225);lower(s,w,'本路 Attention',225);down(s,w,259,285,C.green);text(s,w/2,301,'本路输出','emphasis');
  }
  function compression(s,w,mode){
    text(s,w/2,9,'历史位置','label');row(s,w,27,8,null,'cache');down(s,w,58,82);
    if(mode==='qsa'){
      box(s,w*.13,82,w*.74,32,'压缩 indexer keys','compressed');
      down(s,w,114,134,C.purple);row(s,w,134,4,[1],'compressed',['摘1','摘2','摘3','摘4']);
      text(s,w/2,188,'选块 → 展开原位置','label');row(s,w,211,8,[2,3],'selected');
      line(s,w*.375,165,w*.375,179,C.purple);line(s,w*.375,199,w*.375,211,C.cyan);
      text(s,w/2,281,'主 Attention 读原 token','emphasis');
    }else{
      box(s,w*.13,82,w*.74,32,mode==='csa'?'压缩主 KV':'更强地压缩主 KV','compressed');
      down(s,w,114,143,C.purple);row(s,w,143,mode==='csa'?4:2,mode==='csa'?[1,3]:null,mode==='csa'?'selected':'compressed',mode==='csa'?['摘1','摘2','摘3','摘4']:['摘要 A','摘要 B']);
      text(s,w/2,202,mode==='csa'?'索引器挑选部分摘要':'全部可见摘要都参与','label');
      box(s,8,232,w-16,32,'＋ 近期原 token 窗口','local');
      text(s,w/2,294,'主 Attention 直接读压缩内容','emphasis');
    }
  }
  function state(s,w,mode){
    if(mode==='kv'){
      text(s,w/2,11,'之前的历史','label');row(s,w,32,5,null,'cache');down(s,w,63,73);down(s,w,97,103);
      text(s,w/2,88,'新 token 到来','label');row(s,w,103,6,null,'cache');
      text(s,w/2,164,'追加记录，历史继续增长','label');
      box(s,w/2-36,205,72,32,'当前 Q');down(s,w,237,262,C.cyan);lower(s,w,'逐位置查询历史',262);
    }else{
      text(s,w/2,11,'旧状态的衰减示意','label');
      const sx=w/2-67,cw=40;
      for(let r=0;r<3;r++)for(let c=0;c<3;c++)el('rect',{x:sx+c*46,y:32+r*23,width:cw,height:17,rx:3,fill:C.green,opacity:mode==='gdn'?.43:[.18,.5,.85][r]},s);
      text(s,w/2,119,mode==='gdn'?'整份一起淡化':'各 key 通道分别淡化','label');
      down(s,w,133,156,C.green);box(s,w*.1,156,w*.8,37,'预测 → 对比新值 → 纠正','state');
      text(s,w/2,215,'新信息更新同一形状的状态','label');
      down(s,w,230,255,C.green);lower(s,w,'当前 Q → 从新状态读出',255);
    }
  }
  function reuse(s,w,mode){
    box(s,12,10,w-24,38,mode==='full'?'本层生成全局 KV':'沿用最近 Full 的全局 KV',mode==='full'?'cache':'compressed');
    down(s,w,48,76,mode==='full'?C.cyan:C.purple);
    box(s,20,76,w-40,34,mode==='reuse'?'沿用已有 Top-K 位置':'本层 indexer Q 重新选择');
    down(s,w,110,140);row(s,w,140,6,mode==='full'?[1,4]:[2,5],'selected');
    text(s,w/2,192,mode==='full'?'本例选 2、5':mode==='reindex'?'本例改选 3、6':'沿用前层的 3、6','label');
    box(s,8,221,(w-24)/2,35,'本层 main Q');box(s,16+(w-24)/2,221,(w-24)/2,35,'本层 SWA','local');
    down(s,w,256,279,C.green);text(s,w/2,300,'本层重新计算输出','emphasis');
  }
  function execution(s,w,mode){
    if(mode==='paged'){
      text(s,w/2,11,'逻辑顺序','label');const logical=row(s,w,33,3,null,'cache',['块 A','块 B','块 C']);
      const physical=row(s,w,149,4,[0,2,3],'selected',['C','空','A','B']);
      [2,3,0].forEach((p,i)=>curve(s,logical[i].cx,64,physical[p].cx,149,[C.cyan,C.purple,C.orange][i]));
      text(s,w/2,208,'物理位置不必相邻','label');lower(s,w,'按逻辑顺序读取 KV',260);
    }else{
      box(s,8,8,w-16,34,'显存 HBM：Q / K / V');down(s,w,42,77);
      if(mode==='standard'){
        text(s,w/2,91,'中间分数矩阵写入显存','label');
        const sz=Math.min(19,(w-20)/7),x=w/2-sz*3.5;
        for(let r=0;r<6;r++)for(let c=0;c<6;c++)el('rect',{x:x+c*(sz+2),y:116+r*(sz+2),width:sz,height:sz,rx:2,fill:c<=r?'#2b6b78':'#142936'},s);
        down(s,w,250,273);text(s,w/2,296,'再读取 → 归一化 → 加权','emphasis');
      }else{
        box(s,12,78,w-24,148,'','state');text(s,w/2,100,'片上 SRAM','label');
        for(let i=0;i<3;i++)box(s,25+i*(w-50)/3,126,(w-62)/3,31,'块 '+(i+1),i===1?'selected':'cache');
        text(s,w/2,185,'逐块计算，在线归一化','label');text(s,w/2,207,'累积当前输出','label');
        down(s,w,226,267,C.green);lower(s,w,'最终输出写回显存',267);
      }
    }
  }
  const painters={heads,latent,visibility,nsa,compression,state,reuse,execution};

 const VARIANTS = [
  ['heads','MHA / GQA / MQA'],['mla','MLA'],['swa','SWA'],['dsa','DSA'],
  ['nsa','NSA'],['linear','GDN / KDA'],['qsa','QSA'],['csa','CSA'],['hca','HCA'],
  ['csa2','CSA2'],['flash','FlashAttention'],['paged','PagedAttention']
 ].map(([id,label])=>({id,label}));

 function mount(root) {
  if (!root || root.dataset.avMounted === 'true') return;
  root.dataset.avMounted='true';
  let current='heads';
  root.innerHTML = `  <div class="atlas-surface">
    <header class="atlas-header">
      <span class="atlas-eyebrow">ATTENTION · 机制对照</span>
      <h2>按机制分组，并排对照</h2>
    </header>
    <nav class="atlas-nav" aria-label="机制分组"></nav>
    <section class="atlas-stage" aria-labelledby="atlas-question">
      <div class="atlas-stage-heading">
        <h3 id="atlas-question"></h3>
        <span class="atlas-context"></span>
      </div>
      <div class="atlas-panels"></div>
      <div class="atlas-merge" hidden></div>
      <div class="atlas-takeaway" aria-live="polite"></div>
    </section>
    <footer class="atlas-footer">
      <span>结构示意 · 格子数不代表真实配置或速度</span>
      <span class="atlas-sources"></span>
    </footer>
  </div>
`;
  const nav=root.querySelector('.atlas-nav'),grid=root.querySelector('.atlas-panels');
  boards.forEach(b=>{const btn=document.createElement('button');btn.type='button';btn.textContent=b.tab;btn.dataset.board=b.id;btn.setAttribute('aria-pressed','false');btn.addEventListener('click',()=>{render(b.id);});nav.append(btn)});
  function drawAll(){grid.querySelectorAll('svg').forEach(s=>{const w=Math.round(s.getBoundingClientRect().width);if(!w)return;s.replaceChildren();s.setAttribute('viewBox',`0 0 ${w} 320`);s.setAttribute('height','320');const title=el('title',{},s);title.textContent=s.dataset.title;painters[s.dataset.kind](s,w,JSON.parse(s.dataset.option));})}
  function render(id){
    const b=boards.find(x=>x.id===id)||boards[0];current=b.id;root.dataset.activeBoard=b.id;
    nav.querySelectorAll('button').forEach(n=>n.setAttribute('aria-pressed',String(n.dataset.board===b.id)));
    root.querySelector('#atlas-question').textContent=b.question;root.querySelector('.atlas-context').textContent=b.context;
    grid.classList.toggle('two',b.panels.length===2);grid.replaceChildren();
    b.panels.forEach(([name,sub,bottom,kind,opt])=>{const p=document.createElement('article');p.className='atlas-panel';const h=document.createElement('h4');h.textContent=name;p.append(h);const st=document.createElement('div');st.className='atlas-panel-sub';st.textContent=sub;p.append(st);const s=el('svg',{role:'img','aria-label':`${name}：${sub}；${bottom}`});s.dataset.kind=kind;s.dataset.option=JSON.stringify(opt);s.dataset.title=`${name}：${sub}；${bottom}`;p.append(s);const foot=document.createElement('div');foot.className='atlas-panel-bottom';foot.textContent=bottom;p.append(foot);grid.append(p)});
    const merge=root.querySelector('.atlas-merge');merge.hidden=!b.merge;merge.textContent=b.merge||'';
    root.querySelector('.atlas-takeaway').textContent=b.note;
    const refs=root.querySelector('.atlas-sources');refs.replaceChildren();b.refs.forEach(k=>{const a=document.createElement('a');a.href=sources[k][1];a.textContent=sources[k][0]+' ↗';a.target='_blank';a.rel='noopener noreferrer';refs.append(a)});
    drawAll();
  }
  render('heads');
  let lastWidth=0;new ResizeObserver(()=>{const w=grid.clientWidth;if(w!==lastWidth){lastWidth=w;drawAll()}}).observe(grid);

 }

 function mountAll(){document.querySelectorAll('[data-attention-variants]').forEach(mount);}
 return {headsExample,mlaExample,swaExample,dsaExample,nsaExample,nsaBlockSelection,nsaGateMerge,
  linearStateExample,linearStep,VARIANTS,BOARDS:boards.map(({id,tab,context})=>({id,tab,context})),mount,mountAll};
});
