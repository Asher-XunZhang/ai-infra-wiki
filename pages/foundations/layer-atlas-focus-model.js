/* One deterministic decoder example. These values are not a trained-model trace. */
(function (root) {
  'use strict';
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const mv = (x, w) => w[0].map((_, j) => x.reduce((s, v, i) => s + v * w[i][j], 0));
  const softmax = x => { const m = Math.max(...x), e = x.map(v => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0); return e.map(v => v / z); };
  const silu = x => x / (1 + Math.exp(-x));
  const rms = x => Math.sqrt(dot(x, x) / x.length);
  const normalize = x => x.map(v => v / rms(x));
  const up = [[0.6, 0.2, -1], [0.4, 0.1, 0.2]];
  const gate = [[1.2, -0.8, 0.5], [0.4, 0.2, 0.4]];
  const down = [[1, 0.2], [0.1, -0.5], [0.4, 0.8]];
  const ffn = h => {
    const u = mv(h, up), g = mv(h, gate), a = g.map(silu), z = a.map((v, i) => v * u[i]);
    return { h, u, g, a, z, out: mv(z, down) };
  };
  const tokens = ['法国', '首都', '？'];
  const embeddings = [[0.8, -0.1], [-0.4, 0.7], [0.2, 0.3]];
  const gamma = [1.2, 0.8];
  const theta = 0.6;
  const rotate = ([x, y], angle) => [
    x * Math.cos(angle) - y * Math.sin(angle),
    x * Math.sin(angle) + y * Math.cos(angle)
  ];
  const scale = (matrix, factor) => matrix.map(row => row.map(v => v * factor));
  const norm = input => {
    const scaled = normalize(input);
    return { input, r: rms(input), scaled, gamma, out: scaled.map((v, i) => v * gamma[i]) };
  };
  const headWeights = [
    { q: [[0.7, 0.2], [0.3, -0.2]], k: [[0.9, -0.3], [0.2, 0.6]], v: [[0.7, 0.2], [0.2, 0.7]] },
    { q: [[0.1, 0.7], [-0.2, 0.4]], k: [[0.1, 0.7], [-0.5, 0.3]], v: [[0.4, -0.2], [0.2, 0.4]] }
  ];
  const wo = [[1, 0.1], [0, 1], [0.3, -0.2], [0.1, 0.4]];
  const add = (a, b) => a.map((v, i) => v + b[i]);
  const vocabulary = ['巴黎', '里昂', '北京', '马赛'];
  const vectors = [[0.9, 0.6], [0.1, -0.5], [-0.4, 0.3], [0.2, -0.2]];
  function layer(inputs, index) {
    const a = 1 + index * 0.1;
    const norms = inputs.map(norm);
    const projected = headWeights.map(w => ({
      queries: norms.map(({out}, i) => rotate(mv(out, scale(w.q, a)), i * theta)),
      keys: norms.map(({out}, i) => rotate(mv(out, w.k), i * theta)),
      rawKeys: norms.map(({out}) => mv(out, w.k)),
      rawQueries: norms.map(({out}) => mv(out, scale(w.q, a))),
      values: norms.map(({out}) => mv(out, scale(w.v, 1 / a)))
    }));
    return inputs.map((input, p) => {
      const heads = projected.map(h => {
        const q = h.queries[p], keys = h.keys.slice(0, p + 1), values = h.values.slice(0, p + 1);
        const scores = keys.map(k => dot(q, k) / Math.SQRT2), weights = softmax(scores);
        const out = [0, 1].map(i => values.reduce((s, v, j) => s + weights[j] * v[i], 0));
        return { q, keys, values, scores, weights, out };
      });
      const joined = heads.flatMap(h => h.out);
      const deltaA = mv(joined, wo), x1 = add(input, deltaA);
      const norm2 = norm(x1), mlp = ffn(norm2.out);
      const deltaM = mlp.out.map(v => v / a);
      return { input, norm1: norms[p], heads, joined, deltaA, x1, norm2, mlp, deltaM, out: add(x1, deltaM) };
    }).map((row, p) => ({...row, rawQ: projected[0].rawQueries[p], rawKeys: projected[0].rawKeys}));
  }
  const first = layer(embeddings, 0);
  const middle = layer(first.map(row => row.out), 1);
  const last = layer(middle.map(row => row.out), 2);
  const trace = [first.at(-1), middle.at(-1), last.at(-1)];
  const initial = trace[0];
  const finalNorm = norm(trace.at(-1).out);
  const logits = vectors.map(v => dot(finalNorm.out, v));
  const handoffs = [
    ['Embedding → 第 1 层', initial.input],
    ['RMSNorm → Q/K/V', initial.norm1.out],
    ['Attention → 残差', initial.deltaA],
    ['第一次残差 → RMSNorm', initial.x1],
    ['RMSNorm → FFN', initial.norm2.out],
    ['FFN → 残差', initial.deltaM],
    ['第 1 层 → 第 2 层', initial.out],
    ['Final Norm → LM Head', finalNorm.out],
    ['LM Head → 生成选择', logits]
  ];
  const keyFor = id => ({tokens:'embed',embed:'embed',norm1:'norm',norm2:'norm',finalnorm:'norm',attn:'attention',add1:'residual',add2:'residual',ffn:'ffn',repeat:'layers',head:'head',next:'generation'}[id]);
  function make(kind, variant = 0, id = '') {
    const scene = { kind, variant, id, steps: [], durations: [], data: {} };
    const step = (title, why, formula, duration = 3) => { scene.steps.push({ title, why, formula }); scene.durations.push(duration); };
    if (kind === 'embed') {
      scene.title = 'Embedding · 从编号取出初始向量';
      scene.data = { id: 17, rows: [[5, tokens[1], embeddings[1]], [17, tokens[2], embeddings[2]], [23, tokens[0], embeddings[0]]], out: initial.input };
      step('定位第 17 行', 'ID 用来选行；编号大小本身不表达词义。', 'embedding');
      step('读出同一行的向量', '读出的两个分量成为后续计算的起点，token 数量没有增加。', 'embedding');
    } else if (kind === 'ffn') {
      scene.title = 'FFN / SwiGLU · 展开、调节、再组合'; scene.data = initial.mlp;
      step('两路投影同时展开：2 → 3', '同一个输入并行生成 u 与 g；增加可加工的特征通道。', 'ffn_project', 3.8);
      step('SiLU 连续调节门控响应', '三条门控通道各自经过非线性，保留负响应的可能。', 'silu', 3.2);
      step('逐维相乘：仍是 3 个通道', 'u 与 a 按通道编号一一配对，改变各通道的贡献。', 'gate', 3.4);
      step('下投影组合：3 → 2', '三条通道共同参与两个输出分量，得到可与残差相加的新更新。', 'ffn_down', 3.8);
    } else if (kind === 'norm') {
      scene.title = id === 'finalnorm' ? 'Final Norm · 输出打分前调整尺度' : 'RMSNorm · 调整当前向量的尺度';
      const base = id === 'norm2' ? initial.norm2 : id === 'finalnorm' ? finalNorm : initial.norm1;
      scene.data = variant ? norm(base.input.map(v => v * 10)) : base;
      step('从两个分量求整体幅度', '平方后求均值再开方，得到当前向量的 RMS。', 'rms_demo', 3.2);
      step('每个分量除以同一个 RMS', '缩放不改变维度；整体放大 10 倍的输入会回到相同尺度。', 'rms_divide', 3.2);
      step('乘逐维系数 γ', '训练得到的系数再调节各维，输出保持两个分量。', 'rms_gain', 3);
    } else if (kind === 'residual') {
      scene.title = id === 'add2' ? 'FFN 残差 · 将更新加回 x₁' : 'Attention 残差 · 将更新加回 x';
      const isFFN = id === 'add2';
      scene.data = { input: isFFN ? initial.x1 : initial.input, delta: isFFN ? initial.deltaM : initial.deltaA, out: isFFN ? initial.out : initial.x1, base: isFFN ? 'x₁' : 'x', deltaName: isFFN ? 'ΔM' : 'ΔA' };
      step('原表示等到子层更新', '已有表示沿旁路保留，两个同宽向量就绪后才能相加。', isFFN ? 'residual_f' : 'residual_a', 2.3);
      step('对应分量相加', '每一维独立相加，结果保持 2 维。加法完成后将结果交给下一模块。', isFFN ? 'residual_f' : 'residual_a', 2.6);
    } else if (kind === 'attention' || kind === 'heads') {
      const { heads, joined, deltaA } = initial;
      scene.data = { labels: tokens, input: initial.norm1.out, heads, ...heads[0] };
      if (kind === 'attention') {
        scene.title = 'Attention · 计算联系，按权重汇聚内容';
        step('Q 与各位置的 K 匹配', '三个可见位置同时计算匹配；未来位置被因果掩码阻断。', 'score_demo', 3.7);
        step('分数变成位置权重', '保留各位置的分数，按指数权重归一化。填充的长度只表示权重条展开的进度。', 'softmax_attention', 2);
        step('按权重汇聚 V', '三份加权内容共同形成当前输出，权重决定每个位置贡献多少。', 'attention_mix', 4);
      } else {
        scene.title = '多头 Attention · 分别计算，再混合结果';
        Object.assign(scene.data, { joined, out: deltaA });
        step('两头同时计算各自的联系', '两头读取相同的上下文，使用各自投影。公式中的 a 是头编号，softmax 按每个 Query 的那一行计算。', 'head_attention', 3);
        step('拼接两个头的结果', '本例每头 2 维，拼接后暂时形成 4 维。', 'concat_heads', 3.4);
        step('W_O 混合各头贡献', '本例映射回 2 维；一般输入输出也可能同宽，W_O 的作用是学习混合。', 'attention_projection', 3.6);
      }
    } else if (kind === 'rope') {
      scene.title = 'RoPE · 让匹配包含相对位置';
      const p = 2, r = variant ? 1 : 0, q = initial.rawQ, k = initial.rawKeys[r];
      const qr = rotate(q, p * theta), kr = rotate(k, r * theta);
      scene.data = { theta, p, r, q, k, qr, kr, key: tokens[r], out: [dot(qr, kr)] };
      step('Q 与 K 按各自位置旋转', '旋转保留向量长度。两路独立计算，位置 0 的 K 保持原方向。', 'rope_demo', 3);
      step('将点积缩放为注意力分数', '夹角越接近，固定长度下的点积越大。此分数接回「汇聚信息」中对应的位置。', 'rope_score', 1.6);
    } else if (kind === 'layers') {
      scene.title = '多层堆叠 · 同一位置的表示接续更新';
      const states = trace.map(row => row.out);
      scene.data = { states };
      for (const layerNumber of [2, 3]) step(`第 ${layerNumber} 层更新表示`, '各层从上一层的表示生成本层的 Q/K/V，再更新当前位置。第 3 层的输出接到 Final Norm。', `layer_${layerNumber}`, 2.4);
    } else if (kind === 'head' || kind === 'generation') {
      const input = finalNorm.out, names = vocabulary, scores = logits, probabilities = softmax(scores);
      let selected = scores.indexOf(Math.max(...scores));
      if (variant) { let cumulative = 0; selected = probabilities.findIndex(p => (cumulative += p) >= 0.8); }
      scene.data = { input, names, vectors, scores, probabilities, selected, random: 0.8 };
      if (kind === 'head') {
        scene.title = 'LM Head · 把内部特征映射成词表分数';
        step('同时计算四个候选的分数', '最终表示分别与四组候选权重做点积，再汇成词表分数向量。', 'logit', 4);
      } else {
        scene.title = 'Softmax 与生成选择 · 概率、选择、接回上下文';
        step('候选分数转成概率', 'softmax 保持分数次序，并把数值变成总和为 1 的分布。', 'softmax_vocab', 3.8);
        step(variant ? '按分布采样一个候选' : '选出最高分候选', variant ? '本例取 u 为 0.8，选累计概率首次到达这个数的位置。' : '贪心选择最高分 token，可以直接比较 logits。', variant ? 'sampling' : 'greedy', 3.5);
        step('追加到上下文，准备下一轮', '若未结束生成，选中的 token 接回上下文。', '选中的 token 成为下一轮的新输入位置', 1.5);
      }
    }
    if (!scene.steps.length) throw new Error('Unknown focus kind: ' + kind);
    scene.ends = []; let total = 0; scene.durations.forEach(d => scene.ends.push(total += d)); scene.duration = total;
    return scene;
  }
  function sample(scene, time) {
    const t = Math.max(0, Math.min(scene.duration, time));
    let phase = scene.ends.findIndex(end => t < end); if (phase < 0) phase = scene.ends.length - 1;
    const start = phase ? scene.ends[phase - 1] : 0, p = Math.max(0, Math.min(1, (t - start) / scene.durations[phase]));
    return { time: t, phase, p, complete: i => t >= scene.ends[i], progress: i => Math.max(0, Math.min(1, (t - (i ? scene.ends[i - 1] : 0)) / scene.durations[i])) };
  }
  function fullRun() {
    // Each transfer keeps the output produced by its scene. Junctions carry the same value.
    const chapters = [
      { id:'embed', overview:0, label:'Embedding', edges:['embed-x','x-norm1'], value:initial.input, symbol:'x', branch:'bypass1' },
      { id:'norm1', overview:1, label:'RMSNorm · Attention 前', edges:['norm1-attn'], value:initial.norm1.out, symbol:'h' },
      { id:'attn', overview:2, label:'Attention', edges:['attn-add1'], value:initial.deltaA, symbol:'ΔA',
        parts:[{kind:'rope'},{kind:'attention'},{kind:'heads',skip:1}] },
      { id:'add1', overview:3, label:'Attention 残差', edges:['add1-x1','x1-norm2'], value:initial.x1, symbol:'x₁', branch:'bypass2' },
      { id:'norm2', overview:4, label:'RMSNorm · FFN 前', edges:['norm2-ffn'], value:initial.norm2.out, symbol:'h₂' },
      { id:'ffn', overview:5, label:'FFN / SwiGLU', edges:['ffn-add2'], value:initial.deltaM, symbol:'ΔM' },
      { id:'add2', overview:6, label:'FFN 残差', edges:['add2-y','y-repeat'], value:initial.out, symbol:'y' },
      { id:'repeat', overview:7, label:'第 2、3 层', edges:['repeat-finalnorm'], value:trace.at(-1).out, symbol:'第 3 层输出' },
      { id:'finalnorm', overview:7, label:'Final Norm', edges:['finalnorm-head'], value:finalNorm.out, symbol:'h_final' },
      { id:'head', overview:8, label:'LM Head', edges:['head-next'], value:logits, symbol:'logits' },
      { id:'next', overview:8, label:'生成选择' }
    ];
    const phases = [];
    let duration = 0;
    function append(fields, seconds) {
      const start = duration;
      duration += seconds;
      phases.push({ ...fields, start, end:duration });
    }
    chapters.forEach((chapter, index) => {
      const parts = chapter.parts || [{ kind:keyFor(chapter.id) }];
      chapter.start = duration;
      parts.forEach(({kind, skip}, part) => {
        const scene = make(kind, 0, chapter.id);
        // The two heads compute in parallel. In the merge view, start with both results ready.
        const from = skip ? scene.ends[skip - 1] : 0;
        append({ index, chapter, kind, from, until:scene.duration, type:'compute', enters:part===0 }, (scene.duration - from) / 1.65);
      });
      if (chapter.edges) {
        const {kind,until}=phases.at(-1);
        append({ index, chapter, target:chapters[index+1], kind, until, type:'transfer' }, chapter.edges.length===2 ? 1.9 : 1.4);
      }
    });
    const lastScene = make('generation', 0, 'next').data;
    return { chapters, phases, duration, selected:lastScene.names[lastScene.selected] };
  }

  function sampleRun(run, time) {
    const bounded = Math.max(0, Math.min(run.duration, time));
    const ended = bounded === run.duration;
    const phase = (ended ? run.phases.at(-1) : run.phases.find(p=>bounded<p.end));
    const progress = (bounded-phase.start)/(phase.end-phase.start);
    return { ...phase, time:bounded, progress, ended };
  }
  const api = { make, sample, keyFor, dot, mv, softmax, silu, rms, normalize, ffn, rotate, handoffs, fullRun, sampleRun };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.LayerAtlasFocusModel = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
