/* Chapter 1, lessons 01–04. Compact mechanism boards sized for the shared
 * camera stage: the drawing itself carries input, operation, condition and
 * result. Every node persists across playback; paint() only moves or fades it. */
(() => {
'use strict';
const V = window.MechanismVisuals;
const D = window.TeachingDrawing;
if (!V || !D) return;
const {board, phase, lerp, clamp} = D;
const fmtVec = (v, digits = 2) => '[' + v.map(x => {
  const n = Number(x.toFixed(digits));
  return Number.isInteger(n) ? String(n) : String(n);
}).join(', ') + ']';
const pct = n => (n * 100).toFixed(1) + '%';
const textOf = (e, value) => { if (e.textContent !== value) e.textContent = value; };

/* ------------------------------------------------------------------ 01 tokens */
V['f-tokens'] = (svg, viewport, s) => board(svg, viewport, 'tokens', () => 520, H => {
  const {txt, rect, path, text, box, show, wire, token, move, set} = H;
  const d = s.data;
  const vectors = {104100: [.5, -.2, .4], 100134: [.2, .8, -.1], 104455: [.7, .1, -.3], 1773: [-.1, .4, .6], 102182: [-.4, .3, .5]};
  const occurrences = d.pieces.map((word, i) => ({word, id: d.ids[i], position: i + 1}));
  const unique = [];
  occurrences.forEach(o => {
    let row = unique.find(x => x.id === o.id);
    if (!row) { row = {id: o.id, word: o.word, positions: [], vector: vectors[o.id] || [.2, .2, .2]}; unique.push(row); }
    row.positions.push(o.position);
  });

  const heading = txt('原句 → 可复用片段 → 词表编号 → 一行向量', 'teach-heading');
  const sentenceLabel = txt('你输入的原句（本例未加聊天模板）', 'teach-small');
  const lookupHead = txt('词表与 Embedding 表 · 只列本例用到的项', 'teach-small');
  const outHead = txt('按出现位置排列：同一个编号可以出现两次，位置不能合并', 'teach-small');
  const note = txt('编号是查表索引；向量是手算假设，不是模型权重。p1…p5 表示出现顺序。', 'teach-small'), note2 = txt('p1…p5 表示出现顺序，不是词表编号。', 'teach-small');
  const shape = txt('词表：4 项 · 每行 3 维', 'teach-small');

  const pieces = occurrences.map((o, i) => token('occ-' + (i + 1), o.word, Math.max(42, o.word.length * 15 + 16), 'teach-token teach-color-' + (unique.findIndex(u => u.id === o.id) % 5)));
  const positions = occurrences.map((o, i) => txt('p' + o.position, 'teach-small'));
  const rows = unique.map((u, i) => ({
    bg: rect('teach-row teach-color-' + (i % 5)),
    piece: token('row-piece-' + i, u.word, 72, 'teach-token teach-color-' + (i % 5)),
    id: txt(String(u.id), 'teach-number'),
    occurrences: txt('出现 p' + u.positions.join('、p'), 'teach-small'),
    vector: txt('', 'teach-number'),
    lookup: path('teach-link'),
    embed: path('teach-link')
  }));
  const outputs = occurrences.map((o, i) => {
    const row = unique.findIndex(u => u.id === o.id);
    const bg = rect('teach-row teach-color-' + (row % 5));
    const label = txt('p' + o.position, 'teach-small');
    const node = txt('', 'teach-small');
    return {id: o.id, word: o.word, position: o.position, row, node, bg, label};
  });

  return (p, m) => {
    const W = m ? 360 : 720, chipY = 58, tableY = 132, rowH = 58;
    text(heading, 16, 25, m ? '原句 → 片段 → 编号 → 向量' : '原句 → 可复用片段 → 词表编号 → 一行向量');
    text(sentenceLabel, 16, 44); show(sentenceLabel, p < 3 ? 1 : .55);
    // The sentence is one unit until step 2; pieces then separate continuously,
    // while each occurrence keeps the same token identity and position label.
    const spread = phase(p, 0, 1), sepGap = 6;
    const joinedGap = 1, totalSep = pieces.reduce((sum, q) => sum + q.width, 0) + sepGap * (pieces.length - 1);
    const sepStart = m ? Math.max(16, (W - totalSep) / 2) : 16;
    let joinX = 16, sepX = sepStart;
    pieces.forEach((a, i) => {
      const x = lerp(joinX, sepX, spread);
      move(a, x, chipY, 1);
      text(positions[i], x + a.width / 2 - 6, chipY + 38);
      show(positions[i], p >= 1 ? 1 : 0);
      joinX += a.width + joinedGap;
      sepX += a.width + sepGap;
    });
    text(lookupHead, 16, tableY - 10); show(lookupHead, p >= 1 ? 1 : 0);
    text(shape, 360, tableY - 10); show(shape, !m && p >= 3 ? 1 : 0);
    rows.forEach((r, i) => {
      const y = tableY + i * rowH, ax = m ? 20 : 20, idX = m ? 128 : 136, vecX = m ? 232 : 246;
      box(r.bg, 12, y - 5, m ? 336 : 696, 50); show(r.bg, p >= 1 ? .28 + .5 * phase(p, 1) : 0); r.bg.dataset.tokenRow = String(i + 1);
      move(r.piece, ax, y, p >= 1 ? phase(p, 1) : 0);
      text(r.id, idX, y + 27); show(r.id, p >= 2 ? phase(p, 1.8, 2) : 0);
      text(r.occurrences, idX, y + 44); show(r.occurrences, p >= 2 ? phase(p, 1.8, 2) : 0);
      text(r.vector, vecX, y + 27, fmtVec(unique[i].vector, m ? 1 : 2)); show(r.vector, p >= 3 ? phase(p, 2.8, 3) : 0);
      wire(r.lookup, `M${ax + r.piece.width} ${y + 17} H${idX - 8}`, p >= 2 ? phase(p, 1.8, 2.2) : 0);
      wire(r.embed, `M${idX + (m ? 62 : 76)} ${y + 17} H${vecX - 6}`, p >= 3 ? phase(p, 2.8, 3.2) : 0);
    });
    text(outHead, 16, 380, m ? '同一编号可出现两次；位置不能合并' : '按出现位置排列：同一个编号可以出现两次，位置不能合并'); show(outHead, p >= 4 ? 1 : 0);
    outputs.forEach((o, i) => {
      const n = occurrences.length, gap = m ? 4 : 6, slotW = (W - 32 - gap * (n - 1)) / n;
      const sx = 16 + i * (slotW + gap), sy = 402;
      box(o.bg, sx, sy, slotW, 38); show(o.bg, p >= 4 ? 1 : 0);
      text(o.label, sx, sy - 6); show(o.label, p >= 4 ? 1 : 0);
      const row = rows[o.row], srcX = (m ? 232 : 246) + 4, srcY = tableY + o.row * rowH + 2;
      const u = p >= 4 ? phase(p, 3.55, 4) : 0;
      const compact = '[' + (vectors[o.id] || [.2, .2, .2]).map(x => {
        const v = x.toFixed(1); return v.replace('0.', '.').replace('-.', '-.');
      }).join(',') + ']';
      text(o.node, 0, 0, m ? compact : fmtVec(vectors[o.id] || [.2, .2, .2], 2));
      set(o.node, {'text-anchor': 'middle', class: 'teach-small'});
      move({g: o.node, r: o.bg, t: o.node}, lerp(srcX, sx + slotW / 2, u), lerp(srcY, sy + 25, u), u);
      show(o.node, u);
    });
    text(note, 16, m ? 466 : 486, m ? '编号是查表索引；向量是手算假设，不是模型权重。' : '编号是查表索引；向量是手算假设，不是模型权重。p1…p5 表示出现顺序。'); show(note, 1);
    text(note2, 16, m ? 500 : 506, 'p1…p5 只是出现位置。'); show(note2, m ? 1 : 0);
  };
});

/* -------------------------------------------------------------- 02 generation */
V['f-generation'] = (svg, viewport, s) => board(svg, viewport, 'generation', () => 520, H => {
  const {txt, rect, path, text, box, show, wire, token, move, set} = H;
  const d = s.data, first = d.first, second = d.second;
  // Zones are created before the tokens so a static panel never covers a moving object.
  const inputZone = rect(), modelZone = rect(), candidateZone = rect(), outputZone = rect(), kvZone = rect();
  const inputA = token('input-a', '我', 54, 'teach-token teach-color-0');
  const inputB = token('input-b', '想', 54, 'teach-token teach-color-0');
  const inputC = token('input-copy', first, 58, 'teach-token teach-color-2');
  const out1 = token('output-1', first, 58, 'teach-token teach-color-2');
  const out2 = token('output-2', second, 58, 'teach-token teach-color-3');
  const kv1 = token('kv-my', '我 的 KV', 78, 'teach-token teach-color-0');
  const kv2 = token('kv-want', '想 的 KV', 78, 'teach-token teach-color-0');
  const kv3 = token('kv-first', first + ' 的 KV', 78, 'teach-token teach-color-2');
  const title = txt('', 'teach-heading'), round = txt('', 'teach-small');
  const inputTitle = txt('本轮输入', 'teach-small'), modelTitle = txt('完整模型前向（层结构从略）', 'teach-small');
  const modelNote = txt('', 'teach-small'), candTitle = txt('下一 token 候选', 'teach-small'), candNote = txt('', 'teach-small');
  const outputTitle = txt('输出序列', 'teach-small'), client = txt('', 'teach-small'), kvTitle = txt('历史 KV：只保留已经计算过的输入位置', 'teach-small');
  const kvEmpty = txt('尚无历史；Prefill 完成后才出现', 'teach-small'), kvNote = txt('', 'teach-small');
  const inputLabelC = txt('新输入 · 来自上轮', 'teach-small'), inputLabelCached = txt('已缓存 · 不重送', 'teach-small');
  const cacheBracket = path('teach-link'), newBracket = path('teach-link');
  const inputFlow = path('teach-link'), outputFlow = path('teach-link');
  const candRows = [0, 1, 2].map(i => ({bg: rect('teach-focus'), name: txt('', 'teach-number'), track: rect('teach-track'), bar: rect('teach-value teach-color-' + (i + 2)), value: txt('', 'teach-number')}));
  const interval = [0, 1, 2].map(i => ({bar: rect('teach-contribution teach-color-' + (i + 2)), label: txt('', 'teach-small')}));
  const needle = path('teach-map'), intervalNote = txt('', 'teach-small');
  const waitBox = rect('teach-memory-slot'), waitText = txt('等待模型计算新输入', 'teach-small');
  const loop = path('teach-link'), loopLabel = txt('反馈：同一个 token ID 进入下一轮；输出序列保留原件', 'teach-small');
  const boundary = txt('教学轮次只说明依赖，不表示 GPU 耗时；分布是手算值，不是正确率。', 'teach-small');
  const shape = txt('', 'teach-small');

  return (p, m) => {
    const round2 = p >= 4, showDist1 = p >= 2 && p < 4, showDist2 = p >= 5;
    const inY = m ? 72 : 94, inX = [30, m ? 96 : 106, m ? 162 : 174];
    text(title, 16, 24, round2 ? '第 2 轮 · Decode' : '第 1 轮 · Prefill');
    text(round, 16, 44, innerWidth <= 340 ? (round2 ? '送回上轮输出，只算 1 个新位置；历史仍读取' : '处理 2 个输入位置，留下 K/V，得到候选') : (round2 ? '把上一轮输出送回，只计算 1 个新位置；历史仍需读取' : '处理 2 个输入位置，留下 K/V，得到第一组候选'));
    box(inputZone, 16, m ? 52 : 54, m ? 328 : 300, m ? 98 : 112); box(modelZone, 16, m ? 158 : 176, m ? 328 : 300, m ? 42 : 48);
    box(candidateZone, m ? 16 : 336, m ? 208 : 54, m ? 328 : 368, m ? 122 : 286);
    box(outputZone, 16, m ? 336 : 236, m ? 328 : 300, m ? 80 : 96); box(kvZone, 16, m ? 422 : 344, m ? 328 : 300, m ? 86 : 100);
    text(inputTitle, 28, m ? 70 : 76, round2 ? '本轮送入前向：1 个新 token' : '本轮输入：2 个位置'); text(modelTitle, 28, m ? 174 : 200); text(modelNote, 28, m ? 196 : 218, round2 ? '读取历史，只算新位置' : '处理输入，留下 K/V');
    text(candTitle, m ? 28 : 350, m ? 214 : 80, innerWidth <= 340 ? '下一 token 候选' : '下一 token 候选（手工分布）'); text(candNote, 0, 0, '');
    text(outputTitle, 28, m ? 346 : 258, '模型侧输出序列'); text(client, m ? (innerWidth <= 340 ? 190 : 170) : 430, m ? 410 : 288, p >= 6 ? (m ? '文字另经反分词与传输' : '客户端文字另经反分词与传输') : p >= 3 ? (m ? '客户端尚未收到' : '客户端尚未收到：ID 还要反分词与传输') : '客户端尚未收到文字');
    text(kvTitle, 28, m ? 434 : 366); text(kvEmpty, 28, m ? 466 : 386, '尚无历史；Prefill 完成后才出现'); show(kvEmpty, p < 2);
    text(boundary, 16, 500); show(boundary, m ? 0 : 1); text(shape, 16, m ? 480 : 480, round2 ? '本轮新输入 1 个位置；历史仍需读取' : '本轮输入 2 个位置'); show(shape, p >= 7 || m ? 0 : 1);
    move(inputA, inX[0], inY, 1); move(inputB, inX[1], inY, 1);
    text(inputLabelCached, inX[0], m ? 126 : 148);
    text(inputLabelC, inX[2], m ? 126 : 148);
    show(inputLabelCached, p >= 4 ? 1 : 0); show(inputLabelC, p >= 4 ? 1 : 0);
    const bracketY = m ? 112 : 136, bracketEnd = m ? 150 : 160;
    set(cacheBracket, {d: `M${inX[0]} ${bracketY} H${bracketEnd}`, 'stroke-dasharray': '4 3', 'marker-end': 'none'});
    set(newBracket, {d: `M${inX[2]} ${bracketY} H${inX[2] + 58}`, 'stroke-dasharray': '4 3', 'marker-end': 'none'});
    show(cacheBracket, p >= 4 ? 1 : 0); show(newBracket, p >= 4 ? 1 : 0);
    inputA.g.dataset.context = round2 ? 'cached' : 'new'; inputB.g.dataset.context = round2 ? 'cached' : 'new'; inputC.g.dataset.context = 'new';
    inputA.g.dataset.forwarded = round2 ? 'false' : 'true'; inputB.g.dataset.forwarded = round2 ? 'false' : 'true'; inputC.g.dataset.forwarded = 'true';
    // Only the thin return wire travels. A wide token cannot cross either
    // column on a phone (the panels span almost the full 360px), so the copy
    // appears at the new input when round 2 starts instead of covering text.
    move(inputC, inX[2], inY, p >= 4 ? 1 : 0);
    wire(loop, m ? 'M30 394 H12 V112 H162' : 'M30 284 H12 V130 H174', phase(p, 3.3, 4));
    text(loopLabel, 16, 452); show(loopLabel, p >= 4 && !m ? 1 : 0);
    wire(inputFlow, m ? 'M166 150 V158' : 'M166 166 V176', p >= 1 ? phase(p, .6, 1) : 0);
    wire(outputFlow, m ? 'M344 179 H352 V216 H344' : 'M316 200 H336', p >= 2 ? phase(p, 1, 2) : 0);

    const dist = showDist2 ? d.nextScores : d.scores;
    const choice = showDist2 ? d.nextChoice : p < 4 ? d.choice : null;
    const selectedIndex = p >= 6 ? dist.findIndex(x => x.word === second) : p >= 3 && p < 4 ? dist.findIndex(x => x.word === first) : -1;
    const grow = showDist2 ? phase(p, 4.35, 5) : phase(p, 1.4, 2);
    candRows.forEach((r, i) => {
      const item = dist[i], visible = (showDist1 || showDist2) && !!item;
      const y = m ? 232 + i * 22 : 102 + i * 34, bx = m ? 88 : 420, bw = m ? 150 : 168;
      show(r.bg, visible && selectedIndex === i ? 1 : 0);
      box(r.bg, m ? 22 : 342, y - 5, m ? 312 : 350, 24);
      text(r.name, m ? 28 : 350, y + 13, item ? item.word : ''); show(r.name, visible ? 1 : 0);
      box(r.track, bx, y, bw, 13); show(r.track, visible ? 1 : 0);
      box(r.bar, bx, y, bw * (item ? item.p : 0) * grow, 13); show(r.bar, visible ? 1 : 0);
      text(r.value, m ? 250 : 598, y + 12, item ? pct(item.p) + (m ? ' z=' : ' · z=') + item.z.toFixed(2) : ''); show(r.value, visible ? 1 : 0);
    });
    // Keep the raw-score step next to the probability bars. The bars are p,
    // not z; the chain makes that distinction visible without changing the
    // existing sampling/interval model.
    const chain = showDist2
      ? 'z=[ln4,0] → e^z=[4,1] → p=[.8,.2]（÷5）'
      : 'z=[ln6,ln3,0] → e^z=[6,3,1] → p=[.6,.3,.1]（÷10）';
    // Put the raw-score chain above the probability bars so it clears both the
    // sampling note below the bars and the client text further down.
    text(candNote, 350, 204, chain);
    show(candNote, (showDist1 || showDist2) && !m ? 1 : 0);
    box(waitBox, m ? 28 : 350, m ? 252 : 140, m ? 300 : 340, 38); show(waitBox, p >= 4 && p < 5 ? 1 : 0);
    text(waitText, m ? 40 : 362, m ? 276 : 164, p < 4.5 ? '等待模型计算新输入' : '正在计算新输入，候选尚未产生'); show(waitText, p >= 4 && p < 5 ? 1 : 0);
    const intervals = choice ? choice.intervals : [], start = m ? 28 : 350, iw = m ? 304 : 340, iy = m ? 300 : 214;
    interval.forEach((seg, i) => {
      const item = intervals[i], on = item && choice && choice.draw !== null;
      show(seg.bar, on ? 1 : 0); show(seg.label, on ? 1 : 0);
      if (!item) return;
      box(seg.bar, start + item.start * iw, iy, Math.max(1, item.p * iw), 15);
      text(seg.label, start + item.start * iw + 4, iy + 9, item.word);
    });
    const draw = choice && choice.draw !== null ? choice.draw : null;
    const picked = intervals.find(x => draw !== null && draw >= x.start - 1e-9 && draw <= x.start + x.p + 1e-9);
    const bounds = picked ? `${picked.start.toFixed(1)}–${(picked.start + picked.p).toFixed(1)}` : '';
    wire(needle, `M${start + (draw || 0) * iw} ${iy - 9} V${iy - 1}`, draw !== null ? 1 : 0);
    text(intervalNote, start, iy + 29, draw !== null ? (m ? `u=${draw} → ${choice.selected} 区间` : `固定抽样数 u=${draw} 落在 ${bounds} → ${choice.selected}`) : (showDist1 || showDist2 ? '贪心：取本轮概率最大的候选' : ''));
    show(intervalNote, (showDist1 || showDist2) ? 1 : 0);

    const outY = m ? 358 : 270, sel1 = d.scores.findIndex(x => x.word === first), sel2 = d.nextScores.findIndex(x => x.word === second);
    // Emit at the step boundary. A diagonal flight would cross the candidate
    // rows and the sampling note; the candidate highlight already shows which
    // row was selected, and the output appears when the selection is complete.
    const u1 = p >= 3 ? 1 : 0, u2 = p >= 6 ? 1 : 0;
    move(out1, lerp(m ? 28 : 350, 30, u1), lerp(m ? 232 : 102 + sel1 * 34, outY, u1), u1);
    move(out2, lerp(m ? 28 : 350, m ? 100 : 100, u2), lerp(m ? 232 : 102 + sel2 * 34, outY, u2), u2);
    out1.g.dataset.output = 'first'; out2.g.dataset.output = 'second';

    const kvY = m ? 448 : 386;
    move(kv1, 30, kvY, p >= 2 ? 1 : 0); move(kv2, 110, kvY, p >= 2 ? 1 : 0);
    // The K/V write completes at step 5; showing it in place avoids a vertical
    // flight through the "历史 KV" caption that spans the panel.
    move(kv3, 190, kvY, p >= 5 ? 1 : 0);
    kv1.g.dataset.kv = '1'; kv2.g.dataset.kv = '2'; kv3.g.dataset.kv = '3';
    kv1.g.dataset.kvWritten = p >= 2 ? 'true' : 'false'; kv2.g.dataset.kvWritten = p >= 2 ? 'true' : 'false'; kv3.g.dataset.kvWritten = p >= 5 ? 'true' : 'false';
    inputC.g.dataset.inputCopy = p >= 4 ? 'true' : 'false';
    text(kvNote, 28, m ? 496 : 430, p >= 5 ? `已计算：我、想、${first}；${second} 尚未作为输入` : p >= 4.5 ? '正在计算新输入；K/V 写入尚未完成' : p >= 2 ? '已计算：我、想；新 token 尚未经过模型' : '尚无历史');
    show(kvNote, p >= 1 ? 1 : 0);
  };
});

/* --------------------------------------------------------------- 03 attention */
V['f-attention'] = (svg, viewport, s) => board(svg, viewport, 'attention', m => m ? 696 : 484, H => {
  const {txt, rect, path, text, box, show, wire, token, move, set} = H;
  const d = s.data, allowed = d.query + 1, qIndex = d.query;
  const exps = d.scores.map(x => Number.isFinite(x) ? Math.exp(x) : 0);
  const total = exps.reduce((a, b) => a + b, 0);
  const compactVec = v => '[' + v.map(n => Number.isInteger(n) ? String(n) : String(Number(n.toFixed(1)))) + ']';
  const expText = exps.slice(0, allowed).map(x => x.toFixed(2));
  const denomFirst = '总和 ≈ ' + expText.slice(0, 2).join(' + ');
  const denomRest = expText.slice(2).join(' + ');
  const denomSecond = (denomRest ? '+ ' + denomRest + ' ' : ' ') + '≈ ' + total.toFixed(2);
  const currentWeight = `p${qIndex + 1}：${exps[qIndex].toFixed(2)} ÷ ${total.toFixed(2)} ≈ ${pct(d.weights[qIndex])}`;
  const totalWeight = (d.weights.slice(0, allowed).reduce((a, b) => a + b, 0) * 100).toFixed(1) + '%';
  const roundedSum = exps.slice(0, allowed).reduce((a, b) => a + Number(b.toFixed(2)), 0);
  const roundingGap = Math.abs(roundedSum - total) >= .005;
  const mobileRoundNote = roundingGap ? 'exp(s)=e^s→正数；合计按逐项舍入' : 'exp(s)=e^s→正数；分母求和后舍入';
  const products = [0, 1].map(j => d.weights.slice(0, allowed).map((w, i) => `${(w * 100).toFixed(1)}%×${d.values[i][j]}`));
  const sums = d.mixed.map(x => x.toFixed(2));
  const title = txt('', 'teach-heading'), legend = txt('', 'teach-small'), formula = txt('', 'teach-small');
  const formulaQK = txt('Q、K = 输入', 'teach-small'), formulaV = txt('V = 2×输入', 'teach-small'), formulaS = txt('s = Q·K ÷ √2', 'teach-small');
  const q = token('query', '', 142, 'teach-token teach-color-' + (qIndex % 5));
  q.t.textContent = `Q p${qIndex + 1} ${compactVec(d.Q)}`;
  q.g.dataset.query = String(qIndex + 1);
  const headers = ['位置 / K·V', '分数', 'exp(s)', '权重'].map(x => txt(x, 'teach-small'));
  const rows = d.keys.map((k, i) => {
    const future = i > qIndex;
    return {
      future,
      bg: rect(future ? 'teach-mask' : 'teach-row teach-color-' + i),
      pos: txt('', 'teach-small'),
      key: txt('', 'teach-number'),
      value: txt('', 'teach-number'),
      score: txt('', 'teach-number'),
      exp: txt('', 'teach-number'),
      weight: txt('', 'teach-number'),
      track: rect('teach-track'),
      bar: rect('teach-value teach-color-' + i)
    };
  });
  const denomPanel = rect('teach-panel');
  const denomTitle = txt('softmax 分母', 'teach-heading');
  const denomNote = txt('', 'teach-small'), denomExpNote = txt('', 'teach-small');
  const denomA = txt('', 'teach-number'), denomB = txt('', 'teach-number');
  const denomWeight = txt('', 'teach-small'), denomTotal = txt('', 'teach-small'), denomHint = txt('', 'teach-small');
  const denomRound = txt('', 'teach-small'), denomRound2 = txt('', 'teach-small');
  const dimTitle = txt('各维贡献 = 权重 × V 相加', 'teach-heading');
  const dimLegend = txt('', 'teach-small');
  const dims = [0, 1].map(j => ({
    track: rect('teach-track'),
    segs: d.weights.map((w, i) => ({bar: rect('teach-contribution teach-color-' + i)})),
    sum: txt('', 'teach-number'),
    eq1: txt('', 'teach-small'),
    eq2: txt('', 'teach-small')
  }));
  const result = txt('', 'teach-heading');

  return (p, m) => {
    const W = m ? 360 : 720;
    text(title, 16, 24, `观察位置 ${qIndex + 1} · 可读 p1–p${allowed}`); show(title, 1);
    text(legend, m ? 16 : 166, m ? 44 : 64, allowed === 4 ? '本例无未来行：p1–p4 都参与 softmax' : '灰色虚线 = 未来位置，不进入分数与分母'); show(legend, 1);
    move(q, 16, m ? 58 : 36, 1);
    const formulaOn = p >= 1 ? 1 : .4;
    if (m) {
      text(formulaQK, 166, 64); text(formulaV, 166, 82); text(formulaS, 166, 100);
      show(formulaQK, formulaOn); show(formulaV, formulaOn); show(formulaS, formulaOn); show(formula, 0);
    } else {
      text(formula, 166, 48, 'Q、K = 输入；V = 2×输入；s = Q·K ÷ √2');
      show(formula, formulaOn); show(formulaQK, 0); show(formulaV, 0); show(formulaS, 0);
    }
    const headerY = m ? 130 : 82;
    headers.forEach((e, i) => {
      const xs = m ? [16, 150, 214, 278] : [16, 206, 272, 330];
      text(e, xs[i], headerY); show(e, p >= 1 ? 1 : 0);
    });
    const rowStart = m ? 144 : 96, rowH = m ? 56 : 52;
    rows.forEach((r, i) => {
      const y = rowStart + i * rowH, future = r.future;
      set(r.bg, {class: (future ? 'teach-mask' : 'teach-row teach-color-' + i) + (i === qIndex ? ' is-query' : '')});
      box(r.bg, m ? 10 : 12, y - 2, m ? 340 : 500, m ? 52 : 44);
      show(r.bg, future ? .5 : .3 + .4 * Math.min(1, p));
      r.bg.dataset.futureRow = future ? 'true' : 'false';
      r.bg.dataset.position = String(i + 1);
      text(r.pos, 18, y + 14, `p${i + 1}${i === qIndex ? ' · Q 来源' : ''}`); show(r.pos, future ? .52 : 1);
      text(r.key, 18, y + 32, 'K' + compactVec(d.keys[i]));
      text(r.value, m ? 82 : 105, y + 32, 'V' + compactVec(d.values[i]));
      show(r.key, future ? .38 : 1); show(r.value, future ? .38 : 1);
      text(r.score, m ? 150 : 206, m ? y + 23 : y + 28, future ? '遮住' : p >= 2 ? d.scores[i].toFixed(2) : '');
      show(r.score, future ? (p >= 1 ? 1 : 0) : p >= 2 ? phase(p, 1.6, 2) : 0);
      text(r.exp, m ? 214 : 272, m ? y + 23 : y + 28, future ? '—' : p >= 3 ? exps[i].toFixed(2) : '');
      show(r.exp, p >= 3 ? phase(p, 2.7, 3.1) : 0);
      text(r.weight, m ? 278 : 416, m ? y + 14 : y + 28, future ? '0%' : p >= 3 ? pct(d.weights[i]) : '');
      show(r.weight, p >= 2.9 ? phase(p, 2.9, 3.2) : 0);
      const barX = m ? 278 : 330, barW = m ? 62 : 78;
      box(r.track, barX, m ? y + 34 : y + 20, barW, m ? 6 : 9); show(r.track, !future && p >= 2.7 ? 1 : 0);
      box(r.bar, barX, m ? y + 34 : y + 20, barW * d.weights[i] * phase(p, 2.9, 3.4), m ? 6 : 9);
      show(r.bar, !future && p >= 2.9 ? 1 : 0);
    });
    const panelX = m ? 10 : 526, panelY = m ? 372 : 78, panelW = m ? 340 : 178, panelH = m ? 124 : 238;
    box(denomPanel, panelX, panelY, panelW, panelH);
    const denomOn = p >= 3 ? phase(p, 2.8, 3.15) : 0;
    show(denomPanel, denomOn);
    denomPanel.dataset.denominator = 'true';
    text(denomTitle, panelX + 12, panelY + 20); show(denomTitle, denomOn);
    if (m) {
      text(denomNote, panelX + 12, panelY + 40, mobileRoundNote);
      show(denomNote, denomOn); show(denomExpNote, 0);
    } else {
      text(denomNote, panelX + 12, panelY + 40, 'exp(s) = e^s：分数 → 正数');
      text(denomExpNote, panelX + 12, panelY + 60, '分母 = 这些正数之和');
      show(denomNote, denomOn); show(denomExpNote, denomOn);
    }
    text(denomA, panelX + 12, panelY + (m ? 62 : 82), denomFirst); show(denomA, denomOn);
    text(denomB, panelX + 12, panelY + (m ? 81 : 101), denomSecond); show(denomB, denomOn);
    text(denomWeight, panelX + 12, panelY + (m ? 102 : 122), currentWeight); show(denomWeight, denomOn);
    text(denomTotal, panelX + 12, panelY + (m ? 120 : 140), m ? `合计 ${totalWeight}` : `允许权重合计 ≈ ${totalWeight}`); show(denomTotal, denomOn);
    text(denomRound, panelX + 12, panelY + 158, roundingGap ? `逐项两位相加 = ${roundedSum.toFixed(2)}` : '');
    text(denomRound2, panelX + 12, panelY + 172, roundingGap ? '总和由未舍入值求和后取两位' : '');
    show(denomRound, !m && p >= 3 && roundingGap ? 1 : 0); show(denomRound2, !m && p >= 3 && roundingGap ? 1 : 0);
    text(denomHint, panelX + 12, panelY + (!m && roundingGap ? 196 : 170), allowed === 4 ? '4 个位置都参与' : `未来 p${allowed + 1}–p4 不计入`);
    show(denomHint, !m && p >= 3 ? 1 : 0);
    const dimTop = m ? 516 : 332;
    text(dimTitle, 16, dimTop); show(dimTitle, p >= 3 ? 1 : 0);
    text(dimLegend, 16, dimTop + (m ? 22 : 18), `色段顺序 = p1…p${allowed}`); show(dimLegend, p >= 3 ? 1 : 0);
    dims.forEach((dim, j) => {
      const y = dimTop + (m ? 40 : 34) + j * 54, bx = m ? 84 : 90, maxW = m ? 184 : 320, scale = Math.max(1, ...d.mixed);
      box(dim.track, bx, y, maxW, 10); show(dim.track, p >= 3 ? 1 : 0);
      let xx = bx;
      d.weights.slice(0, allowed).forEach((w, i) => {
        const value = w * d.values[i][j], width = value / scale * maxW * phase(p, 3.25, 4);
        const seg = dim.segs[i];
        box(seg.bar, xx, y, width, 10); show(seg.bar, p >= 3 && value > 0 ? 1 : 0);
        xx += width;
      });
      text(dim.sum, bx + maxW + 10, y + 9, p >= 4 ? sums[j] : ''); show(dim.sum, p >= 4 ? 1 : 0);
      const split = m && allowed > 2;
      if (split) {
        text(dim.eq1, 16, y + 24, `维度${j + 1}：${products[j].slice(0, 2).join(' + ')}`);
        text(dim.eq2, 16, y + 44, `+ ${products[j].slice(2).join(' + ')}`);
      } else {
        text(dim.eq1, 16, y + (m ? 24 : 22), `维度${j + 1}：${products[j].join(' + ')}`);
        text(dim.eq2, 16, y + (m ? 44 : 38), '');
      }
      const eqOn = p >= 4 ? phase(p, 3.8, 4) : 0;
      show(dim.eq1, eqOn); show(dim.eq2, split ? eqOn : 0);
      dim.eq1.dataset.equation = String(j + 1);
    });
    text(result, 16, m ? 682 : 468, p >= 4 ? '汇合得到 ' + fmtVec(d.mixed) : '按维度汇合 · 条形为绘制进度');
    show(result, p >= 3 ? 1 : 0);
    result.dataset.complete = p >= 4 ? 'true' : 'false';
  };
});

/* -------------------------------------------------------------- 04 KV cache */
function cacheCompare(svg, viewport) {
  return board(svg, viewport, 'cache', m => m ? 990 : 642, H => {
    const {txt, rect, text, box, show, move, set, node} = H;
    const M = window.FoundationModel;
    const data = M && M.cacheHistory ? M.cacheHistory() : {calls:[4,5,6],saved:[4,5,6],readCounts:[0,4,5],computedNoCache:15,computedCache:6,readTotal:9};
    const leftCounts = M ? M.snapshot('cache', 0, 3).computed : [4, 5, 6];
    const rightCounts = M ? M.snapshot('cache', 1, 3).computed : [4, 1, 1];
    const titles = ['不保存：每轮重算', '保存：旧读新算'];
    const subs = ['没有跨轮可复用 K/V', '只存一份历史；位置随调用增长'];
    const panels = [rect(), rect()], titleNodes = titles.map(x => txt(x, 'teach-heading')), subNodes = subs.map(x => txt(x, 'teach-small'));
    const legend = txt('', 'teach-small'), legend2 = txt('', 'teach-small');
    function chip(id, label, color, w = 38, h = 26) {
      const g = node('g', {'data-motion-id': id, class: 'teach-token teach-color-' + color});
      const r = node('rect', {width: w, height: h, rx: 5});
      const t = node('text', {x: w / 2, y: h - 8, 'text-anchor': 'middle'});
      t.textContent = label; g.append(r, t); return {g, r, t, width: w};
    }
    const sides = [0, 1].map(side => ({
      counts: side ? rightCounts : leftCounts,
      rounds: [0, 1, 2].map(r => {
        const status = txt('', 'teach-small');
        status.setAttribute('text-anchor', 'end');
        return {
          label: txt('第 ' + (r + 1) + ' 次调用', 'teach-small'),
          status,
          cells: Array.from({length: r + 4}, (_, i) => ({
            node: chip(`cmp-${side}-${r}-${i}`, i < 4 ? 'p' + (i + 1) : 'y' + (i - 3), i % 5),
            fill: rect('teach-compute')
          }))
        };
      })
    }));
    const totals = [txt('', 'teach-heading'), txt('', 'teach-heading')], reads = [txt('', 'teach-small'), txt('', 'teach-small')];
    const summary = txt('', 'teach-heading'), summaryNote = txt('', 'teach-small'), summaryNote2 = txt('', 'teach-small'), summaryNote3 = txt('', 'teach-small');
    const historyTitle = txt('', 'teach-small');
    const historyCount = txt('', 'teach-small');
    const historySlots = ['p1', 'p2', 'p3', 'p4', 'y1', 'y2'].map((label, i) => chip('cmp-store-' + i, label, i % 5, 36, 25));
    const barsTitle = txt('', 'teach-small');
    const newLabel = txt('新算合计', 'teach-small'), readLabel = txt('跨轮读取', 'teach-small');
    const noHead = txt('不保存', 'teach-small'), saveHead = txt('保存', 'teach-small');
    const newTracks = [rect('teach-track'), rect('teach-track')], newFills = [rect('teach-compute'), rect('teach-compute')];
    const readTracks = [rect('teach-track'), rect('teach-track')], readFills = [rect('teach-read'), rect('teach-read')];
    const newValues = [txt('', 'teach-number'), txt('', 'teach-number')], readValues = [txt('', 'teach-number'), txt('', 'teach-number')];
    return (p, m) => {
      const panelH = m ? 314 : 328;
      const panelY = m ? [56, 382] : [42, 42];
      const panelX = m ? [12, 12] : [12, 364], panelW = m ? 336 : 344;
      const rowStart = m ? 54 : 72, rowH = m ? 74 : 72, chipW = m ? 36 : 38, chipGap = m ? 45 : 44;
      const historyY = m ? 724 : 400, summaryY = m ? 793 : 469, barsY = m ? 889 : 543;
      const barX = m ? [96, 268] : [110, 474], barW = m ? 64 : 190;
      panels.forEach((v, side) => box(v, panelX[side], panelY[side], panelW, panelH));
      text(titleNodes[0], panelX[0] + 12, panelY[0] + 24);
      text(titleNodes[1], panelX[1] + 12, panelY[1] + 24);
      text(subNodes[0], panelX[0] + 12, panelY[0] + 44); show(subNodes[0], m ? 0 : 1);
      text(subNodes[1], panelX[1] + 12, panelY[1] + 44); show(subNodes[1], m ? 0 : 1);
      text(legend, 16, 24, m ? '橙/绿 = 进度条：新算/跨轮读取' : '进度条：橙 = 本轮新算 · 绿 = 跨轮缓存读取 · 灰 = 尚未发生；色块 = 位置身份');
      text(legend2, 16, 44, m ? '灰 = 未发生；色块 = 位置身份' : ''); show(legend2, m ? 1 : 0);
      [0, 1].forEach(side => {
        const x = panelX[side], y = panelY[side];
        sides[side].rounds.forEach((row, r) => {
          const active = p >= r, done = p >= r + 1, labelY = y + rowStart + r * rowH;
          const rowStatus = !active ? '待算' : !done ? '进行中' : side === 0 || r === 0 ? `重算 ${row.cells.length}` : `读取 ${data.readCounts[r]} · 新算 1`;
          text(row.label, x + 12, labelY); show(row.label, active ? 1 : .35);
          text(row.status, x + panelW - 12, labelY, rowStatus); show(row.status, active ? 1 : .35);
          row.cells.forEach((cell, i) => {
            const cx = x + 12 + i * chipGap, cy = labelY + 12;
            const reuse = side === 1 && ((r === 1 && i < 4) || (r === 2 && i < 5));
            const planned = side === 0 ? '重算' : reuse ? '复用' : '新算';
            move(cell.node, cx, cy, active ? (done ? 1 : .82) : .2);
            set(cell.node.r, {width: chipW, height: 26});
            text(cell.node.t, chipW / 2, 18);
            cell.node.r.dataset.state = planned;
            box(cell.fill, cx + 2, cy + 30, (chipW - 4) * (done ? 1 : active ? .45 : 0), 3);
            set(cell.fill, {class: side === 1 && reuse ? 'teach-read' : 'teach-compute'});
            show(cell.fill, active ? 1 : 0);
          });
        });
        const doneTerms = sides[side].counts.filter((_, r) => p >= r + 1);
        const computed = doneTerms.reduce((a, b) => a + b, 0);
        text(reads[side], x + 12, y + panelH - 46, side === 0 ? '跨轮缓存读取 0' : p >= 3 ? `跨轮缓存读取 ${data.readCounts[1]} + ${data.readCounts[2]} = ${data.readTotal}` : p >= 2 ? `跨轮缓存读取 ${data.readCounts[1]}` : '跨轮缓存读取 0');
        text(totals[side], x + 12, y + panelH - 16, `新算 ${doneTerms.join(' + ') || '0'}${doneTerms.length ? ' = ' + computed : ''}`);
        totals[side].dataset.computed = String(computed);
      });
      const saved = p >= 3 ? 6 : p >= 2 ? 5 : p >= 1 ? 4 : 0;
      text(historyTitle, 16, historyY, m ? '保存历史位置' : '保存历史的位置（右侧）');
      text(historyCount, m ? 132 : 220, historyY, m ? `0→4→5→6 · 当前 ${saved}/6` : `0 → 4 → 5 → 6：当前 ${saved} 个位置`);
      historySlots.forEach((a, i) => {
        move(a, 16 + i * 46, historyY + 12, i < saved ? 1 : .22);
        set(a.g, {class: 'teach-token ' + (i < saved ? 'teach-color-' + (i % 5) : 'teach-empty')});
        a.g.dataset.saved = String(i < saved);
      });
      text(summary, 16, summaryY, m ? '三次调用完成后的对照' : `三次调用完成后的对照：新算 ${data.computedNoCache} vs ${data.computedCache}（不保存 vs 保存）· 跨轮缓存读取 0 vs ${data.readTotal}`);
      text(summaryNote, 16, summaryY + 24, m ? `新算 ${data.computedNoCache}/${data.computedCache} · 跨轮缓存读取 0/${data.readTotal}` : `${data.computedNoCache}/${data.computedCache} 是 K/V 投影位置数，不是总工作量、耗时或加速比。`);
      text(summaryNote2, 16, summaryY + 46, m ? `${data.computedNoCache}/${data.computedCache} 是位置数，非耗时或加速比` : `跨轮缓存读取 0 = 未复用上一轮缓存；Attention 仍读取本轮所需 K/V。`);
      text(summaryNote3, 16, summaryY + 68, m ? '跨轮读取 0 = 未复用；Attention 仍读 K/V' : ''); show(summaryNote3, m ? 1 : 0);
      text(barsTitle, 16, barsY, m ? '资源条 · 三次调用完成后的累计' : '资源条 · 三次调用完成后的累计（不是时间）');
      text(noHead, 18, barsY + 24); text(saveHead, m ? 190 : 382, barsY + 24);
      text(newLabel, 16, barsY + 52); text(readLabel, 16, barsY + 84);
      [0, 1].forEach(side => {
        const value = side ? data.computedCache : data.computedNoCache;
        box(newTracks[side], barX[side], barsY + 44, barW, 9);
        box(newFills[side], barX[side], barsY + 44, barW * value / data.computedNoCache, 9);
        text(newValues[side], barX[side] + barW + 4, barsY + 54, String(value));
        const read = side ? data.readTotal : 0;
        box(readTracks[side], barX[side], barsY + 76, barW, 9);
        box(readFills[side], barX[side], barsY + 76, barW * read / data.readTotal, 9);
        text(readValues[side], barX[side] + barW + 4, barsY + 86, String(read));
      });
      show(summary, 1); show(summaryNote, 1); show(barsTitle, 1); show(newLabel, 1); show(readLabel, 1);
    };
  });
}
function cachePages(svg, viewport) {
  const panel = window.CacheAdoption?.mount('pages', viewport) || null;
  const inner = board(svg, viewport, 'cache-pages', () => 430, H => {
    const {txt, rect, path, text, box, show, wire, token, move, set} = H;
    const title = txt('逻辑位置 → 页表 → 物理页', 'teach-heading'), logicTitle = txt('R1 的逻辑位置', 'teach-small'), tableTitle = txt('页表：位置 → 页 / 槽', 'teach-small');
    const poolTitle = txt('物理页池（每页 2 槽）', 'teach-small'), note = txt('页号 7、2 是教学物理页号；p1…p3 是请求位置。逻辑顺序连续，物理页可以不连续。', 'teach-small'), note2 = txt('逻辑顺序连续，物理页可以不连续。', 'teach-small');
    const page7 = rect('teach-panel'), page2 = rect('teach-panel'), p7Label = txt('页 7 · 2 槽', 'teach-heading'), p2Label = txt('页 2 · 2 槽', 'teach-heading');
    const positions = [1, 2, 3].map(i => token('page-pos-' + i, 'p' + i, 44, 'teach-token teach-color-0'));
    const posNotes = [1, 2, 3].map(i => txt('待处理', 'teach-small'));
    const rows = [0, 1, 2].map(i => txt('', 'teach-number'));
    const slots = [0, 1, 2, 3].map(i => ({box: rect('teach-memory-slot'), label: txt('空闲', 'teach-small'), sub: txt('', 'teach-small')}));
    const deskPaths = [0, 1, 2, 3, 4, 5].map(() => path('teach-link'));
    const mobilePaths = [0, 1, 2].map(() => path('teach-link'));
    const mapLine = txt('', 'teach-small'), pageState = txt('', 'teach-small');
    return (p, m) => {
      const W = m ? 360 : 720;
      text(title, 16, 24);
      if (!m) {
        text(logicTitle, 16, 56); text(tableTitle, 180, 56); text(poolTitle, 470, 56); show(tableTitle, 1); show(poolTitle, 1);
        const p7x = 470, p7y = 72, p2x = 470, p2y = 214;
        box(page7, p7x, p7y, 220, 130); box(page2, p2x, p2y, 220, 130);
        text(p7Label, p7x + 12, p7y + 22); text(p2Label, p2x + 12, p2y + 22);
        positions.forEach((a, i) => move(a, 24, 76 + i * 60, 1));
        posNotes.forEach((t, i) => { text(t, 26, 116 + i * 60, p >= (i < 2 ? 2 : 3) ? '已写入' : p >= (i < 2 ? 1 : 3) ? '已分配' : '待处理'); });
        rows.forEach((r, i) => {
          const y = 76 + i * 40, page = i < 2 ? 7 : 2, slot = i === 2 ? 0 : i;
          const ready = p >= (i < 2 ? 2 : 3), reserved = p >= (i < 2 ? 1 : 2.7);
          text(r, 180, y, ready ? `p${i + 1} → 页 ${page} / 槽 ${slot} · 已写入` : reserved ? `p${i + 1} → 页 ${page} / 槽 ${slot} · 预留` : `p${i + 1} → 尚未分配`);
        });
        const slotPos = [[p7x + 16, p7y + 48], [p7x + 112, p7y + 48], [p2x + 16, p2y + 48], [p2x + 112, p2y + 48]];
        slots.forEach((s, i) => {
          const [x, y] = slotPos[i], active = i < 3, w = i < 2 ? phase(p, 1.7, 2) : i === 2 ? phase(p, 2.7, 3) : 0, written = active && w >= .99, reserved = active && (i < 2 ? p >= 1 : p >= 2.7);
          box(s.box, x, y, 78, 26); set(s.box, {class: 'teach-memory-slot' + (reserved && !written ? ' is-reserved' : written ? ' is-written' : '')});
          text(s.label, x + 6, y + 17, written ? 'p' + (i + 1) : reserved ? '预留' : '空闲'); text(s.sub, x + 6, y + 43, '槽 ' + (i % 2)); show(s.sub, 1);
        });
        deskPaths.forEach((pathNode, i) => {
          if (i < 3) {
            wire(pathNode, `M136 ${96 + i * 60} H180 V${88 + i * 40}`, i < 2 ? phase(p, 1.7, 2) : phase(p, 2.7, 3));
            pathNode.dataset.map = 'pos-' + (i + 1);
          } else {
            const j = i - 3, [x, y] = slotPos[j], ty = 88 + j * 40;
            wire(pathNode, `M430 ${ty} H450 V${y + 9} H${x - 4}`, j < 2 ? phase(p, 1.7, 2) : phase(p, 2.7, 3));
            pathNode.dataset.map = 'slot-' + (j + 1);
          }
        });
        text(note, 16, 366); show(note2, 0); text(pageState, 16, 390, p >= 3 ? '映射完成：p1、p2 在页 7；p3 在页 2。槽位未写入前仍不可用。' : p >= 1 ? '页 7 已被预留；分配地址不等于已经写入 K/V。' : '空池：还没有为请求分配页。');
      } else {
        text(logicTitle, 16, 44); text(tableTitle, 16, 88); text(poolTitle, 16, 168); show(tableTitle, 0); show(poolTitle, 0);
        positions.forEach((a, i) => move(a, 16 + i * 108, 54, 1));
        [0, 1, 2].forEach(i => {
          const written = p >= (i < 2 ? 2 : 3), reserved = p >= (i < 2 ? 1 : 3);
          text(posNotes[i], 16 + i * 108, 98, written ? '已写入' : reserved ? '已分配' : '待处理'); show(posNotes[i], 1);
          wire(mobilePaths[i], `M${38 + i * 108} 84 V${i < 2 ? 102 : 194}`, 0);
          mobilePaths[i].dataset.map = 'pos-' + (i + 1);
        });
        const p7y = 104, p2y = 196;
        box(page7, 16, p7y, 328, 84); box(page2, 16, p2y, 328, 84);
        text(p7Label, 28, p7y + 22); text(p2Label, 28, p2y + 22);
        const slotPos = [[84, p7y + 40], [184, p7y + 40], [84, p2y + 40], [184, p2y + 40]];
        slots.forEach((s, i) => {
          const [x, y] = slotPos[i], active = i < 3, w = i < 2 ? phase(p, 1.7, 2) : i === 2 ? phase(p, 2.7, 3) : 0, written = active && w >= .99, reserved = active && (i < 2 ? p >= 1 : p >= 2.7);
          box(s.box, x, y, 78, 40); set(s.box, {class: 'teach-memory-slot' + (reserved && !written ? ' is-reserved' : written ? ' is-written' : '')});
          text(s.label, x + 6, y + 16, written ? 'p' + (i + 1) : reserved ? '预留' : '空闲'); text(s.sub, x + 6, y + 32, '槽 ' + (i % 2)); show(s.sub, 1);
        });
        text(mapLine, 16, 296, 'p1、p2 → 页 7；p3 → 页 2。'); show(mapLine, p >= 2 ? 1 : 0);
        text(pageState, 16, 318, p >= 3 ? '映射完成：逻辑位置连续，物理页可以跳号。' : p >= 1 ? '页 7 已预留；地址还不等于可读 K/V。' : '空池：先分配，再写入。');
        text(note, 16, 340, '页号 7、2 是教学物理页号；p1…p3 是请求位置。'); note.setAttribute('font-size', '11');
        text(note2, 16, 356, '逻辑顺序连续，物理页可以不连续。'); note2.setAttribute('font-size', '11');
      }
    };
  });
  return {paint(...args){inner.paint(...args);}, destroy(){panel?.destroy();inner.destroy();}};
}
function cacheShared(svg, viewport) {
  const panel = window.CacheAdoption?.mount('prefix', viewport) || null;
  const inner = board(svg, viewport, 'cache-shared', () => 430, H => {
    const {txt, rect, path, text, box, show, wire, token, move, set} = H;
    const title = txt('两条请求引用同一份物理 K/V', 'teach-heading'), r1 = txt('R1', 'teach-heading'), r2 = txt('R2', 'teach-heading');
    const r1Box = rect('teach-panel'), r2Box = rect('teach-panel'), sharedBox = rect('teach-panel');
    const r1State = txt('', 'teach-small'), r2State = txt('', 'teach-small');
    const sharedLabel = txt('', 'teach-heading'), sharedNote = txt('', 'teach-small');
    const holder = txt('', 'teach-heading'), note1 = txt('', 'teach-small'), note2 = txt('', 'teach-small'), note2b = txt('', 'teach-small');
    const retain = txt('', 'teach-heading');
    const kv = [1, 2, 3].map(i => token('shared-kv-' + i, 'p' + i, 42, 'teach-token teach-color-0'));
    const link1 = path('teach-link'), link2 = path('teach-link');
    const ref1 = txt('引用 1', 'teach-small'), ref2 = txt('引用 2', 'teach-small');
    svg.append(r1, r2);
    return (p, m) => {
      // Holdership and the drawn arrows share one lifecycle. A fractional
      // position stays on the previous completed state until the next step's
      // event happens, so a pause mid-transition never releases R1 early.
      const r1Held = p >= 1 && p < 3, r2Held = p >= 2 && p < 4, refs = (r1Held ? 1 : 0) + (r2Held ? 1 : 0);
      const established = p >= 1;
      text(title, 16, 24);
      box(r1Box, 16, m ? 44 : 50, m ? 160 : 210, m ? 76 : 88); box(r2Box, m ? 184 : 270, m ? 44 : 50, m ? 160 : 210, m ? 76 : 88);
      text(r1, 28, m ? 66 : 72); text(r2, m ? 196 : 282, m ? 66 : 72);
      text(r1State, 28, m ? 92 : 104, r1Held ? '持有共享前缀' : p >= 3 ? 'R1 已结束' : '尚未建立');
      text(r2State, m ? 196 : 282, m ? 92 : 104, r2Held ? '持有共享前缀' : p >= 4 ? 'R2 已结束' : '尚未建立');
      box(sharedBox, 16, m ? 150 : 180, m ? 328 : 454, m ? 96 : 106);
      text(sharedLabel, 28, m ? 174 : 204, !established ? '尚未建立共享历史' : p >= 4 ? '共享历史：仍被缓存保留' : '共享历史：同一份 K/V');
      text(sharedNote, 28, m ? 194 : 226, !established ? '先有可复用前缀，才谈引用。' : refs === 2 ? 'R1、R2 指向同一份数据；没有复制第二份。' : r1Held ? 'R1 指向这份数据；R2 尚未接入。' : r2Held ? 'R1 已结束；R2 仍指向同一份数据。' : m ? '无请求持有；缓存仍保留。' : '没有请求持有；缓存仍保留同一份数据，等待驱逐。');
      kv.forEach((a, i) => move(a, 28 + i * (m ? 78 : 92), m ? 204 : 246, p >= 1 ? 1 : 0));
      wire(link1, m ? 'M96 120 V150' : 'M120 138 V180', r1Held ? 1 : 0);
      wire(link2, m ? 'M264 120 V150' : 'M370 138 V180', r2Held ? 1 : 0);
      link1.dataset.ref = 'r1'; link2.dataset.ref = 'r2';
      text(ref1, m ? 64 : 134, m ? 138 : 164); show(ref1, r1Held ? 1 : 0);
      text(ref2, m ? 232 : 384, m ? 138 : 164); show(ref2, r2Held ? 1 : 0);
      text(holder, m ? 16 : 490, m ? 270 : 206, `持有者：${refs}`); holder.dataset.holder = String(refs); show(holder, established ? 1 : 0);
      text(note1, 16, m ? 298 : 318, m ? '共享是引用，不是复制；' : '共享是引用，不是复制；R1 结束后 R2 仍持有，不能覆盖。');
      text(note2, 16, m ? 318 : 338, m ? 'R1 结束，R2 仍持有，不可覆盖。' : '两条请求都结束后，缓存仍可按策略保留；');
      text(note2b, 16, m ? 336 : 354, m ? '无请求时缓存保留；驱逐另判。' : '驱逐是另一个条件。');
      show(note1, 1); show(note2, 1); show(note2b, 1);
      // Keep the dynamic holder line clear of the three fixed notes above it.
      text(retain, 16, m ? 360 : 386, p >= 4 ? '引用 0：没有活跃请求，缓存仍被保留' : p >= 3 ? 'R1 引用消失，R2 仍持有 → 不可覆盖' : p >= 2 ? '两个引用指向同一份物理数据' : p >= 1 ? 'R1 持有共享前缀；R2 尚未接入' : '');
      retain.dataset.retain = p >= 4 ? 'true' : 'false'; show(retain, p >= 1 ? 1 : 0);
    };
  });
  return {paint(...args){inner.paint(...args);}, destroy(){panel?.destroy();inner.destroy();}};
}
V['f-cache'] = (svg, viewport, s) => s.kind === 'cache-pages' ? cachePages(svg, viewport) : s.kind === 'cache-shared' ? cacheShared(svg, viewport) : cacheCompare(svg, viewport);

})();
