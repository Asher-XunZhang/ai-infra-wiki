/* Isolated companion figure for generation.html: one forward pass, three layers,
 * one KV store per layer, then one Decode step. Objects are created once; the
 * drawing is a pure function of the seekable position p (0…9). */
(() => {
'use strict';
const root = document.querySelector('[data-layer-journey]');
if (!root) return;
const NS = 'http://www.w3.org/2000/svg';
const clamp = (n, a, b) => Math.max(a, Math.min(b, n));
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = t => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
const phase = (p, a, b) => smooth((p - a) / (b - a));
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const STEPS = [
  ['输入位置就位', 'p1…p4 是四个输入位置，不是分词片段，也不是四条请求；三层是教学假设。此时还没有任何 KV。'],
  ['Prefill · 第 1 层', '四个位置一起进入层 1。层 1 写入自己的 p1…p4 K/V，隐藏状态交给层 2。'],
  ['Prefill · 第 2 层', '层 2 写入另一份 p1…p4 K/V，与层 1 的那份独立。仍是同一次前向，不是第二轮请求。'],
  ['Prefill · 第 3 层', '层 3 写入第三份 K/V。最后一个位置 p4 的最终隐藏状态送往输出头。'],
  ['采样 y1', '输出头只取 p4 的最终隐藏状态，得到 logits 并采样 y1。y1 刚选出，还没有作为输入，所以三层都没有 y1 的 KV。'],
  ['y1 成为下一轮输入', '把同一个 y1 送回输入区，成为下一轮唯一的新位置。此时仍没有 y1 的 KV，前向尚未开始。'],
  ['Decode · 第 1 层', '只计算 y1，读取本层 p1…p4 的历史 K/V，并写入 y1 的层 1 K/V；p1…p4 不重算。'],
  ['Decode · 第 2 层', 'y1 的隐藏状态继续经过层 2，写入 y1 的层 2 K/V。只处理新位置，不等于跳过层。'],
  ['Decode · 第 3 层', '层 3 写入 y1 的 K/V。到这里 y1 才遍历完完整模型，输出头将用它采样 y2。'],
  ['采样 y2', 'y2 被采样出来；输出序列保留 y1 和 y2。y2 尚无 KV，要等它作为下一轮输入再逐层写入。']
];
const LAYERS = [1, 2, 3];
const positions = ['p1', 'p2', 'p3', 'p4'];
const esc = x => String(x);
let W = 0, H = 0, narrow = true, mounted = false, p = 0, playing = false, raf = 0, last = 0;
let speed = 1, geom = null, el = null, cells = [], preChips = [], svg = null;

function node(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  parent && parent.append(e);
  return e;
}
function txt(cls, value, parent) {
  const e = node('text', { class: cls }, parent);
  if (value !== undefined) e.textContent = value;
  return e;
}
function chip(parent, label, cls) {
  const g = node('g', { class: 'jl-kv-cell ' + cls }, parent);
  const base = node('rect', { width: 0, height: 26, rx: 4 }, g);
  const fill = node('rect', { width: 0, height: 26, rx: 4, class: 'jl-cell-fill' }, g);
  const t = node('text', { y: 17, 'text-anchor': 'middle', class: 'jl-cell-label' }, g);
  t.textContent = label;
  return { g, base, fill, t, label };
}

function build() {
  W = Math.max(220, Math.min(820, Math.round(root.clientWidth || 320)));
  narrow = W < 520;
  const pad = 8, railW = narrow ? 66 : 180, gap = narrow ? 8 : 18, packetW = railW - 8;
  const rightX = pad + railW + gap, rightW = W - rightX - pad;
  const inputY = 10, inputH = narrow ? 72 : 78;
  const layerTop = inputY + inputH + 12;
  const rowH = narrow ? 92 : 100, rowGap = narrow ? 10 : 14;
  const layerY = [0, 1, 2].map(i => layerTop + i * (rowH + rowGap));
  const outHeadY = layerTop + 3 * (rowH + rowGap), outHeadH = 70;
  const outRowY = outHeadY + outHeadH + 12, outRowH = 72;
  H = outRowY + outRowH + 12;
  const center = inputY + inputH / 2;
  const stations = [center, ...layerY.map(y => y + rowH / 2), outHeadY + outHeadH / 2];
  geom = { W, H, pad, railW, packetW, gap, rightX, rightW, inputY, inputH, layerTop, rowH, rowGap, layerY, outHeadY, outHeadH, outRowY, outRowH, stations, narrow, outCenter: outRowY + 36 };
  root.style.setProperty('--jl-w', W + 'px');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('width', W); svg.setAttribute('height', H); svg.style.width = W + 'px';
  svg.replaceChildren();
  const railX = pad + railW / 2;
  node('line', { x1: railX, y1: stations[0], x2: railX, y2: stations[4], class: 'jl-rail' }, svg);
  stations.forEach((y, i) => node('circle', { cx: railX, cy: y, r: 3, class: 'jl-rail-dot' }, svg));

  // moving forward packet: four prompt positions, then one y1 position
  const pre = node('g', { class: 'jl-packet', 'data-packet': 'prefill' }, svg);
  node('rect', { x: 0, y: 0, width: packetW, height: 62, rx: 8, class: 'jl-packet-bg' }, pre);
  const preTitle = txt('jl-packet-title', '输入 ×4', pre);
  preTitle.setAttribute('x', 6); preTitle.setAttribute('y', 15);
  preChips = positions.map((label, i) => {
    let x, y, w, h, fs;
    if (narrow) { x = 5 + (i % 2) * 28; y = 22 + Math.floor(i / 2) * 19; w = 25; h = 17; fs = 9.5; }
    else { x = 8 + i * 40; y = 24; w = 36; h = 27; fs = 12; }
    const g = node('g', { class: 'jl-packet-chip' }, pre);
    node('rect', { x, y, width: w, height: h, rx: 4 }, g);
    const t = node('text', { x: x + w / 2, y: y + h / 2 + 4, 'text-anchor': 'middle', style: `font-size:${fs}px` }, g);
    t.textContent = label;
    return { g, t };
  });
  const dec = node('g', { class: 'jl-packet', 'data-packet': 'decode' }, svg);
  node('rect', { x: 0, y: 0, width: packetW, height: 54, rx: 8, class: 'jl-packet-bg decode' }, dec);
  const decTitle = txt('jl-packet-title', '同一 y1', dec);
  decTitle.setAttribute('x', 6); decTitle.setAttribute('y', 15);
  const decChip = node('g', { class: 'jl-packet-chip y' }, dec);
  node('rect', { x: (packetW - 44) / 2, y: 22, width: 44, height: 24, rx: 4 }, decChip);
  const decText = node('text', { x: packetW / 2, y: 39, 'text-anchor': 'middle', class: 'jl-token' }, decChip);
  decText.textContent = 'y1';

  // three layer rows, each with its own KV shelf
  cells = LAYERS.map((layer, l) => {
    const y = layerY[l];
    const g = node('g', { class: 'jl-layer l' + l }, svg);
    node('rect', { x: rightX, y, width: rightW, height: rowH - 4, rx: 8, class: 'jl-band' }, g);
    const title = txt('jl-layer-title', '层 ' + layer, g);
    title.setAttribute('x', rightX + 10); title.setAttribute('y', y + 18);
    const sub = txt('jl-layer-sub', 'Attention → MLP', g);
    sub.setAttribute('x', rightX + rightW - 10); sub.setAttribute('y', y + 18); sub.setAttribute('text-anchor', 'end');
    node('rect', { x: rightX + 4, y: y + 26, width: rightW - 8, height: rowH - 34, rx: 6, class: 'jl-shelf l' + l }, g);
    const st = txt('jl-shelf-title l' + l, '层 ' + layer + ' K/V · 独立', g);
    st.setAttribute('x', rightX + 10); st.setAttribute('y', y + 42);
    const status = txt('jl-shelf-status', '空', g);
    status.setAttribute('x', rightX + rightW - 10); status.setAttribute('y', y + 42); status.setAttribute('text-anchor', 'end');
    const cw = Math.min(64, (rightW - 16 - 12) / 5), total = cw * 5 + 12;
    const cx = rightX + (rightW - total) / 2, cy = y + 46;
    const list = [...positions, 'y1'].map((label, i) => {
      const c = chip(g, label, i < 4 ? 'p' : 'y');
      c.g.setAttribute('transform', `translate(${cx + i * (cw + 3)} ${cy})`);
      c.base.setAttribute('width', cw); c.fill.setAttribute('width', cw);
      c.g.setAttribute('data-layer', String(layer)); c.g.setAttribute('data-pos', label);
      c.g.dataset.state = 'empty';
      c.t.setAttribute('x', cw / 2);
      return c;
    });
    return { g, band: g.firstChild, title, sub, status, list, layer, y, cw };
  });

  // output head and retained output sequence
  const oh = node('g', { class: 'jl-outhead' }, svg);
  node('rect', { x: rightX, y: outHeadY, width: rightW, height: outHeadH, rx: 8, class: 'jl-head-box' }, oh);
  const ohTitle = txt('jl-head-title', '输出头 → logits → 采样', oh);
  ohTitle.setAttribute('x', rightX + 10); ohTitle.setAttribute('y', outHeadY + 21);
  const ohSub = txt('jl-head-sub', '等待 p4 的最终隐藏状态', oh);
  ohSub.setAttribute('x', rightX + 10); ohSub.setAttribute('y', outHeadY + 42);
  const ohSub2 = txt('jl-head-sub', '', oh);
  ohSub2.setAttribute('x', rightX + 10); ohSub2.setAttribute('y', outHeadY + 58);

  const or = node('g', { class: 'jl-output' }, svg);
  node('rect', { x: pad, y: outRowY, width: W - 2 * pad, height: outRowH, rx: 8, class: 'jl-out-row' }, or);
  const orTitle = txt('jl-output-title', '输出序列 · 已选出的 token 保留', or);
  orTitle.setAttribute('x', pad + 10); orTitle.setAttribute('y', outRowY + 17);
  const colW = (W - 2 * pad - 12) / 2;
  const slots = [0, 1].map(i => {
    const x = pad + 8 + i * colW;
    node('rect', { x, y: outRowY + 26, width: 38, height: 28, rx: 4, class: 'jl-slot' }, or);
    const g = node('g', { class: 'jl-out-token' }, or);
    const r = node('rect', { x, y: outRowY + 26, width: 38, height: 28, rx: 4 }, g);
    const t = node('text', { x: x + 19, y: outRowY + 45, 'text-anchor': 'middle', class: 'jl-token' }, g);
    t.textContent = i ? 'y2' : 'y1';
    const s = txt('jl-out-status', '', or);
    s.setAttribute('x', x + 45); s.setAttribute('y', outRowY + 45);
    return { g, r, t, s, x };
  });
  node('path', { d: `M${rightX + rightW / 2} ${outHeadY + outHeadH} V${outRowY}`, class: 'jl-drop' }, or);

  // Return lane: the y1 copy never enters the output row. It appears in the
  // rail gutter just above that row and climbs to the input station. A thin
  // dashed route leaves the original y1 slot around the row's left padding so
  // the causal link stays visible without covering the heading or the token.
  const packetX = railX - packetW / 2;
  const linkY = outRowY - 4, laneX = 4, slotLeft = slots[0].x;
  const link = node('path', {
    class: 'jl-return-link',
    d: `M${slotLeft - 2} ${outRowY + 40} H${laneX + 3} Q${laneX} ${outRowY + 40} ${laneX} ${outRowY + 37} ` +
      `V${linkY + 3} Q${laneX} ${linkY} ${laneX + 3} ${linkY} H${packetX}`
  }, svg);
  link.dataset.returnLink = 'y1';
  svg.insertBefore(or, dec);
  svg.insertBefore(link, dec);

  el = { pre, preTitle, dec, decTitle, decChip, oh, ohTitle, ohSub, ohSub2, slots, orTitle, link, linkY };
  root.dataset.jlWidth = String(W);
  render();
}

function setText(e, v) { if (e.textContent !== v) e.textContent = v; }
function setState(c, state) {
  if (c.g.dataset.state !== state) c.g.dataset.state = state;
  c.g.classList.toggle('is-read', state === 'read');
}
function gs(p, l) { return phase(p, 5 + l + 0.4, 5 + l + 0.9); }

function render() {
  if (!el || !geom) return;
  const q = reduced.matches ? Math.round(p) : p;
  const i = Math.round(q);
  const prefill = q < 4.45;
  const railX = geom.pad + geom.railW / 2;
  // Packet rects are authored at x=0 with their full width to the right.
  // Shift the translate left by half the width so the card straddles the
  // rail; using railX alone pushed its right half over layer 1.
  const packetX = railX - geom.packetW / 2;
  const pbgH = 62, dbgH = 54;
  const u = (a, b) => phase(q, a, b);

  // forward packet position
  let py;
  if (q < 4) {
    const a = Math.min(3, Math.floor(q)), t = u(a, a + 1);
    py = lerp(geom.stations[a], geom.stations[a + 1], t);
  } else py = geom.stations[4];
  // Clear the output-head station before the y1 copy appears there: the two
  // packets occupy the same rail gutter, so a crossfade would overlap them.
  const preOp = q < 4.3 ? 1 : 1 - u(4.3, 4.5);
  el.pre.setAttribute('transform', `translate(${packetX} ${py - pbgH / 2})`);
  el.pre.style.opacity = preOp.toFixed(3);
  el.pre.style.display = preOp <= 0.01 ? 'none' : '';
  setText(el.preTitle, q < 1 ? '输入 ×4' : q < 2 ? 'h¹ ×4' : q < 3 ? 'h² ×4' : 'h³ ×4');
  preChips.forEach((c, j) => { c.g.style.opacity = q >= 3.55 && j < 3 ? '.35' : '1'; });

  // decode packet: the y1 copy lifts out of the output row *above* its top
  // edge, never across its heading or original token, then rises to the input
  // station and down all three layers. The card's left edge stays at packetX,
  // left of every layer band and of the output-head text.
  let dx = packetX, dy = geom.stations[0], dop = 0;
  const copyLift = geom.outRowY - 4 - dbgH / 2;
  if (q >= 4.5) {
    dop = u(4.5, 4.55);
    if (q < 5) {
      // Start resting on the output row's top edge in the rail gutter, then
      // climb. No horizontal slide means the card never crosses the row's
      // title or its original y1 slot at any intermediate frame.
      dy = q < 4.55 ? copyLift : lerp(copyLift, geom.stations[0], u(4.55, 5));
    } else {
      const a = Math.min(3, Math.floor(q - 5)), t = u(5 + a, 6 + a);
      dy = lerp(geom.stations[a], geom.stations[a + 1], t);
    }
  }
  el.dec.setAttribute('transform', `translate(${dx} ${dy - dbgH / 2})`);
  el.dec.style.opacity = dop.toFixed(3);
  el.dec.style.display = dop ? '' : 'none';
  setText(el.decTitle, q < 5.5 ? '同一 y1' : q < 6.5 ? 'h¹(y1)' : q < 7.5 ? 'h²(y1)' : 'h³(y1)');

  // Dashed copy route from the original y1 slot, around the row's left
  // padding to the card's current takeoff corner above the row. Tracking the
  // card keeps the causal link while it climbs; the route fades once the card
  // has clearly left the output area.
  const linkOp = q < 4.45 ? 0 : Math.min(u(4.45, 4.52), 1 - u(4.62, 4.8));
  el.link.style.opacity = Math.max(0, linkOp).toFixed(3);
  el.link.style.display = linkOp <= 0.01 ? 'none' : '';
  const cardBottom = (q >= 4.5 ? dy : copyLift) + dbgH / 2, slotMidY = geom.outRowY + 40, laneX = 4;
  el.link.setAttribute('d',
    `M${el.slots[0].x - 2} ${slotMidY} H${laneX + 3} Q${laneX} ${slotMidY} ${laneX} ${slotMidY - 3} ` +
    `V${cardBottom + 3} Q${laneX} ${cardBottom} ${laneX + 3} ${cardBottom} H${packetX}`);

  // per-layer KV shelves
  cells.forEach((L, l) => {
    let state = 'empty', status = '空';
    const pf = u(l + 0.35, l + 0.95);
    if (pf > 0) { state = 'written'; status = pf < .98 ? '写 p1…p4' : '4 个位置'; }
    if (q >= l + 1) status = '4 个位置';
    // A layer starts reading p1…p4 and writing y1 only when the packet has
    // left the previous layer and this layer's step is current. At p=6 that
    // means layer 1 is complete but layer 2 has not started yet.
    const read = q >= 5.5 + l;
    const y1f = gs(q, l);
    if (read) { state = 'read'; status = y1f < .98 ? '读4 · 写y1' : '5 个位置'; }
    L.list.slice(0, 4).forEach((c, j) => {
      const f = phase(q, l + 0.4 + j * 0.04, l + 0.9 + j * 0.04);
      c.fill.style.opacity = (q >= l + 1 ? 1 : f).toFixed(3);
      setState(c, read ? 'read' : q >= l + 1 || f > .5 ? 'written' : 'empty');
    });
    const yc = L.list[4];
    yc.fill.style.opacity = (read ? y1f : 0).toFixed(3);
    setState(yc, y1f > .5 ? 'written' : 'empty');
    L.status.textContent = status;
    L.g.classList.toggle('is-active', (prefill && i === l + 1) || (read && i === l + 6));
    L.band.setAttribute('stroke-width', (prefill && i === l + 1) || (read && i === l + 6) ? 2 : 1);
  });

  // output head wording
  const activeHead = (q >= 3.4 && q < 4.6) || (q >= 8.4);
  el.oh.classList.toggle('is-active', activeHead);
  const hb = el.oh.querySelector('.jl-head-box');
  hb.setAttribute('stroke-width', activeHead ? 2 : 1);
  let h1 = '等待 p4 的最终隐藏状态', h2 = '';
  if (q >= 3.4 && q < 4.6) { h1 = '只取 p4 的 h³'; h2 = q < 4 ? '→ logits' : '→ 采样 y1 · y1 尚无 KV'; }
  if (q >= 4.6 && q < 8.4) { h1 = '等待 y1 走完三层'; h2 = 'Decode 仍遍历完整模型'; }
  if (q >= 8.4) { h1 = '只取 y1 的最终隐藏状态'; h2 = q < 9 ? '→ logits → 采样 y2' : '→ 采样 y2 · y2 尚无 KV'; }
  setText(el.ohSub, h1); setText(el.ohSub2, h2);

  // retained output sequence
  const y1 = q >= 4 ? 1 : 0, y2 = q >= 9 ? 1 : 0;
  el.slots[0].g.style.opacity = y1.toFixed(3);
  el.slots[1].g.style.opacity = y2.toFixed(3);
  let s1 = '尚未选出';
  if (q >= 4 && q < 5) s1 = '刚选出 · 尚无 KV';
  else if (q >= 5 && q < 6) s1 = '等待作为输入';
  else if (q >= 6 && q < 7) s1 = '层 1 已写 KV';
  else if (q >= 7 && q < 8) s1 = '层 1–2 已写 KV';
  else if (q >= 8) s1 = '三层 KV 已写';
  setText(el.slots[0].s, s1);
  setText(el.slots[1].s, y2 ? '刚选出 · 无 KV' : '尚未选出');

  // HTML status and caption
  const phaseName = q < 4.5 ? 'Prefill' : 'Decode';
  const posName = q < 4.5 ? '4 个位置' : '1 个新位置';
  const done = q < 4.5 ? clamp(Math.floor(q), 0, 3) : clamp(Math.floor(q - 5), 0, 3);
  const qPhase = root.querySelector('[data-jl-phase]');
  const qPos = root.querySelector('[data-jl-positions]');
  const qLayers = root.querySelector('[data-jl-layers]');
  if (qPhase) qPhase.textContent = phaseName;
  if (qPos) qPos.textContent = posName;
  if (qLayers) qLayers.textContent = done + ' / 3';
  setText(root.querySelector('[data-jl-caption]'), STEPS[i][1]);
  setText(root.querySelector('[data-jl-stepname]'), STEPS[i][0]);
  const count = root.querySelector('[data-jl-count]');
  if (count) count.textContent = (i + 1) + ' / ' + STEPS.length;
  const range = root.querySelector('[data-jl-range]');
  if (range && document.activeElement !== range) range.value = String(q);
  root.dataset.jlStep = q.toFixed(2);
  root.dataset.jlPhase = phaseName;
  root.dataset.jlLayers = String(done);
  root.dataset.jlOutput = q >= 9 ? 'y1,y2' : q >= 4 ? 'y1' : '';
  const play = root.querySelector('[data-jl-play]');
  if (play) {
    play.textContent = playing ? '暂停' : '▶ 播放';
    play.setAttribute('aria-pressed', playing ? 'true' : 'false');
  }
  const desc = `阶段 ${phaseName}；本轮 ${posName}；已经过 ${done} / 3 层。${STEPS[i][1]}`;
  svg.setAttribute('aria-label', desc);
  const d = root.querySelector('desc');
  if (d) d.textContent = desc;
}

function stop() { playing = false; if (raf) cancelAnimationFrame(raf); raf = 0; render(); }
function tick(t) {
  if (!playing) return;
  if (!last) last = t;
  const dt = Math.min(120, t - last); last = t;
  p = Math.min(9, p + dt / (1900 / speed));
  if (p >= 9) { p = 9; playing = false; }
  render();
  if (playing) raf = requestAnimationFrame(tick); else raf = 0;
}
function play() {
  if (playing) { stop(); return; }
  if (p >= 8.999) p = 0;
  playing = true; last = 0; raf = requestAnimationFrame(tick);
  render();
}
function go(n) { stop(); p = clamp(n, 0, 9); render(); }

function controls() {
  root.innerHTML = `<div class="jl-caption-line"><span class="jl-kicker">逐层观察 · 可暂停/回看</span><strong data-jl-stepname></strong></div>
  <div class="jl-controls">
    <button type="button" data-jl-play aria-pressed="false">▶ 播放</button>
    <button type="button" data-jl-prev>← 上一步</button>
    <span class="jl-count" data-jl-count>1 / 10</span>
    <button type="button" data-jl-next>下一步 →</button>
    <button type="button" data-jl-reset>重置</button>
    <label class="jl-speed">速度 <select data-jl-speed><option value=".75">0.75×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option></select></label>
  </div>
  <div class="jl-meters" aria-live="polite">
    <span>阶段 <b data-jl-phase>Prefill</b></span>
    <span>本轮输入 <b data-jl-positions>4 个位置</b></span>
    <span>已经过 <b data-jl-layers>0 / 3</b> 层</span>
  </div>
  <p class="jl-caption" data-jl-caption aria-live="polite"></p>
  <div class="jl-canvas"></div>
  <label class="jl-scrub">教学进度（不是实测耗时）<input type="range" min="0" max="9" step="0.01" value="0" data-jl-range aria-label="教学进度，可回看中间过程"></label>
  <p class="jl-note">3 层与 4 个输入位置是教学假设；p1…p4 是位置编号，不是分词片段。每一层各有一份 K/V；层是模型内部顺序，不是三轮请求调度。y1 与 y2 的「尚无 KV」表示它们还没有作为输入经过前向。</p>`;
  svg = node('svg', { role: 'img', 'aria-label': '逐层前向：三层模型的 Prefill / Decode 与 KV 写入' });
  const title = node('title', {}, svg); title.textContent = '逐层前向：三层模型的 Prefill / Decode 与 KV 写入';
  node('desc', {}, svg);
  root.querySelector('.jl-canvas').append(svg);
  root.querySelector('[data-jl-play]').onclick = play;
  root.querySelector('[data-jl-prev]').onclick = () => go(Math.ceil(p - .001) - 1);
  root.querySelector('[data-jl-next]').onclick = () => go(Math.floor(p + .001) + 1);
  root.querySelector('[data-jl-reset]').onclick = () => { stop(); p = 0; render(); };
  root.querySelector('[data-jl-speed]').onchange = e => { speed = Number(e.target.value); };
  root.querySelector('[data-jl-range]').oninput = e => {
    // stop() renders and may rewrite an unfocused range back to the old value.
    // Read the new value first, then let stop() render that position.
    const next = clamp(Number(e.target.value), 0, 9);
    p = next; stop();
  };
  root.addEventListener('keydown', e => {
    if (e.target.matches('input,select,button,a')) return;
    if (e.key === ' ') { e.preventDefault(); play(); }
    if (e.key === 'ArrowRight') { e.preventDefault(); go(Math.floor(p + .001) + 1); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); go(Math.ceil(p - .001) - 1); }
  });
}

function mount() {
  if (mounted) return;
  mounted = true;
  controls();
  build();
  new ResizeObserver(() => {
    const w = Math.round(root.clientWidth || 0);
    if (w && Math.abs(w - W) > 24) { stop(); build(); }
  }).observe(root);
}
const details = root.closest('details');
if (details && location.hash === '#mechanism-extension') details.open = true;
if (!details || details.open) mount();
else details.addEventListener('toggle', () => { if (details.open) mount(); }, { once: true });
details && details.addEventListener('toggle', () => { if (!details.open) stop(); });
document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
window.addEventListener('pagehide', stop);
})();
