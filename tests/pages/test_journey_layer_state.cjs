const assert = require('node:assert/strict');
const { deriveState } = require('../../pages/motion/journey-layer.js');

const statuses = q => deriveState(q).layers.map(layer => layer.status);
const active = q => deriveState(q).layers.filter(layer => layer.active).map(layer => layer.layer);
const fills = q => deriveState(q).layers.map(layer => layer.y1Fill);

// p5.45/5.49: y1 is between samples and the first Decode layer. The caption,
// packet, shelf labels and output label must all stay before any forward work.
for (const q of [5.45, 5.49]) {
  const s = deriveState(q);
  assert.equal(s.stepName, 'y1 成为下一轮输入', `p=${q}: step name must not declare Decode layer 1`);
  assert.equal(s.y1Status, '待进层1', `p=${q}: y1 is not yet writing layer 1`);
  assert.deepEqual(active(q), [], `p=${q}: no layer is active before Decode starts`);
  assert.deepEqual(statuses(q), ['4 个位置', '4 个位置', '4 个位置'], `p=${q}: only Prefill KV exists`);
  assert.deepEqual(fills(q), [0, 0, 0], `p=${q}: no y1 KV has been written`);
  assert.match(s.stepCaption, /前向尚未开始/);
  assert.doesNotMatch(s.stepCaption, /写入/);
  assert.equal(s.decodeTitle, '同一 y1');
}

// p5.5/5.6: layer 1 is reading history and writing y1. It must be highlighted,
// and its shelf may not claim all five positions have completed.
for (const q of [5.5, 5.6]) {
  const s = deriveState(q);
  assert.deepEqual(active(q), [1], `p=${q}: layer 1 is the current layer`);
  assert.equal(s.layers[0].status, '读4 · 写y1');
  assert.deepEqual(statuses(q), ['读4 · 写y1', '4 个位置', '4 个位置'], `p=${q}: only layer 1 is writing y1`);
  assert.equal(s.y1Status, '层1 写入中');
  assert.equal(s.done, 0, `p=${q}: no Decode layer has completed`);
  assert.equal(s.decodeTitle, 'y1→层1');
  assert.ok(s.layers[0].y1Fill < 1, `p=${q}: layer 1 y1 write is unfinished`);
}

// p6: layer 1 completes, layer 2 has not started. Completion is decoded from
// the Decode clock, not from the Prefill `q >= l + 1` condition.
{
  const s = deriveState(6);
  assert.deepEqual(statuses(6), ['5 个位置', '4 个位置', '4 个位置']);
  assert.deepEqual(active(6), []);
  assert.equal(s.done, 1);
  assert.equal(s.y1Status, '层 1 已写 KV');
  assert.equal(s.stepName, 'Decode · 第 1 层 · 已写完');
  assert.equal(s.layers[0].y1Written, true);
  assert.equal(s.layers[1].y1Written, false);
  assert.equal(s.decodeTitle, 'h¹(y1)');
}

// p6.6 / p7.6: the current layer has an active band and a "writing" shelf; the
// completed count stays at the layers that actually finished.
{
  const s = deriveState(6.6);
  assert.deepEqual(active(6.6), [2]);
  assert.deepEqual(statuses(6.6), ['5 个位置', '读4 · 写y1', '4 个位置']);
  assert.equal(s.done, 1);
  assert.equal(s.y1Status, '层2 写入中');
  assert.equal(s.decodeTitle, 'h¹→层2');
  assert.ok(s.layers[1].y1Fill > 0 && s.layers[1].y1Fill < 1);
  assert.equal(s.layers[2].y1Fill, 0);
}
{
  const s = deriveState(7.6);
  assert.deepEqual(active(7.6), [3]);
  assert.deepEqual(statuses(7.6), ['5 个位置', '5 个位置', '读4 · 写y1']);
  assert.equal(s.done, 2, 'p7.6: only layers 1 and 2 have completed');
  assert.equal(s.y1Status, '层3 写入中');
  assert.equal(s.decodeTitle, 'h²→层3');
  assert.match(s.stepName, /第 3 层 · 写入中/);
  assert.match(s.stepCaption, /正在写/);
  assert.doesNotMatch(s.stepCaption, /到这里 y1 才遍历完完整模型/);
  assert.ok(s.layers[2].y1Fill > 0 && s.layers[2].y1Fill < 1);
}

// p8: the third Decode layer completes; only then is the full traversal true.
{
  const s = deriveState(8);
  assert.deepEqual(statuses(8), ['5 个位置', '5 个位置', '5 个位置']);
  assert.deepEqual(active(8), []);
  assert.equal(s.done, 3);
  assert.equal(s.y1Status, '三层 KV 已写');
  assert.equal(s.stepName, 'Decode · 第 3 层 · 已写完');
  assert.match(s.stepCaption, /到这里 y1 才遍历完完整模型/);
  assert.equal(s.headActive, true);
}

// Keep the existing sampling boundaries: y1 at p4, y2 at p9. The sampling step
// itself only forms logits before those positions.
{
  let s = deriveState(3.5);
  assert.equal(s.stepName, '输出头 · 形成 y1 logits');
  assert.equal(s.y1Status, '尚未选出');
  assert.equal(s.y2Status, '尚未选出');
  assert.equal(s.output, '');
  assert.deepEqual(fills(3.5), [0, 0, 0]);
  s = deriveState(4);
  assert.equal(s.stepName, '采样 y1');
  assert.equal(s.y1Status, '刚选出 · 尚无 KV');
  assert.equal(s.output, 'y1');
  assert.deepEqual(fills(4), [0, 0, 0], 'p4: y1 is sampled but has no KV yet');
  s = deriveState(8.5);
  assert.equal(s.stepName, '输出头 · 形成 y2 logits');
  assert.equal(s.y2Status, '尚未选出');
  assert.equal(s.output, 'y1');
  assert.match(s.headLine2, /待采样 y2/);
  s = deriveState(9);
  assert.equal(s.stepName, '采样 y2');
  assert.equal(s.y2Status, '刚选出 · 无 KV');
  assert.equal(s.output, 'y1,y2');
  assert.deepEqual(statuses(9), ['5 个位置', '5 个位置', '5 个位置'], 'y2 has no KV at p9');
}

// Prefill has the same completed/current distinction: no KV before p0.5, one
// layer at a time, and integer boundaries report completion rather than the
// next layer's result.
{
  let s = deriveState(0.4);
  assert.deepEqual(statuses(0.4), ['空', '空', '空']);
  assert.deepEqual(active(0.4), []);
  assert.match(s.stepCaption, /还没有任何 KV/);
  s = deriveState(0.5);
  assert.deepEqual(active(0.5), [1]);
  assert.equal(s.layers[0].status, '写 p1…p4');
  assert.equal(s.done, 0);
  s = deriveState(1);
  assert.deepEqual(statuses(1), ['4 个位置', '空', '空']);
  assert.deepEqual(active(1), []);
  assert.equal(s.done, 1);
  assert.match(s.stepName, /第 1 层 · 已写完/);
  assert.equal(s.layers[1].status, '空', 'layer 2 has not started at p1');
  s = deriveState(1.5);
  assert.deepEqual(active(1.5), [2]);
  assert.equal(s.layers[1].status, '写 p1…p4');
  assert.equal(s.done, 1);
  s = deriveState(2);
  assert.deepEqual(statuses(2), ['4 个位置', '4 个位置', '空']);
  assert.deepEqual(active(2), []);
  assert.equal(s.done, 2);
  s = deriveState(2.5);
  assert.deepEqual(active(2.5), [3]);
  assert.equal(s.done, 2);
  s = deriveState(3);
  assert.deepEqual(statuses(3), ['4 个位置', '4 个位置', '4 个位置']);
  assert.deepEqual(active(3), []);
  assert.equal(s.done, 3);
  assert.equal(s.output, '');
}

// Invariants over the whole seekable range: at most one current layer, every
// "writing" claim belongs to an active layer, and the completed counter equals
// the number of finished-layer shelves.
for (let n = 0; n <= 90; n++) {
  const q = n / 10;
  const s = deriveState(q);
  assert.ok(s.done >= 0 && s.done <= 3, `p=${q}: done stays in range`);
  assert.ok(s.layers.filter(layer => layer.active).length <= 1, `p=${q}: one current layer at most`);
  const complete = q < 4.5
    ? s.layers.filter(layer => layer.status === '4 个位置').length
    : q < 5.5 ? 0
    : s.layers.filter(layer => layer.status === '5 个位置').length;
  assert.equal(s.done, complete, `p=${q}: "已经过" counts completed layers only`);
  for (const layer of s.layers) {
    if (layer.status === '读4 · 写y1') assert.equal(layer.active, true, `p=${q}: a writing layer is highlighted`);
    if (layer.status === '写 p1…p4') assert.equal(layer.active, true, `p=${q}: a Prefill writing layer is highlighted`);
    if (layer.y1Written) assert.equal(layer.status, '5 个位置', `p=${q}: only completed Decode shelves claim five positions`);
    if (layer.status === '5 个位置') assert.equal(layer.y1Written, true, `p=${q}: five positions means the y1 write completed`);
  }
  if (/写入中/.test(s.y1Status)) {
    assert.equal(s.layers.filter(layer => layer.active).length, 1, `p=${q}: output write state has a visible active layer`);
  }
}

console.log('PASS: journey-layer Prefill/Decode completion, active layer, per-layer KV labels, sampling boundaries and y2-without-KV semantics across p0…p9; y1=p4 and y2=p9 preserved.');
