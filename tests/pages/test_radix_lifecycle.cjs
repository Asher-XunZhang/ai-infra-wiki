/* Teaching invariants for the RadixCache search / insert / protect / evict lab. */
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const M = require('../../pages/sglang/inference-overview/radix-model.js');

assert.equal(M.pageSize, 4);
assert.equal(M.totalPages, 4);
assert.equal(M.states.length, 12);
assert.equal(M.states.length, M.steps.length);

function view(i) { return M.derive(M.states[i]); }
function node(view, id) { return view.nodes[id]; }

for (let i = 0; i < M.states.length; i += 1) {
  const v = view(i);
  assert.equal(v.step, i);
  assert.equal(v.usedPages + v.freePages, M.totalPages, `page conservation at step ${i}`);
  assert.equal(v.pages.length, M.totalPages);
  assert.equal(v.treeSlots, Object.values(v.nodes).filter(n => n.id !== 'root').reduce((n, x) => n + x.tokens.length, 0));
  assert.equal(v.lockedSlots, Object.values(v.nodes).filter(n => n.id !== 'root' && n.lock > 0).reduce((n, x) => n + x.tokens.length, 0));
  const treeSlots = [];
  for (const n of Object.values(v.nodes)) {
    if (n.id === 'root') continue;
    assert.equal(n.tokens.length, n.slots.length, `token/value length ${n.id}`);
    assert.ok(n.slots.every(slot => slot >= M.pageSize), `page 0 is reserved ${n.id}`);
    for (const slot of n.slots) treeSlots.push(slot);
    const expected = n.lock === 0 && n.children.length === 0;
    assert.equal(v.evictable.includes(n.id), expected, `evictable rule ${n.id} at ${i}`);
  }
  assert.equal(new Set(treeSlots).size, treeSlots.length, `tree nodes do not double-own slots at ${i}`);
  for (const page of v.pages) {
    assert.equal(page.cells.length, 4);
    if (page.kind === 'free') {
      assert.ok(page.cells.every(c => c.token === null), `free page ${page.id} is empty`);
      assert.equal(page.owner, null);
    } else {
      assert.ok(page.owner, `used page ${page.id} has an owner`);
      assert.ok(page.cells.every(c => c.token !== null), `used page ${page.id} is full`);
    }
    for (const cell of page.cells) {
      assert.equal(Math.floor(cell.slot / M.pageSize), page.id);
    }
  }
  for (const req of Object.values(v.requests)) {
    if (req.rowReleased) {
      assert.ok(req.row.every(entry => entry === null), `released row ${req.id}`);
      continue;
    }
    const slots = req.row.filter(Boolean).map(entry => entry.slot);
    assert.equal(new Set(slots).size, slots.length, `row ${req.id} has unique slots at ${i}`);
    for (const entry of req.row.filter(Boolean)) {
      assert.equal(Math.floor(entry.slot / M.pageSize) >= 1, true);
      assert.equal(entry.token, req.tokens[req.row.indexOf(entry)]);
    }
  }
}

assert.equal(node(view(0), 'n1').tokens.join(''), 'ABCDEFGH');
assert.deepEqual(node(view(0), 'n1').slots, [4, 5, 6, 7, 8, 9, 10, 11]);
assert.equal(view(0).pages.find(p => p.id === 1).owner, 'n1');
assert.equal(view(0).pages.find(p => p.id === 2).owner, 'n1');

assert.equal(M.states[1].requests.R1.prefixLen, 8);
assert.deepEqual(M.states[1].requests.R1.prefixIndices, [4, 5, 6, 7, 8, 9, 10, 11]);
assert.ok(M.states[1].requests.R1.row.every(e => e === null), "search returns indices without allocating a request row");
assert.equal(node(view(1), 'n1').lock, 1);
assert.deepEqual(view(1).evictable, []);

assert.deepEqual(node(view(2), 'n3').slots, [12, 13, 14, 15]);
assert.equal(view(2).pages.find(p => p.id === 3).owner, 'n3');
assert.equal(node(view(2), 'n3').lock, 1);
assert.equal(node(view(2), 'n1').lock, 1);
assert.equal(M.states[2].requests.R1.row.length, 12);
assert.equal(view(2).treeSlots, 12);

assert.equal(node(view(3), 'n1').tokens.join(''), 'EFGH');
assert.equal(node(view(3), 'n2').tokens.join(''), 'ABCD');
assert.deepEqual(node(view(3), 'n2').slots, [4, 5, 6, 7]);
assert.deepEqual(node(view(3), 'n1').slots, [8, 9, 10, 11]);
assert.equal(node(view(3), 'n2').lock, 2);
assert.equal(node(view(3), 'n1').lock, 1);
assert.equal(view(3).pages.find(p => p.id === 1).owner, 'n2');
assert.equal(view(3).pages.find(p => p.id === 2).owner, 'n1');
assert.equal(M.states[3].requests.R2.prefixLen, 4);

assert.equal(node(view(4), 'n4').tokens.join(''), 'XYZW');
assert.deepEqual(node(view(4), 'n4').slots, [16, 17, 18, 19]);
assert.equal(node(view(4), 'n2').lock, 2);
assert.equal(node(view(4), 'n4').lock, 1);
assert.equal(view(4).pages.find(p => p.id === 4).owner, 'n4');
assert.equal(view(4).treeSlots, 16);

assert.deepEqual(view(5).evictable, []);
assert.equal(view(5).meta.pressure.reclaimed, 0);

assert.equal(M.states[6].requests.R1.status, 'finished');
assert.equal(M.states[6].requests.R1.rowReleased, true);
assert.equal(node(view(6), 'n3').lock, 0);
assert.equal(node(view(6), 'n1').lock, 0);
assert.equal(node(view(6), 'n2').lock, 1);
assert.deepEqual(view(6).evictable, ['n3']);
assert.equal(view(6).pages.find(p => p.id === 3).owner, 'n3');

assert.equal(view(7).nodes.n3, undefined);
assert.equal(view(7).pages.find(p => p.id === 3).kind, 'free');
assert.deepEqual(view(7).evictable, ['n1']);

assert.equal(view(8).nodes.n1, undefined);
assert.equal(view(8).pages.find(p => p.id === 2).kind, 'free');
assert.deepEqual(view(8).evictable, []);
assert.equal(node(view(8), 'n2').lock, 1);
assert.equal(node(view(8), 'n4').lock, 1);

assert.equal(M.states[9].requests.R2.status, 'finished');
assert.equal(M.states[9].requests.R2.rowReleased, true);
assert.equal(node(view(9), 'n4').lock, 0);
assert.equal(node(view(9), 'n2').lock, 0);
assert.deepEqual(view(9).evictable, ['n4']);

assert.equal(view(10).nodes.n4, undefined);
assert.equal(view(10).pages.find(p => p.id === 4).kind, 'free');
assert.deepEqual(view(10).evictable, ['n2']);

assert.equal(view(11).nodes.n2, undefined);
assert.deepEqual(view(11).nodes.root.children, []);
assert.equal(view(11).usedPages, 0);
assert.equal(view(11).freePages, 4);
assert.deepEqual(view(11).evictable, []);


assert.equal(M.demos.length, 4);
assert.deepEqual(M.demos.map(demo => demo.id), ['journey', 'search', 'insert', 'remove']);
for (const demo of M.demos) {
  assert.equal(demo.states.length, demo.steps.length, `${demo.id} state/step count`);
  assert.equal(demo.states.length, demo.steps.length);
  for (let i = 0; i < demo.states.length; i += 1) {
    const state = demo.states[i];
    const v = M.derive(state);
    assert.equal(state.meta, demo.steps[i], `${demo.id} meta identity ${i}`);
    assert.equal(demo.steps[i].index, i, `${demo.id} local step index`);
    assert.equal(v.meta, demo.steps[i]);
    assert.ok(M.sources[demo.steps[i].source], `${demo.id} source ${i}`);
    if (demo.steps[i].flow) {
      for (const flow of demo.steps[i].flow) {
        assert.ok(flow.label && flow.from && flow.to, `${demo.id} flow shape ${i}`);
      }
    }
  }
}

const searchDemo = M.demos.find(demo => demo.id === 'search');
assert.equal(searchDemo.states.length, 3);
assert.equal(M.derive(searchDemo.states[1]).pages.find(p => p.id === 3).kind, 'free');
assert.equal(searchDemo.states[2].nodes.n3, undefined);
assert.equal(searchDemo.states[2].nodes.n2.lock, 2);
assert.equal(M.derive(searchDemo.states[2]).usedPages, 2);

const insertDemo = M.demos.find(demo => demo.id === 'insert');
assert.equal(insertDemo.states.length, 4);
const ready = M.derive(insertDemo.states[2]);
assert.equal(ready.pages.find(p => p.id === 3).kind, 'private');
assert.equal(ready.pages.find(p => p.id === 3).owner, 'R1');
assert.equal(insertDemo.states[2].nodes.n3, undefined);
assert.equal(ready.treeSlots, 8);
const inserted = M.derive(insertDemo.states[3]);
assert.equal(inserted.pages.find(p => p.id === 3).kind, 'node');
assert.equal(inserted.pages.find(p => p.id === 3).owner, 'n3');
assert.equal(inserted.treeSlots, 12);

const removeDemo = M.demos.find(demo => demo.id === 'remove');
assert.equal(removeDemo.states.length, 7);
assert.equal(M.derive(removeDemo.states[0]).usedPages, 4);
assert.deepEqual(M.derive(removeDemo.states[0]).evictable, []);
assert.equal(M.derive(removeDemo.states[6]).usedPages, 0);
assert.equal(M.derive(removeDemo.states[6]).freePages, 4);

if (process.env.SGLANG_SOURCE_DIR) {
  for (const [key, [file, line, symbol]] of Object.entries(M.sources)) {
    const code = execFileSync('git', ['-C', process.env.SGLANG_SOURCE_DIR, 'show', `${M.revision}:${file}`], { encoding: 'utf8' });
    assert.ok(code.split('\n')[line - 1].includes(symbol), `${key}: ${file}:${line} should contain ${symbol}`);
  }
}

console.log(`PASS: 12-step Radix search/split/insert/protect/evict model, page conservation, row mapping and ${process.env.SGLANG_SOURCE_DIR ? Object.keys(M.sources).length : 0} fixed-source anchors.`);

// Continuous seeking preserves causality, not just the settled teaching steps.
const chain = M.timeline();
const at = (phase, offset) => M.atTime(chain, chain.phases[phase].start + offset).raw;
assert.equal(at(1, 2.84).requests.R1.prefixLen, 0, 'indices do not arrive before the return packet');
assert.deepEqual(at(1, 2.85).requests.R1.prefixIndices, [4,5,6,7,8,9,10,11]);
assert.equal(at(1, 3.54).nodes.n1.lock, 0, 'match_prefix does not protect the path');
assert.equal(at(1, 3.55).nodes.n1.lock, 1, 'admission protects the path');
assert.ok(at(1, 3.8).requests.R1.row.every(x => x === null), 'matching does not allocate a row');
assert.equal(at(2, 1.09).nodes.n3, undefined, 'insert parameters do not create a node before traversal');
assert.equal(at(2, 1.1).nodes.n3.lock, 0, 'new leaf is initially unlocked');
assert.equal(at(2, 2.09).requests.R1.prefixLen, 8);
assert.equal(at(2, 2.1).requests.R1.prefixLen, 12, 'rematch extends prefix_indices');
assert.equal(at(2, 2.84).requests.R1.lastNode, 'n1');
assert.equal(at(2, 2.85).requests.R1.lastNode, 'n3');
assert.equal(at(3, 1.74).nodes.n2, undefined);
assert.equal(at(3, 1.75).nodes.n2.lock, 1, 'split inherits R1 protection, not the future R2 lock');
assert.equal(at(3, 3.55).nodes.n2.lock, 2);
assert.equal(at(6, .9).nodes.n3.lock, 0);
assert.equal(at(6, .9).requests.R1.rowReleased, false, 'unlock precedes request-row release');
assert.equal(at(6, 1.6).requests.R1.rowReleased, true);
assert.equal(M.derive(at(6, 2)).pages[2].kind, 'node', 'finished is not evicted');
assert.equal(M.derive(at(7, 1.09)).pages[2].kind, 'node');
assert.equal(M.derive(at(7, 1.1)).pages[2].kind, 'free');
assert.equal(at(8, 1.5).nodes.n2.lock, 1, 'R2 shared prefix survives eviction');
for (const demo of M.demos) {
  const timeline = M.timeline(demo.id);
  for (const phase of timeline.phases) {
    const final = M.atTime(timeline, phase.end).raw;
    assert.doesNotMatch([phase.meta.title, phase.meta.caption, phase.meta.note, ...phase.events.map(e=>e.label), ...phase.flows.map(f=>f.label)].join(' '), /页|请求行|KV 池|分配器|写入 K.V/, 'tree presentation excludes physical memory management');
    assert.ok(phase.flows.every(f=>!f.from.includes('data-page')&&!f.to.includes('data-page')));
    assert.deepEqual(final.nodes, demo.states[phase.index].nodes);
    assert.deepEqual(final.requests, demo.states[phase.index].requests);
    assert.ok(phase.events.every(e => M.sources[e.source]), 'each semantic event has a source anchor');
  }
  for (let t = 0; t <= timeline.duration; t += .05) {
    const sampled = M.atTime(timeline, t), v = M.derive(sampled.raw);
    assert.equal(v.freePages + v.usedPages, 4);
    for (const id of v.evictable) assert.equal(v.nodes[id].lock, 0);
    const saved = JSON.stringify(sampled.raw);
    M.atTime(timeline, timeline.duration); // Forward then backwards must not retain mutations.
    assert.equal(JSON.stringify(M.atTime(timeline, t).raw), saved);
  }
}
console.log('PASS: event ordering, tree-only presentation, admission versus matching, shared locks, reverse seeking, all four timelines.');
