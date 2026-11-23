const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const M = require('../../pages/sglang/serving-labs/model.js');
const E = require('../../pages/sglang/serving-labs/evidence.js');

const b = M.memoryBudget();
assert.equal(b.slack, 5);
assert.equal(b.rawBudget, 15);
assert.equal(b.tokenBytes, 65536);
assert.equal(b.profiledTokens, 245760);
assert.equal(b.capacity, 99968);
assert.equal(b.demand, 30720);
assert.equal(b.fits, true);
assert.equal(b.executionFits, true);
assert.equal(M.memoryBudget({ fraction: .95 }).fits, true);
assert.equal(M.memoryBudget({ fraction: .95 }).executionFits, false,
  'more KV space must not erase a future activation/graph/workspace peak');
assert.equal(M.memoryBudget({ available: 8, fraction: .75 }).capacity, 0);
assert.equal(M.memoryBudget({ available: 8, fraction: .75 }).fits, false);
assert.equal(M.memoryBudget({ tokenLimit: 1000000 }).capacity, b.profiledTokens);
assert.equal(M.memoryBudget({ context: 32768 }).fits, false);
assert.equal(M.memoryBudget({ bytes: 1, tokenLimit: 0 }).profiledTokens, 2 * b.profiledTokens);
const boundary = M.memoryBudget({ context: 2049 });
assert.equal(boundary.requestedPositions, 2561);
assert.equal(boundary.perRequestPages, 21);
assert.equal(boundary.paddingPerRequest, 127);
assert.equal(boundary.demandPages, 252);
assert.equal(boundary.demand, boundary.usefulPositions + boundary.paddingTotal);
assert.equal(boundary.paddingTotal, 1524);
assert.equal(b.pages, 781);
assert.equal(b.demandPages, 240);
assert.equal(b.alignmentLoss, 32);
assert.equal(b.userCapBinding, true);
assert.equal(M.memoryBudget({ requests: 2, context: 1, reserve: 0 }).demandPages, 2,
  'each unshared request rounds independently, not one aggregate rounding');
for (const page of [16, 128, 256]) {
  for (const context of [1, 127, 128, 129, 2049]) {
    const x = M.memoryBudget({ page, context });
    assert.equal(x.demand, x.demandPages * page);
    assert.equal(x.perRequest, x.requestedPositions + x.paddingPerRequest);
    assert.ok(x.paddingPerRequest >= 0 && x.paddingPerRequest < page);
    assert.equal(x.capacity, x.pages * page);
  }
}
assert.throws(() => M.memoryBudget({ available: 41 }));
assert.throws(() => M.memoryBudget({ requests: -1 }));
assert.deepEqual(M.pageGrowth().map(x => x.newPages - x.oldPages), [0, 1, 0]);
assert.deepEqual(M.pageGrowth([8, 9, 4]).map(x => x.newPages - x.oldPages), [1, 0, 1]);

for (const scenario of ['l3', 'host', 'device-shortage', 'cancel', 'miss']) {
  const p = M.hicache(scenario);
  for (const e of p.events) assert.ok(e.start >= 0 && e.end >= e.start && e.end <= p.duration);
  for (const e of p.events.filter(e => e.id.startsWith('attention-'))) {
    const copy = p.events.find(x => x.id === 'copy-' + e.id.slice(-1));
    assert.ok(copy.end <= e.start, 'a layer cannot consume KV before its own event');
  }
  assert.deepEqual(M.hicacheState(p, 0), M.hicacheState(p, 0), 'replay is deterministic');
}
let p = M.hicache('l3'), s = M.hicacheState(p, 16);
assert.equal(s.ack, false);
assert.equal(s.hostPinned, true);
assert.equal(s.deviceTransferPinned, true);
assert.equal(s.readLayers.length, 1, 'layer 0 may finish before whole-transfer ACK consumption');
s = M.hicacheState(p, 20);
assert.equal(s.ack, true);
assert.equal(s.requestPinned, true, 'transfer retirement must not release the active request');
assert.equal(s.hostValid, true, 'ACK does not evict the host copy');
// Query, completed IO, adoption, per-layer readiness and both ACKs are distinct.
let checkpoints = M.hicache('l3');
assert.equal(M.hicacheState(checkpoints, 2).storageHitKnown, true);
assert.equal(M.hicacheState(checkpoints, 2).hostAllocated, false);
assert.equal(M.hicacheState(checkpoints, 4).ioOwns, true);
assert.equal(M.hicacheState(checkpoints, 4).hostTransferPinned, false);
assert.equal(M.hicacheState(checkpoints, 7.5).storageReadDone, true);
assert.equal(M.hicacheState(checkpoints, 7.5).hostValid, false);
assert.equal(M.hicacheState(checkpoints, 7.5).ioOwns, true);
assert.equal(M.hicacheState(checkpoints, 8).storageAck, true);
assert.equal(M.hicacheState(checkpoints, 8).hostValid, true);
assert.equal(M.hicacheState(checkpoints, 8.5).loadPreparing, true);
assert.equal(M.hicacheState(checkpoints, 8.5).hostTransferPinned, true);
assert.equal(M.hicacheState(checkpoints, 8.5).deviceAllocated, false);
assert.equal(M.hicacheState(checkpoints, 9).deviceAllocated, true);
assert.deepEqual(M.hicacheState(checkpoints, 11.99).readyLayers, []);
assert.deepEqual(M.hicacheState(checkpoints, 12.5).readyLayers, [0]);
assert.deepEqual(M.hicacheState(checkpoints, 12.5).activeLayers, [0]);
assert.equal(M.hicacheState(checkpoints, 18.5).ackPending, true);
assert.equal(M.hicacheState(checkpoints, 18.5).deviceTransferPinned, true);
assert.equal(M.hicacheState(checkpoints, 20.5).hostTransferPinned, false);
assert.equal(M.hicacheState(checkpoints, 20.5).requestPinned, true);
assert.equal(M.hicacheState(checkpoints, 20.5).hostValid, true);
assert.equal(M.hicacheState(M.hicache('device-shortage'), 8.5).hostTransferPinned, true);
assert.equal(M.hicacheState(M.hicache('device-shortage'), 9).hostTransferPinned, false);
assert.equal(M.hicacheState(M.hicache('device-shortage'), 9).hostValid, true);
p = M.hicache('cancel');
s = M.hicacheState(p, 6);
assert.equal(s.canceled, true);
assert.equal(s.ioOwns, true);
assert.equal(s.hostReleased, false);
assert.equal(M.hicacheState(p, 8).hostReleased, false);
assert.equal(M.hicacheState(p, 8).releaseQueued, true);
assert.equal(M.hicacheState(p, 8).ioOwns, false);
assert.equal(M.hicacheState(p, 9).hostReleased, true);
assert.equal(M.hicacheState(p, 9).releaseQueued, false);
assert.equal(M.hicache('device-shortage').events.some(e => e.lane === 'h2d'), false);
assert.equal(M.hicache('miss').events.some(e => e.id === 'host-alloc'), false);
assert.equal(M.deployment('oom').events.some(e => e.id === 'ready'), false);
assert.equal(M.deployment('cancel').events.some(e => e.id === 'finish'), false);
for (const scenario of ['normal', 'probe', 'cancel']) {
  const plan = M.deployment(scenario), by = id => plan.events.find(e => e.id === id);
  const transfer = id => plan.transfers.find(e => e.id === id);
  assert.ok(plan.transfers.every(e => e.start < e.end && e.start >= 0 && e.end <= plan.duration));
  assert.ok(transfer('post-http').start >= by('ready').end, 'request follows readiness check');
  assert.ok(by('prefill').start >= transfer('submit-prefill').end, 'submission precedes GPU execution');
  assert.ok(transfer('return-id').start >= by('prefill').end, 'sample exists before return');
  assert.equal(transfer('return-visible').end, by('text').end, 'client first output has a separate endpoint');
  assert.ok(by('decode').start < by('text').end, 'Decode does not wait for full text return');
  if (scenario === 'cancel') assert.ok(transfer('abort-ipc').end < by('decode').end, 'cancel does not instantly finish GPU work');
}

const rows = [
  { success: true, start: 0, first: .1, end: .34, out: 3 },
  { success: true, start: .05, first: .2, end: .62, out: 5 },
  { success: false, start: .1, first: null, end: .95, out: 0 }
];
let m = M.metrics(rows, 1, { ttft: .2, tpot: .11 });
assert.equal(m.completed, 2);
assert.equal(m.failed, 1);
assert.equal(m.goodput, 1);
assert.equal(m.outputThroughput, 8);
assert.ok(Math.abs(m.p95TTFT - .1475) < 1e-10);
assert.ok(Math.abs(m.p95TPOT - .11925) < 1e-10);
assert.equal(M.metrics([{ success: true, start: 0, first: null, end: 1, out: 3 }], 1).p95TTFT, null);
assert.equal(M.metrics([{ success: true, start: 0, first: .2, end: .3, out: 1 }], 1).tpotSamples, 0);
assert.throws(() => M.metrics(rows, 0));
const normal = M.workload({ concurrency: 1 }), failed = M.workload({ concurrency: 1, failure: true });
assert.deepEqual(normal.rows.map(x => [x.start, x.work, x.end]), failed.rows.map(x => [x.start, x.work, x.end]),
  'changing sample success must not secretly accelerate execution');
assert.ok(normal.rows[1].start > normal.rows[1].scheduled);
assert.ok(M.workload({ concurrency: 8 }).rows[1].start < normal.rows[1].start);

const trace = M.parseTrace({ traceEvents: [
  { ph: 'X', cat: 'kernel', name: 'A', ts: 100, dur: 10, pid: 0, tid: 7, args: { correlation: 1 } },
  { ph: 'X', cat: 'Kernel', name: 'B', ts: 105, dur: 10, pid: 0, tid: 8 },
  { ph: 'X', cat: 'Runtime', name: 'cudaLaunchKernel', ts: 99, dur: 1, pid: 5, tid: 6, args: { correlation: 1 } },
  { ph: 'B', name: 'CPU scope', ts: 98, pid: 5, tid: 6 },
  { ph: 'E', ts: 116, pid: 5, tid: 6 },
  { ph: 'X', name: 'bad', ts: 100, dur: -1 },
  null
] });
const ts = M.traceStats(trace);
assert.equal(ts.observedGpuActive, 15);
assert.equal(ts.summedGpuDuration, 20);
assert.equal(trace.diagnostics.dropped, 1);
assert.equal(M.launchCandidates(trace, trace.spans.find(e => e.name === 'A')).length, 1);
assert.equal(M.traceStats(trace, 7, 12).observedGpuActive, 5);
assert.equal(M.unionDuration([[0, 10], [5, 15], [18, 20]]), 17);
assert.throws(() => M.parseTrace({ traceEvents: [] }));
assert.throws(() => M.parseTrace({}));
const captureRange = M.parseTrace({ traceEvents: [
  { ph: 'M', name: 'process_name', pid: 0, args: { name: 'GPU 0' } },
  { ph: 'X', cat: 'Trace', name: 'capture scope', ts: 0, dur: 100, pid: 0, tid: 1 },
  { ph: 'X', cat: 'cuda_driver', name: 'cuLaunchKernel', ts: 1, dur: 1, pid: 0, tid: 1 },
  { ph: 'X', cat: 'kernel', name: 'actual kernel', ts: 10, dur: 5, pid: 0, tid: 7 }
] });
assert.equal(M.traceStats(captureRange).observedGpuActive, 5,
  'a GPU process name must not turn a whole capture or CPU driver range into device activity');

// Completion gates must not collapse into first text or cancellation notification.
for (const scenario of ['normal', 'cancel']) {
  const plan = M.deployment(scenario), by = id => plan.events.find(e => e.id === id);
  assert.ok(by('text').end < by('terminal').start);
  assert.ok(by('terminal').end <= by(scenario === 'cancel' ? 'retire' : 'finish').start);
  if (scenario === 'cancel') assert.ok(by('cancel-ack').end < by('decode').end,
    'acknowledging notification must not imply device completion');
}
const baseline = M.workload(), failureWorkload = M.workload({ failure: true });
assert.equal(baseline.duration, 33.5);
assert.deepEqual(baseline.rows.map(r => [r.scheduled, r.permit, r.start, r.end]),
  failureWorkload.rows.map(r => [r.scheduled, r.permit, r.start, r.end]), 'failure keeps the same work and denominator');
assert.equal(baseline.rows[2].permit - baseline.rows[2].scheduled, 2.5);
assert.equal(baseline.rows[2].first - baseline.rows[2].start, 10.5);
assert.equal(M.metrics(baseline.rows, baseline.duration).goodputCount, 2);
assert.ok(M.metrics(failureWorkload.rows, failureWorkload.duration).p95TTFT < M.metrics(baseline.rows, baseline.duration).p95TTFT);
const gap = M.largestGpuGap(captureRange);
assert.deepEqual(gap, [15, 100]);
const gapView = M.traceFocusWindow(captureRange, ...gap);
assert.ok(gapView.start <= gap[0] && gapView.end >= gap[1], 'the complete gap must fit');
const cpuOnly = M.parseTrace([{ ph: 'X', cat: 'cpu_op', name: 'host', ts: 0, dur: 20, pid: 1, tid: 1 }]);
assert.equal(M.largestGpuGap(cpuOnly), null, 'a CPU-only trace has no GPU gap evidence');
assert.throws(() => M.parseTrace([{ ph: 'X', cat: 'kernel', name: 'zero', ts: 0, dur: 0 }]));
for (const [a, b] of [[0, 3], [40, 43], [96, 100], [10, 85], [0, 100]]) {
  const w = M.traceFocusWindow({ duration: 100 }, a, b);
  assert.ok(w.start <= a && w.end >= b, 'focus includes the event at boundaries');
}

const samples = path.resolve(__dirname, '../../pages/sglang/trace-lab/samples');
const manifest = JSON.parse(fs.readFileSync(path.join(samples, 'manifest.json'), 'utf8'));
const bytes = fs.readFileSync(path.join(samples, manifest.sample));
assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), manifest.sample_sha256);
const real = M.parseTrace(JSON.parse(bytes));
assert.ok(real.spans.length > 5000);
assert.ok(M.traceStats(real).gpuCount > 1000);
const candidate = real.spans.find(e => e.gpu && M.launchCandidates(real, e).length === 1);
assert.ok(candidate, 'the real legacy Kineto sample retains a CPU-to-GPU correlation');
assert.equal(E.revision, M.revision);
for (const v of Object.values(E.sources)) assert.match(v.url, new RegExp(M.revision));
console.log('Serving labs: memory, HiCache ownership/dependencies, sample statistics, trace intervals and measured provenance passed.');
