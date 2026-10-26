const assert = require('node:assert/strict');
const M = require('../../pages/motion/attention-multihead.js');
const near = (a, b, eps, label) => assert.ok(Math.abs(a - b) <= eps, `${label}: ${a} != ${b}`);
const data = M.example();

assert.equal(data.headCount, 2);
assert.equal(data.headDim, 2);
assert.deepEqual(data.x, [1, 1]);
assert.deepEqual(data.heads[0].q, [1, 0]);
assert.deepEqual(data.heads[1].q, [0, 1]);
assert.deepEqual(data.heads[0].weights.map(w => Number(w.toFixed(8))), [0.66976155, 0.33023845]);
assert.deepEqual(data.heads[1].weights.map(w => Number(w.toFixed(8))), [0.33023845, 0.66976155]);
near(data.heads[0].result[0], 1.3395230986533138, 1e-12, 'h0d0');
near(data.heads[0].result[1], 0.6604769013466862, 1e-12, 'h0d1');
near(data.heads[1].result[0], 0.6604769013466862, 1e-12, 'h1d0');
near(data.heads[1].result[1], 1.3395230986533138, 1e-12, 'h1d1');
for (const head of data.heads) near(head.weights[0] + head.weights[1], 1, 1e-12, 'weights sum');

// Concat keeps four coordinates in head order; it is not an element-wise sum.
assert.equal(data.concat.length, 4);
assert.deepEqual(data.concat, [...data.heads[0].result, ...data.heads[1].result]);
assert.notDeepEqual(data.concat, data.heads[0].result.map((v, i) => v + data.heads[1].result[i]));
assert.equal(data.woIsSpecialAverage, true);
assert.deepEqual(data.output, [1, 1]);
assert.equal(data.output.length, 2);
assert.equal(data.isPerformanceClaim, false);
assert.equal(data.isAccuracyClaim, false);

console.log('PASS: multi-head model — two independent Wq/weights/results, 4-d concat, special 4→2 Wo, no performance claim.');
