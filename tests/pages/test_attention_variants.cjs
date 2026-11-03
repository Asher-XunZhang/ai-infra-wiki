const assert = require('node:assert/strict');
const M = require('../../pages/motion/attention-variants.js');

const near = (actual, expected, label) => {
  assert.ok(Math.abs(actual - expected) < 1e-12, `${label}: ${actual} != ${expected}`);
};
const matrix = (actual, expected, label) => {
  assert.equal(actual.length, expected.length, `${label}: row count`);
  actual.forEach((row, i) => {
    assert.equal(row.length, expected[i].length, `${label}: row ${i} width`);
    row.forEach((value, j) => near(value, expected[i][j], `${label}[${i}][${j}]`));
  });
};

// Shared recurrence: Sbar=D*Sprev; prediction=Sbar^T k; e=v-prediction;
// Snew=Sbar+beta*k*e^T; o=Snew^T q. Column-vector convention.
const s0 = [[2, 0], [0, 4]];
const write = { k: [1, 0], v: [6, 2], q: [1, 1] };
const direct = M.linearStep(s0, write, 'gdn', [0.5, 0.5], 0.5);
matrix(direct.sbar, [[1, 0], [0, 2]], 'direct sbar');
assert.deepEqual(direct.prediction, [1, 0]);
assert.deepEqual(direct.error, [5, 2]);
matrix(direct.update, [[2.5, 1], [0, 0]], 'direct update');
matrix(direct.state, [[3.5, 1], [0, 2]], 'direct state');
assert.deepEqual(direct.output, [3.5, 3]);
assert.deepEqual(s0, [[2, 0], [0, 4]], 'linearStep must not mutate Sprev');
assert.equal(direct.usesDelta, true);

// GDN: D=alpha*I, beta=.5.
const gdn = M.linearStateExample('gdn');
assert.equal(gdn.kind, 'linear-state');
assert.equal(gdn.modeLabel, 'GDN');
assert.equal(gdn.beta, 0.5);
assert.deepEqual(gdn.decay, [0.5, 0.5]);
matrix(gdn.updates[0].sbar, [[1, 0], [0, 2]], 'GDN sbar');
assert.deepEqual(gdn.updates[0].prediction, [1, 0]);
assert.deepEqual(gdn.updates[0].error, [5, 2]);
matrix(gdn.updates[0].update, [[2.5, 1], [0, 0]], 'GDN update');
matrix(gdn.updates[0].state, [[3.5, 1], [0, 2]], 'GDN Snew');
assert.deepEqual(gdn.updates[0].output, [3.5, 3]);
matrix(gdn.states[0], s0, 'GDN states[0]');
matrix(gdn.states[1], gdn.updates[0].state, 'GDN states[1]');
assert.deepEqual(gdn.outputs[0], gdn.updates[0].output);

// KDA: D=diag(.5,.25); only the second row differs from GDN in the first write.
const kda = M.linearStateExample('kda');
assert.deepEqual(kda.decay, [0.5, 0.25]);
assert.equal(kda.decayMode, 'channel');
matrix(kda.updates[0].sbar, [[1, 0], [0, 1]], 'KDA sbar');
assert.deepEqual(kda.updates[0].prediction, [1, 0]);
assert.deepEqual(kda.updates[0].error, [5, 2]);
matrix(kda.updates[0].update, [[2.5, 1], [0, 0]], 'KDA update');
matrix(kda.updates[0].state, [[3.5, 1], [0, 1]], 'KDA Snew');
assert.deepEqual(kda.updates[0].output, [3.5, 2]);

// DeltaNet: D=I, beta=1. Two writes of the same k->v keep row 1 at [6,2].
const delta = M.linearStateExample('delta');
assert.deepEqual(delta.decay, [1, 1]);
assert.equal(delta.beta, 1);
matrix(delta.updates[0].sbar, s0, 'Delta sbar');
matrix(delta.updates[0].state, [[6, 2], [0, 4]], 'Delta first write');
matrix(delta.updates[1].state, [[6, 2], [0, 4]], 'Delta second write');
assert.deepEqual(delta.updates[1].error, [0, 0]);
assert.deepEqual(delta.outputs[0], [6, 6]);
assert.deepEqual(delta.outputs[1], [6, 6]);

// Simple accumulation: D=I, beta=1, no error correction. Same write accumulates.
const add = M.linearStateExample('add');
assert.deepEqual(add.decay, [1, 1]);
assert.equal(add.beta, 1);
assert.equal(add.updates[0].usesDelta, false);
matrix(add.updates[0].state, [[8, 2], [0, 4]], 'add first write');
matrix(add.updates[1].state, [[14, 4], [0, 4]], 'add second write');
assert.deepEqual(add.outputs[0], [8, 6]);
assert.deepEqual(add.outputs[1], [14, 8]);

// GDN / KDA still decay on the second write; the error is not identically zero.
matrix(gdn.updates[1].sbar, [[1.75, 0.5], [0, 1]], 'GDN second sbar');
assert.deepEqual(gdn.updates[1].prediction, [1.75, 0.5]);
assert.deepEqual(gdn.updates[1].error, [4.25, 1.5]);
matrix(gdn.updates[1].state, [[3.875, 1.25], [0, 1]], 'GDN second state');
assert.deepEqual(gdn.outputs[1], [3.875, 2.25]);
matrix(kda.updates[1].sbar, [[1.75, 0.5], [0, 0.25]], 'KDA second sbar');
assert.deepEqual(kda.updates[1].error, [4.25, 1.5]);
matrix(kda.updates[1].state, [[3.875, 1.25], [0, 0.25]], 'KDA second state');
assert.deepEqual(kda.outputs[1], [3.875, 1.5]);
assert.notDeepEqual(gdn.updates[1].error, [0, 0], 'GDN repeat is not zero-error');
assert.notDeepEqual(kda.updates[1].error, [0, 0], 'KDA repeat is not zero-error');

// Parameters are clamped; invalid mode falls back to GDN; add/delta ignore alpha.
const zero = M.linearStateExample('gdn', { alpha: 0 });
matrix(zero.updates[0].sbar, [[0, 0], [0, 0]], 'alpha=0 sbar');
assert.deepEqual(zero.updates[0].state, [[3, 1], [0, 0]]);
assert.deepEqual(zero.updates[0].output, [3, 1]);
const clamped = M.linearStateExample('kda', { alpha1: 9, alpha2: -3 });
assert.deepEqual(clamped.decay, [1, 0]);
assert.equal(M.linearStateExample('nope').mode, 'gdn');
const addAlpha = M.linearStateExample('add', { alpha: 0 });
assert.deepEqual(addAlpha.decay, [1, 1]);
assert.deepEqual(addAlpha.states[2][0], add.states[2][0]);

assert.equal(gdn.sameWriteRepeated, true);
assert.equal(gdn.isPerformanceClaim, false);
assert.ok(M.VARIANTS.some(variant => variant.id === 'linear' && variant.label === 'GDN / KDA'));

// Heads: 4 Q heads stay independent while the number of KV groups changes
// 4 / 2 / 1. GQA keeps both Q projections; only K/V projections are grouped.
const mha = M.headsExample('mha');
const gqa = M.headsExample('gqa');
const mqa = M.headsExample('mqa');
assert.equal(mha.qHeads, 4);
assert.equal(gqa.qHeads, 4);
assert.equal(mqa.qHeads, 4);
assert.equal(mha.kvGroups, 4);
assert.equal(gqa.kvGroups, 2);
assert.equal(mqa.kvGroups, 1);
assert.deepEqual(mha.groups, [[0], [1], [2], [3]]);
assert.deepEqual(gqa.groups, [[0, 1], [2, 3]]);
assert.deepEqual(mqa.groups, [[0, 1, 2, 3]]);
assert.deepEqual(gqa.groupOf, [0, 0, 1, 1]);
assert.equal(gqa.qProjectionsIndependent, true);
assert.notDeepEqual(gqa.heads[0].q, gqa.heads[1].q, 'same GQA group keeps independent Q projections');
assert.notDeepEqual(gqa.heads[0].weights, gqa.heads[1].weights, 'same GQA group keeps independent Q weights');
assert.deepEqual(gqa.heads[0].k, gqa.heads[1].k, 'same GQA group shares K');
assert.deepEqual(gqa.heads[0].v, gqa.heads[1].v, 'same GQA group shares V');

// MLA: the latent expansion gives K=[[1,4],[3,2]] and V=[[1,3],[3,4]];
// both the absorbed and expanded paths produce [2.608859365, 3.804429683].
const mla = M.mlaExample();
matrix(mla.k, [[1, 4], [3, 2]], 'MLA K = U_K c');
matrix(mla.v, [[1, 3], [3, 4]], 'MLA V = U_V c');
assert.ok(Math.abs(mla.output[0] - 2.608859365) < 1e-9, 'MLA output 0');
assert.ok(Math.abs(mla.output[1] - 3.804429683) < 1e-9, 'MLA output 1');
assert.ok(Math.abs(mla.outputExpanded[0] - 2.608859365) < 1e-9, 'MLA expanded output 0');
assert.ok(Math.abs(mla.outputExpanded[1] - 3.804429683) < 1e-9, 'MLA expanded output 1');assert.equal(mla.positionCacheSeparate, true);
assert.equal(mla.ropeAbsorbed, false);

// SWA: t=6, W=3 sees t4..t6 only; earlier positions are outside, later ones
// are future, and neither enters this softmax.
const swa = M.swaExample(6, 3);
assert.deepEqual(swa.visible, [4, 5, 6]);
assert.deepEqual(swa.outside, [1, 2, 3]);
assert.deepEqual(swa.future, [7, 8]);
assert.equal(swa.start, 4);
assert.equal(swa.count, 3);
assert.equal(swa.includesCurrent, true);
const earlySwa = M.swaExample(3, 3);
assert.deepEqual(earlySwa.visible, [1, 2, 3]);
assert.deepEqual(earlySwa.outside, []);
assert.deepEqual(earlySwa.future, [4, 5, 6, 7, 8]);

// DSA: Top-K=2 selects positions 2 and 4; index scores are not final weights.
// Main logits ln3 and 0 give weights .75 / .25 and output [1.5, 1].
const dsa = M.dsaExample(2);
assert.deepEqual(dsa.indexScores, [1, 3, 0, 2]);
assert.deepEqual(dsa.ranking, [1, 3, 0, 2]);
assert.deepEqual(dsa.selected, [1, 3]);
assert.deepEqual(dsa.selectedPositions, [2, 4]);
near(dsa.weights[0], 0.75, 'DSA weight 0');
near(dsa.weights[1], 0.25, 'DSA weight 1');
near(dsa.output[0], 1.5, 'DSA output 0');
near(dsa.output[1], 1, 'DSA output 1');
assert.equal(dsa.indexScoresAreFinalWeights, false);
assert.equal(dsa.deletesUnselected, false);
assert.ok(M.VARIANTS.some(variant => variant.id === 'heads' && variant.label === 'MHA / GQA / MQA'));
assert.ok(M.VARIANTS.some(variant => variant.id === 'mla' && variant.label === 'MLA'));
assert.ok(M.VARIANTS.some(variant => variant.id === 'swa' && variant.label === 'SWA'));
assert.ok(M.VARIANTS.some(variant => variant.id === 'dsa' && variant.label === 'DSA'));

// NSA: 8 visible tokens, four complete 2-token blocks. The two Q-head
// summary weights are softmax rows; their GQA group sum is a selection score,
// not a probability, and Top-n picks whole blocks of raw tokens.
const sum = values => values.reduce((total, value) => total + value, 0);
const nsa = M.nsaExample();
assert.equal(nsa.kind, 'nsa');
assert.equal(nsa.topN, 1);
assert.equal(nsa.visibleTokens, 8);
assert.equal(nsa.blockSize, 2);
assert.equal(nsa.selectionBlockSize, 2);
assert.equal(nsa.stride, 2);
assert.equal(nsa.blockCount, 4);
assert.equal(nsa.completeBlocks, 4);
assert.equal(nsa.blocks.length, 4);
assert.equal(nsa.summaries.length, 4);
assert.equal(nsa.compressor.type, 'learnable-mlp');
assert.equal(nsa.compressor.usesIntraBlockPosition, true);
assert.equal(nsa.compressor.fixedAverage, false);
assert.equal(nsa.compressor.summaryValuesAreTeachingInputs, true);
assert.equal(nsa.branchesUseIndependentKV, true);
assert.equal(nsa.branchesHaveIndependentSoftmax, true);
assert.equal(nsa.groupScoresAreProbabilities, false);
assert.deepEqual(nsa.headWeightSums, [1, 1]);
nsa.headBlockWeights.forEach((row, index) => near(sum(row), 1, `NSA head row ${index} softmax`));
assert.deepEqual(nsa.groupBlockWeights, [0.5, 0.4, 0.9, 0.2]);
near(nsa.groupWeightSum, 2, 'NSA group sum');
assert.deepEqual(nsa.ranking, [2, 0, 1, 3]);
assert.deepEqual(nsa.selectedByRank, [2]);
assert.deepEqual(nsa.selectedBlocks, [2]);
assert.deepEqual(nsa.selectedPositions, [5, 6]);
assert.deepEqual(nsa.selectedTokens.map(token => token.position), [5, 6]);
assert.deepEqual(nsa.selection.positions, [5, 6]);
near(nsa.selection.softmaxSum, 1, 'NSA selection softmax');
assert.deepEqual(nsa.window.positions, [7, 8]);
near(nsa.window.softmaxSum, 1, 'NSA window softmax');
assert.equal(nsa.window.w, 2);
// The gate example is an independent check; its branch outputs are not derived
// from the block weights above.
assert.deepEqual(nsa.gates, [0.5, 0.75, 0.25]);
near(nsa.gateSum, 1.5, 'NSA gate sum');
assert.deepEqual(nsa.branchOutputs, { compressed: [2, 0], selected: [0, 4], window: [2, 2] });
assert.deepEqual(nsa.output, [1.5, 3.5]);
assert.equal(nsa.gatesAreNormalized, false);
assert.equal(nsa.branchOutputsDerivedFromBlockWeights, false);
assert.equal(nsa.isPerformanceClaim, false);

// Deliberately normalize the gates and confirm that it changes the result;
// the implementation must not do this silently.
const normalizedGates = nsa.gates.map(value => value / nsa.gateSum);
const normalizedOutput = [0, 1].map(coordinate => (
  normalizedGates[0] * nsa.branchOutputs.compressed[coordinate]
  + normalizedGates[1] * nsa.branchOutputs.selected[coordinate]
  + normalizedGates[2] * nsa.branchOutputs.window[coordinate]
));
near(normalizedOutput[0], 1, 'normalized gate output 1');
near(normalizedOutput[1], 7 / 3, 'normalized gate output 2');
assert.notDeepEqual(normalizedOutput, nsa.output);

// Top-n changes selection and the fine-grained softmax, but not the
// independent gate example.
const nsa2 = M.nsaExample({ topN: 2 });
assert.deepEqual(nsa2.ranking, [2, 0, 1, 3]);
assert.deepEqual(nsa2.selectedByRank, [2, 0]);
assert.deepEqual(nsa2.selectedBlocks, [0, 2]);
assert.deepEqual(nsa2.selectedPositions, [1, 2, 5, 6]);
assert.deepEqual(nsa2.selectedTokens.map(token => token.position), [1, 2, 5, 6]);
assert.notDeepEqual(nsa2.selection.weights, nsa.selection.weights);
near(nsa2.selection.softmaxSum, 1, 'NSA Top-2 selection softmax');
assert.deepEqual(nsa2.gates, [0.5, 0.75, 0.25]);
assert.deepEqual(nsa2.output, [1.5, 3.5]);
assert.equal(M.nsaExample({ topN: 0 }).topN, 1);
assert.equal(M.nsaExample({ topN: 99 }).topN, 4);
assert.deepEqual(M.nsaExample({ topN: 99 }).selectedPositions, [1, 2, 3, 4, 5, 6, 7, 8]);

// Helpers are deterministic and clamp inputs without normalizing gates.
const blockSelection = M.nsaBlockSelection(1);
assert.deepEqual(blockSelection.groupScores, [0.5, 0.4, 0.9, 0.2]);
assert.deepEqual(blockSelection.selectedPositions, [5, 6]);
const gate = M.nsaGateMerge([0.5, 0.75, 0.25]);
assert.deepEqual(gate.output, [1.5, 3.5]);
assert.equal(gate.normalized, false);
assert.equal(gate.independentSigmoid, true);
assert.deepEqual(M.nsaGateMerge([0.5, 0.5, 0.5]).output, [2, 3]);
assert.deepEqual(M.nsaGateMerge([2, -1, 0.25]).gates, [1, 0, 0.25]);
assert.deepEqual(M.nsaGateMerge([2, -1, 0.25]).output, [2.5, 0.5]);

assert.ok(M.VARIANTS.some(variant => variant.id === 'nsa' && variant.label === 'NSA'));
assert.ok(M.VARIANTS.some(variant => variant.id === 'linear' && variant.label === 'GDN / KDA'));

console.log('PASS: attention variants — linear recurrence values plus NSA block selection, GQA sum, Top-n, independent gates, and non-normalized output.');
