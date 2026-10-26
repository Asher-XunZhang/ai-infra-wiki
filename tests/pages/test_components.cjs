const assert=require('node:assert/strict');
const M=require('../../pages/foundations/components-model.js');
const near=(a,b,eps=1e-6,label='')=>assert.ok(Math.abs(a-b)<=eps,`${label} ${a} != ${b}`);

// RoPE: primary layer-facts example, then the independent -1 invariant.
const rope=M.rope(1,0);
near(rope.dot,Math.SQRT1_2,1e-9,'rope dot');
near(M.rope(4,3).dot,Math.SQRT1_2,1e-9,'common +3 keeps dot');
near(M.rope(3,1).dot,0,1e-9,'relative distance 2 gives 0');
for(const p of [1,2,3])near(M.ropeIndependent(p-1).dot,-1,1e-9,'independent rope -1');
near(M.rope(1,0).qLength,1,1e-9,'rotation keeps length');

// RMSNorm: no mean subtraction, gamma changes direction.
const rms=M.rmsnorm([3,4],[1,2]);
near(rms.rms,3.535534047354091,1e-9,'rms');
near(rms.normalized[0],0.8485281034827336,1e-9,'norm0');
near(rms.normalized[1],1.1313708046436448,1e-9,'norm1');
near(rms.output[0],0.8485281034827336,1e-9,'gamma0');
near(rms.output[1],2.2627416092872896,1e-9,'gamma1');
assert.equal(rms.mean,3.5);
assert.equal(rms.directionPreserved,false);
assert.equal(M.rmsnorm([6,8],[1,2]).directionPreserved,false);
near(M.rmsnorm([6,8],[1,2]).output[1],2.2627416092872896,1e-6,'scale invariance (epsilon is negligible)');
assert.equal(M.rmsnorm([3,4],[1,1]).directionPreserved,true);

// Residual: element-wise, never concatenation.
assert.deepEqual(M.residual([1,2],[.5,-1]).sum,[1.5,1]);
assert.deepEqual(M.residual([1,2],[0,0]).sum,[1,2]);
assert.deepEqual(M.residual([1,2],[.5,-1]).concatenated,[1,2,.5,-1]);
assert.equal(M.residual([1,2],[.5,-1]).isConcatenation,false);

// SwiGLU: SiLU can be negative, product keeps signs, no routing.
const glu=M.swiglu([1,2]);
near(glu.g[0],1);near(glu.g[1],-1);
near(glu.silu[0],0.7310585786300049,1e-9);
near(glu.silu[1],-0.2689414213699951,1e-9);
near(glu.z[0],1.4621171572600098,1e-9);
near(glu.z[1],-0.8068242641099853,1e-9);
assert.equal(glu.hasNegativeSilu,true);
assert.equal(glu.siluRangeIsUnit,false);
assert.equal(glu.isExpertRouting,false);
assert.deepEqual(M.swiglu([1,2]).u,[2,3]);

// K/V head grouping and the 48/24/12 scalar counts.
for(const [kv,count] of [[4,48],[2,24],[1,12]])assert.equal(M.kvHeads(kv).scalars,count);
assert.deepEqual(M.kvHeads(4).mapping.map(m=>m.group),[0,1,2,3]);
assert.deepEqual(M.kvHeads(2).mapping.map(m=>m.group),[0,0,1,1]);
assert.deepEqual(M.kvHeads(1).mapping.map(m=>m.group),[0,0,0,0]);
assert.equal(M.kvHeads(2).qIndependent,true);

// Attention motive: self-only versus allowed context, same Q/K/V.
const self=M.contextMotive('self'),ctx=M.contextMotive('context');
assert.deepEqual(self.weights,[0,1]);assert.deepEqual(self.result,[0,2]);
near(ctx.weights[0],0.3302384506733431,1e-12);near(ctx.weights[1],0.6697615493266569,1e-12);
near(ctx.result[0],0.6604769013466862,1e-12);near(ctx.result[1],1.3395230986533138,1e-12);
near(ctx.weights[0]+ctx.weights[1],1,1e-12);
assert.equal(ctx.isPerformanceClaim,false);assert.equal(ctx.isAccuracyClaim,false);

// Block structure, not a token generator.
assert.deepEqual(M.block.order,['x','RMSNorm','Attention','+x','RMSNorm','SwiGLU MLP','+x₁','y']);
assert.deepEqual(M.block.residualInputs,['x','x₁']);
assert.equal(M.block.residuals,2);assert.equal(M.block.oneLayerIsOneToken,false);
console.log('PASS: component model — RoPE invariants, RMSNorm/gamma, residual sum, SiLU product, KV counts, motive weights, block structure.');
