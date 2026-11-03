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
assert.equal(M.rope(1,0).plainDot,1,'unrotated content dot stays 1');
assert.equal(M.rope(1,0).relativeAngleDegrees,45);
assert.equal(M.rope(4,3).plainDot,1,'common shift does not change content dot');
assert.equal(M.rope(1,0).rotationPreservesLength,true);

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
const uniformRms=M.rmsnorm([3,4],[1,1]);
near(uniformRms.outputRms,1,1e-7,'RMS scale is 1 (epsilon makes it approximate)');
near(uniformRms.outputLength,Math.SQRT2,1e-6,'Euclidean length is sqrt2, not 1');
near(M.rmsnorm([6,8],[1,1]).outputLength,Math.SQRT2,1e-6,'2x output length is still sqrt2');
assert.equal(uniformRms.uniformGamma,true);
assert.equal(M.rmsnorm([3,4],[1,2]).uniformGamma,false);

// Residual: element-wise, never concatenation.
assert.deepEqual(M.residual([1,2],[.5,-1]).sum,[1.5,1]);
assert.deepEqual(M.residual([1,2],[0,0]).sum,[1,2]);
assert.equal(M.residual([1,2],[0,0]).zeroUpdate,true);
assert.deepEqual(M.residual([1,2],[0,0]).unchanged,[1,2]);
assert.equal(M.residual([1,2],[.5,-1]).zeroUpdate,false);
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
// Explicit teaching matrices: both branches come from the same x, then SiLU
// gates the value lane elementwise. The same weights are reused for p2.
const sw1=M.swigluExample([1,2]),sw2=M.swigluExample([2,1]);
assert.deepEqual(sw1.Wg,[[1,1],[1,-1]]);
assert.deepEqual(sw1.Wu,[[0,1],[1,1]]);
assert.deepEqual(sw1.Wd,[[1,0],[0,1]]);
assert.deepEqual(sw1.g,[3,-1]);assert.deepEqual(sw1.u,[2,3]);
near(sw1.silu[0],2.8577223804677105,1e-9,'p1 a0 > 1');
near(sw1.silu[1],-0.2689414213699951,1e-9,'p1 a1 < 0');
near(sw1.z[0],5.715444760935421,1e-9);
near(sw1.z[1],-0.8068242641099853,1e-9);
assert.deepEqual(sw1.y,sw1.z,'Wd=I keeps the teaching product readable');
assert.equal(sw1.gateCanExceedOne,true);
assert.equal(sw1.gateCanBeNegative,true);
assert.deepEqual(sw2.Wg,sw1.Wg,'shared Wg across p1/p2');
assert.deepEqual(sw2.Wu,sw1.Wu,'shared Wu across p1/p2');
assert.deepEqual(sw2.Wd,sw1.Wd,'shared Wd across p1/p2');
assert.deepEqual(sw2.g,[3,1]);assert.deepEqual(sw2.u,[1,3]);
assert.equal(sw2.gateCanBeNegative,false,'p2 has no negative gate');
const purpose=M.swigluPurpose([1,2]);
near(purpose.coefficients[0].value,sw1.silu[0],1e-12);
near(purpose.contributions[1],sw1.z[1],1e-12);
assert.equal(purpose.isRouting,false);assert.equal(purpose.isSwitch,false);

// K/V head grouping and the 48/24/12 scalar counts.
for(const [kv,count] of [[4,48],[2,24],[1,12]])assert.equal(M.kvHeads(kv).scalars,count);
assert.deepEqual(M.kvHeads(4).mapping.map(m=>m.group),[0,1,2,3]);
assert.deepEqual(M.kvHeads(2).mapping.map(m=>m.group),[0,0,1,1]);
assert.deepEqual(M.kvHeads(1).mapping.map(m=>m.group),[0,0,0,0]);
assert.equal(M.kvHeads(2).qIndependent,true);
assert.equal(M.kvHeads(2).qProjectionDimensions,8,'every Q head keeps its own projection');
for(const [kv,count] of [[4,96],[2,48],[1,24]])assert.equal(M.kvHeads(kv,{positions:6}).scalars,count);

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

// Block pass: p1 reads only itself, p3 averages p1–p3, and a changed future
// input cannot change an earlier position's result. The layer adds ΔA to x and
// ΔM to x1; the MLP reads Norm(x1), not Norm(x).
const bp=M.blockPass();
const [bp1,bp2,bp3]=bp.positions;
assert.deepEqual(bp1.readIndices,[0]);
near(bp1.weights[0],1,0,'p1 reads only itself');
for(const w of bp3.weights)near(w,1/3,1e-6,'p3 averages the three readable positions');
for(const p of bp.positions)for(let i=0;i<bp.dim;i++){
 near(p.x1[i],p.x[i]+p.attnOut[i],1e-12,'x1 adds x and ΔA');
 near(p.y[i],p.x1[i]+p.dM[i],1e-12,'y adds x1 and ΔM');
}
const future=M.blockPass({x:[[0,2],[1,1],[9,-5]]});
assert.deepEqual(future.positions[0].y,bp1.y,'changing p3 does not change p1 y');
assert.deepEqual(future.positions[1].y,bp2.y,'changing p3 does not change p2 y');
assert.notDeepEqual(future.positions[2].y,bp3.y,'the changed future position itself changes');
for(const p of [bp1,bp2,bp3]){
 const mlp=M.swigluExample(p.normedX1);
 p.dM.forEach((v,i)=>near(v,mlp.y[i],1e-12,'MLP uses Norm(x1)'));
}
assert.ok(Math.abs(bp3.dM[0]-M.swigluExample(bp3.normedX).y[0])>1e-3,
 'Norm(x1) and Norm(x) are not interchangeable here');
const shared=M.swigluExample([1,2]);
assert.deepEqual(bp.teachingMatrices.Wg,shared.Wg,'shared Wg across positions');
assert.deepEqual(bp.teachingMatrices.Wu,shared.Wu,'shared Wu across positions');
assert.deepEqual(bp.teachingMatrices.Wd,shared.Wd,'shared Wd across positions');
assert.equal(bp.mlp,'per-position-shared');
near(bp3.y[0],3.3859880698054132,1e-12,'default p3 y0');
near(bp3.y[1],2.0341358879499616,1e-12,'default p3 y1');

// The no-bypass counterfactual removes both explicit additions and recomputes
// downstream. Its p3 y is not simply the default y minus x.
const off=M.blockPass({residual:false});
const off3=off.positions[2];
for(const p of off.positions)for(let i=0;i<off.dim;i++){
 near(p.x1[i],p.attnOut[i],1e-12,'no-bypass x1 is ΔA alone');
 near(p.y[i],p.dM[i],1e-12,'no-bypass y is ΔM alone');
}
for(let i=0;i<off.dim;i++){
 near(off3.dM[i],M.swigluExample(off3.normedX1).y[i],1e-12,'downstream MLP is recomputed from Norm(ΔA)');
}
near(off3.y[0],1.7615911115294305,1e-12,'no-bypass p3 y0');
near(off3.y[1],0,1e-12,'no-bypass p3 y1');
assert.ok(Math.abs(off3.y[0]-(bp3.y[0]-bp3.x[0]))>1e-3,'no-bypass y0 is not default y minus x');
assert.ok(Math.abs(off3.y[1]-(bp3.y[1]-bp3.x[1]))>1e-3,'no-bypass y1 is not default y minus x');
console.log('PASS: component model — RoPE invariants, RMSNorm/gamma, residual sum, SiLU product, KV counts, motive weights, block structure, block pass causality/default/no-bypass values.');
