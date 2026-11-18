const assert=require('node:assert/strict');
const M=require('../../pages/foundations/output-chain-model.js');
const near=(a,b,eps=1e-6,label='')=>assert.ok(Math.abs(a-b)<=eps,`${label?label+': ':''}${a} != ${b}`);

const d=M.chain('with-france');
assert.deepEqual(d.tokens,['法国','的','首都','是','哪里','？']);
assert.deepEqual(d.rawScores,[2,0,1,0,1,0]);
near(d.weights.reduce((a,b)=>a+b,0),1,1e-12,'weights sum to 1');
near(d.weights[0],0.466905,1e-6,'法国 weight is the largest');
near(d.attentionMaxScore,2,1e-12,'attention softmax records its stabilising max');
near(d.denominator,d.exps.reduce((a,b)=>a+b,0),1e-12,'attention denominator is Σexp(s-max)');
for(const r of d.rows)near(r.weight,r.rawScore===d.attentionMaxScore?Math.exp(0)/d.denominator:Math.exp(r.rawScore-d.attentionMaxScore)/d.denominator,1e-12,'published weight uses stabilised softmax');
near(d.attention[0],1.158884,1e-6,'attention pooled x');
near(d.attention[1],0.427833,1e-6,'attention pooled y');
near(d.x1[0],d.x6[0]+d.attention[0],1e-12,'first residual keeps x6');
near(d.x1[1],d.x6[1]+d.attention[1],1e-12,'first residual keeps x6');
near(d.h[0],d.x1[0]/d.rms,1e-12,'RMSNorm divides both dimensions');
near(d.h[1],d.x1[1]/d.rms,1e-12,'RMSNorm divides both dimensions');
near(d.g[0],1.763074,1e-6,'gate 0');
near(d.g[1],-0.863779,1e-6,'gate 1');
near(d.g[2],0.890416,1e-6,'gate 2');
near(d.u[2],-1.113110,1e-6,'value lane keeps sign');
near(d.a[0],1.504950,1e-6,'SiLU coefficient > 1');
near(d.a[1],-0.256154,1e-6,'SiLU coefficient < 0');
near(d.z[2],-0.702688,1e-6,'a_i * u_i');
for(let j=0;j<3;j++)near(d.z[j],d.a[j]*d.u[j],1e-12,'elementwise gate product');
for(let i=0;i<2;i++){
 const combined=d.z.reduce((s,z,j)=>s+z*d.WD[j][i],0);
 near(d.dM[i],combined,1e-12,'down projection combines all hidden lanes');
 near(d.y[i],d.x1[i]+d.dM[i],1e-12,'second residual keeps x1');
}
assert.equal(d.ffnContributions.length,3,'one contribution vector per hidden lane');
for(let j=0;j<3;j++){
 assert.equal(d.ffnContributions[j].length,2,'each lane contributes exactly two dimensions');
 for(let i=0;i<2;i++)near(d.ffnContributions[j][i],d.z[j]*d.WD[j][i],1e-12,'z_j times its own Wd row');
}
for(let i=0;i<2;i++)near(d.dM[i],d.ffnContributions.reduce((s,v)=>s+v[i],0),1e-12,'the three 2-D contributions sum to dM');
near(d.y[0],2.597358,1e-6,'layer-end y0');
near(d.y[1],0.511697,1e-6,'layer-end y1');
assert.deepEqual(d.head.map(c=>c.token),['巴黎','里昂','北京','马赛']);
for(const c of d.head)near(c.logit,M.dot(d.y,c.vector),1e-12,'LM Head is a dot product');
near(d.probabilities.reduce((a,b)=>a+b,0),1,1e-12,'head probabilities sum to 1');
near(d.headMaxLogit,Math.max(...d.head.map(c=>c.logit)),1e-12,'head softmax records its stabilising max');
near(d.softmaxDenominator,d.head.reduce((a,c)=>a+Math.exp(c.logit-d.headMaxLogit),0),1e-12,'head denominator is Σexp(s-max)');
for(let i=0;i<d.head.length;i++)near(d.headExps[i],Math.exp(d.head[i].logit-d.headMaxLogit),1e-12,'head numerator is exp(logit-max)');
near(d.head[0].logit,2.0306,1e-4,'巴黎 logit with France context');
near(d.probabilities[0],0.6588,1e-4,'巴黎 probability with France context');
assert.equal(d.selected,'巴黎');
assert.equal(d.teachingOnly,true);
assert.equal(d.realModelObservation,false);

const off=M.chain('without-france');
assert.deepEqual(off.positions,['p2','p3','p4','p5','p6']);
assert.deepEqual(off.WG,d.WG,'the same FFN weights are reused');
assert.deepEqual(off.head.map(c=>c.vector),d.head.map(c=>c.vector),'LM Head weights stay fixed');
near(off.weights.reduce((a,b)=>a+b,0),1,1e-12,'remaining positions are renormalised after p1 is removed');
assert.equal(off.weights[0],off.weights[4],'remaining zero-score positions share the same weight');
assert.ok(off.weights[1]>off.weights[0],'首都 keeps the larger weight in the reduced context');
near(off.attention[1],0.627377,1e-6,'removed context changes attention output');
near(off.y[0],1.556666,1e-6,'removed context changes layer-end representation');
assert.ok(off.head[0].logit<d.head[0].logit,'巴黎 score drops without the 法国 input');
assert.equal(off.selected,'里昂');
assert.equal(off.teachingOnly,true);
assert.equal(off.realModelObservation,false);

console.log('PASS: output chain — attention weights, residual/RMSNorm, SwiGLU vector combination, fixed LM Head, logits/probabilities, greedy selection, and a full re-computation after removing 法国.');
