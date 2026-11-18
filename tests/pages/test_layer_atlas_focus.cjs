const assert = require('node:assert/strict');
const M = require('../../pages/foundations/layer-atlas-focus-model.js');
const near = (a, b) => assert.ok(Math.abs(a-b) < 1e-10, `${a} != ${b}`);
const sameVector = (a,b) => { assert.equal(a.length,b.length);a.forEach((v,i)=>near(v,b[i])); };

const ffn = M.make('ffn');
assert.equal(ffn.data.h.length,2);
for(const v of [ffn.data.u,ffn.data.g,ffn.data.a,ffn.data.z])assert.equal(v.length,3);
assert.equal(ffn.data.out.length,2);
ffn.data.z.forEach((v,i)=>near(v,ffn.data.a[i]*ffn.data.u[i]));
sameVector(ffn.data.out,[ffn.data.z[0]+0.1*ffn.data.z[1]+0.4*ffn.data.z[2],0.2*ffn.data.z[0]-0.5*ffn.data.z[1]+0.8*ffn.data.z[2]]);
assert.ok(ffn.data.a[1]<0,'SiLU is not a 0/1 selector');
assert.notDeepEqual(ffn.data.out,ffn.data.h,'down projection produces an update, not a reconstruction');

sameVector(M.make('norm',0).data.out,M.make('norm',1).data.out);
near(M.make('norm',1).data.r/M.make('norm',0).data.r,10);
const attention=M.make('attention').data;
near(attention.weights.reduce((a,b)=>a+b,0),1);
attention.out.forEach((v,i)=>near(v,attention.values.reduce((s,row,j)=>s+attention.weights[j]*row[i],0)));
assert.equal(M.softmax([1,2,-Infinity])[2],0,'masked position contributes zero');
const heads=M.make('heads').data;
assert.notDeepEqual(heads.heads[0].weights,heads.heads[1].weights);
sameVector(heads.joined,heads.heads.flatMap(h=>h.out));
assert.equal(heads.out.length,2);

for(const variant of [0,1]){
 const r=M.make('rope',variant).data;
 near(M.dot(r.qr,r.qr),M.dot(r.q,r.q));near(M.dot(r.kr,r.kr),M.dot(r.k,r.k));
 near(M.dot(r.qr,r.kr),r.out[0]);
 sameVector(r.qr,attention.q);
 sameVector(r.kr,attention.keys[r.r]);
 near(r.out[0]/Math.SQRT2,attention.scores[r.r]);
}
assert.notEqual(M.make('rope',0).data.out[0],M.make('rope',1).data.out[0]);
const residual=M.make('residual').data;
sameVector(residual.out,residual.input.map((v,i)=>v+residual.delta[i]));
const layers=M.make('layers').data.states;
assert.equal(layers.length,3);layers.forEach(v=>{assert.equal(v.length,2);assert.ok(v.every(Number.isFinite));});
const head=M.make('head').data;
sameVector(head.scores,head.vectors.map(v=>M.dot(head.input,v)));
const greedy=M.make('generation',0).data,sampled=M.make('generation',1).data;
assert.equal(greedy.selected,greedy.scores.indexOf(Math.max(...greedy.scores)));
const cumulative=sampled.probabilities.slice(0,sampled.selected+1).reduce((a,b)=>a+b,0);
assert.ok(cumulative>=sampled.random&&cumulative-sampled.probabilities[sampled.selected]<sampled.random);
assert.notEqual(sampled.selected,greedy.selected);
sameVector(M.make('embed').data.out,M.make('norm',0,'norm1').data.input);
sameVector(M.make('norm',0,'norm1').data.out,attention.input);
sameVector(heads.heads[0].out,attention.out);
sameVector(heads.out,residual.delta);
sameVector(residual.out,M.make('norm',0,'norm2').data.input);
sameVector(M.make('norm',0,'norm2').data.out,ffn.data.h);
const residualFFN=M.make('residual',0,'add2').data;
sameVector(residualFFN.delta,ffn.data.out);
sameVector(residualFFN.out,layers[0]);
sameVector(layers.at(-1),M.make('norm',0,'finalnorm').data.input);
sameVector(M.make('norm',0,'finalnorm').data.out,head.input);
sameVector(head.scores,greedy.scores);

const run=M.fullRun();
assert.equal(run.selected,'巴黎');
assert.deepEqual(run.chapters.map(c=>c.id),['embed','norm1','attn','add1','norm2','ffn','add2','repeat','finalnorm','head','next']);
for(const transfer of run.phases.filter(p=>p.type==='transfer')){
 const {chapter,target,start,end}=transfer;
 assert.equal(M.sampleRun(run,start).type,'transfer');
 assert.equal(M.sampleRun(run,end).chapter.id,target.id);
 assert.equal(M.sampleRun(run,end).type,'compute');
 const from=M.make(transfer.kind,0,chapter.id).data;
 const to=M.make(M.keyFor(target.id),0,target.id).data;
 sameVector(chapter.value,from.out??from.states?.at(-1)??from.scores);
 if(target.id==='add1'||target.id==='add2')sameVector(chapter.value,to.delta);
 else if(target.id==='ffn')sameVector(chapter.value,to.h);
 else if(target.id==='repeat')sameVector(chapter.value,to.states[0]);
 else if(target.id==='next')sameVector(chapter.value,to.scores);
 else sameVector(chapter.value,to.input);
}
const merge=run.phases.find(p=>p.kind==='heads'&&p.type==='compute');
assert.equal(merge.from,M.make('heads').ends[0],'parallel heads already computed when the merge view begins');
assert.equal(M.sampleRun(run,run.duration).ended,true);
assert.equal(M.sampleRun(run,run.duration).progress,1);

for(const kind of ['embed','ffn','norm','residual','attention','heads','rope','layers','head','generation']){
 const scene=M.make(kind);
 for(const t of [0,scene.duration*.29,scene.duration*.73,scene.duration,scene.duration*.29]){
  const f=M.sample(scene,t);
  scene.ends.forEach((end,i)=>assert.equal(f.complete(i),t>=end));
  assert.ok(f.p>=0&&f.p<=1);
  const second=M.sample(scene,t);near(f.p,second.p);assert.equal(f.phase,second.phase);
 }
 assert.equal(M.sample(scene,-10).time,0);assert.equal(M.sample(scene,Infinity).time,scene.duration);
}
console.log('Layer atlas: projections, gates, normalization, attention, RoPE, residuals, selection and reversible phase boundaries passed.');
