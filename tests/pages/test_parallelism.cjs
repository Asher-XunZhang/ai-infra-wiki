const assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const M=require('../../pages/sglang/parallelism/model.js');
for(const kind of ['column','row'])for(const gather of [false,true]){
 const d=M.tensor(kind,gather);assert.deepEqual(d.full,[10,30]);
 assert.deepEqual(M.matvec(d.W,d.X),d.full);
 if(kind==='row'){assert.deepEqual(d.local,[[3,5],[7,25]]);assert.notDeepEqual(d.local.flat(),d.full);}
 else assert.deepEqual(d.local,[[10],[30]]);
 assert.deepEqual(d.rankOutputs,kind==='column'&&!gather?[[10],[30]]:[[10,30],[10,30]]);
}
for(const n of [1,3]){
 const frames=M.pipeline(n),visits=new Map();
 for(const f of frames){
  const running=f.stages.filter(x=>x!==null);assert.equal(new Set(running).size,running.length,'one microbatch cannot occupy both stages in a tick');
  f.stages.forEach((batch,stage)=>{if(batch!==null){const key=`${batch}/${stage}`;assert.ok(!visits.has(key));visits.set(key,f.tick);}});
 }
 for(let batch=1;batch<=n;batch++)assert.equal(visits.get(`${batch}/1`),visits.get(`${batch}/0`)+1,'downstream follows upstream');
 assert.deepEqual(frames[1].stages,[1,null]);assert.deepEqual(frames.at(-1).stages,[null,n]);assert.equal(visits.size,n*2);
}
for(const f of M.dataParallel()){
 assert.deepEqual(f.requests.flat().sort(),Array.from({length:f.step},(_,i)=>i+1));
 assert.equal(new Set(f.requests.flat()).size,f.step,'each request has exactly one replica');
}
for(const skew of [false,true])for(let t=1;t<=4;t++){
 const d=M.experts(skew,t);assert.equal(d.loads.reduce((a,b)=>a+b,0),8);
 assert.equal(new Set(d.selected).size,2);assert.equal(d.weights.reduce((a,b)=>a+b,0),1);
 assert.equal(d.result,.75*t*(d.selected[0]+1)+.25*t*(d.selected[1]+1));
 assert.deepEqual(d.rankLoads,skew?[6,2]:[4,4]);
}
assert.equal(M.experts(false,1).result,1.5);
for(const zigzag of [false,true]){
 const c=M.context(zigzag);assert.deepEqual(c.ranks.flat().sort(),[0,1,2,3,4,5,6,7]);
 assert.deepEqual(c.loads,zigzag?[18,18]:[10,26]);assert.equal(c.loads.reduce((a,b)=>a+b,0),36);
}
assert.deepEqual(M.groups(2,2),{tp:[[0,1],[2,3]],pp:[[0,2],[1,3]]});
assert.throws(()=>M.groups(0,2));assert.throws(()=>M.pipeline(2));assert.throws(()=>M.experts(false,0));
if(process.env.SGLANG_SOURCE_DIR){
 for(const [key,[file,line,symbol]] of Object.entries(M.sources)){
  const code=execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',`${M.revision}:python/sglang/srt/${file}`],{encoding:'utf8'});
  assert.match(code.split('\n')[line-1],new RegExp(`def ${symbol}\\(`),key);
 }
 console.log(`PASS: ${Object.keys(M.sources).length} fixed source anchors.`);
}
console.log('PASS: TP arithmetic and shard semantics, PP causality, DP ownership, EP conservation, CP causal work and rank groups.');

// The implementation lesson has its own continuous, source-pinned timeline.
// Keep the older arithmetic API: other teaching pages still consume it.
const F=require('../../pages/sglang/parallelism/flow-model.js');
assert.deepEqual(F.groups(4,2,2,1,2),{world:8,attnTP:2,moeTP:2,tp:[[0,1,2,3],[4,5,6,7]],pp:[[0,4],[1,5],[2,6],[3,7]],attn:[[0,1],[2,3],[4,5],[6,7]],cp:[[0],[1],[2],[3],[4],[5],[6],[7]]});
assert.deepEqual(F.groups(4,1,1,2).cp,[[0,2],[1,3]]);
assert.throws(()=>F.groups(3,1,2));
assert.deepEqual(F.dispatch(3),[0,1,0]);assert.deepEqual(F.dispatch(3,[false,true]),[1,1,1]);assert.throws(()=>F.dispatch(1,[false,false]));
assert.deepEqual(F.sequenceRows(),[[0,1,2],[3,4,null]]);assert.deepEqual(F.sequenceRows().flat().filter(x=>x!==null),[0,1,2,3,4]);
assert.deepEqual(F.zigzag(8),[[0,1,6,7],[2,3,4,5]]);assert.equal(F.zigzag(3),null);
for(let n=4;n<=15;n++)assert.deepEqual(F.zigzag(n).flat().sort((a,b)=>a-b),Array.from({length:n},(_,i)=>i));
assert.deepEqual(F.mergeAttention([2,6],[1,3]).weights,[.2,.8]);assert.ok(Math.abs(F.mergeAttention([2,6],[1,3]).output-5.2)<1e-12);
assert.ok(Math.abs(F.mergeAttention([2,6],[1+1000,3+1000]).output-5.2)<1e-12,'stable normalization with large LSE');
assert.ok(Math.abs(F.mergeAttention([2,6],[Math.log(2),Math.log(8)],Math.E).output-5.2)<1e-12,'log base and exponent must match');
for(const mode of F.modes){
 const timeline=F.build(mode),end=F.sample(timeline,timeline.duration);assert.equal(end.finished,true);assert.ok(timeline.duration<26);
 for(const phase of timeline.phases){
  const before=F.sample(timeline,phase.start),middle=F.sample(timeline,phase.start+phase.duration/2),after=F.sample(timeline,phase.end);
  assert.equal(before.progress[phase.id],0);assert.ok(Math.abs(middle.progress[phase.id]-.5)<1e-10);assert.equal(after.progress[phase.id],1);
  for(const later of timeline.phases.slice(phase.index+1))assert.equal(middle.progress[later.id],0,'future work cannot complete early');
  assert.deepEqual(F.sample(timeline,phase.start),before,'seeking backward has no accumulated state');
  for(const ref of phase.refs)assert.ok(F.sources[ref],ref);
 }
}
assert.equal(F.build('sp','decode').phases.some(p=>p.id==='gather'),false);
assert.equal(F.build('pcp','short').phases.some(p=>p.id==='kv'),false);
assert.equal(F.build('sp','decode').data.active,false);
if(process.env.SGLANG_SOURCE_DIR){for(const [key,[file,line,text]] of Object.entries(F.sources)){
 const code=execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',`${F.revision}:python/sglang/srt/${file}`],{encoding:'utf8'});
 assert.ok(code.split('\n')[line-1].includes(text),`${key}: ${file}:${line} must contain ${text}`);
}console.log(`PASS: ${Object.keys(F.sources).length} implementation source anchors.`);}
console.log('PASS: implementation groups, request ownership, CP/SP gates, token conservation, LSE normalization and reversible timelines.');
