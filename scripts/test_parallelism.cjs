const assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const M=require('../pages/sglang/parallelism/model.js');
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
