const assert=require('node:assert/strict');
const F=require('../../pages/foundations/model.js'),C=require('../../pages/sglang/communication/model.js'),G=require('../../pages/sglang/advanced-generation/model.js'),R=require('../../pages/sglang/request-runtime/model.js'),O=require('../../pages/sglang/serving-operations/model.js');
const lessons=require('../../pages/motion/detail-scenes.js').create(F,C,G,R,O);
assert.equal(Object.keys(lessons).length,9);
for(const key of ['f-speculation','advanced-generation'])for(const s of lessons[key]){
 const d=s.data;assert.equal(d.predictions.length,d.draft.length+1);
 const result=G.verifyChain(d.draft,d.predictions);assert.equal(result.accepted,d.accepted);assert.deepEqual(result.output,d.output);assert.deepEqual(d.validKV,[d.root,...d.draft.slice(0,d.accepted)]);assert.notDeepEqual(d.output,d.validKV);
}
const accidentalMatch=lessons['f-speculation'][0].data;assert.equal(accidentalMatch.predictions[3],accidentalMatch.draft[3]);assert.equal(G.verifyChain(accidentalMatch.draft,accidentalMatch.predictions).accepted,2,'a match after rejection cannot restart acceptance');
for(const s of lessons.communication){const d=s.data;assert.equal(d.blocks.reduce((n,b)=>n+b.pages*d.bytesPerPage,0),48);assert.deepEqual([...d.dst].sort((a,b)=>a-b),[3,4,9]);}
for(const s of lessons['request-runtime']){assert.equal(s.data.kv,6);assert.equal(s.data.released.slot,false);assert.equal(s.data.released.cached,s.data.cache?6:0);assert.equal(s.data.released.kv,0);}
for(const s of lessons['serving-operations']){assert.equal(s.data.states[2].queued,true);assert.equal(s.data.states[3].frontend,true);assert.equal(s.data.states[4].frontend,false);}
for(const s of lessons['f-hardware'])assert.deepEqual(s.data.weights.map(r=>r.reduce((n,v,i)=>n+v*s.data.input[i],0)),s.data.output);
for(const s of lessons['f-hardware'].filter(s=>s.kind==='hardware-roofline')){
 assert.equal(s.data.work[0]/s.data.rates[0],s.data.move);
 assert.equal(s.data.work[1]/s.data.rates[1],s.data.compute);
 assert.equal(Math.max(s.data.move,s.data.compute),s.data.bound);
}
for(const s of lessons['f-hardware'].filter(s=>s.kind==='hardware-graph')){
 const d=s.data.schedule;
 for(let i=0;i<3;i++){
  assert(d.gpu[i].start>=d.cpu[i].end,'kernel must be submitted before execution');
  assert.equal(d.gpu[i].end-d.gpu[i].start,d.graphGPU[i].end-d.graphGPU[i].start,'graph does not shorten kernel arithmetic');
  if(i)assert(d.gpu[i].start>=d.gpu[i-1].end&&d.graphGPU[i].start>=d.graphGPU[i-1].end,'both retain dependency order');
 }
 assert(d.cpu[1].start<d.gpu[0].end,'ordinary asynchronous launches overlap GPU work');
 assert.equal(d.duration-d.graphGPU[2].end,d.kernelDuration===1?2:0,'launch-bound benefit and covered-launch counterexample');
}
console.log('PASS 9 detailed lessons: model parity, first mismatch, output/KV offset, page coverage and lifecycle boundaries.');
