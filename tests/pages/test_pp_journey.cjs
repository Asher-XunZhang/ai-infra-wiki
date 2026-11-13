const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const M=require('../../pages/sglang/pd-prefill-pp-loop/journey-model.js');
const dir=path.join(__dirname,'../../pages/sglang/pd-prefill-pp-loop/scenarios');
for(const file of fs.readdirSync(dir).filter(f=>f.endsWith('.json'))){
 const model=M.compile(JSON.parse(fs.readFileSync(path.join(dir,file),'utf8')));
 for(const rank of [0,1,2]){
  const source=model.events.filter(e=>e.r===rank&&['cpu','wait'].includes(e.kind)).sort((a,b)=>a.start-b.start),groups=M.cpuGroups(model,rank);
  assert.deepEqual(groups.flatMap(g=>g.members),source,'grouping preserves every CPU and wait event, once and in order');
  assert(groups.length<source.length,'CPU detail is actually condensed');
  for(const group of groups){
   assert.equal(group.start,group.members[0].start);assert.equal(group.end,group.members.at(-1).end);
   for(let i=1;i<group.members.length;i++){assert.equal(group.members[i].start,group.members[i-1].end,'no idle gap hidden');assert.equal(group.members[i].n,group.members[0].n,'no cross-loop merging');assert.equal(group.members[i].kind,'cpu');}
   if(group.kind==='wait')assert.equal(group.members.length,1,'wait boundaries remain independent');
  }
 }
 for(const time of [0,18,24,48,model.end]){const w=M.processWindow(model,time);assert(w.start<=time&&w.end>=time&&w.end-w.start<=24.00001);}
 assert.equal(model.packet.model.sourceCommit,'279339f113b79af84f27fd3ac92d0a13bd3f4cbd');
 for(const batch of model.batches){
  for(const r of batch.ranks){
   const gpu=r.phases[2],kv=r.phases[4],clean=r.phases[5];
   assert.equal(M.rankState(model,batch.batch,r.r,gpu.start).phase,2);
   assert.equal(M.rankState(model,batch.batch,r.r,kv.end).complete,false,'transfer completion is not cleanup');
   assert.equal(M.rankState(model,batch.batch,r.r,clean.end).complete,true);
   assert.equal(M.rankState(model,batch.batch,r.r,kv.start).phase,4,'reverse seek restores transit');
   assert(kv.start>=gpu.end);
  }
  for(const transfer of M.transfers(model,batch.batch)){
   assert.equal(M.active(transfer,transfer.start-.001),false);
   assert.equal(M.active(transfer,transfer.end),false);
   if(transfer.end>transfer.start)assert(M.active(transfer,(transfer.start+transfer.end)/2));
   if(transfer.payload==='hidden states'){
    const consumer=model.nodes.find(n=>n.r===transfer.to&&n.kind==='gpu'&&n.owner===transfer.owner);
    assert(consumer.start>=transfer.end,'activation must arrive before downstream compute');
   }
  }
 }
 for(const e of model.events.filter(e=>e.kind==='wait'&&e.end>e.start)){
  const time=(e.start+e.end)/2,relations=M.crossRelations(model,time,3);
  for(const id of e.deps||[]){const p=model.graph[id];if(p.r!==e.r)assert(relations.some(r=>r.source.id===id&&r.target.id===e.target),'every active cross-rank wait is explained');}
  for(const r of relations){assert(r.target.deps.includes(r.source.id));assert(r.source.r!==r.target.r);assert(r.target.start>=r.source.end-1e-6);if(r.transport)assert.equal(r.source.kind,'message');}
 }
 const id='0:5:recv_out',d=M.dependencies(model,id,0),last=d.latest;
 assert(!M.dependencies(model,id,last-.001).ready);
 assert(M.dependencies(model,id,last+.001).ready);
 assert(d.n.start>=last-.001);
 const coarse=M.dependencies(model,'0:4:gpu',0).deps;
 assert(coarse.some(p=>p.type==='教学简化：等待全部 H2D'));
 const overlaps=model.nodes.filter(e=>e.kind==='gpu').some(g=>model.events.some(e=>e.kind==='cpu'&&e.r===g.r&&e.owner!==g.owner&&e.start<g.end&&e.end>g.start));
 assert(overlaps,'real teaching schedule retains GPU and other CPU work overlap');
}
console.log('PASS: 10 scenarios, source baseline, transfer gates, cleanup boundary, reversible state and parallel activity.');
