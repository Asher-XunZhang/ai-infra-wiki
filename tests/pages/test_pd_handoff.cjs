const assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const M=require('../../pages/sglang/inference-overview/pd-handoff-model.js');
const overlap=(a,b)=>Math.max(a.start,b.start)<Math.min(a.end,b.end);
for(const option of Object.keys(M.cases)){
 const tl=M.build(option);assert.ok(Number.isFinite(tl.duration));
 for(const e of tl.events){
  assert.ok(e.end>e.start&&e.start>=0&&e.end<=tl.duration);
  for(const dep of e.depends)assert.ok(tl.byId[dep].end<=e.start,`${option}: ${dep} precedes ${e.id}`);
  for(const ref of e.refs)assert.ok(M.sources[ref],ref);
 }
 // Resource lanes represent one selected request; active operations may not double-book the lane.
 for(const [lane] of M.lanes){const events=tl.events.filter(e=>e.lane===lane);for(let i=0;i<events.length;i++)for(let j=i+1;j<events.length;j++)assert.ok(!overlap(events[i],events[j]),`${option}: ${lane} overlap`);}
 for(let t=0;t<=tl.duration;t+=.137){
  const s=M.state(tl,t);assert.deepEqual(s,M.state(tl,t),'deterministic reverse seek');
  assert.ok(s.dKV<=s.pKV);assert.ok(!s.dKV||s.reserved);
  if(s.committed)assert.ok(s.rawSuccess&&s.metadataVisible&&s.dKV===4);
  if(s.decodeKV)assert.ok(s.committed);
  if(s.released)assert.ok(t>=tl.byId.post.end,'sender cannot clear before outstanding work drained');
  if(s.dOutput.length===2)assert.ok(s.decodeKV);
 }
 const end=M.state(tl,tl.duration);
 if(option==='capacity'){
  assert.equal(end.reserved,false);assert.equal(end.pKV,0);assert.equal(end.pQueue,'bootstrap_queue');assert.equal(end.dQueue,'prealloc_queue');assert.equal(tl.events.filter(e=>e.kind==='compute').length,0);
 }else{
  assert.ok(overlap(tl.byId.compute2,tl.byId.copy1),'copy completed prefix while GPU computes suffix');
  assert.ok(tl.byId.aux.start>=tl.byId.copy2.end,'last chunk KV before aux');
  assert.ok(tl.byId.compute1.start>=tl.byId.destination.end,'optimistic=0 bootstrap gate');
  assert.ok(tl.byId.release.depends.includes('post')&&!tl.byId.release.depends.includes('notify'),'P release waits for local drain, not remote notification observation');
  assert.equal(M.state(tl,4).dQueue,'transfer_queue','D sender call can return while control message is in flight');
  assert.deepEqual(end.dOutput,['y1','y2']);assert.equal(end.dKV,4);assert.equal(end.decodeKV,true);
 }
}
const delayed=M.build('metadata'),s=M.state(delayed,25);
assert.equal(s.rawSuccess,true);assert.equal(s.metadataVisible,false);assert.equal(s.committed,false);assert.equal(s.dQueue,'transfer_queue');
const late=M.build('latepoll'),mid=M.state(late,28);
assert.equal(mid.released,false);assert.equal(mid.committed,true);assert.ok(mid.active.some(e=>e.id==='decode'),'D does not wait for P resource cleanup');
assert.deepEqual(M.state(M.build(),0).dOutput,[]);
if(process.env.SGLANG_SOURCE_DIR){
 const cache={};
 for(const [key,[file,line,needle]] of Object.entries(M.sources)){
  cache[file]??=execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',`${M.revision}:python/sglang/srt/${file}`],{encoding:'utf8'}).split('\n');
  assert.ok(cache[file][line-1].includes(needle),key+' source anchor');
 }
 console.log(`PASS: ${Object.keys(M.sources).length} fixed-source anchors.`);
}
console.log('PASS: PD allocation/readiness gates, chunk overlap, serial worker dependencies, independent cleanup, KV/token boundaries and reversible state in four conditions.');
