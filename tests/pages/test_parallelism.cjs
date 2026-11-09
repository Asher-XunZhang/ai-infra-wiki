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
  assert.equal(before.phase.id,phase.id,'narration chooses its own phase, not a global barrier');
  assert.deepEqual(F.sample(timeline,phase.start),before,'seeking backward has no accumulated state');
  for(const ref of phase.refs)assert.ok(F.sources[ref],ref);
 }
}
for(const mode of F.modes)for(const option of mode==='dp'?['default','skip']:mode==='sp'?['default','decode']:mode==='pcp'?['default','short']:['default']){
 const tl=F.build(mode,option),events=tl.execution.events;
 assert.ok(tl.execution.lanes.length>0&&tl.execution.lanes.length<=5);
 for(const lane of tl.execution.lanes){
  for(let i=0;i<lane.events.length;i++){
   const e=lane.events[i];assert.ok(e.start>=0&&e.end>e.start&&e.end<=tl.duration,`${mode}: interval ${e.id}`);
   if(i)assert.ok(e.start>=lane.events[i-1].end-1e-6,`${mode}/${lane.id}: exclusive resource overlap`);
   for(const d of e.depends){assert.ok(events[d],`${e.id}: missing prerequisite ${d}`);assert.ok(events[d].end<=e.start+1e-6,`${e.id} started before ${d} completed`);}
   for(const t of [0,e.start,(e.start+e.end)/2,e.end,tl.duration]){
    const s=F.sample(tl,t);assert.ok(s.progress[e.id]>=0&&s.progress[e.id]<=1);
    if(t<=e.start)assert.equal(s.progress[e.id],0);if(t>=e.end)assert.equal(s.progress[e.id],1);
    assert.equal(s.active.some(x=>x.id===e.id),t>=e.start&&t<e.end);
    assert.deepEqual(F.sample(tl,t),s,'reverse seek is deterministic');
   }
  }
 }
}
const pp=F.build('pp'),pe=pp.execution.events;
assert.ok(F.sample(pp,8).active.some(e=>e.id==='gpu02')&&F.sample(pp,8).active.some(e=>e.id==='gpu11'),'different batches overlap across stages');
for(const [t,cpu,gpu]of [[10.8,'process01','gpu02']]){
 const active=F.sample(pp,t).active.map(e=>e.id);assert.ok(active.includes(cpu)&&active.includes(gpu),'CPU previous result overlaps current GPU');
}
assert.ok(pe.submit11.end<pe.proxy1.end&&pe.gpu11.start>=pe.proxy1.end,'CPU submission may precede device readiness');
assert.ok(pe.d2h11.start>=pe.out2.end,'last stage depth=0 copy stream follows output send dependencies');
assert.ok(pe.proxy2.start>=pe.process01.end,'depth=0 posts current proxy after prior result processing');
assert.ok(pe.relay1.end<=pe.process11.start&&pe.relay2.end<=pe.process12.start,'output ring reaches each stage before processing');
const dp=F.build('dp');assert.ok(dp.data.jobs[0].start<dp.execution.events.route2.end,'no all-dispatched barrier');
assert.notEqual(dp.data.jobs[0].start,dp.data.jobs[1].start,'replicas need no simultaneous start');
const skip=F.build('dp','skip');assert.equal(skip.execution.lanes.find(l=>l.id==='replica0').events.length,0);assert.ok(skip.data.jobs[1].start>=skip.data.jobs[0].finish,'selected single-batch example waits for its own prior work');
const tp=F.build('tp').execution.events;assert.ok(tp.down0.start<tp.gate1.end,'no fabricated barrier between local TP layers');
for(const mode of ['sp','pcp'])assert.equal(F.build(mode,mode==='sp'?'decode':'short').execution.lanes[0].events.some(e=>e.kind==='collective'),false);
assert.equal(F.build('sp','decode').phases.some(p=>p.id==='gather'),false);
assert.equal(F.build('pcp','short').phases.some(p=>p.id==='kv'),false);
assert.equal(F.build('sp','decode').data.active,false);
if(process.env.SGLANG_SOURCE_DIR){for(const [key,[file,line,text]] of Object.entries(F.sources)){
 const code=execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',`${F.revision}:python/sglang/srt/${file}`],{encoding:'utf8'});
 assert.ok(code.split('\n')[line-1].includes(text),`${key}: ${file}:${line} must contain ${text}`);
}console.log(`PASS: ${Object.keys(F.sources).length} implementation source anchors.`);}
console.log('PASS: implementation groups, request ownership, CP/SP gates, token conservation, LSE normalization and reversible timelines.');
