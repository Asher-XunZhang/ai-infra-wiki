'use strict';
const assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const M=require('../../pages/sglang/request-state/model.js');
const state=(t,id)=>M.snapshot(t,t.events.find(e=>e.id===id).end).state;
for(const mode of Object.keys(M.modes))for(const condition of mode==='cancel'?['waiting','running']:['waiting']){
 const t=M.timeline(mode,condition),ids=new Set(),ends=new Map();
 for(const e of t.events){assert(!ids.has(e.id));ids.add(e.id);ends.set(e.id,e.end);assert(e.end>e.start);assert(M.sources[e.source]);}
 for(const e of t.events)for(const dep of e.depends||[])assert(ends.get(dep)<=e.start+1e-8,`${mode}/${e.id} starts before ${dep}`);
 for(const lane of ['cpu','gpu','copy']){
  const events=t.events.filter(e=>e.lane===lane).sort((a,b)=>a.start-b.start);
  for(let i=1;i<events.length;i++)assert(events[i-1].end<=events[i].start+1e-8,`${mode}: same ${lane} resource overlap`);
 }
 for(const time of [0,...t.events.flatMap(e=>[e.start,(e.start+e.end)/2,e.end]),t.duration]){
  const s=M.snapshot(t,time).state;assert(s.progress<=s.prompt);assert(s.allocated>=0&&s.written>=0&&s.cached>=0);
  if(s.row===null)assert.equal(s.allocated,0,'row released means no request reservation in scoped trace');
  if(s.protected)assert(s.row!==null&&s.cached>0);
  if(s.consuming)assert(!s.results.some(q=>q.id===s.consuming.id),'popped item cannot remain queued');
  assert.deepEqual(M.snapshot(t,time).state,s,'seek must reconstruct identical state');
 }
 const final=M.snapshot(t,t.duration).state;assert(!final.waiting&&!final.chunk&&!final.row&&!final.consuming&&!final.results.length);
}
const n=M.timeline('normal');
assert.equal(state(n,'merge').running,state(n,'merge').last,'running and last alias same batch when running was empty');
assert.equal(state(n,'finish').row,'q0');assert(state(n,'finish').members.S0.includes('R1'));
assert(!state(n,'release').row&&state(n,'release').cached>0,'cache outlives request row');
assert.deepEqual(state(n,'filter').members.S0,[]);
const c=M.timeline('chunk');
for(let i=0;i<2;i++){
 const s=state(c,'stash-'+i);assert(!s.waiting&&s.chunk&&s.row);assert.deepEqual(s.members['S'+i],[]);assert.deepEqual(s.output,[]);
}
assert(!state(c,'continue-1').chunk&&state(c,'continue-1').progress<6,'pointer clears at last-chunk admission before compute completion');
const r=M.timeline('retract');
assert.deepEqual(state(r,'reset').output,['t1','t2']);assert(state(r,'reset').retracted&&!state(r,'reset').row);
assert(state(r,'requeue').waiting);assert.deepEqual(state(r,'readmit').output,['t1','t2']);assert(!state(r,'readmit').retracted);
const w=M.timeline('cancel','waiting');assert.equal(state(w,'abort-pop').finished,null,'waiting abort is direct response path');
const a=M.timeline('cancel','running');assert.equal(state(a,'abort-mark').pending,'ABORT');assert.equal(state(a,'abort-mark').finished,null);assert(state(a,'abort-mark').row);
assert.equal(state(a,'abort-commit').finished,'ABORT');assert.equal(state(a,'abort-commit').pending,null);
const o=M.timeline('overlap');
assert(state(o,'take0').consuming&&!state(o,'take0').consuming.ready);assert.equal(state(o,'take0').results.length,1);
assert.equal(state(o,'copy0').consuming.token,'t1');
const concurrent=M.snapshot(o,17.1).state;
assert.equal(concurrent.consuming.id,'Q1');assert.equal(concurrent.consuming.token,'EOS');
assert.equal(concurrent.deviceToken,'额外结果','current GPU output is distinct from the result CPU commits');
assert.equal(state(o,'copy2').results.find(q=>q.id==='Q2').token,'额外结果');
const ct=o.events.find(e=>e.id==='commit0'),gpu=o.events.find(e=>e.id==='gpu1');assert(ct.start<gpu.end&&ct.end>gpu.start,'CPU consumes B0 while GPU runs B1');
assert(state(o,'commit1').finished&&state(o,'commit1').results.length,'logical end precedes drain');
assert.deepEqual(state(o,'drain').output,['t1','EOS'],'drain must not append extra token');
assert.deepEqual(M.snapshot(o,0).state,o.initial);
let anchors=0;if(process.env.SGLANG_SOURCE_DIR){
 const cache=new Map();for(const [key,[p,line,needle]] of Object.entries(M.sources)){
  if(!cache.has(p))cache.set(p,execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',`${M.revision}:python/sglang/srt/${p}`],{encoding:'utf8'}).split('\n'));
  assert(cache.get(p)[line-1]?.includes(needle),`${key}: source anchor drift ${p}:${line}`);anchors++;
 }
}
console.log(`PASS: 6 traces, commit boundaries, aliasing, ownership, cancellation, retraction, async dependencies and ${anchors} pinned anchors.`);
