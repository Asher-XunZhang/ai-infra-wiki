const assert=require('node:assert/strict');
const M=require('../../pages/sglang/loop-journey/model.js');
assert.equal(M.validate(),true);
for(const [mode,c]of Object.entries(M.configs)){
 const barrier=c.steps.findIndex(s=>s.id===c.barrier);
 for(let i=0;i<c.steps.length;i++){
  const s=M.snapshot(mode,i),again=M.snapshot(mode,i);
  assert.deepEqual(s,again,`${mode}: snapshots must be deterministic`);
  assert.ok(s.dependencies.every(d=>d.ready),`${mode}: unsatisfied producer`);
  // Backwards reads cannot retain the future host state.
  M.snapshot(mode,c.steps.length-1);assert.deepEqual(M.snapshot(mode,i),again);
 }
 const blocked=M.snapshot(mode,100,{hold:true});
 assert.equal(blocked.index,barrier);assert.equal(blocked.blocked,true);
 assert.ok(!blocked.completed.includes(c.barrier));
 assert.equal(blocked.data.host,M.snapshot(mode,barrier-1).data.host,'blocked action must not commit host results');
 if(mode==='pp'){assert.match(blocked.data.batch,/M2 已提交/);assert.match(blocked.data.device,/激活尚未就绪/);}
 assert.equal(M.snapshot(mode,100,{hold:true,released:true}).last,true);
 assert.equal(M.snapshot(mode,-5).index,0);assert.equal(M.snapshot(mode,Infinity).last,true);
}
const o=M.configs.overlap.steps,at=id=>M.snapshot('overlap',o.findIndex(s=>s.id===id));
assert.match(at('launch-next').data.queue,/B0, B1/);
assert.match(at('ready').data.batch,/B1.*已提交/);
assert.equal(at('finish').data.host,at('drain').data.host,'overshoot cannot add valid output after EOS');
assert.match(at('finish').data.queue,/B2/);assert.match(at('drain').data.queue,/空/);
const pp=M.configs.pp.steps;
const L1=pp.flatMap(e=>e.a).filter(a=>a[3]==='P0:L1');
assert.ok(L1.some(a=>a[0]==='gpu0'&&a[2]==='M2'));
assert.ok(L1.some(a=>a[0]==='cpu0'&&a[2]==='M1'));
assert.ok(pp.findIndex(s=>s.id==='consume0')<pp.findIndex(s=>s.id==='proxy2'));
assert.match(M.snapshot('pp',10).data.host,/PP1.*R2 待处理/);
assert.match(M.snapshot('pp',12).data.request,/两级均完成/);
assert.throws(()=>M.snapshot('invalid'),/Unknown mode/);
console.log('PASS: 3 journeys, 36 events; causal gates, deterministic rewind, blocked state, EOS drain, PP local-loop identity and output-ring closure.');

for(const mode of Object.keys(M.configs)){
 const e=M.execution(mode),ids=Object.fromEntries(e.events.map(x=>[x.id,x]));
 assert.equal(e.times.length,M.configs[mode].steps.length);
 assert.ok(e.times.every((t,i)=>!i||t>=e.times[i-1]));
 for(const x of e.events){assert.ok(x.end>x.start);assert.ok(x.end<=e.duration);for(const d of x.depends){assert.ok(ids[d],d);assert.ok(ids[d].end<=x.start+1e-6,`${mode}: ${x.id} before ${d}`);}}
 for(const l of e.lanes){const a=e.events.filter(x=>x.lane===l.id).sort((a,b)=>a.start-b.start);for(let i=1;i<a.length;i++)assert.ok(a[i-1].end<=a[i].start+1e-6,`${mode}: exclusive lane overlap`);}
 for(let i=0;i<e.times.length;i++){const sample=M.executionSnapshot(mode,i);assert.equal(sample.time,e.times[i]);for(const v of Object.values(sample.progress))assert.ok(v>=0&&v<=1);}
 const blocked=M.executionSnapshot(mode,100,{hold:true});assert.equal(blocked.progress[e.barrierEvent],0);assert.ok(blocked.blocked);
}
const ov=M.execution('overlap'),ovEvent=id=>ov.events.find(e=>e.id===id);
assert.ok(ovEvent('submit1').end<ovEvent('sample0').end,'CPU may submit before token materialization');
assert.ok(ovEvent('process0').start<ovEvent('forward1').end&&ovEvent('process0').end>ovEvent('forward1').start,'old CPU processing overlaps next GPU forward');
console.log('PASS: 90 execution activities share scenario clocks; producer completion, exclusive lanes, hypothetical durations, blocked clock and real overlap windows.');

// Asynchronous completion time does not change the loop that submitted work.
for(const n of [0,1,2]){
 assert.equal(ovEvent('copy'+n).loop,ovEvent('submit'+n).loop);
 assert.notEqual(ovEvent('process'+n).loop,ovEvent('copy'+n).loop);
}
const ppActivities=M.execution('pp').events,ppActivity=id=>ppActivities.find(e=>e.id===id);
assert.ok(ppActivities.every(e=>/^P[01]:L\d$/.test(e.loop)),'every PP activity has a stage-local owner');
for(const e of ppActivities.filter(e=>e.targetLoop))assert.notEqual(e.loop.slice(0,2),e.targetLoop.slice(0,2));
assert.equal(ppActivity('out1').loop,ppActivity('postOut1').loop);
assert.equal(ppActivity('out1').targetLoop,ppActivity('process01').loop);
assert.equal(ppActivity('relay1').loop,ppActivity('postRelay1').loop);
assert.equal(ppActivity('relay1').targetLoop,ppActivity('process11').loop);
assert.equal(ppActivity('waitProxySend1').batch,ppActivity('gpu02').batch,'M2 owns the wait for the previous send');
assert.equal(ovEvent('war0').batch,ovEvent('input1').batch,'B1 owns the wait before writing its metadata');
console.log('PASS: batch ownership, asynchronous loop origin, and message sender/receiver loop identities.');

// Fractional time is reversible; state commits never precede the checkpoint.
for(const mode of Object.keys(M.configs)){
 const scene=M.execution(mode),byId=Object.fromEntries(scene.events.map(e=>[e.id,e]));
 const times=[0,scene.duration,...scene.events.flatMap(e=>[Math.max(0,e.start-.001),e.start,(e.start+e.end)/2,e.end])];
 for(const time of times){
  const sample=M.sample(mode,time);M.sample(mode,scene.duration);
  assert.deepEqual(M.sample(mode,time),sample,mode+': rewind must discard future state');
  assert.equal(sample.time,time);assert.ok(scene.times[sample.index]<=time);
  for(const e of scene.events){const p=sample.execution.progress[e.id];assert.ok(p>=0&&p<=1);if(p>0)for(const dep of e.depends)assert.equal(sample.execution.progress[dep],1,mode+': consumer advanced before '+dep);}
 }
 const gate=byId[scene.barrierEvent],blocked=M.sample(mode,scene.duration,{hold:true});
 assert.equal(blocked.time,gate.start);assert.equal(blocked.execution.progress[gate.id],0);
 assert.ok(!blocked.execution.active.some(e=>e.id===gate.id),'withheld transfer must not appear active');
 assert.equal(blocked.blocked,true);assert.equal(blocked.last,false);
 const released=M.sample(mode,blocked.time,{hold:true,released:true});
 assert.equal(released.time,blocked.time);assert.equal(released.execution.progress[gate.id],0,'releasing a dependency does not skip its transfer');
 assert.equal(released.blocked,false);
 assert.equal(M.sample(mode,-5).time,0);assert.equal(M.sample(mode,Infinity).time,scene.duration);
}
for(const [mode,before,after] of [['normal',8.399,8.4],['overlap',11.599,11.6],['pp',11.199,11.2]])assert.notEqual(M.sample(mode,before).data.host,M.sample(mode,after).data.host,'CPU commit must happen at its boundary');
assert.ok(M.sample('overlap',8).execution.active.some(e=>e.id==='process0'));
assert.ok(M.sample('overlap',8).execution.active.some(e=>e.id==='forward1'));
assert.equal(M.sample('overlap',14.8).data.host,M.sample('overlap',11.6).data.host,'draining never adds output after EOS');
console.log('PASS: fractional clock, boundary commits, producer gates, reversible playback and release without jumping through transfer.');

assert.ok(M.sample('overlap',10,{hold:true}).execution.active.some(e=>e.id==='forward1'),'independent GPU work remains visible at the teaching pause');
