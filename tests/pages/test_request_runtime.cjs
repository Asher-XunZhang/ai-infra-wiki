const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const M = require('../../pages/sglang/request-runtime/model.js');
const D = require('../../pages/sglang/request-runtime/sequence.js');
let checked = 0;
for (const scenario of ['normal','queued-abort','running-abort','invalid']) {
 for (const cache of [false,true]) for (let outputs=1; outputs<=5; outputs++) {
  const frames=M.build({scenario,cache,outputs}), by=id=>frames.find(f=>f.id===id), last=frames.at(-1);
  assert.equal(new Set(frames.map(f=>f.id)).size,frames.length);
  for(const f of frames){
   assert.ok(M.sources[f.source]);
   const visual=D.event(f);assert.ok(visual.edges.length>0);
   for(const [from,to] of visual.edges){assert.ok(from>=0&&from<6&&to>=0&&to<6);}
   if(f.id==='abort-pending')assert.deepEqual(visual.edges,[[2,2,'待结束']]);
   if(f.id==='invalid')assert.ok(visual.edges.every(([a,b])=>a===1&&b===1));
   if(f.id==='prefill'||f.id.startsWith('decode-'))assert.deepEqual(visual.edges.map(e=>e.slice(0,2)),[[2,3],[3,4]]);
   assert.ok(f.visible<=f.sampled,'response cannot expose a token before sampling');
   assert.ok(!f.slot||f.forwards>0,'simplified allocation begins at execution');
   assert.ok(!f.cached||!f.slot,'model moves ownership at release');
   assert.ok(!f.toFinish||!f.finished,'pending abort is not finished');
  }
  assert.equal(last.frontFinished,true);assert.equal(last.slot,false);assert.equal(last.waiting,false);
  if(scenario==='invalid'){
   assert.ok(frames.every(f=>!f.dispatched&&!f.slot&&f.kv===0&&f.forwards===0));
  }else{
   assert.equal(by('dispatch').dispatched,true);assert.equal(by('dispatch').scheduler,'尚无 Req');
   assert.equal(by('queued').forwards,0);
  }
  if(scenario==='normal'){
   assert.equal(by('finish').source,outputs===1?'prefillresult':'result');
   assert.equal(last.visible,outputs);assert.equal(last.forwards,outputs);
   assert.equal(by('finish').kv,4+outputs-1,'last sampled token has not entered forward');
   assert.equal(by('finish').slot,true,'finish and release are separate states');
   assert.equal(by('release').frontFinished,false,'backend resource release is not frontend completion');
   assert.equal(last.cached,cache?4+outputs-1:0);
   assert.equal(by('prefill').visible,0,'sampling is not delivery');
  }else if(scenario==='queued-abort'){
   assert.equal(by('cancel-sent').waiting,true,'sending abort cannot delete backend state');
   assert.equal(by('abort-queued').frontFinished,false,'backend removal is not frontend ACK');
   assert.equal(by('abort-front').source,'abortfront','queued abort uses its own control response handler');
   assert.ok(frames.every(f=>f.forwards===0&&f.kv===0&&f.cached===0&&!f.slot));
  }else if(scenario==='running-abort'){
   assert.equal(by('abort-pending').slot,true);assert.equal(by('abort-pending').kv,4);
   assert.equal(by('abort-pending').toFinish,true);assert.equal(by('abort-pending').finished,false);
   assert.equal(by('abort-result').finished,true);assert.equal(by('abort-result').toFinish,false);
   assert.equal(by('abort-result').slot,true);assert.equal(last.cached,cache?5:0);
   assert.equal(last.visible,1,'model does not claim how extra sampled text is delivered after cancel');
  }
  checked++;
 }
}
assert.throws(()=>M.build({scenario:'foo'}));assert.throws(()=>M.build({outputs:0}));
if(process.env.SGLANG_SOURCE_DIR){
 for(const [key,[file,line,symbol]] of Object.entries(M.sources)){
  const code=execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',`${M.revision}:python/sglang/srt/${file}`],{encoding:'utf8'});
  assert.match(code.split('\n')[line-1],new RegExp(`def ${symbol}\\(`),`source anchor ${key}`);
 }
 console.log(`PASS: ${Object.keys(M.sources).length} source anchors checked against fixed Git objects.`);
}
console.log(`PASS: ${checked} request paths/configurations; ownership, dispatch, cancellation, output and KV invariants.`);

// Continuous presentation must keep arrival, computation and cleanup separate.
const F=require('../../pages/sglang/request-runtime/flow.js');
for(const scenario of ['normal','queued-abort','running-abort','invalid'])for(const cache of [true,false])for(const outputs of [1,3,5]){
 const timeline=F.build({scenario,cache,outputs}),frames=M.build({scenario,cache,outputs});
 assert.deepEqual(F.at(timeline,0).state,F.initial);
 assert.deepEqual(F.at(timeline,timeline.duration).state,frames.at(-1));
 for(const p of timeline.phases){
  assert.deepEqual(F.at(timeline,p.start+p.duration*.98).state,p.frame,'phase completes at its boundary');
  for(const fraction of [.01,.2,.5,.7,.9]){
   const v=F.at(timeline,p.start+p.duration*fraction),s=v.state;
   assert.ok(s.visible<=s.sampled,'cannot deliver before sampling');
   assert.ok(!s.cached||!s.slot,'cache handoff releases request row');
   assert.ok(!s.finished||!s.toFinish,'pending cancellation is not finished');
   if(scenario==='invalid'||scenario==='queued-abort')assert.equal(s.kv+s.cached+s.forwards,0);
   assert.deepEqual(F.at(timeline,v.time),v,'seeking is deterministic');
  }
 }
 const find=id=>timeline.phases.find(p=>p.id===id);
 if(scenario==='normal'){
  const p=find('prefill'),early=F.at(timeline,p.start+p.duration*.2).state;
  assert.ok(early.slot&&!early.waiting,'admission reserves request row before forward');
  assert.equal(early.kv,0);assert.equal(early.sampled,0);
  const sample=F.at(timeline,p.start+p.duration*.7).state;
  assert.equal(sample.kv,4);assert.equal(sample.sampled,1);assert.equal(sample.visible,0);
  const final=find('final'),front=F.at(timeline,final.start+final.duration*.8).state;
  assert.ok(front.frontFinished);assert.ok(front.visible<outputs,'frontend cleanup precedes client arrival');
 }
 for(const e of timeline.events){assert.ok(e.end>e.start);if(e.edge)assert.ok(F.edges[e.edge]);for(const n of e.nodes)assert.ok(F.nodes[n]);}
}
const used=new Set(['normal','queued-abort','running-abort','invalid'].flatMap(scenario=>F.build({scenario}).events.map(e=>e.edge).filter(Boolean)));
assert.deepEqual([...used].sort(),Object.keys(F.edges).sort(),'every drawn connection carries a real event in some scenario');
for(const e of Object.values(F.edges))for(const [id,p]of [[e.from,e.points[0]],[e.to,e.points.at(-1)]]){
 const n=F.nodes[id],inside=p[0]>=n.x&&p[0]<=n.x+n.w&&p[1]>=n.y&&p[1]<=n.y+n.h;
 assert.ok(inside&&(p[0]===n.x||p[0]===n.x+n.w||p[1]===n.y||p[1]===n.y+n.h),'wire meets node boundary');
}
console.log('PASS: continuous arrival/compute/release boundaries, 24 configurations, reverse-seek determinism and connected geometry.');
