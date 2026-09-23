const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const M = require('../pages/sglang/request-runtime/model.js');
const D = require('../pages/sglang/request-runtime/sequence.js');
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
