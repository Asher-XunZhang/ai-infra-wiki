/* Teaching invariants: first-token boundary, KV accounting, admission gates,
 * chunk conservation and metric definitions. Run with Node, no dependencies. */
const assert=require('node:assert/strict');
const model=require('../pages/sglang/inference-overview/model.js');
for(let prompt=1;prompt<=12;prompt++){
  assert.equal(model.generation(prompt,0).kv,0);
  for(let step=1;step<=5;step++){
    const s=model.generation(prompt,step);
    assert.equal(s.kv,prompt+s.outputs-1,'latest sampled output must not have KV yet');
    assert.equal(s.input,step===1?prompt:1);
  }
}
for(let reuse=0;reuse<=7;reuse++)for(let decode=0;decode<=4;decode++){
  const s=model.cache(8,reuse,decode);
  assert.equal(s.computed+s.reuse,8);
  assert.equal(s.kv,8+s.outputs-1);
}
for(let flags=0;flags<8;flags++){
  const gates=[flags&1,flags&2,flags&4];
  for(const step of [3,4]){
    assert.equal(model.handoff(true,step,...gates).blocked,flags!==7,'all handoff gates are necessary');
    assert.equal(model.handoff(false,step,...gates).blocked,false,'unified has no remote handoff gate');
  }
}
for(let chunk=1;chunk<=8;chunk++){
  const s=model.schedule(chunk,99);
  assert.equal(s.rounds.reduce((n,r)=>n+r.prefill,0),8,'chunks conserve input positions');
  assert.equal(s.processed,8);
  assert.equal(s.aTokens,Math.ceil(8/chunk));
}
const normal=model.latency(20,60,0,20),queued=model.latency(100,60,0,20),slow=model.latency(20,60,0,40);
assert.deepEqual([normal.ttft,normal.tpot,normal.total],[80,20,140]);
assert.equal(queued.tpot,normal.tpot);assert.equal(queued.ttft-normal.ttft,80);
assert.equal(slow.ttft,normal.ttft);assert.equal(slow.total-normal.total,60);
console.log('PASS: generation / cache accounting, all handoff gate combinations, all chunk sizes, TTFT and TPOT boundaries');
