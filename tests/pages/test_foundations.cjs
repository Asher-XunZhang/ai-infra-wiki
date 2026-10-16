const assert=require('node:assert/strict');
const M=require('../../pages/foundations/model.js');
for(const [key,d] of Object.entries(M.definitions))for(let v=0;v<d.options.length;v++)for(let s=0;s<d.steps.length;s++){const state=M.snapshot(key,v,s);assert.equal(state.step,s);assert.equal(state.last,d.steps.length-1);}
assert.equal(M.examples[0].pieces.join(''),M.examples[0].text);assert.deepEqual(M.examples[0].ids,[104100,100134,104455,1773]);assert.equal(M.examples[1].ids[1],M.examples[1].ids[3]);
for(const v of [0,1]){const d=M.attention(v);assert(Math.abs(d.weights.reduce((a,b)=>a+b,0)-1)<1e-10);assert(d.weights.every((x,i)=>i<=d.query||x===0));assert.equal(d.output.length,2);}
assert.equal(M.snapshot('generation',0,2).kv.length,2);assert.equal(M.snapshot('generation',0,4).kv.length,2,'new token has not yet completed a forward');assert.equal(M.snapshot('generation',0,5).kv.length,3,'KV appears after computing the new input');assert.equal(M.snapshot('generation',0,6).kv.at(-1),'学习');assert.equal(M.snapshot('generation',1,6).visible,'休息一下');
assert.equal(M.snapshot('cache',0,3).computed.reduce((a,b)=>a+b,0),15);assert.equal(M.snapshot('cache',1,3).computed.reduce((a,b)=>a+b,0),6);
assert.equal(M.snapshot('cache',2,1).ownedPages,1);assert.equal(M.snapshot('cache',2,1).written,0);assert.equal(M.snapshot('cache',2,3).ownedPages,2);assert.equal(M.snapshot('cache',2,3).written,3);assert.deepEqual(M.snapshot('cache',3,3).holders,['R2']);assert.equal(M.snapshot('cache',3,4).retained,true);
assert.deepEqual(M.schedule(0,2).rounds[1],['B']);assert.deepEqual(M.schedule(1,2).rounds[1],['B','C']);for(const v of [0,1])for(let s=0;s<5;s++){const d=M.schedule(v,s),ids=[...d.active,...d.waiting,...d.done];assert.equal(new Set(ids).size,ids.length);assert.equal(ids.length,s?3:2);assert(d.rounds.every(r=>r.length<=2));}
assert.equal(M.snapshot('hardware',2,4).bound,8);assert.equal(M.snapshot('hardware',1,4).bound,4);
assert.deepEqual(M.snapshot('parallel',1,1).lanes,['MB1',null]);assert.deepEqual(M.snapshot('parallel',1,2).lanes,['MB2','MB1']);
assert.deepEqual(M.snapshot('communication',0,4).result,[11,22]);assert.deepEqual(M.snapshot('communication',1,4).result,[1,2,10,20]);assert.equal(M.snapshot('communication',2,1).arrived,false);
assert.equal(M.snapshot('deployment',1,3).decode,false);assert.equal(M.snapshot('deployment',1,4).decode,true);
for(const [v,a] of [[0,2],[1,0],[2,4]]){const d=M.snapshot('speculation',v,4);assert.equal(d.output.length,a+1);assert.deepEqual(d.validKV,['根',...d.draft.slice(0,a)]);assert.notDeepEqual(d.validKV,d.output);}
assert.equal(M.snapshot('serving',2,3).clientDone,true);assert.equal(M.snapshot('serving',2,3).resourceHeld,true);assert.equal(M.snapshot('serving',2,3).cancelReceived,false);
assert.equal(M.snapshot('metrics',2,4).p95,190);assert.equal(M.snapshot('metrics',3,4).p95,180);assert.equal(M.snapshot('metrics',3,4).failed,2);
assert.throws(()=>M.snapshot('tokens',999,0));assert.throws(()=>M.snapshot('tokens',0,-1));


const throughput=M.snapshot('metrics',4,4);assert.equal(throughput.completed/throughput.window,throughput.qps);assert.equal(throughput.outputCounts.reduce((a,b)=>a+b,0)/throughput.window,throughput.outputTPS);

console.log('PASS: 12 teaching models; token fixtures, causal weights, KV ownership, batch conservation, data dependencies, statistics boundaries.');
