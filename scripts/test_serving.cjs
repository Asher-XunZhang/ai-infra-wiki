const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');const M=require('../pages/sglang/serving-operations/model.js');
assert.equal(M.health({starting:true,bypass:true}),503);assert.equal(M.health({exiting:true,generation:false}),503);assert.equal(M.health({generation:false}),200);assert.equal(M.health({last:101}),200);assert.equal(M.health({last:100}),null);assert.equal(M.health({now:120}),503);
assert.deepEqual(M.route('both',3).assigned,[['R1','R3'],['R2']]);for(const c of ['b-down','circuit'])assert.deepEqual(M.route(c,3).assigned,[['R1','R2','R3'],[]]);assert.deepEqual(M.route('none',3).rejected,['R1','R2','R3']);
for(const cap of [1,2])for(let step=0;step<=5;step++){
 const d=M.queue(cap,step);assert.ok(d.waiting.length<=cap);const all=[d.running,...d.waiting,...d.rejected,...d.done].filter(Boolean);assert.equal(new Set(all).size,all.length);assert.equal(all.length,1+Math.min(step,3)+(step===5?1:0));
}assert.deepEqual(M.queue(2,3).rejected,['R3']);assert.equal(M.queue(2,4).running,'R1');assert.deepEqual(M.queue(2,5).waiting,['R2','R4']);assert.throws(()=>M.queue(0,1));
assert.equal(M.waitingTimedOut(100,120,20),false);assert.equal(M.waitingTimedOut(100,121,20),true);assert.equal(M.waitingTimedOut(0,121,20),false);assert.equal(M.waitingTimedOut(100,121,0),false);
for(const kind of ['disconnect','waiting']){assert.equal(M.cancellation(kind,1).queued,true);assert.equal(M.cancellation(kind,2).queued,true);assert.equal(M.cancellation(kind,3).frontend,true);assert.equal(M.cancellation(kind,4).frontend,false);}
assert.equal(M.drain('normal',1).ingressStopped,false);assert.equal(M.drain('normal',2).shutdownSent,false);assert.equal(M.drain('normal',3).shutdownSent,true);assert.equal(M.drain('arrivals',3).shutdownSent,false);assert.equal(M.drain('stuck',5).killFallback,true);assert.equal(M.drain('stuck',5).exitObserved,false);assert.equal(M.drain('stuck',5).schedulerReleased,false);assert.equal(M.drain('normal',5).exitObserved,true);
if(process.env.SGLANG_SOURCE_DIR)for(const [key,[file,line,symbol]] of Object.entries(M.sources)){
 const code=execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',`${M.revision}:${file}`],{encoding:'utf8'});
 assert.ok(code.split('\n')[line-1].includes(symbol),`${key}: ${line}: ${code.split('\n')[line-1]}`);
}console.log(`PASS: health evidence, eligible routing, queue conservation/capacity, timeout boundary, cancel echo, drain and force-exit distinction; ${process.env.SGLANG_SOURCE_DIR?Object.keys(M.sources).length:0} fixed anchors checked.`);
