const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');const M=require('../../pages/sglang/serving-operations/model.js');
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

// P/D governance: asynchronous actors share a teaching clock; gates are causal,
// lengths are illustrative and are not a prediction of runtime performance.
const PD=require('../../pages/sglang/serving-operations/timeline.js');
let traces=0;
for(const [mode,m]of Object.entries(PD.modes))for(const condition of Object.keys(m.cases)){
 const tl=PD.timeline(mode,condition),ids=new Set(tl.events.map(e=>e.id));traces++;
 assert.equal(ids.size,tl.events.length);assert(tl.duration>0);
 for(const e of tl.events){
  assert(e.end>e.start&&e.start>=0,`${mode}/${condition}/${e.id}`);
  assert(PD.sources[e.source]);assert(PD.laneNames[e.lane]);
  for(const id of e.depends||[]){const dep=tl.events.find(x=>x.id===id);assert(dep,`missing dependency ${id}`);assert(dep.end<=e.start+1e-8,`dependency ${id} finishes after ${e.id} starts`);}
  for(const t of [e.start,(e.start+e.end)/2,e.end]){
   const a=PD.snapshot(tl,t);PD.snapshot(tl,tl.duration);assert.deepEqual(PD.snapshot(tl,t).state,a.state,'reverse seek is deterministic');
   assert.equal(a.time,t);assert(a.active.every(x=>x.start<=t&&t<x.end));
  }
 }
}
assert.equal(traces,14);
const state=(m,c,t)=>PD.snapshot(PD.timeline(m,c),t).state;
const active=(m,c,t)=>PD.snapshot(PD.timeline(m,c),t).active.map(e=>e.id);
assert(active('health','pair',1.5).includes('probe-P')&&active('health','pair',1.5).includes('probe-D'));
assert.equal(state('health','stall',5).client,'连接中，等待响应');assert.match(state('health','stall',20).client,/503/);
assert.equal(state('health','startup',5.5).P.http,'Up');assert.equal(state('health','startup',5.5).D.http,'Starting');
assert.deepEqual(active('dispatch','success',2.6),['send-P','send-D']);
assert.deepEqual(active('dispatch','success',13),['p-retire','decode']);
assert.deepEqual(active('dispatch','success',13.8),['p-response','decode']);
assert.equal(state('dispatch','early',13.8).clientDone,false);assert.match(state('dispatch','early',13.8).gateway,/decode_early/);
assert.equal(state('dispatch','none',20).D.tm,'无 R1 登记');
assert.equal(state('cancel','prefill',3.5).clientDone,true);assert.equal(state('cancel','prefill',3.5).abortSent,false);assert.equal(state('cancel','prefill',20).D.allocated,false);
assert(active('cancel','stream',5.5).includes('gpu')&&active('cancel','stream',5.5).includes('mark'));
assert.equal(state('cancel','stream',6.5).D.released,false);assert.equal(state('cancel','stream',9).D.released,true);
assert.equal(state('cancel','transfer',5.5).D.frontDone,true);assert.equal(state('cancel','transfer',5.5).D.released,false);assert.equal(state('cancel','transfer',5.5).ack,false);
assert.equal(state('cancel','transfer',8).ack,true);assert.equal(state('cancel','transfer',8).D.released,false);assert.equal(state('cancel','transfer',10).D.released,true);
assert.equal(state('drain','normal',9.1).processExited,true);assert(active('drain','normal',9.1).includes('d-continues'));
assert.equal(state('drain','arrivals',20).shutdown,false);assert.equal(state('drain','stuck',20).processExited,false);
if(process.env.SGLANG_SOURCE_DIR){
 const cache=new Map(),read=f=>{if(!cache.has(f))cache.set(f,execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',`${PD.revision}:${f}`],{encoding:'utf8'}));return cache.get(f);};
 for(const [key,[f,l,symbol]]of Object.entries(PD.sources))assert(read(f).split('\n')[l-1]?.includes(symbol),`${key}: source anchor ${f}:${l}`);
 const router=read('sgl-model-gateway/src/routers/http/pd_router.rs');
 const dispatch=router.slice(router.indexOf('async fn execute_dual_dispatch_internal'),router.indexOf('async fn select_pd_pair'));
 assert.match(dispatch,/tokio::select!\s*\{\s*biased;[\s\S]*prefill_result = pr;[\s\S]*break;[\s\S]*decode_early = Some\(dr\)/);
 assert(dispatch.indexOf('if prefill_failed')<dispatch.indexOf('let decode_result = match decode_early'));
 const failed=dispatch.slice(dispatch.indexOf('if prefill_failed'),dispatch.indexOf('// Prefill ok:'));
 assert.match(failed,/prefill.record_outcome/);assert.doesNotMatch(failed,/decode.record_outcome/);assert.match(failed,/return response/);
 assert.match(router.slice(router.indexOf('async fn health_generate')),/tokio::join!/);
 const sched=read('python/sglang/srt/managers/scheduler.py'),q=sched.slice(sched.indexOf('def _add_request_to_queue'),sched.indexOf('def _abort_on_queued_limit'));
 assert.match(q,/DisaggregationMode.NULL[\s\S]*_abort_on_queued_limit[\s\S]*DisaggregationMode.PREFILL[\s\S]*bootstrap_queue.add[\s\S]*DisaggregationMode.DECODE[\s\S]*prealloc_queue.add/);
 const abort=sched.slice(sched.indexOf('def abort_request'),sched.indexOf('def abort_request')+9500);
 assert.match(abort,/receiver.abort\(\)[\s\S]*register_deferred_abort_room/);assert.match(abort,/collect_inflight_reqs\(\)[\s\S]*req.to_finish = FINISH_ABORT/);
 const dec=read('python/sglang/srt/disaggregation/decode.py'),deferred=dec.slice(dec.indexOf('def _defer_release'),dec.indexOf('def release_memory_occupation',dec.indexOf('def _defer_release')));
 assert.match(deferred,/required_acks = len\(decode_req.kv_receiver.bootstrap_infos\)/);assert.match(deferred,/if not drained and now < deadline/);
 assert(deferred.indexOf('self._deferred_releases = still_held')<deferred.indexOf('self._do_release(decode_req, idx)'));
 const ack=read('python/sglang/srt/disaggregation/common/conn.py');assert.match(ack,/def _maybe_ack_drained_abort[\s\S]*_staging_outstanding.get\(room, 0\) > 0:[\s\S]*return[\s\S]*_send_abort_ack/);
 console.log(`PASS: ${Object.keys(PD.sources).length} fixed P/D anchors and dispatch, queue, cancellation and ACK source contracts.`);
}
console.log('PASS: 14 P/D traces; concurrency, early D result gate, cancellation/resource ownership, reverse seek and local drain boundaries.');
