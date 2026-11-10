const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process'),M=require('../../pages/sglang/performance-engineering/model.js');
const base=M.timeline(),client=M.timeline('client'),queue=M.timeline('queue');
assert.equal(base.ttft,220);assert.equal(base.e2e,360);assert.equal(base.total,440);assert.equal(client.ttft,base.ttft);assert.equal(client.e2e,base.e2e);assert.equal(client.total,base.total+120);assert.equal(queue.ttft,base.ttft+120);assert.equal(queue.e2e,base.e2e+120);assert.equal(M.timeline('compute').e2e,base.e2e-50);
for(const k of ['base','client','queue','compute']){const d=M.timeline(k);assert.equal(d.segments.reduce((sum,s)=>sum+s.duration,0),d.total);for(let i=1;i<d.segments.length;i++)assert.equal(d.segments[i].start,d.segments[i-1].end);}
assert.deepEqual(M.packets('token').itls,[40,40,40]);assert.deepEqual(M.packets('bundle').itls,[40,40,40]);assert.notEqual(M.packets('bundle').chunks.length,M.packets('token').chunks.length);assert.equal(M.packets().tpot,140/3);assert.equal(M.packets('single').tpot,null);assert.deepEqual(M.packets('single').itls,[]);
const a=M.schedule(1),b=M.schedule(3),fast=M.schedule(3,2);assert.equal(a.wall,720);assert.equal(b.wall,360);assert.ok(b.throughput>a.throughput&&b.e2e>a.e2e);assert.equal(a.readyMean,420);assert.equal(b.readyMean,270);assert.equal(b.wall/fast.wall,4/3);assert.equal(fast.tpot,b.tpot);for(const d of [a,b,fast])assert.deepEqual(d.groups.flatMap(g=>g.ids),[1,2,3,4,5,6]);assert.throws(()=>M.schedule(0));
const all=M.sample(),failed=M.sample('failure');assert.equal(all.mean,185);assert.equal(failed.mean,80);assert.equal(all.throughput,16);assert.equal(failed.throughput,12);assert.equal(failed.failed,1);assert.ok(Math.abs(all.p95-437)<1e-9);assert.equal(M.percentile([],0.95),null);
if(process.env.SGLANG_SOURCE_DIR)for(const [key,[file,line,symbol]] of Object.entries(M.sources)){const code=execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',`${M.revision}:${file}`],{encoding:'utf8'});assert.ok(code.split('\n')[line-1].includes(symbol),`${key}: line ${line}: ${code.split('\n')[line-1]}`);}
console.log(`PASS: timer boundaries, packets versus tokens, missing TPOT, conserved requests, throughput/latency tradeoff, partial speedup and success bias; ${process.env.SGLANG_SOURCE_DIR?Object.keys(M.sources).length:0} fixed anchors.`);

const PD=require('../../pages/sglang/performance-engineering/timeline.js');
const metric=(tl,id)=>tl.metrics.find(m=>m.id===id),value=(tl,id)=>{const m=metric(tl,id);return m.value??m.end-m.start;};
let traces=0;
for(const [mode,m]of Object.entries(PD.modes))for(const condition of Object.keys(m.cases)){
 const tl=PD.timeline(mode,condition);traces++;
 assert.equal(new Set(tl.events.map(e=>e.id)).size,tl.events.length);
 for(const e of tl.events){assert(e.end>e.start&&e.start>=0,e.id);assert(PD.sources[e.source],e.source);for(const id of e.depends){const d=tl.events.find(d=>d.id===id);assert(d,`${e.id} missing ${id}`);assert(d.end<=e.start,`${mode}/${condition}: ${id} ends ${d.end} after ${e.id} starts ${e.start}`);}}
 for(const m of tl.metrics){assert(m.end>=m.start);assert(m.absent||m.observe>=m.end);assert(PD.sources[m.source]);}
 const before=PD.snapshot(tl,tl.duration*.43);PD.snapshot(tl,tl.duration);assert.deepEqual(PD.snapshot(tl,tl.duration*.43).observations,before.observations,'rewind restores observations');
 const f=tl.facts;if(mode!=='sample'){
  assert(f.first>=f.pGate&&f.first>=f.dFirst);assert(f.dWait>=tl.events.find(e=>e.id==='kv-last').end);assert.equal(metric(tl,'p-ttft').absent,true);
  for(const m of tl.metrics.filter(m=>!m.absent)){assert.notEqual(PD.snapshot(tl,m.observe-.01).observations.find(x=>x.id===m.id).status,'已采集');assert.equal(PD.snapshot(tl,m.observe).observations.find(x=>x.id===m.id).status,'已采集');}
 }
}
assert.equal(traces,12);
const normal=PD.timeline('clock','base'),permitPD=PD.timeline('clock','permit'),gatePD=PD.timeline('clock','gate'),pollPD=PD.timeline('clock','poll');
assert.equal(value(normal,'client-ttft'),385);assert.equal(value(normal,'d-ttft'),333);
assert.equal(value(permitPD,'client-ttft'),value(normal,'client-ttft'));assert.equal(permitPD.duration-normal.duration,120);
assert.equal(value(gatePD,'d-ttft'),value(normal,'d-ttft'));assert.equal(value(gatePD,'client-ttft')-value(normal,'client-ttft'),120);
assert.equal(value(pollPD,'d-transfer')-value(normal,'d-transfer'),80);
assert.deepEqual(pollPD.events.find(e=>e.id==='kv-last'),normal.events.find(e=>e.id==='kv-last'));
assert(value(normal,'d-transfer')>normal.events.find(e=>e.id==='kv-last').end-normal.events.find(e=>e.id==='kv-last').start);
assert.equal(value(normal,'p-transfer'),70);assert.equal(value(normal,'kv-latency'),71);
const chunk=PD.timeline('kv','chunked');assert(PD.snapshot(chunk,310).active.some(e=>e.id==='p-gpu2'));assert(PD.snapshot(chunk,310).active.some(e=>e.id==='kv1'));assert(chunk.events.find(e=>e.id==='kv1').start<metric(chunk,'kv-latency').start);
assert.equal(metric(PD.timeline('kv','missing'),'kv-latency').absent,true);
const stream=PD.snapshot(PD.timeline('stream','token'),9999),bundled=PD.snapshot(PD.timeline('stream','bundle'),9999),single=PD.snapshot(PD.timeline('stream','single'),9999);
assert.deepEqual(stream.clientItl,[35,55,30]);assert.deepEqual(bundled.clientItl,[40,40,40]);assert.deepEqual(bundled.tmItl,[40,40,40]);assert.equal(bundled.tpot,140/3);assert.equal(single.tpot,null);assert.equal(single.clientItl.length,0);assert.equal(value(PD.timeline('stream','bundle'),'d-itl'),40);assert.equal(value(PD.timeline('stream','bundle'),'client-tpot'),140/3);assert(!PD.timeline('stream','single').events.some(e=>e.lane==='dg'));
const ok=PD.timeline('sample','all'),bad=PD.timeline('sample','failure');assert(Math.abs(value(ok,'p95')-437)<1e-9);assert.equal(value(bad,'p95'),80);assert.equal(ok.facts.wall,bad.facts.wall);assert.equal(bad.facts.result.outputTokens,12);
if(process.env.SGLANG_SOURCE_DIR){
 const cache=new Map(),read=f=>{if(!cache.has(f))cache.set(f,execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',`${PD.revision}:${f}`],{encoding:'utf8'}));return cache.get(f);};
 for(const [k,[f,l,n]]of Object.entries(PD.sources))assert(read(f).split('\n')[l-1]?.includes(n),`${k}: ${f}:${l}`);
 const ts=read('python/sglang/srt/observability/req_time_stats.py'),tm=read('python/sglang/srt/managers/tokenizer_manager.py'),mc=read('python/sglang/srt/observability/metrics_collector.py');
 assert.match(ts,/stage = RequestStage.DECODE_TRANSFERRED\s+slice_start_time = self.decode_transfer_queue_entry_time/);
 assert.match(ts,/if transfer_metric.transfer_total_bytes is None:\s+return/);assert.match(ts,/if transfer_metric.transfer_latency_s is not None:[\s\S]*self.completion_time - self.prefill_transfer_queue_entry_time/);
 assert.match(tm,/not state.ttft_observed\s+and self.disaggregation_mode != DisaggregationMode.PREFILL/);assert.match(tm,/if self.enable_metrics and state.obj.log_metrics:\s+self.collect_metrics/);
 assert.match(mc,/adjusted_interval = internval \/ num_new_tokens/);assert.match(mc,/his._buckets\[i\].inc\(num_new_tokens\)/);assert.match(tm,/"engine_type": engine_type/);
 const streamer=read('python/sglang/srt/managers/scheduler_components/output_streamer.py');assert.match(streamer,/len\(req.output_ids\) % stream_interval == 1/);assert.match(streamer,/if req.finished\(\):[\s\S]*should_output = True/);
 const p=read('python/sglang/srt/disaggregation/prefill.py');assert.match(p,/send_kv_chunk\(req, last_chunk=True\)\s+req.time_stats.set_prefill_transfer_queue_entry_time\(\)/);assert.match(p,/if isinstance\(req.finished_reason, FINISH_ABORT\):\s+continue[\s\S]*compute_and_observe_kv_transfer_metrics/);
 console.log(`PASS ${Object.keys(PD.sources).length} fixed source anchors and P/D observation contracts.`);
}
console.log('PASS 12 P/D traces: client/server boundaries, parallel waits, late observation, last-chunk fallback, streaming and failed-sample aggregation.');

// Overview is a projection of the same trace, including its concurrent windows.
for(const [mode,m]of Object.entries(PD.modes))for(const condition of Object.keys(m.cases)){
 const tl=PD.timeline(mode,condition),p=PD.overview(tl);assert.equal(Object.keys(p.lanes).length,4);
 for(const e of p.events){assert(p.lanes[e.lane]);assert(e.start>=0&&e.end<=tl.duration&&e.end>e.start);for(const id of e.children||[])assert(tl.events.some(x=>x.id===id));}
 if(mode!=='sample'){assert.equal(p.events.find(e=>e.id==='o-first').start,tl.facts.st);assert.equal(p.events.find(e=>e.id==='o-first').end,tl.facts.first);}
}
const projected=PD.overview(chunk);assert(projected.events.some(e=>e.id==='kv1'&&e.start<310&&e.end>310));assert(projected.events.some(e=>e.id==='o-p-work'&&e.start<310&&e.end>310));
console.log('PASS overview projection preserves original clock, TTFT boundaries and chunk overlap.');
