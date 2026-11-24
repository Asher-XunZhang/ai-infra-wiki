'use strict';
const assert=require('node:assert/strict');
const M=require('../../pages/interactive-lessons/models.js');
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
let s=M.sampling();near(s.probabilities[0],5/11);near(s.mass,8/11);[.625,.375,0,0].forEach((x,i)=>near(s.q[i],x));assert.equal(s.selected,1);
assert.equal(M.sampling({temperature:1}).selected,0);
assert.deepEqual(M.sampling({scores:'signed',repetition:2}).penalized,[1,1,-2,0]);
for(const temperature of [0,.5,1,2])for(const topK of [1,2,3,4])for(const topP of [.1,.6,.85,1])for(const u of [.01,.75,.999]){
 const d=M.sampling({temperature,topK,topP,u});near(d.q.reduce((a,b)=>a+b,0),1);assert.ok(d.q[d.selected]>0);assert.ok(d.q.filter(x=>x>0).length<=(temperature===0?1:topK));assert.ok(d.q.every(Number.isFinite));
}
const edge=s.probabilities[0];
assert.ok(M.sampling({topK:4,topP:edge,rule:'sglang'}).q[1]>0,'frozen PyTorch boundary keeps the equality candidate');
assert.equal(M.sampling({topK:4,topP:edge,rule:'concept'}).q[1],0);
assert.throws(()=>M.sampling({temperature:-1}));assert.throws(()=>M.sampling({u:1}));
const q4=M.quantization(),q8=M.quantization({bits:8});assert.ok(q8.maxError<q4.maxError);
assert.ok(M.quantization({group:1}).total>q4.baseline,'metadata matters for a small array');
assert.equal(M.quantization({scale:'small'}).clipped,3);
for(const row of q4.rows){near(row.restored,row.q*row.scale);assert.ok(Math.abs(row.q)<=q4.qmax);}
assert.equal(M.quantizationPath({supported:'no'}).allowed,false);
for(const distribution of ['hot','spread']){
 const a=M.moe({distribution}),b=M.moe({distribution,replica:'on'});
 assert.deepEqual(a.outputs,b.outputs,'replicas must not change logical expert results');
 assert.equal(a.jobs.length,8);assert.equal(b.jobs.length,8);assert.equal(a.logical.reduce((x,y)=>x+y,0),8);
 for(const d of [a,b]){near(d.loads.reduce((x,y)=>x+y,0),8);assert.equal(d.join,Math.max(...d.events.filter(e=>e.id.startsWith('combine')).map(e=>e.end)));for(const j of d.jobs)assert.equal(j.origin,d.rows.find(r=>r.id===j.token).origin);}
}
assert.deepEqual(M.moe().loads,[8,0]);assert.deepEqual(M.moe({replica:'on'}).loads,[6,2]);
for(const roles of ['single','e-pd','e-p-d']){
 const d=M.multimodal({roles});assert.equal(d.positions.length,d.featureShape[0]+2);
 assert.ok(d.pStart>=d.events.find(e=>e.id==='encode').end);assert.ok(d.dStart>=d.pEnd);
 assert.equal(d.events.some(e=>e.id==='feature'),roles!=='single');assert.equal(d.events.some(e=>e.id==='kv'),roles==='e-p-d');
}
assert.equal(M.multimodal({cache:'hit',media:'changed'}).featureHit,false);
assert.ok(M.multimodal({cache:'hit',readCost:5}).duration>M.multimodal({cache:'miss'}).duration,'a hit does not guarantee a faster path');
let r=M.routing({policy:'cache',residency:'evicted'});assert.equal(r.w.hint,8);assert.equal(r.w.resident,0);assert.equal(r.missing,12);
assert.equal(M.routing({policy:'balanced',backlog:8}).choice,1);assert.equal(M.routing({policy:'balanced',backlog:1}).choice,0);
assert.equal(M.routing({checkpoint:'missing'}).events.some(e=>e.id==='decode'),false);
assert.ok(M.routing({policy:'balanced'}).duration<M.routing({policy:'cache'}).duration,'given wait assumptions must affect the timeline');
for(const scenario of ['oom','collective','pp','cancel']){
 const d=M.faults({scenario,evidence:'missing'});assert.equal(d.conclusive,false);assert.equal(d.logs.some(([rank])=>rank==='rank0'),false);assert.equal(d.events.some(e=>e.lane==='rank0'),false,'missing evidence must not leak in the diagram');
}
for(const capacity of [2,3]){
 const d=M.lora({capacity});assert.equal(d.admitted.length,capacity);assert.equal(d.waiting.length,3-capacity);
 assert.equal(new Set(d.admitted.map(r=>r.slot)).size,capacity);assert.ok(d.admitted.every(r=>r.slot>=0&&r.slot<capacity));
 assert.deepEqual(d.rows.map(r=>r.output),[[3,4],[3,0],[1,2]]);
}
const l=M.lora({revision:'reload'});assert.equal(l.kvHit,false);assert.deepEqual(l.rows[1].output,[5,0]);assert.equal(l.rows[1].slot,M.lora().rows[1].slot);
const vr=M.vllm();
assert.deepEqual(M.vllmState(vr,2),{computed:4,selected:0,received:0,retired:false,allocated:2,covered:1});
assert.deepEqual(M.vllmState(vr,4),{computed:6,selected:1,received:0,retired:false,allocated:2,covered:2});
assert.equal(M.vllmState(vr,5).received,1);
assert.equal(M.vllmState(vr,vr.duration).allocated,0);
assert.equal(M.vllmState(M.vllm({prefix:'miss'}),0).covered,0);
for(const budget of [2,4])for(const prefix of ['hit','miss']){
 const d=M.vllm({budget,prefix});assert.equal(d.totalIds,8);assert.equal(d.computed,7);assert.equal(d.emitted,2);assert.ok(d.steps.every(r=>r.count>0&&r.count<=budget));
 for(let i=1;i<d.steps.length;i++)assert.equal(d.steps[i].from,d.steps[i-1].computed);
 assert.equal(d.steps.filter(r=>r.sampled).length,2);assert.equal(d.steps.at(-1).blocks,2);
}
console.log('Priority lessons: probability/support boundaries, quantization metadata/error, MoE conservation, E/P/D gates, routing residency/recovery, missing evidence, LoRA isolation and V1 counters passed.');
