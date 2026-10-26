const assert=require('node:assert/strict');
const D=require('../../pages/motion/deployment-adoption.js');
/* Derive both work sets from the slot model objects: the same A(4 Decode) and
   B(4 Prefill + first Decode) must complete on both sides. B cannot Decode
   before the t5 transfer and the D readiness gate. */
const workSet=slots=>slots.flatMap(s=>s.tasks||[]).filter(t=>t.op!=='S').map(t=>`${t.req}:${t.op}:${t.n}`).sort();
const sharedWork=workSet(D.baseline.slots),separateWork=workSet([...D.contrast.p.slots,...D.contrast.d.slots]);
assert.deepEqual(separateWork,sharedWork,'both layouts complete the same A/B work set');
assert.equal(sharedWork.filter(k=>k.startsWith('A:D')).length,4);
assert.equal(sharedWork.filter(k=>k.startsWith('B:P')).length,4);
assert.equal(sharedWork.filter(k=>k==='B:D:1').length,1);
assert.deepEqual(D.contrast.d.slots.filter(s=>(s.tasks||[]).some(t=>t.req==='A')).map(s=>s.n),[1,2,3,4],'A stops after its fourth Decode; t5/t6 add no A work');
const firstBDecode=D.contrast.d.slots.find(s=>(s.tasks||[]).some(t=>t.req==='B'&&t.op==='D'));
assert(firstBDecode.n>D.contrast.transfer.to,'B cannot Decode before the transfer slot');
assert.equal(firstBDecode.gate,'open','B waits for the D readiness gate');
assert.equal(D.contrast.d.slots.find(s=>s.n===D.contrast.transfer.to).gate,'wait','receive area waits at the transfer slot');
console.log('PASS deployment comparison: same A4/B5 work set, A stops after t4, B Decode only after t5 transfer and open gate.');
