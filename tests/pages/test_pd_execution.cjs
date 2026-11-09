const assert=require('node:assert/strict');
const T=require('../../pages/sglang/pd-execution/timeline.js'),D=require('../../pages/sglang/pd-dataflow/model.js'),L=require('../../pages/sglang/pd-prefill-lifecycle/scenario-engine.js');
const suites=[['dataflow',D.scenarios.map(s=>({...D.create(s.id),sources:D.sources}))],['lifecycle',Object.values(L.presets).map(p=>L.createScenario(p))]];
let samples=0;
for(const [kind,models]of suites)for(const m of models){
 const tl=T.build(m,kind),byId=Object.fromEntries(tl.events.map(e=>[e.id,e]));
 assert.equal(Object.keys(byId).length,tl.events.length);
 for(const e of tl.events){assert.ok(T.lanes.some(l=>l.id===e.lane));assert.ok(e.end>e.start);for(const d of e.deps)assert.ok(byId[d].end<=e.start+1e-8,`${e.id} starts before ${d} completes`);assert.ok(tl.sourceUrl(e).includes(T.revisions[kind]));}
 const points=new Set([0,tl.duration,...tl.events.flatMap(e=>[e.start,(e.start+e.end)/2,e.end])]),saved=[];
 for(const t of points){const s=tl.sample(t);saved.push([t,JSON.stringify(s.state)]);
  for(const req of m.requests??[{rid:'R0'}]){
   const rows=T.rows(tl,s,req.rid);
   rows.forEach((q,r)=>{assert.ok(q.completed<=q.submitted&&q.submitted<=q.computed,`${kind}/${m.id??m.config.fault} ${t}: completion bounds`);
    for(const c of s.active.filter(e=>e.type==='kv'&&e.rank===r&&e.requests.some(x=>x.rid===req.rid)))assert.ok(q.kv,`${kind}/${c.id} reads released KV`);
    const refs=T.references(tl,s,r,req.rid);assert.ok(+refs.bootstrap + +refs.waiting + +refs.inflight<=1);if(refs.chunk)assert.ok(q.kv);
   });
  }samples++;
 }
 for(const [t,state]of saved.reverse())assert.equal(JSON.stringify(tl.sample(t).state),state);
 for(const lane of T.lanes){const events=tl.events.filter(e=>e.lane===lane.id).sort((a,b)=>a.start-b.start);for(let i=1;i<events.length;i++)assert.ok(events[i-1].end<=events[i].start+1e-8,`${lane.id} overlaps itself`);}
}
const chunks=T.build({...D.create('chunks'),sources:D.sources},'dataflow');
assert.ok(chunks.copies.some(c=>chunks.events.some(e=>e.type==='compute'&&e.id.startsWith('B2')&&e.start<c.end&&c.start<e.end&&c.id.includes('B1'))),'previous chunk transfer must overlap next chunk compute');
assert.ok(chunks.copies.some(a=>chunks.copies.some(b=>a.rank!==b.rank&&a.start<b.end&&b.start<a.end)),'independent workers overlap');
const prep=chunks.events.find(e=>e.id==='B1-allocate-1'),gpu=chunks.events.find(e=>e.id==='B1-compute-0');
assert.ok(prep.start<gpu.end&&gpu.start<prep.end,'downstream local preparation need not wait for upstream GPU completion');
assert.equal(chunks.sample(prep.end).state.ranks[0].computed,0,'out-of-order preparation commits cannot prematurely copy the old serial GPU snapshot');
const append=chunks.events.find(e=>e.id==='B3-result-0');const refs=T.references(chunks,chunks.sample(append.end),0);assert.ok(refs.batch&&refs.inflight&&!refs.chunk,'same local Req can have both batch and inflight references');
console.log(`PASS: ${samples} timed snapshots; completion boundaries, held KV, independent workers, dependencies, references and reverse seek.`);
