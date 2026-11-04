const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process'),M=require('../../pages/sglang/kv-memory/model.js');
for(const target of [8,9,11])for(let step=0;step<=4;step++){
 const d=M.extend(target,step);assert.equal(d.newPages,target===8?0:1);assert.equal(new Set(d.mapping).size,d.mapping.length);assert.deepEqual(d.mapping.slice(0,6),[4,5,6,7,28,29]);assert.equal(d.mapping.length,step?target:6);assert.equal(d.written,step>=3?target:step>=2?8:6);assert.equal(d.ownedPages,2+Number(step>0&&target>8));assert.equal(d.pages.filter(p=>!p.free).length,d.ownedPages);assert.equal(d.pages.flatMap(p=>p.cells).filter(c=>c.state==='written').length,d.written);assert.equal(d.pages.flatMap(p=>p.cells).filter(c=>c.state==='reserved').length,d.mapping.length-d.written);assert.ok(d.mapping.every(s=>Math.floor(s/4)>0));
}assert.deepEqual(M.extend(11,1).added,[30,31,12,13,14]);assert.equal(M.extend(9,4).mapping[8],12);assert.equal(M.extend(8,4).pages[2].free,true);assert.throws(()=>M.extend(7));
for(const common of [3,6,8])for(const ns of [true,false])for(let step=0;step<4;step++){
 const d=M.prefix(common,ns,step);assert.equal(d.request.length,9);assert.equal(d.request.at(-1),'Z');assert.equal(d.hit,ns?({3:0,6:4,8:8})[common]:0);assert.equal(d.hit%4,0);assert.equal(d.recompute+d.hit,9);assert.equal(d.reused.length,step===3?d.hit:0);assert.equal(d.newCopies,0);assert.ok(d.reused.every(s=>[8,9,10,11,20,21,22,23].includes(s)));
}assert.deepEqual(M.prefix(6,true,3).reused,[8,9,10,11]);assert.deepEqual(M.prefix(8,false,3).reused,[]);
for(let step=0;step<=6;step++){const d=M.sharing(step);assert.equal(d.locks,Number(d.r1)+Number(d.r2));assert.equal(d.pages.length,3);assert.equal(d.bufferPages,3);assert.equal(d.freePages+d.pages.filter(p=>p.state!=='free').length,3);assert.equal(d.evictable,d.cached&&d.locks===0);assert.equal(d.pages.filter(p=>p.state==='private').length,d.locks);if(d.locks)assert.equal(d.pages[0].state,'protected');}
assert.equal(M.sharing(3).r1,false);assert.equal(M.sharing(3).r2,true);assert.equal(M.sharing(3).locks,1);assert.equal(M.sharing(4).evictAttempt,true);assert.equal(M.sharing(4).freePages,M.sharing(3).freePages);assert.equal(M.sharing(5).cached,true);assert.equal(M.sharing(5).freePages,2);assert.equal(M.sharing(6).freePages,3);
// The continuous model must preserve causal boundaries under arbitrary seeking.
const at=(tl,phase,t)=>M.sample(tl,tl.phases[phase].start+t).state;
const cell=(s,n)=>s.pages.flatMap(p=>p.slots).find(c=>c.slot===n);
for(const target of [8,9,11]) {
 const tl=M.timeline('extend',{target}),added=M.extend(target,1).added;
 const before=at(tl,1,1.09),reserved=at(tl,1,1.1),mapped=at(tl,1,2.1);
 assert.deepEqual(before.outLoc,[]);assert.equal(before.rows.R1.mapping.length,6);
 assert.deepEqual(reserved.outLoc,added);assert.equal(reserved.rows.R1.mapping.length,6);
 assert.ok(added.every(n=>cell(reserved,n).reserved&&!cell(reserved,n).valid));
 assert.deepEqual(mapped.rows.R1.mapping,[4,5,6,7,28,29,...added]);
 assert.equal(mapped.freePages.length,target===8?1:0);
 assert.equal(mapped.heldEmpty,target===9?3:target===11?1:0);
 assert.ok(added.every(n=>!cell(at(tl,2,2.34),n).valid));
 const written=at(tl,2,2.35);
 assert.ok(added.every((n,i)=>cell(written,n).valid&&cell(written,n).value===String(i+6)));
 assert.equal(M.sample(tl,tl.duration).state.result,`K${target-1} / V${target-1}`);
}
for(const hit of [0,4,8]) {
 const tl=M.timeline('reuse',{hit}),initial=M.sample(tl,0).state,last=M.sample(tl,tl.duration).state;
 assert.deepEqual(last.rows.R2.mapping,[8,9,10,11,20,21,22,23].slice(0,hit));
 assert.deepEqual(last.pages,initial.pages,'copying indices must not create, relocate or overwrite KV');
 assert.equal(last.pages.flatMap(p=>p.slots).filter(c=>c.valid).length,8);
}
const shared=M.timeline('sharing');
const firstReturn=at(shared,3,1.15);
assert.deepEqual(firstReturn.freePages,[3]);assert.equal(firstReturn.rows.R1.released,false);
assert.equal(firstReturn.rows.R1.releasing,true);assert.equal(firstReturn.rows.R1.mapping.length,6);
assert.equal(cell(firstReturn,12).valid,false);assert.equal(cell(firstReturn,12).value,'4');
assert.equal(cell(firstReturn,8).valid,true);assert.equal(firstReturn.freeRows,0);
assert.equal(at(shared,3,2.05).freeRows,1);assert.equal(at(shared,3,2.05).rows.R1.released,true);
assert.deepEqual(at(shared,4,1.2).freePages,[3],'R2 still references the shared page');
assert.deepEqual(at(shared,5,2.05).freePages,[3,4],'cache ownership outlives both request rows');
const evicted=at(shared,6,1.1);
assert.deepEqual(evicted.freePages,[2,3,4]);assert.equal(evicted.freeRows,2);
assert.equal(cell(evicted,8).value,'0');assert.equal(cell(evicted,8).valid,false,'free is not zero-fill');
const reclaimed=at(shared,7,.95);
assert.equal(cell(reclaimed,8).reserved,true);assert.equal(cell(reclaimed,8).valid,false);
assert.equal(cell(reclaimed,8).value,'0');assert.deepEqual(reclaimed.rows.R3.mapping,[8]);
const replaced=at(shared,7,2.55);
assert.equal(cell(replaced,8).value,'Z');assert.equal(cell(replaced,8).valid,true);
assert.equal(cell(replaced,9).value,'1');assert.equal(cell(replaced,9).valid,false,'unused old contents stay invalid');
for(const tl of [M.timeline('extend'),M.timeline('reuse'),shared]) {
 const points=tl.phases.flatMap(p=>p.events.map(e=>p.start+e.at+.001));
 const snapshots=points.map(t=>JSON.stringify(M.sample(tl,t)));
 for(let i=points.length-1;i>=0;i--){assert.equal(JSON.stringify(M.sample(tl,points[i])),snapshots[i],'reverse seek restores every state');}
 for(let t=0;t<=tl.duration;t+=.031){
  const {state:s,flows}=M.sample(tl,t);
  assert.equal(s.pages.length,tl.mode==='reuse'?2:3);
  assert.equal(s.pages.length,s.pages.filter(p=>p.owner).length+s.freePages.length);
  for(const page of s.pages)for(const c of page.slots){assert.ok(!(c.valid&&c.reserved));if(!page.owner)assert.equal(c.valid,false);}
  for(const r of Object.values(s.rows)){if(r.released)assert.deepEqual(r.mapping,[]);else if(!r.releasing)assert.ok(r.mapping.every(n=>cell(s,n).valid||cell(s,n).reserved));}
  assert.ok(flows.every(f=>f.progress>=0&&f.progress<1));
 }
 const mutated=M.sample(tl,tl.duration);mutated.state.pages[0].slots[0].value='bad';
 assert.notEqual(M.sample(tl,tl.duration).state.pages[0].slots[0].value,'bad');
}
if(process.env.SGLANG_SOURCE_DIR)for(const [key,[file,line,symbol]] of Object.entries(M.sources)){const code=execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',`${M.revision}:${file}`],{encoding:'utf8'});assert.ok(code.split('\n')[line-1].includes(symbol),`${key}: line ${line}: ${code.split('\n')[line-1]}`);}
console.log(`PASS: tail-page addresses, allocation/write separation, aligned reuse and namespace, shared ownership, protected eviction and pool conservation; ${process.env.SGLANG_SOURCE_DIR?Object.keys(M.sources).length:0} source anchors.`);
