const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const M=require('../../pages/sglang/advanced-generation/execution-model.js');
let frames=0,cases=0;
for(const mode of Object.keys(M.modes))for(const condition of Object.keys(M.modes[mode].cases)){
 const m=M.build(mode,condition),by=id=>m.events.find(e=>e.id===id);cases++;
 for(const e of m.events){assert.ok(e.end>e.start);assert.ok(M.sources[e.source]);for(const dep of e.deps)assert.ok(by(dep).end<=e.start+1e-7,e.id+' depends on '+dep);if(e.waitFor)assert.equal(e.end,by(e.waitFor).end);}
 for(const lane of m.lanes){const es=m.events.filter(e=>e.lane===lane.id).sort((a,b)=>a.start-b.start);for(let i=1;i<es.length;i++)assert.ok(es[i].start>=es[i-1].end-1e-7,'exclusive lane '+lane.id);}
 const stable=t=>JSON.stringify(m.sample(t),(key,value)=>typeof value==='function'?undefined:value);
 for(let t=0;t<m.duration;t+=.13){const before=stable(t);m.sample(m.duration);assert.equal(stable(t),before);frames++;}
 const end=m.sample(m.duration);
 if(mode==='speculative'){
  assert.deepEqual(end.output.slice(0,-1),end.validKV.slice(1));assert.equal(end.output.at(-1),'z');assert.equal(end.validKV[0],'r');assert.ok(!end.validKV.includes('z'));
  assert.equal(end.cpuLen,8+end.output.length);assert.equal(end.allocated,5,'rejection does not free all reservations');
  const atSelect=m.sample(by('select').end);assert.equal(atSelect.cpuLen,8);assert.equal(atSelect.gpuLen,8+m.data.n+1);assert.deepEqual(atSelect.output,[]);
  assert.ok(by('next').start<by('copy').end&&by('next').end>by('commit').start,'next Draft overlaps current CPU settlement');
  if(condition==='tree')assert.deepEqual(end.acceptIndex,[0,1,3]);
 }
 if(mode==='grammar'){
  if(condition==='invalid'){assert.equal(end.queue,'abort');assert.equal(by('sample'),undefined);}
  else{assert.equal(end.r1,m.data.chosen);assert.equal(end.r2,'');assert.equal(end.shared,true);assert.ok(by('mask').start<by('forward').end,'CPU mask may overlap async forward even without scheduler overlap');}
  if(condition==='cold')assert.ok(by('compile').start<by('other').end);
  if(condition==='hit')assert.equal(by('compile'),undefined);
 }
 if(mode==='combined'){
  const restored=m.sample(by('restore').end);assert.equal(restored.matcher,'{');assert.equal(restored.advanced,false);assert.deepEqual(restored.rows,[0,1,2,3]);assert.deepEqual(restored.output,[]);
  assert.deepEqual(end.output,['"ok"',':','true']);assert.equal(end.matcher,'{"ok":true');assert.equal(end.advanced,true);
  assert.ok(by('barrier').start<by('target').end);assert.ok(by('sample').start>=by('target').end);assert.ok(by('sample').start>=by('restore').end);
  if(condition==='late'){assert.ok(by('wait').start>=by('target').start);assert.ok(by('maskWait').end>by('target').end);}
 }
}
if(process.env.SGLANG_SOURCE_DIR)for(const [key,[file,line,symbol,kind]] of Object.entries(M.sources)){
 const code=execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',M.revision+':python/sglang/srt/'+file],{encoding:'utf8'});
 assert.match(code.split('\n')[line-1],new RegExp((kind||'def')+' '+symbol+'[( :]'),key);
}
console.log('PASS '+cases+' cases, '+frames+' reversible snapshots; dependencies, exclusive resources, KV/output offsets, grammar isolation and overlapping clocks; fixed anchors '+(process.env.SGLANG_SOURCE_DIR?'checked':'NOT_EXECUTED'));
