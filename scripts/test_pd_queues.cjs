'use strict';
const assert=require('node:assert/strict');
const Q=require('../pages/sglang/pd-dataflow/queue-model.js');
const html=require('node:fs').readFileSync(require('node:path').join(__dirname,'../pages/sglang/pd-dataflow/index.html'),'utf8');
for(const match of html.matchAll(/https:\/\/github\.com\/sgl-project\/sglang\/blob\/([a-f0-9]{40})\/python\/sglang\/srt\/([^"<>]+)#L(\d+)/g)){
  assert.equal(match[1],Q.commit);
  assert(Object.values(Q.sources).some(([file,line])=>file===match[2]&&line===Number(match[3])),'HTML source link must have an audited queue anchor');
}
let count=0;
const find=(model,key)=>{const frame=model.frames.find(f=>f.key===key);assert(frame,`${model.id}: missing ${key}`);return frame;};
for(const {id} of Q.scenarios)for(let rank=0;rank<3;rank++){
  const m=Q.create(id,rank),seen=new Set();
  for(const [i,f] of m.frames.entries()){
    assert(!seen.has(f.key));seen.add(f.key);assert(Q.sources[f.source]);
    if(i)assert.deepEqual(f.before,m.frames[i-1].after);
    const s=f.after;
    assert(+s.bootstrap + +s.waiting + +s.inflight<=1,'request queues exclusive in the scoped non-optimistic example');
    if(s.waiting)assert(s.metadata&&!s.pending&&!s.kv);
    if(s.chunk)assert(s.metadata&&!s.pending&&!s.inflight);
    if(s.pendingIds)assert(s.kv&&!s.exit);
    if(s.exit)assert(!s.bootstrap&&!s.waiting&&!s.inflight&&!s.kv&&!s.metadata);
    if(f.kind==='wait')assert.equal(f.before[f.from],s[f.to],'blocked branch must preserve membership');
    if(f.before.kv&&!s.kv)assert(['Success','Failed'].includes(s.poll),'no premature resource release');
    count++;
  }
  if(['abort','bootstrap-fail'].includes(id)){
    assert(m.frames.every(f=>!f.after.waiting&&!f.after.batch&&!f.after.kv));continue;
  }
  const final=find(m,'finalize');assert(final.after.bootstrap&&!final.after.pending&&final.after.metadata,'queue alone is not state');
  const appended=find(m,'append-inflight');assert(appended.after.batch&&appended.after.inflight,'old batch and inflight can alias same Req');
  assert(find(m,'retire').after.inflight,'resource cleanup precedes list rebuild');
  assert(find(m,'remove-inflight').after.batch,'do not invent immediate old-batch removal');
  assert(!m.frames.at(-1).after.batch);
  if(['chunks','skip-chunk-output'].includes(id)){
    assert(find(m,'B1-select').after.chunk&&find(m,'B1-select').after.batch);
    for(const b of ['B1','B2']){const s=find(m,`${b}-filter`).after;assert(s.chunk&&!s.batch&&!s.waiting&&s.kv&&s.pendingIds);}
    const last=find(m,'B3-continue').after;assert(last.batch&&!last.chunk&&!last.waiting);
    assert(!find(m,'send-final').after.pendingIds);
    assert(m.frames.filter(f=>f.after.waiting&&!f.before.waiting).length===1,'ordinary chunk continuation must never requeue');
  }
  if(id==='repoll'&&rank===1){const s=find(m,'repoll').after;assert(s.inflight&&s.kv&&s.metadata);}
  if(id==='transfer-fail')assert.equal(find(m,'retire').after.finished,rank===1?'中止':'完成');
}
assert.equal(Q.create('invalid').id,'normal');
if(process.env.SGLANG_SOURCE_ROOT){
  const {execFileSync}=require('node:child_process');
  for(const [key,[file,line,needle]] of Object.entries(Q.sources)){
    const lines=execFileSync('git',['-C',process.env.SGLANG_SOURCE_ROOT,'show',`${Q.commit}:python/sglang/srt/${file}`],{encoding:'utf8'}).split('\n');
    assert(lines[line-1].includes(needle),`${key}: ${file}:${line}: ${lines[line-1]}`);
  }
}
console.log(`Queue walkthrough: 12 scenarios × 3 ranks / ${count} frames; aliasing, retention, chunk continuation and cleanup passed.${process.env.SGLANG_SOURCE_ROOT?' Pinned anchors verified.':' Source audit not requested.'}`);
