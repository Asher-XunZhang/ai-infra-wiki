'use strict';
const assert=require('node:assert/strict');
const M=require('../pages/sglang/pd-dataflow/model.js');
let count=0;
for(const {id} of M.scenarios){
  const {frames}=M.create(id);const keys=new Set();
  for(const f of frames){
    assert(!keys.has(f.key),`${id}: duplicate ${f.key}`);keys.add(f.key);assert(M.sources[f.source]);
    for(let r=0;r<3;r++){
      const q=f.after.ranks[r],p=f.before.ranks[r];
      assert(q.sent<=q.computed && q.computed<=12,`${id}/${f.key}: sent beyond computed`);
      assert(q.computed>=p.computed && q.sent>=p.sent,`${id}/${f.key}: boundary regressed`);
      if(q.queue==='bootstrap')assert(q.pending);
      if(['batch','chunked','inflight'].includes(q.queue)){assert(q.kv);assert(q.metadata);assert(!q.pending);}
      if(q.token)assert.equal(q.computed,12);
      if(q.queue==='done'){assert.equal(q.poll,'Success');assert(!q.kv&&!q.metadata);}
      if(f.key.startsWith('cleanup-'))assert.deepEqual(f.before.release,['R0']);
      if(p.kv&&!q.kv)assert(['Success','Failed'].includes(p.poll));
    }
    count++;
  }
  frames.slice(1).forEach((f,i)=>assert.deepEqual(f.before,frames[i].after));
  assert(frames.at(-1).after.ranks.every(q=>['done','aborted'].includes(q.queue)));
}
const frame=(id,key)=>M.create(id).frames.find(f=>f.key===key);
for(const polls of [['Success','Failed','Success'],['Failed','Failed','Failed']])assert.deepEqual(M.terminalSet(polls),['R0']);
assert.deepEqual(M.terminalSet(['Success','Failed','Transferring']),[]);
assert.deepEqual(M.consensus(['WaitingForInput','WaitingForInput','WaitingForInput'],[1]),{good:[],bad:['R0']});
assert.deepEqual(M.consensus(['WaitingForInput','Bootstrapping','WaitingForInput']),{good:[],bad:[]});
assert.equal(frame('bootstrap-wait','poll-bootstrap').after.ranks[1].poll,'Bootstrapping');
assert.equal(frame('bootstrap-fail','poll-bootstrap').after.ranks[1].poll,'Failed');
for(const id of ['bootstrap-fail','abort'])assert(M.create(id).frames.every(f=>f.after.ranks.every(q=>!q.kv&&!q.token)));
assert(frame('metadata','metadata-blocked').after.ranks.every(q=>q.queue==='bootstrap'&&!q.metadata));
assert(frame('budget','budget-blocked').after.ranks.every(q=>q.queue==='waiting'&&q.metadata&&!q.kv));
assert(frame('output-wait','d2h-wait').after.ranks.every(q=>!q.token&&q.kv));
for(const id of ['transfer-wait','transfer-fail'])assert(frame(id,'transfer-blocked').after.ranks.every(q=>q.kv&&q.metadata));
assert.equal(frame('transfer-fail','cleanup-1').after.ranks[1].queue,'aborted');
const blocked=frame('repoll','repoll-blocked').after.ranks[1];assert(blocked.kv&&blocked.metadata&&blocked.queue==='inflight');
const chunk=frame('chunks','B1-send-0').after.ranks[0];assert.equal(chunk.sent,4);assert(!chunk.token&&chunk.kv);
assert(!frame('skip-chunk-output','B1-output-0').flow);assert.equal(frame('skip-chunk-output','B3-output-0').flow.kind,'output');
assert.equal(frame('normal','B1-send-0').after.ranks[1].sent,0,'no artificial all-rank send barrier');
// Optional offline source audit against the exact public commit; no SGLang import or checkout mutation.
if(process.env.SGLANG_SOURCE_ROOT){
  const {execFileSync}=require('node:child_process');
  for(const [key,[file,line,text]] of Object.entries(M.sources)){
    const source=execFileSync('git',['-C',process.env.SGLANG_SOURCE_ROOT,'show',`${M.commit}:python/sglang/srt/${file}`],{encoding:'utf8'}).split('\n');
    assert(source[line-1].includes(text),`${key}: source anchor drift at ${file}:${line}: ${source[line-1]}`);
  }
}
console.log(`${M.scenarios.length} scenarios / ${count} snapshots: state, ownership, wait, failure and chunk checks passed.${process.env.SGLANG_SOURCE_ROOT?' Pinned source anchors verified.':' Source anchor audit not requested.'}`);
