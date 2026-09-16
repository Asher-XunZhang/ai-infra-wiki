/* Causal invariants and source-derived examples; no GPU or network simulation. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const api = require('../pages/sglang/pd-prefill-lifecycle/scenario-engine.js');
const html = fs.readFileSync(path.join(__dirname, '../pages/sglang/pd-prefill-lifecycle/index.html'), 'utf8');
const byKey = (m, key) => m.frames.find(f => f.key === key);
const completed = m => m.frames.at(-1).after;
const ranges = m => m.frames.filter(f => f.type === 'cut').map(f => f.work.batch.segments.map(p => [p.rid,p.start,p.end]));
let scenarios = 0, checkedFrames = 0;

function validate(model) {
  scenarios++;
  const {frames,config} = model;
  assert.equal(new Set(frames.map(f=>f.key)).size,frames.length);
  assert.deepEqual(model.events.flatMap(e=>e.frames),frames,'Compression must retain every operation');
  for(let i=0;i<frames.length;i++) {
    const f=frames[i]; checkedFrames++;
    assert.deepEqual(f.before,i?frames[i-1].after:model.snapshot(-1),'Seeking must have continuous snapshots');
    for(const id of f.active)assert(api.nodes[id],`${f.key}: unknown module`);
    for(const link of f.routes)assert(api.nodes[link.from]&&api.nodes[link.to]);
    for(let r=0;r<3;r++)for(const req of model.requests){
      const a=f.before.ranks[r].requests[req.rid],q=f.after.ranks[r].requests[req.rid];
      assert(q.sendEnd<=q.cacheEnd&&q.cacheEnd<=q.selectedEnd&&q.selectedEnd<=req.length,f.key);
      assert(q.sendEnd>=a.sendEnd&&q.cacheEnd>=a.cacheEnd,'Cursors must never go backwards');
      if(q.token)assert.equal(q.cacheEnd,req.length,'Intermediate chunks must not publish t0');
      if(q.cacheEnd>a.cacheEnd){assert(q.holdsKv&&q.metadata);if(r)assert(f.before.ranks[r-1].requests[req.rid].cacheEnd>=q.cacheEnd,'Upstream activation must exist');}
      if(q.sendEnd>a.sendEnd)assert(q.holdsKv&&q.metadata&&!q.released,'Submission must retain source references');
      if(q.released&&!a.released){
        if(q.cacheEnd){assert(q.permit&&f.after.release.includes(req.rid),f.key);assert(['Success','Failed'].includes(a.sender)||f.key.includes('retry'),f.key);}
        else assert(f.after.bootstrap.bad.includes(req.rid),'Only a bad bootstrap conclusion can retire before compute');
      }
      if(q.released)assert(!q.metadata&&!q.holdsKv&&q.queue==='done');
      if(q.reason==='FINISH_ABORT'&&q.cacheEnd)assert.equal(q.sender,'Failed','Failure cleanup does not call sender.clear');
      if(q.reason==='FINISH_LENGTH(0)')assert.equal(q.sender,'Cleared');
    }
    if(f.type==='cut'){
      const b=f.work.batch;
      assert(b.segments.length<=config.batchSize);
      if(config.chunkSize)assert(b.charge<=config.chunkSize,'Shared budget must not be multiplied by batch size');
      assert.equal(b.tokenCount,b.segments.reduce((n,p)=>n+p.end-p.start,0));
      for(const p of b.segments){assert.equal(p.start,f.before.ranks[0].requests[p.rid].cacheEnd);assert(p.end>p.start);if(!p.last)assert.equal(p.end%config.pageSize,0);}
    }
    if(f.type==='consensus'&&f.after.terminal&&f.work.rank!==undefined){
      const c=f.after.terminal;
      for(const rid of c.intersection)for(let r=0;r<=c.through;r++)assert(c.local[r].includes(rid));
    }
  }
  for(const req of completed(model).requests){
    assert.notEqual(req.outcome,'active');
    assert(completed(model).ranks.every(r=>r.requests[req.rid].released));
    if(req.outcome==='success')assert(completed(model).ranks.every(r=>r.requests[req.rid].sendEnd===req.length));
  }
  const first=model.snapshot(0);model.snapshot(frames.length-1).ranks[0].requests.R0.cacheEnd=999;
  assert.deepEqual(model.snapshot(0),first,'Returned snapshots must be independent');
}

const chunks=api.createScenario(),single=api.createScenario(api.presets.single),batching=api.createScenario(api.presets.batching);
assert.deepEqual(ranges(chunks),[[['R0',0,4]],[['R0',4,8]],[['R0',8,12]]]);
assert.deepEqual(ranges(single),[[['R0',0,12]]]);
assert.deepEqual(ranges(batching),[[['R0',0,6],['R1',0,2]],[['R1',2,6],['R2',0,4]],[['R2',4,6]]]);
for(const m of [single,chunks,batching]){
  for(const type of ['budget','cut','pack'])assert(m.events.some(e=>e.frames[0].type===type),'Cut/pack must occur in the visible main flow');
  assert(m.frames.findIndex(f=>f.key==='B1-send-0')<m.frames.findIndex(f=>f.key==='B1-result-1'),'No invented wait-all-results barrier');
}
assert.equal(chunks.events.filter(e=>e.compressed).length,2,'Only subsequent chunks are compressed by default');
for(const group of chunks.events.filter(e=>e.compressed))for(const type of ['cut','pack','forward','proxy','result','send'])assert(group.frames.some(f=>f.type===type));
assert.deepEqual(api.createScenario({...api.defaults,detail:'all'}).frames,chunks.frames,'Detail mode must not alter causal behavior');
const rounded=api.createScenario({requests:2,inputLength:3,pageSize:2,chunkSize:6,batchSize:2});
assert.deepEqual(ranges(rounded),[[['R0',0,3],['R1',0,2]],[['R1',2,3]]],'New requests are billed by page, not raw token count');
assert.equal(byKey(rounded,'B1-cut').work.batch.charge,6);

const boot=api.createScenario(api.presets.bootstrap);
assert.deepEqual(byKey(boot,'bootstrap-1-2').after.bootstrap.bad,['R0','R1']);
assert.deepEqual(byKey(boot,'bootstrap-1-2').after.bootstrap.good,['R2']);
assert.deepEqual(ranges(boot),[[['R2',0,4]]],'Failed handshake requests must not compute');
const allBad=api.createScenario({...api.presets.bootstrap,failed:[['R0'],['R1'],['R2']]});
assert.equal(allBad.batches,0);
const abort=api.createScenario(api.presets.abort),ab=byKey(abort,'bootstrap-1-2').after;
assert.equal(ab.ranks[1].requests.R0.sender,'WaitingForInput');
assert(ab.bootstrap.bad.includes('R0')&&!ab.bootstrap.good.includes('R0'),'Cancellation is bad even when poll remains ready');
const wait=api.createScenario(api.presets.waiting),parked=byKey(wait,'bootstrap-wait').after;
assert(parked.ranks.every(r=>r.requests.R0.queue==='bootstrap'&&!r.requests.R0.metadata));
assert(parked.ranks.every(r=>r.requests.R1.queue==='waiting'));

const transfer=api.createScenario(api.presets.transfer),partial=byKey(transfer,'B1-terminal-2').after;
assert.deepEqual(partial.terminal.intersection,['R1']);
assert.equal(partial.ranks[1].requests.R0.sender,'Failed');
assert.equal(partial.ranks[2].requests.R0.sender,'Transferring');
const held=byKey(transfer,'B1-wait').after;
assert(held.ranks.every(r=>r.requests.R0.holdsKv&&r.requests.R0.metadata&&!r.requests.R0.released));
assert(held.ranks.every(r=>r.requests.R1.released));
assert.equal(held.ranks[0].requests.R1.reason,'FINISH_LENGTH(0)');
assert.equal(held.ranks[2].requests.R1.reason,'FINISH_ABORT','Mixed local outcomes must not be painted as global success');
assert.deepEqual(completed(transfer).requests.map(r=>r.outcome),['transfer-failed','transfer-failed','success']);
const repoll=api.createScenario(api.presets.repoll),guard=byKey(repoll,'B3-finish-1').after.ranks[1].requests.R0;
assert(guard.holdsKv&&guard.metadata&&!guard.released&&!guard.permit,'A release list cannot override a transient local poll');
assert.equal(guard.queue,'inflight');

for(const invalid of [{inputLength:0},{requests:5},{batchSize:0},{chunkSize:3,pageSize:2},{chunkSize:1,pageSize:4},{pageSize:3},{fault:'made-up'},{fault:'transfer_fail',failed:[[],[],[]]}])assert.throws(()=>api.createScenario(invalid));
assert.equal((html.match(/class="life-step"/g)||[]).length,12);
for(const id of Object.keys(api.nodes))assert(html.includes(`data-node="${id}"`));
for(const model of [chunks,single,batching,rounded,boot,allBad,abort,wait,transfer,repoll])validate(model);
// Bounded parameter sweep includes short final pages, queue capacity, and several chunk sizes.
for(let i=0;i<36;i++)validate(api.createScenario({requests:i%4+1,inputLength:[1,3,5,12,17,31][i%6],pageSize:[1,2,4][i%3],chunkSize:i%5===0?0:[1,2,4][i%3]*(i%4+1),batchSize:(i*3)%4+1}));
console.log(`PASS: ${scenarios} scenarios / ${checkedFrames} frames; shared budgets, complete ranges, compression equivalence, per-rank partial failures, local repoll guards, ownership and reverse seeking.`);
