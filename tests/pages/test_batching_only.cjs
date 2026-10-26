const assert=require('node:assert/strict');
const b=require('../../pages/motion/dynamic-batching.js');
const c=require('../../pages/motion/batching-chunk.js');

// B01: the comparison panel must use the same simulate() output as the animation.
const expected={
 0:{batches:[[['A'],0,[0]],[['B'],2,[0]],[['C'],4,[0]]]},
 3:{batches:[[['A','B'],3,[3,1]],[['C'],7,[3]]]},
 5:{batches:[[['A','B','C'],4,[4,2,0]]]}
};
for(const delay of [0,3,5]){
 const run=b.comparison([delay])[0];
 assert.equal(run.batchCount,expected[delay].batches.length);
 assert.deepEqual(run.batches.map(x=>[x.ids,x.time,x.waits]),expected[delay].batches);
 assert.deepEqual(run.frames.at(-1).batches,run.batches);
 for(const id of ['A','B','C']){
  const info=run.byRequest[id];
  assert.equal(info.dispatch-info.at,info.wait);
  assert.ok(info.wait>=0&&info.wait<=delay);
 }
 assert.deepEqual(run.batches.flatMap(x=>x.ids),['A','B','C']);
}
assert.deepEqual(b.comparison([0,3,5]).map(r=>r.batchCount),[3,2,1]);
assert.deepEqual(b.comparison([0,3,5]).map(r=>r.byRequest.A.wait),[0,3,4]);

// B02: both plans keep B1-B8 and A1-A8; only the Prefill grouping changes.
for(const chunk of [1,2,4,8]){
 const p=c.plan(chunk);
 assert.equal(p.prefillRounds,8/chunk);
 assert.equal(p.aDecodes,8);
 assert.equal(p.bPositions,8);
 assert.deepEqual(p.rounds.map(r=>r.a),[1,2,3,4,5,6,7,8]);
 assert.equal(p.rounds.length,8);
 assert.equal(p.fragments.length,8/chunk);
 assert.deepEqual(p.fragments.flatMap(f=>f.positions),[1,2,3,4,5,6,7,8]);
 assert.ok(p.fragments.every(f=>f.positions.length===chunk&&f.to-f.from+1===chunk));
 assert.equal(p.aBeforeBFinish,8/chunk);
 assert.equal(p.aAfterBFinish,8-8/chunk);
 assert.equal(p.rounds.filter(r=>r.fragment).length,8/chunk);
 assert.equal(p.rounds.filter(r=>!r.fragment).length,8-8/chunk);
 const frames=c.create(chunk).frames;
 assert.equal(frames.length,1+2*(8/chunk));
 assert.match(frames.at(-1).caption,/B 的 Prefill 完成/);
}
const whole=c.compare(8);
assert.deepEqual(whole.whole,whole.selected,'chunk 8 is self-consistent with the whole-8 baseline');
assert.equal(whole.same,true);
const four=c.compare(4);
assert.equal(four.whole.prefillRounds,1);
assert.equal(four.selected.prefillRounds,2);
assert.notDeepEqual(four.whole,four.selected);
assert.equal(four.selected.fragments[0].from,1);
assert.equal(four.selected.fragments.at(-1).to,8);
assert.match(c.create(8).frames.at(-1).caption,/A 还有 7 次 Decode/);
assert.match(c.create(1).frames.at(-1).caption,/A 的 8 次 Decode 也恰好/);
console.log('PASS batching-only: B01 0/3/5 waits+batch counts from simulate; B02 A8/B8 conservation, chunk partition, whole-8 self-consistency.');
