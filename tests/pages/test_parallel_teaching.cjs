const assert=require('node:assert/strict'),M=require('../../pages/foundations/parallel-model.js');
for(const kind of ['row','column']){const d=M.tensor(kind);const reconstructed=kind==='row'?d.local[0].map((row,i)=>row.map((v,j)=>v+d.local[1][i][j])):d.local[0].map((row,i)=>[...row,...d.local[1][i]]);assert.deepEqual(reconstructed,[[4,6],[10,12]]);assert.deepEqual(d.result,reconstructed);assert.deepEqual(d.x,[[1,2],[3,4]]);assert.deepEqual(d.w,[[2,0],[1,3]]);}
/* Derive multiply/add counts from the actual X(m×k)·W(k×n) shard shapes,
   not from display strings. k=1 has one product per cell and no addition;
   row split pays 4 merge additions, column split pays 0. Both conserve 8
   multiplications and 4 additions against the unsplit product. */
const opsOf=(x,w)=>{const m=x.length,k=x[0].length,n=w[0].length;return {m,k,n,cells:m*n,mul:m*n*k,add:m*n*(k-1)};};
for(const kind of ['row','column']){
 const d=M.tensor(kind),base=opsOf(d.x,d.w);
 assert.deepEqual(d.total,base,'total ops come from the full X/W shapes');
 assert.equal(d.mergeAdds,kind==='row'?base.cells:0,'row sums one value per output cell; column concatenates');
 let mul=0,adds=0;
 d.shards.forEach((s,i)=>{
  const ops=opsOf(s.x,s.w);
  assert.deepEqual(d.ops[i],ops,'per-card ops come from the real shard shape');
  mul+=ops.mul;adds+=ops.add;
  const first=d.firstCell[i];
  assert.equal(first.terms.length,ops.k,'first cell has one factor per inner-dimension position');
  assert.equal(first.terms.reduce((sum,t)=>sum+t.left*t.right,0),first.value,'displayed first-cell formula reproduces the local value');
  assert.equal(first.value,d.local[i][0][0]);
 });
 assert.equal(mul,base.mul,'cards together perform the same 8 products');
 assert.equal(adds+d.mergeAdds,base.add,'local additions plus merge additions conserve 4');
 if(kind==='row')assert(d.ops.every(o=>o.k===1&&o.add===0),'input-dim shard adds no products locally');
 else assert(d.ops.every(o=>o.k===2&&o.add===2),'output-dim shard performs one addition per output cell');
}
for(const zigzag of [false,true]){const d=M.context(zigzag);assert.deepEqual(d.ranks.flat().sort((a,b)=>a-b),[0,1,2,3,4,5,6,7]);assert.equal(d.work.reduce((a,b)=>a+b),36);assert.equal(d.single,36);if(zigzag)assert.equal(d.work[0],d.work[1]);else assert.ok(d.work[1]>d.work[0]);
 assert.deepEqual(d.owners,zigzag?[0,0,1,1,1,1,0,0]:[0,0,0,0,1,1,1,1],'each Query row has exactly one owner');
 assert.deepEqual(d.copies.map(c=>c.position),[0,1,2,3,4,5,6,7],'one labelled K/V copy per position');
 assert.ok(d.copies.every(c=>c.to===1-c.from&&c.arrive===Number((2+d.localKV[c.from].indexOf(c.position)*.2).toFixed(2))),'copies are bidirectional and arrive in teaching order');
 assert.deepEqual(d.requiredRemoteByRank,zigzag?[[2,3,4,5],[0,1]]:[[],[0,1,2,3]],'only Query rows that need remote K/V wait for it');
 assert.equal(d.queries.reduce((sum,q)=>sum+q.pairs,0),36);
 assert.equal(d.queries[0].readyAt,1.95,'Q1 needs no remote K/V');
 assert.equal(d.queries[7].needRemote,4,'Q8 needs the four remote positions from the other rank');
 assert.equal(d.queries[7].readyAt,2.6);
 assert.equal(d.queries[zigzag?2:4].readyAt,zigzag?2.2:2.6,'the first remote-dependent row waits for its last copy');
 for(const q of d.queries)assert.equal(q.computeEnd,Number((q.computeStart+1).toFixed(2)),'all rows use the same teaching window; pair count is not time');
}
for(const equal of [false,true]){const d=M.decode(equal),direct=d.exp.reduce((s,v,i)=>s+v*d.values[i],0)/d.exp.reduce((a,b)=>a+b);assert.equal(d.result,direct);assert.equal(d.result,equal?4:5);assert.equal(d.weights.reduce((a,b)=>a+b),1);assert.equal(d.reads,4);assert.equal(d.readsPerRank,2);assert.equal(d.merge.numerator,d.numerators[0]+d.numerators[1]);assert.equal(d.merge.denominator,d.sums[0]+d.sums[1]);assert.equal(d.mean,(d.local[0]+d.local[1])/2);const max=Math.max(...d.lse),w=d.lse.map(v=>Math.exp(v-max));assert.ok(Math.abs(d.local.reduce((s,v,i)=>s+v*w[i],0)/w.reduce((a,b)=>a+b)-direct)<1e-12);if(!equal){assert.notEqual(d.mean,d.result);assert.equal(d.mean,4);}}
for(const skew of [false,true]){const d=M.experts(skew);assert.equal(d.loads.reduce((a,b)=>a+b),8);assert.equal(d.selected.length,2);assert.deepEqual(d.selected,skew?[0,1]:[0,2]);assert.deepEqual(d.remote,skew?[0,1,1,2]:[1,1,1,1]);assert.equal(d.crossCombine,d.crossDispatch);assert.equal(d.result,d.outputs.reduce((s,v,i)=>s+v*d.weights[i],0));assert.equal(d.crossDispatch,4);assert.equal(d.crossCombine,4);assert.deepEqual(d.owner,[0,0,1,1]);assert.deepEqual(d.rankLoads,skew?[6,2]:[4,4]);assert.deepEqual(d.dispatchByOrigin,skew?[1,3]:[2,2]);assert.deepEqual(d.outputs,skew?[2,4]:[2,6]);assert.equal(d.result,skew?2.5:3);}
for(const n of [1,2]){const d=M.pipeline(n);assert.equal(d.n,n);assert.equal(d.work,d.single.length);assert.equal(d.single.length,4*n);assert.equal(d.adoptedGpuSlots,2*d.span);assert.equal(d.bubbles,d.adoptedGpuSlots-d.work);assert.equal(d.overlap,2*n-2);assert.equal(d.completion.single,4*n);assert.equal(d.completion.adopted,2*n+2);assert.deepEqual(d.activation.map(a=>a.mb),Array.from({length:n},(_,i)=>i+1));assert.equal(d.stage1.slice(0,2).every(e=>e.kind==='wait'),true,'both stage-1 slots wait before MB1 activation');assert.equal(d.stage0[d.span-1].kind,'drain','stage 0 drains after its last MB');if(n===2){assert.equal(d.span,6);assert.deepEqual(d.stage0.map(e=>e.kind),[ 'work','work','work','work','drain','drain']);assert.deepEqual(d.stage1.map(e=>e.kind),['wait','wait','work','work','work','work']);assert.equal(d.adoptedGpuSlots,12);assert.equal(d.bubbles,4);assert.equal(d.stage0[2].mb,2,'MB2 L1 overlaps MB1 L3 in slot 3');assert.equal(d.stage1[2].mb,1);assert.equal(d.stage0[3].mb,2);assert.equal(d.stage1[3].mb,1);}else{assert.equal(d.span,4);assert.deepEqual(d.stage0.map(e=>e.kind),['work','work','drain','drain']);assert.deepEqual(d.stage1.map(e=>e.kind),['wait','wait','work','work']);assert.equal(d.bubbles,4);}}
const seq=M.sequence(4);assert.deepEqual(seq.tokens,['p1','p2','p3','p4']);assert.deepEqual(seq.shards,[['p1','p2'],['p3','p4']]);assert.equal(seq.rowsPerRank,2);assert.equal(seq.replicatedPerRank,4);assert.equal(seq.replicasBefore,8);assert.equal(seq.replicasAfter,4);
console.log('PASS TP reconstruction, CP single/adopted coverage and balance, DCP weighted merge vs local mean, EP rank load and cross-card dispatch, SP row replicas.');
