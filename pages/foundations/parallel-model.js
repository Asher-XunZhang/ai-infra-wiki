/* Mechanism arithmetic, not measured speed or any engine's launch configuration. */
(function(root){
'use strict';
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
/* Count a dense matrix product from its actual shard shape:
   X(m×k) · W(k×n) has m·n cells, k products per cell and k−1 additions
   between products. A k=1 shard therefore has no arithmetic addition. */
function matrixOps(x,w){
 const m=x.length,k=x[0].length,n=w[0].length;
 return {m,k,n,cells:m*n,mul:m*n*k,add:m*n*Math.max(k-1,0)};
}
function tensor(kind='row'){
 const x=[[1,2],[3,4]],w=[[2,0],[1,3]];
 const multiply=(a,b)=>a.map(row=>b[0].map((_,j)=>row.reduce((sum,v,k)=>sum+v*b[k][j],0)));
 const shards=[0,1].map(rank=>({x:kind==='row'?x.map(row=>[row[rank]]):x.map(row=>[...row]),w:kind==='row'?[[...w[rank]]]:w.map(row=>[row[rank]])}));
 const local=shards.map(s=>multiply(s.x,s.w));
 const ops=shards.map(s=>matrixOps(s.x,s.w));
 const total=matrixOps(x,w);
 /* Row split sums one local value per output cell across two cards; column
    split concatenates output columns and adds nothing across cards. */
 const mergeAdds=kind==='row'?total.cells:0;
 /* First output cell uses x[0][0..k-1] against w[0..k-1][0], derived from
    the shard's inner dimension rather than written as display copy. */
 const firstCell=shards.map((s,rank)=>({
  terms:Array.from({length:matrixOps(s.x,s.w).k},(_,j)=>({left:s.x[0][j],right:s.w[j][0]})),
  value:local[rank][0][0]
 }));
 return {kind,x,w,shards,local,result:multiply(x,w),ops,total,mergeAdds,firstCell};
}
/* One Prefill request, eight Query rows. Query i may read K1..K(i+1), so its
   causal pair count is i+1 and the total is 36. The two ranks exchange the
   local K/V copies (the AllGather teaching choice); source K/V stays put. A
   query is readable only after every remote K/V position it needs has arrived.
   Arrival times are ordering labels for the animation, not measured latency. */
function context(zigzag=false){
 const ranks=zigzag?[[0,1,6,7],[2,3,4,5]]:[[0,1,2,3],[4,5,6,7]];
 const owners=Array(8).fill(0);ranks.forEach((rows,r)=>rows.forEach(i=>owners[i]=r));
 const work=ranks.map(rows=>rows.reduce((sum,i)=>sum+i+1,0));
 const localKV=ranks.map(rows=>[...rows]);
 const copies=[];
 localKV.forEach((positions,from)=>positions.forEach((position,ordinal)=>{
  copies.push({position,from,to:1-from,ordinal,arrive:Number((2+ordinal*.2).toFixed(2))});
 }));
 copies.sort((a,b)=>a.position-b.position||a.from-b.from);
 const copyAt=position=>copies.find(c=>c.position===position);
 const queries=Array.from({length:8},(_,i)=>{
  const owner=owners[i];
  const requiredRemote=localKV[1-owner].filter(position=>position<=i);
  const readyAt=requiredRemote.length?Math.max(...requiredRemote.map(position=>copyAt(position).arrive)):1.95;
  const computeStart=Math.max(readyAt,1.95);
  return {index:i,label:'Q'+(i+1),pairs:i+1,owner,requiredRemote,needRemote:requiredRemote.length,
   readyAt:Number(readyAt.toFixed(2)),computeStart:Number(computeStart.toFixed(2)),
   computeEnd:Number((computeStart+1).toFixed(2))};
 });
 return {zigzag,ranks,owners,work,single:36,pairs:Array.from({length:8},(_,i)=>i+1),
  localKV,remoteKV:[localKV[1],localKV[0]],copies,queries,total:36,
  requiredRemoteByRank:[0,1].map(r=>[...new Set(queries.filter(q=>q.owner===r).flatMap(q=>q.requiredRemote))].sort((a,b)=>a-b))};
}
function decode(equal=false){
 const exp=equal?[1,1,1,1]:[1,1,3,3],values=[1,3,5,7];
 const sums=[0,1].map(r=>exp.slice(r*2,r*2+2).reduce((a,b)=>a+b,0));
 const numerators=[0,1].map(r=>dot(exp.slice(r*2,r*2+2),values.slice(r*2,r*2+2)));
 const local=numerators.map((n,i)=>n/sums[i]),weights=sums.map(s=>s/(sums[0]+sums[1]));
 const result=dot(local,weights);
 return {exp,values,sums,numerators,local,weights,lse:sums.map(Math.log),result,reads:values.length,readsPerRank:values.length/2,mean:(local[0]+local[1])/2,merge:{numerator:numerators[0]+numerators[1],denominator:sums[0]+sums[1]}};
}
function experts(skew=false){
 const routes=skew?[[0,1],[0,2],[0,3],[0,1]]:[[0,2],[1,3],[2,0],[3,1]];
 const selected=routes[0],weights=[.75,.25],outputs=selected.map(e=>2*(e+1)),loads=[0,0,0,0];routes.flat().forEach(e=>loads[e]++);
 const owner=[0,0,1,1],tokenOwner=[0,0,1,1];
 const remote=routes.map((es,t)=>es.filter(e=>owner[e]!==tokenOwner[t]).length);
 const rankLoads=[0,1].map(r=>loads.reduce((sum,n,e)=>sum+(owner[e]===r?n:0),0));
 const dispatchByOrigin=[0,1].map(r=>remote.reduce((sum,n,t)=>sum+(tokenOwner[t]===r?n:0),0));
 return {routes,selected,weights,outputs,loads,owner,tokenOwner,remote,rankLoads,dispatchByOrigin,crossDispatch:remote.reduce((a,b)=>a+b,0),crossCombine:remote.reduce((a,b)=>a+b,0),result:dot(outputs,weights)};
}
/* PP teaching schedule: one layer execution occupies one slot on both sides.
   Both sides use the same MB count and the same four-layer model. The adopted
   side has two stages with two layers each; activation transfer is a dependency
   edge, not an extra slot. */
function pipeline(mbCount=2){
 const n=mbCount===1?1:2;
 const span=2*n+2;
 const single=[];
 for(let mb=1;mb<=n;mb++)for(let layer=1;layer<=4;layer++)single.push({mb,layer,kind:'work'});
 const stage0=[],stage1=[];
 for(let t=0;t<span;t++){
  stage0.push(t<2*n?{mb:Math.floor(t/2)+1,layer:(t%2)+1,kind:'work'}:{kind:'drain'});
  stage1.push(t<2?{kind:'wait'}:t<2*n+2?{mb:Math.floor((t-2)/2)+1,layer:((t-2)%2)+3,kind:'work'}:{kind:'drain'});
 }
 return {
  n,span,single,stage0,stage1,
  completion:{single:4*n,adopted:2*n+2},
  work:4*n,singleGpuSlots:4*n,adoptedGpuSlots:2*span,
  bubbles:2*span-4*n,overlap:Math.max(0,2*n-2),
  activation:[1,2].slice(0,n).map(mb=>({mb,from:2*mb-1,to:2*mb}))
 };
}
function sequence(count=4){
 const tokens=Array.from({length:count},(_,i)=>'p'+(i+1));
 const shards=[tokens.filter((_,i)=>i<count/2),tokens.filter((_,i)=>i>=count/2)];
 return {tokens,shards,rowsPerRank:count/2,replicatedPerRank:count,replicasBefore:count*2,replicasAfter:count};
}
/* Two teaching requests stay identical in both conditions. DP changes request
   assignment and the number of complete parameter copies, never the two-layer
   work inside one request. A single copy may batch or run the requests one
   after another; this model does not order them. */
function dataParallel(replicas=2){
 const n=replicas===1?1:2;
 const requests=[
  {id:'R1',x:1,h:2,y:3,caller:1},
  {id:'R2',x:3,h:6,y:7,caller:2}
 ];
 const copies=n===2
  ?[{id:'A',requests:['R1']},{id:'B',requests:['R2']}]
  :[{id:'A',requests:['R1','R2']}];
 return {
  replicas:n,requests,copies,
  assignments:n===2?[['A'],['B']]:[['A'],['A']],
  paramBlocks:n*2,paths:n,workPerRequest:2,
  batch:n===2?'per-replica':'shared',
  outputs:requests.map(r=>r.y),sameInput:true
 };
}
const api={tensor,context,decode,experts,sequence,pipeline,dataParallel};if(typeof module==='object'&&module.exports)module.exports=api;root.ParallelTeachingModel=api;
})(typeof globalThis==='object'?globalThis:this);
