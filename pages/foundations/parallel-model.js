/* Mechanism arithmetic, not measured speed or any engine's launch configuration. */
(function(root){
'use strict';
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
function tensor(kind='row'){
 const x=[[1,2],[3,4]],w=[[2,0],[1,3]];
 const multiply=(a,b)=>a.map(row=>b[0].map((_,j)=>row.reduce((sum,v,k)=>sum+v*b[k][j],0)));
 const shards=[0,1].map(rank=>({x:kind==='row'?x.map(row=>[row[rank]]):x.map(row=>[...row]),w:kind==='row'?[[...w[rank]]]:w.map(row=>[row[rank]])}));
 const local=shards.map(s=>multiply(s.x,s.w));
 return {kind,x,w,shards,local,result:multiply(x,w)};
}
function context(zigzag=false){
 const ranks=zigzag?[[0,1,6,7],[2,3,4,5]]:[[0,1,2,3],[4,5,6,7]];
 return {ranks,work:ranks.map(rows=>rows.reduce((sum,i)=>sum+i+1,0))};
}
function decode(equal=false){
 const exp=equal?[1,1,1,1]:[1,1,3,3],values=[1,3,5,7];
 const sums=[0,1].map(r=>exp.slice(r*2,r*2+2).reduce((a,b)=>a+b,0));
 const numerators=[0,1].map(r=>dot(exp.slice(r*2,r*2+2),values.slice(r*2,r*2+2)));
 const local=numerators.map((n,i)=>n/sums[i]),weights=sums.map(s=>s/(sums[0]+sums[1]));
 return {exp,values,sums,numerators,local,weights,lse:sums.map(Math.log),result:dot(local,weights)};
}
function experts(skew=false){
 const routes=skew?[[0,1],[0,2],[0,3],[0,1]]:[[0,2],[1,3],[2,0],[3,1]];
 const selected=routes[0],weights=[.75,.25],outputs=selected.map(e=>2*(e+1)),loads=[0,0,0,0];routes.flat().forEach(e=>loads[e]++);
 return {routes,selected,weights,outputs,loads,result:dot(outputs,weights)};
}
const api={tensor,context,decode,experts};if(typeof module==='object'&&module.exports)module.exports=api;root.ParallelTeachingModel=api;
})(typeof globalThis==='object'?globalThis:this);
