/* Independent teaching arithmetic; no SGLang runtime, GPU, or timing simulation. */
(function(root){
 'use strict';
 const revision='279339f113b79af84f27fd3ac92d0a13bd3f4cbd';
 const sources={
  column:['layers/linear.py',492,'forward'],row:['layers/linear.py',1612,'forward'],
  groups:['distributed/parallel_state.py',2507,'initialize_model_parallel'],
  layers:['utils/common.py',1415,'make_layers'],pp:['managers/scheduler_pp_mixin.py',802,'_pp_send_dict_to_next_stage'],
  dp:['managers/data_parallel_controller.py',767,'round_robin_scheduler'],dpworld:['managers/data_parallel_controller.py',371,'launch_dp_schedulers'],
  widths:['runtime_context.py',151,'derive_parallel_widths'],ep:['layers/moe/ep_moe/layer.py',252,'forward_impl'],
  zigzag:['layers/cp/zigzag.py',121,'build_metadata'],cpkv:['layers/cp/zigzag.py',396,'materialize_full_kv']
 };
 const X=[1,2,3,4],W=[[1,1,1,1],[1,2,3,4]];
 function matvec(w,x){return w.map(row=>row.reduce((v,n,i)=>v+n*x[i],0));}
 function tensor(kind='column',gather=true){
  if(!['column','row'].includes(kind))throw Error('Unknown TP split');
  const local=[0,1].map(rank=>kind==='column'?matvec([W[rank]],X):matvec(W.map(row=>row.slice(rank*2,rank*2+2)),X.slice(rank*2,rank*2+2)));
  const full=kind==='column'?local.flat():local[0].map((v,i)=>v+local[1][i]);
  return {kind,gather,X,W,local,full,rankOutputs:kind==='column'&&!gather?local:[full.slice(),full.slice()]};
 }
 function pipeline(n=3){
  if(![1,3].includes(n))throw Error('Use one or three microbatches');
  return Array.from({length:n+2},(_,tick)=>({tick,n,stages:[0,1].map(stage=>{const batch=tick-stage;return batch>=1&&batch<=n?batch:null;}),done:Math.max(0,Math.min(n,tick-2))}));
 }
 function dataParallel(count=3){return Array.from({length:count+1},(_,step)=>({step,target:step?(step-1)%2:null,requests:[0,1].map(rank=>Array.from({length:step},(_,i)=>i+1).filter(i=>(i-1)%2===rank))}));}
 function experts(skew=false,token=1){
  if(!Number.isInteger(token)||token<1||token>4)throw Error('Token must be 1..4');
  const routes=skew?[[0,2],[0,1],[0,3],[0,1]]:[[0,2],[1,3],[2,0],[3,1]],weights=[.75,.25];
  const selected=routes[token-1],values=selected.map(e=>token*(e+1)),loads=Array(4).fill(0);
  routes.flat().forEach(e=>loads[e]++);
  return {token,routes,weights,selected,values,result:values.reduce((sum,v,i)=>sum+v*weights[i],0),loads,rankLoads:[loads[0]+loads[1],loads[2]+loads[3]]};
 }
 function context(zigzag=true){
  const ranks=zigzag?[[0,1,6,7],[2,3,4,5]]:[[0,1,2,3],[4,5,6,7]];
  return {ranks,loads:ranks.map(rows=>rows.reduce((s,q)=>s+q+1,0)),kv:[0,1,2,3,4,5,6,7]};
 }
 function groups(tp,pp){
  if(!Number.isInteger(tp)||!Number.isInteger(pp)||tp<1||pp<1)throw Error('Positive integer widths required');
  return {tp:Array.from({length:pp},(_,stage)=>Array.from({length:tp},(_,r)=>stage*tp+r)),pp:Array.from({length:tp},(_,r)=>Array.from({length:pp},(_,stage)=>stage*tp+r))};
 }
 const url=key=>`https://github.com/sgl-project/sglang/blob/${revision}/python/sglang/srt/${sources[key][0]}#L${sources[key][1]}`;
 const api={revision,sources,tensor,pipeline,dataParallel,experts,context,groups,matvec,url};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ParallelismModel=api;
})(typeof globalThis!=='undefined'?globalThis:this);
