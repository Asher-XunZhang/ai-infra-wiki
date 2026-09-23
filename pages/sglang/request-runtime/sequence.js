/* Graph semantics derived from the request snapshots. Coordinates are teaching steps, not time. */
(function(root){
 'use strict';
 const lanes=[['client','客户端'],['front','前端'],['scheduler','调度'],['worker','执行 / GPU'],['memory','请求行 / KV'],['output','反分词']];
 function event(frame){
  const id=frame.id;
  if(id==='input')return {label:'接入',edges:[[0,1,'R1']],kind:'message'};
  if(id==='tokenize')return {label:'分词',edges:[[1,1,'p1…p4']],kind:'compute'};
  if(id==='dispatch')return {label:'派发',edges:[[1,2,'IPC']],kind:'message',pending:true};
  if(id==='queued')return {label:'排队',edges:[[2,2,'R1 入队']],kind:'wait'};
  if(id==='invalid')return {label:'校验失败',edges:[[1,1,'拒绝']],kind:'abort'};
  if(id==='prefill')return {label:'Prefill',edges:[[2,3,'p1…p4'],[3,4,'写 KV']],kind:'compute',sample:'y1'};
  if(id.startsWith('decode-')||id==='abort-result'){
   const k=frame.sampled;return {label:id==='abort-result'?'取消收尾':`Decode ${k-1}`,edges:[[2,3,`y${k-1}`],[3,4,'写 KV']],kind:id==='abort-result'?'abort':'compute',sample:`y${k}`};
  }
  if(id==='first-output'||id.startsWith('output-'))return {label:'输出回程',edges:[[2,5,'ID'],[5,1,'文字'],[1,0,`y${frame.visible}`]],kind:'message'};
  if(id==='cancel-sent')return {label:'发送取消',edges:[[1,2,'Abort']],kind:'abort',pending:true};
  if(id==='abort-pending')return {label:'标记待结束',edges:[[2,2,'待结束']],kind:'abort'};
  if(id==='abort-queued')return {label:'移出队列',edges:[[2,1,'取消回告']],kind:'abort',pending:true};
  if(id==='abort-front')return {label:'前端收尾',edges:[[1,1,'结束']],kind:'abort'};
  if(id==='finish')return {label:'达到上限',edges:[[2,2,'结束判定']],kind:'finish'};
  if(id==='release')return {label:'资源交接',edges:[[2,4,frame.cached?'缓存接管':'归还 KV']],kind:'memory'};
  if(id==='final')return {label:'结束回程',edges:frame.abortSent?[[2,5,'结束 ID'],[5,1,'结束']]:[[2,5,'结束 ID'],[5,1,'结束'],[1,0,`y${frame.visible}`]],kind:frame.abortSent?'abort':'finish'};
  throw new Error('Missing diagram event: '+id);
 }
 const api={lanes,event};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.RuntimeSequence=api;
})(typeof globalThis!=='undefined'?globalThis:this);
