/* Continuous presentation of the fixed-source teaching snapshots in model.js.
 * Seconds are reading time, not measured SGLang latency. Geometry is shared by
 * wires and travellers; intermediate states are derived, never accumulated. */
(function(root){
'use strict';
const M=typeof module==='object'&&module.exports?require('./model.js'):root.RequestRuntimeModel;
const nodes={
 request:{x:34,y:76,w:136,h:64,title:'客户端',sub:'HTTP 请求 · R1'},
 response:{x:34,y:242,w:136,h:76,title:'流式响应',sub:'客户端可见的文字'},
 http:{x:226,y:76,w:200,h:64,title:'HTTP / 协议入口',sub:'GenerateReqInput'},
 tokenizer:{x:226,y:196,w:200,h:122,title:'TokenizerManager',sub:'分词、派发、关联响应'},
 req:{x:502,y:76,w:220,h:112,title:'Scheduler · Req',sub:'waiting_queue / running_batch'},
 batch:{x:764,y:76,w:230,h:112,title:'ScheduleBatch',sub:'本轮输入 → ForwardBatch'},
 worker:{x:764,y:262,w:230,h:106,title:'TpModelWorker',sub:'ModelRunner → 模型 / GPU'},
 result:{x:502,y:262,w:220,h:106,title:'Scheduler · 结果处理',sub:'更新输出与结束状态'},
 detokenizer:{x:226,y:430,w:200,h:88,title:'DetokenizerManager',sub:'输出 ID → 增量文本'},
 cache:{x:502,y:430,w:220,h:88,title:'RadixCache / 分配器',sub:'缓存接管或归还有效 KV'},
 memory:{x:764,y:430,w:230,h:88,title:'请求行 / GPU KV 池',sub:'地址映射与物理 KV 分开'}
};
const port=(id,side,offset=.5)=>{const n=nodes[id];return side==='l'?[n.x,n.y+n.h*offset]:side==='r'?[n.x+n.w,n.y+n.h*offset]:side==='t'?[n.x+n.w*offset,n.y]:[n.x+n.w*offset,n.y+n.h];};
const edges={};
function wire(id,from,to,kind,points,label,lx,ly){edges[id]={id,from:from[0],to:to[0],kind,points:[port(...from),...points,port(...to)],label,lx,ly};}
wire('http',['request','r'],['http','l'],'request',[],'HTTP',198,96);
wire('prepare',['http','b'],['tokenizer','t'],'local',[],'请求 / rid',365,170);
wire('dispatch',['tokenizer','r',.38],['req','l',.66],'request',[[462,242.36],[462,149.92]],'IPC',480,138);
wire('batch',['req','r'],['batch','l'],'local',[],'Req',743,120);
wire('forward',['batch','b'],['worker','t'],'tensor',[],'输入张量',921,230);
wire('sample',['worker','l',.36],['result','r',.36],'request',[],'ID',743,289);
wire('update',['result','t'],['req','b'],'local',[],'更新 Req',659,230);
wire('kv',['worker','b'],['memory','t'],'kv',[],'写入 K/V',924,401);
wire('release',['result','b'],['cache','t'],'local',[],'资源收尾',657,401);
wire('ownership',['cache','r'],['memory','l'],'local',[],'索引',743,462);
wire('output',['result','l',.5],['detokenizer','t',.8],'request',[[470,315],[470,384],[386,384]],'输出 ID · IPC',378,368);
wire('text',['detokenizer','l'],['tokenizer','b',.27],'text',[[192,474],[192,350],[280,350]],'文本 · IPC',335,336);
wire('client',['tokenizer','l',.69],['response','r'],'text',[],'文本',198,268);
wire('abort-reply',['req','l',.25],['tokenizer','t',.83],'control',[[464,104],[464,176],[392,176]],'取消回告',392,158);
const initial={front:'尚未登记',scheduler:'尚无 Req',batch:'无',waiting:false,slot:false,kv:0,cached:0,sampled:0,visible:0,dispatched:false,abortSent:false,toFinish:false,finished:false,frontFinished:false,forwards:0};
function build(options={}){
 const frames=M.build(options),phases=[],events=[];let total=0;
 for(const [index,f]of frames.entries()){
  let duration=3.4;const work=f.id==='prefill'||f.id.startsWith('decode-')||f.id==='abort-result';
  if(work)duration=f.id==='prefill'?10:7.8;
  else if(f.id==='input')duration=4.6;
  else if(f.id==='tokenize')duration=4;
  else if(f.id==='first-output'||f.id.startsWith('output-')||f.id==='final')duration=f.id==='first-output'?6.8:6;
  else if(f.id==='release')duration=5.2;
  else if(f.id==='finish')duration=2.8;
  const p={id:f.id,start:total,end:total+duration,duration,index,frame:f,previous:frames[index-1]||initial};phases.push(p);
  const e=(edge,a,b,payload,kind,nodesOverride)=>events.push({phase:index,edge,start:total+a*duration,end:total+b*duration,payload,kind:kind||edges[edge]?.kind||'local',nodes:nodesOverride||(edge?[edges[edge].from,edges[edge].to]:[])});
  if(f.id==='input'){e('http',.04,.38,'R1 · HTTP 请求');e('prepare',.42,.76,'GenerateReqInput · rid=R1');e(null,.76,.98,'建立 ReqState，保存响应等待者','local',['tokenizer']);}
  else if(f.id==='tokenize')e(null,.03,.95,'文本 → 输入 ID p1…p4；整理采样参数','request',['tokenizer']);
  else if(f.id==='dispatch')e('dispatch',.06,.88,'TokenizedGenerateReqInput · rid + input_ids + sampling_params');
  else if(f.id==='queued')e(null,.04,.94,'创建 Req(R1)，放入 waiting_queue；尚未获准执行','local',['req']);
  else if(work){
   e('batch',.02,.16,'Req 引用 → 本轮 ScheduleBatch');
   e(null,.16,.33,'请求行已预留；准备本轮 KV 地址（同进程）','local',['memory','batch']);
   e('forward',.19,.34,f.id==='prefill'?'input_ids=p1…p4 + 位置 / KV 地址':'input_ids=y'+(f.sampled-1)+' + 位置 / KV 地址');
   e(null,.34,.65,'GPU 前向：读取输入，计算本轮 hidden states / logits','tensor',['worker']);
   e('kv',.40,.61,f.id==='prefill'?'p1…p4 的 K/V':'y'+(f.sampled-1)+' 的 K/V');
   e('sample',.67,.79,'采样结果 · y'+f.sampled+'（token ID）');
   e('update',.81,.94,'更新 Req.output_ids 与生成进度');
  }else if(f.id==='first-output'||f.id.startsWith('output-')||f.id==='final'){
   const end=f.id==='final',cancel=end&&f.abortSent;
   e('output',.02,.31,cancel?'BatchTokenIDOutput · 取消结束原因':'BatchTokenIDOutput · rid + 输出 ID'+(end?' + 结束原因':''));
   e(null,.31,.46,'反分词：输出 ID → 增量文本','text',['detokenizer']);
   e('text',.46,.73,'BatchStrOutput · rid + 文本'+(end?' + 结束原因':''));
   if(!cancel)e('client',.77,.94,end?'最终响应片段 / 结束':'流式响应片段');
   else e(null,.76,.96,'前端清理等待状态；额外采样文本不在本例模拟','control',['tokenizer']);
  }else if(f.id==='finish')e(null,.04,.96,'Req.finished_reason = length；客户端还在等待最终回程','local',['req','result']);
  else if(f.id==='release'){e('release',.04,.40,'release_kv_cache → cache_finished_req');e('ownership',.44,.76,options.cache===false?'解除映射，相关 KV 归还分配器':'缓存持有有效 KV 的索引；物理 KV 保持原位');e(null,.78,.98,'释放 ReqToTokenPool 请求行','local',['memory','cache']);}
  else if(f.id==='cancel-sent')e('dispatch',.06,.88,'AbortReq · rid=R1（取消控制消息）','control');
  else if(f.id==='abort-pending')e(null,.04,.94,'Req.to_finish = FINISH_ABORT；请求行与 KV 继续保留','control',['req']);
  else if(f.id==='abort-queued')e(null,.04,.94,'移出 waiting_queue；本例尚未分配 KV','control',['req']);
  else if(f.id==='abort-front')e('abort-reply',.04,.88,'AbortReq 回告 → _handle_abort_req → 清理 ReqState');
  else if(f.id==='invalid')e(null,.04,.92,'验证失败：清理未派发的 ReqState；Scheduler 无 Req','control',['http','tokenizer']);
  total+=duration;
 }
 return {phases,events,duration:total,options};
}
function at(timeline,time){
 const t=Math.max(0,Math.min(timeline.duration,Number(time)||0));
 const p=timeline.phases.find(p=>t<p.end)||timeline.phases.at(-1),u=(t-p.start)/p.duration,f=p.frame;
 let state={...p.previous};
 if(u>=.96)state={...f};
 else if(f.id==='input'&&u>=.76)state={...f};
 else if(f.id==='dispatch'&&u>=.06)state={...state,dispatched:true,front:'ReqState · 等待结果'};
 else if(f.id==='queued'&&u>=.45)state={...f};
 else if(f.id==='prefill'||f.id.startsWith('decode-')||f.id==='abort-result'){
  if(u>=.16)Object.assign(state,{waiting:false,slot:true,batch:f.batch,scheduler:'Req · 本轮执行中'});
  if(u>=.61)state.kv=f.kv;
  if(u>=.67)Object.assign(state,{sampled:f.sampled,forwards:f.forwards,scheduler:'Req · 采样结果返回中'});
  if(u>=.94)state={...f};
 }else if(f.id==='release'&&u>=.80)state={...f};
 else if(f.id==='final'){
  if(u>=.73)Object.assign(state,{frontFinished:true,front:f.front});
  if(u>=.94)state={...f};
 }else if(f.id==='first-output'||f.id.startsWith('output-')){if(u>=.94)state={...f};}
 else if(f.id==='cancel-sent'&&u>=.06)state.abortSent=true;
 else if(['abort-pending','abort-queued','invalid','finish'].includes(f.id)&&u>=.65)state={...f};
 else if(f.id==='abort-front'&&u>=.88)state={...f};
 const active=timeline.events.filter(e=>t>=e.start&&t<e.end).map(e=>({...e,progress:(t-e.start)/(e.end-e.start)}));
 return {time:t,phase:p,u,state,active,ended:t===timeline.duration};
}
const api={nodes,edges,initial,build,at};
if(typeof module==='object'&&module.exports)module.exports=api;else root.RequestRuntimeFlow=api;
})(typeof globalThis==='object'?globalThis:this);
