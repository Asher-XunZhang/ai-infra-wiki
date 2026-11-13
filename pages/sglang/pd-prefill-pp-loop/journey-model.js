/* The presentation reads the reviewed dependency model; it never reschedules it. */
(function (scope) {
  'use strict';
  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  const active=(e,t)=>e && e.end>e.start && t>=e.start && t<e.end;
  const progress=(e,t)=>!e?0:e.end===e.start?Number(t>=e.end):clamp((t-e.start)/(e.end-e.start),0,1);
  const phases=['准入与排队','准备输入','本级前向','等待／处理结果','KV 传输','等待共识／清理'];
  const files={pp:'managers/scheduler_pp_mixin.py',scheduler:'managers/scheduler.py',prefill:'disaggregation/prefill.py',policy:'managers/schedule_policy.py',cache:'mem_cache/unified_radix_cache.py',controller:'managers/cache_controller.py'};
  const stages={A:'状态检查',B:'缓存与选批',C:'缓存与选批',D:'输入与提交',E:'输入与提交',F:'结果与共识',G:'结果与共识',H:'结果与共识',I:'转发与保存'};
  // Merge only adjacent CPU service operations. Waits, gaps and local-loop
  // boundaries remain explicit; GPU / transfer intervals are never merged in.
  function cpuGroups(model,rank){
    const groups=[];
    for(const e of model.events.filter(e=>e.r===rank&&['cpu','wait'].includes(e.kind)).sort((a,b)=>a.start-b.start)){
      const label=stages[e.phase]||e.label,prev=groups.at(-1);
      if(e.kind==='cpu'&&prev?.kind==='cpu'&&prev.n===e.n&&prev.label===label&&Math.abs(prev.end-e.start)<1e-7){prev.end=e.end;prev.members.push(e);}
      else groups.push({...e,label:e.kind==='wait'?e.label:label,members:[e],group:true});
    }
    return groups;
  }
  function processWindow(model,time){
    const span=Math.min(24,model.end),start=clamp(Math.floor(time/24)*24,0,Math.max(0,model.end-span));
    return {start,end:start+span};
  }
  function compile(packet){
    const graph=Object.fromEntries(Object.entries(packet.model.graph).map(([id,n])=>[id,{id,...n}]));
    return {packet,graph,nodes:Object.values(graph),end:packet.model.end,batches:packet.quick.batches,loops:packet.quick.loops,events:packet.model.events};
  }
  function rankState(model,batch,rank,time){
    const b=model.batches.find(b=>b.batch===batch).ranks[rank];
    const phase=b.phases.findIndex(p=>active(p,time));
    const complete=time>=b.phases[5].end;
    const cpu=model.events.find(e=>e.r===rank&&['cpu','wait'].includes(e.kind)&&active(e,time));
    const gpu=model.nodes.find(e=>e.r===rank&&e.kind==='gpu'&&active(e,time));
    const loop=model.loops.find(l=>l.r===rank&&active(l,time));
    let label=complete?'本级清理完成':phase<0?'尚未进入本级':phase===0&&time>=b.admitted?'已准入，等待选批':phases[phase];
    if(phase===3){const recv=model.graph[`${rank}:${b.resultLoop}:recv_out`],result=model.graph[`${rank}:${b.resultLoop}:result`],send=model.graph[`${rank}:${b.resultLoop}:send_kv`];
      label=time<recv.start?'等待 output 回流':time<result.start?'等结果就绪／本轮推进':time<result.end?'处理 output，更新请求':time<send.end?'提交本级 KV':'KV 已提交，等待传输';
    }
    if(phase===5)label=time<b.releaseStart?'等共识／本地复查':'正在本地清理';
    return {b,phase,complete,cpu,gpu,loop,label};
  }
  function edgeType(node,p){
    if(p.id===node.prev)return 'CPU 程序顺序';
    if(['h2d','kv'].includes(node.kind)&&p.kind===node.kind)return '教学假设：同类传输串行';
    if(node.kind==='gpu'&&p.kind==='gpu')return 'forward 流顺序';
    if(node.kind==='gpu'&&p.kind==='h2d')return '教学简化：等待全部 H2D';
    if(node.kind==='h2d'&&p.kind==='gpu')return 'schedule_stream → start_event';
    if(node.kind==='copy'&&p.kind==='gpu')return 'schedule_stream → copy_stream';
    if(node.r!==p.r)return /work|commit/.test(node.id)?'通信配对（模型门控）':'跨 PP 数据／消息就绪';
    return '本地数据／event 依赖';
  }
  function dependencies(model,id,time){
    const n=model.graph[id];if(!n)return null;
    const deps=n.deps.map(id=>model.graph[id]).filter(Boolean);
    const latest=deps.length?Math.max(...deps.map(p=>p.end)):0;
    return {n,latest,ready:deps.every(p=>p.end<=time),deps:deps.map(p=>({...p,ready:p.end<=time,critical:Math.abs(p.end-latest)<.001,type:edgeType(n,p)}))};
  }
  function transfers(model,batch){
    return model.nodes.filter(n=>n.owner===`M${batch}` && (n.kind==='kv'||n.id.endsWith(':proxy_message')||n.id.endsWith(':out_message')||n.kind==='h2d')).map(n=>({...n,to:n.kind==='kv'?'D':n.kind==='h2d'?n.r:n.id.endsWith(':out_message')?(n.r+1)%3:n.r+1,payload:n.kind==='kv'?'本级 KV':n.kind==='h2d'?'缓存 K/V':n.id.endsWith(':out_message')?'output 张量':'hidden states'}));
  }
  function crossRelations(model,time,batch,focus=''){
    const result=[],seen=new Set();
    const add=(source,target,wait=null)=>{
      if(!source||!target||source.r===target.r||!target.deps.includes(source.id))return;
      const id=source.id+'>'+target.id;if(seen.has(id))return;seen.add(id);
      const transport=source.kind==='message',parents=source.deps.map(id=>model.graph[id]).filter(Boolean);
      const producer=transport?(parents.find(n=>n.kind==='gpu')||parents.find(n=>n.kind==='cpu')||source):source;
      const payload=source.id.endsWith(':proxy_message')?'激活张量':source.id.endsWith(':out_message')?'output 张量':source.label;
      result.push({id,source,target,producer,wait,transport,payload,type:transport?/proxy_message|out_message/.test(source.id)?'数据传递':'控制消息':edgeType(target,source)});
    };
    for(const wait of model.events.filter(e=>e.kind==='wait'&&active(e,time)))
      for(const id of wait.deps||[])add(model.graph[id],model.graph[wait.target],wait);
    const messages=model.nodes.filter(n=>n.kind==='message'&&/proxy_message|out_message/.test(n.id)).sort((a,b)=>a.start-b.start);
    const consumer=source=>model.nodes.find(n=>n.kind==='cpu'&&n.r!==source.r&&n.deps.includes(source.id));
    for(const message of messages.filter(n=>active(n,time)))add(message,consumer(message));
    // Keep the next payload for the tracked batch visible even before sending.
    const tracked=messages.filter(n=>n.owner===`M${batch}`),next=tracked.find(n=>n.end>time)||tracked.at(-1);
    if(next)add(next,consumer(next));
    if(focus){const [a,b]=focus.split('>');add(model.graph[a],model.graph[b]);}
    return result;
  }
  const url=(ref,commit)=>{const [file,line]=ref.split(':');return `https://github.com/sgl-project/sglang/blob/${commit}/python/sglang/srt/${files[file]}#L${line}`;};
  scope.PP_JOURNEY_MODEL={compile,rankState,dependencies,transfers,crossRelations,cpuGroups,processWindow,stages,active,progress,phases,url,clamp};
  if(typeof module!=='undefined')module.exports=scope.PP_JOURNEY_MODEL;
})(typeof window==='undefined'?globalThis:window);
