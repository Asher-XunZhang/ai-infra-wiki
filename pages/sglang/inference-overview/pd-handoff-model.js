/* Source-grounded, reversible teaching schedule. Times are narration, not measurements. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;root.PDHandoff=api;})(typeof globalThis==='object'?globalThis:this,function(){
'use strict';
const revision='279339f113b79af84f27fd3ac92d0a13bd3f4cbd';
const sources={
 intake:['managers/scheduler.py',3233,'def _add_request_to_queue('],
 prealloc:['disaggregation/decode.py',1123,'def pop_preallocated('],
 alloc:['disaggregation/decode.py',1838,'def _pre_alloc('],
 destination:['disaggregation/mooncake/conn.py',2706,'def send_metadata('],
 bootstrap:['disaggregation/mooncake/conn.py',2189,'def bootstrap_thread('],
 admit:['disaggregation/prefill.py',450,'def pop_bootstrapped('],
 finalize:['disaggregation/prefill.py',395,'def finalize_bootstrap('],
 loop:['disaggregation/prefill.py',641,'def event_loop_normal_disagg_prefill('],
 chunk:['disaggregation/prefill.py',1210,'def process_prefill_chunk('],
 result:['disaggregation/prefill.py',730,'def process_batch_result_disagg_prefill('],
 send:['disaggregation/prefill.py',1311,'def _send_kv_chunk('],
 queue:['disaggregation/mooncake/conn.py',2385,'def add_transfer_request('],
 worker:['disaggregation/mooncake/conn.py',1847,'def transfer_worker('],
 transfer:['disaggregation/mooncake/conn.py',641,'def _transfer_data('],
 buffers:['disaggregation/utils.py',329,'class MetadataBuffers:'],
 aux:['disaggregation/mooncake/conn.py',1247,'def send_aux('],
 conclude:['disaggregation/common/conn.py',507,'def conclude_transfer('],
 status:['disaggregation/common/conn.py',569,'def apply_prefill_status('],
 gate:['disaggregation/utils.py',212,'def _apply_metadata_gate('],
 commit:['disaggregation/decode.py',2158,'def _commit_transfer_to_req('],
 ready:['disaggregation/decode.py',2364,'def pop_transferred('],
 decodeQueue:['disaggregation/decode.py',2839,'def process_decode_queue('],
 prebuilt:['disaggregation/decode_schedule_batch_mixin.py',22,'def prepare_for_prebuilt('],
 relay:['disaggregation/decode_schedule_batch_mixin.py',112,'def process_prebuilt('],
 schedule:['disaggregation/decode.py',2680,'def get_next_disagg_decode_batch_to_run('],
 decodeLoop:['disaggregation/decode.py',2570,'def event_loop_normal_disagg_decode('],
 release:['disaggregation/prefill.py',972,'def process_disagg_prefill_inflight_queue(']
};
const cases={
 normal:['正常交接','两块计算、传输、就绪检查和首次 Decode，沿同一时钟推进。'],
 capacity:['D 容量不足','本窗口内 D 一直无法预分配；P 留在 bootstrap_queue，不启动本请求计算。'],
 metadata:['metadata 暂不可见','Success 已被观察到，但 metadata 的 room 仍为 0；D 暂留 transfer_queue。'],
 latepoll:['P 稍后轮询','改变 P Scheduler 观察发送终态的时刻，检查 D 是否必须等 P 清理完才开算。']
};
const lanes=[['pcpu','P · Scheduler / CPU'],['pgpu','P · GPU'],['worker','P · 传输 worker'],['control','跨机控制消息'],['dcpu','D · Scheduler / CPU'],['dgpu','D · GPU']];
const clamp=x=>Math.max(0,Math.min(1,x));
function build(option='normal'){
 if(!cases[option])option='normal';
 const events=[];
 const e=(id,lane,start,end,label,kind,depends,refs)=>{const v={id,lane,start,end,label,kind,depends,refs};events.push(v);return v;};
 e('pintake','pcpu',0,1.2,'Req / sender','control',[],['intake']);
 e('dintake','dcpu',0,1.2,'Req / receiver','control',[],['intake','prealloc']);
 const phases=[{start:0,title:'两侧各有本地 Req，以同一个 room 配对',text:'R1 已分别送入两侧 Scheduler。prompt 为 p0–p3；room=42 是交接标识。两份 Req、两套模型和两侧缓存各自持有。',refs:['intake','prealloc']}];
 if(option==='capacity'){
  e('pwait','pcpu',1.2,9,'等待 D 目标位置','wait',['pintake'],['admit']);
  e('capacity','dcpu',1.2,9,'预算不足，保留 prealloc_queue','wait',['dintake'],['prealloc']);
  phases.push({start:1.2,title:'D 没有足够的接收预算，P 仍不能计算',text:'pop_preallocated 检查容量，不能满足就不执行 _pre_alloc，也不发送目标 metadata。本例 optimistic_prefill_attempts=0，P 不能从 Bootstrapping 直接越过准入。结束的是观察窗口，请求仍在等待。',refs:['prealloc','admit']});
  return finish(option,events,phases,9);
 }
 e('allocate','dcpu',1.2,3.4,'预分配','control',['dintake'],['prealloc','alloc']);
 e('publish','dcpu',3.4,3.9,'发送目标 metadata','control',['allocate'],['destination']);
 e('destination','control',3.9,5.4,'目标信息抵达 P','control',['publish'],['destination','bootstrap']);
 e('pwait','pcpu',1.2,5.4,'等待目标位置','wait',['pintake'],['admit']);
 e('admit','pcpu',5.4,6.4,'准入 / 选批','control',['pwait','destination'],['admit','finalize','loop']);
 e('compute1','pgpu',6.4,9.8,'C1 · p0 p1','compute',['admit'],['loop','chunk']);
 e('enqueue1','pcpu',9.8,10.7,'提交 C1','control',['compute1'],['chunk','send','queue']);
 e('compute2','pgpu',10.7,15,'C2 · p2 p3','compute',['enqueue1'],['loop','result']);
 e('copy1','worker',10.7,15.4,'C1 · KV 写入 D','kv',['enqueue1'],['worker','transfer']);
 e('enqueue2','pcpu',15,16.2,'记录 y1 / 提交末块','control',['compute2'],['result','send','buffers']);
 e('copy2','worker',16.2,20.3,'C2 · KV 写入 D','kv',['copy1','enqueue2'],['worker','transfer']);
 e('aux','worker',20.3,22.2,'末块 aux','data',['copy2'],['aux','buffers']);
 e('conclude','worker',22.2,22.7,'本地终态','control',['aux'],['conclude']);
 // Host submission and remote observation are separate; CPU return is not a delivery barrier.
 e('post','worker',22.7,23.1,'发通知 / worker 收尾','control',['conclude'],['conclude','worker']);
 e('notify','control',23.1,24,'Success 抵达 D','control',['post'],['conclude','status']);
 const visible=option==='metadata'?26.8:22.2,commitStart=Math.max(24,visible);
 e('receiveWait','dcpu',3.9,24,'R1 留 transfer_queue；循环可处理其他工作','wait',['publish'],['decodeQueue','ready']);
 if(visible>24)e('metaWait','dcpu',24,visible,'Success → Transferring，等 room','wait',['notify'],['gate']);
 e('commit','dcpu',commitStart,commitStart+1.2,'核验 / 提交 y1','control',['notify',...(visible>24?['metaWait']:[])],['gate','commit','ready']);
 e('prebuilt','dcpu',commitStart+1.2,commitStart+3.1,'PREBUILT → 选批','control',['commit'],['decodeQueue','prebuilt','relay','schedule']);
 e('decode','dgpu',commitStart+3.1,commitStart+6.8,'输入 y1 → 采样 y2','compute',['prebuilt'],['decodeLoop']);
 e('output','dcpu',commitStart+6.8,commitStart+8,'本轮结果处理','control',['decode'],['decodeLoop']);
 const pollStart=option==='latepoll'?29.6:23.2;
 e('inflight','pcpu',16.2,pollStart,'Req 保留在 inflight；循环可处理其他工作','wait',['enqueue2'],['release']);
 e('release','pcpu',pollStart,pollStart+1.2,'poll / 释放本侧持有','control',['post','inflight'],['release']);
 phases.push(
  {start:1.2,title:'D 先预留接收位置，再把目标信息发给 P',text:'_pre_alloc 分配请求行和 KV 位置，写入 req_to_token；之后 send_metadata 发布 room、session、目标索引和 aux_index。预留的槽位还没有本请求的有效 KV。',refs:['prealloc','alloc','destination']},
  {start:5.4,title:'P 观察 WaitingForInput，才进入 waiting_queue 与选批',text:'目标信息到达 P 的 bootstrap_thread 后更新状态；pop_bootstrapped 还要分配本地 metadata 槽并初始化 sender。本例关闭乐观 Prefill，P 的计算必须在这之后。',refs:['bootstrap','admit','finalize','loop']},
  {start:9.8,title:'C1 已算出，传输 worker 搬运它；GPU 可以继续算 C2',text:'process_prefill_chunk 将已完成、可发送的范围入队。add_transfer_request 返回后，独立 worker 调用同步搬运；等待发生在该 worker。C2 写新位置，因此本例可以与 C1 的传输重叠。',refs:['chunk','queue','worker','transfer']},
  {start:15,title:'末块生成 y1，KV 与 aux 是两种载荷',text:'P 的 Req.output_ids 记录首 token y1。末块准备 MetadataBuffers；worker 先搬完对应的 GPU KV，再发送 CPU aux（包含 y1、room 等）。KV 表示已处理的 prompt；此时还没有 y1 的 KV。',refs:['result','send','buffers','aux']},
  {start:22.2,title:'P 的发送终态与 D 的接收就绪各自推进',text:option==='latepoll'?'P 尚未再次轮询自己的 inflight_queue；D 可以独立检查接收并继续。源码没有要求 D 等待 P 的 release_kv_cache。':'worker 记录 P 本地终态并通知 D。P Scheduler 后续 poll 才释放本侧持有；D 按自己的接收、metadata 和身份条件决定是否交出请求。两侧没有统一清理屏障。',refs:['conclude','status','release','ready']},
  ...(option==='metadata'?[{start:24,title:'Success 已到，但 room=0：D 暂不提交',text:'_apply_metadata_gate 把本次观察的 Success 降为 Transferring，R1 留在 transfer_queue。图示 aux 的可见性落后于终态观察这一受保护分支；不表示传输 worker 固定先发通知、后搬 aux。',refs:['gate','ready']}]:[]),
  {start:commitStart,title:'核验 room=42，把 y1 提交给 D 的 Req',text:'metadata 可见后仍需核验实际 room 与请求预期一致。_commit_transfer_to_req 把来自 P 的 y1 加入 D 的 output_ids；请求随后进入 waiting_queue。接收可用并不等于本轮已运行。',refs:['gate','commit','ready','decodeQueue']},
  {start:commitStart+1.2,title:'PREBUILT 接续已有状态，不再跑一遍 prompt',text:'prepare_for_prebuilt 准备 batch 元数据，process_prebuilt 把 y1 放进 FutureMap。Scheduler 处理 prebuilt 结果，再过滤 / 合并 running_batch、准备下一次 Decode。这里没有 prompt 模型 forward。',refs:['prebuilt','relay','schedule']},
  {start:commitStart+3.1,title:'D 真正执行：输入 y1，读取 prompt KV，再生成 y2',text:'ModelRunner 执行本轮 Decode：读取已接收的 prompt KV，计算并追加 y1 的 KV，采样 y2。y2 尚未有 KV。本图停在首次 Decode 的结果处理，后续按相同 Decode 循环续写。',refs:['decodeLoop','schedule']}
 );
 return finish(option,events,phases,commitStart+8,{metadataVisibleAt:visible});
}
function finish(option,events,phases,duration,extra={}){
 return {option,events,byId:Object.fromEntries(events.map(e=>[e.id,e])),phases,duration,lanes,...extra};
}
function state(tl,t){
 t=Math.max(0,Math.min(tl.duration,Number(t)||0));
 const progress=Object.fromEntries(tl.events.map(e=>[e.id,clamp((t-e.start)/(e.end-e.start))]));
 const done=id=>progress[id]===1,active=tl.events.filter(e=>e.start<=t&&t<e.end);
 const phase=tl.phases.reduce((a,p)=>p.start<=t?p:a,tl.phases[0]);
 const pQueue=done('release')?'本侧已收尾':done('enqueue2')?'inflight_queue':done('admit')?(done('compute2')?'末块结果处理':'chunked_req / batch'):done('pintake')?'bootstrap_queue':'创建本地 Req';
 const dQueue=done('output')?'本轮完成 / 继续生成':done('prebuilt')?'running_batch':done('commit')?'waiting_queue / PREBUILT':done('publish')?'transfer_queue':done('dintake')?'prealloc_queue':'创建本地 Req';
 return {t,progress,phase,active,pQueue,dQueue,reserved:done('allocate'),pKV:done('compute2')?4:done('compute1')?2:0,dKV:done('copy2')?4:done('copy1')?2:0,
  pOutput:done('enqueue2')?['y1']:[],dOutput:done('output')?['y1','y2']:done('commit')?['y1']:[],
  metadataVisible:tl.metadataVisibleAt!==undefined&&t>=tl.metadataVisibleAt,rawSuccess:done('notify'),committed:done('commit'),released:done('release'),decodeKV:done('decode')};
}
function sourceUrl(key){const [f,line]=sources[key];return `https://github.com/sgl-project/sglang/blob/${revision}/python/sglang/srt/${f}#L${line}`;}
return {revision,sources,cases,lanes,build,state,sourceUrl};
});
