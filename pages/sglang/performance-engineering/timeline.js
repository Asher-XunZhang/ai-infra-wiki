/* Source-grounded observation windows. All millisecond values are constructed. */
(function(root,factory){const api=factory(typeof module==='object'?require('./model.js'):root.PerformanceModel);if(typeof module==='object')module.exports=api;else root.PerformancePD=api;})(globalThis,M=>{
'use strict';
const srt='python/sglang/srt/',stats=srt+'observability/req_time_stats.py',tm=srt+'managers/tokenizer_manager.py',mc=srt+'observability/metrics_collector.py';
const sources={...M.sources,
 created:[tm,3497,'def _init_req_state'],first:[tm,2509,'set_first_token_time'],finish:[tm,2517,'set_finished_time'],metricGate:[tm,2555,'if self.enable_metrics'],labels:[tm,733,'labels ='],apiClock:[stats,479,'def get_interval'],apiOrigin:[stats,404,'def set_created_time'],tmITL:[mc,1868,'def observe_inter_token_latency'],finishedMetric:[mc,1785,'def observe_one_finished_request'],hist:[mc,1743,'self.histogram_time_to_first_token'],
 wait:[stats,746,'def set_wait_queue_entry_time'],forward:[stats,767,'def set_forward_entry_time'],pFinish:[stats,813,'def set_prefill_finished_time'],kvMetric:[stats,913,'def compute_and_observe_kv_transfer_metrics'],pTransfer:[stats,1004,'def set_prefill_kv_transfer_finish_time'],dBootstrap:[stats,1022,'def set_decode_transfer_queue_entry_time'],fakeOutput:[stats,1047,'def set_decode_prebuilt_finish_time'],stageHistogram:[mc,744,'self.per_stage_req_latency_seconds'],
 pChunk:[srt+'disaggregation/prefill.py',946,'self.send_kv_chunk'],pResult:[srt+'disaggregation/prefill.py',804,'set_prefill_finished_time'],pEntry:[srt+'disaggregation/prefill.py',895,'set_prefill_transfer_queue_entry_time'],pObserve:[srt+'disaggregation/prefill.py',1055,'compute_and_observe_kv_transfer_metrics'],dEntry:[srt+'disaggregation/decode.py',1609,'set_decode_transfer_queue_entry_time'],dCommit:[srt+'disaggregation/decode.py',2329,'set_wait_queue_entry_time'],dPrebuilt:[srt+'disaggregation/decode.py',2815,'set_forward_entry_time'],prebuiltResult:[srt+'managers/scheduler_components/batch_result_processor.py',116,'def process_batch_result_prebuilt'],streamInterval:[srt+'managers/scheduler_components/output_streamer.py',418,'def accept'],sender:[srt+'disaggregation/common/conn.py',1485,'def get_transfer_metric'],nixl:[srt+'disaggregation/nixl/conn.py',3115,'self._transfer_metric.transfer_latency_s'],gateway:['sgl-model-gateway/src/routers/http/pd_router.rs',721,'loop {']};
const url=k=>{const [p,l]=sources[k];return `https://github.com/sgl-project/sglang/blob/${M.revision}/${p}#L${l}`;};
const modes={
 clock:{name:'采样位置',question:'同一个首 token，三层秒表在哪里停？',cases:{base:'正常请求',permit:'客户端许可多等 120 ms',gate:'P HTTP 结果晚到 120 ms',poll:'D 接收提交晚轮询 80 ms'}},
 kv:{name:'KV 阶段区间',question:'decode_transferred 为什么长于实际 KV 搬运？',cases:{base:'单块 Prefill · Mooncake',chunked:'两块 Prefill / KV 交叠',missing:'接口未提供 transfer_total_bytes'}},
 stream:{name:'流式上报',question:'ITL 的采样点在输出回程，还是 GPU 上？',cases:{token:'stream_interval = 1',bundle:'stream_interval = 3',single:'只输出 1 个 token'}},
 sample:{name:'样本聚合',question:'慢请求失败后，P95 为什么反而下降？',cases:{all:'R1–R4 均成功',failure:'R4 部分输出后传输异常'}}
};
const lanes={client:'客户端 · native adapter',gateway:'Gateway · HTTP PDRouter',ptm:'P · TokenizerManager',ps:'P · Scheduler CPU',pg:'P · GPU',kv:'P → D · KV worker',ds:'D · Scheduler CPU',dg:'D · GPU',dtm:'D · TokenizerManager'};
function timeline(mode='clock',condition){
 if(!modes[mode])mode='clock';if(!(condition in modes[mode].cases))condition=Object.keys(modes[mode].cases)[0];
 const events=[],metrics=[];const add=(id,start,end,lane,label,detail,source,kind='control',depends=[],payload='')=>events.push({id,start,end,lane,label,detail,source,kind,depends,payload});
 const metric=(id,name,owner,start,end,observe,formula,detail,source,extra={})=>metrics.push({id,name,owner,start,end,observe,formula,detail,source,...extra});
 let scope='文本 R1；HTTP PDRouter + native /generate；P/D 各 TP=PP=DP=1；metrics 与 log_metrics 开启；无投机、HiCache、回撤或 KV offload；本例输出 text 非空。',facts={},duration;
 if(mode==='sample'){
  scope='四条构造的 P/D 请求；只观察客户端 native adapter 的结果集合，后端内部结束时刻未建模。固定 benchmark 窗口 600 ms；不把客户端失败当作后端已释放。';
  const rows=[{id:'R1',ttft:80,end:120},{id:'R2',ttft:80,end:160},{id:'R3',ttft:80,end:180},{id:'R4',ttft:500,end:580}];
  for(const r of rows){add(r.id,0,r.end,r.id,r.id+' 请求在途',r.id==='R4'&&condition==='failure'?'构造 transport exception：先前见到部分内容，随后 adapter 置 success=false。':'此轨道只投影客户端请求生命周期；不能从条长识别后端 GPU 时间。','adapter','wait');add('record-'+r.id,r.end,r.end+16,'records',r.id+(r.id==='R4'&&condition==='failure'?' × 失败':' ✓ 成功'),'保留原始结果；calculate_metrics 之后按 success 字段决定是否计入。','success');}
  add('aggregate',600,640,'aggregate','calculate_metrics','成功请求才进入 TTFT / TPOT / ITL 与输出计数；dur_s 是统一 benchmark 窗口，不是各请求延迟之和。','metrics');
  const d=M.sample(condition==='failure'?'failure':'all');facts={rows,result:{...d,outputTokens:d.throughput,throughput:d.throughput/.6},wall:600};duration=670;
  metric('p95','客户端成功样本 P95','calculate_metrics',0,600,640,'np.percentile(successful_ttfts, 95)','只有在汇总发生后，分布才作为本例最终统计显示。R4 失败会被排除，慢的观察仍留在原始记录中。','success',{value:d.p95,unit:'ms'});
  metric('throughput','成功输出吞吐','benchmark',0,600,640,`${d.throughput} token / 0.600 s`,'分母包括所有已调度请求到完成或失败的 benchmark 窗口；分子按成功样本统计。','throughput',{value:d.throughput/.6,unit:'token/s'});
 }else{
  const shift=condition==='permit'?120:0,lag=condition==='poll'?80:0,gate=condition==='gate'?120:0,one=mode==='stream'&&condition==='single',bundle=mode==='stream'&&condition==='bundle',chunked=mode==='kv'&&condition==='chunked';
  const at=t=>t+shift,dt=t=>t+shift+lag;
  facts={shift,lag,gate,one,bundle,chunked,st:at(80),pBorn:at(110),dBorn:at(112),pWait:at(180),pForward:at(210),pFinished:at(350),pTransfer:at(355),pPoll:at(425),pCompletion:at(426),pTM:at(440),pGate:at(450)+gate,dTransfer:at(180),dWait:dt(410),dForward:dt(420),dPrebuilt:dt(430),dFirst:dt(445),dFinished:dt(one?445:565)};
  const f=facts;
  f.tmOutputs=(one?[445]:bundle?[445,565]:[445,480,535,565]).map((t,i)=>({time:dt(t),n:bundle&&i===1?4:i+1}));
  let prev=0;f.clientOutputs=f.tmOutputs.map(o=>{const t=Math.max(o.time+20,f.pGate+15,prev+15);prev=t;return {time:t,n:o.n};});f.first=f.clientOutputs[0].time;f.end=f.clientOutputs.at(-1).time+20;
  f.kvBytes=condition==='missing'?null:64*1024*1024;duration=Math.max(f.end,at(450)+gate)+30;
  scope+=(chunked?' 两块 Chunked Prefill，P enable_overlap=true，已完成 bootstrap；KV worker 可与下一块 GPU 重叠。':mode==='kv'&&condition==='missing'?' 本条件只检验统计接口缺字段的分支，不声称 Mooncake 正常路径缺字段。':' 本例不分块。');
  if(mode==='stream')scope+=` stream_interval=${bundle?3:1}；输出 ${one?1:4} token；忽略 stop-string 抑制。`;
  add('permit',0,f.st,'client','等待并发许可','limited_request_func 在 semaphore 内才调用 adapter；这段在单请求 st 之前。','permit','wait');
  add('client-send',f.st,at(100),'client','st → HTTP 请求','native adapter 在 session.post 前启动 perf_counter。','start','data',['permit'],'JSON → Gateway');
  add('dispatch',at(100),at(115),'gateway','并发派发 P / D','两个 HTTP send future 在同一 handler 并发 poll；D 先响应会存入 decode_early，仍先检查 P。','gateway','data',[],'JSON + bootstrap room');
  add('p-input',f.pBorn,at(120),'ptm','登记 / 分词 / IPC','_init_req_state 使用 obj.received_time；未提供时由 set_created_time 取本地 perf_counter。本例使用本端起点。','created');
  add('d-input',f.dBorn,at(122),'dtm','独立登记 / IPC','D 有自己的 APIServerReqTimeStats，不能借用 P 的本地时间戳。','created');
  add('p-prepare',at(120),at(125),'ps','入 bootstrap','P Scheduler 收到请求后进入 bootstrap queue；prefill_prepare 在配置中不是 per-stage histogram 观测项。','wait');
  add('p-bootstrap',at(125),f.pWait,'ps','等接收端地址','prefill_bootstrap 到 set_wait_queue_entry_time 才闭合；与 D 端 bootstrap / 分配过程重叠。','wait','wait');
  add('d-prepare',at(122),at(130),'ds','入 prealloc','decode_prepare：scheduler_recv → decode_prealloc_queue_entry。','dBootstrap');
  add('d-bootstrap',at(130),f.dTransfer,'ds','握手 / 分配 / 通知','send_metadata 发布接收位置之后记录 decode_transfer_queue_entry_time，开始后续接收阶段统计。','dEntry','control',[],'接收地址 → P');
  add('p-queue',f.pWait,f.pForward,'ps','waiting_queue','queue_time 只覆盖 waiting_queue 到首次 forward_entry；不会将 bootstrap 再算进这段。','forward','wait',['p-bootstrap']);
  add('p-submit',f.pForward,at(215),'ps','准备 / 提交','CPU set_forward_entry_time 早于 GPU 工作；prefill_forward 是源码调用点间的墙钟区间。','forward','control',['p-queue']);
  if(chunked){
   add('p-gpu1',at(215),at(270),'pg','Prefill 块 1','块 1 设备计算完成后才能发送对应 KV。','pResult','compute',['p-submit']);
   add('p-chunk1-result',at(270),at(280),'ps','块 1 结果','CPU 处理本块结果，再交给传输 worker；不是整请求 prefill_finished。','pEntry','control',['p-gpu1']);
   add('kv1',at(282),at(340),'kv','块 1 KV → D','已生成的块 1 KV 搬运与下一块设备计算可重叠；这是给定合法交错，不是实测重叠比例。','pChunk','data',['p-chunk1-result'],'块 1 KV');
   add('p-gpu2',at(285),at(340),'pg','Prefill 块 2','请求的最后一块；本地 GPU 与 KV worker 是不同执行主体。','pResult','compute',['p-chunk1-result']);
  }else add('p-gpu',at(215),at(340),'pg','Prefill GPU','此条才是构造的设备活动窗口。prefill_forward 的两个时间戳落在 Scheduler 中，不能直接当作 kernel 时长。','pResult','compute',['p-submit']);
  add('p-result',at(340),f.pFinished,'ps','结果处理 / 首 token','CPU 处理 Prefill 结果并记录 prefill_finished_time；首 token 产生在 P，D 后面接收。','pResult','control',[chunked?'p-gpu2':'p-gpu']);
  add('p-send',f.pFinished,f.pTransfer,'ps','send_kv_chunk','最后块 send 调用在前，prefill_transfer_queue_entry 时间戳在后；传输 worker 可能已开始。','pEntry','control',['p-result']);
  add('kv-last',at(352),at(400),'kv',chunked?'块 2 KV + 结果 → D':'KV + 首 token 元数据 → D','独立传输面；本条是给定搬运窗口，不是 decode_transferred histogram 的全部区间。','dCommit','data',['p-result'],chunked?'块 2 KV / 首 token':'KV / 首 token');
  add('p-poll',at(400),f.pPoll,'ps','轮询 Success / 退役','P 观察 sender 成功后记录 prefill_kv_transfer_finish_time；轮询间隔也可能影响测量端点。','pTransfer','wait',['kv-last']);
  add('p-observe',f.pPoll,f.pCompletion,'ps','完成 / 采集','set_completion_time 后读取 sender 指标；没有 transfer_latency_s 时回退到 completion − transfer_queue_entry，只覆盖最后块尾部。','pObserve','control',['p-poll']);
  add('p-out',f.pCompletion,f.pTM,'ptm','P 结束回程','P 仍有本地 E2E、完成请求及输出计数；collect_metrics 的 TTFT 分支明确排除 PREFILL。','server','data',['p-observe'],'P 完成结果');
  add('p-gate',f.pTM,f.pGate,'gateway','接收 P HTTP 结果','此条件可以单独推迟 P HTTP 响应；不改变 D 本地首回程时刻。Gateway 仍先检查 P 结果。','gateway','wait',['p-out']);
  add('d-receive',f.dTransfer,f.dWait,'ds','transfer_queue / 提交','decode_transferred 从公布接收位置后开始，包含等 P 计算、KV 传输和 D 轮询 / metadata gate / 提交。','wait','wait',['d-bootstrap']);
  add('d-wait',f.dWait,f.dForward,'ds','waiting_queue','_commit_transfer_to_req 记录 wait_queue_entry；到 prebuilt 准入时 set_forward_entry_time。','dCommit','wait',['d-receive','kv-last']);
  add('d-prebuilt',f.dForward,f.dPrebuilt,'ds','PREBUILT 首 token','从 P 交接的首 token 走 prebuilt 输出；fake_output 不是再执行一次 Prefill GPU。','prebuiltResult','control',['d-wait']);
  for(let i=0;i<f.tmOutputs.length;i++){
   const o=f.tmOutputs[i],first=i===0;
   add('tm-'+i,o.time-10,o.time,'dtm',first?'首输出批次 → TTFT':`累计 ${o.n} token → ${one?'结束':'ITL'}`,'TM 在 _handle_batch_output 更新 first / finished 时间，再按 enable_metrics && log_metrics 调用 collect_metrics。','first','data',first?['d-prebuilt']:['d-gpu'+o.n],`BatchStrOutput · n=${o.n}`);
   const c=f.clientOutputs[i];add('client-'+i,c.time-12,c.time,'client',first?'非空 text → TTFT':`收到累计 ${c.n} token`,first?'native adapter 只在非空 text 条件下首次记 TTFT，不以 GPU 完成或 D TM 收到批次为终点。':'按 completion_tokens 增量 Δn，把内容包间隔 / Δn 扩展为 Δn 个 ITL 样本。','chunks','data',['tm-'+i,'p-gate'],`SSE JSON · n=${c.n}`);
  }
  if(!one){
   add('d-gpu2',dt(435),dt(470),'dg','生成 t2','D 的下一轮设备工作可以和首输出回程及 P HTTP 收尾重叠。','prebuiltResult','compute',['d-prebuilt']);
   add('d-gpu3',dt(480),dt(525),'dg','生成 t3','设备区间与前端采样区间分别显示，不能用包间隔替代 kernel 事件。','server','compute',['d-gpu2']);
   add('d-gpu4',dt(530),dt(555),'dg','生成 t4','stream_interval 改变输出频率，不将本例设备工作凭空加速。','streamInterval','compute',['d-gpu3']);
  }
  add('tail',f.clientOutputs.at(-1).time,f.end,'client','[DONE] / 流尾','native adapter 每次读到非空流数据先更新 latency，再检查 [DONE]；尾部可能进入 latency，但不产生新的 ITL token 样本。','adapter','data',[], '[DONE]');
  metric('client-ttft','客户端 TTFT','native adapter',f.st,f.first,f.first,'首个非空 text − st','包含请求路径、P/D 处理与 Gateway 放行 / 回传；许可等待在 st 之前。','start');
  metric('d-ttft','D TM TTFT','D TokenizerManager',f.dBorn,f.dFirst,f.dFirst,'first_token_time − created_time','在本地首输出批次处理时打点；此处不等待客户端收到，也不等于 D GPU 首轮时长。','server',{wire:'sglang:time_to_first_token_seconds'});
  metric('p-ttft','P TTFT 不上报','P TokenizerManager',f.pBorn,f.pTM,null,'PREFILL 跳过 TTFT observe','P 的 time_stats 仍可记录首回程，但此 histogram 分支不采样；缺失不等于 0 ms。','server',{absent:true,wire:'sglang:time_to_first_token_seconds'});
  metric('client-e2e','客户端 latency','native adapter',f.st,f.end,f.end,'最后非空流数据时间 − st','本例最后一个非空流数据是 [DONE]，不是 t4 的生成时刻。','adapter');
  metric('d-e2e','D TM E2E','D TokenizerManager',f.dBorn,f.dFinished,f.dFinished,'finished_time − created_time','D 本地 finished 回程结束；与 P 本地 E2E 交叠，不能两端相加还原客户端。','finish',{wire:'sglang:e2e_request_latency_seconds'});
  metric('p-e2e','P TM E2E','P TokenizerManager',f.pBorn,f.pTM,f.pTM,'finished_time − created_time','P 端在成功传输退役后输出完成；并不等于整条业务生成已经结束。','finish',{wire:'sglang:e2e_request_latency_seconds'});
  metric('d-transfer','decode_transferred','D Scheduler',f.dTransfer,f.dWait,f.dWait,'wait_queue_entry − decode_transfer_queue_entry','宽区间覆盖等待 P 计算、传输及本地轮询提交；延迟轮询会拉长该值，即使 KV 搬运区间没变。','wait',{wire:'sglang:per_stage_req_latency_seconds{stage="decode_transferred"}'});
  metric('p-transfer','prefill_transfer_kv_cache','P Scheduler',f.pTransfer,f.pPoll,f.pPoll,'prefill_kv_transfer_finish − transfer_queue_entry','从最后块 send 后的时间戳开始，到成功退役的时间戳结束；不含先前块完整传输区间。','pTransfer',{wire:'sglang:per_stage_req_latency_seconds{stage="prefill_transfer_kv_cache"}'});
  metric('kv-latency','KV transfer latency','P Scheduler / sender',f.pTransfer,f.pCompletion,condition==='missing'?null:f.pCompletion,'backend latency；缺省时 completion − transfer_queue_entry','本例 Mooncake 提供字节数但未提供专用 latency，走最后块尾部回退。若 total_bytes 缺失，函数提前返回，不上报本次值。NIXL 的专用计时另按后端读源码。','kvMetric',{wire:'sglang:kv_transfer_latency_ms',absent:condition==='missing'});
  metric('p-forward','prefill_forward','P Scheduler',f.pForward,f.pFinished,f.pFinished,'prefill_finished − last_forward_entry','CPU 入口到 CPU 结果处理端点；本例普通分块不重置 last_forward_entry，窗口跨两块，含块间间隙。它不是独立 GPU kernel 时间。','pFinish',{wire:'sglang:per_stage_req_latency_seconds{stage="prefill_forward"}'});
  metric('p-bootstrap','prefill_bootstrap','P Scheduler',at(125),f.pWait,f.pWait,'wait_queue_entry − bootstrap_queue_entry','接收端地址与准入前条件的等待；与 D bootstrap / allocation 有重叠。','wait',{wire:'sglang:per_stage_req_latency_seconds'});
  metric('d-bootstrap','decode_bootstrap','D Scheduler',at(130),f.dTransfer,f.dTransfer,'transfer_queue_entry − prealloc_queue_entry','分配和发布接收信息后才结束；不能直接当作 KV payload 搬运。','dBootstrap',{wire:'sglang:per_stage_req_latency_seconds'});
  metric('d-queue','D queue_time / decode_waiting','D Scheduler',f.dWait,f.dForward,f.dForward,'forward_entry − wait_queue_entry','本例从 KV 提交进入 waiting 开始，只到 PREBUILT 准入；该队列窗口不是整条 D 等待时间。','forward',{wire:'sglang:queue_time_seconds'});
  metric('fake','fake_output','D Scheduler',f.dForward,f.dPrebuilt,f.dPrebuilt,'decode_prebuilt_finish − last_forward_entry','首 token 来自 P 的交接结果。PREBUILT 收尾打点并输出，不代表一次 D Prefill GPU 计算。','fakeOutput',{wire:'sglang:per_stage_req_latency_seconds{stage="fake_output"}'});
  if(mode==='stream'){
   add('tpot-collect',f.end+5,f.end+25,'client','汇总 TPOT','adapter 返回后，calculate_metrics 按 success 与 output_len > 1 判断是否追加 TPOT；汇总耗时为教学构造。','tpot','control',['tail']);
   const a=f.tmOutputs.at(-2)||f.tmOutputs[0],b=f.tmOutputs.at(-1),c=f.clientOutputs.at(-2)||f.clientOutputs[0],d=f.clientOutputs.at(-1),dn=b.n-a.n;
   metric('d-itl','本次 D ITL','D TokenizerManager',a.time,b.time,one?null:b.time,'(本次回程 − 上次回程) / Δcompletion_tokens','量尺显示实际回程间隔；采集时除以新增 token 数，同一值记 Δn 次。它不是 GPU 的逐 token 执行时刻。','tmITL',{wire:'sglang:inter_token_latency_seconds',absent:one,value:one?null:(b.time-a.time)/dn});
   metric('client-itl','本次客户端 ITL','native adapter',c.time,d.time,one?null:d.time,'(本次非空 text − 上次非空 text) / Δcompletion_tokens','在客户端读取内容时均摊，而非读取 D TM 的秒表；本例固定额外回传时延。','chunks',{absent:one,value:one?null:(d.time-c.time)/dn});
   metric('client-tpot','本请求 TPOT','客户端 calculate_metrics',f.first,f.end,one?null:f.end+25,'(latency − ttft) / (output_len − 1)','量尺为首内容到流尾；数值还要除以后续 token 数。流尾后已有计算输入；图中稍后的 calculate_metrics 才追加成功且长度大于 1 的样本。','tpot',{absent:one,value:one?null:(f.end-f.first)/(d.n-1)});
  }
 }
 const defaultMetric=mode==='sample'?'p95':mode==='kv'?'d-transfer':mode==='stream'?'d-itl':'client-ttft';
 return {mode,condition,events,metrics,facts,duration,scope,defaultMetric,...modes[mode]};
}
function snapshot(tl,value){const time=Math.max(0,Math.min(tl.duration,Number(value)||0)),active=tl.events.filter(e=>e.start<=time&&time<e.end),ended=tl.events.filter(e=>e.end<=time),focus=active.find(e=>e.lane==='client')||active[0]||ended.at(-1)||tl.events[0];
 const observations=tl.metrics.map(m=>({...m,status:m.absent?'不采样':time<m.start?'未开始':time<m.end?'计时中':time<(m.observe??Infinity)?'等待采集':'已采集',elapsed:m.absent?null:Math.max(0,Math.min(time,m.end)-m.start),value:m.absent?null:(m.value??m.end-m.start)}));
 const f=tl.facts,receive=list=>list.filter(o=>o.time<=time),itls=list=>list.slice(1).flatMap((o,i)=>Array(o.n-list[i].n).fill((o.time-list[i].time)/(o.n-list[i].n)));
 const tmOut=f.tmOutputs?receive(f.tmOutputs):[],clientOut=f.clientOutputs?receive(f.clientOutputs):[];
 return {time,active,ended,focus,observations,tmOut,clientOut,tmItl:itls(tmOut),clientItl:itls(clientOut),tpot:tl.mode==='sample'||time<f.end||f.one?null:(f.end-f.first)/(f.clientOutputs.at(-1).n-1),progress:e=>Math.max(0,Math.min(1,(time-e.start)/(e.end-e.start)))};
}
// Presentation groups are envelopes of existing events, never new execution steps.
// Their lengths include internal gaps; only the expanded lanes distinguish CPU/GPU work.
function overview(tl){
 if(tl.mode==='sample')return {lanes:{R1:'请求 R1',R2:'请求 R2',R3:'请求 R3',R4:'请求 R4'},events:tl.events.filter(e=>/^R[1-4]$/.test(e.id))};
 const events=[],group=(id,lane,label,detail,ids,kind='control')=>{
  const children=ids.map(id=>tl.events.find(e=>e.id===id)).filter(Boolean);if(!children.length)return;
  events.push({id,lane,label,detail,start:Math.min(...children.map(e=>e.start)),end:Math.max(...children.map(e=>e.end)),source:children[0].source,children:children.map(e=>e.id),kind,depends:[]});
 };
 group('o-permit','client','等许可','请求还在客户端等并发许可；这段不计入客户端 TTFT。',['permit'],'wait');
 group('o-first','client','等待首个内容','从发出请求开始计时，收到首个非空内容才停止 TTFT 秒表。',['client-send','client-0'],'wait');
 const first=tl.events.find(e=>e.id==='client-0'),tail=tl.events.find(e=>e.id==='tail');
 events.push({id:'o-output',lane:'client',label:tl.facts.one?'流尾':'接收后续输出',detail:'后续内容与流结束先后到达。客户端收到的时刻，与 D 本地生成和回告的时刻不同。',start:first.end,end:tail.end,source:'adapter',kind:'data',depends:[],children:tl.events.filter(e=>/^client-[1-9]/.test(e.id)||e.id==='tail').map(e=>e.id)});
 group('o-p-ready','P','等 D 地址','P 等待 D 准备好接收位置。两端可以同时推进各自的准备。',['p-prepare','p-bootstrap'],'wait');
 group('o-p-work','P','计算并给出首 token','P 完成输入的计算并处理首 token 结果。这个阶段含 CPU 与 GPU 活动；展开模块过程可看到它们的分工。',['p-submit','p-gpu','p-gpu1','p-chunk1-result','p-gpu2','p-result']);
 group('o-p-done','P','P 收尾','P 确认发送完成，再返回本地结束结果。此时 D 可以继续处理请求。',['p-poll','p-observe','p-out']);
 for(const id of ['kv1','kv-last']){const e=tl.events.find(e=>e.id===id);if(e)events.push({...e,label:id==='kv1'?'块 1 KV':tl.facts.chunked?'块 2 KV':'KV → D',detail:'P 把已有 KV 和必要结果交给 D。D 的接收等待区间可以比真正搬运更长。'});}
 group('o-d-ready','D','准备接收','D 分配接收位置并通知 P；此时还没有收到完整 KV。',['d-prepare','d-bootstrap']);
 group('o-d-wait','D','等待 KV 就绪','D 已经开始等 KV，P 仍可能正在计算。搬运完成后，还要由 D 检查并提交接收结果。',['d-receive'],'wait');
 group('o-d-work','D',tl.facts.one?'回告首 token':'首 token 回告与后续生成','D 使用 P 交接的首 token 回告；需要更多输出时，才继续生成后续 token。它可以与 P 收尾、客户端回传并行。',['d-wait','d-prebuilt','d-gpu2','d-gpu3','d-gpu4',...tl.events.filter(e=>/^tm-/.test(e.id)).map(e=>e.id)]);
 return {lanes:{client:'客户端',P:'Prefill（P）',kv:'P → D · KV',D:'Decode（D）'},events};
}
return {revision:M.revision,sources,url,modes,lanes,timeline,snapshot,overview};
});
