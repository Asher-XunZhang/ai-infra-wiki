(function(root,factory){const api=factory(typeof module==='object'?require('./model.js'):root.ServingModel);if(typeof module==='object')module.exports=api;else root.ServingPD=api;})(globalThis,M=>{
'use strict';
const srt='python/sglang/srt/',router='sgl-model-gateway/src/routers/http/pd_router.rs';
const sources={...M.sources,
 pair:[router,978,'async fn select_pd_pair'],available:[router,1074,'let available_workers'],inject:[router,242,'fn inject_bootstrap_into_value'],dispatch:[router,652,'async fn execute_dual_dispatch_internal'],poll:[router,721,'loop {'],pFailure:[router,744,'if prefill_failed'],dResult:[router,776,'let decode_result'],stream:[router,1110,'fn create_streaming_response'],pairHealth:[router,1466,'async fn health_generate'],warmupPD:[srt+'entrypoints/http_server.py',2161,'async def _send_disaggregation_warmup_requests'],fake:[srt+'disaggregation/decode.py',748,'TransferBackend.FAKE'],pdQueue:[srt+'managers/scheduler.py',3233,'def _add_request_to_queue'],pdAbort:[srt+'managers/scheduler.py',5332,'for req in self.disagg_prefill_bootstrap_queue.queue'],dAbort:[srt+'managers/scheduler.py',5351,'for decode_req in self.disagg_decode_prealloc_queue.queue'],transferAbort:[srt+'managers/scheduler.py',5359,'for decode_req in self.disagg_decode_transfer_queue.queue'],runningAbort:[srt+'managers/scheduler.py',5396,'for req in self.collect_inflight_reqs'],abortTask:[srt+'managers/tokenizer_manager.py',2202,'def create_abort_task'],preallocPoll:[srt+'disaggregation/decode.py',961,'prepare_abort'],transferPoll:[srt+'disaggregation/decode.py',2364,'def pop_transferred'],defer:[srt+'disaggregation/decode.py',2501,'def _defer_release'],resolve:[srt+'disaggregation/decode.py',2525,'def resolve_deferred_releases'],pRetire:[srt+'disaggregation/prefill.py',972,'def process_disagg_prefill_inflight_queue'],finish:[srt+'managers/scheduler_components/batch_result_processor.py',1310,'if req.finished()'],ack:[srt+'disaggregation/common/conn.py',669,'def _maybe_ack_drained_abort'],ackGate:[srt+'disaggregation/common/conn.py',635,'def is_abort_release_safe'],preallocRetire:[srt+'disaggregation/decode.py',1184,'FINISH_ABORT'],batchOutput:[srt+'managers/tokenizer_manager.py',2255,'async def _handle_batch_output']};
const url=k=>{const [p,l]=sources[k];return `https://github.com/sgl-project/sglang/blob/${M.revision}/${p}#L${l}`;};
const modes={
 health:{name:'两端探活',question:'P、D 都返回 200，证明了什么？',cases:{pair:'网关并发探测一对 P/D',stall:'D 没有新回程活动',startup:'两端各自启动与 warmup',light:'Runtime /health 关闭生成检查'}},
 dispatch:{name:'成对派发',question:'选中 P/D 后，两个 HTTP 请求怎样并发推进？',cases:{success:'正常交接与响应',early:'D 先返回错误',none:'找不到可用的 D'}},
 cancel:{name:'失败与取消',question:'连接结束后，哪一侧还持有请求或 KV？',cases:{prefill:'Prefill 失败，D 仍在等待',stream:'流式客户端断连',waiting:'D waiting_queue 等待超时',transfer:'D 接收中取消，延迟归还 KV'}},
 drain:{name:'单端退出',question:'P 退出时，D 是否也必须同时停止？',cases:{normal:'P 本地排空，D 继续生成',arrivals:'仍有直连请求进入 P',stuck:'P Scheduler 超时仍未退出'}}
};
const laneNames={gw:'Gateway · 异步处理',ph:'P · HTTP / TM',ps:'P · Scheduler CPU',pg:'P · GPU',dh:'D · HTTP / TM',ds:'D · Scheduler CPU',dg:'D · GPU',kv:'KV worker / 控制消息'};
const clone=x=>JSON.parse(JSON.stringify(x));
function runtime(){return {http:'Up',tm:'无 R1 登记',queue:'尚未入队',gpu:'本例未安排计算',kv:'无 R1 占用',reply:'尚无响应',allocated:false,released:false,frontDone:false};}
function timeline(mode='health',condition){
 if(!modes[mode])mode='health';if(!(condition in modes[mode].cases))condition=Object.keys(modes[mode].cases)[0];
 const initial={client:'连接中，等待响应',gateway:'尚未开始本次操作',pair:'P / D 尚未选定',payload:'载荷尚未准备',outcome:'尚无结论',P:runtime(),D:runtime(),clientDone:false,abortSent:false,ack:false,shutdown:false,processExited:false};
 const events=[];function add(id,start,end,lane,label,detail,source,apply=()=>{},extra={}){events.push({id,start,end,lane,label,detail,source,apply,...extra});}
 const opts=(depends=[],edge=null,payload='',kind='control')=>({depends,edge,payload,kind});
 let scope='HTTP PDRouter + Python Runtime；P、D 各 TP=PP=DP=1；普通文本；无投机 / HiCache / KV offload / HiSparse；主机放置未指定。',answer='';
 if(mode==='health'){
  answer='这里验证选中 worker 的本地活动或启动条件；不验证真实 P→D KV 链路，也不覆盖所有副本。';
  initial.payload='健康探针与业务请求 R1 分开';initial.P.queue=initial.D.queue='不把探活等同于 R1 准入';
  if(condition==='startup'){
   scope+=' 启用正常 warmup；先假设两个进程均已启动。';initial.P.http=initial.D.http='Starting';initial.gateway='启动由各 Runtime 自己完成';
   for(const [key,lane,end] of [['P','ph',5],['D','dh',7]]){
    add('model-'+key,0,1.2,lane,key+' 接口可访问','先等 /model_info 可响应；Starting 期间健康接口仍返回 503。','warmup',s=>{s[key].reply='/model_info 200';});
    add('warm-'+key,1.2,end,lane,key+' fake-PD warmup','本端向自身 /generate 发 fake bootstrap 请求；DP>1 时 asyncio.gather 按本端 DP rank 并发，本例 DP=1。','warmupPD',s=>{s[key].reply='本端 warmup 全部 200';s[key].http='Up';s[key].kv='未验证真实跨端 KV';},opts(['model-'+key],null,'FAKE_BOOTSTRAP_HOST / room=0','compute'));
   }
   add('report',7,8.2,'gw','比较两端就绪','P 的 Up 不等待 D 的 warmup；两个进程的初始化没有本页虚构的全局屏障。','warmup',s=>{s.outcome='两端分别 Up；真实 PD 请求尚未验证';},opts(['warm-P','warm-D']));
  }else if(condition==='light'){
   initial.gateway='此条件直接检查两个 Runtime /health';initial.payload='生成检查关闭；Starting / 退出门仍优先';
   for(const [key,lane,end] of [['P','ph',1.5],['D','dh',2]])add('light-'+key,0,end,lane,key+' 静态健康分支','Up 且无退出标记时，关闭 /health 生成检查可直接返回 200；/health_generate 不走这个关闭分支。','health',s=>{s[key].reply='/health 200';},opts([],null,'没有生成探针'));
   add('light-end',2,3,'gw','记录覆盖边界','本图不是网关 /health_generate 的执行路径。','health',s=>{s.outcome='静态 200 不证明 Scheduler 或真实 KV 传输前进';},opts(['light-P','light-D']));
  }else{
   initial.gateway='GET /health_generate';
   add('pair',0,1.2,'gw','选择一个 P/D 对','select_pd_pair 使用各角色策略；这里只探测选中这一对，非所有副本。','pairHealth',s=>{s.pair='已选 P1 + D1';s.payload='两份独立 /health_generate；本地 fake bootstrap';});
   for(const [key,lane,end] of [['P','ph',2.2],['D','dh',2.5]])add('probe-'+key,1.2,end,lane,'启动 '+key+' 探针','tokio::join! 并发发送；Runtime 记录 tic 并启动本地生成探针，bootstrap_host=FAKE、room=0。','health',s=>{s[key].tm='独立探针任务；观察全局活动';s[key].kv='FAKE：未验证真实 P→D KV';},opts(['pair'],'g'+key,'GET /health_generate'));
   add('activity-P',2.2,4,'ph','P 有新回程','本地 last_receive_tstamp > tic；普通请求回包也可以满足，不要求业务 R1 或专属探针完成。','health',s=>{s.P.reply='200 · 本地有活动';s.P.tm='探针任务取消 / 登记移除';},opts(['probe-P'],'P'+'g','HTTP 200'));
   add('activity-D',2.5,condition==='stall'?7:5.2,'dh',condition==='stall'?'D 等活动到期限':'D 有新回程','D 的检查与 P 独立。没有新活动时，超过检查期限置 UnHealthy，返回 503；图中 u 不等于秒。','health',s=>{s.D.reply=condition==='stall'?'503 · 活动窗口超时':'200 · 本地有活动';s.D.http=condition==='stall'?'UnHealthy':'Up';s.D.tm='探针任务取消 / 登记移除';},opts(['probe-D'],'Dg',condition==='stall'?'HTTP 503':'HTTP 200',condition==='stall'?'wait':'control'));
   const t=condition==='stall'?7:5.2;add('aggregate',t,t+1.2,'gw','汇总两端 HTTP 结果','join! 等两次请求结束；两端都成功才返回 200，否则 503。成功也不构成真实 KV 交接的端到端验收。','pairHealth',s=>{s.client=condition==='stall'?'503 · 选中对探活失败':'200 · 选中对探活通过';s.outcome=answer;},opts(['activity-P','activity-D'],'gc',condition==='stall'?'503':'200'));
  }
 }else if(mode==='dispatch'){
  answer='HTTP 派发并发；GPU 计算仍遵守 KV 就绪条件。D 先响应时只暂存，源码先处理 P 的结果。';initial.payload='原请求：rid=R1；input_ids；sampling_params';
  add('select',0,1.2,'gw','分角色选 P 和 D','候选按 Prefill / Decode 分组，各自按可用性与策略筛选；不是把整个请求只交给一个混合 worker。','pair',s=>{s.pair=condition==='none'?'P1 可用；无可用 D':'P1 + D1';s.gateway=condition==='none'?'选对失败，尚未派发':'成对选择完成';});
  if(condition==='none')add('reject',1.2,2.5,'gw','返回 503，不派发','缺任一角色时 handle_server_selection_error 返回 503；两个 Runtime 不凭空出现 R1。','pair',s=>{s.client='503 · 无可用 P/D 对';s.outcome='发现 P 地址，不等于存在可服务的 P/D 对';s.clientDone=true;},opts(['select'],'gc','503'));
  else{
   add('inject',1.2,2.2,'gw','注入 bootstrap 元数据','把选中 P 的 host / port 和新 room 注入 JSON；不是把 KV 数据塞进 HTTP。','inject',s=>{s.payload='rid=R1；bootstrap_host=P1；port；room=42';},opts(['select']));
   for(const [key,lane,end] of [['P','ph',3.2],['D','dh',3.6]])add('send-'+key,2.2,end,lane,'HTTP → '+key,'两个 send future 在同一 handler 内被 select! 并发 poll；没有“P 整个完成后才发 D”。','dispatch',s=>{s[key].tm='rid_to_state[R1]';s[key].queue=key==='P'?'bootstrap_queue：R1':'prealloc_queue：R1';},opts(['inject'],'g'+key,'JSON + 同一 room'));
   add('p-bootstrap',3.2,5.8,'ps','P 等接收端准备','P bootstrap 与 D preallocation 是各自的工作；P 的 HTTP 请求先到，不代表能立刻计算。','pdQueue',s=>{s.P.queue='等待 D 元数据';},opts(['send-P'],null,'等待接收端','wait'));
   add('d-alloc',3.6,5.8,'ds','D 准备接收空间','本例成功分配接收 KV 与元数据，随后通知 P；内部容量和握手细节链接 PD 专题。','pdQueue',s=>{s.D.queue='transfer_queue：R1';s.D.kv='接收空间已预留，尚无完整 KV';s.D.allocated=true;},opts(['send-D']));
   add('bootstrap',5.8,6.8,'kv','接收元数据 → P','交接的是接收地址与控制信息，不是 D 已生成 token。','pdQueue',s=>{s.P.queue='waiting / 已可选批';},opts(['p-bootstrap','d-alloc'],'DP','接收元数据'));
   add('prefill',6.8,10.2,'pg','P GPU 前向','本例未分块；计算生成本地 KV 和首 token，输出与传输完成仍需后续处理。','pRetire',s=>{s.P.gpu='本轮 Prefill 完成';s.P.kv='本地 KV 等待传输收尾';s.P.allocated=true;s.P.queue='inflight：R1';},opts(['bootstrap'],null,'计算','compute'));
   add('kv',10.2,12.5,'kv','KV + 交接结果 → D','真正 KV 数据走独立传输面。图概括成功交接，详细 rank / metadata gate 在 PD 课程展开。','transferPoll',s=>{s.D.kv='完整 KV / 交接结果已可用';s.D.queue='waiting_queue：R1';},opts(['prefill'],'PD','KV + 交接结果','data'));
   add('p-retire',12.5,13.5,'ps','P 本地退役','P 轮询传输完成后解锁资源、清理 sender，再输出完成响应。D 不需要等 P 的 HTTP 响应抵达网关才继续。','pRetire',s=>{s.P.queue='inflight 已移除';s.P.kv='请求所有权已释放 / 缓存另算';s.P.released=true;},opts(['kv']));
   add('p-response',13.5,14.5,'ph','P 响应到网关','网关取得并检查 Prefill 响应；不会把它误称为完整 Decode 输出。','dResult',s=>{s.P.reply='200';s.P.tm='R1 本地登记已结束';s.gateway='P 成功；取得 / 等待 D 结果';},opts(['p-retire'],'Pg','P HTTP 200'));
   if(condition==='early'){
    add('d-error',12.5,13.5,'dh','D 先返回 503','条件注入：D 已完成接收，但执行前返回错误；HTTP 错误响应比 P 响应先到。不是“未接收 KV 就已成功生成”。','dispatch',s=>{s.D.reply='503 · 已先到达网关';s.gateway='decode_early 保存 D 错误';s.D.queue='后端清理由错误分支决定';},opts(['kv'],'Dg','D HTTP 503'));
    add('early-wait',13.5,14.5,'gw','暂存 D，仍等 P','select! 不因 decode_early 存在就提前越过 Prefill 结果处理。','poll',()=>{},opts(['d-error'],null,'等 P 响应','wait'));
    add('return',14.5,16,'gw','处理 D 错误','P 已成功，取出暂存的 D 响应，按 D 错误路径构造返回；本例非流式。','dResult',s=>{s.client='503 · Decode 错误';s.clientDone=true;s.outcome='D HTTP 响应先到，不改变 P 结果检查的依赖';},opts(['p-response','early-wait'],'gc','错误响应'));
   }else{
    add('decode',12.5,16,'dg','D GPU 继续生成','D 的设备工作可与 P 本地退役和 HTTP 回程并行；图不强行串行化两端收尾。','transferPoll',s=>{s.D.gpu='本例生成结束';s.D.reply='输出已产生';},opts(['kv'],null,'计算','compute'));
    add('d-response',16,17,'dh','D 响应到网关','选定非流式成功路径；这里演示一次请求闭合，不重复 Decode 每轮内部细节。','dResult',s=>{s.D.reply='200 · 最终输出';s.D.tm='R1 登记已结束';s.D.frontDone=true;s.D.queue='结果已完成';s.D.kv='请求所有权已释放 / 缓存另算';s.D.released=true;},opts(['decode'],'Dg','D 输出响应'));
    add('await-D',14.5,17,'gw','P 已成功，等待 D','网关等待 D 不意味着 P 进程或其他请求必须停止。','dResult',()=>{},opts(['p-response'],null,'等待 D','wait'));
    add('return',17,18.2,'gw','响应客户端','本例无 logprob 合并；最终业务输出由 D 响应构造。','dResult',s=>{s.client='200 · 收到生成结果';s.clientDone=true;s.outcome=answer;},opts(['d-response','await-D'],'gc','生成结果'));
   }
  }
 }else if(mode==='cancel'){
  initial.pair='已选 P1 + D1';initial.payload='逻辑请求 R1；room=42；两端是各自的 Req';initial.P.tm=initial.D.tm='rid_to_state[R1]';
  answer='HTTP / TM 结束、Scheduler 请求退出、KV 可重新使用是不同条件。';
  if(condition==='prefill'){
   initial.P.queue='bootstrap / 失败分支';initial.D.queue='prealloc_queue：R1，未分配 KV';initial.gateway='P、D HTTP 均已派发';
   add('p-fail',0,2,'ph','P 返回失败','条件注入：P HTTP 返回 500；D 请求仍 pending。本场景选择 D 尚未分配接收 KV 的位置。','pFailure',s=>{s.P.reply='HTTP 500';},opts([],'Pg','P HTTP 500'));
   add('close-D',2,3,'gw','返回错误，丢弃 D future','按 P 状态记账；P 的 5xx / 传输错误映射为 502，不把连带取消计为 D 自身熔断失败。','pFailure',s=>{s.gateway='D pending future 已丢弃';s.client='502 · P 失败';s.clientDone=true;s.D.http='上游连接已关闭';},opts(['p-fail'],'gD','连接关闭'));
   add('detect',3,5,'dh','D 检测断连','非后台、非流式等待路径在下一次检查中发现 request.is_disconnected；不能把关闭连接画成后端即时删除。','disconnect',s=>{s.abortSent=true;s.D.tm='abort_sent=true；仍待后端回告';},opts(['close-D'],null,'等待检查','wait'));
   add('abort',5,6,'ds','D receiver.abort()','Scheduler 在 prealloc_queue 分支通知 receiver；该步骤不是从所有队列统一 pop。','dAbort',s=>{s.D.queue='prealloc：receiver 已收到取消';},opts(['detect']));
   add('poll-fail',6,7.8,'ds','轮询失败并退出','本例 receiver 进入 Failed，准入队列沿失败分支输出终态；没有在未分配空间时虚构 KV 释放。','preallocRetire',s=>{s.D.queue='失败项退役';s.D.reply='后端 abort 终态';},opts(['abort']));
   add('echo',7.8,9,'dh','D 本地登记结束','D 回程让本地前端完成收尾；客户端先前已收到 P 失败。P 内部资源的失败退役见 PD 专题，HTTP 500 不能代替该证据。','batchOutput',s=>{s.D.tm='R1 登记已结束';s.D.frontDone=true;s.outcome=answer;},opts(['poll-fail']));
  }else if(condition==='stream'){
   initial.client='正在接收 D 的 SSE';initial.P.tm='R1 已结束';initial.P.queue='P 已完成交接';initial.P.kv='P 本地已收尾';initial.D.queue='running_batch：R1';initial.D.kv='生成中，仍持有 KV';initial.D.allocated=true;initial.gateway='转发 D 流；P 已结束';
   add('gpu',0,7,'dg','D 已提交的一轮计算','客户端断连期间，已提交 GPU 工作不被 HTTP 取消直接撤销；本例选择结果可读后处理终态。','runningAbort',s=>{s.D.gpu='本轮结果可读';},opts([],null,'已提交计算','compute'));
   add('client-close',0,1.5,'gw','客户端关闭流','网关流任务在 tracked.next() 与 tx.closed() 间 select!；已就绪 chunk 优先分支可先被处理。','stream',s=>{s.client='已断连';s.clientDone=true;s.gateway='tx.closed / 发送失败 → 退出转发';},opts([],'cg','连接关闭'));
   add('drop-upstream',1.5,2.5,'dh','D 上游流被关闭','网关丢弃上游流；这不等于向所有 P/D worker 广播 AbortReq。','stream',s=>{s.D.http='流连接关闭，后端仍在执行';},opts(['client-close'],'gD','丢弃 D HTTP 流'));
   add('abort-task',2.5,5,'dh','Runtime 取消任务','StreamingResponse 的后台 abort task 会先 sleep(2)，仍有 rid 登记时再派发 AbortReq；u 不模拟这 2 秒。','abortTask',s=>{s.abortSent=true;s.D.tm='abort_sent=true；仍持有登记';},opts(['drop-upstream'],null,'延后检查','wait'));
   add('mark',5,6,'ds','设置 to_finish=ABORT','运行中请求只先提交待结束标记；finished_reason、KV 和结果引用不能同时凭空消失。','runningAbort',s=>{s.D.queue='running：to_finish=ABORT';},opts(['abort-task']));
   add('commit',7,8.5,'ds','结果路径提交终态','本例普通 Decode 沿结果处理更新 finished，再完成 KV 收尾；不输出“强行中止 GPU kernel”。','finish',s=>{s.D.queue='finished=ABORT';s.D.kv='请求所有权已归还 / 缓存另算';s.D.released=true;},opts(['gpu','mark']));
   add('clear',8.5,9.8,'dh','前端结束登记','输出回程完成本地 ReqState 收尾；断开的客户端不保证收到最后回告。','batchOutput',s=>{s.D.tm='R1 登记已结束';s.D.frontDone=true;s.clientDone=true;s.outcome=answer;},opts(['commit']));
  }else if(condition==='waiting'){
   initial.P.tm='R1 已结束';initial.P.queue='交接完成';initial.D.queue='waiting_queue：R1';initial.D.kv='KV 已到达，仍占接收空间';initial.D.allocated=true;initial.gateway='等 D 响应';
   add('deadline',0,2,'ds','达到等待期限','条件：SGLANG_REQ_WAITING_TIMEOUT>0、入队时间>0。比较为 entry_time < now-timeout，恰等于期限还不满足。','timeout',()=>{},opts([],null,'等待轮询','wait'));
   add('timeout',2,3.5,'ds','轮询产生 AbortReq','这里只扫描 waiting_queue；P bootstrap、D prealloc / transfer 各自有其他超时机制，不能套用这一计时器。','timeout',s=>{s.abortSent=true;s.D.queue='waiting：已产生超时 AbortReq';});
   add('remove',3.5,5,'ds','移除并回告，再释放 KV','等待分支 pop R1，发送带原始 503 原因的 AbortReq 回告；Decode waiting 已有 KV，因此还调用 release_kv_cache。','dequeue',s=>{s.D.queue='waiting_queue 已移除 R1';s.D.kv='本请求 KV 已走释放路径';s.D.released=true;},opts(['timeout']));
   add('response',5,6.5,'dh','TM 清理与非流式 503','本例未发响应头，前端可用非流式 HTTP 错误表达；流式已经发头后另有 SSE 边界。','echo',s=>{s.D.tm='R1 登记已清理';s.D.frontDone=true;s.D.reply='503 · 等待超时';},opts(['remove'],'Dg','503'));
   add('return',6.5,7.8,'gw','返回 D 错误','本例 P 已成功，D 返回的 503 沿网关 Decode 错误路径呈现。','dResult',s=>{s.client='503 · D 等待超时';s.clientDone=true;s.outcome=answer;},opts(['response'],'gc','503'));
  }else{
   scope+=' Mooncake；双方 deferred-release 开关启用；本例收到完整 drain ACK，未走保留超时分支。';
   initial.P.queue='inflight：R1';initial.P.kv='发送 worker 可能仍在写';initial.D.queue='transfer_queue：R1';initial.D.kv='接收 KV + metadata 仍占用';initial.D.allocated=true;initial.gateway='原业务响应仍未结束';initial.payload='D 本地管理入口请求取消 R1';
   add('local-abort',0,1.2,'dh','D TM 派发 AbortReq','本场景只追踪 D 的取消与资源归还；网关仍需处理 P 结果，不假设存在全局取消广播。','abort',s=>{s.abortSent=true;s.D.tm='取消已派发；等待结果';});
   add('receiver-abort',1.2,2.5,'ds','receiver.abort / 注册 room','D 通知对端取消，并建立 room 的 ACK 记账；仍在途的 P 写入不能忽略。','transferAbort',s=>{s.D.queue='receiver 取消已通知；ACK=0/1';},opts(['local-abort'],'DP','ABORT / room=42'));
   add('defer',2.5,4,'ds','移出 transfer，资源留置','poll Failed 后输出失败终态；满足 deferred 条件时移入 _deferred_releases，快照 required_acks，保留 KV 和 metadata。','defer',s=>{s.D.queue='_deferred_releases：R1';s.D.kv='留置：KV + metadata，不可重用';},opts(['receiver-abort']));
   add('front-done',4,5.2,'dh','D 前端失败已结束','失败输出可先让前端登记结束；留置的 KV 尚未释放。HTTP / ReqState 不负责替代传输安全条件。','transferPoll',s=>{s.D.tm='R1 登记已结束';s.D.frontDone=true;s.D.reply='失败终态已返回';s.gateway='已收到 D 失败；仍需按 P 结果处理';},opts(['defer'],'Dg','失败输出'));
   add('drain-writes',2.5,6.5,'kv','P 传输排空','等待这个 room 的在途传输退出；没有把 AbortReq 到达当作 DMA 已全部停止。','ack',s=>{s.P.kv='该 room 的传输已排空';},opts(['receiver-abort'],null,'等待在途传输','wait'));
   add('ack',6.5,7.7,'kv','P → D：drain ACK','本例只通知一个 P rank，因此所需 ACK 为 1；多 rank 要按通知集合聚合，不能只凭任意一个 ACK 放行。','ack',s=>{s.ack=true;s.D.queue='_deferred_releases；ACK=1/1';},opts(['drain-writes'],'PD','ABORT_ACK'));
   add('release',7.7,9,'ds','再次轮询，归还资源','resolve_deferred_releases 检查安全条件后释放 KV、metadata、room 状态并清理 receiver。源码另有 hold timeout 释放，超时不等同于已收到 drain 证明。','resolve',s=>{s.D.kv='KV / metadata 已归还';s.D.released=true;s.D.queue='留置项已清空';s.outcome=answer;},opts(['ack']));
  }
 }else{
  scope+=' 只对 P1 发 SIGTERM；正常分支关闭 force shutdown，初始非 UnHealthy。';answer='本地 rid_to_state 排空控制本端退出；不是所有 P/D 请求一起结束的分布式屏障。';
  initial.pair='P1 正在退出；D1 继续服务';initial.P.tm='rid_to_state[R1]';initial.P.queue='本地收尾中';initial.P.kv='本地尚有收尾工作';initial.D.tm='rid_to_state[R1]';initial.D.queue='running_batch：R1';initial.D.kv='KV 已交接，继续持有';initial.D.allocated=true;
  add('signal',0,1.2,'ph','P 收到 SIGTERM','设置 P TokenizerManager.gracefully_exit=true；D 不会因该本地变量被同步设置退出。','signal',s=>{s.P.http='gracefully_exit=true；探活 503';s.gateway='尚不能据此保证所有入口已摘流';});
  add('ingress',1.2,2.5,'gw',condition==='arrivals'?'直连调用方仍会发请求':'停止向 P1 新投递','本例显式给定部署侧的摘流条件；不是 sigterm_handler 自动阻止所有直连请求。','drain',s=>{s.gateway=condition==='arrivals'?'直连流量未停止':'部署侧已停止向 P1 新投递';},opts(['signal']));
  add('d-continues',0,10,'dg','D 继续已有生成','P 退出期间，D 已取得 KV 的请求可独立推进；图中不新建跨端全局等待。','drain',s=>{s.D.gpu='本例 D 生成完成';s.D.queue='本例已结束';s.D.tm='R1 登记已结束';s.D.frontDone=true;},opts([],null,'D 已有工作','compute'));
  add('p-finish',1.2,4.5,'ps','P 本地请求完成收尾','选择 P 的传输和响应正常结束的路径；本例本地登记将变空，不用 D 的生成结束作依赖。','pRetire',s=>{s.P.queue='R1 本地已收尾';s.P.kv='R1 请求所有权已释放';s.P.released=true;},opts(['signal']));
  add('tm-empty',4.5,5.5,'ph','P ReqState 排空','watchdog 轮询本端 rid_to_state；不读取 D 的登记。','drain',s=>{s.P.tm='rid_to_state={}';},opts(['p-finish']));
  if(condition==='arrivals'){
   add('late',5.5,6.5,'ph','P 又收到直连 R2','教学条件给定：下一次空表检查前，仍有直连请求登记。本地退出标记不能被画成全局入口闸门。','drain',s=>{s.P.tm='rid_to_state[R2]';s.P.queue='新请求 R2 尚未结束';},opts(['tm-empty']));
   add('wait',6.5,10,'ph','继续等本地登记','演示终点保留等待，不能因为时间轴到头就自动 Shutdown。','drain',s=>{s.outcome='P 仍未满足本地排空；D 不因此停止';},opts(['late'],null,'等 R2','wait'));
  }else{
   add('shutdown',5.5,6.5,'ph','发送 ShutdownReq','登记为空后停止子进程 watchdog，派发 ShutdownReq，并观察 Scheduler 进程退出。','drain',s=>{s.shutdown=true;s.P.queue='ShutdownReq 已派发';},opts(['tm-empty','ingress']));
   if(condition==='stuck')add('deadline',6.5,10,'ph','期限后仍有 Scheduler','源码记录 stragglers 并进入 kill_process_tree；这里只观察升级动作，不假装用户态清理已成功。','drain',s=>{s.P.http='升级强制终止路径';s.outcome='已发送退出 / 强制信号，不等于已验证清理与进程消失';},opts(['shutdown'],null,'等进程退出','wait'));
   else{
    add('cleanup',6.5,8,'ps','Scheduler 退出与 finally','handle_shutdown 设置退出标记，事件循环返回；run_scheduler_process 的 finally 执行资源与分布式环境清理。','shutdown',s=>{s.P.queue='事件循环已退出，清理已完成';},opts(['shutdown']));
    add('observe',8,9,'ph','观察 P Scheduler 已退出','本例完成本地进程退出观察；此时 D 的 GPU 轨道仍可继续。','process',s=>{s.processExited=true;s.P.http='本例 P 本地退出完成';s.outcome=answer;},opts(['cleanup']));
   }
  }
 }
 const duration=Math.max(...events.map(e=>e.end))+.8;
 return {mode,condition,initial,events,sorted:events.slice().sort((a,b)=>a.end-b.end||a.start-b.start),duration,scope,answer,...modes[mode]};
}
function snapshot(tl,value){const time=Math.max(0,Math.min(tl.duration,Number(value)||0)),state=clone(tl.initial);for(const e of tl.sorted)if(e.end<=time+1e-8)e.apply(state);const active=tl.events.filter(e=>e.start<=time&&time<e.end),focus=active.find(e=>e.lane==='gw')||active[0]||tl.sorted.filter(e=>e.end<=time).at(-1)||tl.events[0];return {time,state,active,focus,progress:e=>Math.max(0,Math.min(1,(time-e.start)/(e.end-e.start)))};}
return {revision:M.revision,sources,url,modes,laneNames,timeline,snapshot};
});
