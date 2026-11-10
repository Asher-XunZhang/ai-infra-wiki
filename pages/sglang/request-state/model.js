/* Source-pinned teaching traces. Durations are reading time, never GPU timings. */
(function (root) {
'use strict';
const revision='279339f113b79af84f27fd3ac92d0a13bd3f4cbd';
const sources={
 init:['managers/scheduler.py',1273,'def init_running_status'],
 enqueue:['managers/scheduler.py',3233,'def _add_request_to_queue'],
 normal:['managers/scheduler.py',1907,'def event_loop_normal'],
 overlap:['managers/scheduler.py',1942,'def event_loop_overlap'],
 war:['managers/scheduler.py',1893,'def _apply_war_barrier'],
 select:['managers/scheduler.py',4021,'self.waiting_queue ='],
 allocate:['managers/schedule_batch.py',2678,'def prepare_for_extend'],
 merge:['managers/scheduler.py',3672,'Merge the new batch'],
 stash:['managers/scheduler.py',3627,'if self.chunked_req is not None'],
 chunk:['managers/scheduler.py',3893,'self.chunked_req.init_next_round_input'],
 chunkAdd:['managers/schedule_policy.py',950,'def add_chunked_req'],
 filter:['managers/schedule_batch.py',3564,'def filter_batch'],
 decode:['managers/schedule_batch.py',3470,'def prepare_for_decode'],
 prefillResult:['managers/scheduler_components/batch_result_processor.py',257,'def process_batch_result_prefill'],
 decodeResult:['managers/scheduler_components/batch_result_processor.py',920,'def process_batch_result_decode'],
 skipResult:['managers/scheduler_components/batch_result_processor.py',990,'if (self.enable_overlap'],
 finish:['managers/schedule_batch.py',1889,'def update_finish_state'],
 release:['mem_cache/common.py',254,'def release_kv_cache'],
 cache:['mem_cache/radix_cache.py',479,'def cache_finished_req'],
 evict:['mem_cache/radix_cache.py',638,'def evict'],
 capacity:['managers/scheduler.py',4141,'def update_running_batch'],
 retract:['managers/schedule_batch.py',3214,'def retract_decode'],
 releaseRetract:['managers/schedule_batch.py',2209,'def release_req'],
 resetRetract:['managers/schedule_batch.py',1931,'def reset_for_retract'],
 requeue:['managers/scheduler.py',4210,'for req in retracted_reqs'],
 resume:['managers/schedule_batch.py',1556,'def init_next_round_input'],
 abortWait:['managers/scheduler.py',5267,'Delete requests in the waiting queue'],
 abortRun:['managers/scheduler.py',5395,'Delete requests in the running batch'],
 refs:['managers/scheduler.py',5248,'def collect_inflight_reqs'],
 copy:['managers/schedule_batch.py',3719,'def copy'],
 pending:['managers/schedule_batch.py',1117,'If we want to abort'],
};
const url=k=>`https://github.com/sgl-project/sglang/blob/${revision}/python/sglang/srt/${sources[k][0]}#L${sources[k][1]}`;
const modes={
 normal:{label:'正常流转',question:'离开 waiting 后，谁继续持有 R1？',scope:'普通文本 · Normal loop · 无前缀命中 · RadixCache 开启 · page_size=1 · 输入 4 个符号 token。'},
 chunk:{label:'分块续算',question:'还没产生首 token，R1 去哪里了？',scope:'普通文本 · Normal loop · 6 个输入按 2 + 2 + 2 分块 · 不混入 Decode · 预算允许每次续算。'},
 retract:{label:'容量回撤',question:'KV 归还后，已输出的文本会消失吗？',scope:'普通文本 · Normal loop · R1 / R2 同在 Decode 集合 · 本例选择 R1 回撤；R2 完成后 R1 重新准入。'},
 cancel:{label:'取消与收尾',question:'取消已收到，为什么还不能直接删掉所有引用？',scope:'普通文本 · Normal loop · 对比等待中取消与已在 Decode 集合中的取消；不覆盖 PD、Grammar 和中间块取消。'},
 overlap:{label:'异步执行',question:'已经提交、CPU 已处理和请求已完成，为什么是三件事？',scope:'普通文本 · Overlap loop · 单一 GPU forward stream · B0/B1/B2 是执行轮次；S0 是可被原地更新的 batch 对象。'}
};
const clone=x=>JSON.parse(JSON.stringify(x));
function initial(mode,condition){
 const n=mode==='chunk'?6:4;
 const s={waiting:false,chunk:false,current:null,running:null,last:null,members:{},results:[],consuming:null,
  output:[],finished:null,pending:null,retracted:false,row:null,allocated:0,written:0,cached:0,protected:false,
  prompt:n,progress:0,deviceToken:null,resultReady:false,other:'',terminal:'未进入调度',resourceNote:'尚未分配请求行与 KV'};
 if(mode==='retract'||(mode==='cancel'&&condition==='running'))Object.assign(s,{current:null,running:'S0',last:'S0',members:{S0:['R1',...(mode==='retract'?['R2']:[])]},output:mode==='retract'?['t1','t2']:['t1'],row:'q0',allocated:mode==='retract'?5:4,written:mode==='retract'?5:4,cached:4,protected:true,progress:4,terminal:'等待下一轮 Decode',resourceNote:'请求行持有索引；缓存前缀受 R1 保护',other:mode==='retract'?'R2 与 R1 同在 S0；只展开 R1 的资源':''});
 return s;
}
function timeline(mode='normal',condition='waiting'){
 if(!modes[mode])mode='normal';condition=condition==='running'?'running':'waiting';
 const events=[];let clock=0;
 const at=(id,start,end,lane,title,detail,source,apply=()=>{},extra={})=>{events.push({id,start,end,lane,title,detail,source,apply,...extra});return end;};
 const step=(id,d,lane,title,detail,source,apply,extra={})=>{const start=clock;clock=at(id,start,start+d,lane,title,detail,source,apply,{...extra,depends:extra.depends|| (events.length?[events.at(-1).id]:[])});};
 const receive=()=>step('enqueue',1.7,'cpu','接收 R1：waiting_queue 加入引用','普通模式 _add_request_to_queue 追加 Req。尚未选批，尚未分配请求行与计算 KV。','enqueue',s=>{s.waiting=true;s.terminal='等待本轮准入';});
 const select=(id='select',batch='S0',n=4,isChunk=false)=>step(id,2.2,'cpu',isChunk?'选入一块，同时保持续算引用':'通过准入，建立本轮 batch',isChunk?'从 waiting 移除 R1；建立 ScheduleBatch，同时 chunked_req 指向同一个 R1。':'从 waiting 移除的只是引用。ScheduleBatch.reqs 持有 R1；prepare_for_extend 分配请求行与 KV 地址。','select',s=>{s.waiting=false;s.chunk=isChunk;s.current=batch;s.members[batch]=['R1'];s.row='q0';s.allocated=n;s.protected=s.cached>0;s.terminal='已准入，尚无 CPU 输出提交';s.resourceNote='请求持有行与 KV 使用权；槽位已分配，写入尚未完成';});
 const compute=(id,n,batch,label,d=2.6)=>step(id,d,'gpu',label,'图中填充表示本轮模型实际建立 KV；已分配地址不等于对应 KV 已写好。GPU 完成仍需结果可读与 CPU 处理。','allocate',s=>{s.written=n;s.progress=Math.min(n,s.prompt);s.deviceToken=id==='decode'&&mode==='cancel'?'t2':id==='decode'||id==='recompute'?'EOS':id.startsWith('chunk-')&&n<6?null:'t1';s.terminal='设备本轮计算完成';},{kvTo:n,batch});
 const ready=(id='ready')=>step(id,1.1,'copy','本轮结果在 CPU 可读','结果处理依赖对应输出。此条带表示结果可读过程，Normal 模式不假设独立 copy stream。','prefillResult',s=>{s.resultReady=true;s.terminal='结果可读，等待 CPU 提交';});
 const first=(id='first',batch='S0')=>step(id,1.9,'cpu','提交首 token，保存上一批引用','process_batch_result_prefill 追加 t1，未结束则缓存已完成前缀；event loop 随后令 last_batch 指向本轮 batch。','prefillResult',s=>{s.output=['t1'];s.resultReady=false;s.cached=s.prompt;s.protected=true;s.last=batch;s.current=null;s.terminal='已产生首 token；等待下一轮选批';s.resourceNote='前缀已进入缓存，R1 的保护仍在';},{token:'t1'});
 const merge=(batch='S0')=>step('merge',1.7,'cpu','下一轮过滤，再合入 running_batch','本例 running 为空，源码直接 running_batch = last_batch；两个变量指向同一个 ScheduleBatch 对象。不是复制 Req。','merge',s=>{s.running=batch;s.current=batch;s.terminal='本轮选择 Decode';});
 const reserve=(n=5)=>step('reserve',1.2,'cpu','为下一输入 token 预留 KV','prepare_for_decode 准备一个新位置；本轮输入是上次输出 t1，新输出还不存在。','decode',s=>{s.allocated=n;s.current=s.running;s.terminal='Decode 输入已准备';});
 const finish=()=>step('finish',1.8,'cpu','CPU 判断 EOS，R1 逻辑完成','先追加本轮输出，再 update_finish_state。此时旧 batch 仍可引用 R1，逻辑完成不等于所有容器已经过滤。','finish',s=>{s.output.push('EOS');s.resultReady=false;s.finished='EOS';s.pending=null;s.terminal='已逻辑完成；待资源收尾';},{token:'EOS'});
 const release=(id='release')=>step(id,1.8,'cpu','结束请求所有权，缓存可以继续保留','release_kv_cache 处理缓存插入/解锁，再归还请求行。本例 page_size=1、无重复前缀，已计算 KV 保留为可驱逐缓存；不代表已归还 allocator。','release',s=>{s.cached=s.written;s.row=null;s.allocated=0;s.protected=false;s.resourceNote='请求行已归还；KV 由 RadixCache 保留，可在驱逐后复用';},mode==='overlap'?{depends:['commit1']}:{});
 const filter=()=>step('filter',1.6,'cpu','后续 filter_batch 移除完成成员','filter_batch 修改成员与对应行视图。这里删除引用，不再次释放 KV，也不证明 Python 对象此刻被回收。','filter',s=>{for(const a of Object.values(s.members)){const i=a.indexOf('R1');if(i>=0)a.splice(i,1);}s.current=null;s.last=null;s.terminal=s.finished?'R1 已完成，观察到的引用已清理':'R1 已离开本场景';});
 if(mode==='normal'||mode==='chunk'){
  receive();select('select','S0',mode==='chunk'?2:4,mode==='chunk');
  if(mode==='chunk'){
   for(let i=0;i<3;i++){
    const batch='S'+i,n=(i+1)*2;
    compute('chunk-'+i,n,batch,`第 ${i+1} 块：建立输入位置 ${i*2}–${n-1} 的 KV`,i===0?2.6:1.9);ready('ready-'+i);
    if(i<2){
     step('middle-'+i,1.5,'cpu','中间块结束，不提交有效首 token','中间块走独立结果分支，减少 inflight_middle_chunks；保留同一 Req 的续算关系。','prefillResult',s=>{s.last=batch;s.current=null;s.deviceToken=null;s.resultReady=false;s.terminal='输入未完；本块没有有效输出';});
     step('stash-'+i,1.8,'cpu','缓存本块，旧 batch 排除 R1','下一轮 stash_chunked_request 保存进度，再 filter 排除续算请求；不是回到 waiting，也不是释放全部 KV。','stash',s=>{s.cached=n;s.protected=true;s.members[batch]=[];s.resourceNote='已算前缀受保护；行与续算引用继续保留';});
     step('continue-'+i,1.6,'cpu',i===1?'最后一块入新 batch，清空 chunked_req':'续算引用直接进入新 batch','add_chunked_req 使用已有 Req 继续选块；最后一块返回 None，因此续算指针在最后一次计算前清空。','chunkAdd',s=>{s.current='S'+(i+1);s.members[s.current]=['R1'];s.chunk=i===0;s.allocated=n+2;s.terminal='下一块已准备；waiting 始终为空';});
    }else first('first',batch);
   }
  }else{compute('prefill',4,'S0','Prefill：为输入建立 KV');ready();first();}
  merge(mode==='chunk'?'S2':'S0');reserve(mode==='chunk'?7:5);
  compute('decode',mode==='chunk'?7:5,mode==='chunk'?'S2':'S0','Decode：写入 t1 的 KV，采出 EOS',2);ready('decode-ready');finish();release();filter();
 }else if(mode==='retract'){
  step('capacity',2,'cpu','下一步 KV 不够，选择 R1 回撤','Decode 容量检查失败；本例 R1 被选中，R2 保留。回撤是可恢复分支，与中止分支分开。','capacity',s=>{s.terminal='R1 被选中回撤';});
  step('retract-release',2,'cpu','归还 R1 私有尾部，解除前缀保护','release_req 调用 release_kv_cache(is_insert=False)。本例只将尚未缓存的尾部交还 allocator；缓存前缀先解除保护。','releaseRetract',s=>{s.row=null;s.allocated=0;s.written=4;s.protected=false;s.resourceNote='尾部已归还；4 个缓存位置变得可驱逐';});
  step('evict',1.4,'cpu','本例驱逐这段可回收前缀','release_req 随后尝试 evict_from_tree_cache；这里选定 R1 前缀被驱逐的合法示例，不表示所有回撤都必须清空其前缀。','evict',s=>{s.cached=0;s.written=0;s.resourceNote='本例前缀也已驱逐；R1 的 KV 空间全部可复用';});
  step('reset',1.7,'cpu','重置执行状态，保留 output_ids','reset_for_retract 设置 is_retracted、重置 KV 与匹配进度。普通 token 输入保留 [t1,t2]；input_embeds 等特殊路径不在本例。','resetRetract',s=>{s.retracted=true;s.progress=0;s.members.S0=['R2'];s.current=null;s.terminal='R1 可恢复；已输出文本仍在';});
  step('requeue',1.5,'cpu','R1 重新进入 waiting_queue','普通模式 _add_request_to_queue(req, is_retracted=True) 再次追加同一个 Req。R2 留在 running_batch。','requeue',s=>{s.waiting=true;s.terminal='等待资源恢复与重新准入';});
  step('r2',2.4,'gpu','R2 继续执行并完成','省略 R2 的内部状态。其完成与 CPU 收尾后释放容量，让 R1 在随后一轮有机会重新准入。','capacity',s=>{s.other='R2 结果随后收尾，容量恢复';});
  step('r2-finish',1,'cpu','处理 R2 结果，下一轮预算恢复','准入仍由调度器决定，不因 R1 重新入队就立即启动计算。','decodeResult',s=>{s.members.S0=[];});
  step('readmit',2.2,'cpu','以 prompt + 已生成文本重建输入','init_next_round_input 组合原输入与 output_ids，再做前缀匹配。本例缓存已驱逐，6 个位置都需重新计算。','resume',s=>{s.waiting=false;s.current='S1';s.members.S1=['R1'];s.row='q1';s.allocated=6;s.retracted=false;s.terminal='重新 Extend；已有 t1、t2 不重复追加';});
  compute('recompute',6,'S1','重新建立 KV；文本输出账本不倒退',2.6);ready();
  step('resume-output',1.8,'cpu','接着已有文本，追加 EOS','输出保持 [t1,t2] 后追加 EOS；不会重发 t1、t2。本例重新 Extend 的结果已触发结束。','prefillResult',s=>{s.output.push('EOS');s.resultReady=false;s.finished='EOS';s.last='S1';s.current=null;s.terminal='重新准入后正常结束';},{token:'EOS'});release();filter();
 }else if(mode==='cancel'&&condition==='waiting'){
  receive();
  at('other-gpu',1.7,6.5,'gpu','R2 的设备工作继续','R1 从未入选本轮 batch；取消 R1 不要求先等无关的 R2 计算结束。','normal',()=>{}, {batch:'R2',depends:['enqueue']});
  step('abort-message',1.2,'cpu','Scheduler 收到 AbortReq(R1)','此场景 R1 还在 waiting_queue，尚无请求行与计算 KV。','abortWait',s=>{s.terminal='收到取消；尚未选批';},{depends:['enqueue']});
  step('abort-pop',1.8,'cpu','直接移除 waiting 引用并回告','源码直接 pop、清理缓存侧请求状态，再向 TokenizerManager 发送 AbortReq 回告。此分支不要求先设置 Req.finished_reason。','abortWait',s=>{s.waiting=false;s.terminal='已取消，等待引用已移除';s.other='TokenizerManager 收到取消回告后清理前端状态';});
  clock=7;
 }else if(mode==='cancel'){
  step('abort-mark',2,'cpu','先设置 to_finish = FINISH_ABORT','collect_inflight_reqs 从 running_batch、last_batch 收集 Req。此时 finished_reason 仍为空，资源与执行引用继续保留。','abortRun',s=>{s.pending='ABORT';s.terminal='取消待提交；不能提前丢失响应与收尾';});
  reserve(5);compute('decode',5,'S0','本例继续一轮 Decode，沿结果路径收尾',2.5);ready();
  step('abort-commit',2,'cpu','结果处理把待取消标记变成终态','普通结果处理先追加本轮 token，再 update_finish_state 优先采用 to_finish 并清空它。内部账本追加不等于向用户交付一条成功生成结果。','finish',s=>{s.output.push('t2');s.resultReady=false;s.finished='ABORT';s.pending=null;s.terminal='取消已提交；由公共结果路径释放资源';},{token:'t2'});release();filter();
 }else{
  // Activities on different execution entities overlap. All mutations commit at event end.
  receive();select(); // 0..3.9
  const q=(id,forward)=>({id,forward,batch:'S0',ready:false,token:null});
  at('submit0',3.9,4.5,'cpu','提交 B0，保存结果引用','result_queue.append((batch.copy(), result))；Q0 为浅快照，仍引用同一个 R1。入队时 GPU 可以尚未完成。','overlap',s=>{s.results.push(q('Q0','B0'));s.last='S0';},{depends:['select']});
  at('gpu0',4.5,9,'gpu','B0 · Prefill','同一 forward stream 按顺序执行。B1 读取的 future token 必须先由 B0 产生。','overlap',s=>{s.written=4;s.progress=4;s.deviceToken='t1';},{kvTo:4,batch:'B0',depends:['submit0']});
  at('plan1',4.9,6.4,'cpu','准备下一轮 B1','CPU 可提前组织元数据；本例 running 为空，合入的是同一个 S0。共享缓冲写入仍受 WAR 事件约束。','war',s=>{s.running='S0';s.current='S0';s.allocated=5;},{depends:['submit0']});
  at('submit1',6.4,7,'cpu','B1 提交；GPU 等 B0','CPU 返回不表示 B1 开始执行；设备流与 FutureMap 保证输入依赖。','overlap',s=>{s.results.push(q('Q1','B1'));},{depends:['plan1']});
  at('take0',7,7.4,'cpu','取出 Q0，开始处理旧结果','popleft 先发生，随后 process_batch_result 内等待 copy_done；Q0 已不在 result_queue，却仍由 CPU 消费函数持有。','overlap',s=>{s.consuming=s.results.shift();},{depends:['submit1']});
  at('wait0',7.4,10.4,'cpu','等 Q0.copy_done','CPU 等待旧结果期间，已提交的 B1 无需等待 CPU 追加 t1。','prefillResult',()=>{},{kind:'wait',depends:['take0']});
  at('copy0',9,10.4,'copy','B0 · t1 → CPU','copy_done 只表示相应 CPU 副本就绪。','prefillResult',s=>{s.consuming.ready=true;s.consuming.token='t1';},{depends:['gpu0'],resultId:'Q0'});
  at('gpu1',9,14.4,'gpu','B1 · Decode','GPU 读取 B0 设备输出，写 t1 的 KV 并产生 EOS；不要求 CPU 已提交 t1。','overlap',s=>{s.written=5;s.deviceToken='EOS';},{kvTo:5,batch:'B1',depends:['gpu0','submit1']});
  at('commit0',10.4,11.7,'cpu','CPU 提交 t1，B1 仍可计算','这一窗口展示 CPU 处理旧结果与 GPU 执行下一批的合法重叠。','prefillResult',s=>{s.output=['t1'];s.cached=4;s.protected=true;s.consuming=null;s.terminal='CPU 已提交 t1；R1 尚未完成';},{token:'t1',depends:['wait0','copy0']});
  at('plan2',11.7,12.8,'cpu','尚未知道 EOS，准备 B2','Req.finished_reason 仍为空，CPU 可以继续规划下一轮；不能依据未来的 EOS 提前过滤。','overlap',s=>{s.allocated=6;},{depends:['commit0']});
  at('submit2',12.8,13.4,'cpu','提交 B2，保留 Q2','S0 原地更新用于新一轮；Q1/Q2 分别保存执行快照，Req 本身没有复制。','copy',s=>{s.results.push(q('Q2','B2'));},{depends:['plan2']});
  at('take1',13.4,13.8,'cpu','取出 Q1，等 EOS 可读','结果按已提交顺序消费。','overlap',s=>{s.consuming=s.results.shift();},{depends:['submit2']});
  at('wait1',13.8,15.7,'cpu','等 Q1.copy_done','B2 已提交，设备仍按 B1 → B2 执行。','decodeResult',()=>{},{kind:'wait',depends:['take1']});
  at('copy1',14.4,15.7,'copy','B1 · EOS → CPU','只有可读副本就绪后，CPU 才检查停止条件。','decodeResult',s=>{s.consuming.ready=true;s.consuming.token='EOS';},{depends:['gpu1'],resultId:'Q1'});
  at('gpu2',14.4,17,'gpu','B2 · 已提前提交的一轮','本例这轮在 CPU 完成 EOS 收尾前结束；其额外结果不会变成 R1 的有效输出。','skipResult',s=>{s.written=6;s.deviceToken='额外结果';},{kvTo:6,batch:'B2',depends:['gpu1','submit2']});
  at('commit1',15.7,17.3,'cpu','CPU 提交 EOS，R1 逻辑完成','CPU 与 GPU 的完成时刻分开。这里选择设备计算已结束再进入资源收尾的合法教学排程，不声称所有运行都具有相同时长。','finish',s=>{s.output.push('EOS');s.resultReady=false;s.finished='EOS';s.consuming=null;s.terminal='R1 已完成；Q2 仍待排空';},{token:'EOS',depends:['copy1','wait1']});
  at('copy2',17,19.5,'copy','B2 · 额外结果回传','R1 逻辑结束时，已提交的结果拷贝仍可在途。','skipResult',s=>{const q=[s.consuming,...s.results].find(q=>q?.id==='Q2');if(q){q.ready=true;q.token='额外结果';}},{depends:['gpu2'],resultId:'Q2'});
  clock=17.3;release(); // ends 19.1
  step('filter',.9,'cpu','下一轮过滤完成请求，不再提交新批','无新 batch 也不能直接 idle：last_batch 和尚未消费的 Q2 仍需处理。','filter',s=>{s.members.S0=[];s.current=null;});
  step('take2',.6,'cpu','取出 Q2，排空已提交结果','队列为空与当前消费函数仍持有结果是两种状态。','overlap',s=>{s.consuming=s.results.shift();});
  step('drain',1.5,'cpu','跳过已完成 R1，不追加额外 token','process_batch_result_decode 在 overlap 分支跳过 finished / retracted 的 Req。输出保持 [t1, EOS]。','skipResult',s=>{s.consuming=null;s.last=null;s.terminal='结果已排空；本例进入 idle';});
 }
 const sorted=events.slice().sort((a,b)=>a.end-b.end||a.start-b.start);
 const duration=Math.max(clock,...events.map(e=>e.end))+1;
 return {mode,condition,events,sorted,duration,initial:initial(mode,condition),...modes[mode]};
}
function snapshot(tl,time){
 const t=Math.max(0,Math.min(tl.duration,Number.isFinite(Number(time))?Number(time):0)),s=clone(tl.initial),done=[];
 for(const e of tl.sorted)if(e.end<=t+1e-8){e.apply(s);done.push(e.id);}
 const active=tl.events.filter(e=>e.start<=t&&t<e.end);
 const focus=active.filter(e=>e.lane==='cpu').at(-1)||active.at(-1)||tl.sorted.filter(e=>e.end<=t).at(-1)||tl.events[0];
 return {state:s,t,active,focus,done,p:id=>{const e=tl.events.find(e=>e.id===id);return e?Math.max(0,Math.min(1,(t-e.start)/(e.end-e.start))):0;}};
}
const api={revision,sources,url,modes,timeline,snapshot};
if(typeof module!=='undefined')module.exports=api;else root.RequestState=api;
})(typeof globalThis!=='undefined'?globalThis:this);
