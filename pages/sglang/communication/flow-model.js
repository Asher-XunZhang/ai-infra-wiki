/* Source-pinned teaching timelines. Time is explanatory, never a runtime trace. */
(function(root){'use strict';
const revision='279339f113b79af84f27fd3ac92d0a13bd3f4cbd';
// Placement is an explicit teaching condition, not inferred from a rank or API.
const hardware={
 nodes:{cpu0:{x:36,y:108,w:348,h:112},cpu1:{x:656,y:108,w:348,h:112},gpu0:{x:36,y:354,w:348,h:164},gpu1:{x:656,y:354,w:348,h:164},nic0:{x:282,y:258,w:102,h:56},nic1:{x:656,y:258,w:102,h:56},switch:{x:470,y:258,w:100,h:56}},
 route(kind,placement='rdma',reverse=false){
  const cpu=kind==='control'||kind==='aux',local=placement==='local';
  let points=cpu?(local?[[384,184],[656,184]]:[[333,220],[333,270],[707,270],[707,220]]):[[333,354],[333,302],[707,302],[707,354]];
  let nodes=cpu?['cpu0',...(local?[]:['nic0','switch','nic1']),'cpu1']:['gpu0',...(local?[]:['nic0','switch','nic1']),'gpu1'];
  if(reverse){points=points.slice().reverse();nodes=nodes.slice().reverse();}
  return {points,nodes,plane:kind==='control'?'control':'data',memory:cpu?'host':'device'};
 }
};
const sources={
 transferThreads:['disaggregation/mooncake/conn.py',265,'threading.Thread('],
 kvBatch:['disaggregation/mooncake/conn.py',650,'def _send_kvcache_generic('],
 auxMemory:['disaggregation/utils.py',329,'class MetadataBuffers:'],
 engine:['distributed/device_communicators/mooncake_transfer_engine.py',110,'class MooncakeTransferEngine:'],
 dispatch:['managers/tokenizer_manager.py',601,'def _dispatch_to_scheduler('],
 mark:['managers/tokenizer_manager.py',1620,'def _mark_state_dispatched('],
 encode:['managers/io_struct.py',2485,'def sock_send('],
 decode:['managers/io_struct.py',2493,'def sock_recv('],
 receive:['managers/scheduler_components/request_receiver.py',89,'def recv_requests('],
 req:['managers/scheduler.py',2740,'def handle_generate_request('],
 enqueue:['managers/scheduler.py',3233,'def _add_request_to_queue('],
 ppSend:['managers/scheduler_pp_mixin.py',802,'def _pp_send_dict_to_next_stage('],
 ppRecv:['managers/scheduler_pp_mixin.py',825,'def _pp_recv_typed_dict('],
 ppCommit:['managers/scheduler_pp_mixin.py',705,'def _pp_commit_comm_work('],
 ppLaunch:['managers/scheduler_pp_mixin.py',1221,'def _pp_launch_batch('],
 split:['distributed/parallel_state.py',119,'def _split_tensor_dict('],
 sendObject:['distributed/parallel_state.py',1573,'def send_object('],
 recvObject:['distributed/parallel_state.py',1625,'def recv_object('],
 sendTensor:['distributed/parallel_state.py',1744,'def send_tensor_dict('],
 recvTensor:['distributed/parallel_state.py',1799,'def recv_tensor_dict('],
 reduce:['distributed/parallel_state.py',655,'def all_reduce('],
 inplace:['distributed/parallel_state.py',1023,'def _all_reduce_in_place('],
 ncclReduce:['distributed/device_communicators/pynccl.py',143,'def all_reduce('],
 gather:['distributed/parallel_state.py',1326,'def all_gather('],
 ncclGather:['distributed/device_communicators/pynccl.py',192,'def all_gather('],
 exchange:['distributed/parallel_state.py',1171,'def all_to_all_single('],
 ncclExchange:['distributed/device_communicators/pynccl.py',317,'def all_to_all_single('],
 register:['disaggregation/mooncake/conn.py',325,'def register_buffer_to_engine('],
 metadata:['disaggregation/mooncake/conn.py',2706,'def send_metadata('],
 bootstrap:['disaggregation/prefill.py',450,'def pop_bootstrapped('],
 sendChunk:['disaggregation/prefill.py',1311,'def _send_kv_chunk('],
 queue:['disaggregation/mooncake/conn.py',2385,'def add_transfer_request('],
 worker:['disaggregation/mooncake/conn.py',1847,'def transfer_worker('],
 blocks:['disaggregation/common/utils.py',111,'def group_concurrent_contiguous('],
 pointers:['disaggregation/mooncake/conn.py',755,'def set_transfer_blocks('],
 transfer:['disaggregation/mooncake/conn.py',641,'def _transfer_data('],
 aux:['disaggregation/mooncake/conn.py',1247,'def send_aux('],
 notify:['disaggregation/common/conn.py',507,'def conclude_transfer('],
 success:['disaggregation/common/conn.py',569,'def apply_prefill_status('],
 cacheGate:['disaggregation/decode_hicache_mixin.py',172,'def poll('],
 gate:['disaggregation/utils.py',212,'def _apply_metadata_gate('],
 consensus:['disaggregation/utils.py',237,'def poll_and_all_reduce('],
 commit:['disaggregation/decode.py',2158,'def _commit_transfer_to_req('],
 ready:['disaggregation/decode.py',2364,'def pop_transferred('],
 ack:['disaggregation/common/conn.py',635,'def is_abort_release_safe('],
 drain:['disaggregation/common/conn.py',669,'def _maybe_ack_drained_abort('],
 defer:['disaggregation/decode.py',2501,'def _defer_release('],
 free:['disaggregation/decode.py',2510,'def _do_release('],
 release:['disaggregation/decode.py',2525,'def resolve_deferred_releases(']
};
const catalog={
 ipc:['请求 IPC','TokenizerManager 的请求，怎样变成 Scheduler 的 Req？','同机两个 CPU 进程 · Python 服务 · TP=PP=1 · 合法文本请求 · 无 grammar · msgpack IPC','R1 是请求 ID；[101,102] 是示意 input_ids。传递的是序列化消息，接收端会构造自己的 Req；不传 GPU KV。'],
 tensor:['PP 张量','下一 stage 怎样知道接收什么，又把数据放在哪里？','PP=2、TP=1 · CUDA/NCCL · async_send=True · 未启用 CPU blocking wait','hidden_states 与 residual 均缩小为 FP16 [2,4]。发送端连续投递；接收端按描述分配并建立 stream 依赖。不传权重或 KV。'],
 collective:['组内通信','GroupCoordinator 怎样把张量地址交给 NCCL？','CUDA · TP 组 [0,1] · PyNCCL 已启用 · 非编译、无其他加速规约分支','本图选择一个真实后端分支，不代表所有机器的默认选择。部署选项决定图中的硬件路径；不指定 NCCL ring/tree 或网络分包。'],
 kv:['KV 搬运','D 给出目标位置后，P 怎样把 KV 写到正确地址？','Mooncake · 单源/单目标 · 末块 · 普通批量路径，无 staging / custom pool / 索引分批','普通 MHA、无额外 state、关闭乐观 Prefill；只放大一层的 K buffer：每页 16 B、基址 1000/2000 均为教学缩尺。V 和其他层按各自 buffer 处理。'],
 ready:['Decode 就绪','已经收到 Success，为什么还不能交出请求？','Mooncake · 两个有效 P 来源 · 两个 D 参与者 · 普通接收路径','状态通知、元数据落地、其他 D 参与者的进展可以交错发生。时间轴选择一种到达顺序，不把它们画成固定握手协议。'],
 release:['异常释放','请求终止后，D 为什么暂时不能归还接收地址？','延迟释放已启用 · backend 支持 · abort_notified=True · required_acks=2','只看这一延迟释放分支。ACK 表示传输端完成排空；请求 abort、远端停止写入、地址归还不是同一个事件。']
};
const options={ipc:[['default','msgpack 请求']],tensor:[['default','CPU 描述 + CUDA 张量']],collective:[['reduce','All-reduce · 原位写回'],['gather','All-gather · 分配输出'],['exchange','All-to-all · 调用方提供输出']],kv:[['default','目标页 [3,4,9]'],['fragmented','目标页 [3,9,4]']],ready:[['default','元数据与对端晚到'],['cache','另需 HiCache 恢复'],['corrupt','元数据 room 不匹配']],release:[['default','收齐 drain ACK'],['timeout','截止时仍缺一个 ACK']]};
const p=(id,title,duration,summary,detail,refs)=>({id,title,duration,summary,detail,refs});
function transferPlan(fragmented=false){const src=[7,8,12],dst=fragmented?[3,9,4]:[3,4,9],blocks=[];src.forEach((s,i)=>{let b=blocks.at(-1);if(b&&s===b.src+b.count&&dst[i]===b.dst+b.count)b.count++;else blocks.push({src:s,dst:dst[i],count:1});});return {src,dst,blocks:blocks.map(b=>({...b,srcAddr:1000+b.src*16,dstAddr:2000+b.dst*16,bytes:b.count*16})),bytes:48};}
function build(mode='ipc',option){if(!catalog[mode])throw Error('Unknown mechanism');option=option||options[mode][0][0];if(!options[mode].some(x=>x[0]===option))throw Error('Unknown condition');let phases;
if(mode==='ipc')phases=[
 p('object','准备内部请求对象',1.7,'TokenizerManager 持有 TokenizedGenerateReqInput 和本地 ReqState。','rid、input_ids、sampling_params 随对象派发。该实例选择 msgpack；SGLANG_USE_PICKLE_IPC 可切到另一个编码分支。',['dispatch','encode']),
 p('send','sock_send 编码并派发消息',2.8,'msgpack_encode → ZeroMQ socket.send；跨进程的是编码后的字节。','本地 ReqState 不随消息整体搬走；它继续用于等待与组织用户响应。',['encode','dispatch']),
 p('mark','发送侧标记 dispatched',1.5,'_mark_state_dispatched 只更新发送侧状态。','它不是接收确认。图示一次可能的交错：此时接收端尚未轮询；源码不保证标记与接收之间的先后。',['mark']),
 p('receive','SchedulerRequestReceiver 拉取并解码',2.9,'sock_recv(NOBLOCK) → msgpack_decode → recv_reqs。','没有消息时返回本轮接收结果；多 rank 广播与多模态共享内存分支不在此算例中。',['receive','decode']),
 p('req','handle_generate_request 创建 Req',2.8,'从接收对象读取 rid、input_ids、sampling_params，构造 Scheduler 本地 Req。','相同 rid 连接两个进程的状态，但它们不是共享同一个 Python 对象。选择正常、非 session、非 PD 的请求路径。',['req']),
 p('ready','进入等待调度的请求集合',1.6,'_add_request_to_queue 在 DisaggregationMode.NULL 分支追加到 waiting_queue。','本例参数校验通过、无 grammar、未触发排队上限；之后才选批与执行。',['req','enqueue'])];
if(mode==='tensor')phases=[
 p('split','拆开描述与张量引用',2.1,'PP 发送入口加入 __msg_type__=proxy；_split_tensor_dict 分离张量描述与引用。','发送前 current_stream.wait_event(launch_event) 建立对计算的依赖；元数据保留消息类型，tensor_list 保留原张量。',['ppSend','split','sendTensor','ppLaunch']),
 p('meta','先传长度，再传序列化描述',3.3,'send_object 使用 cpu_group 发送长度和 pickle 字节；recv_object 收齐并反序列化。','即使原 tensor 在 CUDA 上，这段描述仍通过 CPU group。GPU 数据没有塞进这份 pickle。',['sendObject','recvObject']),
 p('allocate','接收 CPU 逐项分配、投递并建立依赖',1.8,'CPU 依次执行 empty → irecv → wait，继续投递下一项；GPU 可仍在接收前一项。','本例选择 NCCL 默认非 CPU 阻塞 wait：它建立当前 stream 对通信的依赖；CPU 可继续分配 residual。不是全设备同步。',['recvTensor']),
 p('hidden','GPU 接收 hidden_states，CPU 已继续',3.2,'接收 CPU 已投递两项及后续 forward；设备仍在处理 hidden_states。','发送端可异步提交多个 tensor；图示接收循环的先后，不表示链路必然完全串行。wait 也不应解释为全设备 cudaDeviceSynchronize。',['sendTensor','recvTensor']),
 p('residual','GPU 接收 residual；forward 仍等依赖',3.2,'residual 接收区先前已由 CPU 分配并投递。本例让两项设备通信依次完成，不暗示 CPU 也要依次等完。','P2PWork 保留发送引用；后续 _pp_commit_comm_work 对 work.wait 后 clear，不能只凭 isend 返回就复用 buffer。',['sendTensor','recvTensor','ppCommit']),
 p('usable','接收依赖满足，GPU forward 才能读取',1.8,'CPU 的 proxy 分流和 forward 提交已经完成；此时 GPU 才满足输入依赖。','遇到 output 等其他类型先放入对应 inbox。forward_stream.wait_stream(schedule_stream) 建立消费依赖；不是等待全设备空闲。',['recvTensor','ppRecv','ppLaunch'])];
if(mode==='collective'){const kind=option,reduce=kind==='reduce',gather=kind==='gather';phases=[
 p('buffer','确认本地输入与输出存储',2.5,reduce?'All-reduce 选原位分支：input 的同一地址也用作 output。':gather?'all_gather 分配两倍行数的 output_tensor，输入保留。':'all_to_all_single 接收调用方提供的 input 和 output；两者元素总数相同。',reduce?'这里关闭 custom/quick/MSCCL++/symmetric 等其他候选，未进入编译分支。':gather?'dim=0；两 rank 的 shape 相同，按 rank 顺序收集。':'本 PyNCCL 分支为等分交换；input.numel 必须能被 world_size 整除。',[kind]),
 p('submit','把地址、数量、类型与 stream 传给 NCCL',3,reduce?'GroupCoordinator → _all_reduce_in_place → PyNcclCommunicator.all_reduce。':gather?'GroupCoordinator.all_gather → all_gather_into_tensor → PyNCCL.all_gather。':'GroupCoordinator → PyNCCL.all_to_all_single → ncclGroupStart / Send / Recv / End。','每个 rank 都调用通信接口。传入的是本地 data_ptr、元素数、dtype、communicator 与 CUDA stream。',[kind,reduce?'ncclReduce':gather?'ncclGather':'ncclExchange']),
 p('queued','CPU 调用返回，stream 中任务待执行',1.8,'NCCL 工作已提交到 CUDA stream；主机返回不等于 GPU 数据已经就绪。','图示 GPU 被前序任务占用的一种合法时序；不宣称每次调用后都有这段等待。后续消费者也排在相同 stream 上。',[reduce?'ncclReduce':gather?'ncclGather':'ncclExchange']),
 p('execute','设备通信执行，写入目标 buffer',4,reduce?'两个局部贡献规约后写回原地址：各卡得到 [11,22]。':gather?'按 rank 顺序填入输出：[1,2,10,20]；两卡输入不变。':'发送面向对端的分块，按来源 rank 放入 output：rank 0 得到 [A0,B0]，rank 1 得到 [A1,B1]。','图中移动的是逻辑贡献/分块，未指定 NCCL 内部算法。中途填充仅示意写入过程；最终可用状态在依赖满足后才显示。',[reduce?'ncclReduce':gather?'ncclGather':'ncclExchange']),
 p('consume','同一 stream 的消费者读取完成结果',2,'设备按 stream 顺序执行后续操作；消费者看到完成后的输出。','没有额外虚构 CPU work.wait 或全设备同步。跨 stream 使用需另行建立依赖，本图固定同一 stream。',[kind])];}
if(mode==='kv')phases=[
 p('registered','池已注册，请求目标已预留',2,'传输引擎已有 buffer 基址与长度；D 为 R1 预留目标位置。','注册使引擎能访问内存；请求分配决定写入哪些位置。两者是不同生命周期。初始画面尚未计算本 chunk；本例关闭乐观 Prefill，先交接目标信息。',['register']),
 p('metadata','D → P：交接目标索引',3,'send_metadata 发送 room、session_id、kv_indices、aux_index 等控制信息。','这里只将目标索引放大显示。目标基址来自此前的 rank registration；控制消息不包含待搬运的 K 向量。',['metadata']),
 p('compute','Bootstrap 就绪后，P 计算本 chunk',2.2,'本例 WaitingForInput 后准入 Prefill；计算产生源端 K/V，末块准备 aux。','缩写展示准入与计算，不模拟模型算子；_send_kv_chunk 在末块调用 metadata_buffers.set_buf，随后交给发送器。',['bootstrap','sendChunk']),
 p('queue','P 把 TransferKVChunk 放入发送队列',2.3,'add_transfer_request 保存本 chunk 的源索引、目标切片和末块信息。','transfer_worker 取出任务，结合 D 的目标索引与注册信息组织实际搬运。没有复制整个 GPU 池。',['queue','worker','transferThreads']),
 p('plan','源和目标同时连续，才合并传输段',3.5,'group_concurrent_contiguous 配对索引；每段生成 src_addr、dst_addr、length。','地址 = buffer 基址 + 页号 × item_len；length = 连续页数 × item_len。改变目标位置只改变传输计划，不改变 A/B/C 内容与总字节数。',['blocks','pointers']),
 p('copy','worker 等待 K/V 批次；Scheduler 可继续',4,'Transfer Engine 按地址计划写入 D 的目标区域，A/B/C 各自落到匹配位置。','batch_transfer_sync 阻塞调用它的传输 worker，不阻塞已入队返回的 Scheduler。普通路径合并各层 K/V 为批次；批次成功后才发末块 aux。',['transfer','engine','kvBatch']),
 p('aux','末块继续搬运 aux 数据',2.3,'send_aux 按源、目标 aux_index 计算地址，搬运 output_id、room 等辅助数据。','普通 CUDA 分支的 MetadataBuffers 位于 CPU 内存。aux 是搬运的数据，由 Transfer Engine 传输；Success 是另一条控制消息。',['aux','auxMemory','worker']),
 p('notify','目标处理成功后，再通知 Success',1.6,'conclude_transfer 更新状态并发送终态通知；D 再按来源 rank 汇合。','仅无已记录失败、room 未清理且目标处理成功时发 Success；conclude_transfer 不允许失败 room 被后到的成功覆盖。',['notify','success'])];
if(mode==='ready')phases=[
 p('waiting','接收请求仍在 transfer_queue',1.8,'R1 等待 room=42；本地 raw poll 尚为 Transferring。','所需 P 来源数为 2；P 成功通知的计数，与 D 参与者的状态归并是两个不同集合。',['success','ready']),
 p('first','P0 的 Success 到达',2.3,'apply_prefill_status 把 prefill_rank=0 放入成功集合。','成功来源数 1/2；本地 room 状态还不能变成 Success。',['success']),
 p('duplicate','P0 再次通知，不增加完成数',1.5,'set 去重：{0} 仍是 {0}。','两条通知不等于两个不同来源完成；重复不会代替 P1。',['success']),
 p('second','P1 成功，本地条件仍未满足',2.5,option==='cache'?'成功集合变为 {0,1}，raw poll 为 Success；HiCache PENDING 先将它降为 Transferring。':'成功集合变为 {0,1}，raw poll 为 Success；metadata gate 把本轮结果改回 Transferring。','P0/P1、元数据和可选 HiCache 各自推进；图示一种到达顺序，检查顺序不代表它们串行准备。',['success','cacheGate','gate']),
 p('metadata','本地元数据可见，仍需其他条件',2.5,option==='cache'?'room 已非零，但 HiCache 仍 PENDING，本地 poll 还不能变成 Success。':'metadata buffer 的 room 非零，本地 gate 放行；另一个 D 参与者仍为 Transferring。','此 gate 仅检查非零，不比较 room 身份；身份不匹配要等最终提交检查。',['gate','consensus']),
 ...(option==='cache'?[p('cache','HiCache 本地恢复完成，解除 poll 门槛',3,'hicache_restore_status 从 PENDING 到 READY；包装 receiver 才保留 raw Success。','HiCacheRestoreGatedKVReceiver 在 metadata gate 和 MIN 之前生效；恢复可与传输重叠，不能画成 MIN 已成功后才开始等待。',['cacheGate','ready'])]:[]),
 p('peers','D 参与者通过 MIN 汇合就绪状态',2.5,'各 D 参与者的本地条件都通过，MIN 结果才为 Success。','实际传 CPU uint8 poll 状态；Failed=0、Transferring=3、Success=4。任一 Failed 也会经 MIN 传播。',['consensus']),
 p('commit','提交前核对 room 身份',2.8,option==='corrupt'?'实际 room=99，与 R1 期望的 42 不符：prepare_abort，拒绝提交。':'actual_room == expected_room == 42，才能把 output_id 等 aux 信息写入 Req。','元数据“非零”和“属于这个请求”是两道检查。这个新请求的首个 output_id 取教学值 320；不包含 rebootstrap 重放分支。',['commit']),
 p('done',option==='corrupt'?'终止 R1，不能进入待调度路径':'pop_transferred 返回可继续的 Req',1.7,option==='corrupt'?'传输完成不代表请求身份有效；该请求不被提交。':'Req.output_ids=[320]，传输阶段结束；调度器再决定何时执行 Decode。','这条时间轴聚焦正常接收与身份门槛；失败后的延迟释放条件见独立“异常释放”过程。',['ready'])];
if(mode==='release')phases=[
 p('failed','请求失败，地址仍可能被远端写入',2.2,'R1 已终止，但接收 KV 和 metadata index 暂时保留。','此例已发送 abort 通知且满足延迟释放条件；不是所有失败请求都走这条路径。',['ready','defer']),
 p('defer','保存 required_acks 和 deadline',2.5,'_defer_release 将请求及截止时间加入 deferred_releases。','required_acks 来自 receiver.bootstrap_infos 的数量。图中为 2；时间刻度是教学时间，不代表配置的实际超时秒数。',['defer']),
 p('ack','收到 P0 drain ACK，继续持有地址',3,'is_abort_release_safe 仍未满足两个来源的排空要求。','源端先将 room 标成 Failed；_maybe_ack_drained_abort 在 outstanding 为零时发 ACK。接收侧按来源身份去重。',['drain','ack','release']),
 p('resolve',option==='timeout'?'达到 deadline，仍缺 P1 ACK':'P1 drain ACK 到达',3.5,option==='timeout'?'该固定版本在超时分支仍会调用 _do_release；这不证明远端已全部排空。':'排空确认收齐，resolve_deferred_releases 可以调用 _do_release。','必须区分 drained 与 timed out，不能把两种结果都标成“安全排空完成”。',['release']),
 p('free','归还 KV，清零 metadata room 并回收索引',2.7,'release_kv_cache(is_insert=False)；metadata room=0；free metadata index；清理 receiver。','释放是归还分配权，不表示 GPU KV 字节已擦除。旧数据在图中变灰且不可再属于 R1。',['free']),
 p('done','R1 不再拥有这些地址',1.8,option==='timeout'?'超时释放完成；排空证据仍只有 1/2。':'收到 2/2 排空确认后，资源归还完成。','新请求可由分配器取得地址；本图不额外假设它已经分配给某个新请求。',['free','release'])];
let end=0;phases=phases.map((x,index)=>({...x,index,start:end,end:(end=Number((end+x.duration).toFixed(6)))}));return {mode,option,execution:execution(mode,phases,option),label:catalog[mode][0],question:catalog[mode][1],config:catalog[mode][2],boundary:catalog[mode][3],phases,duration:end,plan:transferPlan(option==='fragmented')};}
// Execution intervals share the teaching clock, but are not the narration chapters.
// Intervals describe one legal ordering, not measured durations or guaranteed overlap.
function execution(mode,phases,option){
 const f=id=>phases.find(p=>p.id===id),a=id=>f(id).start,b=id=>f(id).end,last=phases.at(-1).end;
 const interval=(id,label,start,end,kind='control')=>({id,label,start,end,kind});
 const lane=(id,label,events)=>({id,label,events});let spans={},lanes=[];
 if(mode==='ipc'){
  spans.receive=[a('mark'),b('receive')];
  lanes=[lane('sender','发送 CPU',[interval('encode','编码 / send',a('send'),b('send')),interval('mark','标记',a('mark'),b('mark')),interval('await','等待响应',b('mark'),last,'wait')]),lane('receiver','接收 CPU',[interval('receive','拉取 / 解码',a('mark'),b('receive')),interval('req','构造 Req',a('req'),b('req')),interval('ready','待调度',a('ready'),last,'wait')])];
 }
 if(mode==='tensor'){
  const postEnd=a('meta')+1.5,hPostEnd=a('allocate')+.6,rPostEnd=a('allocate')+1.2;
  spans={sendPosts:[a('meta'),postEnd],rxHidden:[a('allocate'),hPostEnd],rxResidual:[hPostEnd,rPostEnd],forwardSubmit:[rPostEnd,b('allocate')],hidden:[hPostEnd,b('hidden')]};
  lanes=[lane('sendCPU','发送 CPU',[interval('split','拆描述',0,b('split')),interval('sendPosts','连续 isend',a('meta'),postEnd),interval('hold','调用已返；work 保有引用',postEnd,last,'free')]),lane('recvCPU','接收 CPU',[interval('meta','recv 描述 / CPU wait',a('meta'),b('meta'),'wait'),interval('recvPosts','接收调用',a('allocate'),rPostEnd),interval('forwardSubmit','入队',rPostEnd,b('allocate')),interval('returned','CPU 已投递 forward',b('allocate'),last,'free')]),lane('device','GPU / 链路',[interval('hidden','hidden_states',hPostEnd,b('hidden'),'tensor'),interval('residual','residual',a('residual'),b('residual'),'tensor')]),lane('forward','D forward stream',[interval('dependency','等待接收依赖',b('allocate'),b('residual'),'wait'),interval('usable','可读',b('residual'),last,'ok')])];
 }
 if(mode==='collective')lanes=[lane('cpu','两端 CPU',[interval('buffer','准备地址',0,b('buffer')),interval('submit','NCCL 入队',a('submit'),b('submit')),interval('returned','调用已返回；可继续主机工作',b('submit'),last,'free')]),lane('gpu','同一 GPU stream',[interval('prior','前序设备任务',0,b('queued'),'tensor'),interval('execute','通信 / 写结果',a('execute'),b('execute'),'tensor'),interval('consume','消费者',a('consume'),last,'ok')])];
 if(mode==='kv')lanes=[lane('scheduler','P Scheduler',[interval('launch','提交',a('compute'),a('compute')+.6),interval('queue','任务入队',a('queue'),b('queue')),interval('returned','调用返回；可推进其他请求',b('queue'),last,'free')]),lane('gpu','P GPU',[interval('compute','计算本块',a('compute'),b('compute'),'tensor')]),lane('transport','传输数据面',[interval('copy','K/V 批次写入',a('copy'),b('copy'),'kv'),interval('aux','aux 写入',a('aux'),b('aux'),'aux')]),lane('worker','P transfer_worker',[interval('plan','地址计划',a('plan'),b('plan')),interval('sync','batch_transfer_sync 等待',a('copy'),b('copy'),'wait'),interval('auxWait','aux 等待',a('aux'),b('aux'),'wait'),interval('notify','通知',a('notify'),b('notify'))]),lane('decode','D Scheduler',[interval('targets','预留 / 发布位置',0,b('metadata')),interval('poll','本地循环可继续轮询，R1 仍不可 Decode',b('metadata'),last,'free')])];
 if(mode==='ready'){
  spans={second:[a('first'),b('second')],metadata:[0,b('metadata')],peerLocal:[0,a('peers')]};
  if(option==='cache')spans.cache=[0,b('cache')];
  lanes=[lane('p0','P0 来源',[interval('first','P0 通知',a('first'),b('first')),interval('duplicate','重发',a('duplicate'),b('duplicate'))]),lane('p1','P1 来源',[interval('second','P1 独立推进 / 到达',a('first'),b('second'))]),lane('metadata','D 本地 DRAM',[interval('metadata','元数据等待 / 可见',0,b('metadata'),'aux')]),...(option==='cache'?[lane('restore','D HiCache',[interval('restore','独立恢复，持续到 READY',0,b('cache'),'tensor')])]:[]),lane('peer','D1 / MIN',[interval('peerLocal','其他 D 参与者独立准备',0,a('peers'),'control'),interval('peers','MIN 汇合',a('peers'),b('peers'))]),lane('commit','D CPU 提交',[interval('blocked','等待全部门槛',0,b('peers'),'wait'),interval('commit','身份检查',a('commit'),b('commit')),interval('done',option==='corrupt'?'拒绝':'返回 Req',a('done'),last,option==='corrupt'?'wait':'ok')])];
 }
 if(mode==='release'){
  spans={drain0:[0,a('ack')],drain1:[0,option==='timeout'?last+1:a('resolve')]};
  lanes=[lane('p0','P0 后台',[interval('drain0','排空本源在途工作',0,a('ack'),'tensor'),interval('ack','ACK',a('ack'),b('ack'))]),lane('p1','P1 后台',[interval('drain1',option==='timeout'?'尚无排空证据':'与 P0 独立排空',0,option==='timeout'?last:a('resolve'),'wait'),...(option==='timeout'?[]:[interval('resolve','ACK',a('resolve'),b('resolve'))])]),lane('decode','D CPU / 地址',[interval('held','登记延迟释放；持有地址并轮询 ACK / deadline',0,b('resolve'),'wait'),interval('free','归还地址',a('free'),b('free'),'ok'),interval('done','已归还',a('done'),last,'free')])];
 }
 return {spans,lanes};
}
function sample(tl,time){time=Math.max(0,Math.min(tl.duration,Number(time)||0));const phase=tl.phases.find(x=>time<x.end)||tl.phases.at(-1),progress={};tl.phases.forEach(x=>progress[x.id]=time>=x.end?1:time<=x.start?0:(time-x.start)/x.duration);for(const [id,[start,end]]of Object.entries(tl.execution.spans))progress[id]=Math.max(0,Math.min(1,(time-start)/(end-start)));return {time,phase,progress,finished:time===tl.duration};}
function state(tl,time){const s=sample(tl,time),q=s.progress,done=id=>q[id]===1;
return {...s,dispatched:done('mark'),reqConstructed:done('req'),tensorReady:done('residual'),collectiveReady:done('execute'),kvWritten:done('copy'),auxWritten:done('aux'),successNotified:done('notify'),successCount:done('second')?2:done('first')?1:0,rawSuccess:done('second'),metadata:done('metadata')?(tl.option==='corrupt'?99:42):0,peersReady:done('peers'),restored:tl.option!=='cache'||done('cache'),committed:tl.mode==='ready'&&tl.option!=='corrupt'&&done('commit'),rejected:tl.mode==='ready'&&tl.option==='corrupt'&&done('commit'),ackCount:done('resolve')&&tl.option!=='timeout'?2:done('ack')?1:0,released:tl.mode==='release'&&done('free'),timedOut:tl.mode==='release'&&tl.option==='timeout'&&done('resolve')};}
const url=key=>`https://github.com/sgl-project/sglang/blob/${revision}/python/sglang/srt/${sources[key][0]}#L${sources[key][1]}`;
const api={revision,sources,catalog,options,build,sample,state,transferPlan,url,hardware};if(typeof module==='object'&&module.exports)module.exports=api;else root.CommunicationFlow=api;
})(typeof globalThis!=='undefined'?globalThis:this);
