/* Chapter 2 execution profile: vLLM V1 engine + ModelRunner V2 at 4ba787ec.
 * Local teaching states, not a GPU trace. No independent animation clock. */
(() => {
  'use strict';
  const h = window.React.createElement;
  const BASE = 'https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/';
  const RUN = 'vllm/v1/worker/gpu/model_runner.py';
  const src = {
    choice:['vllm/config/vllm.py#L731','V2 selection / fallback'],
    factory:['vllm/v1/worker/gpu_worker.py#L570','Worker._make_model_runner'],
    execute:[RUN+'#L1764','V2.execute_model'],
    sample:[RUN+'#L2149','V2.sample_tokens'],
    state:['vllm/v1/worker/gpu/states.py#L9','RequestState'],
    addresses:[RUN+'#L1595','V2.prepare_attn'],
    blocks:['vllm/v1/worker/gpu/block_table.py#L229','BlockTables.compute_slot_mappings'],
    async:['vllm/v1/worker/gpu/async_utils.py#L116','AsyncOutput / copy_event'],
    queue:['vllm/v1/engine/core.py#L708','EngineCore.step_with_batch_queue'],
    mp:['vllm/v1/executor/multiproc_executor.py#L1096','WorkerProc.handle_output'],
    connector:['vllm/v1/worker/gpu/kv_connector.py#L54','ActiveKVConnector'],
    ready:['vllm/v1/core/sched/scheduler.py#L3520','Scheduler remote-KV gate'],
    cleanup:[RUN+'#L1170','V2.finish_requests'],
    free:['vllm/v1/core/sched/scheduler.py#L3031','Scheduler._free_request'],
    graph:['vllm/v1/worker/gpu/cudagraph_utils.py#L523','CudaGraphManager.dispatch'],
    replay:[RUN+'#L2053','FULL / PIECEWISE / NONE'],
    metric:['vllm/v1/engine/async_llm.py#L879','AsyncLLM.output_handler'],
  };
  const sourceIds = {
    process:['factory','execute','sample'],wire:['mp','execute','async'],
    prefix:['state','addresses','blocks'],batch:['queue','sample','async'],
    disagg:['connector','ready','execute'],crosscut:['cleanup','free','graph','replay'],
    metrics:['sample','async','metric'],
  };
  const text = {
    process:[
      ['Python 前端仍由 AsyncLLM 组织输入与输出；ModelRunner V2 位于 GPU Worker 内，不承担 HTTP 或请求路由。','AsyncLLM still owns frontend input/output. ModelRunner V2 lives inside the GPU Worker and does not route HTTP requests.'],
      ['EngineCore 的 Scheduler 决定本轮 token 预算与 KV block 分配，再把 SchedulerOutput 交给 Executor。V2 执行器内的请求状态不是第二个调度器。','EngineCore Scheduler owns token budgets and KV allocation. Executor delivers SchedulerOutput; V2 request state is not another scheduler.'],
      ['Worker 调用 V2.execute_model：更新持久请求状态、准备输入与地址、执行前向；普通生成再由 sample_tokens 采样，异步复制结果。','Worker calls V2.execute_model to update persistent state, prepare inputs/addresses and run forward. Ordinary generation then calls sample_tokens and copies results asynchronously.'],
      ['AsyncOutput 等 copy_event 后形成 CPU 可读的 ModelRunnerOutput。Scheduler 更新请求并产生 EngineCoreOutputs，AsyncLLM 反分词后流式返回。','After copy_event, AsyncOutput materializes a CPU-readable ModelRunnerOutput. Scheduler updates requests and emits EngineCoreOutputs; AsyncLLM detokenizes and streams them.'],
    ],
    wire:[
      ['前端 ROUTER → EngineCore DEALER 传输入帧，Core PUSH → 前端 PULL 传输出帧；Worker 改用 V2 不改变这层协议。','Frontend ROUTER → Core DEALER carries inputs; Core PUSH → frontend PULL carries outputs. Worker V2 does not change this protocol.'],
      ['Core 消费 CPU 可读的 ModelRunnerOutput 并更新请求后，把 EngineCoreOutputs 从 PUSH 输出通道送到前端 PULL。返回载荷与输入的 ROUTER/DEALER 通道分开；Worker 的 response MQ 属于前一个执行边界。','After consuming CPU-readable ModelRunnerOutput and updating requests, Core sends EngineCoreOutputs over its PUSH output channel to frontend PULL. This return path is separate from ROUTER/DEALER inputs; worker response MQ belongs to the preceding execution boundary.'],
      ['V2 把新请求写入持久 RequestState，已有请求增量更新 block IDs 与进度；再按 idx_mapping 准备本批 GPU 输入。输出 D2H 完成后，Worker response MQ 才能交回可读结果。','V2 adds persistent RequestState for new requests and updates blocks/progress for existing ones. idx_mapping gathers this batch. Worker response MQ returns readable results only after output D2H completes.'],
      ['DP>1 时，独立的 DPCoordinator 收集引擎负载与 request-wave 状态并发布给前端，供路由参考；它不是搬运 KV 的通道。本页默认 DP=1，此处单独展示可选扩展。Worker 内采用 V2 不改变这层控制边界。','At DP>1, a separate DPCoordinator collects engine load and request-wave state and publishes them to frontends for routing; it does not carry KV. This page defaults to DP=1 and shows the optional extension separately. Worker V2 does not change this control boundary.'],
    ],
    prefix:[
      ['前缀 hash 和可复用 block 仍由 EngineCore 内的 KVCacheManager / BlockPool 管理；V2 不在 GPU 上重新做这套全局前缀查找。','KVCacheManager / BlockPool in EngineCore still own prefix hashes and reusable blocks. V2 does not repeat that global lookup on the GPU.'],
      ['本例只命中前两个完整块，共 8/12 token。Scheduler 把已有 block IDs 和 num_computed_tokens=8 交给 V2，未命中尾部另分配；粒度 4 仅作教学示例。','Only the first two complete blocks hit: 8/12 tokens. Scheduler passes existing block IDs and num_computed_tokens=8 to V2 and allocates the tail. Size 4 is illustrative.'],
      ['V2 先将 R1 映射到持久请求槽位，再把 block IDs 写入该槽的 BlockTables。prepare_attn 按本批 idx_mapping 取表，并把 token position 转成 KV 写入位置；请求槽位不等于 KV 槽位。','V2 maps R1 to a persistent request slot and stages its block IDs in BlockTables. prepare_attn gathers by idx_mapping and converts token positions to KV write slots. A request slot is not a KV slot.'],
      ['采样结果留在 last_sampled_tokens 等状态中。下一轮按本批顺序取请求状态，读取旧 KV 并追加新 KV；跨请求前缀保留与引用计数仍在 Scheduler 侧管理。','Sampled tokens remain in last_sampled_tokens and related state. The next batch gathers request state, reads old KV and appends new KV; cross-request retention and references stay scheduler-owned.'],
    ],
    batch:[
      ['EngineCore 的 Scheduler 仍维护 waiting / running，选择本轮工作；V2 只执行已经确定的 SchedulerOutput。','EngineCore Scheduler still owns waiting/running and chooses the work. V2 executes the resulting SchedulerOutput.'],
      ['SchedulerOutput 带每请求 token 数、新请求资料、增量 block IDs、完成/抢占集合。V2 更新持久状态，并为本批构造输入；预算不由 Runner 再算一遍。','SchedulerOutput carries token counts, new requests, block deltas and finished/preempted IDs. V2 updates persistent state and builds inputs without rescheduling the budget.'],
      ['execute_model 先更新状态与地址，再执行前向并保存 ExecuteModelState。普通生成的 sample_tokens 随后读取它并采样；前向调用返回不等于 CPU 已收到 token。','execute_model updates state/addresses, runs forward and saves ExecuteModelState. sample_tokens then reads it and samples. Returning from forward is not CPU receipt of a token.'],
      ['采样后，输出 copy stream 等待采样依赖并做 D2H；main stream 可推进 postprocess_sampled。启用 async scheduling 的 mp Worker 用独立线程等待 copy_event，Core 的 batch queue 可容纳后续批次。','After sampling, the copy stream waits for sampling then performs D2H while the main stream may run postprocess_sampled. With async scheduling, an mp Worker output thread waits on copy_event and Core can queue further batches.'],
    ],
    disagg:[
      ['实例选择由外部路由完成；DP coordinator 负责 DP 协调。Scheduler 与 Worker 各有 connector 职责，V2 通过 ActiveKVConnector 接入 Worker 侧。','External routing chooses instances; the DP coordinator coordinates DP. Scheduler and Worker have separate connector roles; V2 uses ActiveKVConnector on the Worker side.'],
      ['V2.pre_forward 绑定 metadata：has_sync_kv_loads 时先启动加载，否则延至 post_forward 启动。finished_recving 等结果返回 Scheduler；异步接收请求还要通过就绪检查和预算准入才可计算。','V2 pre_forward binds metadata: sync loads start before forward; other loads start in post_forward. finished_recving returns to Scheduler; async receivers still need readiness checks and scheduling budget before execution.'],
      ['分层缓存是可选后端能力。V2 通过相同 connector 生命周期接入，实际保存、加载与完成条件由具体后端实现；CPU offload 不等同于 P/D 搬运。','Tiering is optional backend behavior exposed through the connector lifecycle. Concrete backends own save/load/completion details; CPU offload is not identical to P/D transfer.'],
      ['V2 能力与部署组合分开判断。该提交的 external_launcher + PP>1 属于自动选择 V2 时的回退条件；不能仅凭图中 TP/PP/DP/EP 名称推断任意组合受支持。','V2 capability is distinct from deployment compatibility. external_launcher + PP>1 triggers automatic runner fallback at this commit; TP/PP/DP/EP labels do not imply support for every combination.'],
    ],
    crosscut:[
      ['Scheduler 释放 block 引用，可保留完整块 hash；另一条通知经 finished_req_ids 让 V2 移除本地请求槽位。Connector 可延迟 block 回收：槽位移除、引用归还和 KV 覆写是不同事件。','Scheduler releases block references while complete hashes may remain. finished_req_ids separately removes the V2 request slot. Connectors may delay block reuse: slot removal, reference release and KV overwrite are different events.'],
      ['Scheduler 将请求标记 PREEMPTED、归零进度并放回 waiting；V2 消费 preempted_req_ids，移除本地槽位。恢复时作为新请求重新建状态，prefill_len 可含先前输出，区别于原 prompt_len。','Scheduler marks PREEMPTED, resets progress and requeues. V2 consumes preempted_req_ids and removes the slot. Resumption rebuilds state; prefill_len may include earlier output and differ from prompt_len.'],
      ['Core 处理 ABORT，Scheduler 标记终态并按 delay_free 条件释放资源；V2 在后续 execute_model 消费完成集合、清理请求槽位。CPU 收到取消不代表已发出的 GPU kernel 被抢占。','Core handles ABORT; Scheduler marks terminal state and obeys delay_free. A later V2 execute_model consumes finished IDs to clean the slot. CPU cancellation does not preempt an already issued GPU kernel.'],
      ['V2 的 CudaGraphManager.dispatch 按 batch 描述匹配已捕获候选。FULL 先填输入缓冲再 replay；PIECEWISE 走编译分段或 breakable；没有兼容候选时 NONE 调 model。捕获是准备工作，不是每轮步骤。','V2 CudaGraphManager.dispatch matches captured candidates by batch descriptor. FULL fills input buffers then replays; PIECEWISE uses compiled or breakable segments; no compatible candidate selects NONE. Capture is preparation, not per-step work.'],
    ],
  };
  function extend(copy, chapters) {
    Object.entries(text).forEach(([id,rows])=>rows.forEach((pair,i)=>{copy.chapters[id].steps[i].vl=pair;}));
    copy.header.checked=['源码复核 2026-11-13 · vLLM V1 引擎 / ModelRunner V2 · 默认 TP2 / PP1 / DP1','Source review 2026-11-13 · vLLM V1 engine / ModelRunner V2 · default TP2 / PP1 / DP1'];
    copy.board.vlRoute=['V1 EngineCore → Worker / ModelRunner V2','V1 EngineCore → Worker / ModelRunner V2'];
    copy.sources.checked=['七个主题按 ModelRunner V2 调用链复核 · 2026-11-13','All seven topics checked along ModelRunner V2 call paths · 2026-11-13'];
    copy.sources.body[0]+=' vLLM 统一选择支持 ModelRunner V2 的普通自回归 GPU 路径；V1 指引擎架构，V2 指 Worker 内的执行实现。自动选择及回退见 VllmConfig.use_v2_model_runner。这里只验证固定源码与教学状态，没有运行 GPU / RDMA 实验。';
    copy.sources.body[1]+=' vLLM uses a supported ordinary autoregressive GPU path with ModelRunner V2. V1 names the engine architecture; V2 names the worker implementation. See VllmConfig.use_v2_model_runner for selection/fallback. This is source and teaching-state validation, not a GPU/RDMA experiment.';
    chapters.forEach(c=>{c.sourceKeys.push(...['choice',...(sourceIds[c.id]||[])].map(k=>'sources.v2_'+k));});
    Object.entries(src).forEach(([k,[,label]])=>{copy.sources['v2_'+k]=['vLLM '+label,'vLLM '+label];});
    copy.nodes.process.vl_workers=text.process[2];
    copy.nodes.wire.vl_workers=text.wire[2];
    copy.nodes.batch.vl_workers=text.batch[2];
    copy.nodes.batch.vl_async=text.batch[3];
    copy.nodes.batch.vl_output=text.process[3];
    copy.nodes.disagg.vl_router=text.disagg[0];
    copy.nodes.disagg.vl_connector=text.disagg[1];
    copy.nodes.crosscut.vl_mrv2=text.crosscut[3];
    // Compilation and replay are distinct; both engines have breakable paths.
    copy.chapters.crosscut.steps[3].diff=['两者都依赖形状、地址和后端约束，也都有 breakable 路径；这里比较各自怎样选择与执行图，不将 BCG 视作单边独有能力。','Both depend on shapes, addresses and backend constraints, and both have breakable paths. Compare selection and execution, not an exclusive BCG capability.'];
    copy.chapters.batch.steps[2].diff=['两边都允许 CPU / GPU 工作交错；提交、GPU 完成和 CPU 消费结果是不同边界，是否重叠取决于调度模式和依赖。','Both allow CPU/GPU interleaving. Submission, GPU completion and CPU result consumption are separate boundaries; overlap depends on mode and dependencies.'];
  }
  const sources=Object.entries(src).map(([k,[path]])=>({key:'sources.v2_'+k,url:BASE+path}));
  const choose=(p,en)=>p[en?1:0];
  function Diagram({chapter,step,en}) {
    const [graphMode,setGraphMode]=React.useState('FULL');
    const [loadMode,setLoadMode]=React.useState('async');
    React.useEffect(()=>{setGraphMode('FULL');setLoadMode('async');},[chapter,step]);
    const id=`rv2-${chapter}-${step}`;
    const tr=(zh,english)=>en?english:zh;
    const items=[];
    const label=(x,y,s,cls='rv-label',anchor='start')=>h('text',{x,y,className:cls,textAnchor:anchor},s);
    const box=(x,y,w,title,sub='',key,active=false,graphPath)=>{
      const g=h('g',{className:'rv-node'+(active?' is-active':''),'data-graph-mode':graphPath},
        h('rect',{x,y,width:w,height:53,rx:6}),label(x+10,y+21,title,'rv-title'),sub?label(x+10,y+40,sub,'rv-sub'):null);
      items.push(key?h('a',{key:items.length,href:BASE+src[key][0],target:'_blank',rel:'noreferrer','aria-label':src[key][1]},g):h('g',{key:items.length},g));
    };
    const arrow=(d,active=false,control=false,edgeId)=>items.push(h('g',{key:items.length,'data-graph-edge':edgeId,className:active?'is-active':undefined},
      h('path',{d,className:'rv-line'+(control?' is-control':''),markerEnd:`url(#${id}-arrow)`}),
      active?h('path',{d,className:'rv-pulse'+(control?' is-control':'')}):null));
    const note=(x,y,zh,english)=>items.push(label(x,y,tr(zh,english)));
    let title='';
    if(chapter==='wire') {
      title=tr('控制数据怎样变成本批 GPU 输入？','How does a plan become GPU input?');
      box(22,56,194,'SchedulerOutput','counts · IDs · block deltas','queue',true);
      box(300,56,198,'RequestState (V2)','R1 → slot s · persistent','state',true);
      arrow('M216 82H300',true,true);note(229,69,'计划','plan');
      box(300,153,198,'idx_mapping','batch row → request slot','execute',true);
      arrow('M399 109V153',false,true);
      box(22,153,194,'Input buffers','input_ids · positions','execute',true);
      arrow('M300 180H216',true);note(225,167,'取状态','gather');
      box(22,255,194,'GPU forward','inputs + addresses → KV','execute',true);
      arrow('M119 206V255',true);
      box(300,255,198,'AsyncOutput → CPU','copy_event → response MQ','async');
      arrow('M216 282H300',true);note(227,268,'采样后','sample');
      note(22,346,'CPU token 存储可用 UVA；并非每轮整表 H2D。','Token storage may use UVA; no full-table H2D each step.');
      note(22,369,'虚线：控制/索引；实线：输入或结果；KV 不经 MQ。','Dashed: plan/index. Solid: values. KV does not pass via MQ.');
    } else if(chapter==='prefix') {
      title=tr('请求槽位、block ID、KV 槽位是三件事','Request slot, block ID and KV slot differ');
      const decode=step===3;
      box(22,57,206,'R1 → request slot s','idx_mapping[batch row] = s','state',true);
      box(300,57,198,'BlockTables[s]','[17, 18, 31'+(decode?', 32':'')+']','blocks',true);
      arrow('M228 84H300',false,true);note(239,70,'索引','index');
      note(22,143,'旧前缀 B17、B18；新尾部 B31；旧分支 B19 保留','Prefix B17/B18; new tail B31; old branch B19 retained');
      box(22,174,206,decode?'position = 12':'positions = 8…11',decode?'12 ÷ 4 → block index 3':'⌊position / 4⌋ → block index 2','addresses',true);
      box(300,174,198,decode?'KV slot = 128':'KV slots = 124…127',decode?'32 × 4 + 0 = 128':'31 × 4 + offsets 0…3','blocks',true);
      arrow('M228 201H300',true,true);note(239,187,'查地址','lookup');
      arrow('M399 110V174',false,true);
      box(22,282,206,decode?'input: sampled token':'input: tail tokens',decode?'last_sampled_tokens[s]':'prompt positions 8…11','execute',true);
      box(300,282,198,decode?'GPU KV[128]':'GPU KV[124…127]',tr('前向生成并写入 K/V','forward computes and writes K/V'),'addresses',true);
      arrow('M228 309H300',true);arrow('M399 227V282',false,true);
      note(22,364,'示例：block_size=4、单 KV group、CP1；编号非真实地址。','Example: block size 4, one KV group, CP1; IDs illustrative.');
    } else if(chapter==='batch') {
      if(step===2) {
        title=tr('前向与采样有先后，结果尚未交给 CPU','Forward precedes sampling and CPU result delivery');
        box(22,56,206,'SchedulerOutput','new / cached / finished IDs','execute',true);
        box(300,56,198,'Persistent state','RequestState + BlockTables','state',true);
        arrow('M228 82H300',true,true);
        box(300,153,198,'prepare_inputs / attn','InputBatch · slot_mappings','addresses',true);
        arrow('M399 109V153',true);
        box(22,153,206,'execute_model','GPU forward → hidden states','execute',true);
        arrow('M300 180H228',true);
        box(22,255,206,'ExecuteModelState',tr('保存本轮引用，供采样读取','step references for sampling'),'execute',true);
        arrow('M125 206V255',true,true);
        box(300,255,198,'sample_tokens','logits → sampled token IDs','sample',true);
        arrow('M228 282H300',true,true);
        note(22,359,'普通自回归、PP1；先前向，再采样，再满足结果可读条件。','Ordinary AR, PP1: forward → sample → readable result.');
      } else {
        title=tr('采样后：两条 CUDA stream 的依赖','After sampling: dependencies across two CUDA streams');
        box(22,55,206,'sample_tokens',tr('同一批 B0 的采样结果','sampled values for batch B0'),'sample',true);
        box(300,55,198,'postprocess_sampled','GPU last_sampled_tokens ← t1','sample',true);
        arrow('M228 82H300',true);note(22,38,'GPU main stream','GPU main stream');
        note(22,160,'GPU output copy stream','GPU output copy stream');
        box(22,179,206,'wait_stream(main)',tr('只等此处之前的采样依赖','wait for prior sampling work'),'async',true);
        arrow('M125 108V179',true,true);
        box(300,179,198,'D2H → copy_event',tr('token IDs 复制到 CPU','copy token IDs to CPU'),'async',true);
        arrow('M228 206H300',true);
        note(250,143,'可重叠','may overlap');
        box(300,286,198,'get_output()',tr('等 event，再读 CPU 数组','wait for event, then read CPU'),'async',true);
        arrow('M399 232V286',true,true);
        box(22,286,206,'Scheduler update','ModelRunnerOutput → output','queue',true);
        arrow('M300 313H228',true);
        note(22,365,'启用 async scheduling：输出线程等结果，Core 可继续派发。','Async scheduling: output thread waits; Core may enqueue more.');
      }
    } else if(chapter==='disagg') {
      const sync=loadMode==='sync';
      title=sync?tr('同步分支：先启动加载，再按依赖消费 KV','Sync loads start before forward; consumers obey dependencies'):tr('异步分支：R2 先计算，R1 等 KV 就绪','Async loads: R2 runs while R1 waits for KV');
      box(22,58,206,'pre_forward','bind metadata · preemptions','connector',true);
      box(300,58,198,sync?'start_load_kv':'GPU forward: R2',sync?'has_sync_kv_loads = true':tr('本轮已准入的其他请求','other work admitted to this step'),sync?'connector':'execute',true);
      arrow('M228 85H300',true,true,'load-first');
      if(sync){
        box(22,196,206,'KV transfer','','connector',true);
        note(32,237,'backend','backend');note(179,237,'GPU','GPU');arrow('M87 233H167',true);
        arrow('M399 111V150H125V196',true,true,'load-start');
        box(300,196,198,'GPU forward',tr('消费前由后端保证依赖','backend enforces read readiness'),'execute',true);
        arrow('M228 223H300',true,true,'load-consume');
        box(22,304,206,'post_forward','collect connector results','connector',true);
        arrow('M399 249V277H125V304',true,true);
        box(300,304,198,'Scheduler update',tr('本轮结果与 connector 状态','step results + connector state'),'ready',true);
      }else{
        box(300,196,198,'post_forward',tr('此处 start_load_kv(R1)','start_load_kv(R1) here'),'connector',true);
        arrow('M399 111V196',true,true,'load-start');
        box(22,196,206,'R1 KV transfer','','connector',true);
        note(32,237,'backend','backend');note(179,237,'D GPU','D GPU');arrow('M87 233H167',true);
        arrow('M300 223H228',true,true,'load-request');
        box(22,304,206,'finished_recving: R1',tr('Worker 结果回传 Scheduler','Worker result → Scheduler'),'ready',true);
        arrow('M125 249V304',true,true);
        box(300,304,198,'R1: WAITING → ready',tr('随后仍检查预算与 KV 资源','then check budget and KV resources'),'ready',true);
      }
      arrow('M228 331H300',true,true,'load-result');
      note(22,143,sync?'启动加载不等于全部 KV 已可读；消费依赖由后端落实。':'has_sync_kv_loads = false；当前 forward 不消费 R1 的远端 KV。',sync?'Starting a load does not make all KV ready; backend enforces reads.':'has_sync_kv_loads = false; this forward does not consume R1 KV.');
      note(22,386,sync?'此图只展示同步分支；捕获和重放均保留 connector 边界。':'示例：R1 接收成功；零 token 轮也可推进 connector。',sync?'Sync branch only; capture and replay preserve connector boundaries.':'Example: R1 receives successfully; zero-token steps can service KV.');
    } else if(chapter==='crosscut'&&step<3) {
      title=tr('Scheduler 资源账本与 V2 本地状态分开清理','Scheduler resource accounting and V2 state clean up separately');
      const preempt=step===1;
      box(22,53,206,preempt?'PREEMPTED':step===2?'FINISHED_ABORTED':'FINISHED',preempt?'num_computed_tokens = 0':step===2?tr('示例：R1 已建立执行侧槽位','Example: R1 has a worker slot'):tr('请求终态已记录','terminal status recorded'),'free',true);
      box(300,53,198,'KVCacheManager.free',tr('减引用；可保留完整块 hash','release refs; retain full hashes'),'free',true);
      arrow('M228 80H300',true,true);
      box(22,168,206,preempt?'preempted_req_ids':'finished_req_ids','SchedulerOutput → Worker','cleanup',true);
      arrow('M125 106V168',true,true);
      box(300,168,198,'V2.finish_requests','_remove_request(R1)','cleanup',true);
      arrow('M228 195H300',true,true);
      box(300,275,198,'RequestState',tr('移除 R1 → 槽位 s；槽位可再用','remove R1 → slot s; reusable slot'),'state',true);
      arrow('M399 221V275',true,true);
      note(22,263,preempt?'回到 waiting':'delay_free 可能延迟 blocks 回收',preempt?'Back to waiting':'delay_free may defer block reuse');
      note(22,287,preempt?'恢复时重建状态':'本地请求槽位 ≠ KV 物理块',preempt?'Rebuild state on resume':'Request slot ≠ physical KV block');
      note(22,365,'清理状态不是擦除 KV，也不是取消已发出的 GPU kernel。','State cleanup neither erases KV nor preempts issued kernels.');
    } else {
      title=tr('V2 怎样选择已捕获的图？','How does V2 select a captured graph?');
      box(22,54,206,'Batch shape','tokens · requests · decode shape','graph',true);
      box(300,54,198,'CudaGraphManager','match captured candidates','graph',true);
      arrow('M228 81H300',true,true,'descriptor');
      box(22,176,145,'FULL','fill inputs → replay','replay',graphMode==='FULL','FULL');
      box(188,176,145,'PIECEWISE','compiled / breakable','replay',graphMode==='PIECEWISE','PIECEWISE');
      box(354,176,144,'NONE','model(**inputs)','replay',graphMode==='NONE','NONE');
      arrow('M399 107V143',true,true,'dispatch');
      [['FULL',94],['PIECEWISE',260],['NONE',426]].forEach(([mode,x])=>{
        arrow(`M399 143H${x}V176`,graphMode===mode,true,mode);
        arrow(`M${x} 229V320`,graphMode===mode,false,mode+'-output');
      });
      box(22,320,476,tr('本轮模型输出','Model output for this step'),tr('三条路径均计算本轮输入；这里展示选中路径的结果','Each path computes this step’s inputs; only the selected path runs'),'replay',true);
      note(22,393,graphMode==='NONE'?'无兼容候选：普通前向继续计算，不跳过本轮。':'图已提前捕获；本轮更新输入与 metadata 后执行。',graphMode==='NONE'?'No compatible capture: ordinary forward still computes this step.':'Graphs were captured earlier; update inputs and metadata each step.');
    }
    const svg=h('svg',{viewBox:'0 0 520 410',className:'ia-svg rv-diagram',role:'group','aria-label':title,'data-v2-topic':chapter,'data-load-mode':chapter==='disagg'?loadMode:undefined,'data-graph-selection':chapter==='crosscut'&&step===3?graphMode:undefined},
      h('title',null,title),h('defs',null,h('marker',{id:id+'-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'},h('path',{d:'M0 0L10 5L0 10Z',fill:'var(--ia-vlScene)'}))),
      label(22,22,title,'rv-heading'),items);
    if(chapter==='disagg')return h(React.Fragment,null,
      h('fieldset',{className:'rv-graph-controls'},h('legend',null,tr('示例：选择加载启动时机','Example: choose when the load starts')),
        h('div',{className:'rv-graph-choices'},[['async',tr('异步：前向后启动','Async: after forward')],['sync',tr('同步：前向前启动','Sync: before forward')]].map(([value,caption])=>h('button',{
          key:value,type:'button','aria-pressed':loadMode===value,'data-load-choice':value,onClick:()=>setLoadMode(value),
        },caption)))),svg);
    if(chapter!=='crosscut'||step!==3)return svg;
    return h(React.Fragment,null,
      h('fieldset',{className:'rv-graph-controls'},
        h('legend',null,tr('示例匹配结果：每轮只选一条路径','Example dispatch result: one path per step')),
        h('div',{className:'rv-graph-choices'},['FULL','PIECEWISE','NONE'].map(mode=>h('button',{
          key:mode,type:'button','aria-pressed':graphMode===mode,'data-graph-choice':mode,onClick:()=>setGraphMode(mode),
        },mode)))),svg);
  }
  function supports(chapter,step) {
    return (chapter==='wire'&&step===2)||(chapter==='prefix'&&step>=2)||(chapter==='batch'&&step>=2)||(chapter==='disagg'&&step===1)||chapter==='crosscut';
  }
  window.AtlasV2={extend,sources,Diagram,supports};
})();
