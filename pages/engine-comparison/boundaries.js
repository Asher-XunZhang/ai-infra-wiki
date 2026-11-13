/* Chapter 2: role ownership and metric boundaries, pinned to the atlas releases.
 * This is an explanatory model, not a runtime trace or performance benchmark.
 * Its active paths use the parent atlas pulse clock; it owns no animation timer. */
window.AtlasBoundaries = (() => {
  const h = React.createElement;
  const choose = (pair, en) => pair[en ? 1 : 0];
  const SG = 'https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/';
  const VL = 'https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/';
  const sources = [
    ['sgTiming', SG+'observability/req_time_stats.py#L803', 'SGLang 时间账本', 'SGLang timestamps'],
    ['sgTM', SG+'managers/tokenizer_manager.py#L2731', 'SGLang 前端停表', 'SGLang frontend endpoint'],
    ['sgCollector', SG+'observability/metrics_collector.py#L2161', 'SGLang ITL 样本', 'SGLang ITL observations'],
    ['sgStreaming', SG+'managers/scheduler_components/output_streamer.py#L445', 'SGLang 输出合并条件', 'SGLang output batching conditions'],
    ['vlSampler', VL+'vllm/v1/worker/gpu/sample/sampler.py#L217', 'vLLM V2 普通采样输出', 'vLLM V2 ordinary sampler output'],
    ['vlRunnerChoice', VL+'vllm/config/vllm.py#L730', 'vLLM Runner 选择与回退', 'vLLM runner selection and fallback'],
    ['vlTiming', VL+'vllm/v1/metrics/stats.py#L507', 'vLLM 时间账本', 'vLLM timestamps'],
    ['vlAsync', VL+'vllm/v1/engine/async_llm.py#L879', 'vLLM 前端停表', 'vLLM frontend endpoint'],
    ['sgReceiver', SG+'managers/scheduler_components/request_receiver.py', 'SGLang rank 请求同步', 'SGLang rank request sync'],
    ['sgCacheFactory', SG+'mem_cache/registry.py#L83', 'SGLang 实际缓存类型选择', 'SGLang cache implementation selection'],
    ['vlUni', VL+'vllm/v1/executor/uniproc_executor.py', 'vLLM UniProcExecutor', 'vLLM UniProcExecutor'],
    ['vlParallel', VL+'vllm/config/parallel.py#L1054', 'vLLM 执行器选择', 'vLLM executor selection'],
    ['sgMemory', SG+'mem_cache/kv_cache_configurator.py#L2991', 'SGLang 内存预算', 'SGLang memory budget'],
    ['vlMemory', VL+'vllm/v1/worker/gpu_worker.py#L647', 'vLLM 内存预算', 'vLLM memory budget'],
  ];
  const steps = [
    {
      title:['首 token：在哪里停表？','First token: where does the clock stop?'],
      sg:['普通 Python 文本流式路径：Scheduler 输出 token，经过 Detokenizer，TokenizerManager 处理首输出时记录 first_token_time。','For ordinary Python text streaming, Scheduler output passes through Detokenizer; TokenizerManager records first_token_time when handling the first output.'],
      vl:['AsyncLLM 取到首批 EngineCoreOutputs 后创建 IterationStats。TTFT 使用这个时间戳，早于这一批的反分词。','AsyncLLM creates IterationStats after retrieving the first EngineCoreOutputs. TTFT uses that timestamp, before this batch is detokenized.'],
      diff:['两边都不等于客户端首段文字到达时间；相同名称没有保证相同的停表位置。','Neither is client time to first content. Identical metric names do not imply identical observation boundaries.'],
    },
    {
      title:['请求结束：客户端已经收到吗？','Completion: has the client received it?'],
      sg:['TM 处理终态输出时记录 finished_time。普通文本已由 Detokenizer 还原；HTTP 发送与客户端读取仍在后面。','TM records finished_time while handling the final output. Ordinary text has passed through Detokenizer; HTTP delivery and client reading still follow.'],
      vl:['E2E 使用最终输出批次的 iteration_timestamp − arrival_time；终点在该批反分词和最终 HTTP 返回之前。','E2E uses the final output batch’s iteration_timestamp minus arrival_time, before that batch’s detokenization and final HTTP delivery.'],
      diff:['内部 E2E 与客户端 E2E 分开记录。GPU 完成、前端停表、用户收到，是三个事件。','Record internal and client E2E separately. GPU completion, frontend observation, and client receipt are different events.'],
    },
    {
      title:['排队：只记录第一次等待吗？','Queueing: does it record only the first wait?'],
      sg:['queue_time_seconds 在首次 forward 入口记录首次等待。请求元数据的累计 queue_time 是另一笔账，可累加已结算的再次等待。','queue_time_seconds observes the first wait at first forward entry. Cumulative queue_time in request metadata is a separate ledger and can include settled later waits.'],
      vl:['request_queue_time_seconds 使用首次 SCHEDULED − QUEUED，在请求完成时记录；抢占不覆盖 scheduled_ts。','request_queue_time_seconds uses first SCHEDULED minus QUEUED and is recorded on completion. Preemption does not overwrite scheduled_ts.'],
      diff:['首次排队短，不表示中途没等待。vLLM 的后续等待可能落在 Prefill / Decode / Inference 区间，它们不是纯 GPU 时间。','A short initial queue does not rule out later waiting. Later vLLM waits may fall inside Prefill / Decode / Inference intervals, which are not pure GPU time.'],
    },
    {
      title:['ITL：一个样本到底是什么？','ITL: what does one observation mean?'],
      sg:['在 TM 的观测层，间隔按新增 completion token 数分摊，再贡献相应数量的 Histogram 观测。','At the TM observation layer, the interval is divided by the new completion-token count and contributes that many histogram observations.'],
      vl:['参照路径为普通非投机 Decode：核心这次输出 1 个 token，取相邻 engine_core_timestamp 差并贡献一个 ITL 观测。SGLang 的合并选项不改变这条路径。','The reference is ordinary non-speculative decode: this core output contains one token and contributes one ITL observation from successive engine_core_timestamp values. SGLang’s batching option does not change this path.'],
      diff:['这里比较两种采样单位，不是同工作量压测：左侧可合并多个 token，右侧保留逐 token 核心输出。均分间隔不是逐 token 实测，不能据此比较速度或 p99。','This compares observation units, not equal-workload performance: the left may batch tokens; the right keeps per-token core outputs. Averaged intervals are not individual measured latencies and do not rank speed or p99.'],
    },
  ];
  function extend(copy, chapters) {
    chapters.push({id:'metrics',num:'07',navKey:'chapters.metrics.nav',questionKey:'chapters.metrics.question',steps:4,
      sourceKeys:['sgTM','vlAsync','sgTiming','vlTiming','sgCollector','sgStreaming','vlSampler','vlRunnerChoice'].map(s=>'sources.'+s),
      lessonLinks:[{href:'../sglang/performance-engineering/index.html',label:['性能观测：P/D 指标实现','Performance: P/D instrumentation']}]});
    chapters[0].sourceKeys.push('sources.sgEngine','sources.sgReceiver','sources.vlUni','sources.vlParallel','sources.sgCacheFactory');
    chapters[2].sourceKeys.push('sources.sgMemory','sources.vlMemory','sources.sgCacheFactory');
    copy.chapters.metrics={nav:['延迟口径','Metric boundaries'],question:['指标究竟测了哪一段？','Which interval does a metric measure?'],steps};
    copy.reading.titles.metrics=['当前指标的字段、公式与边界','Fields, formulas, and scope for this metric'];
    copy.header.checked=['原架构复核 2026-11-05 · 本次归属/指标补充复核 2026-11-13 · 默认 TP=2 / PP=1 / DP=1；归属视图可选 TP=1','Original architecture review 2026-11-05 · ownership/metrics additions checked 2026-11-13 · default TP=2 / PP=1 / DP=1; ownership view also offers TP=1'];
    copy.sources.body[0]+=' 进程归属视图另提供 TP=1 uni 对照；前缀缓存示例采用普通 Full Attention、无外部缓存后端的 UnifiedRadixCache，特殊模型/后端另经 registry 选择；延迟口径以图上列出的普通流式、非 PD 条件为准。';
    copy.sources.body[1]+=' The ownership view also offers TP=1 uni. Prefix-cache examples use UnifiedRadixCache for ordinary full attention with no external backend; registry selects other model/backend paths. Metrics use the ordinary streaming, non-PD conditions shown in the diagrams.';
    copy.sources.checked=['原六主题核对 2026-11-05；归属、内存预算与指标补充核对 2026-11-13','Original six topics checked 2026-11-05; ownership, memory budgets, and metric additions checked 2026-11-13'];
    copy.scenes.metrics={note:{sg:['Python 文本服务 · 流式 · 非 PD','Python text streaming · non-PD'],vl:['Python 前端 · 独立 EngineCore · 非 PD','Python frontend · separate EngineCore · non-PD']},foot:{sg:['按采样事件对照，不按图形长度比较速度','Compare observation events, not diagram lengths'],vl:['按采样事件对照，不按图形长度比较速度','Compare observation events, not diagram lengths']}};
    copy.scenes.ownership={note:{sg:['同一实例的 TP ranks 协同推进；不是普通 DP 副本','TP ranks of one instance coordinate; these are not DP replicas'],vl:['TP1 采用 uni；TP2 采用本机 mp。两种布局不混用','TP1 uses uni; TP2 uses local mp. Keep their boundaries distinct']},foot:{sg:['账本是 rank-local 对象；数据驻留对应 GPU','Ledgers are rank-local objects; data resides on each GPU'],vl:['EngineCore 管分配；Worker 管本地执行与数据','EngineCore owns allocation; Worker handles local execution and data']}};
    sources.forEach(([id,,zh,en])=>copy.sources[id]=[zh,en]);
  }
  function Path({label, active=false, write=false}) {
    // Ownership is a containment view, not a second topology. Cross-process
    // routes are named captions; only the adjacent runner→GPU write has a path.
    if(!write)return h('p',{className:'ib-handoff'+(active?' is-active':'')},label);
    return h('div',{className:'ib-path'+(active?' is-active':''),'data-active':active},
      h('svg',{viewBox:'0 0 20 42','aria-hidden':true},
        h('path',{d:'M10 0V40',className:'ib-path-base'}),
        active?h('path',{d:'M10 0V40',className:'ib-path-pulse'}):null,
        h('path',{d:'M6 35L10 41L14 35',className:'ib-path-head'})),h('span',null,label));
  }
  function Role({level,name,detail,active}) {
    return h('div',{className:'ib-role'+(active?' is-active':''),'data-role':level},
      h('span',{className:'ib-role-level'},level),h('div',null,h('strong',null,name),h('small',null,detail)));
  }
  function Ownership({side,step,tp,en}) {
    const sg=side==='sg', tr=p=>choose(p,en), admitted=step>=1, written=step>=2;
    const ledger=h(Role,{level:'L3',name:sg?'UnifiedRadixCache · Pools':'KVCacheManager · BlockPool',detail:admitted?tr(['R1 已取得地址；账本标记持有','R1 has addresses; ownership recorded']):tr(['空闲位置与前缀索引','Free positions and prefix index']),active:step===1});
    const scheduler=h(Role,{level:'L2',name:'Scheduler',detail:admitted?tr(['R1 进入本轮 batch','R1 admitted to this batch']):tr(['候选队列与 token 预算','Candidates and token budget']),active:step===1});
    const worker=rank=>h('div',{className:'ib-worker',key:rank},
      h(Role,{level:'L4',name:sg?'TpModelWorker → ModelRunner':'Worker → ModelRunner',detail:tr(['准备输入并执行本 rank 模型','Prepare inputs and run this model rank']),active:step===2}),
      h(Path,{label:tr(['地址定位后写 K/V','Write K/V at assigned addresses']),active:step===2,write:true}),
      h('div',{className:'ib-gpu','data-written':written},h('strong',null,`GPU ${rank}`),
        h('span',null,written?tr([`R1 的 K/V · rank ${rank} 本地数据`,`R1 K/V · rank ${rank} local data`]):tr(['尚未写入 R1 的 K/V','R1 K/V not written yet'])),
        h('div',{className:'ib-cells','aria-hidden':true},[0,1,2,3].map(i=>h('i',{key:i,className:written?'is-written':admitted?'is-reserved':''})))));
    const process=(label,children,key)=>h('section',{className:'ib-process',key},h('h4',null,label),children);
    return h('div',{className:'ib-ownership','data-tp':tp},
      h('div',{className:'ib-external'},h('b',null,tr(['L5 · 可选集群路由','L5 · Optional cluster router'])),h('small',null,tr(['选服务副本，不选本轮 token；不计入下方进程数','Selects a replica, not this step’s tokens; excluded from counts']))),
      process(tr([sg?'主进程 · HTTP + TokenizerManager':'API 进程 · AsyncLLM',sg?'Main process · HTTP + TokenizerManager':'API process · AsyncLLM']),
        h(Role,{level:'L1',name:tr(['请求 R1 → token IDs','Request R1 → token IDs']),detail:sg?tr(['回程由独立 Detokenizer 转文本','A separate Detokenizer handles returning text']):tr(['InputProcessor / OutputProcessor · 反分词也在此','InputProcessor / OutputProcessor · detokenization here']),active:step===0||step===3})),
      h(Path,{label:tr(['R1 · 请求消息（ZMQ）','R1 · request message (ZMQ)']),active:step===0}),
      sg?h(React.Fragment,null,
        tp===2?h('p',{className:'ib-rank-sync'},tr(['rank 0 接收，再同步请求输入；两边协同推进同一批次','rank 0 receives and synchronizes inputs; both ranks coordinate the same batch'])):null,
        h('div',{className:'ib-ranks'},Array.from({length:tp},(_,rank)=>process(`Scheduler ${tr(['进程','process'])} · rank ${rank}`,
          h(React.Fragment,null,scheduler,ledger,h('p',{className:'ib-local'},tr(['进程内调用 · 本地地址账本','In-process calls · local address ledger'])),worker(rank)),rank)))):
        h(React.Fragment,null,process('EngineCore '+tr(['进程','process']),h(React.Fragment,null,scheduler,ledger,
          tp===1?h(React.Fragment,null,h('p',{className:'ib-local'},'UniProcExecutor · '+tr(['进程内调用','in-process calls'])),worker(0)):null)),
          tp===2?h(React.Fragment,null,h(Path,{label:'SchedulerOutput · '+tr(['本轮计划 + block IDs（IPC）','step plan + block IDs (IPC)']),active:step===2}),
            h('div',{className:'ib-ranks'},[0,1].map(rank=>process(`Worker ${tr(['进程','process'])} · rank ${rank}`,worker(rank),rank)))):null),
      sg?process('Detokenizer '+tr(['进程','process']),h(Role,{level:'L1',name:tr(['token IDs → 文本 → TokenizerManager','token IDs → text → TokenizerManager']),detail:tr(['独立反分词；与输入侧共同承担 L1','Separate detokenization; shares L1 responsibility with input']),active:step===3})):null,
      h(Path,{label:sg?tr(['Scheduler → Detokenizer → TokenizerManager','Scheduler → Detokenizer → TokenizerManager']):tr(['Worker → EngineCore → API 输出处理','Worker → EngineCore → API output processing']),active:step===3}),
      h('p',{className:'ib-count','data-process-count':sg?tp+2:tp===1?2:4},tr([`本图服务主进程：${sg?tp+2:tp===1?2:4} 个`,`Serving processes shown: ${sg?tp+2:tp===1?2:4}`])),
      h('p',{className:'ib-note'},sg?tr(['每 rank 的账本是本地对象。TP ranks 不是独立服务副本，入口同步后仍有 collective；KV 在各 GPU 上的切分或复制取决于模型。','Each ledger is a local object. TP ranks are not independent serving replicas; collectives remain. KV partitioning or replication on GPUs depends on the model.']):tr(['TP1 的 Worker 对象留在 EngineCore；TP2 mp 另起 Worker 进程。KV 在各 GPU 上可切分或复制，数据驻留位置与账本分配权分开。','The TP1 Worker stays inside EngineCore; TP2 mp creates Worker processes. KV may be sharded or replicated on GPUs; residence and allocation authority are distinct.'])));
  }
  function metricModel(side,step,n=2) {
    const sg=side==='sg';
    if(step===3)return {samples:sg?Array(n).fill(200/n):[200],count:sg?n:1,sum:200};
    if(step===2)return {start:0,end:1,nodes:sg?
      [['进入 wait_queue','Enter wait_queue'],['首次 forward 入口','First forward entry'],['执行后回撤 → 再次等待','Retraction → another wait'],['重新进入 forward','Enter forward again']]:
      [['QUEUED','QUEUED'],['首次 SCHEDULED','First SCHEDULED'],['PREEMPTED → 再次等待','PREEMPTED → another wait'],['再次 SCHEDULED','Scheduled again']]};
    return {start:0,end:sg?3:2,nodes:[
      sg?['TM：created_time','TM: created_time']:['前端：arrival_time','Frontend: arrival_time'],
      [step===0?'核心产生首输出':'核心产生终态输出',step===0?'Core produces first output':'Core produces final output'],
      ...(sg?[[ 'Detokenizer：token → 文本','Detokenizer: tokens → text'],[step===0?'TM：首输出停表':'TM：终态停表',step===0?'TM: first-output timestamp':'TM: final-output timestamp']]:
        [[step===0?'AsyncLLM：首批取出停表':'AsyncLLM：最终批次取出停表',step===0?'AsyncLLM: first-batch timestamp':'AsyncLLM: final-batch timestamp'],['OutputProcessor：反分词','OutputProcessor: detokenize']]),
      [step===0?'HTTP → 客户端首内容':'HTTP → 客户端完整读取',step===0?'HTTP → first client content':'HTTP → client finishes reading'],
    ]};
  }
  function ownershipText(side,step,tp,en) {
    const sg=side==='sg';
    const notes=[
      sg?['TokenizerManager 将 R1 变成内部请求，经 ZMQ 交给入口 Scheduler。','TokenizerManager converts R1 into an internal request and sends it to the entry Scheduler via ZMQ.']:['API 前端处理输入，再交给独立 EngineCore。','The API frontend processes input and submits it to the separate EngineCore.'],
      sg?[`TP=${tp}：每个 rank 都有 Scheduler 与缓存账本，协同选择同一批次。`,`TP=${tp}: each rank has a Scheduler and cache ledger; ranks coordinate the same batch.`]:['EngineCore 内的 Scheduler 与 KVCacheManager 决定本轮 token 和 block IDs。','Scheduler and KVCacheManager inside EngineCore decide this step’s tokens and block IDs.'],
      sg?['本 rank 内调用 TpModelWorker；GPU 按本地地址写入本 rank 的 K/V 数据。','Each rank calls its local TpModelWorker; the GPU writes rank-local K/V at assigned addresses.']:
        tp===1?['UniProcExecutor 在 EngineCore 进程内调用 Worker，没有 Core→Worker 的进程 IPC。','UniProcExecutor calls Worker inside EngineCore, with no Core-to-Worker process IPC.']:['MultiprocExecutor 将本轮计划与地址发给两个 Worker；GPU 分别写入本地 K/V。','MultiprocExecutor sends the plan and addresses to two Workers; each GPU writes local K/V.'],
      sg?['token IDs 经独立 Detokenizer 转成文本，再返回 TokenizerManager。','Token IDs pass through a separate Detokenizer and return as text to TokenizerManager.']:['Worker 结果返回 EngineCore，再送 API 前端反分词与输出。','Worker results return through EngineCore to the API frontend for detokenization and output.'],
    ];
    return choose(notes[step],en);
  }
  function Metrics({side,step,en,tokens}) {
    const sg=side==='sg',tr=p=>choose(p,en),m=metricModel(side,step,tokens);
    if(step===3)return h('div',{className:'ib-metricScene','data-samples':m.count},
      h('p',{className:'ib-scene-question'},tr(['假设：非首输出的观测间隔 200 ms','Assumption: non-first output interval is 200 ms'])),
      h('p',null,sg?tr(['观察者：TokenizerManager','Observer: TokenizerManager']):tr(['观察点：EngineCore 输出时间戳','Observation: EngineCore output timestamps'])),
      h(Path,{label:sg?tr([`stream_interval=${tokens} · 此次新增 ${tokens} 个 token`,`stream_interval=${tokens} · this output adds ${tokens} tokens`]):tr(['普通 Decode · 此次核心输出 1 个 token','Ordinary decode · this core output adds one token']),active:true}),
      h('div',{className:'ib-samples'},m.samples.map((v,i)=>h('div',{key:i},h('b',null,`${v} ms`),h('small',null,tr(['一个观测','One observation']))))),
      h('dl',{className:'ib-counters'},h('dt',null,'Δ_count'),h('dd',null,m.count),h('dt',null,'Δ_sum'),h('dd',null,'0.2 s')),
      h('p',{className:'ib-note'},sg?tr(['示例位于首输出之后且非终态：满足未发送 token 数 ≥ stream_interval 后回传。这是平均分摊，没有测到每个 token 单独的生成时刻。','This is a non-first, non-final output: it is emitted once unsent tokens reach stream_interval. Allocation of the interval does not measure each token’s generation time.']):tr(['此参照路径每个核心输出对应一个新 token。投机等单轮多 token 路径不在本例范围内。','In this reference path, each core output has one new token. Speculative and other multi-token-per-step paths are outside this example.'])),
      h('p',{className:'ib-note'},tr(['两边的 200 ms 分别取自各自观察层，仅用于解释记账差异，非共同实测。','Each 200 ms comes from its own observation layer. This illustrates accounting, not a shared measurement.'])));
    return h('div',{className:'ib-metricScene','data-endpoint':m.end},
      h('p',{className:'ib-scene-question'},step===2?tr(['首次 queue Histogram 的计时区间','Interval for the initial queue histogram']):tr([step===0?'内部 TTFT 的终点':'内部 E2E 的终点',step===0?'Internal TTFT endpoint':'Internal E2E endpoint'])),
      h('ol',{className:'ib-event-list'},m.nodes.map((pair,i)=>h('li',{key:i,className:(i<=m.end?'is-measured ':'')+(i===m.end?'is-end':''),'data-event':i},
        h('span',{className:'ib-event-marker'},i===m.start?tr(['起','start']):i===m.end?tr(['止','end']):i+1),
        h('strong',null,tr(pair)),i===m.end?h('small',null,tr(['本指标停表','Metric endpoint'])):null))),
      h('p',{className:'ib-note'},step===2?tr(['后面的再次等待未扩大这条首次 queue 区间；图中长度不表示等待时长。','A later wait does not extend this initial queue interval; lengths do not encode duration.']):tr(['竖线是同一请求的先后依赖；它不是耗时比例，也不表示前端处理时 GPU 必须空闲。','The line shows dependencies for one request, not duration or a requirement for GPU idleness during frontend work.'])),
      step!==2?h('div',{className:'ib-overlap'},tr(['可交叠：前端处理已返回的输出时，核心/GPU 可继续后续轮次；终态不再开启该请求的新轮次。内部首输出也不保证已有可见文字。','May overlap: the core/GPU may advance while the frontend handles returned output. A final output starts no new iteration for that request. First internal output need not contain visible text.'])):null);
  }
  const formulas = {
    sg:[['first_token_time − created_time','tokenizer_manager.py::_get_scheduler_first_token_time','流式取 TM 首输出；非流式且返回的 prefill_finished_time 有效时，可取 Scheduler 时间，否则回退 TM。','Streaming uses TM first output. Non-streaming may use a valid returned prefill_finished_time; otherwise it falls back to TM.'],
      ['finished_time − created_time','req_time_stats.py::get_e2e_latency','终点是 TM 处理终态，不是 HTTP 完成。created_time 可来自 received_time，不是网络首字节的统一定义。','The endpoint is TM final-output handling, not HTTP completion. created_time may use received_time; it is not a universal first-network-byte timestamp.'],
      ['forward_entry_time − wait_queue_entry_time','req_time_stats.py::set_forward_entry_time','首次 forward 记录一个 Histogram 样本；queue_duration_s 累加已结算等待段，不能混作同一指标。','First forward records one histogram sample; queue_duration_s accumulates settled waits. They are different metrics.'],
      ['interval / num_new_tokens × num_new_tokens 次观测','output_streamer.py::accept → tokenizer_manager.py::collect_metrics → metrics_collector.py::observe_inter_token_latency','普通 Decode 也能用 stream_interval 合并回传；首 token 立即发送，终态强制发送。本例取中途的满间隔输出，使用默认 Prometheus collector，跳过 num_new_tokens ≤ 0。','Ordinary decode can batch returns with stream_interval. The first token is immediate and final output is forced. This example uses an intermediate full interval and the default collector; nonpositive token counts are skipped.']],
    vl:[['iteration_timestamp − arrival_time','stats.py::update_from_output','AsyncLLM 取回输出后建立 IterationStats；统计使用该批时间戳，随后才反分词。','AsyncLLM creates IterationStats on retrieval; statistics use that batch timestamp, before detokenization.'],
      ['iteration_timestamp − arrival_time','stats.py::update_from_finished_request','匹配同一正常请求时，可用 E − Q − I 看剩余区间；它混合准备、传递与等待，不是纯 IPC 时间。不同分位数不可相减。','For the same normal request, E − Q − I is a residual interval mixing preparation, transfer, and waiting—not pure IPC time. Do not subtract quantiles.'],
      ['scheduled_ts − queued_ts','stats.py::update_from_events / update_from_finished_request','scheduled_ts 只保留首次；完成时记录。中途抢占等待计入核心 Inference 等区间，不是 GPU kernel 时间。','scheduled_ts preserves the first schedule; recording occurs at completion. Mid-request preemption waits fall inside core inference intervals, not GPU kernel time.'],
      ['engine_core_timestamp − last_token_ts','gpu/model_runner.py → gpu/sample/sampler.py → Scheduler.update_from_output → IterationStats.update_from_output','支持条件下默认走 ModelRunner V2，普通采样返回单 token；V1 回退的普通采样同样每请求一 token。Scheduler 产出对应 EngineCoreOutput，非首输出追加一个 ITL；前端 SSE 聚合不改变核心时间戳采样单位。','When supported, ModelRunner V2 is the default and ordinary sampling returns one token; the V1 fallback also returns one token per request. Scheduler emits EngineCoreOutput and non-first outputs add one ITL. Frontend SSE batching does not change the core-timestamp observation unit.']],
  };
  function MetricReading({step,en}) {
    const tr=p=>choose(p,en);
    return h('div',{className:'ib-reading'},h('div',{className:'ib-reading-pair'},['sg','vl'].map(side=>{
      const f=formulas[side][step];return h('article',{key:side},h('h3',null,side==='sg'?'SGLang':'vLLM'),h('code',null,f[0]),h('p',null,tr(f.slice(2))),h('small',null,f[1]));})),
      h('p',null,tr(['比较用户体验时，以同一客户端定义的首内容／完整响应区间为外部口径；内部指标用于解释各自边界，先对齐 stream、结束原因、输出合并与样本 labels。','For user experience, use identical client definitions of first content and complete response. Internal metrics explain their own boundaries; align streaming, finish reason, output batching, and labels.'])),
      h('p',null,tr(['仅对隔离、无重启且满足记录条件的每请求指标，Δ_count=1 才能将 Δ_sum 归给单请求。ITL 不符合这个单请求计数假设。','Only for isolated per-request metrics with no reset and valid recording conditions can Δ_count=1 attribute Δ_sum to one request. ITL does not follow this request-count rule.'])),
      h(Provenance,{en}));
  }
  function Provenance({en}) {
    const tr=p=>choose(p,en);
    return h('p',{className:'ib-provenance'},tr(['补充问题来自','Questions informed by ']),' ',
      h('a',{href:'https://mp.weixin.qq.com/s/pbzBiFfBR-KpVBpxYGSk4Q',target:'_blank',rel:'noreferrer'},tr(['《五层进程切分》','the five-layer article'])),
      tr(['；实现重新核对本章两份固定提交。文章使用的 v0.5.22 / v0.32 不替代本章版本。延伸：','; implementations rechecked at this chapter’s two commits. The article’s v0.5.22 / v0.32 do not replace this baseline. Further reading: ']),
      h('a',{href:'https://github.com/Asher-XunZhang/ai-infra-wiki/blob/main/大模型推理/推理基础/SGLang%20与%20vLLM%20V1%20架构差异源码学习文档.md',target:'_blank',rel:'noreferrer'},tr(['架构差异学习文档（保留独立基线）','architecture study (independent baselines)'])));
  }
  function MemoryBudget({en}) {
    const tr=p=>choose(p,en);
    return h('section',{className:'ib-reading'},h('h3',null,tr(['两个 0.9，为什么不代表相同 KV 容量？','Why do two values of 0.9 not mean equal KV capacity?'])),
      h('div',{className:'ib-reading-pair'},h('article',null,h('h4',null,'SGLang · --mem-fraction-static'),h('p',null,tr(['静态份额用于权重与 KV。普通路径先预留运行时 slack，再从模型加载后的空闲显存中计算 KV 可用空间。','The static fraction covers weights and KV. The ordinary path reserves runtime slack before deriving KV space from post-load free memory.'])),h('code',null,'slack = pre_model_load_memory × (1 − fraction)')),
        h('article',null,h('h4',null,'vLLM · --gpu-memory-utilization'),h('p',null,tr(['目标 GPU 内存预算扣除 profile 得到的非 KV 占用，再推导 KV 容量；图捕获估算是否扣入由该版本开关决定。','The target GPU budget subtracts profiled non-KV consumption to derive KV capacity. Whether graph estimates are charged depends on the version’s option.'])),h('code',null,'KV budget = requested_memory − non-KV − applied graph estimate'))),
      h('p',{className:'ib-note'},tr(['这是定容逻辑，不是两条按比例画出的实测显存条。Mamba、多模态、显式 KV 字节与捕获后定容另有分支；比较时核对实际 token/block 容量。','This explains sizing, not measured proportional memory bars. Mamba, multimodal, explicit KV bytes, and post-capture sizing have other branches; compare actual token/block capacities.'])),h(Provenance,{en}));
  }
  return {extend,sources:sources.map(([id,url])=>({key:'sources.'+id,url})),Ownership,Metrics,MetricReading,MemoryBudget,Provenance,metricModel,ownershipText};
})();
