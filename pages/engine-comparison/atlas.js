    const React = window.React;
    const h = React.createElement;
    const { useCallback, useEffect, useMemo, useRef, useState } = React;
const ATLAS_LANG = (() => { try { const saved = localStorage.getItem('ai-infra-engine-lang'); return saved === 'en' ? 'en' : 'zh'; } catch { return 'zh'; } })();
const atlasLocale = {
  lang: ATLAS_LANG,
  dictionaries: {},
  listeners: new Set(),
  register(_ns, dictionaries) { this.dictionaries = dictionaries; },
  bind(_ns) { return (key) => (this.dictionaries[this.lang] || {})[key] ?? key; },
  subscribe(listener) { this.listeners.add(listener); return () => this.listeners.delete(listener); },
  setLang(lang) {
    this.lang = lang === 'en' ? 'en' : 'zh';
    try { localStorage.setItem('ai-infra-engine-lang', this.lang); } catch {}
    document.documentElement.lang = this.lang === 'zh' ? 'zh-CN' : 'en';
    this.listeners.forEach((listener) => listener());
  },
};
const chapterFromHash = () => {
  let id = '';
  try { id = decodeURIComponent(location.hash.slice(1)); } catch {}
  return Math.max(0, CHAPTERS.findIndex((chapter) => chapter.id === id));
};

    const NS = 'inference-atlas';
    const PANEL_ID = 'inference-atlas';

    const RELEASES = {
      sg: {
        name: 'SGLang',
        version: 'v0.5.23',
        tag: 'v0.5.23',
        date: '2026-11-02',
        commit: '297e54fd105e974cdf401e5b4d20253a22c16f90',
        commitShort: '297e54f',
        commitUrl: 'https://github.com/sgl-project/sglang/commit/297e54fd105e974cdf401e5b4d20253a22c16f90',
        url: 'https://github.com/sgl-project/sglang/releases/tag/v0.5.23',
      },
      vl: {
        name: 'vLLM',
        version: 'v0.33.0',
        tag: 'v0.33.0rc3',
        date: '2026-11-02',
        commit: '4ba787ec9959e566d3b587a796dff07673324da5',
        commitShort: '4ba787e',
        commitUrl: 'https://github.com/vllm-project/vllm/commit/4ba787ec9959e566d3b587a796dff07673324da5',
        url: 'https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3',
      },
    };

    const CHAPTERS = [
      {
        id: 'process',
        num: '01',
        navKey: 'chapters.process.nav',
        questionKey: 'chapters.process.question',
        steps: 4,
        sourceKeys: ['sources.sgScheduler', 'sources.vlCore'],
        lessonLinks: [
          { href: '../sglang/system-overview/index.html', label: ['系统总览', 'System overview'] },
          { href: '../sglang/request-runtime/index.html', label: ['请求运行时', 'Request runtime'] },
        ],
      },
      {
        id: 'wire',
        num: '02',
        navKey: 'chapters.wire.nav',
        questionKey: 'chapters.wire.question',
        steps: 4,
        sourceKeys: ['sources.sgIpc', 'sources.vlShm'],
        lessonLinks: [
          { href: '../sglang/communication/index.html', label: ['通信与传输', 'Communication'] },
        ],
      },
      {
        id: 'prefix',
        num: '03',
        navKey: 'chapters.prefix.nav',
        questionKey: 'chapters.prefix.question',
        steps: 4,
        sourceKeys: ['sources.sgRadix', 'sources.vlBlockPool'],
        lessonLinks: [
          { href: '../sglang/kv-memory/index.html', label: ['KV Cache 与内存', 'KV cache and memory'] },
        ],
      },
      {
        id: 'batch',
        num: '04',
        navKey: 'chapters.batch.nav',
        questionKey: 'chapters.batch.question',
        steps: 4,
        sourceKeys: ['sources.sgScheduler', 'sources.vlSched'],
        lessonLinks: [
          { href: '../sglang/scheduling/index.html', label: ['调度与批处理', 'Scheduling and batching'] },
        ],
      },
      {
        id: 'disagg',
        num: '05',
        navKey: 'chapters.disagg.nav',
        questionKey: 'chapters.disagg.question',
        steps: 4,
        sourceKeys: ['sources.sgEngine', 'sources.vlSched'],
        lessonLinks: [
          { href: '../sglang/pd-dataflow/index.html', label: ['PD 数据流', 'PD data flow'] },
          { href: '../sglang/inference-overview/deployment.html', label: ['分离部署', 'Disaggregation deployment'] },
        ],
      },
      {
        id: 'crosscut',
        num: '06',
        navKey: 'chapters.crosscut.nav',
        questionKey: 'chapters.crosscut.question',
        steps: 4,
        sourceKeys: ['sources.sgKvCommon', 'sources.sgGraph', 'sources.vlCompile'],
        lessonLinks: [
          { href: '../sglang/model-execution/index.html', label: ['模型执行', 'Model execution'] },
          { href: '../sglang/serving-operations/index.html', label: ['服务运维', 'Serving operations'] },
        ],
      },
    ];

    const COPY = {
      reading: {
        process: {
          title: ['进程职责与请求边界', 'Process ownership and request boundaries'],
          lede: ['对应上图的入口、调度、执行与回程，核对每一步由谁负责。', 'Match entry, scheduling, execution, and the return path to their owners in the diagram.'],
        },
        batch: {
          title: ['准入、预算与执行交叠', 'Admission, budgets, and execution overlap'],
          lede: ['先区分本步放入哪些请求，再看 CPU 准备与 GPU 执行怎样接续。', 'First identify the requests admitted to a step, then follow CPU preparation and GPU execution.'],
        },
        disagg: {
          title: ['跨实例扩展与 KV 交接边界', 'Cross-instance scaling and KV handoff'],
          lede: ['路由、传输和缓存分层承担不同职责；对照上图逐一辨认。', 'Routing, transfer, and cache tiers have different responsibilities; identify each in the diagram.'],
        },
        optional: ['选读', 'Further reading'],
        expand: ['展开阅读', 'Read more'],
        collapse: ['收起', 'Collapse'],
        sources: ['本节来源与版本边界', 'Sources and version scope for this topic'],
        titles: {
          process: ['进程骨架：组件职责与请求边界', 'Process spine: ownership and request boundaries'],
          wire: ['通信面：通道、载荷与执行边界', 'Communication: channels, payloads, and execution boundaries'],
          prefix: ['前缀缓存：索引、共享保护与回收', 'Prefix cache: indexing, shared ownership, and reclamation'],
          batch: ['调度心跳：准入预算与 CPU/GPU 交叠', 'Scheduling: admission budgets and CPU/GPU overlap'],
          disagg: ['分离扩展：路由、缓存分层与 KV 交接', 'Disaggregation: routing, cache tiers, and KV handoff'],
          finish: ['正常结束：KV 的归属与回收', 'Completion: KV ownership and reclamation'],
          overload: ['内存压力：回撤、抢占与重算', 'Memory pressure: retraction, preemption, and recomputation'],
          abort: ['取消请求：队列移除与资源释放', 'Cancellation: queue removal and resource release'],
          graph: ['图捕获：执行边界与回退条件', 'Graph capture: execution boundaries and fallback conditions'],
        },
      },
      sidebar: {
        title: ['推理引擎对照台', 'Inference engine atlas'],
      },
      header: {
        kicker: ['发布基线 · 2026 年 11 月', 'Release baseline · November 2026'],
        titleA: ['同一条请求，', 'One request,'],
        titleB: ['两种调度心跳', 'two scheduling heartbeats'],
        lede: [
          '把固定 release 的 SGLang v0.5.23 与 vLLM v0.33.0 拆成分阶段探索的 2D 剖面：请求如何穿过进程，ZMQ / 共享内存 / NCCL / RDMA 怎样接力，KV 如何被寻址，以及它在收尾、压力、取消和图形捕获时走向哪里。',
          'A step-by-step 2D anatomy of SGLang v0.5.23 and vLLM v0.33.0: how a request crosses process boundaries, how ZMQ, shared memory, NCCL, and RDMA hand off work, how KV is addressed, and where it goes during cleanup, pressure, cancellation, and graph capture.',
        ],
        caution: ['机制示意，不是性能排名。', 'A mechanism explainer, not a performance ranking.'],
        releaseLabel: ['固定 release 与 commit', 'Pinned release and commit'],
        checked: ['静态源码核对 2026-11-05 · 教学拓扑 TP=2 / PP=1 / DP=1 · 单节点', 'Static source check 2026-11-05 · teaching topology TP=2 / PP=1 / DP=1 · single node'],
        sourceCta: ['查看 release', 'Open release'],
      },
      control: {
        chapter: ['章节', 'Chapters'],
        prev: ['上一步', 'Previous step'],
        next: ['下一步', 'Next step'],
        hint: ['点击上一步、下一步或阶段名称切换；图中动态仅示意当前阶段。', 'Use Previous, Next, or a stage name to navigate; motion illustrates the current stage.'],
      },
      board: {
        sgRoute: ['RadixAttention 路线', 'RadixAttention lineage'],
        vlRoute: ['PagedAttention / V1 路线', 'PagedAttention / V1 lineage'],
        hint: ['点击图中的模块查看注释。', 'Select any module in the diagram to inspect its role.'],
        zoom: ['放大图表', 'Zoom in'],
        fit: ['显示全图', 'Fit diagram'],
        fullView: ['完整图表；可用图表标题旁的按钮放大', 'Full diagram; use the adjacent button to zoom in'],
        scrollHint: [
          '左右滑动查看完整图 · 聚焦后可用方向键滚动',
          'Swipe sideways for the full diagram · focus and use arrow keys to scroll',
        ],
      },
      stage: {
        step: ['步骤', 'Step'],
        sgNow: ['SGLang 此刻', 'SGLang now'],
        vlNow: ['vLLM 此刻', 'vLLM now'],
        divergence: ['分叉点', 'Fork point'],
        nodeLabel: ['模块注释', 'Module note'],
        close: ['关闭注释', 'Close note'],
      },
      links: {
        sources: ['固定 release 源码', 'Pinned release source'],
        lessons: ['第三章深入', 'Chapter 3 deep dives'],
        note: [
          '课程链接按各自固定基线整理，版本与源码事实以本页 release 为准。',
          'Lesson links keep their own pinned baselines; releases and source facts follow this page.',
        ],
      },
      chapters: {
        process: {
          nav: ['进程骨架', 'Process spine'],
          question: ['请求先越过哪些进程边界？', 'Which process boundaries does a request cross first?'],
          steps: [
            {
              title: ['到达入口', 'Arrive at the edge'],
              sg: [
                'FastAPI 接住请求；TokenizerManager 在主进程完成 tokenize，再把 token 化请求经 ZMQ 推给调度器。',
                'FastAPI accepts the request. TokenizerManager runs in the main process, tokenizes it, and pushes the tokenized request to the scheduler over ZMQ.',
              ],
              vl: [
                'Python OpenAI 兼容 HTTP 端点接住请求；AsyncLLM 做输入处理，再生成 EngineCoreRequest 送到独立 EngineCore 进程。Rust frontend 是可选部署路径，不在本主图。',
                'A Python OpenAI-compatible HTTP endpoint accepts the request. AsyncLLM processes input and sends an EngineCoreRequest to the separate EngineCore process. The Rust frontend is an optional deployment path and is not drawn in this main topology.',
              ],
              diff: [
                '两边都把用户请求变成内部请求；SGLang 的 tokenize 属于主进程边界，vLLM 的 EngineCore 边界在输入处理之后。',
                'Both turn a user request into an internal request. SGLang tokenizes at the main-process boundary; vLLM places the EngineCore boundary after input processing.',
              ],
            },
            {
              title: ['进入调度', 'Enter scheduling'],
              sg: [
                '固定教学拓扑为 TP=2：两个 TP rank 各有一个进程，每个进程内同时有 Scheduler 与 TpModelWorker；rank 0 接收请求。RadixCache、PrefillAdder 与本 rank 的 TpModelWorker 共同组织执行。',
                'The teaching topology fixes TP=2: each of the two TP ranks has its own process, and each process contains both a Scheduler and a TpModelWorker; rank 0 receives requests. RadixCache, PrefillAdder, and that rank’s TpModelWorker organize execution.',
              ],
              vl: [
                '默认多进程拓扑下，EngineCore 独立进程里汇合 Scheduler、KVCacheManager 与 Executor；它每一步产出 SchedulerOutput。',
                'In the default multiprocess topology, the separate EngineCore process combines Scheduler, KVCacheManager, and Executor. Every step emits a SchedulerOutput.',
              ],
              diff: [
                'SGLang 把缓存与准入和调度放在同一个调度器对象里；vLLM 把调度、KV 管理与执行统一收进 EngineCore。',
                'SGLang keeps cache, admission, and scheduling in one scheduler object; vLLM gathers scheduling, KV management, and execution into EngineCore.',
              ],
            },
            {
              title: ['执行 forward', 'Run forward'],
              sg: [
                'GPU forward 与 CPU 侧结果处理可以错开：overlap scheduler 用 result queue 管理上一批。',
                'GPU forward and CPU-side result processing can overlap: the overlap scheduler keeps the previous batch in a result queue.',
              ],
              vl: [
                'EngineCore 忙循环推进；可选 async scheduling，PP 时再由 batch queue 让 micro-batch 流水起来。',
                'The EngineCore busy loop advances. Optional async scheduling, and a batch queue under PP, keep micro-batches in flight.',
              ],
              diff: [
                '都在拆「调度」和「执行」的时序；SGLang 用 overlap scheduler，vLLM 用 EngineCore 循环加可选流水。',
                'Both separate scheduling from execution in time. SGLang uses an overlap scheduler; vLLM uses the EngineCore loop plus optional pipelining.',
              ],
            },
            {
              title: ['回传 token', 'Return tokens'],
              sg: [
                'Scheduler 把 token ids 交给 DetokenizerManager，文本片段再回 TokenizerManager 的流式响应。',
                'The scheduler hands token ids to DetokenizerManager; text fragments return to TokenizerManager for the streaming response.',
              ],
              vl: [
                'EngineCore 返回 EngineCoreOutputs；AsyncLLM / 前端合并、detokenize，并写出 SSE。',
                'EngineCore returns EngineCoreOutputs. AsyncLLM or the frontend merges, detokenizes, and writes the SSE stream.',
              ],
              diff: [
                'SGLang 的 detokenize 是独立进程边界；vLLM 的输出处理留在前端一侧。',
                'SGLang makes detokenization a separate process boundary; vLLM keeps output processing on the frontend side.',
              ],
            },
          ],
        },
        wire: {
          nav: ['通信面', 'Wire profile'],
          question: ['套接字、序列化和执行面，实际怎样接力？', 'How do sockets, serialization, and the execution plane actually hand off?'],
          steps: [
            {
              title: ['入口 socket', 'Entry sockets'],
              sg: [
                'HTTP 与 TokenizerManager 同在主进程；tokenize 之后，内部请求走 ZMQ PUSH → PULL 到 Scheduler。单节点端点是 ipc://，对象由 recv_pyobj 反序列化。',
                'HTTP and TokenizerManager share the main process. After tokenization, internal requests move from a ZMQ PUSH socket to the Scheduler PULL socket. The single-node endpoint is ipc://, and objects arrive through recv_pyobj.',
              ],
              vl: [
                '在独立 EngineCore 进程的拓扑下，前端绑定 ROUTER 输入、PULL 输出；EngineCore 连接 DEALER 输入与 PUSH 输出。ADD / ABORT / UTILITY / WAKEUP 先以类型帧到达，再跟 msgpack 载荷。',
                'With a separate EngineCore process, the frontend binds a ROUTER input and PULL output; EngineCore connects a DEALER input and PUSH output. ADD, ABORT, UTILITY, and WAKEUP arrive as a type frame followed by msgpack payloads.',
              ],
              diff: [
                'SGLang 普通请求与输出走 PUSH/PULL 请求链；vLLM 用 ROUTER/DEALER 请求加独立 PUSH/PULL 回程，请求与回包不走同一个 socket。',
                'SGLang uses a PUSH/PULL request chain for ordinary requests and outputs; vLLM pairs a ROUTER/DEALER request path with a separate PUSH/PULL return path, so requests and replies do not share one socket.',
              ],
            },
            {
              title: ['回程与负载', 'Return path and payload'],
              sg: [
                'Scheduler 把 token ids 走 detokenizer_ipc_name 推给 DetokenizerManager；DetokenizerManager 再把文本通过同一个 tokenizer PULL socket 推回 TokenizerManager。',
                'The Scheduler pushes token ids to DetokenizerManager over detokenizer_ipc_name. DetokenizerManager pushes text back into the same TokenizerManager PULL socket.',
              ],
              vl: [
                'EngineCore 用 PUSH 把 EngineCoreOutputs 送回前端 PULL；输出在 AsyncLLM / 前端合并并 detokenize。消息是 msgpack 多帧，输出 buffer 可复用。',
                'EngineCore PUSHes EngineCoreOutputs to the frontend PULL socket; AsyncLLM or the frontend merges and detokenizes. Messages are multi-frame msgpack, and output buffers can be reused.',
              ],
              diff: [
                'SGLang 的回程经过两段 ZMQ 和一个独立 detokenizer 进程；vLLM 只有 EngineCore→前端一段。',
                'SGLang’s return path crosses two ZMQ hops and a separate detokenizer process; vLLM uses one EngineCore-to-frontend hop.',
              ],
            },
            {
              title: ['调度 ↔ 执行', 'Scheduler ↔ execution'],
              sg: [
                'Scheduler 与 TpModelWorker 位于同一调度器进程；GPU 侧 TP/PP 主干用 NCCL 集合通信，PP 的请求和 proxy 张量另走点对点控制/数据通道。',
                'Scheduler and TpModelWorker live in the scheduler process. GPU-side TP/PP primarily use NCCL collectives, while PP requests and proxy tensors use point-to-point control and data paths.',
              ],
              vl: [
                'EngineCore 与 Worker 之间走 Executor 的 MessageQueue 抽象：本地小消息可走共享内存 ring，也有 local/remote XPUB/SUB socket 分支与 overflow；ZMQ 还承担通知。不能概括为“core→worker 不用 ZMQ”。',
                'EngineCore and Workers use the Executor’s MessageQueue abstraction: local small messages can use a shared-memory ring, with local/remote XPUB/SUB socket branches and overflow; ZMQ also carries notifications. “Core→worker never uses ZMQ” is too broad.',
              ],
              diff: [
                'SGLang 把 TpModelWorker 放在各 rank 的调度进程内；vLLM 的 core→worker 执行面由 Executor 的共享内存 / socket MessageQueue 或 RPC 承载。',
                'SGLang keeps each TpModelWorker in its rank’s scheduler process; vLLM carries the core-to-worker plane through Executor shared-memory/socket MessageQueues or RPC.',
              ],
            },
            {
              title: ['控制面与扩展', 'Control plane and scale-out'],
              sg: [
                'Engine 到 Scheduler 另有 DEALER RPC socket；开启 DP attention 时，原本 ipc:// 的端点会切换到 TCP（DP attention 本身可以跨节点）。',
                'A separate DEALER RPC socket connects Engine to Scheduler. With DP attention, endpoints that were ipc:// switch to TCP; DP attention itself can span nodes.',
              ],
              vl: [
                '前端 ROUTER 输入 / PULL 回程与 EngineCore 的 DEALER/PUSH 也可服务多进程或多节点前端；DP coordinator 有自己的 ZMQ 通道，数据面再用 NCCL 或 RDMA。',
                'The frontend ROUTER input / PULL return pair and the EngineCore DEALER/PUSH pair can also serve multi-process or multi-node frontends; the DP coordinator has its own ZMQ channel, while the data plane uses NCCL or RDMA.',
              ],
              diff: [
                '两边都把控制面做成显式进程边界，但 SGLang 的控制面以 Scheduler / RPC 为中心，vLLM 以 core client 与 DP coordinator 为中心。',
                'Both make the control plane an explicit process boundary, but SGLang centers it on Scheduler/RPC while vLLM centers it on the engine client and DP coordinator.',
              ],
            },
          ],
        },
        prefix: {
          nav: ['前缀缓存', 'Prefix cache'],
          question: ['复用前缀时，地址是什么形状？', 'When a prefix is reused, what shape is its address?'],
          steps: [
            {
              title: ['把前缀变成地址', 'Turn a prefix into an address'],
              sg: [
                'Token 序列进入 RadixCache。每条边承载一段 token 路径；匹配长度按 page_size 向下对齐，只在完整单位边界分裂。',
                'A token sequence enters RadixCache. Each edge carries a token path; match length is rounded down to page_size and splits only on a complete-unit boundary.',
              ],
              vl: [
                'Prompt 切成固定大小 block，逐块计算 chain hash；hash 表把完整块映射到物理 block。',
                'The prompt is cut into fixed-size blocks and a chained hash is computed block by block. A hash table maps full blocks to physical blocks.',
              ],
              diff: [
                '一个以 token 路径为索引，一个以固定块哈希为索引。',
                'One indexes a token path; the other indexes fixed-block hashes.',
              ],
            },
            {
              title: ['查找可复用前缀', 'Look up reusable prefix'],
              sg: [
                '沿树走最长前缀。本例 page_size=4、已有 12 token 缓存、新输入 12 token，前 10 相同；按完整单位对齐只命中前 8 个，边界 8/12。旧缓存第 3 页（9–12）因 11–12 不同而整页不命中，保留在旧分支；9–10 相同不单独复用。',
                'Walk the longest prefix. With page_size=4, a 12-token cached prompt, and a 12-token input sharing the first 10 tokens, alignment to complete units matches only the first 8, so the boundary is 8/12. The cached third page (9–12) does not match as a unit because tokens 11–12 differ; it stays in the old branch. The shared tokens 9–10 are not reused on their own.',
              ],
              vl: [
                '按块查 hash：只命中前两个完整块，共 8 / 12 token；旧缓存第 3 块（9–12）因 11–12 不同不命中并保留，新输入尾 4 token 另分配一块。',
                'Look up hashes by block: only the first two full blocks hit, 8 of 12 tokens. The cached third block (9–12) does not match because tokens 11–12 differ and stays retained; the trailing four input tokens allocate a new block.',
              ],
              diff: [
                '本例双方都只复用完整单位，边界同为 8/12；真正的差异是索引形状（树路径 vs 链式块 hash）与地址表示（节点/页 vs block table）。4 只是示例粒度，不是推荐配置。',
                'Both sides reuse only complete units here, with the same 8/12 boundary. The real difference is index shape (tree path vs chained block hash) and address representation (node/page vs block table). Four is an illustrative granularity, not a recommended configuration.',
              ],
            },
            {
              title: ['补算与分配', 'Fill and allocate'],
              sg: [
                '未命中的第 3 页（9–12）作为新分支补算，对应的 KV 页随请求路径绑定；旧缓存的 9–12 页仍保留在旧分支，两条分支从第 2 页分开。',
                'The unmatched third page (9–12) grows as a new branch and its KV pages bind to the request path. The cached 9–12 page stays on the old branch; the two branches split after page 2.',
              ],
              vl: [
                '给未命中的尾部 token 分配新块；block table 只更新逻辑位置到物理块的映射，KV 本体在分页缓冲区。部分块索引与 copy-on-write 属于特定分支，本例不展开。',
                'New blocks are allocated for the unmatched tail; the block table only updates logical-to-physical mappings, while KV lives in paged buffers. Partial-block indexing and copy-on-write are conditional branches and are not shown here.',
              ],
              diff: [
                'RadixCache 的共享单位是页对齐的路径片段；vLLM 的共享单位是完整 block/hash 单位。',
                'RadixCache shares path segments; vLLM shares complete blocks.',
              ],
            },
            {
              title: ['下一拍继续生长', 'Grow on the next step'],
              sg: [
                '请求持有路径上的节点锁；解码出的新 token 继续扩展路径，后续请求可在这条路径上再分叉。',
                'The request holds node locks along the path. Decoded tokens extend it, and later requests can fork from the same path.',
              ],
              vl: [
                'block table 保存逻辑位置到物理块的映射；后续请求复用同一批块时只改映射，追加再分配。',
                'The block table maps logical positions to physical blocks. Later requests reuse blocks by changing that mapping and allocate only when they append.',
              ],
              diff: [
                '前者像按页对齐的可变长前缀树，后者像带引用计数的分页表。',
                'The first behaves like a variable-length prefix trie; the second like a reference-counted page table.',
              ],
            },
          ],
        },
        batch: {
          nav: ['调度心跳', 'Scheduling beat'],
          question: ['每一拍的预算，怎样被填满？', 'How is each step’s budget filled?'],
          steps: [
            {
              title: ['收集本拍工作', 'Collect runnable work'],
              sg: [
                'waiting queue 与 running batch 分开维护；调度器每拍先判断有哪些 prefill 可以准入。',
                'The waiting queue and running batch are maintained separately. Each step, the scheduler first decides which prefills may be admitted.',
              ],
              vl: [
                'waiting 与 running 进入同一个调度循环；EngineCore 在每步统一生成 SchedulerOutput。',
                'Waiting and running requests enter one scheduling loop. EngineCore produces a single SchedulerOutput for each step.',
              ],
              diff: [
                '一个围绕「准入」组织，一个围绕「统一输出」组织。',
                'One organizes around admission; the other around a unified output.',
              ],
            },
            {
              title: ['分配这一拍的预算', 'Assign the step budget'],
              sg: [
                'PrefillAdder 用 token / KV 预算判断新 prefill 能否进入；长 prompt 可以按 chunk 跨拍执行。',
                'PrefillAdder uses token and KV budgets to decide which new prefill can enter. Long prompts can run as chunks across steps.',
              ],
              vl: [
                '在统一 token budget 内同时放入 decode token 与 prefill chunk，形成混合 batch。',
                'A unified token budget admits both decode tokens and prefill chunks into the same mixed batch.',
              ],
              diff: [
                'SGLang 的 prefill 准入是独立而缓存感知的决策；vLLM 把本拍工作当成一个统一装箱问题。',
                'SGLang treats prefill admission as a separate, cache-aware decision; vLLM frames the step as one unified packing problem.',
              ],
            },
            {
              title: ['让下一拍提前跑', 'Overlap the next beat'],
              sg: [
                'overlap scheduler：当前 batch 在 GPU 上 forward 时，CPU 处理上一批结果并准备下一批。',
                'Overlap scheduler: while the current batch runs forward on the GPU, the CPU processes the previous result and prepares the next batch.',
              ],
              vl: [
                'EngineCore 忙循环；async scheduling 让调度与执行错开，PP 的 batch queue 让多个 micro-batch 在流水线上重叠。',
                'The EngineCore loop runs continuously. Async scheduling offsets scheduling from execution, while the PP batch queue overlaps multiple micro-batches.',
              ],
              diff: [
                '交叠的层次不同：SGLang 重点在 CPU 调度/结果处理与 GPU forward 之间，vLLM 在 EngineCore 步进与流水线之间。',
                'They overlap at different layers: SGLang between CPU scheduling/result handling and GPU forward, vLLM between EngineCore steps and the pipeline.',
              ],
            },
            {
              title: ['回收 token', 'Return the tokens'],
              sg: [
                '本批结果进入下一步处理，采样出的 token 沿 detokenizer 返回；下一拍继续填预算。',
                'Batch results flow into the next processing stage. Sampled tokens return through the detokenizer, and the next step refills the budget.',
              ],
              vl: [
                'Executor 返回 ModelRunnerOutput，Scheduler 更新请求状态并产出 EngineCoreOutputs。',
                'The executor returns ModelRunnerOutput. The scheduler updates request state and emits EngineCoreOutputs.',
              ],
              diff: [
                '回程路径决定可观测性放在哪：SGLang 看调度器与 detokenizer，vLLM 看 EngineCore 输出。',
                'The return path determines where observability lives: the scheduler and detokenizer in SGLang, EngineCore outputs in vLLM.',
              ],
            },
          ],
        },
        disagg: {
          nav: ['分离扩展', 'Disaggregation'],
          question: ['拆开 prefill 与 decode 后，控制面在哪里？', 'Once prefill and decode split, where is the control plane?'],
          steps: [
            {
              title: ['选择实例', 'Choose an instance'],
              sg: [
                'Rust sgl-router 可按前缀命中做 cache-aware 路由；PD / EPD 把 prefill、decode、encode 分到不同实例池。',
                'The Rust sgl-router can route with cache awareness. PD/EPD place prefill, decode, and encode in separate instance pools.',
              ],
              vl: [
                '前端负责选实例；P/D 分离通过 KVConnectorV1 把 prefill 实例与 decode 实例连起来。DP coordinator 只做 DP 负载统计与 wave 协调，不是通用 P/D router。',
                'The frontend picks an instance. P/D separation connects prefill and decode instances through KVConnectorV1. The DP coordinator coordinates DP load and waves; it is not a general P/D router.',
              ],
              diff: [
                'SGLang 更常把「前缀命中」写进路由判断；vLLM 把分离做成 connector 扩展点。',
                'SGLang more often folds prefix hits into routing; vLLM exposes separation as a connector extension point.',
              ],
            },
            {
              title: ['搬运 KV', 'Move the KV'],
              sg: [
                'Mooncake / NIXL / Mori 等后端搬运 KV；两侧 scheduler 维护角色、握手与释放节奏。',
                'Backends such as Mooncake, NIXL, and Mori move KV. Schedulers on both sides maintain roles, handshakes, and release timing.',
              ],
              vl: [
                'Scheduler 在 SchedulerOutput 里带上 connector metadata；worker 侧把远端 KV 加载到本地物理 KV 缓存，block table 只保存映射。',
                'The scheduler attaches connector metadata to SchedulerOutput; workers load remote KV into local physical KV buffers, with the block table holding only the mapping.',
              ],
              diff: [
                '数据一样是 KV，边界不同：SGLang 侧重路由器与 PD 角色，vLLM 侧重 connector 生命周期。',
                'The data is the same KV; the boundary differs. SGLang emphasizes router and PD roles, vLLM emphasizes connector lifecycles.',
              ],
            },
            {
              title: ['缓存继续下沉', 'Sink the cache'],
              sg: [
                'HiCache 把前缀缓存从 GPU 扩到 CPU / 文件级 L3；MLA + DCP 也可保留前缀。',
                'HiCache extends prefix caching from GPU memory to CPU and file-level L3. MLA with DCP can keep prefixes too.',
              ],
              vl: [
                'KVConnector 与 offload 路径把块搬到 CPU 或远端；前缀缓存仍以块为协议单位。',
                'KVConnector and offload paths move blocks to CPU memory or remote tiers. Prefix caching still uses blocks as its protocol unit.',
              ],
              diff: [
                '一个从 radix 路径向外分层，一个从 block 所有权向外扩展。',
                'One tiers outward from radix paths; the other extends outward from block ownership.',
              ],
            },
            {
              title: ['放大整个集群', 'Scale the cluster'],
              sg: [
                'DP attention、EP / EPLB 与 PD 比例是不同机制/部署选项，不是可任意组合的支持矩阵；路由与缓存命中强耦合。',
                'DP attention, EP/EPLB, and the PD ratio are distinct mechanisms/deployment options, not an arbitrary support matrix. Routing and cache hits stay tightly coupled.',
              ],
              vl: [
                'TP / PP / DP / EP 与弹性 EP 是不同组合选项；EngineCore 边界让前端与执行资源解耦，但不代表任意组合都被支持。',
                'TP/PP/DP/EP and elastic EP are distinct options; the EngineCore boundary decouples frontends from execution resources, but not every combination is supported.',
              ],
              diff: [
                '都在往更大集群扩展，但控制面落点不同。',
                'Both scale out, but their control planes land in different places.',
              ],
            },
          ],
        },
        crosscut: {
          nav: ['生命周期与执行边界', 'Lifecycle & execution boundaries'],
          question: ['同一项工程问题，两个引擎怎样走出不同的线？', 'For the same engineering problem, where do the two engines take different lines?'],
          steps: [
            {
              title: ['请求结束后，KV 去哪？', 'After a request finishes, where does KV go?'],
              sg: [
                '正常收尾时，release_kv_cache(req, checkpoint=True) 先调用 tree_cache.checkpoint()，随后撤销请求的使用关系与锁保护，再回收不再受保护的 KV。仍有保护的前缀留在树上，缓存保留不等于真正驱逐。',
                'On normal finish, release_kv_cache(req, checkpoint=True) first calls tree_cache.checkpoint(), then drops the request’s usage relationship and locks, and finally reclaims KV that is no longer protected. Prefixes still protected stay in the tree; retention in cache is not true eviction.',
              ],
              vl: [
                '正常收尾时，vLLM 释放请求持有的 block：ref_cnt 递减到 0 的块回到 free queue，但已算完的完整块仍保留 hash，可被后续请求命中；驱逐时才清 hash。',
                'On normal finish, vLLM releases the request blocks: blocks whose ref_cnt reaches zero return to the free queue, while fully computed blocks keep their hash for later prefix hits. The hash is cleared only on eviction.',
              ],
              diff: [
                '两者都保留可复用前缀，也都不把请求结束当作真正驱逐；SGLang 先 checkpoint 再撤销保护并回收非保护页，vLLM 先降引用、保留完整块 hash，直到驱逐才清 hash。',
                'Both retain reusable prefixes and neither treats request finish as true eviction: SGLang checkpoints before dropping protection and reclaiming unprotected pages; vLLM drops references and keeps full-block hashes until eviction clears them.',
              ],
            },
            {
              title: ['显存告急，先动谁？', 'When memory runs out, who moves first?'],
              sg: [
                'decode 内存检查失败时，retract_decode 从 running decode 中撤下部分请求，释放 KV、重置状态并放回 waiting；至少保留一个请求，连最后一个都放不下时中止它。',
                'When the decode memory check fails, retract_decode removes some running decode requests, releases KV, resets them, and puts them back in waiting. At least one request is kept; if even the last one cannot fit, it is aborted.',
              ],
              vl: [
                'Scheduler 在 running 中执行 preemption：释放请求 blocks，把 num_computed_tokens 归零并标记 PREEMPTED，前插到 waiting；之后重新 prefill / compute。',
                'The Scheduler preempts a running request: frees its blocks, resets num_computed_tokens to zero, marks it PREEMPTED, and prepends it to waiting. It is recomputed later.',
              ],
              diff: [
                '两者都选择回退而不是让调度器崩溃；SGLang 的入口是 retraction 与 new_token_ratio，vLLM 的入口是 block 释放与 PREEMPTED 重算。',
                'Both fall back instead of crashing the scheduler; SGLang enters through retraction and new_token_ratio, while vLLM enters through block release and PREEMPTED recomputation.',
              ],
            },
            {
              title: ['请求取消，谁清场？', 'When a request is cancelled, who cleans up?'],
              sg: [
                'Abort 先到达调度器输入：waiting 中的请求直接弹出；running 中的请求从 batch 过滤、释放 KV，并向 TokenizerManager 回 AbortReq，让流式状态收尾。',
                'Abort first reaches scheduler input: waiting requests are popped; running requests are filtered out of the batch, KV is released, and an AbortReq is sent back to TokenizerManager to close streaming state.',
              ],
              vl: [
                '前端经 ROUTER→DEALER 向 EngineCore 发送 ABORT 类型帧；Scheduler.finish_requests(FINISHED_ABORTED) 从 waiting / running 移除请求，释放 blocks 与 connector 状态；在途写或 connector 释放受 delay_free 条件约束，不保证立即完成。',
                'The frontend sends an ABORT type frame to EngineCore over ROUTER→DEALER. Scheduler.finish_requests(FINISHED_ABORTED) removes the request from waiting/running and frees blocks and connector state; in-flight writes or connector frees may be delayed by delay_free conditions.',
              ],
              diff: [
                'SGLang 的 AbortReq 在 scheduler 与 tokenizer 状态之间往返；vLLM 的 ABORT 是 frontend→EngineCore 的类型化控制消息，清场集中在 Scheduler。',
                'SGLang’s AbortReq travels between scheduler and tokenizer state; vLLM’s ABORT is a typed frontend-to-EngineCore control message, with cleanup centered in the Scheduler.',
              ],
            },
            {
              title: ['每步 launch 开销，怎样被捕获？', 'How is per-step launch overhead captured?'],
              sg: [
                'decode 用固定形状的 CUDA graph 捕获/重放；prefill 可走 piecewise CUDA graph。可选 breakable CUDA graph（BCG）在 attention/mamba 等显式 eager 边界断开，metadata 在捕获段外重算；它不免除 shape/地址约束，也不是任意变长计算都能塞进已捕获图。',
                'Decode uses captured CUDA graphs for fixed shapes; prefill can use piecewise CUDA graphs. Optional breakable CUDA graph (BCG) breaks at explicit eager boundaries such as attention/mamba and recomputes metadata outside the captured segment; it does not remove shape/address constraints or let arbitrary variable-length work enter a captured graph.',
              ],
              vl: [
                '由图模式（CompilationConfig 与 CUDAGraphMode）决定是否捕获及捕获粒度：NONE、PIECEWISE、FULL 是可选模式，FULL 并不强制依赖 torch.compile，PIECEWISE 有自己的编译条件。dispatcher 按 batch descriptor、允许的模式与已捕获 keys 选择，不能一概画成必选默认。',
                'Graph mode (CompilationConfig and CUDAGraphMode) decides whether and at what granularity to capture: NONE, PIECEWISE, and FULL are selectable modes. FULL does not require torch.compile, and PIECEWISE has its own compilation conditions. The dispatcher chooses by batch descriptor, allowed modes, and captured keys, so this is not a mandatory default path.',
              ],
              diff: [
                '两者都在消除逐层 launch 开销；SGLang 的差异是显式 eager 边界的可断开重放与已有 shape_key，vLLM 的差异是 graph mode、batch descriptor、允许模式与已捕获 keys 的联合选择。都依赖形状桶/填充，具体支持随模型与后端变化。',
                'Both remove per-layer launch overhead. SGLang relies on breakable replay at explicit eager boundaries and an existing shape_key; vLLM jointly selects graph mode, batch descriptor, allowed modes, and captured keys. Both depend on shape buckets or padding, and support varies by model and backend.',
              ],
            },
          ],
        },
      },
      nodes: {
        process: {
          sg_http: [
            'OpenAI 兼容 HTTP 入口。主进程里的 FastAPI 接住请求后交给 TokenizerManager；它不直接调度 GPU。',
            'The OpenAI-compatible HTTP entry point. FastAPI in the main process hands the request to TokenizerManager; it does not schedule the GPU directly.',
          ],
          sg_tokenizer: [
            '负责 tokenize、请求校验、流式输出与请求状态。与 Scheduler 之间通过 ZMQ 传递内部请求。',
            'Owns tokenization, request validation, streaming output, and request state. It exchanges internal requests with the scheduler over ZMQ.',
          ],
          sg_scheduler: [
            '每个模型 rank 进程内有一个 Scheduler：RadixCache、PrefillAdder、ScheduleBatch 与本地模型执行入口。可选 overlap scheduler 将 CPU 侧处理与 GPU forward 错开。',
            'Each model-rank process has a Scheduler: RadixCache, PrefillAdder, ScheduleBatch, and the local model execution entry. The optional overlap scheduler offsets CPU-side work from GPU forward.',
          ],
          sg_workers: [
            '每个 rank 进程内的本地执行对象 TpModelWorker，驱动本 rank 的 TP/PP 组并执行 forward；attention backend、KV 池与采样由具体模型和后端决定。',
            'The local per-rank execution object, TpModelWorker, drives this rank’s TP/PP groups and executes forward. Attention backend, KV pool, and sampling depend on model and backend.',
          ],
          sg_detok: [
            '独立子进程做增量 detokenize，把 token id 转成文本；结果再回 TokenizerManager。',
            'A separate subprocess performs incremental detokenization, turning token ids into text before returning it to TokenizerManager.',
          ],
          vl_frontend: [
            'Python OpenAI 服务或 Rust frontend。Rust 路径把请求渲染与 HTTP 服务从 Python 主路径剥离；启用与否取决于部署。',
            'A Python OpenAI server or the Rust frontend. The Rust path peels request rendering and HTTP serving away from the Python main path; deployment chooses whether to use it.',
          ],
          vl_async: [
            'AsyncLLM 负责输入处理、输出处理、流式回调和引擎客户端。它不直接持有 GPU 调度循环。',
            'AsyncLLM handles input processing, output processing, streaming callbacks, and the engine client. It does not own the GPU scheduling loop.',
          ],
          vl_engine: [
            '默认多进程拓扑下，EngineCore 是独立进程与忙循环：Scheduler、KVCacheManager、Executor、KV connector 的调度侧都在这里汇合。',
            'In the default multiprocess topology, EngineCore is a separate process and busy loop where Scheduler, KVCacheManager, Executor, and the scheduling side of KV connectors meet.',
          ],
          vl_workers: [
            '每个 Worker 进程对应一个模型 rank，执行本 rank 的 ModelRunner；教学拓扑为 ×2。多进程 executor 中它与 EngineCore 分属不同进程，v0.33 的 MRV2、CUDA Graph、attention backend 与量化 kernel 在这里落地。',
            'Each Worker process maps to one model rank and runs that rank’s ModelRunner; the teaching topology uses ×2. Under the multiprocess executor it lives in a process separate from EngineCore. In v0.33, MRV2, CUDA graphs, attention backends, and quantization kernels land here.',
          ],
          vl_stream: [
            'EngineCoreOutputs 回传后，前端合并、detokenize 并写成 SSE；可观测性切面就在这条回程上。',
            'After EngineCoreOutputs returns, the frontend merges, detokenizes, and writes SSE. The observability cut lives on this return path.',
          ],
        },
        wire: {
          sg_tok: [
            'TokenizerManager 的 PULL socket 接收从 DetokenizerManager 或 Scheduler 推回的消息；PUSH socket 把内部请求送到 scheduler_input_ipc_name。端点默认是 ipc:// 临时文件，开启 DP attention 时切换为 tcp://。',
            'TokenizerManager’s PULL socket receives messages pushed back by DetokenizerManager or the Scheduler; its PUSH socket sends internal requests to scheduler_input_ipc_name. Endpoints default to ipc:// temporary files and switch to tcp:// when DP attention is enabled.',
          ],
          sg_sched: [
            'rank 0 的 Scheduler 从 PULL socket 接收请求，向 detokenizer_ipc_name 推 token ids，也可直接向 tokenizer_ipc_name 推回包；另有 DEALER socket 接 Engine RPC。',
            'The rank-0 Scheduler receives requests on a PULL socket, pushes token ids to detokenizer_ipc_name, and can also push replies directly to tokenizer_ipc_name. A separate DEALER socket serves Engine RPC.',
          ],
          sg_detok: [
            '独立进程从 detokenizer_ipc_name 拉取 BatchTokenIDOutput，做增量 detokenize，再把文本 PUSH 到 tokenizer_ipc_name。',
            'A separate process pulls BatchTokenIDOutput from detokenizer_ipc_name, performs incremental detokenization, and PUSHes text to tokenizer_ipc_name.',
          ],
          sg_rpc: [
            'Engine 到 Scheduler 的管理 RPC 走 DEALER socket（rpc_ipc_name）。这条通道与请求数据面分开。',
            'Management RPC from Engine to Scheduler uses a DEALER socket (rpc_ipc_name). This channel is separate from the request data path.',
          ],
          sg_workers: [
            'TpModelWorker 与 Scheduler 同进程，不通过 ZMQ 执行模型；GPU 侧 TP/PP 用 NCCL 集合通信，PP 的 request / proxy 张量走点对点控制与数据通道。',
            'TpModelWorker shares the Scheduler process and does not execute the model over ZMQ. GPU-side TP/PP use NCCL collectives, while PP requests and proxy tensors use point-to-point control and data paths.',
          ],
          sg_protocol: [
            'SGLang 普通请求/输出路径走 ipc_channels 的 PUSH/PULL；io_struct 的 sock_send/sock_recv 还有配置分支，不能概括为全链路始终 pickle。默认 IPC 走 ipc://，开启 DP attention 时改用 TCP。',
            'SGLang’s ordinary request/output path uses PUSH/PULL through ipc_channels; io_struct’s sock_send/sock_recv also has configuration branches, so the whole chain is not always pickle. Default IPC uses ipc://; DP attention switches to TCP.',
          ],
          vl_frontend: [
            '独立 EngineCore 进程的拓扑下，前端绑定 ROUTER 输入与 PULL 输出，输入返回 EngineCoreOutputs；输出合并和 detokenize 留在 AsyncLLM / 前端一侧。Rust frontend 是另一条可选前端路径。',
            'In a separate-EngineCore topology, the frontend binds a ROUTER input and PULL output. EngineCoreOutputs return to the frontend, where merging and detokenization happen. The Rust frontend is an alternative frontend path.',
          ],
          vl_core: [
            'EngineCore 用 DEALER 接前端 ROUTER，用 PUSH 发回输出；请求先有类型帧（ADD / ABORT / UTILITY / WAKEUP），再跟 msgpack 载荷。',
            'EngineCore uses a DEALER socket toward the frontend ROUTER and a PUSH socket for outputs. Requests start with a type frame (ADD, ABORT, UTILITY, WAKEUP) followed by msgpack payloads.',
          ],
          vl_workers: [
            'SchedulerOutput 通过 Executor 的 MessageQueue 抽象广播给 Worker：本地共享内存 ring，另有 local/remote XPUB/SUB 与 overflow；Worker 用各自的 response MQ 返回 ModelRunnerOutput。Ray executor 则改用 Ray RPC。',
            'SchedulerOutput is broadcast to Workers through the Executor MessageQueue abstraction: a local shared-memory ring plus local/remote XPUB/SUB and overflow; each Worker returns ModelRunnerOutput on its response MQ. A Ray executor switches this to Ray RPC.',
          ],
          vl_dp: [
            'DP coordinator 有自己的 ZMQ 输入/输出通道来协调多个 EngineCore；数据并行和模型并行内部再使用进程组 / NCCL。',
            'The DP coordinator uses its own ZMQ input and output channels to coordinate EngineCore instances; data and model parallelism then use process groups and NCCL internally.',
          ],
          vl_protocol: [
            'vLLM 前端与 EngineCore 之间使用多帧 msgpack，并复用输出 buffer；EngineCore 与 Worker 之间不经过这条前端请求通道，而由 Executor 的 MessageQueue 抽象承载（共享内存 ring 与 socket 分支）。',
            'vLLM frontends and EngineCore exchange multi-frame msgpack and can reuse output buffers. The EngineCore-to-Worker plane bypasses this frontend request channel and uses the Executor MessageQueue abstraction (shared-memory ring and socket branches).',
          ],
        },
        prefix: {
          sg_radix: [
            'RadixCache 用树形路径保存前缀；每条边是一段连续 token，命中长度对齐到 page_size 后才分裂。',
            'RadixCache stores prefixes as a tree of paths. Each edge is a contiguous token range and splits only after match length is aligned to page_size.',
          ],
          sg_path: [
            '命中路径会被锁住（引用计数），避免被回收；节点是缓存索引，物理 KV 页另由内存池管理。',
            'The matched path is locked with reference counts to avoid eviction. Request KV pages correspond to nodes on the path.',
          ],
          sg_tail: [
            '未命中的尾部作为新分支插入；同一前缀的请求共享已经命中的节点。',
            'The unmatched tail is inserted as a new branch. Requests with the same prefix share the already-matched nodes.',
          ],
          vl_hash: [
            '逐块计算链式 hash；完整块可复用，表项指向物理 block id。部分块索引是条件分支，不当作所有后端规则。',
            'Chained hashes are computed block by block. Full blocks can be reused, and entries point to physical block ids. Partial-block indexing is conditional and not a rule for every backend.',
          ],
          vl_pool: [
            'BlockPool 管理物理 KV block 的空闲、引用与回收；block table 只是逻辑位置到物理块的映射，不是 KV 本体。',
            'BlockPool manages free, referenced, and reclaimable physical KV blocks. A block table only maps logical positions to blocks; it is not the KV payload.',
          ],
          vl_table: [
            '请求的 block table 指向命中与新分配的物理块；引用计数归零后块才回到池子。',
            'The request block table points at hit and newly allocated physical blocks. A block returns to the pool only when its reference count reaches zero.',
          ],
        },
        batch: {
          sg_queue: [
            '等待队列承载尚未准入的请求；running batch 承载已经进入 decode 的请求。',
            'The waiting queue holds requests not yet admitted; the running batch holds requests already in decode.',
          ],
          sg_adder: [
            'PrefillAdder 根据 token、KV、batch size 与 chunk 约束决定哪些 prefill 进入本拍；这是 SGLang 调度最容易观察到的结构。',
            'PrefillAdder decides which prefills enter the step under token, KV, batch-size, and chunk constraints. It is the most visible SGLang scheduling structure.',
          ],
          sg_batch: [
            'ScheduleBatch 是本拍要执行的批次；可能包含 decode、chunked prefill 或二者合并后的工作。',
            'ScheduleBatch is the batch executed this step. It can contain decode work, chunked prefill, or a merged mix.',
          ],
          sg_overlap: [
            'overlap scheduler 让 CPU 处理上一拍结果与 GPU 执行当前拍错开；它不等同于 PP。',
            'The overlap scheduler offsets CPU handling of the previous result from GPU execution of the current step. It is not the same thing as PP.',
          ],
          sg_gpu: [
            '模型 forward 与采样在 GPU 上执行；结果进入 result queue，CPU 在下一拍处理。',
            'Model forward and sampling execute on GPUs. Results enter a result queue for the CPU to process on the next step.',
          ],
          vl_queue: [
            'waiting 与 running 在同一个调度器里迭代；没有 V0 时代独立的 prefill/decode 引擎分叉。',
            'Waiting and running requests iterate through the same scheduler; there is no separate V0-style prefill/decode engine split.',
          ],
          vl_budget: [
            '统一 token budget 决定本步可以放多少 token；prefill chunk 与 decode token 共同占用。',
            'A unified token budget decides how many tokens fit into the step; prefill chunks and decode tokens draw from it together.',
          ],
          vl_schedout: [
            'SchedulerOutput 是 EngineCore 给 Worker 的执行契约，包含每请求调度 token 数、块变化和 connector metadata。',
            'SchedulerOutput is the execution contract from EngineCore to Workers. It carries per-request scheduled tokens, block changes, and connector metadata.',
          ],
          vl_async: [
            'EngineCore 与执行资源之间的边界；async scheduling 与 PP batch queue 让多拍可以在途。',
            'The boundary between EngineCore and execution resources. Async scheduling and the PP batch queue keep multiple steps in flight.',
          ],
          vl_workers: [
            'Worker 执行 SchedulerOutput；返回的 ModelRunnerOutput 在 EngineCore 内更新状态并生成回包。',
            'Workers execute SchedulerOutput. The returned ModelRunnerOutput updates state in EngineCore and produces the next outputs.',
          ],
        },
        disagg: {
          sg_router: [
            'sgl-router 是 Rust 路由器，可按前缀命中与负载选择实例。PD 路由是它区别于普通负载均衡器的关键。',
            'sgl-router is the Rust router. It can choose instances using prefix hits and load, and PD routing is what distinguishes it from a generic balancer.',
          ],
          sg_kv: [
            'PD/EPD 之间的 KV 传输由 Mooncake、NIXL、Mori 等后端完成；调度器负责角色与生命周期。',
            'KV transfer between PD/EPD roles is carried by backends such as Mooncake, NIXL, and Mori; schedulers own roles and lifecycle.',
          ],
          sg_hicache: [
            'HiCache 将前缀缓存分层到 CPU 和存储；GPU 仍持有活跃 KV。',
            'HiCache tiers prefix caches into CPU memory and storage; GPUs still hold active KV.',
          ],
          sg_scale: [
            'DP attention、EP/EPLB 与 PD 比例是独立旋钮；大规模下路由与缓存策略要一起调。',
            'DP attention, EP/EPLB, and the PD ratio are independent controls. At scale, routing and cache policy must be tuned together.',
          ],
          vl_router: [
            'Python/Rust 前端或 DP coordinator 负责 DP；P/D 使用 KVConnectorV1 连接实例。',
            'A Python/Rust frontend or DP coordinator assigns requests; P/D uses KVConnectorV1 to connect instances.',
          ],
          vl_connector: [
            'KVConnectorV1 由 Scheduler 和 Worker 两侧共同实现：调度侧生成 metadata，worker 侧执行加载与保存。',
            'KVConnectorV1 is implemented on both Scheduler and Worker sides: the scheduler emits metadata, workers perform loads and saves.',
          ],
          vl_offload: [
            'connector 与 offload 以 block 为所有权单位迁移 KV；本地 block table 是解码侧的最终地址。',
            'Connectors and offload paths move KV with block ownership; the local block table is the final address on the decode side.',
          ],
          vl_scale: [
            'TP/PP/DP/EP、弹性 EP 与 KV connector 可以独立组合；EngineCore 是执行资源的控制面。',
            'TP/PP/DP/EP, elastic EP, and KV connectors combine independently. EngineCore is the control plane for execution resources.',
          ],
        },
        crosscut: {
          sg_finish: [
            '请求完成时仍持有 RadixCache 的使用关系与节点锁。release 路径先做 checkpoint，再撤销请求关系与保护，最后回收不再受保护的 KV；不是把请求结束等同于缓存驱逐。',
            'A finished request still holds a RadixCache usage relationship and node locks. The release path checkpoints first, then drops the request relationship and protection, and finally reclaims KV that is no longer protected. Request finish is not cache eviction.',
          ],
          sg_radix: [
            'release_kv_cache(req, checkpoint=True) 先调用 tree_cache.checkpoint()，再退出请求的使用关系、解锁保护，随后回收非保护 KV。受保护的前缀仍留在 radix 树上。',
            'release_kv_cache(req, checkpoint=True) first calls tree_cache.checkpoint(), then drops the request’s usage relationship and locks, and then reclaims unprotected KV. Protected prefixes remain in the radix tree.',
          ],
          sg_pool: [
            'KV 池只在保护关系退出后回收非保护页；仍被 radix 路径或后续请求保护的前缀不会被当作“请求结束”立刻清掉。真正驱逐发生在之后的无锁叶子回收。',
            'The KV pool reclaims unprotected pages only after protection is dropped; prefixes still protected by a radix path or later requests are not cleared merely because a request finished. True eviction happens later when an unlocked leaf is reclaimed.',
          ],
          sg_reuse: [
            '下一个请求沿 radix 树做最长前缀匹配；命中的节点锁引用上升，未命中的尾部继续生长。',
            'The next request runs longest-prefix matching through the radix tree; matched nodes gain locks, and the unmatched tail grows from there.',
          ],
          sg_evict: [
            '没有锁、没有子节点的叶子成为 evictable；回收时释放其 KV 页并从树上删除。',
            'A leaf with no locks and no children becomes evictable; reclaiming it frees its KV pages and removes it from the tree.',
          ],
          vl_finish: [
            '请求完成时，Scheduler 走 finish/free 路径；请求持有的 block 引用逐个减少。',
            'When a request finishes, the Scheduler enters its finish/free path and releases the block references held by the request.',
          ],
          vl_table: [
            '请求的 block table 指向物理块。引用计数归零的块会被归还，而不是立刻丢掉整张表。',
            'The request block table points at physical blocks. Blocks whose reference count reaches zero are returned; the table is not immediately discarded.',
          ],
          vl_pool: [
            'BlockPool 把 ref_cnt 为 0 的块放回 free queue；如果块仍有完整 hash，它仍可被前缀命中。',
            'BlockPool returns ref_cnt=0 blocks to the free queue. A full block with a retained hash can still be hit by prefix lookup.',
          ],
          vl_reuse: [
            '下一个请求按链式 block hash 查找；完整块命中时直接引用，尾部未命中则分配新块。部分块索引与 COW 属于特定后端分支，本例不展开。',
            'The next request looks up chained block hashes. Full-block hits are referenced directly; unmatched tails allocate new blocks. Partial-block indexing and COW are backend-specific branches not shown here.',
          ],
          vl_hash: [
            '完整块的 hash 与物理块绑定，直到该块被驱逐；free queue 中的块仍可能被命中。',
            'A full block hash stays bound to its physical block until eviction; a block in the free queue can still be hit.',
          ],
          vl_evict: [
            '真正驱逐时才移除 hash 映射并回收块；这与“请求结束”不是同一步。',
            'Eviction removes the hash mapping and reclaims the block; that is a different step from request finish.',
          ],
          sg_pressure: [
            'decode 前检查 KV 池余量；不足时不能继续按原样扩展 running batch。',
            'A pre-decode KV capacity check runs before extending the running batch; if it fails, the batch cannot grow as-is.',
          ],
          sg_retract: [
            'retract_decode 选择要撤下的 decode 请求，释放 KV、重置状态并放回 waiting，同时重新估计 new_token_ratio。它也可能等待即将释放的在途结果；不是所有 OOM 都可恢复。',
            'retract_decode withdraws selected decode requests, frees KV, resets them, and requeues them in waiting while re-estimating new_token_ratio. It may also wait for in-flight results about to free KV; not every OOM is recoverable.',
          ],
          sg_release: [
            'retraction 释放 KV 时不把请求路径插回 RadixCache，以便立刻拿回页；请求状态重置后再由 PrefillAdder 重新准入。',
            'Retraction releases KV without inserting the request path back into RadixCache so pages are immediately available. The request resets and later re-enters through PrefillAdder.',
          ],
          sg_ratio: [
            'retract 后 SGLang 会重新估计 new_token_ratio 并 clamp 为 ≤1；它不保证比旧值小。不是所有 OOM 都能靠 retract 恢复；最后一个请求也放不下时会中止它。',
            'After retraction, SGLang re-estimates new_token_ratio and clamps it to ≤1; it is not guaranteed to be lower. Not every OOM is recoverable by retraction; if even the last request cannot fit, it is aborted.',
          ],
          vl_pressure: [
            'Scheduler 在 running 请求之间做内存/水位判断，如果没有可用 block，就需要减少活跃请求。',
            'The Scheduler checks memory and watermarks across running requests; without usable blocks, it must reduce active demand.',
          ],
          vl_preempt: [
            '_preempt_request 释放请求 block、清理 encoder/connector 状态，并把请求标记为 PREEMPTED。',
            '_preempt_request frees the request blocks, clears encoder and connector state, and marks the request PREEMPTED.',
          ],
          vl_reset: [
            'vLLM 把 num_computed_tokens 归零、清 speculative token，把请求前插到 waiting；后续步骤会重新计算 KV。',
            'vLLM resets num_computed_tokens, clears speculative tokens, and prepends the request to waiting; later steps recompute its KV.',
          ],
          vl_recompute: [
            '重新调度后，请求以 waiting 身份再次经历 prefill/计算；preemption 的代价落在重算上。',
            'After re-admission, the request goes through prefill or computation again as a waiting request; the cost of preemption is recomputation.',
          ],
          sg_abort: [
            '取消从请求输入侧到达 Scheduler（AbortReq）；它可能来自客户端断开，也可能是调度器对单个请求的超时/失败收尾。',
            'Cancellation reaches the Scheduler from the request input side as an AbortReq, whether from a client disconnect or a timeout/failure for a single request.',
          ],
          sg_sched: [
            'waiting 中的请求可以直接弹出；running 中的请求需要从当前 batch 过滤，并进入统一的完成/资源释放路径。',
            'A waiting request can be popped directly; a running request must be filtered from the current batch and enter the unified finish/release path.',
          ],
          sg_queues: [
            '队列移除只解决“还会不会被调度”；已经占用的 KV 仍要显式释放，并让 TokenizerManager 知道流已经结束。',
            'Removing a request from queues only answers whether it can be scheduled again. Already-held KV must still be released, and TokenizerManager must learn that the stream ended.',
          ],
          sg_kv: [
            '释放请求持有的 KV 页与 RadixCache 锁；如果请求已经不在 batch 中，仍要保证重复释放不会破坏树。',
            'Release the request KV pages and RadixCache locks; even if the request is no longer in the batch, double-free protection must preserve the tree.',
          ],
          sg_reply: [
            'Scheduler 向 TokenizerManager 发送 AbortReq 回执，后者清理请求状态并结束流；Detokenizer 不再收到后续 token。',
            'The Scheduler sends an AbortReq receipt to TokenizerManager, which clears request state and ends the stream; Detokenizer receives no further tokens.',
          ],
          vl_abort: [
            '客户端断开或上层取消后，前端确定要终止哪些 request id。',
            'After a client disconnect or upstream cancellation, the frontend determines which request ids to terminate.',
          ],
          vl_core: [
            '前端通过输入 ROUTER→EngineCore DEALER 发送 ABORT 类型帧；这是独立控制消息，不走 SchedulerOutput 执行面。',
            'The frontend sends an ABORT type frame over the input ROUTER→EngineCore DEALER path; it is a separate control message, not part of the SchedulerOutput execution plane.',
          ],
          vl_sched: [
            'EngineCore 把 abort 交给 Scheduler.finish_requests(FINISHED_ABORTED)，从 waiting/running 中移除并记录完成集合。',
            'EngineCore passes the abort to Scheduler.finish_requests(FINISHED_ABORTED), which removes the request from waiting/running and records it as finished.',
          ],
          vl_blocks: [
            'Scheduler 释放请求的 KV blocks，并触发 encoder/KV connector 的 finish 钩子；FINISHED_ABORTED 与队列移除不代表在途写或 connector 已立刻完成，delay_free 可延迟回收。',
            'The Scheduler frees the request KV blocks and invokes encoder/KV connector finish hooks; FINISHED_ABORTED and queue removal do not mean in-flight writes or connector work are complete, and delay_free may defer reclamation.',
          ],
          vl_out: [
            'EngineCore 发送 EngineCoreOutputs，前端完成流清理；用户看到的是结束/中止，而不是继续等待 token。',
            'EngineCore emits EngineCoreOutputs and the frontend completes stream cleanup; the user sees a stop/abort rather than more waiting tokens.',
          ],
          sg_graph: [
            'ModelRunner 按执行模式（decode / prefill / idle）从不同的执行器与图形捕获路径中选择。',
            'ModelRunner chooses among execution paths and captured graphs by forward mode (decode, prefill, or idle).',
          ],
          sg_decode: [
            'decode 通常使用按 batch size 捕获的 CUDA graph；固定输入形状的重放减少逐层 launch 开销。',
            'Decode usually uses CUDA graphs captured per batch size; replaying fixed input shapes reduces per-layer launch overhead.',
          ],
          sg_piece: [
            'prefill 可使用 piecewise CUDA graph，把可捕获的连续段拼起来，而不是要求整段 prefill 形状固定。',
            'Prefill can use piecewise CUDA graphs that stitch captureable contiguous segments together instead of requiring one fixed prefill shape.',
          ],
          sg_bcg: [
            '可选的 breakable CUDA graph（BCG）允许重放在动态边界处暂停，跑完主机侧工作或变长尾部后再继续。',
            'An optional breakable CUDA graph (BCG) lets replay pause at a dynamic boundary, run host-side work or a variable-length tail, then resume.',
          ],
          sg_replay: [
            '因此 SGLang 的捕获策略不只是一组形状桶，还包含“在哪断开、何时续放”的控制点；相关能力与默认值随模型和后端变化。',
            'SGLang’s capture strategy is therefore more than a set of shape buckets: it also chooses where to break and when to resume replay. Capability and defaults vary by model and backend.',
          ],
          vl_compile: [
            'vLLM 的 compilation 与 graph replay 是两层。CompilationConfig / CUDAGraphMode 给出 NONE、PIECEWISE、FULL 等模式；FULL 可以使用也可以不使用 torch.compile，PIECEWISE 有各自的编译条件。',
            'Compilation and graph replay are separate layers in vLLM. CompilationConfig / CUDAGraphMode exposes modes such as NONE, PIECEWISE, and FULL; FULL may be used with or without torch.compile, and PIECEWISE has its own compilation conditions.',
          ],
          vl_full: [
            'FULL 为解码批次捕获整段模型执行图，减少每一步的 Python 与 kernel launch 开销；是否启用、是否经过 torch.compile 由配置与后端决定。',
            'FULL captures an entire model execution graph for decode batches, reducing per-step Python and kernel launch overhead; configuration and backend decide whether it is used and whether torch.compile is involved.',
          ],
          vl_piece: [
            'PIECEWISE 把可捕获的区段拼接起来，适配 prefill 的变长与动态控制流；它有自己的编译条件，只有必要部分才离开图形。',
            'PIECEWISE stitches captureable regions together for variable-length prefill and dynamic control flow; it has its own compilation conditions, and only the necessary parts leave the graph.',
          ],
          vl_mix: [
            'FULL_AND_PIECEWISE 只是允许组合之一，不是所有后端或模型的必然默认；dispatcher 还会看 batch descriptor 与已捕获 keys。',
            'FULL_AND_PIECEWISE is only one allowed combination, not an inevitable default for every backend or model; the dispatcher also checks the batch descriptor and captured keys.',
          ],
          vl_mrv2: [
            'Model Runner V2 接收 dispatcher 的选择结果；命中哪条图路径取决于 batch descriptor、允许模式、已捕获 keys 以及形状/后端能力。',
            'Model Runner V2 receives the dispatcher’s selection; which graph path is hit depends on the batch descriptor, allowed modes, captured keys, and shape/backend support.',
          ],
        },
      },
      compare: {
        title: ['差异不是功能表，而是几条结构线', 'The difference is structural, not a checklist'],
        lede: [
          '把两边放到同一组问题上对齐：请求在哪里被看见、消息怎样穿过边界、KV 怎样被寻址、CPU/GPU 如何交叠、每一步怎样被填满。',
          'Align both engines against the same questions: where requests become visible, how messages cross boundaries, how KV is addressed, how CPU/GPU overlap, and how each step is filled.',
        ],
        dimension: ['结构问题', 'Structural question'],
        impact: ['对调优的含义', 'Tuning implication'],
        rows: [
          {
            q: ['请求边界放在哪里？', 'Where are request boundaries drawn?'],
            sg: [
              'HTTP + TokenizerManager（主进程）→ Scheduler（ZMQ）→ DetokenizerManager（ZMQ）',
              'HTTP + TokenizerManager (main process) → Scheduler (ZMQ) → DetokenizerManager (ZMQ)',
            ],
            vl: [
              'AsyncLLM / Rust frontend → EngineCore（ZMQ）→ 输出处理',
              'AsyncLLM / Rust frontend → EngineCore (ZMQ) → output processing',
            ],
            impact: [
              'SGLang 把 token 化与 detoken 化放在调度两侧；vLLM 的主要分界是 EngineCore 与前端。',
              'SGLang places tokenization and detokenization on either side of scheduling; vLLM mostly separates EngineCore from the frontend.',
            ],
          },
          {
            q: ['KV 按什么寻址？', 'How is KV addressed?'],
            sg: [
              'token 路径上的 radix 树，节点可分裂；按最长前缀匹配。',
              'A radix tree over token paths with splittable nodes and longest-prefix matching.',
            ],
            vl: [
              '固定大小 block 的 chain hash / block table；完整块可共享，block table 只记录逻辑块到物理块的映射。',
              'Chained hashes and a block table over fixed-size blocks; full blocks can be shared, and the block table only maps logical blocks to physical blocks.',
            ],
            impact: [
              '复用粒度取决于 page、block 与 hash 配置；索引形状本身不能判定哪一方命中更细。',
              'Reuse granularity depends on page, block, and hash configuration; index shape alone does not determine finer reuse.',
            ],
          },
          {
            q: ['消息怎么穿过边界？', 'How do messages cross boundaries?'],
            sg: [
              'ZMQ PUSH/PULL + DEALER RPC；Python 对象；默认 ipc://，DP attention 时切到 tcp://。',
              'ZMQ PUSH/PULL plus DEALER RPC; Python objects; ipc:// by default, tcp:// under DP attention.',
            ],
            vl: [
              'ZMQ ROUTER/DEALER + 独立 PUSH/PULL 回程；msgpack 类型帧；core→worker 走 Executor 的 MessageQueue 抽象（共享内存 ring 或 socket 分支）。',
              'ZMQ ROUTER/DEALER plus a separate PUSH/PULL return path; msgpack type frames; core→worker uses the Executor MessageQueue abstraction (shared-memory ring or socket branches).',
            ],
            impact: [
              'SGLang 通信围绕 Scheduler 与 Detokenizer；vLLM 分为 frontend↔core 与 core↔worker 两套边界。',
              'SGLang centers communication on Scheduler and Detokenizer; vLLM splits it into frontend↔core and core↔worker planes.',
            ],
          },
          {
            q: ['每一步如何填满？', 'How is each step filled?'],
            sg: [
              'PrefillAdder 做 prefill 准入，配合 running batch / chunked prefill 与 KV 预算。',
              'PrefillAdder admits prefills under token, KV, and chunk constraints alongside the running batch.',
            ],
            vl: [
              '统一 token budget 组织一个混合的 SchedulerOutput（prefill chunk + decode）。',
              'A unified token budget drives one mixed SchedulerOutput (prefill chunks + decode).',
            ],
            impact: [
              'SGLang 的准入是缓存感知的独立决策；vLLM 的预算更像一个统一装箱问题。',
              'SGLang admission is a separate, cache-aware decision; vLLM’s budget behaves more like unified packing.',
            ],
          },
          {
            q: ['怎样把 CPU 与 GPU 错开？', 'How are CPU and GPU offset?'],
            sg: [
              'overlap scheduler：本批 GPU forward 与上一批 CPU 结果处理重叠。',
              'Overlap scheduler: current GPU forward overlaps CPU processing of the previous batch.',
            ],
            vl: [
              'EngineCore 忙循环；async scheduling / PP batch queue 允许流水线交叠。',
              'EngineCore busy loop; async scheduling and the PP batch queue allow pipeline overlap.',
            ],
            impact: [
              '都追求内核不停顿，但交叠发生的位置不同。',
              'Both aim to keep kernels busy, but overlap happens at different layers.',
            ],
          },
          {
            q: ['如何跨实例扩展？', 'How does it scale across instances?'],
            sg: [
              'cache-aware Rust router、PD/EPD、HiCache、DP attention / EP。',
              'Cache-aware Rust router, PD/EPD, HiCache, and DP attention / EP.',
            ],
            vl: [
              'KVConnectorV1、DP coordinator、TP/PP/EP、弹性 EP。',
              'KVConnectorV1, a DP coordinator, TP/PP/EP, and elastic EP.',
            ],
            impact: [
              'SGLang 把前缀命中写进路由；vLLM 把分离做成可插拔 connector。',
              'SGLang writes prefix hits into routing; vLLM turns separation into a pluggable connector.',
            ],
          },
        ],
        note: [
          '前缀密集、角色分离明显的负载，需要重点核对 SGLang 的缓存与路由路径。\n前端和连接器需要替换时，需要重点核对 vLLM V1 的接口边界与后端条件。\n这些是源码阅读与实验方向，不构成性能排名；实际取舍需要相同条件的测量。',
          'For prefix-heavy workloads with distinct roles, inspect SGLang’s cache and routing paths.\nFor replaceable frontends and connectors, inspect vLLM V1 interfaces and backend conditions.\nThese are source-reading and experiment directions, not a performance ranking; tradeoffs require comparable measurements.',
        ],
      },
      wireTable: {
        title: ['通信剖面：同一件事，不同的线', 'Wire profile: the same job over different lines'],
        lede: [
          '两边都用 ZMQ，但套接字图不一样。下表把控制面、回程、执行面与跨实例数据面拆开对齐。',
          'Both use ZMQ, but their socket graphs differ. This table aligns the control plane, return path, execution plane, and cross-instance data path.',
        ],
        channel: ['通道', 'Channel'],
        rows: [
          {
            channel: ['入口 → 调度', 'Entry → scheduler'],
            sg: [
              'ZMQ PUSH → PULL，scheduler_input_ipc_name，recv_pyobj；默认 ipc://，DP attention 时 tcp://。',
              'ZMQ PUSH → PULL on scheduler_input_ipc_name with recv_pyobj; ipc:// by default, tcp:// under DP attention.',
            ],
            vl: [
              '独立 EngineCore 拓扑下，前端 ROUTER ↔ EngineCore DEALER；先发 ADD / ABORT / UTILITY / WAKEUP 类型帧，再跟 msgpack 载荷。',
              'With a separate EngineCore, frontend ROUTER ↔ EngineCore DEALER; an ADD / ABORT / UTILITY / WAKEUP type frame precedes the msgpack payload.',
            ],
          },
          {
            channel: ['调度 → 回程', 'Scheduler → return'],
            sg: [
              'Scheduler PUSH → Detokenizer PULL；Detokenizer PUSH → TokenizerManager PULL；两段 ZMQ 与一个独立进程。',
              'Scheduler PUSH → Detokenizer PULL; Detokenizer PUSH → TokenizerManager PULL. Two ZMQ hops and one separate process.',
            ],
            vl: [
              'EngineCore PUSH → 前端 PULL，送 EngineCoreOutputs；输出合并与 detokenize 在前端，ZMQ 多帧 buffer 可复用。',
              'EngineCore PUSH → frontend PULL for EngineCoreOutputs; merge and detokenize in the frontend, with reusable multi-frame ZMQ buffers.',
            ],
          },
          {
            channel: ['调度 ↔ 执行', 'Scheduler ↔ execution'],
            sg: [
              'Scheduler 与 TpModelWorker 同进程；TP/PP 主干 NCCL / Gloo，PP 的 request / proxy 张量走点对点控制与数据通道。',
              'Scheduler and TpModelWorker share a process; TP/PP use NCCL or Gloo, while PP requests and proxy tensors use point-to-point control and data paths.',
            ],
            vl: [
              'EngineCore → Worker 走 Executor 的 MessageQueue 抽象：本地共享内存 ring，另有 local/remote XPUB/SUB 分支与 overflow 广播 SchedulerOutput；Worker → EngineCore 由各自 response MQ 返回 ModelRunnerOutput；Ray 时改走 Ray RPC。',
              'EngineCore → Workers use the Executor MessageQueue abstraction: a local shared-memory ring plus local/remote XPUB/SUB branches and overflow broadcast SchedulerOutput; each Worker returns ModelRunnerOutput on a response MQ. With Ray, this becomes Ray RPC.',
            ],
          },
          {
            channel: ['控制 / 管理', 'Control / management'],
            sg: [
              'Engine → Scheduler 的 DEALER RPC，rpc_ipc_name；多 tokenizer 与 DP controller 各有自己的通道。',
              'Engine → Scheduler DEALER RPC on rpc_ipc_name; multi-tokenizer and DP controllers have their own channels.',
            ],
            vl: [
              '前端与 EngineCore 的 ROUTER/DEALER 同时承载 utility RPC；DP coordinator 有独立 ZMQ；executor 失败通过控制路径上报。',
              'Frontend ↔ EngineCore ROUTER/DEALER also carries utility RPC; the DP coordinator has separate ZMQ sockets, and executor failures are reported over the control path.',
            ],
          },
          {
            channel: ['跨实例数据', 'Cross-instance data'],
            sg: [
              'sgl-router 做 cache-aware 路由；PD/EPD 的 KV 由 Mooncake / NIXL / Mori 等后端搬运，常走 RDMA 或 TCP。',
              'sgl-router performs cache-aware routing; PD/EPD KV moves through backends such as Mooncake, NIXL, or Mori, often over RDMA or TCP.',
            ],
            vl: [
              'KVConnectorV1 metadata 随 SchedulerOutput 下发；worker 侧加载/保存远端 KV，传输可由 NIXL、Mooncake 等承载。',
              'KVConnectorV1 metadata rides with SchedulerOutput; workers load and save remote KV, with transports such as NIXL or Mooncake.',
            ],
          },
        ],
        note: [
          'SGLang：PUSH/PULL 请求链与独立 DEALER RPC，区分请求、回程和控制通道。\nvLLM 前端与 EngineCore：ROUTER/DEALER 输入，独立 PUSH/PULL 回程。\nvLLM core→worker：Executor 的 MessageQueue 包含本地共享内存 ring、local/remote XPUB/SUB 分支与 overflow；ZMQ 还承担通知。\n适用范围：上述 vLLM 套接字描述对应独立 EngineCore 的多进程拓扑；UniProc、Ray 或 Rust frontend 部署会改变执行面细节。',
          'SGLang: a PUSH/PULL request chain and separate DEALER RPC distinguish requests, returns, and control.\nvLLM frontend and EngineCore: ROUTER/DEALER input with a separate PUSH/PULL return path.\nvLLM core→worker: Executor MessageQueue includes a local shared-memory ring, local/remote XPUB/SUB branches, and overflow; ZMQ also carries notifications.\nScope: these vLLM sockets describe a separate EngineCore in a multiprocess topology. UniProc, Ray, or Rust frontend deployments change execution-plane details.',
        ],
      },
      matrices: {
        kv: {
          title: ['KV 生命周期剖面：同一块显存，不同的归属', 'KV lifecycle: same memory, different ownership'],
          lede: [
            '把「请求结束」和「真正回收」拆开。两边都保留可复用前缀，但保留方式与压力回退不同。',
            'Separate request finish from true reclamation. Both retain reusable prefixes, but through different ownership and pressure paths.',
          ],
          first: ['生命周期问题', 'Lifecycle question'],
          rows: [
            {
              label: ['分配单位', 'Allocation unit'],
              sg: [
                'token 路径 + KV 页（page_size 可调）；radix 节点是缓存索引。',
                'Token path plus KV pages (page_size is configurable); radix nodes are the cache index.',
              ],
              vl: [
                '固定大小 block + block table；块是分配、共享与回收单位。',
                'Fixed-size blocks plus a block table; a block is the unit of allocation, sharing, and reclamation.',
              ],
            },
            {
              label: ['正常结束', 'Normal finish'],
              sg: [
                'release_kv_cache(checkpoint=True)：结束请求的 KV 写入 tree_cache.checkpoint，退出请求的使用关系；解锁后归还此前受保护的 KV，可复用前缀仍留在树上。',
                'release_kv_cache(checkpoint=True): the finished request writes KV into tree_cache.checkpoint, drops its usage relationship, and returns previously protected KV after locks are released; reusable prefixes remain in the tree.',
              ],
              vl: [
                '释放请求 blocks；ref_cnt→0 入 free queue，完整块保留 hash。',
                'Release request blocks; ref_cnt→0 returns a block to the free queue, while a full block keeps its hash.',
              ],
            },
            {
              label: ['复用前提', 'Reuse condition'],
              sg: [
                '沿树做最长前缀匹配；命中后锁住路径节点。',
                'Longest-prefix matching through the tree; matched path nodes are locked.',
              ],
              vl: [
                '链式 block hash 命中完整块；尾部未命中分配新块（部分块/COW 属特定分支）。',
                'Chained block-hash hits on full blocks; unmatched tails allocate new blocks (partial-block/COW paths are backend-specific).',
              ],
            },
            {
              label: ['压力回退', 'Pressure fallback'],
              sg: [
                'retract_decode：释放 KV、重置请求、回 waiting；重新估计 new_token_ratio（clamp ≤1，不保证变小）。',
                'retract_decode frees KV, resets and requeues requests; new_token_ratio is re-estimated (clamped to ≤1, not guaranteed lower).',
              ],
              vl: [
                '_preempt_request：释放 blocks、num_computed_tokens=0、PREEMPTED、前插 waiting。',
                '_preempt_request frees blocks, resets num_computed_tokens to zero, marks PREEMPTED, and prepends to waiting.',
              ],
            },
            {
              label: ['真正驱逐', 'True eviction'],
              sg: [
                '无锁且无子节点的叶子；释放 KV 页并从 radix 树删除。',
                'A leaf with no locks and no children; its KV pages are freed and the node leaves the radix tree.',
              ],
              vl: [
                'free queue 中的 cached block 被选中；清 hash 后回收。',
                'A cached block in the free queue is selected; its hash is cleared before reclamation.',
              ],
            },
          ],
          note: [
            'page_size、block_size、hash 尺寸与是否开启前缀缓存都会改变实现细节；表格只对齐机制，不给固定配置。',
            'Page size, block size, hash size, and whether prefix caching is enabled all change the implementation; this table compares mechanisms, not a fixed configuration.',
          ],
        },
        overload: {
          title: ['过载回退剖面：同样是内存不够，不同的退路', 'Overload fallback: same memory pressure, different retreat'],
          lede: [
            '重点不是「谁会 OOM」，而是系统在压力下先牺牲什么、让谁重算，以及怎样避免调度器崩溃。',
            'The question is not who can OOM, but what the system sacrifices first, who recomputes, and how it avoids a scheduler crash.',
          ],
          first: ['退路问题', 'Fallback question'],
          rows: [
            {
              label: ['触发信号', 'Trigger'],
              sg: [
                'decode 前的 KV 池余量检查失败；也可能来自显式 retraction 模式。',
                'The pre-decode KV capacity check fails; an explicit retraction mode can also trigger it.',
              ],
              vl: [
                '没有可用 block 或触及水位压力；调度器需要缩小 running 集合。',
                'No blocks are available or a watermark is reached; the scheduler must shrink the running set.',
              ],
            },
            {
              label: ['回退单位', 'Unit of retreat'],
              sg: [
                'running decode batch 中的请求；retract_decode 可撤下一组请求。',
                'Requests in the running decode batch; retract_decode can withdraw a group of requests.',
              ],
              vl: [
                '单请求 preemption；_preempt_request 按调度器选中的请求执行。',
                'Per-request preemption; _preempt_request applies to the request selected by the scheduler.',
              ],
            },
            {
              label: ['状态动作', 'State action'],
              sg: [
                '释放 KV、reset_for_retract、回 waiting；重新估计 new_token_ratio 并 clamp 为 ≤1，不保证比旧值小。',
                'Release KV, reset_for_retract, and requeue in waiting; new_token_ratio is re-estimated and clamped to ≤1, not guaranteed lower.',
              ],
              vl: [
                '释放 blocks、num_computed_tokens=0、PREEMPTED、前插 waiting。',
                'Free blocks, reset num_computed_tokens to zero, mark PREEMPTED, and prepend to waiting.',
              ],
            },
            {
              label: ['后续代价', 'Follow-up cost'],
              sg: [
                '由 PrefillAdder 重新准入并重算；若最后一个请求也放不下，则中止它。',
                'PrefillAdder re-admits it and computation restarts; if even the last request cannot fit, it is aborted.',
              ],
              vl: [
                '重新 prefill/计算；async scheduling 下还要处理在途输出与陈旧 token。',
                'Prefill or computation restarts; async scheduling must also handle in-flight outputs and stale tokens.',
              ],
            },
            {
              label: ['其它压力策略', 'Other pressure controls'],
              sg: [
                'retraction 之外还会驱逐可回收前缀；具体策略随版本与缓存层级变化。',
                'Beyond retraction, evictable prefixes can be reclaimed; the policy varies by release and cache tier.',
              ],
              vl: [
                '可结合 watermark、connector 延迟释放与不同 executor；这里只对照默认回归路径。',
                'Watermarks, deferred connector frees, and executor choice can combine; this row compares the default retreat path.',
              ],
            },
          ],
          note: [
            '两套系统还有其他资源管理与预留策略；本表只对齐默认的 decode 内存压力回退，不代表全部 OOM 行为。',
            'Both systems have additional resource-management and reservation policies; this table aligns the default decode-pressure retreat, not every OOM behavior.',
          ],
        },
        abort: {
          title: ['取消回执剖面：同样是终止，不同的清场线', 'Cancellation and receipts: same stop, different cleanup path'],
          lede: [
            '取消不只是「停止按钮」。要决定谁移除请求、谁释放 KV、谁处理在途写入，以及谁结束流。',
            'Cancellation is more than a stop button: it decides who removes the request, who frees KV, who handles in-flight writes, and who ends the stream.',
          ],
          first: ['清场问题', 'Cleanup question'],
          rows: [
            {
              label: ['触发入口', 'Trigger entry'],
              sg: [
                '客户端断开或调度器超时/失败，形成 AbortReq 进入请求输入侧。',
                'A client disconnect or scheduler timeout/failure creates an AbortReq on the request input side.',
              ],
              vl: [
                '前端/客户端断开，形成带 request id 的 ABORT 控制帧。',
                'A frontend or client disconnect creates an ABORT control frame carrying request ids.',
              ],
            },
            {
              label: ['控制路径', 'Control path'],
              sg: [
                'ZMQ PUSH/PULL 调度输入；AbortReq 走请求控制路径而非模型执行。',
                'ZMQ PUSH/PULL into scheduler input; AbortReq rides the request control path, not model execution.',
              ],
              vl: [
                '前端 ROUTER→EngineCore DEALER 的 ABORT 类型帧；不进入 SchedulerOutput 执行面。',
                'A frontend ROUTER→EngineCore DEALER ABORT type frame; it does not enter the SchedulerOutput execution plane.',
              ],
            },
            {
              label: ['队列移除', 'Queue removal'],
              sg: [
                'waiting 直接弹出；running 从当前 batch 过滤。',
                'Waiting requests are popped; running requests are filtered from the current batch.',
              ],
              vl: [
                'finish_requests(FINISHED_ABORTED) 从 waiting/running 移除并记录完成。',
                'finish_requests(FINISHED_ABORTED) removes the request from waiting/running and records it as finished.',
              ],
            },
            {
              label: ['资源清理', 'Resource cleanup'],
              sg: [
                'release_kv_cache 释放 KV 页与 radix 锁；已占用的 KV 不会因出队自动消失。',
                'release_kv_cache frees KV pages and radix locks; held KV does not disappear merely because the request left a queue.',
              ],
              vl: [
                '释放 blocks，触发 encoder/KV connector finish；在途写可延迟回收。',
                'Free blocks and invoke encoder/KV connector finish hooks; in-flight writes may defer reclamation.',
              ],
            },
            {
              label: ['完成回执', 'Completion receipt'],
              sg: [
                'AbortReq 回传 TokenizerManager 结束流；Detokenizer 不再产生新 token。',
                'An AbortReq goes back to TokenizerManager to end the stream; Detokenizer emits no further tokens.',
              ],
              vl: [
                'EngineCoreOutputs 回到前端，前端清理流并结束请求。',
                'EngineCoreOutputs returns to the frontend, which cleans up the stream and finishes the request.',
              ],
            },
          ],
          note: [
            '取消期间可能仍有 token/KV 写入在途；async scheduling、KV connector 与前端重试策略会影响延迟释放和回执时机。',
            'Tokens or KV writes may still be in flight during cancellation; async scheduling, KV connectors, and frontend retry policy affect deferred frees and receipt timing.',
          ],
        },
        graph: {
          title: ['图捕获剖面：同样是减少 launch，不同的捕获边界', 'Graph capture: same launch reduction, different boundaries'],
          lede: [
            '捕获什么、在哪里断开、何时回退到 eager，是两边的真正差异；不是简单的「有/没有 CUDA graph」。',
            'What is captured, where replay breaks, and when to fall back to eager are the real differences — not a yes/no CUDA-graph story.',
          ],
          first: ['捕获问题', 'Capture question'],
          rows: [
            {
              label: ['捕获对象', 'Capture target'],
              sg: [
                'ModelRunner 按 forward mode 选择 decode / prefill / piecewise 执行器与图形路径。',
                'ModelRunner selects decode, prefill, or piecewise execution and graph paths by forward mode.',
              ],
              vl: [
                'CompilationConfig / CUDAGraphMode 决定是否捕获及捕获模式；torch.compile 是可选编译路径，FULL 不强制依赖它。',
                'CompilationConfig / CUDAGraphMode decides whether and at what mode to capture; torch.compile is an optional compilation path, and FULL does not require it.',
              ],
            },
            {
              label: ['decode', 'Decode'],
              sg: [
                '固定 batch 形状的 CudaGraphRunner；按形状桶重放。',
                'CudaGraphRunner for fixed batch shapes; replay from shape buckets.',
              ],
              vl: [
                'FULL graph 可作为 decode 整图路径；是否启用及与 PIECEWISE 的组合由配置与后端决定，不是必选默认。',
                'FULL graph can serve as the whole-graph decode path; configuration and backend decide whether it is enabled and combined with PIECEWISE, so it is not a mandatory default.',
              ],
            },
            {
              label: ['prefill', 'Prefill'],
              sg: [
                'piecewise CUDA graph；捕获可拼接的连续段。',
                'Piecewise CUDA graph; capture contiguous segments that can be stitched.',
              ],
              vl: [
                'PIECEWISE graph；它有自己的编译条件，变长/动态部分不强行塞进整图。',
                'PIECEWISE graph; it has its own compilation conditions, and variable-length or dynamic regions are not forced into one whole graph.',
              ],
            },
            {
              label: ['动态边界', 'Dynamic boundary'],
              sg: [
                '可选的 breakable CUDA graph：在边界断开、处理主机侧工作，再续放。',
                'Optional breakable CUDA graph: break at a boundary, run host-side work, then resume replay.',
              ],
              vl: [
                '由 graph mode 与编译边界决定；不支持时可回退到 PIECEWISE / NONE。',
                'Determined by graph mode and compilation boundaries; unsupported combinations can fall back to PIECEWISE or NONE.',
              ],
            },
            {
              label: ['默认与限制', 'Defaults and limits'],
              sg: [
                'BCG 可选；部分版本特性默认开启，支持随后端、模型与量化变化。',
                'BCG is optional; some release features default on, with support varying by backend, model, and quantization.',
              ],
              vl: [
                'FULL_AND_PIECEWISE 只是允许组合之一；dispatcher 按 batch descriptor、允许模式与已捕获 keys 选择，不支持时会调整或回退。',
                'FULL_AND_PIECEWISE is only one allowed combination; the dispatcher chooses by batch descriptor, allowed modes, and captured keys, then adjusts or falls back when unsupported.',
              ],
            },
          ],
          note: [
            'CUDA graph 的可用形状、LoRA/量化/多模态/PP 等限制随版本和后端变化；这是机制对照，不是完整支持矩阵。',
            'Available shapes and LoRA, quantization, multimodal, and PP constraints vary by release and backend; this is a mechanism comparison, not a full support matrix.',
          ],
        },
      },
      release: {
        title: ['两个固定版本把力气花在哪', 'Where the two pinned releases spend their effort'],
        lede: [
          '只摘取与架构阅读相关的版本信号，不把模型清单当架构差异。',
          'Only the release signals that matter for reading architecture; model lists are not architecture differences.',
        ],
        sgBullets: [
          [
            '低延迟模式：可选的 NVIDIA megakernel / AMD monokernel 把单用户 decode 做成融合持久 kernel。',
            'Low-latency mode: optional NVIDIA megakernel / AMD monokernel paths turn single-user decode into fused persistent kernels.',
          ],
          [
            '可选的 Paged Experts：让超过 GPU 显存的 MoE 专家以分页方式运行。',
            'Optional Paged Experts: run MoE experts that exceed GPU memory through paging.',
          ],
          [
            'HiCache 的文件级 L3 现可保留 MLA + 解码上下文并行（DCP）的前缀。',
            'HiCache can now keep prefixes for MLA with decode context parallelism (DCP) in file-level L3 storage.',
          ],
          [
            '统一 radix tree 已是默认；breakable CUDA graph、PD 与大规模 EP 继续加码。',
            'The unified radix tree is now default; breakable CUDA graphs, PD, and large-scale EP keep advancing.',
          ],
        ],
        vlBullets: [
          [
            'Rust frontend 继续补齐与 Python 路径的 parity；v0.33 的 serving 面更完整。',
            'The Rust frontend continues closing parity gaps with the Python path; v0.33 rounds out the serving surface.',
          ],
          [
            'Model Runner V2 是引擎核心主线；异步调度、PP 与 CUDA Graph 在它上面组合。',
            'Model Runner V2 is a core engine theme; async scheduling, PP, and CUDA graphs compose on top of it.',
          ],
          [
            'PyTorch 2.14 与默认 CUDA 13；ROCm 10 成为默认 ROCm 镜像。',
            'PyTorch 2.14 and CUDA 13 by default; ROCm 10 becomes the default ROCm image.',
          ],
          [
            '投机解码（XPress/FLy、DFlash/DSpark）与 KV connector / 弹性 EP 继续扩展。',
            'Speculative decoding (XPress/FLy, DFlash/DSpark) and KV connector / elastic EP capabilities keep expanding.',
          ],
        ],
        note: [
          '发布说明包含大量模型与 kernel 更新；这里只保留与架构阅读相关的线索。',
          'Release notes include many model and kernel updates; this page keeps only the architecture-relevant thread.',
        ],
      },
      sources: {
        title: ['基线与延伸阅读', 'Baseline and further reading'],
        body: [
          '版本与源码事实按本页列出的固定 release 与 commit 核对（SGLang v0.5.23 / 297e54f，vLLM v0.33.0 / 4ba787e）。主图采用单节点 CUDA 教学路径：Python HTTP 入口、TP=2 / PP=1 / DP=1；4 token 只是分块示例，Rust frontend、PD 分离等另列扩展。',
          'Version and source facts are checked against the pinned releases and commits listed on this page (SGLang v0.5.23 / 297e54f, vLLM v0.33.0 / 4ba787e). The main diagrams use a single-node CUDA teaching path: Python HTTP entry with TP=2 / PP=1 / DP=1. Four tokens is only a block-size example; the Rust frontend, PD separation, and similar paths are listed separately.',
        ],
        sgRelease: ['SGLang v0.5.23 release', 'SGLang v0.5.23 release'],
        vlRelease: ['vLLM v0.33.0 release（tag v0.33.0rc3）', 'vLLM v0.33.0 release (tag v0.33.0rc3)'],
        vlArch: ['vLLM architecture overview', 'vLLM architecture overview'],
        sgKvCommon: ['SGLang release_kv_cache / checkpoint 入口', 'SGLang release_kv_cache / checkpoint entry'],
        sgScheduler: ['SGLang scheduler 源码', 'SGLang scheduler source'],
        sgEngine: ['SGLang engine / ZMQ 结构', 'SGLang engine / ZMQ structure'],
        sgIpc: ['SGLang scheduler IPC 通道', 'SGLang scheduler IPC channels'],
        vlCore: ['vLLM EngineCore 通信源码', 'vLLM EngineCore communication source'],
        vlShm: ['vLLM MessageQueue（共享内存 / socket）', 'vLLM MessageQueue (shared memory / socket)'],
        vlExecutor: ['vLLM executor 源码', 'vLLM executor source'],
        sgGraph: ['SGLang breakable CUDA graph 源码', 'SGLang breakable CUDA graph source'],
        vlCompile: ['vLLM CompilationConfig / CUDAGraphMode', 'vLLM CompilationConfig / CUDAGraphMode'],
        sgRadix: ['SGLang RadixCache 源码', 'SGLang RadixCache source'],
        vlBlockPool: ['vLLM BlockPool 源码', 'vLLM BlockPool source'],
        vlSched: ['vLLM Scheduler / preemption 源码', 'vLLM Scheduler / preemption source'],
        checked: ['最后核对 2026-11-05', 'Last checked 2026-11-05'],
      },
      footer: {
        note: [
          '本页是机制教学示意，不构成性能结论。部署前请以对应版本的源码、release notes 与你自己的压测为准。',
          'This page is a mechanism explainer, not a performance conclusion. Confirm against the exact release source, release notes, and your own benchmarks before deployment.',
        ],
        meta: ['推理引擎对照台 · AI Infra Wiki', 'Inference Engine Atlas · AI Infra Wiki'],
      },
      scenes: {
        process: {
          zone: {
            sgMain: ['主进程', 'Main process'],
            sgSched: ['模型 rank 进程 ×2 · 每个含 Scheduler + TpModelWorker', 'model-rank process ×2 · each with Scheduler + TpModelWorker'],
            sgDetok: ['Detokenizer 子进程', 'Detokenizer subprocess'],
            vlFront: ['前端进程', 'Frontend process'],
            vlCore: ['EngineCore 进程', 'EngineCore process'],
            vlWorkers: ['Worker / executor ×2（每 rank 一个）', 'Worker / executor ×2 (one per rank)'],
            vlOut: ['输出回程', 'Output path'],
          },
          note: {
            sg: [
              'ZMQ 承载 tokenized request、token ids 与文本回包；scheduler 是三者之间的汇聚点。',
              'ZMQ carries tokenized requests, token ids, and text replies; the scheduler is the convergence point.',
            ],
            vl: [
              'EngineCore 与前端通过 ZMQ 往返；Executor 把执行工作分派给 Worker（多进程拓扑下为独立进程）。',
              'EngineCore and the frontend exchange messages over ZMQ; the Executor dispatches execution work to Workers (separate processes in the multiprocess topology).',
            ],
          },
          foot: {
            sg: [
              '主进程 + 调度子进程 + detokenizer 子进程',
              'main + scheduler + detokenizer processes',
            ],
            vl: [
              '前端 / EngineCore / 输出处理边界',
              'frontend / EngineCore / output boundary',
            ],
          },
        },
        wire: {
          note: {
            sg: [
              '默认使用 ipc://；DP attention 时切到 TCP。控制与管理走 DEALER RPC；入站请求链用于调度与输出，模型执行由同进程的 TpModelWorker 完成。',
              'Defaults to ipc://; DP attention switches to TCP. Control and management use DEALER RPC; the inbound request chain carries scheduling and output, while the in-process TpModelWorker executes the model.',
            ],
            vl: [
              '前端 ROUTER + PULL，EngineCore DEALER + PUSH；core→worker 由 Executor 的 MessageQueue 抽象承载（共享内存 ring、socket 分支与 overflow），ZMQ 还承担通知。',
              'Frontend ROUTER + PULL, EngineCore DEALER + PUSH; core→worker uses the Executor MessageQueue abstraction (shared-memory ring, socket branches, and overflow), with ZMQ also carrying notifications.',
            ],
          },
          foot: {
            sg: [
              'PUSH/PULL 请求链 · DEALER RPC · TCP on DP',
              'PUSH/PULL request chain · DEALER RPC · TCP under DP',
            ],
            vl: [
              'ROUTER/DEALER 请求 · PUSH/PULL 回程 · Executor MQ',
              'ROUTER/DEALER request · PUSH/PULL return · Executor MQ',
            ],
          },
        },
        prefix: {
          note: {
            sg: [
              'page_size=4（教学粒度）：已有 12 token 缓存 + 新输入 12 token，前 10 相同；复用 8，旧页 9–12 保留，新页 9–12 补算。9–10 相同不单独复用。',
              'page_size=4 (illustrative): a 12-token cache and a 12-token input share the first 10 tokens; 8 are reused, the old 9–12 page is kept, and the new 9–12 page is computed. The shared 9–10 tokens are not reused on their own.',
            ],
            vl: [
              'block/hash 单位 = 4 token；命中两个完整单位（8 / 12）；旧缓存第 3 块（9–12）保留但不命中，新输入尾 4 token 分配一块。',
              'Block/hash unit = 4 tokens; two complete units hit (8/12). The cached third block (9–12) is retained but does not match; the trailing four input tokens allocate one new block.',
            ],
          },
          foot: {
            sg: [
              'page_size=4 · 复用 8/12 · 旧页保留 / 新页补算（示意）',
              'page_size=4 · reuse 8/12 · old page kept / new page computed (illustrative)',
            ],
            vl: [
              '命中 2 个完整块 = 8/12 token · 旧块保留 / 新块补算（示意）',
              'two full blocks hit = 8/12 tokens · old block kept / new block allocated (illustrative)',
            ],
          },
        },
        batch: {
          note: {
            sg: [
              '两套准入判断：running decode 与 PrefillAdder。overlap 让 CPU 与 GPU 错开。',
              'Two admission decisions: running decode and PrefillAdder. Overlap offsets CPU from GPU.',
            ],
            vl: [
              '一个统一 token budget 组织混合 batch；EngineCore 忙循环推进每一步。',
              'One unified token budget forms a mixed batch; the EngineCore busy loop advances each step.',
            ],
          },
          foot: {
            sg: [
              'PrefillAdder 准入 + overlap scheduler',
              'PrefillAdder admission + overlap scheduler',
            ],
            vl: [
              '统一 token budget + EngineCore 忙循环',
              'unified token budget + EngineCore busy loop',
            ],
          },
        },
        disagg: {
          note: {
            sg: [
              'router 与 PD/EPD 角色决定 KV 从哪来；HiCache 再把前缀缓存分层。',
              'Router and PD/EPD roles determine where KV comes from; HiCache tiers prefix storage further.',
            ],
            vl: [
              'connector metadata 随 SchedulerOutput 下发；远端 KV 最终落到本地 block table。',
              'Connector metadata rides with SchedulerOutput; remote KV ends up in the local block table.',
            ],
          },
          foot: {
            sg: [
              'sgl-router · PD/EPD · HiCache',
              'sgl-router · PD/EPD · HiCache',
            ],
            vl: [
              'KVConnectorV1 · DP coordinator · EP',
              'KVConnectorV1 · DP coordinator · EP',
            ],
          },
        },
        crosscut: {
          decor: {
            header: [
              ['保留账本', 'retention ledger'],
              ['压力路径', 'pressure path'],
              ['控制 / 清场', 'control path / cleanup'],
              ['形状桶 → 捕获', 'shape buckets → capture'],
            ],
            caption: {
              0: {
                sg: ['锁与子节点检查通过后才可回收', 'evictable only after lock and child checks'],
                vl: ['hash 在 finish 后保留，evict 时清除', 'hash survives finish, cleared on eviction'],
              },
              1: {
                sg: ['保留请求或等待在途释放 · ratio clamp ≤1', 'keep requests or wait for in-flight frees · ratio clamped ≤1'],
                vl: ['PREEMPTED → 之后重新计算', 'PREEMPTED → recomputed later'],
              },
              2: {
                sg: ['流结束 · AbortReq 关闭 tokenizer 状态', 'stream ends · AbortReq closes tokenizer state'],
                vl: ['流结束 · outputs 携带 abort 完成态', 'stream ends · outputs carry the abort finish'],
              },
            },
          },
          note: {
            0: {
              sg: [
                '正常结束 → release_kv_cache(checkpoint=True) → tree_cache.checkpoint 写入可复用前缀 → 退出请求使用关系 → 解锁后归还未保护 KV。请求结束不等于缓存立即消失。',
                'Normal finish → release_kv_cache(checkpoint=True) → tree_cache.checkpoint records reusable prefixes → drop the request usage relationship → unlock and return unprotected KV. Finish is not eviction.',
              ],
              vl: [
                '正常结束 → 释放 block → ref_cnt 归零进 free queue → 完整块保留 hash。请求结束不等于立即 evict。',
                'Normal finish → release block → ref_cnt=0 returns it to the free queue → a full block keeps its hash. Finish is not immediate eviction.',
              ],
            },
            1: {
              sg: [
                '压力入口是 decode 内存检查；retraction 释放 KV、重置请求、回 waiting，并重新估计（clamp ≤1 的）new_token_ratio。',
                'The pressure entry is the decode memory check; retraction frees KV, resets and requeues the request, and re-estimates new_token_ratio (clamped to ≤1).',
              ],
              vl: [
                '压力入口是 block 不足/水位判断；preemption 释放 blocks、归零已计算 token、标记 PREEMPTED 并前插 waiting。',
                'The pressure entry is low block availability or a watermark check; preemption frees blocks, zeroes computed tokens, marks PREEMPTED, and prepends waiting.',
              ],
            },
            2: {
              sg: [
                '取消路径从 AbortReq 开始，先移队列，再释放 KV，最后把 AbortReq 回传给 TokenizerManager 收尾。',
                'The cancel path starts with AbortReq, removes queue entries, releases KV, and returns AbortReq to TokenizerManager for cleanup.',
              ],
              vl: [
                '取消路径是 frontend ABORT → EngineCore → Scheduler.finish_requests → 释放 block/connector → EngineCoreOutputs 收尾。',
                'The cancel path is frontend ABORT → EngineCore → Scheduler.finish_requests → release block/connector → EngineCoreOutputs cleanup.',
              ],
            },
            3: {
              sg: [
                '固定形状走 decode graph，可变 prefill 走 piecewise；BCG 可选地把重放在动态点断开。',
                'Fixed shapes use decode graphs, variable prefill uses piecewise graphs; BCG can optionally break replay at dynamic points.',
              ],
              vl: [
                'CompilationConfig / CUDAGraphMode 选择 NONE / PIECEWISE / FULL；torch.compile 可选，FULL 不强制依赖它，组合由 batch descriptor、允许模式与已捕获 keys 决定。',
                'CompilationConfig / CUDAGraphMode selects NONE, PIECEWISE, or FULL; torch.compile is optional and not required by FULL, while the dispatcher decides combinations by batch descriptor, allowed modes, and captured keys.',
              ],
            },
          },
          foot: {
            0: {
              sg: [
                'release_kv_cache(checkpoint=True) → tree_cache.checkpoint → unlock/return',
                'release_kv_cache(checkpoint=True) → tree_cache.checkpoint → unlock/return',
              ],
              vl: [
                'finish → ref_cnt 0 → free queue / hash retained',
                'finish → ref_cnt 0 → free queue / hash retained',
              ],
            },
            1: {
              sg: [
                'retract_decode → normalize → re-admit',
                'retract_decode → normalize → re-admit',
              ],
              vl: [
                'preempt → free blocks → PREEMPTED → recompute',
                'preempt → free blocks → PREEMPTED → recompute',
              ],
            },
            2: {
              sg: [
                'AbortReq → queue/KV cleanup → stream end',
                'AbortReq → queue/KV cleanup → stream end',
              ],
              vl: [
                'ABORT → finish_requests → block/connector free',
                'ABORT → finish_requests → block/connector free',
              ],
            },
            3: {
              sg: [
                'decode graph + piecewise + optional BCG',
                'decode graph + piecewise + optional BCG',
              ],
              vl: [
                'allowed combo · backend constraints',
                'allowed combo · backend constraints',
              ],
            },
          },
        },
      },
    };

    function flattenBilingual(node, prefix, zh, en) {
      if (
        Array.isArray(node) &&
        node.length === 2 &&
        typeof node[0] === 'string' &&
        typeof node[1] === 'string'
      ) {
        zh[prefix] = node[0];
        en[prefix] = node[1];
        return;
      }
      if (Array.isArray(node)) {
        node.forEach((child, index) => {
          flattenBilingual(child, prefix ? `${prefix}.${index}` : String(index), zh, en);
        });
        return;
      }
      if (node && typeof node === 'object') {
        Object.keys(node).forEach((key) => {
          flattenBilingual(node[key], prefix ? `${prefix}.${key}` : key, zh, en);
        });
      }
    }

    const DICT_ZH = {};
    const DICT_EN = {};
    flattenBilingual(COPY, '', DICT_ZH, DICT_EN);
atlasLocale.dictionaries = { zh: DICT_ZH, en: DICT_EN };

    const prefersReducedMotion = () => {
      try {
        return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
      } catch {
        return false;
      }
    };

    function IconBase({ size = 18, children, viewBox = '0 0 24 24', strokeWidth = 1.7 }) {
      return h(
        'svg',
        {
          width: size,
          height: size,
          viewBox,
          fill: 'none',
          stroke: 'currentColor',
          strokeWidth,
          strokeLinecap: 'round',
          strokeLinejoin: 'round',
          'aria-hidden': 'true',
          focusable: 'false',
        },
        children,
      );
    }

    function PanelIcon({ size = 18, active = false }) {
      return h(
        'svg',
        {
          width: size,
          height: size,
          viewBox: '0 0 24 24',
          fill: 'none',
          'aria-hidden': 'true',
          focusable: 'false',
        },
        h('circle', {
          cx: 12,
          cy: 3.7,
          r: 1.7,
          fill: 'currentColor',
          opacity: active ? 1 : 0.9,
        }),
        h('path', {
          d: 'M12 5.4v3.2M7.5 8.6h9',
          stroke: 'currentColor',
          strokeWidth: 1.5,
          strokeLinecap: 'round',
          opacity: 0.85,
        }),
        h('path', {
          d: 'M7.5 8.6v5.1c0 1.2.6 2 1.6 2.7l1.9 1.3',
          stroke: 'currentColor',
          strokeWidth: 1.7,
          strokeLinecap: 'round',
          strokeLinejoin: 'round',
        }),
        h('path', {
          d: 'M16.5 8.6v5.1c0 1.2-.6 2-1.6 2.7l-1.9 1.3',
          stroke: 'currentColor',
          strokeWidth: 1.7,
          strokeLinecap: 'round',
          strokeLinejoin: 'round',
        }),
        h('rect', {
          x: 4.4,
          y: 12.2,
          width: 4.2,
          height: 3.2,
          rx: 0.8,
          stroke: 'currentColor',
          strokeWidth: 1.2,
          opacity: 0.7,
        }),
        h('rect', {
          x: 15.4,
          y: 12.2,
          width: 4.2,
          height: 3.2,
          rx: 0.8,
          stroke: 'currentColor',
          strokeWidth: 1.2,
          opacity: 0.7,
        }),
        h('circle', { cx: 12, cy: 20.2, r: 1.5, fill: 'currentColor', opacity: 0.85 }),
      );
    }

    function StepBackIcon() {
      return h(
        IconBase,
        null,
        h('path', { d: 'M18.4 6v12L9.2 12Z', fill: 'currentColor', stroke: 'none' }),
        h('path', { d: 'M5.6 6v12' }),
      );
    }

    function StepForwardIcon() {
      return h(
        IconBase,
        null,
        h('path', { d: 'M5.6 6v12L14.8 12Z', fill: 'currentColor', stroke: 'none' }),
        h('path', { d: 'M18.4 6v12' }),
      );
    }

    function ExternalIcon() {
      return h(
        IconBase,
        { size: 13, strokeWidth: 1.8 },
        h('path', { d: 'M13.5 4.5H19v5.5M19 4.5l-7.2 7.2' }),
        h('path', { d: 'M17.2 13.8v4.4a1.3 1.3 0 0 1-1.3 1.3H5.1a1.3 1.3 0 0 1-1.3-1.3V7.4a1.3 1.3 0 0 1 1.3-1.3h4.2' }),
      );
    }

    function CloseIcon() {
      return h(
        IconBase,
        { size: 15 },
        h('path', { d: 'M6 6l12 12M18 6 6 18' }),
      );
    }
    const ENGINE = {
      sg: { name: 'SGLang', mark: 'SG', version: RELEASES.sg.version },
      vl: { name: 'vLLM', mark: 'VL', version: RELEASES.vl.version },
    };

    const PROCESS_LAYOUT = {
      sg: {
        zones: [
          { x: 22, y: 30, w: 184, h: 178, labelKey: 'scenes.process.zone.sgMain', labelX: 34, labelY: 47 },
          { x: 226, y: 30, w: 272, h: 284, labelKey: 'scenes.process.zone.sgSched', labelX: 238, labelY: 47 },
          { x: 22, y: 224, w: 184, h: 80, labelKey: 'scenes.process.zone.sgDetok', labelX: 34, labelY: 241 },
        ],
        nodes: [
          { id: 'http', x: 36, y: 54, w: 156, h: 56, label: 'HTTP / OpenAI', tag: 'FastAPI', detailKey: 'nodes.process.sg_http' },
          { id: 'tokenizer', x: 36, y: 142, w: 156, h: 56, label: 'TokenizerManager', tag: 'tokenize + stream', detailKey: 'nodes.process.sg_tokenizer' },
          { id: 'scheduler', x: 244, y: 64, w: 236, h: 66, label: 'Scheduler', tag: 'RadixCache · PrefillAdder', detailKey: 'nodes.process.sg_scheduler' },
          { id: 'workers', x: 244, y: 190, w: 236, h: 56, label: 'TpModelWorker', tag: 'per rank · local forward', detailKey: 'nodes.process.sg_workers' },
          { id: 'detok', x: 36, y: 254, w: 156, h: 48, label: 'Detokenizer', tag: 'incremental text', detailKey: 'nodes.process.sg_detok' },
        ],
        edges: [
          { id: 'http-tokenizer', d: 'M114 110V142' },
          { id: 'tokenizer-scheduler', d: 'M192 170H218V97H244', label: 'ZMQ', lx: 204, ly: 157 },
          { id: 'scheduler-workers', d: 'M362 130V190', label: 'GPU', lx: 371, ly: 166 },
          { id: 'scheduler-detok', d: 'M244 97H214V274H192', label: 'token ids', lx: 218, ly: 238, anchor: 'middle' },
          { id: 'detok-tokenizer', d: 'M114 254V198', label: 'text', lx: 123, ly: 228 },
        ],
        active: [
          { nodes: ['http', 'tokenizer'], edges: ['http-tokenizer'] },
          { nodes: ['tokenizer', 'scheduler'], edges: ['tokenizer-scheduler'] },
          { nodes: ['scheduler', 'workers'], edges: ['scheduler-workers'] },
          { nodes: ['scheduler', 'detok', 'tokenizer'], edges: ['scheduler-detok', 'detok-tokenizer'] },
        ],
        noteKey: 'scenes.process.note.sg',
      },
      vl: {
        zones: [
          { x: 22, y: 30, w: 184, h: 178, labelKey: 'scenes.process.zone.vlFront', labelX: 34, labelY: 47 },
          { x: 226, y: 30, w: 272, h: 124, labelKey: 'scenes.process.zone.vlCore', labelX: 238, labelY: 47 },
          { x: 226, y: 166, w: 272, h: 148, labelKey: 'scenes.process.zone.vlWorkers', labelX: 238, labelY: 183 },
          { x: 22, y: 224, w: 184, h: 80, labelKey: 'scenes.process.zone.vlOut', labelX: 34, labelY: 241 },
        ],
        nodes: [
          { id: 'frontend', x: 36, y: 54, w: 156, h: 56, label: 'OpenAI / Python', tag: 'HTTP · Rust optional', detailKey: 'nodes.process.vl_frontend' },
          { id: 'async', x: 36, y: 142, w: 156, h: 56, label: 'AsyncLLM', tag: 'input + output', detailKey: 'nodes.process.vl_async' },
          { id: 'engine', x: 244, y: 64, w: 236, h: 66, label: 'EngineCore', tag: 'busy loop · scheduler', detailKey: 'nodes.process.vl_engine' },
          { id: 'workers', x: 244, y: 202, w: 236, h: 56, label: 'Workers ×2', tag: 'per rank · MRV2 · CUDA graphs', detailKey: 'nodes.process.vl_workers' },
          { id: 'stream', x: 36, y: 254, w: 156, h: 48, label: 'SSE stream', tag: 'async response', detailKey: 'nodes.process.vl_stream' },
        ],
        edges: [
          { id: 'frontend-async', d: 'M114 110V142' },
          { id: 'async-engine', d: 'M192 170H218V97H244', label: 'ZMQ', lx: 204, ly: 157 },
          { id: 'engine-workers', d: 'M362 130V202', label: 'dispatch', lx: 371, ly: 166 },
          { id: 'engine-async', d: 'M244 90H210V180H192', label: 'outputs', lx: 226, ly: 139 },
          { id: 'async-stream', d: 'M114 198V254', label: 'SSE', lx: 123, ly: 228 },
        ],
        active: [
          { nodes: ['frontend', 'async'], edges: ['frontend-async'] },
          { nodes: ['async', 'engine'], edges: ['async-engine'] },
          { nodes: ['engine', 'workers'], edges: ['engine-workers'] },
          { nodes: ['engine', 'async', 'stream'], edges: ['engine-async', 'async-stream'] },
        ],
        noteKey: 'scenes.process.note.vl',
      },
    };

    const PREFIX_LAYOUT = {
      sg: {
        nodes: [
          { id: 'root', x: 38, y: 184, w: 60, h: 44, label: 'root', detailKey: 'nodes.prefix.sg_radix' },
          { id: 'path1', x: 122, y: 177, w: 94, h: 50, label: 'page 1–4', tag: 'matched', detailKey: 'nodes.prefix.sg_path' },
          { id: 'path2', x: 244, y: 177, w: 94, h: 50, label: 'page 5–8', tag: 'matched', detailKey: 'nodes.prefix.sg_path' },
          { id: 'other', x: 382, y: 132, w: 96, h: 46, label: 'page 9–12', tag: 'cached · no match', detailKey: 'nodes.prefix.sg_path' },
          { id: 'tail', x: 382, y: 232, w: 96, h: 46, label: 'page 9–12', tag: 'new · compute', detailKey: 'nodes.prefix.sg_tail' },
        ],
        edges: [
          { id: 'root-path1', d: 'M98 206H122' },
          { id: 'path1-path2', d: 'M216 202H244' },
          { id: 'path2-other', d: 'M338 196C360 196 360 161 382 161' },
          { id: 'path2-tail', d: 'M338 208C360 208 360 249 382 249' },
        ],
        active: [
          { nodes: [], edges: [] },
          { nodes: ['root', 'path1', 'path2'], edges: ['root-path1', 'path1-path2'] },
          { nodes: ['tail'], edges: ['path2-tail'] },
          { nodes: ['path1', 'path2', 'tail'], edges: ['root-path1', 'path1-path2', 'path2-tail'] },
        ],
        noteKey: 'scenes.prefix.note.sg',
      },
      vl: {
        nodes: [
          { id: 'hash', x: 38, y: 116, w: 204, h: 148, label: 'Hash table', tag: 'chain hash per block', detailKey: 'nodes.prefix.vl_hash', card: true },
          { id: 'pool', x: 276, y: 116, w: 206, h: 148, label: 'BlockPool', tag: 'paged KV blocks', detailKey: 'nodes.prefix.vl_pool', card: true },
          { id: 'table', x: 38, y: 280, w: 444, h: 48, label: 'block table', tag: 'logical → physical', detailKey: 'nodes.prefix.vl_table' },
        ],
        edges: [
          { id: 'hash-pool', d: 'M242 190H276', label: 'lookup', lx: 259, ly: 180, anchor: 'middle' },
        ],
        active: [
          { nodes: [], edges: [] },
          { nodes: ['hash', 'pool'], edges: ['hash-pool'] },
          { nodes: ['pool'], edges: [] },
          { nodes: ['hash', 'pool', 'table'], edges: ['hash-pool'] },
        ],
        noteKey: 'scenes.prefix.note.vl',
      },
    };

    const BATCH_LAYOUT = {
      sg: {
        nodes: [
          { id: 'waiting', x: 30, y: 114, w: 108, h: 56, label: 'waiting', tag: 'not admitted', detailKey: 'nodes.batch.sg_queue' },
          { id: 'adder', x: 158, y: 114, w: 132, h: 56, label: 'PrefillAdder', tag: 'token + KV gate', detailKey: 'nodes.batch.sg_adder' },
          { id: 'batch', x: 310, y: 114, w: 180, h: 64, label: 'ScheduleBatch', tag: 'this step', detailKey: 'nodes.batch.sg_batch' },
          { id: 'running', x: 30, y: 196, w: 108, h: 56, label: 'running', tag: 'decode', detailKey: 'nodes.batch.sg_queue' },
          { id: 'gpu', x: 158, y: 196, w: 132, h: 56, label: 'GPU forward', tag: 'kernels', detailKey: 'nodes.batch.sg_gpu' },
          { id: 'output', x: 310, y: 196, w: 180, h: 56, label: 'token ids', tag: 'detokenizer', detailKey: 'nodes.batch.sg_gpu' },
          { id: 'overlap', x: 30, y: 276, w: 460, h: 76, label: 'overlap scheduler', tag: 'CPU ↔ GPU', detailKey: 'nodes.batch.sg_overlap', card: true },
        ],
        edges: [
          { id: 'waiting-adder', d: 'M138 142H158' },
          { id: 'adder-batch', d: 'M290 142H310' },
          { id: 'running-gpu', d: 'M138 224H158' },
          { id: 'batch-gpu', d: 'M350 178V188H224V196' },
          { id: 'gpu-output', d: 'M290 224H310' },
        ],
        active: [
          { nodes: ['waiting', 'running'], edges: [] },
          { nodes: ['waiting', 'adder', 'batch'], edges: ['waiting-adder', 'adder-batch'] },
          { nodes: ['running', 'batch', 'gpu', 'overlap'], edges: ['running-gpu', 'batch-gpu'] },
          { nodes: ['gpu', 'output'], edges: ['gpu-output'] },
        ],
        noteKey: 'scenes.batch.note.sg',
      },
      vl: {
        nodes: [
          { id: 'queue', x: 34, y: 136, w: 136, h: 54, label: 'waiting + running', tag: 'one queue set', detailKey: 'nodes.batch.vl_queue' },
          { id: 'budget', x: 186, y: 136, w: 128, h: 54, label: 'token budget', tag: 'max tokens', detailKey: 'nodes.batch.vl_budget' },
          { id: 'scheduler', x: 330, y: 136, w: 154, h: 54, label: 'SchedulerOutput', tag: 'per step', detailKey: 'nodes.batch.vl_schedout' },
          { id: 'workers', x: 186, y: 214, w: 128, h: 54, label: 'Workers', tag: 'ModelRunner', detailKey: 'nodes.batch.vl_workers' },
          { id: 'output', x: 330, y: 214, w: 154, h: 54, label: 'EngineCoreOutputs', tag: 'frontend', detailKey: 'nodes.batch.vl_schedout' },
          { id: 'async', x: 30, y: 278, w: 460, h: 76, label: 'async scheduling / PP batch queue', tag: 'multiple steps in flight', detailKey: 'nodes.batch.vl_async', card: true },
        ],
        edges: [
          { id: 'queue-budget', d: 'M170 163H186' },
          { id: 'budget-scheduler', d: 'M314 163H330' },
          { id: 'scheduler-workers', d: 'M360 190V202H250V214', label: 'shm MQ', lx: 366, ly: 209 },
          { id: 'workers-output', d: 'M314 241H330' },
        ],
        active: [
          { nodes: ['queue'], edges: [] },
          { nodes: ['queue', 'budget', 'scheduler'], edges: ['queue-budget', 'budget-scheduler'] },
          { nodes: ['scheduler', 'workers', 'async'], edges: ['scheduler-workers'] },
          { nodes: ['workers', 'output', 'async'], edges: ['workers-output'] },
        ],
        noteKey: 'scenes.batch.note.vl',
      },
    };

    const DISAGG_LAYOUT = {
      sg: {
        nodes: [
          { id: 'router', x: 30, y: 48, w: 138, h: 56, label: 'sgl-router', tag: 'cache-aware', detailKey: 'nodes.disagg.sg_router' },
          { id: 'prefill', x: 190, y: 48, w: 134, h: 56, label: 'Prefill pool', tag: 'PD / EPD', detailKey: 'nodes.disagg.sg_kv' },
          { id: 'decode', x: 354, y: 48, w: 136, h: 56, label: 'Decode pool', tag: 'KV consumer', detailKey: 'nodes.disagg.sg_kv' },
          { id: 'kv', x: 190, y: 128, w: 134, h: 52, label: 'KV transfer', tag: 'Mooncake / NIXL', detailKey: 'nodes.disagg.sg_kv' },
          { id: 'l1', x: 30, y: 212, w: 138, h: 52, label: 'L1 GPU', tag: 'hot KV', detailKey: 'nodes.disagg.sg_hicache' },
          { id: 'l2', x: 190, y: 212, w: 134, h: 52, label: 'L2 CPU', tag: 'HiCache', detailKey: 'nodes.disagg.sg_hicache' },
          { id: 'l3', x: 354, y: 212, w: 136, h: 52, label: 'L3 storage', tag: 'file tier', detailKey: 'nodes.disagg.sg_hicache' },
          { id: 'scale', x: 30, y: 298, w: 460, h: 52, label: 'Cluster controls', tag: 'DP attention · EP / EPLB', detailKey: 'nodes.disagg.sg_scale' },
        ],
        edges: [
          { id: 'router-prefill', d: 'M168 76H190' },
          { id: 'prefill-kv', d: 'M257 104V128', label: 'KV', lx: 266, ly: 119 },
          { id: 'kv-decode', d: 'M324 154H422V104', label: 'KV', lx: 338, ly: 144 },
          { id: 'l1-l2', d: 'M168 238H190', twoWay: true },
          { id: 'l2-l3', d: 'M324 238H354', twoWay: true },
        ],
        active: [
          { nodes: ['router', 'prefill', 'decode'], edges: ['router-prefill'] },
          { nodes: ['prefill', 'kv', 'decode'], edges: ['prefill-kv', 'kv-decode'] },
          { nodes: ['l1', 'l2', 'l3'], edges: ['l1-l2', 'l2-l3'] },
          { nodes: ['scale'], edges: [] },
        ],
        noteKey: 'scenes.disagg.note.sg',
      },
      vl: {
        nodes: [
          { id: 'router', x: 30, y: 48, w: 138, h: 56, label: 'Frontend / DP', tag: 'coordinator', detailKey: 'nodes.disagg.vl_router' },
          { id: 'prefill', x: 190, y: 48, w: 134, h: 56, label: 'Prefill pool', tag: 'KV producer', detailKey: 'nodes.disagg.vl_connector' },
          { id: 'decode', x: 354, y: 48, w: 136, h: 56, label: 'Decode pool', tag: 'KV consumer', detailKey: 'nodes.disagg.vl_connector' },
          { id: 'connector', x: 190, y: 128, w: 134, h: 52, label: 'KVConnectorV1', tag: 'meta + RDMA', detailKey: 'nodes.disagg.vl_connector' },
          { id: 'meta', x: 30, y: 212, w: 138, h: 52, label: 'Scheduler meta', tag: 'per step', detailKey: 'nodes.disagg.vl_connector' },
          { id: 'remote', x: 190, y: 212, w: 134, h: 52, label: 'remote KV', tag: 'block load', detailKey: 'nodes.disagg.vl_offload' },
          { id: 'offload', x: 354, y: 212, w: 136, h: 52, label: 'offload / HMA', tag: 'block owner', detailKey: 'nodes.disagg.vl_offload' },
          { id: 'scale', x: 30, y: 298, w: 460, h: 52, label: 'Parallel controls', tag: 'TP · PP · DP · EP · elastic', detailKey: 'nodes.disagg.vl_scale' },
        ],
        edges: [
          { id: 'router-prefill', d: 'M168 76H190' },
          { id: 'prefill-connector', d: 'M257 104V128', label: 'KV', lx: 266, ly: 119 },
          { id: 'connector-decode', d: 'M324 154H422V104', label: 'KV', lx: 338, ly: 144 },
          { id: 'meta-remote', d: 'M168 238H190', twoWay: true },
          { id: 'remote-offload', d: 'M324 238H354', twoWay: true },
        ],
        active: [
          { nodes: ['router', 'prefill', 'decode'], edges: ['router-prefill'] },
          { nodes: ['prefill', 'connector', 'decode'], edges: ['prefill-connector', 'connector-decode'] },
          { nodes: ['meta', 'remote', 'offload'], edges: ['meta-remote', 'remote-offload'] },
          { nodes: ['scale'], edges: [] },
        ],
        noteKey: 'scenes.disagg.note.vl',
      },
    };

    const WIRE_LAYOUT = {
      sg: {
        nodes: [
          { id: 'tok', x: 26, y: 48, w: 140, h: 58, label: 'TokenizerManager', tag: 'main · PUSH / PULL', detailKey: 'nodes.wire.sg_tok' },
          { id: 'sched', x: 214, y: 48, w: 132, h: 58, label: 'Scheduler', tag: 'rank 0 · PULL', detailKey: 'nodes.wire.sg_sched' },
          { id: 'detok', x: 376, y: 48, w: 118, h: 58, label: 'Detokenizer', tag: 'PULL · text', detailKey: 'nodes.wire.sg_detok' },
          { id: 'rpc', x: 26, y: 142, w: 140, h: 50, label: 'Engine RPC', tag: 'DEALER', detailKey: 'nodes.wire.sg_rpc' },
          { id: 'workers', x: 214, y: 142, w: 132, h: 50, label: 'TpModelWorker', tag: 'same process', detailKey: 'nodes.wire.sg_workers' },
          { id: 'protocol', x: 26, y: 226, w: 468, h: 118, label: 'sockets & payloads', tag: 'ZMQ · recv_pyobj · IPC', detailKey: 'nodes.wire.sg_protocol', card: true },
        ],
        edges: [
          { id: 'tok-sched', d: 'M166 77H214', label: 'PUSH → PULL', lx: 190, ly: 68, anchor: 'middle' },
          { id: 'sched-detok', d: 'M346 77H376', label: 'PUSH', lx: 361, ly: 68, anchor: 'middle' },
          { id: 'detok-tok', d: 'M435 106V124H96V106', label: 'text · PUSH → PULL', lx: 432, ly: 136, anchor: 'end' },
          { id: 'rpc-sched', d: 'M166 167H190V91H214', label: 'DEALER', lx: 190, ly: 143, anchor: 'middle' },
          { id: 'workers-sched', d: 'M280 142V106', label: 'in-process', lx: 287, ly: 127 },
        ],
        active: [
          { nodes: ['tok', 'sched'], edges: ['tok-sched'] },
          { nodes: ['sched', 'detok', 'tok'], edges: ['sched-detok', 'detok-tok'] },
          { nodes: ['sched', 'rpc', 'workers'], edges: ['rpc-sched', 'workers-sched'] },
          { nodes: ['protocol', 'tok', 'sched', 'detok'], edges: [] },
        ],
        protocolRows: [
          ['PUSH / PULL', 'pyobj · ipc:// or tcp://'],
          ['token ids', 'scheduler → detok'],
          ['text', 'detok → tokenizer'],
          ['DEALER RPC', 'control · separate socket'],
        ],
        noteKey: 'scenes.wire.note.sg',
      },
      vl: {
        nodes: [
          { id: 'frontend', x: 26, y: 48, w: 164, h: 58, label: 'AsyncLLM / Rust', tag: 'ROUTER + PULL', detailKey: 'nodes.wire.vl_frontend' },
          { id: 'core', x: 240, y: 48, w: 164, h: 58, label: 'EngineCore', tag: 'DEALER + PUSH', detailKey: 'nodes.wire.vl_core' },
          { id: 'dp', x: 26, y: 142, w: 164, h: 50, label: 'DP coordinator', tag: 'ZMQ control', detailKey: 'nodes.wire.vl_dp' },
          { id: 'workers', x: 240, y: 142, w: 164, h: 50, label: 'Workers', tag: 'response MQ', detailKey: 'nodes.wire.vl_workers' },
          { id: 'protocol', x: 26, y: 226, w: 468, h: 118, label: 'ZMQ envelope + shm MQ', tag: 'msgpack · zero-copy buffers', detailKey: 'nodes.wire.vl_protocol', card: true },
        ],
        edges: [
          { id: 'request', d: 'M190 67H240', label: 'REQ', lx: 215, ly: 59, anchor: 'middle' },
          { id: 'output', d: 'M240 89H190', label: 'OUT', lx: 215, ly: 102, anchor: 'middle' },
          { id: 'core-workers', d: 'M300 106V142', label: 'shm MQ', lx: 307, ly: 127 },
          { id: 'workers-core', d: 'M344 142V106', label: 'response', lx: 351, ly: 127 },
          { id: 'frontend-dp', d: 'M108 106V142', label: 'DP', lx: 115, ly: 126 },
        ],
        active: [
          { nodes: ['frontend', 'core'], edges: ['request'] },
          { nodes: ['core', 'frontend'], edges: ['output'] },
          { nodes: ['core', 'workers'], edges: ['core-workers', 'workers-core'] },
          { nodes: ['dp', 'protocol'], edges: ['frontend-dp'] },
        ],
        protocolRows: [
          ['FRONTEND', 'ROUTER in · PULL out'],
          ['ENGINECORE', 'DEALER in · PUSH out'],
          ['EXECUTOR', 'shm ring · ZMQ fallback'],
          ['WIRE TYPES', 'ADD · ABORT · UTILITY'],
        ],
        noteKey: 'scenes.wire.note.vl',
      },
    };

    const CROSSCUT_LAYOUT = {
      0: {
        sg: {
          nodes: [
            { id: 'req', x: 30, y: 48, w: 124, h: 56, label: 'Finished req', tag: 'usage relation', detailKey: 'nodes.crosscut.sg_finish' },
            { id: 'radix', x: 180, y: 48, w: 150, h: 56, label: 'RadixCache', tag: 'tree_cache.checkpoint', detailKey: 'nodes.crosscut.sg_radix' },
            { id: 'pool', x: 356, y: 48, w: 134, h: 56, label: 'KV pool', tag: 'unprotected return', detailKey: 'nodes.crosscut.sg_pool' },
            { id: 'reuse', x: 30, y: 150, w: 124, h: 52, label: 'Next request', tag: 'longest prefix', detailKey: 'nodes.crosscut.sg_reuse' },
            { id: 'evict', x: 180, y: 150, w: 150, h: 52, label: 'Evictable leaf', tag: 'no lock', detailKey: 'nodes.crosscut.sg_evict' },
            { id: 'keep', x: 356, y: 150, w: 134, h: 52, label: 'Cached path', tag: 'retained', detailKey: 'nodes.crosscut.sg_radix' },
          ],
          edges: [
            { id: 'req-radix', d: 'M154 76H180', label: 'checkpoint', lx: 167, ly: 67, anchor: 'middle' },
            { id: 'radix-pool', d: 'M330 76H356', label: 'unlock/free', lx: 343, ly: 67, anchor: 'middle' },
            { id: 'reuse-radix', d: 'M154 176H168V104H180', label: 'match', lx: 154, ly: 145 },
            { id: 'evict-pool', d: 'M330 176H340V128H423V104', label: 'free', lx: 346, ly: 140 },
          ],
          activeNodes: ['radix', 'pool', 'evict'],
        },
        vl: {
          nodes: [
            { id: 'req', x: 30, y: 48, w: 124, h: 56, label: 'Finished req', tag: 'release()', detailKey: 'nodes.crosscut.vl_finish' },
            { id: 'table', x: 180, y: 48, w: 150, h: 56, label: 'block table', tag: 'ref count', detailKey: 'nodes.crosscut.vl_table' },
            { id: 'pool', x: 356, y: 48, w: 134, h: 56, label: 'BlockPool', tag: 'free queue', detailKey: 'nodes.crosscut.vl_pool' },
            { id: 'reuse', x: 30, y: 150, w: 124, h: 52, label: 'Next request', tag: 'hash lookup', detailKey: 'nodes.crosscut.vl_reuse' },
            { id: 'hash', x: 180, y: 150, w: 150, h: 52, label: 'hash → block', tag: 'kept until evict', detailKey: 'nodes.crosscut.vl_hash' },
            { id: 'evict', x: 356, y: 150, w: 134, h: 52, label: 'eviction', tag: 'hash cleared', detailKey: 'nodes.crosscut.vl_evict' },
          ],
          edges: [
            { id: 'req-table', d: 'M154 76H180', label: 'release', lx: 167, ly: 67, anchor: 'middle' },
            { id: 'table-pool', d: 'M330 76H356', label: 'ref 0', lx: 343, ly: 67, anchor: 'middle' },
            { id: 'reuse-hash', d: 'M154 176H180', label: 'lookup', lx: 167, ly: 167, anchor: 'middle' },
            { id: 'hash-evict', d: 'M330 176H356', label: 'evict', lx: 343, ly: 167, anchor: 'middle' },
            { id: 'hash-pool', d: 'M255 150V126H423V104', label: 'hash', lx: 260, ly: 140 },
          ],
          activeNodes: ['table', 'pool', 'hash'],
        },
      },
      1: {
        sg: {
          nodes: [
            { id: 'pressure', x: 30, y: 48, w: 132, h: 58, label: 'KV pool low', tag: 'decode check', detailKey: 'nodes.crosscut.sg_pressure' },
            { id: 'running', x: 180, y: 48, w: 150, h: 58, label: 'running decode', tag: 'A · B · C', detailKey: 'nodes.crosscut.sg_retract' },
            { id: 'retract', x: 356, y: 48, w: 134, h: 58, label: 'retract_decode', tag: 'victim batch', detailKey: 'nodes.crosscut.sg_retract' },
            { id: 'waiting', x: 30, y: 150, w: 132, h: 52, label: 'waiting queue', tag: 're-admit', detailKey: 'nodes.crosscut.sg_release' },
            { id: 'release', x: 180, y: 150, w: 150, h: 52, label: 'release KV', tag: 'reset request', detailKey: 'nodes.crosscut.sg_release' },
            { id: 'ratio', x: 356, y: 150, w: 134, h: 52, label: 'new_token_ratio', tag: 're-estimate ≤1', detailKey: 'nodes.crosscut.sg_ratio' },
          ],
          edges: [
            { id: 'pressure-running', d: 'M162 77H180', label: 'check', lx: 171, ly: 67, anchor: 'middle' },
            { id: 'running-retract', d: 'M330 77H356', label: 'select', lx: 343, ly: 67, anchor: 'middle' },
            { id: 'retract-release', d: 'M423 106V128H255V150', label: 'free', lx: 380, ly: 120 },
            { id: 'release-waiting', d: 'M180 176H162', label: 'requeue', lx: 171, ly: 168, anchor: 'middle' },
            { id: 'retract-ratio', d: 'M455 106V150', label: 'ratio', lx: 462, ly: 128 },
          ],
          activeNodes: ['running', 'retract', 'release'],
        },
        vl: {
          nodes: [
            { id: 'pressure', x: 30, y: 48, w: 132, h: 58, label: 'blocks low', tag: 'watermark', detailKey: 'nodes.crosscut.vl_pressure' },
            { id: 'running', x: 180, y: 48, w: 150, h: 58, label: 'running queue', tag: 'pick victim', detailKey: 'nodes.crosscut.vl_preempt' },
            { id: 'preempt', x: 356, y: 48, w: 134, h: 58, label: '_preempt_request', tag: 'free blocks', detailKey: 'nodes.crosscut.vl_preempt' },
            { id: 'waiting', x: 30, y: 150, w: 132, h: 52, label: 'waiting (front)', tag: 'recompute', detailKey: 'nodes.crosscut.vl_reset' },
            { id: 'reset', x: 180, y: 150, w: 150, h: 52, label: 'num_computed=0', tag: 'PREEMPTED', detailKey: 'nodes.crosscut.vl_reset' },
            { id: 'recompute', x: 356, y: 150, w: 134, h: 52, label: 'next prefill', tag: 'rebuild KV', detailKey: 'nodes.crosscut.vl_recompute' },
          ],
          edges: [
            { id: 'pressure-running', d: 'M162 77H180', label: 'blocks', lx: 171, ly: 67, anchor: 'middle' },
            { id: 'running-preempt', d: 'M330 77H356', label: 'victim', lx: 343, ly: 67, anchor: 'middle' },
            { id: 'preempt-reset', d: 'M423 106V128H255V150', label: 'free', lx: 380, ly: 120 },
            { id: 'reset-waiting', d: 'M180 176H162', label: 'prepend', lx: 171, ly: 168, anchor: 'middle' },
            { id: 'reset-recompute', d: 'M330 176H356', label: 'recompute', lx: 343, ly: 168, anchor: 'middle' },
          ],
          activeNodes: ['preempt', 'reset', 'waiting'],
        },
      },
      2: {
        sg: {
          nodes: [
            { id: 'abort', x: 30, y: 48, w: 132, h: 56, label: 'Abort / disconnect', tag: 'AbortReq', detailKey: 'nodes.crosscut.sg_abort' },
            { id: 'scheduler', x: 180, y: 48, w: 150, h: 56, label: 'Scheduler', tag: 'waiting / running', detailKey: 'nodes.crosscut.sg_sched' },
            { id: 'queues', x: 356, y: 48, w: 134, h: 56, label: 'queues', tag: 'pop / filter', detailKey: 'nodes.crosscut.sg_queues' },
            { id: 'kv', x: 30, y: 150, w: 132, h: 52, label: 'release KV', tag: 'tree / pool', detailKey: 'nodes.crosscut.sg_kv' },
            { id: 'reply', x: 180, y: 150, w: 150, h: 52, label: 'TokenizerManager', tag: 'abort reply', detailKey: 'nodes.crosscut.sg_reply' },
            { id: 'stream', x: 356, y: 150, w: 134, h: 52, label: 'stream end', tag: 'SSE close', detailKey: 'nodes.crosscut.sg_reply' },
          ],
          edges: [
            { id: 'abort-scheduler', d: 'M162 76H180', label: 'AbortReq', lx: 171, ly: 67, anchor: 'middle' },
            { id: 'scheduler-queues', d: 'M330 76H356', label: 'pop/filter', lx: 343, ly: 67, anchor: 'middle' },
            { id: 'queues-kv', d: 'M423 104V128H96V150', label: 'free KV', lx: 260, ly: 140 },
            { id: 'scheduler-reply', d: 'M255 104V150', label: 'reply', lx: 262, ly: 128 },
            { id: 'reply-stream', d: 'M330 176H356', label: 'close', lx: 343, ly: 167, anchor: 'middle' },
          ],
          activeNodes: ['scheduler', 'queues', 'kv'],
        },
        vl: {
          nodes: [
            { id: 'abort', x: 30, y: 48, w: 132, h: 56, label: 'Abort / disconnect', tag: 'frontend', detailKey: 'nodes.crosscut.vl_abort' },
            { id: 'core', x: 180, y: 48, w: 150, h: 56, label: 'EngineCore', tag: 'ABORT frame', detailKey: 'nodes.crosscut.vl_core' },
            { id: 'scheduler', x: 356, y: 48, w: 134, h: 56, label: 'Scheduler', tag: 'finish_requests', detailKey: 'nodes.crosscut.vl_sched' },
            { id: 'blocks', x: 30, y: 150, w: 132, h: 52, label: 'blocks / connector', tag: 'free', detailKey: 'nodes.crosscut.vl_blocks' },
            { id: 'out', x: 180, y: 150, w: 150, h: 52, label: 'EngineCoreOutputs', tag: 'FINISHED_ABORTED', detailKey: 'nodes.crosscut.vl_out' },
            { id: 'front', x: 356, y: 150, w: 134, h: 52, label: 'frontend cleanup', tag: 'stream end', detailKey: 'nodes.crosscut.vl_out' },
          ],
          edges: [
            { id: 'abort-core', d: 'M162 76H180', label: 'ZMQ', lx: 171, ly: 67, anchor: 'middle' },
            { id: 'core-scheduler', d: 'M330 76H356', label: 'ABORT', lx: 343, ly: 67, anchor: 'middle' },
            { id: 'scheduler-blocks', d: 'M423 104V128H96V150', label: 'free', lx: 260, ly: 140 },
            { id: 'scheduler-out', d: 'M455 104V138H255V150', label: 'outputs', lx: 398, ly: 132 },
            { id: 'out-front', d: 'M330 176H356', label: 'cleanup', lx: 343, ly: 167, anchor: 'middle' },
          ],
          activeNodes: ['core', 'scheduler', 'blocks'],
        },
      },
      3: {
        sg: {
          nodes: [
            { id: 'graph', x: 30, y: 48, w: 132, h: 56, label: 'ModelRunner', tag: 'forward mode', detailKey: 'nodes.crosscut.sg_graph' },
            { id: 'decode', x: 180, y: 48, w: 150, h: 56, label: 'CudaGraphRunner', tag: 'decode shapes', detailKey: 'nodes.crosscut.sg_decode' },
            { id: 'piece', x: 356, y: 48, w: 134, h: 56, label: 'Piecewise graph', tag: 'prefill chunks', detailKey: 'nodes.crosscut.sg_piece' },
            { id: 'host', x: 30, y: 150, w: 132, h: 52, label: 'host boundary', tag: 'break', detailKey: 'nodes.crosscut.sg_bcg' },
            { id: 'bcg', x: 180, y: 150, w: 150, h: 52, label: 'Breakable CUDA Graph', tag: 'optional', detailKey: 'nodes.crosscut.sg_bcg' },
            { id: 'replay', x: 356, y: 150, w: 134, h: 52, label: 'resume replay', tag: 'dynamic tail', detailKey: 'nodes.crosscut.sg_replay' },
          ],
          edges: [
            { id: 'graph-decode', d: 'M162 76H180', label: 'shape', lx: 171, ly: 67, anchor: 'middle' },
            { id: 'graph-piece', d: 'M96 104V122H423V104', label: 'chunk', lx: 260, ly: 116 },
            { id: 'piece-bcg', d: 'M423 104V134H255V150', label: 'capture', lx: 360, ly: 128 },
            { id: 'bcg-host', d: 'M180 176H162', label: 'break', lx: 171, ly: 168, anchor: 'middle' },
            { id: 'bcg-replay', d: 'M330 176H356', label: 'resume', lx: 343, ly: 168, anchor: 'middle' },
          ],
          activeNodes: ['decode', 'piece', 'bcg'],
        },
        vl: {
          nodes: [
            { id: 'compile', x: 30, y: 48, w: 132, h: 56, label: 'CompilationConfig', tag: 'graph mode', detailKey: 'nodes.crosscut.vl_compile' },
            { id: 'full', x: 180, y: 48, w: 150, h: 56, label: 'FULL', tag: 'decode graph · optional', detailKey: 'nodes.crosscut.vl_full' },
            { id: 'piece', x: 356, y: 48, w: 134, h: 56, label: 'PIECEWISE', tag: 'own compile conditions', detailKey: 'nodes.crosscut.vl_piece' },
            { id: 'none', x: 30, y: 150, w: 132, h: 52, label: 'NONE', tag: 'eager fallback', detailKey: 'nodes.crosscut.vl_compile' },
            { id: 'mix', x: 180, y: 150, w: 150, h: 52, label: 'FULL_AND_PIECEWISE', tag: 'one allowed combo', detailKey: 'nodes.crosscut.vl_mix' },
            { id: 'mrv2', x: 356, y: 150, w: 134, h: 52, label: 'MRV2 execute', tag: 'dispatch', detailKey: 'nodes.crosscut.vl_mrv2' },
          ],
          edges: [
            { id: 'compile-full', d: 'M162 76H180', label: 'mode', lx: 171, ly: 67, anchor: 'middle' },
            { id: 'compile-piece', d: 'M96 104V122H423V104', label: 'mode', lx: 260, ly: 116 },
            { id: 'compile-none', d: 'M66 104V150', label: 'off', lx: 72, ly: 128 },
            { id: 'full-mix', d: 'M255 104V150', label: 'FULL', lx: 262, ly: 128 },
            { id: 'piece-mix', d: 'M423 104V134H330V150', label: 'PIECEWISE', lx: 356, ly: 128 },
            { id: 'mix-mrv2', d: 'M330 176H356', label: 'dispatch', lx: 343, ly: 167, anchor: 'middle' },
          ],
          activeNodes: ['full', 'piece', 'mix'],
        },
      },
    };

    function SvgDefs({ uid }) {
      return h(
        'defs',
        null,
        h(
          'marker',
          {
            id: `${uid}-arrow`,
            viewBox: '0 0 10 10',
            refX: 8,
            refY: 5,
            markerWidth: 6,
            markerHeight: 6,
            orient: 'auto-start-reverse',
          },
          h('path', { d: 'M0 1 9 5 0 9Z', className: 'ia-arrowDef' }),
        ),
        h(
          'marker',
          {
            id: `${uid}-arrow-active`,
            viewBox: '0 0 10 10',
            refX: 8,
            refY: 5,
            markerWidth: 6,
            markerHeight: 6,
            orient: 'auto-start-reverse',
          },
          h('path', { d: 'M0 1 9 5 0 9Z', className: 'ia-arrowDef ia-arrowDef--active' }),
        ),
        h(
          'pattern',
          {
            id: `${uid}-grid`,
            width: 24,
            height: 24,
            patternUnits: 'userSpaceOnUse',
          },
          h('circle', { cx: 1.2, cy: 1.2, r: 0.8, className: 'ia-gridDot' }),
        ),
      );
    }

    function CanvasGrid({ uid }) {
      return h('rect', {
        x: 0,
        y: 0,
        width: 520,
        height: 390,
        fill: `url(#${uid}-grid)`,
        'aria-hidden': 'true',
      });
    }

    function Zone({ zone, t }) {
      return h(
        'g',
        { className: 'ia-zone' },
        h('rect', {
          x: zone.x,
          y: zone.y,
          width: zone.w,
          height: zone.h,
          rx: 9,
          className: 'ia-zoneRect',
        }),
        h(
          'text',
          {
            x: zone.labelX,
            y: zone.labelY,
            className: 'ia-zoneLabel',
          },
          t(zone.labelKey),
        ),
      );
    }

    // Short gaps cannot hold an edge caption. Use ports on the top/bottom
    // of both nodes and put the caption on the longer, exterior segment.
    function routeLabelledEdge(edge, nodes) {
      const match = edge.d.match(/^M([\d.]+) ([\d.]+)H([\d.]+)$/);
      if (!edge.label || !match) return edge;
      const [, a, row, b] = match.map(Number);
      if (edge.label.length * 4.6 <= Math.abs(b - a) - 4) return edge;
      const atPort = (x, node) => row > node.y && row < node.y + node.h &&
        (Math.abs(x - node.x) < 1 || Math.abs(x - node.x - node.w) < 1);
      const source = nodes.find(node => atPort(a, node));
      const target = nodes.find(node => atPort(b, node));
      if (!source || !target) return edge;
      const below = row > 140;
      const sy = below ? source.y + source.h : source.y;
      const ty = below ? target.y + target.h : target.y;
      const rail = below ? Math.max(sy, ty) + 20 : Math.min(sy, ty) - 22;
      const sx = source.x + source.w / 2, tx = target.x + target.w / 2;
      return { ...edge, d: `M${sx} ${sy}V${rail}H${tx}V${ty}`,
        lx: (sx + tx) / 2, ly: rail - 6, anchor: 'middle' };
    }

    function Edge({ edge: originalEdge, nodes, active, uid }) {
      const edge = routeLabelledEdge(originalEdge, nodes);
      const activeId = `${uid}-arrow-active`;
      const idleId = `${uid}-arrow`;
      return h(
        'g',
        { className: `ia-edgeGroup${active ? ' is-active' : ''}`, 'data-edge-id': edge.id },
        h('path', {
          d: edge.d,
          className: `ia-edge${edge.solid ? ' is-solid' : ''}${active ? ' is-active' : ''}`,
          markerEnd: `url(#${active ? activeId : idleId})`,
          markerStart: edge.twoWay ? `url(#${active ? activeId : idleId})` : undefined,
        }),
        edge.label
          ? h(
              'text',
              {
                x: edge.lx,
                y: edge.ly,
                textAnchor: edge.anchor,
                className: `ia-edgeLabel${active ? ' is-active' : ''}`,
              },
              edge.label,
            )
          : null,
      );
    }

    function NodeBox({ node, active, selected, onSelect, children }) {
      const interactive = !!node.detailKey;
      const activate = () => {
        if (interactive) onSelect(node);
      };
      return h(
        'g',
        {
          className: [
            'ia-node',
            node.card ? 'is-card' : '',
            active ? 'is-active' : '',
            selected ? 'is-selected' : '',
            interactive ? 'is-clickable' : '',
          ]
            .filter(Boolean)
            .join(' '),
          tabIndex: interactive ? 0 : undefined,
          role: interactive ? 'button' : undefined,
          focusable: interactive ? 'true' : 'false',
          'aria-pressed': interactive ? (selected ? 'true' : 'false') : undefined,
          'aria-label': interactive ? `${node.label}${node.tag ? ', ' + node.tag : ''}` : undefined,
          'data-node-id': node.id,
          onClick: interactive ? activate : undefined,
          onKeyDown: interactive
            ? (event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  activate();
                }
              }
            : undefined,
        },
        h('title', null, node.tag ? `${node.label} — ${node.tag}` : node.label),
        h('rect', {
          x: node.x,
          y: node.y,
          width: node.w,
          height: node.h,
          rx: node.card ? 7 : 4,
          className: 'ia-nodeBox',
        }),
        h(
          'text',
          {
            x: node.x + 12,
            y: node.y + 22,
            className: 'ia-nodeLabel',
          },
          node.label,
        ),
        node.tag
          ? h(
              'text',
              {
                x: node.x + 12,
                y: node.y + 38,
                className: 'ia-nodeTag',
              },
              node.tag,
            )
          : null,
        active ? h('circle', { cx: node.x + node.w - 11, cy: node.y + 11, r: 3.1, className: 'ia-nodePulse' }) : null,
        children,
      );
    }

    function EngineBoard({ side, chapter, step, stepCount, t, footKey, children }) {
      const engine = ENGINE[side];
      const release = RELEASES[side];
      const [zoomed, setZoomed] = React.useState(false);
      const noteKey = chapter === 'crosscut' ? `scenes.crosscut.note.${step}.${side}` : `scenes.${chapter}.note.${side}`;
      return h(
        'article',
        {
          className: `ia-board ia-board--${side}${zoomed ? ' is-zoomed' : ''}`,
          'data-engine': side,
        },
        h(
          'header',
          { className: 'ia-boardHead' },
          h('span', { className: 'ia-engineMark', 'aria-hidden': 'true' }, engine.mark),
          h(
            'div',
            { className: 'ia-boardTitleWrap' },
            h('h3', { className: 'ia-boardName' }, engine.name),
            h('p', { className: 'ia-boardRoute' }, t(side === 'sg' ? 'board.sgRoute' : 'board.vlRoute')),
          ),
          h(
            'div',
            { className: 'ia-boardStamp' },
            h('span', { className: 'ia-boardVersion' }, engine.version),
            h('span', { className: 'ia-boardDate' }, release.date),
          ),
          h('button', {
            className: 'ia-boardZoom', type: 'button', 'aria-pressed': zoomed,
            onClick: () => setZoomed(value => !value),
          }, t(zoomed ? 'board.fit' : 'board.zoom')),
        ),
        h(
          'div',
          {
            className: 'ia-boardStage',
            tabIndex: 0,
            role: 'region',
            'aria-label': t(zoomed ? 'board.scrollHint' : 'board.fullView'),
          },
          h('div', { className: 'ia-boardSvgWrap' }, children),
        ),
        zoomed ? h('p', { className: 'ia-boardScrollHint' }, t('board.scrollHint')) : null,
        h('p', { className: 'ia-sceneNote' }, t(noteKey)),
        h(
          'footer',
          { className: 'ia-boardFoot' },
          h(
            'span',
            { className: 'ia-boardFootText' },
            t(footKey || `scenes.${chapter}.foot.${side}`),
          ),
          h(
            'span',
            { className: 'ia-boardStep' },
            `${String(step + 1).padStart(2, '0')} / ${String(stepCount).padStart(2, '0')}`,
          ),
        ),
      );
    }

    function useActiveState(layout, step) {
      return useMemo(() => {
        const state = layout.active[step] || { nodes: [], edges: [] };
        return {
          nodes: new Set(state.nodes || []),
          edges: new Set(state.edges || []),
        };
      }, [layout, step]);
    }

    function isNodeSelected(selected, side, node) {
      return !!selected && selected.side === side && selected.id === node.id;
    }

    function ProcessScene({ step, t, selected, onSelect }) {
      const renderBoard = (side) => {
        const layout = PROCESS_LAYOUT[side];
        const state = layout.active[step] || { nodes: [], edges: [] };
        const activeNodes = new Set(state.nodes);
        const activeEdges = new Set(state.edges);
        const uid = `process-${side}`;
        return h(
          EngineBoard,
          {
            key: side,
            side,
            chapter: 'process',
            step,
            stepCount: 4,
            t,
          },
          h(
            'svg',
            {
              viewBox: '0 0 520 366',
              className: 'ia-svg',
              role: 'group',
              'aria-label': `${ENGINE[side].name} process architecture`,
            },
            h(SvgDefs, { uid }),
            h(CanvasGrid, { uid }),
            layout.zones.map((zone, index) => h(Zone, { key: index, zone, t })),
            layout.edges.map((edge) =>
              h(Edge, {
                key: edge.id,
                edge,
              nodes: layout.nodes,
                active: activeEdges.has(edge.id),
                uid,
              }),
            ),
            layout.nodes.map((node) =>
              h(NodeBox, {
                key: node.id,
                node,
                active: activeNodes.has(node.id),
                selected: isNodeSelected(selected, side, node),
                onSelect: (picked) => onSelect({ side, id: picked.id, label: picked.label, tag: picked.tag, detailKey: picked.detailKey }),
              }),
            ),
          ),
        );
      };
      return h(React.Fragment, null, renderBoard('sg'), renderBoard('vl'));
    }

    function PrefixTokenRail({ side, step }) {
      const tokens = Array.from({ length: 12 }, (_, index) => index);
      const hit = step >= 1 ? 8 : 0;
      const newFrom = step >= 2 ? 8 : 12;
      const showReadout = step >= 1;
      return h(
        React.Fragment,
        null,
        h(
          'text',
          { x: 34, y: 27, className: 'ia-sceneKicker' },
          side === 'sg' ? 'Radix path · page-aligned' : 'Block hash · block-level',
        ),
        h(
          'text',
          { x: 486, y: 27, textAnchor: 'end', className: 'ia-sceneReadout' },
          showReadout ? `hit ${String(hit).padStart(2, '0')} / 12` : 'lookup',
        ),
        side === 'vl'
          ? tokens
              .filter((index) => index % 4 === 0)
              .map((index) =>
                h(
                  'rect',
                  {
                    key: `bracket-${index}`,
                    x: 31 + (index / 4) * 132,
                    y: 41,
                    width: 132,
                    height: 36,
                    rx: 4,
                    className: `ia-blockBracket${
                      step >= 1 && index / 4 < 2 ? ' is-hit' : step >= 2 && index / 4 >= 2 ? ' is-new' : ''
                    }`,
                  },
                ),
              )
          : null,
        h('text', { x: 378, y: 39, textAnchor: 'middle', className: 'ia-railFlag' }, '≠ 11–12'),
        tokens.map((index) =>
          h(
            'g',
            {
              key: `tok-${index}`,
              className: `ia-token${index < hit ? ' is-hit' : ''}${
                index >= newFrom && index < 12 ? ' is-new' : ''
              }`,
            },
            h('rect', {
              x: 34 + index * 33,
              y: 46,
              width: 28,
              height: 26,
              rx: 3,
              className: 'ia-tokenBox',
            }),
            h(
              'text',
              {
                x: 48 + index * 33,
                y: 63,
                textAnchor: 'middle',
                className: 'ia-tokenText',
              },
              String(index + 1).padStart(2, '0'),
            ),
          ),
        ),
        side === 'vl'
          ? [0, 1, 2].map((block) =>
              h(
                'text',
                {
                  key: `blabel-${block}`,
                  x: 97 + block * 132,
                  y: 91,
                  textAnchor: 'middle',
                  className: `ia-blockLabel${
                    step >= 1 && block < 2 ? ' is-hit' : step >= 2 && block === 2 ? ' is-new' : ''
                  }`,
                },
                `B${block}`,
              ),
            )
          : h(
              React.Fragment,
              null,
              h('path', {
                d: 'M298 41V82',
                className: `ia-splitMarker${step >= 1 ? ' is-active' : ''}`,
              }),
              h(
                'text',
                {
                  x: 298,
                  y: 91,
                  textAnchor: 'middle',
                  className: `ia-blockLabel${step >= 1 ? ' is-active' : ''}`,
                },
                'split',
              ),
            ),
        h('text', { x: 486, y: 100, textAnchor: 'end', className: 'ia-railTail' }, 'tail 09–12 · compute'),
      );
    }

    function PrefixBoard({ side, step, t, selected, onSelect }) {
      const layout = PREFIX_LAYOUT[side];
      const state = layout.active[step] || { nodes: [], edges: [] };
      const activeNodes = new Set(state.nodes);
      const activeEdges = new Set(state.edges);
      const uid = `prefix-${side}`;
      const isVl = side === 'vl';
      const tableTag = step >= 2 ? '[17, 18, 31]' : step >= 1 ? '[17, 18, ?]' : '[?, ?, ?]';
      const nodes = layout.nodes.map((node) =>
        node.id === 'table' ? { ...node, tag: tableTag } : node,
      );

      const hashChildren = h(
        'g',
        { className: 'ia-cardBody' },
        ['h0 · tok 01–04', 'h1 · tok 05–08', 'h2 · tok 09–12', "h2' · tok 09–12"].map((label, index) =>
          h(
            'g',
            {
              key: `hash-${index}`,
              className: `ia-hashRow${
                step >= 1 && index < 2 ? ' is-hit' : step >= 2 && index === 3 ? ' is-new' : ''
              }`,
            },
            h('rect', {
              x: 50,
              y: 168 + index * 24,
              width: 180,
              height: 21,
              rx: 3,
              className: 'ia-rowBox',
            }),
            h(
              'text',
              { x: 59, y: 181 + index * 24, className: 'ia-rowLabel' },
              label,
            ),
            h(
              'text',
              {
                x: 222,
                y: 181 + index * 24,
                textAnchor: 'end',
                className: 'ia-rowValue',
              },
              index === 0 ? '→ blk 17' : index === 1 ? '→ blk 18' : index === 2 ? '→ blk 19' : step >= 2 ? '→ alloc' : '—',
            ),
          ),
        ),
      );

      const poolChildren = h(
        'g',
        { className: 'ia-cardBody' },
        ['B17', 'B18', 'B19', 'B31'].map((label, index) =>
          h(
            'g',
            {
              key: label,
              className: `ia-blockRow${
                step >= 1 && index < 2 ? ' is-hit' : step >= 2 && index === 3 ? ' is-new' : ''
              }`,
            },
            h('rect', {
              x: 288,
              y: 168 + index * 24,
              width: 182,
              height: 21,
              rx: 3,
              className: 'ia-rowBox',
            }),
            h('text', { x: 297, y: 181 + index * 24, className: 'ia-rowLabel' }, label),
            h(
              'text',
              {
                x: 460,
                y: 181 + index * 24,
                textAnchor: 'end',
                className: 'ia-rowValue',
              },
              index < 2 ? (step >= 1 ? 'hit' : 'cached') : index === 2 ? 'kept' : step >= 2 ? 'new' : 'free',
            ),
          ),
        ),
      );

      return h(
        EngineBoard,
        { side, chapter: 'prefix', step, stepCount: 4, t },
        h(
          'svg',
          {
            viewBox: '0 0 520 366',
            className: 'ia-svg',
            role: 'group',
            'aria-label': `${ENGINE[side].name} prefix cache mechanism`,
          },
          h(SvgDefs, { uid }),
          h(CanvasGrid, { uid }),
          h(PrefixTokenRail, { side, step }),
          h('rect', {
            x: 26,
            y: 104,
            width: 468,
            height: 242,
            rx: 8,
            className: 'ia-scenePanel',
          }),
          !isVl ? h(
            'text',
            { x: 38, y: 126, className: 'ia-sceneKicker is-inside' },
            'RadixCache · longest prefix',
          ) : null,
          layout.edges.map((edge) =>
            h(Edge, {
              key: edge.id,
              edge,
              nodes: layout.nodes,
              active: activeEdges.has(edge.id),
              uid,
            }),
          ),
          nodes.map((node) =>
            h(
              NodeBox,
              {
                key: node.id,
                node,
                active: activeNodes.has(node.id),
                selected: isNodeSelected(selected, side, node),
                onSelect: (picked) => onSelect({ side, id: picked.id, label: picked.label, tag: picked.tag, detailKey: picked.detailKey }),
              },
              node.id === 'hash' ? hashChildren : node.id === 'pool' ? poolChildren : null,
            ),
          ),
        ),
      );
    }

    function PrefixScene({ step, t, selected, onSelect }) {
      return h(
        React.Fragment,
        null,
        h(PrefixBoard, { side: 'sg', step, t, selected, onSelect }),
        h(PrefixBoard, { side: 'vl', step, t, selected, onSelect }),
      );
    }

    function BudgetMeters({ side, step }) {
      if (side === 'sg') {
        const decodeWidth = step === 0 ? 138 : 302;
        const prefillWidth = step === 0 ? 48 : 220;
        return h(
          'g',
          null,
          h('text', { x: 30, y: 29, className: 'ia-sceneKicker' }, 'two admission checks'),
          h('text', { x: 30, y: 49, className: 'ia-meterLabel' }, 'decode'),
          h('text', { x: 30, y: 68, className: 'ia-meterLabel' }, 'prefill'),
          h('rect', { x: 118, y: 39, width: 380, height: 10, rx: 5, className: 'ia-meterTrack' }),
          h('rect', {
            x: 118,
            y: 39,
            width: decodeWidth,
            height: 10,
            rx: 5,
            className: 'ia-meterFill ia-meterFill--accent',
          }),
          h('rect', { x: 118, y: 58, width: 380, height: 10, rx: 5, className: 'ia-meterTrack' }),
          h('rect', {
            x: 118,
            y: 58,
            width: prefillWidth,
            height: 10,
            rx: 5,
            className: 'ia-meterFill ia-meterFill--warm',
          }),
          h('text', { x: 498, y: 83, textAnchor: 'end', className: 'ia-meterHint' },
            step >= 2 ? 'chunk + decode in flight' : 'admission pending'),
        );
      }
      return h(
        'g',
        null,
        h('text', { x: 30, y: 29, className: 'ia-sceneKicker' }, 'one unified token budget'),
        h('text', { x: 30, y: 49, className: 'ia-meterLabel' }, 'budget'),
        h('rect', { x: 118, y: 39, width: 380, height: 18, rx: 5, className: 'ia-meterTrack' }),
        h('rect', {
          x: 118,
          y: 39,
          width: 126,
          height: 18,
          rx: 5,
          className: 'ia-meterFill ia-meterFill--accent',
        }),
        h('rect', {
          x: 244,
          y: 39,
          width: 138,
          height: 18,
          rx: 5,
          className: 'ia-meterFill ia-meterFill--warm',
        }),
        h('text', { x: 118, y: 76, className: 'ia-meterHint' }, 'decode'),
        h('text', { x: 244, y: 76, className: 'ia-meterHint' }, 'prefill chunk'),
        h('text', { x: 382, y: 76, className: 'ia-meterHint' }, 'headroom'),
      );
    }

    function BatchBoard({ side, step, t, selected, onSelect }) {
      const layout = BATCH_LAYOUT[side];
      const state = layout.active[step] || { nodes: [], edges: [] };
      const activeNodes = new Set(state.nodes);
      const activeEdges = new Set(state.edges);
      const uid = `batch-${side}`;
      const isSg = side === 'sg';

      const insetChildren = (node) =>
        h(
          'g',
          { className: 'ia-cardBody' },
          isSg
            ? h(
                React.Fragment,
                null,
                h('text', { x: node.x + 12, y: node.y + 52, className: 'ia-insetLabel' }, 'CPU n+1'),
                h('rect', { x: node.x + 86, y: node.y + 44, width: 240, height: 6, rx: 3, className: 'ia-insetTrack' }),
                h('rect', { x: node.x + 86, y: node.y + 44, width: 196, height: 6, rx: 3, className: 'ia-insetFill ia-insetFill--warm' }),
                h('text', { x: node.x + 12, y: node.y + 68, className: 'ia-insetLabel' }, 'GPU n'),
                h('rect', { x: node.x + 86, y: node.y + 60, width: 240, height: 6, rx: 3, className: 'ia-insetTrack' }),
                h('rect', { x: node.x + 86, y: node.y + 60, width: 220, height: 6, rx: 3, className: 'ia-insetFill ia-insetFill--accent' }),
              )
            : h(
                React.Fragment,
                null,
                h('text', { x: node.x + 12, y: node.y + 52, className: 'ia-insetLabel' }, 'step n'),
                h('rect', { x: node.x + 86, y: node.y + 44, width: 240, height: 6, rx: 3, className: 'ia-insetTrack' }),
                h('rect', { x: node.x + 86, y: node.y + 44, width: 156, height: 6, rx: 3, className: 'ia-insetFill ia-insetFill--accent' }),
                h('text', { x: node.x + 12, y: node.y + 68, className: 'ia-insetLabel' }, 'step n+1'),
                h('rect', { x: node.x + 86, y: node.y + 60, width: 240, height: 6, rx: 3, className: 'ia-insetTrack' }),
                h('rect', { x: node.x + 86, y: node.y + 60, width: 214, height: 6, rx: 3, className: 'ia-insetFill ia-insetFill--warm' }),
              ),
        );

      return h(
        EngineBoard,
        { side, chapter: 'batch', step, stepCount: 4, t },
        h(
          'svg',
          {
            viewBox: '0 0 520 366',
            className: 'ia-svg',
            role: 'group',
            'aria-label': `${ENGINE[side].name} scheduling loop mechanism`,
          },
          h(SvgDefs, { uid }),
          h(CanvasGrid, { uid }),
          h(BudgetMeters, { side, step }),
          !isSg
            ? h(
                'g',
                null,
                h('rect', {
                  x: 20,
                  y: 108,
                  width: 480,
                  height: 92,
                  rx: 8,
                  className: 'ia-zoneRect',
                }),
                h(
                  'text',
                  { x: 32, y: 124, className: 'ia-zoneLabel' },
                  'EngineCore busy loop',
                ),
              )
            : null,
          layout.edges.map((edge) =>
            h(Edge, {
              key: edge.id,
              edge,
              nodes: layout.nodes,
              active: activeEdges.has(edge.id),
              uid,
            }),
          ),
          layout.nodes.map((node) =>
            h(
              NodeBox,
              {
                key: node.id,
                node,
                active: activeNodes.has(node.id),
                selected: isNodeSelected(selected, side, node),
                onSelect: (picked) => onSelect({ side, id: picked.id, label: picked.label, tag: picked.tag, detailKey: picked.detailKey }),
              },
              node.id === 'overlap' || node.id === 'async' ? insetChildren(node) : null,
            ),
          ),
        ),
      );
    }

    function BatchScene({ step, t, selected, onSelect }) {
      return h(
        React.Fragment,
        null,
        h(BatchBoard, { side: 'sg', step, t, selected, onSelect }),
        h(BatchBoard, { side: 'vl', step, t, selected, onSelect }),
      );
    }

    function DisaggBoard({ side, step, t, selected, onSelect }) {
      const layout = DISAGG_LAYOUT[side];
      const state = layout.active[step] || { nodes: [], edges: [] };
      const activeNodes = new Set(state.nodes);
      const activeEdges = new Set(state.edges);
      const uid = `disagg-${side}`;
      return h(
        EngineBoard,
        { side, chapter: 'disagg', step, stepCount: 4, t },
        h(
          'svg',
          {
            viewBox: '0 0 520 366',
            className: 'ia-svg',
            role: 'group',
            'aria-label': `${ENGINE[side].name} disaggregation and scaling mechanism`,
          },
          h(SvgDefs, { uid }),
          h(CanvasGrid, { uid }),
          h('text', { x: 30, y: 32, className: 'ia-sceneKicker' },
            side === 'sg' ? 'prefix-aware route · PD roles' : 'connector-owned transport · DP coordination'),
          h('text', { x: 38, y: 202, className: 'ia-sceneKicker is-inside' },
            side === 'sg' ? 'HiCache tiers' : 'block ownership / offload'),
          layout.edges.map((edge) =>
            h(Edge, {
              key: edge.id,
              edge,
              nodes: layout.nodes,
              active: activeEdges.has(edge.id),
              uid,
            }),
          ),
          layout.nodes.map((node) =>
            h(NodeBox, {
              key: node.id,
              node,
              active: activeNodes.has(node.id),
              selected: isNodeSelected(selected, side, node),
              onSelect: (picked) => onSelect({ side, id: picked.id, label: picked.label, tag: picked.tag, detailKey: picked.detailKey }),
            }),
          ),
        ),
      );
    }

    function DisaggScene({ step, t, selected, onSelect }) {
      return h(
        React.Fragment,
        null,
        h(DisaggBoard, { side: 'sg', step, t, selected, onSelect }),
        h(DisaggBoard, { side: 'vl', step, t, selected, onSelect }),
      );
    }

    function WireBoard({ side, step, t, selected, onSelect }) {
      const layout = WIRE_LAYOUT[side];
      const state = layout.active[step] || { nodes: [], edges: [] };
      const activeNodes = new Set(state.nodes);
      const activeEdges = new Set(state.edges);
      const uid = `wire-${side}`;
      const liveRows =
        side === 'sg'
          ? step === 0
            ? [0]
            : step === 1
              ? [1, 2]
              : step === 2
                ? [3]
                : [0, 1, 2, 3]
          : [Math.min(step, 3)];

      const protocolChildren = h(
        'g',
        { className: 'ia-cardBody' },
        layout.protocolRows.map((row, index) =>
          h(
            'g',
            {
              key: `protocol-${index}`,
              className: `ia-protocolRow${liveRows.includes(index) ? ' is-live' : ''}`,
            },
            h('rect', {
              x: 38,
              y: 278 + index * 16,
              width: 444,
              height: 15,
              rx: 2,
              className: 'ia-protocolRowBox',
            }),
            h(
              'text',
              { x: 46, y: 289 + index * 16, className: 'ia-protocolKey' },
              row[0],
            ),
            h(
              'text',
              { x: 160, y: 289 + index * 16, className: 'ia-protocolValue' },
              row[1],
            ),
          ),
        ),
      );

      return h(
        EngineBoard,
        { side, chapter: 'wire', step, stepCount: 4, t },
        h(
          'svg',
          {
            viewBox: '0 0 520 366',
            className: 'ia-svg',
            role: 'group',
            'aria-label': `${ENGINE[side].name} communication topology`,
          },
          h(SvgDefs, { uid }),
          h(CanvasGrid, { uid }),
          layout.edges.map((edge) =>
            h(Edge, {
              key: edge.id,
              edge,
              nodes: layout.nodes,
              active: activeEdges.has(edge.id),
              uid,
            }),
          ),
          layout.nodes.map((node) =>
            h(
              NodeBox,
              {
                key: node.id,
                node,
                active: activeNodes.has(node.id),
                selected: isNodeSelected(selected, side, node),
                onSelect: (picked) => onSelect({ side, id: picked.id, label: picked.label, tag: picked.tag, detailKey: picked.detailKey }),
              },
              node.id === 'protocol' ? protocolChildren : null,
            ),
          ),
        ),
      );
    }

    function WireScene({ step, t, selected, onSelect }) {
      return h(
        React.Fragment,
        null,
        h(WireBoard, { side: 'sg', step, t, selected, onSelect }),
        h(WireBoard, { side: 'vl', step, t, selected, onSelect }),
      );
    }

    function CrosscutDecor({ step, side, t }) {
      const isSg = side === 'sg';
      const panel = h('rect', {
        x: 30,
        y: 226,
        width: 460,
        height: 94,
        rx: 7,
        className: 'ia-scenePanel',
      });
      const header = h(
        'text',
        { x: 42, y: 244, className: 'ia-decorHeader' },
        t(`scenes.crosscut.decor.header.${step}`),
      );

      const chip = (x, y, width, label, accent, key) =>
        h(
          'g',
          {
            key,
            className: `ia-decorChipGroup${accent ? ' is-accent' : ''}`,
          },
          h('rect', {
            x,
            y,
            width,
            height: 24,
            rx: 3,
            className: 'ia-decorChip',
          }),
          h(
            'text',
            {
              x: x + width / 2,
              y: y + 16,
              textAnchor: 'middle',
              className: 'ia-decorChipLabel',
            },
            label,
          ),
        );

      const arrow = (x, y, key) =>
        h(
          'text',
          {
            key,
            x,
            y,
            textAnchor: 'middle',
            className: 'ia-decorArrow',
          },
          '→',
        );

      if (step === 0) {
        const labels = isSg
          ? ['lock_ref 1', 'finish', 'lock_ref 0', 'radix kept']
          : ['ref_cnt 1', 'finish', 'ref_cnt 0', 'hash kept'];
        return h(
          'g',
          { className: 'ia-crosscutDecor', 'aria-hidden': 'true' },
          panel,
          header,
          labels.map((label, index) =>
            chip(42 + index * 106, 258, 96, label, index === labels.length - 1, `chip-${index}`),
          ),
          arrow(143, 275, 'arrow-0'),
          arrow(249, 275, 'arrow-1'),
          arrow(355, 275, 'arrow-2'),
          h(
            'text',
            { x: 42, y: 306, className: 'ia-decorCaption' },
            t(`scenes.crosscut.decor.caption.0.${side}`),
          ),
        );
      }

      if (step === 1) {
        const labels = isSg
          ? ['retract', 'release KV', 're-admit']
          : ['preempt', 'free blocks', 'recompute'];
        return h(
          'g',
          { className: 'ia-crosscutDecor', 'aria-hidden': 'true' },
          panel,
          header,
          h('rect', {
            x: 42,
            y: 252,
            width: 436,
            height: 12,
            rx: 6,
            className: 'ia-gaugeTrack',
          }),
          h('rect', {
            x: 42,
            y: 252,
            width: 401,
            height: 12,
            rx: 6,
            className: 'ia-gaugeFill',
          }),
          h('text', { x: 478, y: 249, textAnchor: 'end', className: 'ia-gaugeLabel' }, '≈92%'),
          labels.map((label, index) =>
            chip(42 + index * 148, 282, 132, label, index === 1, `chip-${index}`),
          ),
          h(
            'text',
            { x: 42, y: 313, className: 'ia-decorCaption' },
            t(`scenes.crosscut.decor.caption.1.${side}`),
          ),
        );
      }

      if (step === 2) {
        const labels = isSg
          ? ['AbortReq', 'queue / batch', 'KV + reply']
          : ['ABORT frame', 'finish_requests', 'blocks + outputs'];
        return h(
          'g',
          { className: 'ia-crosscutDecor', 'aria-hidden': 'true' },
          panel,
          header,
          labels.map((label, index) =>
            chip(42 + index * 148, 260, 132, label, index === 1, `chip-${index}`),
          ),
          arrow(181, 277, 'arrow-0'),
          arrow(329, 277, 'arrow-1'),
          h(
            'text',
            { x: 42, y: 308, className: 'ia-decorCaption' },
            t(`scenes.crosscut.decor.caption.2.${side}`),
          ),
        );
      }

      return h(
        'g',
        { className: 'ia-crosscutDecor', 'aria-hidden': 'true' },
        panel,
        header,
        [1, 2, 4, 8].map((value, index) =>
          h('rect', {
            key: `bar-${value}`,
            x: 42 + index * 62,
            y: 300 - 12 - index * 10,
            width: 46,
            height: 12 + index * 10,
            rx: 2,
            className: `ia-graphBar${index === 3 ? ' is-alt' : ''}`,
          }),
        ),
        [1, 2, 4, 8].map((value, index) =>
          h(
            'text',
            {
              key: `bar-label-${value}`,
              x: 65 + index * 62,
              y: 316,
              textAnchor: 'middle',
              className: 'ia-graphLabel',
            },
            String(value),
          ),
        ),
        isSg
          ? h(React.Fragment, null,
              h('path', {
                d: 'M348 248V310',
                className: 'ia-breakLine',
              }),
              h('text', { x: 356, y: 268, className: 'ia-breakLabel' }, 'break'),
              h('text', { x: 356, y: 288, className: 'ia-decorCaption' }, 'host work'),
              h('text', { x: 356, y: 304, className: 'ia-decorCaption' }, 'resume replay'),
            )
          : h(React.Fragment, null,
              h('text', { x: 348, y: 266, className: 'ia-decorCaption' }, 'FULL'),
              h('text', { x: 348, y: 284, className: 'ia-decorCaption' }, '+ PIECEWISE'),
              h('text', { x: 348, y: 304, className: 'ia-breakLabel' }, 'allowed combo'),
            ),
      );
    }

    function CrosscutBoard({ step, side, t, selected, onSelect }) {
      const layout = CROSSCUT_LAYOUT[step][side];
      const activeNodes = new Set(layout.activeNodes || []);
      const activeEdges = new Set(layout.edges.map((edge) => edge.id));
      const uid = `crosscut-${step}-${side}`;
      return h(
        EngineBoard,
        {
          side,
          chapter: 'crosscut',
          step,
          stepCount: 4,
          t,
          footKey: `scenes.crosscut.foot.${step}.${side}`,
        },
        h(
          'svg',
          {
            viewBox: '0 0 520 366',
            className: 'ia-svg',
            role: 'group',
            'aria-label': `${ENGINE[side].name} crosscut module ${step + 1}`,
          },
          h(SvgDefs, { uid }),
          h(CanvasGrid, { uid }),
          h(CrosscutDecor, { step, side, t }),
          layout.edges.map((edge) =>
            h(Edge, {
              key: edge.id,
              edge,
              nodes: layout.nodes,
              active: activeEdges.has(edge.id),
              uid,
            }),
          ),
          layout.nodes.map((node) =>
            h(NodeBox, {
              key: node.id,
              node,
              active: activeNodes.has(node.id),
              selected: isNodeSelected(selected, side, node),
              onSelect: (picked) => onSelect({ side, id: picked.id, label: picked.label, tag: picked.tag, detailKey: picked.detailKey }),
            }),
          ),
        ),
      );
    }

    function CrosscutScene({ step, t, selected, onSelect }) {
      return h(
        React.Fragment,
        null,
        h(CrosscutBoard, { step, side: 'sg', t, selected, onSelect }),
        h(CrosscutBoard, { step, side: 'vl', t, selected, onSelect }),
      );
    }

    function Scene({ chapter, step, t, selected, onSelect }) {
      if (chapter === 'wire') return h(WireScene, { step, t, selected, onSelect });
      if (chapter === 'prefix') return h(PrefixScene, { step, t, selected, onSelect });
      if (chapter === 'batch') return h(BatchScene, { step, t, selected, onSelect });
      if (chapter === 'disagg') return h(DisaggScene, { step, t, selected, onSelect });
      if (chapter === 'crosscut') return h(CrosscutScene, { step, t, selected, onSelect });
      return h(ProcessScene, { step, t, selected, onSelect });
    }

    const SOURCE_LINKS = [
      {
        key: 'sources.sgRelease',
        url: 'https://github.com/sgl-project/sglang/releases/tag/v0.5.23',
      },
      {
        key: 'sources.vlRelease',
        url: 'https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3',
      },
      {
        key: 'sources.vlArch',
        url: 'https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/docs/design/arch_overview.md',
      },
      {
        key: 'sources.sgScheduler',
        url: 'https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/scheduler.py',
      },
      {
        key: 'sources.sgEngine',
        url: 'https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/entrypoints/engine.py',
      },
      {
        key: 'sources.sgIpc',
        url: 'https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/scheduler_components/ipc_channels.py',
      },
      {
        key: 'sources.vlCore',
        url: 'https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/engine/core.py',
      },
      {
        key: 'sources.vlShm',
        url: 'https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/device_communicators/shm_broadcast.py',
      },
      {
        key: 'sources.vlExecutor',
        url: 'https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/executor/multiproc_executor.py',
      },
      {
        key: 'sources.sgGraph',
        url: 'https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py',
      },
      {
        key: 'sources.vlCompile',
        url: 'https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/config/compilation.py',
      },
      {
        key: 'sources.sgKvCommon',
        url: 'https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/common.py#L400',
      },
      {
        key: 'sources.sgRadix',
        url: 'https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/radix_cache.py#L510',
      },
      {
        key: 'sources.vlBlockPool',
        url: 'https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/block_pool.py',
      },
      {
        key: 'sources.vlSched',
        url: 'https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py',
      },
    ];

    function pad2(value) {
      return String(value).padStart(2, '0');
    }

    function ReleaseCard({ side, t }) {
      const release = RELEASES[side];
      return h(
        'div',
        { className: `ia-releaseCard ia-releaseCard--${side}` },
        h(
          'a',
          {
            className: 'ia-releaseMain',
            href: release.url,
            target: '_blank',
            rel: 'noreferrer',
            'aria-label': `${t('header.sourceCta')}: ${release.name} ${release.version}`,
          },
          h(
            'span',
            { className: 'ia-releaseRow' },
            h('span', { className: 'ia-releaseName' }, release.name),
            h('span', { className: 'ia-releaseVersion' }, release.version),
            h('span', { className: 'ia-releaseArrow', 'aria-hidden': 'true' }, h(ExternalIcon)),
          ),
        ),
        h(
          'span',
          { className: 'ia-releaseMeta' },
          `${release.date} · ${side === 'vl' ? `tag ${release.tag}` : 'GitHub release'} · `,
          h(
            'a',
            {
              className: 'ia-releaseCommit',
              href: release.commitUrl,
              target: '_blank',
              rel: 'noreferrer',
              title: release.commit,
            },
            `@${release.commitShort}`,
          ),
        ),
      );
    }

    function SectionHead({ title, lede }) {
      return h(
        'div',
        { className: 'ia-sectionHead' },
        h('h2', { className: 'ia-sectionTitle' }, title),
        lede ? h('p', { className: 'ia-sectionLede' }, lede) : null,
      );
    }

    function SectionNote({ text }) {
      const items = text.split('\n').map(item => item.trim()).filter(Boolean);
      return items.length > 1
        ? h('ol', { className: 'ia-sectionNote' }, items.map((item, index) => h('li', { key: index }, item)))
        : h('p', { className: 'ia-sectionNote' }, text);
    }

    function ComparisonTable({ t, rows, title, lede }) {
      return h(
        'section',
        { className: 'ia-section', 'aria-labelledby': rows ? 'ia-topic-compare-title' : 'ia-compare-title' },
        h(
          SectionHead,
          {
            title: h('span', { id: rows ? 'ia-topic-compare-title' : 'ia-compare-title' }, title || t('compare.title')),
            lede: lede || t('compare.lede'),
          },
        ),
        h(
          'table',
          { className: 'ia-table ia-compareTable' },
          h(
            'thead',
            null,
            h(
              'tr',
              null,
              h('th', { scope: 'col' }, t('compare.dimension')),
              h('th', { scope: 'col', className: 'ia-thEngine ia-thEngine--sg' }, 'SGLang'),
              h('th', { scope: 'col', className: 'ia-thEngine ia-thEngine--vl' }, 'vLLM'),
              h('th', { scope: 'col' }, t('compare.impact')),
            ),
          ),
          h(
            'tbody',
            null,
            (rows || COPY.compare.rows.map((_, index) => index)).map((index) =>
              h(
                'tr',
                { key: `compare-${index}` },
                h(
                  'td',
                  {
                    'data-label': t('compare.dimension'),
                    className: 'ia-cellQuestion',
                  },
                  t(`compare.rows.${index}.q`),
                ),
                h(
                  'td',
                  {
                    'data-label': 'SGLang',
                    className: 'ia-cellEngine ia-cellEngine--sg',
                  },
                  t(`compare.rows.${index}.sg`),
                ),
                h(
                  'td',
                  {
                    'data-label': 'vLLM',
                    className: 'ia-cellEngine ia-cellEngine--vl',
                  },
                  t(`compare.rows.${index}.vl`),
                ),
                h(
                  'td',
                  {
                    'data-label': t('compare.impact'),
                    className: 'ia-cellImpact',
                  },
                  t(`compare.rows.${index}.impact`),
                ),
              ),
            ),
          ),
        ),
        rows ? null : h(SectionNote, { text: t('compare.note') }),
      );
    }

    function WireTable({ t }) {
      return h(
        'section',
        { className: 'ia-section ia-wireSection', 'aria-labelledby': 'ia-wire-title' },
        h(
          SectionHead,
          {
            title: h('span', { id: 'ia-wire-title' }, t('wireTable.title')),
            lede: t('wireTable.lede'),
          },
        ),
        h(
          'table',
          { className: 'ia-table ia-wireTable' },
          h(
            'thead',
            null,
            h(
              'tr',
              null,
              h('th', { scope: 'col' }, t('wireTable.channel')),
              h('th', { scope: 'col', className: 'ia-thEngine ia-thEngine--sg' }, 'SGLang'),
              h('th', { scope: 'col', className: 'ia-thEngine ia-thEngine--vl' }, 'vLLM'),
            ),
          ),
          h(
            'tbody',
            null,
            COPY.wireTable.rows.map((_, index) =>
              h(
                'tr',
                { key: `wire-${index}` },
                h(
                  'td',
                  {
                    'data-label': t('wireTable.channel'),
                    className: 'ia-cellQuestion',
                  },
                  t(`wireTable.rows.${index}.channel`),
                ),
                h(
                  'td',
                  {
                    'data-label': 'SGLang',
                    className: 'ia-cellEngine ia-cellEngine--sg ia-cellWire',
                  },
                  t(`wireTable.rows.${index}.sg`),
                ),
                h(
                  'td',
                  {
                    'data-label': 'vLLM',
                    className: 'ia-cellEngine ia-cellEngine--vl ia-cellWire',
                  },
                  t(`wireTable.rows.${index}.vl`),
                ),
              ),
            ),
          ),
        ),
        h(SectionNote, { text: t('wireTable.note') }),
      );
    }

    function ProfileTable({ id, t }) {
      const matrix = COPY.matrices[id];
      return h(
        'section',
        {
          className: 'ia-section ia-profileSection',
          'aria-labelledby': `ia-matrix-${id}-title`,
        },
        h(
          SectionHead,
          {
            title: h('span', { id: `ia-matrix-${id}-title` }, t(`matrices.${id}.title`)),
            lede: t(`matrices.${id}.lede`),
          },
        ),
        h(
          'table',
          { className: 'ia-table ia-wireTable ia-profileTable' },
          h(
            'thead',
            null,
            h(
              'tr',
              null,
              h('th', { scope: 'col' }, t(`matrices.${id}.first`)),
              h('th', { scope: 'col', className: 'ia-thEngine ia-thEngine--sg' }, 'SGLang'),
              h('th', { scope: 'col', className: 'ia-thEngine ia-thEngine--vl' }, 'vLLM'),
            ),
          ),
          h(
            'tbody',
            null,
            matrix.rows.map((_, index) =>
              h(
                'tr',
                { key: `${id}-${index}` },
                h(
                  'td',
                  {
                    'data-label': t(`matrices.${id}.first`),
                    className: 'ia-cellQuestion',
                  },
                  t(`matrices.${id}.rows.${index}.label`),
                ),
                h(
                  'td',
                  {
                    'data-label': 'SGLang',
                    className: 'ia-cellEngine ia-cellEngine--sg ia-cellWire',
                  },
                  t(`matrices.${id}.rows.${index}.sg`),
                ),
                h(
                  'td',
                  {
                    'data-label': 'vLLM',
                    className: 'ia-cellEngine ia-cellEngine--vl ia-cellWire',
                  },
                  t(`matrices.${id}.rows.${index}.vl`),
                ),
              ),
            ),
          ),
        ),
        h(SectionNote, { text: t(`matrices.${id}.note`) }),
      );
    }

    function ReleaseRadar({ t }) {
      return h(
        'section',
        { className: 'ia-section ia-releaseSection', 'aria-labelledby': 'ia-release-title' },
        h(
          SectionHead,
          {
            title: h('span', { id: 'ia-release-title' }, t('release.title')),
            lede: t('release.lede'),
          },
        ),
        h(
          'div',
          { className: 'ia-radar' },
          ['sg', 'vl'].map((side) =>
            h(
              'div',
              { key: side, className: `ia-radarCol ia-radarCol--${side}` },
              h(
                'div',
                { className: 'ia-radarHead' },
                h('span', { className: 'ia-radarEngine' }, ENGINE[side].name),
                h('span', { className: 'ia-radarVersion' }, RELEASES[side].version),
              ),
              h(
                'ul',
                { className: 'ia-bullets' },
                COPY.release[`${side}Bullets`].map((_, index) =>
                  h(
                    'li',
                    { key: `${side}-release-${index}` },
                    t(`release.${side}Bullets.${index}`),
                  ),
                ),
              ),
            ),
          ),
        ),
        h('p', { className: 'ia-sectionNote' }, t('release.note')),
      );
    }

    function SourcesSection({ t, chapter }) {
      const keys = new Set(['sources.sgRelease', 'sources.vlRelease', ...chapter.sourceKeys]);
      return h(
        'section',
        { className: 'ia-section ia-sourcesSection', 'aria-labelledby': 'ia-source-title' },
        h(
          'div',
          { className: 'ia-sourceGrid' },
          h(
            'div',
            null,
            h(
              'h2',
              { className: 'ia-sectionTitle', id: 'ia-source-title' },
              t('sources.title'),
            ),
            h('p', { className: 'ia-sourceBody' }, t('sources.body')),
            h('p', { className: 'ia-sourceChecked' }, t('sources.checked')),
          ),
          h(
            'div',
            { className: 'ia-sourceLinks' },
            SOURCE_LINKS.filter(link => keys.has(link.key)).map((link) =>
              h(
                'a',
                {
                  key: link.key,
                  className: 'ia-sourceLink',
                  href: link.url,
                  target: '_blank',
                  rel: 'noreferrer',
                },
                h('span', null, t(link.key)),
                h('span', { className: 'ia-sourceArrow', 'aria-hidden': 'true' }, h(ExternalIcon)),
              ),
            ),
          ),
        ),
      );
    }

    function ChapterLinks({ chapter, t }) {
      const english = atlasLocale.lang === 'en';
      const sources = (chapter.sourceKeys || [])
        .map((key) => SOURCE_LINKS.find((link) => link.key === key))
        .filter(Boolean);
      return h(
        'div',
        { className: 'ia-chapterLinks' },
        sources.length
          ? h(
              'div',
              { className: 'ia-linksGroup' },
              h('span', { className: 'ia-linksLabel' }, t('links.sources')),
              sources.map((link) =>
                h(
                  'a',
                  {
                    key: link.key,
                    className: 'ia-chapterLink',
                    href: link.url,
                    target: '_blank',
                    rel: 'noreferrer',
                  },
                  h('span', null, t(link.key)),
                  h('span', { className: 'ia-linkArrow', 'aria-hidden': 'true' }, '↗'),
                ),
              ),
            )
          : null,
        h(
          'div',
          { className: 'ia-linksGroup' },
          h('span', { className: 'ia-linksLabel' }, t('links.lessons')),
          chapter.lessonLinks.map((link) =>
            h(
              'a',
              { key: link.href, className: 'ia-chapterLink', href: link.href },
              english ? link.label[1] : link.label[0],
            ),
          ),
        ),
        h('p', { className: 'ia-linksNote' }, t('links.note')),
      );
    }

    // Mount only the requested material. A new topic/step gets a fresh closed
    // disclosure, so an open comparison cannot spill into the next lesson.
    function ReadingDisclosure({ title, t, children, className }) {
      const [open, setOpen] = React.useState(false);
      return h('details', {
        className, open,
      },
        h('summary', { onClick: event => {
          event.preventDefault();
          const nextOpen = !open;
          setOpen(nextOpen);
        } },
          h('span', { className: 'ia-readingLabel' }, title),
          h('small', { className: 'ia-readingAction' }, t(open ? 'reading.collapse' : 'reading.expand')),
        ),
        open ? children : null,
      );
    }

    function TopicReading({ chapter, step, t }) {
      const topic = chapter.id;
      const facet = topic === 'crosscut' ? ['finish', 'overload', 'abort', 'graph'][step] : topic;
      const table = topic === 'wire' ? h(WireTable, { t }) :
        topic === 'prefix' ? h(ProfileTable, { id: 'kv', t }) :
        topic === 'crosscut' ? h(ProfileTable, { id: ['kv', 'overload', 'abort', 'graph'][step], t }) :
        h(ComparisonTable, {
          t, rows: { process: [0], batch: [3, 4], disagg: [5] }[topic],
          title: t(`reading.${topic}.title`), lede: t(`reading.${topic}.lede`),
        });
      return h('section', { className: 'ia-topicReading', 'data-topic': topic, 'data-reading': facet },
        h('p', { className: 'ia-readingEyebrow' }, t('reading.optional')),
        h(ReadingDisclosure, {
          key: `${topic}:${step}`, t, className: 'ia-readingDetails', title: t(`reading.titles.${facet}`),
        }, table),
      );
    }

    function ChapterTabs({ chapterIndex, t, onSelect }) {
      return h(
        'div',
        { className: 'ia-chapterTabs', role: 'group', 'aria-label': t('control.chapter') },
        CHAPTERS.map((chapter, index) =>
          h(
            'button',
            {
              key: chapter.id,
              type: 'button',
              className: `ia-chapterTab${index === chapterIndex ? ' is-active' : ''}`,
              'aria-pressed': index === chapterIndex,
              onClick: () => onSelect(index),
            },
            h('span', { className: 'ia-chapterNum', 'aria-hidden': 'true' }, chapter.num),
            h('span', { className: 'ia-chapterName' }, t(chapter.navKey)),
          ),
        ),
      );
    }

    function StepRail({ chapter, step, t, onStep }) {
      return h(
        'nav',
        { className: 'ia-stepRail', 'aria-label': `${t('stage.step')} · ${t(chapter.navKey)}` },
        Array.from({ length: chapter.steps }, (_, index) =>
          h(
            'button',
            {
              key: `${chapter.id}-${index}`,
              type: 'button',
              className: `ia-stepItem${index === step ? ' is-active' : ''}`,
              'aria-current': index === step ? 'step' : undefined,
              onClick: () => onStep(index),
            },
            h('span', { className: 'ia-stepIndex' }, pad2(index + 1)),
            h(
              'span',
              { className: 'ia-stepLabel' },
              t(`chapters.${chapter.id}.steps.${index}.title`),
            ),
          ),
        ),
      );
    }

    function createAtlasPage(locale) {
      return function InferenceAtlasPage() {
        const [localeRevision, setLocaleRevision] = useState(0);
        const [reduceMotion, setReduceMotion] = useState(() => prefersReducedMotion());
        const [chapterIndex, setChapterIndex] = useState(chapterFromHash);
        const [step, setStep] = useState(0);
        const [selected, setSelected] = useState(null);

        useEffect(() => locale.subscribe(() => setLocaleRevision((value) => value + 1)), [locale]);

        useEffect(() => {
          let media = null;
          try {
            media = window.matchMedia('(prefers-reduced-motion: reduce)');
          } catch {
            media = null;
          }
          if (!media) return undefined;
          const onChange = (event) => {
            setReduceMotion(event.matches);
          };
          if (media.addEventListener) media.addEventListener('change', onChange);
          else media.addListener(onChange);
          return () => {
            if (media.removeEventListener) media.removeEventListener('change', onChange);
            else media.removeListener(onChange);
          };
        }, []);

        const t = useMemo(() => locale.bind(NS), [locale, localeRevision]);
        const chapter = CHAPTERS[chapterIndex];
        const stepCount = chapter.steps;
        const stepTitleKey = `chapters.${chapter.id}.steps.${step}.title`;

        useEffect(() => {
          const sync = () => {
            const next = chapterFromHash();
            setChapterIndex(next);
            setStep(0);
            setSelected(null);
          };
          window.addEventListener('hashchange', sync);
          window.addEventListener('popstate', sync);
          return () => {
            window.removeEventListener('hashchange', sync);
            window.removeEventListener('popstate', sync);
          };
        }, []);

        const selectChapter = (index) => {
          setChapterIndex(index);
          setStep(0);
          setSelected(null);
          try { history.pushState(null, '', '#' + CHAPTERS[index].id); } catch {}
        };

        const goToStep = (index) => {
          setStep(Math.max(0, Math.min(index, stepCount - 1)));
          setSelected(null);
        };

        const goRelative = (delta) => {
          setStep((current) => Math.max(0, Math.min(current + delta, stepCount - 1)));
          setSelected(null);
        };

        const handleSelectNode = useCallback(
          (picked) => {
            const key = `${chapter.id}:${picked.side}:${picked.id}`;
            setSelected((current) =>
              current && current.key === key
                ? null
                : { ...picked, key, chapter: chapter.id },
            );
          },
          [chapter.id],
        );

        const stepData = {
          sg: t(`chapters.${chapter.id}.steps.${step}.sg`),
          vl: t(`chapters.${chapter.id}.steps.${step}.vl`),
          diff: t(`chapters.${chapter.id}.steps.${step}.diff`),
        };

        return h(
          React.Fragment,
          null,
          h(
            'main',
            {
              className: 'ia-page',
              'data-reduced-motion': reduceMotion ? 'true' : 'false',
            },
            h('div', { className: 'ia-windowSafe', 'aria-hidden': 'true' }),
            h('div', { className: 'ia-topline', 'aria-hidden': 'true' }, h('i'), h('i')),
            h(
              'header',
              { className: 'ia-header' },
              h(
                'div',
                { className: 'ia-mast' },
                h(
                  'div',
                  { className: 'ia-kicker' },
                  h('span', { className: 'ia-kickerMark', 'aria-hidden': 'true' }),
                  h('span', null, t('header.kicker')),
                ),
                h(
                  'h1',
                  { className: 'ia-title' },
                  h('span', { className: 'ia-titleLine' }, t('header.titleA')),
                  h('span', { className: 'ia-titleLine ia-titleLine--second' }, t('header.titleB')),
                ),
                h('p', { className: 'ia-lede' }, t('header.lede')),
                h(
                  'p',
                  { className: 'ia-caution' },
                  h('span', { className: 'ia-cautionMark', 'aria-hidden': 'true' }),
                  t('header.caution'),
                ),
              ),
              h(
                'aside',
                { className: 'ia-releaseCol', 'aria-label': t('header.releaseLabel') },
                h(
                  'div',
                  { className: 'ia-releaseColHead' },
                  h('span', null, t('header.releaseLabel')),
                  h('span', { className: 'ia-releaseColRule', 'aria-hidden': 'true' }),
                ),
                h(ReleaseCard, { side: 'sg', t }),
                h(ReleaseCard, { side: 'vl', t }),
                h('p', { className: 'ia-releaseChecked' }, t('header.checked')),
              ),
            ),
            h(
              'nav',
              { className: 'ia-controls', 'aria-label': t('control.chapter') },
              h(ChapterTabs, { chapterIndex, t, onSelect: selectChapter }),
              h(
                'div',
                { className: 'ia-transport' },
                h(
                  'button',
                  {
                    type: 'button', className: 'ia-stepButton',
                    onClick: () => goRelative(-1), disabled: step === 0,
                    'aria-label': t('control.prev'),
                  },
                  h(StepBackIcon), h('span', null, t('control.prev')),
                ),
                h(
                  'button',
                  {
                    type: 'button', className: 'ia-stepButton',
                    onClick: () => goRelative(1), disabled: step === stepCount - 1,
                    'aria-label': t('control.next'),
                  },
                  h('span', null, t('control.next')), h(StepForwardIcon),
                ),
              ),
            ),
            h('p', { className: 'ia-controlHint' }, t('control.hint')),
            h(StepRail, { chapter, step, t, onStep: goToStep }),
            h(
              'p',
              { className: 'ia-srOnly', 'aria-live': 'polite' },
              `${t('stage.step')} ${step + 1} / ${stepCount}: ${t(stepTitleKey)}`,
            ),
            h(
              'section',
              { className: 'ia-stage', 'aria-labelledby': 'ia-stage-title' },
              h(
                'div',
                { className: 'ia-stageTop' },
                h(
                  'div',
                  { className: 'ia-stageHeading' },
                  h(
                    'p',
                    { className: 'ia-stageKicker' },
                    h('span', { className: 'ia-stageNum' }, chapter.num),
                    h('span', null, t(chapter.questionKey)),
                  ),
                  h('h2', { className: 'ia-stageTitle', id: 'ia-stage-title' }, t(stepTitleKey)),
                ),
                h(
                  'div',
                  { className: 'ia-stageMeta' },
                  h(
                    'span',
                    { className: 'ia-stepCounter' },
                    `${t('stage.step')} ${pad2(step + 1)} / ${pad2(stepCount)}`,
                  ),
                ),
              ),
              h(
                'div',
                { className: 'ia-boardGrid' },
                h(Scene, {
                  chapter: chapter.id,
                  step,
                  t,
                  selected,
                  onSelect: handleSelectNode,
                }),
              ),
              h(
                'p',
                { className: 'ia-boardHint' },
                h('span', { className: 'ia-hintRule', 'aria-hidden': 'true' }),
                t('board.hint'),
              ),
              h(ChapterLinks, { chapter, t }),
              selected
                ? h(
                    'aside',
                    {
                      className: 'ia-inspector',
                      style: {
                        '--ia-node-accent':
                          selected.side === 'sg' ? 'var(--ia-sg)' : 'var(--ia-vl)',
                      },
                      'aria-label': t('stage.nodeLabel'),
                    },
                    h(
                      'div',
                      { className: 'ia-inspectorHead' },
                      h(
                        'div',
                        { className: 'ia-inspectorTitle' },
                        h(
                          'span',
                          { className: 'ia-inspectorEngine' },
                          selected.side === 'sg' ? 'SGLang' : 'vLLM',
                        ),
                        h('strong', null, selected.label),
                        selected.tag
                          ? h('code', { className: 'ia-inspectorTag' }, selected.tag)
                          : null,
                      ),
                      h(
                        'button',
                        {
                          type: 'button',
                          className: 'ia-closeButton',
                          onClick: () => setSelected(null),
                          'aria-label': t('stage.close'),
                          title: t('stage.close'),
                        },
                        h(CloseIcon),
                      ),
                    ),
                    h('p', { className: 'ia-inspectorBody' }, t(selected.detailKey)),
                  )
                : null,
              h(
                'div',
                { className: 'ia-verdict' },
                h(
                  'div',
                  { className: 'ia-verdictCol ia-verdictCol--sg' },
                  h(
                    'div',
                    { className: 'ia-verdictHead' },
                    h('span', { className: 'ia-verdictDot', 'aria-hidden': 'true' }),
                    t('stage.sgNow'),
                  ),
                  h('p', { className: 'ia-verdictText' }, stepData.sg),
                ),
                h(
                  'div',
                  { className: 'ia-verdictCol ia-verdictCol--vl' },
                  h(
                    'div',
                    { className: 'ia-verdictHead' },
                    h('span', { className: 'ia-verdictDot', 'aria-hidden': 'true' }),
                    t('stage.vlNow'),
                  ),
                  h('p', { className: 'ia-verdictText' }, stepData.vl),
                ),
              ),
              h(
                'div',
                { className: 'ia-fork' },
                h('span', { className: 'ia-forkLabel' }, t('stage.divergence')),
                h('p', null, stepData.diff),
              ),
            ),
            h(TopicReading, { chapter, step, t }),
            h(ReadingDisclosure, {
              key: `sources:${chapter.id}`, t, className: 'ia-referenceDetails', title: t('reading.sources'),
            }, h(SourcesSection, { t, chapter })),
            h(
              'footer',
              { className: 'ia-footer' },
              h('p', null, t('footer.note')),
              h('p', { className: 'ia-footerMeta' }, t('footer.meta')),
            ),
          ),
        );
      };
    }





const atlasRoot = document.getElementById('atlas-root');
ReactDOM.createRoot(atlasRoot).render(React.createElement(createAtlasPage(atlasLocale)));
document.getElementById('lang-toggle')?.addEventListener('click', () => {
  atlasLocale.setLang(atlasLocale.lang === 'zh' ? 'en' : 'zh');
  document.getElementById('lang-toggle').textContent = atlasLocale.lang === 'zh' ? '中文 / EN' : 'EN / 中文';
});
document.documentElement.lang = atlasLocale.lang === 'zh' ? 'zh-CN' : 'en';
