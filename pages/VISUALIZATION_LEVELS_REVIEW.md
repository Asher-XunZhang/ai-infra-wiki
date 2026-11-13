# Pages 可视化细节层级复核

## 第二章 ModelRunner V2 调用链复核（2026-11-13）

第二章保留 SGLang `297e54fd105e974cdf401e5b4d20253a22c16f90` 与 vLLM `4ba787ec9959e566d3b587a796dff07673324da5` 两个固定版本。vLLM 路径明确为 **V1 引擎 + ModelRunner V2**；不更换第三章的 `279339f113`，也不把第三方文章的旧提交混入本章。

### 七个主题逐项处理

| 主题 | 源码确认与可见调整 | 分层与执行轨道取舍 |
| --- | --- | --- |
| 01 进程骨架 | 前端/EngineCore/Executor 边界保留；明确 Worker 选择 `gpu/model_runner.py` 的 V2；区分 `execute_model`、`sample_tokens`、CPU 可读输出。TP1 uni 与 TP2 mp 仍是不同进程布局 | 保留职责/请求两种观察方式与局部源码展开；进程包含关系无需另加时间轨道 |
| 02 通信面 | IPC 协议保留；展开 SchedulerOutput → RequestState → idx_mapping → 输入缓冲，区分计划、索引、token 数据与 GPU KV；结果须等 copy event 后进入 response MQ | 总览保留原通信边界；模块过程显示 V2 输入准备；源码入口分别指向 WorkerProc 与 V2，借调度主题解释并发，不复制播放器 |
| 03 前缀缓存 | 前缀查找仍归 KVCacheManager/BlockPool；V2 按请求槽位维护 BlockTables，再 gather / compute_slot_mappings。沿用 B17/B18 命中、B19 旧分支保留、B31 新尾部的对象身份；下一轮示例追加 B32 | 地址映射与数值变化更适合查表图，不添加甘特。总览保留命中与补算；模块过程展开请求槽位、block ID、KV slot 三层地址；源码再看 staged writes / UVA 与公式 |
| 04 调度心跳 | 预算仍由 Scheduler 决定；前向保存 ExecuteModelState 后采样；AsyncOutput 的 D2H 可与 GPU postprocess_sampled 重叠；async scheduling 下 mp 输出线程等待 event、Core batch queue 可继续派发 | 保留四个手动讲解阶段；模块图按 main/copy stream 分泳道，显式等待箭头表达同批次依赖。没有实测时长，不把条块画成定量甘特；脉冲沿用全页共享时钟，切细节不会另起播放 |
| 05 分离扩展 | V2 ActiveKVConnector.pre_forward 绑定 metadata；sync load 在前向前启动，其余 load 延至 post_forward；finished_recving 回 Scheduler，目标请求通过就绪与预算检查后才运行；零 token 轮仍可 no_forward | 模块过程区分控制箭头与 KV 载荷脉冲，展示两种加载分支的适用条件。同步加载“启动”不等于整批完成；按层等待由后端实现。无统一后端时序时不画假的跨机耗时条 |
| 06 生命周期与执行边界 | Scheduler block 引用回收与 V2 请求槽位清理分开。完成/抢占 ID 经 SchedulerOutput 消费；abort 不抢占已发 GPU kernel；connector 可 delay_free。图选择改为 V2 CudaGraphManager：FULL / PIECEWISE（compiled 或 breakable）/ NONE | 生命周期用分支依赖图防止把两种资源回收合并；图模式用互斥分支，捕获列为初始准备而非每轮行为。不把 breakable 宣称为 SGLang 独有 |
| 07 延迟口径 | V2 sampler → AsyncOutput event → Scheduler / EngineCoreOutputs → AsyncLLM；既有 TTFT/E2E/queue/ITL 端点仍成立，V2 不移动前端停表位置 | 保留事件区间与 ITL 样本图；源码细节补执行端到观测端的入口；不为 Runner 版本添加第二套计时器 |

**本地验证。** 构建、共享导航和学习资料链接检查通过；浏览器覆盖七主题源码入口、十个 V2 模块场景的中英双语/文字边界、总览与模块分离、地址状态回退、共享脉冲暂停、四档视口和主题。人工查看地址映射、采样双 stream、P/D 加载和图模式四幅对照图。原有归属/指标及六主题 24 阶段回归通过。测试创建的独立浏览器与 profile 已清理；预览服务保留供本地查看。

### 关键源码依据

均为公开固定提交的静态阅读；没有修改上游源码，也未执行 GPU、RDMA 或性能测试。以下是选定普通自回归、CUDA、非投机路径的依据，不构成所有模型/connector 的兼容矩阵。

- [Runner 选择](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/config/vllm.py#L731)与 [Worker 工厂](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu_worker.py#L570)：默认支持路径使用 V2；显式环境变量优先、HiSparse/watermark 强制分支及自动不支持回退须分别判断。
- [RequestState](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/states.py#L9)：请求槽位稳定于本地驻留期间；all_token_ids 默认可用 UVA，不能画成全部状态常驻显存。`prefill_len` 恢复时可含既有输出，不总等于 `prompt_len`。
- [V2 状态更新](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/model_runner.py#L1170)、[prepare_attn](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/model_runner.py#L1595)与 [slot mapping kernel](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/block_table.py#L334)：图中的 `block × block_size + offset` 限单 KV group、CP1；block_size=4 只为教学。
- [V2 execute_model](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/model_runner.py#L1764)与 [sample_tokens](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/model_runner.py#L2149)：先采样、建立 AsyncOutput，再 postprocess_sampled；[copy stream](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/async_utils.py#L116) 等之前的主流依赖；`get_output` 同步 copy_event 后读 CPU 数组，不把调用返回画成结果完成。
- [EngineCore batch queue](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/engine/core.py#L708)、[mp 输出线程](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/executor/multiproc_executor.py#L1096)：线程与多批派发条件明确为 async scheduling；V2 的 async copy 本身不等于所有部署都能跨批重叠。
- [ActiveKVConnector](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/kv_connector.py#L54)与 [Scheduler readiness](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py#L3520)：两种 start_load_kv 时机、结果回传与请求准入是不同事件；`finish_forward` 不是通用“传输已完成”屏障。
- [图候选选择](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/cudagraph_utils.py#L523)与 [执行分支](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/model_runner.py#L2053)：FULL 用已填输入缓冲 replay；PIECEWISE 可编译或 breakable；NONE 直接调模型，不强制 `FULL_AND_PIECEWISE`。

**实现范围。** `engine-comparison/runner-v2.js` 集中管理 V2 路径说明、模块图与固定源码入口；总览复用原图，源码层追加与当前主题有关的执行细节。其他章节和配套文章的独立历史基线保留。先前文章带来的进程归属、内存预算和指标补充继续有效，无需再造重复学习文章。


## 2026-11-13：第二章进程归属与指标口径补充

本次完整阅读 [《LLM推理引擎解剖：vLLM与SGLang的五层进程切分》](https://mp.weixin.qq.com/s/pbzBiFfBR-KpVBpxYGSk4Q)及其唯一技术图，并对照仓内[架构差异学习文档](../大模型推理/推理基础/SGLang%20与%20vLLM%20V1%20架构差异源码学习文档.md)、第二章原有六主题、第三章性能观测课。原文、配图和作者归属已经保留在[图源记录](../images/inference-process-latency/SOURCES.md)，不重复下载或新增架构摘要。

第二章新增第 07 主题“延迟口径”，其余增量落在现有第 01、03 主题。下方 2026-11-10 的六主题清单是原复核快照，不代表补充后的主题数量。

| 位置与学习问题 | 细节方案 | 执行轨道、连线和交互取舍 |
| --- | --- | --- |
| [01 进程骨架](engine-comparison/index.html?view=ownership&tp=2#process)：同样的职责分别住在哪个进程，谁拥有 KV 账本与 GPU 数据？ | 保留原请求路径；增加同阶段的“职责与 KV 归属”观察方式。它是观察问题选择，不伪称第三层细节。TP1/TP2 改变真实进程包含关系，源码深读复用现有入口 | 不加泳道：这里解释包含关系及资源状态，不模拟执行耗时。R1 的未分配、已预留、已写入状态可逆；KV 使用 rank-local 数据措辞，不预设均匀切片；同 rank 的写入在同一阶段出现。仅 Runner→GPU 的相邻写入使用连接端点的脉冲；跨进程交接作为命名说明，真实路径仍由原图表达。未画出的 collective 不等于不存在 |
| [03 前缀缓存选读](engine-comparison/index.html?depth=source#prefix)：为何两个显存比例参数不表示相同 KV 容量？ | 复用选读展开，加入静态份额、运行时预留、profile 与实际 KV 容量之间的关系；不增加播放器 | 不加泳道或比例条：定容公式和适用分支才是问题，不虚构显存测量。地址索引、资源分配与实际数据不混为一物 |
| [07 延迟口径](engine-comparison/index.html#metrics)：TTFT、E2E、queue、ITL 各测什么？ | 四个观察问题，分别显示双引擎的停表点、首次等待范围、可操作的合并输出算例；源码层补当前公式、字段和例外 | 复用手动阶段切换；不增加假想运行时钟或甘特：所需证据是同一请求的事件先后与采样端点，事件间距不表示耗时。就近说明前端输出处理可与核心后续执行交叠，不能把事件链读成 GPU 等前端。ITL 的 200 ms 是各自观察层的教学假设，不是共同 Trace |

**源码边界。** 所有新增实现解释按第二章 SGLang `297e54fd105e974cdf401e5b4d20253a22c16f90`、vLLM `4ba787ec9959e566d3b587a796dff07673324da5` 的原始文件重新核对；没有把文章的 `baa9bef` / `5df63a7` 或第三章 `279339f113` 混进第二章。未修改已有 Markdown 的独立基线。新来源入口见 [boundaries.js](engine-comparison/boundaries.js)，每条链接固定到对应提交及文件。

| 行为 | 本次读取的固定源码 |
| --- | --- |
| SGLang rank 内包含 Worker、缓存池与请求同步；一个独立 Detokenizer 的条件 | `scheduler.py::init_tp_model_worker/init_memory_pools`、`scheduler_components/request_receiver.py`、`entrypoints/engine.py::_launch_scheduler_processes/_launch_detokenizer_subprocesses` |
| vLLM 单卡 uni 与 TP2 本机 mp；Core 的 KV 配置与账本 | `config/parallel.py`、`v1/executor/uniproc_executor.py`、`v1/engine/core.py` |
| SGLang 静态预留与 vLLM 非 KV 扣减 | `mem_cache/kv_cache_configurator.py::_profile_available_bytes`；`v1/worker/gpu_worker.py::determine_available_memory` |
| SGLang 前端首末输出、非流式例外、首次/累计等待、ITL 分摊 | `managers/tokenizer_manager.py`、`observability/req_time_stats.py`、`observability/metrics_collector.py::observe_inter_token_latency` |
| vLLM 前端取批计时、首末输出、首次 schedule、ITL 与 TPOT | `v1/engine/async_llm.py::output_handler`、`v1/engine/output_processor.py`、`v1/metrics/stats.py::IterationStats`、`v1/metrics/loggers.py` |

**保留的条件与未采纳内容。** 进程数只在单节点、单 API、单 Detokenizer、PP=DP=1、Python 文本入口及指定执行器条件下说明，辅助进程不计入。指标主图限普通流式、非 PD、非投机、无重试/取消且启用指标；非流式 TTFT、记录时机、累计等待在相应源码层解释。原文的代码行数、架构优劣偏好、历史 33%/29%/38% 和未运行的性能预测不作为本章性能结论。显存图不把 Mamba、多模态、显式 KV 字节或图捕获条件简化成一个通用容量公式。

### 调用链复核后的修正

公众号的措辞不作为当前页面事实依据。二次复核从两个固定 commit 的原始文件追到调用方、数据生产方和分支条件，发现并修正下列教学假设：

| 需要收紧的说法 | 固定源码证据与最终呈现 |
| --- | --- |
| 把旧 `RadixCache` 类名当作默认实例 | [Scheduler 构造缓存](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/scheduler.py#L575)进入 `kv_cache_builder.build_kv_cache → registry.create_tree_cache`；[默认工厂](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/registry.py#L83)的普通 Full Attention、无外部后端路径返回 `UnifiedRadixCache`。旧 `radix_cache.py` 文件仍存在，不代表运行时默认构造它。同步纠正第 01/03 相关类名、生命周期说明中的引用名称和源码入口；图内小节点用变量名 `tree_cache`，不让长类名溢出。纯 SWA、外部后端等保留工厂分支边界 |
| 把两边都设为“普通非投机、一次核心/前端输出两个 token” | [SGLang output_streamer.accept](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/scheduler_components/output_streamer.py#L445)允许中途按 `stream_interval` 合并；首 token 立即回传、终态强制回传。vLLM [普通 _sample 分支](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu_model_runner.py#L3403)调用 [Sampler](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/sample/sampler.py#L208)，每请求的采样结果为单 token；[Scheduler](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py#L2512)创建对应核心输出。控制器因此只改变 SGLang 的合并，vLLM 保留单 token 输出。200 ms 分别是假设的各自观察区间，不能称同工作量对比 |
| 仅链接旧 Runner 的采样实现 | [VllmConfig.use_v2_model_runner](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/config/vllm.py#L730)在无强制覆盖、支持特性和平台条件下默认 V2。[V2 ModelRunner 普通分支](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/model_runner.py#L1681)进入 [V2 Sampler](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/sample/sampler.py#L217)，返回 `sampled.view(-1, 1)`；普通 ModelState 每步一个新采样 token。上行的 V1 调用链作为回退证据保留，页面源码入口改为 V2 并链接选择条件 |
| “每张 GPU 的 KV 都是均匀分片” | 两边模型配置均有至少一个 KV head 的下限及复制语义：[SGLang get_num_kv_heads](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/configs/model_config.py#L1700)、[vLLM get_num_kv_heads](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/config/model.py#L1663)。图中改为“本 rank 的 K/V”，不由进程拓扑推断所有模型的数据切片方式 |
| “最终核心输出→前端停表”适用于所有停止方式 | vLLM [OutputProcessor](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/engine/output_processor.py#L808)可在反分词中匹配 stop string，再向核心发 abort。本图进一步限定 EOS/长度终止、无 stop 字符串；不能让该图替代前端发现终止的返回控制路径 |
| 源码文件/函数名沿用印象 | 当前 SGLang 函数为 `Scheduler.init_tp_model_worker`；缓存构造位于 [`mem_cache/kv_cache_builder.py`](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/kv_cache_builder.py#L350)，其 pool/allocator 来自本进程的 TpModelWorker。已纠正审查表中的函数名，不用文章模块名字替代当前调用链 |

TTFT/E2E 的端点则通过生产/消费两侧确认：vLLM `AsyncLLM.output_handler` 先取核心批次并建 `IterationStats(time.time())`，`OutputProcessor.process_outputs` 再更新统计、反分词、输出；完成统计仍使用已保存的批次时间戳。SGLang 走 `OutputStreamer → DetokenizerManager → TokenizerManager`，在 TM 处理输出时保存 `first_token_time/finished_time`，`collect_metrics` 再将对应值写入 collector。首输出可能没有可见文本；内部 TTFT 不保证等于客户端首内容延迟。vLLM 内部核心事件为另一时钟域，不把前端绝对时间戳和核心时间戳直接相减。

**验证。** [浏览器回归](../tests/pages/test_engine_boundaries_browser.cjs)覆盖 TP1/TP2 包含关系与进程数、R1 状态前进/回退、脉冲暂停/恢复、细节切换、链接刷新/历史返回、四种指标端点、SGLang 1/2/4 token 合并与 vLLM 单 token 核心输出的 ITL sum/count、中英双语、五档宽度、主题及减少动态效果；并冒烟检查原有 24 个阶段。人工查看桌面归属图、TTFT/ITL 和窄屏画面；局部横向滑动沿用双引擎对照约定。仅做源码静态复核与前端验证，未启动推理服务、GPU 或压测。配套文章已形成完整的进程/内存/指标对照，无需再拆出重复学习文档；组件耗时排名仍需同条件实验。

本次判断哪些课程适合采用“总览／模块过程／源码细节”，以及怎样复用已有图、轨道与源码入口。目标是降低首次阅读的负担，同时保留实现细节和真实依赖。下文保留逐课取舍与版本边界；本地实施结果和验证范围见文末。

## 复核范围与依据

| 项目 | 本次范围 |
| --- | --- |
| 复核日期与页面快照 | 2026-11-10；仓库提交 `057f33ef955cd3e19cd7641ccecbbcfcb8d039f6`，本地构建版本 `fe46e6b445240fd2` |
| 课程清单 | [导航生成器](../scripts/build_learning_navigation.py) 的 `ENGINE_TOPICS` 与 `MODULES`：第二章 6 个主题；第三章 13 个模块、23 个页面。下文 `03.2` 表示模块 03 的第二篇，不沿用历史课程编号 |
| 第二章源码边界 | SGLang v0.5.23：`297e54fd105e974cdf401e5b4d20253a22c16f90`；vLLM v0.33.0（tag v0.33.0rc3）：`4ba787ec9959e566d3b587a796dff07673324da5` |
| 第三章源码边界 | Pages 实现分析固定为 `279339f113b79af84f27fd3ac92d0a13bd3f4cbd`。关联历史 Markdown 的独立基线不随本次评估改写 |
| 检查方法 | 阅读页面目标、控制器、模型及已有复核记录；实际打开 6 个主题与 23 个课程入口，查看 1440×1000 视口下的默认图、控件与折叠入口，并截取图所在视口。纯源码页检查正文及互链 |
| 证据边界 | 这是学习层级与表现方式评估，不是新一轮上游源码逐分支审计。未遍历全部条件、窄屏、主题和动画中间态，未运行 SGLang、GPU 或性能实验；实施时仍须核对受影响事件与固定源码 |

共用规范见 [Pages 交互图补充与复核规则](INTERACTION_GUIDELINES.md)。当前图已经按模块职责、数据和状态组织；主要增量是让初学者先看一条完整的因果主线，再打开同一时刻的内部过程。不是再制作一套独立动画，也不是把每张图都改成架构框图。

## 取舍概览

| 范围 | 建议 |
| --- | --- |
| 第二章 | 通信面、前缀缓存、调度心跳、分离扩展适合补完整三层；进程骨架和生命周期四个切面优先保留现有阶段图，以局部展开承接细节 |
| 第三章 | 10 页适合完整三层；并行课按子机制选择；8 页适合两层或局部展开；4 页保留已有分层或专题定位 |
| 优先处理 | Req 状态、通信、PD＋PP 生命周期、PD 数据流、PP 共识。这些图同时承担对象、字段、执行轨道和放行条件的解释，默认密度较高 |
| 可直接复用 | 性能课的同一时刻视图切换；现有主图作为模块过程；已有源码详情作为深读层；PP 综合案例的入门／依赖／源码入口 |
| 不应统一的内容 | 不统一各课的图形、轨道数量、模型拓扑或源码版本；不把机制选项、场景切换、追一批／看一轮当成细节层级 |

“两层／局部展开”指过程图本身已能完成总览，详细字段、表格或源码在原地展开，不另设三套画面。“保留”仍包含后续修改时按规则检查的责任，并非宣称所有交互已完整验收。

## 第二章：逐主题判断

六个主题位于同一个[架构对照页](engine-comparison/index.html)，由 [atlas.js](engine-comparison/atlas.js) 组织。已有每主题四阶段、双引擎并排、模块点击解释、选读对照表和固定源码折叠。阶段切换主要回答“走到哪里”，尚不等于切换细节程度；源码折叠应直接复用。

| 主题与路由 | 现有画面与判断 | 建议怎样展开 | 始终保留什么 |
| --- | --- | --- | --- |
| 01 进程骨架 · `#process` | 进程边界、入口、调度和执行已在两侧清楚排列。**两层／局部展开，后续按需** | 默认保留当前进程图与请求路径；点击进程再看内部对象及本阶段接口；现有源码折叠承接函数。无需再套一张只有“接入→计算→输出”的泛化总览 | Scheduler 与 Worker 是否同进程、两侧真实边界、回程载荷。两侧对齐的是问题，不是假设模块一一对应 |
| 02 通信面 · `#wire` | socket 模式、回程、共享内存与管理通道同时入图，初次阅读容易先陷入协议名。**完整三层，中优先** | 总览：谁向谁交接何种数据；模块过程：按当前阶段展开 IPC、Executor 交接或管理通道；源码：socket 角色、编码与 transport 分支，复用选读表 | 请求、回程、执行分发与控制面的方向；同进程调用不画成网络发送。不默认所有箭头都走网卡 |
| 03 前缀缓存 · `#prefix` | token、树路径、hash、BlockPool 与地址表同时出现。**完整三层，中优先** | 总览：同一输入哪些位置复用、哪些补算；模块过程：左侧 token 路径与索引，右侧块 hash 与 block table；源码：本版本匹配、对齐、分配和引用条件 | 两侧同一教学输入与命中范围。现例共同前缀 10 个位置但可复用 8 个的粒度条件不能折掉；索引查找和物理空间分配分区表达，不把树节点直接称为 KV 数据 |
| 04 调度心跳 · `#batch` | 候选、预算、batch、执行与 overlap 条同时可见。**完整三层，中优先** | 总览：本轮哪些请求进来、输出如何更新；模块过程：各自准入对象、预算和工作交接；源码：额度判断、结果队列与异步循环分支 | 排序不等于准入；两侧按各自固定实现组织。局部 overlap 不能画成整轮都同时进行，也不能据教学条长比较快慢 |
| 05 分离扩展 · `#disagg` | 选择实例、KV 传输、分层缓存、集群控制共用一张全景图。**完整三层，优先** | 总览：Router／前端、P、D 与两类交接；模块过程：当前四阶段对应的传输、缓存或调度主体按需显现；源码：connector、调度元数据和配置条件，复用版本链接 | 控制面与 KV 数据面分开；HiCache、存储和集群扩展不是每个请求必经串行阶段。各后端完成条件不能合成一种 ACK |
| 06 生命周期与执行边界 · `#crosscut` | 完成、压力、取消、图捕获已拆成四个独立问题，各图规模适中。**两层／局部展开，后续按需** | 保留当前切面图；按需展开锁／引用计数、回撤条件、取消收尾和 graph 资格。复用按切面切换的选读表及源码入口 | 这四个切面不是一次请求必经的四段时间。完成不等于驱逐、取消不等于立即停止设备、图捕获也不表示每轮重新 capture |

这里的源码层可以是同一事件旁的字段／条件面板，不要求第三幅全景图。两侧同时切换抽象层级，仍分别标明自己的 release；不能用第三章 `279339f113` 的细节直接替换第二章版本。

执行轨道按问题选择：调度和分离交接有必要解释并行与等待时加入；进程骨架、前缀索引和静态地址关系不因这次分层强加泳道。现有脉冲适合作为传递的视觉语言，未来有时间控制时须把其相位接入统一时钟，而不是直接复制 CSS 无限动画。

## 第三章：逐页面判断

### 第一层：建立全局

| 页面 | 现状与方案 | 总览／过程／细节的内容与边界 |
| --- | --- | --- |
| **01.1** [SGLang 代码架构：模块划分与协作全景](sglang/system-overview/index.html) | 开篇同时展开 P/D 的队列、Worker、Runner、缓存池及 PP；下方已有单实例职责与源码折叠。**完整三层，中优先** | 总览保留 P/D 两侧、请求接入、调度、计算、KV 交接和回程；模块过程复用当前嵌套框图；源码层接现有职责索引。保留 Prefill 的 PP4×CP4×Attention-TP4 和 Decode 逻辑接收端边界，模型内部可折叠，不能把所有 rank 画成一个 Scheduler。完整架构／请求进入等现有按钮是焦点过滤，不重复做成第二组层级 |
| **02.1** [请求生命周期：Req、TokenizerManager 与 Scheduler](sglang/request-runtime/index.html) | 三进程边界、请求与输出序列已较完整，细节按当前阶段回查。**两层／局部展开，中优先** | 默认围绕三个进程看一请求接入、计算、回告；按需展开 ScheduleBatch、执行封装和资源收尾，沿用当前状态与源码详情。保留 ReqState 与 Req 的本地身份、同一 rid 的关联及取消分支；不将整套队列字段搬入此课，深入链接 03.2。已有主图可承担模块层，无需重做全部路径 |

### 第二层：理解单实例

| 页面 | 现状与方案 | 总览／过程／细节的内容与边界 |
| --- | --- | --- |
| **03.1** [调度循环：Normal、Overlap 与 PP 的完整旅程](sglang/loop-journey/index.html) | 已有轨道、完整旅程／看一轮／追一批、对象与事件记录。**完整三层，中优先，PP 与 Overlap 先做** | 总览用同一批次展示准备、提交、执行、消费及等待；模块过程展开 CPU、设备流、结果可读及 PP 交接；源码层显示 future、结果项、slot 与具体依赖。Normal 已较简洁可保留当前轨道；颜色与批次身份、PP 的 stage-local loop、异步深度保持不变。复用现有焦点导航，不将观察范围再当细节层 |
| **03.2** [Req 状态与队列流转：请求为什么还没继续？](sglang/request-state/index.html) | 队列／batch、Req 字段、CPU 消费与 GPU 资源同时可见，信息密度较高。**完整三层，优先** | 总览也并列保留“谁引用／请求是否结束／资源由谁持有”，只减少字段；模块过程复用容器、消费区和资源图；源码层展开 `to_finish`、`finished_reason`、`is_retracted` 等当前变化及事件锚点。正常、分块、回撤、取消和 overlap 共用现有时钟。Q0 消费结果与队列 Q1 必须一直可区分，引用弱线与结果提交脉冲继续分开 |
| **03.3** [ScheduleBatch：Extend 与 Decode 怎样准备输入？](sglang/inference-overview/journey.html) | 两条请求、历史序列、本轮输入及输出去向已经围绕单个问题展开。**两层／局部展开，后续按需** | 保留历史→截取／展平→输入／写地址→下一轮的主图；就地展开长度数组、字段和准备函数。不能把 token 内容、KV 写地址与新采样结果合成一个“数据”方框；`future_map` 与 CPU `output_ids` 的用途差别仍可见。无需新增总览泳道 |
| **03.4** [Scheduler：选批、Chunked Prefill 与 Decode 更新](sglang/inference-overview/scheduling.html) | 三个容器、当前 batch 与后续轮次图较聚焦。**两层／局部展开，后续按需** | 保留谁被选入本轮及 chunk 续算主线，按当前动作展开选择分支、合并和过滤条件。更完整的预算／回撤交给 06.1，不在本页重复。回合框是教学轮次，不是三条可同时执行的流水线；本地 Req 引用不使用跨设备传输脉冲 |
| **04.1** [DeepSeek-V3：Embedding、Decoder Layer 与输出头](sglang/inference-overview/transformer.html) | PP 四级和首级代表层已经上下对应，后续层概括。**两层／局部展开，中优先** | 默认展示各级层区间及 h／residual 交接，点击代表层展开当前已有的 Attention、残差与前馈过程；函数锚点沿用阶段详情。只有首级 embedding、末级输出头，各层缓存本地持有；PP 交接激活不能变成传 KV。不再演示第一章的矩阵推导 |
| **04.2** [ModelRunner 与 CUDA Graph：输入怎样交给 GPU？](sglang/model-execution/index.html) | CPU/GPU 区域内同时显示请求、数组、执行缓冲区、缓存与权重。**完整三层，中优先** | 总览仍直接移动本轮输入、显示 GPU 执行和结果回程；模块过程复用现有地址／缓冲区图；源码层展开 ForwardBatch 字段、graph 资格和 replay 缓冲区条件。保留 Prefill CPU 输入与 Decode GPU 输入差异，以及提交不等于完成。计算／访存延伸保持选读，不用总览架构框替换数据过程 |
| **05.1** [RadixCache 与 Attention：前缀索引怎样连接 KV？](sglang/inference-overview/kv-cache.html) | 树、查询序列、命中结果与可驱逐集合已经分区；源码默认折叠。**两层／局部展开，后续按需** | 树搜索、分裂／插入、锁保护及叶子移除就是主线，默认继续可见；详细 value 索引、锁计数及函数局部展开。整段／搜索／插入／移除是机制范围，不能再强制叠加三套导航。不引入 KV 分页账本，不添加无源码依据的节点合并，也不用传输脉冲替代树结构变化 |
| **05.2** [KV Memory Pool：地址映射、分页与缓存回收](sglang/kv-memory/index.html) | 请求映射、分配器和一层 KV 的地址／内容已能在一图对应，源码与公式折叠。**两层／局部展开，后续按需** | 默认保留请求位置→槽号→本层 K/V 及资源归属；需要时展开页内偏移、分配算法、读写函数和回收条件。页与槽号是本课必要对象，不能为简化删掉映射。三种模式复用现有播放器；不把同 GPU 地址读写画成网卡传输，也不重复 Radix 搜索过程 |
| **06.1** [Batch 调度：准入预算、Chunked Prefill 与 Retract](sglang/scheduling/index.html) | 已按排序准入、分块、回撤分成三个机制，当前只突出一种约束和一本预算。**两层／局部展开，后续按需** | 默认继续看候选、当前约束、入选／等待结果；局部展开完整预算账、比较运算符和异常分支，复用源码详情。排序不保证准入、计划不等于物理分配、首条请求例外及回撤后恢复条件不能藏掉。预算图比新增多条执行轨道更直接 |

### 第三层：扩展到多卡多机

| 页面 | 现状与方案 | 总览／过程／细节的内容与边界 |
| --- | --- | --- |
| **07.1** [TP / PP / DP / EP / CP：SGLang 并行分工](sglang/parallelism/index.html) | 八种策略有专用画法及执行轨道。**按子机制分层，中优先** | 不增加一张包办八种策略的总览。PP、Attention DP、CP 更适合三层，TP／普通 DP 等可用现图加局部展开；具体取舍见下表。每种模式的模型、配置和进程组约束始终可见，不把普通 PP 的轨道直接替代 PD＋PP |
| **08.1** [IPC、Collective 与 KV Transfer：数据何时可用？](sglang/communication/index.html) | 六类机制包含软件对象、硬件通路、数据／控制线、等待及执行轨道。**完整三层，优先** | 总览先回答“谁交什么、何时可读”；模块过程展开该场景实际涉及的 CPU／GPU、网卡／交换机与提交／完成；源码层列 API、poll、计数和准入条件。复用每种机制自己的图及轨道；同机 IPC 不出现无关网卡。P/D 仍分准备、搬运、就绪、释放，而非统一完成标志 |
| **09.1** [PD Disaggregation：Bootstrap、KV 接收与准入](sglang/inference-overview/deployment.html) | 当前 P/D 两侧含 CPU/GPU、目标元数据与传输，底部六条执行轨道。**两层／局部展开，中优先** | 开篇保留预留→P 分块计算与传输→D 校验／PREBUILT→首次 Decode 的简洁主线，按需展开当前模块／硬件图和各阶段源码。P 收尾作为独立分支保留；D 可用与 P 可释放不是一个门槛。不在本导读再展开完整 PP 共识算法 |
| **09.2** [PD + PP：Prefill 请求生命周期与 KV 释放](sglang/pd-prefill-lifecycle/index.html) | 多请求、各级本地状态、三类执行主体和 12 阶段源码同时承载深度；已有参数折叠。**完整三层，优先** | 总览围绕选定 Req 展示各级计算、KV 交接及释放门槛；模块过程复用各级 CPU/GPU/worker 轨道与队列；源码层连接当前 12 阶段、chunk 范围、poll 和共识结果。总览阶段条包含等待时须说明；worker 传输与后续工作可重叠，不画成全级算完才统一发送。沿用请求、stage 和参数选择 |
| **09.3** [PD 数据流：激活、KV Transfer 与 Req 队列引用](sglang/pd-dataflow/index.html) | 主图、局部引用和多主体轨道已同钟，仍需同时读取多种对象。**完整三层，优先** | 总览保留激活／结果、KV 与控制消息的独立路径，以及各进程自己的 Req；模块过程展开 payload、队列引用和传输活动；源码层显示 proxy 字段、切片范围、metadata 和跳过条件。不能折叠成“一个 Req 在各进程之间移动”，不能把引用存在画成 KV 在途；纯中间块的条件继续可达 |
| **09.4** [PP 共识：Bootstrap、Release 与 HiCache ACK](sglang/pp-consensus/index.html) | 三机制各有参与者、名单／计数、传播和并行活动。**完整三层，优先** | 总览保留各级本地观察→传播／汇总→回流后本地决定；模块过程展开名单／计数逐跳变化和本级处理；源码层对应集合运算、ACK 前缀与本地复查。三个算法不能共用一个“全员 ready”布尔灯。已有独立 GPU／worker 活动在总览也保留并行证据 |

并行课需要在子机制内作出取舍，不能只给整个页面加一个总览按钮：

| 子机制 | 推荐层级与主线 | 展开时保留的关键差别 |
| --- | --- | --- |
| TP | 现有两 rank 局部计算→合并图作默认；局部展开形状、权重切片和调用 | down 的贡献相加不等于拼接；不隐藏结果怎样由局部贡献得到 |
| PP | 三层：两个 microbatch 跨级接力→CPU/GPU/通信轨道→slot、proxy、异步深度条件 | 同批依赖与跨批重叠分开；本例普通非 PD，`pp_async_batch_depth=0` 等配置不能因简化消失 |
| 普通 DP | 两层：请求分给独立副本→展开 Controller、每副本 Scheduler 及路由条件 | 副本 0 可在后续请求仍被派发时处理已有请求，无“全部派发完再开算”的共同门槛 |
| Attention DP | 三层：各自 Attention→共享 MLP→本地结果；展开 gather／MLP TP／取片及字段 | 传递激活，不搬整个 KV；共享 MLP 由这些 rank 执行，不新增中央 MLP 进程 |
| EP | 两层或局部展开：token→被选专家→归位；需要时展示路由表、权重与 dispatcher 调用 | 多专家贡献、rank 本地计算并行和恢复 token 顺序要可见，不能只剩三张阶段卡 |
| SP | 两层：本地 token 行与进入／退出布局转换；展开具体 collective 和 shape | 使用本页 Qwen3、EXTEND 与兼容性条件，不把该路径改称所有模型每层通用 SP |
| Prefill CP | 三层：本地 Q 与所需 K/V→汇合／缓存／Attention 轨道→zigzag 元数据与函数 | Q/K/V 的读取范围仍直观；出口另一次汇合是 hidden states，不能与 K/V 汇合混为一条线 |
| Decode CP | 三层：分散 KV 下形成完整输出→Q、局部 O/LSE 与合并过程→校正及 reduce-scatter 调用 | 必须显出局部输出怎样加权，不能画成简单平均；KV 仍分散，与 PCP 不是同一 forward 的前后两步 |

通信课按六种模式分别使用以下总览问题：IPC“对象怎样编码、接收方怎样构造本地 Req”；PP 张量“提交发送后何时能消费 proxy”；collective“各方什么时候得到可用结果”；KV 搬运“谁写目标内存”；Decode 就绪“数据与元数据还需满足哪些条件”；异常释放“仍在使用的资源何时可归还”。软件边界、硬件路径和放行条件不能只藏在源码层。

PP 共识的源码层分别承接 bootstrap 的 good 交集／bad 并集、release 的终态名单与本地再检查、HiCache ACK 的共同可处理数量／连续前缀及本地完成条件；总览仍需简短说明各自放行判据，不把三者当作相同 barrier。

### 第四层：理解功能与服务

| 页面 | 现状与方案 | 总览／过程／细节的内容与边界 |
| --- | --- | --- |
| **10.1** [Speculative Decoding 与约束生成：状态怎样提交？](sglang/advanced-generation/index.html) | 候选、验证窗口、GPU 接受结果、CPU 输出、KV 账目与轨道同屏。**完整三层，中优先** | 总览区分候选、被接受的结果与正式提交；模块过程复用 Draft／Target／CPU 及 grammar 的临时／正式状态；源码层展开接受索引、长度、KV 预留与提交分支。GPU 接受不等于 CPU 已提交；不能为简化把预留空间直接当成有效 KV，或把 grammar 临时遍历状态当正式状态 |
| **11.1** [服务治理：Health Check、Timeout 与 Cancellation](sglang/serving-operations/index.html) | Gateway、P/D 的 HTTP、Scheduler、设备与资源在一图出现，四种服务机制可选。**完整三层，中优先** | 总览先给当前问题的调用方、Gateway、P、D 与可见结果；模块过程展开并发派发、响应汇总、取消传播及设备收尾；源码层显示探活、超时、abort 和释放分支。保留健康响应≠请求准入、前端断连≠设备立即停止、两端退出责任分别判断；错误条件不从总览入口删除 |

### 第五层：综合验证

| 页面 | 现状与方案 | 保留与衔接方式 |
| --- | --- | --- |
| **12.1** [TTFT、ITL、TPOT 与 P95：SGLang 性能观测](sglang/performance-engineering/index.html) | 已有总览／模块过程／源码细节；总览突出关键阶段与一个指标，深层展示打点、公式和字段。**保留，作为参照** | 复用 `timeline.js` 的事件及 `performance.js` 的投影／层级选择。层级切换保持时刻和所选指标；不同采样场景的轨道数不必固定。计时端点、缺失样本、P/D 与客户端来源不因收起而改变；总览阶段条不解释为 GPU 忙碌区间 |
| **13.1** [Prefill Batch 的执行旅程：跨级计算与异步收尾](sglang/pd-prefill-pp-loop/quick.html) | **两层＋局部展开**。总览追一批，执行过程同时展开三个 Stage 的 CPU、GPU 与异步轨道；关联跨级交接两端，点击区间就地查看具体操作、对象、条件与固定源码 | 跟踪对象与细节层分开；同一时钟驱动全部 Stage、依赖状态和传递脉冲。局部迭代选择只作定位，不筛掉其他级，也不将本地 L 编号对齐成全局轮次。完整 CPU 分组图保留独立的槽位／迭代用途，与主图双向同步 Batch、rank、loop。主图从 M1 起步，PP0 L6 作为新 M3／旧 M1 案例 |
| **13.2** [Prefill 流水线的等待与依赖：空档从哪里来](sglang/pd-prefill-pp-loop/index.html) | **保留依赖分析定位，复用三个 Stage 的联合轨道**。默认可同时看各级 CPU、GPU、异步工作及观察操作的前驱；上游消息同时关联其计算与提交。直接依赖连线按需局部放大 | 前驱完成点与消费者起点分别呈现；静态依赖线不播放传输脉冲。区分 CPU 顺序、通信配对门控、流事件和整段 H2D 教学简化。A/B 使用相同绝对窗口，播放器覆盖两侧完整时间；不因 B 较短截断 A |
| **13.3** [Prefill PP 主循环源码：槽位复用与异步收尾](sglang/pd-prefill-pp-loop/notes.html) | **保留连续源码正文，不增加播放器**。进入时以同一模型和时刻列出三个 Stage 的只读状态、当前源码操作与前置条件；正文按入口、对象、槽位、迭代和收尾展开 | 与前两篇互链携带场景、Batch、rank、loop、事件、时间与速度，明确当前场景和独立 B；自定义场景按其参数重建。固定案例显式标明并固定 baseline，与返回当前观察分开。Bootstrap、Release、HiCache ACK 分别解释职责，再连接专门的共识课 |


## 传递脉冲怎样应用到这些页面

脉冲规则已经纳入 [交互图规范](INTERACTION_GUIDELINES.md)；既有语义与历史复核见 [Req 专题记录](MOTION_REVIEW.md)。实施时逐图判断连线承载的是数据、控制、引用还是依赖；只将真实交接改为持续脉冲，其他关系保留各自表达。

| 图的类型 | 适用方式 | 避免的误读 |
| --- | --- | --- |
| 第二章双引擎对照、第三章通信与 P/D | 完整定向路径，在对应传递活动中持续流动；标出激活、KV、metadata、rid 名单等载荷。不同通道按各自事件同时活动 | 不能全程所有线都动；socket 管理消息与 GPU KV 不混成一种搬运，等待 ACK 不等于 KV 还在传输 |
| Req、运行时、批次与 PD 队列引用 | 静态弱线标“引用”；数据交接脉冲从具体结果持有者连到消费者／被更新字段 | 容器指向同一 Req 不是传送 Req。CPU 消费项与仍排队项分开；不能因引用存在而持续“传数据” |
| ModelRunner、模型层与并行模式 | 适用于真实输入复制、激活／结果传递及 collective；保持参与者与本地数据变化 | 脉冲不替代算子执行、矩阵合并或专家贡献；提交调用和设备完成仍是不同事件 |
| Radix Tree、KV 地址、预算与静态源码 | 搜索焦点、树结构变动、地址映射、额度扣减使用各自原生表达；只有确有值交接的动作才考虑脉冲 | 不把查树、指针引用和预算减少全部画成网络数据流；不为统一风格添加不存在的传输主体 |
| 性能轨道与 PP 依赖时间轴 | 保留活动条和完成端点；需要展示载荷路径时才补脉冲，并复用同一时钟 | 不在每条时间条上叠加装饰运动；脉冲密度和阶段长度不当作流量、带宽或 GPU 利用率 |

## 原定实施次序

1. **先做一条完整实现链：** 03.2 Req 状态 → 08.1 通信 → 09.2／09.3 P/D 两篇 → 09.4 PP 共识。每页先定义最小总览，现有图转为模块过程，已有源码说明就近展开。先在同一中间时刻验证身份、依赖、指标和播放状态不变，再推广共享逻辑。
2. **再做入口与复杂机制：** 01.1 系统全景、03.1 循环、04.2 ModelRunner、07.1 指定并行子机制、10.1 生成、11.1 服务；第二章先处理分离扩展，再处理通信、前缀缓存和调度。复用已有专用画法，不把不同机制强制塞进同一种模板。
3. **聚焦课按修改机会小步处理：** 其余两层／局部展开页面，优先调整默认可见字段和就近详情，不安排无差别重画。12.1 与 13.* 保留已成熟的分工，改动时检查深链接衔接即可。

## 本地实施结果（2026-11-10）

本轮按上述取舍覆盖第二章 6 个主题、第三章 23 篇课程。第三章复用既有机制模型与事件时钟；未改模型状态迁移、事件依赖或源码版本。这里的“实现”包括按建议保留成熟专题，不表示每页都新增三个按钮。

| 范围 | 已落地的呈现 |
| --- | --- |
| 第二章 `wire / prefix / batch / disagg` | 总览只展开当前阶段的参与者、输入或活跃交接；模块过程保留原双引擎图；源码细节展开对应对照表及固定版本入口。前缀例保留 10 个相同位置但仅复用 8 个的条件。分离、缓存层次与集群扩展继续作为不同剖面 |
| 第二章 `process / crosscut` | 保留当前过程图与模块点击；增加过程／源码细节切换，不另造泛化总览。切换层级保留主题、阶段与节点选择；脉冲可暂停，两侧共用相位时钟，未把剖面编造成实测执行时间 |
| 第三章 01.1、03.1、03.2、04.2、08.1、09.2–09.4、10.1、11.1 | 完整三层。专用总览分别保留 PP4×CP4×Attention-TP4、批次身份与 CPU/GPU 分离、Q0/Q1 区别、输入/图桶/真实输出行、硬件路径与可读条件、各级 KV 进度与本地 Req、三种共识、GPU 接受与 CPU 提交、P/D 独立治理 |
| 第三章 07.1 | PP、Attention DP、PCP、DCP 使用三层；TP、普通 DP、EP、SP 使用过程／源码细节。沿用原配置、局部数组与各模式执行轨道；不统一它们的算法、时刻或拓扑 |
| 第三章 02.1、03.3、03.4、04.1、05.1、05.2、06.1、09.1 | 两层或局部展开。保留三进程身份、输入展平/槽号、选批、代表层、树结构与锁、物理地址、预算等主图；细节层打开本阶段字段及源码。Radix 的 value 数组按需展开，树结构与锁始终可见。PD 导读先看双侧状态和轨道，再展开硬件与模块 |
| 第三章 12.1、13.1–13.3 | 保留性能课现有三层，以及 PP 综合案例的入门／依赖／源码分工；复核入口可加载，不添加重复播放器 |

### 状态、时钟与连线

[共享层级控件](learning-depth.js)只管理视图、可见区域和 `depth` 深链接。[课程投影](sglang/depth-projections.js)读取各课原始快照；每页仍由自己的播放器推进。暂停、任意拖动、反向回看、展开细节与收起共用同一时刻；层级切换不重新建立模型，也不改变场景选择。原源码说明和固定提交链接继续可达。

[第二章 React 页面](engine-comparison/atlas.js)保留四阶段剖面，由单独的视觉相位驱动活跃路径脉冲；相位不提交课程状态、不冒充算法执行时间。切换阅读深度不重启相位，暂停与减少动态效果设置均受控。

真实交接沿完整定向路径持续脉冲，位置由既有时钟计算；暂停时相位停止。Req 本地引用、树形变化、局部行重排、预算和活动条继续使用各自表达。数据到达、读取依赖、请求提交与资源释放仍在原事件边界确认；动画脉冲本身不能使结果提前可读。

### 验证记录与边界

- 浏览器回归：[test_learning_depth_browser.cjs](../tests/pages/test_learning_depth_browser.cjs)，覆盖第三章 42 个页面/模式组合、第二章 6 个主题，以及 4 个保留入口；检查默认视图、三个中间时刻、切层保留暂停/播放状态、深链接时刻恢复、390 px 窄屏、减少动态效果及脚本/资源错误。图像检查另核对代表总览、源码层及深色主题。
- 既有页面机制测试通过，覆盖请求、循环、KV、调度、并行、通信、PD、共识、生成、服务与性能等模型。共享源码基线检查确认 271 个静态 SGLang 链接及动态基线声明仍为 `279339f113`。
- 对固定 Git 对象运行现有循环、请求、并行、通信、PD 导读、PD 执行、PP 共识、生成源码检查；未修改被分析的 SGLang 工作区。
- Pages 构建、共享导航与学习文档链接检查通过。本节记录部署前的本地验收；没有运行 SGLang 服务、GPU/RDMA 或性能实验。浏览器矩阵不表示遍历了全部故障条件及参数组合。

既有专题已经分别承接输入、执行、资源、通信、共识与性能；本轮不新增拼接式整合文档，主要增量是相同时刻的分层阅读与可追溯状态。

## 第 13 节专项复核（2026-11-11，本地预览）

- **学习问题与范围：** PD 分离下 Prefill PP 如何交错推进新 Batch、旧结果和本地收尾。三篇分别讲执行旅程、等待依赖和主循环源码；保留原 PP=3、depth=0、固定准入／选批和成功路径，Decode 仅为交接边界。
- **轨道选择：** 前两篇复用同一依赖模型，增加连续教学时钟；第三篇保留静态源码阅读。整段阶段范围不解释为持续占用 GPU，L# 为 rank 本地槽位迭代，不是跨级同步屏障。
- **连线与脉冲：** 激活、output、KV 与 host-hit 回载按各自活动区间呈现，暂停／回拖由同一时间决定相位。引用、等待和依赖不冒充搬运；依赖分析图保留静态前驱箭头。
- **源码边界：** 重新读取公开固定 Git 对象 `279339f113b79af84f27fd3ac92d0a13bd3f4cbd`；外部检出工作区有本地改动，未使用这些改动、未修改检出目录。核对 graph 与 step guide 锚点，以及 PP 当前前向／旧结果、copy_stream、逐层加载与 release 分支。原模型未改变排程，仅将笼统的 D2H 标签纠正为结果预处理／就绪事件。全部 H2D 门控、同类传输串行和部分通信配对门控继续明确标为教学假设。
- **已验证：** 十组场景的依赖、交接与清理边界；连续全程播放、暂停、逆向拖动、层级切换、主图／完整图同步、跨三篇及刷新恢复、场景重置；桌面与窄屏、深浅主题和减少动态效果。首屏、传递中与执行轨道分别人工看图。
- **未覆盖：** 未运行 SGLang GPU／RDMA 推理，未测量性能，也未模拟动态选批、容量失败或后端完整流调度。本轮仅本地改造，未部署。

## 第 13.1 篇阅读层级收敛（2026-11-13，本地预览）

- 保留「总览／执行过程」两层；原独立源码层与执行过程共用主图、缺少独立学习价值，改为轨道后的局部展开。第三篇继续承接完整主循环与槽位复用。
- 相邻 CPU 服务操作按五类处理阶段归组，不跨本地迭代、不跨空闲间隔，也不吞并等待；GPU、回载、结果与 KV 的起止时间不变。执行过程按同一绝对时钟分段放大，等待斜纹及具体原因可点击检查。
- 源码展开复用已核对的 step guide，显示当前操作对象、本步处理、后续交接及前置条件。展开、收起、选择细小操作均不修改跟踪 Batch、教学时刻或播放状态；旧 `level=source` 链接映射为执行过程加展开，保留事件与进度。
- 本轮只调整呈现；固定源码、依赖排程和条件假设沿用 `279339f113`。未执行 GPU/RDMA 或性能实验，未部署。
- 验证：十组场景归组前后逐事件一致，等待不被吞并；浏览器核对连续播放、局部源码展开／收起不改变观察状态、等待条件前后及逆向恢复、旧链接与跨页恢复、键盘操作，以及 320／390／1440 px 源码面板。播放期间原生操作选择器保持挂载。

### 跨 Stage 的执行关系（2026-11-13，本地预览）

- **学习问题：** 一个 Stage 等待旧结果时，其他 Stage 的 CPU、GPU 和异步任务是否仍在推进；交接的生产者、载荷与消费条件分别是什么。执行过程始终显示 PP0／PP1／PP2 的 CPU 和 GPU，共用绝对时间窗口；回载、结果就绪与本级 KV 轨道仅在该窗口有活动时出现。
- **联合观察：** 本地迭代选择保留为折叠参考，不改变各级的时间窗口或可见性。CPU 归组与独立等待保留；跨级关系同时标出轨道两端，可查看任一 Stage 的源码而不切走其他级。
- **交接边界：** 关系取自同一个依赖图的真实边，包括当前等待、在途消息与所跟踪 Batch 的后续载荷。激活和 output 只有在消息区间内出现脉冲；控制／Work 完成依赖保持静态。生产者完成、消息到达、消费者其他条件已齐分别判断；保留模型假设，不把教学等待长度解释为真实 CPU 阻塞时长。
- **验证：** 十组场景逐项检查跨级等待及其原始依赖；浏览器核对全部 Stage 同时可见、迭代定位不筛选 Stage、跨级源码展开、传输前／中／后与反向拖动、暂停时脉冲冻结，以及桌面和窄屏布局。本轮未修改固定源码、排程或依赖边，未部署。


### 第 13.2／13.3 篇关联复核（2026-11-13，本地预览）

- **依赖页的缺口：** 直接前驱图只显示输入条件，省略了消息的 GPU 生产者和其他 Stage 的并行工作。现在主图复用三个 Stage 的同窗 CPU／GPU／异步轨道，关联观察操作、直接前驱和消息的生产者；直接依赖连线保留为局部放大。点击轨道与前驱均保留时刻，多个本地 L 不被误当作同步轮次。
- **A/B 时钟：** 两侧使用同一绝对窗口与最长完整时间。短场景不截断原示例；异步读取对照数据时保留 URL 的待恢复时刻，数据齐备后再核对合法范围。
- **源码页的缺口：** 旧说明仍将完整 loop 图当作默认主图，且缺少与当前时刻的直接对应。正文已说明总览、联合执行过程和局部源码的分工；增加只读的三个 Stage 状态摘要，不安装模型播放器、不推进时间。独立对照 B、自定义参数与固定 baseline 案例分别处理，返回当前观察与跳转固定例子明确区分。
- **保留：** 第三篇的槽位表、A–I 调用顺序、三条协调链、固定源码及模型边界继续承接深读；完整依赖图保留缩放、逐项源码追溯的独立用途，不机械增加第三个细节层。
- **验证：** 十组场景、固定源码锚点、全程播放与回看、前驱追溯、A/B 同窗和长短时域、跨三篇刷新恢复、只读状态及自定义场景、桌面与窄屏布局。本轮未改变源码基线或求解依赖，未执行 GPU／RDMA 实验，未部署。
