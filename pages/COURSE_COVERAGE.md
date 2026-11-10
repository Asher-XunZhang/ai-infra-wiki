# Pages 学习课程覆盖与维护

Pages 分为“推理基础”“SGLang 与 vLLM 架构对照”和“SGLang 实现剖析”三章。第一章用十三课建立通用机制；第二章按固定 release 对照六主题、二十四步，并提供第三章深入入口；第三章保留五层十三模块、二十三节课程与固定的源码依据。本文件记录课程覆盖与验证方式。资料归档仍使用仓库的稳定技术领域，课程顺序由导航维护，不移动旧源码课程的阶段目录。

## 三章之间怎样衔接

第一章的 [十三节基础课](foundations/README.md) 依次解释文字与 token、逐步生成、Attention、Transformer 基础部件、KV、批处理、GPU、多卡分工、通信、角色分离、候选验证、服务生命周期和性能口径。每课使用具体例子，提供分步播放、回退、条件切换、自测、来源和无 JavaScript 正文。先修链接与上一课 / 下一课让读者按依赖继续。

第二章的 [架构对照](engine-comparison/index.html) 以 SGLang v0.5.23（commit 297e54f）与 vLLM v0.33.0（tag v0.33.0rc3，commit 4ba787e）为固定基线，按进程、通信、前缀缓存、调度、分离扩展和生命周期六个主题各四步播放；每个主题给出对应的固定 release 源码入口与第三章深入课程。各课基线彼此独立，本页对照不替代各课自己的固定版本。

第三章保留旧网址和固定源码依据，课前只给实现目标与第一章的先修链接，不再嵌入同一套基础播放器。五节旧概览课分别改为批次准备、DeepSeek-V3 调用、前缀索引、调度入口和 PD 交接的源码导读；原有六幅通用机制图及说明归到第一章相应课程的选读区，避免在两章重复学习。

第三章的其他交互保留具体实现问题：请求对象交接、KV 地址与共享保护、准入与回撤、Graph 资格、通信接口与释放、状态回退、服务治理及统计口径。机制上的必要回顾服务于这些实现问题，不另复制第一章课件。

基础课的编辑入口是 `pages/foundations/courses.json`，选读机制图保存在 `extensions.json`；由 `scripts/build_foundations.py` 生成十三页；共享侧栏和课程映射由 `scripts/build_learning_navigation.py` 生成。两类生成一致性均由 `build_pages.py` 检查。

### 通信语义覆盖

[通信基础](foundations/communication.html) 提供 17 个基础语义入口和 1 个“矩阵对照”入口，共 33 个可选场景（29 个基础语义 + 4 个矩阵对照），按分发与收集、归约与扫描、交换与点对点、单边访问、同步与完成分组。包含 Broadcast、Scatter(v)、Gather(v)、AllGather(v)、Reduce、AllReduce、ReduceScatter、Scan / Exscan、AllToAll(v)、邻域交换、Send/Recv、Put / Get、Accumulate / CAS、Barrier 和完成依赖。保留不同图形：固定 rank 间的数据分发与交换、数值归约、端点窗口与事件泳道。NCCL 2.32.3 / MPI 4.1 语义依据与画法参考在课程内列出；不是全部库 API 或底层算法的模拟。

- `node tests/pages/test_communication_teaching.cjs`：输入输出、root、变长片段、rank 顺序、归约、前缀、配对、单边方向与原子更新。
- `node tests/pages/test_communication_teaching_browser.cjs`：29 场景的中间态与响应式排版、稳定对象与回看、完成边界、播放与减少动态效果、无 JavaScript 阅读、共享选择器回归。

### 基础章节验证

- `node tests/pages/test_foundations.cjs`：真实分词固定样例、因果权重、KV 所属与释放、批次成员守恒、通信与 PP 依赖、统计样本边界。
- `node tests/pages/test_foundations_browser.cjs`：全部场景与步骤、320 / 390 / 1440 宽度、深浅主题、自测反馈、键盘、播放暂停、减少动态效果、链接恢复与无 JavaScript 正文。
- `node tests/pages/test_topic_navigation_browser.cjs`：三章目录、第二章六主题入口、二十二节实现课导航、旧路线 URL、跨模块翻页与移动端侧栏。

发布状态以对应提交的 Pages 工作流和线上验证为准。浏览器中的教学模型不代表运行过 GPU 推理。

## 第 03 模块：循环与依赖验证

- `node tests/pages/test_loop_journey.cjs`：依赖门槛、确定性回看、EOS 后有效输出不增加、未就绪不越过屏障、PP 两级完整收尾；90 个轨道活动的单一时钟、生产者完成前提与独占轨道无非法重叠。
- `SGLANG_SOURCE_DIR=<checkout> python3 -B tests/pages/test_loop_journey_source.py`：从固定 Git 对象执行普通 / Overlap 的 CPU 调用顺序，并核对源码锚点；复用并行课的 PP 调用顺序测试。不是 GPU 实测。
- `node tests/pages/test_loop_journey_browser.cjs`：三机制连续播放、暂停、小数时刻拖动、倍速、节点步进、等待与解除、模式和历史、PP 聚焦、同一时钟轨道及依赖点选、键盘、窄屏、主题、无 JS；截图另作人工检查。
- 旧课程保留独立用途：输入组织、选批入口、设备分层、KV 生命周期、资源预算；不把它们重复堆进完整旅程页。

## 第二章架构对照的验证

- `node tests/pages/test_engine_comparison_browser.cjs`：六主题二十四步、节点注释打开与关闭、手动前后步进及首尾边界、停留不自动切换、阶段内连线与节点特效持续运行、hash 直达与刷新历史、中文与英文切换、主题切换、减少动态效果关闭循环特效、1440 / 1280 / 1100 / 1024 / 900 / 768 / 390 / 320 双引擎固定横排、窄屏对照区键盘横向滚动、虚线边界对比度、核心节点角色说明及默认完整图表、主动放大与恢复全图、键盘滚动、中英文 48 个图示状态的文字边界与遮挡、按主题及步骤懒加载的选读区（默认无大表格，展开保留当前阶段，切换收起）、本节来源筛选、有序列表，以及 prefix 双方 8/12 命中。
- 第二章基线固定为 SGLang `297e54fd105e974cdf401e5b4d20253a22c16f90` 与 vLLM `4ba787ec9959e566d3b587a796dff07673324da5`；测试只访问本地预览，不下载、不改写上游源码。
- 页面与共享导航由 `python3 -B scripts/build_pages.py` 构建，生成一致性由同一脚本和 `build_learning_navigation.py --check` 覆盖。

## 连续机制演示审核

2026-10-15 按阅读者需要理解的因果关系复核。24 个页面共 37 个可播放条件、250 个关键状态；基础 token、生成、Attention、KV、角色分离，以及 SGLang 执行、KV 内存、调度、PD 引用与收尾十页采用专用图形。连续批处理保留 token 序列与固定执行位置。八页仍以矩阵、序列或时间轴为主，将通用演示折叠；其余五页保留对象过程图。原参数实验、正文和源码依据保留。八节源码导读 / PP 课程保持原用途。逐课问题、调整理由、保留理由和复核步骤见 [Pages 机制可视化审核](MOTION_REVIEW.md)。

## 每个模块的完成要求

每个模块都需要：明确先修和学习目标；可长期阅读的仓内资料入口；逐行为核对的 SGLang 固定源码基线；解释机制的静态、动态或交互图；图旁解释与自测；正确的模块导航；桌面与手机阅读验证。源码事实、教学假设、未做的运行验证必须分别说明。只有引用一篇相关案例，不能算独立入门模块已经补齐。

图的类型由机制决定，不统一套用一种形式。页面内容不描述使用者的水平。可视化必须直接呈现机制：用泳道、时序箭头、拓扑位置、数据块或资源槽位表达控制流、数据流和状态变化。交互步骤应改变图中的路径、位置、占用或依赖，而非只替换文字卡片、表格和字段值。默认只呈现当前机制所需的元素，完整流程与额外细节按需展开；避免元素堆积。详细说明与实现字段放在图旁的简短提示或展开区；验收要实际观察关键步骤及异常分支，确认不读长文也能辨认当前谁在做什么、数据怎样变化、为什么要这么做，以及前后步骤的依赖。

## 覆盖与补充顺序

这张表是课程补充工作的范围，不把“页面存在”当作全部完成。2026-09-23 开始逐模块复核与补充；其他专题继续保留原有基线。

| 模块 | 已有入口与可视化 | 当前范围与深入方向 |
| --- | --- | --- |
| 01 推理系统全景 | [代码架构](sglang/system-overview/index.html)，一张全局图连接 API、分词、调度、执行、KV、模型与输出；附 TP / DP / PD 拓扑 | 已补系统边界与其余十二模块的按问题衔接；详细启动和运行时变体按独立源码课程展开 |
| 02 请求生命周期与运行时架构 | [普通请求运行时](sglang/request-runtime/index.html)，当前步骤交接图、可展开的完整流程与四路径资源变化 | 已补独立入门课；深入分支连接原有源码章节 |
| 03 调度循环与批次执行旅程 | [完整旅程](sglang/loop-journey/index.html)、[输入准备](sglang/inference-overview/journey.html)、[选批与更新](sglang/inference-overview/scheduling.html)；Normal、Overlap、普通 PP 的事件泳道、结果与资源状态；[Req 状态与队列流转](sglang/request-state/index.html) 单独区分逻辑状态、容器引用与资源归属 | 固定 279339f113；CPU / GPU 依赖、EOS 排空、slot 与输出回环。普通 PP=2/TP=1/depth=0；PD 与 HiCache 在第 13 模块深入 |
| 04 模型执行与设备协作 | [Transformer](sglang/inference-overview/transformer.html) 与 [执行机制](sglang/model-execution/index.html)，执行交接、图重放补齐与成本下界 | 已补 Worker / Runner / backend / kernel 与硬件直觉；复杂模型、kernel 和执行模式连接独立基线源码课程 |
| 05 KV Cache 与资源生命周期 | [KV 原理](sglang/inference-overview/kv-cache.html) 与 [映射和回收](sglang/kv-memory/index.html)，尾页续写、整页命中、共享保护与驱逐 | 已补普通分页路径与固定源码；分层存储、混合状态和容量规划连接独立基线资料 |
| 06 调度策略、批处理与资源约束 | [准入和回撤](sglang/scheduling/index.html)，排序、单项预算、chunk 份额与恢复路径 | 在第 03 模块的循环模型之后解释资源约束；保留不同策略与参数实验 |
| 07 并行与执行拓扑 | [并行分工](sglang/parallelism/index.html)，TP 算例、PP 时间格、DP 路由、EP 任务和 CP 因果矩阵 | 已补独立课程与固定源码锚点；复杂组合和 backend 约束连接深入章节 |
| 08 通信与传输 | [通信机制](sglang/communication/index.html)，消息 / 张量交接、三种 collective、KV 页映射、就绪与失败回收 | 已补独立课程和固定源码锚点；硬件传输实现与真实性能实验保留为深入方向 |
| 09 分离部署与分布式状态交接 | [部署入门](sglang/inference-overview/deployment.html)、[PD 旅程](sglang/pd-prefill-lifecycle/index.html)、[数据流](sglang/pd-dataflow/index.html) | 已补 P 收尾 / D 就绪的两侧边界、通信 / 缓存 / 准入连接；Pages 主线统一 279339f113；各案例分别说明配置与假设 |
| 10 模型结构与高级生成 | [状态与生成](sglang/advanced-generation/index.html)，EAGLE V2 状态结算、Grammar 编译准入、候选树试走 / 回退与正式提交 | 三种实现视角、10 个条件与同一时钟执行轨道；模型状态差异保留补充，其他算法按独立基线衔接 |
| 11 服务部署与运行治理 | [服务治理](sglang/serving-operations/index.html)，P/D 两端探活、成对 HTTP 派发、取消传播、KV 留置与单端退出；14 个条件共享主图与执行轨道 | 已补独立课程；Gateway 额度、协议差异与管理变更按问题衔接源码专题 |
| 12 性能分析与优化方法 | [指标与定位](sglang/performance-engineering/index.html)，P/D 模块执行轨道、可选指标量尺、流式采样与成功集合 | 固定 279339f113；区分客户端 / P / D 计时端点、异步交接、末块 KV 回退及统计归属；Profiler 与真实压测通过深入资料衔接 |
| 13 综合案例与源码实践 | [PP loop](sglang/pd-prefill-pp-loop/quick.html)、[依赖分析](sglang/pd-prefill-pp-loop/index.html)、[源码阅读](sglang/pd-prefill-pp-loop/notes.html) | 已补计时边界、共享 KV 与 PD＋PP 三条跨模块练习；每条含预测、图上操作、固定源码与运行验证边界 |

十三模块的主线入口现已补齐；深入资料继续按独立版本与主题维护。下列验证覆盖教学模型与页面，不能替代真实模型、设备或性能实验。

## 运行时课程的验证

- `node tests/pages/test_request_runtime.cjs`：检查输出 / KV 位置关系、取消回告与资源释放的先后关系、未派发失败没有后端分配等教学不变量。
- `SGLANG_SOURCE_DIR=<source-checkout> node tests/pages/test_request_runtime.cjs`：额外读取固定 Git 对象，校验每个交互源码锚点的行号与函数名。不会修改源码仓库。
- `node tests/pages/test_request_runtime_browser.cjs`：验证精简视图 / 完整流程、消息逐跳运动、减少动态效果、四条路径、参数、播放 / 暂停 / 重置、键盘、深浅主题、320 / 390 / 768 / 1440 布局与无 JS 阅读。
- `node tests/pages/test_topic_navigation_browser.cjs`：首页与全部课程的共享导航和跨模块入口。
- `python3 -B scripts/check_learning_docs.py`、`python3 -B scripts/build_learning_navigation.py --check`、`python3 -B scripts/build_pages.py`：文档引用、共享导航与发布构建。

浏览器测试复用现有 `PLAYWRIGHT_MODULE`、`CHROMIUM_EXECUTABLE`、`OVERVIEW_TEST_URL` 环境变量，默认站点为 `http://127.0.0.1:8765/`。可先以 `python3 -m http.server 8765 --directory .pages-dist` 预览构建输出。网页与源码模型验证不替代 SGLang 实际运行或性能实验。

## Req 状态专题与 Pages 基线的验证

专题放在模块 03 的完整循环旅程之后、批次输入准备之前：先看 event loop，再追踪同一个 Req 的归属。请求运行时、调度准入、PD 数据流、高级生成与服务治理页面保留各自问题，通过交叉入口连接状态专题。分支队列在专题中索引，不重复 PD 和 Grammar 的专属播放器。

- `node tests/pages/test_request_state.cjs`：六条路径的引用、资源、完成条件、依赖顺序与可逆状态；设置 `SGLANG_SOURCE_DIR` 另校验 31 个固定源码锚点。
- `SGLANG_SOURCE_DIR=<source-checkout> python3 -B tests/pages/test_request_state_source.py`：直接读取固定 Git 对象，核对浅快照、结果消费、chunk 续算、回撤和取消分支。
- `node tests/pages/test_request_state_browser.cjs`：六条路径完整播放、暂停、回拖、键盘、深浅主题、五种宽度与无 JS 阅读。
- `python3 -B tests/pages/test_sglang_page_baselines.py`：第三章 Pages 的固定源码链接、动态模型与设计说明均指向 `279339f113`。历史 Markdown 文档不在迁移范围内。

## 并行课程的验证

- `node tests/pages/test_parallelism.cjs`：TP 数值、PP 前后级依赖、DP 单一归属、EP 任务守恒、CP 查询覆盖与因果配对数；设置 `SGLANG_SOURCE_DIR` 可核对 11 个固定源码锚点。
- `node tests/pages/test_parallelism_browser.cjs`：五种机制、参数变化、步骤、播放与重置、键盘、减少动态效果、手机与桌面布局、深浅主题、无 JS 阅读。

## 通信课程的验证

- `node tests/pages/test_communication.cjs`：算子输出、页合并覆盖、源端完成与 ACK 去重、元数据 / rank / 缓存依赖；设置 `SGLANG_SOURCE_DIR` 额外核对固定源码锚点。
- `node tests/pages/test_communication_browser.cjs`：四种机制及分支、移动端与桌面、深浅主题、箭头方向、减少动态效果、键盘、播放和无 JS 阅读。

## 生成课程的验证

- `node tests/pages/test_generation.cjs`：状态检查点、连续接受前缀、输出与 KV 的偏移、规则语言与合法项选择；设置 `SGLANG_SOURCE_DIR` 核对固定源码锚点。
- `node tests/pages/test_generation_execution.cjs`：10 个条件的依赖、资源独占、可逆状态、KV / 输出对齐、规则隔离与异步执行轨道。
- `SGLANG_SOURCE_DIR=… python3 -B tests/pages/test_generation_source.py`：抽取固定源码，检查真实 DFS 回退、mask 构造顺序、编译复制、幂等提交、终止截断与分配保留；不执行 GPU。
- `node tests/pages/test_generation_browser.cjs`：三种机制、10 个条件、连续播放、回退、URL、键盘、减少动态效果、四档屏宽与深浅主题、无 JS 正文。

## 服务课程的验证

- `node tests/pages/test_serving.cjs`：健康证据、候选过滤、队列容量与请求守恒、超时边界、取消回告及退出升级；设置 `SGLANG_SOURCE_DIR` 核对固定锚点。
- `node tests/pages/test_serving_browser.cjs`：四组机制、全部分支、消息方向、参数、播放、键盘、减少动态效果、四档屏宽与深浅主题、无 JS 阅读。

## 性能课程的验证

- `node tests/pages/test_performance.cjs`：12 个 P/D 条件的依赖、观测边界、分块交叠、KV 回退、流式 ITL 与成功样本；设置 `SGLANG_SOURCE_DIR` 核对 41 个固定锚点及源码分支。
- `SGLANG_SOURCE_DIR=<checkout> python3 -B tests/pages/test_performance_source.py`：抽取固定函数，用 CPU 替身验证分块计时、D 提交端点、sender 回退与缺失、P TTFT 分支及 ITL 权重；不导入 SGLang。
- `node tests/pages/test_performance_browser.cjs`：四组 12 个条件的完整播放、指标切换、共同游标、回放、键盘、脉冲；五档宽度 × 两套主题、减少动态与无 JS 阅读。
- `node tests/pages/test_performance_levels_browser.cjs`：默认总览可见性、三层视图保留同一时间与指标、播放中展开、恢复链接与键盘操作；四组主题的桌面和窄屏检查。

## 执行课程的验证

- `node tests/pages/test_execution.cjs`：图桶资格、真实请求守恒、补齐裁剪、矩阵成本与单资源变化；设置 `SGLANG_SOURCE_DIR` 检查固定源码锚点。
- `node tests/pages/test_execution_browser.cjs`：执行数据交接、图资格分支、成本条比例、播放、键盘、减少动态效果、四档宽度与深浅主题、无 JS 阅读。

## KV 课程的验证

- `node tests/pages/test_kv_memory.cjs`：跨页边界、地址唯一性、写入先后、命中页对齐、缓存身份、共享保护与容量守恒；设置 `SGLANG_SOURCE_DIR` 核对固定源码锚点。
- `node tests/pages/test_kv_memory_browser.cjs`：三种图与全部参数、保护分支、四档宽度与深浅主题、键盘与播放、无 JS 阅读。该测试验证原参数实验的占用与连线；连续写入演示由 `test_mechanism_motion_browser.cjs` 另验。

## 调度课程的验证

- `node tests/pages/test_scheduling.cjs`：排序与接纳分离、首条输入例外、KV 严格边界、临时保护释放、chunk 份额守恒、回撤后资源与输出历史；设置 `SGLANG_SOURCE_DIR` 核对固定源码。
- `node tests/pages/test_scheduling_browser.cjs`：三种机制、全部参数与终止分支、四档屏宽和两套主题、播放与键盘、无 JS 阅读。

## 系统全景课程的验证

- `node tests/pages/test_system_overview.cjs`：采样与可见先后、KV / 输出偏移、TP 与 DP / PD 的副本关系、其余十二模块的关联；设置 `SGLANG_SOURCE_DIR` 核对固定源码锚点。
- `node tests/pages/test_system_overview_browser.cjs`：两条逐步链路、三类拓扑、箭头方向、减少动态效果、播放与键盘、深浅主题和四档宽度、无 JS 阅读。

## 分离部署的跨模块验证

- `node tests/pages/test_inference_overview_browser.cjs`：部署静态图、P/D 两侧条件、默认折叠的对照表和四档布局。
- `SGLANG_SOURCE_ROOT=<source-checkout> node tests/pages/test_pd_dataflow.cjs` 与 `node tests/pages/test_pd_queues.cjs`：按各自模型固定版本核对状态、资源与队列；前缀环境变量应分别传给需要源码核对的命令。
- `node tests/pages/test_prefill_lifecycle.cjs` 与 `node tests/pages/test_prefill_lifecycle_browser.cjs`：正常、等待、分块与失败分支；跨模块链接不改变原模型语义。

## 2026-09-23 主线验收

- 五层十三模块已连接到 21 节课程，共享同一套首页、侧栏、面包屑与课程翻页；首页不再把已补齐的模块标为预留。
- 逐模块检查的重点是图中的真实变化：请求交接、地址写入、队列准入、参数切分、载荷与条件门、候选回退、服务选择、计时区间与依赖等待。其余范围留在对应课程的展开说明和深入链接中。
- 本地通过全部 15 份浏览器回归脚本，以及 Pages workflow 的模型与生成检查。覆盖关键分支、键盘、播放、深浅主题、移动端与桌面；无 JS 的可读路径按课程实现检查。
- 新增系统地图的 16 个源码锚点、现有模型锚点和 PP 的 65 个精确锚点均从固定本地 Git 对象核对；新增文档与 HTML 的 145 处静态源码链接也核对了固定对象和行号范围。未切换或修改 SGLang 工作区。
- 综合练习的 TTFT、共享保护和 PP 清理预测已与实际教学模型输出比对；同时检查了三个练习的展开阅读、源码入口和手机布局。
- 文档检查覆盖 223 个 Markdown / HTML、4,329 处本地引用与 185 篇学习正文，零错误。发布仍需以对应提交的 Pages workflow、线上浏览器和构建文件字节比对为准。

这些结果证明本套主线课程和所选教学模型可使用；不表示穷尽 SGLang 全部特性，也不表示做过 GPU、RDMA、吞吐或线上场景复现。

## SGLang 架构入口复核（2026-10-19）

第一层先进入完整代码架构，再进入请求生命周期。ScheduleBatch 输入准备归入调度模块，保留原网址与正文。总览将原跨进程流转、进程内分工两幅交接图合并；通过请求进入、一轮计算、结果返回三个阅读视角强调依赖，不把高亮切换称为执行动画。TP / DP / P/D 对照回答不同问题，继续保留。

图的数据与证据维护在 `sglang/system-overview/architecture.json`，HTML/SVG 由 `scripts/system_architecture.py` 随导航构建生成；每个模块可进入职责、固定源码与详细课程。第三章 Pages 的主图、PD 开篇与 PD 数据流专题均以 `279339f113` 为基线；历史 Markdown 资料保留各自版本。表达参考与适用范围均在页面中说明。

验证入口：`test_system_overview.cjs` 检查固定源码行、目录和部署模型，`test_system_overview_browser.cjs` 检查完整图、视角、模块详情、深链接、键盘、四种宽度与深浅主题、无 JavaScript、首页和下一课。生成与链接校验沿用仓库共用脚本。

## PP 共识专题的验证

[独立专题](sglang/pp-consensus/index.html) 归模块 09 第四篇，模块 07/08/13 提供关联入口。三种机制分别评估并加入同一时钟轨道；11 个条件覆盖名单过滤、跨轮采样、失败终态、本地复查、ACK 前缀与后级并行。`test_pp_consensus.cjs` 检查依赖与状态，`test_pp_consensus_source.py` 从固定 Git 对象执行 CPU 替身检查，`test_pp_consensus_browser.cjs` 检查播放、回拖、状态恢复、主题与窄屏；不是 GPU 或 RDMA 实验。
