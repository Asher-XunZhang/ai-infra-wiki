# Pages 学习课程覆盖与维护

Pages 分为“推理基础”和“SGLang 实现剖析”两章。第一章用十二课建立通用机制，第二章保留五层十二模块、二十节原有课程与各自的源码依据。本文件记录课程覆盖与验证方式。资料归档仍使用仓库的稳定技术领域，课程顺序由导航维护，不移动旧源码课程的阶段目录。

## 两章之间怎样衔接

第一章的 [十二节基础课](foundations/README.md) 依次解释文字与 token、逐步生成、Attention、KV、批处理、GPU、多卡分工、通信、角色分离、候选验证、服务生命周期和性能口径。每课使用具体例子，提供分步播放、回退、条件切换、自测、来源和无 JavaScript 正文。先修链接与上一课 / 下一课让读者按依赖继续。

第二章保留旧网址和固定源码依据，课前只给实现目标与第一章的先修链接，不再嵌入同一套基础播放器。五节旧概览课分别改为批次准备、DeepSeek-V3 调用、前缀索引、调度入口和 PD 交接的源码导读；原有六幅通用机制图及说明归到第一章相应课程的选读区，避免在两章重复学习。

第二章的其他交互保留具体实现问题：请求对象交接、KV 地址与共享保护、准入与回撤、Graph 资格、通信接口与释放、状态回退、服务治理及统计口径。机制上的必要回顾服务于这些实现问题，不另复制第一章课件。

基础课的编辑入口是 `pages/foundations/courses.json`，选读机制图保存在 `extensions.json`；由 `scripts/build_foundations.py` 生成十二页；共享侧栏和课程映射由 `scripts/build_learning_navigation.py` 生成。两类生成一致性均由 `build_pages.py` 检查。

### 通信语义覆盖

[通信基础](foundations/communication.html) 提供 17 个基础语义入口和 1 个“矩阵对照”入口，共 33 个可选场景（29 个基础语义 + 4 个矩阵对照），按分发与收集、归约与扫描、交换与点对点、单边访问、同步与完成分组。包含 Broadcast、Scatter(v)、Gather(v)、AllGather(v)、Reduce、AllReduce、ReduceScatter、Scan / Exscan、AllToAll(v)、邻域交换、Send/Recv、Put / Get、Accumulate / CAS、Barrier 和完成依赖。保留不同图形：固定 rank 间的数据分发与交换、数值归约、端点窗口与事件泳道。NCCL 2.32.3 / MPI 4.1 语义依据与画法参考在课程内列出；不是全部库 API 或底层算法的模拟。

- `node tests/pages/test_communication_teaching.cjs`：输入输出、root、变长片段、rank 顺序、归约、前缀、配对、单边方向与原子更新。
- `node tests/pages/test_communication_teaching_browser.cjs`：29 场景的中间态与响应式排版、稳定对象与回看、完成边界、播放与减少动态效果、无 JavaScript 阅读、共享选择器回归。

### 基础章节验证

- `node tests/pages/test_foundations.cjs`：真实分词固定样例、因果权重、KV 所属与释放、批次成员守恒、通信与 PP 依赖、统计样本边界。
- `node tests/pages/test_foundations_browser.cjs`：全部场景与步骤、320 / 390 / 1440 宽度、深浅主题、自测反馈、键盘、播放暂停、减少动态效果、链接恢复与无 JavaScript 正文。
- `node tests/pages/test_topic_navigation_browser.cjs`：两章目录、原有二十节实现课导航、旧路线 URL、跨模块翻页与移动端侧栏。

本轮仅在本地构建与预览，尚未部署远程。浏览器中的教学模型不代表运行过 GPU 推理。

## 连续机制演示审核

2026-10-15 按阅读者需要理解的因果关系复核。24 个页面共 37 个可播放条件、250 个关键状态；基础 token、生成、Attention、KV、角色分离，以及 SGLang 执行、KV 内存、调度、PD 引用与收尾十页采用专用图形。连续批处理保留 token 序列与固定执行位置。八页仍以矩阵、序列或时间轴为主，将通用演示折叠；其余五页保留对象过程图。原参数实验、正文和源码依据保留。八节源码导读 / PP 课程保持原用途。逐课问题、调整理由、保留理由和复核步骤见 [Pages 机制可视化审核](MOTION_REVIEW.md)。

## 每个模块的完成要求

每个模块都需要：明确先修和学习目标；可长期阅读的仓内资料入口；逐行为核对的 SGLang 固定源码基线；解释机制的静态、动态或交互图；图旁解释与自测；正确的模块导航；桌面与手机阅读验证。源码事实、教学假设、未做的运行验证必须分别说明。只有引用一篇相关案例，不能算独立入门模块已经补齐。

图的类型由机制决定，不统一套用一种形式。页面内容不描述使用者的水平。可视化必须直接呈现机制：用泳道、时序箭头、拓扑位置、数据块或资源槽位表达控制流、数据流和状态变化。交互步骤应改变图中的路径、位置、占用或依赖，而非只替换文字卡片、表格和字段值。默认只呈现当前机制所需的元素，完整流程与额外细节按需展开；避免元素堆积。详细说明与实现字段放在图旁的简短提示或展开区；验收要实际观察关键步骤及异常分支，确认不读长文也能辨认当前谁在做什么、数据怎样变化、为什么要这么做，以及前后步骤的依赖。

## 覆盖与补充顺序

这张表是课程补充工作的范围，不把“页面存在”当作全部完成。2026-09-23 开始逐模块复核与补充；其他专题继续保留原有基线。

| 模块 | 已有入口与可视化 | 当前范围与深入方向 |
| --- | --- | --- |
| 01 推理系统全景 | [代码架构](sglang/system-overview/index.html)，一张全局图连接 API、分词、调度、执行、KV、模型与输出；附 TP / DP / PD 拓扑 | 已补系统边界与其余十一模块的按问题衔接；详细启动和运行时变体按独立源码课程展开 |
| 02 请求生命周期与运行时架构 | [普通请求运行时](sglang/request-runtime/index.html)，当前步骤交接图、可展开的完整流程与四路径资源变化 | 已补独立入门课；深入分支连接原有源码章节 |
| 03 模型执行、硬件与算子 | [Transformer](sglang/inference-overview/transformer.html) 与 [执行机制](sglang/model-execution/index.html)，执行交接、图重放补齐与成本下界 | 已补 Worker / Runner / backend / kernel 与硬件直觉；复杂模型、kernel 和执行模式连接独立基线源码课程 |
| 04 KV Cache 与内存管理 | [KV 原理](sglang/inference-overview/kv-cache.html) 与 [映射和回收](sglang/kv-memory/index.html)，尾页续写、整页命中、共享保护与驱逐 | 已补普通分页路径与固定源码；分层存储、混合状态和容量规划连接独立基线资料 |
| 05 调度与批处理 | [ScheduleBatch 输入准备](sglang/inference-overview/journey.html)、[批处理](sglang/inference-overview/scheduling.html) 与 [准入和回撤](sglang/scheduling/index.html)，候选流转、单项预算、chunk 份额与恢复路径 | 已补普通路径准入、长短请求和回撤；Overlap、优先级与其他预算按独立基线继续深入 |
| 06 并行与执行拓扑 | [并行分工](sglang/parallelism/index.html)，TP 算例、PP 时间格、DP 路由、EP 任务和 CP 因果矩阵 | 已补独立课程与固定源码锚点；复杂组合和 backend 约束连接深入章节 |
| 07 通信与传输 | [通信机制](sglang/communication/index.html)，消息 / 张量交接、三种 collective、KV 页映射、就绪与失败回收 | 已补独立课程和固定源码锚点；硬件传输实现与真实性能实验保留为深入方向 |
| 08 分离部署与分布式状态交接 | [部署入门](sglang/inference-overview/deployment.html)、[PD 旅程](sglang/pd-prefill-lifecycle/index.html)、[数据流](sglang/pd-dataflow/index.html) | 已补 P 收尾 / D 就绪的两侧边界、通信 / 缓存 / 准入连接；各案例保留独立基线和假设 |
| 09 模型结构与高级生成 | [状态与生成](sglang/advanced-generation/index.html)，历史状态、检查点回退、候选验证、输出 / KV 错位与语法筛选 | 已补独立课程及固定源码；量化、多模态与其他生成算法按子方向衔接源码专题 |
| 10 服务部署与运行治理 | [服务治理](sglang/serving-operations/index.html)，启动探活、路由准入、队列拒绝、超时取消与退出收尾 | 已补独立课程；Gateway 额度、协议差异与管理变更按问题衔接源码专题 |
| 11 性能分析与优化方法 | [指标与定位](sglang/performance-engineering/index.html)，请求时间轴、流式数据包、单变量执行条带与失败样本筛选 | 已补独立课程及固定源码；Profiler 与真实压测通过深入资料衔接 |
| 12 综合案例与源码实践 | [PP loop](sglang/pd-prefill-pp-loop/quick.html)、[依赖分析](sglang/pd-prefill-pp-loop/index.html)、[源码阅读](sglang/pd-prefill-pp-loop/notes.html) | 已补计时边界、共享 KV 与 PD＋PP 三条跨模块练习；每条含预测、图上操作、固定源码与运行验证边界 |

十二模块的主线入口现已补齐；深入资料继续按独立版本与主题维护。下列验证覆盖教学模型与页面，不能替代真实模型、设备或性能实验。

## 运行时课程的验证

- `node tests/pages/test_request_runtime.cjs`：检查输出 / KV 位置关系、取消回告与资源释放的先后关系、未派发失败没有后端分配等教学不变量。
- `SGLANG_SOURCE_DIR=<source-checkout> node tests/pages/test_request_runtime.cjs`：额外读取固定 Git 对象，校验每个交互源码锚点的行号与函数名。不会修改源码仓库。
- `node tests/pages/test_request_runtime_browser.cjs`：验证精简视图 / 完整流程、消息逐跳运动、减少动态效果、四条路径、参数、播放 / 暂停 / 重置、键盘、深浅主题、320 / 390 / 768 / 1440 布局与无 JS 阅读。
- `node tests/pages/test_topic_navigation_browser.cjs`：首页与全部课程的共享导航和跨模块入口。
- `python3 -B scripts/check_learning_docs.py`、`python3 -B scripts/build_learning_navigation.py --check`、`python3 -B scripts/build_pages.py`：文档引用、共享导航与发布构建。

浏览器测试复用现有 `PLAYWRIGHT_MODULE`、`CHROMIUM_EXECUTABLE`、`OVERVIEW_TEST_URL` 环境变量，默认站点为 `http://127.0.0.1:8765/`。可先以 `python3 -m http.server 8765 --directory .pages-dist` 预览构建输出。网页与源码模型验证不替代 SGLang 实际运行或性能实验。

## 并行课程的验证

- `node tests/pages/test_parallelism.cjs`：TP 数值、PP 前后级依赖、DP 单一归属、EP 任务守恒、CP 查询覆盖与因果配对数；设置 `SGLANG_SOURCE_DIR` 可核对 11 个固定源码锚点。
- `node tests/pages/test_parallelism_browser.cjs`：五种机制、参数变化、步骤、播放与重置、键盘、减少动态效果、手机与桌面布局、深浅主题、无 JS 阅读。

## 通信课程的验证

- `node tests/pages/test_communication.cjs`：算子输出、页合并覆盖、源端完成与 ACK 去重、元数据 / rank / 缓存依赖；设置 `SGLANG_SOURCE_DIR` 额外核对固定源码锚点。
- `node tests/pages/test_communication_browser.cjs`：四种机制及分支、移动端与桌面、深浅主题、箭头方向、减少动态效果、键盘、播放和无 JS 阅读。

## 生成课程的验证

- `node tests/pages/test_generation.cjs`：状态检查点、连续接受前缀、输出与 KV 的偏移、规则语言与合法项选择；设置 `SGLANG_SOURCE_DIR` 核对固定源码锚点。
- `node tests/pages/test_generation_browser.cjs`：三种机制与全部分支、参数、播放、键盘、减少动态效果、四档屏宽与深浅主题、无 JS 阅读。

## 服务课程的验证

- `node tests/pages/test_serving.cjs`：健康证据、候选过滤、队列容量与请求守恒、超时边界、取消回告及退出升级；设置 `SGLANG_SOURCE_DIR` 核对固定锚点。
- `node tests/pages/test_serving_browser.cjs`：四组机制、全部分支、消息方向、参数、播放、键盘、减少动态效果、四档屏宽与深浅主题、无 JS 阅读。

## 性能课程的验证

- `node tests/pages/test_performance.cjs`：许可内外计时、分包 ITL、TPOT 缺失、单变量收益与成功样本筛选；设置 `SGLANG_SOURCE_DIR` 核对固定锚点。
- `node tests/pages/test_performance_browser.cjs`：四种图、全部分支与步骤、键盘和播放、四档宽度与两套主题、无 JS 阅读。

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

- `node tests/pages/test_system_overview.cjs`：采样与可见先后、KV / 输出偏移、TP 与 DP / PD 的副本关系、其余十一模块的关联；设置 `SGLANG_SOURCE_DIR` 核对固定源码锚点。
- `node tests/pages/test_system_overview_browser.cjs`：两条逐步链路、三类拓扑、箭头方向、减少动态效果、播放与键盘、深浅主题和四档宽度、无 JS 阅读。

## 分离部署的跨模块验证

- `node tests/pages/test_inference_overview_browser.cjs`：部署静态图、P/D 两侧条件、默认折叠的对照表和四档布局。
- `SGLANG_SOURCE_ROOT=<source-checkout> node tests/pages/test_pd_dataflow.cjs` 与 `node tests/pages/test_pd_queues.cjs`：按各自模型固定版本核对状态、资源与队列；前缀环境变量应分别传给需要源码核对的命令。
- `node tests/pages/test_prefill_lifecycle.cjs` 与 `node tests/pages/test_prefill_lifecycle_browser.cjs`：正常、等待、分块与失败分支；跨模块链接不改变原模型语义。

## 2026-09-23 主线验收

- 五层十二模块已连接到 20 节课程，共享同一套首页、侧栏、面包屑与课程翻页；首页不再把已补齐的模块标为预留。
- 逐模块检查的重点是图中的真实变化：请求交接、地址写入、队列准入、参数切分、载荷与条件门、候选回退、服务选择、计时区间与依赖等待。其余范围留在对应课程的展开说明和深入链接中。
- 本地通过全部 15 份浏览器回归脚本，以及 Pages workflow 的模型与生成检查。覆盖关键分支、键盘、播放、深浅主题、移动端与桌面；无 JS 的可读路径按课程实现检查。
- 新增系统地图的 16 个源码锚点、现有模型锚点和 PP 的 65 个精确锚点均从固定本地 Git 对象核对；新增文档与 HTML 的 145 处静态源码链接也核对了固定对象和行号范围。未切换或修改 SGLang 工作区。
- 综合练习的 TTFT、共享保护和 PP 清理预测已与实际教学模型输出比对；同时检查了三个练习的展开阅读、源码入口和手机布局。
- 文档检查覆盖 223 个 Markdown / HTML、4,329 处本地引用与 185 篇学习正文，零错误。发布仍需以对应提交的 Pages workflow、线上浏览器和构建文件字节比对为准。

这些结果证明本套主线课程和所选教学模型可使用；不表示穷尽 SGLang 全部特性，也不表示做过 GPU、RDMA、吞吐或线上场景复现。

## SGLang 架构入口复核（2026-10-19）

第一层先进入完整代码架构，再进入请求生命周期。ScheduleBatch 输入准备归入调度模块，保留原网址与正文。总览将原跨进程流转、进程内分工两幅交接图合并；通过请求进入、一轮计算、结果返回三个阅读视角强调依赖，不把高亮切换称为执行动画。TP / DP / P/D 对照回答不同问题，继续保留。

图的数据与证据维护在 `sglang/system-overview/architecture.json`，HTML/SVG 由 `scripts/system_architecture.py` 随导航构建生成；每个模块可进入职责、固定源码与详细课程。主图基线 `279339f113`，PD 数据流专题仍保留 `882577451e`。表达参考与适用范围均在页面中说明。

验证入口：`test_system_overview.cjs` 检查固定源码行、目录和部署模型，`test_system_overview_browser.cjs` 检查完整图、视角、模块详情、深链接、键盘、四种宽度与深浅主题、无 JavaScript、首页和下一课。生成与链接校验沿用仓库共用脚本。
