# Pages 学习课程覆盖与维护

Pages 按五层十二模块组织。页面正文围绕机制与源码展开；本文件记录课程覆盖与验证方式。资料归档仍使用仓库的稳定技术领域，课程顺序由导航维护，不移动旧源码课程的阶段目录。

## 每个模块的完成要求

每个模块都需要：明确先修和学习目标；可长期阅读的仓内资料入口；逐行为核对的 SGLang 固定源码基线；解释机制的静态、动态或交互图；图旁解释与自测；正确的模块导航；桌面与手机阅读验证。源码事实、教学假设、未做的运行验证必须分别说明。只有引用一篇相关案例，不能算独立入门模块已经补齐。

图的类型由机制决定，不统一套用一种形式。页面内容不描述使用者的水平。可视化必须直接呈现机制：用泳道、时序箭头、拓扑位置、数据块或资源槽位表达控制流、数据流和状态变化。交互步骤应改变图中的路径、位置、占用或依赖，而非只替换文字卡片、表格和字段值。默认只呈现当前机制所需的元素，完整流程与额外细节按需展开；避免元素堆积。详细说明与实现字段放在图旁的简短提示或展开区；验收要实际观察关键步骤及异常分支，确认不读长文也能辨认当前谁在做什么、数据怎样变化、为什么要这么做，以及前后步骤的依赖。

## 覆盖与补充顺序

这张表是课程补充工作的范围，不把“页面存在”当作全部完成。2026-09-23 开始逐模块复核与补充；其他专题继续保留原有基线。

| 模块 | 已有入口与可视化 | 尚需补充或复核 |
| --- | --- | --- |
| 01 推理系统全景 | [请求生成](sglang/inference-overview/journey.html)，Prefill / Decode 交互 | 复核与其余十一模块的概念衔接，补齐总体系统边界 |
| 02 请求生命周期与运行时架构 | [普通请求运行时](sglang/request-runtime/index.html)，当前步骤交接图、可展开的完整流程与四路径资源变化 | 已补独立入门课；深入分支连接原有源码章节 |
| 03 模型执行、硬件与算子 | [Transformer](sglang/inference-overview/transformer.html) 与 [执行机制](sglang/model-execution/index.html)，执行交接、图重放补齐与成本下界 | 已补 Worker / Runner / backend / kernel 与硬件直觉；复杂模型、kernel 和执行模式连接独立基线源码课程 |
| 04 KV Cache 与内存管理 | [KV 原理](sglang/inference-overview/kv-cache.html) 与 [映射和回收](sglang/kv-memory/index.html)，尾页续写、整页命中、共享保护与驱逐 | 已补普通分页路径与固定源码；分层存储、混合状态和容量规划连接独立基线资料 |
| 05 调度与批处理 | [批处理](sglang/inference-overview/scheduling.html)，batch 与调度演示 | 补准入预算、长短请求与回撤边界 |
| 06 并行与执行拓扑 | [并行分工](sglang/parallelism/index.html)，TP 算例、PP 时间格、DP 路由、EP 任务和 CP 因果矩阵 | 已补独立课程与固定源码锚点；复杂组合和 backend 约束连接深入章节 |
| 07 通信与传输 | [通信机制](sglang/communication/index.html)，消息 / 张量交接、三种 collective、KV 页映射、就绪与失败回收 | 已补独立课程和固定源码锚点；硬件传输实现与真实性能实验保留为深入方向 |
| 08 分离部署与分布式状态交接 | [部署入门](sglang/inference-overview/deployment.html)、[PD 旅程](sglang/pd-prefill-lifecycle/index.html)、[数据流](sglang/pd-dataflow/index.html) | 复核通信模块衔接，保留交接条件、资源持有与历史假设 |
| 09 模型结构与高级生成 | [状态与生成](sglang/advanced-generation/index.html)，历史状态、检查点回退、候选验证、输出 / KV 错位与语法筛选 | 已补独立课程及固定源码；量化、多模态与其他生成算法按子方向衔接源码专题 |
| 10 服务部署与运行治理 | [服务治理](sglang/serving-operations/index.html)，启动探活、路由准入、队列拒绝、超时取消与退出收尾 | 已补独立课程；Gateway 额度、协议差异与管理变更按问题衔接源码专题 |
| 11 性能分析与优化方法 | [指标与定位](sglang/performance-engineering/index.html)，请求时间轴、流式数据包、单变量执行条带与失败样本筛选 | 已补独立课程及固定源码；Profiler 与真实压测通过深入资料衔接 |
| 12 综合案例与源码实践 | [PP loop](sglang/pd-prefill-pp-loop/quick.html)、[依赖分析](sglang/pd-prefill-pp-loop/index.html)、[源码阅读](sglang/pd-prefill-pp-loop/notes.html) | 将案例入口关联十二模块，增加从假设到源码、验证与边界的练习路线 |

先补 02 以接通单实例主线，之后补 06、07，再补 09、10、11；同时按上述范围完善已有模块。只有十二模块逐项达到完成要求、检查并部署通过，才能把整项工作标为完成。

## 运行时课程的验证

- `node scripts/test_request_runtime.cjs`：检查输出 / KV 位置关系、取消回告与资源释放的先后关系、未派发失败没有后端分配等教学不变量。
- `SGLANG_SOURCE_DIR=<source-checkout> node scripts/test_request_runtime.cjs`：额外读取固定 Git 对象，校验每个交互源码锚点的行号与函数名。不会修改源码仓库。
- `node scripts/test_request_runtime_browser.cjs`：验证精简视图 / 完整流程、消息逐跳运动、减少动态效果、四条路径、参数、播放 / 暂停 / 重置、键盘、深浅主题、320 / 390 / 768 / 1440 布局与无 JS 阅读。
- `node scripts/test_topic_navigation_browser.cjs`：首页与全部课程的共享导航和跨模块入口。
- `python3 -B scripts/check_learning_docs.py`、`python3 -B scripts/build_learning_navigation.py --check`、`python3 -B scripts/build_pages.py`：文档引用、共享导航与发布构建。

浏览器测试复用现有 `PLAYWRIGHT_MODULE`、`CHROMIUM_EXECUTABLE`、`OVERVIEW_TEST_URL` 环境变量，默认站点为 `http://127.0.0.1:8765/`。可先以 `python3 -m http.server 8765 --directory .pages-dist` 预览构建输出。网页与源码模型验证不替代 SGLang 实际运行或性能实验。

## 并行课程的验证

- `node scripts/test_parallelism.cjs`：TP 数值、PP 前后级依赖、DP 单一归属、EP 任务守恒、CP 查询覆盖与因果配对数；设置 `SGLANG_SOURCE_DIR` 可核对 11 个固定源码锚点。
- `node scripts/test_parallelism_browser.cjs`：五种机制、参数变化、步骤、播放与重置、键盘、减少动态效果、手机与桌面布局、深浅主题、无 JS 阅读。

## 通信课程的验证

- `node scripts/test_communication.cjs`：算子输出、页合并覆盖、源端完成与 ACK 去重、元数据 / rank / 缓存依赖；设置 `SGLANG_SOURCE_DIR` 额外核对固定源码锚点。
- `node scripts/test_communication_browser.cjs`：四种机制及分支、移动端与桌面、深浅主题、运动方向、减少动态效果、键盘、播放和无 JS 阅读。

## 生成课程的验证

- `node scripts/test_generation.cjs`：状态检查点、连续接受前缀、输出与 KV 的偏移、规则语言与合法项选择；设置 `SGLANG_SOURCE_DIR` 核对固定源码锚点。
- `node scripts/test_generation_browser.cjs`：三种机制与全部分支、参数、播放、键盘、减少动态效果、四档屏宽与深浅主题、无 JS 阅读。

## 服务课程的验证

- `node scripts/test_serving.cjs`：健康证据、候选过滤、队列容量与请求守恒、超时边界、取消回告及退出升级；设置 `SGLANG_SOURCE_DIR` 核对固定锚点。
- `node scripts/test_serving_browser.cjs`：四组机制、全部分支、消息方向、参数、播放、键盘、减少动态效果、四档屏宽与深浅主题、无 JS 阅读。

## 性能课程的验证

- `node scripts/test_performance.cjs`：许可内外计时、分包 ITL、TPOT 缺失、单变量收益与成功样本筛选；设置 `SGLANG_SOURCE_DIR` 核对固定锚点。
- `node scripts/test_performance_browser.cjs`：四种图、全部分支与步骤、键盘和播放、四档宽度与两套主题、无 JS 阅读。

## 执行课程的验证

- `node scripts/test_execution.cjs`：图桶资格、真实请求守恒、补齐裁剪、矩阵成本与单资源变化；设置 `SGLANG_SOURCE_DIR` 检查固定源码锚点。
- `node scripts/test_execution_browser.cjs`：执行数据交接、图资格分支、成本条比例、播放、键盘、减少动态效果、四档宽度与深浅主题、无 JS 阅读。

## KV 课程的验证

- `node scripts/test_kv_memory.cjs`：跨页边界、地址唯一性、写入先后、命中页对齐、缓存身份、共享保护与容量守恒；设置 `SGLANG_SOURCE_DIR` 核对固定源码锚点。
- `node scripts/test_kv_memory_browser.cjs`：三种图与全部参数、保护分支、四档宽度与深浅主题、键盘与播放、无 JS 阅读。图无连续运动，步骤改变真实占用与连线。
