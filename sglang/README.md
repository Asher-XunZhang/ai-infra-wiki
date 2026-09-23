# SGLang 学习导航

[Wiki 首页](<../README.md>)

从请求、调度和缓存建立 SGLang 的整体认识，再按部署、模型或性能目标深入。专题资料和系统源码课程是互相连接的两种读法。

## 怎么使用这份导航

目录回答“资料研究什么”，阅读路线回答“先学什么”。起步与主线用于建立基本模型；深入、案例和研究按目标选择；历史与版本参考先核对基线。无需把所有支线顺序读完。

本文是基于仓内资料的学习导航，不改变正文的技术结论、来源和验证边界。不同资料可能使用不同源码版本，不能合并理解为一个版本的行为。

## 入门主线

理解 P/D 后，可用 [普通请求运行时交互课程](../pages/sglang/request-runtime/index.html) 与 [完整笔记](runtime/SGLang%20普通请求运行时与资源生命周期学习文档.md) 连接前端、调度、执行、回程和资源收尾。

1. 不熟悉推理时，先体验 [SGLang 推理全景交互专题](../pages/sglang/inference-overview/journey.html) 或读 [推理全景学习指南](runtime/SGLang%20推理全景学习指南.md)，再补充 [推理基础](<../llm-inference/foundations/README.md>) 或 [源码课程基础篇](<source-study/00-foundations/01-推理系统与SGLang职责地图.md>)。
2. 读 [整体架构与心智模型](<source-study/architecture/README.md>)，先看 M01、M03，认清模块和请求。
3. 进入 [请求、调度与执行主链](<runtime/README.md>)，先读调度总览、请求生命周期，再按需深入预算。
4. 进入 [前缀缓存与分层存储](<kv-cache/README.md>)，首轮读技术主线、命中定义与 KV Pool。
5. 回到 [模型执行分层](<source-study/architecture/06-模型执行与算子分层.md>)，把 Scheduler 与 Worker、Runner、Attention、kernel 接起来。

完成标志：能解释一条请求何时可被接纳、谁持有 KV、谁发起计算，以及请求结束与缓存保留为什么不是一回事。

### 系统源码课程怎样衔接

希望逐阶段读代码时，使用 [source-study 总目录](<source-study/README.md>)。保留其 00–12 阶段、架构导读与附录编号；专题目录不复制这套课程。

| 机制领域 | 课程入口 |
| --- | --- |
| [请求、调度与执行主链](<runtime/README.md>) | [运行环境与最小请求](<source-study/01-getting-started/01-源码目录与最短阅读路线.md>) · [请求与 batch 对象](<source-study/02-request-lifecycle/01-API协议到内部请求对象.md>) · [调度主循环](<source-study/03-scheduling/01-NormalEventLoop与调度主循环.md>) · [Worker 与 ModelRunner](<source-study/05-model-execution/01-Worker与ModelRunner执行边界.md>) · [CUDA Graph 与执行模式](<source-study/05-model-execution/05-CUDAGraph编译与执行模式.md>) |
| [前缀缓存与分层存储](<kv-cache/README.md>) | [缓存架构与资源所有权](<source-study/architecture/05-缓存架构与资源所有权.md>) · [请求视图与分配器](<source-study/04-kv-cache/01-请求视图物理槽位与分配器.md>) · [Unified 与混合状态组件](<source-study/04-kv-cache/04-UnifiedRadix与混合状态组件.md>) |
| [并行拓扑与负载分工](<parallelism/README.md>) | [并行部署与通信拓扑](<source-study/architecture/07-并行部署与通信拓扑.md>) · [Rank 与进程组基础](<source-study/06-parallelism/01-Rank进程组与通信基础.md>) |
| [分离部署与状态交接](<disaggregation/README.md>) | [先补 PD 三条通路](<source-study/architecture/08-PD与Encoder分离的三条通路.md>) · [完整 PD 源码主线](<source-study/07-disaggregation/01-PD分离职责与端到端请求地图.md>) |
| [性能观察与调优案例](<performance-engineering/README.md>) | [Benchmark 设计与指标](<source-study/11-performance-engineering/01-Benchmark设计与指标口径.md>) · [投机解码的提交语义](<source-study/08-advanced-generation/03-投机解码的DraftVerifyCommit.md>) |
| [模型适配与特殊执行机制](<model-support/README.md>) | [模型注册与权重加载](<source-study/09-model-specialization/01-模型注册配置与权重加载.md>) · [投机解码的 Draft、Verify、Commit](<source-study/08-advanced-generation/03-投机解码的DraftVerifyCommit.md>) |
| [版本演进与组合边界](<version-studies/README.md>) | [源码变更与版本回归](<source-study/11-performance-engineering/06-源码变更阅读与版本回归检查.md>) |
| [高级生成与状态提交](<advanced-generation/README.md>) | [高级生成特性全景](<source-study/architecture/09-高级生成特性插入位置.md>) |
| [服务部署与运行治理](<serving-operations/README.md>) | [网关、服务治理与性能诊断](<source-study/architecture/11-网关服务治理与性能诊断全景.md>) |
| 扩展与综合应用 | [DSL、Diffusion 与插件边界](<source-study/architecture/12-DSL与Diffusion及插件边界.md>) |

## 按领域深入

每个入口都提供先修知识、逐篇阅读定位、自测问题和下一步。下面是领域导航，不是强制串行课程。

| 领域 | 主要问题 |
| --- | --- |
| [请求、调度与执行主链](<runtime/README.md>) | 沿一条请求理解 Scheduler 的决策、资源预算与执行提交。 |
| [前缀缓存与分层存储](<kv-cache/README.md>) | 从逻辑前缀到物理 KV，再追踪 Host 和外部存储的回载与释放。 |
| [并行拓扑与负载分工](<parallelism/README.md>) | 区分请求副本、层内切分、流水线和专家负载。 |
| [分离部署与状态交接](<disaggregation/README.md>) | 以 PD 与 PP 的组合为切口，学习跨角色、跨 stage 的请求和资源协调。 |
| [性能观察与调优案例](<performance-engineering/README.md>) | 先形成可解释的测量，再选择参数、缓存或计算优化。 |
| [模型适配与特殊执行机制](<model-support/README.md>) | 理解模型特性如何约束缓存、量化、生成和分布式执行。 |
| [版本演进与组合边界](<version-studies/README.md>) | 在已有机制基础上比较跨模块变化、兼容约束和验证要求。 |

[高级生成与状态提交](<advanced-generation/README.md>) 提供独立主线和交互课程，连接模型状态、投机验证与规则推进。

[服务部署与运行治理](<serving-operations/README.md>) 提供启动、探活、路由准入、取消和退出的交互主线。

性能主线：[指标口径与性能定位](<performance-engineering/SGLang 指标口径与性能定位学习文档.md>)，先核对秒表与样本，再进入 Trace 和调参案例。

[执行分层与硬件机制](<runtime/SGLang 执行分层与硬件机制学习文档.md>) 连接模型层、Runner、具体算子与图重放。

[KV 映射与共享回收](<kv-cache/SGLang KV 映射与共享回收学习文档.md>) 用物理页与共享关系连接请求映射、缓存保护和驱逐。

[调度准入与容量回撤](<runtime/SGLang 调度准入与容量回撤学习文档.md>) 连接候选顺序、输入与 KV 额度，以及回撤后的等待和恢复。

## 按目标选择路线

- **多卡与分离部署**：[并行拓扑与负载分工](<parallelism/README.md>) → [分离部署与状态交接](<disaggregation/README.md>)。
- **性能与模型案例**：[性能观察与调优案例](<performance-engineering/README.md>) → [模型适配与特殊执行机制](<model-support/README.md>)。
- **理解升级影响**：[版本演进与组合边界](<version-studies/README.md>) → [性能观察与调优案例](<performance-engineering/README.md>)。

## 后续资料如何扩展

新增文章先归入主要技术领域，再更新该目录的阅读表和相关路线。一个正文可以被多个入口引用；标题带模型名或版本号，不自动决定归属。
只有形成稳定子主题、积累多篇相关资料且独立导航有助于检索时，才增加一层目录。现有专题不按学习先后编号，插入文章无需重排旧文件。

引擎专题采用共同的分类语言。后续可按实际资料建立 `model-support/`、`advanced-generation/`、`serving-operations/`、`version-studies/` 中尚不存在的目录；已有的直接复用。不为补齐目录树而创建空目录或重复文章。

实例内执行主链归 `runtime/`；实例外 API、网关与运维归 `serving-operations/`。并行切分归 `parallelism/`，服务角色分离与跨角色状态交接归 `disaggregation/`。跨模块版本综述归 `version-studies/`，特定版本的性能案例仍归性能工程。

完整维护规则见 [AGENTS.md：长期分类与学习导航](<../AGENTS.md#长期分类与学习导航>)。
