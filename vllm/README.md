# vLLM 学习导航

[Wiki 首页](<../README.md>)

沿请求执行、分页缓存与资源生命周期理解 vLLM，再选择并行、KV 传输或注意力性能方向。

## 怎么使用这份导航

目录回答“资料研究什么”，阅读路线回答“先学什么”。起步与主线用于建立基本模型；深入、案例和研究按目标选择；历史与版本参考先核对基线。无需把所有支线顺序读完。

本文是基于仓内资料的学习导航，不改变正文的技术结论、来源和验证边界。不同资料可能使用不同源码版本，不能合并理解为一个版本的行为。

## 入门主线

1. 先掌握 [推理基础](<../llm-inference/foundations/README.md>)。
2. [vLLM 从连续批处理到 PagedAttention 的引擎工作流](<runtime/vLLM 从连续批处理到 PagedAttention 的引擎工作流学习文档.md>)：把请求、调度和分页缓存连起来。
3. [vLLM Chunked Prefill 与 Block Size](<runtime/vLLM Chunked Prefill 与 Block Size 学习文档.md>)：分清计算 chunk 与存储 block。
4. [vLLM APC 链式哈希](<kv-cache/vLLM APC 链式哈希学习文档.md>)：理解前缀缓存的块身份。
5. [vLLM V1 KV Cache 管理全生命周期源码](<kv-cache/vLLM V1 KV Cache 管理全生命周期源码学习文档.md>)：沿固定源码追踪分配、引用、抢占与释放。

完成标志：能把一次调度迭代与逻辑 block、物理 KV 和请求生命周期对应起来。概念篇包含历史示例，具体实现以源码篇固定提交为准。

## 按领域深入

每个入口都提供先修知识、逐篇阅读定位、自测问题和下一步。下面是领域导航，不是强制串行课程。

| 领域 | 主要问题 |
| --- | --- |
| [引擎工作流与调度](<runtime/README.md>) | 把调度迭代、计算份额与 KV 存储组织成一条请求链路。 |
| [缓存身份与生命周期](<kv-cache/README.md>) | 从逻辑块身份深入分配、引用、复用、抢占和释放。 |
| [多卡执行与通信](<parallelism/README.md>) | 根据模型层、专家和上下文的切分对象选择阅读支线。 |
| [KV 传输与外部缓存接入](<disaggregation/README.md>) | 把外部缓存和其他实例的 KV 接入本地请求生命周期。 |
| [性能分析与执行优化](<performance-engineering/README.md>) | 从引擎行为继续深入 GPU 工作组织和可验证的性能取舍。 |

## 按目标选择路线

- **多卡执行**：[多卡执行与通信](<parallelism/README.md>)。
- **外部缓存与 PD**：[KV 传输与外部缓存接入](<disaggregation/README.md>)。
- **Attention 性能**：[性能分析与执行优化](<performance-engineering/README.md>)。

## 后续资料如何扩展

新增文章先归入主要技术领域，再更新该目录的阅读表和相关路线。一个正文可以被多个入口引用；标题带模型名或版本号，不自动决定归属。
只有形成稳定子主题、积累多篇相关资料且独立导航有助于检索时，才增加一层目录。现有专题不按学习先后编号，插入文章无需重排旧文件。

引擎专题采用共同的分类语言。后续可按实际资料建立 `model-support/`、`advanced-generation/`、`serving-operations/`、`version-studies/` 中尚不存在的目录；已有的直接复用。不为补齐目录树而创建空目录或重复文章。

实例内执行主链归 `runtime/`；实例外 API、网关与运维归 `serving-operations/`。并行切分归 `parallelism/`，服务角色分离与跨角色状态交接归 `disaggregation/`。跨模块版本综述归 `version-studies/`，特定版本的性能案例仍归性能工程。

完整维护规则见 [AGENTS.md：长期分类与学习导航](<../AGENTS.md#长期分类与学习导航>)。
