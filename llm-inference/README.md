# LLM 推理系统学习导航

[Wiki 首页](<../README.md>)

先理解推理系统的共同问题，再进入 SGLang 或 vLLM 的实现。这里按稳定技术领域归档，阅读路线可以跨目录组合。

## 怎么使用这份导航

目录回答“资料研究什么”，阅读路线回答“先学什么”。起步与主线用于建立基本模型；深入、案例和研究按目标选择；历史与版本参考先核对基线。无需把所有支线顺序读完。

本文是基于仓内资料的学习导航，不改变正文的技术结论、来源和验证边界。不同资料可能使用不同源码版本，不能合并理解为一个版本的行为。

## 入门主线

1. [LLM 推理系统心智模型与 SGLang、vLLM 选型边界](<foundations/LLM 推理系统心智模型与 SGLang、vLLM 选型边界学习文档.md>)：认清一次请求、资源和引擎边界。
2. [LLM Prefill 与 Decode 阶段源码](<foundations/LLM Prefill 与 Decode 阶段源码学习文档.md>)：理解首 token、逐步生成与 KV 时序。
3. [Chunked Prefill 与 Prefill-Decode 共推](<scheduling/Chunked Prefill 与 Prefill-Decode 共推学习文档.md>)：理解每轮计算怎样安排。
4. [KV Cache 容量优化技术地图](<kv-cache/KV Cache 容量优化技术地图学习文档.md>)：建立显存与容量的基本账本。
5. 选择 [SGLang](<../sglang/README.md>) 或 [vLLM](<../vllm/README.md>)，先走一个引擎的请求和缓存主线。

完成标志：能画出一条请求从排队到生成与释放的过程，指出计算、KV 和调度分别在解决什么问题。

## 按领域深入

每个入口都提供先修知识、逐篇阅读定位、自测问题和下一步。下面是领域导航，不是强制串行课程。

| 领域 | 主要问题 |
| --- | --- |
| [推理基础](<foundations/README.md>) | 从一条请求建立计算、状态、资源和指标的共同语言。 |
| [调度方法](<scheduling/README.md>) | 理解调度器如何分配计算机会、控制每轮工作量并处理请求竞争。 |
| [KV Cache 方法与内存层次](<kv-cache/README.md>) | 围绕缓存容量、身份、复用和存储位置组织跨引擎知识。 |
| [并行方法](<parallelism/README.md>) | 先辨认切分对象，再理解通信、负载和容量之间的关系。 |
| [集群服务与分离部署](<distributed-serving/README.md>) | 拆清实例执行、跨实例控制、状态传输和资源规划。 |
| [模型结构与状态语义](<model-architecture/README.md>) | 理解模型结构怎样改变推理状态、缓存恢复和执行约束。 |
| [性能工程与综合优化](<performance-engineering/README.md>) | 把优化方法放回瓶颈、成本、实验条件和证据范围中比较。 |

## 按目标选择路线

- **理解长上下文**：[KV Cache 方法与内存层次](<kv-cache/README.md>) → [并行方法](<parallelism/README.md>) → [多卡执行与通信](<../vllm/parallelism/README.md>)。
- **建设集群服务**：[集群服务与分离部署](<distributed-serving/README.md>) → 选择 [SGLang 分离部署](<../sglang/disaggregation/README.md>) 或 [vLLM KV 传输](<../vllm/disaggregation/README.md>)。
- **理解混合状态**：[模型结构与状态语义](<model-architecture/README.md>) → [前缀缓存与分层存储](<../sglang/kv-cache/README.md>) → [模型适配与特殊执行机制](<../sglang/model-support/README.md>)。
- **研究优化方法**：[性能工程与综合优化](<performance-engineering/README.md>) → 选择 [SGLang 性能案例](<../sglang/performance-engineering/README.md>) 或 [vLLM 执行优化](<../vllm/performance-engineering/README.md>)。

## 后续资料如何扩展

新增文章先归入主要技术领域，再更新该目录的阅读表和相关路线。一个正文可以被多个入口引用；标题带模型名或版本号，不自动决定归属。
只有形成稳定子主题、积累多篇相关资料且独立导航有助于检索时，才增加一层目录。现有专题不按学习先后编号，插入文章无需重排旧文件。

按需扩展 `advanced-generation/`（投机与约束生成）和 `hardware-and-kernels/`（硬件、通信与算子基础）；有实际资料时再创建，不预建空目录。跨系统内容先归本专题，完整独立系统形成自己的学习主线后可再建立大专题。

完整维护规则见 [AGENTS.md：长期分类与学习导航](<../AGENTS.md#长期分类与学习导航>)。
