# SGLang：并行拓扑与负载分工

[专题学习入口](<../README.md>) · [Wiki 首页](<../../README.md>)

区分请求副本、层内切分、流水线和专家负载。

各篇文章的来源、源码版本和验证范围见文首说明。阅读顺序可按自己的问题调整。

## 先修知识

先确认以下基础；已经掌握的部分可以跳过：

- [请求、调度与执行主链](<../运行时/README.md>)
- [前缀缓存与分层存储](<../KV缓存/README.md>)

## 阅读顺序

按表中顺序建立整体认识，再选择深入或选读内容。目录内的归档顺序不构成所有文章都必须读完的要求。

| 顺序 | 资料 | 阅读定位 | 学习重点 |
| --- | --- | --- | --- |
| 1 | [SGLang 并行分工与执行拓扑](<SGLang 并行分工与执行拓扑学习文档.md>) · [交互课程](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/parallelism/) | 主线 | 用权重分块、流水线、请求路由、专家任务与因果矩阵比较 TP / PP / DP / EP / CP。 |
| 2 | [SGLang 数据并行、负载均衡与专家并行边界](<SGLang 数据并行、负载均衡与专家并行边界学习文档.md>) | 起步 | 区分普通 DP、Gateway、DP Attention 和 EP/EPLB。 |
| 3 | [SGLang Pipeline Parallel 模式](<SGLang Pipeline Parallel 模式学习文档.md>) | 深入 | 跟随请求理解 PP 层切分、microbatch 和反馈回路。 |

## 通信主线

完成分工概览后阅读 [SGLang 通信与传输机制](<SGLang 通信与传输机制学习文档.md>) · [交互课程](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/communication/)。从消息、P2P 张量与 collective 进入 KV 页映射，再区分传输完成、Decode 就绪和失败回收。

## 对应源码课程

需要系统读代码时，从下面的课程入口衔接。课程与专题可能采用不同固定提交，阅读前分别核对文首基线。

- [并行部署与通信拓扑](<../source-study/architecture/07-并行部署与通信拓扑.md>)
- [Rank 与进程组基础](<../source-study/06-parallelism/01-Rank进程组与通信基础.md>)
- [MoE：从路由到专家执行](../../pages/sglang/moe-execution/index.html)：先修 EP 分工与 AllToAll，用同一 token 行核对 Dispatch、热点、副本和 Combine。课程采用独立固定提交，时间与专家函数均为教学假设。

## 自测与下一步

能从 rank、stage、模型副本和专家分布解释一次请求在哪些设备上执行。

按目标继续：

- [分离部署与状态交接](<../分离部署/README.md>)
- [多卡执行与通信](<../../vllm/并行方法/README.md>)
- [并行方法](<../../大模型推理/并行方法/README.md>)

## 本目录收录范围

收录 SGLang 的 rank 拓扑、TP/PP/DP/EP/CP 实现与通信、负载分工。PD 与 PP 组合的完整交接流程归 disaggregation。

## PP 状态协调专题

完成 PP 与请求生命周期后，阅读 [SGLang PP 共识与缓存 ACK 机制](<SGLang PP 共识与缓存 ACK 机制学习文档.md>) · [交互课程](../../pages/sglang/pp-consensus/index.html)。这是深入选读：区分 bootstrap 名单、release 终态与 HiCache ACK 计数，跟随逐跳消息和共同时间轴检查本地放行条件。固定基线 `279339f113`，不覆盖其他资料的独立版本。

需要逐条追踪协议时，再读 [SGLang PP 共识机制源码学习文档](<SGLang PP 共识机制源码学习文档.md>)。它以 `72d5c5bb73` 为基线，覆盖普通 PP、PD 队列协调、HiCache 与排障；与上面的短篇分别核对版本。
