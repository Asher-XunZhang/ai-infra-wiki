# SGLang：分离部署与状态交接

[专题学习入口](<../README.md>) · [Wiki 首页](<../../README.md>)

以 PD 与 PP 的组合为切口，学习跨角色、跨 stage 的请求和资源协调。

各篇文章的来源、源码版本和验证范围见文首说明。阅读顺序可按自己的问题调整。

先看 [P/D 状态交接与执行轨道](../../pages/sglang/inference-overview/deployment.html)，配合 [学习指南](../%E8%BF%90%E8%A1%8C%E6%97%B6/SGLang%20%E6%8E%A8%E7%90%86%E5%85%A8%E6%99%AF%E5%AD%A6%E4%B9%A0%E6%8C%87%E5%8D%97.md) 理解完整模型与状态交接，再进入本领域。

## 先修知识

先确认以下基础；已经掌握的部分可以跳过：

- [请求、调度与执行主链](<../运行时/README.md>)
- [前缀缓存与分层存储](<../KV缓存/README.md>)
- [并行拓扑与负载分工](<../并行方法/README.md>)

## 阅读顺序

先追一条请求，再追多批次。步骤表、共识协议和历史复核在遇到具体问题时再查。

| 目的 | 入口 | 本次只抓住什么 |
| --- | --- | --- |
| 首读：走通一条 Prefill 请求 | [PP=3 请求生命周期](<PD Prefill PP=3 请求生命周期源码学习文档.md>) | 握手、前向、KV 交接、释放；先选正常场景，再看分块与失败 |
| 主线：理解多批交错 | [PP loop 交互图说明](<PD Prefill PP loop 交互图.md>) | 当前计算哪一批、处理哪一批旧结果、为什么等待 |
| 查动作：某一步到底做什么 | [生成的步骤详解](<PD Prefill PP loop 步骤详解.md>) | 输入、动作、前置条件与源码；作为查表材料，不要求逐项背诵 |
| 查共识：为什么本级就绪仍未放行 | [PP 共识与缓存 ACK](<../并行方法/SGLang PP 共识与缓存 ACK 机制学习文档.md>) | bootstrap、release 与 ACK 各自的协调对象，固定 `279339f113` |
| 换一个观察问题 | [数据流状态机](<PD Prefill 数据流状态机.md>) | 请求位置与资源状态分开看；该文固定 `882577451e`，不套用上面示例的轮次 |

### 版本深入与维护入口

- [PP 共识源码长文](<../并行方法/SGLang PP 共识机制源码学习文档.md>)：`72d5c5bb73`，覆盖普通 PP、PD 队列与 HiCache；适合查完整协议，不替换短篇的版本。
- [loop 源码复核记录](<PD Prefill PP loop 源码复核记录.md>)：记录图示从 `72d5c5bb73` 到 `279339f113` 的实际变化，供维护者回查。
- [历史 PD+PP 源码长文](<PD 分离下的 PP 源码学习文档.md>)：主体是历史 `muxi-main`，只有明确标注的补充节另用官方提交。先按文首路线选择章节。

## 对应源码课程

需要系统读代码时，从下面的课程入口衔接。课程与专题可能采用不同固定提交，阅读前分别核对文首基线。

- [先补 PD 三条通路](<../source-study/architecture/08-PD与Encoder分离的三条通路.md>)
- [完整 PD 源码主线](<../source-study/07-disaggregation/01-PD分离职责与端到端请求地图.md>)

## 自测与下一步

能分别追踪激活、KV 和控制消息，并解释准入、传输收尾、共识与资源退役的不同条件。

按目标继续：

- [v0.5.20：DSpark 与 PD、DCP 组合](<../版本研究/SGLang v0.5.20 缓存调度与 Serving 协同学习文档.md>)：版本参考；先读状态交接主线，再核对 draft/target 布局与 Responses API 限制。
- [集群服务与分离部署](<../../大模型推理/分布式服务/README.md>)
- [KV 传输与外部缓存接入](<../../vllm/分离部署/README.md>)
- [性能观察与调优案例](<../性能工程/README.md>)

## 本目录收录范围

收录 SGLang 的 PD / Encoder 分离、bootstrap、KV 传输、跨角色状态协调与退役。当前 PP 组合案例只是其中一条阅读支线，不限制后续后端与部署主题。

## 通信机制衔接

[SGLang 通信与传输机制](<../并行方法/SGLang 通信与传输机制学习文档.md>) · [交互课程](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/communication/)：通过消息交接、collective、KV 页映射与就绪 / 回收门槛区分各层责任。文章使用独立固定提交，不替换本目录资料的历史基线。

先用 [P/D 状态交接与执行轨道](../../pages/sglang/inference-overview/deployment.html#pd-handoff-title) 跟随一条请求，再用 [部署边界图](../../pages/sglang/inference-overview/deployment.html#handoff-boundaries) 分别判断 P 侧收尾和 D 侧就绪，再观察 [KV 持有与释放](../../pages/sglang/kv-memory/index.html)、[选批准入](../../pages/sglang/scheduling/index.html)。通信课程与生命周期页采用 `279339f113`；数据流状态机保留 `882577451e`。两侧条件之间的阅读连接不表示这些场景共享全局时钟，也不覆盖全部接收后端。

## 相关机制选读

[PD 与投机解码的状态交接](<../../大模型推理/高级生成/PD 与投机解码的状态交接、回滚及容量学习文档.md>)：跨引擎机制补充，先修投机流程；关注 Draft 初始化与回滚边界。

## PP 共识深入阅读

在三篇 PD 机制课程之后，进入 [PP 共识交互专题](../../pages/sglang/pp-consensus/index.html) · [源码学习文档](<../并行方法/SGLang PP 共识与缓存 ACK 机制学习文档.md>)。先修请求生命周期与数据流，重点理解 bootstrap 实际回流名单、release 本地复查，并与 HiCache ACK 的计数 / event 协调作区分；不重复完整 PP loop。
