# SGLang：分离部署与状态交接

[专题学习入口](<../README.md>) · [Wiki 首页](<../../README.md>)

以 PD 与 PP 的组合为切口，学习跨角色、跨 stage 的请求和资源协调。

本文是学习导航，分类与阅读顺序由整理者归纳。技术资料继续保留各自的来源、版本和验证边界；导航不表示已完成运行或性能复现。

初次接触 PD 时，先看 [推理全景：合并部署与 PD 分离](../../pages/sglang/inference-overview/deployment.html)，配合 [学习指南](../runtime/SGLang%20推理全景学习指南.md) 理解完整模型与状态交接，再进入本领域。

## 先修知识

先确认以下基础；已经掌握的部分可以跳过：

- [请求、调度与执行主链](<../runtime/README.md>)
- [前缀缓存与分层存储](<../kv-cache/README.md>)
- [并行拓扑与负载分工](<../parallelism/README.md>)

## 阅读顺序

先理解普通调度、缓存与 PP，再补下方 PD 三条通路。正文按“单请求 → 多 batch → 操作 → 共识”推进；复核记录与历史分支材料不计入必读主线。

| 顺序 | 资料 | 阅读定位 | 学习重点 |
| --- | --- | --- | --- |
| 1 | [PD Prefill PP=3 请求生命周期源码](<PD Prefill PP=3 请求生命周期源码学习文档.md>) | 起步 | 先跟一条请求走完 Prefill，再观察 chunk 与异常分支。 |
| 2 | [PD Prefill PP loop 交互图](<PD Prefill PP loop 交互图.md>) | 主线 | 转到多 batch 视角，理解 loop、槽位和依赖。 |
| 3 | [PD Prefill PP loop 步骤详解](<PD Prefill PP loop 步骤详解.md>) | 深入 | 逐项核对输入、动作、推进条件和源码入口。 |
| 4 | [SGLang PP 共识机制源码](<SGLang PP 共识机制源码学习文档.md>) | 深入 | 比较不同队列的共识规则与资源生命周期边界。 |
| 5 | [PD Prefill PP loop 源码复核记录](<PD Prefill PP loop 源码复核记录.md>) | 参考：复核 | 用于追溯图示基线和修正依据，不作为入门正文。 |
| 6 | [PD 分离下的 PP 源码](<PD 分离下的 PP 源码学习文档.md>) | 参考：历史 | 主体为历史 muxi-main 分析，独立补充章节另有官方基线；不能合并成同一版本。 |

[数据流状态机](<PD Prefill 数据流状态机.md>)作为交互对照材料，补充正常、等待、分块与失败场景中的状态和资源边界；固定开源 `882577451e`，与上表资料分别核对版本。

## 对应源码课程

需要系统读代码时，从下面的课程入口衔接。课程与专题可能采用不同固定提交，阅读前分别核对文首基线。

- [先补 PD 三条通路](<../source-study/architecture/08-PD与Encoder分离的三条通路.md>)
- [完整 PD 源码主线](<../source-study/07-disaggregation/01-PD分离职责与端到端请求地图.md>)

## 自测与下一步

能分别追踪激活、KV 和控制消息，并解释准入、传输收尾、共识与资源退役的不同条件。

按目标继续：

- [v0.5.20：DSpark 与 PD、DCP 组合](<../version-studies/SGLang v0.5.20 缓存调度与 Serving 协同学习文档.md>)：版本参考；先读状态交接主线，再核对 draft/target 布局与 Responses API 限制。
- [集群服务与分离部署](<../../llm-inference/distributed-serving/README.md>)
- [KV 传输与外部缓存接入](<../../vllm/disaggregation/README.md>)
- [性能观察与调优案例](<../performance-engineering/README.md>)

## 本目录收录范围

收录 SGLang 的 PD / Encoder 分离、bootstrap、KV 传输、跨角色状态协调与退役。当前 PP 组合案例只是其中一条阅读支线，不限制后续后端与部署主题。

后续文章先加入阅读表，标明先修、定位和学习重点。当形成多篇相互关联且需要独立导航的子主题时，再建立子目录；跨主题内容保留一份正文，由相关 README 交叉引用。
