# SGLang：请求、调度与执行主链

[专题学习入口](<../README.md>) · [Wiki 首页](<../../README.md>)

沿一条请求理解 Scheduler 的决策、资源预算与执行提交。

本文是学习导航，分类与阅读顺序由整理者归纳。技术资料继续保留各自的来源、版本和验证边界；导航不表示已完成运行或性能复现。

## 先修知识

先确认以下基础；已经掌握的部分可以跳过：

- [推理基础](<../../llm-inference/foundations/README.md>)
- [调度方法](<../../llm-inference/scheduling/README.md>)

## 阅读顺序

初学者先用推理全景建立宏观认识；首轮再读总览和请求生命周期，再读显存预算。CUDA Graph 是执行支线，先补下方 Worker / ModelRunner 与 Graph 课程入口。

| 顺序 | 资料 | 阅读定位 | 学习重点 |
| --- | --- | --- | --- |
| 起步 | [SGLang 推理全景学习指南](<SGLang 推理全景学习指南.md>) · [交互专题](../../pages/sglang/inference-overview/journey.html) | 零基础入口 | 从 P/D、Transformer 与 KV 走到部署差异、调度和延迟。 |
| 1 | [SGLang 调度机制总览与学习路线](<SGLang 调度机制总览与学习路线.md>) | 起步 | 先区分路由、排序、准入和执行时序。 |
| 2 | [SGLang 调度器请求生命周期与重叠调度](<SGLang 调度器请求生命周期与重叠调度学习文档.md>) | 主线 | 从普通循环走到 Overlap、FutureMap 和结果处理。 |
| 3 | [SGLang Chunked Prefill 与调度器显存预算](<SGLang Chunked Prefill 与调度器显存预算学习文档.md>) | 深入 | 理解 chunk 额度与请求、KV、显存预算的共同约束。 |
| 4 | [SGLang Breakable CUDA Graph 与 Prefill 捕获](<SGLang Breakable CUDA Graph 与 Prefill 捕获学习文档.md>) | 选读：执行 | 掌握执行分层和 CUDA Graph 基础后，再读动态形状与 Prefill 捕获。 |

## 对应源码课程

需要系统读代码时，从下面的课程入口衔接。课程与专题可能采用不同固定提交，阅读前分别核对文首基线。

- [运行环境与最小请求](<../source-study/01-getting-started/01-源码目录与最短阅读路线.md>)
- [请求与 batch 对象](<../source-study/02-request-lifecycle/01-API协议到内部请求对象.md>)
- [调度主循环](<../source-study/03-scheduling/01-NormalEventLoop与调度主循环.md>)
- [Worker 与 ModelRunner](<../source-study/05-model-execution/01-Worker与ModelRunner执行边界.md>)
- [CUDA Graph 与执行模式](<../source-study/05-model-execution/05-CUDAGraph编译与执行模式.md>)

## 自测与下一步

能追踪 waiting、EXTEND、running 与结束释放，并区分排序、准入、Overlap 和 Graph 的作用。

按目标继续：

- [v0.5.20：HRRN、Graph 与采样协同](<../version-studies/SGLang v0.5.20 缓存调度与 Serving 协同学习文档.md>)：版本案例；掌握调度和执行主线后，比较实例内排序、实例路由和采样同步。
- [前缀缓存与分层存储](<../kv-cache/README.md>)
- [并行拓扑与负载分工](<../parallelism/README.md>)
- [性能观察与调优案例](<../performance-engineering/README.md>)

## 本目录收录范围

收录 SGLang 的启动、请求、调度、Worker/Runner 与执行提交主链，包括 CUDA Graph 等执行机制。实例外服务治理按需归 serving-operations。

后续文章先加入阅读表，标明先修、定位和学习重点。当形成多篇相互关联且需要独立导航的子主题时，再建立子目录；跨主题内容保留一份正文，由相关 README 交叉引用。
