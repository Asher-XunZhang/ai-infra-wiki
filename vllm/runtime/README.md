# vLLM：引擎工作流与调度

[专题学习入口](<../README.md>) · [Wiki 首页](<../../README.md>)

把调度迭代、计算份额与 KV 存储组织成一条请求链路。

本文是学习导航，分类与阅读顺序由整理者归纳。技术资料继续保留各自的来源、版本和验证边界；导航不表示已完成运行或性能复现。

## 先修知识

先确认以下基础；已经掌握的部分可以跳过：

- [推理基础](<../../llm-inference/foundations/README.md>)
- [调度方法](<../../llm-inference/scheduling/README.md>)

## 阅读顺序

按表中顺序建立整体认识，再选择深入或选读内容。目录内的归档顺序不构成所有文章都必须读完的要求。

| 顺序 | 资料 | 阅读定位 | 学习重点 |
| --- | --- | --- | --- |
| 1 | [vLLM 从连续批处理到 PagedAttention 的引擎工作流](<vLLM 从连续批处理到 PagedAttention 的引擎工作流学习文档.md>) | 起步 | 建立引擎、调度与 PagedAttention 的整体关系；概念示例含不同历史版本。 |
| 2 | [vLLM Chunked Prefill 与 Block Size](<vLLM Chunked Prefill 与 Block Size 学习文档.md>) | 主线 | 拆清 chunk、block 与每轮预算的联系。 |

## 自测与下一步

能解释连续批处理如何补入和移出请求，以及计算 chunk 为什么不等于存储 block。

按目标继续：

- [缓存身份与生命周期](<../kv-cache/README.md>)
- [多卡执行与通信](<../parallelism/README.md>)
- [请求、调度与执行主链](<../../sglang/runtime/README.md>)

## 本目录收录范围

收录 vLLM 的启动、请求、调度、Worker/Runner 与执行主链；后续相关机制按这一职责范围归档。

后续文章先加入阅读表，标明先修、定位和学习重点。当形成多篇相互关联且需要独立导航的子主题时，再建立子目录；跨主题内容保留一份正文，由相关 README 交叉引用。
