# vLLM：多卡执行与通信

[专题学习入口](<../README.md>) · [Wiki 首页](<../../README.md>)

根据模型层、专家和上下文的切分对象选择阅读支线。

本文是学习导航，分类与阅读顺序由整理者归纳。技术资料继续保留各自的来源、版本和验证边界；导航不表示已完成运行或性能复现。

## 先修知识

先确认以下基础；已经掌握的部分可以跳过：

- [引擎工作流与调度](<../runtime/README.md>)
- [缓存身份与生命周期](<../kv-cache/README.md>)

## 阅读顺序

PP、EP、DCP 是三条按需求选择的支线，不是互相必修的三级课程。DCP 先补通用 CP 总览；EP 先确认模型的 MoE 基础。

| 顺序 | 资料 | 阅读定位 | 学习重点 |
| --- | --- | --- | --- |
| 1 | [vLLM Pipeline Parallel 流水线并行](<vLLM Pipeline Parallel 流水线并行学习文档.md>) | 选读：PP | 沿层切分和 stage 间中间张量理解请求流水。 |
| 2 | [vLLM Expert Parallel 与 EPLB](<vLLM Expert Parallel 与 EPLB 学习文档.md>) | 选读：EP | 沿 MoE token 分发、专家计算和结果合并理解 EPLB。 |
| 3 | [vLLM DCP KV Cache 去重与 LSE 合并](<vLLM DCP KV Cache 去重与 LSE 合并学习文档.md>) | 选读：DCP | 先补 CP 总览，再研究 KV 去重和 LSE 合并。 |

补充先修：[Context Parallel、PCP 与 DCP 总体](<../../llm-inference/parallelism/Context Parallel、PCP 与 DCP 总体学习文档.md>)。

## 自测与下一步

能分别解释 PP 的层切分、EP 的 dispatch/combine 和 DCP 的 KV 分片，不混淆它们的收益。

按目标继续：

- [KV 传输与外部缓存接入](<../disaggregation/README.md>)
- [性能分析与执行优化](<../performance-engineering/README.md>)
- [并行拓扑与负载分工](<../../sglang/parallelism/README.md>)

## 本目录收录范围

收录 vLLM 的 TP/PP/DP/EP/CP 拓扑、通信与负载机制。多卡不是一条固定路线，各并行方式按切分对象组织。

后续文章先加入阅读表，标明先修、定位和学习重点。当形成多篇相互关联且需要独立导航的子主题时，再建立子目录；跨主题内容保留一份正文，由相关 README 交叉引用。
