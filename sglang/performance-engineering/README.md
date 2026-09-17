# SGLang：性能观察与调优案例

[专题学习入口](<../README.md>) · [Wiki 首页](<../../README.md>)

先形成可解释的测量，再选择参数、缓存或计算优化。

本文是学习导航，分类与阅读顺序由整理者归纳。技术资料继续保留各自的来源、版本和验证边界；导航不表示已完成运行或性能复现。

## 先修知识

先确认以下基础；已经掌握的部分可以跳过：

- [请求、调度与执行主链](<../runtime/README.md>)
- [前缀缓存与分层存储](<../kv-cache/README.md>)

## 阅读顺序

按表中顺序建立整体认识，再选择深入或选读内容。目录内的归档顺序不构成所有文章都必须读完的要求。

| 顺序 | 资料 | 阅读定位 | 学习重点 |
| --- | --- | --- | --- |
| 1 | [SGLang Torch Profiler 与 Trace 性能分析](<SGLang Torch Profiler 与 Trace 性能分析学习文档.md>) | 起步 | 按模型结构、shape 和调用次数读取 Trace。 |
| 2 | [SGLang v0.5.16 24GB 显存调优案例](<SGLang v0.5.16 24GB 显存调优案例学习文档.md>) | 案例：显存 | 学习有限显存下的预算与单变量调优，不照抄版本参数。 |
| 3 | [SGLang GLM-5.2 NVFP4 推理优化案例](<SGLang GLM-5.2 NVFP4 推理优化案例学习文档.md>) | 案例：计算 | 结合投机、量化和并行基础，分析 NVFP4 模型优化的收益来源。 |

## 对应源码课程

需要系统读代码时，从下面的课程入口衔接。课程与专题可能采用不同固定提交，阅读前分别核对文首基线。

- [Benchmark 设计与指标](<../source-study/11-performance-engineering/01-Benchmark设计与指标口径.md>)
- [投机解码的提交语义](<../source-study/08-advanced-generation/03-投机解码的DraftVerifyCommit.md>)

## 自测与下一步

能区分 CPU 提交和 GPU 执行，写出包含负载、基线、单变量变化与适用范围的对照计划。

按目标继续：

- [模型适配与特殊执行机制](<../model-support/README.md>)
- [版本演进与组合边界](<../version-studies/README.md>)
- [性能工程与综合优化](<../../llm-inference/performance-engineering/README.md>)

## 本目录收录范围

收录 SGLang 的 Profiler、benchmark、性能诊断、调参方法和优化案例。文章带版本或模型名时，仍按性能问题归档。

后续文章先加入阅读表，标明先修、定位和学习重点。当形成多篇相互关联且需要独立导航的子主题时，再建立子目录；跨主题内容保留一份正文，由相关 README 交叉引用。
