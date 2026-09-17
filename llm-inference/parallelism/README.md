# LLM 推理系统：并行方法

[专题学习入口](<../README.md>) · [Wiki 首页](<../../README.md>)

先辨认切分对象，再理解通信、负载和容量之间的关系。

本文是学习导航，分类与阅读顺序由整理者归纳。技术资料继续保留各自的来源、版本和验证边界；导航不表示已完成运行或性能复现。

## 先修知识

先确认以下基础；已经掌握的部分可以跳过：

- [推理基础](<../foundations/README.md>)
- [KV Cache 方法与内存层次](<../kv-cache/README.md>)

## 阅读顺序

按表中顺序建立整体认识，再选择深入或选读内容。目录内的归档顺序不构成所有文章都必须读完的要求。

| 顺序 | 资料 | 阅读定位 | 学习重点 |
| --- | --- | --- | --- |
| 1 | [Context Parallel、PCP 与 DCP 总体](<Context Parallel、PCP 与 DCP 总体学习文档.md>) | 起步 | 建立 CP、PCP、DCP 的整体坐标系。 |
| 2 | [PCP 长上下文 Prefill 并行](<PCP 长上下文 Prefill 并行学习文档.md>) | 深入 | 追踪长上下文 Prefill 的 token 分片与注意力计算。 |

## 自测与下一步

能解释 PCP 与 DCP 切分什么，以及局部 Attention 结果为什么需要合并。

按目标继续：

- [并行拓扑与负载分工](<../../sglang/parallelism/README.md>)
- [多卡执行与通信](<../../vllm/parallelism/README.md>)
- [集群服务与分离部署](<../distributed-serving/README.md>)

## 本目录收录范围

收录 TP、PP、DP、EP、CP 等切分方法、通信关系与组合原则。具体引擎的实现进入相应引擎专题。

后续文章先加入阅读表，标明先修、定位和学习重点。当形成多篇相互关联且需要独立导航的子主题时，再建立子目录；跨主题内容保留一份正文，由相关 README 交叉引用。
