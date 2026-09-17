# LLM 推理系统：模型结构与状态语义

[专题学习入口](<../README.md>) · [Wiki 首页](<../../README.md>)

理解模型结构怎样改变推理状态、缓存恢复和执行约束。

本文是学习导航，分类与阅读顺序由整理者归纳。技术资料继续保留各自的来源、版本和验证边界；导航不表示已完成运行或性能复现。

## 先修知识

先确认以下基础；已经掌握的部分可以跳过：

- [推理基础](<../foundations/README.md>)
- [KV Cache 方法与内存层次](<../kv-cache/README.md>)

## 阅读顺序

先用 Kimi K3 建立状态地图，再深入 KDA 检查点。涉及投机回滚时，可补读 SGLang 源码课程中的 Draft、Verify、Commit。

| 顺序 | 资料 | 阅读定位 | 学习重点 |
| --- | --- | --- | --- |
| 1 | [Kimi K3 混合注意力缓存与 Mooncake 状态传输](<Kimi K3 混合注意力缓存与 Mooncake 状态传输学习文档.md>) | 起步 | 先建立混合注意力、检查点和状态传输的整体图。 |
| 2 | [KDA Cache 检查点、前缀复用与投机回滚](<KDA Cache 检查点、前缀复用与投机回滚学习文档.md>) | 深入 | 进一步追踪前缀复用、投机回滚与状态提交；投机基础可补读源码课程。 |

投机基础：[Draft、Verify、Commit](<../../sglang/source-study/08-advanced-generation/03-投机解码的DraftVerifyCommit.md>)。

## 自测与下一步

能说明 token 前缀长度为什么不等于合法状态恢复边界，并区分共享状态和请求私有状态。

按目标继续：

- [模型适配与特殊执行机制](<../../sglang/model-support/README.md>)
- [前缀缓存与分层存储](<../../sglang/kv-cache/README.md>)
- [集群服务与分离部署](<../distributed-serving/README.md>)

## 本目录收录范围

收录 Attention、MoE、混合状态等模型结构对推理计算、容量和恢复语义的影响；具体模型的引擎适配进入对应引擎专题。

后续文章先加入阅读表，标明先修、定位和学习重点。当形成多篇相互关联且需要独立导航的子主题时，再建立子目录；跨主题内容保留一份正文，由相关 README 交叉引用。
