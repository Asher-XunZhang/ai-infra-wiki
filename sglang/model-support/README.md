# SGLang：模型适配与特殊执行机制

[专题学习入口](<../README.md>) · [Wiki 首页](<../../README.md>)

理解模型特性如何约束缓存、量化、生成和分布式执行。

本文是学习导航，分类与阅读顺序由整理者归纳。技术资料继续保留各自的来源、版本和验证边界；导航不表示已完成运行或性能复现。

## 先修知识

先确认以下基础；已经掌握的部分可以跳过：

- [请求、调度与执行主链](<../runtime/README.md>)
- [前缀缓存与分层存储](<../kv-cache/README.md>)
- [模型结构与状态语义](<../../llm-inference/model-architecture/README.md>)

## 阅读顺序

模型专题按目标模型选择。当前案例涉及混合状态与投机提交，建议先读下方投机基础；后续模型适配、量化与多模态资料也按此领域归档。

| 顺序 | 资料 | 阅读定位 | 学习重点 |
| --- | --- | --- | --- |
| 1 | [SGLang Kimi K3 推理协同优化](<SGLang Kimi K3 推理协同优化学习文档.md>) | 案例：混合状态 | 沿 KDA、DSpark、PP 与 DCP 理解协同约束，先补投机基础再读提交细节。 |

## 对应源码课程

需要系统读代码时，从下面的课程入口衔接。课程与专题可能采用不同固定提交，阅读前分别核对文首基线。

- [模型注册与权重加载](<../source-study/09-model-specialization/01-模型注册配置与权重加载.md>)
- [投机解码的 Draft、Verify、Commit](<../source-study/08-advanced-generation/03-投机解码的DraftVerifyCommit.md>)

## 自测与下一步

能把模型结构、状态提交、计算格式和并行方式分开解释，而不是把模型支持当作单一开关。

按目标继续：

- [性能观察与调优案例](<../performance-engineering/README.md>)
- [版本演进与组合边界](<../version-studies/README.md>)

## 本目录收录范围

收录 SGLang 的模型注册与适配、量化计算路径、多模态和特殊状态执行约束。只以模型为例讨论测量和瓶颈的文章归性能工程。

后续文章先加入阅读表，标明先修、定位和学习重点。当形成多篇相互关联且需要独立导航的子主题时，再建立子目录；跨主题内容保留一份正文，由相关 README 交叉引用。

## 状态与生成衔接

[模型状态与高级生成机制](<../advanced-generation/SGLang 模型状态与高级生成机制学习文档.md>) · [交互课程](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/advanced-generation/)：观察历史状态、候选提交和规则进度，再进入本目录模型案例。
