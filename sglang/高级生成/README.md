# SGLang：高级生成与状态提交

[专题入口](../README.md) · [Wiki 首页](../../README.md)

本目录围绕约束生成、投机解码和分支状态，研究候选怎样成为有效输出，以及输出、KV 与模型状态如何保持一致。模型注册、量化与输入适配归 [model-support](../模型适配/README.md)。

## 先修与主线

先理解 [Prefill / Decode 与 KV](../%E8%BF%90%E8%A1%8C%E6%97%B6/SGLang%20%E6%8E%A8%E7%90%86%E5%85%A8%E6%99%AF%E5%AD%A6%E4%B9%A0%E6%8C%87%E5%8D%97.md)、[请求运行时](../运行时/README.md)与[模型状态](../模型适配/README.md)。

1. [模型状态与高级生成机制](SGLang%20%E6%A8%A1%E5%9E%8B%E7%8A%B6%E6%80%81%E4%B8%8E%E9%AB%98%E7%BA%A7%E7%94%9F%E6%88%90%E6%9C%BA%E5%88%B6%E5%AD%A6%E4%B9%A0%E6%96%87%E6%A1%A3.md) · [交互课程](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/advanced-generation/)：用状态槽、验证列和词表筛选观察三种机制。
2. [Draft、Verify、Commit](../source-study/08-advanced-generation/03-投机解码的DraftVerifyCommit.md)：深入对象、采样、KV 账本与停止边界。
3. [结构化输出与 Grammar](../source-study/08-advanced-generation/01-结构化输出与Grammar状态.md)：规则编译、缓存模板、matcher 与调度队列。

## 按问题深入

- 候选来自哪里：[EAGLE 与 MTP](../source-study/08-advanced-generation/04-EAGLE与MTP的源码主线.md)、[DFlash / Ngram / 自适应投机](../source-study/08-advanced-generation/05-DFlashNgram与自适应投机.md)。
- 状态怎样分支与并行推进：[Beam Search](../source-study/08-advanced-generation/02-BeamSearch与请求分支状态.md)、[Overlap 与组合约束](../source-study/08-advanced-generation/06-投机中的OverlapKV与组合约束.md)。
- 混合模型有什么额外条件：[MLA、稀疏注意力与混合状态](../source-study/09-model-specialization/04-MLA稀疏注意力与混合状态模型.md)。

资料分别保留自己的固定源码版本，导航不表示已经验证所有功能组合。

## 自测与下一步

能区分草稿、验证行、正式输出与有效 KV，解释首个分歧之后为何不能继续提交，并说明规则进度为何属于单个请求。

下一步读 [性能分析](../性能工程/README.md)，通过测量比较草稿成本与接受收益；或进入 [模型适配](../模型适配/README.md)核对实际模型的状态契约。

## 相关机制选读

[PD 与投机解码的状态交接](<../../大模型推理/高级生成/PD 与投机解码的状态交接、回滚及容量学习文档.md>)：组合专题，先修 Draft/Verify/Commit；公开问题与设计提案保留各自版本。
