# SGLang：高级生成与状态提交

[专题入口](../README.md) · [Wiki 首页](../../README.md)

本目录围绕约束生成、投机解码和分支状态，研究候选怎样成为有效输出，以及输出、KV 与模型状态如何保持一致。模型注册、量化与输入适配归 [model-support](../model-support/README.md)。

## 先修与主线

先理解 [Prefill / Decode 与 KV](../runtime/SGLang%20推理全景学习指南.md)、[请求运行时](../runtime/README.md)与[模型状态](../model-support/README.md)。

1. [模型状态与高级生成机制](SGLang%20模型状态与高级生成机制学习文档.md) · [交互课程](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/advanced-generation/)：用状态槽、验证列和词表筛选观察三种机制。
2. [Draft、Verify、Commit](../source-study/08-advanced-generation/03-投机解码的DraftVerifyCommit.md)：深入对象、采样、KV 账本与停止边界。
3. [结构化输出与 Grammar](../source-study/08-advanced-generation/01-结构化输出与Grammar状态.md)：规则编译、缓存模板、matcher 与调度队列。

## 按问题深入

- 候选来自哪里：[EAGLE 与 MTP](../source-study/08-advanced-generation/04-EAGLE与MTP的源码主线.md)、[DFlash / Ngram / 自适应投机](../source-study/08-advanced-generation/05-DFlashNgram与自适应投机.md)。
- 状态怎样分支与并行推进：[Beam Search](../source-study/08-advanced-generation/02-BeamSearch与请求分支状态.md)、[Overlap 与组合约束](../source-study/08-advanced-generation/06-投机中的OverlapKV与组合约束.md)。
- 混合模型有什么额外条件：[MLA、稀疏注意力与混合状态](../source-study/09-model-specialization/04-MLA稀疏注意力与混合状态模型.md)。

资料分别保留自己的固定源码版本，导航不表示已经验证所有功能组合。

## 自测与下一步

能区分草稿、验证行、正式输出与有效 KV，解释首个分歧之后为何不能继续提交，并说明规则进度为何属于单个请求。

下一步读 [性能分析](../performance-engineering/README.md)，通过测量比较草稿成本与接受收益；或进入 [模型适配](../model-support/README.md)核对实际模型的状态契约。
