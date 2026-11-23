# SGLang：性能观察与调优案例

[专题学习入口](<../README.md>) · [Wiki 首页](<../../README.md>)

先形成可解释的测量，再选择参数、缓存或计算优化。

各篇文章的来源、源码版本和验证范围见文首说明。阅读顺序可按自己的问题调整。

## 先修知识

先确认以下基础；已经掌握的部分可以跳过：

- [请求、调度与执行主链](<../运行时/README.md>)
- [前缀缓存与分层存储](<../KV缓存/README.md>)

## 指标与机制主线

先读 [SGLang 指标口径与性能定位](<SGLang 指标口径与性能定位学习文档.md>)，并操作 [交互课程](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/performance-engineering/)：P/D 指标采集位置、阶段统计窗口、流式回告与成功样本聚合。然后根据现象选择下面的 Trace 与调优案例。

## 阅读顺序

按表中顺序建立整体认识，再选择深入或选读内容。目录内的归档顺序不构成所有文章都必须读完的要求。

| 顺序 | 资料 | 阅读定位 | 学习重点 |
| --- | --- | --- | --- |
| 1 | [SGLang Torch Profiler 与 Trace 性能分析](<SGLang Torch Profiler 与 Trace 性能分析学习文档.md>) | 起步 | 按模型结构、shape 和调用次数读取 Trace。 |
| 2 | [SGLang v0.5.16 24GB 显存调优案例](<SGLang v0.5.16 24GB 显存调优案例学习文档.md>) | 案例：显存 | 学习有限显存下的预算与单变量调优，不照抄版本参数。 |
| 3 | [SGLang GLM-5.2 NVFP4 推理优化案例](<SGLang GLM-5.2 NVFP4 推理优化案例学习文档.md>) | 案例：计算 | 结合投机、量化和并行基础，分析 NVFP4 模型优化的收益来源。 |
| 4 | [SGLang Kimi K3 推理协同优化](<SGLang Kimi K3 推理协同优化学习文档.md>) | 案例：混合状态与并行 | 先补 KDA 与投机基础，再分析 DSpark、内核、PP/DCP 的配合及各项性能数字的条件。 |

## 对应源码课程

需要系统读代码时，从下面的课程入口衔接。课程与专题可能采用不同固定提交，阅读前分别核对文首基线。

- [Benchmark 设计与指标](<../source-study/11-performance-engineering/01-Benchmark设计与指标口径.md>)
- [投机解码的提交语义](<../source-study/08-advanced-generation/03-投机解码的DraftVerifyCommit.md>)

## 自测与下一步

能区分 CPU 提交和 GPU 执行，写出包含负载、基线、单变量变化与适用范围的对照计划。

按目标继续：

- [v0.5.20：模拟器与性能数字的适用范围](<../版本研究/SGLang v0.5.20 缓存调度与 Serving 协同学习文档.md>)：方法案例；读过指标基础后，比较模拟预测、实卡测试、算子耗时和完整请求指标。
- [模型适配与特殊执行机制](<../模型适配/README.md>)
- [版本演进与组合边界](<../版本研究/README.md>)
- [性能工程与综合优化](<../../大模型推理/性能工程/README.md>)

## 本目录收录范围

收录 SGLang 的 Profiler、benchmark、性能诊断、调参方法和优化案例。文章带版本或模型名时，仍按性能问题归档。

## 压测与 Trace 实验

先做[标准压测实验](../../pages/sglang/benchmark-lab/index.html)，再进入 [Trace 定位](../../pages/sglang/trace-lab/index.html)。前者保留负载、许可等待和成功样本，后者附带明确来源的历史 ResNet-50 实测片段并支持导入自己的记录；两者不构成 SGLang GPU 跑分。
