# 推理基础课程

本目录是 Pages 第一章，面向尚未掌握推理术语的读者。以可操作的教学算例建立直觉，不把通用概念归为 SGLang 专属机制。第三章的 SGLang 实现课各自保留固定源码、配置约束与历史基线。

## 三章连读

第一次阅读可从 [首页的 17 步入门路线](../index.html#reading-routes) 开始：先读本章 01–07 与 12，再对照请求路径、前缀缓存和调度，最后进入 SGLang 的六节单实例实现课。带有 `learn=beginner` 的链接使用同一条路线翻页；也可随时退出，按下方完整目录阅读。多卡、P/D、候选验证和性能课程按需要进入，不是走通一轮执行的前提。

[同一问题的三章索引](../index.html#topic-paths) 说明原理、引擎对照和源码分别增加什么。各课算例独立；第二章与第三章各保留自己的源码基线。第一章的 nearest-rank 百分位算例与第三章 benchmark 的线性插值分别标注，不把两个结果当成同一口径。

## 推荐阅读路线

1. [文字、token 与向量](tokens.html) → [Transformer 前向总览](components.html) → [Attention 小算例](attention.html) → [自回归生成](generation.html) → [KV 与内存](cache.html)。
2. [批处理与分块](batching.html)。
3. [GPU 计算与搬运](hardware.html) → [多卡分工](parallel.html) → [通信语义](communication.html) → [角色分离](deployment.html)。
4. 按需阅读 [候选验证](speculation.html)、[服务生命周期](serving.html)、[性能口径](metrics.html)。

每课列出先修、解释用到的词，并提供自测、无 JavaScript 阅读路径和对应实现入口。自测应回答图中当前问题：谁在计算、数据怎样变化、为什么等待、哪些事实尚未得到证明。

## 内容与来源

三章路线和问题映射维护在 `../learning-paths.json`；共享导航生成器检查基础先修顺序和 13 / 7 / 34 项覆盖，第二章另有 vLLM 独立导读。`courses.json` 是课程正文、先修关系、自测和来源的编辑入口；`build_foundations.py` 生成十三个 HTML。`extensions.json` 保存从旧概览课归并而来的六幅选读机制图及原说明；第三章只保留对应实现导读，不再嵌入基础播放器。选读区按需初始化，收起时停止播放。`model.js` 提供独立教学状态，`player.js` 按机制绘制不同图形。数字与边界写在各课，不能把教学轮次、资源下界或手工分数当成 GPU 观测。

第 02 课用 [Transformer 架构图](components.html#layer-atlas) 串起一次前向，维护入口为 `scripts/templates/layer-atlas-section.html.in` 与 `layer-atlas*.js`。同页 [单位置对照实验](components.html#output-chain) 由 `scripts/templates/output-chain-section.html.in` 与 `output-chain*.js` 维护，使用独立的二维单层模型。第 03 课聚焦可见范围、加权汇聚与多头，第 04 课跟踪选出的 token 怎样成为下一轮输入，并可选读各层 K/V 的写入时刻。

[采样参数与输出控制](generation.html#sampling-params)沿固定分数核对概率、候选集合、重复惩罚和停止条件；[量化数值直觉](hardware.html#quantization-intuition)将 scale、还原误差与元数据计入账本。维护入口为 `../interactive-lessons/lessons.json` 与 `build_priority_lessons.py`，由基础课生成器嵌入，不另建基础章节。

token 样例来自 Qwen/Qwen2.5-0.5B-Instruct 的 `7ae557604adf67be50417f59c2c2f167def9a775` 版本，使用 `tokenizers 0.23.3` 对原始文本执行 encode（不添加特殊 token）与 decode 核对。两组句子不应用聊天模板，也不运行模型。`tokenizer.json` SHA-256 为 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`。固定文件的公开来源列在第一课。

## 维护与验证

```bash
python3 -B scripts/build_foundations.py
python3 -B scripts/build_learning_navigation.py
python3 -B scripts/build_pages.py
python3 -B scripts/check_learning_docs.py
node tests/pages/test_foundations.cjs
```

构建会检查课程与导航生成一致性。浏览器检查使用 `tests/pages/test_foundations_browser.cjs`，通过环境变量提供浏览器、Playwright 模块与本地预览地址。逐页审核从首页三章目录进入，不需要改变页面 URL 或失效旧书签。
