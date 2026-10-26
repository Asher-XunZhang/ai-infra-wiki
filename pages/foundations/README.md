# 推理基础课程

本目录是 Pages 第一章，面向尚未掌握推理术语的读者。以可操作的教学算例建立直觉，不把通用概念归为 SGLang 专属机制。第二章的 SGLang 实现课各自保留固定源码、配置约束与历史基线。

## 推荐阅读路线

1. [文字、token 与向量](tokens.html) → [逐步生成](generation.html) → [Attention 小算例](attention.html) → [Transformer 基础部件](components.html)。
2. [KV 与内存](cache.html) → [批处理与分块](batching.html)。
3. [GPU 计算与搬运](hardware.html) → [多卡分工](parallel.html) → [通信语义](communication.html) → [角色分离](deployment.html)。
4. 按需阅读 [候选验证](speculation.html)、[服务生命周期](serving.html)、[性能口径](metrics.html)。

每课列出先修、解释用到的词，并提供自测、无 JavaScript 阅读路径和对应实现入口。自测应回答图中当前问题：谁在计算、数据怎样变化、为什么等待、哪些事实尚未得到证明。

## 内容与来源

`courses.json` 是课程正文、先修关系、自测和来源的编辑入口；`build_foundations.py` 生成十三个 HTML。`extensions.json` 保存从旧概览课归并而来的六幅选读机制图及原说明；第二章只保留对应实现导读，不再嵌入基础播放器。选读区按需初始化，收起时停止播放。`model.js` 提供独立教学状态，`player.js` 按机制绘制不同图形。数字与边界写在各课，不能把教学轮次、资源下界或手工分数当成 GPU 观测。

token 样例来自 Qwen/Qwen2.5-0.5B-Instruct 的 `7ae557604adf67be50417f59c2c2f167def9a775` 版本，使用 `tokenizers 0.23.3` 对原始文本执行 encode（不添加特殊 token）与 decode 核对。两组句子不应用聊天模板，也不运行模型。`tokenizer.json` SHA-256 为 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`。固定文件的公开来源列在第一课。

## 维护与验证

```bash
python3 -B scripts/build_foundations.py
python3 -B scripts/build_learning_navigation.py
python3 -B scripts/build_pages.py
python3 -B scripts/check_learning_docs.py
node tests/pages/test_foundations.cjs
```

构建会检查课程与导航生成一致性。浏览器检查使用 `tests/pages/test_foundations_browser.cjs`，通过环境变量提供浏览器、Playwright 模块与本地预览地址。逐页审核从首页两章目录进入，不需要改变页面 URL 或失效旧书签。
