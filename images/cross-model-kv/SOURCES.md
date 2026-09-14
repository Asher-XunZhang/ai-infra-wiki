# 跨模型 KV 转换图片与来源记录

对应文档：[学习文档](<../../llm-inference/跨模型 KV Cache 转换与 Prefill 复用学习文档.md>)。读取日期：2026-09-14。

- 原文：[NVIDIA 论文解读：KV Cache 跨模型转换](https://mp.weixin.qq.com/s/YTrcAvoBxp7eNCLODmOzWQ?scene=1&click_id=1)
- 作者：Realtime AI Lab；发布时间：2026-09-09T22:02:00+08:00。
- 已定位并完整提取 `#js_content`，标准库 parser 与 lxml 独立检查正文、末尾和图片数量；排除验证页。正文结构化文本 7110 字符，HTML 3609824 字节。
- 原始 HTML SHA-256：`2718084733582fa37293e716d0941f337a7f952911b5b9a3ea40d562bbedfdcd`。
- 结构化正文 SHA-256：`dd5cb9d60ce26959c2ba1993a1a8584270ec3dedba760458b647e99a391b60bf`。
- 原始整页 HTML 与抓取脚本只保留在临时目录；仓内保留原创笔记与用于解读的关键原图。
- 原文没有技术位图、SVG 或 CSS 背景图；补充图来自所引用论文 v1。
- 图片逐张检查了含义、尺寸和白色背景可读性；保持下载字节不变，未 AI 重绘。窄屏下复杂图需点击放大，正文提供文字解释。

| 文件 | 原始来源 | 像素尺寸 | SHA-256 |
| --- | --- | --- | --- |
| [01-pipeline.png](01-pipeline.png) | [论文原图](https://arxiv.org/html/2608.03893v1/figures/pipeline_overview.drawio.png) | 881×190 | `ab35d67a250ebb223f0357624c1a6d633b54ac25b1e7cc3ae4b0ac37a40ebe1e` |
| [02-mapper.png](02-mapper.png) | [论文原图](https://arxiv.org/html/2608.03893v1/figures/mapper_architecture.drawio.png) | 1031×370 | `28c01219f47fe212bbecbfa05e8346aa23219d87e6b84876579bef8a12669a34` |
| [03-latency.png](03-latency.png) | [论文原图](https://arxiv.org/html/2608.03893v1/figures/latency_vs_seqlen.png) | 1500×900 | `25dc9a915a3dcbcdb80e2d77357df1f172a63b6fe269e5b5e554eeeca4987766` |
