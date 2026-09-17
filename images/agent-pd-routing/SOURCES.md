# 多轮路由与跨站 Prefill图片与来源记录

对应文档：[学习文档](<../../llm-inference/distributed-serving/多轮 Agent 的 PD 路由与跨数据中心 Prefill 学习文档.md>)。读取日期：2026-09-14。

- 原文：[Agent 时代，PD 分离裂开三道缝：AMPD、PPD、PrfaaS 怎么补](https://mp.weixin.qq.com/s/iOudrmkN-4844Is9F3ZC6g)
- 作者：HyperAI；发布时间：2026-08-31T08:49:47+08:00。
- 已定位并完整提取 `#js_content`，标准库 parser 与 lxml 独立检查正文、末尾和图片数量；排除验证页。正文结构化文本 2140 字符，HTML 3455492 字节。
- 原始 HTML SHA-256：`be72778afe2a5970aa75c3251bca8991159efd084d6c9711b7acf75cdd783364`。
- 结构化正文 SHA-256：`39f1f42cb38fa1beabbfd38cacf484015d890b914ce3296d8ac1773af72dd12e`。
- 原始整页 HTML 与抓取脚本只保留在临时目录；仓内保留原创笔记与用于解读的关键原图。
- 原文没有技术位图、SVG 或 CSS 背景图；补充图来自所引用论文 v1。
- 图片逐张检查了含义、尺寸和白色背景可读性；保持下载字节不变。窄屏下复杂图需点击放大，正文提供完整机制解释；AMPD 原图只有 446×415，保留原图并明确分辨率限制。

| 文件 | 原始来源 | 像素尺寸 | SHA-256 |
| --- | --- | --- | --- |
| [01-ampd-overview.png](01-ampd-overview.png) | [论文原图](https://arxiv.org/html/2602.14516v1/system_overview.png) | 446×415 | `d895151c987f4f5369912216112e47b7267a5f8237fa2b2fd9f5d9c8488bcc05` |
| [02-append-interference.png](02-append-interference.png) | [论文原图](https://arxiv.org/html/2603.13358v1/figures/interference_tpot.png) | 1320×1020 | `740235207d51f674baaaa384f2a3da2c86ad3472a00ae744089b1f7d5405e415` |
| [03-prfaas-topology.png](03-prfaas-topology.png) | [论文原图](https://arxiv.org/html/2604.15039v1/paas_pd_architecture.png) | 1273×760 | `8a78db4cd032ff46180ed5404cd90116a6eaf86d771a7545349f429e73486ef5` |
