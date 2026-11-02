# Engram 条件记忆概念图来源

- 原文：[《DeepSeek V4.1 Flash为什么“不再坚持Decoder-only”？》](https://mp.weixin.qq.com/s/QrFXRFBMuuiNoYgHrjH7Xg)，作者“崴和Gpt合砍83分”，公众号“崴什么”。页面发布时间 2026-09-10 20:08:00（Asia/Shanghai）；读取于 2026-11-02。
- 已完整读取正文及 6 张图片。本图为第 6 张，其余 5 张围绕 CED、计算预算及 Attention 演进，相关主线已有文档，本篇不重复保存。
- 保留原始 PNG，未修改、裁剪或重绘。图像为 881×352 RGBA；浅色底上文字可读，窄屏的关键含义由正文对象表及流程解释承载。
- 原图是作者的概念对照，未画出上下文 gate、表分片与等待条件。图中文字 196B 是 Engram 参数规模，不是 KV 容量或单步激活量；不能从左右布局推出性能倍数。
- 本篇另用 Mermaid 绘制固定源码的依赖图，明确区分发起 fetch、提交 wait、满足设备等待和实际消费；图不表达实测时间或完整 RDMA 协议。
- 对应正文：[Engram 条件记忆与 KV 分层缓存](<../../大模型推理/模型架构/Engram 条件记忆与 KV 分层缓存学习文档.md>)。

| 文件 | 原文图序 | 原图链接 | SHA-256 |
| --- | --- | --- | --- |
| [01-conditional-memory.png](01-conditional-memory.png) | 6 | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/q2micJwsGlP43xQcgarCzJXPxtBmtyibgsj1iboP6piaz4aMBZYOQwZw9Jtqhssd2JrmC5nQE1hoYehPrKicmNd0D6JKrSJTib4VZL3mC0PTDHJAs/640?wx_fmt=png&from=appmsg) | `e8de5b18559f2ca86688f2907902d8d17e610a213c3eb8b439c4be24f8bd67dc` |
