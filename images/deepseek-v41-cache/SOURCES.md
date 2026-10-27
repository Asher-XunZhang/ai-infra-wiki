# DeepSeek V4.1 Flash 架构与缓存图源

- 原文：[DeepSeek V4.1 Flash：架构学习笔记](https://mp.weixin.qq.com/s/a0syC_KNq49t0jZT-90-Rw)，薛定谔的爆米花，2026-09-11；读取于 2026-10-27。
- 原图来自 DeepSeek-AI 技术报告，本文保留文章中全部 4 张技术配图，原样保存，无重绘。版权归原作者。
- 对照 [arXiv v1](https://arxiv.org/html/2609.19969v1) 的 Figures 2–5 核对图意与标题；未声称文章所用 PDF 与 arXiv v1 文件字节一致。
- 原图为白色背景，中文含义和窄屏阅读所需的关键信息另以正文表格解释；第四张是精度加权 FLOPs，不是实测性能曲线。
- 对应正文：[CED、CSA2 与缓存分工](<../../大模型推理/模型架构/DeepSeek V4.1 Flash 的 CED、CSA2 与缓存分工学习文档.md>)。

| 本地文件 | 报告图号 | 尺寸 | 原文图片地址 | SHA-256 |
| --- | --- | --- | --- | --- |
| [01-ced-architecture.png](01-ced-architecture.png) | Figure 3 | 1080×809 | [原图](https://mmbiz.qpic.cn/mmbiz_png/TeGEb3s7mCocWM2DuEjDCVlkMNcz1tdAXFk6z386Jia7MibGwKU1sjdpagnr2LQYyb328JOMeYBpiblDXEp1jVrogb9oOAyNAFTIqGFoySZREM/640?wx_fmt=png&from=appmsg) | `7006a004bf87055b14cc6a6e1d772deb8360df04e600b20c4d60b796d2cacc3a` |
| [02-csa2-modes.png](02-csa2-modes.png) | Figure 4 | 1000×495 | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/TeGEb3s7mCpGtMZEfiatiamvBUk2pViaTtofEuuWvuY5Qm3sOgKaUFqkqEX4O4Pa9A7Ub45cosDkdFGSyoyhGcFibgE74mibTEtox1oIq1Lx5Dqk/640?wx_fmt=png&from=appmsg) | `55ed4135b6bb38532409bac15b11f3fc7a21b880ce24be3b9b336747d37277df` |
| [03-hierarchical-indexer.png](03-hierarchical-indexer.png) | Figure 5 | 1000×614 | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/TeGEb3s7mCpH454eweLtVM9ZROJp8VMiaTlh32ytQric9iaEZAk2teYibdc3cwicwibUyzVJ0Lq1CLFcstuiak4udevtpZM41GaqGUTiagbnefz9l4w/640?wx_fmt=png&from=appmsg) | `f5d6e67eb630f7368defb40b193430d3652665f8ce28a4f3c157d61ff787b171` |
| [04-decode-flops.png](04-decode-flops.png) | Figure 2 | 1000×462 | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/TeGEb3s7mCp2j5NmqPcudL0UVCHT0ZuIvtqXf7u6RrEyoJtpSjNK8SrZgjCraQGo9P9nfIksa0n62IAStpKic1CcGjeNaKibSffxPAia4LicSpM/640?wx_fmt=png&from=appmsg) | `ce19180df221a65dfce1cb737337648bcc05a508740304e3aa0cd016ca16c6a8` |
