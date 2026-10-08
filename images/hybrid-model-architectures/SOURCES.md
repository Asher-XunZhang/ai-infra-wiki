# 面试官：GLM/Kimi/DS/Qwen架构有什么区别？：图片来源与提取记录

读取时间：2026-10-08。关联[学习文档](<../../llm-inference/model-architecture/混合注意力、索引压缩与残差流架构对比学习文档.md>)。

- 来源：[原文](https://mp.weixin.qq.com/s/N7TbIq01pMbJ0VtqyH5tBg)，作者 panda，网页发布时间 2026-09-16T15:13:36+08:00。
- 使用完整桌面浏览器 User-Agent 与 scene=1 取得公开 HTML；按 js_content 提取，标准库 HTMLParser 与 lxml 独立核对图片数量及正文结尾。自闭合 br、img 按 void elements 处理，避免提前截断。
- HTML SHA-256：1af1059a61395140df6711f1fd8d2eb98e552a0f6aa95eeb3e93d518ea76e134。临时抓取材料不入仓。
- 原文共有 5 张位图，前 4 张为技术图，第 5 张为作者交流二维码，未入库。原链接的 wx_fmt=webp 不等于实际编码；下载的四图均为 JPEG，保存原字节。原文另有 22 个 MathJax SVG 公式，已结合节点字符、上下标结构和一手报告理解，关键表达改为正文公式；不将其误判为装饰。
- 原图已逐张查看，正文均有原创图意解读；保留水印与来源。图像说明不等于实验复现。

| 文件 | 下载来源 | SHA-256 |
| --- | --- | --- |
| [01-glm-indexpool.jpg](01-glm-indexpool.jpg) | [正文图 3](https://mmbiz.qpic.cn/sz_mmbiz_jpg/J3xeWzJT3Ncsn4AUpgcKtDaUMibtzIQVOOStn0C0hZc04yOG9gmUzicSiavNevSOIib3vkkGRibQrbzfcLr5Do1R9zWrZuZ3udx2vGXlnfmLA32o/640?wx_fmt=webp&from=appmsg) | 4efe030dc35322362d2c062ee52f3e1666b7adaff46e599662d432d349627393 |
| [02-deepseek-csa-hca.jpg](02-deepseek-csa-hca.jpg) | [正文图 1](https://mmbiz.qpic.cn/mmbiz_jpg/J3xeWzJT3NdejnwVicIUsR7TIEj7WollvjFicj2ia1sqxPBb4LVSprzxMjVeibq0SckVichvSK9ViapOYr7uwOiaWXtxg8liaclkOQib22Adq1P9c3gI/640?wx_fmt=webp&from=appmsg) | dd669a21f4308e68d169b0e7277ecbb1f8a03f07d7908dab111a1572e52cfb65 |
| [03-qwen-gdn-qsa-gr.jpg](03-qwen-gdn-qsa-gr.jpg) | [正文图 2](https://mmbiz.qpic.cn/sz_mmbiz_jpg/J3xeWzJT3NfgjiaL8tiatoMHOahtVgvricXp084B2t4VCUFYENzh3B66tfeC2dHibYymjZnVZP0b5OLAzNtIlIm74LT9KJuYykBLEibymU9icjW9g/640?wx_fmt=webp&from=appmsg) | de92a7e85cc8cb00255fc1081042146a9ad83af0a206b4e80c19846a82a9c7e9 |
| [04-kimi-kda-mla-attnres.jpg](04-kimi-kda-mla-attnres.jpg) | [正文图 4](https://mmbiz.qpic.cn/mmbiz_jpg/J3xeWzJT3Ndq2nH0GIiaQ0gSBUrIX3ha36F7gmJib0XV4vrVEUUBv5M3lN3OuNeMhZS3cCnKJLpiaAjShVzV4Hdyp94y3BvzwYXLjmfQaHC6mM/640?wx_fmt=webp&from=appmsg) | 93c926eb50c1180c2a3cd5e720864b8e128cff3bad3b49e9be12860e004341eb |
