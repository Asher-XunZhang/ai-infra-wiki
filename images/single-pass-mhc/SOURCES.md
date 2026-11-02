# Single-Pass mHC 公式图来源

- 原文：[《Deepseek v4.1 flash 架构速览》](https://mp.weixin.qq.com/s/09VTlU5evR4awCjwCa5h0w)。作者 rainjoe，公众号“棍子上的萝卜”；发布时间 2026-10-30 15:50:07（Asia/Shanghai），读取时间 2026-11-02。
- 完整读取正文七节和全部四张图；第 1、2 图分别为 CED 架构与 CSA2 模式，已有专题覆盖，不重复保存。第 3、4 图对应原版 mHC 和 Single-Pass mHC 的公式，保留原始 PNG 字节，未重绘、裁剪或改写。
- 两张图均为不透明灰度 PNG，白底可读；宽度为 724/770 像素，窄屏公式会缩小，因此正文另用可维护文字和表格解释符号及依赖。
- 一手核验：[DeepSeek V4.1 技术报告 v1 §2.4.1](https://arxiv.org/html/2609.19969v1#S2.SS4.SSS1)，式 (2)–(6) 与局部激活读写量。图片展示数学依赖，不是 kernel trace 或质量实验结果；本文未验证第一 block 的初始化、实现调度或端到端性能。
- 正文：[混合注意力、索引压缩与残差流架构对比](<../../大模型推理/模型架构/混合注意力、索引压缩与残差流架构对比学习文档.md#single-pass-mhc提前准备读入权重消除一次残差重读>)。

| 文件 | 原文图序 | 尺寸 | 内容 | 原图链接 | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| [01-original-mhc.png](01-original-mhc.png) | 3 | 724×88 | 当前状态预测 A、B、C，当前 A 用于输入混合 | [原图](https://mmbiz.qpic.cn/mmbiz_png/mq4uHlAkNapssAW6kts26bGf2LfIicLkLAUFC6LYcdm4w2OOqIQUgDpYw81uzWOUkNN50YX3ick22au9hRNqvXA6SzpzYJ76nPYBv3XpXxqog/640?wx_fmt=png&from=appmsg) | `4147018c5cc1fb27d0702bdb20a24bced4efc6ad48597af8e9be7060fee360a9` |
| [02-shifted-input-mixing.png](02-shifted-input-mixing.png) | 4 | 770×92 | 使用上一 block 的 A 混合当前残差 | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/mq4uHlAkNarGBib6mAcPz9xkxGk3MDjEhia6Eyhzib3wKnFRuHibLD6AibFLRD5NOvQSzdxCIUM36O1UrFp2rgXuURjY6vDl183yubmYdMPTKdJM/640?wx_fmt=png&from=appmsg) | `d2c232fc8e9a25bdbf84f3d38d86a40108356fd1cb50f7ad2db68d5ea86de815` |
