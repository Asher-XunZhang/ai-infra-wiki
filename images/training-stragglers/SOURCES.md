# 训练infra踩坑：万卡集群为何变慢？：图片来源与提取记录

读取时间：2026-10-08。关联[学习文档](<../../llm-inference/performance-engineering/3D 并行训练的慢 Rank 定位与通信等待传播学习文档.md>)。

- 来源：[原文](https://mp.weixin.qq.com/s/2Y9W9U2fwgabKZjP8zkK8A)，作者 小白玩推理，网页发布时间 2026-09-29T22:20:22+08:00。
- 使用完整桌面浏览器 User-Agent 与 scene=1 取得公开 HTML；按 js_content 提取，标准库 HTMLParser 与 lxml 独立核对图片数量及正文结尾。自闭合 br、img 按 void elements 处理，避免提前截断。
- HTML SHA-256：a182f9d5641f1d1afa43136e796cac7834e7bf37fa7306968538ddfed64237c6。临时抓取材料不入仓。
- 原文两图均为技术资料。第一张实际格式 JPEG，第二张为 PNG，按内容格式保存原字节。截图中的 HCOM 名称与原文未提供的硬件型号分开处理。
- 原图已逐张查看，正文均有原创图意解读；保留水印与来源。图像说明不等于实验复现。

| 文件 | 下载来源 | SHA-256 |
| --- | --- | --- |
| [01-dp-communication-matrix.jpg](01-dp-communication-matrix.jpg) | [正文图 1](https://mmbiz.qpic.cn/mmbiz_jpg/bnRkHqDib9pAAIMEvg2Y4V0OBH5VpIqv0Q3W7ALxMeIsg3qBicreUx9lQEBg5XFyEapJuBOhU6n12ZwLWa1P9xoosd4d3AkYNO9z8lA7yuul8/640?wx_fmt=jpeg&from=appmsg#imgIndex=1) | 03093c5cfc74e35f88ac28d7a8377a4173af33ac1ff29641b44076dd4ce447dd |
| [02-allreduce-wait-trace.png](02-allreduce-wait-trace.png) | [正文图 2](https://mmbiz.qpic.cn/mmbiz_png/bnRkHqDib9pDuQnMpA8UuU7kNyz9BUaXhlQJKJuhiaZFKFiajwPS8Rfc6AtZDBlEdb6uLoezWSKSBOpqHkxVKh1L3edOrVU7DAibu9Z2nTFXqeE/640?wx_fmt=png&from=appmsg#imgIndex=2) | 84a469bdc46bbc423f9cbb7fa6076f0ae692c0e39411a57297b08ff8210a2b28 |
