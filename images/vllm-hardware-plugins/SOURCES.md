# vLLM 硬件插件配图来源

- 原文：[vLLM 如何识别并运行在国产 GPU 上？](https://mp.weixin.qq.com/s/CsX8BdUvlVh_Y1DqZcuW_A)，张量叙事。
- 发布时间：2026-11-19 09:49:30，Asia/Shanghai；读取日期：2026-11-19。
- 完整查看正文 5 图。题图与执行层次图信息重复，不收录；保留正文 4 张技术图，按页面提供的正文资源 URL 下载，未重绘、裁剪或修改。
- 这些图是原文的教学示意，没有固定源码版本，也不是进程拓扑或实测 GPU 时间线。平台注册、general callbacks 与运行时补丁的实际顺序及条件，见正文的固定源码复核。
- 没有性能图；不从示意图推断设备性能或功能验证结果。

| 文件 | 图意 | 来源资源 | SHA-256 |
| --- | --- | --- | --- |
| [01-plugin-discovery.png](01-plugin-discovery.png) | 插件发现与注册示意 | [原文资源](https://mmbiz.qpic.cn/mmbiz_png/KlV8ppcRQBS2u2ZxqqWshnbqeYdicuw97f0lGetcCt4UqDBEQ3mMVzNXeviagubiaXT5mQIic8tV39Mm91YHMjUiaaIC4ibksxaYEmssVF0JLsfo0/640?wx_fmt=png&from=appmsg) | `39346d67c85dd4577b83610f19289569786cc19c8ff4ef2d454c86193090892d` |
| [02-execution-stack.png](02-execution-stack.png) | 从调度到设备执行的逻辑层次 | [原文资源](https://mmbiz.qpic.cn/mmbiz_png/KlV8ppcRQBTD1exOBDD5hDC3r9jiaWiaHOA4NDaiaUN2zMndiar9Yzhcw2ZcWxrKPeNzEvaaLxeeH4eFhtxp1Tybuko0uMribep5Om7snNQ20ibCg/640?wx_fmt=png&from=appmsg) | `1ff7aaa516a63287feccd3bbb0629ac88edf4df7999d49021080847faa53148f` |
| [03-hcu-runtime.png](03-hcu-runtime.png) | HCU 插件、Worker/Runner 与 Runtime Patch | [原文资源](https://mmbiz.qpic.cn/mmbiz_png/KlV8ppcRQBSjWeMficppcaZaFZGq5WoA6q8XjnibafhHnRticydiaKqvjQAhNicxWX90ribfMn4NlsqgOdSZnDdBTr5HIaq1JHeLPiauMNJkJI5Iw4/640?wx_fmt=png&from=appmsg) | `5ed5cd9d7796a70bf077a0aa0df056a6394f61fdbb6727f1729350bda51a9de8` |
| [04-hcu-ascend-comparison.png](04-hcu-ascend-comparison.png) | HCU 与 Ascend 的逻辑接入路线 | [原文资源](https://mmbiz.qpic.cn/mmbiz_png/KlV8ppcRQBSGDq81T71n5r3a7ibK1LwgA9UFGJP4lqWYqdvg5FczKUmhwYcwT5XxnXiawTo8Leb5k3hBy5ichv8saKOIpu36ibmL1gMmQewdKwA/640?wx_fmt=png&from=appmsg) | `bdda19bec56552dca6bc0aed93c7bc579d2487961d330ee9bc24e89bcc7429e6` |

正文：[vLLM 硬件插件、平台选择与执行适配](<../../vllm/运行时/vLLM 硬件插件、平台选择与执行适配学习文档.md>)。
