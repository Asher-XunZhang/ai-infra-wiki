# 图片来源与阅读边界

原文：[从ViT到全模态Omni：一文读懂多模态大模型演进史](https://mp.weixin.qq.com/s/jzcgVGhSOzXlPeZUibZnTA)。读取：2026-10-14。原文件保持不变；技术解释和纠偏见对应学习文档。

原图为原作者或原论文图示，权利归原权利人；这里只用于图意分析与学习。

| 本地文件 | 原文序号 | 尺寸 | SHA-256 | 原始下载地址 |
| --- | --- | --- | --- | --- |
| [01-vit.png](01-vit.png) | 1 | 663 × 492 | `9b3643599b094226cf944fc4106782d79cd89666a0606495a3d1bf04116df88d` | [原图](https://mmbiz.qpic.cn/mmbiz_png/xG65qWYOXTH6ByP4Ku4nsfQW2wiaVLg86Jz5uNvXKVsYj2uBdmZBB7tg4ufJeU1rLnnmsYVSjjtb66rhIicPzdbcFiaYCkYjuU7Ckg4Xgr1rVA/640?wx_fmt=png&from=appmsg) |
| [02-clip.png](02-clip.png) | 2 | 1080 × 386 | `d860579d4c685ace0e54ca6784900ed5c069c2ba3c4094bed9f867ceea48bf2d` | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/xG65qWYOXTHghhq6beZn03eMfP5Jia2vaFkiclA3oJ3oicOpJ20BXvVDjFHMVbHZYo2KbZFhG9VcmIn16xHIWr3wl4v5IdVmZlK533Vnyrsrsg/640?wx_fmt=png&from=appmsg) |
| [03-llava-family.png](03-llava-family.png) | 3 | 973 × 457 | `a5c00116c3929b87b09e1da716099799b6adab65e7a8ad301ff89f1705c619dd` | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/xG65qWYOXTFkmcsib9CgDHYmrla7qLpCozwrIQnJEAnWNbwELLUCMGLicEG8b1sSo3VXpUJJPpVrsiaibebia1qgrjv9QibXqKy7yBZWnUTWrTtF4/640?wx_fmt=png&from=appmsg) |
| [04-internvl.png](04-internvl.png) | 4 | 1080 × 325 | `b84f5c7926dd6ceac4c573ba855f895dcb78c06b1f64aac3a25d5cc49581bbae` | [原图](https://mmbiz.qpic.cn/mmbiz_png/xG65qWYOXTGoUrIZVicOCPgk4pWQ97DteuZPwVQC9ic6ZaUABf3tibdq05RLyYTTeLSJFYdkoibJcTErYLtSTv7CicnDoeiccvmcvln3ZZXNq1jVY/640?wx_fmt=png&from=appmsg) |
| [05-qwen2-vl.png](05-qwen2-vl.png) | 5 | 1080 × 687 | `8edebd9430a8f74740eb6c4d16856459165d6bafbfab725f0daa68d951dd8237` | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/xG65qWYOXTFVvic2iaN0ZaUCvsNQtvoscLk7QSP3zt9U65D4ibk9dCAEiaQxGhmZbJ0uUMDaT4kegwGqUXtjKqWYB5gDvHhEZXKWZvbqHI1NlWU/640?wx_fmt=png&from=appmsg) |
| [06-vq-vae.png](06-vq-vae.png) | 6 | 984 × 374 | `55480eacf6fc30d4974908e1c4ccac11acf5624b807626d43b48fdf8735fbb9a` | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/xG65qWYOXTGpUGZlkfLicKbU6WgcXGuVT2mFCicNhRqo07EF9ZWIqsRwOjica3Q55E2NgsGibetHtYnh3uloZ2Q7WbRh5Q4a6GrOibLyrdSuyppc/640?wx_fmt=png&from=appmsg) |
| [07-contrastive-encoder.png](07-contrastive-encoder.png) | 7 | 548 × 360 | `990c6726e6f9678ec3e4c52233d31d767848d6bc053d649928de8a3843ef393a` | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/xG65qWYOXTGYSQYxicEcteDxicwpvMMoBbdhvYdicVt1hTj5DZAOdlLD5sPEwhRia3hy2SOM07pcLic3b8AGXvLQAPGxzT1TyxHnHZgkJga9IjF8/640?wx_fmt=png&from=appmsg) |
| [08-chameleon.png](08-chameleon.png) | 8 | 986 × 527 | `9a7365bf273211b64e94a652a0676dc8976607b4f353a45e4acc463fc863cec9` | [原图](https://mmbiz.qpic.cn/mmbiz_png/xG65qWYOXTFnaYvCrBpdVwGeDriaHa5SaT4ywaYtYSibC4loOicIdtgXjJlR6q8kno5icibiaKeibO3e7Zd6smywibcaMpeCxF2DQB7Gh0hMtH3j7Xw/640?wx_fmt=png&from=appmsg) |
| [09-janus.png](09-janus.png) | 9 | 1080 × 452 | `d41e1a4d9f569c0e4fa0405df34a9a7a4b6421f551360f6fdea973edc8c365a9` | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/xG65qWYOXTF3oxk0lxHyIgKibrS5JhQbeMMjeFJ2ZeWXNiaYJ5wCBRxV41fF5WxunepEIic7ZqHz0mQ1e3dQibRtDMr594Wyl51aplNiadZp0438/640?wx_fmt=png&from=appmsg) |
| [10-onecat.png](10-onecat.png) | 10 | 1080 × 422 | `63238f64603f5e43b927f97d4e76651435d4c44b0e2b796b29c16b1711929900` | [原图](https://mmbiz.qpic.cn/mmbiz_png/xG65qWYOXTEpYZtSAAj5Hu3ddOdCOCZDbxI9j5bxnJ83zdOib6qw3w8DQO0EFfYAv0feqDjyic7LynmI5MSTnqjXMkH39BTXJ91EYkAetV1r0/640?wx_fmt=png&from=appmsg) |
| [11-bagel.png](11-bagel.png) | 11 | 1080 × 396 | `4acdff897a10c15710a9b4792bf3f39e0b65c344774663c459a6d2ce19b93ac1` | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/xG65qWYOXTFwBjdqIIR1ap7gUuFHpGYLINCofMDFojGTmYFyESa6icbDMevCBBtobCUic0Zn7deER5R0w6jP2TphEFfe8SSulgEmG27v4Jian0/640?wx_fmt=png&from=appmsg) |
| [12-capability-axes.png](12-capability-axes.png) | 12 | 1080 × 300 | `664ce800ee517fa67d1990255f7c3ede4d51497eab9bde1b57a8e7bbab82b121` | [原图](https://mmbiz.qpic.cn/mmbiz_png/xG65qWYOXTEabZvCdc6TwmHyrPiaEVj4Dvq3NAgw7GWWkoAYOZicb0nkIMiaKQKUlwYJ0U3yVcibz7YG3jMlibGD8fFxM4HgL1odgH77f1P6RES0/640?wx_fmt=png&from=appmsg) |
| [13-ming-omni.png](13-ming-omni.png) | 13 | 847 × 780 | `17fa068f6c2dcb4da494c4f415fc429fb1ecb7f20b81706b4e05d16bff782e1b` | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/xG65qWYOXTGrzG2OpcpcOBTwFM7pNdy3a2I6KTcymcbIfr5eubEHiaibdg5AtiaYyb0lp8zcPkZLdYH08VzSbWcwibU0Dke94Bl3tuN7lBpG2eQ/640?wx_fmt=png&from=appmsg) |

逐张检查了全部 13 张原图。第 5 张图内明确标注 Qwen2-VL，原文图注却称 Qwen2.5-VL；第 7 张实际展示 CLIP 对比学习，而非独立 ViT 结构；第 11 张包含生成专家和编码器，不能用来支持 BAGEL 无编码器、无扩散的说法。正文逐项说明。
