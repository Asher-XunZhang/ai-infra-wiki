# 图片来源与阅读边界

原文：[PD 分离还不够：多模态推理为什么开始拆分 Encode？](https://mp.weixin.qq.com/s/k9nZgwRBY_tYAe6oRG4txg)。读取：2026-10-14。原文件保持不变；技术解释和纠偏见对应学习文档。

原图为原作者或原论文图示，权利归原权利人；这里只用于图意分析与学习。

| 本地文件 | 原文序号 | 尺寸 | SHA-256 | 原始下载地址 |
| --- | --- | --- | --- | --- |
| [01-media-pipeline.png](01-media-pipeline.png) | 2 | 1080 × 608 | `cd87be80b3c0ffe178a6ab9fb601308fe6dec567dfe55dd6fc033b8ab3746603` | [原图](https://mmbiz.qpic.cn/mmbiz_png/KlV8ppcRQBS17WialImMkETxibE4BXYicSf3ibl1PSu73I2YuLkrNO93BB87MItsHOeyDUPF0etG4Mm6MZpiaR8kRDJZon3z4KaVZe7siaApFh8zc/640?wx_fmt=png&from=appmsg) |
| [02-resource-contention.png](02-resource-contention.png) | 3 | 1080 × 463 | `33695e8818f2f5e79c4a9e0ea13107567316deee006c0499b5ebfadc6561317b` | [原图](https://mmbiz.qpic.cn/mmbiz_png/KlV8ppcRQBRApq6oqYyicMGze4MgapcKNVqVibiaQQZUOowfIlsS0mPFxwzjh6I43Www0l7zQ3VQUhStYW1J3y4icibARyEdKySlbJgrliacbbH8A/640?wx_fmt=png&from=appmsg) |
| [03-deployment-topologies.png](03-deployment-topologies.png) | 4 | 1080 × 608 | `475e8089a298b5292e86ff380b4b3536f49a006605745c903b9f3e39653669e4` | [原图](https://mmbiz.qpic.cn/mmbiz_png/KlV8ppcRQBQUvKAjbqd73IkYZ3L8j4XhjBYtEYcNN8PFCCI9Xjyc9e5G9YibXB9GhxUiaERtUu41rj72YHtDSicBUA3LS073Q4LLPFaxlpI2icQ/640?wx_fmt=png&from=appmsg) |
| [04-embedding-kv-transfer.png](04-embedding-kv-transfer.png) | 5 | 1080 × 608 | `5b13f72372bcbec5fa7b9b7f92fa60a864a94211ee72fb0fc34876ca97b82d46` | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/KlV8ppcRQBTQbWb51wOzBztoJgEX3Ft2LycO9tFSWj9CVRbOZbJuLXlakPibVoXfb7S9xcJNiaCDIkIf5xT3ozRS57yYu1M9zcyTOrV3JfEws/640?wx_fmt=png&from=appmsg) |
| [05-request-flow.png](05-request-flow.png) | 6 | 1080 × 540 | `33f627432a00e5ca19f9934767d4c0b4827a41c4ea2d41bbd81ef61384f1c5b4` | [原图](https://mmbiz.qpic.cn/mmbiz_png/KlV8ppcRQBQ3Am0AY0ms3Y3VqtkC3tETCJ7UvZr18P2cvicbjBN7UmhbHVjEsWhkBVdWfygVTEwPnwicK6VV4mI1E4ibfpV5Pk9Ok6AwI19bJI/640?wx_fmt=png&from=appmsg) |
| [06-resource-scheduling.png](06-resource-scheduling.png) | 7 | 1080 × 608 | `d4822020731e2420f113e79d9588f29fc06c3f8d9cedc5c87fae1ca2c6276e3e` | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/KlV8ppcRQBRzWd82xREicQGAhMQV6wMkGohPHjlpe2oyTwAo2Z7cyHH2TSdwMibZrBWicfDhkHLHmicLialAd5DbGVib1icq8n2eOlhvLJLbMjtibibc/640?wx_fmt=png&from=appmsg) |

原文第 1 张是题图，信息与正文图重复，未收录。示意图中的利用率、加速比和容量没有配套实验条件，不作为性能证据。
