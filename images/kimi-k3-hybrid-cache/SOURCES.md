# 图片来源与校验

读取时间：2026-09-09。以下文件保留下载原字节，未裁剪、重绘或重新编码。序号指 `js_content` 内的图片顺序。

| 文件 | 原文图序 | 尺寸 | SHA256 |
| --- | --- | --- | --- |
| [01-hybrid-architecture.png](01-hybrid-architecture.png) | 1 | 1080 × 934 | `c97f55cfb3a7afdbc7fc2f451cb7f5e7482686dbaff3ae85b1c80723462098cc` |
| [02-checkpoint-capacity.png](02-checkpoint-capacity.png) | 2 | 1080 × 475 | `4a6098c4596aca7c0f65fa93d38fd4e19c14bca4193bb9644aa6ada84d3a0b5d` |
| [03-state-ownership.png](03-state-ownership.png) | 3 | 1080 × 579 | `74b7e8bab40b7ac224a083b59477c53fee918549cd97364331ff783ca29410f9` |
| [04-unified-memory.png](04-unified-memory.png) | 4 | 1080 × 599 | `b2582e1e465946d529d87f374d42e4583b64cb6e5aaf04f75982c775a122174b` |
| [05-two-cache-representations.png](05-two-cache-representations.png) | 5 | 1080 × 532 | `3eda53b75449dba3a30d8ddab89b14a23b4b06cc84c4fd9b0682425be83f1b29` |
| [06-partial-prefix-hit.png](06-partial-prefix-hit.png) | 6 | 1080 × 796 | `f0b1dbdc9d73e980efc139aebae4e514a57a6f059923199c867973faed5a2c9f` |
| [07-interval-retention.png](07-interval-retention.png) | 7 | 1080 × 288 | `a6cdf8a021fa01f260252bb2c4d34d43377f1bd948fc3a36e48f9ca64b89c12c` |
| [08-selective-retention.gif](08-selective-retention.gif) | 8 | 1079 × 392 | `f00954a8a929eb6f545aca588d461c9297499b9ef5d2e045110ca1180883a7c3` |
| [09-flat-kv-layout.png](09-flat-kv-layout.png) | 9 | 1080 × 595 | `cc632a4a3f9486fcf9f2bfa35b612783c092c69cc8175b62496ea7799e891a2a` |
| [10-epd-data-plane.png](10-epd-data-plane.png) | 10 | 1080 × 595 | `892066e88539f52f4ed08f1eae1fbd003b1934520cb4eb8ad2ef9a35e0e418bb` |

原文：[文章入口](https://mp.weixin.qq.com/s/Yxmt-Foq2D7b46sYOk7WAg)。图片版权归原作者或原发布机构；各图的解释和适用边界见引用它的学习文档。

## 原始图片地址

- 原文图 1：[原始文件](https://mmbiz.qpic.cn/mmbiz_png/U63nXREMfAj5KY9UvQrqDLCws32vj3RibQqeCrxDPVSHVEggzniabygia7Znz61nl7RUDTDA0x4qUpjdVbLvic9H8MCtd29j0DUyLpmBsMY2NBw/640?wx_fmt=png&from=appmsg)。
- 原文图 2：[原始文件](https://mmbiz.qpic.cn/mmbiz_png/U63nXREMfAh6DAonnZicqo3HO8r7fdMk9tUAupsvo6bI0AibxGGDxbVuAYia0ohAOnKIX5o490MTAKGRShVJpSFJRnTUfGiboQGl1NwaxZdmCFI/640?wx_fmt=png&from=appmsg)。
- 原文图 3：[原始文件](https://mmbiz.qpic.cn/mmbiz_png/U63nXREMfAiaAH7jsTgoLUPialQ3MlJVkwx6ORyzaXbjfcBHS9Pc64sKiaV1vuUgyrDia7oxS7ia5TlicVtr0RmIIay8xyAtw5O3h0X8jKrnc4YPc/640?wx_fmt=png&from=appmsg)。
- 原文图 4：[原始文件](https://mmbiz.qpic.cn/mmbiz_png/U63nXREMfAiajKk0QXU3VCXelcDDsaV7qFnuPNibS6yxwgpjpsLnTJ2QOibTeTqbYwiaQhdEwibr8sULMEmBUYmUxxAbQ5N8vsbnsY59AYsk6Eick/640?wx_fmt=png&from=appmsg)。
- 原文图 5：[原始文件](https://mmbiz.qpic.cn/mmbiz_png/U63nXREMfAg9sVOcbHHUIdGvB3BPUmYGsdQYBBg0Frj89RpFjAg0icIwXUpJzGAib0icibmZMVicdqu47xfziabnDhJdyIZkczicXt0kkibQgBTxzqc/640?wx_fmt=png&from=appmsg)。
- 原文图 6：[原始文件](https://mmbiz.qpic.cn/mmbiz_png/U63nXREMfAhian5PRjgga4E38sQOibo75yIXGswDlJLn97FRUsAQV1mw8FGdHSgffoYm1icKEQHRJ1zojp4tsmz3TksWkk0wVDQPoMm1jRP1xc/640?wx_fmt=png&from=appmsg)。
- 原文图 7：[原始文件](https://mmbiz.qpic.cn/mmbiz_png/U63nXREMfAgzY46oupibKk0aLjG7orxdcR5UKnnhgyLbnzTTMF6Qdx2FpumsEnP1JcoegofOuGp6rrdAWA1M2ic7xNSdk1En9FR4IM7TVoCc4/640?wx_fmt=png&from=appmsg)。
- 原文图 8：[原始文件](https://mmbiz.qpic.cn/mmbiz_gif/U63nXREMfAjqF4jFNAlPcq3gsCjWO6y9Xn5z5gJBFHzKOCJJeNY7j3qpLE0ErFp73nBPU2mwjFka1GpzSH9Y4na499exC1aDDv4q90pjw1s/640?wx_fmt=gif&from=appmsg)。
- 原文图 9：[原始文件](https://mmbiz.qpic.cn/sz_mmbiz_png/U63nXREMfAhMPMZsJWWU7Jge32C9qvsHHPoCfjZq3jibiboLdKaWUrdGraqG0Nv1ApuhmcPfA4SnIUicLib9LE9Ypqx7kic5PjibyhuDQgb7saWV4/640?wx_fmt=png&from=appmsg)。
- 原文图 10：[原始文件](https://mmbiz.qpic.cn/sz_mmbiz_png/U63nXREMfAialoGXYDL4x8z2ldt5giaTRRd9QHcqr8GQKiccm86b1OMWy4iaP78gdCyqnBdgc6V1o1w4xzGhFuoqmTWYoxYUEJkD814D95Wj8SQ/640?wx_fmt=png&from=appmsg)。

第 2 篇与第 3 篇的图 1、3–10 SHA256 完全相同；图 2 为相同内容的 JPEG/PNG 两种编码，保留第 3 篇 PNG。动态图 8 保留 GIF。图 9 含透明背景，深色预览下标题较弱，可用浅色背景查看；正文另写完整布局说明。
