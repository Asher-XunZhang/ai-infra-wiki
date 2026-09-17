# 原始配图来源与处理记录

读取时间：2026-09-09（Asia/Shanghai）。图片来自正文 `js_content` 的 `data-src`；保留下载文件原始字节，未重绘或修改图中文字。SHA256 用于识别本地副本，不代表技术结论经过实验验证。

## 来源

- S04：[《Prefill chunk size 从 64K 调到 16K：省显存，也能提吞吐吗？》](https://mp.weixin.qq.com/s/8fPRQaX8ik03r4yAtmRMjA)；作者 魏新宇；发布 2026-09-07T22:53:53+08:00。

## 保留的技术图

| 文件 | 正文图序 | 内容与证据边界 | 原图 URL | SHA256 |
| --- | --- | --- | --- | --- |
| [01-oom-tuning.png](./01-oom-tuning.png) | S04 图 1 | 官方调优说明的原文截图；示例参数不是普适推荐。 | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/x7l3NT3zibfyibV7h8qUphPTZ6IBfq6IqQE6j45prwvZLFozdDvYLytat7BXPwtVXUlZxOgIibPfKqFfBwm7NGmH0Xd35QgqB46CNO1stNRf9c/640?wx_fmt=png&from=appmsg) | `42e050903d35281e7a2c42ecd1a6b07214c3e0dca45d903ea0c83b61029c7f16` |
| [02-sarathi-prefill-overhead.png](./02-sarathi-prefill-overhead.png) | S04 图 2 | Sarathi-Serve v3 Figure 14；Yi-34B、TP2、512/1024/2048 chunk，不是 SGLang 16K/64K 实测。 | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/x7l3NT3zibfzcabMgo2znWtricqoTS3KuyLf9al4XXg28OVeh8SdxSVPeIlZDqCVDESsZpXjwsaoWyDaqGRWdZFLicgvrViar4o56m8NM2vfbq4/640?wx_fmt=png&from=appmsg) | `54bc9a4edad5407ee2b0b8050ec23d678ece1b9228a1c901c8936ec443bd7c47` |
| [03-memory-capacity-conditions.png](./03-memory-capacity-conditions.png) | S04 图 3 | 原文章作者绘制的条件关系图，不是本次生成图，也不是性能测量。 | [原图](https://mmbiz.qpic.cn/mmbiz_png/x7l3NT3zibfx44ibgHukSmnGntjL0Nf7HpN8I0icyvO48n0LCFMuibmrDVUnmZlQiaxmVykYH9m8QAAfHjNWSHozaiacyl991fS4rjIia0W6zJEOrc/640?wx_fmt=png&from=appmsg) | `3dd258b899e10ca24cbc48178dda0a79182419c15ef42c2e8fd54f7213ee7997` |
| [04-sarathi-scheduling-timeline.png](./04-sarathi-scheduling-timeline.png) | S04 图 4 | Sarathi-Serve v3 Figure 7；历史调度示意，不是当前框架排名。 | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/x7l3NT3zibfxPHEynJcDb8icJF3cia1iap5G8lu3WHnfNU5ycicz9Q53uX6a0kWpqdv4pbwDBoh7BTe3icKCvqPDgG2ic0X8xXwmmuppmcQUNXoIxY/640?wx_fmt=png&from=appmsg) | `14f4bfced4ab34a0472e299bd334c47d57f8a1e56c06e02752c5caf5a9b1a154` |

## 性能图原始出处

- [Sarathi-Serve，arXiv:2403.02310v3](https://arxiv.org/html/2403.02310v3)：Figure 7、Figure 14、§5.4。原文图 2、图 4 是论文图片的转载，不能视作两组独立实验。
- [SGLang 固定版本调优文档](https://github.com/sgl-project/sglang/blob/6e312af8c25ccedd1dcd2583358be038ab4875b0/docs/docs/advanced_features/hyperparameter_tuning.mdx)：核对原文图 1 所述取舍。

学习正文：[Chunked Prefill 与调度器显存预算](<../../sglang/runtime/SGLang Chunked Prefill 与调度器显存预算学习文档.md>) 第 14～16 节。
