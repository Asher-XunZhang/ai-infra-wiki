# 原始配图来源与处理记录

读取时间：2026-09-09（Asia/Shanghai）。图片来自正文 `js_content` 的 `data-src`；保留下载文件原始字节，未重绘或修改图中文字。SHA256 用于识别本地副本，不代表技术结论经过实验验证。

## 来源

- S02：[《从 KV Cache 到 Zero Overhead Scheduling，一文读懂 SGLang 的调度巧思》](https://mp.weixin.qq.com/s/-O5W_4CGD0XJMAtHckn3nw)；作者 关注AI Infra；发布 2026-01-12T12:00:33+08:00。
- S03：[《小进探索sglang：sglang中的scheduler调度原理和代码解析》](https://mp.weixin.qq.com/s/baB0ozQrVuaqZrTphSCUvg)；作者 lil2j；发布 2025-12-08T11:52:13+08:00。
- S05：[《SGLang推理优化-调度器核心ScheduleBatch》](https://mp.weixin.qq.com/s/e--Z3OKzilcZuJFoi7Hizg)；作者 kason_zhang；发布 2026-06-05T12:00:00+08:00。

## 保留的技术图

| 文件 | 正文图序 | 内容与证据边界 | 原图 URL | SHA256 |
| --- | --- | --- | --- | --- |
| [02-end-to-end-components.jpg](./02-end-to-end-components.jpg) | S02 图 1 | 逻辑组件图，不等于进程拓扑。 | [原图](https://mmbiz.qpic.cn/sz_mmbiz_jpg/DPAHibibAl3vQuJ41G9xKQV7rDfTV5LWwHyJeTnmZE8aILS24VHIuwY7fVazzTU4aXqcGFP5SiaLU773gg1Y6XK4Q/640?wx_fmt=jpeg&from=appmsg) | `26af7c5a51a287911ff6a693c50e923b8b6a6369d8bdbb231a898cd16d57e397` |
| [03-event-loop-call-graph.jpg](./03-event-loop-call-graph.jpg) | S02 图 2 | 请求入口、选批、执行与结果处理调用图。 | [原图](https://mmbiz.qpic.cn/sz_mmbiz_jpg/DPAHibibAl3vQuJ41G9xKQV7rDfTV5LWwHJiaVCJiaAZGZian910kbJN4LVhibMFjl4AckVp6Rwr3XHwALPHZfsHIKxA/640?wx_fmt=jpeg&from=appmsg) | `c43cc6b808016e6b4845270ffd953207ebdc7dd1a392e2abb7c846368f441086` |
| [04-prefill-decode-batch-transition.jpg](./04-prefill-decode-batch-transition.jpg) | S02 图 3 | EXTEND 过滤并入 running 的历史流程图；图内函数拼写以正文和固定源码锚点为准。 | [原图](https://mmbiz.qpic.cn/sz_mmbiz_jpg/DPAHibibAl3vQuJ41G9xKQV7rDfTV5LWwHN68C4ODmp1Uo3icFd0rFlm5Yp2fLc8cDDicyicd8ia52g7KXpPaRiaJA8zQ/640?wx_fmt=jpeg&from=appmsg) | `bf257a3bcdfe416a718e753a3aa51c867cf45a25098712d7261610aa5536f59a` |
| [06-prefix-slot-mapping.jpg](./06-prefix-slot-mapping.jpg) | S02 图 4 | 逻辑索引、请求寻址与物理 KV；图中 L1/L2/L3 是作者逻辑分层，不是 HiCache 存储层级。 | [原图](https://mmbiz.qpic.cn/sz_mmbiz_jpg/DPAHibibAl3vQuJ41G9xKQV7rDfTV5LWwH0W6zibymKI3B6BafE0nA0kMPTUBDeVBtCvB2qVicC0eUFI98G4MoGJbQ/640?wx_fmt=jpeg&from=appmsg) | `0550ed6057b596189ef5fc2333df87405d2c33f9475ace4ea6a00bcf6ee9236a` |
| [01-scheduler-overview.png](./01-scheduler-overview.png) | S03 图 1 | 调度与执行总览，保留高层职责关系。 | [原图](https://mmbiz.qpic.cn/mmbiz_png/URBeoJhAhbqMa7icic8bcCsZWzN8CKKzQ0lEKWqy6y9tyxJSDpwtW5Lia6oJDm4Pwria8piapO4BLibWO4pca6R6GS9A/640?wx_fmt=png&from=appmsg) | `7dfdb67262bf5ed8e2b6caa9a0284084a87731d47bde0a58bf7a0eabbe2d77a6` |
| [07-request-merge-timeline.png](./07-request-merge-timeline.png) | S03 图 5 | 新请求跨轮合并的状态表；首 token 通常在最终 Prefill 产生，不沿用原文相邻文字的 token 计数。 | [原图](https://mmbiz.qpic.cn/mmbiz_png/URBeoJhAhbqMa7icic8bcCsZWzN8CKKzQ0sd5eIsdMmrDmlIAl0kiapVlaB59AedzJy4XBLUUeACCoiaUM5hemFGJQ/640?wx_fmt=png&from=appmsg) | `1d8682d5c554f483c28e9497b1897511f80efea25c4a95eac56659b33eef864b` |
| [05-schedule-batch-infographic.png](./05-schedule-batch-infographic.png) | S05 图 1 | Batch 主流程与示例；不同版本签名、特殊模式和预算规则需单独核对。 | [原图](https://mmbiz.qpic.cn/sz_mmbiz_png/0iapsZ8kL4iaCDcAWK8b0LLUPUZSDcDIBs3sg9uSOIvp1ryfNWEibSrbfA6zZr2JslaLWgsYNGwH8eWzANhq4TQH6nCv9VMyaK8vgmCfroAzFU/640?wx_fmt=png&from=appmsg) | `52c9213172d635a44e461de6a2e1b768c245e46ee8d2f537c551fde8d7ca348e` |

## 排除的图像

| 来源 | 正文图序 | 排除原因 |
| --- | --- | --- |
| S02 | 图 5 | 交流群二维码，无技术信息。 |
| S03 | 图 2 | 初始化海报含大量不可可靠辨读的源码名、参数拼写；不作为机制证据。 |
| S03 | 图 3 | `batch empty` 的 yes/no 与执行分支对应错误；不嵌入学习正文。 |
| S03 | 图 4 | 多处文字错误、箭头和条件不清，容易误导 Prefill/Decode 选择；不嵌入学习正文。 |

S02 的 15 个内嵌 SVG 均为宽度为零的空占位，无技术内容。S03 没有内嵌 SVG；所检查正文均无 CSS 背景技术图。排除图像未写入仓库。

学习正文：[请求生命周期与重叠调度](<../../sglang/SGLang 调度器请求生命周期与重叠调度学习文档.md>)。
