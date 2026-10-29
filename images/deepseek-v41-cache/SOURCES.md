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


## 2026-10-29：模型结构详解的两项增量

- 来源：[《详细分析 DeepSeek-V4.1 Flash 技术报告(1) : 模型结构》](https://mp.weixin.qq.com/s/2Apx01jPq98D_MNxmKLjpw)，作者渣B（Zartbot）；发布时间 2026-09-17 08:26:00，Asia/Shanghai，按页面 `ct` 换算。另有[作者桌面版](https://zartbot.github.io/blog/model_arch/dsv41flash_arch/index.html)。
- 完整读取正文及末尾 9 项参考资料，公式从 HTML 的 `data-formula` 属性提取；网页脚本未执行。查看正文 33 处图片，共 29 份不同图像：第 2/13/27 处相同，第 15/23 处相同，第 28/33 处相同。
- 本次保留与正文新增解释直接相关的第 25、30 处原图，原样保存；服务器返回的真实格式为 JPEG，故使用 `.jpg`。其他图片已阅读，但不因收录一条链接而重复保存全部文章；原有四张官方报告图继续保留。
- 两图均无透明通道，浅色背景可直接阅读。原图需放大查看，正文另有中文图意解读、容量表及短算例，承载窄屏阅读所需信息。
- 第 25 处图仅用于 main KV 分组和打包字节解释，不把图中的舍入误差、参考实现路径当成本仓源码核验结果。第 30 处图按固定全局字典的简化读出理解，不能推广到逐层变化的 SWA 和完整层输出。

| 本地文件 | 原文顺序 | 尺寸 | 采用范围 | 原文图片地址 | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| [05-main-kv-fp4-layout.jpg](05-main-kv-fp4-layout.jpg) | 第 25 处 | 1080×872 | main KV 分组与字节账目 | [原图](https://mmbiz.qpic.cn/mmbiz_jpg/llFnu58nOPa8nsTZpNUF7axUslE7oJdzLKaIFEyMBmD9He4iadbk29axVVvmj7B34ialdNyiaW9FfRicxGx3Yyl3t3DYX20JkJXqdWQDwtKqCSU/640?wx_fmt=webp&from=appmsg) | `4ca3c439f00cd2ad6394f4c5dae30114e9733904bdc4ddfb060be3a48de8b763` |
| [06-fixed-dictionary-readout.jpg](06-fixed-dictionary-readout.jpg) | 第 30 处 | 1080×553 | 固定全局字典的读出边界 | [原图](https://mmbiz.qpic.cn/mmbiz_jpg/llFnu58nOPbHEZcr6ibADXaHbD2Tn7ctgRxP1tbf3g4fC3GYDDGYOCCJg4IzEAvOjLkHOU8Hq8icc57EPUtPeI4DaiaiaD774j8Jc4icia4EIlLHQ/640?wx_fmt=webp&from=appmsg) | `504d328185d0b58331506533e964c9f80c5af965671b6a09a32cbe95560c68ee` |

### 其余配图的复核与取舍

| 原文顺序 | 图意与处理 |
| --- | --- |
| 1、3、4、14、18 | 官方模型/容量/计算图及三模式、HSI 架构；机制已由正文和既有原图覆盖，不重复引入榜单或 FLOPs 结论。 |
| 2/13/27、15/23、16、28/33、32 | 作者的体系结构类比和层组展开；辅助理解来源层、索引层和查询。Cache/页表/时钟域属于类比，不表示动态 cache miss、硬件存放位置或权重绑定。 |
| 5–7、20–22、26、29 | 视觉、残差、Engram、计算机比喻和 Transformer 注入点；超出本篇缓存主线的新增范围，未据此扩展这些专题。 |
| 8–10 | Agent 追加输入与 YOCO 提前退出；本篇已有 CED、冷 Prefill 算例与多轮恢复窗口说明。 |
| 11–12 | CSA/CSA2 压缩器和 indexer K 来源；正文已区分缓存来源，本次不扩成跨代压缩器实现专题。 |
| 17、19 | HISA 与 HSI 的分层选择；已有候选池、Top-K 和全扫描边界。未核验图内参考代码、最新块强制保留及估算加速比，不将其补成引擎事实。 |
| 24 | 通用 NVFP4 双级 scale 示意；V4.1 main KV 省略第二级 scale，不能直接照图多计一份全局 scale。 |
| 31 | 查询指数重加权示意；采用其代数思路但未采用数值柱图。按图列出的基线与乘数归一化，第三/第四项约为 0.34/0.08，右侧柱图却标成约 0.08/0.34。正文另用可复算的两条记录示例。 |

架构与维度复核依据：[官方报告 v1 §2.3、§2.4.4、§4.2.1](https://arxiv.org/html/2609.19969v1)；indexer MXFP4 缩放粒度依据 [OCP MX 规范 v1.0 Table 1](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。本次无模型运行、性能复现或完整源码审计。
