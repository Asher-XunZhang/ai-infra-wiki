# Attention 容量与计算口径图源

## 来源与采用范围

- 原文：[《大模型加速优化：Attention架构演进》](https://mp.weixin.qq.com/s/fJglfOaTUreYXjcIOdLH3A)，页面作者“程序员Owen”，公众号“AI修炼技术派”。发布时间 2026-10-30 08:30:00，Asia/Shanghai，由 `ct=1793320200` 换算；读取于 2026-11-02。
- 完整读取六节正文及全部 17 张技术图，末尾为 HCA 缓存图和总结。没有技术图之外的二维码或推广图片；正文未提取到独立 SVG / `data-formula`，计算公式位于第 3、6、9、12、15 张图片中。脚本未执行。
- 算子与缓存图带“知乎 @骑虎南下”水印；保留原图与水印，不将页面作者视为已证实的图像原创者。版权归原作者，未核定最初发表页面。
- 本次只保留支撑增量说明的 3 张原图，原始字节为 JPEG，未重绘或修改；其余图完成阅读后不重复收录。图片无透明通道，白底下可读；长图与横向图在窄屏会缩小，正文以维度表和条目数量表承载全部关键结论。
- 对应正文：[KV Cache 容量地图](<../../大模型推理/KV缓存/KV Cache 容量优化技术地图学习文档.md>)、[混合注意力与索引压缩比较](<../../大模型推理/模型架构/混合注意力、索引压缩与残差流架构对比学习文档.md>)。不是性能复现、模型完整源码审计或新的统一排名。

| 本地文件 | 原文图序 | 尺寸 | 采用范围 | 原图链接 | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| [01-mla-projections.jpg](01-mla-projections.jpg) | 7 | 1080×1715 | 512 维 latent、64 维位置 key 与展开后 K/V 的区别 | [原图](https://mmbiz.qpic.cn/sz_mmbiz_jpg/dHwlicgnJnmShlGg3ic32mZb5q6mOogZggeuSZUmsMTuzS5EhZGibZ8cE7jYa3r1Qyptibvqy6hmQQXgHwTpfKaNAjZJwlkKZJERVnMOC3Gfc94/640?wx_fmt=jpeg&from=appmsg) | `98277541bf481a594eaff4d1a571a2b933df769e3c30b6141a6d7ac8e3efca73` |
| [02-csa-state.jpg](02-csa-state.jpg) | 14 | 1080×320 | 局部窗口、压缩历史与索引记录三条路径 | [原图](https://mmbiz.qpic.cn/mmbiz_jpg/dHwlicgnJnmQCTn1uJITpzsj4G5ZZfkpDLGOvotTkkMEIbTshvGTKHEbpibnuPT2pSyBWX4nvVgIPItPxn5ypyJsCBibkicKPHhoYPDhL2urquQ/640?wx_fmt=jpeg&from=appmsg) | `d5907f8fa64b1b141208ad04e1fab82f2222f5a3f0b94b08ddf69ed4cfc2d8a0` |
| [03-hca-state.jpg](03-hca-state.jpg) | 17 | 1080×399 | 局部窗口加全部高压缩历史，无 Indexer | [原图](https://mmbiz.qpic.cn/sz_mmbiz_jpg/dHwlicgnJnmTibHzueltbfXZoBWZXNy73PJnZOBjES9UV2Mkicj8lEViauds2ydRXOiapneCXvQTQ3IlbrAyANoV7TYJWbtga9uWlSkOQbdGjn9Q/640?wx_fmt=jpeg&from=appmsg) | `8d8da4138e82d985db7d8bf6cbec112d1b1be8ddabbb3851b816b9b086488ceb` |

## 全部配图的复核与取舍

| 原文图序 | 内容 | 处理与边界 |
| --- | --- | --- |
| 1–3 | MHA 算子、缓存形状、两次乘法 | 已有 Transformer 与容量地图覆盖；公式中的两个 head dim 只有相等时才能共用同一个符号，矩形 S×T 也不是因果 Prefill 的精确有效配对数。 |
| 4–6 | Qwen3 GQA 算子、共享缓存、乘法 | 保留其“KV 头减少不同比减少主注意力乘法”的问题意识。第 4 图错误包含 V RMSNorm、K-RoPE 64 头和输出的每头 80 维标签，正文按固定配置与实现校正，不保存该图作为结构依据。 |
| 7–9 | MLA 展开路径、合并缓存、MQA 公式 | 保留第 7 图；第 8 图的共享缓存概念已有说明。第 9 图把两个乘法维度都写成 QHD，不作为 MLA 的精确 FLOPs 公式，正文分别列出展开与吸收路径。 |
| 10–12 | DSA 的 Indexer、两路缓存、Top-K 公式 | 主缓存和 indexer key 不是相同对象；Top-K=2048 只描述主注意力的选择预算。索引打分与 scale 另计，参考实现的全矩阵加 mask 也不能算成已执行 sparse kernel。 |
| 13–15 | CSA 算子、缓存、128+512 公式 | 保留第 14 图。第 13 图把 indexer 缓存宽度写成 512，和配置 128 维不符；第 15 图只适用于可见条目充足的主乘法口径，不包含 Compressor/Indexer/Top-K。 |
| 16–17 | HCA 算子与缓存 | 保留第 17 图。第 16 图计算框沿用了含“DSA”的标签，不能据此说 HCA 使用稀疏 Top-K；HCA 遍历压缩条目。正文补上 T/128 与 512 的长度条件，并指出第六节“2048”混用了 V3.2 的配置。 |

## 一手核验基线

读取日期均为 **2026-11-02**。以下仅下载公开文本到临时位置、静态读取指定配置/函数，无源码仓库检出或工作区改动，无模型执行、训练或 GPU 实验。Hugging Face 项目使用读取时 `main` 的固定 commit；Transformers 使用公开 tag `v4.51.3` 对应 commit，tag URL 与 commit URL 返回的代码字节一致。

| 核验对象 | 公开仓库与固定 commit | 文件/函数与支撑事实 |
| --- | --- | --- |
| Qwen3-32B | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/tree/9216db5781bf21249d130ec9da846c4624c16137)，`9216db5781bf21249d130ec9da846c4624c16137` | `config.json`：64 Q 头、8 KV 头、head_dim=128、hidden_size=5120。 |
| Qwen3Attention | [huggingface/transformers](https://github.com/huggingface/transformers/tree/5f4ecf2d9f867a1255131d2461d75793c0cf1db2)，`5f4ecf2d9f867a1255131d2461d75793c0cf1db2` | `src/transformers/models/qwen3/modeling_qwen3.py::Qwen3Attention`：仅 Q/K Norm，V 直接投影；输出投影 8192→5120。 |
| DeepSeek-R1 | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1/tree/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad)，`56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad` | `config.json`：KV latent 512、Q/K 内容维 128、RoPE 64、V 128；用来核对原文 R1 配置，不据此宣称已审计其全部实现。 |
| DeepSeek-V3.2 | [deepseek-ai/DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6)，`a7e62ac04ecb2c0a54d736dc46601c5606cf10a6` | `config.json`；`inference/model.py::Indexer, MLA.forward`：indexer key/scale，Top-K=2048，MLA 展开/吸收路径，全矩阵加 mask 的参考实现边界。 |
| DeepSeek-V4-Flash-0731 | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/7872f01b1d1fe23eabc4c98b48bffcef5a386062)，`7872f01b1d1fe23eabc4c98b48bffcef5a386062` | `config.json`；`inference/model.py::Indexer.forward, Attention.forward, get_compress_topk_idxs`：压缩率 4/128、窗口 128、CSA Top-K=512、main 512 维/indexer 128 维，以及可见压缩条目数。 |

论文交叉核对：[DeepSeek-V3 报告 v1 §2.1.1](https://arxiv.org/html/2412.19437v1#S2.SS1.SSS1)、[DeepSeek-V3.2 报告 v1 §2.1 与附录 A](https://arxiv.org/html/2512.02556v1)、[DeepSeek-V4 报告 v1 §2.3 与 §4.2.1](https://arxiv.org/html/2606.19348v1)。V4-Flash-0731 与既有 V4 报告基线分别标识，不将其配置外推到 V4.1。

### 本次读取文本的校验值

这些校验值定位本次静态核验的输入，不代表对权重、运行镜像或模型输出做了校验。

| 文本文件 | SHA-256 |
| --- | --- |
| qwen3-config.json | `97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb` |
| qwen3-model-pinned.py | `704c914530530a1acb0b443add1f520404e3ac2c28c0ab7e16f80f86cfe8ccb2` |
| r1-config.json | `79ddea672a62e95d3f0be27be434375538e1988975971cffcf87c96c1de84a65` |
| v32-config.json | `c7fa8b191e9936d8e6a57d864baab82b792fae16a116416cdd3a75ba76bc5af1` |
| v32-inference-model.py | `bfe5b89186b579910b6d59fa7e0cf1f7a75ceb4ec5d0f5b54f0ccbe77cdb6a63` |
| v4-config.json | `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023` |
| v4-inference-model.py | `c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f` |
