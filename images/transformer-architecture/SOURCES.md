# Transformer 架构学习图：来源与复现

对应正文：[Transformer 架构与推理流程学习文档](<../../llm-inference/model-architecture/Transformer 架构与推理流程学习文档.md>)。

读取与制作日期：2026-10-08。图 01–26 为整理者绘制的 SVG 教学图，以数字带、表格抽行、读取权重、向量相加、数轴和逐 token 缓存展示机制。所有数值是人工例子，不是模型运行截图或性能数据。图 01 使用经典 Encoder–Decoder / Post-LN；图 22 单独展示 Pre-RMSNorm / RoPE / 门控 FFN 的 Decoder-only 结构示例，不代表全部语言模型。图 20 的权重与数字为手工构造。图 23 展开经典架构的全部主要推理模块，MoE 仅标为可选 FFN 替换；图 24–26 拆解专家选择、汇总和参数口径。

当前直接维护本目录 SVG 与本来源说明。一次性专用绘图生成器已清理；需要复现原始生成过程时，可查阅[固定提交中的生成器](https://github.com/Asher-XunZhang/ai-infra-wiki/blob/d95a982380b2136e4b0ab3e6dd356084a459df06/scripts/build_transformer_diagrams.py)，并在该版本的完整仓库中运行。它不是当前工作区提供的命令。

修改图后核对正文数值、图意、文字布局与不同屏宽阅读，并执行 `python3 -B scripts/check_learning_docs.py` 检查路径；来源和原图校验值继续保留。

## 视觉表达参考

下列资料于 2026-10-08 阅读，用于借鉴教学表达方法。已实际查看其中的注意力汇总、分头拼接、向量和上下投影配图。本文重新设计画面、构造数值，不直接复制这些教材的图像，不把教学中的假想头分工写成真实模型机制。

| 原始资料 / 作者 | 借鉴的方法 | 本文如何落到图里 |
| --- | --- | --- |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) · Jay Alammar | 把每个 token 的向量画出来，逐项追踪加权汇总与多头拼接 | 图 04、06–08 展示数值格、不同读取比例和特征带拼接 |
| [Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention/) · 3Blue1Brown / Grant Sanderson | 用位置间关系与向量更新解释注意力 | 图 08、10、14、16 展示粗细连线、几何相加、逐位置路径 |
| [How might LLMs store facts](https://www.3blue1brown.com/lessons/mlp/) · 3Blue1Brown / Grant Sanderson | 把矩阵变换理解成特征组合与方向汇总 | 图 13、20 用 2→4→2 数值网络和 ReLU 折线展示升维、非线性、降维 |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) · Georgia Tech Polo Club | 把模型数据变化与可观察的视觉表示对应起来 | 图 18、19、21 展示词表分数/概率、生成阶梯与每层缓存增长；本文不宣称运行了该站的 GPT-2 |

机制本身仍以正文列出的论文与官方文档为准。简单矩阵/向量图的值能按正文手算复核；全局图中的特征颜色、头权重和神经元节点仅用于帮助读图，不作为模型可解释性证据。

## 原始技术图

作者：Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N. Gomez、Łukasz Kaiser、Illia Polosukhin。论文：[Attention Is All You Need，arXiv:1706.03762v7](https://arxiv.org/html/1706.03762v7)。

论文页面声明：Google 允许在给予适当署名的情况下，将论文图表复现用于新闻或学术作品。本目录保留以下原图用于学习与图文核对，原作者归属不因保存或整理改变。未经裁切、重绘或修改原图字节。

| 本地图片 | 论文图号 / 原始地址 | 字节数 | SHA-256 |
| --- | --- | --- | --- |
| [90-original-transformer.png](90-original-transformer.png) | Figure 1：[原图](https://arxiv.org/html/1706.03762v7/Figures/ModalNet-21.png) | 159152 | `56d69ccc192b90c3b93e5e285ff892f74753f970103939b40291945e106e9f79` |
| [91-original-attention.png](91-original-attention.png) | Figure 2 左：[原图](https://arxiv.org/html/1706.03762v7/Figures/ModalNet-19.png) | 26323 | `6d40a44e2ec3f0cdab207c768a9085b6b908bbefa82c5af6613dc8233bee6a42` |
| [92-original-multi-head.png](92-original-multi-head.png) | Figure 2 右：[原图](https://arxiv.org/html/1706.03762v7/Figures/ModalNet-20.png) | 65853 | `ba8a3ede5d40183ce8c18de249be60bec692646d53db108a3df51797a3c8634a` |

原图不替代正文中的推理循环、缓存时序和现代变体说明。图意解读位于正文各图之后。

## MoE 资料与原图

机制来源：Albert Q. Jiang 等（Mistral AI），[Mixtral of Experts，arXiv:2401.04088v1](https://arxiv.org/html/2401.04088v1)，2024-01-08；William Fedus、Barret Zoph、Noam Shazeer（Google），[Switch Transformers，arXiv:2101.03961v3](https://arxiv.org/html/2101.03961v3)，2022-06-16。读取时间均为 2026-10-08。仅整理推理架构与参数口径，未进行模型运行或性能复现。

图 24 的逐 token 路由对照参考两篇论文对独立专家 FFN 的说明；图 25 采用 Mixtral 式 Top-k 后对选中项归一化的规则，但四位专家、输入和输出全部为人工例子。图 26 的 `4P / 2P` 只计算专家参数，不是速度比或整模型参数量。Switch 的 Top-1 概率缩放不同，不能把图 25 的规则推广到所有 MoE。

保留 Mixtral Figure 1 作为结构核对材料，原创教学图不替代论文证据。原图版权与作者归属不变，保存不代表重新许可；未修改原图字节。

| 本地图片 | 原始地址 | 字节数 | SHA-256 |
| --- | --- | --- | --- |
| [93-original-moe.png](93-original-moe.png) | [Mixtral Figure 1](https://arxiv.org/html/2401.04088v1/images/smoe.png) | 145105 | `1347eb75040f1144f7ea5820cae7b4bd60c49b6bf351fb1c9e77b4c20e642c46` |

## 原创图索引

| 编号 | 内容 | 对应正文模块 |
| --- | --- | --- |
| 01 | token 数字带、源/目标层堆叠与生成回路 | 全局地图 |
| 02–03 | 文字切块、ID 圆点、起止标记与批次补齐 | 输入准备 |
| 04–05 | 按 ID 抽取矩阵一行、位置相加与 RoPE 指针旋转 | 初始表示 |
| 06–08 | 三种小矩阵投影、权重柱、Value 汇总、多头读取与拼接 | 注意力核心 |
| 09 | 屏蔽 PAD 前后的读取份额对比 | 有效位置 |
| 10–12 | 向量首尾相接、数轴刻度整理、残差旁路位置 | 子层包装 |
| 13、20 | 逐 token 小网络、2→4→4→2 数值网络与 ReLU 折线 | 逐位置特征加工 |
| 14 | 跨位置读取与逐位置加工对照 | 源序列编码 |
| 15–17 | 遮挡矩阵、源/目标二部关系图、Decoder 三道工序 | 目标前缀处理 |
| 18–19 | 分数/概率条、实框与虚框组成的生成阶梯 | 输出与循环 |
| 21–22 | 分层缓存格、新 Query 读取范围、单栈生成 | LLM 推理映射 |
| 23 | 经典双栈全部子层、各自残差与 LN、掩码、输出循环、MoE 位置 | 全模块总览 |
| 24–26 | Dense/MoE 对照、四选二数值分支、全部参数与激活参数 | 稀疏专家加工 |

颜色用于追踪图内对象：常用蓝色表示已有表示，绿色表示读取或汇总，橙色表示当前位置或增量，紫色表示特征加工，红色表示禁止或终止，灰色表示占位或未参与项。每张图的直接标签优先于概括图例。数值条高度/长度表达相应数值，特征小格的深浅辅助区分数值，不表示某个固定语义维度。箭头在几何图表示向量，在关系图表示读取，在执行图表示数据依赖，含义由图题与正文图意解读说明。
