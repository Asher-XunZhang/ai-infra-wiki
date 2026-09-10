# SGLang 源码学习系列：借鉴图片来源档案

本档案为源码学习系列中的第三方原图建立来源记录。图片只辅助理解；SGLang 行为以正文固定 commit `72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a` 为准。全部来源与图片读取时间为 **2026-09-10**。

共 **37 张不同原图、45 处正文引用，覆盖 39 篇**。所有图片已逐张查看，保留下载原始字节；其中 0 张复用 Wiki 顶层 images 下已有的相同文件，其余存于本目录。未裁剪、重绘、翻译或改写图内标注。

原图版权归原作者或机构。这里记录引用来源，不把网页可访问或项目开源解释成图片已获额外授权。原站若另有图片使用条款，应沿来源核对；本档案不替代其授权说明。

仓库分支上的浮动图片 URL 以本次下载的 SHA-256 固定内容；论文图使用下列具体 arXiv 版本。图中历史实现、训练结构、示例尺寸与跑图耗时均不作为本次 SGLang 运行证据。宽图、密集论文图可从正文的“查看原尺寸”放大阅读；gRPC 透明底图宜使用浅色背景。

[返回学习目录](../../sglang/source-study/README.md) · [查看全部 75 篇的配图选择](../../sglang/source-study/appendices/06-学习进度与版本变更记录.md#6-原图补充与逐篇配图记录)

## 来源登记

| 编号 | 原文标题与链接 | 作者 / 机构 | 发布时间 / 版本 |
| --- | --- | --- | --- |
| R01 | [Building a tokenizer, block by block](https://huggingface.co/learn/llm-course/chapter6/8?fw=pt) | Hugging Face LLM Course | 未标独立发布日期 |
| R02 | [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate) | Patrick von Platen / Hugging Face | 2020-03-01；页面注明 2023-07 更新 |
| R03 | [Continuous batching from first principles](https://huggingface.co/blog/continuous_batching) | Rémi Ouazan Reboul、Arthur Zucker、Luc Georges / Hugging Face | 2025-11-25 |
| R04 | [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/) | Lianmin Zheng、Liangsheng Yin 等 / LMSYS | 2024-01-17 |
| R05 | [SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/) | SGLang Team | 2024-12-04 |
| R06 | [Unified Radix Cache: One Tree for Hybrid Model Prefix Caching](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache/) | Zhangheng Huang、Ke Bao、Yi Zhang、Jialin Ouyang、Sicheng Pan | 2026-08-11 |
| R07 | [SGLang HiCache: Fast Hierarchical KV Caching with Your Favorite Storage Backends](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/) | Zhiqiang Xie | 2025-09-10 |
| R08 | [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) | Vinh Nguyen、Michael Carilli 等 / PyTorch | 2021-10-26；网页另标 2024-11-15 更新 |
| R09 | [NCCL User Guide — Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) | NVIDIA | 未标独立发布日期 |
| R10 | [MegatronLM: Training Billion+ Parameter Language Models Using GPU Model Parallelism](https://research.nvidia.com/labs/adlr/MegatronLM/) | NVIDIA ADLR | 2019-08-13 |
| R11 | [Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/) | SGLang Team | 2025-05-05 |
| R12 | [Achieving Efficient, Flexible, and Portable Structured Generation with XGrammar](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar) | MLC Community | 2024-11-22 |
| R13 | [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/html/2503.01840v3) | Yuhui Li、Fangyun Wei、Chao Zhang、Hongyang Zhang | arXiv v3：2025-04-23 |
| R14 | [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/html/2602.06036v2) | Jian Chen、Yesheng Liang、Zhijian Liu | arXiv v2：2026-05-28 |
| R15 | [Using FP8 and FP4 with Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) | NVIDIA Transformer Engine | 未标独立发布日期 |
| R16 | [LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) | Hugging Face PEFT | 未标独立发布日期 |
| R17 | [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/html/2405.04434v5) | DeepSeek-AI | arXiv v5：2024-06-19 |
| R18 | [Qwen2.5-VL Technical Report](https://arxiv.org/html/2502.13923v1) | Qwen Team / Alibaba Group | arXiv v1：2025-02-19 |
| R19 | [Retrieve & Re-Rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) | Sentence Transformers | 未标独立发布日期 |
| R20 | [Introduction to gRPC](https://grpc.io/docs/what-is-grpc/introduction/) | gRPC | 未标独立发布日期 |
| R21 | [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | PyTorch | 未标独立发布日期 |
| R22 | [Stable Diffusion with 🧨 Diffusers](https://huggingface.co/blog/stable_diffusion) | Suraj Patil、Pedro Cuenca、Nathan Lambert、Patrick von Platen / Hugging Face | 2022-08-22 |
| R23 | [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://raw.githubusercontent.com/deepspeedai/DeepSpeed/master/blogs/deepspeed-ulysses/README.md) | Sam Ade Jacobs 等 / DeepSpeed | 2023；原项目文章 |

## 图片与使用位置

<a id="f01"></a>
### F01 · 文字经过分词流水线

- 本地原图：[01-tokenizer-pipeline.svg](01-tokenizer-pipeline.svg)；0 0 1776 1276；118,437 bytes；本次归档。
- 来源：R01；[原始图片 URL](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/tokenization_pipeline.svg)；读取时间：2026-09-10。
- SHA-256：`c3c5caf940349c90cacf3cddc6d88f37ed8432c4df6443fb7970e3fdfe74d712`。
- 使用位置：[00-03 第 2 节](../../sglang/source-study/00-foundations/03-从文本到Token再到模型输出.md)、[02-02 第 3 节](../../sglang/source-study/02-request-lifecycle/02-Tokenizer与进程间消息通路.md)。
- 图意与限制：沿左侧从上到下读：文字规范化、预切分、子词切分，最后加入特殊标记。右侧的 Model 指分词算法，不是负责生成回答的语言模型。 这是 Hugging Face 的分词示例；转小写、WordPiece 的 ##、CLS/SEP 都取决于 tokenizer，不是所有 SGLang 模型的固定步骤。图中还未画出词表 ID 到 embedding 的计算。

<a id="f02"></a>
### F02 · 概率最高不等于每次都被选中

- 本地原图：[02-temperature.png](02-temperature.png)；2283 × 602；31,572 bytes；本次归档。
- 来源：R02；[原始图片 URL](https://huggingface.co/blog/assets/02_how-to-generate/sampling_search_with_temp.png)；读取时间：2026-09-10。
- SHA-256：`816eab79cdfd4efa9966ef3e4fc1a9071a0ea227bf2b483935462d320aceb250`。
- 使用位置：[00-03 第 4 节](../../sglang/source-study/00-foundations/03-从文本到Token再到模型输出.md)。
- 图意与限制：蓝柱是同一上下文后的候选概率，红柱标出一次抽到的 house。先看整张分布，再区分“取最大值”和“按分布抽样”。 这是教学分布，不是本系列的模型输出，也不是两种温度的对照实验。具体过滤、温度和 logprob 的计算顺序见正文。

<a id="f03"></a>
### F03 · 当前 token 写 KV，再预测下一个 token

- 本地原图：[03-kv-growth.png](03-kv-growth.png)；1348 × 466；35,735 bytes；本次归档。
- 来源：R03；[原始图片 URL](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/kv_cache.png)；读取时间：2026-09-10。
- SHA-256：`44931e339df5ebc418385dcb1880a75da44eeeaf5c4ae3a6aab853d8916bd680`。
- 使用位置：[00-04 第 4 节](../../sglang/source-study/00-foundations/04-Prefill与Decode及KVCache入门.md)、[02-04 第 6 节](../../sglang/source-study/02-request-lifecycle/04-一次Prefill到多轮Decode.md)。
- 图意与限制：蓝色列表示历史 K/V，白色 will 列是本轮新增输入；当前 Q 读取这两部分后，箭头才指向预测结果 be。右端的 be 此时还没有经过下一次 forward。 这是单请求、单步的逻辑注意力图；横向排列不代表 SGLang 物理 KV 连续存储，也不展示预分配、页表或异步完成事件。

<a id="f04"></a>
### F04 · 拼在一个 batch 里，仍是两个请求

- 本地原图：[04-ragged-batch.png](04-ragged-batch.png)；1855 × 1496；169,193 bytes；本次归档。
- 来源：R03；[原始图片 URL](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/ragged_batching.png)；读取时间：2026-09-10。
- SHA-256：`2be503cd93c27dd92aec3fd490995a25fcb747cff75c4d5eea29d158d2b27aad`。
- 使用位置：[02-03 第 6 节](../../sglang/source-study/02-request-lifecycle/03-Req与多种Batch对象的分工.md)、[05-01 第 3 节](../../sglang/source-study/05-model-execution/01-Worker与ModelRunner执行边界.md)。
- 图意与限制：沿横轴看拼接后的 token，沿纵轴看本轮 query。两个绿色三角块分别属于两个请求；块外空白说明不允许互读。批内共享一次执行，不等于共享上下文。 这是逻辑 Attention 掩码示意，不表示后端实际创建一张完整的大矩阵；请求隔离还要靠长度、起点和 KV 索引共同实现。

<a id="f05"></a>
### F05 · 每轮重新安排 token 工作量

- 本地原图：[05-continuous-batching.png](05-continuous-batching.png)；2722 × 2569；229,856 bytes；本次归档。
- 来源：R03；[原始图片 URL](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/continuous_batching.png)；读取时间：2026-09-10。
- SHA-256：`c57b1deeacea8c2e591687d1eb105722fd352492d86cdfd74c05f166fc96068d`。
- 使用位置：[03-02 第 9 节](../../sglang/source-study/03-scheduling/02-连续批处理与队列状态.md)、[03-04 第 9 节](../../sglang/source-study/03-scheduling/04-ChunkedPrefill与长请求调度.md)。
- 图意与限制：从上到下看三轮，红虚线隔开 forward。中间一轮把旧请求的一个 Decode token 与新请求的四个 Prefill token 放在一起；下一轮继续剩余输入。蓝色是已有 KV，绿色表示允许注意的位置。 原图同时引入分块和 mixed batch，示例每轮最多处理 5 个新 token。它不是 SGLang 的默认配置，也不能用来证明只开启 chunk 就一定混入 Decode。

<a id="f06"></a>
### F06 · 不同应用为什么会共享一段输入

- 本地原图：[06-radix-sharing.jpg](06-radix-sharing.jpg)；7214 × 2759；1,378,184 bytes；本次归档。
- 来源：R04；[原始图片 URL](https://www.lmsys.org/images/blog/sglang/sharing_wide.jpg)；读取时间：2026-09-10。
- SHA-256：`10309577cc86bd813dc4342f15c0b8667b832fe01bd22ad48b76460e8e189ccd`。
- 使用位置：[03-03 第 4 节](../../sglang/source-study/03-scheduling/03-排序策略与准入预算.md)、[12-01 第 8 节](../../sglang/source-study/12-extensions-and-capstone/01-FrontendDSL与Runtime的关系.md)、[12-05 第 2 节](../../sglang/source-study/12-extensions-and-capstone/05-从需求到源码与验证的综合案例.md)。
- 图意与限制：四列分别展示少样本提示、多次候选、多轮对话和树状推理。先找重复的蓝色前缀，再找各分支自己的绿色输入与黄色输出。 图说明前缀复用的应用动机；可见文字重复不保证实际 token、模型状态和缓存命名空间兼容，也不保证仍有可复用 KV。

<a id="f07"></a>
### F07 · CPU 与 GPU 怎样错开批次

- 本地原图：[07-overlap-scheduling.jpg](07-overlap-scheduling.jpg)；3314 × 334；130,211 bytes；本次归档。
- 来源：R05；[原始图片 URL](https://www.lmsys.org/images/blog/sglang_v0_4/scheduler.jpg)；读取时间：2026-09-10。
- SHA-256：`b6069f00b9361b44dcdfb8c98d42c464daeb01a56f6843b3facf09dd298be351`。
- 使用位置：[03-05 第 1 节](../../sglang/source-study/03-scheduling/05-Overlap中的CPU与GPU依赖.md)。
- 图意与限制：分别沿 CPU、GPU 两条时间轴阅读：CPU 可以在 GPU 计算当前批次时处理前一批结果、准备下一批。跨行对应关系比单个色块长度更重要。 这是 SGLang v0.4 的设计图，只用于建立重叠直觉；本次源码的 relay、stream、event 和延迟采样依赖以正文为准，图中长度不是本次测量。

<a id="f08"></a>
### F08 · 前缀树怎样生长、分裂与淘汰

- 本地原图：[08-radix-tree.jpg](08-radix-tree.jpg)；5816 × 4642；1,680,837 bytes；本次归档。
- 来源：R04；[原始图片 URL](https://www.lmsys.org/images/blog/sglang/radix_attn.jpg)；读取时间：2026-09-10。
- SHA-256：`6fdab756a9ebc87cd88c13ce172f15a83fc72acc5ea22e8ab5f8e1d0b183e354`。
- 使用位置：[04-02 第 2 节](../../sglang/source-study/04-kv-cache/02-RadixAttention与前缀匹配.md)。
- 图意与限制：按编号观察九个树快照：新输入延伸已有路径，也可能在共同前缀处分叉；图中虚线和叉号标出淘汰部分。每条边保存一段 token，而非整条请求。 这是 2024 年 RadixAttention 原图。它没有画出当前 Unified 的组件、保护链和分层回载；普通树的具体锁与淘汰规则仍需读正文。

<a id="f09"></a>
### F09 · 会话引用是保留倾向，不是硬锁

- 本地原图：[09-session-eviction.svg](09-session-eviction.svg)；0 0 1200 680；9,013 bytes；本次归档。
- 来源：R06；[原始图片 URL](https://www.lmsys.org/images/blog/unified-radix-cache/image6.svg)；读取时间：2026-09-10。
- SHA-256：`3d85bb4947f0d7d3f002cb5149cf376b14c49d2cebfca845199250214057e7b8`。
- 使用位置：[04-03 第 8 节](../../sglang/source-study/04-kv-cache/03-命中条件与缓存隔离.md)。
- 图意与限制：左边看哪些会话覆盖同一段树，右边看引用怎样影响淘汰次序；底部 Close 表示解除会话引用，Generation 用来拒绝过时代次的登记。 只对应 Unified 的 session references。图里的“仍可淘汰”针对会话引用这一软信号，不允许绕过活跃请求锁或在途传输保护，也不等同于 StreamingSession 的槽位接管。

<a id="f10"></a>
### F10 · 一棵 token 树挂接不同状态

- 本地原图：[10-unified-topology.svg](10-unified-topology.svg)；0 0 1440 790；9,833 bytes；本次归档。
- 来源：R06；[原始图片 URL](https://www.lmsys.org/images/blog/unified-radix-cache/image1.svg)；读取时间：2026-09-10。
- SHA-256：`5e99e3fbf6eef63e243aabe1723401d989e0dfbcda099569e69698e08d22f89b`。
- 使用位置：[04-04 第 1 节](../../sglang/source-study/04-kv-cache/04-UnifiedRadix与混合状态组件.md)。
- 图意与限制：左侧共用一条前缀路径，下面的蓝、绿、黄框分别表示 Full 路径、SWA 尾窗与 Mamba 检查点；右侧再独立描述数据驻留层级。 组件语义与驻留位置是两个维度；图中并列展示能力，不代表任意模型都启用全部组件，也不代表任意布局与后端组合均已验证。

<a id="f11"></a>
### F11 · 走到最深节点，不一定能复用到那里

- 本地原图：[11-unified-boundary.svg](11-unified-boundary.svg)；0 0 1440 820；9,440 bytes；本次归档。
- 来源：R06；[原始图片 URL](https://www.lmsys.org/images/blog/unified-radix-cache/image2.svg)；读取时间：2026-09-10。
- SHA-256：`b7720543cd3ce65bf12c0dafad7214888582f805940a59912b14f53b24790363`。
- 使用位置：[04-04 第 4 节](../../sglang/source-study/04-kv-cache/04-UnifiedRadix与混合状态组件.md)。
- 图意与限制：沿上方路径走到 n4，再按列检查下面三种组件。图中最后全通过的位置是 n2；Mamba 在 n4 恢复通过，仍不能弥补同列 SWA 不通过。 这是预设树快照。正文的 N4/N8/N12/N16 是另一组教学数据，二者只共享“全部组件通过才更新边界”的规则，不能混用节点编号。

<a id="f12"></a>
### F12 · 索引控制与 KV 搬运分开看

- 本地原图：[12-hicache-control.png](12-hicache-control.png)；2520 × 1980；296,640 bytes；本次归档。
- 来源：R07；[原始图片 URL](https://www.lmsys.org/images/blog/hicache/hicache_overview.png)；读取时间：2026-09-10。
- SHA-256：`f9266ae2fc4ffdc318dd3d3238b7083ed835d71869ebd4c80d0808ded52650bc`。
- 使用位置：[04-05 第 1 节](../../sglang/source-study/04-kv-cache/05-HiCache分层存储与回载.md)。
- 图意与限制：先看上方 Scheduler 怎样询问前缀树，再看 Controller 连接 GPU、CPU 和存储的搬运路径。树记录“在哪里、能否复用”，池保存真正的数据。 2025 年原图使用 HiRadixTree 等历史命名；当前 Unified、HybridCacheController 与 L2TransferEngine 的职责见正文。图中的 ACK 不能泛化成任意传输已安全退役的证明。

<a id="f13"></a>
### F13 · 同一份 KV 可以有不同排列顺序

- 本地原图：[13-hicache-layout.png](13-hicache-layout.png)；2805 × 884；150,382 bytes；本次归档。
- 来源：R07；[原始图片 URL](https://www.lmsys.org/images/blog/hicache/hicache_layout.png)；读取时间：2026-09-10。
- SHA-256：`a91d32a985c130b3a8cd2c7c5990ffa6c4f1c3da2939f13e28769cde99c8b355`。
- 使用位置：[04-05 第 3 节](../../sglang/source-study/04-kv-cache/05-HiCache分层存储与回载.md)。
- 图意与限制：用 A、B 两页追踪相同颜色：按层排列适合逐层计算，按页排列方便把一页的多层数据一起搬运。改变的是字节排列与访问方式，不是前缀身份。 这是 HiCache 的代表布局对照；实际池类型、传输接口及特殊模型布局由当前配置决定，不是所有设备与主机池都采用图中的一种固定顺序。

<a id="f14"></a>
### F14 · 把 Attention 的三次关键变换连起来

- 本地原图：[14-attention-matrices.png](14-attention-matrices.png)；3713 × 2514；262,484 bytes；本次归档。
- 来源：R03；[原始图片 URL](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/attention.png)；读取时间：2026-09-10。
- SHA-256：`64a3e6c77db40bcc6d0c13010a204a8a42d4e7dd7b6103eb15ccc4dbc8a8bb13`。
- 使用位置：[05-02 第 5 节](../../sglang/source-study/05-model-execution/02-以Llama为例读懂模型Forward.md)。
- 图意与限制：从左到右找 Q 与 K 的乘积、带因果限制的权重，再看权重如何与 V 组合。三角形表示当前位置不能读未来 token。 这是单头的逻辑计算图；Llama 的 GQA、RoPE 与实际后端实现需要正文补齐，不能据此认定融合后端会把整张注意力矩阵写入显存。

<a id="f15"></a>
### F15 · 候选数随概率集中程度改变

- 本地原图：[15-top-p.png](15-top-p.png)；2591 × 1128；69,972 bytes；本次归档。
- 来源：R02；[原始图片 URL](https://huggingface.co/blog/assets/02_how-to-generate/top_p_sampling.png)；读取时间：2026-09-10。
- SHA-256：`6cdf2ce91023691463116196e2a27629344c497a14660df76d5cf3e49eda6931`。
- 使用位置：[05-04 第 5 节](../../sglang/source-study/05-model-execution/04-Logits采样与输出概率.md)。
- 图意与限制：两幅图都按概率排序，蓝色区域累计到阈值后停止扩展。分布越集中，达到同一阈值通常需要的候选越少；边界 token 会使累计值略超过阈值。 这是 top-p 的概念示意，不是本系列实测分布。保留边界、重新归一化以及与 top-k 的组合顺序按正文对应后端核对。

<a id="f16"></a>
### F16 · 重放一组工作，减少逐个启动的开销

- 本地原图：[16-cuda-launch.png](16-cuda-launch.png)；1891 × 703；193,145 bytes；本次归档。
- 来源：R08；[原始图片 URL](https://pytorch.org/assets/images/cuda-image-2.png)；读取时间：2026-09-10。
- SHA-256：`0067710c0299743c23eccc02fab976fd88cb9742c15363c65462b411ced064a2`。
- 使用位置：[05-05 第 1 节](../../sglang/source-study/05-model-execution/05-CUDAGraph编译与执行模式.md)、[11-03 第 7 节](../../sglang/source-study/11-performance-engineering/03-从现象定位性能瓶颈.md)。
- 图意与限制：上半部分 CPU 逐个启动 A—E，GPU 中间留下空隙；下半部分预先建立图后一次发起重放，仍执行 A—E。 这是 CUDA Graph 的概念图，不是算子融合图，也不是当前 SGLang 的速度测量；能否重放仍取决于形状、地址、数据依赖和所选 runner。

<a id="f17"></a>
### F17 · AllReduce：规约结果发回每个 rank

- 本地原图：[17-allreduce.png](17-allreduce.png)；650 × 200；20,420 bytes；本次归档。
- 来源：R09；[原始图片 URL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/allreduce.png)；读取时间：2026-09-10。
- SHA-256：`56ba8399a2d0ae820a989cc1130d37264c5923d36f8852f5eeb057919537a675`。
- 使用位置：[06-01 第 4 节](../../sglang/source-study/06-parallelism/01-Rank进程组与通信基础.md)。
- 图意与限制：把各 rank 同位置的数据按操作合并，结果在每个 rank 上都有一份。图示以求和为例；不要把它理解成把不同分片依次拼接。 NCCL 文档图只定义集合通信语义；SGLang 的一次调用也可能选择其他通信实现，调用返回是否代表设备完成仍需检查异步契约。

<a id="f18"></a>
### F18 · AllGather：收齐每个 rank 的分片

- 本地原图：[18-allgather.png](18-allgather.png)；650 × 205；23,453 bytes；本次归档。
- 来源：R09；[原始图片 URL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/allgather.png)；读取时间：2026-09-10。
- SHA-256：`93b7bf717381477043bc7b8dc34d76c519eb71339af2b67401f37ac1ef0e8c81`。
- 使用位置：[06-01 第 4 节](../../sglang/source-study/06-parallelism/01-Rank进程组与通信基础.md)。
- 图意与限制：从左边每个 rank 的一块数据出发，右边每个 rank 都获得按 rank 顺序组织的完整集合。分片内容被收集，不做求和。 图省略张量维度与具体后端；实际拼接维度、rank 组和缓冲区大小由调用点决定。

<a id="f19"></a>
### F19 · AllToAll：按目标 rank 交换分片

- 本地原图：[19-alltoall.png](19-alltoall.png)；650 × 205；29,154 bytes；本次归档。
- 来源：R09；[原始图片 URL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/alltoall.png)；读取时间：2026-09-10。
- SHA-256：`664155da9f45512acce0ebce36c63c32a8cc13456a2632c685626f20749fc8bf`。
- 使用位置：[06-01 第 4 节](../../sglang/source-study/06-parallelism/01-Rank进程组与通信基础.md)。
- 图意与限制：用颜色跟踪目的地：每个发送方把不同分片交给对应接收方，接收方再收齐来自不同发送方的那一份。 这张图不是专家路由表；MoE 还需要 token 到 expert 的选择、排列、重复和回收映射。集合通信的数学语义不能替代这些账本。

<a id="f20"></a>
### F20 · 列切分和行切分为什么成对出现

- 本地原图：[20-tensor-parallel.jpg](20-tensor-parallel.jpg)；1036 × 1096；554,872 bytes；本次归档。
- 来源：R10；[原始图片 URL](https://research.nvidia.com/labs/adlr/images/megatronlm/MLP_SelfAttention.jpg)；读取时间：2026-09-10。
- SHA-256：`dfa5aae2db312f3388cc041d5656d3928ce5a685de2ce2df9d56be9220ff2549`。
- 使用位置：[06-02 第 5 节](../../sglang/source-study/06-parallelism/02-TensorParallel与层内通信.md)。
- 图意与限制：上图先把第一层输出分成两路，再让第二层各自产生部分和并在 g 处合并；下图把 Attention head 分到两路。读图时持续追踪每条路拿到的是完整输入还是部分输出。 这是 Megatron 的训练模型原图，含 GeLU、Dropout 及训练背景；当前 Llama 推理使用的激活、GQA、QKV 合并与通信分支以正文为准。图中矩阵写法也不等于 PyTorch weight 的存储轴。

<a id="f21"></a>
### F21 · 请求分组与专家分组可以不同

- 本地原图：[21-ep-parallel.png](21-ep-parallel.png)；5400 × 2350；564,696 bytes；本次归档。
- 来源：R11；[原始图片 URL](https://www.lmsys.org/images/blog/large_scale_ep/parallel-design.png)；读取时间：2026-09-10。
- SHA-256：`3513fd32f7b2793d9c996fdbcb0029fc7181d95b5e5e54ee57ca67581afd6403`。
- 使用位置：[06-05 第 2 节](../../sglang/source-study/06-parallelism/05-MoE专家并行与负载均衡.md)。
- 图意与限制：左半图的 Dense FFN 与右半图的稀疏 FFN 分开看。右侧 Dispatch 将 token 送往专家，Combine 把专家结果带回对应请求。 这是 2025 年 DeepSeek 部署的并行设计，不能套成所有 Dense 模型的 DP Attention 路径，也不保证当前任何 TP/EP 配置都使用同一 DeepEP 模式。

<a id="f22"></a>
### F22 · P/D 分工与 EP 组是两层拓扑

- 本地原图：[22-ep-architecture.png](22-ep-architecture.png)；6000 × 4500；847,528 bytes；本次归档。
- 来源：R11；[原始图片 URL](https://www.lmsys.org/images/blog/large_scale_ep/overall-arch.png)；读取时间：2026-09-10。
- SHA-256：`7fe26ae1ff93cf0212c2306c6ad1a9ed83ee6b58bde03155283ff81a7650b0eb`。
- 使用位置：[07-01 第 1 节](../../sglang/source-study/07-disaggregation/01-PD分离职责与端到端请求地图.md)。
- 图意与限制：先按 Prefill、Decode 两个大区域辨认工作，再看各区域内部的 DP Attention 和 EP；跨区域箭头是 KV 交接，不是普通层内专家通信。 原图对应 2025 年 96 张 H100 的特定部署（3 个 P 节点、9 个 D 节点）。它仅帮助区分两层拓扑，不是本篇教学配置或本次运行证据。

<a id="f23"></a>
### F23 · 接收位置先就绪，KV 字节后到达

- 本地原图：[23-pd-transfer.png](23-pd-transfer.png)；1434 × 1120；125,296 bytes；本次归档。
- 来源：R11；[原始图片 URL](https://www.lmsys.org/images/blog/large_scale_ep/pd-disaggregation.png)；读取时间：2026-09-10。
- SHA-256：`46f720676348ee5076ea65d9f7ec8d574dcc2250ab85b12026a754add4487a20`。
- 使用位置：[07-02 第 3 节](../../sglang/source-study/07-disaggregation/02-Prefill侧Bootstrap与传输状态.md)、[07-03 第 5 节](../../sglang/source-study/07-disaggregation/03-Decode侧预分配接收与就绪.md)。
- 图意与限制：沿两条竖线从上往下看：D 预分配并告知目标，P 计算并发送 KV，D 收到之后才接续计算。握手和数据箭头是不同的交接。 这是正常路径的简化时序；没有画出当前源码的队列门槛、跨 rank 检查、PREBUILT、取消和退役。收到一条通知不能单独证明可以释放或复用原地址。

<a id="f24"></a>
### F24 · 语法状态每走一步，允许的 token 就改变

- 本地原图：[24-grammar-decoding.png](24-grammar-decoding.png)；1760 × 864；226,104 bytes；本次归档。
- 来源：R12；[原始图片 URL](https://blog.mlc.ai/img/xgrammar/constrained-decoding.png)；读取时间：2026-09-10。
- SHA-256：`079cec9529fa1ba4937a7cc37c10a30290d040b66dea678f5cf1716003d1324a`。
- 使用位置：[08-01 第 4 节](../../sglang/source-study/08-advanced-generation/01-结构化输出与Grammar状态.md)。
- 图意与限制：观察下方两个黄色框：在 done 后只允许符合布尔字段的续写；值完成后，允许集合随语法位置改变。约束发生在生成过程，不是输出完再修 JSON。 token mask 约束的是格式与语法，不能保证内容事实正确；图中的可见词只是讲解粒度，真实 bitmask 对应 tokenizer 的词表 token。

<a id="f25"></a>
### F25 · CPU 可以提前准备规则，采样仍要等待正确 mask

- 本地原图：[25-grammar-overlap.png](25-grammar-overlap.png)；1999 × 675；190,718 bytes；本次归档。
- 来源：R12；[原始图片 URL](https://blog.mlc.ai/img/xgrammar/constrained-decoding-pipeline-overlap.png)；读取时间：2026-09-10。
- SHA-256：`5c7675a6934db02121945397e0f6eed8a6040d801ce5893ddac4fa9c9d67b41a`。
- 使用位置：[08-01 第 5 节](../../sglang/source-study/08-advanced-generation/01-结构化输出与Grammar状态.md)。
- 图意与限制：上下两条时间线对照串行与重叠：CPU 上的编译或 mask 准备可与 GPU forward 交错，但一次采样必须使用对应前缀状态的约束。 这是 XGrammar 的设计示意，不是当前 SGLang 每一种后端与投机组合的执行记录；延迟采样、状态推进和回滚条件以正文为准。

<a id="f26"></a>
### F26 · 比较整条候选路径，不只看下一步

- 本地原图：[26-beam-search.png](26-beam-search.png)；600 × 495；37,824 bytes；本次归档。
- 来源：R02；[原始图片 URL](https://huggingface.co/blog/assets/02_how-to-generate/beam_search.png)；读取时间：2026-09-10。
- SHA-256：`e1db549ab759682f802dbc4dbf6e3d357bafef10f71f300069cdd583eafe568a`。
- 使用位置：[08-02 第 4 节](../../sglang/source-study/08-advanced-generation/02-BeamSearch与请求分支状态.md)。
- 图意与限制：沿树把路径概率相乘：dog 的首步概率低于 nice，但继续到 has 后的累计概率可以更高。保留多个候选，才有机会发现这种路径。 这是短搜索树示例，不展示 SGLang 的 BeamGroup、累计 logprob、最终长度评分或 KV 共享结构；搜索树节点不能直接当成独立 Req。

<a id="f27"></a>
### F27 · 目标模型特征怎样帮助草稿继续生成

- 本地原图：[27-eagle-method.png](27-eagle-method.png)；647 × 987；94,100 bytes；本次归档。
- 来源：R13；[原始图片 URL](https://arxiv.org/html/2503.01840v3/method.png)；读取时间：2026-09-10。
- SHA-256：`1fc5aea1023b11258bcc47d75f528b4e0224364089c2bbdba20a0c0aeb0f86c1`。
- 使用位置：[08-04 第 3 节](../../sglang/source-study/08-advanced-generation/04-EAGLE与MTP的源码主线.md)。
- 图意与限制：从左侧目标模型抽出的低、中、高层特征开始，经中部拼接与映射送到右侧草稿；右侧编号展示草稿继续向前生成。蓝色箭头传递特征，方框内的词是 token。 这是 EAGLE-3 论文的机制图，不能覆盖旧 EAGLE、所有 MTP 或 Frozen-KV 变体；候选仍需目标模型验证，图中的训练标记不表示本次执行了训练。

<a id="f28"></a>
### F28 · 草稿块怎样同时看到上下文与待填位置

- 本地原图：[28-dflash-design.svg](28-dflash-design.svg)；0 0 624 281；107,198 bytes；本次归档。
- 来源：R14；[原始图片 URL](https://arxiv.org/html/2602.06036v2/dflash_inference_design.svg)；读取时间：2026-09-10。
- SHA-256：`3e2c3e1fb9ee3851dcac777c404822337d77d6608c03032d93e5a4e9bb2421b4`。
- 使用位置：[08-05 第 2 节](../../sglang/source-study/08-advanced-generation/05-DFlashNgram与自适应投机.md)。
- 图意与限制：蓝色块是目标上下文特征，黄色块是已知的目标 token，绿色块是待预测的 mask 位置。沿中间草稿层向右看，双向 Attention 在块内生成候选，再由输出头给出 token。 这是 DFlash 论文的草稿设计。图的右端不是“候选已全部提交”；验证、接受前缀和 KV 收尾还要走正文的 worker 链路，也不适用于 Ngram 候选来源。

<a id="f29"></a>
### F29 · 8 位如何分给符号、指数与尾数

- 本地原图：[29-fp8-formats.png](29-fp8-formats.png)；1280 × 720；12,935 bytes；本次归档。
- 来源：R15；[原始图片 URL](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/_images/fp8_formats.png)；读取时间：2026-09-10。
- SHA-256：`975d48f49f46ee22524541e4cfa9857d7c2b949a8787662bda5d5e73223b3443`。
- 使用位置：[09-02 第 3 节](../../sglang/source-study/09-model-specialization/02-量化格式与计算路径.md)。
- 图意与限制：逐行比较彩色位格：指数位影响可表示范围，尾数位影响同一区间的分辨率。E4M3 与 E5M2 都是 8 位，但取舍不同。 这张格式图不等于量化算法；scale、block 粒度、累加精度、重排与具体 kernel 仍需分别核对，不能由位数直接推导模型质量或吞吐。

<a id="f30"></a>
### F30 · 共享主权重，再叠加各自的小增量

- 本地原图：[30-lora-path.png](30-lora-path.png)；1247 × 528；69,174 bytes；本次归档。
- 来源：R16；[原始图片 URL](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_diagram.png)；读取时间：2026-09-10。
- SHA-256：`4313422c5f2755897fb8ddfc5b99251358f679647ec0f2d120a3f1ff060defe7`。
- 使用位置：[09-03 第 1 节](../../sglang/source-study/09-model-specialization/03-MultiLoRA加载调度与隔离.md)。
- 图意与限制：左侧蓝色 W 与橙色 A/B 两条路径的输出相加；中部等式说明这个加法关系。读 MultiLoRA 时，重点是不同请求如何选到各自的 A/B。 原图左侧带训练初始化，右侧画出可合并权重的情形；SGLang 的动态 MultiLoRA 不意味着每次把 adapter 永久合入 W，缩放及驻留管理按正文核对。

<a id="f31"></a>
### F31 · 改变历史状态的保存方式

- 本地原图：[31-mla-heads.png](31-mla-heads.png)；1957 × 529；100,873 bytes；本次归档。
- 来源：R17；[原始图片 URL](https://arxiv.org/html/2405.04434v5/dsattn.png)；读取时间：2026-09-10。
- SHA-256：`4595663dfc123e89769ebefd6f59fdf38f10d7d7d980d29c35af600e3dc03f9b`。
- 使用位置：[09-04 第 2 节](../../sglang/source-study/09-model-specialization/04-MLA稀疏注意力与混合状态模型.md)。
- 图意与限制：从左往右对比 MHA、GQA、MQA 与 MLA：前几种改变 K/V head 的共享方式，MLA 则引入压缩的 latent 表示。先找标为缓存的部分，再看它与各 head 的联系。 这是 DeepSeek-V2 的结构对照；RoPE 的独立部分及当前后端实际保存格式仍须读正文，也不能把 MLA 图当作 DSA、KDA 或 GDN 的状态图。

<a id="f32"></a>
### F32 · 媒体先编码，再进入语言模型序列

- 本地原图：[32-qwen-vl.jpeg](32-qwen-vl.jpeg)；4579 × 2999；1,003,229 bytes；本次归档。
- 来源：R18；[原始图片 URL](https://arxiv.org/html/2502.13923v1/figures/qwen2.5vl_arc.jpeg)；读取时间：2026-09-10。
- SHA-256：`94267a81f1932038a9fec59d256ba46faea8dd225d052e8398651da38c2df8ce`。
- 使用位置：[09-05 第 5 节](../../sglang/source-study/09-model-specialization/05-图像视频音频输入的处理链路.md)。
- 图意与限制：从底部图像与视频向上读，经 Vision Encoder 变成不同长度的视觉表示，再与文字位置一起进入 LM Decoder。视频支路额外显示帧采样与时间位置。 这是 Qwen2.5-VL 论文原图；token 数、图像尺寸和帧率是图中示例，不是所有请求默认值。音频和其他视觉模型的预处理不能直接套用。

<a id="f33"></a>
### F33 · 先找候选，再逐对比较

- 本地原图：[33-retrieve-rerank.png](33-retrieve-rerank.png)；945 × 279；19,724 bytes；本次归档。
- 来源：R19；[原始图片 URL](https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/img/InformationRetrieval.png)；读取时间：2026-09-10。
- SHA-256：`36aa9b1033831db51882f81831f1f2b757c3f6308b5b2928730283e0df3709b0`。
- 使用位置：[09-06 第 6 节](../../sglang/source-study/09-model-specialization/06-EmbeddingRerank与模型适配清单.md)。
- 图意与限制：左边的查询先进入检索器，从语料中拿候选；右边 Cross-Encoder 再联合阅读查询与候选，给出排序。两段模型解决的问题不同。 语料库和检索系统不由本篇 SGLang 接口自动提供；图中的应用流程也不表示所有 rerank 都采用分类式 cross-encoder，解码式打分见正文。

<a id="f34"></a>
### F34 · 客户端 stub 与服务端按协议交接

- 本地原图：[34-grpc-concept.svg](34-grpc-concept.svg)；0 19.282 552 326.936；114,389 bytes；本次归档。
- 来源：R20；[原始图片 URL](https://grpc.io/img/landing-2.svg)；读取时间：2026-09-10。
- SHA-256：`12a3552cc1f9c0f0c155d46edf65367c9e042220abc30c516f67b67bef825760`。
- 使用位置：[10-02 第 3 节](../../sglang/source-study/10-serving-operations/02-HTTPgRPC与Rust服务边界.md)。
- 图意与限制：沿请求、响应两组箭头看：客户端通过 stub 发送协议消息，服务端返回协议定义的结果。语言不同仍能交接，前提是双方遵守相同契约。 这是 gRPC 官方的 C++/Ruby/Android 示例，不是 SGLang 的部署图。原生 Generate 的 proto、Rust/Python bridge 与流式终态以正文为准。 原图为透明背景，深色预览下箭头对比较弱，建议在浅色背景打开原尺寸。

<a id="f35"></a>
### F35 · 把 CPU 线程与 GPU stream 对齐阅读

- 本地原图：[35-profiler-trace.png](35-profiler-trace.png)；3016 × 498；245,046 bytes；本次归档。
- 来源：R21；[原始图片 URL](https://docs.pytorch.org/tutorials/_images/trace_img.png)；读取时间：2026-09-10。
- SHA-256：`ee1600300ce66949edbdd66ecb17aa9d8e0473f86991414d33ffe81b19681d02`。
- 使用位置：[11-02 第 7 节](../../sglang/source-study/11-performance-engineering/02-Profiler与时间线阅读.md)。
- 图意与限制：先找 CPU 线程行和 GPU stream 行，再比较色块起止与空隙。上层算子范围和下层 kernel 范围不应简单逐块一一对应。 这是 PyTorch 官方 ResNet/conv2d 示例截图，不是本系列采集的 SGLang trace；图中的耗时、算子名和利用情况不能用作本次性能结论。

<a id="f36"></a>
### F36 · 反复更新 latent，最后才解码成图像

- 本地原图：[36-latent-diffusion.png](36-latent-diffusion.png)；578 × 769；54,639 bytes；本次归档。
- 来源：R22；[原始图片 URL](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/stable_diffusion.png)；读取时间：2026-09-10。
- SHA-256：`916bcd0725a61474e008b4b889c198142d9f3f1539c8faa4629dc96bdadfedb9`。
- 使用位置：[12-02 第 2 节](../../sglang/source-study/12-extensions-and-capstone/02-Diffusion服务与生成流程入门.md)。
- 图意与限制：从上方噪声与右侧文本条件进入，中间循环更新 latent，最后交给 VAE decoder 形成图片。左侧 Scheduler algorithm 指数值更新规则。 这是 Stable Diffusion/UNet 的背景图；本篇主线使用 FLUX，模型结构、文本编码器与尺寸不同。数值 scheduler 也不等同于服务层的请求调度器。

<a id="f37"></a>
### F37 · 在序列切分与 head 切分之间换布局

- 本地原图：[37-ulysses-exchange.png](37-ulysses-exchange.png)；1408 × 649；114,290 bytes；本次归档。
- 来源：R23；[原始图片 URL](https://raw.githubusercontent.com/deepspeedai/DeepSpeed/master/blogs/deepspeed-ulysses/media/image3.png)；读取时间：2026-09-10。
- SHA-256：`0c41d8e6105bfda47d57280eb992d450a5ed9dcbdcd820d3a9663b8f9346903d`。
- 使用位置：[12-03 第 3 节](../../sglang/source-study/12-extensions-and-capstone/03-Diffusion并行缓存与性能地图.md)。
- 图意与限制：跟随两根红箭头：第一次 AllToAll 把各卡的局部序列换成部分 head 的完整序列，Attention 后第二次交换恢复序列分片。 这是 DeepSpeed Ulysses 的训练项目原图，此处只借用前向布局关系；图设 P=hc=4，本篇两卡 FLUX 配置不能照抄该数字，也不能代替 Ring、PCP 或 DCP 的通信图。
