# 仓内文章整理与可读性审查

基线：本仓库 `main@279a86fe1f90fb2c5827adbda9efd28058882972`，加原有本地修改。初审与实施日期均为 2026-11-23。用户确认按方案调整后，原九组候选已完成本轮整合、分工和导航重组。

## 本轮实施结果

初审覆盖 212 份学习 Markdown：185 份正文、练习或附录，以及 27 份导航。现在其中四份旧文已退出独立正文，只保留旧地址的简短迁移说明；活跃正文、练习和附录为 181 份。没有按标题相似批量删除，也没有把固定版本统一改成“最新实现”。

六组内容已确定主文与深入资料的关系，另三组完成阅读入口、版本地图和附录查阅结构。75 篇源码课程的重复入口说明已精简，其中 17 篇长课补充了具体阅读路线和可选分支。12 篇架构导读共用一份图例，各篇仍保留自己的完整基线。下面的逐篇清单保留初审行数，方便回查原范围。

本轮只修改 Wiki 与维护检查，不改动被阅读的外部源码，不执行模型、GPU 或性能实验。发布范围只包含本轮文章整理与维护检查。原有容量地图、KDA 补充、技能文件和其他未提交工作保留在原工作区，不纳入本轮提交。更新前和大调整前分别保存了恢复备份。

## 已实施的九组调整

<a id="i01"></a>
### I01：调度沿一次批次执行讲清楚

主文为[调度循环与批次执行旅程](<../sglang/运行时/SGLang 调度循环与批次执行旅程学习文档.md>)：先区分路由、排序、准入和执行时序，再走 Normal、Overlap 与普通 PP。原调度总览的八条来源、采用或排除理由及 `6e312af8` 抽查移到主文来源附录；`279339f113` 的主线和 `72d5c5bb73` 的源码课程仍分别标明版本。

[较早的请求生命周期长文](<../sglang/运行时/SGLang 调度器请求生命周期与重叠调度学习文档.md>)保留历史对象和选批细节，作为参考，不再要求初学者先读两轮总览。运行时导航直接指向当前主文与相应深入问题。

<a id="i02"></a>
### I02：把前缀、地址和可读数据接起来

[KV Pool 主文](<../sglang/KV缓存/SGLang KV Pool、请求视图与 HiCache 工程学习文档.md#prefix-match-to-readable-kv>)统一解释计算身份、索引匹配、请求映射和数据可读条件。压缩树、节点分裂及原始技术图从旧命中定义迁入；页大小为 4、相同 6 个 token 只复用 4 个的例子明确属于仓内固定源码资料，不套到所有缓存配置。

原稿关于 child_key 哈希与碰撞的说法保留为第三方解释模型，并指出仍需核对候选保存与冲突处理。技术演进留在技术主线选读，HiCache 的匹配与搬运深读分别进入，没有拼成更长的大全。

<a id="i03"></a>
### I03：APC 归入 KV 全生命周期

[V1 KV Cache 主文](<../vllm/KV缓存/vLLM V1 KV Cache 管理全生命周期源码学习文档.md#apc-design-and-tail>)用 16-token 块和 33-token 前缀区分身份、存在、匹配粒度、固定引用与 COW。原 APC 的链式身份、尾块、登记时机和碰撞解释迁入对应步骤。

本轮重新只读核对了 `f727951d` 的 Full Attention 查找函数：所读路径先逐块查完整前缀，再按条件探测部分尾块。旧文的二分搜索模型没有固定版本，不能由链式哈希直接推成当前查找算法。新文把这处差别放在具体例子后说明，保留原文作者、链接和读取日期。

<a id="i04"></a>
### I04：插件主文与固定版本附录各有用途

[硬件插件主文](<../vllm/运行时/vLLM 硬件插件、平台选择与执行适配学习文档.md>)给出前端、EngineCore、WorkerWrapper、HCU Worker 的检查路线。[启动附录](<../vllm/运行时/vLLM 多进程启动中的插件注册与补丁生效顺序学习文档.md>)保留 HCU 目标版本的完整调用、fork/spawn 与导入协调细节。

本轮按 `33fa579e` 与 `58ad1f3b` 读取四组固定源码文件。general loader 的函数 AST 相同，Core 中先加载插件再建 Executor 的相对顺序一致；Worker 的 Triton override 插入点、平台异常日志和多模态接收缓存初始化存在差异。这是保留版本附录的具体原因，不将两个提交拼成一条启动分支。

<a id="i05"></a>
### I05：混合状态统一使用一个恢复例子

[KDA 主文](<../大模型推理/模型架构/KDA Cache 检查点、前缀复用与投机回滚学习文档.md>)集中解释共享 5000 个 token、从 4096 状态补算 904 个位置的例子，以及 COW、Snapshot、Donate 和投机提交。原所有权图连同解读迁入这里；internal checkpoint、ReplaySSM 和性能条件均保留；原有本地补充留在原工作区。

[Kimi K3 缓存与传输](<../大模型推理/模型架构/Kimi K3 混合注意力缓存与 Mooncake 状态传输学习文档.md>)不再重复另一套恢复流程，聚焦检查点密度、各引擎布局、Mooncake 和 TokenSpeed/EPD 的传输。模型配置、精度和布局仍归各自来源，没有混成统一性能结论。

<a id="i06"></a>
### I06：DeepInfer 的执行核心与服务分工集中说明

[DeepInfer 主文](<../大模型推理/分布式服务/DeepInfer 专用推理引擎与生产系统边界学习文档.md>)集中说明执行模型、提供 API 和组织集群三层交付，补入 recipe mock 示例、三引擎机制入口和状态重建／输出重复性的区别。重复的 ZOS、动态验证、Engram 细节由已有专题继续解释。

旧比较文的八张原图及逐图解读、所有独立来源、部署规模例子、确定性开关和近似恢复条件已迁入。`55374155` 的 DeepInfer、`72d5c5bb73` 的 SGLang、在线 vLLM 文档及后续 `05a303fd` 专项仍分别标明，未形成统一性能排名。

<a id="i07"></a>
### I07：PD/PP 保留一条首读路线

[分离部署导航](../sglang/分离部署/README.md)按“单请求 → 多批次 → 查步骤 → 查共识”组织。生成的步骤详解保留生成合同，复核记录退出首读主线，历史 `muxi-main` 长文继续标明历史范围。普通状态机、PP 共识短篇和长篇按观察问题与固定提交区分，没有改已发布页面 URL。

<a id="i08"></a>
### I08：四篇版本综述改为按问题回查

[版本地图](../sglang/版本研究/README.md#按问题回查版本资料)按缓存、调度、角色切换、缓存内核和性能证据连接四篇版本资料及机制先修。表格只说明本库哪份资料讨论该问题，不把提到能力当作最早引入。四篇开头突出各自变化，历史正文保留。

<a id="i09"></a>
### I09：六份附录按查阅任务展开

术语、调用链、配置、排障和模板按请求、调度、缓存、执行、并行与服务分组展开；保留每个原编号章节、表格内容和源码引用。不同用途下的同名词条没有机械去重。

[实验模板](../sglang/source-study/appendices/05-实验记录与证据模板.md)先显示通用骨架，再按问题查专项字段；历史静态填写示例移到[进度记录](../sglang/source-study/appendices/06-学习进度与版本变更记录.md#historical-reading-example)。长篇验收记录按需展开，不再挡在复制模板的路径上。

## 四篇旧文的退役与信息去向

旧文件保留约七行的迁移说明，维护已分享的 Markdown 地址；它们不再作为四篇课程列入学习导航。检查器要求这些入口有真实正文链接、有效锚点，且不能形成迁移链或继续夹带长文。

| 旧址入口 | 独立内容现在在哪里 |
| --- | --- |
| [调度总览旧址](<../sglang/运行时/SGLang 调度机制总览与学习路线.md>) | 调度旅程的四种决定、Normal/Overlap 主线及来源附录 |
| [前缀命中定义旧址](<../sglang/KV缓存/SGLang RadixAttention 前缀缓存命中定义学习文档.md>) | KV Pool 主文的前缀例子、树分裂、原图与版本说明 |
| [APC 旧址](<../vllm/KV缓存/vLLM APC 链式哈希学习文档.md>) | KV 全生命周期的链式身份、尾块、查找和登记解释 |
| [三引擎比较旧址](<../大模型推理/分布式服务/DeepInfer、vLLM 与 SGLang 的交付范围与机制比较学习文档.md>) | DeepInfer 的交付、recipe、重复性说明和原图来源附录 |

正文内部的旧文链接已改指实际章节。图片仍在顶层 `images/`，没有删除原始技术图；两份恢复文中的所有权图随讲解迁移。

## 可读性问题的处理结果

| 任务 | 本轮处理 |
| --- | --- |
| H01 调度总览与长文 | 当前主文解释一批执行，历史来源后置，旧实现作为深入参考 |
| H02 Radix、KV Pool 与 HiCache | 从一次前缀命中走到物理地址与可读数据，工程分支按需展开 |
| H03 APC 旧文 | 撤去独立正文，先给尾块例子，再解释旧模型与实际代码差别 |
| H04 硬件插件 | 每一步注明当前进程、所选实现与仍未发生的动作，保留具体版本差异 |
| H05 Kimi/KDA | 使用同一组 token 位置，恢复与所有权集中讲，布局和传输另讲 |
| H06 DeepInfer | 服务分工先行，机制沿专题继续；来源和八张原图保留为附录 |
| H07 历史 PD+PP 长文 | 增加首读路线，rank/层切分、后端和组合分支展开查，源码片段可折叠 |
| H08 PP 共识长文 | 开头先讲各级本地状态，补写过程移到来源节，协议与社区材料按需查 |
| H09 两篇 HiCache 深读 | 明确分别回答匹配长度与四向搬运；保留数值流程、失败和释放条件 |
| H10 十二篇架构导读 | 共同图例集中，单篇保留短提示、基线、图与源码链接 |
| H11 术语与源码索引 | 分组查阅；相同符号在不同问题中的用途继续保留 |
| H12 配置与排障索引 | 按问题展开原表格，保留必要限定，不把复杂单元格截成错误短结论 |
| H13 模板与进度记录 | 通用模板先行，专项字段按需展开，历史填写示例迁出模板 |
| H14 阶段 09、11、12 | 17 篇补具体阅读路径；模型变体、CI、特殊后端与配置原文按需展开 |
| H15 版本与总导航 | 四篇突出各自变化，版本地图按问题进入，维护者说明退出阅读主线 |

语言整理使用 `humanizer-zh` 的保留事实原则。已清楚的例子和必要的“不能／不等于”保留；本轮没有按词表判定文章质量，也没有把所有专业段落改成口语。75 篇课程另统一精简重复的架构导读提示，保留全部链接。

## 目录与来源保留

上一阶段完成的三处归档继续有效：Kimi K3 协同优化归 `sglang/性能工程/`，通用 PP 共识长文归 `sglang/并行方法/`，未定位一手实现的 PersistentKV 资料归 `大模型推理/性能工程/`。本轮没有再增加细碎主题目录，`source-study/` 阶段编号保留。

历史分支、不同固定提交、生成产物和静态验收记录各有用途，不因篇幅长或主题相同而整篇删除。迁移核对同时检查原图、外部来源和独立数值条件；重复例子收敛为一个例子，不把每组教学数字当成必须重复保存的独立结论。

## 验证记录

文档检查、五项迁移入口测试、导航与基础课生成一致性、PP 步骤检查、Pages 构建及 `git diff --check` 已通过。全量 212 份学习文档的引用定义与折叠结构检查通过；四组迁移的外部来源与合并范围内的原始技术图引用均完整保留。除已退役正文与明确收敛的 Kimi 重复算例外，130 份有改动的学习文件保留原有全部代码／Mermaid 图块；生成的 PP 步骤详解逐字节未变。验证范围包括本地链接与锚点、迁移入口、引用定义、折叠结构、生成文档一致性和 Pages 构建；不包含真实模型或设备执行。

## 逐篇处置清单

以下记录本轮审查基线的全部 185 份非 README 学习文件。`Ixx` 对应上文已实施的调整；“保留”表示保留独立问题和固定版本，不表示全文所有句子均已改写。H 编号标出本轮处理的结构或语言问题；具体动作以上表为准。行数取审查开始时快照，迁移与局部改写后可能小幅变化。

| 文章 | 类型 | 原行数 | 内容与目录决定 | 可读性处理 |
| --- | --- | ---: | --- | --- |
| [HiCache 下 SGLang L1、L2、L3 与上传回载源码学习文档](<../sglang/KV缓存/HiCache 下 SGLang L1、L2、L3 与上传回载源码学习文档.md>) | 专题文章 | 821 | 保留独立问题与正文 | 分清主线与后端选项，按方向查阅 |
| [HiCache 前缀命中源码学习文档](<../sglang/KV缓存/HiCache 前缀命中源码学习文档.md>) | 专题文章 | 618 | 保留独立问题与正文 | 先走一次命中，再展开实现分支 |
| [Mooncake 与 SGLang HiCache 学习文档](<../sglang/KV缓存/Mooncake 与 SGLang HiCache 学习文档.md>) | 专题文章 | 347 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang KV Pool、请求视图与 HiCache 工程学习文档](<../sglang/KV缓存/SGLang KV Pool、请求视图与 HiCache 工程学习文档.md>) | 专题文章 | 497 | [I02](#i02)；已落实正文分工 | 先解释池、映射和树，再讲所有权与工程选项 |
| [SGLang KV 映射与共享回收学习文档](<../sglang/KV缓存/SGLang KV 映射与共享回收学习文档.md>) | 专题文章 | 164 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang RadixAttention 与 HiCache KV Cache 技术主线学习文档](<../sglang/KV缓存/SGLang RadixAttention 与 HiCache KV Cache 技术主线学习文档.md>) | 专题文章 | 313 | [I02](#i02)；已落实正文分工 | 保留；逐句润色未执行 |
| [SGLang RadixAttention 前缀缓存命中定义学习文档](<../sglang/KV缓存/SGLang%20KV%20Pool、请求视图与%20HiCache%20工程学习文档.md#prefix-match-to-readable-kv>) | 专题文章 | 219 | [I02](#i02)；独立正文已退役，保留旧址入口 | 保留；逐句润色未执行 |
| [SGLang Unified Radix Cache 学习文档](<../sglang/KV缓存/SGLang Unified Radix Cache 学习文档.md>) | 专题文章 | 281 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [推理系统与 SGLang 职责地图](<../sglang/source-study/00-foundations/01-推理系统与SGLang职责地图.md>) | 源码课程 | 225 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [读源码必备的 Python 与并发基础](<../sglang/source-study/00-foundations/02-读源码必备的Python与并发基础.md>) | 源码课程 | 257 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [从文本到 Token 再到模型输出](<../sglang/source-study/00-foundations/03-从文本到Token再到模型输出.md>) | 源码课程 | 249 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Prefill 与 Decode 及 KV Cache 入门](<../sglang/source-study/00-foundations/04-Prefill与Decode及KVCache入门.md>) | 源码课程 | 245 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [吞吐、延迟与显存的基本账本](<../sglang/source-study/00-foundations/05-吞吐延迟与显存的基本账本.md>) | 源码课程 | 278 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [源码目录与最短阅读路线](<../sglang/source-study/01-getting-started/01-源码目录与最短阅读路线.md>) | 源码课程 | 240 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [环境依赖与最小运行准备](<../sglang/source-study/01-getting-started/02-环境依赖与最小运行准备.md>) | 源码课程 | 233 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [从 CLI 到服务进程启动](<../sglang/source-study/01-getting-started/03-从CLI到服务进程启动.md>) | 源码课程 | 249 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [从参数声明到最终生效配置](<../sglang/source-study/01-getting-started/04-从参数声明到最终生效配置.md>) | 源码课程 | 275 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [最小请求与启动故障定位](<../sglang/source-study/01-getting-started/05-最小请求与启动故障定位.md>) | 源码课程 | 281 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [API 协议到内部请求对象](<../sglang/source-study/02-request-lifecycle/01-API协议到内部请求对象.md>) | 源码课程 | 303 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Tokenizer 与进程间消息通路](<../sglang/source-study/02-request-lifecycle/02-Tokenizer与进程间消息通路.md>) | 源码课程 | 325 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Req 与多种 Batch 对象的分工](<../sglang/source-study/02-request-lifecycle/03-Req与多种Batch对象的分工.md>) | 源码课程 | 300 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [一次 Prefill 到多轮 Decode](<../sglang/source-study/02-request-lifecycle/04-一次Prefill到多轮Decode.md>) | 源码课程 | 334 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Detokenizer 与流式输出](<../sglang/source-study/02-request-lifecycle/05-Detokenizer与流式输出.md>) | 源码课程 | 353 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [完成、取消与资源释放](<../sglang/source-study/02-request-lifecycle/06-完成取消与资源释放.md>) | 源码课程 | 417 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Normal Event Loop 与调度主循环](<../sglang/source-study/03-scheduling/01-NormalEventLoop与调度主循环.md>) | 源码课程 | 395 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [连续批处理与队列状态](<../sglang/source-study/03-scheduling/02-连续批处理与队列状态.md>) | 源码课程 | 419 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [排序策略与准入预算](<../sglang/source-study/03-scheduling/03-排序策略与准入预算.md>) | 源码课程 | 460 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Chunked Prefill 与长请求调度](<../sglang/source-study/03-scheduling/04-ChunkedPrefill与长请求调度.md>) | 源码课程 | 440 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Overlap 中的 CPU 与 GPU 依赖](<../sglang/source-study/03-scheduling/05-Overlap中的CPU与GPU依赖.md>) | 源码课程 | 427 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [回撤、背压、饥饿与调度排障](<../sglang/source-study/03-scheduling/06-回撤背压饥饿与调度排障.md>) | 源码课程 | 447 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [请求视图、物理槽位与分配器](<../sglang/source-study/04-kv-cache/01-请求视图物理槽位与分配器.md>) | 源码课程 | 420 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [RadixAttention 与前缀匹配](<../sglang/source-study/04-kv-cache/02-RadixAttention与前缀匹配.md>) | 源码课程 | 494 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [命中条件与缓存隔离](<../sglang/source-study/04-kv-cache/03-命中条件与缓存隔离.md>) | 源码课程 | 465 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [UnifiedRadix 与混合状态组件](<../sglang/source-study/04-kv-cache/04-UnifiedRadix与混合状态组件.md>) | 源码课程 | 573 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [HiCache 分层存储与回载](<../sglang/source-study/04-kv-cache/05-HiCache分层存储与回载.md>) | 源码课程 | 560 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [容量规划、碎片与显存回收](<../sglang/source-study/04-kv-cache/06-容量规划碎片与显存回收.md>) | 源码课程 | 474 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [KV 缓存全生命周期与排障](<../sglang/source-study/04-kv-cache/07-KV缓存全生命周期与排障.md>) | 源码课程 | 445 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Worker 与 ModelRunner 执行边界](<../sglang/source-study/05-model-execution/01-Worker与ModelRunner执行边界.md>) | 源码课程 | 478 | 保留课程／练习与固定基线 | 复制与结果消费长段已改，继续按一次 batch 检查 |
| [以 Llama 为例读懂模型 Forward](<../sglang/source-study/05-model-execution/02-以Llama为例读懂模型Forward.md>) | 源码课程 | 511 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Attention 后端与执行元数据](<../sglang/source-study/05-model-execution/03-Attention后端与执行元数据.md>) | 源码课程 | 423 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Logits 采样与输出概率](<../sglang/source-study/05-model-execution/04-Logits采样与输出概率.md>) | 源码课程 | 446 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [CUDA Graph、编译与执行模式](<../sglang/source-study/05-model-execution/05-CUDAGraph编译与执行模式.md>) | 源码课程 | 438 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Kernel 注册、选择与实现阅读](<../sglang/source-study/05-model-execution/06-Kernel注册选择与实现阅读.md>) | 源码课程 | 470 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Rank、进程组与通信基础](<../sglang/source-study/06-parallelism/01-Rank进程组与通信基础.md>) | 源码课程 | 490 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Tensor Parallel 与层内通信](<../sglang/source-study/06-parallelism/02-TensorParallel与层内通信.md>) | 源码课程 | 532 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Data Parallel 与 DP Attention](<../sglang/source-study/06-parallelism/03-DataParallel与DPAttention.md>) | 源码课程 | 497 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Pipeline Parallel 与 Microbatch](<../sglang/source-study/06-parallelism/04-PipelineParallel与Microbatch.md>) | 源码课程 | 530 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [MoE 专家并行与负载均衡](<../sglang/source-study/06-parallelism/05-MoE专家并行与负载均衡.md>) | 源码课程 | 562 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Context Parallel 与并行组合](<../sglang/source-study/06-parallelism/06-ContextParallel与并行组合.md>) | 源码课程 | 523 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [PD 分离职责与端到端请求地图](<../sglang/source-study/07-disaggregation/01-PD分离职责与端到端请求地图.md>) | 源码课程 | 469 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Prefill 侧 Bootstrap 与传输状态](<../sglang/source-study/07-disaggregation/02-Prefill侧Bootstrap与传输状态.md>) | 源码课程 | 477 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Decode 侧预分配、接收与就绪](<../sglang/source-study/07-disaggregation/03-Decode侧预分配接收与就绪.md>) | 源码课程 | 516 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [KV 传输接口与后端实现地图](<../sglang/source-study/07-disaggregation/04-KV传输接口与后端实现地图.md>) | 源码课程 | 457 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [PD 异常取消与资源退役](<../sglang/source-study/07-disaggregation/05-PD异常取消与资源退役.md>) | 源码课程 | 486 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [PD 与 PP 及 Encoder 分离的组合](<../sglang/source-study/07-disaggregation/06-PD与PP及Encoder分离的组合.md>) | 源码课程 | 428 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [结构化输出与 Grammar 状态](<../sglang/source-study/08-advanced-generation/01-结构化输出与Grammar状态.md>) | 源码课程 | 501 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Beam Search 与请求分支状态](<../sglang/source-study/08-advanced-generation/02-BeamSearch与请求分支状态.md>) | 源码课程 | 552 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [投机解码的 Draft、Verify、Commit](<../sglang/source-study/08-advanced-generation/03-投机解码的DraftVerifyCommit.md>) | 源码课程 | 496 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [EAGLE 与 MTP 的源码主线](<../sglang/source-study/08-advanced-generation/04-EAGLE与MTP的源码主线.md>) | 源码课程 | 538 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [DFlash、Ngram 与自适应投机](<../sglang/source-study/08-advanced-generation/05-DFlashNgram与自适应投机.md>) | 源码课程 | 557 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [投机中的 Overlap、KV 与组合约束](<../sglang/source-study/08-advanced-generation/06-投机中的OverlapKV与组合约束.md>) | 源码课程 | 525 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [模型注册、配置与权重加载](<../sglang/source-study/09-model-specialization/01-模型注册配置与权重加载.md>) | 源码课程 | 565 | 保留课程／练习与固定基线 | 长篇课程分主线与例外；逐句改写未执行 |
| [量化格式与计算路径](<../sglang/source-study/09-model-specialization/02-量化格式与计算路径.md>) | 源码课程 | 569 | 保留课程／练习与固定基线 | 长篇课程分主线与例外；逐句改写未执行 |
| [MultiLoRA 加载、调度与隔离](<../sglang/source-study/09-model-specialization/03-MultiLoRA加载调度与隔离.md>) | 源码课程 | 537 | 保留课程／练习与固定基线 | 长篇课程分主线与例外；逐句改写未执行 |
| [MLA、稀疏注意力与混合状态模型](<../sglang/source-study/09-model-specialization/04-MLA稀疏注意力与混合状态模型.md>) | 源码课程 | 668 | 保留课程／练习与固定基线 | 长篇课程分主线与例外；逐句改写未执行 |
| [图像、视频、音频输入的处理链路](<../sglang/source-study/09-model-specialization/05-图像视频音频输入的处理链路.md>) | 源码课程 | 635 | 保留课程／练习与固定基线 | 长篇课程分主线与例外；逐句改写未执行 |
| [Embedding、Rerank 与模型适配清单](<../sglang/source-study/09-model-specialization/06-EmbeddingRerank与模型适配清单.md>) | 源码课程 | 611 | 保留课程／练习与固定基线 | 长篇课程分主线与例外；逐句改写未执行 |
| [Model Gateway 注册、路由与缓存亲和](<../sglang/source-study/10-serving-operations/01-ModelGateway注册路由与缓存亲和.md>) | 源码课程 | 569 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [HTTP、gRPC 与 Rust 服务边界](<../sglang/source-study/10-serving-operations/02-HTTPgRPC与Rust服务边界.md>) | 源码课程 | 521 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [健康检查、超时、限流与优雅退出](<../sglang/source-study/10-serving-operations/03-健康检查超时限流与优雅退出.md>) | 源码课程 | 586 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Metrics、日志与 Trace 关联](<../sglang/source-study/10-serving-operations/04-Metrics日志与Trace关联.md>) | 源码课程 | 531 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [权重更新、暂停恢复与 RL 接口](<../sglang/source-study/10-serving-operations/05-权重更新暂停恢复与RL接口.md>) | 源码课程 | 557 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [Benchmark 设计与指标口径](<../sglang/source-study/11-performance-engineering/01-Benchmark设计与指标口径.md>) | 源码课程 | 475 | 保留课程／练习与固定基线 | 长篇课程分主线与例外；逐句改写未执行 |
| [Profiler 与时间线阅读](<../sglang/source-study/11-performance-engineering/02-Profiler与时间线阅读.md>) | 源码课程 | 552 | 保留课程／练习与固定基线 | 长篇课程分主线与例外；逐句改写未执行 |
| [从现象定位性能瓶颈](<../sglang/source-study/11-performance-engineering/03-从现象定位性能瓶颈.md>) | 源码课程 | 547 | 保留课程／练习与固定基线 | 长篇课程分主线与例外；逐句改写未执行 |
| [参数对照与性能实验记录](<../sglang/source-study/11-performance-engineering/04-参数对照与性能实验记录.md>) | 源码课程 | 652 | 保留课程／练习与固定基线 | 长篇课程分主线与例外；逐句改写未执行 |
| [测试分层、精度与失败复现](<../sglang/source-study/11-performance-engineering/05-测试分层精度与失败复现.md>) | 源码课程 | 705 | 保留课程／练习与固定基线 | 日志段落已改，CI 细节需按排查任务分层 |
| [源码变更阅读与版本回归检查](<../sglang/source-study/11-performance-engineering/06-源码变更阅读与版本回归检查.md>) | 源码课程 | 583 | 保留课程／练习与固定基线 | 长篇课程分主线与例外；逐句改写未执行 |
| [Frontend DSL 与 Runtime 的关系](<../sglang/source-study/12-extensions-and-capstone/01-FrontendDSL与Runtime的关系.md>) | 源码课程 | 646 | 保留课程／练习与固定基线 | 长篇课程分主线与例外；逐句改写未执行 |
| [Diffusion 服务与生成流程入门](<../sglang/source-study/12-extensions-and-capstone/02-Diffusion服务与生成流程入门.md>) | 源码课程 | 666 | 保留课程／练习与固定基线 | 长篇课程分主线与例外；逐句改写未执行 |
| [Diffusion 并行缓存与性能地图](<../sglang/source-study/12-extensions-and-capstone/03-Diffusion并行缓存与性能地图.md>) | 源码课程 | 672 | 保留课程／练习与固定基线 | 长篇课程分主线与例外；逐句改写未执行 |
| [硬件后端插件与生态边界](<../sglang/source-study/12-extensions-and-capstone/04-硬件后端插件与生态边界.md>) | 源码课程 | 581 | 保留课程／练习与固定基线 | 长篇课程分主线与例外；逐句改写未执行 |
| [从需求到源码与验证的综合案例](<../sglang/source-study/12-extensions-and-capstone/05-从需求到源码与验证的综合案例.md>) | 源码课程 | 566 | 保留课程／练习与固定基线 | 长篇课程分主线与例外；逐句改写未执行 |
| [术语与对象速查](<../sglang/source-study/appendices/01-术语与对象速查.md>) | 附录 | 1358 | [I09](#i09)；已落实正文分工 | 按问题入口已补；表格精简待做，历史记录保留 |
| [源码入口与调用链索引](<../sglang/source-study/appendices/02-源码入口与调用链索引.md>) | 附录 | 2310 | [I09](#i09)；已落实正文分工 | 按问题入口已补；表格精简待做，历史记录保留 |
| [配置解析与功能兼容矩阵](<../sglang/source-study/appendices/03-配置解析与功能兼容矩阵.md>) | 附录 | 2027 | [I09](#i09)；已落实正文分工 | 按问题入口已补；表格精简待做，历史记录保留 |
| [症状到源码的排障索引](<../sglang/source-study/appendices/04-症状到源码的排障索引.md>) | 附录 | 1566 | [I09](#i09)；已落实正文分工 | 按问题入口已补；表格精简待做，历史记录保留 |
| [实验记录与证据模板](<../sglang/source-study/appendices/05-实验记录与证据模板.md>) | 附录 | 1405 | [I09](#i09)；已落实正文分工 | 按问题入口已补；表格精简待做，历史记录保留 |
| [学习进度与版本变更记录](<../sglang/source-study/appendices/06-学习进度与版本变更记录.md>) | 附录 | 588 | [I09](#i09)；已落实正文分工 | 按问题入口已补；表格精简待做，历史记录保留 |
| [M01 · SGLang全景与四种架构视图](<../sglang/source-study/architecture/01-SGLang全景与四种架构视图.md>) | 架构导读 | 153 | 保留架构层，不与源码课机械合并 | 图例已集中，单篇基线与细节按需展开 |
| [M02 · 启动装配与运行时边界](<../sglang/source-study/architecture/02-启动装配与运行时边界.md>) | 架构导读 | 121 | 保留架构层，不与源码课机械合并 | 图例已集中，单篇基线与细节按需展开 |
| [M03 · 请求对象与生命周期全景](<../sglang/source-study/architecture/03-请求对象与生命周期全景.md>) | 架构导读 | 138 | 保留架构层，不与源码课机械合并 | 图例已集中，单篇基线与细节按需展开 |
| [M04 · 调度器架构与时间模型](<../sglang/source-study/architecture/04-调度器架构与时间模型.md>) | 架构导读 | 126 | 保留架构层，不与源码课机械合并 | 图例已集中，单篇基线与细节按需展开 |
| [M05 · 缓存架构与资源所有权](<../sglang/source-study/architecture/05-缓存架构与资源所有权.md>) | 架构导读 | 125 | 保留架构层，不与源码课机械合并 | 图例已集中，单篇基线与细节按需展开 |
| [M06 · 模型执行与算子分层](<../sglang/source-study/architecture/06-模型执行与算子分层.md>) | 架构导读 | 123 | 保留架构层，不与源码课机械合并 | 图例已集中，单篇基线与细节按需展开 |
| [M07 · 并行部署与通信拓扑](<../sglang/source-study/architecture/07-并行部署与通信拓扑.md>) | 架构导读 | 133 | 保留架构层，不与源码课机械合并 | 图例已集中，单篇基线与细节按需展开 |
| [M08 · PD与Encoder分离的三条通路](<../sglang/source-study/architecture/08-PD与Encoder分离的三条通路.md>) | 架构导读 | 144 | 保留架构层，不与源码课机械合并 | 图例已集中，单篇基线与细节按需展开 |
| [M09 · 高级生成特性插入位置](<../sglang/source-study/architecture/09-高级生成特性插入位置.md>) | 架构导读 | 138 | 保留架构层，不与源码课机械合并 | 图例已集中，单篇基线与细节按需展开 |
| [M10 · 模型装配状态形态与输入输出](<../sglang/source-study/architecture/10-模型装配状态形态与输入输出.md>) | 架构导读 | 145 | 保留架构层，不与源码课机械合并 | 图例已集中，单篇基线与细节按需展开 |
| [M11 · 网关服务治理与性能诊断全景](<../sglang/source-study/architecture/11-网关服务治理与性能诊断全景.md>) | 架构导读 | 130 | 保留架构层，不与源码课机械合并 | 图例已集中，单篇基线与细节按需展开 |
| [M12 · DSL与Diffusion及插件边界](<../sglang/source-study/architecture/12-DSL与Diffusion及插件边界.md>) | 架构导读 | 133 | 保留架构层，不与源码课机械合并 | 图例已集中，单篇基线与细节按需展开 |
| [SGLang 跨模块机制验证练习](<../sglang/source-study/机制验证练习.md>) | 源码课程 | 113 | 保留课程／练习与固定基线 | 保留；逐句润色未执行 |
| [PD Prefill：PP=3、5 个 micro-batch 的 loop 细分](<../sglang/分离部署/PD Prefill PP loop 交互图.md>) | 专题文章 | 245 | [I07](#i07)；已落实正文分工 | 保留；逐句润色未执行 |
| [PD Prefill PP loop：逐步理解行为与目的](<../sglang/分离部署/PD Prefill PP loop 步骤详解.md>) | 专题文章 | 872 | [I07](#i07)；已落实正文分工 | 从生成源改善重复步骤解释，不能直接手改 |
| [PD Prefill PP loop Pages 源码复核记录](<../sglang/分离部署/PD Prefill PP loop 源码复核记录.md>) | 专题文章 | 144 | [I07](#i07)；已落实正文分工 | 保留；逐句润色未执行 |
| [PD Prefill PP=3 请求生命周期源码学习文档](<../sglang/分离部署/PD Prefill PP=3 请求生命周期源码学习文档.md>) | 专题文章 | 428 | [I07](#i07)；已落实正文分工 | 保留；逐句润色未执行 |
| [PD Prefill 数据流状态机](<../sglang/分离部署/PD Prefill 数据流状态机.md>) | 专题文章 | 199 | [I07](#i07)；已落实正文分工 | 保留；逐句润色未执行 |
| [PD 分离下的 PP 源码学习文档](<../sglang/分离部署/PD 分离下的 PP 源码学习文档.md>) | 专题文章 | 2931 | [I07](#i07)；已落实正文分工 | 历史长文先给最短路线；制作说明已改 |
| [SGLang PP 共识机制源码学习文档](<../sglang/并行方法/SGLang PP 共识机制源码学习文档.md>) | 专题文章 | 1094 | [I07](#i07)；已落实正文分工；已迁移目录 | 三轮阅读路线需减负，复核过程后移 |
| [SGLang PP 共识与缓存 ACK 机制](<../sglang/并行方法/SGLang PP 共识与缓存 ACK 机制学习文档.md>) | 专题文章 | 178 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang Pipeline Parallel 模式学习文档](<../sglang/并行方法/SGLang Pipeline Parallel 模式学习文档.md>) | 专题文章 | 628 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang 并行分工与执行拓扑](<../sglang/并行方法/SGLang 并行分工与执行拓扑学习文档.md>) | 专题文章 | 227 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang 数据并行、负载均衡与专家并行边界学习文档](<../sglang/并行方法/SGLang 数据并行、负载均衡与专家并行边界学习文档.md>) | 专题文章 | 436 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang 通信与传输机制学习文档](<../sglang/并行方法/SGLang 通信与传输机制学习文档.md>) | 专题文章 | 297 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang GLM-5.2 NVFP4 推理优化案例学习文档](<../sglang/性能工程/SGLang GLM-5.2 NVFP4 推理优化案例学习文档.md>) | 专题文章 | 473 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang Torch Profiler 与 Trace 性能分析学习文档](<../sglang/性能工程/SGLang Torch Profiler 与 Trace 性能分析学习文档.md>) | 专题文章 | 246 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang v0.5.16 24GB 显存调优案例学习文档](<../sglang/性能工程/SGLang v0.5.16 24GB 显存调优案例学习文档.md>) | 专题文章 | 632 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang 指标口径与性能定位](<../sglang/性能工程/SGLang 指标口径与性能定位学习文档.md>) | 专题文章 | 239 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang 服务生命周期与请求治理学习文档](<../sglang/服务运维/SGLang 服务生命周期与请求治理学习文档.md>) | 专题文章 | 243 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang Kimi K3 推理协同优化学习文档](<../sglang/性能工程/SGLang Kimi K3 推理协同优化学习文档.md>) | 专题文章 | 182 | 保留独立问题与正文；已迁移目录 | 保留；逐句润色未执行 |
| [SGLang v0.5.18 推理系统协同演进学习文档](<../sglang/版本研究/SGLang v0.5.18 推理系统协同演进学习文档.md>) | 专题文章 | 240 | [I08](#i08)；已落实正文分工 | 突出该版变化；来源条件靠近具体结论 |
| [SGLang v0.5.19 功能组合与升级边界学习文档](<../sglang/版本研究/SGLang v0.5.19 功能组合与升级边界学习文档.md>) | 专题文章 | 193 | [I08](#i08)；已落实正文分工 | 突出该版变化；来源条件靠近具体结论 |
| [SGLang v0.5.20 缓存调度与 Serving 协同学习文档](<../sglang/版本研究/SGLang v0.5.20 缓存调度与 Serving 协同学习文档.md>) | 专题文章 | 345 | [I08](#i08)；已落实正文分工 | 突出该版变化；来源条件靠近具体结论 |
| [SGLang v0.5.21 动态角色、缓存内核与升级边界学习文档](<../sglang/版本研究/SGLang v0.5.21 动态角色、缓存内核与升级边界学习文档.md>) | 专题文章 | 139 | [I08](#i08)；已落实正文分工 | 突出该版变化；来源条件靠近具体结论 |
| [SGLang Breakable CUDA Graph 与 Prefill 捕获学习文档](<../sglang/运行时/SGLang Breakable CUDA Graph 与 Prefill 捕获学习文档.md>) | 专题文章 | 265 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang Chunked Prefill 与调度器显存预算学习文档](<../sglang/运行时/SGLang Chunked Prefill 与调度器显存预算学习文档.md>) | 专题文章 | 707 | 保留独立问题与正文 | 长篇预算解释按一个请求展开，减少例外前置 |
| [SGLang Req 状态与队列流转学习文档](<../sglang/运行时/SGLang Req 状态与队列流转学习文档.md>) | 专题文章 | 186 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang 启动、平台选择与插件生效顺序学习文档](<../sglang/运行时/SGLang 启动、平台选择与插件生效顺序学习文档.md>) | 专题文章 | 216 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang 执行分层与硬件机制](<../sglang/运行时/SGLang 执行分层与硬件机制学习文档.md>) | 专题文章 | 162 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang 推理全景学习指南](<../sglang/运行时/SGLang 推理全景学习指南.md>) | 专题文章 | 193 | 保留独立问题与正文 | 操作长段已分开，后续按入门主线复核 |
| [SGLang 普通请求运行时与资源生命周期学习文档](<../sglang/运行时/SGLang 普通请求运行时与资源生命周期学习文档.md>) | 专题文章 | 227 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang 系统边界与学习地图学习文档](<../sglang/运行时/SGLang 系统边界与学习地图学习文档.md>) | 专题文章 | 173 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang 调度准入与容量回撤学习文档](<../sglang/运行时/SGLang 调度准入与容量回撤学习文档.md>) | 专题文章 | 174 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang 调度器请求生命周期与重叠调度学习文档](<../sglang/运行时/SGLang 调度器请求生命周期与重叠调度学习文档.md>) | 专题文章 | 665 | [I01](#i01)；已落实正文分工 | 历史对象图与现行讲解分开，收拢重复总览 |
| [SGLang 调度循环与批次执行旅程](<../sglang/运行时/SGLang 调度循环与批次执行旅程学习文档.md>) | 专题文章 | 226 | [I01](#i01)；已落实正文分工 | 保留；逐句润色未执行 |
| [SGLang 调度机制总览与学习路线](<../sglang/运行时/SGLang%20调度循环与批次执行旅程学习文档.md#scheduler-decisions>) | 专题文章 | 209 | [I01](#i01)；独立正文已退役，保留旧址入口 | 先讲调度职责，再放来源筛选；局部已改 |
| [SGLang 模型状态与高级生成机制学习文档](<../sglang/高级生成/SGLang 模型状态与高级生成机制学习文档.md>) | 专题文章 | 248 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [vLLM APC 链式哈希学习文档](<../vllm/KV缓存/vLLM%20V1%20KV%20Cache%20管理全生命周期源码学习文档.md#apc-design-and-tail>) | 专题文章 | 228 | [I03](#i03)；独立正文已退役，保留旧址入口 | 旧模型纠偏与机制解释放在同一处 |
| [vLLM V1 KV Cache 管理全生命周期源码学习文档](<../vllm/KV缓存/vLLM V1 KV Cache 管理全生命周期源码学习文档.md>) | 专题文章 | 842 | [I03](#i03)；已落实正文分工 | 先走块生命周期，再查特殊缓存与清零 |
| [Mooncake 与 vLLM 接入地图学习文档](<../vllm/分离部署/Mooncake 与 vLLM 接入地图学习文档.md>) | 专题文章 | 369 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [vLLM V1 KV Connector 架构与实现地图源码学习文档](<../vllm/分离部署/vLLM V1 KV Connector 架构与实现地图源码学习文档.md>) | 专题文章 | 925 | 保留独立问题与正文 | 公共流程先走通，全部后端能力表后读 |
| [vLLM DCP KV Cache 去重与 LSE 合并学习文档](<../vllm/并行方法/vLLM DCP KV Cache 去重与 LSE 合并学习文档.md>) | 专题文章 | 565 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [vLLM Expert Parallel 与 EPLB 学习文档](<../vllm/并行方法/vLLM Expert Parallel 与 EPLB 学习文档.md>) | 专题文章 | 640 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [vLLM Pipeline Parallel 流水线并行学习文档](<../vllm/并行方法/vLLM Pipeline Parallel 流水线并行学习文档.md>) | 专题文章 | 615 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [PersistentKV 长上下文注意力调度学习文档](<../大模型推理/性能工程/PersistentKV 长上下文注意力调度学习文档.md>) | 专题文章 | 396 | 保留独立问题与正文；已迁移目录 | 保留；逐句润色未执行 |
| [vLLM Chunked Prefill 与 Block Size 学习文档](<../vllm/运行时/vLLM Chunked Prefill 与 Block Size 学习文档.md>) | 专题文章 | 389 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [vLLM 从连续批处理到 PagedAttention 的引擎工作流学习文档](<../vllm/运行时/vLLM 从连续批处理到 PagedAttention 的引擎工作流学习文档.md>) | 专题文章 | 362 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [vLLM 多进程启动中的插件注册与补丁生效顺序学习文档](<../vllm/运行时/vLLM 多进程启动中的插件注册与补丁生效顺序学习文档.md>) | 专题文章 | 185 | [I04](#i04)；已落实正文分工 | 同一插件贯穿三个进程，先解释装配术语 |
| [vLLM 硬件插件、平台选择与执行适配学习文档](<../vllm/运行时/vLLM 硬件插件、平台选择与执行适配学习文档.md>) | 专题文章 | 282 | [I04](#i04)；已落实正文分工 | 启动链与设备案例按层展开 |
| [Agent 暂停与恢复时的 KV 保留与状态所有权学习文档](<../大模型推理/KV缓存/Agent 暂停与恢复时的 KV 保留与状态所有权学习文档.md>) | 专题文章 | 227 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [KV Cache Salt 全链路键空间学习文档](<../大模型推理/KV缓存/KV Cache Salt 全链路键空间学习文档.md>) | 专题文章 | 417 | 保留独立问题与正文 | 先用一组相同 token 解释隔离，再追全链路键 |
| [KV Cache 与 MoE 权重的分层内存学习文档](<../大模型推理/KV缓存/KV Cache 与 MoE 权重的分层内存学习文档.md>) | 专题文章 | 167 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [KV Cache 容量优化技术地图学习文档](<../大模型推理/KV缓存/KV Cache 容量优化技术地图学习文档.md>) | 专题文章 | 430 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [前缀缓存命中后的恢复条件与状态所有权对比学习文档](<../大模型推理/KV缓存/前缀缓存命中后的恢复条件与状态所有权对比学习文档.md>) | 专题文章 | 261 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [跨模型 KV Cache 转换与 Prefill 复用学习文档](<../大模型推理/KV缓存/跨模型 KV Cache 转换与 Prefill 复用学习文档.md>) | 专题文章 | 170 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [Attention–FFN 分离的收益边界与部署选型学习文档](<../大模型推理/分布式服务/Attention-FFN 分离的收益边界与部署选型学习文档.md>) | 专题文章 | 198 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [DeepInfer 专用推理引擎与生产系统边界学习文档](<../大模型推理/分布式服务/DeepInfer 专用推理引擎与生产系统边界学习文档.md>) | 专题文章 | 372 | [I06](#i06)；已落实正文分工 | 执行流程与生产组件范围分层阅读 |
| [DeepInfer、vLLM 与 SGLang 的交付范围与机制比较学习文档](<../大模型推理/分布式服务/DeepInfer%20专用推理引擎与生产系统边界学习文档.md#delivery-boundary>) | 专题文章 | 189 | [I06](#i06)；独立正文已退役，保留旧址入口 | 比较条件靠近表格，避免重复跨框架总览 |
| [LLM 弹性扩缩容、模型驻留与冷启动学习文档](<../大模型推理/分布式服务/LLM 弹性扩缩容、模型驻留与冷启动学习文档.md>) | 专题文章 | 219 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [P/D 分离的 RDMA、IB 与 GPU 可见性学习文档](<../大模型推理/分布式服务/P-D 分离的 RDMA、IB 与 GPU 可见性学习文档.md>) | 专题文章 | 324 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [前缀亲和调度与缓存感知路由源码学习文档](<../大模型推理/分布式服务/前缀亲和调度与缓存感知路由源码学习文档.md>) | 专题文章 | 189 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [多模态 Encode 分离与 EPD 状态交接学习文档](<../大模型推理/分布式服务/多模态 Encode 分离与 EPD 状态交接学习文档.md>) | 专题文章 | 151 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [多轮 Agent 的 PD 路由与跨数据中心 Prefill 学习文档](<../大模型推理/分布式服务/多轮 Agent 的 PD 路由与跨数据中心 Prefill 学习文档.md>) | 专题文章 | 207 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [推理引擎与集群推理层分工学习文档](<../大模型推理/分布式服务/推理引擎与集群推理层分工学习文档.md>) | 专题文章 | 302 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [Context Parallel、PCP 与 DCP 总体学习文档](<../大模型推理/并行方法/Context Parallel、PCP 与 DCP 总体学习文档.md>) | 专题文章 | 408 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [PCP 长上下文 Prefill 并行学习文档](<../大模型推理/并行方法/PCP 长上下文 Prefill 并行学习文档.md>) | 专题文章 | 507 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [3D 并行训练的慢 Rank 定位与通信等待传播学习文档](<../大模型推理/性能工程/3D 并行训练的慢 Rank 定位与通信等待传播学习文档.md>) | 专题文章 | 172 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [KV Cache 与推理调度协同优化学习文档](<../大模型推理/性能工程/KV Cache 与推理调度协同优化学习文档.md>) | 专题文章 | 378 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [大模型推理全链路优化学习文档](<../大模型推理/性能工程/大模型推理全链路优化学习文档.md>) | 专题文章 | 214 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [LLM Prefill 与 Decode 阶段源码学习文档](<../大模型推理/推理基础/LLM Prefill 与 Decode 阶段源码学习文档.md>) | 专题文章 | 683 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [LLM 推理系统心智模型与 SGLang、vLLM 选型边界学习文档](<../大模型推理/推理基础/LLM 推理系统心智模型与 SGLang、vLLM 选型边界学习文档.md>) | 专题文章 | 390 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang 与 vLLM V1 架构差异源码学习文档](<../大模型推理/推理基础/SGLang 与 vLLM V1 架构差异源码学习文档.md>) | 专题文章 | 879 | 保留独立问题与正文 | 879 行多版本补充，按问题分主线与深入 |
| [DeepSeek V4.1 Flash 的 CED、CSA2 与缓存分工学习文档](<../大模型推理/模型架构/DeepSeek V4.1 Flash 的 CED、CSA2 与缓存分工学习文档.md>) | 专题文章 | 246 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [Engram 条件记忆与 KV 分层缓存学习文档](<../大模型推理/模型架构/Engram 条件记忆与 KV 分层缓存学习文档.md>) | 专题文章 | 188 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [KDA Cache 检查点、前缀复用与投机回滚学习文档](<../大模型推理/模型架构/KDA Cache 检查点、前缀复用与投机回滚学习文档.md>) | 专题文章 | 313 | [I05](#i05)；已落实正文分工 | 先建立检查点位置，再讨论各实现和性能口径 |
| [Kimi K3 混合注意力缓存与 Mooncake 状态传输学习文档](<../大模型推理/模型架构/Kimi K3 混合注意力缓存与 Mooncake 状态传输学习文档.md>) | 专题文章 | 250 | [I05](#i05)；已落实正文分工 | 统一贯穿例子，分开各系统的状态布局 |
| [Transformer 架构与推理流程学习文档](<../大模型推理/模型架构/Transformer 架构与推理流程学习文档.md>) | 专题文章 | 929 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [从 ViT 到 Omni 的多模态表示与生成路线学习文档](<../大模型推理/模型架构/从 ViT 到 Omni 的多模态表示与生成路线学习文档.md>) | 专题文章 | 187 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [混合注意力、索引压缩与残差流架构对比学习文档](<../大模型推理/模型架构/混合注意力、索引压缩与残差流架构对比学习文档.md>) | 专题文章 | 300 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [Chunked Prefill 与 Prefill-Decode 共推学习文档](<../大模型推理/调度方法/Chunked Prefill 与 Prefill-Decode 共推学习文档.md>) | 专题文章 | 335 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [LLM 推理调度的分层决策与 SLO 取舍学习文档](<../大模型推理/调度方法/LLM 推理调度的分层决策与 SLO 取舍学习文档.md>) | 专题文章 | 210 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [SGLang Overlap 与 DeepInfer ZOS 的依赖和状态提交对比学习文档](<../大模型推理/调度方法/SGLang Overlap 与 DeepInfer ZOS 的依赖和状态提交对比学习文档.md>) | 专题文章 | 275 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [DFlash 与 DSpark 并行投机解码学习文档](<../大模型推理/高级生成/DFlash 与 DSpark 并行投机解码学习文档.md>) | 专题文章 | 207 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [PD 与投机解码的状态交接、回滚及容量学习文档](<../大模型推理/高级生成/PD 与投机解码的状态交接、回滚及容量学习文档.md>) | 专题文章 | 142 | 保留独立问题与正文 | 保留；逐句润色未执行 |
| [动态投机验证预算：DeepInfer、SGLang 与 vLLM 的取舍学习文档](<../大模型推理/高级生成/动态投机验证预算：DeepInfer、SGLang 与 vLLM 的取舍学习文档.md>) | 专题文章 | 296 | 保留独立问题与正文 | 保留；逐句润色未执行 |

## 导航清单

19 份领域导航已精简共用说明，其余保留原结构；下表用于回查本轮覆盖范围。模型适配与 vLLM 性能导航仍保留跨目录选读入口，不复制迁出的正文。

| 导航 | 本轮处理 |
| --- | --- |
| [sglang/KV缓存/README.md](<../sglang/KV缓存/README.md>) | 共用说明已精简；相关入链已更新 |
| [sglang/README.md](<../sglang/README.md>) | 保留；纳入入口与课程分层复核 |
| [sglang/source-study/README.md](<../sglang/source-study/README.md>) | 保留；纳入入口与课程分层复核 |
| [sglang/source-study/architecture/README.md](<../sglang/source-study/architecture/README.md>) | 保留；纳入入口与课程分层复核 |
| [sglang/分离部署/README.md](<../sglang/分离部署/README.md>) | 共用说明已精简；相关入链已更新 |
| [sglang/并行方法/README.md](<../sglang/并行方法/README.md>) | 共用说明已精简；相关入链已更新 |
| [sglang/性能工程/README.md](<../sglang/性能工程/README.md>) | 共用说明已精简；相关入链已更新 |
| [sglang/服务运维/README.md](<../sglang/服务运维/README.md>) | 保留；纳入入口与课程分层复核 |
| [sglang/模型适配/README.md](<../sglang/模型适配/README.md>) | 共用说明已精简；相关入链已更新 |
| [sglang/版本研究/README.md](<../sglang/版本研究/README.md>) | 共用说明已精简；相关入链已更新 |
| [sglang/运行时/README.md](<../sglang/运行时/README.md>) | 共用说明已精简；相关入链已更新 |
| [sglang/高级生成/README.md](<../sglang/高级生成/README.md>) | 保留；纳入入口与课程分层复核 |
| [vllm/KV缓存/README.md](<../vllm/KV缓存/README.md>) | 共用说明已精简；相关入链已更新 |
| [vllm/README.md](<../vllm/README.md>) | 保留；纳入入口与课程分层复核 |
| [vllm/分离部署/README.md](<../vllm/分离部署/README.md>) | 共用说明已精简；相关入链已更新 |
| [vllm/并行方法/README.md](<../vllm/并行方法/README.md>) | 共用说明已精简；相关入链已更新 |
| [vllm/性能工程/README.md](<../vllm/性能工程/README.md>) | 共用说明已精简；相关入链已更新 |
| [vllm/运行时/README.md](<../vllm/运行时/README.md>) | 共用说明已精简；相关入链已更新 |
| [大模型推理/KV缓存/README.md](<../大模型推理/KV缓存/README.md>) | 共用说明已精简；相关入链已更新 |
| [大模型推理/README.md](<../大模型推理/README.md>) | 保留；纳入入口与课程分层复核 |
| [大模型推理/分布式服务/README.md](<../大模型推理/分布式服务/README.md>) | 共用说明已精简；相关入链已更新 |
| [大模型推理/并行方法/README.md](<../大模型推理/并行方法/README.md>) | 共用说明已精简；相关入链已更新 |
| [大模型推理/性能工程/README.md](<../大模型推理/性能工程/README.md>) | 共用说明已精简；相关入链已更新 |
| [大模型推理/推理基础/README.md](<../大模型推理/推理基础/README.md>) | 共用说明已精简；相关入链已更新 |
| [大模型推理/模型架构/README.md](<../大模型推理/模型架构/README.md>) | 共用说明已精简；相关入链已更新 |
| [大模型推理/调度方法/README.md](<../大模型推理/调度方法/README.md>) | 共用说明已精简；相关入链已更新 |
| [大模型推理/高级生成/README.md](<../大模型推理/高级生成/README.md>) | 保留；纳入入口与课程分层复核 |
