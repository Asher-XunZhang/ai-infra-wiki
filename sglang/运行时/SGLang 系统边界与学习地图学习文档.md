# SGLang 系统边界与学习地图学习文档

先用一张代码框架图找到主要模块，再沿请求辨认职责、进程、设备资源与模型副本。本篇连接 Pages 的十二模块；各深入资料保留自己的固定源码基线。

[交互课程：SGLang 代码架构与模块协作](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/system-overview/) · [本领域路线](README.md)

## 0. 源码基线与范围

本文是源码分析型学习资料。主线采用普通 Python HTTP、单 tokenizer、单卡合并部署、无 overlap 的文本生成；拓扑图另设 TP=2、DP=2 或 P/D 两侧各单卡的独立配置。

| 项目 | 内容 |
| --- | --- |
| 公开仓库 | [sgl-project/sglang](https://github.com/sgl-project/sglang) |
| 分支与 commit | 公开上游 `main` 快照：[279339f113b79af84f27fd3ac92d0a13bd3f4cbd](https://github.com/sgl-project/sglang/tree/279339f113b79af84f27fd3ac92d0a13bd3f4cbd) |
| 读取时间 | 初读 2026-09-23；代码框架复核 2026-10-19 |
| 工作区状态 | 复核时源码检出目录干净；只读固定 Git 对象，当前检出版本与本文快照分开，不修改源码。 |
| 操作边界 | 静态源码分析、教学模型和页面验证；未运行模型、GPU、网络或性能实验。 |
| 教学设定 | 两个输入位置 p0/p1、一个首输出 y1；拓扑图用三层代表权重。不是实际 tokenization、模型预测或部署配置。 |
| 不展开 | 多 tokenizer、Ray、Rust runtime、全部并行组合、特殊模型、错误恢复及完整协议。 |

先修：[自回归生成、Prefill 与 Decode](https://asher-xunzhang.github.io/ai-infra-wiki/foundations/generation.html)。


## 1. 先看代码框架：模块怎样共同完成生成

在线服务要同时接住新请求、安排有限的计算资源、保存历史状态并持续返回文字。仅看模型 `forward` 无法解释这些工作；先认识模块之间的协作，再进入具体函数。

下面聚焦 `python/sglang/srt/`，即 SGLang Runtime（推理运行时）。箭头表达依赖和载荷；本图采用单 tokenizer、单卡、同机的普通 Python 文本服务。GPU 是设备资源，Worker 与 Runner 是调度进程内对象。

```mermaid
flowchart TB
    C[客户端]
    subgraph Front[入口进程]
        A[API 接入：entrypoints]
        T[分词与请求管理：TokenizerManager]
        A -->|文本与生成配置| T
        T -->|文字结果| A
    end
    subgraph Back[Scheduler 进程]
        S[调度与结果处理：Scheduler / ScheduleBatch]
        E[执行封装与采样：TpModelWorker / ModelRunner]
        K[KV 管理：前缀索引 / 地址池 / 分配器]
        M[模型与算子：models / layers / backend]
        G[GPU 设备资源：权重 / 激活 / KV]
        S -->|本轮 batch| E
        E -->|本轮结果| S
        S <-->|匹配与空间管理| K
        E <-->|模型调用与分数| M
        K -->|管理 KV 位置| G
        M <-->|设备计算与访存| G
    end
    subgraph Output[反分词进程]
        D[DetokenizerManager]
    end
    C -->|HTTP 请求| A
    A -->|HTTP 响应| C
    T -->|IPC：token ID / 配置 / rid| S
    S -->|IPC：输出 token ID / rid| D
    D -->|IPC：文字 / 结束信息 / rid| T
```

**读图：** `rid` 是请求标识，用于关联不同进程里的请求状态；IPC 是进程间通信。跨进程传递的是消息，进程内箭头表示调用、返回或资源管理。Scheduler 消费本轮结果后继续选批，未结束的请求可在后续轮次继续执行。图中没有模拟耗时，也不要求实际网络回程与下一轮计算完全串行。

| 模块 | 固定版本中的代码入口 | 负责什么 |
| --- | --- | --- |
| API 接入 | [`python/sglang/srt/entrypoints/http_server.py`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/entrypoints/http_server.py#L911) | 把客户端协议接到引擎，并把生成结果返回连接。 |
| 分词与请求管理 | [`python/sglang/srt/managers/tokenizer_manager.py`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/tokenizer_manager.py#L1592) | 模型需要 token ID；返回结果还要找到原来的请求连接。 |
| 调度与结果处理 | [`python/sglang/srt/managers/scheduler.py`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler.py#L1907) | 请求不断到来，计算与 KV 容量有限，需要决定本轮接纳谁、执行多少。 |
| 执行封装与采样 | [`python/sglang/srt/managers/tp_worker.py`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/tp_worker.py#L593) | 调度描述本轮做什么；执行层要把它变成模型可用的张量、位置和执行方式。 |
| KV 管理 | [`python/sglang/srt/managers/scheduler.py`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler.py#L615) | 复用已有历史，并让新 token 有可写位置；请求结束后还要按共享关系回收。 |
| 模型与算子 | [`python/sglang/srt/models/deepseek_v2.py`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/models/deepseek_v2.py#L3165) | 执行层需要调用具体的模型结构与适合当前设备的运算实现。 |
| 输出反分词 | [`python/sglang/srt/managers/detokenizer_manager.py`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/detokenizer_manager.py#L443) | 模型选择的是 token ID，客户端需要文字及结束信息。 |

DeepSeek-V3 的模型例子对应 `models/deepseek_v2.py::DeepseekV3ForCausalLM`。`models/`、`layers/`、`model_executor/`、`mem_cache/` 位于 `srt/` 内；`python/sglang/kernels/` 是其同级算子目录，不能误写成 `srt/kernels/`。底层计算实现还可能来自外部后端。

### 主图之外的目录怎样衔接

- `srt/distributed/`：多 rank 的进程组与通信，衔接并行和通信课程。
- `sgl-model-gateway/` 与 `srt/disaggregation/`：分别处理实例外网关和 P/D 分离的状态交接，不是普通单实例请求必经的两跳。
- `srt/speculative/`、`srt/constrained/`、`srt/observability/`：分别进入投机、约束生成和观测专题。
- `python/sglang/lang/` 是语言前端；`python/sglang/multimodal_gen/` 是另一类生成路径。本图没有把它们混入普通文本请求链。

这些目录归属已按本文固定提交核对。第二章主体使用 `279339f113`；[PD 数据流专题](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/pd-dataflow/) 单独固定在 `882577451e`，不把它的细节当成本图版本的实现。

**表达参考：** Ziyi Xu，[Mini-SGLang: Efficient Inference Engine in a Nutshell](https://www.lmsys.org/blog/2025-12-17-minisgl/)，LMSYS，2025-12-17；读取于 2026-10-19。参考其架构图中请求闭环与执行侧缓存相邻的组织方式，上图为按完整 SGLang 固定源码重新组织的代码框架，不直接采用 Mini-SGLang 的类名或进程划分。

## 2. 服务路径：先数进程，再追踪载荷

```mermaid
flowchart TB
    Client[客户端] -->|生成请求| Front[入口进程：HTTP / TokenizerManager]
    Front -->|IPC：token IDs 与配置| Scheduler[Scheduler 进程]
    Scheduler -->|IPC：输出 token IDs| Detok[Detokenizer 进程]
    Detok -->|文本结果与 rid| Front
    Front -->|响应或流式块| Client
```

普通启动路径把 TokenizerManager 放在主进程，另起 Scheduler 与 Detokenizer。HTTP handler 使用前端 manager 提供的结果生成响应。这里画的是所选路径的主要进程，不是所有启动配置的完整进程树。

服务实例可以含多个进程，也可以含多张 GPU。反过来，一个职责名也不一定对应独立进程：单卡示例中的 Worker 和 ModelRunner 由 Scheduler 进程内调用，不是另外两个必须经过的网络服务。

请求经历的载荷也在变化：原始输入、内部请求、模型执行输入、输出 ID、文本结果和网络块不能互换。一个流式块可能携带多个 token 的结果；不能从网络块数倒推出模型 forward 次数。

交互总览将模型计算结果与客户端响应画在不同边界。沿“结果返回”视角可以观察中间还经过哪些模块；后续计算不必等待前一段文字完全返回后才继续。

## 3. 实例内部：职责协作与设备资源

```mermaid
flowchart TB
    S[Scheduler：决定本轮工作] -->|匹配、准入与位置准备| C[缓存索引 / 分配器]
    C -->|状态与地址| S
    S -->|本轮 batch| W[Worker / Runner：执行与采样]
    W -->|模型调用与写位置| G[GPU：模型计算与各层 KV]
    G -->|模型结果| W
    W -->|本轮结果| S
```

这张图画职责与资源，不能再把四个框数成四个进程。GPU 内存不是另一个服务器；主机对象管理设备 buffer，并组织实际设备计算。

Scheduler 先选出本轮工作，准备输入与地址。缓存索引回答能复用什么，分配器提供可写位置，模型层实际写入 K/V。能够写一个位置与决定何时释放它，是不同职责。

本例 Prefill 处理 p0、p1，写下这两个位置在各层的 KV；模型给出分数，再采样 y1。y1 尚未再次作为输入，所以它还没有自己的 KV。之后 Scheduler 消费结果，决定继续、结束或进入其他分支。

Worker、Runner、模型层、Attention backend 和 kernel 之间还有更细的调用关系，见[执行分层与硬件机制](SGLang 执行分层与硬件机制学习文档.md)。本图不把它们都铺在主画面上。

## 4. 两张卡，不一定是两个完整副本

页面分别展示三种独立拓扑：

- **TP=2：** 同一副本的可切分矩阵分给两个 rank 协作；各 rank 仍参与模型各层的计算。图中半块只代表一个可切分矩阵，不能推导所有参数、Norm 或激活都刚好减半。
- **DP=2、每副本 TP=1：** controller 为每个请求选择一个执行副本。本例用 round-robin 将 R1、R2 分给不同副本；两者共享服务入口，但分别调度并持有本地请求状态。显式分派和失效副本的规则不在图中。
- **P/D 各单卡：** 两个服务角色接力处理同一请求，两侧都覆盖完整模型。P 侧做 Prefill，D 侧续写，需要交接 KV、首输出与配对等状态；不是把模型前半层与后半层分给 P、D。

这三个维度不能只用“卡数”概括。并行还包括 PP、EP、CP，它们切分的对象不同；详细数学与通信关系见[并行课程](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/parallelism/)。多 rank 情况下也要重新核对 Scheduler 参与者和进程外框，不能照搬单卡进程图。

## 5. 用边界选择学习模块

| 当前问题 | 对应模块 | 要追踪的证据 |
| --- | --- | --- |
| 请求进入后为什么没回答 | [02 请求运行时](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/request-runtime/) → [10 服务治理](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/serving-operations/) → [11 性能分析](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/performance-engineering/) | 最后完成的交接、就绪/准入状态、指标计时边界。 |
| 谁决定本轮算哪些位置 | [05 调度](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/scheduling/) → [03 执行](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/model-execution/) | 候选列表、预算、执行输入与返回结果。 |
| 历史状态存在哪里 | [04 KV 与内存](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/kv-memory/) → [09 模型与生成](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/advanced-generation/) | 地址映射、共享保护、检查点与提交语义。 |
| 跨卡或跨实例传什么 | [06 并行](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/parallelism/) → [07 通信](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/communication/) → [08 分离部署](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/inference-overview/deployment.html) | 切分对象、通信组、数据就绪与资源退役。 |
| 怎样验证机制判断 | [12 综合实践](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/pd-prefill-pp-loop/quick.html) | 预测、单变量变化、固定源码与真实运行验证的边界。 |

这些链接组织阅读关系，不合并所有页面的源码版本，也不保证所有特性可以任意同时启用。

## 6. 固定源码入口

以下路径相对于公开 SGLang 仓库，均固定到本文 commit。

| 行为 | 入口 |
| --- | --- |
| 普通进程启动关系 | [`Engine._launch_subprocesses`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/entrypoints/engine.py#L1051) |
| HTTP 请求与流式响应 | [`generate_request`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/entrypoints/http_server.py#L911) |
| 前端派发内部请求 | [`TokenizerManager._send_one_request`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/tokenizer_manager.py#L1592) |
| 选批、执行、结果消费 | [`Scheduler.event_loop_normal`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler.py#L1907) |
| 输入与地址准备 | [`ScheduleBatch.prepare_for_extend`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/schedule_batch.py#L2678) |
| 执行与采样调用 | [`TpModelWorker.forward_batch_generation`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/tp_worker.py#L593) |
| 每层 KV 写入 | [`MHATokenToKVPool.set_kv_buffer`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/memory_pool.py#L2526) |
| Prefill 结果处理 | [`process_batch_result_prefill`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L257) |
| 输出 ID 转成文本结果 | [`DetokenizerManager.handle_batch_token_id_out`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/detokenizer_manager.py#L443) |
| 前端处理返回结果 | [`TokenizerManager._handle_batch_output`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/tokenizer_manager.py#L2255) |
| TP 矩阵分片计算 | [`ColumnParallelLinear.forward`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/layers/linear.py#L492)、[`RowParallelLinear.forward`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/layers/linear.py#L1612) |
| DP 启动与轮询分派 | [`launch_dp_schedulers`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/data_parallel_controller.py#L371)、[`round_robin_scheduler`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/data_parallel_controller.py#L767) |
| D 侧提交交接状态 | [`_commit_transfer_to_req`](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/decode.py#L2158) |

## 7. 自测与下一步

面对一个框，先回答它是职责、进程、资源还是副本；面对一根箭头，先回答它是调用、消息、张量还是状态交接。若请求没继续推进，再找最后一个已经满足的条件，而不是只看一个利用率数字。

继续阅读：[普通请求运行时与资源生命周期](SGLang 普通请求运行时与资源生命周期学习文档.md)、[系统架构课程](../source-study/architecture/README.md)。后者使用独立基线，阅读时分别核对版本。
