# PD Prefill：PP=3、5 个 micro-batch 的 loop 细分

## 0. 场景：PD 分离下，Prefill 侧的 PP 调度

**第一次阅读建议从[快速入门版](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/pd-prefill-pp-loop/quick.html)开始。** 它以 loop 为主图，选择 M1–M5 后在同一张图中高亮相关操作，并贴附该 batch 的进度带；详细版继续保留逐项操作与前置依赖。

入门版把源码中的连续步骤合成五组：A 检查状态；B+C 缓存与选批；D+E 输入 / 提交前向；F+G+H 结果与共识；I 转发保存。每个外框仍代表一次本地槽位迭代，段宽保留原模型时间。bootstrap 轮询与共识回流仍分处前后，避免为“合并概念”而修改执行顺序。

每级下沿的生命周期进度带与 loop 共用时间轴，沿用 micro-batch 的 M1–M5 编号，按各级的六个连续状态区间显示：准入与排队、准备缓存 / 输入、本级 GPU 前向、等结果 / 处理、KV 传输、等共识 / 释放。除 GPU 前向段外，这些区间可能包含排队或资源等待，不意味着连续占用 CPU / GPU。共同准入状态来自同一请求集合，不能当成五份独立资源消耗相加。清理终点取本地 release 操作结束，release 名单记录的时间则是进入清理的时刻。

可以按住图中任意位置左右拖动平移，用按钮或 Ctrl / ⌘ + 滚轮缩放；全程已显示时需先放大。点击 loop 查看该轮大步骤，点击进度带查看该生命周期区间，再点对应 L# 定位；batch 高亮始终保留。

实线高亮表示大步骤中包含所选 batch 的操作，虚线表示多请求共享操作，均不表示整段耗时独属于它。归属来自原模型的明确对象及实际 release 名单；未列出请求成员的通用 bootstrap / 终态轮询不额外猜测归属。

loop、进度带与高亮归属使用同一份详细版模型。仓库内 `scripts/build_pp_quick_data.py` 从详细页面提取模型并生成 `quick-data.js`，检查 48 次本地 loop 的连续覆盖、全部五份 batch 在三个流水级上的生命周期、激活先后及 release 所属；重建命令为 `python3 scripts/build_pp_quick_data.py`。这是图的逻辑一致性检查，没有新增 GPU 运行验证。

### 切换耗时场景进行对照

快速入门和依赖分析都提供“耗时场景”选择器，共用同一份依赖模型。原示例已经包含不同 PP 级、不同 batch 的 GPU 前向时间差异，以及 M2/M4 不同的 L2 回载时间；新选择器用于对照其他时间条件。

| 场景 | 相对原示例只修改哪些服务时间 |
| --- | --- |
| 原示例 | 保留原有不均匀时间 |
| GPU 计算均衡 | 15 段 GPU 前向均设为 5 u |
| PP1 计算偏慢 | PP1 的五段前向乘 2.5 |
| M3 计算偏长 | M3 在三个 PP 级的前向乘 2.5 |
| 整体计算较短 / 较长 | 全部前向分别乘 0.4 / 2.5 |
| L2 回载偏慢 | M2/M4 各级 H2D 时间乘 3 |
| 激活 / output 传输偏慢 | 激活与 output 消息在途时间乘 8 |
| PD KV 传输偏慢 | 各级 KV 在途时间乘 4 |
| CPU 调度偏慢 | CPU 操作的服务时间乘 2.5；依赖等待重新求解 |

每个场景重新求解依赖图，重新推导 L2 ACK 发布轮次、本地 event 等待、transfer 终态集合和 release 名单，并由同一模型派生快速入门的 loop 与生命周期。不是对最终图块做比例拉伸。末尾继续展开到所有 batch 完成本地清理，因此不同场景展示的 loop 数可以不同。

选择 M1–M5 仍可跟踪生命周期；两页跳转保留场景。“统一全程时间轴”使用相同 0–170 u 窗口对比，避免自动聚焦让不同时长看起来相同。参数表列出本场景每级每批的实际示意 GPU 时间；指标比较最后一段 GPU 完成、全部本地清理完成及展示轮数。指标均不是 Decode 生成完成时间。

例如慢 KV 场景中，GPU 全部完成仍为 75.05 u，而全部本地清理由 101.72 u 延至 162 u，每级展示 38 轮。**这是固定选批且没有新增容量压力反馈的示意模型结果**，不代表真实服务中的慢 KV 一定不影响 GPU。

所有场景固定 PP=3、depth=0、L1–L3 准入、L4–L8 每轮选择一份 batch，未模拟真实请求流、动态选批、后端竞争、逐层 CUDA 行为或失败恢复，不能计算吞吐、加速比或穷举真实工况。本文后续走读中的固定时间数字仍对应原示例。

场景生成器为 `scripts/pp_timing_model.py`；用 `python3 -B scripts/build_pp_scenarios.py` 重建两种视图的数据与场景目录。生成时校验原示例完全一致、所有前置先完成、五份 batch 的跨级前向与 KV / release 顺序，以及生命周期完整覆盖。GitHub Pages 发布流程会重新运行这些校验。

这张图展示 **SGLang 在 Prefill / Decode 分离部署时，Prefill 侧使用流水线并行（PP）的调度逻辑**。Prefill 处理输入 prompt、执行前向并产生 KV cache；Decode 接收所需 KV 后继续逐 token 生成。图中还包含 Prefill 为此推进的 bootstrap 状态、结果处理、KV 发送及请求释放。

**PP0、PP1、PP2 全部属于 Prefill 侧**，分别执行该流水级的模型层；同一份 micro-batch 的激活依次经过这三个流水级。Decode 是图外的 KV 接收方，没有作为第四条流水级画入，也未展开其调度循环、GPU 计算或设备拓扑。PP rank 不必等于一颗 GPU，级内 TP / CP 等并行维度未展开。

本例用五份 micro-batch 和三个环形槽位，说明“一次本地 loop 正在准备谁、提交谁、处理谁的旧结果”。M# 表示实际 batch，s# 表示可复用状态槽位，L# 表示一次本地槽位迭代；三者不能混用。CPU、GPU 与 I/O 行共用时间轴，呈现调度线程、异步前向和通信 / 拷贝之间的依赖。

建议先在[交互图页面](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/pd-prefill-pp-loop/)阅读场景介绍，再看默认 PP0 L6：这一轮提交 M3 前向，同时接收并处理旧 M1 的结果。页面下方的读图指南解释 A–I 阶段、PD bootstrap 与 L2 ACK 的区别、环形槽位关系、操作方法及模型假设；本说明继续提供逐项源码锚点。

图中 u 是人为设定的示意时间单位。它用于解释源码调用顺序与依赖，不是设备 trace；“槽位占满必然产生空泡”以及具体吞吐、延迟、加速比都不是本图的结论。

## 1. 源码基线与图的边界

**2026-09-17 基线升级：** 本图、快速入门和本说明已对齐本地 `D:/Codefiles/sglang` 的 `main` / `279339f113b79af84f27fd3ac92d0a13bd3f4cbd`。读取时源码工作区干净；仅做静态源码对照、教学模型和页面验证，未启动 SGLang 或进行 GPU 推理。旧图的 `72d5c5bb73` 不再作为这些页面的基线；本地当前提交也不代表远端最新版本。

**缓存实现选择：** 本例采用普通 FULL attention、未指定自定义缓存 backend 的默认 `UnifiedRadixCache`，启用 HiCache 的 `cache` 模式，关闭 external linker。不是旧 `HiRadixCache`，也不涵盖 `buffer_only`、SWA 或 Mamba 的特殊恢复路径。当前基线的默认工厂见 [registry.py](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/registry.py#L80)；ACK 合并同步见 [check_hicache_events](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L3290)。

- 源码目录：`D:/Codefiles/sglang`。
- 源码读取时间：2026-09-17；源码工作树干净；只读源码分析。
- 分支：`main`；固定提交 `279339f113b79af84f27fd3ac92d0a13bd3f4cbd`。后续本地 HEAD 变化仍需重新复核，不会自动视为适用。
- 模式：PD Prefill、PP=3、`pp_async_batch_depth=0`、CUDA 路径、UnifiedRadixCache + HiCache L2 开启。
- 五份实际 micro-batch 为 M1–M5；三份环形槽位为 s0–s2。**micro-batch 数量不等于槽位数量，也不等于循环次数。**
- 为突出所有步骤，示例假设五份请求在开头进入 bootstrap；每轮最多选一份单请求 batch；M2/M4 有 L2 host hit；完整 Prefill、正常成功路径。关闭图中不需要的分支：中间 chunk、L3 storage、buffer_only、staging、DP MLP sync、speculative/prebuilt/skip-output、sampling mask 输出、KV checksum、失败或 optimistic pending-bootstrap；本图不适用于 NPU/HCCL 的 PP=2 专门收发路径。
- 缓存按 write_back 且本例没有淘汰写回处理，所以 write ACK count 可以是 0；代码仍调用其计数同步路径。

图中 u 是假设时间单位。所有耗时均为演示值，不是 ms、profiler trace 或吞吐实测。图按源码调用顺序及关键消息依赖建立简化模型，省略 CUDA/NCCL 的实际流调度、通信资源竞争、CPU 调度抖动与逐层计算。尤其 `Work.wait()` 对 host/stream 的具体影响取决于后端；图将其作为依赖等待门控，不声称其宽度就是真实 CPU 阻塞时间。

### 从旧基线升级，哪些讲解变了？

以本地当前提交为准，旧图不是直接换一个 commit 标记就继续使用。以下差异已同步到图中说明、数据和源码锚点：

| 对照项 | 当前处理 | 源码 |
| --- | --- | --- |
| 缓存事件入口 | 从选批内部移至 PP loop 的显式 `_process_hicache_events()`；B 在 C 之前，空队列也推进 ACK。 | [scheduler_pp_mixin.py:251](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L251) |
| 准入与回载 | 先检查预算/延迟准入，再准备 host hit，最后提交准入；图中将原“选批”块拆出 `commit_admission`，拒绝路径不假定已发起 H2D。 | [schedule_policy.py:1185](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/schedule_policy.py#L1185) |
| 预算实现 | `PrefillAdder` 通过 allocator 创建 `PrefillBudget`；本图容量充足且固定每轮一份，不模拟新的预算拒绝、回载失败或重算路径。 | [prefill_budget.py:61](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/prefill_budget.py#L61) |
| 成功收尾 | 请求按 `(rid, attempt_id)` 标识缓存尝试；增加 `finish(SUCCESS)` 调用，默认成功实现不取消缓存异步工作，也没有新 barrier。 | [base_prefix_cache.py:374](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/base_prefix_cache.py#L374) |
| 新增可选分支 | NPU PP=2 批量收发、sampling mask 输出、可选 KV checksum、入口拒绝与失败重试均不纳入本 CUDA PP=3 成功场景。 | [scheduler_pp_mixin.py:1037](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1037) |
| 仍成立的主线 | depth=0 的先 launch 后 output、旧槽位、bootstrap/release 共识、合并 L2 ACK、H2D 与 D2H 流依赖均保持。 | [scheduler_pp_mixin.py:221](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L221) |

本次没有添加新的 GPU/通信等待边；成功收尾也没有虚构额外等待。为展示准入顺序，将原来的 0.48 u 选批示意时间拆为 0.36 u 预检查和 0.12 u 提交，中间保留回载准备。因此主要前向、传输和释放时间保持不变；这仅是教学占位值的分配，不是两个版本性能相同的证据。

## 2. 一个条块究竟是什么

CPU 行的每个外框 L# 表示 `event_loop_pp_disagg_prefill()` 中一次 **`for mb_id` 槽位迭代**，从恢复槽位状态、接收请求，到保存本轮状态。它不是外层 `while True` 完整遍历三个槽位的一圈，也不是某个 micro-batch 从进入到释放的整个生命周期。

PP0/PP1/PP2 各自有本地循环。相同 L# 在不同 rank 上不要求同时开始或结束。横轴统一，因而条块位置、长度和等待都可跨 rank 对照。

GPU 行仍以 M# 表示本级前向计算，因为 GPU 是异步资源，某份前向可以跨越 CPU loop 边界。外框内部的 CPU 细条按 A–I 区分，点击外框或选择 loop 可展开细项。

入口与循环：[scheduler.py:5748](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler.py#L5748)、[scheduler_pp_mixin.py:221](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L221)。

## 3. 一次 loop 的源码顺序

下列阶段是便于读图的分组；函数在源码里没有 A–I 这些名称。

| 阶段 | 实际行为与细分 | 所属对象 |
|---|---|---|
| A 请求与状态 | 恢复本槽位状态；`ingest_requests()`；等待上次请求发送 work；轮询 bootstrap 状态并聚合 good 交集/bad 并集，保存 `bmbs[i]`；轮询 KV transfer 终态并求交集，保存 `tmbs[i]`；回收相关历史发送 work | 新请求集合、bootstrap 候选集合、历史 inflight 集合 |
| B chunk / L2 ACK | `process_prefill_chunk()`；显式调用 `_process_hicache_events()`；UnifiedRadixCache 回收此前的 PP count 发送；合并同步 write/load ACK count；随后处理本地 write/load event | 当前槽位的旧 chunk，以及全局缓存 ACK 队列；本例没有中间 chunk |
| C 当前 batch 准备 | 空队列则返回 None；否则前缀匹配、排序；`PrefillBudget` 与 `_select_prefill_admission()` 检查预算，再通过延迟准入；host hit 时 `init_load_back()`；成功后 `_commit_prefill_admission()` 锁定前缀并加入 `can_run_list`，再构造 batch；`ready_to_load_host_cache()` → `start_loading()`，设置 consumer index；`prepare_for_extend()`；保存当前 batch | **这一次即将提交的 M#**；不是固定的下一份 batch |
| D 激活与 work | 当前 batch 非空时，非首级接收上级 hidden states；随后回收历史 proxy send work | 当前 M# 的输入、上次提交的激活发送 |
| E 当前 forward | `_pp_launch_batch()`：forward stream 等待 schedule stream，运行本级层，记录 launch event；末级入 output 队列 | 当前 M#；CPU 提交与 GPU 完成分开 |
| F output | 本例 depth=0，所以在 launch 后执行；回收旧 output send work；末级发送当前 output，中间级转发上一轮保存的 output；接收 `mbs[j]` 的旧 output，安排 D2H 与结果预处理 | 旧结果槽位 j；发送与接收可能对应不同 M# |
| G PP 控制共识回流 | 发 bootstrap 共识、release 共识；按 `bmbs[j]` 接收 bootstrap 回流并移入 waiting queue；回收 bootstrap 共识发送；按 `tmbs[j]` 接收 release 名单并回收发送 | 候选请求集合和已终态请求集合；两者也不一定相同 |
| H 旧结果与 KV | 旧 batch 存在时 `d2h_event.synchronize()`、处理结果、更新 `last_mbs[j]`；正常 final Prefill 进入 inflight 并提交本级 KV 到 Decode；另按 release 名单复查终态，成功时 `release_kv_cache()` → `tree_cache.finish(handle, SUCCESS)` → `sender.clear()`，归还 metadata 并移出 inflight | `mbs[j]` 的旧 batch，以及更早的传输请求 |
| I 转发与保存 | 非末级转发请求、bootstrap 状态、transfer 终态；有当前 batch 时，schedule stream `wait_event(launch_event)` 后异步发 hidden states；保存 output、release/bootstrap 状态，复位 batch_is_full | 当前激活和队列快照；保存供下一轮转发的结果 |

关键定位：[主循环:235](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L235)、[output helper:1018](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1018)、[launch:1221](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1221)。

若 `pp_async_batch_depth>0`，F 的 helper 被移动到 E 前面，但 H 的最终结果处理仍在 E 后面；环长度也随之改变。本图不能直接套用到 depth>0。

## 4. 为什么不是固定的 i+1 / i / i-1

源码计算：

```python
pp_loop_size = pp_size + pp_async_batch_depth
next_mb_id = (mb_id + 1) % pp_loop_size
```

本例环长 3。在连续每轮提交一份新 batch 时，当前槽位 i 的下一槽 j 保存的是**两次本地迭代之前**提交的 batch。因此计算 M3 的迭代处理 M1，计算 M4 的迭代处理 M2。槽位下标的“+1”是环形索引，不能解释成“未来一份 batch”。空槽、预算、chunk、额外异步深度都会改变简单的 batch 编号关系。

| 本地 loop | 当前槽位 | 提交前向 | 旧结果处理 |
|---|---|---|---|
| L1–L3 | s0 → s1 → s2 | 无，等待 bootstrap 准入 | 无 |
| L4 | s0 | M1 | 无 |
| L5 | s1 | M2 | 无 |
| L6 | s2 | M3 | M1 |
| L7 | s0 | M4 | M2 |
| L8 | s1 | M5 | M3 |
| L9 | s2 | 无 | M4 |
| L10 | s0 | 无 | M5 |
| 后续 | 环形继续 | 无 | 继续推进终态与 release |

三轮 bootstrap 暖场和上述 batch 分配是本图的明确初始条件，不是所有启动场景都必须恰好如此。服务若已有准入请求，可省去这段暖场。五份 batch 全部计算结束后也不能立即停止循环：KV 传输和 release 共识还要继续推进。

## 5. 两类“共识”要分开看

### PD bootstrap / transfer / release

- `_pp_pd_get_bootstrapped_ids()`：每级检查本地发送端 bootstrap；good 取交集，bad 取并集，再向后传播。末级得到聚合状态，回流后各级才把相应请求从 bootstrap 队列移入 waiting queue。
- 因为“接收 bootstrap 共识并准入”在本轮选 batch 之后，这些新准入请求最早可在后续迭代参加选批。但数量和归属由队列及预算决定，不能把这个集合统一写成 M(i+1)。
- `_pp_pd_get_prefill_transferred_ids()` 聚合本地 Success/Failed 终态集合。它与 bootstrap 是独立路径。
- release 回流后 `process_disagg_prefill_inflight_queue()` 再复查本地状态。成功分支释放请求 KV 引用、调用 `tree_cache.finish(req.cache_request_handle, SUCCESS)`、清理发送端和 metadata。当前默认缓存继承的 `finish(SUCCESS)` 不取消异步缓存工作、不新增等待；请求结束不表示所有缓存物理页都被回收。
- output tensor 环也独立于这两条控制路径。接收到 token/output 不代表 Decode 已收全 KV。

源码：[bootstrap:593](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L593)、[终态聚合:635](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L635)、[release:972](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L972)。

### UnifiedRadixCache + HiCache L2 ACK 与逐层 H2D 等待

这里的 L2 是 host memory 层，不是 PD bootstrap，也不是 L3 storage。

1. PP loop 在 `process_prefill_chunk()` 后、`get_new_batch_prefill()` 前调用 `_process_hicache_events()`，由它进入 `check_hicache_events()`；不再从选批函数内部调用。即使 waiting queue 为空，启用 HiCache 的 loop 仍会推进 L2 ACK。
2. PP0 分别统计 write/load 队列中连续 ready ACK 的数量，由 `_sync_hicache_ready_counts()` 合在一个张量中同步，并在本级相关 TP/CP 组做 MIN 归约。同步张量还带有 write_back 回收一致性摘要；本例无 L3 存储队列。
3. `_pp_sync()` 把数量沿 PP0 → PP1 → PP2 传播。它**不是把所有 PP rank 的 ready 数一起做 MIN 的全 PP barrier**。
4. 每级按相同数量弹出自己的 ACK，执行本地 `finish_event.synchronize()`；load 完成后释放这次回载的 device/host 锁，并结束加载保护，host 副本仍可保留。因此 PP0 已完成的 load，在后级可能仍触发本地等待。
5. 当前选中 batch 的 host hit 由 `init_load_back()` 和 `start_loading()` 提交；GPU 使用 KV 时另有逐层 event 等待，可与层计算重叠。图把整份 H2D 粗化为一个完成点，并让 GPU 前向等这个完成点；这是模型额外采用的整批门控，源码没有这个整批 barrier，不能用图里的整段等待估算真实逐层 overlap 收益。

源码：[选批前缓存入口:3597](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler.py#L3597)、[UnifiedRadix _all_reduce/_pp_sync:333](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L333)、[loading_check:3187](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L3187)、[start_loading:941](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/cache_controller.py#L941)、[layer wait:54](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/cache_controller.py#L54)。

### CUDA 流之间还有两条必要依赖

CPU 已经提交异步操作，不代表对应设备流已经执行到那里：

- **H2D 启动依赖：** `start_loading()` 在当前 `schedule_stream` 上记录 `start_event`，H2D 流等待这个事件。上一轮非末级为激活发送、末级为 output 发送排入的 `wait_event(launch_event/q_event)` 仍在调度流上，因此本例 M2/M4 的回载不能越过上一本级前向完成点。[start_loading](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/cache_controller.py#L950)、[H2D start_event](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/l2_transfer.py#L74)。
- **末级旧结果 D2H 依赖：** depth=0 的 CUDA 路径先提交当前 output 发送，其前面有 `schedule_stream.wait_event(q_event)`；随后旧结果的 `copy_stream.wait_stream(schedule_stream)` 承接该约束。因此末级旧结果的 D2H 也要等本轮当前前向完成。比如 PP2 L6 拷贝 M1，必须晚于本级 M3 前向结束；不能因为对象是旧 M1 就忽略 M3 的事件。[output 发送](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1000)、[D2H 等待](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1084)。

图中显式保留这两条必要 event 依赖；`wait_event` 本身不是 CPU 阻塞。图仍省略通信后端的完整流调度，不用于推断真实 host 等待时间。

## 6. 默认展开 PP0 L6 的读法

此 loop 占示意时间 **13.75–29.80 u**，槽位 s2，当前 batch M3，旧结果 M1。

- A/B：接收状态、推进缓存 ACK；其中历史 L2 count 发送在本例产生等待。
- C：准备 M3。本例 M3 无 host hit；不会执行实际 `init_load_back()`。
- D：等待历史 M2 proxy send work。
- E：提交 M3，GPU 在 **18.00–24.00 u** 计算。
- F：CPU 路径在 **18.00–27.72 u** 等待 M1 output。前 6 u 与本级 M3 的 GPU 计算重叠；M3 在 24.00 u 结束后，依赖尚未就绪。
- G/H：接收并处理控制共识、处理 M1 结果、提交 M1 KV。此时“提交 M1 KV”不代表它已完成。
- I：本轮转发、保存后结束。之后才进入 L7 准备 M4；M4 在本例还有 L2 host hit，最终于 **34.61 u** 开始本级计算。

所以 PP0 从 M3 结束到 M4 开始的 **10.61 u 空泡是多段原因叠加**：旧 output 等待的尾段、其他收尾/下一轮准备、以及 M4 的 模型的整份 H2D 门控。总览给空泡标注的是主要相交等待，不能把整段都归因于单一通信操作。

**槽位占满不必然产生空泡。** 环上旧结果和相关通信若及时完成，CPU 准备若被已有 GPU 工作覆盖，下一份可以及时接续；本图选择不均衡耗时来展示依赖来不及完成时出现的空泡。没有用该示意推导性能数字。

## 7. 验证与交互入口

[打开交互图](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/pd-prefill-pp-loop/)。

依赖模型验证了前驱约束、每级五份前向（共 15 段）、本级 KV 提交晚于前向完成，以及五份请求的各级 release。2026-09-17 新增的 `scripts/test_pp_timing_model.py` 对全部十个场景检查 H2D / D2H 的必要源码依赖及合并 ACK 路径，并通过故意删除依赖的反例确认检查能拦住原来的遗漏。构建同时重建详细图与入门图数据。以上属于源码静态复核和教学模型检查，未运行 SGLang GPU 推理。

## 8. 怎样查看每一步的前置关系

总览和原先的 loop 放大图已合并为一张可缩放、平移的时间轴。三个 PP rank 共用同一时间窗口，CPU 的所有阶段仍在各自的一条横向泳道。选择「步骤与前置条件」或点击一个步骤，会显示它的入边，并在下方列出前置节点；点击前置节点可以继续追溯。

- 用「＋ / −」或 Ctrl / ⌘ + 滚轮缩放，横向拖动或移动「时间窗口」滑块平移。
- 「聚焦依赖」自动容纳当前 loop 和直接前置的完成点；「显示全程」恢复整个流水线。
- 放大时直接展开 CPU 条块里的分段，并显示同一时间范围内各级的 GPU、CPU、H2D/KV 在途状态。
- 窗口外的前置节点使用边缘箭头提示；节点的真实示意时间仍在前置列表中，边缘箭头不代表它在边界时刻发生。

- 同一泳道上时间连续、直接相接的前置分段与选中步骤，用彩色共边表示依赖，不再画箭头。相接按模型时间判断，不因缩小后看起来靠近就合并；中间有等待分段时仍保留箭头。
- 其他依赖使用从前置完成时刻指向选中步骤开始时刻的箭头：实线表示同一个 PP rank，虚线表示跨 PP rank，不再附加圆点或菱形。PP rank 代表流水级；若一级内配置 TP/CP，它不等于单颗芯片。本图没有展开级内芯片。
- 异步 message、copy 节点在选中它或其直接后继时，补显为 I/O 行内的通信 / 拷贝条块；H2D、KV 保持原有条块。这些条块表示异步活动，不表示 CPU 在那里进行计算。连线起止于真实条块边界，通信完成时刻仍与 CPU 提交时刻区分。
- 条块、选中边框、共边与连线共用泳道坐标和时间映射。窗口内的端点不再向内偏移；窗口外的前置完成点截在时间窗口边界，并用方向提示区分。为闭合依赖而补显的后续 loop 节点同样使用真实模型时间。
- 「CPU 程序顺序」只说明调度线程先后执行，不意味着后一步必须消费前一步的数据。
- 「通信就绪」「数据 / event」「源码数据条件」区分激活、结果、名单与事件依赖。
- 「模型串行假设」「通信配对（模型门控）」是示意模型采用的约束，不能解释成所有传输后端都具有相同的串行或 host 阻塞行为。
- bootstrap、terminal、release 的本地 poll、级内 TP/CP 聚合及 Decode 侧条件单独列为未展开的状态输入；轮询本身不等于等待所有请求 ready。

例如默认 PP0 L6 的旧 M1 output 接收：本级在提交 M3 之后进入该操作，同时依赖 PP2 L4 的 M1 output 到达。继续点击 PP2 的 output，可看到它依赖本级发送提交以及 M1 GPU 完成。两个 rank 的相同 L# 无须同时发生。

其他可直接检查的例子：

| 选中步骤 | 本级关系 | 跨级关系 |
| --- | --- | --- |
| PP1 L5 接收 M2 激活 | 当前 batch 的准备路径 | PP0 L5 的 proxy 消息；继续追溯其发送和 GPU 完成 |
| PP1 L5 的 L2 合并 ACK 计数 | 本地检查的程序顺序 | PP0 L5 公布的 write/load ACK counts |
| PP0 L5 的 M2 GPU | forward 提交、上份 GPU、本例的 整份 H2D 门控（模型粗化） | 首级无上游激活；中间级经 recv_proxy 链追溯 |
| PP0 L10 的 release | release 名单、本地 M1 传输终态及当前线程顺序 | 名单接收继续追溯 PP2 的终态共识回流 |

这是一张固定成功路径、固定 batch 分配下的源码依赖示意，不是完整运行时 trace。图中共导出 1619 个已解析模型节点；这包括用于闭合边界的后续节点，不表示实际运行必然执行该数量的操作。
