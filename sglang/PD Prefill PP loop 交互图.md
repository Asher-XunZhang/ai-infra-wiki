# PD Prefill：PP=3、5 个 micro-batch 的 loop 细分

## 1. 源码基线与图的边界

- 本地开源 worktree：`/Users/mac/Documents/Documents/工作/sglang-source-study`。
- 源码读取时间：2026-09-16；源码工作树干净；只读源码分析。
- 分支：`codex/main`，跟踪 `upstream/main`；固定提交 `72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a`。本次不更新该基线。
- 模式：PD Prefill、PP=3、`pp_async_batch_depth=0`、CUDA 路径、HiRadix L2 开启。
- 五份实际 micro-batch 为 M1–M5；三份环形槽位为 s0–s2。**micro-batch 数量不等于槽位数量，也不等于循环次数。**
- 为突出所有步骤，示例假设五份请求在开头进入 bootstrap；每轮最多选一份单请求 batch；M2/M4 有 L2 host hit；完整 Prefill、正常成功路径。关闭图中不需要的分支：中间 chunk、L3 storage、staging、DP MLP sync、speculative/prebuilt/skip-output、失败或 optimistic pending-bootstrap。
- 缓存按 write_back 且本例没有淘汰写回处理，所以 write ACK count 可以是 0；代码仍调用其计数同步路径。

图中 u 是假设时间单位。所有耗时均为演示值，不是 ms、profiler trace 或吞吐实测。图按源码调用顺序及关键消息依赖建立简化模型，省略 CUDA/NCCL 的实际流调度、通信资源竞争、CPU 调度抖动与逐层计算。尤其 `Work.wait()` 对 host/stream 的具体影响取决于后端；图将其作为依赖等待门控，不声称其宽度就是真实 CPU 阻塞时间。

## 2. 一个条块究竟是什么

CPU 行的每个外框 L# 表示 `event_loop_pp_disagg_prefill()` 中一次 **`for mb_id` 槽位迭代**，从恢复槽位状态、接收请求，到保存本轮状态。它不是外层 `while True` 完整遍历三个槽位的一圈，也不是某个 micro-batch 从进入到释放的整个生命周期。

PP0/PP1/PP2 各自有本地循环。相同 L# 在不同 rank 上不要求同时开始或结束。横轴统一，因而条块位置、长度和等待都可跨 rank 对照。

GPU 行仍以 M# 表示本级前向计算，因为 GPU 是异步资源，某份前向可以跨越 CPU loop 边界。外框内部的 CPU 细条按 A–I 区分，点击外框或选择 loop 可展开细项。

入口与循环：[scheduler.py:5662](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/managers/scheduler.py#L5662)、[scheduler_pp_mixin.py:220](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/managers/scheduler_pp_mixin.py#L220)。

## 3. 一次 loop 的源码顺序

下列阶段是便于读图的分组；函数在源码里没有 A–I 这些名称。

| 阶段 | 实际行为与细分 | 所属对象 |
|---|---|---|
| A 请求与状态 | 恢复本槽位状态；`ingest_requests()`；等待上次请求发送 work；轮询 bootstrap 状态并聚合 good 交集/bad 并集，保存 `bmbs[i]`；轮询 KV transfer 终态并求交集，保存 `tmbs[i]`；回收相关历史发送 work | 新请求集合、bootstrap 候选集合、历史 inflight 集合 |
| B chunk / L2 ACK | `process_prefill_chunk()`；进入 `get_new_batch_prefill()`；HiRadix 回收此前的 PP count 发送；write ACK count 同步与本地 event 处理；load ACK count 同步与本地 event 处理 | 当前槽位的旧 chunk，以及全局缓存 ACK 队列；本例没有中间 chunk |
| C 当前 batch 准备 | 空队列则返回 None；否则前缀匹配、排序、预算与 admission；host hit 时 `init_load_back()`；构造 batch；`ready_to_load_host_cache()` → `start_loading()`，设置 consumer index；`prepare_for_extend()`；保存当前 batch | **这一次即将提交的 M#**；不是固定的下一份 batch |
| D 激活与 work | 当前 batch 非空时，非首级接收上级 hidden states；随后回收历史 proxy send work | 当前 M# 的输入、上次提交的激活发送 |
| E 当前 forward | `_pp_launch_batch()`：forward stream 等待 schedule stream，运行本级层，记录 launch event；末级入 output 队列 | 当前 M#；CPU 提交与 GPU 完成分开 |
| F output | 本例 depth=0，所以在 launch 后执行；回收旧 output send work；末级发送当前 output，中间级转发上一轮保存的 output；接收 `mbs[j]` 的旧 output，安排 D2H 与结果预处理 | 旧结果槽位 j；发送与接收可能对应不同 M# |
| G PP 控制共识回流 | 发 bootstrap 共识、release 共识；按 `bmbs[j]` 接收 bootstrap 回流并移入 waiting queue；回收 bootstrap 共识发送；按 `tmbs[j]` 接收 release 名单并回收发送 | 候选请求集合和已终态请求集合；两者也不一定相同 |
| H 旧结果与 KV | 旧 batch 存在时 `d2h_event.synchronize()`、处理结果、更新 `last_mbs[j]`；正常 final Prefill 进入 inflight 并提交本级 KV 到 Decode；另按 release 名单检查旧 inflight 并清理 | `mbs[j]` 的旧 batch，以及更早的传输请求 |
| I 转发与保存 | 非末级转发请求、bootstrap 状态、transfer 终态；有当前 batch 时，schedule stream `wait_event(launch_event)` 后异步发 hidden states；保存 output、release/bootstrap 状态，复位 batch_is_full | 当前激活和队列快照；保存供下一轮转发的结果 |

关键定位：[主循环:234](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/managers/scheduler_pp_mixin.py#L234)、[output helper:1009](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/managers/scheduler_pp_mixin.py#L1009)、[launch:1078](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/managers/scheduler_pp_mixin.py#L1078)。

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
- release 回流后 `process_disagg_prefill_inflight_queue()` 再复查本地状态并释放请求引用/清理发送端；不表示所有缓存物理页都被回收。
- output tensor 环也独立于这两条控制路径。接收到 token/output 不代表 Decode 已收全 KV。

源码：[bootstrap:591](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/managers/scheduler_pp_mixin.py#L591)、[终态聚合:633](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/managers/scheduler_pp_mixin.py#L633)、[release:918](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/disaggregation/prefill.py#L918)。

### HiRadix L2 ACK 与逐层 H2D 等待

这里的 L2 是 host memory 层，不是 PD bootstrap，也不是 L3 storage。

1. `check_hicache_events()` 在空 waiting queue 的提前返回之前调用，所以空 batch loop 仍会推进 L2 ACK。
2. PP0 统计连续 ready ACK 的数量，并在本级相关 TP/CP 组做 MIN 归约。
3. `_pp_sync()` 把数量沿 PP0 → PP1 → PP2 传播。它**不是把所有 PP rank 的 ready 数一起做 MIN 的全 PP barrier**。
4. 每级按相同数量弹出自己的 ACK，执行本地 `finish_event.synchronize()` 并解锁相应缓存节点。因此 PP0 已完成的 load，在后级可能仍触发本地等待。
5. 当前选中 batch 的 host hit 由 `init_load_back()` 和 `start_loading()` 提交；GPU 使用 KV 时另有逐层 event 等待，可与层计算重叠。图只用“首个需要 KV 的层”的门控作简化，不能用图里的整段等待估算真实逐层 overlap 收益。

源码：[空队列前检查:3709](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/managers/scheduler.py#L3709)、[HiRadix _all_reduce/_pp_sync:239](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/mem_cache/hiradix_cache.py#L239)、[loading_check:1098](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/mem_cache/hiradix_cache.py#L1098)、[start_loading:914](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/managers/cache_controller.py#L914)、[layer wait:53](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/managers/cache_controller.py#L53)。

## 6. 默认展开 PP0 L6 的读法

此 loop 占示意时间 **13.75–29.80 u**，槽位 s2，当前 batch M3，旧结果 M1。

- A/B：接收状态、推进缓存 ACK；其中历史 L2 count 发送在本例产生等待。
- C：准备 M3。本例 M3 无 host hit；不会执行实际 `init_load_back()`。
- D：等待历史 M2 proxy send work。
- E：提交 M3，GPU 在 **17.21–23.21 u** 计算。
- F：CPU 路径在 **17.21–27.72 u** 等待 M1 output。前 6 u 与本级 M3 的 GPU 计算重叠；M3 在 23.21 u 结束后，依赖尚未就绪。
- G/H：接收并处理控制共识、处理 M1 结果、提交 M1 KV。此时“提交 M1 KV”不代表它已完成。
- I：本轮转发、保存后结束。之后才进入 L7 准备 M4；M4 在本例还有 L2 host hit，最终于 **34.61 u** 开始本级计算。

所以 PP0 从 M3 结束到 M4 开始的 **11.40 u 空泡是多段原因叠加**：旧 output 等待的尾段、其他收尾/下一轮准备、以及 M4 的 L2 layer gate。总览给空泡标注的是主要相交等待，不能把整段都归因于单一通信操作。

**槽位占满不必然产生空泡。** 环上旧结果和相关通信若及时完成，CPU 准备若被已有 GPU 工作覆盖，下一份可以及时接续；本图选择不均衡耗时来展示依赖来不及完成时出现的空泡。没有用该示意推导性能数字。

## 7. 验证与交互入口

[打开交互图](https://asher-xunzhang.github.io/ai-infra-wiki/sglang/pd-prefill-pp-loop/)。

依赖模型验证了前驱约束、每级五份前向（共 15 段）、本级 KV 提交晚于前向完成，以及五份请求的各级 release。浏览器检查了 rank/loop 选择、源码链接和不同屏幕宽度下的显示。这些属于静态分析与图的逻辑检查，未运行 SGLang GPU 推理。

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
| PP1 L5 的 L2 load ACK | 本地检查的程序顺序 | PP0 L5 公布的 load ACK count |
| PP0 L5 的 M2 GPU | forward 提交、上份 GPU、本例的 H2D layer gate | 首级无上游激活；中间级经 recv_proxy 链追溯 |
| PP0 L10 的 release | release 名单、本地 M1 传输终态及当前线程顺序 | 名单接收继续追溯 PP2 的终态共识回流 |

这是一张固定成功路径、固定 batch 分配下的源码依赖示意，不是完整运行时 trace。图中共导出 1669 个已解析模型节点；这包括用于闭合边界的后续节点，不表示实际运行必然执行该数量的操作。
