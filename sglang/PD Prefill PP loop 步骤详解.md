# PD Prefill PP loop：逐步理解行为与目的

本文是[依赖分析交互图说明](PD%20Prefill%20PP%20loop%20交互图.md)的逐项阅读版。每个示例说明“在做什么、为什么需要、完成后推进什么”。交互页面会按所选场景和步骤实时替换对象与等待条件；本文的数字和对象只对应原示例。

## 1. 阅读基线

| 项目 | 内容 |
| --- | --- |
| 源码仓库 | [sgl-project/sglang](https://github.com/sgl-project/sglang) |
| 分支 | `main` |
| 固定提交 | `279339f113b79af84f27fd3ac92d0a13bd3f4cbd` |
| 读取时间 | 2026-09-17 |
| 工作区状态 | 读取时干净，无本地改动或未跟踪文件 |
| 操作边界 | 只读源码；验证教学模型与页面，未运行 SGLang 或 GPU 实验 |

范围固定为 PP=3、depth=0、CUDA、UnifiedRadixCache + HiCache cache 模式，完整 Prefill 成功路径。M2/M4 有 host hit；无中间 chunk、L3、sampling mask 或推测解码。时间 u 是模型示意值；选批安排固定，资源串行和整批 H2D 屏障属于模型简化。

PP0/PP1/PP2 都在 Prefill 侧。M# 是 batch；L# 是本级槽位迭代。同一轮的新前向、旧结果和队列级名单可能属于不同请求。

| 对象 | 如何理解 |
| --- | --- |
| proxy / hidden states | 上一级模型层的中间特征，送往下一 PP 级继续计算 |
| output | 末级产生的 token 等结果，沿 PP 环回流，供各级更新请求 |
| KV | 各级本地层的注意力缓存，发送给图外 Decode |
| ACK / event | ACK 是完成队列条目；event 是异步工作完成的标记，仍需本级确认 |
| Work | 异步通信的句柄，回收它不等于执行新的发送 |

## 2. 各类操作

以下按调度、计算、结果、共识和通信整理；实际先后以交互图时间轴及前置依赖为准。

### 2.1. 接收请求，建立本级调度状态

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：后续请求集合（例中为空）。

**在做什么：** PP1 检查本轮接收的请求列表；固定示例的 M1–M5 已在开头到达，本轮列表为空，因此不会在这里新增请求状态。

**为什么需要：** 每个流水级只计算自己的模型层，却都需要知道在处理哪些请求、使用哪些缓存，以及把 KV 交给哪个 Decode 接收方。

**完成后：** 本轮收到的列表保留到本轮末尾，再向下一流水级转发；新请求能否成为可运行 batch，要继续经过 bootstrap 共识和选批。

**读图边界：** 收到请求不等于 GPU 已可运行。本例假设 M1–M5 在开头到达，后续很多轮收到的是空列表。

**源码：** [scheduler:2064](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler.py#L2064) · [pp:327](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L327)

### 2.2. 检查与 Decode 的准备状态，汇总 bootstrap 候选

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：请求候选集合；并非固定 M(i+1)。

**在做什么：** PP1 接收上一级的候选名单，再结合本级轮询结果：good 取交集，bad 取并集；中止的请求也进入失败处理。

**为什么需要：** 同一请求在各级都要准备好 KV 传输相关状态，才能让它一致地进入后续调度；某一级失败也必须让其他级知道。

**完成后：** 候选沿 PP0→PP1→PP2 汇总，之后还要通过 bootstrap 共识回流，成功请求才会移入 waiting_queue。

**读图边界：** 这是读取当前状态的一次轮询，不是原地等所有请求 ready；bootstrap 也不是 L2 缓存加载完成。

**源码：** [pp:593](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L593) · [pp:305](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L305)

### 2.3. 检查 KV 传输是否进入终态

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：inflight 请求集合。当前模型这一轮没有进入终态候选的请求。

**在做什么：** PP1 轮询已经处于 inflight 队列的请求，挑出 sender 报告 Success 或 Failed 的请求，并与上一级传来的终态名单取交集。

**为什么需要：** 本级发完 KV 不代表其他流水级也发完；释放 Prefill 侧资源前，需要逐级汇总传输终态，避免仍在发送的级过早清理。

**完成后：** 终态候选继续向末级汇总，随后形成 release 回流名单；真正清理由后面的 release 步骤完成。

**读图边界：** 终态包含成功和失败，源码分别处理。本图只建模成功路径；传输终态不代表 Decode 已生成完全部回答。

**源码：** [pp:635](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L635) · [prefill:972](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L972)

### 2.4. 整理上一批和未完成的 Prefill 分块

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：本例无中间 chunk。

**在做什么：** 检查是否有未完成的 chunked request，必要时缓存已完成的前缀、推进该分块的 KV 发送，并从上一批中滤掉不应继续保留的请求。

**为什么需要：** 长 prompt 可以分多轮计算；选下一批之前必须保存旧进度并整理 batch，避免把已经处理过或应暂时移出的请求再次计算。

**完成后：** 调度器获得整理后的上一批与分块状态，再处理缓存事件并选择本轮的新 batch。

**读图边界：** 本图采用完整 Prefill、没有中间 chunk，因此这里只表示检查和整理路径，不能把这段看成又执行了一次 GPU 前向。

**源码：** [prefill:1210](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L1210)

### 2.5. 统一这一轮可以处理多少条缓存 ACK

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：缓存 ACK 队列；write=0，load 按完成队列统计。本轮公布 write=0、load=0；这些条目的本地完成情况在后续 ACK 步骤确认。

**在做什么：** PP1 接收由 PP0 发布、上一级转来的缓存完成计数；如果还有下一流水级，继续异步转发相同计数。

**为什么需要：** 各级的拷贝进度可以不同，但缓存完成队列必须按一致的数量和顺序消费，否则缓存节点的生命周期可能分叉。

**完成后：** writing_check / loading_check 按公布的数量处理 ACK；每一级仍要确认对应的本地完成事件。

**读图边界：** 这里不是对所有 PP 的完成时间取最小值。PP0 发布计数，其他级可能随后等待本地 event。本例没有淘汰写回，write=0；空 batch 轮也要参加。

**源码：** [cache:3073](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L3073) · [cache:333](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L333) · [cache:3290](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L3290)

### 2.6. 确认本地回载完成，收尾缓存 ACK

**当前例子：** PP1 · L6：本轮前向 M3；本轮旧结果 M1。所选步骤的对象：已公布 ACK：M2 本地 H2D event。

**在做什么：** PP1 按本轮公布的数量处理 M2 的 load ACK，逐个同步本地 finish_event，移除 ongoing_load_back 记录并解除这次回载的临时保护。

**为什么需要：** 收到统一计数只说明该处理这些队首条目了，本级数据未必已经复制完；先等本地事件，再解除临时保护，才能安全管理缓存。

**完成后：** 已确认的条目退出回载中的状态，缓存节点可以按正常引用和淘汰规则管理，调度继续进入选批。

**读图边界：** 解除的是回载期间的临时引用，不是把正在服务请求的 KV 全部释放。当前请求对前缀的使用引用是另一层生命周期。

**源码：** [cache:3187](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L3187) · [cache:3210](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L3210)

### 2.7. 匹配前缀并判断本轮是否可以选批

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：M2。

**在做什么：** 本例本轮选择 M2：先匹配可复用的 KV 前缀，再检查 Prefill token、显存等预算，决定本轮接纳的计算范围，并通过延迟准入检查。

**为什么需要：** 请求已经到达、完成 bootstrap，也可能因预算或调度条件不能立刻运行。先判断可接纳范围，才能避免超额分配资源。

**完成后：** 通过预检查后，才准备必要的 host 缓存回载，并正式提交准入。

**读图边界：** 本图固定 L4–L8 每轮选 M1–M5，假设容量充足；改时间不会重新模拟真实预算竞争、选批策略或请求到达。

**源码：** [scheduler:3959](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler.py#L3959) · [policy:1185](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/schedule_policy.py#L1185)

### 2.8. 为 host 命中的前缀准备回载

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：M2。

**在做什么：** M2 的一部分前缀 KV 只在主机内存中。调度器调用 init_load_back，为回载准备 GPU 位置和缓存节点保护，取得可接入请求前缀的设备索引。

**为什么需要：** 命中主机缓存可以少算一段前缀，但 GPU 做 attention 前还需要对应 KV 和有效的设备地址。

**完成后：** 回载准备成功后，把设备索引纳入请求的前缀，再提交准入；实际异步 H2D 由后面的 start_loading 发起。

**读图边界：** “拿到设备索引”不代表数据已复制完成。本图只有 M2、M4 有 host hit，其他 batch 不应被讲成也在回载。

**源码：** [policy:1208](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/schedule_policy.py#L1208) · [cache:3243](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L3243)

### 2.9. 正式接纳请求并保护它要使用的前缀

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：M2。

**在做什么：** 把 M2 的本轮计算范围写入请求，增加前缀使用引用，将请求加入 can_run_list，并扣减这一轮的调度预算。

**为什么需要：** 前面的预算判断和回载准备通过后，才可以正式占用预算并保护前缀，防止后续选批或缓存回收破坏这次计算。

**完成后：** 调度器随后将已接纳的请求构造成 ScheduleBatch，进入缓存提交和输入准备。

**读图边界：** 这是准入状态提交，不是 GPU 提交；它在回载准备成功之后、实际 H2D 提交之前。

**源码：** [policy:1347](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/schedule_policy.py#L1347) · [scheduler:4037](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler.py#L4037)

### 2.10. 构造 batch，提交已准备的缓存回载

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：M2。

**在做什么：** 调度器将已接纳的 M2 构造成 batch。start_loading 合并待回载任务，记录 start_event，提交 H2D，并把完成事件放进 ACK 队列；batch 保存 consumer index。

**为什么需要：** 把准备好的缓存任务交给异步拷贝流，同时保留批次与逐层完成事件的对应关系，计算时才能等待正确的数据。

**完成后：** CPU 继续准备输入；若确有回载，H2D 独立推进，完成后还需要在后续缓存事件轮消费 ACK。

**读图边界：** 提交异步拷贝不等于拷贝完成；consumer index 用来定位这次回载的事件组，不是 micro-batch 编号。

**源码：** [scheduler:4054](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler.py#L4054) · [controller:941](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/cache_controller.py#L941) · [controller:960](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/cache_controller.py#L960)

### 2.11. 把请求整理成可执行的 Prefill 输入

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：M2。

**在做什么：** 为 M2 准备本次需要计算的 token、序列长度、已命中前缀长度以及新 KV 的写入位置等 batch 字段，并保存到当前 PP 槽位。

**为什么需要：** 模型执行需要连续的输入张量和明确的缓存位置，不能直接拿调度队列里的请求对象进行前向。

**完成后：** 接下来接收上一级传来的中间激活，再提交本级计算。

**读图边界：** 这里准备的是本轮未被缓存覆盖的计算输入；部分输入搬运会延迟到 forward stream，不能把准备阶段结束等同于所有设备工作结束。

**源码：** [scheduler:4058](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler.py#L4058) · [pp:256](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L256)

### 2.12. 接收上一级算好的中间激活

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：M2。

**在做什么：** PP1 从 PP0 接收 M2 的 proxy 张量（hidden states，即中间特征），并包装成供本级模型继续计算的输入。

**为什么需要：** PP 把模型层拆在不同级上；后一级没有前面层的输出，就无法继续执行自己持有的后续层。

**完成后：** 当前 batch 的上级输入就绪，随后在本地提交前向；同一通信通道中的 output 消息由类型标记区分。

**读图边界：** proxy 是 Prefill 流水级之间的中间激活，既不是发给 Decode 的 KV，也不是最终 token 结果。PP0 没有这一步。

**源码：** [pp:858](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L858) · [pp:825](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L825)

### 2.13. 向 GPU 流提交本级前向

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：M2。

**在做什么：** PP1 为 M2 在 forward stream 上安排 run_batch，保存本批结果所需的元数据，并在流上记录 launch_event。

**为什么需要：** CPU 调度与 GPU 计算异步推进；后续发送激活或结果时，需要一个事件来保证不会读到尚未计算完成的数据。

**完成后：** CPU 可以继续这一轮的结果、共识或转发工作；GPU 在自己的流依赖满足后实际执行。

**读图边界：** 记录 event 只是安排完成标记，不能理解为 CPU 此刻已经等到 GPU 算完；图中 launch 和 GPU 前向是两段不同工作。

**源码：** [pp:1221](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1221) · [pp:1247](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1247)

### 2.14. 执行当前 batch 在本级的模型层

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：M2。

**在做什么：** PP1 的 GPU 实际计算 M2 在本级持有的层，读取所需前缀 KV，并产生本级新增 KV和传给下一级的中间激活。

**为什么需要：** 同一 batch 必须依次通过 PP0、PP1、PP2 的模型层，才能得到完整 Prefill 结果；各级也因此拥有需要交给 Decode 的本地 KV。

**完成后：** 前向完成后，M2 的中间激活可以传给 PP2，让后续层继续计算。

**读图边界：** 源码按层等待 KV 就绪，允许回载与前向重叠。本图把它粗化成整份 H2D 完成后再画 GPU 前向，不能据此断言真实程序有整批屏障。

**源码：** [pp:1229](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1229) · [controller:64](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/cache_controller.py#L64)

### 2.15. 把命中的前缀 KV 从主机搬回 GPU

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：M2。

**在做什么：** PP1 的拷贝流将 M2 命中的主机缓存 KV 从 L2 搬到 GPU HBM，按层记录完成事件。

**为什么需要：** 复用已有前缀能够避免重复计算，但 attention 执行时需要相应 KV 在设备上可用。

**完成后：** 逐层事件让计算知道哪些层已可使用；最终完成事件则由后面的 load ACK 检查消费，用于缓存状态收尾。

**读图边界：** 源码按层等待 KV 就绪，允许回载与前向重叠。本图把它粗化成整份 H2D 完成后再画 GPU 前向，不能据此断言真实程序有整批屏障。 图中同一拷贝资源的串行安排也是模型简化。

**源码：** [controller:960](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/cache_controller.py#L960) · [controller:64](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/cache_controller.py#L64) · [cache:3187](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L3187)

### 2.16. 安排把本级激活发给下一流水级

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：M2。

**在做什么：** PP1 先在发送所在流上等待 M2 的 launch_event，再异步提交带 proxy 类型的张量发送到 PP2，保存 send_proxy_work。

**为什么需要：** 发送必须排在本级前向产出激活之后；通过流事件建立先后关系，可以让 CPU 继续调度，而不必在此同步等待 GPU 完成。

**完成后：** 本级 GPU 输出就绪后，激活传输才能真正推进；发送句柄留到后续轮次回收。

**读图边界：** wait_event 是设备流依赖，不是 event.synchronize()。本步骤结束只能说明发送已安排，不能说明对方已收到。

**源码：** [pp:336](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L336) · [pp:340](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L340)

### 2.17. 中间激活传到下一流水级

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：M2。

**在做什么：** PP1 正在把 M2 在本级算出的 hidden states（中间特征张量）传给 PP2。模型要求发送已提交，且本级 GPU 前向已经产出这些数据。

**为什么需要：** PP2 只持有后续模型层，必须拿到这份输入，才能继续同一个 M2 的计算。

**完成后：** PP2 的 recv_proxy 可以接收这份激活，再推进本级前向。

**读图边界：** 这是 Prefill 级间激活传输，不是 PD KV 传输，也不是 output 回流。条宽代表模型中的在途时长，不是 CPU 调用 send 的耗时。

**源码：** [pp:340](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L340) · [pp:858](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L858)

### 2.18. 提交或转发末级产出的 output

**当前例子：** PP1 · L7：本轮前向 M4；本轮旧结果 M2。所选步骤的对象：M1。

**在做什么：** PP1 把之前收到并保存的 M1 output 转发给 PP2；这里使用的是保存的 pp_outputs，并不是重新做一次末级计算。

**为什么需要：** 只有末级拥有完整模型输出，但每一级都要用同一结果更新本地请求并推进 KV 发送，所以结果沿 PP2→PP0→PP1→PP2 回流。

**完成后：** 数据在依赖满足后传输，接收级把它关联到相应的旧 batch，发送 work 留待后续回收。

**读图边界：** output 主要携带 next_token_ids 等结果字段；它和顺着 PP0→PP1→PP2 流动的 proxy 激活是两类消息。

**源码：** [pp:983](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L983) · [pp:769](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L769)

### 2.19. output 沿结果环路传输

**当前例子：** PP1 · L7：本轮前向 M4；本轮旧结果 M2。所选步骤的对象：M1。

**在做什么：** M1 的 output 正从 PP1 传往 PP2。这是对已收到结果的接力转发。

**为什么需要：** 让持有不同模型层和本地 KV 的各级，对同一请求使用一致的 token 结果进行后处理。

**完成后：** 接收级的 recv_out 得到旧 batch 的输出，准备结果对象和就绪事件，再进入请求后处理。

**读图边界：** 这里传的是结果张量，不是把 Prefill 的全部 KV 绕 PP 环传一遍；KV 另由各级发送给 Decode。

**源码：** [pp:983](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L983) · [pp:1072](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1072)

### 2.20. 接收要在本轮处理的旧 batch 结果

**当前例子：** PP1 · L6：本轮前向 M3；本轮旧结果 M1。所选步骤的对象：M1。

**在做什么：** PP1 从 PP0 接收 M1 的 output，使用该旧 batch 保存的元数据解释结果，得到 next_token_ids 等字段。

**为什么需要：** 本轮提交的可能是另一份新 batch；旧 batch 的结果稍后才沿环路返回，需要根据槽位和元数据找到正确的请求。

**完成后：** 接着在 copy stream 上预处理结果并记录就绪事件；等到后处理位置再同步事件、更新旧请求。

**读图边界：** 本例 depth=0，结果接收位于当前前向提交之后；不能把这里的旧结果默认当作本轮刚 launch 的 batch。

**源码：** [pp:1072](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1072) · [pp:1084](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1084)

### 2.21. 预处理旧结果并记录就绪事件（图中 D2H）

**当前例子：** PP1 · L6：本轮前向 M3；本轮旧结果 M1。所选步骤的对象：M1。

**在做什么：** 为 M1 构造 GenerationBatchResult，处理结果字段，并在 copy stream 上记录 d2h_event；该流先等待 schedule_stream 的先行工作。

**为什么需要：** 后续 CPU 请求处理要使用这份结果，需要一个明确的完成点，确保预处理及按配置发起的异步拷贝已达到可使用的状态。

**完成后：** 后面的 wait_copy 同步这个事件，然后进入 process_batch_result_disagg_prefill。

**读图边界：** 图中 D2H 是结果预处理／就绪区间的简写。源码根据输出配置决定拷贝内容；本例关闭 sampling mask，相关分支只记录 event，不能理解为这里必然完整复制 token 张量，更不是 KV 回载。

**源码：** [pp:1084](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1084) · [pp:1091](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1091) · [pp:1100](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1100)

### 2.22. 在使用旧结果前同步完成事件

**当前例子：** PP1 · L6：本轮前向 M3；本轮旧结果 M1。所选步骤的对象：M1。

**在做什么：** PP1 调用 d2h_event.synchronize()，确认 M1 的结果就绪事件已经完成，才允许 CPU 进入后处理。

**为什么需要：** 异步流已经提交任务并不保证数据可读；这里是从异步结果准备跨到 CPU 使用结果的明确同步点。

**完成后：** 读取旧结果、更新请求并推进其 Prefill KV 发送。

**读图边界：** 若事件早已完成，这个调用可以很快返回；若尚未完成，图中此前的斜线等待段表示依赖尚未满足的区间，勿与此调用的服务段重复计算。

**源码：** [pp:317](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L317) · [prefill:800](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L800)

### 2.23. 把旧结果写回请求，转入等待 KV 传输阶段

**当前例子：** PP1 · L6：本轮前向 M3；本轮旧结果 M1。所选步骤的对象：M1。

**在做什么：** 处理 M1 的 Prefill 结果：把 next_token_ids 转成 CPU 可用的值，更新请求的 output_ids，按缓存策略保存未结束请求的前缀，并加入 disagg_prefill_inflight_queue。

**为什么需要：** GPU 返回的是 batch 结果，调度器需要把它落到逐个请求；这些请求的 KV 还要交给 Decode，因此此时不能直接丢弃本地状态。

**完成后：** 正常完整 Prefill 路径继续提交本级 KV；请求留在 inflight 队列中，等待后续终态轮询与清理。

**读图边界：** 源码在同一个结果处理函数内推进这些动作；图为便于阅读把后面的 send_kv 单独画出。加入 inflight 不代表传输已经完成。

**源码：** [prefill:781](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L781) · [prefill:800](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L800) · [prefill:859](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L859)

### 2.24. 把本级 KV 交给传输后端发往 Decode

**当前例子：** PP1 · L6：本轮前向 M3；本轮旧结果 M1。所选步骤的对象：M1。

**在做什么：** 为 M1 找出本级已计算好的 KV 范围，转换为传输需要的物理页索引，交给 disagg_kv_sender.send；完整 Prefill 在这里提交最后一块及所需附带状态。

**为什么需要：** Decode 要接着生成 token，需要继承 Prefill 已计算的上下文 KV。每个 PP 级负责自己持有层的 KV。

**完成后：** 传输后端异步推进 KV 发送，后续通过 sender.poll() 观察是否完成。

**读图边界：** 目的地是图外的 Decode 接收端，不是下一 PP 级；本步骤结束表示发送已提交，资源释放还要等终态和共识。

**源码：** [prefill:893](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L893) · [prefill:1502](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L1502)

### 2.25. 本级 KV 正在传向 Decode

**当前例子：** PP1 · L6：本轮前向 M3；本轮旧结果 M1。所选步骤的对象：M1。

**在做什么：** PP1 已为 M1 提交 KV，传输后端正在把这些缓存页送往 Decode。图把它表示为一段异步在途区间。

**为什么需要：** 计算与传输可以分开推进；只看 GPU 前向结束，无法判断 Prefill 请求什么时候可以安全清理。

**完成后：** 本地传输终态会在后续轮询中被观察，经过 PP 共识回流和本地复查后才允许清理。

**读图边界：** 本图假设每级的 KV 在途段串行，真实后端的传输并发与链路争用未建模。条段结束既不等于马上释放，也不等于 Decode 生成结束。

**源码：** [prefill:1502](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L1502) · [pp:635](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L635) · [prefill:972](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L972)

### 2.26. 将 bootstrap 共识沿环路传回各级

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：请求候选集合。这次发送使用 PP1 · L4 的已收共识。

**在做什么：** PP1 将上一轮已经收到并保存的 bootstrap 共识转发给 PP2。

**为什么需要：** 各级需要看到同一份成功和失败名单，才能一致地把成功请求放入可调度队列，并处理失败请求。

**完成后：** 对应级收到名单后调用 process_bootstrapped_queue；发送 work 还需要在本轮后面的 commit_bc 回收。

**读图边界：** 发送的是请求 ID 和状态名单，不传激活或 KV；名单来自队列快照，不能按“本轮当前 batch”理解其成员。

**源码：** [pp:660](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L660) · [pp:305](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L305)

### 2.27. 按 bootstrap 共识把请求移入等待队列

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：候选集合（本例无新准入）。

**在做什么：** PP1 接收共识 good/bad 名单，调用 process_bootstrapped_queue 处理 bootstrap 队列：成功请求进入 waiting_queue，失败请求走对应处理。

**为什么需要：** 把“各级已同意准备就绪”转换成真正可参与选批的调度状态，避免某一级先计算其他级还没准备好的请求。

**完成后：** 成功请求可以在后续选批中被接纳；因为本轮的选批位置已经过去，所以不是收到名单就立即补做本轮前向。

**读图边界：** 固定示例后续没有新准入，仍可传递和处理空名单；本步骤不表示每轮都增加新请求。

**源码：** [pp:305](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L305) · [pp:589](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L589)

### 2.28. 将允许复查清理的终态名单传回各级

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：终态请求集合。这次发送使用 PP1 · L4 的已收共识。

**在做什么：** PP1 将之前收到并保存的 release 名单转发给 PP2。

**为什么需要：** 各级需要协调请求的传输生命周期；仅凭某一级的发送完成就回收资源，可能早于其他级的完成时刻。

**完成后：** 各级接收名单后，还会在本地 inflight 队列中复查状态，确定本轮实际能清理的请求。

**读图边界：** 名单中的对象可能是更早的请求，与本轮当前 batch 或正在处理的旧 batch 没有固定一一对应关系。

**源码：** [pp:683](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L683) · [prefill:972](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L972)

### 2.29. 接收 release 名单，限制本轮清理范围

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：已终态的历史请求集合。

**在做什么：** PP1 接收对应槽位回流的请求 ID 名单，保存为 next_release_rids，供后面的 inflight 队列清理使用。

**为什么需要：** 将各级已经汇总的终态信息带回本地，限定哪些请求可以进入本轮的清理检查。

**完成后：** 先完成本轮需要的旧结果处理，再用这份名单调用 process_disagg_prefill_inflight_queue。

**读图边界：** 收到名单尚未释放资源；名单也可能为空。只有本地复查仍满足条件的请求才会被清理。

**源码：** [pp:312](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L312) · [prefill:972](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L972)

### 2.30. 复查传输状态，清理 Prefill 请求资源

**当前例子：** PP1 · L10：本轮前向 无；本轮旧结果 M5。所选步骤的对象：释放 M1。

**在做什么：** PP1 本轮实际清理 M1：在 release 名单限定的范围内复查本地 sender，成功后依次 release_kv_cache、finish(SUCCESS)、sender.clear，归还元数据并移出 inflight 队列。

**为什么需要：** 避免 KV 仍在被传输后端使用时回收请求资源，同时及时归还已经完成 Prefill 交接的请求占用。

**完成后：** 清理完成的请求退出本地传输生命周期；未完成的请求留待后续轮询，Decode 端继续自己的生成流程。

**读图边界：** 释放请求引用不等于所有前缀缓存都立即从显存消失。当前默认缓存的 finish(SUCCESS) 不取消异步缓存工作；图只表示成功路径。

**源码：** [prefill:972](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L972) · [prefill:1025](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L1025) · [prefill:1074](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L1074)

### 2.31. 保存回流状态，结束这一轮槽位迭代

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：队列级。

**在做什么：** 把本轮收到的 output、bootstrap 共识和 release 名单保存到调度器状态，更新 batch_is_full 等标记，然后进入下一次槽位迭代。

**为什么需要：** 结果和共识沿流水线分轮传播，下次转发需要使用这次保存的状态；环形槽位也要继续被复用。

**完成后：** CPU 推进下一轮接收和状态检查，已经提交的 GPU、拷贝与通信任务仍按各自依赖异步运行。

**读图边界：** loop 结束不等于当前 batch 结束，更不等于服务器所有任务结束。L# 是本级迭代编号，不是全局同步屏障。

**源码：** [pp:346](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L346) · [pp:223](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L223)

### 2.32. 提交请求的逐级转发

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：队列级。

**在做什么：** PP1 把本轮收到的请求列表异步发送给 PP2，并保存发送 work，供之后确认完成。

**为什么需要：** 使后续流水级也建立相同请求的本地调度状态。

**完成后：** 消息在通信路径上推进，CPU 可以继续本轮剩余工作；后续轮次再回收发送句柄。

**读图边界：** 这是队列级控制消息，名单可能为空，也可能涉及多份请求；其成员不由当前前向 batch 决定。

**源码：** [pp:327](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L327) · [pp:735](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L735)

### 2.33. 请求消息正在传输

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：本轮队列快照。

**在做什么：** 本轮收到的请求列表正从 PP1 传往 PP2；这是前面异步发送提交后的在途阶段。

**为什么需要：** 使后续流水级也建立相同请求的本地调度状态。

**完成后：** PP2 的 recv_req 可以消费消息，继续本级请求或状态处理。

**读图边界：** 图中的在途段是教学模型单独拆出的通信时长，不是源码中又调用了一个同名 CPU 函数；控制消息不携带完整激活或 KV。

**源码：** [pp:327](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L327)

### 2.34. 提交bootstrap 候选的逐级转发

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：队列级。

**在做什么：** PP1 把本级汇总的 bootstrap good/bad 候选名单异步发送给 PP2，并保存发送 work，供之后确认完成。

**为什么需要：** 让下一级结合自己的准备状态继续求 good 交集、bad 并集。

**完成后：** 消息在通信路径上推进，CPU 可以继续本轮剩余工作；后续轮次再回收发送句柄。

**读图边界：** 这是队列级控制消息，名单可能为空，也可能涉及多份请求；其成员不由当前前向 batch 决定。

**源码：** [pp:331](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L331) · [pp:735](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L735)

### 2.35. bootstrap 候选消息正在传输

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：本轮队列快照。

**在做什么：** 本级汇总的 bootstrap good/bad 候选名单正从 PP1 传往 PP2；这是前面异步发送提交后的在途阶段。

**为什么需要：** 让下一级结合自己的准备状态继续求 good 交集、bad 并集。

**完成后：** PP2 的 boot_poll 可以消费消息，继续本级请求或状态处理。

**读图边界：** 图中的在途段是教学模型单独拆出的通信时长，不是源码中又调用了一个同名 CPU 函数；控制消息不携带完整激活或 KV。

**源码：** [pp:331](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L331)

### 2.36. 提交transfer 终态候选的逐级转发

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：队列级。

**在做什么：** PP1 把本级汇总的 KV 传输终态候选名单异步发送给 PP2，并保存发送 work，供之后确认完成。

**为什么需要：** 让下一级结合本地传输状态继续求交集，最终形成 release 名单。

**完成后：** 消息在通信路径上推进，CPU 可以继续本轮剩余工作；后续轮次再回收发送句柄。

**读图边界：** 这是队列级控制消息，名单可能为空，也可能涉及多份请求；其成员不由当前前向 batch 决定。

**源码：** [pp:334](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L334) · [pp:735](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L735)

### 2.37. transfer 终态候选消息正在传输

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：本轮队列快照。

**在做什么：** 本级汇总的 KV 传输终态候选名单正从 PP1 传往 PP2；这是前面异步发送提交后的在途阶段。

**为什么需要：** 让下一级结合本地传输状态继续求交集，最终形成 release 名单。

**完成后：** PP2 的 term_poll 可以消费消息，继续本级请求或状态处理。

**读图边界：** 图中的在途段是教学模型单独拆出的通信时长，不是源码中又调用了一个同名 CPU 函数；控制消息不携带完整激活或 KV。

**源码：** [pp:334](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L334)

### 2.38. 确认并回收上一轮请求发送

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：队列级。

**在做什么：** PP1 对上一轮请求发送保留的 work 调用 wait()，随后清空列表。这里处理的是已提交的通信，不重新发送一份数据。

**为什么需要：** 避免未完成的请求发送句柄无限累积，再保存新一轮发送任务。

**完成后：** 调度继续执行后续步骤；下方跨 PP 的前置关系可以追到对方对应的接收操作。

**读图边界：** 源码调用 Work.wait() 并清空句柄列表；其对 CPU 或设备流的具体影响取决于通信后端。图中的等待宽度是依赖模型的示意值。

**源码：** [pp:238](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L238) · [pp:705](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L705)

### 2.39. 确认并回收上一轮 bootstrap 候选发送

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：队列级。

**在做什么：** PP1 对上一轮 bootstrap 候选发送保留的 work 调用 wait()，随后清空列表。这里处理的是已提交的通信，不重新发送一份数据。

**为什么需要：** 维持候选消息的有序推进，再保存新一轮发送任务。

**完成后：** 调度继续执行后续步骤；下方跨 PP 的前置关系可以追到对方对应的接收操作。

**读图边界：** 源码调用 Work.wait() 并清空句柄列表；其对 CPU 或设备流的具体影响取决于通信后端。图中的等待宽度是依赖模型的示意值。

**源码：** [pp:242](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L242) · [pp:705](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L705)

### 2.40. 确认并回收上一轮 transfer 终态候选发送

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：队列级。

**在做什么：** PP1 对上一轮 transfer 终态候选发送保留的 work 调用 wait()，随后清空列表。这里处理的是已提交的通信，不重新发送一份数据。

**为什么需要：** 有序推进传输状态消息并回收通信句柄。

**完成后：** 调度继续执行后续步骤；下方跨 PP 的前置关系可以追到对方对应的接收操作。

**读图边界：** 源码调用 Work.wait() 并清空句柄列表；其对 CPU 或设备流的具体影响取决于通信后端。图中的等待宽度是依赖模型的示意值。

**源码：** [pp:245](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L245) · [pp:705](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L705)

### 2.41. 确认并回收上一轮激活发送

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：M1。

**在做什么：** PP1 对上一轮激活发送保留的 work 调用 wait()，随后清空列表。这里处理的是已提交的通信，不重新发送一份数据。

**为什么需要：** 确认此前提交的激活发送达到后端的完成条件，再推进当前 batch 的提交。

**完成后：** 调度继续执行后续步骤；下方跨 PP 的前置关系可以追到对方对应的接收操作。

**读图边界：** 源码调用 Work.wait() 并清空句柄列表；其对 CPU 或设备流的具体影响取决于通信后端。图中的等待宽度是依赖模型的示意值。

**源码：** [pp:272](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L272) · [pp:705](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L705)

### 2.42. 确认并回收上一轮 output 发送

**当前例子：** PP1 · L8：本轮前向 M5；本轮旧结果 M3。所选步骤的对象：M1。

**在做什么：** PP1 对上一轮 output 发送保留的 work 调用 wait()，随后清空列表。这里处理的是已提交的通信，不重新发送一份数据。

**为什么需要：** 回收旧结果发送句柄，再启动这一轮的结果收发。

**完成后：** 调度继续执行后续步骤；下方跨 PP 的前置关系可以追到对方对应的接收操作。

**读图边界：** 源码调用 Work.wait() 并清空句柄列表；其对 CPU 或设备流的具体影响取决于通信后端。图中的等待宽度是依赖模型的示意值。

**源码：** [pp:710](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L710) · [pp:705](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L705)

### 2.43. 确认并回收本轮 bootstrap 共识发送

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：队列级。

**在做什么：** PP1 对本轮 bootstrap 共识发送保留的 work 调用 wait()，随后清空列表。这里处理的是已提交的通信，不重新发送一份数据。

**为什么需要：** 确认共识消息的发送工作得到处理，避免留下未回收的通信句柄。

**完成后：** 调度继续执行后续步骤；下方跨 PP 的前置关系可以追到对方对应的接收操作。

**读图边界：** 源码调用 Work.wait() 并清空句柄列表；其对 CPU 或设备流的具体影响取决于通信后端。图中的等待宽度是依赖模型的示意值。

**源码：** [pp:311](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L311) · [pp:705](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L705)

### 2.44. 确认并回收本轮 release 共识发送

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：队列级。

**在做什么：** PP1 对本轮 release 共识发送保留的 work 调用 wait()，随后清空列表。这里处理的是已提交的通信，不重新发送一份数据。

**为什么需要：** 确认终态回流消息的发送工作得到处理，再继续本地后处理。

**完成后：** 调度继续执行后续步骤；下方跨 PP 的前置关系可以追到对方对应的接收操作。

**读图边界：** 源码调用 Work.wait() 并清空句柄列表；其对 CPU 或设备流的具体影响取决于通信后端。图中的等待宽度是依赖模型的示意值。

**源码：** [pp:314](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L314) · [pp:705](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L705)

### 2.45. 回收上一轮缓存计数的异步发送

**当前例子：** PP1 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：队列级。

**在做什么：** PP1 在开始新的缓存事件轮之前，等待 work_list 中上一轮缓存计数发送的 work，并清空这个列表。

**为什么需要：** 限制未完成发送的积累；当下游流水级落后时，通过这里形成反压，使缓存控制消息按轮推进。

**完成后：** 然后发布或接收本轮 write/load 完成计数，再处理本地缓存 ACK。

**读图边界：** 这里回收的是缓存计数通信，不是等待所有 KV 回载，也不是请求的 PD 传输释放。

**源码：** [cache:321](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L321) · [cache:3290](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L3290)

## 3. 分支与等待：同名步骤不一定在做同样的工作

空轮、无 host hit、末级发出与中间级转发，必须结合当前上下文阅读。斜线等待是前置条件未满足的区间，不是重复执行一次操作。

### 3.1. 匹配前缀并判断本轮是否可以选批

**当前例子：** PP0 · L1：本轮前向 无；本轮旧结果 无。所选步骤的对象：无可计算 batch。

**在做什么：** 本例这一轮没有选出可计算的请求，get_new_batch_prefill 返回 None；调度仍继续推进已有请求的状态、通信与清理。

**为什么需要：** 请求已经到达、完成 bootstrap，也可能因预算或调度条件不能立刻运行。先判断可接纳范围，才能避免超额分配资源。

**完成后：** 跳过本轮新 batch 的准备和前向；后续旧结果处理或控制通信仍可能有工作。

**读图边界：** 本图固定 L4–L8 每轮选 M1–M5，假设容量充足；改时间不会重新模拟真实预算竞争、选批策略或请求到达。

**源码：** [scheduler:3959](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler.py#L3959) · [policy:1185](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/schedule_policy.py#L1185)

### 3.2. 为 host 命中的前缀准备回载

**当前例子：** PP0 · L4：本轮前向 M1；本轮旧结果 无。所选步骤的对象：M1。

**在做什么：** M1 在本例没有需要回载的 host 命中，因此跳过 init_load_back；这段表示分支检查，不会产生 KV 拷贝。

**为什么需要：** 命中主机缓存可以少算一段前缀，但 GPU 做 attention 前还需要对应 KV 和有效的设备地址。

**完成后：** 回载准备成功后，把设备索引纳入请求的前缀，再提交准入；实际异步 H2D 由后面的 start_loading 发起。

**读图边界：** “拿到设备索引”不代表数据已复制完成。本图只有 M2、M4 有 host hit，其他 batch 不应被讲成也在回载。

**源码：** [policy:1208](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/schedule_policy.py#L1208) · [cache:3243](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L3243)

### 3.3. 构造 batch，提交已准备的缓存回载

**当前例子：** PP0 · L4：本轮前向 M1；本轮旧结果 无。所选步骤的对象：M1：load 队列为空。

**在做什么：** 调度器构造 M1：load 队列为空 的 batch 并检查回载队列；本例该队列为空，start_loading 返回 -1，不提交 H2D。

**为什么需要：** 把准备好的缓存任务交给异步拷贝流，同时保留批次与逐层完成事件的对应关系，计算时才能等待正确的数据。

**完成后：** CPU 继续准备输入；若确有回载，H2D 独立推进，完成后还需要在后续缓存事件轮消费 ACK。

**读图边界：** 提交异步拷贝不等于拷贝完成；consumer index 用来定位这次回载的事件组，不是 micro-batch 编号。

**源码：** [scheduler:4054](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler.py#L4054) · [controller:941](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/cache_controller.py#L941) · [controller:960](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/cache_controller.py#L960)

### 3.4. 确认本地回载完成，收尾缓存 ACK

**当前例子：** PP0 · L1：本轮前向 无；本轮旧结果 无。所选步骤的对象：已公布 ACK：本轮计数为 0。

**在做什么：** PP0 检查本轮缓存完成计数；当前 load ACK 数为 0，没有需要同步并收尾的回载条目。

**为什么需要：** 收到统一计数只说明该处理这些队首条目了，本级数据未必已经复制完；先等本地事件，再解除临时保护，才能安全管理缓存。

**完成后：** 已确认的条目退出回载中的状态，缓存节点可以按正常引用和淘汰规则管理，调度继续进入选批。

**读图边界：** 解除的是回载期间的临时引用，不是把正在服务请求的 KV 全部释放。当前请求对前缀的使用引用是另一层生命周期。

**源码：** [cache:3187](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L3187) · [cache:3210](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L3210)

### 3.5. 复查传输状态，清理 Prefill 请求资源

**当前例子：** PP0 · L3：本轮前向 无；本轮旧结果 无。所选步骤的对象：候选终态请求；不等于当前旧 batch。

**在做什么：** PP0 检查 release 名单和本地 inflight 状态；当前场景这一轮没有实际清理的请求，未满足条件的请求继续保留。

**为什么需要：** 避免 KV 仍在被传输后端使用时回收请求资源，同时及时归还已经完成 Prefill 交接的请求占用。

**完成后：** 清理完成的请求退出本地传输生命周期；未完成的请求留待后续轮询，Decode 端继续自己的生成流程。

**读图边界：** 释放请求引用不等于所有前缀缓存都立即从显存消失。当前默认缓存的 finish(SUCCESS) 不取消异步缓存工作；图只表示成功路径。

**源码：** [prefill:972](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L972) · [prefill:1025](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L1025) · [prefill:1074](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L1074)

### 3.6. 接收请求，建立本级调度状态

**当前例子：** PP0 · L1：本轮前向 无；本轮旧结果 无。所选步骤的对象：M1–M5 请求集合。

**在做什么：** PP0 从请求接收器取得本轮新请求，处理输入并建立本级请求状态；PD Prefill 的新生成请求随后需要经过 bootstrap 握手准备。

**为什么需要：** 每个流水级只计算自己的模型层，却都需要知道在处理哪些请求、使用哪些缓存，以及把 KV 交给哪个 Decode 接收方。

**完成后：** 本轮收到的列表保留到本轮末尾，再向下一流水级转发；新请求能否成为可运行 batch，要继续经过 bootstrap 共识和选批。

**读图边界：** 收到请求不等于 GPU 已可运行。本例假设 M1–M5 在开头到达，后续很多轮收到的是空列表。

**源码：** [scheduler:2064](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler.py#L2064) · [pp:327](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L327)

### 3.7. 统一这一轮可以处理多少条缓存 ACK

**当前例子：** PP0 · L1：本轮前向 无；本轮旧结果 无。所选步骤的对象：缓存 ACK 队列；write=0，load 按完成队列统计。本轮公布 write=0、load=0；这些条目的本地完成情况在后续 ACK 步骤确认。

**在做什么：** PP0 统计缓存 write/load 完成队列中从队首开始连续就绪的 ACK 数，将计数合并同步；级内参与进程取最小值，然后把结果向后续 PP 级传播。

**为什么需要：** 各级的拷贝进度可以不同，但缓存完成队列必须按一致的数量和顺序消费，否则缓存节点的生命周期可能分叉。

**完成后：** writing_check / loading_check 按公布的数量处理 ACK；每一级仍要确认对应的本地完成事件。

**读图边界：** 这里不是对所有 PP 的完成时间取最小值。PP0 发布计数，其他级可能随后等待本地 event。本例没有淘汰写回，write=0；空 batch 轮也要参加。

**源码：** [cache:3073](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L3073) · [cache:333](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L333) · [cache:3290](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L3290)

### 3.8. 提交或转发末级产出的 output

**当前例子：** PP2 · L4：本轮前向 M1；本轮旧结果 无。所选步骤的对象：M1。

**在做什么：** PP2 取出 M1 的前向完成事件和输出，在发送流上等待该事件，再异步发往 PP0。

**为什么需要：** 只有末级拥有完整模型输出，但每一级都要用同一结果更新本地请求并推进 KV 发送，所以结果沿 PP2→PP0→PP1→PP2 回流。

**完成后：** 数据在依赖满足后传输，接收级把它关联到相应的旧 batch，发送 work 留待后续回收。

**读图边界：** output 主要携带 next_token_ids 等结果字段；它和顺着 PP0→PP1→PP2 流动的 proxy 激活是两类消息。

**源码：** [pp:983](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L983) · [pp:769](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L769)

### 3.9. 预处理旧结果并记录就绪事件（图中 D2H）

**当前例子：** PP2 · L6：本轮前向 M3；本轮旧结果 M1。所选步骤的对象：M1。

**在做什么：** 为 M1 构造 GenerationBatchResult，处理结果字段，并在 copy stream 上记录 d2h_event；该流先等待 schedule_stream 的先行工作。

**为什么需要：** 后续 CPU 请求处理要使用这份结果，需要一个明确的完成点，确保预处理及按配置发起的异步拷贝已达到可使用的状态。

**完成后：** 后面的 wait_copy 同步这个事件，然后进入 process_batch_result_disagg_prefill。

**读图边界：** 图中 D2H 是结果预处理／就绪区间的简写。源码根据输出配置决定拷贝内容；本例关闭 sampling mask，相关分支只记录 event，不能理解为这里必然完整复制 token 张量，更不是 KV 回载。末级 copy stream 还会继承等待当前前向完成事件的流依赖。

**源码：** [pp:1084](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1084) · [pp:1091](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1091) · [pp:1100](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1100)

### 3.10. 将 bootstrap 共识沿环路传回各级

**当前例子：** PP2 · L4：本轮前向 M1；本轮旧结果 无。所选步骤的对象：请求候选集合。这次发送使用 PP2 · L4 的本轮候选。

**在做什么：** PP2 将逐级汇总后的 bootstrap good/bad 名单发回 PP0，开始共识回流。

**为什么需要：** 各级需要看到同一份成功和失败名单，才能一致地把成功请求放入可调度队列，并处理失败请求。

**完成后：** 对应级收到名单后调用 process_bootstrapped_queue；发送 work 还需要在本轮后面的 commit_bc 回收。

**读图边界：** 发送的是请求 ID 和状态名单，不传激活或 KV；名单来自队列快照，不能按“本轮当前 batch”理解其成员。

**源码：** [pp:660](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L660) · [pp:305](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L305)

### 3.11. 将允许复查清理的终态名单传回各级

**当前例子：** PP2 · L4：本轮前向 M1；本轮旧结果 无。所选步骤的对象：终态请求集合。这次发送使用 PP2 · L4 的本轮候选。

**在做什么：** PP2 把逐级取交集后的传输终态名单发回 PP0，启动 release 共识回流。

**为什么需要：** 各级需要协调请求的传输生命周期；仅凭某一级的发送完成就回收资源，可能早于其他级的完成时刻。

**完成后：** 各级接收名单后，还会在本地 inflight 队列中复查状态，确定本轮实际能清理的请求。

**读图边界：** 名单中的对象可能是更早的请求，与本轮当前 batch 或正在处理的旧 batch 没有固定一一对应关系。

**源码：** [pp:683](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L683) · [prefill:972](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/disaggregation/prefill.py#L972)

### 3.12. 等待前置条件：回收上一轮缓存计数的异步发送

**当前例子：** PP0 · L2：本轮前向 无；本轮旧结果 无。所选步骤的对象：队列级。

**正在等待：** 本地前一步已经走完，但所选操作尚不能开始。当前等待：PP1 · L1 的 L2 write/load ACK：合并计数→逐级传播（缓存 ACK 队列；write=0，load 按完成队列统计，2.30 u 完成）。

**等到后做什么：** PP0 在开始新的缓存事件轮之前，等待 work_list 中上一轮缓存计数发送的 work，并清空这个列表。

**为什么需要：** 限制未完成发送的积累；当下游流水级落后时，通过这里形成反压，使缓存控制消息按轮推进。

**完成后：** 这些前置条件满足后，才从 2.30 u 开始所选操作。然后发布或接收本轮 write/load 完成计数，再处理本地缓存 ACK。

**读图边界：** 这里回收的是缓存计数通信，不是等待所有 KV 回载，也不是请求的 PD 传输释放。 斜线区间是模型推导出的依赖等待，不是额外执行的一遍操作，也不能直接当作实测 CPU 阻塞。

**源码：** [cache:321](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L321) · [cache:3290](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/mem_cache/unified_radix_cache.py#L3290)

### 3.13. GPU 尚未开始 M1 的前向

**当前例子：** PP0 · L4：本轮前向 M1；本轮旧结果 无。所选步骤的对象：M1。

**正在等待：** 本级当前没有执行图中的 GPU 前向，M1 还没有完成提交所需的调度和准备。此区间 CPU 可能在接收消息、检查缓存、选批或等待依赖；不能用一个原因概括整段空档。

**等到后做什么：** PP0 的 GPU 实际计算 M1 在本级持有的层，读取所需前缀 KV，并产生本级新增 KV和传给下一级的中间激活。

**为什么需要：** 同一 batch 必须依次通过 PP0、PP1、PP2 的模型层，才能得到完整 Prefill 结果；各级也因此拥有需要交给 Decode 的本地 KV。

**完成后：** 前向完成后，M1 的中间激活可以传给 PP1，让后续层继续计算。

**读图边界：** 这是本图两段 GPU 前向之间的未计算区间，不表示设备没有任何通信或拷贝活动，也不是源码中的单独“idle”函数。

**源码：** [pp:223](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L223) · [pp:1229](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1229) · [controller:64](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/cache_controller.py#L64)

### 3.14. GPU 等待 M2 的缓存数据

**当前例子：** PP0 · L5：本轮前向 M2；本轮旧结果 无。所选步骤的对象：M2。

**正在等待：** 当前 batch 的 launch 已完成，但本图仍在等本级 H2D 完成，才允许 M2 的 GPU 前向开始。

**等到后做什么：** PP0 的 GPU 实际计算 M2 在本级持有的层，读取所需前缀 KV，并产生本级新增 KV和传给下一级的中间激活。

**为什么需要：** 同一 batch 必须依次通过 PP0、PP1、PP2 的模型层，才能得到完整 Prefill 结果；各级也因此拥有需要交给 Decode 的本地 KV。

**完成后：** 前向完成后，M2 的中间激活可以传给 PP1，让后续层继续计算。

**读图边界：** 源码按层等待 KV 就绪，允许回载与前向重叠。本图把它粗化成整份 H2D 完成后再画 GPU 前向，不能据此断言真实程序有整批屏障。

**源码：** [controller:64](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/cache_controller.py#L64) · [pp:1229](https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/managers/scheduler_pp_mixin.py#L1229)

## 4. 维护与验证

文字维护在 `pages/sglang/pd-prefill-pp-loop/step-guide.js`。运行 `python -B scripts/build_pp_step_guide.py` 同步 iframe 和本文；`build_pp_scenarios.py` 重建场景时也会同步。

`test_pp_step_guide.py` 检查全部操作、等待、空轮、ACK、release、发送者分支和自定义时间场景，防止说明沿用错误的 batch 或固定轮次。`check_pp_source.py --source-root "$SGLANG_SOURCE_ROOT"` 检查固定基线与逐行源码锚点；`SGLANG_SOURCE_ROOT` 由读者设置为官方仓库的检出目录。
