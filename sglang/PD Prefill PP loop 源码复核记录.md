# PD Prefill PP loop Pages 源码复核记录

## 当前结论：已对齐本地 279339f113

旧的快速入门、依赖分析与源码说明固定在 `72d5c5bb73`，不能直接称为本地当前版本。本次已将页面、模型、十个场景与源码锚点统一到本地 `279339f113`。下面先记录版本升级，再保留第一次纠正旧图的历史记录；历史部分的旧行号不用于当前页面。

### 阅读基线与范围

| 项目 | 内容 |
| --- | --- |
| 源码目录 | `D:/Codefiles/sglang` |
| 分支 / commit | `main` / `279339f113b79af84f27fd3ac92d0a13bd3f4cbd` |
| 读取时间 | 2026-09-17 |
| 源码工作区 | 读取时干净；只读，未修改 SGLang |
| 对照方式 | `git show` / `git diff` 读取旧提交，对照当前文件；未切换源码分支 |
| Wiki 工作区 | 开始时已有未提交的请求生命周期专题、README 与导航改动；保留已有内容，只协调相关版本交叉引用 |
| 场景 | CUDA、PD Prefill、PP=3、depth=0、普通 FULL attention、UnifiedRadixCache + HiCache cache 模式；五份单请求 batch、M2/M4 有 host hit、容量充足且固定选批 |
| 操作边界 | 静态源码对照、教学模型检查和页面验证；未启动 SGLang，未做 GPU/NCCL/Mooncake 实验 |

“当前”指上述本地提交，不代表远端最新版本。原 `PD 分离下的 PP 源码学习文档.md` 主体仍是历史 `muxi-main` 资料；只将独立第 11.5 节与这些图同步，已在文首明确区分，没有把全仓历史文档改称当前源码。

### 实质差异与处理

| 差异 | 当前逻辑与页面调整 | 当前源码 |
| --- | --- | --- |
| 缓存事件入口外移 | PP loop 在处理旧 chunk 后显式调用 `_process_hicache_events()`，然后才选批；不再藏在 `_get_new_batch_prefill_raw()` 内。B 仍先于 C，空队列也推进 ACK | `scheduler_pp_mixin.py:248–252`；`scheduler.py:3589` |
| 准入分阶段提交 | `_select_prefill_admission()` 和延迟准入通过后才准备 host hit，成功后 `_commit_prefill_admission()` 锁定前缀、加入 can_run_list 并记账。图拆出“提交准入”，避免将选中候选误读为已提交 batch | `schedule_policy.py:1183–1263`、`:1347` |
| 预算与回载接口变化 | allocator 创建 `PrefillBudget`；回载可返回 None，未兑现 host hit 时重检重算预算。图限定容量充足、准入成功，不模拟这些拒绝/重试分支 | `schedule_policy.py:595`；`prefill_budget.py:61`；`unified_radix_cache.py:3243` |
| 成功收尾新增 finish | `release_kv_cache` → `tree_cache.finish(handle, SUCCESS)` → `sender.clear`，再输出/归还 metadata/移出 inflight。handle 是 `(rid, attempt_id)`。默认成功 finish 不取消异步缓存工作，也不新增等待，合入原 release 步骤 | `prefill.py:1025–1078`；`base_prefix_cache.py:45`、`:374` |
| 新增可选路径 | NPU PP=2 批量收发、sampling mask 结果、KV checksum、入口拒绝和乐观重试均不能直接套入本图；已明确 CUDA PP=3 成功路径及关闭的分支 | `scheduler_pp_mixin.py:1037`、`:1090`；`prefill.py:417`、`:1292`、`:1515` |

上述文件均相对 `python/sglang/srt/`。可点击的逐项源码链接和更完整说明见 [交互图说明：基线升级](PD%20Prefill%20PP%20loop%20交互图.md)。

成功收尾的 `finish(SUCCESS)` 是接口语义变化，不是新的 GPU barrier。当前 UnifiedRadixCache 继承 BasePrefixCache 实现，仅非 SUCCESS 才调用 abort 清理。因此释放请求仍不等于清空前缀缓存物理页。

### 核对后仍成立的主线

- 环长仍为 PP size + depth；本地一次 loop 仍是一次 `for mb_id`。本例连续选批、depth=0 时，提交 M3 的轮次处理旧 M1。
- CUDA depth=0 仍先 launch 后处理 output；bootstrap good∩/bad∪、transfer 终态交集和 release 回流仍分开。
- UnifiedRadixCache 仍先合并同步 write/load ACK 数量，再处理本地完成事件；没有改成对所有 PP rank 同时求 MIN 的 barrier。
- `start_loading` 的 start_event 仍承接 schedule_stream 的前序约束；末级旧 D2H 的 copy_stream 仍等待 schedule_stream，保留先前修正的两条必要依赖。
- 真正 KV 可读性仍按 layer event 等待；整份 H2D 完成后才画整段 GPU 前向，仍是教学模型的额外粗化。

### 验证与数据变化

- 完整源码 commit 集中在 `scripts/pp_source_baseline.py`；详细图源码 URL 使用所加载模型的 commit，快速入门继承同一数据版本。
- `python -B scripts/check_pp_source.py --source-root D:/Codefiles/sglang` 检查本地 HEAD、源码工作区、42 个精确模型锚点、页面源码链接范围，以及缓存入口、准入、ACK 和成功清理的 AST 调用顺序。它不导入 SGLang，也不证明完整运行时行为。
- 五项模型回归覆盖十个场景的必要依赖、删除 H2D/D2H/准入边的反例，以及拒绝旧版本数据派生当前入门图。
- 原示例仍有 48 次可见本地 loop 和 15 段 GPU 前向；拆出准入提交后，已解析模型节点从 1604 变为 1619。
- 原 0.48 u 选批占位拆为 0.36 u 预检查和 0.12 u 提交，中间保留回载准备；十个场景主要前向、KV 与最终释放时间保持一致。这是教学占位分配，不是版本间性能没有变化的证据。
- Pages 构建流程增加基线一致性检查；浏览器核验当前版本显示、新准入步骤、场景切换和源码链接。未做 SGLang 设备运行验证。

本地新增的请求生命周期专题本来已采用 `279339f113`。现已统一交叉引用的版本说明，但该专题关闭 HiCache，本图假设 M2/M4 有 host hit，不能照搬彼此的示例 L# 或时间。

---

## 历史：旧基线的第一次复核

本记录是源码分析型复核，检查交互教学页是否与其声明的 SGLang 版本一致。复核发现原页面存在缓存实现混用和必要流依赖遗漏，已在本地修正。以下结论只覆盖声明的成功路径和教学模型，不表示完成了真实设备运行验证。

### 1. 读取基线与范围

| 项目 | 内容 |
| --- | --- |
| 读取日期 | 2026-09-17 |
| 源码目录 | `D:/Codefiles/sglang` |
| 当前分支与 HEAD | `main` / `279339f113b79af84f27fd3ac92d0a13bd3f4cbd` |
| 页面固定基线 | `72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a`；通过 `git show` 读取，未切换源码分支 |
| 源码工作区 | 复核前干净；本次未修改源码 |
| Wiki 工作区 | `D:/Codefiles/ai-infra-wiki`；开始时干净；本次修改页面、教学模型、生成数据、检查和对应说明 |
| 页面范围 | `pages/index.html`、专题的 `quick.html`、`index.html`、`notes.html`，以及十个耗时场景与入门数据 |
| 场景 | CUDA，PD Prefill，PP=3，depth=0，五份单请求 batch；普通 FULL attention，默认 UnifiedRadixCache + HiCache cache 模式，关闭 external linker、L3、staging、投机、中间 chunk 等未展开分支 |
| 验证范围 | 静态源码、模型必要依赖、生成结果、浏览器交互；未导入/启动 SGLang，未做 GPU、NCCL、传输或性能实验 |

### 2. 发现的问题与修正

#### 2.1 将旧 HiRadixCache 的 ACK 路径当成了默认路径

**原问题：** 原图按 `HiRadixCache` 的 PP 分支，分别画 write/load ACK 两次计数同步。固定版本的普通默认工厂实际进入 `UnifiedRadixCache`，其 `check_hicache_events()` 先合并同步 counts，再分别处理本地完成事件。

**源码事实：** [默认缓存选择](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/mem_cache/registry.py#L80)、[合并 counts](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/mem_cache/unified_radix_cache.py#L2741)、[检查入口](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/mem_cache/unified_radix_cache.py#L2957)。

**修正：** 改成一个 `l2_counts` 节点，更新源码链接，声明默认实现和 cache 模式。仍保留正确的“PP0 在级内归约，再沿 PP 传播，各级等待本地 event”解释；这不等于跨全部 PP rank 求 ready-count 的 MIN。

#### 2.2 H2D 可以越过调度流前面的 launch_event

**原问题：** 模型只让 H2D 等提交和上一份 H2D，遗漏了 `start_event` 所在调度流的约束。原示例 PP0 的 M2 H2D 被画成 11.13 u 开始，但上一轮 M1 GPU 到 11.92 u 才完成。

**源码事实：** 非末级在发送激活前排入 [wait_event(launch_event)](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/managers/scheduler_pp_mixin.py#L334)；末级发送 output 前也排入 [wait_event(q_event)](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/managers/scheduler_pp_mixin.py#L991)。下一轮 [start_loading](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/managers/cache_controller.py#L923) 在调度流记录 start_event，[H2D 流等待该事件](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/mem_cache/l2_transfer.py#L74)。

**修正：** 增加上一轮前向到 H2D 的必要因果依赖；重算所有场景。

#### 2.3 末级旧结果 D2H 缺少本轮前向依赖

**原问题：** 模型仅让旧结果 D2H 等待旧 output 到达。原图 PP2 L6 的 M1 D2H 在 43.11 u 开始，而该轮 M3 GPU 到 49.90 u 才完成。

**源码事实：** CUDA、depth=0 的末级先提交当前 output 发送，在调度流上等待当前 q_event；之后旧结果 [copy_stream.wait_stream(schedule_stream)](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/managers/scheduler_pp_mixin.py#L1061) 继承此约束。对象是旧 batch，不代表它能绕过当前 batch 的流事件。

**修正：** 显式增加当前末级前向到旧结果 D2H 的依赖；修正后该 D2H 在 49.90–50.14 u，依赖表可追溯 M3 和旧 M1 output 两条路径。`wait_event` 是设备流约束，不应表述成该调用阻塞 CPU。

#### 2.4 逐层回载的模型简化说明不准确

**原问题：** 文字称“首个需要 KV 的层的门控”，但图实际上等待整份 H2D 完成后才画整段前向，容易让读者把这个整批门控当成源码事实。

**源码事实：** [LayerLoadingEvent.wait](https://github.com/sgl-project/sglang/blob/72d5c5bb73cadd7ffbf5114e5f81e29d36b6c61a/python/sglang/srt/managers/cache_controller.py#L53) 按 layer 等事件；L2 transfer 逐层记录完成，具备回载与计算重叠的机制。

**修正：** 正文、图中节点、依赖标签和两张图顶部均明确标注“等待全部 H2D”是模型粗化，源码无此整批 barrier。没有把教学图扩展成逐层 CUDA/NCCL 仿真，也不从图中估算真实 overlap 收益。

### 3. 已核对成立的主要讲解

下表是固定版本的源码事实；具体 L#、M# 分配和耗时仍是示例假设。

| 页面讲解 | 源码依据 |
| --- | --- |
| loop 是一次本地 `for mb_id` 迭代，不是外层完整一圈 | `scheduler_pp_mixin.py:220–226` |
| 环长为 pp_size + depth；depth=0 时连续提交 M3 的迭代处理 M1 | `init_pp_loop_state():552` 与旧槽索引 `:226` |
| 当前 batch 准备、前向提交、旧结果处理面对不同对象 | 主循环 `:247–320` |
| depth>0 时 output helper 移到 launch 前，不能直接套本图 | `:263–287` |
| CUDA output 环中，末级发送当前结果，非末级转发上轮结果 | `_pp_send_output_to_next_stage():974` |
| bootstrap good 求交、bad 求并；共识回流后才能进入 waiting_queue | `_pp_pd_get_bootstrapped_ids():591`、`process_bootstrapped_queue()`、主循环 `:302–309` |
| transfer Success/Failed 终态求交，release 与 output 回流相互独立 | `_pp_pd_get_prefill_transferred_ids():633`、`_pp_pd_send_consensus_release_ids():681` |
| 本例最终 Prefill 结果处理先进入 inflight，再提交 KV | `disaggregation/prefill.py:805–840` |
| release 名单后仍复查本地终态，释放请求引用/清理 sender；不等于清空缓存物理页 | `prefill.py:918–976`、`mem_cache/common.py:238` |
| 空 batch loop 也能推进缓存完成事件和传输收尾 | `scheduler.py:3704–3720`、PP 主循环中的队列处理 |

### 4. 与本地当前 HEAD 的差别

页面继续固定旧基线，未声称自动跟随 main。对比本地 `279339f113` 后确认：

- PP loop 已显式调用 `_process_hicache_events()`；缓存事件检查从 `_get_new_batch_prefill_raw()` 内移出。宏观顺序仍是选批前推进事件，但旧函数定位不能照搬。
- output helper 新增 NPU PP=2 特殊路径及 sampling-mask 处理；本页 CUDA PP=3、无相关特殊输出的限定不覆盖它们。
- HiCache 接口、混合状态和传输代码还有其他变化；本记录不声称整个较新版本已被此教学模型覆盖。

### 5. 验证与交付边界

- 三项回归测试通过：十个场景的源码必要约束检查，以及分别删除 H2D、D2H 依赖后必须失败的两个反例。
- 十个场景全部重建；每个场景验证 15 段前向、跨级激活依赖、KV 提交与释放顺序及五份 batch 的完整生命周期。原示例有 48 次可见本地迭代、1604 个用于闭合依赖的已解析模型节点。
- 入门与详细模型来自同一生成结果。PP0 L6 的 M3 GPU 区间由旧图 17.21–23.21 u 改为 18.00–24.00 u；到 M4 开始的示意空档改为 10.61 u，正文已同步。
- Pages 构建通过；检查页面本地链接、模型源码链接和生成数据一致性。浏览器检查十个场景切换、batch 生命周期、末级 D2H 依赖与源码链接；未见页面脚本错误。
- 复核阶段仅修改本地 Wiki，未修改 SGLang 源码。用户随后于 2026-09-17 授权合入；发布通过资料仓的 GitHub Pages 工作流完成。

教学模型保留固定请求到达、固定选批、通信配对门控、串行传输假设和整份 H2D 粗化。复核后的结论是“已检查的控制流、对象关系和必要 event 依赖与固定源码一致”，不是“每个时间条块就是实际 GPU trace”。
