# Pages 学习课程覆盖与维护

Pages 按五层十二模块组织。页面正文服务初学者；本文件记录课程覆盖与验证方式。资料归档仍使用仓库的稳定技术领域，课程顺序由导航维护，不移动旧源码课程的阶段目录。

## 每个模块的完成要求

每个模块都需要：明确先修和学习目标；可长期阅读的仓内资料入口；逐行为核对的 SGLang 固定源码基线；解释机制的静态、动态或交互图；图旁解释与自测；正确的模块导航；桌面与手机阅读验证。源码事实、教学假设、未做的运行验证必须分别说明。只有引用一篇相关案例，不能算独立入门模块已经补齐。

## 覆盖与补充顺序

这张表是课程补充工作的范围，不把“页面存在”当作全部完成。2026-09-23 开始逐模块复核与补充；其他专题继续保留原有基线。

| 模块 | 已有入口与可视化 | 尚需补充或复核 |
| --- | --- | --- |
| 01 推理系统全景 | [请求生成](sglang/inference-overview/journey.html)，Prefill / Decode 交互 | 复核与其余十一模块的概念衔接，补齐总体系统边界 |
| 02 请求生命周期与运行时架构 | [普通请求运行时](sglang/request-runtime/index.html)，进程图与四路径状态实验 | 已补独立入门课；深入分支连接原有源码章节 |
| 03 模型执行、硬件与算子 | [Transformer](sglang/inference-overview/transformer.html)，逐层执行与 attention 图 | 补 Worker / Runner / 后端 / kernel 的层次与硬件瓶颈直觉 |
| 04 KV Cache 与内存管理 | [KV 入门](sglang/inference-overview/kv-cache.html)，KV 位置与复用 | 补物理页、逻辑映射、共享与驱逐的资源视图 |
| 05 调度与批处理 | [批处理](sglang/inference-overview/scheduling.html)，batch 与调度演示 | 补准入预算、长短请求与回撤边界 |
| 06 并行与执行拓扑 | 现有 PP 案例可交叉阅读 | 建立 TP / PP / DP / EP 的独立拓扑入门；按输入、权重、激活和状态的切分讲解 |
| 07 通信与传输 | PD 数据流可交叉阅读 | 建立 IPC、collective、点对点、KV 传输的独立主线；区分控制消息、激活与 KV；补握手、就绪与失败边界 |
| 08 分离部署与分布式状态交接 | [部署入门](sglang/inference-overview/deployment.html)、[PD 旅程](sglang/pd-prefill-lifecycle/index.html)、[数据流](sglang/pd-dataflow/index.html) | 复核通信模块衔接，保留交接条件、资源持有与历史假设 |
| 09 模型结构与高级生成 | 仓内模型支持与高级生成源码章节 | 补模型状态差异、约束生成、草拟 / 验证 / 接受 / 回退的可视化入门 |
| 10 服务部署与运行治理 | 仓内服务治理源码章节 | 补实例启动、健康检查、路由、背压、超时取消与退出的可视化闭环 |
| 11 性能分析与优化方法 | PP 耗时案例与仓内性能资料 | 补 TTFT / TPOT / 吞吐口径、时间线定位、单变量实验和反例 |
| 12 综合案例与源码实践 | [PP loop](sglang/pd-prefill-pp-loop/quick.html)、[依赖分析](sglang/pd-prefill-pp-loop/index.html)、[源码阅读](sglang/pd-prefill-pp-loop/notes.html) | 将案例入口关联十二模块，增加从假设到源码、验证与边界的练习路线 |

先补 02 以接通单实例主线，之后补 06、07，再补 09、10、11；同时按上述范围完善已有模块。只有十二模块逐项达到完成要求、检查并部署通过，才能把整项工作标为完成。

## 运行时课程的验证

- `node scripts/test_request_runtime.cjs`：检查输出 / KV 位置关系、取消回告与资源释放的先后关系、未派发失败没有后端分配等教学不变量。
- `SGLANG_SOURCE_DIR=<source-checkout> node scripts/test_request_runtime.cjs`：额外读取固定 Git 对象，校验每个交互源码锚点的行号与函数名。不会修改源码仓库。
- `node scripts/test_request_runtime_browser.cjs`：验证四条路径、参数、播放 / 暂停 / 重置、键盘、深浅主题、320 / 390 / 768 / 1440 布局与无 JS 阅读。
- `node scripts/test_topic_navigation_browser.cjs`：首页与全部课程的共享导航和跨模块入口。
- `python3 -B scripts/check_learning_docs.py`、`python3 -B scripts/build_learning_navigation.py --check`、`python3 -B scripts/build_pages.py`：文档引用、共享导航与发布构建。

浏览器测试复用现有 `PLAYWRIGHT_MODULE`、`CHROMIUM_EXECUTABLE`、`OVERVIEW_TEST_URL` 环境变量，默认站点为 `http://127.0.0.1:8765/`。可先以 `python3 -m http.server 8765 --directory .pages-dist` 预览构建输出。网页与源码模型验证不替代 SGLang 实际运行或性能实验。
