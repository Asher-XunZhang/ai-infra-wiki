# 第二章的 Archify 图稿

这些 JSON 是课程图的可编辑内容源。Archify 负责节点布局与箭头路径；页面复用生成的 SVG，由本课的步骤、选中状态和同一个脉冲相位控制显示。没有把 Archify Viewer 的全图流动当作引擎执行轨迹，也不加载独立播放器。

## 图的分工

| 学习问题 | 图稿 | 页面中的用途 |
| --- | --- | --- |
| 每个组件负责什么 | `roles-*` | 逻辑职责与对象分组；不以对象分组冒充进程边界 |
| 职责在哪个进程，数据在哪张 GPU | `process-*-tp1` / `process-*-tp2` | 单节点、单 API、PP=DP=1；vLLM 分别采用 uni / 本机 mp。GPU 驻留区与执行进程分开 |
| 谁连接谁，一次消息怎样返回 | `wire-*` / `messages-*` / `*-messages-*` | 通道拓扑与消息时序；管理 RPC、可选 DP 协调分别展开。时序间距不表示耗时 |
| 为什么本轮能够准入 | `admission-*` | token 与 KV 条件的简化流程，保留原来的数值预算图；不是完整源码分支顺序 |
| 请求结束、回撤或取消后怎样收尾 | `lifecycle-*` | 教学状态与条件；资源账本分别说明 KV 引用、缓存保留和执行侧请求槽位 |
| 本轮用哪种图执行 | `graph-*` | 互斥分支；图提前捕获，本轮只计算所选路径 |
| P/D 有哪些角色，KV 怎样可用 | `pd-overview-*` / `pd-*` / `kv-flow-*` | 先看实例，再区分 KV 载荷、控制、引用和准入。vLLM 数据流采用异步接收示例，同步启动路径另行展开 |
| 分层 KV 怎样恢复 | `restore-*` | 独立的可选缓存场景。SGLang 只播放 L3 → L2 → L1 恢复；vLLM 区分 Connector 控制与后端写 GPU |

前缀命中、block/request/KV 槽位映射、执行与采样、CUDA stream 依赖，以及指标端点和样本账本仍由原来的专用图呈现。课程不再叠放同一问题的新旧两套架构图。

## 固定源码与验证边界

- SGLang：公开上游 `sgl-project/sglang`，`v0.5.23`，commit `297e54fd105e974cdf401e5b4d20253a22c16f90`。
- vLLM：公开上游 `vllm-project/vllm`，`v0.33.0rc3`，commit `4ba787ec9959e566d3b587a796dff07673324da5`；本章选择支持 ModelRunner V2 的普通自回归 GPU 路径。
- 读取与图稿复核日期：2026-11-23。读取固定 Git 对象或对应的临时干净检出，不修改原有外部源码工作区。
- 图中的状态、编号、容量及资源条件用于教学。图稿与浏览器检查不代表运行过 GPU、RDMA 或性能实验。

节点的 `sources` 保存固定文件和行号。请求终态、可复用缓存、请求槽位清理与物理内存可重用是不同事件；控制、引用及依赖也不自动代表有字节正在传输。

## 修改和生成

在安装了 Archify 的环境中编辑相应 JSON，按其中的 `diagram_type` 运行完整 `finalize`，并提供匹配固定提交的源码检出。`ARCHIFY_CLI`、`SPEC`、`SOURCE_CHECKOUT` 与 `REVIEW_DIR` 由执行者传入，不把个人路径写入仓库。输出目录中的文件名应与图稿的 `meta.output` 一致；同一目录的生成任务顺序执行。

```bash
node "$ARCHIFY_CLI" finalize architecture "$SPEC" roles-sg.html \
  --repo-root "$SOURCE_CHECKOUT" --quality showcase --out-dir "$REVIEW_DIR" --json
```

检查通过后，从存放这些原生 HTML 及对应检查回执的目录导入：

```bash
python3 -B scripts/build_engine_archify.py --import-dir "$ARCHIFY_OUTPUT_DIR"
python3 -B scripts/build_engine_archify.py --check
python3 -B scripts/build_pages.py
```

导入器只接受源文件、原生 HTML 与通过回执的 SHA-256 全部一致的结果。可发布的 SVG 统一放在顶层 `images/engine-comparison/`；`generated.json` 记录摘要和四项通过状态，`archify-assets.js` 是可离线加载的运行时资源。临时原生 Viewer、浏览器回执和作者目录不进入公开页面。

页面适配只改变配色、字幕换行和装饰性留白，不重新计算节点或箭头路径。活动路径沿生成的同一路径叠加脉冲；本地引用和依赖保持静态。除原生检查外，还要在实际课程页检查节点选择、暂停、切步、TP、源码、语言、窄屏、旧链接与浏览历史。

## 课程页回归

构建后通过 `OVERVIEW_TEST_URL` 指向本地预览，使用当前环境已安装的 Playwright 与 Chromium。浏览器脚本会记录自己的会话、主进程和 profile，并在退出时清理；`ENGINE_TEST_SCREENSHOTS` 可指定仓库外的截图目录。

```bash
node tests/pages/test_engine_archify_browser.cjs
node tests/pages/test_engine_comparison_browser.cjs
node tests/pages/test_engine_boundaries_browser.cjs
node tests/pages/test_cross_chapter_browser.cjs
python3 -B scripts/check_learning_docs.py
```

第一项覆盖 27 张原生图的中英文节点、边界与 1440 / 390 px 布局；后续脚本检查原有 28 步、节点与源码、TP、共享暂停、保留的数值算例和跨章导航。截图需另外实际阅读，不能只凭脚本通过认定布局清楚。
