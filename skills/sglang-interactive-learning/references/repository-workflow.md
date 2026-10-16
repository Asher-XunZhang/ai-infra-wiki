# ai-infra-wiki 工作流

此文件是定位入口，不是永久固定的仓库清单。执行时先核对文件和命令仍存在，并遵守当前 `AGENTS.md`。不从本机路径推导公开文档链接。

## 读哪些文件

| 目的 | 优先入口 |
| --- | --- |
| 课程覆盖、学习目标与验收 | `pages/COURSE_COVERAGE.md`、`pages/MOTION_REVIEW.md`、根 README、对应主题 README |
| 共享交互与样式 | `pages/site.js`、`pages/site.css`、`pages/diagrams.js` |
| 基础课内容与生成 | `pages/foundations/courses.json`、`model.js`、`extensions.json` 与 `scripts/build_foundations.py` |
| 连贯机制图 | `pages/motion/player.js` 共享控制；`teaching-scenes.js` / `teaching-visuals.js`、`engine-scenes.js` / `engine-visuals.js`、`batching.js` 按机制渲染；先核对当前使用点 |
| 专题状态与渲染 | `pages/sglang/<topic>/` 下的 HTML、`model.js` 和专题 JS/CSS |
| 实现课概念动机与外部依据 | `pages/sglang/design-notes.json`；每课保留自己的源码基线，不能把所有页面改为同一提交 |
| 导航单一来源 | `scripts/build_learning_navigation.py`，先读生成逻辑再改入口 |
| 构建 | `scripts/build_pages.py`，当前产物为 `.pages-dist` |
| 模型、浏览器与源码检查 | `tests/README.md`、`tests/pages/`、`tests/pp/`、Pages workflow |
| 写作节奏 | `sglang/分离部署/PD 分离下的 PP 源码学习文档.md`；历史分支证据不能移作上游结论 |

先用 `git status --short` 记录原有改动。修改共享层前搜索使用点，不用重建页面覆盖已有模型与源码锚点。站点导航、Markdown 文档索引和主题归档是相关但不同的层次，应保持相互可达。

学习路线按先修关系安排，而不是按文件夹顺序展示。保留已有深读资料入口，通用原理与 SGLang 特定实现分别标明。稳定标题不使用“新增”“刚上线”等临时标签。

本仓已确认的模型执行教学用例使用 DeepSeek-V3，后续若改用 V4 应另核对其源码基线；不要默默换回 Llama。tokenizer 样例按各自已核实的来源保留，不能为了统一模型名字编造 token ID。普通 MHA 分页示例与 MLA 历史布局分别声明适用范围。

## 验证入口

以下命令从仓库根目录执行；选择与实际改动匹配的范围：

```bash
python3 -B scripts/check_learning_docs.py
python3 -B scripts/build_learning_navigation.py --check
python3 -B scripts/build_pages.py
node tests/pages/test_scheduling.cjs
```

最后一条只是调度课的模型检查示例，其他课程使用自己的测试。改共享时间线可复用 `tests/pages/test_mechanism_motion_browser.cjs`；改专用图可复用 `test_teaching_visuals_browser.cjs`，按当前测试实际覆盖补充关键边界。浏览器测试外仍需看图，不能把节点存在或帧数统计当作初学者验收。涉及源码核对时先阅读脚本，按它接受的参数提供检出目录；当前有 `SGLANG_SOURCE_DIR` 与 `SGLANG_SOURCE_ROOT` 等不同接口，不能统一猜测。

预览构建产物，端口冲突时选空闲端口并同步测试 URL：

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory .pages-dist
node tests/pages/test_scheduling_browser.cjs
```

浏览器脚本需要 Playwright 模块及 Chromium；CLI Skill 可用于人工式检查，但 CLI 可运行不等于仓库脚本能解析模块。按脚本支持的 `PLAYWRIGHT_MODULE`、`CHROMIUM_EXECUTABLE`、`OVERVIEW_TEST_URL` 或专题变量配置，先查脚本，不硬编码机器路径进仓库。仅供本轮自动测试的临时服务验证后关闭；用户还要本地审阅时，保留独立于短时工具会话的预览服务，确认原链接、脚本资源和交互可用后交付，不在收尾时关掉用户要看的入口。

重点检查 320、390、768、1440 宽度以及已有主题和减少动态效果支持；只对受影响范围执行。对于大型图，允许局部滚动，但控件、正文和关键状态不能依赖整页横向溢出或仅悬停交互。

如修改生成器，运行对应生成一致性检查；如移动正文或图片，全仓更新相关引用。文档图片统一放在仓库根部 `images/<topic>/`。构建和测试结束后再次检查 diff/status，区分原有变更和本次变更。
