# 教学模型与页面回归测试

这里保存可重复运行的专题检查。`scripts/` 中的全站工具与本目录的专题测试分开维护；目录调整不改变断言范围。

| 目录 | 内容 | 依赖 |
| --- | --- | --- |
| `pages/` | 页面教学模型、队列、导航及浏览器交互测试 | Node.js；`*_browser.cjs` 另需 Playwright、Chromium 与本地静态服务 |
| `pp/` | PP 时序、依赖、自定义时间及步骤说明测试 | Python 标准库、Node.js |
| `pp/` 中的辅助模块 | 共享时序模型、源码基线、生成器与源码核对器 | 多个测试及 CI 共同依赖，不能当作无用脚本删除 |

## 模型与生成检查

在仓库根目录执行，Node.js 需在 PATH 中：

```bash
python3 -B tests/pp/test_pp_timing_model.py
python3 -B tests/pp/build_pp_scenarios.py
python3 -B tests/pp/test_pp_custom_timing.py
python3 -B tests/pp/test_pp_dependency_view.py
python3 -B tests/pp/test_pp_step_guide.py
python3 -B tests/pp/check_pp_source.py

for test_file in tests/pages/test_*.cjs; do
  case "$test_file" in *_browser.cjs) continue ;; esac
  node "$test_file" || exit 1
done
```

场景生成器会同步页面数据与步骤文档，生成后检查 Git diff；源码核对器的 `--source-root` 和模型测试的源码环境变量保持原接口。固定版本与验证边界见各专题正文。

## 浏览器检查

先构建，再在独立终端启动预览服务：

```bash
python3 -B scripts/build_pages.py
python3 -m http.server 8765 --directory .pages-dist
```

在可访问 Playwright / Chromium 的环境运行：

```bash
for test_file in tests/pages/test_*_browser.cjs; do
  node "$test_file" || exit 1
done
```

非默认安装可设置 `PLAYWRIGHT_MODULE`、`CHROMIUM_EXECUTABLE`；Prefill 生命周期检查使用 `PREFILL_PLAYWRIGHT_MODULE`。服务地址覆盖参数仍按各测试文件提供的环境变量使用。

CI 保留原有模型、PP 生成与源码基线检查，并增加通用文档检查。浏览器测试仍为本地运行入口；迁移目录不代表此前未进入 CI 的浏览器测试已经自动执行。基础章节另有 `test_foundations.cjs` 与 `test_foundations_browser.cjs`，检查十三课的机制不变量、场景切换与完整阅读路径。具体复核范围见 [课程维护表](../pages/COURSE_COVERAGE.md)。

`pages/test_concept_focus_browser.cjs` 检查 Chunked Prefill 的分轮累计、CUDA Graph 的提交与计算依赖、性能指标的样本和窗口边界，以及按概念设置不同时长后的拖动、回看、途中调速和旧链接恢复。它使用与其他 Pages 浏览器检查相同的 `PLAYWRIGHT_MODULE`、`CHROMIUM_EXECUTABLE`、`OVERVIEW_TEST_URL`，结束时关闭并核对自有浏览器进程。

并行基础的 `test_parallel_teaching.cjs` 核对 TP 重构、PCP 因果配对与 DCP 的稳定归一化合并；`test_detail_mechanisms_browser.cjs` 覆盖各并行场景的中间态。删除旧播放器后，相关专题浏览器测试通过 `migrated_checks.cjs` 继续检查迁入主动画的原有分支，不因入口整合而删掉这些机制断言。

通信基础的 `test_communication_teaching.cjs` 与 `test_communication_teaching_browser.cjs` 覆盖集合通信、点对点、单边访问及完成条件；浏览器测试记录独立进程与 profile，并在退出时清理。原 `test_communication*.cjs` 仍检查 SGLang 实现课程，不与基础语义混用。

## 调度循环与批次旅程

`node tests/pages/test_loop_journey.cjs` 检查依赖、排空不变量与连续时间下的状态提交边界；`SGLANG_SOURCE_DIR=<checkout> python3 -B tests/pages/test_loop_journey_source.py` 核对固定 Git 对象中的 CPU 调用顺序；PP 顺序另由 `test_parallelism_source.py` 覆盖。浏览器使用 `node tests/pages/test_loop_journey_browser.cjs`，支持 `PLAYWRIGHT_MODULE`、`CHROMIUM_EXECUTABLE`、`OVERVIEW_TEST_URL` 与 `LOOP_SCREENSHOT_DIR`。覆盖三机制的完整播放、暂停、倒拖、倍速、旧 `step` 链接与小数 `t` 链接、依赖门槛、键盘和响应式布局。`test_site_update_browser.cjs` 另验证显式更新后恢复时间、速度与阅读位置。共享导航仍运行 `test_topic_navigation_browser.cjs`。

## 发布版本与阅读位置

`python3 -B tests/pages/test_site_update_build.py` 检查篇章 URL、元数据与资源哈希，随部署 CI 执行。构建并启动本地预览后，运行 `node tests/pages/test_site_update_browser.cjs`；支持上述浏览器环境变量、`OVERVIEW_TEST_URL` 与可选 `UPDATE_SCREENSHOT_DIR`。该检查使用本地双版本响应模拟更新，不触发远程发布，覆盖顶部入口、七种窗口宽度 / 深浅主题、吸顶控件留白、单次导航、场景 / 步骤 / 滚动恢复、资源缺失、离线、发布延迟、会话存储失败及历史恢复；结束时关闭独立浏览器并检查进程和 profile。


## PD＋PP 综合实践

`node tests/pages/test_pp_journey.cjs` 从十个已生成场景核对激活可用、前驱门槛、KV 结束与本地清理分界，以及并行区间和逆向状态重建。`node tests/pages/test_pp_journey_browser.cjs` 验证连续播放、暂停／回拖、脉冲活动区间、观察对象与完整图同步、跨三篇及刷新恢复、场景切换、响应式与减少动态效果。支持 `PLAYWRIGHT_MODULE`、`CHROMIUM_EXECUTABLE`、`OVERVIEW_TEST_URL` 和可选的 `PP_JOURNEY_QA_DIR`；自动清理独立浏览器及 profile。

原有 `tests/pp/test_pp_timing_model.py`、`test_pp_custom_timing.py`、`test_pp_dependency_view.py`、`test_pp_step_guide.py` 和 `check_pp_source.py` 继续检查生成模型、参数求解、完整图几何、步骤说明及固定源码锚点。修改依赖模型或步骤说明后，用 `python3 -B tests/pp/build_pp_scenarios.py --refresh-baseline` 同步场景与嵌入数据，再核对生成 diff；只有明确审查过模型变化后才使用此选项。
