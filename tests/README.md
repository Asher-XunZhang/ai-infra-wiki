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

CI 保留原有模型、PP 生成与源码基线检查，并增加通用文档检查。浏览器测试仍为本地运行入口；迁移目录不代表此前未进入 CI 的浏览器测试已经自动执行。具体复核范围见 [课程维护表](../pages/COURSE_COVERAGE.md)。
