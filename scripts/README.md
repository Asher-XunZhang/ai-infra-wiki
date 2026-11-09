# 仓库通用维护工具

本目录只放跨专题使用的全站维护入口。可重复执行的专题回归测试集中在 [tests](../tests/README.md)，测试所依赖的模型、固定基线与生成工具随测试一起维护。

| 文件 | 复用范围 | 用法 |
| --- | --- | --- |
| [check_learning_docs.py](check_learning_docs.py) | 全仓本地链接、锚点、图片目录和学习导航覆盖 | `python3 -B scripts/check_learning_docs.py` |
| [build_foundations.py](build_foundations.py) | 推理基础十三课正文、先修、自测与来源 | `python3 -B scripts/build_foundations.py`；检查用 `--check` |
| [build_learning_navigation.py](build_learning_navigation.py) | 全部 Pages 共用的课程映射、侧栏、面包屑与翻页 | `python3 -B scripts/build_learning_navigation.py`；检查用 `--check` |
| [build_pages.py](build_pages.py) | 全站发布构建与内容哈希资源名 | `python3 -B scripts/build_pages.py`；输出为 `.pages-dist/` |

四个入口只依赖 Python 标准库；文档检查另需 Git。构建会自动检查基础课程正文与共享导航是否同步。

## 维护边界

全站工具归本目录；教学模型与页面行为的回归断言归 `tests/`；浏览器运行代码留在对应页面目录。测试名称带有专题，不代表它是一次性文件；只要还用于防止回归或被 CI 调用，就保留其检查与依赖。

单次内容制作、已无调用的一次性辅助程序可清理。图片、已发布的数据与来源记录保留；需要历史复现时指向固定 Git 提交。本次仅删除独立的 Transformer 配图生成器，没有删除课程回归断言。

调度循环课使用 `pages/sglang/loop-journey/` 的事件模型；第二层四模块与所有页面编号统一在 `build_learning_navigation.py` 维护，旧页面 URL 保持稳定。

## 页面更新与缓存

构建输出的篇章链接保持稳定，删除旧 `v`，保留场景参数与锚点。CSS / JavaScript 仍按内容生成哈希文件名；`site-release.json` 与每页的 `site-release` 元数据使用同一发布标识。更新引导直接嵌入 HTML，确保旧哈希资源被移除后仍有更新入口。

页面初次加载、回到前台及从浏览器历史恢复时检查版本，自动检查间隔至少一分钟；全站顶部导航提供“检查更新”。发现新版后，导航显示“发现新版”标记和“更新并继续”，正文提示按正常布局占位，不使用遮挡内容的浮层；关闭正文提示后顶部入口仍然可见。由读者点击“更新并继续”，先验证目标 HTML 已发布，再用一次性会话记录保存课程状态、展开项与滚动锚点。临时 `_refresh` 仅用于请求新文档，载入后清除；不会留在分享链接中。离线、发布未同步或无法保存状态时保留当前页，不循环刷新。

独立课程通过 `window.SiteUpdate.register(name, capture)` 保存自己的进度，用 `take(name)` 消费一次恢复记录；共享播放器沿用已有 URL 状态。恢复后保持暂停。协议覆盖当前状态格式，不承诺已删除的小节或跨不兼容模型版本仍能原样恢复。原先尚未包含更新引导的旧页面，需要先刷新一次加载本机制。
