# 仓库通用维护工具

本目录只放跨专题使用的全站维护入口。可重复执行的专题回归测试集中在 [tests](../tests/README.md)，测试所依赖的模型、固定基线与生成工具随测试一起维护。

| 文件 | 复用范围 | 用法 |
| --- | --- | --- |
| [check_learning_docs.py](check_learning_docs.py) | 全仓本地链接、锚点、图片目录和学习导航覆盖 | `python3 -B scripts/check_learning_docs.py` |
| [build_learning_navigation.py](build_learning_navigation.py) | 全部 Pages 共用的课程映射、侧栏、面包屑与翻页 | `python3 -B scripts/build_learning_navigation.py`；检查用 `--check` |
| [build_pages.py](build_pages.py) | 全站发布构建与内容哈希资源名 | `python3 -B scripts/build_pages.py`；输出为 `.pages-dist/` |

三个入口只依赖 Python 标准库；文档检查另需 Git。构建会自动检查共享导航是否同步。

## 维护边界

全站工具归本目录；教学模型与页面行为的回归断言归 `tests/`；浏览器运行代码留在对应页面目录。测试名称带有专题，不代表它是一次性文件；只要还用于防止回归或被 CI 调用，就保留其检查与依赖。

单次内容制作、已无调用的一次性辅助程序可清理。图片、已发布的数据与来源记录保留；需要历史复现时指向固定 Git 提交。本次仅删除独立的 Transformer 配图生成器，没有删除课程回归断言。
