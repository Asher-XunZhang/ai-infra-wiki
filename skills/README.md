# 本仓使用的 Skills

这些规则用于维护学习文档、交互页面与发布流程。按任务选用，不需要每次加载全部 Skill。仓库中的 `AGENTS.md` 和用户当前要求优先。

## 外部开源 Skills

仅列名称，不复制外部实现：

- frontend-design
- web-design-guidelines
- playwright
- gsap-core
- gsap-timeline
- gsap-plugins
- gsap-performance
- skill-creator
- skill-installer
- create-plan

## 本机定制与本仓维护 Skills

| Skill | 用途 | 文件 |
| --- | --- | --- |
| sglang-interactive-learning | 学习路线、机制动画、术语、交接边界、概念动机与证据；包含逐章检查和仓库工作流 | [入口](sglang-interactive-learning/SKILL.md) |
| ai-infra-wiki-pages-publication | 交互课程的构建、检查、隔离发布与线上内容核对 | [入口](ai-infra-wiki-pages-publication/SKILL.md) |
| publish-exact-files-from-dirty-worktree | 只发布明确文件集合并保留原工作区；整站发布使用上一项 | [入口](publish-exact-files-from-dirty-worktree/SKILL.md) |
| codex-token-usage | 读取本地已落盘的任务用量；附统计脚本和可选 hook，默认不启用 hook | [入口](codex-token-usage/SKILL.md) |

定制 Skill 已收录正文及其引用的资源。为便于他人使用，仓库副本去除了本机 Python/Node 绝对路径，使用可用运行时或调用参数，并将两个发布 Skill 的 frontmatter 规范为 Codex 支持的字段；不包含本机日志、使用记录或凭据。

## 使用与维护

可按工具约定从本目录加载，也可以将需要的子目录复制到本机 Skill 目录。外部 Skill 按名称从可信上游安装。本目录不替代依赖安装，也不自动开启发布、hook 或后台任务。

本仓课程规范以 `sglang-interactive-learning/` 为可审查副本。修改本机版本时同步该目录，检查相对引用与 frontmatter；共享前核对没有机器专属目录。使用官方说明解释概念、固定提交核对实现，不能把两类证据混为同一版本事实。
