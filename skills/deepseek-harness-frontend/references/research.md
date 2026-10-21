# 来源与取舍

读取日期：2026-10-20。检索近期 DeepSeek Harness 文章、1017 发布与社区实践，优先匹配本机 `0.2.1-rc.2`。没有跨平台可靠热度榜；搜索排名、项目 stars 和文章阅读量不是同一指标，因此不把选读列表称为“全网最热门”。

## 第一手依据

| 来源 | 日期或版本 | 本 Skill 采用的事实 |
| --- | --- | --- |
| [DeepSeek 1017 发布说明](https://api-docs.deepseek.com/news/news261017/) | 2026-10-17 | 官方把该模型与新版 Harness 配套推荐；不据此承诺本仓页面的速度或质量提升幅度。 |
| [官方 Harness rc.2 release](https://github.com/deepseek-ai/deepseek-harness/releases/tag/dsh-v0.2.1-rc.2) | 2026-10-17 | 此版本加入 1017 支持；作为本机版本对应基线。 |
| [官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | 1017，读取于本日 | 可查模型与图像输入相关信息；模型能力不等于桌面工具已配置，也不能用基准成绩替代前端验收。 |
| [官方 agent-experience Skill](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/preset/agent-preset/skills/agent-experience/SKILL.md) | 固定 rc.2 commit | 先给必要目标和约束，清楚索引延迟读取的资料，限制无关输出；评估总工作量，不只压缩首轮提示。 |
| [官方使用指南](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/docs/user/guide/index.zh.md) | 同上 | 工作区选择与模型配置属于实际运行环境；任务应在正确 workspace 内执行。 |
| [官方技能发现文档](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/skill/skill-filesystem/README.md) | 同上 | 使用项目 Skill 入口；不能假定 Codex 的安装路径被 Harness 自动发现。 |
| [官方工作区指令文档](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/context/agent-instructions/README.md) | 同上 | AGENTS 的加载有上下文预算及刷新条件；关键任务范围仍写入交接包，变更后明确要求重新读取。 |
| [官方 CLI](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/apps/cli/README.md)、[headless](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/bundle/headless/README.md) | 同上 | 程序化交接可以用 profile、stdin、JSON 事件和会话恢复；桌面登录与 headless 默认认证不同。具体使用边界见 runtime 文档。 |

## 社区选读与采用边界

| 来源 | 时效/可见信号 | 借鉴与不采用的部分 |
| --- | --- | --- |
| [陈大黄：DeepSeek Harness 四种模式详解、PTC 与 Preset 系列](https://chendahuang.com/blog/) | 博客列表显示四种模式条目为 2026-08-15、约 1 万浏览、46 赞；这是网站展示值，未独立审计 X 统计 | 用于发现 Standard/PTC 等实践选择。其“PTC 更快、更省”是经验线索；本工作流按任务判断，并回查当前官方组合，不照搬旧版 Minimal 能力表。 |
| [DeepSeek V4.1 Flash 1017 与 Pi/Harness 的十任务比较](https://www.reddit.com/r/LocalLLaMA/comments/1x8mq6f/deepseek_v41_flash_1017_vs_v41_flash_pi_vs/) | 检索于 2026-10-20，近期发布；作者描述 10 个日常任务、每项运行 3 次 | 第一手小样本实验提示要一起观察成功率、时间、用量与重复稳定性。不是前端教学动画的测评，不据其 High/Low 结果更改本项目 Max，也不使用评论中的工具推断作为配置依据。 |
| [eimoon：从安装配置到 Coding、Web Search 与 Vision](https://blog.eimoon.com/p/deepseek-harness-tutorial/) | 页面日期 2026-09-04，示例版本 0.1.0-rc.7 | 作为旧版上手路线参考，具体操作回查官方新版文档。旧文额外视觉桥接与多模型插件不自动适用于 1017；本次不据此添加 Gemini 或 ModLens。 |
| [独立社区：V4.1 Flash 1017 与 Harness](https://deepseek-harness.io/deepseek-v4-1-flash-1017) | 2026-10-17 更新主题 | 用于发现发布说明链接，不作为官方站点。其模型重定向、价格或 API 配置陈述均须另查官方，本 Skill 不纳入这些未经验证的结论。 |

社区材料用于找问题和候选方法；可执行命令、认证行为、版本能力以官方固定资料及本机核验为准。没有证据证明某段“万能提示词”或某个预设能保证最优结果。

## 本项目的具体决策

以下是基于用户要求、现有课程经验与上面的资料做出的工作流选择，不冒称官方推荐：

1. 模型与 Max 固定遵从用户要求，优化资料组织和反馈闭环，不以降档换速度。
2. Codex 把源码事实、设计动机、最小算例及边界交清；Harness 选择适合机制的图形并负责前端实现。
3. 一个会话只处理有界的任务阶段；跨阶段用检查点开新会话，同阶段短返修先检查上下文预算。少量入口文件与按需加载替代整仓倾倒。
4. 保持原工作区和源码证据，修改范围明确；实现阶段只有一个写入者负责同一文件。
5. 构建与测试后仍实际看图、操作中途状态。缺陷集中反馈，依据阶段和预算选择恢复或新会话，独立验收通过才向用户呈现。
6. 性能按可取得的耗时、用量、返工次数及最终验收结果讨论。没有测量时只称工作流改进，不宣称已证明最快、最省或效果最佳。

## 长任务上下文与成本复核

2026-10-20 补查。以下机制来自官方文档或工程博客；具体预算阈值是本项目的可调整起点，不是厂商保证的最优值。

| 来源 | 采用的机制与边界 |
| --- | --- |
| [Anthropic：Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | 按需读取路径指向的资料；用外部进度记录跨上下文保留决策和未解决问题。压缩不可丢掉关键条件，也不能把完整历史换个文件再全部读回。 |
| [Anthropic：Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) | 长任务拆成可验收的增量，每次会话读取进度并交付可继续的状态。借鉴其交接方式，不照搬自动提交、并行代理或特定产品命令。 |
| [DeepSeek：compaction-basic](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/compaction/compaction-basic/README.md) | 自动压缩按 context window、输出余量与阈值触发，会调用摘要模型。它不等于低成本任务预算；不等到窗口将满才拆任务，也不把普通 headless 输入 `/compact` 当作已执行 UI 命令。 |
| [DeepSeek：token-meter](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/llm/token-meter/README.md) | 本地 tokenUsage 与 contextPressure 可用于观察累计用量和上下文压力；启发式分解不是账单。读取这些状态不需要额外模型请求。 |
| [DeepSeek：价格说明](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/) | 缓存命中输入仍有价格，需区分缓存输入、未缓存输入和输出。实际费用以账户账单、当时模型与时段为准，不用累计 token 数直接推断余额。 |

据此移除“同一页面始终沿用原会话”的规则，采用 [上下文与调用预算](context-budget.md)。不把私人账单、会话原文或账户标识写进公共 Skill；不声称离线流程验证已经证明实际节省比例。

## 一次交接与简短目标：证据和采用方式

2026-10-20 核对近期原始报告，采用“给清楚目标、减少过程指挥”，不将其解释成提示词越短越好。

| 第一手来源 | 实际支持与限制 |
| --- | --- |
| [DeepSeek 10 月 17 日发布说明](https://api-docs.deepseek.com/news/news261017/) | 官方以 Wish-Driven Programming 为主题，介绍通过自反馈强化学习改善端到端项目开发，并展示游戏与 3D 案例；没有提供同一任务“短指令 vs 长指令”的受控实验。 |
| [EmpirioLabs 10 月 18 日双任务实测](https://empiriolabs.ai/blog/deepseek-v4-1-flash-1017-vs-deepseek-v4-1-flash) | 作者给出提示词、设置、首版渲染及用量：水族箱与夜景均一次返回可运行动画；1017 一个任务更省、另一个更费。是默认思考预算下的两项 HTML 生成测试，不等同于本机 Harness / Max 的教学站重构。 |
| [作者发布的 Max / zero-shot 游戏对比](https://www.reddit.com/r/DeepSeek/comments/1x8exxe/v41_flash_1017_is_a_massive_jump_same_zeroshot/) | 作者注明同一个长提示词、Max、各跑一次，评论说明使用 Harness 桌面版。这支持自主首版能力的个案观察，不支持“一句话胜过详细指令”的因果结论，也不是完整功能验收。 |
| [官方 agent-experience](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/preset/agent-preset/skills/agent-experience/SKILL.md) | 明确从最小上下文开始、细节按需读取、关键约束可见、同一事实只说一次，并评估总工作量。这是减少重复提示与过程干预的直接依据。 |

本项目将普通交接缩成目标、必要边界、资料入口；复杂事实按需读取，由 Harness 主导实现和自测，Codex 核验并呈现。一次交接不等于一次模型调用；保留上下文检查、用户要求的方案审批和功能验证。是否更好看、更省，需要在后续真实任务中观察首版质量、返修与总用量，不把短提示本身当成成功指标。
