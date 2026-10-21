# 本机接入与配置核验

核验日期：2026-10-20。桌面安装实测 `0.2.1-rc.2`；官方对应 tag 为 `dsh-v0.2.1-rc.2`，commit `687f282e8625c6b8d11ff58211a90524be2e4317`。这些是检查时的记录，后续执行重新确认。

## 已验证入口

- 官方桌面应用名称：`DeepSeek Harness`。
- 安装自带 CLI 相对应用包路径：`Contents/Resources/runtime/cli/bin/dsh`。
- 用户命令目录已配置 `dsh` 链接；执行 `command -v dsh`、`dsh --version` 和 `dsh --help` 确认解析到安装自带版本。移动应用或升级后若链接失效，重新定位；不要为了补 PATH 再装一套不同版本的 npm 包。
- UI 检查过“设置 → 模型”的 DeepSeek 账号已登录；会话模型菜单提供 Flash 的 `1017` 标签，推理档位包含 `Max`。账户状态不写入仓库中的个人信息或凭据。

用户明确反馈桌面弹窗影响开发，因此后续默认后台执行；不激活 Harness 窗口、不打开有头自动化浏览器，不复用用户浏览器。若用户明确要求桌面操作，再通过受支持的原生应用控制工具读取实时状态并操作；不能覆盖用户草稿或正在运行的会话。

## CLI 与桌面账号的区别

官方 [CLI 文档](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/apps/cli/README.md) 定义 `dsh --profile headless`、`sdk` 和 `acp`；它们是不同 profile。CLI 不允许启动保留给 Electron 的 `desktop` profile。

官方 [headless 文档](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/bundle/headless/README.md) 明确：默认 headless 禁用托管凭据、authorization 和 DeepSeek account 消费方，GUI 保存的登录不会自动用于该模式。该文档也提供恢复已有账号认证的 overlay；本机已使用此方式成功完成最小调用，返回 `READY`、`turn_end=completed`、退出码 0，随后启动实际后台审核任务。

本机 overlay 位于用户 Harness 配置下的 `overlays/ai-infra-wiki-flash-max.yml`，不含密钥，内容如下。它让官方组件读取已有登录，不提取、复制或打印凭据，也不改变工具审批或文件访问策略。

```yaml
- id: credentials
  disabled: false
- id: authorization
  disabled: false
- id: deepseek-account
  disabled: false
- id: agent-default-model
  config:
    provider: deepseek-account
    model: deepseek-flash
    reasoningEffort: max
```

`deepseek-flash` 是路由别名，版本仍需结合账户目录与官方发布核对，不能当作永久固定的 1017 标识。升级后重新验证；不猜不存在的 `--model` 或 `--reasoning-effort` 参数。只输出允许的模型配置字段，不将完整配置或日志中的敏感内容打印到对话。

新实施批次从本仓 cwd 使用 [受管启动器](guarded-jobs.md)，`harness_overlay`、`handoff_file`、`skill_dir` 和尚不存在的 `new_run_dir` 均由调度方确认：

```sh
python3 "$skill_dir/scripts/harness_job.py" start \
  --cwd "$PWD" --overlay "$harness_overlay" \
  --handoff "$handoff_file" --run-dir "$new_run_dir"
```

运行器保存 PID、会话事件、stderr 与退出结果，日志放网站之外；使用 watcher 等待，不逐回合读取长日志。受管启动器只创建新会话，以短检查点接续文件；旧会话的只读诊断仍可用 [预算检查](context-budget.md)，不以检查通过代替运行中钩子。核对相同 cwd、preset 和唯一执行者，不让 CLI 和 UI 同时驱动同一会话。验收需要进程退出、正确终态与真实产物，不能只看 `final` 或把 `checkpoint` 当作全部完成。首次启动 profile 可能初始化配置，不将它描述为完全无副作用的查询。

## 让 Harness 读取教学 Skill

官方 [skill-filesystem](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/skill/skill-filesystem/README.md) 支持项目 `.dsh/skills`、`.agents/skills` 和用户 Skill 根目录，默认不扫描 Codex 专用的 Skill 目录。项目 Skill 只发现根下一级的 `name/SKILL.md`。

本仓通过 `.dsh/skills/` 下的相对链接指向仓内可审查 Skill，正文仍由 `skills/` 单处维护；不复制外部开源 Skill 到本仓，也不把本机绝对链接提交进去。在正确工作区检查 Skill 目录或让实现者明确读取入口，不能假定发现成功。

本机的用户级 Harness Skill 目录还链接了已安装的 `frontend-design`、`web-design-guidelines`、`playwright`、`gsap-core`、`gsap-timeline`、`gsap-plugins` 和 `gsap-performance`。这些是本机接入配置，不随仓库分发；换机器时按来源安装并核对当前会话是否可发现。Skill 内若使用某工具专有的环境变量或路径，应先核实实际可用运行时，不能把文档已加载当成依赖已可运行。

安装附带的 [agent-preset 文档](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/preset/agent-preset/README.md) 提及标准模式工具结果的 8192 字符裁剪阈值。让新 Skill 入口保持简短；大型教学规范按需、分段读取，检查截断提示。文件实际存在、目录发现和完整读到正文是三个不同检查。

本次已通过桌面会话做只读交接验证：实际 cwd 为本仓，两项项目 Skill 均由 Skill 工具完整加载，实现者正确识别分工与本地修改边界。调度方从界面核对 `Flash / 1017 / Max`；实现者没有把 `deepseek-flash` 路由名当作版本自证。这次验证确认交接可用，未执行前端实现、渲染验收或性能对照。
