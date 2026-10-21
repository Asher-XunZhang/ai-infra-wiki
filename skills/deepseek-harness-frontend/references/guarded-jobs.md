# 有预算的后台任务与本地事件等待

这套工具替代 `sleep → 读整份日志 → 再 sleep` 的模型往返，并把预算检查接入 Harness 内部的调用边界。适配已核验的 Harness `0.2.1-rc.2`；Python 标准库、Node 和本机官方 CLI 即可。当前运行器使用 POSIX 文件锁，适用于本机 macOS。

## 启动一批

在目标工作区中运行，下列变量由调度方核对；运行目录必须尚不存在，放网站服务目录之外。

```sh
python3 "$skill_dir/scripts/harness_job.py" start \
  --cwd "$PWD" --handoff "$handoff_file" --overlay "$harness_overlay" \
  --run-dir "$new_run_dir"
```

启动器固定使用新 headless 会话，不接受旧 session ID；模型配置仍取已核对的 Flash-1017 / Max overlay，不修改全局配置。先挂载本 Skill 的 `budget_guard.cjs`，收到本次 run ID 与 PID 匹配的 `armed.json` 后才发送任务。钩子加载失败时不会发送模型提示。其他安装布局可通过 `--llm-module` 指定官方 `dsh-llm` 模块路径；不可用时失败，不回退到无预算运行。

同工作区的受管任务使用 OS 文件锁，避免两个受管进程同时写入；锁随 Harness 子进程保留。它不识别外部未接入工具的旧进程，首次迁移前仍须确认旧执行者退出。

## 工具实际执行的预算

默认 40 次模型回复或最近输入 128k 时进入收尾；可通过 `--max-responses`、`--warn-tokens` 调整。计数包含缓存输入。达到阈值后：

1. 在官方 `agent/pre-step` 钩子中追加一次收尾消息，此时上一组工具调用已结束；不发送进度查询、不重复插话。
2. 给至多 8 次额外回复完成当前最小单元、保存文件、清理检查进程，写 `delivery.json`。由模型选择自然交付点，工具核对 run ID、必填字段与仓内路径。
3. 合法检查点提交后，钩子拒绝下一次模型调用，进程退出后运行器报告 `checkpoint`。这是阶段交接，不表示用户全部目标完成；Codex 仍核对真实产物、检查与清理证据。
4. 超过宽限回复数或输入达到 256k，钩子在下一次调用前阻止继续，报告 `budget_blocked`。保留文件和日志，明确待接续，不伪装完成。输入突然跨过 256k 时也会阻止下一次调用。

硬限制检查的是上一次已知输入和已完成回复，并非费用硬上限：不能预知下一次输入精确长度，也不取消正在执行的模型请求、工具或同一次请求内部的重试。通过 `--grace-responses`、`--hard-tokens` 调整时须说明理由。受管批次为单执行者，不允许子代理绕过这份预算另开执行。

手动请求自然收尾：

```sh
python3 "$skill_dir/scripts/harness_job.py" request-checkpoint --run-dir "$run_dir"
```

它只写本地信号，由已加载钩子在下一次调用前读取。不会发送重复模型任务或强杀进程。

## 等待只交给一个本地进程

```sh
python3 "$skill_dir/scripts/harness_job.py" watch --run-dir "$run_dir" --terminal-only
```

只创建一个 watcher，让它在本地等待；无变化时不输出。完成、检查点交接、预算阻断或失败时返回简短 JSON；15 分钟日志无进展则返回 `needs_attention`，不自动判断失败或杀进程。可用 `--stall-seconds` 调整。

如果执行工具先返回 session ID，保持并等待**同一个**执行会话，不再另开 `sleep`、读取日志、重复创建 watcher。按工具允许的等待时长使用 `write_stdin` / 执行会话等待；平台的超时续等仍可能要求模型往返，工具不能承诺自动推送或零唤醒。期间只做独立工作，不解读未变化的状态。

省略 `--terminal-only` 可在首次达到预算时提前返回 `closing`；此时进程未退出，不能启动第二个写入者。之后用 `--terminal-only` 等待真正退出，不反复读取同一预警。可选 `--timeout` 到期静默退出码 124；这不是任务失败。

## 当前旧 headless 进程的迁移

已经运行且未加载钩子的 headless 进程不能热插入预算策略。不要另起进程恢复同一会话或改写会话日志。可立即切换等待方式：

```sh
python3 "$skill_dir/scripts/harness_job.py" watch --run-dir "$old_task_dir" \
  --legacy-log "$old_task_dir/batch1.jsonl" --legacy-exit "$old_task_dir/batch1.exit" \
  --terminal-only
```

这只观察旧进程结束，不为旧进程提供预算强制收束。旧进程退出并保留其文件后，核对修改与未完成项，用短检查点通过 `start` 开新小批次；不要为接入工具回滚当前修改。

## 验证与依据

离线检查：`node scripts/test_budget_guard.cjs`、`python3 -B scripts/test_harness_job.py`（路径相对本 Skill）。覆盖阈值、只注入一次、宽限/硬上限、检查点身份、进程退出边界、启动失败不发送任务与静默等待。真实 CLI 的空 stdin 测试可验证钩子挂载，空任务在模型调用前拒绝；不为测试而付费调用模型。

官方固定版本依据：

- [agent-loop](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/core/agent-loop/README.md)：`agent/pre-step` 与 reject 边界、工具执行与取消语义。
- [headless](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/bundle/headless/README.md)：加载完成后读取任务、一次任务运行到退出。
- [SDK protocol](https://github.com/deepseek-ai/deepseek-harness/blob/687f282e8625c6b8d11ff58211a90524be2e4317/packages/sdk/protocol/README.md)：SDK 的 prompt 是排队输入，没有独立取消方法，不能把它伪装成现有 headless 进程的收尾控制。
