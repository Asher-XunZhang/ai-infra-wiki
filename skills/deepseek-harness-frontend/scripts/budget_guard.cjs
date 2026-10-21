/* Harness rc.2 public agent/pre-step extension. No model calls or process signals. */
const fs = require('node:fs');
const path = require('node:path');

function atomic(file, value) {
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(value, null, 2));
  fs.renameSync(tmp, file);
}

function validCheckpoint(value, config) {
  if (!value || value.run_id !== config.runId || value.ready !== true) return false;
  if (typeof value.unit !== 'string' || !value.unit.trim()) return false;
  if (!Array.isArray(value.changed_files) || !Array.isArray(value.remaining) ||
      !Array.isArray(value.checks) || value.cleanup_complete !== true) return false;
  if (![...value.remaining, ...value.checks].every(x => typeof x === 'string')) return false;
  return value.changed_files.every(p => typeof p === 'string' && p.length &&
    !path.isAbsolute(p) && !path.relative(config.cwd, path.resolve(config.cwd, p)).startsWith('..'));
}

function apply(ctx, config) {
  for (const key of ['warnTokens', 'hardTokens', 'maxResponses', 'graceResponses']) {
    if (!Number.isSafeInteger(config[key]) || config[key] < 1) throw new Error(`Invalid ${key}`);
  }
  if (config.warnTokens >= config.hardTokens) throw new Error('Invalid context thresholds');
  if (path.resolve(config.cwd) !== process.cwd()) throw new Error('Guard workspace mismatch');
  const createUserMessage = config.createUserMessage ||
    require(config.llmModule).createUserMessage;
  const states = new Map();
  let rootId;
  const get = session => {
    if (!states.has(session.id)) states.set(session.id, {
      session_id: session.id, responses: 0, input_tokens: 0, output_tokens: 0,
      latest_input: 0, warning_at: null, phase: 'running',
    });
    return states.get(session.id);
  };
  const save = state => atomic(config.stateFile, {
    ...state, run_id: config.runId, updated_at: new Date().toISOString(),
  });
  ctx.on('session/event', (session, event) => {
    if (event.type !== 'assistant/message' || !event.data.usage) return;
    const s = get(session), u = event.data.usage;
    const input = (u.inputTokens || 0) + (u.cacheReadTokens || 0) + (u.cacheWriteTokens || 0);
    if (!Number.isFinite(input) || input < 0) throw new Error('Invalid provider usage');
    s.responses += 1;
    s.latest_input = input;
    s.input_tokens += input;
    s.output_tokens += u.outputTokens || 0;
    save(s);
  });
  ctx.on('agent/pre-step', async ({ agent, signal }, next) => {
    // One guarded batch has one writer. Child work must not evade the job budget.
    rootId ??= agent.session.id;
    if (agent.session.id !== rootId) return { kind: 'reject' };
    const s = get(agent.session);
    if (signal.aborted) return next();
    if (s.warning_at !== null) {
      let checkpoint;
      try { checkpoint = JSON.parse(fs.readFileSync(config.checkpointFile, 'utf8')); }
      catch (e) { if (e.code !== 'ENOENT' && !(e instanceof SyntaxError)) throw e; }
      if (validCheckpoint(checkpoint, config)) {
        s.phase = 'checkpoint_accepted';
        s.unit = checkpoint.unit;
        save(s);
        return { kind: 'reject' };
      }
      if (s.responses >= s.warning_at + config.graceResponses) {
        s.phase = 'budget_blocked';
        s.reason = 'No valid delivery checkpoint before the bounded grace expired';
        save(s);
        return { kind: 'reject' };
      }
    }
    if (s.latest_input >= config.hardTokens) {
      s.phase = 'budget_blocked';
      s.reason = 'Context ceiling reached; no further model request admitted';
      save(s);
      return { kind: 'reject' };
    }
    const decision = await next();
    if (decision.kind === 'reject') return decision;
    if (s.warning_at === null && (s.responses >= config.maxResponses ||
        s.latest_input >= config.warnTokens || fs.existsSync(config.requestFile))) {
      s.warning_at = s.responses;
      s.phase = 'closing';
      save(s);
      const schema = {
        run_id: config.runId, ready: true, unit: '当前已完成的最小单元',
        changed_files: ['仓内相对路径'], checks: ['实际检查及结果，未测须说明'],
        remaining: ['下一批待办'], cleanup_complete: true,
      };
      const message = createUserMessage({
        source: { kind: 'user' },
        content: [{ type: 'text', text:
          `执行预算已到，停止扩展范围。完成当前最小单元，保存已有修改，清理本任务后台检查进程；` +
          `在最多 ${config.graceResponses} 次模型回复内将真实检查点写入 ${config.checkpointFile}。` +
          `格式：${JSON.stringify(schema)}。写好后直接结束，不继续下一单元。` +
          `cleanup_complete 仅在实际清理完成后填 true；未完成的工作如实写 remaining。` }],
      });
      return { ...decision, messages: [...decision.messages, message] };
    }
    return decision;
  }, { prepend: true });
  // Last action: runner does not send a task until all hooks above are registered.
  atomic(config.armedFile, { run_id: config.runId, pid: process.pid });
}

module.exports = { name: 'ai-wiki-budget-guard', apply, validCheckpoint };
