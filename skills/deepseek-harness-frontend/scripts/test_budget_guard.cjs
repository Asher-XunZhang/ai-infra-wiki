const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { apply } = require('./budget_guard.cjs');

async function fixture(test) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'guard-test-'));
  const hooks = {};
  const config = {
    cwd: process.cwd(), runId: 'test', warnTokens: 100, hardTokens: 200,
    maxResponses: 2, graceResponses: 2,
    stateFile: path.join(dir, 'state'), armedFile: path.join(dir, 'armed'),
    checkpointFile: path.join(dir, 'delivery'), requestFile: path.join(dir, 'request'),
    createUserMessage: value => value,
  };
  const session = { id: 'root' }, agent = { session };
  let nextCalls = 0;
  const next = async () => { nextCalls++; return { kind: 'enter', messages: [] }; };
  try {
    apply({ on(name, fn) { hooks[name] = fn; } }, config);
    const step = () => hooks['agent/pre-step']({ agent, signal: { aborted: false } }, next);
    const usage = input => hooks['session/event'](session, {
      type: 'assistant/message', data: { usage: {
        inputTokens: 1, cacheReadTokens: input - 1, outputTokens: 2,
      } },
    });
    await test({ dir, config, step, usage, hooks, next, count: () => nextCalls });
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
}

(async () => {
  await fixture(async ({ step, usage, config, count }) => {
    assert.equal((await step()).messages.length, 0);
    usage(100);
    assert.equal((await step()).messages.length, 1);
    assert.equal((await step()).messages.length, 0, 'warning injected only once');
    usage(110); usage(120);
    const before = count();
    assert.equal((await step()).kind, 'reject');
    assert.equal(count(), before, 'hard stop does not invoke next request assembly');
    assert.equal(JSON.parse(fs.readFileSync(config.stateFile)).phase, 'budget_blocked');
  });
  await fixture(async ({ step, usage, config }) => {
    usage(200);
    assert.equal((await step()).kind, 'reject', 'large jump cannot bypass context ceiling');
  });
  await fixture(async ({ step, usage, config }) => {
    usage(20); usage(20);
    assert.equal((await step()).messages.length, 1, 'reply count also triggers closing');
    const checkpoint = { run_id: 'wrong', ready: true, unit: 'token lookup',
      changed_files: ['pages/test.js'], checks: ['unit check passed'],
      remaining: ['next unit'], cleanup_complete: true };
    fs.writeFileSync(config.checkpointFile, JSON.stringify(checkpoint));
    assert.equal((await step()).kind, 'enter', 'stale checkpoint ignored');
    checkpoint.run_id = 'test';
    checkpoint.changed_files = ['../outside'];
    fs.writeFileSync(config.checkpointFile, JSON.stringify(checkpoint));
    assert.equal((await step()).kind, 'enter', 'outside path rejected');
    checkpoint.changed_files = ['pages/test.js'];
    fs.writeFileSync(config.checkpointFile, JSON.stringify(checkpoint));
    assert.equal((await step()).kind, 'reject');
    assert.equal(JSON.parse(fs.readFileSync(config.stateFile)).phase, 'checkpoint_accepted');
    assert.equal(JSON.parse(fs.readFileSync(config.checkpointFile)).unit, 'token lookup');
  });
  await fixture(async ({ step, config, hooks, next }) => {
    await step();
    fs.writeFileSync(config.requestFile, '');
    assert.equal((await step()).messages.length, 1, 'manual checkpoint uses same safe hook');
    const child = { session: { id: 'child' } };
    assert.equal((await hooks['agent/pre-step']({ agent: child,
      signal: { aborted: false } }, next)).kind, 'reject', 'child cannot bypass batch budget');
  });
  console.log('Budget hook: warning once, grace, hard ceiling, checkpoint identity/path, manual request, child boundary PASS.');
})().catch(error => { console.error(error); process.exitCode = 1; });
