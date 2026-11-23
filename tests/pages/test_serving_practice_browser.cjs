const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), os = require('node:os');
const cp = require('node:child_process'), zlib = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.OVERVIEW_TEST_URL || 'http://127.0.0.1:8765/';
const output = process.env.SERVING_LABS_QA_DIR;
const slugs = ['memory-budget', 'hicache-loadback', 'deployment-lab', 'benchmark-lab', 'trace-lab'];
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
function processes() {
  return cp.execFileSync('ps', ['-axo', 'pid=,ppid=,stat=,command='], { encoding: 'utf8' }).trim().split('\n')
    .map(line => { const m = line.trim().match(/^(\d+)\s+(\d+)\s+(\S+)\s+(.*)$/); return m && { pid: +m[1], parent: +m[2], stat: m[3], command: m[4] }; }).filter(Boolean);
}
function descendants(pid) {
  const all = processes(), ids = new Set([pid]); let changed = true;
  while (changed) { changed = false; for (const p of all) if (ids.has(p.parent) && !ids.has(p.pid)) { ids.add(p.pid); changed = true; } }
  return all.filter(p => ids.has(p.pid));
}
function alive(targets) {
  const current = processes();
  return targets.filter(p => current.some(q => p.pid === q.pid && p.command === q.command && !q.stat.startsWith('Z')));
}

(async () => {
  const server = await chromium.launchServer({ headless: true,
    ...(process.env.CHROMIUM_EXECUTABLE ? { executablePath: process.env.CHROMIUM_EXECUTABLE } : {}) });
  const owned = server.process();
  const info = { session: 'serving-practice-review-' + process.pid, controller: process.pid, browser: owned.pid,
    profile: owned.spawnargs.find(a => a.startsWith('--user-data-dir=')) };
  console.log(JSON.stringify(info));
  if (output) { fs.mkdirSync(output, { recursive: true }); fs.writeFileSync(path.join(output, 'session.json'), JSON.stringify(info, null, 2)); }
  let browser;
  const seen = new Map();
  const remember = () => { for (const p of descendants(owned.pid)) seen.set(p.pid, p); };
  remember();
  let targets = [];
  try {
    browser = await chromium.connect(server.wsEndpoint());
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    page.setDefaultTimeout(20000);
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    async function go(slug) {
      await page.goto(base + 'sglang/' + slug + '/?learn=labs');
      await page.locator('[data-serving-lab][data-ready=true]').waitFor();
      await page.waitForFunction(() => !!document.querySelector('#sl-svg title'));
      assert.equal(await page.locator('.sl-transport [name=speed]').count(), 1);
      assert.equal(await page.locator('#lab .animation-speed').count(), 1, 'one effective speed control');
      remember();
    }
    const time = v => page.locator('[name=seek]').evaluate((e, v) => { e.value = v; e.dispatchEvent(new Event('input', { bubbles: true })); }, v);
    const clock = async () => +await page.locator('#lab').getAttribute('data-time');
    const shot = async name => { if (output) await page.locator('#lab').screenshot({ path: path.join(output, name + '.png') }); };
    async function importTrace(events, name = 'window-test.json') {
      await page.locator('#sl-trace-file').setInputFiles({ name, mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({ traceEvents: events })) });
      await page.waitForFunction(n => document.querySelector('#sl-record-label').textContent.includes(n), name);
    }
    await go('deployment-lab');
    await time(13.1);
    assert.match(await page.locator('#sl-stage').innerText(), /看见首个文本/);
    assert.match(await page.locator('#sl-stage').innerText(), /尚未确认/);
    await time(14.6);
    assert.match(await page.locator('#sl-stage').innerText(), /已收到完整响应/);
    await page.locator('[name=scenario]').selectOption('cancel');
    await time(13.8);
    assert.match(await page.locator('#sl-stage').innerText(), /通知已返回，设备可能仍在途/);
    await page.locator('[data-depth-choice=modules]').click();
    assert.equal(await clock(), 13.8);
    await page.locator('[data-source-toggle]').click();
    assert.equal(await clock(), 13.8);
    await shot('deployment-cancel');
    await time(12.8);
    const phase = () => page.locator('#sl-svg [data-flow=return-visible] path').getAttribute('stroke-dashoffset');
    const a = await phase(); await page.waitForTimeout(100); assert.equal(await phase(), a, 'pulse freezes while paused');
    await time(12.9); assert.notEqual(await phase(), a, 'pulse follows shared teaching clock');
    await time(13.1); assert.equal(await page.locator('#sl-svg [data-flow=return-visible]').getAttribute('data-active'), 'false');
    await time(15.2); assert.match(await page.locator('#sl-stage').innerText(), /服务仍在运行/);
    await page.locator('[name=scenario]').selectOption('oom'); await time(7.1);
    assert.match(await page.locator('#sl-stage').innerText(), /初始化失败，没有进入生成/);
    await page.locator('[name=scenario]').selectOption('probe'); await time(2.5);
    assert.match(await page.locator('#sl-stage').innerText(), /过早探测/);

    await go('benchmark-lab');
    await page.locator('[name=requestFocus]').selectOption('2');
    assert.match(await page.locator('[data-benchmark-reading]').innerText(), /固定查看 R3/);
    assert.match(await page.locator('[data-benchmark-reading]').innerText(), /2.5 u/);
    assert.match(await page.locator('[data-benchmark-reading]').innerText(), /10.5 u/);
    await page.locator('#sl-results details').evaluate(e => e.open = true);
    await page.locator('[data-action=play]').click(); await page.waitForTimeout(200);
    assert.equal(await page.locator('[name=requestFocus]').inputValue(), '2');
    assert.equal(await page.locator('#sl-results details').getAttribute('open'), '', 'playback keeps expanded records');
    await page.locator('[data-depth-choice=modules]').click();
    assert.equal(await page.locator('#lab').getAttribute('data-playing'), 'true');
    await page.locator('[data-action=play]').click();
    await time(7); assert.equal(await page.locator('[name=requestFocus]').inputValue(), '2');
    await page.locator('[name=slaTTFT]').selectOption('12'); assert.equal(await clock(), 7);
    assert.equal(await page.locator('[name=requestFocus]').inputValue(), '2');
    await page.locator('[name=slaTTFT]').selectOption('6');
    await shot('benchmark-request-ledger');
    const stats = async () => page.locator('#sl-results').innerText();
    assert.match(await stats(), /10.5 u/); assert.match(await stats(), /33.5 u/);
    await page.locator('[name=failure]').check();
    assert.match(await stats(), /5 \/ 6/); assert.match(await stats(), /10.1 u/); assert.match(await stats(), /33.5 u/);
    await page.locator('[name=requestFocus]').selectOption('2');
    assert.match(await page.locator('[data-benchmark-reading]').innerText(), /没有可用于成功 TPOT 的样本/);
    const row = { id: 'measured-1', scheduled: 0, permit: .2, start: .25, first: .5, end: .8, out: 4, success: true };
    await page.locator('#sl-benchmark-file').setInputFiles({ name: 'run.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({ schema: 'ai-infra-serving-lab/v1', unit: 's', evidence: 'measured', duration: 1, rows: [row] })) });
    await page.waitForFunction(() => document.querySelector('#sl-record-label').textContent.includes('原始采集记录'));
    assert.match(await page.locator('[data-benchmark-reading]').innerText(), /0.25 s/);
    assert.match(await page.locator('[data-benchmark-reading]').innerText(), /0.1 s\/token/);
    assert.equal(await page.locator('[name=warm]').isDisabled(), true);
    assert.doesNotMatch(await page.locator('#sl-svg').textContent(), /输入处理/);

    await go('trace-lab');
    await page.locator('[name=windowSpan]').selectOption('5');
    await page.locator('[name=traceFilter]').selectOption('gpu');
    await page.locator('[data-action=trace-pair]').click();
    assert.equal(await page.locator('[name=traceFilter]').inputValue(), 'all');
    assert.equal(await page.locator('#lab').getAttribute('data-depth'), 'modules');
    assert.match(await page.locator('#sl-stage').innerText(), /一条较早的 CPU runtime 候选/);
    assert.ok(await page.locator('#sl-svg rect[stroke="var(--fg)"]').count() >= 2, 'GPU event and CPU candidate are visible');
    const eventNode = page.locator('#sl-svg [data-trace]').first();
    await eventNode.evaluate(e => e.setAttribute('data-qa-retained', 'true'));
    await page.locator('[data-action=play]').click(); await page.waitForTimeout(150);
    assert.equal(await page.locator('#sl-svg [data-qa-retained]').count(), 1, 'playback keeps event geometry and focus');
    await page.locator('[data-action=play]').click();
    await shot('trace-real-correlation');
    const custom = [
      { ph: 'X', cat: 'cpu_op', name: 'capture', ts: 1000, dur: 100000, pid: 1, tid: 1 },
      { ph: 'X', cat: 'cuda_runtime', name: 'cudaLaunchKernel', ts: 2000, dur: 100, pid: 1, tid: 2, args: { correlation: 7 } },
      { ph: 'X', cat: 'kernel', name: 'far-away-device', ts: 90000, dur: 10000, pid: 9, tid: 10, args: { correlation: 7 } }
    ];
    await importTrace(custom);
    await page.locator('[name=windowSpan]').selectOption('5');
    await page.locator('[data-action=trace-pair]').click();
    assert.equal(await page.locator('[name=windowSpan]').inputValue(), '100', 'both distant intervals fit');
    assert.equal(await page.locator('#sl-svg [data-trace="1"]').count(), 1);
    assert.equal(await page.locator('#sl-svg [data-trace="2"]').count(), 1);
    await page.locator('[data-action=trace-gap]').click();
    assert.match(await page.locator('#sl-stage').innerText(), /覆盖完整间隔/);
    assert.equal(await page.locator('[name=windowSpan]').inputValue(), '100', 'whole 89ms gap fits');
    await shot('trace-full-gap');
    await importTrace([
      { ph: 'X', cat: 'kernel', name: 'left', ts: 0, dur: 10, pid: 1, tid: 1 },
      { ph: 'X', cat: 'kernel', name: 'right', ts: 99000, dur: 10, pid: 1, tid: 1 }
    ], 'empty-middle.json');
    await page.locator('[name=windowSpan]').selectOption('5');
    await page.locator('[name=windowStart]').evaluate(e => { e.value = 50; e.dispatchEvent(new Event('input', { bubbles: true })); });
    assert.match(await page.locator('#sl-stage').innerText(), /当前窗口没有可识别/);
    await importTrace([custom[0]], 'cpu-only.json');
    await page.locator('[data-action=trace-gap]').click();
    assert.match(await page.locator('#sl-error').innerText(), /没有识别到 GPU/);
    await page.locator('#sl-trace-file').setInputFiles({ name: 'zero.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify([{ ph: 'X', cat: 'kernel', ts: 0, dur: 0 }])) });
    await page.waitForFunction(() => document.querySelector('#sl-error').textContent.includes('导入失败'));
    assert.match(await page.locator('#sl-record-label').innerText(), /cpu-only.json/);

    for (const slug of ['deployment-lab', 'benchmark-lab', 'trace-lab']) {
      await go(slug);
      if (slug === 'trace-lab') await page.locator('[data-action=trace-pair]').click();
      else if (slug === 'benchmark-lab') await page.locator('[name=requestFocus]').selectOption('2');
      else await time(12.8);
      for (const width of [1440, 390, 320]) {
        await page.setViewportSize({ width, height: 900 });
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), slug + ' page stays within viewport ' + width);
        const missing = await page.locator('#lab button,#lab input,#lab select').evaluateAll(els => els.filter(e => !e.closest('[hidden]') && getComputedStyle(e).display !== 'none' && e.getBoundingClientRect().width < 1).map(e => e.outerHTML));
        assert.deepEqual(missing, [], 'usable controls');
        if (width !== 320) await shot(slug + '-' + width);
        await page.locator('[data-depth-choice=modules]').click();
        const outside = await page.locator('#sl-svg').evaluate(el => [...el.querySelectorAll('text')].filter(e => { const b=e.getBBox(); return b.x < -1 || b.x+b.width > el.viewBox.baseVal.width+1; }).map(e => e.textContent));
        assert.deepEqual(outside, [], slug + ' SVG text within bounds');
      }
      await page.locator('.theme-toggle').click(); await shot(slug + '-dark'); await page.locator('.theme-toggle').click();
    }
    assert.deepEqual(errors, []);
    console.log('Practice review: completion evidence, request timing/sample boundaries, trace focus/gaps, stable playback and responsive layouts passed.');
  } finally {
    remember(); targets = [...seen.values()];
    if (browser) await browser.close().catch(() => {});
    await server.close().catch(() => {});
    for (let i = 0; i < 20 && alive(targets).length; i++) await delay(100);
    for (const p of alive(targets)) process.kill(p.pid, 'SIGTERM');
    for (let i = 0; i < 20 && alive(targets).length; i++) await delay(100);
    for (const p of alive(targets)) process.kill(p.pid, 'SIGKILL');
    await delay(100);
    const remaining = alive(targets);
    console.log(JSON.stringify({ session: info.session, cleaned: remaining.length === 0, remaining }));
    assert.equal(remaining.length, 0, 'owned browser and helpers must exit');
    if (info.profile) {
      const profile = info.profile.split('=').slice(1).join('=');
      if (remaining.length === 0 && profile.startsWith(os.tmpdir()) &&
          path.basename(profile).startsWith('playwright_chromiumdev_profile-')) {
        fs.rmSync(profile, { recursive: true, force: true });
      }
      assert.equal(fs.existsSync(profile), false, 'the owned temporary browser profile must be removed');
    }
  }
})().catch(e => { console.error(e); process.exitCode = 1; });
