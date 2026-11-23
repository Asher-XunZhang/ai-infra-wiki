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
  const info = { session: 'serving-labs-review-' + process.pid, controller: process.pid, browser: owned.pid,
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
    async function go(slug, modules = true) {
      await page.goto(base + 'sglang/' + slug + '/?learn=labs');
      await page.locator('[data-serving-lab][data-ready=true]').waitFor();
      await page.waitForFunction(() => !!document.querySelector('#sl-svg title'));
      if (modules) await page.locator('[data-depth-choice=modules]').click();
      remember();
      assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-playing'), 'false');
    }
    async function time(value) {
      await page.locator('[name=seek]').evaluate((e, v) => {
        e.value = String(v); e.dispatchEvent(new Event('input', { bubbles: true }));
      }, value);
    }
    async function shot(name, canvas = true) {
      if (!output) return;
      if (canvas) await page.locator('.sl-canvas').screenshot({ path: path.join(output, name + '.png') });
      else {
        await page.locator('#lab').evaluate(e => window.scrollTo({ top: e.getBoundingClientRect().top + scrollY - 90, behavior: 'instant' }));
        await page.screenshot({ path: path.join(output, name + '.png') });
      }
    }

    for (const slug of slugs) {
      await go(slug, false);
      assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-depth'), 'overview');
      await time(slug === 'trace-lab' ? 13000 : 3.375);
      const state = async () => ({ time: await page.locator('[data-serving-lab]').getAttribute('data-time'),
        playing: await page.locator('[data-serving-lab]').getAttribute('data-playing'), results: await page.locator('#sl-results').innerText() });
      const before = await state();
      await page.locator('[data-depth-choice=modules]').click();
      assert.deepEqual(await state(), before, slug + ' module projection preserves state');
      await page.locator('[data-source-toggle]').click();
      assert.deepEqual(await state(), before, slug + ' source overlay preserves state');
      assert.equal(+await page.locator('.ld-source-panel').getAttribute('data-time'), +before.time);
      await page.locator('[data-depth-choice=overview]').click();
      assert.deepEqual(await state(), before, slug + ' collapse preserves state');
      await shot(slug + '-overview', false);
      await page.reload();
      await page.waitForFunction(() => !!document.querySelector('.ld-overview svg,.ld-overview [data-memory-overview]'));
      assert.equal(+await page.locator('[data-serving-lab]').getAttribute('data-time'), +before.time, slug + ' URL restores progress');
      assert.equal(await page.locator('[data-source-toggle]').getAttribute('aria-expanded'), 'true');
      await page.locator('[data-action=play]').click();
      await page.waitForTimeout(100);
      const moving = +(await state()).time;
      await page.locator('[data-depth-choice=modules]').click();
      assert.equal((await state()).playing, 'true');
      assert.ok(+(await state()).time >= moving);
      await page.locator('[data-action=play]').click();
    }
    await go('hicache-loadback');
    async function activeFlows() {
      return page.locator('#sl-svg [data-flow][data-active=true]').evaluateAll(es => es.map(e => ({
        id: e.dataset.flow, kind: e.dataset.flowKind, d: e.querySelector('path').getAttribute('d'),
        phase: e.querySelector('path').getAttribute('stroke-dashoffset') })));
    }
    for (const value of [2.9, 7, 8.99, 18]) { await time(value); assert.deepEqual(await activeFlows(), []); }
    await time(10.25); const pulse = await activeFlows();
    assert.equal(pulse[0].id, 'copy-0');
    await page.waitForTimeout(150); assert.deepEqual(await activeFlows(), pulse, 'pause freezes pulse phase');
    await time(10.5); const shifted = await activeFlows();
    assert.equal(shifted[0].d, pulse[0].d); assert.notEqual(shifted[0].phase, pulse[0].phase);
    await time(10.25); assert.deepEqual(await activeFlows(), pulse, 'rewind restores exact phase');
    await time(12.5); assert.equal((await activeFlows())[0].id, 'copy-1');
    assert.match(await page.locator('#sl-stage').innerText(), /同时推进/);
    await shot('hicache-parallel-pulse');
    await go('deployment-lab'); await time(12.05);
    assert.equal((await activeFlows()).length, 2, 'Decode submission overlaps first ID return');
    await shot('deployment-parallel-pulse');
    await time(12.45); assert.deepEqual(await activeFlows(), [], 'CPU detokenization is not transport');

    await go('memory-budget');
    assert.match(await page.locator('#sl-results').innerText(), /245,760/);
    assert.match(await page.locator('#sl-svg').textContent(), /99,968/);
    await page.locator('[data-action=play]').click();
    await page.waitForTimeout(230);
    const before = +await page.locator('[data-serving-lab]').getAttribute('data-time');
    assert.ok(before > 0);
    await page.locator('[name=speed]').fill('1.5');
    assert.ok(+await page.locator('[data-serving-lab]').getAttribute('data-time') >= before);
    await page.locator('[name=context]').selectOption('32768');
    assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-time'), '0');
    assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-playing'), 'false');
    assert.match(await page.locator('#sl-svg').textContent(), /不能接纳/);
    await page.locator('[name=available]').selectOption('8');
    await page.locator('[name=fraction]').selectOption('0.75');
    assert.match(await page.locator('#sl-svg').textContent(), /运行预留还差 2 GiB/);
    await shot('memory-shortage');
    await time(8); await time(0);
    assert.match(await page.locator('#sl-svg').textContent(), /当前可用/);

    await go('hicache-loadback');
    await time(16);
    assert.match(await page.locator('#sl-results').innerText(), /复制未全部完成/);
    assert.match(await page.locator('#sl-svg').textContent(), /设备与主机仍受保护/);
    await shot('hicache-layer-ready');
    await time(20);
    assert.match(await page.locator('#sl-svg').textContent(), /继续持有 L1 前缀/);
    await page.locator('[name=scenario]').selectOption('cancel');
    await time(6);
    assert.match(await page.locator('#sl-svg').textContent(), /仍在持有/);
    assert.match(await page.locator('#sl-svg').textContent(), /R1 已取消/);
    await shot('hicache-cancel');
    await time(9);
    assert.match(await page.locator('#sl-svg').textContent(), /已归还目标/);
    await time(4);
    assert.doesNotMatch(await page.locator('#sl-svg').textContent(), /R1 已取消/);
    for (const scene of ['host', 'device-shortage', 'miss']) {
      await page.locator('[name=scenario]').selectOption(scene);
      assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-time'), '0');
      await page.locator('[data-action=next]').click();
    }

    await go('deployment-lab');
    await page.locator('[name=scenario]').selectOption('oom');
    await time(7);
    assert.match(await page.locator('#sl-svg').textContent(), /初始化失败/);
    assert.doesNotMatch(await page.locator('#sl-svg').textContent(), /身份 \/ 就绪已检查/);
    await page.locator('[name=scenario]').selectOption('cancel');
    await time(13.25);
    assert.match(await page.locator('#sl-svg').textContent(), /取消通知已发出/);
    await shot('deployment-cancel');
    await page.locator('[data-action=reset]').click();
    await page.locator('[data-action=next]').focus();
    await page.keyboard.press('Enter');
    assert.ok(+await page.locator('[data-serving-lab]').getAttribute('data-time') > 0);

    await go('benchmark-lab');
    assert.match(await page.locator('#sl-results').innerText(), /6 \/ 6/);
    await page.locator('[name=failure]').check();
    assert.match(await page.locator('#sl-results').innerText(), /5 \/ 6/);
    await time(18);
    await shot('benchmark-failure');
    const record = { schema: 'ai-infra-serving-lab/v1', unit: 's', evidence: 'protocol-test',
      duration: 1, rows: [{ id: '<img src=x onerror=alert(1)>', scheduled: 0, permit: .1, start: .11,
        first: .2, end: .6, out: 3, success: true }] };
    await page.locator('#sl-benchmark-file').setInputFiles({ name: 'run.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(record)) });
    await page.waitForFunction(() => document.querySelector('[name=interval]').disabled);
    assert.match(await page.locator('#sl-results').innerText(), /1 \/ 1/);
    assert.equal(await page.locator('#sl-stage img').count(), 0, 'imported IDs remain text');
    await page.locator('#sl-benchmark-file').setInputFiles({ name: 'bad.json', mimeType: 'application/json', buffer: Buffer.from('{"ttfts":[0.1]}') });
    await page.waitForFunction(() => document.querySelector('#sl-error').textContent.includes('导入失败'));
    assert.match(await page.locator('#sl-results').innerText(), /1 \/ 1/);
    await page.locator('[data-action=example]').click();
    assert.equal(await page.locator('[name=interval]').isDisabled(), false);

    await go('trace-lab');
    await page.waitForFunction(() => document.querySelector('#sl-record-label').textContent.includes('ResNet'));
    assert.ok(await page.locator('[data-trace]').count() > 100);
    await page.locator('.sl-tops button').first().click();
    assert.match(await page.locator('#sl-stage').innerText(), /原始起点/);
    await page.locator('[name=windowSpan]').selectOption('10');
    await page.locator('[name=windowStart]').fill('35');
    await page.locator('[name=traceFilter]').selectOption('gpu');
    await shot('trace-measured-window');
    const sample = { traceEvents: [
      { ph: 'X', cat: 'kernel', name: 'GPU sample', ts: 100, dur: 20, pid: 0, tid: 7, args: { correlation: 2 } },
      { ph: 'X', cat: 'cuda_runtime', name: 'cudaLaunchKernel', ts: 98, dur: 1, pid: 5, tid: 6, args: { correlation: 2 } }
    ] };
    await page.locator('#sl-trace-file').setInputFiles({ name: 'own.json.gz', mimeType: 'application/gzip', buffer: zlib.gzipSync(JSON.stringify(sample)) });
    await page.waitForFunction(() => document.querySelector('#sl-record-label').textContent.includes('own.json.gz'));
    assert.match(await page.locator('#sl-results').innerText(), /0.02 ms/);
    await page.locator('#sl-trace-file').setInputFiles({ name: 'empty.json', mimeType: 'application/json', buffer: Buffer.from('{"traceEvents":[]}') });
    await page.waitForFunction(() => document.querySelector('#sl-error').textContent.includes('没有可读'));
    assert.match(await page.locator('#sl-record-label').innerText(), /own.json.gz/);

    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }, { width: 320, height: 740 }]) {
      await page.setViewportSize(viewport);
      for (const slug of slugs) {
        await go(slug);
        await page.waitForTimeout(80);
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
          slug + ' must not overflow the whole page at ' + viewport.width);
        const outside = await page.locator('#sl-svg').evaluate(svg => {
          const b = svg.viewBox.baseVal;
          return [...svg.querySelectorAll('text')].filter(e => {
            const x = e.getBBox(); return x.width && (x.x < -1 || x.y < -1 || x.x + x.width > b.width + 1 || x.y + x.height > b.height + 1);
          }).map(e => e.textContent);
        });
        assert.deepEqual(outside, [], slug + ' SVG text bounds');
        await page.locator('[data-depth-choice=overview]').click();
        const overviewOutside = slug === 'memory-budget' ? [] : await page.locator('.sl-overview-canvas svg').evaluate(svg => {
          const b = svg.viewBox.baseVal;
          return [...svg.querySelectorAll('text')].filter(e => {
            const x = e.getBBox(); return x.width && (x.x < -1 || x.y < -1 || x.x + x.width > b.width + 1 || x.y + x.height > b.height + 1);
          }).map(e => e.textContent);
        });
        assert.deepEqual(overviewOutside, [], slug + ' overview text bounds');
        if (viewport.width === 390) await shot(slug + '-mobile', false);
        if (viewport.width === 1440) await shot(slug + '-desktop', false);
        await page.locator('.theme-toggle').click();
        await page.locator('.theme-toggle').click();
      }
    }
    const context = await browser.newContext({ javaScriptEnabled: false, viewport: { width: 390, height: 844 } });
    const plain = await context.newPage();
    for (const slug of slugs) {
      await plain.goto(base + 'sglang/' + slug + '/');
      assert.ok(await plain.locator('#evidence').innerText());
      assert.ok(await plain.locator('#check details').count() >= 3);
    }
    await context.close();
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await go('hicache-loadback');
    assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-playing'), 'false');
    await time(10.5);
    assert.equal(await page.locator('#sl-svg [data-active=true] path[stroke-dashoffset]').count(), 0, 'reduced motion uses static activity highlight');
    assert.deepEqual(errors, []);
    remember(); targets = [...seen.values()];
    console.log('Five labs: responsive layouts, controls, replay, imports, measured sample, no-JS and reduced motion passed.');
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
