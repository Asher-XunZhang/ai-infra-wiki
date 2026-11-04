/* Real-page acceptance: one clock owns semantics, packets and geometry. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { execFileSync } = require('node:child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const M = require('../../pages/sglang/inference-overview/radix-model.js');

(async () => {
  const out = process.env.RADIX_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(), 'radix-qa-'));
  fs.mkdirSync(out, { recursive: true });
  const server = await chromium.launchServer({ headless: true,
    ...(process.env.CHROMIUM_EXECUTABLE ? { executablePath: process.env.CHROMIUM_EXECUTABLE } : {}) });
  const processHandle = server.process(), browserPID = processHandle.pid;
  const command = execFileSync('ps', ['-p', String(browserPID), '-o', 'command='], { encoding: 'utf8' });
  const profile = command.match(/--user-data-dir=([^ ]+)/)?.[1];
  const lifecycle = { controllerPID: process.pid, browserPID, profile };
  fs.writeFileSync(path.join(out, 'browser-lifecycle.json'), JSON.stringify(lifecycle, null, 2));
  try {
    const browser = await chromium.connect(server.wsEndpoint());
    const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    const base = process.env.OVERVIEW_TEST_URL || 'http://127.0.0.1:8765/';
    await page.goto(base + 'sglang/inference-overview/kv-cache.html#radix-lab-title');
    await page.locator('[data-radix-lab]').waitFor({ state: 'visible' });
    await page.clock.install();
    const lab = page.locator('[data-radix-lab]');
    const seek = async t => page.locator('#radix-seek').evaluate((el, t) => {
      el.value = String(t); el.dispatchEvent(new Event('input', { bubbles: true }));
    }, t);
    const click = selector => page.locator(selector).evaluate(el => el.click());
    const scroll = () => lab.evaluate(el => window.scrollTo({ top: scrollY + el.getBoundingClientRect().top - 85, behavior: 'instant' }));
    const snapshot = () => page.evaluate(() => ({
      time: document.querySelector('[data-radix-lab]').dataset.time,
      nodes: document.querySelector('#radix-tree-nodes').innerHTML,
      rows: document.querySelector('#radix-requests').innerHTML,
      flow: document.querySelector('#radix-flow').innerHTML,
      caption: document.querySelector('#radix-step-title').textContent
    }));
    const shot = async name => { await scroll(); await page.screenshot({ path: path.join(out, name + '.png') }); };
    assert.equal(await page.locator('.animation-speed').count(), 0, 'no independent shared playback controller');
    assert.equal(await page.locator('#radix-timeline [data-step]').count(), 12);
    assert.equal(await lab.locator('.radix-page,.radix-pool-panel,[data-page]').count(), 0, 'no physical KV pool or pages in tree lesson');
    assert.match(await page.locator('#radix-requests-title').innerText(), /查询与匹配结果/);
    assert.match(await page.locator('#radix-source a').getAttribute('href'), new RegExp(M.revision));
    const timeline = M.timeline(), phase = i => timeline.phases[i].start;
    await shot('initial-1280');

    await seek(phase(1) + 2.2); // Return packet in flight.
    assert.equal(await page.locator('.radix-packet').count(), 1);
    assert.match(await page.locator('[data-request="R1"] .radix-request-fields').innerText(), /索引长度 0/);
    await click('#radix-play'); await page.clock.runFor(180); await click('#radix-play');
    const paused = await snapshot();
    await page.clock.runFor(1200);
    assert.deepEqual(await snapshot(), paused, 'pause freezes packets, nodes, text and time');
    assert.equal(await page.evaluate(() => document.getAnimations().length), 0, 'no CSS/WAAPI clock bypasses pause');
    await shot('return-paused');
    await seek(phase(1) + 2.9);
    assert.equal(await page.locator('[data-node="n1"]').getAttribute('data-lock'), '0');
    assert.match(await page.locator('[data-request="R1"]').innerText(), /prefix_indices/);
    await seek(phase(1) + 3.6);
    assert.equal(await page.locator('[data-node="n1"]').getAttribute('data-lock'), '1');

    await seek(phase(2) + .6);
    assert.equal(await page.locator('[data-node="n3"]').count(), 0);
    assert.match(await page.locator('[data-request="R1"] .radix-insert-argument').innerText(), /待插入 value/);
    await shot('insert-parameters');
    await seek(phase(2) + 2);
    assert.equal(await page.locator('[data-node="n3"]').count(), 1);
    assert.match(await page.locator('[data-request="R1"] .radix-request-fields').innerText(), /索引长度 8/);
    const beforeInsert = await snapshot();
    await seek(timeline.duration); await seek(phase(2) + 2);
    assert.deepEqual(await snapshot(), beforeInsert, 'backwards scrubbing reconstructs exact DOM state');
    await seek(phase(3) + 1.9); await shot('split-mid-motion');
    const split = await snapshot(); await seek(phase(4)); await seek(phase(3) + 1.9);
    assert.deepEqual(await snapshot(), split, 'mid-morph geometry is deterministic');
    await seek(timeline.phases[4].end); await shot('both-protected-1280');
    const endpoints = await page.evaluate(() => {
      const svg=document.querySelector('#radix-tree-edges'), nodes=window.RadixLifecycleModel.atTime(window.RadixLifecycleModel.timeline(),Number(document.querySelector('[data-radix-lab]').dataset.time)).raw.nodes;
      return [...svg.querySelectorAll(':scope > path')].map((path,i) => {
        const id=Object.keys(nodes).filter(id=>nodes[id].parent)[i], parent=document.querySelector(`[data-node="${nodes[id].parent}"]`).getBoundingClientRect(), child=document.querySelector(`[data-node="${id}"]`).getBoundingClientRect();
        const point=(length)=>{const p=path.getPointAtLength(length);return new DOMPoint(p.x,p.y).matrixTransform(svg.getScreenCTM());};
        const start=point(0),end=point(path.getTotalLength());
        return Math.max(Math.abs(start.x-parent.right),Math.abs(start.y-(parent.top+parent.height/2)),Math.abs(end.x-child.left),Math.abs(end.y-(child.top+child.height/2)));
      });
    });
    assert.ok(endpoints.every(error=>error<1), 'tree connectors meet the exact rendered card edges');

    await seek(phase(6) + 1.2);
    assert.equal(await page.locator('[data-node="n3"]').getAttribute('data-lock'), '0');
    assert.equal(await page.locator('[data-request="R1"] .is-released').count(), 0);
    await seek(timeline.phases[8].end);
    assert.equal(await page.locator('[data-node="n1"]').count(), 0);
    assert.equal(await page.locator('[data-node="n2"]').getAttribute('data-lock'), '1');
    await shot('shared-prefix-survives');

    const speedProgress = async speed => {
      await seek(0); await page.locator('#radix-speed').selectOption(speed);
      await click('#radix-play'); await page.clock.runFor(1000); await click('#radix-play');
      return Number(await lab.getAttribute('data-time'));
    };
    const one = await speedProgress('1'), two = await speedProgress('2');
    assert.ok(Math.abs(two / one - 2) < .08, `speed ratio ${two / one}`);

    for (const demo of M.demos) {
      await click(`button[data-mode="${demo.id}"]`); await seek(0);
      await page.evaluate(() => {
        window.radixObserved = new Set();
        window.radixObserver = new MutationObserver(() => {
          const lab = document.querySelector('[data-radix-lab]');
          window.radixObserved.add(lab.dataset.phase + '/' + lab.dataset.event);
        });
        window.radixObserver.observe(document.querySelector('[data-radix-lab]'), { attributes: true });
      });
      await page.locator('#radix-speed').selectOption('2');
      await click('#radix-play'); await page.clock.runFor(M.timeline(demo.id).duration * 500 + 200);
      assert.doesNotMatch(await lab.innerText(), /页 [0-9]|KV 池|ReqToTokenPool|空闲表|分页位置/);
      assert.equal(await lab.getAttribute('data-playing'), 'false', `${demo.id} finishes in one play`);
      assert.equal(Number(await lab.getAttribute('data-time')), Number(M.timeline(demo.id).duration.toFixed(3)));
      const observed = await page.evaluate(() => { window.radixObserver.disconnect(); return [...window.radixObserved]; });
      for (const p of M.timeline(demo.id).phases) for (const e of p.events) {
        assert.ok(observed.includes(p.index + '/' + e.at), `${demo.id} renders event ${p.index}/${e.at}`);
      }
    }
    await click('button[data-mode="journey"]');
    assert.equal(Number(await lab.getAttribute('data-time')), 0);
    await page.locator('#radix-speed').selectOption('1');
    await seek(timeline.phases[4].end);
    const layout = [];
    for (const width of [1440, 1280, 820, 390, 320]) {
      await page.setViewportSize({ width, height: width < 600 ? 844 : 900 });
      await page.clock.runFor(20);
      await shot('layout-' + width);
      const dimensions = await page.evaluate(() => ({
        width: innerWidth, document: document.documentElement.scrollWidth,
        lab: document.querySelector('[data-radix-lab]').getBoundingClientRect().toJSON(),
        figure: document.querySelector('.radix-figure').getBoundingClientRect().toJSON(),
        overflowingNodes: [...document.querySelectorAll('.radix-node')].filter(n => n.scrollHeight > n.clientHeight + 1).map(n => n.dataset.node)
      }));
      assert.ok(dimensions.document <= width, `no document overflow at ${width}`);
      assert.deepEqual(dimensions.overflowingNodes, [], `node text fits at ${width}`);
      layout.push(dimensions);
    }
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.emulateMedia({ colorScheme: 'dark', reducedMotion: 'reduce' });
    await seek(phase(3) + 1.9); await shot('dark-reduced-motion');
    assert.equal(await page.locator('.radix-packet').count(), 0);
    assert.equal(await page.locator('[data-node="n2"]').getAttribute('data-lock'), '1');
    await page.emulateMedia({ colorScheme: 'light', reducedMotion: 'no-preference' });
    await seek(phase(1) + 2.9); const deepLink = page.url();
    await page.goto(deepLink);
    assert.equal(Number(await lab.getAttribute('data-time')), Number((phase(1) + 2.9).toFixed(3)), 'deep link restores continuous time');
    const noJS = await browser.newContext({ javaScriptEnabled: false });
    const staticPage = await noJS.newPage(); await staticPage.goto(base + 'sglang/inference-overview/kv-cache.html');
    assert.ok(await staticPage.locator('#radix-conditions-title').isVisible());
    await noJS.close();
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(out, 'acceptance.json'), JSON.stringify({ source: M.revision, url: base, errors, speedRatio: two / one, layout, checks: ['pause', 'speed', 'causality', 'reverse seek', 'mid-morph seek', 'all events in four complete plays', 'shared eviction', 'responsive', 'reduced motion', 'deep link', 'no JS'] }, null, 2));
    await browser.close();
    console.log('PASS: clock, causality, complete playback, responsiveness and accessible fallbacks. Evidence: ' + out);
  } finally {
    await server.close();
    lifecycle.processExited = processHandle.exitCode !== null || processHandle.signalCode !== null;
    lifecycle.profileRemoved = !profile || !fs.existsSync(profile);
    const processes = execFileSync('ps', ['-axo', 'pid=,command='], { encoding: 'utf8' });
    lifecycle.residual = profile ? processes.split('\n').filter(line => line.includes(profile)) : [];
    fs.writeFileSync(path.join(out, 'browser-lifecycle.json'), JSON.stringify(lifecycle, null, 2));
    assert.ok(lifecycle.processExited && lifecycle.profileRemoved && !lifecycle.residual.length, 'owned browser and profile cleaned');
  }
})().catch(e => { console.error(e); process.exitCode = 1; });
