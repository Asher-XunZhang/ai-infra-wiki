/* Compact regression for the parallel/communication readability pass.
 * It checks the concrete collisions found in the real screenshots: reduction
 * values sharing the delivery lane, TP call-outs sitting on the local cells,
 * SP/AllGather copies occupying one lane, DCP O/Z copies crossing, and EP
 * dispatch tokens covering the annotation band. */
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const choose = require('./choose_motion_scene.cjs');
const base = process.env.OVERVIEW_TEST_URL || 'http://127.0.0.1:18765/';
const chrome = process.env.CHROMIUM_EXECUTABLE || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

(async () => {
  const server = await chromium.launchServer({ headless: true, executablePath: chrome });
  const proc = server.process(), profileArg = proc.spawnargs.find(a => a.startsWith('--user-data-dir='));
  console.log(JSON.stringify({ session: 'readability-parallel-comm', controller: process.pid, browser: proc.pid, profile: profileArg }));
  let browser;
  try {
    browser = await chromium.connect(server.wsEndpoint());
    const page = await browser.newPage();
    const seek = p => page.locator('.motion-scrub input').evaluate((e, v) => {
      e.value = v; e.dispatchEvent(new Event('input', { bubbles: true }));
    }, p);
    const overlap = (a, b) => {
      const w = Math.min(a.right, b.right) - Math.max(a.left, b.left);
      const h = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
      return w > 1 && h > 1 ? w * h : 0;
    };

    await page.goto(base + 'foundations/communication.html');
    for (const width of [320, 390, 768, 1440, 1994]) {
      await page.setViewportSize({ width, height: 1100 });
      for (const index of [8, 9, 10, 11, 12, 13, 14]) {
        await choose(page, index);
        for (const p of [2.6, 3]) {
          await seek(p);
          const tokens = await page.locator('.mechanism-film svg').evaluate(svg => [...svg.querySelectorAll('g[data-motion-id^="comm-result-"]')]
            .filter(g => Number(g.getAttribute('opacity') ?? 1) > .02)
            .map(g => ({ text: g.textContent.trim(), r: g.getBoundingClientRect().toJSON() })));
          assert.ok(tokens.length >= 2, `scene ${index} has result actors`);
          for (let i = 0; i < tokens.length; i++) for (let j = i + 1; j < tokens.length; j++) {
            if (tokens[i].text === tokens[j].text) continue; // copies of one value may overlap
            assert.equal(overlap(tokens[i].r, tokens[j].r), 0,
              `scene ${index} width ${width} p${p}: ${tokens[i].text} and ${tokens[j].text} cover each other`);
          }
        }
      }
    }

    await page.goto(base + 'foundations/parallel.html');
    for (const width of [320, 390, 768, 1440, 1994]) {
      await page.setViewportSize({ width, height: 1100 });
      // TP: the call-out must clear the local-contribution cells, and the
      // numeric result must appear with the final grids while the check hides.
      for (const index of [0, 1]) {
        await choose(page, index);
        await seek(3.5);
        const bad = await page.locator('.mechanism-film svg').evaluate(svg => {
          const texts = [...svg.querySelectorAll('text')].filter(e => !e.closest('g.teach-token') && /×.*=/.test(e.textContent));
          const cells = [...svg.querySelectorAll('g[data-motion-id^="tp-local"]')].filter(g => Number(g.getAttribute('opacity') ?? 1) > .02);
          const hit = [];
          for (const t of texts) for (const c of cells) {
            const a = t.getBoundingClientRect(), b = c.getBoundingClientRect();
            if (Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1 && Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 1) hit.push(t.textContent);
          }
          return hit;
        });
        assert.deepEqual(bad, [], `TP scene ${index} width ${width}: call-out overlaps local cells`);
        await seek(5.5);
        const state = await page.locator('.mechanism-film svg').evaluate(svg => {
          const values = [...svg.querySelectorAll('g[data-motion-id^="tp-final"] text')].map(e => e.textContent);
          const check = svg.querySelector('[data-tp-check="same-coord"]');
          return { values, hasCheck: !!check, check: Number(check?.getAttribute('opacity') ?? 0) };
        });
        assert.ok(state.values.length === 8 && state.values.every(v => /^\d+$/.test(v)), 'TP final grid shows numbers before the check');
        assert.ok(state.hasCheck, 'TP same-coordinate check exists');
        assert.ok(state.check < .02, 'TP check waits for the completed grid');
      }
      await choose(page, 7); await seek(2.5);
      const gather = await page.locator('.mechanism-film svg').evaluate(svg => [...svg.querySelectorAll('[data-motion-id^="sp-gather-"]')]
        .map(g => ({ x: g.dataset.x, y: g.dataset.y, op: Number(g.getAttribute('opacity') ?? 1) })).filter(g => g.op > .02));
      assert.ok(gather.length >= 2, 'SP copies are visible');
      assert.equal(new Set(gather.map(g => `${g.x},${g.y}`)).size, gather.length, `SP copies share a lane at width ${width}`);
      await choose(page, 10); await seek(2.5);
      const incoming = await page.locator('.mechanism-film svg').evaluate(svg => [...svg.querySelectorAll('[data-motion-id^="dcp-incoming-"]')]
        .map(g => g.getBoundingClientRect().toJSON()));
      assert.equal(overlap(incoming[0], incoming[1]), 0, `DCP O/Z copies cross at width ${width}`);
      await choose(page, 5); await seek(1.5);
      const hidden = await page.locator('.mechanism-film svg').evaluate(svg => [...svg.querySelectorAll('text')]
        .filter(e => /E[02]：|两个分支按路由/.test(e.textContent))
        .every(e => Number(e.getAttribute('opacity') ?? 1) < .02));
      assert.ok(hidden, 'EP dispatch band is cleared while tokens travel');
    }
    console.log('PASS compact readability regression: reduction values, TP call-outs, SP AllGather lanes, DCP O/Z lanes, EP dispatch annotations.');
  } finally {
    await browser?.close().catch(() => {});
    await server.close().catch(() => {});
    let cleanupError = '';
    try { process.kill(proc.pid, 0); cleanupError = 'owned browser alive ' + proc.pid; }
    catch (e) { if (e.code !== 'ESRCH') cleanupError = e.message; }
    if (profileArg) {
      const dir = profileArg.slice('--user-data-dir='.length);
      try { require('node:fs').rmSync(dir, { recursive: true, force: true }); } catch (e) { cleanupError ||= e.message; }
      if (require('node:fs').existsSync(dir)) cleanupError ||= 'owned profile remains ' + dir;
    }
    if (cleanupError) { console.error('CLEANUP-FAIL', cleanupError); process.exitCode = 1; }
    else console.log('CLEANED readability browser PID ' + proc.pid);
  }
})().catch(e => { console.error(e); process.exitCode = 1; });
