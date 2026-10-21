/* Chapter 1 basics readability regression: tokens, generation, deployment and
 * the two P95 distribution scenes. Geometry is measured from rendered boxes,
 * not from hard-coded coordinates, so a layout change must stay collision-free
 * on every checked viewport. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const chooseScene = require('./choose_motion_scene.cjs');

const BASE = process.env.OVERVIEW_TEST_URL || process.env.BASICS_TEST_URL || 'http://127.0.0.1:18802/';
const CHROME = process.env.CHROMIUM_EXECUTABLE || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const LESSONS = {
  tokens: { variants: [0, 1], positions: [0, .5, 1, 2, 3, 4] },
  generation: { variants: [0, 1], positions: [0, 1, 2, 2.5, 2.8, 3, 3.5, 4, 4.5, 4.9, 5, 6] },
  deployment: { variants: [0], positions: [0, 1, 2, 2.1, 2.3, 2.5, 2.6, 2.7, 2.9, 3, 4, 5] },
  metrics: { variants: [2, 3], positions: [0, .5, 1, 2, 2.5, 3, 4] },
};
const WIDTHS = [320, 390, 768, 1440, 1994];

let server;
let browser;
let page;
let owned;
let profilePath;
let shuttingDown;

const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

function pidState(pid) {
  try {
    process.kill(pid, 0);
    return 'alive';
  } catch (error) {
    return error.code === 'ESRCH' ? 'gone' : `error:${error.code || error.message}`;
  }
}

function profileProcesses() {
  if (!profilePath) return [];
  let output;
  try {
    output = execFileSync('pgrep', ['-fl', profilePath], { encoding: 'utf8' });
  } catch (error) {
    if (error.status === 1) return [];
    throw error;
  }
  return output.trim().split('\n').filter(Boolean).map(line => {
    const match = line.match(/^(\d+)\s+(.*)$/);
    return match ? { pid: Number(match[1]), command: match[2] } : null;
  }).filter(entry => entry && pidState(entry.pid) !== 'gone');
}

function signalPid(pid, signal) {
  try {
    process.kill(pid, signal);
    return null;
  } catch (error) {
    return error.code === 'ESRCH' ? null : (error.code || error.message);
  }
}

async function waitForGone(pid, timeout) {
  const deadline = Date.now() + timeout;
  let state = pidState(pid);
  while (state !== 'gone' && Date.now() < deadline) {
    await delay(100);
    state = pidState(pid);
  }
  return state;
}

function isOwnedTemporaryProfile(candidate) {
  if (!candidate || !/playwright/i.test(path.basename(candidate))) return false;
  const candidates = [path.resolve(candidate)];
  try { candidates.push(fs.realpathSync(candidate)); } catch {}
  const roots = [path.resolve(os.tmpdir())];
  try { roots.push(fs.realpathSync(os.tmpdir())); } catch {}
  return candidates.some(file => roots.some(root => file.startsWith(root + path.sep)));
}

async function shutdown(reason) {
  if (shuttingDown) return shuttingDown;
  shuttingDown = (async () => {
    await page?.close().catch(() => {});
    await browser?.close().catch(() => {});
    await server?.close().catch(() => {});

    const failures = [];
    const residualBefore = profileProcesses().map(entry => entry.pid);
    let browserState = owned?.pid ? await waitForGone(owned.pid, 1500) : 'not-created';
    let browserFailure = '';
    for (const signal of ['SIGTERM', 'SIGKILL']) {
      if (browserState === 'gone') break;
      // Only signal a PID whose command line still carries this launch's
      // unique profile; a reused PID must never be mistaken for our browser.
      if (!profileProcesses().some(entry => entry.pid === owned.pid)) {
        browserFailure = `browser pid ${owned.pid} is ${browserState} but is not confirmed by profile; not touching it`;
        break;
      }
      const error = signalPid(owned.pid, signal);
      if (error) failures.push(`browser pid ${owned.pid} ${signal}: ${error}`);
      browserState = await waitForGone(owned.pid, signal === 'SIGTERM' ? 1500 : 1000);
    }
    if (owned?.pid && browserState !== 'gone' && !browserFailure) {
      browserFailure = `browser pid ${owned.pid} not gone (${browserState})`;
    }
    if (browserFailure) failures.push(browserFailure);

    let remaining = profileProcesses();
    if (remaining.length) {
      for (const signal of ['SIGTERM', 'SIGKILL']) {
        for (const entry of remaining) {
          const error = signalPid(entry.pid, signal);
          if (error) failures.push(`profile pid ${entry.pid} ${signal}: ${error}`);
        }
        await delay(signal === 'SIGTERM' ? 250 : 150);
        remaining = profileProcesses();
        if (!remaining.length) break;
      }
      if (remaining.length) failures.push(`${remaining.length} profile process(es) still alive: ${remaining.map(entry => entry.pid).join(',')}`);
    }

    const profileOwned = isOwnedTemporaryProfile(profilePath);
    let profileRemoved = !profilePath || !fs.existsSync(profilePath);
    if (profilePath && !profileRemoved && profileOwned && !remaining.length) {
      try { fs.rmSync(profilePath, { recursive: true, force: true }); } catch (error) {
        failures.push(`profile removal failed: ${error.code || error.message}`);
      }
      profileRemoved = !fs.existsSync(profilePath);
    }

    console.log(JSON.stringify({
      event: 'cleanup',
      reason,
      controller: process.pid,
      browser: owned?.pid ?? null,
      browserState,
      profile: profilePath,
      profileOwned,
      profileRemoved,
      residualProfileCount: residualBefore.length,
      residualProfilePids: residualBefore,
      remainingProfilePids: remaining.map(entry => entry.pid),
    }));
    if (failures.length) throw new Error(`browser cleanup incomplete (${reason}): ${failures.join('; ')}`);
    return { browserState, profileProcesses: 0 };
  })();
  return shuttingDown;
}

for (const [signal, code] of [['SIGINT', 130], ['SIGTERM', 143]]) {
  process.once(signal, () => {
    shutdown(signal)
      .then(() => process.exit(code))
      .catch(error => {
        console.error(error);
        process.exit(1);
      });
  });
}


async function seek(page, position) {
  await page.locator('.mechanism-film .motion-scrub input').evaluate((el, p) => {
    el.value = String(p);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, position);
  await page.waitForTimeout(35);
}

function geometry() {
  const film = document.querySelector('.mechanism-film');
  const svg = film.querySelector('svg');
  const visible = e => {
    for (let a = e; a && a !== svg; a = a.parentElement) {
      const s = getComputedStyle(a);
      if (Number(s.opacity) < .05 || s.display === 'none' || s.visibility === 'hidden') return false;
    }
    return true;
  };
  const box = e => {
    const r = e.getBoundingClientRect();
    return { x: r.x, y: r.y, w: r.width, h: r.height, right: r.right, bottom: r.bottom };
  };
  const order = [...svg.querySelectorAll('*')];
  const z = new Map(order.map((e, i) => [e, i]));
  const texts = [...svg.querySelectorAll('text')].filter(e => e.textContent.trim() && visible(e)).map(e => {
    const g = e.closest('g[data-motion-id]');
    return { text: e.textContent.trim(), id: g?.dataset.motionId || '', z: z.get(e), group: g || e.parentElement, ...box(e) };
  });
  const tokens = [...svg.querySelectorAll('g[data-motion-id] > rect')].filter(visible).map(e => {
    const g = e.parentElement;
    return { id: g.dataset.motionId, z: z.get(g), ...box(e) };
  });
  const textHits = [];
  for (let i = 0; i < texts.length; i++) for (let j = i + 1; j < texts.length; j++) {
    const a = texts[i], b = texts[j];
    const ix = Math.min(a.right, b.right) - Math.max(a.x, b.x);
    const iy = Math.min(a.bottom, b.bottom) - Math.max(a.y, b.y);
    if (ix > 1 && iy > 1) textHits.push(`${a.text} × ${b.text}`);
  }
  const covered = [];
  for (const t of texts) for (const r of tokens) {
    if (t.id === r.id) continue;
    const cx = (t.x + t.right) / 2, cy = (t.y + t.bottom) / 2;
    if (cx > r.x + 1 && cx < r.right - 1 && cy > r.y + 1 && cy < r.bottom - 1 && r.z > (t.group ? z.get(t.group) : t.z)) {
      covered.push(`${t.text} ← ${r.id}`);
      break;
    }
  }
  const stage = box(film.querySelector('.motion-stage'));
  const outside = texts.filter(t => t.x < stage.x - 2 || t.right > stage.right + 2 || t.y < stage.y - 2 || t.bottom > stage.bottom + 2).map(t => t.text);
  const samples = [...svg.querySelectorAll('g[data-motion-id^=sample-]')].filter(visible).map(g => ({
    id: g.dataset.motionId, text: g.querySelector('text')?.textContent || '', ...box(g.querySelector('rect')),
  }));
  const sampleHits = [];
  for (let i = 0; i < samples.length; i++) for (let j = i + 1; j < samples.length; j++) {
    const a = samples[i], b = samples[j];
    const ix = Math.min(a.right, b.right) - Math.max(a.x, b.x);
    const iy = Math.min(a.bottom, b.bottom) - Math.max(a.y, b.y);
    if (ix > 1 && iy > 1) sampleHits.push(`${a.text} × ${b.text}`);
  }
  return { textHits, covered, outside, sampleHits, texts, samples, filmText: film.textContent };
}

(async () => {
  let failure = null;
  try {
    server = await chromium.launchServer({ headless: true, executablePath: CHROME });
    owned = server.process();
    const profileArg = owned.spawnargs.find(arg => arg.startsWith('--user-data-dir='));
    profilePath = profileArg ? profileArg.slice('--user-data-dir='.length) : null;
    console.log(JSON.stringify({ session: 'test_readability_basics_browser', controller: process.pid, browser: owned.pid, profile: profilePath }));
    browser = await chromium.connect(server.wsEndpoint());
    page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));

    for (const [key, spec] of Object.entries(LESSONS)) {
      await page.goto(BASE + 'foundations/' + key + '.html', { waitUntil: 'networkidle' });
      await page.locator('.mechanism-film').waitFor();
      await page.evaluate(() => document.fonts.ready);
      for (const variant of spec.variants) {
        await chooseScene(page, variant);
        await page.waitForTimeout(80);
        const max = Number(await page.locator('.mechanism-film .motion-scrub input').getAttribute('max'));
        for (const width of WIDTHS) {
          await page.setViewportSize({ width, height: width < 800 ? 900 : 1000 });
          await page.waitForTimeout(120);
          for (const raw of spec.positions) {
            if (raw > max) continue;
            await seek(page, raw);
            const g = await page.evaluate(geometry);
            assert.deepEqual(g.textHits, [], `${key} v${variant} ${width}px p${raw}: text/text overlap`);
            assert.deepEqual(g.covered, [], `${key} v${variant} ${width}px p${raw}: text covered by an object`);
            assert.deepEqual(g.outside, [], `${key} v${variant} ${width}px p${raw}: text outside stage`);
            assert.deepEqual(g.sampleHits, [], `${key} v${variant} ${width}px p${raw}: sample cards overlap`);

            if (key === 'tokens' && width === 320 && raw === 0) {
              const notes = g.texts.filter(t => /编号是查表索引|p1…p5 只是/.test(t.text));
              assert.equal(notes.length, 2, 'both token notes visible at 320');
              const gap = Math.min(...notes.map(n => n.y)) - Math.max(...notes.map(n => n.bottom));
              assert.ok(gap === gap, 'token note boxes measurable');
              const first = notes.find(n => /编号/.test(n.text)), second = notes.find(n => /p1/.test(n.text));
              assert.ok(second.y - first.bottom >= 10, `token notes need line spacing, got ${(second.y - first.bottom).toFixed(1)}`);
            }
            if (key === 'generation' && raw === 2) {
              for (const pct of ['60.0%', '30.0%', '10.0%']) assert.ok(g.filmText.includes(pct), `candidate probability ${pct}`);
            }
            if (key === 'metrics') {
              const values = g.samples.map(s => s.text.replace(' ×', ''));
              assert.equal(new Set(values).size, 20, 'all 20 samples present');
              assert.deepEqual(values.map(Number).sort((a, b) => a - b), Array.from({ length: 20 }, (_, i) => (i + 1) * 10));
              if (variant === 2 && raw === 4) {
                assert.match(g.filmText, /P95：第 19 项 = 190 ms/);
                assert.match(g.filmText, /成功 20 \/ 20 · 成功率 100%/);
              }
              if (variant === 3 && raw === 4) {
                assert.match(g.filmText, /P95：第 18 项 = 180 ms/);
                assert.match(g.filmText, /成功 18 \/ 20 · 成功率 90%/);
                assert.match(g.filmText, /190 ×/);
                assert.match(g.filmText, /200 ×/);
              }
            }
            if (key === 'deployment') {
              const roles = await page.locator('.mechanism-film [data-motion-id^=source-kv-]').evaluateAll(es => es.map(e => e.dataset.role));
              assert.deepEqual(roles, ['source', 'source'], `source KV keeps its role at p${raw}`);
              const source = Number(await page.locator('[data-motion-id=source-kv-0]').getAttribute('opacity'));
              if (raw === 0) {
                assert.equal(source, 0, 'source KV waits for the first step output');
                assert.match(g.filmText, /输入处理中/);
              } else {
                assert.equal(source, 1, `source KV appears once the first step completes at p${raw}`);
              }
              const state = await page.evaluate(() => {
                const req = document.querySelector('.mechanism-film [data-copies]');
                const total = document.querySelector('.mechanism-film [data-ready]');
                return {
                  copies: req?.dataset.copies, y1: req?.dataset.y1, gate: req?.dataset.gate,
                  copyProgress: Number(req?.dataset.copyProgress), y1Progress: Number(req?.dataset.y1Progress),
                  ready: total?.dataset.ready, available: total?.dataset.available,
                };
              });
              assert.equal(state.copies, raw >= 2.6 ? '2' : '0', `KV copy count at p${raw}`);
              assert.equal(state.y1, raw >= 3 ? '1' : '0', `y1 arrival at p${raw}`);
              assert.equal(state.gate, raw >= 3 ? 'open' : 'closed', `D gate at p${raw}`);
              assert.equal(state.ready, raw >= 3 ? 'true' : 'false', `data-ready at p${raw}`);
              if (raw === 2.5) {
                assert.equal(state.copyProgress, .926, 'both historical KV copies move together');
                assert.equal(state.y1Progress, 0, 'y1 copy has not started with the KV copies');
              }
              if (raw === 2.7) {
                assert.equal(state.copyProgress, 1, 'historical copies have arrived');
                assert.ok(state.y1Progress > 0 && state.y1Progress < 1, 'y1 is still independently in transit');
                assert.equal(state.available, '2', 'two history KV copies are available before y1');
              }
              if (state.copyProgress === 0) assert.match(g.filmText, /副本 0\/2 · 等待/);
              else if (raw < 2.55) assert.match(g.filmText, /副本在途 · y1 待发/);
              else if (raw < 2.6) assert.match(g.filmText, /副本在途 · y1 在途/);
              else if (raw < 3) assert.match(g.filmText, /副本 2\/2 · y1 在途/);
              else assert.match(g.filmText, /副本 2\/2 · y1 已到/);
              if (raw < 2.6) assert.match(g.filmText, /可用 0\/3/);
              if (raw >= 2.6 && raw < 3) assert.match(g.filmText, /可用 2\/3 · 等 y1/);
              if (raw >= 3 && raw < 4) assert.match(g.filmText, /可用 2\/3 · 计算 y1 中/);
              if (raw >= 4) assert.match(g.filmText, /可用 3\/3 · y1 KV 完成/);
              const dest = await page.locator('.mechanism-film [data-motion-id^=target-kv-]').evaluateAll(es => es.map(e => e.textContent));
              assert.deepEqual(dest, raw >= 2.6 ? ['副本 1', '副本 2'] : ['等待副本', '等待副本'], `destination slot state at p${raw}`);
              if (raw === 2.9) assert.ok(!g.texts.some(t => t.text === '等待 y1'), 'waiting ghost hidden once the input copy moves');
            }
          }
        }
      }
      await page.setViewportSize({ width: 1440, height: 1000 });
      // Reduced motion turns play/next into deterministic whole-step jumps, so
      // the transport controls can be checked without racing the tween.
      await page.emulateMedia({ reducedMotion: 'reduce' });
      await page.locator('[data-motion-action=reset]').click();
      await seek(page, 1.2);
      const before = await page.locator('.mechanism-film .motion-scrub input').inputValue();
      await page.locator('[data-motion-action=play]').click();
      await page.waitForTimeout(140);
      await page.locator('[data-motion-action=play]').click();
      const paused = await page.locator('.mechanism-film .motion-scrub input').inputValue();
      await page.waitForTimeout(180);
      assert.equal(await page.locator('.mechanism-film .motion-scrub input').inputValue(), paused, `${key}: pause holds position`);
      assert.ok(Number(paused) >= Number(before), `${key}: playback advances`);
      await page.locator('[data-motion-action=reset]').click();
      assert.equal(await page.locator('.mechanism-film .motion-scrub input').inputValue(), '0', `${key}: reset returns to start`);
      await page.locator('[data-motion-action=next]').click();
      assert.ok(Number(await page.locator('.mechanism-film .motion-scrub input').inputValue()) > 0, `${key}: next advances`);
      await page.locator('[data-motion-action=previous]').click();
      assert.equal(await page.locator('.mechanism-film .motion-scrub input').inputValue(), '0', `${key}: previous rewinds`);
      await page.emulateMedia({ reducedMotion: 'no-preference' });
    }
    assert.deepEqual(errors, [], 'no page errors');
    console.log(JSON.stringify({ ok: true, lessons: Object.keys(LESSONS), widths: WIDTHS }));
  } catch (error) {
    failure = error;
    console.error(error);
    process.exitCode = 1;
  } finally {
    try {
      await shutdown(failure ? 'failure' : 'complete');
    } catch (error) {
      console.error(error);
      process.exitCode = 1;
    }
  }
})();
