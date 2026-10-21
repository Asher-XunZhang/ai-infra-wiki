/* Mid-transition readability regression for the chapter-1 communication
 * course. The reported defects were paused states where a moving copy covered
 * a static caption or another copy, so every evaluated state comes from a real
 * seek on the built page and uses rendered SVG boxes. A token border may brush
 * another token while copies separate; only text collisions and substantial
 * different-value body overlap are treated as readability defects. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const choose = require('./choose_motion_scene.cjs');
const M = require('../../pages/foundations/communication-model.js');

const BASE = process.env.OVERVIEW_TEST_URL || 'http://127.0.0.1:18802/';
const CHROME = process.env.CHROMIUM_EXECUTABLE || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const WIDTHS = [320, 390, 768, 1440, 1994];
const HEIGHTS = [1000, 1100];
const CASES = [
  { scenes: [8, 9, 10, 11, 12, 13, 14], positions: [.5, 1.5, 1.8, 2.5] },
  { scenes: [27], positions: [0, 2.5, 3.5] },
  { scenes: [29, 30, 31, 32], positions: [1.5, 1.8, 2, 2.2, 2.5, 4] },
  { scenes: [15], positions: [1.8, 2.2, 2.5, 3] },
];

let server;
let browser;
let page;
let owned;
let profilePath;
let shuttingDown;

const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const pidState = pid => {
  try { process.kill(pid, 0); return 'alive'; }
  catch (error) { return error.code === 'ESRCH' ? 'gone' : `error:${error.code || error.message}`; }
};
function profileProcesses() {
  if (!profilePath) return [];
  try {
    return execFileSync('pgrep', ['-fl', profilePath], { encoding: 'utf8' }).trim().split('\n')
      .filter(Boolean).map(line => line.match(/^(\d+)\s+(.*)$/)).filter(Boolean)
      .map(m => ({ pid: Number(m[1]), command: m[2] })).filter(e => pidState(e.pid) !== 'gone');
  } catch (error) { if (error.status === 1) return []; throw error; }
}
async function waitGone(pid, timeout) {
  const end = Date.now() + timeout;
  let state = pidState(pid);
  while (state !== 'gone' && Date.now() < end) { await delay(80); state = pidState(pid); }
  return state;
}
function signal(pid, sig) {
  try { process.kill(pid, sig); return null; }
  catch (error) { return error.code === 'ESRCH' ? null : error.code || error.message; }
}
function ownedProfile(candidate) {
  if (!candidate || !/playwright/i.test(path.basename(candidate))) return false;
  const roots = [path.resolve(os.tmpdir())];
  try { roots.push(fs.realpathSync(os.tmpdir())); } catch {}
  const files = [path.resolve(candidate)];
  try { files.push(fs.realpathSync(candidate)); } catch {}
  return files.some(file => roots.some(root => file.startsWith(root + path.sep)));
}
async function shutdown(reason) {
  if (shuttingDown) return shuttingDown;
  shuttingDown = (async () => {
    const failures = [];
    await page?.close().catch(() => {});
    await browser?.close().catch(() => {});
    await server?.close().catch(() => {});
    let state = owned?.pid ? await waitGone(owned.pid, 1500) : 'not-created';
    for (const sig of ['SIGTERM', 'SIGKILL']) {
      if (state === 'gone') break;
      if (!profileProcesses().some(e => e.pid === owned.pid)) { failures.push(`pid ${owned.pid} is not our browser`); break; }
      const error = signal(owned.pid, sig); if (error) failures.push(`browser ${sig}: ${error}`);
      state = await waitGone(owned.pid, sig === 'SIGTERM' ? 1200 : 800);
    }
    if (owned?.pid && state !== 'gone') failures.push(`browser pid ${owned.pid} remains (${state})`);
    let remaining = profileProcesses();
    for (const sig of ['SIGTERM', 'SIGKILL']) {
      for (const entry of remaining) { const error = signal(entry.pid, sig); if (error) failures.push(`profile ${entry.pid} ${sig}: ${error}`); }
      await delay(180); remaining = profileProcesses(); if (!remaining.length) break;
    }
    if (remaining.length) failures.push(`profile processes remain: ${remaining.map(e => e.pid).join(',')}`);
    let removed = !profilePath || !fs.existsSync(profilePath);
    if (profilePath && !removed && ownedProfile(profilePath) && !remaining.length) {
      try { fs.rmSync(profilePath, { recursive: true, force: true }); } catch (error) { failures.push(`profile: ${error.code || error.message}`); }
      removed = !fs.existsSync(profilePath);
    }
    console.log(JSON.stringify({ event: 'cleanup', reason, controller: process.pid, browser: owned?.pid ?? null, browserState: state, profile: profilePath, profileRemoved: removed, remaining: remaining.length }));
    if (failures.length) throw new Error(`cleanup incomplete (${reason}): ${failures.join('; ')}`);
  })();
  return shuttingDown;
}
for (const [sig, code] of [['SIGINT', 130], ['SIGTERM', 143]]) process.once(sig, () => shutdown(sig).then(() => process.exit(code)).catch(e => { console.error(e); process.exit(1); }));

async function seek(position) {
  await page.locator('.mechanism-film .motion-scrub input').evaluate((el, p) => {
    el.value = String(p); el.dispatchEvent(new Event('input', { bubbles: true }));
  }, position);
  await page.waitForTimeout(35);
}

function geometry() {
  const svg = document.querySelector('.mechanism-film svg');
  const visible = e => {
    for (let a = e; a && a !== svg; a = a.parentElement) {
      const s = getComputedStyle(a);
      if (s.display === 'none' || s.visibility === 'hidden' || Number(s.opacity) < .02) return false;
    }
    return true;
  };
  const box = e => { const b = e.getBoundingClientRect(); return { x: b.x, y: b.y, r: b.right, bt: b.bottom }; };
  const texts = [...svg.querySelectorAll('text')].filter(e => e.textContent.trim() && visible(e)).map(e => ({
    text: e.textContent.trim(), id: e.closest('[data-motion-id]')?.dataset.motionId || '', ...box(e),
  }));
  const tokens = [...svg.querySelectorAll('g[data-motion-id] > rect')].filter(visible).map(e => ({
    id: e.parentElement.dataset.motionId, text: e.parentElement.querySelector('text')?.textContent.trim() || '', ...box(e),
  }));
  const overlap = (a, b, min = 1) => {
    const w = Math.min(a.r, b.r) - Math.max(a.x, b.x);
    const h = Math.min(a.bt, b.bt) - Math.max(a.y, b.y);
    return w > min && h > min ? { w, h, area: w * h } : null;
  };
  const hit = (a, b) => !!overlap(a, b);
  const textHits = [];
  for (let i = 0; i < texts.length; i++) for (let j = i + 1; j < texts.length; j++) {
    if (texts[i].text !== texts[j].text && hit(texts[i], texts[j])) textHits.push(`${texts[i].text} × ${texts[j].text}`);
  }
  const tokenHits = [];
  const size = t => (t.r - t.x) * (t.bt - t.y);
  for (let i = 0; i < tokens.length; i++) for (let j = i + 1; j < tokens.length; j++) {
    // Copies of one value may briefly overlap as they separate. Border contact
    // between different values is also harmless while both labels stay clear;
    // flag only a substantial body collision, not a few overlapping pixels.
    if (tokens[i].text === tokens[j].text) continue;
    const shared = overlap(tokens[i], tokens[j], 2);
    if (shared && shared.area / Math.min(size(tokens[i]), size(tokens[j])) >= .5) {
      tokenHits.push(`${tokens[i].id} × ${tokens[j].id}`);
    }
  }
  const covered = [];
  for (const t of texts) for (const a of tokens) {
    if (a.id === t.id || a.text === t.text) continue;
    // A value replicated to several ranks may sit on an identical label; that
    // is normal copying. A token edge may also brush the end of a caption, so
    // require a visible bite (roughly a character wide) or a quarter of a short
    // label before treating the text as covered.
    const shared = overlap(t, a);
    const textArea = (t.r - t.x) * (t.bt - t.y);
    if (shared && shared.h >= 6 && (shared.w >= 12 || shared.area >= textArea * .25)) {
      covered.push(`${t.text} ← ${a.id}`); break;
    }
  }
  const stage = svg.getBoundingClientRect();
  const outside = texts.filter(t => t.x < stage.x - 2 || t.r > stage.right + 2 || t.y < stage.y - 2 || t.bt > stage.bottom + 2).map(t => t.text);
  return { textHits, tokenHits, covered, outside };
}

(async () => {
  let failure = null;
  try {
    server = await chromium.launchServer({ headless: true, executablePath: CHROME });
    owned = server.process();
    profilePath = owned.spawnargs.find(a => a.startsWith('--user-data-dir='))?.slice('--user-data-dir='.length) || null;
    console.log(JSON.stringify({ session: 'readability-communication', controller: process.pid, browser: owned.pid, profile: profilePath }));
    browser = await chromium.connect(server.wsEndpoint());
    page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    const scenes = M.scenes();
    await page.goto(BASE + 'foundations/communication.html', { waitUntil: 'networkidle' });
    await page.locator('.mechanism-film').waitFor();
    await page.evaluate(() => document.fonts.ready);
    for (const height of HEIGHTS) for (const width of WIDTHS) {
      await page.setViewportSize({ width, height });
      for (const { scenes: ids, positions } of CASES) for (const index of ids) {
        await choose(page, index);
        await page.waitForTimeout(50);
        for (const p of positions) {
          await seek(p);
          const g = await page.evaluate(geometry);
          const tag = `${index} ${width}x${height} p${p}`;
          assert.deepEqual(g.textHits, [], `${tag}: text/text overlap`);
          assert.deepEqual(g.tokenHits, [], `${tag}: token/token overlap`);
          assert.deepEqual(g.covered, [], `${tag}: text covered by a token`);
          assert.deepEqual(g.outside, [], `${tag}: text outside stage`);
          if ([8, 9, 10, 11, 12, 13, 14].includes(index) && p === 2.5) {
            const state = await page.locator('.mechanism-film svg').evaluate(svg => ({
              copies: [...svg.querySelectorAll('[data-motion-id^="comm-contribution-"][data-arrived="true"]')].length,
              text: svg.textContent,
            }));
            assert.equal(state.copies, scenes[index].data.math.reduce((n, row) => n + row.sources.length, 0), `${tag}: contribution copies reached the matrix`);
            assert.match(state.text, /=/, `${tag}: arithmetic remains visible`);
          }
          if ([29, 30, 31, 32].includes(index) && p === 4) {
            const values = await page.locator('.mechanism-film [data-motion-id^="matrix-result-"] text').allTextContents();
            assert.deepEqual(values, scenes[index].data.output.flat().map(String), `${tag}: output values preserved`);
            const inputs = await page.locator('.mechanism-film [data-motion-id^="matrix-input-"] text').allTextContents();
            assert.deepEqual(inputs, scenes[index].data.input.flat().map(String), `${tag}: original inputs preserved`);
          }
          if (index === 15 && p === 3) {
            const values = await page.locator('.mechanism-film [data-motion-id^="comm-copy-"]:not([opacity="0.000"]) text').allTextContents();
            assert.deepEqual(values.sort(), scenes[index].data.output.flat().slice().sort(), `${tag}: AllToAll outputs preserved`);
          }
        }
      }
    }
    assert.deepEqual(errors, [], 'no page errors');
    console.log(JSON.stringify({ ok: true, widths: WIDTHS, heights: HEIGHTS, states: CASES.reduce((n, c) => n + c.scenes.length * c.positions.length, 0) * WIDTHS.length * HEIGHTS.length }));
  } catch (error) {
    failure = error; console.error(error); process.exitCode = 1;
  } finally {
    try { await shutdown(failure ? 'failure' : 'complete'); }
    catch (error) { console.error(error); process.exitCode = 1; }
  }
})();
