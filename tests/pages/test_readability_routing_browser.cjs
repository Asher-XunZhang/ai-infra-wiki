/* Focused readability regression for DP scenario 4 and EP scenarios 5/6.
 * Checks the three real defects: DP heading/rank vs the L1 copy and request,
 * EP return values vs the expert math, and the weighted formula vs E1/E2.
 * Also exercises both endpoints at five widths plus 1000x600.
 * Replicated copies of one value (e.g. two "L1 · ×2" replicas) are not treated
 * as errors when they are merely adjacent; only visible text coverage and
 * different return values are asserted. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const chrome = process.env.CHROMIUM_EXECUTABLE || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const base = process.env.OVERVIEW_TEST_URL || process.env.PREVIEW_URL || 'http://127.0.0.1:18803/';
const session = 'readability-routing-browser';
const WIDTHS = [320, 390, 768, 1440, 1994];
const FRAMES = [0.5, 1.5, 3.5, 4, 4.5];

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

(async () => {
  let failure = null;
  try {
    server = await chromium.launchServer({ headless: true, executablePath: chrome });
    owned = server.process();
    profilePath = owned.spawnargs.find(a => a.startsWith('--user-data-dir='))?.slice('--user-data-dir='.length) || null;
    console.log(JSON.stringify({ session, controller: process.pid, browser: owned.pid, profile: profilePath }));
    browser = await chromium.connect(server.wsEndpoint());
    page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));

    const hit = (a, b) => Math.min(a.right, b.right) - Math.max(a.left, b.left) > 2.5 &&
      Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 2.5;

    async function open(index, p) {
      await page.goto(base + 'foundations/parallel.html?mv=' + index + '&ms=' + p, { waitUntil: 'networkidle' });
      await page.locator('.mechanism-film').waitFor();
      await page.evaluate(() => document.fonts.ready);
      await page.waitForTimeout(120);
    }
    async function seek(p) {
      await page.locator('.motion-scrub input').evaluate((e, v) => { e.value = String(v); e.dispatchEvent(new Event('input', { bubbles: true })); }, p);
      await page.waitForTimeout(70);
    }
    async function snapshot() {
      return page.locator('.mechanism-film svg').evaluate(svg => {
        const shown = e => {
          for (let n = e; n && n !== svg; n = n.parentElement) {
            const o = n.getAttribute('opacity');
            if (o !== null && Number(o) < .02) return false;
          }
          return Number(getComputedStyle(e).opacity || 1) > .02;
        };
        const box = e => { const r = e.getBoundingClientRect(); return { x: r.x, y: r.y, right: r.right, bottom: r.bottom, w: r.width, h: r.height }; };
        const labels = [...svg.querySelectorAll('text.teach-heading,text.teach-small,text.teach-number')]
          .filter(e => e.textContent.trim() && shown(e) && !e.closest('g[data-motion-id]'))
          .map(e => ({ text: e.textContent.trim(), cls: e.getAttribute('class') || '', ...box(e) }));
        const tokens = [...svg.querySelectorAll('g[data-motion-id]')]
          .filter(shown).map(g => ({ id: g.dataset.motionId, text: g.textContent.trim(), ...box(g) }));
        const tokenText = new Set(tokens.map(t => t.text.trim()));
        return { labels, tokens, tokenText: [...tokenText] };
      });
    }
    for (const width of WIDTHS) {
      await page.setViewportSize({ width, height: 1100 });
      await page.waitForTimeout(120);
      for (const index of [4, 5, 6]) {
        await open(index, 0);
        for (const p of FRAMES) {
          await seek(p);
          const { labels, tokens } = await snapshot();
          if (index === 4) {
            const heads = labels.filter(l => /^完整副本 [AB]$|^P\d \/ Rank/.test(l.text));
            const moving = tokens.filter(t => /^dp-L1-|^dp-R1$|^dp-R2$/.test(t.id));
            for (const h of heads) for (const t of moving) {
              assert.ok(!hit(h, t), `DP width ${width} p${p}: ${h.text} is covered by ${t.id} (${t.text})`);
            }
            assert.ok(labels.some(l => /^完整副本 [AB]$/.test(l.text) || /^P\d \/ Rank/.test(l.text)), `DP width ${width} p${p}: an identity label stays visible`);
          } else {
            const returns = tokens.filter(t => t.id.startsWith('ep-return-'));
            const math = labels.filter(l => /^E[0-3]\(t1\)：/.test(l.text));
            const formula = labels.find(l => / × 0\.75 \+ /.test(l.text));
            for (let i = 0; i < returns.length; i++) for (let j = i + 1; j < returns.length; j++) {
              assert.ok(!hit(returns[i], returns[j]), `EP${index} width ${width} p${p}: return ${returns[i].text} and ${returns[j].text} overlap`);
            }
            for (const m of math) for (const r of returns) {
              assert.ok(!hit(m, r), `EP${index} width ${width} p${p}: ${m.text} is covered by return ${r.text}`);
            }
            if (formula) for (const m of math) {
              assert.ok(!hit(formula, m), `EP${index} width ${width} p${p}: formula covers ${m.text}`);
            }
            if (p >= 4) {
              const expected = index === 5 ? ['2', '6'] : ['2', '4'];
              assert.deepEqual(returns.map(r => r.text), expected, `EP${index} width ${width} p${p}: return identity/values`);
              assert.ok(labels.some(l => /E[0-3]：0\.75/.test(l.text)), `EP${index} width ${width} p${p}: routing weight stays visible`);
            }
          }
        }
        // Endpoint snapshots: first frame and final frame must paint without a page error.
        for (const p of [0, 5]) {
          await seek(p);
          const state = await page.locator('.mechanism-film').evaluate(film => ({
            position: Number(film.dataset.position),
            status: film.querySelector('.motion-status')?.textContent || '',
            svg: !!film.querySelector('svg'),
            visible: film.getBoundingClientRect().height > 0
          }));
          assert.ok(state.svg && state.visible, `scene ${index} width ${width} p${p}: film is painted`);
          assert.ok(Math.abs(state.position - p) < .01, `scene ${index} width ${width}: endpoint p${p} reached`);
        }
      }
    }

    // Short 1000x600 viewport: both endpoints still reach the film and no crash.
    await page.setViewportSize({ width: 1000, height: 600 });
    for (const index of [4, 5, 6]) {
      await open(index, 0);
      for (const p of [0, 5]) {
        await seek(p);
        const visible = await page.locator('.mechanism-film').evaluate((film, want) =>
          film.getBoundingClientRect().height > 0 && !!film.querySelector('svg') &&
          Math.abs(Number(film.dataset.position) - want) < .01, p);
        assert.ok(visible, `scene ${index} 1000x600 endpoint p${p}`);
      }
    }
    assert.deepEqual(errors, [], 'no page errors');
    console.log('PASS rolling readability: DP heading/rank clear of L1 and requests; EP return values clear of expert math; weighted formula clear of E1/E2; return identities distinct; endpoints at 320/390/768/1440/1994 and 1000x600.');
  } catch (error) {
    failure = error; console.error(error); process.exitCode = 1;
  } finally {
    try { await shutdown(failure ? 'failure' : 'complete'); }
    catch (error) { console.error(error); process.exitCode = 1; }
  }
})();
