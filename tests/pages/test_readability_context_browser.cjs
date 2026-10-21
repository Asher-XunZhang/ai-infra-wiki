/* Targeted regression for the mid-flight PCP / DCP context visuals.
 *
 * PCP: eight moving Queries must stay separable at p0.5 (including the zigzag
 * ownership swap), retain their rank at rest, and return as eight distinct O
 * tiles at p3.5.
 * DCP: the four KV shards must be traceable while moving in (p0.5) without
 * covering each other or the Query at 320/390; the final Z-weighted value and
 * weights stay visible at p4.
 * Identical text copies are allowed to touch; different values and titles are
 * not. Also runs the same checks at a 1000x600 window.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const choose = require('./choose_motion_scene.cjs');

const BASE = process.env.OVERVIEW_TEST_URL || 'http://127.0.0.1:18765/';
const CHROME = process.env.CHROMIUM_EXECUTABLE || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const WIDTHS = [320, 390, 768, 1440, 1994];

let server, browser, page, owned, profile, shuttingDown;

function pidGone(pid) {
  try { process.kill(pid, 0); return false; } catch (error) { return error.code === 'ESRCH'; }
}

async function shutdown() {
  if (shuttingDown) return shuttingDown;
  shuttingDown = (async () => {
    await page?.close().catch(() => {});
    await browser?.close().catch(() => {});
    await server?.close().catch(() => {});
    if (owned?.pid) for (let i = 0; i < 20 && !pidGone(owned.pid); i++) await new Promise(r => setTimeout(r, 100));
    if (profile && /playwright/i.test(path.basename(profile)) && profile.startsWith(fs.realpathSync(os.tmpdir()))) {
      fs.rmSync(profile, { recursive: true, force: true });
    }
    console.log(JSON.stringify({
      event: 'cleanup',
      browser: owned?.pid ?? null,
      browserGone: !owned?.pid || pidGone(owned.pid),
      profile,
      profileGone: !profile || !fs.existsSync(profile),
    }));
    if (owned?.pid && !pidGone(owned.pid)) process.exitCode = 1;
  })();
  return shuttingDown;
}

for (const [signal, code] of [['SIGINT', 130], ['SIGTERM', 143]]) {
  process.once(signal, () => shutdown().finally(() => process.exit(code)));
}

async function seek(position) {
  await page.locator('.mechanism-film .motion-scrub input').evaluate((el, p) => {
    el.value = String(p); el.dispatchEvent(new Event('input', { bubbles: true }));
  }, position);
  await page.waitForTimeout(80);
}

async function open(width, height = 1000) {
  await page.setViewportSize({ width, height });
  await page.waitForTimeout(120);
}

function geometry() {
  const svg = document.querySelector('.mechanism-film svg');
  const vis = e => {
    for (let a = e; a && a !== svg.parentElement; a = a.parentElement) {
      const s = getComputedStyle(a);
      if (s.display === 'none' || s.visibility === 'hidden' || Number(s.opacity) < .02) return false;
    }
    return true;
  };
  const box = e => { const b = e.getBoundingClientRect(); return { x: b.left, y: b.top, r: b.right, b: b.bottom }; };
  const hit = (a, b) => Math.min(a.r, b.r) - Math.max(a.x, b.x) > 1 && Math.min(a.b, b.b) - Math.max(a.y, b.y) > 1;
  const groups = [...svg.querySelectorAll('g[data-motion-id]')].filter(vis).map(g => ({
    id: g.dataset.motionId, text: g.textContent.trim(), ...box(g),
  }));
  const texts = [...svg.querySelectorAll('text')]
    .filter(e => e.textContent.trim() && vis(e) && !e.closest('g[data-motion-id]'))
    .map(e => ({ text: e.textContent.trim(), ...box(e) }));
  const tokenHits = [];
  for (let i = 0; i < groups.length; i++) for (let j = i + 1; j < groups.length; j++) {
    if (groups[i].text === groups[j].text) continue; // same-value copies may separate
    if (hit(groups[i], groups[j])) tokenHits.push(`${groups[i].id} × ${groups[j].id}`);
  }
  const covered = [];
  for (const t of texts) for (const g of groups) {
    if (hit(t, g)) { covered.push(`${t.text} ← ${g.id}`); break; }
  }
  const textHits = [];
  for (let i = 0; i < texts.length; i++) for (let j = i + 1; j < texts.length; j++) {
    if (texts[i].text !== texts[j].text && hit(texts[i], texts[j])) textHits.push(`${texts[i].text} × ${texts[j].text}`);
  }
  return { tokenHits, covered, textHits, ids: groups.map(g => g.id) };
}

async function check(label) {
  const g = await page.evaluate(geometry);
  assert.deepEqual(g.tokenHits, [], `${label}: different tokens overlap`);
  assert.deepEqual(g.covered, [], `${label}: token covers a caption/title`);
  assert.deepEqual(g.textHits, [], `${label}: different captions overlap`);
  return g;
}

(async () => {
  try {
    server = await chromium.launchServer({ headless: true, executablePath: CHROME });
    owned = server.process();
    profile = owned.spawnargs.find(a => a.startsWith('--user-data-dir='))?.slice('--user-data-dir='.length) || null;
    console.log(JSON.stringify({ session: 'readability-context', controller: process.pid, browser: owned.pid, profile }));
    browser = await chromium.connect(server.wsEndpoint());
    page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(BASE + 'foundations/parallel.html', { waitUntil: 'networkidle' });
    await page.locator('.mechanism-film svg').waitFor();
    await page.evaluate(() => document.fonts.ready);

    for (const width of WIDTHS) {
      await open(width);
      for (const index of [8, 9]) {
        await choose(page, index);
        await page.waitForTimeout(80);
        await seek(.5);
        const mid = await check(`PCP v${index} ${width}px p0.5`);
        assert.equal(mid.ids.filter(id => id.startsWith('pcp-query-')).length, 8, `PCP v${index} ${width}px: eight moving Queries`);
        await seek(1.2);
        const owners = await page.evaluate(() => {
          const svg = document.querySelector('.mechanism-film svg');
          const m = (svg.getAttribute('viewBox') || '').startsWith('0 0 360 ');
          const q = [...svg.querySelectorAll('g[data-motion-id^="pcp-query-"]')].map(g => ({
            id: g.dataset.motionId, x: Number(g.dataset.x), rank: /^Q([1-8])$/.test(g.textContent.trim()) ? Number(g.textContent.trim().slice(1)) - 1 : -1,
          }));
          return { m, q, x: [18, m ? 190 : 382], w: m ? 152 : 320 };
        });
        // Read the scenario ranks from its visible label instead of trusting geometry.
        const ranks = await page.evaluate(i => window.TeachingLessons['f-parallel'][i].data.ranks, index);
        for (const q of owners.q) {
          const expected = ranks[0].includes(q.rank) ? 0 : 1;
          assert.ok(q.x >= owners.x[expected] + 8 && q.x <= owners.x[expected] + owners.w - 8,
            `PCP v${index} ${width}px: ${q.id} lost rank ${expected}`);
        }
        assert.deepEqual(ranks.map(r => r.length), [4, 4], 'PCP keeps four Queries per rank');
        await seek(3.5);
        const done = await check(`PCP v${index} ${width}px p3.5`);
        const outputs = await page.locator('.mechanism-film [data-motion-id^="pcp-output-"] text').allTextContents();
        assert.deepEqual(outputs.sort(), ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8'], `PCP v${index} ${width}px: eight output labels`);
        assert.equal(done.ids.filter(id => id.startsWith('pcp-output-')).length, 8, `PCP v${index} ${width}px: eight distinct O tiles`);
      }

      for (const index of [10, 11]) {
        await choose(page, index);
        await page.waitForTimeout(80);
        await seek(.5);
        const mid = await check(`DCP v${index} ${width}px p0.5`);
        assert.equal(mid.ids.filter(id => id.startsWith('dcp-kv-')).length, 4, `DCP v${index} ${width}px: four KV shards visible`);
        assert.ok(mid.ids.includes('dcp-query') && mid.ids.includes('dcp-query-copy'), `DCP v${index} ${width}px: Query and copy visible`);
        await seek(4);
        await check(`DCP v${index} ${width}px p4`);
        const final = await page.evaluate(i => {
          const svg = document.querySelector('.mechanism-film svg');
          return { text: svg.textContent, data: window.TeachingLessons['f-parallel'][i].data };
        }, index);
        assert.match(final.text, new RegExp(`全局 O = .* = ${final.data.result}`), `DCP v${index} ${width}px: Z-weighted result`);
        assert.match(final.text, new RegExp(`全局分母：${final.data.sums[0]} \\+ ${final.data.sums[1]} = ${final.data.sums[0] + final.data.sums[1]}`), `DCP v${index} ${width}px: global denominator`);
        assert.ok(final.text.includes(`片 0：${Math.round(final.data.weights[0] * 100)}%`), `DCP v${index} ${width}px: weight 0`);
        assert.ok(final.text.includes(`片 1：${Math.round(final.data.weights[1] * 100)}%`), `DCP v${index} ${width}px: weight 1`);
      }
    }

    await open(1000, 600);
    for (const [index, positions] of [[8, [.5, 3.5]], [9, [.5, 3.5]], [10, [.5, 4]], [11, [.5, 4]]]) {
      await choose(page, index);
      await page.waitForTimeout(80);
      for (const p of positions) {
        await seek(p);
        await check(`1000x600 v${index} p${p}`);
      }
    }

    assert.deepEqual(errors, [], 'no page errors');
    console.log(`PASS context readability: PCP v8/9 and DCP v10/11 at ${WIDTHS.join('/')} and 1000x600.`);
  } catch (error) {
    console.error(error);
    process.exitCode = 1;
  } finally {
    await shutdown().catch(error => { console.error(error); process.exitCode = 1; });
  }
})();
