/* Short-window regression for the shared foundation camera.
 *
 * The camera compensates type with a cap, so the stage must not collapse to a
 * vh-only box on a 600px-tall laptop. These checks use settled endpoints, not
 * mid-transition flying copies: a temporary shrunken copy is not body text. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const choose = require('./choose_motion_scene.cjs');

const BASE = process.env.OVERVIEW_TEST_URL || 'http://127.0.0.1:18765/';
const CHROME = process.env.CHROMIUM_EXECUTABLE || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const CASES = [
  { width: 1000, height: 600, course: 'communication', variant: 32, position: 0 },
  { width: 1000, height: 600, course: 'parallel', variant: 5, position: 0 },
  { width: 1000, height: 600, course: 'parallel', variant: 2, position: 1.5 },
  { width: 1000, height: 600, course: 'attention', variant: 1, position: 3 },
  { width: 1000, height: 600, course: 'cache', variant: 0, position: 0 },
  { width: 1000, height: 600, course: 'batching', variant: 0, position: 4 },
  { width: 390, height: 700, course: 'communication', variant: 32, position: 0 },
  { width: 390, height: 700, course: 'parallel', variant: 5, position: 0 },
  { width: 390, height: 700, course: 'batching', variant: 0, position: 4 },
];

let server;
let browser;
let page;
let owned;
let profile;
let shuttingDown;

function pidGone(pid) {
  try { process.kill(pid, 0); return false; } catch (error) { return error.code === 'ESRCH'; }
}

async function shutdown() {
  if (shuttingDown) return shuttingDown;
  shuttingDown = (async () => {
    await page?.close().catch(() => {});
    await browser?.close().catch(() => {});
    await server?.close().catch(() => {});
    if (owned?.pid) {
      for (let i = 0; i < 20 && !pidGone(owned.pid); i++) await new Promise(r => setTimeout(r, 100));
      assert.ok(pidGone(owned.pid), `browser ${owned.pid} still alive`);
    }
    if (profile && /playwright/i.test(path.basename(profile)) && profile.startsWith(fs.realpathSync(os.tmpdir()))) {
      fs.rmSync(profile, { recursive: true, force: true });
    }
    console.log(JSON.stringify({ event: 'cleanup', browser: owned?.pid ?? null, browserGone: !owned?.pid || pidGone(owned.pid), profile, profileGone: !profile || !fs.existsSync(profile) }));
  })();
  return shuttingDown;
}

for (const [signal, code] of [['SIGINT', 130], ['SIGTERM', 143]]) {
  process.once(signal, () => shutdown().finally(() => process.exit(code)));
}

async function seek(position) {
  await page.locator('.mechanism-film .motion-scrub input').evaluate((el, p) => {
    el.value = String(p);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, position);
  await page.waitForTimeout(700);
}

function measure() {
  const film = document.querySelector('.mechanism-film');
  const svg = film.querySelector('svg');
  const camera = film.querySelector('.motion-camera');
  const stage = film.querySelector('.motion-stage');
  const tools = film.querySelector('.motion-camera-tools');
  const visible = e => {
    for (let a = e; a && a !== svg.parentElement; a = a.parentElement) {
      const s = getComputedStyle(a);
      if (s.display === 'none' || s.visibility === 'hidden' || Number(s.opacity) < .45) return false;
    }
    return true;
  };
  const texts = [...svg.querySelectorAll('text')].filter(e => e.textContent.trim() && visible(e)).map(e => {
    const b = e.getBoundingClientRect(), ctm = e.getScreenCTM();
    return { text: e.textContent, px: parseFloat(getComputedStyle(e).fontSize) * Math.hypot(ctm.a, ctm.b), box: b };
  });
  const overlaps = [];
  for (let i = 0; i < texts.length; i++) for (let j = i + 1; j < texts.length; j++) {
    const a = texts[i], b = texts[j];
    if (a.text === b.text) continue;
    const dx = Math.min(a.box.right, b.box.right) - Math.max(a.box.left, b.box.left);
    const dy = Math.min(a.box.bottom, b.box.bottom) - Math.max(a.box.top, b.box.top);
    if (dx > 2 && dy > 2) overlaps.push(`${a.text} × ${b.text}`);
  }
  const stageBox = stage.getBoundingClientRect();
  const cameraBox = camera.getBoundingClientRect();
  const clipped = texts.filter(t =>
    t.box.left < cameraBox.left - 1 || t.box.right > cameraBox.right + 1 ||
    t.box.top < cameraBox.top - 1 || t.box.bottom > cameraBox.bottom + 1
  ).map(t => t.text);
  const controls = [...tools.querySelectorAll('button')].map(button => {
    const box = button.getBoundingClientRect();
    return {
      action: button.dataset.motionZoomIn !== undefined ? 'zoom-in'
        : button.dataset.motionZoomOut !== undefined ? 'zoom-out'
          : button.dataset.motionZoomFit !== undefined ? 'zoom-fit' : 'unknown',
      disabled: button.disabled,
      visible: box.width > 0 && box.height > 0 && getComputedStyle(button).display !== 'none' && getComputedStyle(button).visibility !== 'hidden',
      inside: box.left >= stageBox.left - 1 && box.right <= stageBox.right + 1 &&
        box.top >= stageBox.top - 1 && box.bottom <= stageBox.bottom + 1,
    };
  });
  return {
    stage: stageBox,
    camera: cameraBox,
    textCount: texts.length,
    minPx: Math.min(...texts.map(t => t.px)),
    under9: texts.filter(t => t.px < 9).map(t => `${t.text}:${t.px.toFixed(1)}`),
    overlaps,
    clipped,
    controls,
  };
}

function assertReadable(result, label) {
  assert.ok(result.textCount > 0, `${label}: no settled text measured`);
  assert.deepEqual(result.under9, [], `${label}: type below 9px`);
  assert.deepEqual(result.overlaps, [], `${label}: settled text overlap`);
  assert.deepEqual(result.clipped, [], `${label}: settled text clipped by the camera`);
  assert.deepEqual(result.controls.map(c => c.action).sort(), ['zoom-fit', 'zoom-in', 'zoom-out'], `${label}: zoom controls missing`);
  for (const control of result.controls) {
    assert.ok(control.visible && control.inside, `${label}: ${control.action} hidden or clipped`);
  }
  assert.equal(result.controls.find(c => c.action === 'zoom-in').disabled, false, `${label}: zoom-in unavailable`);
  assert.equal(result.controls.find(c => c.action === 'zoom-fit').disabled, false, `${label}: zoom-fit unavailable`);
}

(async () => {
  let failure;
  try {
    server = await chromium.launchServer({ headless: true, executablePath: CHROME });
    owned = server.process();
    const profileArg = owned.spawnargs.find(arg => arg.startsWith('--user-data-dir='));
    profile = profileArg && profileArg.slice('--user-data-dir='.length);
    console.log(JSON.stringify({ session: 'test_readability_short_viewport_browser', controller: process.pid, browser: owned.pid, profile }));
    browser = await chromium.connect(server.wsEndpoint());
    page = await browser.newPage({ viewport: { width: 1000, height: 900 } });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));

    for (const item of CASES) {
      await page.setViewportSize({ width: item.width, height: item.height });
      await page.goto(BASE + 'foundations/' + item.course + '.html', { waitUntil: 'networkidle' });
      await page.locator('.mechanism-film svg').waitFor();
      await choose(page, item.variant);
      await page.waitForTimeout(350);
      await seek(item.position);
      const result = await page.evaluate(measure);
      const label = `${item.course} v${item.variant} ${item.width}x${item.height}`;
      assertReadable(result, label);
      console.log(JSON.stringify({ label, stage: Math.round(result.stage.height), minPx: +result.minPx.toFixed(2) }));
    }

    // A normal-height window keeps the same readable settled state.
    await page.setViewportSize({ width: 1000, height: 1000 });
    await page.goto(BASE + 'foundations/communication.html', { waitUntil: 'networkidle' });
    await choose(page, 32);
    await seek(0);
    const normal = await page.evaluate(measure);
    assertReadable(normal, 'communication v32 1000x1000');

    // Zoom in/out/reset controls use the markup's data-motion-zoom-* hooks.
    const camera = page.locator('.mechanism-film .motion-camera');
    const zoomIn = page.locator('.mechanism-film [data-motion-zoom-in]');
    const zoomFit = page.locator('.mechanism-film [data-motion-zoom-fit]');
    const zoomValue = page.locator('.mechanism-film [data-motion-zoom-value]');
    await zoomIn.click();
    assert.equal(await camera.getAttribute('data-zoom'), '1.25');
    assert.equal(await zoomValue.textContent(), '125%');
    // A mouse click leaves focus on the button; the camera deliberately ignores
    // key events from buttons, so return focus before testing the '+' shortcut.
    assert.equal(await zoomIn.evaluate(el => el === document.activeElement), true);
    await camera.focus();
    assert.equal(await camera.evaluate(el => el === document.activeElement), true);
    await page.keyboard.press('+');
    assert.equal(await camera.getAttribute('data-zoom'), '1.50');
    assert.equal(await zoomValue.textContent(), '150%');
    await zoomFit.click();
    assert.equal(await camera.getAttribute('data-zoom'), '1.00');
    assert.equal(await zoomValue.textContent(), '100%');
    assert.equal(await camera.evaluate(el => el.classList.contains('is-zoomed')), false);

    assert.deepEqual(errors, [], 'no page errors');
    console.log(JSON.stringify({ ok: true, cases: CASES.length, normalStage: Math.round(normal.stage.height) }));
  } catch (error) {
    failure = error;
    console.error(error);
    process.exitCode = 1;
  } finally {
    try { await shutdown(); } catch (error) { console.error(error); process.exitCode = 1; }
  }
})();
