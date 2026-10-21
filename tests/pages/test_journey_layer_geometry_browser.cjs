/* Geometry regression for generation.html's optional layer-by-layer figure.
 * The moving packet is authored at x=0, so its translate origin must be the
 * packet center. This checks the actual frame/text boxes against every layer
 * band and against the output row's heading/original tokens, not just whether
 * text stayed inside the SVG viewBox. The 4.55–5 return must lift above the
 * output row instead of crossing its heading or the original y1. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const BASE = process.env.OVERVIEW_TEST_URL || 'http://127.0.0.1:8765/';
const CHROME = process.env.CHROMIUM_EXECUTABLE || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

// 384/512/720 are 200% page-zoom equivalents for 768/1024/1440 CSS-pixel windows.
const WIDTHS = [
  { width: 320, label: '320' },
  { width: 384, label: '384 (768 @ 200%)' },
  { width: 390, label: '390' },
  { width: 512, label: '512 (1024 @ 200%)' },
  { width: 720, label: '720 (1440 @ 200%)' },
  { width: 768, label: '768' },
  { width: 1024, label: '1024' },
  { width: 1440, label: '1440' },
  { width: 1994, label: '1994' },
];

const PREFILL_POSITIONS = [0, 1, 1.5, 2.5, 3.5, 3.9];
const RETURN_POSITIONS = [4.55, 4.65, 4.75, 4.85, 4.95];
const DECODE_POSITIONS = [5.5, 5.8, 6, 6.2, 6.5, 7.5, 8.5];

let server;
let browser;
let page;
let owned;
let profilePath;
let shuttingDown;

const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

function profileProcesses() {
  if (!profilePath) return [];
  try {
    return require('node:child_process')
      .execFileSync('pgrep', ['-fl', profilePath], { encoding: 'utf8' })
      .trim()
      .split('\n')
      .filter(Boolean);
  } catch (error) {
    if (error.status === 1) return [];
    throw error;
  }
}

async function shutdown(reason) {
  if (shuttingDown) return shuttingDown;
  shuttingDown = (async () => {
    await page?.close().catch(() => {});
    await browser?.close().catch(() => {});
    await server?.close().catch(() => {});
    if (owned?.pid) {
      let alive = true;
      for (let i = 0; i < 20 && alive; i++) {
        try { process.kill(owned.pid, 0); alive = true; } catch (error) { alive = error.code !== 'ESRCH'; }
        if (alive) await delay(100);
      }
      if (alive) {
        try { process.kill(owned.pid, 'SIGTERM'); } catch {}
        await delay(250);
        try { process.kill(owned.pid, 0); } catch (error) { alive = error.code !== 'ESRCH'; }
      }
      if (alive) {
        try { process.kill(owned.pid, 'SIGKILL'); } catch {}
        await delay(150);
      }
    }
    const remaining = profileProcesses();
    if (profilePath && (profilePath.startsWith(os.tmpdir()) || profilePath.includes('playwright'))) {
      fs.rmSync(profilePath, { recursive: true, force: true });
    }
    const browserAlive = owned?.pid ? (() => {
      try { process.kill(owned.pid, 0); return true; } catch (error) { return error.code !== 'ESRCH'; }
    })() : false;
    console.log('CLEANUP ' + JSON.stringify({
      reason,
      controllerPid: process.pid,
      browserPid: owned?.pid || null,
      profile: profilePath,
      browserAlive,
      profileProcesses: remaining.length,
      profileRemoved: !profilePath || !fs.existsSync(profilePath),
    }));
    if (browserAlive || remaining.length) throw new Error(`browser cleanup incomplete: ${reason}`);
    return { browserAlive, remaining: remaining.length };
  })();
  return shuttingDown;
}

for (const [signal, code] of [['SIGINT', 130], ['SIGTERM', 143]]) {
  process.once(signal, () => {
    shutdown(signal)
      .then(() => process.exit(code))
      .catch(error => { console.error(error); process.exit(code); });
  });
}

async function seek(value) {
  await page.locator('[data-layer-journey] [data-jl-range]').evaluate((range, next) => {
    range.value = String(next);
    range.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await page.waitForTimeout(24);
}

async function geometry() {
  return page.evaluate(() => {
    const root = document.querySelector('[data-layer-journey]');
    const packets = [...root.querySelectorAll('.jl-packet')];
    const visible = packets.filter(packet => {
      const style = getComputedStyle(packet);
      return style.display !== 'none' && Number(style.opacity) > 0.01;
    });
    const box = rect => ({
      x: rect.x, y: rect.y, width: rect.width, height: rect.height,
      right: rect.right, bottom: rect.bottom,
    });
    const canvas = box(root.querySelector('.jl-canvas').getBoundingClientRect());
    const layerBands = [...root.querySelectorAll('.jl-layer')].map((layer, index) => ({
      layer: index + 1,
      band: box(layer.querySelector('.jl-band').getBoundingClientRect()),
      title: box(layer.querySelector('.jl-layer-title').getBoundingClientRect()),
      shelf: box(layer.querySelector('.jl-shelf').getBoundingClientRect()),
    }));
    const outputTokenEls = [...root.querySelectorAll('.jl-out-token')];
    const outputTokens = outputTokenEls.map(token => ({
      value: token.querySelector('text').textContent,
      opacity: Number(getComputedStyle(token).opacity),
      rect: box(token.querySelector('rect').getBoundingClientRect()),
      text: box(token.querySelector('text').getBoundingClientRect()),
    }));
    const outputStatuses = [...root.querySelectorAll('.jl-out-status')].map(status => ({
      value: status.textContent,
      box: box(status.getBoundingClientRect()),
    }));
    const outputTitleEl = root.querySelector('.jl-output-title');
    const linkEl = root.querySelector('.jl-return-link');
    const hitAt = element => {
      const r = element.getBoundingClientRect();
      const node = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
      return {
        hit: node?.className?.baseVal || node?.className || node?.tagName || '',
        inPacket: [...root.querySelectorAll('.jl-packet')].some(packet => packet.contains(node)),
      };
    };
    return {
      step: Number(root.dataset.jlStep),
      phase: root.dataset.jlPhase,
      packetCount: packets.length,
      visiblePackets: visible.length,
      packets: visible.map(packet => ({
        kind: packet.dataset.packet,
        title: packet.querySelector('.jl-packet-title')?.textContent || '',
        opacity: Number(getComputedStyle(packet).opacity),
        frame: box(packet.querySelector('.jl-packet-bg').getBoundingClientRect()),
        texts: [...packet.querySelectorAll('text')].map(text => ({
          value: text.textContent,
          box: box(text.getBoundingClientRect()),
        })),
      })),
      canvas,
      layerBands,
      outputRow: box(root.querySelector('.jl-out-row').getBoundingClientRect()),
      outputTitle: { value: outputTitleEl.textContent, box: box(outputTitleEl.getBoundingClientRect()), hit: hitAt(outputTitleEl) },
      outputTokens: outputTokens.map((token, index) => ({
        ...token,
        hit: hitAt(outputTokenEls[index].querySelector('text')),
      })),
      outputStatuses,
      returnLink: linkEl ? {
        opacity: Number(getComputedStyle(linkEl).opacity),
        box: box(linkEl.getBoundingClientRect()),
      } : null,
      pageScrollWidth: document.documentElement.scrollWidth,
      viewportWidth: innerWidth,
      output: root.dataset.jlOutput,
      statuses: [...root.querySelectorAll('.jl-shelf-status')].map(status => status.textContent),
      hitTests: visible.map(packet => {
        const frame = packet.querySelector('.jl-packet-bg').getBoundingClientRect();
        const point = { x: frame.x + frame.width / 2, y: frame.y + frame.height / 2 };
        const hit = document.elementFromPoint(point.x, point.y);
        return { kind: packet.dataset.packet, hit: hit?.className?.baseVal || hit?.className || hit?.tagName || '', inside: !!hit && packet.contains(hit) };
      }),
    };
  });
}

(async () => {
  server = await chromium.launchServer({ headless: true, executablePath: CHROME });
  owned = server.process();
  const profileArg = owned.spawnargs.find(arg => arg.startsWith('--user-data-dir='));
  profilePath = profileArg ? profileArg.slice('--user-data-dir='.length) : null;
  console.log(JSON.stringify({ session: 'test_journey_layer_geometry_browser', controller: process.pid, browser: owned.pid, profile: profilePath }));

  browser = await chromium.connect(server.wsEndpoint());
  page = await browser.newPage({ viewport: { width: 1994, height: 1150 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('response', response => {
    if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`);
  });

  await page.goto(BASE + 'foundations/generation.html#mechanism-extension', { waitUntil: 'networkidle' });
  const root = page.locator('[data-layer-journey]');
  await root.locator('[data-jl-play]').waitFor();
  // Geometry is driven synchronously by transform; disabling the opacity
  // transition keeps the takeoff/crossfade frame visible to computed style.
  await page.addStyleTag({ content: '.jl .jl-packet{transition:none!important}' });
  assert.equal(await page.locator('#mechanism-extension').evaluate(details => details.open), true, 'hash opens the supplement');
  assert.equal(await page.locator('#mechanism-extension .jl-token').filter({ hasText: /^y1$/ }).count() > 0, true, 'y1 remains the moving object');

  let checked = 0;
  let returnChecked = 0;
  for (const { width, label } of WIDTHS) {
    await page.setViewportSize({ width, height: 1100 });
    await page.waitForTimeout(140);
    await page.locator('[data-layer-journey] .jl-canvas').scrollIntoViewIfNeeded();
    await page.waitForTimeout(40);

    for (const position of [...PREFILL_POSITIONS, ...RETURN_POSITIONS, ...DECODE_POSITIONS]) {
      await seek(position);
      const state = await geometry();
      const where = `${label} @ p=${position}`;
      assert.ok(state.visiblePackets >= 1 && state.visiblePackets <= 2, `${where}: moving packets stay present`);
      assert.equal(state.packetCount, 2, `${where}: one prefill and one decode object, never duplicated`);
      const expectedKind = position < 4.5 ? 'prefill' : 'decode';
      assert.ok(state.packets.some(packet => packet.kind === expectedKind), `${where}: expected ${expectedKind} object`);

      for (const packet of state.packets) {
        const frame = packet.frame;
        const hitTest = state.hitTests.find(test => test.kind === packet.kind);
        assert.ok(hitTest?.inside, `${where}: ${packet.kind} packet is hidden behind another object (hit ${hitTest?.hit || 'nothing'})`);
        const canvas = state.canvas;
        assert.ok(frame.x >= canvas.x - 0.5 && frame.right <= canvas.right + 0.5,
          `${where}: ${packet.kind} frame clipped horizontally (${frame.x}..${frame.right} vs ${canvas.x}..${canvas.right})`);
        assert.ok(frame.y >= canvas.y - 0.5 && frame.bottom <= canvas.bottom + 0.5,
          `${where}: ${packet.kind} frame clipped vertically (${frame.y}..${frame.bottom} vs ${canvas.y}..${canvas.bottom})`);

        for (const layer of state.layerBands) {
          assert.equal(overlapArea(frame, layer.band), 0,
            `${where}: ${packet.kind} frame overlaps layer ${layer.layer} band`);
          assert.equal(overlapArea(frame, layer.title), 0,
            `${where}: ${packet.kind} frame overlaps layer ${layer.layer} name`);
          assert.equal(overlapArea(frame, layer.shelf), 0,
            `${where}: ${packet.kind} frame overlaps layer ${layer.layer} K/V shelf`);
        }

        for (const text of packet.texts) {
          assert.ok(text.box.x >= frame.x - 0.75 && text.box.right <= frame.right + 0.75,
            `${where}: ${packet.kind} text ${text.value} escapes the packet frame`);
          for (const layer of state.layerBands) {
            assert.equal(overlapArea(text.box, layer.band), 0,
              `${where}: ${packet.kind} text ${text.value} overlaps layer ${layer.layer}`);
          }
        }

        // While the packet is level with a layer band, leave a visible gutter.
        const verticalOverlap = state.layerBands.some(layer =>
          Math.min(frame.bottom, layer.band.bottom) - Math.max(frame.y, layer.band.y) > 0);
        if (verticalOverlap) {
          const gap = Math.min(...state.layerBands.map(layer => layer.band.x - frame.right));
          assert.ok(gap >= 8, `${where}: only ${gap}px gutter between ${packet.kind} and layer panel`);
        }
      }

      assert.ok(state.pageScrollWidth <= state.viewportWidth + 1, `${where}: page overflows horizontally`);

      if (position >= 4.55 && position <= 4.95) {
        // Real regression from the 390px Decode screenshot: the returning y1
        // card crossed the output row and covered both the heading and the
        // original y1. The card must take off above the row's top edge, keep
        // the heading/original readable, and stay one single visible object.
        const decode = state.packets.find(packet => packet.kind === 'decode');
        assert.ok(decode, `${where}: copied y1 card is missing during the return`);
        assert.ok(decode.opacity > 0.01, `${where}: copied y1 card is not visible during the return`);
        assert.ok(decode.frame.bottom <= state.outputRow.y - 3.5,
          `${where}: copied y1 card ${decode.frame.bottom} enters the output row top ${state.outputRow.y}`);
        assert.equal(overlapArea(decode.frame, state.outputRow), 0,
          `${where}: copied y1 card overlaps the output row`);
        assert.equal(overlapArea(decode.frame, state.outputTitle.box), 0,
          `${where}: copied y1 card overlaps the output heading`);
        assert.equal(state.outputTitle.value, '输出序列 · 已选出的 token 保留', `${where}: output heading text changed`);
        assert.equal(state.outputTitle.hit.inPacket, false,
          `${where}: output heading is covered by the moving card (hit ${state.outputTitle.hit.hit})`);
        for (const [index, token] of state.outputTokens.entries()) {
          assert.equal(overlapArea(decode.frame, token.rect), 0,
            `${where}: copied y1 card overlaps output token ${token.value} (${index})`);
          if (token.opacity > 0.01) {
            assert.equal(overlapArea(decode.frame, token.text), 0,
              `${where}: copied y1 card overlaps output token text ${token.value}`);
            assert.equal(token.hit.inPacket, false,
              `${where}: output token ${token.value} is covered by the moving card`);
          }
        }
        for (const status of state.outputStatuses) {
          assert.equal(overlapArea(decode.frame, status.box), 0,
            `${where}: copied y1 card overlaps output status "${status.value}"`);
        }
        for (const text of decode.texts) {
          assert.ok(text.box.x >= decode.frame.x - 0.75 && text.box.right <= decode.frame.right + 0.75,
            `${where}: copied y1 text ${text.value} escapes the card`);
        }
        if (state.returnLink && state.returnLink.opacity > 0.01) {
          assert.equal(overlapArea(state.returnLink.box, state.outputTitle.box), 0,
            `${where}: return route overlaps the output heading`);
          for (const token of state.outputTokens) {
            assert.equal(overlapArea(state.returnLink.box, token.rect), 0,
              `${where}: return route overlaps output token ${token.value}`);
          }
        }
        returnChecked++;
      }
      checked++;
    }
  }

  // Preserve the existing teaching states while changing only packet geometry.
  await page.setViewportSize({ width: 1024, height: 1100 });
  await seek(6);
  let state = await geometry();
  assert.equal(state.phase, 'Decode');
  assert.equal(state.output, 'y1');
  assert.deepEqual(state.statuses, ['5 个位置', '4 个位置', '4 个位置'], 'Decode layer 1 writes y1 only');
  assert.equal(state.packets.find(packet => packet.kind === 'decode').title, 'h¹(y1)');
  await seek(8.5);
  state = await geometry();
  assert.deepEqual(state.statuses, ['5 个位置', '5 个位置', '5 个位置'], 'a new position traverses all layers');
  await seek(9);
  state = await geometry();
  assert.equal(state.output, 'y1,y2');
  assert.equal(await page.locator('[data-layer-journey]').getAttribute('data-jl-layers'), '3');

  // Pause/back/reset still drive the same seekable teaching position.
  await seek(6);
  await root.locator('[data-jl-prev]').click();
  assert.equal(await root.getAttribute('data-jl-step'), '5.00');
  await root.locator('[data-jl-next]').click();
  assert.equal(await root.getAttribute('data-jl-step'), '6.00');
  await root.locator('[data-jl-reset]').click();
  assert.equal(await root.getAttribute('data-jl-step'), '0.00');

  // Collapsing the supplement stops playback; reopening mounts the same figure.
  await root.locator('[data-jl-play]').click();
  await page.waitForTimeout(120);
  await page.locator('#mechanism-extension > summary').click();
  await page.waitForTimeout(180);
  const collapsedStep = await root.getAttribute('data-jl-step');
  await page.waitForTimeout(500);
  assert.equal(await root.getAttribute('data-jl-step'), collapsedStep, 'collapse stops playback');
  assert.equal(await root.locator('[data-jl-play]').getAttribute('aria-pressed'), 'false');
  await page.locator('#mechanism-extension > summary').click();
  await root.locator('[data-jl-play]').waitFor();

  assert.deepEqual(errors, []);
  console.log(`PASS: journey-layer packet stays clear of all layer bands/names/KV and the output row's heading/original tokens across ${WIDTHS.length} widths (${returnChecked} return positions and full Prefill/Decode traversal); existing KV timing, output, pause/back/reset/collapse preserved.`);
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
}).finally(async () => {
  await shutdown('complete');
});

function overlapArea(a, b) {
  return Math.max(0, Math.min(a.right, b.right) - Math.max(a.x, b.x))
    * Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.y, b.y));
}
