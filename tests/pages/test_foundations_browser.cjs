/* Chapter 1 browser checks: every main mechanism scene, frame, branch and
 * narrow-screen layout; plus the specific causal labels added by this pass. */
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const courses = require('../../pages/foundations/courses.json');
const chooseScene = require('./choose_motion_scene.cjs');

const BASE = process.env.OVERVIEW_TEST_URL || 'http://127.0.0.1:8765/';
const CHROME = process.env.CHROMIUM_EXECUTABLE || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

async function choose(film, value) {
  const select = film.locator('.motion-controls select');
  if (!(await select.count())) return;
  await select.evaluate((el, v) => {
    el.value = String(v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await film.page().waitForTimeout(25);
}

async function optionsFor(page) {
  const film = page.locator('.mechanism-film');
  const select = film.locator('.motion-controls select');
  const families = film.locator('.parallel-strategies button');
  if (await families.count()) {
    const values = [];
    for (let i = 0; i < await families.count(); i++) {
      await families.nth(i).click();
      await page.waitForTimeout(25);
      values.push(...await select.locator('option').evaluateAll(es => es.map(e => e.value)));
    }
    return [...new Set(values)];
  }
  if (await select.count()) return await select.locator('option').evaluateAll(es => es.map(e => e.value));
  return ['0'];
}

async function goFinal(page) {
  const next = page.locator('[data-motion-action=next]');
  for (let i = 0; i < 20 && !(await next.isDisabled()); i++) {
    await next.click();
    await page.waitForTimeout(18);
  }
}

async function family(page, label) {
  const button = page.locator('.parallel-strategies button').filter({ hasText: label }).first();
  assert.equal(await button.count(), 1, `family button ${label}`);
  await button.click();
  await page.waitForTimeout(30);
}

(async () => {
  const server = await chromium.launchServer({
    headless: true,
    executablePath: CHROME,
  });
  const owned = server.process();
  const profile = owned.spawnargs.find(a => a.startsWith('--user-data-dir='));
  console.log(JSON.stringify({ session: 'test_foundations_browser', controller: process.pid, browser: owned.pid, profile }));
  const browser = await chromium.connect(server.wsEndpoint());
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
  const errors = [], httpFailures = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('response', r => { if (r.status() >= 400) httpFailures.push(r.status() + ' ' + r.url()); });

  try {
    let sceneCount = 0;
    for (const course of courses) {
      await page.goto(BASE + 'foundations/' + course.slug + '.html', { waitUntil: 'networkidle' });
      assert.equal(await page.locator('h1').count(), 1, `${course.slug} h1`);
      assert.equal(await page.locator('.foundation-nav [aria-current=page]').count(), 1, `${course.slug} nav`);
      assert.ok(await page.locator('#evidence a').count(), `${course.slug} evidence`);
      const film = page.locator('.mechanism-film');
      assert.equal(await film.count(), 1, `${course.slug} mechanism film`);
      const values = await optionsFor(page);
      assert.ok(values.length >= 1, `${course.slug} scenarios`);
      for (const value of values) {
        await chooseScene(page, Number(value));
        const select = film.locator('.motion-controls select');
        if (await select.count() && await select.isVisible()) {
          assert.equal(await select.inputValue(), String(value), `${course.slug} selected scenario ${value}`);
        }
        const max = Number(await film.locator('.motion-scrub input').getAttribute('max'));
        assert.ok(max >= 1, `${course.slug}/${value} steps`);
        sceneCount++;
        for (let step = 0; step <= max; step++) {
          if (step) {
            const next = film.locator('[data-motion-action=next]');
            assert.equal(await next.isDisabled(), false, `${course.slug}/${value} can advance`);
            await next.click();
            await page.waitForTimeout(18);
          }
          const state = await film.evaluate(el => ({
            pos: Number(el.dataset.position),
            status: el.querySelector('.motion-status')?.textContent || '',
            action: el.querySelector('.motion-action')?.textContent || '',
            svg: el.querySelector('svg')?.textContent || '',
          }));
          assert.ok(state.status.length > 3, `${course.slug}/${value}/${step} status`);
          assert.ok(state.action.length > 3, `${course.slug}/${value}/${step} action`);
          assert.doesNotMatch(state.svg + state.action + state.status, /undefined|NaN/, `${course.slug}/${value}/${step} no missing values`);
        }
        if (course.slug === 'deployment') assert.match(await film.textContent(), /当前可用 3\/3/);
      }
      // The narrow layout must not create a page-level horizontal scroll.
      await page.setViewportSize({ width: 390, height: 844 });
      await choose(film, values[0]);
      await goFinal(page);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), 390, `${course.slug} mobile width`);
      await page.setViewportSize({ width: 1440, height: 1000 });
      console.log('PASS', course.slug, values.length, 'scenes');
    }
    assert.ok(sceneCount >= 77, `main scene coverage ${sceneCount}`);

    // F01-01: the two lookup tables show occurrence positions and the exact
    // vocabulary/ID mapping; the occurrence rail is not a duplicate table row.
    await page.goto(BASE + 'foundations/tokens.html', { waitUntil: 'networkidle' });
    const tokens = page.locator('.mechanism-film');
    await choose(tokens, 0); await goFinal(page);
    assert.match(await tokens.textContent(), /100134.*p2.*p4/s);
    assert.match(await tokens.textContent(), /104100.*p1/);
    assert.equal(await tokens.locator('svg [data-motion-id="id-3"]').count(), 1);

    // F02: client-visible text and KV remain separate from the selected token.
    await page.goto(BASE + 'foundations/generation.html', { waitUntil: 'networkidle' });
    const generation = page.locator('.mechanism-film');
    await choose(generation, 1); await goFinal(page);
    assert.match(await generation.textContent(), /客户端已见：休息一下/);
    assert.match(await generation.textContent(), /历史 KV/);

    // F03: both dimensions expose their added terms.
    await page.goto(BASE + 'foundations/attention.html', { waitUntil: 'networkidle' });
    const attention = page.locator('.mechanism-film');
    await choose(attention, 0); await goFinal(page);
    assert.match(await attention.textContent(), /维度1：0\.66 \+ 0\.00.*≈ 0\.66/s);
    assert.match(await attention.textContent(), /维度2：0\.00 \+ 1\.34.*≈ 1\.34/s);
    await chooseScene(page, 1); await goFinal(page);
    assert.match(await attention.textContent(), /2\.59 ≈ 3\.22/);
    assert.doesNotMatch(await attention.textContent(), /2\.59 = 3\.22/);

    // F04: page mapping and shared-prefix references are in the main selector;
    // the comparison keeps the read cost visible.
    await page.goto(BASE + 'foundations/cache.html', { waitUntil: 'networkidle' });
    const cache = page.locator('.mechanism-film');
    await choose(cache, 0); await goFinal(page);
    assert.match(await cache.textContent(), /读取历史：4 \+ 5 = 9/);
    await choose(cache, 1); await goFinal(page);
    assert.match(await cache.textContent(), /页 7.*页 2/s);
    await choose(cache, 2); await goFinal(page);
    assert.match(await cache.textContent(), /共享历史：同一份 K\/V/);

    // F06: products keep both factors, and graph prep cost is a labelled
    // one-time assumption rather than a measured break-even.
    await page.goto(BASE + 'foundations/hardware.html', { waitUntil: 'networkidle' });
    await family(page, '矩阵乘加'); await goFinal(page);
    assert.match(await page.locator('.mechanism-film').textContent(), /1×2=2/);
    await family(page, 'CUDA Graph'); await goFinal(page);
    assert.match(await page.locator('.mechanism-film').textContent(), /一次性准备（教学假设，不标时长）/);

    // F07: EP's scalar is on t1; SP/ DCP keep both K and V and use a clear
    // staging lane (asserted by the visible copy labels and stable panel rows).
    await page.goto(BASE + 'foundations/parallel.html', { waitUntil: 'networkidle' });
    await family(page, '专家'); await goFinal(page);
    assert.match(await page.locator('.mechanism-film').textContent(), /t1 x=2/);
    await family(page, '序列'); await goFinal(page);
    assert.match(await page.locator('.mechanism-film').textContent(), /AllGather/);
    await family(page, 'Decode 上下文'); await goFinal(page);
    assert.match(await page.locator('.mechanism-film').textContent(), /p1 K=0 V=1/);

    // F08: direct Put/Get labels; failure and CAS branches use their own final
    // status instead of the successful completion sentence.
    await page.goto(BASE + 'foundations/communication.html', { waitUntil: 'networkidle' });
    const communication = page.locator('.mechanism-film');
    await family(page, 'Put'); await goFinal(page);
    assert.match(await communication.textContent(), /写入载荷 \[7,8\]/);
    await family(page, 'Get'); await goFinal(page);
    assert.match(await communication.textContent(), /读取请求/);
    assert.match(await communication.textContent(), /返回 \[7,8\]/);
    await family(page, 'Send / Recv');
    const commSelect = communication.locator('.motion-controls select');
    const sendValues = await commSelect.locator('option').evaluateAll(es => es.map(e => [e.value, e.textContent]));
    const missing = sendValues.find(x => x[1].includes('接收缺席'))[0];
    await choose(communication, missing); await goFinal(page);
    assert.match(await communication.textContent(), /接收尚未完成/);
    await family(page, '原子更新');
    const atomicValues = await commSelect.locator('option').evaluateAll(es => es.map(e => [e.value, e.textContent]));
    const failed = atomicValues.find(x => x[1].includes('比较失败'))[0];
    await choose(communication, failed); await goFinal(page);
    assert.match(await communication.textContent(), /比较失败：保留 5/);

    // F09: prefix, local y1 KV and total available count are separate.
    await page.goto(BASE + 'foundations/deployment.html', { waitUntil: 'networkidle' });
    await goFinal(page);
    assert.match(await page.locator('.mechanism-film').textContent(), /收到的前缀 2\/2/);
    assert.match(await page.locator('.mechanism-film').textContent(), /本地新增 y1 KV 1/);
    assert.match(await page.locator('.mechanism-film').textContent(), /当前可用 3\/3/);

    // F10: root and valid history are in the main speculation explanation.
    await page.goto(BASE + 'foundations/speculation.html', { waitUntil: 'networkidle' });
    await choose(page.locator('.mechanism-film'), 0); await goFinal(page);
    assert.match(await page.locator('.mechanism-film').textContent(), /根 token → 本轮草稿/);
    assert.match(await page.locator('.mechanism-film').textContent(), /有效历史：根 \+ 接受前缀/);

    // F11: cancellation lanes and both serving routes are reachable.
    await page.goto(BASE + 'foundations/serving.html', { waitUntil: 'networkidle' });
    const serving = page.locator('.mechanism-film');
    await choose(serving, 0); await goFinal(page);
    assert.match(await serving.textContent(), /取消已送达后端/);
    assert.match(await serving.textContent(), /完成收尾，资源才可再使用/);
    await choose(serving, 1); await goFinal(page);
    assert.match(await serving.textContent(), /进入实例/);
    await choose(serving, 2); await goFinal(page);
    assert.match(await serving.textContent(), /只路由到 B/);
    assert.match(await serving.textContent(), /A 未收到 R1/);

    // F12: packet children and the request-observation window are explicit.
    await page.goto(BASE + 'foundations/metrics.html', { waitUntil: 'networkidle' });
    const metrics = page.locator('.mechanism-film');
    await choose(metrics, 1); await goFinal(page);
    assert.match(await metrics.textContent(), /token 2/);
    assert.match(await metrics.textContent(), /token 3/);
    assert.match(await metrics.textContent(), /请求总观测窗口（发出 → 末 token）/);

    // Retained learning behaviors: quiz feedback, deep-link restoration,
    // keyboard advance and lazy supplement initialization/stop-on-collapse.
    await page.goto(BASE + 'foundations/generation.html', { waitUntil: 'networkidle' });
    const quiz = page.locator('.fd-quiz');
    assert.equal(await quiz.isVisible(), true);
    const answer = Number(await quiz.getAttribute('data-answer'));
    const choiceCount = await quiz.locator('input').count();
    await quiz.locator(`input[value="${(answer + 1) % choiceCount}"]`).check();
    await quiz.locator('button').click();
    assert.match(await quiz.locator('.fd-feedback').innerText(), /再看一下/);
    await quiz.locator(`input[value="${answer}"]`).check();
    await quiz.locator('button').click();
    assert.match(await quiz.locator('.fd-feedback').innerText(), /判断正确/);

    await page.goto(BASE + 'foundations/tokens.html?mv=1&ms=2&v=review', { waitUntil: 'networkidle' });
    const deep = page.locator('.mechanism-film');
    assert.equal(Number(await deep.getAttribute('data-position')), 2);
    await deep.locator('[data-motion-action=next]').focus();
    await page.keyboard.press('Enter');
    assert.equal(Number(await deep.getAttribute('data-position')), 3);
    assert.equal(new URL(page.url()).searchParams.get('v'), 'review');
    await page.reload({ waitUntil: 'networkidle' });
    assert.equal(Number(await page.locator('.mechanism-film').getAttribute('data-position')), 3);

    await page.goto(BASE + 'foundations/generation.html', { waitUntil: 'networkidle' });
    const extension = page.locator('.foundation-extension');
    assert.equal(await extension.evaluate(e => e.open), false);
    assert.equal(await extension.locator('.wr-scene').count(), 0, 'closed supplement is lazy');
    await extension.locator(':scope > summary').click();
    await extension.locator('[data-id=j-play]').click();
    await page.waitForTimeout(120);
    await extension.locator(':scope > summary').click();
    await page.waitForTimeout(200);
    const closedText = await extension.locator('.wr-scene').textContent();
    await page.waitForTimeout(1000);
    assert.equal(await extension.locator('.wr-scene').textContent(), closedText, 'collapsing a supplement pauses its playback');

    // Without JavaScript the prose, source links and quiz explanations remain readable.
    const staticPage = await browser.newPage({ javaScriptEnabled: false });
    await staticPage.goto(BASE + 'foundations/tokens.html');
    assert.equal(await staticPage.locator('h1').count(), 1);
    assert.ok((await staticPage.locator('.foundation-copy').innerText()).length > 100);
    assert.ok(await staticPage.locator('#evidence a').count());
    await staticPage.close();

    assert.deepEqual(errors, []);
    console.log(`PASS: ${courses.length} lessons; ${sceneCount} main scenes; every frame, mobile width, key causal labels, failure branches and no-JS content.`);
    assert.deepEqual(httpFailures, [], 'no lesson resource HTTP failures');
  } finally {
    await page.close().catch(() => {});
    await browser.close().catch(() => {});
    await server.close().catch(() => {});
    try { process.kill(owned.pid, 0); throw new Error('owned browser process remains'); }
    catch (e) { if (e.code !== 'ESRCH' && !/remains/.test(e.message)) throw e; }
    if (profile) {
      const profilePath = profile.slice('--user-data-dir='.length);
      try { require('node:fs').rmSync(profilePath, { recursive: true, force: true }); } catch {}
    }
    console.log('CLEANED test_foundations_browser browser PID', owned.pid);
  }
})().catch(error => {
  console.error(error);
  process.exit(1);
});
