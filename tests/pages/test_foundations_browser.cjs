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

async function seek(page, position) {
  await page.locator('.mechanism-film .motion-scrub input').evaluate((el, p) => {
    el.value = String(p);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, position);
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
        if (course.slug === 'deployment') {
          assert.match(await film.textContent(), /可用 3\/3 · y1 KV 完成/);
          assert.equal(await film.locator('svg text[data-ready="true"]').count(), 1, 'deployment ready state is rendered');
        }
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

    // F01-01: one row per vocabulary item, and both occurrences of a repeated
    // token remain positions of that one row rather than duplicate table rows.
    await page.goto(BASE + 'foundations/tokens.html', { waitUntil: 'networkidle' });
    const tokens = page.locator('.mechanism-film');
    await choose(tokens, 0); await goFinal(page);
    assert.match(await tokens.textContent(), /100134.*p2.*p4/s);
    assert.match(await tokens.textContent(), /104100.*p1/);
    assert.equal(await tokens.locator('svg [data-token-row]').count(), 4);
    assert.equal(await tokens.locator('svg [data-motion-id^="row-piece-"]').count(), 4);

    // F02: the selected token is copied back to the model input; the output
    // node and the KV history remain separate until the new input is computed.
    await page.goto(BASE + 'foundations/generation.html', { waitUntil: 'networkidle' });
    const generation = page.locator('.mechanism-film');
    await choose(generation, 1);
    await generation.locator('[data-motion-action=reset]').click();
    for (let i = 0; i < 4; i++) { await generation.locator('[data-motion-action=next]').click(); await page.waitForTimeout(18); }
    assert.equal(await generation.locator('[data-kv-written="true"]').count(), 2, 'y1 KV is not written before Decode');
    assert.equal(await generation.locator('[data-input-copy="true"]').count(), 1, 'the output copy is the new input');
    await goFinal(page);
    assert.match(await generation.textContent(), /客户端已见：休息一下/);
    assert.match(await generation.textContent(), /历史 KV/);
    assert.equal(await generation.locator('[data-kv-written="true"]').count(), 3);
    assert.equal(await generation.locator('[data-output="first"]').count(), 1);
    assert.equal(await generation.locator('[data-output="second"]').count(), 1);

    // F03: both dimensions expose their added terms, and future rows stay out
    // of the denominator.
    await page.goto(BASE + 'foundations/attention.html', { waitUntil: 'networkidle' });
    const attention = page.locator('.mechanism-film');
    await choose(attention, 0); await goFinal(page);
    const attentionEqs = await attention.locator('svg [data-equation]').evaluateAll(es => es.map(e => e.textContent));
    assert.deepEqual(attentionEqs, ['维度1：33.0%×2 + 67.0%×0', '维度2：33.0%×0 + 67.0%×2']);
    assert.equal(await attention.locator('svg [data-complete="true"]').textContent(), '汇合得到 [0.66, 1.34]');
    assert.equal(await attention.locator('svg [data-future-row="true"]').count(), 2);
    assert.match(await attention.textContent(), /允许权重合计 ≈ 100\.0%/);
    await chooseScene(page, 1); await goFinal(page);
    assert.equal(await attention.locator('svg [data-equation="1"]').textContent(), '维度1：15.7%×2 + 3.8%×0 + 15.7%×2 + 64.7%×4');
    assert.equal(await attention.locator('svg [data-complete="true"]').textContent(), '汇合得到 [3.22, 0.39]');
    assert.match(await attention.textContent(), /16\.92.*26\.15/);

    // F04: the comparison keeps read and compute costs separate; the page
    // mapping is a real arrow relation; shared-prefix holders are explicit.
    await page.goto(BASE + 'foundations/cache.html', { waitUntil: 'networkidle' });
    const cache = page.locator('.mechanism-film');
    await choose(cache, 0); await goFinal(page);
    assert.match(await cache.textContent(), /读取历史：4 \+ 5 = 9/);
    assert.match(await cache.textContent(), /新算位置数：15.*vs 6/);
    await choose(cache, 1); await goFinal(page);
    assert.match(await cache.textContent(), /页 7.*页 2/s);
    assert.match(await cache.textContent(), /p1 → 页 7 \/ 槽 0/);
    assert.ok(await cache.locator('svg [data-map]').count() >= 6, 'page table and slot mappings are drawn');
    await choose(cache, 2);
    // Midpoints only exist outside reduced-motion mode; the shared player
    // rounds scrub positions when reduced motion is requested.
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await seek(page, 0.5);
    const earlyLabels = await cache.locator('svg text').evaluateAll(es => es.map(e => e.textContent));
    assert.ok(earlyLabels.includes('尚未建立共享历史'), 'history is not announced before R1 writes it');
    assert.equal(earlyLabels.filter(label => label === '共享历史：同一份 K/V').length, 0);
    assert.equal(await cache.locator('svg [data-holder]').getAttribute('aria-hidden'), 'true');
    const sharedMidpoints = [
      { position: 1.5, refs: ['r1'], text: /R1 指向这份数据；R2 尚未接入/, forbid: /R1、R2 指向|引用 0|R1 引用消失/ },
      { position: 2.5, refs: ['r1', 'r2'], text: /R1、R2 指向同一份数据；没有复制第二份/, forbid: /引用 0|R1 引用消失/ },
      { position: 3.5, refs: ['r2'], text: /R1 已结束；R2 仍指向同一份数据/, forbid: /引用 0|R1、R2 指向/ },
    ];
    for (const { position, refs, text, forbid } of sharedMidpoints) {
      await seek(page, position);
      const state = await cache.evaluate(el => ({
        position: Number(el.dataset.position),
        holder: Number(el.querySelector('svg [data-holder]')?.dataset.holder),
        holderText: el.querySelector('svg [data-holder]')?.textContent,
        refs: [...el.querySelectorAll('svg [data-ref]')]
          .filter(node => Number(node.getAttribute('opacity')) > 0.5)
          .map(node => node.dataset.ref),
        svg: el.querySelector('svg')?.textContent || '',
      }));
      assert.equal(state.position, position);
      assert.deepEqual(state.refs, refs, `visible reference arrows at ${position}`);
      assert.equal(state.holder, refs.length, `holder count matches arrows at ${position}`);
      assert.equal(state.holderText, `持有者：${refs.length}`, `holder label at ${position}`);
      assert.match(state.svg, text, `explanation at ${position}`);
      assert.doesNotMatch(state.svg, forbid, `no premature or stale release copy at ${position}`);
    }
    await seek(page, 4);
    assert.match(await cache.textContent(), /共享历史：仍被缓存保留/);
    assert.match(await cache.textContent(), /引用 0：没有活跃请求，缓存仍被保留/);
    assert.match(await cache.textContent(), /持有者：0/);
    assert.equal(await cache.locator('svg [data-holder="0"]').count(), 1);
    await page.emulateMedia({ reducedMotion: 'reduce' });

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
    assert.match(await page.locator('.mechanism-film').textContent(), /源 KV 2\/2 · 仍保留/);
    assert.match(await page.locator('.mechanism-film').textContent(), /副本 2\/2 · y1 已到/);
    assert.match(await page.locator('.mechanism-film').textContent(), /y1 KV 已写入 · 选出 y2/);
    assert.match(await page.locator('.mechanism-film').textContent(), /可用 3\/3 · y1 KV 完成/);
    assert.equal(await page.locator('.mechanism-film svg text[data-ready="true"]').count(), 1, 'deployment ready state is rendered');

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
    const journeyRoot = extension.locator('[data-layer-journey]');
    assert.equal(await extension.evaluate(e => e.open), false);
    assert.equal(await journeyRoot.locator('[data-jl-play]').count(), 0, 'closed supplement is lazy');
    await extension.locator(':scope > summary').click();
    const jlPlay = journeyRoot.locator('[data-jl-play]');
    await jlPlay.waitFor();
    await jlPlay.click();
    await page.waitForFunction(() => Number(document.querySelector('[data-layer-journey]').dataset.jlStep) > 0, null, { timeout: 2500 });
    assert.equal(await jlPlay.getAttribute('aria-pressed'), 'true', 'journey play state is pressed');
    assert.ok(Number(await journeyRoot.getAttribute('data-jl-step')) > 0, 'journey playback advances');
    await extension.locator(':scope > summary').click();
    await page.waitForTimeout(200);
    const closedStep = await journeyRoot.getAttribute('data-jl-step');
    assert.equal(await jlPlay.getAttribute('aria-pressed'), 'false', 'collapse clears pressed state');
    await page.waitForTimeout(700);
    assert.equal(await journeyRoot.getAttribute('data-jl-step'), closedStep, 'collapsing a supplement pauses its playback');

    // Without JavaScript the prose, source links and quiz explanations remain readable.
    const staticPage = await browser.newPage({ javaScriptEnabled: false });
    await staticPage.goto(BASE + 'foundations/tokens.html');
    assert.equal(await staticPage.locator('h1').count(), 1);
    assert.ok((await staticPage.locator('.foundation-copy').innerText()).length > 100);
    assert.ok(await staticPage.locator('#evidence a').count());
    await staticPage.goto(BASE + 'foundations/generation.html');
    const staticExtension = staticPage.locator('.foundation-extension');
    assert.match(await staticExtension.innerText(), /逐层前向/);
    await staticExtension.locator(':scope > summary').click();
    assert.match(await staticExtension.locator('noscript').innerText(), /请启用 JavaScript/);
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
