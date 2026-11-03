/* Chapter 1 browser checks: every main mechanism scene, frame, branch and
 * narrow-screen layout; plus the specific causal labels added by this pass. */
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const courses = require('../../pages/foundations/courses.json');
const model = require('../../pages/foundations/model.js');
const chooseScene = require('./choose_motion_scene.cjs');

const BASE = process.env.OVERVIEW_TEST_URL || 'http://127.0.0.1:8765/';
const CHROME = process.env.CHROMIUM_EXECUTABLE || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

// Attention publishes vectors through two display formats: fmt() keeps a
// space and trims trailing zeros, while dataset.output is fixed to two
// decimals without spaces. Compare the rendered numbers, allowing the
// rounding implied by two-decimal display, instead of matching a string shape.
function closeNumbers(text, expected, message, tolerance = 0.0051) {
  const match = String(text).match(/\[\s*([^\]]+?)\s*\]/);
  assert.ok(match, `${message}: expected a numeric array, got ${text}`);
  const actual = match[1].split(',').map(part => Number(part.trim()));
  assert.equal(actual.length, expected.length, `${message}: ${text} vs ${expected.length} values`);
  actual.forEach((value, i) => {
    assert.ok(Number.isFinite(value) && Math.abs(value - expected[i]) <= tolerance,
      `${message}: ${text} at index ${i}, expected ${expected[i]}`);
  });
}

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
      // Enumerate the registered catalogue itself, so family-only traversal
      // still reaches the two legacy token links that have no family button.
      const values = Array.from({ length: await chooseScene.count(page) }, (_, i) => String(i));
      assert.ok(values.length >= 1, `${course.slug} scenarios`);
      for (const value of values) {
        await chooseScene(page, Number(value));
        await film.locator('[data-motion-action=reset]').click();
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
    assert.ok(sceneCount >= 95, `main scene coverage ${sceneCount}`);

    // F01-01: the old sample is a URL-stable legacy entry, not the default
    // BPE concept. Keep its original two-row drawing assertions explicit.
    await page.goto(BASE + 'foundations/tokens.html?mv=0', { waitUntil: 'networkidle' });
    const tokens = page.locator('.mechanism-film');
    assert.equal(new URL(page.url()).searchParams.get('mv'), '0');
    await goFinal(page);
    assert.match(await tokens.textContent(), /100134.*p2.*p4/s);
    assert.match(await tokens.textContent(), /104100.*p1/);
    assert.equal(await tokens.locator('svg [data-token-row]').count(), 4);
    assert.equal(await tokens.locator('svg [data-motion-id^="row-piece-"]').count(), 4);

    // F02: the selected token is copied back to the model input; the output
    // node and the KV history remain separate until the new input is computed.
    await page.goto(BASE + 'foundations/generation.html', { waitUntil: 'networkidle' });
    const generation = page.locator('.mechanism-film');
    await generation.getByLabel('连续演示条件').selectOption({ label: '完整自回归链 · 固定抽样' });
    await generation.locator('[data-motion-action=reset]').click();
    for (let i = 0; i < 4; i++) { await generation.locator('[data-motion-action=next]').click(); await page.waitForTimeout(18); }
    assert.equal(await generation.locator('[data-kv-written="true"]').count(), 2, 'y1 KV is not written before Decode');
    assert.equal(await generation.locator('[data-input-copy="true"]').count(), 1, 'the output copy is the new input');
    await goFinal(page);
    assert.match(await generation.textContent(), /客户端文字另经反分词与传输/);
    assert.doesNotMatch(await generation.textContent(), /客户端已见/);
    assert.match(await generation.textContent(), /历史 KV/);
    assert.equal(await generation.locator('[data-kv-written="true"]').count(), 3);
    assert.equal(await generation.locator('[data-output="first"]').count(), 1);
    assert.equal(await generation.locator('[data-output="second"]').count(), 1);

    // F03: the current QKV/weights/mask graph publishes only readable
    // contributions, the causal denominator and the weighted sum. The
    // data-current value is pending/mixing until every source arrives; the
    // final data-output is a model check, never a mid-mix visible value.
    await page.goto(BASE + 'foundations/attention.html', { waitUntil: 'networkidle' });
    const attention = page.locator('.mechanism-film');
    const attentionFacts = () => attention.locator('.motion-attention-facts strong').allTextContents();
    const attentionView = () => attention.locator('svg').evaluate(svg => {
      const q = Number(svg.dataset.queryPosition);
      const cells = [...svg.querySelectorAll('.att-maskcell')];
      const row = cells.slice((q - 1) * 4, q * 4);
      const out = svg.querySelector('[data-current]');
      return {
        query: q,
        allowed: Number(svg.dataset.allowed),
        future: Number(svg.dataset.futureCount),
        denominator: Number(svg.dataset.denominator),
        weightSum: Number(svg.dataset.weightSum),
        output: svg.dataset.output,
        current: out?.getAttribute('data-current'),
        state: out?.getAttribute('data-mix-state'),
        count: Number(out?.getAttribute('data-mix-count')),
        total: Number(out?.getAttribute('data-mix-total')),
        self: out?.getAttribute('data-self-contribution'),
        history: out?.getAttribute('data-history-contribution'),
        maskFuture: row.filter(c => c.classList.contains('is-future')).length,
        maskOpen: row.filter(c => c.classList.contains('is-open')).length,
        maskText: svg.querySelector('[data-att-mode=mask]')?.textContent || '',
        maskOutput: svg.querySelector('[data-att-mode=mask] [data-output]')?.textContent || '',
        // Desktop mask rows print the percentage alone (33.0%), not w=33.0%.
        // Future rows print 0%; keeping only one-decimal tokens preserves row order.
        maskWeights: [...svg.querySelectorAll('[data-att-mode=mask] text.att-w.att-num')]
          .map(e => e.textContent.trim())
          .filter(text => /^\d+\.\d%$/.test(text))
          .map(text => Number(text.slice(0, -1))),
        weightsText: svg.querySelector('[data-att-mode=weights]')?.textContent || '',
      };
    });
    for (const variant of [0, 1, 2]) {
      const expected = model.attention(variant);
      await chooseScene(page, variant);
      await goFinal(page);
      const done = await attentionView();
      assert.equal(done.query, expected.queryPosition, `attention query ${variant}`);
      assert.equal(done.allowed, expected.allowed, `attention readable positions ${variant}`);
      assert.equal(done.future, 4 - expected.allowed, `attention causal future ${variant}`);
      assert.equal(done.maskFuture, 4 - expected.allowed, `mask row future ${variant}`);
      assert.equal(done.maskOpen, expected.allowed, `mask row readable ${variant}`);
      assert.equal(done.denominator.toFixed(4), expected.denominator.toFixed(4), `attention denominator ${variant}`);
      assert.equal(done.weightSum, 1, `attention weights sum ${variant}`);
      closeNumbers(done.output, expected.mixed, `attention model output ${variant}`);
      closeNumbers(done.current, expected.mixed, `attention published output ${variant}`);
      assert.equal(done.state, 'complete', `attention complete ${variant}`);
      assert.equal(done.count, expected.allowed, `attention arrived contributions ${variant}`);
      assert.equal(done.total, expected.allowed, `attention contribution total ${variant}`);
      const self = expected.contributions[expected.query];
      const history = [0, 1].map(j => expected.contributions.slice(0, expected.query).reduce((sum, c) => sum + c[j], 0));
      closeNumbers(done.self, self, `attention self contribution ${variant}`);
      closeNumbers(done.history, history, `attention history contribution ${variant}`);
      assert.match(done.maskText, new RegExp(`Σ e\\^s（仅可见） = ${expected.denominator.toFixed(2)}`), `attention denominator copy ${variant}`);
      assert.match(done.maskText, /w 合计 = 100\.0%/, `attention weight copy ${variant}`);
      closeNumbers(done.maskOutput, expected.mixed, `attention mask output ${variant}`);
      assert.equal(done.maskWeights.length, expected.allowed, `attention mask row weights ${variant}`);
      for (let i = 0; i < expected.allowed; i++) {
        assert.ok(Math.abs(done.maskWeights[i] - expected.weights[i] * 100) <= 0.051,
          `attention weight ${variant}/${i}: ${done.maskWeights[i]}% vs ${expected.weights[i] * 100}%`);
      }
      assert.deepEqual(await attentionFacts(), [
        expected.denominator.toFixed(2),
        '100.0%',
        `[${expected.mixed.map(x => x.toFixed(2)).join(', ')}]`,
      ], `attention side facts ${variant}`);

      await page.emulateMedia({ reducedMotion: 'no-preference' });
      for (const position of [3.9, 4.2]) {
        await seek(page, position);
        const mid = await attentionView();
        assert.equal(mid.state, 'mixing', `attention mixes at ${variant}/${position}`);
        assert.ok(['pending', 'mixing'].includes(mid.current), `attention current is not final at ${variant}/${position}`);
        assert.ok(mid.count < mid.total, `attention contributions incomplete at ${variant}/${position}`);
        assert.match(mid.weightsText, /汇合中/, `attention main says mixing at ${variant}/${position}`);
        assert.doesNotMatch(mid.weightsText, /m [0-9]/, `attention main has no numeric m at ${variant}/${position}`);
        assert.equal((await attentionFacts())[2], '汇合中', `attention side says mixing at ${variant}/${position}`);
      }
      await seek(page, 4.6);
      const mixed = await attentionView();
      assert.equal(mixed.state, 'complete', `attention completes at ${variant}/4.6`);
      closeNumbers(mixed.current, expected.mixed, `attention final at ${variant}/4.6`);
      assert.equal(mixed.count, expected.allowed, `attention all contributions at ${variant}/4.6`);
      assert.notEqual((await attentionFacts())[2], '汇合中', `attention side establishes at ${variant}/4.6`);
      await page.emulateMedia({ reducedMotion: 'reduce' });
    }

    // F04: the comparison keeps read and compute costs separate; the page
    // mapping is a real arrow relation; shared-prefix holders are explicit.
    await page.goto(BASE + 'foundations/cache.html', { waitUntil: 'networkidle' });
    const cache = page.locator('.mechanism-film');
    await choose(cache, 0); await goFinal(page);
    const cacheFinal = await cache.textContent();
    assert.match(cacheFinal, /跨轮缓存读取 4 \+ 5 = 9/, 'cache reads 4 and 5 while reusing history');
    assert.match(cacheFinal, /新算 4 \+ 5 \+ 6 = 15/, 'no-cache K/V positions sum to 15');
    assert.match(cacheFinal, /新算 4 \+ 1 \+ 1 = 6/, 'cached K/V positions sum to 6');
    assert.match(cacheFinal, /三次调用完成后的对照：新算 15 vs 6/, 'three-call comparison keeps 15 vs 6');
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

    // F10: the 860-unit board is fully in frame, and root/history are checked
    // as objects rather than by an old sentence about the draft row.
    await page.goto(BASE + 'foundations/speculation.html', { waitUntil: 'networkidle' });
    const speculation = page.locator('.mechanism-film');
    await choose(speculation, 0); await goFinal(page);
    assert.equal(await speculation.locator('svg').getAttribute('viewBox'), '0 0 860 800');
    assert.match(await speculation.textContent(), /有效历史：根 \+ 接受前缀/);
    const rootToken = speculation.locator('svg [data-motion-id="root-token"]');
    assert.equal(await rootToken.count(), 1);
    assert.equal((await rootToken.textContent()).trim(), 't0');
    assert.equal((await speculation.locator('svg [data-motion-id="verify-input-0"]').textContent()).trim(), '根');
    const validHistory = await speculation.locator('svg [data-motion-id^="valid-history-"]').evaluateAll(es => es.map(e => e.textContent.trim()));
    assert.deepEqual(validHistory, ['根', '我', '想'], 'valid history is root plus the accepted prefix');
    const committed = await speculation.locator('svg [data-motion-id^="committed-output-"]').evaluateAll(es => es.map(e => e.textContent.trim()));
    assert.deepEqual(committed, ['我', '想', '吃'], 'committed output ends with the replacement/bonus token');

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
