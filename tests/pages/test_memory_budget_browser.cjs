const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), os = require('node:os');
const cp = require('node:child_process'), zlib = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.OVERVIEW_TEST_URL || 'http://127.0.0.1:8765/';
const output = process.env.MEMORY_BUDGET_QA_DIR;
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
function processes() {
  return cp.execFileSync('ps', ['-axo', 'pid=,ppid=,stat=,command='], { encoding: 'utf8' }).trim().split('\n')
    .map(line => { const m = line.trim().match(/^(\d+)\s+(\d+)\s+(\S+)\s+(.*)$/); return m && { pid: +m[1], parent: +m[2], stat: m[3], command: m[4] }; }).filter(Boolean);
}
function descendants(pid) {
  const all = processes(), ids = new Set([pid]); let changed = true;
  while (changed) { changed = false; for (const p of all) if (ids.has(p.parent) && !ids.has(p.pid)) { ids.add(p.pid); changed = true; } }
  return all.filter(p => ids.has(p.pid));
}
function alive(targets) {
  const current = processes();
  return targets.filter(p => current.some(q => p.pid === q.pid && p.command === q.command && !q.stat.startsWith('Z')));
}

(async () => {
  const server = await chromium.launchServer({ headless: true,
    ...(process.env.CHROMIUM_EXECUTABLE ? { executablePath: process.env.CHROMIUM_EXECUTABLE } : {}) });
  const owned = server.process();
  const info = { session: 'memory-budget-review-' + process.pid, controller: process.pid, browser: owned.pid,
    profile: owned.spawnargs.find(a => a.startsWith('--user-data-dir=')) };
  console.log(JSON.stringify(info));
  if (output) { fs.mkdirSync(output, { recursive: true }); fs.writeFileSync(path.join(output, 'session.json'), JSON.stringify(info, null, 2)); }
  let browser;
  const seen = new Map();
  const remember = () => { for (const p of descendants(owned.pid)) seen.set(p.pid, p); };
  remember();
  let targets = [];
  try {
    browser = await chromium.connect(server.wsEndpoint());
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    page.setDefaultTimeout(20000);
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    async function time(value) {
      await page.locator('[name=seek]').evaluate((e, v) => {
        e.value = String(v); e.dispatchEvent(new Event('input', { bubbles: true }));
      }, value);
    }
    async function openMemory(query = '') {
      await page.goto(base + 'sglang/memory-budget/?learn=labs' + query + '#lab');
      await page.locator('[data-memory-overview]').waitFor(); remember();
    }
    async function overviewText() { return page.locator('[data-memory-overview]').innerText(); }
    async function shotMemory(name) {
      if (!output) return;
      await page.locator('#lab').evaluate(e => window.scrollTo({ top: e.getBoundingClientRect().top + scrollY - 90, behavior: 'instant' }));
      await page.screenshot({ path: path.join(output, name + '.png'), fullPage: false });
      await page.locator('[data-memory-overview]').screenshot({ path: path.join(output, name + '-ledger.png') });
    }
    await openMemory();
    assert.match(await overviewText(), /15.*GiB KV 预算/s);
    assert.match(await overviewText(), /245,760/);
    assert.match(await overviewText(), /781 页 = 99,968 槽位/);
    assert.match(await overviewText(), /20 页\/条/);
    assert.match(await overviewText(), /240 页/);
    assert.match(await page.locator('[data-memory-cap]').innerText(), /增加字节预算.*不会突破/s);
    assert.equal(await page.locator('.sl-transport input[type=range]').count(),1,'one effective speed control');
    await shotMemory('default-desktop');
    await page.locator('[name=fraction]').selectOption('0.95');
    assert.match(await overviewText(), /294,912/);
    assert.match(await overviewText(), /781 页 = 99,968 槽位/);
    assert.match(await overviewText(), /新增峰值超出预留/);
    await page.locator('[data-memory-preset=uncap]').click();
    assert.match(await overviewText(), /2,304 页 = 294,912 槽位/);
    assert.equal(await page.locator('[name=fraction]').inputValue(), '0.95');
    assert.equal(await page.locator('[name=tokenLimit]').inputValue(), '0');
    await shotMemory('uncapped-budget');
    await page.locator('[data-memory-preset=boundary]').click();
    assert.equal(await page.locator('[name=context]').inputValue(), '2049');
    assert.match(await overviewText(), /21 页\/条/);
    assert.match(await page.locator('[data-memory-padding]').innerText(), /127/);
    assert.match(await overviewText(), /252 页/);
    assert.equal(await page.locator('.ml-page').count(), 21);
    await shotMemory('per-request-rounding');
    await time(5.25);
    const t = await page.locator('[data-serving-lab]').getAttribute('data-time');
    await page.locator('[data-depth-choice=modules]').click();
    await page.locator('[data-source-toggle]').click();
    assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-time'), t);
    assert.match(await page.locator('#sl-svg').textContent(), /1,524/);
    if(output) await page.locator('.sl-canvas').screenshot({path:path.join(output,'units-and-rounding-modules.png')});
    await page.reload(); await page.locator('[data-serving-lab][data-ready=true]').waitFor();
    assert.equal(await page.locator('[name=context]').inputValue(),'2049');
    assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-time'), t);
    assert.equal(await page.locator('[data-source-toggle]').getAttribute('aria-expanded'), 'true');
    await page.locator('[data-depth-choice=overview]').click();
    await page.locator('[data-action=play]').click(); await page.waitForTimeout(180);
    const moving=+await page.locator('[data-serving-lab]').getAttribute('data-time');
    await page.locator('[data-depth-choice=modules]').click();
    assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-playing'),'true');
    assert.ok(+await page.locator('[data-serving-lab]').getAttribute('data-time')>=moving);
    await page.locator('[data-memory-preset=default]').click();
    assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-time'),'0');
    assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-playing'),'false');
    await page.locator('[name=context]').selectOption('32768');
    assert.match(await page.locator('#sl-svg').textContent(), /还差 2,339 页/);
    await page.locator('[data-depth-choice=overview]').click();
    assert.match(await page.locator('[data-memory-verdict]').innerText(), /超过池容量/);
    await shotMemory('request-shortage');
    await page.locator('[name=available]').selectOption('8');
    await page.locator('[name=fraction]').selectOption('0.75');
    assert.match(await overviewText(), /不能建立正的 KV 预算/);
    assert.match(await overviewText(), /0 页 = 0 槽位/);
    await shotMemory('byte-budget-shortage');
    for (const width of [1440,768,390,320]) {
      await page.setViewportSize({width,height:900});
      await openMemory('&lab-context=2049');
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'page width '+width);
      assert.ok(await page.locator('[data-memory-overview]').evaluate(e=>e.scrollWidth<=e.clientWidth+1),'ledger width '+width);
      await page.locator('details.sl-reading').first().evaluate(e=>e.open=true);
      await page.locator('[name=page]').selectOption('256');
      assert.match(await overviewText(),/11 页\/条/);
      await shotMemory('responsive-'+width);
      await page.locator('details.sl-reading').first().evaluate(e=>e.open=false);
      await page.locator('[data-memory-overview]').scrollIntoViewIfNeeded();
      if(output&&width===390)await page.screenshot({path:path.join(output,'mobile-ledger-viewport.png')});
      await page.locator('#formula').scrollIntoViewIfNeeded();
      if(output&&width===390)await page.screenshot({path:path.join(output,'mobile-math-viewport.png')});
      await page.locator('[data-depth-choice=modules]').click();
      const outside=await page.locator('#sl-svg').evaluate(svg=>{
        const b=svg.viewBox.baseVal;return [...svg.querySelectorAll('text')].filter(e=>{
          const x=e.getBBox();return x.width&&(x.x<0||x.y<0||x.x+x.width>b.width+1||x.y+x.height>b.height+1);
        }).map(e=>e.textContent);
      });assert.deepEqual(outside,[],'module bounds '+width);
      await page.locator('[data-depth-choice=overview]').click();
      await page.locator('.theme-toggle').click();
      if(width===390)await shotMemory('mobile-dark');
      await page.locator('.theme-toggle').click();
      assert.equal(await page.locator('#formula math').count(),3);
    }
    await page.emulateMedia({reducedMotion:'reduce'});await openMemory();
    await time(7.5);await page.locator('[data-action=reset]').click();
    assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-time'),'0');
    const plainContext=await browser.newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});
    const plain=await plainContext.newPage();await plain.goto(base+'sglang/memory-budget/');
    assert.match(await plain.locator('#formula').innerText(),/2561|2,561/);
    assert.equal(await plain.locator('#formula math').count(),3);
    assert.match(await plain.locator('#pages').innerText(),/独立的 P=4/);
    await plainContext.close();
    assert.deepEqual(errors,[]);
    console.log('PASS memory ledger: byte budget vs user cap; independent request rounding; page padding; preset reset; depth/play/URL persistence; 4 widths; themes; MathML; reduced motion and no-JS.');
  } finally {
    remember(); targets = [...seen.values()];
    if (browser) await browser.close().catch(() => {});
    await server.close().catch(() => {});
    for (let i = 0; i < 20 && alive(targets).length; i++) await delay(100);
    for (const p of alive(targets)) process.kill(p.pid, 'SIGTERM');
    for (let i = 0; i < 20 && alive(targets).length; i++) await delay(100);
    for (const p of alive(targets)) process.kill(p.pid, 'SIGKILL');
    await delay(100);
    const remaining = alive(targets);
    console.log(JSON.stringify({ session: info.session, cleaned: remaining.length === 0, remaining }));
    assert.equal(remaining.length, 0, 'owned browser and helpers must exit');
    if (info.profile) {
      const profile = info.profile.split('=').slice(1).join('=');
      if (remaining.length === 0 && profile.startsWith(os.tmpdir()) &&
          path.basename(profile).startsWith('playwright_chromiumdev_profile-')) {
        fs.rmSync(profile, { recursive: true, force: true });
      }
      assert.equal(fs.existsSync(profile), false, 'the owned temporary browser profile must be removed');
    }
  }
})().catch(e => { console.error(e); process.exitCode = 1; });
