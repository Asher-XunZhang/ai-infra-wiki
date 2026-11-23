const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), os = require('node:os');
const cp = require('node:child_process'), zlib = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.OVERVIEW_TEST_URL || 'http://127.0.0.1:8765/';
const output = process.env.HICACHE_QA_DIR;
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
  const info = { session: 'hicache-review-' + process.pid, controller: process.pid, browser: owned.pid,
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
    async function openHi(query='') {
      await page.goto(base+'sglang/hicache-loadback/?learn=labs'+query+'#lab');
      await page.locator('[data-serving-lab][data-ready=true]').waitFor();
      await page.locator('.sl-overview-canvas svg').waitFor();remember();
    }
    async function flowState(selector='#sl-svg') {
      return page.locator(selector+' [data-flow][data-active=true]').evaluateAll(es=>es.map(e=>({id:e.dataset.flow,
        path:e.querySelector('path').getAttribute('d'),phase:e.querySelector('path').getAttribute('stroke-dashoffset')})));
    }
    async function shotHi(name,modules=false) {
      if(!output)return;
      await page.locator(modules?'.sl-canvas':'.sl-overview-canvas').screenshot({path:path.join(output,name+'.png')});
    }
    const modelState=()=>page.evaluate(()=>ServingLabModel.hicacheState(ServingLabModel.hicache(document.querySelector('[name=scenario]').value),+document.querySelector('[data-serving-lab]').dataset.time));
    await openHi();
    if(output){await page.locator('#lab').evaluate(e=>scrollTo({top:e.getBoundingClientRect().top+scrollY-100,behavior:'instant'}));await page.screenshot({path:path.join(output,'lesson-first-viewport.png')});}
    assert.equal(await page.locator('.sl-transport input[type=range]').count(),1,'one real speed control');
    await page.locator('[data-hicache-moment=hit]').click();
    assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-time'),'2');
    assert.match(await page.locator('#sl-stage').innerText(),/只确认存在/);
    assert.deepEqual(await flowState(),[]);
    await shotHi('hit-is-not-readable');
    await time(4);let st=await modelState();assert.equal(st.ioOwns,true);assert.equal(st.hostValid,false);assert.equal(st.hostTransferPinned,false);
    await time(7.5);st=await modelState();assert.equal(st.storageReadDone,true);assert.equal(st.hostValid,false);assert.equal(st.ioOwns,true);assert.equal(st.storageAck,false);
    assert.match(await page.locator('#sl-stage').innerText(),/尚未被缓存侧采用/);await shotHi('read-done-before-adoption');
    await time(8.5);st=await modelState();assert.equal(st.storageAck,true);assert.equal(st.hostTransferPinned,true);assert.equal(st.deviceAllocated,false);
    await page.locator('[data-source-toggle]').click();assert.match(await page.locator('.ld-source-panel a').getAttribute('href'),/#L1472$/);
    await time(10.25);const pulse=await flowState();assert.equal(pulse[0].id,'copy-0');
    await delay(120);assert.deepEqual(await flowState(),pulse,'pause freezes phase');
    await time(10.5);assert.equal((await flowState())[0].path,pulse[0].path);assert.notEqual((await flowState())[0].phase,pulse[0].phase);
    await time(10.25);assert.deepEqual(await flowState(),pulse,'rewind restores phase');
    await time(11.99);assert.match(await page.locator('#sl-svg [data-hc-tier="2"][data-hc-layer="0"] text').textContent(),/尚不可读/);
    await time(12.5);st=await modelState();assert.deepEqual(st.readyLayers,[0]);assert.deepEqual(st.activeLayers,[0]);assert.equal((await flowState())[0].id,'copy-1');
    const paused={time:await page.locator('[data-serving-lab]').getAttribute('data-time'),results:await page.locator('#sl-results').innerText(),phase:(await flowState())[0].phase};
    await shotHi('layer0-read-with-layer1-copy');
    await page.locator('[data-depth-choice=modules]').click();
    assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-time'),paused.time);assert.equal(await page.locator('#sl-results').innerText(),paused.results);
    assert.equal((await flowState())[0].phase,paused.phase);
    // Every H2D arrow uses the exact source and destination cell ports.
    const ports=await page.locator('#sl-svg').evaluate(svg=>[0,1,2].map(layer=>{
      const a=svg.querySelector('[data-hc-tier="1"][data-hc-layer="'+layer+'"] .hc-cell'),b=svg.querySelector('[data-hc-tier="2"][data-hc-layer="'+layer+'"] .hc-cell');
      const d=svg.querySelector('[data-flow="copy-'+layer+'"] path').getAttribute('d');return {d,start:[+a.getAttribute('x') + +a.getAttribute('width'),+a.getAttribute('y') + +a.getAttribute('height')/2],end:[+b.getAttribute('x'),+b.getAttribute('y') + +b.getAttribute('height')/2]};
    }));for(const a of ports)assert.deepEqual(a.d.match(/-?\d+(?:\.\d+)?/g).map(Number),[...a.start,...a.end]);
    await shotHi('module-ports-and-ownership',true);
    await page.reload();await page.locator('[data-serving-lab][data-ready=true]').waitFor();assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-time'),paused.time);
    assert.equal(await page.locator('[data-source-toggle]').getAttribute('aria-expanded'),'true');
    await page.locator('[data-action=play]').click();await page.waitForTimeout(200);const moving=+await page.locator('[data-serving-lab]').getAttribute('data-time');
    await page.locator('[name=speed]').fill('1.75');await page.locator('[data-depth-choice=overview]').click();assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-playing'),'true');
    assert.ok(+await page.locator('[data-serving-lab]').getAttribute('data-time')>=moving);await page.locator('[data-action=play]').click();
    await page.locator('[data-hicache-moment=copied]').click();st=await modelState();assert.equal(st.ackPending,true);assert.equal(st.deviceTransferPinned,true);assert.equal(st.requestPinned,true);
    assert.deepEqual(await flowState(),[],'no transport while only ACK consumption is pending');assert.match(await page.locator('#sl-stage').innerText(),/没有 KV 正在传输/);await shotHi('copied-but-ack-pending');
    assert.match(await page.locator('.ld-source-panel a').getAttribute('href'),/#L2855$/);
    await page.locator('[data-hicache-moment=ack]').click();st=await modelState();assert.equal(st.ack,true);assert.equal(st.hostTransferPinned,false);assert.equal(st.requestPinned,true);assert.equal(st.hostValid,true);
    assert.deepEqual(st.activeLayers,[2]);assert.match(await page.locator('#sl-stage').innerText(),/仍持有 L1 前缀/);await shotHi('ack-does-not-release-request');
    await page.locator('[data-hicache-moment=end]').click();assert.equal((await modelState()).requestPinned,true);assert.match(await page.locator('#sl-stage').innerText(),/没有结束 R1/);
    await page.locator('[name=scenario]').selectOption('cancel');assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-time'),'0');assert.equal(await page.locator('[data-serving-lab]').getAttribute('data-playing'),'false');
    assert.equal(await page.locator('[data-hicache-moment=ack]').isDisabled(),true);
    await time(6);st=await modelState();assert.equal(st.canceled,true);assert.equal(st.ioOwns,true);assert.equal((await flowState())[0].id,'storage-read');await shotHi('cancel-still-owns-io-target');
    await time(8.25);st=await modelState();assert.equal(st.storageAck,true);assert.equal(st.ioOwns,false);assert.equal(st.releaseQueued,true);assert.equal(st.hostReleased,false);assert.match(await page.locator('#sl-stage').innerText(),/尚待实际归还/);await shotHi('final-ack-before-pool-free');
    await time(9);st=await modelState();assert.equal(st.hostReleased,true);assert.equal(st.ioOwns,false);assert.equal(st.hostValid,false);assert.equal(st.deviceAllocated,false);assert.match(await page.locator('.sl-overview-canvas').innerText(),/目标已归还/);await shotHi('cancel-after-final-ack');
    await time(4);assert.equal((await modelState()).canceled,false);assert.equal((await modelState()).hostReleased,false);
    await page.locator('[name=scenario]').selectOption('device-shortage');await time(8.5);assert.equal((await modelState()).hostTransferPinned,true);
    await time(9.5);st=await modelState();assert.equal(st.hostValid,true);assert.equal(st.hostTransferPinned,false);assert.equal(st.deviceAllocated,false);assert.deepEqual(await flowState(),[]);await shotHi('allocation-failure-unpins');
    await page.locator('[name=scenario]').selectOption('miss');await time(4);assert.deepEqual(await flowState(),[]);assert.equal((await modelState()).storageHitKnown,false);assert.match(await page.locator('#sl-stage').innerText(),/不走回载/);
    await page.locator('[name=scenario]').selectOption('host');await page.locator('[data-hicache-moment=layer]').click();assert.equal((await modelState()).storageAck,false);assert.equal((await modelState()).hostValid,true);await shotHi('host-hit-skips-l3');
    for(const width of [1440,768,390,320]){
      await page.setViewportSize({width,height:900});await openHi('&t=12.5');
      if(output&&width===390){await page.locator('#lab').evaluate(e=>scrollTo({top:e.getBoundingClientRect().top+scrollY-100,behavior:'instant'}));await page.screenshot({path:path.join(output,'mobile-first-viewport.png')});}
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'page bounds '+width);
      for(const mode of ['overview','modules']){
        await page.locator('[data-depth-choice='+mode+']').click();const selector=mode==='overview'?'.sl-overview-canvas svg':'#sl-svg';
        const outside=await page.locator(selector).evaluate(svg=>{const b=svg.viewBox.baseVal;return [...svg.querySelectorAll('text')].filter(e=>{const x=e.getBBox();return x.width&&(x.x<0||x.y<0||x.x+x.width>b.width+1||x.y+x.height>b.height+1)}).map(e=>e.textContent)});
        assert.deepEqual(outside,[],'text bounds '+width+' '+mode);
      }
      await page.locator('[data-depth-choice=overview]').click();
      if(width===390){await page.locator('.sl-overview-canvas').scrollIntoViewIfNeeded();if(output)await page.screenshot({path:path.join(output,'mobile-viewport.png')});}
      await page.locator('.theme-toggle').click();if(width===1440)await shotHi('overview-dark');await page.locator('.theme-toggle').click();
    }
    await page.emulateMedia({reducedMotion:'reduce'});await time(12.5);assert.equal(await page.locator('#sl-svg [data-active=true] path[stroke-dashoffset]').count(),0);assert.match(await page.locator('#sl-activity').innerText(),/静态高亮.*传递中/);
    const plainContext=await browser.newContext({javaScriptEnabled:false,viewport:{width:390,height:844}}),plain=await plainContext.newPage();await plain.goto(base+'sglang/hicache-loadback/');assert.match(await plain.locator('#ownership').innerText(),/传输/);assert.ok(await plain.locator('#check details').count()>=3);await plainContext.close();
    assert.deepEqual(errors,[]);console.log('PASS HiCache gates, both ACKs, reversible ownership, active-only sustained pulses, exact per-layer ports, one clock/speed, state-preserving depth/URL, 5 scenarios, 4 widths, themes, reduced motion and no-JS.');
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
