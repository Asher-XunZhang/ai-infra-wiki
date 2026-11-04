const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const M = require('../../pages/sglang/kv-memory/model.js');
const base = process.env.OVERVIEW_TEST_URL || 'http://127.0.0.1:8765/';
const retained = !!process.env.KV_QA_DIR;
const artifacts = process.env.KV_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(), 'kv-memory-qa-'));
fs.mkdirSync(artifacts, {recursive:true});
(async () => {
 let server, metadata;
 try {
  server = await chromium.launchServer({headless:true, ...(process.env.CHROMIUM_EXECUTABLE ? {executablePath:process.env.CHROMIUM_EXECUTABLE} : {})});
  const pid = server.process().pid;
  const command = execFileSync('ps', ['-p', String(pid), '-o', 'command='], {encoding:'utf8'});
  metadata = {controller:process.pid, pid, profile:command.match(/--user-data-dir=([^ ]+)/)?.[1], command};
  fs.writeFileSync(path.join(artifacts,'browser.json'), JSON.stringify(metadata,null,2));
  const browser = await chromium.connect(server.wsEndpoint());
  const page = await browser.newPage({viewport:{width:1280,height:800}});
  const errors = [];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base+'sglang/kv-memory/');
  await page.evaluate(()=>document.fonts.ready);
  await page.clock.install({time:new Date('2026-01-01T00:00:00Z')});
  await page.clock.pauseAt(new Date('2026-01-01T00:00:01Z'));
  const lab=page.locator('[data-memory-lab]');
  const choose=async mode=>page.locator(`button[data-mode="${mode}"]`).click();
  const seek=async t=>page.locator('#memory-seek').evaluate((el,t)=>{el.value=t;el.dispatchEvent(new Event('input',{bubbles:true}));},t);
  const currentTime=async()=>Number(await lab.getAttribute('data-time'));
  const state=async()=>page.locator('#memory-scene-layer').innerHTML();
  const screenshot=async name=>{
   if(!retained)return;
   const box=await lab.boundingBox();
   await page.evaluate(y=>window.scrollTo({top:y-76,behavior:'instant'}),box.y+await page.evaluate(()=>scrollY));
   await page.screenshot({path:path.join(artifacts,name+'.png')});
  };
  assert.equal(await page.locator('.mechanism-film,.batch-extra,.animation-speed').count(),0,'one playback owner');
  assert.equal(await lab.isVisible(),true);
  for(const target of [8,9,11]) {
   await page.selectOption('#extend-length',String(target));
   await seek(2.3);
   assert.equal(await page.locator('[data-map="R1:6"]').getAttribute('data-slot'),'');
   assert.equal(await page.locator('[data-slot-id="30"]').getAttribute('data-state'),'reserved');
   await seek(3.3);
   assert.equal(await page.locator('[data-map="R1:6"]').getAttribute('data-slot'),'30');
   await seek(5.1);
   assert.equal(await page.locator('.memory-packet.data').count(),target-6);
   assert.equal(await page.locator('[data-slot-id="30"]').getAttribute('data-state'),'reserved');
   if(target===11)await screenshot('write-in-flight');
   await seek(6.1);
   assert.equal(await page.locator('[data-slot-id="30"]').getAttribute('data-state'),'written');
   if(target===8)assert.equal(await page.locator('[data-page="3"]').getAttribute('data-owner'),'free');
  }
  // Pausing freezes both the explanation clock and moving payload coordinates.
  await seek(4.4);await page.locator('#memory-play').click();await page.clock.runFor(400);await page.locator('#memory-play').click();
  const paused=await currentTime(),positions=await page.locator('#memory-flow-layer').innerHTML();
  await page.clock.runFor(1200);assert.equal(await currentTime(),paused);assert.equal(await page.locator('#memory-flow-layer').innerHTML(),positions);
  await page.selectOption('#memory-speed','2');assert.equal(await currentTime(),paused);
  await page.locator('#memory-play').click();await page.clock.runFor(600);await page.locator('#memory-play').click();
  const accelerated=await currentTime()-paused;
  assert.ok(accelerated>1 && accelerated<1.25,`2x advanced ${accelerated}s`);
  await page.locator('#memory-play').click();await page.selectOption('#extend-length','9');
  assert.equal(await currentTime(),0);assert.equal(await lab.getAttribute('data-playing'),'false');
  await page.selectOption('#memory-speed','1');
  for(const mode of ['extend','reuse','sharing']) {
   await choose(mode);
   const options={target:Number(await page.locator('#extend-length').inputValue()),hit:Number(await page.locator('#reuse-length').inputValue())};
   const tl=M.timeline(mode,options),snapshots=[];
   for(const phase of tl.phases)for(const event of phase.events){const t=phase.start+event.at+.02;await seek(t);snapshots.push([t,await state()]);}
   for(const [t,html] of snapshots.reverse()){await seek(t);assert.equal(await state(),html,`${mode}: reverse seek`);}
   await page.locator('#memory-reset').click();await page.locator('#memory-play').click();
   for(const phase of tl.phases){await page.clock.runFor(Math.ceil(phase.duration*1000));}
   await page.clock.runFor(80);
   assert.equal(await lab.getAttribute('data-playing'),'false');assert.equal(await currentTime(),Number(tl.duration.toFixed(3)));
   assert.equal(await page.locator('#memory-play').innerText(),'重新播放');
   assert.equal(await lab.getAttribute('data-phase'),String(tl.phases.length-1));
  }
  await choose('reuse');
  for(const hit of [0,4,8]){
   await page.selectOption('#reuse-length',String(hit));await seek(6.5);
   assert.equal(await page.locator('[data-map]:not([data-slot=""])').count(),hit);
   assert.equal(await page.locator('[data-state="written"]').count(),8);
   assert.equal(await page.locator('#memory-scene-layer [data-page]').count(),2);
  }
  await seek(2.4);await screenshot('reuse-indices');
  await choose('sharing');
  for(const [time,name]of [[8.1,'separate-release'],[15,'stale-after-free'],[17.1,'reserved-before-overwrite'],[19.4,'space-reused']]){await seek(time);await screenshot(name);}
  await seek(15);assert.equal(await page.locator('[data-slot-id="8"]').getAttribute('data-state'),'free');assert.equal(await page.locator('[data-slot-id="8"]').getAttribute('data-value'),'0');
  await seek(17.1);assert.equal(await page.locator('[data-slot-id="8"]').getAttribute('data-state'),'reserved');
  await seek(19.4);assert.equal(await page.locator('[data-slot-id="8"]').getAttribute('data-value'),'Z');
  await page.locator('#memory-reset').focus();await page.keyboard.press('Home');assert.equal(await currentTime(),0);
  await page.keyboard.press('End');assert.equal(await currentTime(),19.4);
  await page.keyboard.press('ArrowLeft');assert.equal(await currentTime(),16);
  // Reduced motion uses static labeled arrows while retaining playback/seek semantics.
  await choose('extend');await page.emulateMedia({reducedMotion:'reduce'});await seek(5);
  const reduced=await page.locator('#memory-flow-layer').innerHTML();await seek(5.2);
  assert.equal(await page.locator('#memory-flow-layer').innerHTML(),reduced);
  assert.ok(await page.locator('.memory-flow-line[marker-end]').count()>0);
  await page.emulateMedia({reducedMotion:'no-preference'});await page.clock.runFor(40);
  // Geometry in the actual navigation shell, light/dark and narrow layouts.
  for(const width of [1440,1280,820,390,320])for(const theme of ['light','dark']){
   await page.setViewportSize({width,height:900});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
   for(const mode of ['extend','reuse','sharing']){
    await choose(mode);await seek(mode==='extend'?5.1:mode==='reuse'?2.3:8.1);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${width}/${theme}/${mode}`);
    const outside=await page.locator('#memory-scene-layer text').evaluateAll(es=>es.filter(e=>{const b=e.getBBox();return b.x<0||b.x+b.width>920||b.y<0||b.y+b.height>430}).map(e=>e.textContent));
    assert.deepEqual(outside,[],`${width}/${theme}/${mode}: SVG text bounds`);
    assert.deepEqual(await page.locator('.memory-flow-line').evaluateAll(es=>es.filter(e=>!/^[ML\d., -]+$/.test(e.getAttribute('d'))).map(e=>e.getAttribute('d'))),[]);
    if(mode==='extend'&&theme==='dark'&&[1280,390].includes(width))await screenshot(`dark-${width}`);
   }
  }
  await page.goto(base+'sglang/kv-memory/?mode=extend&target=8&t=6.3');
  assert.equal(await currentTime(),6.3);assert.equal(await page.locator('#extend-length').inputValue(),'8');
  await page.evaluate(()=>{history.pushState(null,'','?mode=sharing&t=17.1');dispatchEvent(new PopStateEvent('popstate'));});
  assert.equal(await lab.getAttribute('data-mode'),'sharing');assert.equal(await currentTime(),17.1);
  assert.equal(await page.locator('[data-slot-id="8"]').getAttribute('data-state'),'reserved');
  const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto(base+'sglang/kv-memory/');
  assert.equal(await plain.locator('[data-memory-lab]').isVisible(),false);await plain.locator('.topology-detail summary').first().click();
  assert.equal(await plain.locator('.memory-static').first().isVisible(),true);await plain.close();
  assert.deepEqual(errors,[]);await browser.close();
  console.log('PASS: allocation/map/write causality, all conditions, continuous playback, pause/speed/reverse seek, stale data and reuse, keyboard/URL/reduced motion/no-JS, five widths × two themes.');
 } finally {
  if(server)await server.close();
  if(metadata){
   metadata.browserExited=server.process().exitCode!==null||server.process().signalCode!==null;
   metadata.profileRemoved=!fs.existsSync(metadata.profile);
   const processes=execFileSync('ps',['-axo','pid=,command='],{encoding:'utf8'});
   metadata.remaining=processes.split('\n').filter(s=>metadata.profile&&s.includes(metadata.profile));
   fs.writeFileSync(path.join(artifacts,'browser.json'),JSON.stringify(metadata,null,2));
   assert.equal(metadata.browserExited,true);assert.equal(metadata.profileRemoved,true);assert.deepEqual(metadata.remaining,[]);
  }
  if(!retained)fs.rmSync(artifacts,{recursive:true,force:true});
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
