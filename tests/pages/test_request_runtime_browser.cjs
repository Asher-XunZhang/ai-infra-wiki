const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const {execFileSync}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const F=require('../../pages/sglang/request-runtime/flow.js');
(async()=>{
 const artifact=process.env.RUNTIME_SCREENSHOT_DIR||fs.mkdtempSync(path.join(os.tmpdir(),'runtime-flow-qa-'));
 fs.mkdirSync(artifact,{recursive:true});
 const server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 const child=server.process(),profile=child.spawnargs.find(x=>x.startsWith('--user-data-dir='))?.slice(16),record={controller:process.pid,browser:child.pid,profile};
 const browser=await chromium.connect(server.wsEndpoint());
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
  await page.goto(base+'sglang/request-runtime/');await page.waitForLoadState('load');
  assert.ok(await page.locator('[data-runtime-lab]').isVisible());
  assert.equal(await page.locator('.mechanism-film').count(),0,'one player for the whole lifecycle');
  assert.equal(await page.locator('.animation-speed').count(),0,'legacy global player cannot inject another speed');
  const seek=async t=>page.locator('#step').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));},t);
  const observed=()=>page.locator('#runtime-snapshot').evaluate(e=>({...e.dataset}));
  const geometry=[];
  for(const scenario of ['normal','queued-abort','running-abort','invalid'])for(const cache of [true,false])for(const outputs of scenario==='normal'?[1,3,5]:[3]){
   await page.selectOption('#scenario',scenario);await page.locator('#cache').setChecked(cache);if(scenario==='normal')await page.selectOption('#outputs',String(outputs));
   const tl=F.build({scenario,cache,outputs});
   for(const p of tl.phases){
    await seek(p.start+p.duration*.98);const v=await observed();
    assert.equal(v.frame,p.id);for(const key of ['kv','cached','sampled','visible'])assert.equal(Number(v[key]),p.frame[key]);
    assert.equal(await page.locator('.rf-node').count(),11,'architecture remains visible throughout');
    geometry.push(...await page.locator('.rf-node').evaluateAll(nodes=>nodes.flatMap(n=>{const box=n.querySelector('.rf-node-bg').getBBox();return [...n.querySelectorAll('text')].filter(t=>getComputedStyle(t).display!=='none'&&t.textContent&&t.closest('.rf-cell')?.style.display!=='none').flatMap(t=>{const r=t.getBBox();return r.width>box.width-15?[{node:n.dataset.node,text:t.textContent,width:r.width,available:box.width-15}]:[];});})));
   }
   await seek(tl.duration);assert.equal((await observed()).frame,tl.phases.at(-1).id);
   await seek(0);assert.equal((await observed()).sampled,'0');assert.equal((await observed()).cached,'0');
  }
  assert.deepEqual(geometry,[],'node labels fit without clipping');
  await page.selectOption('#scenario','normal');await page.selectOption('#outputs','3');await page.locator('#cache').check();
  const tl=F.build(),event=tl.events.find(e=>e.edge==='dispatch');
  const positions=[];
  for(const u of [.2,.5,.8]){await seek(event.start+(event.end-event.start)*u);positions.push(await page.locator('.rf-packet[visibility="visible"]').first().evaluate(e=>({transform:e.getAttribute('transform'),edge:e.dataset.edge})));}
  assert.equal(new Set(positions.map(p=>p.transform)).size,3,'mid-path positions move continuously');assert.ok(positions.every(p=>p.edge==='dispatch'));
  await seek(event.start+(event.end-event.start)*.5);assert.deepEqual(await page.locator('.rf-packet[visibility="visible"]').first().evaluate(e=>({transform:e.getAttribute('transform'),edge:e.dataset.edge})),positions[1],'reverse seek exactly restores geometry');
  await page.locator('#reset').click();await page.clock.install();await page.selectOption('#runtime-speed','2');await page.locator('#runtime-toggle').click();
  await page.clock.runFor(2700);assert.ok(Number(await page.locator('[data-runtime-lab]').getAttribute('data-time'))>5);
  await page.locator('#runtime-toggle').click();const paused=await page.locator('[data-runtime-lab]').getAttribute('data-time');await page.clock.runFor(2000);assert.equal(await page.locator('[data-runtime-lab]').getAttribute('data-time'),paused);
  await page.locator('#runtime-toggle').click();await page.clock.runFor(35000);assert.equal(await page.locator('#runtime-toggle').getAttribute('aria-pressed'),'false');assert.equal((await observed()).visible,'3');assert.equal((await observed()).cached,'6');
  await page.locator('#runtime-toggle').click();await page.clock.runFor(1000);await page.selectOption('#scenario','running-abort');assert.equal(await page.locator('[data-runtime-lab]').getAttribute('data-time'),'0.000');assert.equal(await page.locator('#runtime-toggle').getAttribute('aria-pressed'),'false');
  await page.selectOption('#scenario','normal');await page.locator('#step').focus();await page.keyboard.press('ArrowRight');assert.ok(Number(await page.locator('#step').inputValue())>0);
  await page.emulateMedia({reducedMotion:'reduce'});await seek(event.start+1);assert.equal(await page.locator('.rf-packet[visibility="visible"]').count(),0);assert.ok(await page.locator('.rf-wire.is-active').count());await page.emulateMedia({reducedMotion:'no-preference'});
  for(const width of [1440,768,390,320])for(const theme of ['light','dark']){
   await page.setViewportSize({width,height:1100});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
   await seek(20.3);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
   if(width===1440||width===390)await page.locator('[data-runtime-lab]').screenshot({path:path.join(artifact,`${width}-${theme}.png`)});
  }
  await page.setViewportSize({width:1440,height:1100});await page.evaluate(()=>document.documentElement.dataset.theme='light');
  for(const [name,t]of [['initial',0],['return',29.5],['handoff',60.9],['final',tl.duration]]){await seek(t);await page.locator('[data-runtime-lab]').screenshot({path:path.join(artifact,name+'.png')});}
  const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto(base+'sglang/request-runtime/');await plain.locator('.runtime-reading-details>summary').click();assert.ok(await plain.locator('.runtime-map').isVisible());assert.equal(await plain.locator('.runtime-source-grid a').count(),6);await plain.close();
  assert.deepEqual(errors,[]);record.result='PASS';record.motion=positions;record.layouts='1440,768,390,320 × light/dark';
  console.log('PASS: all paths and phase boundaries; continuous motion, backward seek, full playback, pause, speed, reset, keyboard, reduced motion, layouts and no-JS reading.');
 }finally{
  await browser.close();await server.close();
  let alive=false;try{process.kill(child.pid,0);alive=true;}catch{}
  record.cleanup={browserExited:!alive,profileRemoved:profile?!fs.existsSync(profile):null,profileProcesses:profile?execFileSync('ps',['-axo','pid=,ppid=,command='],{encoding:'utf8'}).split('\n').filter(s=>s.includes(profile)):[]};
  fs.writeFileSync(path.join(artifact,'review.json'),JSON.stringify(record,null,2));
  assert.equal(alive,false);assert.deepEqual(record.cleanup.profileProcesses,[]);
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
