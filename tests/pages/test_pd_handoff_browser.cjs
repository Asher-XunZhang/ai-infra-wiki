const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {execFileSync}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const M=require('../../pages/sglang/inference-overview/pd-handoff-model.js');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
const retained=!!process.env.PD_HANDOFF_QA_DIR,dir=process.env.PD_HANDOFF_QA_DIR||fs.mkdtempSync(path.join(os.tmpdir(),'pd-handoff-qa-'));
fs.mkdirSync(dir,{recursive:true});
(async()=>{
 let server,metadata;
 try{
  server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
  const pid=server.process().pid,command=execFileSync('ps',['-p',String(pid),'-o','command='],{encoding:'utf8'});
  metadata={session:'pd-handoff-qa',controller:process.pid,pid,profile:command.match(/--user-data-dir=([^ ]+)/)?.[1],command};
  fs.writeFileSync(path.join(dir,'browser.json'),JSON.stringify(metadata,null,2));
  const browser=await chromium.connect(server.wsEndpoint()),page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400&&r.url().startsWith(base))errors.push(r.status()+' '+r.url());});
  await page.goto(base+'sglang/inference-overview/deployment.html#pd-handoff-title');await page.evaluate(()=>document.fonts.ready);
  const scene=page.locator('#pd-handoff-scene'),toggle=page.locator('#pd-handoff-toggle');
  await page.waitForFunction(()=>document.querySelector('#pd-handoff-scene').children.length>0);
  await page.locator('#pd-handoff-title').evaluate(e=>e.scrollIntoView({block:'start',behavior:'instant'}));
  if(retained)await page.screenshot({path:path.join(dir,'default-viewport.png')});
  assert.ok(await scene.evaluate(e=>e.getBoundingClientRect().bottom<=innerHeight),'complete desktop first-viewport diagram and tracks');
  await page.clock.install({time:new Date('2026-01-01T00:00:00Z')});await page.clock.pauseAt(new Date('2026-01-01T00:00:01Z'));
  const choose=option=>page.selectOption('#pd-handoff-case',option);
  const seek=t=>page.locator('#pd-handoff-seek').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));},t);
  const time=async()=>Number(await scene.getAttribute('data-time'));
  const frame=async name=>{if(retained)await scene.screenshot({path:path.join(dir,name+'.png')});};
  const bounds=async label=>{
   const outside=await scene.locator('text').evaluateAll(es=>es.filter(e=>{const b=e.getBBox();return b.x<0||b.x+b.width>960||b.y<0||b.y+b.height>700;}).map(e=>e.textContent));
   assert.deepEqual(outside,[],label+': SVG bounds');
  };
  for(const option of Object.keys(M.cases)){
   await choose(option);assert.equal(await time(),0);const tl=M.build(option),snapshots=[];
   for(const phase of tl.phases){await seek(phase.start+.2);await bounds(option+'/'+phase.title);snapshots.push([await time(),await scene.innerHTML()]);
    const refs=await page.locator('#pd-handoff-source a').evaluateAll(es=>es.map(e=>e.href));assert.ok(refs.length&&refs.every(u=>u.includes(M.revision)));}
   for(const [t,html]of snapshots.reverse()){await seek(t);assert.equal(await scene.innerHTML(),html,'reverse seeks must reconstruct identical state');}
   for(const e of tl.events){await seek((e.start+e.end)/2);assert.equal(await scene.locator(`[data-event="${e.id}"]`).getAttribute('data-active'),'true');assert.equal(Number(await scene.locator('.pd-tracks').getAttribute('data-clock')),await time());}
   await page.locator('#pd-handoff-reset').click();await toggle.click();await page.clock.runFor(Math.ceil(tl.duration*1000)+100);
   assert.equal(await time(),tl.duration,option+': play to end');assert.equal(await toggle.getAttribute('aria-pressed'),'false');await frame(option+'-end');
  }
  await choose('normal');await seek(12);assert.equal(await scene.locator('[data-device=p]').getAttribute('data-running'),'compute2');assert.equal(await scene.locator('[data-event=copy1]').getAttribute('data-active'),'true');
  const before=await scene.locator('[data-motion=copy1]').getAttribute('data-x');await frame('compute-transfer-overlap');await seek(13);assert.notEqual(await scene.locator('[data-motion=copy1]').getAttribute('data-x'),before,'continuous load movement');
  await seek(4.6);await frame('destination-indices');await seek(26);await frame('prebuilt');await seek(13);
  await toggle.click();await page.clock.runFor(450);await toggle.click();const paused=await time(),html=await scene.innerHTML();await page.clock.runFor(900);assert.equal(await time(),paused);assert.equal(await scene.innerHTML(),html);
  await page.selectOption('#pd-handoff-rate','2');await toggle.click();await page.clock.runFor(600);await toggle.click();assert.ok(await time()-paused>1&&await time()-paused<1.3);
  await toggle.click();await choose('metadata');await page.clock.runFor(600);assert.equal(await time(),0,'condition change resets and stops');
  await seek(25);assert.match(await scene.textContent(),/room=0/);assert.equal(await scene.locator('[data-device=d]').getAttribute('data-running'),'');await frame('metadata-gate');
  await choose('latepoll');await seek(28);assert.equal(await scene.locator('[data-device=d]').getAttribute('data-running'),'decode');assert.match(await page.locator('#pd-handoff-proof').innerText(),/P 本侧收尾\s*否/);await frame('independent-cleanup');
  await page.emulateMedia({reducedMotion:'reduce'});assert.equal(await scene.locator('[data-motion]').count(),0);await seek(12);await bounds('reduced');await page.emulateMedia({reducedMotion:'no-preference'});
  await page.locator('#pd-handoff-reset').click();await page.locator('#pd-handoff-next').focus();await page.keyboard.press('Enter');assert.ok(await time()>0,'keyboard step');
  for(const width of [1440,1280,942,768,390,320])for(const theme of ['light','dark']){
   await page.setViewportSize({width,height:1000});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await choose('normal');await seek(12);await bounds(`${width}/${theme}`);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'no page overflow');
   if(width>=768)assert.equal(await page.locator('.pd-handoff-scroll').evaluate(e=>e.scrollWidth>e.clientWidth),false);
   else{assert.equal(await page.locator('.pd-handoff-pan').isVisible(),true);await page.locator('.pd-handoff-scroll').evaluate(e=>e.scrollLeft=400);assert.ok(await page.locator('.pd-handoff-scroll').evaluate(e=>e.scrollLeft)>0);}
   if(retained&&theme==='light'&&[1440,390].includes(width)){await page.locator('.pd-handoff-scroll').evaluate(e=>e.scrollLeft=0);await page.locator('#pd-handoff-title').evaluate(e=>e.scrollIntoView({block:'start',behavior:'instant'}));await page.screenshot({path:path.join(dir,`page-${width}.png`)});}
  }
  const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto(base+'sglang/inference-overview/deployment.html');assert.equal(await plain.locator('[data-pd-handoff]').isVisible(),false);assert.equal(await plain.locator('#handoff-boundaries').isVisible(),true);await plain.close();
  await page.goto(base+'sglang/inference-overview/deployment.html?pdcase=latepoll#pd-handoff-title');assert.equal(await scene.getAttribute('data-case'),'latepoll');assert.equal(await time(),0);
  assert.deepEqual(errors,[]);await browser.close();console.log('PASS: four PD conditions, concurrent work, continuous transfers, reversible same-clock state, complete playback, keyboard, reduced motion, no-JS, six widths/two themes.');
 }finally{
  if(server)await server.close();
  if(metadata){metadata.browserExited=server.process().exitCode!==null||server.process().signalCode!==null;metadata.profileRemoved=!fs.existsSync(metadata.profile);metadata.remaining=execFileSync('ps',['-axo','pid=,command='],{encoding:'utf8'}).split('\n').filter(s=>metadata.profile&&s.includes(metadata.profile));fs.writeFileSync(path.join(dir,'browser.json'),JSON.stringify(metadata,null,2));assert.ok(metadata.browserExited&&metadata.profileRemoved);assert.deepEqual(metadata.remaining,[]);}
  if(!retained)fs.rmSync(dir,{recursive:true,force:true});
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
