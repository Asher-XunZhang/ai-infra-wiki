const assert=require('node:assert/strict');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {execFileSync}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const M=require('../../pages/sglang/parallelism/flow-model.js');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
const retained=!!process.env.PARALLEL_QA_DIR;
const artifacts=process.env.PARALLEL_QA_DIR||fs.mkdtempSync(path.join(os.tmpdir(),'parallel-qa-'));
fs.mkdirSync(artifacts,{recursive:true});
(async()=>{
 let server,metadata;
 try{
  server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
  const pid=server.process().pid,command=execFileSync('ps',['-p',String(pid),'-o','command='],{encoding:'utf8'});
  metadata={session:'parallel-execution-lanes-qa',controller:process.pid,pid,profile:command.match(/--user-data-dir=([^ ]+)/)?.[1],command};
  fs.writeFileSync(path.join(artifacts,'browser.json'),JSON.stringify(metadata,null,2));
  const browser=await chromium.connect(server.wsEndpoint()),page=await browser.newPage({viewport:{width:1280,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400&&r.url().startsWith(base))errors.push(r.status()+' '+r.url());});
  await page.goto(base+'sglang/parallelism/#parallel-title');await page.evaluate(()=>document.fonts.ready);
  await page.waitForFunction(()=>{const r=document.querySelector('#parallel-title').getBoundingClientRect();return r.top>=75&&r.top<150&&document.querySelector('#parallel-scene').children.length>0;});if(retained)await page.screenshot({path:path.join(artifacts,'default-viewport.png')});
  await page.clock.install({time:new Date('2026-01-01T00:00:00Z')});await page.clock.pauseAt(new Date('2026-01-01T00:00:01Z'));
  const scene=page.locator('#parallel-scene'),toggle=page.locator('#parallel-toggle');
  const choose=async mode=>page.locator(`button[data-mode="${mode}"]`).click();
  const seek=async t=>page.locator('#parallel-seek').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));},t);
  const time=async()=>Number(await scene.getAttribute('data-time'));
  const sceneText=async()=>scene.textContent();
  const frame=async name=>{if(retained)await scene.screenshot({path:path.join(artifacts,name+'.png')});};
  const bounds=async label=>{
   const outside=await scene.locator('text').evaluateAll(es=>es.filter(e=>{const b=e.getBBox();return b.x<0||b.x+b.width>900||b.y<0||b.y+b.height>e.ownerSVGElement.viewBox.baseVal.height;}).map(e=>e.textContent));
   assert.deepEqual(outside,[],label+': SVG text bounds');
  };
  assert.equal(await page.locator('.animation-speed,.mechanism-film').count(),0,'only one playback owner');
  assert.equal(await page.locator('[data-parallel-lab]').isVisible(),true);
  for(const mode of M.modes){
   await choose(mode);await page.locator('#parallel-reset').click();const tl=M.build(mode),snapshots=[];
   assert.equal(await time(),0);assert.equal(await scene.getAttribute('data-mode'),mode);
   assert.equal(await page.locator('#parallel-option-wrap').isVisible(),['dp','sp','pcp'].includes(mode));
   for(const phase of tl.phases){
    for(const progress of [.05,.5,1]){
     const t=Number((phase.start+phase.duration*progress).toFixed(2));await seek(t);await bounds(`${mode}/${phase.id}/${progress}`);
     snapshots.push([t,await scene.innerHTML()]);
    }
    const refs=await page.locator('#parallel-source a').evaluateAll(es=>es.map(e=>e.href));
    assert.ok(refs.length>0&&refs.every(u=>u.includes(M.revision)),mode+': pinned source refs');
   }
   await frame(mode+'-end');
   for(const [t,html] of snapshots.reverse()){await seek(t);assert.equal(await scene.innerHTML(),html,mode+': reversible state');}
   const middle={tp:'reduce',pp:'proxy',dp:'route1',dpa:'gather',ep:'dispatch',sp:'slice',pcp:'kv',dcp:'lse'}[mode];
   const phase=tl.phases.find(p=>p.id===middle);await seek(phase.start+phase.duration*.4);const before=await scene.innerHTML();await frame(mode+'-moving');
   await seek(phase.start+phase.duration*.6);assert.notEqual(await scene.innerHTML(),before,mode+': continuous data movement');
   await page.locator('#parallel-reset').click();await toggle.click();
   for(const phase of tl.phases)await page.clock.runFor(Math.ceil(phase.duration*1000));await page.clock.runFor(80);
   assert.equal(await time(),Number(tl.duration.toFixed(3)),mode+': complete playback');assert.equal(await toggle.getAttribute('aria-pressed'),'false');
   assert.equal(await page.locator('#parallel-next').isDisabled(),true);
  }
  // Semantic checkpoints: future results must disappear when seeking back.
  await choose('pp');await seek(6);assert.equal(await scene.locator('[data-microbatch="MB1"][data-stage="1"]').getAttribute('data-state'),'等待激活可用');
  await seek(8);assert.equal(await scene.locator('[data-actor="gpu0"]').getAttribute('data-active'),'gpu02');assert.equal(await scene.locator('[data-actor="gpu1"]').getAttribute('data-active'),'gpu11');await frame('pp-cross-stage');
  await seek(10.8);assert.equal(await scene.locator('[data-actor="cpu0"]').getAttribute('data-active'),'process01');assert.equal(await scene.locator('[data-actor="gpu0"]').getAttribute('data-active'),'gpu02');await frame('pp-cpu-gpu');if(retained){await page.locator('#parallel-title').evaluate(e=>e.scrollIntoView({block:'start',behavior:'instant'}));await page.screenshot({path:path.join(artifacts,'pp-full-viewport.png')});}
  await seek(14.6);assert.equal(await scene.locator('[data-actor="cpu1"]').getAttribute('data-active'),'waitRelay1');assert.equal(await scene.locator('[data-actor="gpu1"]').getAttribute('data-active'),'gpu12');await frame('pp-last-stage-wait');
  await seek(3);assert.notEqual(await scene.locator('[data-microbatch="MB1"][data-stage="1"]').getAttribute('data-state'),'正在计算');
  await choose('dp');await seek(6);assert.equal(await scene.locator('[data-request="R1"]').getAttribute('data-state'),'GPU forward');assert.equal(await scene.locator('[data-activity="route1"]').getAttribute('data-active'),'true');await frame('dp-dispatch-compute');
  for(const mode of M.modes){await choose(mode);for(const [id,e]of Object.entries(M.build(mode).execution.events)){await seek((e.start+e.end)/2);assert.equal(await scene.locator(`[data-activity="${id}"]`).getAttribute('data-active'),'true');assert.equal(Number(await scene.locator('.execution-lanes').getAttribute('data-clock')),await time());}}
  await choose('dcp');await seek(M.build('dcp').duration);assert.match(await sceneText(),/5\.2/);await seek(5);assert.doesNotMatch(await sceneText(),/0\.8 × 6/);
  await choose('dp');await page.selectOption('#parallel-option','skip');await seek(M.build('dp','skip').duration);
  assert.match(await sceneText(),/status=False/);assert.equal(await scene.locator('[data-chip="R1"],[data-chip="R2"],[data-chip="R3"]').count(),3);
  for(const [mode,option,forbidden] of [['sp','decode','Column g → GEMM'],['pcp','short','run_attention · prev']]){
   await choose(mode);await page.selectOption('#parallel-option',option);await seek(M.build(mode,option).duration);
   assert.doesNotMatch(await sceneText(),new RegExp(forbidden));assert.equal(await scene.locator('[data-motion],[data-row]').count(),0);
   assert.match(await sceneText(),/False/);await page.selectOption('#parallel-option','default');assert.equal(await time(),0);
  }
  await choose('tp');await seek(9);await toggle.click();await page.clock.runFor(400);await toggle.click();
  const paused=await time(),frozen=await scene.innerHTML();await page.clock.runFor(1300);assert.equal(await time(),paused);assert.equal(await scene.innerHTML(),frozen);
  await page.selectOption('#parallel-rate','2');await toggle.click();await page.clock.runFor(600);await toggle.click();
  assert.ok(await time()-paused>1&&await time()-paused<1.25,'2x clock');
  await toggle.click();await choose('pp');await page.clock.runFor(800);assert.equal(await time(),0,'mode switch stops old clock');
  await page.locator('#parallel-next').focus();await page.keyboard.press('Enter');assert.ok(await time()>0,'keyboard step');
  // Reduced motion holds geometry within a phase; no in-flight cards or future results.
  await page.emulateMedia({reducedMotion:'reduce'});await seek(8);const reduced=await scene.locator('#parallel-drawing').innerHTML();await seek(9);
  assert.equal(await scene.locator('#parallel-drawing').innerHTML(),reduced);assert.equal(await scene.locator('[data-motion]').count(),0);assert.equal(await scene.locator('[data-microbatch=MB1][data-stage="1"]').getAttribute('data-state'),'正在计算');
  await seek(11.3);assert.equal(await scene.locator('[data-microbatch=MB1][data-stage="0"]').getAttribute('data-state'),'结果已处理');
  await page.emulateMedia({reducedMotion:'no-preference'});
  for(const width of [1440,1280,1121,942,820,390,320])for(const theme of ['light','dark']){
   await page.setViewportSize({width,height:1000});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
   await choose('pp');await seek(10.8);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${width}/${theme}: document overflow`);
   await bounds(`${width}/${theme}`);
   if(width>=641)assert.equal(await page.locator('.parallel-canvas-scroll').evaluate(e=>e.scrollWidth>e.clientWidth),false,`${width}: complete desktop canvas`);
   if(retained&&theme==='light'&&[942,390].includes(width)){
    await page.locator('#parallel-title').evaluate(e=>e.scrollIntoView({block:'start',behavior:'instant'}));await page.screenshot({path:path.join(artifacts,`page-${width}.png`)});
   }
  }
  await page.goto(base+'sglang/parallelism/?mode=sp&case=decode');assert.equal(await scene.getAttribute('data-mode'),'sp');assert.equal(await page.locator('#parallel-option').inputValue(),'decode');
  const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto(base+'sglang/parallelism/');
  assert.equal(await plain.locator('[data-parallel-lab]').isVisible(),false);assert.equal(await plain.locator('.parallel-map table').isVisible(),true);await plain.close();
  assert.deepEqual(errors,[]);await browser.close();
  console.log('PASS: same-clock execution lanes, causal CPU/GPU states, eight source-pinned implementations, data movement, gating, reversible state, uninterrupted playback/pause/rate, keyboard/no-JS/reduced-motion, seven widths and two themes.');
 }finally{
  if(server)await server.close();
  if(metadata){
   metadata.browserExited=server.process().exitCode!==null||server.process().signalCode!==null;metadata.profileRemoved=!fs.existsSync(metadata.profile);
   metadata.remaining=execFileSync('ps',['-axo','pid=,command='],{encoding:'utf8'}).split('\n').filter(s=>metadata.profile&&s.includes(metadata.profile));
   fs.writeFileSync(path.join(artifacts,'browser.json'),JSON.stringify(metadata,null,2));
   assert.equal(metadata.browserExited,true);assert.equal(metadata.profileRemoved,true);assert.deepEqual(metadata.remaining,[]);
  }
  if(!retained)fs.rmSync(artifacts,{recursive:true,force:true});
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
