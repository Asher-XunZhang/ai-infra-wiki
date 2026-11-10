'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright'),M=require('../../pages/sglang/performance-engineering/timeline.js');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/',shots=process.env.PERFORMANCE_SCREENSHOT_DIR,visualOnly=process.env.PERFORMANCE_VISUAL_ONLY;
(async()=>{
 const server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 const proc=server.process(),profile=proc.spawnargs.find(a=>a.startsWith('--user-data-dir=')).slice('--user-data-dir='.length);
 const identity={session:'performance-pd-review',controller:process.pid,browser:proc.pid,profile};console.log(JSON.stringify(identity));let browser;
 try{
  browser=await chromium.connect(server.wsEndpoint());const page=await browser.newPage({viewport:{width:1440,height:1080}}),errors=[],root=page.locator('[data-performance-pd]');
  page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`);});
  const go=async(mode='clock',condition='base',t=0,metric='')=>{await page.goto(base+`sglang/performance-engineering/?view=source&mode=${mode}&condition=${condition}&t=${t}&metric=${metric}#perf-title`);await page.waitForFunction(()=>window.PerformancePD&&document.querySelector('[data-performance-pd]').dataset.time!==undefined);};
  const seek=async t=>page.locator('[data-role=clock]').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));},t);
  const state=async()=>JSON.parse(await page.locator('[data-role=accessible]').textContent());
  const expected=(tl,t)=>{const s=M.snapshot(tl,t);return {time:s.time,observations:s.observations.map(m=>({id:m.id,status:m.status,value:m.status==='已采集'?m.value:null})),tmOut:s.tmOut,clientOut:s.clientOut,tmItl:s.tmItl,clientItl:s.clientItl,tpot:s.tpot};};
  const settle=async()=>{await page.locator('#perf-title').evaluate(async e=>{await new Promise(requestAnimationFrame);document.activeElement?.blur();window.scrollTo({top:scrollY+e.getBoundingClientRect().top-document.querySelector('.site-header').getBoundingClientRect().bottom-16,behavior:'instant'});});await page.waitForTimeout(100);};
  const shot=async name=>{if(shots){await settle();await page.screenshot({path:path.join(shots,name+'-viewport.png')});await root.screenshot({path:path.join(shots,name+'-panel.png')});}};
  if(shots){fs.mkdirSync(shots,{recursive:true});fs.writeFileSync(path.join(shots,'session.json'),JSON.stringify(identity));}
  await go();assert(await page.locator('[data-role=diagram] rect').count()>30);await shot('default');
  if(!visualOnly)for(const [mode,m]of Object.entries(M.modes))for(const condition of Object.keys(m.cases)){
   if(process.env.PERFORMANCE_CASE_FILTER&&!new RegExp(process.env.PERFORMANCE_CASE_FILTER).test(mode+'/'+condition))continue;
   await go(mode,condition);const tl=M.timeline(mode,condition);
   for(const e of tl.events){await seek(e.end);assert.deepEqual(await state(),expected(tl,e.end));assert.equal(await root.getAttribute('data-time'),await page.locator('[data-role=diagram]').getAttribute('data-clock'));assert.match(await page.locator('[data-role=event-source]').getAttribute('href'),new RegExp(M.revision));}
   await seek(0);await page.locator('[data-role=speed]').selectOption('2');await page.locator('[data-role=play]').click();await page.waitForFunction(()=>Number(document.querySelector('[data-performance-pd]').dataset.time)>2);
   await page.locator('[data-role=play]').click();const paused=await root.getAttribute('data-time');await page.waitForTimeout(120);assert.equal(await root.getAttribute('data-time'),paused);
   await page.locator('[data-role=play]').click();await page.waitForFunction(()=>document.querySelector('[data-performance-pd]').dataset.playing==='false',{},{timeout:Math.ceil(tl.duration/80*1000)+5000});
   assert.equal(Number(await root.getAttribute('data-time')),tl.duration);assert.deepEqual(await state(),expected(tl,tl.duration));await seek(0);assert.deepEqual(await state(),expected(tl,0));console.log('PASS continuous playback',mode,condition);
  }
  await go('kv','chunked',310);assert.equal(await page.locator('[data-event=p-gpu2].active').count(),1);assert.equal(await page.locator('[data-event=kv1].active').count(),1);await shot('chunk-overlap');
  const full=await page.locator('[data-pulse=kv1]').getAttribute('d'),phase=await page.locator('[data-pulse=kv1]').getAttribute('stroke-dashoffset');
  await seek(315);assert.equal(await page.locator('[data-pulse=kv1]').getAttribute('d'),full);assert.notEqual(await page.locator('[data-pulse=kv1]').getAttribute('stroke-dashoffset'),phase);
  const later=await page.locator('[data-pulse=kv1]').getAttribute('stroke-dashoffset');await page.waitForTimeout(120);assert.equal(await page.locator('[data-pulse=kv1]').getAttribute('stroke-dashoffset'),later);
  const saved=await page.locator('[data-role=diagram]').innerHTML();await seek(600);await seek(315);assert.equal(await page.locator('[data-role=diagram]').innerHTML(),saved);
  await page.locator('[data-role=metric]').selectOption('kv-latency');assert.equal(Number(await root.getAttribute('data-time')),315);assert.equal(await page.locator('[data-measure=kv-latency]').count(),1);
  await seek(425);assert.match(await page.locator('[data-role=measure-value]').textContent(),/计时中/);await seek(426);assert.equal(await page.locator('[data-role=measure-value]').textContent(),'71 ms');await shot('last-chunk-metric');
  await page.locator('[data-role=condition]').selectOption('missing');assert.equal(Number(await root.getAttribute('data-time')),0);await page.locator('[data-role=metric]').selectOption('kv-latency');await seek(600);assert.equal(await page.locator('[data-role=measure-value]').textContent(),'无本次样本');
  await page.locator('[data-role=events]').selectOption('d-prebuilt');assert.equal(Number(await root.getAttribute('data-time')),420);assert.match(await page.locator('[data-role=event-title]').textContent(),/PREBUILT/);
  await page.locator('[data-event=p-result]').focus();await page.keyboard.press('Enter');assert.equal(Number(await root.getAttribute('data-time')),340);assert.match(await page.locator('[data-role=dependency]').textContent(),/Prefill GPU/);
  await page.locator('[data-role=reset]').click();await page.locator('[data-role=clock]').focus();await page.keyboard.press('ArrowRight');assert(Number(await root.getAttribute('data-time'))>0);
  await go('clock','gate',500);await shot('gate-delay');await go('stream','bundle',605);assert.deepEqual((await state()).clientItl,[40,40,40]);assert.equal(await page.locator('[data-role=measure-value]').textContent(),'40 ms');await shot('bundled-itl');await page.locator('[data-role=metric]').selectOption('client-tpot');assert.match(await page.locator('[data-role=measure-value]').textContent(),/等待采集/);await seek(630);assert.equal(await page.locator('[data-role=measure-value]').textContent(),'46.7 ms');await go('sample','failure',640);await shot('failed-sample');
  if(!visualOnly)for(const width of [1440,1100,768,390,320])for(const theme of ['light','dark']){
   await page.setViewportSize({width,height:1080});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
   for(const [mode,m]of Object.entries(M.modes))for(const condition of Object.keys(m.cases)){
   if(process.env.PERFORMANCE_CASE_FILTER&&!new RegExp(process.env.PERFORMANCE_CASE_FILTER).test(mode+'/'+condition))continue;
    await page.locator(`button[data-mode=${mode}]`).click();await page.locator('[data-role=condition]').selectOption(condition);await seek(M.timeline(mode,condition).duration/2);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${width}/${theme}/${mode}/${condition} outer overflow`);
    const overflow=await page.locator('[data-role=diagram]').evaluate(svg=>Array.from(svg.querySelectorAll('text')).filter(t=>{const b=t.getBBox();return b.x<0||b.x+b.width>1101;}).map(t=>t.textContent));assert.deepEqual(overflow,[],`${mode}/${condition} svg clipping`);
   }
   if((width===1440&&theme==='dark')||(width===390&&theme==='light'))await shot(`${width}-${theme}`);
  }
  await page.setViewportSize({width:1440,height:1080});await page.emulateMedia({reducedMotion:'reduce'});await go('kv','chunked',310);
  const reducedPhase=await page.locator('[data-pulse=kv1]').getAttribute('stroke-dashoffset');await seek(315);assert.equal(await page.locator('[data-pulse=kv1]').getAttribute('stroke-dashoffset'),reducedPhase);assert.equal(await root.getAttribute('data-playing'),'false');
  const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto(base+'sglang/performance-engineering/');assert.match(await plain.locator('noscript').textContent(),/固定源码/);assert(await plain.locator('.pm-prose').count()>0);await plain.close();assert.deepEqual(errors,[]);
  console.log('PASS: source sampling windows, continuous playback, shared clock, rewind, keyboard, full pulses, reduced motion, no-JS; '+(visualOnly?'visual samples.':(process.env.PERFORMANCE_CASE_FILTER?'filtered '+process.env.PERFORMANCE_CASE_FILTER:'12 conditions')+' × 5 widths × 2 themes.'));
 }finally{
  if(browser)await browser.close();await server.close();for(let i=0;i<20&&(proc.exitCode===null&&proc.signalCode===null);i++)await new Promise(r=>setTimeout(r,50));
  let alive=false;try{process.kill(proc.pid,0);alive=true;}catch{}assert.equal(alive,false,'owned Chromium still alive');assert.equal(fs.existsSync(profile),false,'owned profile remains');console.log('CLEANED',JSON.stringify(identity));
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
