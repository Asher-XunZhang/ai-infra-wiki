'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright'),M=require('../../pages/sglang/serving-operations/timeline.js');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/',shots=process.env.SERVING_SCREENSHOT_DIR,visualOnly=process.env.SERVING_VISUAL_ONLY;
(async()=>{
 const server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 const proc=server.process(),profile=proc.spawnargs.find(a=>a.startsWith('--user-data-dir=')).slice('--user-data-dir='.length);
 const identity={session:'serving-pd-review',controller:process.pid,browser:proc.pid,profile};console.log(JSON.stringify(identity));let browser;
 try{
  browser=await chromium.connect(server.wsEndpoint());const page=await browser.newPage({viewport:{width:1440,height:1080}}),errors=[];const root=page.locator('[data-serving-pd]');
  page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`);});
  const go=async(mode='health',condition='pair',t=0)=>{await page.goto(base+`sglang/serving-operations/?mode=${mode}&condition=${condition}&t=${t}#ops-title`);await page.waitForFunction(()=>window.ServingPD&&document.querySelector('[data-serving-pd]').dataset.time!==undefined);};
  const seek=async t=>{await page.locator('[data-role=clock]').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));},t);};
  const state=async()=>JSON.parse(await page.locator('[data-role=accessible]').textContent());
  const settle=async()=>{await page.locator('#ops-title').evaluate(async e=>{await new Promise(requestAnimationFrame);document.activeElement?.blur();window.scrollTo({top:scrollY+e.getBoundingClientRect().top-document.querySelector('.site-header').getBoundingClientRect().bottom-16,behavior:'instant'});});await page.waitForTimeout(180);};
  const shot=async name=>{if(shots){await settle();await page.screenshot({path:path.join(shots,name+'-viewport.png')});await root.screenshot({path:path.join(shots,name+'-panel.png')});}};
  if(shots){fs.mkdirSync(shots,{recursive:true});fs.writeFileSync(path.join(shots,'session.json'),JSON.stringify(identity));}
  await go();assert.equal(await page.locator('[data-runtime]').count(),2);assert(await page.locator('[data-role=diagram] rect').count()>4);await shot('default');
  if(!visualOnly&&!process.env.SERVING_SKIP_PLAYBACK)for(const [mode,m] of Object.entries(M.modes))for(const condition of Object.keys(m.cases)){
   await go(mode,condition);const tl=M.timeline(mode,condition);
   for(const e of tl.events){await seek(e.end);assert.deepEqual(await state(),M.snapshot(tl,e.end).state);assert.equal(await root.getAttribute('data-time'),await page.locator('[data-role=tracks]').getAttribute('data-clock'));assert.match(await page.locator('[data-role=source]').getAttribute('href'),new RegExp(M.revision));}
   await seek(0);await page.locator('[data-role=speed]').selectOption('2');await page.locator('[data-role=play]').click();await page.waitForFunction(()=>Number(document.querySelector('[data-serving-pd]').dataset.time)>.2);
   await page.locator('[data-role=play]').click();const paused=await root.getAttribute('data-time');await page.waitForTimeout(120);assert.equal(await root.getAttribute('data-time'),paused);
   await page.locator('[data-role=play]').click();await page.waitForFunction(()=>document.querySelector('[data-serving-pd]').dataset.playing==='false',{},{timeout:Math.ceil(tl.duration*500)+5000});
   assert.equal(Number(await root.getAttribute('data-time')),Number(tl.duration.toFixed(3)));await seek(0);assert.deepEqual(await state(),tl.initial);console.log('PASS continuous playback',mode,condition);
  }
  await go('dispatch','success',2.6);assert.equal(await page.locator('[data-event-flow]').count(),2);
  const full=await page.locator('[data-event-flow=send-P] path').getAttribute('d'),phase=await page.locator('[data-event-flow=send-P] path').getAttribute('stroke-dashoffset');
  await seek(2.9);assert.equal(await page.locator('[data-event-flow=send-P] path').getAttribute('d'),full);assert.notEqual(await page.locator('[data-event-flow=send-P] path').getAttribute('stroke-dashoffset'),phase);
  const later=await page.locator('[data-event-flow=send-P] path').getAttribute('stroke-dashoffset');await page.waitForTimeout(120);assert.equal(await page.locator('[data-event-flow=send-P] path').getAttribute('stroke-dashoffset'),later);
  const pts=await page.locator('[data-event-flow=send-P] path').evaluate(p=>{const a=p.getPointAtLength(0),b=p.getPointAtLength(p.getTotalLength());return [a.x,a.y,b.x,b.y];});assert(Math.abs(pts[0]-495.175)<.001);assert.deepEqual(pts.slice(1),[168,270,230]);
  await shot('parallel-dispatch');await seek(13.8);assert.match(await page.locator('[data-runtime=D]').textContent(),/D GPU 继续生成/);await shot('compute-and-response');
  const saved=await page.locator('[data-role=diagram]').innerHTML();await seek(17);await seek(13.8);assert.equal(await page.locator('[data-role=diagram]').innerHTML(),saved);
  await go('cancel','transfer',5.5);assert.equal((await state()).D.frontDone,true);assert.equal((await state()).D.released,false);await shot('frontend-done-kv-held');
  await page.locator('[data-event=release]').focus();await page.keyboard.press('Enter');assert.match(await page.locator('[data-role=dependency]').textContent(),/ACK/);assert.equal(Number(await root.getAttribute('data-time')),7.7);assert.match(await page.locator('[data-role=event-title]').textContent(),/归还资源/);await page.locator('[data-role=events]').selectOption('drain-writes');assert.equal(await page.locator('[data-role=event-title]').textContent(),'P 传输排空');
  await page.locator('[data-role=reset]').click();await page.locator('[data-role=clock]').focus();await page.keyboard.press('ArrowRight');assert(Number(await root.getAttribute('data-time'))>0);
  await go('cancel','stream',5.5);await shot('abort-and-gpu');await go('drain','normal',9.1);await shot('p-exited-d-running');
  if(!visualOnly)for(const width of [1440,1100,768,390,320])for(const theme of ['light','dark']){
   await page.setViewportSize({width,height:1080});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
   for(const [mode,m]of Object.entries(M.modes))for(const condition of Object.keys(m.cases)){
    await page.locator(`button[data-mode=${mode}]`).click();await page.locator('[data-role=condition]').selectOption(condition);await seek(M.timeline(mode,condition).duration/2);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${width}/${theme}/${mode}/${condition} outer overflow`);
    const overflow=await page.locator('[data-role=diagram]').evaluate(svg=>Array.from(svg.querySelectorAll('text')).filter(t=>{const b=t.getBBox();return b.x<0||b.x+b.width>1101;}).map(t=>t.textContent));assert.deepEqual(overflow,[],`${mode}/${condition} svg clipping`);
   }
   if((width===1440&&theme==='dark')||(width===390&&theme==='light'))await shot(`${width}-${theme}`);
  }
  await page.setViewportSize({width:1440,height:1080});await page.emulateMedia({reducedMotion:'reduce'});await go('dispatch','success',2.6);
  const reducedPhase=await page.locator('[data-event-flow=send-P] path').getAttribute('stroke-dashoffset');await seek(2.9);assert.equal(await page.locator('[data-event-flow=send-P] path').getAttribute('stroke-dashoffset'),reducedPhase);assert.equal(await root.getAttribute('data-playing'),'false');
  const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto(base+'sglang/serving-operations/');assert.match(await plain.locator('noscript').textContent(),/固定源码/);assert(await plain.locator('.sp-prose').count()>0);await plain.close();assert.deepEqual(errors,[]);
  console.log('PASS: P/D shared-clock rendering, full pulses, connected ports, rewind, keyboard, reduced motion, no-JS; '+(visualOnly?'visual samples.':'14 conditions × 5 widths × 2 themes.'));
 }finally{
  if(browser)await browser.close();await server.close();for(let i=0;i<20&&(proc.exitCode===null&&proc.signalCode===null);i++)await new Promise(r=>setTimeout(r,50));
  let alive=false;try{process.kill(proc.pid,0);alive=true;}catch{}assert.equal(alive,false,'owned Chromium still alive');assert.equal(fs.existsSync(profile),false,'owned profile remains');console.log('CLEANED',JSON.stringify(identity));
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
