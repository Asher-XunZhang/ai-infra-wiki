'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const M=require('../../pages/sglang/request-state/model.js');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
const shots=process.env.REQUEST_STATE_SCREENSHOT_DIR;
(async()=>{
 const server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 const proc=server.process(),profile=proc.spawnargs.find(a=>a.startsWith('--user-data-dir=')).slice('--user-data-dir='.length);
 const identity={session:'request-state-review',controller:process.pid,browser:proc.pid,profile};console.log(JSON.stringify(identity));
 let browser;
 try{
  browser=await chromium.connect(server.wsEndpoint());const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)errors.push(r.status()+' '+r.url());});
  const root=page.locator('[data-request-state]');
  const seek=async t=>{await page.locator('[data-role=clock]').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));},t);};
  const wireStyle=e=>{const s=getComputedStyle(e);return [s.strokeWidth,s.strokeDasharray,s.strokeLinecap,s.strokeLinejoin];};
  await page.goto(base+'engine-comparison/index.html#process');await page.locator('.ia-edge.is-active').first().waitFor({state:'attached'});
  const referenceStyle=await page.locator('.ia-edge.is-active').first().evaluate(wireStyle);
  if(shots){fs.mkdirSync(shots,{recursive:true});await page.locator('.ia-boardStage').first().screenshot({path:path.join(shots,'chapter-two-reference.png')});}
  await page.goto(base+'sglang/request-state/#req-state-title');await page.waitForFunction(()=>document.querySelector('[data-request-state]').dataset.time==='0.000');
  const settleAnchor=async()=>{await page.locator('#req-state-title').evaluate(async e=>{await new Promise(requestAnimationFrame);await new Promise(requestAnimationFrame);document.activeElement?.blur();const h=document.querySelector('.site-header').getBoundingClientRect().bottom;window.scrollTo({top:scrollY+e.getBoundingClientRect().top-h-18,behavior:'instant'});});await page.waitForTimeout(150);const position=await page.locator('#req-state-title').evaluate(e=>({top:e.getBoundingClientRect().top,header:document.querySelector('.site-header').getBoundingClientRect().bottom,height:innerHeight}));assert(position.top>=position.header-1&&position.top<position.height/3,JSON.stringify(position));};
  if(shots){fs.mkdirSync(shots,{recursive:true});fs.writeFileSync(path.join(shots,'session.json'),JSON.stringify(identity));await settleAnchor();await page.screenshot({path:path.join(shots,'default-viewport.png')});}
  if(!process.env.REQUEST_STATE_VISUAL_ONLY)for(const mode of Object.keys(M.modes))for(const condition of mode==='cancel'?['waiting','running']:['waiting']){
   await page.goto(base+`sglang/request-state/?mode=${mode}&case=${condition}#req-state-title`);
   const tl=M.timeline(mode,condition);await page.waitForFunction(m=>document.querySelector('[data-request-state]').dataset.mode===m,mode);
   assert.equal(await root.getAttribute('data-time'),'0.000');assert.equal(await root.getAttribute('data-playing'),'false');
   for(const event of tl.events){
    await seek(event.end);const s=M.snapshot(tl,event.end).state;
    assert.equal(Number(await root.getAttribute('data-time')),Number(event.end.toFixed(3)));
    assert.equal(await page.locator('[data-role=tracks]').getAttribute('data-clock'),await root.getAttribute('data-time'));
    assert.match(await page.locator('[data-role=accessible-state]').textContent(),new RegExp('finished_reason='+String(s.finished||'None')));
    assert.match(await page.locator('[data-role=source]').getAttribute('href'),new RegExp(M.revision));
   }
   await seek(0);await page.locator('[data-role=speed]').selectOption('2');await page.locator('[data-role=play]').click();
   await page.waitForFunction(()=>Number(document.querySelector('[data-request-state]').dataset.time)>.3);
   await page.locator('[data-role=play]').click();const paused=await root.getAttribute('data-time');await page.waitForTimeout(180);assert.equal(await root.getAttribute('data-time'),paused);
   await page.locator('[data-role=play]').click();
   await page.waitForFunction(()=>document.querySelector('[data-request-state]').dataset.playing==='false',{},{timeout:Math.ceil(tl.duration*500)+6000});
   assert.equal(Number(await root.getAttribute('data-time')),Number(tl.duration.toFixed(3)),'one play reaches end');
   await seek(0);assert.match(await page.locator('[data-role=accessible-state]').textContent(),/finished_reason=None/);
   console.log('PASS browser trace',mode,condition);
  }
  await page.goto(base+'sglang/request-state/?mode=overlap#req-state-title');await page.waitForFunction(()=>window.RequestState);
  const consumer=page.locator('[data-region=consumer]');
  const queued=()=>page.locator('[data-queued-result]').evaluateAll(es=>es.map(e=>e.dataset.queuedResult));
  await seek(7.39);assert.equal(await consumer.getAttribute('data-result-id'),'');assert.deepEqual(await queued(),['Q0','Q1']);
  await seek(7.4);assert.equal(await consumer.getAttribute('data-result-id'),'Q0');assert.deepEqual(await queued(),['Q1']);
  const pulse=()=>page.locator('[data-flow-kind=commit] .flow-pulse').getAttribute('stroke-dashoffset');
  const completePath=()=>page.locator('[data-flow-kind=commit] .flow-path').getAttribute('d');
  await seek(10.39);assert.equal(await page.locator('[data-flow-kind=commit]').count(),0);
  await seek(10.41);const route=await completePath();assert.equal(await page.locator('[data-flow-kind=commit] .flow-path').getAttribute('marker-end'),'url(#rq-ref)','arrow already complete at activity start');
  await seek(10.9);assert.deepEqual(await page.locator('[data-flow-kind=commit] .flow-pulse').evaluate(wireStyle),referenceStyle,'same active-wire appearance as chapter 2');const a=await pulse();await seek(11.3);const b=await pulse();assert.notEqual(a,b,'pulse travels throughout the active interval');assert.equal(await completePath(),route,'full route does not grow');
  assert.equal(await consumer.getAttribute('data-result-id'),'Q0');assert.match(await consumer.textContent(),/t1/);assert.deepEqual(await queued(),['Q1']);
  assert.equal(await page.locator('[data-flow-kind=commit]').getAttribute('data-flow-source'),'consumer-result');
  assert.equal(await page.locator('[data-flow-kind=commit]').getAttribute('data-flow-target'),'output-ids');
  assert(await page.locator('[data-reference-from]').count()>0);
  assert((await page.locator('[data-reference-from] .reference-label').allTextContents()).every(t=>t==='引用'));
  const ports=await page.locator('[data-role=diagram]').evaluate(svg=>{
   const source=svg.querySelector('[data-consumer-result]').getBBox(),target=svg.querySelector('[data-output-field]').getBBox(),path=svg.querySelector('[data-flow-kind=commit] path');
   const a=path.getPointAtLength(0),b=path.getPointAtLength(path.getTotalLength());
   return {start:[a.x,a.y],end:[b.x,b.y],source:[source.x+source.width,source.y+source.height/2],target:[target.x,target.y+target.height/2]};
  });assert.deepEqual(ports.start,ports.source);assert.deepEqual(ports.end,ports.target);
  if(shots){await settleAnchor();await page.screenshot({path:path.join(shots,'q0-commit-viewport.png')});await page.locator('[data-role=diagram]').screenshot({path:path.join(shots,'q0-commit-diagram.png')});}
  await seek(11.69);assert.equal(await completePath(),route);await seek(11.7);assert.equal(await page.locator('[data-flow-kind=commit]').count(),0,'pulse ends with the operation');
  await seek(9.2);assert.equal(await page.locator('[data-flow-kind=copy]').count(),1,'device to CPU output is active before commit');
  await seek(17.1);assert.equal(await page.locator('[data-flow-kind]').count(),2,'independent copy and commit can pulse concurrently');
  assert.equal(await consumer.getAttribute('data-result-id'),'Q1');assert.match(await consumer.textContent(),/EOS/);assert.doesNotMatch(await consumer.textContent(),/额外结果/);assert.deepEqual(await queued(),['Q2']);
  assert.equal(await page.locator('[data-flow-kind=copy]').getAttribute('data-flow-target'),'Q2','new GPU result goes to Q2 while CPU commits Q1');
  await seek(20.7);assert.equal(await consumer.getAttribute('data-result-id'),'Q2');assert.deepEqual(await queued(),[]);assert.match(await consumer.textContent(),/额外结果/);assert.equal(await page.locator('[data-flow-kind=commit]').count(),0,'finished request does not commit Q2');
  await seek(10.9);const pausedPulse=await pulse();await page.waitForTimeout(180);assert.equal(await pulse(),pausedPulse,'pause freezes pulse phase');
  await page.locator('[data-role=play]').click();await page.waitForFunction(()=>Number(document.querySelector('[data-request-state]').dataset.time)>11.05);await page.locator('[data-role=play]').click();assert.notEqual(await pulse(),pausedPulse,'play resumes pulse with the shared clock');
  await seek(11.3);
  const first=await page.locator('[data-role=diagram]').innerHTML();await seek(18);await seek(11.3);assert.equal(await page.locator('[data-role=diagram]').innerHTML(),first,'reverse seek restores geometry and state');
  await page.locator('[data-event=take0]').focus();await page.keyboard.press('Enter');assert.match(await page.locator('[data-role=dependency]').textContent(),/取出 Q0/);
  if(shots){await settleAnchor();await page.screenshot({path:path.join(shots,'overlap-mid.png')});await page.setViewportSize({width:1440,height:1450});await seek(17.1);await settleAnchor();await page.screenshot({path:path.join(shots,'concurrent-pulses.png')});await page.setViewportSize({width:1440,height:1000});await seek(8);await settleAnchor();await page.screenshot({path:path.join(shots,'overlap-wait.png')});}
  await page.locator('[data-mode=chunk]').click();assert.equal(await root.getAttribute('data-time'),'0.000');await page.goBack();assert.equal(await root.getAttribute('data-mode'),'overlap');assert.equal(await root.getAttribute('data-playing'),'false');
  for(const width of [1440,1024,768,390,320])for(const theme of ['light','dark']){
   await page.setViewportSize({width,height:900});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
   await seek(10.9);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${width}/${theme}: body overflow`);
   const scroller=page.locator('.rq-scroll').first();if(await scroller.evaluate(e=>e.scrollWidth>e.clientWidth)){assert(await page.locator('.rq-pan').isVisible(),'overflow has a visible hint');await scroller.focus();await page.keyboard.press('ArrowRight');await scroller.evaluate(e=>e.scrollLeft=e.scrollWidth);assert(await scroller.evaluate(e=>e.scrollLeft>0),'resource panel reachable by scrolling');await scroller.evaluate(e=>e.scrollLeft=0);}
   const bounds=await page.locator('[data-role=diagram] text').evaluateAll(es=>es.filter(e=>{const b=e.getBBox(),v=e.ownerSVGElement.viewBox.baseVal;return b.x<0||b.x+b.width>v.width||b.y<0||b.y+b.height>v.height;}).map(e=>e.textContent));assert.deepEqual(bounds,[]);
   if(shots&&width===390){await settleAnchor();await page.screenshot({path:path.join(shots,`mobile-${theme}.png`)});}
  }
  await page.emulateMedia({reducedMotion:'reduce'});await seek(10.9);assert.equal(Number(await root.getAttribute('data-time')),10.9);const still=await pulse();await seek(11.3);assert.equal(await pulse(),still,'reduced motion preserves a static full direction');
  const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto(base+'sglang/request-state/');assert.equal(await plain.locator('#four-ledgers').count(),1);assert.equal(await plain.locator('#queue-atlas tbody tr').count(),7);await plain.close();
  await page.setViewportSize({width:1440,height:1000});await page.goto(base+'sglang/loop-journey/');await page.locator('.lesson-pager .next').click();assert.match(page.url(),/request-state/);await page.locator('.lesson-pager .next').click();assert.match(page.url(),/inference-overview\/journey/);
  assert.deepEqual(errors,[]);console.log('PASS: '+(process.env.REQUEST_STATE_VISUAL_ONLY?'visual follow-up':'playback to completion and boundaries')+', reverse seek, keyboard, history, 5 widths, 2 themes, scroll access, reduced motion, no-JS and navigation.');
 }finally{
  if(browser)await browser.close();await server.close();
  let alive=false;try{process.kill(proc.pid,0);alive=true;}catch(e){if(e.code!=='ESRCH')throw e;}
  const cleanup={browserExited:!alive,profileRemoved:!fs.existsSync(profile)};console.log('CLEANUP',JSON.stringify(cleanup));if(shots&&fs.existsSync(shots))fs.writeFileSync(path.join(shots,'cleanup.json'),JSON.stringify(cleanup));assert(!alive&&cleanup.profileRemoved,'owned browser/profile cleanup');
 }
})().catch(e=>{console.error(e);process.exit(1);});
