const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const M=require('../pages/sglang/request-runtime/model.js');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 try{
  const page=await browser.newPage(),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
  await page.goto(base+'sglang/request-runtime/');await page.waitForLoadState('load');
  assert.equal(await page.locator('[data-runtime-lab]').isVisible(),true);
  assert.equal(await page.locator('[data-module=request-runtime] .lesson-link[aria-current=page]').count(),1);
  const step=async i=>page.locator('#step').evaluate((e,i)=>{e.value=i;e.dispatchEvent(new Event('input',{bubbles:true}));},i);
  for(const width of [1440,768,390,320]){
   await page.setViewportSize({width,height:1000});
   for(const theme of ['light','dark']){
    await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
    for(const scenario of ['normal','queued-abort','running-abort','invalid']){
     await page.selectOption('#scenario',scenario);
     for(const cache of [true,false]){
      await page.locator('#cache').setChecked(cache);
      const frames=M.build({scenario,cache});
      for(let i=0;i<frames.length;i++){
       await step(i);
       assert.equal(await page.locator('#runtime-snapshot').getAttribute('data-frame'),frames[i].id);
       assert.equal(await page.locator('.seq-event.current').getAttribute('data-step'),String(i));
       assert.equal(await page.locator('.seq-event').count(),1,'focus view shows only the current handoff');
       assert.equal(await page.locator('#kv-cells .filled').count(),frames[i].kv+frames[i].cached);
       assert.equal(await page.locator('#sample-cells .filled').count(),frames[i].sampled);
       assert.equal(await page.locator('#visible-cells .filled').count(),frames[i].visible);
       assert.equal(await page.locator('#step-source').getAttribute('href'),M.sourceURL(frames[i].source));
       assert.equal(await page.locator('.runtime-ledger').count(),0,'text cards must not replace the swimlane');
       assert.equal(await page.locator('.sequence-label').count(),6);
       assert.equal(await page.locator('.seq-event.current .seq-arrow').count()>0,true);
       assert.equal(await page.locator('.runtime-step-details').evaluate(e=>e.open),false);
       assert.equal(await page.locator('#sequence-scroll').evaluate(el=>[...el.querySelectorAll('.current .seq-stop')].every(node=>{const b=node.getBBox();return b.x>=0&&b.x+b.width<=el.clientWidth;})),true,'current status marks remain inside the focus viewport');
       assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${width}/${theme}/${scenario}/${i}`);
      }
      assert.equal(await page.locator('#next').isDisabled(),true);
     }
    }
   }
  }
  await page.selectOption('#scenario','normal');
  for(const outputs of [1,2,4,5]){
   await page.selectOption('#outputs',String(outputs));
   await step(M.build({outputs}).length-1);
   assert.equal(await page.locator('#visible-cells .filled').count(),outputs);
  }
  await page.selectOption('#outputs','3');await page.locator('#reset').click();
  await page.locator('#step').focus();await page.keyboard.press('ArrowRight');
  assert.equal(await page.locator('#runtime-snapshot').getAttribute('data-frame'),'tokenize');
  await page.locator('#show-sequence').check();
  assert.equal(await page.locator('.seq-event').count(),M.build().length,'full flow is available on demand');
  await page.locator('.seq-event[data-step="3"]').focus();await page.keyboard.press('Enter');
  assert.equal(await page.locator('#runtime-snapshot').getAttribute('data-frame'),'queued');
  await page.locator('#show-sequence').uncheck();
  await page.locator('#reset').click();await page.locator('#next').click();
  await page.locator('#previous').click();assert.equal(await page.locator('#previous').isDisabled(),true);
  // Verify the picture actually carries messages along the correct path, in hop order.
  await step(M.build().findIndex(f=>f.id==='first-output'));
  const motion=await page.evaluate(()=>{
   const svg=document.querySelector('#sequence-svg');svg.pauseAnimations();
   const packets=[...svg.querySelectorAll('.current .seq-packet')];
   const at=t=>{svg.setCurrentTime(t);return packets.map(p=>({y:p.getCTM().f,visible:getComputedStyle(p).visibility}));};
   return {early:at(.275),middle:at(.825),late:at(1.375)};
  });
  assert.ok(Math.abs(motion.early[0].y-216)<2,'first hop travels Scheduler → Detokenizer');
  assert.equal(motion.early[1].visible,'hidden','second hop waits for first hop');
  assert.ok(Math.abs(motion.middle[1].y-184)<2,'second hop returns text to frontend');
  assert.equal(motion.middle[2].visible,'hidden','client hop waits for frontend text');
  assert.ok(Math.abs(motion.late[2].y-88)<2,'final hop reaches toward client');
  await page.emulateMedia({reducedMotion:'reduce'});await step(4);
  assert.equal(await page.locator('animateMotion').count(),0,'reduced motion retains static arrows');
  assert.equal(await page.locator('.current .seq-computation').count(),1,'forward has a GPU work block');
  assert.equal(await page.locator('.current .compute-bar').first().evaluate(e=>getComputedStyle(e).animationName),'none');
  await page.emulateMedia({reducedMotion:'no-preference'});await page.locator('#reset').click();
  await page.clock.install();await page.locator('#play').click();await page.clock.runFor(2300);
  assert.equal(await page.locator('#runtime-snapshot').getAttribute('data-frame'),'tokenize');
  await page.locator('#play').click();await page.clock.runFor(4500);
  assert.equal(await page.locator('#runtime-snapshot').getAttribute('data-frame'),'tokenize','pause stops advance');
  await page.locator('#play').click();await page.selectOption('#scenario','queued-abort');await page.clock.runFor(4500);
  assert.equal(await page.locator('#runtime-snapshot').getAttribute('data-frame'),'input','scenario resets and stops');
  assert.equal(await page.locator('#play').getAttribute('aria-pressed'),'false');
  await page.selectOption('#scenario','normal');await page.locator('#cache').check();
  if(process.env.RUNTIME_SCREENSHOT_DIR){
   fs.mkdirSync(process.env.RUNTIME_SCREENSHOT_DIR,{recursive:true});
   for(const width of [1440,390])for(const theme of ['light','dark']){
    await page.setViewportSize({width,height:1100});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
    await step(M.build().findIndex(f=>f.id==='release'));
    await page.locator('[data-runtime-lab]').scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(process.env.RUNTIME_SCREENSHOT_DIR,`${width}-${theme}.png`),fullPage:true});
   }
  }
  const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto(base+'sglang/request-runtime/');
  await plain.locator('.runtime-reading-details>summary').click();
  assert.equal(await plain.locator('.runtime-map').isVisible(),true);
  assert.match(await plain.locator('noscript').innerText(),/进程图/);
  assert.equal(await plain.locator('.runtime-source-grid a').count(),6);
  assert.equal(await plain.locator('[data-runtime-lab]').isVisible(),false);
  await plain.close();assert.deepEqual(errors,[]);
  console.log('PASS: runtime scenarios, states/source links, output parameters, keyboard, playback/reset, six-lane diagram, sequential moving messages, reduced motion and live resource glyphs, 4 widths × 2 themes, no-JS reading.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
