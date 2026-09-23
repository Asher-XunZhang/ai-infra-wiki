const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const M=require('../pages/sglang/communication/model.js');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 try{
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';await page.goto(base+'sglang/communication/');await page.waitForLoadState('load');
  const choose=async mode=>page.locator(`button[data-mode=${mode}]`).click();
  const next=async()=>page.locator('#comm-next').click();
  const end=async()=>{while(!await page.locator('#comm-next').isDisabled())await next();};
  for(const width of [1440,768,390,320]){
   await page.setViewportSize({width,height:1000});
   for(const theme of ['light','dark']){
    await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
    for(const mode of ['message','collective','kv','gates']){
     await choose(mode);
     do{
      assert.equal(await page.locator('[data-settings]:visible').count(),1);
      assert.equal(await page.locator('.parallel-details').evaluate(e=>e.open),false);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${width}/${theme}/${mode}`);
      assert.match(await page.locator('#comm-source').getAttribute('href'),new RegExp(M.revision));
      if(await page.locator('#comm-next').isDisabled())break;
      await next();
     }while(true);
    }
   }
  }
  await choose('message');await next();assert.match(await page.locator('#comm-caption').innerText(),/不等于/);await next();assert.match(await page.locator('.endpoint').nth(1).innerText(),/Req R1/);
  await page.selectOption('#message-kind','tensor');await next();await next();
  assert.equal(await page.locator('.endpoint').nth(1).locator('.chip.empty').count(),2);await next();await next();
  assert.deepEqual(await page.locator('.endpoint').nth(1).locator('.chip').allTextContents(),['3','5']);
  await choose('collective');for(const kind of ['reduce','gather','exchange']){
   await page.selectOption('#collective-kind',kind);await end();
   const outputs=await page.locator('.comm-pair').nth(1).locator('.endpoint').evaluateAll(es=>es.map(e=>[...e.querySelectorAll('.chip')].map(c=>c.textContent)));
   assert.deepEqual(outputs,M.collective(kind).output.map(xs=>xs.map(String)));
  }
  await choose('kv');await end();assert.match(await page.locator('.result-note').innerText(),/2 段/);
  await page.locator('#kv-fragment').check();await end();assert.match(await page.locator('.result-note').innerText(),/3 段/);
  assert.deepEqual(await page.locator('.registered').nth(1).locator('.chip').allTextContents(),['A','C','B']);
  await page.locator('#comm-reset').click();await next();await next();await next();
  const points=await page.locator('.comm-wire svg').evaluate(svg=>{svg.pauseAnimations();const circle=svg.querySelector('circle');return [.1,.8].map(t=>{svg.setCurrentTime(t);const m=circle.getCTM();return {x:m.e,y:m.f};});});
  assert.ok(points[1].y>points[0].y,'KV travels down from source to destination');
  await page.emulateMedia({reducedMotion:'reduce'});await page.waitForFunction(()=>document.querySelectorAll('animateMotion').length===0);assert.equal(await page.locator('animateMotion').count(),0);await page.emulateMedia({reducedMotion:'no-preference'});
  await choose('gates');for(const kind of ['normal','cache','corrupt','abort','timeout']){
   await page.selectOption('#gate-kind',kind);await end();
   if(['normal','cache'].includes(kind))assert.match(await page.locator('.request-track .parking').nth(1).innerText(),/R1/);
   if(kind==='corrupt'){assert.equal(await page.locator('.gate.failed').count(),1);assert.match(await page.locator('.result-note').innerText(),/R1 终止/);assert.equal(await page.locator('.parking.occupied').count(),0);}
   if(kind==='abort')assert.match(await page.locator('.result-note').innerText(),/2 \/ 2 drain ACK/);
   if(kind==='timeout'){assert.match(await page.locator('.result-note').innerText(),/超时释放/);assert.equal(await page.locator('.ack-dot.done').count(),1);assert.match(await page.locator('.parking').innerText(),/释放/);}
  }
  await choose('message');await page.selectOption('#message-kind','ipc');await page.clock.install();await page.locator('#comm-play').click();await page.clock.runFor(2300);assert.equal(await page.locator('#comm-scene').getAttribute('data-step'),'1');
  await page.locator('#comm-play').click();await page.clock.runFor(4500);assert.equal(await page.locator('#comm-scene').getAttribute('data-step'),'1');
  await page.locator('#comm-next').focus();await page.keyboard.press('Enter');assert.equal(await page.locator('#comm-scene').getAttribute('data-step'),'2');
  await page.locator('#comm-reset').click();assert.equal(await page.locator('#comm-previous').isDisabled(),true);
  const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto(base+'sglang/communication/');assert.equal(await plain.locator('[data-comm-lab]').isVisible(),false);await plain.locator('.topology-detail summary').click();assert.equal(await plain.locator('.comm-static').isVisible(),true);await plain.close();
  assert.deepEqual(errors,[]);console.log('PASS: communication modes/branches, numerical output, page destinations, readiness/release, motion, reduced motion, keyboard, clock, four widths × two themes, no-JS.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
