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
       assert.equal(await page.locator('.runtime-event h3').innerText(),frames[i].title);
       assert.equal(await page.locator('.runtime-evidence a').getAttribute('href'),M.sourceURL(frames[i].source));
       assert.equal(await page.locator('.runtime-ledger').count(),4);
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
   assert.match(await page.locator('.runtime-ledger').nth(3).innerText(),new RegExp(`示例中已回传：${outputs}`));
  }
  await page.selectOption('#outputs','3');await page.locator('#reset').click();
  await page.locator('#step').focus();await page.keyboard.press('ArrowRight');
  assert.equal(await page.locator('#runtime-snapshot').getAttribute('data-frame'),'tokenize');
  await page.locator('#previous').click();assert.equal(await page.locator('#previous').isDisabled(),true);
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
  assert.equal(await plain.locator('.runtime-map').isVisible(),true);
  assert.match(await plain.locator('noscript').innerText(),/上方进程图/);
  assert.equal(await plain.locator('.runtime-source-grid a').count(),6);
  assert.equal(await plain.locator('[data-runtime-lab]').isVisible(),false);
  await plain.close();assert.deepEqual(errors,[]);
  console.log('PASS: runtime scenarios, states/source links, output parameters, keyboard, playback/reset, 4 widths × 2 themes, no-JS reading.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
