const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
  await page.goto(base);assert.equal(await page.locator('.topic-group').count(),3);
  assert.equal(await page.getByText('专题导览',{exact:true}).count(),0);
  for(const group of await page.locator('.topic-group').all()){
   const summary=group.locator(':scope > summary');
   assert.equal(await group.evaluate(el=>el.open),false);
   await summary.click();assert.equal(await group.locator('nav').isVisible(),true);
   await summary.focus();await page.keyboard.press('Enter');assert.equal(await group.locator('nav').isVisible(),false);
   await page.keyboard.press('Space');assert.equal(await group.locator('nav').isVisible(),true);
  }
  await page.getByRole('link',{name:/从推理全景开始/}).click();
  // Cross-page clicks may resolve before deferred navigation state initializes.
  await page.waitForLoadState('load');
  assert.match(page.url(),/journey.html/);assert.equal(await page.locator('[data-topic=overview]').evaluate(el=>el.open),true);
  assert.equal(await page.locator('[data-topic=pp-loop]').evaluate(el=>el.open),true,'other topic expansion persists');
  await page.locator('[data-topic=pp-loop]>summary').click();
  await page.locator('[data-topic=overview] a[href*="kv-cache.html"]').click();
  await page.waitForLoadState('load');
  assert.equal(await page.locator('[data-topic=pp-loop]').evaluate(el=>el.open),false,'collapsed topic stays collapsed');
  assert.equal(await page.locator('[data-topic=overview] [aria-current=page]').count(),1);
  for(const width of [390,320]){
   await page.setViewportSize({width,height:844});
   await page.locator('.sidebar-menu>summary').click();
   await page.locator('[data-topic=overview]>summary').click();assert.equal(await page.locator('[data-topic=overview] nav').isVisible(),false);
   await page.locator('[data-topic=overview]>summary').click();assert.equal(await page.locator('[data-topic=overview] nav').isVisible(),true);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
   await page.locator('.sidebar-menu>summary').click();
  }
  await page.goto(base+'sglang/inference-overview/index.html?v=old');await page.waitForURL('**/journey.html*');
  assert.equal(await page.getByText('专题导览',{exact:true}).count(),0);
  await page.setViewportSize({width:1440,height:1000});
  for(const route of ['sglang/pd-prefill-lifecycle/index.html','sglang/pd-prefill-pp-loop/quick.html','sglang/pd-prefill-pp-loop/index.html','sglang/pd-prefill-pp-loop/notes.html','sglang/pd-dataflow/index.html']){
   await page.goto(base+route);
   assert.equal(await page.locator('.topic-group').count(),3,route);
   assert.equal(await page.locator('[data-topic=pp-loop] .course-link').count(),4,route);
   const active=page.locator('.topic-group:has([aria-current=page])');
   assert.equal(await active.count(),1,route);assert.equal(await active.evaluate(el=>el.open),true,route);
   assert.equal(await page.locator('[data-topic=overview] .course-link').count(),5,route);
  }
  assert.deepEqual(errors,[]);console.log('PASS: no guide entry; all topic headings toggle by click/keyboard; current topic expands; state persists; mobile; old entry redirects; existing lessons and dataflow retain complete navigation');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
