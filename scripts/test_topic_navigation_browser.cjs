const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const courses=[
 ['sglang/scheduling/index.html','scheduling','instance'],
 ['sglang/kv-memory/index.html','kv-memory','instance'],
 ['sglang/model-execution/index.html','model-execution','instance'],
 ['sglang/performance-engineering/index.html','performance','practice'],
 ['sglang/serving-operations/index.html','serving-operations','serving'],
 ['sglang/advanced-generation/index.html','advanced-generation','serving'],
 ['sglang/communication/index.html','communication','distributed'],
 ['sglang/parallelism/index.html','parallelism','distributed'],
 ['sglang/request-runtime/index.html','request-runtime','foundations'],
 ['sglang/inference-overview/journey.html','system-overview','foundations'],
 ['sglang/inference-overview/transformer.html','model-execution','instance'],
 ['sglang/inference-overview/kv-cache.html','kv-memory','instance'],
 ['sglang/inference-overview/scheduling.html','scheduling','instance'],
 ['sglang/inference-overview/deployment.html','disaggregation','distributed'],
 ['sglang/pd-prefill-lifecycle/index.html','disaggregation','distributed'],
 ['sglang/pd-dataflow/index.html','disaggregation','distributed'],
 ['sglang/pd-prefill-pp-loop/quick.html','case-studies','practice'],
 ['sglang/pd-prefill-pp-loop/index.html','case-studies','practice'],
 ['sglang/pd-prefill-pp-loop/notes.html','case-studies','practice'],
];
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
  await page.goto(base);
  const signature=()=>page.locator('.site-sidebar [data-module]').evaluateAll(nodes=>nodes.map(n=>n.dataset.module));
  const expectedModules=await signature();assert.equal(expectedModules.length,12);
  assert.equal(await page.locator('.framework-stage').count(),5);
  assert.equal(await page.locator('.module-card').count(),12);
  assert.equal(await page.locator('.module-card[data-coverage=planned]').count(),0);
  assert.equal(await page.locator('a[href="#"],a:not([href]),button[disabled]').count(),0);
  assert.match(await page.locator('#communication').innerText(),/已有课程[\s\S]*开始学习/);
  assert.match(await page.locator('#performance').innerText(),/已有课程[\s\S]*开始学习/);
  assert.match(await page.locator('#serving-operations').innerText(),/已有课程[\s\S]*开始学习/);
  assert.deepEqual(await page.locator('a[href^="#"]').evaluateAll(links=>links.filter(a=>!document.getElementById(a.hash.slice(1))).map(a=>a.hash)),[]);
  for(const group of await page.locator('.topic-group').all()){
   assert.equal(await group.evaluate(el=>el.open),false);
   const summary=group.locator(':scope > summary');await summary.click();
   assert.equal(await group.locator('nav').isVisible(),true);
   await summary.focus();await page.keyboard.press('Enter');assert.equal(await group.locator('nav').isVisible(),false);
   await page.keyboard.press('Space');assert.equal(await group.locator('nav').isVisible(),true);
  }
  await page.locator('.site-sidebar [data-module=communication] > summary').click();
  await page.locator('.site-sidebar [data-module=communication] .module-overview-link').click();
  await page.waitForURL(url=>url.hash==='#communication');assert.equal(await page.locator('#communication:target').count(),1);
  const lessons=page.locator('#system-overview .module-lessons');
  await lessons.locator('summary').focus();await page.keyboard.press('Enter');assert.equal(await lessons.locator('ol').isVisible(),true);
  await page.keyboard.press('Space');assert.equal(await lessons.locator('ol').isVisible(),false);
  for(const width of [1440,1024,900,768,390,320]){
   await page.setViewportSize({width,height:1000});
   for(const theme of ['light','dark']){
    await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`home ${width} ${theme}`);
   }
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('.theme-toggle').click();assert.equal(await page.locator('html').getAttribute('data-theme'),'light');
  await page.locator('.theme-toggle').click();assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
  await page.getByRole('link',{name:/从推理全景开始/}).click();await page.waitForLoadState('load');
  assert.match(page.url(),/journey.html/);
  assert.deepEqual(await signature(),expectedModules,'entering a lesson retains the entire framework');
  await page.locator('[data-module=kv-memory] > summary').click();
  await page.locator('[data-module=kv-memory] .lesson-link').first().click();await page.waitForLoadState('load');
  assert.match(page.url(),/kv-cache.html/);assert.match(await page.locator('.learning-breadcrumb').innerText(),/04 KV Cache/);
  await page.locator('[data-topic=framework-practice]>summary').click();
  const wasOpen=await page.locator('[data-topic=framework-practice]').evaluate(e=>e.open);
  await page.reload();assert.equal(await page.locator('[data-topic=framework-practice]').evaluate(e=>e.open),wasOpen);
  // The framework pager follows the module sequence, not the old five-page topic order.
  await page.locator('.lesson-pager .next').click();await page.waitForLoadState('load');
  assert.match(page.url(),/kv-memory\/index.html/);assert.match(await page.locator('.learning-breadcrumb').innerText(),/04 KV Cache/);
  await page.locator('.lesson-pager .next').click();await page.waitForLoadState('load');
  assert.match(page.url(),/scheduling.html/);assert.match(await page.locator('.learning-breadcrumb').innerText(),/05 调度/);
  for(const [route,module,stage] of courses){
   await page.goto(base+route);await page.waitForLoadState('load');
   assert.equal(await page.locator('.topic-group').count(),5,route);
   assert.deepEqual(await signature(),expectedModules,route);
   assert.equal(await page.locator('.site-sidebar .lesson-link').count(),courses.length,route);
   assert.equal(await page.locator('.site-sidebar [aria-current=page]').count(),1,route);
   assert.equal(await page.locator(`[data-module=${module}] .lesson-link[aria-current=page]`).count(),1,route);
   assert.equal(await page.locator(`[data-module=${module}]`).evaluate(e=>e.open),true,route);
   assert.equal(await page.locator(`[data-topic=framework-${stage}]`).evaluate(e=>e.open),true,route);
   assert.equal(await page.locator('.learning-breadcrumb').count(),1,route);
   assert.equal(await page.locator('.lesson-kicker').innerText().then(t=>/专题\s*0|\d\d \/ 0[45]/.test(t)),false,route);
   assert.equal(await page.locator('[data-topic=overview],[data-topic=lifecycle],[data-topic=pp-loop]').count(),0,route);
   for(const width of [1440,390,320]){
    await page.setViewportSize({width,height:900});
    // MediaQueryList change is delivered asynchronously after viewport resize.
    await page.waitForFunction(()=>document.querySelector('.sidebar-menu').open===!matchMedia('(max-width:900px)').matches, null, {timeout:5000});
    if(width<901){
     assert.equal(await page.locator('.sidebar-menu').evaluate(e=>e.open),false,`${route} ${width}`);
     await page.locator('.sidebar-menu>summary').click();
    }
    assert.equal(await page.locator(`[data-module=${module}] .lesson-link[aria-current=page]`).isVisible(),true);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${route} ${width}`);
    if(process.env.NAV_SCREENSHOT_DIR&&['system-overview','kv-memory'].includes(module)&&width!==320){
     fs.mkdirSync(process.env.NAV_SCREENSHOT_DIR,{recursive:true});
     await page.screenshot({path:path.join(process.env.NAV_SCREENSHOT_DIR,`${module}-${width}.png`)});
    }
    if(width<901)await page.locator('.sidebar-menu>summary').click();
   }
   await page.setViewportSize({width:1440,height:1000});
   // Every breadcrumb is a real module/stage anchor on the homepage.
   const moduleLink=page.locator('.learning-breadcrumb a').last();
   await moduleLink.click();await page.waitForLoadState('load');
   assert.equal(await page.locator(`#${module}:target`).count(),1);
   assert.deepEqual(await signature(),expectedModules);
  }
  await page.goto(base+'sglang/inference-overview/index.html?v=old');await page.waitForURL('**/journey.html*');
  assert.equal(await page.locator('.topic-group').count(),5,'legacy URL also enters the new shell');
  const staticPage=await browser.newPage({javaScriptEnabled:false,viewport:{width:1440,height:1000}});
  await staticPage.goto(base+'sglang/inference-overview/kv-cache.html');
  assert.equal(await staticPage.locator('[data-module=kv-memory] .lesson-link[aria-current=page]').isVisible(),true,'navigation is in HTML, not swapped after load');
  await staticPage.close();
  assert.deepEqual(errors,[]);console.log('PASS: one framework on home and all lessons; canonical module/current state, nested keyboard navigation, cross-module pager, breadcrumbs, placeholder anchors, desktop/mobile and no-JS navigation.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
