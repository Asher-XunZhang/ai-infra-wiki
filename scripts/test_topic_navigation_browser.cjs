const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
  await page.goto(base);assert.equal(await page.locator('.topic-group').count(),5);
  assert.equal(await page.locator('.framework-stage').count(),5);
  assert.equal(await page.locator('.module-card').count(),12);
  assert.equal(await page.locator('.module-card[data-coverage=planned]').count(),4);
  const missingAnchors=await page.locator('a[href^="#"]').evaluateAll(links=>links.filter(a=>!document.getElementById(a.hash.slice(1))).map(a=>a.hash));
  assert.deepEqual(missingAnchors,[],'all roadmap and route anchors resolve');
  assert.equal(await page.locator('a[href="#"],a:not([href]),button[disabled]').count(),0,'no empty course links or fake entry buttons');
  // A reserved module may refer to an existing case, but must not claim it is its own course.
  assert.match(await page.locator('#communication').innerText(),/预留模块[\s\S]*待建设[\s\S]*关联案例/);
  assert.equal(await page.locator('#request-runtime a,#advanced-generation a,#serving-operations a').count(),0);
  const coursePaths=await page.locator('main a[href*="sglang/"]').evaluateAll(links=>[...new Set(links.map(a=>new URL(a.href).pathname))]);
  assert.equal(coursePaths.length,10,'all ten existing content pages remain reachable from the home framework');
  for(const pathname of coursePaths){const response=await page.request.get(new URL(pathname,base).href);assert.equal(response.status(),200,pathname);}
  assert.equal(await page.getByText('专题导览',{exact:true}).count(),0);
  for(const group of await page.locator('.topic-group').all()){
   const summary=group.locator(':scope > summary');
   assert.equal(await group.evaluate(el=>el.open),false);
   await summary.click();assert.equal(await group.locator('nav').isVisible(),true);
   await summary.focus();await page.keyboard.press('Enter');assert.equal(await group.locator('nav').isVisible(),false);
   await page.keyboard.press('Space');assert.equal(await group.locator('nav').isVisible(),true);
  }
  await page.locator('[data-topic=framework-distributed] a[href="#communication"]').click();
  await page.waitForURL('**/#communication');
  assert.equal(await page.locator('#communication:target').count(),1,'sidebar selects an actual module');
  const lessons=page.locator('#system-overview .module-lessons');
  await lessons.locator('summary').focus();await page.keyboard.press('Enter');
  assert.equal(await lessons.locator('ol').isVisible(),true,'lesson list opens with keyboard');
  await page.keyboard.press('Space');assert.equal(await lessons.locator('ol').isVisible(),false);
  for(const width of [1440,1024,900,768,390,320]){
   await page.setViewportSize({width,height:1000});
   for(const theme of ['light','dark']){
    await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`home ${width} ${theme} overflow`);
    const clipped=await page.locator('.module-card').evaluateAll(cards=>cards.filter(c=>c.scrollWidth>c.clientWidth+1).map(c=>c.id));
    assert.deepEqual(clipped,[],`module content fits at ${width} ${theme}`);
    if(process.env.HOME_SCREENSHOT_DIR&&[1440,390].includes(width)){
     fs.mkdirSync(process.env.HOME_SCREENSHOT_DIR,{recursive:true});
     await page.screenshot({path:path.join(process.env.HOME_SCREENSHOT_DIR,`home-${width}-${theme}.png`),fullPage:true});
    }
   }
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('.theme-toggle').click();assert.equal(await page.locator('html').getAttribute('data-theme'),'light');
  await page.locator('.theme-toggle').click();assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
  await page.getByRole('link',{name:/从推理全景开始/}).click();
  // Cross-page clicks may resolve before deferred navigation state initializes.
  await page.waitForLoadState('load');
  assert.match(page.url(),/journey.html/);assert.equal(await page.locator('[data-topic=overview]').evaluate(el=>el.open),true);
  assert.equal(await page.locator('[data-topic=pp-loop]').evaluate(el=>el.open),false,'home framework and lesson topics have separate expansion state');
  await page.locator('[data-topic=pp-loop]>summary').click();
  await page.reload();assert.equal(await page.locator('[data-topic=pp-loop]').evaluate(el=>el.open),true,'lesson topic expansion persists');
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
  assert.deepEqual(errors,[]);console.log('PASS: five-stage home, twelve modules, honest placeholders, all ten lessons reachable, anchors, keyboard, themes, six responsive widths; lesson navigation persists and old entry redirects');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
