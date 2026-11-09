const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const courses=[
 ['sglang/loop-journey/index.html','loop-journey','instance'],
 ['sglang/system-overview/index.html','system-overview','foundations'],
 ['sglang/scheduling/index.html','scheduling','instance'],
 ['sglang/kv-memory/index.html','kv-memory','instance'],
 ['sglang/model-execution/index.html','model-execution','instance'],
 ['sglang/performance-engineering/index.html','performance','practice'],
 ['sglang/serving-operations/index.html','serving-operations','serving'],
 ['sglang/advanced-generation/index.html','advanced-generation','serving'],
 ['sglang/communication/index.html','communication','distributed'],
 ['sglang/parallelism/index.html','parallelism','distributed'],
 ['sglang/request-runtime/index.html','request-runtime','foundations'],
 ['sglang/inference-overview/journey.html','loop-journey','instance'],
 ['sglang/inference-overview/transformer.html','model-execution','instance'],
 ['sglang/inference-overview/kv-cache.html','kv-memory','instance'],
 ['sglang/inference-overview/scheduling.html','loop-journey','instance'],
 ['sglang/inference-overview/deployment.html','disaggregation','distributed'],
 ['sglang/pd-prefill-lifecycle/index.html','disaggregation','distributed'],
 ['sglang/pd-dataflow/index.html','disaggregation','distributed'],
 ['sglang/pp-consensus/index.html','disaggregation','distributed'],
 ['sglang/pd-prefill-pp-loop/quick.html','case-studies','practice'],
 ['sglang/pd-prefill-pp-loop/index.html','case-studies','practice'],
 ['sglang/pd-prefill-pp-loop/notes.html','case-studies','practice'],
];
// Opening another chapter must keep its courses as separate readable links.
async function checkFoundationNavigation(page,label,interactive=true){
 if(interactive)assert.equal(await page.locator('.site-header [data-site-check]').count(),1,label+': header update entry');
 assert.equal(await page.locator('.site-footer [data-site-check]').count(),0,label+': no footer duplicate');
 assert.equal(await page.locator('.site-update-notice').count(),0,label+': no body update notice');
 const chapter=page.locator('.site-sidebar [data-chapter=foundations]');
 const wasOpen=await chapter.evaluate(el=>el.open);
 if(!wasOpen)await chapter.locator(':scope > summary').click();
 const links=chapter.locator('.foundation-nav a');
 assert.equal(await links.count(),13,label);
 if(process.env.NAV_SCREENSHOT_DIR&&label==='engine-process-1440'){
  fs.mkdirSync(process.env.NAV_SCREENSHOT_DIR,{recursive:true});
  await chapter.locator(':scope > summary').scrollIntoViewIfNeeded();
  await page.locator('.site-sidebar').screenshot({path:path.join(process.env.NAV_SCREENSHOT_DIR,'foundation-sidebar.png')});
 }
 const rows=await links.evaluateAll(nodes=>nodes.map(el=>{
  const b=el.getBoundingClientRect(),style=getComputedStyle(el);
  return {top:b.top,bottom:b.bottom,left:b.left,right:b.right,display:style.display,fragments:el.getClientRects().length,client:el.clientWidth,scroll:el.scrollWidth};
 }));
 rows.forEach((row,i)=>{
  assert.notEqual(row.display,'inline',`${label}: course ${i+1} must have its own row`);
  assert.equal(row.fragments,1,`${label}: course ${i+1} is fragmented inline text`);
  assert.ok(row.scroll<=row.client+1,`${label}: course ${i+1} overflows`);
  if(i)assert.ok(row.top>=rows[i-1].bottom-1,`${label}: courses ${i} and ${i+1} overlap`);
 });
 if(!wasOpen)await chapter.locator(':scope > summary').click();
}
(async()=>{
 const server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})}),owned=server.process();
 console.log(JSON.stringify({session:'architecture-navigation-review',controller:process.pid,browser:owned.pid,profile:owned.spawnargs.find(a=>a.startsWith('--user-data-dir='))}));
 const browser=await chromium.connect(server.wsEndpoint());
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
  await page.goto(base);
  const signature=()=>page.locator('.site-sidebar [data-module]').evaluateAll(nodes=>nodes.map(n=>n.dataset.module));
  const expectedModules=await signature();assert.equal(expectedModules.length,13);
  assert.equal(await page.locator('.framework-stage').count(),5);
  assert.equal(await page.locator('.module-card').count(),13);
  assert.equal(await page.locator('.module-card[data-coverage=available]').count(),13);
  assert.equal(await page.locator('a[href="#"],a:not([href]),button[disabled]').count(),0);
  assert.match(await page.locator('#communication').innerText(),/已有课程[\s\S]*开始学习/);
  assert.match(await page.locator('#performance').innerText(),/已有课程[\s\S]*开始学习/);
  assert.match(await page.locator('#serving-operations').innerText(),/已有课程[\s\S]*开始学习/);
  assert.deepEqual(await page.locator('a[href^="#"]').evaluateAll(links=>links.filter(a=>!document.getElementById(a.hash.slice(1))).map(a=>a.hash)),[]);
  assert.equal(await page.locator('.chapter-nav').count(),3);
  assert.equal(await page.locator('.foundation-lessons:not(.engine-topic-links) a').count(),13);
  const engineChapter=page.locator('[data-chapter=engine-comparison]');
  assert.equal(await engineChapter.evaluate(e=>e.open),true);
  const engineLinks=engineChapter.locator('.lesson-link');
  assert.equal(await engineLinks.count(),6);
  assert.equal(await engineLinks.first().getAttribute('href').then(h=>h.endsWith('#process')),true);
  assert.equal(await engineLinks.last().getAttribute('href').then(h=>h.endsWith('#crosscut')),true);
  await engineLinks.first().click();await page.waitForURL(url=>url.hash==='#process'&&url.pathname.endsWith('/engine-comparison/index.html'));
  assert.match(await page.locator('.ia-chapterTab[aria-pressed=true]').innerText(),/进程骨架/);
  assert.equal(await page.locator('.ia-stepItem').count(),4);
  await page.locator('[data-chapter=engine-comparison] .lesson-link').last().click();
  await page.waitForURL(url=>url.hash==='#crosscut');
  assert.match(await page.locator('.ia-chapterTab[aria-pressed=true]').innerText(),/生命周期/);
  // Reproduce the reported chapter-1 sidebar problem from every chapter-2 topic.
  for(const width of [1440,1024,390,320]){
   await page.setViewportSize({width,height:1000});
   for(const topic of ['process','wire','prefix','batch','disagg','crosscut']){
    await page.goto(base+'engine-comparison/index.html#'+topic);await page.waitForLoadState('load');
    if(width<901&&!await page.locator('.sidebar-menu').evaluate(e=>e.open))await page.locator('.sidebar-menu>summary').click();
    await checkFoundationNavigation(page,`engine-${topic}-${width}`);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
   }
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.goto(base+'foundations/tokens.html');await page.waitForLoadState('load');
  await checkFoundationNavigation(page,'chapter-1-current');
  assert.equal(await page.locator('.foundation-nav a[aria-current=page]').count(),1);
  await page.goto(base);await page.waitForLoadState('load');
  await checkFoundationNavigation(page,'home');
  await page.locator('[data-chapter=sglang]>summary').click();
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
  const lessons=page.locator('#disaggregation .module-lessons');
  await lessons.locator('summary').focus();await page.keyboard.press('Enter');assert.equal(await lessons.locator('ul').isVisible(),true);
  await page.keyboard.press('Space');assert.equal(await lessons.locator('ul').isVisible(),false);
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
  await page.goto(base+'sglang/inference-overview/journey.html');await page.waitForLoadState('load');
  assert.match(page.url(),/journey.html/);
  assert.deepEqual(await signature(),expectedModules,'entering a lesson retains the entire framework');
  await page.locator('[data-module=kv-memory] > summary').click();
  await page.locator('[data-module=kv-memory] .lesson-link').first().click();await page.waitForLoadState('load');
  assert.match(page.url(),/kv-cache.html/);assert.match(await page.locator('.learning-breadcrumb').innerText(),/05 KV Cache/);
  await page.locator('[data-topic=framework-practice]>summary').click();
  const wasOpen=await page.locator('[data-topic=framework-practice]').evaluate(e=>e.open);
  await page.reload();assert.equal(await page.locator('[data-topic=framework-practice]').evaluate(e=>e.open),wasOpen);
  // The framework pager follows the module sequence, not the old five-page topic order.
  await page.locator('.lesson-pager .next').click();await page.waitForLoadState('load');
  assert.match(page.url(),/kv-memory\/index.html/);assert.match(await page.locator('.learning-breadcrumb').innerText(),/05 KV Cache/);
  await page.locator('.lesson-pager .next').click();await page.waitForLoadState('load');
  assert.match(page.url(),/scheduling\/index.html/);assert.match(await page.locator('.learning-breadcrumb').innerText(),/06 调度/);
  for(const [route,module,stage] of courses){
   await page.goto(base+route);await page.waitForLoadState('load');
   assert.equal(await page.locator('.topic-group').count(),5,route);
   assert.equal(await page.locator('[data-foundation-lab]').count(),0,route+' must not replay foundations');
   assert.equal(await page.locator('script[src*="foundations/player"]').count(),0,route);
   if(route.includes('inference-overview/')){assert.equal(await page.locator('.implementation-map li').count(),3);assert.equal(await page.locator('[data-diagram]').count(),0);}
   assert.deepEqual(await signature(),expectedModules,route);
   assert.equal(await page.locator('[data-chapter=sglang] .lesson-link').count(),courses.length,route);
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
    await checkFoundationNavigation(page,`${route}-${width}`);
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
  // Bookmarked routes retain their existing lesson sequence, with the new loop primer inserted before it.
  await page.goto(base+'sglang/inference-overview/journey.html?learn=first&v=legacy-review');
  const firstRoute=['journey.html','transformer.html','kv-cache.html','scheduling.html'];
  for(let i=0;i<firstRoute.length;i++){
   assert.equal(new URL(page.url()).pathname.endsWith(firstRoute[i]),true);
   assert.equal(new URL(page.url()).searchParams.get('learn'),'first');
   assert.equal(new URL(page.url()).searchParams.has('v'),false,'reading routes use stable URLs');
   assert.equal(await page.locator('.lesson-pager').isVisible(),false);
   assert.equal(await page.locator('.reading-route-pager:visible').count(),1);
   assert.match(await page.locator('.active-reading-route').innerText(),new RegExp(`${i+3} / 6`));
   assert.deepEqual(await signature(),expectedModules);
   if(i<firstRoute.length-1){await page.locator('.reading-route-pager:visible .next').click();await page.waitForLoadState('load');}
  }
  await page.locator('.reading-route-pager:visible .next').click();await page.waitForLoadState('load');
  assert.equal(new URL(page.url()).hash,'#reading-routes');
  await page.goto(base+'sglang/inference-overview/journey.html?learn=unknown');
  assert.equal(await page.locator('.lesson-pager').isVisible(),true,'unknown route keeps the canonical pager');
  assert.equal(await page.locator('.reading-route-pager:visible').count(),0);
  const staticPage=await browser.newPage({javaScriptEnabled:false,viewport:{width:1440,height:1000}});
  await staticPage.goto(base);
  assert.equal(await staticPage.locator('.chapter-choice a:visible').count(),3,'chapter links remain readable without JavaScript');
  assert.equal(await staticPage.locator('.foundation-lessons:not(.engine-topic-links) a').count(),13);
  assert.equal(await staticPage.locator('.module-card').count(),13);
  await staticPage.goto(base+'engine-comparison/index.html');
  await checkFoundationNavigation(staticPage,'chapter-2-no-JS',false);
  await staticPage.goto(base+'sglang/inference-overview/kv-cache.html');
  assert.equal(await staticPage.locator('[data-module=kv-memory] .lesson-link[aria-current=page]').isVisible(),true,'navigation is in HTML, not swapped after load');
  await staticPage.close();
  assert.deepEqual(errors,[]);console.log('PASS: one framework on home and all lessons; canonical module/current state, nested keyboard navigation, cross-module pager, breadcrumbs, placeholder anchors, desktop/mobile and no-JS navigation; chapter-1 links remain separate rows in every chapter-2 topic and all chapter-3 lessons.');
 }finally{await browser.close();await server.close();try{process.kill(owned.pid,0);throw Error('owned browser remains');}catch(e){if(e.code!=='ESRCH')throw e;}}
})().catch(e=>{console.error(e);process.exit(1);});
