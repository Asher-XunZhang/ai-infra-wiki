/* Full chapter checks: actual controls, transitions, branch outcomes and readable fallbacks. */
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const M=require('../../pages/foundations/model.js');
const courses=require('../../pages/foundations/courses.json');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)errors.push(r.status()+' '+r.url());});
  const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
  const lab=page.locator('[data-foundation-lab][data-full=true]'),btn=a=>lab.locator(`[data-action=${a}]`);
  for(const c of courses){
   await page.goto(base+'foundations/'+c.slug+'.html');
   if(await page.locator('.batch-extra').count())await page.locator('.batch-extra>summary').click();
   assert.equal(await page.locator('h1').count(),1);
   assert.equal(await page.locator('.foundation-nav [aria-current=page]').count(),1);
   assert.equal(await page.locator('[data-chapter=foundations]').evaluate(e=>e.open),true);
   assert.equal(await page.locator('.chapter-nav').count(),2);
   assert.equal(await page.locator('.site-sidebar .lesson-link').count(),20);
   assert.ok(await page.locator('#evidence a').count());
   const def=M.definitions[c.slug],allowed=require('../../pages/motion/integration.js').foundation[c.slug]??def.options.map((_,i)=>i);
   for(const width of [1440,390,320]){
    await page.setViewportSize({width,height:1000});
    for(const v of allowed){
     await lab.locator('select').selectOption(String(v));
     assert.equal(await lab.getAttribute('data-step'),'0','changing a scenario resets it');
     let first='';
     for(let s=0;s<def.steps.length;s++){
      assert.equal(await lab.getAttribute('data-step'),String(s));
      const scene=await lab.locator('.fd-scene').innerText();if(s===0)first=scene;
      assert.ok(scene.length>20);assert.doesNotMatch(scene,/undefined|NaN/);
      assert.ok((await lab.locator('.fd-caption').innerText()).length>0,`${c.slug}/${v}/${s} explanation`);
      for(const theme of ['light','dark']){
       await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
       assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${c.slug}/${v}/${s}/${width}/${theme}`);
      }
      if(s<def.steps.length-1)await btn('next').click();
      else assert.notEqual(scene,first,'steps must visibly change data or state');
     }
     assert.equal(await btn('next').isDisabled(),true);
     await btn('prev').click();assert.equal(await lab.getAttribute('data-step'),String(def.steps.length-2));
     await btn('reset').click();assert.equal(await btn('prev').isDisabled(),true);
    }
   }
   // Wrong predictions receive an explanation, then a corrected answer is acknowledged.
   await page.locator(`.fd-quiz input[value="${(c.answer+1)%c.choices.length}"]`).check();
   await page.locator('.fd-quiz button').click();assert.match(await page.locator('.fd-feedback').innerText(),/再看一下/);
   await page.locator(`.fd-quiz input[value="${c.answer}"]`).check();
   await page.locator('.fd-quiz button').click();assert.match(await page.locator('.fd-feedback').innerText(),/判断正确/);
   const broken=await page.locator('a[href^="#"]').evaluateAll(es=>es.filter(e=>!document.getElementById(e.hash.slice(1))).map(e=>e.hash));assert.deepEqual(broken,[]);
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.goto(base+'foundations/tokens.html?mv=1&ms=2&v=review');
  const film=page.locator('.mechanism-film');assert.equal(await lab.count(),0);
  assert.equal(await film.getAttribute('data-position'),'2.0000');
  assert.match(await film.textContent(),/104100.*100134.*102182.*100134.*1773/);
  await page.emulateMedia({reducedMotion:'reduce'});
  await film.locator('[data-motion-action=next]').focus();await page.keyboard.press('Enter');
  assert.equal(await film.getAttribute('data-position'),'3.0000');
  assert.equal(new URL(page.url()).searchParams.get('v'),'review');
  await page.reload();assert.equal(await film.getAttribute('data-position'),'3.0000');
  await page.clock.install();
  await page.goto(base+'foundations/generation.html');
  const extension=page.locator('.foundation-extension');
  assert.equal(await extension.evaluate(e=>e.open),false);
  assert.equal(await extension.locator('.wr-scene').count(),0,'closed supplements initialize on demand');
  await extension.locator(':scope>summary').click();
  const oldPlay=extension.locator('[data-id=j-play]');await oldPlay.click();await page.clock.runFor(2300);
  await oldPlay.click();const paused=await extension.locator('.wr-scene').textContent();
  await oldPlay.click();await extension.locator(':scope>summary').click();const closed=await extension.locator('.wr-scene').textContent();await page.clock.runFor(9000);
  assert.equal(await extension.locator('.wr-scene').textContent(),closed,'closed supplement pauses');
  await page.goto(base+'sglang/inference-overview/journey.html');
  assert.equal(await page.locator('[data-foundation-lab]').count(),0);
  assert.equal(await page.locator('.implementation-map li').count(),3);
  // No JS: both chapters, all lesson body sections, prerequisites, evidence and quiz explanations remain.
  const staticPage=await browser.newPage({javaScriptEnabled:false});
  for(const c of courses){
   await staticPage.goto(base+'foundations/'+c.slug+'.html');
   assert.equal(await staticPage.locator('.foundation-copy section').count(),c.sections.length);
   assert.equal(await staticPage.locator('.fd-quiz').isVisible(),false);
   await staticPage.locator('.foundation-fallback summary').click();assert.ok((await staticPage.locator('.foundation-fallback p').innerText()).length>20);
   assert.equal(await staticPage.locator('.foundation-nav a').count(),12);
   assert.equal(await staticPage.locator('.lesson-pager a').count(),2);
  }
  await staticPage.close();assert.deepEqual(errors,[]);
  console.log('PASS: 12 foundation lessons; every scenario/step at 1440/390/320px, light/dark, quiz feedback, real tokenizer IDs, keyboard, playback, reduced motion, URL restore, supplement pause and no-JS content.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
