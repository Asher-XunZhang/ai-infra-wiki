/* Serve the built pages, install Playwright/Chromium, then run this browser regression.
 * Optional: PREFILL_TEST_URL, PREFILL_PLAYWRIGHT_MODULE, CHROMIUM_EXECUTABLE, PREFILL_SCREENSHOT_DIR. */
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PREFILL_PLAYWRIGHT_MODULE || 'playwright');
async function capture(target,name){if(process.env.PREFILL_SCREENSHOT_DIR){require('node:fs').mkdirSync(process.env.PREFILL_SCREENSHOT_DIR,{recursive:true});await target.screenshot({path:require('node:path').join(process.env.PREFILL_SCREENSHOT_DIR,name)});}}
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 const page=await browser.newPage({viewport:{width:1600,height:1050}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`);});
 await page.clock.install();await page.goto(process.env.PREFILL_TEST_URL || 'http://127.0.0.1:8765/sglang/pd-prefill-lifecycle/');
 const report=await page.evaluate(()=>{
  const api=window.PrefillLifecycle,issues=[],rectIssues=[],routes=[],stats=[];
  const seek=(id,n)=>{const el=document.getElementById(id);el.value=n;el.dispatchEvent(new Event('input',{bubbles:true}));};
  for(const [name,config] of Object.entries(api.presets)){
   const sel=document.getElementById('config-preset');sel.value=name;sel.dispatchEvent(new Event('change',{bubbles:true}));
   const model=api.createScenario(config);
   for(let e=0;e<model.events.length;e++){
    seek('player-seek',e);
    for(let f=0;f<model.events[e].frames.length;f++){
     if(f)seek('frame-seek',f);
     const frame=model.events[e].frames[f],title=document.getElementById('operation-title').textContent,text=document.getElementById('operation-visual').textContent;
     if(title!==frame.title||!text||/undefined|NaN/.test(text))issues.push(`${name}:${frame.key} render`);
     const current=[...document.querySelectorAll('.pipeline-node.is-active')].map(n=>n.dataset.node);
     if(current.length!==frame.active.length||current.some(id=>!frame.active.includes(id)))issues.push(`${name}:${frame.key} highlight`);
     if(document.documentElement.scrollWidth>innerWidth)issues.push(`${name}:${frame.key} overflow`);
     for(const line of document.querySelectorAll('.active-route')){
      const length=line.getTotalLength();
      for(let k=2;k<98;k++){
       const p=line.getPointAtLength(length*k/100);
       for(const [id,n] of Object.entries(api.nodes))if(!frame.active.includes(id)&&!frame.routes.some(w=>w.from===id||w.to===id)&&p.x>n.x+3&&p.x<n.x+n.w-3&&p.y>n.y+3&&p.y<n.y+n.h-3)routes.push(`${name}:${frame.key} crosses ${id}`);
      }
     }
    }
   }
   stats.push([name,model.events.length,model.frames.length]);
  }
  for(const node of document.querySelectorAll('.pipeline-node')){
   const box=node.querySelector('rect').getBBox();for(const t of node.querySelectorAll('text')){const r=t.getBBox();if(r.x<box.x||r.x+r.width>box.x+box.width||r.y<box.y||r.y+r.height>box.y+box.height)rectIssues.push(node.dataset.node);}
  }
  return {issues,rectIssues,routes:[...new Set(routes)],stats};
 });
 assert.deepEqual(report.issues,[]);assert.deepEqual(report.rectIssues,[]);assert.deepEqual(report.routes,[]);
 await page.selectOption('#config-preset','chunks');
 const seek=async(id,n)=>page.locator('#'+id).evaluate((el,n)=>{el.value=n;el.dispatchEvent(new Event('input',{bubbles:true}));},n);
 const groups=await page.evaluate(()=>window.PrefillLifecycle.createScenario().events.map((e,i)=>({i,compressed:e.compressed,length:e.frames.length,key:e.key})).filter(e=>e.compressed));
 await seek('player-seek',groups[0].i);await page.locator('#play-group').click();await page.clock.runFor(700);assert.equal(await page.locator('#frame-counter').innerText(),`2 / ${groups[0].length}`);
 await page.locator('#player-play').click();const paused=await page.locator('#operation-title').innerText();await page.clock.runFor(2000);assert.equal(await page.locator('#operation-title').innerText(),paused);
 await page.locator('#frame-prev').click();assert.equal(await page.locator('#frame-counter').innerText(),`1 / ${groups[0].length}`);
 await page.locator('#play-group').click();await page.clock.runFor(650*(groups[0].length+1));assert.equal(await page.locator('#player-play').getAttribute('aria-pressed'),'false');assert.equal(await page.locator('#player-counter').innerText(),`${groups[0].i+1} / 34`);
 await seek('player-seek',12);await page.locator('#player-play').click();await page.clock.runFor(100);assert(await page.locator('.token.selected').first().evaluate(el=>el.getAnimations().length>0));await page.locator('#player-play').click();
 await page.locator('#config-requests').fill('3');await page.locator('#config-requests').dispatchEvent('change');await page.locator('#config-inputLength').fill('5');await page.locator('#config-chunkSize').fill('8');await page.locator('#config-batchSize').fill('3');await page.locator('button[type=submit]').click();assert.match(await page.locator('#scenario-summary').innerText(),/3 条请求/);
 await seek('player-seek',13);const saved=page.url(),savedTitle=await page.locator('#operation-title').innerText();await page.reload();assert.equal(await page.locator('#operation-title').innerText(),savedTitle);assert.equal(await page.locator('#config-requests').inputValue(),'3');assert.equal(page.url(),saved);
 await page.locator('#config-chunkSize').fill('3');await page.locator('button[type=submit]').click();assert.match(await page.locator('#config-error').innerText(),/整除/);assert.equal(await page.locator('#operation-title').innerText(),savedTitle);
 await page.selectOption('#config-preset','transfer');assert.equal(await page.locator('#failure-matrix input').count(),9);await page.locator('[data-fail-rank="0"][value="R2"]').check();await page.locator('button[type=submit]').click();await seek('player-seek',999);assert.match(await page.locator('#operation-visual').innerText(),/R2 · 交接异常/);
 await page.selectOption('#config-preset','batching');await page.locator('.scenario-settings summary').click();await seek('player-seek',13);await page.locator('#operation-lab').scrollIntoViewIfNeeded();await capture(page,'batching-desktop.png');
 await page.evaluate(()=>document.documentElement.dataset.theme='dark');await capture(page.locator('#pipeline-player'),'batching-dark.png');
 await page.emulateMedia({reducedMotion:'reduce'});await page.locator('#player-play').click();assert.equal(await page.locator('.packed-tokens span').first().evaluate(el=>el.getAnimations().length),0);await page.locator('#player-play').click();
 await page.setViewportSize({width:390,height:844});await page.evaluate(()=>document.documentElement.dataset.theme='light');await page.locator('.scenario-settings summary').click();await page.locator('#config-requests').fill('4');await page.locator('#config-inputLength').fill('48');await page.locator('#config-chunkSize').fill('0');await page.locator('#config-batchSize').fill('4');await page.locator('button[type=submit]').click();await seek('player-seek',12);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await page.locator('.scenario-settings summary').click();await page.locator('#operation-lab').scrollIntoViewIfNeeded();await capture(page,'mobile.png');
 assert.deepEqual(errors,[]);console.log(JSON.stringify({pass:true,...report,errors,checks:['all preset frames','route/module collision','node label containment','compressed-group playback/seek','pause','animation','custom generation','URL restore','invalid budget','custom failure matrix','dark theme','reduced motion','390px max input']}));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
