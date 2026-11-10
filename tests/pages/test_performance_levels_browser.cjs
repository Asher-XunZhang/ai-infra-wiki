'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/',shots=process.env.PERFORMANCE_SCREENSHOT_DIR;
(async()=>{
 const server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 const proc=server.process(),profile=proc.spawnargs.find(a=>a.startsWith('--user-data-dir=')).slice('--user-data-dir='.length);
 const identity={session:'performance-levels-review',controller:process.pid,browser:proc.pid,profile};console.log(JSON.stringify(identity));let browser;
 try{
  browser=await chromium.connect(server.wsEndpoint());const page=await browser.newPage({viewport:{width:1440,height:1080}}),errors=[],root=page.locator('[data-performance-pd]');
  page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`);});
  if(shots){fs.mkdirSync(shots,{recursive:true});fs.writeFileSync(path.join(shots,'session.json'),JSON.stringify(identity));}
  const go=async(query='')=>{await page.goto(base+'sglang/performance-engineering/?'+query+'#perf-title');await page.waitForFunction(()=>document.querySelector('[data-performance-pd]').dataset.time!==undefined);};
  const seek=async t=>page.locator('[data-role=clock]').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));},t);
  const state=()=>page.locator('[data-role=accessible]').textContent();
  const shot=async name=>{if(!shots)return;await page.locator('#perf-title').evaluate(e=>{document.activeElement?.blur();window.scrollTo({top:scrollY+e.getBoundingClientRect().top-document.querySelector('.site-header').getBoundingClientRect().bottom-16,behavior:'instant'});});await page.waitForTimeout(100);await page.screenshot({path:path.join(shots,name+'-viewport.png')});await root.screenshot({path:path.join(shots,name+'-panel.png')});};
  await go();assert.equal(await root.getAttribute('data-view'),'overview');assert.equal(await page.locator('[data-role=diagram] .lane-name').count(),5);assert.equal(await page.locator('[data-role=options]').getAttribute('open'),null);assert.equal(await page.locator('[data-role=formula]').isVisible(),false);assert.equal(await page.locator('[data-role=event-source]').isVisible(),false);assert(await page.locator('[data-role=diagram]').isVisible());await shot('default');
  for(const query of ['mode=clock&condition=gate','mode=kv&condition=chunked','mode=stream&condition=bundle','mode=sample&condition=failure']){
   await go(query+'&t=310');const before=await state(),metric=await page.locator('[data-role=metric]').inputValue();
   for(const view of ['modules','source','overview']){
    await page.locator(`button[data-view=${view}]`).click();assert.equal(await state(),before);assert.equal(await page.locator('[data-role=metric]').inputValue(),metric);assert.equal(await root.getAttribute('data-time'),await page.locator('[data-role=diagram]').getAttribute('data-clock'));assert.equal(await page.locator('[data-role=formula]').isVisible(),view==='source');assert.equal(new URL(page.url()).searchParams.get('view'),view);
    if(query.startsWith('mode=kv'))await shot('chunk-'+view);
   }
   await page.reload();await page.waitForFunction(()=>document.querySelector('[data-performance-pd]').dataset.time!==undefined);assert.equal(await state(),before);assert.equal(await root.getAttribute('data-view'),'overview');
  }
  // A metric available only in the source menu remains selected when simplifying.
  await go('view=source&mode=clock&t=450&metric=p-ttft');const frozen=await state();await page.locator('button[data-view=overview]').click();assert.equal(await page.locator('[data-role=metric]').inputValue(),'p-ttft');assert.equal(await state(),frozen);
  await go();await page.locator('[data-role=options] summary').click();await page.locator('[data-role=condition]').selectOption('poll');assert.equal(await page.locator('[data-role=options]').getAttribute('open'),'');assert.equal(await root.getAttribute('data-time'),'0.00');
  await page.locator('[data-role=play]').click();await page.waitForFunction(()=>+document.querySelector('[data-performance-pd]').dataset.time>5);await page.locator('button[data-view=modules]').click();assert.equal(await root.getAttribute('data-playing'),'true');await page.locator('[data-role=speed]').selectOption('2');await page.locator('button[data-view=overview]').click();await page.waitForFunction(()=>document.querySelector('[data-performance-pd]').dataset.playing==='false',{},{timeout:15000});assert.equal(+await root.getAttribute('data-time'),715);
  // Overview click is mapped to the original teaching-clock start, then expands in place.
  await go('mode=kv&condition=chunked');await page.locator('[data-event=o-p-work]').focus();await page.keyboard.press('Enter');assert.equal(+await root.getAttribute('data-time'),210);await page.locator('button[data-view=modules]').click();assert.equal(+await root.getAttribute('data-time'),210);
  for(const width of [1440,768,390])for(const theme of ['light','dark']){
   await page.setViewportSize({width,height:1080});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
   for(const mode of ['clock','kv','stream','sample']){
    await page.locator(`button[data-mode=${mode}]`).click();await seek(310);
    for(const view of ['overview','modules','source']){
     await page.locator(`button[data-view=${view}]`).click();assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${width}/${theme}/${mode}/${view}`);
     const outside=await page.locator('[data-role=diagram]').evaluate(svg=>[...svg.querySelectorAll('text')].filter(t=>{const b=t.getBBox();return b.x<0||b.x+b.width>1101;}).map(t=>t.textContent));assert.deepEqual(outside,[]);
    }
   }
   if(width===390&&theme==='light'){await page.locator('button[data-mode=clock]').click();await page.locator('button[data-view=overview]').click();await shot('mobile');}
  }
  await page.setViewportSize({width:1440,height:1080});await page.locator('button[data-mode=kv]').click();await page.locator('button[data-view=overview]').click();await page.emulateMedia({reducedMotion:'reduce'});await seek(370);assert.equal(await page.locator('[data-pulse=kv-last]').getAttribute('stroke-dashoffset'),'0.00');await shot('dark-overview');assert.deepEqual(errors,[]);
  console.log('PASS default overview, all three levels, unchanged clock/metric/state, playing switch, reload, keyboard, full playback, 4 modes × 3 widths × 2 themes.');
 }finally{
  if(browser)await browser.close();await server.close();for(let i=0;i<20&&(proc.exitCode===null&&proc.signalCode===null);i++)await new Promise(r=>setTimeout(r,50));
  let alive=false;try{process.kill(proc.pid,0);alive=true;}catch{}assert.equal(alive,false,'owned Chromium still alive');assert.equal(fs.existsSync(profile),false,'owned profile remains');console.log('CLEANED',JSON.stringify(identity));
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
