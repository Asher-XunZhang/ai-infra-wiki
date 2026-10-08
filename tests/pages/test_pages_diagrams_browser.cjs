/* PP guides, lifecycle focus and one-object references on the built site.
 * Optional PAGES_TEST_URL, PLAYWRIGHT_MODULE, CHROMIUM_EXECUTABLE, DIAGRAM_SCREENSHOT_DIR. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 try{
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],base=process.env.PAGES_TEST_URL||'http://127.0.0.1:8765/';
 page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`);});
 const id=n=>page.locator(`[data-id="${n}"]`),reader=()=>page.locator('[data-pp-reader]');
 const go=p=>page.goto(base+'sglang/'+p);
 const check=async()=>{assert.doesNotMatch(await reader().innerText(),/undefined|NaN/);assert.equal(await reader().locator('[viewBox*="NaN"]').count(),0);};
 async function shot(target,name){if(process.env.DIAGRAM_SCREENSHOT_DIR){fs.mkdirSync(process.env.DIAGRAM_SCREENSHOT_DIR,{recursive:true});await target.screenshot({path:path.join(process.env.DIAGRAM_SCREENSHOT_DIR,name+'.png')});}}
 await go('pd-prefill-pp-loop/quick.html');await id('pp-rank').waitFor();assert.match(await reader().innerText(),/13.75–29.8/);assert.equal(await page.locator('#chart').isVisible(),false);
 const scenes=await page.evaluate(()=>PP_SCENARIOS.map(s=>s.id));
 for(const scene of scenes){await page.selectOption('#scenario-select',scene);await page.waitForFunction(id=>window.PP_SCENARIO_STATE?.id===id,scene);
   for(const rank of [0,1,2]){await id('pp-rank').selectOption(String(rank));const loops=await id('pp-loop-index').locator('option').evaluateAll(es=>es.map(e=>e.value));for(const loop of [loops[0],'5',loops.at(-1)]){await id('pp-loop-index').selectOption(loop);await check();}}
   await id('pp-batch').click();for(const batch of [1,2,3,4,5]){await id('pp-batch-index').selectOption(String(batch));await check();assert.match(await reader().innerText(),new RegExp(`M${batch}`));}await id('pp-loop').click();
 }
 await page.selectOption('#scenario-select','baseline');await page.waitForFunction(()=>window.PP_SCENARIO_STATE?.id==='baseline');await id('pp-rank').selectOption('0');await id('pp-loop-index').selectOption('5');await shot(reader(),'pp-guide-desktop');
 await page.locator('#full-timeline-view>summary').click();assert.equal(await page.locator('#chart').isVisible(),true);await page.selectOption('#batch','3');assert.match(await page.locator('#selection-title').innerText(),/M3/);
 await go('pd-prefill-pp-loop/quick.html?batch=2&view=batch&scenario=slow-link');await id('pp-batch-index').waitFor();assert.equal(await id('pp-batch-index').inputValue(),'2');await page.waitForFunction(()=>window.PP_SCENARIO_STATE?.id==='slow-link');await check();
 await go('pd-prefill-pp-loop/index.html');await id('dependency-event').waitFor();assert.equal(await reader().locator('svg').count(),2);assert.match(await reader().innerText(),/27.72 → 32.9/);
 await id('dependency-compare').selectOption('baseline');await page.waitForFunction(()=>document.querySelector('[data-pp-reader]').getAttribute('aria-busy')===null);assert.match(await reader().innerText(),/27.72 → 27.72/);
 await id('dependency-compare').selectOption('slow-link');await page.waitForFunction(()=>document.querySelector('[data-pp-reader]').getAttribute('aria-busy')===null);
 for(const key of ['0:5:proxy_work','0:5:gpu','0:5:recv_out']){await id('dependency-event').selectOption(key);await check();}
 await reader().locator('summary').first().click();await id('dependency-1').click();assert.equal(await id('dependency-back').count(),1);await check();await id('dependency-back').click();assert.equal(await id('dependency-event').inputValue(),'0:5:recv_out');
 assert.equal(await page.locator('#timeline iframe').count(),1);await shot(reader(),'dependency-desktop');
 await go('pd-prefill-lifecycle/index.html');assert.equal(await page.locator('#operation-visual .operation-focus .wr-three .wr-node').count(),3);await page.locator('#player-next').click();assert.equal(await page.locator('#operation-visual .operation-detail').isVisible(),true);await page.locator('#operation-visual .operation-detail>summary').click();assert.equal(await page.locator('#operation-visual .operation-detail').evaluate(e=>e.open),true);await page.locator('#player-next').click();assert.equal(await page.locator('#operation-visual .operation-detail').evaluate(e=>e.open),true);assert.equal(await page.locator('.life-step').count(),12);
 await go('pd-dataflow/index.html');assert.equal(await page.locator('#data-routes-overview').count(),1);await page.emulateMedia({reducedMotion:'reduce'});
 const queueReport=await page.evaluate(()=>{const issues=[];for(const scenario of PDQueues.scenarios)for(const rank of [0,1,2]){const sel=document.getElementById('q-scenario');sel.value=scenario.id;sel.dispatchEvent(new Event('change'));const rs=document.getElementById('q-rank');rs.value=rank;rs.dispatchEvent(new Event('change'));const model=PDQueues.create(scenario.id,rank);for(let i=0;i<model.frames.length;i++){const step=document.getElementById('q-steps');step.value=i;step.dispatchEvent(new Event('change'));const expected=PDQueues.nodes.filter(n=>n.id!=='exit'&&model.frames[i].after[n.id]).map(n=>n.id).sort(),actual=[...document.querySelectorAll('#q-reference-lines path')].map(p=>p.dataset.reference).sort();if(JSON.stringify(expected)!==JSON.stringify(actual))issues.push(`${scenario.id}:${rank}:${i}`);if(document.querySelectorAll('#q-map .q-request').length!==1)issues.push('duplicated Req');}}return issues;});assert.deepEqual(queueReport,[]);
 await page.selectOption('#q-scenario','normal');await page.selectOption('#q-rank','0');await page.selectOption('#q-steps','6');assert.equal(await page.locator('#q-reference-lines path').count(),2);await shot(page.locator('#q-map'),'queue-desktop');
 for(const width of [390,320])for(const url of ['pd-prefill-pp-loop/quick.html','pd-prefill-pp-loop/index.html','pd-prefill-lifecycle/index.html','pd-dataflow/index.html']){
   await page.setViewportSize({width,height:1000});await go(url);if(url.includes('pp-loop'))await reader().locator('svg').first().waitFor();
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${url} ${width} overflow`);
   const bad=await page.locator('.lesson-diagram').evaluateAll(roots=>roots.flatMap(root=>{const box=root.getBoundingClientRect(),bad=[];for(const e of root.querySelectorAll('*')){if(!e.getClientRects().length||e.tagName==='OPTION')continue;const r=e.getBoundingClientRect();if(r.width&&(r.right>box.right+2||r.left<box.left-2))bad.push(e.tagName+':'+e.textContent.slice(0,30));}return bad;}));assert.deepEqual(bad,[],`${url} ${width}`);
   if(width===390){const target=url.includes('pp-loop')?reader():url.includes('lifecycle')?page.locator('#operation-visual'):page.locator('#q-map');await shot(target,url.split('/')[0]+'-'+(url.includes('quick')?'quick':'detail')+'-mobile');}
 }
 assert.deepEqual(errors,[]);console.log('PASS: all PP scenarios × ranks/edge loops/batches; original timeline, causal comparison/drilldown, lifecycle focus; 493 queue frames keep one Req and exact references; mobile layouts');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
