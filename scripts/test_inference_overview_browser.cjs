/* Serve .pages-dist; optional OVERVIEW_TEST_URL, PLAYWRIGHT_MODULE,
 * CHROMIUM_EXECUTABLE, OVERVIEW_SCREENSHOT_DIR. */
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 try{
 const page=await browser.newPage({viewport:{width:1440,height:1050}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`);});
 const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
 const go=async slug=>{await page.goto(base+'sglang/inference-overview/'+slug+'.html');if(slug==='index')await page.waitForURL('**/journey.html*');};
 const range=async(id,value)=>page.locator('#'+id).evaluate((el,v)=>{el.value=v;el.dispatchEvent(new Event('input',{bubbles:true}));},value);
 async function capture(name){if(process.env.OVERVIEW_SCREENSHOT_DIR){fs.mkdirSync(process.env.OVERVIEW_SCREENSHOT_DIR,{recursive:true});await page.screenshot({path:path.join(process.env.OVERVIEW_SCREENSHOT_DIR,name+'.png'),fullPage:true});}}
 await page.goto(base);await page.getByRole('link',{name:/从推理全景开始/}).click();assert.match(page.url(),/inference-overview/);assert.match(page.url(),/journey.html/);assert.equal(await page.locator('[data-topic=overview] .course-link').count(),5);
 await go('journey');await page.locator('#journey-next').click();assert.match(await page.locator('#journey-output').innerText(),/Prefill 完成/);assert.equal(await page.locator('#journey-kv .token').count(),4);assert.equal(await page.locator('#journey-output-tokens .token').count(),1);
 for(let i=0;i<4;i++)await page.locator('#journey-next').click();assert.equal(await page.locator('#journey-kv .token').count(),8);assert.equal(await page.locator('#journey-next').isDisabled(),true);
 await range('prompt-size',12);assert.equal(await page.locator('#journey-prev').isDisabled(),true);assert.equal(await page.locator('#journey-kv .token').count(),0);await page.locator('#journey-next').click();await capture('journey-desktop');
 await go('transformer');assert.equal(await page.locator('.attention-cell.allowed').count(),10);assert.equal(await page.locator('.attention-cell.masked').count(),6);
 await page.locator('#phase-decode').click();assert.equal(await page.locator('.attention-cell.allowed').count(),5);assert.equal(await page.locator('#transformer-input .token').count(),1);
 await page.selectOption('#model-part','mlp');assert.match(await page.locator('#transformer-output').innerText(),/前馈网络/);assert.equal(await page.locator('[data-part="mlp"]').getAttribute('class'),'flow-node active');
 await page.locator('#phase-prefill').click();await capture('transformer-desktop');
 await go('kv-cache');await range('reuse-count',7);await range('decode-count',4);assert.equal(await page.locator('#cache-compute').innerText(),'1');assert.equal(await page.locator('#cache-count').innerText(),'12');assert.equal(await page.locator('#cache-output-count').innerText(),'5');assert.equal(await page.locator('.kv-layer').count(),3);
 await go('deployment');assert.equal(await page.locator('.service-instance').count(),1);assert.equal(await page.locator('[data-instance=A] [data-role]').count(),2);assert.equal(await page.locator('.instance-bridge').count(),0);await capture('deployment-unified');await page.selectOption('#deployment-gpus','2');assert.equal(await page.locator('.gpu-chip').count(),2);assert.equal(await page.locator('.service-instance').count(),1);assert.equal(await page.locator('#handoff-controls').isVisible(),false);await page.selectOption('#deployment-mode','split');await page.locator('#handoff-controls').evaluate(el=>el.open=true);assert.equal(await page.locator('.service-instance').count(),2);assert.equal(await page.locator('.gpu-chip').count(),4);assert.equal(await page.locator('[data-instance=P] [data-role=D]').count(),0);assert.equal(await page.locator('[data-instance=D] [data-role=P]').count(),0);assert.equal(await page.locator('.model-residency').count(),2);for(let i=0;i<4;i++){await page.locator('#deployment-next').click();if(i===1)await capture('deployment-transfer');}
 for(const gate of ['gate-kv','gate-meta','gate-slot']){await page.locator('#'+gate).uncheck();assert.match(await page.locator('#deployment-output').innerText(),/等待中/);assert.match(await page.locator('[data-admission=waiting]').innerText(),/检查可执行条件/);assert.equal(await page.locator('[data-instance=D] [data-role=D].is-current').count(),0);assert.match(await page.locator('[data-instance=D]').innerText(),/尚未执行/);await page.locator('#'+gate).check();assert.doesNotMatch(await page.locator('#deployment-output').innerText(),/等待中/);}
 await page.locator('#gate-meta').uncheck();await capture('deployment-wait');await page.selectOption('#deployment-mode','unified');assert.equal(await page.locator('#deployment-step').innerText(),'1 / 5');
 await go('scheduling');for(const n of [1,3,8]){await range('chunk-size',n);for(let i=0;i<Math.ceil(8/n);i++)await page.locator('#schedule-next').click();assert.match(await page.locator('#schedule-output').innerText(),/8 \/ 8/);assert.equal(await page.locator('#schedule-next').isDisabled(),true);}
 await range('latency-q',100);assert.equal(await page.locator('#ttft-value').innerText(),'160 ms');assert.equal(await page.locator('#tpot-value').innerText(),'20 ms');await range('latency-d',40);assert.equal(await page.locator('#ttft-value').innerText(),'160 ms');assert.equal(await page.locator('#total-value').innerText(),'280 ms');
 // Check narrow screens, dark/light, anchors, and readable layouts on every chapter.
 for(const width of [1440,1024,768,390,320])for(const slug of ['index','journey','transformer','kv-cache','deployment','scheduling']){
   await page.setViewportSize({width,height:900});await go(slug);
   for(const theme of ['light','dark']){
     await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
     assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${slug}: ${width}px ${theme} overflow`);
   }
   if(slug==='deployment'){await page.selectOption('#deployment-mode','split');await page.locator('#handoff-controls').evaluate(el=>el.open=true);await page.selectOption('#deployment-gpus','2');for(let i=0;i<4;i++)await page.locator('#deployment-next').click();await page.locator('#gate-kv').uncheck();assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${width}px split overflow`);assert.equal(await page.locator('[data-instance=D] .kv-piece').count(),0);assert.equal(await page.locator('[data-instance=P] .kv-piece').count(),4);assert.equal(await page.locator('.service-instance').count(),2);}
   assert.equal(await page.locator('h1').count(),1);
   const anchors=await page.locator('a[href^="#"]').evaluateAll(links=>links.filter(a=>!document.getElementById(a.hash.slice(1))).map(a=>a.hash));assert.deepEqual(anchors,[]);
   if(width===390&&['transformer','deployment','scheduling'].includes(slug))await capture(slug+'-mobile-dark');
 }
 await page.setViewportSize({width:1440,height:1050});await go('transformer');await page.emulateMedia({reducedMotion:'reduce'});
 await page.locator('#phase-decode').focus();await page.keyboard.press('Enter');assert.equal(await page.locator('#phase-decode').getAttribute('aria-pressed'),'true');
 await page.goto(base+'sglang/pd-prefill-lifecycle/index.html');assert.equal(await page.locator('.site-sidebar a[href*="inference-overview"]').count(),5);
 if(!await page.locator('[data-topic=overview]').evaluate(el=>el.open))await page.locator('[data-topic=overview]>summary').click();await page.locator('.site-sidebar a[href*="inference-overview/journey.html"]').click();assert.match(await page.locator('h1').innerText(),/一条请求怎样变成回答/);
 assert.deepEqual(errors,[]);console.log('PASS: six interactive modules, gating, boundary inputs, chapter navigation, 320/390/768/1024/1440px, instance boundaries, GPU groups, themes, keyboard; no browser errors');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
