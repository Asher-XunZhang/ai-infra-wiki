const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:18765/',out=process.env.MOTION_SCREENSHOT_DIR;
const pages=[...['tokens','generation','attention','cache','hardware','batching','parallel','communication','deployment','speculation','serving'].map(k=>'foundations/'+k+'.html'),...['system-overview','request-runtime','model-execution','kv-memory','scheduling','parallelism','communication','advanced-generation','serving-operations','pd-dataflow','pd-prefill-lifecycle'].map(k=>'sglang/'+k+'/index.html')];
(async()=>{const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE});try{
 const p=await browser.newPage(),errors=[],links=new Set();p.on('pageerror',e=>errors.push(e.message));
 for(const url of pages){await p.goto(base+url);const panels=p.locator('[data-boundary-panel]');assert.ok(await panels.count(),url+' has handoff context');
  for(const width of [320,390,768,1440]){await p.setViewportSize({width,height:1000});assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,url+' width '+width);}
  for(const href of await panels.locator('a').evaluateAll(es=>es.map(e=>e.href)))links.add(href);
 }
 for(const url of links){const r=await p.request.get(url);assert.equal(r.status(),200,url);}
 await p.goto(base+'foundations/parallel.html');await p.locator('[data-parallel-family=PP]').click();const seek=async v=>p.locator('.motion-scrub input').evaluate((e,v)=>{e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}));},v);
 await seek(3.2);const before=await p.locator('[data-motion-id=MB1]').getAttribute('transform');
 await p.getByLabel('PP 部署位置').selectOption('cross-host');assert.equal(await p.locator('[data-motion-id=MB1]').getAttribute('transform'),before,'placement must not alter simulated computation');assert.match(await p.locator('.mechanism-film svg').textContent(),/跨主机/);assert.match(await p.locator('.handoff-endpoint').last().textContent(),/主机 B.*GPU 1/s);
 await p.locator('.mechanism-film select[aria-label="连续演示条件"]').selectOption('3');assert.match(await p.locator('.mechanism-film svg').textContent(),/跨主机/,'scenario reset retains selected placement');
 if(out){fs.mkdirSync(out,{recursive:true});for(const width of [390,1440])for(const placement of ['same-host','cross-host']){await p.setViewportSize({width,height:1000});await p.getByLabel('PP 部署位置').selectOption(placement);await seek(3.2);await p.locator('.mechanism-film').screenshot({path:path.join(out,`pp-boundary-${width}-${placement}.png`)});}}
 await p.goto(base+'sglang/parallelism/');for(const mode of ['tp','pp','dp','ep','cp']){await p.locator(`button[data-mode=${mode}]`).click();assert.equal(await p.locator('[data-parallel-lab] [data-boundary-panel]').getAttribute('data-boundary-panel'),'parallel-'+mode);}
 await p.locator('button[data-mode=pp]').click();await p.getByLabel('PP 部署位置').selectOption('cross-host');await p.locator('#parallel-next').click();assert.equal(await p.getByLabel('PP 部署位置').inputValue(),'cross-host');
 assert.deepEqual(errors,[]);console.log(`PASS ${pages.length} courses: boundary context, four widths, ${links.size} local destinations, PP topology independent of timeline, five parallel modes.`);
 }finally{await browser.close();}})().catch(e=>{console.error(e);process.exit(1)});
