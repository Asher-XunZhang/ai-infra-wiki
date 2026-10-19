const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
const output=process.env.MOTION_SCREENSHOT_DIR;if(output)fs.mkdirSync(output,{recursive:true});
const capture=async(target,file)=>{if(output)await target.screenshot({path:path.join(output,file)});};
const foundation=['tokens','generation','attention','cache','batching','hardware','parallel','communication','deployment','speculation','serving','metrics'];
const engine=['request-runtime','model-execution','kv-memory','scheduling','communication','advanced-generation','serving-operations','pd-dataflow','pd-prefill-lifecycle'];
const pages=[...foundation.map(x=>`foundations/${x}.html`),...engine.map(x=>`sglang/${x}/index.html`)];
(async()=>{const server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});const owned=server.process();console.log(JSON.stringify({session:'test_mechanism_motion_browser.cjs',controller:process.pid,browser:owned.pid,profile:owned.spawnargs.find(a=>a.startsWith('--user-data-dir='))}));const browser=await chromium.connect(server.wsEndpoint());try{
 const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 const seek=async n=>page.locator('.motion-scrub input').evaluate((e,n)=>{e.value=n;e.dispatchEvent(new Event('input',{bubbles:true}));},n);
 let variants=0,frames=0;
 for(const url of pages){await page.goto(base+url);const film=page.locator('.mechanism-film');await film.waitFor({state:'attached'});assert.equal(await film.count(),1,url);
  const supplement=page.locator('.motion-supplement').filter({has:film});
  if(await supplement.count()){assert.equal(await supplement.evaluate(e=>e.open),false,'mechanism-specific view comes first');await supplement.locator(':scope>summary').click();}
  const choice=film.locator('select[aria-label="连续演示条件"]'),count=await require('./choose_motion_scene.cjs').count(page);
  for(let variant=0;variant<count;variant++){if(count>1)await require('./choose_motion_scene.cjs')(page,variant);variants++;
   const max=Number(await film.locator('.motion-scrub input').getAttribute('max'));
   await film.locator('[data-motion-id]').first().evaluate(e=>window.testIdentityNode=e);
   for(let n=0;n<=max;n++){await seek(n);frames++;assert.equal(await film.locator('[data-motion-id]').first().evaluate(e=>e===window.testIdentityNode),true,url+' must retain actor nodes');
    assert.ok((await film.locator('.motion-action').innerText()).length>0,url);
    const overflow=await film.locator('svg text').evaluateAll(es=>es.filter(e=>e.closest('[opacity="0.000"]')===null&&e.classList.contains('motion-object-title')).filter(e=>e.getComputedTextLength()>184).map(e=>e.textContent));assert.deepEqual(overflow,[],url+' label overflow');
   }
   // Rewind reconstructs the same initial state rather than running inverse side effects.
   await seek(0);const before=await film.locator('[data-motion-id]').evaluateAll(es=>es.map(e=>e.getAttribute('transform')));await seek(max);await seek(0);assert.deepEqual(await film.locator('[data-motion-id]').evaluateAll(es=>es.map(e=>e.getAttribute('transform'))),before);
  }
  const href=await film.locator('.motion-evidence').getAttribute('href');if(href.startsWith('#'))assert.equal(await page.locator(href).count(),1,url+' evidence target');else assert.match(href,/^https:\/\/github.com\/.+\/[0-9a-f]{40}\//);
  for(const width of [390,320,1440]){await page.setViewportSize({width,height:1100});for(const theme of ['dark','light']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,url+' page overflow');}}
 }
 await page.goto(base+'foundations/batching.html');const film=page.locator('.mechanism-film');await film.scrollIntoViewIfNeeded();await require('./choose_motion_scene.cjs')(page,0);
 const A=film.locator('[data-motion-id=A]'),getX=async()=>Number(await A.getAttribute('data-x'));
 await seek(0);const x0=await getX();await seek(.5);const xm=await getX();await seek(1);const x1=await getX();assert.ok(x0<xm&&xm<x1,'midpoint is actual travel, not a card replacement');
 await seek(0);await film.locator('[data-motion-action=play]').click();await page.waitForTimeout(650);await film.locator('[data-motion-action=play]').click();const paused=await getX();assert.ok(paused>x0&&paused<x1);await page.waitForTimeout(400);assert.equal(await getX(),paused,'pause must freeze geometry');
 await capture(film,'batching-mid-pause.png');
 await film.locator('[data-motion-action=play]').click();await page.waitForTimeout(300);assert.ok(await getX()>paused);await film.locator('[data-motion-action=reset]').click();await page.waitForTimeout(350);assert.equal(await getX(),x0,'reset cancels old animation');
 await seek(3);assert.match(await film.locator('.batch-empty').first().textContent(),/空位/);assert.equal(await film.locator('.batch-empty').first().getAttribute('opacity'),'1');
 const B=film.locator('[data-motion-id=B]'),stationary=await B.getAttribute('transform');
 await seek(3.5);assert.equal(await B.getAttribute('transform'),stationary,'B must retain its slot while C enters');await capture(film,'batching-refill.png');
 await seek(4.5);for(const id of ['B','C']){const filled=await film.locator(`[data-motion-id=${id}] [data-output-token]`).evaluateAll(es=>es.map(e=>Number(e.dataset.fill)));assert.ok(filled.some(n=>n>0&&n<1),'a new token actually grows during the round');}
 assert.equal(await B.getAttribute('transform'),stationary,'generation grows sequence, not request position');
 const slotGeometry=await film.locator('[data-batch-slot]').evaluateAll(es=>es.map(e=>e.outerHTML));
 for(const n of [2,2.5,3,3.5,4,4.5,5,6,7,8]){await seek(n);assert.deepEqual(await film.locator('[data-batch-slot]').evaluateAll(es=>es.map(e=>e.outerHTML)),slotGeometry,'execution slots stay fixed');}
 assert.deepEqual(await film.locator('.batch-round-members').allTextContents(),['A + B','B + C','B']);
 assert.equal(await film.locator('[data-output-token][data-fill="1.000"]').count(),5);
 await capture(film,'batching-complete.png');
 await require('./choose_motion_scene.cjs')(page,1);await seek(4.5);
 assert.equal(await film.locator('[data-motion-id=C]').getAttribute('data-count'),'0.000','fixed-batch C keeps waiting');
 assert.match(await film.locator('.batch-empty').first().textContent(),/等 B 结束/);
 await seek(10);assert.deepEqual(await film.locator('.batch-round-members').allTextContents(),['A + B','B','B','C']);
 await capture(film,'batching-fixed-complete.png');
 await require('./choose_motion_scene.cjs')(page,0);
 for(const width of [320,390,1440]){await page.setViewportSize({width,height:1100});await seek(4.5);await page.waitForTimeout(100);assert.equal(await film.locator('.batch-viewport').evaluate(e=>e.scrollWidth>e.clientWidth),false,'batching must fit narrow screens without horizontal panning');}

 await require('./choose_motion_scene.cjs')(page,1);assert.equal(await film.getAttribute('data-position'),'0.0000');
 await film.locator('[data-motion-action=next]').click();await page.waitForTimeout(500);await film.locator('[data-motion-action=play]').click();const stepPaused=await getX();await page.waitForTimeout(300);assert.equal(await getX(),stepPaused,'pause also freezes single-step motion');
 await page.emulateMedia({reducedMotion:'reduce'});await film.locator('[data-motion-action=reset]').click();await film.locator('[data-motion-action=next]').focus();await page.keyboard.press('Enter');assert.equal(await film.getAttribute('data-position'),'1.0000');await page.waitForTimeout(300);assert.equal(await film.getAttribute('data-position'),'1.0000');await page.emulateMedia({reducedMotion:'no-preference'});
 for(const [url,n,file]of [['foundations/tokens.html',3,'tokens-mapping.png'],['sglang/model-execution/index.html',5.5,'deepseek-mla-handoff.png'],['sglang/kv-memory/index.html',2.5,'kv-write.png'],['sglang/pd-dataflow/index.html',5.5,'pd-references.png']]){await page.goto(base+url);const extra=page.locator('.motion-supplement').filter({has:page.locator('.mechanism-film')});if(await extra.count())await extra.locator(':scope>summary').click();await seek(n);await capture(page.locator('.mechanism-film'),file);}
 await page.setViewportSize({width:390,height:1000});await page.goto(base+'foundations/tokens.html');await seek(2.5);await capture(page.locator('.mechanism-film'),'tokens-mobile.png');
 const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto(base+'foundations/tokens.html');assert.ok((await plain.locator('.foundation-copy').innerText()).includes('token'));await plain.close();
 assert.deepEqual(errors,[]);console.log(`PASS: ${pages.length} pages, ${variants} variants, ${frames} endpoint frames; persistent DOM, real midpoint, pause/resume/reset, backwards seek, condition reset, reduced motion/keyboard, 3 widths × 2 themes, no-JS and evidence links.`);
}finally{await browser.close();await server.close();try{process.kill(owned.pid,0);throw Error('owned browser still running');}catch(e){if(e.code!=='ESRCH')throw e;}}})().catch(e=>{console.error(e);process.exit(1);});
