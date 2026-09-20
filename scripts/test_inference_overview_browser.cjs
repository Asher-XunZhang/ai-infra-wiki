/* Serve .pages-dist. Optional OVERVIEW_TEST_URL, PLAYWRIGHT_MODULE,
 * CHROMIUM_EXECUTABLE, OVERVIEW_SCREENSHOT_DIR. */
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 try{
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`);});
 const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
 const go=slug=>page.goto(base+'sglang/inference-overview/'+slug+'.html');
 const id=x=>page.locator(`[data-id="${x}"]`),root=m=>page.locator(`[data-diagram="${m}"]`);
 const range=async(k,v)=>id(k).evaluate((e,v)=>{e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}));},v);
 const text=m=>root(m).innerText();
 await page.goto(base);await page.getByRole('link',{name:/从推理全景开始/}).click();await page.waitForURL('**/journey.html*');
 await go('journey');assert.match(await text('journey'),/没有任何层的 KV/);
 for(let layer=1;layer<=3;layer++){await id('j-next').click();assert.match(await text('journey'),new RegExp(`当前完成第 ${layer} 层，尚未采样`));assert.equal(await root('journey').locator('.wr-node.active').count(),1);}
 await id('j-next').click();assert.match(await text('journey'),/缓存中只有已处理的 p1…p4，尚无 y1/);
 await id('j-next').click();assert.match(await text('journey'),/输入 y1 → 各层新增它的 KV → 采样 y2/);assert.equal(await id('j-next').isDisabled(),true);
 await page.clock.install();await id('j-play').click();await page.clock.runFor(2250);assert.match(await text('journey'),/当前完成第 1 层/);await id('j-play').click();const paused=await text('journey');await page.clock.runFor(5000);assert.equal(await text('journey'),paused);
 await go('transformer');await id('tok-1').click();assert.equal(await root('transformer').locator('.wr-token.mask').count(),2);await id('t-d').click();assert.equal(await root('transformer').locator('.wr-token.read').count(),5);assert.match(await text('transformer'),/仍运行完整模型/);
 await go('kv-cache');await range('c-reuse',7);await range('c-decode',4);assert.match(await text('cache'),/Prefill 新算 1/);assert.match(await text('cache'),/每层覆盖 12/);assert.match(await text('cache'),/输出 y5 尚无 KV/);for(const l of [1,2,3])assert.match(await text('cache'),new RegExp(`K\\[${l}, p8\\]`));assert.equal(await root('cache').locator('select').count(),0);
 await id('c-reuse').focus();await page.keyboard.press('ArrowLeft');await page.keyboard.press('ArrowLeft');assert.equal(await id('c-reuse').inputValue(),'5');
 await go('deployment');assert.equal(await root('deployment').locator('button,input,select').count(),0);assert.match(await text('deployment'),/合并部署/);assert.match(await text('deployment'),/PD 分离/);assert.match(await text('deployment'),/三个条件要同时满足/);
 await go('scheduling');for(const n of [1,2,4,8]){await id('s-chunk').selectOption(String(n));for(let i=1;i<8/n;i++)await id('s-next').click();assert.match(await text('scheduling'),/8 \/ 8/);assert.equal(await id('s-next').isDisabled(),true);}
 await range('l-q',100);assert.match(await text('latency'),/首字等待 160 ms/);await range('l-d',40);assert.match(await text('latency'),/完成 280 ms/);
 for(const width of [1440,768,390,320])for(const slug of ['journey','transformer','kv-cache','deployment','scheduling']){
   await page.setViewportSize({width,height:1000});await go(slug);
   for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${slug} ${width} ${theme} overflow`);}
   assert.equal(await page.locator('.lab-viewport').count(),0,'no fixed canvas or forced zoom');
   assert.equal(await page.locator('h1').count(),1);
   const bad=await page.locator('a[href^="#"]').evaluateAll(es=>es.filter(e=>!document.getElementById(e.hash.slice(1))).map(e=>e.hash));assert.deepEqual(bad,[]);
   if(process.env.OVERVIEW_SCREENSHOT_DIR&&[390,1440].includes(width)){fs.mkdirSync(process.env.OVERVIEW_SCREENSHOT_DIR,{recursive:true});await page.locator('.lab').first().screenshot({path:path.join(process.env.OVERVIEW_SCREENSHOT_DIR,`${slug}-${width}.png`)});}
 }
 assert.deepEqual(errors,[]);console.log('PASS: 6 overview figures, Prefill/decode and KV boundaries, static deployment, playback/pause, parameter effects, themes, navigation, 320/390/768/1440px');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
