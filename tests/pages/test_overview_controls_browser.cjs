/* Reading view regression: real-size text and directly reachable controls. */
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 try{
 const page=await browser.newPage({hasTouch:true}),base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
 for(const width of [1440,768,390,320])for(const slug of ['journey','transformer','kv-cache','deployment','scheduling']){
   await page.setViewportSize({width,height:850});await page.goto(base+'sglang/inference-overview/'+slug+'.html');
   const bad=await page.locator('.lesson-diagram').evaluateAll(roots=>roots.flatMap(root=>{
     const box=root.getBoundingClientRect(),bad=[];
     for(const el of root.querySelectorAll('*')){if(!el.getClientRects().length||el.tagName==='OPTION')continue;const r=el.getBoundingClientRect();if(r.width&&(r.left<box.left-2||r.right>box.right+2))bad.push(el.tagName+':'+el.textContent.slice(0,30));if(el.tagName==='text'&&parseFloat(getComputedStyle(el).fontSize)<11)bad.push('small SVG label');}return bad;
   }));assert.deepEqual(bad,[],`${slug} ${width}: content must reflow, not scale down`);
   const button=page.locator('.lesson-diagram button:not(:disabled)').first();if(await button.count()){await button.scrollIntoViewIfNeeded();assert.equal(await button.evaluate(e=>{const r=e.getBoundingClientRect();return e.contains(document.elementFromPoint(r.x+r.width/2,r.y+r.height/2));}),true);await button.focus();await page.keyboard.press('Enter');}
 }
 console.log('PASS: reading view fits 320/390/768/1440px, SVG labels remain readable, controls are reachable without canvas gestures');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
