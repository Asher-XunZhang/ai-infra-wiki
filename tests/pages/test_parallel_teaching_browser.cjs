const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict'),fs=require('node:fs');
const dir=process.env.MOTION_SCREENSHOT_DIR;if(dir)fs.mkdirSync(dir,{recursive:true});
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:18765/';
(async()=>{console.log('controller PID',process.pid);const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE});try{
const page=await browser.newPage(),errors=[],overflow=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto(base+'foundations/parallel.html');const film=page.locator('.mechanism-film'),sel=film.locator('select[aria-label="连续演示条件"]');
const seek=async p=>page.locator('.motion-scrub input').evaluate((e,p)=>{e.value=p;e.dispatchEvent(new Event('input',{bubbles:true}));},p);
const families=['TP','TP','PP','PP','DP','EP','EP','SP','PCP','PCP','DCP','DCP'];
for(let v=0;v<12;v++){const family=families[v];await page.locator(`[data-parallel-family=${family}]`).click();if(!['DP','SP'].includes(family))await sel.selectOption(String(v));assert.equal(await film.getAttribute('data-position'),'0.0000');
assert.equal(await page.locator('.parallel-strategies [aria-pressed=true]').count(),1);assert.equal(await sel.locator('option').count(),families.filter(f=>f===family).length);
const max=Number(await page.locator('.motion-scrub input').getAttribute('max'));
for(const width of [320,390,768,1440]){await page.setViewportSize({width,height:1000});await page.waitForTimeout(40);for(let p=0;p<=max;p+=.5){await seek(p);const bad=await film.locator('svg').evaluate(svg=>{const b=svg.getBoundingClientRect();return [...svg.querySelectorAll('text')].filter(e=>{for(let a=e;a&&a!==svg;a=a.parentElement)if(Number(a.getAttribute('opacity')??1)<.02)return false;return !!e.textContent;}).filter(e=>{const r=e.getBoundingClientRect();return r.left<b.left-2||r.right>b.right+2||r.top<b.top-2||r.bottom>b.bottom+2;}).map(e=>e.textContent);});if(bad.length)overflow.push({v,width,p,bad});}
assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
if(dir&&[390,1440].includes(width))for(const step of v===0||v===1?[0,2,3.5,5.5,max]:[0,max]){await seek(step);await film.screenshot({path:`${dir}/v${v}-${width}-p${step}.png`});}
}
}
// Switching mechanisms resets playback; URL restores both semantic levels.
await page.locator('[data-parallel-family=TP]').click();await seek(0);
const actor=page.locator('[data-motion-id="tp-x0-0-0"]');const x0=await actor.getAttribute('transform');await seek(2.5);const xm=await actor.getAttribute('transform');await seek(3);const x1=await actor.getAttribute('transform');assert.notEqual(x0,xm);assert.notEqual(xm,x1);
await seek(4);assert.deepEqual(await page.locator('[data-motion-id^="tp-local0-"] text').allTextContents(),['2','0','6','0']);
assert.deepEqual(await page.locator('[data-motion-id^="tp-local1-"] text').allTextContents(),['2','6','4','12']);await seek(6);assert.deepEqual(await page.locator('[data-motion-id^="tp-final0-"] text').allTextContents(),['4','6','10','12']);
await sel.selectOption('1');await seek(4);assert.deepEqual(await page.locator('[data-motion-id^="tp-local0-"] text').allTextContents(),['4','10']);assert.deepEqual(await page.locator('[data-motion-id^="tp-local1-"] text').allTextContents(),['6','12']);
await page.locator('[data-parallel-family=DCP]').focus();await page.keyboard.press('Enter');assert.equal(await film.getAttribute('data-position'),'0.0000');await sel.selectOption('11');await seek(2.5);await page.reload();await film.waitFor();assert.equal(await page.locator('[data-parallel-family=DCP]').getAttribute('aria-pressed'),'true');assert.equal(await sel.inputValue(),'11');assert.equal(await film.getAttribute('data-position'),'2.5000');
await page.locator('[data-parallel-family=DP]').click();await seek(2);const dp=page.locator('[data-motion-id=dp-R1]');const before=await dp.getAttribute('transform');await film.locator('[data-motion-action=play]').click();await page.waitForTimeout(400);await film.locator('[data-motion-action=play]').click();const paused=await dp.getAttribute('transform');assert.notEqual(before,paused);await page.waitForTimeout(150);assert.equal(await dp.getAttribute('transform'),paused);await film.locator('[data-motion-action=reset]').click();assert.equal(await film.getAttribute('data-position'),'0.0000');
await page.emulateMedia({reducedMotion:'reduce'});await film.locator('[data-motion-action=next]').click();assert.equal(await film.getAttribute('data-position'),'1.0000');await page.emulateMedia({reducedMotion:'no-preference'});
await page.locator('[data-parallel-family=PP]').click();await seek(3.2);assert.equal(Number(await page.locator('.mechanism-film .teach-value').nth(4).getAttribute('width')),0);await page.getByLabel('PP 部署位置').selectOption('cross-host');assert.match(await film.locator('svg').textContent(),/跨主机/);await seek(3.5);assert.ok(Number(await page.locator('.mechanism-film .teach-value').nth(4).getAttribute('width'))>0);
for(const width of [390,1440]){await page.setViewportSize({width,height:1000});for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.locator('[data-parallel-family=TP]').click();await seek(5.5);if(dir)await film.locator('svg').screenshot({path:`${dir}/tp-detail-${width}-${theme}.png`});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);}}
await page.evaluate(()=>document.documentElement.dataset.theme='light');
if(dir)for(const id of ['parallel-overview','read-pcp','read-dcp'])await page.locator('#'+id).screenshot({path:`${dir}/${id}.png`});
const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto(base+'foundations/parallel.html');assert.equal(await plain.locator('.parallel-reading-section').count(),8);await plain.locator('#read-pcp summary').click();assert.match(await plain.locator('#read-pcp').innerText(),/padding/);await plain.locator('#parallel-glossary summary').click();assert.match(await plain.locator('#parallel-glossary').innerText(),/Log-Sum-Exp/);await plain.close();
if(dir)fs.writeFileSync(dir+'/findings.json',JSON.stringify({errors,overflow},null,2));assert.deepEqual(errors,[]);assert.deepEqual(overflow,[]);console.log('PASS 12 parallel scenarios; 4 widths and half-steps; 2 themes; TP arithmetic/geometry; PP activation gating; keyboard, URL restore, pause/reset, reduced motion and no-JS reading.');
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1});
