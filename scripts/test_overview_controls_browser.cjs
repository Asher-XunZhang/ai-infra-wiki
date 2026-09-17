/* Verify actual viewport interaction without Playwright auto-scrolling controls. */
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 try{
  const page=await browser.newPage({hasTouch:true}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
  const cases=[['journey',0,'#journey-next','#journey-output'],['transformer',0,'#phase-decode','#phase-summary'],['kv-cache',0,'#reuse-count','#cache-compute'],['deployment',0,'#deployment-next','#deployment-step'],['scheduling',0,'#schedule-next','#schedule-output'],['scheduling',1,'#latency-q','#ttft-value']];
  async function mouseControl(locator){const r=await locator.boundingBox();assert(r);await page.mouse.click(r.x+r.width/2,r.y+r.height/2);}
  for(const width of [1440,768,390,320])for(const [slug,n,control,result] of cases){
   await page.setViewportSize({width,height:width<500?740:1000});await page.goto(base+'sglang/inference-overview/'+slug+'.html');
   await page.evaluate(()=>document.documentElement.style.scrollBehavior='auto');
   if(slug==='deployment')await page.selectOption('#deployment-mode','split');
   const frame=page.locator('.lab-workbench').nth(n),viewport=frame.locator('.lab-viewport');
   await frame.evaluate(el=>window.scrollBy(0,el.getBoundingClientRect().top-document.querySelector('.site-header').getBoundingClientRect().height-5));
   const geometry=await frame.evaluate(el=>{const r=el.getBoundingClientRect(),v=el.querySelector('.lab-viewport').getBoundingClientRect(),d=el.querySelector('.lab-dock').getBoundingClientRect();return {top:r.top,bottom:r.bottom,viewportTop:v.top,viewportHeight:v.height,dockBottom:d.bottom,windowHeight:innerHeight,headerBottom:document.querySelector('.site-header').getBoundingClientRect().bottom};});
   assert(geometry.top>=geometry.headerBottom-1,`${slug}: covered by header`);
   assert(geometry.bottom<=geometry.windowHeight,`${slug} ${width}: frame does not fit screen`);
   assert(geometry.viewportHeight>180,`${slug}: too little room for diagram`);
   assert(geometry.dockBottom<geometry.viewportTop,`${slug}: controls overlap drawing`);
   const scroll=await page.evaluate(()=>scrollY),before=await page.locator(result).innerText();
   const hit=await page.locator(control).evaluate(el=>{const r=el.getBoundingClientRect();return el.contains(document.elementFromPoint(r.x+r.width/2,r.y+r.height/2));});assert(hit,`${slug}: inaccessible control`);
   await mouseControl(page.locator(control));if(control==='#reuse-count'||control==='#latency-q')await page.keyboard.press('ArrowRight');
   assert.notEqual(await page.locator(result).innerText(),before);
   assert(Math.abs(await page.evaluate(()=>scrollY)-scroll)<3,`${slug}: operating control scrolls the page`);
   const fitted=Number(await viewport.getAttribute('data-scale'));
   await mouseControl(frame.locator('[data-camera=in]'));assert(Number(await viewport.getAttribute('data-scale'))>fitted);
   await mouseControl(frame.locator('[data-camera=reset]'));assert.equal(Number(await viewport.getAttribute('data-scale')),1);
   const box=await viewport.boundingBox(),oldX=Number(await viewport.getAttribute('data-x'));
   await page.mouse.move(box.x+box.width*.7,box.y+box.height*.6);await page.mouse.down();await page.mouse.move(box.x+box.width*.4,box.y+box.height*.45,{steps:6});await page.mouse.up();
   assert.notEqual(Number(await viewport.getAttribute('data-x')),oldX,`${slug}: dragging does not pan`);
   assert(Math.abs(await page.evaluate(()=>scrollY)-scroll)<3,`${slug}: dragging scrolls the page`);
   await viewport.focus();await page.keyboard.press('+');assert(Number(await viewport.getAttribute('data-scale'))>1);
   await page.keyboard.press('0');
   const fitBounds=await viewport.evaluate(el=>{const v=el.getBoundingClientRect(),s=el.querySelector('.lab-scene').getBoundingClientRect();return {inside:s.left>=v.left-1&&s.right<=v.right+1&&s.top>=v.top-1&&s.bottom<=v.bottom+1};});assert(fitBounds.inside,`${slug}: fit clips content`);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
   if(process.env.OVERVIEW_SCREENSHOT_DIR&&['kv-cache','deployment'].includes(slug)&&[1440,390].includes(width)){fs.mkdirSync(process.env.OVERVIEW_SCREENSHOT_DIR,{recursive:true});await page.screenshot({path:path.join(process.env.OVERVIEW_SCREENSHOT_DIR,`${slug}-canvas-${width}.png`)});}
  }
  // Pinch and modifier-wheel zoom inside the canvas; ordinary page scroll remains available outside.
  await page.goto(base+'sglang/inference-overview/kv-cache.html');await page.evaluate(()=>document.documentElement.style.scrollBehavior='auto');
  const viewport=page.locator('.lab-viewport');await viewport.scrollIntoViewIfNeeded();
  let scale=Number(await viewport.getAttribute('data-scale'));const box=await viewport.boundingBox();
  await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.keyboard.down('Control');await page.mouse.wheel(0,-80);await page.keyboard.up('Control');
  await page.waitForFunction(s=>Number(document.querySelector('.lab-viewport').dataset.scale)>s,scale);
  await page.locator('[data-camera=fit]').click();scale=Number(await viewport.getAttribute('data-scale'));
  const cdp=await page.context().newCDPSession(page),cx=box.x+box.width/2,cy=box.y+box.height/2;
  await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:cx-35,y:cy,id:1},{x:cx+35,y:cy,id:2}]});
  await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:cx-65,y:cy,id:1},{x:cx+65,y:cy,id:2}]});
  await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
  assert(Number(await viewport.getAttribute('data-scale'))>scale,'pinch failed');await cdp.detach();
  // Original PD gate listeners survive reparenting into the fixed controls.
  await page.goto(base+'sglang/inference-overview/deployment.html');await page.selectOption('#deployment-mode','split');
  await page.locator('#handoff-controls summary').click();await page.locator('#gate-meta').uncheck();
  for(let i=0;i<4;i++)await page.locator('#deployment-next').click();assert.equal(await page.locator('[data-admission=waiting]').count(),1);
  await page.locator('#gate-meta').focus();await page.keyboard.press('Escape');assert.equal(await page.locator('#handoff-controls').evaluate(el=>el.open),false);
  assert.equal(await page.locator('#handoff-controls summary').evaluate(el=>el===document.activeElement),true);
  await page.selectOption('#deployment-mode','unified');assert.equal(await page.locator('#handoff-controls').isVisible(),false);
  assert.deepEqual(errors,[]);console.log('PASS: 6 canvases × 4 widths; visible fixed controls, pan, zoom, fit, reset, keyboard, wheel, pinch, no page jumps, PD gates');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
