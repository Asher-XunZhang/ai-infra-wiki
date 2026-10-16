const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const {scenarios}=require('../../pages/motion/dynamic-batching.js');
const output=process.env.MOTION_SCREENSHOT_DIR;
(async()=>{const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});try{
 const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto((process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/')+'foundations/batching.html');
 const film=page.locator('.mechanism-film'),choice=film.locator('select[aria-label="连续演示条件"]'),svg=film.locator('svg'),range=film.locator('.motion-scrub input');
 const seek=n=>range.evaluate((e,n)=>{e.value=n;e.dispatchEvent(new Event('input',{bubbles:true}));},n);
 assert.equal(await choice.inputValue(),'2');
 for(let mode=0;mode<3;mode++){
  await require('./choose_motion_scene.cjs')(page,Number(String(mode+2)));const frames=scenarios[mode].frames;
  const A=film.locator('[data-motion-id=A]');await A.evaluate(e=>window.actorA=e);
  for(let i=0;i<frames.length;i++){
   await seek(i);assert.equal(await svg.getAttribute('data-clock'),frames[i].time.toFixed(3));assert.equal(Number(await svg.getAttribute('data-dispatched')),frames[i].batches.length);
   assert.equal(await A.evaluate(e=>e===window.actorA),true);
   if(i&&frames[i].event==='dispatch'){
    const y1=Number(await A.getAttribute('data-y'));await seek(i-1);const y0=Number(await A.getAttribute('data-y'));await seek(i-.5);const ym=Number(await A.getAttribute('data-y'));
    if(y0!==y1)assert.ok(ym>y0&&ym<y1,'request really moves between queue and batch');
    assert.equal(Number(await svg.getAttribute('data-dispatched')),frames[i-1].batches.length,'in-flight batch must not be committed');
   }
  }
  await seek(frames.length-1);assert.match(await svg.textContent(),/等了/);await seek(0);assert.equal(await svg.getAttribute('data-dispatched'),'0');
 }
 await require('./choose_motion_scene.cjs')(page,Number('2'));await film.scrollIntoViewIfNeeded();await film.locator('[data-motion-action=play]').click();await page.waitForTimeout(500);await film.locator('[data-motion-action=play]').click();const paused=await film.getAttribute('data-position');await page.waitForTimeout(250);assert.equal(await film.getAttribute('data-position'),paused);assert.ok(Number(paused)>0&&Number(paused)<1);
 // Switching renderer must clear the other renderer's layout classes and reset time.
 for(const [label,mode] of [['② 固定批次','1'],['② 连续批处理','0'],['① 动态组批','2']]){await film.getByRole('button',{name:label,exact:true}).click();assert.equal(await choice.inputValue(),mode);assert.equal(await film.getAttribute('data-position'),'0.0000');assert.equal(await film.locator('.batch-viewport').count(),mode==='2'?0:1);}
 if(output)fs.mkdirSync(output,{recursive:true});
 for(const width of [320,390,768,1440]){await page.setViewportSize({width,height:1100});for(const theme of ['light','dark']){
  await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
  for(const mode of [0,1,2,3,4]){await require('./choose_motion_scene.cjs')(page,Number(String(mode)));const last=Number(await range.getAttribute('max'));await seek(mode<2?3.5:last);await page.waitForTimeout(60);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
   assert.equal(await film.locator('.motion-viewport').evaluate(e=>e.scrollWidth>e.clientWidth),false);
   const clipped=await svg.locator('text').evaluateAll(es=>es.filter(e=>e.getAttribute('opacity')!=='0.000'&&e.closest('[opacity="0.000"]')===null).filter(e=>{const b=e.getBBox(),v=e.ownerSVGElement.viewBox.baseVal;return b.x<0||b.x+b.width>v.width+1;}).map(e=>e.textContent));assert.deepEqual(clipped,[],`${width}/${theme}/${mode} text outside SVG`);
   if(output&&[390,1440].includes(width)&&((theme==='light'&&[0,1,2,3].includes(mode))||(theme==='dark'&&mode===4)))await film.screenshot({path:path.join(output,`batch-${mode}-${width}-${theme}.png`)});
  }
 }}
 await require('./choose_motion_scene.cjs')(page,Number('1'));await seek(7);assert.match(await svg.textContent(),/C 现在可加入下一批/);assert.doesNotMatch(await svg.textContent(),/空位留着，等 B 结束/);
 await require('./choose_motion_scene.cjs')(page,Number('2'));await seek(1.5);assert.equal(await svg.getAttribute('data-clock'),'1.000');assert.equal(await film.locator('[data-queue-count]').getAttribute('data-queue-count'),'1');
 if(output)await film.screenshot({path:path.join(output,'dynamic-wait-midpoint.png')});
 await seek(6.5);if(output)await film.screenshot({path:path.join(output,'dynamic-dispatch-midpoint.png')});
 await page.emulateMedia({reducedMotion:'reduce'});await require('./choose_motion_scene.cjs')(page,Number('2'));await film.locator('[data-motion-action=next]').focus();await page.keyboard.press('Enter');assert.equal(await film.getAttribute('data-position'),'1.0000');
 const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto((process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/')+'foundations/batching.html');await plain.locator('#read-assemble summary').click();assert.match(await plain.locator('.foundation-copy').innerText(),/AB 在 t=3/);await plain.close();assert.deepEqual(errors,[]);
 console.log('PASS: dynamic snapshots, in-flight identity, reversible geometry, pause, renderer switching, 5 modes × 4 widths × 2 themes, keyboard/reduced-motion and no-JS.');
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exit(1);});
