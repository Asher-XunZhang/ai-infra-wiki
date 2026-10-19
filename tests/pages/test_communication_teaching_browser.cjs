const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const M=require('../../pages/foundations/communication-model.js');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/',output=process.env.MOTION_SCREENSHOT_DIR;
(async()=>{
 const server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 const proc=server.process(),pid=proc.pid;console.log(JSON.stringify({session:'communication-teaching',controller:process.pid,browser:pid,profile:proc.spawnargs.find(a=>a.startsWith('--user-data-dir='))}));
 let browser;try{
 browser=await chromium.connect(server.wsEndpoint());const page=await browser.newPage(),errors=[],overflow=[];page.on('pageerror',e=>errors.push(e.message));
 if(output)fs.mkdirSync(output,{recursive:true});
 await page.goto(base+'foundations/communication.html');
 const film=page.locator('.mechanism-film'),select=film.locator('[aria-label="连续演示条件"]');
 const seek=p=>film.locator('.motion-scrub input').evaluate((e,p)=>{e.value=p;e.dispatchEvent(new Event('input',{bubbles:true}));},p);
 assert.equal(await page.locator('.foundation-lab').count(),0,'old duplicate lab removed');assert.equal(await film.locator('[data-lesson-family]').count(),17);
 for(const [v,s]of M.scenes().entries()){
  await film.locator(`[data-lesson-family="${s.op}"]`).click();if(await select.locator('option').count()>1)await select.selectOption(String(v));
  for(const width of [320,390,768,1440]){
   await page.setViewportSize({width,height:1100});await page.waitForTimeout(25);
   for(const p of [0,.5,1,1.5,2,2.5,3,3.5,4]){
    await seek(p);
    const bad=await film.locator('svg').evaluate(svg=>{const b=svg.getBoundingClientRect();return [...svg.querySelectorAll('text')].filter(e=>{for(let a=e;a&&a!==svg;a=a.parentElement)if(Number(a.getAttribute('opacity')??1)<.02)return false;const r=e.getBoundingClientRect();return e.textContent&&(r.left<b.left-2||r.right>b.right+2||r.top<b.top-2||r.bottom>b.bottom+2);}).map(e=>e.textContent);});
    if(bad.length)overflow.push({v,width,p,bad});
   }
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'page overflow');
   if(output&&[29,30,31,32].includes(v)&&[390,1440].includes(width)){await seek(2.5);await film.screenshot({style:'.site-header{visibility:hidden}',path:path.join(output,`${s.op}-${v}-${width}.png`)});}
  }
 }
 fs.writeFileSync(output?path.join(output,'overflow.json'):'/tmp/comm-overflow.json',JSON.stringify(overflow,null,2));assert.deepEqual(overflow,[],'labels fit all frames');
 // Independent peers must actually move at the same time, not only share final outputs.
 for(const width of [390,1440]){
  await page.setViewportSize({width,height:1100});
  for(const op of ['Broadcast','Scatter','Gather','AllGather','AllToAll','Neighbor']){
   await film.locator(`[data-lesson-family="${op}"]`).click();
   assert.equal(await film.locator('svg').getAttribute('data-teaching-kind'),'communication-direct');
   assert.doesNotMatch(await film.locator('svg').textContent(),/路由矩阵|独立交接|每行一对/);
   const snapshot=()=>film.locator('[data-motion-id^="comm-copy-"][data-in-flight="true"]').evaluateAll(nodes=>nodes.map(e=>({id:e.dataset.motionId,x:Number(e.dataset.x),y:Number(e.dataset.y),arrived:e.dataset.arrived,src:e.dataset.source,dst:e.dataset.target,rect:e.getBoundingClientRect().toJSON()})));
   await seek(1.8);const before=await snapshot();await seek(2.2);const after=await snapshot();
   assert.ok(before.length>=2,op+' has multiple transfers in flight');
   assert.equal(before.length,after.length);
   for(const a of after){const b=before.find(b=>b.id===a.id);assert.ok(a.x!==b.x||a.y!==b.y,op+' independent peer was stalled');assert.equal(a.arrived,'false');assert.notEqual(a.src,a.dst,'local copy is not network traffic');}
   for(let i=0;i<after.length;i++)for(let j=i+1;j<after.length;j++){const a=after[i].rect,b=after[j].rect;assert.ok(a.right<=b.left+1||b.right<=a.left+1||a.bottom<=b.top+1||b.bottom<=a.top+1,op+' concurrent payloads overlap');}
  }
  for(const op of ['Reduce','AllReduce','ReduceScatter','Scan','Exscan']){
   await film.locator(`[data-lesson-family="${op}"]`).click();await seek(1.9);
   assert.equal(await film.locator('[data-motion-id^="comm-result-"][opacity="1.000"]').count(),0,'result cannot precede its contributions');
   await seek(2.4);const actors=film.locator('[data-motion-id^="comm-result-"][data-in-flight="true"]'),before=await actors.evaluateAll(es=>es.map(e=>e.getAttribute('transform')));assert.ok(before.length>=2);
   await seek(2.6);const after=await actors.evaluateAll(es=>es.map(e=>e.getAttribute('transform')));assert.equal(before.length,after.length);before.forEach((v,i)=>assert.notEqual(v,after[i],op+' result delivery was serialized'));
   assert.equal(await film.locator('[data-motion-id^="comm-result-"][data-arrived="true"]').count(),0);
  }
  for(const op of ['SendRecv','Put','Get']){
   await film.locator(`[data-lesson-family="${op}"]`).click();await seek(1.2);const copies=film.locator('[data-motion-id^="comm-copy-"][data-in-flight="true"]');assert.equal(await copies.count(),2);
   const first=await copies.evaluateAll(es=>es.map(e=>e.getAttribute('transform')));await seek(1.4);const second=await copies.evaluateAll(es=>es.map(e=>e.getAttribute('transform')));first.forEach((v,i)=>assert.notEqual(v,second[i],'message elements advance together on narrow and wide screens'));
  }
 }
 // Whole matrix fragments retain coordinates and cross concurrently in separate curves.
 for(const width of [390,1440])for(const v of [29,30,31,32]){await page.setViewportSize({width,height:1100});await require('./choose_motion_scene.cjs')(page,v);const originals=await film.locator('[data-motion-id^="matrix-input-"]').evaluateAll(es=>es.map(e=>e.getAttribute('transform')));
  for(const t of [1.8,2,2.2]){await seek(t);const rects=await film.locator('[data-in-flight=true]').evaluateAll(es=>es.map(e=>e.getBoundingClientRect().toJSON()));for(let i=0;i<rects.length;i++)for(let j=i+1;j<rects.length;j++){const a=rects[i],b=rects[j];assert.ok(a.right<=b.left+1||b.right<=a.left+1||a.bottom<=b.top+1||b.bottom<=a.top+1,'matrix copies remain distinguishable');}}
  await seek(4);assert.deepEqual(await film.locator('[data-motion-id^="matrix-input-"]').evaluateAll(es=>es.map(e=>e.getAttribute('transform'))),originals);assert.deepEqual(await film.locator('[data-motion-id^="matrix-result-"] text').allTextContents(),M.scenes()[v].data.output.flat().map(String));
 }
 // Stable identity, original buffer preservation, backward seek and reload.
 await film.locator('[data-lesson-family="AllToAll"]').click();await seek(0);const original=await film.locator('[data-motion-id="comm-input-0-0"]').getAttribute('transform');
 await seek(1);assert.equal(await film.locator('[data-motion-id^="comm-route-"]').count(),0,'no intermediate routing view');const actor=film.locator('[data-motion-id="comm-copy-0-1-1-0"]');await actor.evaluate(e=>e.dataset.identity='persist');const start=await actor.getAttribute('transform');await seek(2);assert.notEqual(await actor.getAttribute('transform'),start);assert.equal(await actor.getAttribute('data-arrived'),'false');assert.equal(await actor.getAttribute('data-identity'),'persist');assert.equal(await film.locator('[data-motion-id="comm-input-0-0"]').getAttribute('transform'),original);
 await seek(3);assert.equal(await actor.getAttribute('data-arrived'),'true');await seek(1);assert.equal(await actor.getAttribute('transform'),start);await seek(2.5);await page.reload();assert.equal(await film.getAttribute('data-position'),'2.5000');
 await film.locator('[data-lesson-family="SendRecv"]').click();await select.selectOption('19');await seek(4);assert.match(await film.locator('svg').textContent(),/接收尚未完成/);assert.equal(await film.locator('[data-arrived="true"]').count(),0);
 await film.locator('[data-lesson-family="Barrier"]').click();await select.selectOption('26');await seek(4);assert.match(await film.locator('svg').textContent(),/屏障不会成功返回/);assert.match(await film.locator('svg').textContent(),/等待 Rank 2 进入/);assert.doesNotMatch(await film.locator('svg').textContent(),/全组进入的最早位置/);assert.equal(await film.locator('[data-released=true]').count(),0);
 await select.selectOption('25');await seek(2.5);assert.equal(await film.locator('[data-released=true]').count(),0);await seek(3);assert.equal(await film.locator('[data-released=true]').count(),3,'all members entering releases the barrier condition');
 for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await film.locator('[data-lesson-family="AllReduce"]').click();await seek(3);if(output)await film.screenshot({style:'.site-header{visibility:hidden}',path:path.join(output,`AllReduce-${theme}.png`)});}
 await page.emulateMedia({reducedMotion:'reduce'});await film.locator('[data-motion-action=reset]').click();await film.locator('[data-motion-action=next]').click();assert.equal(await film.getAttribute('data-position'),'1.0000');
 await page.emulateMedia({reducedMotion:'no-preference'});await film.locator('[data-motion-action=reset]').click();await film.locator('[data-motion-action=play]').click();await page.waitForTimeout(350);await film.locator('[data-motion-action=play]').click();assert.ok(Number(await film.getAttribute('data-position'))>0,'actual timed playback');
 // Shared controls still work for already separated mechanism selectors.
 await page.goto(base+'foundations/parallel.html');await page.locator('[data-parallel-family=PP]').click();assert.match(await page.locator('.mechanism-film h2').textContent(),/PP/);
 await page.goto(base+'foundations/batching.html');await page.locator('[data-lesson-family=dynamic]').click();assert.ok(await page.locator('.mechanism-film input[type=range]').count());
 const noJS=await browser.newContext({javaScriptEnabled:false});try{const plain=await noJS.newPage();await plain.goto(base+'foundations/communication.html');assert.match(await plain.locator('body').textContent(),/ReduceScatter.*111/s);assert.ok(await plain.locator('#why-rma').count());}finally{await noJS.close();}
 assert.deepEqual(errors,[]);console.log('PASS 33 scenarios × 4 widths × 9 times; concurrent peer movement and non-overlap, causal gates, ownership, seek/reload, negative cases, playback, themes, reduced motion, no-JS and shared selectors.');
 }finally{await browser?.close();await server.close();try{process.kill(pid,0);throw Error('Owned browser still alive: '+pid);}catch(e){if(e.code!=='ESRCH')throw e;}console.log('CLEANED communication-teaching browser PID '+pid);}
})().catch(e=>{console.error(e);process.exitCode=1;});
