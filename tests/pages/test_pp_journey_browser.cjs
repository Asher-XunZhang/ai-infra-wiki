const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{execFileSync}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/',keep=!!process.env.PP_JOURNEY_QA_DIR,dir=process.env.PP_JOURNEY_QA_DIR||fs.mkdtempSync(path.join(os.tmpdir(),'pp-journey-qa-'));
fs.mkdirSync(dir,{recursive:true});
(async()=>{let server,meta;try{
 server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 const pid=server.process().pid,command=execFileSync('ps',['-p',String(pid),'-o','command='],{encoding:'utf8'});
 meta={session:'pp-journey-qa',controller:process.pid,pid,profile:command.match(/--user-data-dir=([^ ]+)/)?.[1]};console.log(meta);
 const browser=await chromium.connect(server.wsEndpoint()),page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400&&r.url().startsWith(base))errors.push(r.status()+' '+r.url());});
 const go=async file=>{await page.goto(base+'sglang/pd-prefill-pp-loop/'+file);await page.waitForFunction(()=>!!window.PP_JOURNEY?.model);};
 const state=()=>page.evaluate(()=>({...PP_JOURNEY.state}));
 const seek=t=>page.locator('[data-control=time]').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));},t);
 await go('quick.html');await page.evaluate(()=>document.fonts.ready);
 assert((await page.locator('.pp-ranks').boundingBox()).y<800,'default viewport includes main visual');
 assert.equal(await page.locator('.animation-speed').count(),0,'no competing legacy speed control');
 await page.clock.install();await page.clock.pauseAt(new Date());
 await seek(29.3);const shot=await page.locator('.pp-scene').innerHTML();
 const pulse=await page.locator('[data-transfer="0:5:kv"] [data-pulse]').getAttribute('stroke-dashoffset');
 await page.clock.runFor(500);assert.equal(await page.locator('[data-transfer="0:5:kv"] [data-pulse]').getAttribute('stroke-dashoffset'),pulse,'paused pulse freezes');
 await seek(29.5);assert.notEqual(await page.locator('[data-transfer="0:5:kv"] [data-pulse]').getAttribute('stroke-dashoffset'),pulse);
 await seek(90);await seek(29.3);assert.equal(await page.locator('.pp-scene').innerHTML(),shot,'reverse seek is deterministic');
 for(const level of ['process','overview']){await page.selectOption('[data-control=level]',level);assert.equal((await state()).time,29.3);assert.equal((await state()).playing,false);}

 assert.deepEqual(await page.locator('[data-control=level] option').allTextContents(),['总览','执行过程']);
 await page.selectOption('[data-control=level]','process');await page.selectOption('[data-control=batch]','3');await seek(18.2);
 for(const rank of [0,1,2]){assert.equal(await page.locator(`[data-rail-rank="${rank}"][data-lane="cpu"]`).count(),1);assert.equal(await page.locator(`[data-rail-rank="${rank}"][data-lane="gpu"]`).count(),1);}
 assert.match(await page.locator('.pp-relations').innerText(),/PP2.*PP0/s);
 assert.equal(await page.locator('[data-cross-pulse]').count(),0,'unstarted payload must not pulse');
 const beforeInspect=await state();
 await page.locator('.pp-rails [data-inspect="0:5:recv_out:wait"]').click();
 assert.equal((await state()).batch,beforeInspect.batch);assert.equal((await state()).time,beforeInspect.time);assert.equal((await state()).playing,false);
 assert.match(await page.locator('.pp-inline-source').innerText(),/M1/);assert.match(await page.locator('.pp-inline-source').innerText(),/未满足/);assert.match(await page.locator('.pp-operation-status').innerText(),/等待中/);assert.match(await page.locator('.pp-source-grid').innerText(),/条件满足后才执行/);
 await seek(27.6);assert.equal(await page.locator('[data-cross-pulse="2:3:out_message"]').count(),1);
 const crossPhase=await page.locator('[data-cross-pulse]').getAttribute('stroke-dashoffset');await page.clock.runFor(200);assert.equal(await page.locator('[data-cross-pulse]').getAttribute('stroke-dashoffset'),crossPhase);
 await seek(27.65);assert.notEqual(await page.locator('[data-cross-pulse]').getAttribute('stroke-dashoffset'),crossPhase);
 await seek(27.73);assert.equal(await page.locator('[data-gate="2:3:out_message"]').getAttribute('data-ready'),'true');
 await seek(18.2);assert.equal(await page.locator('[data-gate="2:3:out_message"]').getAttribute('data-ready'),'false');
 await page.locator('[data-action=play]').click();const inspectTime=(await state()).time;
 await page.locator('[data-action=close-source]').click();assert((await state()).playing);assert.equal((await state()).time,inspectTime);
 await page.locator('.pp-rails [data-inspect="0:5:recv_req"]').click();assert((await state()).playing);assert.equal((await state()).time,inspectTime);
 await page.locator('[data-action=operations]').click();await page.selectOption('[data-control=operation]','0:5:boot_poll');assert.equal((await state()).time,inspectTime);assert((await state()).playing);
 const pickerHandle=await page.locator('[data-control=operation]').elementHandle();await page.clock.runFor(200);
 assert(await pickerHandle.evaluate(e=>e.isConnected),'native picker survives advancing frames');assert((await state()).time>inspectTime);
 await page.locator('[data-action=play]').click();await page.selectOption('[data-control=level]','overview');
 await page.selectOption('[data-control=batch]','4');assert.equal((await state()).batch,4);
 await page.locator('#full-timeline-view>summary').click();assert.equal(await page.locator('#batch').inputValue(),'4');
 await page.selectOption('#batch','2');assert.equal(await page.locator('[data-control=batch]').inputValue(),'2');
 await page.selectOption('[data-control=batch]','4');await seek(31.25);
 // Cross-page and reload round trip keep the scene, batch, focus and fractional clock.
 let link=await page.locator('.pp-local-nav a[href*="index.html"]').first().getAttribute('href');await page.goto(link);await page.waitForFunction(()=>!!PP_JOURNEY.model);await page.waitForTimeout(50);
 assert.equal((await state()).batch,4);assert.equal((await state()).time,31.25);assert.equal((await state()).event,'0:6:gpu','iframe initialization must not overwrite the deep-linked event');
 await page.locator('#full-dependency-view>summary').click();
 const frame=page.frameLocator('#timeline iframe');await frame.locator('#pp-loop-detail').waitFor();
 await seek(32.5);await page.waitForFunction(()=>document.querySelector('#timeline iframe')!=null);await page.waitForTimeout(50);
 assert.equal(await frame.locator('#pp-loop-detail').getAttribute('data-time'),'32.500');
 await page.locator('#scenario-options>summary').click();await page.selectOption('#scenario-select','slow-kv');await page.waitForFunction(()=>PP_JOURNEY.scenario==='slow-kv');
 assert.equal((await state()).time,0);assert.equal((await state()).playing,false);
 assert.match(await page.locator('.pp-dependency-panel').last().innerText(),/PD KV 传输偏慢/);
 await seek(23.125);link=await page.locator('.pp-local-nav a[href*="notes.html"]').first().getAttribute('href');await page.goto(link);
 assert.match(await page.locator('[data-pp-reading-context]').innerText(),/M4.*23.13/);
 link=await page.locator('.pp-local-nav a[href*="quick.html"]').first().getAttribute('href');await page.goto(link);await page.waitForFunction(()=>PP_JOURNEY.scenario==='slow-kv'&&PP_JOURNEY.model);
 assert.equal((await state()).time,23.125);assert.equal((await state()).batch,4);
 await page.reload();await page.waitForFunction(()=>!!PP_JOURNEY.model);assert.equal((await state()).time,23.125);
 // Playback actually traverses the scene rather than jumping between a fixed beat array.
 await go('quick.html');await page.selectOption('[data-control=speed]','4');await page.locator('[data-action=play]').click();await page.clock.runFor(500);
 assert((await state()).time>0);await page.selectOption('[data-control=level]','process');assert((await state()).playing,'level switch preserves playback');
 await page.locator('[data-action=play]').click();const paused=(await state()).time;await page.clock.runFor(1000);assert.equal((await state()).time,paused);
 await page.selectOption('[data-control=level]','overview');await page.locator('[data-action=play]').click();await page.clock.runFor(40000);
 const end=await page.evaluate(()=>PP_JOURNEY.model.end);assert.equal((await state()).time,end);assert.equal((await state()).playing,false);assert.equal(await page.locator('.pp-rank strong').filter({hasText:'本级清理完成'}).count(),3);
 await page.locator('[data-action=reset]').focus();await page.keyboard.press('Enter');assert.equal((await state()).time,0);
 for(const width of [1440,768,390,320])for(const theme of ['light','dark']){
  await page.setViewportSize({width,height:1000});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await seek(29.3);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`overflow ${width}/${theme}`);
  if(keep&&[1440,390].includes(width)){await page.locator('#quick-timeline').evaluate(e=>e.scrollIntoView({block:'start',behavior:'instant'}));await page.screenshot({path:path.join(dir,`quick-${width}-${theme}.png`)});}
 }
 await page.emulateMedia({reducedMotion:'reduce'});await seek(29.3);assert.equal(await page.locator('[data-pulse]').first().evaluate(e=>getComputedStyle(e).strokeDasharray),'none');
 await page.clock.resume();await page.emulateMedia({reducedMotion:'no-preference'});await page.setViewportSize({width:1440,height:1000});

 await go('quick.html?level=source&event=0%3A5%3Agpu&batch=3&t=18.2');
 assert.equal((await state()).level,'process');assert.equal((await state()).time,18.2);assert.equal(await page.locator('.pp-inline-source').count(),1);
 await page.reload();await page.waitForFunction(()=>!!PP_JOURNEY.model);assert.equal(await page.locator('.pp-inline-source').count(),1);
 await go('quick.html?batch=3&rank=1&view=loop&level=process&t=18.2&loop=5');
 for(const rank of [0,1,2])assert.equal(await page.locator(`[data-rail-rank="${rank}"][data-lane="cpu"]`).count(),1);
 assert.equal((await state()).time,18.2);await page.selectOption('[data-control=rank]','2');
 for(const rank of [0,1,2])assert.equal(await page.locator(`[data-rail-rank="${rank}"][data-lane="gpu"]`).count(),1);
 await page.locator('.pp-rails [data-inspect="0:5:recv_out:wait"]').click();
 assert.match(await page.locator('.pp-inline-source h3').innerText(),/PP0/);assert.equal((await state()).rank,2,'source expansion does not change iteration reference');
 await go('quick.html?batch=3&level=process&t=27.6');
 await page.locator('.pp-relation-tabs button').filter({hasText:'控制／完成依赖'}).first().click();
 assert.equal(await page.locator('.pp-relation-chain').getAttribute('data-wire-kind'),'dependency');assert.equal(await page.locator('[data-cross-pulse]').count(),0);
 await go('index.html?compare=slow-cpu&t=164&event=0%3A9%3Arelease');
 await page.waitForFunction(()=>PP_JOURNEY.duration>164);
 assert.equal((await state()).time,164,'long independent comparison survives initial restore');
 assert.match(await page.locator('.pp-dependency-panel').last().innerText(),/独立对照/);
 await page.reload();await page.waitForFunction(()=>PP_JOURNEY.duration>164);assert.equal((await state()).time,164);
 // Dependency view keeps all stages, upstream production and a common A/B clock.
 await go('index.html?event=0%3A5%3Arecv_out&t=18.2');
 await page.waitForSelector('[data-model-panel="dep-a"]');
 for(const rank of [0,1,2])for(const lane of ['cpu','gpu'])assert.equal(await page.locator(`[data-model-panel="dep-a"] [data-rail-rank="${rank}"][data-lane="${lane}"]`).count(),1);
 assert.match(await page.locator('[data-producer-chain="2:3:out_message"]').innerText(),/GPU 前向 27.50.*27.50–27.72/s);
 assert.equal(await page.locator('[data-rail-rank="2"][data-lane="gpu"] [data-node="2:3:gpu"]').getAttribute('data-related'),'source');
 await page.locator('[data-direct-graph]>summary').click();await seek(27.6);assert(await page.locator('[data-direct-graph]').evaluate(e=>e.open));
 await page.locator('[data-rail-rank="1"][data-lane="cpu"] [data-node]').first().click();assert.equal((await state()).time,27.6);
 await go('index.html?scenario=short-gpu&event=0%3A5%3Arecv_out&t=90');
 await page.waitForSelector('[data-model-panel="dep-b"]');
 assert.equal((await state()).time,90,'incoming baseline clock survives a shorter current scene');await seek(18.2);
 const clockMax=await page.evaluate(()=>PP_JOURNEY.duration);assert(clockMax>=101.7,'shorter B must not truncate baseline A');
 const ticks=await page.locator('[data-model-panel] .pp-rails').evaluateAll(nodes=>nodes.filter((_,i)=>i%2===0).map(e=>[...e.querySelectorAll(':scope>text')].map(t=>t.textContent)));
 assert.deepEqual(ticks[0],ticks[1],'A/B context uses identical time window');
 for(const width of [1440,390,320])for(const theme of ['light','dark']){
  await page.setViewportSize({width,height:1000});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`dependency context overflow ${width}/${theme}`);
  if(keep&&theme==='light'){await page.locator('.pp-dependency-panel').first().evaluate(e=>e.scrollIntoView({block:'start',behavior:'instant'}));await page.screenshot({path:path.join(dir,`dependency-context-${width}.png`)});}
 }
 await page.setViewportSize({width:1440,height:1000});await page.evaluate(()=>document.documentElement.dataset.theme='light');
 // The source reading page freezes the exact incoming clock, even with play=1.
 await page.goto(base+'sglang/pd-prefill-pp-loop/notes.html?batch=3&event=0%3A5%3Agpu&inspect=0%3A5%3Arecv_out%3Await&t=18.2&play=1');
 await page.waitForSelector('[data-snapshot-rank="2"]');
 assert.equal(await page.locator('[data-pp-reading-snapshot]').getAttribute('data-snapshot-time'),'18.200');
 assert.equal(await page.locator('[data-snapshot-rank]').count(),3);assert.match(await page.locator('[data-reading-operation]').innerText(),/M1|output/);
 await page.waitForTimeout(150);assert.equal((await state()).time,18.2);assert.equal(await page.locator('[data-control=time]').count(),0);
 const returned=new URL(await page.locator('.pp-reading-snapshot a[href*="quick.html"]').getAttribute('href'));assert.equal(returned.searchParams.get('t'),'18.200');assert.equal(returned.searchParams.get('play'),'1');
 for(const a of await page.locator('a[data-pp-fixed]').evaluateAll(nodes=>nodes.map(n=>n.href)))assert.equal(new URL(a).searchParams.get('scenario'),'baseline');
 await page.goto(base+'sglang/pd-prefill-pp-loop/notes.html?scenario=short-gpu&compare=slow-cpu&t=164&batch=4');
 await page.waitForSelector('[data-snapshot-rank="2"]');assert.equal(await page.locator('[data-pp-reading-snapshot]').getAttribute('data-snapshot-scenario'),'slow-cpu');assert.match(await page.locator('.pp-reading-snapshot').innerText(),/独立对照 B/);
 for(const width of [1440,390,320]){
  await page.setViewportSize({width,height:1000});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`reading overflow ${width}`);
  if(keep)await page.screenshot({path:path.join(dir,`reading-context-${width}.png`)});
 }
 await page.setViewportSize({width:1440,height:1000});
 await go('quick.html?scenario=custom&timing=invalid');assert.equal(await page.evaluate(()=>PP_JOURNEY.scenario),'baseline');
 await page.locator('.pp-advanced>summary').first().click();await page.locator('.timing-edit').click();
 await page.waitForFunction(()=>PP_JOURNEY.scenario==='custom');await seek(30);
 const field=page.locator('.timing-editor input[type=number]').first();await field.fill('0');await field.dispatchEvent('input');
 await page.waitForFunction(()=>PP_JOURNEY.state.time===0);assert.equal((await state()).playing,false);
 assert(!/undefined|NaN/.test(await page.locator('.pp-scene').innerText()),'visible values must be defined');
 link=await page.locator('.pp-local-nav a[href*="notes.html"]').first().getAttribute('href');await page.goto(link);await page.waitForSelector('[data-snapshot-rank="2"]');assert.equal(await page.locator('[data-pp-reading-snapshot]').getAttribute('data-snapshot-scenario'),'custom');
 for(const width of [1440,390,320]){
  await page.setViewportSize({width,height:1000});await go('quick.html?batch=3&rank=0&level=process&t=18.2');
  await page.locator('.pp-track-title').scrollIntoViewIfNeeded();
  if(keep)await page.screenshot({path:path.join(dir,`process-${width}.png`)});
  await page.locator('.pp-rails [data-inspect="0:5:recv_out:wait"]').click();
  assert.match(await page.locator('.pp-inline-source').innerText(),/M1/);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`source overflow ${width}`);
  if(keep)await page.screenshot({path:path.join(dir,`source-${width}.png`)});
  await page.locator('.pp-rails [data-inspect="0:5:recv_req"]').focus();await page.keyboard.press('Enter');
  assert.equal((await state()).inspect,'0:5:recv_req');
  await page.locator('[data-action=operations]').click();
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`expanded source overflow ${width}`);
 }
 await page.setViewportSize({width:1440,height:1000});
 if(keep){
  await go('quick.html?level=process&view=loop&rank=0&loop=5&batch=3&t=20');
  await page.locator('.pp-track-title').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(dir,'execution-tracks.png')});
  await go('quick.html?batch=3&level=process&t=27.6');
  await page.locator('.pp-track-title').evaluate(e=>e.scrollIntoView({block:'start',behavior:'instant'}));await page.screenshot({path:path.join(dir,'joint-handoff.png')});
  await go('index.html?scenario=slow-l2&event=0%3A4%3Agpu&t=12');
  assert.equal((await state()).event,'0:4:gpu');assert.equal((await state()).batch,2,'event focus and tracked batch agree');await page.locator('.pp-dependency-panel').first().scrollIntoViewIfNeeded();await page.screenshot({path:path.join(dir,'dependencies.png')});
  await go('index.html?event=0%3A5%3Arecv_out&t=18.2');await page.locator('.pp-dependency-panel').first().evaluate(e=>e.scrollIntoView({block:'start',behavior:'instant'}));await page.screenshot({path:path.join(dir,'dependency-joint.png')});
  await page.goto(base+'sglang/pd-prefill-pp-loop/notes.html?batch=3&t=18.2&event=0%3A5%3Arecv_out');await page.waitForSelector('[data-snapshot-rank="2"]');await page.locator('#source-map').evaluate(e=>e.scrollIntoView({block:'start',behavior:'instant'}));await page.screenshot({path:path.join(dir,'source-reading.png')});
 }
 assert.deepEqual(errors,[]);await browser.close();console.log('PASS: first viewport, continuous end-to-end play, pause/reverse, pulse intervals, levels, shared focus, iframe, cross-page/reload, scenario reset, responsive themes and reduced motion.');
}finally{if(server)await server.close();if(meta){meta.exited=server.process().exitCode!==null||server.process().signalCode!==null;meta.profileRemoved=!fs.existsSync(meta.profile);meta.remaining=execFileSync('ps',['-axo','pid=,command='],{encoding:'utf8'}).split('\n').filter(l=>l.includes(meta.profile));fs.writeFileSync(path.join(dir,'browser.json'),JSON.stringify(meta,null,2));assert(meta.exited&&meta.profileRemoved);assert.deepEqual(meta.remaining,[]);}if(!keep)fs.rmSync(dir,{recursive:true,force:true});}})().catch(e=>{console.error(e);process.exitCode=1;});
