const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{execFileSync}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const M=require('../../pages/sglang/pp-consensus/model.js'),base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:51459/';
const keep=!!process.env.PP_CONSENSUS_QA_DIR,dir=process.env.PP_CONSENSUS_QA_DIR||fs.mkdtempSync(path.join(os.tmpdir(),'pp-consensus-qa-'));fs.mkdirSync(dir,{recursive:true});
(async()=>{let server,meta;try{
 server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 const pid=server.process().pid,command=execFileSync('ps',['-p',String(pid),'-o','command='],{encoding:'utf8'});meta={session:'pp-consensus-qa',controller:process.pid,pid,command,profile:command.match(/--user-data-dir=([^ ]+)/)?.[1]};fs.writeFileSync(path.join(dir,'browser.json'),JSON.stringify(meta,null,2));
 const browser=await chromium.connect(server.wsEndpoint()),page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.url().startsWith(base)&&r.status()>=400)errors.push(r.status()+' '+r.url());});
 await page.goto(base+'sglang/pp-consensus/#consensus-title');await page.locator('#consensus-lab[data-ready=true]').waitFor();await page.evaluate(()=>document.fonts.ready);
 await page.clock.install({time:new Date('2026-01-01T00:00:00Z')});await page.clock.pauseAt(new Date('2026-01-01T00:00:01Z'));
 const host=page.locator('#consensus-lab'),scene=host.locator('[data-role=scene]'),tracks=host.locator('[data-role=tracks]');
 const seek=t=>host.locator('[data-role=seek]').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));},t);
 const screenshot=async name=>{if(keep){await page.locator('#consensus-title').evaluate(e=>e.scrollIntoView({block:'start',behavior:'instant'}));await page.screenshot({path:path.join(dir,name+'.png')});}};
 await screenshot('default');
 for(const [mode,info]of Object.entries(M.modes)){
  await host.locator(`[data-mode=${mode}]`).click();assert.equal(await host.getAttribute('data-time'),'0.000');
  for(const condition of Object.keys(info.cases)){
   await host.locator('[data-role=condition]').selectOption(condition);const tl=M.build(mode,condition);assert.equal(await host.getAttribute('data-time'),'0.000');
   for(const e of tl.events.filter(e=>e.kind!=='device'&&e.start<tl.duration)){
    await seek((e.start+Math.min(e.end,tl.duration))/2);
    assert.equal(await tracks.locator(`[data-event="${e.id}"]`).getAttribute('data-active'),'true');assert.equal(await scene.getAttribute('data-clock'),await tracks.getAttribute('data-clock'));
    const overflow=await scene.locator('text').evaluateAll(es=>es.filter(e=>{const b=e.getBBox();return b.x<0||b.x+b.width>1001||b.y<0||b.y+b.height>366;}).map(e=>e.textContent));assert.deepEqual(overflow,[],`${mode}/${condition}/${e.id} text bounds`);
   }
   await seek(8.7);const snapshot=await scene.innerHTML();await seek(tl.duration);await seek(8.7);assert.equal(await scene.innerHTML(),snapshot,'reverse rendering');
   if(mode==='bootstrap'&&condition==='metadata'){await seek(tl.duration);for(let r=0;r<3;r++)assert.equal(await scene.locator(`[data-req="${r}-R1"]`).getAttribute('data-queue'),'bootstrap');await screenshot('metadata-filter');}
   if(mode==='release'&&condition==='repoll'){await seek(tl.duration);assert.equal(await scene.locator('[data-req="1-R0"]').getAttribute('data-queue'),'inflight');await screenshot('local-repoll');}
   if(mode==='hicache'&&condition==='normal'){await seek(8);assert.equal(await scene.locator('[data-ack="1-L1"]').getAttribute('data-state'),'popped');assert.equal(await scene.locator('[data-ack="2-L1"]').getAttribute('data-state'),'finished');await screenshot('hicache-overlap');}
   await seek(0);await host.locator('[data-role=speed]').selectOption('2');await host.locator('[data-action=play]').click();await page.clock.runFor(Math.ceil(tl.duration*500)+200);assert.ok(Math.abs(Number(await host.getAttribute('data-time'))-tl.duration)<.001);assert.equal(await host.locator('[data-action=play]').getAttribute('aria-pressed'),'false');
  }
  await host.locator('[data-role=condition]').selectOption('normal');const tl=M.build(mode),move=tl.events.find(e=>e.kind==='message');await seek(move.start+.2);const before=await scene.locator('[data-motion]').getAttribute('data-x');await seek(move.end-.2);assert.notEqual(await scene.locator('[data-motion]').getAttribute('data-x'),before,'message must move');await screenshot(mode+'-message');
  await host.locator('[data-action=play]').click();await page.clock.runFor(350);await host.locator('[data-action=play]').click();const paused=await host.getAttribute('data-time');await page.clock.runFor(500);assert.equal(await host.getAttribute('data-time'),paused);
  for(const width of [1440,1280,942,768,390,320])for(const theme of ['light','dark']){
   await page.setViewportSize({width,height:1100});await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);await seek(8);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${mode}/${width}/${theme} page overflow`);
   if(keep&&width===390)await screenshot(`${mode}-${width}-${theme}`);
  }
  await page.setViewportSize({width:1440,height:1100});await page.evaluate(()=>document.documentElement.dataset.theme='light');
 }
 await page.emulateMedia({reducedMotion:'reduce'});await seek(2.5);const still=await scene.locator('[data-motion]').getAttribute('data-x');await seek(3);assert.equal(await scene.locator('[data-motion]').getAttribute('data-x'),still);await page.emulateMedia({reducedMotion:'no-preference'});
 await host.locator('[data-action=reset]').focus();await page.keyboard.press('Enter');assert.equal(await host.getAttribute('data-time'),'0.000');await host.locator('[data-action=next]').focus();await page.keyboard.press('Enter');assert.ok(Number(await host.getAttribute('data-time'))>0);
 await page.goto(base+'sglang/pp-consensus/?mode=release&condition=repoll&t=17#consensus-title');await page.locator('#consensus-lab[data-ready=true]').waitFor();assert.equal(await host.getAttribute('data-time'),'17.000');assert.equal(await host.locator('[data-role=condition]').inputValue(),'repoll');
 assert.equal(await page.locator('.site-sidebar [data-module=disaggregation] [aria-current=page]').count(),1);
 const nojs=await browser.newPage({javaScriptEnabled:false});await nojs.goto(base+'sglang/pp-consensus/');assert.ok(await nojs.locator('noscript').isVisible());assert.equal(await nojs.locator('#source-map').count(),1);await nojs.close();assert.deepEqual(errors,[]);await browser.close();
 console.log('PASS: 11 scenarios, shared clock, reversible state, full playback, actual message motion, pause, keyboard, source bounds, six widths/two themes, restored URL, navigation and no-JS.');
}finally{if(server)await server.close();if(meta){meta.browserExited=server.process().exitCode!==null||server.process().signalCode!==null;meta.profileRemoved=!fs.existsSync(meta.profile);meta.remaining=execFileSync('ps',['-axo','pid=,command='],{encoding:'utf8'}).split('\n').filter(s=>meta.profile&&s.includes(meta.profile));fs.writeFileSync(path.join(dir,'browser.json'),JSON.stringify(meta,null,2));assert.ok(meta.browserExited&&meta.profileRemoved);assert.deepEqual(meta.remaining,[]);}if(!keep)fs.rmSync(dir,{recursive:true,force:true});}})().catch(e=>{console.error(e);process.exitCode=1;});
