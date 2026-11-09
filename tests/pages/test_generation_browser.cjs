const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{execFileSync}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright'),M=require('../../pages/sglang/advanced-generation/execution-model.js');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:51459/',keep=!!process.env.GENERATION_QA_DIR,dir=process.env.GENERATION_QA_DIR||fs.mkdtempSync(path.join(os.tmpdir(),'generation-qa-'));
fs.mkdirSync(dir,{recursive:true});
(async()=>{let server,meta;try{
 server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 const pid=server.process().pid,command=execFileSync('ps',['-p',String(pid),'-o','command='],{encoding:'utf8'});
 meta={session:'generation-execution-qa',controller:process.pid,pid,command,profile:command.match(/--user-data-dir=([^ ]+)/)?.[1]};fs.writeFileSync(path.join(dir,'browser.json'),JSON.stringify(meta,null,2));
 const browser=await chromium.connect(server.wsEndpoint()),page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.url().startsWith(base)&&r.status()>=400)errors.push(r.status()+' '+r.url());});
 await page.goto(base+'sglang/advanced-generation/');await page.locator('#generation-execution[data-ready=true]').waitFor();await page.evaluate(()=>document.fonts.ready);
 if(keep)await page.screenshot({path:path.join(dir,'initial-viewport.png')});
 const host=page.locator('#generation-execution'),scene=host.locator('[data-role=scene]'),tracks=host.locator('[data-role=tracks]');
 const seek=t=>host.locator('[data-role=seek]').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));},t);
 const shot=async name=>{if(keep){await page.locator('#gen-title').evaluate(e=>e.scrollIntoView({block:'start',behavior:'instant'}));await page.screenshot({path:path.join(dir,name+'.png')});await host.screenshot({path:path.join(dir,name+'-full.png')});}};
 await shot('default');
 await page.clock.install({time:new Date('2026-01-01T00:00:00Z')});await page.clock.pauseAt(new Date('2026-01-01T00:00:01Z'));
 for(const [mode,info]of Object.entries(M.modes)){
  await host.locator('[data-mode='+mode+']').click();
  for(const condition of Object.keys(info.cases)){
   await host.locator('[data-role=condition]').selectOption(condition);const tl=M.build(mode,condition);
   assert.equal(await host.getAttribute('data-time'),'0.000');
   if(condition==='invalid')assert.match(await scene.textContent(),/regex \/ \( （无效）/);
   for(const e of tl.events){
    await seek((e.start+e.end)/2);assert.equal(await tracks.locator('[data-event='+e.id+']').getAttribute('data-active'),'true');
    assert.equal(await scene.getAttribute('data-clock'),await tracks.getAttribute('data-clock'));
    const overflow=await scene.locator('text').evaluateAll(es=>es.filter(e=>{const b=e.getBBox();return b.x<0||b.x+b.width>1101||b.y<0||b.y+b.height>391;}).map(e=>e.textContent));
    assert.deepEqual(overflow,[],mode+'/'+condition+'/'+e.id+' text bounds');
   }
   await seek(5.1);const before=await scene.innerHTML();await seek(tl.duration);await seek(5.1);assert.equal(await scene.innerHTML(),before,'rewind is deterministic');
   await seek(0);await host.locator('[data-role=speed]').selectOption('2');await host.locator('[data-action=play]').click();await page.clock.runFor(Math.ceil(tl.duration*500)+200);
   assert.ok(Math.abs(Number(await host.getAttribute('data-time'))-tl.duration)<.001,'full playback '+mode+'/'+condition);assert.equal(await host.locator('[data-action=play]').getAttribute('aria-pressed'),'false');
   console.log('Playback '+mode+'/'+condition);
  }
  if(mode==='grammar')await host.locator('[data-role=condition]').selectOption('hit');
  for(const width of [1440,768,390,320])for(const theme of ['light','dark']){
   await page.setViewportSize({width,height:1100});await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);await seek(7.8);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,mode+' document overflow '+width);
   if([1440,390].includes(width))await shot(mode+'-'+width+'-'+theme);
  }
 }
 await page.setViewportSize({width:1440,height:1100});await page.evaluate(()=>document.documentElement.dataset.theme='light');
 await host.locator('[data-mode=speculative]').click();await host.locator('[data-role=condition]').selectOption('tree');
 await seek(10.2);const start=await scene.locator('[data-kv-node="3"] rect').getAttribute('x');await seek(11);assert.notEqual(await scene.locator('[data-kv-node="3"] rect').getAttribute('x'),start);assert.ok(Number(await scene.locator('[data-predict-node="3"] rect').getAttribute('x'))>380,'predict follows same noncontiguous path');await shot('tree-compaction');
 await host.locator('[data-mode=combined]').click();await seek(4.7);const x=await scene.locator('[data-motion]').getAttribute('data-x');await seek(5);assert.notEqual(await scene.locator('[data-motion]').getAttribute('data-x'),x,'continuous traversal');
 await host.locator('[data-action=play]').click();await page.clock.runFor(200);await host.locator('[data-action=play]').click();const stopped=await scene.innerHTML();await page.clock.runFor(400);assert.equal(await scene.innerHTML(),stopped,'pause freezes geometry');
 await host.locator('[data-role=condition]').selectOption('late');await seek(5);await shot('late-barrier');await tracks.locator('[data-event=maskWait]').focus();await page.keyboard.press('Enter');assert.match(await host.locator('[data-role=phase]').innerText(),/mask 尚未提交/);await seek(11.4);await shot('late-mask');await seek(M.build('combined','late').duration);await shot('formal-commit');
 await host.locator('[data-action=reset]').focus();await page.keyboard.press('Enter');assert.equal(await host.getAttribute('data-time'),'0.000');await tracks.locator('[data-event=enter1]').focus();await page.keyboard.press('Enter');assert.ok(Number(await host.getAttribute('data-time'))>0);
 await page.emulateMedia({reducedMotion:'reduce'});await seek(8.5);const still=await scene.locator('[data-motion]').getAttribute('data-x');await seek(8.8);assert.equal(await scene.locator('[data-motion]').getAttribute('data-x'),still);await page.emulateMedia({reducedMotion:'no-preference'});
 await page.goto(base+'sglang/advanced-generation/?mode=combined&case=late&t=5#gen-title');await host.waitFor();assert.equal(await host.getAttribute('data-time'),'5.000');assert.equal(await host.locator('[data-role=condition]').inputValue(),'late');
 assert.equal(await page.locator('.site-sidebar [data-module=advanced-generation] [aria-current=page]').count(),1);
 // Real explicit site update with mocked next-release HTML: controller checkpoint survives, paused.
 const current=await page.locator('meta[name=site-release]').getAttribute('content'),next='bbbbbbbbbbbbbbbb',html=fs.readFileSync(path.resolve(__dirname,'../../.pages-dist/sglang/advanced-generation/index.html'),'utf8');
 await page.route('**/*',route=>{const u=new URL(route.request().url());if(u.pathname.endsWith('/site-release.json'))return route.fulfill({json:{schema:1,release:next}});if(u.searchParams.has('_refresh'))return route.fulfill({contentType:'text/html',body:html.replace('content="'+current+'"','content="'+next+'"')});return route.continue();});
 await host.locator('[data-role=speed]').selectOption('1.5');await page.locator('[data-site-check]').click();await page.locator('.site-update-notice [data-site-update]').waitFor({state:'visible'});await page.locator('.site-update-notice [data-site-update]').click();
 await page.waitForFunction(()=>document.querySelector('meta[name=site-release]').content==='bbbbbbbbbbbbbbbb');await page.waitForLoadState('load');
 assert.equal(await host.getAttribute('data-time'),'5.000');assert.equal(await host.getAttribute('data-mode'),'combined');assert.equal(await host.getAttribute('data-condition'),'late');assert.equal(await host.locator('[data-role=speed]').inputValue(),'1.5');assert.equal(await host.locator('[data-action=play]').getAttribute('aria-pressed'),'false');
 const nojs=await browser.newPage({javaScriptEnabled:false});await nojs.goto(base+'sglang/advanced-generation/');assert.equal(await nojs.locator('noscript').isVisible(),true);assert.match(await nojs.locator('#generation-tree').locator('..').innerText(),/traverse_tree.*父节点 mask/s);assert.equal(await nojs.locator('#source-baseline a').count()>0,true);await nojs.close();
 assert.deepEqual(errors,[]);await browser.close();console.log('PASS: 10 cases/full playback, shared clock, reversible frames, real motion, pause, keyboard, URL and update checkpoint, four widths/two themes, no-JS.');
}finally{if(server)await server.close();if(meta){meta.browserExited=server.process().exitCode!==null||server.process().signalCode!==null;meta.profileRemoved=!fs.existsSync(meta.profile);meta.remaining=execFileSync('ps',['-axo','pid=,command='],{encoding:'utf8'}).split('\n').filter(s=>meta.profile&&s.includes(meta.profile));fs.writeFileSync(path.join(dir,'browser.json'),JSON.stringify(meta,null,2));assert.ok(meta.browserExited&&meta.profileRemoved);assert.deepEqual(meta.remaining,[]);}if(!keep)fs.rmSync(dir,{recursive:true,force:true});}
})().catch(e=>{console.error(e);process.exitCode=1;});
