const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{execFileSync}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const T=require('../../pages/sglang/pd-execution/timeline.js'),D=require('../../pages/sglang/pd-dataflow/model.js'),L=require('../../pages/sglang/pd-prefill-lifecycle/scenario-engine.js');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:51459/',keep=!!process.env.PD_EXECUTION_QA_DIR,dir=process.env.PD_EXECUTION_QA_DIR||fs.mkdtempSync(path.join(os.tmpdir(),'pd-execution-qa-'));
fs.mkdirSync(dir,{recursive:true});
(async()=>{let server,meta;try{
 server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 const pid=server.process().pid,command=execFileSync('ps',['-p',String(pid),'-o','command='],{encoding:'utf8'});
 meta={session:'pd-execution-qa',controller:process.pid,pid,command,profile:command.match(/--user-data-dir=([^ ]+)/)?.[1]};
 const browser=await chromium.connect(server.wsEndpoint()),page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400&&r.url().startsWith(base))errors.push(r.status()+' '+r.url());});
 await page.clock.install({time:new Date('2026-01-01T00:00:00Z')});await page.clock.pauseAt(new Date('2026-01-01T00:00:01Z'));
 for(const kind of ['lifecycle','dataflow']){
  const slug=kind==='lifecycle'?'pd-prefill-lifecycle':'pd-dataflow',id=kind==='lifecycle'?'lifecycle-execution':'dataflow-execution';
  await page.goto(base+'sglang/'+slug+'/#'+(kind==='lifecycle'?'pipeline-heading':'lab-title'));
  const host=page.locator('#'+id);await host.locator('[data-role=board] .px-stage').first().waitFor({state:'attached'});
  assert.equal(await host.locator('.px-stage').count(),3);assert.equal(await page.locator('[data-role=seek]').count(),1,'one shared clock');
  await page.evaluate(()=>document.fonts.ready);await host.evaluate(e=>e.scrollIntoView({block:'start',behavior:'instant'}));
  if(keep)await page.screenshot({path:path.join(dir,kind+'-initial.png')});
  const model=kind==='lifecycle'?L.createScenario():{...D.create('chunks'),sources:D.sources},tl=T.build(model,kind);
  const seek=t=>host.locator('[data-role=seek]').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));},t);
  const snapshots=[];
  for(const e of tl.events){const at=(e.start+e.end)/2;await seek(at);assert.equal(await host.locator(`[data-event="${e.id}"]`).getAttribute('data-active'),'true',kind+'/'+e.id);assert.equal(await host.getAttribute('data-time'),await host.locator('[data-role=tracks]').getAttribute('data-clock'));}
  for(const e of tl.commits.filter(e=>/B1-send-0|B2-(compute|forward)-0|B3-result-0|release-1|finish-1/.test(e.id))){await seek(e.end);snapshots.push([e.end,await host.locator('[data-role=board]').innerHTML()]);}
  for(const [t,html]of snapshots.reverse()){await seek(t);assert.equal(await host.locator('[data-role=board]').innerHTML(),html,'reverse reconstruction');}
  const overlap=tl.events.find(e=>e.type==='compute'&&e.id.startsWith('B2')&&tl.copies.some(c=>c.id.includes('B1')&&c.start<e.start&&c.end>e.end));
  assert.ok(overlap);await seek(overlap.start+.2);const move=host.locator('[data-role=routes] [data-motion]').first(),before=await move.getAttribute('data-x');await seek(overlap.start+.6);assert.notEqual(await move.getAttribute('data-x'),before);
  if(keep)await host.screenshot({path:path.join(dir,kind+'-overlap.png')});
  if(kind==='dataflow'){await seek(tl.events.find(e=>e.id==='B3-result-0').end);assert.equal(await host.locator('[data-ref=batch]').getAttribute('data-held'),'true');assert.equal(await host.locator('[data-ref=inflight]').getAttribute('data-held'),'true');if(keep)await host.screenshot({path:path.join(dir,'same-req-references.png')});}
  await seek(0);await host.locator('[data-action=play]').click();await page.clock.runFor(500);await host.locator('[data-action=play]').click();const paused=await host.getAttribute('data-time');await page.clock.runFor(500);assert.equal(await host.getAttribute('data-time'),paused);
  await host.locator('[data-role=speed]').selectOption('3');await host.locator('[data-action=play]').click();await page.clock.runFor(Math.ceil(tl.duration*1000/3)+1500);assert.ok(Math.abs(Number(await host.getAttribute('data-time'))-tl.duration)<.001);assert.equal(await host.locator('[data-action=play]').getAttribute('aria-pressed'),'false');
  await host.locator('[data-action=reset]').focus();await page.keyboard.press('Enter');assert.equal(Number(await host.getAttribute('data-time')),0);
  if(kind==='dataflow')for(const scenario of D.scenarios){await page.selectOption('#scenario',scenario.id);await seek(99999);assert.match(await host.locator('[data-role=source]').getAttribute('href'),new RegExp(D.commit));}
  else{await page.locator('.scenario-settings>summary').click();for(const preset of Object.keys(L.presets)){await page.selectOption('#config-preset',preset);await seek(99999);}await page.selectOption('#config-preset','chunks');await page.locator('.scenario-settings>summary').click();}
  for(const width of [1440,1280,942,768,390,320])for(const theme of ['light','dark']){await page.setViewportSize({width,height:1100});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await seek(12);assert.deepEqual(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth?[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+2&&!e.closest('.px-scroll')).map(e=>[e.tagName,e.className,e.getBoundingClientRect().width]).slice(-12):[]),[],`${kind}/${width}/${theme} overflow`);if(keep&&width===390&&theme==='light')await host.screenshot({path:path.join(dir,kind+'-mobile.png')});}
  await page.emulateMedia({reducedMotion:'reduce'});await seek(10);await page.emulateMedia({reducedMotion:'no-preference'});await page.setViewportSize({width:1440,height:1100});
 }
 for(const slug of ['pd-prefill-lifecycle','pd-dataflow']){
  const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto(base+'sglang/'+slug+'/');assert.ok(await plain.locator('noscript').first().isVisible());await plain.close();
 }
 await page.goto(base+'sglang/pd-dataflow/?scenario=repoll&t=10.500&rank=2#lab-title');await page.locator('#dataflow-execution[data-ready=true]').waitFor();assert.equal(await page.locator('#dataflow-execution').getAttribute('data-time'),'10.500');assert.equal(await page.locator('[data-role=rank]').inputValue(),'2');
 assert.deepEqual(errors,[]);await browser.close();console.log('PASS: both lessons, default rendering, continuous overlap, single clock, reversible state, full playback, scenario branches, restored URLs, no-JS, six widths and two themes.');
}finally{if(server)await server.close();if(meta){meta.browserExited=server.process().exitCode!==null||server.process().signalCode!==null;meta.profileRemoved=!fs.existsSync(meta.profile);meta.remaining=execFileSync('ps',['-axo','pid=,command='],{encoding:'utf8'}).split('\n').filter(x=>meta.profile&&x.includes(meta.profile));fs.writeFileSync(path.join(dir,'browser.json'),JSON.stringify(meta,null,2));assert.ok(meta.browserExited&&meta.profileRemoved);assert.deepEqual(meta.remaining,[]);}if(!keep)fs.rmSync(dir,{recursive:true,force:true});}})().catch(e=>{console.error(e);process.exitCode=1;});
