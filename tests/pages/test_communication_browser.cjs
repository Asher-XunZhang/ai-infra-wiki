const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {execFileSync}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const M=require('../../pages/sglang/communication/flow-model.js');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
const retained=Boolean(process.env.COMM_QA_DIR),artifacts=process.env.COMM_QA_DIR||fs.mkdtempSync(path.join(os.tmpdir(),'communication-qa-'));
fs.mkdirSync(artifacts,{recursive:true});
(async()=>{let server,metadata;
try{
 server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 const pid=server.process().pid,command=execFileSync('ps',['-p',String(pid),'-o','command='],{encoding:'utf8'});
 metadata={session:'communication-implementation-qa',controller:process.pid,pid,profile:command.match(/--user-data-dir=([^ ]+)/)?.[1],command};fs.writeFileSync(path.join(artifacts,'browser.json'),JSON.stringify(metadata,null,2));
 const browser=await chromium.connect(server.wsEndpoint()),page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400&&r.url().startsWith(base))errors.push(r.status()+' '+r.url());});
 await page.goto(base+'sglang/communication/#comm-title');await page.evaluate(()=>document.fonts.ready);
 await page.waitForFunction(()=>{const title=document.querySelector('#comm-title').getBoundingClientRect(),board=document.querySelector('#comm-canvas').getBoundingClientRect();return title.top>=60&&title.top<150&&board.top<innerHeight&&board.height>0;});
 if(retained)await page.screenshot({path:path.join(artifacts,'default-viewport.png')});
 await page.clock.install({time:new Date('2026-01-01T00:00:00Z')});await page.clock.pauseAt(new Date('2026-01-01T00:00:01Z'));
 const scene=page.locator('#comm-canvas'),drawing=page.locator('#comm-drawing'),toggle=page.locator('#comm-toggle');
 const choose=async(mode,option)=>{await page.locator(`button[data-mode="${mode}"]`).click();if(option&&M.options[mode].length>1)await page.selectOption('#comm-option',option);};
 const seek=async t=>page.locator('#comm-time').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));},t);
 const time=async()=>Number(await scene.getAttribute('data-time'));
 const checkBounds=async label=>{const outside=await scene.locator('text').evaluateAll(es=>{const svg=es[0].ownerSVGElement,frame=svg.getBoundingClientRect(),scale=svg.viewBox.baseVal.width/frame.width;return es.filter(e=>{const b=e.getBoundingClientRect();return (b.left-frame.left)*scale<-.5||(b.right-frame.left)*scale>1040.5||(b.top-frame.top)*scale<-.5||(b.bottom-frame.top)*scale>svg.viewBox.baseVal.height+.5;}).map(e=>({text:e.textContent}));});assert.deepEqual(outside,[],label);};
 assert.equal(await page.locator('.animation-speed,.mechanism-film').count(),0,'one playback owner');
 assert.equal(await page.locator('[data-comm-lab]').isVisible(),true);
 const duplicated=await page.locator('[id]').evaluateAll(es=>es.map(e=>e.id).filter((v,i,arr)=>arr.indexOf(v)!==i));assert.deepEqual(duplicated,[],'unique IDs');
 for(const mode of Object.keys(M.catalog))for(const [option] of M.options[mode]){
  await choose(mode,option);const tl=M.build(mode,option),snapshots=[];assert.equal(await time(),0);
  for(const phase of tl.phases){
   for(const progress of [.08,.5,1]){const t=Number((phase.start+phase.duration*progress).toFixed(2));await seek(t);await checkBounds(`${mode}/${option}/${phase.id}`);snapshots.push([t,await drawing.innerHTML()]);}
   const refs=await page.locator('#comm-sources a').evaluateAll(es=>es.map(e=>e.href));assert.ok(refs.length&&refs.every(x=>x.includes(M.revision)));
  }
  await seek(100);assert.equal(await page.locator('#comm-next').isDisabled(),true,`${mode}/${option}: slider reaches exact end`);if(retained)await scene.screenshot({path:path.join(artifacts,`${mode}-${option}-end.png`)});
  for(const [t,html]of snapshots.reverse()){await seek(t);assert.equal(await drawing.innerHTML(),html,`${mode}/${option}: reverse seek`);}
  const phase=tl.phases.find(p=>p.id===({ipc:'send',tensor:'hidden',collective:'execute',kv:'copy',ready:'second',release:'free'}[mode]));
  await seek(phase.start+phase.duration*.4);const a=await scene.locator('.moving-payload').first().evaluate(e=>[e.dataset.x,e.dataset.y].join(','));
  if(retained)await scene.screenshot({path:path.join(artifacts,`${mode}-${option}-moving.png`)});
  await seek(phase.start+phase.duration*.6);const b=await scene.locator('.moving-payload').first().evaluate(e=>[e.dataset.x,e.dataset.y].join(','));assert.notEqual(a,b,mode+': continuous payload');
  await page.locator('#comm-reset').click();await toggle.click();await page.clock.runFor(Math.ceil(tl.duration*1000)+100);assert.equal(await time(),Number(tl.duration.toFixed(3)),mode+': complete play');assert.equal(await toggle.getAttribute('aria-pressed'),'false');assert.equal(await page.locator('#comm-next').isDisabled(),true);
 }
 // Hardware routes must agree with the selected placement and payload memory.
 for(const mode of ['tensor','collective'])for(const place of ['local','rdma']){
  await choose(mode);await page.selectOption('#comm-placement',place);assert.equal(await time(),0);assert.equal(await toggle.getAttribute('aria-pressed'),'false');
  const tl=M.build(mode),phase=tl.phases.find(p=>p.id===(mode==='tensor'?'hidden':'execute'));await seek(phase.start+phase.duration*.5);await checkBounds(mode+'/'+place);
  assert.equal(await scene.locator('[data-hardware="nic0"]').count(),place==='rdma'?1:0);assert.equal(await scene.locator('[data-hardware="switch"]').count(),place==='rdma'?1:0);
  const routes=await scene.locator('.hardware-route[data-residency="device"]').evaluateAll(es=>es.map(e=>e.dataset.nodes));assert.ok(routes.length&&routes.every(x=>!x.includes('cpu')));
  if(retained)await scene.screenshot({path:path.join(artifacts,mode+'-'+place+'-hardware.png')});
 }
 await choose('kv');const auxPhase=M.build('kv').phases.find(p=>p.id==='aux');await seek(auxPhase.start+auxPhase.duration*.5);
 assert.equal(await scene.locator('.hardware-route[data-payload^="aux"]').getAttribute('data-plane'),'data');assert.equal(await scene.locator('.hardware-route[data-payload^="aux"]').getAttribute('data-residency'),'host');
 if(retained)await scene.screenshot({path:path.join(artifacts,'kv-aux-hardware.png')});
 await choose('tensor');await seek(6.9);assert.equal(await scene.locator('[data-activity=hidden][data-active=true]').count(),1);assert.equal(await scene.locator('[data-activity=forwardSubmit][data-active=true]').count(),1);
 await choose('ready','cache');await seek(3);for(const id of ['first','second','metadata','restore'])assert.equal(await scene.locator(`[data-activity=${id}][data-active=true]`).count(),1);if(retained)await scene.screenshot({path:path.join(artifacts,'ready-parallel-conditions.png')});
 // Content causality independent of the generated phase labels.
 const text=async()=>drawing.textContent();
 await choose('tensor');await seek(8);
 const receiver=name=>scene.locator(`[data-buffer="${name}"][data-side="receiver"]`);
 assert.equal(await receiver('hidden_states').getAttribute('data-allocated'),'true');assert.equal(await receiver('hidden_states').getAttribute('data-ready'),'false');assert.equal(await receiver('residual').getAttribute('data-allocated'),'true');
 assert.deepEqual(await receiver('hidden_states').locator('[data-element]').evaluateAll(es=>es.map(e=>e.dataset.element)),Array(8).fill('·'));
 assert.match(await text(),/forward 已入队；GPU 等接收依赖/);
 await seek(M.build('tensor').phases.find(p=>p.id==='hidden').end);
 assert.deepEqual(await receiver('hidden_states').locator('[data-element]').evaluateAll(es=>es.map(e=>e.dataset.element)),Array.from({length:8},(_,i)=>'h'+i));
 for(const condition of ['default','fragmented']){
  await choose('kv',condition);const tl=M.build('kv',condition),copy=tl.phases.find(p=>p.id==='copy');await seek(copy.start+copy.duration*.5);
  const blocks=await scene.locator('[data-transfer-block]').evaluateAll(es=>es.map(e=>({pages:Number(e.dataset.pages),bytes:Number(e.dataset.bytes)})));
  assert.equal(blocks.length,condition==='default'?2:3);assert.equal(blocks.reduce((n,b)=>n+b.bytes,0),48);assert.equal(blocks.reduce((n,b)=>n+b.pages,0),3);
  if(condition==='default')assert.match(await scene.locator('.moving-payload').first().textContent(),/A\+B · 32 B/);
  const aux=tl.phases.find(p=>p.id==='aux');await seek(aux.start+aux.duration*.5);assert.match(await scene.locator('.moving-payload').textContent(),/aux 数据/);assert.doesNotMatch(await scene.locator('.moving-payload').textContent(),/Success/);
  await seek(aux.end+.5);assert.match(await scene.locator('.moving-payload').textContent(),/Success/);assert.match(await text(),/aux 已写入/);
 }

 await choose('kv');await seek(0);assert.match(await text(),/尚待计算/);assert.doesNotMatch(await text(),/A · 16 B/);
 await choose('ready','cache');const cacheTl=M.build('ready','cache');await seek(cacheTl.phases.find(p=>p.id==='metadata').end);assert.match(await text(),/HiCache PENDING/);assert.match(await text(),/D MIN: Transferring/);assert.doesNotMatch(await text(),/metadata gate: Success/);await seek(cacheTl.phases.find(p=>p.id==='cache').end);assert.match(await text(),/metadata gate: Success/);assert.match(await text(),/D MIN: Transferring/);
 await choose('ready','corrupt');await seek(15);assert.match(await text(),/room = 99/);assert.doesNotMatch(await text(),/99 ≠ 42 · 拒绝/);await seek(100);assert.match(await text(),/99 ≠ 42 · 拒绝/);assert.doesNotMatch(await text(),/\[320\]/);
 await choose('release','timeout');await seek(100);assert.match(await text(),/独立来源 1 \/ 2/);assert.match(await text(),/旧字节失效/);assert.doesNotMatch(await text(),/drained=True/);
 await choose('collective','gather');await seek(7);assert.match(await text(),/通信排队/);assert.doesNotMatch(await text(),/结果依赖已满足/);
 // Playback, speed and keyboard keep the same clock.
 await choose('tensor');await seek(7);await toggle.click();await page.clock.runFor(550);await toggle.click();const paused=await time(),frozen=await drawing.innerHTML();await page.clock.runFor(1000);assert.equal(await time(),paused);assert.equal(await drawing.innerHTML(),frozen);
 await page.selectOption('#comm-speed','2');assert.equal(await time(),paused);await toggle.click();await page.clock.runFor(600);await toggle.click();assert.ok(await time()-paused>1&&await time()-paused<1.25,'2x');
 await toggle.click();await choose('kv');await page.clock.runFor(600);assert.equal(await time(),0);await page.locator('#comm-next').focus();await page.keyboard.press('Enter');assert.ok(await time()>0);
 await page.emulateMedia({reducedMotion:'reduce'});await seek(13);assert.equal(await scene.locator('.moving-payload').count(),0);assert.doesNotMatch(await text(),/A · 已写入/);await page.emulateMedia({reducedMotion:'no-preference'});
 for(const width of[1440,1280,1024,768,390,320]){await page.setViewportSize({width,height:1000});for(const theme of['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);for(const mode of Object.keys(M.catalog)){await choose(mode);await seek(M.build(mode).duration*.6);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${width}/${theme}/${mode}: page overflow`);await checkBounds(`${width}/${theme}/${mode}`);}}}
 await page.setViewportSize({width:1440,height:1100});await page.evaluate(()=>document.documentElement.dataset.theme='light');await choose('kv');await seek(12.8);await page.locator('[data-comm-lab]').scrollIntoViewIfNeeded();if(retained)await page.screenshot({path:path.join(artifacts,'page-desktop.png')});
 await page.setViewportSize({width:390,height:900});await page.locator('.cf-controls').scrollIntoViewIfNeeded();if(retained)await page.screenshot({path:path.join(artifacts,'page-mobile.png')});
 const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto(base+'sglang/communication/');assert.equal(await plain.locator('[data-comm-lab]').isVisible(),false);await plain.locator('.cf-static:has(.comm-static) summary').click();assert.equal(await plain.locator('.comm-static').isVisible(),true);await plain.close();
 assert.deepEqual(errors,[]);console.log('PASS: 6 mechanisms, all 12 conditions, reversible frames, payload geometry, continuous full playback, causality, keyboard, 6 widths × 2 themes, reduced motion, no-JS, one player, no console/resource errors.');
}finally{
 if(server)await server.close();
 if(metadata){try{process.kill(metadata.pid,0);metadata.browserExited=false;}catch{metadata.browserExited=true;}metadata.profileRemoved=!fs.existsSync(metadata.profile);metadata.remaining=execFileSync('ps',['-axo','pid=,command='],{encoding:'utf8'}).split('\n').filter(s=>metadata.profile&&s.includes(metadata.profile));fs.writeFileSync(path.join(artifacts,'browser.json'),JSON.stringify(metadata,null,2));assert.equal(metadata.browserExited,true,'owned browser exited');assert.equal(metadata.profileRemoved,true,'owned profile removed');assert.deepEqual(metadata.remaining,[],'owned subprocesses gone');}
 if(!retained)fs.rmSync(artifacts,{recursive:true,force:true});
}
})().catch(e=>{console.error(e);process.exitCode=1;});
