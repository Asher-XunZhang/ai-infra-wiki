const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),os=require('node:os'),{execFileSync}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright'),F=require('../../pages/sglang/model-execution/flow-model.js');
(async()=>{
 const out=process.env.EXECUTION_SCREENSHOT_DIR||fs.mkdtempSync(path.join(os.tmpdir(),'execution-flow-qa-'));fs.mkdirSync(out,{recursive:true});
 const server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})}),proc=server.process(),profile=proc.spawnargs.find(x=>x.startsWith('--user-data-dir='))?.slice(16);
 const browser=await chromium.connect(server.wsEndpoint()),report={controller:process.pid,browser:proc.pid,profile},base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1200}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
  const seek=t=>page.locator('#mf-seek').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));},t);
  const go=async path=>{await page.goto(base+path);await page.waitForLoadState('load');assert.ok(await page.locator('[data-model-flow]').isVisible());assert.equal(await page.locator('.mechanism-film,.animation-speed').count(),0);};
  const fit=()=>page.locator('.mf-node').evaluateAll(nodes=>nodes.flatMap(n=>{const box=n.querySelector('.mf-node-bg').getBBox(),w=box.width;return [...n.querySelectorAll(':scope>text')].flatMap(t=>t.textContent&&(t.getBBox().width>w-20||t.getBBox().y+t.getBBox().height>box.height-2)?[{node:n.dataset.node,text:t.textContent,width:t.getBBox().width,available:w-20}]:[]);}));
  const deviceFit=()=>page.locator('svg[data-surface] text').evaluateAll(nodes=>nodes.flatMap(t=>{const b=t.getBBox(),p=t.parentElement,transform=p.transform?.baseVal?.consolidate()?.matrix,x=b.x+(transform?.e||0),y=b.y+(transform?.f||0);return x<0||x+b.width>1040||y<0||y+b.height>610?[{text:t.textContent,x,y,width:b.width}]:[];}));
  let overflow=[];
  await go('sglang/inference-overview/transformer.html');await page.locator('[data-model-flow]').screenshot({path:path.join(out,'model-initial.png')});
  for(const mode of ['prefill','decode']){
   await page.selectOption('#mf-mode',mode);const tl=F.build('model',{mode});
   for(const p of tl.phases){await seek(p.start+p.duration*.5);overflow.push(...await fit());assert.equal(await page.locator('[data-model-flow]').getAttribute('data-phase'),p.id);}
   await seek(tl.duration);assert.match(await page.locator('[data-node="token"]').textContent(),mode==='decode'?/y2.*尚无/:/y1.*尚无/);
   assert.equal(await page.locator('[data-node="layerkv"] .filled').count(),mode==='decode'?5:4);await seek(0);assert.equal(await page.locator('[data-node="layerkv"] .filled').count(),mode==='decode'?4:0);
  }
  await page.selectOption('#mf-mode','prefill');const model=F.build('model'),move=model.events.find(e=>e.edge==='pp0'),points=[];
  for(const u of [.2,.5,.8]){await seek(move.start+(move.end-move.start)*u);points.push(await page.locator('.mf-packet[visibility=visible]').first().getAttribute('transform'));}
  assert.equal(new Set(points).size,3);await seek(move.start+(move.end-move.start)*.5);assert.equal(await page.locator('.mf-packet[visibility=visible]').first().getAttribute('transform'),points[1]);
  await page.locator('[data-model-flow]').screenshot({path:path.join(out,'model-pp.png')});
  const condensed=model.phases.find(p=>p.id==='local-2');
  await seek(condensed.start+condensed.duration*.3);const before=await page.locator('[data-node=layerkv] .filled').count();
  const caption=await page.locator('#mf-payload').textContent();assert.ok(Number(await page.locator('[data-node=stage2] .mf-workbar').getAttribute('width'))>0);
  await seek(condensed.start+condensed.duration*.7);assert.ok(await page.locator('[data-node=layerkv] .filled').count()>before);assert.equal(await page.locator('#mf-payload').textContent(),caption);
  assert.match(await page.locator('[data-node=stage1]').textContent(),/本级层区间已完成/);assert.match(await page.locator('[data-node=stage3]').textContent(),/等待本次输入/);
  await page.locator('[data-model-flow]').screenshot({path:path.join(out,'model-condensed.png')});await seek(condensed.start+condensed.duration*.3);assert.equal(await page.locator('[data-node=layerkv] .filled').count(),before);
  const detailed=model.phases.find(p=>p.id==='attention');await seek(detailed.start+detailed.duration*.1);const detailCaption=await page.locator('#mf-payload').textContent();await seek(detailed.start+detailed.duration*.6);assert.equal(await page.locator('#mf-payload').textContent(),detailCaption);

  await page.clock.install();await page.locator('#mf-reset').click();await page.selectOption('#mf-speed','2');await page.locator('#mf-toggle').click();await page.clock.runFor(3000);const progressed=Number(await page.locator('[data-model-flow]').getAttribute('data-time'));assert.ok(progressed>5);
  await page.locator('#mf-toggle').click();const paused=Number(await page.locator('[data-model-flow]').getAttribute('data-time'));await page.clock.runFor(1000);assert.equal(Number(await page.locator('[data-model-flow]').getAttribute('data-time')),paused);await page.locator('#mf-toggle').click();await page.clock.runFor(model.duration*500+100);assert.equal(await page.locator('#mf-toggle').getAttribute('aria-pressed'),'false');assert.match(await page.locator('[data-node=token]').textContent(),/y1.*尚无/);
  await page.locator('#mf-toggle').click();await page.selectOption('#mf-mode','decode');assert.equal(await page.locator('[data-model-flow]').getAttribute('data-time'),'0.000');assert.equal(await page.locator('#mf-toggle').getAttribute('aria-pressed'),'false');
  await go('sglang/model-execution/');
  for(const view of ['chain','graph','hardware']){
   await page.locator(`[data-exec-view=${view}]`).click();assert.equal(await page.locator('[data-model-flow]').getAttribute('data-time'),'0.000');const t=F.build(view);
   for(const p of t.phases){await seek(p.start+p.duration*.85);overflow.push(...await fit());}
   await page.locator('[data-model-flow]').screenshot({path:path.join(out,view+'-end.png')});
   await page.locator('#mf-reset').click();await page.selectOption('#mf-speed','2');await page.locator('#mf-toggle').click();await page.clock.runFor(t.duration*500+100);assert.equal(await page.locator('#mf-toggle').getAttribute('aria-pressed'),'false');
  }
  await page.locator('[data-exec-view=chain]').click();
  for(const origin of ['prefill','decode']){
   await page.selectOption('#execution-input',origin);const t=F.build('chain',{origin});assert.equal(await page.locator('[data-model-flow]').getAttribute('data-time'),'0.000');
   for(const id of ['resolve','bind','load','metadata','launch','project','compute','result','return']){
    const p=t.phases.find(p=>p.id===id);await seek(p.start+p.duration*.6);overflow.push(...await deviceFit());
    await page.locator('[data-model-flow]').screenshot({path:path.join(out,`device-${origin}-${id}.png`)});
    if(id==='resolve'){assert.equal(await page.locator('[data-transfer=h2d]').count(),origin==='prefill'?5:0);assert.equal(await page.locator('[data-resident=weights]').count(),1);}
    if(id==='bind')assert.equal(await page.locator('[data-transfer=reference]').count(),1);
    if(id==='load')assert.equal(await page.locator('[data-transfer=d2d]').count(),origin==='prefill'?15:6);
    if(id==='metadata')assert.equal(await page.locator('[data-mapping]').count(),origin==='prefill'?5:2);
   }
   const transfer=t.phases.find(p=>p.id==='resolve'),kind=origin==='prefill'?'h2d':'gpu-reuse',motion=[];
   for(const u of [.25,.45,.60]){await seek(transfer.start+transfer.duration*u);motion.push(await page.locator(`[data-transfer=${kind}]`).first().getAttribute('transform'));}
   assert.equal(new Set(motion).size,3);await seek(transfer.start+transfer.duration*.45);assert.equal(await page.locator(`[data-transfer=${kind}]`).first().getAttribute('transform'),motion[1]);
   assert.equal(await page.locator('[data-origin-token]').count(),origin==='prefill'?5:2,'CPU request records remain present during copies');
   const weights=await page.locator('[data-resident=weights]').evaluate(e=>[e.getAttribute('x'),e.getAttribute('y')]);
   await seek(t.duration);assert.deepEqual(await page.locator('[data-resident=weights]').evaluate(e=>[e.getAttribute('x'),e.getAttribute('y')]),weights,'weights remain resident');assert.equal(await page.locator('[data-cpu-result]').count(),2);assert.equal(await page.locator('[data-slot][data-filled=true]').count(),origin==='prefill'?5:7);
   await seek(0);assert.equal(await page.locator('[data-cpu-result]').count(),0);assert.equal(await page.locator('[data-slot][data-filled=true]').count(),origin==='prefill'?0:5);
  }
  await page.locator('[data-exec-view=graph]').click();
  for(const count of [1,3,4,5])for(const condition of ['ready','off','override']){
   await page.selectOption('#graph-count',String(count));await page.selectOption('#graph-condition',condition);const t=F.build('graph',{count,condition});await seek(t.duration);overflow.push(...await fit());assert.equal(await page.locator('[data-valid-output]').count(),count);overflow.push(...await deviceFit());
   if(count===3&&condition==='ready'){assert.equal(await page.locator('[data-pad]').count(),1);await page.locator('[data-model-flow]').screenshot({path:path.join(out,'graph-padding.png')});}
  }
  await page.locator('[data-exec-view=hardware]').click();for(const rows of [1,16,256])for(const upgrade of ['base','compute','memory']){
   await page.selectOption('#matrix-rows',String(rows));await page.selectOption('#hardware-upgrade',upgrade);await seek(F.build('hardware').duration);overflow.push(...await fit());assert.match(await page.locator('[data-node=bound]').textContent(),rows===1?/访存项/:rows===256?/计算项/:/项更长/);
  }
  report.textOverflows=overflow;assert.deepEqual(overflow,[]);
  await page.locator('[data-exec-view=chain]').click();await page.locator('#mf-seek').focus();await page.keyboard.press('ArrowRight');assert.ok(Number(await page.locator('#mf-seek').inputValue())>0);
  await page.emulateMedia({reducedMotion:'reduce'});const launch=F.build('chain').phases.find(p=>p.id==='launch');await seek(launch.start+launch.duration*.5);assert.equal(await page.locator('.mf-packet[visibility=visible]').count(),0);assert.ok(await page.locator('[data-transfer=submit]').count());await page.emulateMedia({reducedMotion:'no-preference'});
  for(const [file,view]of [['sglang/inference-overview/transformer.html','model'],['sglang/model-execution/','chain']]){
   await go(file);for(const width of [1440,820,768,390,320])for(const theme of ['light','dark']){
    await page.setViewportSize({width,height:1200});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await seek(view==='model'?11:27);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    if(width===1440||width===390)await page.locator('[data-model-flow]').screenshot({path:path.join(out,`${view}-${width}-${theme}.png`)});
   }
  }
  const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto(base+'sglang/inference-overview/transformer.html');assert.ok(await plain.locator('.implementation-map').isVisible());assert.equal(await plain.locator('[data-model-flow]').isVisible(),false);await plain.goto(base+'sglang/model-execution/');await plain.locator('.topology-detail>summary').first().click();assert.ok(await plain.locator('.exec-flow').first().isVisible());await plain.close();assert.deepEqual(errors,[]);
  assert.deepEqual(overflow,[]);report.result='PASS';report.layouts='2 pages × 5 widths × light/dark';report.motion=points;console.log('PASS: two lessons, four timelines, CPU/GPU copies and references, PP/MLA state, Graph padding/fallback, hardware cases, full playback, reverse seek, controls, themes and no-JS.');
 }finally{
  await browser.close();await server.close();let alive=false;try{process.kill(proc.pid,0);alive=true;}catch{}
  report.cleanup={browserExited:!alive,profileRemoved:profile?!fs.existsSync(profile):null,remaining:profile?execFileSync('ps',['-axo','pid=,ppid=,command='],{encoding:'utf8'}).split('\n').filter(line=>line.includes(profile)):[]};fs.writeFileSync(path.join(out,'review.json'),JSON.stringify(report,null,2));assert.equal(alive,false);assert.deepEqual(report.cleanup.remaining,[]);
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
