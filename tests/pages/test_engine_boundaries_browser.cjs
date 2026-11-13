// Regression: process ownership and statistical boundaries at chapter 2 releases.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {execFileSync}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=(process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/').replace(/\/$/,'');
const shots=process.env.ENGINE_TEST_SCREENSHOTS;
const alive=pid=>{try{process.kill(pid,0);return true;}catch{return false;}};
(async()=>{
 let server,browser,pid,profile;
 try{
  server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
  pid=server.process().pid;profile=server.process().spawnargs.find(x=>x.startsWith('--user-data-dir='))?.slice(16);
  console.log(JSON.stringify({session:'engine-boundaries',controller:process.pid,browser:pid,profile}));
  browser=await chromium.connect(server.wsEndpoint());
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const goto=async q=>{await page.goto(base+'/engine-comparison/'+q);await page.locator('.ia-page').waitFor();};
  const snapshot=async name=>{if(shots){fs.mkdirSync(shots,{recursive:true});await page.locator('.ia-boardGrid').screenshot({path:path.join(shots,name+'.png')});}};
  const noOverflow=async()=>assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'page overflow');
  await goto('?view=ownership&tp=2&step=2#process');
  assert.equal(await page.locator('.ia-chapterTab').count(),7);
  assert.deepEqual(await page.locator('[data-process-count]').evaluateAll(es=>es.map(e=>e.dataset.processCount)),['4','4']);
  assert.equal(await page.locator('.ia-board--sg .ib-process').count(),4);
  assert.equal(await page.locator('.ia-board--vl .ib-process').count(),4);
  assert.equal(await page.locator('[data-written=true]').count(),4);
  await snapshot('roles-tp2');
  await page.getByLabel('张量并行规模',{exact:true}).selectOption('1');
  assert.deepEqual(await page.locator('[data-process-count]').evaluateAll(es=>es.map(e=>e.dataset.processCount)),['3','2']);
  const core=page.locator('.ia-board--vl .ib-process').filter({has:page.locator('h4', {hasText:'EngineCore'})});
  assert.equal(await core.locator('.ib-gpu').count(),1,'uni GPU worker belongs to the Core process');
  assert.match(await page.locator('.ia-verdictCol--vl').innerText(),/进程内/);
  await snapshot('roles-tp1');
  await page.reload();await page.locator('.ib-ownership').first().waitFor();
  assert.equal(await page.getByLabel('张量并行规模',{exact:true}).inputValue(),'1');
  for(let s=0;s<4;s++){
   await page.locator('.ia-stepItem').nth(s).click();
   assert.equal(await page.locator('[data-written=true]').count(),s>=2?2:0);
   assert.equal(await page.locator('.ib-cells .is-reserved').count(),s===1?8:0);
  }
  await page.locator('.ia-stepItem').nth(2).click();
  await page.locator('[data-pulse-toggle]').click();
  const phase=()=>page.locator('.ib-path-pulse').first().evaluate(e=>getComputedStyle(e).strokeDashoffset);
  const frozen=await phase();await page.waitForTimeout(140);assert.equal(await phase(),frozen);
  await page.locator('[data-depth-choice=source]').click();
  assert.equal(await phase(),frozen);assert.equal(await page.locator('.ia-page').getAttribute('data-step'),'2');
  await page.locator('[data-depth-choice=modules]').click();
  await page.locator('[data-pulse-toggle]').click();await page.waitForTimeout(160);assert.notEqual(await phase(),frozen);
  await goto('?step=0#metrics');
  for(let step=0;step<4;step++){
   await page.locator('.ia-stepItem').nth(step).click();
   if(step<2){
    assert.deepEqual(await page.locator('.ib-metricScene').evaluateAll(es=>es.map(e=>e.dataset.endpoint)),['3','2']);
    assert.match(await page.locator('.ia-board--sg .ib-event-list .is-end').innerText(),/TM/);
    assert.match(await page.locator('.ia-board--vl .ib-event-list .is-end').innerText(),/AsyncLLM/);
   }
   if(step===2){
    assert.equal(await page.locator('.ib-event-list .is-measured').count(),4,'both first-wait intervals exclude requeue');
    assert.match(await page.locator('.ia-verdictCol--vl').innerText(),/完成时/);
   }
   await page.locator('[data-depth-choice=source]').click();
   assert.equal(await page.locator('.ia-readingDetails[open] .ib-reading-pair article').count(),2);
   await page.locator('[data-depth-choice=modules]').click();
   await snapshot('metric-'+step);
  }
  for(const n of [1,2,4]){
   await page.getByLabel('SGLang 合并 token 数',{exact:true}).selectOption(String(n));
   assert.match(await page.locator('.ia-board--vl .ib-handoff').innerText(),/核心输出 1 个 token/);
   assert.deepEqual(await page.locator('[data-samples]').evaluateAll(es=>es.map(e=>Number(e.dataset.samples))),[n,1]);
   assert.deepEqual(await page.locator('.ia-board--sg .ib-samples b').allInnerTexts(),Array(n).fill(200/n+' ms'));
   assert.deepEqual(await page.locator('.ib-counters dd').allInnerTexts(),[String(n),'0.2 s','1','0.2 s']);
  }
  await page.reload();await page.locator('[data-samples]').first().waitFor();
  assert.equal(await page.getByLabel('SGLang 合并 token 数',{exact:true}).inputValue(),'4');
  await page.locator('.ia-chapterTab').nth(0).click();
  await page.goBack();await page.locator('[data-samples]').first().waitFor();
  assert.equal(await page.locator('.ia-page').getAttribute('data-step'),'3');
  for(const lang of ['en','zh']){
   await page.locator('#lang-toggle').click();
   for(const topic of ['process','metrics']){
    await goto('?view=ownership&tp=2&step=2#'+topic);
    await noOverflow();
    assert.equal(await page.locator('.ia-board').count(),2);
   }
  }
  await goto('?view=ownership&step=2#process');
  for(const width of [1440,1024,768,390,320]){
   await page.setViewportSize({width,height:1000});await noOverflow();
   const bad=await page.locator('.ib-ownership,.ib-process,.ib-role,.ib-metricScene').evaluateAll(es=>es.filter(e=>e.scrollWidth>e.clientWidth+1).map(e=>e.className));
   assert.deepEqual(bad,[],`internal overflow at ${width}`);
  }
  await snapshot('roles-mobile');
  await page.setViewportSize({width:1440,height:1000});
  await goto('?step=0#metrics');await page.locator('.theme-toggle').click();await noOverflow();await snapshot('metric-theme');
  await goto('?depth=source#prefix');assert.match(await page.locator('.ib-reading').innerText(),/mem-fraction-static/);
  // Existing six topics remain usable after extending the topic registry.
  for(let chapter=0;chapter<6;chapter++){
   await page.locator('.ia-chapterTab').nth(chapter).click();
   if(chapter===0)await page.locator('[data-process-view=request]').click();
   await page.locator('[data-depth-choice=modules]').click();
   for(let step=0;step<4;step++){await page.locator('.ia-stepItem').nth(step).click();assert.equal(await page.locator('.ia-board svg.ia-svg').count(),2);}
  }
  // V2: the high-level boundary is preserved while module views expose the
  // real runner path. Reversing steps must restore the same block identities.
  const v2Cases=[['wire',2],['prefix',2],['prefix',3],['batch',2],['batch',3],['disagg',1],...Array.from({length:4},(_,i)=>['crosscut',i])];
  for(const lang of ['zh','en']){
   await page.evaluate(lang=>{if(atlasLocale.lang!==lang)atlasLocale.setLang(lang);},lang);
   for(const [topic,s] of v2Cases){
    await goto(`?depth=modules&step=${s}#${topic}`);
    const svg=page.locator('.rv-diagram');assert.equal(await svg.count(),1);
    const overflow=await svg.locator('.rv-node').evaluateAll(es=>es.flatMap(e=>{
     const b=e.querySelector('rect').getBBox();return [...e.querySelectorAll('text')].filter(t=>{const r=t.getBBox();return r.x<b.x||r.x+r.width>b.x+b.width-3;}).map(t=>t.textContent);
    }));assert.deepEqual(overflow,[],`${lang} ${topic}/${s} node text overflow`);
    const outside=await svg.locator('text').evaluateAll(es=>es.filter(e=>{const b=e.getBBox();return b.x<0||b.x+b.width>520||b.y+b.height>410;}).map(e=>e.textContent));
    assert.deepEqual(outside,[],`${lang} ${topic}/${s} text outside viewBox`);
    if(topic==='prefix')assert.match(await svg.textContent(),s===2?/31 × 4.*124/:/32 × 4.*128/);
    if(lang==='zh'&&[['prefix',2],['batch',3],['disagg',1],['crosscut',3]].some(([c,i])=>c===topic&&i===s))await snapshot(`v2-${topic}-${s}`);
    await page.locator('[data-depth-choice=source]').click();
    assert.equal(await page.locator('.rv-trace').count(),1);
    const links=await page.locator('.rv-trace a').evaluateAll(es=>es.map(e=>e.href));
    assert(links.every(s=>s.includes('/4ba787ec9959e566d3b587a796dff07673324da5/')));
    assert(!links.some(s=>s.includes('/gpu_model_runner.py')),'V1 fallback must not be the selected execution source');
    if(topic!=='crosscut'){
     await page.locator('[data-depth-choice=overview]').click();
     assert.equal(await page.locator('.rv-diagram').count(),0,'overview must remain a separate abstraction');
    }
   }
  }
  await page.evaluate(()=>atlasLocale.setLang('zh'));
  await goto('?depth=modules&step=2#prefix');
  const prefixFrame=await page.locator('.rv-diagram').textContent();
  await page.locator('.ia-stepItem').nth(3).click();await page.locator('.ia-stepItem').nth(2).click();
  assert.equal(await page.locator('.rv-diagram').textContent(),prefixFrame);
  await goto('?depth=modules&step=3#batch');
  await page.locator('[data-pulse-toggle]').click();
  const v2Phase=()=>page.locator('.rv-pulse').first().evaluate(e=>getComputedStyle(e).strokeDashoffset);
  const v2Frozen=await v2Phase();await page.waitForTimeout(160);assert.equal(await v2Phase(),v2Frozen);
  await page.locator('[data-depth-choice=source]').click();assert.equal(await v2Phase(),v2Frozen);
  await page.locator('[data-pulse-toggle]').click();await page.waitForTimeout(160);assert.notEqual(await v2Phase(),v2Frozen);
  for(const width of [1440,768,390,320]){
   await page.setViewportSize({width,height:1000});await noOverflow();
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('.theme-toggle').click();await snapshot('v2-batch-theme');
  for(const topic of ['process','wire','prefix','batch','disagg','crosscut','metrics']){
   await goto('?depth=source#'+topic);assert.equal(await page.locator('.rv-trace').count(),1);
   assert.match(await page.locator('.ia-board--vl .ia-boardRoute').textContent(),/V1 EngineCore.*ModelRunner V2/);
  }
  console.log('PASS: V2 ten module scenes in two languages, address/state reversibility, source pin/path, distinct depth views, shared pause clock, four widths and theme.');
  const reduced=await browser.newContext({reducedMotion:'reduce'});
  try{const p=await reduced.newPage();await p.goto(base+'/engine-comparison/?view=ownership&step=2#process');await p.locator('.ib-ownership').first().waitFor();
   assert.equal(await p.locator('.ib-path-pulse:visible').count(),0);
   await p.locator('.ia-stepItem').nth(0).click();assert.equal(await p.locator('[data-written=true]').count(),0);
   await p.goto(base+'/engine-comparison/?depth=modules&step=3#batch');await p.locator('.rv-diagram').waitFor();
   assert.equal(await p.locator('.rv-pulse:visible').count(),0);
  }finally{await reduced.close();}
  assert.deepEqual(errors,[]);
  console.log('PASS: process TP1/TP2 ownership/counts, reversible address/write state, shared pulse pause, detail/history/reload, four metric boundaries, ITL count/sum for 1/2/4 tokens, fixed source details, bilingual/5 widths/theme/reduced-motion, existing 24 stages.');
 }finally{
  try{if(browser)await browser.close();}finally{if(server)await server.close();}
  const remaining=profile?execFileSync('ps',['-axo','pid=,command='],{encoding:'utf8'}).split('\n').filter(l=>l.includes(profile)):[];
  assert(!pid||!alive(pid),'owned browser still alive');assert.equal(remaining.length,0,'profile processes remain');
  assert(!profile||!fs.existsSync(profile),'temporary profile remains');
  console.log(JSON.stringify({cleanup:{pid,profile,processes:remaining.length,stopped:true}}));
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
