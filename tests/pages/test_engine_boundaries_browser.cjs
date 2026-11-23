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
  const sourceToggle=page.locator('[data-source-toggle]');
  const noOverflow=async()=>assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'page overflow');
  // Archify geometry remains editable upstream; the lesson owns selection,
  // process assumptions and the one pulse clock. Native IDs must survive import.
  await goto('#process');
  assert.equal(await page.locator('.ia-page').getAttribute('data-process-view'),'roles');
  assert.equal(await page.locator('.ir-map').count(),2);
  assert.equal(await page.locator('.ia-stepRail').isVisible(),false);
  const vlRoles=page.locator('.ia-board--vl .ir-map');
  for(const id of ['core','scheduler','executor','worker','runner','model','attention','kernel']){
   const node=vlRoles.locator(`[data-node-id="${id}"]`);
   assert.equal(await node.count(),1);assert(await node.getAttribute('aria-label'));
  }
  assert.equal(await page.locator('.ia-board--sg [data-node-id="executor"]').count(),0);
  await vlRoles.locator('[data-node-id="executor"]').focus();await page.keyboard.press('Enter');
  assert.match(await vlRoles.locator('.ac-selection').innerText(),/uni.*mp.*execute_model/s);
  assert.match(await vlRoles.locator('.ac-selection a').getAttribute('href'),/4ba787ec.*executor\/abstract.py/);
  await sourceToggle.click();assert.equal(await page.locator('.ir-map [data-module-role] a').count(),17);
  assert.equal(await page.locator('.ia-source-step').count(),0);
  await vlRoles.locator('[data-locate-role="executor"]').click();
  assert.equal(await page.locator('.ia-page').getAttribute('data-process-view'),'request');
  assert.equal(await page.locator('.ia-board--vl [data-node-id="executor"].is-selected').count(),1);
  await page.locator('.ia-stepItem').nth(2).click();await page.locator('[data-pulse-toggle]').click();
  const roleTime=await page.locator('.ia-page').getAttribute('data-pulse-time');
  await page.locator('button[data-process-view="roles"]').click();
  await vlRoles.locator('[data-node-id="runner"]').click();
  await vlRoles.locator('[data-locate-role="runner"]').click();
  assert.equal(await page.locator('.ia-page').getAttribute('data-step'),'2');
  assert.equal(await page.locator('.ia-page').getAttribute('data-pulse-time'),roleTime);
  assert.equal(await page.locator('[data-pulse-toggle]').getAttribute('aria-pressed'),'false');
  assert.equal(await page.locator('.ia-board--vl [data-node-id="runner"].is-selected').count(),1,'locate the selected logical role in its process');
  for(const lang of ['en','zh']){
   await page.evaluate(lang=>atlasLocale.setLang(lang),lang);
   await page.locator('button[data-process-view="roles"]').click();
   for(const width of [1440,390]){
    await page.setViewportSize({width,height:1000});await noOverflow();
    const boxes=await page.locator('.ia-board').evaluateAll(es=>es.map(e=>{const r=e.getBoundingClientRect();return{y:r.y,bottom:r.bottom};}));
    if(width===390){
     assert(boxes[1].y>=boxes[0].bottom,'mobile duties read vertically');
     assert.equal(await page.locator('.ac-reader[open]').count(),2,'mobile responsibilities read without hover');
     assert.deepEqual(await page.locator('.ac-reader,.ac-note,.ac-selection').evaluateAll(es=>es.filter(e=>e.scrollWidth>e.clientWidth+1).map(e=>e.className)),[],'responsibilities must not be clipped');
    }
    else assert(Math.abs(boxes[0].y-boxes[1].y)<1);
   }
  }
  await snapshot('module-roles-mobile');await page.setViewportSize({width:1440,height:1000});
  await goto('?view=ownership&tp=2&step=2#process');
  assert.deepEqual(await page.locator('[data-process-count]').evaluateAll(es=>es.map(e=>e.dataset.processCount)),['4','4']);
  assert.equal(await page.locator('[data-node-id^="gpu"]').count(),4);
  assert.equal(await page.locator('[data-node-id^="gpu"]').filter({hasText:'已写入'}).count(),4);
  await page.getByLabel('张量并行规模',{exact:true}).selectOption('1');
  assert.deepEqual(await page.locator('[data-process-count]').evaluateAll(es=>es.map(e=>e.dataset.processCount)),['3','2']);
  await page.reload();await page.locator('[data-process-count]').first().waitFor();
  assert.equal(await page.getByLabel('张量并行规模',{exact:true}).inputValue(),'1');
  for(const s of [0,1,2,3,1,0]){
   await page.locator('.ia-stepItem').nth(s).click();
   const labels=await page.locator('[data-node-id^="gpu"] [data-detail="context"]').allTextContents();
   assert(labels.every(t=>s>=2?t.includes('已写入'):s===1?t.includes('尚未写'):t.includes('尚未分配')),labels.join(','));
  }
  const phase=()=>page.locator('.ac-pulse').first().evaluate(e=>getComputedStyle(e).strokeDashoffset);
  await goto('?wire=sequence&step=0#wire');
  assert.equal(await page.locator('[data-archify="messages-sg"]').count(),1);
  let moving=await phase();await page.waitForTimeout(160);assert.notEqual(await phase(),moving);
  await page.locator('[data-pulse-toggle]').click();
  const frozen=await phase();await page.waitForTimeout(160);assert.equal(await phase(),frozen);
  await page.locator('[data-wire-view="topology"]').click();
  assert.equal(await page.locator('[data-archify="wire-sg"]').count(),1);
  assert.equal(await phase(),frozen);await sourceToggle.click();assert.equal(await phase(),frozen);
  await page.locator('.ia-stepItem').nth(3).click();
  assert.equal(await page.locator('.ia-board--sg path[data-edge-id="rpc-request"].is-active').count(),1);
  await page.locator('[data-wire-view="sequence"]').click();
  assert.equal(await page.locator('[data-archify="rpc-messages-sg"]').count(),1);
  assert.equal(await page.locator('[data-archify="coord-messages-vl"]').count(),1);
  assert.match(await page.locator('.ia-board--vl').innerText(),/DP > 1/);
  await page.reload();await page.locator('[data-archify="rpc-messages-sg"]').waitFor();
  assert.equal(await page.locator('[data-wire-view="sequence"]').getAttribute('aria-pressed'),'true');
  // Every inline marker is prefixed per SVG; two engines cannot steal arrows.
  assert.deepEqual(await page.locator('.ac-native').evaluateAll(es=>es.flatMap(svg=>[...svg.querySelectorAll('[marker-end]')].filter(n=>!svg.querySelector('[id="'+n.getAttribute('marker-end').slice(5,-1)+'"]')).map(n=>n.getAttribute('marker-end')))),[]);
  const sgBoard=page.locator('.ia-board--sg');
  await goto('?depth=overview&step=1#disagg');
  assert.equal(await page.locator('[data-archify^="pd-overview"] [data-node-id]').count(),6);
  await page.locator('[data-depth-choice="modules"]').click();
  assert.equal(await page.locator('[data-archify^="kv-flow"] [data-node-id]').count(),12);
  for(const id of ['bind','readable'])assert.equal(await page.locator(`[data-edge-id="${id}"] + .ac-pulse`).count(),0,'references and dependencies are static');
  assert.equal(await page.locator('[data-edge-id="payload"] + .ac-pulse').count(),2);
  await page.locator('.ia-board--vl .ac-local-detail>summary').click();
  await page.locator('[data-pulse-toggle]').click();
  const loadPhase=()=>page.locator('.rv-pulse').first().evaluate(e=>getComputedStyle(e).strokeDashoffset);
  const loadFrozen=await loadPhase();
  for(const mode of ['async','sync']){
   await page.locator(`[data-load-choice="${mode}"]`).click();
   assert.equal(await page.locator('.rv-diagram').getAttribute('data-load-mode'),mode);
   assert.equal(await loadPhase(),loadFrozen);
  }
  await goto('?step=2#disagg');
  assert.equal(await page.locator('[data-archify="restore-sg"]').count(),1);
  assert.equal(await page.locator('[data-depth-choice]').count(),0,'one restoration view avoids a duplicate overview');
  await goto('?step=2#crosscut');
  assert.equal(await page.locator('.ac-pulse').count(),0,'state transitions do not claim byte transfers');
  assert.match(await sgBoard.locator('.ac-ledger').innerText(),/尚未分配 KV/);
  assert.equal(await page.locator('[data-pulse-toggle]').isDisabled(),true);
  await goto('?step=3#crosscut');
  await page.locator('[data-pulse-toggle]').click();const graphFrozen=await phase();
  for(const [side,modes] of [['sg',['decode','piece','eager']],['vl',['FULL','PIECEWISE','NONE']]]){
   const board=page.locator('.ia-board--'+side);
   for(const mode of modes){
    await board.locator(`[${side==='sg'?'data-sg-graph-choice':'data-graph-choice'}="${mode}"]`).click();
    assert.equal(await board.locator('.ac-pulse').count(),2,'only one selected branch and its output flow');
    assert.deepEqual(await board.locator('path.ac-edge.is-active').evaluateAll(es=>es.map(e=>e.dataset.edgeId).sort()),['choose-'+mode,mode+'-output'].sort());
    assert.equal(await phase(),graphFrozen);await sourceToggle.click();assert.equal(await phase(),graphFrozen);await sourceToggle.click();
   }
  }
  await page.locator('#lang-toggle').click();assert.equal(await page.locator('[data-graph-choice="NONE"]').getAttribute('aria-pressed'),'true');
  await page.locator('#lang-toggle').click();await page.locator('[data-pulse-toggle]').click();await page.waitForTimeout(160);assert.notEqual(await phase(),graphFrozen);
  for(const query of ['?depth=overview&step=1#prefix','?step=0#batch','?step=1#batch','?step=3#disagg','?step=0#metrics','?view=ownership&step=0#process']){
   await goto(query);assert.equal(await page.locator('[data-pulse-toggle]').isDisabled(),true,query);
   assert((await page.locator('[data-flow-hint]').textContent()).length>15);
  }
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
   await sourceToggle.click();
   assert.equal(await page.locator('.ia-source-step').count(),2);
   assert.equal(await page.locator('.ia-readingDetails[open]').count(),0,'open only the in-diagram annotations');
   await sourceToggle.click();
   await snapshot('metric-'+step);
  }
  await page.locator('.ia-readingDetails>summary').click();
  await sourceToggle.click();
  for(const n of [1,2,4]){
   await page.getByLabel('SGLang 合并 token 数',{exact:true}).selectOption(String(n));
   assert.match(await page.locator('.ia-board--vl .ib-handoff').innerText(),/核心输出 1 个 token/);
   assert.deepEqual(await page.locator('[data-samples]').evaluateAll(es=>es.map(e=>Number(e.dataset.samples))),[n,1]);
   assert.deepEqual(await page.locator('.ia-board--sg .ib-samples b').allInnerTexts(),Array(n).fill(200/n+' ms'));
   assert.deepEqual(await page.locator('.ib-reading-pair article').first().locator('math mn').allTextContents(),['200',String(n),String(200/n)]);
   assert.deepEqual(await page.locator('.ib-counters dd').allInnerTexts(),[String(n),'0.2 s','1','0.2 s']);
   assert.deepEqual(await page.locator('.ia-source-step-result').allTextContents(),[`${n} × ${200/n} ms`,'1 × 200 ms']);
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
  await goto('?depth=source#prefix');
  assert.equal(await sourceToggle.getAttribute('aria-expanded'),'true','legacy depth=source opens inline annotations');
  assert.equal(await page.locator('.ia-page').getAttribute('data-depth'),'modules');
  await page.locator('.ia-readingDetails>summary').click();
  assert.match(await page.locator('.ib-reading').innerText(),/mem-fraction-static/);
  // The current step has useful annotations for both engines in every topic.
  const topics=['process','wire','prefix','batch','crosscut','disagg','metrics'];
  const threeLayerTopics=['prefix','disagg'];
  for(const topic of topics){
   await goto('?view=request&depth=modules#'+topic);
   assert.equal(await page.locator('[data-depth-choice]').count(),threeLayerTopics.includes(topic)?2:0);
   assert.equal(await page.locator('.ia-source-step').count(),0);
   await sourceToggle.click();
   const observations=[];
   for(let step=0;step<4;step++){
    await page.locator('.ia-stepItem').nth(step).click();
    assert.equal(await page.locator('.ia-board').count(),2);
    assert.equal(await page.locator('.ia-board .ia-source-step').count(),2);
    const anchors=await page.locator('.ia-source-step-head a').evaluateAll(links=>links.map(link=>link.href));
    assert.match(anchors[0],/sgl-project\/sglang\/blob\/297e54fd105e974cdf401e5b4d20253a22c16f90\//);
    assert.match(anchors[1],/vllm-project\/vllm\/blob\/4ba787ec9959e566d3b587a796dff07673324da5\//);
    observations.push((await page.locator('.ia-source-step').allTextContents()).join(' '));
   }
   assert.equal(new Set(observations).size,4,`${topic} annotations must track the current step`);
  }
  // V2: the high-level boundary is preserved while module views expose the
  // real runner path. Reversing steps must restore the same block identities.
  const v2Cases=[['prefix',2],['prefix',3],['batch',2],['batch',3]];
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
    await sourceToggle.click();
    assert.equal(await svg.count(),1,'annotations keep the existing V2 module diagram');
    assert.equal(await page.locator('.ia-board--vl .ia-source-step').count(),1);
    const links=await page.locator('.ia-board--vl .ia-source-step a').evaluateAll(es=>es.map(e=>e.href));
    assert(links.every(s=>s.includes('/4ba787ec9959e566d3b587a796dff07673324da5/')));
    assert(!links.some(s=>s.includes('/gpu_model_runner.py')),'V1 fallback must not be the selected execution source');
    if(topic!=='crosscut'){
     await page.locator('[data-depth-choice=overview]').click();
     assert.equal(await page.locator('.rv-diagram').count(),0,'overview must remain a separate abstraction');
     assert.equal(await sourceToggle.getAttribute('aria-expanded'),'true','annotation remains an independent choice');
     assert.equal(await page.locator('.ia-source-step').count(),2);
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
  const serializedDiagram=await page.locator('.rv-diagram').evaluate(svg=>svg.outerHTML);
  await sourceToggle.click();assert.equal(await v2Phase(),v2Frozen);
  assert.equal(await page.locator('.rv-diagram').evaluate(svg=>svg.outerHTML),serializedDiagram);
  assert.equal(await page.locator('.ia-source-step').count(),2);
  await snapshot('source-batch-desktop');
  await page.setViewportSize({width:390,height:1000});
  await snapshot('source-batch-mobile');
  await page.locator('[data-pulse-toggle]').click();await page.waitForTimeout(160);assert.notEqual(await v2Phase(),v2Frozen);
  for(const width of [1440,768,390,320]){
   await page.setViewportSize({width,height:1000});await noOverflow();
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('.theme-toggle').click();await snapshot('v2-batch-theme');
  await goto('?depth=overview&source=1&step=1#prefix');
  await page.reload();await page.locator('.ia-source-step').first().waitFor();
  assert.equal(await page.locator('.ia-page').getAttribute('data-depth'),'overview');
  assert.equal(await page.locator('.ia-page').getAttribute('data-step'),'1');
  assert.match(await page.locator('.ia-source-step-result').first().innerText(),/8 \/ 12/);
  await page.locator('.ia-chapterTab').filter({hasText:'调度与批处理'}).click();
  await page.goBack();await page.waitForSelector('.ia-page[data-chapter=prefix][data-step="1"]');
  assert.equal(await sourceToggle.getAttribute('aria-expanded'),'true');
  await sourceToggle.click();assert.equal(await page.locator('.ia-source-step').count(),0);
  console.log('PASS: preserved V2 input/async scenes in two languages, address/state reversibility, source pin/path, distinct diagram views, per-step pinned source annotations, shared pause clock, four widths and theme.');
  const reduced=await browser.newContext({reducedMotion:'reduce'});
  try{const p=await reduced.newPage();await p.goto(base+'/engine-comparison/?view=ownership&step=2#process');await p.locator('[data-process-count]').first().waitFor();
   assert.equal(await p.locator('.ac-pulse:visible').count(),0);
   await p.locator('.ia-stepItem').nth(0).click();assert.equal(await p.locator('[data-node-id^=gpu]').filter({hasText:'已写入'}).count(),0);
   await p.goto(base+'/engine-comparison/?depth=modules&step=3#batch');await p.locator('.rv-diagram').waitFor();
   assert.equal(await p.locator('.rv-pulse:visible').count(),0);
   await p.goto(base+'/engine-comparison/?step=3#crosscut');await p.locator('[data-archify=graph-vl]').waitFor();
   await p.locator('[data-graph-choice="PIECEWISE"]').click();
   assert.equal(await p.locator('[data-graph-choice=PIECEWISE]').getAttribute('aria-pressed'),'true');
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
