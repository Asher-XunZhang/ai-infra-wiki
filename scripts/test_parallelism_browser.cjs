const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const M=require('../pages/sglang/parallelism/model.js');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 try{
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';await page.goto(base+'sglang/parallelism/');await page.waitForLoadState('load');
  const choose=async mode=>page.locator(`button[data-mode=${mode}]`).click();
  const advance=async()=>{await page.locator('#parallel-next').click();};
  const check=async mode=>{
   assert.equal(await page.locator('#parallel-scene').getAttribute('data-mode'),mode);
   assert.equal(await page.locator('[data-settings]:visible').count(),1,'only relevant controls shown');
   assert.equal(await page.locator('.parallel-details').evaluate(e=>e.open),false);
   assert.equal(await page.locator('.parallel-lab table').count(),0);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
   assert.match(await page.locator('#parallel-source').getAttribute('href'),new RegExp(M.revision));
  };
  for(const width of [1440,768,390,320]){
   await page.setViewportSize({width,height:1000});
   for(const theme of ['light','dark']){
    await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
    for(const mode of ['tp','pp','dp','ep','cp']){
     await choose(mode);assert.equal(await page.locator('#parallel-scene').getAttribute('data-step'),'0');
     await check(mode);
     while(!await page.locator('#parallel-next').isDisabled()){await advance();await check(mode);}
    }
   }
  }
  await choose('tp');await page.selectOption('#tp-kind','row');await advance();
  assert.deepEqual(await page.locator('.local-result .number-vector').allTextContents(),['35','725']);
  await advance();assert.equal(await page.locator('.parallel-result .number-vector').innerText(),'10\n30');
  assert.equal(await page.locator('#parallel-source').getAttribute('href'),M.url('row'));
  const positions=await page.locator('.parallel-bridge svg').evaluate(svg=>{svg.pauseAnimations();const p=svg.querySelector('circle');const at=t=>{svg.setCurrentTime(t);const m=p.getCTM();return {x:m.e,y:m.f}};return [at(.1),at(.6)];});
  assert.ok(positions[1].x>positions[0].x&&positions[1].y>positions[0].y,'message travels toward collective merge');
  await page.selectOption('#tp-kind','column');await page.locator('#tp-gather').uncheck();await advance();await advance();
  assert.equal(await page.locator('.parallel-bridge').count(),0);assert.equal(await page.locator('.shard-retained').isVisible(),true);
  await page.emulateMedia({reducedMotion:'reduce'});await page.locator('#tp-gather').check();await advance();await advance();
  assert.equal(await page.locator('animateMotion').count(),0);await page.emulateMedia({reducedMotion:'no-preference'});
  await choose('pp');await page.selectOption('#pp-count','1');await advance();
  assert.equal(await page.locator('.pipeline-cell.batch-1.now').count(),1);await advance();assert.equal(await page.locator('#parallel-next').isDisabled(),true);
  assert.match(await page.locator('.activation-handoff').innerText(),/M1[\s\S]*激活输入/);
  await page.selectOption('#pp-count','3');await page.locator('[data-tick="2"]').first().focus();await page.keyboard.press('Enter');
  assert.equal(await page.locator('.pipeline-cell.now:not(.bubble)').count(),2);
  await choose('dp');await advance();await advance();await advance();
  assert.deepEqual(await page.locator('.replica-requests').allTextContents(),['R1R3','R2']);
  await choose('ep');await page.locator('#ep-skew').check();await page.selectOption('#ep-token','4');
  assert.deepEqual(await page.locator('.expert.selected').evaluateAll(es=>es.map(e=>Number(e.dataset.expert))),[0,1]);
  await advance();assert.equal(await page.locator('.expert-token.arriving').count(),2);
  await advance();assert.equal(await page.locator('.expert-token.arriving').count(),0,'compute does not redispatch the token');
  await advance();
  assert.match(await page.locator('.weighted-result').innerText(),new RegExp('= '+M.experts(true,4).result+'$'));
  assert.deepEqual(await page.locator('.expert-load>div>b').allTextContents(),['4','2','1','1']);
  await choose('cp');await advance();assert.equal(await page.locator('.causal-matrix i:not(.masked)').count(),36);
  assert.deepEqual(await page.locator('.cp-work b').allTextContents(),['18 对 Q–K','18 对 Q–K']);
  await page.selectOption('#cp-layout','contiguous');await advance();
  assert.deepEqual(await page.locator('.cp-work b').allTextContents(),['10 对 Q–K','26 对 Q–K']);
  assert.equal(await page.locator('#parallel-next').isDisabled(),true,'counterfactual partition does not claim real KV path');
  await page.selectOption('#cp-layout','zigzag');await advance();await advance();assert.equal(await page.locator('.kv-reunited .number-vector b').count(),16);
  await choose('tp');await page.clock.install();await page.locator('#parallel-play').click();await page.clock.runFor(2100);
  assert.equal(await page.locator('#parallel-scene').getAttribute('data-step'),'1');
  await page.locator('#parallel-play').click();await page.clock.runFor(4100);assert.equal(await page.locator('#parallel-scene').getAttribute('data-step'),'1');
  await page.locator('#parallel-play').click();await choose('dp');await page.clock.runFor(4100);assert.equal(await page.locator('#parallel-scene').getAttribute('data-step'),'0');
  await page.locator('#parallel-next').focus();await page.keyboard.press('Enter');assert.equal(await page.locator('#parallel-scene').getAttribute('data-step'),'1');
  const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto(base+'sglang/parallelism/');
  assert.equal(await plain.locator('[data-parallel-lab]').isVisible(),false);await plain.locator('.topology-detail>summary').first().click();assert.equal(await plain.locator('.rank-grid').first().isVisible(),true);await plain.close();
  assert.deepEqual(errors,[]);console.log('PASS: five parallel mechanisms, parameters, numerical visuals, moving messages, source links, keyboard/playback, reduced motion, four widths × two themes, no-JS diagrams.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
