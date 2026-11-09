const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const M=require('../../pages/sglang/loop-journey/model.js');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
const shots=process.env.LOOP_SCREENSHOT_DIR;
(async()=>{
 const server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})}),owned=server.process();
 const identity={session:'loop-journey-review',controller:process.pid,browser:owned.pid,profile:owned.spawnargs.find(a=>a.startsWith('--user-data-dir='))};console.log(JSON.stringify(identity));
 const browser=await chromium.connect(server.wsEndpoint());
 if(shots){fs.mkdirSync(shots,{recursive:true});fs.writeFileSync(path.join(shots,'session.json'),JSON.stringify(identity));}
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
  const root=page.locator('[data-loop-journey]');
  for(const mode of Object.keys(M.configs)){
   await page.goto(base+'sglang/loop-journey/index.html#'+mode);
   await page.waitForFunction(m=>document.querySelector('[data-loop-journey]').dataset.mode===m,mode);
   assert.equal(await root.getAttribute('data-step'),'0');
   assert.equal(await page.locator('[data-loop-reading]:visible').count(),1);
   const stroke=id=>page.locator(`[data-track-event="${id}"] .lj-track-slot`).evaluate(e=>getComputedStyle(e).stroke);
   const batchPeers=mode==='pp'?['submit01','gpu01','proxy1','gpu11','out1','process01','process11']:['submit0','forward0','copy0','process0'];
   assert.equal(new Set(await Promise.all(batchPeers.map(stroke))).size,1,'a batch keeps its color across owners');
   assert.notEqual(await stroke(batchPeers[0]),await stroke(mode==='pp'?'gpu02':'forward1'));
   await page.locator('#lj-color').selectOption('loop');
   if(mode==='overlap'){assert.equal(await stroke('copy0'),await stroke('submit0'));assert.notEqual(await stroke('copy0'),await stroke('process0'));}
   if(mode==='pp'){assert.notEqual(await stroke('submit02'),await stroke('submit12'));assert.equal(await stroke('out1'),await stroke('postOut1'));assert.notEqual(await stroke('out1'),await stroke('process01'));}
   await page.locator('#lj-color').selectOption('batch');
   for(let i=0;i<M.configs[mode].steps.length;i++){
    assert.equal(await root.getAttribute('data-step'),String(i));
    assert.equal(await page.locator('#lj-title').textContent(),M.configs[mode].steps[i].title);
    assert.equal(Number(await page.locator('#lj-tracks-svg').getAttribute('data-clock')),M.execution(mode).times[i]);
    assert.match(await page.locator('#lj-source').getAttribute('href'),new RegExp(M.revision));
    const clipping=await page.locator('.lj-action').evaluateAll(nodes=>nodes.filter(e=>e.clientWidth>0&&(e.scrollHeight>e.clientHeight+1||e.scrollWidth>e.clientWidth+1)).map(e=>e.textContent));assert.deepEqual(clipping,[]);
    if(i===Math.floor(M.configs[mode].steps.length/2)&&shots){await page.evaluate(()=>scrollTo({top:document.querySelector('[data-loop-journey]').getBoundingClientRect().top+scrollY-85,behavior:'instant'}));await page.screenshot({path:path.join(shots,mode+'-desktop.png')});await page.locator('.lj-tracks').screenshot({path:path.join(shots,mode+'-tracks.png')});}
    if(i<M.configs[mode].steps.length-1)await page.locator('#lj-next').click();
   }
   assert.equal(await page.locator('#lj-next').isDisabled(),true);
   await page.locator('#lj-hold').check();assert.equal(await root.getAttribute('data-step'),'0');
   const barrier=M.configs[mode].steps.findIndex(s=>s.id===M.configs[mode].barrier);
   for(let i=0;i<barrier;i++)await page.locator('#lj-next').click();
   assert.equal(await root.getAttribute('data-blocked'),'true');assert.equal(await page.locator('#lj-next').isDisabled(),true);
   const held=await page.locator('#lj-ledger').innerText();await page.waitForTimeout(350);assert.equal(await page.locator('#lj-ledger').innerText(),held);
   const heldClock=await page.locator('#lj-tracks-svg').getAttribute('data-clock');
   for(const color of ['loop','kind','batch']){
    await page.locator('#lj-color').selectOption(color);
    assert.equal(await root.getAttribute('data-blocked'),'true');assert.equal(await root.getAttribute('data-step'),String(barrier));
    assert.equal(await page.locator('#lj-tracks-svg').getAttribute('data-clock'),heldClock);assert.equal(await page.locator('#lj-ledger').innerText(),held);
    assert.ok((await page.locator('.lj-track-event.wait .lj-track-slot').evaluateAll(es=>es.map(e=>getComputedStyle(e).strokeDasharray))).every(v=>v!=='none'));
   }
   // Jumping to a future event cannot bypass unresolved dependency.
   await page.locator('.lj-event-detail>summary').click();await page.locator('.lj-column').last().click();assert.equal(await root.getAttribute('data-step'),String(barrier));
   if(shots)await root.screenshot({path:path.join(shots,mode+'-blocked.png')});
   await page.locator('#lj-release').click();assert.equal(await root.getAttribute('data-blocked'),'false');
   await page.locator('#lj-color').selectOption('loop');assert.equal(await root.getAttribute('data-blocked'),'false');
   await page.locator('#lj-next').click();assert.equal(await root.getAttribute('data-step'),String(barrier+1));
   await page.locator('#lj-reset').click();await page.locator('#lj-hold').uncheck();await page.locator('.lj-event-detail>summary').click();
  }
  await page.locator('button[data-mode=pp]').click();
  await page.locator('#lj-view').selectOption('loop');await page.locator('#lj-focus').selectOption('P0:L1');
  assert.equal(await page.locator('#lj-color').inputValue(),'loop');
  assert.equal(await page.locator('[data-track-event=out1]').getAttribute('data-color-group'),'P1:L0');
  assert.equal(await page.locator('[data-track-event=out1]').evaluate(e=>e.classList.contains('is-muted')),false,'incoming message stays visible');
  const bright=await page.locator('.lj-action:not(.is-muted)').allTextContents();assert.ok(bright.some(t=>t.includes('M1'))&&bright.some(t=>t.includes('M2')));
  await page.locator('#lj-view').selectOption('batch');await page.locator('#lj-focus').selectOption('M1');
  assert.equal(await page.locator('#lj-color').inputValue(),'batch');
  await page.locator('#lj-color').selectOption('kind');assert.equal(await page.locator('#lj-focus').inputValue(),'M1');assert.equal(await page.locator('#lj-view').inputValue(),'batch');
  await page.locator('#lj-color').selectOption('batch');
  assert.ok((await page.locator('.lj-action:not(.is-muted)').allTextContents()).every(t=>t.includes('M1')));
  await page.locator('[data-track-event=gpu11]').focus();await page.keyboard.press('Enter');assert.match(await page.locator('#lj-track-detail').innerText(),/M1.*后段/);
  await page.locator('button[data-mode=overlap]').click();assert.equal(await root.getAttribute('data-step'),'0');
  await page.goBack();assert.equal(await root.getAttribute('data-mode'),'pp');assert.equal(await root.getAttribute('data-step'),'0');
  await page.locator('#lj-next').focus();await page.keyboard.press('Enter');assert.equal(await root.getAttribute('data-step'),'1');
  await page.locator('a[href="#objects"]').click();assert.equal(await root.getAttribute('data-mode'),'pp');
  // Short windows, narrow screens, themes, and readable scroll containment.
  for(const width of [1440,1024,768,390,320]){
   await page.setViewportSize({width,height:width===1440?700:900});
   for(const theme of ['light','dark']){
    await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
    for(const mode of ['normal','overlap','pp']){
     await page.locator(`button[data-mode=${mode}]`).click();
     assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${width}/${theme}/${mode}`);
     assert.ok(await page.locator('#lj-tracks-svg').evaluate(e=>e.getBoundingClientRect().width>=960));
     for(const color of ['batch','loop','kind']){
      await page.locator('#lj-color').selectOption(color);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${width}/${theme}/${mode}/${color}`);
      const palette=await page.locator('#lj-track-legend span').evaluateAll(es=>es.map(e=>getComputedStyle(e).borderLeftColor));
      assert.equal(new Set(palette).size,palette.length,'legend colors are distinct');
      const clippedLabels=await page.locator('.lj-track-event').evaluateAll(es=>es.filter(e=>{const t=e.querySelector('text'),r=e.querySelector('rect');if(!t)return false;const a=t.getBBox(),b=r.getBBox();return a.x<b.x-1||a.x+a.width>b.x+b.width+1;}).map(e=>e.dataset.trackEvent));assert.deepEqual(clippedLabels,[]);
     }
     await page.locator('#lj-color').selectOption('loop');
     const outside=await page.locator('.lj-toolbar button,.lj-now,.lj-ledger').evaluateAll(nodes=>nodes.filter(e=>{if(!e.checkVisibility())return false;const r=e.getBoundingClientRect();return r.left<0||r.right>innerWidth+1;}).map(e=>e.id));assert.deepEqual(outside,[]);
     if(shots&&[390,320].includes(width)&&mode==='pp'&&theme==='dark'){await page.locator('#journey-lab').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(shots,`pp-${width}-${theme}.png`)});}
    }
   }
  }
  await page.emulateMedia({reducedMotion:'reduce'});assert.equal(await page.locator('.lj-cell.is-current .lj-action').first().evaluate(e=>getComputedStyle(e,'::after').animationName),'none');
  const nojs=await browser.newPage({javaScriptEnabled:false});await nojs.goto(base+'sglang/loop-journey/');
  assert.equal(await nojs.locator('[data-loop-reading]:visible').count(),3);assert.ok(await nojs.locator('[data-loop-reading] li').count()>=13);await nojs.close();
  if(shots){
   await page.setViewportSize({width:1440,height:1000});await page.emulateMedia({reducedMotion:'no-preference'});
   await page.goto(base+'sglang/loop-journey/');await page.evaluate(()=>document.documentElement.dataset.theme='light');
   await page.screenshot({path:path.join(shots,'default-desktop.png')});
   await page.goto(base+'sglang/loop-journey/?step=4#pp');
   await page.locator('.lj-tracks').screenshot({path:path.join(shots,'pp-parallel-window.png')});
   await page.locator('#lj-color').selectOption('loop');await page.reload();
   assert.equal(await root.getAttribute('data-step'),'4');assert.equal(await page.locator('#lj-color').inputValue(),'loop');
   await page.locator('.lj-tracks').screenshot({path:path.join(shots,'pp-loop-colors.png')});
   await page.locator('#lj-view').selectOption('loop');await page.locator('#lj-focus').selectOption('P0:L1');
   await page.locator('.lj-tracks').screenshot({path:path.join(shots,'pp-loop-focus.png')});
   await page.locator('#lj-view').selectOption('all');
   await page.setViewportSize({width:390,height:844});
   await page.locator('.lj-tracks').screenshot({path:path.join(shots,'pp-mobile-tracks.png')});
   await page.setViewportSize({width:320,height:844});await page.evaluate(()=>document.documentElement.dataset.theme='dark');
   await page.locator('.lj-tracks').screenshot({path:path.join(shots,'pp-mobile-dark-tracks.png')});
   await page.goto(base);await page.locator('#stage-instance').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(shots,'modules-mobile.png')});
   await page.setViewportSize({width:1440,height:1000});await page.locator('#stage-instance').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(shots,'modules-desktop.png')});
  }
  assert.deepEqual(errors,[]);console.log('PASS: 36 events, 3 blocked/release branches, 3 color modes, batch/loop ownership, PP focus, keyboard/history, 5 widths, 2 themes, reduced motion and no-JS reading.');
 }finally{await browser.close();await server.close();try{process.kill(owned.pid,0);throw Error('owned browser remains');}catch(e){if(e.code!=='ESRCH')throw e;}if(shots)fs.writeFileSync(path.join(shots,'cleanup.json'),JSON.stringify({browserExited:true,profileRemoved:!fs.existsSync(identity.profile.split('=').slice(1).join('='))}));}
})().catch(e=>{console.error(e);process.exit(1);});
