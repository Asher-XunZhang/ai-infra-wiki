const assert=require('node:assert/strict');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const BASE=(process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/').replace(/\/+$/,'')+'/';
const CHAPTERS=['process','wire','prefix','batch','disagg','crosscut'];
const alive=pid=>{try{process.kill(pid,0);return true;}catch(e){return e.code!=='ESRCH';}};
const waitGone=async(pid,ms=3000)=>{const end=Date.now()+ms;while(Date.now()<end){if(!alive(pid))return true;await new Promise(r=>setTimeout(r,60));}return !alive(pid);};
const stepAt=page=>page.locator('.ia-stepItem').evaluateAll(items=>items.findIndex(i=>i.getAttribute('aria-current')==='step'));
const activeTab=page=>page.locator('.ia-chapterTab[aria-pressed=true]');
const noOverflow=async(page,label)=>{
 const size=await page.evaluate(()=>({html:document.documentElement.scrollWidth,body:document.body.scrollWidth,inner:innerWidth}));
 assert.ok(size.html<=size.inner,`${label}: html ${size.html} > ${size.inner}`);
 assert.ok(size.body<=size.inner,`${label}: body ${size.body} > ${size.inner}`);
};
const horizontalPair=async(page,label)=>{
 const boxes=await page.locator('.ia-board').evaluateAll(nodes=>nodes.map(n=>{const b=n.getBoundingClientRect();return {x:b.x,y:b.y,width:b.width,right:b.right};}));
 assert.equal(boxes.length,2,label);
 assert.ok(Math.abs(boxes[0].y-boxes[1].y)<1,`${label}: engines must share a row ${JSON.stringify(boxes)}`);
 assert.ok(boxes[1].x>=boxes[0].right,`${label}: vLLM must stay to the right`);
 assert.ok(Math.abs(boxes[0].width-boxes[1].width)<1,`${label}: equal comparison widths`);
};
const visibleBoundaries=async(page)=>{
 const ratios=await page.locator('.ia-zoneRect,.ia-blockBracket,.ia-splitMarker').evaluateAll(nodes=>{
  const rgb=value=>value.startsWith('#')?value.slice(1).match(/../g).map(v=>parseInt(v,16)):value.match(/[\d.]+/g).map(Number);
  const lum=c=>c.slice(0,3).map(v=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4}).reduce((sum,v,i)=>sum+v*[.2126,.7152,.0722][i],0);
  return nodes.map(n=>{
   const style=getComputedStyle(n),base=rgb(getComputedStyle(n.closest('.ia-page')).getPropertyValue('--ia-scene-2').trim());
   const fill=style.fill==='none'?[...base,0]:rgb(style.fill),alpha=fill[3]??1;
   const background=base.map((v,i)=>v*(1-alpha)+fill[i]*alpha);
   const stroke=rgb(style.stroke),a=stroke[3]??1,ink=background.map((v,i)=>v*(1-a)+stroke[i]*a);
   return (Math.max(lum(ink),lum(background))+.05)/(Math.min(lum(ink),lum(background))+.05);
  });
 });
 assert.ok(ratios.every(r=>r>=3),`dashed boundaries must contrast with the canvas: ${ratios}`);
};
const diagramGeometry=async(page,label)=>{
 const issues=await page.evaluate(()=>{
  const found=[];
  const intersects=(a,b)=>Math.min(a.x+a.width,b.x+b.width)-Math.max(a.x,b.x)>1 && Math.min(a.y+a.height,b.y+b.height)-Math.max(a.y,b.y)>1;
  for(const svg of document.querySelectorAll('.ia-board .ia-svg')){
   const side=svg.closest('.ia-board').dataset.engine,view=svg.viewBox.baseVal;
   const texts=[...svg.querySelectorAll('text')];
   for(const text of texts){
    const b=text.getBBox(),node=text.closest('.ia-node');
    if(b.x<0||b.y<0||b.x+b.width>view.width||b.y+b.height>view.height)found.push(`${side}: clipped text ${text.textContent}`);
    if(node&&/ia-node(Label|Tag)/.test(text.getAttribute('class'))){
     const r=node.querySelector('.ia-nodeBox').getBBox();
     if(b.x<r.x+2||b.x+b.width>r.x+r.width-2||b.y<r.y||b.y+b.height>r.y+r.height-2)found.push(`${side}: node caption escapes ${text.textContent}`);
    }
    if(text.classList.contains('ia-edgeLabel'))for(const rect of svg.querySelectorAll('.ia-nodeBox')){
     if(intersects(b,rect.getBBox()))found.push(`${side}: node covers edge label ${text.textContent}`);
    }
   }
   for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){
    if(intersects(texts[i].getBBox(),texts[j].getBBox()))found.push(`${side}: overlapping ${texts[i].textContent} / ${texts[j].textContent}`);
   }
  }
  return found;
 });
 assert.deepEqual(issues,[],label);
};
(async()=>{
 let server,browser,owned,browserPid=0,profilePath='',cleanupDone=false,report={};
 const cleanup=async()=>{
  if(cleanupDone)return report;cleanupDone=true;
  try{if(browser)await browser.close();}catch(e){report.browserCloseError=e.message;}
  try{if(server)await server.close();}catch(e){report.serverCloseError=e.message;}
  let stopped=true;
  if(browserPid){
   stopped=await waitGone(browserPid,2500);
   if(!stopped){try{process.kill(browserPid,'SIGTERM');}catch(e){}stopped=await waitGone(browserPid,2000);}
   if(!stopped){try{process.kill(browserPid,'SIGKILL');}catch(e){}stopped=await waitGone(browserPid,1000);}
  }
  if(profilePath&&fs.existsSync(profilePath)&&(profilePath.includes('playwright')||profilePath.startsWith(os.tmpdir()))){
   try{fs.rmSync(profilePath,{recursive:true,force:true});}catch(e){report.profileRemoveError=e.message;}
  }
  report={...report,session:'engine-comparison-browser',controller:process.pid,browser:browserPid,profile:profilePath,stopped,residue:browserPid&&alive(browserPid)?1:0,profileResidue:!!(profilePath&&fs.existsSync(profilePath))};
  console.log(JSON.stringify({cleanup:report}));
  if(process.env.ENGINE_TEST_REPORT)try{fs.writeFileSync(process.env.ENGINE_TEST_REPORT,JSON.stringify(report,null,2));}catch(e){}
  return report;
 };
 const shutdown=signal=>{cleanup().finally(()=>process.exit(signal==='SIGINT'?130:143));};
 process.once('SIGINT',()=>shutdown('SIGINT'));process.once('SIGTERM',()=>shutdown('SIGTERM'));
 try{
  const launch={headless:true,handleSIGINT:false,handleSIGTERM:false};
  if(process.env.CHROMIUM_EXECUTABLE)launch.executablePath=process.env.CHROMIUM_EXECUTABLE;
  server=await chromium.launchServer(launch);
  owned=server.process();browserPid=owned.pid;
  const profileArg=owned.spawnargs.find(a=>a.startsWith('--user-data-dir='));
  profilePath=profileArg?profileArg.slice('--user-data-dir='.length):'';
  console.log(JSON.stringify({session:'engine-comparison-browser',controller:process.pid,browser:browserPid,profile:profilePath}));
  browser=await chromium.connect(server.wsEndpoint());
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(BASE+'engine-comparison/index.html#process');await page.waitForLoadState('load');
  assert.match(await page.locator('.ia-releaseCol').innerText(),/297e54f/);
  await page.locator('.ia-referenceDetails').last().locator('summary').click();
  assert.match(await page.locator('.ia-sourceBody').innerText(),/297e54f/);
  assert.match(await page.locator('.ia-sourceBody').innerText(),/TP=2 \/ PP=1 \/ DP=1/);
  await page.locator('.ia-referenceDetails').last().locator('summary').click();
  assert.doesNotMatch(await page.locator('.ia-footerMeta').innerText(),/Harness/i);

  // Six topics × four steps, including per-topic release source and chapter 3 links.
  for(let chapter=0;chapter<CHAPTERS.length;chapter++){
   const tab=page.locator('.ia-chapterTab').nth(chapter);
   await tab.click();await page.waitForFunction(id=>location.hash==='#'+id,CHAPTERS[chapter]);
   assert.equal(await tab.getAttribute('aria-pressed'),'true',CHAPTERS[chapter]);
   assert.equal(await page.locator('.ia-stepItem').count(),4,CHAPTERS[chapter]);
   const hrefs=await page.locator('.ia-chapterLinks a').evaluateAll(as=>as.map(a=>a.href));
   assert.ok(hrefs.some(h=>h.includes('/blob/')),`${CHAPTERS[chapter]} pinned source`);
   assert.ok(hrefs.some(h=>h.includes('/sglang/')&&new URL(h).pathname.endsWith('.html')),`${CHAPTERS[chapter]} chapter 3 link`);
   await page.locator('.ia-referenceDetails>summary').click();
   const sourceLinks=await page.locator('.ia-sourceLinks a').evaluateAll(as=>as.map(a=>a.href));
   assert.deepEqual(sourceLinks.filter(h=>h.includes('/blob/')).sort(),hrefs.filter(h=>h.includes('/blob/')).sort(),'sources must match this topic');
   assert.equal(sourceLinks.filter(h=>h.includes('/releases/tag/')).length,2);
   await page.locator('.ia-referenceDetails>summary').click();
   for(let step=0;step<4;step++){
    const item=page.locator('.ia-stepItem').nth(step);
    await item.click();
    assert.equal(await item.getAttribute('aria-current'),'step',`${CHAPTERS[chapter]} step ${step+1}`);
    assert.equal(await page.locator('.ia-stepItem[aria-current=step]').count(),1);
    assert.equal((await page.locator('#ia-stage-title').innerText()).trim(),(await item.locator('.ia-stepLabel').innerText()).trim());
    assert.equal(await page.locator('.ia-board').count(),2);
    assert.equal(await page.getByRole('button',{name:'上一步',exact:true}).isDisabled(),step===0);
    assert.equal(await page.getByRole('button',{name:'下一步',exact:true}).isDisabled(),step===3);
    assert.ok(await page.locator('.ia-edge.is-active,.ia-nodePulse,.ia-splitMarker.is-active').evaluateAll(nodes=>nodes.every(el=>getComputedStyle(el).animationPlayState==='running')),`${CHAPTERS[chapter]} step ${step+1}: existing effects must run`);
    await diagramGeometry(page,`${CHAPTERS[chapter]} step ${step+1}`);
    await horizontalPair(page,`${CHAPTERS[chapter]} step ${step+1}`);
    await visibleBoundaries(page);
    assert.equal(await page.locator('.ia-topicReading').getAttribute('data-topic'),CHAPTERS[chapter]);
    assert.equal(await page.locator('.ia-table').count(),0,'unrequested tables must not be mounted');
    assert.equal(await page.locator('.ia-readingDetails[open]').count(),0,'a new step starts with concise reading');
    await page.locator('.ia-readingDetails>summary').click();
    assert.equal(await page.locator('.ia-topicReading table:visible').count(),1);
    assert.equal(await page.locator('.ia-table').count(),1,'only this topic comparison may be mounted');
    if(chapter===5)assert.match(await page.locator('.ia-topicReading h2').innerText(),[/KV/,/过载/,/取消/,/图捕获/][step]);
    await noOverflow(page,`${CHAPTERS[chapter]} step ${step+1} 1440`);
   }
  }

  // Topic-specific reading and semantic lists; global references remain optional.
  await page.locator('.ia-chapterTab').nth(1).click();
  await page.locator('.ia-readingDetails>summary').click();
  assert.equal(await stepAt(page),0,'opening a reading must retain the stage');
  assert.equal(await page.locator('.ia-topicReading ol.ia-sectionNote li').count(),4);
  assert.match(await page.locator('.ia-topicReading ol li').last().innerText(),/适用范围/);
  assert.equal(await page.locator('.ia-referenceDetails[open]').count(),0);
  assert.equal(await page.getByText('差异不是功能表，而是几条结构线',{exact:true}).count(),0);
  assert.equal(await page.locator('.ia-releaseSection').count(),0);
  await page.locator('.ia-readingDetails>summary').click();
  assert.equal(await page.locator('.ia-table').count(),0,'closing must remove the full comparison');
  await page.locator('#lang-toggle').click();
  for(let topic=0;topic<6;topic++){
   await page.locator('.ia-chapterTab').nth(topic).click();
   for(let step=0;step<4;step++){
    await page.locator('.ia-stepItem').nth(step).click();
    await diagramGeometry(page,`English ${CHAPTERS[topic]} step ${step+1}`);
    assert.equal(await page.locator('.ia-table').count(),0);
    await page.locator('.ia-readingDetails>summary').click();
    assert.equal(await page.locator('.ia-topicReading table:visible').count(),1);
   }
  }
  await page.locator('#lang-toggle').click();

  // Node notes retain the stage and leave the stage effects running.
  await page.goto(BASE+'engine-comparison/index.html#process');await page.waitForLoadState('load');
  await page.locator('.ia-board--sg .ia-node[data-node-id=scheduler]').click();
  assert.equal(await page.locator('.ia-inspector').isVisible(),true);
  assert.match(await page.locator('.ia-inspector').innerText(),/Scheduler/);
  assert.equal(await stepAt(page),0);
  await page.locator('.ia-closeButton').click();
  assert.equal(await page.locator('.ia-inspector').count(),0);

  // Core components explain their role before implementation details.
  await page.locator('.ia-board--vl .ia-node[data-node-id=engine]').click();
  assert.match(await page.locator('.ia-inspectorRole').innerText(),/调度中枢/);
  assert.match(await page.locator('.ia-inspectorBody').innerText(),/Worker/);
  await page.locator('#lang-toggle').click();
  assert.match(await page.locator('.ia-inspectorRole').innerText(),/scheduling hub/);
  await page.locator('#lang-toggle').click();
  if(process.env.ENGINE_TEST_SCREENSHOTS){
   fs.mkdirSync(process.env.ENGINE_TEST_SCREENSHOTS,{recursive:true});
   await page.locator('.ia-inspector').scrollIntoViewIfNeeded();
   await page.screenshot({path:path.join(process.env.ENGINE_TEST_SCREENSHOTS,'enginecore-role.png')});
  }
  await page.locator('.ia-closeButton').click();
  await page.locator('.ia-chapterTab').nth(3).click();
  await page.locator('.ia-board--vl .ia-node[data-node-id=output]').click();
  assert.match(await page.locator('.ia-inspectorRole').innerText(),/前端结果包/);
  assert.match(await page.locator('.ia-inspectorBody').innerText(),/EngineCoreOutputs/);
  await page.locator('.ia-chapterTab').nth(4).click();
  for(const side of ['sg','vl'])for(const [node,role] of [['prefill',/输入计算者/],['decode',/续写执行者/]]){
   await page.locator(`.ia-board--${side} .ia-node[data-node-id=${node}]`).click();
   assert.match(await page.locator('.ia-inspectorRole').innerText(),role);
  }
  await page.locator('.ia-chapterTab').nth(0).click();

  // Stages only advance on reader input; decorative motion remains independent.
  assert.equal(await page.locator('.ia-playButton,.ia-speed,.ia-playProgress,.ia-liveStatus').count(),0);
  const prev=page.getByRole('button',{name:'上一步',exact:true});
  const next=page.getByRole('button',{name:'下一步',exact:true});
  assert.equal(await prev.isDisabled(),true);
  await next.focus();await page.keyboard.press('Enter');assert.equal(await stepAt(page),1);
  await next.click();assert.equal(await stepAt(page),2);
  await next.click();assert.equal(await stepAt(page),3);assert.equal(await next.isDisabled(),true);
  await prev.focus();await page.keyboard.press('Space');assert.equal(await stepAt(page),2);
  await prev.click();await prev.click();assert.equal(await stepAt(page),0);assert.equal(await prev.isDisabled(),true);
  const effects=()=>page.locator('.ia-edge.is-active,.ia-nodePulse,.ia-splitMarker.is-active').evaluateAll(nodes=>nodes.map(el=>({
   style:getComputedStyle(el).animationPlayState,
   name:getComputedStyle(el).animationName,
   offset:getComputedStyle(el).strokeDashoffset,
   opacity:getComputedStyle(el).opacity,
   time:el.getAnimations()[0]?.currentTime,
  })));
  const before=await effects();assert.ok(before.length>0);
  assert.ok(before.every(e=>e.style==='running'&&e.name!=='none'));
  await page.waitForTimeout(350);
  const after=await effects();
  assert.ok(after.some((e,i)=>e.time>before[i].time&& (e.offset!==before[i].offset||e.opacity!==before[i].opacity)),'stage effects must actually move');
  await page.locator('.ia-board--sg .ia-node[data-node-id=scheduler]').click();
  await page.locator('.ia-readingDetails>summary').click();
  assert.ok((await effects()).every(e=>e.style==='running'),'reading and inspecting must not freeze stage effects');
  await page.waitForTimeout(6000);assert.equal(await stepAt(page),0,'idle reader must never be advanced by a timer');
  await page.locator('.ia-closeButton').click();
  await page.locator('.ia-readingDetails>summary').click();
  if(process.env.ENGINE_TEST_SCREENSHOTS){
   fs.mkdirSync(process.env.ENGINE_TEST_SCREENSHOTS,{recursive:true});
   await page.locator('.ia-controls').scrollIntoViewIfNeeded();
   await page.screenshot({path:path.join(process.env.ENGINE_TEST_SCREENSHOTS,'manual-desktop.png')});
  }

  // Direct hash, history and reload.
  await page.goto(BASE+'engine-comparison/index.html#prefix');await page.waitForLoadState('load');
  assert.match(await activeTab(page).innerText(),/前缀缓存/);
  await page.locator('.ia-chapterTab').nth(4).click();
  await page.waitForFunction(()=>location.hash==='#disagg');
  assert.match(await activeTab(page).innerText(),/分离扩展/);
  await page.goBack().catch(()=>{});
  await page.waitForFunction(()=>location.hash==='#prefix');
  assert.match(await activeTab(page).innerText(),/前缀缓存/);
  await page.reload();await page.waitForLoadState('load');
  assert.match(await activeTab(page).innerText(),/前缀缓存/);assert.equal(await stepAt(page),0);

  // Prefix reuse is 8/12 on both engines.
  await page.locator('.ia-stepItem').nth(1).click();
  const feet=await page.locator('.ia-boardFootText').allInnerTexts();
  assert.equal(feet.length,2);
  feet.forEach(text=>assert.match(text,/8\s*\/\s*12/));

  // Chinese / English and light / dark.
  await page.locator('#lang-toggle').click();
  assert.equal(await page.locator('html').getAttribute('lang'),'en');
  assert.match(await page.locator('.ia-title').innerText(),/One request/);
  assert.match(await activeTab(page).innerText(),/Prefix cache/);
  assert.match(await page.locator('.ia-chapterLinks').innerText(),/Chapter 3 deep dives/i);
  await page.locator('#lang-toggle').click();
  assert.equal(await page.locator('html').getAttribute('lang'),'zh-CN');
  assert.match(await page.locator('.ia-chapterLinks').innerText(),/第三章深入/);
  await page.locator('.theme-toggle').click();assert.equal(await page.locator('html').getAttribute('data-theme'),'light');
  await noOverflow(page,'1440 light');
  await page.locator('.theme-toggle').click();assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');

  // Fixed horizontal pairing at desktop, tablet and phone widths; only the pair scrolls.
  for(const width of [1440,1280,1100,1024,900,768,390,320]){
   await page.setViewportSize({width,height:1000});
   await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
   await noOverflow(page,`${width} dark`);
   await horizontalPair(page,`${width} dark`);
   if(width>=1024)assert.ok(await page.locator('.ia-boardGrid').evaluate(e=>e.scrollWidth<=e.clientWidth+1),`desktop pair fits at ${width}`);
   if(process.env.ENGINE_TEST_SCREENSHOTS&&[1280,1024,390].includes(width)){
    await page.locator('.ia-comparison').scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(process.env.ENGINE_TEST_SCREENSHOTS,`horizontal-${width}.png`)});
   }
   for(const stage of await page.locator('.ia-boardStage').all()){
    const size=await stage.evaluate(el=>({scroll:el.scrollWidth,client:el.clientWidth}));
    assert.ok(size.scroll<=size.client+1,`default diagram must fit at ${width}: ${JSON.stringify(size)}`);
   }
  }
  await page.setViewportSize({width:390,height:1000});
  await page.locator('.theme-toggle').click();
  assert.equal(await page.locator('html').getAttribute('data-theme'),'system');
  await noOverflow(page,'390 system');

  const pair=page.locator('.ia-boardGrid');
  assert.equal(await page.locator('.ia-comparisonHint').isVisible(),true);
  const pairSize=await pair.evaluate(el=>({scroll:el.scrollWidth,client:el.clientWidth,role:el.getAttribute('role'),tab:el.tabIndex}));
  assert.ok(pairSize.scroll>pairSize.client);assert.equal(pairSize.role,'region');assert.equal(pairSize.tab,0);
  assert.ok(await page.locator('.ia-board--sg').evaluate(el=>el.clientWidth<=el.closest('.ia-boardGrid').clientWidth),'a single engine card must fit the narrow viewport');
  await pair.evaluate(el=>{el.scrollLeft=0;el.focus();});
  await page.keyboard.press('ArrowRight');await page.waitForTimeout(150);
  assert.ok(await pair.evaluate(el=>el.scrollLeft>0),'keyboard must scroll the comparison');
  await pair.evaluate(el=>el.scrollLeft=el.scrollWidth);
  assert.ok(await page.locator('.ia-board--vl').evaluate(el=>{const b=el.getBoundingClientRect(),g=el.closest('.ia-boardGrid').getBoundingClientRect();return b.right<=g.right+1&&b.right>g.left;}),'the right-hand engine is reachable');
  await noOverflow(page,'horizontal pair scrolled');await pair.evaluate(el=>el.scrollLeft=0);

  // The full diagram fits by default; explicit zoom offers a keyboard-accessible detail view.
  const stage=page.locator('.ia-board--sg .ia-boardStage');
  const zoom=page.locator('.ia-board--sg .ia-boardZoom');
  assert.equal(await zoom.getAttribute('aria-pressed'),'false');
  await zoom.click();
  assert.equal(await zoom.getAttribute('aria-pressed'),'true');
  const mobile=await stage.evaluate(el=>({
   scrollWidth:el.scrollWidth,clientWidth:el.clientWidth,tabIndex:el.tabIndex,
   role:el.getAttribute('role'),label:el.getAttribute('aria-label'),
  }));
  assert.ok(mobile.scrollWidth>mobile.clientWidth+5,`mobile board should scroll: ${JSON.stringify(mobile)}`);
  assert.equal(mobile.tabIndex,0);
  assert.equal(mobile.role,'region');
  assert.match(mobile.label,/左右滑动/);
  assert.equal(await page.locator('.ia-boardScrollHint').first().isVisible(),true);
  const scrolled=await stage.evaluate(el=>{
   el.scrollLeft=el.scrollWidth;el.focus();
   return {left:el.scrollLeft,focused:document.activeElement===el,doc:document.documentElement.scrollWidth,body:document.body.scrollWidth,inner:innerWidth,winX:scrollX};
  });
  assert.ok(scrolled.left>0,`stage should scroll right: ${JSON.stringify(scrolled)}`);
  assert.equal(scrolled.focused,true);
  assert.ok(scrolled.doc<=scrolled.inner);assert.ok(scrolled.body<=scrolled.inner);assert.equal(scrolled.winX,0);
  const beforeKey=scrolled.left;
  await page.keyboard.press('ArrowLeft');await page.waitForTimeout(120);
  const afterKey=await stage.evaluate(el=>el.scrollLeft);
  assert.ok(afterKey<beforeKey,`keyboard should scroll stage left: ${afterKey} !< ${beforeKey}`);
  const foot=await page.locator('.ia-board--sg .ia-boardFootText').evaluate(el=>{
   const style=getComputedStyle(el);
   return {client:el.clientWidth,scroll:el.scrollWidth,whiteSpace:style.whiteSpace,textOverflow:style.textOverflow,text:el.innerText.trim()};
  });
  assert.ok(foot.text.length>0);
  assert.ok(foot.scroll<=foot.client+1,`footer text is clipped: ${JSON.stringify(foot)}`);
  assert.notEqual(foot.whiteSpace,'nowrap');
  await zoom.click();
  assert.equal(await zoom.getAttribute('aria-pressed'),'false');
  assert.ok(await stage.evaluate(el=>el.scrollWidth<=el.clientWidth+1));

  // Reduced motion suppresses looping effects; manual navigation still works.
  const reduced=await browser.newContext({reducedMotion:'reduce',viewport:{width:390,height:900}});
  try{
   const reducedPage=await reduced.newPage();reducedPage.on('pageerror',e=>errors.push(e.message));
   await reducedPage.goto(BASE+'engine-comparison/index.html#process');await reducedPage.waitForLoadState('load');
   assert.equal(await reducedPage.locator('.ia-page').getAttribute('data-reduced-motion'),'true');
   assert.equal(await reducedPage.locator('.ia-playButton,.ia-speed,.ia-playProgress').count(),0);
   await reducedPage.waitForTimeout(100);
   assert.equal(await reducedPage.locator('.ia-board').evaluateAll(boards=>boards.flatMap(b=>b.getAnimations({subtree:true})).filter(a=>a.playState==='running').length),0);
   await reducedPage.getByRole('button',{name:'下一步',exact:true}).click();assert.equal(await stepAt(reducedPage),1);
   assert.equal(await reducedPage.locator('.ia-page').getAttribute('data-reduced-motion'),'true');
  }finally{await reduced.close();}
  if(process.env.ENGINE_TEST_SCREENSHOTS){
   await page.setViewportSize({width:390,height:1000});
   await page.locator('.ia-controls').scrollIntoViewIfNeeded();
   await page.screenshot({path:path.join(process.env.ENGINE_TEST_SCREENSHOTS,'manual-mobile.png')});
  }

  assert.deepEqual(errors,[]);
  console.log('PASS: engine comparison six topics / 24 steps, pinned source and chapter 3 links, node notes without freezing effects, manual previous/next and boundary states, idle stage retention and animated edges/nodes, hash/history/reload, zh/EN, light/dark/system, eight widths with fixed horizontal engine pairs, visible dashed boundaries, role-first core descriptions, and complete diagrams, explicit zoom/fit and keyboard scrolling, 48 bilingual geometry states, topic/step-specific reading and ordered notes, prefix 8/12, reduced motion without looping effects.');
 }finally{await cleanup();}
})().catch(e=>{console.error(e);process.exit(1);});
