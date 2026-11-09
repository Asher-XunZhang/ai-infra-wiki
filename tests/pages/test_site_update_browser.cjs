const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(__dirname,'../../.pages-dist'),base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
const current=JSON.parse(fs.readFileSync(path.join(root,'site-release.json'))).release,next=current==='bbbbbbbbbbbbbbbb'?'cccccccccccccccc':'bbbbbbbbbbbbbbbb';
const shots=process.env.UPDATE_SCREENSHOT_DIR;
(async()=>{
 const server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 const proc=server.process(),identity={session:'site-update-review',controller:process.pid,browser:proc.pid,profile:proc.spawnargs.find(a=>a.startsWith('--user-data-dir='))};console.log(JSON.stringify(identity));
 if(shots){fs.mkdirSync(shots,{recursive:true});fs.writeFileSync(path.join(shots,'session.json'),JSON.stringify(identity));}
 const browser=await chromium.connect(server.wsEndpoint());
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],fixture={release:current,ready:true,offline:false,assetsMissing:false,delay:0,probes:0,navigations:0};
  page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/*',async route=>{
   const request=route.request(),url=new URL(request.url());
   if(url.origin!==new URL(base).origin)return route.continue();
   if(request.resourceType()==='document')fixture.navigations++;
   if(url.pathname.endsWith('/site-release.json')){
    fixture.probes++;if(fixture.delay)await new Promise(resolve=>setTimeout(resolve,fixture.delay));assert.equal(request.headers()['cache-control'],'no-cache');
    if(fixture.offline)return route.abort('internetdisconnected');
    return route.fulfill({json:{schema:1,release:fixture.release}});
   }
   if(fixture.assetsMissing&&/\.[a-f0-9]{16}\.(?:js|css)$/.test(url.pathname))return route.fulfill({status:404,body:'missing old fingerprint'});
   if(url.searchParams.has('_refresh')){
    const relative=decodeURIComponent(url.pathname.slice(new URL(base).pathname.length));
    const file=path.join(root,relative+(!relative||relative.endsWith('/')?'index.html':''));
    const html=fs.readFileSync(file,'utf8').replace(/name="site-release" content="[a-f0-9]{16}"/,`name="site-release" content="${fixture.ready?fixture.release:current}"`);
    return route.fulfill({contentType:'text/html',body:html});
   }
   return route.continue();
  });
  const check=async()=>{await page.locator('[data-site-check]').focus();await page.locator('[data-site-check]').press('Enter');await page.waitForFunction(()=>document.querySelector('[data-site-check]').getAttribute('aria-busy')==='false');};
  const announce=async()=>{await page.waitForFunction(()=>document.querySelector('[data-site-check]')?.getAttribute('aria-busy')==='false');fixture.release=next;await check();assert.equal(await page.locator('.site-update-notice').isVisible(),true,JSON.stringify({url:page.url(),footer:await page.locator('.site-update-tools').innerText(),release:fixture.release,probes:fixture.probes,meta:await page.locator('meta[name=site-release]').getAttribute('content')}));};
  const apply=async()=>{
   const previous=fixture.navigations;
   await page.locator('.site-update-notice [data-site-update]').evaluate(e=>e.click());
   await page.waitForFunction(release=>document.querySelector('meta[name="site-release"]').content===release,next);
   await page.waitForLoadState('load');await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
   assert.equal(fixture.navigations,previous+1,'one navigation per accepted update');
   assert.equal(new URL(page.url()).searchParams.has('_refresh'),false);
   assert.equal(new URL(page.url()).searchParams.has('v'),false);
  };
  // A manual request joining an automatic probe still gets visible feedback.
  fixture.delay=1000;await page.goto(base,{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>document.querySelector('[data-site-check]')?.getAttribute('aria-busy')==='true');
  await check();assert.match(await page.locator('.site-update-tools').innerText(),/已是最新/);fixture.delay=0;
  // Stable URLs and current-version checks.
  await page.goto(base+'sglang/loop-journey/?v=legacy&step=4&color=loop&learn=first#pp');
  await check();assert.match(await page.locator('.site-update-tools').innerText(),/已是最新/);
  assert.equal(new URL(page.url()).searchParams.has('v'),false);assert.equal(await page.locator('[data-loop-journey]').getAttribute('data-step'),'4');
  await page.locator('#lj-view').selectOption('loop');await page.locator('#lj-focus').selectOption('P0:L1');
  await page.locator('.lj-state>summary').click();await announce();
  for(const width of [1440,390,320])for(const theme of ['light','dark']){
   await page.setViewportSize({width,height:900});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
   await page.locator('.site-update-notice').scrollIntoViewIfNeeded();
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
   assert.equal(await page.locator('.site-update-notice').evaluate(e=>getComputedStyle(e).position),'static');
   if(shots)await page.screenshot({path:path.join(shots,`notice-${width}-${theme}.png`)});
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('#lj-tracks-title').evaluate(e=>scrollTo({top:e.getBoundingClientRect().top+scrollY-100,behavior:'instant'}));
  const beforeY=await page.evaluate(()=>scrollY);await apply();
  assert.equal(await page.locator('[data-loop-journey]').getAttribute('data-step'),'4');
  assert.equal(await page.locator('#lj-view').inputValue(),'loop');assert.equal(await page.locator('#lj-focus').inputValue(),'P0:L1');
  assert.equal(await page.locator('#lj-color').inputValue(),'loop');assert.equal(new URL(page.url()).searchParams.get('learn'),'first');assert.equal(new URL(page.url()).hash,'#pp');
  assert.equal(await page.locator('.lj-state').evaluate(e=>e.open),true);assert.ok(Math.abs(await page.evaluate(()=>scrollY)-beforeY)<100,'reading scroll retained');
  // Missing/cached bootstrap and partial publication cannot trap the reader.
  fixture.assetsMissing=true;fixture.release=current;await page.goto(base+'sglang/loop-journey/?step=2#overlap');await check();assert.ok(await page.evaluate(()=>Boolean(window.SiteUpdate)));fixture.assetsMissing=false;await page.goto(base+'sglang/loop-journey/?step=2#overlap');
  fixture.offline=true;await check();assert.match(await page.locator('.site-update-tools').innerText(),/暂时无法检查/);fixture.offline=false;
  await announce();fixture.ready=false;const stays=page.url(),nav=fixture.navigations;
  await page.locator('.site-update-notice [data-site-update]').click();await page.waitForFunction(()=>document.querySelector('.site-update-tools').dataset.state==='available');
  assert.match(await page.locator('.site-update-notice').innerText(),/同步/);assert.equal(page.url(),stays);assert.equal(fixture.navigations,nav);
  fixture.ready=true;await apply();assert.equal(await page.locator('[data-loop-journey]').getAttribute('data-step'),'2');
  // A release gate already opened is not re-closed by an update.
  fixture.release=current;await page.goto(base+'sglang/loop-journey/?hold=1&step=100#pp');
  await page.locator('#lj-release').click();assert.equal(await page.locator('[data-loop-journey]').getAttribute('data-blocked'),'false');await announce();await apply();
  assert.equal(await page.locator('[data-loop-journey]').getAttribute('data-blocked'),'false');
  // Chapter 2 keeps its topic and manually selected stage.
  fixture.release=current;await page.goto(base+'engine-comparison/#wire');
  await page.locator('.ia-stepItem').nth(2).click();
  const atlasText=await page.locator('#ia-stage-title').innerText();await announce();await apply();
  assert.equal(new URL(page.url()).hash,'#wire');assert.equal(await page.locator('.ia-stepItem').nth(2).getAttribute('aria-current'),'step');assert.equal(await page.locator('#ia-stage-title').innerText(),atlasText);
  // Independent continuous clocks retain exact positions and resume paused.
  for(const [route,seek,host,attribute] of [
   ['sglang/parallelism/?mode=pp','#parallel-seek','#parallel-scene','data-time'],
   ['sglang/system-overview/','#op-range','#sglang-opening','data-progress'],
   ['sglang/inference-overview/deployment.html','#pd-handoff-seek','#pd-handoff-scene','data-time'],
   ['sglang/inference-overview/journey.html','#sf-seek','[data-sf-lab]','data-time'],
   ['sglang/pd-dataflow/','[data-role=seek]','#dataflow-execution','data-time']
  ]){
   fixture.release=current;await page.goto(base+route);await page.locator(seek).waitFor();
   await page.locator(seek).evaluate(e=>{e.value=Number(e.max)*.4;e.dispatchEvent(new Event('input',{bubbles:true}));});
   const before=await page.locator(host).getAttribute(attribute);await announce();await apply();assert.equal(await page.locator(host).getAttribute(attribute),before,route);
  }
  // PP consensus retains the chosen branch, live clock and speed after an explicit update.
  fixture.release=current;await page.goto(base+'sglang/pp-consensus/?mode=bootstrap&condition=metadata&t=8.25#consensus-title');
  const consensus=page.locator('#consensus-lab');await consensus.locator('[data-role=speed]').selectOption('1.5');await announce();
  await consensus.locator('[data-action=play]').click();await page.waitForTimeout(100);await apply();
  assert.equal(await consensus.locator('[data-mode=bootstrap]').getAttribute('aria-pressed'),'true');
  assert.equal(await consensus.locator('[data-role=condition]').inputValue(),'metadata');
  assert.equal(await consensus.locator('[data-role=speed]').inputValue(),'1.5');
  const resumedTime=Number(await consensus.getAttribute('data-time'));assert.ok(resumedTime>8.25&&resumedTime<18);
  assert.equal(await consensus.locator('[data-action=play]').getAttribute('aria-pressed'),'false');
  await page.waitForTimeout(100);assert.equal(Number(await consensus.getAttribute('data-time')),resumedTime);
  // Shared first-chapter player stores fractional progress in its existing URL.
  fixture.release=current;await page.goto(base+'foundations/generation.html?concept=sampling&ms=1.5#mechanism-process');
  await page.waitForSelector('.mechanism-film');await announce();const state=new URL(page.url()).searchParams.get('ms');await apply();assert.equal(new URL(page.url()).searchParams.get('ms'),state);
  // Discrete performance stages and the detailed PP dependency inspector.
  fixture.release=current;await page.goto(base+'sglang/performance-engineering/');
  await page.locator('[data-performance-lab] [data-mode=packet]').click();await page.locator('#perf-next').click();await page.locator('#perf-next').click();
  await announce();await apply();assert.equal(await page.locator('#perf-scene').getAttribute('data-mode'),'packet');assert.equal(await page.locator('#perf-scene').getAttribute('data-step'),'2');
  fixture.release=current;await page.goto(base+'sglang/pd-prefill-pp-loop/?scenario=baseline#timeline');
  const frame=page.frameLocator('#timeline iframe');await frame.locator('[data-rank]').selectOption('2');await frame.locator('[data-loop]').selectOption('3');
  await frame.locator('[data-zoom-in]').click();
  const reading=await frame.locator('body').evaluate(()=>window.PPReadingState.capture());await announce();await apply();
  await page.frames().find(f=>f.url()==='about:srcdoc').waitForFunction(expected=>JSON.stringify(window.PPReadingState?.capture())===JSON.stringify(expected),reading);
  fixture.release=current;await page.goto(base+'sglang/pd-prefill-pp-loop/quick.html?scenario=slow-link#quick-timeline');
  await page.waitForFunction(()=>document.querySelector('#chart').dataset.scenario==='slow-link');
  await page.locator('#full-timeline-view>summary').click();await page.locator('#rank').selectOption('1');await page.locator('#loop').selectOption('4');await page.locator('#batch').selectOption('3');await page.locator('#life-phase').selectOption('2');
  await page.locator('[data-id=pp-loop-index]').selectOption('3');await page.locator('[data-id=pp-story-next]').click();
  const quick=await page.locator('#chart').textContent(),story=await page.locator('[data-pp-reader] [data-id=wr-scene]').textContent();
  await announce();await apply();await page.waitForFunction(()=>document.querySelector('#chart').dataset.scenario==='slow-link');
  assert.equal(await page.locator('#chart').textContent(),quick);assert.equal(await page.locator('[data-pp-reader] [data-id=wr-scene]').textContent(),story);
  // Returning from bfcache triggers a check without user interaction/reload.
  fixture.release=current;await page.goto(base);await check();fixture.release=next;const count=fixture.probes;
  await page.evaluate(()=>dispatchEvent(new PageTransitionEvent('pageshow',{persisted:true})));await page.waitForFunction(()=>!document.querySelector('.site-update-notice').hidden);assert.equal(fixture.probes,count+1);
  // Failure to checkpoint never silently discards the current reader state.
  await page.evaluate(()=>{Storage.prototype.setItem=()=>{throw new Error('blocked');};});
  const before=fixture.navigations;await page.locator('.site-update-notice [data-site-update]').click();await page.waitForFunction(()=>document.querySelector('.site-update-notice').textContent.includes('无法保存'));
  assert.equal(fixture.navigations,before);
  assert.deepEqual(errors,[]);console.log('PASS: stable URLs, version checks, single update, state/scroll restore, 3 widths/2 themes, missing-asset recovery, offline/partial deploy/storage failures, bfcache and independent lesson clocks.');
 }finally{await browser.close();await server.close();let exited=false;try{process.kill(proc.pid,0);}catch(e){exited=e.code==='ESRCH';}assert.ok(exited);if(shots)fs.writeFileSync(path.join(shots,'cleanup.json'),JSON.stringify({browserExited:exited,profileRemoved:!fs.existsSync(identity.profile.split('=').slice(1).join('='))}));}
})().catch(e=>{console.error(e);process.exitCode=1;});
