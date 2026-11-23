'use strict';
// Exercise real cross-page/hash navigation; content and animation models retain their own tests.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const catalog=require('../../pages/learning-paths.json');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
const shots=process.env.CROSS_CHAPTER_SCREENSHOT_DIR;
const processes=()=>cp.execFileSync('ps',['-axo','pid=,ppid=,command='],{encoding:'utf8'}).trim().split('\n').map(s=>{const m=s.trim().match(/^(\d+)\s+(\d+)\s+(.*)$/);return m&&{pid:+m[1],parent:+m[2],cmd:m[3]}}).filter(Boolean);
const alive=pid=>{try{process.kill(pid,0);return true}catch{return false}};
(async()=>{let browser,server,identity,owned=[];try{
 server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});const proc=server.process();identity={session:'cross-chapter-review',controller:process.pid,browser:proc.pid,profile:proc.spawnargs.find(a=>a.startsWith('--user-data-dir=')).slice(16)};console.log(JSON.stringify(identity));
 browser=await chromium.connect(server.wsEndpoint());const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)errors.push(r.status()+' '+r.url())});
 const go=async target=>{await page.goto(new URL(target,base).href);await page.waitForLoadState('load');};
 const pager=()=>page.locator('.reading-route-pager:visible');
 const assertStep=async i=>{const expected=new URL(catalog.beginner.steps[i].path,base);assert.equal(new URL(page.url()).pathname.replace(/index\.html$/,''),expected.pathname.replace(/index\.html$/,''),'route step '+(i+1));assert.equal(new URL(page.url()).hash,expected.hash);assert.equal(new URL(page.url()).searchParams.get('learn'),'beginner');await pager().waitFor({timeout:6000});assert.equal(await pager().count(),1);assert.match(await page.locator('.active-reading-route').innerText(),new RegExp((i+1)+' / 17'));};
 const shot=async name=>{if(!shots)return;fs.mkdirSync(shots,{recursive:true});await page.screenshot({path:path.join(shots,name+'.png')});};
 await go('');assert.equal(await page.locator('[data-beginner-start]').count(),1);await shot('home-1440');
 await page.locator('#topic-paths>summary').click();
 assert.equal(await page.locator('#topic-paths tbody tr').count(),9);
 const deepLinks=await page.locator('#topic-paths td:nth-child(4) a').evaluateAll(nodes=>nodes.map(a=>a.pathname));assert.equal(new Set(deepLinks).size,28);assert.equal(deepLinks.length,28);
 await page.locator('#topic-paths>summary').click();await page.locator('[data-beginner-start]').click();
 for(let i=0;i<catalog.beginner.steps.length;i++){
  await assertStep(i);
  const chapter=i<8?'foundations':i<11?'engine-comparison':'sglang';
  assert(await page.locator(`.site-sidebar [data-chapter="${chapter}"]`).evaluate(e=>e.open),'current chapter opens');
  if(i===8||i===11)assert.equal(await page.locator('.site-sidebar .chapter-nav[open]').count(),1,'crossing chapters focuses the current chapter');
  if(catalog.beginner.steps[i].path.includes('engine-comparison/')){
   if(await page.locator('button[data-process-view="request"]').count())await page.locator('button[data-process-view="request"]').click();
   await page.locator('.ia-stepItem').last().click();assert.equal(await page.locator('.ia-stepItem[aria-current=step]').count(),1);assert.equal(await page.locator('.ia-topic-pager').isVisible(),false);
  }
  if(i===8){
   await pager().locator('a').first().click();await page.waitForLoadState('load');await assertStep(7);
   await pager().locator('.next').click();await page.waitForLoadState('load');await assertStep(8);
  }
  if(i===9){await page.reload();await assertStep(i);assert.equal(await page.locator('[data-topic-connection=prefix]').count(),1);}
  if(i<catalog.beginner.steps.length-1){await pager().locator('.next').click();await page.waitForLoadState('load');}
 }
 await pager().locator('.next').click();await page.waitForLoadState('load');assert.equal(new URL(page.url()).hash,'#reading-routes');
 console.log('PASS 17-step route across all three chapters, backward boundary, hash topics and reload.');
 // Changing topic through the single-page tabs must update the matching route pager.
 await go('engine-comparison/?learn=beginner#process');await page.locator('button.ia-chapterTab').nth(2).click();await page.waitForURL(u=>u.hash==='#prefix');await assertStep(9);
 await page.locator('button.ia-chapterTab').nth(1).click();await page.waitForURL(u=>u.hash==='#wire');assert.match(await page.locator('.active-reading-route').innerText(),/当前主题为选读/);assert.equal(await pager().count(),0);assert(await page.locator('.ia-topic-pager').isVisible());
 await page.goBack();await page.waitForURL(u=>u.hash==='#prefix');await assertStep(9);
 // Leaving the route changes navigation only, preserving the current experiment.
 await go('sglang/model-execution/?learn=beginner&view=graph&t=4.3');const root=page.locator('[data-model-flow]');await root.waitFor();const time=await root.getAttribute('data-time');
 await page.locator('[data-route-exit]').click();assert.equal(new URL(page.url()).searchParams.has('learn'),false);assert.equal(await root.getAttribute('data-time'),time);assert(await page.locator('.lesson-pager').isVisible());assert.equal(await pager().count(),0);
 console.log('PASS tab/history synchronization and route exit preserves teaching time.');
 // Default chapter reading also has an explicit topic-to-topic continuation.
 await go('engine-comparison/#process');
 for(const topic of ['process','wire','prefix','batch','crosscut','disagg','metrics']){
  await page.locator(`[data-topic-connection=${topic}]`).waitFor();assert(await page.locator('.ia-topic-pager').isVisible());
  assert((await page.locator('.ia-chapter-handoff a[href*="foundations/"]').count())>0);
  if(topic!=='metrics'){await page.locator('.ia-topic-pager .next').click();await page.waitForFunction(t=>document.querySelector('.ia-page').dataset.chapter===t,['wire','prefix','batch','crosscut','disagg','metrics'][['process','wire','prefix','batch','crosscut','disagg'].indexOf(topic)]);}
 }
 await page.locator('.ia-topic-pager .next').click();await page.waitForLoadState('load');assert(new URL(page.url()).pathname.endsWith('/sglang/system-overview/index.html'));
 console.log('PASS all seven topic handoffs and chapter-two to chapter-three continuation.');
 // Same-chapter navigation retains deliberate expansion and desktop scroll.
 await go('sglang/inference-overview/journey.html');await page.locator('.site-sidebar .topic-group,.site-sidebar .module-nav').evaluateAll(nodes=>nodes.forEach(n=>n.open=true));
 const nextLesson=page.locator('.site-sidebar [data-module=model-execution] .lesson-link').first();await nextLesson.scrollIntoViewIfNeeded();const savedTop=await page.locator('.site-sidebar').evaluate(e=>e.scrollTop);
 await nextLesson.click();await page.waitForLoadState('load');assert(Math.abs(await page.locator('.site-sidebar').evaluate(e=>e.scrollTop)-savedTop)<2);assert.equal(await page.locator('.site-sidebar .topic-group[open]').count(),5);assert.equal(await page.locator('.site-sidebar .module-nav[open]').count(),13);
 console.log('PASS same-chapter expansion and scroll remain intact.');
 for(const width of [1440,768,390,320])for(const target of ['', 'foundations/cache.html?learn=beginner','engine-comparison/?learn=beginner#batch','sglang/loop-journey/?learn=beginner']){
  await page.setViewportSize({width,height:1000});await go(target);
  if(!target)await page.locator('#topic-paths>summary').click();
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),width+'/'+target+' no outer overflow');
  if(width===390||width===1440)await shot((target?'lesson-'+target.split('/')[0]:'index')+'-'+width);
 }
 const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto(base);await plain.locator('#route-beginner>summary').click();assert.equal(await plain.locator('#route-beginner .route-steps a').count(),17);await plain.locator('#route-beginner .route-steps a').first().click();assert((await plain.locator('main').innerText()).includes('Tokenization'));assert(await plain.locator('.lesson-pager').isVisible());await plain.close();
 assert.deepEqual(errors,[]);console.log('PASS four widths, expanded cross-chapter index and no-JS reading.');
 }finally{if(identity){const list=processes(),set=new Set([identity.browser]);for(let i=0;i<8;i++)for(const p of list)if(set.has(p.parent))set.add(p.pid);owned=[...set];}await browser?.close().catch(()=>{});await server?.close().catch(()=>{});await new Promise(r=>setTimeout(r,200));for(const signal of ['SIGTERM','SIGKILL']){for(const pid of owned.filter(alive)){const p=processes().find(p=>p.pid===pid);if(p&&/Chrome|Chromium|chrome/.test(p.cmd))process.kill(pid,signal);}await new Promise(r=>setTimeout(r,200));}const remaining=owned.filter(alive);console.log(JSON.stringify({cleanup:remaining,profileExists:identity&&fs.existsSync(identity.profile)}));assert.deepEqual(remaining,[]);if(identity)assert(!fs.existsSync(identity.profile));}
})().catch(e=>{console.error(e);process.exitCode=1});
