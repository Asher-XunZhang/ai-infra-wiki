'use strict';
// Cross-lesson contract: depth changes cannot seek, pause, rebuild the model, or
// make data readable earlier. Mechanism tests keep checking the original models.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
const shots=process.env.DEPTH_SCREENSHOT_DIR;
const pages=[
 ['system-overview/',true],['request-runtime/',false],['loop-journey/',true],['loop-journey/#overlap',true],['loop-journey/#pp',true],
 ['request-state/',true],['request-state/?mode=overlap',true],['request-state/?mode=cancel',true],
 ['inference-overview/journey.html',false],['inference-overview/scheduling.html',false],['inference-overview/transformer.html',false],['model-execution/',true],['model-execution/?view=graph',true],['model-execution/?execution-input=decode',true],
 ['inference-overview/kv-cache.html',false],['kv-memory/',false],['kv-memory/?mode=reuse',false],['scheduling/',false],
 ...['tp','pp','dp','dpa','ep','sp','pcp','dcp'].map(m=>['parallelism/?mode='+m,['pp','dpa','pcp','dcp'].includes(m)]),
 ...['ipc','tensor','collective','kv','ready','release'].map(m=>['communication/?mode='+m,true]),
 ['inference-overview/deployment.html',false],['pd-prefill-lifecycle/',true],['pd-dataflow/',true],
 ...['bootstrap','release','hicache'].map(m=>['pp-consensus/?mode='+m,true]),
 ...['speculative','grammar','combined'].map(m=>['advanced-generation/?mode='+m,true]),['serving-operations/',true]
];
const processes=()=>cp.execFileSync('ps',['-axo','pid=,ppid=,command='],{encoding:'utf8'}).trim().split('\n').map(s=>{const m=s.trim().match(/^(\d+)\s+(\d+)\s+(.*)$/);return m&&{pid:+m[1],parent:+m[2],cmd:m[3]}}).filter(Boolean);
const alive=pid=>{try{process.kill(pid,0);return true}catch{return false}};
(async()=>{let browser,server,identity,owned=[];try{
 server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 const proc=server.process();identity={session:'learning-depth-review',controller:process.pid,browser:proc.pid,profile:proc.spawnargs.find(a=>a.startsWith('--user-data-dir=')).slice(16)};console.log(JSON.stringify(identity));
 browser=await chromium.connect(server.wsEndpoint());const page=await browser.newPage({viewport:{width:1440,height:1000}});let errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)errors.push(r.status()+' '+r.url())});
 if(shots)fs.mkdirSync(shots,{recursive:true});
 for(const [url,full]of pages.filter(([url])=>!process.env.DEPTH_CASE_FILTER||url.includes(process.env.DEPTH_CASE_FILTER))){errors=[];await page.goto(base+'sglang/'+url);const root=page.locator('[data-learning-depth]').first();await root.waitFor();await page.waitForFunction(()=>Number.isFinite(Number(document.querySelector('[data-learning-depth]').dataset.time)));
  assert.equal(await root.getAttribute('data-depth'),full?'overview':'modules',url+' initial depth');
  if(full)assert((await root.locator('.ld-overview svg').count())>0,url+' initial projection');
  const initial=await root.getAttribute('data-time');assert.equal(Number(initial),0,url+' initial clock');
  const seek=async fraction=>root.evaluate((el,f)=>{const slider=[...el.querySelectorAll('input[type=range]')].find(e=>e.getClientRects().length);if(!slider)throw Error('missing visible clock');slider.value=Number(slider.min||0)+(Number(slider.max)-Number(slider.min||0))*f;slider.dispatchEvent(new Event('input',{bubbles:true}));},fraction);
  for(const f of [0.27,.61,.999]){await seek(f);const time=await root.getAttribute('data-time');assert(Number(time)>0,url+' seeks');
   for(const level of ['source','modules',...(full?['overview']:[])]){await root.locator(`[data-depth-choice=${level}]`).click();assert.equal(await root.getAttribute('data-time'),time,url+' depth preserves clock');assert.equal(await root.getAttribute('data-playing'),'false',url+' depth preserves pause');if(level==='source')assert((await root.locator('.ld-source-panel').innerText()).length>50,url+' current source state');}
   const invalid=await root.evaluate(e=>[...e.querySelectorAll('svg [d],svg [width],svg [height],svg [stroke-dashoffset]')].filter(n=>[...n.attributes].some(a=>/NaN|Infinity|undefined/.test(a.value))).slice(0,3).map(n=>n.outerHTML));assert.deepEqual(invalid,[],url+' finite geometry');
  }
  if(shots){await seek(.61);await root.locator('.ld-controls').evaluate(e=>scrollTo({top:scrollY+e.getBoundingClientRect().top-100,behavior:'instant'}));await page.screenshot({path:path.join(shots,url.replace(/[^a-z0-9]+/gi,'-')+'.png')});}
  await seek(.2);await root.evaluate(el=>{const b=[...el.querySelectorAll('button')].find(e=>e.getClientRects().length&&/播放|继续/.test(e.textContent.trim()));if(!b)throw Error('missing play button');b.click();});
  await page.waitForTimeout(80);const runningTime=Number(await root.getAttribute('data-time'));await root.locator('[data-depth-choice=source]').click();assert.equal(await root.getAttribute('data-playing'),'true',url+' depth preserves running state');await page.waitForTimeout(80);assert(Number(await root.getAttribute('data-time'))>runningTime,url+' clock continues');await seek(.61);
  // The permalink must recover the same view and clock in a fresh render.
  await root.locator('[data-depth-choice=source]').click();const before=Number(await root.getAttribute('data-time'));await page.reload();await root.waitFor();assert(Math.abs(Number(await root.getAttribute('data-time'))-before)<.02,url+' permalink recovers clock');assert.equal(await root.getAttribute('data-depth'),'source');
  await page.setViewportSize({width:390,height:844});await page.emulateMedia({reducedMotion:'reduce'});await root.locator('[data-depth-choice=modules]').click();assert(await root.locator('.ld-controls').isVisible());
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),url+' viewport overflow');
  assert.deepEqual(errors,[],url+' console / resources');await page.emulateMedia({reducedMotion:'no-preference'});await page.setViewportSize({width:1440,height:1000});console.log('PASS',url);
 }
 // Chapter 2 has stage snapshots, not fabricated measured time intervals.
 for(const topic of ['process','wire','prefix','batch','disagg','crosscut']){await page.goto(base+'engine-comparison/#'+topic);const root=page.locator('.ia-page');await root.waitFor();const stepButtons=page.locator('.ia-stepRail button');if(await stepButtons.count())await stepButtons.nth(2).click();const step=await root.getAttribute('data-step');
  await root.locator('[data-pulse-toggle]').click();const pulse=Number(await root.getAttribute('data-pulse-time'));
  for(const level of ['source','modules',...(!['process','crosscut'].includes(topic)?['overview']:[])]){await root.locator(`[data-depth-choice=${level}]`).click();assert.equal(await root.getAttribute('data-step'),step);assert(Math.abs(Number(await root.getAttribute('data-pulse-time'))-pulse)<.05);}
  if(shots){await page.locator('.ia-stage').evaluate(e=>scrollTo({top:scrollY+e.getBoundingClientRect().top-100,behavior:'instant'}));await page.screenshot({path:path.join(shots,'chapter2-'+topic+'.png')});}assert.deepEqual(errors,[]);console.log('PASS chapter2',topic);
 }
 // Active transfer paths stay complete; only the dash phase changes with time.
 for(const url of ['communication/?mode=tensor&t=8.5','pp-consensus/?mode=bootstrap&t=3.5','advanced-generation/?mode=grammar&t=6']){
  await page.goto(base+'sglang/'+url+'&depth=modules');const root=page.locator('[data-learning-depth]');await root.waitFor();
  const slider=root.locator('input[type=range]').first(),max=Number(await slider.getAttribute('max'));let found=false;
  for(let t=.4;t<max;t+=.4){await slider.evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));},t);const wires=root.locator('path[data-flow-active=true]');if(!await wires.count())continue;
   const a=await wires.first().evaluate(e=>({d:e.getAttribute('d'),offset:getComputedStyle(e).strokeDashoffset}));await page.waitForTimeout(60);assert.equal(await wires.first().evaluate(e=>getComputedStyle(e).strokeDashoffset),a.offset,'paused pulse is stable');
   await slider.evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));},t+.04);const b=await wires.first().evaluate(e=>({d:e.getAttribute('d'),offset:getComputedStyle(e).strokeDashoffset}));assert.equal(a.d,b.d,'full transfer geometry is stable');assert.notEqual(a.offset,b.offset,'pulse follows the teaching clock');found=true;break;
  }assert(found,url+' has an observable transfer pulse');console.log('PASS pulse',url);
 }
 for(const url of ['system-overview/','request-state/?mode=overlap','pd-dataflow/','inference-overview/deployment.html','advanced-generation/?mode=combined']){
  await page.goto(base+'sglang/'+url);await page.locator('[data-learning-depth]').waitFor();await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.locator('.ld-controls').first().evaluate(e=>scrollTo({top:scrollY+e.getBoundingClientRect().top-100,behavior:'instant'}));if(shots)await page.screenshot({path:path.join(shots,'dark-'+url.replace(/[^a-z0-9]+/gi,'-')+'.png')});await page.evaluate(()=>document.documentElement.dataset.theme='light');
 }
 // Existing mature levels and pure source pages retain their direct entries.
 for(const url of ['performance-engineering/','pd-prefill-pp-loop/quick.html','pd-prefill-pp-loop/','pd-prefill-pp-loop/notes.html']){await page.goto(base+'sglang/'+url);assert.equal(await page.locator('[data-learning-depth]').count(),0);assert((await page.locator('main').innerText()).length>100);console.log('PASS retained',url);}
 }finally{if(identity){const list=processes(),set=new Set([identity.browser]);for(let i=0;i<8;i++)for(const p of list)if(set.has(p.parent))set.add(p.pid);owned=[...set];}await browser?.close().catch(()=>{});await server?.close().catch(()=>{});await new Promise(r=>setTimeout(r,250));for(const signal of ['SIGTERM','SIGKILL']){for(const id of owned.filter(alive)){const p=processes().find(p=>p.pid===id);if(p&&/Chrome|Chromium|chrome/.test(p.cmd))process.kill(id,signal);}await new Promise(r=>setTimeout(r,200));}const remaining=owned.filter(alive);console.log(JSON.stringify({cleanup:remaining,profileExists:identity&&fs.existsSync(identity.profile)}));assert.deepEqual(remaining,[],'owned browser exited');if(identity)assert(!fs.existsSync(identity.profile),'owned profile removed');}
})().catch(e=>{console.error(e);process.exitCode=1});
