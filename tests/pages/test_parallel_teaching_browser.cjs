const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {execFileSync}=require('node:child_process');
const dir=process.env.MOTION_SCREENSHOT_DIR;if(dir)fs.mkdirSync(dir,{recursive:true});
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:18765/';
let server,browser,owned,profilePath,shuttingDown;
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const pidState=pid=>{try{process.kill(pid,0);return 'alive';}catch(error){return error.code==='ESRCH'?'gone':`error:${error.code||error.message}`;}};
function profileProcesses(){
 if(!profilePath)return [];
 try{return execFileSync('pgrep',['-fl',profilePath],{encoding:'utf8'}).trim().split('\n').filter(Boolean).map(line=>line.match(/^(\d+)\s+(.*)$/)).filter(Boolean).map(m=>({pid:Number(m[1]),command:m[2]})).filter(entry=>pidState(entry.pid)!=='gone');}
 catch(error){if(error.status===1)return [];throw error;}
}
async function waitGone(pid,timeout){const end=Date.now()+timeout;let state=pidState(pid);while(state!=='gone'&&Date.now()<end){await delay(80);state=pidState(pid);}return state;}
function signal(pid,sig){try{process.kill(pid,sig);return null;}catch(error){return error.code==='ESRCH'?null:error.code||error.message;}}
function ownedProfile(candidate){
 if(!candidate||!/playwright/i.test(path.basename(candidate)))return false;
 const roots=[path.resolve(os.tmpdir())];try{roots.push(fs.realpathSync(os.tmpdir()));}catch{}
 const files=[path.resolve(candidate)];try{files.push(fs.realpathSync(candidate));}catch{}
 return files.some(file=>roots.some(root=>file.startsWith(root+path.sep)));
}
async function shutdown(reason){
 if(shuttingDown)return shuttingDown;
 shuttingDown=(async()=>{
  const failures=[];
  await browser?.close().catch(error=>failures.push(`browser close: ${error.message}`));
  await server?.close().catch(error=>failures.push(`server close: ${error.message}`));
  let state=owned?.pid?await waitGone(owned.pid,1500):'not-created';
  for(const sig of ['SIGTERM','SIGKILL']){
   if(state==='gone')break;
   if(!profileProcesses().some(entry=>entry.pid===owned.pid)){failures.push(`pid ${owned.pid} is not our browser`);break;}
   const error=signal(owned.pid,sig);if(error)failures.push(`browser ${sig}: ${error}`);
   state=await waitGone(owned.pid,sig==='SIGTERM'?1200:800);
  }
  if(owned?.pid&&state!=='gone')failures.push(`browser pid ${owned.pid} remains (${state})`);
  let remaining=profileProcesses();
  for(const sig of ['SIGTERM','SIGKILL']){
   for(const entry of remaining){const error=signal(entry.pid,sig);if(error)failures.push(`profile ${entry.pid} ${sig}: ${error}`);}
   await delay(180);remaining=profileProcesses();if(!remaining.length)break;
  }
  if(remaining.length)failures.push(`profile processes remain: ${remaining.map(entry=>entry.pid).join(',')}`);
  let removed=!profilePath||!fs.existsSync(profilePath);
  if(profilePath&&!removed&&ownedProfile(profilePath)&&!remaining.length){
   try{fs.rmSync(profilePath,{recursive:true,force:true});}catch(error){failures.push(`profile: ${error.code||error.message}`);}
   removed=!fs.existsSync(profilePath);
  }
  console.log(JSON.stringify({event:'cleanup',reason,controller:process.pid,browser:owned?.pid??null,browserState:state,profile:profilePath,profileRemoved:removed,remaining:remaining.length}));
  if(failures.length)throw new Error(`cleanup incomplete (${reason}): ${failures.join('; ')}`);
 })();
 return shuttingDown;
}
for(const [sig,code] of [['SIGINT',130],['SIGTERM',143]])process.once(sig,()=>shutdown(sig).then(()=>process.exit(code)).catch(error=>{console.error(error);process.exit(1);}));
(async()=>{console.log('controller PID',process.pid);
 server=await chromium.launchServer({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE});
 owned=server.process();profilePath=owned.spawnargs.find(a=>a.startsWith('--user-data-dir='))?.slice('--user-data-dir='.length)||null;
 console.log(JSON.stringify({session:'parallel-teaching',controller:process.pid,browser:owned.pid,profile:profilePath}));
 browser=await chromium.connect(server.wsEndpoint());
 try{
const page=await browser.newPage(),errors=[],overflow=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto(base+'foundations/parallel.html');const film=page.locator('.mechanism-film'),sel=film.locator('select[aria-label="连续演示条件"]');
const seek=async p=>page.locator('.motion-scrub input').evaluate((e,p)=>{e.value=p;e.dispatchEvent(new Event('input',{bubbles:true}));},p);
const families=['TP','TP','PP','PP','DP','EP','EP','SP','PCP','PCP','DCP','DCP'];
for(let v=0;v<12;v++){const family=families[v];await page.locator(`[data-parallel-family=${family}]`).click();if(!['DP','SP'].includes(family))await sel.selectOption(String(v));assert.equal(await film.getAttribute('data-position'),'0.0000');
assert.equal(await page.locator('.parallel-strategies [aria-pressed=true]').count(),1);assert.equal(await sel.locator('option').count(),families.filter(f=>f===family).length);
const max=Number(await page.locator('.motion-scrub input').getAttribute('max'));
for(const width of [320,390,768,1440]){await page.setViewportSize({width,height:1000});await page.waitForTimeout(40);for(let p=0;p<=max;p+=.5){await seek(p);const bad=await film.locator('svg').evaluate(svg=>{const b=svg.getBoundingClientRect();return [...svg.querySelectorAll('text')].filter(e=>{for(let a=e;a&&a!==svg;a=a.parentElement)if(Number(a.getAttribute('opacity')??1)<.02)return false;return !!e.textContent;}).filter(e=>{const r=e.getBoundingClientRect();return r.left<b.left-2||r.right>b.right+2||r.top<b.top-2||r.bottom>b.bottom+2;}).map(e=>e.textContent);});if(bad.length)overflow.push({v,width,p,bad});}
assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
if(dir&&[390,1440].includes(width))for(const step of v===0||v===1?[0,2,3.5,5.5,max]:[0,max]){await seek(step);await film.screenshot({path:`${dir}/v${v}-${width}-p${step}.png`});}
}
}
// EP is one integrated canvas: no separate adoption panel repeats the same
// single-card / two-card comparison above the player.
await page.locator('[data-parallel-family=EP]').click();await sel.selectOption('5');await seek(0);
assert.equal(await page.locator('.parallel-adoption').count(),0,'EP has no duplicate adoption panel');
const epSvg=film.locator('svg');
assert.equal(await epSvg.getAttribute('data-ep-mode'),'balanced');
assert.equal(await epSvg.getAttribute('data-ep-selected'),'0,2');
assert.equal(await epSvg.getAttribute('data-ep-rank-loads'),'4/4');
assert.equal(await epSvg.getAttribute('data-ep-result'),'pending');assert.equal(await epSvg.getAttribute('data-ep-weight-result'),'3');
await sel.selectOption('6');
assert.equal(await epSvg.getAttribute('data-ep-mode'),'hotspot');
assert.equal(await epSvg.getAttribute('data-ep-selected'),'0,1');
assert.equal(await epSvg.getAttribute('data-ep-rank-loads'),'6/2');
assert.equal(await epSvg.getAttribute('data-ep-result'),'pending');assert.equal(await epSvg.getAttribute('data-ep-weight-result'),'2.5');
await seek(5);assert.equal(await epSvg.getAttribute('data-ep-result'),'2.5');assert.match(await epSvg.textContent(),/2 × 0\.75 \+ 4 × 0\.25 = 2\.5/);
// Switching mechanisms resets playback; URL restores both semantic levels.
await page.locator('[data-parallel-family=TP]').click();await seek(0);
const actor=page.locator('[data-motion-id="tp-x0-0-0"]');const x0=await actor.getAttribute('transform');await seek(2.25);const xm=await actor.getAttribute('transform');await seek(3);const x1=await actor.getAttribute('transform');assert.notEqual(x0,xm);assert.notEqual(xm,x1);assert.notEqual(x0,x1);
await seek(4);assert.deepEqual(await page.locator('[data-motion-id^="tp-local0-"] text').allTextContents(),['2','0','6','0']);
assert.deepEqual(await page.locator('[data-motion-id^="tp-local1-"] text').allTextContents(),['2','6','4','12']);const rangeMax=Number(await page.locator('.motion-scrub input').getAttribute('max'));await seek(rangeMax);assert.deepEqual(await page.locator('[data-motion-id^="tp-final0-"] text').allTextContents(),['4','6','10','12']);
await sel.selectOption('1');await seek(4);assert.deepEqual(await page.locator('[data-motion-id^="tp-local0-"] text').allTextContents(),['4','10']);assert.deepEqual(await page.locator('[data-motion-id^="tp-local1-"] text').allTextContents(),['6','12']);
await page.locator('[data-parallel-family=DCP]').focus();await page.keyboard.press('Enter');assert.equal(await film.getAttribute('data-position'),'0.0000');await sel.selectOption('11');await seek(2.5);await page.reload();await film.waitFor();assert.equal(await page.locator('[data-parallel-family=DCP]').getAttribute('aria-pressed'),'true');assert.equal(await sel.inputValue(),'11');assert.equal(await film.getAttribute('data-position'),'2.5000');
await page.locator('[data-parallel-family=DP]').click();await seek(2);const dp=page.locator('[data-motion-id=dp-R1]');const before=await dp.getAttribute('transform');await film.locator('[data-motion-action=play]').click();await page.waitForTimeout(400);await film.locator('[data-motion-action=play]').click();const paused=await dp.getAttribute('transform');assert.notEqual(before,paused);await page.waitForTimeout(150);assert.equal(await dp.getAttribute('transform'),paused);await film.locator('[data-motion-action=reset]').click();assert.equal(await film.getAttribute('data-position'),'0.0000');
await page.emulateMedia({reducedMotion:'reduce'});await film.locator('[data-motion-action=next]').click();assert.equal(await film.getAttribute('data-position'),'1.0000');await page.emulateMedia({reducedMotion:'no-preference'});
await page.locator('[data-parallel-family=PP]').click();
const ppGate=()=>film.locator('[data-pp-gate]'),ppCell=id=>page.locator(`[data-motion-id="${id}"]`);
assert.equal(await page.locator('.pp-tab').evaluateAll(gs=>gs.every(g=>{const r=g.querySelector('rect').getBoundingClientRect(),t=g.querySelector('text').getBoundingClientRect();return t.left>=r.left-1&&t.right<=r.right+1&&t.top>=r.top-1&&t.bottom<=r.bottom+1;})),true,'1/2 MB labels stay inside their buttons');
await seek(3.5);assert.equal(await ppGate().getAttribute('data-pp-inflight'),'true','MB1 activation is in flight after L2');assert.equal(await ppGate().getAttribute('data-pp-ready'),'false','L3 cannot run before MB1 activation arrives');assert.equal(await ppCell('pp-stage0-1').getAttribute('data-pp-state'),'complete','MB1 L2 completes before the transfer');assert.equal(await ppCell('pp-stage1-2').getAttribute('data-pp-state'),'hidden','L3 is hidden while the activation is in flight');
await page.getByLabel('PP 部署位置').selectOption('cross-host');assert.match(await film.locator('svg').textContent(),/跨主机/);assert.equal(await ppGate().getAttribute('data-pp-ready'),'false','cross-host keeps the activation dependency');assert.equal(await ppCell('pp-stage1-2').getAttribute('data-pp-state'),'hidden');
await seek(3.995);assert.equal(await ppGate().getAttribute('data-pp-ready'),'false','p=3.995 is just before the MB1 arrival at 4.0');assert.equal(await ppGate().getAttribute('data-pp-inflight'),'true','MB1 is still in flight just before arrival');assert.equal(await ppCell('pp-stage1-2').getAttribute('data-pp-state'),'hidden','L3 cannot start before the MB1 transfer reaches 1');
await seek(4);assert.equal(await ppGate().getAttribute('data-pp-ready'),'true','MB1 readiness begins exactly at arrival');assert.equal(await ppGate().getAttribute('data-pp-inflight'),'false','arrival clears the in-flight state');assert.equal(await ppCell('pp-stage1-2').getAttribute('data-pp-state'),'active','L3 starts at the MB1 arrival boundary');
await seek(5.995);assert.equal(await ppGate().getAttribute('data-pp-ready'),'false','p=5.995 is just before the MB2 arrival at 6.0');assert.equal(await ppGate().getAttribute('data-pp-inflight'),'true','MB2 is still in flight just before arrival');assert.equal(await ppCell('pp-stage1-4').getAttribute('data-pp-state'),'hidden','L3 for MB2 cannot start before its transfer reaches 1');
await seek(6);assert.equal(await ppGate().getAttribute('data-pp-ready'),'true','MB2 readiness begins exactly at arrival');assert.equal(await ppGate().getAttribute('data-pp-inflight'),'false','MB2 arrival clears the in-flight state');assert.equal(await ppCell('pp-stage1-4').getAttribute('data-pp-state'),'active','MB2 L3 starts at the arrival boundary');
await seek(4.2);assert.equal(await ppGate().getAttribute('data-pp-ready'),'true','L3 is ready after MB1 activation arrives');assert.equal(await ppCell('pp-stage1-2').getAttribute('data-pp-state'),'active');assert.equal(await ppCell('pp-stage1-2').getAttribute('data-pp-mb'),'1');
await seek(8);assert.equal(await ppCell('pp-stage1-5').getAttribute('data-pp-state'),'complete','two stages finish their sixth cell in slot 6');assert.equal(await ppCell('pp-single-6').getAttribute('data-pp-state'),'active','single card continues at cell 7');assert.match(await film.locator('svg').textContent(),/已完成 · 第 6 格 · 单卡继续/);assert.equal(JSON.parse(await film.getAttribute('data-step-durations')).length,Number(await page.locator('.motion-scrub input').getAttribute('max')),'stepDurations covers every frame transition');
await sel.selectOption('3');await seek(6);assert.equal(await page.locator('.mechanism-film [data-pp-mb="2"]').count(),0,'n1 has no MB2 work cells');assert.equal(await page.locator('.mechanism-film [data-pp-state="active"]').count(),0,'n1 terminal has no active cell');assert.equal(await ppCell('pp-single-3').getAttribute('data-pp-state'),'complete');assert.equal(await ppCell('pp-stage1-3').getAttribute('data-pp-state'),'complete');assert.match(await ppCell('pp-single-3').getAttribute('aria-label'),/已完成/);assert.match(await film.locator('svg').textContent(),/已完成 · 4 \/ 4 格/);assert.match(await film.locator('svg').textContent(),/8 GPU·格.*气泡 4/);assert.doesNotMatch(await ppGate().textContent(),/MB2|两个 MB/);assert.equal(JSON.parse(await film.getAttribute('data-step-durations')).length,Number(await page.locator('.motion-scrub input').getAttribute('max')),'n1 stepDurations covers every frame transition');
for(const width of [390,1440]){await page.setViewportSize({width,height:1000});for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.locator('[data-parallel-family=TP]').click();await seek(5.5);if(dir)await film.locator('svg').screenshot({path:`${dir}/tp-detail-${width}-${theme}.png`});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);}}
await page.evaluate(()=>document.documentElement.dataset.theme='light');
if(dir)for(const id of ['parallel-overview','read-pcp','read-dcp'])await page.locator('#'+id).screenshot({path:`${dir}/${id}.png`});
const plain=await browser.newPage({javaScriptEnabled:false});await plain.goto(base+'foundations/parallel.html');assert.equal(await plain.locator('.parallel-reading-section').count(),8);await plain.locator('#read-pcp summary').click();assert.match(await plain.locator('#read-pcp').innerText(),/padding/);await plain.locator('#parallel-glossary summary').click();assert.match(await plain.locator('#parallel-glossary').innerText(),/Log-Sum-Exp/);await plain.close();
if(dir)fs.writeFileSync(dir+'/findings.json',JSON.stringify({errors,overflow},null,2));assert.deepEqual(errors,[]);assert.deepEqual(overflow,[]);console.log('PASS 12 parallel scenarios; 4 widths and half-steps; 2 themes; TP arithmetic/geometry; PP causal activation gating, 1/2 MB terminal states and duration contract; keyboard, URL restore, pause/reset, reduced motion and no-JS reading.');
}finally{try{await shutdown('complete');}catch(error){console.error(error);process.exitCode=1;}}})().catch(e=>{console.error(e);process.exitCode=1});
