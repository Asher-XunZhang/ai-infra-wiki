const assert=require('node:assert/strict');
const fs=require('node:fs');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=(process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:18765/').replace(/\/?$/,'/');

let browser;
let server;
let proc;
let profile;
let cleanupPromise;

const waitGone=(pid,timeout)=>{
 return new Promise(resolve=>{
  const started=Date.now();
  const check=()=>{
   try{process.kill(pid,0);}catch(error){if(error.code==='ESRCH'){resolve(true);return;}}
   if(Date.now()-started>=timeout){resolve(false);return;}
   setTimeout(check,50);
  };
  check();
 });
};

const cleanup=()=>cleanupPromise||(cleanupPromise=(async()=>{
 try{await browser?.close();}catch(error){console.error('browser close:',error.message);}
 try{await server?.close();}catch(error){console.error('server close:',error.message);}
 if(proc){
  let gone=await waitGone(proc.pid,2500);
  if(!gone){
   try{process.kill(proc.pid,'SIGTERM');}catch(error){if(error.code!=='ESRCH')console.error('SIGTERM:',error.message);}
   gone=await waitGone(proc.pid,1500);
  }
  if(!gone){
   try{process.kill(proc.pid,'SIGKILL');}catch(error){if(error.code!=='ESRCH')console.error('SIGKILL:',error.message);}
   gone=await waitGone(proc.pid,1000);
  }
  assert.equal(gone,true,`owned browser ${proc.pid} remains`);
  console.log(`CLEANED browser pid ${proc.pid}`);
 }
 if(profile&&fs.existsSync(profile))fs.rmSync(profile,{recursive:true,force:true,maxRetries:3,retryDelay:200});
})());

for(const signal of ['SIGINT','SIGTERM']){
 process.once(signal,()=>{cleanup().finally(()=>process.exit(signal==='SIGINT'?130:143));});
}

(async()=>{
 try{
  server=await chromium.launchServer({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE});
  proc=server.process();
  profile=proc.spawnargs.find(arg=>arg.startsWith('--user-data-dir='))?.slice('--user-data-dir='.length);
  console.log(JSON.stringify({session:'output-chain-reduced-motion',controller:process.pid,browser:proc.pid,profile}));
  browser=await chromium.connect(server.wsEndpoint());
  const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce'});
  const page=await context.newPage();
  const errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});

  await page.goto(base+'foundations/components.html#output-chain');
  const root=page.locator('[data-output-chain]');
  await root.waitFor();
  await page.waitForFunction(()=>document.querySelector('[data-output-chain]')?.dataset.ocReady==='true');
  assert.equal(await page.evaluate(()=>matchMedia('(prefers-reduced-motion: reduce)').matches),true,'reduced motion is emulated');

  const state=()=>root.evaluate(node=>({step:Number(node.dataset.step),state:node.dataset.ocState,progress:Number(node.dataset.ocProgress)}));
  const toggle=root.locator('[data-oc-toggle]');
  const status=root.locator('[data-oc-status]');
  assert.deepEqual(await state(),{step:0,state:'idle',progress:1},'the lesson starts at step 1');

  await toggle.click();
  assert.deepEqual(await state(),{step:0,state:'playing',progress:1},'play enters the playing state');
  await page.waitForTimeout(1700);
  const advanced=await state();
  assert.ok(advanced.step>=1,`reduced-motion playback must advance past step 0 (got ${advanced.step})`);
  assert.equal(advanced.state,'playing','reduced-motion playback keeps a real playing state');
  assert.equal(advanced.progress,1,'each reduced-motion step is fully rendered');

  await page.waitForFunction(()=>{
   const node=document.querySelector('[data-output-chain]');
   return node?.dataset.ocState==='idle'&&node?.dataset.step==='4'&&node?.dataset.ocProgress==='1.000';
  },null,{timeout:6000});
  assert.equal((await status.textContent()).includes('播放结束'),true,'reduced-motion playback reaches the endpoint');

  const learned=await root.textContent();
  for(const correction of ['没有把三通道索引当成二维坐标','不把分母标成','本页没有显示未平移的 Σexp(s)','本页不把它记成']){
   assert.equal(learned.includes(correction),false,`correction wording stays removed: ${correction}`);
  }
  assert.ok(learned.includes('稳定 softmax 的分母'),'softmax captions show the stable normalization denominator');
  assert.equal(await root.locator('.oc-math-block').nth(1).locator('math mfrac').count(),2,'attention explains both scaling and normalization');
  await root.locator('[data-oc-step="3"]').click();
  assert.equal(await root.locator('.oc-ffn-sum math').count(),1,'the FFN contribution sum is rendered as MathML');

  await root.locator('[data-oc-reset]').click();
  assert.deepEqual(await state(),{step:0,state:'idle',progress:1},'reset returns to step 1');
  await toggle.click();
  await toggle.click();
  const paused=await state();
  assert.equal(paused.state,'paused','the second click pauses reduced-motion playback');
  assert.equal(paused.progress,1,'pause keeps the current step fully rendered');
  await page.waitForTimeout(900);
  assert.deepEqual(await state(),paused,'pause stops the reduced-motion hold timer');
  await toggle.click();
  await page.waitForTimeout(900);
  const resumed=await state();
  assert.ok(resumed.step>paused.step&&resumed.state==='playing',`resume continues reduced-motion playback (step ${resumed.step}, ${resumed.state})`);
  await page.waitForFunction(()=>{
   const node=document.querySelector('[data-output-chain]');
   return node?.dataset.ocState==='idle'&&node?.dataset.step==='4';
  },null,{timeout:6000});

  await root.locator('[data-oc-step="2"]').click();
  assert.deepEqual(await state(),{step:2,state:'idle',progress:1},'manual steps still settle instantly without a false playing state');
  assert.deepEqual(errors,[],'no page or console errors');
  console.log('PASS reduced-motion output chain: playback advances and ends, pause holds, resume continues, manual steps stay idle.');
 }finally{
  await cleanup();
 }
})().catch(error=>{console.error(error);process.exitCode=1;});
