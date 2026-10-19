const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/',choose=require('./choose_motion_scene.cjs');
(async()=>{
 const server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})}),owned=server.process();
 console.log(JSON.stringify({session:'test_concept_focus_browser',controller:process.pid,browser:owned.pid,profile:owned.spawnargs.find(a=>a.startsWith('--user-data-dir='))}));
 const browser=await chromium.connect(server.wsEndpoint());
 try{
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  const film=page.locator('.mechanism-film'),svg=film.locator('svg'),seek=p=>film.locator('.motion-scrub input').evaluate((e,p)=>{e.value=p;e.dispatchEvent(new Event('input',{bubbles:true}));},p);
  for(const slug of ['metrics','batching','hardware']){
   await page.goto(base+'foundations/'+slug+'.html');const count=await choose.count(page);
   for(let v=0;v<count;v++){
    await choose(page,v);const max=Number(await film.locator('.motion-scrub input').getAttribute('max')),durations=JSON.parse(await film.getAttribute('data-step-durations'));
    assert.equal(durations.length,max);assert(durations.every(d=>Number.isFinite(d)&&d>0));
    for(const width of [320,768]){await page.setViewportSize({width,height:1100});
     for(let p=0;p<=max;p+=.5){await seek(p);assert(Math.abs(Number(await film.getAttribute('data-position'))-p)<.001,'nonuniform duration seek');
      const outside=await svg.evaluate(e=>{const b=e.getBoundingClientRect();return [...e.querySelectorAll('text')].filter(t=>{for(let a=t;a&&a!==e;a=a.parentElement)if(Number(a.getAttribute('opacity')??1)<.02)return false;const r=t.getBoundingClientRect();return t.textContent&&(r.left<b.left-2||r.right>b.right+2||r.top<b.top-2||r.bottom>b.bottom+2);}).map(t=>t.textContent);});assert.deepEqual(outside,[],slug+'/'+v+'/'+p+'/'+width);
     }
    }
    if(slug==='batching'&&v>=5){const chunk=[1,2,4,8][v-5];assert.equal(Number(await svg.getAttribute('data-processed')),8);assert.equal(Number(await svg.getAttribute('data-decode-outputs')),8/chunk);await seek(1.5);assert.equal(Number(await svg.getAttribute('data-processed')),0,'in-progress chunk is not completed');assert.equal(Number(await svg.getAttribute('data-decode-outputs')),0);}
   }
  }
  await page.goto(base+'foundations/hardware.html?mv=4&ms=1.5');
  const progress=id=>svg.locator('[data-motion-id='+id+']').getAttribute('data-progress').then(Number);
  for(let i=0;i<2;i++)for(let j=0;j<3;j++)assert.equal(await progress('kernel-'+i+'-'+j),0,'calculation does not precede submission');
  await seek(2);assert.equal(await progress('kernel-0-0'),1);assert.equal(await progress('kernel-0-1'),0);assert.equal(await progress('kernel-1-1'),.5);assert.equal(await progress('launch-0-1'),.75,'CPU submission overlaps GPU work');
  await seek(2.5);assert.equal(await svg.locator('text').filter({hasText:'本轮完成 · 输出 9'}).count(),1,'graph completes while ordinary submission is unfinished');
  await seek(3);assert.equal(await svg.locator('text').filter({hasText:'本轮完成 · 输出 9'}).count(),2);await seek(5);assert.equal(await svg.locator('text').filter({hasText:'本轮完成 · 输出 13'}).count(),2);
  await choose(page,5);await seek(2.5);for(let j=0;j<3;j++)assert.equal(await progress('kernel-0-'+j),await progress('kernel-1-'+j),'long kernels cover later CPU launches');
  await choose(page,2);await seek(2);assert.equal(await svg.locator('[data-motion-id^=resource-work-0-]').count(),8);assert.equal(await svg.locator('[data-motion-id^=resource-work-1-]').count(),6);assert.match(await svg.textContent(),/6 份 ÷ 4 份\/秒 = 1.5 秒/);
  await page.goto(base+'foundations/metrics.html?mv=0&ms=1.5');assert.equal(Number(await film.getAttribute('data-position')),1.5);assert.equal(Number(await svg.getAttribute('data-observed-time')),120);
  await choose(page,1);await seek(1.5);assert.equal(Number(await svg.getAttribute('data-observed-time')),140);await seek(4);assert.match(await svg.textContent(),/包间隔为 80/);
  await choose(page,3);await seek(4);assert.equal(await svg.locator('[data-included=true]').count(),18);assert.equal(await svg.locator('[data-percentile=true]').textContent(),'180');await seek(0);const unsorted=await svg.locator('[data-motion-id=sample-1]').getAttribute('transform');await seek(.5);assert.notEqual(await svg.locator('[data-motion-id=sample-1]').getAttribute('transform'),unsorted,'sorting is continuous');await seek(1);assert.equal(await svg.locator('[data-included=true]').count(),20);
  await choose(page,4);await seek(3.5);assert.equal(Number(await svg.getAttribute('data-completed')),3);assert.equal(Number(await svg.getAttribute('data-output-tokens')),6);await seek(4);assert.equal(Number(await svg.getAttribute('data-completed')),4);assert.equal(Number(await svg.getAttribute('data-output-tokens')),10);
  await page.goto(base+'foundations/metrics.html?fv=3&fs=3');assert.equal(await film.getByLabel('连续演示条件').inputValue(),'3');assert.equal(Number(await film.getAttribute('data-position')),3,'old metric URLs retain meaning');
  // Single-step playback uses the remaining duration, including after changing speed mid-step.
  await choose(page,0);await page.setViewportSize({width:1440,height:1100});await film.scrollIntoViewIfNeeded();await seek(.2);
  const speed=film.getByLabel('演示速度');await speed.evaluate(e=>{e.value=.25;e.dispatchEvent(new Event('input',{bubbles:true}));});await film.locator('[data-motion-action=next]').click();await page.waitForTimeout(150);const slow=Number(await film.getAttribute('data-position'));assert(slow>.2&&slow<.5);await speed.evaluate(e=>{e.value=3;e.dispatchEvent(new Event('input',{bubbles:true}));});await page.waitForTimeout(700);assert.equal(Number(await film.getAttribute('data-position')),1);
  await seek(1.25);await film.locator('[data-motion-action=play]').click();await page.waitForTimeout(100);await film.locator('[data-motion-action=play]').click();const frozen=await film.getAttribute('data-position');await page.waitForTimeout(180);assert.equal(await film.getAttribute('data-position'),frozen);
  await page.emulateMedia({reducedMotion:'reduce'});await seek(2.2);await film.locator('[data-motion-action=previous]').click();assert.equal(Number(await film.getAttribute('data-position')),1);
  assert.deepEqual(errors,[]);console.log('PASS concept focus: 20 variants, all half-steps at 2 widths, work/rate conservation, asynchronous launch/dependency timing, long-kernel comparison, chunk conservation, sample ordering/failures, measurement boundaries, URL migration and nonuniform timing/rate/seek.');
 }finally{await browser.close();await server.close();try{process.kill(owned.pid,0);throw Error('owned browser remains');}catch(e){if(e.code!=='ESRCH')throw e;}}
})().catch(e=>{console.error(e);process.exit(1);});
