/* Systems readability regression: batching, hardware and speculation. */
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const choose=require('./choose_motion_scene.cjs');
const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
const chrome=process.env.CHROMIUM_EXECUTABLE;
const WIDTHS=[320,390,768,1440,1994],THEMES=['light','dark'];
const seek=(page,p)=>page.locator('.motion-scrub input').evaluate((e,p)=>{e.value=p;e.dispatchEvent(new Event('input',{bubbles:true}));},p);
const film=page=>page.locator('.mechanism-film');

/* Screen-space readability: painted text vs other painted text and movable
 * token/object groups. A token's own label is inside its group, so it is not
 * an occlusion. Boxes/borders themselves are intentionally not compared. */
function scan(page,positions){
 return page.evaluate(positions=>{
  const film=document.querySelector('.mechanism-film'),svg=film.querySelector('svg'),range=film.querySelector('.motion-scrub input');
  const vis=e=>{if(!e.textContent.trim())return false;for(let a=e;a;a=a.parentElement){const attr=Number(a.getAttribute?.('opacity')??1);if(attr<.05)return false;const cs=getComputedStyle(a);if(cs.display==='none'||cs.visibility==='hidden'||Number(cs.opacity)<.02)return false;}return true;};
  const R=e=>{const r=e.getBoundingClientRect();return{l:r.left,t:r.top,r:r.right,b:r.bottom};};
  const hit=(a,b)=>{const x=Math.min(a.r,b.r)-Math.max(a.l,b.l),y=Math.min(a.b,b.b)-Math.max(a.t,b.t);return x>2.5&&y>2.5?{x:+x.toFixed(1),y:+y.toFixed(1)}:null;};
  const bad=[];
  for(const p of positions){
   range.value=String(p);range.dispatchEvent(new Event('input',{bubbles:true}));
   const bounds=R(svg),texts=[...svg.querySelectorAll('text')].filter(vis).map(e=>({e,text:e.textContent.trim().replace(/\s+/g,' ').slice(0,60),r:R(e)}));
   const objs=[...svg.querySelectorAll('[data-motion-id]')].filter(vis).map(e=>({e,id:e.getAttribute('data-motion-id'),r:R(e)}));
   const tt=[],to=[],out=[];
   for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){if(texts[i].text===texts[j].text)continue;const h=hit(texts[i].r,texts[j].r);if(h)tt.push({a:texts[i].text,b:texts[j].text,...h});}
   for(const t of texts)for(const o of objs){if(o.e.contains(t.e))continue;if(t.e.closest('[data-motion-id]'))continue;const h=hit(t.r,o.r);if(h)to.push({text:t.text,id:o.id,...h});}
   for(const t of texts){const r=t.r;if(r.l<bounds.l-2.5||r.r>bounds.r+2.5||r.t<bounds.t-2.5||r.b>bounds.b+2.5)out.push(t.text);}
   if(tt.length||to.length||out.length)bad.push({p,tt:tt.slice(0,5),to:to.slice(0,5),out:out.slice(0,4)});
   if(bad.length>=8)break;
  }
  return bad;
 },positions);
}
const visible=page=>page.evaluate(()=>[...document.querySelectorAll('.mechanism-film [data-motion-id^="committed-output"]')].filter(e=>Number(e.getAttribute('opacity'))>.5).length);

(async()=>{
 const server=await chromium.launchServer({headless:true,...(chrome?{executablePath:chrome}:{})}),owned=server.process();
 console.log(JSON.stringify({session:'test_readability_systems_browser',controller:process.pid,browser:owned.pid,profile:owned.spawnargs.find(a=>a.startsWith('--user-data-dir='))}));
 const browser=await chromium.connect(server.wsEndpoint());
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100},reducedMotion:'no-preference'}),errors=[];page.on('pageerror',e=>errors.push(e.message));
  const expected={batching:9,hardware:6,speculation:3};
  for(const [course,count] of Object.entries(expected)){
   await page.goto(base+'foundations/'+course+'.html');await page.locator('.mechanism-film svg').waitFor();
   assert.equal(await choose.count(page),count,course+' scene count');
   for(let v=0;v<count;v++){
    await choose(page,v);
    const max=Number(await page.locator('.motion-scrub input').getAttribute('max')),positions=[];
    for(let p=0;p<=max+1e-6;p+=.5)positions.push(+p.toFixed(2));if(positions.at(-1)!==max)positions.push(max);
    for(const width of WIDTHS){await page.setViewportSize({width,height:1100});await page.waitForTimeout(90);
     for(const theme of THEMES){await page.evaluate(t=>{document.documentElement.dataset.theme=t;},theme);const bad=await scan(page,positions);assert.deepEqual(bad,[],`${course}/${v}/${width}/${theme} screen-space text occlusion or clipping`);}
    }
   }
  }

  // Budget/height-specific regression: at 768x1000 the hardware stage is
  // shorter, so the camera enlarges SVG type. Scene 5's lane captions must stay
  // outside the event-label column through both graph rounds.
  await page.setViewportSize({width:768,height:1000});await page.evaluate(()=>{document.documentElement.dataset.theme='light';});
  await page.goto(base+'foundations/hardware.html');await page.locator('.mechanism-film svg').waitFor();
  await choose(page,5);
  const tabletPositions=[];for(let p=1.5;p<=5+1e-6;p+=.5)tabletPositions.push(+p.toFixed(2));
  const tabletBad=await scan(page,tabletPositions);
  assert.deepEqual(tabletBad,[],'hardware/5/768x1000 timeline captions must not be covered by A/图 or other event text');
  assert.equal(await page.locator('.mechanism-film svg').getAttribute('data-graph-clock')!==null,true,'graph clock state remains exposed');

  // Numbers and identities at the final state, at the narrow reading width.
  await page.setViewportSize({width:390,height:1100});await page.evaluate(()=>{document.documentElement.dataset.theme='light';});
  await page.goto(base+'foundations/batching.html');await page.locator('.mechanism-film svg').waitFor();
  await choose(page,0);await seek(page,8);assert.deepEqual(await page.locator('[data-motion-id]').evaluateAll(es=>es.map(e=>e.dataset.count)),['1.000','3.000','1.000']);
  await choose(page,1);await seek(page,10);assert.match(await film(page).textContent(),/第 4 轮[\s\S]*C/);
  await choose(page,4);await seek(page,2.5);assert.equal(await page.locator('.mechanism-film svg').getAttribute('data-dispatched'),'0');await seek(page,3);assert.equal(await page.locator('.mechanism-film svg').getAttribute('data-dispatched'),'1');await seek(page,12);assert.equal(await page.locator('.mechanism-film svg').getAttribute('data-dispatched'),'3');
  await choose(page,5);await seek(page,1.5);assert.equal(await page.locator('.mechanism-film svg').getAttribute('data-processed'),'0');await seek(page,16);assert.equal(await page.locator('.mechanism-film svg').getAttribute('data-processed'),'8');assert.equal(await page.locator('.mechanism-film svg').getAttribute('data-decode-outputs'),'8');

  await page.goto(base+'foundations/hardware.html');await page.locator('.mechanism-film svg').waitFor();
  for(const v of [0,1,2]){await choose(page,v);await seek(page,3);const d=await page.evaluate(v=>window.TeachingLessons['f-hardware'][v].data,v),bound=Math.max(d.work[0]/d.rates[0],d.work[1]/d.rates[1]);assert.equal(d.bound,bound);assert.match(await film(page).textContent(),new RegExp('整体至少 '+bound+' 秒'));}
  await choose(page,3);await seek(page,3);assert.match(await film(page).textContent(),/8/);assert.match(await film(page).textContent(),/18/);
  for(const v of [4,5]){await choose(page,v);await seek(page,3);assert.match(await film(page).textContent(),/完成 · 9/);await seek(page,4);assert.equal(await page.locator('.mechanism-film svg').getAttribute('data-graph-round'),'2');assert.match(await film(page).textContent(),/第 2 轮 · 输入 4/);await seek(page,5);assert.match(await film(page).textContent(),/完成 · 13/);}

  await page.goto(base+'foundations/speculation.html');await page.locator('.mechanism-film svg').waitFor();
  for(const v of [0,1,2]){const d=await page.evaluate(v=>window.TeachingLessons['f-speculation'][v].data,v);assert.equal(d.output.length,d.accepted+1);assert.equal(d.validKV.length,d.accepted+1);await choose(page,v);await seek(page,3);assert.equal(await page.locator('.mechanism-film svg').getAttribute('data-compared-prefix'),d.draft.slice(0,d.accepted).join(''));assert.equal(await visible(page),0);await seek(page,4);assert.equal(await visible(page),d.output.length);assert.equal(await page.locator('.mechanism-film [data-motion-id^="valid-history"]').count(),d.validKV.length);assert.match(await film(page).textContent(),new RegExp('输出 '+d.output.length+' = 接受 '+d.accepted+' \\+ 补出 1'));}
  await choose(page,0);await page.locator('[data-motion-action=play]').click();await page.waitForTimeout(260);await page.locator('[data-motion-action=play]').click();const paused=await film(page).getAttribute('data-position');await page.waitForTimeout(180);assert.equal(await film(page).getAttribute('data-position'),paused);assert.ok(Number(paused)>0);
  await page.locator('[data-motion-action=reset]').click();assert.equal(await film(page).getAttribute('data-position'),'0.0000');await seek(page,4);assert.equal(await visible(page),3);await choose(page,1);assert.equal(await film(page).getAttribute('data-position'),'0.0000');
  assert.deepEqual(errors,[]);
  console.log('PASS: 9 batching + 6 hardware + 3 speculation scenes at 320/390/768/1440/1994, light/dark, half-step transitions and final states; numeric identities, committed-output/valid-history counts, pause/reset.');
 }finally{
  await browser.close();await server.close();let alive=true;try{process.kill(owned.pid,0);}catch(e){if(e.code==='ESRCH')alive=false;else throw e;}if(alive)throw Error('owned browser still running');console.log('CLEANUP: owned browser gone');
 }
})().catch(e=>{console.error(e);process.exit(1);});
