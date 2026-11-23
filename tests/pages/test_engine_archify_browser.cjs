'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=(process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765').replace(/\/$/,'');
const shots=process.env.ENGINE_TEST_SCREENSHOTS;
const cases=[
 ['roles','?view=roles#process'],['tp1','?view=ownership&tp=1&step=2#process'],['tp2','?view=ownership&tp=2&step=2#process'],
 ['topology','?wire=topology&step=0#wire'],['request','?wire=sequence&step=0#wire'],['execute','?wire=sequence&step=2#wire'],['management','?wire=sequence&step=3#wire'],
 ['admission','?step=1#batch'],['lifecycle','?step=0#crosscut'],['graph','?step=3#crosscut'],
 ['pd-overview','?depth=overview&step=0#disagg'],['pd-instances','?depth=modules&step=0#disagg'],['kv-flow','?depth=modules&step=1#disagg'],['restore','?step=2#disagg']
];
(async()=>{let server,browser,identity;try{
 server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});const proc=server.process();
 identity={session:'engine-archify',controller:process.pid,browser:proc.pid,profile:proc.spawnargs.find(a=>a.startsWith('--user-data-dir=')).slice(16)};console.log(JSON.stringify(identity));
 browser=await chromium.connect(server.wsEndpoint());const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],issues=[],seen=new Set();page.on('pageerror',e=>errors.push(e.message));
 for(const lang of ['zh','en'])for(const [name,query] of cases){
  await page.goto(base+'/engine-comparison/'+query,{waitUntil:'domcontentloaded'});await page.locator('.ac-native').first().waitFor();
  await page.evaluate(lang=>atlasLocale.setLang(lang),lang);
  const check=await page.locator('[data-archify]').evaluateAll((es,english)=>es.map(el=>{
   const svg=el.querySelector('svg'),vb=svg.viewBox.baseVal,problems=[];
   for(const node of svg.querySelectorAll('[data-node-id]')){
    const r=node.querySelector('rect.c-mask');if(!r)continue;const b=r.getBBox();
    for(const t of node.querySelectorAll('text')){const x=t.getBBox();if(x.x<b.x+1||x.x+x.width>b.x+b.width-1||x.y<b.y||x.y+x.height>b.y+b.height)problems.push('caption outside node: '+t.textContent);}
   }
   for(const t of svg.querySelectorAll('text')){
    const b=t.getBBox();if(b.x<vb.x-1||b.y<vb.y-1||b.x+b.width>vb.x+vb.width+1||b.y+b.height>vb.y+vb.height+1)problems.push('clipped: '+t.textContent);
    if(english&&/[\u4e00-\u9fff]/.test(t.textContent))problems.push('untranslated: '+t.textContent);
   }
   const ids=[...svg.querySelectorAll('[id]')].map(e=>e.id);if(new Set(ids).size!==ids.length)problems.push('duplicate IDs');
   return {asset:el.dataset.archify,problems};
  }),lang==='en');
  for(const item of check){seen.add(item.asset);for(const problem of item.problems)issues.push(`${lang}/${name}/${item.asset}: ${problem}`);}
  for(const width of [1440,390]){
   await page.setViewportSize({width,height:1000});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`${name}/${lang}/${width}: page overflow`);
   if(shots&&lang==='zh'&&['roles','topology','kv-flow','graph'].includes(name)){
    fs.mkdirSync(shots,{recursive:true});await page.locator('.ia-boardGrid').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(shots,`${name}-${width}.png`)});
   }
  }
  await page.setViewportSize({width:1440,height:1000});
 }
 assert.equal(seen.size,27,'all native diagrams rendered in the course');assert.deepEqual(issues,[]);assert.deepEqual(errors,[]);
 console.log('PASS: 27 native Archify diagrams, bilingual node containment and SVG bounds, stable identifiers, desktop1440/mobile390 without page overflow.');
}finally{
 await browser?.close().catch(()=>{});await server?.close().catch(()=>{});
 if(identity){const remains=cp.execFileSync('ps',['-axo','pid=,command='],{encoding:'utf8'}).split('\n').filter(s=>s.includes(identity.profile));assert.deepEqual(remains,[]);assert(!fs.existsSync(identity.profile));console.log(JSON.stringify({cleanup:{...identity,remaining:0}}));}
}})().catch(e=>{console.error(e);process.exitCode=1});
