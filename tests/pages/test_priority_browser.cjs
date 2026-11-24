const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), os = require('node:os');
const cp = require('node:child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const lessons = require('../../pages/interactive-lessons/lessons.json');
const models = require('../../pages/interactive-lessons/models.js');
const base = process.env.OVERVIEW_TEST_URL || 'http://127.0.0.1:8765/';
const output = process.env.PRIORITY_QA_DIR;
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
function processes() {
  return cp.execFileSync('ps', ['-axo', 'pid=,ppid=,stat=,command='], {encoding:'utf8'}).trim().split('\n')
    .map(line => { const m=line.trim().match(/^(\d+)\s+(\d+)\s+(\S+)\s+(.*)$/); return m && {pid:+m[1],parent:+m[2],stat:m[3],command:m[4]}; }).filter(Boolean);
}
function descendants(pid) {
  const all=processes(), ids=new Set([pid]); let changed=true;
  while(changed){changed=false;for(const p of all)if(ids.has(p.parent)&&!ids.has(p.pid)){ids.add(p.pid);changed=true;}}
  return all.filter(p=>ids.has(p.pid));
}
function alive(targets) {const now=processes();return targets.filter(p=>now.some(q=>p.pid===q.pid&&p.command===q.command&&!q.stat.startsWith('Z')));}
(async()=>{
  const server=await chromium.launchServer({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
  const owned=server.process(), info={session:'priority-lessons-'+process.pid,controller:process.pid,browser:owned.pid,profile:owned.spawnargs.find(a=>a.startsWith('--user-data-dir='))};
  console.log(JSON.stringify(info));
  if(output){fs.mkdirSync(output,{recursive:true});fs.writeFileSync(path.join(output,'session.json'),JSON.stringify(info,null,2));}
  let browser;const seen=new Map(),remember=()=>{for(const p of descendants(owned.pid))seen.set(p.pid,p);};remember();
  try{
    browser=await chromium.connect(server.wsEndpoint());
    const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));page.setDefaultTimeout(10000);
    const route=key=>lessons[key].path||(key==='quantization-intuition'?'foundations/hardware.html':'foundations/generation.html');
    const host=key=>page.locator('[data-priority-key="'+key+'"]');
    const go=async key=>{await page.goto(base+route(key));await host(key).locator('[data-priority-clock]').waitFor();await page.waitForFunction(k=>document.querySelector('[data-priority-key="'+k+'"]').dataset.ready==='true',key);remember();};
    const seek=async(key,t)=>host(key).locator('[data-priority-seek]').evaluate((e,t)=>{e.value=t;e.dispatchEvent(new Event('input',{bubbles:true}));},String(t));
    const time=async key=>+await host(key).getAttribute('data-time');
    const flows=async key=>host(key).locator('[data-priority-flow][data-active=true]').evaluateAll(es=>es.map(e=>({id:e.dataset.priorityFlow,path:e.querySelector('path').getAttribute('d'),phase:e.querySelector('path').getAttribute('stroke-dashoffset')})));
    const shot=async(key,suffix,selector='.plab-figure')=>{if(output)await host(key).locator(selector).screenshot({path:path.join(output,key+'-'+suffix+'.png')});};
    for(const [key,lesson] of Object.entries(lessons)){
      await go(key);const h=host(key),layered=key!=='stopping';
      assert.equal(await h.getAttribute('data-playing'),'false',key+' starts paused');
      assert.equal(await h.locator('[data-priority-speed]').count(),1,key+' one speed control');
      assert.equal(await h.locator('[data-priority-action=play]').count(),1,key+' one play control');
      await seek(key,1.375);const before=await time(key);
      if(layered){
        await h.locator('[data-depth-choice=modules]').click();assert.equal(await time(key),before);
        await h.locator('[data-source-toggle]').click();assert.equal(await time(key),before);
        assert.equal(+await h.locator('.ld-source-panel').getAttribute('data-time'),before);
        await h.locator('[data-depth-choice=overview]').click();assert.equal(await time(key),before);
      }
      await page.reload();await page.waitForFunction(k=>document.querySelector('[data-priority-key="'+k+'"]').dataset.ready==='true',key);
      assert.equal(await time(key),before,key+' URL restores progress');
      if(layered)assert.equal(await h.locator('[data-source-toggle]').getAttribute('aria-expanded'),'true');
      await h.locator('[data-priority-action=play]').click();await delay(110);
      assert.equal(await h.getAttribute('data-playing'),'true');assert.ok(await time(key)>before);
      if(layered){await h.locator('[data-depth-choice=modules]').click();assert.equal(await h.getAttribute('data-playing'),'true',key+' view change keeps playback');}
      await h.locator('[data-priority-speed]').selectOption('2');assert.equal(await h.getAttribute('data-playing'),'true');
      await h.locator('[data-priority-action=play]').click();const frozen=await time(key);await delay(90);assert.equal(await time(key),frozen);
      await h.locator('[data-priority-action=reset]').click();assert.equal(await time(key),0);
      await h.locator('[data-priority-action=next]').focus();await page.keyboard.press('Enter');assert.ok(await time(key)>0);
      await h.locator('[data-priority-action=prev]').click();assert.equal(await time(key),0);
      // Every selector changes a real model, and changing conditions resets the shared clock.
      for(const control of lesson.controls){
        await seek(key,1.5);const other=control.values.find(([v])=>String(v)!==String(control.value))[0];
        await h.locator('[data-param="'+control.name+'"]').selectOption(String(other));
        assert.equal(await time(key),0);assert.equal(await h.getAttribute('data-playing'),'false');
        await h.locator('[data-param="'+control.name+'"]').selectOption(String(control.value));
      }
      const d=models[lesson.kind](Object.fromEntries(lesson.controls.map(c=>[c.name,c.value])));
      if(layered)await h.locator('[data-depth-choice=modules]').click();
      const pulseEvent=(d.events||[]).find(e=>e.kind==='transfer');
      if(pulseEvent && ['moe','multimodal','routing','lora','vllm'].includes(key)){
        const t=(pulseEvent.start+pulseEvent.end)/2;await seek(key,t);const first=await flows(key);
        assert.ok(first.length,key+' has a transfer pulse');await delay(100);assert.deepEqual(await flows(key),first,key+' pause freezes phase');
        await seek(key,t+.05);assert.notDeepEqual(await flows(key),first,key+' pulse advances within interval');
        await seek(key,t);assert.deepEqual(await flows(key),first,key+' rewind restores phase');
      }
      await seek(key,d.duration??(key==='sampling'?7:5));assert.deepEqual(await flows(key),[],key+' no pulse beyond activity');
      await shot(key,'desktop');
      const outside=await h.locator('svg').evaluateAll(es=>es.flatMap(svg=>[...svg.querySelectorAll('text')].filter(e=>{const b=e.getBBox(),v=svg.viewBox.baseVal;return b.x<-.5||b.y<-.5||b.x+b.width>v.width+.5||b.y+b.height>v.height+.5;}).map(e=>e.textContent)));
      assert.deepEqual(outside,[],key+' SVG text stays within viewBox');
      console.log(key+': controls, projection, clock, replay and diagram bounds passed');
    }
    await go('sampling');await seek('sampling',5.5);assert.match(await host('sampling').locator('.plab-figure').innerText(),/62.5%/);
    await host('sampling').locator('[data-param=temperature]').selectOption('0');await seek('sampling',5.5);assert.match(await host('sampling').innerText(),/贪心/);
    await go('stopping');await host('stopping').locator('[data-param=scenario]').selectOption('cancel');await seek('stopping',2.5);assert.match(await host('stopping').innerText(),/仍|尚|不能/);
    await go('quantization-intuition');await host('quantization-intuition').locator('[data-param=group]').selectOption('1');assert.match(await host('quantization-intuition').innerText(),/13 bytes/);
    await go('quantization');await host('quantization').locator('[data-param=supported]').selectOption('no');await seek('quantization',4);assert.match(await host('quantization').innerText(),/阻塞/);
    await go('faults');await host('faults').locator('[data-depth-choice=modules]').click();assert.equal(await host('faults').locator('[data-priority-event="oom"]').count(),1);await host('faults').locator('[data-param=evidence]').selectOption('missing');assert.equal(await host('faults').locator('[data-priority-event="oom"]').count(),0);assert.match(await host('faults').innerText(),/缺少 rank0/);
    await go('multimodal');await host('multimodal').locator('[data-depth-choice=modules]').click();assert.equal(await host('multimodal').locator('[data-priority-flow=mm-kv]').count(),1);
    await host('multimodal').locator('[data-param=roles]').selectOption('single');assert.equal(await host('multimodal').locator('[data-priority-flow=mm-kv]').count(),0);await seek('multimodal',6);assert.deepEqual(await flows('multimodal'),[]);
    await go('routing');await host('routing').locator('[data-param=checkpoint]').selectOption('missing');await seek('routing',5);assert.match(await host('routing').innerText(),/恢复条件未齐/);
    await go('vllm');await host('vllm').locator('[data-depth-choice=modules]').click();await seek('vllm',99);assert.match(await host('vllm').innerText(),/7/);assert.match(await host('vllm').innerText(),/8/);
    assert.equal(await page.locator('[data-chapter=engine-comparison] .engine-deep-reading [aria-current=page]').count(),1,'V1 deep read belongs to chapter 2');
    assert.equal(await page.locator('.engine-topic-nav .lesson-link').count(),7,'deep read does not add a comparison topic');
    assert.equal(await page.locator('[data-chapter=sglang] .lesson-link').count(),34);
    await seek('vllm',2);
    const fact=label=>host('vllm').locator('.ld-overview .plab-facts>div').filter({has:page.locator('dt',{hasText:label})}).locator('dd');
    await host('vllm').locator('[data-depth-choice=overview]').click();
    assert.equal(await fact('本请求已分配 block').innerText(),'2');assert.equal(await fact('计算记录覆盖 block').innerText(),'1');
    await seek('vllm',4);assert.equal(await fact('设备已选输出 ID').innerText(),'1 / 目标 2');assert.equal(await fact('前端已收输出 ID').innerText(),'0 / 目标 2');
    for(const width of [1440,768,390,320]){
      await page.setViewportSize({width,height:1000});
      for(const key of Object.keys(lessons)){
        await go(key);if(key!=='stopping')await host(key).locator('[data-depth-choice=modules]').click();
        await seek(key,2.5);
        for(const theme of ['light','dark']){
          await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
          assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,key+' '+width+' '+theme+' page overflow');
          assert.equal(await host(key).locator('select').evaluateAll(es=>es.some(e=>e.getBoundingClientRect().width>innerWidth)),false);
        }
        if(width===390){await shot(key,'mobile','.plab-controls');if(output&&['moe','multimodal','vllm'].includes(key)){await host(key).locator('.plab-transport').evaluate(e=>scrollTo({top:e.getBoundingClientRect().top+scrollY-95,behavior:'instant'}));await page.screenshot({path:path.join(output,key+'-mobile-context.png')});}}
      }
    }
    await page.emulateMedia({reducedMotion:'reduce'});await go('moe');await host('moe').locator('[data-depth-choice=modules]').click();await seek('moe',2);
    assert.equal(await host('moe').locator('[data-active=true] path[stroke-dashoffset]').count(),0,'reduced motion uses static activity');
    const context=await browser.newContext({javaScriptEnabled:false,viewport:{width:390,height:844}}),plain=await context.newPage();
    for(const key of Object.keys(lessons)){await plain.goto(base+route(key));assert.ok(await plain.locator('#'+(lessons[key].anchor?'evidence-'+key:'evidence')).innerText());assert.ok(await plain.locator('.self-check').count()>=lessons[key].checks.length,key+' no-JS self-checks');}
    await context.close();assert.deepEqual(errors,[]);console.log('Ten experiments: beginner branches, responsive themes, no-JS, reduced motion and clock invariants passed.');
  }finally{
    remember();const targets=[...seen.values()];
    if(browser)await browser.close().catch(()=>{});await server.close().catch(()=>{});
    for(let i=0;i<20&&alive(targets).length;i++)await delay(100);
    for(const p of alive(targets))process.kill(p.pid,'SIGTERM');
    for(let i=0;i<20&&alive(targets).length;i++)await delay(100);
    for(const p of alive(targets))process.kill(p.pid,'SIGKILL');await delay(100);
    const remaining=alive(targets);console.log(JSON.stringify({session:info.session,cleaned:remaining.length===0,remaining}));assert.equal(remaining.length,0);
    if(info.profile){const profile=info.profile.split('=').slice(1).join('=');if(!remaining.length&&profile.startsWith(os.tmpdir())&&path.basename(profile).startsWith('playwright_chromiumdev_profile-'))fs.rmSync(profile,{recursive:true,force:true});assert.equal(fs.existsSync(profile),false);}
  }
})().catch(e=>{console.error(e);process.exitCode=1;});
