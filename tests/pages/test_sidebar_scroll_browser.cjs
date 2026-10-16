const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');

(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_EXECUTABLE?{executablePath:process.env.CHROMIUM_EXECUTABLE}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:800}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const base=process.env.OVERVIEW_TEST_URL||'http://127.0.0.1:8765/';
  await page.goto(base+'sglang/inference-overview/journey.html');
  await page.locator('.topic-group,.module-nav').evaluateAll(groups=>groups.forEach(group=>group.open=true));
  const sidebar=page.locator('.site-sidebar');
  const target=page.locator('[data-module=case-studies] .lesson-link').last();
  await target.scrollIntoViewIfNeeded();
  const before=await sidebar.evaluate(el=>el.scrollTop);
  assert.ok(before>500,'the test must navigate from well below the sidebar top');
  await target.click();await page.waitForURL('**/notes.html*');
  const expectPosition=async(label,top=before)=>{
   await page.waitForFunction(expected=>Math.abs(document.querySelector('.site-sidebar').scrollTop-expected)<2,top);
   assert.ok(Math.abs(await sidebar.evaluate(el=>el.scrollTop)-top)<2,label);
  };
  await expectPosition('chapter navigation retains the sidebar position');
  assert.equal(await page.locator('.site-sidebar [aria-current=page]').innerText(),'PP loop 源码阅读');
  assert.equal(await page.evaluate(()=>window.scrollY),0,'new chapter still starts at the top of the document');
  assert.equal(await page.locator('.topic-group[open]').count(),5);
  assert.equal(await page.locator('.module-nav[open]').count(),12);
  await page.reload();await expectPosition('reload retains the sidebar position');
  const next=page.locator('[data-module=case-studies] .lesson-link').nth(1);
  await next.click();await page.waitForURL('**/pd-prefill-pp-loop/index.html*');
  await expectPosition('a second chapter switch retains position');
  await page.goBack();await expectPosition('back retains position');
  await page.goForward();await expectPosition('forward retains position');

  // Scrolling without a link click must also survive a reload.
  await sidebar.evaluate(el=>el.scrollTop=400);
  await page.reload();await expectPosition('manual scrolling survives reload',400);
  // The narrow layout must not overwrite the saved desktop offset with zero.
  await page.setViewportSize({width:390,height:800});
  await page.waitForFunction(()=>!document.querySelector('.sidebar-menu').open);
  await page.locator('.sidebar-menu>summary').click();
  await page.locator('[data-module=case-studies] .lesson-link').last().click();
  await page.waitForURL('**/notes.html*');
  assert.equal(await page.locator('.sidebar-menu').evaluate(el=>el.open),false);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.setViewportSize({width:1440,height:800});
  await expectPosition('returning to desktop restores its position',400);

  // Storage restrictions must not break navigation or the remaining site script.
  const blocked=await browser.newPage({viewport:{width:1440,height:800}});
  blocked.on('pageerror',e=>errors.push(e.message));
  await blocked.addInitScript(()=>{
   Storage.prototype.getItem=Storage.prototype.setItem=()=>{throw new DOMException('blocked','SecurityError');};
  });
  await blocked.goto(base+'sglang/pd-prefill-pp-loop/notes.html');
  assert.equal(await blocked.locator('[data-module=case-studies]').evaluate(el=>el.open),true);
  await blocked.locator('.theme-toggle').click();
  assert.equal(await blocked.locator('html').getAttribute('data-theme'),'light');
  await blocked.locator('[data-module=case-studies] .lesson-link').first().click();
  await blocked.waitForURL('**/quick.html*');
  assert.deepEqual(errors,[]);
  console.log('PASS: sidebar position across chapter switches, reload, back/forward, mobile layout and blocked storage.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
