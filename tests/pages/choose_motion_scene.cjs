/* Scenario IDs remain URL-stable even when the UI has multiple selection levels. */
async function choose(page,index){
 const film=page.locator('.mechanism-film'),key=await film.getAttribute('data-motion');
 if(key==='f-parallel'){const family=['TP','TP','PP','PP','DP','EP','EP','SP','PCP','PCP','DCP','DCP'][index];await film.locator(`[data-lesson-family="${family}"]`).click();}
 if(key==='f-communication'){const family=await page.evaluate(i=>window.TeachingLessons['f-communication'][i].op,index);await film.locator(`[data-lesson-family="${family}"]`).click();}
 if(key==='f-batching')await film.locator(`[data-lesson-family="${index>=5?'chunk':index>=2?'dynamic':index===1?'fixed':'continuous'}"]`).click();
 if(key==='f-hardware'){const kind=await page.evaluate(i=>window.TeachingLessons['f-hardware'][i].kind,index);await film.locator(`[data-lesson-family="${kind.replace('hardware-','')}"]`).click();}
 if(key==='f-metrics')await film.locator(`[data-lesson-family="${index<2?'latency':index<4?'distribution':'throughput'}"]`).click();
 if(key==='scheduling'){
  await film.locator(`[data-lesson-family="${index>=12?'chunk':'admission'}"]`).click();
  if(index<12){const policy=await page.evaluate(i=>window.TeachingLessons.scheduling[i].policy,index);await film.getByLabel('准入排序策略').selectOption(policy);}
 }
 const select=film.getByLabel('连续演示条件');if(await select.isVisible())await select.selectOption(String(index));
}
choose.count=async page=>page.locator('.mechanism-film').evaluate(e=>(window.TeachingLessons?.[e.dataset.motion]||window.MechanismScenes[e.dataset.motion]).length);
module.exports=choose;
