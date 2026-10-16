/* Assertions follow mechanisms moved out of the former parameter players. */
const assert=require('node:assert/strict');
module.exports=async(page,key)=>{
 const film=page.locator('.mechanism-film'),choice=film.getByLabel('连续演示条件');
 const seek=async p=>film.locator('input[type=range]').evaluate((e,p)=>{e.value=p;e.dispatchEvent(new Event('input',{bubbles:true}));},p);
 const variants=await require('./choose_motion_scene.cjs').count(page);
 for(let v=0;v<variants;v++){
  if(variants>1)await require('./choose_motion_scene.cjs')(page,v);const max=Number(await film.locator('input[type=range]').getAttribute('max'));
  for(let p=0;p<=max;p++){await seek(p);assert.doesNotMatch(await film.textContent(),/undefined|NaN/);if(key==='scheduling'&&v<12){
   const expected=await page.evaluate(({v,p})=>window.TeachingLessons.scheduling[v].events[p].admitted,{v,p});
   assert.deepEqual(await film.locator('[data-motion-id^="plan-"][data-committed="1.000"]').evaluateAll(es=>es.sort((a,b)=>Number(a.dataset.x)-Number(b.dataset.x)).map(e=>e.textContent)),expected);
  }}
  if(key==='model-execution'){assert.match(await film.textContent(),/DeepSeek/);assert.match(await film.locator('.motion-evidence').getAttribute('href'),/github.com.+[0-9a-f]{40}/);}
  if(key==='kv-memory'){const target=[8,9,11][v];assert.equal(await film.locator('[data-slot][data-written="1.000"]').count(),target);assert.match(await film.locator('svg').textContent(),new RegExp(`已写入 ${target}`));}
  if(key==='communication'){assert.equal(await film.locator('[data-written=true]').count(),3);assert.deepEqual(await film.locator('[data-motion-id^="payload-"] text').allTextContents(),['A','B','C']);assert.match(await film.locator('svg').textContent(),new RegExp(`${v?3:2} 个连续块`));}
  if(key==='advanced-generation'){const d=await page.evaluate(v=>window.TeachingLessons['advanced-generation'][v].data,v);assert.deepEqual(await film.locator('[data-motion-id^="committed-output"][opacity="1.000"] text').allTextContents(),d.output);assert.deepEqual(await film.locator('[data-motion-id^="valid-kv"][opacity="1.000"] text').allTextContents(),d.validKV);}
  if(key==='serving-operations'){assert.equal(await film.locator('[data-motion-id="front-state"].teach-empty').count(),1);assert.equal(await film.locator('[data-motion-id="scheduler-req"].teach-empty').count(),1);await seek(2.5);assert.equal(Number(await film.locator('[data-motion-id="abort-msg"]').getAttribute('opacity'))>0,v===0,'timeout is local to Scheduler');}
 }
};
