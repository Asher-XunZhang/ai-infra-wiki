(() => {
  'use strict';
  const catalog=window.PP_SCENARIOS;
  const picker=document.querySelector('#scenario-select');
  if(!picker||!catalog)return;
  const frame=document.querySelector('#timeline iframe');
  const base=new URL('.',document.currentScript.src);
  const cache=new Map();
  const extent=Math.ceil(Math.max(...catalog.map(s=>s.summary.end))/10)*10;
  let active='baseline',packet=null,request=0,frameReady=false;
  const scale=document.querySelector('#scenario-scale');
  const status=document.querySelector('#scenario-status');
  const params=new URLSearchParams(location.search);
  function sendToFrame(){
    if(!frameReady)return;
    frame.contentWindow.postMessage({type:'pp-timing-scale',end:scale.checked?extent:0},'*');
    if(packet)frame.contentWindow.postMessage({type:'pp-timing-model',id:active,model:packet.model},'*');
  }
  function links(){
    document.querySelectorAll('a[href]').forEach(a=>{
      const url=new URL(a.href,location.href);
      if(url.origin===location.origin&&/\/pd-prefill-pp-loop\/(?:quick\.html|index\.html)?$/.test(url.pathname)){
        url.searchParams.set('scenario',active);
        scale.checked?url.searchParams.set('scale','all'):url.searchParams.delete('scale');
        a.href=url.href;
      }
    });
  }
  function persist(){
    const url=new URL(location.href);url.searchParams.set('scenario',active);
    scale.checked?url.searchParams.set('scale','all'):url.searchParams.delete('scale');
    history.replaceState(null,'',url);links();
  }
  function summary(scene){
    document.querySelector('#scenario-change').textContent=scene.change;
    document.querySelector('#scenario-observe').textContent='观察点：'+scene.observe;
    const metrics=document.querySelector('#scenario-metrics');metrics.replaceChildren();
    const values=[['GPU 全部完成',scene.summary.gpuEnd,'u'],['全部本地清理完成',scene.summary.releaseEnd,'u'],['每级展示的 loop',scene.summary.loops,'轮']];
    values.forEach(([name,value,unit],i)=>{
      const item=document.createElement('div'),label=document.createElement('span'),number=document.createElement('strong'),delta=document.createElement('small');
      label.textContent=name;number.textContent=value+' '+unit;
      const original=[catalog[0].summary.gpuEnd,catalog[0].summary.releaseEnd,catalog[0].summary.loops][i];
      const difference=Number((value-original).toFixed(2));
      delta.textContent=scene.id==='baseline'?'原示例参考值':'较原示例 '+(difference>0?'+':'')+difference+' '+unit;
      item.append(label,number,delta);metrics.append(item);
    });
    const rows=document.querySelector('#scenario-parameters');rows.replaceChildren();
    scene.summary.gpu.forEach((values,r)=>{
      const row=document.createElement('tr'),heading=document.createElement('th');heading.scope='row';heading.textContent='PP'+r;row.append(heading);
      values.forEach(value=>{const cell=document.createElement('td');cell.textContent=value;row.append(cell);});rows.append(row);
    });
    document.body.dataset.scenario=scene.id;
    document.querySelectorAll('[data-original-scene-note]').forEach(e=>e.hidden=scene.id==='baseline');
  }
  async function choose(id){
    const scene=catalog.find(s=>s.id===id)||catalog[0];
    const token=++request;status.textContent='正在切换场景…';picker.setAttribute('aria-busy','true');
    try{
      if(!cache.has(scene.id)){
        const pending=fetch(new URL(scene.file,base)).then(response=>{if(!response.ok)throw new Error('场景资源加载失败');return response.json();});
        cache.set(scene.id,pending);pending.catch(()=>{if(cache.get(scene.id)===pending)cache.delete(scene.id);});
      }
      const next=await cache.get(scene.id);
      if(token!==request)return;
      packet=next;active=scene.id;picker.value=active;
      window.dispatchEvent(new CustomEvent('pp-timing-change',{detail:{id:active,packet}}));
      summary(scene);sendToFrame();persist();
      status.textContent='已切换：'+scene.name+'。两张图使用相同的时间模型。';
    }catch(error){if(token===request){picker.value=active;status.textContent='切换未完成，仍显示原场景。请重试或重新加载页面。';}}
    finally{if(token===request)picker.removeAttribute('aria-busy');}
  }
  catalog.forEach(scene=>{const option=document.createElement('option');option.value=scene.id;option.textContent=scene.name;picker.append(option);});
  picker.addEventListener('change',()=>choose(picker.value));
  scale.checked=params.get('scale')==='all';
  document.querySelector('#scenario-scale-label').textContent='切换场景时统一全程时间轴（0–'+extent+' u），便于比较条块长度';
  function applyScale(){
    window.dispatchEvent(new CustomEvent('pp-timing-scale',{detail:{end:scale.checked?extent:0}}));
    sendToFrame();persist();
  }
  scale.addEventListener('change',applyScale);
  if(frame){
    window.addEventListener('message',event=>{
      if(event.source!==frame.contentWindow)return;
      if(event.data?.type==='pp-timing-ready'){frameReady=true;sendToFrame();}
    });
    frame.addEventListener('load',()=>{frameReady=true;sendToFrame();});
    frame.contentWindow?.postMessage({type:'pp-timing-ping'},'*');
  }
  summary(catalog[0]);
  const initial=params.get('scenario');
  applyScale();
  if(initial&&initial!=='baseline')choose(initial);
})();
