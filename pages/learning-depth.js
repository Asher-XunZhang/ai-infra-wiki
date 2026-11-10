/* Reading depth is a projection of an existing player, never another clock. */
(()=>{'use strict';
 const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const reduced=()=>matchMedia('(prefers-reduced-motion: reduce)').matches;
 const text=(x,y,value,cls='',anchor='start')=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(value)}</text>`;
 const box=(x,y,w,h,cls='')=>`<rect x="${x}" y="${y}" width="${Math.max(0,w)}" height="${h}" rx="7" class="ld-box ${cls}"/>`;
 const chips=(x,y,items,{width=46,gap=8,cls='',prefix=''}={})=>items.map((v,i)=>`<g data-value="${esc(v)}">${box(x+i*(width+gap),y,width,29,cls)}${text(x+i*(width+gap)+width/2,y+20,prefix+v,'ld-value','middle')}</g>`).join('');
 const path=points=>'M'+points.map(p=>p.join(' ')).join('L');
 function wire(points,{active=false,time=0,start=0,label='',kind='data',id=''}={}){
  const d=path(points),offset=reduced()?0:-(time-start)*22/.9;
  return `<g data-depth-flow="${esc(id)}" data-flow-kind="${kind}" data-active="${active}"><title>${esc(label)}</title><path d="${d}" class="ld-wire ${kind} ${active?'active':''}" marker-end="url(#ld-arrow)"${active?` stroke-dashoffset="${offset.toFixed(3)}"`:''}/></g>`;
 }
 function pulse(markup,time,active=true){return markup.replace(/<path /g,`<path data-flow-active="${active}" ${active?`style="stroke-dasharray:4 7;stroke-dashoffset:${reduced()?0:-time*22/.9}"`:''} `);}
 function svg(body,height=340,label='同一时刻的机制总览'){
  return `<div class="ld-viewport" tabindex="0" aria-label="机制总览，窄屏可横向查看"><svg viewBox="0 0 1100 ${height}" role="img"><title>${esc(label)}</title><defs><marker id="ld-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="context-stroke"/></marker></defs>${body}</svg></div>`;
 }
 function facts(rows){return `<dl class="ld-facts">${rows.filter(r=>r[1]!==undefined).map(([k,v])=>`<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('')}</dl>`;}
 function tracks(events,lanes,time,duration,{label='同一时钟 · 关键活动',subLabel='条块保留原事件区间；空白不是阻塞等待'}={}){
  const x=t=>185+Math.max(0,t)/Math.max(1,duration)*885;
  let out=text(16,22,label,'ld-heading')+text(16,43,subLabel,'ld-small');
  lanes.forEach((lane,i)=>{const y=74+i*39;out+=text(16,y+18,lane.label,'ld-small')+box(185,y,885,25,'ld-lane');
   for(const e of events.filter(lane.match)){const a=x(e.start),w=Math.max(1,x(e.end)-a),p=Math.max(0,Math.min(1,(time-e.start)/(e.end-e.start||1))),kind=e.kind||e.type||'control';out+=`<g data-depth-event="${esc(e.id)}"><title>${esc(e.label||e.title)} · ${e.start}–${e.end}</title>${box(a,y,w,25,'ld-event '+kind)}<rect x="${a}" y="${y+20}" width="${w*p}" height="4" class="ld-progress"/>${w>75?text(a+w/2,y+17,(e.short||e.label||e.title||'').slice(0,9),'ld-small','middle'):''}</g>`;}
  });
  for(let i=0;i<=4;i++)out+=text(x(duration*i/4),65,(duration*i/4).toFixed(1),'ld-small',i===4?'end':'start');
  out+=`<path d="M${x(time)} 68V${76+lanes.length*39}" class="ld-cursor"/>`;
  return svg(out,88+lanes.length*39,label);
 }
 function mount({host,before,module=[],source=[],compact=false,compactOverview=false,onChange=()=>{},persist=()=>{}}){
  if(!host)return null;
  const levels=compact?['modules','source']:['overview','modules','source'];
  let allowed=levels.slice(),view=levels.includes(new URL(location.href).searchParams.get('depth'))?new URL(location.href).searchParams.get('depth'):levels[0];
  const bar=document.createElement('div');bar.className='ld-controls';bar.setAttribute('role','group');bar.setAttribute('aria-label','细节程度');
  const labels=compact?{modules:'过程',source:'细节与源码'}:{overview:'总览',modules:'模块过程',source:'源码细节'};
  bar.innerHTML='<span>细节程度</span>'+levels.map(v=>`<button type="button" data-depth-choice="${v}">${labels[v]}</button>`).join('')+'<small data-depth-hint></small>';
  const overview=document.createElement('div');overview.className='ld-overview';
  const detail=document.createElement('section');detail.className='ld-source-panel';detail.setAttribute('aria-label','当前事件的源码与状态');
  const anchor=typeof before==='string'?host.querySelector(before):before;
  (anchor||host.firstChild)?.before(bar);if(!bar.isConnected)host.prepend(bar);
  bar.after(overview);host.append(detail);host.dataset.learningDepth='true';host.dataset.compactOverview=String(compactOverview);
  const modules=module.map(s=>typeof s==='string'?host.querySelector(s):s).filter(Boolean),sources=source.map(s=>typeof s==='string'?host.querySelector(s):s).filter(Boolean);
  modules.forEach(e=>e.classList.add('ld-module-region'));sources.forEach(e=>e.classList.add('ld-source-region'));
  function apply(){host.dataset.depth=view;overview.hidden=view!=='overview'&&!(compactOverview&&view==='modules');detail.hidden=view!=='source';
   for(const b of bar.querySelectorAll('button')){b.hidden=!allowed.includes(b.dataset.depthChoice);b.setAttribute('aria-pressed',String(b.dataset.depthChoice===view));}
   bar.querySelector('small').textContent=view==='overview'?'先看关键变化；展开保留当前时刻':view==='modules'?'处理过程与必要状态':'当前字段、条件与固定源码';
   sources.forEach(e=>{if(e.tagName==='DETAILS')e.open=view==='source';});
  }
  function set(next,remember=true){if(!allowed.includes(next))next=allowed[0];view=next;apply();onChange(view);if(remember){persist();const u=new URL(location.href);u.searchParams.set('depth',view);history.replaceState(history.state,'',u);}}
  bar.addEventListener('click',e=>{const b=e.target.closest('[data-depth-choice]');if(b)set(b.dataset.depthChoice);});
  window.addEventListener('popstate',()=>set(new URL(location.href).searchParams.get('depth'),false));
  apply();
  return {get view(){return view;},set,bar,overview,detail,
   allow(next){const list=next.filter(x=>levels.includes(x));if(list.join()===allowed.join())return;allowed=list.length?list:levels;if(!allowed.includes(view))view=allowed[0];apply();},
   update({time,body='',rows=[],event='',explanation='',href='',sourceLabel='固定源码 ↗'}){
    overview.dataset.time=String(time);detail.dataset.time=String(time);
    if(!overview.hidden)overview.innerHTML=body;
    if(view==='source')detail.innerHTML=`<p class="ld-state-note">下列为当前教学状态快照；真实字段、分支与条件以本阶段源码说明为准。</p><h3>${esc(event||'当前源码状态')}</h3>${explanation?`<p>${esc(explanation)}</p>`:''}${facts(rows)}${href?`<a href="${esc(href)}" target="_blank" rel="noreferrer">${esc(sourceLabel)}</a>`:''}`;
   }
  };
 }
 window.LearningDepth={mount,pulse,esc,text,box,chips,wire,svg,facts,tracks,reduced};
})();
