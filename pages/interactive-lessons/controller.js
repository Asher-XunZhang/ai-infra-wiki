/* One clock per independent experiment; every view is a projection of it. */
(function(){
'use strict';
const M=window.PriorityLessonModels,V=window.PriorityLessonViews;
if(!M||!V||!window.LearningDepth)return;
for(const host of document.querySelectorAll('[data-priority-lab]')){
 if(host.dataset.ready==='true')continue;
 const kind=host.dataset.priorityLab,key=host.dataset.priorityKey||kind,config=JSON.parse(host.dataset.priorityConfig),$=s=>host.querySelector(s);
 let options={},data,t=0,running=false,raf=0,last=null,speed=1,lastSaved=-Infinity,depth;
 const htmlCache=new WeakMap(),put=(el,html)=>{if(htmlCache.get(el)!==html){el.innerHTML=html;htmlCache.set(el,html);}};
 const controls=[...host.querySelectorAll('[data-param]')],query=new URL(location.href).searchParams;
 for(const el of controls){const value=query.get('lab-'+key+'-'+el.dataset.param);if(value!==null&&[...el.options].some(o=>o.value===value))el.value=value;}
 const read=()=>Object.fromEntries(controls.map(el=>[el.dataset.param,el.dataset.type==='number'?Number(el.value):el.value]));
 const duration=()=>data.duration??(kind==='sampling'?7:5);
 const reduced=()=>matchMedia('(prefers-reduced-motion:reduce)').matches;
 function save(force=false){
  if(running&&!force&&Math.abs(t-lastSaved)<1)return;lastSaved=t;
  const u=new URL(location.href);u.searchParams.set('lab-'+key+'-t',String(Number(t.toFixed(3))));
  for(const [k,v] of Object.entries(options))u.searchParams.set('lab-'+key+'-'+k,String(v));
  u.searchParams.set('lab-'+key+'-speed',String(speed));history.replaceState(history.state,'',u);
 }
 function pause(){running=false;cancelAnimationFrame(raf);last=null;}
 function render(){
  const view=V[kind](data,t,{reduced:reduced()}),source=config.sources[Math.min(view.source||0,config.sources.length-1)];
  host.dataset.time=String(t);host.dataset.playing=String(running);
  $('[data-priority-action=play]').textContent=running?'暂停':'播放';$('[data-priority-action=play]').setAttribute('aria-pressed',String(running));
  $('[data-priority-seek]').max=String(duration());$('[data-priority-seek]').value=String(t);
  $('[data-priority-clock]').textContent=V.f(t)+' / '+V.f(duration())+(config.unit==='step'?' 段':' u');
  if(kind!=='stopping')put($('.plab-figure'),view.detail);put($('.plab-state'),'<h3>'+V.esc(view.title)+'</h3><p>'+V.esc(view.description)+'</p>');
  $('.plab-state').setAttribute('aria-live',running?'off':'polite');
  depth.update({time:t,body:view.overview,event:view.title,explanation:view.description,rows:view.rows,href:source?.url||'#evidence',sourceLabel:source?.label||'来源与边界'});
  save();
 }
 function rebuild(restore=false){
  pause();options=read();data=M[kind](options);t=0;
  if(restore){const n=Number(query.get('lab-'+key+'-t'));if(Number.isFinite(n))t=M.clamp(n,0,duration());const s=Number(query.get('lab-'+key+'-speed'));if([.5,1,1.5,2].includes(s))speed=s;}
  $('[data-priority-speed]').value=String(speed);$('[data-priority-speed-label]').textContent=speed+'×';render();
 }
 function frame(now){
  if(!running)return;
  if(last!==null)t=Math.min(duration(),t+(now-last)/1000*speed*.8);last=now;
  if(t>=duration())pause();render();if(running)raf=requestAnimationFrame(frame);
 }
 const boundaries=()=>[...new Set([0,duration(),...(data.events?data.events.flatMap(e=>[e.start,e.end]):Array.from({length:duration()+1},(_,i)=>i))])].sort((a,b)=>a-b);
 host.addEventListener('click',e=>{
  const action=e.target.closest('[data-priority-action]')?.dataset.priorityAction;if(!action)return;
  if(action==='play'){if(running)pause();else{if(t>=duration())t=0;running=true;last=null;raf=requestAnimationFrame(frame);}}
  else{pause();if(action==='reset')t=0;else{const a=boundaries();t=action==='next'?a.find(n=>n>t+1e-7)??duration():a.filter(n=>n<t-1e-7).at(-1)??0;}}
  render();
 });
 host.addEventListener('change',e=>{if(e.target.matches('[data-param]'))rebuild();});
 $('[data-priority-seek]').addEventListener('input',e=>{pause();t=M.clamp(Number(e.target.value),0,duration());render();});
 $('[data-priority-speed]').addEventListener('change',e=>{speed=Number(e.target.value);$('[data-priority-speed-label]').textContent=speed+'×';save(true);});
 document.addEventListener('visibilitychange',()=>{if(document.hidden){pause();render();}});window.addEventListener('pagehide',pause);
 depth=kind==='stopping'?{update:view=>put($('.plab-figure'),view.body)}:window.LearningDepth.mount({host,before:'.plab-figure',module:['.plab-figure'],persist:()=>save(true),onChange:()=>{if(data)render();}});
 rebuild(true);host.dataset.ready='true';
}
})();
