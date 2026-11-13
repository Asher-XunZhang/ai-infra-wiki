/* Shared observation and teaching clock, including links to the source-reading page. */
(() => {
  'use strict';
  const query=new URLSearchParams(location.search),listeners=new Set(),M=window.PP_JOURNEY_MODEL;
  const number=(key,fallback)=>query.has(key)&&Number.isFinite(+query.get(key))?+query.get(key):fallback;
  let s={batch:M.clamp(Math.round(number('batch',1)),1,5),rank:M.clamp(Math.round(number('rank',0)),0,2),loop:Math.max(0,Math.round(number('loop',M.clamp(Math.round(number('batch',1)),1,5)+2))),mode:query.get('view')==='loop'?'loop':'batch',level:['process','source'].includes(query.get('level'))?'process':'overview',inspect:query.get('inspect')||(query.get('level')==='source'?(query.get('event')||'0:3:gpu'):''),time:Math.max(0,number('t',0)),speed:M.clamp(number('speed',1),.25,4),playing:false,relation:query.get('relation')||'',compare:query.get('compare')||'current',event:query.get('event')||(document.querySelector('[data-pp-reader=dependency]')?'0:5:recv_out':`${M.clamp(Math.round(number('rank',0)),0,2)}:${M.clamp(Math.round(number('batch',1)),1,5)+2}:gpu`)};
  if(/^\d+:\d+:[a-z_]+$/.test(s.event)){const [rank,loop]=s.event.split(':').map(Number);if(!query.has('rank'))s.rank=M.clamp(rank,0,2);if(!query.has('loop'))s.loop=loop;}
  let comparisonEnd=0;
  let model=null,scenario=query.get('scenario')||'baseline',raf=0,last=0,lastSaved=0;
  const initialPlaying=query.get('play')==='1'&&!matchMedia('(prefers-reduced-motion: reduce)').matches;
  const duration=()=>Math.max(model?.end??0,comparisonEnd);
  function write(url){
    for(const [key,value] of Object.entries({batch:s.batch,rank:s.rank,loop:s.loop,view:s.mode,level:s.level,inspect:s.inspect,relation:s.relation,t:s.time.toFixed(3),speed:s.speed,event:s.event,compare:s.compare,play:(s.playing||(!model&&initialPlaying))?'1':'0'}))url.searchParams.set(key,value);
    return url;
  }
  function links(){
    document.querySelectorAll('a[href]').forEach(a=>{
      const u=new URL(a.href,location.href);
      if(u.origin!==location.origin||! /\/pd-prefill-pp-loop\/(?:quick\.html|index\.html|notes\.html)?$/.test(u.pathname)||a.hasAttribute('data-pp-fixed'))return;
      for(const k of ['scenario','timing','seed','scale']){const v=new URL(location.href).searchParams.get(k);v===null?u.searchParams.delete(k):u.searchParams.set(k,v);}
      a.href=write(u).href;
    });
  }
  function persist(){history.replaceState(null,'',write(new URL(location.href)));links();lastSaved=performance.now();}
  function notify(reason){listeners.forEach(fn=>fn(s,reason));window.dispatchEvent(new CustomEvent('pp-observation',{detail:{...s,reason}}));}
  function tick(now){
    if(!s.playing||!model)return;
    const dt=last?Math.min(.15,(now-last)/1000):0;last=now;
    const transfers=s.level==='process'?model.batches.flatMap(b=>M.transfers(model,b.batch)):M.transfers(model,s.batch),inTransfer=transfers.some(e=>M.active(e,s.time));
    const next=transfers.filter(e=>e.start>s.time+.000001).sort((a,b)=>a.start-b.start)[0]?.start??duration();
    s.time=Math.min(duration(),s.time+dt*(inTransfer?.45:3)*s.speed,next);
    if(s.time>=duration())s.playing=false;
    notify('clock');if(now-lastSaved>1000||!s.playing)persist();
    if(s.playing)raf=requestAnimationFrame(tick);
  }
  function update(patch,reason='observation'){
    const was=s.playing;s={...s,...patch};
    if(!('inspect' in patch)&&['rank','loop','batch'].some(k=>k in patch))s.inspect='';
    if(model&&'event' in patch&&model.graph[s.event]){
      const event=model.graph[s.event],batch=String(event.owner).match(/^M([1-5])(?:$|\b|：)/)?.[1];
      s.rank=event.r;s.loop=event.n;
      if(batch&&!('batch' in patch))s.batch=+batch;
    }
    if(model&&!('event' in patch)&&('batch' in patch||'rank' in patch)){const r=model.batches.find(b=>b.batch===s.batch)?.ranks[s.rank];if(r){s.event=`${s.rank}:${r.currentLoop}:gpu`;s.loop=r.currentLoop;}}
    if(model&&!('event' in patch)&&'loop' in patch)s.event=model.graph[`${s.rank}:${s.loop}:gpu`]?`${s.rank}:${s.loop}:gpu`:`${s.rank}:${s.loop}:recv_req`;
    s.time=M.clamp(s.time,0,model?duration():Number.MAX_SAFE_INTEGER);
    if(model){s.loop=Math.min(s.loop,Math.max(...model.loops.filter(l=>l.r===s.rank).map(l=>l.n)));if(!model.graph[s.event])s.event='0:5:recv_out';}
    if(!s.playing){cancelAnimationFrame(raf);last=0;}
    notify(reason);persist();
    if(s.playing&&!was){last=0;raf=requestAnimationFrame(tick);}
  }
  function install(packet,id){
    const first=!model,changed=!first;
    cancelAnimationFrame(raf);last=0;model=M.compile(packet);scenario=id;
    s.playing=first&&initialPlaying;if(first&&(s.compare!=='current'||document.querySelector('[data-pp-reader=dependency]')))comparisonEnd=Math.max(comparisonEnd,s.time);if(!first)comparisonEnd=0;s.time=first?Math.min(s.time,Math.max(duration(),s.compare!=='current'?s.time:0)):0;
    if(changed){s.playing=false;s.inspect='';s.relation='';}
    update(first&&query.has('event')?{event:s.event}:{},'model');
    if(s.playing)raf=requestAnimationFrame(tick);
  }
  const frame=document.querySelector('#timeline iframe');
  let frameFocus='';
  function sendFrame(reason){
    if(!frame||!model||reason==='frame')return;
    const key=`${scenario}:${s.rank}:${s.loop}:${s.event}:${s.batch}`;
    frame.contentWindow?.postMessage({type:'pp-observation',state:{...s},focus:key!==frameFocus&&reason!=='clock'},'*');frameFocus=key;
  }
  listeners.add((state,reason)=>sendFrame(reason));
  window.addEventListener('message',e=>{
    if(!frame||e.source!==frame.contentWindow)return;
    if(e.data?.type==='pp-timing-applied'||e.data?.type==='pp-timing-ready'){frameFocus='';sendFrame('model');}
    if(model&&e.data?.type==='pp-event-selected')update({event:e.data.id,rank:e.data.rank,loop:e.data.loop},'frame');
  });
  window.addEventListener('pagehide',()=>{persist();cancelAnimationFrame(raf);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&s.playing)update({playing:false},'pause');});
  // Save the exact outgoing clock before navigation, not just the last one-second URL snapshot.
  document.addEventListener('click',e=>{if(e.target.closest('a[href]'))links();},true);
  window.PP_JOURNEY={get state(){return s;},get model(){return model;},get scenario(){return scenario;},get duration(){return duration();},setComparisonEnd(end){comparisonEnd=end;update({},'comparison-ready');},install,update,persist,links,subscribe(fn){listeners.add(fn);return()=>listeners.delete(fn);}};
  window.addEventListener('pp-timing-change',e=>install(e.detail.packet,e.detail.id));
  links();
  const context=document.querySelector('[data-pp-reading-context]');
  if(context){context.textContent=`观察对象 M${s.batch} · PP${s.rank} · L${s.loop+1} · 教学时刻 ${s.time.toFixed(2)} u。返回动画会恢复同一场景与进度。`;}
})();
