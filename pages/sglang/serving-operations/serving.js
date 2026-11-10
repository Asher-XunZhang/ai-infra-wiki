(()=>{'use strict';
const M=window.ServingPD,root=document.querySelector('[data-serving-pd]');if(!M||!root)return;
const $=k=>root.querySelector(`[data-role="${k}"]`),esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const txt=(x,y,v,c='',anchor='start')=>`<text x="${x}" y="${y}" class="${c}" text-anchor="${anchor}">${esc(v)}</text>`,box=(g,c='box',extra='')=>`<rect x="${g.x}" y="${g.y}" width="${g.w}" height="${g.h}" rx="7" class="${c}" ${extra}/>`;
const G={c:{x:20,y:36,w:235,h:132},g:{x:295,y:36,w:785,h:132},P:{x:20,y:230,w:500,h:208},D:{x:580,y:230,w:500,h:208}};
const point=(id,side,at=.5)=>{const n=G[id];return side==='top'?[n.x+n.w*at,n.y]:side==='bottom'?[n.x+n.w*at,n.y+n.h]:side==='left'?[n.x,n.y+n.h*at]:[n.x+n.w,n.y+n.h*at];};
const paths={cg:[point('c','right'),point('g','left')],gP:[point('g','bottom',.255),[495.175,195],[270,195],point('P','top')],gD:[point('g','bottom',.745),[879.825,195],[830,195],point('D','top')],PD:[point('P','right',.74),point('D','left',.74)]};
for(const k of ['cg','gP','gD','PD'])paths[k.split('').reverse().join('')]=paths[k].slice().reverse();
const reduced=matchMedia('(prefers-reduced-motion: reduce)');let tl,time=0,playing=false,raf,last=0,inspected='';
function wrap(v,max){const lines=[];let line='',width=0;for(const c of String(v)){const w=/[\x00-\xff]/.test(c)?.55:1;if(width+w>max){lines.push(line);line='';width=0;}line+=c;width+=w;}if(line)lines.push(line);return lines;}
function lines(x,y,v,max,c='small',height=17){return wrap(v,max).map((s,i)=>txt(x,y+i*height,s,c)).join('');}
function diagram(snap){
 const s=snap.state;let html='<title>P/D 服务链：HTTP 状态、请求登记和资源归属</title><defs><marker id="sp-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="context-stroke"/></marker></defs>';
 html+=box(G.c)+txt(34,59,'客户端 / 调用方','heading')+lines(34,86,s.client,17,'value',20)+txt(34,151,'连接结束 ≠ 后端已收尾','muted');
 html+=box(G.g,'box gateway')+txt(310,59,'Gateway · HTTP PDRouter · Rust 异步任务','heading')+txt(310,83,s.pair,'value')+lines(310,105,s.gateway,61)+lines(310,128,s.payload,61,'code',17);
 for(const key of ['P','D']){
  const n=G[key],v=s[key],gpu=snap.active.find(e=>e.lane===(key==='P'?'pg':'dg')),active=snap.active.filter(e=>e.lane.startsWith(key.toLowerCase()));
  html+=`<g data-runtime="${key}">`+box(n,active.length?'box runtime active':'box runtime')+txt(n.x+16,n.y+24,key==='P'?'Prefill Runtime · P1':'Decode Runtime · D1','heading');
  html+=txt(n.x+16,n.y+48,'HTTP / TM · '+v.http,'small')+txt(n.x+16,n.y+69,v.tm+' · '+v.reply,'tiny')+txt(n.x+16,n.y+92,'Scheduler · '+v.queue,'small');
  const gg={x:n.x+16,y:n.y+112,w:216,h:79},kg={x:n.x+248,y:n.y+112,w:236,h:79};
  html+=box(gg,'subbox')+txt(gg.x+10,gg.y+19,'GPU / 设备工作','label')+lines(gg.x+10,gg.y+40,gpu?gpu.label:v.gpu,16,'small',16);
  if(gpu)html+=`<rect x="${gg.x+9}" y="${gg.y+69}" width="${198*snap.progress(gpu)}" height="4" class="compute-progress"/>`;
  html+=box(kg,v.allocated&&!v.released?'subbox held':'subbox')+txt(kg.x+10,kg.y+19,'KV / 资源归属','label')+lines(kg.x+10,kg.y+41,v.kv,18,'small',17)+'</g>';
 }
 for(const e of snap.active.filter(e=>e.edge)){
  const pts=paths[e.edge],d='M'+pts.map(p=>p.join(' ')).join('L'),offset=reduced.matches?0:-(snap.time-e.start)*22/.9;
  html+=`<g data-event-flow="${e.id}" data-edge="${e.edge}"><title>${esc(e.payload)}</title><path d="${d}" class="flow ${e.kind==='data'?'data':''}" marker-end="url(#sp-arrow)" stroke-dashoffset="${offset.toFixed(3)}"/></g>`;
  if(e.edge.includes('c'))html+=txt(275,92,'HTTP','wire-label','middle');
  else if(e.edge.includes('g')){const x=e.edge.includes('P')?270:830;html+=txt(x,219,e.payload,'wire-label','middle');}
  else html+=txt(550,375,e.kind==='data'?'KV':'控制','wire-label','middle');
 }
 html+=txt(20,461,'同一逻辑请求在两端各有 Req / ReqState；HTTP 携带 JSON 或响应，KV 走独立传输面。','muted');
 $('diagram').innerHTML=html;
 $('payload').textContent=snap.active.filter(e=>e.edge).map(e=>`${({g:'Gateway',c:'客户端',P:'P1',D:'D1'})[e.edge[0]]} → ${({g:'Gateway',c:'客户端',P:'P1',D:'D1'})[e.edge[1]]}：${e.payload}`).join('；')||'当前没有画出的跨端交接；观察本地活动与等待条件。';
 $('outcome').textContent=s.outcome;
 $('accessible').textContent=JSON.stringify(s);
}
function tracks(snap){
 const lanes=Object.keys(M.laneNames).filter(l=>tl.events.some(e=>e.lane===l)),ys=Object.fromEntries(lanes.map((l,i)=>[l,38+i*42])),h=lanes.length*42+47,x=t=>182+t/tl.duration*895;
 $('tracks').setAttribute('viewBox',`0 0 1100 ${h}`);$('tracks').dataset.clock=time.toFixed(3);
 let html='<title>同一教学时钟下的执行轨道</title>';
 for(let t=0;t<=tl.duration;t+=2)html+=txt(x(t),19,t+'u','tiny','middle')+`<path d="M${x(t)} 27V${h-11}" class="grid"/>`;
 for(const l of lanes)html+=txt(10,ys[l]+21,M.laneNames[l],'small')+box({x:182,y:ys[l],w:895,h:30},'track-bg');
 for(const e of tl.events){const left=x(e.start),w=x(e.end)-left,y=ys[e.lane],short=wrap(e.label,Math.max(1,(w-10)/10))[0];html+=`<g class="track-event ${e.kind||'control'} ${snap.active.includes(e)?'is-active':''}" role="button" tabindex="0" data-event="${e.id}" aria-label="${esc(e.label+'，'+e.start+' 至 '+e.end+' 教学单位，定位并查看依赖')}"><title>${esc(e.label)}</title>`+box({x:left,y,w,h:30},'track')+box({x:left,y,w:w*snap.progress(e),h:30},'track-fill')+(w>24?txt(left+w/2,y+20,short,'tiny','middle'):'')+'</g>';}
 html+=`<path d="M${x(time)} 27V${h-11}" class="cursor"/>`;$('tracks').innerHTML=html;
 const e=tl.events.find(e=>e.id===inspected)||snap.focus;$('dependency').textContent=e.label+' · 前提：'+((e.depends||[]).map(id=>tl.events.find(x=>x.id===id).label).join('；')||'本场景明确给定的初始条件');
}
function render(){const snap=M.snapshot(tl,time);snap.focus=tl.events.find(e=>e.id===inspected)||snap.focus;root.dataset.mode=tl.mode;root.dataset.condition=tl.condition;root.dataset.time=time.toFixed(3);root.dataset.playing=String(playing);$('title').textContent=tl.question;$('scope').textContent=tl.scope;$('clock').max=tl.duration;$('clock').value=time;$('time').textContent=`${time.toFixed(1)} / ${tl.duration.toFixed(1)} u`;$('play').textContent=playing?'暂停':time>=tl.duration?'重播':'从这里播放';$('prev').disabled=time<=0;$('next').disabled=time>=tl.duration;
 root.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===tl.mode)));$('event-title').textContent=snap.focus.label;$('detail').textContent=snap.focus.detail;$('source').href=M.url(snap.focus.source);const [p,l]=M.sources[snap.focus.source];$('source').textContent=p.split('/').at(-1)+':'+l+' ↗';$('events').value=snap.focus.id;diagram(snap);tracks(snap);depth.update({time,body:DepthProjections.serving(tl,snap),event:snap.focus.label,explanation:snap.focus.detail,href:M.url(snap.focus.source),rows:['P','D'].flatMap(k=>{const v=snap.state[k];return [[k+' HTTP / TM',v.http+' / '+v.tm],[k+' Scheduler',v.queue],[k+' KV / released',v.kv+' / '+v.released]]})});
}
function stop(){playing=false;cancelAnimationFrame(raf);last=0;}
function save(push=false){const u=new URL(location.href);u.searchParams.set('mode',tl.mode);u.searchParams.set('condition',tl.condition);u.searchParams.set('t',time.toFixed(3));history[push?'pushState':'replaceState']({},'',u);}
function seek(t){stop();time=Math.min(tl.duration,Math.max(0,Number(t)||0));inspected='';render();save();}
function configure(mode,condition,t=0,fromURL=false){stop();tl=M.timeline(mode==='admission'?'dispatch':mode,condition);time=Math.min(tl.duration,Math.max(0,Number(t)||0));inspected='';$('condition').innerHTML=Object.entries(tl.cases).map(([k,v])=>`<option value="${k}">${v}</option>`).join('');$('condition').value=tl.condition;$('events').innerHTML=tl.events.slice().sort((a,b)=>a.start-b.start).map(e=>`<option value="${e.id}">${e.label}</option>`).join('');render();if(!fromURL)save(true);}
function locate(id){const e=tl.events.find(e=>e.id===id);if(!e)return;stop();time=e.start;inspected=id;render();save();}
function read(){const u=new URL(location.href);configure(u.searchParams.get('mode')||'health',u.searchParams.get('condition'),u.searchParams.get('t'),true);}
function tick(ts){if(!playing)return;if(last)time=Math.min(tl.duration,time+(ts-last)/1000*Number($('speed').value));last=ts;if(time>=tl.duration)stop();render();if(playing)raf=requestAnimationFrame(tick);else save();}
root.addEventListener('click',e=>{const b=e.target.closest('button[data-mode]');if(b){if(b.dataset.mode!==tl.mode)configure(b.dataset.mode);return;}const bar=e.target.closest('[data-event]');if(bar){locate(bar.dataset.event);}});
root.addEventListener('keydown',e=>{const bar=e.target.closest('[data-event]');if(bar&&['Enter',' '].includes(e.key)){e.preventDefault();bar.dispatchEvent(new MouseEvent('click',{bubbles:true}));root.querySelector(`[data-event="${bar.dataset.event}"]`)?.focus();}});
$('condition').addEventListener('change',()=>configure(tl.mode,$('condition').value));$('events').addEventListener('change',()=>locate($('events').value));$('clock').addEventListener('input',()=>seek($('clock').value));$('reset').addEventListener('click',()=>seek(0));
$('prev').addEventListener('click',()=>seek([0,...tl.events.map(e=>e.end)].sort((a,b)=>b-a).find(t=>t<time-.01)??0));$('next').addEventListener('click',()=>seek(tl.events.map(e=>e.end).sort((a,b)=>a-b).find(t=>t>time+.01)??tl.duration));
$('play').addEventListener('click',()=>{if(playing){stop();render();save();}else{if(time>=tl.duration)time=0;playing=true;inspected='';last=0;render();raf=requestAnimationFrame(tick);}});
const depth=LearningDepth.mount({host:root,before:$('diagram').parentElement,module:[$('diagram').parentElement],onChange:()=>render(),persist:()=>save()});
window.addEventListener('popstate',read);window.addEventListener('pagehide',stop);document.addEventListener('visibilitychange',()=>{if(document.hidden){stop();render();save();}});reduced.addEventListener('change',render);read();
})();
