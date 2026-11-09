(() => {
  'use strict';
  const M=window.PPConsensus,host=document.querySelector('#consensus-lab');if(!host||!M)return;
  const $=name=>host.querySelector(`[data-role="${name}"]`),btn=name=>host.querySelector(`[data-action="${name}"]`),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const reduced=matchMedia('(prefers-reduced-motion: reduce)'),C={control:'var(--pc-green)',merge:'var(--pc-green)',apply:'var(--pc-green)',message:'var(--pc-purple)',device:'var(--pc-gold)',wait:'var(--pc-gray)',notice:'var(--pc-blue)'};
  const centres=[170,500,830],left=[20,350,680],top=60,bottom=282;
  let tl,t=0,playing=false,raf,last=0,renderedFocus='';
  const text=(x,y,s,cls='',attrs='')=>`<text x="${x}" y="${y}" class="${cls}" ${attrs}>${esc(s)}</text>`;
  const set=a=>a?.length?'{'+a.join(' ' )+'}':'∅';
  const payload=p=>Array.isArray(p)?(tl.mode==='hicache'?'['+p.join(', ')+']':set(p)):`G ${set(p?.good)} / B ${set(p?.bad)}`;
  function route(from,to){const back=from===2&&to===0,y=back?317:36,a=centres[from],b=centres[to],sy=back?bottom:top;return {points:[[a,sy],[a,y],[b,y],[b,sy]],d:`M${a} ${sy}V${y}H${b}V${sy}`,labelY:back?339:18};}
  function point(points,p){const lengths=points.slice(1).map((v,i)=>Math.hypot(v[0]-points[i][0],v[1]-points[i][1]));let distance=p*lengths.reduce((a,b)=>a+b,0);for(let i=0;i<lengths.length;i++){if(distance<=lengths[i]||i===lengths.length-1){const u=distance/lengths[i];return [points[i][0]+(points[i+1][0]-points[i][0])*u,points[i][1]+(points[i+1][1]-points[i][1])*u];}distance-=lengths[i];}return points[0];}
  const statusName={WaitingForInput:'就绪',Bootstrapping:'握手中',Success:'成功',Failed:'失败',Transferring:'传输中'};
  const queueName={bootstrap:'留队',waiting:'待调度',failed:'失败清理',inflight:'持有',done:'已收尾'};
  function badge(x,y,w,label,kind='muted',extra=''){return `<g ${extra}><rect x="${x}" y="${y}" width="${w}" height="25" rx="4" class="pc-chip ${kind}"/>${text(x+w/2,y+17,label,'pc-chip-text','text-anchor="middle"')}</g>`;}
  function draw(s){
    let out='<defs><marker id="pc-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="context-stroke"/></marker></defs>';
    const msg=s.active.find(e=>e.kind==='message');
    for(const [a,b]of [[0,1],[1,2],...(tl.mode==='hicache'?[]:[[2,0]])]){
      const r=route(a,b),active=msg?.from===a&&msg?.to===b;
      out+=`<path d="${r.d}" class="pc-route${active?' active':''}" marker-end="url(#pc-arrow)"/>`;
      if(active){const p=point(r.points,reduced.matches?.5:s.progress(msg));out+=`<g data-motion="${msg.id}" data-x="${p[0].toFixed(2)}" data-y="${p[1].toFixed(2)}"><rect x="${p[0]-7}" y="${p[1]-5}" width="14" height="10" rx="2" fill="var(--pc-purple)"/></g>`;out+=text((centres[a]+centres[b])/2,r.labelY,`${msg.back?'回流':'发送'} ${payload(msg.payload)}`,'pc-payload','text-anchor="middle"');}
    }
    if(!msg)out+=text(500,18,tl.mode==='hicache'?'计数单向传播：PP0 → PP1 → PP2':'候选向后汇总；末级 → 首级 → 中间级 → 末级回流','pc-muted','text-anchor="middle"');
    s.stages.forEach((stage,r)=>{
      const x=left[r],active=s.active.find(e=>e.rank===r&&e.lane==='cpu'+r);
      out+=`<g data-stage="${r}" transform="translate(0,-16)"><rect x="${x}" y="${top+16}" width="300" height="${bottom-top}" rx="7" class="pc-stage ${active?'working':''}"/>${text(x+14,101,`PP${r} · Scheduler / CPU`,'pc-stage-title')}`;
      if(tl.mode==='hicache'){
        out+=text(x+14,125,'ACK 顺序记录 · ✓ 已出队并完成','pc-muted');
        out+=text(x+14,151,'write','pc-code');out+=badge(x+68,134,76,'W0 '+(stage.acks.W0==='finished'?'✓':'就绪'),stage.acks.W0==='finished'?'done':'good');
        out+=text(x+14,185,'load','pc-code');
        ['L0','L1','L2'].forEach((id,k)=>{const e=tl.events.find(e=>e.id===`copy${r}-${k}`),state=stage.acks[id],ready=s.t>=e.end;out+=badge(x+62+k*76,168,70,id+' '+(state==='finished'?'✓':state==='popped'?'取出':ready?'就绪':'拷贝'),state==='finished'?'done':state==='popped'?'wait':ready?'good':'muted',`data-ack="${r}-${id}" data-state="${state}"`);if(state!=='finished'&&!ready){out+=`<rect x="${x+64+k*76}" y="190" width="${66*s.progress(e)}" height="3" fill="var(--pc-gold)"/>`;}});
        out+=text(x+14,217,'收到计数 '+(stage.counts?payload(stage.counts):'尚未收到'),'pc-code');
        out+=text(x+14,242,`已收尾 write=${stage.acks.W0==='finished'?1:0} / load=${['L0','L1','L2'].filter(id=>stage.acks[id]==='finished').length}`);
        const done=tl.events.find(e=>e.id==='continue'+r),waiting=active?.kind==='wait';
        out+=text(x+14,278,waiting?`CPU 阻塞：等 ${active.ack} 本地 event`:s.t>=done.start?'本级缓存事件结束 → 后续选批':'队列条目仍保持各自的加载保护',waiting?'pc-wait-label':'pc-muted');
      }else{
        out+=text(x+14,125,'本轮 poll 快照 · 未采样时不作判断','pc-muted');
        M.ids.forEach((id,k)=>{const status=stage.observed?.[id];out+=badge(x+12+k*94,136,88,id+' '+(statusName[status]||'—'),status==='Failed'?'bad':status==='Success'||status==='WaitingForInput'?'good':'muted');});
        out+=text(x+14,187,'汇合 '+(stage.aggregate===null?'尚未采样':payload(stage.aggregate)),'pc-code');
        out+=text(x+14,212,'回流 '+(stage.returned===null?'尚未到达':payload(stage.returned)),'pc-code');
        out+=text(x+14,238,tl.mode==='bootstrap'?'本地 Req → 队列归属':'本地 Req → 资源持有 / 收尾','pc-muted');
        M.ids.forEach((id,k)=>{const q=stage.queue[id];out+=badge(x+12+k*94,249,88,id+' '+queueName[q],q==='failed'?'bad':['waiting','done'].includes(q)?'good':'muted',`data-req="${r}-${id}" data-queue="${q}"`);});
        // The result of a merge is committed at its end; a moving ID shows local selection in progress.
        if(active?.kind==='merge'&&!reduced.matches){const y=162+18*s.progress(active);out+=`<circle data-selection="${active.id}" cx="${x+16}" cy="${y}" r="3" fill="var(--pc-green)"/>`;}
      }
      out+='</g>';
    });
    if(tl.mode==='hicache')out+=text(500,317,'pop ACK → 等本地 event → 解除保护；计数到达不是数据完成','pc-muted','text-anchor="middle"');
    else if(!msg||!(msg.from===2&&msg.to===0))out+=text(500,339,'本地 Req 不跨进程搬运；线上传的是 rid 名单','pc-muted','text-anchor="middle"');
    $('scene').innerHTML=out;$('scene').dataset.clock=t.toFixed(3);
  }
  function tracks(s){
    const H=tl.lanes.length*24+39,x=n=>168+n/tl.duration*812;
    let out=`<defs><pattern id="pc-wait-hatch" width="6" height="6" patternUnits="userSpaceOnUse"><path d="M0 6L6 0" stroke="var(--pc-gray)" stroke-width="1"/></pattern></defs>`;
    for(let i=0;i<=6;i++){const at=i*tl.duration/6;out+=text(x(at),14,at.toFixed(1),'pc-axis',`text-anchor="${i===6?'end':'start'}"`)+`<path d="M${x(at)} 21V${H-12}" class="pc-grid"/>`;}
    tl.lanes.forEach((lane,i)=>{const y=26+i*24;out+=text(155,y+15,lane.name,'pc-lane','text-anchor="end"')+`<path d="M168 ${y+23}H980" class="pc-grid"/>`;
      tl.events.filter(e=>e.lane===lane.id&&e.start<tl.duration).forEach(e=>{const w=x(Math.min(tl.duration,e.end))-x(e.start),active=e.start<=t&&t<e.end,done=t>=e.end;
        out+=`<g data-event="${e.id}" data-active="${active}" role="button" tabindex="0" aria-label="${esc(e.label)}，${e.start.toFixed(1)} 至 ${e.end.toFixed(1)}" class="pc-event"><rect x="${x(e.start)}" y="${y}" width="${w}" height="21" rx="3" fill="${e.kind==='wait'?'url(#pc-wait-hatch)':C[e.kind]}" opacity="${active?1:done?.65:.25}" stroke="${active?'var(--fg)':'none'}"/><title>${esc(e.label+'：'+e.detail)}</title>${w>48?text(x(e.start)+4,y+15,e.kind==='device'?(tl.mode==='bootstrap'?'已有前向':e.ack||'KV 传输'):e.kind==='wait'?`等 ${e.ack}`:e.kind==='message'?'名单 / 计数':w>88?e.label.slice(0,9):'CPU','pc-bar-label'):''}</g>`;
      });
    });
    out+=`<path d="M${x(t)} 20V${H-12}" class="pc-cursor"/><circle cx="${x(t)}" cy="20" r="3" fill="var(--fg)"/>`;
    $('tracks').setAttribute('viewBox',`0 0 1000 ${H}`);$('tracks').innerHTML=out;$('tracks').dataset.clock=t.toFixed(3);
  }
  function render(){const s=tl.sample(t);host.dataset.time=t.toFixed(3);$('seek').value=t;$('time').textContent=`${t.toFixed(1)} / ${tl.duration.toFixed(1)} · 教学时间，非实测`;draw(s);tracks(s);
    const f=s.focus;const focusKey=f.id+(s.t<f.start?'pending':'active');if(focusKey!==renderedFocus){renderedFocus=focusKey;$('phase').textContent=(s.t<f.start?'即将开始 · ':'')+f.label;$('detail').textContent=f.detail;$('source').href=M.sourceURL(f.source);$('source').textContent='本阶段固定源码 ↗';$('steps').value=f.id;}
    $('result').innerHTML=s.stages.map((r,i)=>`<span><b>PP${i}</b> ${tl.mode==='hicache'?(['L0','L1','L2'].filter(id=>r.acks[id]==='finished').length+' 条 load ACK 已收尾'):M.ids.map(id=>`${id}：${esc(r.outcome[id]||'尚未判定')}`).join('；')}</span>`).join('');
    btn('prev').disabled=t<=0;btn('next').disabled=t>=tl.duration;
  }
  function persist(){const u=new URL(location.href);u.searchParams.set('mode',tl.mode);u.searchParams.set('condition',tl.condition);u.searchParams.set('t',t.toFixed(3));history.replaceState(null,'',u);}
  function pause(){playing=false;cancelAnimationFrame(raf);btn('play').textContent=tl&&t>=tl.duration?'重播':'播放全程';btn('play').setAttribute('aria-pressed','false');}
  function seek(value,write=true){pause();t=Math.min(tl.duration,Math.max(0,Number(value)||0));render();if(write)persist();}
  function tick(now){if(!playing)return;t=Math.min(tl.duration,t+Math.min(.1,(now-last)/1000)*Number($('speed').value));last=now;render();if(t>=tl.duration){pause();persist();}else raf=requestAnimationFrame(tick);}
  function play(){if(playing){pause();persist();return;}if(t>=tl.duration)t=0;playing=true;last=performance.now();btn('play').textContent='暂停';btn('play').setAttribute('aria-pressed','true');raf=requestAnimationFrame(tick);}
  function load(mode,condition,time=0,write=true){pause();tl=M.build(mode,condition);renderedFocus='';$('condition').innerHTML=Object.entries(M.modes[tl.mode].cases).map(([v,label])=>`<option value="${v}">${label}</option>`).join('');$('condition').value=tl.condition;host.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===tl.mode)));$('question').textContent=M.modes[tl.mode].question;$('scope').textContent=M.modes[tl.mode].scope;$('rule').textContent=M.modes[tl.mode].rule;$('steps').innerHTML=tl.events.filter(e=>e.kind!=='device'&&e.start<tl.duration).sort((a,b)=>a.start-b.start).map(e=>`<option value="${e.id}">${e.start.toFixed(1)} · ${e.label}</option>`).join('');$('seek').max=tl.duration;seek(time,write);host.dataset.ready='true';}
  host.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>load(b.dataset.mode,'normal')));
  $('condition').addEventListener('change',()=>load(tl.mode,$('condition').value));$('seek').addEventListener('input',()=>seek($('seek').value));$('steps').addEventListener('change',()=>seek(tl.events.find(e=>e.id===$('steps').value).start));
  btn('play').addEventListener('click',play);btn('reset').addEventListener('click',()=>seek(0));btn('prev').addEventListener('click',()=>seek([...tl.steps].reverse().find(n=>n<t-.01)||0));btn('next').addEventListener('click',()=>seek(tl.steps.find(n=>n>t+.01)??tl.duration));
  const eventSeek=e=>{const node=e.target.closest('[data-event]');if(node)seek(tl.events.find(x=>x.id===node.dataset.event).start);};$('tracks').addEventListener('click',eventSeek);$('tracks').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();eventSeek(e);}});
  document.addEventListener('visibilitychange',()=>{if(document.hidden){pause();persist();}});window.addEventListener('pagehide',pause);reduced.addEventListener('change',render);
  function restore(){const q=new URL(location.href).searchParams;load(q.get('mode'),q.get('condition'),q.get('t'),false);}window.addEventListener('popstate',restore);restore();
  const resume=window.SiteUpdate?.take('pp-consensus');
  if(resume&&M.modes[resume.mode]){
    load(resume.mode,resume.condition,resume.time,false);
    if([...$('speed').options].some(option=>option.value===String(resume.speed)))$('speed').value=String(resume.speed);
  }
  window.SiteUpdate?.register('pp-consensus',()=>{pause();persist();return {mode:tl.mode,condition:tl.condition,time:t,speed:$('speed').value};});
})();
