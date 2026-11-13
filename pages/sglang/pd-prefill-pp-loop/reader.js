/* One continuous clock for batch state, module activity and dependency inspection. */
(() => {
  'use strict';
  const root=document.querySelector('[data-pp-reader]');if(!root)return;
  const J=window.PP_JOURNEY,M=window.PP_JOURNEY_MODEL,dependency=root.dataset.ppReader==='dependency';
  const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt=v=>Number(v).toFixed(2),color=b=>b>=1&&b<=5?`var(--pp-b${b})`:'var(--muted)';
  const name=id=>window.PP_SCENARIOS.find(c=>c.id===id)?.name||'自定义时间';
  const short=n=>n.kind==='copy'?'结果预处理／就绪':n.label;
  const button=(id,text)=>`<button type="button" data-action="${id}">${text}</button>`;
  let sourceStep='',operationsOpen=false;
  let original=null,comparison=null,comparisonId=J.state.compare==='current'?null:J.state.compare,request=0,history=[],lastKey='',packetSeen=null;
  const cache=new Map(),base=new URL('.',document.currentScript.src);
  const get=async id=>{if(!cache.has(id)){const c=window.PP_SCENARIOS.find(c=>c.id===id);cache.set(id,fetch(new URL(c.file,base)).then(r=>{if(!r.ok)throw Error('场景加载失败');return r.json();}));}return M.compile(await cache.get(id));};
  const phaseId=n=>String(n.owner).match(/^M([1-5])(?:$|：)/)?.[1];
  const status=(e,t)=>t<e.start?'尚未开始':M.active(e,t)?e.kind==='wait'?'等待中':'执行中':e.kind==='wait'?'等待已解除':'已完成';
  const opt=(v,label,selected)=>`<option value="${esc(v)}" ${v==selected?'selected':''}>${esc(label)}</option>`;
  const select=(id,label,options,value)=>`<label>${label} <select data-control="${id}">${options.map(([v,label])=>opt(v,label,value)).join('')}</select></label>`;
  root.classList.add('pp-reader');
  root.innerHTML=`<div class="pp-focus-controls"></div><div class="pp-clock-controls motion-controls">${button('play','播放全程')}${button('prev','上一事件')}${button('next','下一事件')}${button('reset','回到起点')}<label>速度 <select data-control="speed">${[.5,1,1.5,2,4].map(v=>opt(v,v+'×',J.state.speed)).join('')}</select></label><output data-clock-value></output></div><label class="pp-scrubber">教学时钟 <input data-control="time" type="range" min="0" max="110" step="0.001" value="0"></label><p class="pp-clock-note">u 为假设时间，非实测。主图与轨道共用此时钟；短交接自动减速，时轴区间不变；空白不等于阻塞等待。</p><div class="pp-current" role="status" aria-live="polite"></div><div class="pp-scene"></div><div class="pp-inspector"></div><div class="pp-payloads"></div><div class="pp-evidence"></div><noscript>本图需要 JavaScript；可阅读本页说明及第三篇的源码顺序表。</noscript>`;
  const $=q=>root.querySelector(q);
  function controls(){
    const activeControl=document.activeElement?.dataset.control;
    const s=J.state,m=J.model;
    const common=select('batch','跟踪',m.batches.map(b=>[b.batch,`M${b.batch}`]),s.batch);
    if(!dependency){
      $('.pp-focus-controls').innerHTML=common+select('mode','观察方式',[['batch','追一批'],['loop','看一次本地迭代']],s.mode)+select('level','视图',[['overview','总览'],['process','执行过程']],s.level)+(s.mode==='loop'?select('rank','迭代定位',[[0,'PP0'],[1,'PP1'],[2,'PP2']],s.rank):'')+(s.mode==='loop'?select('loop','局部迭代',m.loops.filter(l=>l.r===s.rank).map(l=>[l.n,`L${l.n+1} · 新 ${l.current?'M'+l.current:'—'} / 旧 ${l.old?'M'+l.old:'—'}`]),s.loop):'')+button('example','案例：新 M3／旧 M1');
    }else{
      const b=m.batches.find(b=>b.batch===s.batch).ranks[s.rank];
      const options=[['0:5:recv_out','M1 旧结果到达'],['0:5:proxy_work','M2 历史发送完成'],['0:5:gpu','M3 GPU 启动'],['1:3:recv_proxy','M1 上级激活到达'],['0:4:gpu','M2 回载与前向'],[`0:${m.batches[0].ranks[0].releaseLoop}:release`,'本地收尾（当前场景释放 M1）'],[`${s.rank}:${b.currentLoop}:gpu`,`当前 M${s.batch} · PP${s.rank} 前向`]];
      if(!options.some(o=>o[0]===s.event)&&m.graph[s.event])options.push([s.event,short(m.graph[s.event])]);
      $('.pp-focus-controls').innerHTML=common+select('rank','流水级',[[0,'PP0'],[1,'PP1'],[2,'PP2']],s.rank)+select('event','追踪依赖',options,s.event)+button('focus-event','定位到等待区间')+(history.length?button('back','返回前驱选择'):'')+`<details class="pp-compare-options"><summary>对照条件</summary>${select('compare','对照 B',[['current','跟随当前场景'],...window.PP_SCENARIOS.map(c=>[c.id,c.name])],comparisonId||'current')}<p>默认 A 为原示例，B 跟随页面场景；独立选择只改变 B，并回到起点暂停。</p></details>`;
    }
    if(activeControl)$(`[data-control="${activeControl}"]`)?.focus({preventScroll:true});
  }
  function activityRows(start=0,end=J.duration,m=J.model,forceProcess=false){
    const s=J.state;
    if(!forceProcess&&s.level==='overview')return m.batches.find(b=>b.batch===s.batch).ranks.map(r=>({label:`PP${r.r} · M${s.batch}`,bars:r.phases.map((p,i)=>({...p,label:M.phases[i],color:color(s.batch),phase:i,range:i!==2&&i!==4})),note:'生命周期阶段范围，含等待；不是资源忙碌率'}));
    const rows=[];
    for(const rank of [0,1,2]){
      const nodes=m.nodes.filter(n=>n.r===rank),groups=M.cpuGroups(m,rank);
      rows.push({label:`PP${rank} CPU`,rank,lane:'cpu',stageStart:true,bars:groups},{label:`PP${rank} GPU`,rank,lane:'gpu',bars:nodes.filter(n=>n.kind==='gpu')});
      for(const [kind,label] of [['h2d','回载 KV → GPU'],['copy','结果就绪事件'],['kv','本级 KV → D']]){
        const bars=nodes.filter(n=>n.kind===kind);
        if(bars.some(n=>n.end>start&&n.start<end))rows.push({label,rank,lane:kind,bars});
      }
    }
    return rows.map(r=>({...r,bars:r.bars.map(n=>{
      const batch=n.members?new Set(n.members.map(phaseId)):null;
      const owner=batch?.size===1?[...batch][0]:batch?null:phaseId(n);
      const caption=n.kind==='wait'?`等 ${n.owner.match(/^M[1-5]$/)?n.owner+' ':''}${n.label.replace(/^等：|等待\/接收|等待/g,'').replace('旧 batch output','output').replace('上轮 proxy send work','激活发送完成')}`:n.group?n.label:owner?'M'+owner:short(n);
      return {...n,caption,label:caption,color:owner?color(+owner):'var(--accent)',range:n.kind==='wait'};
    })}));
  }
  function rail(rows,start,end,time,id,dependencyInfo=null,relation=null){
    const joint=id==='joint'||id.startsWith('context-'),w=1000,left=148,right=16,top=30,rowHeight=joint?32:42,barHeight=joint?22:26,h=top+rows.length*rowHeight+22,x=t=>left+(t-start)/(end-start)*(w-left-right);
    let svg=`<svg class="pp-rails" viewBox="0 0 ${w} ${h}" role="img" aria-label="${joint?'三个 PP Stage 同时执行的轨道':esc(id)}"><defs><pattern id="${id}-wait" width="7" height="7" patternUnits="userSpaceOnUse"><path d="M0 7L7 0" stroke="currentColor" opacity=".18"/></pattern></defs>`;
    for(let i=0;i<=5;i++){const t=start+(end-start)*i/5;svg+=`<path d="M${x(t)} 24V${h-16}" class="pp-grid"/><text x="${x(t)}" y="16" text-anchor="${i===5?'end':'middle'}">${fmt(t)}</text>`;}
    rows.forEach((r,i)=>{
      const y=top+i*rowHeight;
      if(joint)svg+=`<g data-rail-rank="${r.rank}" data-lane="${r.lane}"><rect x="0" y="${y}" width="1000" height="${rowHeight}" fill="var(--soft)" opacity="${r.rank%2?.65:.2}"/>${r.stageStart?`<path d="M0 ${y-1}H1000" stroke="var(--border)" stroke-width="2"/>`:''}`;
      svg+=`<text x="0" y="${y+(joint?19:21)}" class="pp-lane-label">${esc(r.label)}</text><path d="M${left} ${y+rowHeight-6}H${w-right}" class="pp-grid"/>`;
      r.bars.forEach(n=>{
        if(n.end<start||n.start>end)return;
        const xx=x(Math.max(start,n.start)),ww=Math.max(1,x(Math.min(end,n.end))-xx),done=M.progress(n,time),c=n.color||'var(--accent)';
        const nodeId=n.target||n.id,selectable=nodeId&&(J.model.graph[nodeId]||n.group);
        const selected=!dependency&&J.state.inspect===n.id;
        const ids=n.members?.flatMap(m=>[m.id,m.target].filter(Boolean))||[n.id];
        const linked=relation?(ids.includes(relation.target.id)?'target':ids.some(id=>relation.sources?.includes(id)||id===relation.source?.id||id===relation.producer?.id)?'source':''):'';
        svg+=`<g ${selectable?`data-${dependency?'node':'inspect'}="${esc(dependency?nodeId:n.id)}" data-related="${linked}" aria-pressed="${selected}" tabindex="0" role="button" aria-label="${esc(n.label)}，${fmt(n.start)} 至 ${fmt(n.end)} u"`:''}><title>${esc(n.label)} · ${fmt(n.start)}–${fmt(n.end)} u${n.range?'（含等待或阶段范围）':''}</title><rect x="${xx}" y="${y+3}" width="${ww}" height="${barHeight}" rx="3" fill="${c}" fill-opacity=".12" stroke="${c}" stroke-width="${selected||linked?3:M.active(n,time)?2:.6}"/>`;
        if(done>0)svg+=`<rect x="${xx}" y="${y+3}" width="${Math.min(ww,Math.max(0,x(Math.min(n.end,time,end))-xx))}" height="${barHeight}" rx="3" fill="${n.range?`url(#${id}-wait)`:c}" fill-opacity="${n.range?1:.28}"/>`;
        if(n.start<start)svg+=`<path data-continuation="before" d="M${xx+7} ${y+9}l-4 7 4 7" fill="none" stroke="currentColor" stroke-width="1.5"/>`;
        if(n.end>end)svg+=`<path data-continuation="after" d="M${xx+ww-7} ${y+9}l4 7-4 7" fill="none" stroke="currentColor" stroke-width="1.5"/>`;
        const caption=typeof n.phase==='number'?M.phases[n.phase]:n.caption||(n.kind==='wait'?'等前置条件':phaseId(n)?'M'+phaseId(n):'');
        if(ww>Math.max(32,[...caption].reduce((w,c)=>w+(c.charCodeAt(0)>255?12:7),14)))svg+=`<text x="${xx+ww/2}" y="${y+21}" text-anchor="middle">${esc(caption)}</text>`;
        svg+='</g>';
      });
      if(joint)svg+='</g>';
    });
    if(dependencyInfo){
      const target=dependencyInfo.n,ty=top+(rows.length-1)*rowHeight+3;
      svg+=`<defs><marker id="${id}-dependency-arrow" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L8 4L0 8" fill="currentColor"/></marker></defs>`;
      dependencyInfo.deps.forEach((p,i)=>{const sx=x(p.end),sy=top+i*rowHeight+29,tx=x(target.start),bend=ty-7-i*3;
        svg+=`<path data-dependency-edge="${p.id}" d="M${sx} ${sy}V${bend}H${tx}V${ty}" fill="none" stroke="currentColor" stroke-width="1.2" opacity=".45" ${p.r!==target.r?'stroke-dasharray="4 3"':''} marker-end="url(#${id}-dependency-arrow)"/>`;
      });
    }
    if(time>=start&&time<=end)svg+=`<path data-clock-cursor d="M${x(time)} 24V${h-15}" class="pp-cursor"/><circle cx="${x(time)}" cy="25" r="4" fill="var(--accent)"/>`;
    return `<div class="pp-chart-scroll">${svg}</svg></div>`;
  }
  function flowList(){
    const s=J.state,m=J.model;
    const list=M.transfers(m,s.batch);
    // Identical payload routes may occur at different times; each keeps its own interval.
    return `<div class="pp-flows">${list.map(n=>{
      const a=M.active(n,s.time),done=s.time>=n.end,from=n.kind==='h2d'?`PP${n.r} host`:`PP${n.r}`,to=n.kind==='h2d'?`PP${n.r} GPU`:n.to==='D'?'Decode':`PP${n.to}`;
      return `<div class="pp-flow ${a?'active':''}" data-transfer="${n.id}" data-active="${a}"><span>${from} → ${to}<small>M${s.batch} · ${esc(n.payload)}</small></span><svg viewBox="0 0 180 20" role="img" aria-label="${esc(n.payload)} ${a?'传递中':done?'已交接':'尚未发送'}"><defs><marker id="arrow-${n.id.replaceAll(':','-')}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10" fill="${color(s.batch)}"/></marker></defs><path d="M4 10H174" fill="none" stroke="${color(s.batch)}" stroke-width="1.5" opacity="${a?.55:.4}" marker-end="url(#arrow-${n.id.replaceAll(':','-')})"/>${a?`<path d="M4 10H174" fill="none" stroke="${color(s.batch)}" stroke-width="3" stroke-dasharray="9 7" stroke-dashoffset="${-((s.time-n.start)*32)%16}" data-pulse="1"/>`:''}</svg><small>${a?'传递中':done?'已交接':'尚未发送'}</small></div>`;
    }).join('')}</div>`;
  }
  function rankCards(compact=false){
    const s=J.state,m=J.model;
    return `<div class="pp-ranks ${compact?'pp-ranks-compact':''}">${[0,1,2].map(r=>{
      const v=M.rankState(m,s.batch,r,s.time),p=v.b.phases[2];
      return `<article class="pp-rank" data-rank-state="${r}" style="--batch-color:${color(s.batch)}"><h3>PP${r} <small>Prefill Stage ${r}</small></h3><strong>M${s.batch} · ${v.label}</strong><div class="pp-compute-label">M${s.batch} 本级前向进度</div><div class="pp-compute" aria-label="本级前向的教学进度"><span style="width:${M.progress(p,s.time)*100}%"></span></div><p>GPU：${v.gpu?esc(v.gpu.owner)+' 前向计算':'无前向执行'}</p><p>CPU：${v.cpu?`${v.loop?'L'+(v.loop.n+1)+' · ':''}${esc((M.stages[v.cpu.phase]||short(v.cpu))+(v.cpu.kind==='wait'?' · 等前置条件':''))}`:'暂无调度活动'}</p></article>`;
    }).join('')}</div>`;
  }
  function loopContext(){
    const s=J.state,l=J.model.loops.find(l=>l.r===s.rank&&l.n===s.loop);
    const active=M.active(l,s.time),next=(l.slot+1)%3;
    return `<div class="pp-loop-context"><strong>观察 PP${s.rank} · L${s.loop+1}（${fmt(l.start)}–${fmt(l.end)} u）</strong><span>${active?'此刻正在本次迭代':s.time<l.start?'本次迭代尚未开始':'本次迭代已结束，异步活动可继续'}</span><div class="pp-slots">${[0,1,2].map(i=>`<div><b>槽位 s${i}</b><span>${i===l.slot?`mb_id · ${l.current?'当前 M'+l.current:'无新 Batch'}`:i===next?`next_mb_id · ${l.old?'旧 M'+l.old:'无旧结果'}`:'其他历史状态'}</span></div>`).join('')}</div><p>此处是所选迭代的对象关系，非此刻的槽位快照。清理候选来自独立的 release 名单；本次实际清理 ${l.released.map(b=>'M'+b).join('、')||'无'}。</p>${button('loop-start','定位本次迭代起点')}</div>`;
  }
  function sourceDetails(){
    const s=J.state,m=J.model;
    if(!s.inspect)return `<p class="pp-inspect-hint">点击轨道上的阶段、等待或传输，展开对应操作与源码。${button('inspect-current','查看此刻 CPU 操作')}</p>`;
    const groups=M.cpuGroups(m,+s.inspect.split(':')[0]),group=groups.find(g=>g.id===s.inspect);
    const members=group?.members||[m.graph[s.inspect]].filter(Boolean);
    if(!members.length)return '';
    const e=members.find(e=>e.id===sourceStep)||members.find(e=>M.active(e,s.time))||members.find(e=>e.start>s.time)||members.at(-1);
    const n=m.graph[e.target||e.id],d=M.dependencies(m,n.id,s.time),guide=window.PP_STEP_GUIDE.describe(m.packet.model,n,e);
    const l=m.loops.find(l=>l.r===n.r&&l.n===n.n);
    const target=new URL('index.html',location.href);target.search=location.search;target.hash='dependency-comparison';target.searchParams.set('event',n.id);target.searchParams.set('rank',n.r);target.searchParams.set('loop',n.n);target.searchParams.set('t',s.time.toFixed(3));target.searchParams.delete('inspect');
    return `<section class="pp-source pp-inline-source" aria-label="所选阶段的源码" data-inspection="${esc(s.inspect)}"><div class="pp-source-heading"><h3>PP${n.r} · L${n.n+1} · ${esc((group?.label||short(n)).replace(/^等：/,''))}</h3>${button('close-source','收起源码')}</div><p class="pp-note">所选区间 ${fmt(group?.start??n.start)}–${fmt(group?.end??n.end)} u；观察时刻 ${fmt(s.time)} u。展开不改变跟踪对象或播放状态。</p>${members.length>1?`<div class="pp-operation-picker">${button('operations',`${operationsOpen?'收起':'展开'} ${members.length} 个具体操作`)}${operationsOpen?select('operation','选择操作',[['','随时钟查看本阶段'],...members.map(x=>[x.id,x.label])],sourceStep):''}</div>`:''}<div class="pp-source-body"><h4>${esc(guide.title)} <span class="pp-operation-status">${status(e,s.time)}</span></h4><p class="pp-object-line"><b>对象：</b>${esc(n.owner)}。本轮前向 ${l.current?'M'+l.current:'无'}；旧结果 ${l.old?'M'+l.old:'无'}；槽位 mb_id = ${l.slot}，next_mb_id = ${(l.slot+1)%3}。</p><div class="pp-source-grid"><div><b>${e.kind==='wait'&&s.time<e.end?'条件满足后才执行':'本步处理'}</b><p>${esc(guide.what)}</p></div><div><b>交给后续什么</b><p>${esc(guide.next)}</p></div></div><p class="pp-note">${esc(guide.boundary)}</p><div class="pp-gates"><strong>开始条件 · 当前教学时刻</strong>${d.deps.length?`<ul>${d.deps.map(p=>`<li data-gate="${esc(p.id)}" data-ready="${p.ready}"><span>${p.ready?'✓ 已满足':'○ 未满足'}</span> ${esc(short(p))} · ${esc(p.owner)} · ${fmt(p.end)} u<br><small>${esc(p.type)}</small></li>`).join('')}</ul>`:'<p>模型入口，无前置事件。</p>'}</div>${guide.wait?`<p>${esc(guide.wait)}</p>`:''}<p class="pp-source-links">${guide.refs.map(ref=>`<a href="${M.url(ref,m.packet.model.sourceCommit)}" target="_blank" rel="noopener">${esc(ref)} ↗</a>`).join(' · ')} · <a data-pp-fixed href="${esc(target.href)}">追查依赖 →</a> · <a href="notes.html#section-${n.id.endsWith(':release')?'5':'4'}">主循环源码 →</a></p></div></section>`;
  }
  let inspectorKey='',pickerKey='';
  function renderInspector(){
    const host=$('.pp-inspector');
    if(J.state.level!=='process'){host.replaceChildren();inspectorKey='';return;}
    const html=sourceDetails(),key=J.state.inspect,controlKey=JSON.stringify([key,sourceStep,operationsOpen]);
    if(!key||key!==inspectorKey||!host.querySelector('.pp-source-body')){
      host.innerHTML=html;inspectorKey=key;pickerKey=controlKey;return;
    }
    // Keep the native operation picker mounted while the clock advances.
    // Only its live explanation and readiness indicators need new content.
    const template=document.createElement('template');template.innerHTML=html;
    const panel=host.querySelector('.pp-inline-source'),next=template.content.querySelector('.pp-inline-source');
    panel.querySelector(':scope>.pp-note').textContent=next.querySelector(':scope>.pp-note').textContent;
    panel.querySelector('.pp-source-body').innerHTML=next.querySelector('.pp-source-body').innerHTML;
    if(controlKey!==pickerKey){
      const old=panel.querySelector('.pp-operation-picker'),picker=next.querySelector('.pp-operation-picker');
      if(old&&picker)old.replaceWith(picker);else if(old)old.remove();else if(picker)panel.insertBefore(picker,panel.querySelector('.pp-source-body'));
      pickerKey=controlKey;
    }
  }
  function relationPanel(relations,selected){
    const s=J.state,m=J.model;
    if(!selected)return '<p class="pp-note">此刻没有本图可追踪的跨级等待或载荷交接。</p>';
    const r=selected,p=r.source,n=r.target,producer=r.producer,d=M.dependencies(m,n.id,s.time);
    const sending=r.transport&&M.active(p,s.time),arrived=s.time>=p.end;
    const consumer=s.time<n.start?arrived?d.ready?'前置条件已齐，尚未开始本步':'此项已满足，仍等本地其他条件':'等待此项前置条件':s.time<n.end?'正在接收／处理':'本步已完成';
    const wireLabel=r.transport?`${r.payload} · ${sending?'传递中':arrived?'已到达':'尚未发送'}`:'完成依赖 · '+(arrived?'已满足':'未满足');
    const c=color(+phaseId(p)),wire=r.transport?'data':'dependency';
    return `<section class="pp-relations" aria-label="跨 Stage 交接与依赖"><div class="pp-relation-tabs">${relations.map(x=>`<button type="button" data-relation="${esc(x.id)}" aria-pressed="${x.id===r.id}">PP${x.source.r} → PP${x.target.r} · ${x.transport?esc(x.source.owner)+' '+esc(x.payload):'控制／完成依赖'}</button>`).join('')}${s.relation?button('relation-auto','跟随当前等待'):''}</div><div class="pp-relation-chain" data-cross-relation="${esc(r.id)}" data-wire-kind="${wire}"><button class="pp-relation-end" data-inspect="${producer.id}"><b>PP${producer.r} · ${producer.kind==='gpu'?'GPU 前向':'CPU 操作'}</b><span>${esc(producer.owner)} · ${status(producer,s.time)}</span><small>${esc(short(producer))} · ${fmt(producer.start)}–${fmt(producer.end)} u</small></button><div class="pp-relation-wire"><b>${esc(wireLabel)}</b><svg viewBox="0 0 200 22" aria-label="${esc(wireLabel)}"><defs><marker id="cross-arrow" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L8 4L0 8" fill="${c}"/></marker></defs><path d="M3 11H192" fill="none" stroke="${c}" opacity=".55" stroke-width="1.5" ${r.transport?'':'stroke-dasharray="3 4"'} marker-end="url(#cross-arrow)"/>${sending?`<path data-cross-pulse="${p.id}" d="M3 11H192" fill="none" stroke="${c}" stroke-width="3" stroke-dasharray="9 7" stroke-dashoffset="${-((s.time-p.start)*32)%16}"/>`:''}</svg><small>${r.transport?`${fmt(p.start)}–${fmt(p.end)} u；到达后仍按本地条件推进`:esc(r.type)+'；不表示载荷正在传输'}</small></div><button class="pp-relation-end" data-inspect="${r.wait?.id||n.id}"><b>PP${n.r} · CPU L${n.n+1}</b><span>${esc(n.owner)} · ${consumer}</span><small>${esc(short(n))} · ${fmt(n.start)} u 起</small></button></div><p class="pp-note">轨道粗边框关联两端操作；可点击端点查看源码。${r.transport&&s.time<p.start&&s.time>=producer.end?'本级计算已完成，发送还需等 CPU 提交及其余前置条件。':''}${!r.transport?'控制／Work 关系沿模型依赖显示，不能把等待长度当作真实 CPU 阻塞时长。':''}</p></section>`;
  }
  function renderQuick(){
    const s=J.state,m=J.model;
    const stages=[0,1,2].map(r=>M.rankState(m,s.batch,r,s.time));
    $('.pp-current').textContent=`M${s.batch}：${stages.map((v,r)=>`PP${r} ${v.label}`).join('；')}`;
    const process=s.level==='process';
    const transferOpen=$('.pp-transfer-details')?.open??!process;
    let min=0,max=J.duration;
    if(process){const window=M.processWindow(m,s.time);min=window.start;max=window.end;}
    const rows=activityRows(min,max),relations=process?M.crossRelations(m,s.time,s.batch,s.relation):[],related=relations.find(r=>r.id===s.relation)||relations[0];
    const flows=`<details class="pp-transfer-details" ${transferOpen?'open':''}><summary>M${s.batch} 的全部载荷路径 · 激活、output 与 KV</summary>${flowList()}<p class="pp-note">完整箭头表示路线；只有模型中的在途区间出现脉冲。output 环路为 PP2 → PP0 → PP1 → PP2；KV 各自发往 Decode。物理主机放置未指定。</p></details>`;
    const before=document.activeElement?.dataset,focus=before?.inspect?['inspect',before.inspect]:before?.control==='operation'?['control','operation']:before?.relation?['relation',before.relation]:before?.action?['action',before.action]:null;
    const loopOpen=$('.pp-loop-reference')?.open;
    $('.pp-scene').innerHTML=rankCards(process)+(process?'':flows)+(s.mode==='loop'?`<details class="pp-loop-reference" ${loopOpen?'open':''}><summary>局部迭代参考：PP${s.rank} · L${s.loop+1}（全部 Stage 仍同步显示）</summary>${loopContext()}</details>`:'')+`<h3 class="pp-track-title">${process?'跨 Stage · 同一时钟下的执行过程':'同一时钟下的生命周期'} <small>${fmt(min)}–${fmt(max)} u${process?' · 全部 Stage 共用窗口':''}</small></h3><p class="pp-note">${process?'三个 Stage 同时显示；L 是各级本地迭代。CPU 操作归组，斜纹单独表示等待；回载、结果就绪与 KV 按独立轨道推进。只省略本窗口内没有活动的异步轨道。':'斜纹阶段包含等待；实色前向与 KV 段表示对应活动。'}</p>`+rail(rows,min,max,s.time,process?'joint':'journey',null,related)+(process?relationPanel(relations,related):'')+`<p class="pp-note">${process?'数据传输只在在途区间有脉冲；控制／完成依赖保持静态。边缘尖括号表示活动延续到窗口外。':'展开执行过程，可同时查看各级 CPU、GPU 与异步工作。'}窄屏可横向查看轨道。</p>`;
    renderInspector();$('.pp-payloads').innerHTML=process?flows:'';
    if(focus)$(`[data-${focus[0]}="${focus[1]}"]`)?.focus({preventScroll:true});
    $('.pp-evidence').innerHTML=`<p>固定条件：<code>279339f113</code> · PP=3 · depth=0 · 固定选批 · 5 个单请求 microbatch。L 是各 rank 的局部迭代，非全局同步轮次。<strong>P 清理完成不等于 D 生成完成。</strong></p><details><summary>模型简化与源码边界</summary><p>HiCache 回载在本图按“全部 H2D 完成后前向”粗化，源码支持按层等待；同类回载／KV 串行及部分 Work 配对门控也是模型假设，非性能测量。结果预处理记录 d2h_event；仅 sampling_mask_output 等指定分支触发 copy_to_cpu，不能把所有就绪事件都画成 D2H 载荷。<a href="notes.html#section-1">完整基线与边界 →</a></p></details>`;
  }
  function dependencyPanel(model,label,id){
    const s=J.state,d=M.dependencies(model,s.event,s.time);if(!d)return `<p>${esc(label)} 中没有此事件，请选择共同事件。</p>`;
    const prev=model.graph[d.n.prev];
    const rows=d.deps.map(p=>({label:`PP${p.r} ${p.kind==='cpu'?'CPU':p.kind.toUpperCase()}`,bars:[{...p,label:short(p),color:color(+phaseId(p))}]}));
    rows.push({label:`消费者 PP${d.n.r}`,bars:[...(prev&&prev.end<d.n.start?[{start:prev.end,end:d.n.start,label:'等待前置条件',range:true,color:'var(--muted)'}]:[]),{...d.n,label:short(d.n),color:'var(--accent)'}]});
    const first=M.dependencies(original,s.event,s.time),second=M.dependencies(comparison||J.model,s.event,s.time);
    const times=[first,second].filter(Boolean).flatMap(v=>[v.n.start,v.n.end,...v.deps.flatMap(p=>[p.start,p.end])]);
    const min=Math.max(0,Math.min(...times)-1),max=Math.max(...times)+2;
    const pending=d.deps.filter(p=>!p.ready),gate=s.time<d.n.start?pending.length?`等待：${pending.map(p=>short(p)).join('、')}`:'前置条件已齐，尚未到本图安排的起点':status(d.n,s.time);
    const released=model.packet.model.releases.filter(r=>r.r===d.n.r&&r.n===d.n.n).flatMap(r=>r.batches);
    const releaseNote=d.n.id.endsWith(':release')?`<p class="pp-note">本次收尾调用实际清理：${released.map(b=>'M'+b).join('、')||'无'}。调用开始不代表任意请求已满足 release 条件；同一局部迭代在不同场景可以清理不同 Batch。</p>`:'';
    const upstream=d.deps.flatMap(p=>p.kind==='message'?p.deps.map(id=>model.graph[id]).filter(Boolean):[]);
    const highlight={target:d.n,sources:[...d.deps,...upstream].map(p=>p.id)};
    const window=M.processWindow({end:J.duration},s.time),context=activityRows(window.start,window.end,model,true);
    // Payload / event tracks are included only when relevant to the selected
    // dependency. Their durations remain those of the original model.
    for(const rank of [0,1,2]){
      const messages=[d.n,...d.deps,...upstream].filter(p=>p.r===rank&&p.kind==='message');
      if(messages.length){const at=context.findIndex(r=>r.rank>rank);context.splice(at<0?context.length:at,0,{label:'消息在途',rank,lane:'message',bars:[...new Map(messages.map(p=>[p.id,p])).values()].map(p=>({...p,label:short(p),color:color(+phaseId(p))}))});}
    }
    const upstreamNote=d.deps.filter(p=>p.kind==='message').map(p=>{
      const parents=p.deps.map(id=>model.graph[id]).filter(Boolean);
      return `<p class="pp-note" data-producer-chain="${p.id}">PP${p.r} · ${esc(p.owner)}：${parents.map(q=>`${q.kind==='gpu'?'GPU 前向':'CPU 提交'} ${fmt(q.end)} u 完成`).join(' ＋ ')} → 消息 ${fmt(p.start)}–${fmt(p.end)} u 在途 → PP${d.n.r} ${fmt(d.n.start)} u 开始本步。<button data-dependency="${p.id}">追查这次发送</button></p>`;
    }).join('');
    const graphOpen=$(`[data-direct-graph="${id}"]`)?.open;
    return `<article class="pp-dependency-panel" data-model-panel="${id}"><h3>${esc(label)} <small>${fmt(d.n.start)} u 开始</small></h3><p class="pp-gate">${esc(gate)}</p>${rail(context,window.start,window.end,s.time,'context-'+id,null,highlight)}<p class="pp-note">三个 Stage 共用 ${fmt(window.start)}–${fmt(window.end)} u 窗口；实边框关联前置，虚边框关联观察步骤。窄屏可横向查看；窗口外操作在下方保留准确时刻。</p>${upstreamNote}<h4 class="pp-prerequisite-title">本步开始前，以下条件必须全部满足</h4><div class="pp-dep-list">${d.deps.map(p=>`<button data-dependency="${p.id}" class="${p.ready?'ready':'pending'}"><span>${p.ready?'✓ 已满足':'○ 未完成'} · PP${p.r} · ${esc(p.owner)}</span><strong>PP${p.r} · ${esc(short(p))}</strong><small>${esc(p.type)} · ${fmt(p.end)} u${p.critical?' · 决定本图起点':''}</small></button>`).join('')}</div><details class="pp-direct-graph" data-direct-graph="${id}" ${graphOpen?'open':''}><summary>直接前置关系 · 同一绝对时间的局部放大</summary>${rail(rows,min,max,s.time,id,d)}<p class="pp-note">箭头表示前置完成 → 本步开始，保持静态；点击前驱可继续追溯。此图只放大依赖，不代表其他 Stage 停止执行。</p></details>${releaseNote}${d.n.inputs?.length?`<p class="pp-note">另有源码状态输入：${d.n.inputs.map(i=>esc(i.note)).join('；')}。这些引用用于解释数据依据，不额外制造阻塞等待。</p>`:''}</article>`;
  }
  const comparisonDuration=()=>Math.max(original?.end||0,comparison?.end||0,(!original||(comparisonId&&!comparison))?J.state.time:0);
  function renderDependency(){
    if(!original||!J.model)return;
    if(comparisonId&&!comparison){$('.pp-current').textContent='正在读取独立对照场景…';$('.pp-scene').innerHTML='';return;}
    const s=J.state,b=comparison||J.model,n=b.graph[s.event];
    if(!n)return;
    const a=original.graph[s.event],different=comparisonId?comparisonId!=='baseline':J.scenario!=='baseline';
    $('.pp-current').textContent=`观察 ${n.owner} · PP${n.r} ${short(n)}。${different?'A／B 在相同刻度下对照同一事件。':'当前为原示例；改变下方场景后再对照起点变化。'}`;
    $('.pp-scene').innerHTML=dependencyPanel(original,'A · 原示例','dep-a')+(different?dependencyPanel(b,`B · ${name(comparisonId||J.scenario)}${comparisonId?'（独立对照）':'（当前场景）'}`,'dep-b'):'')+`<p class="pp-note">${different&&a?`操作起点 ${fmt(a.start)} → ${fmt(n.start)} u；` :''}长条是模型安排，不代表实测性能。依赖线无传输脉冲；前置条件的完成与消费者启动分别标注。</p>`;
    $('.pp-evidence').innerHTML=`<p><strong>同一时钟、同一观察事件。</strong> ${comparisonId?'B 使用明确选定的独立场景；完整图仍对应“当前场景”。':'完整图与 B 都跟随当前场景。'} 查看 <a href="quick.html#quick-timeline">Batch 执行旅程</a> 或 <a href="notes.html#section-3">主循环源码顺序</a>。</p><p class="pp-note">GPU 等 H2D 是整段回载的教学简化；Work 配对门控依赖模型和通信后端，不能据此认定 CPU 在真实运行中阻塞了同样时长。</p>`;
  }
  function render(s,reason){
    const m=J.model;if(!m)return;
    root.dataset.time=s.time.toFixed(3);root.dataset.batch=s.batch;root.dataset.level=s.level;root.dataset.scenario=J.scenario;
    const key=JSON.stringify([s.batch,s.rank,s.loop,s.mode,s.level,s.event,J.scenario,comparisonId,history.length]);
    if(key!==lastKey||reason==='model'){controls();lastKey=key;}
    $('[data-action="play"]').textContent=s.playing?'暂停':'播放全程';
    $('[data-clock-value]').textContent=`${fmt(s.time)} / ${fmt(J.duration)} u`;
    const slider=$('[data-control="time"]');slider.max=J.duration;slider.value=s.time;
    $('[data-control="speed"]').value=s.speed;
    dependency?renderDependency():renderQuick();
    if(reason!=='clock')J.links();
  }
  function milestones(){
    const s=J.state,m=J.model;
    if(dependency){const d=M.dependencies(comparison||m,s.event,s.time);return [...new Set([0,...d.deps.flatMap(p=>[p.start,p.end]),d.n.start,d.n.end,J.duration])].sort((a,b)=>a-b);}
    const b=m.batches.find(b=>b.batch===s.batch);
    return [...new Set([0,J.duration,...b.ranks.flatMap(r=>[r.admitted,r.releaseStart,...r.phases.flatMap(p=>[p.start,p.end])])])].sort((a,b)=>a-b);
  }
  root.addEventListener('click',async e=>{
    const action=e.target.closest('[data-action]')?.dataset.action,node=e.target.closest('[data-node]')?.dataset.node,dep=e.target.closest('[data-dependency]')?.dataset.dependency;
    const s=J.state;if(!J.model)return;
    const inspect=e.target.closest('[data-inspect]')?.dataset.inspect;
    if(inspect){sourceStep='';operationsOpen=false;J.update({inspect},'inspect');$('.pp-inline-source')?.scrollIntoView({block:'nearest',behavior:'instant'});return;}
    if(e.target.closest('[data-relation]')){J.update({relation:e.target.closest('[data-relation]').dataset.relation},'relation');return;}
    if(action==='relation-auto'){J.update({relation:''},'relation');return;}
    if(action==='close-source'){J.update({inspect:''},'inspect');return;}
    if(action==='operations'){operationsOpen=!operationsOpen;renderQuick();return;}
    if(action==='inspect-current'){const g=M.cpuGroups(J.model,s.rank).find(g=>M.active(g,s.time));if(g){sourceStep='';operationsOpen=false;J.update({inspect:g.id},'inspect');$('.pp-inline-source')?.scrollIntoView({block:'nearest',behavior:'instant'});}return;}
    if(node){const n=(comparison||J.model).graph[node]||original?.graph[node];if(!n)return;J.update({event:node,rank:n.r,loop:n.n},'focus');return;}
    if(dep){history.push(s.event);const n=(comparison||J.model).graph[dep];J.update({event:dep,rank:n.r,loop:n.n},'focus');return;}
    if(action==='play')J.update({playing:!s.playing,...(s.time>=J.duration?{time:0}:{})},'play');
    if(action==='reset')J.update({time:0,playing:false},'seek');
    if(action==='prev'||action==='next'){const times=milestones();J.update({playing:false,time:action==='prev'?[...times].reverse().find(t=>t<s.time-.01)??0:times.find(t=>t>s.time+.01)??J.duration},'seek');}
    if(action==='example')J.update({batch:3,rank:0,loop:5,event:'0:5:gpu',mode:'loop',level:'process',time:18,playing:false},'seek');
    if(action==='loop-start')J.update({time:J.model.loops.find(l=>l.r===s.rank&&l.n===s.loop).start,playing:false},'seek');
    if(action==='focus-event'){const d=M.dependencies(comparison||J.model,s.event,s.time),prev=(comparison||J.model).graph[d.n.prev];J.update({time:prev?.end??Math.min(...d.deps.map(p=>p.start),d.n.start),playing:false},'seek');}
    if(action==='back')J.update({event:history.pop()},'focus');
  });
  root.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.matches('[data-node],[data-inspect]')){e.preventDefault();e.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));}});
  root.addEventListener('input',e=>{if(e.target.dataset.control==='time')J.update({time:+e.target.value,playing:false},'seek');});
  root.addEventListener('change',async e=>{
    const key=e.target.dataset.control,value=e.target.value;if(!key||key==='time')return;
    if(key==='operation'){sourceStep=value;renderQuick();return;}
    if(key==='compare'){
      const token=++request;comparisonId=value==='current'?null:value;comparison=null;J.setComparisonEnd(original?.end||0);J.update({compare:value,time:0,playing:false},'compare');
      if(comparisonId)try{const next=await get(comparisonId);if(token!==request)return;comparison=next;J.setComparisonEnd(comparisonDuration());render(J.state,'compare');}catch(err){$('.pp-current').textContent='对照场景读取失败，请重选。';}return;
    }
    if(key==='event'){history=[];const n=J.model.graph[value];J.update({event:value,rank:n.r,loop:n.n},'focus');}
    else J.update({[key]:['rank','loop','batch','speed'].includes(key)?+value:value},key==='level'?'level':'observation');
  });
  window.addEventListener('pp-timing-error',()=>{$('.pp-current').textContent='时间模型未能读取，请刷新页面重试。';});
  J.subscribe((s,reason)=>{
    if(packetSeen!==J.model){packetSeen=J.model;history=[];if(!comparisonId)comparison=null;if(dependency)J.setComparisonEnd(comparisonDuration());}
    render(s,reason);
  });
  if(dependency&&comparisonId)get(comparisonId).then(m=>{comparison=m;J.setComparisonEnd(comparisonDuration());}).catch(()=>{comparisonId=null;J.update({compare:'current'},'compare');});
  if(dependency)get('baseline').then(m=>{original=m;J.setComparisonEnd(comparisonDuration());}).catch(()=>$('.pp-current').textContent='基线场景读取失败，请刷新重试。');
  $('.pp-current').textContent='正在读取固定版本的时间模型…';
  if(J.model)render(J.state,'model');
})();
