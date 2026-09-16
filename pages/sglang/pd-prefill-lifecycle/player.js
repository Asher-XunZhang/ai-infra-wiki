(() => {
  'use strict';
  const baseModel=window.PrefillLifecycle, $=id=>document.getElementById(id);
  let model=baseModel.scenario($('player-scenario').value==='single'?'single':'chunked');
  const player=$('pipeline-player'),svg=$('pipeline-svg'),routes=$('pipeline-routes');
  const articles=[...document.querySelectorAll('.life-step')];
  const phaseLinks=[...document.querySelectorAll('.life-nav a')];
  const nodeElements=[...svg.querySelectorAll('[data-node]')];
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  let position=0,playing=false,timer=null,inspected=null;
  const phaseStart=phase=>model.events.findIndex(event=>event.phase===phase);
  const labels={control:'请求 / 控制消息',proxy:'中间激活',output:'结果 t₀',kv:'KV / 交接数据'};
  function element(tag,attrs={}) {
    const el=document.createElementNS('http://www.w3.org/2000/svg',tag);
    Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));return el;
  }
  function routePath(wire) {
    const a=model.nodes[wire.from],b=model.nodes[wire.to];
    const ax=a.x+a.w/2,bx=b.x+b.w/2;
    const ay=a.y+(a.kind==='compute'?82:a.h/2),by=b.y+(b.kind==='compute'?82:b.h/2);
    if(wire.route==='entry')return `M${ax} ${a.y} V270 H200 V${by} H${b.x}`;
    if(wire.route==='return'&&a.kind==='release')return `M${a.x} ${ay} H${a.x-16} V1015 H200 V${by} H${b.x}`;
    if(wire.route==='return')return `M${a.x+a.w} ${ay} H1370 V20 H200 V${by} H${b.x}`;
    if(wire.route==='output-return')return `M${a.x+a.w} ${ay} H1320 V643 H${bx} V${b.y}`;
    if(wire.route==='response')return `M${a.x} ${ay} H200 V${by} H${b.x+b.w}`;
    if(wire.route==='handshake')return `M${a.x} ${ay} H1355 V${90+b.rank*10} H${bx} V${b.y}`;
    if(wire.route==='kv')return `M${a.x+a.w} ${ay} H${a.x+a.w+22} V${930+a.rank*18} H1355 V${by} H${b.x}`;
    if(a.kind==='release'&&b.kind==='release')return `M${a.x} ${ay} H${a.x-16} V1015 H${b.x-16} V${by} H${b.x}`;
    if(wire.kind==='output')return `M${ax} ${a.y} V658 H${bx} V${b.y}`;
    if(a.rank===b.rank&&a.rank!==undefined)return `M${ax} ${a.y+a.h} V${(a.y+a.h+b.y)/2} H${bx} V${b.y}`;
    if(a.x===b.x)return `M${ax} ${a.y+a.h} V${b.y}`;
    return `M${a.x+a.w} ${ay} H${b.x} V${by}`;
  }
  function drawRoutes(event) {
    routes.replaceChildren();
    const defs=element('defs'),marker=element('marker',{id:'flow-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'});
    marker.append(element('path',{d:'M0 0 L10 5 L0 10 Z',fill:'var(--flow-current)'}));defs.append(marker);routes.append(defs);
    for(const wire of event.routes){
      const d=routePath(wire);
      routes.append(element('path',{d,class:'active-route','marker-end':'url(#flow-arrow)'}));
      const packet=element('g',{class:'flow-packet',style:`offset-path:path('${d}');offset-distance:50%`});
      packet.append(element('rect',{x:-18,y:-13,width:36,height:26,rx:5}));
      const label=element('text',{y:5});label.textContent={control:event.id==='terminal-1-wait'?'∅':'R',proxy:'h',output:event.middle?'B':'t₀',kv:'KV'}[wire.kind];
      packet.append(label);routes.append(packet);
    }
  }
  function inspectNode(id,state=model.snapshot(position)) {
    inspected=id;const node=model.nodes[id];let detail=node.role;
    if(node.rank!==undefined){
      const r=state.ranks[node.rank];
      detail+=` 本小步结束后：${model.queueLabels[r.queue]}；${model.kvLabels[r.kv]}；sender ${model.senderLabels[r.sender]}；metadata ${r.metadata?'仍占用':'未占用'}；本地 Req ${r.token?'已记录 t₀':'尚未记录 t₀'}。`;
    }
    if(id==='decode-kv')detail+=` 已有 ${state.ranks.filter(r=>r.sent).length}/3 个级提交发送，${state.ranks.filter(r=>r.terminal).length}/3 个级报告成功。`;
    $('module-description').textContent=`${node.title}：${detail} ${model.events[position].active.includes(id)?'当前小步正在操作此模块。':'此模块不是当前小步的操作对象。'}`;
    $('module-actions').replaceChildren();
    const previous=model.events.map((event,i)=>({event,i})).filter(({event,i})=>i<position&&event.active.includes(id)).pop();
    const next=model.events.findIndex((event,i)=>i>position&&event.active.includes(id));
    for(const [name,index] of [['回看此模块的上一次操作',previous?.i],['跳到此模块的下一次操作',next]]){
      if(index===undefined||index<0)continue;
      const button=document.createElement('button');button.type='button';button.textContent=name;
      button.addEventListener('click',()=>{pause();show(index,true);player.scrollIntoView({block:'start'});});$('module-actions').append(button);
    }
  }
  function show(index,writeHash=false) {
    position=Math.max(0,Math.min(model.events.length-1,index));
    const event=model.events[position],state=model.snapshot(position);
    player.style.setProperty('--flow-current',`var(--flow-${event.kind})`);
    const active=new Set(event.active),visited=new Set(state.visited);
    nodeElements.forEach(node=>{
      const id=node.dataset.node;node.classList.toggle('is-active',active.has(id));node.classList.toggle('is-visited',visited.has(id));
      node.setAttribute('aria-label',`${model.nodes[id].title}，${active.has(id)?'当前操作':visited.has(id)?'已访问':'尚未访问'}，点击查看模块状态`);
    });
    state.ranks.forEach((r,i)=>{
      svg.querySelector(`[data-rank-queue="${i}"]`).textContent=model.queueLabels[r.queue];
      svg.querySelector(`[data-rank-kv="${i}"]`).textContent=r.released?'KV · 请求引用已释放':`KV · 已写 ${r.cacheEnd}/12 · ${r.holdsKv?'引用保留':'尚未分配'}`;
      svg.querySelector(`[data-rank-sender="${i}"]`).textContent=`sender · ${model.senderLabels[r.sender]}`;
      const tank=svg.querySelector(`[data-node="p${i}-transfer"]`);
      tank.classList.toggle('has-reservation',r.kv==='reserved');
      tank.classList.toggle('has-kv',r.kv==='written');
      tank.classList.toggle('is-released',r.released);
      svg.querySelector(`[data-node="p${i}-compute"]`).classList.toggle('has-computed',r.computed);
      svg.querySelector(`[data-received="${i}"]`).classList.toggle('is-received',r.terminal);
      for(let part=0;part<3;part++)svg.querySelector(`[data-kv-chunk="${i}-${part}"]`).classList.toggle('is-filled',r.cacheEnd>part*4&&!r.released);
    });
    drawRoutes(event);
    $('player-counter').textContent=`${String(position+1).padStart(2,'0')} / ${model.events.length}`;
    $('player-seek').value=position;
    $('player-seek').setAttribute('aria-valuetext',`${position+1} / ${model.events.length}：${event.title}`);
    $('event-location').textContent=`${labels[event.kind]} · ${event.active.map(id=>model.nodes[id].title).join(' → ')}`;
    $('player-current-title').textContent=event.title;
    $('scenario-note').textContent=event.chunk?`第 ${event.chunk.index} / ${12/model.size} 块 · 新 token [${event.chunk.start}, ${event.chunk.end})`:`同一条 R · 12 token · ${12/model.size} 个 chunk`;
    $('event-title').textContent=event.title;$('event-description').textContent=event.description;$('event-gate').textContent=event.gate;
    $('event-source').href=`#${articles[event.phase].id}`;$('event-source').textContent=`阅读第 ${event.phase+1} 阶段的源码说明 ↓`;
    articles.forEach((article,i)=>{article.hidden=i!==event.phase;});
    phaseLinks.forEach((link,i)=>{if(i===event.phase)link.setAttribute('aria-current','step');else link.removeAttribute('aria-current');});
    [...$('event-list').children].forEach((item,i)=>{if(i===position)item.firstElementChild.setAttribute('aria-current','step');else item.firstElementChild.removeAttribute('aria-current');});
    for(const id of ['player-prev','life-prev'])$(id).disabled=position===0;
    for(const id of ['player-next','life-next'])$(id).disabled=position===model.events.length-1;
    $('life-progress').textContent=`动作 ${position+1} / ${model.events.length}`;
    if(inspected)inspectNode(inspected,state);
    window.PrefillOperationLab?.update({model,event,state,before:model.snapshot(position-1)});
    if(playing)window.PrefillOperationLab?.play(4200/Number($('player-speed').value));
    if(writeHash)history.replaceState(null,'',`#${model.mode==='chunked'?'chunked-':''}event-${event.id}`);
    if(position===model.events.length-1||!playing)pause();
  }
  function pause() {
    playing=false;clearTimeout(timer);timer=null;player.classList.remove('is-playing');
    window.PrefillOperationLab?.stop();
    $('player-play').textContent=position===model.events.length-1?'↺ 重播':'▶ 播放';$('player-play').setAttribute('aria-pressed','false');
    $('player-mode').textContent=position===model.events.length-1?'P 侧已完成':'已暂停 · 可单步';
  }
  function schedule() {
    clearTimeout(timer);if(!playing)return;
    const duration=4200/Number($('player-speed').value);player.style.setProperty('--flow-duration',`${duration}ms`);
    timer=setTimeout(()=>{if(!playing)return;show(position+1,true);if(playing)schedule();},duration);
  }
  function play() {
    if(playing){pause();return;}if(position===model.events.length-1)show(0,true);
    playing=true;player.classList.add('is-playing');$('player-play').textContent='Ⅱ 暂停';$('player-play').setAttribute('aria-pressed','true');
    $('player-mode').textContent='播放中 · 可随时暂停';schedule();window.PrefillOperationLab?.play(4200/Number($('player-speed').value));
  }
  function advance(delta){pause();show(position+delta,true);}
  function fromHash(){
    const mode=location.hash.startsWith('#chunked-event-')?'chunked':location.hash.startsWith('#event-')?'single':model.mode;
    if(mode!==model.mode){model=baseModel.scenario(mode);$('player-scenario').value=mode;populateEvents();}
    const hash=location.hash.replace('#chunked-event-','#event-');
    let index=model.events.findIndex(event=>`#event-${event.id}`===hash);
    if(index<0){const phase=articles.findIndex(article=>`#${article.id}`===location.hash);if(phase>=0)index=phaseStart(phase);}
    if(index>=0){pause();show(index);player.scrollIntoView({block:'start'});}
  }
  function populateEvents(){
  $('event-list').replaceChildren();
  model.events.forEach((event,index)=>{
    const li=document.createElement('li'),button=document.createElement('button');button.type='button';button.textContent=event.title;
    button.addEventListener('click',()=>{pause();show(index,true);player.scrollIntoView({block:'start'});});li.append(button);$('event-list').append(li);
  });
  $('player-seek').max=model.events.length-1;$('player-seek').disabled=false;
  }
  populateEvents();
  $('player-scenario').addEventListener('change',event=>{pause();model=baseModel.scenario(event.target.value);window.PrefillOperationLab?.follow();populateEvents();show(0,true);});
  document.querySelector('.player-toolbar').hidden=false;
  document.querySelector('.life-walkthrough').classList.add('is-interactive');document.querySelector('.life-controls').hidden=false;
  $('life-prev').textContent='← 上一小步';$('life-next').textContent='下一小步 →';
  $('player-play').addEventListener('click',play);
  $('player-reset').addEventListener('click',()=>{pause();show(0,true);});
  ['player-prev','life-prev'].forEach(id=>$(id).addEventListener('click',()=>advance(-1)));
  ['player-next','life-next'].forEach(id=>$(id).addEventListener('click',()=>advance(1)));
  $('player-seek').addEventListener('input',event=>{pause();show(Number(event.target.value),true);});
  $('player-speed').addEventListener('change',schedule);
  $('player-zoom').addEventListener('change',event=>{svg.style.width=event.target.value==='fit'?'100%':`${1680*Number(event.target.value)/100}px`;});
  $('event-source').addEventListener('click',event=>{event.preventDefault();pause();const article=articles[model.events[position].phase];article.scrollIntoView({block:'start'});article.querySelector('h3').focus({preventScroll:true});});
  phaseLinks.forEach((link,phase)=>{
    link.addEventListener('click',event=>{if(event.metaKey||event.ctrlKey||event.shiftKey||event.altKey)return;event.preventDefault();pause();show(phaseStart(phase),true);player.scrollIntoView({block:'start'});});
    link.addEventListener('keydown',event=>{
      const moves={ArrowDown:phase+1,ArrowRight:phase+1,ArrowUp:phase-1,ArrowLeft:phase-1,Home:0,End:11};
      if(!(event.key in moves))return;event.preventDefault();const target=Math.max(0,Math.min(11,moves[event.key]));pause();show(phaseStart(target),true);phaseLinks[target].focus();
    });
  });
  nodeElements.forEach(node=>{
    const inspect=()=>{pause();inspectNode(node.dataset.node);document.querySelector('.module-inspector').open=true;window.PrefillOperationLab?.inspect(node.dataset.node);};
    node.addEventListener('click',inspect);node.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();event.stopPropagation();inspect();}});
  });
  player.addEventListener('keydown',event=>{
    if(event.target.closest('input,select,button,a,summary,[data-node]'))return;
    if(event.key==='ArrowLeft'){event.preventDefault();advance(-1);}if(event.key==='ArrowRight'){event.preventDefault();advance(1);}if(event.key===' '){event.preventDefault();play();}
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});window.addEventListener('pagehide',pause);window.addEventListener('hashchange',fromHash);
  window.addEventListener('prefill-lab-play',pause);
  const motion=()=>player.classList.toggle('reduce-motion',reduced.matches);reduced.addEventListener('change',motion);motion();
  show(0);pause();fromHash();
})();
