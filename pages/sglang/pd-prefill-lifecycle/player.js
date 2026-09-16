(() => {
  'use strict';
  const api=window.PrefillLifecycle,$=id=>document.getElementById(id),player=$('pipeline-player'),svg=$('pipeline-svg');
  const fields=['requests','inputLength','chunkSize','batchSize','pageSize','fault','waitRank','waitRid','detail'];
  const articles=[...document.querySelectorAll('.life-step')],phaseLinks=[...document.querySelectorAll('.life-nav a')];
  let model,eventIndex=0,frameIndex=0,playing=false,onlyGroup=false,timer=null,draftFailures=api.defaults.failed.map(x=>[...x]);
  const activeFrame=()=>model.events[eventIndex].frames[frameIndex];
  const activeGroup=()=>model.events[eventIndex];
  function svgEl(tag,attributes){const el=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attributes))el.setAttribute(k,v);return el;}
  function path(wire){
    const a=model.nodes[wire.from],b=model.nodes[wire.to],ax=a.x+a.w/2,bx=b.x+b.w/2,ay=a.y+a.h/2,by=b.y+b.h/2;
    if(wire.path==='entry')return `M${ax} ${a.y} V102 H155 V${by} H${b.x}`;
    if(wire.path==='handshake')return `M${a.x} ${ay} H1024 V${96+b.rank*3} H${bx} V${b.y+b.h}`;
    if(wire.path==='return'&&a.kind==='bootstrap')return `M${a.x+a.w} ${ay} H1014 V102 H155 V${by} H${b.x}`;
    if(a.kind==='release')return `M${a.x} ${ay} H${a.x-6} V464 H${b.x-6} V${by} H${b.x}`;
    if(wire.path==='output-return')return `M${a.x+a.w} ${ay} H1008 V279 H${bx} V${b.y}`;
    if(wire.kind==='output')return `M${ax} ${a.y} V292 H${bx} V${b.y}`;
    if(wire.path==='kv')return `M${a.x+a.w} ${ay} H${a.x+a.w+6} V${407+a.rank*16} H${1000+a.rank*16} V${325+a.rank*15} H${b.x-2}`;
    return `M${a.x+a.w} ${ay} H${b.x} V${by}`;
  }
  function draw(f){
    player.style.setProperty('--flow-current',`var(--flow-${f.kind})`);
    for(const el of svg.querySelectorAll('[data-node]')){const id=el.dataset.node;el.classList.toggle('is-active',f.active.includes(id));el.classList.toggle('is-visited',f.after.visited.includes(id));el.setAttribute('aria-label',`${model.nodes[id].title}，${f.active.includes(id)?'当前操作':'查看职责'}`);}
    const routes=$('pipeline-routes');routes.replaceChildren();
    const defs=svgEl('defs',{}),marker=svgEl('marker',{id:'flow-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto'});marker.append(svgEl('path',{d:'M0 0 L10 5 L0 10 Z',fill:'var(--flow-current)'}));defs.append(marker);routes.append(defs);
    for(const wire of f.routes){const d=path(wire);routes.append(svgEl('path',{class:'active-route',d,'marker-end':'url(#flow-arrow)'}));const p=svgEl('circle',{r:4,class:'flow-packet',style:`offset-path:path('${d}');offset-distance:50%`});routes.append(p);}
  }
  function show(write=true){
    const g=activeGroup(),f=activeFrame();
    $('player-counter').textContent=`${eventIndex+1} / ${model.events.length}`;$('player-seek').value=eventIndex;
    $('player-current-title').textContent=g.title;$('group-label').textContent=g.compressed?'后续块 · 一段完整流程':'详细动作';
    $('frame-controls').hidden=!g.compressed;$('frame-counter').textContent=`${frameIndex+1} / ${g.frames.length}`;$('frame-seek').max=g.frames.length-1;$('frame-seek').value=frameIndex;
    $('frame-prev').disabled=frameIndex===0;$('frame-next').disabled=frameIndex===g.frames.length-1;
    $('player-prev').disabled=eventIndex===0;$('player-next').disabled=eventIndex===model.events.length-1;
    $('player-seek').setAttribute('aria-valuetext',`${eventIndex+1}：${g.title}`);
    window.PrefillOperationLab.render(f,model);draw(f);$('module-description').textContent='当前模块：'+f.active.map(id=>model.nodes[id].title).join(' · ')+'。点击组件可查看职责。';
    articles.forEach((a,i)=>a.hidden=i!==f.phase);phaseLinks.forEach((a,i)=>{if(i===f.phase)a.setAttribute('aria-current','step');else a.removeAttribute('aria-current');});
    [...$('event-list').children].forEach((li,i)=>li.firstElementChild.setAttribute('aria-current',i===eventIndex?'step':'false'));
    if(write)history.replaceState(null,'',scenarioUrl(f.key));
    if(!playing){$('player-play').textContent=eventIndex===model.events.length-1?'↺ 重播':'▶ 播放全程';$('player-mode').textContent='已暂停 · 可单步';}
  }
  function pause(){playing=false;onlyGroup=false;clearTimeout(timer);timer=null;player.classList.remove('is-playing');$('player-play').textContent='▶ 播放全程';$('player-play').setAttribute('aria-pressed','false');$('play-group').textContent='▶ 播放这一整段';$('player-mode').textContent='已暂停 · 可单步';}
  function schedule(){clearTimeout(timer);if(!playing)return;const duration=(activeGroup().compressed?650:1800)/Number($('player-speed').value);player.style.setProperty('--flow-duration',`${duration}ms`);timer=setTimeout(()=>{
    if(frameIndex+1<activeGroup().frames.length)frameIndex++;
    else if(onlyGroup||eventIndex+1===model.events.length){pause();show();return;}
    else{eventIndex++;frameIndex=0;}
    show();schedule();
  },duration);}
  function play(groupOnly=false){if(playing){pause();show();return;}if(!groupOnly&&eventIndex===model.events.length-1){eventIndex=0;frameIndex=0;show();}if(groupOnly&&frameIndex===activeGroup().frames.length-1)frameIndex=0;playing=true;onlyGroup=groupOnly;player.classList.add('is-playing');$('player-play').textContent='Ⅱ 暂停';$('player-play').setAttribute('aria-pressed','true');$('play-group').textContent='Ⅱ 暂停';$('player-mode').textContent=groupOnly?'播放当前整段':'播放全程';show();schedule();}
  function jump(e,f=0){pause();eventIndex=Math.max(0,Math.min(model.events.length-1,e));frameIndex=Math.max(0,Math.min(activeGroup().frames.length-1,f));show();}
  function populate(){
    $('event-list').replaceChildren();model.events.forEach((g,i)=>{const li=document.createElement('li'),button=document.createElement('button');button.type='button';button.textContent=g.title;button.addEventListener('click',()=>{jump(i);$('operation-lab').scrollIntoView({block:'start'});});li.append(button);$('event-list').append(li);});$('player-seek').max=model.events.length-1;
    $('scenario-summary').textContent=`${model.config.requests} 条请求 · ${model.batches} 个 batch · ${model.events.length} 个阅读步骤`;
  }
  function collectFailures(){return [0,1,2].map(r=>[...document.querySelectorAll(`[data-fail-rank="${r}"]:checked`)].map(x=>x.value));}
  function failureFields(){
    const count=Math.max(1,Math.min(4,Number($('config-requests').value)||1)),fault=$('config-fault').value;
    $('failure-field').hidden=!['bootstrap_fail','bootstrap_abort','transfer_fail'].includes(fault);
    $('wait-field').hidden=!['bootstrap_wait','transfer_fail','repoll_wait'].includes(fault);
    $('failure-matrix').replaceChildren();
    for(let r=0;r<3;r++){const row=document.createElement('div'),title=document.createElement('strong');title.textContent=`PP${r}`;row.append(title);for(let i=0;i<count;i++){const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.dataset.failRank=r;input.value=`R${i}`;input.checked=draftFailures[r]?.includes(input.value);input.addEventListener('change',()=>draftFailures=collectFailures());label.append(input,document.createTextNode(input.value));row.append(label);}$('failure-matrix').append(row);}
    const rid=$('config-waitRid').value;$('config-waitRid').replaceChildren();for(let i=0;i<count;i++){const option=document.createElement('option');option.value=option.textContent=`R${i}`;$('config-waitRid').append(option);}$('config-waitRid').value=Number(rid.slice(1))<count?rid:'R0';
  }
  function fill(config){for(const field of fields)if(field!=='waitRid')$(`config-${field}`).value=config[field];draftFailures=config.failed.map(a=>[...a]);failureFields();$('config-waitRid').value=config.waitRid;}
  function readForm(){return {...Object.fromEntries(fields.map(k=>[k,$(`config-${k}`).value])),failed:collectFailures()};}
  function scenarioUrl(key){const url=new URL(location.href);url.search='';for(const k of fields)url.searchParams.set(k,model.config[k]);url.searchParams.set('failed',JSON.stringify(model.config.failed));url.hash=`action=${encodeURIComponent(key)}`;return url.pathname+url.search+url.hash;}
  function apply(raw,write=true){try{const next=api.createScenario(raw);pause();model=next;eventIndex=frameIndex=0;populate();$('config-error').hidden=true;$('config-status').textContent=`已生成 ${model.batches} 个 batch。`;show(write);return true;}catch(e){$('config-error').hidden=false;$('config-error').textContent=e.message;return false;}}
  function fromHash(){
    let key;try{key=decodeURIComponent(location.hash.replace(/^#action=/,''));}catch{return;}
    const legacy={'#event-intake':'intake','#event-proxy-01':'B1-proxy-0-1','#event-batch-0':'B1-pack','#chunked-event-chunk-2-cut':'B2-cut','#event-handoff':'finish'};key=legacy[location.hash]??key;
    for(let e=0;e<model.events.length;e++){const f=model.events[e].frames.findIndex(x=>x.key===key);if(f>=0){jump(e,f);return;}}
    const phase=articles.findIndex(a=>'#'+a.id===location.hash);if(phase>=0){const e=model.events.findIndex(g=>g.frames.some(f=>f.phase===phase));if(e>=0)jump(e,model.events[e].frames.findIndex(f=>f.phase===phase));}
  }
  $('scenario-form').addEventListener('submit',e=>{e.preventDefault();if(apply(readForm()))$('config-preset').value='custom';});
  $('config-preset').addEventListener('change',e=>{if(!api.presets[e.target.value])return;const config=api.normalize(api.presets[e.target.value]);fill(config);apply(config);});
  $('config-requests').addEventListener('change',()=>{draftFailures=collectFailures();failureFields();});$('config-fault').addEventListener('change',failureFields);
  $('player-reset').addEventListener('click',()=>jump(0));$('player-prev').addEventListener('click',()=>jump(eventIndex-1));$('player-next').addEventListener('click',()=>jump(eventIndex+1));$('player-play').addEventListener('click',()=>play());
  $('player-seek').addEventListener('input',e=>jump(Number(e.target.value)));$('frame-prev').addEventListener('click',()=>jump(eventIndex,frameIndex-1));$('frame-next').addEventListener('click',()=>jump(eventIndex,frameIndex+1));$('frame-seek').addEventListener('input',e=>jump(eventIndex,Number(e.target.value)));$('play-group').addEventListener('click',()=>play(true));$('player-speed').addEventListener('change',schedule);
  $('share-scenario').addEventListener('click',async()=>{const url=new URL(scenarioUrl(activeFrame().key),location.href).href;try{await navigator.clipboard.writeText(url);$('config-status').textContent='已复制当前参数和动作位置。';}catch{$('config-status').textContent='参数已保存在地址栏，可复制当前网址。';}});
  phaseLinks.forEach((a,phase)=>a.addEventListener('click',e=>{if(e.ctrlKey||e.metaKey)return;e.preventDefault();const index=model.events.findIndex(g=>g.frames.some(f=>f.phase===phase));if(index>=0){jump(index,model.events[index].frames.findIndex(f=>f.phase===phase));$('operation-lab').scrollIntoView({block:'start'});}}));
  for(const el of svg.querySelectorAll('[data-node]')){const inspect=()=>{pause();const id=el.dataset.node;$('module-description').textContent=`${model.nodes[id].title}：${model.nodes[id].role}`;const flat=model.frames.indexOf(activeFrame()),next=model.frames.findIndex((f,i)=>i>flat&&f.active.includes(id));if(next>=0){const button=document.createElement('button');button.textContent='跳到该组件的下一次操作';button.type='button';button.addEventListener('click',()=>{const target=model.frames[next];for(let e=0;e<model.events.length;e++){const f=model.events[e].frames.indexOf(target);if(f>=0){jump(e,f);$('operation-lab').scrollIntoView({block:'start'});break;}}});$('module-description').append(button);}};el.addEventListener('click',inspect);el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();inspect();}});}
  player.addEventListener('keydown',e=>{if(e.target.closest('input,select,button,a,summary,[data-node]'))return;if(e.key==='ArrowRight'){e.preventDefault();jump(eventIndex+1);}if(e.key==='ArrowLeft'){e.preventDefault();jump(eventIndex-1);}if(e.key===' '){e.preventDefault();play();}});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});window.addEventListener('pagehide',pause);window.addEventListener('hashchange',fromHash);
  document.querySelector('.life-walkthrough').classList.add('is-interactive');
  const params=new URLSearchParams(location.search),raw={};for(const k of fields)if(params.has(k))raw[k]=params.get(k);let initial,invalidLink=false;
  try{if(params.has('failed'))raw.failed=JSON.parse(params.get('failed'));initial=api.normalize(raw);}catch(e){initial=api.normalize();invalidLink=true;}
  fill(initial);apply(initial,false);$('config-preset').value=Object.keys(api.presets).find(k=>{const p=api.normalize(api.presets[k]);return fields.every(f=>p[f]===initial[f])&&JSON.stringify(p.failed)===JSON.stringify(initial.failed);})??'custom';fromHash();if(invalidLink)$('config-status').textContent='链接参数无效，已使用默认示例。';
})();
