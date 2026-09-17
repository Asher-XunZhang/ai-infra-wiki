(() => {
  'use strict';
  const Q=window.PDQueues,$=id=>document.getElementById(id);
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let model,index=0,timer=null,selected='bootstrap',animation=null;
  const labels={bootstrap:'bootstrap 列表',waiting:'waiting_queue',inflight:'inflight 列表',batch:'batch.reqs',chunk:'chunked_req',exit:'教学出口'};
  $('q-scenario').innerHTML=Q.scenarios.map(s=>`<option value="${s.id}">${s.label}</option>`).join('');
  $('q-nodes').innerHTML=Q.nodes.map(n=>`<button type="button" data-q-node="${n.id}" class="q-node ${['batch','chunk'].includes(n.id)?'q-reference':''}"><span class="q-type">${n.kind}</span><code>${n.name}</code><span class="q-slot" id="q-slot-${n.id}"></span><span class="q-count" id="q-count-${n.id}"></span></button>`).join('');
  function pause(){clearInterval(timer);timer=null;$('q-play').textContent='播放入队 / 出队';$('q-play').setAttribute('aria-pressed','false');}
  function stopMotion(){if(animation){animation.cancel();animation=null;}$('q-traveler').hidden=true;}
  function load(id,rank){pause();stopMotion();model=Q.create(id,rank);index=0;$('q-scenario').value=model.id;$('q-rank').value=model.rank;$('q-steps').innerHTML=model.frames.map((f,i)=>`<option value="${i}">${i+1} · ${esc(f.title)}</option>`).join('');render(false);}
  function detail(){const n=Q.nodes.find(x=>x.id===selected);$('q-node-title').textContent=n.name;$('q-node-detail').textContent=n.note;$('q-node-source').href=Q.sourceUrl(n.source);$('q-nodes').querySelectorAll('[data-q-node]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.qNode===selected)));}
  function move(f){
    if(!f.from||!f.to||matchMedia('(prefers-reduced-motion: reduce)').matches)return;
    const board=$('q-map').getBoundingClientRect(),a=$(`q-slot-${f.from}`).getBoundingClientRect(),b=$(`q-slot-${f.to}`).getBoundingClientRect(),traveler=$('q-traveler');
    const x=a.left+a.width/2-board.left-20,y=a.top+a.height/2-board.top-13,dx=b.left+b.width/2-a.left-a.width/2,dy=b.top+b.height/2-a.top-a.height/2;
    traveler.hidden=false;traveler.style.left=`${x}px`;traveler.style.top=`${y}px`;traveler.classList.toggle('q-ref-flight',f.kind==='reference');
    animation=traveler.animate(f.from===f.to?[{transform:'scale(1)'},{transform:'scale(1.3)'},{transform:'scale(1)'}]:[{transform:'translate(0,0)',opacity:1},{transform:`translate(${dx}px,${dy}px)`,opacity:1}],{duration:1100,easing:'ease-in-out'});
    animation.onfinish=()=>{traveler.hidden=true;animation=null;};
  }
  function render(animate=true){
    stopMotion();const f=model.frames[index],s=f.after;
    $('q-position').textContent=`PP${model.rank} · ${index+1} / ${model.frames.length} · ${s.batchName}`;$('q-steps').value=index;$('q-progress').max=model.frames.length-1;$('q-progress').value=index;
    $('q-prev').disabled=index===0;$('q-next').disabled=index===model.frames.length-1;
    $('q-title').textContent=f.title;$('q-description').textContent=f.detail;$('q-code').textContent=f.code;$('q-source').href=Q.sourceUrl(f.source);
    $('q-operation').textContent=f.from?`${labels[f.from]} ${f.from===f.to?'↺ 保留':f.kind==='reference'?'⇢ 引用关系':'→'} ${f.from===f.to?'':labels[f.to]}`:'本地操作 · 观察成员与字段';
    Q.nodes.forEach(n=>{const active=s[n.id],changed=active!==f.before[n.id];$(`q-slot-${n.id}`).innerHTML=active?`<span class="q-chip">R0${n.id==='chunk'?' ↗':''}</span>`:'<span class="q-empty">∅</span>';$(`q-count-${n.id}`).textContent=n.id==='exit'?(active?s.finished:'未收尾'):n.id==='chunk'?(active?'指向 R0':'None'):`${active?1:0} 个 Req 引用`;
      const el=$(`q-slot-${n.id}`).closest('button');el.classList.toggle('q-occupied',active);el.classList.toggle('q-changed',changed);el.classList.toggle('q-blocked',f.kind==='wait'&&f.to===n.id);});
    $('q-fields').innerHTML=[['最近 poll',s.poll],['pending_bootstrap',String(s.pending)],['metadata 槽',s.metadata?'持有':'无'],['请求 KV 所有权',s.kv?'持有':'无'],['finished',s.finished],['pending_chunk_rids',s.pendingIds?'{R0}':'∅']].map(([k,v])=>`<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
    $('q-announcement').textContent=`PP${model.rank}，${f.title}。${f.detail}`;detail();if(animate)move(f);
  }
  function go(i){pause();index=Math.max(0,Math.min(model.frames.length-1,i));render();}
  $('q-prev').onclick=()=>go(index-1);$('q-next').onclick=()=>go(index+1);$('q-restart').onclick=()=>go(0);$('q-replay').onclick=()=>{pause();render();};
  $('q-steps').onchange=e=>go(Number(e.target.value));$('q-progress').oninput=e=>go(Number(e.target.value));
  $('q-play').onclick=()=>{if(timer){pause();return;}if(index===model.frames.length-1){index=0;render();}$('q-play').textContent='暂停';$('q-play').setAttribute('aria-pressed','true');timer=setInterval(()=>{index++;render();if(index===model.frames.length-1)pause();},3600);};
  $('q-scenario').onchange=e=>{$('q-follow').checked=false;load(e.target.value,$('q-rank').value);};
  $('q-rank').onchange=e=>load(model.id,e.target.value);
  $('q-follow').onchange=e=>{if(e.target.checked)load($('scenario').value,$('q-rank').value);};
  $('scenario').addEventListener('change',e=>{if($('q-follow').checked)load(e.target.value,$('q-rank').value);});
  window.addEventListener('popstate',()=>{if($('q-follow').checked)load($('scenario').value,$('q-rank').value);});
  $('q-nodes').onclick=e=>{const b=e.target.closest('[data-q-node]');if(b){selected=b.dataset.qNode;detail();}};
  // Only one walkthrough plays at a time; changing the main scenario also pauses this panel.
  $('q-play').addEventListener('click',()=>{if(timer&&$('play').getAttribute('aria-pressed')==='true')$('play').click();});
  $('play').addEventListener('click',()=>{if($('play').getAttribute('aria-pressed')==='true')pause();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden){pause();stopMotion();}});window.addEventListener('pagehide',()=>{pause();stopMotion();});window.addEventListener('resize',stopMotion);
  load($('scenario').value,0);
})();
