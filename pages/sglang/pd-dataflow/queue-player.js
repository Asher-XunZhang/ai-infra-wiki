(() => {
  'use strict';
  const Q=window.PDQueues,$=id=>document.getElementById(id);
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let model,index=0,timer=null,selected='bootstrap';
  const labels={bootstrap:'bootstrap 列表',waiting:'waiting_queue',inflight:'inflight 列表',batch:'batch.reqs',chunk:'chunked_req',exit:'教学出口'};
  $('q-scenario').innerHTML=Q.scenarios.map(s=>`<option value="${s.id}">${s.label}</option>`).join('');
  $('q-nodes').innerHTML=Q.nodes.map(n=>`<button type="button" data-q-node="${n.id}" class="q-node ${['batch','chunk'].includes(n.id)?'q-reference':''}"><code>${n.name}</code><span class="q-slot" id="q-slot-${n.id}"></span><span class="q-count" id="q-count-${n.id}"></span></button>`).join('');
  function pause(){clearInterval(timer);timer=null;$('q-play').textContent='播放入队 / 出队';$('q-play').setAttribute('aria-pressed','false');}
  function stopMotion(){for(const motion of $('q-reference-lines').getAnimations({subtree:true}))motion.cancel();}
  function load(id,rank){pause();stopMotion();model=Q.create(id,rank);index=0;$('q-scenario').value=model.id;$('q-rank').value=model.rank;$('q-steps').innerHTML=model.frames.map((f,i)=>`<option value="${i}">${i+1} · ${esc(f.title)}</option>`).join('');render(false);}
  function detail(){const n=Q.nodes.find(x=>x.id===selected);$('q-node-title').textContent=n.name;$('q-node-detail').textContent=n.note;$('q-node-source').href=Q.sourceUrl(n.source);$('q-nodes').querySelectorAll('[data-q-node]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.qNode===selected)));}
  function drawReferences(animate=false){
    const f=model.frames[index],state=f.after,svg=$('q-reference-lines'),board=$('q-map').getBoundingClientRect(),obj=$('q-request').getBoundingClientRect();
    svg.setAttribute('viewBox',`0 0 ${board.width} ${board.height}`);svg.replaceChildren();
    for(const n of Q.nodes){if(!state[n.id]||n.id==='exit')continue;const box=$(`q-slot-${n.id}`).closest('button').getBoundingClientRect(),x=box.right-board.left,y=box.top+box.height/2-board.top,tx=obj.left-board.left,ty=obj.top+obj.height/2-board.top;
      const p=document.createElementNS('http://www.w3.org/2000/svg','path');p.setAttribute('d',`M ${x} ${y} C ${tx-16} ${y},${tx-16} ${ty},${tx-3} ${ty}`);p.dataset.reference=n.id;svg.append(p);
      if(animate&&state[n.id]!==f.before[n.id]&&!matchMedia('(prefers-reduced-motion: reduce)').matches){const len=p.getTotalLength();p.animate([{strokeDasharray:len,strokeDashoffset:len},{strokeDasharray:len,strokeDashoffset:0}],{duration:800,easing:'ease-out'});}
    }
  }
  function render(animate=true){
    stopMotion();const f=model.frames[index],s=f.after;
    $('q-position').textContent=`PP${model.rank} · ${index+1} / ${model.frames.length} · ${s.batchName}`;$('q-steps').value=index;$('q-progress').max=model.frames.length-1;$('q-progress').value=index;
    $('q-prev').disabled=index===0;$('q-next').disabled=index===model.frames.length-1;
    $('q-title').textContent=f.title;$('q-description').textContent=f.detail;$('q-code').textContent=f.code;$('q-source').href=Q.sourceUrl(f.source);
    $('q-operation').textContent=f.from?`${labels[f.from]} ${f.from===f.to?'↺ 保留':f.kind==='reference'?'⇢ 引用关系':'→'} ${f.from===f.to?'':labels[f.to]}`:'本地操作 · 观察成员与字段';
    Q.nodes.forEach(n=>{const active=s[n.id],changed=active!==f.before[n.id];$(`q-slot-${n.id}`).innerHTML=active?'<span class="q-reference-label">'+(n.id==='exit'?'已收尾':'引用 →')+'</span>':'<span class="q-empty">∅</span>';$(`q-count-${n.id}`).textContent=n.id==='exit'?(active?s.finished:'未收尾'):n.id==='chunk'?(active?'持有引用':'None'):`${active?1:0} 个 Req 引用`;
      const el=$(`q-slot-${n.id}`).closest('button');el.classList.toggle('q-occupied',active);el.classList.toggle('q-changed',changed);el.classList.toggle('q-blocked',f.kind==='wait'&&f.to===n.id);});
    $('q-fields').innerHTML=[['最近 poll',s.poll],['pending_bootstrap',String(s.pending)],['metadata 槽',s.metadata?'持有':'无'],['请求 KV 所有权',s.kv?'持有':'无'],['finished',s.finished],['pending_chunk_rids',s.pendingIds?'{R0}':'∅']].map(([k,v])=>`<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
    $('q-request-rank').textContent=`PP${model.rank} 本地对象`;
    $('q-reference-note').textContent=Q.nodes.filter(n=>n.id!=='exit'&&s[n.id]).map(n=>n.name).join('、')||'图中容器暂无引用；不据此断言对象已被回收';
    $('q-announcement').textContent=`PP${model.rank}，${f.title}。${f.detail}`;detail();drawReferences(animate);
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
  document.addEventListener('visibilitychange',()=>{if(document.hidden){pause();stopMotion();}});window.addEventListener('pagehide',()=>{pause();stopMotion();});new ResizeObserver(()=>{if(model)drawReferences();}).observe($('q-map'));
  load($('scenario').value,0);
})();
