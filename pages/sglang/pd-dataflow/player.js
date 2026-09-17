(() => {
  'use strict';
  const M=window.PDDataflow, $=id=>document.getElementById(id);
  const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fields={queue:'所在阶段',poll:'最近 poll',metadata:'metadata 槽',kv:'请求 KV 所有权',computed:'已计算边界',sent:'已提交边界',token:'有效首 token',pending:'pending_bootstrap'};
  const format=(key,value)=>key==='queue'?M.queues[value]:typeof value==='boolean'?(key==='pending'?String(value):value?'有':'无'):['computed','sent'].includes(key)?`[0, ${value})`:value;
  const names={control:'控制 / 共识',activation:'激活',output:'结果',kv:'KV / metadata'};
  const defaults={control:['PP0 → PP1 → PP2','PP2 → PP0 → PP1 → PP2'],activation:['PP0 → PP1 → PP2','hidden_states / residual'],output:['PP2 → PP0 → PP1 → PP2','batch 对应的结果'],kv:['各 PP → Decode','本级层的 KV；最终块携带 metadata']};
  let model, index=0, rank=0, timer=null;
  $('scenario').innerHTML=M.scenarios.map(s=>`<option value="${s.id}">${s.label}</option>`).join('');
  function pause(){clearInterval(timer);timer=null;$('play').textContent='播放';$('play').setAttribute('aria-pressed','false');}
  function writeUrl(){const url=new URL(location.href);url.searchParams.set('scenario',model.id);url.searchParams.set('step',model.frames[index].key);url.searchParams.set('rank',rank);history.replaceState(null,'',url);}
  function load(id,step){pause();model=M.create(id);index=Math.max(0,model.frames.findIndex(f=>f.key===step));$('scenario').value=model.id;$('scenario-description').textContent=M.scenarios.find(s=>s.id===model.id).description;
    $('milestone').innerHTML=model.frames.map((f,i)=>`<option value="${i}">${String(i+1).padStart(2,'0')} · ${escape(f.title)}</option>`).join('');$('progress').max=model.frames.length-1;render();}
  function render(){
    const f=model.frames[index],s=f.after,old=f.before,q=s.ranks[rank],prev=old.ranks[rank];
    $('position').textContent=`${index+1} / ${model.frames.length} · ${s.batch}`;$('progress').value=index;$('milestone').value=index;
    $('prev').disabled=index===0;$('next').disabled=index===model.frames.length-1;
    $('event-title').textContent=f.title;$('event-detail').textContent=f.detail;$('event-kind').textContent=f.flow?names[f.flow.kind]:'本地动作 / 状态检查';
    $('source').href=M.sourceUrl(f.source);$('source').title=M.sources[f.source][0]+':'+M.sources[f.source][1];
    $('gate').hidden=!f.gate;$('gate').textContent=f.gate?`暂不能前进 · ${f.gate}`:'';
    $('rank-cards').innerHTML=s.ranks.map((row,r)=>`<article class="rank-card ${r===rank?'selected':''}"><button type="button" data-rank="${r}" aria-pressed="${r===rank}">PP${r}<small>${r===rank?'正在观察':'查看状态'}</small></button><p class="queue">${M.queues[row.queue]}</p><dl>${['poll','kv','metadata','pending','token'].map(key=>`<div><dt>${fields[key]}</dt><dd class="${row[key]!==old.ranks[r][key]?'changed':''}">${escape(format(key,row[key]))}</dd></div>`).join('')}</dl></article>`).join('');
    $('routes').innerHTML=Object.entries(defaults).map(([kind,info])=>{const active=f.flow?.kind===kind;return `<div class="flow-route ${active?'active':''}"><strong>${names[kind]}</strong><div class="route-track"><span>${escape(active?f.flow.from:info[0])}</span>${active?`<span>→ ${escape(f.flow.to)}</span>`:''}</div><span class="route-payload">${escape(active?f.flow.payload:info[1])}</span></div>`;}).join('');
    const set=a=>a.length?`{${a.join(', ')}}`:'∅';
    $('bootstrap-sets').textContent=`good ${set(s.good)} · bad ${set(s.bad)}`;$('terminal-set').textContent=set(s.release);
    $('rank').value=rank;$('inspector-title').textContent=`PP${rank} · 状态与持有关系`;
    const node=key=>{const available=model.frames.some(frame=>frame.after.ranks[rank].queue===key);return `<button type="button" data-state="${key}" class="state-node ${q.queue===key?'current':prev.queue===key&&prev.queue!==q.queue?'previous':''}" aria-pressed="${q.queue===key}" ${available?'':'disabled'}>${M.queues[key]}${q.queue===key?' · 当前':''}</button>`;};
    $('state-path').innerHTML=`<div class="state-main">${['bootstrap','waiting','batch','inflight','done'].map((key,i)=>`${i?'<span class="path-arrow" aria-hidden="true">→</span>':''}${node(key)}`).join('')}</div><div class="state-branch"><span>本地 batch</span><span>→ 中间块 →</span>${node('chunked')}<span>↩ 下一块回到 batch</span></div><div class="state-branch"><span>握手失败 / 取消，或在途传输失败</span><span>→</span>${node('aborted')}</div>`;
    const changes=Object.keys(fields).filter(key=>q[key]!==prev[key]);
    $('changes').innerHTML=changes.length?changes.map(key=>`<li>${fields[key]}：${escape(format(key,prev[key]))} → <strong>${escape(format(key,q[key]))}</strong></li>`).join(''):'<li>本步未改变本级请求字段；可能正在传递数据、汇总名单，或推进其它级。</li>';
    $('rank-hint').textContent=q.queue==='aborted'?'本级以中止收尾。其它级的局部 Success 不能证明端到端请求成功。':q.queue==='done'?'本地请求已收尾；图中仍保留历史 poll 和 token 区间供对照。':q.pending?'先等待有效共识与本地 finalize，不能直接进入计算。':q.queue==='waiting'?'已握手就绪；选批仍受 token 预算、batch 容量等约束。':q.kv?'本地请求仍持有 KV 所有权；不能按“已提交”的长度直接释放。':'';
    $('tokens').innerHTML=['computed','sent'].map(key=>`<div class="token-row"><span>${fields[key]} [0, ${q[key]})</span><div class="token-cells" role="img" aria-label="${fields[key]} [0, ${q[key]})，上一帧 [0, ${prev[key]})">${Array.from({length:12},(_,i)=>`<i aria-hidden="true" class="${i<q[key]?'filled':''} ${i>=prev[key]&&i<q[key]?'delta':''}">${i}</i>`).join('')}</div></div>`).join('');
    $('announcement').textContent=`步骤 ${index+1}，${f.title}${f.gate?'，等待：'+f.gate:''}`;writeUrl();
  }
  function go(next){pause();index=Math.max(0,Math.min(model.frames.length-1,next));render();}
  $('prev').onclick=()=>go(index-1);$('next').onclick=()=>go(index+1);$('restart').onclick=()=>go(0);
  $('progress').oninput=e=>go(Number(e.target.value));$('milestone').onchange=e=>go(Number(e.target.value));
  $('scenario').onchange=e=>load(e.target.value);
  $('rank').onchange=e=>{rank=Number(e.target.value);render();};
  $('state-path').onclick=e=>{const b=e.target.closest('[data-state]');if(!b)return;const key=b.dataset.state;const candidates=model.frames.map((f,i)=>({q:f.after.ranks[rank].queue,p:f.before.ranks[rank].queue,i})).filter(x=>x.q===key&&(x.p!==key||x.i===0));const next=candidates.find(x=>x.i>index)||candidates[0];if(next){go(next.i);$('state-path').querySelector(`[data-state="${key}"]`).focus({preventScroll:true});}};
  $('rank-cards').onclick=e=>{const b=e.target.closest('[data-rank]');if(b){rank=Number(b.dataset.rank);render();$('rank-cards').querySelector(`[data-rank="${rank}"]`).focus({preventScroll:true});}};
  $('play').onclick=()=>{if(timer){pause();return;}if(index===model.frames.length-1){index=0;render();}$('play').textContent='暂停';$('play').setAttribute('aria-pressed','true');timer=setInterval(()=>{index++;render();if(index===model.frames.length-1)pause();},2400);};
  $('share').onclick=async()=>{try{await navigator.clipboard.writeText(location.href);$('announcement').textContent='已复制当前状态链接';$('share').textContent='已复制';}catch{$('announcement').textContent='复制不可用，请复制浏览器地址栏中的当前链接';$('share').textContent='请复制地址栏链接';}};
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});window.addEventListener('pagehide',pause);
  function restore(){const p=new URL(location.href).searchParams;rank=Math.max(0,Math.min(2,Number(p.get('rank'))||0));rank=Math.floor(rank);load(p.get('scenario'),p.get('step'));}
  window.addEventListener('popstate',restore);restore();
})();
