/* One renderer and one authoritative playback state for overview and data operations. */
(() => {
  'use strict';
  const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const set=a=>a?.length?`{${a.join(', ')}}`:'∅';
  const pill=(s,cls='')=>`<span class="data-pill ${cls}">${esc(s)}</span>`;
  const queues={none:'未接入',bootstrap:'bootstrap_queue',waiting:'waiting_queue',batch:'本轮 batch',chunked:'保留 chunked_req',inflight:'inflight_queue',done:'已退出'};
  const doneLabel=s=>s==='success'?'成功交接':s==='bootstrap-failed'?'握手失败 / 取消':s==='transfer-failed'?'交接异常':'处理中';
  function tokenRow(req,segment,mode='cut',before=0,after=0){
    return `<div class="token-request"><div class="request-label req-${req.index}"><b>${req.rid}</b><span>${segment?`前缀 ${segment.start} · 本轮 ${segment.end-segment.start} · 尾部 ${req.length-segment.end}`:`完整序列 · ${req.length} token`}</span></div><div class="token-row">`+req.tokenIds.map((id,i)=>{
      let cls=segment?(i<segment.start?'prefix':i<segment.end?'selected':'future'):'neutral';
      if(mode==='cache')cls=i<before?'prefix':i<after?'written':'future';
      if(mode==='send')cls=i<before?'prefix':i<after?'sending':'future';
      return `<span class="token ${cls} ${segment&&i===segment.start?'cut-start':''} ${segment&&i===segment.end-1?'cut-end':''}" style="--order:${i}" title="${req.rid} 的位置 ${i}，预设 ID ${id}"><small>${i}</small><b>${id}</b></span>`;
    }).join('')+'</div></div>';
  }
  function chain(items){return '<div class="data-flow">'+items.map(x=>`<div>${x}</div>`).join('<span class="data-arrow" aria-hidden="true">→</span>')+'</div>';}
  function batchView(f,model){
    const b=f.work.batch,s=f.after,config=model.config;
    if(f.type==='budget')return `<div class="budget-meters"><div><span>本批 token 预算</span><strong>${b.budget??'不限'}</strong><small>按页计费 ${b.charge} · 实际新 token ${b.tokenCount}</small></div><div><span>请求容量</span><strong>${b.segments.length} / ${b.capacity}</strong><small>${model.requests.length-b.segments.length} 条请求未在本批中</small></div></div><div class="candidate-list">${s.requests.map(req=>{const p=b.segments.find(x=>x.rid===req.rid);return `<div class="${p?'chosen':''}"><b>${req.rid}</b><span>${req.outcome!=='active'?doneLabel(req.outcome):p?`入选 [${p.start}, ${p.end}) · 计费 ${p.charge}`:s.ranks[0].requests[req.rid].cacheEnd===req.length?'已完成计算':'继续等待容量 / 预算'}</span></div>`;}).join('')}</div>`;
    if(f.type==='cut')return `<div class="visual-key"><span>绿 = 已计算前缀</span><span>蓝 = 本轮新增</span><span>灰 = 未处理尾部</span><b>${b.id} · ${b.budget===null?'不切块':'共享预算 '+b.budget}</b></div>`+model.requests.filter(q=>b.segments.some(x=>x.rid===q.rid)).map(q=>tokenRow(q,b.segments.find(x=>x.rid===q.rid))).join('');
    const packed=b.segments.map(p=>{const req=model.requests.find(q=>q.rid===p.rid);return `<div class="packed-request req-${req.index}"><strong>${p.rid}</strong><span>prefix_len=${p.start} · extend_len=${p.end-p.start}</span><div class="packed-tokens">${req.tokenIds.slice(p.start,p.end).map((id,j)=>`<span style="--order:${j}">${id}</span>`).join('')}</div>${f.work.allocate?`<small>本级写入映射：${p.rid}[${p.start}:${p.end}] → PP${f.work.rank} KV slots（示意）</small>`:''}</div>`;}).join('');
    return chain([`请求队列<br>${b.segments.map(p=>pill(p.rid)).join('')}`,`${b.id} · EXTEND<br>${b.tokenCount} 个新 token / ${b.segments.length} 条请求`,`PP${f.work.rank??0} 本地准备<br>input_ids + 长度 + KV 位置`])+`<div class="batch-envelope"><div class="envelope-heading">${b.id}.input_ids = 按请求边界拼接的序列 <span>page_size=${config.pageSize}</span></div>${packed}</div>`;
  }
  function statusTable(state,model,highlight){
    return `<div class="state-scroll"><table class="sender-matrix"><thead><tr><th>本地状态</th>${model.requests.map(q=>`<th>${q.rid}</th>`).join('')}</tr></thead><tbody>${state.ranks.map((r,i)=>`<tr class="${i===highlight?'selected-rank':''}"><th>PP${i}</th>${model.requests.map(req=>{const q=r.requests[req.rid];return `<td class="${q.reason==='FINISH_ABORT'||q.sender==='Failed'?'status-bad':q.sender==='Success'||q.reason==='FINISH_LENGTH(0)'?'status-good':''}">${q.released?esc(q.reason):esc(q.sender)}</td>`;}).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  function consensusView(f,model){
    const s=f.after,r=f.work.rank,boot=['bootstrap','admission'].includes(f.type);
    let content=statusTable(s,model,r),c=boot?s.bootstrap:s.terminal;
    if(boot&&model.config.fault==='bootstrap_abort')content+=`<p class="branch-note">本地取消集合：${model.config.failed.map((ids,i)=>`PP${i} ${set(ids)}`).join(' · ')}。即使 poll 仍为 WaitingForInput，也要从 good 移除并加入 bad。</p>`;
    if(!c||f.work.poll)return content;
    if(boot){
      const k=c.through;
      content+=`<div class="set-equations"><div class="good-set"><b>good · 交集</b><code>${k===0?set(c.localGood[0]):`${set(c.upstream.good)} ∩ ${set(c.localGood[k])}`} = ${set(c.good)}</code></div><div class="bad-set"><b>bad · 并集</b><code>${k===0?set(c.localBad[0]):`${set(c.upstream.bad)} ∪ ${set(c.localBad[k])}`} = ${set(c.bad)}</code></div></div>`;
      if(f.type==='admission')content+=chain([`返回 PP${r} 的名单`,`${pill('good '+set(c.good),'good')} ${pill('bad '+set(c.bad),'bad')}`,`${c.good.length} 条分配 metadata → 等待选批<br>${c.bad.length} 条失败清理 → 退出`]);
    }else{
      content+=`<div class="terminal-sets">${c.local.map((ids,i)=>`<div class="${i===c.through?'current':''}"><b>PP${i} 终态集合</b><code>${set(ids)}</code><small>Success 或 Failed</small></div>`).join('<span>∩</span>')}</div><div class="consensus-answer"><b>合并到 PP${c.through}：${set(c.intersection)}</b><span>这是收尾名单，不是成功名单。</span></div>`;
    }
    return content;
  }
  function render(f,model){
    const s=f.after,b=f.work.batch,r=f.work.rank??0;
    document.getElementById('operation-title').textContent=f.title;
    document.getElementById('operation-description').textContent=f.description;
    document.getElementById('operation-source').href='https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/'+f.source;
    let html='';
    if(f.type==='intake')html=chain(f.work.rank===undefined?['输入文本 / 生成参数','Tokenizer → 完整 IDs','rid / bootstrap_room']:[r===0?'来自 Tokenizer 的请求':`PP${r-1} 转发的请求`,`PP${r} 本地 Req · 复用完整 IDs`,`sender + bootstrap_room<br>加入 bootstrap_queue`])+model.requests.map(q=>tokenRow(q)).join('');
    else if(['budget','cut','pack'].includes(f.type))html=batchView(f,model);
    else if(['bootstrap','admission','consensus'].includes(f.type))html=consensusView(f,model);
    else if(f.type==='forward'){
      html=chain([r===0?`${b.tokenCount} 个新 token<br>Embedding`:`PP${r-1} 的激活<br>[${b.tokenCount}, hidden_size]`,`PP${r} · L${r*10}–L${r*10+9}`,r===2?'输出头 / 本块结果':'交给下一级的激活']);
      html+=`<div class="layer-strip">${Array.from({length:10},(_,i)=>`<span style="--order:${i}">L${r*10+i}<small>写 K / V</small></span>`).join('')}</div>`;
      for(const seg of b.segments){const req=model.requests.find(q=>q.rid===seg.rid),before=f.before.ranks[r].requests[req.rid].cacheEnd;html+=`<p class="data-caption">${req.rid} · PP${r} 的 KV：${before} → ${seg.end} 个 token（仅本级层）</p>`+tokenRow(req,null,'cache',before,seg.end);}
    }else if(f.type==='proxy')html=chain([`PP${r-1}<br>${pill(b.id)} ${pill('hidden_states / residual')}`,`激活 [${b.tokenCount}, hidden_size]<br>${b.segments.map(p=>pill(`${p.rid}:${p.end-p.start}`)).join('')}`,`PP${r} 对应的 ${b.id}`])+`<p class="branch-note">KV 留在各级。batch 的请求边界保持一致，不能把 h 当成 KV 搬运。</p>`;
    else if(f.type==='output')html=chain([`末级 ${b.id} 结果`,`${b.segments.map(p=>pill(`${p.rid}：${p.last?'最终块 t₀':'中间块结果'}`)).join('')}`,`PP${r}：匹配旧 batch 槽<br>等待必要 D2H 事件`]);
    else if(f.type==='result')html=`<div class="result-cards">${b.segments.map(p=>`<div><strong>${p.rid} · ${p.last?'最终块':'中间块'}</strong><code>output_ids：${f.before.ranks[r].requests[p.rid].token?'[t₀]':'[]'} → ${p.last?'[t₀]':'[]'}</code><span>${p.last?'加入 inflight_queue':'保留 chunked_req / 已计算前缀'}</span></div>`).join('')}</div>`;
    else if(f.type==='send'){
      html=chain([`PP${r} 本级层的源 KV`,'按页索引提交传输','Decode 的对应层位置']);
      for(const send of f.work.sends){const req=model.requests.find(q=>q.rid===send.rid);html+=`<p class="data-caption">${send.rid} · start_send_idx ${send.start} → ${send.end} · ${send.last?'最终块，附 metadata':'非最终块，只提交整页'}</p><div class="page-list">`;
        for(let i=send.start;i<send.end;i+=model.config.pageSize)html+=`<div class="page-unit"><b>逻辑页 ${Math.floor(i/model.config.pageSize)}</b><span>token [${i}, ${Math.min(i+model.config.pageSize,send.end)})</span><small>K / V · 物理索引另行映射</small></div>`;
        html+='</div>'+tokenRow(req,null,'send',send.start,send.end);
      }
      html+='<p class="branch-note">已提交 ≠ 传输成功。源 KV 请求引用继续保留。</p>';
    }else if(['release','cleanup'].includes(f.type)){
      html=`<div class="release-list">返回 PP${r} 的 release_rids：${pill(set(s.release))}</div><div class="result-cards">`+f.work.ids.map(id=>{const a=f.before.ranks[r].requests[id],q=s.ranks[r].requests[id];return `<div class="${q.reason==='FINISH_ABORT'?'failed-card':''}"><strong>${id} · ${q.released?esc(q.reason):q.permit?'本地复查通过':'仍需等待'}</strong><span>本地 poll：${esc(a.sender)} → ${esc(q.sender)}</span><span>KV 引用：${a.holdsKv?'持有':'未持有'} → ${q.holdsKv?'仍持有':'未持有'}</span><span>metadata：${a.metadata?'占用':'未占用'} → ${q.metadata?'仍占用':'未占用'}</span><span>队列：${queues[a.queue]} → ${queues[q.queue]}</span><div class="resource-pair"><i class="${q.holdsKv?'held':'returned'}">KV 引用 ${q.holdsKv?'●':'○'}</i><i class="${q.metadata?'held':'returned'}">metadata ${q.metadata?'●':'○'}</i></div></div>`;}).join('')+'</div>';
    }else html=`<div class="outcome-cards">${s.requests.map(q=>`<div class="${q.outcome==='success'?'good':'bad'}"><strong>${q.rid} · ${doneLabel(q.outcome)}</strong><p>${q.outcome==='success'?'已完成本次 KV 交接，Decode 可继续。':'不把终态共识当作成功，检查下面各级原因。'}</p>${s.ranks.map((rank,i)=>`<small>PP${i}：${esc(rank.requests[q.rid].reason??rank.requests[q.rid].sender)}</small>`).join('')}</div>`).join('')}</div>`;
    document.getElementById('operation-visual').innerHTML=html;
    document.getElementById('request-state').innerHTML='<div class="state-scroll"><table><thead><tr><th>请求 / PP</th><th>队列</th><th>本地状态 / 原因</th><th>KV 已算 / 已提交</th><th>请求引用 / metadata</th></tr></thead><tbody>'+model.requests.flatMap(req=>s.ranks.map((rank,i)=>{const q=rank.requests[req.rid];return `<tr><th>${req.rid} / PP${i}</th><td>${queues[q.queue]}</td><td>${esc(q.reason??q.sender)}</td><td>${q.cacheEnd} / ${q.sendEnd}</td><td>${q.holdsKv?'持有':'无'} / ${q.metadata?'占用':'无'}</td></tr>`;})).join('')+'</tbody></table></div>';
  }
  window.PrefillOperationLab={render};
})();
