(function(){
 'use strict';
 const M=window.ParallelismModel,host=document.querySelector('[data-parallel-lab]');if(!M||!host)return;
 const $=id=>host.querySelector('#'+id),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 let mode='tp',step=0,timer=null;
 const titles={tp:'TP · 切一层的权重',pp:'PP · 按层接力',dp:'DP · 分配不同请求',ep:'EP · 分配专家任务',cp:'CP · 分担上下文查询'};
 const vector=(values,cls='')=>`<span class="number-vector ${cls}">${values.map(v=>`<b>${esc(v)}</b>`).join('')}</span>`;
 const layers=()=>'<div class="layer-stack" aria-label="全部四层">'+[1,2,3,4].map(i=>`<i>L${i}</i>`).join('')+'</div>';
 function packet(path,x,y,animate=true){
  const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
  return `<circle class="travel" r="4" cx="${reduced||!animate?x:0}" cy="${reduced||!animate?y:0}">${reduced||!animate?'':`<animateMotion dur=".8s" path="${path}" fill="freeze"/>`}</circle>`;
 }
 function bridge(text,starts=[125,375]){
  const paths=starts.map((x,i)=>`M${x} 2 V16 L${240+i*20} 48`);
  return `<div class="parallel-bridge"><svg viewBox="0 0 500 56" preserveAspectRatio="none" aria-hidden="true"><defs><marker id="p-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z"/></marker></defs>${paths.map((d,i)=>`<path class="flow-line" d="${d}" marker-end="url(#p-arrow)"/>${packet(d,240+i*20,48)}`).join('')}</svg><strong>${text}</strong></div>`;
 }
 function tp(){
  const d=M.tensor($('tp-kind').value,$('tp-gather').checked),row=d.kind==='row';
  $('tp-gather-wrap').hidden=row;
  const cards=[0,1].map(rank=>{
   const matrix=d.W.map((r,ri)=>r.map((v,ci)=>`<b class="${(row?Math.floor(ci/2):ri)===rank?'owned':'other'}">${v}</b>`).join('')).join('');
   return `<article class="gpu rank-${rank}"><h3>GPU ${rank}</h3><div class="gpu-input">${vector(row?d.X.slice(rank*2,rank*2+2):d.X)}<span>↓</span></div><div class="weight-matrix" aria-label="GPU ${rank} 的权重分片">${matrix}</div><small>W · 输出 × 输入</small><div class="local-result ${step>=1?'shown':''}"><span>${row?'局部贡献':'输出列'}</span>${vector(d.local[rank])}</div></article>`;
  });
  let html=`<div class="two-gpus">${cards.join('')}</div>`;
  if(step===2)html+=(!row&&!d.gather?'<div class="shard-retained">↓ 各自保留输出列，交给后续分片计算</div>':bridge(row?'＋ 按相同位置求和':'拼接不同输出列')+`<div class="parallel-result">${vector(d.full)}<small>两张 GPU 各得到这份完整结果</small></div>`);
  return {html,source:row?'row':'column',phase:['分配权重','两卡本地计算','合并或保留分片'][step],caption:step===0?(row?'每张卡只保存 W 的一部分输入列，也只读取对应输入。':'每张卡保存 W 的不同输出行；它们读取同一份输入。'):step===1?(row?'[3,5] 与 [7,25] 都有两列，但都只是贡献，尚未得到完整答案。':'GPU 0 算出第一个输出 10；GPU 1 算出第二个输出 30。'):row?'相同位置相加得到 [10,30]；把贡献拼接起来会得到错误的形状与含义。':d.gather?'All-gather 将两段输出按原顺序拼接，得到 [10,30]。':'下一层若能直接消费这些分片，就不必在这里收齐。',why:'本例使用无 bias 的小矩阵，显示存储权重 W，计算 X × Wᵀ。Column 名称来自数学权重 A=Wᵀ 的列方向。Row 演示启用结果规约的普通路径；源码还支持由调用方融合、延后或替换通信，不能推断每层总有同一次 All-reduce。'};
 }
 function pp(){
  const n=Number($('pp-count').value),ticks=n+1;
  const header=Array.from({length:ticks},(_,i)=>`<small>t${i+1}</small>`).join('');
  let html=`<div class="pipeline-grid" style="--ticks:${ticks}"><div></div><div class="tick-head">${header}</div>`;
  for(let stage=0;stage<2;stage++){
   const cells=Array.from({length:ticks},(_,i)=>{const batch=i+1-stage,valid=batch>0&&batch<=n;return `<button type="button" class="pipeline-cell ${valid?'batch-'+batch:'bubble'} ${i+1===step?'now':i+1<step?'past':'future'}" data-tick="${i+1}" aria-label="t${i+1}，阶段 ${stage}，${valid?'M'+batch:'空闲'}">${valid?'M'+batch:'·'}</button>`;}).join('');
   html+=`<div class="stage-label"><strong>GPU ${stage}</strong><span>L${stage*2+1} · L${stage*2+2}</span><small>本级权重 / KV</small></div><div class="tick-row">${cells}</div>`;
  }
  html+='</div>';
  if(step>=2&&step<=n+1)html+=`<div class="activation-handoff"><span>M${step-1} · GPU 0 的结果</span><b class="handoff-arrow">→</b><span>GPU 1 · 激活输入</span></div>`;
  return {html,phase:step===0?'按层分配':`t${step} · 当前执行`,source:step===0?'layers':'pp',caption:step===0?'两级持有不同层；同一个 microbatch 必须先过 L1–2，再过 L3–4。':step===n+1?`GPU 1 正在完成 M${n}；GPU 0 已无后续 microbatch。`:step===1?'M1 先进入 GPU 0。GPU 1 还没有上一级的激活，不能提前算。':`GPU 0 处理 M${step}，同时 GPU 1 处理 M${step-1}；两张卡此刻处理的是不同 microbatch。`,why:'两级各用一个等长时隙是教学假设，用来展示前后级依赖和填充 / 排空。不是 SGLang event loop 的逐次复刻或吞吐预测。跨级激活继续下一层计算；本级 KV 随本级 Attention 层保存，不能把 PP 激活交接理解为 P/D 的 KV 搬运。'};
 }
 function dp(){
  const d=M.dataParallel()[step];
  const pending=[1,2,3].filter(r=>r>step).map(r=>`<span class="request-chip">R${r}</span>`).join('');
  const html=`<div class="request-arrivals"><small>待分配</small>${pending||'<span class="quiet">已分配完</span>'}</div>${step?`<div class="route-choice to-${d.target}"><b>R${step}</b> <span>→ 副本 ${d.target}</span></div>`:''}<div class="two-gpus">${d.requests.map((rs,r)=>`<article class="gpu rank-${r} ${d.target===r?'receiving':''}"><h3>副本 ${r} · GPU ${r}</h3>${layers()}<small>完整模型 · 独立的队列与 KV 空间</small><div class="replica-requests">${rs.map(q=>`<span class="request-chip ${q===step?'arriving':''}">R${q}</span>`).join('')||'<span class="quiet">空</span>'}</div></article>`).join('')}</div>`;
  return {html,phase:step?`分配 R${step}`:'两个模型副本',source:step?'dp':'dpworld',caption:step?`R${step} 只交给副本 ${d.target}。另一个副本可以处理自己的请求，两边不为同一层合并计算贡献。`:'普通 DP 为每个副本准备完整模型；这里的每个副本仅使用一张 GPU。',why:'选择两个健康副本上的 round-robin 教学路径。源码还会检查外部路由、活跃集合与健康状态；本图没有模拟服务网关、缓存感知路由或完成速度。普通 DP 的各个副本还可以分别由 TP × PP 组成，不能把这里的 GPU 数套给 DP Attention。'};
 }
 function ep(){
  const d=M.experts($('ep-skew').checked,Number($('ep-token').value));
  const destinations=d.selected.map(e=>[75,175,325,425][e]);
  let html=`<div class="expert-input">${vector(['t'+d.token])}<small>本图跟随一个 token · 箭头表示逻辑派发</small></div><svg class="expert-routing" viewBox="0 0 500 58" preserveAspectRatio="none" aria-label="选中的两份专家任务" role="img"><defs><marker id="route-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z"/></marker></defs>${destinations.map((x,j)=>{const path=`M250 0 L${x} 52`;return `<path class="flow-line" d="${path}" marker-end="url(#route-arrow)"/>${packet(path,x,52,step===1)}<text x="${x}" y="40" text-anchor="middle">${d.weights[j]}</text>`;}).join('')}</svg><div class="two-gpus">`;
  for(let rank=0;rank<2;rank++)html+=`<article class="gpu rank-${rank}"><h3>GPU ${rank}</h3><div class="expert-pair">${[rank*2,rank*2+1].map(e=>{const j=d.selected.indexOf(e);return `<div class="expert ${j>=0?'selected':''}" data-expert="${e}"><strong>E${e}</strong>${j>=0?`<span class="expert-token ${step>=1?'arriving':''}">${step>=1?'t'+d.token:'待派发'}</span>${step>=2?`<b>${d.values[j]}</b>`:''}`:'<span>—</span>'}</div>`;}).join('')}</div></article>`;
  html+='</div>';
  if(step===3)html+=bridge('按路由权重合并',destinations)+`<div class="parallel-result"><span class="weighted-result">${d.weights[0]} × ${d.values[0]} + ${d.weights[1]} × ${d.values[1]} = <b>${d.result}</b></span></div>`;
  html+=`<div class="expert-load"><small>四个 token 共 8 份专家任务</small>${d.loads.map((load,e)=>`<div><span>E${e}</span><span class="load-track"><i style="width:${load/4*100}%"></i></span><b>${load}</b></div>`).join('')}</div>`;
  return {html,phase:['选专家','派发任务','专家计算','合并回 token'][step],source:'ep',caption:step===0?`t${d.token} 选择 E${d.selected.join(' 和 E')}；top-k=2 产生两份专家任务，仍是一个 token。`:step===1?'选中的专家获得该 token 的输入。图示逻辑派发，是否发生网络搬运由 dispatcher / backend 决定。':step===2?'每位专家执行自己的权重计算，产生各自贡献。这里只用标量函数代替专家网络。':`两份贡献按 0.75 / 0.25 合成 t${d.token} 的输出，并恢复其 token 身份。`,why:'本例两张 GPU 各存两个完整专家，MoE TP=1；用 fₑ(t)=t×(e+1) 解释加权，不是模型真实输出。负载条统计整个四-token 批次的逻辑专家任务数，不代表耗时；切换偏斜后，卡数不变，任务却可能集中。EP 不是 HTTP 路由，也不保证所有实现使用 All-to-All。'};
 }
 function cp(){
  const zigzag=$('cp-layout').value==='zigzag',d=M.context(zigzag);
  let html='';
  if(step===1){
   html='<div class="causal-scene"><div><p>列：可访问的 K/V 位置 →</p><div class="causal-matrix" role="img" aria-label="8 个查询位置的因果 Attention 三角，每行只能访问当前及更早位置">'+Array.from({length:64},(_,i)=>{const q=Math.floor(i/8),k=i%8,r=d.ranks[0].includes(q)?0:1;return `<i class="${k<=q?'rank-'+r:'masked'}" title="q${q} → k${k}">${k===0?q:''}</i>`;}).join('')+'</div><small>行：查询位置 0–7；颜色表示负责的 rank</small></div><div class="cp-work">'+d.loads.map((n,r)=>`<div class="rank-${r}"><strong>rank ${r}</strong><span class="load-track"><i style="width:${n/36*100}%"></i></span><b>${n} 对 Q–K</b></div>`).join('')+'</div></div>';
  }else html=`<div class="two-gpus">${d.ranks.map((rows,r)=>`<article class="gpu rank-${r}"><h3>CP rank ${r}</h3><small>负责的查询位置</small>${vector(rows)}${step===2?`<div class="kv-reunited"><span>本层汇合后的 K/V 位置</span>${vector(d.kv)}</div>`:''}</article>`).join('')}</div>`;
  return {html,phase:['分配查询位置','比较因果依赖','本层汇合 K/V'][step],source:step===2?'cpkv':'zigzag',caption:step===0?(zigzag?'首尾配对：rank 0 负责 0、1、6、7；rank 1 负责 2、3、4、5。':'连续切分对照：前四个与后四个查询分给不同 rank。'):step===1?`因果可见关系共有 36 对。当前分工为 ${d.loads[0]} / ${d.loads[1]} 对，${zigzag?'首尾配对平衡这项工作量。':'靠后查询能看到更多历史，工作量更大。'}`:'在所选 Zigzag Prefill 路径中，局部 K/V 会汇合并恢复顺序；切分查询不等于 KV 永久只存半份。',why:'只解释无前缀、8 个位置、CP=2 的 Zigzag Prefill 思路。连续切分是教学对照，非本页声称的源码执行路径；Q–K 配对数不是 kernel 耗时。Decode CP、稀疏 Attention、滑窗、padding、模型与 backend 的准入条件需另读对应源码，不能直接套用这幅三角图。'};
 }
 function maximum(){return mode==='tp'?2:mode==='pp'?Number($('pp-count').value)+1:mode==='dp'||mode==='ep'?3:$('cp-layout').value==='zigzag'?2:1;}
 function stop(){clearInterval(timer);timer=null;$('parallel-play').textContent='播放';$('parallel-play').setAttribute('aria-pressed','false');}
 function render(){
  const f=({tp,pp,dp,ep,cp}[mode])(),max=maximum();
  $('parallel-title').textContent=titles[mode];$('parallel-scene').innerHTML=f.html;$('parallel-scene').dataset.mode=mode;$('parallel-scene').dataset.step=step;
  $('parallel-status').textContent=`${step+1} / ${max+1} · ${f.phase}`;$('parallel-caption').textContent=f.caption;$('parallel-why').textContent=f.why;
  $('parallel-source').href=M.url(f.source);$('parallel-source').textContent='源码 · '+M.sources[f.source][2]+' ↗';
  $('parallel-previous').disabled=step===0;$('parallel-next').disabled=step===max;
  host.querySelectorAll('[data-settings]').forEach(el=>el.hidden=el.dataset.settings!==mode);
  if(step===max)stop();
 }
 function reset(){stop();step=0;render();}
 host.querySelectorAll('button[data-mode]').forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.mode;host.querySelectorAll('button[data-mode]').forEach(el=>el.setAttribute('aria-pressed',String(el===b)));reset();}));
 ['tp-kind','tp-gather','pp-count','ep-skew','ep-token','cp-layout'].forEach(id=>$(id).addEventListener('change',reset));
 $('parallel-next').addEventListener('click',()=>{stop();step=Math.min(maximum(),step+1);render();});
 $('parallel-previous').addEventListener('click',()=>{stop();step=Math.max(0,step-1);render();});
 $('parallel-reset').addEventListener('click',reset);
 $('parallel-play').addEventListener('click',()=>{if(timer){stop();return;}if(step===maximum()){step=0;render();}$('parallel-play').textContent='暂停';$('parallel-play').setAttribute('aria-pressed','true');timer=setInterval(()=>{step++;render();},2000);});
 $('parallel-scene').addEventListener('click',e=>{const b=e.target.closest('[data-tick]');if(b){stop();step=Number(b.dataset.tick);render();}});
 document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});window.addEventListener('pagehide',stop);
 host.hidden=false;reset();
})();
