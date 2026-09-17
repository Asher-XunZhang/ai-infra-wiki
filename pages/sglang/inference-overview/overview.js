(() => {
  'use strict';
  const api=window.InferenceOverview,$=id=>document.getElementById(id);
  const tokens=(n,prefix,kind='')=>Array.from({length:n},(_,i)=>`<span class="token ${kind}">${prefix}${i+1}</span>`).join('');
  const listen=(ids,render)=>ids.forEach(id=>$(id).addEventListener('input',render));
  const page=document.body.dataset.lesson;
  if(page==='journey'){
    let step=0;
    function render(){
      const s=api.generation($('prompt-size').value,step);
      $('prompt-size-value').textContent=s.prompt;
      $('journey-step').textContent=`${step} / 5 · ${s.phase}`;
      $('journey-prev').disabled=step===0;$('journey-next').disabled=step===5;
      $('journey-flow').innerHTML=['接入与分词','排队 / 准入','Prefill','Decode × 4','输出结束'].map((x,i)=>`<div class="flow-node ${i===(step===0?1:step===1?2:step<5?3:4)?'active':''}">${x}<small>${['文本 → token IDs','先有资源，再计算','全部输入经过模型','上一输出作为新输入','达到教学输出上限'][i]}</small></div>`).join('<span class="flow-arrow" aria-hidden="true">→</span>');
      $('journey-input').innerHTML=step===0?tokens(s.prompt,'p'):step===1?tokens(s.prompt,'p','new'):`<span class="token new">y${step-1}</span>`;
      $('journey-output-tokens').innerHTML=step?tokens(step,'y','new'):'尚未生成';
      $('journey-kv').innerHTML=step?tokens(s.prompt,'p')+tokens(step-1,'y'):'尚未写入';
      $('generation-loop').innerHTML=`<div class="generation-chain"><div><small>本轮送入</small><strong>${step<=1?'p1…p'+s.prompt:'y'+(step-1)}</strong></div><span aria-hidden="true">→</span><div class="generation-model"><small>同一个完整模型</small><strong>层 1 → … → 层 L</strong></div><span aria-hidden="true">→</span><div><small>${step?'刚采样':'尚未采样'}</small><strong>${step?'y'+step:'?'}</strong></div></div><div class="feedback-path">${step===0?'等待准入，尚未进入模型':step===5?'已达到本例输出上限 · 不再送回模型':`↶ 下一轮把 y${step} 送回输入端，再经过完整模型`}</div>`;
      $('journey-output').innerHTML=step===0?'请求已被分词为示例 token p1…pN，正在等调度器接纳。此时还没有 KV，也没有输出。':`<strong>${s.phase} 完成：已有 ${step} 个输出 token，KV 覆盖 ${s.kv} 个位置。</strong> ${step===1?'最后一个输入位置的 logits 用于采样 y1。Prefill 已经产出了第一个输出 token。':`把 y${step-1} 输入完整模型，写入它的 KV，再采样 y${step}。历史 token 的 K/V 直接复用。`} 最新输出 y${step} 尚未经过下一轮前向，因此这里还没有它的 KV。${step===5?'本例到此停止；生产中还可能由 EOS、停止串、长度上限或取消结束。':''}`;
    }
    $('journey-prev').onclick=()=>{step=Math.max(0,step-1);render();};
    $('journey-next').onclick=()=>{step=Math.min(5,step+1);render();};
    $('journey-reset').onclick=()=>{step=0;render();};
    listen(['prompt-size'],()=>{step=0;render();});render();
  }
  if(page==='transformer'){
    let phase='prefill';
    const descriptions={embedding:'把本轮输入的 token ID 查成向量。Prefill 输入多个位置；普通 Decode 每条请求输入一个位置。',attention:'在每一层为当前位置计算 Q/K/V；K/V 写入该层缓存，Q 与可见历史 K/V 进行 attention。Decode 没有跳过 attention。',mlp:'每一层还要执行前馈网络（MLP）以及归一化、残差。Decode 并不是只读 KV，不做模型计算。',head:'最终隐藏状态经输出头得到词表 logits，再由采样步骤选出下一个 token。采样是生成流程的一步，不是另一半 Transformer。'};
    function render(){
      const p=phase==='prefill';
      $('phase-prefill').setAttribute('aria-pressed',String(p));$('phase-decode').setAttribute('aria-pressed',String(!p));
      $('transformer-input').innerHTML=p?tokens(4,'p','new'):'<span class="token new">y1</span>';
      $('transformer-shape').textContent=p?'4 个新位置 × hidden size':'1 个新位置 × hidden size';
      $('attention-title').textContent=p?'同一层：4 个输入位置的因果可见性':'同一层：y1 可以读取 4 个 prompt 位置和自身';
      const cols=p?4:5;
      $('attention-grid').style.gridTemplateColumns=`repeat(${cols+1},minmax(0,1fr))`;
      const keys=p?['p1','p2','p3','p4']:['p1','p2','p3','p4','y1'];
      const header='<span class="attention-axis">Q ↓<br>K/V →</span>'+keys.map(t=>`<span class="attention-axis">${t}</span>`).join('');
      $('attention-grid').innerHTML=header+(p?Array.from({length:4},(_,r)=>`<span class="attention-axis">p${r+1}</span>`+Array.from({length:4},(_,c)=>`<span class="attention-cell ${c<=r?'allowed':'masked'}" aria-label="p${r+1} ${c<=r?'可看':'不可看'} p${c+1}">${c<=r?'可看':'遮住'}</span>`).join('')).join(''):'<span class="attention-axis">y1</span>'+keys.map(t=>`<span class="attention-cell allowed" aria-label="y1 可看 ${t}">可看</span>`).join(''));
      $('phase-summary').textContent=p?'Prefill：这些位置可以组成一次批量前向，但因果遮罩仍阻止它们看到未来。最后位置用于预测 y1。':'Decode：只新增 y1 这个位置的计算；各层读取自己的历史 KV，最终预测 y2。';
      $('transformer-output').innerHTML=`<strong>${$('model-part').selectedOptions[0].textContent}</strong> · ${descriptions[$('model-part').value]}`;
      document.querySelectorAll('[data-part]').forEach(n=>n.classList.toggle('active',n.dataset.part===$('model-part').value));
    }
    $('phase-prefill').onclick=()=>{phase='prefill';render();};$('phase-decode').onclick=()=>{phase='decode';render();};listen(['model-part'],render);render();
  }
  if(page==='kv-cache'){
    function render(){
      const s=api.cache(8,$('reuse-count').value,$('decode-count').value);
      $('reuse-value').textContent=s.reuse;$('decode-value').textContent=s.decode;
      $('cache-compute').textContent=s.computed;$('cache-count').textContent=s.kv;$('cache-output-count').textContent=s.outputs;
      const kvCell=(label,reused)=>`<span class="cache-cell ${reused?'reused':'written'}"><b>${label}</b><span><i>K</i><i>V</i></span><small>${reused?'复用':'新写'}</small></span>`;
      $('kv-layers').innerHTML=Array.from({length:3},(_,i)=>`<div class="kv-layer"><strong>层 ${i+1}</strong><div class="token-row">${Array.from({length:8},(_,j)=>kvCell('p'+(j+1),j<s.reuse)).join('')}${Array.from({length:s.decode},(_,j)=>kvCell('y'+(j+1),false)).join('')}<span class="cache-cell not-written"><b>y${s.outputs}</b><span><i>—</i><i>—</i></span><small>尚未写入</small></span></div></div>`).join('');
      $('cache-output').innerHTML=`<strong>复用前 ${s.reuse} 个位置，这次 Prefill 新算 ${s.computed} 个位置。</strong> ${s.reuse?'复用的前缀必须完全匹配，并且缓存已可用；新后缀仍需读取这些 KV。':'没有前缀命中，8 个输入位置都需要计算。'} 此后执行 ${s.decode} 轮 Decode，得到 ${s.outputs} 个输出；每层的 KV 覆盖 ${s.kv} 个已处理位置。图中的相同标签代表同一 token，不代表不同层存着相同数值。`;
    }
    listen(['reuse-count','decode-count'],render);render();
  }
  if(page==='deployment'){
    let step=0;
    function render(){
      const split=$('deployment-mode').value==='split';
      const s=api.handoff(split,step,$('gate-kv').checked,$('gate-meta').checked,$('gate-slot').checked);
      $('handoff-controls').hidden=!split;$('deployment-step').textContent=`${step+1} / 5`;
      $('deployment-prev').disabled=step===0;$('deployment-next').disabled=step===4;
      const names=split?['接入 / 选择 P、D','P：完整模型前向','P → D：交接状态','D：检查可执行条件','D：完整模型续写']:['接入 / 选择实例','本实例：完整模型前向','KV 留在本实例','本实例：调度下一轮','本实例：完整模型续写'];
      $('deployment-flow').innerHTML=window.renderDeploymentScene(s,{
        gpus:Number($('deployment-gpus').value),kv:$('gate-kv').checked,metadata:$('gate-meta').checked,
      });
      $('deployment-gates').innerHTML=split?[['gate-kv','KV 数据可用'],['gate-meta','元数据已匹配'],['gate-slot','本地执行资源可用']].map(([id,t])=>`<div class="gate ${step>=3&&$(id).checked?'ready':''}">${step<3?'交接后将检查':$(id).checked?'✓ 已满足':'○ 仍在等待'} · ${t}</div>`).join(''):`<div class="gate ${step?'ready':''}">${step?'KV 已在本实例保留':'Prefill 完成后，KV 将保留在本实例'}；没有跨 P/D 的接收门槛。本地调度和资源检查仍然存在。</div>`;
      const detail=split?['路由协调两个服务角色。请求控制消息与大块 KV 数据可以走不同通路；不是客户端把完整模型搬来搬去。','P 侧服务组执行完整的模型层，处理 prompt 并采样 y1。P 不是模型的前半层。','交接 prompt 的 KV、首个输出 token 与必要元数据。层间 hidden states 与 P→D 的 KV 不是同一份对象；模型权重不会随每条请求传输。',s.ready?'教学门槛均已满足，请求可以进入后续可执行调度。实际实现还会受所选缓存、传输和并行路径约束。':'尚不能执行 Decode。切换下面的条件，观察“KV 已到”为什么仍然不够。',s.blocked?'等待中：缺失条件尚未满足，不能把本步骤画成已经完成的 Decode。':'D 侧服务组输入 y1，读取各层 prompt KV，经过完整模型后采样 y2；后续继续逐轮生成。']:['请求进入一个承担 P 与 D 两种工作的服务实例（实例可以包含多张 GPU）。','同一个服务组完成 Prefill，写入各层 KV，并采样 y1。','KV 在原实例保留，通过本地请求映射继续使用，无需跨角色复制。','调度器决定何时让它参加下一批执行；其他请求的 Prefill 可能占用执行时间。','本实例输入 y1，复用 KV 并采样 y2。P/D 的计算语义没有因为合并部署而改变。'];
      $('deployment-output').innerHTML=`<strong>${s.blocked?'等待条件 · ':''}${names[step]}</strong><br>${detail[step]}`;
    }
    $('deployment-prev').onclick=()=>{step=Math.max(0,step-1);render();};$('deployment-next').onclick=()=>{step=Math.min(4,step+1);render();};
    listen(['deployment-mode'],()=>{step=0;render();});listen(['deployment-gpus','gate-kv','gate-meta','gate-slot'],render);render();
  }
  if(page==='scheduling'){
    let round=0;
    function renderSchedule(){
      const s=api.schedule($('chunk-size').value,round);round=s.round;
      $('chunk-size-value').textContent=s.chunk;$('schedule-round').textContent=`${round} / ${s.rounds.length} 轮`;
      $('schedule-next').disabled=round===s.rounds.length;
      $('schedule-rounds').innerHTML=s.rounds.map((r,i)=>`<div class="round batch-round ${i===round-1?'active':''}"><strong>轮 ${i+1}<small>${i<round?'已执行':'未执行'}</small></strong><div class="batch-container"><div class="batch-label">一个 batch · ${1+r.prefill} 个新位置</div><div class="batch-work"><span class="batch-token from-a">A<small>Decode</small></span><b aria-hidden="true">＋</b><div class="batch-prefill">${Array.from({length:r.prefill},(_,j)=>`<span class="batch-token from-b">B·p${i*s.chunk+j+1}<small>Prefill</small></span>`).join('')}</div></div>${i===s.rounds.length-1?'<small class="batch-finish">最后一块完成 → B 采样 y1 → 下一轮可加入 Decode</small>':''}</div></div>`).join('');
      $('batch-a').innerHTML=round?tokens(round,'续写','new'):'本实验尚未推进';$('batch-b').innerHTML=tokens(s.processed,'p','new')+`<span class="token pending">剩余 ${8-s.processed}</span>`;
      $('schedule-output').innerHTML=`<strong>B 的 Prefill 已处理 ${s.processed} / 8 个输入位置。</strong> A 在这些教学轮次里续写了 ${s.aTokens} 个 token。${s.processed===8?'B 的最后一块完成，才在本例中采样 y1；下一轮它可参加 Decode。':'B 的中间块只积累状态，不把每个 chunk 当成一个输出 token。'} ${s.chunk===8?'一整块完成 B 的 Prefill，单轮工作较多。':'小块给调度器更多交错执行机会，但增加调度轮数。'} `;
    }
    $('schedule-next').onclick=()=>{round++;renderSchedule();};$('schedule-reset').onclick=()=>{round=0;renderSchedule();};listen(['chunk-size'],()=>{round=0;renderSchedule();});renderSchedule();
    function renderLatency(){
      const s=api.latency(...['latency-q','latency-p','latency-h','latency-d'].map(id=>$(id).value));
      ['latency-q','latency-p','latency-h','latency-d'].forEach((id,i)=>$(id+'-value').textContent=s.values[i]);
      $('ttft-value').textContent=s.ttft+' ms';$('tpot-value').textContent=s.tpot+' ms';$('total-value').textContent=s.total+' ms';
      const parts=[['排队',s.values[0]],['Prefill',s.values[1]],['首 token 可见前的额外等待',s.values[2]],['D1',s.values[3]],['D2',s.values[3]],['D3',s.values[3]]];
      $('latency-track').innerHTML=parts.filter(p=>p[1]).map(([label,n])=>`<div class="time-segment" style="flex:${n}" role="img" aria-label="${label}：${n} 毫秒" title="${label}：${n} ms"></div>`).join('');
      $('latency-legend').textContent=parts.map(([l,n])=>`${l} ${n} ms`).join(' · ');
      $('latency-output').textContent='教学假设：首 token 在排队、Prefill 和额外等待之后对客户端可见；此后 3 轮 Decode 各输出一个 token，忽略其他开销与重叠。改变排队时间只改变本模型的 TTFT；改变 Decode 单步时间影响后续输出间隔。真实 PD 中交接是否计入 TTFT，取决于首 token 何时被发送到客户端。';
    }
    listen(['latency-q','latency-p','latency-h','latency-d'],renderLatency);renderLatency();
  }
})();
