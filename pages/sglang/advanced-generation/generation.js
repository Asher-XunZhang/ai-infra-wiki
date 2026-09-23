(()=>{
'use strict';
const M=GenerationModel,$=s=>document.querySelector(s),lab=$('[data-generation-lab]');let mode='state',step=0,timer=null;
const prefs=()=>({state:$('#state-kind').value,accept:Number($('#spec-accept').value),grammar:$('#grammar-prefer').value});
const last=()=>mode==='state'?5:mode==='speculative'?4:M.grammar(prefs().grammar).length-1;
const token=(x,cls='')=>`<span class="gen-token ${cls}">${x}</span>`;
function stateView(){
 const type=prefs().state,d=M.state(step),recurrent=type!=='attention';
 let graph=`<p class="gen-label">依次处理四个输入 · 当前 ${d.n} / 4</p><div class="gen-row">${M.inputs.map((x,i)=>token(`t${i+1}`,i<d.n?'accepted':'pending')).join('')}</div>`;
 if(type!=='recurrent')graph+=`<div class="gen-band"><h3>Attention 层 · 每个位置保存 K / V</h3><div class="memory-slots">${M.inputs.map((x,i)=>`<div class="memory-slot ${i<d.n?'filled':''}">${i<d.n?'<div class="kv-pair"><i>K</i><i>V</i></div>':'·'}<small>t${i+1}</small></div>`).join('')}</div></div>`;
 if(recurrent)graph+=`<div class="gen-band"><h3>递推层 · 更新活动状态</h3><div class="state-flow">${token(d.restored?'快照':step?`x=${M.inputs[step-1]}`:'等待',d.restored?'bonus':step?'':'pending')}<span class="state-arrow ${step?'moving':''}" aria-hidden="true"></span><div class="state-box"><strong>${d.value}</strong><small>当前状态 s</small></div></div><div class="checkpoint ${d.restored?'restoring':''}">${d.restored?'恢复到 t2 的状态':'t2 检查点'}<b>${d.checkpoint===null?'尚未保存':d.checkpoint}</b></div></div>`;
 if(type==='attention')graph+='<p class="gen-result">有效 token 位置随前缀增长</p>';
 const caption=d.restored?(recurrent?'回到 t2 不能只丢掉后两个 token 的标签；这里恢复已保存的 s=4。混合模型还要同步调整 Attention 的有效前缀。':'教学回退到 t2：有效 KV 只覆盖前两个位置。图表示有效范围，不表示池的物理容量同步缩小。'):step===0?'选择代表结构，再逐步输入。模型结构决定历史以什么形式被保存，不能靠切换后端任意互换。':recurrent?`本步用 x=${M.inputs[step-1]} 更新同一个状态槽：s = 0.5 × ${d.history[step-1]} + ${M.inputs[step-1]} = ${d.value}。这是教学递推式，不是 GDN 的计算公式。`:'每处理一个输入，新增对应位置的 K/V。历史记录保持可按位置访问；本图省略 head、层和页映射。';
 return {graph,title:'历史变长，究竟增加什么状态？',label:d.restored?'教学回退到 t2':step?`处理 t${step}`:'选择状态形态',caption,why:'Attention KV 与递推层的 conv / temporal 状态是不同对象。固定大小的活动状态不等于整个模型内存恒定：层数、请求数、检查点、投机中间状态及混合 Attention 都另占空间。回退示例假设在 t2 保存检查点；真实投机提交还可能使用按接受位置 scatter、ReplaySSM 或其他专用路径。',source:d.restored?'stateCommit':type==='attention'?'kv':type==='hybrid'?'hybrid':'state'};
}
function speculative(){
 const d=M.speculate(prefs().accept),a=d.accepted;
 let graph=`<p class="gen-label">${step<2?'已有输出末尾 r · 它的目标 KV 尚待本轮计算':'验证根 r · 本轮已计算它与候选的目标 KV'}</p>`;
 if(step<2){graph+=`<div class="gen-row">${token('r','bonus')}<span aria-hidden="true">→</span>${d.draft.map(x=>token(step?x:'·',step?'':'pending')).join('')}</div><p class="gen-result">${step?'Draft 提议 a b c d':'等待草稿'}</p>`;}
 else{
  graph+='<p class="gen-note">每列上方是验证输入，下方是目标模型预测的后继。</p><div class="verify-grid">';
  graph+=d.verify.map((x,i)=>`<div class="verify-cell ${step>=3&&i>a?'invalid':''}">${token(x,i===0?'bonus':step>=3?(i<=a?'accepted':'rejected'):'')}<span class="verify-arrow">↓</span>${token(d.predictions[i],step>=3?(i<a?'accepted':i===a?'bonus':'unused'):'')}<small>${step>=3?(i<a?'接受草稿':i===a?'补出 z':'错误前缀'):'因果验证'}</small></div>`).join('')+'</div>';
 }
 if(step===4)graph+=`<div class="gen-band"><h3>本轮正式输出 · ${a} 个草稿 + 1 个补充 token</h3><div class="gen-row output-tokens">${d.output.map((x,i)=>token(x,i===a?'bonus':'accepted')).join('')}</div></div><p class="gen-label">本轮新增的有效目标 KV</p><div class="gen-row valid-kv">${d.validKV.map(x=>token(x,'accepted')).join('')}${token('z 待计算','pending')}</div>`;
 const captions=['上轮生成了 r，但它自身的 K/V 尚未进入目标模型缓存；它将成为本轮验证的根。','草稿只提出候选，不能直接当作正式输出。topk=1 的教学链中，根 r 加四个候选构成五个验证位置。','目标模型一次计算五行，每行仍只能看合法前缀。并行验证没有取消自回归的因果关系。',a===4?'四个候选全部通过；最后一行还产生新的 z。':`前 ${a} 个候选通过，随后目标模型预测 z。分歧之后的行基于错误前缀，即使计算过也不能续接为正式输出。`,'输出是已接受草稿加 z；有效目标 KV 是旧根 r 加已接受草稿。两个长度相同，内容却错开一位，z 要在后续输入时才有自己的 KV。'];
 return {graph,title:'草稿可以先算，提交必须沿正确前缀',label:['上一轮的种子','Draft · 提议','Verify · 目标计算','Accept · 找到边界','Commit · 更新有效状态'][step],caption:captions[step],why:'限定单请求、链式 topk=1、贪心验证，无 EOS / 长度截断、grammar、penalty、Overlap 或模拟接受长度。分歧位置由教学控件指定，不是模型测量。随机采样不能直接套用相等判断；树模式还有祖先与兄弟关系。未接受位置可以位于已分配空间中，图不把无效 KV 等同于立即释放物理页。',source:step<2?'draft':step===2?'verify':'accept'};
}
function grammar(){
 const d=M.grammar(prefs().grammar)[step],masked=d.phase!=='scores';
 const route=prefs().grammar==='yes'?['起点','y','yes','结束']:['起点','no','结束'];
 const current=d.ended?'结束':d.after||'起点';
 const graph=`<div class="grammar-path">${route.map((x,i)=>`${i?'<b>→</b>':''}<span class="${x===current?'current':''}">${x}</span>`).join('')}</div><p class="gen-label">本轮 mask 依据前缀「${d.prefix||'空'}」</p><div class="vocab-bars">${M.vocab.map((t,i)=>`<div class="vocab-row ${masked&&!d.valid[i]?'masked':''} ${d.token===t?'selected':''}" data-token="${t}"><span>${t}</span><div class="vocab-track"><div class="vocab-fill" style="--w:${d.scores[i]*10}%"></div></div><span>${masked&&!d.valid[i]?'−∞':d.scores[i]}${d.token===t?' ✓':''}</span></div>`).join('')}</div><p class="grammar-result">${d.ended?'输出完成':`当前已输出：${d.after||'（空）'}`}${d.token&&!d.ended?` · 接受 ${d.token}`:''}</p>`;
 return {graph,title:'分数最高，也可能被规则排除',label:d.phase==='scores'?'模型给出分数':d.phase==='mask'?'规则筛选候选':d.ended?'接受 EOS':'接受 token 并推进规则',caption:d.phase==='scores'?'固定五个教学 token，目标语言只有 yes 或 no。最高分的 ! 会破坏规则；先看原始分数，再应用 mask。':d.phase==='mask'?'当前前缀决定哪些 token 可以继续。被禁止项置为 −∞，随后贪心选择只在合法项中比较分数。':d.ended?'只有完整的 yes 或 no 才允许 EOS。词表 token 可以包含多个字符，规则不能简单按单字符判断。':`接受 ${d.token} 后，matcher 推进到「${d.after}」。下一轮需要按新前缀重新计算 mask。`,why:'分数是教学 logits，不是概率；词表和分词方式为人工缩小示例，不复刻某个 tokenizer。这里只演示普通约束生成与贪心选择。XGrammar copy 复用编译上下文，却创建新的 matcher；两个请求不能共享已接受前缀。格式约束不保证答案事实正确。',source:d.phase==='scores'?'sample':d.phase==='mask'?'mask':'advance'};
}
function stop(){clearInterval(timer);timer=null;$('#gen-play').textContent='播放';$('#gen-play').setAttribute('aria-pressed','false');}
function render(){const d=({state:stateView,speculative,grammar})[mode]();$('#gen-title').textContent=d.title;$('#gen-scene').innerHTML=d.graph;$('#gen-scene').dataset.mode=mode;$('#gen-scene').dataset.step=step;$('#gen-caption').textContent=d.caption;$('#gen-why').textContent=d.why;$('#gen-source').href=M.url(d.source);$('#gen-source').textContent=`源码 · ${M.sources[d.source][2]} ↗`;$('#gen-status').textContent=`${step+1} / ${last()+1} · ${d.label}`;$('#gen-previous').disabled=step===0;$('#gen-next').disabled=step===last();if(step===last())stop();}
function reset(){stop();step=0;render();}
lab.hidden=false;lab.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;lab.querySelectorAll('[data-mode]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));lab.querySelectorAll('[data-settings]').forEach(x=>x.hidden=x.dataset.settings!==mode);reset();});lab.querySelectorAll('select').forEach(x=>x.onchange=reset);
$('#gen-previous').onclick=()=>{stop();if(step>0)step--;render();};$('#gen-next').onclick=()=>{stop();if(step<last())step++;render();};$('#gen-reset').onclick=reset;$('#gen-play').onclick=()=>{if(timer){stop();return;}if(step===last())step=0;render();$('#gen-play').textContent='暂停';$('#gen-play').setAttribute('aria-pressed','true');timer=setInterval(()=>{step++;render();},2400);};document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});render();
})();
