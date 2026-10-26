/* V16.1: complete Pre-Norm layer context for the one-head Attention player.
 * The full-layer chain is symbolic: x -> xhat = Norm(x) -> A = Attention(xhat)
 * -> x1 = x + A -> xhat1 = Norm(x1) -> B = MLP(xhat1) -> y = x1 + B.
 * Local Attention toy numbers stay in a separate explanation; they are never
 * spliced into the full-layer chain. The layer view advances one operation at
 * a time with a JS-owned progress value, so pausing freezes the visible motion. */
(function(){
'use strict';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>Math.abs(x-Math.round(x))<1e-9?String(Math.round(x)):Number(x).toFixed(2);
const vec=v=>'['+v.map(fmt).join(', ')+']';
const pct=x=>(x*100).toFixed(1)+'%';
const clamp=n=>Math.max(0,Math.min(1,n));
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
const STAGES=[
 {label:'x',title:'输入 x：这一层从哪里开始',kicker:'来源 · 上一层 / Embedding',opName:'读取输入',resultName:'x',duration:0,
  pending:'x 是本层输入；下一步做 Norm。'},
 {label:'Norm',title:'先归一化：x̂ = Norm(x)',kicker:'操作 · RMSNorm',opName:'RMSNorm(x)',resultName:'x̂',duration:900,
  pending:'完成 RMSNorm 后出现 x̂。'},
 {label:'Attention',title:'Attention：A = Attention(x̂)',kicker:'操作 · 单头 Attention',opName:'Attention(x̂)',resultName:'A',duration:1200,
  pending:'完成 Attention 后出现 A。'},
 {label:'+x',title:'残差 1：x₁ = x + A',kicker:'操作 · 残差相加',opName:'x + A',resultName:'x₁',duration:900,
  pending:'完成残差相加后出现 x₁。'},
 {label:'Norm',title:'再归一化：x̂₁ = Norm(x₁)',kicker:'操作 · RMSNorm',opName:'RMSNorm(x₁)',resultName:'x̂₁',duration:900,
  pending:'完成 RMSNorm 后出现 x̂₁。'},
 {label:'MLP',title:'MLP：B = MLP(x̂₁)',kicker:'操作 · SwiGLU',opName:'MLP(x̂₁)',resultName:'B',duration:1300,
  pending:'完成 MLP 后出现 B。'},
 {label:'+x₁',title:'残差 2：y = x₁ + B',kicker:'操作 · 残差相加',opName:'x₁ + B',resultName:'y',duration:900,
  pending:'完成残差相加后出现 y。'},
 {label:'下一层',title:'y 离开本层，进入下一层',kicker:'去向 · 同一前向继续',opName:'传递 y',resultName:'下一层',duration:800,
  pending:'完成传递后进入下一层。'}
];
const OBJECTS=[
 {stage:0,name:'x',role:'输入'},
 {stage:1,name:'x̂',role:'Norm(x)'},
 {stage:2,name:'A',role:'Attention(x̂)'},
 {stage:3,name:'x₁',role:'x + A'},
 {stage:4,name:'x̂₁',role:'Norm(x₁)'},
 {stage:5,name:'B',role:'MLP(x̂₁)'},
 {stage:6,name:'y',role:'x₁ + B'},
 {stage:7,name:'下一层',role:'去向'}
];

function formula(text){return `<p class="attl-formula">${text}</p>`;}
function note(text){return `<p class="attl-note">${text}</p>`;}
function obj(name,role,meta='',cls=''){
 return `<div class="attl-obj ${cls}" data-object="${esc(name)}"><span class="attl-obj-name">${esc(name)}</span><span class="attl-obj-role">${esc(role)}</span>${meta?`<small>${esc(meta)}</small>`:''}</div>`;
}
function chip(label,value,cls=''){
 return `<span class="attl-chip ${cls}"><small>${esc(label)}</small><b>${esc(value)}</b></span>`;
}
function relayHTML(label,from,to,token){
 return `<div class="attl-relay" data-attl-relay style="--p:0">
  <div class="attl-relay-labels"><span>${esc(from)}</span><span>${esc(to)}</span></div>
  <div class="attl-relay-track" role="progressbar" aria-label="${esc(label)}操作进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
   <i class="attl-relay-fill"></i><span class="attl-relay-token" aria-hidden="true">${esc(token)}</span>
  </div>
  <div class="attl-relay-meta"><span>${esc(label)}</span><b data-attl-progress>0%</b></div>
 </div>`;
}
function mergeHTML(left,right,symbol='+'){
 return `<div class="attl-merge">${left}<span class="attl-plus">${symbol}</span>${right}</div>`;
}
function localAttentionHTML(s){
 const d=s.data,q=d.query,focus=s.focus;
 const qkv=d.inputs.map((input,i)=>{
  const read=i<d.allowed,cur=i===q;
  return `<div class="attl-local-row${cur?' is-query':''}${read?'':' is-masked'}" role="row">
   <b>p${i+1}</b>
   <span class="attl-qkv-cell"><i>Q</i><code>${vec(d.Q)}</code></span>
   <span class="attl-qkv-cell"><i>K</i><code>${vec(d.keys[i])}</code></span>
   <span class="attl-qkv-cell"><i>V</i><code>${vec(d.values[i])}</code></span>
   <small class="attl-local-status">${cur?'当前 Query':read?'可读':'未来 · 遮罩'}</small>
  </div>`;
 }).join('');
 const weights=d.inputs.map((input,i)=>{
  const read=i<d.allowed,cur=i===q,w=read?d.weights[i]:0;
  return `<div class="attl-wrow${cur?' is-query':''}${read?'':' is-masked'}">
   <b>p${i+1}</b>
   <span class="attl-score">${read?'s='+fmt(d.scores[i]):'s=−∞'}</span>
   <span class="attl-track"><i style="--w:${(w*100).toFixed(2)}%"></i></span>
   <strong>${pct(w)}</strong>
   <small>${read?'w×V '+vec(d.contributions[i]):'不进分母'}</small>
  </div>`;
 }).join('');
 const focusNote=focus==='mask'?'当前聚焦遮罩：未来位置 w=0，不进入分母。':focus==='qkv'?'当前聚焦 Q/K/V 分工：Q 只来自当前行，K/V 每个可读位置各一份。':focus==='weights'?'当前聚焦权重：先有 s 与 e^s，再除分母得到 w，最后用 w 混合 V。':'当前为自动镜头；局部例的完整结果不随层视图的操作进度改变。';
 return `<section class="attl-local" aria-label="局部手算例，独立于完整层数值链">
  <div class="attl-local-head"><div><p class="attl-local-kicker">局部手算例 · 独立说明</p><h5>这一页的 Attention 子层怎样手算</h5></div><span class="attl-chip is-muted">完整层 A 仍是符号</span></div>
  <p class="attl-local-note">为手算，WQ=WK=I、WV=2I，并把 Norm 折叠掉；下面直接用本页 x 作为 Q/K，V=2x。这组数字只解释局部 Attention，不是上方 x̂ → A 的数值。</p>
  <div class="attl-local-qkv" role="table" aria-label="局部 Q/K/V 与可读位置">
   <div class="attl-local-row is-head" role="row"><b>位置</b><span>Q</span><span>K</span><span>V</span><span>可读</span></div>
   ${qkv}
  </div>
  <div class="attl-local-weights" aria-label="局部分数与权重">${weights}</div>
  <div class="attl-local-result"><span>局部例完整结果</span><strong>m = Σw·V = ${vec(d.mixed)}</strong><small>Σe^s=${d.denominator.toFixed(2)} · Σw=${pct(d.weightSum)}</small></div>
  <p class="attl-local-note">这是局部例的完整结果；本地时间线可能停在其间。完整层的 A 不从这里取数。${focusNote}</p>
 </section>`;
}

function stageContent(s,stage){
 const d=s.data,q=d.query;
 switch(stage){
  case 0:
   return {
    source:`<p class="attl-label">来源 · 上一层 / Embedding</p>${obj('x','本层输入','4 个位置，每个位置一个向量','is-source')}${note('x 是这一层第一次出现的对象；后面的残差还要复用它。')}`,
    op:`<div class="attl-op-copy"><strong>读取输入 x</strong><p>本层从 x 开始。Prefill 可以一次送入多个位置；Decode 通常只送入一个新位置，但入口相同。</p></div>${relayHTML('读取','上一层','x','→')}${formula('同一套层权重处理每个位置；位置 p 是序列位置，不是层号。')}`,
    result:`${obj('x','本层输入已就位','不是新算出的结果；是这一层的起点','is-source')}${note('下一步：对 x 做 RMSNorm。')}`,
    note:'x 进入本层后先归一化；残差会在 Attention 之后把同一个 x 加回来。',
    local:'',pending:STAGES[0].pending,resultName:'x',duration:0
   };
  case 1:
   return {
    source:`<p class="attl-label">来源 · 本层输入</p>${obj('x','同一个对象','4 个位置；逐位置处理','is-source')}`,
    op:`<div class="attl-op-copy"><strong>RMSNorm：每个位置独立归一化</strong><p>对每个位置单独求均方根并缩放；不跨位置混合。</p></div>${relayHTML('RMSNorm','x','x̂','N')}${formula('x̂_p = x_p / √(mean(x_p²) + ε) · γ₁')}${note('γ₁ 是 Norm1 自己的可学习参数；Norm2 的 γ₂ 另行学习，不共享。这里只写符号，不代入数值。')}`,
    result:`${obj('x̂','结果 · 归一化后的 x','符号对象；形状不变','is-ok')}${note('完整层里，下一步的 Q/K/V 来自 x̂ = Norm(x)，不是原始 x。')}`,
    note:'本页局部手算例把 Norm 与投影折叠掉；因此这里不展示 x̂ 的数值。完整层只保留 x → x̂ 的符号关系。',
    local:'',pending:STAGES[1].pending,resultName:'x̂',duration:STAGES[1].duration
   };
  case 2:
   return {
    source:`<p class="attl-label">来源 · Norm(x)</p>${obj('x̂','Attention 的输入','Q/K/V 都从 x̂ 投影得到','is-source')}${note('不是从原始 x 直接得到 Q/K/V。')}`,
    op:`<div class="attl-op-copy"><strong>A = Attention(x̂)</strong><p>Q=x̂W_Q，K=x̂W_K，V=x̂W_V；因果掩码只让当前位置读取可读位置。多头输出先拼接，再过 W_O。</p></div>${relayHTML('Attention','x̂','A','Attn')}${formula('A = [softmax(QKᵀ/√d + mask) · V] W_O')}${note('单头时括号内结果直接过 W_O 得 A；这里只保留符号对象 A，不写完整层的 Attention 数值。')}`,
    result:`${obj('A','结果 · Attention 子层输出','符号对象；不是整层输出','is-ok')}${note('A 还要加回 x，得到 x₁。')}`,
    note:'A 是 Attention(Norm(x)) 的符号结果。下方“局部手算例”的 m 是另一套独立数字，不能替代 A。',
    local:localAttentionHTML(s),pending:STAGES[2].pending,resultName:'A',duration:STAGES[2].duration
   };
  case 3:
   return {
    source:`<p class="attl-label">来源 · 残差的两边</p>${mergeHTML(obj('x','旁路来源 · 保留','残差要加回同一个 x','is-residual'),obj('A','子层输出','来自 Attention(Norm(x))','is-ok'))}${note('残差是逐元素相加，不是拼接；x 没有被 A 替换。')}`,
    op:`<div class="attl-op-copy"><strong>x₁ = x + A</strong><p>把 Attention 的输出加回同一个 x。</p></div>${relayHTML('残差相加','x + A','x₁','+')}${formula('x₁ = x + Attention(Norm(x))')}`,
    result:`${obj('x₁','结果 · 第一个子层后的表示','新对象；x 仍然存在','is-ok')}${note('x₁ 会进入第二个 Norm；第二个残差以后要复用它。')}`,
    note:'第二个残差保留的是这个 x₁，不是最初的 x。',
    local:'',pending:STAGES[3].pending,resultName:'x₁',duration:STAGES[3].duration
   };
  case 4:
   return {
    source:`<p class="attl-label">来源 · 第一个残差的结果</p>${obj('x₁','不是最初的 x','输入身份已经换成 x₁','is-source')}${note('把 x₁ 的每一行按位置单独归一化；位置之间仍不混合。')}`,
    op:`<div class="attl-op-copy"><strong>RMSNorm(x₁)</strong><p>与第一个 Norm 是同一类操作，但输入已经换成了 x₁。</p></div>${relayHTML('RMSNorm','x₁','x̂₁','N')}${formula('x̂₁_p = x₁_p / √(mean(x₁_p²) + ε) · γ₂')}${note('γ₂ 是 Norm2 自己的可学习参数，与 Norm1 的 γ₁ 分开学习；这里不代入数值。')}`,
    result:`${obj('x̂₁','结果 · 归一化后的 x₁','符号对象；形状不变','is-ok')}${note('下一步送入 MLP。')}`,
    note:'这里的重点是输入身份已经变成 x₁，而不是把最初的 x 再归一化一次。',
    local:'',pending:STAGES[4].pending,resultName:'x̂₁',duration:STAGES[4].duration
   };
  case 5:
   return {
    source:`<p class="attl-label">来源 · Norm(x₁)</p>${obj('x̂₁','逐位置进入 MLP','每个位置独立变换','is-source')}${note('MLP 不做 Q/K 匹配，也不混合其他位置。')}`,
    op:`<div class="attl-op-copy"><strong>B = MLP(x̂₁)</strong><p>SwiGLU 的 gate、up、down 三条投影都逐位置作用。</p></div>${relayHTML('MLP','x̂₁','B','MLP')}<div class="attl-mlp"><div class="attl-mlp-lane"><small>gate</small><span>x̂₁ · Wg</span><i>→</i><b>g</b></div><div class="attl-mlp-lane"><small>up</small><span>x̂₁ · Wu</span><i>→</i><b>u</b></div><div class="attl-mlp-mix"><span>z = SiLU(g) ⊙ u</span><small>逐元素相乘</small></div><div class="attl-mlp-lane"><small>down</small><span>z · Wd</span><i>→</i><b>B</b></div></div>`,
    result:`${obj('B','结果 · MLP 子层输出','符号对象；不是整层输出','is-ok')}${note('B 还没有加回 x₁。')}`,
    note:'SwiGLU 的中间宽度通常远大于输入；这里只画结构，不伪造维度或数值。',
    local:'',pending:STAGES[5].pending,resultName:'B',duration:STAGES[5].duration
   };
  case 6:
   return {
    source:`<p class="attl-label">来源 · 第二个残差的两边</p>${mergeHTML(obj('x₁','旁路来源 · 保留','第二个残差要复用它','is-residual'),obj('B','子层输出','来自 MLP(Norm(x₁))','is-ok'))}${note('第二个旁路从 x₁ 取，不从最初的 x 取。')}`,
    op:`<div class="attl-op-copy"><strong>y = x₁ + B</strong><p>把 MLP 的输出加回同一个 x₁。</p></div>${relayHTML('残差相加','x₁ + B','y','+')}${formula('y = x₁ + MLP(Norm(x₁))')}`,
    result:`${obj('y','结果 · 本层输出','x₁ 与 B 逐元素相加','is-ok')}${note('y 离开本层，进入下一层。')}`,
    note:'把第二个残差画成“加原 x”会丢失第一个子层的更新；这里保留 x₁ 的身份。',
    local:'',pending:STAGES[6].pending,resultName:'y',duration:STAGES[6].duration
   };
  default:
   return {
    source:`<p class="attl-label">来源 · 本层输出</p>${obj('y','这一层已经完成','同一前向继续','is-source')}`,
    op:`<div class="attl-op-copy"><strong>进入下一层</strong><p>下一层会重新做 Norm → Attention → 残差 → Norm → MLP → 残差；不是下一个 token，也不是跳过 MLP。</p></div>${relayHTML('传递','y','下一层','→')}${note('Decode 的下一 token 仍要先走完所需各层。')}`,
    result:`${obj('下一层','去向','重新开始完整层','is-next')}${note('一层只是一次前向中的一段；生成一个 token 需要走完所需各层，再由输出头与采样决定。')}`,
    note:'区别在于这次送进来多少位置、能读取哪些历史 K/V，而不是“Prefill 只走一半、Decode 跳过 MLP”。',
    local:'',pending:STAGES[7].pending,resultName:'下一层',duration:STAGES[7].duration
   };
 }
}

function mount(section,api){
 if(!section||section.dataset.attentionLayerMounted)return;
 section.dataset.attentionLayerMounted='true';
 const theater=section.querySelector('.motion-theater');
 if(!theater)return;
 const controls=section.querySelector('.motion-controls');
 const dock=document.createElement('div');
 dock.className='attl-dock';
 dock.innerHTML=`<div class="attl-dock-head">
   <div><p class="attl-kicker">这一页在完整 Pre-Norm 层里的位置</p>
   <p class="attl-formula">x₁ = x + Attention(Norm(x)) · y = x₁ + MLP(Norm(x₁))</p></div>
   <button type="button" class="attl-open" aria-expanded="false" aria-controls="attl-view">进入完整层 <span aria-hidden="true">→</span></button>
  </div>
  <div class="attl-dock-body">
   <div class="attl-mini" aria-label="完整层结构：x 归一化后进入 Attention，加回 x 得到 x₁，再归一化进入 MLP，加回 x₁ 得到 y">
    ${['x','Norm','Attention','+x','x₁','Norm','MLP','+x₁','y'].map((label,i)=>`${i?'<span class="attl-arr" aria-hidden="true">→</span>':''}<span class="attl-node${label==='Attention'?' is-active':''}">${label}${label==='Attention'?'<small>本页放大</small>':''}</span>`).join('')}
   </div>
   <p class="attl-dock-state" aria-live="polite"></p>
  </div>`;
 theater.parentNode.insertBefore(dock,theater);
 const view=document.createElement('section');
 view.className='attl-view';
 view.id='attl-view';
 view.hidden=true;
 view.setAttribute('role','region');
 view.setAttribute('aria-label','完整 Pre-Norm 层：来源、操作与结果');
 view.innerHTML=`<header class="attl-view-head">
   <div><p class="attl-kicker">完整 Pre-Norm 层 · 当前局部帧已暂停</p>
   <h3>Attention 在这一层里的位置</h3>
   <p class="attl-view-sub">同一 Query 的来源 → 操作 → 结果；对象只在对应操作完成后出现，看完回到原来的暂停帧。</p></div>
   <button type="button" class="attl-close" aria-label="返回当前 Attention 暂停帧">← 返回当前 Attention 帧</button>
  </header>
  <div class="attl-statebar" aria-live="polite"></div>
  <nav class="attl-pipeline" aria-label="完整层步骤">
   ${STAGES.map((st,i)=>`<button type="button" data-attl-stage="${i}" aria-pressed="false" aria-label="第 ${i+1} 步 ${esc(st.label)}：${esc(st.title)}"><small>0${i+1}</small><strong>${esc(st.label)}</strong></button>`).join('')}
  </nav>
  <div class="attl-stage-head"><p class="attl-stage-kicker"></p><h4 class="attl-stage-title"></h4><span class="attl-stage-state" data-state="idle">待运行</span></div>
  <div class="attl-objects" aria-label="本层已经产生的对象"></div>
  <div class="attl-work">
   <div class="attl-grid">
    <article class="attl-col is-source"><h5><span>1</span>来源 · source</h5><div data-attl-slot="source"></div></article>
    <div class="attl-arrow" aria-hidden="true">→</div>
    <article class="attl-col is-op"><h5><span>2</span>操作 · operation</h5><div data-attl-slot="op"></div></article>
    <div class="attl-arrow" aria-hidden="true">→</div>
    <article class="attl-col is-result"><h5><span>3</span>结果 · result</h5><div data-attl-slot="result"></div></article>
    <div class="attl-local-slot" data-attl-slot="local" hidden></div>
   </div>
   <p class="attl-stage-note" aria-live="polite"></p>
  </div>
  <footer class="attl-controls">
   <button type="button" data-attl-action="play">播放完整层</button>
   <button type="button" data-attl-action="prev">上一步</button>
   <button type="button" data-attl-action="next">下一步</button>
   <button type="button" data-attl-action="start">从输入看</button>
   <span class="attl-count"></span>
   <button type="button" class="attl-jump" data-attl-action="attention">定位 Attention</button>
  </footer>`;
 theater.append(view);
 const dockState=dock.querySelector('.attl-dock-state');
 const openBtn=dock.querySelector('.attl-open');
 const closeBtn=view.querySelector('.attl-close');
 const pageBtns=[...view.querySelectorAll('[data-attl-stage]')];
 const slots={source:view.querySelector('[data-attl-slot=source]'),op:view.querySelector('[data-attl-slot=op]'),result:view.querySelector('[data-attl-slot=result]'),local:view.querySelector('[data-attl-slot=local]')};
 const stageKicker=view.querySelector('.attl-stage-kicker'),stageTitle=view.querySelector('.attl-stage-title'),stageState=view.querySelector('.attl-stage-state'),stageNote=view.querySelector('.attl-stage-note'),objectsEl=view.querySelector('.attl-objects'),work=view.querySelector('.attl-work'),count=view.querySelector('.attl-count'),statebar=view.querySelector('.attl-statebar'),playBtn=view.querySelector('[data-attl-action=play]'),prevBtn=view.querySelector('[data-attl-action=prev]'),nextBtn=view.querySelector('[data-attl-action=next]');
 let state=null,stage=0,doneMax=0,progress=0,current=null,open=false,lastDock='',playing=false,motionActive=false;
 let raf=null,playTimer=null,runStart=0,runFrom=0,runTo=1,runDur=0;
 function currentState(){try{return api.getState();}catch(e){return null;}}
 function renderDock(){
  const s=currentState();if(!s||!s.data)return;
  const d=s.data;
  const html=`<span class="attl-chip is-query">Query p${d.queryPosition}</span><span class="attl-chip">聚焦 ${esc(s.focusLabel)}</span><span class="attl-chip ${s.playing?'is-live':'is-paused'}">${s.playing?'播放中 · 进入时会暂停':'已暂停'}</span><span class="attl-chip is-muted">进入后从 x 开始</span>`;
  if(html!==lastDock){lastDock=html;dockState.innerHTML=html;}
  openBtn.setAttribute('aria-label',`进入完整层，带着 Query p${d.queryPosition} 与当前暂停位置`);
 }
 function sync(){renderDock();if(open){const s=currentState();if(s&&s.data){state=s;renderState();}}}
 function cancelTimers(){
  if(raf){cancelAnimationFrame(raf);raf=null;}
  if(playTimer){clearTimeout(playTimer);playTimer=null;}
  motionActive=false;
 }
 function pauseMotion(){
  cancelTimers();playing=false;updateStageState();renderControls();
 }
 function renderState(){
  if(!state||!state.data)return;
  const d=state.data;
  statebar.innerHTML=`<b>局部帧</b><span class="is-paused">已暂停</span><span>Query p${d.queryPosition}</span><span>聚焦 ${esc(state.focusLabel)}</span><span>${state.preview?'完整状态快照':'局部时间线'}</span><span class="is-note">层视图不推进它</span>`;
 }
 function updateStageState(){
  const p=Math.round(progress*100);
  let text='待运行',key='idle';
  if(progress>=1){text='已完成';key='done';}
  else if(motionActive){text=`运行中 ${p}%`;key='running';}
  else if(progress>0){text=`已暂停 ${p}%`;key='paused';}
  stageState.textContent=text;
  stageState.dataset.state=key;
 }
 function updateProgress(p){
  progress=clamp(p);
  const relay=slots.op.querySelector('[data-attl-relay]');
  if(relay){
   relay.style.setProperty('--p',progress.toFixed(4));
   const track=relay.querySelector('[role=progressbar]');
   if(track)track.setAttribute('aria-valuenow',String(Math.round(progress*100)));
   const label=relay.querySelector('[data-attl-progress]');
   if(label)label.textContent=Math.round(progress*100)+'%';
  }
  updateStageState();
 }
 function renderResult(){
  if(!current)return;
  if(progress>=1){
   slots.result.innerHTML=current.result;
   slots.result.dataset.state='done';
  }else{
   slots.result.innerHTML=`<div class="attl-pending"><span class="attl-pending-mark" aria-hidden="true">?</span><b>结果尚未产生</b><small>${esc(current.pending)}</small></div>`;
   slots.result.dataset.state='pending';
  }
 }
 function renderObjects(){
  const shown=OBJECTS.filter(o=>o.stage<=doneMax||o.stage===0);
  const residual=(name)=>(name==='x'&&stage===3)||(name==='x₁'&&stage===6);
  let html=shown.map((o,i)=>{
   const isResidual=residual(o.name);
   return `${i?'<span class="attl-object-arrow" aria-hidden="true">→</span>':''}<span class="attl-object${isResidual?' is-residual':''}" data-object="${esc(o.name)}"><b>${esc(o.name)}</b><small>${esc(o.role)}</small>${isResidual?'<i>旁路</i>':''}</span>`;
  }).join('');
  if(progress<1&&stage>0)html+=`<span class="attl-object-arrow" aria-hidden="true">→</span><span class="attl-object is-pending"><b>?</b><small>待产生 ${esc(STAGES[stage].resultName)}</small></span>`;
  objectsEl.innerHTML=html;
 }
 function renderPipeline(){
  pageBtns.forEach((b,i)=>{
   const active=i===stage;
   b.setAttribute('aria-pressed',String(active));
   b.classList.toggle('is-done',i<=doneMax&&!active);
   b.classList.toggle('is-current',active);
   b.classList.toggle('is-complete',i<=doneMax&&active);
  });
 }
 function updateCount(){count.textContent=`第 ${stage+1} / ${STAGES.length} 步`;}
 function renderControls(){
  const active=motionActive||playing;
  playBtn.textContent=active?'暂停':(progress<1?'继续播放':(stage>=STAGES.length-1?'再走一遍':'播放完整层'));
  playBtn.setAttribute('aria-pressed',String(active));
  nextBtn.textContent=progress<1?'完成这一步':(stage>=STAGES.length-1?'已到最后一步':'下一步');
  nextBtn.disabled=motionActive||(progress>=1&&stage>=STAGES.length-1);
  prevBtn.disabled=stage===0&&progress>=1;
  updateCount();
 }
 function renderStage(){
  if(!current)return;
  stageKicker.textContent=STAGES[stage].kicker;
  stageTitle.textContent=STAGES[stage].title;
  slots.source.innerHTML=current.source;
  slots.op.innerHTML=current.op;
  slots.local.innerHTML=current.local||'';
  slots.local.hidden=!current.local;
  stageNote.textContent=current.note;
  view.dataset.stage=String(stage);
  renderResult();renderObjects();renderPipeline();renderControls();
 }
 function selectStage(n,nextProgress=0){
  cancelTimers();playing=false;
  stage=Math.max(0,Math.min(STAGES.length-1,n));
  doneMax=Math.max(0,stage-1);
  progress=stage===0?1:clamp(nextProgress);
  if(progress>=1)doneMax=Math.max(doneMax,stage);
  current=stageContent(state,stage);
  renderStage();
  updateProgress(progress);
  renderControls();
  work.scrollTop=0;
 }
 function commitCurrent(){
  cancelTimers();
  progress=1;doneMax=Math.max(doneMax,stage);
  updateProgress(1);
  renderResult();renderObjects();renderPipeline();updateCount();renderControls();
  if(playing){
   if(stage>=STAGES.length-1){playing=false;renderControls();return;}
   playTimer=setTimeout(()=>{
    playTimer=null;
    if(!open||!playing)return;
    selectStage(stage+1,0);
    playing=true;
    runCurrent(true);
   },650);
  }
 }
 function tick(ts){
  if(!open){raf=null;motionActive=false;return;}
  if(!runStart)runStart=ts;
  const t=Math.min(1,(ts-runStart)/runDur);
  updateProgress(runFrom+(runTo-runFrom)*t);
  if(t>=1){raf=null;motionActive=false;commitCurrent();}
  else raf=requestAnimationFrame(tick);
 }
 function runCurrent(autoplay){
  playing=autoplay;
  if(progress>=1){commitCurrent();return;}
  if(reduced.matches){progress=1;commitCurrent();return;}
  motionActive=true;
  runFrom=progress;runTo=1;runDur=Math.max(180,STAGES[stage].duration*(runTo-runFrom));runStart=0;
  updateStageState();renderControls();
  raf=requestAnimationFrame(tick);
 }
 function startPlayback(){
  if(playing||motionActive){pauseMotion();return;}
  if(stage>=STAGES.length-1&&progress>=1)selectStage(0,1);
  if(progress>=1){
   if(stage>=STAGES.length-1){playing=false;renderControls();return;}
   selectStage(stage+1,0);
  }
  playing=true;
  runCurrent(true);
 }
 function stepOnce(){
  if(motionActive)return;
  cancelTimers();playing=false;
  if(progress>=1){
   if(stage>=STAGES.length-1)return;
   selectStage(stage+1,0);
  }
  runCurrent(false);
 }
 function prevStep(){
  cancelTimers();playing=false;
  selectStage(stage<=0?0:stage-1,1);
 }
 function openView(){
  if(open)return;
  api.pause();
  state=currentState();if(!state||!state.data)return;
  open=true;section.dataset.attentionLayer='open';
  view.hidden=false;openBtn.setAttribute('aria-expanded','true');
  renderState();selectStage(0,1);
  if(controls){controls.setAttribute('aria-hidden','true');controls.inert=true;}
  document.addEventListener('keydown',onKeyCapture,true);
  requestAnimationFrame(()=>closeBtn.focus());
 }
 function closeView(){
  if(!open)return;
  open=false;
  cancelTimers();playing=false;motionActive=false;
  view.hidden=true;section.removeAttribute('data-attention-layer');openBtn.setAttribute('aria-expanded','false');
  document.removeEventListener('keydown',onKeyCapture,true);
  try{view.getAnimations({subtree:true}).forEach(a=>a.cancel());}catch(e){}
  if(controls){controls.removeAttribute('aria-hidden');controls.inert=false;}
  openBtn.focus();
 }
 function onKeyCapture(e){
  if(e.key==='Escape'){e.preventDefault();e.stopPropagation();closeView();}
 }
 openBtn.addEventListener('click',openView);
 closeBtn.addEventListener('click',closeView);
 view.addEventListener('keydown',e=>{
  if(e.key==='Escape'){e.preventDefault();e.stopPropagation();closeView();return;}
  if(e.key===' '||e.key==='ArrowLeft'||e.key==='ArrowRight')e.stopPropagation();
 });
 view.addEventListener('click',e=>{
  const b=e.target.closest('button');if(!b)return;
  if(b.dataset.attlStage!==undefined){cancelTimers();playing=false;selectStage(Number(b.dataset.attlStage),Number(b.dataset.attlStage)===0?1:0);return;}
  if(b.dataset.attlReturn!==undefined){closeView();return;}
  const a=b.dataset.attlAction;
  if(a==='play')startPlayback();
  else if(a==='prev')prevStep();
  else if(a==='next')stepOnce();
  else if(a==='start'){cancelTimers();playing=false;selectStage(0,1);}
  else if(a==='attention'){cancelTimers();playing=false;selectStage(2,0);}
 });
 window.AttentionLayer={mount,sync,open:openView,close:closeView};
 renderDock();
}
window.AttentionLayer={mount,sync(){},open(){},close(){}};
})();
