/* Interactive walkthrough for the one-layer "法国的首都是哪里？" teaching chain.
 * Every displayed number comes from output-chain-model.js; this file only
 * formats, lays out and animates that model. Geometry is drawn from model
 * vectors, and the playback state controls every transition. */
(function(){
'use strict';
const deepRead=document.getElementById('output-chain');
function openDeepRead(){if(deepRead&&location.hash==='#output-chain')deepRead.open=true;}
openDeepRead();
window.addEventListener('hashchange',openDeepRead);
const M=window.OutputChainModel;
if(!M)return;
const roots=[...document.querySelectorAll('[data-output-chain]')];
if(!roots.length)return;

const DURATION=1500;
const HOLD=700;
const NS='http://www.w3.org/2000/svg';
const STEPS=[
 {id:'context',label:'问题与位置'},
 {id:'attention',label:'Attention 汇合'},
 {id:'residual',label:'残差与 Norm'},
 {id:'ffn',label:'SwiGLU 加工'},
 {id:'head',label:'输出头打分'}
];

const clamp=(v,lo=0,hi=1)=>Math.max(lo,Math.min(hi,v));
const smooth=t=>{t=clamp(t);return t*t*(3-2*t);};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(v,d=3)=>Number(v.toFixed(d)).toFixed(d);
const fmt=(v,d=3)=>'['+v.map(x=>num(x,d)).join(', ')+']';
const pct=(v,d=1)=>(v*100).toFixed(d)+'%';
const equation=key=>document.getElementById('equation-'+key).innerHTML;
function vectorMath(vector, decimals){
 const entries=vector.map(value=>`<mn>${num(value,decimals)}</mn>`).join('<mo>,</mo>');
 return `<mrow><mo>[</mo>${entries}<mo>]</mo></mrow>`;
}
function weightMatrix(label, matrix){
 const rows=matrix.map(row=>'<mtr>'+row.map(v=>`<mtd><mn>${num(v,3)}</mn></mtd>`).join('')+'</mtr>').join('');
 return `<math display="block"><msub><mi>W</mi><mi>${label}</mi></msub><mo>=</mo><mrow><mo>[</mo><mtable>${rows}</mtable><mo>]</mo></mrow></math>`;
}

function originPoint(x,y,view){return {x:view.x+x*view.scale,y:view.y-y*view.scale};}
function drawArrow(group,from,to,headSize=7){
 if(!group)return;
 const line=group.querySelector('.oc-arrow-line');
 const head=group.querySelector('.oc-arrow-head');
 if(!line||!head)return;
 line.setAttribute('x1',from.x);line.setAttribute('y1',from.y);
 line.setAttribute('x2',to.x);line.setAttribute('y2',to.y);
 const angle=Math.atan2(to.y-from.y,to.x-from.x);
 const wing=.46;
 const p1={x:to.x-headSize*Math.cos(angle-wing),y:to.y-headSize*Math.sin(angle-wing)};
 const p2={x:to.x-headSize*Math.cos(angle+wing),y:to.y-headSize*Math.sin(angle+wing)};
 head.setAttribute('points',`${to.x},${to.y} ${p1.x},${p1.y} ${p2.x},${p2.y}`);
}
function placeLabel(group,to,dx=0,dy=0){
 const label=group&&group.querySelector('.oc-arrow-label');
 if(!label)return;
 label.setAttribute('x',to.x+dx);label.setAttribute('y',to.y+dy);
}

function conditionCopy(d,step){
 const p=d.head.find(c=>c.selected);
 if(step===0)return {
  purpose:`确定预测位置 p6，并看清当前条件允许它读取的 ${d.tokens.length} 个片段。`,
  handoff:`p6 的 Q=${fmt(d.Q,2)} 将与各位置 K 比较；x6=${fmt(d.x6,2)} 留在残差主路。`
 };
 if(step===1)return {
  purpose:`用 Q 与 ${d.rows.length} 个 K 算分，做数值稳定的 softmax，再把各位置的 w·V 首尾相加。`,
  handoff:`o=${fmt(d.attention,3)}；下一步把它与 x6 相加，得到 x1=${fmt(d.x1,3)}。`
 };
 if(step===2)return {
  purpose:`残差保留 x6，加上 o 得 x1；RMSNorm 用同一个 r 缩放两个维度，得到交给 FFN 的 h。`,
  handoff:`r=${num(d.rms,3)}，h=${fmt(d.h,3)}；下一步三条 SwiGLU 通道都读取这一个 h。`
 };
 if(step===3)return {
  purpose:`SwiGLU 只在 p6 内工作：a=SiLU(g)，z=a·u，三条通道的 z 分别乘自己的 Wd 行。`,
  handoff:`ΔM=${fmt(d.dM,3)}；y=x1+ΔM=${fmt(d.y,3)}，交给输出头。`
 };
 return {
  purpose:`输出头用同一个 y 与四个候选方向做点积，softmax 后按最高概率选一个。`,
  handoff:`当前条件选出“${d.selected}”，p=${pct(p.probability)}；切换条件会从第 1 步重置整条链。`
 };
}

function mount(root){
 const stage=root.querySelector('[data-oc-stage]');
 const math=root.querySelector('[data-oc-math]');
 const stepButtons=[...root.querySelectorAll('[data-oc-step]')];
 const conditionButtons=[...root.querySelectorAll('[data-oc-mode]')];
 const prev=root.querySelector('[data-oc-prev]');
 const next=root.querySelector('[data-oc-next]');
 const toggle=root.querySelector('[data-oc-toggle]');
 const reset=root.querySelector('[data-oc-reset]');
 const label=root.querySelector('[data-oc-step-label]');
 const conditionText=root.querySelector('[data-oc-condition-text]');
 const purpose=root.querySelector('[data-oc-purpose]');
 const handoff=root.querySelector('[data-oc-handoff]');
 const status=root.querySelector('[data-oc-status]');
 const pipeNodes=[...root.querySelectorAll('[data-oc-pipe-node]')];
 let mode='with-france';
 let data=M.chain(mode);
 let step=0;
 let progress=1;
 let playing=false;
 let paused=false;
 let raf=0;
 let holdTimer=0;
 let animStart=0;
 const views=new Map();

 function reduceMotion(){return window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;}

 function updateShell(){
  stepButtons.forEach((b,i)=>{
   const active=i===step;
   b.classList.toggle('is-current',active);
   if(active)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');
  });
  conditionButtons.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.ocMode===mode)));
  conditionText.textContent=data.condition;
  label.textContent=`步骤 ${step+1} / ${STEPS.length} · ${STEPS[step].label}`;
  prev.disabled=step===0;
  next.disabled=step===STEPS.length-1;
  toggle.textContent=playing?'暂停':'播放';
  toggle.setAttribute('aria-pressed',String(playing));
  toggle.title=step===STEPS.length-1&&progress>=1?'从头播放':'播放整条链';
  const copy=conditionCopy(data,step);
  purpose.textContent=copy.purpose;
  handoff.textContent=copy.handoff;
  status.textContent=`步骤 ${step+1} / ${STEPS.length}：${STEPS[step].label}。${playing?'正在播放。':progress>=1?'已停在当前步骤。':'已暂停或等待继续。'}`;
  root.dataset.step=String(step);
  root.dataset.ocState=playing?'playing':paused?'paused':'idle';
  root.dataset.ocProgress=progress.toFixed(3);
  updatePipeline();
 }

 function updatePipeline(){
  const top=data.head.find(c=>c.selected);
  const values=[
   {main:`${data.tokens.length} 个片段`,sub:data.positions.join(' · ')},
   {main:fmt(data.attention,3),sub:'o'},
   {main:fmt(data.x1,3),sub:`h ${fmt(data.h,3)}`},
   {main:fmt(data.y,3),sub:`ΔM ${fmt(data.dM,3)}`},
   {main:data.selected,sub:`p ${pct(top.probability)}`}
  ];
  pipeNodes.forEach((node,i)=>{
   const done=i<=step;
   node.classList.toggle('is-done',i<step);
   node.classList.toggle('is-active',i===step);
   node.classList.toggle('is-pending',i>step);
   if(i===step)node.setAttribute('aria-current','step');else node.removeAttribute('aria-current');
   node.querySelector('[data-oc-pipe-main]').textContent=done?values[i].main:'—';
   node.querySelector('[data-oc-pipe-sub]').textContent=done?values[i].sub:'待计算';
  });
 }

 function updateProgress(p){
  progress=clamp(p);
  root.dataset.ocProgress=progress.toFixed(3);
  root.style.setProperty('--oc-p',progress.toFixed(4));
  const view=views.get(step);
  if(view)applyProgress(view,step,progress);
 }

 function buildContext(d){
  const chips=d.rows.map((r,i)=>`<div class="oc-token-card oc-lane-${Number(r.id.slice(1))} ${r.id==='p6'?'is-target':''} ${r.id==='p1'?'is-clue':''}" data-oc-chip="${i}">
    <b>${esc(r.id)}</b><span>${esc(r.token)}</span><small>${esc(r.role)}</small>
   </div>`).join('');
  return `<div class="oc-view" data-oc-view="context">
   <div class="oc-view-head"><div><span class="oc-eyebrow">第 1 步 · 输入与目标</span><h3>先确定“谁在问”，再看它允许读什么</h3></div><span class="oc-badge">预测位置 p6</span></div>
   <div class="oc-sequence" aria-label="当前条件的教学片段">${chips}</div>
   <div class="oc-context-grid">
    <div class="oc-context-note"><p><strong>目标是在当前上下文末尾预测下一个 token。</strong> 这一层只在 p6“？”这个位置产生下一 token 的候选分数。前面的法国、首都、哪里等片段，只有被 Attention 读取并汇入 p6 后，才会影响结果。</p>
    <p>p6 的层输入是 <b>x6=${fmt(d.x6,3)}</b>；教学 Q 为 <b>${fmt(d.Q,2)}</b>。真实模型的 Q/K/V 来自输入投影，本切片直接给定，并省略 p6 输入投影与 Q/K/V 之前的 Norm。</p></div>
    <div class="oc-flow-note"><span>下一步</span><b>p6 的 Q</b><i aria-hidden="true">→</i><b>${d.rows.length} 个可见 K</b><small>先打分，再汇合 V</small></div>
   </div>
   <p class="oc-callout">屏蔽「法国」时，p1 的 K/V 不再出现在下一步的打分表里。p6 仍是同一个预测位置，因此可以直接横向比较其余位置的分数与权重。</p>
  </div>`;
 }

 function buildAttention(d){
  const rows=d.rows.map((r,i)=>{
   const id=Number(r.id.slice(1));
   return `<div class="oc-attn-row oc-lane-${id}" data-oc-row="${i}">
    <span class="oc-pos"><b>${esc(r.id)}</b><span>${esc(r.token)}</span></span>
    <span class="oc-score">${num(r.rawScore,3)}</span>
    <span class="oc-weight-cell"><i class="oc-weight-track"><span data-oc-weight-fill></span></i><em>${pct(r.weight)}</em></span>
    <span class="oc-contrib">${fmt(r.contribution,3)}</span>
   </div>`;
  }).join('');
  const arrows=d.rows.map((r,i)=>`<g class="oc-arrow oc-lane-${Number(r.id.slice(1))}" data-oc-attn-arrow="${i}" opacity="0">
    <line class="oc-arrow-line"></line><polygon class="oc-arrow-head"></polygon><text class="oc-arrow-label">${esc(r.id)}</text>
   </g>`).join('');
  return `<div class="oc-view" data-oc-view="attention">
   <div class="oc-view-head"><div><span class="oc-eyebrow">第 2 步 · 跨位置汇合</span><h3>Q 与 K 打分，softmax 成权重，再把 w·V 加起来</h3></div><span class="oc-badge">Q=${fmt(d.Q,2)}</span></div>
   <p class="oc-view-note">分数的最大值是 ${num(d.attentionMaxScore,3)}。先从每个分数减去它，再取指数并求和，得到分母 ${num(d.denominator,3)}。这一步保持权重不变，同时避免指数过大。</p>
   <div class="oc-split">
    <div class="oc-attn-list" aria-label="各位置的分数、权重与加权 V">
     <div class="oc-attn-head"><span>位置 / 片段</span><span>s</span><span>w</span><span>w·V</span></div>
     ${rows}
    </div>
    <div class="oc-vector-figure">
     <p class="oc-figure-title">把 w·V 按位置首尾相加；粗箭头就是 o</p>
     <svg class="oc-vector-svg" viewBox="0 0 380 230" role="img" aria-label="加权 V 按位置首尾相接形成 Attention 汇合向量">
      <circle class="oc-origin" data-oc-attn-origin r="3"></circle>
      <g class="oc-arrow is-total" data-oc-attn-total opacity="0"><line class="oc-arrow-line"></line><polygon class="oc-arrow-head"></polygon><text class="oc-arrow-label">o</text></g>
      ${arrows}
     </svg>
     <p class="oc-legend"><span><i class="oc-swatch"></i>每段 = 一个位置的 w·V</span><span><i class="oc-swatch is-total"></i>粗箭头 = 汇合 o</span></p>
    </div>
   </div>
   <p class="oc-result">Attention 汇合 <strong>o=${fmt(d.attention,3)}</strong>；它仍是一个二维向量，不是文字答案。</p>
   <p class="oc-view-note">箭头长度按同一比例绘制；播放中的长度只表示汇合进度，终态读数由模型给出。</p>
  </div>`;
 }

 function buildResidual(d){
  return `<div class="oc-view" data-oc-view="residual">
   <div class="oc-view-head"><div><span class="oc-eyebrow">第 3 步 · 接回主路并调整尺度</span><h3>先加回末位向量，再按当前均方根缩放</h3></div><span class="oc-badge">残差 → RMSNorm</span></div>
   <div class="oc-split">
    <div class="oc-equation-panel">
     <div class="oc-equation">
      <div class="oc-eq-card is-left"><small>x6 · 层输入</small><b>${fmt(d.x6,3)}</b></div><span class="oc-op" aria-hidden="true">+</span>
      <div class="oc-eq-card is-attn"><small>o · Attention</small><b>${fmt(d.attention,3)}</b></div><span class="oc-op" aria-hidden="true">=</span>
      <div class="oc-eq-card is-result"><small>x1 · 第一次残差</small><b>${fmt(d.x1,3)}</b></div>
     </div>
     <p class="oc-callout">残差保留的是 p6 原来的 x6，不是把 o 替换进去。下一步 FFN 的旁路又会保留 x1；两个旁路的对象不同。</p>
     <p class="oc-result">第一次残差结果的均方根是 <strong>${num(d.rms,3)}</strong>，两维都除以这个数得到 <strong>${fmt(d.h,3)}</strong>。本算例让 γ 的两个分量都取 1，并省略 ε。</p>
    </div>
    <div class="oc-vector-figure">
     <p class="oc-figure-title">左：向量相加；右：同一方向除以 r</p>
     <svg class="oc-vector-svg" viewBox="0 0 380 220" role="img" aria-label="残差把 x6 与 o 相加得到 x1，RMSNorm 再把 x1 缩放到 h">
      <circle class="oc-origin" cx="34" cy="186" r="3"></circle>
      <circle class="oc-origin" cx="230" cy="186" r="3"></circle>
      <g class="oc-arrow oc-lane-1" data-oc-res-x6 opacity="0"><line class="oc-arrow-line"></line><polygon class="oc-arrow-head"></polygon><text class="oc-arrow-label">x6</text></g>
      <g class="oc-arrow oc-lane-2" data-oc-res-o opacity="0"><line class="oc-arrow-line"></line><polygon class="oc-arrow-head"></polygon><text class="oc-arrow-label">o</text></g>
      <g class="oc-arrow is-total" data-oc-res-x1 opacity="0"><line class="oc-arrow-line"></line><polygon class="oc-arrow-head"></polygon><text class="oc-arrow-label">x1</text></g>
      <g class="oc-arrow is-ghost" data-oc-norm-x1 opacity="0"><line class="oc-arrow-line"></line><polygon class="oc-arrow-head"></polygon><text class="oc-arrow-label">x1</text></g>
      <g class="oc-arrow oc-lane-3" data-oc-norm-h opacity="0"><line class="oc-arrow-line"></line><polygon class="oc-arrow-head"></polygon><text class="oc-arrow-label">h</text></g>
      <text class="oc-panel-label" x="24" y="20">残差相加</text><text class="oc-panel-label" x="218" y="20">RMSNorm</text>
     </svg>
     <p class="oc-legend"><span><i class="oc-swatch"></i>箭头按模型值缩放</span><span><i class="oc-swatch is-total"></i>粗箭头 = 当前结果</span></p>
    </div>
   </div>
  </div>`;
 }

 function buildFfn(d){
  const maxA=Math.max(...d.a.map(v=>Math.abs(v)),1e-9);
  const lanes=d.g.map((_,j)=>{
   const neg=d.a[j]<0;
   const width=Math.abs(d.a[j])/maxA*100;
   return `<article class="oc-lane-card oc-lane-${j+1}" data-oc-lane="${j}">
    <header><b>h${j+1}</b><span>隐藏通道 ${j+1}</span><small>W<sub>d,${j+1}</sub>=${fmt(d.WD[j],2)}</small></header>
    <dl class="oc-lane-values">
     <div><dt>g</dt><dd>${num(d.g[j],3)}</dd></div>
     <div><dt>a=SiLU(g)</dt><dd>${num(d.a[j],3)}</dd></div>
     <div><dt>u</dt><dd>${num(d.u[j],3)}</dd></div>
     <div><dt>z=a·u</dt><dd>${num(d.z[j],3)}</dd></div>
    </dl>
    <div class="oc-gate-meter ${neg?'is-negative':''}"><span>a 的相对大小（条形只表示 |a|，颜色示正负）</span><i><b data-oc-gate-fill style="width:0%"></b></i></div>
    <p class="oc-lane-contrib">z·W<sub>d,${j+1}</sub> = <strong>${fmt(d.ffnContributions[j],3)}</strong></p>
   </article>`;
  }).join('');
  const arrows=d.z.map((_,j)=>`<g class="oc-arrow oc-lane-${j+1}" data-oc-ffn-arrow="${j}" opacity="0">
    <line class="oc-arrow-line"></line><polygon class="oc-arrow-head"></polygon><text class="oc-arrow-label">h${j+1}</text>
   </g>`).join('');
  return `<div class="oc-view" data-oc-view="ffn">
   <div class="oc-view-head"><div><span class="oc-eyebrow">第 4 步 · 逐位置加工</span><h3>同一个 h 进入三条 SwiGLU 通道，三个二维贡献再相加</h3></div><span class="oc-badge">只处理 p6</span></div>
   <p class="oc-view-note">将 g 经过 SiLU 得到连续系数 a，再与 u 逐元素相乘得到 z。用每个 z 分量乘以下投影对应的一行权重，得到一份二维贡献。三份贡献逐维相加得到 ΔM。</p>
   <div class="oc-lane-grid">${lanes}</div>
   <div class="oc-split">
    <div class="oc-vector-figure">
     <p class="oc-figure-title">三条带方向与长度的贡献首尾相接，终点即 ΔM</p>
     <svg class="oc-vector-svg" viewBox="0 0 380 230" role="img" aria-label="三条 SwiGLU 通道的二维贡献首尾相加，得到 ΔM">
      <circle class="oc-origin" data-oc-ffn-origin r="3"></circle>
      <g class="oc-arrow is-total" data-oc-ffn-total opacity="0"><line class="oc-arrow-line"></line><polygon class="oc-arrow-head"></polygon><text class="oc-arrow-label">ΔM</text></g>
      ${arrows}
     </svg>
     <p class="oc-legend"><span><i class="oc-swatch"></i>每段 = z<sub>j</sub>·W<sub>d,j</sub>（二维）</span><span><i class="oc-swatch is-total"></i>粗箭头 = ΔM</span></p>
    </div>
    <div class="oc-ffn-summary">
     <div><span>第二次残差基底</span><b>x1=${fmt(d.x1,3)}</b></div>
     <div><span>FFN 更新</span><b>ΔM=${fmt(d.dM,3)}</b></div>
     <div class="is-result"><span>层末表示</span><b>y=${fmt(d.y,3)}</b></div>
     <p>三份贡献逐维相加，界面数值保留三位小数。</p>
     <div class="oc-ffn-sum lesson-formulas"><math display="block">${d.ffnContributions.map(v=>vectorMath(v,3)).join('<mo>+</mo>')}<mo>≈</mo>${vectorMath(d.dM,3)}</math></div>
    </div>
   </div>
   <p class="oc-callout">h1–h3 只是本算例的通道编号，不绑定“国家”“首都”等语义。W<sub>g</sub>、W<sub>u</sub>、W<sub>d</sub> 在推理时固定；换上下文会改变 h、a、u、z，因而改变 ΔM，但不会临时改写这些参数。</p>
  </div>`;
 }

 function buildHead(d){
  const top=d.head.find(c=>c.selected);
  const cards=d.head.map((c,i)=>`<article class="oc-candidate oc-lane-${i+1} ${c.selected?'is-selected':''}" data-oc-candidate="${i}" role="listitem">
    <header><b>${esc(c.token)}</b><span>c=${fmt(c.vector,2)}</span></header>
    <div class="oc-candidate-score"><span>点积分数</span><strong>${num(c.logit,3)}</strong></div>
    <div class="oc-prob"><i><b data-oc-prob-fill="${i}" style="width:0%"></b></i><span>${pct(c.probability)}</span></div>
    <p>${c.selected?'最高分候选':'其他候选'}</p>
   </article>`).join('');
  return `<div class="oc-view" data-oc-view="head">
   <div class="oc-view-head"><div><span class="oc-eyebrow">第 5 步 · 映射到候选词表</span><h3>y 与四个候选方向做点积，softmax 后选最高概率</h3></div><span class="oc-badge">4 个教学候选</span></div>
   <p class="oc-view-note">和 Attention 中的稳定 softmax 一样，从各候选分数中减去最大的 ${num(d.headMaxLogit,3)}，再对指数求和。本步归一化的分母是 ${num(d.softmaxDenominator,3)}。</p>
   <div class="oc-head-grid" role="list" aria-label="四个教学候选的 logits 与概率">${cards}</div>
   <div class="oc-head-result">
    <div><span>稳定 softmax 的分母</span><b>${num(d.softmaxDenominator,3)}</b><small>概率与 logits 来自同一组 y 与候选方向</small></div>
    <div class="oc-winner" data-oc-winner><span>argmax → 下一 token</span><b>${esc(top.token)}</b><small>p=${pct(top.probability)}（教学候选表）</small></div>
   </div>
   <p class="oc-result">在「${esc(d.condition)}」这个教学条件下，最高分候选是<strong>「${esc(d.selected)}」</strong>。</p>
  </div>`;
 }

 function buildView(index,d){
  const html=index===0?buildContext(d):index===1?buildAttention(d):index===2?buildResidual(d):index===3?buildFfn(d):buildHead(d);
  const holder=document.createElement('div');
  holder.innerHTML=html;
  return holder.firstElementChild;
 }

 function ensureView(){
  if(views.has(step))return views.get(step);
  stage.querySelectorAll(':scope > noscript').forEach(n=>n.remove());
  const view=buildView(step,data);
  stage.append(view);
  views.set(step,view);
  for(const [key,node] of views)node.hidden=key!==step;
  return view;
 }

 function showOnlyView(){
  for(const [key,node] of views)node.hidden=key!==step;
  const view=ensureView();
  for(const [key,node] of views)node.hidden=key!==step;
  return view;
 }

 function applyContext(view,p){
  const chips=[...view.querySelectorAll('[data-oc-chip]')];
  chips.forEach((el,i)=>{
   const t=smooth(clamp((p-i*.035)/.38));
   el.style.opacity=(.02+.98*t).toFixed(3);
   el.style.transform=`translateY(${(1-t)*7}px)`;
  });
 }

 function applyAttention(view,p,d){
  const box={x:44,y:184,scale:112};
  const origin=originPoint(0,0,box);
  const originDot=view.querySelector('[data-oc-attn-origin]');
  if(originDot){originDot.setAttribute('cx',origin.x);originDot.setAttribute('cy',origin.y);}
  let from=[0,0];
  d.rows.forEach((r,i)=>{
   const t=smooth(clamp((p-.05-i*.045)/.45));
   const to=[from[0]+r.contribution[0]*t,from[1]+r.contribution[1]*t];
   const g=view.querySelector(`[data-oc-attn-arrow="${i}"]`);
   if(g){
    drawArrow(g,originPoint(from[0],from[1],box),originPoint(to[0],to[1],box),7);
    placeLabel(g,originPoint(to[0],to[1],box),i%2?7:-7,i%2?-5:13);
    g.setAttribute('opacity',t>.01?'1':'0');
   }
   const row=view.querySelector(`[data-oc-row="${i}"]`);
   if(row){
    const fill=row.querySelector('[data-oc-weight-fill]');
    if(fill)fill.style.width=(r.weight*100*t).toFixed(2)+'%';
   }
   from=to;
  });
  const total=view.querySelector('[data-oc-attn-total]');
  if(total){
   drawArrow(total,origin,originPoint(from[0],from[1],box),9);
   total.setAttribute('opacity',p>.02?'1':'0');
   placeLabel(total,originPoint(from[0],from[1],box),12,-8);
  }
 }

 function applyResidual(view,p,d){
  const left={x:34,y:186,scale:68};
  const right={x:230,y:186,scale:68};
  const o=originPoint(0,0,left), r0=originPoint(0,0,right);
  const tX6=smooth(clamp(p/.3));
  const tO=smooth(clamp((p-.12)/.5));
  const x6=[d.x6[0]*tX6,d.x6[1]*tX6];
  const x6tip=originPoint(x6[0],x6[1],left);
  const x6Arrow=view.querySelector('[data-oc-res-x6]');
  if(x6Arrow){drawArrow(x6Arrow,o,x6tip,7);placeLabel(x6Arrow,x6tip,-8,-7);x6Arrow.setAttribute('opacity',tX6>.01?'1':'0');}
  const partial=[x6[0]+d.attention[0]*tO,x6[1]+d.attention[1]*tO];
  const partialPt=originPoint(partial[0],partial[1],left);
  const oArrow=view.querySelector('[data-oc-res-o]');
  if(oArrow){drawArrow(oArrow,x6tip,partialPt,7);placeLabel(oArrow,partialPt,7,12);oArrow.setAttribute('opacity',tO>.01?'1':'0');}
  const tX1=smooth(clamp((p-.48)/.25));
  const x1Arrow=view.querySelector('[data-oc-res-x1]');
  if(x1Arrow){
   const to=originPoint(partial[0],partial[1],left);
   drawArrow(x1Arrow,o,to,9);placeLabel(x1Arrow,to,11,-8);
   x1Arrow.setAttribute('opacity',tX1>.01?'1':'0');
  }
  const tH=smooth(clamp((p-.68)/.32));
  const hPoint=originPoint(d.h[0]*tH,d.h[1]*tH,right);
  const x1Full=originPoint(d.x1[0],d.x1[1],right);
  const ghost=view.querySelector('[data-oc-norm-x1]');
  if(ghost){drawArrow(ghost,r0,x1Full,7);placeLabel(ghost,x1Full,-8,-8);ghost.setAttribute('opacity',tH>.01?'.34':'0');}
  const hArrow=view.querySelector('[data-oc-norm-h]');
  if(hArrow){drawArrow(hArrow,r0,hPoint,9);placeLabel(hArrow,hPoint,12,8);hArrow.setAttribute('opacity',tH>.01?'1':'0');}
 }

 function applyFfn(view,p,d){
  const box={x:72,y:188,scale:108};
  const origin=originPoint(0,0,box);
  const dot=view.querySelector('[data-oc-ffn-origin]');
  if(dot){dot.setAttribute('cx',origin.x);dot.setAttribute('cy',origin.y);}
  let from=[0,0];
  d.ffnContributions.forEach((c,j)=>{
   const t=smooth(clamp((p-.05-j*.12)/.48));
   const to=[from[0]+c[0]*t,from[1]+c[1]*t];
   const g=view.querySelector(`[data-oc-ffn-arrow="${j}"]`);
   if(g){
    drawArrow(g,originPoint(from[0],from[1],box),originPoint(to[0],to[1],box),7);
    placeLabel(g,originPoint(to[0],to[1],box),j===1?-10:8,j===2?13:-7);
    g.setAttribute('opacity',t>.01?'1':'0');
   }
   const card=view.querySelector(`[data-oc-lane="${j}"]`);
   if(card){
    const fill=card.querySelector('[data-oc-gate-fill]');
    if(fill){const maxA=Math.max(...d.a.map(v=>Math.abs(v)),1e-9);fill.style.width=(Math.abs(d.a[j])/maxA*100*t).toFixed(1)+'%';}
    card.style.opacity=(.45+.55*smooth(clamp((p-j*.05)/.45))).toFixed(3);
   }
   from=to;
  });
  const total=view.querySelector('[data-oc-ffn-total]');
  if(total){
   drawArrow(total,origin,originPoint(from[0],from[1],box),9);
   total.setAttribute('opacity',p>.02?'1':'0');
   placeLabel(total,originPoint(from[0],from[1],box),12,-8);
  }
 }

 function applyHead(view,p,d){
  const cards=[...view.querySelectorAll('[data-oc-candidate]')];
  cards.forEach((el,i)=>{
   const c=d.head[i];
   const t=smooth(clamp((p-i*.04)/.48));
   el.style.opacity=(.3+.7*t).toFixed(3);
   const fill=el.querySelector('[data-oc-prob-fill]');
   if(fill)fill.style.width=(c.probability*100*t).toFixed(2)+'%';
   el.classList.toggle('is-hot',c.selected&&p>.82);
  });
  const winner=view.querySelector('[data-oc-winner]');
  if(winner)winner.classList.toggle('is-hot',p>.82);
 }

 function applyProgress(view,index,p){
  if(index===0)applyContext(view,p);
  else if(index===1)applyAttention(view,p,data);
  else if(index===2)applyResidual(view,p,data);
  else if(index===3)applyFfn(view,p,data);
  else applyHead(view,p,data);
 }

 function cancelAnimation(){
  if(raf)cancelAnimationFrame(raf);
  raf=0;
  if(holdTimer)clearTimeout(holdTimer);
  holdTimer=0;
 }

 function completeStep(){
  updateProgress(1);
  if(!playing){updateShell();return;}
  if(step===STEPS.length-1){finishPlayback();return;}
  holdTimer=setTimeout(()=>{
   holdTimer=0;
   if(!playing)return;
   goTo(step+1,{animate:true,keepPlaying:true});
  },HOLD);
 }

 function tick(ts){
  if(!animStart)animStart=ts;
  const p=clamp((ts-animStart)/DURATION);
  updateProgress(p);
  if(p<1){raf=requestAnimationFrame(tick);return;}
  raf=0;
  completeStep();
 }

 function startTick(){
  cancelAnimation();
  if(reduceMotion()){completeStep();return;}
  animStart=performance.now()-progress*DURATION;
  raf=requestAnimationFrame(tick);
 }

 function finishPlayback(){
  playing=false;
  paused=false;
  cancelAnimation();
  updateProgress(1);
  updateShell();
  status.textContent=`播放结束：${STEPS[STEPS.length-1].label} 已显示最终结果。`;
 }

 function goTo(n,opts={}){
  const animate=opts.animate!==false;
  if(!opts.keepPlaying){playing=false;paused=false;}
  cancelAnimation();
  step=clamp(n,0,STEPS.length-1);
  progress=animate?0:1;
  stage.querySelectorAll(':scope > noscript').forEach(node=>node.remove());
  showOnlyView();
  updateProgress(progress);
  updateShell();
  if(animate)startTick();
 }

 function startPlayback(){
  if(playing)return;
  playing=true;
  paused=false;
  updateShell();
  if(step===STEPS.length-1&&progress>=1){
   goTo(0,{animate:true,keepPlaying:true});
  }else if(progress<1){
   startTick();
  }else{
   goTo(step,{animate:true,keepPlaying:true});
  }
 }

 function pausePlayback(){
  if(!playing)return;
  playing=false;
  paused=true;
  cancelAnimation();
  updateShell();
 }

 function stopPlayback(){
  playing=false;
  paused=false;
  cancelAnimation();
 }

 function changeMode(nextMode){
  if(mode===nextMode)return;
  stopPlayback();
  mode=nextMode;
  data=M.chain(mode);
  views.clear();
  stage.replaceChildren();
  math.innerHTML=renderMath(data);
  goTo(0,{animate:false});
 }

 function renderMath(d){
  const attentionRows=d.rows.map((r,i)=>`<tr><th scope="row">${esc(r.id)} · ${esc(r.token)}</th><td>${fmt(r.key,1)}</td><td>${num(r.rawScore,3)}</td><td>${num(d.exps[i],4)}</td><td>${num(r.weight,4)}</td><td>${fmt(r.value,2)}</td><td>${fmt(r.contribution,4)}</td></tr>`).join('');
  const ffnRows=d.g.map((_,j)=>`<tr><th scope="row">h${j+1}</th><td>${num(d.g[j],4)}</td><td>${num(d.a[j],4)}</td><td>${num(d.u[j],4)}</td><td>${num(d.z[j],4)}</td><td>${fmt(d.WD[j],3)}</td><td>${fmt(d.ffnContributions[j],4)}</td></tr>`).join('');
  const headRows=d.head.map((c,i)=>`<tr class="${c.selected?'is-selected':''}"><th scope="row">${esc(c.token)}</th><td>${fmt(c.vector,2)}</td><td>${num(c.logit,4)}</td><td>${num(d.headExps[i],4)}</td><td>${num(c.probability,4)}</td></tr>`).join('');
  return `<div class="oc-math-grid">
   <section class="oc-math-block"><h3>同一组教学参数</h3>
    <div class="lesson-formulas"><math display="block"><mi>Q</mi><mo>≈</mo>${vectorMath(d.Q,3)}</math><math display="block"><msub><mi>x</mi><mn>6</mn></msub><mo>=</mo>${vectorMath(d.x6,3)}</math></div>
    <div class="lesson-formulas">${weightMatrix('g',d.WG)}${weightMatrix('u',d.WU)}${weightMatrix('d',d.WD)}</div>
    <p><b>候选方向</b>：${d.head.map(c=>`${esc(c.token)} ${fmt(c.vector,2)}`).join('；')}</p>
    <p class="oc-math-note">这组数直接给出 Q、K、V，并将层末向量直接送进输出头。本对照只计算一次子层更新链，便于固定投影值并追踪局部变化。</p>
   </section>
   <section class="oc-math-block"><h3>Attention：稳定 softmax 与加权和</h3>
    <div class="oc-formula lesson-formulas">${equation('score_lab')}${equation('softmax_lab_attention')}${equation('attention_mix_lab')}</div>
    <div class="oc-table-wrap"><table class="oc-table"><caption>最高位置分数为 ${num(d.attentionMaxScore,3)}，稳定 softmax 的分母为 ${num(d.denominator,4)}。</caption><thead><tr><th>位置</th><th>K</th><th>分数</th><th>减最大值后取指数</th><th>权重</th><th>V</th><th>加权 V</th></tr></thead><tbody>${attentionRows}</tbody></table></div>
   </section>
   <section class="oc-math-block"><h3>第一次残差与 RMSNorm</h3>
    <div class="oc-formula lesson-formulas">${equation('lab_residual_a')}${equation('rms_lab')}</div>
    <p>这里各维增益都取 1，并省略 ε。两维共用的均方根约为 ${num(d.rms,4)}，缩放后得到给 FFN 用的 h。</p>
   </section>
   <section class="oc-math-block"><h3>SwiGLU：三条通道各给二维贡献</h3>
    <div class="oc-formula lesson-formulas">${equation('ffn_lab')}${equation('residual_f')}</div>
    <div class="oc-table-wrap"><table class="oc-table"><caption>最后一列给出每个通道的二维贡献，逐维相加得到 ΔM。</caption><thead><tr><th>通道</th><th>g</th><th>a</th><th>u</th><th>z</th><th>下投影的对应行</th><th>二维贡献</th></tr></thead><tbody>${ffnRows}</tbody></table></div>
    <div class="lesson-formulas"><math display="block"><mi>y</mi><mo>≈</mo>${vectorMath(d.y,4)}</math></div>
   </section>
   <section class="oc-math-block"><h3>输出头：同一组 y 与四个候选</h3>
    <div class="oc-formula lesson-formulas">${equation('lab_logit')}${equation('softmax_vocab')}${equation('greedy')}</div>
    <div class="oc-table-wrap"><table class="oc-table"><caption>最高候选分数为 ${num(d.headMaxLogit,3)}，稳定 softmax 的分母为 ${num(d.softmaxDenominator,4)}。贪心选中「${esc(d.selected)}」。</caption><thead><tr><th>候选</th><th>候选权重向量</th><th>logit</th><th>减最大值后取指数</th><th>概率</th></tr></thead><tbody>${headRows}</tbody></table></div>
   </section>
  </div>`;
 }

 stepButtons.forEach((b,i)=>b.addEventListener('click',()=>{stopPlayback();goTo(i,{animate:true});}));
 conditionButtons.forEach(b=>b.addEventListener('click',()=>changeMode(b.dataset.ocMode)));
 prev.addEventListener('click',()=>{stopPlayback();goTo(step-1,{animate:true});});
 next.addEventListener('click',()=>{stopPlayback();goTo(step+1,{animate:true});});
 toggle.addEventListener('click',()=>{playing?pausePlayback():startPlayback();});
 reset.addEventListener('click',()=>{stopPlayback();goTo(0,{animate:false});status.textContent='已回到第 1 步；播放按钮可从开始处重新播放。';});
 root.addEventListener('keydown',e=>{
  if(e.target!==root)return;
  if(e.key==='ArrowRight'){e.preventDefault();stopPlayback();goTo(step+1,{animate:true});}
  if(e.key==='ArrowLeft'){e.preventDefault();stopPlayback();goTo(step-1,{animate:true});}
 });
 root.setAttribute('tabindex','0');
 math.innerHTML=renderMath(data);
 goTo(0,{animate:false});
 root.dataset.ocReady='true';
}

roots.forEach(mount);
})();
