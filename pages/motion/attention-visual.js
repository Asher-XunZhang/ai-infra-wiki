/* 03 Attention: one query, three causal lenses.
 *
 * The board is a mechanism animation, not a finished table with a cursor on
 * top. At every timeline position the visible numbers are the ones computed so
 * far: inputs form Q/K/V roles, Q is compared with readable K rows, e^s is
 * accumulated into the denominator, bars grow into weights, and only then are
 * w x V contributions added to the running output. Selecting a concept first
 * shows that lens in its complete, checkable state; play or seek then drops the
 * snapshot while keeping the lens, so the same concept animates its mechanism.
 *
 * Full four-row numbers remain in the player's "深入核对" disclosure.
 */
(function(){
'use strict';
const V=window.MechanismVisuals,D=window.TeachingDrawing;
if(!V||!D)return;
const {board}=D;
const NS='http://www.w3.org/2000/svg';
const fmt=v=>'['+v.map(x=>Number(x.toFixed(2))).join(', ')+']';
const tight=v=>'['+v.map(x=>Number(x.toFixed(2))).join(',')+']';
const pct=w=>(w*100).toFixed(1)+'%';
const clamp=n=>Math.max(0,Math.min(1,n));
const lerp=(a,b,t)=>a+(b-a)*t;
const ease=t=>t*t*(3-2*t);
const phase=(p,a,b)=>ease(clamp((p-a)/(b-a)));
const qVariant={2:0,4:1,3:2};
const MODE_META={
 qkv:{num:'①',title:'输入怎样分成 Q、K、V？',sub:'Q 去提问，K 被比较，V 带内容。'},
 weights:{num:'②',title:'哪些位置该多分一点？',sub:'s → e^s → w，再按 w 混合 V。'},
 mask:{num:'③',title:'为什么未来位置读不到？',sub:'第 i 行只看 p1…pi，未来不进分母。'}
};
const STAGE_LABELS=['① 输入 → Q/K/V','② Q 比 K → 分数 s','③ e^s → 分母','④ 归一化 → 权重 w','⑤ w×V → 汇合 m','⑥ 遮罩回看'];
const STAGE_NOTES=[
 '输入逐个形成 Q、K、V；Q 只来自当前 Query。',
 'Q 与每个可读 K 比较，得到分数 s。',
 'e^s 只对可读项累加，得到分母 Σ。',
 '分母归一化，每个位置得到权重 w。',
 'w 乘以各自的 V，再逐维相加得到 m。',
 '遮罩限定可读范围；未来项不进入分母。'
];

V['f-attention']=(svg,viewport,s)=>{
 return board(svg,viewport,'attention',m=>m?560:480,H=>{
  const d=s.data,query=d.queryPosition,qIndex=d.query;
  const allowed=d.allowed;
  const mk=(tag,attrs={},parent=svg)=>{const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(parent)parent.append(e);return e;};
  const attrs=(e,o)=>{for(const[k,v]of Object.entries(o))e.setAttribute(k,v);};
  const tx=(parent,value,cls,x,y,extra={})=>{const e=mk('text',Object.assign({class:cls,x,y},extra),parent);e.textContent=value;return e;};
  const setText=(e,v)=>{if(e.textContent!==v)e.textContent=v;};
  const rect=(parent,cls,x,y,w,h,rx=7,extra={})=>mk('rect',Object.assign({class:cls,x,y,width:w,height:h,rx},extra),parent);
  const setBox=(e,x,y,w,h)=>attrs(e,{x,y,width:w,height:h});
  const vis=(e,on)=>{e.style.display=on?'':'none';e.setAttribute('aria-hidden',on?'false':'true');};
  const opacity=(e,v)=>{e.style.opacity=clamp(v).toFixed(3);};
  const textAt=(e,x,y)=>attrs(e,{x,y});
  const on=(e,c,v)=>e.classList.toggle(c,!!v);
  const line=(parent,cls,x1,y1,x2,y2)=>mk('line',Object.assign({class:cls,x1,y1,x2,y2},{}),parent);

  const bg=rect(svg,'att-board',0,0,720,480,16);
  const topline=rect(svg,'att-topline',0,0,720,3,1);
  const gHead=mk('g',{class:'att-head'},svg);
  const gQkv=mk('g',{class:'att-mode','data-att-mode':'qkv'},svg);
  const gWeights=mk('g',{class:'att-mode','data-att-mode':'weights'},svg);
  const gMask=mk('g',{class:'att-mode','data-att-mode':'mask'},svg);
  const modeGroups={qkv:gQkv,weights:gWeights,mask:gMask};

  /* ---------------------------------------------------------------- header */
  const kicker=tx(gHead,'ATTENTION · 单头 · 2 维教学例','att-kicker',16,18);
  const modeNum=tx(gHead,'','att-mode-num',16,42);
  const modeTitle=tx(gHead,'','att-mode-title',44,42);
  const modeSub=tx(gHead,'','att-mode-sub',16,62);
  const queryLabel=tx(gHead,'Query','att-kicker',526,18);
  const rangeNote=tx(gHead,'','att-dim',526,70);
  const tabs=[2,3,4].map(position=>{
   const variant=qVariant[position],g=mk('g',{class:'att-tab',role:'button',tabindex:'0','aria-label':`把 p${position} 设为 Query`},gHead);
   const r=mk('rect',{rx:6,width:42,height:24},g),t=tx(g,`p${position}`,'att-tab-text',0,0,{'text-anchor':'middle'});
   const go=()=>svg.dispatchEvent(new CustomEvent('attention-query',{bubbles:true,detail:{variant}}));
   g.addEventListener('click',go);
   g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.stopPropagation();go();}});
   return {position,variant,g,r,t};
  });

  /* --------------------------------------------------------------- Q/K/V view */
  const qkvFormula=tx(gQkv,'教学投影：Q = x_q WQ，K_i = x_i WK，V_i = x_i WV；Q 只取当前行。','att-note',14,92);
  const qkvPanelL=rect(gQkv,'att-panel',14,102,254,302);
  const qkvLHead=tx(gQkv,'① 每个位置：输入投影成三种角色','att-stage-title',28,124);
  const qkvNote1=tx(gQkv,'Q 只从当前 Query 行取；','att-dim',28,336);
  const qkvNote2=tx(gQkv,'K、V 每个位置各一份。','att-dim',28,354);
  const qkvNote3=tx(gQkv,'教学例 WQ=WK=I、WV=2I。','att-dim',28,372);
  const qkvNote6=tx(gQkv,'真实模型用三组可学习投影。','att-dim',28,390);
  const qkvCols=['位置','输入 x','Q','K','V'].map((label,i)=>tx(gQkv,label,'att-colhead',[28,64,126,178,230][i],146));
  const qkvRows=d.inputs.map((input,i)=>{
   const rowbg=rect(gQkv,'att-rowbg',0,0,0,0,7);
   const pos=tx(gQkv,'','att-num',0,0);
   const x=tx(gQkv,'','att-num',0,0);
   const q=tx(gQkv,'','att-q att-num',0,0);
   const k=tx(gQkv,'','att-k att-num',0,0);
   const v=tx(gQkv,'','att-v att-num',0,0);
   const score=tx(gQkv,'','att-score',0,0);
   const vLine=tx(gQkv,'','att-note',0,0);
   return {i,input,bg:rowbg,pos,x,q,k,v,score,vLine};
  });
  const qkvMCard=rect(gQkv,'att-qcard',12,104,336,98);
  const qkvMLine1=tx(gQkv,'','att-strong',24,130);
  const qkvMLine2=tx(gQkv,'','att-q',24,156);
  const qkvMSection=tx(gQkv,'K 被比较 · V 等权重确定后混合','att-note',24,180);

  const qkvPanelR=rect(gQkv,'att-panel',282,102,424,302);
  const qkvRHead=tx(gQkv,'② Q 与每个可读 K 比较','att-stage-title',296,124);
  const qkvQLine=tx(gQkv,'','att-strong',296,148);
  const qkvRCols=['位置','K','s = Q·K / √2','本次读取'].map((label,i)=>tx(gQkv,label,'att-colhead',[296,332,410,545][i],170));
  const qkvRRows=d.inputs.map((_,i)=>{
   const rowbg=rect(gQkv,'att-rowbg',0,0,0,0,7);
   const pos=tx(gQkv,'','att-num',0,0);
   const k=tx(gQkv,'','att-k att-num',0,0);
   const score=tx(gQkv,'','att-score',0,0);
   const status=tx(gQkv,'','att-note',0,0);
   return {i,bg:rowbg,pos,k,score,status};
  });
  const qkvNote4=tx(gQkv,'s 只是“像不像”的分数，还不是权重；','att-dim',296,356);
  const qkvNote5=tx(gQkv,'下一步才对可读项做 e^s / Σ。','att-dim',296,378);

  const qkvChainLine=line(gQkv,'att-chain',90,438,630,438);
  const qkvChainDots=[0,1,2].map(()=>mk('circle',{class:'att-chain-dot',r:4},gQkv));
  const qkvChainLabels=['输入 → Q/K/V','Q 比 K','V 等权重'].map(label=>tx(gQkv,label,'att-dim',0,0,{'text-anchor':'middle'}));
  const qkvPulse=mk('circle',{class:'att-pulse',r:3.2},gQkv);
  const qkvStageNote=tx(gQkv,'','att-dim',24,466);
  const qkvFooter=tx(gQkv,'','att-foot',14,544);

  /* -------------------------------------------------------------- weight view */
  const wFormula=tx(gWeights,'s = Q·K / √2   →   e^s / Σ   →   w × V   →   Σ w·V','att-note',14,92);
  const wStage=tx(gWeights,'','att-stage-chip',540,92);
  const wDenomPanel=rect(gWeights,'att-panel',14,102,692,64);
  const wDenomTitle=tx(gWeights,'Σ e^s（只加可读项）','att-stage-title',28,124);
  const wDenomLine=tx(gWeights,'','att-w',200,124);
  wDenomLine.setAttribute('data-denominator','true');
  const wWeightLine=tx(gWeights,'','att-w',540,124);
  wWeightLine.setAttribute('data-weight-sum','true');
  const wDenomNote=tx(gWeights,'未来项 e^s = 0，不进入这一行求和。','att-dim',28,146);
  const wCols=['位置','K','s = Q·K/√2','e^s','w','w × V'].map((label,i)=>tx(gWeights,label,'att-colhead',[28,64,150,226,300,500][i],184));
  const wRows=d.inputs.map((_,i)=>{
   const rowbg=rect(gWeights,'att-rowbg',0,0,0,0,7);
   const pos=tx(gWeights,'','att-num',0,0);
   const k=tx(gWeights,'','att-k att-num',0,0);
   const score=tx(gWeights,'','att-score',0,0);
   const exp=tx(gWeights,'','att-w att-num',0,0);
   const track=rect(gWeights,'att-track',0,0,0,7,3);
   const bar=rect(gWeights,'att-wbar',0,0,0,7,3);
   const wlabel=tx(gWeights,'','att-w att-num',0,0);
   const contrib=tx(gWeights,'','att-v att-num',0,0);
   return {i,bg:rowbg,pos,k,score,exp,track,bar,wlabel,contrib};
  });
  const wOutPanel=rect(gWeights,'att-panel',14,374,692,84);
  const wOutEq=tx(gWeights,'','att-out',28,402);
  wOutEq.setAttribute('data-output','true');
  wOutEq.setAttribute('data-complete','true');
  const wOutNote=tx(gWeights,'两维分别相加，不是拼接','att-dim',450,424);
  const wMixNote=tx(gWeights,'','att-dim',450,448);
  const wTrack1=rect(gWeights,'att-track',28,418,400,9,3);
  const wTrack2=rect(gWeights,'att-track',28,434,400,9,3);
  const wOutSegs=[];
  for(let i=0;i<4;i++)for(let j=0;j<2;j++){
   const e=rect(gWeights,'att-contrib',0,0,0,9,2);
   e.style.fill=['#5fd6e8','#b9a6f5','#f2b45c','#c4a6f5'][i];
   e.setAttribute('data-contribution',`p${i+1}-d${j+1}`);
   e.setAttribute('data-row',String(i+1));
   wOutSegs.push({i,j,e});
  }
  const wFooter=tx(gWeights,'','att-foot',14,544);

  /* ---------------------------------------------------------------- mask view */
  const mPanel=rect(gMask,'att-panel',14,94,302,336);
  const mHead=tx(gMask,'因果遮罩','att-stage-title',28,116);
  const mSub=tx(gMask,'行 = Query · 列 = Key','att-note',28,134);
  const mCells=[];
  for(let r=0;r<4;r++)for(let c=0;c<4;c++){
   const g=mk('g',{},gMask);
   const cell=rect(g,'att-maskcell',0,0,0,0,5);
   const dot=mk('circle',{class:'att-openmark',r:2.2},g);
   const x1=mk('path',{class:'att-maskmark'},g),x2=mk('path',{class:'att-maskmark'},g);
   mCells.push({r,c,g,cell,dot,x1,x2});
  }
  const mCols=[0,1,2,3].map(i=>tx(gMask,`p${i+1}`,'att-note',0,0,{'text-anchor':'middle'}));
  const mRows=[0,1,2,3].map(i=>{
   const e=tx(gMask,`p${i+1}`,'att-maskrow',0,0);
   if(i>0){
    e.setAttribute('role','button');e.setAttribute('tabindex','0');e.setAttribute('aria-label',`观察 p${i+1} 作为 Query`);e.classList.add('is-clickable');
    const go=()=>svg.dispatchEvent(new CustomEvent('attention-query',{bubbles:true,detail:{variant:qVariant[i+1]}}));
    e.addEventListener('click',go);
    e.addEventListener('keydown',ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();ev.stopPropagation();go();}});
   }
   return e;
  });
  const mRowBox=rect(gMask,'att-matrix-current',0,0,0,0,5);
  const mLegend=tx(gMask,'✓ 可读 · × 未来：e^s = 0，不进分母','att-dim',28,358);

  const mDetailPanel=rect(gMask,'att-panel',330,94,376,336);
  const mDetailHead=tx(gMask,'当前 Query 的可见项','att-stage-title',344,116);
  const mDetailSub=tx(gMask,'s = Q·K/√2 · 只对 ✓ 求和','att-note',344,134);
  const mDetailCols=['位置','s','e^s','w','读取'].map((label,i)=>tx(gMask,label,'att-colhead',[344,390,452,524,620][i],158));
  const mDetailRows=d.inputs.map((_,i)=>{
   const rowbg=rect(gMask,'att-rowbg',0,0,0,0,6);
   const pos=tx(gMask,'','att-num',0,0);
   const score=tx(gMask,'','att-score',0,0);
   const exp=tx(gMask,'','att-w att-num',0,0);
   const weight=tx(gMask,'','att-w att-num',0,0);
   const status=tx(gMask,'','att-note',0,0);
   return {i,bg:rowbg,pos,score,exp,weight,status};
  });
  const mSummaryPanel=rect(gMask,'att-panel',12,366,336,118);
  const mReadNote=tx(gMask,'','att-note',24,390);
  const mDenom=tx(gMask,'','att-w',344,340);
  mDenom.setAttribute('data-denominator','true');
  const mWeightSum=tx(gMask,'','att-w',344,360);
  mWeightSum.setAttribute('data-weight-sum','true');
  const mOut=tx(gMask,'','att-out',344,388);
  mOut.setAttribute('data-output','true');
  mOut.setAttribute('data-complete','true');
  const mNote=tx(gMask,'','att-dim',344,410);
  const mFooter=tx(gMask,'','att-foot',14,540);

  /* ---------------------------------------------------------------- header */
  function paintHeader(m,mode){
   const meta=MODE_META[mode];
   attrs(bg,{width:m?360:720,height:m?560:480});
   attrs(topline,{width:m?360:720});
   textAt(kicker,m?14:16,18);
   setText(kicker,m?'ATTENTION · 2 维教学例':'ATTENTION · 单头 · 2 维教学例');
   textAt(modeNum,m?14:16,42);setText(modeNum,meta.num);
   textAt(modeTitle,m?38:44,42);setText(modeTitle,meta.title);
   textAt(modeSub,m?14:16,62);setText(modeSub,meta.sub);
   textAt(queryLabel,m?14:526,m?88:18);setText(queryLabel,'Query');
   tabs.forEach((tab,i)=>{
    const x=m?64+i*46:526+i*46,y=m?70:26;
    setBox(tab.r,x,y,42,24);textAt(tab.t,x+21,y+17);
    tab.g.classList.toggle('is-query',tab.position===query);
    tab.g.setAttribute('aria-pressed',String(tab.position===query));
   });
   textAt(rangeNote,m?212:526,m?88:70);
   setText(rangeNote,d.allowed===4?'可读 p1–p4 · 无未来':`可读 p1–p${d.allowed} · ${d.allowed} 项`);
  }

  /* ------------------------------------------------------------- Q/K/V paint */
  function paintQKV(m,p,pinned,stage){
   const roleT=pinned?1:phase(p,0,1);
   const scoreT=pinned?1:phase(p,1,2);
   const qLocal=pinned?1:phase(p,.12,.62);
   textAt(qkvFormula,14,92);
   vis(qkvFormula,!m);
   vis(qkvPanelL,!m);vis(qkvLHead,!m);vis(qkvNote1,!m);vis(qkvNote2,!m);vis(qkvNote3,!m);vis(qkvNote6,!m);
   qkvCols.forEach(e=>vis(e,!m));
   vis(qkvMCard,m);vis(qkvMLine1,m);vis(qkvMLine2,m);vis(qkvMSection,m);
   if(m){
    textAt(qkvMLine1,24,130);setText(qkvMLine1,`当前 p${query} 输入 x=${fmt(d.inputs[qIndex])}`);
    textAt(qkvMLine2,24,156);
    setText(qkvMLine2,qLocal>.4?`→ Q = ${fmt(d.Q)} · 只来自当前 Query`:'→ Q = … · 正在形成当前 Query');
   }
   qkvRows.forEach((row,i)=>{
    const readable=i<=qIndex,future=!readable;
    const local=pinned?1:phase(p,.04+i*.12,.5+i*.12);
    const shownScore=pinned?1:phase(p,1+i*.1,1.3+i*.1);
    on(row.bg,'is-future',future);on(row.bg,'is-query',i===qIndex);
    on(row.bg,'is-scan',!future&&scoreT>0&&scoreT<1&&i===Math.min(qIndex,Math.floor(scoreT*allowed)));
    if(m){
     const y=214+i*56;
     setBox(row.bg,12,y,336,52);
     textAt(row.pos,24,y+19);setText(row.pos,`p${i+1}`);
     textAt(row.x,60,y+19);setText(row.x,tight(d.inputs[i]));
     textAt(row.k,150,y+19);setText(row.k,local>.45?`→ K${tight(d.keys[i])}`:'→ K …');
     opacity(row.k,.35+.65*local);
     textAt(row.score,270,y+19);
     setText(row.score,readable?(shownScore>.2?`s=${d.scores[i].toFixed(2)}`:'s=…'):'s=−∞');
     opacity(row.score,readable?.35+.65*shownScore:1);
     row.score.classList.toggle('att-hot',readable&&scoreT>0&&scoreT<1);
     textAt(row.vLine,40,y+39);
     setText(row.vLine,readable?(local>.55?`V = 2x = ${tight(d.values[i])} · 等 w 决定后混合`:'V = 2x · 等待角色形成'):'V 仍在，但本次不混合');
     opacity(row.vLine,readable?.45+.55*local:.72);
     vis(row.q,0);vis(row.v,0);vis(row.score,1);vis(row.vLine,1);
    }else{
     const y=156+i*42;
     setBox(row.bg,22,y,238,38);
     textAt(row.pos,28,y+20);setText(row.pos,`p${i+1}`);
     textAt(row.x,64,y+20);setText(row.x,tight(d.inputs[i]));
     textAt(row.q,126,y+20);setText(row.q,i===qIndex?(qLocal>.4?tight(d.Q):'…'):'—');
     opacity(row.q,i===qIndex?.4+.6*qLocal:.65);
     textAt(row.k,178,y+20);setText(row.k,local>.45?tight(d.keys[i]):'…');
     opacity(row.k,.35+.65*local);
     textAt(row.v,230,y+20);setText(row.v,local>.55?tight(d.values[i]):'…');
     opacity(row.v,.35+.65*local);
     vis(row.q,1);vis(row.v,1);vis(row.score,0);vis(row.vLine,0);
    }
    if(m&&i===qIndex)row.bg.classList.add('is-query');
   });
   vis(qkvPanelR,!m);vis(qkvRHead,!m);vis(qkvQLine,!m);vis(qkvNote4,!m);vis(qkvNote5,!m);
   qkvRCols.forEach(e=>vis(e,!m));
   if(!m){
    textAt(qkvQLine,296,148);
    setText(qkvQLine,qLocal>.4?`Q = ${fmt(d.Q)} · 来自 p${query}`:'Q = … · 正在形成');
   }
   qkvRRows.forEach((row,i)=>{
    const readable=i<=qIndex,future=!readable;
    const shownScore=pinned?1:phase(p,1+i*.1,1.3+i*.1);
    const y=180+i*40;
    setBox(row.bg,290,y,408,34);
    on(row.bg,'is-future',future);on(row.bg,'is-query',i===qIndex);
    on(row.bg,'is-scan',!future&&scoreT>0&&scoreT<1&&i===Math.min(qIndex,Math.floor(scoreT*allowed)));
    textAt(row.pos,296,y+18);setText(row.pos,`p${i+1}`);
    textAt(row.k,332,y+18);setText(row.k,tight(d.keys[i]));
    opacity(row.k,.45+.55*Math.max(roleT,shownScore));
    textAt(row.score,410,y+18);
    setText(row.score,readable?(shownScore>.2?`s=${d.scores[i].toFixed(2)}`:'s=…'):'s=−∞');
    opacity(row.score,readable?.35+.65*shownScore:1);
    row.score.classList.toggle('att-hot',readable&&scoreT>0&&scoreT<1);
    textAt(row.status,545,y+18);
    setText(row.status,i===qIndex?'当前 Query':readable?'可读':'× 未来');
    row.status.classList.toggle('att-mask',future);
    vis(row.bg,!m);vis(row.pos,!m);vis(row.k,!m);vis(row.score,!m);vis(row.status,!m);
   });
   vis(qkvChainLine,1);vis(qkvPulse,1);
   const chainX=m?[55,180,305]:[90,360,630],chainY=m?500:438;
   attrs(qkvChainLine,{x1:chainX[0],x2:chainX[2],y1:chainY,y2:chainY});
   const chainStage=stage<2?0:stage<4?1:2;
   const pulseX=lerp(chainX[0],chainX[2],clamp(p/5));
   attrs(qkvPulse,{cx:pulseX,cy:chainY});
   qkvChainDots.forEach((dot,i)=>{
    attrs(dot,{cx:chainX[i],cy:chainY});
    on(dot,'is-active',i===chainStage);
    vis(dot,1);
   });
   qkvChainLabels.forEach((e,i)=>{
    textAt(e,chainX[i],chainY+18);
    vis(e,1);
   });
   textAt(qkvStageNote,24,466);vis(qkvStageNote,m);
   setText(qkvStageNote,pinned?'完整状态 · 可回查每个位置的 Q/K/V 与分数':STAGE_NOTES[stage]);
   textAt(qkvFooter,14,544);vis(qkvFooter,m||!pinned);
   setText(qkvFooter,'教学例 WQ=WK=I、WV=2I；真实模型三组投影。');
   if(!m){vis(qkvStageNote,0);vis(qkvFooter,1);textAt(qkvFooter,14,472);setText(qkvFooter,pinned?'完整状态 · 可回查每个位置的 Q/K/V 与分数':STAGE_NOTES[stage]);}
  }

  /* ------------------------------------------------------------ weight paint */
  function paintWeights(m,p,pinned,stage){
   const sDone=pinned||p>=1.35+allowed*.1;
   const eDone=pinned||p>=2.4+allowed*.1;
   const wDone=pinned||p>=3.8;
   const mixDone=pinned||p>=4.6;
   const wT=pinned?1:phase(p,3,3.8);
   const eReveal=i=>pinned?1:phase(p,2+i*.1,2.35+i*.1);
   const cReveal=i=>pinned?1:phase(p,3.8+i*.1,4.15+i*.1);
   textAt(wFormula,14,92);vis(wFormula,!m);
   textAt(wStage,m?12:540,m?112:92);
   vis(wStage,1);
   setText(wStage,pinned?'完整状态 · 数值可回查':STAGE_LABELS[stage]);
   on(wStage,'att-hot',!pinned&&(stage===3||stage===4));

   setBox(wDenomPanel,m?12:14,m?120:102,m?336:692,72);
   textAt(wDenomTitle,m?24:28,m?140:124);vis(wDenomTitle,1);
   textAt(wDenomLine,m?24:200,m?162:124);
   textAt(wWeightLine,m?24:540,m?180:124);
   textAt(wDenomNote,28,146);vis(wDenomNote,!m);

   const shownExps=[];
   for(let i=0;i<allowed;i++){if(pinned||eReveal(i)>=.5)shownExps.push(d.exps[i]);}
   let denomText;
   if(eDone){
    denomText=`e^s: ${d.exps.slice(0,allowed).map(x=>x.toFixed(2)).join(' + ')} = ${d.denominator.toFixed(2)}`;
   }else if(shownExps.length){
    denomText=`e^s: ${shownExps.map(x=>x.toFixed(2)).join(' + ')}${shownExps.length<allowed?' + …':''} ≈ ${shownExps.reduce((a,b)=>a+b,0).toFixed(2)}（累加中）`;
   }else{
    denomText='e^s: 等待分数 s';
   }
   setText(wDenomLine,denomText);
   setText(wWeightLine,wDone?'Σw = 100.0% · 只含可读项':'Σw = 归一化中 · 只含可读项');
   vis(wDenomLine,1);vis(wWeightLine,1);

   wCols.forEach((e,i)=>{
    vis(e,!m);
    on(e,'is-active',(stage===1&&i===2)||(stage===2&&i===3)||(stage===3&&i===4)||(stage===4&&i===5));
   });
   wRows.forEach((row,i)=>{
    const readable=i<=qIndex,future=!readable;
    const shownScore=pinned?1:phase(p,1+i*.1,1.3+i*.1);
    const eT=eReveal(i);
    const cT=cReveal(i);
    let bx,by,trackW;
    if(m){
     const y=200+i*58;
     setBox(row.bg,12,y,336,54);
     textAt(row.pos,24,y+18);setText(row.pos,`p${i+1}`);
     textAt(row.k,60,y+18);setText(row.k,tight(d.keys[i]));
     textAt(row.score,150,y+18);setText(row.score,readable?(shownScore>.2?`s=${d.scores[i].toFixed(2)}`:'s=…'):'s=−∞');
     textAt(row.exp,238,y+18);setText(row.exp,readable?(eT>.2?`e=${d.exps[i].toFixed(2)}`:'e=…'):'e=0');
     bx=24;by=y+27;trackW=110;
     setBox(row.track,bx,by,trackW,6);setBox(row.bar,bx,by,0,6);
     textAt(row.wlabel,142,y+34);setText(row.wlabel,readable?(wDone?`w=${pct(d.weights[i])}`:'w=…'):'w=0');
     textAt(row.contrib,24,y+49);setText(row.contrib,readable?(cT>=.5?`w×V = ${tight(d.contributions[i])}`:'w×V = …'):'—');
    }else{
     const y=198+i*44;
     setBox(row.bg,22,y,676,38);
     textAt(row.pos,30,y+19);setText(row.pos,`p${i+1}`);
     textAt(row.k,64,y+19);setText(row.k,tight(d.keys[i]));
     textAt(row.score,150,y+19);setText(row.score,readable?(shownScore>.2?`s=${d.scores[i].toFixed(2)}`:'s=…'):'s=−∞');
     textAt(row.exp,226,y+19);setText(row.exp,readable?(eT>.2?d.exps[i].toFixed(2):'…'):'0');
     bx=288;by=y+14;trackW=120;
     setBox(row.track,bx,by,trackW,7);setBox(row.bar,bx,by,0,7);
     textAt(row.wlabel,418,y+19);setText(row.wlabel,readable?(wDone?`w=${pct(d.weights[i])}`:'w=…'):'w=0');
     textAt(row.contrib,500,y+19);setText(row.contrib,readable?(cT>=.5?tight(d.contributions[i]):'…'):'—');
    }
    const barW=readable?trackW*d.weights[i]*(pinned?1:wT):0;
    attrs(row.bar,{x:bx,width:String(barW)});
    row.bar.setAttribute('data-row',String(i+1));
    row.bar.setAttribute('data-width',barW.toFixed(2));
    on(row.bg,'is-future',future);on(row.bg,'is-query',i===qIndex);
    row.score.classList.toggle('att-mask',future);
    row.exp.classList.toggle('att-mask',future);
    row.wlabel.classList.toggle('att-mask',future);
    row.contrib.classList.toggle('att-mask',future);
    row.score.classList.toggle('att-hot',stage===1&&!future&&shownScore>0);
    row.exp.classList.toggle('att-hot',stage===2&&!future&&eT>0);
    row.wlabel.classList.toggle('att-hot',stage===3&&!future&&wT>0);
    row.contrib.classList.toggle('att-hot',stage===4&&!future&&cT>0);
    on(row.track,'is-future',future);on(row.bar,'is-future',future);
    vis(row.bg,1);vis(row.pos,1);vis(row.k,1);vis(row.score,1);vis(row.exp,1);vis(row.track,1);vis(row.bar,barW>0.5);vis(row.wlabel,1);vis(row.contrib,1);
   });

   setBox(wOutPanel,m?12:14,m?440:374,m?336:692,m?80:84);
   textAt(wOutEq,m?24:28,m?464:402);
   const revealed=d.inputs.map((_,i)=>i<=qIndex&&(pinned||cReveal(i)>=.5));
   const partial=[0,1].map(j=>d.contributions.reduce((sum,c,i)=>sum+(revealed[i]?c[j]:0),0));
   const count=revealed.filter(Boolean).length;
   if(count===0){
    setText(wOutEq,'m = Σ w·V · 等待权重');
    wOutEq.setAttribute('data-current','pending');
   }else if(count<allowed||!mixDone){
    setText(wOutEq,`m = ${fmt(partial)} · 汇合 ${count}/${allowed}`);
    wOutEq.setAttribute('data-current',fmt(partial));
   }else{
    setText(wOutEq,`m = Σ w·V = ${fmt(d.mixed)}`);
    wOutEq.setAttribute('data-current',fmt(d.mixed));
   }
   wOutEq.setAttribute('data-value',fmt(d.mixed));
   wOutEq.setAttribute('data-mix-count',String(count));
   wOutEq.classList.toggle('att-hot',mixDone&&stage>=4);
   textAt(wOutNote,450,424);vis(wOutNote,!m);
   textAt(wMixNote,m?24:450,m?510:448);
   setText(wMixNote,count===0?'逐项汇合：等待权重完成':count<allowed?`已汇合 ${count}/${allowed} 项`:(mixDone?'全部可读项已汇合':'逐项汇合中'));
   vis(wMixNote,1);
   const barW=m?300:400,baseX=m?24:28,trackY1=m?474:418,trackY2=m?488:434;
   attrs(wTrack1,{x:baseX,y:trackY1,width:barW,height:9});
   attrs(wTrack2,{x:baseX,y:trackY2,width:barW,height:9});
   const cursor=[0,0];
   wOutSegs.forEach(({i,j,e})=>{
    const show=revealed[i];
    const width=show?d.contributions[i][j]/Math.max(d.mixed[j],0.0001)*barW:0;
    attrs(e,{x:baseX+cursor[j],y:j?trackY2:trackY1,width,height:9});
    if(show)cursor[j]+=width;
    vis(e,show&&width>0.01);
   });
   textAt(wFooter,14,m?544:472);
   setText(wFooter,m?'s→e^s→w→ΣwV；e^s 只对可读项求和。':'s = Q·K/√2；e^s 只对可读项求和；w×V 按维相加。');
   vis(wFooter,1);
   vis(wOutPanel,1);vis(wTrack1,1);vis(wTrack2,1);vis(wOutEq,1);
   svg.dataset.mixProgress=(pinned?1:phase(p,3.8,4.6)).toFixed(3);
  }

  /* -------------------------------------------------------------- mask paint */
  function paintMask(m,p,pinned){
   const cell=m?38:42,gap=m?6:5,gridX=m?62:64,gridY=m?170:154;
   if(m){
    setBox(mPanel,12,104,336,252);
    textAt(mHead,24,126);textAt(mSub,24,146);setText(mSub,'行 Q · 列 K');
    setBox(mSummaryPanel,12,366,336,118);vis(mSummaryPanel,1);
    textAt(mReadNote,24,390);
    setText(mReadNote,d.allowed===4?'可读 p1–p4 · 无未来项':`可读 p1–p${d.allowed} · 未来 p${d.allowed+1}${d.allowed<3?'、p'+(d.allowed+2):''}${d.allowed<2?'、p'+(d.allowed+3):''}`);
    vis(mReadNote,1);
    vis(mDetailPanel,0);vis(mDetailHead,0);vis(mDetailSub,0);mDetailCols.forEach(e=>vis(e,0));mDetailRows.forEach(row=>{vis(row.bg,0);vis(row.pos,0);vis(row.score,0);vis(row.exp,0);vis(row.weight,0);vis(row.status,0);});
    textAt(mDenom,24,416);textAt(mWeightSum,190,416);textAt(mOut,24,448);textAt(mNote,24,470);
    textAt(mFooter,14,540);
   }else{
    setBox(mPanel,14,94,302,336);
    textAt(mHead,28,116);textAt(mSub,28,134);setText(mSub,'行 = Query · 列 = Key');
    vis(mSummaryPanel,0);vis(mReadNote,0);
    vis(mDetailPanel,1);vis(mDetailHead,1);vis(mDetailSub,1);mDetailCols.forEach(e=>vis(e,1));
    textAt(mDenom,344,340);textAt(mWeightSum,344,360);textAt(mOut,344,388);textAt(mNote,344,410);
    textAt(mFooter,14,470);
   }
   textAt(mHead,m?24:28,m?126:116);setText(mHead,'因果遮罩');
   mCells.forEach(x=>{
    const future=x.c>x.r,cur=x.r===qIndex;
    const px=gridX+x.c*(cell+gap),py=gridY+x.r*(cell+gap);
    setBox(x.cell,px,py,cell,cell);
    x.cell.classList.toggle('is-open',!future);
    x.cell.classList.toggle('is-future',future);
    x.cell.classList.toggle('is-current',cur&&!future);
    const cx=px+cell/2,cy=py+cell/2;
    attrs(x.dot,{cx,cy,r:cell*.07});vis(x.dot,!future);
    const rr=cell*.19;
    x.x1.setAttribute('d',`M${cx-rr},${cy-rr}L${cx+rr},${cy+rr}`);
    x.x2.setAttribute('d',`M${cx+rr},${cy-rr}L${cx-rr},${cy+rr}`);
    vis(x.x1,future);vis(x.x2,future);
    x.g.setAttribute('aria-label',`p${x.r+1} 读 p${x.c+1}：${future?'未来，不可读':'可读'}`);
   });
   mCols.forEach((e,c)=>{textAt(e,gridX+c*(cell+gap)+cell/2,gridY-8);vis(e,1);});
   mRows.forEach((e,r)=>{
    textAt(e,m?44:44,gridY+r*(cell+gap)+cell-(m?10:13));
    e.classList.toggle('is-current',r===qIndex);
    e.setAttribute('fill',r===qIndex?'#9ff0ff':'#91a9bb');
    vis(e,1);
   });
   setBox(mRowBox,gridX-3,gridY+qIndex*(cell+gap)-3,4*(cell+gap)-gap+6,cell+6);
   textAt(mLegend,m?24:28,m?352:358);
   setText(mLegend,m?'✓ 可读 · × 未来：e^s=0':'✓ 可读 · × 未来：e^s = 0，不进分母');
   vis(mLegend,1);

   mDetailRows.forEach((row,i)=>{
    const readable=i<=qIndex,future=!readable;
    const y=170+i*36;
    setBox(row.bg,336,y,362,30);
    textAt(row.pos,344,y+20);setText(row.pos,`p${i+1}`);
    textAt(row.score,390,y+20);setText(row.score,readable?`s=${d.scores[i].toFixed(2)}`:'s=−∞');
    textAt(row.exp,452,y+20);setText(row.exp,readable?d.exps[i].toFixed(2):'0');
    textAt(row.weight,524,y+20);setText(row.weight,readable?pct(d.weights[i]):'0%');
    textAt(row.status,620,y+20);setText(row.status,readable?'✓ 可读':'× 未来');
    on(row.bg,'is-future',future);on(row.bg,'is-query',i===qIndex);
    row.score.classList.toggle('att-mask',future);
    row.exp.classList.toggle('att-mask',future);
    row.weight.classList.toggle('att-mask',future);
    row.status.classList.toggle('att-mask',future);
    vis(row.bg,!m);vis(row.pos,!m);vis(row.score,!m);vis(row.exp,!m);vis(row.weight,!m);vis(row.status,!m);
   });

   setText(mDenom,m?`Σe^s = ${d.denominator.toFixed(2)}`:`Σ e^s（仅可见） = ${d.denominator.toFixed(2)}`);
   setText(mWeightSum,m?'w 合计 = 100%':`w 合计 = ${(d.weightSum*100).toFixed(1)}%`);
   setText(mOut,`m = ${fmt(d.mixed)}`);
   mOut.classList.toggle('att-hot',pinned||p>=4.6);
   setText(mNote,d.allowed===4?'本例 Query 在最后位置：没有未来项。':'未来项 e^s = 0，从未加入求和。');
   setText(mFooter,d.allowed===4?'遮罩只决定可读范围；本例没有未来项。':'遮罩不改变 K/V，只限制本次可读范围。');
   vis(mDenom,1);vis(mWeightSum,1);vis(mOut,1);vis(mNote,!m);vis(mFooter,1);
  }

  /* ---------------------------------------------------------------- dispatch */
  return (p,m)=>{
   const focus=svg.dataset.attentionFocus;
   const pinned=!!focus&&svg.dataset.attentionFocusPreview==='true';
   const mode=focus==='qkv'||focus==='weights'||focus==='mask'
    ?focus
    :(p>=4.7?'mask':p>=1.5?'weights':'qkv');
   const stage=p>=4.7?5:p>=3.8?4:p>=3?3:p>=2?2:p>=1?1:0;
   svg.dataset.queryPosition=String(query);
   svg.dataset.allowed=String(d.allowed);
   svg.dataset.futureCount=String(4-d.allowed);
   svg.dataset.weightSum=d.weightSum.toFixed(3);
   svg.dataset.denominator=d.denominator.toFixed(4);
   svg.dataset.output='['+d.mixed.map(x=>x.toFixed(2)).join(',')+']';
   svg.dataset.attentionMode=mode;
   svg.dataset.causalStage=String(stage);
   paintHeader(m,mode);
   paintQKV(m,p,pinned,stage);
   paintWeights(m,p,pinned,stage);
   paintMask(m,p,pinned);
   for(const[k,g]of Object.entries(modeGroups))vis(g,k===mode);
   vis(qkvFooter,mode==='qkv');
   vis(wFooter,mode==='weights');
   vis(mFooter,mode==='mask');
   const denom=mode==='weights'?wDenomLine:mDenom;
   const output=mode==='weights'?wOutEq:mOut;
   denom.setAttribute('data-visible-mode',mode);
   output.setAttribute('data-visible-mode',mode);
  };

 });
};
})();
