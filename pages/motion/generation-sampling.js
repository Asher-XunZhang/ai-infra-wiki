/* Chapter 1, lesson 02. One independent sampling concept:
 * logits z -> e^z -> denominator -> probabilities p -> choice.
 * The old mv0/1 generation boards remain available as legacy entries; this
 * module only replaces the default scene and keeps the old visual untouched.
 *
 * Stable entry: generation.html?concept=sampling (mv=2 is an alias).
 * Legacy entries: generation.html?mv=0 (greedy) and ?mv=1 (fixed draw). */
(function(root){
'use strict';
const V=root.MechanismVisuals,D=root.TeachingDrawing,M=root.FoundationModel;
if(!V||!D||!M)return;
if(root.__generationSampling)return;
root.__generationSampling=true;
const {board,phase}=D;
const U_VALUES=[.25,.75,.95];
const FALLBACK=[
 {word:'学习',p:.6,z:Math.log(6)},
 {word:'休息',p:.3,z:Math.log(3)},
 {word:'。',p:.1,z:0}
];
const fmtNum=x=>{
 const n=Math.round(x);
 return Math.abs(x-n)<1e-9?String(n):String(Number(x.toFixed(2)));
};
const fmtP=p=>p.toFixed(2);
const compactP=p=>p.toFixed(2).replace(/^0/,'');
const pct=p=>Math.round(p*100)+'%';
const displayWord=w=>w==='。'?'句号':w;

function rowsFrom(s){
 const source=s&&s.data&&Array.isArray(s.data.scores)&&s.data.scores.length===3?s.data.scores:FALLBACK;
 return source.map((r,i)=>({
  word:r.word||FALLBACK[i].word,
  p:Number.isFinite(r.p)?r.p:FALLBACK[i].p,
  z:Number.isFinite(r.z)?r.z:Math.log(FALLBACK[i].p)
 }));
}
function makeButton(H,label,ariaLabel){
 const g=H.node('g',{class:'gs-btn',role:'button',tabindex:'-1','aria-pressed':'false','aria-label':ariaLabel||label,opacity:'0'});
 const r=H.node('rect',{rx:7,width:10,height:10});
 const t=H.node('text',{x:0,y:0,'text-anchor':'middle'},label);
 g.append(r,t);
 const run=()=>g.gsRun?.();
 g.addEventListener('click',e=>{e.stopPropagation();run();});
 g.addEventListener('keydown',e=>{
  if(e.key==='Enter'||e.key===' '||e.key==='Spacebar'){e.preventDefault();e.stopPropagation();run();}
 });
 return {g,r,t,label};
}
function placeButton(H,btn,x,y,w,h,enabled,active,visible){
 H.set(btn.g,{transform:`translate(${x} ${y})`,tabindex:enabled?'0':'-1',style:enabled?'':'pointer-events:none'});
 H.set(btn.r,{width:w,height:h});
 H.text(btn.t,w/2,h/2+4);
 H.show(btn.g,visible?1:0);
 btn.g.setAttribute('aria-pressed',active?'true':'false');
 btn.g.classList.toggle('is-on',active);
}

function samplingBoard(svg,viewport,s){
 const rows=rowsFrom(s);
 const exps=rows.map(r=>Math.exp(r.z));
 const denom=exps.reduce((a,b)=>a+b,0);
 const maxExp=Math.max(...exps);
 const baseChoice=M.chooseToken(rows,null);
 const intervals=baseChoice.intervals;
 const greedy=baseChoice.selected;
 const pSum=rows.reduce((a,r)=>a+r.p,0);
 const zSum=rows.reduce((a,r)=>a+r.z,0);
 let method='greedy',u=.75,lastP=0,lastM=false;
 let repaint=()=>{};
 function pauseFilm(){
  const film=svg.closest('.mechanism-film');
  const b=film?.querySelector('[data-motion-action=play]');
  if(b&&b.getAttribute('aria-pressed')==='true')b.click();
 }
 function change(){pauseFilm();repaint();}

 return board(svg,viewport,'generation-sampling',()=>590,H=>{
  const {txt,rect,text,box,show,token,move,set}=H;

  // Backgrounds first. Later text, bars and tokens must stay above them.
  const tablePanel=rect('teach-panel'),selectPanel=rect('teach-panel');
  const rowBgs=rows.map((r,i)=>rect('teach-row teach-color-'+(i+2)));
  const expTracks=rows.map(()=>rect('teach-track'));
  const pTracks=rows.map(()=>rect('teach-track'));
  const expBars=rows.map((r,i)=>rect('teach-value teach-color-'+(i+2)));
  const pBars=rows.map((r,i)=>rect('teach-value teach-color-'+(i+2)));
  const axisBg=rect('gs-axis-bg');
  const segs=rows.map((r,i)=>rect('teach-contribution teach-color-'+(i+2)));
  const ticks=[0,1].map(()=>H.node('line',{class:'gs-tick'}));
  const selectedOutline=rect('gs-selected');
  const needle=H.node('g',{class:'gs-needle'});
  const needleLine=H.node('line',{});
  const needleTip=H.node('path',{d:'M-5,-8 L5,-8 L0,-1 Z'});
  needle.append(needleLine,needleTip);
  const greedyMark=H.node('path',{class:'gs-marker',d:'M-6,-8 L6,-8 L0,0 Z'});

  const heading=txt('','teach-heading'),sub=txt('','teach-small');
  const ctxLabel=txt('上下文','teach-small');
  const ctxA=token('sampling-ctx-a','我',40,'teach-token teach-color-0');
  const ctxB=token('sampling-ctx-b','想',40,'teach-token teach-color-0');

  const hCand=txt('','teach-small'),hZ=txt('','teach-small'),hExp=txt('','teach-small'),hP=txt('','teach-small');
  const rowNames=rows.map(r=>txt('','gs-word'));
  const zVals=rows.map(()=>txt('','teach-number'));
  const zSubs=rows.map(()=>txt('','teach-small'));
  const expVals=rows.map(()=>txt('','teach-number'));
  const pPcts=rows.map(()=>txt('','teach-number'));
  const arrows1=rows.map(()=>txt('','gs-arrow'));
  const arrows2=rows.map(()=>txt('','teach-small'));
  const sumExpr=txt('','gs-formula'),normExpr=txt('','gs-formula'),zNote=txt('','teach-small');
  const previewNote=txt('','teach-small');

  const axisTitle=txt('','teach-heading');
  const segWords=rows.map(()=>txt('','gs-seg-word'));
  const segPcts=rows.map(()=>txt('','gs-seg-pct'));
  const needleLabel=txt('','teach-small'),greedyLabel=txt('','teach-small');
  const pickLabel=txt('','teach-small'),verdict=txt('','gs-formula'),pending=txt('','teach-small');
  const compare=txt('','teach-small'),consequence=txt('','teach-small');
  const separator=H.node('line',{class:'gs-separator'});
  const pickChip=token('sampling-pick','',68,'teach-token teach-color-2');
  const foot=txt('','teach-small');

  const methodGreedy=makeButton(H,'贪心 · 取最高','选择规则：贪心，取最高概率');
  const methodSampling=makeButton(H,'固定抽样 · 用 u','选择规则：固定抽样');
  methodGreedy.g.gsRun=()=>{method='greedy';change();};
  methodSampling.g.gsRun=()=>{method='sampling';change();};
  const uButtons=U_VALUES.map(v=>{
   const b=makeButton(H,'u='+v.toFixed(2),'固定抽样数 u='+v.toFixed(2)+'，按当前分布重新计算落点');
   b.g.gsRun=()=>{u=v;method='sampling';change();};
   return b;
  });

  function draw(p,m){
   lastP=p;lastM=m;
   const controlsOn=p>=4;
   const choice=p>=5?M.chooseToken(rows,method==='sampling'?u:null):null;
   const picked=choice?choice.selected:null;
   const pickedIndex=picked?rows.findIndex(r=>r.word===picked):-1;
   const pickedInterval=picked?choice.intervals.find(x=>x.word===picked):null;
   const result=Boolean(picked)&&p>=5;
   const selectOn=phase(p,2.78,3);

   const panelX=m?8:16,panelW=m?344:688,tableY=m?58:60;
   const rowY0=m?108:118,rowGap=m?44:46;
   const zX=m?64:96,expX=m?132:216,expW=m?44:64,expValX=m?182:288;
   const pX=m?228:348,pW=m?80:240,pPctX=m?314:598;
   const arrow1X=m?112:184,arrow2X=318,headerY=m?80:86;

   // One persistent header. Context tokens are omitted on the narrow board;
   // the same "我 想" is written into the subtitle there.
   text(heading,m?14:16,m?24:26,m?'从分数到选择':'从分数到选择：logits 不是概率');
   text(sub,m?14:16,m?44:47,m?'z 不是概率 → e^z → p → 选择':'同一上下文「我 想」；z 是原始分数，不是概率。先 exp，再归一化，最后按规则选择。');
   if(m){show(ctxLabel,0);show(ctxA.g,0);show(ctxB.g,0);}
   else{
    text(ctxLabel,482,28);show(ctxLabel,1);
    move(ctxA,540,12,1);move(ctxB,584,12,1);
   }

   box(tablePanel,panelX,tableY,panelW,234);show(tablePanel,1);
   text(hCand,m?18:34,headerY,'候选');show(hCand,1);
   text(hZ,zX,headerY,m?'z 分数':'z 原始分数');show(hZ,1);
   text(hExp,expX,headerY,'e^z');show(hExp,phase(p,.2,.8));
   text(hP,pX,headerY,m?'p':(p>=3?'p 概率（合计 1）':'p 概率'));show(hP,phase(p,1.9,2.7));

   rows.forEach((r,i)=>{
    const y=rowY0+i*rowGap;
    const isPicked=pickedIndex===i&&result;
    set(rowBgs[i],{class:'teach-row teach-color-'+(i+2)+(isPicked?' is-query':'')});
    box(rowBgs[i],m?14:24,y-24,m?332:672,40);show(rowBgs[i],1);
    text(rowNames[i],m?18:34,y+4,displayWord(r.word));show(rowNames[i],1);
    text(zVals[i],zX,y+1,fmtNum(r.z));show(zVals[i],1);
    text(zSubs[i],zX,y+15,'ln '+fmtNum(exps[i]));show(zSubs[i],1);
    // A bar keeps its real share from the first visible frame. The construction
    // step reveals rows one by one instead of animating a partial width that
    // would disagree with the printed e^z / p value. e^z uses max(e^z)=6 as the
    // full track; p uses 1.00 as the full track.
    const expIn=phase(p,.18+i*.14,.5+i*.14);
    text(arrows1[i],arrow1X,y+4,'→');show(arrows1[i],expIn);

    box(expTracks[i],expX,y-10,expW,12);show(expTracks[i],phase(p,.08+i*.1,.35+i*.1));
    box(expBars[i],expX,y-10,expW*(exps[i]/maxExp),12);show(expBars[i],expIn);
    text(expVals[i],expValX,y+4,fmtNum(exps[i]));show(expVals[i],expIn);

    if(!m){text(arrows2[i],arrow2X,y+4,'÷10');show(arrows2[i],phase(p,2.05,2.7));}
    else show(arrows2[i],0);

    const pIn=phase(p,2.16+i*.14,2.5+i*.14);
    box(pTracks[i],pX,y-10,pW,12);show(pTracks[i],phase(p,2.05+i*.08,2.35+i*.08));
    box(pBars[i],pX,y-10,pW*r.p,12);show(pBars[i],pIn);
    text(pPcts[i],pPctX,y+4,pct(r.p));show(pPcts[i],pIn);
   });

   const sumLine=m
    ? `Σe^z = ${exps.map(fmtNum).join(' + ')} = ${fmtNum(denom)}`
    : `分母 Σe^z = ${exps.map(fmtNum).join(' + ')} = ${fmtNum(denom)}`;
   const normLine=m
    ? `p = e^z ÷ ${fmtNum(denom)}，Σp = ${pSum.toFixed(3)}`
    : `p = e^z ÷ ${fmtNum(denom)} → Σp = ${pSum.toFixed(3)}`;
   text(sumExpr,m?16:30,m?230:258,sumLine);show(sumExpr,phase(p,1.1,1.85));
   text(normExpr,m?16:348,m?248:258,normLine);show(normExpr,phase(p,2.1,2.85));
   const zNoteText=m
    ?(p>=3?`z 合计 ${zSum.toFixed(2)}；p 合计 ${pSum.toFixed(3)}。`:`z 合计 ${zSum.toFixed(2)} ≠ 1，不是概率。`)
    :(p>=3?`logits z 合计 ${zSum.toFixed(2)}；p 合计 ${pSum.toFixed(3)}，只有 p 是概率。`:`z 是原始分数：可正可负，合计 ${zSum.toFixed(2)} ≠ 1，不能当概率。`);
   text(zNote,m?16:30,m?268:280,zNoteText);show(zNote,1);

   const scaleWide='条长尺度：e^z 满格 = 6（最大项）；p 满格 = 1.00；p 条长 = p 值。';
   const scaleNarrow='条长：e^z 满格=6；p 满格=1.00；p 条长=p 值。';
   text(previewNote,m?16:30,354,m?scaleNarrow:scaleWide);
   show(previewNote,p>=.4&&p<2.9?1:0);

   // The probability axis appears only after every p bar has been revealed.
   box(selectPanel,panelX,m?308:304,panelW,m?252:256);show(selectPanel,selectOn);
   text(axisTitle,m?16:30,m?322:330,p>=4?(m?'同一组 p':'同一组 p：按规则选择'):'归一化后的概率轴');
   show(axisTitle,selectOn);

   const axisX=m?16:40,axisY=m?386:368,axisW=m?328:640,axisH=m?40:44;
   box(axisBg,axisX,axisY,axisW,axisH);show(axisBg,selectOn);
   segs.forEach((seg,i)=>{
    const start=intervals[i].start,end=intervals[i].end;
    const sx=axisX+start*axisW,sw=Math.max(1,(end-start)*axisW);
    box(seg,sx,axisY,sw,axisH);show(seg,selectOn);
    text(segWords[i],sx+sw/2,axisY+(m?22:25),displayWord(rows[i].word));show(segWords[i],selectOn);
    text(segPcts[i],sx+sw/2,axisY+(m?35:40),pct(rows[i].p));show(segPcts[i],selectOn);
   });
   ticks.forEach((tick,i)=>{
    const x=axisX+intervals[i].end*axisW;
    set(tick,{x1:x,y1:axisY,x2:x,y2:axisY+axisH});
    show(tick,selectOn);
   });

   if(result&&pickedInterval){
    box(selectedOutline,axisX+pickedInterval.start*axisW,axisY,Math.max(1,(pickedInterval.end-pickedInterval.start)*axisW),axisH);
    show(selectedOutline,1);
   }else show(selectedOutline,0);

   const greedyStart=intervals[rows.findIndex(r=>r.word===greedy)].start;
   const greedyWidth=intervals[rows.findIndex(r=>r.word===greedy)].p;
   const greedyX=axisX+(greedyStart+greedyWidth/2)*axisW;
   set(greedyMark,{transform:`translate(${greedyX} ${axisY-2})`});
   show(greedyMark,result&&method==='greedy'?1:0);
   text(greedyLabel,greedyX,axisY-16,'最高 p');
   show(greedyLabel,result&&method==='greedy'?1:0);

   const needleX=axisX+u*axisW;
   set(needle,{transform:`translate(${needleX} ${axisY})`});
   set(needleLine,{x1:0,y1:-13,x2:0,y2:-6});
   show(needle,result&&method==='sampling'?1:0);
   text(needleLabel,needleX,axisY-16,'u='+u.toFixed(2));
   set(needleLabel,{'text-anchor':needleX>axisX+axisW-40?'end':'middle'});
   show(needleLabel,result&&method==='sampling'?1:0);

   const uOn=controlsOn&&method==='sampling';
   uButtons.forEach((b,i)=>{
    placeButton(H,b,m?16+i*84:40+i*76,m?434:422,m?76:70,26,uOn,Math.abs(u-U_VALUES[i])<1e-9,uOn);
   });
   placeButton(H,methodGreedy,m?14:438,m?330:310,m?158:120,28,controlsOn,method==='greedy',controlsOn);
   placeButton(H,methodSampling,m?188:566,m?330:310,m?158:120,28,controlsOn,method==='sampling',controlsOn);

   const pendingOn=p>=3&&p<5;
   text(pending,m?16:40,m?494:488,p>=4?'尚未选择：先确认规则，下一步看落点。':'现在只有概率；下一步才选择。');
   show(pending,pendingOn?1:0);

   show(pickChip.g,result?1:0);show(pickLabel,result?1:0);show(verdict,result?1:0);
   if(result){
    const color=pickedIndex+2;
    text(pickChip.t,34,22,displayWord(picked));
    set(pickChip.g,{class:'teach-token teach-color-'+color});
    pickChip.g.setAttribute('aria-label','选中的 token：'+displayWord(picked));
    move(pickChip,m?52:76,m?472:464,1);
    text(pickLabel,m?16:40,m?494:486,'选中');
    let verdictText;
    if(method==='greedy'){
     verdictText=m
      ?`贪心 → ${displayWord(picked)}（p=${fmtP(rows[pickedIndex].p)}）`
      :`贪心：取最高 p → ${displayWord(picked)}（p=${fmtP(rows[pickedIndex].p)}）`;
    }else{
     verdictText=m
      ?`u=${compactP(u)} → ${displayWord(picked)} [${compactP(pickedInterval.start)},${compactP(pickedInterval.end)})`
      :`u=${u.toFixed(2)} ∈ [${pickedInterval.start.toFixed(2)}, ${pickedInterval.end.toFixed(2)}) → ${displayWord(picked)}`;
    }
    text(verdict,m?124:156,m?494:488,verdictText);
   }

   const compareOn=p>=6;
   set(separator,{x1:panelX+24,y1:m?515:509,x2:panelX+panelW-24,y2:m?515:509});
   show(separator,compareOn?1:0);
   const samplingWord=M.chooseToken(rows,u).selected;
   text(compare,m?16:40,m?534:528,m
    ?`同分布：贪心→${displayWord(greedy)}；u=${compactP(u)}→${displayWord(samplingWord)}。`
    :`同一组 p：贪心 → ${displayWord(greedy)}；固定抽样 u=${u.toFixed(2)} → ${displayWord(samplingWord)}。`);
   show(compare,compareOn?1:0);
   text(consequence,m?16:40,m?552:546,m
    ?'选中项进入下一轮；下一轮 p 要重算。'
    :'选中的 token 成为下一轮输入；上下文变了，下一轮 p 要重算。');
   show(consequence,compareOn?1:0);

   text(foot,16,578,m
    ?'教学假设，不是真实模型分数或正确率。'
    :'数值是教学假设；[ln6, ln3, 0] 这组分数只用于本例，不是训练模型的真实分数或正确率。');
   show(foot,1);

   svg.dataset.samplingMethod=method;
   svg.dataset.samplingU=u.toFixed(2);
   svg.dataset.samplingPicked=result?picked:'';
   svg.dataset.samplingStage=p<1?'logits':p<2?'exp':p<3?'sum':p<4?'normalize':p<5?'rules':p<6?'selected':'compare';
   svg.dataset.samplingDenominator=fmtNum(denom);
   svg.dataset.samplingProbabilities=rows.map(r=>fmtP(r.p)).join(',');
  }
  repaint=()=>draw(lastP,lastM);
  return draw;
 });
}

const samplingScene={
 concept:'sampling',mv:2,default:true,
 name:'logits → softmax → 选择',
 title:'Logits → Softmax → 选择：分数怎样变成一次选择？',
 intro:'同一上下文「我 想」下，模型给三个候选原始分数 z。先把 z 变成正数并求和，再归一化成概率 p，最后用贪心或固定 u 选择。',
 boundary:'数值是教学假设：[ln6, ln3, 0] → [6, 3, 1] → 分母 10 → [.6, .3, .1]，不是训练模型的真实分数或正确率。贪心取最高分；同一条分布上，固定抽样 u=.75 落在休息区间 [.6, .9)，区间左闭右开。改变方法或 u 会按当前分布重新计算；选择结果改变下一轮输入，下一轮分布要重算。',
 data:M.snapshot('generation',0,6),
 kind:'generation-sampling',
 frames:[
  {title:'先看原始分数 z',caption:'上下文固定为「我 想」。三个候选各有原始分数 z；z 可正可负，合计也不等于 1，所以它不是概率。'},
  {title:'指数变换：每个 z 变成 e^z',caption:'对每个 z 取 e^z：分数变成正数，大小顺序不变，但数值还不是份额。'},
  {title:'求和：分母是 10',caption:'把三个 e^z 相加得到分母 Σ=10；它只表示这组候选的指数总量。'},
  {title:'归一化：p 合计为 1',caption:'每个 e^z ÷ 10 得到 p：0.60、0.30、0.10。现在三份合计才是 1。'},
  {title:'先选规则，不选 token',caption:'同一组 p 可以用不同规则选择：贪心比大小，固定抽样让 u 落进概率区间。这里只确定规则，还没有选 token。'},
  {title:'按当前规则落点并选中',caption:'按当前规则在概率轴上落点；被选中的区间、候选行和结果使用同一种身份色。'},
  {title:'同一分布的比较与后果',caption:'同一组 p 下，方法不同会选择不同 token；切换方法或 u 会按同一分布重算。选中的 token 成为下一轮输入，下一轮要用新上下文重算 p。'}
 ],
 stepDurations:[2.2,2.0,1.8,2.2,2.0,3.2]
};

const legacy=(root.TeachingLessons&&root.TeachingLessons['f-generation']||[]).map((scene,i)=>({
 ...scene,legacy:true,mv:i,
 name:'自回归完整过程 · '+(i?'固定抽样':'贪心'),
 frames:scene.frames.map(f=>({...f}))
}));
root.TeachingLessons=root.TeachingLessons||{};
root.TeachingLessons['f-generation']=[samplingScene,...legacy];

const legacyVisual=V['f-generation'];
V['f-generation']=(svg,viewport,s)=>{
 const film=svg.closest('.mechanism-film');
 if(film)film.classList.toggle('is-generation-sampling',s.concept==='sampling');
 if(s.concept!=='sampling')return legacyVisual?legacyVisual(svg,viewport,s):null;
 return samplingBoard(svg,viewport,s);
};
})(typeof globalThis==='object'?globalThis:this);
