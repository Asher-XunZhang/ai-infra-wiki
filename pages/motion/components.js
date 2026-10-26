/* Six independent Transformer component lessons. One player, six stable
 * concept entries. Every board shows one input, one local operation and one
 * result; the in-board buttons change the teaching condition without changing
 * the accepted Attention scenes. Numbers come from components-model.js. */
(function(){
'use strict';
const V=window.MechanismVisuals,D=window.TeachingDrawing,M=window.ComponentModel;
if(!V||!D||!M)return;
const NS='http://www.w3.org/2000/svg';
const {board}=D;
let serial=0;
const fmt=(v,d=2)=>'['+v.map(x=>{const n=Number(x.toFixed(d));return String(n);}).join(', ')+']';
const pct=w=>(w*100).toFixed(1)+'%';
const clamp=n=>Math.max(0,Math.min(1,n));
const phase=(p,a,b)=>{const t=clamp((p-a)/(b-a));return t*t*(3-2*t);};
const num=(v,d=4)=>{if(!Number.isFinite(v))return '—';if(Math.abs(v)<5e-9)return '0';return String(Number(v.toFixed(d)));};

function mk(tag,attrs={},parent){
 const e=document.createElementNS(NS,tag);
 for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);
 if(parent)parent.append(e);
 return e;
}
function txt(parent,cls,x,y,value=''){
 const e=mk('text',{class:cls,x,y},parent);
 if(value!==undefined)e.textContent=value;
 return e;
}
function rect(parent,cls,x,y,w,h,rx=7){
 return mk('rect',{class:cls,x,y,width:w,height:h,rx},parent);
}
function put(e,x,y,w,h){e.setAttribute('x',x);e.setAttribute('y',y);e.setAttribute('width',w);e.setAttribute('height',h);}
function place(e,x,y){e.setAttribute('x',x);e.setAttribute('y',y);}
function setText(e,value){if(e.textContent!==String(value))e.textContent=String(value);}
function show(e,on){e.style.display=on?'':'none';e.setAttribute('aria-hidden',on?'false':'true');}
function fade(e,v){e.style.opacity=clamp(v).toFixed(3);}
function size(e,px){e.style.fontSize=px+'px';return e;}
function hit(g,label,fn){
 g.setAttribute('role','button');g.setAttribute('tabindex','0');g.setAttribute('aria-label',label);g.classList.add('cmp-hit');
 g.addEventListener('click',e=>{e.stopPropagation();fn();});
 g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '||e.key==='Spacebar'){e.preventDefault();e.stopPropagation();fn();}});
}
function pressed(g,on){g.setAttribute('aria-pressed',on?'true':'false');}
function btn(parent,x,y,w,h,label,fn,cls=''){
 const g=mk('g',{class:'cmp-btn '+cls,role:'button',tabindex:'0','aria-pressed':'false'},parent);
 const r=mk('rect',{rx:7,width:w,height:h},g);
 const t=mk('text',{x:w/2,y:h/2+5,'text-anchor':'middle'},g);t.textContent=label;
 hit(g,label,fn);g.cmpRect=r;g.cmpLabel=t;g.cmpLabelText=label;g.cmpWidth=w;g.cmpHeight=h;
 return g;
}
function placeBtn(g,x,y,w,h,label,on){
 g.setAttribute('transform',`translate(${x} ${y})`);
 put(g.cmpRect,0,0,w,h);place(g.cmpLabel,w/2,h/2+5);
 if(g.cmpLabel.textContent!==label)g.cmpLabel.textContent=label;
 pressed(g,on);
}
function backToBlock(svg,x,y,w=150){
 return btn(svg,x,y,w,26,'← 回到 Block 第 1 层',()=>{
  svg.dispatchEvent(new CustomEvent('component-focus',{bubbles:true,detail:{concept:'block',step:2}}));
 },'cmp-back');
}
function header(svg,title,sub,kicker){
 const bg=mk('rect',{class:'att-board',rx:14},svg);
 const band=mk('rect',{class:'cmp-band'},svg);
 const t1=txt(svg,'att-mode-title',18,30,title);
 const t2=txt(svg,'att-mode-sub',18,50,sub);
 const t3=txt(svg,'att-kicker',18,70,kicker);
 return {bg,band,t1,t2,t3};
}
function placeHeader(h,m,W){
 put(h.bg,0,0,W,m?700:560);put(h.band,0,0,W,78);
 if(m)h.t2.style.fontSize='10.5px';else h.t2.style.fontSize='';
 place(h.t1,m?14:18,m?27:30);place(h.t2,m?14:18,m?47:50);place(h.t3,m?14:18,m?67:70);
}
function footer(svg,text,m,HT){
 return txt(svg,'att-foot',m?14:18,HT-12,text);
}
function pause(svg){const b=svg.closest('.mechanism-film')?.querySelector('[data-motion-action=play]');if(b&&b.getAttribute('aria-pressed')==='true')b.click();}
function seekTo(svg,step){svg.dispatchEvent(new CustomEvent('component-seek',{bubbles:true,detail:{step}}));}
function marker(svg,color='#7893a4'){
 const id='cmp-arrow-'+(++serial);
 const defs=mk('defs',{},svg);
 const m=mk('marker',{id,viewBox:'0 0 10 10',refX:'8',refY:'5',markerWidth:'5',markerHeight:'5',orient:'auto-start-reverse'},defs);
 mk('path',{d:'M0 0L10 5L0 10Z',fill:color},m);
 return id;
}
function wire(parent,id,d,cls='cmp-wire'){
 return mk('path',{class:cls,d,fill:'none','marker-end':`url(#${id})`},parent);
}

/* --------------------------------------------------------------- 01 RoPE */
function ropeBoard(svg,H){
 const h=header(svg,'RoPE：位置决定 Q、K 的旋转角','q=k=[1,0]，θ=π/4；Q 和 K 都用各自位置旋转。','位置 → 角度；播放进度只表示旋转比例，不表示位置。');
 const opts=[
  {label:'pQ=1, pK=0',pq:1,pk:0},
  {label:'两者 +3',pq:4,pk:3},
  {label:'pQ=3, pK=1',pq:3,pk:1}
 ];
 const buttons=opts.map((c,i)=>btn(svg,0,0,200,30,c.label,()=>{
  cond=i;pause(svg);seekTo(svg,3);draw(lastP,lastM);
 }));
 const circle=mk('circle',{class:'cmp-circle',cx:180,cy:300,r:104},svg);
 const axisX=mk('line',{class:'cmp-axis',x1:76,y1:300,x2:284,y2:300},svg);
 const axisY=mk('line',{class:'cmp-axis',x1:180,y1:196,x2:180,y2:404},svg);
 const base=mk('line',{class:'cmp-dash',x1:180,y1:300,x2:284,y2:300},svg);
 const qTarget=mk('line',{class:'cmp-target-q',x1:180,y1:300,x2:180,y2:300},svg);
 const kTarget=mk('line',{class:'cmp-target-k',x1:180,y1:300,x2:180,y2:300},svg);
 const qLine=mk('line',{class:'cmp-vector-q',x1:180,y1:300,x2:180,y2:300},svg);
 const kLine=mk('line',{class:'cmp-vector-k',x1:180,y1:300,x2:180,y2:300},svg);
 const qArc=mk('path',{class:'cmp-arc-q',fill:'none'},svg);
 const kArc=mk('path',{class:'cmp-arc-k',fill:'none'},svg);
 const qTip=mk('circle',{class:'cmp-tip-q',r:4},svg);
 const kTip=mk('circle',{class:'cmp-tip-k',r:4},svg);
 const qName=txt(svg,'att-q',0,0,'Q');
 const kName=txt(svg,'att-k',0,0,'K');
 const baseName=txt(svg,'att-dim',0,0,'基准 [1,0]');
 const track=rect(svg,'cmp-track',40,454,280,6,3);
 const trackFill=rect(svg,'cmp-track-fill',40,454,0,6,3);
 const trackLabel=txt(svg,'att-dim',0,0,'旋转进度 t');
 const trackPct=txt(svg,'att-score',0,0,'0%');
 const panel=rect(svg,'att-panel',0,0,0,0);
 const pTitle=txt(svg,'att-strong',0,0,'位置条件（与播放进度分开）');
 const pQ=txt(svg,'att-q',0,0,'');
 const pK=txt(svg,'att-k',0,0,'');
 const pProg=txt(svg,'att-score',0,0,'');
 const pAng=txt(svg,'att-num',0,0,'');
 const pQr=txt(svg,'att-num',0,0,'');
 const pKr=txt(svg,'att-num',0,0,'');
 const pDotNow=txt(svg,'att-w',0,0,'');
 const pDotFinal=txt(svg,'att-num',0,0,'');
 const pNote=txt(svg,'att-note',0,0,'');
 const pInv=txt(svg,'att-note',0,0,'');
 const pCheck1=txt(svg,'att-dim',0,0,'');
 const pCheck2=txt(svg,'att-dim',0,0,'');
 const pFoot=txt(svg,'att-dim',0,0,'V 不参与 RoPE；这里只旋转 Q/K。');
 const foot=footer(svg,'θ=π/4；q=k=[1,0]。点积只由相对角决定；真实模型各维频率可不同。',false,560);
 let cond=0,lastP=0,lastM=false;
 function arcPath(cx,cy,r,angle){
  if(Math.abs(angle)<1e-6)return '';
  const x=cx+r*Math.cos(angle),y=cy-r*Math.sin(angle);
  const large=angle>Math.PI?1:0;
  return `M${cx+r} ${cy} A${r} ${r} 0 ${large} 0 ${x} ${y}`;
 }
 function draw(p,m){
  lastP=p;lastM=m;const W=m?360:720,HT=m?660:560;
  placeHeader(h,m,W);
  buttons.forEach((b,i)=>{
   const bw=m?332:214;
   placeBtn(b,m?14:18+i*228,m?82+i*34:86,bw,28,opts[i].label,i===cond);
  });
  const c=opts[cond];
  const r=M.rope(c.pq,c.pk);
  const t=phase(p,1,2.8);
  const cx=m?180:178,cy=m?292:300,R=m?90:104;
  const qAng=r.qAngle*t,kAng=r.kAngle*t;
  const qCur=M.rotate(r.q,qAng),kCur=M.rotate(r.k,kAng);
  const qDot=M.dot(qCur,kCur);
  const qx=cx+R*Math.cos(qAng),qy=cy-R*Math.sin(qAng);
  const kx=cx+R*Math.cos(kAng),ky=cy-R*Math.sin(kAng);
  const qtx=cx+R*Math.cos(r.qAngle),qty=cy-R*Math.sin(r.qAngle);
  const ktx=cx+R*Math.cos(r.kAngle),kty=cy-R*Math.sin(r.kAngle);
  circle.setAttribute('cx',cx);circle.setAttribute('cy',cy);circle.setAttribute('r',R);
  axisX.setAttribute('x1',cx-R-20);axisX.setAttribute('y1',cy);axisX.setAttribute('x2',cx+R+20);axisX.setAttribute('y2',cy);
  axisY.setAttribute('x1',cx);axisY.setAttribute('y1',cy-R-20);axisY.setAttribute('x2',cx);axisY.setAttribute('y2',cy+R+20);
  base.setAttribute('x1',cx);base.setAttribute('y1',cy);base.setAttribute('x2',cx+R);base.setAttribute('y2',cy);
  qTarget.setAttribute('x1',cx);qTarget.setAttribute('y1',cy);qTarget.setAttribute('x2',qtx);qTarget.setAttribute('y2',qty);
  kTarget.setAttribute('x1',cx);kTarget.setAttribute('y1',cy);kTarget.setAttribute('x2',ktx);kTarget.setAttribute('y2',kty);
  qLine.setAttribute('x1',cx);qLine.setAttribute('y1',cy);qLine.setAttribute('x2',qx);qLine.setAttribute('y2',qy);
  kLine.setAttribute('x1',cx);kLine.setAttribute('y1',cy);kLine.setAttribute('x2',kx);kLine.setAttribute('y2',ky);
  qArc.setAttribute('d',arcPath(cx,cy,R*0.44,qAng));kArc.setAttribute('d',arcPath(cx,cy,R*0.70,kAng));
  qTip.setAttribute('cx',qx);qTip.setAttribute('cy',qy);kTip.setAttribute('cx',kx);kTip.setAttribute('cy',ky);
  setText(qName,`Q ${(qAng*180/Math.PI).toFixed(1)}°`);place(qName,qx+8,qy-8);
  setText(kName,`K ${(kAng*180/Math.PI).toFixed(1)}°`);place(kName,kx+8,ky+16);
  place(baseName,cx,cy+R+14);
  // Left progress rail: rotation fraction only, never the position condition.
  const tx=m?40:40,ty=m?420:454,tw=m?280:280;
  put(track,tx,ty,tw,6);put(trackFill,tx,ty,tw*t,6);
  place(trackLabel,tx,ty-10);place(trackPct,tx+tw,ty-10);trackPct.setAttribute('text-anchor','end');
  setText(trackPct,pct(t));
  // Right/lower panel.
  const px=m?14:356,py=m?430:120,pw=m?332:350,ph=m?206:404;
  put(panel,px,py,pw,ph);
  place(pTitle,px+12,py+(m?22:26));
  if(m){
   setText(pQ,`pQ=${c.pq} → ${Math.round(r.qAngle*180/Math.PI)}° · pK=${c.pk} → ${Math.round(r.kAngle*180/Math.PI)}°`);
   place(pQ,px+12,py+44);setText(pK,'');
   place(pProg,px+12,py+66);setText(pProg,`t=${pct(t)} · 只表示旋转动画`);
   place(pAng,px+12,py+88);setText(pAng,`当前 Q ${(qAng*180/Math.PI).toFixed(1)}° / K ${(kAng*180/Math.PI).toFixed(1)}°`);
   place(pQr,px+12,py+110);setText(pQr,`q′ = ${fmt(qCur,4)}`);
   place(pKr,px+12,py+132);setText(pKr,`k′ = ${fmt(kCur,4)}`);
   place(pDotNow,px+12,py+158);setText(pDotNow,`当前 q′·k′ = ${num(qDot,8)}`);
   place(pDotFinal,px+12,py+182);setText(pDotFinal,`完整旋转后 = ${num(r.dot,8)}`);
   place(pNote,px+12,py+204);setText(pNote,'t 只控制比例；pQ/pK 才是位置。');
   show(pInv,false);show(pCheck1,false);show(pCheck2,false);
  }else{
   place(pQ,px+12,py+52);setText(pQ,`pQ=${c.pq} → Q 目标 ${Math.round(r.qAngle*180/Math.PI)}°`);
   place(pK,px+12,py+74);setText(pK,`pK=${c.pk} → K 目标 ${Math.round(r.kAngle*180/Math.PI)}°`);
   place(pProg,px+12,py+102);setText(pProg,`播放进度 t=${pct(t)}（只表示旋转动画）`);
   place(pAng,px+12,py+128);setText(pAng,`当前 Q ${(qAng*180/Math.PI).toFixed(1)}° / K ${(kAng*180/Math.PI).toFixed(1)}°`);
   place(pQr,px+12,py+156);setText(pQr,`当前 q′ = ${fmt(qCur,4)}`);
   place(pKr,px+12,py+180);setText(pKr,`当前 k′ = ${fmt(kCur,4)}`);
   place(pDotNow,px+12,py+210);setText(pDotNow,`当前 q′·k′ = ${num(qDot,8)}`);
   place(pDotFinal,px+12,py+236);setText(pDotFinal,`完整旋转后 q′·k′ = ${num(r.dot,8)}`);
   place(pNote,px+12,py+264);setText(pNote,'点积 = cos(当前夹角)；旋转保持长度 1。');
   place(pInv,px+12,py+292);setText(pInv,cond===1?'两者都多转 135°，相对角仍 45°：点积不变。':cond===2?'相对角变成 90°：完整旋转后的点积为 0。':'共同平移时两角同时增加，差值不变。');
   place(pCheck1,px+12,py+322);setText(pCheck1,'主例核对：1/0 与 4/3 → 0.70710678');
   place(pCheck2,px+12,py+344);setText(pCheck2,'　　　　　3/1 → 0（相对角 90°）');
   place(pFoot,px+12,py+376);
   show(pInv,true);show(pCheck1,true);show(pCheck2,true);
  }
  show(pK,!m);show(pFoot,!m);
  setText(foot,m?'θ=π/4；点积只由相对角决定；真实各维频率可不同。':'θ=π/4；q=k=[1,0]。点积只由相对角决定；真实模型各维频率可不同。');
  place(foot,m?14:18,HT-12);show(foot,true);
  svg.dataset.component='rope';
  svg.dataset.dot=num(r.dot,8);
  svg.dataset.currentDot=num(qDot,8);
  svg.dataset.pq=String(c.pq);svg.dataset.pk=String(c.pk);
  svg.dataset.relativeAngle=(r.relativeAngle*180/Math.PI).toFixed(3);
  svg.dataset.progress=t.toFixed(3);
 }
 return draw;
}

/* ------------------------------------------------------------ 02 RMSNorm */
function rmsBoard(svg,H){
 const h=header(svg,'RMSNorm：均方根缩放与 gamma','先平方求均方根；不减均值。gamma 逐维缩放。','输入 → 平方/均方 → 缩放 → gamma；条件可改尺度与 gamma。');
 const inOpts=[{label:'x=[3,4]',x:3},{label:'x=[6,8]',x:6}];
 const gOpts=[{label:'γ=[1,1]',g:[1,1]},{label:'γ=[1,2]',g:[1,2]}];
 const inButtons=inOpts.map((c,i)=>btn(svg,0,0,158,28,c.label,()=>{scale=i;pause(svg);draw(lastP,lastM);}));
 const gButtons=gOpts.map((c,i)=>btn(svg,0,0,158,28,c.label,()=>{gammaIdx=i;pause(svg);draw(lastP,lastM);}));
 const inPanel=rect(svg,'att-panel',0,0,0,0),midPanel=rect(svg,'att-panel',0,0,0,0),outPanel=rect(svg,'att-panel',0,0,0,0);
 const inTitle=txt(svg,'att-strong',0,0,'输入 x（有符号）');
 const scaleNote=txt(svg,'att-dim',0,0,'同一纵轴 0–8');
 const zeroLine=mk('line',{class:'cmp-axis',x1:0,y1:0,x2:0,y2:0},svg);
 const bars=[0,1].map(()=>mk('rect',{class:'cmp-bar-in'},svg));
 const barVals=[0,1].map(()=>txt(svg,'att-num',0,0,''));
 const barNames=[0,1].map((_,i)=>txt(svg,'att-dim',0,0,`x${i}`));
 const inNote=txt(svg,'att-dim',0,0,'不减均值；输入 ×2 时柱高也 ×2。');
 const midTitle=txt(svg,'att-strong',0,0,'计算 r');
 const midLines=[0,1,2].map(()=>txt(svg,'att-num',0,0,''));
 const rmsBig=txt(svg,'att-out',0,0,'');
 const meanNote=txt(svg,'att-note',0,0,'');
 const outTitle=txt(svg,'att-strong',0,0,'缩放与 gamma');
 const outRatio=txt(svg,'att-num',0,0,'');
 const dir=txt(svg,'att-note',0,0,'');
 const outScale=txt(svg,'att-dim',0,0,'同一纵轴 0–3');
 const outZero=mk('line',{class:'cmp-axis',x1:0,y1:0,x2:0,y2:0},svg);
 const outLabels=[0,1,2,3].map((_,i)=>txt(svg,'att-dim',0,0,['x/r 0','x/r 1','y 0','y 1'][i]));
 const outBars=[0,1,2,3].map(()=>mk('rect',{class:'cmp-bar-out'},svg));
 const outVals=[0,1,2,3].map(()=>txt(svg,'att-num',0,0,''));
 outLabels.forEach(t=>t.setAttribute('text-anchor','middle'));
 outVals.forEach(t=>t.setAttribute('text-anchor','middle'));
 const foot=footer(svg,'x 与 r 同倍放大时 x/r 基本不变；γ 改变比例，方向不保证不变。',false,560);
 let scale=0,gammaIdx=1,lastP=0,lastM=false;
 function draw(p,m){
  lastP=p;lastM=m;const W=m?360:720,HT=m?700:560;
  placeHeader(h,m,W);
  if(m){
   inButtons.forEach((b,i)=>placeBtn(b,14+i*168,82,160,28,inOpts[i].label,i===scale));
   gButtons.forEach((b,i)=>placeBtn(b,14+i*168,120,160,28,gOpts[i].label,i===gammaIdx));
  }else{
   inButtons.forEach((b,i)=>placeBtn(b,18+i*168,86,158,28,inOpts[i].label,i===scale));
   gButtons.forEach((b,i)=>placeBtn(b,18+2*168+i*168,86,158,28,gOpts[i].label,i===gammaIdx));
  }
  const x=[scale?6:3,scale?8:4],gamma=[1,gammaIdx?2:1];
  const r=M.rmsnorm(x,gamma);
  // Input panel: one fixed 0–8 scale, so [3,4] and [6,8] are visibly different.
  const ix=14,iy=m?160:132,iw=m?332:210,ih=m?176:390;
  put(inPanel,ix,iy,iw,ih);place(inTitle,ix+10,iy+22);
  if(!m){place(scaleNote,ix+10,iy+44);setText(inTitle,'输入 x（有符号）');}
  else setText(inTitle,'输入 x（纵轴 0–8）');
  show(scaleNote,!m);
  const iz=m?iy+122:iy+320,unit=(iz-(m?iy+42:iy+90))/8;
  zeroLine.setAttribute('x1',ix+12);zeroLine.setAttribute('y1',iz);zeroLine.setAttribute('x2',ix+iw-10);zeroLine.setAttribute('y2',iz);
  bars.forEach((b,i)=>{
   const bx=m?80+i*90:48+i*68,bh=x[i]*unit;
   put(b,bx,iz-bh,m?34:34,bh,3);
   place(barVals[i],bx+(m?17:17),iz-bh-8);setText(barVals[i],String(x[i]));
   place(barNames[i],bx+(m?17:17),iz+18);setText(barNames[i],`x${i}`);
  });
  place(inNote,ix+10,m?iy+163:iy+370);setText(inNote,'不减均值；输入 ×2 时柱高也 ×2。');
  // Middle panel: no duplicate big r after the formula line.
  const mx=m?14:232,my=m?344:132,mw=m?332:218,mh=m?140:390;
  put(midPanel,mx,my,mw,mh);place(midTitle,mx+10,my+22);
  const lines=[
   `平方 ${x[0]}² + ${x[1]}² = ${r.squares[0]} + ${r.squares[1]} = ${r.squares[0]+r.squares[1]}`,
   `均方 ${r.squares[0]+r.squares[1]} ÷ 2 = ${r.meanSquare}`,
   `√(${r.meanSquare} + 1e−6)`
  ];
  midLines.forEach((t,i)=>{place(t,mx+10,my+(m?46:52)+i*(m?22:28));setText(t,lines[i]);size(t,m?11.5:12.5);});
  place(rmsBig,mx+10,my+(m?118:142));setText(rmsBig,`r = ${r.rms.toFixed(4)}`);size(rmsBig,m?17:24);
  place(meanNote,mx+10,my+(m?134:196));setText(meanNote,`均值 ${r.mean}：只记录，不减`);size(meanNote,m?10.5:12);
  // Output panel: labels are y0/y1, not y2/y3.
  const ox=m?14:458,oy=m?492:132,ow=m?332:248,oh=m?162:390;
  put(outPanel,ox,oy,ow,oh);place(outTitle,ox+10,oy+22);
  place(outRatio,ox+10,oy+(m?46:50));setText(outRatio,`x/r = ${fmt(r.normalized,4)}`);size(outRatio,m?11:12.5);
  place(dir,ox+10,oy+(m?66:74));setText(dir,gammaIdx?'γ 改变比例：3:4 → 3:8':'γ=[1,1]：共同缩放，方向保持');size(dir,m?10.5:11.5);
  dir.setAttribute('class',gammaIdx?'att-mask':'att-w');
  const oz=m?oy+138:oy+348,ounit=(oz-(m?oy+62:oy+250))/3;
  if(!m)place(outScale,ox+10,oz-ounit*3-8);
  outZero.setAttribute('x1',ox+12);outZero.setAttribute('y1',oz);outZero.setAttribute('x2',ox+ow-10);outZero.setAttribute('y2',oz);
  const vals=[...r.normalized,...r.output];
  // All four bars use the same 0–3 axis and x/r 0, x/r 1, y 0, y 1 order.
  // Keep them inside the output panel: the old x origin belonged to the input panel.
  const outW=m?26:24,outStep=m?66:56,outStart=ox+(m?54:28);
  outBars.forEach((b,i)=>{
   const bx=outStart+i*outStep,bh=Math.abs(vals[i])*ounit;
   put(b,bx,oz-bh,outW,bh,3);
   b.setAttribute('class',i<2?'cmp-bar-norm':'cmp-bar-gamma');
   place(outVals[i],bx+outW/2,oz-bh-8);setText(outVals[i],vals[i].toFixed(2));size(outVals[i],m?10.5:11.5);
   place(outLabels[i],bx+outW/2,oz+18);setText(outLabels[i],i<2?`x/r ${i}`:`y ${i-2}`);size(outLabels[i],m?10:10.5);
  });
  outBars.forEach((b,i)=>fade(b,i<2?(p>=2?1:.4):(p>=3?1:.4)));
  outVals.forEach((b,i)=>fade(b,i<2?(p>=2?1:.4):(p>=3?1:.4)));
  fade(rmsBig,p>=1?1:.35);
  place(foot,m?14:18,HT-12);show(foot,true);
  svg.dataset.component='rmsnorm';
  svg.dataset.rms=r.rms.toFixed(6);
  svg.dataset.output=fmt(r.output,6);
  svg.dataset.gamma=fmt(gamma,0);
  svg.dataset.direction=String(r.directionPreserved);
  svg.dataset.input=fmt(x,0);
 }
 return draw;
}

/* ------------------------------------------------------------ 03 Residual */
function residualBoard(svg,H){
 const h=header(svg,'残差连接：原输入与子层结果相加','x 沿旁路保留；与 F(x) 逐元素相加，不是拼接。','同一维相加：结果形状与 x 相同；条件可把 F 设为 0。');
 const fOpts=[[.5,-1],[0,0],[1,1]];
 const labelsText=['F=[0.5,−1]','F=[0,0]','F=[1,1]'];
 const buttons=fOpts.map((fx,i)=>btn(svg,0,0,150,28,labelsText[i],()=>{fi=i;pause(svg);draw(lastP,lastM);}));
 const rowX=rect(svg,'att-panel',0,0,0,0),rowF=rect(svg,'att-panel',0,0,0,0),rowY=rect(svg,'att-panel',0,0,0,0);
 const labels=['原输入 x','子层 F(x)','结果 y=x+F(x)'].map(s=>txt(svg,'att-strong',0,0,s));
 const xVals=[0,1].map(()=>txt(svg,'att-out',0,0,'')),fVals=[0,1].map(()=>txt(svg,'att-num',0,0,'')),yVals=[0,1].map(()=>txt(svg,'att-out',0,0,''));
 const plus=[0,1].map(()=>{const g=mk('g',{class:'cmp-plus'},svg),c=mk('circle',{r:14},g),t=mk('text',{y:5,'text-anchor':'middle'},g);t.textContent='+';return{g,c,t};});
 const arrows=[0,1].map(()=>mk('line',{class:'cmp-line'},svg));
 const bypassArrow=marker(svg,'#7fe0c3');
 const bypass=mk('path',{class:'cmp-bypass',fill:'none'},svg);
 const bypassTips=[0,1].map(()=>mk('path',{class:'cmp-bypass',fill:'none','marker-end':`url(#${bypassArrow})`},svg));
 svg.insertBefore(bypass,plus[0].g);
 const panel=rect(svg,'att-panel',0,0,0,0);
 const pTitle=txt(svg,'att-strong',0,0,'逐元素，不是拼接');
 const pLine1=txt(svg,'att-num',0,0,''),pLine2=txt(svg,'att-num',0,0,''),pLine3=txt(svg,'att-num',0,0,'');
 const pFormula=[0,1].map(()=>txt(svg,'att-num',0,0,''));
 const pConcat=txt(svg,'att-mask',0,0,''),pNote=txt(svg,'att-note',0,0,'');
 const foot=footer(svg,'F=[0,0] 时 y=x；F 改变时只改对应维的贡献，旁路仍保留原输入。',false,560);
 let fi=0,lastP=0,lastM=false;
 function draw(p,m){
  lastP=p;lastM=m;const W=m?360:720,HT=m?660:560;
  placeHeader(h,m,W);
  buttons.forEach((b,i)=>placeBtn(b,m?14:18+i*160,m?82+i*36:86,m?332:150,28,labelsText[i],i===fi));
  const r=M.residual([1,2],fOpts[fi]);
  // Each dimension completes its own addition; dim 1 finishes inside the
  // "逐元素相加" frame so a half-computed state is visible during playback.
  const d0=p>=2,d1=p>=2.6;
  const left=m?14:24,lw=m?332:390;
  const rowXy=m?330:160,rowFy=m?410:250,rowYy=m?490:390,rh=m?56:64;
  put(rowX,left,rowXy,lw,rh);put(rowF,left,rowFy,lw,rh);put(rowY,left,rowYy,lw,rh);
  place(labels[0],left+12,rowXy+22);place(labels[1],left+12,rowFy+22);place(labels[2],left+12,rowYy+22);
  const chipX=[left+(m?150:210),left+(m?250:300)];
  xVals.forEach((t,i)=>{place(t,chipX[i],rowXy+46);setText(t,`[${r.x[i]}]`);});
  fVals.forEach((t,i)=>{place(t,chipX[i],rowFy+46);setText(t,`[${r.fx[i]}]`);});
  yVals.forEach((t,i)=>{
   const done=i===0?d0:d1;
   place(t,chipX[i],rowYy+46);setText(t,done?`[${r.sum[i]}]`:'?');fade(t,done?1:.5);
  });
  const plusY=m?474:360;
  plus.forEach((q,i)=>{
   const done=i===0?d0:d1;
   q.g.setAttribute('transform',`translate(${chipX[i]+18} ${plusY})`);
   show(q.g,done);q.c.setAttribute('class',done?'cmp-plus-on':'');
  });
  arrows.forEach((a,i)=>{
   const done=i===0?d0:d1;
   a.setAttribute('x1',chipX[i]+18);a.setAttribute('x2',chipX[i]+18);
   a.setAttribute('y1',rowFy+rh);a.setAttribute('y2',plusY-16);
   show(a,done);
  });
  // One x bypass rail feeds both same-dimension adders. It starts at dim 0 and
  // only extends to dim 1 when that dimension's addition becomes active.
  const routeX=left-10;
  const busEnd=(d1?chipX[1]:chipX[0])-30;
  bypass.setAttribute('d',`M${left} ${rowXy+32} H${routeX} V${plusY} H${busEnd}`);
  show(bypass,d0);fade(bypass,.75);
  bypassTips.forEach((t,i)=>{
   const done=i===0?d0:d1;
   t.setAttribute('d',`M${chipX[i]-30} ${plusY} H${chipX[i]+1}`);
   show(t,done);fade(t,.75);
  });
  const px=m?14:434,py=m?196:150,pw=m?332:272,ph=m?114:330;
  put(panel,px,py,pw,ph);place(pTitle,px+12,py+24);
  place(pLine1,px+12,py+52);
  if(m)setText(pLine1,d0?`${r.x[0]} + ${r.fx[0]} = ${r.sum[0]}`:`x = ${fmt(r.x,1)}，F = ${fmt(r.fx,1)}`);
  else setText(pLine1,`x = ${fmt(r.x,1)}，F(x) = ${fmt(r.fx,1)}`);
  place(pLine2,px+12,py+76);
  if(m)setText(pLine2,d1?`${r.x[1]} + ${r.fx[1]} = ${r.sum[1]}`:d0?`y0 = ${r.sum[0]}；y1 待相加`:'两维分别相加');
  else setText(pLine2,d1?`y = ${fmt(r.sum,1)}；形状仍为 ${r.shape} 维`:d0?`y0 = ${r.sum[0]} 已完成；y1 待相加`:'逐维相加后得到 y。');
  place(pLine3,px+12,py+104);
  if(m)setText(pLine3,d1?`y = [${r.sum.join(', ')}]`:'同一维相加；不是拼接。');
  else setText(pLine3,'同一维相加；不是拼接。');
  pFormula.forEach((t,i)=>{place(t,px+12,py+136+i*26);setText(t,`${r.x[i]} + ${r.fx[i]} = ${r.sum[i]}`);show(t,!m&&(i===0?d0:d1));});
  place(pConcat,px+12,py+196);setText(pConcat,p>=4?`拼接会得到 ${fmt(r.concatenated,1)}（4 维）：这不是残差。`:'如果拼接，会得到 4 维——但这里不这么做。');
  show(pConcat,!m);fade(pConcat,p>=4?1:.45);
  place(pNote,px+12,py+226);setText(pNote,'F=[0,0] 时 y=x；旁路把原输入直接送到加法。');show(pNote,!m);
  setText(foot,m?'F=[0,0] 时 y=x；旁路仍保留原输入。':'F=[0,0] 时 y=x；F 改变时只改对应维的贡献，旁路仍保留原输入。');
  place(foot,m?14:18,HT-12);show(foot,true);
  svg.dataset.component='residual';svg.dataset.sum=fmt(r.sum,6);svg.dataset.fx=fmt(r.fx,6);
 }
 return draw;
}

/* ------------------------------------------------------------- 04 SwiGLU */
function swigluBoard(svg,H){
 const h=header(svg,'SwiGLU：SiLU 门控与另一分支逐元素相乘','g=xWg，u=xWu；a=SiLU(g)，z=a⊙u，y=zWd。','门控可负，不是 0–1 开关，也不是专家路由。');
 const opts=[{label:'x=[1,2]',x:[1,2]},{label:'x=[2,2]',x:[2,2]}];
 const buttons=opts.map((c,i)=>btn(svg,0,0,160,28,c.label,()=>{xi=i;pause(svg);draw(lastP,lastM);}));
 const mid=marker(svg);
 const wires=mk('g',{class:'cmp-wires'},svg);
 const wireInG=wire(wires,mid,'M218 176 H238');
 const wireG=wire(wires,mid,'M280 192 V200');
 const wireA=wire(wires,mid,'M280 240 V258');
 const wireInU=wire(wires,mid,'M218 350 H238');
 const wireMultA=wire(wires,mid,'M352 282 C372 282 380 294 386 298');
 const wireMultU=wire(wires,mid,'M352 352 C372 352 380 320 386 308');
 const wireOut=wire(wires,mid,'M418 300 H454');
 const inPanel=rect(svg,'att-panel',0,0,0,0),resPanel=rect(svg,'att-panel',0,0,0,0);
 const inTitle=txt(svg,'att-strong',0,0,'输入与固定投影');
 const inLines=[0,1,2].map(()=>txt(svg,'att-num',0,0,''));
 const inNote=txt(svg,'att-dim',0,0,'g、u 都是 2 维');
 const gLabel=txt(svg,'att-dim',0,0,'门控分支 g=xWg');
 const gVal=txt(svg,'att-num',0,0,'');
 const siluBox=rect(svg,'cmp-op',0,0,0,0);
 const siluText=txt(svg,'cmp-op-t',0,0,'SiLU');
 const aLabel=txt(svg,'att-dim',0,0,'a=SiLU(g)');
 const aVal=txt(svg,'att-num',0,0,'');
 const uLabel=txt(svg,'att-dim',0,0,'值分支 u=xWu');
 const uVal=txt(svg,'att-num',0,0,'');
 const mult=mk('g',{class:'cmp-mult'},svg);
 const multC=mk('circle',{r:20},mult);const multT=mk('text',{y:5,'text-anchor':'middle'},mult);multT.textContent='⊙';
 const resTitle=txt(svg,'att-strong',0,0,'乘积与输出');
 const zLine=txt(svg,'att-num',0,0,'');
 const zVal=txt(svg,'att-out',0,0,'');
 const d0=txt(svg,'att-num',0,0,''),d1=txt(svg,'att-num',0,0,'');
 const negNote=txt(svg,'att-mask',0,0,'');
 const wd=txt(svg,'att-num',0,0,'');
 const yVal=txt(svg,'att-num',0,0,'');
 const yNote=txt(svg,'att-dim',0,0,'');
 const notRoute=txt(svg,'att-dim',0,0,'');
 const warn=txt(svg,'att-note',0,0,'');
 const foot=footer(svg,'教学简化 Wd=I；SiLU 可负，不是 0–1 门或专家路由。',false,560);
 let xi=0,lastP=0,lastM=false;
 function draw(p,m){
  lastP=p;lastM=m;const W=m?360:720,HT=m?700:560;
  placeHeader(h,m,W);
  buttons.forEach((b,i)=>placeBtn(b,m?14:18+i*180,m?82:86,m?160:160,28,opts[i].label,i===xi));
  const r=M.swiglu(opts[xi].x);
  // Input block.
  const ix=14,iy=m?124:132,iw=m?332:204,ih=m?110:176;
  put(inPanel,ix,iy,iw,ih);place(inTitle,ix+10,iy+22);
  const lines=[`x = ${fmt(r.x,1)}`,`Wg = [[1,−1],[0,0]]`,`Wu = [[0,1],[1,1]]`];
  inLines.forEach((t,i)=>{place(t,ix+10,iy+(m?46:54)+i*(m?22:28));setText(t,lines[i]);size(t,m?11.5:12);});
  place(inNote,ix+10,iy+(m?96:150));size(inNote,11);
  if(m){
   // Narrow flow: gate on the left, value branch on the right, product below.
   place(gLabel,24,258);place(gVal,24,280);setText(gVal,`g = ${fmt(r.g,2)}`);
   put(siluBox,24,296,86,34);place(siluText,67,318);
   place(aLabel,150,304);place(aVal,150,326);setText(aVal,`a = ${fmt(r.silu,4)}`);size(aVal,10.5);
   place(uLabel,238,258);place(uVal,238,280);setText(uVal,`u = ${fmt(r.u,2)}`);size(uVal,11);
   mult.setAttribute('transform','translate(206 390)');
   // Re-route the fixed wires for the narrow layout.
   wireInG.setAttribute('d','M100 234 V246 H64 V254');
   wireG.setAttribute('d','M64 288 V296');
   wireA.setAttribute('d','M110 313 H144');
   wireInU.setAttribute('d','M250 234 V246');
   wireMultA.setAttribute('d','M150 336 C150 360 180 372 198 380');
   wireMultU.setAttribute('d','M250 290 C250 340 226 364 212 374');
   wireOut.setAttribute('d','M206 410 V430');
   const rx=14,ry=430,rw=332,rh=240;
   put(resPanel,rx,ry,rw,rh);
   place(resTitle,rx+12,ry+24);
   place(zLine,rx+12,ry+54);setText(zLine,'z = a ⊙ u');
   place(zVal,rx+12,ry+82);setText(zVal,fmt(r.z,4));size(zVal,18);
   place(d0,rx+12,ry+116);setText(d0,`d0：${r.silu[0].toFixed(4)} × ${r.u[0]} = ${r.z[0].toFixed(4)}`);
   place(d1,rx+12,ry+140);setText(d1,`d1：${r.silu[1].toFixed(4)} × ${r.u[1]} = ${r.z[1].toFixed(4)}`);
   place(negNote,rx+12,ry+170);setText(negNote,'负门控 → 该维乘积为负');
   place(wd,rx+12,ry+200);setText(wd,'Wd=I → y = z');
   place(yVal,rx+12,ry+226);setText(yVal,fmt(r.y,4));size(yVal,13);
   show(yNote,false);show(notRoute,false);show(warn,false);
   fade(zVal,p>=3?1:.35);fade(zLine,p>=3?1:.35);fade(yVal,p>=4?1:.4);fade(wd,p>=4?1:.4);
  }else{
   place(gLabel,238,160);place(gVal,238,184);setText(gVal,`g = ${fmt(r.g,2)}`);
   put(siluBox,238,202,84,36);place(siluText,280,225);
   place(aLabel,238,264);place(aVal,238,288);setText(aVal,`a = ${fmt(r.silu,4)}`);
   place(uLabel,238,338);place(uVal,238,362);setText(uVal,`u = ${fmt(r.u,2)}`);
   mult.setAttribute('transform','translate(396 300)');
   wireInG.setAttribute('d','M218 176 H234');
   wireG.setAttribute('d','M280 192 V200');
   wireA.setAttribute('d','M280 240 V258');
   wireInU.setAttribute('d','M218 350 H234');
   wireMultA.setAttribute('d','M280 296 C310 320 370 305 386 298');
   wireMultU.setAttribute('d','M280 370 C320 370 370 320 386 308');
   wireOut.setAttribute('d','M418 300 H454');
   put(resPanel,454,132,252,390);place(resTitle,466,156);
   const rx=454,ry=132;
   place(zLine,rx+12,ry+54);setText(zLine,'z = a ⊙ u');
   place(zVal,rx+12,ry+86);setText(zVal,fmt(r.z,4));size(zVal,19);
   place(d0,rx+12,ry+124);setText(d0,`d0：${r.silu[0].toFixed(4)} × ${r.u[0]} = ${r.z[0].toFixed(4)}`);size(d0,11.5);
   place(d1,rx+12,ry+148);setText(d1,`d1：${r.silu[1].toFixed(4)} × ${r.u[1]} = ${r.z[1].toFixed(4)}`);size(d1,11.5);
   place(negNote,rx+12,ry+178);setText(negNote,'负门控 → 该维乘积为负');size(negNote,11.5);
   place(wd,rx+12,ry+212);setText(wd,'Wd = I → y = z');
   place(yVal,rx+12,ry+240);setText(yVal,fmt(r.y,4));size(yVal,14);
   place(yNote,rx+12,ry+270);setText(yNote,'真实 Wd 回投到模型宽度。');size(yNote,11);
   place(notRoute,rx+12,ry+298);setText(notRoute,'两路都来自同一个 x。');size(notRoute,11);
   place(warn,rx+12,ry+324);setText(warn,'没有专家选择；a 可负，不是 0–1 开关。');size(warn,10.5);
   show(yNote,true);show(notRoute,true);show(warn,true);
   fade(zVal,p>=3?1:.35);fade(zLine,p>=3?1:.35);fade(yVal,p>=4?1:.35);fade(yNote,p>=4?1:.35);
  }
  show(mult,p>=3);
  show(wires,p>=1);fade(wires,p>=3?1:.35);
  setText(foot,m?'Wd=I；SiLU 可负，不是 0–1 门或专家路由。':'教学简化 Wd=I；SiLU 可负，不是 0–1 门或专家路由。');
  place(foot,m?14:18,HT-12);show(foot,true);
  svg.dataset.component='swiglu';
  svg.dataset.z=fmt(r.z,6);
  svg.dataset.silu=fmt(r.silu,6);
  svg.dataset.negative=String(r.hasNegativeSilu);
 }
 return draw;
}

/* ------------------------------------------------------- 05 MHA/GQA/MQA */
function headsBoard(svg,H){
 const h=header(svg,'MHA/GQA/MQA：Q 头共享 K/V','4 个 Q 头、3 个位置、每头 2 维；改动 KV 组数。','每个 Q 仍独立查询；48/24/12 只是该层 K+V 标量数。');
 const kvOptions=[4,2,1];
 const buttons=kvOptions.map((kv,i)=>btn(svg,0,0,104,28,`${kv} 组 KV`,()=>{
  pause(svg);seekTo(svg,[2,3,4][i]);
 }));
 const qPanel=rect(svg,'att-panel',0,0,0,0),kvPanel=rect(svg,'att-panel',0,0,0,0),cPanel=rect(svg,'att-panel',0,0,0,0);
 const qTitle=txt(svg,'att-strong',0,0,'4 个 Q 头 · 各自 2 维');
 const qRows=[0,1,2,3].map(i=>{const g=mk('g',{class:'cmp-qrow'},svg);const r=mk('rect',{rx:6},g);const t=mk('text',{x:8,y:22},g);const s=mk('text',{x:8,y:38,class:'att-dim'},g);return {g,r,t,s};});
 const kvTitle=txt(svg,'att-strong',0,0,'K/V 组');
 const kvRows=[0,1,2,3].map(i=>{const g=mk('g',{class:'cmp-kvrow'},svg);const r=mk('rect',{rx:6},g);const t=mk('text',{x:8,y:20},g);const s=mk('text',{x:8,y:37,class:'att-dim'},g);return {g,r,t,s};});
 const lines=[0,1,2,3].map(()=>mk('path',{class:'cmp-map',fill:'none'},svg));
 const cTitle=txt(svg,'att-strong',0,0,'该层 K+V 标量');
 const formula=txt(svg,'att-num',0,0,'');
 const count=txt(svg,'att-out',0,0,'');
 const mode=txt(svg,'att-strong',0,0,'');
 const perQ=txt(svg,'att-note',0,0,'4 个 Q 仍各自查询；共享的是 K/V。');
 const outNote=txt(svg,'att-dim',0,0,'concat 后仍需 output projection。');
 const todo=txt(svg,'att-dim',0,0,'独立头权重与 concat 数值尚未展开。');
 const limit=txt(svg,'att-dim',0,0,'48/24/12 只是该层 K+V 标量。');
 const foot=footer(svg,'K+V 标量 = 2 × 3 × 2 × KV 组数；不是模型全部显存，也不表示改训练后头数无损。',false,560);
 let lastP=0,lastM=false;
 function draw(p,m){
  lastP=p;lastM=m;const W=m?360:720,HT=m?660:560;
  placeHeader(h,m,W);
  const groups=p<2.5?4:p<3.5?2:1;
  const r=M.kvHeads(groups);
  buttons.forEach((b,i)=>placeBtn(b,m?14+i*110:18+i*130,m?82:86,m?104:120,28,`${kvOptions[i]} 组 KV`,groups===kvOptions[i]));
  if(m){
   const qx=14,qy=124,qw=150,qh=280;
   put(qPanel,qx,qy,qw,qh);place(qTitle,qx+8,qy+20);size(qTitle,11);
   const rowY=[0,1,2,3].map(i=>qy+44+i*50);
   qRows.forEach((row,i)=>{
    row.g.setAttribute('transform',`translate(${qx+8} ${rowY[i]})`);
    row.r.setAttribute('width',qw-16);row.r.setAttribute('height',40);row.r.setAttribute('class','cmp-qbox');
    row.t.setAttribute('x',7);row.t.setAttribute('y',17);row.t.textContent=`Q${i} · 2 维`;
    row.s.setAttribute('x',7);row.s.setAttribute('y',33);row.s.setAttribute('text-anchor','start');row.s.textContent=`→ KV${r.mapping[i].group}`;
    show(row.s,true);show(row.g,true);
   });
   const kx=172,ky=124,kw=174,kh=280;
   put(kvPanel,kx,ky,kw,kh);place(kvTitle,kx+8,ky+20);size(kvTitle,11.5);
   const slot=(kh-58)/Math.max(groups,1),hgt=Math.max(42,slot-8);
   const groupY=[];
   for(let g=0;g<4;g++){
    const visible=g<groups,gy=ky+42+g*slot;
    kvRows[g].g.setAttribute('transform',`translate(${kx+8} ${gy})`);
    kvRows[g].r.setAttribute('width',kw-16);kvRows[g].r.setAttribute('height',hgt);
    kvRows[g].r.setAttribute('class',g<groups?'cmp-kvbox':'');
    kvRows[g].t.textContent=`KV${g}`;kvRows[g].t.setAttribute('x',8);kvRows[g].t.setAttribute('y',19);
    kvRows[g].s.setAttribute('x',8);kvRows[g].s.setAttribute('y',35);kvRows[g].s.textContent='K、V 各 3×2';
    show(kvRows[g].g,visible);
    if(visible)groupY[g]=gy+hgt/2;
   }
   lines.forEach(line=>show(line,false));
   const cx=14,cy=414,cw=332,ch=168;
   put(cPanel,cx,cy,cw,ch);place(cTitle,cx+10,cy+22);
   place(formula,cx+10,cy+50);setText(formula,`2 × 3 × 2 × ${groups} = ${r.scalars}`);size(formula,13);
   place(count,cx+10,cy+86);setText(count,String(r.scalars));size(count,24);
   place(mode,cx+10,cy+112);setText(mode,`${groups===4?'MHA':groups===2?'GQA':'MQA'} · ${groups} 组 KV`);size(mode,12);
   place(perQ,cx+10,cy+136);setText(perQ,'4 个 Q 仍各自查询；共享的是 K/V。');size(perQ,10.5);
   place(todo,cx+10,cy+158);setText(todo,'独立头权重与 concat 数值尚未展开。');size(todo,10);
   show(outNote,false);show(limit,false);
  }else{
   const qx=14,qy=150,qw=180,qh=330;
   put(qPanel,qx,qy,qw,qh);place(qTitle,qx+8,qy+20);
   const rowY=[0,1,2,3].map(i=>qy+50+i*58);
   qRows.forEach((row,i)=>{
    row.g.setAttribute('transform',`translate(${qx+8} ${rowY[i]})`);
    row.r.setAttribute('width',qw-16);row.r.setAttribute('height',44);row.r.setAttribute('class','cmp-qbox');
    row.t.setAttribute('x',7);row.t.setAttribute('y',20);row.t.textContent=`Q${i} · [·,·]`;
    row.s.setAttribute('x',qw-12);row.s.setAttribute('y',20);row.s.setAttribute('text-anchor','end');row.s.textContent='独立查询';
    show(row.s,true);show(row.g,true);
   });
   const kx=210,ky=150,kw=190,kh=330;
   put(kvPanel,kx,ky,kw,kh);place(kvTitle,kx+8,ky+20);
   const slot=(kh-72)/Math.max(groups,1),hgt=Math.max(38,slot-10);
   const groupY=[];
   for(let g=0;g<4;g++){
    const visible=g<groups,gy=ky+48+g*slot;
    kvRows[g].g.setAttribute('transform',`translate(${kx+8} ${gy})`);
    kvRows[g].r.setAttribute('width',kw-16);kvRows[g].r.setAttribute('height',hgt);
    kvRows[g].r.setAttribute('class',g<groups?'cmp-kvbox':'');
    kvRows[g].t.textContent=`KV${g}`;kvRows[g].t.setAttribute('x',8);kvRows[g].t.setAttribute('y',20);
    kvRows[g].s.setAttribute('x',8);kvRows[g].s.setAttribute('y',38);kvRows[g].s.textContent='K、V 各 3×2';
    show(kvRows[g].g,visible);
    if(visible)groupY[g]=gy+hgt/2;
   }
   lines.forEach((line,i)=>{
    const target=r.mapping[i].group;
    if(!(target in groupY)){show(line,false);return;}
    const sx=qx+qw-8,sy=rowY[i]+22,tx=kx+8,ty=groupY[target];
    line.setAttribute('d',`M${sx} ${sy} C${sx+18} ${sy} ${tx-18} ${ty} ${tx} ${ty}`);
    show(line,p>=1);fade(line,p>=2?1:.4);
   });
   const cx=418,cy=150,cw=288,ch=330;
   put(cPanel,cx,cy,cw,ch);place(cTitle,cx+8,cy+20);
   place(formula,cx+8,cy+48);setText(formula,`2 × 3 × 2 × ${groups} = ${r.scalars}`);size(formula,13.5);
   place(count,cx+8,cy+96);setText(count,String(r.scalars));size(count,30);
   place(mode,cx+8,cy+128);setText(mode,`${groups===4?'MHA':groups===2?'GQA':'MQA'} · ${groups} 组 KV`);size(mode,13);
   place(perQ,cx+8,cy+158);setText(perQ,'4 个 Q 仍各自查询；共享的是 K/V。');
   place(outNote,cx+8,cy+186);setText(outNote,'concat 后仍需 output projection。');
   place(todo,cx+8,cy+214);setText(todo,'独立头权重与 concat 数值尚未展开。');
   place(limit,cx+8,cy+242);setText(limit,'48/24/12 只是该层 K+V 标量数。');
   show(outNote,true);show(limit,true);
  }
  setText(foot,m?'K+V 标量 = 2×3×2×KV 组；不是整模型显存。':'K+V 标量 = 2 × 3 × 2 × KV 组数；不是模型全部显存，也不表示改训练后头数无损。');
  place(foot,m?14:18,HT-12);show(foot,true);
  svg.dataset.component='heads';
  svg.dataset.kv=String(groups);
  svg.dataset.scalars=String(r.scalars);
  svg.dataset.mode=groups===4?'MHA':groups===2?'GQA':'MQA';
 }
 return draw;
}

/* --------------------------------------------------------- 06 Transformer Block */
function blockBoard(svg,H){
 const h=header(svg,'Transformer Block：Pre-Norm 一层','x → Norm → Attention → +x → Norm → MLP → +x₁。','第二个旁路保留 x₁；点击标出的部件进入小演示。');
 const rows=[
  {id:'x',label:'x',sub:'当前位置表示'},
  {id:'norm1',label:'RMSNorm',sub:'先归一化',concept:'rmsnorm'},
  {id:'attn',label:'Attention',sub:'RoPE · 4/2/1 KV'},
  {id:'plus1',label:'+ x',sub:'残差 1（来自 x）',concept:'residual'},
  {id:'norm2',label:'RMSNorm',sub:'第二个子层',concept:'rmsnorm'},
  {id:'mlp',label:'SwiGLU MLP',sub:'逐位置变换',concept:'swiglu'},
  {id:'plus2',label:'+ x₁',sub:'残差 2（来自 x₁）',concept:'residual'},
  {id:'y',label:'y',sub:'进入下一层'}
 ];
 const boxes=rows.map(r=>{
  const g=mk('g',{class:'cmp-block'},svg);const rectEl=mk('rect',{rx:7},g);
  const t=mk('text',{x:12,y:25},g);t.textContent=r.label;const s=mk('text',{x:12,y:44},g);s.textContent=r.sub;
  if(r.concept)hit(g,`${r.label}：进入 ${r.concept} 小演示`,()=>svg.dispatchEvent(new CustomEvent('component-focus',{bubbles:true,detail:{concept:r.concept,step:0}})));
  return {g,r,rect:rectEl,t,s};
 });
 const arrows=rows.slice(1).map(()=>mk('line',{class:'cmp-line'},svg));
 const arc1=mk('path',{class:'cmp-bypass',fill:'none'},svg),arc2=mk('path',{class:'cmp-bypass',fill:'none'},svg);
 const arc1t=txt(svg,'att-dim',0,0,'x 旁路');
 const arc2t=txt(svg,'att-dim',0,0,'x₁ 旁路');
 const ropetag=txt(svg,'att-q',0,0,'RoPE');
 const headtag=txt(svg,'att-k',0,0,'头分组');
 const panel=rect(svg,'att-panel',0,0,0,0);
 const pTitle=txt(svg,'att-strong',0,0,'Pre-Norm 连接');
 const f1=txt(svg,'att-num',0,0,'x₁ = x + Attention(Norm(x))');
 const f2=txt(svg,'att-num',0,0,'y = x₁ + MLP(Norm(x₁))');
 const pLine1=txt(svg,'att-note',0,0,'第二个残差旁路来自 x₁，');
 const pLine2=txt(svg,'att-note',0,0,'不是最初的 x。');
 const pOne=txt(svg,'att-mask',0,0,'一层 ≠ 生成一个 token');
 const pLine3=txt(svg,'att-dim',0,0,'一次前向经过全部层后，');
 const pLine4=txt(svg,'att-dim',0,0,'输出头给出下一 token 的分数。');
 const pLine5=txt(svg,'att-dim',0,0,'层号 / 序列位置 / token 是不同对象。');
 const one=txt(svg,'att-mask',0,0,'一层 ≠ 生成一个 token');
 const next=txt(svg,'att-note',0,0,'');
 const foot=footer(svg,'Pre-Norm 顺序固定：每个子层前先归一化；第二个残差加的是 x₁。',false,560);
 hit(ropetag,'RoPE 小演示',()=>svg.dispatchEvent(new CustomEvent('component-focus',{bubbles:true,detail:{concept:'rope',step:0}})));
 hit(headtag,'头分组小演示',()=>svg.dispatchEvent(new CustomEvent('component-focus',{bubbles:true,detail:{concept:'heads',step:0}})));
 let lastP=0,lastM=false;
 function draw(p,m){
  lastP=p;lastM=m;const W=m?360:720,HT=m?660:560;
  placeHeader(h,m,W);
  const bx=m?16:70,bw=m?200:250,rh=42,gap=4,start=m?88:92;
  boxes.forEach((b,i)=>{
   const y=start+i*(rh+gap);
   b.g.setAttribute('transform',`translate(${bx} ${y})`);
   b.rect.setAttribute('width',bw);b.rect.setAttribute('height',rh);
   b.rect.setAttribute('class','cmp-block-box'+((b.r.id==='attn'&&p>=1)||(b.r.id==='norm1'&&p>=1)||(b.r.id==='plus1'&&p>=2)||(b.r.id==='norm2'&&p>=3)||(b.r.id==='mlp'&&p>=3)||(b.r.id==='plus2'&&p>=4)?' is-on':''));
   b.t.setAttribute('x',10);b.t.setAttribute('y',20);b.s.setAttribute('x',10);b.s.setAttribute('y',35);
   b.s.textContent=b.r.id==='attn'&&m?'RoPE · 头分组':b.r.sub;
  });
  arrows.forEach((a,i)=>{
   const y=start+i*(rh+gap)+rh;
   a.setAttribute('x1',bx+bw/2);a.setAttribute('x2',bx+bw/2);a.setAttribute('y1',y);a.setAttribute('y2',y+gap);
  });
  const rowCenter=i=>start+i*(rh+gap)+rh/2;
  const yX=bx+bw+10;
  arc1.setAttribute('d',`M${bx+bw} ${rowCenter(0)} H${yX+(m?38:60)} V${rowCenter(3)} H${bx+bw}`);
  arc2.setAttribute('d',`M${bx+bw} ${rowCenter(3)+rh/2} H${yX+(m?48:69)} V${rowCenter(6)} H${bx+bw}`);
  show(arc1,p>=2);show(arc2,p>=4);
  place(arc1t,bx+bw+6,rowCenter(0)-6);show(arc1t,!m);
  place(arc2t,bx+bw+6,rowCenter(3)+rh/2-4);show(arc2t,!m);
  place(ropetag,bx+bw+6,rowCenter(2)-4);show(ropetag,true);
  place(headtag,bx+bw+6,rowCenter(2)+12);show(headtag,true);
  if(m){
   place(f1,16,470);place(f2,16,492);size(f1,10.5);size(f2,10.5);
   place(pOne,16,526);one.style.display='none';
   place(next,16,554);setText(next,p>=5?'经过后续层，再由输出头得到下一 token 的分数。':'一层只是一次前向中的一段。');
   show(panel,false);show(pTitle,false);show(pLine1,false);show(pLine2,false);show(pLine3,false);show(pLine4,false);show(pLine5,false);
  }else{
   put(panel,414,92,292,300);place(pTitle,432,116);
   place(f1,432,146);place(f2,432,170);size(f1,12);size(f2,12);
   place(pLine1,432,208);place(pLine2,432,230);
   place(pOne,432,270);size(pOne,13);
   place(pLine3,432,300);place(pLine4,432,322);
   place(pLine5,432,352);size(pLine5,10.5);
   place(next,70,480);setText(next,p>=5?'经过后续层，再由输出头得到下一 token 的分数。':'一层只是一次前向中的一段。');
   show(panel,true);show(pTitle,true);show(pLine1,true);show(pLine2,true);show(pLine3,true);show(pLine4,true);show(pLine5,true);
   show(one,false);
  }
  setText(foot,m?'Pre-Norm：每个子层前先归一化；第二个残差加 x₁。':'Pre-Norm 顺序固定：每个子层前先归一化；第二个残差加的是 x₁。');
  place(foot,m?14:18,HT-12);show(foot,true);
  svg.dataset.component='block';svg.dataset.layerStep=String(Math.min(5,Math.floor(p+1e-6)));
 }
 return draw;
}

const frames=(...items)=>items.map(([title,caption])=>({title,caption}));
const choices=[
 {concept:'rope',name:'01 位置与 RoPE',title:'RoPE：位置决定 Q、K 的旋转角',
  intro:'同一对维度随位置旋转；只看旋转后的 Q·K。点条件比较共同平移与相对距离。',
  boundary:'θ=π/4 的教学单对维度；真实模型各维频率可不同。播放进度只表示旋转比例，不是位置。',
  data:{pQ:1,pK:0,theta:Math.PI/4},
  frames:frames(
   ['输入一对维度','q=[1,0]、k=[1,0]；位置来自序列，不是动画时间。下方按钮可切换位置条件。'],
   ['位置先变成角度','θ=π/4；Q、K 的目标角分别由 pQ、pK 决定，两个向量都旋转。'],
   ['旋转 Q 和 K','播放进度只控制旋转比例；当前角度、向量和点积随进度更新。'],
   ['用旋转后的向量做点积','当前 q′·k′ 由当前两向量计算；完整旋转后只由相对角决定。'],
   ['共同平移','两个位置同时增加相同量时，目标角同时增加，相对角不变；点上方「两者 +3」核对。'],
   ['改相对距离','只改一个位置会改变相对角；点上方「pQ=3, pK=1」核对 90° 与点积 0。']
  ),stepDurations:[1.2,1.6,2.0,1.8,1.6]},
 {concept:'rmsnorm',name:'02 RMSNorm',title:'RMSNorm：均方根缩放与 gamma',
  intro:'先平方求均方根，再按共同标量缩放；gamma 逐维缩放。改输入尺度与 gamma 核对。',
  boundary:'x=[3,4]、eps=1e-6 的教学例；不减均值。gamma 后方向不保证不变，也不把柱子画成等高。',
  data:{x:[3,4],gamma:[1,2],eps:1e-6},
  frames:frames(
   ['输入两个有符号分量','先看平方，不先减均值；下方可切换输入尺度。'],
   ['先算均方根','平方和、均方和 r 都显示在中栏；r 是整条向量的共同标量。'],
   ['按共同标量缩放','未乘 gamma 时 x/r 的比例与 x 相同。'],
   ['gamma 逐维缩放','gamma 逐维缩放，比例和方向都可能改变；看 y0/y1 两根柱。'],
   ['改输入尺度','输入 ×2 时 r 也约 ×2，x/r 基本不变；固定纵轴下柱高会变化。'],
   ['常见误解','RMSNorm 没有减均值；gamma 不保证方向总不变，也不能把柱子画成等高。']
  ),stepDurations:[1.1,1.8,1.5,1.8,1.4]},
 {concept:'residual',name:'03 残差连接',title:'残差连接：逐元素相加，不是拼接',
  intro:'原输入沿旁路保留，子层结果与它按同一维相加；改 F 看结果如何变化。',
  boundary:'x=[1,2]、F=[0.5,−1] 的教学例；实际结构可能有不同归一化顺序，本课固定 Pre-Norm。',
  data:{x:[1,2],fx:[.5,-1]},
  frames:frames(
   ['原输入先保留','x=[1,2] 沿旁路保留；它不是备份，后面要参与相加。'],
   ['子层给出 F(x)','F(x) 的形状与 x 相同，都是 2 维；下方可切换 F。'],
   ['逐元素相加','同一维相加，两个加法互不混合。'],
   ['结果形状不变','y 仍是 2 维；不是把 x 与 F(x) 拼成 4 维。'],
   ['改子层贡献','F=[0,0] 时 y=x；F=[1,1] 时每一维各加 1。'],
   ['旁路的作用','子层只需学习增量；原表示仍可直接进入下一段计算。']
  ),stepDurations:[1.2,1.5,1.7,1.3,1.5]},
 {concept:'swiglu',name:'04 MLP / SwiGLU',title:'SwiGLU：门控与另一分支逐元素乘',
  intro:'同一输入经两个投影得到 g、u；SiLU(g) 与 u 逐元素相乘，再做输出投影。',
  boundary:'Wd=I 的教学简化；数值为手算例。SiLU 可负，不是 0–1 开关，也不是专家路由。',
  data:{x:[1,2]},
  frames:frames(
   ['同一输入分两路投影','g=xWg，u=xWu；两路都来自同一个 x。'],
   ['门控先过 SiLU','a=SiLU(g)；SiLU 可以取负，不是 0–1 开关。'],
   ['另一路保留数值','u 没有被选中或丢弃；它要和 a 逐元素相乘。'],
   ['逐元素相乘','z=a⊙u；负门控会把对应维的乘积推向负值。'],
   ['输出投影','本例 Wd=I，所以 y=z；真实 FFN 还要把扩展维回投到模型宽度。'],
   ['改输入并排除误解','改输入后 g、a、z 同步变化；这不是在多个专家中路由。']
  ),stepDurations:[1.3,1.8,1.5,1.7,1.5]},
 {concept:'heads',name:'05 MHA / GQA / MQA',title:'MHA / GQA / MQA：4 个 Q 头怎样共享 K/V',
  intro:'固定 4 个 Q 头、3 个历史位置、每头 2 维；切换 4/2/1 组 KV，看映射与标量数。',
  boundary:'48/24/12 是该层 K+V 标量数，不是模型全部显存；每 Q 仍独立，改训练后头数不声称无损。',
  data:{kvHeads:4,qHeads:4,positions:3,dim:2},
  frames:frames(
   ['先固定 4 个 Q 头','4 个 Q 头各自 2 维，每头独立查询；共享 K/V 不等于合并 Q。'],
   ['历史 K/V 的规模','3 个历史位置，每个 KV 头每位置 2 维；K 和 V 各一份。'],
   ['4 组 KV：MHA','Q0→KV0，Q1→KV1，Q2→KV2，Q3→KV3；标量 2×3×2×4=48。'],
   ['2 组 KV：GQA','Q0、Q1→KV0；Q2、Q3→KV1；同样 3 位置，标量 24。'],
   ['1 组 KV：MQA','4 个 Q 都读 KV0；标量 12，但每个 Q 仍独立计算。'],
   ['输出与边界','各头输出 concat 后还要 output projection；48/24/12 只是该层 KV 数。']
  ),stepDurations:[1.2,1.6,1.6,1.5,1.5]},
 {concept:'block',name:'06 Transformer Block',title:'Transformer Block：Pre-Norm 完整层',
  intro:'把已说明的归一化、Attention、残差与 SwiGLU MLP 接回一层；点击部件可进入小演示。',
  boundary:'只讲一层的前向连接；一层不等于生成一个 token。层号、位置和 token 是不同对象。',
  data:{},
  frames:frames(
   ['一层从一个表示开始','x 是当前位置的表示；层号 L 与位置是两件事。'],
   ['先归一化，再做 Attention','Pre-Norm：RMSNorm(x) 进入 Attention；Attention 内含 RoPE 与头分组。'],
   ['加上原输入','x + Attention(Norm(x))；旁路保留原表示，Attention 只是增量。'],
   ['再做一次 Norm 和 MLP','第二个子层是 SwiGLU MLP；它逐位置变换，不混合历史位置。'],
   ['第二个残差','y = x₁ + MLP(Norm(x₁))；这一层的输出接着进入下一层。'],
   ['一层不等于一个 token','一次前向经过全部层，再由输出头得到下一 token 的分数；生成还要选择。']
  ),stepDurations:[1.1,1.8,1.7,1.5,1.5]}
];

globalThis.TeachingLessons=globalThis.TeachingLessons||{};
globalThis.TeachingLessons['f-components']=choices;

V['f-components']=(svg,viewport,s)=>{
 const film=svg.closest('.mechanism-film');
 if(film)film.classList.toggle('is-components',true);
 const compact=s.concept==='rmsnorm'||s.concept==='swiglu'?700:660;
 return board(svg,viewport,'components-'+s.concept,m=>m?compact:560,H=>{
  if(s.concept==='rope')return ropeBoard(svg,H);
  if(s.concept==='rmsnorm')return rmsBoard(svg,H);
  if(s.concept==='residual')return residualBoard(svg,H);
  if(s.concept==='swiglu')return swigluBoard(svg,H);
  if(s.concept==='heads')return headsBoard(svg,H);
  return blockBoard(svg,H);
 });
};
})();
