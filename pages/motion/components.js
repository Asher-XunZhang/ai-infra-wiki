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
const sgn=(v,d=4)=>{if(!Number.isFinite(v))return '—';if(Math.abs(v)<5e-9)return '0';return (v>0?'+':'−')+num(Math.abs(v),d);};

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
 if(g.cmpLabel.textContent!==label){g.cmpLabel.textContent=label;g.cmpLabelText=label;}
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
/* A changed in-board condition starts a new teaching run: the shared player
 * pauses at 0 so the option is visible before the user presses play. Clicking
 * the active option keeps the current position; detail toggles do not come here. */
function changeCondition(svg,current,next,apply){
 if(current===next){pause(svg);return;}
 apply(next);
 seekTo(svg,0);
}
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
/* One fixed coordinate plane carries the whole lesson: a sequence-position
 * track says where Q/K come from, the unit circle shows the actual rotation of
 * one Q/K feature pair, and three rows compare the same Q with K1/K2/K3 through
 * Δθ and a signed dot product. A whole +3 is taught as a common rotation from
 * the base condition, so the track and the vectors move together while the
 * three rows stay constant. */
function ropeBoard(svg,H){
 const h=header(svg,'RoPE：同样的内容，为什么比较会不同？','', '旋转 Q/K 的一对特征分量，不是 token 实体');
 const conds=[
  {id:'base',label:'起点',q:2,ks:[0,1,2]},
  {id:'q1',label:'只移 Q +1',q:3,ks:[0,1,2]},
  {id:'shift3',label:'整段 +3',q:5,ks:[3,4,5]}
 ];
 const FROM_Q=2,FROM_KS=[0,1,2];
 const K_COLORS=['#c9b8ff','#a98cf5','#e3d4ff'];
 const Q_COLOR='#62d8e9';
 let condIdx=0,detail=false;
 const rad=d=>d*Math.PI/180;
 const norm=d=>((d%360)+360)%360;
 const angGap=(a,b)=>Math.abs(((a-b+540)%360)-180);
 const signed=(d)=>((d+540)%360)-180;
 const round=n=>Math.round(n);
 const pairAngle=(name,cur,target)=>`${name} ${round(cur)}°→${round(target)}°`;
 const pairValue=(cur,target)=>`${sgn(cur,3)}→${sgn(target,3)}`;
 function verdict(dot){
  if(dot>0.995)return '同向';
  if(dot<-0.995)return '反向';
  if(Math.abs(dot)<0.005)return '垂直';
  if(dot<0)return '负比较';
  return '有夹角';
 }
 function headerSub(c,m){
  if(c.id==='q1')return m?'只移 Q：p2→p3；K 不动，Q 多转 45°。':'只移 Q：Q 从 p2 到 p3，K 停在 p0–p2；只有 Q 多转 45°。';
  if(c.id==='shift3')return m?'整段 +3：轨道右移；四个向量共同 +135°。':'整段 +3：轨道右移 3 格，四个向量共同多转 135°；相对角与点积不变。';
  return m?'起点：Q p2，K p0–p2；按位置转开。':'起点：Q 在 p2，三个 K 在 p0/p1/p2；同一份内容开始按各自位置旋转。';
 }
 const buttons=conds.map((c,i)=>btn(svg,0,0,126,26,c.label,()=>{
  changeCondition(svg,condIdx,i,v=>{condIdx=v;});
 }));
 const detailBtn=btn(svg,0,0,126,26,'展开计算',()=>{
  detail=!detail;pause(svg);draw(lastP,lastM);
 });
 const controlHint=txt(svg,'att-dim',0,0,'换条件 → 回到 0 并暂停');

 // Sequence-position track: cards move horizontally; they never rotate.
 const trackPanel=rect(svg,'att-panel',0,0,0,0);
 const trackTitle=txt(svg,'att-strong',0,0,'① 序列位置：Q 与同一段内容在哪一格');
 const trackSub=txt(svg,'att-dim',0,0,'');
 const qSlot=rect(svg,'cmp-qslot',0,0,0,0);
 const targetCards=[0,1,2].map(()=>{
  const g=mk('g',{class:'cmp-target-card'},svg);
  const bg=rect(g,'cmp-target-card-bg',0,0,0,0,8);
  const label=txt(g,'cmp-target-label',0,14,'');
  label.setAttribute('text-anchor','middle');
  return {g,bg,label};
 });
 const cards=[0,1,2].map(()=>{
  const g=mk('g',{class:'cmp-kcard'},svg);
  const bg=rect(g,'cmp-kcard-bg',0,0,0,0,8);
  const id=txt(g,'cmp-kcard-id',0,0,'K1');
  const pos=txt(g,'cmp-kcard-pos',0,0,'p0');
  const content=txt(g,'cmp-kcard-content',0,0,'内容 A');
  const selfDot=mk('circle',{class:'cmp-self-dot',r:3.2},g);
  return {g,bg,id,pos,content,selfDot};
 });
 const slotTicks=[0,1,2,3,4,5].map(()=>mk('circle',{class:'cmp-slot-tick',r:2.2},svg));
 const slotLabels=[0,1,2,3,4,5].map(i=>{
  const t=txt(svg,'cmp-slot-label',0,0,'p'+i);
  t.setAttribute('text-anchor','middle');
  return t;
 });
 const qConnector=mk('line',{class:'cmp-qconnector'},svg);
 const qPin=mk('g',{class:'cmp-qpin'},svg);
 const qPinBg=rect(qPin,'cmp-qpin-bg',0,0,48,16,8);
 const qPinText=txt(qPin,'cmp-qpin-text',24,11.4,'Q p2');
 qPinText.setAttribute('text-anchor','middle');

 // One fixed coordinate plane; every vector endpoint is derived by rotation.
 const circlePanel=rect(svg,'att-panel',0,0,0,0);
 const circleTitle=txt(svg,'att-strong',0,0,'② 向量平面：Q/K 真的在转');
 const circleSub=txt(svg,'att-dim',0,0,'青=Q，紫=K；虚线=目标角/起点残影');
 const circleBg=mk('circle',{class:'cmp-circle',r:0},svg);
 const axisX=mk('line',{class:'cmp-axis'},svg);
 const axisY=mk('line',{class:'cmp-axis'},svg);
 const axisXLabel=txt(svg,'cmp-axis-label',0,0,'+x');
 const axisYLabel=txt(svg,'cmp-axis-label',0,0,'+y');
 const axisTicks=[0,1,2,3,4,5,6,7].map(()=>mk('line',{class:'cmp-axis-tick'},svg));
 const targetRays=[0,1,2,3].map(i=>mk('line',{class:'cmp-target-ray '+(i===0?'is-q':'is-k'+i)},svg));
 const ghostRays=[0,1,2,3].map(i=>mk('line',{class:'cmp-ghost-ray '+(i===0?'is-q':'is-k'+i)},svg));
 const kLines=[0,1,2].map(i=>mk('line',{class:'cmp-vector-line cmp-vector-k cmp-vector-k'+(i+1),'data-key':'K'+(i+1)},svg));
 const qLine=mk('line',{class:'cmp-vector-line cmp-vector-q','data-key':'Q'},svg);
 const kTips=[0,1,2].map(i=>mk('circle',{class:'cmp-tip cmp-tip-k'+(i+1),r:i===2?8:3.2},svg));
 const qTip=mk('circle',{class:'cmp-tip cmp-tip-q',r:4.4},svg);
 const qLabelLeader=mk('line',{class:'cmp-label-leader is-q'},svg);
 const kLabelLeaders=[0,1,2].map(i=>mk('line',{class:'cmp-label-leader is-k'+(i+1)},svg));
 const qLabel=txt(svg,'cmp-vector-label is-q',0,0,'Q');
 const kLabels=[0,1,2].map(i=>txt(svg,'cmp-vector-label is-k'+(i+1),0,0,'K'+(i+1)));
 const qk3Label=txt(svg,'cmp-vector-label is-joint',0,0,'Q 与 K3 同向');
 const allLabel=txt(svg,'cmp-vector-label is-joint',0,0,'Q 与 K1–K3 同向');
 const projLine=mk('line',{class:'cmp-projection','data-key':'K2'},svg);
 const projDrop=mk('line',{class:'cmp-projection-drop'},svg);
 const projFoot=mk('circle',{class:'cmp-projection-foot',r:3.2},svg);
 const projLabel=txt(svg,'cmp-projection-label',0,0,'');
 const arcMarker=marker(svg,'#f2b45c');
 const axisMarkerId=marker(svg,'#6f8a9b');
 const commonArc=mk('path',{class:'cmp-arc',fill:'none','marker-end':`url(#${arcMarker})`},svg);
 const arcLabel=txt(svg,'cmp-arc-label',0,0,'');
 const angleHeader=txt(svg,'cmp-angle-readout',0,0,'当前角 → 目标角');
 const qAngleText=txt(svg,'cmp-angle-readout is-q',0,0,'');
 const kAngleTexts=[0,1,2].map(i=>txt(svg,'cmp-angle-readout is-k'+(i+1),0,0,''));
 const legendIds=[0,1,2,3].map(i=>txt(svg,'cmp-legend-id is-'+(i===0?'q':'k'+i),0,0,['Q','K1','K2','K3'][i]));
 const projNote=txt(svg,'cmp-projection-label',0,0,'');

 // Three rows share the same current vectors and the same signed scale.
 const rowsPanel=rect(svg,'att-panel',0,0,0,0);
 const rowsTitle=txt(svg,'att-strong',0,0,'③ 三行比较：Q 分别对 K1/K2/K3');
 const rowsSub=txt(svg,'att-dim',0,0,'用当前 q′、k′ 算 Δθ 与点积；零居中，负值向左');
 const rowsGroup=mk('g',{class:'cmp-rows'},svg);
 const rows=[0,1,2].map(()=>{
  const bg=rect(rowsGroup,'cmp-row',0,0,0,0,8);
  const label=txt(rowsGroup,'cmp-row-label',0,0,'K1 · p0');
  const delta=txt(rowsGroup,'cmp-row-delta',0,0,'Δθ 90°');
  const value=txt(rowsGroup,'cmp-row-value',0,0,'q′·k′ 0.00');
  const barBg=rect(rowsGroup,'cmp-compare-track',0,0,0,0,4);
  const barFill=rect(rowsGroup,'cmp-compare-fill',0,0,0,0,4);
  const zeroMark=mk('line',{class:'cmp-compare-zero'},rowsGroup);
  const zero=txt(rowsGroup,'cmp-scale-tick',0,0,'0');
  const negTick=txt(rowsGroup,'cmp-scale-tick',0,0,'−1');
  const posTick=txt(rowsGroup,'cmp-scale-tick',0,0,'+1');
  return {bg,label,delta,value,barBg,barFill,zeroMark,zero,negTick,posTick};
 });

 const note1=txt(svg,'att-note',0,0,'');
 const note2=txt(svg,'att-note',0,0,'');
 const boundary=txt(svg,'att-dim',0,0,'');

 // Optional derivation layer. On a phone it replaces the three rows while open;
 // it never changes the condition or the clock.
 const detailBg=rect(svg,'att-panel',0,0,0,0);
 const detailTitle=txt(svg,'att-strong',0,0,'推导：先旋转，再点积（当前→目标）');
 const detailSub1=txt(svg,'cmp-detail-sub',0,0,'');
 const detailSub2=txt(svg,'cmp-detail-sub',0,0,'');
 const detailSub3=txt(svg,'cmp-detail-sub',0,0,'');
 const detailCols=[0,1,2].map(()=>txt(svg,'cmp-detail-line',0,0,''));
 const detailSubCols=[0,1,2].map(()=>txt(svg,'cmp-detail-line cmp-detail-subline',0,0,''));
 const detailFoot=txt(svg,'cmp-detail-sub',0,0,'比较值是 softmax 前的 Q·K，不是概率；不推出“越近越高”。');
 show(detailBg,false);show(detailTitle,false);show(detailSub1,false);show(detailSub2,false);show(detailSub3,false);
 detailCols.forEach(e=>show(e,false));detailSubCols.forEach(e=>show(e,false));show(detailFoot,false);

 const foot=footer(svg,'',false,560);
 // A compact key for the two visual languages that carry the comparison:
 // dashed target rays are the reference, bars are the signed dot value.
 const legend=txt(svg,'cmp-legend',0,0,'虚线=方向参照 · 条形=点积');
 let lastP=0,lastM=false;

 function pointAt(cx,cy,r,deg){return{x:cx+Math.cos(rad(deg))*r,y:cy-Math.sin(rad(deg))*r};}
 function setLine(e,a,b){e.setAttribute('x1',a.x.toFixed(2));e.setAttribute('y1',a.y.toFixed(2));e.setAttribute('x2',b.x.toFixed(2));e.setAttribute('y2',b.y.toFixed(2));}
 /* Identity chips are short on purpose: the fixed key below the circle carries
  * the current→target numbers. Chips still need to stay apart when Q, K3, K2
  * occupy a narrow fan just after rotation starts, so resolve their boxes with
  * a small spring/repulsion pass and keep a leader back to the endpoint. */
 function layoutChips(items,cx,cy,r,bounds){
  const gap=15,placed=items.map((it,i)=>{
   const base=pointAt(cx,cy,r,it.angle);
   return {...it,ox:base.x,oy:base.y,x:base.x,y:base.y,index:i};
  });
  for(let pass=0;pass<24;pass++){
   for(const c of placed){c.x+=(c.ox-c.x)*.12;c.y+=(c.oy-c.y)*.12;}
   for(let i=0;i<placed.length;i++)for(let j=i+1;j<placed.length;j++){
    const a=placed[i],b=placed[j];
    let dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy);
    if(d>=gap)continue;
    if(d<.01){dx=0;dy=j-i;d=Math.hypot(dx,dy);}
    const need=(gap-d)/2,ux=dx/d,uy=dy/d;
    a.x-=ux*need;a.y-=uy*need;b.x+=ux*need;b.y+=uy*need;
   }
   for(const c of placed){c.x=Math.max(bounds.x0,Math.min(bounds.x1,c.x));c.y=Math.max(bounds.y0,Math.min(bounds.y1,c.y));}
  }
  // The last pass enforces spacing after the spring, so chips never land on
  // top of one another even when the desired points are identical.
  for(let i=0;i<placed.length;i++)for(let j=i+1;j<placed.length;j++){
   const a=placed[i],b=placed[j];
   let dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy);
   if(d>=gap)continue;
   if(d<.01){dx=0;dy=j-i;d=Math.hypot(dx,dy);}
   const need=(gap-d)/2,ux=dx/d,uy=dy/d;
   a.x-=ux*need;a.y-=uy*need;b.x+=ux*need;b.y+=uy*need;
  }
  return placed;
 }
 function placeChip(e,text,chip,visible){
  if(!visible){show(e,false);e.setAttribute('data-visible','false');return;}
  e.setAttribute('x',chip.x.toFixed(1));e.setAttribute('y',chip.y.toFixed(1));
  e.setAttribute('text-anchor','middle');
  e.setAttribute('data-identity',chip.id);e.setAttribute('data-label-x',chip.x.toFixed(1));e.setAttribute('data-label-y',chip.y.toFixed(1));
  e.setAttribute('data-visible','true');
  setText(e,text);show(e,true);fade(e,1);
 }
 function placeLeader(line,tip,chip,visible,colorClass){
  if(!visible||Math.hypot(chip.x-tip.x,chip.y-tip.y)<4){show(line,false);return;}
  const dx=chip.x-tip.x,dy=chip.y-tip.y,d=Math.hypot(dx,dy),ux=dx/d,uy=dy/d;
  setLine(line,{x:tip.x+ux*5,y:tip.y+uy*5},{x:chip.x-ux*5,y:chip.y-3-uy*3});
  show(line,true);
 }
 function arcData(cx,cy,r,from,to){
  const a=pointAt(cx,cy,r,from),b=pointAt(cx,cy,r,to),large=Math.abs(to-from)>180?1:0;
  return {d:`M${a.x.toFixed(2)} ${a.y.toFixed(2)} A${r.toFixed(2)} ${r.toFixed(2)} 0 ${large} 0 ${b.x.toFixed(2)} ${b.y.toFixed(2)}`,mid:(from+to)/2};
 }

 function draw(p,m){
  lastP=p;lastM=m;const W=m?360:720,HT=m?720:560;
  placeHeader(h,m,W);
  if(m)put(h.bg,0,0,W,HT);
  if(m){h.t1.style.fontSize='14.5px';h.t2.style.fontSize='9.2px';h.t3.style.fontSize='8.4px';}
  buttons.forEach((b,i)=>{
   if(m)placeBtn(b,10+(i%2)*172,84+Math.floor(i/2)*34,162,28,conds[i].label,i===condIdx);
   else placeBtn(b,14+i*136,84,126,26,conds[i].label,i===condIdx);
  });
  if(m)placeBtn(detailBtn,10,152,334,28,detail?'收起计算':'展开计算',detail);
  else placeBtn(detailBtn,422,84,126,26,detail?'收起计算':'展开计算',detail);
  detailBtn.setAttribute('aria-label',detail?'收起计算':'展开计算');
  detailBtn.setAttribute('aria-expanded',detail?'true':'false');
  fade(controlHint,m?0:1);show(controlHint,!m);place(controlHint,558,101);size(controlHint,8.4);
  place(legend,m?152:706,m?438:63);size(legend,m?6.8:8.2);
  legend.setAttribute('text-anchor',m?'start':'end');show(legend,true);

  const c=conds[condIdx];
  setText(h.t2,headerSub(c,m));
  const rot=phase(p,0.75,2.65);
  const fromQAngle=c.id==='base'?0:FROM_Q*45;
  const fromKAngles=c.id==='base'?[0,0,0]:FROM_KS.map(v=>v*45);
  const qPos=FROM_Q+(c.q-FROM_Q)*rot;
  const kPos=FROM_KS.map((v,i)=>v+(c.ks[i]-v)*rot);
  const qDeg=norm(fromQAngle+(c.q*45-fromQAngle)*rot);
  const kDegs=fromKAngles.map((v,i)=>norm(v+(c.ks[i]*45-v)*rot));
  const qVec=M.rotate([1,0],rad(qDeg));
  const kVecs=kDegs.map(v=>M.rotate([1,0],rad(v)));
  const qDotVals=kVecs.map(v=>M.dot(qVec,v));
  const targetQ=c.q*45;
  const targetKs=c.ks.map(v=>v*45);
  const targetDots=kVecs.map((_,i)=>M.dot(M.rotate([1,0],rad(targetQ)),M.rotate([1,0],rad(targetKs[i]))));
  const targetRels=targetKs.map(v=>signed(targetQ-v));
  const curRels=kDegs.map(v=>signed(qDeg-v));

  // ------- track: same cards persist; only x changes. Q is a pin above the rail.
  const trackX=m?8:14,trackY=m?182:112,trackW=m?344:692,trackH=m?104:108;
  put(trackPanel,trackX,trackY,trackW,trackH);
  place(trackTitle,trackX+12,trackY+18);size(trackTitle,m?10.5:13.5);
  setText(trackTitle,m?'① 序列位置：Q/K 在哪一格':'① 序列位置：Q 与同一段内容在哪一格');
  place(trackSub,trackX+12,trackY+34);size(trackSub,m?8.4:9.3);
  setText(trackSub,c.id==='base'?(m?'Q p2；K p0–p2。旋转在下方。':'Q p2；K p0 / p1 / p2。这里只显示位置，旋转在下方。'):
   c.id==='q1'?(rot>=.999?'Q 已到 p3；K 仍在 p0–p2。':(m?'Q p2→p3；K 不动。':'当前 Q p2 → 目标 p3；K 停在 p0–p2。')):
   (rot>=.999?'Q 已到 p5；K 已到 p3–p5。':(m?'整体 p0–p2 → p3–p5。':'当前 Q p2 → p5；K p0–p2 → p3–p5。')));
  const n0=m?40:88,step=m?50:103,cardW=m?42:64,cardH=m?32:36;
  const cardY=trackY+(m?60:62),pinY=trackY+(m?40:38);
  const qx=n0+qPos*step;
  put(qSlot,qx-cardW/2-3,cardY-2,cardW+6,cardH+4);
  const nearK=kPos.some(v=>Math.abs(v-qPos)<.16);
  qSlot.setAttribute('class','cmp-qslot'+(nearK?' is-filled':''));
  qConnector.setAttribute('x1',qx.toFixed(2));qConnector.setAttribute('x2',qx.toFixed(2));
  qConnector.setAttribute('y1',(pinY+16).toFixed(2));qConnector.setAttribute('y2',cardY.toFixed(2));
  qPin.setAttribute('transform',`translate(${(qx-24).toFixed(2)} ${pinY})`);
  setText(qPinText,Math.abs(qPos-c.q)>.02?`Q→p${c.q}`:`Q p${c.q}`);
  slotTicks.forEach((dot,i)=>{
   dot.setAttribute('cx',(n0+i*step).toFixed(2));
   dot.setAttribute('cy',(cardY+cardH+(m?2:3)).toFixed(2));
  });
  slotLabels.forEach((t,i)=>{
   place(t,(n0+i*step).toFixed(2),(cardY+cardH+(m?10:8)).toFixed(2));
   size(t,m?7.2:8);
  });
  targetCards.forEach((card,i)=>{
   const tx=n0+c.ks[i]*step;
   card.g.setAttribute('transform',`translate(${(tx-cardW/2).toFixed(2)} ${cardY})`);
   put(card.bg,0,0,cardW,cardH);
   place(card.label,cardW/2,m?12:13);
   setText(card.label,'目标 p'+c.ks[i]);
   const moving=Math.abs(kPos[i]-c.ks[i])>.02;
   show(card.g,moving);fade(card.g,moving?.9:0);
  });
  cards.forEach((card,i)=>{
   const x=n0+kPos[i]*step;
   const moving=Math.abs(kPos[i]-c.ks[i])>.02;
   card.g.setAttribute('transform',`translate(${(x-cardW/2).toFixed(2)} ${cardY})`);
   put(card.bg,0,0,cardW,cardH);
   card.g.setAttribute('class','cmp-kcard'+(Math.abs(kPos[i]-qPos)<.16?' is-self':''));
   place(card.id,7,13);size(card.id,m?8:9.5);
   setText(card.id,'K'+(i+1));
   place(card.pos,cardW-6,13);size(card.pos,m?6.2:6.8);card.pos.setAttribute('text-anchor','end');
   setText(card.pos,moving?`p${FROM_KS[i]}→p${c.ks[i]}`:`p${c.ks[i]}`);
   place(card.content,cardW/2,m?24:28);size(card.content,m?8:9.5);card.content.setAttribute('text-anchor','middle');
   setText(card.content,m?'A':'内容 A');
   card.selfDot.setAttribute('cx',cardW-7);card.selfDot.setAttribute('cy',cardH-6);
   show(card.selfDot,Math.abs(kPos[i]-qPos)<.16);
  });

  // ------- fixed axes and unit circle. Endpoints, labels, projection and dots
  // all come from the same qVec/kVecs; the radius is the visible length proof.
  const circleX=m?8:14,circleY=m?292:226,circleW=m?344:332,circleH=m?150:222;
  put(circlePanel,circleX,circleY,circleW,circleH);
  place(circleTitle,circleX+12,circleY+(m?18:14));size(circleTitle,m?10.5:13.5);
  place(circleSub,circleX+12,circleY+(m?34:28));size(circleSub,m?7.6:9.3);
  const cx=m?88:180,cy=m?384:344,R=m?38:60;
  circleBg.setAttribute('cx',cx);circleBg.setAttribute('cy',cy);circleBg.setAttribute('r',R);
  setLine(axisX,{x:cx-R-10,y:cy},{x:cx+R+12,y:cy});
  setLine(axisY,{x:cx,y:cy+R+10},{x:cx,y:cy-R-12});
  axisX.setAttribute('marker-end',`url(#${axisMarkerId})`);
  axisY.setAttribute('marker-end',`url(#${axisMarkerId})`);
  place(axisXLabel,cx+R+14,cy+3);size(axisXLabel,7.2);
  place(axisYLabel,cx+4,cy-R-10);size(axisYLabel,7.2);
  show(axisXLabel,!m);show(axisYLabel,false);
  axisTicks.forEach((t,i)=>{
   const a=i*45,R1=R-3,R2=R+3;
   setLine(t,pointAt(cx,cy,R1,a),pointAt(cx,cy,R2,a));
  });
  const targetOn=(i)=>angGap(qDeg,i===0?targetQ:kDegs[i-1])>1.2&&angGap(rot>=.999?qDeg:qDeg,targetQ)>0.8;
  targetRays.forEach((ray,i)=>{
   const target=i===0?targetQ:targetKs[i-1];
   const cur=i===0?qDeg:kDegs[i-1];
   const on=angGap(cur,target)>1.2&&Math.abs(target-cur)<180;
   setLine(ray,{x:cx,y:cy},pointAt(cx,cy,R+2,target));
   show(ray,on);
  });
  ghostRays.forEach((ray,i)=>{
   const base=i===0?FROM_Q*45:FROM_KS[i-1]*45;
   const cur=i===0?qDeg:kDegs[i-1];
   const on=c.id==='shift3'&&rot>.03&&angGap(cur,base)>1.2;
   setLine(ray,{x:cx,y:cy},pointAt(cx,cy,R+2,base));
   show(ray,on);
  });
  kLines.forEach((line,i)=>{
   const tip=pointAt(cx,cy,R,kDegs[i]);
   setLine(line,{x:cx,y:cy},tip);
   line.setAttribute('data-angle',kDegs[i].toFixed(3));
   line.setAttribute('data-length',String(R));
  });
  const qTipPoint=pointAt(cx,cy,R,qDeg);
  setLine(qLine,{x:cx,y:cy},qTipPoint);
  qLine.setAttribute('data-angle',qDeg.toFixed(3));
  qLine.setAttribute('data-length',String(R));
  kTips.forEach((tip,i)=>{
   const p=pointAt(cx,cy,R,kDegs[i]);
   tip.setAttribute('cx',p.x.toFixed(2));tip.setAttribute('cy',p.y.toFixed(2));
  });
  qTip.setAttribute('cx',qTipPoint.x.toFixed(2));qTip.setAttribute('cy',qTipPoint.y.toFixed(2));
  const allTogether=Math.max(qDeg,...kDegs)-Math.min(qDeg,...kDegs)<4;
  const qk3=angGap(qDeg,kDegs[2])<3;
  const chipVisible=!m&&!allTogether;
  const chips=layoutChips([
   {id:'Q',e:qLabel,text:'Q',angle:qDeg,tip:qTipPoint,leader:qLabelLeader,color:'is-q'},
   {id:'K1',e:kLabels[0],text:'K1',angle:kDegs[0],tip:pointAt(cx,cy,R,kDegs[0]),leader:kLabelLeaders[0],color:'is-k1'},
   {id:'K2',e:kLabels[1],text:'K2',angle:kDegs[1],tip:pointAt(cx,cy,R,kDegs[1]),leader:kLabelLeaders[1],color:'is-k2'},
   {id:'K3',e:kLabels[2],text:'K3',angle:kDegs[2],tip:pointAt(cx,cy,R,kDegs[2]),leader:kLabelLeaders[2],color:'is-k3'}
  ],cx,cy,R+(m?11:20),{x0:24,y0:262,x1:336,y1:404});
  chips.forEach(ch=>{
   placeChip(ch.e,ch.text,ch,chipVisible);
   placeLeader(ch.leader,ch.tip,ch,chipVisible);
  });
  // The endpoint is a dot; the badge names both identities when they coincide.
  // It never uses "·", which is reserved for the dot-product readouts.
  const badge=allTogether?allLabel:qk3?qk3Label:null;
  [allLabel,qk3Label].forEach(e=>show(e,e===badge));
  if(badge){
   place(badge,m?344:332,m?436:414);size(badge,m?6.8:7.8);
   badge.setAttribute('text-anchor','end');
   setText(badge,allTogether?'Q 与 K1–K3 同向':'Q 与 K3 同向');
  }

  // Projection uses the K2 row's own dot. It is a numeric correspondence, not a
  // requirement that screen lengths in different panels be equal.
  const k2dot=qDotVals[1];
  const foot2={x:cx+qVec[0]*k2dot*R,y:cy-qVec[1]*k2dot*R};
  const k2tip=pointAt(cx,cy,R,kDegs[1]);
  const projOn=rot>.04&&angGap(qDeg,kDegs[1])>2;
  setLine(projLine,{x:cx,y:cy},foot2);
  setLine(projDrop,k2tip,foot2);
  projLine.setAttribute('data-value',String(k2dot));
  projLine.setAttribute('data-key','K2');
  projDrop.setAttribute('data-value',String(k2dot));
  projFoot.setAttribute('cx',foot2.x.toFixed(2));projFoot.setAttribute('cy',foot2.y.toFixed(2));
  show(projLine,projOn&&Math.abs(k2dot)>.012);show(projDrop,projOn);show(projFoot,projOn);
  if(m){
   place(projLabel,0,0);show(projLabel,false);
  }else{
   place(projLabel,170,414);size(projLabel,7.4);projLabel.setAttribute('text-anchor','start');
   setText(projLabel,`K2 投影 ${sgn(k2dot,2)}`);show(projLabel,projOn);
  }

  const arcFrom=c.id==='base'?0:FROM_Q*45;
  const arcTo=qDeg;
  // The arc label reads the span that is actually on screen. arcTo already
  // includes the current progress, so multiplying by rot again would show 4°
  // beside an 18° endpoint (and the same error for q1).
  const arcSpan=arcTo-arcFrom;
  const arcR=R+(m?9:13);
  if(rot>.02&&Math.abs(arcSpan)>.5){
   const a=arcData(cx,cy,arcR,arcFrom,arcTo);
   commonArc.setAttribute('d',a.d);show(commonArc,true);
   place(arcLabel,m?8:26,m?436:414);size(arcLabel,m?6.8:7.4);arcLabel.setAttribute('text-anchor','start');
   setText(arcLabel,c.id==='shift3'?`共同 +${round(arcSpan)}°`:`Q 弧 +${round(arcSpan)}°`);
   show(arcLabel,true);
  }else{show(commonArc,false);show(arcLabel,false);}

  // Current → target angles stay visible in a fixed key. Keeping the numbers
  // out of the endpoint chips is what removes the small-angle label pile-up.
  if(m){
   show(legendIds[0],false);show(legendIds[1],false);show(legendIds[2],false);show(legendIds[3],false);
   place(angleHeader,152,340);size(angleHeader,7.2);show(angleHeader,true);
   place(qAngleText,152,356);size(qAngleText,8.2);setText(qAngleText,pairAngle('Q',qDeg,targetQ));show(qAngleText,true);
   kAngleTexts.forEach((t,i)=>{place(t,152,372+i*16);size(t,8.2);setText(t,pairAngle('K'+(i+1),kDegs[i],targetKs[i]));show(t,true);});
   place(projNote,152,424);size(projNote,7.2);setText(projNote,`K2 投影 = q′·k′ ${sgn(k2dot,2)}`);show(projNote,true);
  }else{
   show(angleHeader,false);show(projNote,false);
   const keyX=[50,132,214,296],keyTexts=[qAngleText,...kAngleTexts],keyIds=['Q','K1','K2','K3'];
   keyTexts.forEach((t,i)=>{
    place(t,keyX[i],442);size(t,7.4);t.setAttribute('text-anchor','middle');
    setText(t,`${round(i===0?qDeg:kDegs[i-1])}°→${round(i===0?targetQ:targetKs[i-1])}°`);show(t,true);
    place(legendIds[i],keyX[i],430);size(legendIds[i],8.2);legendIds[i].setAttribute('text-anchor','middle');
    show(legendIds[i],true);
   });
  }

  // ------- three rows, zero in the middle, negative grows left. Every value
  // here is the dot of the same qVec/kVecs used by the circle.
  const rowsX=m?8:352,rowsY=m?448:226,rowsW=m?344:354,rowsH=m?190:222;
  put(rowsPanel,rowsX,rowsY,rowsW,rowsH);
  place(rowsTitle,rowsX+12,rowsY+(m?16:14));size(rowsTitle,m?10.5:13.5);
  setText(rowsTitle,m?'③ Q 分别比较 K1–K3':'③ 三行比较：Q 分别对 K1/K2/K3');
  place(rowsSub,rowsX+12,rowsY+(m?32:28));size(rowsSub,m?8:9.3);
  setText(rowsSub,m?'当前 q′·k′；Δθ=Q−K，零居中':'用当前 q′、k′ 算 Δθ 与点积；Δθ=Q−K，零居中');
  const rowX=m?18:360,rowW=m?324:338,rowH=m?46:50,rowGap=m?6:10;
  const rowY0=rowsY+(m?38:38);
  const positionText=i=>Math.abs(kPos[i]-c.ks[i])>.02?`p${FROM_KS[i]}→p${c.ks[i]}`:`p${c.ks[i]}`;
  rows.forEach((r,i)=>{
   const rowY=rowY0+i*(rowH+rowGap);
   const cur=kVecs[i],dot=qDotVals[i],rel=curRels[i],tgt=targetDots[i],trel=targetRels[i];
   const self=c.ks[i]===c.q;
   put(r.bg,rowX,rowY,rowW,rowH,7);
   r.bg.setAttribute('class','cmp-row'+(self?' is-self':'')+(i===1?' is-example':''));
   r.bg.setAttribute('data-key','K'+(i+1));
   place(r.label,rowX+9,rowY+(m?14:15));size(r.label,m?8.4:9.5);
   r.label.style.fill=K_COLORS[i];
   setText(r.label,`K${i+1} · ${positionText(i)}`);
   place(r.delta,rowX+(m?78:86),rowY+(m?14:15));size(r.delta,m?7.6:8.4);
   setText(r.delta,`Δθ ${round(rel)}°`+(Math.abs(rel-trel)>.5?`→${round(trel)}°`:''));
   place(r.value,rowX+rowW-8,rowY+(m?14:15));size(r.value,m?7.6:8.6);r.value.setAttribute('text-anchor','end');
   setText(r.value,`q′·k′ ${sgn(dot,2)} · ${verdict(dot)}`);
   const barX=rowX+9,barW=rowW-18,barY=rowY+(m?25:28),barH=m?5:6;
   const mid=barX+barW/2,half=barW/2,mag=Math.abs(dot)<5e-9?0:Math.abs(dot);
   put(r.barBg,barX,barY,barW,barH,m?3:4);
   put(r.barFill,dot<0?mid-half*mag:mid,barY,half*mag,barH,m?3:4);
   r.barFill.setAttribute('class','cmp-compare-fill'+(dot<0?' is-negative':'')+(mag===0?' is-zero':''));
   r.barFill.setAttribute('fill',dot<0?'#f0808b':K_COLORS[i]);
   r.barFill.setAttribute('data-width',(half*mag).toFixed(2));
   r.barFill.setAttribute('data-value',String(dot));
   r.barFill.setAttribute('data-sign',dot<-5e-9?'negative':dot>5e-9?'positive':'zero');
   r.barFill.setAttribute('data-identity','K'+(i+1));
   r.zeroMark.setAttribute('x1',mid);r.zeroMark.setAttribute('x2',mid);
   r.zeroMark.setAttribute('y1',barY-2);r.zeroMark.setAttribute('y2',barY+barH+2);
   place(r.negTick,barX,barY+barH+(m?9:9));size(r.negTick,6.8);r.negTick.setAttribute('text-anchor','start');
   place(r.zero,mid,barY+barH+(m?9:9));size(r.zero,6.8);r.zero.setAttribute('text-anchor','middle');
   place(r.posTick,barX+barW,barY+barH+(m?9:9));size(r.posTick,6.8);r.posTick.setAttribute('text-anchor','end');
  });
  show(rowsGroup,!(m&&detail));
  show(rowsPanel,true);show(rowsTitle,true);show(rowsSub,true);

  // ------- plain-language cause, then the optional derivation.
  const noteY=m?654:464;
  if(detail){
   const dx=m?8:14,dy=m?448:451,dw=m?344:692,dh=m?248:85;
   put(detailBg,dx,dy,dw,dh);
   place(detailTitle,dx+12,dy+(m?20:14));size(detailTitle,m?9.5:10.5);
   place(detailSub1,dx+12,dy+(m?38:26));size(detailSub1,m?7.4:8.4);
   place(detailSub2,dx+12,dy+(m?52:36));size(detailSub2,m?7.1:8.1);
   place(detailSub3,dx+12,dy+(m?65:46));size(detailSub3,m?7.1:8.1);
   const sub1=c.id==='shift3'?'整段 +3：轨道整体右移，四个向量共同 +135°；Δθ 与点积不变。':
    c.id==='q1'?'只移动 Q：Q 多转 45°，三个 K 的角度不动；每行 Δθ 都增加 45°。':
    '起点：Q/K 从 0° 开始按各自位置旋转；p 不同，转过的角不同。';
   const sub2=m?'一般 q′·k′=|q||k|cos(Δθ−φ)，φ 是原内容中 K 相对 Q 的有向角。':
    '本例 q=k=[1,0]、|q|=|k|=1 ⇒ q′·k′=cosΔθ；Δθ=Q−K 是旋转后的位置夹角。';
   const sub3=m?'本例 φ=0、|q|=|k|=1 ⇒ cosΔθ；高维每对频率不同。':
    '一般 q′·k′=|q||k|cos(Δθ−φ)，φ 是原内容中 K 相对 Q 的有向夹角（arg k−arg q）；高维每对频率不同。';
   setText(detailSub1,sub1);setText(detailSub2,sub2);setText(detailSub3,sub3);
   rows.forEach((r,i)=>{
    const qa=pairAngle('Q',qDeg,targetQ),ka=pairAngle('K',kDegs[i],targetKs[i]);
    const da=`Δθ ${round(curRels[i])}°→${round(targetRels[i])}°`;
    const dv=`q′·k′ ${pairValue(qDotVals[i],targetDots[i])}`;
    const ps=positionText(i);
    const full=`K${i+1} · ${ps}：${qa} · ${ka} · ${da} · ${dv}`;
    const first=`K${i+1} · ${ps} · ${qa} · ${ka}`;
    const second=`${da} · ${dv}`;
    const primaryY=dy+(m?88+i*36:58+i*11);
    place(detailCols[i],dx+12,primaryY);size(detailCols[i],m?7.2:8.4);
    setText(detailCols[i],m?first:full);show(detailCols[i],true);
    if(m){place(detailSubCols[i],dx+12,primaryY+13);size(detailSubCols[i],7.2);setText(detailSubCols[i],second);show(detailSubCols[i],true);}
    else{setText(detailSubCols[i],'');show(detailSubCols[i],false);}
   });
   if(m){place(detailFoot,dx+12,dy+218);size(detailFoot,7.2);show(detailFoot,true);}
   else show(detailFoot,false);
   show(detailBg,true);show(detailTitle,true);show(detailSub1,true);show(detailSub2,true);show(detailSub3,true);
   [note1,note2,boundary].forEach(e=>show(e,false));
  }else{
   show(detailBg,false);show(detailTitle,false);show(detailSub1,false);show(detailSub2,false);show(detailSub3,false);
   detailCols.forEach(e=>show(e,false));detailSubCols.forEach(e=>show(e,false));show(detailFoot,false);
   place(note1,14,noteY);size(note1,m?8.4:10.5);
   setText(note1,'位置不是贴编号：Q/K 真的转过一个角，点积看的是旋转后的夹角。');
   place(note2,14,noteY+(m?16:18));size(note2,m?8.2:9.6);
   setText(note2,'整段一起平移时，所有向量共同旋转，夹角不变，所以三行数值不变。');
   place(boundary,14,noteY+(m?32:36));size(boundary,m?7.2:8.4);
   setText(boundary,m?'只画一对维度 · 比较值是 softmax 前 Q·K，不是概率 · 不推出“越近越高”。':'只画一对维度 · 比较值是 softmax 前的 Q·K，不是概率 · 不推出“越近越高”。');
   [note1,note2,boundary].forEach(e=>show(e,true));
  }
  place(foot,m?14:18,HT-12);size(foot,m?7.2:9);
  setText(foot,m?'Q/K 一对特征分量，不是 token；V 不参与。softmax 前 Q·K，不是概率。':'只画一对维度：旋转 Q/K 的一对特征分量，不是 token；V 不参与。比较值是 softmax 前的 Q·K，不是概率。');
  show(foot,true);

  svg.dataset.component='rope';
  svg.dataset.condition=c.id;
  svg.dataset.qp=String(c.q);
  svg.dataset.pks=c.ks.join(',');
  svg.dataset.currentQp=qPos.toFixed(3);
  svg.dataset.currentPks=kPos.map(v=>v.toFixed(3)).join(',');
  svg.dataset.detail=detail?'true':'false';
  svg.dataset.progress=rot.toFixed(3);
  svg.dataset.plainDot='1';
  svg.dataset.radius=String(R);
  svg.dataset.commonRotation=String(c.id==='shift3'?135:0);
  svg.dataset.arcRotation=String(Number(arcSpan.toFixed(3)));
  svg.dataset.arcFrom=String(Number(arcFrom.toFixed(3)));
  svg.dataset.arcTo=String(Number(arcTo.toFixed(3)));
  svg.dataset.arcLabel=arcLabel.textContent;
  svg.dataset.jointLabel=badge?badge.textContent:'';
  svg.dataset.vectorAngles=[qDeg,...kDegs].map(v=>v.toFixed(3)).join(',');
  svg.dataset.projectionDot=String(qDotVals[1]);
  const rowsData=rows.map((r,i)=>({
   key:i+1,pos:c.ks[i],targetPos:c.ks[i],currentPos:Number(kPos[i].toFixed(3)),
   fromPos:FROM_KS[i],distance:Math.abs(c.q-c.ks[i]),
   angle:Number(targetRels[i].toFixed(3)),
   dot:targetDots[i],
   currentAngle:Number(curRels[i].toFixed(3)),
   currentDot:qDotVals[i],
   projection:qDotVals[i],
   qAngle:Number(qDeg.toFixed(3)),
   kAngle:Number(kDegs[i].toFixed(3)),
   fromQAngle:fromQAngle,
   fromKAngle:fromKAngles[i],
   targetQAngle:Number(targetQ.toFixed(3)),
   targetKAngle:Number(targetKs[i].toFixed(3)),
   commonRotation:c.id==='shift3'?135:0,
   self:c.ks[i]===c.q,
   verdict:verdict(qDotVals[i])
  }));
  svg.dataset.rows=JSON.stringify(rowsData);
  svg.dataset.relativeAngle=String(rowsData[0].angle);
  svg.dataset.dot=String(rowsData[0].dot);
  svg.dataset.currentDot=String(rowsData[0].currentDot);
  svg.dataset.currentAngle=String(rowsData[0].currentAngle);
  svg.dataset.targetAngle=String(rowsData[0].angle);
 }
 return draw;
}

/* ------------------------------------------------------------ 02 RMSNorm */
function rmsBoard(svg,H){
 const h=header(svg,'RMSNorm：先拉齐幅度，再逐维调整','x 与 2x 方向相同、幅度 1:2；先缩到近似共同尺度，再乘 γ。','先消整体幅度差，保留分量关系；γ 不是第二次归一化。');
 const gOpts=[{label:'γ=[1,1]',g:[1,1]},{label:'γ=[1,2]',g:[1,2]}];
 const gButtons=gOpts.map((c,i)=>btn(svg,0,0,100,28,c.label,()=>changeCondition(svg,gammaIdx,i,v=>{gammaIdx=v;})));
 const calcBtn=btn(svg,0,0,150,28,'展开计算',()=>{advanced=!advanced;pause(svg);draw(lastP,lastM);});
 const gammaHint=txt(svg,'att-dim',0,0,'默认 γ=[1,2]：dim1 拉长；切到 [1,1] 看方向保持');
 const inPanel=rect(svg,'att-panel',0,0,0,0),outPanel=rect(svg,'att-panel',0,0,0,0);
 const summaryPanel=rect(svg,'att-panel',0,0,0,0),detailPanel=rect(svg,'att-panel',0,0,0,0);
 const arrowInX=marker(svg,'#62d8e9'),arrowIn2=marker(svg,'#f2b45c');
 const arrowOutX=marker(svg,'#7fe0c3'),arrowOut2=marker(svg,'#f2b45c'),arrowGamma=marker(svg,'#bda9f7');
 const inRay=mk('line',{class:'cmp-rms-ray'},svg);
 const inArrow2=mk('line',{class:'cmp-input-arrow cmp-input-2x','marker-end':`url(#${arrowIn2})`},svg);
 const inArrowX=mk('line',{class:'cmp-input-arrow cmp-input-x','marker-end':`url(#${arrowInX})`},svg);
 const inTipX=mk('circle',{class:'cmp-input-tip-x',r:3.4},svg),inTip2=mk('circle',{class:'cmp-input-tip-2x',r:3.4},svg);
 const bracketLine=mk('path',{class:'cmp-rms-bracket',fill:'none'},svg);
 const bracketLabel=txt(svg,'att-dim',0,0,'2×');
 const outRay=mk('line',{class:'cmp-rms-ray'},svg);
 const outRayLabel=txt(svg,'att-dim',0,0,'原方向');
 const ghostX=mk('line',{class:'cmp-rms-ray'},svg),ghost2=mk('line',{class:'cmp-rms-ray'},svg);
 const outArrow2=mk('line',{class:'cmp-output-y cmp-output-2x','marker-end':`url(#${arrowOut2})`},svg);
 const outArrowX=mk('line',{class:'cmp-output-norm','marker-end':`url(#${arrowOutX})`},svg);
 const outTip2=mk('circle',{class:'cmp-tip-2x',r:7},svg),outTipX=mk('circle',{class:'cmp-tip-norm',r:4},svg);
 const preTip=mk('circle',{class:'cmp-pretip',r:4},svg);
 const gammaStretch=mk('line',{class:'cmp-gamma-stretch','marker-end':`url(#${arrowGamma})`},svg);
 const gammaStretchLabel=txt(svg,'att-k',0,0,'');
 const altArrow=mk('line',{class:'cmp-compare','marker-end':`url(#${arrowGamma})`},svg);
 const altTip=mk('circle',{class:'cmp-compare-tip',r:4},svg);
 const inTitle=txt(svg,'att-strong',0,0,'① 输入：方向相同，幅度差 2 倍');
 const inSub=txt(svg,'att-dim',0,0,'');
 const outTitle=txt(svg,'att-strong',0,0,'② 归一化：两条箭头缩到近似共同尺度');
 const outSub=txt(svg,'att-dim',0,0,'');
 const gammaTag=txt(svg,'att-score',0,0,'');
 const ratioTag=txt(svg,'att-note',0,0,''),epsTag=txt(svg,'att-dim',0,0,'');
 const rXTag=txt(svg,'att-q',0,0,''),r2Tag=txt(svg,'att-score',0,0,'');
 const inLabelX=txt(svg,'att-q',0,0,''),inLabel2=txt(svg,'att-score',0,0,'');
 const outLabelX=txt(svg,'att-w',0,0,'x'),outLabel2=txt(svg,'att-score',0,0,'2x');
 const altLabel=txt(svg,'att-k',0,0,'对照');
 const preLabel=txt(svg,'att-dim',0,0,'γ 前');
 const statusOut=txt(svg,'att-note',0,0,'');
 const compareTag=txt(svg,'att-w',0,0,'二维箭头近似重合');
 const dirBadge=txt(svg,'att-w',0,0,''),dirSub=txt(svg,'att-dim',0,0,'');
 const rmsLine=txt(svg,'att-dim',0,0,'');
 const summaryTitle=txt(svg,'att-strong',0,0,'');
 const summaryA=txt(svg,'att-num',0,0,''),summaryB=txt(svg,'att-num',0,0,'');
 const detailLines=[0,1,2,3].map(()=>txt(svg,'att-dim',0,0,''));
 const boundaryLines=[0,1].map(()=>txt(svg,'att-dim',0,0,''));
 const foot=footer(svg,'先统一整体幅度，再逐维乘 γ；归一化逐位置进行，不跨 token。',false,560);
 let gammaIdx=1,advanced=false,lastP=0,lastM=false;
 function setLine(e,a,b){e.setAttribute('x1',a.x);e.setAttribute('y1',a.y);e.setAttribute('x2',b.x);e.setAttribute('y2',b.y);}
 function vecPoint(o,v,s){return{x:o.x+v[0]*s,y:o.y-v[1]*s};}
 function draw(p,m){
  lastP=p;lastM=m;const W=m?360:720,HT=m?700:560;
  placeHeader(h,m,W);
  if(m){h.t1.style.fontSize='15px';h.t2.style.fontSize='9.8px';h.t3.style.fontSize='8.9px';}
  else{h.t1.style.fontSize='';h.t2.style.fontSize='';h.t3.style.fontSize='';}
  if(m){
   gButtons.forEach((b,i)=>placeBtn(b,14+i*108,84,102,28,gOpts[i].label,i===gammaIdx));
   placeBtn(calcBtn,230,84,116,28,advanced?'收起计算':'展开计算',advanced);
   show(gammaHint,false);
  }else{
   gButtons.forEach((b,i)=>placeBtn(b,18+i*106,88,100,28,gOpts[i].label,i===gammaIdx));
   placeBtn(calcBtn,552,88,150,28,advanced?'收起计算':'展开计算',advanced);
   place(gammaHint,240,106);show(gammaHint,true);
  }
  calcBtn.setAttribute('aria-expanded',advanced?'true':'false');
  const x0=[1.2,1.6],x2=[2.4,3.2],gamma=[1,gammaIdx?2:1];
  const rx=M.rmsnorm(x0,gamma),r2=M.rmsnorm(x2,gamma);
  const ux=3/5,uy=4/5,scale=m?22:42;
  const L=m?
   {in:[14,120,332,190],out:[14,318,332,190],sum:[14,516,332,54],detail:[14,576,332,92],inO:{x:60,y:296},outO:{x:60,y:480}}:
   {in:[18,124,326,248],out:[376,124,326,248],sum:[18,384,684,62],detail:[18,454,684,76],inO:{x:72,y:348},outO:{x:424,y:348}};
  const [ix,iy,iw,ih]=L.in,[ox,oy,ow,oh]=L.out,[sx,sy,sw,sh]=L.sum,[dx,dy,dw,dh]=L.detail;
  put(inPanel,ix,iy,iw,ih);put(outPanel,ox,oy,ow,oh);
  put(summaryPanel,sx,sy,sw,sh);put(detailPanel,dx,dy,dw,dh);
  place(inTitle,ix+12,iy+22);size(inTitle,m?10.8:11.5);
  place(inSub,ix+12,iy+40);size(inSub,m?8.8:9.7);
  place(outTitle,ox+12,oy+22);size(outTitle,m?10.8:11.5);
  place(outSub,ox+12,oy+40);size(outSub,m?8.8:9.7);
  place(gammaTag,ox+ow-12,oy+22);size(gammaTag,m?9.2:9.7);gammaTag.setAttribute('text-anchor','end');
  const inO=L.inO,outO=L.outO,inScale=scale,outScale=scale;
  const xTip=vecPoint(inO,x0,inScale),x2Tip=vecPoint(inO,x2,inScale);
  const inEnd=vecPoint(inO,[ux*4.9,uy*4.9],inScale);
  setLine(inRay,inO,inEnd);setLine(inArrowX,inO,xTip);setLine(inArrow2,inO,x2Tip);
  inTipX.setAttribute('cx',xTip.x);inTipX.setAttribute('cy',xTip.y);
  inTip2.setAttribute('cx',x2Tip.x);inTip2.setAttribute('cy',x2Tip.y);
  const perp={x:-uy,y:ux};
  setText(inSub,'两图同一把尺 · 2x 只是沿同一方向更长');
  setText(inLabelX,`x = ${fmt(x0,1)}`);setText(inLabel2,`2x = ${fmt(x2,1)}`);
  place(inLabelX,xTip.x+8,xTip.y+14);size(inLabelX,m?8.8:9.4);
  place(inLabel2,x2Tip.x+8,x2Tip.y+2);size(inLabel2,m?8.8:9.4);
  const measure=phase(p,.25,1);
  const boff=m?7:9,bA={x:xTip.x+perp.x*boff,y:xTip.y+perp.y*boff},bB={x:x2Tip.x+perp.x*boff,y:x2Tip.y+perp.y*boff};
  const cap=4;
  bracketLine.setAttribute('d',`M${bA.x} ${bA.y} L${bB.x} ${bB.y} M${bA.x-perp.x*cap} ${bA.y-perp.y*cap} L${bA.x+perp.x*cap} ${bA.y+perp.y*cap} M${bB.x-perp.x*cap} ${bB.y-perp.y*cap} L${bB.x+perp.x*cap} ${bB.y+perp.y*cap}`);
  place(bracketLabel,(bA.x+bB.x)/2-4,(bA.y+bB.y)/2+4);bracketLabel.setAttribute('text-anchor','end');size(bracketLabel,m?8.2:8.8);
  fade(bracketLine,measure);fade(bracketLabel,measure);show(bracketLine,measure>.01);show(bracketLabel,measure>.01);
  const norm=phase(p,1.05,2),n1=phase(p,1.05,1.7),n2=phase(p,1.25,1.95);
  const g=phase(p,3.05,3.97),cmp=phase(p,4.05,4.95);
  const gammaDone=g>=1,normDone=p>=2;
  const uniform=gamma.every(v=>Math.abs(v-gamma[0])<=1e-12);
  const normTip=vecPoint(outO,rx.normalized,outScale);
  const curTip=vecPoint(outO,rx.output,outScale);
  const drawVecX=x0.map((v,i)=>v+(rx.normalized[i]-v)*n1);
  const drawVec2=x2.map((v,i)=>v+(r2.normalized[i]-v)*n2);
  const activeVecX=drawVecX.map((v,i)=>v+(rx.normalized[i]*gamma[i]-v)*g);
  const activeVec2=drawVec2.map((v,i)=>v+(r2.normalized[i]*gamma[i]-v)*g);
  const drawX=vecPoint(outO,activeVecX,outScale),draw2=vecPoint(outO,activeVec2,outScale);
  const ghostOn=phase(p,.9,1.5);
  setLine(ghostX,outO,vecPoint(outO,x0,outScale));setLine(ghost2,outO,vecPoint(outO,x2,outScale));
  fade(ghostX,.32*ghostOn);fade(ghost2,.32*ghostOn);
  show(ghostX,ghostOn>.01);show(ghost2,ghostOn>.01);
  setLine(outArrowX,outO,drawX);setLine(outArrow2,outO,draw2);
  outTipX.setAttribute('cx',drawX.x);outTipX.setAttribute('cy',drawX.y);
  outTip2.setAttribute('cx',draw2.x);outTip2.setAttribute('cy',draw2.y);
  show(outArrowX,true);show(outArrow2,true);show(outTipX,true);show(outTip2,true);
  const outEnd=vecPoint(outO,[ux*4.9,uy*4.9],outScale);setLine(outRay,outO,outEnd);
  const rayOn=phase(p,1.65,2);fade(outRay,.5*rayOn);show(outRay,rayOn>.01);
  place(outRayLabel,outEnd.x+6,outEnd.y-3);show(outRayLabel,!m&&p>=2.2);
  setText(outSub,gammaDone?(uniform?'γ 均匀：方向不变':'γ 非均匀：只拉长 dim1'):g>0?'γ 逐维缩放中':normDone?'归一化完成：两条近似重合':'各自 ÷ 自身 RMS');
  setText(gammaTag,`× γ=${fmt(gamma,0)}`);
  setText(outTitle,gammaDone?'③ γ 完成：逐维缩放已生效':g>0?'③ 逐维 ×γ：dim1 拉长':normDone?'② 归一化完成：两条到近似共同尺度':'② 各自 ÷ 自己的 RMS：回到同一尺度');
  setText(rXTag,normDone?`x：RMS ${rx.rms.toFixed(2)} → 1.00`:`x：RMS ${rx.rms.toFixed(2)}`);
  setText(r2Tag,normDone?`2x：RMS ${r2.rms.toFixed(2)} → 1.00`:`2x：RMS ${r2.rms.toFixed(2)}`);
  place(rXTag,ox+12,oy+60);size(rXTag,m?8.4:9);
  place(r2Tag,ox+12,oy+76);size(r2Tag,m?8.4:9);
  show(rXTag,p>=.55);show(r2Tag,p>=.55);
  // Math-check readouts use the current drawn endpoints, never the final
  // endpoint while the clock is still at an earlier stage.
  const currentRmsX=Math.sqrt((activeVecX[0]*activeVecX[0]+activeVecX[1]*activeVecX[1])/2);
  const currentRms2=Math.sqrt((activeVec2[0]*activeVec2[0]+activeVec2[1]*activeVec2[1])/2);
  const rmsRatio=Math.max(currentRmsX,currentRms2)/Math.min(currentRmsX,currentRms2);
  const outGap=Math.hypot(activeVecX[0]-activeVec2[0],activeVecX[1]-activeVec2[1]);
  const gapStage=gammaDone?'γ 后':normDone?'归一化后':'当前';
  setText(ratioTag,`${gapStage} RMS 比 ${rmsRatio.toFixed(3)}×`);
  setText(epsTag,`x/2x ${gapStage}端点差≈${outGap.toExponential(1).replace('-','−')}${normDone?'（ε=1e−6）':''}`);
  place(ratioTag,ox+ow-12,oy+60);size(ratioTag,m?8:8.6);ratioTag.setAttribute('text-anchor','end');
  place(epsTag,ox+ow-12,oy+76);size(epsTag,m?7.4:8.2);epsTag.setAttribute('text-anchor','end');
  show(ratioTag,advanced&&p>=.55);show(epsTag,advanced&&p>=.55);
  const labelsOn=p>=.12;
  show(outLabelX,labelsOn);show(outLabel2,labelsOn);
  let labelX='x',label2='2x';
  if(gammaDone){labelX='y(x)';label2='y(2x)';}
  else if(g>0){labelX='来自 x';label2='来自 2x';}
  else if(normDone){labelX='来自 x';label2='来自 2x';}
  else if(p>=1){labelX='x ÷ 幅度';label2='2x ÷ 幅度';}
  setText(outLabelX,labelX);setText(outLabel2,label2);
  place(outLabelX,drawX.x+8,drawX.y-7);size(outLabelX,m?8.6:9.2);
  place(outLabel2,draw2.x+8,draw2.y+16);size(outLabel2,m?8.6:9.2);
  const statusOn=p>1&&p<1.96;show(statusOut,statusOn);
  // Fixed right-bottom note slot: the shrinking 2x tip passes through the old
  // centre-top position around p=1.3, so keep this caption off the arrow path.
  place(statusOut,ox+ow-12,oy+oh-28);statusOut.setAttribute('text-anchor','end');size(statusOut,m?8.6:9.2);
  setText(statusOut,'两条箭头正沿原方向缩短…');
  show(compareTag,p>=1.96);place(compareTag,ox+12,oy+94);size(compareTag,m?8.8:9.2);
  const moved=!uniform&&g>.01;
  show(preTip,moved);preTip.setAttribute('cx',normTip.x);preTip.setAttribute('cy',normTip.y);
  setLine(gammaStretch,normTip,curTip);show(gammaStretch,moved);
  place(gammaStretchLabel,ox+ow-12,oy+40);gammaStretchLabel.setAttribute('text-anchor','end');size(gammaStretchLabel,m?8:8.6);
  setText(gammaStretchLabel,`dim0 ×${gamma[0]} · dim1 ×${gamma[1]}`);show(gammaStretchLabel,g>.18);
  show(preLabel,moved&&g>.18);place(preLabel,normTip.x+8,normTip.y+30);size(preLabel,m?8.2:8.8);
  const altOn=cmp>.01;
  const altGamma=uniform?[1,2]:[1,1];
  const altTarget=[rx.normalized[0]*altGamma[0],rx.normalized[1]*altGamma[1]];
  const altTipP=vecPoint(outO,[rx.normalized[0]+(altTarget[0]-rx.normalized[0])*cmp,rx.normalized[1]+(altTarget[1]-rx.normalized[1])*cmp],outScale);
  setLine(altArrow,outO,altTipP);fade(altArrow,.85);show(altArrow,altOn);
  altTip.setAttribute('cx',altTipP.x);altTip.setAttribute('cy',altTipP.y);show(altTip,altOn);
  place(altLabel,altTipP.x+8,altTipP.y+15);size(altLabel,m?8.4:9);show(altLabel,cmp>.15);
  setText(altLabel,uniform?'虚线：γ=[1,2]':'虚线：γ=[1,1] 保方向');
  const a0=Math.atan2(-uy,ux);
  const a1=Math.atan2(drawX.y-outO.y,drawX.x-outO.x);
  const angleDiff=Math.abs(a1-a0)*180/Math.PI;
  const cmpTip=vecPoint(outO,[rx.normalized[0],rx.normalized[1]*2],outScale);
  const aCmp=Math.atan2(cmpTip.y-outO.y,cmpTip.x-outO.x);
  const cmpAngle=Math.abs(aCmp-a0)*180/Math.PI;
  const rmsOn=p>=1;show(rmsLine,rmsOn);place(rmsLine,ox+12,oy+oh-12);size(rmsLine,m?8.2:9.2);
  if(gammaDone)setText(rmsLine,`γ 完成：两条近似重合 · ${uniform?'方向不变':`偏转≈${angleDiff.toFixed(1)}°`}`);
  else if(g>0)setText(rmsLine,'γ 途中：逐维缩放…');
  else if(normDone)setText(rmsLine,'归一化完成：两条到近似共同尺度');
  else setText(rmsLine,'归一化途中：各自 ÷ 自身幅度');
  const dirOn=p>=3.75;show(dirBadge,dirOn);show(dirSub,dirOn);
  place(dirBadge,ox+ow-12,oy+94);dirBadge.setAttribute('text-anchor','end');size(dirBadge,m?8.8:9.2);
  place(dirSub,ox+ow-12,oy+110);dirSub.setAttribute('text-anchor','end');size(dirSub,m?8:8.6);
  dirBadge.setAttribute('class',uniform?'att-w':'att-score');
  if(uniform){
   if(cmp>.15){setText(dirBadge,'方向保持');setText(dirSub,cmp>.95?`虚线对照 γ=[1,2]：偏转≈${cmpAngle.toFixed(1)}°`:'虚线 γ=[1,2]：dim1 继续拉长');}
   else{setText(dirBadge,'γ 均匀');setText(dirSub,'两条仍沿原方向');}
  }else if(gammaDone){setText(dirBadge,'方向改变');setText(dirSub,`偏转≈${angleDiff.toFixed(1)}°${cmp>.15?' · 虚线 γ=[1,1] 保方向':''}`);}
  else{setText(dirBadge,'逐维缩放中');setText(dirSub,'dim1 正在拉长');}
  place(summaryTitle,sx+12,sy+22);size(summaryTitle,m?10.2:10.8);
  const angleText=`${angleDiff.toFixed(1)}°`;
  if(p<1){
   setText(summaryTitle,'① 两个输入：方向相同，幅度 1:2');
   setText(summaryA,'x=[1.2, 1.6] · 2x=[2.4, 3.2]');
   setText(summaryB,'2x 只是沿同一方向更长');
  }else if(p<2){
   setText(summaryTitle,'② 各自 ÷ 自己的整体幅度');
   setText(summaryA,`x ÷ ${rx.rms.toFixed(2)} · 2x ÷ ${r2.rms.toFixed(2)}`);
   setText(summaryB,'长的 2x 缩得更多，方向不变');
  }else if(p<3.05){
   setText(summaryTitle,'② 完成：幅度已接近共同尺度');
   setText(summaryA,'两条箭头长度已很接近');
   setText(summaryB,'方向与各分量的比例保留');
  }else if(!gammaDone){
   setText(summaryTitle,'③ γ 途中：dim0×γ0，dim1×γ1');
   setText(summaryA,uniform?'γ=[1,1]：dim0、dim1 都不变':'γ=[1,2]：dim0 不变，dim1 拉长');
   setText(summaryB,'γ 是逐维缩放，不是第二次归一化');
  }else if(uniform){
   setText(summaryTitle,'γ=[1,1] 完成：方向保持');
   setText(summaryA,`y=[${rx.output.map(v=>v.toFixed(3)).join(', ')}]`);
   setText(summaryB,cmp>.15?'虚线：dim1 拉长，方向偏转≈16.3°':'两条仍沿原方向');
  }else{
   setText(summaryTitle,'γ=[1,2] 完成：dim1 拉长，方向改变');
   setText(summaryA,`y=[${rx.output.map(v=>v.toFixed(3)).join(', ')}]`);
   setText(summaryB,`x 与 2x 近似重合 · 偏转≈${angleText}`);
  }
  if(m){place(summaryA,sx+12,sy+38);place(summaryB,sx+12,sy+50);}
  else{place(summaryA,sx+12,sy+44);place(summaryB,sx+344,sy+44);}
  size(summaryA,m?8.2:9.4);size(summaryB,m?8.2:9.4);
  const d0=m?`r_x≈${rx.rms.toFixed(3)} · r_2x≈${r2.rms.toFixed(3)}（ε=1e−6）`:`r_x≈${rx.rms.toFixed(6)} · r_2x≈${r2.rms.toFixed(6)}（r=√(均方+ε)）`;
  const d1=m?'x/r_x ≈ 2x/r_2x ≈ [0.8485, 1.1314]':'x/r_x ≈ 2x/r_2x ≈ [0.848528, 1.131371]（近似相等）';
  const d2=m?'γ 前 RMS≈1；二维长度≈1.414（RMS≠长度）':'γ 前分量 RMS≈1，二维欧氏长度≈1.414；RMS 不是长度';
  const d3=m?`γ=${fmt(gamma,0)} → y≈${fmt(rx.output,3)} · RMS≈${rx.outputRms.toFixed(3)}`:`γ=${fmt(gamma,0)} → y=${fmt(rx.output,6)} · RMS≈${rx.outputRms.toFixed(6)} · ${uniform?'方向不变':'偏转≈16.31°'}`;
  setText(detailLines[0],d0);setText(detailLines[1],d1);setText(detailLines[2],d2);setText(detailLines[3],d3);
  detailLines.forEach((t,i)=>{place(t,dx+12,dy+(m?20:17)+i*15);size(t,m?8.4:9.2);});
  const b0=m?'两条输入方向相同，缩完会收敛到近似共同尺度。':'两条输入方向相同，各缩各的幅度后会收敛到近似共同尺度。';
  const b1=m?'幅度差基本拉平，方向与分量比例保留；γ 再调整。':'整体幅度差基本拉平，方向与分量比例保留；γ 再做逐维调整。';
  setText(boundaryLines[0],b0);setText(boundaryLines[1],b1);
  boundaryLines.forEach((t,i)=>{place(t,dx+12,dy+(m?20:17)+i*(m?17:16));size(t,m?8.4:9.2);});
  detailLines.forEach(t=>show(t,advanced));boundaryLines.forEach(t=>show(t,!advanced));
  setText(foot,m?'先统一整体幅度，再逐维乘 γ；逐位置，不减均值，不跨 token。':'先统一整体幅度，再逐维乘 γ；逐位置处理，不减均值、不跨 token 平均；RMS≈1 不是欧氏长度 1。');
  place(foot,m?14:18,HT-12);show(foot,true);
  svg.dataset.component='rmsnorm';
  svg.dataset.rms=rx.rms.toFixed(6);
  svg.dataset.r2Rms=r2.rms.toFixed(6);
  svg.dataset.output=fmt(rx.output,6);
  svg.dataset.gamma=fmt(gamma,0);
  svg.dataset.direction=String(rx.directionPreserved);
  svg.dataset.input=fmt(x0,2);
  svg.dataset.normalized=fmt(rx.normalized,6);
  svg.dataset.outputLength=rx.outputLength.toFixed(6);
  svg.dataset.outputRms=rx.outputRms.toFixed(6);
  svg.dataset.progress=p.toFixed(3);
  svg.dataset.phase=p<1?'input':p<2?'normalize':p<3.05?'measure':p<3.97?'gamma':'compare';
  svg.dataset.normProgress=norm.toFixed(3);
  svg.dataset.gammaProgress=g.toFixed(3);
  svg.dataset.compareProgress=cmp.toFixed(3);
  svg.dataset.angleDegrees=angleDiff.toFixed(2);
  svg.dataset.compareAngleDegrees=(uniform?cmpAngle:0).toFixed(2);
  svg.dataset.rmsRatio=rmsRatio.toFixed(6);
  svg.dataset.currentRms=currentRmsX.toFixed(6);
  svg.dataset.currentRms2=currentRms2.toFixed(6);
  svg.dataset.outputGap=outGap.toExponential(6);
  svg.dataset.outputGapStage=gapStage;
  svg.dataset.outDraw=`${drawX.x.toFixed(2)},${drawX.y.toFixed(2)}`;
  svg.dataset.outTip=`${curTip.x.toFixed(2)},${curTip.y.toFixed(2)}`;
 }
 return draw;
}

/* ------------------------------------------------------------ 03 Residual */
function residualBoard(svg,H){
 const h=header(svg,'残差：已有表示继续走，子层只加更新','同一个 x 分两路：主路原样保留；子层只产生可正可负的 F(x)。','教学例：每维一把尺；先看基底，再看更新怎样到达并合并。');
 const fOpts=[[.5,-1],[0,0],[1,1]];
 const labelsText=['F=[0.5, −1]','F=[0,0]','F=[1,1]'];
 const buttons=fOpts.map((fx,i)=>btn(svg,0,0,150,28,labelsText[i],()=>changeCondition(svg,fi,i,v=>{fi=v;})));
 const sourceNote=txt(svg,'att-dim',0,0,'x=[1,2] · F 读同一个 x');
 const hint=txt(svg,'att-dim',0,0,'换条件 → 回到 0 并暂停');
 const mainPanel=rect(svg,'att-panel',0,0,0,0),statusPanel=rect(svg,'att-panel',0,0,0,0);
 const mainTitle=txt(svg,'att-strong',0,0,'每维一把尺：x 基底保留，F 只改对应维');
 const legend=[
  {kind:'rect',cls:'cmp-xbar',label:'x 基底'},
  {kind:'rect',cls:'cmp-res-fonly',label:'+ 更新'},
  {kind:'rect',cls:'cmp-res-fonly is-negative',label:'− 更新'},
  {kind:'circle',cls:'cmp-res-result',label:'y 结果'}
 ].map(it=>{
  const mark=it.kind==='rect'?mk('rect',{rx:2,class:it.cls},svg):mk('circle',{r:4,class:it.cls},svg);
  const label=txt(svg,'cmp-res-legend-label',0,0,it.label);
  return {...it,mark,label};
 });
 const dimRows=[0,1].map(i=>{
  const label=txt(svg,'cmp-res-dim',0,0,'dim'+i);
  const laneLabels=[
   txt(svg,'cmp-res-lane',0,0,'x · 旁路'),
   txt(svg,'cmp-res-lane',0,0,'只传 F'),
   txt(svg,'cmp-res-lane',0,0,'x + F')
  ];
  const values=[
   txt(svg,'cmp-res-value',0,0,''),
   txt(svg,'cmp-res-value',0,0,''),
   txt(svg,'cmp-res-pending',0,0,'')
  ];
  return {label,laneLabels,values};
 });
 const axes=[0,1].map(()=>{
  const axis=mk('line',{class:'cmp-res-axis'},svg);
  const zero=mk('line',{class:'cmp-res-zero'},svg);
  const tickLines=[-1,0,1,2,3].map(()=>mk('line',{class:'cmp-res-tick'},svg));
  const tickLabels=[-1,0,1,2,3].map(v=>txt(svg,'cmp-res-tick-label',0,0,v<0?'−1':String(v)));
  return {axis,zero,tickLines,tickLabels};
 });
 const xBars=[0,1].map(()=>mk('rect',{class:'cmp-xbar',rx:2},svg));
 const fBars=[0,1].map(()=>mk('rect',{class:'cmp-res-fonly',rx:2},svg));
 const fZeroDots=[0,1].map(()=>mk('circle',{class:'cmp-res-zero-dot',r:3.5},svg));
 const yBases=[0,1].map(()=>mk('rect',{class:'cmp-res-base',rx:3},svg));
 const yDeltas=[0,1].map(()=>mk('rect',{class:'cmp-res-delta',rx:2},svg));
 const diffBands=[0,1].map(()=>mk('rect',{class:'cmp-res-diff',rx:2},svg));
 const diffLabels=[0,1].map(()=>txt(svg,'cmp-res-diff-label',0,0,''));
 const pluses=[0,1].map(()=>{
  const g=mk('g',{class:'cmp-plus'},svg),c=mk('circle',{r:8},g),t=mk('text',{y:4,'text-anchor':'middle'},g);
  t.textContent='+';
  return {g,c,t};
 });
 const chipPaths=[0,1].map(()=>mk('path',{class:'cmp-update-path'},svg));
 const chips=[0,1].map(()=>{
  const g=mk('g',{class:'cmp-update-chip'},svg),r=mk('rect',{rx:8,width:48,height:20},g),t=mk('text',{x:24,y:14,'text-anchor':'middle'},g);
  return {g,r,t};
 });
 const resultStems=[0,1].map(()=>mk('line',{class:'cmp-res-result-stem'},svg));
 const resultDots=[0,1].map(()=>mk('circle',{class:'cmp-res-result',r:5},svg));
 const resultTags=[0,1].map(()=>txt(svg,'cmp-res-result-tag',0,0,''));
 const progressLabels=[0,1].map(()=>txt(svg,'cmp-res-progress',0,0,''));
 const compareNote=txt(svg,'cmp-res-note',0,0,'');
 const sTitle=txt(svg,'att-strong',0,0,''),sLine0=txt(svg,'att-score',0,0,''),sLine1=txt(svg,'att-score',0,0,'');
 const sShape=txt(svg,'cmp-res-note',0,0,''),sCondition=txt(svg,'cmp-res-note',0,0,''),sCompare=txt(svg,'cmp-res-note',0,0,'');
 const foot=footer(svg,'',false,560);
 let fi=0,lastP=0,lastM=false;
 function signed(v){const n=Math.abs(v);return v===0?'0':(v>0?'+':'−')+(Number.isInteger(n)?String(n):n.toFixed(1));}
 function vecText(v){return '['+v.map(n=>n<0?'−'+Math.abs(n):String(n)).join(', ')+']';}
 function setLine(e,a,b){e.setAttribute('x1',a.x);e.setAttribute('y1',a.y);e.setAttribute('x2',b.x);e.setAttribute('y2',b.y);}
 function putBar(e,left,width,y,height){
  const w=Math.max(0,width);
  e.setAttribute('x',left.toFixed(2));e.setAttribute('y',y.toFixed(2));
  e.setAttribute('width',w.toFixed(2));e.setAttribute('height',height);
  show(e,w>=.5);
 }
 function draw(p,m){
  lastP=p;lastM=m;const W=m?360:720,HT=m?700:560;
  placeHeader(h,m,W);
  if(m){h.t1.style.fontSize='15px';h.t2.style.fontSize='9.8px';h.t3.style.fontSize='8.9px';}
  else{h.t1.style.fontSize='';h.t2.style.fontSize='';h.t3.style.fontSize='';}
  buttons.forEach((b,i)=>{
   if(m)placeBtn(b,14+i*108,84,102,28,labelsText[i],i===fi);
   else placeBtn(b,18+i*152,84,152,28,labelsText[i],i===fi);
  });
  if(m){show(sourceNote,false);show(hint,false);}
  else{
   place(sourceNote,512,94);size(sourceNote,9.2);show(sourceNote,true);
   place(hint,512,108);size(hint,9);show(hint,true);
  }
  const r=M.residual([1,2],fOpts[fi]);
  const L=m?
   {panel:[10,120,340,332],status:[10,460,340,206],title:[20,141],
    dims:[{ruler:184,x:208,f:236,y:264},{ruler:300,x:324,f:352,y:380}],
    ruler0:116,ruler1:344,compare:[20,414],dimX:16,laneX:42,valueX:108}:
   {panel:[14,120,692,310],status:[14,430,692,94],title:[28,141],
    dims:[{ruler:182,x:206,f:234,y:262},{ruler:288,x:312,f:340,y:368}],
    ruler0:170,ruler1:680,compare:[28,406],dimX:28,laneX:60,valueX:160};
  const [px,py,pw,ph]=L.panel,[tx,ty,tw,th]=L.status;
  put(mainPanel,px,py,pw,ph);put(statusPanel,tx,ty,tw,th);
  place(mainTitle,L.title[0],L.title[1]);size(mainTitle,m?11.4:12.5);
  const legendX=m?[20,105,190,275]:[28,190,352,514];
  legend.forEach((it,i)=>{
   const x=legendX[i],y=m?153:154;
   if(it.kind==='rect')put(it.mark,x,y,9,9);
   else{it.mark.setAttribute('cx',x+4);it.mark.setAttribute('cy',y+4);}
   place(it.label,x+(m?13:14),y+8);size(it.label,m?7.4:8.5);
  });
  const MIN=-1.4,MAX=3.4;
  const scale=(L.ruler1-L.ruler0)/(MAX-MIN);
  const vx=v=>L.ruler0+(v-MIN)*scale;
  const baseT=phase(p,0,.28),fT=phase(p,.35,1.05);
  // A contribution must finish arriving before that dimension starts merging.
  const arrive0=phase(p,1.05,1.5),arrive1=phase(p,1.6,2.05);
  const merge0=phase(p,1.5,1.85),merge1=phase(p,2.05,2.4);
  const arrivals=[arrive0,arrive1],merges=[merge0,merge1];
  const done=[merge0>=.999,merge1>=.999],complete=done[0]&&done[1];
  const rowNow=[r.x[0]+r.fx[0]*merge0,r.x[1]+r.fx[1]*merge1];
  const compare=phase(p,2.9,3.55);
  axes.forEach((axis,di)=>{
   const g=L.dims[di];
   setLine(axis.axis,{x:L.ruler0,y:g.ruler},{x:L.ruler1,y:g.ruler});
   setLine(axis.zero,{x:vx(0),y:g.ruler},{x:vx(0),y:g.y+10});
   [-1,0,1,2,3].forEach((v,j)=>{
    const x=vx(v);
    setLine(axis.tickLines[j],{x,y:g.ruler-4},{x,y:g.ruler+4});
    place(axis.tickLabels[j],x,g.ruler-7);axis.tickLabels[j].setAttribute('text-anchor','middle');
    size(axis.tickLabels[j],m?7.2:8);
   });
  });
  dimRows.forEach((row,i)=>{
   const g=L.dims[i];
   place(row.label,L.dimX,g.x+4);size(row.label,m?8.6:10);
   place(row.laneLabels[0],L.laneX,g.x+4);setText(row.laneLabels[0],m?'x':'x · 旁路');size(row.laneLabels[0],m?8:9.2);
   place(row.laneLabels[1],L.laneX,g.f+4);setText(row.laneLabels[1],'只传 F');size(row.laneLabels[1],m?8:9.2);
   place(row.laneLabels[2],L.laneX,g.y+4);setText(row.laneLabels[2],'x + F');size(row.laneLabels[2],m?8:9.2);
   place(row.values[0],L.valueX,g.x+4);row.values[0].setAttribute('text-anchor','end');
   setText(row.values[0],String(r.x[i]));row.values[0].setAttribute('class','cmp-res-value');size(row.values[0],m?11.5:13.5);
   place(row.values[1],L.valueX,g.f+4);row.values[1].setAttribute('text-anchor','end');
   setText(row.values[1],signed(r.fx[i]));
   row.values[1].setAttribute('class','cmp-res-value'+(r.fx[i]<0?' is-negative':'')+(r.zeroUpdate?' is-zero':''));size(row.values[1],m?10:11);
   place(row.values[2],L.valueX,g.y+4);row.values[2].setAttribute('text-anchor','end');
   if(done[i]){setText(row.values[2],String(r.sum[i]));row.values[2].setAttribute('class','cmp-res-value is-final');size(row.values[2],m?13:16);}
   else{setText(row.values[2],'待合并');row.values[2].setAttribute('class','cmp-res-pending');size(row.values[2],m?8.4:9.2);}
   const baseW=Math.abs(vx(r.x[i])-vx(0))*baseT;
   const barH=m?10:12,baseH=m?12:14;
   putBar(xBars[i],vx(0),baseW,g.x-(m?5:6),barH);
   putBar(yBases[i],vx(0),baseW,g.y-(m?6:7),baseH);
   const fw=Math.abs(vx(r.fx[i])-vx(0))*fT;
   if(r.zeroUpdate){
    show(fBars[i],false);
    fZeroDots[i].setAttribute('cx',vx(0));fZeroDots[i].setAttribute('cy',g.f);
    show(fZeroDots[i],fT>.05);fade(fZeroDots[i],fT);
   }else{
    putBar(fBars[i],r.fx[i]<0?vx(0)-fw:vx(0),fw,g.f-(m?4:4.5),m?8:9);
    fBars[i].setAttribute('class','cmp-res-fonly'+(r.fx[i]<0?' is-negative':''));
    show(fZeroDots[i],false);
   }
   const deltaW=Math.abs(vx(rowNow[i])-vx(r.x[i]));
   putBar(yDeltas[i],r.fx[i]<0?vx(r.x[i])-deltaW:vx(r.x[i]),deltaW,g.y-(m?3.5:4),m?7:8);
   yDeltas[i].setAttribute('class','cmp-res-delta'+(r.fx[i]<0?' is-negative':'')+(deltaW<.5?' is-zero':''));
   const dotX=vx(rowNow[i]);
   resultDots[i].setAttribute('cx',dotX);resultDots[i].setAttribute('cy',g.y);
   resultStems[i].setAttribute('x1',dotX);resultStems[i].setAttribute('x2',dotX);
   resultStems[i].setAttribute('y1',g.y-(m?11:12));resultStems[i].setAttribute('y2',g.y+(m?8:9));
   show(resultDots[i],baseT>.05);fade(resultDots[i],baseT);
   show(resultStems[i],baseT>.05);fade(resultStems[i],baseT);
   pluses[i].g.setAttribute('transform',`translate(${vx(r.x[i]).toFixed(2)} ${g.y})`);
   pluses[i].g.setAttribute('class','cmp-plus'+(r.zeroUpdate?' cmp-plus-zero':'')+(done[i]?' is-complete':'')+(arrivals[i]>=.999?' is-arrived':''));
   pluses[i].t.style.fontSize=(m?8.5:9.5)+'px';
   show(pluses[i].g,baseT>.05);fade(pluses[i].g,baseT);
   const sx=vx(r.fx[i]),sy=g.f,tx0=vx(r.x[i]),ty0=g.y;
   chipPaths[i].setAttribute('d',`M${sx.toFixed(2)} ${sy} C${sx.toFixed(2)} ${sy+16} ${tx0.toFixed(2)} ${ty0-16} ${tx0.toFixed(2)} ${ty0}`);
   const total=chipPaths[i].getTotalLength();
   const pt=total>0?chipPaths[i].getPointAtLength(total*arrivals[i]):{x:sx,y:sy};
   chips[i].g.setAttribute('transform',`translate(${(pt.x-24).toFixed(2)} ${(pt.y-10).toFixed(2)})`);
   chips[i].g.setAttribute('class','cmp-update-chip'+(r.zeroUpdate?' is-zero':''));
   setText(chips[i].t,signed(r.fx[i]));
   const chipAlpha=fT*(1-phase(arrivals[i],.92,1));
   show(chips[i].g,chipAlpha>.01);fade(chips[i].g,chipAlpha);
   show(chipPaths[i],chipAlpha>.01);fade(chipPaths[i],.45*chipAlpha);
   if(!done[i]&&merges[i]>.01){
    setText(progressLabels[i],Math.round(merges[i]*100)+'%');
    place(progressLabels[i],dotX,g.y-(m?12:13));progressLabels[i].setAttribute('text-anchor','middle');
    show(progressLabels[i],true);fade(progressLabels[i],merges[i]);size(progressLabels[i],m?8:8.5);
   }else show(progressLabels[i],false);
   if(done[i]){
    setText(resultTags[i],`y${i}`);place(resultTags[i],dotX,g.y-(m?12:13));resultTags[i].setAttribute('text-anchor','middle');
    show(resultTags[i],true);fade(resultTags[i],1);size(resultTags[i],m?8:8.5);
   }else show(resultTags[i],false);
   const bandTop=g.f+4,bandH=g.y-g.f-8;
   const fEnd=vx(r.fx[i]),yEnd=dotX,left=Math.min(fEnd,yEnd),width=Math.abs(fEnd-yEnd);
   put(diffBands[i],left,bandTop,width,bandH);
   diffBands[i].setAttribute('class','cmp-res-diff'+(r.fx[i]<0?' is-negative':''));
   setText(diffLabels[i],`差 = x${i} = ${r.x[i]}`);
   place(diffLabels[i],left+width/2,(g.f+g.y)/2+3);diffLabels[i].setAttribute('text-anchor','middle');
   diffLabels[i].setAttribute('class','cmp-res-diff-label'+(r.fx[i]<0?' is-negative':'')+(r.zeroUpdate?' is-zero':''));
   show(diffBands[i],compare>.01);fade(diffBands[i],compare);
   show(diffLabels[i],compare>.01);fade(diffLabels[i],compare);size(diffLabels[i],m?8:8.5);
  });
  let stageTitle;
  if(!complete){
   if(p<.35)stageTitle='① x 先走主路：青色基底先出现';
   else if(p<1.05)stageTitle='② 子层读同一个 x，只产生 F';
   else if(p<1.5)stageTitle='③ dim0 更新在路上；到达前不合并';
   else if(p<1.85)stageTitle='③ dim0 合并中：y₀ = x₀ + F₀';
   else if(p<2.05)stageTitle='④ dim1 更新在路上；到达前不合并';
   else stageTitle='④ dim1 合并中：y₁ = x₁ + F₁';
  }else if(p<2.9)stageTitle=r.zeroUpdate?'合并完成：F=0，y 仍是 x':'合并完成：y = x + F（仍是 2 维）';
  else stageTitle=r.zeroUpdate?'F=0：只传 F 得 0，x+F 仍得 x':'对照：只传 F ≠ x+F，差正好是 x';
  const lineFor=i=>{
   const base=`dim${i} · x=${r.x[i]} · F=${signed(r.fx[i])}`;
   if(done[i])return `${base} · 完成 y${i}=${r.sum[i]}`;
   if(arrivals[i]>=.999)return `${base} · 更新已到 · 合并 ${Math.round(merges[i]*100)}%`;
   if(arrivals[i]>.01)return `${base} · 更新在途 ${Math.round(arrivals[i]*100)}%`;
   return `${base} · 待到达`;
  };
  setText(sTitle,stageTitle);place(sTitle,tx+(m?10:14),ty+22);size(sTitle,m?10.4:11.5);
  setText(sLine0,lineFor(0));setText(sLine1,lineFor(1));
  sLine0.setAttribute('class',done[0]?'att-num':'att-score');sLine1.setAttribute('class',done[1]?'att-num':'att-score');
  if(m){place(sLine0,20,ty+48);place(sLine1,20,ty+70);size(sLine0,9.2);size(sLine1,9.2);}
  else{place(sLine0,tx+14,ty+46);place(sLine1,tx+14,ty+66);size(sLine0,10);size(sLine1,10);}
  const doneCount=done.filter(Boolean).length;
  const shapeText=complete?`y=${fmt(r.sum,2)} · ${r.shape} 维，不拼接`:doneCount===1?'一维已写出；另一维仍待合并；不是拼接。':'y 暂不写出；等每维更新到达后再写精确值。';
  const conditionText=r.zeroUpdate?'F=0：子层没有增量；只传 F 得 [0,0]，主路仍把 x 送到 y。':r.fx.some(v=>v<0)?'F 含负值：dim1 的 −1 让 y₁ 小于 x₁；主路 x 仍在。':'F 全为正：两维都被推长；主路 x 仍是基底。';
  const compareText=compare>.05&&complete?
   (r.zeroUpdate?`只传 F ${vecText(r.fx)} vs y ${fmt(r.sum,2)}；F=0 时差值就是全部 x。`:`只传 F ${vecText(r.fx)} vs y ${fmt(r.sum,2)}；差=${fmt(r.x,0)}（就是主路 x）。`):
   (complete?'合并完成；下一步对照只传 F 与 x+F。':`合并进度 ${Math.round(Math.min(merge0,merge1)*100)}%；精确 y 只在完成后写出。`);
  if(m){
   place(sShape,20,ty+98);size(sShape,8.8);
   place(sCondition,20,ty+122);size(sCondition,8.8);
   place(sCompare,20,ty+146);size(sCompare,8.8);
  }else{
   place(sShape,tx+380,ty+24);size(sShape,9.6);
   place(sCondition,tx+380,ty+47);size(sCondition,9.4);
   place(sCompare,tx+380,ty+70);size(sCompare,9.4);
  }
  setText(sShape,shapeText);setText(sCondition,conditionText);setText(sCompare,compareText);
  setText(compareNote,m?'对照：只传 F 停在 F；x+F 还要加主路 x，差正好是 x。':'对照：只传 F 会停在 F 的端点；x+F 还要加上主路 x，所以两者差正好是 x，不是同一个输出。');
  place(compareNote,L.compare[0],L.compare[1]);size(compareNote,m?8.8:9.5);show(compareNote,compare>.01);fade(compareNote,compare);
  setText(foot,m?'教学例；F 来自同一个 x，逐元素相加；不声称加法无损。':'教学例；F 由同一个 x 得到，逐元素加到 x；不是拼接，也不声称每次加法都无损。');
  place(foot,m?14:18,HT-12);show(foot,true);
  svg.dataset.component='residual';
  svg.dataset.condition=String(fi);
  svg.dataset.x=fmt(r.x,6);svg.dataset.fx=fmt(r.fx,6);svg.dataset.onlyF=fmt(r.fx,6);
  svg.dataset.sum=fmt(r.sum,6);svg.dataset.zero=String(r.zeroUpdate);
  svg.dataset.complete=String(complete);svg.dataset.shape=String(r.shape);
  svg.dataset.progress=p.toFixed(3);
  svg.dataset.arrival0=arrive0.toFixed(3);svg.dataset.arrival1=arrive1.toFixed(3);
  svg.dataset.merge0=merge0.toFixed(3);svg.dataset.merge1=merge1.toFixed(3);
  svg.dataset.mergeProgress=Math.min(merge0,merge1).toFixed(3);
  svg.dataset.compare=compare.toFixed(3);
  svg.dataset.resultState=complete?'complete':done[0]||done[1]?'merging':'pending';
  svg.dataset.y=complete?fmt(r.sum,6):'';
  svg.dataset.diff=fmt(r.x,6);
  svg.dataset.phase=p<.35?'base':p<1.05?'sub':p<1.5?'dim0-arrive':p<1.85?'dim0-merge':p<2.05?'dim1-arrive':p<2.4?'dim1-merge':p<2.9?'complete':'compare';
  svg.dataset.rulerStart=String(L.ruler0);svg.dataset.rulerEnd=String(L.ruler1);
  svg.dataset.done0=String(done[0]);svg.dataset.done1=String(done[1]);
 }
 return draw;
}

/* ------------------------------------------------------------- 04 SwiGLU */
/* Three positions share one small MLP (Wg/Wu/Wd). The focus column is drawn
 * as two visible roles: the gate lane turns x into a continuous coefficient a,
 * while the value lane supplies the feature u that a scales. The lower ruler
 * then places u and z = a ⊙ u on one zero line, so a > 1, 0 < a < 1 and a < 0
 * become length/reversal changes rather than another formula box. Wd = I only
 * keeps the hand check readable. */
function swigluBoard(svg,H){
 const h=header(svg,'MLP / SwiGLU：同一套权重，怎样逐位置连续调制特征？','Attention 已把上下文写进 p1–p3；三个位置共用 Wg/Wu/Wd，却各用自己的 x 独立计算。','用途：把已有特征乘上当前输入决定的连续系数，再回投；不跨位置，不选专家。');
 const posData=[{id:'p1',x:[1,2]},{id:'p2',x:[2,1]},{id:'p3',x:[0,2]}];
 const cards=posData.map((c,i)=>{
  const g=mk('g',{},svg);
  const box=mk('rect',{rx:7},g);
  const name=txt(g,'att-q',10,18,c.id);
  const vec=txt(g,'att-num',10,36,'');
  const gate=txt(g,'att-score',10,50,'');
  const tag=txt(g,'att-score',0,18,'');
  hit(g,`聚焦 ${c.id}，查看这个位置的共享权重与连续系数`,()=>changeCondition(svg,focus,i,v=>{focus=v;}));
  return {g,box,name,vec,gate,tag};
 });
 const checkBtn=btn(svg,0,0,150,26,'显示手算核对',()=>{advanced=!advanced;pause(svg);draw(lastP,lastM);});
 const topNote=txt(svg,'att-dim',0,0,'三个位置共用 Wg/Wu/Wd；每列用本列 x 计算，列间没有箭头。');
 const base=mk('g',{},svg),check=mk('g',{},svg);

 /* One focus identity, reused by the graph title, the input chip and the data
  * contract. Every visible label reads from posData[focus], never from p1. */
 const graphPanel=rect(base,'att-panel',0,0,0,0);
 const graphTitle=txt(base,'att-strong',0,0,'');
 graphTitle.setAttribute('data-role','graph-title');
 const focusChip=rect(base,'cmp-focus-chip',0,0,0,0,8);
 const focusName=txt(base,'att-q',0,0,'');
 focusName.setAttribute('data-role','focus-label');
 const focusVec=txt(base,'att-num',0,0,'');
 focusVec.setAttribute('data-role','focus-input');

 /* Role 1: the gate lane produces a from x. */
 const gateTitle=txt(base,'att-strong',0,0,'① 门支路 · 生成输入相关调节系数 a');
 const gBox=rect(base,'cmp-branch-box is-gate',0,0,0,0,8);
 const gTitle=txt(base,'att-strong',0,0,'g = xWg');
 const gVal=txt(base,'att-num',0,0,'');
 const actBox=rect(base,'cmp-branch-box is-act',0,0,0,0,8);
 const actTitle=txt(base,'att-strong',0,0,'SiLU → a');
 const actVal=txt(base,'att-score',0,0,'');
 const gateTag=txt(base,'cmp-role-tag is-gate',0,0,'产生系数');

 /* Role 2: the value lane supplies u, the feature that will be modulated. */
 const valueTitle=txt(base,'att-strong',0,0,'② 值支路 · 提供被调节特征 u');
 const uBox=rect(base,'cmp-branch-box is-value',0,0,0,0,8);
 const uTitle=txt(base,'att-strong',0,0,'u = xWu');
 const uVal=txt(base,'att-num',0,0,'');
 const valueTag=txt(base,'cmp-role-tag is-value',0,0,'提供特征');

 /* A compact sign key: the three possible effects stay visible while the
  * lower ruler shows the selected values. */
 const legendTitle=txt(base,'att-dim',0,0,'a 的符号 → 特征变化');
 const legendItems=[
  {sym:'a > 1',desc:'同向变长',cls:'is-grow'},
  {sym:'0 < a < 1',desc:'同向缩短',cls:'is-shrink'},
  {sym:'a < 0',desc:'反向',cls:'is-negative'}
 ].map(item=>{
  item.sym=txt(base,'cmp-role-legend',0,0,item.sym);
  item.desc=txt(base,'cmp-role-legend',0,0,item.desc);
  return item;
 });

 /* The effect stage: u and z share one signed ruler. z is drawn from zero at
  * the same scale as u, and a horizontal arrow runs from the u tip to the z
  * tip, so lengthening, shortening and reversal are geometric. */
 const effectTitle=txt(base,'att-strong',0,0,'③ a 逐元素作用在 u 上：同尺看变长 / 缩短 / 反向');
 const effectAxis=mk('line',{class:'cmp-effect-axis'},base);
 const effectTicks=[-3,0,3,6].map(v=>{
  const line=mk('line',{class:'cmp-scale-tick'},base);
  const label=txt(base,'cmp-axis-text',0,0,String(v));
  label.setAttribute('text-anchor','middle');
  return {v,line,label};
 });
 const effectZero=mk('line',{class:'cmp-zero-line'},base);
 const effectRows=[0,1].map(i=>{
  const head=txt(base,'att-strong',0,0,'');
  const dir=txt(base,'cmp-effect-dir',0,0,'');
  const uTrack=rect(base,'cmp-track',0,0,0,0,4);
  const uFill=rect(base,'cmp-u-fill',0,0,0,0,4);
  const zTrack=rect(base,'cmp-track',0,0,0,0,4);
  const zFill=rect(base,'cmp-z-fill',0,0,0,0,4);
  const uLabel=txt(base,'att-dim',0,0,`u${i}`);
  const zLabel=txt(base,'att-dim',0,0,`z${i}`);
  const uVal=txt(base,'cmp-bar-value',0,0,'');
  const zVal=txt(base,'cmp-bar-value',0,0,'');
  const arrow=mk('path',{class:'cmp-effect-arrow',fill:'none'},base);
  uLabel.setAttribute('text-anchor','end');
  zLabel.setAttribute('text-anchor','end');
  return {head,dir,uTrack,uFill,zTrack,zFill,uLabel,zLabel,uVal,zVal,arrow};
 });

 /* Down projection stays a small output rail, not the main picture. */
 const outSep=mk('line',{class:'cmp-role-sep'},base);
 const zBox=rect(base,'cmp-branch-box is-out',0,0,0,0,8);
 const zTitle=txt(base,'att-strong',0,0,'z = a ⊙ u');
 const zResult=txt(base,'att-score',0,0,'等待相乘');
 const wdBox=rect(base,'cmp-down-box',0,0,0,0,8);
 const wdTitle=txt(base,'att-strong',0,0,'Wd 下投影');
 const wdSub=txt(base,'att-dim',0,0,'y = z · Wd');
 const yVal=txt(base,'att-num',0,0,'');

 /* Optional hand check: explicit matrices and the same p1/p2/p3 example. */
 const checkPanel=rect(check,'att-panel',0,0,0,0);
 const checkTitle=txt(check,'att-strong',0,0,'手算核对：显式教学矩阵与逐项乘法');
 const checkSub=txt(check,'att-dim',0,0,'只在点开时显示；默认图不铺矩阵和逐项乘法。');
 const checkLines=Array.from({length:10},()=>txt(check,'att-num',0,0,''));
 const checkNote1=txt(check,'att-dim',0,0,'g0=x0+x1，g1=x0−x1；u0=x1，u1=x0+x1。');
 const checkNote2=txt(check,'att-dim',0,0,'两条支路只是输入维度的线性组合，不绑定人类语义。');
 const checkNote3=txt(check,'att-dim',0,0,'Wd = I 只为看清乘积；真实 FFN 会把中间维回投模型宽度。');
 const foot=footer(svg,'',false,560);
 const arrowId=marker(svg);
 const wires={
  xgate:mk('path',{class:'cmp-flow',fill:'none'},base),
  xvalue:mk('path',{class:'cmp-flow',fill:'none'},base),
  gact:mk('path',{class:'cmp-flow',fill:'none'},base),
  zwd:mk('path',{class:'cmp-flow',fill:'none'},base)
 };
 let focus=0,advanced=false,lastP=0,lastM=false;

 function flow(key,d,progress){
  const e=wires[key],on=clamp(progress);
  e.setAttribute('d',d);
  e.style.strokeDasharray='1';e.style.strokeDashoffset=String(1-on);
  e.style.opacity=on>.001?'1':'0';
  if(on>=.999)e.setAttribute('marker-end',`url(#${arrowId})`);else e.removeAttribute('marker-end');
 }

 function draw(p,m){
  lastP=p;lastM=m;const W=m?360:720,HT=m?700:560;
  placeHeader(h,m,W);
  size(h.t1,m?13.5:18);size(h.t2,m?9:11.5);size(h.t3,m?8.4:10.5);
  setText(h.t1,m?'MLP / SwiGLU：逐位置连续调制':'MLP / SwiGLU：同一套权重，怎样逐位置连续调制特征？');
  setText(h.t2,m?'三个位置共用 Wg/Wu/Wd；各用自己的 x 独立计算。':'Attention 已把上下文写进 p1–p3；三个位置共用 Wg/Wu/Wd，却各用自己的 x 独立计算。');
  setText(h.t3,m?'用途：已有特征乘连续系数；不跨位置。':'用途：把已有特征乘上当前输入决定的连续系数，再回投；不跨位置，不选专家。');
  if(m)placeBtn(checkBtn,12,84,160,26,advanced?'收起手算核对':'显示手算核对',advanced);
  else placeBtn(checkBtn,552,86,150,26,advanced?'收起手算核对':'显示手算核对',advanced);
  checkBtn.setAttribute('aria-expanded',advanced?'true':'false');

  const id=posData[focus].id;
  const ex=M.swigluExample(posData[focus].x);
  const r=M.swigluPurpose(posData[focus].x);
  const reveal=phase(p,.2,.9),project=phase(p,.8,1.7),gate=phase(p,1.7,2.6);
  const multiply=phase(p,2.6,4.0),down=phase(p,4.0,4.9);
  const multiplied=multiply>=.999,settled=down>=.999;

  cards.forEach((c,i)=>{
   const rr=M.swigluPurpose(posData[i].x);
   const w=m?106:216,x=m?10+i*116:18+i*228,y=m?116:118,hh=m?56:54;
   c.g.setAttribute('transform',`translate(${x} ${y})`);
   put(c.box,0,0,w,hh);
   setText(c.name,posData[i].id);
   setText(c.vec,`x = ${fmt(posData[i].x,1)}`);
   setText(c.gate,m?`a=${fmt(rr.coefficients.map(v=>v.value),2)}`:`a = SiLU(g) = ${fmt(rr.coefficients.map(v=>v.value),2)}`);
   setText(c.tag,i===focus?'焦点':'同权重');
   c.tag.setAttribute('text-anchor','end');
   if(m){
    place(c.name,8,17);place(c.vec,8,34);place(c.gate,8,48);place(c.tag,w-8,17);
    size(c.name,9.4);size(c.vec,9);size(c.gate,7.6);size(c.tag,7.4);
   }else{
    place(c.name,10,18);place(c.vec,10,36);place(c.gate,10,50);place(c.tag,w-10,18);
    size(c.name,10.5);size(c.vec,10.5);size(c.gate,8.4);size(c.tag,8.6);
   }
   c.box.setAttribute('class',i===focus?'cmp-poscard is-focus':'cmp-poscard');
   c.g.setAttribute('opacity',String(i===focus?1:(m?.74:.84)));
   pressed(c.g,i===focus);
  });
  setText(topNote,m?'同一套 Wg/Wu/Wd；每列用自己的 x 计算，列间没有箭头。':'三个位置共用 Wg/Wu/Wd；每列用本列 x 计算，列间没有箭头。');
  if(m){place(topNote,12,184);size(topNote,8.6);}else{place(topNote,18,106);size(topNote,10.2);}
  show(topNote,true);

  setText(graphTitle,`焦点 ${id} 的调制链`);
  setText(focusName,`焦点 ${id}`);
  setText(focusVec,`x = ${fmt(r.x,1)}`);
  focusChip.setAttribute('aria-label',`焦点 ${id}，输入 x = ${fmt(r.x,1)}`);

  const scaleMin=-3,scaleMax=6;
  const layout=m?{
   panel:[10,194,340,390],title:[20,212],chip:[20,220,320,34],name:[30,242],vec:[110,242],
   gateTitle:[20,268],gate:[20,276,146,42],gTitle:[30,294],gVal:[30,310],act:[174,276,166,42],actTitle:[184,294],actVal:[184,310],
   valueTitle:[20,332],u:[20,340,320,34],uTitle:[30,360],uVal:[128,360],
   legendTitle:[20,390],effectTitle:[20,416],axis:[94,424,212],rowY:[444,464,484,504],headY:[438,478],dirY:[456,496],arrowY:[458,498],
   barX:94,barW:212,zBox:[20,520,320,28],zTitle:[30,538],zResult:[130,538],wdBox:[20,550,320,30],wdTitle:[30,569],wdSub:[130,569],yVal:[130,569],
   foot:HT-12
  }:{
   panel:[18,182,684,352],title:[30,200],chip:[30,214,104,52],name:[40,234],vec:[40,254],
   gateTitle:[150,210],gate:[150,220,104,40],gTitle:[158,236],gVal:[158,252],act:[266,220,122,44],actTitle:[274,236],actVal:[274,254],
   valueTitle:[150,272],u:[150,282,200,36],uTitle:[158,298],uVal:[158,314],
   legendTitle:[540,218],effectTitle:[30,344],axis:[200,362,270],rowY:[398,428,462,492],headY:[402,466],dirY:[418,482],arrowY:[418,482],
   barX:200,barW:270,zBox:[530,378,156,56],zTitle:[540,396],zResult:[540,414],wdBox:[530,446,156,72],wdTitle:[540,464],wdSub:[540,480],yVal:[540,496],
   foot:HT-12
  };

  put(graphPanel,...layout.panel);
  place(graphTitle,...layout.title);size(graphTitle,m?9.6:11.2);
  put(focusChip,...layout.chip);place(focusName,...layout.name);size(focusName,m?9.2:10.4);
  place(focusVec,...layout.vec);size(focusVec,m?9.4:10.8);
  place(gateTitle,...layout.gateTitle);size(gateTitle,m?9.6:10.4);
  put(gBox,...layout.gate);place(gTitle,...layout.gTitle);size(gTitle,m?9.2:9.6);place(gVal,...layout.gVal);size(gVal,m?9.4:9.8);
  put(actBox,...layout.act);place(actTitle,...layout.actTitle);size(actTitle,m?9.2:9.8);place(actVal,...layout.actVal);size(actVal,m?8.7:9.2);
  place(valueTitle,...layout.valueTitle);size(valueTitle,m?9.6:10.4);
  put(uBox,...layout.u);place(uTitle,...layout.uTitle);size(uTitle,m?9.2:9.6);place(uVal,...layout.uVal);size(uVal,m?9.6:10.2);
  place(gateTag,396,240);size(gateTag,9.4);
  place(valueTag,360,304);size(valueTag,9.4);
  if(m){show(gateTag,false);show(valueTag,false);show(wdSub,false);}

  setText(gVal,`g = ${fmt(ex.g,0)}`);
  setText(actVal,`a = ${fmt(ex.silu,3)}`);
  setText(uVal,`u = ${fmt(ex.u,2)}`);
  gBox.setAttribute('class','cmp-branch-box is-gate'+(project>.05?' is-active':''));
  uBox.setAttribute('class','cmp-branch-box is-value'+(project>.05?' is-active':''));
  zBox.setAttribute('class','cmp-branch-box is-out'+(multiply>.05?' is-active':''));
  setText(gTitle,'g = xWg');
  setText(uTitle,'u = xWu');
  gateTag.textContent='产生系数';
  valueTag.textContent='提供特征';

  /* Gate lane and value lane use the same x, so the split starts at the chip. */
  if(m){
   flow('xgate','M40 254 V276',project);
   flow('xvalue','M40 254 V340',project);
   flow('gact','M166 297 H174',project);
  }else{
   flow('xgate','M134 240 H150',project);
   flow('xvalue','M134 252 H142 V300 H150',project);
   flow('gact','M254 240 H266',project);
  }

  /* Legend: fixed rows on PC, one sentence on the narrow board. */
  if(m){
   place(legendTitle,20,390);size(legendTitle,8.8);
   setText(legendTitle,'a>1 变长 · 0<a<1 缩短 · a<0 反向');
   legendItems.forEach(it=>{show(it.sym,false);show(it.desc,false);});
  }else{
   place(legendTitle,...layout.legendTitle);size(legendTitle,10.2);
   setText(legendTitle,'a 的符号 → 特征变化');
   legendItems.forEach((it,i)=>{
    const y=246+i*28;
    place(it.sym,540,y);size(it.sym,10);
    place(it.desc,it.cls==='is-shrink'?610:600,y);size(it.desc,10);
    it.sym.setAttribute('class','cmp-role-legend '+it.cls);
    it.desc.setAttribute('class','cmp-role-legend '+it.cls);
    show(it.sym,true);show(it.desc,true);
   });
  }

  /* The modulation ruler. Both dimensions use the same -3..6 scale and the
   * same zero line; z is the drawn result, u is the input feature. */
  place(effectTitle,...layout.effectTitle);size(effectTitle,m?9.6:11.2);
  setText(effectTitle,m?'③ 同尺：a 把 u 变成 z':'③ a 逐元素作用在 u 上：同尺看变长 / 缩短 / 反向');
  const barX=layout.barX,barW=layout.barW,axisY=layout.axis[1],axisX=layout.axis[0],axisW=layout.axis[2];
  const xOf=v=>barX+(v-scaleMin)/(scaleMax-scaleMin)*barW;
  const zeroX=xOf(0);
  const [uY0,zY0,uY1,zY1]=layout.rowY;
  effectAxis.setAttribute('x1',axisX);effectAxis.setAttribute('x2',axisX+axisW);
  effectAxis.setAttribute('y1',axisY);effectAxis.setAttribute('y2',axisY);
  effectTicks.forEach(t=>{
   const x=xOf(t.v);
   t.line.setAttribute('x1',x);t.line.setAttribute('x2',x);
   t.line.setAttribute('y1',axisY-5);t.line.setAttribute('y2',axisY+5);
   place(t.label,x,axisY+15);size(t.label,m?7.6:8.4);
  });
  effectZero.setAttribute('x1',zeroX);effectZero.setAttribute('x2',zeroX);
  effectZero.setAttribute('y1',axisY+(m?18:20));effectZero.setAttribute('y2',m?512:508);

  effectRows.forEach((row,i)=>{
   const a=ex.silu[i],u=ex.u[i],z=ex.z[i];
   const uY=i?uY1:uY0,zY=i?zY1:zY0;
   const uX=xOf(u),zX=xOf(z),zNow=zeroX+(zX-zeroX)*multiply;
   const dir=z<0&&u>0||z>0&&u<0?'反向':Math.abs(z)>Math.abs(u)?'同向变长':'同向缩短';
   place(row.head,30,layout.headY[i]);size(row.head,m?10.4:12.6);
   setText(row.head,`a${i} = ${num(a,3)}`);
   row.head.setAttribute('class',a<0?'att-mask':'att-score');
   place(row.dir,30,layout.dirY[i]);size(row.dir,m?8.4:9.6);
   setText(row.dir,`${dir} · ×${num(a,2)}`);
   row.dir.setAttribute('class','cmp-effect-dir '+(dir==='反向'?'is-negative':dir==='同向变长'?'is-grow':'is-shrink'));

   put(row.uTrack,barX,uY,barW,m?9:11);
   put(row.uFill,Math.min(zeroX,uX),uY,Math.abs(uX-zeroX),m?9:11);
   put(row.zTrack,barX,zY,barW,m?10:13);
   put(row.zFill,Math.min(zeroX,zNow),zY,Math.abs(zNow-zeroX),m?10:13);
   row.uFill.setAttribute('class','cmp-u-fill'+(u<0?' is-negative':''));
   row.zFill.setAttribute('class','cmp-z-fill'+(z<0?' is-negative':''));
   place(row.uLabel,barX-8,uY+(m?8:9));size(row.uLabel,m?8.2:9);
   place(row.zLabel,barX-8,zY+(m?9:10));size(row.zLabel,m?8.2:9);
   row.uVal.setAttribute('text-anchor',uX>=zeroX?'start':'end');
   place(row.uVal,uX>=zeroX?uX+5:uX-5,uY+(m?8:9));
   setText(row.uVal,`u${i} = ${num(u,2)}`);
   size(row.uVal,m?8.2:9.2);
   row.zVal.setAttribute('text-anchor',zX>=zeroX?'start':'end');
   place(row.zVal,zX>=zeroX?zX+5:zX-5,zY+(m?9:10));
   setText(row.zVal,multiplied?`z${i} = ${num(z,3)}`:'');
   size(row.zVal,m?8.2:9.2);
   row.arrow.setAttribute('d',`M${uX} ${layout.arrowY[i]} H${zNow}`);
   row.arrow.setAttribute('class','cmp-effect-arrow '+(z<0?'is-negative':'is-positive'));
   show(row.arrow,multiply>.3&&Math.abs(zNow-uX)>3);
  });

  /* Output rail: product and down projection stay readable without becoming
   * the dominant picture. */
  put(zBox,...layout.zBox);place(zTitle,...layout.zTitle);size(zTitle,m?9.4:9.8);
  place(zResult,...layout.zResult);size(zResult,m?9:9.4);
  zResult.setAttribute('text-anchor','start');
  if(multiplied)setText(zResult,fmt(ex.z,3));
  else if(multiply>.001)setText(zResult,`相乘 ${pct(multiply)}`);
  else setText(zResult,'等待相乘');
  zResult.setAttribute('class',ex.z.some(v=>v<0)?'att-mask':'att-score');
  put(wdBox,...layout.wdBox);place(wdTitle,...layout.wdTitle);size(wdTitle,m?9.2:9.8);
  place(wdSub,...layout.wdSub);size(wdSub,m?8.4:8.8);
  show(wdSub,!m);
  place(yVal,...layout.yVal);size(yVal,m?9:9.4);
  if(settled)setText(yVal,`y = ${fmt(ex.y,3)}`);
  else if(down>.001)setText(yVal,`Wd ${pct(down)}`);
  else setText(yVal,'等待 Wd');
  outSep.setAttribute('x1',m?20:520);outSep.setAttribute('x2',m?340:520);
  outSep.setAttribute('y1',m?516:356);outSep.setAttribute('y2',m?516:524);
  if(m){
   flow('zwd','M180 548 V550',down);
  }else{
   flow('zwd','M608 434 V446',down);
  }

  if(!m){
   legendItems.forEach(it=>{show(it.sym,gate>.01);show(it.desc,gate>.01);});
  }

  /* Stage reveal follows the mechanism, not the layout. */
  const at=(e,prog)=>{show(e,prog>.01);if(prog>.01)fade(e,prog);};
  [focusChip,focusName,focusVec,graphPanel,graphTitle].forEach(e=>at(e,reveal));
  [effectTitle,effectAxis,effectZero].forEach(e=>at(e,reveal));
  effectTicks.forEach(t=>{at(t.line,reveal);at(t.label,reveal);});
  [gateTitle,gBox,gTitle,gVal].forEach(e=>at(e,project));
  [actBox,actTitle,actVal].forEach(e=>at(e,gate));
  if(!m)at(gateTag,gate);
  [valueTitle,uBox,uTitle,uVal].forEach(e=>at(e,project));
  if(!m)at(valueTag,project);
  [legendTitle].forEach(e=>at(e,gate));
  if(!m)legendItems.forEach(it=>{at(it.sym,gate);at(it.desc,gate);});
  effectRows.forEach((row,i)=>{
   at(row.head,gate);at(row.dir,multiply);
   at(row.uTrack,project);at(row.uFill,project);at(row.uLabel,project);at(row.uVal,project);
   at(row.zTrack,multiply);at(row.zFill,multiply);at(row.zLabel,multiply);at(row.zVal,multiply);
   at(row.arrow,multiply);
  });
  [zBox,zTitle,zResult].forEach(e=>at(e,multiply));
  [outSep,wdBox,wdTitle,yVal].forEach(e=>at(e,down));
  if(!m)at(wdSub,down);
  cards.forEach(c=>at(c.gate,gate));
  show(base,!advanced&&reveal>.01);
  if(!advanced)fade(base,reveal);
  show(check,advanced);

  if(m){
   put(checkPanel,12,190,336,490);
   place(checkTitle,24,212);size(checkTitle,10.2);place(checkSub,24,230);size(checkSub,7.8);
   checkLines.forEach((t,i)=>{place(t,24,254+i*24);size(t,8.4);});
   place(checkNote1,24,506);size(checkNote1,7.6);
   place(checkNote2,24,522);size(checkNote2,7.6);
   place(checkNote3,24,538);size(checkNote3,7.4);
  }else{
   put(checkPanel,18,182,684,344);
   place(checkTitle,30,204);size(checkTitle,11.5);place(checkSub,30,222);size(checkSub,8.6);
   [0,1,2,3,4,5].forEach(i=>{place(checkLines[i],36,252+i*26);size(checkLines[i],9.6);});
   [6,7,8,9].forEach((n,i)=>{place(checkLines[n],366,252+i*26);size(checkLines[n],9.6);});
   place(checkNote1,36,460);size(checkNote1,8.6);
   place(checkNote2,36,478);size(checkNote2,8.6);
   place(checkNote3,36,496);size(checkNote3,8.6);
  }
  const mat=v=>'[['+v.map(row=>row.join(', ')).join('], [')+']]';
  setText(checkLines[0],`Wg = ${mat(ex.Wg)}`);
  setText(checkLines[1],`Wu = ${mat(ex.Wu)}`);
  setText(checkLines[2],'Wd = I（2×2 教学例）');
  setText(checkLines[3],`x = ${fmt(ex.x,1)}`);
  setText(checkLines[4],`g = xWg = ${fmt(ex.g,0)}`);
  setText(checkLines[5],`u = xWu = ${fmt(ex.u,2)}`);
  setText(checkLines[6],`a = SiLU(g) = ${fmt(ex.silu,4)}`);
  setText(checkLines[7],`z0 = a0·u0 = ${num(ex.silu[0],4)}×${num(ex.u[0],2)} = ${num(ex.z[0],4)}`);
  setText(checkLines[8],`z1 = a1·u1 = ${num(ex.silu[1],4)}×${num(ex.u[1],2)} = ${num(ex.z[1],4)}`);
  setText(checkLines[9],`y = zWd = ${fmt(ex.y,4)}`);
  pressed(checkBtn,advanced);
  setText(foot,m?'三个位置权重共享、各自计算；a 连续可负/可>1；列间无箭头。':'三个位置权重共享、各自计算；a 连续可负/可>1；z=a⊙u 后由 Wd 回投。');
  place(foot,m?12:18,HT-12);size(foot,m?9:10);show(foot,true);

  svg.dataset.component='swiglu';
  svg.dataset.focus=id;
  svg.dataset.focusLabel=`焦点 ${id}`;
  svg.dataset.focusInput=fmt(r.x,1);
  svg.dataset.condition=String(focus);
  svg.dataset.shared='Wg,Wu,Wd';
  svg.dataset.branches='g,u';
  svg.dataset.positionCount='3';
  svg.dataset.crossPosition='false';
  svg.dataset.g=fmt(ex.g,6);
  svg.dataset.u=fmt(ex.u,6);
  svg.dataset.silu=fmt(ex.silu,6);
  svg.dataset.z=fmt(ex.z,6);
  svg.dataset.y=settled?fmt(ex.y,6):'';
  svg.dataset.a0=num(ex.silu[0],6);svg.dataset.a1=num(ex.silu[1],6);
  svg.dataset.negative=String(ex.gateCanBeNegative);
  svg.dataset.exceedsOne=String(ex.gateCanExceedOne);
  svg.dataset.gateMin=num(Math.min(...ex.silu),4);
  svg.dataset.gateMax=num(Math.max(...ex.silu),4);
  svg.dataset.gateProgress=gate.toFixed(3);
  svg.dataset.multiplyProgress=multiply.toFixed(3);
  svg.dataset.applyProgress=multiply.toFixed(3);
  svg.dataset.downProgress=down.toFixed(3);
  svg.dataset.resultState=settled?'complete':multiplied?'multiplied':'multiplying';
  svg.dataset.phase=p<.8?'positions':p<1.7?'project':p<2.6?'silu':p<4?'multiply':p<4.9?'down':'complete';
  svg.dataset.detail=advanced?'shown':'hidden';
  svg.dataset.scaleMin=String(scaleMin);
  svg.dataset.scaleMax=String(scaleMax);
  svg.dataset.zeroX=num(zeroX,3);
 }
 return draw;
}

/* ------------------------------------------------------- 05 MHA/GQA/MQA */
/* The three topologies are always on screen together. The only condition is
 * history length; Q stays four independent queries in every column. Each
 * K/V group owns its own history cells, so "shared" means fewer copies, not a
 * lossless merge of an old MHA cache. */
function headsBoard(svg,H){
 const h=header(svg,'MHA / GQA / MQA：Q 头都保留 4 个，只改 K/V 怎么共享',
  '三种方案并排：Q0–Q3 各自查询，独立 K/V 组数 4/2/1；历史先写满，再读取。',
  '不是把旧 MHA 缓存无损合并；连线只表示逻辑上读哪一组。');
 const histButtons=[3,6].map(n=>btn(svg,0,0,88,26,`历史 ${n}`,()=>changeCondition(svg,hist,n,v=>{hist=v;})));
 const note1=txt(svg,'att-dim',0,0,'并排对照：Q0–Q3 始终各自查询；共享只减少 K/V 组数，不合并 Q。');
 const note2=txt(svg,'att-dim',0,0,'每个 KV 组先按历史长度写满格子，写满后才让 Q 读取所属组。');
 const topologyPanel=rect(svg,'att-panel',0,0,0,0);
 const topologyTitle=txt(svg,'att-strong',0,0,'三种独立拓扑并排：谁共享哪一组 K/V');
 const topologySub=txt(svg,'att-dim',0,0,'同一层、同一个历史长度；每组各自保存一份 K/V 历史。');
 const modes=[
  {name:'MHA',kv:4,sub:'4 个 Q 各自查询；Qi 读 KVi'},
  {name:'GQA',kv:2,sub:'4 个 Q 各自查询；Q0/Q1→KV0，Q2/Q3→KV1'},
  {name:'MQA',kv:1,sub:'4 个 Q 各自查询；四个 Q 都读 KV0'}
 ];
 const columns=modes.map(def=>{
  const colBg=rect(svg,'cmp-heads-col',0,0,0,0,8);
  const title=txt(svg,'att-strong',0,0,`${def.name} · ${def.kv} 组 K/V`);
  const sub=txt(svg,'att-dim',0,0,def.sub);
  const qCards=Array.from({length:4},(_,i)=>{
   const g=mk('g',{class:'cmp-headq'},svg);
   const box=rect(g,'cmp-qbox',0,0,0,0,5);
   const name=txt(g,'att-q',7,15,`Q${i}`);
   return {g,box,name};
  });
  const kvCards=Array.from({length:4},(_,i)=>{
   const g=mk('g',{class:'cmp-headkv'},svg);
   const box=rect(g,'cmp-kvbox',0,0,0,0,5);
   const name=txt(g,'att-q',7,14,`KV${i}`);
   const meta=txt(g,'att-dim',7,27,'');
   const ticks=Array.from({length:6},()=>mk('rect',{rx:1,class:'cmp-kvtick'},g));
   const count=txt(g,'att-dim',0,0,'');
   return {g,box,name,meta,ticks,count};
  });
  const lines=Array.from({length:4},()=>mk('path',{class:'cmp-read',fill:'none'},svg));
  const pulses=Array.from({length:4},()=>mk('circle',{class:'cmp-read-pulse',r:3},svg));
  const status=txt(svg,'att-dim',0,0,'');
  return {def,colBg,title,sub,qCards,kvCards,lines,pulses,status};
 });
 const topologyNote=txt(svg,'att-dim',0,0,'并排只比较拓扑与元素数；不把旧 MHA 缓存无损合并，也不保证输出或质量相同。');
 const mobileStatus=txt(svg,'att-score',0,0,'');
 const cachePanel=rect(svg,'att-panel',0,0,0,0);
 const cacheTitle=txt(svg,'att-strong',0,0,'公平缓存对照：同一层、每头 dim=2、历史 3 与 6');
 const cacheFormula=txt(svg,'att-dim',0,0,'K+V 元素 = 2 × 历史 × 2 × KV 组数；柱长同一刻度');
 const cacheHead3=txt(svg,'att-dim',0,0,'历史 3');
 const cacheHead6=txt(svg,'att-dim',0,0,'历史 6');
 const cacheRows=[4,2,1].map(kv=>{
  const label=txt(svg,'att-strong',0,0,`${kv===4?'MHA':kv===2?'GQA':'MQA'} ${kv}组`);
  const bar3=rect(svg,'cmp-cache-bar',0,0,0,0,4);
  const bar6=rect(svg,'cmp-cache-bar',0,0,0,0,4);
  const count3=txt(svg,'att-num',0,0,'');
  const count6=txt(svg,'att-num',0,0,'');
  return {kv,label,bar3,bar6,count3,count6,
   h3:M.kvHeads(kv,{positions:3}).scalars,h6:M.kvHeads(kv,{positions:6}).scalars};
 });
 const cacheNotes=[
  txt(svg,'att-dim',0,0,'共享：K/V 组 4→2→1'),
  txt(svg,'att-dim',0,0,'每位置 K+V：16→8→4 个元素'),
  txt(svg,'att-dim',0,0,'历史 3→6：各组翻倍'),
  txt(svg,'att-w',0,0,'Q 头仍 4 个；不保证输出相同')
 ];
 const cacheMobileNote=txt(svg,'att-dim',0,0,'历史 3→6 翻倍；Q 仍 4 个；不把旧缓存无损合并。');
 const foot=footer(svg,'K+V 元素 = 2 × 历史 × 每头维度 × KV 组数；只是该层保存量，不是总显存、带宽或加速比。',false,560);
 let hist=3,lastP=0,lastM=false;
 const pulseColors=['#62d8e9','#bda9f7','#7fe0c3','#f2b45c'];
 function draw(p,m){
  lastP=p;lastM=m;const W=m?360:720,HT=m?700:560;
  placeHeader(h,m,W);
  const write=phase(p,.08,.95),readOut=phase(p,1,1.8),readBack=phase(p,1.8,2.45);
  const shown=Math.max(0,Math.min(hist,Math.ceil(hist*write-1e-9)));
  const ready=shown===hist;
  const state=!ready?`写入 ${shown}/${hist} · 未就绪`:
   readOut<=.01?`就位 ${hist}/${hist} · 待读取`:
   readOut<1?`Q→KV 读取 ${Math.round(readOut*100)}%`:
   readBack<1?`KV→Q 返回 ${Math.round(readBack*100)}%`:
   `完成 · 历史 ${hist}/${hist}`;
  if(m){
   histButtons.forEach((b,i)=>placeBtn(b,14+i*98,84,88,26,`历史 ${[3,6][i]}`,hist===[3,6][i]));
   place(note1,14,124);size(note1,8.4);place(note2,14,138);size(note2,7.9);
  }else{
   histButtons.forEach((b,i)=>placeBtn(b,18+i*98,88,88,26,`历史 ${[3,6][i]}`,hist===[3,6][i]));
   place(note1,18,124);size(note1,9.6);place(note2,18,138);size(note2,8.8);
  }
  if(m){put(topologyPanel,12,146,336,366);place(topologyTitle,24,164);size(topologyTitle,9.8);place(topologySub,24,179);size(topologySub,7.7);}
  else{put(topologyPanel,18,144,684,250);place(topologyTitle,30,160);size(topologyTitle,11.5);place(topologySub,30,176);size(topologySub,8.6);}
  const colW=m?324:208,colX0=m?20:30;
  columns.forEach((col,ci)=>{
   const colX=m?colX0:colX0+ci*226;
   const blockY=m?188+ci*104:0;
   const qY=m?blockY+33:218,qH=m?20:22;
   const kvY=m?blockY+65:276,kvH=m?31:72;
   const qW=m?(colW-18)/4:(colW-15)/4,qGap=m?6:5;
   const kvW=(colW-(col.def.kv-1)*6)/col.def.kv;
   put(col.colBg,colX-(m?4:6),m?blockY:184,colW+(m?8:12),m?100:176);
   place(col.title,colX,m?blockY+14:196);size(col.title,m?9.4:10.4);
   place(col.sub,colX,m?blockY+27:210);size(col.sub,m?7:7.2);
   col.qCards.forEach((card,i)=>{
    card.g.setAttribute('transform',`translate(${colX+i*(qW+qGap)} ${qY})`);
    put(card.box,0,0,qW,qH);place(card.name,7,qH-7);size(card.name,m?8:8.6);
    show(card.g,true);fade(card.g,.55+.45*phase(p,0,.5));
   });
   col.kvCards.forEach((card,g)=>{
    const visible=g<col.def.kv;show(card.g,visible);if(!visible)return;
    card.g.setAttribute('transform',`translate(${colX+g*(kvW+6)} ${kvY})`);
    put(card.box,0,0,kvW,kvH);
    fade(card.g,.5+.5*Math.min(1,write*2));
    setText(card.name,`KV${g}`);
    const meta=col.def.kv===4?`仅 Q${g}`:col.def.kv===2?`Q${g*2}、Q${g*2+1} 共用`:'Q0–Q3 共用';
    setText(card.meta,meta);
    if(m){
     place(card.name,6,11);size(card.name,7.6);
     place(card.meta,32,11);size(card.meta,6.6);
    }else{
     place(card.name,7,14);size(card.name,col.def.kv===4?8:8.8);
     place(card.meta,7,28);size(card.meta,col.def.kv===4?6.8:7.4);
    }
    const tw=(kvW-12)/hist;
    card.ticks.forEach((tick,j)=>{
     const on=j<shown;show(tick,on);if(!on)return;
     put(tick,6+j*tw,m?16:42,Math.max(2,tw-1),m?5:7);
     tick.setAttribute('class',j%2?'cmp-kvtick is-alt':'cmp-kvtick');
     fade(tick,j===shown-1?Math.max(.2,Math.min(1,(write*hist-j)*3)):1);
    });
    place(card.count,kvW-6,m?28:67);card.count.setAttribute('text-anchor','end');size(card.count,m?6.4:6.9);
    setText(card.count,`K/V ${shown}/${hist}`);
   });
   col.lines.forEach((line,i)=>{
    const qx=colX+i*(qW+qGap)+qW/2,qBottom=qY+qH;
    const target=Math.floor(i*col.def.kv/4);
    const tx=colX+target*(kvW+6)+kvW/2,ty=kvY;
    const lift=Math.max(6,Math.min(14,(ty-qBottom)*.45));
    line.setAttribute('d',`M${qx} ${qBottom} C${qx} ${qBottom+lift} ${tx} ${ty-lift} ${tx} ${ty}`);
    line.setAttribute('class',`cmp-read cmp-read-${i}`);
    line.setAttribute('pathLength','1');line.style.strokeDasharray='1';line.style.strokeDashoffset=String(1-readOut);
    show(line,readOut>.01);fade(line,.2+.8*readOut);
    const cubic=(a,b,c,d,t)=>{const u=1-t;return u*u*u*a+3*u*u*t*b+3*u*t*t*c+t*t*t*d;};
    const back=readBack>.001,t=back?1-readBack:readOut;
    const px=cubic(qx,qx,tx,tx,t),py=cubic(qBottom,qBottom+lift,ty-lift,ty,t);
    const pulse=col.pulses[i];
    pulse.setAttribute('cx',px);pulse.setAttribute('cy',py);
    pulse.setAttribute('class',`cmp-read-pulse cmp-pulse-${i}`);
    pulse.style.fill=pulseColors[i]||'#f2b45c';
    show(pulse,readOut>.02&&readBack<.98);fade(pulse,readOut>.02?1:0);
   });
   if(m){show(col.status,false);}
   else{
    place(col.status,colX+colW/2,364);col.status.setAttribute('text-anchor','middle');size(col.status,7.2);
    setText(col.status,`${col.def.name} ${state}`);
    col.status.setAttribute('class',!ready?'att-dim':readOut<1?'att-score':'att-w');
    show(col.status,true);
   }
  });
  if(m){
   place(mobileStatus,180,508);mobileStatus.setAttribute('text-anchor','middle');size(mobileStatus,8.2);
   setText(mobileStatus,state);mobileStatus.setAttribute('class',!ready?'att-dim':readOut<1?'att-score':'att-w');
   show(mobileStatus,true);show(topologyNote,false);
  }else{
   show(mobileStatus,false);
   place(topologyNote,30,382);size(topologyNote,7.6);show(topologyNote,true);
  }
  const cacheX=m?12:18,cacheY=m?520:402,cacheW=m?336:684,cacheH=m?122:126;
  put(cachePanel,cacheX,cacheY,cacheW,cacheH);
  place(cacheTitle,cacheX+12,cacheY+18);size(cacheTitle,m?9.4:11.2);
  setText(cacheTitle,m?'缓存对照：同一把尺看历史 3/6':'公平缓存对照：同一层、每头 dim=2、历史 3 与 6');
  place(cacheFormula,cacheX+12,cacheY+34);size(cacheFormula,m?7.2:8.4);
  setText(cacheFormula,m?'K+V=2×历史×2×组数；同一把尺':'K+V 元素 = 2 × 历史 × 2 × KV 组数；柱长同一刻度');
  const bar3X=m?72:92,maxW=m?64:135,bar6X=m?168:272;
  const count3X=m?154:258,count6X=m?246:436;
  const headY=cacheY+(m?46:52),rowY0=cacheY+(m?58:66),rowStep=m?16:21,barH=m?7:9;
  place(cacheHead3,bar3X+maxW/2,headY);cacheHead3.setAttribute('text-anchor','middle');size(cacheHead3,m?7.4:8.2);
  place(cacheHead6,bar6X+maxW/2,headY);cacheHead6.setAttribute('text-anchor','middle');size(cacheHead6,m?7.4:8.2);
  cacheRows.forEach((row,i)=>{
   const y=rowY0+i*rowStep;
   place(row.label,cacheX+12,y+barH);size(row.label,m?7.2:8.2);
   const w3=Math.max(2,maxW*row.h3/96),w6=Math.max(2,maxW*row.h6/96);
   put(row.bar3,bar3X,y,w3,barH,3);put(row.bar6,bar6X,y,w6,barH,3);
   row.bar3.setAttribute('fill','#62d8e9');row.bar6.setAttribute('fill','#f2b45c');
   row.bar3.setAttribute('opacity',hist===3?'.95':'.38');
   row.bar6.setAttribute('opacity',hist===6?'.95':'.38');
   place(row.count3,count3X,y+barH);row.count3.setAttribute('text-anchor','end');place(row.count6,count6X,y+barH);row.count6.setAttribute('text-anchor','end');
   size(row.count3,m?7.6:8.6);size(row.count6,m?7.6:8.6);
   setText(row.count3,String(row.h3));setText(row.count6,String(row.h6));
   row.count3.setAttribute('class',hist===3?'att-num':'att-dim');row.count6.setAttribute('class',hist===6?'att-num':'att-dim');
  });
  if(m){
   cacheNotes.forEach(n=>show(n,false));
   place(cacheMobileNote,cacheX+12,cacheY+112);size(cacheMobileNote,7);show(cacheMobileNote,true);
  }else{
   show(cacheMobileNote,false);
   cacheNotes.forEach((n,i)=>{place(n,cacheX+432,cacheY+56+i*18);size(n,7.4);show(n,true);});
  }
  setText(foot,m?'K+V 元素=2×历史×2×组数；只是该层保存量，不是带宽或总显存。':'K+V 元素 = 2 × 历史 × 每头维度 × KV 组数；只是该层保存量，不是总显存、带宽或加速比。');
  place(foot,m?12:18,HT-12);show(foot,true);
  const cur=[M.kvHeads(4,{positions:hist}).scalars,M.kvHeads(2,{positions:hist}).scalars,M.kvHeads(1,{positions:hist}).scalars];
  svg.dataset.component='heads';
  svg.dataset.layout='parallel';
  svg.dataset.mode='parallel';
  svg.dataset.modes='MHA,GQA,MQA';
  svg.dataset.qHeads='4';
  svg.dataset.kvHeads='4,2,1';
  svg.dataset.mapping='0,1,2,3;0,0,1,1;0,0,0,0';
  svg.dataset.history=String(hist);
  svg.dataset.filledPositions=String(shown);
  svg.dataset.historyReady=String(ready);
  svg.dataset.readOut=readOut.toFixed(3);svg.dataset.readBack=readBack.toFixed(3);
  svg.dataset.readProgress=readOut.toFixed(3);
  svg.dataset.readPhase=readBack>=1?'done':readBack>0?'return':readOut>0?'out':'wait';
  svg.dataset.scalars=cur.join(',');
  svg.dataset.scalars3='48,24,12';svg.dataset.scalars6='96,48,24';
  svg.dataset.cacheScale='96';
  svg.dataset.perHeadDim='2';
  svg.dataset.qIndependent='true';svg.dataset.qMerged='false';
  svg.dataset.mergedFromMHA='false';svg.dataset.losslessMerge='false';
  svg.dataset.outputQualityEqual='false';svg.dataset.bandwidthClaim='false';
  svg.dataset.readSource='own-group';
 }
 return draw;
}

/* --------------------------------------------------------- 06 Transformer Block */
/* One Pre-Norm layer drawn as a single live chain. Three position lanes run
 * from x through two sublayers; every stage shows the actual teaching vectors
 * and how the residual base is combined with the sublayer delta. Attention is
 * cross-position and causal; the MLP is per-position with shared parameters.
 * The optional no-bypass state is a deterministic counterfactual, clearly
 * labelled, not the default architecture. */
function blockBoard(svg,H){
 const h=header(svg,'Transformer Block：一层内，位置表示怎样被更新两次？',
  'Pre-Norm 顺序：Norm(x) → Attention 增量 ΔA → 加回 x 得 x1；再 Norm(x1) → MLP 增量 ΔM → 加回 x1 得 y。',
  'Attention 跨位置因果读取 · MLP 每位置共享参数 · RoPE 只进 Q/K · RMS 控尺度 · 输出仍是 3 个位置表示');
 let focus=2,ablation=false,lastP=0,lastM=false;
 const focusButtons=[0,1,2].map(i=>btn(svg,0,0,100,24,`聚焦 p${i+1}`,()=>changeCondition(svg,focus,i,v=>{focus=v;})));
 const ablationBtn=btn(svg,0,0,142,24,'断开旁路（对照）',()=>changeCondition(svg,ablation?1:0,ablation?0:1,v=>{ablation=!!v;}));
 const hint=txt(svg,'att-dim',0,0,'换焦点 / 断开旁路 → 回到 0 并暂停');
 const inputLabel=txt(svg,'att-strong',0,0,'');
 const inputCards=[0,1,2].map(i=>{
  const g=mk('g',{},svg),box=rect(g,'cmp-poscard',0,0,0,0,7),name=txt(g,'att-q',8,14,`p${i+1} · x`),val=txt(g,'att-num',8,32,''),tag=txt(g,'att-dim',0,14,'同层');
  return {g,box,name,val,tag};
 });
 const band=rect(svg,'cmp-mainband',0,0,0,0);
 const bandTitle=txt(svg,'att-strong',0,0,'');
 const bandSub=txt(svg,'att-dim',0,0,'');
 const normChips=[0,1,2].map(()=>{
  const g=mk('g',{},svg),box=rect(g,'cmp-normchip',0,0,0,0,4),t=txt(g,'att-dim',6,12,'');
  return {g,box,t};
 });
 const kNodes=[0,1,2].map(i=>{
  const g=mk('g',{},svg),box=mk('rect',{rx:4,class:'cmp-attn-node'},g),t=txt(g,'att-k',0,0,`K${i+1} ↻`);
  return {g,box,t};
 });
 const qNodes=[0,1,2].map(i=>{
  const g=mk('g',{},svg),box=mk('rect',{rx:4,class:'cmp-attn-node'},g),t=txt(g,'att-q',0,0,`Q${i+1} ↻`);
  return {g,box,t};
 });
 const beams=[0,1,2].map(()=>mk('path',{class:'cmp-beam',fill:'none'},svg));
 const beamPulses=[0,1,2].map(()=>mk('circle',{class:'cmp-read-pulse',r:2.7},svg));
 const readout=txt(svg,'att-dim',0,0,'');
 const x1Label=txt(svg,'att-strong',0,0,'');
 const colorLegend=txt(svg,'att-dim',0,0,'青=基底 · 橙/粉=增量 · 绿=结果');
 /* The two merge bars sit only 5.5 units apart, less than one text line.
  * Explain the row order once here instead of printing d0/d1 inside every card. */
 const mergeOrder=txt(svg,'att-dim',0,0,'条形行序 上→下：d0 / d1');
 const merge1Cards=[0,1,2].map(()=>makeMergeCard(svg));
 const mlpPanel=rect(svg,'att-panel',0,0,0,0);
 const mlpTitle=txt(svg,'att-strong',0,0,'');
 const mlpSub=txt(svg,'att-dim',0,0,'');
 const mlpNormChips=[0,1,2].map(()=>{
  const g=mk('g',{},svg),box=rect(g,'cmp-normchip',0,0,0,0,4),t=txt(g,'att-dim',6,12,'');
  return {g,box,t};
 });
 const sharedLine=mk('path',{class:'cmp-block-shared',fill:'none'},svg);
 const sharedText=txt(svg,'cmp-block-shared-text',0,0,'');
 const mlpBoxes=[0,1,2].map(i=>{
  const g=mk('g',{},svg),box=rect(g,'cmp-mlpbox',0,0,0,0,6);
  const gp=rect(g,'cmp-pill is-gate',0,0,0,0,3),gt=txt(g,'cmp-pill-text is-gate',0,0,'g=xWg→SiLU');
  const up=rect(g,'cmp-pill is-value',0,0,0,0,3),ut=txt(g,'cmp-pill-text is-value',0,0,'u=xWu');
  const row=txt(g,'cmp-mlp-row',0,0,'a ⊙ u → Wd');
  const val=txt(g,'cmp-mlp-delta',0,0,'');
  return {g,box,gp,gt,up,ut,row,val};
 });
 const merge2Label=txt(svg,'att-strong',0,0,'');
 const merge2Cards=[0,1,2].map(()=>makeMergeCard(svg));
 const outTitle=txt(svg,'att-strong',0,0,'');
 const outSub=txt(svg,'att-dim',0,0,'');
 const xBus=mk('path',{class:'cmp-block-skip is-x',fill:'none'},svg);
 const xStubTop=mk('path',{class:'cmp-block-stub is-x',fill:'none'},svg);
 const xStubBottom=mk('path',{class:'cmp-block-stub is-x',fill:'none'},svg);
 const x1Bus=mk('path',{class:'cmp-block-skip is-x1',fill:'none'},svg);
 const x1StubTop=mk('path',{class:'cmp-block-stub is-x1',fill:'none'},svg);
 const x1StubBottom=mk('path',{class:'cmp-block-stub is-x1',fill:'none'},svg);
 const xBusTag=txt(svg,'cmp-block-skip-tag is-x',0,0,'x 旁路');
 const x1BusTag=txt(svg,'cmp-block-skip-tag is-x1',0,0,'x1 旁路');
 const rail=rect(svg,'att-panel',0,0,0,0);
 const railTitle=txt(svg,'att-strong',0,0,'为什么分两步？');
 const why1=txt(svg,'att-dim',0,0,'Attention 要读别的历史位置；');
 const why2=txt(svg,'att-dim',0,0,'MLP 只加工本位置；');
 const why3=txt(svg,'att-dim',0,0,'两者都只写回增量，不换掉位置。');
 const scopeTitle=txt(svg,'att-strong',0,0,'对象边界');
 const scope1=txt(svg,'att-num',0,0,'一层输出 = 3 个位置表示');
 const notToken=rect(svg,'cmp-block-warn-chip',0,0,0,0,6);
 const notTokenText=txt(svg,'cmp-block-warn-text',0,0,'不是 1 个新 token');
 const next1=txt(svg,'att-dim',0,0,'下一 token：全部层之后');
 const next2=txt(svg,'att-dim',0,0,'输出头给分数，生成再选择');
 const object=txt(svg,'att-dim',0,0,'层号 L ≠ 位置 p ≠ token');
 const detailTitle=txt(svg,'att-dim',0,0,'单项详情 · 点击进入');
 const detailDefs=[
  {full:'RMSNorm · 控尺度',short:'RMS',concept:'rmsnorm'},
  {full:'RoPE · 位置进 Q/K',short:'RoPE',concept:'rope'},
  {full:'Attention · 跨位置',short:'Attn',href:'attention.html#mechanism-process'},
  {full:'残差 · 保留原表示',short:'残差',concept:'residual'},
  {full:'SwiGLU · 逐位置',short:'SwiGLU',concept:'swiglu'}
 ];
 const detailChips=detailDefs.map(d=>{
  const g=d.href?mk('a',{href:d.href,class:'cmp-hit',tabindex:'0','aria-label':`${d.full}：进入 Attention 课程`},svg):mk('g',{},svg);
  const box=rect(g,'cmp-block-detail',0,0,0,0,5),t=txt(g,'cmp-block-detail-text',8,15,d.full);
  if(!d.href)hit(g,`${d.full}：进入小演示`,()=>svg.dispatchEvent(new CustomEvent('component-focus',{bubbles:true,detail:{concept:d.concept,step:0}})));
  return {g,box,t,full:d.full,short:d.short};
 });
 const hint2=txt(svg,'att-dim',0,0,'换焦点 / 断开旁路 → 回到 0 并暂停');
 const teaching1=txt(svg,'att-dim',0,0,'二维教学例：单头、W_O=I、γ=[1,1]');
 const teaching2=txt(svg,'att-dim',0,0,'数值由固定公式算出');
 const teaching3=txt(svg,'att-dim',0,0,'不是模型观测或真实 hidden state');
 const ablationNote=txt(svg,'cmp-block-warn-text',0,0,'对照：无显式相加基底，下游重算');
 const foot=footer(svg,'',false,700);

 function vtext(v,d=2){return '['+v.map(n=>n<0?'−'+Math.abs(n).toFixed(d):n.toFixed(d)).join(', ')+']';}
 function shortVec(v){return '['+v.map(n=>Number.isInteger(n)?String(n):n.toFixed(2)).join(', ')+']';}
 function placeMergeCard(card,x,y,w,h,base,delta,result,showBase,grow,done,label,tagText,tagWarn,baseName){
  card.g.setAttribute('transform',`translate(${x} ${y})`);
  put(card.box,0,0,w,h);
  place(card.name,8,13);size(card.name,w<120?8.2:9.4);setText(card.name,label);
  place(card.tag,w-7,13);card.tag.setAttribute('text-anchor','end');size(card.tag,w<120?6.8:7.6);
  setText(card.tag,tagText);card.tag.setAttribute('class',tagWarn?'cmp-block-tag is-warn':'att-dim');
  place(card.eq,8,27);size(card.eq,w<120?6.8:7.6);
  place(card.res,8,41);size(card.res,w<120?8:9.2);
  const barX0=w<120?18:20,barW=w-barX0-7,rowY=w<120?46:47,rowGap=w<120?4.5:5.5,bh=w<120?2.4:2.8;
  const min=-.8,max=4;
  const vx=v=>barX0+Math.max(0,Math.min(1,(v-min)/(max-min)))*barW;
  const zero=vx(0);
  card.zero.setAttribute('x1',zero);card.zero.setAttribute('x2',zero);card.zero.setAttribute('y1',rowY-2);card.zero.setAttribute('y2',rowY+rowGap+bh+2);
  card.rows.forEach((r,i)=>{
   const b=base[i],v=result[i];
   const baseEnd=showBase?vx(b):zero;
   const end=baseEnd+(vx(v)-baseEnd)*grow;
   if(showBase){
    const w0=Math.abs(baseEnd-zero);
    r.base.setAttribute('x',Math.min(zero,baseEnd));r.base.setAttribute('y',rowY+i*rowGap);r.base.setAttribute('width',w0);r.base.setAttribute('height',bh);
    show(r.base,w0>=.5);
   }else show(r.base,false);
   const w1=Math.abs(end-baseEnd);
   r.delta.setAttribute('x',Math.min(baseEnd,end));r.delta.setAttribute('y',rowY+i*rowGap);r.delta.setAttribute('width',w1);r.delta.setAttribute('height',bh);
   r.delta.setAttribute('class','cmp-block-delta'+(v<b?' is-negative':''));
   show(r.delta,w1>=.5);
   r.dot.setAttribute('cx',end);r.dot.setAttribute('cy',rowY+i*rowGap+bh/2);
   r.dot.setAttribute('class','cmp-block-dot'+(done?' is-done':''));
   show(r.dot,grow>.02);
  });
  card.g.dataset.merge=label;card.g.dataset.base=shortVec(base);card.g.dataset.delta=shortVec(delta);
  card.g.dataset.result=done?shortVec(result):'';card.g.dataset.grow=grow.toFixed(3);card.g.dataset.done=String(done);
  if(done){
   setText(card.res,`= ${vtext(result)}`);
   card.res.setAttribute('class','cmp-block-result');
  }else{
   setText(card.res,grow>.02?'合并中…':'待合并');
   card.res.setAttribute('class','cmp-block-pending');
  }
  const deltaName=tagText.includes('ΔM')?'ΔM':'ΔA';
  setText(card.eq,showBase?`${baseName}${shortVec(base)} + ${deltaName}${shortVec(delta)}`:`仅 ${deltaName}${shortVec(delta)}`);
 }
 function makeMergeCard(parent){
  const g=mk('g',{class:'cmp-block-merge'},parent);
  const box=rect(g,'cmp-block-card',0,0,0,0,7);
  const name=txt(g,'att-num',8,13,'');
  const tag=txt(g,'att-dim',0,13,'');
  const eq=txt(g,'att-dim',8,27,'');
  const res=txt(g,'cmp-block-pending',8,41,'');
  const zero=mk('line',{class:'cmp-block-zero'},g);
  const rows=[0,1].map(()=>({base:mk('rect',{class:'cmp-block-base',rx:1},g),delta:mk('rect',{class:'cmp-block-delta',rx:1},g),dot:mk('circle',{class:'cmp-block-dot',r:2.4},g)}));
  return {g,box,name,tag,eq,res,zero,rows};
 }
 function draw(p,m){
  lastP=p;lastM=m;const W=m?360:720,HT=700;
  placeHeader(h,m,W);
  put(h.bg,0,0,W,700);
  if(m){
   h.t1.style.fontSize='15px';h.t2.style.fontSize='9.2px';h.t3.style.fontSize='8.2px';
   setText(h.t1,'Transformer Block：两次更新');
   setText(h.t2,'Norm→Attention→+x=x1；Norm→MLP→+x1=y');
   setText(h.t3,'Attention 跨位置 · MLP 逐位置');
  }else{
   h.t1.style.fontSize='';h.t2.style.fontSize='';h.t3.style.fontSize='';
   setText(h.t1,'Transformer Block：一层内，位置表示怎样被更新两次？');
   setText(h.t2,'Pre-Norm 顺序：Norm(x) → Attention 增量 ΔA → 加回 x 得 x1；再 Norm(x1) → MLP 增量 ΔM → 加回 x1 得 y。');
   setText(h.t3,'Attention 跨位置因果读取 · MLP 每位置共享参数 · RoPE 只进 Q/K · RMS 控尺度 · 输出仍是 3 个位置表示');
  }
  focusButtons.forEach((b,i)=>{
   if(m)placeBtn(b,12+i*76,84,72,24,`聚焦 p${i+1}`,i===focus);
   else placeBtn(b,18+i*106,84,100,24,`聚焦 p${i+1}`,i===focus);
  });
  if(m)placeBtn(ablationBtn,244,84,100,24,ablation?'恢复本例':'断开旁路',ablation);
  else placeBtn(ablationBtn,342,84,142,24,ablation?'恢复本例旁路':'断开旁路（对照）',ablation);
  ablationBtn.setAttribute('aria-label',ablation?'恢复本例旁路':'断开旁路（对照）');
  if(m){show(hint,false);}else{place(hint,18,124);size(hint,8);show(hint,true);}
  const b=M.blockPass({residual:!ablation});
  const pos=b.positions,fd=pos[focus];
  const inputIn=phase(p,0,.25),attnIn=phase(p,.85,1),normOn=phase(p,1,1.2);
  const readOut=phase(p,1.2,1.5),readBack=phase(p,1.5,1.8);
  const merge1Card=phase(p,1.8,2),merge1Bar=phase(p,2,2.5),merge1Done=p>=2.5;
  const mlpIn=phase(p,3,3.15),mlpWork=phase(p,3.15,3.65),mlpDone=mlpWork>=1;
  const merge2Card=phase(p,3.8,4),merge2Bar=phase(p,4,4.5),merge2Done=p>=4.5;
  const laneX=m?[12,128,244]:[18,186,354],laneW=m?104:146,laneC=laneX.map(x=>x+laneW/2);
  const mainX=m?12:14,mainW=m?336:500;
  const cardY=m?130:140,cardH=m?42:46;
  const bandY=m?180:194,bandH=m?128:140;
  const merge1Y=m?322:350,mergeH=m?54:60;
  const mlpY=m?384:432,mlpH=m?118:126;
  const merge2Y=m?516:574;
  const busTop=m?172:190,busBottom=m?382:414,bus2Top=m?382:420,bus2Bottom=m?576:640;
  const busX=m?3:8,bus2X=m?357:514;

  place(inputLabel,mainX,m?124:134);size(inputLabel,m?8.4:10.2);
  setText(inputLabel,m?'① 同一层 3 个位置表示（不是 token）':'① 输入 x：同一层的 3 个位置表示（不是 3 个 token 候选）；看它们怎样被两次加增量更新');
  inputCards.forEach((c,i)=>{
   c.g.setAttribute('transform',`translate(${laneX[i]} ${cardY})`);put(c.box,0,0,laneW,cardH);
   c.box.setAttribute('class',i===focus?'cmp-poscard is-focus':'cmp-poscard');
   place(c.name,8,m?14:14);size(c.name,m?8.6:10.2);
   place(c.val,8,m?30:32);size(c.val,m?8.2:10.2);setText(c.val,vtext(pos[i].x,2));
   place(c.tag,laneW-6,14);c.tag.setAttribute('text-anchor','end');size(c.tag,m?6.6:7.8);setText(c.tag,i===focus?'焦点':'同层');
   fade(c.g,(i===focus?1:.78)*inputIn);show(c.g,inputIn>.01);
  });

  put(band,mainX,bandY,mainW,bandH);
  place(bandTitle,mainX+12,bandY+18);size(bandTitle,m?9.4:11.2);
  setText(bandTitle,m?'② Attention：Norm→RoPE→因果读':'② Pre-Norm + Attention：RMS 控尺度，RoPE 进 Q/K，因果跨位置读');
  place(bandSub,mainX+12,bandY+33);size(bandSub,m?7.6:8.8);
  setText(bandSub,m?'RMS 控尺度；RoPE 只转 Q/K；Q 只读 j≤i。':'每个 Q_i 只读 j ≤ i；V 的加权和写回位置 i，得到增量 ΔA。');
  show(band,attnIn>.01);fade(band,attnIn);show(bandTitle,attnIn>.01);fade(bandTitle,attnIn);show(bandSub,attnIn>.01);fade(bandSub,attnIn);
  const normY=bandY+(m?36:40),normH=m?16:18;
  normChips.forEach((c,i)=>{
   c.g.setAttribute('transform',`translate(${laneX[i]+2} ${normY})`);put(c.box,0,0,laneW-4,normH);
   const r=normOn>.45?`${pos[i].rmsX.toFixed(2)}→1.00`:'…';
   setText(c.t,m?`Norm x ${r}`:`Norm(x)  RMS ${r}  · γ=1`);
   place(c.t,6,normH-4);size(c.t,m?6.8:8.1);
   show(c.g,normOn>.01);fade(c.g,normOn);
  });
  const ky=bandY+(m?58:66),qy=bandY+(m?90:100),nodeW=m?58:70,nodeH=m?14:16;
  kNodes.forEach((n,i)=>{
   n.g.setAttribute('transform',`translate(${laneC[i]-nodeW/2} ${ky})`);put(n.box,0,0,nodeW,nodeH);
   n.box.setAttribute('class',i<=focus?'cmp-attn-node':'cmp-attn-node is-dim');
   place(n.t,nodeW/2,nodeH-4);n.t.setAttribute('text-anchor','middle');size(n.t,m?7.2:8.6);setText(n.t,`K${i+1} ↻`);
   show(n.g,attnIn>.01);fade(n.g,(i<=focus?1:.35)*attnIn);
  });
  qNodes.forEach((n,i)=>{
   n.g.setAttribute('transform',`translate(${laneC[i]-nodeW/2} ${qy})`);put(n.box,0,0,nodeW,nodeH);
   n.box.setAttribute('class',i===focus?'cmp-attn-node is-focus':'cmp-attn-node is-dim');
   place(n.t,nodeW/2,nodeH-4);n.t.setAttribute('text-anchor','middle');size(n.t,m?7.2:8.6);setText(n.t,`Q${i+1} ↻`);
   show(n.g,attnIn>.01);fade(n.g,(i===focus?1:.4)*attnIn);
  });
  const cubic=(a,b2,c,d,t)=>{const u=1-t;return u*u*u*a+3*u*u*t*b2+3*u*t*t*c+t*t*t*d;};
  beams.forEach((beam,j)=>{
   const on=j<=focus&&attnIn>.01;
   if(on){
    const sx=laneC[focus],sy=qy,tx=laneC[j],ty=ky+nodeH;
    const cy1=sy-14,cy2=ty+14;
    beam.setAttribute('d',`M${sx} ${sy} C${sx} ${cy1} ${tx} ${cy2} ${tx} ${ty}`);
    beam.setAttribute('pathLength','1');beam.style.strokeDasharray='1';beam.style.strokeDashoffset=String(1-readOut);
    beam.style.stroke=j===focus?'#f2b45c':'#d59a4a';
    const t=readBack>.001?1-readBack:readOut;
    const px=cubic(sx,sx,tx,tx,t),py=cubic(sy,cy1,cy2,ty,t);
    const pulse=beamPulses[j];pulse.setAttribute('cx',px);pulse.setAttribute('cy',py);
    pulse.setAttribute('fill',readBack>.001?'#7fe0c3':'#f2b45c');
    show(pulse,readOut>.02&&readBack<.98);fade(pulse,readOut>.02?1:0);
   }else{show(beamPulses[j],false);}
   show(beam,on&&readOut>.01);fade(beam,on?1:.2);
  });
  place(readout,mainX+12,bandY+bandH-6);size(readout,m?7.4:8.8);
  let readText;
  if(p<1||normOn<1)readText='先 Norm(x)，完成后才按位置旋转 Q/K 并读取 j≤i。';
  else if(readOut<=.01)readText='Norm(x) 已完成；Q 开始读取 j≤i。';
  else if(readBack<1)readText=`Q${focus+1} 读取 K1–K${focus+1}…（权重由 Q·K 打分）`;
  else readText=`Q${focus+1} 读 K1–K${focus+1}（w ${fd.weights.map(w=>w.toFixed(2)).join(' / ')}）→ ΔA=${vtext(fd.attnOut)}`;
  setText(readout,readText);
  readout.setAttribute('class',readBack>=1?'att-score':'att-dim');
  show(readout,attnIn>.01);fade(readout,attnIn);

  place(x1Label,mainX,m?316:344);size(x1Label,m?8.8:10.5);
  setText(x1Label,m?'③ x1 = x + ΔA（旁路保留 x）':'③ 第一次残差：x1 = x + ΔA；旁路把 x 原样带到合并点');
  show(x1Label,merge1Card>.01);fade(x1Label,merge1Card);
  if(!m){place(colorLegend,mainX+mainW,344);colorLegend.setAttribute('text-anchor','end');size(colorLegend,7.6);show(colorLegend,merge1Card>.01);fade(colorLegend,merge1Card);}
  else show(colorLegend,false);
  merge1Cards.forEach((c,i)=>{
   const base=pos[i].x,delta=pos[i].attnOut,result=pos[i].x1;
   const tag=ablation?'仅 ΔA（对照）':'x + ΔA';
   placeMergeCard(c,laneX[i],merge1Y,laneW,mergeH,base,delta,result,!ablation,merge1Bar,merge1Done,`x1 p${i+1}`,tag,ablation,'x');
   show(c.g,merge1Card>.01);fade(c.g,merge1Card*(i===focus?1:.8));
  });
  place(mergeOrder,m?mainX+mainW:538,m?316:516);
  mergeOrder.setAttribute('text-anchor',m?'end':'start');size(mergeOrder,m?8.0:7.8);
  show(mergeOrder,merge1Card>.01);fade(mergeOrder,merge1Card);

  put(mlpPanel,mainX,mlpY,mainW,mlpH);
  place(mlpTitle,mainX+12,mlpY+18);size(mlpTitle,m?9.4:11.2);
  setText(mlpTitle,m?'④ MLP：Norm(x1) 后逐位置共享 SwiGLU':'④ MLP 逐位置：先 Norm(x1)，再用同一套 SwiGLU 加工每一列');
  place(mlpSub,mainX+12,mlpY+33);size(mlpSub,m?7.4:8.8);
  setText(mlpSub,m?'共享 Wg/Wu/Wd；列间无箭头，不混合位置。':'三列共享 Wg/Wu/Wd，但各用本列 x1；g 经 SiLU 得 a，a⊙u 后由 Wd 回投；列间没有箭头，不混合位置。');
  show(mlpPanel,mlpIn>.01);fade(mlpPanel,mlpIn);show(mlpTitle,mlpIn>.01);fade(mlpTitle,mlpIn);show(mlpSub,mlpIn>.01);fade(mlpSub,mlpIn);
  const mlpNormY=mlpY+(m?38:38),mlpNormH=m?14:16;
  mlpNormChips.forEach((c,i)=>{
   c.g.setAttribute('transform',`translate(${laneX[i]+2} ${mlpNormY})`);put(c.box,0,0,laneW-4,mlpNormH);
   setText(c.t,m?`Norm x1 ${pos[i].rmsX1.toFixed(2)}→1.00`:`Norm(x1)  RMS ${pos[i].rmsX1.toFixed(2)}→1.00  · γ=1`);
   place(c.t,6,mlpNormH-4);size(c.t,m?6.6:8.1);
   show(c.g,mlpIn>.01);fade(c.g,mlpIn);
  });
  const sharedY=mlpY+(m?56:64);
  sharedLine.setAttribute('d',`M${laneX[0]+6} ${sharedY} H${laneX[2]+laneW-6}`);
  place(sharedText,mainX+mainW/2,sharedY-2);sharedText.setAttribute('text-anchor','middle');
  setText(sharedText,m?'共享 Wg/Wu/Wd · 输入各自不同':'同一套 Wg / Wu / Wd · 每列输入不同');
  size(sharedText,m?6.8:8.0);show(sharedLine,mlpIn>.01);fade(sharedLine,.55*mlpIn);show(sharedText,mlpIn>.01);fade(sharedText,mlpIn);
  const boxY=mlpY+(m?64:72),boxH=m?48:56;
  mlpBoxes.forEach((c,i)=>{
   c.g.setAttribute('transform',`translate(${laneX[i]} ${boxY})`);put(c.box,0,0,laneW,boxH);
   const p1w=m?62:86,p2w=m?36:46,pillH=m?13:14;
   put(c.gp,6,6,p1w,pillH);place(c.gt,6+p1w/2,6+pillH-3);c.gt.setAttribute('text-anchor','middle');size(c.gt,m?6.2:7.2);setText(c.gt,m?'g→SiLU':'g=xWg→SiLU');
   put(c.up,6+p1w+4,6,p2w,pillH);place(c.ut,6+p1w+4+p2w/2,6+pillH-3);c.ut.setAttribute('text-anchor','middle');size(c.ut,m?6.2:7.2);setText(c.ut,m?'u':'u=xWu');
   place(c.row,6,m?29:32);size(c.row,m?7.2:8.2);setText(c.row,'a ⊙ u → Wd');
   place(c.val,6,m?42:48);size(c.val,m?7.4:8.6);
   if(mlpDone){setText(c.val,`ΔM = ${vtext(pos[i].dM)}`);c.val.setAttribute('class','cmp-mlp-delta is-done');}
   else{setText(c.val,'ΔM …');c.val.setAttribute('class','cmp-mlp-delta');}
   show(c.g,mlpWork>.01);fade(c.g,mlpWork*(i===focus?1:.82));
  });

  place(merge2Label,mainX,m?510:568);size(merge2Label,m?8.8:10.5);
  setText(merge2Label,m?'⑤ y = x1 + ΔM（旁路来自 x1）':'⑤ 第二次残差：y = x1 + ΔM；第二旁路来自 x1，不是 x');
  show(merge2Label,merge2Card>.01);fade(merge2Label,merge2Card);
  merge2Cards.forEach((c,i)=>{
   const base=pos[i].x1,delta=pos[i].dM,result=pos[i].y;
   const tag=ablation?'仅 ΔM（对照）':'x1 + ΔM';
   placeMergeCard(c,laneX[i],merge2Y,laneW,mergeH,base,delta,result,!ablation,merge2Bar,merge2Done,`y p${i+1}`,tag,ablation,'x1');
   show(c.g,merge2Card>.01);fade(c.g,merge2Card*(i===focus?1:.8));
  });

  if(ablation){
   xBus.style.display='none';xStubTop.style.display='none';xStubBottom.style.display='none';x1Bus.style.display='none';x1StubTop.style.display='none';x1StubBottom.style.display='none';
   xBusTag.style.display='none';x1BusTag.style.display='none';
  }else{
   xBus.setAttribute('d',`M${busX} ${busTop} H${laneX[2]+laneW} M${busX} ${busTop} V${busBottom} M${busX} ${busBottom} H${laneX[2]+laneW}`);
   xStubTop.setAttribute('d',laneC.map(x=>`M${x} ${busTop} V${busTop-4}`).join(' '));
   xStubBottom.setAttribute('d',laneC.map(x=>`M${x} ${busBottom} V${busBottom-4}`).join(' '));
   x1Bus.setAttribute('d',`M${laneX[0]} ${bus2Top} H${bus2X} M${bus2X} ${bus2Top} V${bus2Bottom} M${laneX[0]} ${bus2Bottom} H${bus2X}`);
   x1StubTop.setAttribute('d',laneC.map(x=>`M${x} ${bus2Top} V${bus2Top-7}`).join(' '));
   x1StubBottom.setAttribute('d',laneC.map(x=>`M${x} ${bus2Bottom} V${bus2Bottom-6}`).join(' '));
   show(xBus,merge1Bar>.01);fade(xBus,.8*merge1Bar);show(xStubTop,merge1Bar>.01);fade(xStubTop,.8*merge1Bar);show(xStubBottom,merge1Bar>.01);fade(xStubBottom,.8*merge1Bar);
   show(x1Bus,merge2Bar>.01);fade(x1Bus,.8*merge2Bar);show(x1StubTop,merge2Bar>.01);fade(x1StubTop,.8*merge2Bar);show(x1StubBottom,merge2Bar>.01);fade(x1StubBottom,.8*merge2Bar);
   if(m){show(xBusTag,false);show(x1BusTag,false);}
   else{
    place(xBusTag,4,busTop+(busBottom-busTop)/2-16);xBusTag.setAttribute('transform',`rotate(-90 4 ${busTop+(busBottom-busTop)/2-16})`);size(xBusTag,7.6);show(xBusTag,merge1Bar>.2);
    place(x1BusTag,bus2X-3,bus2Top+(bus2Bottom-bus2Top)/2-16);x1BusTag.setAttribute('transform',`rotate(-90 ${bus2X-3} ${bus2Top+(bus2Bottom-bus2Top)/2-16})`);size(x1BusTag,7.6);show(x1BusTag,merge2Bar>.2);
   }
  }

  place(outTitle,mainX,m?582:652);size(outTitle,m?8.8:10.5);
  if(ablation){setText(outTitle,'对照输出：没有显式旁路把 x 作为相加基底');outTitle.setAttribute('class','cmp-block-warn-text');}
  else{setText(outTitle,'输出 y：仍是 3 个位置表示（p1–p3），不是 1 个新 token');outTitle.setAttribute('class','att-strong');}
  place(outSub,mainX,m?596:666);size(outSub,m?7.2:8.6);
  setText(outSub,m?'下一 token 要等全部层之后':'下一 token 要等全部层之后：输出头给分数，生成过程再选择。');
  show(outTitle,merge2Card>.01);fade(outTitle,merge2Card);show(outSub,merge2Card>.01);fade(outSub,merge2Card);

  if(m){
   put(rail,12,604,336,58);show(rail,true);
   place(railTitle,24,620);size(railTitle,8);setText(railTitle,'输出仍是 3 个位置表示，不是 token');railTitle.setAttribute('class','att-w');
   [why1,why2,why3,scopeTitle,notToken,notTokenText,next1,next2,object,hint2,teaching1,teaching2,teaching3,ablationNote].forEach(e=>show(e,false));
   show(scope1,false);show(detailTitle,false);
   detailChips.forEach((c,i)=>{
    const x=16+i*66,y=628;c.g.setAttribute('transform',`translate(${x} ${y})`);put(c.box,0,0,64,18);place(c.t,6,12);size(c.t,6.6);setText(c.t,c.short);show(c.g,true);
   });
   show(colorLegend,false);
  }else{
   put(rail,524,118,184,550);show(rail,true);
   place(railTitle,538,140);size(railTitle,10.8);railTitle.setAttribute('class','att-strong');
   place(why1,538,159);place(why2,538,173);place(why3,538,187);[why1,why2,why3].forEach(t=>{size(t,8.2);show(t,true);});
   place(scopeTitle,538,218);size(scopeTitle,10.2);
   place(scope1,538,238);size(scope1,9.0);
   put(notToken,538,246,156,20);place(notTokenText,546,260);size(notTokenText,9.0);
   place(next1,538,282);place(next2,538,296);[next1,next2].forEach(t=>{size(t,8.2);show(t,true);});
   place(object,538,314);size(object,8.0);
   place(detailTitle,538,342);size(detailTitle,9.0);
   detailChips.forEach((c,i)=>{const y=354+i*27;c.g.setAttribute('transform',`translate(538 ${y})`);put(c.box,0,0,156,22);place(c.t,8,15);size(c.t,8.4);setText(c.t,c.full);show(c.g,true);});
   place(hint2,538,502);size(hint2,7.4);show(hint2,true);
   place(colorLegend,538,530);colorLegend.setAttribute('text-anchor','start');size(colorLegend,7.8);show(colorLegend,true);
   place(teaching1,538,552);place(teaching2,538,566);place(teaching3,538,580);[teaching1,teaching2,teaching3].forEach(t=>{size(t,7.4);show(t,true);});
   place(ablationNote,538,602);size(ablationNote,7.6);show(ablationNote,ablation);
   [scopeTitle,scope1,notToken,notTokenText,next1,next2,object,detailTitle].forEach(e=>show(e,true));
   show(railTitle,true);
  }
  setText(foot,ablation?'反事实：两条旁路断开并重算下游；没有显式相加基底，子层仍读输入，不等于信息必然丢失。':'教学例：二维表示 · 单头注意力（Wq=Wk=Wv=Wo=I）· γ=[1,1] · Wd=I；数值由固定公式算出，不是模型观测。');
  foot.setAttribute('class',ablation?'att-foot cmp-block-warn-text':'att-foot');
  place(foot,m?12:18,688);size(foot,m?7.2:9.4);show(foot,true);

  svg.dataset.component='block';svg.dataset.layerStep=String(Math.min(5,Math.floor(p+1e-6)));
  svg.dataset.focus=`p${focus+1}`;svg.dataset.readRange=fd.readRange;
  svg.dataset.readOut=readOut.toFixed(3);svg.dataset.readBack=readBack.toFixed(3);
  svg.dataset.mergeA=merge1Bar.toFixed(3);svg.dataset.mlpProgress=mlpWork.toFixed(3);svg.dataset.mergeM=merge2Bar.toFixed(3);
  svg.dataset.residualInputs='x,x1';svg.dataset.preNorm='true';svg.dataset.generalizesAllTransformers='false';
  svg.dataset.firstBypassFrom=ablation?'none-counterfactual':'x';svg.dataset.secondBypassFrom=ablation?'none-counterfactual':'x1';
  svg.dataset.normDone=String(normOn>=.999);svg.dataset.mlpDone=String(mlpDone);svg.dataset.mlpInput='x1-norm';svg.dataset.mlpShared='Wg,Wu,Wd';
  svg.dataset.attention='causal-cross-position';svg.dataset.mlp='per-position-shared';
  svg.dataset.rope='q,k';svg.dataset.rms='scale';svg.dataset.positions='3';svg.dataset.oneLayerIsOneToken='false';
  svg.dataset.teaching='explicit-deterministic';svg.dataset.modelObservation='false';
  svg.dataset.residual=ablation?'counterfactual-off':'on';
  svg.dataset.x=vtext(pos[focus].x,6);svg.dataset.attn=readBack>=.999?vtext(pos[focus].attnOut,6):'';
  svg.dataset.x1=merge1Done?vtext(pos[focus].x1,6):'';svg.dataset.mlpDelta=mlpDone?vtext(pos[focus].dM,6):'';
  svg.dataset.y=merge2Done?vtext(pos[focus].y,6):'';
  svg.dataset.weights=fd.weights.map(w=>w.toFixed(6)).join(',');
  svg.dataset.resultState=merge2Done?'complete':merge2Bar>0?'merging':mlpDone?'mlp':merge1Done?'x1':merge1Card>0?'x1-pending':readBack>0?'attention':'input';
  svg.dataset.phase=p<1?'input':p<2?'attention':p<3?'x1':p<4?'mlp':p<4.5?'mlp-complete':p<5?'y':'review';
  svg.dataset.ablation=String(ablation);
 }
 return draw;
}

const frames=(...items)=>items.map(([title,caption])=>({title,caption}));
const choices=[
 {concept:'rope',name:'01 位置与 RoPE',title:'RoPE：同样的内容，为什么比较会不同？',
  intro:'Q 在 p2，三个 Key 内容相同、位置不同。先看位置在轨道上的分布，再看 Q/K 的一对特征分量怎样按位置真正旋转，最后逐行比较 Q 与 K1/K2/K3，并用整段 +3 检查：位置平移对应共同旋转，相对角与点积不变。',
  boundary:'θ=π/4、q=k=[1,0] 的教学例；只展开一对维度，Q 在 p2，K 在 p0–p2（整段 +3 时到 p3–p5）。Δθ=(pQ−pK)θ。比较值是 softmax 前的 Q·K，不是注意力权重；本例原内容同向、长度 1，点积才化为 cosΔθ，换内容后不推出“越近越高”。高维有多对维度、不同频率；旋转的是 Q/K 特征分量，不是 token；V 不参与。',
  data:{pQ:2,pK:1,theta:Math.PI/4,thetaDeg:45,dimensions:1,conditions:['base','q1','shift3'],projectionKey:'K2'},
  frames:frames(
   ['先分清两件事：轨道位置与向量角度','轨道上的卡片只平移；坐标平面里的 Q/K 才旋转。起点从 0° 开始，只移 Q 与整段 +3 从基准角开始。'],
   ['位置进入：Q/K 按 p·45° 真正旋转','端点由旋转得到，沿单位圆走；半径不变，所以长度不变。K3 与 Q 同位置时始终重合。'],
   ['夹角决定点积：Q 分别比较 K1–K3','每行用当前 q′、k′ 算 Δθ 与点积；K2 的投影和条形取同一个数。负值在零线左侧。'],
   ['换条件检查：哪些变了，哪些没变','只移 Q +1 时只有 Q 多转 45°，负比较出现；整段 +3 时轨道右移、所有向量共同 +135°，三行数值不变。条件只在你点击时切换。']
  ),stepDurations:[1.2,2.0,1.6,1.2]},
 {concept:'rmsnorm',name:'02 RMSNorm',title:'RMSNorm：先拉齐整体幅度，再逐维调整',
  intro:'默认 γ=[1,2]：方向相同的 x 与 2x 幅度差 2 倍；归一化先把两条箭头沿原方向缩到近似共同尺度，再把 dim1 拉长。',
  boundary:'x=[1.2,1.6]、2x=[2.4,3.2]、eps=1e−6；逐位置处理。归一化后分量 RMS≈1，但不是欧氏长度 1（二维长度≈1.414）；γ=[1,2] 后长度≈2.416、RMS≈1.709。ε 使 2x 的结果只是近似相同；γ 不均匀会改变方向，γ 不是第二次归一化。',
  data:{x:[1.2,1.6],gamma:[1,2],eps:1e-6},
  frames:frames(
   ['同一方向，幅度差 2 倍','x=[1.2,1.6]、2x=[2.4,3.2] 方向相同；2x 的箭头长一倍，两图用同一把尺。'],
   ['各自除以自己的整体幅度','两条箭头沿原方向缩短；长的 2x 缩得更多，短的 x 缩得少。'],
   ['收敛到近似共同尺度','两条箭头端点近似重合；方向与分量比例没变，整体幅度差基本拉平。'],
   ['再逐维乘 γ','默认 γ=[1,2]：dim0 不变、dim1 拉长；切到 γ=[1,1] 则两维都不变。'],
   ['同一 γ 下仍近似重合','x 与 2x 经过同一个 γ 得到近似相同的输出；默认 [1,2] 改变方向，切到 [1,1] 保方向。'],
   ['均匀与不均匀 γ 对照','默认 [1,2] 下偏转约 16.3°，虚线对照 [1,1] 保方向；切到 [1,1] 则虚线对照 [1,2]。']
  ),stepDurations:[1.2,1.6,1.6,1.6,1.6]},
 {concept:'residual',name:'03 残差连接',title:'残差：原表示主路 + 逐维更新',
  intro:'同一个 x 分两路：主路原样保留；子层只产生可正可负的 F(x)，到达后按维度加到 x。只把 F 当输出会丢掉主路。',
  boundary:'x=[1,2]、F=[0.5,−1] 的教学例；F 由同一个 x 得到，逐元素相加，不是拼接；不声称每次加法都无损，也不演示训练梯度。',
  data:{x:[1,2],fx:[.5,-1]},
  frames:frames(
   ['原表示先走主路','青色的 x 基底先出现：x=[1,2]，它不经过子层，沿旁路保留。'],
   ['子层读同一个 x，只产生 F','F(x) 与 x 同形，可正可负；橙色/粉色条是增量，不是替代 x。'],
   ['dim0 更新到达后才合并','+0.5 先沿支路到达；到达前不合并，到达后蓝色基底上才长出 +0.5 的变化段，y₀=1.5。'],
   ['dim1 负更新让结果变小','−1 到达后从 x₁=2 向左退回 1；基底仍保留在 2，结果标记落在 1，不把结果看回原长。'],
   ['只传 F 与 x+F 的差','对照带显示：只传 F 停在 F 的端点，x+F 还要加上主路 x；差正好是 x。F=0 时前者是 0，后者仍是 x。'],
   ['换 F 核对','点按钮换到 F=[0,0] 或 F=[1,1]；切换会回到 0 并暂停。每个维度各自相加，仍是 2 维。']
  ),stepDurations:[1.2,1.5,1.7,1.5,1.6,1.3]},
 {concept:'swiglu',name:'04 MLP / SwiGLU',title:'MLP / SwiGLU：位置内部加工与连续调制',
  intro:'默认用焦点 p1 看完整链路：两路投影 → SiLU 连续系数 → 逐元素相乘 → Wd 下投影；上方三个位置共享同一套参数。',
  boundary:'显式教学矩阵 Wg=[[1,1],[1,−1]]、Wu=[[0,1],[1,1]]、Wd=I；g/u 只是输入维度的线性组合，不绑定真实神经元的固定语义。a=SiLU(g) 连续、可负、可大于 1，不是概率、0/1 开关或专家路由；Wd=I 只为看清乘法，真实 FFN 会把中间维回投模型宽度。',
  data:{x:[1,2]},
  frames:frames(
   ['三个位置，各带自己的 x','p1–p3 共用 Wg/Wu/Wd，但每列用自己的 x 独立计算；列间没有箭头，不互相读取。'],
   ['两路投影：门支路与值支路','以 p1 x=[1,2] 为例：g=xWg=[3,−1] 走门支路，u=xWu=[2,3] 走值支路；两条支路都来自同一个 x。'],
   ['SiLU 把 g 变成连续系数','a=SiLU(g)≈[2.8577,−0.2689]；刻度上 a 可大于 1，也可小于 0，不是 0/1 门或概率。'],
   ['逐元素相乘：a 改写 u','z0=2.8577×2=5.7154，同号变长；z1=−0.2689×3=−0.8068，与 u1 反向；⊙ 是逐元素乘，不是矩阵乘。'],
   ['Wd 下投影回投表示','Wd=I 时 y=z=[5.7154,−0.8068]；真实 FFN 的 Wd 会把中间维回投模型宽度，不是把多维压成概率。'],
   ['换输入，权重不变','切到 p2 x=[2,1]：g=[3,1]、u=[1,3]、a≈[2.8577,0.7311]、z≈[2.8577,2.1932]；变的是输入相关调制，不是专家路由。']
  ),stepDurations:[1.0,1.1,1.4,1.5,1.3]},
 {concept:'heads',name:'05 MHA / GQA / MQA',title:'MHA / GQA / MQA：Q 头保留，并排看 K/V 怎么共享',
  intro:'三种方案同屏并排：每列都有 4 个独立 Q 头，只改独立 K/V 组数 4/2/1。先看每个 Q 读哪一组、历史怎样先写满再读取，再用同一刻度对照历史 3/6 的缓存元素。',
  boundary:'历史 3 时 48/24/12、历史 6 时 96/48/24，只是该层 K+V 元素数；三种拓扑是并排比较，不是把已训练 MHA 的 K/V 无损压缩或把旧缓存并入 GQA，也不是总显存、带宽或加速比。连线只表示逻辑上读哪一组；实际 kernel 读取次数与带宽取决于实现，Q 头数、concat 与 output projection 不变，输出与质量不保证相同。',
  data:{kvHeads:4,qHeads:4,positions:3,dim:2,history:3},
  frames:frames(
   ['三种方案同屏，Q 头都保留 4 个','MHA、GQA、MQA 每列都有 Q0–Q3；共享只改变 K/V 组数，不合并 Q，也不合并 output projection。'],
   ['谁共享哪一组：看连线终点','MHA 的 Qi→KVi；GQA 的 Q0/Q1→KV0、Q2/Q3→KV1；MQA 四个 Q 都→KV0。线只表示逻辑读取。'],
   ['历史先写满，再读取','每个 KV 组先按历史 3 或 6 写入自己的 K/V 格；没写满前 Q 不读取，写满后才开始 Q→KV。'],
   ['读取返回 Q，历史仍保留','Q 从所属组读取并返回；读取不清空历史，四个 Q 仍各自形成查询，再 concat 后做 output projection。'],
   ['公平缓存对照：同一把尺','该层 K+V=2×历史×2×组数；历史 3 为 48/24/12，历史 6 为 96/48/24，柱长可直接比较。'],
   ['换历史，看共享的差别怎样放大','历史 3→6 各组翻倍；共享减少的是 K/V 副本，不是 Q 头、带宽或质量。并排不是旧 MHA 缓存的无损合并。']
  ),stepDurations:[1.2,1.6,1.8,1.4,1.6,1.3]},
 {concept:'block',name:'06 Transformer Block',title:'Transformer Block：先交互，再逐位置加工',
  intro:'默认用过程看二维教学例：Attention 跨位置、MLP 逐位置，两条残差分别来自 x 与 x₁；数值由固定公式算出。',
  boundary:'只讲一层前向。二维表示、单头注意力、Wq=Wk=Wv=Wo=I、γ=[1,1]、Wd=I 都是固定教学设定，不是实际模型观测；本例固定 Pre-Norm（现代 Transformer 常见，不泛化所有模型）。第二个旁路来自 x₁；断开旁路只是移除显式相加基底并重算下游，不证明信息必然丢失。一层输出是 3 个位置的更新表示，不是 1 个新 token。',
  data:{positions:3},
  frames:frames(
   ['先出现 3 个位置','p1–p3 是同一层的三个序列位置，不是 3 个 token 候选；每列有自己的表示 x。'],
   ['Attention 跨位置','先 Norm(x) 完成，再按位置旋转 Q/K；聚焦 p3 时 Q3 读 K/V p1–p3；p1 只读自己。RoPE 进 Q/K，KV 组数另选。'],
   ['第一个残差保留 x','x₁ = x + Attention(Norm(x))；先显示 x 基底，再让 ΔA 逐分量生长，合并完成才发布 x₁。'],
   ['MLP 逐位置','每列先 Norm(x₁)，再走同一套 SwiGLU MLP；三列输入不同、参数共享，MLP 完成后才发布 ΔM。'],
   ['第二个残差保留 x₁','y = x₁ + MLP(Norm(x₁))；第二个旁路的基底是 x₁，不是最初的 x；合并完成才发布 y，输出仍是 3 个位置表示。'],
   ['一层 ≠ 一个 token','全部层之后由输出头给下一 token 分数，生成再选择；层号、位置、token 是不同对象。']
  ),stepDurations:[1.3,1.8,1.6,1.7,1.6,1.5]}

];

globalThis.TeachingLessons=globalThis.TeachingLessons||{};
globalThis.TeachingLessons['f-components']=choices;

V['f-components']=(svg,viewport,s)=>{
 const film=svg.closest('.mechanism-film');
 if(film)film.classList.toggle('is-components',true);
 const compact=s.concept==='rope'?720:['rmsnorm','residual','swiglu','heads','block'].includes(s.concept)?700:660;
 return board(svg,viewport,'components-'+s.concept,m=>m?compact:(s.concept==='block'?700:560),H=>{
  if(s.concept==='rope')return ropeBoard(svg,H);
  if(s.concept==='rmsnorm')return rmsBoard(svg,H);
  if(s.concept==='residual')return residualBoard(svg,H);
  if(s.concept==='swiglu')return swigluBoard(svg,H);
  if(s.concept==='heads')return headsBoard(svg,H);
  return blockBoard(svg,H);
 });
};
})();
