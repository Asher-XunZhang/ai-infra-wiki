/* Chapter 1, lesson 01. Five compact, independently selectable token concepts.
 * The board is deliberately its own dark mechanism canvas: one input, one local
 * operation and one result per concept. Numbers are teaching values from
 * tokens-facts.md; real Qwen IDs and the 0..3 toy embedding table never mix.
 *
 * Stable entries: tokens.html?concept=bpe|position|lookup|boundary|contract
 * Legacy entries: tokens.html?mv=0 (我想学习人工智能。) and ?mv=1
 * (我想学习机器学习。) keep the original fixed samples and their older drawing.
 * mv=10..14 are also accepted as stable numeric aliases for the five concepts. */
(function(root){
'use strict';
const V=root.MechanismVisuals,D=root.TeachingDrawing;
if(!V||!D)return;
const {board,phase,clamp}=D;
const legacyVisual=V['f-tokens'];
const fmt=(v,d=1)=>'['+v.map(x=>{const n=Number(x.toFixed(d));return String(n);}).join(', ')+']';
const BPE_RULES=[
 {n:'①',text:'机 + 器 → 机器',sub:'只合并这一对相邻单位'},
 {n:'②',text:'学 + 习 → 学习',sub:'只合并这一对相邻单位'},
 {n:'③',text:'机器 + 学习 → 机器学习',sub:'最后合成一个单位'}
];
const TOY=[
 {id:0,word:'我',v:[1,0,0]},
 {id:1,word:'学习',v:[.2,.8,-.1]},
 {id:2,word:'机器',v:[0,1,0]},
 {id:3,word:'。',v:[0,0,1]}
];
/* Training shows this update in row 2 as a teaching contrast, never as a
 * persistent write. Both modes start from the same initial table E. */
const UPDATED=[.23,.76,-.08];
const EX2={text:'我想学习机器学习。',pieces:['我想','学习','机器','学习','。'],ids:[104100,100134,102182,100134,1773]};
const frameList=(...items)=>items.map(([title,caption])=>({title,caption}));
/* A click may reveal its selected result immediately, but that snapshot must
 * not become a permanent terminal state. The player controls bump the epoch;
 * the next play/seek/reset clears the pin and phase(p) owns the reveal again. */
let interactionEpoch=0,pauseFilm=()=>{};
function watchProgress(svg){
 const film=svg.closest('.mechanism-film');
 if(!film||film.dataset.tokWatched)return;
 film.dataset.tokWatched='1';
 film.addEventListener('input',e=>{if(e.target.closest?.('.motion-scrub'))interactionEpoch++;});
 film.addEventListener('click',e=>{if(e.target.closest?.('.motion-controls button,.motion-panel button,.motion-panel select'))interactionEpoch++;});
}
function isPinned(pin,p){
 if(pin.p===null)return false;
 if(pin.epoch!==interactionEpoch||Math.abs(p-pin.p)>1e-4){pin.p=null;return false;}
 return true;
}
function pinHere(pin,p){pauseFilm();pin.p=p;pin.epoch=interactionEpoch;}

function hit(g,label,fn){
 g.setAttribute('role','button');g.setAttribute('tabindex','0');g.classList.add('tok-hit');g.setAttribute('aria-label',label);
 g.addEventListener('click',e=>{e.stopPropagation();fn();});
 g.addEventListener('keydown',e=>{
  if(e.key==='Enter'||e.key===' '||e.key==='Spacebar'){
   e.preventDefault();e.stopPropagation();fn();
  }
 });
}
function pressed(g,on){g.setAttribute('aria-pressed',on?'true':'false');}
function button(H,x,y,w,h,label,fn,cls){
 const g=H.node('g',{class:'tok-btn '+(cls||'')});
 const r=H.node('rect',{rx:7,width:w,height:h});
 const t=H.node('text',{x:w/2,y:h/2+5,'text-anchor':'middle'},label);
 g.append(r,t);hit(g,label,fn);g.tokRect=r;g.tokLabel=t;g.setAttribute('transform',`translate(${x} ${y})`);return g;
}
function placeButton(H,g,x,y,w,h,label,on){
 H.set(g,{transform:`translate(${x} ${y})`});
 H.set(g.tokRect,{width:w,height:h});
 H.set(g.tokLabel,{x:w/2,y:h/2+5});
 if(g.tokLabel.textContent!==label)g.tokLabel.textContent=label;
 pressed(g,on);
}

/* ------------------------------------------------------------------ 01 BPE */
function bpeBoard(H){
 const {txt,rect,text,box,show,set,node}=H;
 const bg=node('rect',{class:'tok-board'}),band=node('rect',{class:'tok-band'});
 const resultBg=node('rect',{class:'tok-result-box'});
 const ticks=Array.from({length:6},()=>rect('tok-tick'));
 const c=Array.from({length:6},(_,i)=>H.token('bpe-c'+i,'机',44,'tok-token'));
 const m1=H.token('bpe-m1','机器',86,'tok-token tok-amber'),m2=H.token('bpe-m2','学习',86,'tok-token tok-amber'),m3=H.token('bpe-m3','机器学习',136,'tok-token tok-cyan');
 const rules=BPE_RULES.map((r,i)=>{
  const g=node('g',{class:'tok-rule'}),b=node('rect',{rx:6,width:200,height:56});
  const n=node('text',{class:'tok-rule-n',x:12,y:22},r.n);
  const t=node('text',{class:'tok-rule-t',x:34,y:22},r.text);
  const s=node('text',{class:'tok-rule-s',x:34,y:43},r.sub);
  g.append(b,n,t,s);return {g,b,n,t,s,rule:r};
 });
 const title=txt('','tok-title'),sub=txt('','tok-sub'),inLabel=txt('','tok-kicker'),curLabel=txt('','tok-kicker');
 const inputLine=txt('我  爱  机  器  学  习','tok-input-line');
 const count=txt('','tok-count'),seq=txt('6 → 5 → 4 → 3','tok-dim');
 const result=txt('','tok-result'),resultSub=txt('','tok-result-sub');
  const note=txt('','tok-note');

 function layout(m){
  const wc=m?42:54,wm=m?68:86,w3=m?104:136,gap=m?5:8,x0=16;
  const stageSets=[
   [['c0','我',wc],['c1','爱',wc],['c2','机',wc],['c3','器',wc],['c4','学',wc],['c5','习',wc]],
   [['c0','我',wc],['c1','爱',wc],['m1','机器',wm],['c4','学',wc],['c5','习',wc]],
   [['c0','我',wc],['c1','爱',wc],['m1','机器',wm],['m2','学习',wm]],
   [['c0','我',wc],['c1','爱',wc],['m3','机器学习',w3]]
  ];
  return stageSets.map(units=>{const map={};let x=x0;for(const [k,,w]of units){map[k]=x;x+=w+gap;}return map;});
 }
 function draw(p,m){
  const W=m?360:720,HT=m?560:500;
  box(bg,0,0,W,HT);box(band,0,0,W,62);
  text(title,16,27,m?'BPE：把句子合成可复用单位':'分词与 BPE：把句子切成可复用的单位');
  text(sub,16,47,m?'规则固定，只合并相邻单位':'规则来自训练语料；推理只应用，不重新学习');
  text(inLabel,16,m?76:82,m?'输入 · 原句 6 个字符':'输入 · 原句的 6 个字符');
  text(inputLine,16,m?102:110);
  show(inputLine,1);
  text(curLabel,16,m?132:146,'当前序列 · 合并只发生在相邻单位之间');
  const lay=layout(m),wc=m?42:54,wm=m?68:86,w3=m?104:136;
  const curY=m?148:162;
  const stage=Math.min(3,Math.floor(p+1e-9)),u=clamp(p-stage),next=Math.min(3,stage+1);
  const pos=(key,st)=>{
   const map=lay[Math.min(3,st)];
   if(map[key]!==undefined)return map[key];
   if(key==='c2'||key==='c3')return (map.m1??0)+(wm-wc)/2;
   if(key==='c4'||key==='c5')return (map.m2??0)+(wm-wc)/2;
   if(key==='m1')return map.c2??map.m1??0;
   if(key==='m2')return map.c4??0;
   if(key==='m3')return map.m3??map.m1??0;
   return 0;
  };
  const xFor=k=>pos(k,stage)+(pos(k,next)-pos(k,stage))*u;
  const op={c0:1,c1:1,
   c2:1-phase(p,.65,1),c3:1-phase(p,.65,1),
   c4:1-phase(p,1.65,2),c5:1-phase(p,1.65,2),
   m1:phase(p,.65,1)*(1-phase(p,2.65,3)),
   m2:phase(p,1.65,2)*(1-phase(p,2.65,3)),
   m3:phase(p,2.65,3)};
  const chars=['我','爱','机','器','学','习'];
  c.forEach((t,i)=>{set(t.r,{width:wc});set(t.t,{x:wc/2});if(t.t.textContent!==chars[i])t.t.textContent=chars[i];H.move(t,xFor('c'+i),curY,op['c'+i]);});
  [[m1,'m1',wm],[m2,'m2',wm],[m3,'m3',w3]].forEach(([t,k,w])=>{set(t.r,{width:w});set(t.t,{x:w/2});H.move(t,xFor(k),curY,op[k]);});
  const n=p<1?6:p<2?5:p<3?4:3;
  text(count,16,m?204:218,'单位数 '+n);
  text(seq,m?102:118,m?204:218,'6 → 5 → 4 → 3');
  ticks.forEach((t,i)=>{box(t,16+i*(m?22:24),m?212:226,m?12:14,m?12:14);set(t,{class:'tok-tick'+(i<n?' on':'')});});
  const rx=m?16:470,ry=m?236:84,rw=m?328:234,rh=m?46:56;
  rules.forEach((r,i)=>{
   const done=p>=i+1,active=!done&&p>=i;
   set(r.g,{transform:`translate(${rx} ${ry+i*(rh+(m?6:8))})`});
   box(r.b,0,0,rw,rh);set(r.b,{class:'tok-rule'+(active?' is-on':'')});
   text(r.n,12,m?20:22,r.n.textContent);
   text(r.t,34,m?20:22,r.rule.text);
   text(r.s,34,m?38:43,done?'已完成 · 单位数 '+(6-i)+' → '+(5-i):active?'正在应用 · 只动相邻单位':'待应用 · '+r.rule.sub);
  });
  box(resultBg,16,m?400:318,m?328:688,m?76:74);
  text(result,30,m?428:346,p<3?'合并中… 当前 '+n+' 个单位':'结果 · 3 个 token');
  text(resultSub,30,m?452:370,p<3?'继续按固定规则合并相邻单位':'我 / 爱 / 机器学习');
  text(note,16,m?518:430,'BPE 规则在训练语料上固定；推理不会重新学习这套规则。');
 }
 return draw;
}

/* ------------------------------------------------------- 02 ID and position */
function positionBoard(H){
 const {txt,text,box,show,set,node,path}=H;
 const bg=node('rect',{class:'tok-board'}),band=node('rect',{class:'tok-band'});
 const panel=node('rect',{class:'tok-panel'}),stat1Bg=node('rect',{class:'tok-stat-box'}),stat2Bg=node('rect',{class:'tok-stat-box tok-stat-pos'});
 const link=path('tok-link tok-link-amber');
 const cards=EX2.pieces.map((w,i)=>{
  const g=node('g',{class:'tok-card'}),b=node('rect',{rx:8,width:112,height:110});
  const piece=node('text',{class:'tok-piece',x:56,y:34,'text-anchor':'middle'},w);
  const pos=node('text',{class:'tok-pos',x:56,y:62,'text-anchor':'middle'},'p'+(i+1));
  const id=node('text',{class:'tok-id',x:56,y:90,'text-anchor':'middle'},String(EX2.ids[i]));
  g.append(b,piece,pos,id);
  hit(g,`${w}，位置 p${i+1}，ID ${EX2.ids[i]}`,()=>{selected=i;pinHere(pin,lastP);repaint();});
  return {g,b,piece,pos,id};
 });
 const title=txt('','tok-title'),sub=txt('','tok-sub'),kicker=txt('','tok-kicker');
 const col1=txt('片段','tok-small'),col2=txt('位置','tok-small'),col3=txt('token ID','tok-small');
 const pairLabel=txt('','tok-small tok-amber-text');
 const panelTitle=txt('','tok-panel-title');
 const idLabel=txt('token ID','tok-stat-label'),idValue=txt('','tok-stat-value'),idNote=txt('','tok-stat-note');
 const posLabel=txt('出现位置','tok-stat-label'),posValue=txt('','tok-stat-value'),posNote=txt('','tok-stat-note');
 const pairText=txt('','tok-note'),panelNote=txt('','tok-note');
 const note=txt('真实 token ID 来自 Qwen2.5 固定样例；p 只是位置，不是 ID。','tok-note');
 const note2=txt('','tok-note');
 let selected=3,pin={p:null,epoch:-1},lastP=0,lastM=false;
 function repaint(){draw(lastP,lastM);}
 function draw(p,m){
  lastP=p;lastM=m;const pinned=isPinned(pin,p);
  const W=m?360:720,HT=m?560:500;
  box(bg,0,0,W,HT);box(band,0,0,W,62);
  text(title,16,27,m?'ID 与位置：两件不同的事':'Token ID 与位置：编号相同，位置不同');
  text(sub,16,47,m?'同一 ID 可以出现两次；位置不能合并':'两次“学习”取同一行，但占据两个不同的输入位置');
  text(kicker,16,m?82:82,m?'输入 · 5 次出现（真实 Qwen2.5 样例）':'输入 · 原句的 5 次出现（真实 Qwen2.5 样例）');
  const sameSelected=selected===1||selected===3;
  const linkStart=Math.max(phase(p,1.2,2),pinned?1:0);
  const linkOn=linkStart>0||sameSelected;
  if(m){
   text(col1,28,97);text(col2,120,97);text(col3,190,97);
   cards.forEach((c,i)=>{
    const y=104+i*49,on=i===selected,same=EX2.ids[i]===100134;
    set(c.g,{transform:`translate(16 ${y})`});
    set(c.b,{width:328,height:45,class:'tok-card'+(on?' is-on':'')+(same?' is-same':'')});
    set(c.piece,{x:12,y:29,'text-anchor':'start'});
    set(c.pos,{x:104,y:29,'text-anchor':'start'});
    set(c.id,{x:174,y:29,'text-anchor':'start'});
    pressed(c.g,on);
   });
   const top=104+49+22.5,bot=104+3*49+22.5;
   set(link,{d:`M344 ${top} H350 V${bot} H344`,opacity:linkOn?'1':'0'});
   show(pairLabel,linkOn?1:0);
   text(pairLabel,180,362,'同一 ID 100134 · 位置 p2 / p4');set(pairLabel,{'text-anchor':'middle'});
  }else{
   show(col1,0);show(col2,0);show(col3,0);
   const cw=112,gap=10,total=cw*5+gap*4,x0=(W-total)/2,y=94,h=110;
   const centers=[];
   cards.forEach((c,i)=>{
    const x=x0+i*(cw+gap),same=EX2.ids[i]===100134,on=i===selected;
    set(c.g,{transform:`translate(${x} ${y})`});
    set(c.b,{width:cw,height:h,class:'tok-card'+(on?' is-on':'')+(same?' is-same':'')});
    set(c.piece,{x:cw/2,y:34,'text-anchor':'middle'});
    set(c.pos,{x:cw/2,y:64,'text-anchor':'middle'});
    set(c.id,{x:cw/2,y:96,'text-anchor':'middle'});
    pressed(c.g,on);centers.push(x+cw/2);
   });
   set(link,{d:`M${centers[1]} ${y+h} V${y+h+14} H${centers[3]} V${y+h}`,opacity:linkOn?'1':'0'});
   show(pairLabel,linkOn?1:0);
   text(pairLabel,(centers[1]+centers[3])/2,y+h+35,'同一 ID 100134 · 位置 p2 / p4');set(pairLabel,{'text-anchor':'middle'});
  }
  const py=m?374:252,ph=m?150:200;
  set(panel,{x:m?16:24,y:py,width:m?328:672,height:ph});
  text(panelTitle,m?32:40,py+(m?22:26),p<.5?(m?'点一张卡查看 ID 与位置':'先看 5 次出现；点一张卡查看 ID 与位置'):'选中：p'+(selected+1)+' · '+EX2.pieces[selected]);
  const sw=m?150:300,sh=m?68:76,s1=m?28:40,s2=m?184:360,sy=py+(m?32:40);
  set(stat1Bg,{x:s1,y:sy,width:sw,height:sh});set(stat2Bg,{x:s2,y:sy,width:sw,height:sh});
  const reveal=Math.max(phase(p,.8,1.6),pinned?1:0);
  text(idLabel,s1+10,sy+(m?16:17));text(idValue,s1+10,sy+(m?40:46),'ID '+EX2.ids[selected]);
  const idMsg=sameSelected?(m?'与 p4 同 ID':'与另一次“学习”相同'):(m?'本次唯一出现':'这个 ID 本次只出现一次');
  const posMsg=sameSelected?(selected===3?(m?'第 2 次出现':'两次“学习”的第 2 次'):(m?'第 1 次出现':'两次“学习”的第 1 次')):(m?'第 '+(selected+1)+' 个位置':'第 '+(selected+1)+' 个输入位置');
  text(idNote,s1+10,sy+(m?58:66),idMsg);
  text(posLabel,s2+10,sy+(m?16:17));text(posValue,s2+10,sy+(m?40:46),'p'+(selected+1));
  text(posNote,s2+10,sy+(m?58:66),posMsg);
  set(idValue,{opacity:(.35+.65*reveal).toFixed(3)});set(posValue,{opacity:(.35+.65*reveal).toFixed(3)});
  set(idNote,{opacity:(.3+.7*reveal).toFixed(3)});set(posNote,{opacity:(.3+.7*reveal).toFixed(3)});
  text(pairText,m?32:40,py+(m?126:142),sameSelected?
   (m?'同一 ID：p2、p4 是两次出现。':'同一个 ID 指向同一词表项；两次出现的位置分别是 p2 与 p4。'):
   (m?'ID 是词表项；p 是出现次序。':'ID 是词表项；位置是出现次序，不能合并。'));
  text(panelNote,m?32:40,py+(m?143:164),m?'位置只记次序，不是 token ID。':'位置编号只记录次序；它不是 token ID，也不改变词表项。');
  text(note,m?16:24,m?540:476,m?'真实 token ID 来自 Qwen2.5 固定样例；':'真实 token ID 来自 Qwen2.5 固定样例；p 只是位置，不是 ID。');
  text(note2,16,554,m?'p 只是出现位置，不是 token ID。':'');
  show(note2,m?1:0);
 }
 return draw;
}

/* --------------------------------------------------------- 03 lookup a row */
function lookupBoard(H){
 const {txt,text,box,show,set,node,path,wire}=H;
 const bg=node('rect',{class:'tok-board'}),band=node('rect',{class:'tok-band'});
 const panel=node('rect',{class:'tok-panel'}),link=path('tok-link tok-link-amber');
 const rows=TOY.map((r,i)=>{
  const g=node('g',{class:'tok-card'}),b=node('rect',{rx:6,width:360,height:52});
  const id=node('text',{class:'tok-rowid',x:24,y:33},String(r.id));
  const v=node('text',{class:'tok-vec',x:88,y:33},fmt(r.v));
  const nth=node('text',{class:'tok-stat-note',x:330,y:33,'text-anchor':'end'},'第 '+(i+1)+' 行');
  g.append(b,id,v,nth);
  hit(g,`ID ${r.id}，读取第 ${i+1} 行的 ${fmt(r.v)}`,()=>{selected=i;pinHere(pin,lastP);repaint();});
  return {g,b,id,v,nth};
 });
 const title=txt('','tok-title'),sub=txt('','tok-sub'),kicker=txt('','tok-kicker');
 const tableTitle=txt('E · 玩具表（4 行 × 3 列）','tok-small');
 const outTitle=txt('结果 · 取出一整行','tok-panel-title');
 const outId=txt('','tok-big'),outLine=txt('','tok-note'),outVec=txt('','tok-out'),outNote=txt('','tok-note'),outShape=txt('','tok-small');
 const note=txt('ID 0–3 是玩具表行号；与上一概念的真实 token ID 分开。','tok-note');
 let selected=1,pin={p:null,epoch:-1},lastP=0,lastM=false;
 function repaint(){draw(lastP,lastM);}
 function draw(p,m){
  lastP=p;lastM=m;const W=m?360:720,HT=m?560:500;
  box(bg,0,0,W,HT);box(band,0,0,W,62);
  text(title,16,27,m?'Embedding：按 ID 取一行':'Embedding：按 ID 取出一行参数');
  text(sub,16,47,m?'行号就是 ID；取出整行':'ID 是行号；取出一整行，得到 1×3 向量');
  text(kicker,16,m?76:82,'输入 · 选择一个 ID，按行号读取');
  const tx=m?16:24,ty=m?112:122,tw=m?328:360,th=m?44:52,step=th+(m?5:6);
  text(tableTitle,tx,m?98:104);
  TOY.forEach((r,i)=>{
   const y=ty+i*step,on=i===selected;
   set(rows[i].g,{transform:`translate(${tx} ${y})`});
   set(rows[i].b,{width:tw,height:th,class:'tok-row'+(on?' is-on':'')});
   set(rows[i].id,{x:m?20:24,y:m?28:33});
   set(rows[i].v,{x:m?72:88,y:m?28:33});
   set(rows[i].nth,{x:m?310:330,y:m?28:33,'text-anchor':'end'});
   pressed(rows[i].g,on);
  });
  const px=m?16:420,py=m?320:122,pw=m?328:276,ph=m?170:240;
  set(panel,{x:px,y:py,width:pw,height:ph});
  const pinned=isPinned(pin,p);
  const reveal=Math.max(phase(p,.8,1.8),pinned?1:0);
  const vecReveal=Math.max(phase(p,1.6,2.4),pinned?1:0);
  text(outTitle,px+12,py+24);
  text(outId,px+12,py+(m?60:62),'ID '+selected);set(outId,{opacity:(.35+.65*reveal).toFixed(3)});
  text(outLine,px+12,py+(m?86:90),'→ 第 '+(selected+1)+' 行');set(outLine,{opacity:(.35+.65*reveal).toFixed(3)});
  text(outVec,px+12,py+(m?120:124),fmt(TOY[selected].v));set(outVec,{opacity:(.25+.75*vecReveal).toFixed(3)});
  text(outNote,px+12,py+(m?146:154),'整行读取；不做 one-hot，不复制整张表');set(outNote,{opacity:(.4+.6*reveal).toFixed(3)});
  text(outShape,px+12,py+(m?166:218),'输出形状 1 × 3');set(outShape,{opacity:(.35+.65*reveal).toFixed(3)});
  const sy=ty+selected*step+th/2;
  if(m){wire(link,`M348 ${sy} V316`,1);}else{wire(link,`M${tx+tw} ${sy} H${px}`,1);}
  text(note,16,m?520:386,'ID 0–3 是玩具表行号；与上一概念的真实 token ID 分开。');
 }
 return draw;
}

/* ------------------------------------------------------- 04 train / infer */
function boundaryBoard(H){
 const {txt,text,box,show,set,node,path,wire}=H;
 const bg=node('rect',{class:'tok-board'}),band=node('rect',{class:'tok-band'});
 const panel=node('rect',{class:'tok-panel'});
 const rows=TOY.map((r,i)=>{
  const g=node('g',{class:'tok-card'}),b=node('rect',{rx:6,width:300,height:48});
  const id=node('text',{class:'tok-rowid',x:20,y:31},String(r.id));
  const v=node('text',{class:'tok-vec',x:64,y:31},fmt(r.v));
  const tag=node('text',{class:'tok-stat-note',x:280,y:31,'text-anchor':'end'});
  const before=node('text',{class:'tok-note',x:64,y:43});
  g.append(b,id,v,tag,before);return {g,b,id,v,tag,before};
 });
 const update=path('tok-link tok-link-amber'),read=path('tok-link tok-link-mint');
 const lock=node('g',{class:'tok-lock'});const lockB=node('rect',{rx:6,width:48,height:24}),lockT=node('text',{x:24,y:17,'text-anchor':'middle'},'只读');lock.append(lockB,lockT);
 const tabTrain=button(H,0,0,148,30,'训练 / 更新',()=>{override='train';interacted=true;repaint();});
 const tabInfer=button(H,0,0,148,30,'推理 / 只读',()=>{override='infer';interacted=true;repaint();});
 const title=txt('','tok-title'),sub=txt('','tok-sub'),kicker=txt('','tok-kicker');
 const tableLabel=txt('','tok-small');
 const modeTitle=txt('','tok-panel-title'),line1=txt('','tok-note'),line2=txt('','tok-note');
 const vals1=txt('','tok-mono'),vals2=txt('','tok-mono'),panelNote=txt('','tok-note'),note=txt('','tok-note');
 let override=null,interacted=false,lastP=0,lastM=false;
 function repaint(){draw(lastP,lastM);}
 function draw(p,m){
  lastP=p;lastM=m;const mode=override||(p>=3?'train':'infer'),train=mode==='train';const W=m?360:720,HT=m?560:500;
  box(bg,0,0,W,HT);box(band,0,0,W,62);
  text(title,16,27,m?'同一条参数：可写 / 只读':'Embedding 的学习与推理：可写与只读');
  text(sub,16,47,m?'训练可更新；推理只读':'同一行参数来自训练；普通推理不因一句话改表');
  text(kicker,16,m?76:82,'输入 · ID 1 读取第 2 行；训练时才写回');
  const tx=m?16:24,ty=m?104:108,tw=m?328:300,th=m?42:48,step=th+(m?5:6);
  TOY.forEach((r,i)=>{
   const y=ty+i*step,on=i===1;
   set(rows[i].g,{transform:`translate(${tx} ${y})`});
   set(rows[i].b,{width:tw,height:th,class:'tok-row'+(on?' is-on':'')+(train&&on?' is-updated':'')});
   set(rows[i].id,{x:m?18:20,y:m?27:31});
   if(train&&on){
    text(rows[i].v,m?56:64,m?22:26,fmt(UPDATED,2));set(rows[i].v,{class:'tok-vec tok-updated'});
    text(rows[i].tag,tw-10,m?22:26,'更新后');show(rows[i].tag,1);
    text(rows[i].before,m?56:64,m?36:41,'更新前 '+fmt(r.v));show(rows[i].before,1);
   }else{
    text(rows[i].v,m?56:64,m?27:31,fmt(r.v));set(rows[i].v,{class:'tok-vec'});
    show(rows[i].tag,on?1:0);
    if(on)text(rows[i].tag,tw-10,m?27:31,'原值');
    show(rows[i].before,0);
   }
  });
  text(tableLabel,tx,m?96:100,train?'E · 第 2 行显示更新后（原值见行内）':'E · 第 2 行：初始原值 · 只读');
  const px=m?16:360,py=m?310:108,pw=m?328:336,ph=m?210:250;
  set(panel,{x:px,y:py,width:pw,height:ph});
  text(modeTitle,px+12,py+(m?22:26),train?'训练 · 更新可写':'推理 · 只读');
  placeButton(H,tabTrain,px+12,py+(m?36:40),148,30,'训练 / 更新',train);
  placeButton(H,tabInfer,px+168,py+(m?36:40),148,30,'推理 / 只读',!train);
  text(line1,px+12,py+(m?78:92),train?'误差产生 Δ，写回第 2 行（本例）':'本次输入只读取第 2 行；不写回 E');
  text(line2,px+12,py+(m?100:114),train?'表中第 2 行显示更新后的值':'参数保持初始 E；这句不训练');
  const vy=m?py+124:py+146;
  text(vals1,px+12,vy,train?'更新前 '+fmt(TOY[1].v):'读取 '+fmt(TOY[1].v));
  text(vals2,px+12,vy+22,train?'更新后 '+fmt(UPDATED,2):'没有写回；E 不变');
  set(vals2,{class:train?'tok-mono':'tok-note'});
  text(panelNote,px+12,py+(m?184:214),train?'Δ 是教学假设；不运行优化器。':'写回路径被锁住；推理只读。');
  set(lock,{transform:`translate(${px+pw-62} ${m?py+4:py+8})`,opacity:train?'0':'1'});set(lockB,{width:48,height:m?24:36});set(lockT,{x:24,y:m?17:23});
  if(train){
   const d=m?`M350 ${py} V${ty+th*1.5} H${tx+tw}`:`M${px} ${py+198} C${px-24} ${py+198},${tx+tw+24} ${ty+th*1.5},${tx+tw} ${ty+th*1.5}`;
   wire(update,d,1);set(read,{opacity:'0'});
  }else{
   const d=m?`M${tx+tw} ${ty+th*1.5} H350 V${py-4}`:`M${tx+tw} ${ty+th*1.5} H${px}`;
   wire(read,d,1);set(update,{opacity:'0'});
  }
  text(note,16,m?544:470,'训练/推理是同一初始 E 的两个独立对照，不是连续写表。');
 }
 return draw;
}

/* ------------------------------------------------------ 05 contract / template */
function contractBoard(H){
 const {txt,text,box,show,set,node,path}=H;
 const bg=node('rect',{class:'tok-board'}),band=node('rect',{class:'tok-band'});
 const link1=path('tok-link tok-link-mint'),link2=path('tok-link tok-link-mint');
 const panel=node('rect',{class:'tok-panel'}),tpl=node('rect',{class:'tok-template'});
 const blocks=[0,1,2].map(i=>{
  const g=node('g',{class:'tok-block'}),b=node('rect',{rx:8,width:200,height:70});
  const t=node('text',{class:'tok-block-t',x:12,y:26},['Tokenizer A','A 词表 / ID','A 权重 E'][i]);
  const s=node('text',{class:'tok-block-s',x:12,y:49},['文本 → 片段','片段 ↔ 编号','ID → 一行参数'][i]);
  g.append(b,t,s);return {g,b,t,s};
 });
 const tabA=button(H,0,0,92,32,'匹配 A',()=>{mismatch=false;pinHere(pin,lastP);repaint();});
 const tabB=button(H,0,0,150,32,'只换 tokenizer',()=>{mismatch=true;pinHere(pin,lastP);repaint();});
 const tabT=button(H,0,0,140,32,'模板 关',()=>{template=!template;pinHere(pin,lastP);repaint();});
 const title=txt('','tok-title'),sub=txt('','tok-sub'),kicker=txt('','tok-kicker');
 const status=txt('','tok-status');
 const inputTitle=txt('输入文本','tok-small'),inputWord=txt('学习','tok-input-word');
 const templateTitle=txt('','tok-small'),templateText=txt('','tok-template-text');
 const mapTitle=txt('','tok-panel-title'),map1=txt('','tok-mono'),map2=txt('','tok-mono');
 const mapNote1=txt('','tok-note'),mapNote2=txt('','tok-note');
 const note=txt('模板格式依模型而定；示意格式不写通用 special ID。','tok-note');
 let mismatch=false,template=false,pin={p:null,epoch:-1},lastP=0,lastM=false;
 function repaint(){draw(lastP,lastM);}
 function draw(p,m){
  lastP=p;lastM=m;const W=m?360:720,HT=m?560:500;
  box(bg,0,0,W,HT);box(band,0,0,W,62);
  text(title,16,27,m?'三件套要匹配；模板另算':'Tokenizer、词表与权重：三件套要匹配');
  text(sub,16,47,m?'模板是包装，不是语义':'模板是模型约定的包装；三件套匹配才是硬条件');
  text(kicker,16,m?76:82,'三件套：tokenizer — 词表 / ID — 权重');
  const bw=m?104:200,bx=m?12:24,by=m?88:92,gap=m?8:34;
  blocks.forEach((b,i)=>{
   const x=bx+i*(bw+gap);
   const labels=i===0?(mismatch?'Tokenizer B':'Tokenizer A'):['A 词表 / ID','A 权重 E'][i-1];
   const subs=m?['文本 → 片段','片段 ↔ ID','ID → 向量'][i]:['文本 → 片段','片段 ↔ 编号','ID → 一行参数'][i];
   set(b.g,{transform:`translate(${x} ${by})`});
   set(b.b,{width:bw,height:m?64:70,class:'tok-block'+(mismatch&&i===0?' is-bad':'')});
   set(b.t,{x:12,y:m?25:26});if(b.t.textContent!==labels)b.t.textContent=labels;
   set(b.s,{x:12,y:m?47:49});if(b.s.textContent!==subs)b.s.textContent=subs;
  });
  const y1=by+(m?32:35),x1=bx+bw,x2=x1+gap+bw,x3=x2+gap;
  set(link1,{d:`M${x1} ${y1} H${x1+gap}`,class:'tok-link '+(mismatch?'tok-link-coral':'tok-link-mint')});
  set(link2,{d:`M${x2} ${y1} H${x2+gap}`,class:'tok-link '+(mismatch?'tok-link-coral':'tok-link-mint')});
  placeButton(H,tabA,m?16:24,m?168:180,92,32,'匹配 A',!mismatch);
  placeButton(H,tabB,m?116:124,m?168:180,150,32,'只换 tokenizer',mismatch);
  placeButton(H,tabT,m?16:282,m?206:180,140,32,template?'模板 开':'模板 关',template);
  text(status,m?276:600,m?226:196,mismatch?'不对应':'三件套匹配');set(status,{class:'tok-status '+(mismatch?'bad':'good')});set(status,{'text-anchor':m?'end':'start'});
  const ix=m?16:24,iy=m?246:222,iw=m?328:300,ih=m?108:112;
  box(tpl,ix-4,iy-6,iw+8,ih+4);show(tpl,template?1:0);
  text(inputTitle,ix+12,iy+22);
  text(inputWord,ix+24,iy+(m?56:58));
  text(templateTitle,ix+12,iy+(m?82:86),template?'聊天模板 · 示意格式':'未加模板');
  text(templateText,ix+12,iy+(m?102:104),template?'[角色: 用户] 学习 [轮次结束]':'原文直接进入 tokenizer');
  const px=m?16:350,py=m?368:222,pw=m?328:346,ph=m?156:184;
  set(panel,{x:px,y:py,width:pw,height:ph});
  const pinned=isPinned(pin,p);
  const reveal=Math.max(phase(p,.4,1.2),pinned?1:0);
  const reveal2=Math.max(phase(p,1.2,2),pinned?1:0);
  text(mapTitle,px+14,py+26,mismatch?'B · 只换 tokenizer':'A · 匹配');
  text(map1,px+14,py+(m?54:58),mismatch?'B 词表：学习 → ID 2':'学习 → ID 1');
  text(map2,px+14,py+(m?80:86),mismatch?'沿用 A 权重：E[2] = [0, 1, 0]':'E[1] = [0.2, 0.8, -0.1]');
  set(map1,{opacity:(.35+.65*reveal).toFixed(3)});set(map2,{opacity:(.35+.65*reveal2).toFixed(3)});
  text(mapNote1,px+14,py+(m?106:116),mismatch?'B 词表内部一致：学习 的编号是 2。':'tokenizer A、A 词表与 A 权重相互对应。');
  text(mapNote2,px+14,py+(m?128:140),mismatch?'但 A 权重第 3 行（ID 2）是机器行。':'ID 1 取出的就是学习行。');
  set(mapNote1,{opacity:(.35+.65*reveal).toFixed(3)});set(mapNote2,{opacity:(.35+.65*reveal2).toFixed(3)});
  text(note,16,m?544:472,m?'模板格式依模型而定；不写通用 special ID。':'模板格式依模型而定；示意格式不写通用 special ID。');
 }
 return draw;
}

const choices=[
 {concept:'bpe',mv:10,name:'01 分词与 BPE',title:'分词与 BPE：把句子变成可复用单位',
  intro:'先看一句话为什么不能整句作为一个 ID；再按固定规则只合并相邻单位，单位数从 6 变为 3。',
  boundary:'合并规则是训练后固定的，推理时只应用，不按当前句的词义或频率临时改变。本例从字符开始；字节级 BPE 的起始单位是字节。',
  data:{},frames:frameList(
   ['输入是原句','整句组合太多，不可能给每句新话准备一个 ID。先把它切成 6 个可复用单位。'],
   ['规则①：合并相邻的“机”“器”','只合并相邻单位，得到“机器”；单位数 6 → 5。规则来自训练语料，不是按词义判断。'],
   ['规则②：合并相邻的“学”“习”','得到“学习”；单位数 5 → 4。每一步只改变相邻的这一对。'],
   ['规则③：再合并“机器”“学习”','得到“机器学习”；单位数 4 → 3。规则顺序固定。'],
   ['结果是 3 个 token','我 / 爱 / 机器学习。它们下一步才去词表查 ID；推理不会重新学习这套规则。']
  ),stepDurations:[1.1,1.7,1.7,1.8]},
 {concept:'position',mv:11,name:'02 ID 与位置',title:'Token ID 与位置：编号相同，位置不同',
  intro:'同一句话里两次出现“学习”。它们查到同一个真实 token ID，但占据两个不同的位置。',
  boundary:'ID 是词表索引，position 是出现次序；p1…p5 只表示位置，不是 token ID。真实 ID 来自 Qwen2.5-0.5B-Instruct 固定 tokenizer 的已有样例，与后面的 0–3 玩具表分开。',
  data:EX2,frames:frameList(
   ['输入是 5 次出现','原句被切成 5 个片段；每个片段都有自己的出现位置。'],
   ['两次“学习”查到同一个 ID','词表把“学习”映射为 100134；两次出现都指向同一项。'],
   ['但位置分别是 p2、p4','ID 没变，位置变了。它们不是同一个输入位置。'],
   ['点另一张“学习”卡','选中第二次出现：ID 仍是 100134，位置显示 p4。'],
   ['ID 与位置分开保留','后续计算还要结合位置和上下文；不能把 p 标签当成 ID。']
  ),stepDurations:[1,1.5,1.5,1.4]},
 {concept:'lookup',mv:12,name:'03 按行查表',title:'Embedding：按 ID 取出一行参数',
  intro:'Embedding 表有 4 行、每行 3 个数。ID 是行号；取出一整行，得到形状 1×3 的向量。',
  boundary:'这一页的 ID 0–3 是玩具词表行号，不是上一概念的真实 token ID。向量是教学假设，不是 Qwen 权重；不构造巨大 one-hot。',
  data:{rows:TOY},frames:frameList(
   ['先看玩具表 E','4 行 × 3 列；每行是一组可学习参数。行号 0–3 只是索引，大小没有语义。'],
   ['取 ID 1，定位第 2 行','ID 不是把数字拆开计算，而是直接作为行索引。'],
   ['读取这一整行','取出 [0.2, 0.8, -0.1]；不做 one-hot，也不复制整张表。'],
   ['结果是 1×3 向量','这一行按输入位置交给后续模型。形状来自表宽，不是 ID 的长度。'],
   ['换一行也同步更新','点 ID 2 或 3，表中对应行与右侧输出一起变化。']
  ),stepDurations:[1,1.3,1.4,1.3]},
 {concept:'boundary',mv:13,name:'04 训练 / 推理',title:'Embedding 的学习与推理：可写与只读',
  intro:'同一行参数来自训练。训练时误差可以写回这一行；普通推理只读取它，不会因为当前一句话改表。',
  boundary:'只展示“参数会不会更新”的边界，不展开反向传播或优化器。训练与推理都从同一个初始 E 出发，是两个独立对照例；训练态的第 2 行显示教学更新值，推理态读取初始值且不写回，两者不是在同一张持久参数表上连续切换。',
  data:{rows:TOY},frames:frameList(
   ['同一行，两种状态','这行参数既能被训练更新，也能被推理读取。先分清当前是哪一种。'],
   ['推理：按 ID 1 读取第 2 行','读取只发生一次；拿到向量后送入模型前向计算。'],
   ['推理没有写回','这句话不会改写 E；写回路径被锁住。'],
   ['训练：误差更新同一行','训练例从同一初始 E 出发：第 2 行显示更新后的值；只画 Δ，不展开算法。'],
   ['边界是“是否写参数”','训练例可写、推理例只读；两者是独立对照，不是同一张表连续切换。不要把一次请求说成一次训练。']
  ),stepDurations:[1,1.4,1.4,1.6]},
 {concept:'contract',mv:14,name:'05 模板与匹配',title:'Tokenizer、词表与权重：三件套要匹配',
  intro:'tokenizer 决定片段与 ID，词表/权重决定 ID 对应哪一行参数。聊天模板是另一层包装，必须按模型约定使用。',
  boundary:'模板只是示意格式；控制标记与特殊 ID 依模型而异，不编造通用 special ID。模板会改变实际输入序列，但不改变“学习”这个词元在 A 约定下的向量含义。换 tokenizer 后若沿用 A 权重，ID 2 会落到 A 的机器行。',
  data:{rows:TOY},frames:frameList(
   ['三件套必须对得上','tokenizer、词表/ID、模型权重是同一套契约；模板单独一层。'],
   ['匹配：学习 → ID 1 → 学习行','A 的词表与权重对应，取到学习那一行。'],
   ['只换 tokenizer：学习 → ID 2','B 的词表内部把“学习”编为 ID 2；若沿用 A 权重，会取到 A 的第 3 行（ID 2），那是机器行。'],
   ['模板是另一层包装','开关只改变角色/内容怎样包装；核心 token 仍是“学习”。'],
   ['模板格式依模型而定','示意格式不写通用 special ID；三件套匹配才是硬条件。']
  ),stepDurations:[1,1.5,1.7,1.5]}
];

function legacyChoice(scene,mv,index){
 if(!scene)return null;
 return {...scene,legacy:true,mv,name:'旧样例 '+(index+1)+' · '+scene.data.text,title:scene.title,
  intro:'这是旧链接保留的固定样例。可用上方的概念入口继续观察分词、位置、查表、训练/推理与模板匹配。',frames:scene.frames.map(f=>({title:f.title,caption:f.caption}))};
}
const oldLessons=(root.TeachingLessons&&root.TeachingLessons['f-tokens'])||[];
const legacy=[
 legacyChoice(oldLessons.find(s=>s.data&&s.data.text==='我想学习人工智能。'),0,0),
 legacyChoice(oldLessons.find(s=>s.data&&s.data.text==='我想学习机器学习。'),1,1)
].filter(Boolean);

V['f-tokens']=(svg,viewport,s)=>{
 const film=svg.closest('.mechanism-film');
 if(film)film.classList.toggle('is-tokens-legacy',!!s.legacy);
 if(s.legacy&&legacyVisual)return legacyVisual(svg,viewport,s);
 watchProgress(svg);
 pauseFilm=()=>{const b=film?.querySelector('[data-motion-action=play]');if(b&&b.getAttribute('aria-pressed')==='true')b.click();};
 return board(svg,viewport,'tokens-'+s.concept,m=>m?560:500,H=>{
  if(s.concept==='bpe')return bpeBoard(H);
  if(s.concept==='position')return positionBoard(H);
  if(s.concept==='lookup')return lookupBoard(H);
  if(s.concept==='boundary')return boundaryBoard(H);
  return contractBoard(H);
 });
};
root.TeachingLessons=root.TeachingLessons||{};
root.TeachingLessons['f-tokens']=choices.concat(legacy);
})(typeof globalThis==='object'?globalThis:this);
