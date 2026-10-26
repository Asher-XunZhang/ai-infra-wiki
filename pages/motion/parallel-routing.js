(()=>{'use strict';
const {board,phase,lerp}=window.TeachingDrawing;
window.DataParallelVisual=(svg,viewport,s)=>board(svg,viewport,'replicated-model',()=>860,H=>{
 const {txt,rect,path,text,box,show,wire,token,move}=H;
 const title=txt('同一模型：L1(x)=2x → L2(h)=h+1','teach-heading'),legend=txt('x 是输入特征；L1 / L2 是两层教学运算','teach-small');
 const source=[token('dp-model-L1','L1 · ×2',100),token('dp-model-L2','L2 · +1',100)];
 const models=[0,1].map(r=>({panel:rect(),name:txt('','teach-heading'),layers:[token('dp-L1-'+r,'L1 · ×2',100),token('dp-L2-'+r,'L2 · +1',100)],wire:path(),calc:txt('','teach-number'),result:txt('','teach-small')}));
 const req=[token('dp-R1','R1 · x=1',112,'teach-token teach-color-0'),token('dp-R2','R2 · x=3',112,'teach-token teach-color-1')];
 const note=txt('','teach-heading'),step=txt('','teach-small'),returnLines=[path(),path()];
 return(p,m)=>{const x=[18,m?190:382],w=m?152:320;
  text(title,18,34);if(m)title.setAttribute('font-size','14');text(legend,18,62);
  move(source[0],m?30:150,90,1-.65*phase(p,1));move(source[1],m?200:460,90,1-.65*phase(p,1));
  models.forEach((r,i)=>{const u=phase(p,0);box(r.panel,x[i],200,w,470);show(r.panel,u);
   // The replica heading and rank sit above the layer stack. The copied L1
   // token and then the request both travel through that band, so fade each
   // label only while one of those moving rectangles intersects it. The rank
   // stays visible while L1 passes and the heading stays visible while the
   // request passes, so the panel keeps an identity label at every frame.
   const labelX=x[i]+10;
   const layerTops=[lerp(90,285,u),lerp(90,450,u)];
   const reqTop=p<2?lerp(150,342,phase(p,1)):p<3?lerp(342,507,phase(p,2)):p<4?507:lerp(507,750,phase(p,4));
   const clear=(top,bottom)=>![...layerTops,reqTop].some(y=>y+34>top&&y<bottom);
   text(r.name,labelX,230,'完整副本 '+(i?'B':'A'));show(r.name,clear(210,232)?u:0);
   r.layers.forEach((l,j)=>move(l,lerp(m?(j?200:30):(j?460:150),x[i]+16,u),lerp(90,285+j*165,u),u));
   wire(r.wire,`M${x[i]+66} 322 V442`,p>=1?1:0);
   const ry=reqTop;
   move(req[i],x[i]+12,ry);req[i].t.textContent=p<3?`R${i+1} · x=${i?3:1}`:p<4?`R${i+1} · h=${i?6:2}`:`R${i+1} · y=${i?7:3}`;
   text(r.calc,x[i]+12,580,p<3?(p>=2?`${i?3:1} × 2 → ${i?6:2}`:''):p<4?`${i?6:2} + 1 → ${i?7:3}`:`y = ${i?7:3}`);show(r.calc,p<4?1:0);
   text(r.result,labelX,262,'P'+i+' / Rank '+i+' / GPU '+i);show(r.result,clear(248,266)?u:0);
   wire(returnLines[i],`M${x[i]+66} 670 V690 M${x[i]+66} 725 V742`,phase(p,4));
  });
  text(note,18,816,p<4?'两条请求同时使用各自的完整副本':'分别归还给 R1 / R2 的调用方');
  text(step,18,848,p>=5?'输出为 3 和 7；不把两条请求求和。':'完整副本复制参数；请求在副本间分工。');
  svg.dataset.parallelKind='dp';
 };
});
/* EP: one board keeps the same 8 top-2 routes while the learner switches the
   placement (one GPU with four expert groups, or two GPUs with two each).
   The animated branches, returns and weighted combine are pure functions of p,
   so stepping backwards restores the exact intermediate state. */
window.ExpertParallelVisual=(svg,viewport,s)=>{
 const film=viewport.closest('.mechanism-film');if(film)film.classList.add('is-ep');
 const inner=board(svg,viewport,'expert-routing',m=>m?660:600,H=>{
 const {txt,rect,path,text,box,show,set,token,move}=H,d=s.data,sel=d.selected;
 const NS='http://www.w3.org/2000/svg';
 const raw=(tag,attrs={},value)=>{const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(value!==undefined)e.textContent=value;return e;};
 const layout=m=>m?{
  W:360,H:660,titleY:22,legendY:40,
  mode:{y:50,h:70,w:168,x:[8,184],cw:30,ch:18,gap:4,groupGap:8,titleY:16,chipY:24,footerY:60},
  headY:136,
  mx:{ownerX:8,ownerW:14,tokenX:24,tokenW:34,x:62,headY:150,rowY:158,rowH:20,gapY:4,cw:38,gapX:3},
  crossLegendY:650,sumY:[274,290],
  t1:{x:234,y:152,w:118,h:96},
  exp:{panelY:300,before:{x:8,y:300,w:344,h:190},top:{x:8,y:300,w:344,h:88},bottom:{x:8,y:396,w:344,h:94},headY:314,bottomHeadY:408,cardX:[16,184,16,184],cardY:[318,318,412,412],cardW:160,cardH:64},
  bot:{dockX:8,dockY:500,dockW:170,dockH:96,formulaX:184,formulaY:500,formulaW:168,formulaH:96,bayX:[18,98],bayY:534,bayW:70,bayH:32,chipW:36,chipH:22,fTitleY:516,fMainY:540,fSubY:564,weightY:580},
  notes:{x:12,y:[616,636]}
 }:{
  W:720,H:600,titleY:26,legendY:46,
  mode:{y:60,h:80,w:338,x:[18,364],cw:48,ch:24,gap:6,groupGap:14,titleY:18,chipY:28,footerY:68},
  headY:158,
  mx:{ownerX:14,ownerW:18,tokenX:38,tokenW:52,x:96,headY:180,rowY:192,rowH:22,gapY:4,cw:46,gapX:4},
  crossLegendY:592,sumY:[],
  t1:{x:0,y:0,w:0,h:0},
  exp:{panelY:304,before:{x:14,y:304,w:692,h:100},top:{x:14,y:304,w:336,h:100},bottom:{x:354,y:304,w:352,h:100},headY:320,bottomHeadY:320,cardX:[24,184,370,530],cardY:[328,328,328,328],cardW:152,cardH:64},
  bot:{dockX:18,dockY:414,dockW:360,dockH:90,formulaX:394,formulaY:414,formulaW:308,formulaH:90,bayX:[40,146],bayY:454,bayW:82,bayH:32,chipW:42,chipH:22,fTitleY:436,fMainY:462,fSubY:486,weightY:500},
  notes:{x:18,y:[526,548,570]}
 };
 let mode='after',lastP=0,lastM=false,draw=null;
 const fmt=v=>String(Math.round(v*100)/100),clamp01=v=>Math.max(0,Math.min(1,v));
 try{if(new URL(location.href).searchParams.get('ep')==='before')mode='before';}catch(error){}
 const title=txt('','teach-heading'),legend=txt('','teach-small');
 const routeHead=txt('','teach-small'),routeHint=txt('','teach-small');
 const modeGroups=[];
 for(const key of ['before','after']){
  const g=raw('g',{class:'ep-mode',role:'button',tabindex:'0','aria-pressed':'false'});
  const bg=raw('rect',{class:'teach-panel ep-mode-panel',rx:7});g.append(bg);
  const name=raw('text',{class:'teach-heading'});g.append(name);
  const current=raw('text',{class:'teach-small ep-current'},'当前');g.append(current);
  const foot=raw('text',{class:'teach-small'});g.append(foot);
  const chips=[];
  for(let e=0;e<4;e++){
   const cg=raw('g',{class:'teach-token ep-mode-chip'});
   const cr=raw('rect',{rx:4}),ct=raw('text',{'text-anchor':'middle'},'E'+e);
   cg.append(cr,ct);g.append(cg);chips.push({g:cg,r:cr,t:ct});
  }
  svg.append(g);modeGroups.push({key,g,bg,name,current,foot,chips});
 }
 const setMode=next=>{if(next===mode)return;mode=next;if(draw)draw(lastP,lastM);svg.dispatchEvent(new CustomEvent('parallel-ep-mode',{bubbles:true,detail:{mode}}));};
 for(const m of modeGroups){
  m.g.addEventListener('click',()=>setMode(m.key));
  m.g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.stopPropagation();setMode(m.key);}});
 }
 const owners=[0,1,2,3].map(i=>token('ep-owner-'+i,'0',18,'teach-token ep-owner-chip'));
 const inputs=[0,1,2,3].map(i=>token('ep-input-'+i,'t'+(i+1),52,'teach-token teach-color-'+i));
 const focus=rect('teach-focus');
 const heads=[0,1,2,3].map(i=>txt('E'+i,'teach-small'));
 const cells=Array.from({length:16},()=>rect('teach-mask'));
 const divider=path('ep-divider');
 const routeLegend=txt('','teach-small');
 const ledgerPanel=rect('teach-panel'),ledgerTitle=txt('','teach-text');
 const ledgerHeads=[txt('未采用','teach-small'),txt('采用后','teach-small')];
 const ledgerRows=['GPU 数','每卡专家组','跨卡分支','激活负载'].map(v=>txt(v,'teach-small'));
 const ledgerBefore=['1','4','0','8'].map(v=>txt(v,'teach-text')),ledgerAfter=['2','2+2','4 去 / 4 回','4/4'].map(v=>txt(v,'teach-text'));
 const sum1=txt('','teach-small'),sum2=txt('','teach-small');
 const t1Panel=rect('teach-panel'),t1Title=txt('','teach-text'),t1Line=[txt('','teach-small'),txt('','teach-small'),txt('','teach-small'),txt('','teach-small')];
 const panels=[rect('teach-panel'),rect('teach-panel'),rect('teach-panel')];
 const panelTitles=[txt('','teach-small'),txt('','teach-small'),txt('','teach-small')];
 const rankDivider=path('ep-divider');
 const cards=[0,1,2,3].map(e=>({bg:rect('teach-panel ep-expert'),name:txt('','teach-text ep-expert-name'),load:txt('','teach-small'),math:txt('','teach-small')}));
 const branches=d.routes.flatMap((es,t)=>es.map((e,k)=>({t,e,k,cross:d.owner[e]!==d.tokenOwner[t]})));
 const slots=[0,1,2,3].map(e=>branches.filter(b=>b.e===e));
 const dispatch=branches.map(b=>{
  const a=token('ep-dispatch-'+b.t+'-'+b.e,'t'+(b.t+1),24,'teach-token teach-color-'+b.t);
  set(a.r,{height:14,rx:4});set(a.t,{x:12,y:10,'font-size':'10'});
  return {...b,a};
 });
 const dockPanel=rect('teach-panel'),dockTitle=txt('','teach-text');
 const bays=[0,1].map(i=>rect('teach-gate')),bayLabels=[0,1].map(i=>txt('E'+sel[i]+' 分支','teach-small')),weights=[0,1].map(i=>txt('','teach-small ep-cost'));
 const formulaPanel=rect('teach-panel'),formulaTitle=txt('','teach-small'),formulaMain=txt('','teach-text'),formulaSub=txt('','teach-small');
 const contrib=[0,1].map(i=>{const a=token('ep-contrib-'+i,fmt(d.outputs[i]*d.weights[i]),36,'teach-token teach-color-0');set(a.r,{height:22,rx:4});set(a.t,{x:18,y:15,'font-size':'11'});return a;});
 const plus=txt('+','teach-heading'),mergeRing=raw('circle',{r:0,fill:'none',stroke:'#2a8478','stroke-width':2});
 const lane=path('ep-lane');
 const returns=[0,1].map(i=>{const a=token('ep-return-'+i,String(d.outputs[i]),42,'teach-token teach-color-0');return a;});
 const final=token('ep-final','t1 ← '+d.result,76,'teach-token teach-color-2');
 const notes=[0,1,2].map(()=>txt('','teach-small'));
 draw=(p,m)=>{
  lastP=p;lastM=m;
  const L=layout(m),after=mode==='after',hot=d.rankLoads[0]!==d.rankLoads[1];
  const rowY=i=>L.mx.rowY+i*(L.mx.rowH+L.mx.gapY);
  const cellX=e=>L.mx.x+e*(L.mx.cw+L.mx.gapX);
  const cross=(t,e)=>after&&d.owner[e]!==d.tokenOwner[t];
  text(title,m?12:18,L.titleY,m?'EP · 同一路由，专家放 2 卡':'EP · 同一路由，把专家放到 2 张 GPU');
  text(legend,m?12:18,L.legendY,m?'8 分支 · t1 x=2 · 权重 .75/.25':'4 个 token × top-2 = 8 个分支 · t1 x=2 · E_i(x)=(i+1)x · 权重 0.75 / 0.25');
  // Adoption switch: both placements stay visible; the active card drives the board.
  for(const mg of modeGroups){
   const on=mg.key===mode,isAfter=mg.key==='after',MO=L.mode,i=modeGroups.indexOf(mg);
   set(mg.g,{transform:`translate(${MO.x[i]} ${MO.y})`,class:'ep-mode'+(on?' is-active':''),'aria-pressed':String(on),'aria-label':(isAfter?'采用后：2 张 GPU，每卡 2 组专家，跨卡 4 去 4 回':'采用前：1 张 GPU 放 4 组专家，全部本地，0 跨卡')+'。按 Enter 或空格切换。'});
   box(mg.bg,0,0,MO.w,MO.h);
   text(mg.name,10,MO.titleY,isAfter?(m?'采用后 · 2 GPU':'采用后 · 2 张 GPU'):(m?'采用前 · 1 GPU':'采用前 · 1 张 GPU'));
   show(mg.current,m?0:(on?1:0));text(mg.current,MO.w-10,MO.titleY,'当前');mg.current.setAttribute('text-anchor','end');
   text(mg.foot,10,MO.footerY,isAfter?(m?'2+2 · 4去/4回':'2+2 组 · 跨卡 4 去 / 4 回'):(m?'4组 · 0跨卡':'4 组/卡 · 跨卡 0'));
   const total=4*MO.cw+2*MO.gap+(isAfter?MO.groupGap:0),start=(MO.w-total)/2;
   mg.chips.forEach((c,e)=>{
    const x=start+e*(MO.cw+MO.gap)+(isAfter&&e>=2?MO.groupGap:0);
    set(c.r,{width:MO.cw,height:MO.ch,rx:4});set(c.t,{x:MO.cw/2,y:Math.round(MO.ch*.72)});
    set(c.g,{transform:`translate(${x} ${MO.chipY})`});
   });
  }
  text(routeHead,m?12:18,L.headY,'同一路由 · 8 个分支');
  text(routeHint,m?352:702,L.headY,m?'0/1 = 起点 GPU':'行首 0/1 = 起点 GPU · 点上方卡切换放置');routeHint.setAttribute('text-anchor','end');
  // Inputs stay in place; the owner tag changes from all-GPU0 to the adopted split.
  for(let i=0;i<4;i++){
   const owner=after?d.tokenOwner[i]:0;
   set(owners[i].r,{width:L.mx.ownerW,height:L.mx.rowH,rx:3});set(owners[i].t,{x:L.mx.ownerW/2,y:Math.round(L.mx.rowH*.72)});
   owners[i].t.textContent=String(owner);owners[i].g.dataset.rank=String(owner);
   move(owners[i],L.mx.ownerX,rowY(i),1);
   set(inputs[i].r,{width:L.mx.tokenW,height:L.mx.rowH,rx:4});set(inputs[i].t,{x:L.mx.tokenW/2,y:Math.round(L.mx.rowH*.72)});
   inputs[i].t.textContent=i===0&&!m?'t1 x=2':'t'+(i+1);
   move(inputs[i],L.mx.tokenX,rowY(i),1);
  }
  box(focus,6,rowY(0)-2,L.mx.x+4*(L.mx.cw+L.mx.gapX)-8,L.mx.rowH+4);show(focus,p>=1?.55:0);
  heads.forEach((h,e)=>{text(h,cellX(e)+L.mx.cw/2,L.mx.headY,'E'+e);h.setAttribute('text-anchor','middle');});
  cells.forEach((c,i)=>{
   const t=Math.floor(i/4),e=i%4,chosen=d.routes[t].includes(e),isCross=cross(t,e);
   box(c,cellX(e),rowY(t),L.mx.cw,L.mx.rowH);
   set(c,{class:chosen?'teach-contribution teach-color-'+t:'teach-mask','stroke-dasharray':isCross?'4 2':(chosen?'none':'4 3'),'stroke-width':(t===0&&chosen&&p>=1)?2.4:1});
   show(c,chosen?(p<1?.5:1):(p<1?.6:.5));
  });
  const midX=L.mx.x+2*(L.mx.cw+L.mx.gapX)-L.mx.gapX/2;
  set(divider,{d:m?`M${midX} 146 V${rowY(3)+L.mx.rowH}`:`M${midX} 174 V${rowY(3)+L.mx.rowH}`,class:'ep-divider'});show(divider,after?1:0);
  text(routeLegend,m?12:18,L.crossLegendY,after?(m?'虚线 = 跨卡（4/8）':'虚线格 = 跨卡分支（4 / 8）'):(m?'单卡 · 0 跨卡':'单卡：8 个分支全本地（0 跨卡）'));
  // The ledger answers the placement question before the animation moves.
  const ledgerOn=!m;
  [ledgerPanel,ledgerTitle,...ledgerHeads,...ledgerRows,...ledgerBefore,...ledgerAfter].forEach(el=>show(el,ledgerOn?1:0));
  if(ledgerOn){
   box(ledgerPanel,350,156,352,136);text(ledgerTitle,362,176,'前后账本（总专家仍 4）');
   text(ledgerHeads[0],540,196,'未采用');text(ledgerHeads[1],640,196,'采用后');ledgerHeads.forEach(h=>h.setAttribute('text-anchor','middle'));
   const rows=['1','4','0','8'],vals=['2','2+2','4 去 / 4 回',d.rankLoads.join('/')];
   ledgerRows.forEach((t,i)=>{text(t,362,218+i*22,['GPU 数','每卡专家组','跨卡分支','激活负载'][i]);text(ledgerBefore[i],540,218+i*22,rows[i]);text(ledgerAfter[i],640,218+i*22,vals[i]);ledgerBefore[i].setAttribute('text-anchor','middle');ledgerAfter[i].setAttribute('text-anchor','middle');});
  }
  show(sum1,m?1:0);show(sum2,m?1:0);
  if(m){text(sum1,12,L.sumY[0],'GPU 1→2 · 每卡 4→2组 · 总专家 4');text(sum2,12,L.sumY[1],`代价：跨卡 4去/4回 · 负载 ${d.rankLoads.join('/')}`);}
  // Mobile-only t1 focus card keeps the selected route readable at 360 units.
  const t1On=!!m;[t1Panel,t1Title,...t1Line].forEach(el=>show(el,t1On?1:0));
  if(t1On){
   box(t1Panel,L.t1.x,L.t1.y,L.t1.w,L.t1.h);text(t1Title,L.t1.x+8,L.t1.y+16,p<1?'t1 的输入':'t1 选中');
   text(t1Line[0],L.t1.x+8,L.t1.y+38,'x = 2');text(t1Line[1],L.t1.x+8,L.t1.y+56,p>=1?`E${sel[0]} + E${sel[1]}`:'');
   text(t1Line[2],L.t1.x+8,L.t1.y+74,p>=1?'w .75 / .25':'');text(t1Line[3],L.t1.x+8,L.t1.y+92,p>=1?`跨卡 ${after?d.remote[0]:0} 去 / ${after?d.remote[0]:0} 回`:'');
  }
  // Expert placement: one panel before; two rank panels after. Cards never move.
  const E=L.exp;
  box(panels[0],E.before.x,E.before.y,E.before.w,E.before.h);show(panels[0],after?0:1);
  box(panels[1],E.top.x,E.top.y,E.top.w,E.top.h);show(panels[1],after?1:0);
  box(panels[2],E.bottom.x,E.bottom.y,E.bottom.w,E.bottom.h);show(panels[2],after?1:0);
  text(panelTitles[0],E.before.x+10,E.headY,(m?'GPU0 · 4 组 · 全部本地':'GPU0 · 4 组专家参数 · 全部本地'));show(panelTitles[0],after?0:1);
  text(panelTitles[1],E.top.x+10,E.headY,`GPU0 · E0 E1 · 负载 ${after?d.rankLoads[0]:8}`);show(panelTitles[1],after?1:0);
  text(panelTitles[2],E.bottom.x+10,E.bottomHeadY,`GPU1 · E2 E3 · 负载 ${d.rankLoads[1]}`);show(panelTitles[2],after?1:0);
  set(rankDivider,{d:m?`M${E.top.x+2} ${E.bottom.y-2} H${E.bottom.x+E.bottom.w-2}`:`M${E.bottom.x-2} ${E.top.y+2} V${E.top.y+E.top.h-2}`,class:'ep-divider'});show(rankDivider,after?1:0);
  const cardPos=e=>({x:E.cardX[e],y:E.cardY[e]});
  // Stagger the eight already-routed branches only so each identity stays readable.
  // The order runs from the far expert to the near expert, so a later chip never
  // catches an earlier one; this is presentation order, not an execution dependency.
  const order=[...dispatch].sort((a,b)=>(E.cardX[b.e]+E.cardW-8)-(E.cardX[a.e]+E.cardW-8)||E.cardY[b.e]-E.cardY[a.e]||a.t-b.t||a.k-b.k);
  order.forEach((b,q)=>{b.start=1+q*.1;b.tx=E.cardX[b.e]+E.cardW-8;});
  const arrivedByExpert=[0,1,2,3].map(e=>dispatch.filter(b=>b.e===e&&p>=b.start+.15).length);
  cards.forEach((c,e)=>{
   const pos=cardPos(e),chosen=sel.includes(e)&&p>=1,oi=sel.indexOf(e);
   box(c.bg,pos.x,pos.y,E.cardW,E.cardH);
   set(c.bg,{class:'teach-panel ep-expert'+(chosen?' is-selected':'')});
   text(c.name,pos.x+10,pos.y+20,`E${e}(x)=${e===0?'x':(e+1)+'x'}`);
   text(c.load,pos.x+10,pos.y+38,p<1?`路由 ${d.loads[e]}`:p<2?`到达 ${arrivedByExpert[e]}/${d.loads[e]}`:`到达 ${d.loads[e]}`);
   let math='',op=0;
   if(oi>=0&&p>=3){math=`t1 2×${e+1}=${d.outputs[oi]}`;op=1;}
   else if(oi>=0&&p>=2.2){math='t1 计算中…';op=phase(p,2.2,2.5)*(1-phase(p,2.97,3));}
   text(c.math,pos.x+10,pos.y+56,math);show(c.math,op);
  });
  dispatch.forEach(b=>{
   const laneY=E.panelY-12;
   const sx=cellX(b.e)+L.mx.cw-24,sy=rowY(b.t)+(L.mx.rowH-14)/2;
   const u=clamp01((p-b.start)/.15),isCross=cross(b.t,b.e);
   let x=sx,y=sy;
   if(u>0){
    if(u<.22)y=lerp(sy,laneY,u/.22);
    else if(u<.8){x=lerp(sx,b.tx,(u-.22)/.58);y=laneY;}
    else{x=b.tx;y=lerp(laneY,E.cardY[b.e]-6,(u-.8)/.2);}
   }
   set(b.a.r,{'stroke-dasharray':isCross?'3 2':'none'});b.a.g.dataset.cross=String(isCross);
   move(b.a,x,y,u>0&&u<1?1:0);
  });
  // Return values are independent until the p5 weighted combine. The right-aligned
  // dock caption keeps the two branch slots and their weight lines free.
  const B=L.bot;
  returns.forEach((r,i)=>{
   const targetX=B.bayX[i]+(B.bayW-B.chipW)/2,targetY=B.bayY+(B.bayH-B.chipH)/2,u=phase(p,3.04+i*.06,3.9);
   const sourceX=m?targetX:cardPos(sel[i]).x+E.cardW/2-B.chipW/2;
   const sourceY=m?B.bayY-14:cardPos(sel[i]).y+E.cardH;
   let x=targetX,y=targetY;
   if(!m&&u>0){const returnLane=B.dockY-16;
    if(u<.18){x=sourceX;y=lerp(sourceY,returnLane,u/.18);}
    else if(u<.82){x=lerp(sourceX,targetX,(u-.18)/.64);y=returnLane;}
    else{x=targetX;y=lerp(returnLane,targetY,(u-.82)/.18);}
   }else if(m&&u>0){x=targetX;y=lerp(sourceY,targetY,u);}
   set(r.r,{width:B.chipW,height:B.chipH,rx:4});set(r.t,{x:B.chipW/2,y:Math.round(B.chipH*.7)});
   move(r,x,y,u>0?1:0);r.g.dataset.returned=String(u>=1);
  });
  set(lane,{d:`M${B.dockX} ${B.dockY-16} H${B.formulaX+B.formulaW}`,class:'ep-lane'});show(lane,!m&&p>=3?1:0);
  box(dockPanel,B.dockX,B.dockY,B.dockW,B.dockH);
  text(dockTitle,B.dockX+B.dockW-10,B.dockY+20,p<3?'t1 原位置 · 等待':p<4?'t1 原位置 · 返回中':p<5?'t1 原位置 · 已返回':'t1 原位置 · 合并');
  dockTitle.setAttribute('text-anchor','end');
  bays.forEach((b,i)=>{
   const x=B.bayX[i],y=B.bayY;
   box(b,x,y,B.bayW,B.bayH);set(b,{class:'teach-gate'+(p>=4?' is-ready':'')});
   const weighted=fmt(d.outputs[i]*d.weights[i]);
   const label=p<4?(m?`E${sel[i]}`:`E${sel[i]} 分支`):p<4.2?(m?`× ${d.weights[i]}`:`E${sel[i]} × ${d.weights[i]}`):(m?`× ${d.weights[i]} = ${weighted}`:`E${sel[i]} × ${d.weights[i]} = ${weighted}`);
   text(bayLabels[i],x+B.bayW/2,B.weightY,label);bayLabels[i].setAttribute('text-anchor','middle');bayLabels[i].setAttribute('font-size',m?'10':'12');
   show(weights[i],0);
  });
  box(formulaPanel,B.formulaX,B.formulaY,B.formulaW,B.formulaH);
  text(formulaTitle,B.formulaX+10,B.fTitleY,'t1 的合并点（Combine）');
  let main='',sub='',num=false;
  const o0=d.outputs[0],o1=d.outputs[1],fw0=fmt(d.weights[0]),fw1=fmt(d.weights[1]),c0=fmt(o0*d.weights[0]),c1=fmt(o1*d.weights[1]);
  const mergeOn=!m&&p>=4.42&&p<4.98;
  if(p<1){main=m?'先对齐路由与放置':'先对齐路由与专家放置';sub=m?`t1 的 top-2：E${sel[0]} + E${sel[1]}`:`t1 的 top-2：E${sel[0]} + E${sel[1]} · 权重 ${fw0} / ${fw1}`;}
  else if(p<2){const n=dispatch.filter(b=>p>=b.start+.15).length;main=m?`派发 ${n}/8 条`:(n<8?`派发中：${n}/8 条分支已到专家`:'派发完成：8/8 条分支已到专家');sub=m?`t1 → E${sel[0]} / E${sel[1]}`:`t1 → E${sel[0]} / E${sel[1]} · 错开显示只为看清，不是执行先后`;}
  else if(p<2.2){main=m?'两分支到达专家':'两条分支到达专家 · 开始计算';sub=m?'开始计算':`E${sel[0]} 与 E${sel[1]} 分别处理 t1 的输入`;}
  else if(p<3){main=m?`计算 E${sel[0]} / E${sel[1]}`:`计算中：E${sel[0]}(2)=2×${sel[0]+1} · E${sel[1]}(2)=2×${sel[1]+1}`;sub=m?`2×${sel[0]+1}=${2*(sel[0]+1)} · 2×${sel[1]+1}=${2*(sel[1]+1)}`:'两条分支独立计算，尚未返回';}
  else if(p<3.1){main=`计算完成：${o0} / ${o1}`;sub=m?'两分支各自返回':'两条结果按原 token 身份开始返回';}
  else if(p<4){main=m?'返回 t1 原位置':`返回中：${o0} → t1 原位置 · ${o1} → t1 原位置`;sub=m?`${o0} 与 ${o1} 各自回`:'两条输出各自返回，不混合不同 token';}
  else if(p<4.2){main=m?'各自乘权重':`各自乘权重：${o0} × ${fw0}`;sub=m?`${o0}×${fw0} · ${o1}×${fw1}`:`${o1} × ${fw1}`;}
  else if(p<4.45){main=m?`${o0}×${fw0}=${c0}`:`${o0} × ${fw0} = ${c0}`;sub=m?`${o1}×${fw1}=${c1}`:`${o1} × ${fw1} = ${c1}`;}
  else if(p<5){main=m?`${c0} + ${c1}`:'';sub=m?'合并回 t1':(p>=4.85?'合并中…':'两项贡献相加 · 合并后回到 t1');}
  else{num=true;main=m?`${o0}×.75 + ${o1}×.25`:`${o0} × 0.75 + ${o1} × 0.25 = ${d.result}`;sub=m?`= ${d.result} · 前后相同`:'权重合并回 t1 · 前后结果相同';}
  set(formulaMain,{class:num&&!m?'teach-number':'teach-text'});text(formulaMain,B.formulaX+10,B.fMainY,main);text(formulaSub,B.formulaX+10,B.fSubY,sub);show(formulaMain,mergeOn?0:1);
  const cu=phase(p,4.5,4.85),cfade=1-phase(p,4.72,4.84);
  move(contrib[0],lerp(B.formulaX+10,B.formulaX+58,cu),B.formulaY+36,p>=4.42&&p<4.9?cfade:0);
  move(contrib[1],lerp(B.formulaX+116,B.formulaX+88,cu),B.formulaY+36,p>=4.5&&p<4.9?cfade:0);
  text(plus,B.formulaX+100,B.formulaY+52,'+');plus.setAttribute('text-anchor','middle');show(plus,p>=4.5&&p<4.72?1:0);
  set(mergeRing,{cx:B.formulaX+92,cy:B.formulaY+47,r:(8*phase(p,4.78,4.92)).toFixed(2)});show(mergeRing,p>=4.78&&p<4.98?1-phase(p,4.84,4.94):0);
  set(final.r,{width:76,height:26,rx:4});set(final.t,{x:38,y:18});move(final,610,B.formulaY+8,0);show(final.g,(!m&&p>=5)?1:0);
  // The note keeps the causal claim and the cost next to the picture.
  const nY=L.notes.y;
  text(notes[0],L.notes.x,nY[0],after
   ?(m?'每卡 4→2 组，总专家 4；跨卡 4去/4回。':'为什么：每卡专家参数 4→2 组（总专家仍 4），不同专家放到不同 rank；GPU 数 1→2。')
   :(m?'1 GPU · 8 次激活都在 GPU0':'未采用：1 GPU 保存 4 组专家，8 个分支全部在 GPU0 本地，0 跨卡。'));
  text(notes[1],L.notes.x,nY[1],after
   ?(m?'错开演示不代表执行先后；top-2 稀疏不称加速。':`代价：8 个分支中 4 个跨卡去、4 个回；${hot?'热点 E0 被选 4 次，负载 6/2':'两卡负载 4/4'}。`)
   :(m?'8 次激活同卡；热点只改专家分布，单卡总激活仍 8 次。':'未采用没有跨卡开销；热点改变的是专家分布，单卡总激活仍 8 次，8 次都落在 GPU0；不是把单卡总负载从 8 增大。'));
  text(notes[2],L.notes.x,nY[2]??nY[1]+20,'同一路由与专家函数下前后输出相同；MoE top-2 稀疏不是 EP 收益，不声称加速。');show(notes[2],m?0:1);
  svg.dataset.parallelKind='ep';
  svg.dataset.epMode=hot?'hotspot':'balanced';
  svg.dataset.epSelected=sel.join(',');
  svg.dataset.epRankLoads=d.rankLoads.join('/');
  svg.dataset.epCross=String(d.crossDispatch);
  svg.dataset.epWeightResult=String(d.result);
  svg.dataset.epAdoption=mode;
  svg.dataset.epActualCross=after?String(d.crossDispatch):'0';
  svg.dataset.epComputed=p>=3?'1':'0';
  svg.dataset.epReturned=p>=4?'1':'0';
  svg.dataset.epResult=p>=5?String(d.result):'pending';
 };
 return draw;
 });
 const destroy=inner.destroy;inner.destroy=()=>{film?.classList.remove('is-ep');destroy();};
 return inner;
};
})();
