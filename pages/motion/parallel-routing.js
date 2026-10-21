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
window.ExpertParallelVisual=(svg,viewport,s)=>board(svg,viewport,'expert-routing',()=>910,H=>{
 const {txt,rect,text,box,show,token,move,set}=H,d=s.data;
 const title=txt('4 个 token → Router → 4 个专家','teach-heading'),legend=txt('t1…t4 = 输入位置；E0…E3 = 不同专家','teach-small');
 const input=Array.from({length:4},(_,i)=>token('ep-input-'+i,i?'t'+(i+1):'t1 x=2',i?54:66,'teach-token teach-color-'+i));
 const owners=txt('输入归属：t1/t2 在 Rank 0，t3/t4 在 Rank 1','teach-small');
 const matrix=Array.from({length:16},()=>rect('teach-mask')),heads=[0,1,2,3].map(i=>txt('E'+i,'teach-small')),rows=[0,1,2,3].map(i=>txt('t'+(i+1),'teach-small'));
 const ranks=[rect(),rect()],rankNames=[0,1].map(i=>txt('Rank '+i+' / P'+i+' / GPU '+i,'teach-small'));
 const ex=[0,1,2,3].map(i=>({bg:rect(),name:txt('E'+i+'(x) = '+(i+1)+'x','teach-heading'),load:txt('','teach-small')}));
 const dispatch=d.routes.flatMap((es,t)=>es.map((e,k)=>({t,e,k,a:token('ep-dispatch-'+t+'-'+e,'t'+(t+1),28,'teach-token teach-color-'+t)})));
 const out=d.outputs.map((n,i)=>token('ep-return-'+i,String(n),24,'teach-token teach-color-0'));
 const returned=txt('t1 原位置 · Rank 0','teach-small');
 const routeWeights=d.weights.map((v,i)=>txt('E'+d.selected[i]+'：'+v,'teach-small'));
 const instruction=txt('','teach-heading'),formula=txt('','teach-number'),math=[txt('','teach-small'),txt('','teach-small')],final=token('ep-final','t1: '+d.result,100,'teach-token teach-color-2'),note=txt('','teach-small');
 return(p,m)=>{const rx=[18,m?190:382],w=m?152:320,cw=m?48:95,left=m?86:170;
  const focused=p>=2&&!d.loads.some(n=>n>2),background=focused?.28:1;
  // Dispatch tokens cross the annotation band between the matrix and the rank
  // cards. Fade those labels for that window so moving tokens never sit on them.
  const annotation=p<1?1:p<1.12?1-phase(p,1,1.12):p<1.86?0:phase(p,1.86,2);
  text(title,18,28);text(legend,18,56,m?'t = 输入位置；E = 专家':'t1…t4 = 输入位置；E0…E3 = 不同专家');input.forEach((t,i)=>move(t,28+i*(m?78:170),82,i?background:1));
  text(owners,18,134,m?'t1/t2→Rank 0；t3/t4→Rank 1':'输入归属：t1/t2 在 Rank 0，t3/t4 在 Rank 1');
  heads.forEach((t,i)=>text(t,left+i*cw,158));rows.forEach((t,i)=>text(t,left-38,192+i*34));
  matrix.forEach((c,i)=>{const r=Math.floor(i/4),e=i%4,chosen=d.routes[r].includes(e);box(c,left+e*cw,170+r*34,cw-6,30);set(c,{class:chosen&&p>=1?'teach-contribution teach-color-'+r:'teach-mask'});show(c,r?background:1);});
  rows.forEach((t,i)=>show(t,i?background:1));
  routeWeights.forEach((t,i)=>{text(t,18+i*(m?172:360),324);show(t,annotation);});
  text(instruction,18,354,p<1?'路由表：每个 token 选择两个专家':p<2?'t1 的两个分支按路由同时派发':p<3?'到达目标专家；颜色保留 token 身份':p<4?'聚焦 t1：所选专家分别计算':'按 t1 身份返回，再按权重合并');show(instruction,annotation);
  [0,1].forEach(r=>{box(ranks[r],rx[r],370,w,304);text(rankNames[r],rx[r]+6,392);show(rankNames[r],annotation);});
  ex.forEach((e,i)=>{const r=Math.floor(i/2),y=406+(i%2)*134,opacity=d.selected.includes(i)?1:background;box(e.bg,rx[r]+8,y,w-16,128);text(e.name,rx[r]+16,y+26);text(e.load,rx[r]+16,y+112,'派发数：'+(p>=2?d.loads[i]:0));[e.bg,e.name,e.load].forEach(t=>show(t,opacity));});
  dispatch.forEach(({a,t,e})=>{
   const r=Math.floor(e/2),members=dispatch.filter(v=>v.e===e),slot=members.findIndex(v=>v.a===a);
   const startX=left+e*cw,startY=170+t*34,cardY=406+(e%2)*134;
   const finalX=rx[r]+10+(slot%4)*30,finalY=cardY+52;
   const entryX=r===0?rx[r]+w-30:rx[r]-30;
   // Stagger each token's two branches. They leave the matrix into the empty
   // annotation band, descend at the outer edge of the target rank, and only
   // then slide under the expert name. Names and load labels are never crossed.
   const u=phase(p,1+t*.18,1.18+t*.18);let x,y;
   if(u<.18){const k=u/.18;x=startX;y=lerp(startY,336,k);}
   else if(u<.68){const k=(u-.18)/.5;x=lerp(startX,entryX,k);y=336;}
   else if(u<.86){const k=(u-.68)/.18;x=entryX;y=lerp(336,finalY,k);}
   else {const k=(u-.86)/.14;x=lerp(entryX,finalX,k);y=finalY;}
   move(a,x,y,phase(p,0)*(t?background:1)*(1-phase(p,2.15,2.4)));
  });
  out.forEach((t,i)=>{
   const e=d.selected[i],r=Math.floor(e/2),cardY=406+(e%2)*134;
   const startX=rx[r]+14+(i?64:0),startY=cardY+52;
   // Return lanes are per output, not per rank. When both selected experts
   // live on one rank (the skewed scene), the old shared outer lane made the
   // 2 and 4 values merge at p3.5. Stay inside the empty right edge of the
   // expert card while descending, then slide left below every card to the
   // two final positions. The left-to-right order never changes, so the two
   // different return values cannot cross.
   const laneX=rx[r]+w-(i?30:62),drop=phase(p,3,3.35),toFinal=phase(p,3.6,3.85);
   const x=toFinal?lerp(laneX,30+i*76,toFinal):laneX;
   const y=drop?lerp(startY,696,drop):startY;
   move(t,x,y,phase(p,2.45,2.7));
   text(math[i],18,758+i*24,`E${e}(t1)：2 × ${e+1} = ${d.outputs[i]}`);show(math[i],p>=3?1:0);
  });
  text(returned,140,686);show(returned,phase(p,3));
  text(formula,18,814,`${d.outputs[0]} × 0.75 + ${d.outputs[1]} × 0.25 = ${d.result}`);show(formula,phase(p,4));move(final,30,824,phase(p,4));
  text(note,18,890,p>=4?'每条分支按路由权重贡献；合并结果归回 t1。':(m?'两条分支按路由权重合并回 t1。':'绿色两分支都来自 t1；专家编号区分计算。'));
  svg.dataset.parallelKind='ep';
 };
});
})();
