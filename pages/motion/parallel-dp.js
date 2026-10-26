/* DP: one integrated canvas for the same two requests under one or two complete
 * model copies. The switch changes request assignment and parameter blocks, not
 * the two layers inside a request. A single copy may batch or run the requests
 * one after another; this picture does not impose an order.
 *
 * Player contract: integer p is the arrived state of frames[p]; the interval
 * p -> p+1 is labelled "完成这一步后" and shows frames[p+1]. The phase gates
 * below therefore finish at integers: assignment at 2, L1 at 3, L2 at 4,
 * return at 5, ledger review at 6. */
(()=>{'use strict';
const {board,phase,lerp}=window.TeachingDrawing;
window.DPMechanismVisual=(svg,viewport)=>{
 const film=viewport.closest('.mechanism-film');if(film)film.classList.add('is-dp');
 const inner=board(svg,viewport,'parallel-dp',m=>m?680:600,H=>{
  const {txt,rect,path,text,box,show,token,move,set,node}=H;
  let replicas=2;
  try{replicas=new URL(location.href).searchParams.get('dp')==='1'?1:2;}catch(error){}
  let lastP=0,lastM=false;
  const fallback=n=>({
   replicas:n,
   requests:[{id:'R1',x:1,h:2,y:3,caller:1},{id:'R2',x:3,h:6,y:7,caller:2}],
   assignments:n===2?[['A'],['B']]:[['A'],['A']],
   paramBlocks:n*2,paths:n,workPerRequest:2,batch:n===2?'per-replica':'shared',outputs:[3,7]
  });
  const data=()=>window.ParallelTeachingModel?.dataParallel?window.ParallelTeachingModel.dataParallel(replicas):fallback(replicas);

  const title=txt('DP · 同一输入：1 份与 2 份完整模型','teach-heading');
  const subtitle=txt('R1 x=1、R2 x=3；每请求都走 L1 ×2 → L2 +1','teach-small dp-tiny');
  const status1=txt('','teach-small dp-tiny'),status2=txt('','teach-small dp-tiny');
  const tabs=[1,2].map(n=>{
   const g=node('g',{class:'dp-tab','data-dp-replicas':String(n),role:'button',tabindex:'0','aria-label':n===1?'切换到一个完整副本':'切换到两个完整副本','aria-pressed':'false'});
   const r=node('rect',{rx:7,class:'dp-tab-bg'}),t=node('text',{'text-anchor':'middle',class:'dp-tab-label'},n===1?'1 副本':'2 副本');
   g.append(r,t);return{g,r,t,n};
  });
  const sched=rect('teach-panel dp-sched');
  const pathNote=txt('','teach-small dp-sched-text dp-tiny');

  const cards=[0,1].map(i=>({
   i,
   panel:rect('teach-panel dp-card'),
   name:txt('完整副本 '+(i?'B':'A'),'teach-heading dp-card-name'),
   sub:txt('','teach-small dp-tiny dp-card-sub'),
   foot:txt('','teach-small dp-tiny dp-card-foot'),
   l1:rect('teach-panel dp-layer'),l2:rect('teach-panel dp-layer'),
   l1label:txt('L1 · ×2','teach-number dp-layer-label'),
   l2label:txt('L2 · +1','teach-number dp-layer-label'),
   l1f:[txt('','teach-number dp-formula'),txt('','teach-number dp-formula')],
   l2f:[txt('','teach-number dp-formula'),txt('','teach-number dp-formula')],
   out:token('dp-out-R'+(i+1),'','150','teach-token '+(i?'teach-color-1':'teach-color-0')),
   outNote:txt('','teach-small dp-tiny dp-out-note'),
   ret:path('teach-link')
  }));
  // Card names and copy roles sit on the right edge, except the narrow one-copy
  // card where R2's lane needs the right side; there they end between the lanes.
  cards.forEach(card=>{
   [card.name,card.sub,card.foot,card.l1label,card.l2label].forEach(t=>t.setAttribute('text-anchor','end'));
  });
  const req=[
   token('dp-R1','R1 · x=1',110,'teach-token teach-color-0'),
   token('dp-R2','R2 · x=3',110,'teach-token teach-color-1')
  ];
  const pblocks=[0,1,2,3].map(k=>{
   const copy=k<2?'A':'B',layer=k%2?'L2':'L1';
   const t=token('dp-param-'+k,'','120','teach-token dp-param');
   t.g.dataset.copy=copy;t.g.dataset.layer=layer;
   return t;
  });
  const ledger=[0,1,2,3,4].map(()=>({
   cell:rect('teach-panel dp-ledger-cell'),label:txt('','dp-tiny dp-ledger-label'),
   value:txt('','dp-ledger-value'),other:txt('','dp-tiny dp-ledger-other')
  }));
  const returnNote=txt('','teach-small dp-note');

  function choose(n){
   if(n===replicas)return;
   replicas=n;
   try{const u=new URL(location.href);u.searchParams.set('dp',String(n));history.replaceState(null,'',u);}catch(error){}
   try{svg.dispatchEvent(new CustomEvent('parallel-dp-replicas',{bubbles:true,detail:{replicas:n}}));}catch(error){}
   render(lastP,lastM);
  }
  tabs.forEach(tab=>{
   tab.g.addEventListener('click',()=>choose(tab.n));
   tab.g.addEventListener('keydown',event=>{
    if(event.key==='Enter'||event.key===' '){event.preventDefault();event.stopPropagation();choose(tab.n);}
   });
  });

  function fitToken(t,w,h=24){
   set(t.r,{width:w,height:h});set(t.t,{x:w/2,y:Math.round(h*.68)});t.width=w;
  }
  function showLayer(block,on){set(block,{class:'teach-panel dp-layer'+(on?' is-active':'')});}

  function render(p,m){
   const d=data(),two=d.replicas===2,W=m?360:720,canvasH=m?680:600;
   const vh=canvasH;
   svg.setAttribute('viewBox','0 0 '+W+' '+vh);
   svg.dataset.parallelKind='dp';svg.dataset.dpReplicas=String(d.replicas);
   svg.dataset.dpParamBlocks=String(d.paramBlocks);svg.dataset.dpPaths=String(d.paths);
   svg.dataset.dpWorkPerRequest=String(d.workPerRequest);svg.dataset.dpBatch=d.batch;
   svg.dataset.dpOutputs=d.outputs.join(',');svg.dataset.dpExpectedOutputs=d.outputs.join(',');
   svg.setAttribute('aria-label','DP 教学画布：同一 R1 x=1、R2 x=3，'+(two?'两个完整副本，参数块 4，结果分别返回':'一个完整副本，参数块 2，两请求可合批或先后')+'。');

   const cardY=m?(two?166:170):146,cardH=m?(two?200:230):258;
   const cardX=two?(m?[10,188]:[28,368]):[m?12:96];
   const cardW=two?(m?162:324):(m?336:528);
   const inputY=m?(two?140:152):138;
   const chipW=m?(two?58:66):(two?100:78);

   text(title,18,m?22:26);text(subtitle,18,m?40:46);
   const tabW=m?160:82,tabH=m?30:28,tabX=m?12:536,tabY=m?54:12;
   tabs.forEach((tab,i)=>{
    box(tab.r,tabX+i*(tabW+8),tabY,tabW,tabH);text(tab.t,tabX+i*(tabW+8)+tabW/2,tabY+tabH/2+5);
    set(tab.g,{class:'dp-tab'+(tab.n===d.replicas?' is-active':''),'aria-pressed':String(tab.n===d.replicas)});
   });
   text(status1,18,m?100:66,d.replicas===2
    ?(m?'当前：2 副本 · 4 个参数块 · 2 条独立路径':'当前：2 个完整副本 · 4 个参数块 · 2 条独立路径')
    :(m?'当前：1 副本 · 2 个参数块 · 共享容量':'当前：1 个完整副本 · 2 个参数块 · 共享容量'));
   text(status2,18,m?116:82,two
    ?(m?'R1 → A、R2 → B；每请求 2 层':'R1 → A、R2 → B；每请求仍为 2 层，结果分别返回')
    :(m?'可同批或先后；每请求仍 2 层':'两请求可同批或先后（由调度决定）；每请求仍为 2 层'));

   // The path note and the request chips get separate vertical bands. In the
   // one-copy case the note stays inside the shared-capacity band; in both
   // cases the chips start below all explanatory text.
   const bandY=m?120:96,bandH=m?30:34,noteY=two?(m?134:117):(m?141:117);
   if(!two){box(sched,m?12:96,bandY,m?336:528,bandH);show(sched,1);}else{show(sched,0);}
   text(pathNote,two?(m?18:18):(m?20:108),noteY,two
    ?(m?'R1 → A，R2 → B；每请求仍完整两层':'两条独立执行路径：R1 → A，R2 → B；每请求仍完整两层')
    :(m?'共享容量：可同批或先后；不等于串行':'共享执行容量：可同批，也可先后；单副本不等于只能串行'));
   show(pathNote,1);

   // Card frames and layer blocks. Only a visible copy gets geometry; the
   // hidden B copy keeps its last finite frame and stays fully transparent.
   const layout=two
    ?(m?{l1:[42,68,60,null,56,78],l2:[114,68,132,null,128,150]}:{l1:[52,76,72,null,68,90],l2:[140,76,160,null,156,178]})
    :(m?{l1:[44,78,62,92,58,80],l2:[130,78,148,178,144,166]}:{l1:[52,80,72,102,68,90],l2:[142,82,162,192,158,180]});
   // L1 runs in 2 -> 3, L2 in 3 -> 4. Integer 3/4 are completed states, so the
   // layer is no longer marked active there; h/y chips carry the result.
   const l1Active=p>=2.04&&p<3.0,l2Active=p>=3.04&&p<4.0;
   cards.forEach((card,i)=>{
    const on=two||i===0;
    if(!on){
     show(card.panel,0);show(card.name,0);show(card.sub,0);show(card.foot,0);
     show(card.l1,0);show(card.l2,0);show(card.l1label,0);show(card.l2label,0);
     card.l1f.forEach(t=>show(t,0));card.l2f.forEach(t=>show(t,0));
     return;
    }
    const x=cardX[i],y=cardY,w=cardW;
    show(card.panel,1);show(card.name,1);show(card.sub,1);show(card.foot,0);
    box(card.panel,x,y,w,cardH);
    const labelX=!two&&m?x+180:x+w-12;
    text(card.name,labelX,y+(m?18:20));
    text(card.sub,labelX,y+(m?34:36),m?(two?'独立路径':'共享容量'):(i?'完整模型 · 独立路径':'完整模型 · '+(two?'独立路径':'共享容量')));
    [card.l1,card.l2].forEach((block,j)=>{
     const z=layout[j?'l2':'l1'];box(block,x+12,y+z[0],w-24,z[1]);show(block,1);
    });
    text(card.l1label,x+w-14,y+layout.l1[4]);text(card.l2label,x+w-14,y+layout.l2[4]);
    show(card.l1label,1);show(card.l2label,1);
    const rowY=(z,row)=>y+z[2]+(row&&z[3]?z[3]-z[2]:0);
    const fX=x+(two?(m?86:124):(m?142:150));
    card.l1f.forEach((t,row)=>{
     const use=row===0||(!two&&i===0);
     text(t,fX,rowY(layout.l1,row)+18,row?'3 × 2 = 6':'1 × 2 = 2');
     show(t,use?phase(p,2.06,2.9):0);
    });
    card.l2f.forEach((t,row)=>{
     const use=row===0||(!two&&i===0);
     text(t,fX,rowY(layout.l2,row)+18,row?'6 + 1 = 7':'2 + 1 = 3');
     show(t,use?phase(p,3.06,3.9):0);
    });
    // B's second-row formula is replaced by R2's first-row formula; row 1 is
    // R1/R2 only in the one-copy layout.
    if(two&&i===1){text(card.l1f[0],fX,y+layout.l1[2]+18,'3 × 2 = 6');text(card.l2f[0],fX,y+layout.l2[2]+18,'6 + 1 = 7');}
    showLayer(card.l1,l1Active&&on);showLayer(card.l2,l2Active&&on);
   });

   // Request chips: start below the note, enter the card at 2, finish L1 at 3,
   // finish L2 at 4, then move to the result slot and disappear at 5.
   const outY=()=>m?(two?374:410):414;
   const outX=i=>{
    if(two)return m?(i?194:16):(cardX[i]+14);
    return m?(i?190:16):(cardX[0]+(i?270:44));
   };
   const outW=150;
   const hDone=p>=3,yDone=p>=4,outDone=p>=5;
   const u0=phase(p,1.04,1.94),u1=phase(p,2.04,2.94),u2=phase(p,3.04,3.94),u3=phase(p,4.04,4.94);
   const starts=two
    ?(m?[{x:cardX[0]+10,y:inputY},{x:cardX[1]+10,y:inputY}]:[{x:cardX[0]+14,y:inputY},{x:cardX[1]+14,y:inputY}])
    :[{x:cardX[0]+14,y:inputY},{x:cardX[0]+(m?202:234),y:inputY}];
   req.forEach((r,k)=>{
    const cardIdx=two?k:0,row=two?0:k,x=cardX[cardIdx],cy=cardY;
    fitToken(r,chipW,24);
    // In the one-copy layout the two requests keep separate horizontal lanes
    // from start to result: R1 on the left, R2 right of the formulas. A shared
    // entry point would paint the two 24-high chips on top of each other.
    const lane=two?x+14:x+(k?(m?202:234):14);
    const entry={x:lane,y:cy+26};
    const l1={x:lane,y:cy+layout.l1[2]+(row&&layout.l1[3]?layout.l1[3]-layout.l1[2]:0)};
    const l2={x:lane,y:cy+layout.l2[2]+(row&&layout.l2[3]?layout.l2[3]-layout.l2[2]:0)};
    const out={x:outX(k)+8,y:outY(k)-32};
    let from=starts[k],to=entry,u=0;
    if(p>=4.04){from=l2;to=out;u=u3;}
    else if(p>=3.04){from=l1;to=l2;u=u2;}
    else if(p>=2.04){from=entry;to=l1;u=u1;}
    else if(p>=1.04){from=starts[k];to=entry;u=u0;}
    let px=lerp(from.x,to.x,u),py=lerp(from.y,to.y,u);
    // On the narrow one-copy board R2 descends in its own lane, then crosses
    // below the L2 formula into caller 2's slot. This keeps the whole return
    // rectangle clear of "6 + 1 = 7" without moving the result or its caller.
    if(!two&&k===1&&m&&p>=4.04){
      const split=.72;
      if(u<=split){px=lane;py=lerp(l2.y,out.y,u/split);}
      else{px=lerp(lane,out.x,(u-split)/(1-split));py=out.y;}
    }
    // The request chip fades just before the result slot commits its value.
    move(r,px,py,1-phase(p,4.9,4.98));
    r.t.textContent='R'+(k+1)+' · '+(yDone?'y='+(k?7:3):hDone?'h='+(k?6:2):'x='+(k?3:1));
    // The moving chip is the request identity; layer labels and formulas stay fixed.
    if(l1Active||l2Active)r.g.dataset.stage=l1Active?'L1':'L2';else delete r.g.dataset.stage;
   });

   // Return slots. They exist from the start as destinations; values appear only
   // after the request chip has arrived (movement ends at 4.94, y appears at 5).
   cards.forEach((card,i)=>{
    const x=outX(i),y=outY(i);
    fitToken(card.out,outW,30);
    move(card.out,x,y,outDone?1:.35);
    card.out.t.textContent='R'+(i+1)+' · '+(outDone?'y='+(i?7:3):'等待返回');
    set(card.out.g,{class:'teach-token '+(i?'teach-color-1':'teach-color-0')});
    text(card.outNote,x,y+46,'调用方 '+(i+1));
    show(card.outNote,outDone?1:0);
    // A short dashed lane from the model to the result slot. It is only a
    // local return edge; the two values never merge.
    set(card.ret,{d:'M'+(x+outW/2)+' '+(y-24)+' V'+(y-2)});
    show(card.ret,phase(p,4.06,4.96));
    card.ret.setAttribute('marker-end',outDone?'':'none');
   });
   svg.dataset.dpReturned=outDone?d.outputs.join(','):'';

   // Parameter copies: 2 blocks for one complete model, 4 for two. The strip
   // follows the switch immediately, so request assignment and block count stay
   // in sync at every paused position.
   const py=m?(two?432:468):494;
   text(returnNote,W/2,m?0:482,outDone
    ?'每请求 2 层 · 完整前向；R1 / R2 结果分别返回，不相加'
    :'每请求 2 层 · 完整前向；1 / 2 副本都不减少单请求层数');
   returnNote.setAttribute('text-anchor','middle');show(returnNote,m?0:1);
   const pbW=m?128:(two?118:150),pbH=24;
   const pbPos=two
    ?(m?[[60,py],[198,py],[60,py+32],[198,py+32]]:[[100,py],[226,py],[412,py],[538,py]])
    :(m?[[60,py],[198,py]]:[[100,py],[268,py]]);
   pblocks.forEach((b,k)=>{
    const visible=two||k<2;if(visible){fitToken(b,pbW,pbH);move(b,pbPos[k][0],pbPos[k][1],1);}
    show(b.g,visible?1:0);
    b.t.textContent=(k<2?'A':'B')+' · '+(k%2?'L2':'L1');
    set(b.g,{class:'teach-token dp-param'+(p>=1&&p<5?' is-active':''),'data-copy':k<2?'A':'B','data-layer':k%2?'L2':'L1'});
   });

   // Same-input ledger. Both conditions remain visible; only the current one is
   // large. This is the compact replacement for the old stacked adoption panel.
   const labels=['完整副本','执行路径','参数块','每请求层数','返回 R1/R2 · 预期'];
   const active=[
    String(d.replicas),d.replicas===2?'2 独立':'1 可合批',String(d.paramBlocks),'2 层',d.outputs.join(' / ')
   ];
   const other=[
    '对照 '+(d.replicas===2?'1':'2'),'对照 '+(d.replicas===2?'1 可合批':'2 独立'),'对照 '+(d.replicas===2?'2':'4'),'两种条件都是 2',
    outDone?'已返回核对值 '+d.outputs.join(' / '):'预期核对值，未返回'
   ];
   const ly=m?500:526,lh=m?48:62,lgap=m?6:0;
   ledger.forEach((cell,k)=>{
    const col=m?k%2:k,row=m?Math.floor(k/2):0;
    const x=m?(col?184:12):(18+k*139),w=m?164:130,y=ly+row*(lh+lgap);
    box(cell.cell,x,y,w,lh);
    set(cell.cell,{class:'teach-panel dp-ledger-cell'+(p>=5.8?' is-review':'')});
    text(cell.label,x+6,y+(m?11:15),labels[k]);
    text(cell.value,x+6,y+(m?28:38),active[k]);
    text(cell.other,x+6,y+(m?43:55),other[k]);
   });

   // Selected tab styling and keyboard hint.
   const tabHint=m?'点按上方副本数切换':'点按右上角副本数切换';
   subtitle.setAttribute('aria-label',tabHint);
  }

  return (p,m)=>{lastP=p;lastM=m;render(p,m);};
 });
 const destroy=inner.destroy;inner.destroy=()=>{film?.classList.remove('is-dp');destroy();};
 return inner;
};
})();
