/* Each parallel scheme visualizes the object it partitions. */
(()=>{'use strict';
const {board,phase,lerp}=window.TeachingDrawing,V=window.MechanismVisuals,pp=V['f-parallel'];
V['f-parallel']=(svg,viewport,s)=>{
 if(s.kind==='pipeline'){
  // The partition captions sit between the model strip and the stage grid.
  // Keep them out of the layer-token row entirely: they appear with the grid,
  // below the explanation, instead of sharing the strip with L1…L4.
  const inner=pp(svg,viewport,s);
  return {paint(...args){inner.paint(...args);const p=args[3]??0;svg.querySelectorAll('text').forEach(e=>{if(/^Stage [01] · GPU [01]$/.test(e.textContent.trim())){e.setAttribute('y','232');e.setAttribute('opacity',p>=1?'1':'0');}});},destroy(){inner.destroy();}};
 }
 if(s.kind==='parallel-dp')return window.DataParallelVisual(svg,viewport,s);
 if(s.kind==='parallel-ep')return window.ExpertParallelVisual(svg,viewport,s);
 if(s.kind==='parallel-tp')return window.TensorParallelVisual(svg,viewport,s);
 return board(svg,viewport,s.kind,()=>s.kind==='parallel-sp'?626:s.kind==='parallel-pcp'?870:790,H=>{
 const {txt,rect,path,text,box,show,token,move,set,wire}=H,d=s.data,kind=s.kind.slice(9);
 const header=txt('','teach-heading'),hint=txt('','teach-small'),panels=[rect(),rect()],names=[0,1].map(r=>txt('','teach-small'));
 const note=txt('','teach-small'),result=txt('','teach-heading'),line2=txt('','teach-small');
 const tile=(id,label,color=0,w=64)=>token(id,label,w,'teach-token teach-color-'+color);
 let paint;
 if(kind==='sp'){
  const fit=(t,width)=>{set(t.r,{width,height:30});set(t.t,{x:width/2,y:19});};
  const original=Array.from({length:4},(_,i)=>tile('sp-source-'+i,'p'+(i+1)+' [a'+(i+1)+',b'+(i+1)+']',i<2?0:1,116));
  const local=Array.from({length:4},(_,i)=>tile('sp-local-'+i,'p'+(i+1)+' [a'+(i+1)+',b'+(i+1)+']',i<2?0:1,116));
  const copies=Array.from({length:4},(_,i)=>tile('sp-gather-'+i,'n'+(i+1)+' [a′,b′]',i<2?0:1,116));
  const contrib=Array.from({length:8},(_,i)=>tile('sp-contrib-'+i,'c'+(i%4+1)+(i<4?'₀':'₁'),i<4?0:1,54));
  const reducing=Array.from({length:8},(_,i)=>tile('sp-reduce-'+i,'c'+(i%4+1)+(i<4?'₀':'₁'),i<4?0:1,54));
  const outputs=Array.from({length:4},(_,i)=>tile('sp-output-'+i,'y'+(i+1)+' [u,v]',i<2?0:1,100));
  const labels=[txt('完整激活 · 4 个位置 × 2 个特征','teach-small'),txt('行分片：每行的两个特征仍在一起','teach-small')];
  const links=Array.from({length:4},()=>path('teach-link'));
  paint=(p,m,x,w)=>{
   text(header,18,28,'SP：序列行分片与 TP 布局转换');text(hint,18,56,'p = 输入行；n = 逐 token 变换后的行');
   text(labels[0],18,72);text(labels[1],18,166);show(labels[1],phase(p,.7,.95));
   // Group the source rows by owner before the split: p1/p2 sit over Rank 0
   // and p3/p4 over Rank 1. Each pair then drops straight into its panel, so
   // the left/right groups never cross through one another.
   const sx=m?[18,18,190,190]:[18,18,382,382],sy=[86,122,86,122];
   original.forEach((t,i)=>{fit(t,116);move(t,sx[i],sy[i],1-phase(p,.75,1.05));});
   const panelTop=176,rows=[220,254,288,322];
   [0,1].forEach(r=>{box(panels[r],x[r],panelTop,w,196);text(names[r],x[r]+6,198,m?'Rank '+r:'Rank '+r+' · P'+r+' / GPU '+r);show(names[r],phase(p,.9,1.1));});
   local.forEach((t,i)=>{const r=i<2?0:1,j=i%2,u=phase(p,0);fit(t,116);move(t,lerp(sx[i],x[r]+12,u),lerp(sy[i],rows[j],u),u);t.t.textContent=(p>=2?'n':'p')+(i+1)+(p>=2?' [a′,b′]':' [a'+(i+1)+',b'+(i+1)+']');});
   // AllGather copies arrive in the receiver's empty lower rows. The dashed
   // link shows the cross-rank origin; the copy itself only slides a short
   // distance into its slot, so paused frames never have two long cards
   // crossing each other or the rank captions.
   copies.forEach((t,i)=>{const r=i<2?1:0,j=i<2?i:i-2,u=phase(p,2);fit(t,116);t.t.textContent='n'+(i+1)+' [a′,b′]';move(t,lerp(x[r]+28,x[r]+12,u),rows[2+j],u);});
   links.forEach((line,i)=>{const from=i<2?0:1,to=1-from,j=i<2?i:i-2;const fy=rows[i<2?i:0]+15,ty=rows[2+j]+15;wire(line,`M${x[from]+128} ${fy} L${x[to]+12} ${ty}`,phase(p,2)*(1-phase(p,3.2,3.4)));});
   text(result,18,390,p<1?'按序列切行，每个 token 的特征保持完整':p<2?'各 rank 独立执行逐 token 算子':p<3?'AllGather：收齐四行的副本':p<4?'TP：每卡产生四行的局部贡献':'ReduceScatter：求和后各 rank 留本行');
   const bx0=m?18:180,bx1=bx0+72,bx2=bx1+72,by0=414;
   const contribOp=phase(p,3)*(1-phase(p,4.15,4.35)),reduceOp=phase(p,4.35,4.5)*(1-phase(p,4.85,5)),outOp=phase(p,4.85,5),pull=phase(p,4.5,4.85)*6;
   contrib.forEach((t,i)=>{const q=Math.floor(i/4),j=i%4;fit(t,54);move(t,q?bx1:bx0,by0+j*38,contribOp);});
   reducing.forEach((t,i)=>{const q=Math.floor(i/4),j=i%4;fit(t,54);move(t,(q?bx1:bx0)+(q?-pull:pull),by0+j*38,reduceOp);});
   outputs.forEach((t,i)=>{fit(t,100);move(t,bx2,by0+i*38,outOp);});
   text(note,18,578,'cᵢ₀ + cᵢ₁ = yᵢ；i 是位置，0/1 是贡献来源 rank。');show(note,p>=4?1:0);
   text(line2,18,604,p>=4?'回到序列分片：每个位置的特征仍然完整。':'这里只追踪激活的分工；TP 算法见 TP 示例。');
  };
 }else if(kind==='pcp'){
  const source=Array.from({length:8},(_,i)=>tile('pcp-source-'+i,'p'+(i+1),d.ranks[0].includes(i)?0:1,32));
  const qs=Array.from({length:8},(_,i)=>tile('pcp-query-'+i,'Q'+(i+1),d.ranks[0].includes(i)?0:1,48));
  const outputs=Array.from({length:8},(_,i)=>tile('pcp-output-'+i,'O'+(i+1),d.ranks[0].includes(i)?0:1,48));
  const originals=[tile('pcp-original-0','原 K/V 片 0',0,120),tile('pcp-original-1','原 K/V 片 1',1,120)];
  const kv=[tile('pcp-kv-0','K/V 片 0 副本',0,120),tile('pcp-kv-1','K/V 片 1 副本',1,120)];
  const cells=Array.from({length:64},()=>rect('teach-mask'));
  const rows=Array.from({length:8},(_,i)=>txt('Q'+(i+1),'teach-small')),cols=Array.from({length:8},(_,i)=>txt('K'+(i+1),'teach-small'));
  const explain=txt('同一条输入的八个位置','teach-small'),dependency=txt('','teach-small'),localLabels=[txt('','teach-small'),txt('','teach-small')],workSums=[txt('','teach-small'),txt('','teach-small')];
  const links=[path('teach-link'),path('teach-link')];
  paint=(p,m,x,w)=>{
   text(header,18,28,'PCP：沿 Query 行分担 Prefill');text(hint,18,60,'矩阵格 = 一次 Q–K 配对；灰格禁止读取');
   source.forEach((t,i)=>move(t,18+i*(m?41:84),100,1-phase(p,.75,1.05)));
   text(explain,18,80);show(explain,p<.5?1:0);
   const panelTop=182;
   [0,1].forEach(r=>{box(panels[r],x[r],panelTop,w,248);text(names[r],x[r]+6,204,m?'Rank '+r:'Rank '+r+' · P'+r+' / GPU '+r);show(names[r],phase(p,.9,1.1));});
   // Each rank gets one Query row. Rank 1 drops below the source row first so
   // the zigzag ownership move (Q7/Q8 travel left) passes over a clear lane
   // instead of through another moving Q. Rank 0 forms its row after that.
   const qw=m?30:48,qGap=m?36:64,lane0=140,lane1=178;
   const slot=i=>{const r=d.ranks[0].includes(i)?0:1,j=d.ranks[r].indexOf(i);return{r,x:x[r]+12+j*qGap};};
   const drop0=phase(p,0,.1),drop1=phase(p,0,.1),topSlide=phase(p,.08,.26),laneSlide=phase(p,.24,.42),down0=phase(p,.40,.62),down1=phase(p,.60,.82),qShow=phase(p,0,.08)*(1-phase(p,3,3.25));
   qs.forEach((t,i)=>{const {r,x:tx}=slot(i);set(t.r,{width:qw});set(t.t,{x:qw/2});const qx=lerp(18+i*(m?41:84),tx,r?laneSlide:topSlide),qy=lerp(lerp(100,r?lane1:lane0,r?drop1:drop0),246,r?down1:down0);move(t,qx,qy,qShow);});
   localLabels.forEach((t,r)=>{text(t,x[r]+12,226,'本地位置 '+d.ranks[r].map(i=>i+1).join('/'));show(t,phase(p,.9,1.1));});
   originals.forEach((t,r)=>move(t,x[r]+12,330,phase(p,0)));
   kv.forEach((t,r)=>{const u=phase(p,1);move(t,lerp(x[r]+30,x[r]+12,u),370,u);});
   links.forEach((line,r)=>wire(line,`M${x[r]+72} 364 C${x[r]+92} 402 ${x[1-r]+92} 402 ${x[1-r]+72} 386`,phase(p,1)*(1-phase(p,2.7,3))));
   // Outputs settle into the same four slots that held their Query rows, so
   // O1…O8 stay individually readable instead of stacking two per rank.
   outputs.forEach((t,i)=>{const {x:tx}=slot(i),u=phase(p,3.3,3.6);set(t.r,{width:qw});set(t.t,{x:qw/2});move(t,tx,238+8*u,u);});
   const size=m?28:30,left=m?64:166,top=460;
   cells.forEach((c,i)=>{const row=Math.floor(i/8),col=i%8,owner=d.ranks[0].includes(row)?0:1,ordinal=d.ranks[owner].indexOf(row),allowed=row>=col,computed=phase(p,2)*4>=ordinal+1;box(c,left+col*size,top+row*size,size-3,size-3);set(c,{'data-query':row,'data-key':col,'data-owner':owner,'data-computed':computed&&allowed,class:allowed?(computed?'teach-contribution teach-color-'+owner:'teach-graph-slot'):'teach-mask'});show(c,allowed?1:.2);c.classList.toggle('tp-active',p>=1&&p<2&&row===6&&allowed);});
   rows.forEach((t,i)=>text(t,left-36,top+i*size+18));cols.forEach((t,i)=>text(t,left+i*size,top-14));
   text(result,18,722,p<1?'先看因果遮罩，再按行分工':'Query 归属固定；需要远端历史 K/V');
   const remote=d.ranks[1-(d.ranks[0].includes(6)?0:1)].filter(i=>i<=6).map(i=>i+1);
   text(dependency,18,750,'Q7 读取 K1…K7；远端位置 '+remote.join('/'));
   const finished=Math.min(4,Math.floor(phase(p,2)*4));
   text(note,18,778,'每个 Query 行：只计入未被遮罩的格子。');
   workSums.forEach((t,r)=>{const terms=d.ranks[r].slice(0,finished).map(i=>i+1);text(t,18,802+r*24,`Rank ${r}：${terms.join(' + ')||'0'} = ${terms.reduce((a,b)=>a+b,0)} 个配对`);show(t,p>=2?1:0);t.dataset.pairs=String(terms.reduce((a,b)=>a+b,0));});
   text(line2,18,856,'配对数是工作量示意，不是实测耗时。');
  };
 }else if(kind==='dcp'){
  const query=tile('dcp-query','Q = 1',4,80),copy=tile('dcp-query-copy','Q = 1',4,80);
  const source=d.values.map((v,i)=>tile('dcp-source-'+i,'V'+(i+1)+'='+v,i<2?0:1,72));
  const kLabel=i=>d.exp[i]===1?'0':d.exp[i]===3?'ln3':String(d.exp[i]);
  const kv=d.values.map((v,i)=>tile('dcp-kv-'+i,'p'+(i+1)+' K='+kLabel(i)+' V='+v,i<2?0:1,132));
  const stats=[0,1].map(i=>tile('dcp-stats-'+i,`O=${d.local[i]}, Z=${d.sums[i]}`,i,136));
  const incoming=[0,1].map(r=>tile('dcp-incoming-'+r,`O=${d.local[1-r]}, Z=${d.sums[1-r]}`,1-r,136));
  const weightBars=[rect('teach-contribution teach-color-0'),rect('teach-contribution teach-color-1')],weightLabels=[txt('','teach-small'),txt('','teach-small')];
  const equations=[0,1].map(()=>Array.from({length:5},()=>txt('','teach-small'))),before=txt('','teach-small'),derive=txt('','teach-small'),merge=txt('','teach-small');
  paint=(p,m,x,w)=>{
   text(header,18,28,'DCP：分片历史，共同算一个 Query');text(hint,18,50,'同一历史 p1…p4；每个位置都保存 K 和 V');
   source.forEach((t,i)=>move(t,18+i*(m?82:174),100,1-phase(p,0,.15)));
   text(before,18,78,d.exp[2]===3?'K=[0,0,ln3,ln3]；V=[1,3,5,7]':'K=[0,0,0,0]；V=[1,3,5,7]');show(before,p<.5?1:0);
   text(derive,18,160,'score=Q×K；exp(0)=1，exp(ln3)=3');show(derive,p>=1?1:0);
   // Q waits below both KV rows, outside every transfer path; the copy crosses
   // only the empty band under the rows.
   const qy=324;
   move(query,x[0]+12,qy);move(copy,x[1]+12,qy,phase(p,.4,.7));
   // Each shard appears just under the source row and enters one at a time,
   // lower slot first, so a later slab never passes through an earlier one.
   const kvWindows=[[.18,.38],[0,.2],[.48,.68],[.36,.56]];
   kv.forEach((t,i)=>{const r=i<2?0:1,u=phase(p,...kvWindows[i]);move(t,lerp(18+i*(m?82:174),x[r]+10,u),lerp(144,246+(i%2)*38,u),phase(p,kvWindows[i][0],kvWindows[i][0]+.04));});
   [0,1].forEach(r=>{const e=d.exp.slice(r*2,r*2+2),v=d.values.slice(r*2,r*2+2),lines=['exp 分数：'+e.join(', '),`分母 Z = ${e[0]}+${e[1]} = ${d.sums[r]}`,`分子 = ${e[0]}×${v[0]}+${e[1]}×${v[1]}`,`         = ${d.numerators[r]}`,`局部 O = ${d.numerators[r]}/${d.sums[r]} = ${d.local[r]}`];equations[r].forEach((t,i)=>{text(t,x[r]+10,386+i*26,lines[i]);show(t,p>=1?Math.min(1,phase(p,1)*5-i):0);});move(stats[r],x[r]+8,502,p>=2?1:0);move(incoming[r],lerp(x[1-r]+8,x[r]+8,phase(p,2)),540+r*38,phase(p,2));});
   text(result,18,634,p<2?'各片只读本地 KV，独立得到 O 与 Z':'交换 O / Z 副本；本地结果保留');show(result,p>=1?1:0);
   text(note,18,662,`全局分母：${d.sums[0]} + ${d.sums[1]} = ${d.sums[0]+d.sums[1]}`);show(note,phase(p,3));
   text(merge,18,690,`全局 O = (${d.sums[0]}×${d.local[0]}+${d.sums[1]}×${d.local[1]}) / ${d.sums[0]+d.sums[1]} = ${d.result}`);show(merge,phase(p,3));
   weightBars.forEach((b,i)=>{const bx=18+(i?d.weights[0]*(m?324:684):0),bw=d.weights[i]*(m?324:684);box(b,bx,726,bw,24);show(b,phase(p,3));text(weightLabels[i],bx+6,743,'片 '+i+'：'+Math.round(d.weights[i]*100)+'%');show(weightLabels[i],phase(p,3));});
   text(line2,18,774,p>=4?`权重 ${d.weights[0]} / ${d.weights[1]}；历史 KV 留在原卡。`:'');
  };
 }
 return(p,m)=>{const x=[18,m?190:382],w=m?152:320;if(kind==='dcp'){[0,1].forEach(i=>{box(panels[i],x[i],160,w,460);text(names[i],x[i]+6,184,m?'Rank '+i:'Rank '+i+' · P'+i+' / GPU '+i);show(names[i],p>=.9?1:0);});}svg.dataset.parallelKind=kind;svg.dataset.step=Math.floor(p);paint(p,m,x,w);if(m){header.setAttribute('font-size','16');result.setAttribute('font-size','15');hint.setAttribute('font-size','12');}};
 });
};
})();
