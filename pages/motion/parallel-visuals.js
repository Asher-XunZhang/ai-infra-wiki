/* Each parallel scheme visualizes the object it partitions. */
(()=>{'use strict';
const {board,phase,lerp}=window.TeachingDrawing,V=window.MechanismVisuals,pp=V['f-parallel'];
V['f-parallel']=(svg,viewport,s)=>{
 if(s.kind==='pipeline')return pp(svg,viewport,s);
 if(s.kind==='parallel-dp')return window.DataParallelVisual(svg,viewport,s);
 if(s.kind==='parallel-ep')return window.ExpertParallelVisual(svg,viewport,s);
 if(s.kind==='parallel-tp')return window.TensorParallelVisual(svg,viewport,s);
 return board(svg,viewport,s.kind,()=>1090,H=>{
 const {txt,rect,text,box,show,token,move,set}=H,d=s.data,kind=s.kind.slice(9);
 const header=txt('','teach-heading'),hint=txt('','teach-small'),panels=[rect(),rect()],names=[0,1].map(r=>txt('Rank '+r+' · P'+r+' / GPU '+r,'teach-small'));
 const note=txt('','teach-small'),result=txt('','teach-heading'),line2=txt('','teach-small');
 const tile=(id,label,color=0,w=64)=>token(id,label,w,'teach-token teach-color-'+color);
 let paint;
 if(kind==='sp'){
  const original=Array.from({length:4},(_,i)=>tile('sp-source-'+i,'p'+(i+1)+' [a'+(i+1)+',b'+(i+1)+']',i<2?0:1,116));
  const local=Array.from({length:4},(_,i)=>tile('sp-local-'+i,'p'+(i+1)+' [a'+(i+1)+',b'+(i+1)+']',i<2?0:1,116));
  const copies=Array.from({length:4},(_,i)=>tile('sp-gather-'+i,'n'+(i+1)+' [a′,b′]',i<2?0:1,116));
  const contrib=Array.from({length:8},(_,i)=>tile('sp-contrib-'+i,'c'+(i%4+1)+(i<4?'₀':'₁'),i<4?0:1,54));
  const reducing=Array.from({length:8},(_,i)=>tile('sp-reduce-'+i,'c'+(i%4+1)+(i<4?'₀':'₁'),i<4?0:1,54));
  const outputs=Array.from({length:4},(_,i)=>tile('sp-output-'+i,'y'+(i+1)+' [u,v]',i<2?0:1,116));
  const labels=[txt('完整激活 · 4 个位置 × 2 个特征','teach-small'),txt('行分片：每行的两个特征仍在一起','teach-small')];
  paint=(p,m,x,w)=>{
   text(header,18,28,'SP：序列行分片与 TP 布局转换');text(hint,18,54,'p = 输入行；n = 逐 token 变换后的行');
   text(labels[0],18,83);text(labels[1],18,244);original.forEach((t,i)=>move(t,m?128:310,94+i*34,1-.65*phase(p,0)));
   const layoutContext=1-.65*phase(p,3);
   local.forEach((t,i)=>{const r=i<2?0:1,u=phase(p,0);move(t,lerp(m?128:310,x[r]+12,u),lerp(94+i*34,294+(i%2)*42,u),u*layoutContext);t.t.textContent=(p>=2?'n':'p')+(i+1)+(p>=2?' [a′,b′]':' [a'+(i+1)+',b'+(i+1)+']');});
   // Route the AllGather copies through a staging lane above the local rows.
   // They remain concurrent but never cross the rows they are copying.
   copies.forEach((t,i)=>{const r=i<2?0:1,u=phase(p,2),lane=408+(i%2)*44,fromY=294+(i%2)*42,toY=398+(i%2)*42;let cx,cy;
    if(u<.28){const k=u/.28;cx=x[r]+12;cy=lerp(fromY,lane,k);}
    else if(u<.72){const k=(u-.28)/.44;cx=lerp(x[r]+12,x[1-r]+12,k);cy=lane;}
    else {const k=(u-.72)/.28;cx=x[1-r]+12;cy=lerp(lane,toY,k);}
    move(t,cx,cy,u*layoutContext);});
   text(result,18,530,p<1?'按序列切行，每个 token 的特征保持完整':p<2?'各 rank 独立执行逐 token 算子':p<3?'AllGather：收齐四行的副本':p<4?'TP：每卡产生四行的局部贡献':'ReduceScatter：求和后保留本 rank 的行');
   contrib.forEach((t,i)=>move(t,x[Math.floor(i/4)]+12,564+(i%4)*44,phase(p,3)));
   reducing.forEach((t,i)=>{const r=Math.floor(i/4),q=i%4;move(t,lerp(x[r]+12,x[q<2?0:1]+12+r*58,phase(p,4,4.35)),lerp(564+q*44,820+(q%2)*48,phase(p,4.35,4.8)),phase(p,4)*(1-phase(p,4.8,5)));});
   outputs.forEach((t,i)=>move(t,x[i<2?0:1]+12,820+(i%2)*48,phase(p,4.8,5)));
   text(note,18,960,'cᵢ₀ + cᵢ₁ = yᵢ；i 是位置，0/1 是贡献来源 rank。');show(note,p>=4?1:0);
   text(line2,18,1010,p>=4?'回到序列分片：每个位置的特征仍然完整。':'这里只追踪激活的分工；TP 算法见 TP 示例。');
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
  paint=(p,m,x,w)=>{
   text(header,18,28,'PCP：沿 Query 行分担 Prefill');text(hint,18,54,'矩阵格 = 一次 Q–K 配对；灰格禁止读取');
   source.forEach((t,i)=>move(t,18+i*(m?41:84),88));text(explain,18,145);
   qs.forEach((t,i)=>{const r=d.ranks[0].includes(i)?0:1,j=d.ranks[r].indexOf(i),u=phase(p,0);move(t,lerp(18+i*(m?41:84),x[r]+12+(j%2)*64,u),lerp(88,294+Math.floor(j/2)*44,u),u*(1-phase(p,3)));});
   originals.forEach((t,r)=>move(t,x[r]+12,394,phase(p,0)));
   kv.forEach((t,r)=>move(t,lerp(x[r]+12,x[1-r]+12,phase(p,1)),lerp(394,440,phase(p,1)),phase(p,1)));
   localLabels.forEach((t,r)=>{text(t,x[r]+12,384,'本地位置 '+d.ranks[r].map(i=>i+1).join('/'));show(t,phase(p,0));});
   const size=m?31:42,left=m?64:166,top=610;
   // Both ranks make independent progress; the reveal order is not a GPU schedule.
   cells.forEach((c,i)=>{const row=Math.floor(i/8),col=i%8,owner=d.ranks[0].includes(row)?0:1,ordinal=d.ranks[owner].indexOf(row),allowed=row>=col,computed=phase(p,2)*4>=ordinal+1;box(c,left+col*size,top+row*size,size-3,size-3);set(c,{'data-query':row,'data-key':col,'data-owner':owner,'data-computed':computed&&allowed,class:allowed?(computed?'teach-contribution teach-color-'+owner:'teach-graph-slot'):'teach-mask'});show(c,allowed?1:.2);c.classList.toggle('tp-active',p>=1&&p<2&&row===6&&allowed);});
   outputs.forEach((t,i)=>{const r=d.ranks[0].includes(i)?0:1,j=d.ranks[r].indexOf(i),u=phase(p,3);move(t,lerp(left+i*size,x[r]+12+(j%2)*64,u),lerp(top+i*size,294+Math.floor(j/2)*44,u),u);});
   rows.forEach((t,i)=>text(t,left-36,top+i*size+20));cols.forEach((t,i)=>text(t,left+i*size,top-14));
   text(result,18,535,p<1?'先看因果遮罩，再按行分工':'Query 归属固定；需要远端历史 K/V');
   const remote=d.ranks[1-(d.ranks[0].includes(6)?0:1)].filter(i=>i<=6).map(i=>i+1);
   text(dependency,18,568,'Q7 读取 K1…K7；远端位置 '+remote.join('/'));
   const finished=Math.min(4,Math.floor(phase(p,2)*4));
   text(note,18,985,'每个 Query 行：只计入未被遮罩的格子。');
   workSums.forEach((t,r)=>{const terms=d.ranks[r].slice(0,finished).map(i=>i+1);text(t,18,1011+r*25,`Rank ${r}：${terms.join(' + ')||'0'} = ${terms.reduce((a,b)=>a+b,0)} 个配对`);show(t,p>=2?1:0);t.dataset.pairs=String(terms.reduce((a,b)=>a+b,0));});
   text(line2,18,1070,'配对数是工作量示意，不是实测耗时。');
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
   text(header,18,28,'DCP：分片历史，共同算一个 Query');text(hint,18,54,'同一历史 p1…p4；每个位置都保存 K 和 V');source.forEach((t,i)=>move(t,18+i*(m?82:174),88,1-.65*phase(p,0)));
   text(before,18,150,d.exp[2]===3?'K=[0,0,ln3,ln3]；V=[1,3,5,7]':'K=[0,0,0,0]；V=[1,3,5,7]');
   text(derive,18,190,'score=Q×K；exp(0)=1，exp(ln3)=3');show(derive,p>=1?1:0);
   move(query,x[0]+12,294);move(copy,lerp(x[0]+12,x[1]+12,phase(p,0)),294,phase(p,0));
   kv.forEach((t,i)=>{const r=i<2?0:1,u=phase(p,0);move(t,lerp(18+i*(m?82:174),x[r]+10,u),lerp(88,350+(i%2)*44,u),u);});
   [0,1].forEach(r=>{const e=d.exp.slice(r*2,r*2+2),v=d.values.slice(r*2,r*2+2),lines=['exp 分数：'+e.join(', '),`分母 Z = ${e[0]}+${e[1]} = ${d.sums[r]}`,`分子 = ${e[0]}×${v[0]}+${e[1]}×${v[1]}`,`         = ${d.numerators[r]}`,`局部 O = ${d.numerators[r]}/${d.sums[r]} = ${d.local[r]}`];equations[r].forEach((t,i)=>{text(t,x[r]+10,470+i*29,lines[i]);show(t,p>=1?Math.min(1,phase(p,1)*5-i):0);});move(stats[r],x[r]+8,630,p>=2?1:0);move(incoming[r],lerp(x[1-r]+8,x[r]+8,phase(p,2)),lerp(630,744,phase(p,2)),phase(p,2));});
   text(result,18,712,p<2?'各片只读本地 KV，独立得到 O 与 Z':'交换 O / Z 副本；本地结果保留');show(result,p>=1?1:0);
   text(note,18,838,`全局分母：${d.sums[0]} + ${d.sums[1]} = ${d.sums[0]+d.sums[1]}`);show(note,phase(p,3));
   text(merge,18,887,`全局 O = (${d.sums[0]}×${d.local[0]}+${d.sums[1]}×${d.local[1]}) / ${d.sums[0]+d.sums[1]} = ${d.result}`);show(merge,phase(p,3));
   weightBars.forEach((b,i)=>{const bx=18+(i?d.weights[0]*(m?324:684):0),bw=d.weights[i]*(m?324:684);box(b,bx,935,bw,24);show(b,phase(p,3));text(weightLabels[i],bx+6,952,'片 '+i+'：'+Math.round(d.weights[i]*100)+'%');show(weightLabels[i],phase(p,3));});
   text(line2,18,985,p>=4?`权重 ${d.weights[0]} / ${d.weights[1]}；历史 KV 留在原卡。`:'');
  };
 }
 return(p,m)=>{const x=[18,m?190:382],w=m?152:320;[0,1].forEach(i=>{box(panels[i],x[i],250,w,kind==='dcp'?430:240);text(names[i],x[i]+6,274);if(m)names[i].setAttribute('font-size','11');});svg.dataset.parallelKind=kind;svg.dataset.step=Math.floor(p);paint(p,m,x,w);if(m){header.setAttribute('font-size','16');result.setAttribute('font-size','15');hint.setAttribute('font-size','12');}};
 });
};
})();
