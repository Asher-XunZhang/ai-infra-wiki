/* Matrix geometry, slice identity and arithmetic are the TP explanation. */
(()=>{'use strict';
const {board,phase,lerp}=window.TeachingDrawing;
window.TensorParallelVisual=(svg,viewport,s)=>board(svg,viewport,'tensor-matrices',()=>600,H=>{
 const {txt,rect,path,text,box,show,wire,token,move,set}=H,d=s.data,row=d.kind==='row';
 const title=txt('Y = XW · 同一个线性层','teach-heading');
 const legend=txt('X 行 = token；X 列 / W 行 = 输入特征','teach-small');
 const names=['X · 2×2','W · 2×2','Y · 2×2'].map(t=>txt(t,'teach-small'));
 const symbol=[txt('×','teach-heading'),txt('=','teach-heading')];
 const cell=(id,v,color=4)=>{const a=token(id,String(v),26,'teach-token teach-color-'+color);set(a.r,{width:26,height:26,rx:4});set(a.t,{x:13,y:18});return {...a,value:v};};
 const grid=(id,values,color=4)=>values.flatMap((r,i)=>r.map((v,j)=>({...cell(id+'-'+i+'-'+j,v,color),i,j})));
 const full=[grid('tp-full-x',d.x),grid('tp-full-w',d.w),grid('tp-reference',d.result,2)];
 const shards=d.shards.map((sh,r)=>({
  x:sh.x.flatMap((rowVals,i)=>rowVals.map((v,j)=>({...cell('tp-x'+r+'-'+i+'-'+j,v,r),i,j}))),
  w:sh.w.flatMap((rowVals,i)=>rowVals.map((v,j)=>({...cell('tp-w'+r+'-'+i+'-'+j,v,r),i,j}))),
  y:d.local[r].flatMap((rowVals,i)=>rowVals.map((v,j)=>({...cell('tp-local'+r+'-'+i+'-'+j,v,r),i,j,value:v})))
 }));
 const final=[0,1].map(r=>d.result.flatMap((rowVals,i)=>rowVals.map((v,j)=>({...cell('tp-final'+r+'-'+i+'-'+j,v,row?2:j),i,j,value:v}))));
 const panels=[rect(),rect()];
 const first=svg.querySelector('defs')?.nextSibling||svg.firstChild;panels.forEach(e=>svg.insertBefore(e,first));
 const rankNames=[0,1].map(i=>txt('','teach-heading')),rankSubs=[0,1].map(i=>txt('GPU '+i,'teach-small'));
 const labels=[0,1].map(()=>[txt('','teach-small'),txt('','teach-small'),txt('','teach-small')]);
 const calls=[txt('','teach-small'),txt('','teach-small')];
 const cut=[path('teach-map'),path('teach-map')];
 const exchange=txt('','teach-heading'),formula=txt('','teach-small'),detail=txt('','teach-small'),subtitle=txt('','teach-heading');
 const finalNames=[0,1].map(i=>txt('','teach-small')),check=txt('','teach-small'),foot=txt('','teach-small');
 const place=(cells,x,y,gap,size,opacity=1)=>cells.forEach(c=>{set(c.r,{width:size,height:size});set(c.t,{x:size/2,y:size*.7});move(c,x+c.j*gap,y+c.i*gap,opacity);});
 const owner=(k,c)=>k===0?(row?c.j:4):(row?c.i:c.j);
 return(p,m)=>{
  const size=m?26:30,gap=size+3,W=m?360:720,bx=m?[72,168,264]:[60,290,520],by=76;
  const pw=m?162:330,px=m?[12,186]:[24,366],py=232,ph=228;
  text(title,18,26);text(legend,18,46);names.forEach((t,i)=>text(t,bx[i],by-10));symbol.forEach((t,i)=>text(t,m?[140,236][i]:[236,466][i],by+18));
  full.forEach((cells,k)=>place(cells,bx[k],by,gap,size,p>=3?Math.max(.45,1-.55*phase(p,3)):1));
  full[2].forEach(c=>{c.t.textContent=p>=6?String(c.value):(p>=1&&c.i===0&&c.j===0?'4':'?');});
  full[0].forEach(c=>{c.g.setAttribute('class','teach-token teach-color-'+(p>=2?owner(0,c):4));c.r.classList.toggle('tp-active',p>=1&&p<2&&c.i===0);});
  full[1].forEach(c=>{c.g.setAttribute('class','teach-token teach-color-'+(p>=2?owner(1,c):4));c.r.classList.toggle('tp-active',p>=1&&p<2&&c.j===0);});
  // The two cut segments mark exactly which operand dimension is partitioned.
  const line=(e,d,n)=>{set(e,{d,'stroke-dasharray':'5 4','stroke-dashoffset':0});show(e,n);};
  if(row){
   line(cut[0],`M${bx[0]+size+1.5} ${by-4} V${by+2*size+4}`,phase(p,2));
   line(cut[1],`M${bx[1]-4} ${by+size+1.5} H${bx[1]+2*size+4}`,phase(p,2));
  }else{
   line(cut[0],`M${bx[1]+size+1.5} ${by-4} V${by+2*size+4}`,phase(p,2));
   line(cut[1],`M${bx[1]-4} ${by+size+1.5} H${bx[1]+2*size+4}`,phase(p,2));
  }
  text(subtitle,18,196,p<1?'先看完整问题：同一个 X 与 W':p<2?'先算左上角：X 的第 1 行乘 W 的第 1 列':p<3?(row?'沿输入特征切：X 切列、W 切对应行':'沿输出特征切：W 切列、X 保持完整'):p<4?'两卡同时算各自的局部块':p<5?(row?'两份贡献矩阵已就绪':'两列输出分片已就绪'):p<6?(row?'AllReduce：同坐标贡献相加':'AllGather：按列拼接输出'):'与未切分的结果核对');
  if(p<1){text(formula,18,160,'X：2 个 token × 2 个输入特征');text(detail,18,178,'W：2 个输入 × 2 个输出；Y：2 × 2');}
  else if(p<2){text(formula,18,160,'左上角：1×2 + 2×1 = 4');text(detail,18,178,'同一行与同一列逐项相乘，再把乘积相加。');}
  else{text(formula,18,160,row?'切分轴：X 的输入列 + W 的对应行':'切分轴：W 的输出列；X 两卡各一份');text(detail,18,178,row?'同色 X 列与 W 行必须交给同一个 rank。':'两卡各算一列输出；输入副本不改变权重分片。');}
  [0,1].forEach(r=>{
   const x=px[r],w=pw;
   box(panels[r],x,py,w,ph);text(rankNames[r],x+10,py+22,'Rank '+r);text(rankSubs[r],x+10,py+38,'GPU '+r+' · P'+r);
   show(panels[r],phase(p,2));show(rankNames[r],phase(p,2));show(rankSubs[r],phase(p,2));
   const left=x+10,right=m?x+86:x+166,oy=py+68;
   text(labels[r][0],left,py+50,row?'X 列 · 2×1':'X 完整 · 2×2');
   text(labels[r][1],right,py+50,row?'W 行 · 1×2':'W 列 · 2×1');
   [labels[r][0],labels[r][1]].forEach(t=>show(t,phase(p,2)));
   shards[r].x.forEach(c=>{const src=row?full[0][c.i*2+r]:full[0][c.i*2+c.j];const u=phase(p,2);move(c,lerp(src?bx[0]+(src.j)*gap:left,left+c.j*gap,u),lerp(src?by+src.i*gap:oy,oy+c.i*gap,u),u);});
   shards[r].w.forEach(c=>{const src=row?full[1][r*2+c.j]:full[1][c.i*2+r];const u=phase(p,2);move(c,lerp(src?bx[1]+src.j*gap:right,right+c.j*gap,u),lerp(src?by+src.i*gap:oy,oy+c.i*gap,u),u);});
   const total=shards[r].y.length,progress=p>=4?1:phase(p,3),idx=Math.min(total-1,Math.floor(progress*total));
   text(labels[r][2],left,py+136,row?'局部贡献 · 2×2':'输出列 · 2×1');show(labels[r][2],phase(p,2));
   shards[r].y.forEach((c,i)=>{move(c,left+c.j*gap,py+146+c.i*size,Math.max(phase(p,2)*.18,progress*total>=i+1?1:phase(p,2)*.18));c.t.textContent=progress*total>=i+1?String(c.value):'?';c.r.classList.toggle('tp-active',p>=3&&p<4&&i===idx);});
   const active=shards[r].y[idx];
   if(row){const xv=shards[r].x[active.i].value,wv=shards[r].w[active.j].value;text(calls[r],left,py+212,p>=4?'局部完成，待合并':`${xv}×${wv} = ${active.value}（格 ${active.i+1},${active.j+1}）`);}
   else{const x0=shards[r].x[active.i*2].value,x1=shards[r].x[active.i*2+1].value;text(calls[r],left,py+212,p>=4?'两列完成，待拼接':`${x0}×${shards[r].w[0].value} + ${x1}×${shards[r].w[1].value} = ${active.value}`);}
   show(calls[r],phase(p,3));
  });
  text(exchange,18,476,p<4?'合并前先确认：贡献还是输出列？':p<5?(row?'AllReduce：同坐标贡献相加':'AllGather：按列拼接输出'):p<6?(row?'AllReduce 合并中':'AllGather 拼接中'):(row?'AllReduce 完成：两份完整 Y':'AllGather 完成：两份完整 Y'));
  const fx=m?[40,210]:[150,420],fy=506;
  final.forEach((cells,r)=>{text(finalNames[r],fx[r],fy-12,'Rank '+r+' · 完整 Y');show(finalNames[r],p>=5);cells.forEach(c=>{const u=phase(p,5);move(c,fx[r]+c.j*(m?23:27),fy+c.i*(m?23:27),u);c.t.textContent=p>=6?String(c.value):'?';});});
  text(check,18,m?562:570,row?'同坐标：2+2=4，6+4=10':'第 1 列 [4,10]；第 2 列 [6,12]');show(check,p>=5?1:0);
  text(foot,18,m?578:586,p>=7?'结果一致；没有引入中央汇总 GPU。':'虚线 = 切分轴；局部结果尚未合并。');
  if(m){exchange.setAttribute('font-size','13');check.setAttribute('font-size','12');foot.setAttribute('font-size','11');}
  svg.dataset.parallelKind='tp';svg.dataset.step=Math.floor(p);
 };
});
})();
