/* Matrix geometry, slice identity and arithmetic are the TP explanation. */
(()=>{'use strict';
const {board,phase,lerp}=window.TeachingDrawing;
window.TensorParallelVisual=(svg,viewport,s)=>board(svg,viewport,'tensor-matrices',()=>1100,H=>{
 const {txt,rect,path,text,box,show,wire,token,move,set}=H,d=s.data,row=d.kind==='row';
 const title=txt('Y = XW · 同一个线性层','teach-heading'),legend=txt('行：token；X 列 / W 行：输入特征','teach-small');
 const subtitle=txt('','teach-heading'),detail=txt('','teach-small'),operator=[txt('×','teach-heading'),txt('=','teach-heading')];
 const names=['X · 2×2','W · 2×2','Y · 2×2'].map(t=>txt(t,'teach-small'));
 const grid=(id,values,color=0)=>values.flatMap((r,i)=>r.map((v,j)=>{const a=token(id+'-'+i+'-'+j,String(v),32,'teach-token teach-color-'+color);set(a.r,{height:30});set(a.t,{y:20});return {...a,i,j,value:v};}));
 const full=[grid('tp-full-x',d.x),grid('tp-full-w',d.w),grid('tp-reference',d.result,2)];
 const cuts=[path('teach-map'),path('teach-map')],panels=[rect(),rect()],rankNames=[0,1].map(i=>txt('Rank '+i+' · P'+i+' / GPU '+i,'teach-small'));
 const local=d.shards.map((sh,i)=>({x:grid('tp-x'+i,sh.x,i),w:grid('tp-w'+i,sh.w,i),y:grid('tp-local'+i,d.local[i],i),xl:txt('','teach-small'),wl:txt('','teach-small'),yl:txt('','teach-small'),calc:txt('','teach-small')}));
 const incoming=[0,1].map(i=>grid('tp-incoming'+i,d.local[1-i],1-i));
 const final=[0,1].map(i=>grid('tp-final'+i,d.result,2));
 const exchange=txt('','teach-heading'),returnNames=[0,1].map(i=>txt('','teach-small')),checks=[0,1].map(()=>txt('','teach-small'));
 const link=[path(),path()],check=txt('','teach-heading'),foot=txt('原矩阵是逻辑参照；虚线标切分维度。','teach-small');
 function place(cells,x,y,gap,opacity=1){cells.forEach(c=>move(c,x+c.j*gap,y+c.i*gap,opacity));}
 return(p,m)=>{
  const gap=m?38:48,bx=m?[18,142,270]:[48,298,550],rx=[18,m?190:382],pw=m?152:320;
  text(title,18,28);text(legend,18,52);names.forEach((t,i)=>text(t,bx[i],82));operator.forEach((t,i)=>text(t,m?[108,242][i]:[194,474][i],130));
  full.forEach((cells,k)=>{place(cells,bx[k],100,gap,k<2?1-.65*phase(p,2):1);cells.forEach(c=>{
   if(k===2){c.t.textContent=p>=7?String(c.value):(p>=1&&c.i===0&&c.j===0?'4':'?');}
   else{const owner=k===0?(row?c.j:0):(row?c.i:c.j);c.g.setAttribute('class','teach-token teach-color-'+(p>=2?owner:0));c.r.classList.toggle('tp-active',p>=1&&p<2&&(k===0?c.i===0:c.j===0));}
  });});
  const cut=phase(p,1)*(1-phase(p,2));wire(cuts[0],row?`M${bx[0]+gap-3} 94 V${100+gap+36}`:'',row?cut:0);
  wire(cuts[1],row?`M${bx[1]-4} ${100+gap-4} H${bx[1]+gap+36}`:`M${bx[1]+gap-3} 94 V${100+gap+36}`,cut);
  text(subtitle,18,220,p<1?'先看两个乘数和输出形状':p<2?'一格怎样算：1×2 + 2×1 = 4':p<3?(row?'切 X 的列、W 的对应行':'切 W 的列；完整 X 各用一份'):p<4?'两卡各自执行矩阵乘法':p<5?'交换局部结果的副本':p<6?(row?'同坐标相加 → 完整输出':'按列号拼接 → 完整输出'):'与未切分的结果核对');
  text(detail,18,250,p<2?'W 的列对应输出特征；数字是教学权重。':row?'绿色：输入特征 1；蓝色：输入特征 2。':'绿色：输出列 1；蓝色：输出列 2。');
  local.forEach((l,r)=>{
   box(panels[r],rx[r],284,pw,550);text(rankNames[r],rx[r]+8,308);show(panels[r],phase(p,1));show(rankNames[r],phase(p,1));
   const sx=rx[r]+12,u=phase(p,2),calcIndex=Math.min(l.y.length-1,Math.floor(phase(p,3)*l.y.length)),active=l.y[calcIndex];
   text(l.xl,sx,338,row?'X 列分片 · 2×1':'X 完整副本 · 2×2');text(l.wl,sx,468,row?'W 行分片 · 1×2':'W 列分片 · 2×1');
   [l.xl,l.wl].forEach(t=>show(t,u));
   for(const [k,cells,y]of [[0,l.x,350],[1,l.w,480]])cells.forEach(c=>{
    const oi=k===0?c.i:(row?r:c.i),oj=k===0?(row?r:c.j):(row?c.j:r);
    move(c,lerp(bx[k]+oj*gap,sx+c.j*gap,u),lerp(100+oi*gap,y+c.i*gap,u),phase(p,1));c.r.classList.toggle('tp-active',p>=3&&p<4&&(k===0?c.i===active.i:c.j===active.j));
   });
   const total=l.y.length,progress=phase(p,3),idx=Math.min(total-1,Math.floor(progress*total)),target=l.y[idx];
   const terms=d.shards[r].x[target.i].map((v,k)=>v+'×'+d.shards[r].w[k][target.j]);
   text(l.calc,sx,580,p>=4?'局部乘积已完成':terms.join('+')+(progress*total>idx+.8?'='+target.value:''));show(l.calc,p>=3?1:0);
   text(l.yl,sx,610,row?'贡献矩阵 · 2×2':'输出列 · 2×1');show(l.yl,phase(p,2));
   l.y.forEach((c,i)=>{move(c,sx+c.j*gap,624+c.i*gap,p>=3?1:0);c.t.textContent=progress*total>=i+1?String(c.value):'?';c.r.classList.toggle('tp-active',p>=3&&p<4&&i===idx);c.g.dataset.complete=String(progress*total>=i+1);});
   text(returnNames[r],sx,724,'来自 Rank '+(1-r));show(returnNames[r],phase(p,4));
   incoming[r].forEach(c=>{const t=phase(p,4);move(c,lerp(rx[1-r]+12+c.j*gap,sx+c.j*gap,t),lerp(624+c.i*gap,740+c.i*gap,t),t);});
   final[r].forEach((c,i)=>{const complete=phase(p,5)*4>=i+1;move(c,sx+c.j*gap,930+c.i*gap,p>=5?1:0);c.t.textContent=complete?String(c.value):'?';c.g.dataset.complete=String(complete);c.r.classList.toggle('tp-active',p>=5&&p<6&&Math.floor(phase(p,5)*4)===i);});
   text(checks[r],sx,1025,p>=6?(row?'对应格求和后均为 Y':'两列按原顺序得到 Y'):'等待汇总完成');show(checks[r],p>=5?1:0);
   wire(link[r],`M${sx+120} 808 V832 M${sx+120} 870 V890`,phase(p,4));
  });
  text(exchange,18,858,row?'AllReduce：相同坐标的贡献相加':'AllGather：不同输出列按位置拼接');show(exchange,p>=4?1:0);
  const mergedIndex=Math.min(3,Math.floor(phase(p,5)*4)),mr=Math.floor(mergedIndex/2),mc=mergedIndex%2;
  text(check,18,904,p>=5&&p<6?(row?`格 (${mr+1},${mc+1})：${d.local[0][mr][mc]} + ${d.local[1][mr][mc]} = ${d.result[mr][mc]}`:`第 ${mc+1} 列第 ${mr+1} 行 → ${d.result[mr][mc]}`):p>=6?(row?'左上角：2+2=4；左下角：6+4=10':'第 1 列 [4,10]；第 2 列 [6,12]'):'');
  if(m){exchange.setAttribute('font-size','14');check.setAttribute('font-size','13');}
  text(foot,18,1080,p>=7?'结果一致；没有引入中央汇总 GPU。':'原矩阵是逻辑参照；虚线标切分维度。');
  svg.dataset.parallelKind='tp';svg.dataset.step=Math.floor(p);
 };
});
})();
