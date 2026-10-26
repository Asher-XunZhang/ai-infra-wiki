/* TP: one board carries the single-GPU question, the two-GPU split, the local
   multiply/add, the exchange and the merged result. The same X, W and Y stay
   visible; the split tabs swap the scenario without rebuilding the lesson. */
(()=>{'use strict';
const {board,phase}=window.TeachingDrawing;
const poly=(t,pts)=>{
 const u=Math.max(0,Math.min(1,t))*(pts.length-1),i=Math.min(pts.length-2,Math.floor(u)),f=u-i,e=f*f*(3-2*f);
 return {x:pts[i][0]+(pts[i+1][0]-pts[i][0])*e,y:pts[i][1]+(pts[i+1][1]-pts[i][1])*e};
};
window.TensorParallelVisual=(svg,viewport,s)=>{
 const film=viewport.closest('.mechanism-film');if(film)film.classList.add('is-tp');
 const inner=board(svg,viewport,'tensor-matrices',m=>m?806:560,H=>{
  const {txt,rect,path,text,box,show,wire,token,move,set,node}=H;
  const d=s.data,row=d.kind==='row';
  const cell=(id,v,color=4)=>{const a=token(id,String(v),24,'teach-token teach-color-'+color);set(a.r,{width:24,height:24,rx:4});set(a.t,{x:12,y:17});return {...a,value:v};};
  const grid=(prefix,mat,color)=>mat.flatMap((r,i)=>r.map((v,j)=>({...cell(prefix+'-'+i+'-'+j,v,color),i,j,value:v})));

  /* Panels are created before the contents so no moving cell is ever hidden
     behind a white rectangle. */
  const basePanel=rect('teach-panel'),gpuPanels=[rect('teach-panel tp-gpu-panel'),rect('teach-panel tp-gpu-panel')];
  const title=txt('TP · 同一个 Y = XW，单卡与两卡','teach-heading');
  const sub=txt('X 行 = token · W 行 = 输入特征，列 = 输出特征','teach-small');
  const switchLabel=txt('切法','teach-small');
  const tabRects=[rect('tp-switch'),rect('tp-switch')];
  const tabTexts=['① 切输入维','② 切输出维'].map(v=>txt(v,'tp-switch-text'));
  const tabs=tabTexts.map((t,i)=>{
   const g=node('g',{class:'tp-switch'+(i===(row?0:1)?' is-active':''),role:'button',tabindex:'0','aria-pressed':String(i===(row?0:1)),'aria-label':i?'沿输出维切分':'沿输入维切分'});
   g.append(tabRects[i],t);return g;
  });
  const baseHead=txt('未采用 · 单卡','teach-heading');
  const baseMeta=txt('','teach-small tp-meta');
  const fieldLabels=[txt('X · 2×2','teach-small tp-label-x'),txt('W · 2×2','teach-small tp-label-w'),txt('Y · 2×2','teach-small tp-label-y')];
  const baseNotes=[0,1,2].map(()=>txt('','teach-small'));
  const fullX=grid('tp-full-x',d.x,0),fullW=grid('tp-full-w',d.w,2),fullY=grid('tp-reference',d.result,1);
  const cutX=path('teach-map'),cutW=path('teach-map');
  const flow=txt('切分 → 2 张 GPU，各算一部分','teach-small tp-flow'),flowPath=path('tp-flow-line');

  const shardX=d.shards.map((sh,r)=>sh.x.flatMap((vals,i)=>vals.map((v,j)=>({...cell('tp-x'+r+'-'+i+'-'+j,v,row?r:0),i,j,value:v}))));
  const shardW=d.shards.map((sh,r)=>sh.w.flatMap((vals,i)=>vals.map((v,j)=>({...cell('tp-w'+r+'-'+i+'-'+j,v,r),i,j,value:v}))));
  const local=d.local.map((mat,r)=>mat.flatMap((vals,i)=>vals.map((v,j)=>({...cell('tp-local'+r+'-'+i+'-'+j,v,r),i,j,value:v}))));
  const final=d.result.map((vals,i)=>vals.map((v,j)=>({...cell('tp-final0-'+i+'-'+j,v,1),i,j,value:v})));
  const final1=d.result.map((vals,i)=>vals.map((v,j)=>({...cell('tp-final1-'+i+'-'+j,v,1),i,j,value:v})));

  const gpuNames=[0,1].map(r=>txt('GPU '+r+' · P'+r,'teach-heading'));
  const gpuMeta=[0,1].map(()=>txt('','teach-small tp-meta'));
  const opLabels=[0,1].map(()=>[txt('','teach-small'),txt('','teach-small')]);
  const opMul=[0,1].map(()=>txt('×','teach-text tp-operator'));
  const localLabels=[0,1].map(()=>txt('','teach-small'));
  const formulas=[0,1].map(()=>[0,1,2].map(()=>txt('','teach-small tp-formula')));
  const statuses=[0,1].map(()=>txt('','teach-small tp-status'));
  const commTitle=txt('','teach-heading tp-comm-title'),commSub=txt('','teach-small tp-comm-sub');
  const commArrows=[path('tp-comm-line'),path('tp-comm-line')];
  const finalLabels=[0,1].map(r=>txt('GPU '+r+' · 完整 Y','teach-small tp-final-label'));
  const check=txt('','teach-text tp-check');check.dataset.tpCheck='same-coord';
  const foot1=txt('','teach-small tp-foot'),foot2=txt('','teach-small tp-foot');

  /* Payloads and the merge station make the exchange visible. They are kept
     above the final matrices so a paused frame still shows what is moving. */
  const mergeChips=[0,1].map(r=>token('tp-merge-chip-'+r,'',32,'teach-token tp-merge-chip teach-color-'+r));
  const mergePlus=txt('+','teach-text tp-operator'),mergeEq=txt('=','teach-text tp-operator'),mergeJoin=txt('∥','teach-text tp-operator');
  const mergeResult=token('tp-merge-result','',24,'teach-token teach-color-1');
  [mergePlus,mergeEq,mergeJoin].forEach(t=>t.setAttribute('text-anchor','middle'));

  const activate=kind=>{if(kind===(row?'row':'column'))return;svg.dispatchEvent(new CustomEvent('parallel-tp-split',{bubbles:true,detail:{kind}}));};
  tabs.forEach((g,i)=>{
   const kind=i?'column':'row';
   g.addEventListener('click',()=>activate(kind));
   g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.stopPropagation();activate(kind);}});
  });

  const put=(c,x,y,size,opacity=1)=>{set(c.r,{width:size,height:size,rx:4});set(c.t,{x:size/2,y:size*.7});move(c,x+c.j*(size+2),y+c.i*(size+2),opacity);};

  return(p,m)=>{
   const W=m?360:720,baseX=m?[24,88,152]:[32,100,168],baseY=m?140:112,cs=m?22:24,step=cs+2;
   const a=d.local[0],b=d.local[1];
   title.setAttribute('x',m?12:18);title.setAttribute('y',m?24:26);
   sub.setAttribute('x',m?12:18);sub.setAttribute('y',m?42:46);
   switchLabel.setAttribute('x',m?0:484);switchLabel.setAttribute('y',32);show(switchLabel,m?0:1);
   const tabW=m?162:86,tabY=m?52:14,tabX=m?[12,186]:[512,604];
   tabs.forEach((g,i)=>{set(g,{transform:'translate('+tabX[i]+' '+tabY+')',class:'tp-switch'+(i===(row?0:1)?' is-active':''),'aria-pressed':String(i===(row?0:1))});box(tabRects[i],0,0,tabW,28);text(tabTexts[i],tabW/2,18,tabTexts[i].textContent);tabTexts[i].setAttribute('text-anchor','middle');});

   /* Single-GPU reference: the same matrices remain visible throughout. */
   const baseRight=(m?348:702),baseBottom=(m?216:188);
   box(basePanel,m?12:18,m?90:62,m?336:684,m?126:126);
   text(baseHead,m?24:30,m?112:84);text(baseMeta,baseRight-12,m?112:84);
   baseMeta.setAttribute('text-anchor','end');
   const baseOp=p>=3.6?.62:1;
   [fieldLabels[0],fieldLabels[1],fieldLabels[2]].forEach((t,i)=>{text(t,baseX[i],m?134:104);t.setAttribute('class','teach-small '+(i===0?'tp-label-x':i===1?'tp-label-w':'tp-label-y'));});
   fullX.forEach(c=>{const cls='teach-token '+(p>=2&&row?'teach-color-'+c.j:'teach-color-0');set(c.g,{class:cls});put(c,baseX[0],baseY,cs,baseOp);});
   fullW.forEach(c=>{const cls='teach-token '+(p>=2?(row?'teach-color-'+c.i:'teach-color-'+c.j):'teach-color-2');set(c.g,{class:cls});put(c,baseX[1],baseY,cs,baseOp);});
   if(p>=2&&!row)fullX.forEach(c=>c.g.classList.add('tp-shared'));
   fullY.forEach(c=>{set(c.g,{class:'teach-token teach-color-1'});put(c,baseX[2],baseY,cs,baseOp);});
   const note1={x:m?24:240,y:m?204:108},note2={x:m?24:240,y:130},note3={x:m?24:240,y:152};
   const n1=p<1?'单卡一次算完 4 个输出格。':p<2?'左上角：1×2 + 2×1 = 4':p<3?(row?'沿输入维切：X 切列，W 切对应行':'沿输出维切：W 切列，X 两卡各一份'):p<4?'分片来自同一份 X 与 W。':p<6?'两卡各算 4 个乘积项。':'两侧 Y 必须逐格相同。';
   const n2=p<1?'每个 Y 格 = 2 个乘积相加。':p<2?'同一行乘同一列，再相加。':p<3?'同色 = 同一 rank；虚线 = 切分轴。':p<4?'数值不变；只是换一张卡算。':p<6?'合并前先确认：贡献还是输出列？':'[[4,6],[10,12]]';
   const n3=p<1?'W 的 4 个格子全在这张卡上。':p<2?'其余格用同一规则。':p<3?(row?'X 第 1/2 列分别配对 W 第 1/2 行':'两卡各持一列 W；X 复制到两卡。'):p<4?(row?'注意：局部贡献还不是完整 Y。':'输出列已经覆盖自己那一列。'):p<6?'合并通信另计，不折算加速。':'完整恢复 Y 是本例的选择。';
   text(baseNotes[0],note1.x,note1.y,n1);text(baseNotes[1],note2.x,note2.y,n2);text(baseNotes[2],note3.x,note3.y,n3);
   show(baseNotes[1],m?0:1);show(baseNotes[2],m?0:1);
   baseMeta.textContent=m?'1 GPU · W 4 格':'1 GPU · W 4 格 · 8 乘 · 4 加';

   const cutOp=phase(p,1.5,2.05)*(1-phase(p,3.05,3.35));
   if(row){wire(cutX,'M'+(baseX[0]+cs+1)+' '+(baseY-5)+' V'+(baseY+2*cs+7),cutOp);show(cutX,cutOp);wire(cutW,'M'+(baseX[1]-5)+' '+(baseY+cs+1)+' H'+(baseX[1]+2*cs+7),cutOp);show(cutW,cutOp);}
   else{wire(cutW,'M'+(baseX[1]+cs+1)+' '+(baseY-5)+' V'+(baseY+2*cs+7),cutOp);show(cutW,cutOp);show(cutX,0);}

   /* The flow strip is hidden only while shard payloads occupy the band under
      the reference. That keeps p2–3 pauses free of text over moving digits. */
   const flowOn=p<1.9||p>=3.05;
   text(flow,W/2,m?230:204,'切分 → 2 张 GPU，各算一部分');flow.setAttribute('text-anchor','middle');
   flow.setAttribute('opacity',flowOn?(p>=3?'.45':'1'):'0');
   if(m)wire(flowPath,'M120 236 H240',1);else wire(flowPath,'M320 214 H400',1);flowPath.classList.add('tp-flow-line');
   show(flowPath,flowOn?1:0);

   /* Two GPU panels: operands, local result, formula and a short status. */
   const gpuY=m?[242,442]:[224,224],gpuX=m?[12,12]:[18,368],gpuW=m?336:334,gpuH=188;
   const flight=row
    ?[{x:[2,2.55],w:[1.95,2.35]},{x:[2.3,2.9],w:[2.52,3]}]
    :[{x:[2.05,2.4],w:[1.95,2.3]},{x:[2.15,2.75],w:[2.15,2.75]}];
   [0,1].forEach(r=>{
    const gx=gpuX[r],gy=gpuY[r],xTarget=gx+22,wTarget=gx+(row?64:88),localX=gx+232,localY=gy+48;
    box(gpuPanels[r],gx,gy,gpuW,gpuH);gpuPanels[r].classList.toggle('is-pending',p<2.9);
    text(gpuNames[r],gx+12,gy+24);
    text(gpuMeta[r],gx+gpuW-12,gy+24,row?'2 格 · 4 乘 · 0 加':'2 格 · 4 乘 · 2 加');gpuMeta[r].setAttribute('text-anchor','end');
    text(opLabels[r][0],xTarget,gy+40,row?'X 列 '+(r+1):'X 完整');text(opLabels[r][1],wTarget,gy+40,row?'W 行 '+(r+1):'W 列 '+(r+1));
    text(localLabels[r],localX,gy+40,row?'局部贡献 C'+r:'输出列 C'+r);
    const labelsOn=p>=2.95?1:0;[opLabels[r][0],opLabels[r][1],localLabels[r]].forEach(t=>show(t,labelsOn));
    text(opMul[r],gx+(row?55:79),gy+68,'×');opMul[r].setAttribute('text-anchor','middle');show(opMul[r],p>=3?1:0);

    /* Shards leave the reference from below its numeric rows, then follow a
       clean corridor inside the target panel. No path crosses a static matrix
       number or a panel label; the row's second W row waits until the first
       column shard has cleared the tray. */
    const ty=gy+48,fade=phase(p,1.86,1.99),fadeW=r===1&&row?phase(p,2.46,2.58):fade;
    const tX=phase(p,flight[r].x[0],flight[r].x[1]),tW=phase(p,flight[r].w[0],flight[r].w[1]);
    if(m){
     shardX[r].forEach(c=>{const q=poly(tX,[[xTarget,gy+28],[xTarget,ty]]);move(c,q.x+c.j*step,q.y+c.i*step,fade);});
     shardW[r].forEach(c=>{const q=poly(tW,[[wTarget,gy+28],[wTarget,ty]]);move(c,q.x+c.j*step,q.y+c.i*step,fadeW);});
    }else if(row){
     const xPath=r===0?[[baseX[0],164],[2,164],[2,ty],[xTarget,ty]]:[[baseX[0]+step,164],[352,164],[352,ty],[xTarget,ty]];
     const wPath=r===0?[[baseX[1],164],[150,164],[150,ty],[wTarget,ty]]:[[baseX[1],190],[520,190],[520,ty],[wTarget,ty]];
     shardX[r].forEach(c=>{const q=poly(tX,xPath);move(c,q.x,q.y+c.i*step,fade);});
     shardW[r].forEach(c=>{const q=poly(tW,wPath);move(c,q.x+c.j*step,q.y+c.i*step,fadeW);});
    }else{
     /* X is replicated in the column split, so its copies settle in place
        instead of crossing the W corridor. */
     const drop=phase(p,2,2.35);
     shardX[r].forEach(c=>move(c,xTarget+c.j*step,ty+c.i*step+7*(1-drop),drop));
     const wPath=r===0?[[baseX[1],164],[150,164],[150,ty],[wTarget,ty]]:[[180,164],[520,164],[520,ty],[wTarget,ty]];
     shardW[r].forEach(c=>{const q=poly(tW,wPath);move(c,q.x,q.y+c.i*step,fade);});
    }

    const total=d.local[r].length*d.local[r][0].length,progress=p>=4?1:phase(p,3,4),idx=Math.min(total-1,Math.floor(progress*total));
    local[r].forEach((c,k)=>{move(c,localX+c.j*step,localY+c.i*step,p>=3?1:0);c.t.textContent=(p>=4||progress*total>=k+1)?String(c.value):'?';c.r.classList.toggle('tp-active',p>=3&&p<4&&k===idx);});
    const active=local[r][idx],i=active.i,j=active.j;
    let f1='',f2='',f3='';
    if(p<3){f1='切分后：每卡只算自己的一份';f2='局部数值尚未计算';}
    else if(row){const xv=shardX[r][i*1+0].value;/* X shard is 2×1 */const wv=shardW[r][j].value;f1='C'+r+'['+i+','+j+'] = '+xv+'×'+wv+' = '+active.value;f2='本卡 4 乘 · 0 局部加（每格 1 乘积）';}
    else{const x0=shardX[r][i*2].value,x1=shardX[r][i*2+1].value,w0=shardW[r][0].value,w1=shardW[r][1].value;f1='C'+r+'['+i+',0] = '+x0+'×'+w0+' + '+x1+'×'+w1+' = '+active.value;f2='本卡 4 乘 · 2 局部加（每格 2 乘积）';}
    if(p>=3){
     if(p<4.4)f3=row?'权重 2 格；同坐标贡献待合并':'权重 2 格；输出列待拼接';
     else if(p<5.65)f3=row?'交换局部贡献：等待同坐标相加':'交换输出列：等待另一列到达';
     else f3=row?('同坐标已相加：'+a[0][0]+'+'+b[0][0]+'='+(a[0][0]+b[0][0])+'、'+a[1][0]+'+'+b[1][0]+'='+(a[1][0]+b[1][0])):('已拼接：列1 ['+a[0][0]+','+a[1][0]+']←P0；列2 ['+b[0][0]+','+b[1][0]+']←P1');
    }
    formulas[r].forEach((t,k)=>text(t,gx+14,gy+118+k*18,[f1,f2,f3][k]));
    const status=p<2?'等待切分':p<2.95?'接收分片':p<4?'本地计算':p<4.9?(row?'待合并':'待拼接'):p<5.65?'交换结果分片':p<6.75?(row?'同坐标相加中':'按列拼接中'):'完整 Y 已就绪';
    text(statuses[r],gx+gpuW-12,gy+178,status);statuses[r].setAttribute('text-anchor','end');
   });

   /* Communication and merge: one strip, two payload-carrying arrows, then a
      station that shows the tracked operation before any final value appears. */
   const cy=m?646:426,csY=m?662:442,ay=m?672:452,by=m?680:460;
   const draw=phase(p,4.4,4.85);
   text(commTitle,W/2,cy,p<4.4?'合并前：局部结果还缺什么？':p<5.5?(row?'AllReduce：交换局部贡献':'AllGather：交换输出列'):p<6.75?(row?'AllReduce：逐坐标相加':'AllGather：逐列拼回'):(row?'AllReduce 完成：4 次数学加法':'AllGather 完成：0 次跨卡数学加法'));commTitle.setAttribute('text-anchor','middle');
   text(commSub,W/2,csY,p<4.4?(row?'贡献矩阵还不是完整 Y；每个输出格只收到一半乘积。':'每个输出列只覆盖自己那一列；另一列在另一张卡。'):p<5.5?(row?'两卡交换局部贡献；同坐标还缺另一半。':'两卡交换输出列；每卡还缺另一列。'):p<6.75?(row?'贡献已到齐；逐坐标相加，其余坐标同规则。':'列色即来源：青 = GPU0，蓝 = GPU1。'):'每个 rank 都得到完整 Y；通信次数/成本未计。');commSub.setAttribute('text-anchor','middle');show(commSub,(p>=4.45&&p<5.3)?0:1);
   if(m){wire(commArrows[0],'M60 '+ay+' H150',draw);wire(commArrows[1],'M300 '+by+' H210',draw);}
   else{wire(commArrows[0],'M170 '+ay+' H340',draw);wire(commArrows[1],'M550 '+by+' H380',draw);}

   const mergeY=m?742:520;
   const rowChipW=m?28:32,colChipW=m?47:54;
   const rowAX=m?124:170,rowBX=m?162:214,rowPlusX=m?158:208,rowEqX=m?196:252,rowResX=m?200:260;
   const colAX=m?122:156,colJoinX=m?175:218,colBX=m?181:226;
   const chipTargets=row?[[rowAX,mergeY-12],[rowBX,mergeY-12]]:[[colAX,mergeY-12],[colBX,mergeY-12]];
   const chipVals=row?[String(a[0][0]),String(b[0][0])]:['['+a[0][0]+','+a[1][0]+']','['+b[0][0]+','+b[1][0]+']'];
   mergeChips.forEach((c,i)=>{
    const cw=row?rowChipW:colChipW;
    c.t.textContent=chipVals[i];set(c.r,{width:cw,height:24,rx:4});set(c.t,{x:cw/2,y:17});
    c.g.setAttribute('aria-label',row?'GPU '+i+' 的局部贡献 '+chipVals[i]:'GPU '+i+' 的输出列 '+chipVals[i]);
   });
   const travel=phase(p,4.55,5.25),chipOp=phase(p,4.5,4.65)*(1-phase(p,5.65,5.8));
   const start0=m?[18,ay-12]:[178,ay-12],start1=m?[282,by-12]:[530,by-12];
   const mid0=m?[100,ay-12]:[300,ay-12],mid1=m?[250,by-12]:[400,by-12];
   const q0=poly(travel,[start0,mid0,chipTargets[0]]),q1=poly(travel,[start1,mid1,chipTargets[1]]);
   move(mergeChips[0],q0.x,q0.y,chipOp);move(mergeChips[1],q1.x,q1.y,chipOp);

   const symIn=phase(p,5.25,5.38),symOut=1-phase(p,5.65,5.8),resIn=phase(p,5.4,5.52);
   text(mergePlus,rowPlusX,mergeY+4,'+');text(mergeEq,rowEqX,mergeY+4,'=');text(mergeJoin,colJoinX,mergeY+4,'∥');
   [mergePlus,mergeEq].forEach(t=>show(t,row?symIn*symOut:0));show(mergeJoin,row?0:symIn*symOut);
   set(mergeResult.r,{width:24,height:24,rx:4});set(mergeResult.t,{x:12,y:17});mergeResult.t.textContent=String(a[0][0]+b[0][0]);
   move(mergeResult,rowResX,mergeY-12,row?resIn*symOut:0);

   const checkText=row?'✓ 同坐标：2+2=4，6+4=10':'✓ 第 1 列 [4,10]、第 2 列 [6,12]';
   text(check,W/2,m?694:478,checkText);check.setAttribute('text-anchor','middle');show(check,p>=6.85?1:0);

   /* Final Y waits for the exchange. The tracked row coordinate emerges from
      the 2+2 station; columns slide from their arriving strips. */
   const finalOp=phase(p,5.1,5.28);
   const rowReveal=[[5.9,6.35],[6.15,6.55]],colReveal=[5.85,6.1];
   const fx=m?[70,230]:[34,384],fy=m?714:498,fs=22;
   [final,final1].forEach((rows,r)=>{
    const lab=finalLabels[r];text(lab,fx[r],m?704:492);show(lab,finalOp);
    rows.flat().forEach(c=>{
     const reveal=row?rowReveal[c.i][c.j]:colReveal[c.j];
     const t=phase(p,reveal,reveal+.24);
     const tx=fx[r]+c.j*(fs+2),ty2=fy+c.i*(fs+2);
     let sx=tx,sy=ty2+8;
     if(row&&c.i===0&&c.j===0){sx=rowResX+(r?6:-6);sy=mergeY-12;}
     else if(!row){sx=tx;sy=mergeY-12+c.i*(fs+2);}
     move(c,sx+(tx-sx)*t,sy+(ty2-sy)*t,t);
     c.t.textContent=t>0?String(c.value):'';
     const cls='teach-token '+(row?'teach-color-1':'teach-color-'+c.j)+(row&&c.i===0&&c.j===0?' tp-active':'');
     set(c.g,{class:cls});
     c.g.dataset.origin=row?'gpu0+gpu1':'gpu'+c.j;
     c.g.dataset.reveal=reveal.toFixed(2);
    });
   });

   if(m){foot1.setAttribute('x',180);foot1.setAttribute('y',776);foot1.setAttribute('text-anchor','middle');foot1.textContent='数学工作量 8 乘 · 4 加；额外 1 GPU + 通信换分担，';foot2.setAttribute('x',180);foot2.setAttribute('y',792);foot2.setAttribute('text-anchor','middle');foot2.textContent='不是 2 倍性能；通信成本另计，完整 Y 为本例选择。';}
   else{foot1.setAttribute('x',360);foot1.setAttribute('y',554);foot1.setAttribute('text-anchor','middle');foot1.textContent='本例数学工作量 8 乘 · 4 加；通信执行次数/成本另计，不是 2 倍性能。';foot2.textContent='';}
   /* The last row cell lands at 6.79; the footer must not share glyph space
      with a cell that is still sliding into its final grid. */
   const footOp=phase(p,6.8,6.95);show(foot1,footOp);show(foot2,footOp);

   svg.dataset.parallelKind='tp';svg.dataset.step=Math.floor(p);
  };
 });
 const destroy=inner.destroy;inner.destroy=()=>{film?.classList.remove('is-tp');destroy();};return inner;
};
})();
