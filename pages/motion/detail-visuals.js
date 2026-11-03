/* Dedicated encodings: arithmetic, dependencies, input/output alignment and ownership.
 * The timeline only reveals model facts. DOM objects survive reverse seeking. */
(()=>{
'use strict';
const {board,phase,lerp,clamp}=window.TeachingDrawing,V=window.MechanismVisuals;
function resourceComparison(svg,viewport,s){return board(svg,viewport,'resource-comparison',()=>710,H=>{
 const {txt,rect,path,text,box,show,wire,token,move,set}=H,d=s.data;
 const baseline=d.baseline||{work:[8,6],rates:[1,2]},heading=txt('','teach-heading'),legend=txt('','teach-small'),legendCompare=txt('','teach-small');
 const rows=d.work.map((count,i)=>({bg:rect(),title:txt('','teach-heading'),rate:txt('','teach-text'),badge:txt('×2','teach-badge'),note:txt('每列 1 秒','teach-small'),
  units:Array.from({length:count},(_,j)=>token('resource-work-'+i+'-'+j,String(j+1),26,'teach-token teach-color-'+(i?2:1))),
  slots:Array.from({length:8},()=>rect('teach-slot')),ticks:Array.from({length:9},(_,j)=>txt(String(j),'teach-small')),
  axis:path(),bar:rect(i?'teach-compute':'teach-value teach-color-1'),refBar:rect('teach-reference-bar'),tip:txt('','teach-small'),refNote:txt('','teach-small'),formula:txt('','teach-number')}));
 const sumA=txt('','teach-small'),result=txt('','teach-heading'),why=txt('','teach-small'),explain=txt('','teach-small'),reference=txt('','teach-small');
 return(p,m)=>{const W=m?360:720,x=24,cw=(W-48)/8,u=phase(p,1),timeVisible=phase(p,2),concluded=p>=2.4;
  text(heading,18,30,m?'先数工作量，再按秒分组':'先数工作量，再按每秒容量分组');
  text(legend,18,51,m?'数字是编号；两组同为 8 / 6 份工作。':'同色格代表等量工作；格内数字是编号，当前与原配使用同一组 8 / 6 份工作。');
  text(legendCompare,18,66,m?'实线当前；虚线原配，同用 8 秒刻度。':'实心条为当前条件；虚线为原配，两者同用 8 秒刻度。');
  rows.forEach((r,i)=>{const y=(m?84:86)+i*238,rate=d.rates[i],seconds=d.work[i]/rate,refRate=baseline.rates[i],refSeconds=baseline.work[i]/refRate,panelW=W-24,changed=d.changed===(i?'compute':'bandwidth');
   r.bg.classList.toggle('is-changed',changed);
   box(r.bg,12,y,panelW,236);text(r.title,26,y+27,i?'计算量：6 份计算':'数据量：8 份数据');
   text(r.rate,26,y+51,m?(i?'算力 ':'带宽 ')+rate+' 份/秒':(i?'算力：每秒 '+rate+' 份计算':'带宽：每秒 '+rate+' 份数据'));
   text(r.note,panelW-2,y+51,'每列 1 秒');set(r.note,{'text-anchor':'end'});
   text(r.badge,panelW-2,y+27,'×2');set(r.badge,{'text-anchor':'end'});show(r.badge,changed?1:0);
   r.slots.forEach((slot,j)=>{box(slot,x-6+j*cw,y+62,cw-3,102);show(slot,u*.8);});
   r.units.forEach((a,j)=>{const col=Math.floor(j/rate),row=j%rate;
    move(a,lerp(x+j*cw+6,x+col*cw+6,u),lerp(y+66,y+70+row*23,u),1);set(a.r,{height:22,rx:3});set(a.t,{y:16});
   });
   wire(r.axis,`M${x-6} ${y+200} H${W-18}`,u);
   r.ticks.forEach((tick,j)=>{text(tick,j===8?W-18:x-9+j*cw,y+216,j===8?'8 秒':String(j));set(tick,{'text-anchor':j===8?'end':'start'});show(tick,u);});
   box(r.bar,x-6,y+172,seconds*cw*u,8);r.bar.dataset.motionId='resource-demand-'+i;r.bar.dataset.seconds=String(seconds);r.bar.dataset.work=String(d.work[i]);r.bar.dataset.rate=String(rate);
   box(r.refBar,x-6,y+186,refSeconds*cw*u,7);r.refBar.dataset.motionId='resource-reference-'+i;r.refBar.dataset.seconds=String(refSeconds);r.refBar.dataset.work=String(baseline.work[i]);r.refBar.dataset.rate=String(refRate);show(r.refBar,u*.9);
       const tipX=m&&rate>1?Math.min(W-18,x+(Math.ceil(d.work[i]/rate)-1)*cw+36):x-6+seconds*cw;
    text(r.tip,tipX,y+168,seconds+' 秒');set(r.tip,{'text-anchor':m&&rate>1?'start':'end'});show(r.tip,u);
   text(r.formula,26,y+232,`${d.work[i]} 份 ÷ ${rate} 份/秒 = ${seconds} 秒`);show(r.formula,u);
   text(r.refNote,panelW-2,y+232,refSeconds===seconds?'原配同为 '+refSeconds+' 秒':'原配 '+refSeconds+' 秒');set(r.refNote,{'text-anchor':'end'});show(r.refNote,timeVisible);
  });
  const moveSeconds=d.work[0]/d.rates[0],computeSeconds=d.work[1]/d.rates[1],baseMove=baseline.work[0]/baseline.rates[0],baseCompute=baseline.work[1]/baseline.rates[1];
  const explainText=concluded?d.changed==='compute'?(m?`原配算 ${baseCompute} 秒，现在 ${computeSeconds} 秒；搬运仍 ${moveSeconds} 秒。`:`原配计算 ${baseCompute} 秒；算力翻倍后 ${computeSeconds} 秒。搬运同为 ${moveSeconds} 秒，下界取 ${d.bound} 秒。`):d.changed==='bandwidth'?(m?`原配搬 ${baseMove} 秒，现在 ${moveSeconds} 秒；计算同为 ${computeSeconds} 秒。`:`原配搬运 ${baseMove} 秒；带宽翻倍后 ${moveSeconds} 秒。计算同为 ${computeSeconds} 秒，下界取 ${d.bound} 秒。`):(m?`原配：搬 ${baseMove} 秒，算 ${baseCompute} 秒，取 ${d.bound} 秒。`:`原配即当前条件：搬运 ${baseMove} 秒，计算 ${baseCompute} 秒，下界取 ${d.bound} 秒。`):'每列 1 秒；数据与计算分开核算。';
  text(sumA,18,576,m?`搬运 ${moveSeconds} 秒 · 计算 ${computeSeconds} 秒`:`搬运下界 ${moveSeconds} 秒 · 计算下界 ${computeSeconds} 秒`);
  show(sumA,timeVisible);
  text(result,18,604,concluded?`整体至少 ${d.bound} 秒`:'两项资源分别核算');
  text(why,18,630,concluded?(d.bound===moveSeconds?'受带宽限制':'受算力限制'):'取两项中的较大值');
  text(explain,18,656,explainText);
  text(reference,18,684,concluded?(m?'max 只是下界，不是实测时间/加速比':'整体下界 = max(搬运下界, 计算下界)；不是实测时间或加速比'):'');show(reference,timeVisible);
  if(m){set(result,{'font-size':17});set(explain,{'font-size':13});}
 };
});}
function matrixArithmetic(svg,viewport,s){return board(svg,viewport,'arithmetic',()=>670,H=>{
 const {txt,rect,text,box,show,token,move,set}=H,d=s.data;
 const heading=txt('y = W × x','teach-heading'),subtitle=txt('','teach-small'),wTitle=txt('W · 权重 2×2','teach-small'),xTitle=txt('x · 输入 2×1','teach-small');
 const weights=d.weights.map((row,i)=>row.map((n,j)=>token('weight-'+i+'-'+j,String(n),40,'teach-token teach-color-'+i)));
 const input=d.input.map((n,i)=>token('operand-'+i,String(n),40,'teach-token'));
 const multiply=txt('×','teach-heading'),rows=[0,1].map(i=>({title:txt('第 '+(i+1)+' 行','teach-heading'),parts:[0,1].map(j=>token('product-'+i+'-'+j,d.weights[i][j]+'×'+d.input[j],74,'teach-token teach-color-'+i)),plus:txt('+','teach-heading'),equals:txt('=','teach-heading'),result:token('row-result-'+i,String(d.output[i]),54,'teach-token teach-color-2')}));
 const outputTitle=txt('按原行号组成 y','teach-heading'),outputPanel=rect(),outputLabel=txt('y','teach-heading'),output=d.output.map((n,i)=>token('output-'+i,String(n),54,'teach-token teach-color-2'));
 const note=txt('','teach-small');
 return(p,m)=>{const start=m?30:130,xpos=m?238:440,rowX=m?26:142,cardW=m?84:74,span=m?100:144;
  text(heading,18,30);text(subtitle,18,54,m?'对应相乘，再按行相加':'对应位置相乘，再按行相加；两行独立计算。');
  text(wTitle,start,84);text(xTitle,xpos-10,84,m?'x · 2×1':'x · 输入 2×1');text(multiply,m?194:344,128);
  weights.forEach((row,i)=>row.forEach((t,j)=>move(t,start+j*52,104+i*44)));input.forEach((t,i)=>move(t,xpos,104+i*44));
  rows.forEach((r,i)=>{const y=244+i*120;text(r.title,18,y-16);
   r.parts.forEach((t,j)=>{move(t,lerp(start+j*52,rowX+j*span,phase(p,0)),lerp(104+i*44,y,phase(p,0)),phase(p,0));set(t.r,{width:cardW});set(t.t,{x:cardW/2});t.t.textContent=d.weights[i][j]+'×'+d.input[j]+'='+String(d.weights[i][j]*d.input[j]);});
   text(r.plus,rowX+cardW+(span-cardW)/2,y+24);set(r.plus,{'text-anchor':'middle'});show(r.plus,phase(p,1));text(r.equals,rowX+span+cardW+12,y+24);set(r.equals,{'text-anchor':'middle'});show(r.equals,phase(p,1));move(r.result,rowX+span+cardW+24,y,phase(p,1));
  });
  const panelX=m?246:500,panelY=m?452:456,panelW=m?100:150,panelH=m?128:150;
  box(outputPanel,panelX,panelY,panelW,panelH);show(outputPanel,phase(p,2));
  text(outputTitle,18,m?434:440,m?'按原行号组成 y':'按原行号组成 y · 2×1');show(outputTitle,phase(p,2));
  text(outputLabel,panelX+10,panelY+24);show(outputLabel,phase(p,2));
  output.forEach((t,i)=>{const u=phase(p,2),fromX=rowX+span+cardW+24,toY=panelY+38+i*44,lane=panelX+panelW-70;
   // Enter from the right-hand lane so the first output never crosses the
   // second row result or the y caption; only the last leg slides into place.
   let x=lane,y=lerp(244+i*120,toY,u);
   if(u<.2)x=lerp(fromX,lane,u/.2);
   else if(u<.82)x=lane;
   else x=lerp(lane,panelX+16,(u-.82)/.18);
   if(u<.2)y=lerp(244+i*120,toY,u);
   move(t,x,y,u);
  });
  text(note,18,620,m?'两行独立计算；结果是新数值。':'先相乘，再按行相加；两行可以独立计算。');
  if(m){note.setAttribute('font-size','13');subtitle.setAttribute('font-size','13');}
 };
});}
function graphSubmission(svg,viewport,s){return board(svg,viewport,'graph-submission',()=>730,H=>{
 const {txt,rect,path,text,box,show,wire,set}=H,d=s.data.schedule;
 const title=txt('','teach-heading'),meaning=txt('','teach-small'),meaning2=txt('','teach-small');
 const prep=[rect('teach-graph-slot'),rect('teach-graph-slot')],prepText=[txt('','teach-small'),txt('','teach-small')],prepLink=path();
 const prepNote=txt('','teach-small'),prepCost=txt('','teach-small'),roundLabel=txt('','teach-heading'),history=txt('','teach-small'),axis=path();
 const ticks=Array.from({length:d.duration+1},(_,i)=>txt(String(i),'teach-small'));
 function event(e,id,cls){return {...e,name:e.id,outline:rect('teach-slot'),fill:rect(cls),label:txt(e.id,'teach-small'),id};}
 const panels=[0,1].map(i=>({bg:rect(),title:txt(i?'CUDA Graph · 一次图提交':'普通方式 · 逐项提交','teach-heading'),
  calls:(i?d.graphCPU:d.cpu).map((e,j)=>event(e,'launch-'+i+'-'+j,'teach-value teach-color-4')),
  kernels:(i?d.graphGPU:d.gpu).map((e,j)=>event(e,'kernel-'+i+'-'+j,'teach-value teach-color-'+j)),
  arrows:(i?[d.graphGPU[0]]:d.gpu).map(()=>path()),
  gaps:i?[]:d.gpu.slice(1).filter((e,j)=>e.start>d.gpu[j].end).map((e,j)=>({start:d.gpu[j].end,end:e.start,id:e.id,bg:rect('teach-mask'),label:txt('等','teach-small')})),
  cursor:path(),cpuLabel:txt('CPU 提交','teach-small'),gpuLabel:txt('GPU 执行','teach-small'),result:txt('','teach-small')}));
 const takeaway=txt('','teach-heading'),explain=txt('','teach-small'),explain2=txt('','teach-small'),legend=txt('','teach-small'),legend2=txt('','teach-small'),scaleNote=txt('','teach-small');
 return(p,m)=>{const W=m?360:720;
  // Keep the CPU/GPU row names in a fixed left gutter. At tablet heights the
  // camera enlarges SVG type to stay legible, so a caption placed under the
  // previous lane can cover the first event label; the timeline now starts
  // after the caption column instead of sharing its x range.
  const x=m?112:124,span=W-x-18,scale=span/d.duration;
  const round=p>=3.999?1:0,local=round?clamp(p-4):clamp(p-1),clock=local*d.duration,ready=phase(p,0),input=round?4:2,output=round?13:9;
  svg.dataset.graphClock=clock.toFixed(4);svg.dataset.graphRound=String(round+1);
  text(title,18,30,m?'A → B → C 依赖链':'同一条依赖链：A → B → C');
  text(meaning,18,54,m?'A 加 1 · B 乘 2 · C 加 3':'A：加 1；B：乘 2；C：加 3。每轮都重新算。');
  text(meaning2,18,74,m?'每轮都用新输入重算':'');show(meaning2,m?1:0);
  const prepW=m?150:300;
  prep.forEach((r,i)=>{box(r,18+i*(prepW+10),92,prepW,32);show(r,.25+.75*ready);text(prepText[i],26+i*(prepW+10),113,m?(i?'② 实例化':'① 捕获'):(i?'实例化：可重放的图':'捕获：记录操作与依赖'));});
  wire(prepLink,`M${18+prepW} 108 H${18+prepW+10}`,ready);
  text(prepNote,18,134,p<1?(m?'捕获不执行 GPU 运算':'首次准备：捕获不执行 GPU 运算'):'首次准备已完成 · 本轮复用');
  text(prepCost,18,156,m?'准备成本不计入时间轴':'一次性准备（教学假设，不标时长）');
  text(roundLabel,18,184,`第 ${round+1} 轮 · 输入 ${input}`);
  text(history,W-18,172,'上一轮 9');set(history,{'text-anchor':'end'});show(history,round?1:0);
  wire(axis,`M${x} 214 H${x+span}`);ticks.forEach((t,i)=>{const last=i===d.duration,prior=i===d.duration-1;text(t,last?x+span-2:x+i*scale-3,206,last?d.duration+' 秒':String(i));set(t,{'text-anchor':last||prior?'end':'start'});show(t,m&&((d.duration>8&&i%2&&!last)||i===d.duration-1)?0:1);});
  panels.forEach((r,i)=>{const y=226+i*188,finish=r.kernels.at(-1).end;box(r.bg,12,y,W-24,176);
   text(r.title,26,y+22);text(r.cpuLabel,26,y+45);text(r.gpuLabel,26,y+91);
   function paint(events,offset){events.forEach(e=>{const amount=clamp((clock-e.start)/(e.end-e.start)),bx=x+e.start*scale,bw=(e.end-e.start)*scale;
    box(e.outline,bx,y+offset,bw,24);show(e.outline,.5*ready);box(e.fill,bx,y+offset,bw*amount,24);e.fill.dataset.motionId=e.id;set(e.fill,{'data-start':e.start,'data-end':e.end,'data-progress':amount.toFixed(4)});
    text(e.label,bx+bw/2,y+offset+16,e.name==='Graph'?'图':e.name);set(e.label,{'text-anchor':'middle'});show(e.label,ready);
    set(e.label,{class:amount>.7?'hardware-event-label is-filled':'hardware-event-label'});
   });}
   paint(r.calls,56);paint(r.kernels,102);
   r.arrows.forEach((a,j)=>{const call=r.calls[j],kernel=r.kernels[j],start=call.end,end=kernel.start;
    wire(a,`M${x+start*scale} ${y+82} V${y+92} H${x+end*scale} V${y+102}`,clock>=start?1:0);});
   r.gaps.forEach(g=>{const amount=clamp((clock-g.start)/(g.end-g.start));box(g.bg,x+g.start*scale,y+102,(g.end-g.start)*scale*amount,24);text(g.label,x+(g.start+g.end)/2*scale,y+118,m?'等':'等 '+g.id);set(g.label,{'text-anchor':'middle'});show(g.label,amount>.65?1:0);});
   wire(r.cursor,`M${x+clock*scale} ${y+40} V${y+136}`,p>=1&&clock<finish?1:0);set(r.cursor,{'stroke-dasharray':'3 3','marker-end':'none'});show(r.cursor,p>=1&&clock<finish?.6:0);
   const done=clock>=finish-1e-9;
   text(r.result,26,y+158,done?`完成 · ${output} = (${input}+1)×2+3`:clock<=0?'等待本轮提交':i?'图已提交，GPU 继续算':'提交与计算可重叠');
  });
  text(takeaway,18,610,d.kernelDuration===1?(m?'少等提交，kernel 没变短':'少等提交，三个 kernel 没有变短'):(m?'CPU 少提交，总耗时不变':'CPU 少提交，但本例总耗时相同'));
  text(explain,18,636,d.kernelDuration===1?(m?'灰色空档 = GPU 等提交':'虚线空档：GPU 等下一次提交完成。'):(m?'长 kernel 覆盖提交时间':'较长计算覆盖了后续提交，GPU 原本就能接续执行。'));
  text(explain2,18,658,d.kernelDuration===1?(m?'三个 kernel 本身没变短':'三个 kernel 的时长没有变化。'):(m?'两种方式同刻结束':'两种方式在同一时刻结束。'));show(explain2,m?1:0);
  text(legend,18,682,m?'同色 = 同一 kernel':'同色块 = 同一个 kernel；块宽 = 计算用时。');
  text(legend2,18,704,m?'块宽 = 计算用时':'');show(legend2,m?1:0);
  text(scaleNote,18,726,m?'横轴：教学时间，非实测。':'横轴：同一教学时间刻度；不是硬件实测值。');
  if(m){set(takeaway,{'font-size':17});set(explain,{'font-size':13});}
 };
});}
V['f-hardware']=(svg,viewport,s)=>s.kind==='hardware-roofline'?resourceComparison(svg,viewport,s):s.kind==='hardware-graph'?graphSubmission(svg,viewport,s):matrixArithmetic(svg,viewport,s);
V['f-communication']=(svg,viewport,s)=>board(svg,viewport,'collective',()=>710,H=>{
 const {txt,rect,text,box,show,token,move}=H,d=s.data,gather=s.kind==='gather';
 const panels=[rect(),rect()],heads=[txt('Rank 0','teach-heading'),txt('Rank 1','teach-heading')];
 const input=d.input.map((v,i)=>token('input-'+i,'['+v+']',108,'teach-token teach-color-'+i));
 const copies=d.input.map((v,i)=>token('copy-'+i,'['+v+']',108,'teach-token teach-color-'+i));
 const operation=txt('逻辑运算示意 · 不是一台中央设备','teach-heading'),math=[txt('','teach-number'),txt('','teach-number')],result=token('merged','['+d.result+']',150,'teach-token teach-color-2');
 const dest=[0,1].map(i=>({label:txt('Rank '+i+' 收到结果副本','teach-small'),token:token('result-'+i,'['+d.result+']',150,'teach-token teach-color-2')}));
 const note=txt('原输入保持可见；两份结果属于不同参与方。','teach-small');
 return(p,m)=>{const W=m?360:720,x=i=>m?20+i*170:58+i*360;
  panels.forEach((e,i)=>{box(e,x(i)-4,22,m?162:290,138);text(heads[i],x(i)+10,52);move(input[i],x(i)+10,80);const t=phase(p,0);move(copies[i],lerp(x(i)+10,x(i)+10,t),lerp(80,228,t),t);});
  text(operation,18,290);if(m)operation.setAttribute('font-size','15');
  text(math[0],28,328,gather?'按顺序保留四个位置：':'位置 0：1 + 10'+(p>=3?' = 11':''));text(math[1],28,364,gather?'[1,2] 接着 [10,20]':'位置 1：2 + 20'+(p>=3?' = 22':''));math.forEach(e=>show(e,phase(p,1)));
  move(result,(W-150)/2,400,p>=3?1:0);
  dest.forEach((r,i)=>{text(r.label,x(i),610);show(r.label,phase(p,3));move(r.token,lerp((W-150)/2,x(i),phase(p,3)),lerp(400,540,phase(p,3)),phase(p,3));});
  text(note,18,668);if(m)note.setAttribute('font-size','12');
 };
});
V['f-parallel']=(svg,viewport,s)=>board(svg,viewport,'pipeline-grid',()=>860,H=>{
 const {txt,rect,path,wire,text,box,token,move}=H;
 const heading=txt('列 = 教学时隙；行 = PP stage','teach-heading');
 const heads=[1,2,3].map(i=>txt('时隙 '+i,'teach-small'));
 const stages=[0,1].map(i=>({label:txt('PP','teach-heading'),index:txt('Stage '+i,'teach-small'),layers:txt(i?'L3–L4':'L1–L2','teach-small')}));
 const cells=Array.from({length:6},()=>({bg:rect('teach-graph-slot'),fill:rect('teach-value'),label:txt('','teach-small')}));
 const transfers=Array.from({length:s.data.n},()=>path()),crossing=txt('','teach-small');
 const completed=txt('','teach-heading'),reason=txt('','teach-small');
 const jobs=Array.from({length:s.data.n},(_,i)=>token('MB'+(i+1),'MB'+(i+1),56,'teach-token teach-color-'+i));
 const paths=[txt('MB = microbatch（微批次）','teach-small'),txt('MB1 = {R1,R2}；MB2 = {R3,R4}','teach-small'),txt('同一微批次：Stage 0 → Stage 1 → 完成','teach-small'),txt('移动表示激活交接；填条表示阶段内计算。','teach-small')];
 const grid=document.createElementNS('http://www.w3.org/2000/svg','g');
 [...svg.children].filter(e=>e.tagName!=='defs').forEach(e=>grid.append(e));svg.append(grid);grid.setAttribute('transform','translate(0 240)');
 const whole=txt('完整模型：L1 → L2 → L3 → L4','teach-heading');
 const partition=[txt('Stage 0 · GPU 0','teach-small'),txt('Stage 1 · GPU 1','teach-small')];
 const layerActors=[1,2,3,4].map(i=>token('pp-layer-'+i,'L'+i,54,'teach-token teach-color-'+(i<3?0:1)));
 const explanation=txt('先按模型层切分，再让微批次依次经过各阶段。','teach-small');
 return(raw,m)=>{const p=Math.max(0,raw-2),u=phase(raw,0),cw=m?84:190,start=m?92:108;
  text(whole,18,28);partition.forEach((t,i)=>{text(t,m?18+i*180:80+i*350,87);t.setAttribute('opacity',u);});
  layerActors.forEach((a,i)=>move(a,lerp(18+i*(m?84:174),(m?18:80)+Math.floor(i/2)*(m?180:350)+(i%2)*66,u),lerp(45,105,u)));
  text(explanation,18,181,raw<1?'L1…L4 是模型层；顺序决定计算依赖。':raw<2?'MB1={R1,R2}；MB2={R3,R4} 是工作单元。':'下方每一行对应上方一个 PP stage。');
  grid.setAttribute('opacity',phase(raw,1));
  text(heading,18,30);heads.forEach((h,i)=>text(h,start+i*cw+4,68));stages.forEach((h,i)=>{const y=136+i*145;text(h.label,18,y);text(h.index,18,y+24);text(h.layers,18,y+46);});
  cells.forEach((c,i)=>{const row=Math.floor(i/3),col=i%3,x=start+col*cw,y=88+row*145,job=col-row,used=job>=0&&job<s.data.n,t=row?clamp((phase(p,col)-.3)/.7):phase(p,col);box(c.bg,x,y,cw-10,112);box(c.fill,x+2,y+84,(cw-14)*(used?t:0),24);text(c.label,x+6,y+24,used?'MB'+(job+1):row===1&&col===0?'等激活':row===0?'无新任务':'无输入');});
  transfers.forEach((line,i)=>wire(line,`M${start+i*cw+38} 202 L${start+(i+1)*cw+38} 230`));
  text(crossing,18,396,svg.dataset.placement==='cross-host'?'激活交接：跨进程 · GPU 间 · 跨主机':'激活交接：跨进程 · GPU 间 · 同机');if(m)crossing.setAttribute('font-size','13');
  jobs.forEach((j,i)=>{const progress=clamp(phase(p,i+1)/.3),x0=start+i*cw+8,x1=start+(i+1)*cw+8;move(j,lerp(x0,x1,progress),lerp(136,281,progress),p>=i?1:0);});
  const done=Math.min(s.data.n,Math.max(0,Math.floor(p)-1));const passed=done?Array.from({length:done},(_,i)=>'MB'+(i+1)).join('、'):'';text(completed,18,440,s.data.n===1?(done?'MB1 已依次完成 Stage 0 和 Stage 1':'尚无微批次完成两个 stage'):(done>=s.data.n?'MB1、MB2 均完成 Stage 0→1':done?`${passed} 已完成两段；MB${done+1} 尚未`:'尚无微批次完成两个 stage'));if(m)completed.setAttribute('font-size','16');text(reason,18,478,p<1?'Stage 1 等待 Stage 0 产生激活。':p<2?'激活抵达 Stage 1 后，它才能开始计算。':p<3?'完成的是微批次，不是 stage。':'时间格保留各 stage 的计算与空闲记录。');if(m)reason.setAttribute('font-size','13');paths.forEach((h,i)=>{text(h,18,500+i*36,i===1&&s.data.n===1?'本条件只送入 MB1 = {R1,R2}':undefined);if(m)h.setAttribute('font-size','13');});
 };
});
function spec(svg,viewport,s){const engine=s.kind==='engine-speculation';return board(svg,viewport,'verification',()=>engine?820:650,H=>{
 const {txt,rect,text,box,show,token,move}=H,d=s.data,inputs=[d.root,...d.draft];
 const title=txt('每列：输入位置 → 预测下一个 token','teach-heading'),focus=rect('teach-focus'),commit=rect('teach-read'),keep=rect('teach-read');
 const labels=['验证输入','目标预测','候选对照','正式输出','有效 KV'].map(x=>txt(x,'teach-heading'));
 const input=inputs.map((x,i)=>token('verify-input-'+i,x,48,'teach-token teach-color-'+i%5));
 const predictions=d.predictions.slice(0,5).map((x,i)=>token('prediction-'+i,x,48,'teach-token teach-color-2'));
 const draft=d.draft.map((x,i)=>token('candidate-'+i,x,48,'teach-token teach-color-'+(i+1)%5));
 const checks=Array.from({length:4},()=>txt('','teach-small'));
 const output=d.output.map((x,i)=>token('committed-output-'+i,x,48,'teach-token teach-color-'+(i<d.accepted?(i+1)%5:2)));
 const kv=d.validKV.map((x,i)=>token('valid-kv-'+i,x,48,'teach-token teach-color-'+i%5));
 const why=txt('','teach-small'),offset=txt('','teach-small'),state=txt('','teach-heading');
 return(p,m)=>{const W=m?360:720,start=m?22:70,gap=m?64:128,at=i=>start+i*gap;
  text(title,18,28);labels.forEach((l,i)=>{text(l,18,[64,190,320,506,660][i]);show(l,i===4&&!engine?0:1);});
  input.forEach((t,i)=>move(t,at(i),86,i===0?1:phase(p,0)));predictions.forEach((t,i)=>move(t,at(i),218,phase(p,1)));
  draft.forEach((t,i)=>{move(t,at(i),346,phase(p,0));t.g.classList.toggle('teach-empty',p>=3&&i>=d.accepted);}); // Prediction at input i compares to candidate i, not input i+1.
  const checked=Math.floor(phase(p,2)*4+1e-8);box(focus,at(Math.min(d.accepted,Math.min(3,phase(p,2)*4)))-4,210,56,178);show(focus,p>=2&&p<3?1:0);
  checks.forEach((e,i)=>{text(e,at(i),406,i>=checked?'待判断':i<d.accepted?'通过':i===d.accepted?'分歧':'不再接受');});
  text(why,18,446,d.accepted===4?'全部通过：最后一列预测还能补出一个 token。':'分歧后的输入前缀不再有效，后面不能跳着接受。');show(why,p>=3?1:0);if(m)why.setAttribute('font-size','12');
  output.forEach((t,i)=>{const u=phase(p,3);move(t,at(i),lerp(i<d.accepted?346:218,534,u),p>3?1:0);});
  kv.forEach((t,i)=>{move(t,at(i),688,engine&&p>=5?1:0);});
  box(commit,at(0),580,(d.output.length*gap-16)*phase(p,3),8);box(keep,at(0),734,(d.validKV.length*gap-16)*phase(p,4),8);
  show(keep,engine?1:0);text(offset,18,610,engine?'输出末尾的补出 token 尚未输入；KV 从根位置开始。':'正式输出 = 接受的连续前缀 + 补出 token。');if(m)offset.setAttribute('font-size','12');show(offset,p>=4?1:0);
  show(state,engine?1:0);text(state,18,780,p>=5?'有效历史 ≠ 刚刚发出的输出序列':'验证临时状态不等于已提交历史');
 };
});}
function foundationSpeculation(svg,viewport,s){return board(svg,viewport,'speculative-prefix',m=>m?1040:800,H=>{
 const {txt,rect,text,box,show,token,move,set}=H,d=s.data;
 // The proposal row and the verification table deliberately share five
 // columns. The table is the same input sequence seen through one target call,
 // so predictions can be read against the candidate that produced them.
 const bonusLabel=d.accepted===4?'补出':'替代';
 const context=txt('','teach-small'),heads=['① 草稿候选 · 尚未确认','② 目标模型 · 1 次验证','③ 提交正式输出'].map(t=>txt(t,'teach-heading'));
 const root=txt('','teach-small'),rootToken=token('root-token','t0',48,'teach-token teach-color-4');
 const draft=d.draft.map((v,i)=>token('candidate-'+i,v,48,'teach-token teach-color-'+i));
 const input=[d.root,...d.draft].map((v,i)=>token('verify-input-'+i,v,48,'teach-token teach-color-'+(i===0?4:i-1)));
 const predicted=d.predictions.map((v,i)=>token('prediction-'+i,v,48,'teach-token teach-color-4'));
 const checks=Array.from({length:4},()=>txt('','teach-small'));
 const output=d.output.map((v,i)=>token('committed-output-'+i,v,48,'teach-token teach-color-'+(i<d.accepted?i:4)));
 const outputTags=Array.from({length:4},()=>txt('','teach-small'));
 const validTitle=txt('有效历史：根 + 接受前缀','teach-small'),validHistory=d.validKV.map((v,i)=>token('valid-history-'+i,v,48,'teach-token teach-color-'+(i===0?4:i)));
 const accepted=rect('teach-read'),rejected=rect('teach-focus'),rejectedDraft=rect('teach-focus'),focus=rect('teach-map'),compareDraft=rect('teach-map');
 [focus,rejected,rejectedDraft,compareDraft].forEach(e=>e.style.fill='none');
 const columnLabels=['根','草1','草2','草3','草4'].map(v=>txt(v,'teach-small'));
 const rowInput=txt('','teach-small'),rowPrediction=txt('','teach-small'),rowJudge=txt('','teach-small');
 const proposalNote=txt('候选未提交 · 颜色=身份','teach-small'),predictionRule=txt('1 次调用 × 5 个位置','teach-small');
 const reason=txt('','teach-heading'),legend=txt('候选按身份着色；预测同色','teach-small'),legend2=txt('','teach-small'),bonus=txt('','teach-small');
 // The call ledger is a process account, not a post-hoc table. Every call
 // exists from p0 with an explicit pending state; a value appears only after
 // the action that produced it. One verification over five positions remains
 // one call, and the ordinary path is an explanatory comparison, not a second
 // simultaneous execution.
 const callPanel=rect('teach-panel spec-ledger'),callTitle=txt('','teach-heading'),callNote=txt('','teach-small');
 const callDraftLabel=txt('','teach-heading'),callVerifyLabel=txt('','teach-heading'),callOrdinaryLabel=txt('','teach-heading');
 const draftBoxes=Array.from({length:4},()=>rect('spec-call-card'));
 const draftTints=Array.from({length:4},()=>rect('spec-identity'));
 const draftLabels=Array.from({length:4},()=>txt('','teach-small spec-card-label'));
 const draftValues=Array.from({length:4},()=>txt('','teach-text spec-call-value'));
 const draftStates=Array.from({length:4},()=>txt('','teach-small spec-call-state'));
 const verifyBox=rect('spec-verify'),verifyTint=rect('spec-identity teach-color-4'),verifyState=txt('','teach-small spec-call-state'),verifyCoverage=txt('','teach-small');
 const verifyPips=Array.from({length:5},()=>rect('spec-pip'));
 const verifyPipLabels=Array.from({length:5},()=>txt('','teach-small spec-pip-label'));
 const verifyProgress=rect('spec-progress'),verifyNote=txt('同一次调用覆盖 5 个位置。','teach-small');
 const ordinaryBoxes=Array.from({length:5},()=>rect('spec-call-card'));
 const ordinaryTints=Array.from({length:5},()=>rect('spec-identity'));
 const ordinaryLabels=Array.from({length:5},()=>txt('','teach-small spec-card-label'));
 const ordinaryValues=Array.from({length:5},()=>txt('','teach-text spec-call-value'));
 const ordinaryStates=Array.from({length:5},()=>txt('','teach-small spec-call-state'));
 const ordinaryPlaceholder=rect('spec-call-card is-pending'),ordinaryPlaceholderTitle=txt('输出确认后展开','teach-small spec-card-label'),ordinaryPlaceholderNote=txt('逐 token 对照，不另行执行','teach-small spec-call-state');
 const ordinaryNote=txt('同一输出；不另行执行。','teach-small'),costTitle=txt('成本边界','teach-heading'),cost1=txt('','teach-small'),cost2=txt('','teach-small');
 const callNodes=[callPanel,callTitle,callNote,callDraftLabel,callVerifyLabel,callOrdinaryLabel,verifyBox,verifyTint,verifyState,verifyCoverage,verifyProgress,verifyNote,ordinaryPlaceholder,ordinaryPlaceholderTitle,ordinaryPlaceholderNote,ordinaryNote,costTitle,cost1,cost2,...draftBoxes,...draftTints,...draftLabels,...draftValues,...draftStates,...verifyPips,...verifyPipLabels,...ordinaryBoxes,...ordinaryTints,...ordinaryLabels,...ordinaryValues,...ordinaryStates];
 const callLayer=document.createElementNS('http://www.w3.org/2000/svg','g');
 callNodes.forEach(n=>callLayer.append(n));svg.append(callLayer);
 const cardState=(e,state)=>{for(const s of ['pending','running','done'])e.classList.toggle('is-'+s,state===s);};
 return(p,m)=>{const W=m?360:860,colGap=m?60:112,colX=i=>m?18+i*colGap:86+i*colGap;
  const draftY=m?78:84,verifyHeadY=m?134:146,inputY=m?178:194,lineY=m?216:232,predY=m?242:242;
  const checkY=m?310:292,reasonY=m?334:316,commitHeadY=m?360:338,outputY=m?370:348,validY=m?448:430;
  text(context,18,24,m?'根 t0 已提交；4 个候选待验证。':'根 t0 已提交；4 个草稿候选等待目标模型一次验证。');
  text(heads[0],18,50);text(heads[1],18,verifyHeadY);text(heads[2],18,commitHeadY);
  text(root,20,m?0:106,'根');show(root,m?0:1);
  move(rootToken,colX(0),draftY,1);
  const draftProgress=i=>phase(p,i/4,(i+1)/4);
  draft.forEach((t,i)=>{move(t,colX(i+1),draftY,draftProgress(i));t.g.classList.toggle('teach-empty',p>=3&&i>=d.accepted);});
  text(proposalNote,610,106);show(proposalNote,m?0:1);
  text(rowInput,20,m?168:214,m?'验证输入：根 + 4 候选':'验证输入');
  text(rowPrediction,20,m?226:262,m?'目标预测（1 次调用）':'目标预测');
  text(rowJudge,20,m?296:292,m?'判断':'判断');
  columnLabels.forEach((e,i)=>{text(e,colX(i)+24,m?0:184,['根','草1','草2','草3','草4'][i]);set(e,{'text-anchor':'middle'});show(e,m?0:1);});
  input.forEach((t,i)=>{move(t,colX(i),inputY,i===0?1:phase(p,.92,1.15));if(i)t.g.classList.toggle('teach-empty',p>=3&&i>d.accepted);});
  const checked=Math.min(d.accepted===4?4:d.accepted+1,Math.floor(phase(p,2)*4+1e-8));
  checks.forEach((e,i)=>{text(e,colX(i)+24,checkY,i>=checked?(checked>d.accepted?'舍弃':''):i<d.accepted?'接受':i===d.accepted?'分歧':'舍弃');set(e,{'text-anchor':'middle'});});
  predicted.forEach((t,i)=>{const visible=!(i===4&&d.accepted<4);move(t,colX(i),predY,visible?phase(p,1):0);});
  const n=Math.min(d.accepted,checked);box(accepted,colX(0)-2,lineY,n?(n*colGap+52):0,4);show(accepted,p>=2&&n?1:0);
  const comparing=Math.min(d.accepted,Math.floor(phase(p,2)*4)),validPrefix=d.draft.slice(0,comparing),firstDiff=d.accepted<4?d.accepted:-1;
  // The prediction is in column i, while the candidate it checks sits one
  // column to the right. Pair the outlines and name both values so the
  // judgment is not read as accepting the root input in column 0.
  const diverged=firstDiff>=0&&(checked>firstDiff||p>=3);
  box(rejected,colX(Math.min(d.accepted,3))-4,predY-4,56,42);show(rejected,p>=3&&d.accepted<4?1:0);
  box(rejectedDraft,colX(Math.min(d.accepted,3)+1)-4,draftY-4,56,42);show(rejectedDraft,p>=3&&d.accepted<4?1:0);
  box(focus,colX(comparing)-4,inputY-2,56,predY+38-(inputY-2));show(focus,p>=2&&p<3?1:0);
  box(compareDraft,colX(comparing+1)-4,draftY-4,56,42);show(compareDraft,p>=2&&p<3?1:0);
  let why='';
  if(p>=1){
   if(p<2)why='候选逐个生成；随后由目标模型一次验证。';
   else if(diverged)why=m?`首次分歧：预测${d.predictions[d.accepted]}↔草${d.accepted+1}${d.draft[d.accepted]}；后续不再接受。`:`首次分歧：预测下一 token「${d.predictions[d.accepted]}」 ↔ 草${d.accepted+1}「${d.draft[d.accepted]}」；后面不能跳着接受。`;
   else if(d.accepted===4&&p>=3)why=m?`全部接受；第 5 列补出「${d.predictions[4]}」。`:`全部接受；第 5 列预测「${d.predictions[4]}」作为补出。`;
   else{const k=Math.min(checked,3);why=m?`第${k+1}次：预测「${d.predictions[k]}」↔草${k+1}「${d.draft[k]}」`:`第 ${k+1} 次比较：预测下一 token「${d.predictions[k]}」 ↔ 草${k+1}「${d.draft[k]}」`;}
  }
  text(reason,18,reasonY,why);show(reason,p>=1?1:0);
  text(predictionRule,610,214);show(predictionRule,m?0:1);
  // Keep the submitted tokens in the ③ row: a tiny upward lane plus a short
  // horizontal slide, clear of the prediction, judgment and reason rows. The
  // 草N / 替代 labels on the same baseline name where each token came from.
  const laneGap=m?10:16,laneY=outputY-3,tagY=m?420:402;
  output.forEach((t,i)=>{const t0=3.02+i*.06,t1=3.96,u=clamp((p-t0)/(t1-t0)),drop=phase(u,0,.25),slide=clamp((u-.2)/.8);
  move(t,lerp(colX(i)+laneGap,colX(i),slide),lerp(laneY,outputY,drop),phase(p,3,3.14));});
 outputTags.forEach((e,i)=>{const accepted=i<d.accepted;text(e,colX(i)+24,tagY,'草'+(i+1));set(e,{'text-anchor':'middle'});show(e,accepted?phase(p,3.08+i*.06,3.24+i*.06):0);});
  text(bonus,colX(d.accepted)+24,tagY,bonusLabel);set(bonus,{'text-anchor':'middle'});show(bonus,p>3?1:0);
  text(legend,610,m?0:356);show(legend,m?0:1);
  text(validTitle,20,m?438:420);show(validTitle,p>=4?1:0);
  validHistory.forEach((t,i)=>move(t,colX(i),validY,p>=4?1:0));
  text(legend2,18,m?500:484,p>=4?(m?'根 + 接受前缀；补出/替代未回灌，本轮暂无自身 KV。':'有效历史 = 根 + 接受前缀；补出/替代 token 本轮尚未回灌，尚无自己的有效 KV。'):'');show(legend2,p>=4?1:0);
  const ledgerY=m?510:498,ledgerH=m?520:288,ledgerX=12,ledgerW=W-24;
  const comparisonStarted=p>=3,outputSubmitted=p>=4,ordinaryCount=d.targetCallsSequential;
  const ledgerNote=m?'灰待处理 · 琥珀执行 · 绿完成；普通对照不另行执行。':'灰=待处理 · 琥珀=进行中 · 绿=已完成；普通对照解释同一输出，不另行执行。';
  const pendingLedgerNote=m?'输出提交中；普通对照待 token 到达后计 1 次。':'正式输出提交中；普通对照暂为预计，token 到达后才计 1 次，不另行执行。';
  box(callPanel,ledgerX,ledgerY,ledgerW,ledgerH);
  text(callTitle,26,m?534:522,m?'调用账目 · 已发生动作':'调用账目 · 只记已经发生的动作');
  text(callNote,26,m?558:543,comparisonStarted&&!outputSubmitted?pendingLedgerNote:ledgerNote);
  const draftTitleY=m?582:570,draftCardY=m?592:580,draftH=m?70:80,draftW=m?70:54,draftGap=6;
  text(callDraftLabel,26,draftTitleY,'草稿模型 · 4 次');
  let completedDrafts=0;
  draftBoxes.forEach((r,i)=>{const u=draftProgress(i),done=u>=1-1e-6,running=u>0&&!done;
   if(done)completedDrafts++;
   const x=26+i*(draftW+draftGap);
   box(r,x,draftCardY,draftW,draftH);cardState(r,done?'done':running?'running':'pending');
   box(draftTints[i],x+4,draftCardY+8,3,draftH-16);set(draftTints[i],{rx:2});draftTints[i].setAttribute('class','spec-identity teach-color-'+i);
   text(draftLabels[i],x+draftW/2,draftCardY+(m?16:18),'草'+(i+1));
   text(draftValues[i],x+draftW/2,draftCardY+(m?36:40),done?d.draft[i]:'');
   text(draftStates[i],x+draftW/2,draftCardY+(m?56:60),done?'已完成':running?'生成中':'待生成');cardState(draftStates[i],done?'done':running?'running':'pending');
   [draftLabels[i],draftValues[i],draftStates[i]].forEach(e=>set(e,{'text-anchor':'middle'}));
   show(r,1);show(draftTints[i],1);show(draftLabels[i],1);show(draftValues[i],done?1:0);show(draftStates[i],1);
  });
  const verifyTitleY=m?684:570,verifyY=m?694:580,verifyW=m?308:244,verifyH=m?86:80,verifyX=m?26:316;
  text(callVerifyLabel,verifyX,verifyTitleY,'目标模型 · 1 次验证');
  box(verifyBox,verifyX,verifyY,verifyW,verifyH);
  box(verifyTint,verifyX+4,verifyY+8,3,verifyH-16);set(verifyTint,{rx:2});
  const verifyStarted=p>=1-1e-6,verifyDone=p>=2-1e-6,verifyRunning=verifyStarted&&!verifyDone,verifyU=verifyDone?1:phase(p,1,2);
  cardState(verifyBox,verifyDone?'done':verifyRunning?'running':'pending');
  text(verifyCoverage,verifyX+10,verifyY+18,'覆盖 5 个位置');
  text(verifyState,verifyX+verifyW-10,verifyY+18,verifyDone?'已完成':verifyRunning?'执行中':'待开始');set(verifyState,{'text-anchor':'end'});cardState(verifyState,verifyDone?'done':verifyRunning?'running':'pending');
  const pipW=m?52:40,pipGap=m?6:5,pipY=verifyY+26,pipH=m?24:20,pipX=verifyX+10;
  ['根','草1','草2','草3','草4'].forEach((name,i)=>{const x=pipX+i*(pipW+pipGap);
   box(verifyPips[i],x,pipY,pipW,pipH);
   cardState(verifyPips[i],verifyDone?'done':verifyRunning?'running':'pending');
   text(verifyPipLabels[i],x+pipW/2,pipY+(m?16:15),name);set(verifyPipLabels[i],{'text-anchor':'middle'});
  });
  box(verifyProgress,pipX,verifyY+(m?60:54),(verifyW-20)*verifyU,5);show(verifyProgress,verifyU>0?1:0);
  text(verifyNote,verifyX,verifyY+(m?102:101));show(verifyNote,1);
  const ordinaryTitleY=m?818:570,ordinaryY=m?828:580,ordinaryH=m?70:80,ordinaryW=m?56:44,ordinaryGap=m?6:5,ordinaryX=m?26:596;
  text(callOrdinaryLabel,ordinaryX,ordinaryTitleY,outputSubmitted?`普通对照 · ${ordinaryCount} 次`:comparisonStarted?`普通对照 · 预计 ${ordinaryCount} 次`:'普通对照 · 待定');
  if(!comparisonStarted){
   box(ordinaryPlaceholder,ordinaryX,ordinaryY,m?308:244,ordinaryH);
   text(ordinaryPlaceholderTitle,ordinaryX+12,ordinaryY+24,'输出确认后展开');
   text(ordinaryPlaceholderNote,ordinaryX+12,ordinaryY+46,'逐 token 对照，不另行执行');
   show(ordinaryPlaceholder,1);show(ordinaryPlaceholderTitle,1);show(ordinaryPlaceholderNote,1);
   ordinaryBoxes.forEach((r,i)=>{show(r,0);show(ordinaryTints[i],0);show(ordinaryLabels[i],0);show(ordinaryValues[i],0);show(ordinaryStates[i],0);});
  }else{
   show(ordinaryPlaceholder,0);show(ordinaryPlaceholderTitle,0);show(ordinaryPlaceholderNote,0);
   ordinaryBoxes.forEach((r,i)=>{
    const on=i<ordinaryCount,u=phase(p,3+i*.12,4),revealed=u>=1,submitting=u>0&&!revealed;
    show(r,on?1:0);show(ordinaryTints[i],on?1:0);show(ordinaryLabels[i],on?1:0);show(ordinaryValues[i],on&&revealed?1:0);show(ordinaryStates[i],on?1:0);
    if(!on)return;
    const x=ordinaryX+i*(ordinaryW+ordinaryGap);
    box(r,x,ordinaryY,ordinaryW,ordinaryH);cardState(r,revealed?'reference':'pending');
    box(ordinaryTints[i],x+4,ordinaryY+8,3,ordinaryH-16);set(ordinaryTints[i],{rx:2});ordinaryTints[i].setAttribute('class','spec-identity teach-color-'+(i<d.accepted?i:4));
    text(ordinaryLabels[i],x+ordinaryW/2,ordinaryY+(m?16:18),'同'+(i+1));
    text(ordinaryValues[i],x+ordinaryW/2,ordinaryY+(m?36:40),revealed?d.output[i]:'');
    text(ordinaryStates[i],x+ordinaryW/2,ordinaryY+(m?56:60),revealed?'计1次':submitting?'提交中':'待对照');
    [ordinaryLabels[i],ordinaryValues[i],ordinaryStates[i]].forEach(e=>set(e,{'text-anchor':'middle'}));
   });
  }
  text(ordinaryNote,ordinaryX,m?920:681);
  text(costTitle,26,m?960:719,'成本边界');
  if(p<1-1e-6){
   text(cost1,26,m?982:739,`草稿 ${completedDrafts}/4 完成，验证未开始。`);
   text(cost2,26,m?1004:759,'调用数尚不可比较。');
  }else if(p<2-1e-6){
   text(cost1,26,m?982:739,'草稿 4/4 完成，验证执行中。');
   text(cost2,26,m?1004:759,m?'覆盖 5 个位置，不是 5 次调用。':'1 次验证覆盖 5 个位置，不是 5 次调用。');
  }else if(!comparisonStarted){
   text(cost1,26,m?982:739,'验证完成，但输出尚未提交。');
   text(cost2,26,m?1004:759,m?'普通对照未展开，调用数不可比。':'普通对照尚未展开，调用数不可比较。');
  }else if(!outputSubmitted){
   text(cost1,26,m?982:739,'正式输出提交中；有效历史尚未提交。');
   text(cost2,26,m?1004:759,m?`对照预计 ${ordinaryCount} 次；token 到达后计 1 次，不另行执行。`:`普通对照预计 ${ordinaryCount} 次；待 token 到达后才计 1 次，不另行执行。`);
  }else if(d.accepted===0){
   text(cost1,26,m?982:739,m?'接受0：对照1、验证1；目标未减少。':'接受 0：普通对照 1 次，本路径验证 1 次；目标调用未减少。');
   text(cost2,26,m?1004:759,m?'对照不另行执行；另付4次草稿，成本不等，非固定加速。':'普通对照不另行执行；另付 4 次草稿，成本不等，也不是固定加速。');
  }else{
   text(cost1,26,m?982:739,m?`接受${d.accepted}：对照${ordinaryCount}、验证1、草稿4。`:`接受 ${d.accepted}：普通对照 ${ordinaryCount} 次，本路径验证 1 次；另付 4 次草稿。`);
   text(cost2,26,m?1004:759,m?'对照不另行执行；验证覆盖5位置，成本不等。':'普通对照不另行执行；1 次验证覆盖 5 个位置，成本不等，也不是固定加速。');
  }
  show(callLayer,1);
  svg.dataset.comparedPrefix=validPrefix.join('');
 };
},m=>m?360:860);}
V['f-speculation']=foundationSpeculation;V['advanced-generation']=spec;
V.communication=(svg,viewport,s)=>board(svg,viewport,'page-copy',()=>770,H=>{
 const {txt,rect,text,box,show,token,move,set}=H,d=s.data;
 const heads=[txt('源端 · 原页保持不动','teach-heading'),txt('目标端 · 地址保持不动','teach-heading')],pairs=txt('映射说明源与目标的对应；写入状态单独标注。','teach-small');
 const rows=d.src.map((page,i)=>({source:rect(),dest:rect('teach-memory-slot'),fill:rect('teach-memory-fill'),sl:txt('源页 '+page,'teach-small'),dl:txt('目标页 '+d.dst[i],'teach-small'),original:token('source-'+page,d.payload[i],48,'teach-token teach-color-'+i),copy:token('payload-'+i,d.payload[i],48,'teach-token teach-color-'+i),map:txt(page+' → '+d.dst[i],'teach-number'),status:txt('','teach-small')}));
 const summary=txt('','teach-heading'),blocks=d.blocks.map(()=>txt('','teach-small')),boundary=txt('写完本图三个目标页 ≠ 整条 PD 请求已可运行。','teach-small');
 return(p,m)=>{const W=m?360:720,sx=24,dx=m?230:520;
  text(heads[0],18,30);text(heads[1],dx-4,30,m?'目标端 · 固定地址':'目标端 · 地址保持不动');if(m)heads.forEach(e=>e.setAttribute('font-size','14'));text(pairs,18,64);
  rows.forEach((r,i)=>{const y=98+i*150,dy=98+[...d.dst].sort((a,b)=>a-b).indexOf(d.dst[i])*150,u=phase(p,i+1);box(r.source,sx,y,100,100);box(r.dest,dx,dy,100,100);box(r.fill,dx+2,dy+76,96*u,20);text(r.sl,sx+8,y+22);text(r.dl,dx+8,dy+22);move(r.original,sx+26,y+40);
   text(r.map,sx,y+124);show(r.map,phase(p,0));const mid=W/2-24,px=u<.3?lerp(sx+26,mid,u/.3):u<.7?mid:lerp(mid,dx+26,(u-.7)/.3),py=u<.3?y+40:u<.7?lerp(y+40,dy+40,(u-.3)/.4):dy+40;move(r.copy,px,py,u>0?1:0);text(r.status,dx+5,dy+124,u===1?'已写入':u>0?'写入途中':'尚未写入');r.dest.dataset.written=String(u===1);});
  text(summary,18,594,p>=5?`${d.blocks.length} 个连续块 · 总载荷 48 字节`:'哪些相邻映射可以合并？');blocks.forEach((e,i)=>{const b=d.blocks[i];text(e,18,630+i*26,`源 ${b.src} → 目标 ${b.dst}：${b.pages} 页 × ${d.bytesPerPage} 字节`);show(e,p>=5?1:0);});text(boundary,18,748);if(m)boundary.setAttribute('font-size','12');
 };
});
V['request-runtime']=(svg,viewport,s)=>board(svg,viewport,'ownership',()=>750,H=>{
 const {txt,rect,path,text,box,show,wire,token,move,set}=H,d=s.data;
 const request=rect(),cache=rect(),reqTitle=txt('请求 R1 / 请求行','teach-heading'),cacheTitle=txt('前缀缓存管理','teach-heading'),reqState=txt('','teach-small'),cacheState=txt('','teach-small');
 const memoryTitle=txt('固定的 KV 位置 · 不随归属变化搬动','teach-heading');
 const cells=Array.from({length:6},(_,i)=>({t:token('physical-kv-'+i,i<4?'p'+(i+1):'y'+(i-3),44,'teach-token teach-color-'+i%5),label:txt('位置 '+i,'teach-small'),owner:path()}));
 const cached=path(),requestLink=path(),bus=path(),last=token('last-output','y3',60,'teach-token teach-color-3'),lastLabel=txt('最后输出 · 尚未作为模型输入','teach-small'),note=txt('','teach-heading'),detail=txt('','teach-small');
 return(p,m)=>{const W=m?360:720,rx=18,cx=m?190:400;
  box(request,rx,24,m?152:280,126);box(cache,cx,24,m?152:300,126);text(reqTitle,rx+10,52);text(cacheTitle,cx+10,52);if(m){reqTitle.setAttribute('font-size','14');cacheTitle.setAttribute('font-size','14');}
  text(reqState,rx+10,88,p>=3?'请求行已归还':p>=1?'已记录结束原因':'仍持有请求行');text(cacheState,cx+10,88,d.cache?(p>=2?'保留 KV，仍占内存':'尚未接管'):'本例关闭缓存');
  text(memoryTitle,18,366);if(m)memoryTitle.setAttribute('font-size','15');
  const step=m?53:110;cells.forEach((c,i)=>{const x=24+i*step;move(c.t,x,280);text(c.label,x,338);c.t.g.classList.toggle('teach-empty',p>=3&&!d.cache);wire(c.owner,`M${x+22} 242 V280`,d.cache?1:1-phase(p,2));});
  wire(requestLink,`M${rx+76} 150 V190 H${W/2} V242`,1-phase(p,2));wire(cached,`M${cx+76} 150 V190 H${W/2} V242`,d.cache?phase(p,1):0);
  set(bus,{d:`M46 242 H${46+5*step}`});show(bus,d.cache?1:1-phase(p,2));
  text(lastLabel,18,402);move(last,28,424);
  text(note,18,548,p>=3?(d.cache?'请求行空闲；KV 仍由缓存保留':'请求行与 KV 位置都可重新分配'):'结束原因不会自动归还资源');if(m)note.setAttribute('font-size','16');
  text(detail,18,590,p>=3?(d.cache?'实框：六个 KV 位置仍被缓存占用。':'虚框：可再分配，不表示字节已被清零。'):'线表示持有关系；位置 0…5 始终不动。');
 };
});

function servingRoute(svg,viewport,s){const inner=board(svg,viewport,'serving-route',m=>m?540:440,H=>{
 const {txt,rect,path,text,box,token,move,show,wire}=H,d=s.data,variant=d.variant||0;
 const client=rect(),panels=[rect(),rect()],titles=[txt('实例 A · 完整模型','teach-heading'),txt('实例 B · 完整模型','teach-heading')];
 const health=[txt('health ✓','teach-small'),txt('health ✓','teach-small')],queue=[txt('queue: 0','teach-small'),txt('queue: 0','teach-small')];
 const panelState=[txt('','teach-small'),txt('','teach-small')],routeLabel=txt('','teach-small'),idleLabel=txt('A 未收到 R1','teach-small'),summary=txt('health ✓ 只说明在线；本例每实例 1 个执行槽','teach-small');
 const r1=token('route-r1','R1',56,'teach-token teach-color-0');
 const arrow=path(),back=path(),note=txt('','teach-small'),note2=txt('','teach-small'),state=txt('','teach-heading'),clientState=txt('','teach-small'),clientTitle=txt('客户端','teach-heading');
 const track=rect('teach-track'),bar=rect('teach-compute');
 return(p,m)=>{const W=m?360:720,clientW=m?W-40:196,x=[m?20:238,m?20:470],y=[m?150:70,m?300:70],pw=m?W-40:214,ph=m?126:180;
  box(client,20,58,clientW,84);text(clientTitle,32,84);text(clientState,32,112,p>=5?'已看到结果':p>=1?'等待 R1':'准备发送 R1');
  // Queue count follows the same teaching state as the status line. A request
  // waits from acceptance (or, for routing, from selection) until it gets the
  // execution slot; then it is no longer a waiting item.
  const route=d.route||{},waiting=variant?(p>=2&&p<3):(p>=1&&p<3);
  for(let i=0;i<(variant?2:1);i++){
   box(panels[i],x[i],y[i],pw,ph);text(titles[i],x[i]+10,y[i]+24);text(health[i],x[i]+10,y[i]+52);text(queue[i],x[i]+10,y[i]+76);
   let queueText='queue: 0',stateText='',queueCount=0;
   if(variant&&i===0){stateText='未接到 R1';}
   else if(variant&&i===1&&p<2){stateText=p<1?'尚未接收':'等待服务层选择';}
   else if(p<1){stateText='尚未接收';}
   else if(waiting&&(!variant||p>=2)){queueText='queue: 1 · R1 等待';queueCount=1;stateText=variant?'已选中 · 等待执行槽':p<2?'已接收 · 等待执行':'排队等待执行槽';}
   else if(p<3){stateText='已接收 · 等待执行';queueCount=1;queueText='queue: 1 · R1 等待';}
   else if(p<4){stateText='R1 执行中 · 占用执行槽';}
   else if(p<5){stateText='R1 已完成 · 结果回程';}
   else{stateText='R1 已完成 · 结果已送达';}
   text(queue[i],x[i]+10,y[i]+76,queueText);text(panelState[i],x[i]+10,y[i]+108,stateText);
   queue[i].dataset.queueCount=String(queueCount);panelState[i].dataset.state=stateText;
  }
  text(summary,20,m?440:266);
  if(!variant){[titles[1],health[1],queue[1],panelState[1]].forEach(e=>show(e,0));}
  const ex=variant?x[1]+pw/2:x[0]+pw/2,ey=variant?y[1]+ph/2:y[0]+ph/2;
  // R1 sits at the client edge after admission, then enters the selected
  // instance. In the two-instance trace it must wait for routing first.
  const targetX=variant?x[1]+30:x[0]+30,targetY=variant?y[1]+64:y[0]+64;
  const travel=phase(p,variant?2:1,variant?2.6:1.8),startX=Math.min(clientW+20,W-62);
  move(r1,lerp(startX,targetX,travel),lerp(100,targetY,travel),p>=1&&p<4?1:0);
  const forward=variant?(p<2?(m?`M${startX} 100 H${W-12} V44`:`M${startX} 100 V36 H300`):(m?`M${startX} 100 H${W-12} V${y[1]-6} H${ex}`:`M${startX} 100 V36 H${ex} V${y[1]-6}`)):(m?`M${clientW/2+20} 142 V${y[0]-6}`:`M${startX} 100 H${ex}`);wire(arrow,forward,p>=1?1:0);
  text(routeLabel,variant?(p<2?(m?150:300):x[1]+10):238,p>=1?(variant?40:56):56,variant?(p<2?'等待服务层选择':'只路由到 B'):'进入实例');show(routeLabel,p>=1);
  show(idleLabel,0);
  const active=variant?1:0,slotY=y[active]+88;
  box(track,x[active]+10,slotY,pw-20,7);box(bar,x[active]+10,slotY,pw-20,7);show(bar,p>=3?1:0);
  const backward=m?(variant?`M${ex} ${y[1]-28} H${W-12} V100 H${clientW+20}`:`M${ex} ${y[0]-28} V142 H${clientW/2+20}`):(variant?`M${ex} ${ey-28} V28 H${clientW+20} V100`:`M${ex} ${ey-28} V100 H${clientW+20}`);wire(back,backward,p>=4?1:0);
  const notes=p>=5?(variant?['结果已从 B 返回客户端；','A 的健康状态不等于 R1 被 A 执行。']:['结果已返回客户端；','本图不虚构网络时延。']):p>=4?(variant?['结果正在从 B 回程；','A 未执行 R1；客户端仍在等待。']:['结果正在回程；','客户端仍在等待，执行完成不等于已看见。']):p>=3?(variant?['R1 在 B 内执行；','执行完成前，结果尚未回程。']:['R1 正在目标实例内执行；','执行完成前，结果尚未回程。']):(variant?(p>=2?['B 已选中 R1；','R1 等待 B 的执行槽。']:p>=1?['R1 仍要等服务层选择；','等待目标实例接纳。']:['R1 尚未发送；','健康检查通过不等于 R1 被接纳。']):(p>=1?['健康通过不等于接纳；','请求仍要等待目标实例的执行槽。']:['R1 尚未发送；','健康检查只说明实例在线。']));
  if(m){
   text(note,20,464,notes[0]);text(note2,20,482,notes[1]);show(note2,1);
  }else{
   text(note,20,380,notes[0]+notes[1]);text(note2,20,400,'');show(note2,0);
  }
  text(state,20,m?510:340,p>=5?'完成：客户端可见结果':p>=4?'完成：结果回程中':p>=3?'执行：R1 执行中':p>=1?'接收 / 等待':'准备发送');if(m){state.setAttribute('font-size','16');note.setAttribute('font-size','11');note2.setAttribute('font-size','11');}
  svg.dataset.routeState=p<1?'arriving':p<2?(variant?'routing':'accepted'):p<3?'waiting':p<4?'executing':p<5?'returning':'visible';
  svg.dataset.selected=route.selected||'';
 };
});
 const panel=window.ServingAdoption?.mount('contrast',viewport,s.data)||null;
 return {paint(...args){inner.paint(...args);},destroy(){panel?.destroy();inner.destroy();}};
}
V['f-serving']=(svg,viewport,s)=>{
 if(s.kind==='serving-route')return servingRoute(svg,viewport,s);
 window.ServingAdoption?.clear(viewport);
 return board(svg,viewport,'cancel',()=>570,H=>{
 const {txt,rect,text,box,token,move,show,set}=H;
 const client=rect(),backend=rect(),head=txt('客户端','teach-heading'),worker=txt('后端 · 请求 R1','teach-heading');
 const clientState=txt('','teach-small'),workState=txt('','teach-small'),resourceState=txt('','teach-small');
 const message=token('cancel-message','取消 R1',90,'teach-token teach-color-2'),resource=token('resource-r1','R1 占用',102,'teach-token teach-color-0');
 const lanes=[rect('teach-track'),rect('teach-track'),rect('teach-track')],work=rect('teach-compute');
 const labels=[txt('客户端 / 控制通知','teach-small'),txt('后端在途操作','teach-small'),txt('资源使用权','teach-small')];
 const eventControl=txt('','teach-small'),eventWork=txt('','teach-small'),note=txt('','teach-heading'),why=txt('断连 → 送达取消 → 在途使用结束 → 归还','teach-small');
 return(p,m)=>{const W=m?360:720;box(client,16,24,W-32,96);box(backend,16,150,W-32,350);text(head,28,54);text(worker,28,178);
  text(clientState,28,88,p>=1?'已断开；不再等待':'连接中；等待 R1');
  const laneX=120,laneW=W-148;
  labels.forEach((e,i)=>text(e,28,232+i*92));
  [0,1,2].forEach(i=>box(lanes[i],laneX,236+i*92,laneW,8));
  move(message,m?W-178:lerp(W-178,W-150,phase(p,1)),m?258:lerp(108,252,phase(p,1)),p>=1?1:0);
  text(eventControl,28,270,p>=2?'取消已送达后端':'取消通知在途');show(eventControl,p>=1);
  box(work,laneX,328,laneW*(.15+.85*Math.min(1,p/3)),12);text(workState,28,364,p>=3?'在途操作已结束；不再安排后续工作':p>=2?'收到取消，等待安全收尾':'当前操作仍在执行');
  move(resource,m?W-178:W-178,420);resource.t.textContent=p>=4?'可再分配':'R1 占用';resource.g.classList.toggle('teach-empty',p>=4);
  text(resourceState,28,464,p>=4?'已归还使用权':p>=3?'安全条件满足，准备归还':'仍不可复用');
  text(note,18,494,p>=4?'完成收尾，资源才可再使用':'取消通知不会立即终止在途操作');text(why,18,532);if(m){note.setAttribute('font-size','16');why.setAttribute('font-size','12');}
  svg.dataset.cancelState=p>=4?'released':p>=3?'draining':p>=2?'received':p>=1?'sent':'running';
 };
});};

V['serving-operations']=(svg,viewport,s)=>board(svg,viewport,'queued-cancel',()=>800,H=>{
 const {txt,rect,text,box,show,token,move}=H,d=s.data;
 const panels=[rect(),rect()],titles=[txt('前端 TokenizerManager','teach-heading'),txt('后端 Scheduler','teach-heading')],states=[txt('','teach-small'),txt('','teach-small')];
 const front=token('front-state','ReqState · R1',150,'teach-token teach-color-0'),req=token('scheduler-req','Req · R1',150,'teach-token teach-color-1');
 const abort=token('abort-msg','AbortReq(R1)',130,'teach-token teach-color-2'),echo=token('echo-msg','取消处理回告',130,'teach-token teach-color-3');
 const client=txt('','teach-small'),resource=txt('GPU KV：整个示例均未分配','teach-heading'),why=txt('同一 rid 关联两个对象；消息不是把对象搬过去。','teach-small'),result=txt('','teach-heading');
 return(p,m)=>{const W=m?360:720,idx=Math.min(4,Math.floor(p)),state=d.states[idx];
  panels.forEach((r,i)=>box(r,16,44+i*336,W-32,170));titles.forEach((h,i)=>text(h,28,74+i*336));
  move(front,34,100);front.g.classList.toggle('teach-empty',!state.frontend);move(req,34,436);req.g.classList.toggle('teach-empty',!state.queued);
  text(states[0],28,172,state.frontend?'本地状态仍在，等后端回告':'本地状态已清理');text(states[1],28,532,state.queued?'waiting_queue 仍引用 Req':'已移出等待队列，回告可发送');
  move(abort,m?198:490,lerp(134,414,phase(p,2)),d.kind==='disconnect'?phase(p,1):0);move(echo,m?198:490,lerp(474,112,phase(p,3)),p>=3?1:0);
  text(client,18,24,state.client==='断连'?'客户端已断连，不会再收到响应':'客户端连接仍在');text(why,18,606);if(m)why.setAttribute('font-size','12');text(resource,18,658);
  text(result,18,720,p>=4?(d.kind==='waiting'?'前端收尾；本例返回 503':'前端收尾；不是客户端收包确认'):'取消已派发 ≠ 后端已处理');if(m)result.setAttribute('font-size','16');
 };
});
})();
