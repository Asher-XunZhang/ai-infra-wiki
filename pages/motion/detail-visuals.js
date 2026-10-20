/* Dedicated encodings: arithmetic, dependencies, input/output alignment and ownership.
 * The timeline only reveals model facts. DOM objects survive reverse seeking. */
(()=>{
'use strict';
const {board,phase,lerp,clamp}=window.TeachingDrawing,V=window.MechanismVisuals;
function resourceComparison(svg,viewport,s){return board(svg,viewport,'resource-comparison',()=>710,H=>{
 const {txt,rect,path,text,box,show,wire,token,move,set}=H,d=s.data;
 const heading=txt('先数工作量，再按每秒容量分组','teach-heading');
 const legend=txt('同色格代表等量工作；格内数字是编号。','teach-small');
 const rows=d.work.map((count,i)=>({bg:rect(),title:txt(i?'计算量：6 份计算':'数据量：8 份数据','teach-heading'),rate:txt('','teach-small'),
  units:Array.from({length:count},(_,j)=>token('resource-work-'+i+'-'+j,String(j+1),26,'teach-token teach-color-'+(i?2:1))),
  slots:Array.from({length:8},()=>rect('teach-slot')),ticks:Array.from({length:9},(_,j)=>txt(String(j),'teach-small')),
  axis:path(),bar:rect(i?'teach-compute':'teach-value teach-color-1'),formula:txt('','teach-number')}));
 const result=txt('','teach-heading'),explain=txt('','teach-small'),reference=txt('','teach-small');
 return(p,m)=>{const W=m?360:720,x=30,cw=(W-60)/8,u=phase(p,1),timeVisible=phase(p,2);
  text(heading,18,30);text(legend,18,59);
  rows.forEach((r,i)=>{const y=82+i*238,rate=d.rates[i],seconds=d.work[i]/rate;
   box(r.bg,16,y,W-32,220);text(r.title,28,y+29);text(r.rate,28,y+55,(i?'算力':'带宽')+'：每秒 '+rate+' 份'+(i?'计算':'数据'));
   r.slots.forEach((slot,j)=>{box(slot,x+j*cw,y+72,cw-3,84);show(slot,u*.8);});
   r.units.forEach((a,j)=>{const col=Math.floor(j/rate),row=j%rate;
    move(a,lerp(x+j*cw+(cw-26)/2,x+col*cw+(cw-26)/2,u),lerp(y+86,y+77+row*19,u));set(a.r,{height:17,rx:3});set(a.t,{y:13,'font-size':11});
   });
   wire(r.axis,`M${x} ${y+174} H${W-27}`,u);
   r.ticks.forEach((tick,j)=>{text(tick,x+j*cw-3,y+193,j===8?'8 秒':String(j));show(tick,u);});
   box(r.bar,x,y+164,seconds*cw*u,7);r.bar.dataset.motionId='resource-demand-'+i;
   text(r.formula,28,y+213,`${d.work[i]} 份 ÷ ${rate} 份/秒 = ${seconds} 秒`);show(r.formula,u);
  });
  text(result,18,604,p<3?'比较两项资源至少需要的时间':`整体至少 ${d.bound} 秒 · 带宽是当前限制`);
  text(explain,18,638,p<3?'每一列容量对应 1 秒；计算与搬运分别核算。':s.name==='算力翻倍'?'计算更快了，但搬完这批数据仍至少要 8 秒。':s.name==='带宽翻倍'?'同样 8 份数据，每秒搬 2 份，搬运下界减半。':'即使计算能力够用，数据也必须搬够才能结束。');
  text(reference,18,675,'整体下界 = max(搬运下界, 计算下界)');show(reference,timeVisible);
  if(m){set(result,{'font-size':16});set(explain,{'font-size':12});}
 };
});}
function matrixArithmetic(svg,viewport,s){return board(svg,viewport,'arithmetic',()=>640,H=>{
 const {txt,rect,text,box,show,token,move}=H,d=s.data;
 const heading=txt('y = W × x','teach-heading'),wTitle=txt('W · 权重 2×2','teach-small'),xTitle=txt('x · 输入 2×1','teach-small');
 const weights=d.weights.map((row,i)=>row.map((n,j)=>token('weight-'+i+'-'+j,String(n),40,'teach-token teach-color-'+i)));
 const input=d.input.map((n,i)=>token('operand-'+i,String(n),40,'teach-token'));
 const multiply=txt('×','teach-heading'),rows=[0,1].map(i=>({title:txt('第 '+(i+1)+' 行','teach-heading'),parts:[0,1].map(j=>token('product-'+i+'-'+j,d.weights[i][j]+'×'+d.input[j],74,'teach-token teach-color-'+i)),plus:txt('+','teach-heading'),equals:txt('=','teach-heading'),result:token('row-result-'+i,String(d.output[i]),54,'teach-token teach-color-2')}));
 const outputTitle=txt('按原行号组成 y · 2×1','teach-heading'),output=d.output.map((n,i)=>token('output-'+i,String(n),54,'teach-token teach-color-2'));
 const note=txt('先相乘，再按行相加；两行可以独立计算。','teach-small');
 return(p,m)=>{const start=m?35:130,xpos=m?245:440;
  text(heading,18,30);text(wTitle,start,68);text(xTitle,xpos-12,68);text(multiply,m?187:344,130);
  weights.forEach((row,i)=>row.forEach((t,j)=>move(t,start+j*52,86+i*44)));input.forEach((t,i)=>move(t,xpos,86+i*44));
  rows.forEach((r,i)=>{const y=244+i*120,x=m?28:142,span=m?96:144; text(r.title,18,y-20);
   r.parts.forEach((t,j)=>{move(t,lerp(start+j*52,x+j*span,phase(p,0)),lerp(86+i*44,y,phase(p,0)),phase(p,0));t.t.textContent=d.weights[i][j]+'×'+d.input[j]+'='+String(d.weights[i][j]*d.input[j]);});
   text(r.plus,x+81,y+24);show(r.plus,phase(p,1));text(r.equals,x+span+82,y+24);show(r.equals,phase(p,1));move(r.result,x+2*span+8,y,phase(p,1));
  });
  text(outputTitle,18,510);show(outputTitle,phase(p,2));output.forEach((t,i)=>{const u=phase(p,2),x=(m?28:142)+2*(m?96:144)+8;move(t,lerp(x,m?262:530,u),lerp(244+i*120,530+i*42,u),u);});
  text(note,18,620);if(m)note.setAttribute('font-size','13');
 };
});}
function graphSubmission(svg,viewport,s){return board(svg,viewport,'graph-submission',()=>760,H=>{
 const {txt,rect,path,text,box,show,wire,set}=H,d=s.data.schedule;
 const title=txt('同一条依赖链：A → B → C','teach-heading');
 const meaning=txt('A：加 1；B：乘 2；C：加 3。每轮都重新算。','teach-small');
 const prep=[rect('teach-graph-slot'),rect('teach-graph-slot')],prepText=[txt('捕获：记录操作与依赖','teach-small'),txt('实例化：可重放的图','teach-small')],prepLink=path();
 const prepNote=txt('首次准备 · 后续重复使用','teach-small'),prepCost=txt('一次性准备（教学假设，不标时长）','teach-small'),roundLabel=txt('','teach-heading'),axis=path();
 const ticks=Array.from({length:d.duration+1},(_,i)=>txt(String(i),'teach-small'));
 function event(e,id,cls){return {...e,name:e.id,outline:rect('teach-slot'),fill:rect(cls),label:txt(e.id,'teach-small'),id};}
 const panels=[0,1].map(i=>({bg:rect(),title:txt(i?'CUDA Graph · 一次图提交':'普通方式 · 三次逐项提交','teach-heading'),cpu:txt('CPU','teach-small'),gpu:txt('GPU','teach-small'),
  calls:(i?d.graphCPU:d.cpu).map((e,j)=>event(e,'launch-'+i+'-'+j,'teach-value teach-color-4')),
  kernels:(i?d.graphGPU:d.gpu).map((e,j)=>event(e,'kernel-'+i+'-'+j,'teach-value teach-color-'+j)),
  arrows:(i?[d.graphGPU[0]]:d.gpu).map(()=>path()),
  gaps:i?[]:d.gpu.slice(1).filter((e,j)=>e.start>d.gpu[j].end).map((e,j)=>({start:d.gpu[j].end,end:e.start,bg:rect('teach-mask'),label:txt('等','teach-small')})),
  cursor:path(),end:path(),result:txt('','teach-small')}));
 const takeaway=txt('','teach-heading'),explain=txt('','teach-small'),legend=txt('','teach-small'),scaleNote=txt('横轴：同一教学时间刻度；不是硬件实测值。','teach-small');
 return(p,m)=>{const W=m?360:720,x=64,span=W-x-26,scale=span/d.duration,round=p>3?1:0,local=clamp((p-(round?3:1))/2),clock=local*d.duration,ready=phase(p,0),input=round?4:2;
  svg.dataset.graphClock=clock.toFixed(4);svg.dataset.graphRound=String(round+1);
  text(title,18,30);text(meaning,18,56);
  prep.forEach((r,i)=>{box(r,18+i*(W/2),78,W/2-38,37);show(r,.25+.75*ready);text(prepText[i],26+i*W/2,101);});
  wire(prepLink,`M${W/2-16} 97 H${W/2+12}`,ready);text(prepNote,18,140,p<1?'首次准备：捕获不执行 GPU 运算':'首次准备已完成 · 本轮复用');
  text(prepCost,18,168);
  text(roundLabel,18,188,`第 ${round+1} 轮 · 输入 ${input}`);
  wire(axis,`M${x} 211 H${W-18}`);ticks.forEach((t,i)=>{text(t,x+i*scale-3,202);show(t,m&&d.duration>8&&i%2&&i!==d.duration?0:1);});
  panels.forEach((r,i)=>{const y=225+i*190,finish=r.kernels.at(-1).end;box(r.bg,16,y,W-32,176);text(r.title,28,y+27);text(r.cpu,24,y+69);text(r.gpu,24,y+120);
   function paint(events,offset){events.forEach(e=>{const amount=clamp((clock-e.start)/(e.end-e.start)),bx=x+e.start*scale,bw=(e.end-e.start)*scale;
    box(e.outline,bx,y+offset,bw,28);show(e.outline,.5*ready);box(e.fill,bx,y+offset,bw*amount,28);e.fill.dataset.motionId=e.id;set(e.fill,{'data-start':e.start,'data-end':e.end,'data-progress':amount.toFixed(4)});
    text(e.label,bx+bw/2,y+offset+19,e.name==='Graph'?'图':e.name);set(e.label,{'text-anchor':'middle'});show(e.label,ready);
    // Labels stay legible on filled and planned portions in both themes.
    set(e.label,{class:amount>.7?'hardware-event-label is-filled':'hardware-event-label'});
   });}
   paint(r.calls,49);paint(r.kernels,100);
   r.arrows.forEach((a,j)=>{const start=r.calls[j].end,end=r.kernels[j].start;wire(a,`M${x+start*scale} ${y+79} L${x+end*scale+2} ${y+98}`,clock>=start?1:0);});
   r.gaps.forEach(g=>{const amount=clamp((clock-g.start)/(g.end-g.start));box(g.bg,x+g.start*scale,y+100,(g.end-g.start)*scale*amount,28);text(g.label,x+(g.start+g.end)/2*scale,y+119);set(g.label,{'text-anchor':'middle'});show(g.label,amount>.65?1:0);});
   wire(r.cursor,`M${x+clock*scale} ${y+40} V${y+134}`,p>1&&clock<finish?1:0);set(r.cursor,{'stroke-dasharray':'3 3','marker-end':'none'});show(r.cursor,p>1&&clock<finish?.6:0);
   wire(r.end,`M${x+finish*scale} ${y+96} V${y+140}`,clock>=finish?1:0);
   text(r.result,28,y+160,clock>=finish?`本轮完成 · 输出 ${round?13:9}`:clock===0?'准备本轮提交':i&&clock>=2?'CPU 已完成图提交，GPU 继续计算':'提交与计算可重叠进行');
  });
  text(takeaway,18,642,d.kernelDuration===1?'少等提交，三个 kernel 没有变短':'CPU 少提交，但本例总耗时相同');
  text(explain,18,674,d.kernelDuration===1?'虚线空档：GPU 等下一次提交完成。':'较长计算覆盖了后续提交，GPU 原本就能接续执行。');
  text(legend,18,706,'同色块 = 同一个 kernel；块宽 = 计算用时。');text(scaleNote,18,738);
  if(m){set(explain,{'font-size':11});set(takeaway,{'font-size':16});}
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
V['f-parallel']=(svg,viewport,s)=>board(svg,viewport,'pipeline-grid',()=>960,H=>{
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
  const done=Math.min(s.data.n,Math.max(0,Math.floor(p)-1));text(completed,18,440,'已完成两个 stage：'+(done?Array.from({length:done},(_,i)=>'MB'+(i+1)).join('、'):'暂无'));if(m)completed.setAttribute('font-size','16');text(reason,18,478,p<1?'Stage 1 等待 Stage 0 产生激活。':p<2?'激活抵达 Stage 1 后，它才能开始计算。':'时间格保留各 stage 的计算与空闲记录。');if(m)reason.setAttribute('font-size','13');paths.forEach((h,i)=>{text(h,18,540+i*36,i===1&&s.data.n===1?'本条件只送入 MB1 = {R1,R2}':undefined);if(m)h.setAttribute('font-size','13');});
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
function foundationSpeculation(svg,viewport,s){return board(svg,viewport,'speculative-prefix',()=>700,H=>{
 const {txt,rect,path,text,box,show,wire,token,move}=H,d=s.data;
 const context=txt('根 token → 本轮草稿','teach-small'),predictionRule=txt('预测只依赖该候选之前的前缀','teach-small'),heads=['① 草稿候选 · 尚未确认','② 目标模型 · 一次验证','③ 正式输出'].map(t=>txt(t,'teach-heading'));
 const root=txt('根','teach-small'),rootToken=token('root-token','t0',48,'teach-token teach-color-4');
 const draft=d.draft.map((v,i)=>token('candidate-'+i,v,48,'teach-token teach-color-'+i));
 const predicted=d.predictions.map((v,i)=>token('prediction-'+i,v,48,'teach-token teach-color-4'));
 const checks=Array.from({length:4},()=>txt('','teach-small'));
 const output=d.output.map((v,i)=>token('committed-output-'+i,v,48,'teach-token teach-color-'+(i<d.accepted?i:4)));
 const validTitle=txt('有效历史：根 + 接受前缀','teach-small'),validHistory=d.validKV.map((v,i)=>token('valid-history-'+i,v,48,'teach-token teach-color-'+(i===0?4:i)));
 const accepted=rect('teach-read'),rejected=rect('teach-focus'),focus=rect('teach-map'),bonus=txt('补出','teach-small');focus.style.fill='none';
 const prefix=txt('','teach-small'),comparison=txt('','teach-small'),cause=txt('','teach-small');
 const reason=txt('','teach-heading'),legend=txt('候选颜色跟随身份；目标预测使用统一颜色。','teach-small');
 return(p,m)=>{const start=m?24:76,gap=m?64:128,at=i=>start+i*gap;
  text(context,18,28);text(root,18,88);move(rootToken,18,100,1);heads.forEach((h,i)=>text(h,18,[70,206,540][i]));
  draft.forEach((t,i)=>{const u=phase(p,i/4,(i+1)/4);move(t,at(i),lerp(92,104,u),u);t.g.classList.toggle('teach-empty',p>=3&&i>=d.accepted);});
  text(predictionRule,18,232);show(predictionRule,phase(p,1));predicted.forEach((t,i)=>move(t,at(i),250,phase(p,1)*(i===4&&d.accepted<4?0:p>=3&&i>d.accepted?.3:1)));
  const checked=Math.min(d.accepted===4?4:d.accepted+1,Math.floor(phase(p,2)*4+1e-8));
  checks.forEach((t,i)=>{text(t,at(i),308,i>=checked?(checked>d.accepted?'舍弃':''):i<d.accepted?'接受':i===d.accepted?'分歧':'舍弃');});
  const n=Math.min(d.accepted,checked);box(accepted,at(0)-4,143,n?Math.max(56,(n-1)*gap+56):0,5);show(accepted,p>=2&&n?1:0);
  box(rejected,at(Math.min(d.accepted,3))-4,99,56,42);rejected.style.fill='none';show(rejected,d.accepted<4&&checked>d.accepted?1:0);
  const comparing=Math.min(d.accepted,Math.floor(phase(p,2)*4)),validPrefix=d.draft.slice(0,comparing);
  box(focus,at(comparing)-4,246,56,42);show(focus,p>=2?1:0);
  text(prefix,18,360,'输入前缀：已有文本'+(validPrefix.length?' + '+validPrefix.join(' / '):''));show(prefix,p>=2?1:0);
  text(comparison,18,390,comparing===4?`目标预测“${d.predictions[4]}” → 多得到一个 token`:`预测“${d.predictions[comparing]}” ↔ 候选“${d.draft[comparing]}”：${d.predictions[comparing]===d.draft[comparing]?'相同，可接受':'不同，停止接受'}`);show(comparison,p>=2?1:0);
  text(cause,18,420,p>=3&&d.accepted<4?'前缀已改变，后面的匹配不再有效。':'每个位置只使用它之前的前缀进行预测。');show(cause,p>=2?1:0);
  text(bonus,at(d.accepted),456,'补出“'+d.replacement+'”');show(bonus,p>=3?1:0);
  output.forEach((t,i)=>{const u=phase(p,3);move(t,at(i),lerp(i<d.accepted?104:246,574,u),p>3?1:0);});
  text(validTitle,18,620);show(validTitle,p>=3);validHistory.forEach((t,i)=>move(t,m?18+i*52:110+i*56,636,p>=3?1:0));
  text(reason,18,490,p>=3?(d.accepted===4?'全部接受，再补出一个 token。':'分歧处停下，其后候选都不接受。'):'');
  text(legend,18,678,p>=4?(m?`输出 ${d.output.length} = 接受 ${d.accepted} + 补出 1；有效历史含根与前缀。`:`输出数量 = ${d.accepted} 个接受 + 1 个补出 = ${d.output.length} 个；有效历史只含根与接受前缀。`):'候选颜色跟随身份；目标预测使用统一颜色。');if(m)legend.style.fontSize='10px';if(m){reason.setAttribute('font-size','16');legend.setAttribute('font-size','12');}
  svg.dataset.comparedPrefix=validPrefix.join('');
 };
});}
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

function servingRoute(svg,viewport,s){return board(svg,viewport,'serving-route',m=>m?540:440,H=>{
 const {txt,rect,path,text,box,token,move,show,wire}=H,d=s.data,variant=d.variant||0;
 const client=rect(),panels=[rect(),rect()],titles=[txt('实例 A · 完整模型','teach-heading'),txt('实例 B · 完整模型','teach-heading')];
 const health=[txt('health ✓','teach-small'),txt('health ✓','teach-small')],queue=[txt('queue: 0','teach-small'),txt('queue: 0','teach-small')];
 const panelState=[txt('','teach-small'),txt('','teach-small')],routeLabel=txt('','teach-small'),idleLabel=txt('A 未收到 R1','teach-small'),summary=txt('健康 / 队列 / 执行资源分列','teach-small');
 const r1=token('route-r1','R1',56,'teach-token teach-color-0'),result=token('route-result','结果',56,'teach-token teach-color-2');
 const arrow=path(),back=path(),note=txt('','teach-small'),state=txt('','teach-heading'),clientState=txt('','teach-small'),clientTitle=txt('客户端','teach-heading');
 const track=rect('teach-track'),bar=rect('teach-compute');
 return(p,m)=>{const W=m?360:720,clientW=m?W-40:196,x=[m?20:238,m?20:470],y=[m?150:70,m?300:70],pw=m?W-40:214,ph=m?126:180;
  box(client,20,58,clientW,84);text(clientTitle,32,84);text(clientState,32,112,p>=5?'已看到结果':p>=1?'等待 R1':'准备发送 R1');
  for(let i=0;i<(variant?2:1);i++){box(panels[i],x[i],y[i],pw,ph);text(titles[i],x[i]+10,y[i]+24);text(health[i],x[i]+10,y[i]+52);text(queue[i],x[i]+10,y[i]+76);text(panelState[i],x[i]+10,y[i]+108,variant===0?(p>=3?'R1 执行中':'等待执行资源'):i===0?'空闲 · 未接到 R1':p>=3?'R1 执行中':'等待 R1');}
  text(summary,20,m?440:266);
  if(!variant){[titles[1],health[1],queue[1],panelState[1]].forEach(e=>show(e,0));}
  const ex=variant?x[1]+pw/2:x[0]+pw/2,ey=variant?y[1]+ph/2:y[0]+ph/2;
  move(r1,20,118,p>=1?1:0);
  const forward=m?(variant?`M${clientW+20} 100 H${W-12} V${y[1]-6} H${ex}`:`M${clientW/2+20} 142 V${y[0]-6}`):(variant?`M${clientW+20} 100 V36 H${ex} V${y[1]-6}`:`M${clientW+20} 100 H${ex}`);wire(arrow,forward,p>=1?1:0);
  text(routeLabel,variant?x[1]+10:238,p>=1?(variant?40:56):56,variant?'只路由到 B':'进入实例');show(routeLabel,p>=1);
  show(idleLabel,0);
  box(track,ex-40,ey+48,80,8);box(bar,ex-40,ey+48,80*Math.min(1,Math.max(0,(p-2)/1)),8);show(bar,p>=3?1:0);
  move(result,ex,ey+ph+8,p>=4?1:0);
  const backward=m?(variant?`M${ex} ${y[1]-28} H${W-12} V100 H${clientW+20}`:`M${ex} ${y[0]-28} V142 H${clientW/2+20}`):(variant?`M${ex} ${ey-28} V28 H${clientW+20} V100`:`M${ex} ${ey-28} V100 H${clientW+20}`);wire(back,backward,p>=4?1:0);
  text(note,20,m?470:380,p>=4?(variant?'结果从 B 返回；A 的健康状态不等于 R1 被 A 执行。':'结果从实例返回客户端；本图不虚构网络时延。'):'健康通过不等于接纳；请求仍要等待目标实例的资源。');
  text(state,20,m?510:340,p>=5?'客户端可见结果':p>=3?'执行 / 回程':'接收 / 等待');if(m){state.setAttribute('font-size','16');note.setAttribute('font-size','11');}
 };
});}
V['f-serving']=(svg,viewport,s)=>s.kind==='serving-route'?servingRoute(svg,viewport,s):board(svg,viewport,'cancel',()=>570,H=>{
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
});

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
