/* Source-facing diagrams: keep implementation structure and data identity distinct. */
(() => {
'use strict';
const {board,phase,lerp,clamp}=window.TeachingDrawing,V=window.MechanismVisuals;
V['model-execution']=(svg,viewport,s)=>s.kind==='graph'?graph(svg,viewport,s):execution(svg,viewport,s);
function graph(svg,viewport,s){return board(svg,viewport,'graph',m=>m?780:570,H=>{
 const {txt,rect,path,text,box,show,wire,token,move}=H;
 const titles=['真实输入 · 3 条','容量 4 的图桶','真实 logits · 3 行'].map(x=>txt(x,'teach-heading'));
 const bucket=rect(),slots=Array.from({length:4},()=>rect('teach-graph-slot'));
 const input=[1,2,3].map(i=>token('request-R'+i,'R'+i,78,'teach-token teach-color-'+(i-1)));
 const output=[1,2,3].map(i=>token('logits-R'+i,'R'+i+' 分数',98,'teach-token teach-color-'+(i-1)));
 const pad=token('pad-input','PAD',78,'teach-token teach-empty'),blocked=txt('PAD 无对外输出','teach-small'),count=txt('','teach-heading'),run=rect('teach-compute'),note=txt('','teach-small');
 return(p,m)=>{
  const by=m?176:74,bx=m?50:250,bw=m?260:210,oy=m?676:132;
  text(titles[0],18,32);text(titles[1],m?50:250,m?160:48);text(titles[2],m?18:520,m?650:48);
  box(bucket,bx,by,bw,374);show(bucket,phase(p,0));show(titles[1],phase(p,0));show(titles[2],phase(p,3));
  slots.forEach((a,i)=>{box(a,bx+18,by+24+i*82,bw-36,62);show(a,phase(p,0));});
  input.forEach((a,i)=>{const t=phase(p,1),sx=m?24+i*108:42,sy=m?64:by+38+i*82;move(a,lerp(sx,bx+36,t),lerp(sy,by+38+i*82,t));});
  move(pad,bx+36,by+38+3*82,phase(p,1));
  output.forEach((a,i)=>{const t=phase(p,3),ex=m?18+i*112:536,ey=m?oy:by+38+i*82;move(a,lerp(bx+36,ex,t),lerp(by+38+i*82,ey,t),t);});
  box(run,bx+12,by+360,(bw-24)*phase(p,2),5);show(run,phase(p,2));text(blocked,bx+18,by+350);show(blocked,phase(p,3));
  text(count,18,m?750:520,p<2?'真实请求数始终为 3':p<4?'图桶容量 4 ≠ 请求数 4':'裁掉补齐行，再交给后续采样');
 };
});}
function execution(svg,viewport,s){return board(svg,viewport,'execution',m=>m?1470:970,H=>{
 const {txt,rect,path,text,box,show,wire,token,move,set}=H;
 const top=['Scheduler','Worker','ModelRunner'].map(x=>({r:rect(),t:txt(x,'teach-heading')}));
 const model=rect(),modelTitle=txt('本轮执行地图 · 按职责展开','teach-heading');
 const names=['Embedding（模型）','MLA Attention（模型）','FFN / MoE（模型）','后续层 / 输出头（模型）','Runner 准备元数据','压缩历史与位置部分','Triton attention kernel','Worker 采样'];
 const nodes=names.map(x=>({r:rect(),t:txt(x,'teach-heading')}));
 const callLinks=Array.from({length:7},()=>path()),topLinks=Array.from({length:3},()=>path());
 const ids=token('batch-input','ID [41,52]',112,'teach-token teach-color-0'),features=token('hidden-states','向量 X',100,'teach-token teach-color-1'),query=token('mla-query','查询 Q',90,'teach-token teach-color-2'),result=token('attention-result','Attention 结果',138,'teach-token teach-color-3'),logits=token('vocab-logits','词表分数',100,'teach-token teach-color-4'),selected=token('next-ids','下一 ID [8,7]',132,'teach-token teach-color-0');
 const metadata=txt('本轮位置 / 长度 / KV 索引','teach-small'),history=Array.from({length:3},(_,i)=>rect('teach-contribution teach-color-'+i));
 const note=txt('图示调用与张量依赖；不是新增网络服务','teach-small');
 const highlight=rect('teach-focus');svg.append(highlight);highlight.style.fill='none';
 return(p,m)=>{
  top.forEach((a,i)=>{const x=m?20:18+i*234,y=m?18+i*90:18;box(a.r,x,y,m?320:216,m?72:48);text(a.t,x+14,y+30);});
  const base=m?340:150;box(model,16,base,m?328:688,m?884:624);text(modelTitle,30,base+28);
  const positions=m?[[34,base+66],[34,base+176],[34,base+610],[34,base+734],[34,base+286],[34,base+390],[34,base+500],[34,base+942]]:[[38,base+70],[38,base+210],[38,base+380],[38,base+510],[380,base+70],[380,base+210],[380,base+380],[38,base+690]];
  nodes.forEach((a,i)=>{const[x,y]=positions[i];box(a.r,x,y,m?292:300,74);text(a.t,x+12,y+26);});
  const mid=i=>({x:positions[i][0]+146,y:positions[i][1]+74});
  [[0,1],[4,0],[5,6],[1,6],[6,2],[2,3],[3,7]].forEach(([a,b],i)=>{
   const [ax,ay]=positions[a],[bx,by]=positions[b],w=m?292:300;
   if(ax!==bx){const right=ax<bx,start=right?ax+w:ax,end=right?bx:bx+w,channel=(start+end)/2;wire(callLinks[i],`M${start} ${ay+37} H${channel} V${by+37} H${end}`);}
   else if(by>ay&&by-ay<180)wire(callLinks[i],`M${ax+w/2} ${ay+74} V${by}`);
   else{const channel=by<ay?(m?348:696):20,edge=by<ay?ax+w:ax;wire(callLinks[i],`M${edge} ${ay+37} H${channel} V${by+37} H${edge}`);}
  });
  for(let i=0;i<2;i++)wire(topLinks[i],m?`M180 ${90+i*90} V${108+i*90}`:`M${234+i*234} 42 H${252+i*234}`);
  wire(topLinks[2],m?`M180 270 V300 H350 V${positions[4][1]+37} H326`:`M594 66 V126 H526 V${positions[4][1]}`);
  text(metadata,positions[4][0]+12,positions[4][1]+53);show(metadata,phase(p,1));
  history.forEach((a,i)=>{box(a,positions[5][0]+16+i*82,positions[5][1]+42,72,18);show(a,phase(p,1));});
  const start={x:m?42:38,y:m?54:80},forward={x:m?42:272,y:m?144:80},embed={x:positions[0][0]+14,y:positions[0][1]+36},mla={x:positions[1][0]+14,y:positions[1][1]+36},kernel={x:positions[6][0]+14,y:positions[6][1]+36},ffn={x:positions[2][0]+14,y:positions[2][1]+36},head={x:positions[3][0]+14,y:positions[3][1]+36},sample={x:positions[7][0]+14,y:positions[7][1]+36};
  const meta={x:m?42:510,y:m?234:80};
  const idFrom=p<1?start:p<2?forward:meta,idTo=p<1?forward:p<2?meta:embed,idProgress=p<1?phase(p,0):p<2?phase(p,1):phase(p,2,2.6);
  move(ids,lerp(idFrom.x,idTo.x,idProgress),lerp(idFrom.y,idTo.y,idProgress),1-phase(p,2.6,3));
  move(features,lerp(embed.x,mla.x,phase(p,3)),lerp(embed.y,mla.y,phase(p,3)),phase(p,2.6,3)*(1-phase(p,3.6,4)));
  move(query,lerp(mla.x,kernel.x,phase(p,4)),lerp(mla.y,kernel.y,phase(p,4)),phase(p,3.6,4)*(1-phase(p,5,5.1)));
  const rt=phase(p,5.1,5.45);
  move(result,lerp(kernel.x,ffn.x,rt),lerp(kernel.y,ffn.y,rt),phase(p,5.1,5.15)*(1-phase(p,5.45,5.65)));
  move(logits,lerp(head.x,sample.x,phase(p,6)),lerp(head.y,sample.y,phase(p,6)),phase(p,5.8,6));
  move(selected,m?50:396,positions[7][1]+(m?104:32),phase(p,6));
  const active=p<1?-1:p<2?-2:p<3?4:p<4?0:p<5?1:p<5.45?6:p<5.8?2:p<6.7?3:7;
  if(active>=0){box(highlight,positions[active][0]-5,positions[active][1]-5,m?302:310,84);show(highlight,1);}else show(highlight,0);
  text(note,18,m?1446:948);if(m)note.setAttribute('font-size','11');
 };
});}
V['kv-memory']=(svg,viewport,s)=>board(svg,viewport,'memory',m=>m?1010:650,H=>{
 const {txt,rect,path,text,box,show,wire,token,move,set}=H,K=s.model;
 const title=txt('R1 的逻辑位置 · 按原始顺序','teach-heading'),legend=txt('虚框：预留地址，尚未写入 · 实色：已写入','teach-small');
 const logical=Array.from({length:s.target},(_,i)=>token('logical-'+i,'p'+i,42,'teach-token teach-color-1'));
 const names=[1,7,3].map(id=>txt('物理页 '+id,'teach-heading'));
 const cells=Array.from({length:12},(_,i)=>({slot:[4,28,12][Math.floor(i/4)]+i%4,bg:rect('teach-memory-slot'),fill:rect('teach-memory-fill'),label:txt('','teach-number'),address:txt('','teach-small')}));
 const mapping=path('teach-map'),pointer=token('mapping-cursor','地址映射',80,'teach-token teach-color-3');
 const explanation=txt('','teach-heading'),owned=txt('','teach-small'),hint=txt('点击上方 p 编号，可追踪它指向的物理槽','teach-small');let focus=null,last=0;
 logical.forEach((a,i)=>{set(a.g,{role:'button',tabindex:0,'aria-label':'查看逻辑位置 p'+i+' 的物理槽'});const choose=()=>{focus=i;draw(last,viewport.clientWidth<600);};a.g.addEventListener('click',choose);a.g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}});});
 function draw(p,m){last=p;if(p<1)focus=null;const a=K.extend(s.target,Math.floor(p)),b=K.extend(s.target,Math.min(4,Math.floor(p)+1)),u=phase(p,Math.floor(p));
  text(title,18,28);text(legend,18,m?190:116);if(m)legend.setAttribute('font-size','11');
  const positions=logical.map((t,i)=>{const x=m?20+(i%6)*54:20+i*61,y=m?54+Math.floor(i/6)*58:54;move(t,x,y,i<6?1:phase(p,0));return{x,y};});
  names.forEach((t,r)=>text(t,18,(m?250:192)+r*(m?208:124)));
  cells.forEach((c,i)=>{const r=Math.floor(i/4),j=i%4,x=m?24+(j%2)*166:148+j*136,y=(m?270:170)+r*(m?208:124)+(m?Math.floor(j/2)*86:0),w=m?142:120;
   const ca=a.pages[r].cells[j],cb=b.pages[r].cells[j],written=lerp(Number(ca.state==='written'),Number(cb.state==='written'),u),reserved=ca.state==='reserved'||cb.state==='reserved';
   box(c.bg,x,y,w,68);c.bg.setAttribute('class',written===1?'teach-memory-slot is-written':reserved?'teach-memory-slot is-reserved':'teach-memory-slot');box(c.fill,x+2,y+2,(w-4)*written,34);text(c.label,x+12,y+24,ca.p>=0?'p'+ca.p:cb.p>=0&&p>0?'p'+cb.p:'空位');text(c.address,x+12,y+53,'槽 '+c.slot);c.bg.dataset.slot=c.slot;c.bg.dataset.written=written.toFixed(3);c.x=x;c.y=y;c.w=w;
  });
  const selected=focus!==null&&focus<a.mapping.length?focus:p>=4?s.target-1:p>=3?s.target-1:p>=2?7:p>=1?6:5,slot=(p>=1?K.extend(s.target,1):a).mapping[selected],cell=cells.find(c=>c.slot===slot),origin=positions[selected];
  if(cell&&origin){const x=cell.x+cell.w/2,y=cell.y;wire(mapping,`M${origin.x+21} ${origin.y+34} V${m?222:140} H${m?350:126} V${cell.y+34} H${m?cell.x+cell.w:cell.x}`,1);move(pointer,m?230:540,m?206:130);text(explanation,18,m?934:586,`p${selected} → 槽 ${slot} = 页 ${Math.floor(slot/4)} / 偏移 ${slot%4}`);}
  text(owned,18,m?972:624,`当前持有 ${a.ownedPages} 页；映射 ${a.mapping.length} 个位置，已写入 ${a.written} 个。`);text(hint,18,m?998:648);if(m)hint.setAttribute('font-size','11');
 }
 return draw;
});
V.scheduling=(svg,viewport,s)=>s.kind==='chunk'?window.ChunkPrefillVisual(svg,viewport,s):board(svg,viewport,'admission',m=>m?1130:760,H=>{
 const {txt,rect,path,text,box,show,wire,token,move,set}=H,S=s.model;
 const title=txt('候选请求 · 深色为命中历史，浅色为新输入','teach-heading'),orderText=txt('','teach-small');
 const reqs=S.requests.map((r,i)=>({r,label:token(r.id,r.id,56,'teach-token teach-color-'+i),pieces:Array.from({length:r.input},(_,j)=>rect(j<r.hit?'teach-read':'teach-contribution teach-color-'+i)),note:txt('','teach-small'),status:txt('','teach-small')}));
 const meterNames=['本轮输入额度','KV 预算账本','请求行'];
 const meters=meterNames.map(x=>({panel:rect(),title:txt(x,'teach-heading'),track:rect('teach-track'),fill:rect('teach-value'),value:txt('','teach-number')}));
 const check=rect('teach-check'),checkTitle=txt('','teach-heading'),formula=txt('','teach-number'),prefix=txt('','teach-small');
 const plan=rect('teach-plan'),planTitle=txt('can_run_list · 本轮计划，尚未执行','teach-heading');
 const members=S.requests.map((r,i)=>token('plan-'+r.id,r.id,64,'teach-token teach-color-'+i));
 return(p,m)=>{
  const index=Math.floor(p),a=s.events[index],b=s.events[Math.min(index+1,s.events.length-1)],u=phase(p,index),current=u>0?b:a;
  text(title,18,28);text(orderText,18,54,'本轮检查顺序：'+a.order.join(' → '));if(m)title.setAttribute('font-size','12');
  reqs.forEach(({r,label,pieces,note,status},i)=>{const y=80+i*108,x=24,last=s.events.slice(0,index+1).reverse().find(e=>e.current===r.id),rejected=last?.kind==='blocked';
   move(label,x,y);pieces.forEach((cell,j)=>box(cell,x+72+j*28,y+4,23,26));text(note,x+72,y+53,`输入 ${r.input} · 命中 ${r.hit} · 新算 ${r.input-r.hit}`);text(status,x,y+84,a.admitted.includes(r.id)?'✓ 已提交计划':b.admitted.includes(r.id)&&u>0?'正在提交计划':rejected?'未加入：'+({kv:'KV 检查未过',input:'输入额度不足',rows:'请求行不足'}[last.reason]||last.reason):current.current===r.id?'正在检查 / 处理':a.kind==='ready'?'本轮未轮到':'等待检查');
   label.g.dataset.admitted=String(a.admitted.includes(r.id));
  });
  const mx=m?18:378,my=m?436:70,mw=m?324:324;
  const values=[lerp(a.usedInput,b.usedInput,u),lerp(a.offset+a.protectedTokens+a.trial,b.offset+b.protectedTokens+b.trial,u),lerp(a.admitted.length,b.admitted.length,u)],limits=[a.config.input,a.config.kv,a.config.rows];
  meters.forEach((meter,i)=>{const y=my+i*102;box(meter.panel,mx,y,mw,88);text(meter.title,mx+12,y+25);box(meter.track,mx+12,y+44,mw-24,16);box(meter.fill,mx+12,y+44,(mw-24)*clamp(values[i]/limits[i]),16);const before=[a.usedInput,a.offset+a.protectedTokens+a.trial,a.admitted.length][i],after=[b.usedInput,b.offset+b.protectedTokens+b.trial,b.admitted.length][i];text(meter.value,mx+12,y+80,u>0&&u<1&&before!==after?`${before} → ${after} / ${limits[i]}（变更中）`:`${u>=1?after:before} / ${limits[i]}`);meter.fill.dataset.used=values[i].toFixed(3);});
  const display=current.current?current:s.events.slice(0,index+1).reverse().find(e=>e.current&&['blocked','commit','stop'].includes(e.kind))||current;
  const cy=m?766:416;box(check,18,cy,m?324:684,114);text(checkTitle,30,cy+26,display.current?`${display.current} 的${current.kind==='ready'?'最后':'本次'}检查`:'等待下一次检查');
  const request=S.requests.find(r=>r.id===display.current),need=request?request.input-request.hit+request.maxNew+1:0;
  text(formula,30,cy+55,request?`KV：${request.input-request.hit} + 2 + 1 = ${need}`:'先排序，再逐条核对条件');
  text(prefix,30,cy+84,request?display.reason==='kv'?`要求 ${need} < ${display.checkAvailable}；不成立，未加入计划`:`临时保护 ${display.trial}；可用 ${display.checkAvailable??display.available}；${display.kind==='commit'?'检查已通过':display.kind==='blocked'?'未通过 '+({rows:'请求行',input:'输入额度'}[display.reason]||display.reason):'需求须严格小于可用量'}`:'输入、KV 和请求行是不同的约束');if(m)prefix.setAttribute('font-size','11');
  const py=m?926:578;box(plan,18,py,m?324:684,116);text(planTitle,30,py+28);if(m)planTitle.setAttribute('font-size','14');
  members.forEach((t,i)=>{const start=a.admitted.includes(S.requests[i].id)?1:0,end=b.admitted.includes(S.requests[i].id)?1:0,amount=lerp(start,end,u);const ai=a.admitted.indexOf(S.requests[i].id),bi=b.admitted.indexOf(S.requests[i].id),position=ai<0?(bi<0?i:bi):lerp(ai,bi<0?ai:bi,u);move(t,38+position*96,py+56,amount);t.g.dataset.committed=amount.toFixed(3);});
 };
});
})();

/* Queue membership is a reference graph; the Req never travels between boxes. */
(() => {
'use strict';
const {board,phase,lerp}=window.TeachingDrawing,V=window.MechanismVisuals;
V.dataflow=(svg,viewport,s)=>board(svg,viewport,'references',m=>m?760:660,H=>{
 const {txt,rect,path,text,box,show,wire,token,move,set}=H;
 const definitions=[['bootstrap','bootstrap_queue'],['waiting','waiting_queue'],['batch','ScheduleBatch'],['chunk','chunked_req'],['inflight','inflight_queue']];
 const refs=definitions.map(([id,name])=>({id,box:rect(),name:txt(name,'teach-number'),state:txt('无引用','teach-small'),line:path()}));
 const req=token('Req','本地 Req R0',156,'teach-token teach-color-0'),scope=txt('只观察 PP0 的一个本地对象','teach-heading'),legend=txt('容器 → Req 的箭头表示引用；不是复制或搬运请求','teach-small'),resource=rect('teach-gate'),resourceText=txt('','teach-small'),exit=txt('','teach-heading');
 const index=id=>s.actors.findIndex(x=>x.id===id);
 return(p,m)=>{
  const n=Math.floor(p),a=s.frames[n],b=s.frames[Math.min(n+1,s.frames.length-1)],u=phase(p,n),rx=m?190:282,ry=m?314:290;
  text(scope,18,30);if(m)scope.setAttribute('font-size','14');text(legend,18,m?720:620);if(m)legend.setAttribute('font-size','10');move(req,rx,ry);
  refs.forEach((r,i)=>{const x=m?16:i<3?18+i*236:136+(i-3)*280,y=m?72+i*116:i<3?90:470,w=m?150:208,value=lerp(a.actors[index(r.id)].opacity,b.actors[index(r.id)].opacity,u);
   box(r.box,x,y,w,72);text(r.name,x+10,y+27);if(m)r.name.setAttribute('font-size','12');text(r.state,x+10,y+54,value===1?'持有引用':value===0?'无引用':'引用正在变化');r.box.classList.toggle('is-reference',value>0);
   const endX=m?rx:rx+78,endY=ry+17;wire(r.line,m?`M${x+w} ${y+36} C182 ${y+36} 182 ${endY} ${endX} ${endY}`:`M${x+w/2} ${i<3?y+72:y} L${endX} ${i<3?ry:ry+34}`,value);r.line.dataset.reference=r.id;r.line.dataset.amount=value.toFixed(3);
  });
  box(resource,m?rx:510,m?386:320,156,64);text(resourceText,m?rx+10:520,m?412:346,a.actors[0].note.includes('无本地持有')?'本地持有已解除':'仍有本地资源持有');if(m)resourceText.setAttribute('font-size','11');
  text(exit,18,m?686:574,lerp(a.actors[index('exit')].opacity,b.actors[index('exit')].opacity,u)>0?'到达教学出口（不是源码队列）':'同一个 Req 可以同时被多个容器引用');if(m)exit.setAttribute('font-size','13');
 };
});
V.lifecycle=(svg,viewport,s)=>board(svg,viewport,'release',m=>150+s.frames.length*94,H=>{
 const {txt,rect,path,text,box,show,wire,token,move}=H;
 const heads=['控制名单','本级 sender','本地资源'].map(x=>txt(x,'teach-heading'));
 const timeline=path(),cursor=token('release-progress','当前',48,'teach-token teach-color-3');
 const rows=s.frames.map((f,i)=>({time:txt(String(i+1),'teach-small'),control:txt(i===0?'名单含 R0':i===1?'交给本级处理':'','teach-small'),poll:txt('','teach-small'),held:rect('teach-held'),label:txt('','teach-small'),link:path()}));
 const note=txt('纵向是教学步骤，不是实测时间；收尾不等于 D 已可运行','teach-small');
 const sender=s.actors.findIndex(a=>a.id==='sender'),resource=s.actors.findIndex(a=>a.id==='resource');
 return(p,m)=>{
  const xs=m?[32,138,250]:[90,310,540];heads.forEach((a,i)=>text(a,xs[i],35));
  rows.forEach((r,i)=>{const y=96+i*94,f=s.frames[i],active=p>=i,st=f.actors[sender].note,held=f.actors[resource].count>0;
   text(r.time,8,y+21);text(r.control,xs[0],y+21);text(r.poll,xs[1],y+21,st.includes('Transferring')?'仍在途':st.includes('Success')?'本地成功':st.includes('完成')?'已收尾':'等待复查');
   box(r.held,xs[2],y,m?90:140,44);r.held.classList.toggle('is-free',!held);text(r.label,xs[2]+10,y+27,held?'继续保留':'解除持有');
   [r.time,r.control,r.poll,r.held,r.label].forEach(e=>show(e,active?1:.18));
   if(i===1)wire(r.link,`M${xs[0]+(m?60:100)} ${y+30} H${xs[1]-8}`,phase(p,0));
  });
  const y=96+p*94;move(cursor,m?200:430,y+40);wire(timeline,`M20 76 V${96+p*94}`,1);text(note,18,130+s.frames.length*94);if(m)note.setAttribute('font-size','10');
 };
});
})();
