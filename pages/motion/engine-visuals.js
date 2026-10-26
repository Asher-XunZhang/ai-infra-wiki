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
V.scheduling=(svg,viewport,s)=>s.kind==='chunk'?window.ChunkPrefillVisual(svg,viewport,s):board(svg,viewport,'admission',m=>m?910:520,H=>{
  const {txt,rect,text,box,show,token,move,set}=H,S=s.model;
  const title=txt('候选请求 · 深色为命中历史，浅色为新输入','teach-heading'),orderText=txt('','teach-small');
  const reqs=S.requests.map((r,i)=>({r,label:token(r.id,r.id,56,'teach-token teach-color-'+i),pieces:Array.from({length:r.input},(_,j)=>rect(j<r.hit?'teach-read':'teach-contribution teach-color-'+i)),note:txt('','teach-small'),status:txt('','schedule-status teach-small is-unchecked')}));
  const budget=rect(),budgetTitle=txt('本轮预算账本','teach-heading'),budgetNote=txt('KV 预留含命中保护；尚未写入物理 KV','teach-small');
  const meters=['本轮输入','KV 账本','请求行'].map(name=>({title:txt(name,'teach-heading'),track:rect('teach-track'),fill:rect('teach-value'),value:txt('','teach-number')}));
  const check=rect('teach-check'),checkTitle=txt('','teach-heading schedule-check-title'),formula=txt('','teach-number schedule-formula');
  const details=[0,1,2].map(()=>txt('','teach-small schedule-detail'));
  const plan=rect('teach-plan'),planTitle=txt('本轮计划','teach-heading schedule-plan-title');
  const decisions=[0,1].map(()=>txt('','schedule-decision'));
  const reasons=[0,1].map(()=>txt('','teach-small schedule-reason'));
  const members=S.requests.map((r,i)=>{const t=token('plan-'+r.id,r.id,44,'teach-token teach-color-'+i);set(t.r,{width:44,height:26});set(t.t,{x:22,y:18,'font-size':12});return t;});
  const shortReason={kv:'KV 检查未通过',input:'输入追加未通过',rows:'请求行准入未通过'};
  const needOf=r=>r.input-r.hit+r.maxNew+1;
  const list=ids=>ids.length?ids.join('、'):'无';
  function checkFor(event){
    if(!event?.current)return null;
    if(event.kind==='check'||(event.kind==='blocked'&&event.reason!=='rows'))return event;
    const at=s.events.indexOf(event);
    for(let i=at-1;i>=0;i--){const e=s.events[i];if(e.kind==='check'&&e.current===event.current)return e;if(e.kind==='blocked'&&e.reason==='rows')break;}
    return null;
  }
  function charWidth(ch,size){
    if(ch===' ')return size*.34;
    if(/[A-Za-z0-9_.+\/%]/.test(ch))return size*.56;
    if(/[≥→]/.test(ch))return size*.96;
    return size;
  }
  function measure(value,size){let total=0;for(const ch of value)total+=charWidth(ch,size);return total;}
  // SVG text has no automatic wrapping. Split on ASCII words and CJK characters,
  // leaving enough width for the punctuation that follows a line.
  function wrapLines(value,size,maxWidth,limit=3){
    const lines=[];
    for(const paragraph of String(value).split('\n')){
      const units=paragraph.match(/[A-Za-z0-9_.+\/%≥→]+| +|./gu)||[];let current='';
      for(let i=0;i<units.length;i++){
        const unit=units[i];
        if(current&&measure(current+unit,size)>maxWidth){
          lines.push(current.trimEnd());if(lines.length>=limit)return lines;
          current=unit.trimStart();
        }else current+=unit;
      }
      if(current)lines.push(current.trim());
      if(lines.length>=limit)return lines;
    }
    if(!lines.length)lines.push('');
    return lines.slice(0,limit);
  }
  function putLines(nodes,lines,x,y,lineHeight){
    nodes.forEach((node,i)=>{const value=lines[i];if(value){text(node,x,y+i*lineHeight,value);show(node,1);}else{node.textContent='';show(node,0);}});
  }
  function afterOf(event){
    if(!event?.current)return [];
    return event.order.slice(event.order.indexOf(event.current)+1).filter(id=>!event.admitted.includes(id));
  }
  function blockedLine(event){
    const r=S.requests.find(x=>x.id===event.current),after=afterOf(event),tail=after.length?`。\n${after.join('、')} 未检查。`:'。';
    if(event.reason==='rows')return `请求行已满（${event.admitted.length}/${event.config.rows}）；${event.current} 未入计划${tail}`;
    if(event.reason==='kv')return `KV 检查未通过；${event.current} 未入计划${tail}`;
    return `${event.current} 新输入 ${r.input-r.hit} ≥ 剩余 ${event.config.input-event.usedInput}；未入计划${tail}`;
  }
  function stopLine(event,before){
    const after=afterOf(event),tail=after.length?`。\n${after.join('、')} 未检查。`:'。';
    return before?`首条提交后新输入 ${event.usedInput}/${event.config.input} 已超额度${tail}`:`提交后新输入 ${event.usedInput}/${event.config.input} 用完${tail}`;
  }
  return(p,m)=>{
    const index=Math.min(Math.floor(p+1e-8),s.events.length-1),a=s.events[index],b=s.events[Math.min(index+1,s.events.length-1)],u=phase(p,index);
    const gate=s.events.slice(0,index+1).reverse().find(e=>e.kind==='blocked'||e.kind==='stop');
    const gatePos=gate?a.order.indexOf(gate.current):-1;
    text(title,18,28);text(orderText,18,52,'本轮检查顺序：'+a.order.join(' → '));
    reqs.forEach(({r,label,pieces,note,status},i)=>{
      const x=m?16:18,y=(m?62:64)+i*(m?88:94),admitted=a.admitted.includes(r.id),isCurrent=a.kind==='check'&&a.current===r.id,pos=a.order.indexOf(r.id);
      let cls='is-unchecked',copy='未检查 · 等待轮次';
      if(admitted){cls='is-committed';copy='✓ 已提交计划'+(gate?.kind==='stop'&&gate.current===r.id?' · 触发停止':'');}
      else if(gate?.kind==='blocked'&&gate.current===r.id){cls='is-blocked';copy=shortReason[gate.reason];}
      else if(gate&&pos>gatePos){copy=gate.kind==='stop'?'本轮未检查 · 前序已停止':'本轮未检查 · 前序未通过';}
      else if(isCurrent){cls='is-checking';copy='正在检查 · 尚未决定';}
      else if(a.kind==='ready'){copy='本轮未检查 · 未入计划';}
      move(label,x,y);pieces.forEach((cell,j)=>box(cell,x+68+j*26,y+5,23,26));
      text(note,x+68,y+47,`输入 ${r.input} · 命中 ${r.hit} · 新算 ${r.input-r.hit}`);
      text(status,x,y+72,copy);status.setAttribute('class','schedule-status teach-small '+cls);
      status.dataset.status=cls.replace('is-','');label.g.dataset.admitted=String(admitted);label.g.dataset.status=cls.replace('is-','');label.g.dataset.needInput=String(r.input-r.hit);
    });
    const bx=m?16:414,by=m?330:64,bw=m?328:288,bh=m?236:276;
    box(budget,bx,by,bw,bh);text(budgetTitle,bx+12,by+22);
    const values=[lerp(a.usedInput,b.usedInput,u),lerp(a.offset+a.protectedTokens+a.trial,b.offset+b.protectedTokens+b.trial,u),lerp(a.admitted.length,b.admitted.length,u)];
    const limits=[a.config.input,a.config.kv,a.config.rows];
    meters.forEach((meter,i)=>{
      const y=by+(m?38:42)+i*(m?66:74),before=[a.usedInput,a.offset+a.protectedTokens+a.trial,a.admitted.length][i],after=[b.usedInput,b.offset+b.protectedTokens+b.trial,b.admitted.length][i];
      text(meter.title,bx+12,y+12);
      meter.value.setAttribute('text-anchor','end');
      text(meter.value,bx+bw-12,y+13,`${['已用','预留','已用'][i]} ${u>0&&u<1&&before!==after?`${before} → ${after}`:`${u>=1?after:before}`} / ${limits[i]}`);
      box(meter.track,bx+12,y+24,bw-24,10);box(meter.fill,bx+12,y+24,(bw-24)*clamp(values[i]/limits[i]),10);
      meter.fill.dataset.used=values[i].toFixed(3);
    });
    text(budgetNote,bx+12,by+bh-10);
    const display=a.current?a:(a.kind==='ready'?(gate||s.events.slice(0,index).reverse().find(e=>e.kind==='commit')||a):a);
    const rowsBlocked=display?.kind==='blocked'&&display.reason==='rows';
    const isFinal=a.kind==='ready';
    let checkHead='等待下一次检查';
    if(display?.current){
      if(rowsBlocked)checkHead=`${isFinal?'最后准入':'请求行准入'} · ${display.current} 未通过`;
      else if(isFinal)checkHead=`最后检查 · ${display.current} ${display.kind==='blocked'?'未通过':'通过'}`;
      else if(display.kind==='check')checkHead=`${display.current} 的准入检查 · 进行中`;
      else checkHead=`${display.current} 的准入检查 · ${display.kind==='blocked'?'未通过':'通过'}`;
    }
    text(checkTitle,30,(m?578:352)+24,checkHead);
    let formulaText='先排序，再核对请求行、KV 与输入';
    if(display?.current){
      if(rowsBlocked)formulaText=`请求行 ${display.admitted.length} / ${display.config.rows} 已占，需再加 1 行`;
      else{const r=S.requests.find(x=>x.id===display.current),ce=checkFor(display),avail=ce?ce.checkAvailable:display.available;formulaText=`KV：${r.input-r.hit} + ${r.maxNew} + 1 = ${needOf(r)}（可用 ${avail}）`;}
    }
    text(formula,30,(m?578:352)+50,formulaText);
    let lines=[];
    if(!display?.current)lines=['排序只决定检查顺序，不改变预算。','请求行、KV 与输入额度是三道独立条件。'];
    else if(rowsBlocked)lines=[`请求行已占 ${display.admitted.length}/${display.config.rows}，还需 1 行。`,'请求行检查未通过；未进行 KV 计算。',`${list(afterOf(display))} 未检查、未入计划。`];
    else{
      const r=S.requests.find(x=>x.id===display.current),ce=checkFor(display),avail=ce?ce.checkAvailable:display.available,need=needOf(r),extend=r.input-r.hit,after=afterOf(display);
      if(display.kind==='check')lines=[`请求行已通过；KV 要求 ${need} < ${avail}。`,'比较尚未完成；下一步才提交或停止。'];
      else if(display.kind==='commit')lines=[`要求 ${need} < ${avail} 成立。`,'已加入计划；尚未分配物理 KV。','提交不等于执行模型。'];
      else if(display.kind==='stop'){const over=display.usedInput>display.config.input;lines=[`KV 检查通过：${need} < ${avail}。`,`提交后输入 ${display.usedInput}/${display.config.input} ${over?'已超额度':'用完'}，停止追加。`,`${list(after)} 未检查；已提交请求仍会执行。`];}
      else if(display.reason==='kv')lines=[`要求 ${need} < ${avail} 不成立。`,'临时保护已退回；未加入计划。',`${list(after)} 未检查。`];
      else if(display.reason==='input')lines=[`KV 检查通过：${need} < ${avail}。`,`追加输入 ${extend} ≥ 剩余 ${display.config.input-display.usedInput}（已用 ${display.usedInput}/${display.config.input}）。`,`${display.current} 未入计划；${list(after)} 未检查。`];
      else lines=['检查已通过；尚未执行模型。'];
    }
    putLines(details,lines,30,(m?578:352)+78,m?22:20);
    text(planTitle,(m?16:360)+12,(m?734:352)+22);
    members.forEach((t,i)=>{
      const start=a.admitted.includes(S.requests[i].id)?1:0,end=b.admitted.includes(S.requests[i].id)?1:0,amount=lerp(start,end,u);
      const ai=a.admitted.indexOf(S.requests[i].id),bi=b.admitted.indexOf(S.requests[i].id),position=ai<0?(bi<0?i:bi):lerp(ai,bi<0?ai:bi,u);
      move(t,(m?16:360)+112+position*50,(m?734:352)+7,amount);t.g.dataset.committed=amount.toFixed(3);
    });
    let d1,d2;
    if(a.kind==='waiting'){d1='尚未开始准入检查';d2=`等待顺序：${S.requests.map(r=>r.id).join(' → ')}；尚未占用预算。`;}
    else if(a.kind==='sorted'){d1='检查顺序已确定';d2=`${a.order.join(' → ')}；排序本身不占用预算。`;}
    else if(a.kind==='check'){d1=`正在检查 ${a.current}，尚未提交`;d2='通过后才加入本轮计划；提交不等于执行。';}
    else if(a.kind==='commit'){d1=`已提交 ${a.current}，继续检查后续`;d2=`输入 ${a.usedInput}/${a.config.input} · KV 预留 ${a.offset+a.protectedTokens+a.trial}/${a.config.kv} · 请求行 ${a.admitted.length}/${a.config.rows}`;}
    else if(a.kind==='blocked'){d1=`本轮计划：${list(a.admitted)}；已停止追加`;d2=blockedLine(a);}
    else if(a.kind==='stop'){d1=`本轮计划：${list(a.admitted)}；已停止追加`;d2=stopLine(a,a.usedInput>a.config.input);}
    else{d1=`本轮计划：${list(a.admitted)}`;d2=gate?.kind==='blocked'?blockedLine(gate):gate?.kind==='stop'?stopLine(gate,gate.usedInput>gate.config.input):'输入、KV 与请求行均未阻止；模型尚未执行。';}
    const px=m?16:360,py=m?734:352,pw=m?328:342;
    putLines(decisions,wrapLines(d1,15,pw-32,2),px+12,py+48,18);
    const decisionCount=Math.max(1,wrapLines(d1,15,pw-32,2).length);
    putLines(reasons,wrapLines(d2,m?14:12,pw-34,3),px+12,py+48+decisionCount*18+6,m?20:16);
    check.dataset.state=display?.kind||'none';check.dataset.reason=display?.reason||'';check.dataset.request=display?.current||'';
    plan.dataset.admitted=a.admitted.join(',');plan.dataset.remaining=a.waiting.join(',');
    box(check,18,m?578:352,m?324:330,m?144:150);
    box(plan,px,py,pw,m?160:150);
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
  const y=96+p*94;move(cursor,m?200:430,y+40);wire(timeline,`M20 76 V${96+p*94}`,1);text(note,18,130+s.frames.length*94);note.style.fontSize='10px';
 };
});
})();
