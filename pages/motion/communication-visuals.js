/* Communication diagrams encode ownership, arithmetic or causal time as appropriate. */
(()=>{
'use strict';
const {board,phase,lerp}=window.TeachingDrawing;
function route(a,b,t,x){if(t<.2)return{x:lerp(a.x,x,t/.2),y:a.y};if(t<.8)return{x,y:lerp(a.y,b.y,(t-.2)/.6)};return{x:lerp(x,b.x,(t-.8)/.2),y:b.y};}
const html=(tag,cls,value)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(value)e.textContent=value;return e;};
function context(viewport,s){
 const d=s.data,box=html('section','motion-handoff comm-context is-compact');box.setAttribute('aria-label','交接边界与载荷');
 const stream=d.op==='Completion'&&d.mode==='stream',rma=['Put','Get','Atomic'].includes(d.op),pair=!!d.matrix||rma||d.op==='SendRecv'||d.op==='Completion';
 if(stream)box.append(html('strong','','CPU 提交与 GPU stream 依赖'));
 const text=html('p'),label=html('label','','部署示例 '),select=html('select');select.setAttribute('aria-label','通信部署位置');
 for(const [v,t]of [['local','同机、不同进程'],['remote','跨主机、不同进程']]){const o=html('option','',t);o.value=v;select.append(o);}label.append(select);if(!stream)box.append(label);
 const render=()=>{text.textContent=stream?'CPU 提交通信；通信 stream → 完成事件 → 消费者 stream。跨 rank 的载荷交换由 NCCL 推进，图中聚焦本 rank 的消费依赖。':`${pair?'Rank 0 / P0 与 Rank 1 / P1':'Rank 0 / P0、Rank 1 / P1、Rank 2 / P2'} ：${select.value==='local'?'同机、不同进程。':(pair?'P0 在主机 A，P1 在主机 B。':'P0 在主机 A，P1、P2 在主机 B。')+'跨主机交接需通过机间路径。'}${rma?'窗口属于各自进程；MPI 示例不限定 CPU 或 GPU 存储。':d.op==='Barrier'?'屏障表示进程会合，不指定设备张量传输。':'本例各用一张 GPU。'}`;};select.addEventListener('change',render);render();box.append(text);
 const payload=html('p','comm-payload',d.op==='Barrier'?'载荷：屏障同步信息；不搬运用户张量。':d.op==='Completion'?'载荷：KV 字节或通信结果；控制信息与完成事件分别表示。':d.op==='Get'?'发起：Rank 0 → Rank 1 的读取请求；载荷：Rank 1 窗口 → Rank 0 本地缓冲区。':d.op==='Put'?'载荷：Rank 0 本地缓冲区 → Rank 1 窗口；窗口地址和访问许可需预先建立。':d.op==='Atomic'?'载荷：更新参数 → 远端窗口；CAS 还返回旧值。':`载荷：图中编号数据块${d.math.length?'或同一下标的数值贡献':''}；直接按语义呈现交接。`);box.append(payload);
 const nav=html('nav');for(const [t,u]of [['数学分片与并行','parallel.html'],['SGLang 通信与传输','../sglang/communication/index.html'],['完成条件说明','#why-ready']]){const a=html('a','',t+' ↗');a.href=u;nav.append(a);}box.append(nav);viewport.prepend(box);return box;
}
function between(a,b,t){return{x:lerp(a.x,b.x,t),y:lerp(a.y,b.y,t)};}
// The same rank columns show the before and after states. Copies travel directly
// between their owners; a curve separates opposing directions, not another stage.
function directTransfer(svg,viewport,s){
 const d=s.data,exchange=['AllToAll','Neighbor'].includes(d.op),maxIn=Math.max(...d.input.map(a=>a.length)),maxOut=Math.max(...d.output.map(a=>a.length));
 const inputHeight=70+maxIn*40,outputTop=60+inputHeight+110,outputHeight=80+maxOut*40,height=outputTop+outputHeight+68;
 return board(svg,viewport,'communication-direct',()=>height,H=>{
  const {txt,rect,path,wire,text,box,show,token,move,set}=H;
  const color=(r,i)=>d.op==='Scatter'?d.flows.find(f=>f.src===r&&f.si===i)?.dst??r:r;
  const panels=[0,1,2].map(()=>({input:rect(),output:rect()}));
  const names=[0,1,2].map(()=>({input:txt('','teach-heading'),output:txt('','teach-heading')}));
  const empty=[0,1,2].map(()=>({input:txt('无输入','teach-small'),output:txt('无输出','teach-small')}));
  const before=txt('开始时','teach-heading'),after=txt('完成后','teach-heading');
  const lines=d.flows.map(f=>({...f,line:path('teach-link comm-direct-link')}));
  const originals=d.input.flatMap((a,r)=>a.map((v,i)=>({r,i,t:token('comm-input-'+r+'-'+i,String(v),36,'teach-token teach-color-'+color(r,i))})));
  const slots=d.output.flatMap((a,r)=>a.map((v,i)=>({r,i,t:token('comm-slot-'+r+'-'+i,'…',36,'teach-token teach-empty')})));
  const copies=d.flows.map(f=>({...f,t:token('comm-copy-'+f.id,String(f.value),36,'teach-token teach-color-'+color(f.src,f.si))}));
  for(const a of [...originals,...slots,...copies]){set(a.t.r,{height:28});set(a.t.t,{y:19});}
  const legend=txt('','teach-small'),summary=txt('','teach-heading');
  const behavior={Broadcast:'root 的同一份数据 → 每个 rank 一份',Scatter:'root 的不同分片 → 各自的目标 rank',Gather:'各 rank 的片段 → root 按顺序收齐',AllGather:'各 rank 的片段 → 每个 rank 都收齐',AllToAll:'各 rank 按目标发送 → 接收方按来源排列',Neighbor:'只交换相邻 rank 的数据：0 ↔ 1 ↔ 2'};
  return(p,m)=>{const W=m?360:720,cw=(W-36)/3,cx=r=>18+r*cw,x=r=>cx(r)+cw/2-18,t=phase(p,1,3);
   text(before,18,30);text(after,18,outputTop-18);
   panels.forEach((b,r)=>{box(b.input,cx(r),60,cw-8,inputHeight);box(b.output,cx(r),outputTop,cw-8,outputHeight);
    const name=`Rank ${r}${['Broadcast','Scatter','Gather'].includes(d.op)&&r===d.root?' ★':''}`;
    text(names[r].input,cx(r)+8,90,name);text(names[r].output,cx(r)+8,outputTop+outputHeight-18,name);
    text(empty[r].input,cx(r)+8,145);show(empty[r].input,d.input[r].length?0:1);text(empty[r].output,cx(r)+8,outputTop+45);show(empty[r].output,d.output[r].length?0:1);
   });
   const source=f=>({x:x(f.src),y:116+f.si*40}),target=f=>({x:x(f.dst),y:outputTop+24+f.di*40});
   const bend=f=>{const limit=(target(f).y-source(f).y)*.46,offset=Math.sign(f.src-f.dst)*46+(f.di-f.si)*24;return Math.max(-limit,Math.min(limit,offset));};
   originals.forEach(a=>move(a.t,x(a.r),116+a.i*40,p>1&&p<3?.4:1));slots.forEach(a=>move(a.t,x(a.r),outputTop+24+a.i*40));
   lines.forEach(f=>{const a=source(f),b=target(f);wire(f.line,`M${a.x+18} ${a.y+14} Q${(a.x+b.x)/2+18} ${(a.y+b.y)/2+14+bend(f)} ${b.x+18} ${b.y+14}`,1);show(f.line,f.src!==f.dst&&p>=1&&p<3?.24:0);});
   copies.forEach(f=>{const a=source(f),b=target(f),local=f.src===f.dst;
    const q=local?b:{x:lerp(a.x,b.x,t),y:lerp(a.y,b.y,t)+2*(1-t)*t*bend(f)};
    move(f.t,q.x,q.y,local?phase(p,0):p>1?1:0);
    set(f.t.g,{'data-arrived':String(p>=(local?1:3)),'data-in-flight':String(!local&&p>1&&p<3),'data-source':f.src,'data-target':f.dst});
   });
   text(legend,18,height-42,exchange?(m?'i→j：来源 i、目标 j；b = 第二个元素':'数据标记 i→j 表示来源 rank i、目标 rank j；b 是发给同一目标的第二个元素。'):['Broadcast','Scatter','Gather'].includes(d.op)?'★ 标记 root；颜色跟随同一份数据。':'颜色表示数据来源；收集保持 rank 顺序。');
   text(summary,18,height-10,m?{Broadcast:'同一份数据复制到每个 rank',Scatter:'不同分片交给对应 rank',Gather:'各份数据按顺序收集到 root',AllGather:'每个 rank 都得到全部片段',AllToAll:'按目标发送，按来源排列',Neighbor:'沿 0 ↔ 1 ↔ 2 交换数据'}[d.op]:behavior[d.op]);
  };
 });
}
function pointToPoint(svg,viewport,s){const d=s.data,get=d.op==='Get',put=d.op==='Put',src=get?1:0,dst=1-src;return board(svg,viewport,'communication-endpoints',m=>m?880:640,H=>{
 const {txt,rect,path,wire,text,box,token,move,show,set}=H;
 const panels=[rect(),rect()],names=[0,1].map(r=>txt('Rank '+r+' · 进程 P'+r,'teach-heading'));
 const labels=[txt('','teach-small'),txt('','teach-small')],states=[txt('','teach-small'),txt('','teach-small')];
 const originals=[7,8].map((v,i)=>token('comm-input-'+src+'-'+i,String(v),52,'teach-token teach-color-'+src));
 const slots=[0,1].map(i=>token('comm-slot-'+dst+'-'+i,'…',52,'teach-token teach-empty'));
 const copies=[7,8].map((v,i)=>token('comm-copy-'+i,String(v),52,'teach-token teach-color-'+src));
 const payload=path(),request=path('teach-link comm-control-link'),direction=txt('','teach-small'),control=txt('','teach-small'),controlDetail=txt('','teach-small'),requestLabel=txt('','teach-small'),payloadLabel=txt('','teach-small'),result=txt('','teach-heading');
 return(p,m)=>{const W=m?360:720,pos=r=>m?{x:40,y:80+r*370}:{x:46+r*360,y:80};
  for(let r=0;r<2;r++){const q=pos(r);box(panels[r],q.x-20,q.y-35,m?320:320,230);text(names[r],q.x,q.y);text(labels[r],q.x,q.y+42,r===src?'源缓冲区 · 读取后仍保留':d.op==='Put'?'目标窗口 · 位置 0、1':'目标缓冲区 · 位置 0、1');text(states[r],q.x,q.y+158,r===src?'原件 [7,8]':d.unmatched?'Recv 缺席':p<3?'内容尚未齐备':p<4?'载荷写好；等待完成条件':'本例使用条件已满足');}
  const a=pos(src),b=pos(dst);originals.forEach((t,i)=>move(t,a.x+i*70,a.y+70));slots.forEach((t,i)=>move(t,b.x+i*70,b.y+70));
  const reqA=pos(0),reqB=pos(1),direct=!m&&(get||d.op==='Put');
  // On a narrow screen the two participants are stacked: use a straight
  // vertical request/payload arrow instead of a U-shaped desktop route.
  const mobileRequest=get&&m?`M${reqA.x+150} ${reqA.y+230} V${reqB.y}`:m?`M${reqA.x+220} ${reqA.y+50} V${reqB.y+50}`:`M${reqA.x+220} ${reqA.y+20} H${reqB.x}`;
  wire(request,direct?`M${reqA.x+55} ${reqA.y+20} H${reqB.x}`:mobileRequest,get?phase(p,0):0);
  const mobilePayload=put&&m?`M${a.x+200} ${a.y+230} V${b.y}`:get&&m?`M${a.x+250} ${a.y} V${b.y+230}`:null;
  const pathD=put&&m?mobilePayload:get&&m?mobilePayload:direct?`M${a.x+150} ${a.y+70} H${b.x+20}`:m?`M${a.x+55} ${a.y+87} V${b.y+87} H${b.x+55}`:`M${a.x+55} ${a.y+112} V${a.y+220} H${b.x+55} V${b.y+112}`;
  wire(payload,pathD,d.unmatched?0:phase(p,1));
  const midX=(a.x+b.x)/2+55,midY=a.y+62;
  if(m&&put){text(payloadLabel,207,380,'写入载荷 [7,8]');}else if(m&&get){text(requestLabel,120,380,'读取请求');text(payloadLabel,228,404,'返回 [7,8]');}else if(put){text(payloadLabel,midX,midY,'写入载荷 [7,8]');}else if(get){text(requestLabel,midX,reqA.y+12,'读取请求');text(payloadLabel,midX,a.y+62,'返回 [7,8]');}
  [requestLabel,payloadLabel].forEach(t=>show(t,phase(p,1)));
  copies.forEach((t,i)=>{const u=phase(p,1,3);let x,y;if(m){const point=route({x:a.x+i*70,y:a.y+70},{x:b.x+i*70,y:b.y+70},u,220+i*62);x=point.x;y=point.y;}else{x=lerp(a.x+i*70,b.x+i*70,u);y=a.y+70+Math.sin(u*Math.PI)*180;}move(t,x,y,!d.unmatched&&p>1?1:0);set(t.g,{'data-arrived':String(p>=3&&!d.unmatched),'data-in-flight':String(p>1&&p<3&&!d.unmatched)});});
  text(control,18,m?430:390,m?(get?'读请求：0 → 1':d.op==='Put'?'写入：0 → 1':'Send 0 → Recv 1'):get?'读请求 0 → 1；返回数据 1 → 0':d.op==='Put'?'写入载荷 0 → 1；目标不逐条提交 Recv':'Send 0 → Recv 1；两方必须匹配');text(controlDetail,18,378,get?'返回载荷：1 → 0':d.op==='Put'?'无需逐条匹配 Recv':'双方必须匹配');show(controlDetail,0);show(control,m?0:1);text(direction,18,m?765:444,'箭头就地说明发起方与载荷方向；不规定协议阶段。');
  text(result,18,m?820:506,d.unmatched?'接收尚未完成':p<3?'载荷尚未完整到达':p<4?'继续确认完成与消费依赖':'[7,8] 可供本例消费者读取');
 };
});}
function reductions(svg,viewport,s){const d=s.data,scan=['Scan','Exscan'].includes(d.op);return board(svg,viewport,'communication-reduction',m=>m?490:1120,H=>{
 const {txt,rect,text,box,show,token,move,set}=H;
 const heads=[txt('① 输入 · 每一行属于一个 rank','teach-heading'),txt(scan?'② 前缀范围：只取阶梯内的贡献':'② 相同下标对齐，再执行归约','teach-heading'),txt('③ 各 rank 的输出','teach-heading')];
 const originals=d.input.flatMap((a,r)=>a.map((v,i)=>({r,i,t:token('comm-input-'+r+'-'+i,String(v),48,'teach-token teach-color-'+r)})));
 const inputLabels=[0,1,2].map(r=>txt('Rank '+r,'teach-small'));
 const rows=d.math.map((line,i)=>({line,i,label:txt(line.label,'teach-small'),cells:[0,1,2].map(r=>({bg:rect(line.sources.some(a=>a.src===r)?'teach-graph-slot':'teach-mask'),tag:txt('','teach-small')})),copies:line.sources.map(a=>({...a,t:token('comm-contribution-'+i+'-'+a.src,String(d.input[a.src][a.si]),48,'teach-token teach-color-'+a.src)})),op:txt('','teach-small'),result:token('comm-math-'+i,line.value===null?'未定义':String(line.value),50,'teach-token teach-color-3')}));
 const columns=[0,1,2].map(r=>txt('来自 '+r,'teach-small'));
 const outputLabels=[0,1,2].map(r=>txt('Rank '+r,'teach-small'));
 const results=d.math.flatMap((line,i)=>line.destinations.filter(()=>line.value!==null).map(dest=>({i,...dest,t:token('comm-result-'+dest.dst+'-'+dest.di,line.value===null?'未定义':String(line.value),50,'teach-token teach-color-3')})));
 const outputPanels=[0,1,2].map(()=>rect());
 const empties=d.output.map((a,r)=>({r,out:a,t:txt(a[0]===null?'未定义':a.length?'':'无归约输出','teach-small')}));
 const equals=d.math.map(()=>txt('＝','teach-number'));
 const note=txt('','teach-small'),formula=txt('','teach-number'),status=txt('','teach-heading');
 return(p,m)=>{const mobile=!!m;
  const sx=mobile?66:200,cw=mobile?38:98,mx=mobile?58:160,mc=mobile?40:105,rx=mobile?204:540,rowY=i=>mobile?162+i*56:374+i*86;
  // Keep the first heading below the top edge: camera type compensation can
  // make its ascender taller than the 20-unit baseline on wide, short stages.
  heads.forEach((e,i)=>{text(e,mobile?14:18,[mobile?32:38,mobile?136:256,mobile?334:320][i]);show(e,i===0?1:i===1?(p>=1?1:0):(p>=2?1:0));});
  inputLabels.forEach((e,r)=>text(e,mobile?14:20,mobile?52+r*30:94+r*65));
  originals.forEach(a=>{if(mobile){set(a.t.r,{width:30,height:26});text(a.t.t,15,18);move(a.t,sx+a.i*cw,36+a.r*30);}else{set(a.t.r,{width:48,height:34});text(a.t.t,24,22);move(a.t,sx+a.i*cw,72+a.r*65);}});
  columns.forEach((e,r)=>{if(mobile){text(e,mx+r*mc+10,151,'R'+r);e.setAttribute('text-anchor','middle');}else{text(e,mx+r*mc,348,'来自 '+r);e.removeAttribute('text-anchor');}});
  rows.forEach((row,i)=>{const y=rowY(i);
   text(row.label,mobile?14:18,mobile?y+19:y+24);
   row.cells.forEach((c,r)=>{if(mobile){box(c.bg,mx+r*mc-2,y-2,36,30);text(c.tag,mx+r*mc+10,y+21,'—');}else{box(c.bg,mx+r*mc-3,y-3,54,44);text(c.tag,mx+r*mc+8,y+24,'—');}show(c.tag,row.line.sources.some(a=>a.src===r)?0:1);});
   row.copies.forEach(a=>{const t=phase(p,0);
    if(mobile){set(a.t.r,{width:32,height:26});text(a.t.t,16,18);move(a.t,lerp(sx+a.si*cw,mx+a.src*mc+2,t),lerp(36+a.src*30,y+2,t),t);}
    else{set(a.t.r,{width:48,height:34});text(a.t.t,24,22);move(a.t,lerp(sx+a.si*cw,mx+a.src*mc,t),lerp(72+a.src*65,y,t),t);}
   });
   if(mobile){
    show(equals[i],p>=2?1:0);text(equals[i],176,y+20);
    text(row.op,58,y+46,row.line.value===null?'空前缀':d.reduce==='max'?'MAX：取最大':'SUM：'+row.line.sources.map(a=>d.input[a.src][a.si]).join(' + '));show(row.op,phase(p,1));
    set(row.result.r,{width:44,height:26});text(row.result.t,22,18);move(row.result,204,y+2,p>=2?1:0);
   }else{
    show(equals[i],0);
    text(row.op,mx,y+64,row.line.value===null?'空前缀':d.reduce==='max'?'MAX：取最大':'SUM：'+row.line.sources.map(a=>d.input[a.src][a.si]).join(' + '));show(row.op,phase(p,1));
    set(row.result.r,{width:50,height:34});text(row.result.t,25,22);move(row.result,rx,y,p>=2?1:0);
   }
  });
  const W=mobile?360:720,ow=(W-36)/3,ox=r=>18+r*ow+ow/2-25;
  outputPanels.forEach((b,r)=>{if(mobile)box(b,12+r*115,344,104,100);else box(b,18+r*ow,706,ow-8,272);b.parentNode.insertBefore(b,heads[0]);});
  outputLabels.forEach((e,r)=>text(e,mobile?20+r*115:26+r*ow,mobile?438:958,`Rank ${r}${d.op==='Reduce'&&r===d.root?' ★':''}`));
  empties.forEach(a=>text(a.t,mobile?20+a.r*115:26+a.r*ow,mobile?395:762,a.out[0]===null?'未定义':a.out.length?'':'无归约输出'));
  results.forEach(a=>{
   if(mobile){
    // Lift a finished value into the free lane above its own row before moving
    // it sideways. The old direct diagonal crossed the row's contribution cell.
    const from={x:204,y:rowY(a.i)+2},lift={x:204,y:rowY(a.i)-22},across={x:12+a.dst*115+38,y:rowY(a.i)-22},to={x:12+a.dst*115+38,y:368+a.di*22},u=phase(p,2,3);
    const point=u<.25?between(from,lift,phase(p,2,2.25)):u<.6?between(lift,across,phase(p,2.25,2.6)):between(across,to,phase(p,2.6,3));
    move(a.t,point.x,point.y,p>2?1:0);
    set(a.t.r,{width:42,height:20});text(a.t.t,21,15);
   }else{
    const from={x:rx,y:374+a.i*86},gutter={x:W-56,y:from.y},below={x:gutter.x,y:668+a.i*44},fan={x:ox(a.dst),y:below.y},to={x:ox(a.dst),y:744+a.di*52};
    const point=p<2.15?between(from,gutter,phase(p,2,2.15)):p<2.5?between(gutter,below,phase(p,2.15,2.5)):p<2.8?between(below,fan,phase(p,2.5,2.8)):between(fan,to,phase(p,2.8,3));
    move(a.t,point.x,point.y,p>2?1:0);
    set(a.t.r,{width:50,height:34});text(a.t.t,25,22);
   }
   set(a.t.g,{'data-arrived':String(p>=3),'data-in-flight':String(p>2&&p<3),'data-target':a.dst});
  });
  text(note,18,mobile?386:1058,scan?'灰格不参与；按 rank 顺序取前缀。':'此处是算式，不是一台中央设备。');show(note,mobile?0:1);
  text(formula,mobile?14:18,mobile?464:1023,d.op==='ReduceScatter'?'归约整体 → 按 rank 分片':d.op==='Reduce'?'归约整体 → 仅 root':d.op==='AllReduce'?'归约整体 → 每个 rank 一份':d.op==='Exscan'?'Rank 0：MPI 接收结果未定义':'每个 rank 取得自己的前缀和');
  text(status,mobile?14:18,mobile?480:1100,p<2?'状态：归约尚未完成':p<3?(mobile?'各份结果并发交给 rank':'状态：各份结果并发交到指定参与者'):(mobile?'核对各 rank 输出与算式':'核对每个 rank 的输出与对应算式'));
 };
});}

function synchronization(svg,viewport,s){const d=s.data,barrier=d.op==='Barrier';return board(svg,viewport,'communication-events',()=>690,H=>{
 const {txt,rect,path,wire,text,box,token,move,show}=H;
 const heading=txt(barrier?'各 rank 何时到达屏障？':'控制返回与数据可用是两条时间线','teach-heading');
 const labels=(barrier?['Rank 0','Rank 1','Rank 2']:d.mode==='stream'?['CPU 提交','通信 stream','消费 stream']:['控制通知','KV 载荷','消费者']).map(v=>txt(v,'teach-small'));
 const tracks=labels.map(()=>({bg:rect('teach-track'),fill:rect('teach-value'),wait:rect('teach-mask'),word:txt('','teach-small')}));
 const actors=labels.map((_,i)=>token('comm-event-'+i,barrier?'R'+i:i===0?(d.mode==='stream'?'提交':'通知'):i===1?'数据':'计算',48,'teach-token teach-color-'+i));
 const fence=path(),gate=txt('','teach-small'),lines=[txt('','teach-heading'),txt('','teach-small'),txt('','teach-small')];
 return(p,m)=>{const W=m?360:720,x=96,length=W-122,at=t=>x+length*t/4;
  text(heading,18,30,m?(barrier?'Barrier：等全组进入':'完成：什么时候能读？'):(barrier?'各 rank 何时到达屏障？':'控制返回与数据可用是两条时间线'));
  labels.forEach((l,i)=>{const y=125+i*135;text(l,18,y-26);box(tracks[i].bg,x,y,length,8);});
  wire(fence,`M${at(3)} 85 V447`,barrier&&d.missing?0:1);text(gate,barrier&&d.missing?18:at(3)-(m?75:40),66,barrier?(d.missing?'Rank 2 未进入：没有满足返回条件':'全组进入的最早位置'):'相关传输完成');
  tracks.forEach((t,i)=>{const y=125+i*135;
   let progress,word,start=0,end=0;
   if(barrier){const arrival=i+1,arrived=p>=arrival&&!(d.missing&&i===2),released=!d.missing&&p>3;progress=Math.min(p,arrival);if(d.missing&&i===2)progress=0;if(released)progress=lerp(arrival,4,phase(p,3));start=arrival;end=d.missing&&i===2?start:Math.max(start,Math.min(p,d.missing?4:3));word=released?'可返回':arrived?(d.missing?'等待 Rank 2 进入':p>=3?'全组已进入 · 可返回':'已进入 · 等待其余成员'):d.missing&&i===2?'未进入':'到达前';actors[i].g.dataset.released=String(!d.missing&&p>=3);}
   else{progress=i===0?Math.min(p,1):i===1?Math.max(0,Math.min(3,p)-1):p>3?lerp(0,4,phase(p,3)):0;word=i===0?(p>=1?'提交返回 / 通知已到':'提交前'):i===1?(p>=3?'载荷完成':p>1?'写入中':'等待'):p>=4?'可读并接续计算':p>=3?'依赖确认中':'等待完成依赖';start=i===2?0:0;end=i===2?Math.min(p,3):0;}
   box(t.fill,x,y,Math.max(0,at(progress)-x),8);box(t.wait,at(start),y-8,Math.max(0,at(end)-at(start)),24);move(actors[i],at(progress)-24,y-45);text(t.word,x,y+46,word);
  });
  text(lines[0],18,540,barrier?(d.missing?'缺席时：屏障不会成功返回':p<3?'仍有成员未进入，先到者等待':'全部进入后才允许各自返回'):p<3?'通知已到，数据仍可能不可用':p<4?'传输完成，继续满足消费依赖':'本例消费者的使用条件已满足');
  text(lines[1],18,581,barrier?'Barrier 不携带用户张量。':'横轴是事件先后，填条不是实测带宽。');
  text(lines[2],18,618,barrier?'它不自动等待无关的设备工作。':d.mode==='stream'?'跨 stream：事件记录 + 等待建立依赖。':'控制信息、载荷、身份匹配分别确认。');
 };
});}
function atomic(svg,viewport,s){const d=s.data,cas=d.atomic==='cas';return board(svg,viewport,'communication-atomic',()=>680,H=>{
 const {txt,rect,path,wire,text,box,token,move,show}=H;
 const heads=[txt('Rank 0 · 发起方','teach-heading'),txt('Rank 1 · 暴露的窗口','teach-heading')],panels=[rect(),rect()];
 const command=token('comm-atomic-command',cas?`==${d.expected} → 9`:'+ 2',110,'teach-token teach-color-0'),value=token('comm-atomic-value','5',70,'teach-token teach-color-1'),old=token('comm-atomic-return','旧值 5',100,'teach-token teach-color-1');
 const line=path(),math=txt('','teach-number'),out=txt('','teach-heading'),note=txt('一次操作针对一个整数位置','teach-small');
 return(p,m)=>{const W=m?360:720,x=m?40:80,tx=m?190:450;panels.forEach((e,i)=>{box(e,i?tx-12:x-12,90,i?W-tx-12:140,310);text(heads[i],i?tx-12:x-12,61);});move(value,tx,230);value.t.textContent=p>=3?String(d.output[1][0]):'5';
  const t=phase(p,0,2);move(command,lerp(x,tx-12,t),lerp(130,170,t));wire(line,`M${x+60} 340 H${tx+35}`,phase(p,0));
  text(math,18,460,cas?`比较：5 ${d.expected===5?'=':'≠'} ${d.expected} → ${d.expected===5?'写入 9':'保留 5'}`:'目标值：5 + 2 = 7');show(math,p>=2?1:0);
  move(old,lerp(tx,x,phase(p,2)),lerp(280,300,phase(p,2)),cas&&p>2?1:0);
  text(out,18,530,p<3?'更新尚未确认完成':cas?(d.expected===5?'比较成功：写入 9；发起方得到旧值 5':'比较失败：目标保留 5；发起方得到旧值 5'):'目标更新完成；不返回旧值');text(note,18,578);text(heads[1],m?166:tx-12,61,m?'Rank 1 · 窗口':'Rank 1 · 暴露的窗口');
 };
});}
function matrixTransfer(svg,viewport,s){const d=s.data,sum=d.op==='AllReduce';return board(svg,viewport,'communication-matrix',()=>840,H=>{
 const {txt,rect,text,box,show,token,move,set}=H;
 const panels=[0,1].map(()=>({before:rect(),after:rect()})),names=[0,1].map(r=>[txt('Rank '+r+(d.root===r&&['Broadcast','Scatter'].includes(d.op)?' · root':''),'teach-heading'),txt('Rank '+r,'teach-heading')]);
 const title=txt('开始时','teach-heading'),after=txt('完成后','teach-heading'),shape=[0,1].map(()=>[txt('','teach-small'),txt('','teach-small')]),equation=txt('','teach-heading'),note=txt('','teach-small');
 const color=(r,i)=>d.op==='Scatter'?Math.floor(i/4):r;
 const cell=(id,value,c)=>{const t=token(id,String(value),36,'teach-token teach-color-'+c);set(t.r,{height:30});set(t.t,{y:20});return t;};
 const originals=d.input.flatMap((a,r)=>a.map((v,i)=>({r,i,t:cell('matrix-input-'+r+'-'+i,v,color(r,i))})));
 const copies=(sum?[0,1].flatMap(src=>d.input[src].map((value,i)=>({src,si:i,dst:1-src,di:i,value,id:src+'-'+i}))):d.flows).map(f=>({...f,t:cell('matrix-copy-'+f.id,f.value,color(f.src,f.si))}));
 const results=d.output.flatMap((a,r)=>a.map((v,i)=>({r,i,v,t:cell('matrix-result-'+r+'-'+i,v,sum?2:d.op==='AllGather'?Math.floor(i/4):d.op==='Scatter'?r:0)})));
 return(p,m)=>{const W=m?360:720,sx=m?30:80,tx=m?228:500,lane=r=>24+r*430,xy=(r,i,output)=>({x:(output?tx:sx)+Math.floor(i/4)*42,y:lane(r)+88+(i%4)*36}),t=phase(p,1,3),calc=phase(p,3),index=Math.min(7,Math.floor(calc*8));
  show(title,0);show(after,0);
  panels.forEach((v,r)=>{box(v.before,16,lane(r),W-32,270);show(v.after,0);text(names[r][0],30,lane(r)+30);text(names[r][1],tx,lane(r)+62,'输出');text(shape[r][0],sx,lane(r)+250,d.inputColumns[r]?'输入 4×'+d.inputColumns[r]:'无输入');text(shape[r][1],tx,lane(r)+250,'输出 4×'+d.outputColumns[r]);});
  // Inputs stay on the left and outputs on the right of the same rank lane.
  // Opposing transfers use different curves, keeping entire matrix fragments legible.
  originals.forEach(a=>{const q=xy(a.r,a.i,false);move(a.t,q.x,q.y);a.t.r.classList.toggle('comm-matrix-active',sum&&p>=3&&p<4&&a.i===index);});
  copies.forEach(a=>{const q=xy(a.src,a.si,false),z=xy(a.dst,a.di,true),local=a.src===a.dst,u=local?phase(p,0):t,bend=local?0:Math.sign(a.src-a.dst)*(m?88:120)*Math.sin(Math.PI*u);move(a.t,lerp(q.x,z.x,u)+bend,lerp(q.y,z.y,u),p>1&&(!sum||p<3)?1:0);set(a.t.g,{'data-in-flight':String(!local&&p>1&&p<3),'data-arrived':String(p>=3),'data-source':a.src,'data-target':a.dst});});
  results.forEach(a=>{const q=xy(a.r,a.i,true),done=!sum?p>=3:calc*8>=a.i+1;move(a.t,q.x,q.y,p>=3?1:0);a.t.t.textContent=done?String(a.v):'…';a.t.r.classList.toggle('comm-matrix-active',sum&&p>=3&&p<4&&a.i===index);set(a.t.g,{'data-complete':String(done)});});
  text(equation,18,790,sum?(p<3?'相同坐标的数值相加':`${d.input[0][index]} + ${d.input[1][index]} = ${d.output[0][index]}`):{Broadcast:'完整矩阵 → 各方一份',Scatter:'不同列 → 各自的 rank',AllGather:'两列 → 每个 rank 都收齐'}[d.op]);
  text(note,18,824,sum?'同坐标相加；矩阵形状保持不变。':'本例按列组织元素；颜色跟随来源。');

 };
});}
window.MechanismVisuals['f-communication']=(svg,viewport,s)=>{
 const panel=context(viewport,s),v=s.data.matrix?matrixTransfer(svg,viewport,s):['Barrier','Completion'].includes(s.op)?synchronization(svg,viewport,s):s.op==='Atomic'?atomic(svg,viewport,s):['Put','Get','SendRecv'].includes(s.op)?pointToPoint(svg,viewport,s):s.data.math.length?reductions(svg,viewport,s):directTransfer(svg,viewport,s);
 return {paint:v.paint,destroy(){v.destroy();panel.remove();}};
};
})();
