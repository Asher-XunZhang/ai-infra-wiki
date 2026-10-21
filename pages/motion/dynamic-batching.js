/* Request aggregation before execution. Pure snapshots, not timer callbacks.
 * All arrivals at a decision time are collected before testing capacity/deadline. */
(function(root){
'use strict';
function simulate(delay, arrivals=[{id:'A',at:0},{id:'B',at:2},{id:'C',at:4}], capacity=3){
 if(!Number.isFinite(delay)||delay<0||!Number.isInteger(capacity)||capacity<1)throw new Error('Invalid batch policy');
 if(arrivals.some(r=>!Number.isFinite(r.at)||r.at<0)||new Set(arrivals.map(r=>r.id)).size!==arrivals.length)throw new Error('Invalid arrivals');
 const pending=arrivals.map(r=>({...r})).sort((a,b)=>a.at-b.at),frames=[];
 let time=0,queue=[],batches=[],arrived=[];
 const save=(title,caption,event='decision')=>frames.push({title,caption,event,time,queue:queue.map(r=>r.id),arrived:[...arrived],deadline:queue.length?queue[0].at+delay:null,batches:batches.map(b=>({...b,ids:[...b.ids],waits:[...b.waits]}))});
 save('先看同一组到达请求','A 在 0、B 在 2、C 在 4 到达；每批最多 3 条。每张卡代表一条完整请求，不是 token。','start');
 while(pending.length||queue.length){
  const deadline=queue.length?queue[0].at+delay:Infinity;
  const next=Math.min(pending[0]?.at??Infinity,deadline);
  if(next>time){time=next;save('时间向前，已到达的请求继续等待','等候上限从队列中最早到达的请求开始算；后来到达的请求不会重置它。','clock');}
  const now=[];while(pending.length&&pending[0].at<=time)now.push(pending.shift());
  if(now.length){queue.push(...now);arrived.push(...now.map(r=>r.id));save(now.map(r=>r.id).join('、')+' 到达等待区',`已到达 ${queue.map(r=>r.id).join('、')}。${queue.length>=capacity?'已凑满，可以发出。':delay===0?'不额外等待，发出当前可用请求。':time>=queue[0].at+delay?'等待已到期，可以发出。':'还没凑满，且尚未到期，继续等。'}`,'arrival');}
  while(queue.length&&(queue.length>=capacity||time>=queue[0].at+delay)){
   const reason=queue.length>=capacity?'凑满':delay===0?'不额外等待':'等待到期';
   const members=queue.slice(0,capacity),ids=members.map(r=>r.id);
   save(reason+' → 可以发出',`${ids.join(' + ')} 将一起发往执行端；${reason==='凑满'?'达到容量上限，不必等到截止时刻。':reason==='等待到期'?'即使不足 3 条，也不能无限等下去。':'本例此刻只有这些请求可用。'}`);
   queue=queue.slice(members.length);batches.push({ids,time,reason,waits:members.map(r=>time-r.at)});
   save('第 '+batches.length+' 批已发出',`${ids.join(' + ')} 在 t=${time} 发出；各自等了 ${members.map(r=>r.id+'：'+(time-r.at)).join('，')} 个教学时间单位。发出不等于执行完成。`,'dispatch');
  }
 }
 save('保留组批结果，比较谁等了多久','同一组到达请求，只改变允许额外等待的上限。批次更大不保证更快；这里没有模拟执行耗时或计算吞吐。','summary');
 return frames;
}
const definitions=[{delay:5,name:'动态组批 · 最多等 5',default:true},{delay:3,name:'动态组批 · 最多等 3'},{delay:0,name:'动态组批 · 不额外等待'}];
const scenarios=definitions.map(d=>({...d,kind:'dynamic-batching',group:'① 执行前：什么时候发出一批？',title:'Dynamic Batching（动态组批）：凑满，还是等到期？',intro:`用三条图片分类请求拆开组批过程：A、B、C 各带一张同规格图片，分别在 0、2、4 到达。每批最多 3 条，最多额外等 ${d.delay} 个教学时间单位。先猜第一批会有哪些成员，再播放。`,boundary:'这里采用无状态、单次调用的请求组批例子。请求彼此兼容，执行端在每次决策时可接收；不模拟执行时长。t 是假设的到达时间，不是毫秒、播放秒数或生成轮次。参考 Triton 的动态组批规则，不代表 SGLang 的默认策略。',frames:simulate(d.delay)}));
// Waiting is visible without making every clock update a full teaching pause.
scenarios.forEach(s=>{s.stepDurations=s.frames.slice(1).map(f=>({clock:.65,arrival:.8,dispatch:1.2,summary:.7})[f.event]||.9);});
if(typeof module==='object'&&module.exports){module.exports={simulate,scenarios};return;}
const old=root.MechanismScenes['f-batching'];old.forEach(s=>{s.group='② 多轮生成：什么时候换成员？';});old.push(...scenarios);
root.DynamicBatchingVisual=(svg,viewport,s)=>root.TeachingDrawing.board(svg,viewport,'dynamic-batching',()=>500+58*(s.frames.at(-1).batches.length-1)+34,H=>{
 const {txt,rect,text,box,show,token,move,set}=H;
 const heading=txt('① 执行前 · 一张卡 = 一条图片请求','teach-heading');
 const ruler=rect('teach-track'),elapsed=rect('teach-value'),deadlineLine=rect('teach-compute'),clock=txt('','teach-number');
 const ticks=Array.from({length:9},(_,i)=>txt(String(i),'teach-small'));
 const waitLabel=txt('','teach-small'),waitTrack=rect('teach-track'),waitFill=rect('teach-compute');
 const arrivalLabel=txt('到达计划 · 先认出三条请求','teach-heading');
 const arrivals=[0,2,4].map((at,i)=>txt(`${'ABC'[i]} · t=${at} 到达`,'teach-small'));
 const queuePanel=rect(),queueTitle=txt('等待区 · A/B/C 是三条请求','teach-heading'),capacityLabel=txt('每批最多 3 条','teach-small');
 const slots=[0,1,2].map(()=>rect('teach-memory-slot'));
 const count=txt('','teach-small'),reason=txt('','teach-badge');
 const outTitle=txt('发往执行端','teach-heading'),outNote=txt('≠ 执行完成','teach-small');
 const rows=s.frames.at(-1).batches.map(()=>({bg:rect(),label:txt('','teach-small'),time:txt('','teach-small'),why:txt('','teach-small'),wait:[0,1,2].map(()=>txt('','teach-small'))}));
 const note=txt('A / B / C 列保持身份；空列表示未在这一批。','teach-small');
 const cards=['A','B','C'].map((id,i)=>token(id,id+' · 图片',78,'teach-token teach-color-'+i));
 return(p,m)=>{
  const W=m?360:720,x=i=>m?110+i*76:260+i*150,at=i=>[0,2,4][i];
  const index=Math.min(Math.floor(p+1e-8),s.frames.length-1),a=s.frames[index],b=s.frames[Math.min(index+1,s.frames.length-1)],raw=p-index,u=raw*raw*(3-2*raw),time=a.time+(b.time-a.time)*u;
  text(heading,18,24,m?'① 执行前 · 一张卡 = 一条请求':'① 执行前 · 一张卡 = 一条图片请求');text(clock,18,50,`t = ${Number(time.toFixed(1))} · 教学时间`);box(ruler,28,62,W-56,5);box(elapsed,28,62,(W-56)*time/8,5);ticks.forEach((e,i)=>text(e,25+(W-56)*i/8,84));
  box(deadlineLine,28+(W-56)*(a.deadline??0)/8,55,2,20);show(deadlineLine,a.deadline!==null?1:0);
  const first=a.queue[0],start=first?at('ABC'.indexOf(first)):time,wait=Math.max(0,time-start);
  text(waitLabel,18,106,first?`最早 ${first} 已等 ${Number(wait.toFixed(1))} / ${s.delay} · 截止 t=${a.deadline}`:(a.batches.length?'当前没有待组批请求 · 上批计时已结束':'没有等待中的请求 · 计时未启动'));box(waitTrack,18,116,W-36,7);box(waitFill,18,116,s.delay&&first?(W-36)*Math.min(1,wait/s.delay):0,7);
  text(arrivalLabel,18,170);arrivals.forEach((e,i)=>{text(e,x(i)+(m?32:0),190);e.setAttribute('text-anchor',m?'middle':'start');if(m)e.textContent=`t=${at(i)}`;});
  box(queuePanel,16,244,W-32,110);text(queueTitle,28,270);text(capacityLabel,28,292);slots.forEach((r,i)=>box(r,x(i)-3,300,m?70:84,38));
  text(count,28,346,`入队 ${a.queue.length} / 3`);count.dataset.queueCount=a.queue.length;
  const outgoing=b.event==='dispatch'&&raw>0;
  text(note,18,404+rows.length*58+16);
  text(reason,18,144,outgoing?'发往执行端途中':a.queue.length>=3?'凑满 → 发出':a.queue.length&&time>=a.deadline?(s.delay?'到期 → 发出':'不再等 → 发出'):a.queue.length?'未满且未到期 → 等待':a.arrived.length===3?'本例全部已发出':'等待下一条到达');
  text(outTitle,18,370,m?'执行端':'发往执行端');text(outNote,18,392);
  rows.forEach((r,i)=>{const y=404+i*58,batch=a.batches[i],planned=b.batches[i];box(r.bg,16,y,W-32,56);
   text(r.label,28,y+14,`第 ${i+1} 批`);text(r.time,28,y+34,batch?`t=${batch.time}`:outgoing&&planned?'发出途中':'尚未发出');text(r.why,28,y+53,batch?batch.reason:'');
   r.wait.forEach((e,j)=>{const k=batch?.ids.indexOf('ABC'[j])??-1;text(e,x(j)+(m?16:0),y+54,k>=0?`等了 ${batch.waits[k]}`:'');});show(r.bg,batch||planned?1:.35);
  });
  function pos(f,i){const id='ABC'[i],row=f.batches.findIndex(z=>z.ids.includes(id));return row>=0?{y:405+58*row,o:1}:f.queue.includes(id)?{y:300,o:1}:{y:204,o:.3};}
  cards.forEach((c,i)=>{const from=pos(a,i),to=pos(b,i),w=m?64:78;set(c.r,{width:w});set(c.t,{x:w/2});
   let cx=x(i),cy=from.y+(to.y-from.y)*u;
   // Queue slots sit below their own captions. Enter through the right aisle
   // beside the caption, then slide across the slot row once the card has
   // descended past the labels; a straight drop would cut through the title.
   if(from.y===204&&to.y===300){
    const aisle=m?Math.max(x(i),282):Math.max(x(i),312);
    if(u<.85){cx=x(i)+(aisle-x(i))*Math.min(1,u/.2);cy=204+(300-204)*(u/.85);}
    else{cx=aisle+(x(i)-aisle)*((u-.85)/.15);cy=300;}
   }
   move(c,cx,cy,from.o+(to.o-from.o)*u);c.g.setAttribute('aria-label',`请求 ${'ABC'[i]}，${raw&&from.y!==to.y?'移动中':a.queue.includes('ABC'[i])?'等待组批':a.arrived.includes('ABC'[i])?'已发出':'尚未到达'}`);});
  svg.dataset.clock=time.toFixed(3);svg.dataset.dispatched=a.batches.length;
 };
});
})(typeof window==='object'?window:globalThis);
