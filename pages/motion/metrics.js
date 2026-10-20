/* Measurement boundaries are the mechanism: a clock, observed events and a
 * declared sample/window. Playback speed never changes the measured values. */
(function(root){
'use strict';
const M=root.FoundationModel;if(!M)return;
const explanations=[
 ['从发出请求开始计时','首个 token 到达，TTFT 区间结束','第二个 token 到达，出现第一个后续间隔','第三个 token 到达；后续两个间隔均为 40 ms','TPOT = (180 − 100) / (3 − 1) = 40 ms'],
 ['从发出请求开始计时','首包带来 token 1','第二包同时带来 token 2、3','只能观测包间隔，不能恢复包内生成时刻','分包方式会影响可观测的间隔；没有凭空增加生成时间'],
 ['先看同一组 20 个样本','延迟从小到大排列，每格是一条请求','所有 20 条请求均成功，全部纳入分布','nearest-rank：ceil(0.95 × 20) = 19','第 19 项为 190 ms；同时保留成功率 100%'],
 ['先看原来的 20 个请求','保留成功与失败的记录','最后两项失败，成功延迟分布剩下 18 项','nearest-rank：ceil(0.95 × 18) = 18','P95 为 180 ms，但成功率只有 90%'],
 ['固定 0–10 秒统计窗口','2 秒时 R1 完成，贡献 2 个输出 token','4 秒时 R2 完成，再贡献 3 个输出 token','7 秒时 R3 完成，再贡献 1 个输出 token','R4 在 9 秒完成；10 秒关窗，按同一窗口计算吞吐']
];
const def=M.definitions.metrics;
const scenes=def.options.map((name,v)=>({name,kind:'metrics',family:v<2?'latency':v<4?'distribution':'throughput',title:v<2?'TTFT 与 ITL：先确定计时的两个端点':v<4?'P95：在声明的样本中找到位置':'吞吐：完成工作量除以同一个统计窗口',intro:v<2?'沿客户端时间轴看输出到达。TTFT 是等待首个输出的时间；后续间隔从相邻到达事件量起。':v<4?'每格一条请求，按延迟从小到大排列。P95 的位置取决于纳入统计的样本；同时保留失败记录。':'只数窗口内的已完成请求和它们产生的输出 token。请求数与 token 数使用不同单位，分母都是 10 秒。',variant:v,data:M.snapshot('metrics',v,4),frames:explanations[v].map((caption,i)=>({title:M.snapshot('metrics',v,i).title,caption})),stepDurations:v<2?[1.6,1.3,1.3,1.8]:v<4?[.7,1.1,1.8,.9]:[.9,.9,1.2,1.6],boundary:'所有时间和样本均为教学设定，不是实测。客户端观测边界、统计成功范围与窗口始终固定；调节播放速度不改变数值。'}));
root.TeachingLessons??={};root.TeachingLessons['f-metrics']=scenes;
const {board,phase,lerp}=root.TeachingDrawing;
root.MechanismVisuals['f-metrics']=(svg,viewport,s)=>board(svg,viewport,'metrics',()=>600,H=>{
 const {txt,rect,path,text,box,wire,show,token,move,set}=H,d=s.data,v=s.variant;
 const heading=txt('','teach-heading'),unit=txt('','teach-small'),axis=path(),cursor=path('teach-map'),clock=token('metric-clock','0',68,'teach-token teach-color-0');
 const labels=Array.from({length:6},()=>txt('','teach-small'));
 const bars=[0,1,2].map(()=>rect('teach-value'));
 const events=Array.from({length:4},(_,i)=>({point:token('metric-event-'+i,'',60,'teach-token teach-color-'+i),caption:txt('','teach-small'),line:path()}));
 const samples=d.samples.map((value,i)=>({value,i,t:token('sample-'+i,String(value),48,'teach-token teach-color-0')}));
 const percentileMark=rect('teach-map');set(percentileMark,{fill:'none','stroke-width':2});
 const verdict=txt('','teach-heading'),detail=txt('','teach-small'),detail2=txt('','teach-small');
 const packetCaption=txt('','teach-small'),windowNote=txt('','teach-small');
 const packetTokens=['token 2','token 3'].map((label,i)=>token('packet-'+i,label,56,'teach-token teach-color-'+(i+1)));
 return(p,m)=>{const W=m?360:720,left=m?26:52,right=W-48,span=right-left;
  samples.forEach(a=>show(a.t.g,v>=2&&v<4?1:0));show(percentileMark,v>=2&&v<4&&p>2?1:0);events.forEach(a=>{show(a.point.g,v<2||v===4?1:0);show(a.caption,v<2||v===4?1:0);show(a.line,0);});labels.forEach(a=>show(a,0));bars.forEach(a=>show(a,0));show(axis,v>=2&&v<4?0:1);show(cursor,v>=2&&v<4?0:1);show(clock.g,v>=2&&v<4?0:1);
  if(v<2){
   packetTokens.forEach(t=>show(t.g,0));show(packetCaption,0);show(windowNote,0);
   const i=Math.min(3,Math.floor(p)),times=v===0?[0,100,140,180]:[0,100,180,180],now=lerp(times[i],times[Math.min(3,i+1)],phase(p,i)),at=t=>left+span*t/200;
   text(heading,18,30,'发出请求 → 首个输出 → 后续输出');text(unit,18,58,'横轴：客户端观测时间（ms）');wire(axis,`M${left} 158 H${right}`,1);wire(cursor,`M${at(now)} 122 V390`,1);move(clock,Math.min(W-78,Math.max(8,at(now)-34)),80);clock.t.textContent=Math.round(now)+' ms';
   [0,100,200].forEach((t,j)=>{text(labels[j],at(t)-8,188,String(t));show(labels[j],1);});
   const eventTimes=v===0?[0,100,140,180]:[0,100,180],names=v===0?['发出','token 1','token 2','token 3']:['发出','包 1','包 2'];
   events.forEach((a,j)=>{if(j>=eventTimes.length){show(a.point.g,0);show(a.caption,0);return;}const t=eventTimes[j],arrived=now>=t-1e-6,x=at(t),y=j%2?270:212;move(a.point,Math.min(W-66,Math.max(6,x-30)),y,arrived?1:.15);a.point.t.textContent=names[j];wire(a.line,`M${x} 158 V${y}`,arrived?1:.15);text(a.caption,Math.min(W-65,Math.max(6,x-30)),y+53,t+' ms');a.point.g.dataset.arrived=String(arrived);});
   box(bars[0],at(0),350,span*Math.min(now,100)/200,12);show(bars[0],1);text(labels[3],18,385,now<100?'TTFT · 等首个输出':'TTFT = 100 ms');show(labels[3],1);
   if(v===0){box(bars[1],at(100),410,span*Math.max(0,Math.min(now,140)-100)/200,10);box(bars[2],at(140),410,span*Math.max(0,now-140)/200,10);show(bars[1],1);show(bars[2],1);text(verdict,18,474,p>=3?'后续间隔：40 ms、40 ms':'首个输出之后，才开始量后续间隔');text(detail,18,515,p>=4?'TPOT = (180 − 100) ÷ 2 = 40 ms':'同一时间轴，起止点随指标改变。');text(detail2,18,552,'这里每包一个 token，包间隔可用于观测 ITL。');}
   else{text(verdict,18,450,p>=2?'第二包内有 2 个 token':'同包到达，只有一个观测时刻');text(detail,18,490,'100 → 180 ms：包间隔为 80 ms。');text(detail2,18,530,'不能据此拆出两个 token 各自的生成耗时。');text(packetCaption,18,300,'包 2 内有两个 token；只有一个到达时刻。');show(packetCaption,phase(p,2,3));packetTokens.forEach((t,i)=>move(t,at(180)-28+i*58,326,phase(p,2,3)));}
   text(windowNote,18,570,p>=3?'请求总观测窗口（发出 → 末 token）= 180 ms；结束消息可能更晚。':'');show(windowNote,p>=3);
   svg.dataset.observedTime=now.toFixed(3);
  }else if(v<4){
   text(heading,18,30,'P95 取决于哪些样本被统计');text(unit,18,60,p<1?'将样本按延迟从小到大排列 · ms':'样本已按延迟从小到大排列 · ms');
   const cols=m?5:10,gap=m?66:65,failed=v===3&&p>=2,n=failed?18:20,idx=Math.ceil(.95*n)-1;
   samples.forEach(a=>{const excluded=failed&&a.i>=18,chosen=p>=3&&a.i===idx,original=(a.i*7)%20,order=phase(p,0,1);move(a.t,18+lerp(original%cols,a.i%cols,order)*gap,100+lerp(Math.floor(original/cols),Math.floor(a.i/cols),order)*64,v===3&&a.i>=18?1-.75*phase(p,1,2):1);a.t.t.textContent=a.value+(excluded?' ×':'');a.t.g.setAttribute('class','teach-token '+(chosen?'teach-color-2':'teach-color-0'));a.t.g.dataset.included=String(!excluded);a.t.g.dataset.percentile=String(chosen);});
   const scanned=Math.min(idx,Math.floor(idx*phase(p,2,3)));box(percentileMark,15+(scanned%cols)*gap,97+Math.floor(scanned/cols)*64,54,40);
   text(verdict,18,420,p<3?'先定样本，再找 95% 的位置':`P95：第 ${idx+1} 项 = ${d.p95} ms`);text(detail,18,466,`成功 ${n} / 20 · 成功率 ${n*5}%`);text(detail2,18,510,p>=3?`nearest-rank：ceil(0.95 × ${n}) = ${idx+1}`:'× 表示失败，仍保留记录。');
  }else{
   const times=[0,2,4,7,10],i=Math.min(4,Math.floor(p)),now=lerp(times[i],times[Math.min(4,i+1)],phase(p,i)),at=t=>left+span*t/10,completed=d.completions.filter(t=>t<=now+1e-6).length,total=d.outputCounts.slice(0,completed).reduce((a,b)=>a+b,0);
   text(heading,18,30,'同一窗口：分别数请求与输出 token');text(unit,18,58,'横轴：0–10 秒；标记是完成事件');wire(axis,`M${left} 140 H${right}`,1);wire(cursor,`M${at(now)} 122 V410`,1);move(clock,Math.min(W-78,Math.max(8,at(now)-34)),80);clock.t.textContent=now.toFixed(1)+' s';
   [0,5,10].forEach((t,j)=>{text(labels[j],at(t)-8,168,String(t));show(labels[j],1);});
   events.forEach((a,j)=>{const x=at(d.completions[j]),y=200+j*49,arrived=j<completed;move(a.point,x-30,y,arrived?1:.15);a.point.t.textContent='R'+(j+1);text(a.caption,18,y+24,`${d.completions[j]}s · ${d.outputCounts[j]} token`);});
   text(verdict,18,448,`完成 ${completed} 条 · 输出 ${total} token`);text(detail,18,490,p>=4?'请求吞吐：4 ÷ 10 = 0.4 请求/秒':'尚未关窗，先保留累计数。');text(detail2,18,530,p>=4?'输出吞吐：10 ÷ 10 = 1 token/秒':'两种工作量使用同一统计窗口。');svg.dataset.completed=completed;svg.dataset.outputTokens=total;
  }
 };
});
})(window);
