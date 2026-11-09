(function(){
'use strict';
const M=window.ParallelismFlow,host=document.querySelector('[data-parallel-lab]');if(!M||!host)return;
const $=id=>host.querySelector('#'+id),escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let mode='tp',option='default',timeline=M.build(mode),time=0,playing=false,rate=1,raf=0,last=0,lastPhase='',S,backdrops=[];
const clamp=x=>Math.max(0,Math.min(1,x)),ease=x=>{x=clamp(x);return x*x*(3-2*x);};
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
const p=id=>{const value=S.progress[id]||0;return reduced.matches?(value>=1?1:0):value;},done=id=>(S.progress[id]||0)>=1,moving=id=>(S.progress[id]||0)>0&&(S.progress[id]||0)<1;
const text=(x,y,label,cls='',anchor='start')=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${escape(label)}</text>`;
const box=(x,y,w,h,cls='')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" class="${cls}"/>`;
function chip(x,y,label,cls='activation',w=40){return `<g class="chip ${cls}" data-chip="${escape(label)}">${box(x,y,w,26)}${text(x+w/2,y+17,label,'chip-label','middle')}</g>`;}
function row(labels,x,y,cls='activation',width=40,gap=6){return labels.map((v,i)=>chip(x+i*(width+gap),y,v===null?'∅':v,v===null?'padding':cls,width)).join('');}
function panel(x,y,w,h,title,sub){backdrops.push(box(x,y,w,h,'rank-panel'));return text(x+16,y+25,title,'rank-title')+text(x+16,y+46,sub,'sub');}
function module(x,y,w,name,sub,active=false){return `<g class="module ${active?'active':''}">${box(x,y,w,50)}${text(x+12,y+20,name,'code')}${text(x+12,y+39,sub,'sub')}</g>`;}
function path(points,cls='activation',arrow=true){return `<path d="${points.map((a,i)=>(i?'L':'M')+a.join(',')).join(' ')}" class="wire ${cls}" ${arrow?`marker-end="url(#pf-arrow-${cls})"`:''}/>`;}
function flow(id,points,label,cls='activation',width=102){
 if(p(id)<=0)return '';
 const lengths=points.slice(1).map((a,i)=>Math.hypot(a[0]-points[i][0],a[1]-points[i][1])),total=lengths.reduce((a,b)=>a+b,0);let distance=ease((p(id)-.08)/.84)*total,index=0;
 while(index<lengths.length-1&&distance>lengths[index])distance-=lengths[index++];
 const a=points[index],b=points[index+1],f=lengths[index]?distance/lengths[index]:1,x=a[0]+(b[0]-a[0])*f,y=a[1]+(b[1]-a[1])*f;
 return path(points,cls)+(!done(id)?`<g class="moving-payload" data-motion="${id}" data-x="${x.toFixed(3)}" data-y="${y.toFixed(3)}">${chip(x-width/2,y-13,label,cls,width)}</g>`:'');
}
function moveRows(id,labels,from,to,cls='activation',width=38){
 const f=ease((p(id)-.08)/.84);
 return labels.map((label,i)=>{const a=from(i),b=to(i),x=a[0]+(b[0]-a[0])*f,y=a[1]+(b[1]-a[1])*f;return `<g data-row="${escape(label)}" data-x="${x.toFixed(3)}" data-y="${y.toFixed(3)}">${chip(x,y,label,label==='∅'?'padding':cls,width)}</g>`;}).join('');
}
function progress(x,y,w,id){return box(x,y,w,5,'progress-track')+`<rect x="${x}" y="${y}" width="${w*p(id)}" height="5" rx="2" class="progress-fill"/>`;}
const rankX=r=>24+r*462,inner=r=>rankX(r)+20;
function title(main,sub){return text(24,26,main,'board-title')+text(24,49,sub,'sub');}
function result(label){return box(24,428,852,36,'result-strip')+text(40,451,label,'result-text');}
function TP(){
 let out=title('DeepseekV2MLP.forward：同一批 token，特征维分片','TP 组 [0,1] · 权重驻留本地 · 沿模型真实调用顺序向下读');
 for(let r=0;r<2;r++){
  const x=rankX(r),a=inner(r);out+=panel(x,65,390,348,`rank ${r} / GPU ${r}`,`gate/up 权重：F/2 输出维 · down 权重：F/2 输入维`);
  out+=text(a,140,'输入 X [T=2,H]','sub');out+=row(['t0','t1'],a+190,121);
  out+=module(a,162,350,'gate_up_proj · Column','本地 gate、up → SiluAndMul',moving('gate'+r))+progress(a,209,350,'gate'+r);
  if(done('gate'+r))out+=row([`g${r} [T,F/2]`,`u${r} [T,F/2]`],a+16,226,'activation',146,10);
  if(p('down'+r)>0)out+=module(a,269,350,'down_proj · Row',done('reduce')?'完整 Y [T,H]':`Y${r} [T,H] · 尚缺另一卡贡献`,moving('down'+r))+progress(a,316,350,'down'+r);
  if(done('down'+r))out+=chip(a+92,355,done('reduce')?'Y₀ + Y₁':'Y'+r+' · 局部贡献',done('reduce')?'result':'activation',166);
 }
 out+=flow('reduce',[[394,364],[454,364],[454,343],[506,343]],'Y₀','activation',66);
 out+=flow('reduce',[[506,389],[446,389],[446,402],[394,402]],'Y₁','activation',66);
 return out+result(done('reduce')?'两卡得到同样的完整特征；token 行数仍为 2。':'关键观察：gate/up 不先收齐；down 的同形状输出还要相加。');
}
function PP(){
 let out=title('两个 microbatch：同批先后、跨批并行、CPU / GPU 重叠','MB1 / MB2 为微批次 · h+r = hidden_states + residual · CUDA / async depth=0');
 for(let r=0;r<2;r++){
  const a=inner(r),cpu=S.active.find(e=>e.lane==='cpu'+r),gpu=S.active.find(e=>e.lane==='gpu'+r);
  out+=panel(rankX(r),65,390,348,`PP Stage ${r} / 独立 Scheduler`,r?'后段 Decoder → norm / LM head':'Embedding → 前段 Decoder');
  out+=`<g data-actor="cpu${r}" data-active="${cpu?.id||''}">`+module(a,128,350,'CPU · Scheduler / 结果处理',cpu?cpu.label:time>=20?'两批结果已处理':'本窗口未安排 CPU 活动',!!cpu&&cpu.kind!=='wait')+'</g>';
  for(let n=1;n<=2;n++){
   const id='gpu'+r+n,req='req'+n,state=done('process'+r+n)?'结果已处理':done(id)?'本级 forward 完成':moving(id)?'正在计算':r&&!done(req)?'等待请求':r&&!done('proxy'+n)?'等待激活可用':'等待本级执行';
   out+=`<g data-microbatch="MB${n}" data-stage="${r}" data-state="${state}">`+text(a+12,202+(n-1)*21,`MB${n} · ${state}`,'code')+'</g>';
  }
  out+=`<g data-actor="gpu${r}" data-active="${gpu?.id||''}">`+module(a,248,350,'GPU · ModelRunner / 本级模型',gpu?gpu.label:done('gpu'+r+'2')?'本窗口两次 forward 已完成':'等待 forward / 输入依赖',gpu?.kind==='compute')+(gpu?.kind==='compute'?progress(a,295,350,gpu.id):'')+'</g>';
  out+=box(a,335,166,52,'kv')+text(a+12,356,'本级 KV buffer','code')+text(a+12,376,'始终留在本 Stage','sub');
  for(let n=1;n<=2;n++)if(done('process'+r+n))out+=chip(a+190,333+(n-1)*29,`MB${n} 已处理`,'result',145);
 }
 for(let n=1;n<=2;n++){
  out+=flow('req'+n,[[394,150],[506,150]],`MB${n} 请求`,'control',86);
  out+=flow('proxy'+n,[[394,274],[506,274]],`MB${n} h+r`,'activation',94);
  out+=flow('out'+n,[[506,286],[458,286],[458,168],[394,168]],`MB${n} output`,'activation',100);
  out+=flow('relay'+n,[[394,178],[506,178]],`MB${n} 转发`,'activation',86);
 }
 const msg=time<6.7?'请求先到；CPU 提交返回后，GPU 仍必须等待输入与流依赖。':time<9.4?'Stage 0 正在算 MB2，Stage 1 正在算 MB1：跨批次重叠。':time<11.7?'Stage 0 CPU 处理 MB1 输出时，本级 GPU 仍可计算 MB2。':time<14.3?'MB2 激活交接；MB1 输出转发已投递，末级仍需等流依赖。':time<16.2?'末级仍在算 MB2；CPU 等输出流依赖，不能提前处理 MB1。':time<20?'等待 MB2 输出及 D2H，再分别更新各级 batch。':'两批均已处理；时间轴保留依赖、通信、计算和等待，可任意回看。';
 return out+result(msg);
}
function DP(){
 let out=title('DataParallelController：派发与各副本执行可交错','R1–R3 各示意一次 forward；具体组批时刻为示例，各副本独立决定何时执行');
 const routing=S.active.some(e=>e.lane==='controller');
 out+=module(240,66,420,'round_robin_scheduler(req)','选择可用副本 → sock_send；不等待其他请求全部派发',routing);
 for(let r=0;r<2;r++){
  const a=inner(r),jobs=timeline.data.jobs.filter(j=>j.owner===r),work=jobs.find(j=>time>=j.start&&time<j.finish),arrived=jobs.filter(j=>time>=j.arrival);
  out+=panel(rankX(r),208,390,205,`副本 ${r} / Scheduler ${r}`,r===0&&!timeline.data.healthy?'status=False · 本例跳过此槽位':'TP=1、PP=1 · 完整模型与独立 KV');
  out+=text(a,282,'本副本请求：等待 / 本轮执行 / 已处理','sub');
  arrived.forEach((j,i)=>{const state=time<j.start?'等待本副本':time<j.end?'GPU forward':time<j.finish?'结果处理':'本轮已处理';out+=`<g data-request="R${j.index+1}" data-owner="${r}" data-state="${state}">`+chip(a+i*112,294,'R'+(j.index+1),time>=j.finish?'result':'control',100)+text(a+i*112+50,336,state,'sub','middle')+'</g>';});
  const label=work?`R${work.index+1} · ${time<work.end?'GPU forward':'CPU 结果处理'}`:!jobs.length?'本例无请求，不执行这批工作':arrived.some(j=>time<j.start)?'请求已到，等待本副本调度':arrived.length?'本窗口已处理的请求保持原副本归属':'等待本副本请求';
  out+=`<g data-actor="replica${r}" data-active="${work?'R'+(work.index+1):''}">`+module(a,345,350,'Scheduler → ModelRunner',label,!!work)+(work?progress(a,394,350,'compute'+work.index):'')+'</g>';
 }
 timeline.data.owners.forEach((r,i)=>{out+=flow('route'+i,[[450,116],[450,156],[inner(r)+160,156],[inner(r)+160,208]],'R'+(i+1),'control',62);});
 return out+result(timeline.option==='skip'?'副本 0 不可用，所有请求进入副本 1；后到请求可等待本副本前批。':'Controller 继续派发时，已获请求的副本可以开算；两副本无需统一起止。');
}
function DPA(){
 let out=title('LayerCommunicator：Attention 局部行 ↔ MLP 全局行','Attention DP=2：不同请求 · MLP TP=2：相同 token 行，不同权重分片');
 for(let r=0;r<2;r++){
  const a=inner(r);out+=panel(rankX(r),65,390,348,`rank ${r} · Attention TP 组 [${r}]`,`请求 ${r?'R2':'R1'} 的 KV 始终留在本组`);
  out+=module(a,127,350,'prepare_attn → Attention',r?'本地 b0、b1':'本地 a0、a1',moving('attention'+r))+progress(a,174,350,'attention'+r);
  out+=text(a,220,'prepare_mlp 的激活缓冲','sub');
  const own=r?['b0','b1']:['a0','a1'];out+=row(own,a+(r?112:0),234);
  if(done('gather')){const foreign=r?['a0','a1']:['b0','b1'];out+=row(foreign,a+(r?0:112),234);}
  if(done('gather'))out+=module(a,282,350,'MLP TP · gate/up → down',done('mlpReduce')?'TP 求和完成；随后本地取回本组行':done('mlp'+r)?'局部贡献已算出，TP 求和中':'本地 gate/up → down：同一全局行序',moving('mlp'+r)||moving('mlpReduce'))+progress(a,329,350,'mlp');
  if(p('scatter')>0){out+=moveRows('scatter',own,i=>[a+(r?112:0)+i*46,334],i=>[a+112+i*46,372],'result');out+=text(a,362,'本地取回','sub');}
 }
 out+=flow('gather',[[394,239],[506,239]],'a0,a1','activation',68)+flow('gather',[[506,269],[394,269]],'b0,b1','activation',68);
 if(moving('gather'))out+=text(450,194,'激活 gather','sub','middle');
 return out+result(done('scatter')?'下一层 Attention 又各自处理 a 行 / b 行；KV 未发生跨组搬运。':'全局缓冲包含 token 激活；请求队列与 KV 池不参与这次 gather。');
}
function EP(){
 let out=title('DeepEP：token 身份随 dispatch / combine 保留','顶部合看 r0 的 tA 与 r1 的 tB；专家任务在两个 rank 之间派发与归并');
 out+=row(['r0: tA','r1: tB'],349,72,'activation',94,14);
 if(p('topk')>0)out+=text(44,143,'E0 ← tA .75 / tB .25','code')+text(506,143,'E1 ← tA .25 / tB .75','code');
 for(let r=0;r<2;r++){
  const a=inner(r);out+=panel(rankX(r),165,390,198,`EP rank ${r} · 代表专家 E${r}`,'专家权重在本地；每行保留原 token / 路由身份');
  if(done('dispatch'))out+=row(['tA','tB'],a+86,234,'activation',64,28);
  out+=module(a,284,350,'run_moe_core',done('experts'+r)?`E${r} 已算出，等待 Combine`:'按本地专家重排的行执行 GEMM',moving('experts'+r))+progress(a,332,350,'experts'+r);
  out+=flow('combine',[[a+175,363],[a+175,392],[405+r*74,392]],r?'E1 输出':'E0 输出','result',95);
 }
 out+=flow('dispatch',[[450,108],[450,247],[310,247]],'tA,tB','activation',72)+flow('dispatch',[[450,108],[450,247],[590,247]],'tA,tB','activation',72);
 if(done('combine'))out+=row(['tA′','tB′'],382,380,'result',62,12);
 return out+result(done('combine')?'tA′=.75 E0(tA)+.25 E1(tA)；tB′=.75 E1(tB)+.25 E0(tB)。':'TopKOutput 包含专家 ID 和权重；dispatch 重排的是专家任务，不是服务请求。');
}
function SP(){
 const active=timeline.data.active;
 let out=title('LayerNorm SP：序列片只在指定区域保留','Qwen3 dense 的实现分支 · t0–t4 共 5 行 · ∅ 为补齐行');
 if(!active)return out+module(130,130,640,'runs_sp(DECODE) = False','本轮沿普通 TP 执行；不开启序列分片区域',true)+text(450,245,'没有 entry_scatter → 没有 SP g / g-bar','board-title','middle')+result('模式门槛失败：不能仅因为启动时开启 SP，就给 Decode 画上 SP 通信。');
 for(let r=0;r<2;r++){
  const a=inner(r),labels=timeline.data.rows[r].map(n=>n===null?'∅':'t'+n);
  out+=panel(rankX(r),65,390,348,`TP rank ${r} · 相同模型的协作者`,'SP 局部行 ↔ 线性层所需的全序列');
  out+=text(a,140,'本地 entry slice / Norm','sub');
  if(p('slice')===0)out+=row(['t0','t1','t2','t3','t4'],a,155,'activation',40,7);
  if(p('slice')>0&&p('gather')===0)out+=moveRows('slice',labels,i=>[a+(r*3+i)*47,155],i=>[a+80+i*47,197]);
  if(p('gather')>0){out+=moveRows('gather',['t0','t1','t2','t3','t4'],i=>[inner(i<3?0:1)+80+(i%3)*47,197],i=>[a+i*47,236]);}
  const task=S.active.find(e=>e.lane===String(r));out+=module(a,277,350,'SP 区域 · 当前动作',task?task.label:'本地输入已准备',task&&task.kind!=='wait')+(task?progress(a,324,350,task.id):'');
  if(p('reduce')>0)out+=moveRows('reduce',labels,i=>[a+(r*3+i)*47,333],i=>[a+80+i*47,366],'result');
 }
 if(p('exit')>0){out+=box(110,346,680,62,'exit-overlay')+text(128,370,'模型出口','sub')+moveRows('exit',['t0','t1','t2','t3','t4'],i=>[inner(i<3?0:1)+80+(i%3)*47,366],i=>[300+i*52,367],'result');}
 return out+result(done('exit')?'恢复 5 个真实 token 行后进入 LM head；padding ∅ 已移除。':'entry 是本地切片；Column 做 All-gather，Row 做 Reduce-scatter。');
}
function PCP(){
 let out=title('ZigzagCPStrategy：Q 分片、K/V 汇合、出口恢复顺序','GQA / FA3 分支 · Attention CP 组 [0,1] · 同一请求的 8 个位置');
 if(!timeline.data.active)return out+row(['t0','t1','t2'],370,138)+module(130,210,640,'can_apply：3 < 2 × CP → False','本轮不创建 Zigzag 分片；沿该模型的非 CP forward 继续',true)+result('真实门槛：不只检查 CP 开关，还要检查 forward mode 和每条请求的 extend 长度。');
 const positions=Array.from({length:8},(_,i)=>i);
 for(let r=0;r<2;r++){
  const a=inner(r),indices=timeline.data.rows[r];out+=panel(rankX(r),103,390,310,`CP rank ${r} · 局部查询`,indices.map(i=>'t'+i).join('、'));
  if(p('shard')>0)out+=moveRows('shard',indices.map(i=>'t'+i),i=>[248+indices[i]*50,64],i=>[a+62+i*52,170]);
  out+=text(a,222,p('qkv'+r)>0?'本地 Q 保持分片；K/V 准备汇合':'等待本地 Q/K/V','sub');
  if(done('qkv'+r))out+=done('kv')?row(positions.map(i=>'KV'+i),a,266,'kv',39,4):row(indices.map(i=>'KV'+i),a+62,238,'kv',39,13);
  out+=module(a,316,350,'run_attention · prev / next','本卡 Q + 对应长度的 KV → 本地 hidden',moving('attn'+r))+progress(a,364,350,'attn'+r);
 }
 out+=flow('kvTransfer',[[394,276],[506,276]],'KV0,1,6,7','kv',88)+flow('kvTransfer',[[506,301],[394,301]],'KV2,3,4,5','kv',88);
 if(p('shard')===0)out+=row(positions.map(i=>'t'+i),248,64,'control',40,10);
 if(p('exit')>0){out+=box(152,363,596,48,'exit-overlay');out+=moveRows('exit',positions.map(i=>'h'+i),i=>{const owner=timeline.data.rows.findIndex(rows=>rows.includes(i));return [inner(owner)+62+timeline.data.rows[owner].indexOf(i)*52,371];},i=>[252+i*50,375],'result');}
 return out+result(done('exit')?'出口 hidden 已恢复原 token 顺序；每层 KV 汇合与出口 hidden 汇合是两件事。':'metadata 保留分片和反向索引，防止 Gather 后按 rank 顺序误当 token 原顺序。');
}
function DCP(){
 let out=title('DCP / MLA：保留分散 KV，合并完整 Attention 贡献','ag_rs：Q heads 汇合 → 局部 O + LSE → 校正 → Reduce-scatter');
 for(let r=0;r<2;r++){
  const a=inner(r);out+=panel(rankX(r),65,390,348,`DCP rank ${r} · owner = slot % 2`,'KV 不随 Q 和输出的交换搬到对方');
  out+=text(a,137,'已有 widened KV slots → 本地行','sub')+row([r,r+2,r+4].map((v,i)=>`${v}→${i}`),a+50,150,'kv',72,12);
  out+=text(a,214,done('query')?'该组 Q heads 已收齐':'本地 Q head 分片','sub');
  out+=row(done('query')?['Q₀','Q₁']:[r?'Q₁':'Q₀'],a+110,226,'activation',52);
  out+=module(a,272,350,'FlashInfer MLA · local Attention','本地 KV → O 与 LSE（以 2 为底）',moving('local'+r))+progress(a,319,350,'local'+r);
  if(done('local'+r))out+=text(a+175,350,`O${r}=${r?6:2}    LSE${r}=${r?3:1}`,'value','middle');
  if(done('lse'))out+=text(a+175,378,`校正贡献：${r?'0.8 × 6 = 4.8':'0.2 × 2 = 0.4'}`,'result-text','middle');
 }
 out+=flow('query',[[394,233],[506,233]],'Q₀','activation',56)+flow('query',[[506,253],[394,253]],'Q₁','activation',56);
 out+=flow('lseTransfer',[[394,343],[506,343]],'LSE₀','activation',65)+flow('lseTransfer',[[506,364],[394,364]],'LSE₁','activation',65);
 out+=flow('reduce',[[394,394],[506,394]],'0.4','result',58)+flow('reduce',[[506,409],[394,409]],'4.8','result',58);
 return out+result(done('reduce')?'对应 head 得到 5.2，再保留各自 head 分片；简单平均 4 不等于该结果。':'LSE 保留每块 KV 的归一化分母信息；它决定局部 O 应占多少权重。');
}
const draw={tp:TP,pp:PP,dp:DP,dpa:DPA,ep:EP,sp:SP,pcp:PCP,dcp:DCP};
function executionLanes(){
 const x0=132,w=744,y=516,step=25,scale=w/timeline.duration,lanes=timeline.execution.lanes;
 const shorten=(label,width)=>{let n=0,out='';for(const c of label){n+=/[\u0000-\u007f]/.test(c)?5.7:10.5;if(n>width-12)return out.length>1?out+'…':'';out+=c;}return out;};
 let out=`<g class="execution-lanes" data-clock="${time.toFixed(3)}">`+text(24,488,'同一时钟下的执行轨道','rank-title')+text(876,488,'示例交错 / 非实测；同列为同时刻','sub','end');
 for(let t=0;t<=timeline.duration;t+=5)out+=text(x0+t*scale,507,`${t}s`,'time-tick',t===0?'start':'middle');
 lanes.forEach((l,i)=>{
  const top=y+i*step;out+=text(24,top+15,l.label,'lane-label')+box(x0,top,w,20,'lane-background');
  for(const e of l.events){
   const x=x0+e.start*scale,width=(e.end-e.start)*scale,active=time>=e.start&&time<e.end;
   const value=clamp((time-e.start)/(e.end-e.start)),f=reduced.matches?(value>=1?1:0):value;
   out+=`<g class="execution-event ${e.kind}" data-activity="${e.id}" data-owner="${l.id}" data-active="${active}" data-start="${e.start}" data-end="${e.end}"><title>${escape(l.label+'：'+e.label+'；示例 '+e.start+'–'+e.end+' 秒')}</title>`+box(x,top,width,20,'event-slot')+`<rect x="${x}" y="${top}" width="${width*f}" height="20" rx="3" class="event-progress"/>`+text(x+width/2,top+14,shorten(e.label,width),'event-label','middle')+'</g>';
  }
 });
 const bottom=y+lanes.length*step,cx=x0+time*scale;
 out+=`<path d="M${cx},511V${bottom-3}" class="execution-cursor"/>`;
 out+=text(24,bottom+18,'实色：计算 / 数据动作　紫色：CPU / 控制　虚框：等待　空轨：本窗口未安排活动','sub');
 return out+'</g>';
}
const definitions='<defs>'+['activation','control','kv','result'].map(c=>`<marker id="pf-arrow-${c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" class="${c}"/></marker>`).join('')+'</defs>';
function render(){
 S=M.sample(timeline,time);backdrops=[];const foreground=draw[mode]();$('parallel-scene').setAttribute('viewBox',`0 0 900 ${550+timeline.execution.lanes.length*25}`);$('parallel-scene').innerHTML=definitions+backdrops.join('')+`<g id="parallel-drawing">${foreground}</g>`+executionLanes();$('parallel-scene').dataset.mode=mode;$('parallel-scene').dataset.phase=S.phase.id;$('parallel-scene').dataset.time=time.toFixed(3);
 $('parallel-seek').value=time;$('parallel-clock').textContent=`${time.toFixed(1)} / ${timeline.duration.toFixed(1)} s`;
 $('parallel-previous').disabled=time===0;$('parallel-next').disabled=time>=timeline.duration;
 $('parallel-toggle').textContent=playing?'暂停':'播放';$('parallel-toggle').setAttribute('aria-pressed',String(playing));
 $('parallel-seek').setAttribute('aria-valuetext',`${S.phase.title}，讲解时间 ${time.toFixed(1)} 秒`);
 $('parallel-active').innerHTML=timeline.execution.lanes.map(l=>{const active=S.active.filter(e=>e.lane===l.id);return `<div><strong>${escape(l.label)}</strong><span>${escape(active.length?active.map(e=>e.label).join('；'):time>=Math.max(...l.events.map(e=>e.end))&&l.events.length?'本窗口活动已完成':'本窗口未安排活动')}</span></div>`;}).join('');
 if(lastPhase!==S.phase.id){
  lastPhase=S.phase.id;$('parallel-status').textContent=`讲解焦点 ${S.phase.index+1} / ${timeline.phases.length} · ${S.phase.title}`;$('parallel-caption').textContent=S.phase.summary;$('parallel-detail').textContent=S.phase.detail;
  $('parallel-source').innerHTML=S.phase.refs.map(key=>`<a href="${M.url(key)}" target="_blank" rel="noopener">${escape(M.sources[key][0])} · ${escape(M.sources[key][2].replace('def ','').replace('(','') )} ↗</a>`).join('');
  host.querySelectorAll('[data-event]').forEach(b=>b.setAttribute('aria-current',String(Number(b.dataset.event)===S.phase.index)));
 }
}
function url(){const u=new URL(location.href);u.searchParams.set('mode',mode);if(option==='default')u.searchParams.delete('case');else u.searchParams.set('case',option);u.searchParams.delete('t');u.searchParams.delete('step');history.replaceState(null,'',u);}
function stop(){playing=false;cancelAnimationFrame(raf);raf=0;last=0;}
function select(next,choice='default',update=true){
 stop();mode=next;option=choice;timeline=M.build(mode,option);time=0;lastPhase='';
 $('parallel-question').textContent=timeline.question;$('parallel-config').textContent=timeline.config;$('parallel-boundary').textContent=timeline.boundary;
 $('parallel-seek').max=timeline.duration;
 const options=mode==='sp'?[['default','EXTEND / Prefill'],['decode','DECODE / 不启用 SP']]:mode==='pcp'?[['default','extend 长度 8'],['short','extend 长度 3 / 不启用 CP']]:mode==='dp'?[['default','两个副本可用'],['skip','副本 0 不可用']]:[];
 $('parallel-option-wrap').hidden=!options.length;$('parallel-option').innerHTML=options.map(([v,label])=>`<option value="${v}">${label}</option>`).join('');$('parallel-option').value=option;
 host.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode)));
 $('parallel-events').innerHTML=timeline.phases.map(p=>`<button type="button" data-event="${p.index}">${p.index+1}. ${escape(p.title)}</button>`).join('');
 if(update)url();render();
}
function tick(now){if(!playing)return;if(last)time=Math.min(timeline.duration,time+(now-last)/1000*rate);last=now;if(time>=timeline.duration){stop();}render();if(playing)raf=requestAnimationFrame(tick);}
host.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{if(mode!==b.dataset.mode)select(b.dataset.mode);}));
$('parallel-option').addEventListener('change',e=>select(mode,e.target.value));
$('parallel-toggle').addEventListener('click',()=>{if(playing)stop();else{if(time>=timeline.duration){time=0;lastPhase='';}playing=true;last=0;raf=requestAnimationFrame(tick);}render();});
$('parallel-seek').addEventListener('input',e=>{stop();time=Number(e.target.value);render();});
$('parallel-rate').addEventListener('change',e=>{rate=Number(e.target.value);last=0;});
$('parallel-reset').addEventListener('click',()=>{stop();time=0;lastPhase='';render();});
$('parallel-next').addEventListener('click',()=>{stop();time=timeline.phases.find(p=>p.end>time+.001)?.end||timeline.duration;render();});
$('parallel-previous').addEventListener('click',()=>{stop();time=[...timeline.phases].reverse().find(p=>p.start<time-.001)?.start||0;render();});
$('parallel-events').addEventListener('click',e=>{const b=e.target.closest('[data-event]');if(b){stop();time=timeline.phases[Number(b.dataset.event)].start;render();}});
document.addEventListener('visibilitychange',()=>{if(document.hidden){stop();render();}});window.addEventListener('pagehide',stop);
function fromLocation(){const u=new URL(location.href),m=u.searchParams.get('mode');select(M.modes.includes(m)?m:'tp',u.searchParams.get('case')||'default',false);}
window.addEventListener('popstate',fromLocation);
reduced.addEventListener('change',render);
// Give the two-rank canvas the full width before the global mobile breakpoint.
const compactNav=matchMedia('(max-width:1160px)'),menu=document.querySelector('.sidebar-menu');
function fitNavigation(){if(menu)menu.open=!compactNav.matches;}
compactNav.addEventListener('change',fitNavigation);window.addEventListener('resize',fitNavigation);fitNavigation();
host.hidden=false;fromLocation();
const resume=window.SiteUpdate?.take('parallelism');
if(resume&&M.modes.includes(resume.mode)){select(resume.mode,resume.option,false);time=Math.max(0,Math.min(timeline.duration,Number(resume.time)||0));render();}
window.SiteUpdate?.register('parallelism',()=>{stop();render();return {mode,option,time};});
if(location.hash==='#parallel-title')requestAnimationFrame(()=>$('parallel-title').scrollIntoView({block:'start',behavior:'instant'}));
})();
