(function(){
'use strict';
const M=window.ParallelismFlow,host=document.querySelector('[data-parallel-lab]');if(!M||!host)return;
const $=id=>host.querySelector('#'+id),escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let mode='tp',option='default',timeline=M.build(mode),time=0,playing=false,rate=1,raf=0,last=0,lastPhase='',S,backdrops=[];
const clamp=x=>Math.max(0,Math.min(1,x)),ease=x=>{x=clamp(x);return x*x*(3-2*x);};
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
const p=id=>{const value=S.progress[id]||0;return reduced.matches?(value>=1?1:0):value;},done=id=>p(id)>=1,moving=id=>p(id)>0&&p(id)<1;
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
  out+=module(a,162,350,'gate_up_proj · Column','本地 gate、up → SiluAndMul',moving('gate'))+progress(a,209,350,'gate');
  if(done('gate'))out+=row([`g${r} [T,F/2]`,`u${r} [T,F/2]`],a+16,226,'activation',146,10);
  if(p('down')>0)out+=module(a,269,350,'down_proj · Row',done('reduce')?'完整 Y [T,H]':`Y${r} [T,H] · 尚缺另一卡贡献`,moving('down'))+progress(a,316,350,'down');
  if(done('down'))out+=chip(a+92,355,done('reduce')?'Y₀ + Y₁':'Y'+r+' · 局部贡献',done('reduce')?'result':'activation',166);
 }
 out+=flow('reduce',[[394,364],[454,364],[454,343],[506,343]],'Y₀','activation',66);
 out+=flow('reduce',[[506,389],[446,389],[446,402],[394,402]],'Y₁','activation',66);
 return out+result(done('reduce')?'两卡得到同样的完整特征；token 行数仍为 2。':'关键观察：gate/up 不先收齐；down 的同形状输出还要相加。');
}
function PP(){
 let out=title('SchedulerPPMixin：请求控制与 GPU 激活走不同通道','普通 event_loop_pp · 一个 microbatch 的依赖关系 · 时间轴为讲解进度');
 for(let r=0;r<2;r++){
  const x=rankX(r),a=inner(r);out+=panel(x,65,390,348,`PP Stage ${r} / Scheduler 进程 ${r}`,r?'接收 proxy 后，继续自己的层区间':'首先从 input_ids 建立 hidden_states');
  out+=module(a,128,350,'Scheduler / ScheduleBatch',r?(done('proxy')?'请求与 proxy 已就绪；可继续计算':done('request')?'请求对象已到；等待模型输入':'等待请求对象'):'持有本轮请求与 microbatch',moving('request'));
  out+=module(a,224,350,'ModelRunner → DeepseekV2Model',r?'本级 Decoder → norm / LM head':'Embedding → 本级 Decoder',moving(r?'last':'first'))+progress(a,271,350,r?'last':'first');
  out+=box(a,335,166,52,'kv')+text(a+12,356,'本级 KV buffer','code')+text(a+12,376,done(r?'last':'first')?'已有本级层的 KV':'只存本级层的 KV','sub');
  if(done(r?'last':'first'))out+=chip(a+190,345,r?'output_ids':'hidden + residual',r?'control':'activation',154);
 }
 out+=flow('request',[[394,152],[506,152]],'recv_reqs','control',100);
 out+=flow('proxy',[[394,296],[440,296],[440,205],[506,205]],'hidden + residual','activation',146);
 if(moving('proxy'))out+=text(450,317,'proxy · GPU 张量','sub','middle');
 out+=flow('result',[[506,402],[394,402]],'output_ids','control',102);
 return out+result(done('proxy')?'Stage 1 可以消费 proxy；Stage 0 的 KV 没有搬走。':'收到请求对象只解决“算谁”；收到 proxy 才得到“接着算什么”。');
}
function DP(){
 let out=title('DataParallelController：选择请求的执行副本','蓝色卡片为 Req / input_ids 等内部请求数据 · 两个副本的状态互不合并');
 out+=module(240,66,420,'round_robin_scheduler(req)','检查 active / status → sock_send 到入口 socket',S.phase.id.startsWith('route'));
 for(let r=0;r<2;r++){
  const a=inner(r);out+=panel(rankX(r),208,390,205,`副本 ${r} / Scheduler ${r}`,r===0&&!timeline.data.healthy?'status=False · 本例跳过此槽位':'TP=1、PP=1 · 完整模型与独立 KV');
  const requests=timeline.data.owners.map((owner,i)=>owner===r&&done('route'+i)?'R'+(i+1):null).filter(Boolean);
  out+=text(a,284,'本副本请求队列','sub')+row(requests,a,296,'control',68);
  const owns=timeline.data.owners.includes(r);out+=module(a,345,350,'Scheduler → ModelRunner',!owns?'本例未接收请求，不执行该批次':done('compute')?'按本副本 batch 继续结果处理':'各自组批、执行、维护 KV',owns&&moving('compute'))+(owns?progress(a,394,350,'compute'):'');
 }
 timeline.data.owners.forEach((r,i)=>{out+=flow('route'+i,[[450,116],[450,156],[inner(r)+160,156],[inner(r)+160,208]],'R'+(i+1),'control',62);});
 return out+result(done('route2')?'三个请求各有一个所属副本；Controller 不决定本副本内部的 batch。':'待路由 R1、R2、R3：检查的是副本可用状态，不是专家 ID。');
}
function DPA(){
 let out=title('LayerCommunicator：Attention 局部行 ↔ MLP 全局行','Attention DP=2：不同请求 · MLP TP=2：相同 token 行，不同权重分片');
 for(let r=0;r<2;r++){
  const a=inner(r);out+=panel(rankX(r),65,390,348,`rank ${r} · Attention TP 组 [${r}]`,`请求 ${r?'R2':'R1'} 的 KV 始终留在本组`);
  out+=module(a,127,350,'prepare_attn → Attention',r?'本地 b0、b1':'本地 a0、a1',moving('attention'))+progress(a,174,350,'attention');
  out+=text(a,220,'prepare_mlp 的激活缓冲','sub');
  const own=r?['b0','b1']:['a0','a1'];out+=row(own,a+(r?112:0),234);
  if(done('gather')){const foreign=r?['a0','a1']:['b0','b1'];out+=row(foreign,a+(r?0:112),234);}
  if(done('gather'))out+=module(a,282,350,'MLP TP · gate/up → down','两卡都按 a0,a1,b0,b1 的顺序处理',moving('mlp'))+progress(a,329,350,'mlp');
  if(p('scatter')>0){out+=moveRows('scatter',own,i=>[a+(r?112:0)+i*46,334],i=>[a+112+i*46,372],'result');out+=text(a,362,'本地取回','sub');}
 }
 out+=flow('gather',[[394,239],[506,239]],'a0,a1','activation',68)+flow('gather',[[506,269],[394,269]],'b0,b1','activation',68);
 if(moving('gather'))out+=text(450,194,'激活 gather','sub','middle');
 return out+result(done('scatter')?'下一层 Attention 又各自处理 a 行 / b 行；KV 未发生跨组搬运。':'全局缓冲包含 token 激活；请求队列与 KV 池不参与这次 gather。');
}
function EP(){
 let out=title('DeepEP：token 身份随 dispatch / combine 保留','每个 token 两份专家任务 → 本地专家计算 → 按权重恢复原 token 行');
 out+=row(['tA','tB'],382,72,'control',62,12);
 if(p('topk')>0)out+=text(44,143,'E0 ← tA .75 / tB .25','code')+text(506,143,'E1 ← tA .25 / tB .75','code');
 for(let r=0;r<2;r++){
  const a=inner(r);out+=panel(rankX(r),165,390,198,`EP rank ${r} · 代表专家 E${r}`,'专家权重在本地；每行保留原 token / 路由身份');
  if(done('dispatch'))out+=row(['tA','tB'],a+86,234,'activation',64,28);
  out+=module(a,284,350,'run_moe_core',done('experts')?`E${r}(tA)、E${r}(tB) 已算出`:'按本地专家重排的行执行 GEMM',moving('experts'))+progress(a,332,350,'experts');
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
  out+=module(a,277,350,'Column g → GEMM → Row g-bar','收齐行；权重仍分片；求和后分回序列片',moving('reduce'))+progress(a,324,350,'reduce');
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
  out+=text(a,222,p('qkv')>0?'本地 Q 保持分片；K/V 准备汇合':'等待本地 Q/K/V','sub');
  if(done('qkv'))out+=done('kv')?row(positions.map(i=>'KV'+i),a,266,'kv',39,4):row(indices.map(i=>'KV'+i),a+62,238,'kv',39,13);
  out+=module(a,316,350,'run_attention · prev / next','本卡 Q + 对应长度的 KV → 本地 hidden',moving('attn'))+progress(a,364,350,'attn');
 }
 out+=flow('kv',[[394,276],[506,276]],'KV0,1,6,7','kv',88)+flow('kv',[[506,301],[394,301]],'KV2,3,4,5','kv',88);
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
  out+=row(done('query')?['Q₀','Q₁']:[r?'Q₁':'Q₀'],a+110,226,'control',52);
  out+=module(a,272,350,'FlashInfer MLA · local Attention','本地 KV → O 与 LSE（以 2 为底）',moving('local'))+progress(a,319,350,'local');
  if(done('local'))out+=text(a+175,350,`O${r}=${r?6:2}    LSE${r}=${r?3:1}`,'value','middle');
  if(done('lse'))out+=text(a+175,378,`校正贡献：${r?'0.8 × 6 = 4.8':'0.2 × 2 = 0.4'}`,'result-text','middle');
 }
 out+=flow('query',[[394,233],[506,233]],'Q₀','control',56)+flow('query',[[506,253],[394,253]],'Q₁','control',56);
 out+=flow('lse',[[394,343],[506,343]],'LSE₀','control',65)+flow('lse',[[506,364],[394,364]],'LSE₁','control',65);
 out+=flow('reduce',[[394,394],[506,394]],'0.4','result',58)+flow('reduce',[[506,409],[394,409]],'4.8','result',58);
 return out+result(done('reduce')?'对应 head 得到 5.2，再保留各自 head 分片；简单平均 4 不等于该结果。':'LSE 保留每块 KV 的归一化分母信息；它决定局部 O 应占多少权重。');
}
const draw={tp:TP,pp:PP,dp:DP,dpa:DPA,ep:EP,sp:SP,pcp:PCP,dcp:DCP};
const definitions='<defs>'+['activation','control','kv','result'].map(c=>`<marker id="pf-arrow-${c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" class="${c}"/></marker>`).join('')+'</defs>';
function render(){
 S=M.sample(timeline,time);backdrops=[];const foreground=draw[mode]();$('parallel-scene').innerHTML=definitions+backdrops.join('')+foreground;$('parallel-scene').dataset.mode=mode;$('parallel-scene').dataset.phase=S.phase.id;$('parallel-scene').dataset.time=time.toFixed(3);
 $('parallel-seek').value=time;$('parallel-clock').textContent=`${time.toFixed(1)} / ${timeline.duration.toFixed(1)} s`;
 $('parallel-previous').disabled=time===0;$('parallel-next').disabled=time>=timeline.duration;
 $('parallel-toggle').textContent=playing?'暂停':'播放';$('parallel-toggle').setAttribute('aria-pressed',String(playing));
 $('parallel-seek').setAttribute('aria-valuetext',`${S.phase.title}，讲解时间 ${time.toFixed(1)} 秒`);
 if(lastPhase!==S.phase.id){
  lastPhase=S.phase.id;$('parallel-status').textContent=`${S.phase.index+1} / ${timeline.phases.length} · ${S.phase.title}`;$('parallel-caption').textContent=S.phase.summary;$('parallel-detail').textContent=S.phase.detail;
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
if(location.hash==='#parallel-title')requestAnimationFrame(()=>$('parallel-title').scrollIntoView({block:'start',behavior:'instant'}));
})();
