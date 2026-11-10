/* Each lesson projects its own state. No inferred completion or extra timers. */
(()=>{'use strict';const D=window.LearningDepth;if(!D)return;
 const {text:T,box:B,chips:C,wire:W,svg:S,facts:F,tracks:K}=D;
 const value=x=>Array.isArray(x)?'['+x.join(', ')+']':x==null?'—':typeof x==='object'?JSON.stringify(x):String(x);
 const note=s=>`<p class="ld-projection-note">${D.esc(s)}</p>`;
 const state=(x,y,title,lines,w=310,h=140)=>B(x,y,w,h)+T(x+16,y+28,title,'ld-heading')+lines.map((v,i)=>T(x+16,y+56+i*24,v,i?'ld-small':'')).join('');
 function req(tl,snap){const s=snap.state,cpu=s.consuming,copy=snap.active.find(e=>e.lane==='copy'),commit=snap.active.find(e=>e.token),owners=[s.waiting&&'waiting',s.chunk&&'chunked_req',...Object.keys(s.members).filter(k=>s.members[k].includes('R1'))].filter(Boolean);
  let out=state(20,35,'CPU 当前消费',[cpu?`${cpu.id} · ${cpu.ready?'可读':'等 copy_done'}`:s.resultReady?'本轮结果可读':'尚无可读结果',cpu?.token||s.resultReady&&s.deviceToken||'—'],300,115);
  out+=state(20,175,'仍在结果队列',[s.results.map(q=>q.id+(q.ready?' 可读':' 在途')).join(' / ')||'空'],300,80);
  out+=T(25,292,'持有 R1 的容器','ld-small')+T(25,318,owners.join(' / ')||'这些容器未持有','ld-value');
  out+=B(410,35,310,290,'active')+T(430,65,'R1 · 同一个请求对象','ld-heading')+T(430,96,s.terminal,'ld-small')+T(430,131,'正式输出 output_ids','ld-small')+C(430,146,s.output.length?s.output:['—'],{width:52})+T(430,211,s.pending?'取消待收尾：'+s.pending:s.finished?'结束原因：'+s.finished:'请求尚未结束')+T(430,248,`输入已处理 ${s.progress} / ${s.prompt}`)+T(430,295,'入队、选批与回撤不会更换对象','ld-small');
  out+=state(805,35,'资源归属',[s.row?'R1 持有请求行':'R1 未持有请求行',`KV 预留 ${s.allocated} · 已写 ${s.written}`,`缓存保留 ${s.cached}`],275,140);
  out+=C(817,193,Array.from({length:8},(_,i)=>i<s.written?'KV':i<s.allocated?'待写':'·'),{width:26,gap:5});
  out+=state(805,260,'设备输出',[s.deviceToken||'尚未产生'],275,76);
  if(owners.length)out+=W([[320,306],[410,306]],{kind:'reference',label:'容器引用 R1'})+T(365,294,'引用','ld-small','middle');
  if(cpu||s.resultReady)out+=W([[320,70],[410,70]],{kind:'reference',label:'CPU 消费函数引用 R1'})+T(365,58,'引用','ld-small','middle');
  if(commit)out+=W([[320,129],[370,129],[370,160],[430,160]],{active:true,time:snap.t,start:commit.start,id:commit.id,label:'可读结果提交至 output_ids'})+T(361,184,'提交','ld-small','middle');
  if(copy)out+=W([[940,336],[940,362],[170,362],[170,copy.resultId&&s.results.some(q=>q.id===copy.resultId)?255:150]],{active:true,time:snap.t,start:copy.start,id:copy.id,label:'设备结果到对应 CPU 结果项'})+T(535,354,'设备回传到对应结果项；可读后才能提交','ld-small','middle');
  return S(out,388,'Req 身份、持有者、结果与资源分别变化')+note('引用是本地关系；脉冲表示对应结果的交接。Q0 已出队时，Q1 仍可在队列里等待。');
 }
 function pd(tl,s,rank,rid){const M=window.PDExecution,rows=M.rows(tl,s,rid),length=tl.kind==='lifecycle'?tl.model.config.inputLength:12;
  let out=T(18,26,`${rid} · 各级只计算并持有本级层的 KV`,'ld-heading');
  rows.forEach((r,i)=>{const y=55+i*92;out+=T(20,y+24,'PP'+i,'ld-heading')+T(20,y+49,r.kv?'源 KV 仍持有':'本请求不持有 KV','ld-small');
   out+=T(185,y+12,'已计算','ld-small')+B(185,y+24,235,21,'ld-lane')+B(185,y+24,235*r.computed/length,21,'good')+T(429,y+40,`${r.computed}/${length}`,'ld-value');
   out+=T(525,y+12,'已提交 / 搬运完成','ld-small')+B(525,y+24,300,21,'ld-lane')+B(525,y+24,300*r.submitted/length,21,'wait')+B(525,y+24,300*r.completed/length,21,'good')+T(835,y+40,`${r.submitted} / ${r.completed}`,'ld-value');
   const e=s.active.find(e=>e.type==='kv'&&e.rank===i);out+=W([[912,y+34],[1000,y+34]],{active:!!e,time:s.t,start:e?.start||0,id:e?.id||'',kind:'kv',label:'本级层 KV → Decode'});out+=T(950,y+17,'KV','ld-small','middle')+T(1006,y+39,'D 接收','ld-small');
   out+=T(185,y+69,`本地状态：${r.queue} · 最近 poll：${r.poll}`,'ld-small');
  });
  const flows=s.active.filter(e=>e.lane==='link'&&e.type!=='wait'&&e.frame&&(e.frame.flow||e.frame.routes?.length));
  const px=[210,550,890];for(let r=0;r<3;r++)out+=B(px[r]-110,340,220,35)+T(px[r],363,`PP${r} 本地 Req ${rid}`,'ld-value','middle');
  for(const e of flows){const f=e.frame.flow||e.frame.routes[0],from=Number(f.from.match(/(?:PP|p)(\d)/)?.[1]??-1),to=Number(f.to.match(/(?:PP|p)(\d)/)?.[1]??-1);if(from>=0&&to>=0&&from!==to){const kind=e.type==='activation'||e.type==='proxy'?'data':'control',y=from>to?407:324;out+=W([[px[from],from>to?375:340],[px[from],y],[px[to],y],[px[to],from>to?375:340]],{active:true,time:s.t,start:e.start,kind,id:e.id,label:e.title})+T((px[from]+px[to])/2,y-7,e.type==='output'?'batch 结果回流':kind==='data'?'hidden_states / residual':'rid 名单 / 控制','ld-small','middle');}}
  out+=T(20,438,flows.length?flows.map(e=>e.title).join('；'):'各级是不同的本地 Req 对象；没有跨进程迁移一个 Python 对象','ld-small');
  const lanes=[0,1,2].flatMap(r=>[{label:`PP${r} · GPU`,match:e=>e.rank===r&&e.type==='compute'},{label:`PP${r} · KV worker`,match:e=>e.rank===r&&e.type==='kv'}]);
  return S(out,460,'已计算、已提交、搬运完成与释放分开')+K(tl.events,lanes,s.t,tl.duration,{subLabel:'计算和搬运保留各自区间；CPU / 共识在模块过程展开'})+note('D 的准入由接收方另行判断；P 各级释放还须通过终态共识和本地复查。进度条为已提交的整数边界，不等同在途完成百分比。');
 }
 function consensus(tl,s){let out=T(18,25,tl.mode==='hicache'?'共同 ACK 数量到达后，仍需等待本地 event':'本地观察 → 名单汇合与回流 → 本地放行','ld-heading');
  s.stages.forEach((v,i)=>{const x=20+i*365;out+=B(x,65,325,190)+T(x+15,95,`PP${i} · 本地判断`,'ld-heading');
   if(tl.mode==='hicache'){out+=T(x+15,125,'收到计数：'+value(v.counts),'ld-value');out+=C(x+15,142,['L0','L1','L2'].map(k=>k+':'+({queued:'等待',popped:'取出',finished:'完成'})[v.acks[k]]),{width:90,gap:6});out+=T(x+15,212,'取出 ACK ≠ 本地拷贝已完成','ld-small');}
   else{out+=T(x+15,125,'观察：'+(v.observed?'已采样':'尚未采样'),'ld-small');out+=T(x+15,151,v.aggregate===null?'尚未汇合':Array.isArray(v.aggregate)?'终态候选：'+v.aggregate.join(' '):'good '+(v.aggregate.good.join(' ')||'∅')+' / bad '+(v.aggregate.bad.join(' ')||'∅'),'ld-small');out+=C(x+15,171,['R0','R1','R2'].map(id=>id+':'+({bootstrap:'留队',waiting:'准入',inflight:'持有',done:'收尾',failed:'失败'})[v.queue[id]]),{width:90,gap:6});out+=T(x+15,232,v.returned===null?'回流尚未到达':'回流已到；按本地条件处理','ld-small');}
  });
  for(const e of s.active.filter(e=>e.kind==='message')){const a=182+365*e.from,b=182+365*e.to,y=e.from>e.to?300:46;out+=W([[a,e.from>e.to?255:65],[a,y],[b,y],[b,e.from>e.to?255:65]],{active:true,time:s.t,start:e.start,id:e.id,kind:'control',label:e.label});out+=T((a+b)/2,y-8,e.back?'回流名单':tl.mode==='hicache'?'ACK 计数':'候选名单','ld-small','middle');}
  return S(out,327)+note(tl.mode==='bootstrap'?'good 取交集、bad 取并集；只有处理本地返回结果后才改变队列。':tl.mode==='release'?'终态候选还要经过本地 poll 复查；传输 worker 可继续独立运行。':'传播共同可处理的计数；本地队列顺序与 event 完成决定解除保护。');
 }
 function communication(tl,s,placement){const done=id=>s.progress[id]===1,active=id=>(s.progress[id]||0)>0&&(s.progress[id]||0)<1,send=(id,pts,label,kind='data')=>W(pts,{active:active(id),time:s.time,start:tl.phases.find(p=>p.id===id)?.start||0,label,kind,id});let out='';
  if(tl.mode==='ipc'){
   out=state(20,55,'TokenizerManager · CPU',['本地 ReqState 保留',s.dispatched?'已标记 dispatched':'尚未标记 dispatched'],355,155)+state(725,55,'Scheduler · CPU',[s.reqConstructed?'本地 Req(R1) 已构造':'等待接收并构造 Req',done('ready')?'进入 waiting_queue':'尚未进入待调度集合'],355,155);
   out+=C(40,159,['R1','101','102'],{width:62})+send('send',[[375,122],[725,122]],'编码字节：rid、input_ids、sampling','control')+T(550,105,'本机 IPC · 编码字节','ld-small','middle')+T(550,159,'接收端解码后构造自己的对象','ld-small','middle');
  }else if(tl.mode==='tensor'||tl.mode==='collective'){
   out=state(20,55,'rank 0 · CPU / GPU',[done(tl.mode==='tensor'?'sendPosts':'submit')?'CPU 已提交':'CPU 准备调用','GPU 显存中的本地数据'],330,165)+state(750,55,'rank 1 · CPU / GPU',[done(tl.mode==='tensor'?'forwardSubmit':'submit')?'CPU 已提交后续工作':'CPU 准备接收 / 调用',(tl.mode==='tensor'?s.tensorReady:s.collectiveReady)?'消费依赖满足':'消费者尚不可读取结果'],330,165);
   out+=T(550,92,placement==='local'?'机内 NVLink / PCIe':'网卡 → 交换机 → 网卡','ld-small','middle');
   if(tl.mode==='tensor'){out+=C(40,165,['h','r'],{width:62})+C(770,165,[done('hidden')?'h':'待 h',done('residual')?'r':'待 r'],{width:62});out+=send('meta',[[350,119],[750,119]],'CPU 描述','control')+T(550,111,'CPU 描述','ld-small','middle')+send('hidden',[[350,180],[750,180]],'hidden_states')+send('residual',[[350,204],[750,204]],'residual')+T(550,168,'GPU 激活 h / residual r','ld-small','middle');}
   else{const input=tl.option==='exchange'?[['A0','A1'],['B0','B1']]:[[1,2],[10,20]],output=tl.option==='reduce'?[[11,22],[11,22]]:tl.option==='gather'?[[1,2,10,20],[1,2,10,20]]:[['A0','B0'],['A1','B1']];[40,770].forEach((x,i)=>{out+=C(x,163,s.collectiveReady?output[i]:input[i],{width:48,gap:5});});out+=send('execute',[[350,165],[750,165]],'组内张量交接')+send('execute',[[750,195],[350,195]],'组内张量交接')+T(550,147,tl.option==='reduce'?'局部贡献求和':tl.option==='gather'?'按 rank 顺序收集':'按目标交换并归位','ld-small','middle');}
  }else if(tl.mode==='kv'){
   const p=window.CommunicationFlow.transferPlan(tl.option==='fragmented');out=state(20,55,'P · 本级层 KV',[done('compute')?'本块计算完成':'等待本块计算',done('queue')?'已提交 worker':'尚未提交 worker'],330,155)+state(750,55,'D · 接收位置',[s.kvWritten?'KV 已写入':'预留，尚未完成写入',s.auxWritten?'aux 已写入':'等待 aux'],330,155)+C(40,158,p.src.map(v=>'源 '+v),{width:70})+C(770,158,p.dst.map(v=>'目标 '+v),{width:80});out+=send('metadata',[[750,88],[350,88]],'目标索引','control')+T(550,76,'D → P：目标索引','ld-small','middle')+send('copy',[[350,150],[750,150]],'本层 KV','kv')+T(550,138,'GPU → 网卡 / 交换机 → GPU','ld-small','middle')+send('aux',[[350,200],[750,200]],'首 token / aux')+T(550,190,'首 token / aux','ld-small','middle');
  }else if(tl.mode==='ready'){
   out=state(20,55,'来源通知',[`独立 Success：${s.successCount} / 2`,'重复通知不增加来源数'],310,140)+state(395,55,'D 本地门槛',[`metadata room：${s.metadata}`,s.restored?'缓存恢复满足':'缓存恢复未完成',s.peersReady?'参与者共识满足':'等待其他参与者'],310,170)+state(770,55,'请求交出',[s.rejected?'身份检查失败':s.committed?'已提交到本地 Req':'仍在 transfer_queue',s.committed?'随后再由 Scheduler 选批':'Success 通知不等于可执行'],310,140);out+=send('second',[[330,104],[395,104]],'来源终态','control')+send('commit',[[705,104],[770,104]],'校验后的结果');
  }else{
   out=state(20,55,'P0 / P1 · 各自排空',[`已收到 drain ACK：${s.ackCount} / 2`,'每个来源只证明自己已排空'],330,140)+state(400,55,'D · 延迟释放',[s.released?'地址已归还':'仍持有接收地址',s.timedOut?'deadline 到达，仍缺 ACK':'等待排空条件或 deadline'],320,140)+state(770,55,'KV / metadata',[s.released?'R1 已不持有':'R1 继续持有','释放不等于擦除 GPU 字节'],310,140);out+=send('ack',[[350,112],[400,112]],'drain ACK','control')+send('free',[[720,112],[770,112]],'归还索引');
  }
  out+=T(20,260,s.phase.title,'ld-heading');
  const lanes=tl.execution.lanes.map(l=>({label:l.label,match:e=>e._lane===l.id}));const events=tl.execution.lanes.flatMap(l=>l.events.map(e=>({...e,_lane:l.id})));
  return S(out,290,tl.question)+K(events,lanes,s.time,tl.duration)+note(s.phase.summary);
 }
 function generation(tl,s){let out='';
  if(tl.mode==='speculative'){
   out=T(20,30,'一轮候选怎样变成正式输出？','ld-heading')+T(20,79,'Draft 候选')+C(240,58,s.done('draft')?tl.data.tokens:['r','·','·','·','·'],{width:62,gap:10})+T(20,137,'GPU 接受的结果')+C(240,116,s.done('select')?tl.data.output:['等待验证'],{width:86})+T(20,195,'CPU 正式输出')+C(240,174,s.output.length?s.output:['尚未提交'],{width:86})+T(20,259,`有效 KV：GPU ${s.gpuLen} · CPU 账本 ${s.cpuLen}；预留空间另行持有`,'ld-small');
   const e=s.active.find(e=>e.id==='copy'||e.id==='commit');if(e)out+=W([[800,137],[830,137],[830,190],[780,190]],{active:true,time:s.t,start:e.start,id:e.id,label:'可读结果与 CPU 提交'});
  }else if(tl.mode==='grammar'){
   out=state(20,45,'规则编译与缓存',[s.queue,'编译 ctx 可共享'],300,160)+state(395,45,'各请求独立 matcher',['R1 前缀：'+(s.r1||'空'),'R2 前缀：'+(s.r2||'空')],310,160)+state(770,45,'GPU 筛选与采样',[tl.data.bad?'编译失败，无合法 mask':s.masked?'logits 已按 mask 筛选':'等待词表 mask',s.chosen?'采样：'+s.chosen:'尚未采样'],310,160);
   out+=C(790,165,s.chosen?[s.chosen]:['·'],{width:72})+T(20,257,'编译 Future、mask 交接与 matcher 状态分别推进；无效规则不进入采样。','ld-small');
  }else{
   out=state(20,30,'候选验证',['R1 matcher 临时 accept / rollback','当前试走：'+(s.matcher||'空')],300,180)+C(36,160,tl.data.tokens,{width:44,gap:5});
   out+=state(395,30,'GPU 本轮接受结果',[s.done('sample')?'接受索引 [0, 1, 3]':'等待 logits + mask','接受结果还未必被 CPU 提交'],310,180)+C(411,160,s.done('sample')?tl.data.output:['·'],{width:70});
   out+=state(770,30,'正式状态',[s.advanced?'本轮 FSM 已推进':s.done('barrier')?'已提交前缀 {；试走后还原':'等待上一轮 FSM','CPU output_ids'],310,180)+C(786,160,s.output.length?s.output:['·'],{width:70})+T(20,257,'DFS 临时推进同一个 matcher 后还原；本轮接受结果在正式 advance / commit 路径生效。','ld-small');
  }
  return S(out,290)+note(tl.rule);
 }
 function serving(tl,snap){const s=snap.state;
  let out=state(20,30,'客户端',[s.client],280,90)+state(355,30,'Gateway · PDRouter',[s.pair,s.gateway],725,90);
  for(const [i,k]of ['P','D'].entries()){const v=s[k],x=20+i*570;out+=state(x,195,k==='P'?'Prefill P1':'Decode D1',[v.http+' · '+v.reply,v.queue,v.allocated&&!v.released?'请求仍持有 KV / 地址':v.kv],510,135);}
  const routes={cg:[[300,75],[355,75]],gP:[[515,120],[515,155],[275,155],[275,195]],gD:[[875,120],[875,195]],PD:[[530,275],[590,275]]};for(const k of Object.keys(routes))routes[k.split('').reverse().join('')]=routes[k].slice().reverse();
  for(const e of snap.active.filter(e=>e.edge)){out+=W(routes[e.edge],{active:true,time:snap.time,start:e.start,id:e.id,kind:e.kind==='data'?'kv':'control',label:e.payload});}
  return S(out,355)+F([['当前交接',snap.active.filter(e=>e.edge).map(e=>e.payload).join('；')||'本地处理或等待'],['当前结果',s.outcome]])+note('P、D 各自推进；健康响应、HTTP 收尾、请求终态和资源释放不是同一件事。');
 }

 function handoff(tl,s,M){
  let out=state(20,25,'P · Prefill',[`本块已计算 KV：${s.pKV}`,s.released?'P 本地收尾完成':'P 仍需观察传输终态与收尾条件'],440,135)+state(640,25,'D · Decode',[s.reserved?'接收目标已预留':'等待接收预算',`已接收 prompt KV：${s.dKV}`,s.committed?'y1 已提交；可继续调度':'尚未通过准入'],440,135);
  out+=W([[640,65],[460,65]],{active:s.active.some(e=>e.id==='destination'),time:s.t,kind:'control',label:'目标位置与接收信息'})+T(550,51,'接收信息','ld-small','middle')+W([[460,127],[640,127]],{active:s.active.some(e=>e.kind==='kv'||e.id==='aux'),time:s.t,kind:'kv',label:'prompt KV / aux'})+T(550,114,'KV / aux','ld-small','middle');
  out+=T(20,198,'D 输出：'+(s.dOutput.join(' · ')||'尚无')+'；D 可执行与 P 可释放分别判断','ld-small');
  return S(out,220)+K(tl.events,M.lanes.map(([id,label])=>({label,match:e=>e.lane===id})),s.t,tl.duration);
 }
 function system(progress,time,flows,stages){
  let out=state(20,20,'接入与回程',['R1 · Client / Router','文本 → IDs；输出 IDs → 文字'],260,105);
  out+=B(20,170,650,230)+T(38,197,'Prefill · PP4 × CP4 × Attention-TP4','ld-heading')+T(38,222,'每个 rank 有本地执行与 KV；每级展示 4 × 4 rank','ld-small');
  for(let r=0;r<4;r++){const x=38+r*157;out+=B(x,244,140,130,'active')+T(x+10,265,'PP'+r+' · '+stages[r],'ld-value');for(let cp=0;cp<4;cp++)for(let tp=0;tp<4;tp++)out+=B(x+12+tp*29,276+cp*21,21,15);if(r<3)out+=W([[x+140,310],[x+157,310]],{active:flows.some(e=>e.edge==='pp'+(r+1)),time,label:'hidden_states / residual'});}
  out+=state(790,170,'Decode · 逻辑接收端',[progress<.11?'预留接收位置':progress<.678?'等待 KV / metadata 核验':progress<.72?'已准入，准备首次 Decode':progress<.90?'读历史、写本轮 KV、采样':'流式回程与请求收尾','本图不展开 Decode rank'],290,140);
  const paths={requestP:[[100,125],[100,170]],requestD:[[260,125],[260,150],[1010,150],[1010,170]],receive:[[790,207],[725,207],[725,185],[670,185]],kv:[[670,275],[790,275]],metadata:[[670,300],[740,300],[740,330],[800,330],[800,310]],out:[[935,170],[935,70],[280,70]]};
  const matches={requestP:e=>e.edge==='request-p',requestD:e=>e.edge==='request-d',receive:e=>e.edge==='receive-info',kv:e=>e.edge==='kv',metadata:e=>e.edge==='metadata',out:e=>e.edge==='router-client'};
  for(const [key,path]of Object.entries(paths))out+=W(path,{active:flows.some(matches[key]),time,id:key,kind:key==='kv'?'kv':key==='out'?'data':'control'});
  out+=T(703,253,'KV','ld-small')+T(690,351,'metadata：y1 / 身份','ld-small')+T(760,143,'D → P 接收信息','ld-small')+T(370,430,'PP 交接 h / residual；KV 留在各级，另由 Sender 交给 D','ld-small','middle');
  return S(out,450)+note(flows.map(e=>e.payload).join('；')||'本地计算、等待或已完成；展开模块可查看本时刻的具体归属。');
 }
 function loop(s,mode,colorBy){const ex=s.execution;let out='';const batches=[...new Set(ex.events.map(e=>e.batch).filter(Boolean))];
  batches.forEach((batch,i)=>{const y=20+i*105,events=ex.events.filter(e=>e.batch===batch),active=ex.active.filter(e=>e.batch===batch),completed=events.filter(e=>(ex.progress[e.id]||0)>=1).length;out+=B(20,y,1060,92)+T(38,y+25,batch,'ld-heading')+T(145,y+25,active.map(e=>e.label).join(' / ')|| (completed===events.length?'本窗口活动完成':'尚未执行 / 等待本批活动'),'ld-value')+T(38,y+53,`已完成 ${completed} / ${events.length} 项活动；CPU 提交、设备执行、CPU 消费分别确认`,'ld-small');out+=B(38,y+67,1020,8,'ld-lane')+`<rect x="38" y="${y+67}" width="${1020*completed/events.length}" height="8" style="fill:var(${colorBy==='batch'?'--lj-group-'+i:'--accent'},var(--accent))"/>`;});
  return S(out,30+batches.length*105)+F([['设备侧',s.data.device],['CPU 已提交',s.data.host],['待处理结果',s.data.queue],['资源',s.data.resource]])+note(mode==='pp'?'各 stage 分别计 loop；同一个 microbatch 保持身份，slot 可复用。下方轨道保留本地轮次和跨批重叠。':colorBy==='batch'?'下方同色条块属于同一批工作；GPU 输出与 CPU 已提交状态分别观察。':'下方轨道沿用所选颜色分组；GPU 输出与 CPU 已提交状态分别观察。');
 }
 function execution(tl,v,M){const s=v.state,graph=tl.view==='graph',prefill=tl.options.origin==='prefill';let out='';
  if(graph){const d=M.graph(Number(tl.options.count),tl.options.condition,4),n=Number(tl.options.count),cap=d.eligible?(d.bucket||4):n;
   out+=state(20,25,'GPU 当前输入',[n+' 条真实请求',d.eligible?'使用已捕获图':'本轮走 Eager'],285,105)+C(38,145,Array.from({length:n},(_,i)=>'R'+(i+1)),{width:40});
   out+=state(405,25,'GPU 执行缓冲区',[s.padded?'本轮输入已写入':'等待更新输入',d.eligible?'固定地址 / 既有图桶':'保留全部真实输入'],290,105)+C(425,145,Array.from({length:cap},(_,i)=>s.padded?(i<n?'R'+(i+1):'PAD'):'·'),{width:40});
   out+=state(795,25,'GPU 输出',[s.trimmed?'只交真实行给后续采样':s.raw?'原始 logits 已生成':'等待执行完成','本图未发生 D2H 回程'],285,105)+C(815,145,s.trimmed?Array.from({length:n},(_,i)=>'R'+(i+1)):['·'],{width:40});
   out+=W([[305,90],[405,90]],{active:v.phase.id==='load',time:v.time,label:'D2D'})+W([[695,90],[795,90]],{active:v.phase.id==='execute',time:v.time,label:'模型输出'});
   return S(out,215)+note(d.reason+'；PAD 是形状补齐，不是新增用户请求。');
  }
  const ids=prefill?[11,21,41,32,52]:[41,52];
  out+=state(20,30,prefill?'CPU · 本轮输入':'GPU · FutureMap 输入',[prefill?'R1: 11,21,41 / R2: 32,52':'R1: 41 / R2: 52',prefill?'异步 H2D 交接 ID 数组':'本轮输入已在 GPU'],290,108)+C(38,160,ids,{width:40});
  out+=state(405,30,'GPU · 当前执行',[s.loaded?'执行缓冲区已更新':s.inputReady?'输入已可用':'等待本轮输入',s.kernelDone?'Attention 完成':s.stored?'本层 KV 已写入':'本层 KV 随计算更新'],290,108)+C(425,160,s.inputReady?ids:ids.map(()=>'·'),{width:40});
  out+=state(795,30,'CPU · 请求结果',[s.delivered?'R1 / R2 得到下一 token':'等待结果可读','权重和历史 KV 留在 GPU'],285,108)+C(815,160,s.delivered?['y₁','y₂']:['·','·']);
  out+=W([[310,85],[405,85]],{active:v.phase.id==='resolve',time:v.time,label:prefill?'H2D':'设备内提取'})+W([[695,85],[795,85]],{active:v.phase.id==='return',time:v.time,label:'D2H · 输出 IDs'});
  out+=T(328,68,prefill?'H2D':'设备内','ld-small')+T(718,68,'D2H','ld-small')+T(425,224,s.sampled?'GPU 已产生下一 ID，CPU 尚需消费':s.logits?'词表分数已产生，随后采样':'CPU 提交不表示设备已完成','ld-small');
  return S(out,250)+note('引用 ForwardBatch 不复制 Python 对象；图中数字为教学 token / 槽号，未测量 GPU 耗时。');
 }
 function parallel(tl,s,time){const p=id=>s.progress[id]||0,done=id=>p(id)>=1,active=id=>p(id)>0&&p(id)<1;let out='';const mode=tl.mode;
  if(mode==='pp'){
   for(let r=0;r<2;r++){const x=20+r*570;out+=state(x,20,'Stage '+r+' · 本地 Scheduler / GPU',['普通 PP · 非 PD','本级 KV 始终本地持有'],490,85);for(let n=1;n<=2;n++){const id='gpu'+r+n,y=127+(n-1)*64;out+=B(x+18,y,450,49)+T(x+30,y+20,`MB${n} · ${done('process'+r+n)?'CPU 已处理':done(id)?'GPU 已完成':active(id)?'GPU 正在执行':r&&!done('proxy'+n)?'等待输入激活':'等待本级执行'}`,'ld-value')+`<rect x="${x+18}" y="${y+42}" width="${450*p(id)}" height="5" class="ld-progress"/>`;}}
   for(let n=1;n<=2;n++)out+=W([[510,145+(n-1)*64],[590,145+(n-1)*64]],{active:active('proxy'+n),time,label:'MB'+n+' h / residual'});
  }else if(mode==='dpa'){
   for(let r=0;r<2;r++){const x=20+r*570,own=r?['b0','b1']:['a0','a1'];out+=state(x,20,'rank '+r+' · 请求 '+(r?'R2':'R1'),[done('attention'+r)?'本组 Attention 完成':'本组 Attention',done('gather')?'全局 token 行参与 MLP TP':'MLP 前汇合激活'],490,110)+C(x+18,150,done('gather')&&!done('scatter')?['a0','a1','b0','b1']:own)+T(x+18,220,done('scatter')?'取回本组行，进入下一层':'KV 留在本 Attention 组','ld-small');}
   out+=W([[510,155],[590,155]],{active:active('gather'),time,label:'a 行激活'})+W([[590,181],[510,181]],{active:active('gather'),time,label:'b 行激活'});
  }else if(mode==='pcp'){
   if(!tl.data.active)return S(state(20,20,'本轮不启用 CP',['extend 长度不足；走原模型非 CP forward'],1060,110),155);
   for(let r=0;r<2;r++){const x=20+r*570,ids=tl.data.rows[r];out+=state(x,20,'CP rank '+r,['本地 Q：'+ids.map(i=>'t'+i).join(' '),done('kv')?'Attention 可读取本轮所需 K/V':'等待 K/V 汇合与本地写缓存'],490,110)+C(x+18,152,(done('kv')?Array.from({length:8},(_,i)=>i):ids).map(i=>'KV'+i),{width:47,gap:7})+T(x+18,220,done('exit')?'出口 hidden 恢复原 token 顺序':done('attn'+r)?'本地查询计算完成':'本地查询只计算所分位置','ld-small');}
   out+=W([[510,168],[590,168]],{active:active('kvTransfer'),time,kind:'kv'})+W([[590,193],[510,193]],{active:active('kvTransfer'),time,kind:'kv'});
  }else{
   for(let r=0;r<2;r++){const x=20+r*570;out+=state(x,20,'DCP rank '+r,['KV 按 slot % 2 分散持有',done('query')?'Q heads 已汇合':'本地 Q heads'],490,110)+T(x+18,167,done('local'+r)?`局部 O=${r?6:2}，LSE₂=${r?3:1}`:'等待本地 O / LSE','ld-value')+T(x+18,207,done('lse')?(r?'权重 0.8 × 6 = 4.8':'权重 0.2 × 2 = 0.4'):'LSE 尚未汇合，不能合并局部 O','ld-small');}
   out+=W([[510,165],[590,165]],{active:active('query')||active('lseTransfer')||active('reduce'),time})+W([[590,190],[510,190]],{active:active('query')||active('lseTransfer')||active('reduce'),time});out+=T(550,261,done('reduce')?'完整贡献 5.2；各自保留对应 head 分片':'以 2 为底的 LSE 决定权重；KV 不搬走','ld-value','middle');
  }
  const lanes=tl.execution.lanes,events=lanes.flatMap(l=>l.events.map(e=>({...e,lane:l.id})));
  return S(out,285)+K(events,lanes.map(l=>({label:l.label,match:e=>e.lane===l.id})),time,tl.duration)+note(s.phase.summary);
 }
 window.DepthProjections={req,pd,consensus,communication,generation,serving,handoff,system,loop,execution,parallel,value};
})();
