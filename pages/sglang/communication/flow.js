(function(){'use strict';
const M=window.CommunicationFlow,host=document.querySelector('[data-comm-lab]');if(!M||!host)return;
const $=id=>host.querySelector('#'+id),esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let placement='rdma',timeline,time=0,playing=false,rate=1,raf=0,last=0,phaseKey='',S;
const reduced=matchMedia('(prefers-reduced-motion: reduce)'),clamp=x=>Math.max(0,Math.min(1,x));
const p=id=>{const v=S.progress[id]||0;return reduced.matches?(v===1?1:0):v;},done=id=>p(id)===1;
const txt=(x,y,t,c='',a='start')=>`<text x="${x}" y="${y}" class="${c}" text-anchor="${a}">${esc(t)}</text>`;
const rect=(x,y,w,h,c,rx=8)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" class="${c}"/>`;
const chip=(x,y,w,t,c='')=>`<g class="chip ${c}" data-value="${esc(t)}">${rect(x,y,w,30,'cell',5)}${txt(x+w/2,y+20,t,'chip-label','middle')}</g>`;
const row=(x,y,items,c='',w=64)=>items.map((v,i)=>chip(x+i*(w+8),y,w,v,c)).join('');
const panel=(x,y,w,h,t,sub)=>rect(x,y,w,h,'panel')+txt(x+16,y+26,t,'heading')+txt(x+16,y+48,sub,'sub');
const mod=(x,y,w,t,sub,active=false)=>rect(x,y,w,54,'module'+(active?' active':''))+txt(x+12,y+23,t,'code')+txt(x+12,y+43,sub,'sub');
const bar=(x,y,w,f)=>rect(x,y,w,5,'track',2)+rect(x,y,w*clamp(f),5,'fill',2);
const title=(a,b)=>txt(26,27,a,'heading')+txt(26,49,b,'sub');
const result=(s,warn=false)=>rect(26,459,988,35,warn?'warning-strip':'strip')+txt(40,482,s,'result');
function wire(points,c='',state='idle'){return `<path d="${points.map((v,i)=>(i?'L':'M')+v.join(',')).join(' ')}" class="wire ${c}" data-state="${state}" marker-end="url(#cf-arrow-${c||'tensor'})"/>`;}
function flow(id,points,label,c='',w=100,begin=0,end=1){const f=clamp((p(id)-begin)/(end-begin)),lengths=points.slice(1).map((v,i)=>Math.hypot(v[0]-points[i][0],v[1]-points[i][1]));let distance=f*lengths.reduce((a,b)=>a+b,0),i=0;while(i<lengths.length-1&&distance>lengths[i])distance-=lengths[i++];const a=points[i],b=points[i+1],t=lengths[i]?distance/lengths[i]:0,x=a[0]+(b[0]-a[0])*t,y=a[1]+(b[1]-a[1])*t;return wire(points,c,f===0?'future':f===1?'past':'active')+(f>0&&f<1?`<g class="moving-payload" data-motion="${id}" data-x="${x}" data-y="${y}">${chip(x-w/2,y-15,w,label,c)}</g>`:'');}
function IPC(){let out=title('请求对象 → 编码字节 → 接收对象 → Scheduler.Req','同一主机 A：CPU / DRAM → 本机 IPC → CPU / DRAM；本段不使用 GPU、网卡或网络交换机');
 out+=panel(26,67,418,370,'TokenizerManager / CPU 进程','主机 A · DRAM 中保有 ReqState')+panel(596,67,418,370,'Scheduler / CPU 进程','同一主机 A · DRAM 中独立构造 Req');
 out+=txt(48,144,'TokenizedGenerateReqInput','code')+chip(48,160,91,'rid=R1','control')+chip(149,160,146,'[101,102]','control')+chip(305,160,115,'sampling','control');
 out+=mod(48,216,374,'sock_send → msgpack_encode','ZMQ socket.send(bytes)',S.phase.id==='send');
 out+=chip(48,324,179,S.dispatched?'dispatched=True':'dispatched=False',S.dispatched?'ok':'empty')+txt(48,384,'ReqState 仍在这里等待用户响应','sub');
 out+=mod(618,143,374,'SchedulerRequestReceiver','sock_recv(NOBLOCK) → msgpack_decode',S.phase.id==='receive');
 if(done('send'))out+=chip(618,216,152,done('receive')?'已解码 R1':'消息待拉取',done('receive')?'control':'empty');
 out+=mod(618,290,374,'handle_generate_request','读取字段 → 构造 Req → 登记待调度',S.phase.id==='req');
 out+=chip(618,379,177,done('req')?'Req(R1)':'等待 Req',done('req')?'ok':'empty')+txt(813,400,done('ready')?'等待选批':'尚未进入执行','sub');
 out+=flow('send',[[422,242],[516,242],[516,170],[618,170]],'msgpack bytes','control',116);
 out+=flow('receive',[[828,197],[828,268],[805,268],[805,290]],'R1 / ids','control',92);
 out+=flow('req',[[706,344],[706,379]],'Req(R1)','control',86);
 return out+result(done('req')?'相同 rid 连接两个进程的独立状态；Scheduler 已获得 Req，尚未开始 GPU 计算。':S.dispatched?'发送侧 dispatched=True 不是接收 ACK；接收端何时拉取由它的循环决定。':'请求身份、token IDs 和参数跨过进程边界；发送侧状态对象保留在本地。');}
function hardwareNode(id,label,detail=''){
 const n=M.hardware.nodes[id];return `<g data-hardware="${id}">${rect(n.x,n.y,n.w,n.h,'hw-node '+(id.startsWith('gpu')?'hw-gpu':id.startsWith('cpu')?'hw-cpu':'hw-network'))}${txt(n.x+12,n.y+23,label,id.startsWith('nic')||id==='switch'?'code':'heading')}${detail?txt(n.x+12,n.y+44,detail,'sub'):''}</g>`;
}
function hardwareBase(left,right){
 let out=placement==='local'?rect(20,62,1000,476,'hw-host')+txt(36,88,'同一主机 · 两个进程 / 两块 GPU','heading'):rect(20,62,380,476,'hw-host')+rect(640,62,380,476,'hw-host')+txt(36,88,'主机 A · '+left,'heading')+txt(656,88,'主机 B · '+right,'heading');
 for(const i of [0,1]){out+=hardwareNode('cpu'+i,'CPU 执行 / 主机内存 DRAM');out+=hardwareNode('gpu'+i,'GPU '+i+' / 显存 HBM');}
 return out;
}
function hardwareNetwork(){
 if(placement==='local')return `<g data-hardware="p2p">${rect(450,258,140,56,'hw-node hw-network')}${txt(520,287,'NVLink / PCIe','code','middle')}</g>`+txt(520,240,'本例不经过网卡和网络交换机','sub','middle');
 return ['nic0','nic1'].map(id=>{const n=M.hardware.nodes[id];return `<g data-hardware="${id}">${rect(n.x,n.y,n.w,n.h,'hw-node hw-network')}${txt(n.x+n.w/2,n.y+33,'RDMA 网卡','code','middle')}</g>`;}).join('')+`<g data-hardware="switch">${rect(470,258,100,56,'hw-node hw-network')}${txt(520,291,'网络交换机','code','middle')}</g>`+txt(520,240,'本例：跨机 RoCE / IB 网络','sub','middle')+txt(260,335,'PCIe','sub')+txt(758,335,'PCIe','sub');
}
function hardwareFlow(id,kind,label,color,reverse=false,begin=0,end=1){
 const route=M.hardware.route(kind,placement,reverse),f=clamp((p(id)-begin)/(end-begin)),pts=route.points,lengths=pts.slice(1).map((v,i)=>Math.hypot(v[0]-pts[i][0],v[1]-pts[i][1]));let distance=f*lengths.reduce((a,b)=>a+b,0),i=0;while(i<lengths.length-1&&distance>lengths[i])distance-=lengths[i++];const t=distance/lengths[i],x=pts[i][0]+(pts[i+1][0]-pts[i][0])*t,y=pts[i][1]+(pts[i+1][1]-pts[i][1])*t;
 return `<g class="hardware-route" data-plane="${route.plane}" data-residency="${route.memory}" data-nodes="${route.nodes.join(' ')}" data-payload="${esc(label)}">${wire(pts,color,f===0?'future':f===1?'past':'active')}${f>0&&f<1?`<g class="moving-payload ${color}" data-motion="${id}" data-x="${x}" data-y="${y}"><title>${esc(label)}</title><circle cx="${x}" cy="${y}" r="6" class="hw-packet"/></g>`:''}</g>`;
}
function hardwareResult(text){return rect(20,551,1000,36,'strip')+txt(34,574,text,'result');}
function hwCPU(i,line,content,foot){const x=i?656:36;return txt(x+12,155,line,'code')+chip(x+12,168,324,content,'control')+txt(x+12,211,foot,'sub');}
function tensorBuffer(i,y,name,letter,allocated,received,receiving,source=false){const x=i?668:48,cls=source?'':received?'ok':'empty';let out=`<g class="tensor-buffer ${cls}" data-buffer="${name}" data-side="${source?'source':'receiver'}" data-allocated="${allocated}" data-ready="${received}">${rect(x,y,324,48,'cell')}${txt(x+10,y+18,name,'code')}${txt(x+10,y+37,source?'FP16 [2,4]':received?'接收依赖满足':receiving?'接收中':allocated?'已分配 · 未可读':'尚未分配','sub')}`;
 if(allocated)for(let j=0;j<8;j++){const cx=x+164+(j%4)*37,cy=y+5+Math.floor(j/4)*21,v=source||received?letter+j:'·';out+=`<g class="tensor-element ${cls}" data-element="${v}">${rect(cx,cy,32,18,'cell',2)}${txt(cx+16,cy+13,v,'chip-label','middle')}</g>`;}return out+'</g>';}
function Tensor(){
 let out=title('PP：CPU 组织描述与调用，GPU 间搬运激活','紫色虚线是控制/描述；蓝色实线是 hidden_states、residual；主机放置是所选教学条件');
 out+=hardwareBase('PP Stage 0 / rank 0','PP Stage 1 / rank 1');
 out+=hwCPU(0,'Scheduler → send_tensor_dict','DRAM: proxy / cuda / FP16 / [2,4]',done('sendPosts')?'isend 已返回；work 保有发送引用':'依次 isend 长度 / 描述 / h / r');
 out+=hwCPU(1,'recv_tensor_dict → typed inbox',done('meta')?'DRAM: 描述已解码 · proxy':'DRAM: 等待描述',done('forwardSubmit')?(done('residual')?'GPU forward 可以读取':'forward 已入队；GPU 等接收依赖'):'按描述逐项 empty / irecv / wait');
 out+=tensorBuffer(0,390,'hidden_states','h',true,true,false,true)+tensorBuffer(0,448,'residual','r',true,true,false,true);
 out+=tensorBuffer(1,390,'hidden_states','h',done('rxHidden'),done('hidden'),p('hidden')>0&&!done('hidden'))+tensorBuffer(1,448,'residual','r',done('rxResidual'),done('residual'),p('residual')>0&&!done('residual'));
 out+=bar(668,439,324,p('hidden'))+bar(668,498,324,p('residual'));
 out+=flow('split',[[82,220],[82,354]],'提交','control',50)+flow('allocate',[[950,220],[950,354]],'分配','control',50);
 out+=hardwareNetwork();
 out+=hardwareFlow('meta','control','size / pickle 描述','control',false,0,1)+hardwareFlow('hidden','gpu','h0…h7 · FP16 · 16 B','',false)+hardwareFlow('residual','gpu','r0…r7 · FP16 · 16 B','',false);
 const payload=S.phase.id==='meta'?(p('meta')<.25?'长度 size':'pickle 描述（无激活值）'):p('hidden')>0&&!done('hidden')?'h0…h7 · FP16 · 16 B':S.phase.id==='residual'?'r0…r7 · FP16 · 16 B':S.phase.id==='allocate'?'接收 GPU 分配 empty':done('usable')?'接收依赖满足':'CPU 保有张量引用';
 out+=txt(520,119,'当前动作 / 载荷','sub','middle')+txt(520,143,payload,'code','middle')+txt(520,334,placement==='local'?'GPU P2P · 载荷不经 CPU 内存':'GPUDirect RDMA · 载荷不经 CPU 内存','sub','middle');
 return out+hardwareResult(done('residual')?'CPU 建立接收与 forward 的 stream 依赖；GPU 读取本机 HBM 中的激活，不读取对端权重或 KV。':placement==='local'?'CPU 提交调用和元数据；GPU 激活走蓝线。本例经机内 P2P 传输，不使用外部网络。':'CPU 提交调用和元数据；GPU 激活走蓝线。网卡/交换机负责传输，不执行 Scheduler 或模型层。');
}
function Collective(){const kind=timeline.option,reduce=kind==='reduce',gather=kind==='gather',method=reduce?'all_reduce':gather?'all_gather':'all_to_all_single';
 let out=title('Collective：CPU 提交，GPU 张量按组内语义交换','通信组已初始化；本图展示一次张量操作。交换机只表示网络转发，不表示代替 GPU 求和');out+=hardwareBase('TP rank 0','TP rank 1');
 for(let i=0;i<2;i++){const x=i?668:48;out+=hwCPU(i,'GroupCoordinator → PyNCCL',method+'(ptr, count, dtype, stream)',done('submit')?'CPU 调用已返回，设备仍按依赖执行':'输入/输出地址指向本机 GPU 显存');
 out+=row(x,390,[done('queued')?'前序完成':'前序任务',done('execute')?'通信完成':p('execute')>0?'通信执行':'通信排队',done('consume')?'消费完成':'消费者等待'],'empty',102);
 const input=reduce&&done('execute')?[11,22]:kind==='exchange'?(i?['B0','B1']:['A0','A1']):(i?[10,20]:[1,2]);
 if(reduce){out+=txt(x,444,'HBM: input / output 同一地址','sub')+row(x+56,458,input,done('execute')?'ok':'',100);}
 else{out+=txt(x,440,'input','sub')+row(x+65,425,input,'',92);const values=done('execute')?(gather?[1,2,10,20]:i?['A1','B1']:['A0','B0']):Array(gather?4:2).fill('待写');out+=txt(x,488,'output','sub')+row(x+65,473,values,done('execute')?'ok':'empty',gather?57:110);}
 out+=flow('submit',[[i?950:82,220],[i?950:82,354]],'提交','control',50);}
 out+=hardwareNetwork();out+=hardwareFlow('execute','gpu',reduce?'贡献 [1,2]':gather?'[1,2]':'A1','',false,0,.9)+hardwareFlow('execute','gpu',reduce?'贡献 [10,20]':gather?'[10,20]':'B0','',true,.1,1);
 out+=txt(520,119,'蓝线：双向张量交换','sub','middle')+txt(520,143,reduce?'GPU 侧完成规约':gather?'按 rank 拼接输出':'按目的 rank 交换分块','code','middle')+txt(520,334,placement==='local'?'机内 GPU P2P':'GPUDirect RDMA · 无主机中转','sub','middle');
 return out+hardwareResult('提交调用、设备通信与消费者执行是三件事；图中路径为部署示例，不指定 NCCL 的 ring/tree 或网络分包。');
}
function KV(){const plan=timeline.plan;let out=title('KV 交接：控制消息、GPU 数据、CPU aux 各走什么路径？','本例：跨机 Mooncake / RDMA、GPUDirect 可用；仅放大一层 K buffer，16 B / 页为教学缩尺');out+=hardwareBase('Prefill / P','Decode / D');
 out+=txt(48,155,done('queue')?'Scheduler: 入队已返回，可继续':'Scheduler: 计算交接 → 任务入队','code')+txt(48,179,done('notify')?'worker: 本块完成':p('aux')>0&&!done('aux')?'worker: 同步等待 aux':p('copy')>0&&!done('copy')?'worker: batch_transfer_sync 等待':done('queue')?'worker: 地址计划 / 末块处理':'worker: 等待队列任务','code')+txt(48,208,done('compute')?'DRAM aux: output_id=320 / room=42':'DRAM aux: 等待末块结果','sub');
 out+=hwCPU(1,'Receiver / metadata buffer',`DRAM: 目标页 [${plan.dst}] 已预留`,done('aux')?'aux 已写入: output_id=320 / room=42':'aux: CPU 内存中尚待写入');
 for(let j=0;j<3;j++){const y=389+j*39;out+=chip(48,y,92,done('compute')?`${'ABC'[j]} · 16 B`:'尚待计算',done('compute')?'kv':'empty')+txt(152,y+20,`page ${plan.src[j]} · ${1000+plan.src[j]*16}`,'code');out+=chip(668,y,125,done('copy')?`${'ABC'[j]} · 已写入`:'预留 · 未写入',done('copy')?'kv':'empty')+txt(805,y+20,`page ${plan.dst[j]} · ${2000+plan.dst[j]*16}`,'code');}
 out+=flow('compute',[[82,220],[82,354]],'计算','control',50);
 out+=hardwareNetwork()+hardwareFlow('metadata','control',`目标索引 [${plan.dst}]`,'control',true)+hardwareFlow('aux','aux','aux 数据 · room / output_id','aux')+hardwareFlow('notify','control','Success · 终态通知','control');
 out+=txt(520,119,'当前动作 / 载荷','sub','middle')+txt(520,143,done('queue')?'发送队列 → 地址计划':'等待计算与任务入队','code','middle');
 out+=txt(520,378,`${plan.blocks.length} 段 · 48 B · 地址计划`,'code','middle');
 plan.blocks.forEach((b,j)=>{const label='ABC'.slice(plan.src.indexOf(b.src),plan.src.indexOf(b.src)+b.count).split('').join('+')+' · '+b.bytes+' B';out+=`<g data-transfer-block="${j}" data-pages="${b.count}" data-bytes="${b.bytes}" data-src="${b.srcAddr}" data-dst="${b.dstAddr}">${txt(520,405+j*25,`${b.srcAddr} → ${b.dstAddr} : ${b.bytes} B`,'code','middle')}`+hardwareFlow('copy','gpu',label,'kv',false,j*.08,1)+'</g>';});
 const label=S.phase.id==='metadata'?'控制：目标索引 D → P':S.phase.id==='copy'?'数据：K 向量 P → D':S.phase.id==='aux'?'数据：aux（CPU 内存 → CPU 内存）':S.phase.id==='notify'?'控制：Success 通知 P → D':'GPU 数据与 CPU aux 分开搬运';out+=txt(520,334,label,'sub','middle');
 return out+hardwareResult(done('notify')?'Success 是控制通知；D 还需检查 aux、参与者与 room 身份。KV 保存在 D 的 GPU 显存。':'金色实线搬 K 数据；粉色实线搬 CPU aux；紫色虚线传地址与状态。注册内存不等于为请求分配。');
}
function Ready(){let out=title('P 报告各自进度；D 汇合本地条件，再交给 Scheduler','左：P 来源通知　中：D0 本地内存与 gate　右：D 参与者汇合与 Req；各条件独立准备');
 out+=panel(26,67,288,370,'P 端 / 来源 CPU','独立发送；抵达 D 才计入成功来源')+panel(369,67,328,370,'D0 / 本地 CPU 与 DRAM','接收记录、HiCache 与元数据检查')+panel(752,67,262,370,'D 端 / 汇合与提交','D1 是另一个参与者，非另一请求');
 const first=done('first'),second=done('second'),meta=done('metadata');
 out+=chip(46,139,248,p('first')>0?'P0 已发出 Success':'P0 通知尚未发出',first?'control':'empty')+chip(46,205,248,p('second')>0?'P1 已发出 Success':'P1 通知尚未发出',second?'control':'empty');
 out+=bar(46,176,248,p('first'))+bar(46,242,248,p('second'))+chip(46,280,248,p('duplicate')>0?'P0 重发同一个来源':'可能重发同源通知',done('duplicate')?'control':'empty');
 out+=txt(46,352,'这里只发送状态，不携带 KV 字节','sub')+txt(46,378,'GPU KV 数据路径见「KV 搬运」','sub');
 out+=chip(389,139,288,`成功来源 {${second?'0,1':first?'0':''}} · ${S.successCount}/2`,second?'ok':'empty');
 out+=chip(389,201,288,`metadata room = ${S.metadata}`,meta?'aux':'empty')+bar(389,237,288,p('metadata'));
 out+=chip(389,263,288,timeline.option==='cache'?(S.restored?'HiCache READY · 放行':'HiCache PENDING · 同时恢复'):'HiCache: 本例无需恢复',S.restored?'ok':'empty');
 if(timeline.option==='cache')out+=bar(389,299,288,p('cache'));
 out+=chip(389,325,288,!second?'poll: Transferring':!S.restored?'poll 仍为 Transferring':meta?'metadata gate: Success':'room=0 → Transferring',second&&meta&&S.restored?'ok':'empty');
 out+=txt(389,401,'先检查恢复，再检查 metadata 非零','sub');
 out+=chip(772,139,222,done('peerLocal')?'D1 本地已就绪':'D1 独立准备中',done('peerLocal')?'ok':'empty');
 out+=chip(772,193,222,S.peersReady?'D MIN: Success':'D MIN: Transferring',S.peersReady?'ok':'empty');
 out+=chip(772,247,222,done('commit')?(S.rejected?'99 ≠ 42 · 拒绝':'42 = 42 · 校验通过'):'等待 room 身份校验',S.rejected?'bad':done('commit')?'ok':'empty');
 out+=chip(772,301,222,S.committed?'Req.output_ids=[320]':'Req.output_ids=[]',S.committed?'ok':'empty');
 out+=chip(772,365,222,S.rejected?'ABORTED':S.committed?'可返回给调度器':'留在 transfer_queue',S.rejected?'bad':S.committed?'ok':'empty');
 out+=flow('first',[[294,154],[389,154]],'P0','control',44)+flow('duplicate',[[294,295],[345,295],[345,154],[389,154]],'P0','control',44)+flow('second',[[294,220],[334,220],[334,154],[389,154]],'P1','control',44);
 out+=flow('peers',[[677,340],[724,340],[724,208],[772,208]],'poll','control',44)+flow('commit',[[883,223],[883,247]],'校验','control',44);
 return out+result(S.rejected?'非零 room 通过早期 gate，身份不符仍拒绝提交；P 的通知不能替 D 决定请求可用。':S.committed?'独立条件均满足，MIN 与身份检查通过：写入 aux 中的 output_id，返回 Req 供后续调度。':'准备过程可以重叠；D 按 HiCache → metadata → MIN → room 身份的检查顺序汇合条件。',S.rejected);}
function Release(){let out=title('P 排空在途工作；D 持有地址，按 ACK 或 deadline 释放','左：P 的两个来源后台　中：D 的 GPU / DRAM 资源　右：D CPU 延迟释放队列');
 out+=panel(26,67,330,370,'P 端 / 传输后台','两个来源分别停止准入并排空')+panel(414,67,260,370,'D 端 / 持有资源','请求结束不等于内存可复用')+panel(732,67,282,370,'D CPU / 释放队列','resolve_deferred_releases');
 const drained=S.ackCount===2;
 out+=chip(48,139,286,done('ack')?'P0 drain ACK 已到 D':done('drain0')?'P0 排空，ACK 在途':'P0 在途工作排空中',done('ack')?'ok':'empty')+bar(48,176,286,p('drain0'));
 out+=chip(48,221,286,drained?'P1 drain ACK 已到 D':done('drain1')?'P1 排空，ACK 在途':'P1 独立排空，尚无 ACK',drained?'ok':'empty')+bar(48,258,286,timeline.option==='timeout'?0:p('drain1'));
 out+=txt(48,321,'每个来源只为自己的排空条件作证','sub')+txt(48,348,timeline.option==='timeout'?'P1 未确认，不能显示排空完成':'P1 可以晚于 P0 完成','sub');
 out+=chip(434,139,220,'R1 · ABORTED','bad')+txt(434,207,'GPU HBM / KV 接收区域','code')+row(434,225,['A','B','C'],S.released?'stale':'kv',66)+txt(434,288,S.released?'旧字节失效 · 分配权已归还':'仍被 R1 持有，不得复用','sub');
 out+=chip(434,319,220,S.released?'metadata room=0':'metadata index 仍占用',S.released?'ok':'empty')+txt(434,377,S.released?'KV / metadata 索引已归还':'DRAM 与 HBM 都保持所属权','sub');
 out+=chip(752,139,242,done('defer')?'进入 deferred_releases':'等待登记释放条件',done('defer')?'control':'empty');
 out+=chip(752,193,242,`独立来源 ${S.ackCount} / 2`,drained?'ok':'empty');
 out+=chip(752,247,242,done('resolve')?(S.timedOut?'deadline 到达':'drained=True'):'持有地址，轮询 ACK / deadline',S.timedOut?'warn':drained?'ok':'empty');
 out+=bar(752,291,242,p('resolve'))+mod(752,327,242,'_do_release',S.released?'资源归还完成':'条件满足后再执行',S.phase.id==='free');
 out+=flow('ack',[[334,154],[380,154],[380,423],[704,423],[704,208],[752,208]],'ACK0','control',44);
 if(timeline.option!=='timeout')out+=flow('resolve',[[334,236],[380,236],[380,423],[704,423],[704,208],[752,208]],'ACK1','control',44);
 out+=flow('free',[[752,354],[704,354],[704,423],[674,423]],'归还','ok',44);
 return out+result(S.released?(S.timedOut?'源码在超时后仍释放；只有 1/2 ACK，不代表远端已全部排空，也不表示 GPU 字节已擦除。':'收到 2/2 drain ACK 后归还 KV 与 metadata 索引；两路排空独立，D 持续检查完成条件。'):'P0 / P1 的排空与 D 的等待同时推进；D 等的是这些地址的条件，不是让全系统停止。',S.timedOut);}
function executionLanes(y){const lanes=timeline.execution.lanes,x0=170,width=840,scale=width/timeline.duration;let out=`<g class="execution-lanes" data-clock="${time.toFixed(3)}">`+txt(26,y+17,'同一时钟下的执行轨道','code')+txt(1014,y+17,'示例交错 / 非实测；重叠可并行，虚框为等待','sub','end');
 lanes.forEach((lane,i)=>{const top=y+29+i*24;out+=txt(26,top+14,lane.label,'lane-label')+rect(x0,top,width,18,'lane-track',3);for(const e of lane.events){const x=x0+e.start*scale,w=(e.end-e.start)*scale,progress=clamp((time-e.start)/(e.end-e.start)),active=time>=e.start&&time<e.end;out+=`<g class="execution-event ${e.kind}" data-activity="${e.id}" data-owner="${lane.id}" data-active="${active}" data-start="${e.start}" data-end="${e.end}"><title>${esc(lane.label+'：'+e.label)}</title>${rect(x,top,w,18,'event-slot',3)}${rect(x,top,w*progress,18,'event-progress',3)}`;if(w>44)out+=txt(x+w/2,top+13,e.label,'event-label','middle');out+='</g>';}});
 const cx=x0+time*scale;out+=`<path d="M${cx},${y+25}V${y+29+lanes.length*24}" class="time-cursor"/>`;return out+'</g>';}
const renderers={ipc:IPC,tensor:Tensor,collective:Collective,kv:KV,ready:Ready,release:Release};
function render(){S=M.state(timeline,time);const transport=['tensor','collective','kv'].includes(timeline.mode),base=transport?600:510,total=base+40+timeline.execution.lanes.length*24;$('comm-canvas').setAttribute('viewBox',`0 0 1040 ${total}`);$('comm-drawing').innerHTML=renderers[timeline.mode]()+executionLanes(base);$('comm-canvas').dataset.mode=timeline.mode;$('comm-canvas').dataset.phase=S.phase.id;$('comm-canvas').dataset.time=time.toFixed(3);$('comm-time').value=time;$('comm-time').setAttribute('aria-valuetext',`${time.toFixed(1)} 秒，${S.phase.title}`);$('comm-clock').textContent=`${time.toFixed(1)} / ${timeline.duration.toFixed(1)} s`;$('comm-toggle').textContent=playing?'暂停':S.finished?'重新播放':'播放全程';$('comm-toggle').setAttribute('aria-pressed',String(playing));$('comm-previous').disabled=time===0;$('comm-next').disabled=S.finished;
 const key=timeline.mode+timeline.option+S.phase.id;if(key!==phaseKey){phaseKey=key;$('comm-phase').textContent=`讲解焦点 ${S.phase.index+1} / ${timeline.phases.length} · ${S.phase.title}`;$('comm-caption').textContent=S.phase.summary;$('comm-detail').textContent=S.phase.detail;$('comm-sources').innerHTML=S.phase.refs.map(k=>`<a href="${M.url(k)}" target="_blank" rel="noopener">${esc(M.sources[k][0])}:${M.sources[k][1]} · ${esc(M.sources[k][2].replace('def ','').replace('(',''))} ↗</a>`).join('');$('comm-svg-title').textContent=timeline.question;$('comm-svg-desc').textContent=S.phase.title+'。'+S.phase.summary;host.querySelectorAll('[data-phase]').forEach(e=>e.setAttribute('aria-current',String(Number(e.dataset.phase)===S.phase.index)));}}
function pause(){playing=false;cancelAnimationFrame(raf);last=0;}
function tick(now){if(!playing)return;if(last)time=Math.min(timeline.duration,time+(now-last)/1000*rate);last=now;if(time>=timeline.duration)pause();render();if(playing)raf=requestAnimationFrame(tick);}
function choose(mode,option,where=placement){pause();placement=['tensor','collective'].includes(mode)&&where==='local'?'local':'rdma';timeline=M.build(mode,option);$('comm-placement').value=placement;host.querySelector('.cf-placement').hidden=!['tensor','collective'].includes(mode);$('comm-hardware-note').textContent=['tensor','collective'].includes(mode)?(placement==='local'?'教学部署：两个进程在同一主机，CUDA P2P 可用；数据经 NVLink / PCIe，图中不使用外部网络。':'教学部署：两个 rank 分处两台主机；本例指定 NCCL 走可用的 GPUDirect RDMA，网卡与 GPU 经 PCIe 互访（非部署探测结果）。'):mode==='kv'?'教学部署：P/D 跨机，Mooncake 使用 RDMA 且 GPUDirect 可用；普通 CUDA aux 位于 CPU DRAM。':mode==='ipc'?'本机请求 IPC：两处是进程边界，不表示两颗物理 CPU。':mode==='ready'?'P 来源通知与 D 本地条件独立推进；MIN 只交换 CPU poll 状态。图中 D0 / D1 的物理放置未指定。':'P0 / P1 各自排空并回报 ACK；D CPU 轮询条件，持有 GPU 接收地址与 CPU 元数据槽。';time=0;phaseKey='';$('comm-question').textContent=timeline.question;$('comm-config').textContent=timeline.config;$('comm-boundary').textContent=timeline.boundary;$('comm-option').innerHTML=M.options[mode].map(([v,n])=>`<option value="${v}">${esc(n)}</option>`).join('');$('comm-option').value=timeline.option;host.querySelector('.cf-settings').hidden=M.options[mode].length<2;$('comm-time').max=timeline.duration;host.querySelector('.cf-events').innerHTML=timeline.phases.map(p=>`<button type="button" data-phase="${p.index}">${p.index+1}. ${esc(p.title)}</button>`).join('');host.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode)));const u=new URL(location.href);u.searchParams.set('mode',mode);u.searchParams.set('placement',placement);u.searchParams.set('condition',timeline.option);u.searchParams.delete('step');u.searchParams.delete('t');history.replaceState(null,'',u);render();}
host.querySelector('.cf-modes').innerHTML=Object.entries(M.catalog).map(([id,c],i)=>`<button type="button" data-mode="${id}" aria-pressed="false"><span aria-hidden="true">0${i+1} · </span>${c[0]}</button>`).join('');
$('comm-canvas').querySelector('defs').innerHTML=Object.entries({tensor:'#85c8f2',control:'#b8b4fb',kv:'#edc879',ok:'#79dbc0',aux:'#e7a4cb'}).map(([k,c])=>`<marker id="cf-arrow-${k}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill="${c}"/></marker>`).join('');
host.addEventListener('click',e=>{const mode=e.target.closest('[data-mode]'),phase=e.target.closest('[data-phase]');if(mode)choose(mode.dataset.mode);if(phase){pause();time=timeline.phases[Number(phase.dataset.phase)].start;render();}});
$('comm-placement').addEventListener('change',e=>choose(timeline.mode,timeline.option,e.target.value));$('comm-option').addEventListener('change',e=>choose(timeline.mode,e.target.value));$('comm-toggle').addEventListener('click',()=>{if(playing)pause();else{if(time>=timeline.duration)time=0;playing=true;last=0;const board=$('comm-canvas').getBoundingClientRect();if(board.height<innerHeight-170&&board.bottom>innerHeight)host.querySelector('.cf-controls').scrollIntoView({block:'start',behavior:'instant'});raf=requestAnimationFrame(tick);}render();});$('comm-speed').addEventListener('change',e=>{rate=Number(e.target.value);last=0;});$('comm-time').addEventListener('input',e=>{pause();time=Number(e.target.value);render();});$('comm-reset').addEventListener('click',()=>{pause();time=0;render();});$('comm-next').addEventListener('click',()=>{pause();time=timeline.phases.find(x=>x.start>time+.001)?.start??timeline.duration;render();});$('comm-previous').addEventListener('click',()=>{pause();time=[...timeline.phases].reverse().find(x=>x.start<time-.001)?.start??0;render();});
reduced.addEventListener('change',render);document.addEventListener('visibilitychange',()=>{if(document.hidden){pause();render();}});window.addEventListener('pagehide',pause);
const params=new URLSearchParams(location.search),mode=M.catalog[params.get('mode')]?params.get('mode'):'ipc',option=M.options[mode].some(x=>x[0]===params.get('condition'))?params.get('condition'):undefined;choose(mode,option,params.get('placement'));host.hidden=false;
// The fragment target was inside a hidden lab during native anchor resolution.
if(location.hash==='#comm-title')requestAnimationFrame(()=>$('comm-title').scrollIntoView({block:'start',behavior:'instant'}));
})();
