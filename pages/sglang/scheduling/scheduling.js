/* One deterministic clock; data and geometry are derived from the same state. */
(()=>{'use strict';
const host=document.querySelector('[data-sf-lab]'),F=window.SchedulingFlow,M=window.SchedulingModel;if(!host||!F)return;
const $=id=>host.querySelector('#sf-'+id),NS='http://www.w3.org/2000/svg',reduced=matchMedia('(prefers-reduced-motion: reduce)'),clamp=x=>Math.max(0,Math.min(1,x));
let view=host.dataset.sfLab,timeline,time=0,snap,svg,playing=false,last=0,raf=0;
const make=(tag,a={},text)=>{const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;};
const add=(tag,a,text)=>{const e=make(tag,a,text);svg.append(e);return e;},text=(x,y,t,cls='sf-text',a={})=>add('text',{x,y,class:cls,...a},t),rect=(x,y,w,h,cls='sf-box',a={})=>add('rect',{x,y,width:w,height:h,rx:6,class:cls,...a}),path=(pts,cls='sf-wire',a={})=>add('path',{d:pts.map((p,i)=>(i?'L':'M')+p.join(' ')).join(' '),class:cls,...a});
function cell(x,y,label,cls='',w=42,a={}){const g=add('g',{transform:`translate(${x} ${y})`,class:'sf-cell '+cls,...a});g.append(make('rect',{width:w,height:30,rx:4}),make('text',{x:w/2,y:20,'text-anchor':'middle'},label));return g;}
function point(pts,u){const ds=pts.slice(1).map((p,i)=>Math.hypot(p[0]-pts[i][0],p[1]-pts[i][1]));let n=ds.reduce((a,b)=>a+b,0)*clamp(u);for(let i=0;i<ds.length;i++){if(n<=ds[i]){const f=ds[i]?n/ds[i]:0;return[pts[i][0]+(pts[i+1][0]-pts[i][0])*f,pts[i][1]+(pts[i+1][1]-pts[i][1])*f];}n-=ds[i];}return pts.at(-1);}
function move(label,pts,u,cls='',w=42,attrs={}){path(pts,'sf-wire '+cls,{'marker-end':'url(#sf-arrow)'});if(u<=0||u>=1)return;const p=point(pts,reduced.matches?0:u);cell(p[0]-w/2,p[1]-15,label,cls,w,{'data-moving':label,...attrs});}
function start(title){svg=$('svg');svg.replaceChildren(make('title',{id:'sf-svg-title'},title));svg.setAttribute('viewBox','0 0 1040 560');const defs=add('defs'),m=make('marker',{id:'sf-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:5,markerHeight:5,orient:'auto'});m.append(make('path',{d:'M0 0L10 5L0 10Z',class:'sf-arrow'}));defs.append(m);}
const active=id=>snap.phase.id===id,progress=(id,a=.08,b=.94)=>{const p=timeline.phases.find(p=>p.id===id);return p?clamp((time-p.start-p.duration*a)/(p.duration*(b-a))):0;};
function batch(){
 const d=timeline.data,s=snap.state,tx=i=>138+i*48,flatX=i=>560+i*48,decode=s.decodePrepared;
 start('ScheduleBatch：截取、展平、分配、执行和下一轮输入');
 text(24,28,'Req 历史 → 本轮输入 → 下一轮输入','sf-title');
 d.requests.forEach((r,j)=>{const y=94+j*116;rect(24,y-42,460,98,'sf-request');text(40,y-15,`${r.id} · 完整输入 ${r.tokens.length} · 前缀命中 ${r.hit}`,'sf-label');r.tokens.forEach((t,i)=>cell(tx(i),y,t,r.id+(i<r.hit?' cached':s.cut?' selected':''),38,{'data-history':r.id+':'+i}));});
 rect(24,290,460,78,'sf-work');text(40,314,'本轮片段','sf-label');text(40,340,decode?'R1: [a]      R2: [b]':s.cut?`R1: [C,D,E,F]   R2: [${d.requests[1].tokens.slice(d.requests[1].hit).join(',')}]`:'等待截取本轮输入','sf-code');
 rect(544,52,472,316,'sf-batch');text(560,78,decode?'DECODE · ScheduleBatch':'EXTEND · ScheduleBatch','sf-label');text(560,105,decode?'seq_lens = [7, 6]':`prefix_lens = [2, ${d.requests[1].hit}]`,'sf-code');text(560,131,decode?'token_per_req = 1':`extend_lens = [4, ${5-d.requests[1].hit}] · seq_lens = [6, 5]`,'sf-code');
 text(560,173,decode?'GPU input_ids':s.executed?'已执行 Extend 输入快照':'prefill_input_ids_cpu · pinned CPU','sf-code');
 if(decode)d.requests.forEach((r,i)=>cell(flatX(i),214,s.gathered?r.output:'—',r.id+(s.gathered?' selected':' pending'),38,{'data-batch-token':r.id}));
 else if(s.packed||active('pack'))d.flat.forEach((t,i)=>{if(s.packed||(progress('pack')>=.48&&t.request==='R1'))cell(flatX(i),214,t.token,t.request,38,{'data-batch-token':t.request+':'+t.position});});
 if(!s.packed&&!decode&&(!active('pack')||progress('pack')<.48))text(560,236,'等待片段展平','sf-muted');if(decode&&!s.gathered)text(675,236,'None · 等待 resolve','sf-muted');
 text(560,279,'out_cache_loc · 本轮写地址','sf-code');if(decode)d.requests.forEach((r,i)=>cell(flatX(i),302,r.nextSlot,'address',38,{'data-write-address':r.id}));else if(s.packed)d.flat.forEach((t,i)=>cell(flatX(i),302,s.allocated?t.slot:'—','address',38,{'data-write-address':i}));
 if(active('pack')){let offset=0;d.requests.forEach((r,j)=>{const n=r.tokens.length-r.hit,dest=flatX(offset)+(n-1)*24+19;const u=clamp((progress('pack')-j*.5)/.48);if(u>0&&u<1)move(r.id+': '+r.tokens.slice(r.hit).join(''),[[484,124+j*116],[510,124+j*116],[510,196],[dest,196],[dest,214]],u,r.id,100,{'data-transfer':'pack'});offset+=n;});}
 rect(24,422,310,108,'sf-work');text(40,449,'模型执行 / 结果处理','sf-label');text(40,477,s.decoded?'Decode 已处理 a、b':s.executed?'Extend 已补齐本轮 KV':'新 KV 尚未产生','sf-muted');text(40,507,s.decoded?'新采样 c、d 还没有自己的 KV':s.executed?'采样 a、b；等待下一轮使用':'准备输入与地址 ≠ 执行完成','sf-muted');
 rect(370,422,300,108,'sf-relay');text(386,449,'GPU future_map','sf-label');text(386,474,'按 req_pool_indices 中转','sf-muted');if(s.relayed)d.requests.forEach((r,i)=>cell(392+i*68,488,s.decoded?r.next:r.output,r.id,42,{'data-relay':r.id}));
 rect(706,422,310,108,'sf-request');text(722,449,'CPU Req.output_ids','sf-label');text(722,475,'结果记录 · 不直接充当 GPU 输入','sf-muted');text(722,507,s.decoded?'R1: [a,c]    R2: [b,d]':s.relayed?'R1: [a]      R2: [b]':'R1: []       R2: []','sf-code');
 if(active('relay'))d.requests.forEach((r,i)=>move(r.output,[[334,464+i*32],[413+i*68,503]],progress('relay'),r.id,42,{'data-transfer':'relay'}));
 if(active('gather'))move('[a,b]',[[670,503],[690,503],[690,394],[1006,394],[1006,254],[603,254],[603,244]],progress('gather'),'R2',60,{'data-transfer':'gather'});
 if(active('forward')||active('decode-run')){const u=progress(snap.phase.id);rect(40,539,294*(reduced.matches?(u>=1?1:0):u),4,'sf-progress');}
 return `本轮 token 数 ${decode?2:d.flat.length} · 请求数 2 · 已有 KV ${s.decoded?13:s.executed?11:2+d.requests[1].hit} 个 · 新采样 token 的 KV 在下一轮生成`;
}
function loop(){
 const s=snap.state,d=F.loopState(s.step),id=snap.phase.id,round=d.round;
 start('Scheduler：等待队列、chunked_req、running_batch 与本轮选择');
 const zones=[{x:24,title:'waiting_queue',sub:'尚未接纳的 Req',ids:d.waiting},{x:370,title:'chunked_req',sub:'未完成 Prefill 的 Req 引用',ids:d.chunk?[d.chunk]:[]},{x:716,title:'running_batch',sub:'可继续 Decode 的 Req 引用',ids:d.running}];
 zones.forEach(z=>{rect(z.x,32,300,152,'sf-zone');text(z.x+16,59,z.title,'sf-code');text(z.x+16,82,z.sub,'sf-muted');z.ids.forEach((r,i)=>{cell(z.x+18+i*82,108,r,r,64,{'data-zone':z.title,'data-request':r});text(z.x+20+i*82,161,d.finished.includes(r)?'已结束':r==='L'?'长输入':r==='S'?'短输入':'续写中','sf-muted');});if(!z.ids.length)text(z.x+16,135,'空','sf-muted');});
 if(!['choose-1','choose-2','merge','choose-decode'].includes(id))text(24,223,s.step===3?'L 上一块已保存，但还不能合入 Decode':s.step===5?'S 已结束；L 只是完成 Prefill，仍需续写':s.step===8?'结果已标记结束；下一轮过滤成员':s.step===9?'没有可运行 batch → on_idle':'get_next_batch_to_run：先试 Prefill；无新 batch 时再更新 Decode','sf-label');
 rect(24,258,992,130,'sf-batch');text(410,282,'本轮 batch · Req 引用 + 本轮输入范围','sf-label');
 const tokens=round===1?[['L','L0'],['L','L1'],['L','L2'],['L','L3']]:round===2?[['L','L4'],['L','L5'],['S','S0'],['S','S1']]:round===3?[['D','d1'],['L','l1']]:[];
 if(d.current.length){tokens.forEach(([r,t],i)=>cell(42+i*80,310,t,r+(d.history>=round?' executed':''),58,{'data-current-token':t}));text(410,330,round<3?'ForwardMode.EXTEND':'ForwardMode.DECODE','sf-code');text(410,361,round===1?'中间 chunk：不发布输出':round===2?'完成末块：L → l1；S → s1（结束）':'每请求处理一个末尾 token','sf-muted');}
 else if(!['choose-1','choose-2','choose-decode'].includes(id))text(42,335,s.step===3?'保存 chunk / 过滤 last_batch':s.step===6?'last_batch 只留下 L → 合并进 [D, L]':s.step===9?'空':'尚未选出本轮输入','sf-muted');
 if(active('choose-1'))move('L[0:4]',[[74,138],[74,241],[112,241],[112,310]],progress(id),'L',72,{'data-transfer':'choose'});
 if(active('choose-2')){const u=progress(id);if(u<.48)move('L[4:6]',[[420,138],[420,241],[112,241],[112,310]],clamp(u/.48),'L',72,{'data-transfer':'continue'});else if(s.step<4){cell(42,310,'L4','L',58);cell(122,310,'L5','L',58);}if(u>.5)move('S[0:2]',[[74,138],[74,201],[680,201],[680,241],[272,241],[272,310]],clamp((u-.5)/.5),'S',72,{'data-transfer':'choose'});}
 if(active('merge'))move('L',[[112,340],[112,406],[970,406],[970,208],[848,208],[848,138]],progress(id),'L',54,{'data-transfer':'merge'});
 if(active('choose-decode'))move('D + L',[[807,184],[807,230],[352,230],[352,365],[111,365],[111,340]],progress(id),'D',70,{'data-transfer':'decode'});
 const histories=[['回合 1 · Extend','L0 L1 L2 L3','D 等待；L 尚无可发布输出'],['回合 2 · Extend','L4 L5 + S0 S1','L、S 各有首输出；S 已结束'],['回合 3 · Decode','d1 + l1','D、L 续写；随后达到停止条件']];
 histories.forEach((h,i)=>{const x=24+i*338,done=d.history>i;rect(x,430,316,106,'sf-history'+(done?' complete':''),{'data-round':i+1,'data-complete':done});text(x+14,455,h[0],'sf-label');text(x+14,482,done?h[1]:'待执行','sf-code');text(x+14,512,done?h[2]:'示意回合，不表示实际耗时','sf-muted');});
 if(['run-1','run-2','run-decode'].includes(id)){const r=id==='run-1'?0:id==='run-2'?1:2,u=progress(id);rect(38+r*338,522,288*(reduced.matches?(u>=1?1:0):u),4,'sf-progress');}
 return `本轮输入 ${d.current.length?tokens.length:0} 个 · 等待 ${d.waiting.length} 条 · chunk 引用 ${d.chunk||'无'} · running 引用 ${d.running.join(', ')||'空'} · 已结束 ${d.finished.join(', ')||'无'}`;
}
function admission(){
 const e=timeline.data.events[snap.state.event],target=snap.phase.event,prev=snap.phase.previous,r=M.requests.find(r=>r.id===target.current),c=e.config,kind=c.resource,cap=c[kind==='input'?'input':kind==='kv'?'kv':'rows'],consumed=kind==='input'?e.usedInput:kind==='kv'?e.offset+e.protectedTokens+e.trial:e.admitted.length;
 start('PrefillAdder：候选排序、临时保护、资源检查与提交');
 text(24,30,'等待队列 · 排序只决定尝试顺序','sf-title');text(1016,30,'预算账本 ≠ 物理 KV 分配','sf-muted',{'text-anchor':'end'});
 const order=e.order;
 order.forEach((reqId,i)=>{const q=M.requests.find(r=>r.id===reqId),from=prev.order.indexOf(reqId),y=active('admit-1')?(reduced.matches?(snap.state.event>=1?72+i*92:72+from*92):72+from*92+(i-from)*92*progress('admit-1')):72+i*92;
  if(e.admitted.includes(reqId))return;const checking=target.current===reqId;rect(24,y,258,74,'sf-request '+reqId+(checking?' focused':''),{'data-waiting':reqId});text(40,y+25,reqId,'sf-label');text(40,y+51,`输入 ${q.input} · 命中 ${q.hit} · 待算 ${q.input-q.hit}`,'sf-muted');
 });
 const names={input:'本轮输入额度',kv:'KV 总预算',rows:'可用请求行'};
 rect(328,58,688,254,'sf-gate');text(346,85,'PrefillAdder · '+names[kind],'sf-label');
 text(346,113,`输入上限 ${c.input}    KV 初始预算 ${c.kv}    请求行 ${c.rows}`,'sf-muted');
 const need=r?(kind==='input'?r.input-r.hit:kind==='kv'?r.input-r.hit+r.maxNew+1:1):0;
 text(346,150,r?`${r.id} 本次需求：${need}${kind==='rows'?' 行':' token'}`:'等待候选检查','sf-title');
 const avail=cap-consumed,scale=Math.max(cap,consumed,1),barX=346,barY=183,w=624;
 rect(barX,barY,w,28,'sf-budget-bg');const amounts=kind==='kv'?[{n:e.offset,cls:'budget-used'},{n:e.protectedTokens,cls:'budget-pin'},{n:e.trial,cls:'budget-trial'}]:[{n:consumed,cls:'budget-used'}];let offset=0;
 amounts.forEach(a=>{if(a.n){rect(barX+offset/scale*w,barY,a.n/scale*w,28,'sf-'+a.cls);offset+=a.n;}});if(consumed>cap)path([[barX+cap/scale*w,barY-8],[barX+cap/scale*w,barY+36]],'sf-limit');
 text(346,237,`已记账 / 不可用 ${consumed} · 当前余额 ${avail} · 初始 ${cap}`,'sf-code');
 let rule=kind==='kv'?`检查：需求 ${need} < 余额 ${target.checkAvailable??e.available}`:kind==='input'?(e.admitted.length?'追加候选要求：输入 < 剩余输入额度':'首条无 chunk 请求可以超过输入额度'):'行数必须留得下本条请求';
 if(target.kind==='blocked')rule+=' → 本轮停止追加';if(target.kind==='stop')rule='已接纳后返回 OTHER：停止追加，不撤销当前计划';text(346,275,rule,'sf-muted');
 text(400,378,'can_run_list · 本轮计划，尚未分配或执行','sf-label');rect(24,397,992,84,'sf-batch');e.admitted.forEach((id,i)=>cell(44+i*108,423,id,id,74,{'data-admitted':id}));if(!e.admitted.length)text(44,447,'空','sf-muted');
 if(target.kind==='commit'&&!reduced.matches){const from=prev.order.indexOf(target.current),to=target.admitted.length-1;move(target.current,[[153,72+from*92+74],[304,72+from*92+74],[304,362],[81+to*108,362],[81+to*108,423]],progress(snap.phase.id),target.current,74,{'data-transfer':'commit'});}
 if(kind==='kv')text(24,518,`已记账 ${e.offset} + 持续保护 ${e.protectedTokens} + 临时保护 ${e.trial} + 余额 ${e.available} = ${c.kv}`,'sf-code');else text(24,518,'准备 → 检查 → 提交；can_run_list 的成员变化与返回值要一起看','sf-muted');
 return `已接纳 ${e.admitted.join(', ')||'无'} · 等待 ${e.waiting.join(', ')||'无'} · 临时保护 ${e.trial} · 本轮输入记账 ${e.usedInput}`;
}
function chunks(){
 const target=snap.phase.event,d=timeline.data.events[snap.state.event],plan=target.plan||{l:0,s:0,round:0},before=target.kind==='done'?snap.phase.previous:d,doneL=10-before.left,doneS=2-before.shortLeft,working=target.kind==='plan'||target.kind==='done',cap=Number(timeline.options.cap);
 start('Chunked Prefill：已有长请求续算，短请求利用剩余额度');
 text(24,30,'固定输入身份：L0…L9 与 S0、S1','sf-title');
 for(let i=0;i<10;i++)cell(166+i*77,64,'L'+i,'L'+(i<10-d.left?' executed':working&&i>=doneL&&i<doneL+plan.l?' selected':''),52,{'data-long-token':i,'data-computed':i<10-d.left});
 for(let i=0;i<10-d.left;i++)text(192+i*77,113,'已算','sf-small',{'text-anchor':'middle'});text(24,85,'长请求 L','sf-label');for(let i=0;i<2;i++)cell(166+i*77,132,'S'+i,'S'+(i<2-d.shortLeft?' executed':working&&plan.s?' selected':''),52,{'data-short-token':i,'data-computed':i<2-d.shortLeft});text(24,153,'短请求 S','sf-label');text(360,153,working?`L 本轮取 ${plan.l}；剩余 ${cap-plan.l} ${plan.s?'足够 S 的 2 个位置':'不足以接纳 S'}`:'先看 L 的剩余输入，再判断 S 能否放入','sf-muted');
 rect(24,202,992,140,'sf-batch');text(40,230,`本轮 chunk 额度 ${cap} · 拟输入 ${plan.l+plan.s}`,'sf-label');
 for(let i=0;i<cap;i++){const who=i<plan.l?'L':i<plan.l+plan.s?'S':'',token=who==='L'?'L'+(doneL+i):who==='S'?'S'+(doneS+i-plan.l):'空';cell(166+i*89,264,token,who+(target.kind==='plan'?' pending':''),60,{'data-chunk-position':i});}
 if(target.kind==='plan'){const u=reduced.matches?(snap.state.event===snap.phase.index?1:0):progress(snap.phase.id);if(plan.l)rect(163+doneL*77,60,((plan.l-1)*77+58)*u,38,'sf-selection L');if(plan.s&&u>.5)rect(163,128,135*clamp((u-.5)*2),38,'sf-selection S');}
 text(24,382,'已执行的回合 · 位置总数守恒','sf-label');
 const rounds=timeline.data.events.at(-1).history;
 rounds.forEach((p,i)=>{const x=24+i*338,complete=d.history.length>i;rect(x,402,316,120,'sf-history'+(complete?' complete':''),{'data-chunk-round':i+1,'data-complete':complete});text(x+16,428,'回合 '+(i+1),'sf-label');text(x+16,458,complete?`L: ${p.l}   +   S: ${p.s}`:'尚未执行','sf-code');text(x+16,490,complete?(d.history.slice(0,i+1).reduce((n,p)=>n+p.l,0)===10?'L 已完成 Prefill，可产生首输出':'L 仍是中间 chunk，不发布输出'):'每回合长短不表示实际耗时','sf-muted');});
 return `L 已算 ${10-d.left} / 10 · S 已算 ${2-d.shortLeft} / 2 · 已完成 ${d.history.length} 回合 · 调小 chunk 不保证短请求下一轮就能进入`;
}
function retract(){
 const s=snap.state,d=F.recovery(timeline,s),scenario=timeline.options.recovery,id=snap.phase.id,ids=Array.from({length:d.capacity},(_,i)=>i+4),slotX=i=>166+(i%7)*118,slotY=i=>260+Math.floor(i/7)*84;
 start('Retract：先回收缓存，必要时释放请求，保留文本历史并重新准入');
 const waiting=['released','waiting','aborted'].includes(d.location),zones=[{x:24,title:'running_batch',labels:[...(d.r1?['R1']:[]),...(d.row&&d.location!=='prefill'?['R2']:[])]},{x:370,title:d.location==='aborted'?'R2 已终止':d.location==='prefill'?'R2 · 恢复 Extend':'waiting_queue',labels:waiting||d.location==='prefill'?['R2']:[]},{x:716,title:'R2 已输出的文本',labels:d.output}];
 zones.forEach((z,i)=>{rect(z.x,30,300,126,'sf-zone'+(i===1&&d.location==='aborted'?' aborted':''));text(z.x+16,57,z.title,'sf-label');z.labels.forEach((v,j)=>cell(z.x+18+j*76,80,v,i===2?'R2':v,54,{'data-location':i===0?'running':i===1?d.location:'output'}));if(!z.labels.length)text(z.x+18,106,'空','sf-muted');text(z.x+16,140,i===2?'释放 KV 不会删除普通文本历史':i===1?d.location==='released'?'资源已释放，尚待回队':d.location==='prefill'?'重新通过准入并准备输入':'Req 生命周期状态':'能否继续，先检查下一步容量','sf-muted');});
 text(24,201,`固定 KV 槽位 · 容量 ${d.capacity} · 空闲 ${d.free} · 可驱逐缓存 ${d.cache}`,'sf-label');text(24,225,scenario==='resume'&&s.step>=5?'R1 缓存分为 prompt 7 + 输出后缀 2；恢复时只需再回收后缀 2':'本图 page_size=1；数字是池内地址，位置不随状态移动','sf-muted');
 const owners={};
 if(scenario==='cache'){
  for(let i=0;i<8;i++)owners[4+i]={who:'R1',label:'R1 KV'};for(let i=0;i<4;i++)owners[12+i]={who:'R2',label:'R2 KV'};
  if(s.step===0){owners[16]={who:'cache',label:'缓存'};owners[17]={who:'cache',label:'缓存'};}
  if(s.step===3){owners[16]={who:'R1',label:'R1 新 KV'};owners[17]={who:'R2',label:'R2 新 KV'};}
 }else{
  if(d.r1)for(let i=0;i<d.r1;i++)owners[4+i]={who:'R1',label:'R1 KV'};
  if(d.cache)for(let i=0;i<d.cache;i++)owners[4+i]={who:'cache',label:i<7?'R1 prompt':'R1 输出'};
  if(s.step<2)for(let i=0;i<4;i++)owners[12+i]={who:'R2',label:'R2 KV'};
  if(s.step>=6)[13,14,15,11,12].forEach((slot,i)=>owners[slot]={who:'R2'+(d.reserved?' reserved':''),label:d.reserved?'预留 '+[...'uvwx','a'][i]:'KV '+[...'uvwx','a'][i]});
 }
 text(24,287,'物理池','sf-label');text(24,311,'固定地址','sf-muted');
 ids.forEach((slot,i)=>{const o=owners[slot],x=slotX(i),y=slotY(i);rect(x,y,102,64,'sf-slot '+(o?.who||'free'),{'data-pool-slot':slot,'data-owner':o?.who||'free','data-valid':!!o&&!o.who.includes('reserved')});text(x+51,y+25,o?.label||'空闲','sf-small',{'text-anchor':'middle'});text(x+51,y+49,'槽 '+slot,'sf-code',{'text-anchor':'middle'});});
 rect(24,447,992,90,'sf-batch');text(40,472,'R2 的恢复输入与结果','sf-label');
 if(s.step>=6&&scenario==='resume'){[...'uvwx','a'].forEach((t,i)=>cell(278+i*67,484,t,'R2',44,{'data-recompute-token':t}));text(650,505,s.written?'五个历史位置已重建 KV':d.reserved?'只有地址，KV 尚未写入':'待执行','sf-muted');}
 else text(278,505,scenario==='cache'?'无需回撤；R2 继续 Decode':d.location==='aborted'?'回队失败，本分支终止':s.step>=2?'原输入 u v w x + 已有输出 a，都还保留':'普通文本历史始终可以追踪','sf-muted');
 if(active('release'))move('R2',[[192,110],[346,110],[346,175],[440,175],[440,156]],progress(id),'R2',54,{'data-transfer':'retract'});
 if(active('requeue')&&scenario!=='full')move('R2',[[440,175],[440,110]],progress(id),'R2',54,{'data-transfer':'requeue'});
 if(active('recompute')){const u=progress(id);rect(650,519,326*(reduced.matches?(s.written?1:0):u),4,'sf-progress');}
 return `R1 持有 ${d.r1} · R2 持有 ${d.r2}（其中预留未写 ${d.reserved}）· 缓存 ${d.cache} · 空闲 ${d.free} = ${d.capacity}；R2 输出 [${d.output.join(',')}]`;
}
const fmt=n=>Math.floor(n/60)+':'+String(Math.floor(n%60)).padStart(2,'0');
function render(){snap=F.at(timeline,time);const p=snap.phase;host.dataset.view=view;host.dataset.time=time.toFixed(3);host.dataset.phase=p.id;host.dataset.playing=playing;host.querySelector('[data-sf-meaning]').textContent=({batch:'浅色前缀已命中；写地址与 KV 内容分开',loop:'同色追踪同一请求；箭头表示选择或合并',admission:'色条是预算记账；空白是余额',chunks:'已算 = 输入已处理；虚线是计划输入',retract:'虚线槽位 = 空闲或预留；有效 KV 用实线'})[view];$('phase').textContent=`${p.index+1} / ${timeline.phases.length} · ${p.title}`;$('caption').textContent=p.summary;$('detail').textContent=p.detail;$('source').href=F.sourceURL(p.source);$('ledger').textContent=({batch,loop,admission,chunks,retract})[view]();$('seek').value=time;$('seek').setAttribute('aria-valuetext',fmt(time)+' '+p.title);$('time').textContent=fmt(time)+' / '+fmt(timeline.duration);$('toggle').textContent=playing?'暂停':snap.ended?'重新播放':'播放全程';$('toggle').setAttribute('aria-pressed',String(playing));$('prev').disabled=time===0;$('next').disabled=snap.ended;}
function options(){return {hit:Number($('hit')?.value||3),policy:$('policy')?.value||'fcfs',scenario:$('case')?.value||'compute',cap:Number($('cap')?.value||4),recovery:$('recovery')?.value||'resume'};}
function pause(){playing=false;last=0;cancelAnimationFrame(raf);render();}
function reset(){playing=false;last=0;cancelAnimationFrame(raf);time=0;timeline=F.build(view,options());$('seek').max=timeline.duration;host.querySelectorAll('[data-sf-view]').forEach(e=>e.setAttribute('aria-pressed',String(e.dataset.sfView===view)));host.querySelectorAll('[data-sf-settings]').forEach(e=>e.hidden=e.dataset.sfSettings!==view);render();}
function tick(now){if(!playing)return;if(last)time=Math.min(timeline.duration,time+(now-last)/1000*Number($('speed').value));last=now;if(time===timeline.duration)playing=false;render();if(playing)raf=requestAnimationFrame(tick);}
$('toggle').onclick=()=>{if(playing)return pause();if(time===timeline.duration)time=0;playing=true;last=0;render();raf=requestAnimationFrame(tick);};$('seek').oninput=()=>{time=Number($('seek').value);pause();};$('reset').onclick=reset;$('speed').onchange=()=>{last=0;};$('next').onclick=()=>{time=timeline.phases[snap.phase.index+1]?.start??timeline.duration;pause();};$('prev').onclick=()=>{time=time>snap.phase.start+.3?snap.phase.start:timeline.phases[Math.max(0,snap.phase.index-1)].start;pause();};host.querySelectorAll('[data-sf-view]').forEach(e=>e.onclick=()=>{if(view===e.dataset.sfView)return;view=e.dataset.sfView;reset();});host.querySelectorAll('.sf-settings select').forEach(e=>e.onchange=reset);document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});window.addEventListener('pagehide',pause);reduced.addEventListener('change',render);host.hidden=false;reset();
})();
