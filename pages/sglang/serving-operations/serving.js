(()=>{
'use strict';const M=ServingModel,$=s=>document.querySelector(s),lab=$('[data-serving-lab]');let mode='health',step=0,timer=null;
const p=()=>({health:$('#health-kind').value,admission:$('#admission-kind').value,workers:$('#worker-kind').value,capacity:Number($('#queue-capacity').value),cancel:$('#cancel-kind').value,drain:$('#drain-kind').value});
const last=()=>mode==='health'?(p().health==='startup'?4:2):mode==='admission'?(p().admission==='route'?3:5):mode==='cancel'?4:p().drain==='arrivals'?3:5;
const chip=(text,cls='')=>`<span class="ops-chip ${cls}">${text}</span>`;
const chips=(items,empty='空')=>`<div class="ops-chips">${items.length?items.map(x=>chip(x)).join(''):chip(empty,'empty')}</div>`;
const node=(title,body,note='',cls='')=>`<div class="ops-node ${cls}"><h3>${title}</h3>${body}${note?`<small>${note}</small>`:''}</div>`;
const badge=(text,cls='')=>`<div class="ops-center"><span class="ops-badge ${cls}">${text}</span></div>`;
function wire(paths,active,label){const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;return `<p class="ops-label">${label}</p><svg class="ops-wire" viewBox="0 0 600 72" preserveAspectRatio="none" aria-label="${label}" role="img">${paths.map(([x1,x2,reverse=false])=>{const y1=reverse?68:4,y2=reverse?4:68,d=`M ${x1} ${y1} C ${x1} 36 ${x2} 36 ${x2} ${y2}`;return `<path d="${d}" class="${active?'live':''}"/>${active?`<circle cx="${reduced?x2:0}" cy="${reduced?y2:0}" r="4">${reduced?'':`<animateMotion dur="1.6s" repeatCount="indefinite" path="${d}"/>`}</circle>`:''}`;}).join('')}</svg>`;}
function health(){
 const kind=p().health;let graph,caption,label;
 if(kind==='startup'){
  const titles=['HTTP 接口','warmup','状态 Up'];const done=[step>=1,step>=3,step>=3];
  graph=`<div class="ops-steps">${titles.map((t,i)=>`<div class="ops-stage ${done[i]?'done':step===2&&i===1?'active':''}">${done[i]?'✓ ':''}${t}</div>`).join('')}</div>`;
  graph+=node('普通 Python Runtime',chips(step===2?['warmup']:step>=3?['已预热']:[],'Starting'),step===1?'/model_info 可响应':'启用正常 warmup 路径');
  graph+=badge(step<3?'探活：503 · Starting':step===3?'状态 Up · 可以开始探测':'探活：200 · 观察到回程活动',step<3?'bad':step===4?'good':'');
  label=['进程启动','接口可响应','发送 warmup','warmup 成功','健康探测观察到活动'][step];
  caption=['进程出现，不代表模型服务已经准备好。','warmup 会先等待 /model_info；接口可响应与 warmup 完成是两个阶段。','warmup 请求实际经过运行时。这里仍不能仅凭 HTTP 已启动就宣布就绪。','普通 Python 路径的 warmup 成功后设置 Up。健康生成检查还会观察自己的探测窗口。','在健康检查窗口内观察到回程活动，返回 200；这不是每个模型能力或每条请求成功的证明。'][step];
 }else{
  const light=kind==='light',active=kind==='activity',result=step===0?null:M.health({generation:!light,last:active&&step===2?101:99,now:!active&&step===2?121:101});
  graph=node('TokenizerManager · 活动观测',chips(step===2&&active?['R2 回包']:[],'等待回程'),light?'/health · 关闭生成检查':'/health_generate · 记录探测起点 tic');
  graph+=wire([[450,300,true]],active&&step===1,light?'本分支不发生成探针':active?'普通请求 R2 的回程也会更新活动时间':'窗口内没有观察到新活动');
  graph+=`<div class="ops-pair">${node('R1',chips(['排队']),'本例尚未执行')}${node('R2',chips(active&&step>=1?['回包']:[],'无回包'))}</div>`;
  graph+=badge(result===null?'探活结果：等待':`探活结果：${result}`,result===200?'good':result===503?'bad':'');
  label=step===0?'开始检查':step===1?(light?'静态分支返回':active?'R2 有回程活动':'继续等活动'):'得到探活结果';
  caption=light?'Starting 与退出标记仍先返回 503；在 Up 状态关闭 /health 的生成检查后，该路径可以直接 200，不能据此证明 Scheduler 正在前进。':active?'判据是全局 last_receive_tstamp 推进，不要求专属探针完成。R2 回包可以让探活通过，即使 R1 还在排队。':'直到配置的检查期限仍无新回程活动，状态置为 UnHealthy，返回 503；这本身不能定位故障在计算、通信还是回程处理。';
 }
 return {graph,title:'接口响应、预热完成、运行时活动是三件事',label,caption,why:'限定普通 Python HTTP、非 P/D、未启用诊断旁路。/health 与 /health_generate 共享入口，但 /health 可通过配置关闭生成检查。图中事件顺序是教学场景，不测量启动或探测耗时。',source:kind==='startup'&&step<4?'warmup':'health'};
}
function admission(){
 if(p().admission==='route'){
  const d=M.route(p().workers,step),reason=p().workers==='circuit'?'健康，但熔断拒绝':'不健康';
  let graph=node('Gateway · Round-robin',chips(step?[`R${step}`]:[],'待分配'),'本例候选已经匹配模型 / 角色');
  graph+=wire(d.selected===null?[]:[[300,d.selected===0?150:450]],step>0,step?'本次只选择一个可用 worker':'先过滤候选，再轮流选择');
  graph+=`<div class="ops-pair">${[0,1].map(i=>node(`Worker ${i?'B':'A'}`,chips(d.assigned[i]),d.eligible.includes(i)?'可选':i===1?reason:'不健康',d.selected===i?'selected':d.eligible.includes(i)?'':'off')).join('')}</div>`;
  if(d.rejected.length)graph+=badge('没有可用候选 · 未转交请求','bad');
  return {graph,title:'路由先看可用性，再执行选择策略',label:step?`处理 R${step}`:'候选名单',caption:'Round-robin 只在健康且熔断器允许的候选中循环。本例请求留在所选实例；网关选择不等于该实例已经完成调度和计算。',why:'固定候选顺序 A、B，策略计数从 0 开始，不加入并发更新、重试或其他模型。半开熔断的真实准入由 circuit breaker 决定，图只展示“允许 / 不允许”的结果。Cache-aware 等策略另有规则；本图不保证 KV 命中。',source:'route'};
 }
 const d=M.queue(p().capacity,step);
 const graph=`<div>${node(`等待队列 · 容量 ${p().capacity}`,`<div class="ops-chips">${Array.from({length:p().capacity},(_,i)=>chip(d.waiting[i]||'空',d.waiting[i]?'':'empty')).join('')}</div>`)}</div>${wire(step===4?[[300,300]]:[],step===4,step===4?'R0 完成，示例将队首 R1 交给执行':'队列额度与执行位置分开计数')}${node('执行位置 · 1 个',chips(d.running?[d.running]:[]))}<div class="ops-rejected">${d.rejected.length?d.rejected.map(r=>chip(r,'reject')).join(' ')+' · 503 队列已满':'尚无拒绝'}</div>${d.done.length?badge('R0 已完成','good'):''}`;
 return {graph,title:'队列满了，继续等待还是明确拒绝？',label:['R0 正在执行','R1 到达','R2 到达','R3 到达','完成并接纳队首','R4 到达'][step],caption:step===3?'普通无优先级路径中，加入后会超过 max_queued_requests，就拒绝新请求并回告 503。不是把所有请求无限堆在队列里。':step>=4?'已完成请求让出执行位置，等待请求推进后也腾出队列容量。新的请求是否能入队，需要重新判断。':'本图只画一个 Runtime 的等待队列；执行位置只有一个是教学假设。Gateway 的额度、HTTP 在途数与这个队列上限不是同一个对象。',why:'关闭 priority scheduling，示例的 KV / token 预算足够，按队首推进；不模拟完整 Scheduler。源码队列检查只说明 max_queued_requests 边界，不能据此推断 max_running_requests 或 GPU 利用率。优先级开启时可能改为淘汰已有低优先级请求。',source:'queue'};
}
function cancel(){
 const kind=p().cancel,d=M.cancellation(kind,step);
 let graph=badge(`客户端：${d.client}`,d.client==='断连'?'bad':'');
 graph+=node('TokenizerManager · 请求登记',chips(d.frontend?['R1']:[]),step===4?'回告已处理':kind==='disconnect'&&step>=2?'已发出取消 / 等待回告':'仍持有 ReqState');
 graph+=wire((step===2&&kind==='disconnect')||step===3?[[300,300,step===3]]:[],(step===2&&kind==='disconnect')||step===3,step===2?(kind==='disconnect'?'AbortReq → Scheduler':'Scheduler 轮询产生 AbortReq'):step===3?'Abort 回告 → TokenizerManager':'消息尚未完成收尾');
 graph+=node('Scheduler · 等待队列',chips(d.queued?['R1']:[]),d.queued?'本例尚未开始模型执行':'请求已移出队列');
 if(d.code)graph+=badge('本例非流式请求：503 等待超时','bad');
 return {graph,title:'调用方停止等待，后端不会凭空消失',label:['R1 等待执行',kind==='disconnect'?'客户端断开':'超过等待期限',kind==='disconnect'?'检测断连并发取消':'轮询产生取消','后端移除并回告','前端登记清理'][step],caption:kind==='disconnect'?['前端与后端各自持有请求状态；现在只是等待，没有开始模型执行。','客户端不再等待，后端队列里的 R1 此刻仍存在。','所选非后台等待路径检查到断连后派发 AbortReq。发出消息仍不等于后端已经处理。','Scheduler 从等待队列移除请求并发送回告，通知前端结束对应状态。','前端收到回告后标记结束、删除登记并唤醒等待者；断开的客户端不保证收到结果。'][step]:['队列开始计时，客户端仍保持连接。','超过等待期限不代表此刻已从队列删除；检查发生在 Scheduler 的轮询点。','轮询产生携带 503 原因的 AbortReq，再按一致的请求处理路径消费。','Scheduler 移除等待请求并保留原始超时原因回告前端。','本例非流式返回 503，前端登记被删除。流式已经发送响应头后的错误表达另有边界。'][step],why:'限定普通非 P/D、未进入执行的文本请求，非 background。前端事件等待超时只是再次检查断连的时机，不是自动中止一切请求的总时限。运行中取消还要处理执行引用；P/D、Grammar、混合状态各有资源条件，见运行时与通信课程。',source:kind==='waiting'&&step<=2?'timeout':step<=2?'disconnect':step===3?'dequeue':'echo'};
}
function drain(){
 const kind=p().drain,d=M.drain(kind,step);
 let graph=`<div class="ops-gates"><span class="${d.exiting?'closed':'open'}">健康入口：${d.exiting?'503':'正常'}</span><span class="${d.ingressStopped?'closed':'open'}">新请求：${d.ingressStopped?'外部已摘流':'仍可能到达'}</span></div>`;
 graph+=node('前端登记 · 在途请求',chips(d.pending),'登记排空后才走正常 ShutdownReq 分支');
 graph+=wire(d.shutdownSent?[[300,300]]:[],step===3&&d.shutdownSent,d.shutdownSent?'ShutdownReq → Scheduler':kind==='arrivals'&&step===3?'R3 新到达 · 排空条件仍不满足':'等待在途请求结束');
 graph+=node('Scheduler 进程',chips(d.exitObserved?[]:[d.schedulerReleased?'已执行清理':d.killFallback?'强制终止路径':'仍存在']),d.exitObserved?'本例已观察到退出':d.killFallback?'不能据此证明用户态清理完整':'停止循环与释放资源属于后续步骤',d.exitObserved?'off':'');
 const captions=['服务仍在处理 R1、R2。','SIGTERM 设置退出标记，健康入口返回 503；这不是所有调用方都已停止投递的证明。',kind==='arrivals'?'R1 结束，但本场景仍允许直连请求进入。':'本场景假设部署侧已停止新流量；R1 结束，仍等待 R2。',kind==='arrivals'?'R2 结束时又收到 R3。登记仍非空，无法沿正常排空分支继续退出。':'登记清空后发送 ShutdownReq。只有发出消息，还不能认定 Scheduler 已退出。',kind==='stuck'?'Scheduler 在等待期限内仍未退出，继续观测实际进程。':'Scheduler 设置退出标记，离开事件循环，在 finally 路径清理资源与通信环境。',kind==='stuck'?'等待到期仍有 Scheduler 存活，代码会告警并进入强制清理进程树路径；这不是完整释放资源的证明。':'本例观察到 Scheduler 已退出，再收尾其余子进程与服务。部署验收还应核对真实进程、端口和资源状态。'];
 return {graph,title:'先停止新增工作，再排空并释放',label:['正常运行','收到 SIGTERM','在途请求减少',kind==='arrivals'?'新流量阻止排空':'发送 ShutdownReq','等待 Scheduler 退出',kind==='stuck'?'期限到达 · 升级终止':'观测退出并收尾'][step],caption:captions[step],why:'本图是普通健康服务的 SIGTERM 分支。停止新投递是部署侧协作假设，不是 gracefully_exit 自动完成。UnHealthy 或 SGL_FORCE_SHUTDOWN 可以跳过正常等待；退出信号、ShutdownReq、finally 清理和进程实际退出是不同证据。未实际发送信号或执行退出实验。',source:step<=1?'signal':step===4?'process':'drain'};
}
function stop(){clearInterval(timer);timer=null;$('#ops-play').textContent='播放';$('#ops-play').setAttribute('aria-pressed','false');}
function render(){const d=({health,admission,cancel,drain})[mode]();$('#ops-title').textContent=d.title;$('#ops-scene').innerHTML=d.graph;$('#ops-scene').dataset.mode=mode;$('#ops-scene').dataset.step=step;$('#ops-caption').textContent=d.caption;$('#ops-why').textContent=d.why;$('#ops-source').href=M.url(d.source);$('#ops-source').textContent=`源码 · ${M.sources[d.source][2]} ↗`;$('#ops-status').textContent=`${step+1} / ${last()+1} · ${d.label}`;$('#ops-previous').disabled=step===0;$('#ops-next').disabled=step===last();$('#worker-setting').hidden=p().admission!=='route';$('#queue-setting').hidden=p().admission!=='queue';if(step===last())stop();}
function reset(){stop();step=0;render();}lab.hidden=false;lab.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;lab.querySelectorAll('[data-mode]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));lab.querySelectorAll('[data-settings]').forEach(x=>x.hidden=x.dataset.settings!==mode);reset();});lab.querySelectorAll('select').forEach(x=>x.onchange=reset);$('#ops-previous').onclick=()=>{stop();if(step>0)step--;render();};$('#ops-next').onclick=()=>{stop();if(step<last())step++;render();};$('#ops-reset').onclick=reset;$('#ops-play').onclick=()=>{if(timer){stop();return;}if(step===last())step=0;render();$('#ops-play').textContent='暂停';$('#ops-play').setAttribute('aria-pressed','true');timer=setInterval(()=>{step++;render();},2400);};document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change',render);render();
})();
