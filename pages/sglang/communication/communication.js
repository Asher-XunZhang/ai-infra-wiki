(()=>{
'use strict';
const M=CommunicationModel,$=s=>document.querySelector(s),lab=$('[data-comm-lab]');
let mode='message',step=0,timer=null;
const kinds=()=>({message:$('#message-kind').value,collective:$('#collective-kind').value,kv:$('#kv-fragment').checked,gates:$('#gate-kind').value});
const last=()=>mode==='message'?(kinds().message==='ipc'?2:4):mode==='collective'?2:mode==='kv'?5:['abort','timeout'].includes(kinds().gates)?3:kinds().gates==='cache'?6:5;
const chip=(x,i=0,empty=false)=>`<span class="chip ${empty?'empty':i===1?'b':i===2?'c':''}">${x}</span>`;
const chips=xs=>`<div class="chips">${xs.map((v,i)=>chip(v,typeof v==='string'&&/^[AB][01]$/.test(v)?(v[0]==='A'?0:1):i)).join('')}</div>`;
const endpoint=(title,body,note='',active=false)=>`<div class="endpoint ${active?'active':''}"><h3>${title}</h3>${body}${note?`<small>${note}</small>`:''}</div>`;
function wire(paths,active,label,loop=false,reverse=false){
 const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
 return `<p class="wire-label">${label}</p><div class="comm-wire"><svg viewBox="0 0 600 76" preserveAspectRatio="none" aria-label="${label}" role="img">${paths.map(([x1,x2],i)=>{const end=loop||reverse?3:72;const d=`M ${x1} ${reverse?72:3} C ${x1} ${loop?90:38} ${x2} ${loop?90:38} ${x2} ${end}`;return `<path d="${d}" class="${active?'live':''}"/>${active?`<circle cx="${reduced?x2:0}" cy="${reduced?end:0}" r="4">${reduced?'':`<animateMotion dur="1.6s" begin="${i*.08}s" repeatCount="indefinite" path="${d}"/>`}</circle>`:''}`;}).join('')}</svg></div>`;
}
function message(){
 const ipc=kinds().message==='ipc',labels=ipc?['准备派发','发送侧已派发','调度侧收到请求']:['准备张量','发送形状与类型','接收侧分配空间','传递张量内容','等待完成后使用'];
 const left=ipc?chips(['R1']):chips([3,5]);
 const right=ipc?(step===2?chips(['Req R1']):chips(['…'])):step>=4?chips([3,5]):step>=2?`<div class="chips">${chip('?',0,true)}${chip('?',0,true)}</div>`:chips(['…']);
 let graph=`<div class="comm-pair">${endpoint(ipc?'TokenizerManager':'上游 rank',left,ipc?'请求对象':'tensor · shape [2]',step===0)}${endpoint(ipc?'Scheduler':'下游 rank',right,ipc?(step===2?'本地请求对象':'尚未处理'):(step>=2?'已分配 tensor':'尚无接收 buffer'),step===last())}</div>`;
 graph+=wire([[150,450]],ipc?step===1:[1,3].includes(step),ipc?(step===0?'进程间消息':step===1?'R1 · 正在交接':'同一请求，不同对象'):step<3?'元数据：shape [2] · dtype':'数据：[3, 5]',true);
 const captions=ipc?['前端持有 R1 的状态；调度侧还没有这个请求对象。','发送侧把对象交给通信层。派发标记不等于调度侧已经接收或执行。','调度侧处理收到的对象，建立自己的 Req；跨进程传递不会共享同一个 Python 对象。']:['上游已有 [3,5]；下游需要先知道形状和类型，才能准备接收空间。','先交接 TensorMetadata。图中的两个数还留在上游。','下游按元数据分配空间。问号表示内容尚不可用。','张量内容经过对应进程组发送；异步发送返回的 work 句柄不是完成证明。','所选 recv_tensor_dict 路径等待接收 work 完成后返回；下游才能使用 [3,5]。'];
 return {graph,title:ipc?'消息跨进程，对象各自持有':'先告诉形状，再交接张量',label:labels[step],caption:captions[step],why:ipc?'本图只画普通请求的逻辑交接。底层 sock_send 及共享内存封装有自己的路径；“已派发”不是对端处理回执。':'只展示两个 rank、一个张量、无可选 send-allgather 的路径。CPU 元数据组与实际张量组有不同职责；异步发送使用 P2PWork 保留 payload 引用。',source:ipc?'ipc':step>=2?'receive':'send'};
}
function collective(){
 const kind=kinds().collective,d=M.collective(kind),name={reduce:'All-reduce · 逐位置求和',gather:'All-gather · 收齐分片',exchange:'All-to-all · 按目的地交换'}[kind];
 let graph=`<div class="comm-pair">${d.input.map((xs,i)=>endpoint(`rank ${i} · 输入`,chips(xs))).join('')}</div>`;
 graph+=wire([[150,150],[150,450],[450,150],[450,450]],step===1,step===0?'同一个进程组':step===1?'逻辑数据依赖 · 非物理网络线路':'通信结果');
 graph+=`<div class="comm-pair">${d.output.map((xs,i)=>endpoint(`rank ${i} · 输出`,step===2?chips(xs):`<div class="chips">${chip('等待',0,true)}</div>`,'',step===2)).join('')}</div>`;
 const note={reduce:'对应位置相加：1+10=11，2+20=22；两个 rank 都获得相同结果。',gather:'沿教学约定的顺序拼接，数值没有相加；每个 rank 都收齐四个数。',exchange:'A 表示发往 rank 0，B 表示发往 rank 1；每份元素只有一个目的地，不是复制到所有 rank。'}[kind];
 return {graph,title:name,label:['各自输入','组内交换','各自输出'][step],caption:step===2?note:step===0?'先预测结果：相加、拼接、重排是三种不同的算子语义。': '箭头表示参与关系；实际通信可由不同算法和后端完成，不代表必须同时发送四条网络消息。',why:note+' 示例使用两个 rank、等长分片、sum 归约；SGLang 的具体层会按布局选择算子。EP dispatcher 也不保证固定使用这里的 all_to_all_single。',source:kind};
}
function kv(){
 const d=M.pages(kinds().kv),sorted=[...d.dst].sort((a,b)=>a-b),pos=[100,300,500];
 const row=(target)=>`<div class="pool-row">${(target?sorted:d.src).map((p,i)=>{const index=target?d.dst.indexOf(p):i;const filled=target?step>=4:step>=2;return `<div class="page-slot ${target&&step>=1?'reserved':''}" data-page="${p}">${chip(filled?d.payload[index]:'·',index,!filled)}<small>page ${p}</small></div>`;}).join('')}</div>`;
 const paths=d.src.map((p,i)=>[pos[i],pos[sorted.indexOf(d.dst[i])]]);
 const graph=`<div class="registered"><p class="pool-label">P · 已注册的内存池</p>${row(false)}<div class="copy-segments">${d.blocks.map(b=>`<span style="grid-column:span ${b.pages}">${b.pages*16} B</span>`).join('')}</div></div>${wire(step===0?[]:step===1?[[300,300]]:paths,step===1||step===3,step===0?'池注册不等于请求已经占页':step===1?`D → P：目标页 [${d.dst.join(',')}]`:step===2?'数据已就绪 · 即将入队':step===3?'按对应关系写入 · 不按相同页号复制':step===4?'数据已写入 · 等完成通知':'传输完成 · 继续检查 Decode 就绪',false,step===1)}<div class="registered"><p class="pool-label">D · ${step>=1?'R1 占用目标页':'内存池尚未为 R1 预留'}</p>${row(true)}</div><div class="result-note">${d.blocks.length} 段 × 本图 1 个 K buffer · 共 48 B</div>`;
 return {graph,title:'同一份 KV，落在不同的物理页',label:['注册内存池','D 预留并发布位置','P 产生 KV','入队后执行搬运','目标页已有内容','传输层报告完成'][step],caption:['注册让传输引擎认识一段内存；本图只显示三个候选位置，并不表示整池只有三页。','D 为 R1 预留自己的页，通过控制消息告诉 P 目标索引；索引本身不是 KV 内容。','A、B、C 是请求的连续三段 KV。P 的物理页 [7,8,12] 并不连续。','send() 先入队。本帧继续展示工作线程执行搬运；发送调用返回本身不能证明写入完成。','目标页的内容与源页一一对应。P 的源数据不会因为复制自动消失。','这里只到传输完成。切换“就绪与回收”，看为什么不能立刻认定请求可执行。'][step],why:`每页按 4 token × 1 head × 2 dimensions × FP16 = 16 B；仅画一层的 K，未画 V、其他层、TP/PP 切片或 staging。连续合并必须源与目标同时相邻。本布局为 ${d.blocks.map(b=>`${b.src}→${b.dst}：${b.pages*16} B`).join('；')}。页号各属于自己的池，不是全局地址。`,source:step===0?'register':step===1?'metadata':step===3?'queue':'blocks'};
}
function gates(){
 const kind=kinds().gates,d=M.gates(kind,step);
 if(d.failed){
  const graph=`<div class="request-track">${chip('R1 失败',1)}<span class="rail"></span><div class="parking ${d.held?'occupied':''}">${d.held?'D 槽位占用':'D 槽位释放'}</div></div><div class="ack-dots">${[0,1].map(i=>`<span class="ack-dot ${d.acks.includes(i)?'done':''}">${d.acks.includes(i)?'✓ ':''}P${i}</span>`).join('')}</div><p class="result-note">${d.drained?'2 / 2 drain ACK':d.timedOut?'超时释放 · ACK 未收齐':`${d.acks.length} / 2 drain ACK`}</p>`;
  return {graph,title:'请求失败，目标页何时能复用？',label:['失败并通知源端','等待 drain ACK','收到一个 ACK',kind==='abort'?'收齐 ACK 并释放':'等待超时并释放'][step],caption:step===3?(d.drained?'所需源端都回报 drain ACK，源码释放该请求持有的页。':'该固定版本在超时后仍释放页；没有收齐 ACK，不能据此断言所有远端写入都已经排空。'):'本分支启用延迟回收、后端支持且 abort 已通知。请求结束与物理槽位释放分开处理，避免立刻复用仍可能被写入的位置。',why:'独立教学场景：有两个可能写入的源端。ACK 用源端身份去重，重复 ACK 不增加计数。没有启用或不满足延迟回收条件时，pop_transferred 走其他释放路径；本图不泛化为所有取消流程。',source:'release'};
 }
 const done=[d.kv,d.metadata!==0&&d.peers,kind==='cache'?d.restored:step>=5],names=['KV 收齐','元数据 + 同步',kind==='cache'?'恢复缓存':'提交检查'];
 const graph=`<div class="gate-track">${names.map((name,i)=>`<div class="gate ${kind==='corrupt'&&step===5&&i===2?'failed':done[i]?'done':'pending'}">${name}</div>`).join('')}</div><div class="request-track"><div class="parking ${!d.ready&&!(d.corrupt&&step===5)?'occupied':''}">${d.corrupt&&step===5?'交接已清理':!d.ready?chip('R1',1):'交接队列'}</div><span class="rail"></span><div class="parking ${d.ready?'occupied':''}">${d.ready?chip('R1'):'可调度队列'}</div></div><div class="ack-dots">${[0,1].map(i=>`<span class="ack-dot ${d.sources.includes(i)?'done':''}">${d.sources.includes(i)?'✓ ':''}P${i}</span>`).join('')}</div>${d.corrupt&&step===5?'<p class="result-note">R1 终止 · 目标页释放</p>':''}`;
 const labels=['尚未收齐','一个源端完成','两个源端完成','元数据到达','参与 rank 达成就绪',kind==='cache'?'缓存恢复完成':'提交检查', '提交检查'];
 const captions=['本场景有两个源端。请求仍在交接队列，页已经为它预留。','一个源端报 Success；另一个未完成，不能推进请求。','KV 已收齐，但元数据仍未到达，整体 poll 继续保持 Transferring。','非零元数据到达只是第一层 gate；它还不等于 room 身份已经校验通过。','本图假设所有参与 rank 均满足前置条件；源码用 MIN 归约保持推进一致。',kind==='cache'?'HiCache 恢复结束后，还要提交并校验请求元数据。':kind==='corrupt'?'提交时发现 room 99 与预期 42 不同，终止请求，不能进入可调度队列。':'room 等元数据检查通过，请求才从交接队列进入后续可调度流程。','缓存恢复与元数据检查都通过，请求进入后续可调度流程。'];
 return {graph,title:'写入完成，还缺哪些就绪条件？',label:labels[step],caption:captions[step],why:'用两个源端说明 unique-rank 计数；图不复用上一实验的单源拓扑。HiCache 是可选门槛。本图只画成功条件汇合，实际 gate 可重叠完成；调度、batch 和执行还在后续。元数据非零 gate 与最终 bootstrap_room 一致性检查分属两处源码。',source:step<=2?'success':step<=4?'gate':kind==='cache'&&step===5?'ready':'commit'};
}
function stop(){clearInterval(timer);timer=null;$('#comm-play').textContent='播放';$('#comm-play').setAttribute('aria-pressed','false');}
function render(){
 const d=({message,collective,kv,gates})[mode]();
 $('#comm-title').textContent=d.title;$('#comm-scene').innerHTML=d.graph;$('#comm-scene').dataset.mode=mode;$('#comm-scene').dataset.step=step;
 $('#comm-caption').textContent=d.caption;$('#comm-why').textContent=d.why;$('#comm-source').href=M.url(d.source);$('#comm-source').textContent=`源码 · ${M.sources[d.source][2]} ↗`;
 $('#comm-status').textContent=`${step+1} / ${last()+1} · ${d.label}`;$('#comm-previous').disabled=step===0;$('#comm-next').disabled=step===last();
 if(step===last())stop();
}
function reset(){stop();step=0;render();}
lab.hidden=false;
lab.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.mode;lab.querySelectorAll('[data-mode]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));lab.querySelectorAll('[data-settings]').forEach(x=>x.hidden=x.dataset.settings!==mode);reset();}));
lab.querySelectorAll('select,input').forEach(x=>x.addEventListener('change',reset));
$('#comm-previous').onclick=()=>{stop();if(step>0)step--;render();};$('#comm-next').onclick=()=>{stop();if(step<last())step++;render();};$('#comm-reset').onclick=reset;
$('#comm-play').onclick=()=>{if(timer){stop();return;}if(step===last())step=0;render();$('#comm-play').textContent='暂停';$('#comm-play').setAttribute('aria-pressed','true');timer=setInterval(()=>{step++;render();},2200);};
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change',render);
render();
})();
