/* Handoff context belongs to the picture, separately from its timing model.
 * Paths are relative to this file so nested courses and a Pages subpath work. */
(function(root){
'use strict';
const script=typeof document==='object'?document.currentScript:null;
const base=script?new URL('../',script.src):null;
const links={communication:['通信原理','foundations/communication.html'],transport:['SGLang 通信与传输','sglang/communication/index.html'],parallel:['并行与执行拓扑','sglang/parallelism/index.html'],system:['进程与调用边界','sglang/system-overview/index.html'],hardware:['GPU 计算与访存','foundations/hardware.html'],memory:['KV 地址与内存','sglang/kv-memory/index.html'],deployment:['P/D 角色与交接','foundations/deployment.html'],serving:['服务与回程','foundations/serving.html'],execution:['模型执行','sglang/model-execution/index.html'],generation:['候选验证','sglang/advanced-generation/index.html']};
const row=(from,payload,to,boundary,direction='forward')=>({from,payload,to,boundary,direction});
const entry=(rows,note,more)=>({rows,note,more});
const entries={
 'f-tokens':entry([row('分词器 · 文本','token IDs（词表编号）','模型 · Embedding 查表','逻辑数据依赖')],'分词通常由 CPU 处理，Embedding 属于模型计算；本图只解释两次查表。ID 进入执行端的进程通信与设备搬运见实现课。',['system','execution']),
 'f-generation':entry([row('模型执行端','输出 ID → 反分词 → 文本','客户端','响应回程'),row('本轮采样结果','同一 token ID','下一轮模型输入','引擎内部反馈')],'客户端可与服务端同机或跨机。下一轮输入由引擎保留；网络回程与模型内部反馈是两条路径。',['system','serving']),
 'f-attention':entry([row('允许读取的 K / V','读取、点积、加权求和','当前查询的输出','单设备计算示例')],'图中的贡献汇合表示数值运算。多设备下的上下文切分与数据交换在 CP 专题展开。',['hardware','parallel']),
 'f-cache':entry([row('本设备历史 KV','读取已有 K/V','本轮 Attention','设备内存读取')],'左右两栏是同一任务的两种做法，没有互相传输。跨设备 KV 交接另见 P/D 课程。',['memory','deployment']),
 'f-hardware':entry([row('GPU 显存中的输入 / 权重','通过存储层级读取','同一 GPU 的计算单元','设备内部访存')],'本图省略缓存与寄存器层级；读取副本属于同一设备的计算供数。',['hardware']),
 'f-chunk':entry([row('B 的长输入','按轮次分块处理','当前输入块','单实例调度示例')],'A 的 Decode 位置与 B 的当前输入块同批执行。',['execution','system']),
 'f-batching':entry([row('调度器等待集合','选入、退出、补入','本轮 batch 成员','调度状态更新')],'图中位置表示逻辑成员关系。组批之后，执行层还要准备输入与地址并提交设备工作。',['execution','system']),
 'f-communication':entry([row('进程 P0 / Rank 0 · GPU 0','张量片段或局部贡献','进程 P1 / Rank 1 · GPU 1','跨进程、GPU 间 collective','both')],'两方共同参与 AllReduce / AllGather；图中运算区是逻辑示意。设备可同机或跨机，实际链路由拓扑和通信后端决定。',['transport','parallel']),
 'f-deployment':entry([row('Prefill 进程 / GPU','数据面：KV 载荷','Decode 进程 / GPU','跨进程、GPU 间传输'),row('P 侧控制组件','控制面：身份、地址、就绪信息','D 侧控制组件','控制消息')],'P / D 可部署在同一主机或不同主机。数据与控制分别推进，实际传输协议取决于所选后端。',['transport']),
 'f-speculation':entry([row('草稿生成','候选 token 与验证输入','目标模型验证','模型之间的数据依赖')],'本图未指定草稿与目标模型的进程或设备放置。正式输出和有效 KV 两行表示提交集合，不能据此推断发生网络传输。',['generation','execution']),
 'f-serving':entry([row('客户端连接','断连 / 取消事件','服务入口','连接边界'),row('服务入口','取消控制消息','执行后端','内部控制路径')],'本图只描述安全生命周期；入口与后端如何跨进程通信，在 SGLang 服务课结合具体对象展开。',['system','transport']),
 'model-execution':entry([row('Scheduler → Worker → Runner','本轮输入、地址与调用','模型层','本例同一执行进程'),row('主机侧运行时','提交设备工作','本 rank 的 GPU','主机控制与设备执行')],'此处是单 rank 示例。模型层之间传递张量；图中的函数框不是独立网络服务。',['system','hardware']),
 'kv-memory':entry([row('请求的逻辑 token 位置','地址索引 / 引用','本 rank 的物理 KV 槽','本地地址映射')],'映射线用于查地址；写入由设备计算产生的 KV 完成。远端 KV 复制在通信课单独说明。',['hardware','transport']),
 'scheduling-chunk':entry([row('已有 chunked_req / 等待队列','本轮输入位置与预算','Prefill batch','同一 Scheduler 内的计划与执行衔接')],'L1 等编号标记输入位置；移动的是计划中的工作份额。选入计划后还要执行 forward，完成后才推进已计算历史。',['execution','memory']),
 'scheduling':entry([row('Scheduler 等待集合','预算检查与本轮计划','can_run_list','同一进程的调度决策')],'准入通过后才进入输入准备和设备执行；本图成员移动不表示请求跨机迁移。',['system','execution']),
 'request-runtime':entry([row('请求行 / 缓存管理','建立或解除持有关系','本 rank 的 KV 存储','本地资源生命周期')],'连线表示持有关系。物理 KV 保持原位；跨进程输出和远端数据搬运属于另外的路径。',['memory','system']),
 'communication':entry([row('源进程 · 源 GPU 页','KV 字节载荷','目标进程 · 目标 GPU 页','跨进程、GPU 间传输')],'本课 P/D 页复制示例可对应同机或跨机部署。地址、身份与完成通知走控制路径；载荷通过通信后端的数据路径传输，链路不能仅由页号推断。',['deployment','parallel']),
 'advanced-generation':entry([row('草稿生成 → 目标验证','候选与验证预测','接受前缀 / 有效 KV','本课线性验证的数据依赖')],'图按 token 位置排列，未画模型的设备部署。跨 rank 通信由模型并行配置决定；提交集合的变化在本地状态中表示。',['execution','parallel']),
 'serving-operations':entry([row('TokenizerManager 进程','AbortReq(rid) →；处理回告 ←','Scheduler 进程','本例跨进程 IPC','both')],'IPC 是进程间通信。取消与回告属于控制消息；客户端连接另走服务响应路径，本例没有 GPU KV 载荷传输。',['system','transport']),
 'system-overview':entry([row('TokenizerManager','请求 / token IDs','Scheduler','跨进程 IPC'),row('Scheduler','方法调用','Worker / Runner','本例同进程'),row('Scheduler → Detokenizer → 入口','输出 ID → 文本','客户端','内部 IPC 与外部响应回程')],'进程、函数对象与 GPU 是不同边界；客户端和服务端是否跨机取决于部署。',['transport','execution']),
 'dataflow':entry([row('Scheduler 本地容器','引用同一个 Req','本地 Req 对象','同进程对象关系'),row('Prefill PP Stage i','activation（激活）','PP Stage i+1','跨 rank、GPU 间交接'),row('Prefill 实例','KV 载荷 / 控制信息','Decode 实例','跨实例传输')],'同一个流程同时包含引用、PP 激活通信与 P/D KV 传输，三种箭头按载荷区分。同机或跨机由部署决定。',['transport','parallel']),
 'lifecycle':entry([row('PP 各 stage 的进程','控制名单 / 共识信息','本级 Scheduler','跨 rank 控制通信'),row('本级 sender','本地 poll / 完成状态','本级资源收尾','本地状态检查')],'跨级控制结果与本地传输完成是两个条件。KV 数据面仍由传输后端推进；远端接收位置和链路见通信课。',['transport','parallel'])
};
entries['queued-timeout']=entry([row('Scheduler 进程','本地等待时限检查','waiting_queue 中的 Req','同进程状态更新'),row('Scheduler 进程','超时处理回告','TokenizerManager 进程','跨进程 IPC')],'本例从未执行的等待请求不持有 GPU KV；前端接到回告后清理 ReqState，并按该路径返回 503。',['system','transport']);
entries.parallelism=entries['f-communication'];
entries['parallel-tp']=entries['f-communication'];
entries['parallel-dp']=entry([row('入口 / DP Controller','分派不同请求','不同模型副本','副本之间独立处理请求')],'请求分派经过入口与执行进程间的控制路径；副本可同机或跨机。每个副本内部还可使用 TP / PP。',['system','transport']);
entries['parallel-ep']=entry([row('token 所在 rank','派发所选专家的输入','专家所在 rank','同 rank 本地计算 / 跨 rank 数据交换'),row('专家输出','按路由权重汇合','原 token 对应位置','返回与合并')],'目标专家在本 rank 时可本地执行；在其他 rank 时需设备间数据交换。同机或跨机及通信算法由拓扑与 dispatcher 后端决定。',['transport']);
entries['parallel-cp']=entry([row('各 CP rank 的局部 K/V','本课 Zigzag Prefill 路径汇合','参与 Attention 的 rank','跨进程、GPU 间 collective','both')],'查询位置的分工不等于物理卡的位置。参与设备可同机或跨机；连续切分选项只作负载对照，不声称复现同一通信路径。',['transport']);
entries['parallel-sp']=entry([row('TP 组各 rank 的激活分片','AllGather：收齐序列行','TP 计算区域','跨进程、GPU 间 collective','both'),row('各 rank 的输出贡献','ReduceScatter：归约后按序列分发','逐 token 算子区域','跨进程、GPU 间 collective','both')],'本图采用经典 Megatron SP 前向布局。两 rank 同机、各用一张 GPU；这是激活布局转换，不是直接拆分完整 Attention 的历史。',['communication','parallel']);
entries['parallel-pcp']=entry([row('进程 P0 / Rank 0 · GPU 0','局部 K/V 的副本 · AllGather','进程 P1 / Rank 1 · GPU 1','本例同机、跨进程、GPU 间通信','both')],'本图选择收齐 K/V 的教学路径。Query 行保持各自归属；其他 CP 实现可循环传块，实际通信与内存开销由算法决定。',['communication','parallel']);
entries['parallel-dcp']=entry([row('当前 Query 所在 rank','Q 副本','保存其他历史 KV 的 rank','本例跨进程、GPU 间传递'),row('各 rank 的局部 Attention','局部输出 + LSE 等归一化统计','全局合并后的输出','跨进程、GPU 间通信','both')],'本图两 rank 同机各用一张 GPU，历史 KV 留在各自设备；实际 Query 分布与通信算法由并行布局及后端决定。',['communication','parallel']);
function context(key,placement='same-host'){
 if(key==='f-parallel'||key==='parallel-pp')return entry([
  row('主机 A · 进程 P0 · GPU 0\nPP Stage 0 / L1–L2','MB 的 activation（激活）',placement==='cross-host'?'主机 B · 进程 P1 · GPU 1\nPP Stage 1 / L3–L4':'主机 A · 进程 P1 · GPU 1\nPP Stage 1 / L3–L4',placement==='cross-host'?'跨进程 + GPU 间 + 跨主机通信':'跨进程 + GPU 间通信 · 同一主机')
 ],placement==='cross-host'?'GPU 0 / 1 是本图设备标签。本例两张 GPU 位于不同主机，载荷经过跨机互联；具体协议和中转路径由后端与硬件决定。':'GPU 0 / 1 是本图设备标签。本例两张 GPU 位于同一主机，通过可用的机内数据路径交接；是否直连或经过中转取决于后端与硬件。',['communication','transport']);
 return entries[key]||null;
}
function mount(viewport,key,onChange){
 if(!context(key))return {update(){}};
 const box=document.createElement('section');box.className='motion-handoff';box.setAttribute('aria-label','交接边界与载荷');box.dataset.boundaryPanel=key;
 const lessonCompact=Boolean(document.body.dataset.lesson);const compact=lessonCompact||key==='scheduling'||key==='scheduling-chunk';if(compact)box.classList.add('is-compact');
 const title=document.createElement('strong');title.textContent='读图：交接跨越什么边界';if(!compact||!lessonCompact)box.append(title);
 let placement='same-host';
 if(key==='f-parallel'||key==='parallel-pp'){
  const label=document.createElement('label');label.textContent='部署示例 ';const select=document.createElement('select');select.setAttribute('aria-label','PP 部署位置');
  for(const[value,text]of [['same-host','同机两卡'],['cross-host','跨机两卡']]){const o=document.createElement('option');o.value=value;o.textContent=text;select.append(o);}label.append(select);box.append(label);
  select.addEventListener('change',()=>{placement=select.value;render();onChange?.(placement);});
 }
 const diagram=document.createElement('div'),note=document.createElement('p'),nav=document.createElement('nav');nav.setAttribute('aria-label','交接机制详解');box.append(diagram,note,nav);viewport.prepend(box);
 function render(){const c=context(key,placement);box.dataset.placement=placement;diagram.replaceChildren();
  if(compact){const line=document.createElement('p');line.className='handoff-caption';const basic={'f-tokens':'文字 → token ID → Embedding 向量：同一输入的三种表示。','f-generation':'选中的 token ID → 下一轮模型输入；反馈在模型侧完成。','f-attention':'Q 与 K 比较，权重乘 V；本例在一个设备内计算。','f-cache':'左右是两种计算方式的对照，不互相传输数据。','f-chunk':'B 的长输入分块续算；A 的 Decode 与当前输入块同批执行。','f-batching':'请求进入或退出 batch：本例表示调度成员变化。','f-hardware':'分开观察矩阵乘加、算力与带宽约束、CPU 提交与 GPU 执行。','f-deployment':'P 进程 / GPU → D 进程 / GPU：传递历史 KV 与输入 ID；可同机或跨机。','f-speculation':'草稿 token → 目标模型验证 → 接受前缀；图中表示计算依赖。','f-serving':'客户端 → 服务端：取消请求；已提交工作结束后才释放资源。',scheduling:'等待集合 → 预算检查与本轮计划 → can_run_list：同一进程的调度决策；通过后才准备输入并提交设备执行，成员移动不表示请求跨机迁移。','scheduling-chunk':'已有 chunked_req / 等待队列 → 本轮输入位置与预算 → Prefill batch：同一 Scheduler 内的计划与执行衔接；移动的是本轮计划份额，执行完成后才推进已计算历史。'};line.textContent=basic[key]||c.rows.map(r=>r.from.replace(/\n/g,' · ')+' → '+r.to.replace(/\n/g,' · ')+'：'+r.payload+'（'+r.boundary+'）').join('；');diagram.append(line);}else for(const r of c.rows){const line=document.createElement('div');line.className='handoff-line';for(const[type,value]of [['endpoint',r.from],['edge',r.payload],['endpoint',r.to]]){const e=document.createElement('div');e.className='handoff-'+type+(type==='edge'&&r.direction==='both'?' handoff-both':'');e.textContent=value;if(type==='edge'){const b=document.createElement('small');b.textContent=r.boundary;e.append(b);}line.append(e);}diagram.append(line);}
  note.hidden=compact;note.textContent=c.note+(key==='f-parallel'||key==='parallel-pp'?' 切换只改变部署示意；下方时隙不计真实通信耗时。':'');nav.replaceChildren();for(const k of c.more){const a=document.createElement('a');a.textContent=links[k][0]+' ↗';a.href=new URL(links[k][1],base).href;nav.append(a);}
 }
 render();return {update(){render();}};
}
const api={context,mount};if(typeof module==='object'&&module.exports)module.exports=api;root.HandoffBoundaries=api;
})(typeof globalThis==='object'?globalThis:this);
