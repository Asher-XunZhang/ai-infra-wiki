/* Communication contracts, independent of animation and transport algorithms.
 * Three ranks by default; matrix examples use two. Disjoint input/output buffers. */
(function(root){
'use strict';
const ranks=[0,1,2],empty=()=>ranks.map(()=>[]);
const groups={distribution:'分发与收集',reduction:'归约与扫描',exchange:'交换与点对点',rma:'单边访问',synchronization:'同步与完成'};
const definitions=[
 ['Broadcast','distribution','广播','一份共同输入要供多个参与者使用。root 把相同数据复制给所有 rank。'],
 ['Scatter','distribution','分发','完整输入在 root，但各 rank 只负责其中一段。按目标 rank 分发片段。'],
 ['Gather','distribution','收集','每个 rank 有独立片段，只有 root 需要完整结果。按 rank 顺序收集，不做算术。'],
 ['AllGather','distribution','全收集','每个 rank 都需要全部片段；各方得到相同的有序拼接。'],
 ['Reduce','reduction','归约到根','不同 rank 贡献同一位置的数值，只有 root 需要最终归约值。'],
 ['AllReduce','reduction','全归约','每个 rank 都需要同一归约结果；对齐下标计算，再让各方得到结果。'],
 ['ReduceScatter','reduction','归约后分片','先合并同一位置的贡献，但后续各 rank 只需要结果的一段。'],
 ['Scan','reduction','包含自身的前缀归约','各 rank 需要知道截至自身的累计量，而非全组总和。'],
 ['Exscan','reduction','不含自身的前缀归约','分配连续存储区时，各 rank 需要知道排在自己之前的总量。'],
 ['AllToAll','exchange','全交换','每个 rank 都有发往不同目标的数据；将“按来源分组”转换为“按目标分组”。'],
 ['Neighbor','exchange','邻域交换','只需相邻参与者的数据时，只沿事先定义的邻接关系交换。'],
 ['SendRecv','exchange','点对点收发','一方把指定载荷交给另一方；发送和接收必须匹配。'],
 ['Put','rma','单边写入','发起方已知远端暴露的窗口和位置，直接指定要写入的载荷与地址。'],
 ['Get','rma','单边读取','发起方需要远端窗口的一段数据，指定远端位置并取回本地。'],
 ['Atomic','rma','远端原子更新','多个参与者更新同一远端位置时，读、判断、写需要遵循原子操作的约定。'],
 ['Barrier','synchronization','组内屏障','参与者进入下一阶段之前，需要确认同组所有 rank 都已到达这个同步点。'],
 ['Completion','synchronization','完成与可用','提交、载荷完成、消费者依赖满足是不同事件；观察下一步到底在等什么。']
].map(([id,group,label,why])=>({id,group,label,why}));
function calculate(op,opts={}){
 if(opts.matrix)return matrixExample(op);
 const rootRank=opts.root??0,variable=!!opts.variable,reduce=opts.reduce||'sum';
 let input=empty(),output=empty(),flows=[],math=[],counts=null;
 const add=(src,si,dst,di)=>{const value=input[src][si];output[dst][di]=value;flows.push({src,si,dst,di,value,id:`${src}-${si}-${dst}-${di}`});};
 const numeric=['Reduce','AllReduce','ReduceScatter','Scan','Exscan'].includes(op);
 if(numeric){
  input=['Scan','Exscan'].includes(op)?[[2],[3],[4]]:[[1,2,3],[10,20,30],[100,200,300]];
  const merge=vs=>reduce==='max'?Math.max(...vs):vs.reduce((a,b)=>a+b,0);
  if(op==='Scan'||op==='Exscan'){
   ranks.forEach(r=>{const sources=ranks.slice(0,r+(op==='Scan'?1:0));output[r]=sources.length?[merge(sources.map(i=>input[i][0]))]:[null];math.push({sources:sources.map(src=>({src,si:0})),value:output[r][0],destinations:[{dst:r,di:0}],label:`Rank ${r}`});});
  }else input[0].forEach((_,j)=>{const sources=ranks.map(src=>({src,si:j})),value=merge(sources.map(a=>input[a.src][a.si]));const dest=op==='Reduce'?[rootRank]:op==='ReduceScatter'?[j]:ranks;const destinations=dest.map(dst=>({dst,di:op==='ReduceScatter'?0:j}));destinations.forEach(({dst,di})=>output[dst][di]=value);math.push({sources,value,destinations,label:`下标 ${j}`});});
 }else if(op==='Broadcast'){input[rootRank]=[7,8];ranks.forEach(dst=>input[rootRank].forEach((_,i)=>add(rootRank,i,dst,i)));}
 else if(op==='Scatter'){counts=variable?[1,2,1]:[1,1,1];input[rootRank]=Array.from({length:counts.reduce((a,b)=>a+b)},(_,i)=>i+1);let k=0;ranks.forEach(dst=>{for(let i=0;i<counts[dst];i++)add(rootRank,k++,dst,i);});}
 else if(op==='Gather'||op==='AllGather'){input=variable?[[1],[2,3],[4]]:[[1],[2],[3]];counts=input.map(a=>a.length);(op==='Gather'?[rootRank]:ranks).forEach(dst=>{let di=0;ranks.forEach(src=>input[src].forEach((_,si)=>add(src,si,dst,di++)));});}
 else if(op==='AllToAll'||op==='Neighbor'){
  counts=op==='Neighbor'?[[0,1,0],[1,0,1],[0,1,0]]:variable?[[1,2,0],[0,1,1],[2,0,1]]:[[1,1,1],[1,1,1],[1,1,1]];
  ranks.forEach(src=>{ranks.forEach(dst=>{for(let k=0;k<counts[src][dst];k++)input[src].push(`${src}→${dst}${k?"b":""}`);});});
  const next=[0,0,0];ranks.forEach(src=>{let si=0;ranks.forEach(dst=>{for(let k=0;k<counts[src][dst];k++)add(src,si++,dst,next[dst]++);});});
 }else if(op==='SendRecv'){input[0]=[7,8];if(!opts.unmatched)input[0].forEach((_,si)=>add(0,si,1,si));}
 else if(op==='Put'||op==='Get'){const src=op==='Put'?0:1,dst=1-src;input[src]=[7,8];input[src].forEach((_,si)=>add(src,si,dst,si));}
 else if(op==='Atomic'){const cas=opts.atomic==='cas';input=[cas?[opts.expected,9]:[2],[5],[]];output=[cas?[5]:[],[cas?(opts.expected===5?9:5):7],[]];}
 return {op,root:rootRank,variable,reduce,input,output,flows,math,counts,...opts};
}
// Column-major examples preserve the rank order of contiguous gathered pieces.
function matrixExample(op){
 const x=[0,2,4,6,1,3,5,7],y0=[0,20,40,60,0,60,120,180],y1=[20,60,100,140,40,120,200,280];
 const input=op==='AllGather'?[[20,80,140,200],[40,180,320,460]]:op==='AllReduce'?[y0,y1]:[x,[]];
 const output=[[],[]],flows=[],math=[];
 const add=(src,si,dst,di)=>{const value=input[src][si];output[dst][di]=value;flows.push({src,si,dst,di,value,id:`${src}-${si}-${dst}-${di}`});};
 if(op==='AllReduce'){y0.forEach((v,i)=>{const value=v+y1[i],destinations=[0,1].map(dst=>({dst,di:i}));destinations.forEach(({dst})=>output[dst][i]=value);math.push({sources:[{src:0,si:i},{src:1,si:i}],value,destinations});});}
 else if(op==='Broadcast')[0,1].forEach(dst=>x.forEach((_,i)=>add(0,i,dst,i)));
 else if(op==='Scatter')[0,1].forEach(dst=>{for(let i=0;i<4;i++)add(0,dst*4+i,dst,i);});
 else [0,1].forEach(dst=>[0,1].forEach(src=>input[src].forEach((_,i)=>add(src,i,dst,src*4+i))));
 return {op,matrix:true,root:0,reduce:'sum',input,output,flows,math,rows:4,inputColumns:input.map(v=>v.length/4),outputColumns:output.map(v=>v.length/4)};
}
const configs=[
 ['Broadcast',{root:0},'root = Rank 0'],['Broadcast',{root:2},'root = Rank 2'],
 ['Scatter',{},'等长分片'],['Scatter',{variable:true},'Scatterv · 不等长分片'],
 ['Gather',{},'等长片段 → Rank 0'],['Gather',{variable:true,root:2},'Gatherv · 不等长 → Rank 2'],
 ['AllGather',{},'等长片段'],['AllGather',{variable:true},'AllGatherv · 不等长片段'],
 ['Reduce',{},'SUM → Rank 0'],['Reduce',{reduce:'max',root:2},'MAX → Rank 2'],
 ['AllReduce',{},'SUM · 逐元素求和'],['AllReduce',{reduce:'max'},'MAX · 逐元素取最大'],
 ['ReduceScatter',{},'SUM · 等长结果分片'],['Scan',{},'SUM · 包含自身'],['Exscan',{},'SUM · 不含自身'],
 ['AllToAll',{},'AllToAll · 每对一个元素'],['AllToAll',{variable:true},'AllToAllv · 每对数量不同'],
 ['Neighbor',{},'线形邻接 0 ↔ 1 ↔ 2'],
 ['SendRecv',{},'匹配的发送与接收'],['SendRecv',{unmatched:true},'接收缺席 · 不宣告完成'],
 ['Put',{},'Rank 0 写入 Rank 1'],['Get',{},'Rank 0 读取 Rank 1'],
 ['Atomic',{atomic:'sum'},'Accumulate · 远端加 2'],['Atomic',{atomic:'cas',expected:5},'CAS · 比较成功'],['Atomic',{atomic:'cas',expected:4},'CAS · 比较失败'],
 ['Barrier',{},'Rank 2 较晚进入'],['Barrier',{missing:true},'Rank 2 未进入'],
 ['Completion',{mode:'stream'},'NCCL · 跨 stream 依赖'],['Completion',{mode:'notification'},'控制通知先于 KV 载荷'],
 ...['Broadcast','Scatter','AllGather','AllReduce'].map(op=>[op,{matrix:true},'矩阵例子 · '+({Broadcast:'复制完整输入',Scatter:'按列分发',AllGather:'收齐两列',AllReduce:'对应格求和'}[op])])
];
function scenes(){return configs.map(([op,opts,name])=>{
 const def=definitions.find(d=>d.id===op),data=calculate(op,opts);let steps;
 if(data.matrix)steps=[['先看矩阵归属',op==='Broadcast'?'root 持有 4×2 矩阵 X；另一个 rank 尚无这份输入。':op==='Scatter'?'root 持有 4×2 矩阵 X；不同颜色对应将要分发的两列。':op==='AllGather'?'每个 rank 持有一个 4×1 列片段；两列属于同一矩阵的不同位置。':'两个 rank 各有一个 4×2 贡献矩阵，形状相同、数值不同。'],['确定本操作的行为',{Broadcast:'复制整份 X，元素与形状都不变。',Scatter:'第一列交给 Rank 0，第二列交给 Rank 1。',AllGather:'两方都收齐 Rank 0、Rank 1 的列片段，保持来源顺序。',AllReduce:'对两个矩阵的相同坐标求和，每个 rank 都得到完整结果。'}[op]],['数据直接到达参与方','同一颜色跟随同一来源；独立传输并发进行，原输入保留。本图不指定网络算法。'],['得到所需结果',op==='AllReduce'?'对应位置相加，例如第一行第一列 0+20=20；不是把两个矩阵并排加宽。':'每个元素落在约定的位置，数值不变；收集与分发不做加法。'],['核对形状与数值',op==='Scatter'?'输出为每个 rank 一个 4×1 分片。':'两个 rank 都有完整 4×2 输出；图中按列组织元素，颜色保留来源。']];
 else if(data.math.length)steps=[['看清原始输入','每一行属于一个 rank；相同下标表示同一个输出位置的贡献。'],['对齐参与计算的数',op==='Scan'||op==='Exscan'?'每个输出行只取它对应的 rank 前缀；阶梯形阴影显示参与范围。':'每列单独归约。副本按下标进入逻辑计算区，源输入保留。'],['执行归约',data.reduce==='max'?'比较每组贡献，选择最大值，输出长度由该操作约定。':'将同一组的贡献相加；这些算式说明数学含义，不指定网络算法。'],['把结果交给指定参与者',op==='Reduce'?'只有 root 获得归约输出；其他 rank 仍保留自己的输入。':op==='ReduceScatter'?'本例三个结果位置按 rank 分片，每个 rank 只取得一个位置。':op==='Exscan'?'Rank 0 的 MPI_Exscan 接收结果未定义；应用计算偏移时可另行把自己的起点设为 0。':'各份已得到的结果并发交到目的 rank，无需按 rank 排队。本图分开表示计算与交付；真实算法还可按数据块重叠归约与通信。'],['检查结果与用途',op==='Scan'?'[2,3,4] 的包含式前缀和为 [2,5,9]，不同于 AllReduce 的三个 9。':op==='Exscan'?'Rank 1 获得 2，Rank 2 获得 5；前缀和可帮助计算变长片段的起始偏移。':'每份输出都能回查对应的贡献与运算；改变通信语义会改变数学结果或结果归属。']];
 else if(['Barrier','Completion'].includes(op))steps=op==='Barrier'?[['进入前','三条泳道表示同一组的三个进程；横轴仅表示事件先后。'],['Rank 0 到达','Rank 0 等待其他成员，不能越过屏障。'],['Rank 1 到达','Rank 1 也开始等待；此时仍不能放行。'],[opts.missing?'Rank 2 缺席':'Rank 2 到达',opts.missing?'缺席成员使屏障持续等待，不把超时画成成功。':'全部进入后，各方才满足屏障返回条件；不要求同时返回。'],['检查同步边界',opts.missing?'其他两方仍在等待。屏障不会替缺席进程完成工作。':'Barrier 只表达这个组的会合条件，不替无关 GPU 工作或其他传输确认完成。']]:[
 ['区分控制与执行',opts.mode==='stream'?'CPU 调用 NCCL，将通信工作提交到通信 stream；消费者位于另一 stream。':'控制面发送“R1 的 KV 将写入块 B”，数据面另外传输实际字节。'],
 ['提交／通知已经返回',opts.mode==='stream'?'CPU 返回后设备工作仍可在途；本例消费者先建立等待完成事件的依赖。':'收到请求与地址信息时，块 B 仍可能为空；身份信息不是完成证明。'],
 ['载荷正在推进','填充条仅表示本教学场景的传输进度，数据尚未完整可用。'],
 ['相关传输完成',opts.mode==='stream'?'通信 stream 上记录的完成事件发生；消费者 stream 的等待依赖才可满足。':'后端确认相关载荷写完；还要核对请求身份与消费者的可见性、执行依赖。'],
 ['消费者可以继续',opts.mode==='stream'?'消费者在正确依赖之后读取结果；同一 stream 的后续工作则由该 stream 的顺序约束。':'本例再满足身份匹配和消费侧依赖后，才使用块 B。真正完成条件由具体传输协议定义。']];
 else if(op==='Atomic')steps=[['目标已有一个值','Rank 1 窗口中的值是 5；Rank 0 发起更新，Rank 2 不参与。'],['发出原子操作',opts.atomic==='cas'?`CAS（Compare And Swap，比较并交换）：期望 ${opts.expected}，新值 9。`:'Accumulate：把 2 按 SUM 归约到远端值。'],['在目标执行',opts.atomic==='cas'?`读取旧值 5，与期望 ${opts.expected} 比较；判断和条件更新属于同一次原子操作。`:'目标位置执行 5 + 2 = 7。本例只更新一个整数，不推广为整个数组事务。'],['确认结果',opts.atomic==='cas'?'把旧值 5 返回发起方；是否写入 9 取决于比较结果。':'目标更新为 7；MPI_Accumulate 本身不返回旧值。'],['完成与后续访问','窗口、合法访问期和相应完成同步是前提。原子性按所选接口与目标元素定义，不等于整个程序全局同步。']];
 else steps=[['看清谁拥有什么',op==='Get'?'Rank 1 的窗口持有 [7,8]；Rank 0 是发起者，也将是数据接收方。':op==='Put'?'Rank 0 持有 [7,8]，目标是 Rank 1 已暴露窗口的位置。':'源输入始终保留；输出缓冲区用虚线槽表示，本例不使用原地操作。'],
 ['确定交接关系',op==='AllToAll'?'每个数据块标明来源和目标 rank；全交换按目标重新组织，v 变体允许每对数量不同。':op==='Neighbor'?'只沿 0↔1、1↔2 交换；0 与 2 之间没有直接语义边。':op==='SendRecv'?(opts.unmatched?'Rank 1 没有提供匹配接收；本例保留待处理状态。':'Rank 0 的发送对应 Rank 1 的接收；两方同意数量与类型。'):op==='Get'?'Rank 0 提出读请求；请求方向与返回载荷方向相反。':op==='Put'?'Rank 0 指定目标窗口和偏移；目标无需为这一条 Put 单独提交匹配 Recv。':'按 root、rank 顺序与数量约定确定每个片段的目的位置。'],
 ['复制载荷',opts.unmatched?'不能确认接收结果；具体发送侧是否先缓冲返回由接口和协议决定。':'各份数据直接从来源 rank 移到目标 rank，独立传输同时推进；原始输入保留。'],
 ['输出位置写好',opts.unmatched?'接收缺席，接收输出仍不可用。':'每份副本落在约定的输出位置；普通收集与交换不改变元素数值。'],
 ['按契约使用结果',op==='Put'||op==='Get'?'此处假定窗口访问合法且完成同步已满足。Put/Get 不自动保证应用已消费数据，继续阅读完成语义。':opts.unmatched?'需要补齐匹配接收或走错误处理，不能跳到“计算可用”。':'核对各 rank 的输出、元素顺序与数量，再交给后续计算。']];
 const stepDurations=data.matrix?(op==='AllReduce'?[.7,1.15,1.15,2.4]:[.65,1.05,1.05,.7]):data.math.length?[1.1,1.6,1.2,.7]:['Barrier','Completion'].includes(op)?[1,1,1.4,1.2]:op==='Atomic'?[.8,1.1,1.5,.7]:['Put','Get','SendRecv'].includes(op)?[.9,1,1,1.2]:[.65,1,1,.65];
 if(opts.unmatched)steps[4]=['接收尚未完成','接收侧仍没有匹配接收；结果不可用，不能跳到“计算可用”。'];if(op==='Atomic'&&opts.atomic==='cas'&&opts.expected!==5)steps[4]=['比较失败：保留 5','期望 4 与实际 5 不同；目标不写入，发起方得到旧值 5。'];
 return {stepDurations,name,scenarioLabel:name,op,matrix:!!opts.matrix,kind:'communication-contract',group:def.group,title:`${op==='SendRecv'?'Send / Recv':op==='Atomic'?'Accumulate / CAS':op==='Completion'?'异步完成':op}：${def.label}`,intro:def.why,data,frames:steps.map(([title,caption])=>({title,caption})),boundaryKey:'communication-contract',boundary:'教学示例：rank 是组内进程编号。图中呈现数据去向与结果，不指定 ring/tree、物理链路或真实耗时。MPI 扩展语义不表示同名 NCCL 接口或当前 SGLang 已使用它。',evidence:'#evidence'};
 });}
const api={definitions,groups,calculate,scenes};if(typeof module==='object'&&module.exports)module.exports=api;root.CommunicationTeaching=api;
if(root.document)(root.TeachingLessons||(root.TeachingLessons={}))['f-communication']=scenes();
})(typeof globalThis==='object'?globalThis:this);
