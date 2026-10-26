(function(root){
'use strict';
const M=root.ParallelTeachingModel;if(!M)return;
const scope='教学拓扑：两个 rank 各用一张 GPU，同机不同进程；跨机放置也可能，但不在本图推导具体链路或耗时。编号只在本例内有效。数值为手算示例，不是模型权重或实测性能。';
const make=(name,kind,intro,steps,data={},boundary='')=>({name,kind:'parallel-'+kind,title:name,intro,boundary:scope+' '+boundary,data,frames:steps.map(([title,caption])=>({title,caption})),boundaryKey:'parallel-'+kind});
const tp=['row','column'].map(kind=>make(kind==='row'?'TP · 切输入维：局部乘积求和':'TP · 切输出维：结果按列拼接','tp','同一个线性层 Y = XW：X 的两行是两个 token 的输入特征，W 把两个输入特征变成两个输出特征。先看完整矩阵，再切分、计算和恢复结果。',[
 ['完整问题：两行输入乘同一份权重','X 为 2×2：行表示 token，列表示输入特征。W 为 2×2：行对应输入特征，列对应输出特征。本例没有偏置与激活函数。'],
 ['先算一个输出格','Y 的左上角取 X 第一行与 W 第一列：1×2 + 2×1 = 4。其余格采用相同规则；矩阵乘法不是对应位置相乘。'],
 ['标出切分轴',kind==='row'?'沿相乘的输入特征维切：X 切列、W 切对应行。同色部分必须成对交给同一 rank。':'沿输出特征维切：W 切列，X 在两卡各保留完整副本。两卡各自负责不同的输出列。'],
 ['把对应分块放到两张卡',kind==='row'?'Rank 0 接 X 第 1 列与 W 第 1 行；Rank 1 接 X 第 2 列与 W 第 2 行。左侧参考矩阵只作原问题对照。':'两卡使用相同 X；Rank 0 持有 W 第 1 列，Rank 1 持有 W 第 2 列。复制的是输入激活，权重按列分片。'],
 ['各卡计算局部矩阵',kind==='row'?'两份局部乘积均为 2×2；同一坐标是同一输出格的部分贡献。两卡同时计算；图逐格揭示算式，真实 GEMM 可并行计算。':'两卡同时计算，每卡得到 2×1 输出列：第一卡为 [4,10]，第二卡为 [6,12]。输出列延续所属权重列的颜色。'],
 ['交换需要的结果分片',kind==='row'?'本例用两卡交换局部乘积，再逐元素相加说明 AllReduce 的语义；本地贡献保持可见。':'本例用 AllGather 交换输出列的副本。原输出列保持不动，各卡取得缺少的那一列。'],
 ['按元素含义汇总',kind==='row'?'同一输出坐标相加：例如 2+2=4、6+4=10；每个 rank 都得到完整 Y，合并通信另计。':'两列同时移到原列号对应的位置：第一列 [4,10]、第二列 [6,12]；每个 rank 都得到完整 Y。'],
 ['核对完整结果','两种切法都恢复 Y=[[4,6],[10,12]]。实际网络可继续使用分片输出，未必每层都 AllGather；这里为核对结果选择完整恢复。']
 ],M.tensor(kind),'采用 Y=XW 行向量布局；输入维切分对应 Row Parallel，输出维切分对应 Column Parallel。切分动画解释逻辑布局，不表示每次推理都重新切权重。两卡交换是 collective 语义示例，不限定 NCCL 算法。'));
const dp=make('DP · 完整模型副本与请求分派','dp','无论 1 份还是 2 份完整模型副本，每条请求都执行 L1(x)=2x、L2(h)=h+1；R1 输入 1，R2 输入 3。先看参数复制与请求分派，再观察两条输入分别计算并返回。',[
 ['同一份完整模型','用两个标量运算缩小模型，只研究请求分工；x 是输入特征，不是词表 ID。'],
 ['建立 1 或 2 份完整参数副本','1 副本：A 保存 L1 的 ×2 与 L2 的 +1，共 2 个参数块；2 副本：A、B 各保存一份，共 4 个参数块。副本内部仍可再组合 TP / PP。'],
 ['把请求送入完整副本','1 副本时 R1、R2 进入同一份共享执行容量，可同批也可先后；2 副本时 R1 → A、R2 → B。分派属于服务控制路径，不改变每请求的两层计算。'],
 ['L1 完成：h=2 / 6','R1 的 L1 完成，h=2；R2 的 L1 完成，h=6。两条请求都在自己所在的完整模型内乘 2；1 副本不等于只能串行，2 副本也不让单条请求少算一层。'],
 ['L2 完成：y=3 / 7','R1 的 L2 完成，y=3；R2 的 L2 完成，y=7。没有交换或合并两条请求的中间结果。'],
 ['返回到各自调用方','R1 的调用方收到 3，R2 的调用方收到 7；两值不相加。单副本共享容量可合批或先后，2 副本是两条独立路径，都不是单请求加速比。'],
 ['前后账本回顾','底部账本对照两种条件：1 副本 2 个参数块、1 条共享容量；2 副本 4 个参数块、2 条独立路径。两种条件每请求都算 2 层；3 / 7 是预期核对值，返回到达后才成为已返回值。']
],{},'两层标量算例只解释推理 DP，不模拟语言生成、KV 增长或训练梯度同步。');
dp.boundary='DP 教学边界：同一 R1 x=1、R2 x=3，每请求固定两层 L1 ×2 → L2 +1。1 副本有 2 个参数块和一条共享执行容量，可合批也可先后；2 副本有 4 个参数块和两条独立路径，输出分别返回、不相加。两层标量算例只解释推理 DP，不模拟语言生成、KV 增长或训练梯度同步，也不把副本数当作单请求加速比。';
const ep=[false,true].map(skew=>{
 const d=M.experts(skew),hot=skew;
 const dispatch=hot
  ?'未采用（1 GPU）：8 个分支全部在 GPU0 本地，0 个跨卡。采用后（2 GPU）：t1 的 E0/E1 都在 Rank 0 本地；4 个跨卡分支来自 t2/t3/t4，其中 t4 的两条从 Rank 1 跨到 Rank 0。派发只去选中的专家，不是广播；画面错开只为看清，不代表执行先后。'
  :'未采用（1 GPU）：8 个分支全部在 GPU0 本地，0 个跨卡。采用后（2 GPU）：t1 的 E0 在 Rank 0 本地，E2 跨到 Rank 1；t2/t3/t4 也各有一条跨卡分支，共 4 去。派发只去选中的专家，不是广播；画面错开只为看清，不代表执行先后。';
 const returned=hot
  ?'未采用（1 GPU）：t1 的 '+d.outputs[0]+' 与 '+d.outputs[1]+' 都在 GPU0 本地，0 个跨卡返回。采用后（2 GPU）：t1 的两条输出仍都在 Rank 0 本地；4 个跨卡分支各返回一次，共 4 回，其中 t4 的两条从 Rank 0 回到 Rank 1。'
  :'未采用（1 GPU）：8 个分支全部本地返回，0 个跨卡返回。采用后（2 GPU）：E0 的 '+d.outputs[0]+' 在本地返回，E2 的 '+d.outputs[1]+' 跨回 Rank 0；t2/t3/t4 也各有一条跨卡返回，共 4 回。';
 return make(skew?'EP · 热点路由：6 / 2 为什么仍偏载':'EP · 同路由对照：1 GPU 4 专家 vs 2 GPU 各 2 专家','ep','Expert Parallelism（专家并行）把 MoE 的专家参数放到不同 rank。同一组 t1…t4、同一 top-2 路由与同一标量专家函数下，比较 1 张 GPU 放 4 个专家和 2 张 GPU 各放 2 个专家；追踪 t1 的两条分支，并核对采用前后结果。',[
  ['同一输入、同一路由：先对齐两种放置','t1…t4 是四个输入位置，四个专家是四组不同参数，不是四份完整 LLM。每个 token 选 top-2，共 8 个专家分支；t1 的标量特征 x=2，权重 0.75 / 0.25。此刻只对齐路由与专家放置，尚未开始派发。'],
  ['放置 4 组专家参数：1 卡 vs 2 卡','未采用：GPU0 保存 E0–E3 共 4 组专家参数。采用后：每卡 2 组，E0/E1 在 Rank 0，E2/E3 在 Rank 1；总专家仍 4 组。变化是参数放置与 GPU 数，不是专家函数或路由。'],
  ['按同一路由派发 8 个分支',dispatch],
  ['所选专家分别计算','E0(t1)='+d.outputs[0]+'、E'+d.selected[1]+'(t1)='+d.outputs[1]+'；两条输出保持独立。同一路由条件内，采用前后使用相同的专家函数、选中集合与权重，所以输出相同。其它 token 在本例只更新路由与负载计数。'],
  ['Combine：两条分支按原 token 身份返回',returned+' 本阶段只完成返回；下一步分别乘权重，再把两项贡献相加。'],
  ['按路由权重汇合：前后结果相同',d.outputs[0]+' × '+d.weights[0]+' = '+(d.outputs[0]*d.weights[0])+'、'+d.outputs[1]+' × '+d.weights[1]+' = '+(d.outputs[1]*d.weights[1])+'；再把两项贡献相加得到 '+d.result+'。未采用（1 GPU）：8 次激活都在 GPU0，8 个分支也都本地，参数 4 组同在 GPU0，没有两 rank 负载分配。采用后（2 GPU）：'+(hot?'热点下 E0 被选 4 次，E0/E1 又同在 Rank 0，两 rank 激活负载 6/2；':'均衡时两 rank 各 4 次激活；切到热点后 E0 被选 4 次、E1 两次，负载变成 6/2；')+'EP 只换放置，不换路由，同一路由条件下前后结果相同。']
 ],d,'EP 教学边界：单卡与两卡使用同一组 t1…t4、同一 top-2 路由、同一权重与同一标量函数 E_i(x)=(i+1)x；每个路由条件内采用前后结果相同。8 个分支是 top-2 × 4 tokens，MoE 稀疏激活不是 EP 带来的收益。参数 4→2 是每卡专家参数组数，不是加速比；跨卡 dispatch/combine 本例按 4+4 计数，不模拟网络时延、带宽或吞吐。');
});
const sp=make('SP · 序列分片与 TP 布局转换','sp','Sequence Parallelism（序列并行）：采用 Megatron 经典 SP 定义，在 LayerNorm 等逐 token 区域按序列分片激活，并通过 AllGather / ReduceScatter 衔接 TP 区域。画布内可切换 TP 基线与 TP+SP；两种情况始终是同一 TP 组的两张 GPU、同一组 4 行。',[
 ['同一输入：4 个位置，每行两维','p1…p4 是同一序列的四个位置；每行 [a_i,b_i] 是该位置完整的两维特征。TP 与 TP+SP 使用同一输入、同一输出定义；切换条件只改变逐 token 区域的激活行副本。'],
 ['同一 2 GPU 上的行分工','TP 基线：逐 token 区域每卡复制 4 行，共 8 行副本。TP+SP：GPU0 持 p1/p2，GPU1 持 p3/p4，各 2 行，共 4 行。切的是序列行，不是每行的特征列。'],
 ['逐 token 算子在本地完整行上执行','LayerNorm 等逐位置算子可独立处理每个位置。TP 基线每卡处理 n1…n4；TP+SP 每卡只处理本地两行。n_i 只表示变换后的行，不伪造归一化数值。'],
 ['AllGather 收齐进入 TP 区域所需的行','TP+SP 用 AllGather 收齐 n1…n4；本地行保留，副本补齐到 4/4 后才进入需要完整序列的 TP 区域。TP 基线本来每卡已有 4 行，无需这步序列收齐，但 TP 区域仍有自己的 collective。'],
 ['TP 区域产生部分贡献，还不是最终行','TP 区域按 TP 规则对 4 个位置分别得到部分贡献 c_i^r；c_i^0 与 c_i^1 需要按同一位置相加，此时不能提前写成完整 y_i。'],
 ['同位置求和，再按所属 rank 得到完整行','y_i = c_i^0 + c_i^1。TP+SP 由 ReduceScatter 求和后只把 y1/y2 留给 GPU0、y3/y4 留给 GPU1；TP 基线由 AllReduce 让两卡都保留 4 行。两条件最终输出相同。']
],{},'这是训练框架中常用的 SP 前向布局示意，用于区分概念；推理时通常不启用 Dropout。它不是 SGLang 通用启动配置，也不等同于将完整 Attention 历史切开。同 2 GPU、同 4 行：TP 基线每卡 4 行，TP+SP 每卡 2 行；AG/RS 是布局转换的通信语义，不把总通信写成固定新增两次，也不把激活分片等同于 KV 分片。');
const pcp=[false,true].map(zigzag=>{
 const d=M.context(zigzag);
 const q7=d.queries[6];
 const remote=q7.requiredRemote.map(i=>'K'+(i+1));
 const remoteText=remote.length?remote[0]+'–'+remote[remote.length-1]:'无';
 const rowsText=rows=>rows.map(i=>'Q'+(i+1)).join(' ');
 const splitName=zigzag?'前后配对':'连续切分';
 return make(
  zigzag?'PCP · 前后配对，平衡因果计算量':'PCP · 连续切分，观察因果负载',
  'pcp',
  'Prefill Context Parallelism（预填充上下文并行）让多个 rank 共同处理一条长 prompt。各 rank 负责部分 Query 行，同时必须取得这些行需要的 K/V。',
  [
   ['同一条长请求的八个位置','p1…p8 是同一请求的八个 Query 位置，不是八条请求，也不是八个调度轮次。Qi 只允许读 K1…Ki：Q1 有 1 个配对，Q8 有 8 个；总数 1+…+8=36。'],
   ['把 Query 行分给两张卡',splitName+'：Rank 0 负责 '+rowsText(d.ranks[0])+'（'+d.work[0]+' 对），Rank 1 负责 '+rowsText(d.ranks[1])+'（'+d.work[1]+' 对）。Query 归谁，O_i 就归谁；归属改变不改变每行自己的因果范围。'],
   ['交换所需的 K/V','本图选 AllGather 局部 K/V 的示例：原 K/V 留在本 rank，副本按位置送到对端。Q7 需要 K1…K7，其中远端 '+remoteText+'；远端副本逐份到达，齐备前不能当成已收齐。部分副本到达不等于 collective 完成，全量进度在画布中单列。'],
   ['收齐后才计算自己的行','所需 K/V 齐备后才计算：Q1 只读本地 K1，没有远端依赖，不需要等副本；Q7 等远端 '+remoteText+' 的副本到齐才开算。本图按所需数据齐备开算，开算先后只是教学顺序，不是实测耗时；部分到达不当作 AllGather 完成。'],
   ['保留各自负责的输出行','O1…O8 跟随各自 Query 的 rank，不做跨 rank 求和。单卡 36、连续 10/26、前后 18/18 都是允许配对数；计数不是耗时，也不承诺加速。']
  ],
  d,
  'PCP 是 Prefill 阶段的一类 CP 分工；与 Chunked Prefill 按时间分多轮处理不同，二者可以组合。'
 );
});
const dcp=[false,true].map(equal=>make(equal?'DCP · 两片归一化权重相等':'DCP · 两片归一化权重不同','dcp','Decode Context Parallelism（解码上下文并行）把同一请求的历史 KV 按上下文位置分散保存；一个新 Query 必须汇合各片 Attention 的贡献。',[
 ['先看同一请求的完整历史','本例 Q=1，K 为标量，score=Q×K（特征维度为 1）。图先展示完整历史，再按位置分片。Rank 0 保存 p1/p2，Rank 1 保存 p3/p4。新 Query Q 来自同一个 Decode 位置；这里只展示它读取四个历史位置的子过程。'],
 ['Q 已到两张卡','同一个 Q 从中央分发到两张卡；两卡都持有 Q，仍只读各自本地 K/V，尚未算出局部 O/Z。本图按中央 Q 同时送达两卡示意，实际分发或复制关系依赖 TP / DCP 布局。'],
 ['只读本地 K/V，尚未交换','Q 到达后，各卡读取本地的 p1/p2 与 p3/p4；局部 O/Z 的逐项计算尚未开始，交换门尚未打开。'],
 ['逐项算局部 O/Z','各卡读取本地 K/V 后，按 Q×K 得到指数分数，再累加 Z、分子并得到局部 O；仍只读本地 K/V，尚未交换。'],
 ['交换后合并','两卡交换 O/Z（实际实现通常交换 LSE 并稳定合并）；两片都到齐后按全局 Z 加权，得到全局 O=5 或 4。只有 Z0=Z1 时才等于简单平均。']
 ],M.decode(equal),'用一维 V 演示注意力合并，省略新位置 KV 写入和完整模型前向；不限定 collective 算法，也不推导 SGLang 当前版本是否支持某个启动组合。'));
const pp=root.TeachingLessons['f-parallel'].map(s=>{
 const n=Number(s.data?.n)||2;
 const frames=n===2?[
  {title:'先看完整模型的层顺序',caption:'L1 到 L4 是依次执行的四层。单卡保存全部四层权重；两 stage 各保存两层，权重留在负责该层的 GPU。'},
  {title:'第 1 格：两侧同时开始 L1',caption:'单卡与 GPU 0 都开始 MB1 L1；GPU 1 还没有激活，当前只显示第 1 个等待格，第 2 个等待格在下一格出现。1 格表示一层计算，两侧使用同一刻度。'},
  {title:'第 2 格：L1 完成，L2 开始',caption:'L1 完成后才进入 L2；GPU 0 正在算 MB1 L2，只有 L2 完成后才能把中间激活交给 GPU 1。'},
  {title:'交接：L2 完成，激活在途',caption:'MB1 L2 已完成，中间激活正从 GPU 0 送往 GPU 1；传输不另计格，但到达前 L3 不能执行。'},
  {title:'第 3 格：激活到达，L3 开始',caption:'MB1 激活到达后 GPU 1 才开始 L3；GPU 0 同期算 MB2 L1。这是不同 MB 的重叠，不是同一 MB 跨 stage 并行。'},
  {title:'第 4 格：MB1 L4 与 MB2 L2 重叠',caption:'GPU 1 算 MB1 L4，GPU 0 算 MB2 L2。同一个 MB 仍按 L1→L4 依赖；只有不同 MB 才能让两个 stage 同时忙。'},
  {title:'第 5 格：MB2 激活到达，下游继续',caption:'MB2 L2 完成并发激活，到达后 GPU 1 算 MB2 L3；GPU 0 没有新 MB，进入排空。'},
  {title:'第 6 格：两 stage 的最后一格',caption:'GPU 1 算 MB2 L4，GPU 0 继续排空。两 stage 在第 6 格结束；单卡此时还在第 6 格，后面还有 L3、L4。'},
  {title:'第 7 格：两 stage 已完成，单卡继续',caption:'两 stage 保持 6 格完成；单卡还在算 MB2 L3，当前 7/8 格。对比两边的进度，流水收益来自不同 MB 的重叠。'},
  {title:'第 8 格：单卡最后一层',caption:'单卡算 MB2 L4，第 8 格才完成；两 stage 仍停在 6 格，不再增加 GPU 格。'},
  {title:'回顾：两个 MB 才有重叠',caption:'单卡 8 格、两 stage 6 格；总层工作仍是 8 格，2×6=12 GPU·格中有 4 格是填充/排空。完成格数不是性能倍数。'}
 ]:[
  {title:'先看完整模型的层顺序',caption:'L1 到 L4 是依次执行的四层。这里只送入一个 MB；单卡与两 stage 使用同一组层和同一 1 格单位。'},
  {title:'第 1 格：两侧开始 L1',caption:'单卡与 GPU 0 都开始 MB1 L1；GPU 1 没有激活，当前只显示第 1 个等待格，第 2 个等待格在下一格出现。'},
  {title:'第 2 格：L1 完成，L2 开始',caption:'L1 完成后才进入 L2；GPU 0 正在算 MB1 L2，只有 L2 完成后才能把激活交给 GPU 1。'},
  {title:'交接：L2 完成，激活在途',caption:'MB1 L2 已完成，中间激活正从 GPU 0 送往 GPU 1；传输不另计格，但到达前 L3 不能执行。'},
  {title:'第 3 格：激活到达，L3 开始',caption:'MB1 激活到达后 GPU 1 才开始 L3；GPU 0 已经算完两层，没有第二个 MB 可接，开始排空。'},
  {title:'第 4 格：L4 执行，两侧同格结束',caption:'GPU 1 算 MB1 L4；单卡也在第 4 格完成。两 stage 用两张 GPU，但完成时刻没有提前。'},
  {title:'回顾：单 MB 没有流水收益',caption:'两侧同 4 格；2×4=8 GPU·格，工作 4 格、气泡 4 格。没有第二个 MB 填空，不能把“两卡”当作吞吐收益。'}
 ];
 return {...s,intro:'同一四层模型、同一 MB 数量：上面比较单卡，下面比较两 stage。1 格 = 1 层计算；激活到达后才可执行下游；传输延迟未计。',boundaryKey:'parallel-pp',frames,stepDurations:n===2?[1.6,1.8,2.2,1.8,2.0,1.8,1.8,1.8,1.7,1.8]:[1.6,1.8,2.2,1.8,1.9,2.0]};
});
root.TeachingLessons['f-parallel']=[...tp,...pp,dp,...ep,sp,...pcp,...dcp];
const labels=['沿输入维切分','沿输出维切分','两个微批次','一个微批次','完整模型副本','均衡路由','热点专家','序列布局转换','连续位置分片','前后配对分片','归一化权重不同','归一化权重相同'];
const pacing={
 'parallel-tp':[2.2,1.5,1.6,3.2,1.4,2.8,1.1],
 'parallel-dp':[1.1,1.4,1.8,1.8,1.4,1.6],
 'parallel-ep':[1.6,1.5,2,1.4,2.6],
 'parallel-sp':[1.4,1.8,1.6,1.8,2.6],
 'parallel-pcp':[1.6,1.5,2.6,1.5],
 'parallel-dcp':[1.6,3.6,1.4,3.2]
};
root.TeachingLessons['f-parallel'].forEach((s,i)=>{s.scenarioLabel=labels[i];s.stepDurations=pacing[s.kind]||s.stepDurations||s.frames.slice(1).map((_,step)=>step<2?1.3:1.8);});
})(typeof globalThis==='object'?globalThis:this);
