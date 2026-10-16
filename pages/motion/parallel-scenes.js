(function(root){
'use strict';
const M=root.ParallelTeachingModel;if(!M)return;
const scope='教学拓扑：两个 rank 各用一张 GPU，同机不同进程；跨机放置也可能，但不在本图推导具体链路或耗时。编号只在本例内有效。数值为手算示例，不是模型权重或实测性能。';
const make=(name,kind,intro,steps,data={},boundary='')=>({name,kind:'parallel-'+kind,title:name,intro,boundary:scope+' '+boundary,data,frames:steps.map(([title,caption])=>({title,caption})),boundaryKey:'parallel-'+kind});
const tp=['row','column'].map(kind=>make(kind==='row'?'TP · 切输入维：局部乘积求和':'TP · 切输出维：结果按列拼接','tp','同一个线性层 Y = XW：X 的两行是两个 token 的输入特征，W 把两个输入特征变成两个输出特征。先看完整矩阵，再切分、计算和恢复结果。',[
 ['完整问题：两行输入乘同一份权重','X 为 2×2：行表示 token，列表示输入特征。W 为 2×2：行对应输入特征，列对应输出特征。本例没有偏置与激活函数。'],
 ['先算一个输出格','Y 的左上角取 X 第一行与 W 第一列：1×2 + 2×1 = 4。其余格采用相同规则；矩阵乘法不是对应位置相乘。'],
 ['标出切分轴',kind==='row'?'沿相乘的输入特征维切：X 切列、W 切对应行。同色部分必须成对交给同一 rank。':'沿输出特征维切：W 切列，X 在两卡各保留完整副本。两卡各自负责不同的输出列。'],
 ['把对应分块放到两张卡',kind==='row'?'Rank 0 接 X 第 1 列与 W 第 1 行；Rank 1 接 X 第 2 列与 W 第 2 行。上方淡色矩阵只作原问题参照。':'两卡使用相同 X；Rank 0 持有 W 第 1 列，Rank 1 持有 W 第 2 列。复制的是输入激活，权重按列分片。'],
 ['各卡计算局部矩阵',kind==='row'?'两份局部乘积均为 2×2；同一坐标是同一输出格的部分贡献。图逐格揭示结果，真实 GEMM 可并行计算。':'每卡得到 2×1 输出列：第一卡为 [4,10]，第二卡为 [6,12]。图逐格揭示点积结果。'],
 ['交换需要的结果分片',kind==='row'?'本例用两卡交换局部乘积，再逐元素相加说明 AllReduce 的语义；本地贡献保持可见。':'本例用 AllGather 交换输出列的副本。原输出列保持不动，各卡取得缺少的那一列。'],
 ['按元素含义汇总',kind==='row'?'同一输出坐标相加：例如 2+2=4、6+4=10。不能把两份矩阵当作不同列直接拼接。':'按原列号恢复位置：第一列 [4,10]、第二列 [6,12]。它们不是同一坐标的贡献，不能相加。'],
 ['核对完整结果','两种切法都恢复 Y=[[4,6],[10,12]]。实际网络可继续使用分片输出，未必每层都 AllGather；这里为核对结果选择完整恢复。']
 ],M.tensor(kind),'采用 Y=XW 行向量布局；输入维切分对应 Row Parallel，输出维切分对应 Column Parallel。切分动画解释逻辑布局，不表示每次推理都重新切权重。两卡交换是 collective 语义示例，不限定 NCCL 算法。'));
const dp=make('DP · 复制完整模型，分派不同请求','dp','两个副本都执行 L1(x)=2x、L2(h)=h+1；R1 输入 1，R2 输入 3。先看完整模型如何复制，再观察两条独立输入经过相同计算。',[
 ['同一份完整模型','用两个线性标量运算缩小模型，只研究请求分工；x 是输入特征，不是词表 ID。'],
 ['建立两份完整参数副本','A、B 都保留 L1 的 ×2 和 L2 的 +1。它们有相同参数，分别由独立进程和 GPU 执行。'],
 ['不同请求分别进入副本','R1 输入 1，R2 输入 3；每条请求只走一个完整副本。图中输入分派属于服务控制路径。'],
 ['各自执行第一层','两个副本都乘 2：R1 得 2，R2 得 6。相同模型不要求输入和结果相同。'],
 ['各自执行第二层','分别加 1，得到 3 与 7。没有交换或合并两条请求的中间结果。'],
 ['分别返回各自调用方','R1 收到 3，R2 收到 7；两者不相加。真实模型副本内部可以组合 TP / PP，复制并不自动加速单条请求。']
 ],{},'两层标量算例只解释推理 DP，不模拟语言生成、KV 增长或训练梯度同步。');
const ep=[false,true].map(skew=>make(skew?'EP · 热点专家与不均匀负载':'EP · 路由、派发与加权合并','ep','Expert Parallelism（专家并行）把 MoE 的专家放到不同 rank。E0/E1 位于 Rank 0，E2/E3 位于 Rank 1；追踪 token t1 的两条专家分支。',[
 ['完整输入与专家路由表','t1…t4 是四个输入位置。聚焦 t1 的标量特征 x=2；路由表每行只选择两列。四个专家是四组不同参数；不表示四个完整模型副本。下面的负载计数来自四个教学 token 的路由。'],
 ['Router 为 t1 选择两个专家','本例 top-2 路由，权重为 0.75 和 0.25；t1 的两份分支仍属于同一个输入位置。'],
 ['Dispatch 到目标专家','目标在本 rank 时本地交接；目标在远端 rank 时交换 token 激活及路由信息。派发不等于把每个 token 广播给全部专家。'],
 ['所选专家分别计算','专家输出是中间特征，本例用标量方便手算；未被选中的专家不处理 t1。'],
 ['Combine 返回原 token 位置','结果按 token 身份回到原位置。返回路径仍可能跨进程与 GPU。'],
 ['按路由权重汇合','对两个专家结果分别乘 0.75、0.25 后相加。热点专家收到更多 token 时，均匀放置参数也不能保证负载均匀。']
 ],M.experts(skew)));
const sp=make('SP · 序列分片与 TP 布局转换','sp','Sequence Parallelism（序列并行）：这里采用 Megatron 经典 SP 定义，在 LayerNorm 等逐 token 算子处按序列分片激活，并通过 AllGather / ReduceScatter 衔接 TP 计算。',[
 ['完整激活按序列切分','上方四行分别是四个 token 的完整特征。沿序列维切行，特征列保持完整。本例 Rank 0 持有位置 p1/p2，Rank 1 持有 p3/p4；每行仍包含该 token 的完整特征维。'],
 ['完成序列行分工','Rank 0 保留位置 1/2，Rank 1 保留位置 3/4。a、b 代表特征分量，每个 token 的完整特征保存在同一卡。'],
 ['逐 token 算子在本地处理','LayerNorm 等算子可独立处理每个位置。图用 n1…n4 表示变换后的激活，不伪造归一化数值。'],
 ['AllGather 衔接 TP 区域','需要完整序列的 TP 计算区收齐激活行的副本；权重与特征计算仍按 TP 规则切分。'],
 ['TP 计算产生部分输出贡献','这里只画 TP 区域的输入／输出布局，算子细节见 TP 示例；部分贡献尚不能直接作为完整输出。'],
 ['ReduceScatter：先求和再按序列分发','同一输出位置的 TP 贡献求和后，各 rank 仅保留分给自己的序列行；恢复 SP 布局。']
 ],{},'这是训练框架中常用的 SP 前向布局示意，用于区分概念；推理时通常不启用 Dropout。它不是 SGLang 通用启动配置，也不等同于将完整 Attention 历史切开。');
const pcp=[false,true].map(zigzag=>make(zigzag?'PCP · 前后配对，平衡因果计算量':'PCP · 连续切分，观察因果负载','pcp','Prefill Context Parallelism（预填充上下文并行）让多个 rank 共同处理一条长 prompt。各 rank 负责部分 Query 行，同时必须取得这些行需要的 K/V。',[
 ['同一条长请求的八个位置','p1…p8 是序列位置，不是八条请求，也不是八个调度轮次。'],
 ['把 Query 行分给两张卡',zigzag?'前后配对：Rank 0 处理 p1/p2/p7/p8，Rank 1 处理 p3/p4/p5/p6。':'连续切分：Rank 0 处理 p1…p4，Rank 1 处理 p5…p8。后半段的 Query 要读取更多历史。'],
 ['交换所需的 K/V','本图选 AllGather 局部 K/V 的示例：本地 Query 保持归属，K/V 副本交给另一 rank。其他 CP 算法可循环传块，不要求总是物化全量副本。'],
 ['按因果遮罩计算自己的行','矩阵一行对应一个 Query，一列对应一个 Key；仅下三角允许读取。蓝/绿行分别由两个 rank 计算。'],
 ['保留各自负责的输出行','输出按 Query 位置归属返回后续层。计数是允许的 Q–K 配对数；不是实测 GPU 工作量或加速比。']
 ],M.context(zigzag),'PCP 是 Prefill 阶段的一类 CP 分工；与 Chunked Prefill 按时间分多轮处理不同，二者可以组合。'));
const dcp=[false,true].map(equal=>make(equal?'DCP · 两片归一化权重相等':'DCP · 两片归一化权重不同','dcp','Decode Context Parallelism（解码上下文并行）把同一请求的历史 KV 按上下文位置分散保存；一个新 Query 必须汇合各片 Attention 的贡献。',[
 ['先看同一请求的完整历史','本例 Q=1，K 为标量，score=Q×K（特征维度为 1）。图先展示完整历史，再按位置分片。Rank 0 保存 p1/p2，Rank 1 保存 p3/p4。新 Query Q 来自同一个 Decode 位置；这里只展示它读取四个历史位置的子过程。'],
 ['Query 送到参与 rank','本例 Q 从 Rank 0 复制到 Rank 1；实际分发或复制关系依赖 TP / DCP 布局，不必搬走所有历史 KV。'],
 ['分别得到局部结果与归一化量','每片计算局部输出 O 和分母 Z=sum(exp(score))。指数分数由上方 Q、K 推出，再将每个 V 按指数分数加权求和、除以 Z。'],
 ['交换局部输出和统计量','全局结果需要两个分片的归一化信息。实际实现常交换 log-sum-exp（LSE），通过稳定的缩放合并。'],
 ['按全局归一化权重合并','全局 O=(Z0×O0+Z1×O1)/(Z0+Z1)。只有 Z0=Z1 时才等于简单平均，不能机械地相加或平均局部 Attention 输出。']
 ],M.decode(equal),'用一维 V 演示注意力合并，省略新位置 KV 写入和完整模型前向；不限定 collective 算法，也不推导 SGLang 当前版本是否支持某个启动组合。'));
const pp=root.TeachingLessons['f-parallel'].map(s=>({...s,boundaryKey:'parallel-pp',frames:[{title:'先看完整模型的层顺序',caption:'L1 到 L4 是依次执行的四层。本例把相邻两层交给同一个 PP stage，层的权重留在负责该层的 GPU。'},{title:'按层划分 Stage，再准备微批次',caption:'Stage 0 负责 L1/L2；Stage 1 负责 L3/L4。MB1 与 MB2 各包含两条请求；相同微批次必须依次经过两个阶段。'},...s.frames]}));
root.TeachingLessons['f-parallel']=[...tp,...pp,dp,...ep,sp,...pcp,...dcp];
const labels=['沿输入维切分','沿输出维切分','两个微批次','一个微批次','完整模型副本','均衡路由','热点专家','序列布局转换','连续位置分片','前后配对分片','归一化权重不同','归一化权重相同'];
root.TeachingLessons['f-parallel'].forEach((s,i)=>s.scenarioLabel=labels[i]);
})(typeof globalThis==='object'?globalThis:this);
