/* Source-pinned implementation paths. All durations are explanatory seconds. */
(function(root){
'use strict';
const revision='279339f113b79af84f27fd3ac92d0a13bd3f4cbd';
const sources={
 widths:['runtime_context.py',151,'def derive_parallel_widths('],
 groups:['distributed/parallel_state.py',2507,'def initialize_model_parallel('],
 column:['layers/linear.py',492,'def forward('],row:['layers/linear.py',1612,'def forward('],
 mlp:['models/deepseek_v2.py',302,'def forward('],
 layers:['utils/common.py',1415,'def make_layers('],
 ppModel:['models/deepseek_v2.py',2735,'def forward('],
 ppLoop:['managers/scheduler_pp_mixin.py',63,'def event_loop_pp('],
 ppSend:['managers/scheduler_pp_mixin.py',802,'def _pp_send_dict_to_next_stage('],
 ppRecv:['managers/scheduler_pp_mixin.py',858,'def _pp_recv_proxy_tensors('],
 dpLaunch:['managers/data_parallel_controller.py',371,'def launch_dp_schedulers('],
 dpRoute:['managers/data_parallel_controller.py',767,'def round_robin_scheduler('],
 prepareAttn:['layers/communicator.py',657,'def prepare_attn('],
 prepareMlp:['layers/communicator.py',860,'def prepare_mlp('],
 dpGather:['layers/communicator.py',1234,'def _gather_hidden_states_and_residual('],
 dpScatter:['layers/communicator.py',1499,'def _scatter_hidden_states('],
 ep:['layers/moe/ep_moe/layer.py',252,'def forward_impl('],
 deepEP:['layers/moe/token_dispatcher/deepep.py',1011,'def dispatch('],
 spGate:['layers/layernorm_sp.py',72,'def runs_sp('],
 spEntry:['layers/layernorm_sp.py',107,'def sp_entry_scatter('],
 spColumn:['layers/layernorm_sp.py',198,'def column_parallel_g_matmul('],
 spRow:['layers/layernorm_sp.py',222,'def row_parallel_gbar_matmul('],
 spExit:['layers/layernorm_sp.py',127,'def sp_exit_gather('],
 cpGate:['layers/cp/zigzag.py',109,'def can_apply('],
 cpMeta:['layers/cp/zigzag.py',121,'def build_metadata('],
 cpKV:['layers/cp/zigzag.py',396,'def materialize_full_kv('],
 cpAttention:['layers/cp/zigzag.py',343,'def run_attention('],
 cpExit:['layers/cp/utils.py',224,'def cp_gather_after_forward('],
 dcpLayout:['layers/dcp/layout.py',44,'def maybe_dcp_kernel_indices('],
 dcpQ:['layers/dcp/comm.py',254,'def all_gather_q_for_mla_decode('],
 dcpMerge:['layers/dcp/comm.py',116,'def cp_lse_ag_out_rs_mla('],
 dcpLSE:['layers/attention/flashinfer_mla_backend.py',813,'# for decode and dcp_world_size > 1'],
};
const modes=['tp','pp','dp','dpa','ep','sp','pcp','dcp'];
const catalog={
 tp:['TP','线性层怎样接住上一层的分片？','DeepSeek-V3 的 dense MLP · TP=2 · 常规未融合规约路径','两个 rank 是同一层的协作者；每卡一份权重分片。H 是隐藏维度，F 是 MLP 中间维度，T 是本轮 token 行数。'],
 pp:['PP','下一 stage 究竟收到什么，何时能算？','DeepSeek-V3 · PP=2、TP=1 · event_loop_pp · 普通非 PD、非投机路径','两级是不同 Scheduler 进程；主机位置未指定。只跟踪一个 microbatch 的因果链，不用等长格子模拟真实排程。'],
 dp:['普通 DP','Controller 怎样把请求交给某个实例？','DP=2、TP=1、PP=1 · ROUND_ROBIN · 无 routed_dp_rank','两个独立模型副本各有 Scheduler、请求队列和 KV 池。R1–R3 是三个请求，IPC 箭头携带内部请求对象。'],
 dpa:['Attention DP','不同请求的激活，怎样一起进入共享的 MLP？','DeepSeek-V3 dense 层 · TP=2、Attention DP=2、CP=1 · 关闭规约融合','Attention TP=1；两个 rank 处理不同请求。图中 MLP 仍由这两个 rank 共同执行，没有第三个中央进程。'],
 ep:['EP','MoE 怎样把 token 送到专家，再恢复 token 顺序？','DeepSeek-V3 MoE · DeepEP dispatcher · EP=2、MoE TP=1','只画两个 token 及被选中的代表专家；top-k=2、权重 0.75/0.25 是缩小的教学配置，不是 DeepSeek-V3 默认值。'],
 sp:['SP','LayerNorm 为什么能只持有一部分 token 行？','Qwen3 dense · TP=2 · enable_layernorm_sp · 普通未融合路径','该提交的支持列表只有 Qwen3ForCausalLM；SP 仅用于 EXTEND，与 Attention DP、投机配置不兼容。这里明确切换模型。'],
 pcp:['Prefill CP','一次 Prefill 怎样分配 Q，又怎样让 Q 看到所需 K/V？','Qwen3-MoE / GQA · TP=2、CP=2 · Zigzag · FA3 · 无命中前缀','t0–t7 是同一请求的 token 位置。只跟踪 Attention 分支和模型末尾恢复顺序；MoE 层间布局另由 LayerCommunicator 处理。'],
 dcp:['Decode CP','KV 分散在两卡后，怎样得到完整 Attention 输出？','DeepSeek-V3 / MLA · TP=2、DCP=2 · FlashInfer MLA · ag_rs','DCP 的 KV 布局从 Prefill 就已建立。这里选择非 replicated-Q 路径；不把 Prefill CP 与 DCP 当作同一次 forward 的连续步骤。'],
};
const phase=(id,title,duration,summary,detail,refs)=>({id,title,duration,summary,detail,refs});
function groups(tp,pp=1,dp=1,cp=1,ep=1){
 for(const v of [tp,pp,dp,cp,ep])if(!Number.isInteger(v)||v<1)throw Error('Positive integer widths required');
 if(tp%(dp*cp)||tp%ep)throw Error('Widths must divide the TP world');
 const atp=tp/dp/cp,range=(n,start=0,stride=1)=>Array.from({length:n},(_,i)=>start+i*stride);
 return {world:tp*pp,attnTP:atp,moeTP:tp/ep,tp:range(pp).map(s=>range(tp,s*tp)),pp:range(tp).map(r=>range(pp,r,tp)),attn:range(pp*dp*cp).map(g=>range(atp,g*atp)),cp:range(pp*dp).flatMap(g=>range(atp).map(r=>range(cp,g*tp/dp+r,atp)))};
}
function dispatch(count,healthy=[true,true]){let cursor=0;const owners=[];if(!healthy.some(Boolean))throw Error('No active healthy worker');for(let i=0;i<count;i++){while(!healthy[cursor%healthy.length])cursor++;owners.push(cursor++%healthy.length);}return owners;}
function sequenceRows(n=5,tp=2){const padded=Math.ceil(n/tp)*tp;return Array.from({length:tp},(_,r)=>Array.from({length:padded/tp},(_,i)=>{const p=r*padded/tp+i;return p<n?p:null;}));}
function zigzag(n){if(n<4)return null;const sizes=Array.from({length:4},(_,i)=>Math.floor(n/4)+(i<n%4?1:0)),blocks=[];let p=0;for(const size of sizes){blocks.push(Array.from({length:size},()=>p++));}return [blocks[0].concat(blocks[3]),blocks[1].concat(blocks[2])];}
function mergeAttention(outputs,lse,base=2){if(outputs.length!==lse.length||!outputs.length)throw Error('Mismatched partials');const m=Math.max(...lse),w=lse.map(v=>base**(v-m)),sum=w.reduce((a,b)=>a+b,0),weights=w.map(x=>x/sum);return {weights,output:outputs.reduce((a,b,i)=>a+b*weights[i],0)};}
function build(mode='tp',option='default'){
 if(!modes.includes(mode))throw Error('Unknown implementation path');
 const [label,question,config,boundary]=catalog[mode];let phases,data={};
 if(mode==='tp')phases=[
  phase('group','同一层，两个本地权重分片',1.8,'initialize_model_parallel 建立 TP 组 [0,1]；ModelRunner 中的模型在各 rank 构造本地线性层。','输入 X[T,H] 已在两卡可用；不额外虚构每层广播。本图沿 DeepseekV2MLP 普通路径，省略 bias、量化和 fused GEMM。',['groups','mlp']),
  phase('gate','gate_up_proj 产生本地 gate / up',3,'MergedColumnParallelLinear 生成两个 [T,F/2] 分片。','gather_output=False：保留本地中间维度；随后 SiluAndMul 在本地配对 gate 与 up，不先 All-gather 全部中间结果。',['column','mlp']),
  phase('down','SiluAndMul → down_proj 的局部贡献',3,'两个 rank 各产生 [T,H]，但每份仅包含自己的中间维度贡献。','RowParallelLinear 消费本地 F/2 输入片。相同 shape 不代表完整结果；仍需把两个贡献按位置相加。',['row','mlp']),
  phase('reduce','TP All-reduce：相同位置求和',3.4,'Y = Y₀ + Y₁；两个 rank 都得到相同的完整输出。','选择 reduce_results=True 且不跳过规约的分支。其他模型调用可由 LayerCommunicator 延迟、融合或替换规约，本页不把这次 All-reduce 写成每层必有。',['row']),
  phase('output','完整 hidden_states 交给后续层',1.4,'输入行身份没有改变；改变的是它们的特征表示。','跨 GPU 载荷是局部输出张量；权重仍驻留在各自 rank，不随 token 一起移动。',['mlp'])];
 if(mode==='pp')phases=[
  phase('group','make_layers 只实例化本级层',2,'每个 stage 有独立 Scheduler 和 ModelRunner；首级有 Embedding，末级负责输出头。','层区间由 get_pp_indices 决定。图中“本级层”表示真实 start_layer:end_layer 范围，未人为把模型改成四层。',['layers','ppModel']),
  phase('request','先向下一级传递请求对象',2.5,'event_loop_pp 转发 recv_reqs，各级按相同请求顺序组织 microbatch。','控制路径携带请求及调度所需信息；它不能替代模型下一层所需的 GPU 激活。',['ppLoop']),
  phase('first','Stage 0 执行自己的 Decoder 层',3,'Embedding → 本级模型层；本级 Attention 写本级 KV。','只展开同一 microbatch 的因果链。实际 loop 可以重叠其他 microbatch 和结果处理，动画长度不代表独占 GPU 耗时。',['ppModel']),
  phase('proxy','PPProxyTensors 跨级传递激活',3.6,'载荷是 hidden_states + residual；__msg_type__ = proxy。','发送等待本级 launch_event；接收端 _pp_recv_proxy_tensors 收到 proxy 后才能把它交给模型。本级 KV 留在原 stage。所选非 DSA 路径不含 topk_indices。',['ppSend','ppRecv','ppModel']),
  phase('last','Stage 1 从 proxy 继续后半模型',3,'读取 hidden_states / residual → 本级层 → final norm / logits / sampling。','下一级不是再次 tokenize 或重算 Embedding，而是继续同一 microbatch 的模型表示。',['ppModel','ppLoop']),
  phase('result','输出消息反馈到流水线调度',3,'输出 token ID 等结果沿 output 通道反馈，更新相应 microbatch。','proxy 是向后续模型层传的激活；output 是末级产生、供请求结果处理使用的输出。示意只画两级，不展开异步队列深度。',['ppLoop']),
  phase('output','两种消息，两类仍留本地的状态',1.4,'请求对象 → 调度；激活 → 下一级计算；输出 ID → 结果更新。','权重与本级 KV 始终在各自 stage。P/D KV 传输属于后续分离部署课。',['ppModel'])];
 if(mode==='dp'){
  data.owners=dispatch(3,option==='skip'?[false,true]:[true,true]);data.healthy=option!=='skip';
  phases=[phase('group','启动两个独立 Scheduler 副本',1.8,'launch_dp_schedulers 为各副本创建独立模型并行组与入口 socket。','本例每副本 TP=1、PP=1；一般普通 DP 每个副本可包含 TP×PP 个模型进程。',['dpLaunch']),
  ...data.owners.map((owner,i)=>phase('route'+i,`R${i+1} → 副本 ${owner}`,i===0?3:2.2,`round_robin_scheduler 选中可用槽位 ${owner}，sock_send 发送内部请求对象。`,option==='skip'?'副本 0 的 status=False；轮转跳过它。这里演示已经给定的健康状态，不模拟检测或恢复协议。':'Controller 选择接收方；请求进入副本后，由该副本的 Scheduler 决定何时组批，不由 Controller 直接发 GPU kernel。',['dpRoute'])),
  phase('compute','两个 Scheduler 各自组批和执行',3,'请求历史与 KV 归所属副本维护，副本之间不为同一层交换计算贡献。','请求卡片的归属保持不变。并行的两条执行线只表达独立性，不承诺两边同时完成。',['dpLaunch']),
  phase('output','结果沿各自请求通道返回',1.8,'R1、R2、R3 都只有一个被选择的副本。','外部 routed_dp_rank、预算路由和 PD bootstrap 路由是独立分支；本页固定 ROUND_ROBIN。',['dpRoute'])];
 }
 if(mode==='dpa')phases=[
  phase('group','derive_parallel_widths：Attention TP = 1',2,'TP=2 / Attention DP=2 / CP=1；Attention 是 [0]、[1]，MLP TP 组仍是 [0,1]。','这个案例只有两个模型进程，不把 Attention DP 再乘成四个。',['widths','groups']),
  phase('attention','两边对不同请求执行 Attention',3,'rank 0 处理 R1 的 a0、a1；rank 1 处理 R2 的 b0、b1；各自读取本地 KV。','行标签 a/b 表示请求身份。Attention 完成后，每组仍持有自己的 hidden_states，而不是对方的 KV。',['prepareAttn']),
  phase('gather','prepare_mlp：收集并对齐 token 行',3.8,'LayerCommunicator 调用 dp_gather_replicate，把 [a0,a1]、[b0,b1] 组织为相同全局行序。','选择 Attention TP=1 的 gather 分支，示例两组行数相等。实际缓冲还根据 global_num_tokens 和 padding mode 确定有效行与偏移。',['prepareMlp','dpGather']),
  phase('mlp','原来两卡共同执行 MLP TP',3,'每卡都处理 [a0,a1,b0,b1] 的不同权重分片，再按 TP 路径完成规约。','图中四行是激活，不是把请求队列或 KV 池合并。此处选择普通 dense 层、关闭规约融合。',['mlp']),
  phase('scatter','postprocess_layer：取回自己的输出行',3.5,'dp_scatter 按本组偏移，将完整结果切回 a 行与 b 行。','本分支是从已规约的全局激活缓冲作本地取片；不能看到 scatter 就画成新的网络派发。其他分支可采用 Reduce-scatter。',['dpScatter']),
  phase('output','各自的 token 行进入下一层 Attention',1.6,'MLP 共享协作范围，Attention 恢复各组的请求范围。','两份本地 KV 从未随激活 gather 移到对方。',['prepareAttn'])];
 if(mode==='ep')phases=[
  phase('group','MoE 层已有 hidden_states 与路由输入',1.8,'只跟踪 tA、tB 的专家分支；每个 rank 保有自己的专家权重。','共享专家分支不在本图展开。E0/E1 是两个代表专家，非完整专家数量。',['ep']),
  phase('topk','TopKOutput 保留专家 ID 与权重',2.8,'tA → E0(0.75)、E1(0.25)；tB → E1(0.75)、E0(0.25)。','一个 token 产生两份带身份的专家任务；topk_ids 与 topk_weights 共同进入 dispatcher。',['ep']),
  phase('dispatch','dispatcher.dispatch 重排成专家输入',3.8,'两卡分别得到属于本地专家的 token 行；派发同时保留返程所需映射。','选择 DeepEP 后端；跨卡载荷包含 hidden states 与路由信息。图示语义派发，不描绘底层 RDMA 包或固定网卡拓扑。',['ep','deepEP']),
  phase('experts','run_moe_core 执行本地专家权重',3,'每个专家分别变换收到的行，得到不同的专家输出。','任务编号不等于新 token ID。图中不编造真实模型输出数值，也不以任务数推断耗时。',['ep']),
  phase('combine','dispatcher.combine 反向归并',3.8,'按返程映射恢复 tA、tB，并按路由权重合并各专家贡献。','权重参与合并的具体落点由 kernel/dispatcher 配合完成；图画最终加权语义，不声称 combine 内一定重复乘权重。',['ep']),
  phase('output','回到原 token 顺序的 MoE 输出',1.8,'tA 的两份贡献合成一行，tB 的两份贡献合成另一行。','原 token 的请求归属保留；EP 不负责把 HTTP 请求分到独立服务副本。',['ep'])];
 if(mode==='sp'){
  data.rows=sequenceRows();data.active=option!=='decode';
  phases=data.active?[
   phase('group','检查模型与本轮 ForwardMode',1.8,'Qwen3ForCausalLM、TP>1、开关开启，且本轮为 EXTEND。','当前固定提交拒绝不支持的模型、Attention DP 与投机组合。',['spGate']),
   phase('slice','sp_entry_scatter：补齐后本地取片',3,'5 行补到 6 行；rank 0 取 t0–t2，rank 1 取 t3、t4、∅。','输入已经在 TP 各卡复制；entry_scatter 是本地 tensor_split，不是跨卡发送。∅ 是 padding 行，不是第六个 token。',['spEntry']),
   phase('norm','LayerNorm / residual 在本地行上处理',2.3,'两卡只保留各自的序列片，进行逐行归一化和残差处理。','LayerCommunicator 的 SP variant 保持这些区域为 SCATTERED。',['prepareAttn']),
   phase('gather','Column 线性层先 All-gather 序列行',3.4,'column_parallel_g_matmul 收齐真实 token 行，再用本地 QKV / gate_up 权重计算。','在普通路径中，sp_exit_gather 会去掉 padding 后执行本地矩阵乘法。Attention 与 MLP 的线性层参与各自的 g/g-bar 配对。',['spColumn']),
   phase('reduce','Row 线性层按 token 维 Reduce-scatter',3.5,'本地 GEMM 贡献按位置求和后，结果分回两个序列片。','row_parallel_gbar_matmul 在需要时补齐行数。这里的 Reduce-scatter 同时完成求和与分发，不是简单截断张量。',['spRow']),
   phase('exit','模型出口 Gather 并丢弃 padding',3,'LM head 前恢复 [t0,t1,t2,t3,t4] 的完整顺序。','后续 Decoder 层重复同类配对；本图概括为一个代表区间。输出头看到真实行数 5，不含 ∅。',['spExit'])]:[
   phase('group','相同开关，换成 DECODE',2,'runs_sp 检查 ForwardMode：本轮不等于 EXTEND。','支持某模型不等于所有 forward 都启用 SP。',['spGate']),
   phase('bypass','sp_active=False，走普通 TP 路径',2.5,'不会进入 entry_scatter、SP g/g-bar 和 SP exit gather。','保持已有 TP 张量布局；不把未执行的 SP 通信画成在后台运行。',['spGate','prepareAttn'])];
 }
 if(mode==='pcp'){
  data.count=option==='short'?3:8;data.rows=zigzag(data.count);data.active=Boolean(data.rows);
  phases=data.active?[
   phase('group','can_apply 判断本轮能否 Zigzag',1.8,'CP=2，每条请求本轮 extend 长度至少为 4；本例长度 8。','模式也要满足 is_context_parallel_extend。开关开启不是唯一条件。',['cpGate']),
   phase('shard','build_metadata → 首尾块本地分片',3.5,'rank 0：t0,t1,t6,t7；rank 1：t2,t3,t4,t5，position IDs 同步分片。','metadata 保存 split_len、反向索引等，用于恢复正确 token 顺序。这里没有前缀命中。',['cpMeta']),
   phase('qkv','各自为局部 token 计算 Q/K/V',2.4,'Q 继续只负责自己的查询行；新 K/V 还只是局部片。','图中 GQA 使用真实 K/V 表示，不用 MLA latent 代替。',['cpKV']),
   phase('kv','materialize_full_kv：Gather → 重排 → 写缓存',3.8,'两片 K/V 按原序恢复为 t0–t7，再写入各卡本层 KV buffer。','实际先拼接 K/V 载荷、Gather，再按维度拆回 K、V；Q 不跟着扩成全部查询。',['cpKV']),
   phase('attn','run_attention 对本卡 Q 执行 Attention',3,'FA3 路径按 prev / next 两段的元数据执行，并拼回本地结果。','每张卡查询位置不同，所需上下文由元数据给定。这里不重演第一章的因果矩阵工作量比较。',['cpAttention']),
   phase('exit','模型结束后恢复 hidden_states 的全局行序',3.3,'Gather 各 rank 的结果，并按 reverse index 恢复 t0–t7，再交给 logits。','这是模型出口的 hidden states 汇合，与每层 K/V 汇合是两次不同载荷的通信。',['cpExit'])]:[
   phase('group','CP=2，但 extend 只有 3 个 token',2,'3 < 2×CP，ZigzagCPStrategy.can_apply 返回 False。','逐请求长度也受检查，不能只看整个 batch 的 token 总数。',['cpGate']),
   phase('bypass','本轮 CP 不激活',2.5,'保持该模型的非 CP forward 路径，不生成 Zigzag 分片。','不把不满足门槛的输入强塞进两个 rank，再补画一条不存在的 Gather。',['cpGate'])];
 }
 if(mode==='dcp'){
  data.merge=mergeAttention([2,6],[1,3]);
  phases=[phase('group','KV 已按 DCP owner 规则分散持有',2.3,'示例 widened slots 0–5：偶数属于 rank 0，奇数属于 rank 1；本地行号为 slot//2。','槽号为教学设定，不等同于所有请求的 position ID；真实地址还可能由 pool translator 翻译。',['dcpLayout']),
   phase('query','All-gather 该 DCP 组的 Q heads',3,'q_pe 与 q_nope 先打包，沿 head 维汇合，再拆回查询张量。','每卡读取本地 KV，却需要同一组完整 Q，才能形成可合并的局部 Attention 结果。',['dcpQ']),
   phase('local','FlashInfer MLA 产出局部 O 与 LSE',3.2,'每卡只用本地 KV 计算；返回局部输出与对应归一化信息。','图中的 O₀=2、O₁=6、log₂ 分母 LSE₀=1、LSE₁=3 是独立标量算例，不是模型实测输出。',['dcpLSE']),
   phase('lse','All-gather LSE，校正局部贡献',3.6,'分母贡献 2¹:2³=1:4 → 权重 0.2 / 0.8。','FlashInfer MLA 的 LSE 为 base-2；correct_attn_out 必须使用匹配的指数底。只平均 O₀、O₁ 会得到错误结果。',['dcpMerge']),
   phase('reduce','Reduce-scatter 沿 head 维合并',3.4,'0.2×2 + 0.8×6 = 5.2；合并后各卡取回自己负责的头。','并非把所有 KV Gather 到每张卡。图示一个头的数值；实际为多头多维张量，返回本地 head 分片。',['dcpMerge']),
   phase('output','后续投影消费本地 head 输出',1.6,'Q 在组内汇合，O/LSE 用于结果合并，KV 仍分散持有。','只演示 ag_rs 后端；a2a / fi_a2a 是另一实现分支，不能拼成同一条通信流程。',['dcpMerge'])];
 }
 let end=0;phases=phases.map((p,index)=>({...p,index,start:end,end:(end+=p.duration)}));
 return {mode,option,label,question,config,boundary,phases,duration:end,data};
}
function sample(timeline,time){
 time=Math.max(0,Math.min(timeline.duration,Number.isFinite(time)?time:0));
 const phase=timeline.phases.find(p=>time<p.end)||timeline.phases.at(-1);
 const progress={};for(const p of timeline.phases)progress[p.id]=time>=p.end?1:time<=p.start?0:(time-p.start)/p.duration;
 return {time,phase,progress,finished:time===timeline.duration};
}
const url=key=>`https://github.com/sgl-project/sglang/blob/${revision}/python/sglang/srt/${sources[key][0]}#L${sources[key][1]}`;
const api={revision,sources,modes,catalog,build,sample,groups,dispatch,sequenceRows,zigzag,mergeAttention,url};
if(typeof module==='object'&&module.exports)module.exports=api;else root.ParallelismFlow=api;
})(typeof globalThis!=='undefined'?globalThis:this);
