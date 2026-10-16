/* Mechanism model: fixed source boundaries, invented payloads and hardware rates. */
(function(root){
'use strict';
const revision='279339f113b79af84f27fd3ac92d0a13bd3f4cbd',srt='python/sglang/srt/';
const sources={
 scheduler:[srt+'managers/scheduler.py',4285,'def run_batch'],
 worker:[srt+'managers/tp_worker.py',593,'def forward_batch_generation'],
 input:[srt+'model_executor/forward_batch_info.py',835,'def init_new'],
 runner:[srt+'model_executor/model_runner.py',1782,'def _forward_raw'],
 eager:[srt+'model_executor/runner/eager_runner.py',244,'def _execute_decode'],
 metadata:[srt+'layers/attention/triton_backend.py',775,'def init_forward_metadata'],
 model:[srt+'models/deepseek_v2.py',2735,'def forward'],
 mla:[srt+'models/deepseek_common/attention_forward_methods/forward_mla.py',761,'attn_output = self.attn_mqa'],
 layer:[srt+'models/deepseek_v2.py',2434,'def forward'],
 attention:[srt+'layers/radix_attention.py',290,'return get_attn_backend().forward'],
 kernelCall:[srt+'layers/attention/triton_backend.py',2301,'self.decode_attention_fwd'],
 kernel:['python/sglang/kernels/ops/attention/decode_attention.py',1156,'def decode_attention_fwd('],
 logits:[srt+'models/deepseek_v2.py',3071,'return self.logits_processor'],
 sample:[srt+'managers/tp_worker.py',672,'batch_result.next_token_ids = self.model_runner.sample'],
 graphGate:[srt+'model_executor/runner/decode_cuda_graph_runner.py',632,'def can_run_graph'],
 graphLoad:[srt+'model_executor/runner/decode_cuda_graph_runner.py',1302,'self._pad_to_bucket(raw_bs, self.capture_bs)'],
 replay:[srt+'model_executor/runner/decode_cuda_graph_runner.py',1431,'self.backend.replay'],
 trim:[srt+'model_executor/runner/decode_cuda_graph_runner.py',1450,'output.next_token_logits[: self.raw_num_token]']
};
const trace=[
 {from:'Scheduler',to:'Worker',action:'交出本轮工作',data:'requests',source:'scheduler',caption:'本轮 R1、R2 各做一个 Decode 位置。谁入选已由 Scheduler 决定。'},
 {from:'Worker',to:'ModelRunner',action:'整理执行输入',data:'forward',source:'worker',caption:'请求对象转换成 ForwardBatch：本轮 token、逻辑位置、长度与 KV 写位置一起交接。'},
 {from:'EagerRunner',to:'Attention backend',action:'先准备本轮索引',data:'indices',source:'eager',caption:'本例需要初始化元数据：先整理每条请求的历史索引，再进入模型计算。'},
 {from:'EagerRunner',to:'模型 forward',action:'进入网络计算',data:'hidden',source:'model',caption:'DeepSeek-V3 把 token 编号变成数字特征，再执行本级模型层。层内经过 MLA 注意力，以及普通前馈网络或按路由选择的 MoE 专家。'},
 {from:'模型 Attention 层',to:'Attention backend',action:'交接查询与压缩历史',data:'qkv',source:'mla',caption:'本例 MLA 路径把变换后的查询、压缩 KV 与位置部分交给 attn_mqa。Triton 后端将本轮状态写入对应槽位；它不重新决定请求顺序。'},
 {from:'Triton backend',to:'设备算子',action:'用索引读取 KV 并计算',data:'kernel',source:'kernelCall',caption:'变换后的 Q、MLA 缓存视图与分段索引进入具体实现。一个 Python 入口可以发起多个设备 kernel。'},
 {from:'模型层与输出头',to:'Worker · logits',action:'返回词表分数',data:'logits',source:'logits',caption:'Attention 结果还要经过输出投影、普通前馈层或 MoE 专家，以及后续层。直到输出头，才得到词表分数。'},
 {from:'Worker 协调采样',to:'Scheduler',action:'返回下一 token',data:'sample',source:'sample',caption:'本例用贪心选择示意：每条请求只提交一个下一 token；Scheduler 再更新请求与下一轮。'}
];
function graph(n=3,condition='ready',step=0){
 if(![1,3,4,5].includes(n))throw Error('Unsupported request count');
 const bucket=[1,2,4].find(b=>b>=n),eligible=condition==='ready'&&bucket!==undefined;
 const reason=condition==='off'?'未启用图 runner':condition==='override'?'动态 embedding override':!bucket?'超过最大图桶 4':'模式、形状和本例条件均满足';
 const real=Array.from({length:n},(_,i)=>'R'+(i+1));
 const slots=step>=2&&eligible?[...real,...Array(bucket-n).fill(null)]:real;
 return {real,bucket:eligible?bucket:null,eligible,reason,slots,outputs:step>=4?[...real]:[],path:eligible?'graph':'eager'};
}
function matrix(m=1,upgrade='base'){
 if(![1,16,256].includes(m))throw Error('Unsupported row count');
 const k=4096,n=4096,flops=2*m*k*n,bytes=2*(m*k+k*n+m*n);
 const compute=8e12*(upgrade==='compute'?2:1),bandwidth=100e9*(upgrade==='memory'?2:1);
 const mathMs=flops/compute*1000,memoryMs=bytes/bandwidth*1000;
 return {m,k,n,flops,bytes,weightBytes:2*k*n,intensity:flops/bytes,mathMs,memoryMs,lowerMs:Math.max(mathMs,memoryMs),limit:mathMs>memoryMs?'math':'memory'};
}
const api={revision,sources,trace,graph,matrix};if(typeof module==='object'&&module.exports)module.exports=api;root.ExecutionModel=api;
})(typeof globalThis==='object'?globalThis:this);
