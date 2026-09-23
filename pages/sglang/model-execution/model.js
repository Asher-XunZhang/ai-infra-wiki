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
 layer:[srt+'models/llama.py',340,'def forward'],
 attention:[srt+'layers/radix_attention.py',290,'return get_attn_backend().forward'],
 kernelCall:[srt+'layers/attention/triton_backend.py',2301,'self.decode_attention_fwd'],
 kernel:['python/sglang/kernels/ops/attention/decode_attention.py',1156,'def decode_attention_fwd('],
 logits:[srt+'models/llama.py',585,'return self.logits_processor'],
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
 {from:'EagerRunner',to:'模型 forward',action:'进入网络计算',data:'hidden',source:'layer',caption:'token 先变成 hidden states，再经过模型层。模型层会继续调用 Attention 与 MLP。'},
 {from:'模型 Attention 层',to:'Attention backend',action:'交接 Q / K / V',data:'qkv',source:'attention',caption:'Q 查询历史；本轮 K/V 写入对应层的 KV 池。后端不重新决定请求顺序。'},
 {from:'Triton backend',to:'设备算子',action:'用索引读取 KV 并计算',data:'kernel',source:'kernelCall',caption:'Q、每层 KV buffer 与分段索引进入具体实现。一个 Python 入口可以发起多个设备 kernel。'},
 {from:'模型层与输出头',to:'Worker · logits',action:'返回词表分数',data:'logits',source:'logits',caption:'Attention 结果还要经过输出投影、MLP 与后续层。直到输出头，才得到词表分数。'},
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
