/* Scope maps for the fixed ordinary Python runtime; labels are teaching data. */
(function(root){
'use strict';
const revision='279339f113b79af84f27fd3ac92d0a13bd3f4cbd',p='python/sglang/srt/';
const sources={startup:[p+'entrypoints/engine.py',1051,'def _launch_subprocesses'],http:[p+'entrypoints/http_server.py',911,'async def generate_request'],dispatch:[p+'managers/tokenizer_manager.py',1592,'async def _send_one_request'],loop:[p+'managers/scheduler.py',1907,'def event_loop_normal'],prepare:[p+'managers/schedule_batch.py',2678,'def prepare_for_extend'],worker:[p+'managers/tp_worker.py',593,'def forward_batch_generation'],sample:[p+'managers/tp_worker.py',672,'batch_result.next_token_ids = self.model_runner.sample'],write:[p+'mem_cache/memory_pool.py',2526,'def set_kv_buffer'],result:[p+'managers/scheduler_components/batch_result_processor.py',257,'def process_batch_result_prefill'],detokenize:[p+'managers/detokenizer_manager.py',443,'def handle_batch_token_id_out'],front:[p+'managers/tokenizer_manager.py',2255,'def _handle_batch_output'],column:[p+'layers/linear.py',492,'def forward'],row:[p+'layers/linear.py',1612,'def forward'],dp:[p+'managers/data_parallel_controller.py',767,'def round_robin_scheduler'],dpLaunch:[p+'managers/data_parallel_controller.py',371,'def launch_dp_schedulers'],handoff:[p+'disaggregation/decode.py',2158,'def _commit_transfer_to_req']};
const modules={runtime:['02 请求运行时','../request-runtime/'],execution:['03 模型执行','../model-execution/'],memory:['04 KV 与内存','../kv-memory/'],scheduling:['05 调度与批处理','../scheduling/'],parallel:['06 并行拓扑','../parallelism/'],communication:['07 通信与传输','../communication/'],pd:['08 分离部署','../inference-overview/deployment.html'],generation:['09 模型与生成','../advanced-generation/'],serving:['10 服务治理','../serving-operations/'],performance:['11 性能分析','../performance-engineering/'],practice:['12 综合实践','../pd-prefill-pp-loop/quick.html']};
const service=[
 {from:'client',to:'front',path:'M140 40 V89',x:104,y:64,payload:'请求',source:'http',title:'入口接收请求',caption:'客户端先把生成请求交给 HTTP 入口；此时还没有模型结果。',links:['serving','runtime']},
 {from:'front',to:'scheduler',path:'M120 143 L80 223',x:78,y:181,payload:'p0 · p1',source:'dispatch',title:'跨 IPC 传递内部请求',caption:'TokenizerManager 把文本变成 token IDs 和配置，再派发到 Scheduler。派发不等于已经开始计算。',links:['runtime','scheduling']},
 {from:'scheduler',to:'scheduler',path:'M62 277 V304 H98 V277',x:150,y:304,payload:'前向 · 采样',source:'worker',title:'调度进程内组织执行',caption:'Scheduler 选批，调用同进程的 Worker / Runner。它们协调设备计算；不是再经过两个网络服务。',links:['execution','memory']},
 {from:'scheduler',to:'detokenizer',path:'M135 250 H165',x:150,y:217,payload:'y1',source:'result',title:'输出 ID 进入反分词进程',caption:'本轮产生的 token ID 先经过后端结果处理，再发往反分词。采样发生了，客户端还未必可见。',links:['communication','generation']},
 {from:'detokenizer',to:'front',path:'M220 223 L180 143',x:226,y:181,payload:'文本片段',source:'detokenize',title:'文字按请求身份返回入口',caption:'Detokenizer 将 ID 转成文本结果；TokenizerManager 按 rid 找到对应的响应等待者。',links:['runtime','communication']},
 {from:'front',to:'client',path:'M160 89 V40',x:212,y:64,payload:'流式块',source:'http',title:'客户端收到可见内容',caption:'本图假设连接正常。流式块与 token 未必一一对应；测量首字延迟时要包含实际回程。',links:['performance','practice']}
];
const inside=[
 {from:'scheduler',to:'cache',path:'M180 122 L220 188',x:225,y:153,payload:'匹配 · 准入',source:'loop',title:'先确定本轮能做什么',caption:'Scheduler 结合队列与资源决定本轮计划。缓存索引和分配器提供状态与容量信息。',links:['scheduling','memory']},
 {from:'cache',to:'scheduler',path:'M204 188 L160 122',x:155,y:152,payload:'写位置',source:'prepare',title:'准备本轮输入与地址',caption:'本例接纳 R1，为两个新位置准备写地址。拿到地址还不代表 KV 值已经写好。',links:['memory','runtime']},
 {from:'scheduler',to:'runner',path:'M120 122 L80 188',x:70,y:153,payload:'本轮 batch',source:'worker',title:'调用 Worker / Runner',caption:'执行层得到本轮 token、长度与地址等输入。此处是进程内调用，不能画成新的 IPC 跳转。',links:['execution','generation']},
 {from:'runner',to:'gpu',path:'M80 242 L130 285',x:70,y:263,payload:'计算 · 写 KV',source:'write',title:'设备计算并写入历史状态',caption:'各模型层计算本轮结果，写入各自的 K/V。图只画两个位置，省略真实张量维度。',links:['execution','memory']},
 {from:'gpu',to:'runner',path:'M110 285 L64 242',x:183,y:263,payload:'词表分数',source:'worker',title:'模型结果尚不是最终文本',caption:'整套模型得到词表分数，之后还要采样与结果处理。这里只把模型内部计算压缩成一个设备节点。',links:['execution','generation']},
 {from:'runner',to:'scheduler',path:'M100 188 L140 122',x:145,y:153,payload:'y1',source:'sample',title:'采样结果交回 Scheduler',caption:'本例已写 p0、p1 的 KV，并采样 y1；y1 尚未作为输入执行，所以还没有自己的 KV。',links:['runtime','performance']}
];
function frame(mode='service',step=0){const list=mode==='service'?service:mode==='inside'?inside:null;if(!list||!Number.isInteger(step)||step<0||step>=list.length)throw Error('Unknown frame');return {...list[step],step,kv:mode==='service'?(step>=2?2:0):(step>=3?2:0),sampled:mode==='service'?(step>=2?1:0):(step>=5?1:0),visible:mode==='service'&&step===5?1:0};}
function topology(kind='tp'){
 if(!['tp','dp','pd'].includes(kind))throw Error('Unknown topology');
 return {kind,replicas:kind==='tp'?1:2,ranks:2,layers:[0,1,2],fractions:kind==='tp'?[.5,.5]:[1,1],assignments:kind==='tp'?[['R1'],['R1']]:kind==='dp'?[['R1'],['R2']]:[['R1 · P'],['R1 · D']],source:kind==='tp'?'row':kind==='dp'?'dp':'handoff',links:kind==='pd'?['communication','pd']:['parallel','serving']};
}
const api={revision,sources,modules,service,inside,frame,topology};if(typeof module==='object'&&module.exports)module.exports=api;root.SystemOverviewModel=api;
})(typeof globalThis==='object'?globalThis:this);
