/* Source-grounded teaching timelines. Reading seconds are not GPU timings. */
(function(root){'use strict';
const M=typeof module==='object'&&module.exports?require('./model.js'):root.ExecutionModel;
const sources={...M.sources,modelReturn:['python/sglang/srt/models/deepseek_v2.py',2857,'if not self.pp_group.is_last_rank:'],finalNorm:['python/sglang/srt/models/deepseek_v2.py',2885,'if not forward_batch.forward_mode.is_idle():'],v3:['python/sglang/srt/models/deepseek_v2.py',3165,'class DeepseekV3ForCausalLM'],resolve:['python/sglang/srt/managers/overlap_utils.py',87,'def resolve_forward_inputs'],eagerLoad:['python/sglang/srt/model_executor/runner/eager_runner.py',183,'def load_batch'],eagerExtend:['python/sglang/srt/model_executor/runner/eager_runner.py',273,'def _execute_extend'],returnCPU:['python/sglang/srt/managers/utils.py',166,'self.next_token_ids = _async_d2h']};
function layout(){const nodes={},edges={};return {nodes,edges,height:628,
 node(id,x,y,w,h,title,sub){nodes[id]={id,x,y,w,h,title,sub};},
 wire(id,a,b,kind,label,points=[],labelPosition){
  const port=([id,side,offset=.5])=>{const n=nodes[id];return side==='l'?[n.x,n.y+n.h*offset]:side==='r'?[n.x+n.w,n.y+n.h*offset]:side==='t'?[n.x+n.w*offset,n.y]:[n.x+n.w*offset,n.y+n.h];};
  const start=port(a),end=port(b);edges[id]={id,from:a[0],to:b[0],kind,label,points:[start,...points,end],labelPosition:labelPosition||[(start[0]+end[0])/2,(start[1]+end[1])/2-13]};
 }};}
function geometry(view){const g=layout(),n=g.node.bind(g),w=g.wire.bind(g);
 if(view==='model'){
  for(let s=0;s<4;s++)n('stage'+s,28+s*252,32,224,120,'PP Stage '+s,s===0?'Embedding → 本级层区间':s===3?'本级层区间 → Norm / 输出头':'接收中间状态 → 本级层区间');
  for(let s=0;s<3;s++)w('pp'+s,['stage'+s,'r'],['stage'+(s+1),'l'],'activation','h / r');
  n('norm1',30,252,164,90,'prepare_attn','残差整理 + RMSNorm');n('mla',232,252,164,90,'self_attn · MLA','参考同层上下文');n('norm2',434,252,164,90,'prepare_mlp','残差合并 + RMSNorm');n('mlp',636,252,164,90,'mlp','依层配置 Dense / MoE');n('layerout',838,252,164,90,'postprocess_layer','整理两路层输出');
  w('norm-attn',['norm1','r'],['mla','l'],'activation','h');w('attn-norm',['mla','r'],['norm2','l'],'activation','h');w('norm-mlp',['norm2','r'],['mlp','l'],'activation','h');w('mlp-out',['mlp','r'],['layerout','l'],'activation','h');
  n('layerkv',232,390,164,84,'本层 MLA 缓存','压缩历史 + 位置部分');
  w('write',['mla','b',.25],['layerkv','t',.25],'kv','写本轮',[],[238,376]);w('read',['layerkv','t',.75],['mla','b',.75],'kv','读上下文',[],[397,376]);
  w('residual1',['norm1','b'],['norm2','b'],'reference','residual',[[112,365],[516,365]],[465,388]);
  w('residual2',['norm2','b',.73],['layerout','b'],'reference','residual 随中间状态保留',[[553.72,493],[920,493]],[733,484]);
  n('norm',232,544,164,72,'最终 RMSNorm','仅最后一个 PP 阶段');n('logits',434,544,164,72,'lm_head / logits','隐藏状态 → 词表分数');n('sample',636,544,164,72,'后续采样','Worker 协调选择 ID');n('token',838,544,164,72,'下一 token','返回 Scheduler');
  w('to-norm',['layerout','r',.72],['norm','t'],'activation','末级完成全部层后',[[1022,316.8],[1022,525],[314,525]],[760,512]);
  w('to-logits',['norm','r'],['logits','l'],'activation','h');w('to-sample',['logits','r'],['sample','l'],'logits','分数');w('to-token',['sample','r'],['token','l'],'token','ID');
 }else if(view==='chain'||view==='graph'){
  g.height=610;
 }else{
  g.height=554;n('x',40,50,240,138,'输入 X · M × 4096','M = 本轮新位置总数');n('w',400,50,240,138,'同一份权重 W','4096 × 4096 · 2 bytes / 元素');n('y',760,50,240,138,'输出 Y · M × 4096','每个新位置得到一行结果');
  w('x-y',['x','b'],['y','b',.3],'activation','多行输入复用同一份 W',[[160,222],[832,222]],[478,214]);w('w-y',['w','r'],['y','l'],'activation','矩阵乘法');
  n('compute',40,296,440,92,'计算下界','FLOPs / 假设算力');n('memory',560,296,440,92,'访存下界','bytes / 假设带宽');n('bound',300,446,440,86,'较大的一项约束理想下界','不是完整模型或真实 GPU 的耗时预测');
 }
 return g;
}
function build(view='model',options={}){
 if(!['model','chain','graph','hardware'].includes(view))throw Error('Unknown view');
 const opts={mode:'prefill',origin:'prefill',count:3,condition:'ready',rows:1,upgrade:'base',...options},g=geometry(view),phases=[],events=[],milestones=[];
 const initial={embedded:false,residualReady:false,arrived:0,completed:-1,kv:opts.mode==='decode'?[4,4,4,4]:[0,0,0,0],logits:false,sampled:false,delivered:false,inputReady:false,metadata:false,stored:false,kernelDone:false,padded:false,raw:false,trimmed:false,calculated:false};let duration=0;
 function phase(id,title,d,source,explanation,stage=-1){const p={id,title,duration:d,start:duration,end:duration+d,source,explanation,stage,index:phases.length};phases.push(p);duration+=d;return p;}
 function e(p,edge,a,b,payload,kind,nodes){events.push({phase:p.index,edge,start:p.start+a*p.duration,end:p.start+b*p.duration,payload,kind:kind||g.edges[edge]?.kind||'reference',nodes:nodes||(edge?[g.edges[edge].from,g.edges[edge].to]:[])});}
 function mark(p,at,patch){milestones.push({time:p.start+p.duration*at,patch});}
 function local(p,node,payload,a=.06,b=.9,kind='activation'){e(p,null,a,b,payload,kind,[node]);}
 if(view==='model'){
  let p=phase('embedding','首级：token ID 变成 hidden states',3,'model','首个 PP 阶段执行 Embedding；非首级读取上级中间状态。本例采用普通 token 输入，省略 input_embeds 替代入口。',0);local(p,'stage0','input_ids → embed_tokens → hidden_states；residual 初始为空');mark(p,.9,{embedded:true});
  for(let s=0;s<4;s++){
   if(s===0){
    p=phase('attention','代表层：准备输入，读写缓存并计算 Attention',7,'layer','prepare_attn 整理 residual 与归一化输入，再交给 MLA。Prefill 参考因果范围内的上下文；Decode 在历史后加入新位置。此处展开一层的接口顺序，具体布局与后端调用在执行课继续。',s);
    local(p,'norm1','prepare_attn：整理 residual，准备归一化输入',.02,.20);mark(p,.20,{residualReady:true});e(p,'norm-attn',.21,.31,'归一化后的 hidden_states');e(p,'write',.33,.48,opts.mode==='decode'?'写入 y1 的压缩历史与位置部分':'写入 p1…p4 的本层历史表示');mark(p,.48,{cacheStage:s,cacheCount:opts.mode==='decode'?5:4});e(p,'read',.50,.67,'本层可读上下文；不读取其他层的缓存');local(p,'mla','查询与可读历史参与 Attention 计算',.69,.96);
    p=phase('feedforward','代表层：合并残差，前馈计算并交出层结果',6.4,'layer','prepare_mlp 合并 Attention 更新与主干，再由 mlp 按层配置执行 Dense 或 MoE；postprocess_layer 整理 hidden_states、residual。其余本级层沿相同接口继续，全部完成后才交接。',s);
    e(p,'attn-norm',.02,.19,'Attention 的输出');e(p,'residual1',.02,.19,'本层保留的 residual','reference');local(p,'norm2','残差合并 + RMSNorm',.20,.33);e(p,'norm-mlp',.34,.44,'归一化后的输入特征');local(p,'mlp','按层配置执行 Dense 或 MoE',.45,.65);e(p,'mlp-out',.66,.82,'mlp 产生的 hidden_states');e(p,'residual2',.66,.82,'保留 residual，与 hidden_states 一起交出','reference');local(p,'layerout','整理层输出，继续完成本级其余层',.83,.97);mark(p,.97,{completed:s});
   }else{
    p=phase('local-'+s,'Stage '+s+'：执行本级层区间',2.6,'model','接收上一级的 hidden_states、residual，执行属于本级的各层；层内沿用下方接口，但各层的计算与 MLA 历史独立。这里只概括重复结构，未省去实际计算；教学时长不代表本级工作更少。',s);
    local(p,'stage'+s,'本级各层继续计算，使用各自的 MLA 缓存',.02,.97);mark(p,.55,{cacheStage:s,cacheCount:opts.mode==='decode'?5:4});mark(p,.97,{completed:s});
   }
   if(s<3){p=phase('pp-'+s,'PP 交接：Stage '+s+' → '+(s+1),s===0?3:1.8,'modelReturn',s===0?'非末级返回 PPProxyTensors；运行时沿 PP 通道交接 hidden_states、residual。各层 MLA 历史留在对应阶段，不随激活迁移。':'本级层区间已完成，把更新后的 hidden_states、residual 交给下一级；本级缓存继续由本级保存。',s);e(p,'pp'+s,.03,.94,s===0?'PPProxyTensors：hidden_states + residual（不是历史 KV）':'Stage '+s+' → '+(s+1)+'：更新后的 hidden_states + residual');mark(p,.94,{arrived:s+1});}
  }
  p=phase('norm','末级：最终归一化',2.6,'finalNorm','只有最后一个 PP 阶段执行模型末尾的 RMSNorm；必要时与携带的 residual 一起归一化。',3);e(p,'to-norm',.02,.58,'最后层的 hidden_states 与 residual');local(p,'norm','最终 RMSNorm',.60,.94);
  p=phase('logits','末级：从隐藏状态得到词表分数',3.2,'logits','logits_processor 使用 lm_head 计算词表分数。Prefill 的普通续写按所需输出位置取结果；这些分数还不是已经选中的 token。',3);e(p,'to-logits',.03,.40,'最终 hidden_states');local(p,'logits','lm_head / logits_processor → next_token_logits',.42,.94,'logits');mark(p,.94,{logits:true});
  p=phase('sample','模型之外：采样并交回下一 token',3.6,'sample','Worker 协调 ModelRunner.sample。图中只标记下一输出的身份，不编造模型生成的文本或词表分数。新采样 token 尚未作为下一轮输入，因此没有它的 KV。',3);e(p,'to-sample',.02,.34,'next_token_logits');mark(p,.63,{sampled:true});local(p,'sample','从候选分数中选出一个 token ID',.35,.63,'token');e(p,'to-token',.64,.94,opts.mode==='decode'?'下一输出 y2；尚无 y2 的 KV':'首个输出 y1；尚无 y1 的 KV');mark(p,.94,{delivered:true});
 }else if(view==='chain'){
  const prefill=opts.origin==='prefill';
  let p=phase('pack','CPU：选定请求，整理本轮输入',3,'scheduler','本例只跟随一个批次：R1、R2。Prefill 展示完整新输入；Decode 展示两个请求各继续一个位置。地址已经由调度侧准备，图中不模拟分配策略。');p.summary=prefill?'R1 的 3 个 token 与 R2 的 2 个 token 按请求顺序打包':'R1、R2 各继续一个位置；输入来自上一轮采样';
  p=phase('resolve',prefill?'CPU → GPU：把新输入拷入显存':'GPU 内：沿用上一轮输出作为本轮输入',4,'resolve','resolve_forward_inputs 将 Prefill 的 pinned CPU 输入异步拷入设备；Decode 可从显存 FutureMap 提取上一轮输出。图中演示后者，不把所有 Decode 都画成 H2D。');p.summary=prefill?'复制的是 token ID 数组；请求对象和源码函数不会被搬进 GPU':'Decode 的 token 已在显存；只在 GPU 内整理，不绕回 CPU 再上传';mark(p,.94,{inputReady:true});
  p=phase('bind','Worker：组织字段与张量引用',2.6,'input','ForwardBatch.init_new 引用 input_ids、seq_lens、out_cache_loc 等既有张量，并准备 positions 等字段。图中的虚线表示引用，不代表再次拷贝这些字节。');p.summary='ForwardBatch 留在 CPU 侧组织调用；字段引用显存张量';mark(p,.94,{bound:true});
  p=phase('load','Runner：更新本轮执行缓冲区',3,'eagerLoad','此固定版本的 EagerRunner 默认也使用静态缓冲区；load_batch 通过 registry.fill_from 更新它们。这里是显存内的数据更新，不是把 Python 对象上传。SGLANG_EAGER_INPUT_NO_COPY 分支不在本例展开。');p.summary='GPU 当前输入 → GPU 执行缓冲区；数据内容更新，缓冲区位置保留';mark(p,.94,{loaded:true});
  p=phase('metadata','Backend：建立本层历史的读取索引',3.5,prefill?'eagerExtend':'eager','运行模型前准备 Attention 元数据。槽号是显存缓存地址索引；逻辑位置是序列中的位置；token ID 是词表编号，三者不能互换。Prefill 按因果范围读取；Decode 读取已有历史与本轮位置。');p.summary='out_cache_loc 决定写哪里；历史索引决定读哪些槽；格子的物理位置保持不动';mark(p,.94,{metadata:true});
  p=phase('launch','CPU 提交计算，GPU 执行算子',2.5,prefill?'eagerExtend':'eager','模型 forward 的 Python 调用组织张量运算并发起设备工作。虚线标记提交关系，与实线数据拷贝不同；图中只画一个代表性提交，不模拟真实 kernel 数量或流并发。');p.summary='提交的是计算任务及张量地址；权重与历史 KV 已在显存';mark(p,.94,{launched:true});
  p=phase('project','GPU：读取输入与权重，生成层内特征',3.4,'model','GPU 算子从设备缓冲区读取本轮输入，使用常驻权重执行 Embedding 与层内投影，产生查询和本轮 MLA 历史表示。这里只画数据依赖，不模拟具体矩阵数值。');p.summary='显存中的 input_ids + 常驻权重 → 隐藏特征 → 查询与本轮 KV';mark(p,.94,{projected:true});
  p=phase('compute','GPU：写入本轮 KV，再读取可见上下文',6,prefill?'attention':'kernelCall','放大一层 Attention 的数据依赖。先将本轮历史表示写入预留槽，再按当前请求的索引访问可见历史；Prefill 仍遵守因果范围。MLA 缓存画成格子仅表示位置，不代表完整逐头 K/V。');p.summary='本轮 KV 落入自己的槽位；查询只读取同一请求、同一层的可见历史';mark(p,.38,{stored:true});mark(p,.94,{kernelDone:true});
  p=phase('result','GPU：完成模型并采样出下一 ID',4.8,'sample','Attention 结果仍是中间张量；其余层与输出头完成后产生 logits，再由 Worker 协调采样。这里只呈现结果类型，不编造真实模型分数或生成文本。');p.summary='中间特征 → 每请求的词表分数 → 每请求一个下一 token ID';mark(p,.42,{logits:true});mark(p,.94,{sampled:true});
  p=phase('return','GPU → CPU：返回处理请求所需的结果',2.4,'returnCPU','GenerationBatchResult.copy_to_cpu 按需拷回结果。图中只追踪下一 token ID；不返回整份权重或历史 KV。Scheduler 消费结果，再安排下一轮。');p.summary='返回下一 token ID，更新 R1 / R2；权重和 KV 继续留在 GPU';mark(p,.94,{delivered:true});
 }else if(view==='graph'){
  const d=M.graph(Number(opts.count),opts.condition,4);initial.eligible=d.eligible;
  let p=phase('gate','先核对真实请求与可用图桶',3.2,'graphGate','图桶是已捕获形状的容量，不代表新增请求。示例改变容量约束、是否启用与动态 embedding override；其余条件假设满足。');p.summary=d.reason;
  p=phase('load',d.eligible?'在显存内更新固定地址，必要时补齐':'保留真实输入，准备 Eager 执行',4,d.eligible?'graphLoad':'eagerLoad','当前输入已经在 GPU；图 runner 按图桶更新静态设备缓冲区。PAD 只占形状，不创建用户请求。本例 Eager 不做图桶补齐。');p.summary=d.eligible?'同一批真实输入 → 固定地址中的本轮数据；空余行标记为补齐':'不满足图条件仍在 GPU 上计算全部真实输入';mark(p,.75,{padded:true});
  p=phase('execute',d.eligible?'CPU 提交重放，GPU 执行已捕获的工作':'CPU 发起普通调用，GPU 执行模型',3.4,d.eligible?'replay':'runner','图重放复用已捕获的执行工作，不复用旧答案；输入、KV 与必要元数据每轮仍需更新。图内仅列代表计算，不表示真实 kernel 数量或耗时比例。');p.summary=d.eligible?'replay(选中的图桶) → 当前输入重新经过设备计算':'Eager 逐次发起普通算子调用；不是 CPU 回退';
  p=phase('raw','取得本轮 GPU 输出',2,d.eligible?'trim':'logits','先获得当前输入产生的输出，再按真实行数决定交付范围。图桶的占位行不能成为用户结果。');mark(p,.93,{raw:true,logits:true});
  p=phase('trim','只保留真实请求对应的分数行',3.4,'trim',d.eligible?'按 raw_num_token 裁回真实输出行。这里交给后续采样，尚未返回 CPU。':'保留全部真实输入对应的 logits，之后再采样。');mark(p,.93,{trimmed:true,delivered:true});
 }else{
  let p=phase('rows','看清 M：本轮有多少个新位置',2.5,null,'M 可以来自多个 Decode 请求，也可以来自一段 Prefill。图中的矩阵块是形状示意，不按完整维度逐个画元素。');local(p,'x','X 有 M 行，每行对应一个新位置');local(p,'w','所有输入行使用同一份权重 W',.05,.94);
  p=phase('multiply','每一行都与同一份权重相乘',3,null,'Y 的行数与 X 一致。增加新位置会增加乘加工作，但不会扩大这份权重矩阵。');e(p,'x-y',.04,.78,'输入行参与线性变换');e(p,'w-y',.04,.78,'复用相同权重');mark(p,.94,{calculated:true});
  p=phase('cost','分别估算计算和访存下界',4.2,null,'教学假设：K=N=4096，每元素 2 bytes；矩阵元素理想读写一次。算力 8 TFLOP/s、带宽 100 GB/s 均非实测设备参数。两条柱使用相同的毫秒刻度，虚线框保留原配值。');local(p,'compute','FLOPs = 2MKN；计算项 = FLOPs / 算力');local(p,'memory','bytes = 2(MK + KN + MN)；访存项 = bytes / 带宽');
  p=phase('bound','比较哪一个约束更强',2.8,null,'理想下界取两项的最大值。它不表示一定完全重叠，也不包含启动、缓存重读、Attention 历史 KV、通信等成本，不能据此预测整个模型。');local(p,'bound','max(计算下界, 访存下界)');mark(p,.94,{delivered:true});
 }
 // Keep the payload caption stable while several short sub-actions run.
 const summaries={model:{attention:'hidden_states → 查询与本层 MLA 缓存；读取可见历史，产生 Attention 输出',feedforward:'Attention 输出 + residual → 前馈计算 → hidden_states + residual',sample:'next_token_logits → 选出下一 token ID；新输出尚无自己的 KV'},chain:{forward:'本轮 token ID、位置、长度、KV 写地址 → ForwardBatch',metadata:'请求行与缓存地址 → 历史索引和分段信息',kernel:'Q + 缓存视图 + 分段索引 → Attention 输出',result:'最终 hidden_states → next_token_logits → next_token_ids / GenerationBatchResult'},graph:{execute:'当前输入与 KV 已就绪 → 执行模型 → 等待本轮输出'},hardware:{}};
 for(const p of phases)p.summary=p.summary||summaries[view][p.id];
 milestones.sort((a,b)=>a.time-b.time);return {view,options:opts,geometry:g,phases,events,milestones,initial,duration};
}
function at(timeline,time){const t=Math.max(0,Math.min(timeline.duration,Number(time)||0)),p=timeline.phases.find(p=>t<p.end)||timeline.phases.at(-1),state={...timeline.initial,kv:[...timeline.initial.kv]};for(const m of timeline.milestones){if(m.time>t)break;const {cacheStage,cacheCount,...patch}=m.patch;Object.assign(state,patch);if(cacheStage!==undefined)state.kv[cacheStage]=cacheCount;}return {time:t,phase:p,u:(t-p.start)/p.duration,state,ended:t===timeline.duration,active:timeline.events.filter(e=>t>=e.start&&t<e.end).map(e=>({...e,progress:(t-e.start)/(e.end-e.start)}))};}
const api={revision:M.revision,sources,geometry,build,at,sourceURL:key=>{const [file,line]=sources[key];return `https://github.com/sgl-project/sglang/blob/${M.revision}/${file}#L${line}`;}};if(typeof module==='object'&&module.exports)module.exports=api;else root.ModelExecutionFlow=api;
})(typeof globalThis==='object'?globalThis:this);
