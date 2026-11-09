/* Source-pinned execution examples. Units are explanatory time, not measurements. */
(function(root){
'use strict';
const revision='279339f113b79af84f27fd3ac92d0a13bd3f4cbd';
const sources={
 worker:['speculative/eagle_worker_v2.py',1272,'forward_batch_generation'],draft:['speculative/eagle_worker_v2.py',596,'draft'],
 build:['speculative/eagle_worker_common.py',316,'build_eagle_verify_input'],verify:['speculative/eagle_worker_common.py',461,'run_eagle_verify'],
 compact:['speculative/eagle_worker_common.py',406,'_finalize_accept_tree_path'],sample:['speculative/eagle_utils.py',724,'eagle_sample'],
 extend:['speculative/eagle_worker_v2.py',1013,'_draft_extend_for_decode'],allocation:['mem_cache/allocation_sizing.py',62,'page_aligned_decode_alloc_lens'],
 result:['managers/scheduler_components/batch_result_processor.py',744,'_resolve_spec_v2_tokens'],output:['managers/scheduler_components/batch_result_processor.py',920,'process_batch_result_decode'],
 barrier:['managers/scheduler.py',2054,'_advance_pending_grammar'],advance:['managers/scheduler_components/batch_result_processor.py',851,'advance_grammar_fsm'],
 retain:['managers/scheduler_components/batch_result_processor.py',820,'_accept_grammar_tokens'],
 compile:['constrained/base_grammar_backend.py',283,'get_cached_or_future_value'],manager:['constrained/grammar_manager.py',145,'process_req_with_grammar'],ready:['constrained/grammar_manager.py',203,'get_ready_grammar_requests'],
 copy:['constrained/xgrammar_backend.py',144,'copy'],mask:['constrained/xgrammar_backend.py',118,'fill_vocab_mask'],apply:['constrained/xgrammar_backend.py',125,'apply_vocab_mask'],accept:['constrained/xgrammar_backend.py',92,'accept_token'],
 samplingInfo:['sampling/sampling_batch_info.py',263,'update_regex_vocab_mask'],runner:['model_executor/model_runner.py',1882,'_preprocess_logits'],
 tree:['speculative/spec_utils.py',473,'traverse_tree'],treeCopy:['speculative/spec_utils.py',591,'GrammarTree','class'],treeMask:['speculative/spec_utils.py',649,'build_grammar_vocab_mask'],
 publish:['managers/overlap_utils.py',571,'publish'],loop:['managers/scheduler.py',1942,'event_loop_overlap'],copyResult:['managers/utils.py',133,'copy_to_cpu']
};
const url=k=>{const [file,line]=sources[k];return `https://github.com/sgl-project/sglang/blob/${revision}/python/sglang/srt/${file}#L${line}`;};
const modes={
 speculative:{label:'EAGLE · 验证与提交',question:'GPU 已选出接受路径，Req 的输出和 KV 账本何时才更新？',cases:{partial:'单链：接受 2 个',reject:'单链：首个分歧',all:'单链：全部接受',tree:'树：接受非连续节点'},scope:'CUDA · 单层 EAGLEWorkerV2 · PP=TP=1 · 普通 Attention 目标 · greedy · page_size=1 · Overlap 开；plan stream / grammar 关。',rule:'图中分开记录设备长度、CPU Req 账本和物理预留。下一轮设备工作可以先于当前轮 CPU 输出收尾；同一请求的 Draft → Verify → Draft Extend 仍有依赖。'},
 grammar:{label:'Grammar · 准入与采样',question:'同一份规则如何复用，而两个请求的匹配进度互不干扰？',cases:{cold:'首次编译规则',hit:'命中已编译规则',no:'命中：模型选择 no',invalid:'规则编译失败'},scope:'CUDA · XGrammar · PP=TP=1 · 普通非投机 decode · Overlap 关。小词表 y / es / no / ! / EOS 为教学设定。',rule:'线程池编译可以与其他已就绪请求的 GPU 工作并行。编译上下文 ctx 可复用，每个请求持有自己的 matcher；图中只展开 R1 的一次采样。'},
 combined:{label:'EAGLE + Grammar · 协同',question:'怎样遍历候选树却不污染真实 matcher，并让规则处理与目标计算重叠？',cases:{normal:'上一轮结果已可读',late:'上一轮 copy_done 较晚'},scope:'CUDA · 单层 EAGLEWorkerV2 + XGrammar · PP=TP=1 · topk=2 · 普通 Attention · Overlap 开、plan stream 关。只跟踪请求 R1。',rule:'当前 target forward 与 CPU barrier / DFS 可以重叠；mask 依赖上一轮 FSM 已推进、树元数据已到 CPU。采样必须等 logits 和本轮 mask 都可用。'}
};
function create(mode,condition){
 if(!modes[mode])mode='speculative';if(!modes[mode].cases[condition])condition=Object.keys(modes[mode].cases)[0];
 const events=[],lanes=[];
 const lane=(id,name)=>lanes.push({id,name});
 const event=(id,lane,start,end,title,detail,source,deps=[],kind='cpu',extra={})=>{const e={id,lane,start,end,title,detail,source,deps,kind,...extra};events.push(e);return end;};
 let data;
 if(mode==='speculative'){
  lane('cpu','Scheduler · CPU');lane('gpu','GPU · forward stream');lane('copy','GPU → CPU · copy stream');
  const tree=condition==='tree',n=condition==='reject'?0:condition==='all'?4:2,path=tree?[0,1,3]:Array.from({length:n+1},(_,i)=>i),tokens=tree?['r','a','x','b','y']:['r','a','b','c','d'];
  const end=tree?11.4:10.6,publish=end+.4,extend=publish+4,copy=extend+1.4,commit=copy+1.4;
  event('reserve','cpu',0,1.2,'保留请求映射与预留空间','eagle_prepare_for_decode 按已提交长度与 reserve 补足空间；已分配长度不会因本轮拒绝而自动缩短。图中只截取本轮 5 个验证槽。','allocation');
  event('draft','gpu',1.2,4.2,'Draft：提出候选','EagleDraftWorker.draft 消费上一轮 bonus、hidden / topk，执行草稿步骤。候选尚未写入 Req.output_ids。','draft',['reserve'],'compute');
  event('build','gpu',4.2,5.2,'组织 EagleVerifyInput','build_eagle_verify_input 生成 draft_token、positions、tree mask 和 retrieve 索引。root r 已在上一轮输出，但其目标 KV 尚未写入。','build',['draft'],'data');
  event('verify','gpu',5.2,8.8,'Target Verify：写候选位置 KV','目标 worker 处理全部验证输入，得到每个节点的 logits 与 hidden states，并写入对应 out_cache_loc。计算过的节点不等于最终有效路径。','verify',['build'],'compute');
  event('select','gpu',8.8,10,'eagle_sample：确定路径与 bonus',`本例 accept_index=${JSON.stringify(path)}；accept_lens=${n+1}（接受 ${n} 个草稿 + 1 个 bonus）。new_seq_lens=8+${n+1}。这是 GPU 结果，CPU Req 尚未结算。`,'sample',['verify'],'compute');
  event('compact','gpu',10,end,tree?'非连续路径移到连续前部':'单链：接受路径已连续',tree?'_finalize_accept_tree_path 按 accept_index 移动目标 KV、predict 与 hidden_states，节点 3 的 b 来到连续槽 2。只展示有效前部，尾部仍属于预留窗口。':'topk=1 不调用树路径压紧；有效目标 KV 是 root + 接受的草稿，bonus z 尚无自身目标 KV。','compact',['select'],'data');
  event('publish','gpu',end,publish,'写入长度 relay，标记 publish_ready','主机 on_publish 回调在 Verify 后、Draft Extend 前安排 new_seq_lens_buf 写入并记录 publish_ready；轨道画的是设备写入与就绪点，不是 GPU 执行 Python。长度可用不等于全部后继草稿输入已刷新。','publish',['compact'],'data');
  event('extend','gpu',publish,extend,'Draft Extend：刷新下一轮种子','使用目标 hidden_states、predict 和 accept_lens 刷新草稿状态，选出接受边界的 topk / hidden；bonus z 成为下一轮根。','extend',['publish'],'compute');
  event('copy','copy',extend,copy,'异步复制本轮结果到 CPU','Scheduler 在 copy_stream 等待本轮 forward，再复制 next_token_ids / accept_lens 等，并记录 copy_done。copy stream 不串行阻塞下一轮 forward stream。','copyResult',['extend'],'data');
  event('next','gpu',extend,extend+4,'下一轮 Draft','示例请求继续生成。新的草稿依赖前轮 Draft Extend / relay；设备可在当前轮 CPU 输出结算前开始。这里只展开下一轮 Draft，不假装下一轮已完成。','loop',['extend'],'compute');
  event('wait','cpu',extend+.4,copy,'CPU 等本轮 copy_done','CPU 结果处理器等待结果副本可读；GPU 下一轮 Draft 可以继续。','output',[],'wait',{waitFor:'copy'});
  event('commit','cpu',copy,commit,'结算 Req 并输出','_resolve_spec_v2_tokens 按每请求 stride / accept_lens 取有效 token，推进 kv_committed_len；process_batch_result_decode 追加 output_ids 并检查结束。本例不触发 EOS。','result',['copy']);
  data={tokens,path,n,output:[...path.slice(1).map(i=>tokens[i]),'z'],prefix:8,commitEnd:commit};
 }else if(mode==='grammar'){
  lane('cpu','Scheduler / matcher · CPU');lane('compile','GrammarCompiler · 线程池');lane('gpu','GPU · forward / mask / sample');
  const cold=condition==='cold'||condition==='invalid',bad=condition==='invalid',chosen=condition==='no'?'no':'y',ready=cold?6.6:2;
  event('lookup','cpu',0,1.2,'GrammarManager：以规则为 key 查缓存','key 由规则类型和规则文本组成；这里只演示同一个规则 yes|no。进入 waiting_queue 前，必须取得可用的 grammar 对象。','manager');
  if(cold){
   event('submit','cpu',1.2,1.8,'Future 入 grammar_queue','缓存未命中，executor.submit 异步编译；req.grammar 暂为 Future。请求 R1 等待规则，不是已经进入 GPU 采样。','compile',['lookup']);
   event('compile','compile',1.8,5.5,bad?'编译返回无效规则':'编译规则上下文','线程池执行规则编译；不占用 Scheduler 主线程，也不阻止其他已准入请求计算。时长为示例。','compile',['submit'],'compute');
   event('other','gpu',0,5,'其他请求 B0 的前向','B0 已有可用规则 / 无需规则。该工作与等待编译的 R1 是不同请求，不依赖 R1 的 Future。','loop',[],'compute');
   event('poll','cpu',2.1,2.8,'短暂轮询 Future：尚未完成','R1 留在 grammar_queue，Scheduler 可继续其他工作；未绘出的空白不表示持续忙等。','ready',['submit']);
   event('ready','cpu',5.5,ready,bad?'失败请求走 abort':'取结果并缓存可复用副本',bad?'InvalidGrammarObject 触发 set_finish_with_abort；请求进入既有中止处理，不会正常进入本图的 mask / GPU 路径。':'取 Future.result() 给 R1；缓存 req.grammar.copy()，复制创建新 matcher，共享已编译 ctx。','ready',['compile']);
  }else event('ready','cpu',1.2,ready,'缓存命中：copy 创建 R1 matcher','BaseGrammarBackend 调用 value.copy()。XGrammarGrammar.copy 共享 ctx，构造一个从起点开始的 GrammarMatcher，不复制别人的已接受 token 进度。','copy',['lookup']);
  if(!bad){
   event('r2','cpu',ready,ready+1,'R2 复用 ctx，获得独立 matcher','R2 后到并命中同一份规则。两个对象共享编译上下文，accepted_tokens 与 matcher 各自从起点开始。','copy',['ready']);
   event('forward','gpu',ready+1,ready+3.8,'R1 模型前向得到 logits','本例普通非投机、关闭 Overlap。这里只展开 R1 一轮；R2 的采样未展开，不表示框架要求 R2 等 R1 结束。','runner',['r2'],'compute');
   event('mask','cpu',ready+1.3,ready+2.8,'SamplingBatchInfo：生成本请求 bitmask','前向的主机调用返回后，CPU 可在 GPU 仍计算时生成 mask；关闭 Scheduler Overlap 不等于 CUDA 调用都变同步。update_regex_vocab_mask 调用 matcher 填行，初始合法项为 y / no。','samplingInfo',['r2']);
   event('h2d','gpu',ready+3.8,ready+4.4,'mask 非阻塞搬到 GPU','CPU 已生成 mask；同一设备流按前向 → 搬运 → 应用的顺序执行。XGrammarGrammar.move_vocab_mask(..., non_blocking=True) 搬的是词表位掩码，不是 matcher 或整个规则。','mask',['mask','forward'],'data');
   event('apply','gpu',ready+4.4,ready+5.5,'应用 mask，再执行采样','ModelRunner._preprocess_logits 经 SamplingBatchInfo 应用 mask，随后 Sampler 选 token。分数最高但不合法的 ! 被屏蔽。无额外 logit_bias / penalty。','runner',['h2d'],'compute');
   event('sample','gpu',ready+5.5,ready+6.2,`Sampler 选中 ${chosen}`,'模型只在本轮合法项中选择；这里用人工 logits 与 greedy，token 文字不代表真实 tokenizer 的切分。','apply',['apply'],'compute');
   event('d2h','gpu',ready+6.2,ready+6.9,'结果回到 CPU','普通非投机结果回传后，CPU 才能追加输出并推进该请求的 grammar。','output',['sample'],'data');
   event('advance','cpu',ready+6.9,ready+8.1,'只推进 R1 的 matcher',`R1 接受 ${chosen}，output_ids 和 matcher 前进；R2 仍在起点。下一轮 R1 允许 ${chosen==='y'?'es':'EOS'}，R2 仍允许 y / no。`,'accept',['d2h']);
  }
  data={cold,bad,chosen,ready,vocab:['y','es','no','!','EOS'],scores:[4,6,chosen==='no'?7:3,9,2],legal:[true,false,true,false,false]};
 }else{
  lane('cpu','Scheduler / grammar · CPU');lane('gpu','GPU · forward stream');lane('copy','GPU → CPU · copy stream');
  const prev=condition==='late'?7:2.6;
  event('draft','gpu',0,2,'本轮候选树就绪','EagleVerifyInput 包含 root r 与四个候选；图中节点编号是 verify 行索引，不是 token ID。root 是上一轮已经选出的 {。','build',[],'compute');
  event('treeCopy','gpu',2,3,'树元数据 D2H','GrammarTree.from_device 在目标 forward 提交前，非阻塞复制 child / sibling / draft_token，记录本流完成事件。只复制索引与 token，不复制模型 KV。','treeCopy',['draft'],'data');
  event('target','gpu',3,11,'本轮 Target Verify','设备计算当前候选树 logits / KV；同一时段 CPU 可以处理上一轮 grammar 与当前树的 mask。不能把目标模型计算画成等待 DFS 才开始。','verify',['treeCopy'],'compute');
  event('previous','copy',0,prev,'上一轮结果 D2H','这里是前一轮结果（root {）。其 copy_done 可能比当前 target forward 提交晚；两条 CUDA 流不等于两块 GPU。','copyResult',[],'data');
  let cursor=3.2;
  if(prev>cursor){event('wait','cpu',cursor,prev,'barrier 等上一轮 copy_done','_advance_pending_grammar → advance_grammar_fsm 同步上一轮事件；当前 target forward 已在运行。','advance',[],'wait',{waitFor:'previous'});cursor=prev;}
  cursor=event('barrier','cpu',cursor,cursor+.7,'先补齐上一轮真实 FSM','上一轮输出 root { 进入 R1 matcher，result.grammar_advanced 标为 true。此后 DFS 的起点才与已提交前缀一致。','barrier',['previous']);
  const actions=[
   ['root',.7,'root 只读规则，填第 0 行','root 已经由上一轮提交；DFS 不再 accept root。第 0 行允许 "ok" / "err"。',0,'{',[0]],
   ['enter1',.7,'临时 accept 节点 1','沿 "ok" 分支推进 matcher，为节点 1 填下一 token mask：只允许 :。',1,'{"ok"',[0,1]],
   ['enter3',.7,'临时 accept 节点 3','节点 3 是 :；填第 3 行，允许 true。每一行 mask 对应目标模型同一 verify 行的后继预测。',3,'{"ok":',[0,1,3]],
   ['back3',.5,'rollback(1)：退出节点 3','撤销这次试走的 :，回到节点 1 的规则状态；保留已经写好的第 3 行 mask。',1,'{"ok"',[0,1,3]],
   ['back1',.5,'rollback(1)：回到分叉点','撤销试走的 "ok"，回到 {，再进入兄弟分支。不是从第一条分支末尾继续判断兄弟。',0,'{',[0,1,3]],
   ['enter2',.7,'临时 accept 节点 2','从相同父状态试走 "err"，第 2 行同样只允许 :。',2,'{"err"',[0,1,2,3]],
   ['prune4',.7,'父 mask 拒绝节点 4','! 不在节点 2 的允许集合中，DFS 跳过该节点，不 accept、不生成它的行；该行保持初始化值，采样不会从合法路径到达它。',4,'{"err"',[0,1,2,3]],
   ['restore',.6,'rollback(1)：还原真实前缀','退出最后一条试走分支，matcher 恢复为 {。只有 mask 留下来；候选树遍历没有正式提交任何候选。',0,'{',[0,1,2,3]]
  ];
  let dependency='barrier';for(const [id,d,title,detail,node,prefix,rows]of actions){cursor=event(id,'cpu',cursor,cursor+d,title,detail,'tree',[dependency,'treeCopy'],'cpu',{node,prefix,rows});dependency=id;}
  const maskStart=Math.max(cursor,11);
  if(cursor>11)event('maskWait','gpu',11,cursor,'设备空闲：mask 尚未提交','目标 logits 已算完，但 CPU 仍在生成本轮 mask。这里是缺少后续工作提交，不是虚构一个等待 kernel。','treeMask',['target'],'wait',{waitFor:'restore'});
  event('h2d','gpu',maskStart,maskStart+.6,'本轮 mask H2D','build_grammar_vocab_mask 把位掩码非阻塞送入当前流，并清掉 sampling_info 中旧 mask；流内顺序确保 apply 在搬运之后。','treeMask',['restore','target'],'data');
  event('sample','gpu',maskStart+.6,maskStart+1.7,'应用树 mask，选出接受路径','eagle_sample 先 mask logits，再验证。教学结果 accept_index=[0,1,3]、accept_lens=3，新增输出为 "ok"、:、true。','sample',['h2d'],'compute');
  event('compact','gpu',maskStart+1.7,maskStart+2.5,'整理路径并发布长度','非连续索引 0、1、3 的 KV / predict / hidden 移到连续前部；matcher 仍为 {，尚未接受本轮输出。','compact',['sample'],'data');
  event('extend','gpu',maskStart+2.5,maskStart+4.1,'Draft Extend','刷新后继草稿状态。GPU 数据准备与 CPU matcher 提交是不同的状态变化。','extend',['compact'],'compute');
  event('copy','copy',maskStart+4.1,maskStart+4.8,'本轮结果 D2H','只有本轮结果 copy_done 后，CPU 才能消费本轮 token / accept_lens。','copyResult',['extend'],'data');
  event('advance','cpu',maskStart+4.8,maskStart+6,'正式推进本轮 FSM','advance_grammar_fsm 顺序 accept 保留的输出，记录 grammar_retained_tokens，设置 grammar_advanced。函数在后续结果处理再次调用时直接返回，避免重复推进。','advance',['copy']);
  event('commit','cpu',maskStart+6,maskStart+7,'CPU 提交保留输出与 KV 长度','_resolve_spec_v2_tokens 使用已缓存的保留前缀，随后追加 output_ids；若 grammar 提前结束，只提交保留前缀，不能把过量后缀输出。当前例未结束。','result',['advance']);
  data={prev,tokens:['{','"ok"','"err"',':','!'],children:[1,3,4,-1,-1],siblings:[-1,2,-1,-1,-1],path:[0,1,3],maskStart,output:['"ok"',':','true']};
 }
 const duration=Math.max(...events.map(e=>e.end))+.6,steps=[...new Set([0,...events.map(e=>e.start),duration])].sort((a,b)=>a-b);
 const sample=time=>{
  const t=Math.max(0,Math.min(duration,Number(time)||0)),done=id=>t>=events.find(e=>e.id===id)?.end,p=id=>{const e=events.find(e=>e.id===id);return e?Math.max(0,Math.min(1,(t-e.start)/(e.end-e.start))):0;};
  const active=events.filter(e=>e.start<=t&&t<e.end),candidates=active.filter(e=>e.kind!=='compute'||e.id==='verify'||e.id==='target');
  const focus=candidates.filter(e=>e.lane==='cpu').at(-1)||candidates.at(-1)||active.at(-1)||events.filter(e=>e.end<=t).sort((a,b)=>a.end-b.end).at(-1)||events[0];
  const state={t,done,p,active,focus};
  if(mode==='speculative')Object.assign(state,{cpuLen:8+(done('commit')?data.n+1:0),gpuLen:8+(done('select')?data.n+1:0),output:done('commit')?data.output:[],validKV:done('select')?data.path.map(i=>data.tokens[i]):[],acceptIndex:done('select')?data.path:[],allocated:5});
  if(mode==='grammar')Object.assign(state,{queue:!done('lookup')?'待检查':data.cold&&!done('ready')?'grammar_queue':data.bad?'abort':t>=data.ready+1?'已准入 / 运行中':'waiting_queue',r1:done('advance')?data.chosen:'',r2:'',shared:done('r2'),chosen:done('sample')?data.chosen:null,masked:done('apply')});
  if(mode==='combined'){
   const visit=events.filter(e=>e.prefix&&done(e.id)).at(-1);
   Object.assign(state,{matcher:done('advance')?'{"ok":true':visit?.prefix||(done('barrier')?'{':''),rows:visit?.rows||[],node:active.find(e=>e.node!==undefined)?.node??visit?.node??-1,output:done('commit')?data.output:[],advanced:done('advance'),restored:done('restore')});
  }
  return state;
 };
 return {mode,condition,...modes[mode],events,lanes,data,duration,steps,sample};
}
const api={revision,sources,url,modes,build:create};if(typeof module!=='undefined')module.exports=api;else root.GenerationExecution=api;
})(typeof globalThis!=='undefined'?globalThis:this);
