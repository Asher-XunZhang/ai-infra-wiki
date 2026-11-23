/* A static responsibility map, sharing selection with the existing request
 * diagram. Groups represent objects and responsibilities, not processes. */
window.AtlasRoles = (() => {
  'use strict';
  const h=React.createElement;
  const SG='https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/';
  const VL='https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/';
  const role=(id,name,summary,detail,path,node,primary=false)=>({id,name,summary,detail,path,node,primary});
  const common=side=>[
    role('model','Model',['连接 Attention、MLP、Norm 等子层，定义模型的前向计算。','Connects attention, MLP and normalization layers into the model’s forward computation.'],['Model 是加载后的模型对象。ModelRunner 的执行路径调用它；模型内各层再调用对应的算子实现。','Model is the loaded model object. ModelRunner’s execution path calls it, and its layers invoke their operator implementations.'],side==='sg'?SG+'model_executor/runner/eager_runner.py#L268':VL+'vllm/v1/worker/gpu/model_runner.py#L2053','workers'),
    role('attention','Attention',['利用当前输入和历史 KV，计算注意力输出。','Uses the current input and historical KV to compute attention output.'],['Attention 是模型中的计算子层；执行时使用本轮 attention metadata 与 KV 地址，具体计算交给所选后端。','Attention is a model sublayer. It uses this step’s attention metadata and KV addresses, with computation handled by the selected backend.'],side==='sg'?SG+'layers/radix_attention.py#L114':VL+'vllm/model_executor/layers/attention/attention.py','workers'),
    role('kernel','Kernel',['在设备上执行矩阵乘、归约等算子，支撑各个子层。','Executes operators such as matrix multiplication and reductions to support model sublayers.'],['Kernel 支撑 Attention、MLP、归一化等子层。一次前向通常调用多个算子；这里的源码以可选 Attention 后端如何调用算子为例。','Kernels support attention, MLP and normalization. A forward pass usually invokes many operators; this source illustrates operator calls from one optional attention backend.'],side==='sg'?SG+'layers/attention/flashinfer_backend.py':VL+'vllm/v1/attention/backends/flash_attn.py#L1222','workers'),
  ];
  const data={
    sg:[
      role('input','HTTP / TokenizerManager',['整理文本和采样参数，维护请求与流式响应。','Prepares text and sampling settings, and tracks requests and streaming responses.'],['TokenizerManager 将文本编码成 token IDs，交给 Scheduler；回程接收文本片段，并归到对应请求。','TokenizerManager encodes text into token IDs for Scheduler, then associates returning text fragments with their requests.'],SG+'managers/tokenizer_manager.py','tokenizer',true),
      role('scheduler','Scheduler',['结合 token 预算和 KV 空间，从队列中选出本轮执行的请求。','Selects this step’s requests from queues using token budget and KV capacity.'],['组织 ScheduleBatch，调用本 rank 的 TpModelWorker，再消费执行结果、更新请求状态。TpModelWorker 与 Scheduler 位于同一进程。','Builds ScheduleBatch, calls the rank-local TpModelWorker, then consumes results and updates requests. TpModelWorker shares its process.'],SG+'managers/scheduler.py#L4614','scheduler',true),
      role('worker','TpModelWorker',['接收本 rank 的批次，调用 ModelRunner 并整理返回结果。','Receives the rank-local batch, calls ModelRunner and packages returned results.'],['TpModelWorker 持有 ModelRunner，将调度批次转换为执行输入，调用前向与采样，再整理执行结果。这里没有额外引入独立 Executor。','TpModelWorker owns ModelRunner, converts batch inputs, invokes forward and sampling, and packages the results. No separate Executor is added here.'],SG+'managers/tp_worker.py#L622','workers',true),
      role('runner','ModelRunner',['整理计算输入，组织模型前向、采样和图执行。','Prepares computation inputs and organizes model forward, sampling and graph execution.'],['根据 ForwardBatch 与后端条件准备计算，选择适用的图或普通执行路径；模型和 KV 访问落实在本 rank。','Uses ForwardBatch and backend conditions to prepare computation and select graph or ordinary execution. Model execution and KV access are rank-local.'],SG+'model_executor/model_runner.py#L1813','workers'),
      ...common('sg'),
      role('return','Detokenizer → TokenizerManager',['将 token IDs 增量还原成文本，再交回对应响应。','Incrementally converts token IDs to text and returns it to the matching response.'],['Scheduler 推出 token IDs，独立 Detokenizer 处理可显示的文本片段，再送回 TokenizerManager。进程与 socket 的位置在后两种视图中展开。','Scheduler emits token IDs; the separate Detokenizer builds displayable text and sends it to TokenizerManager. The other views expand process and socket boundaries.'],SG+'managers/detokenizer_manager.py','detok',true),
    ],
    vl:[
      role('input','API / AsyncLLM',['准备请求，并在回程处理输出、反分词和流式响应。','Prepares requests, then handles outputs, detokenization and streaming responses.'],['普通 Python 服务路径中，前端将输入交给 EngineCore；拿到 EngineCoreOutputs 后进行输出处理。','In ordinary Python serving, the frontend submits inputs to EngineCore and processes returning EngineCoreOutputs.'],VL+'vllm/v1/engine/async_llm.py#L879','async',true),
      role('core','EngineCore',['组织调度循环、模型执行和结果更新。','Organizes the scheduling loop, model execution and result updates.'],['EngineCore 调用 Scheduler 取得计划，再通过 Executor 执行并更新请求。异步模式可以交错多轮工作；本图只表达职责关系。','EngineCore gets a plan from Scheduler, executes it through Executor, and updates requests. Async modes can interleave steps; this map shows responsibilities.'],VL+'vllm/v1/engine/core.py#L665','engine',true),
      role('scheduler','Scheduler',['选择本轮请求与 token 工作量，并结合 KV 资源安排任务。','Chooses requests and token work for this step while accounting for KV resources.'],['维护 waiting / running，检查 token 预算与 KV 分配，产生 SchedulerOutput；本轮计算由执行侧落实。','Maintains waiting/running requests, checks token budgets and KV allocations, and produces SchedulerOutput for execution.'],VL+'vllm/v1/core/sched/scheduler.py','engine'),
      role('executor','Executor',['派发执行调用，协调 Worker 执行及结果返回。','Dispatches execution calls and coordinates workers and returned results.'],['接收 SchedulerOutput，通过所选 uni、mp 等实现派发 execute_model / sample_tokens。执行器决定派发方式，Worker 管理各自的执行环境。','Receives SchedulerOutput and dispatches execute_model/sample_tokens through the selected uni, mp or other implementation. Workers manage their own execution environments.'],VL+'vllm/v1/executor/abstract.py#L265','engine'),
      role('worker','Worker',['管理本 rank 的设备、分布式环境和执行生命周期。','Manages this rank’s device, distributed environment and execution lifecycle.'],['初始化设备和分布式环境，创建 ModelRunner，并将本 rank 的执行调用交给它。TP1 uni 与 TP2 mp 的进程放置见归属视图。','Initializes device/distributed state, creates ModelRunner, and delegates rank-local execution to it. The ownership view shows TP1 uni versus TP2 mp placement.'],VL+'vllm/v1/worker/gpu_worker.py#L449','workers',true),
      role('runner','ModelRunner V2',['整理输入和 KV 地址等执行信息，组织前向与采样。','Prepares inputs and execution information such as KV addresses, then organizes forward and sampling.'],['更新持久请求状态与 BlockTables，准备本批输入、attention metadata 和 KV 槽位；execute_model 与后续 sample_tokens 分别推进前向和采样。','Updates persistent request state/BlockTables and prepares batch inputs, attention metadata and KV slots. execute_model and sample_tokens advance forward and sampling separately.'],VL+'vllm/v1/worker/gpu/model_runner.py#L1764','workers'),
      ...common('vl'),
    ],
  };
  function Body({side,en,selected,onSelect,onLocate,showSources}){
    const tr=p=>p[en?1:0],items=data[side],find=id=>items.find(x=>x.id===id);
    const render=id=>{
      const item=find(id),chosen=selected?.side===side&&(selected.roleId?selected.roleId===id:item.primary&&selected.id===item.node);
      const panel=`role-${side}-${id}`;
      return h('div',{className:'ir-item'+(chosen?' is-selected':''),key:id,'data-module-role':id},
        h('button',{type:'button',className:'ir-role','aria-expanded':!!chosen,'aria-controls':chosen?panel:undefined,onClick:()=>onSelect({side,id:item.node,roleId:id,label:item.name,role:item.summary,description:item.detail,source:item.path})},
          h('strong',null,item.name),h('span',null,tr(item.summary))),
        chosen?h('div',{className:'ir-detail',id:panel},h('p',null,tr(item.detail)),
          h('div',{className:'ir-detail-links'},h('a',{href:item.path,target:'_blank',rel:'noreferrer'},tr(['固定源码 ↗','Pinned source ↗'])),h('button',{type:'button','data-locate-role':id,onClick:()=>onLocate()},tr(['在请求图定位','Locate in request diagram'])))):showSources?h('a',{className:'ir-source',href:item.path,target:'_blank',rel:'noreferrer'},tr(['固定源码 ↗','Pinned source ↗'])):null);
    };
    const group=(label,ids)=>h('section',{className:'ir-group','aria-label':label},render(ids[0]),h('div',{className:'ir-subgroup'},ids.slice(1).map(render)));
    const handoff=label=>h('p',{className:'ir-handoff'},h('span',{'aria-hidden':true},'↓'),label);
    return h('div',{className:'ir-map',id:'ia-module-roles-'+side},
      render('input'),handoff(tr(['提交内部请求','Submit the internal request'])),
      side==='sg'?h('section',{className:'ir-group','aria-label':tr(['调度与本 rank 执行','Scheduling and rank-local execution'])},render('scheduler'),h('div',{className:'ir-subgroup'},render('worker'),h('div',{className:'ir-subgroup'},render('runner')))):h(React.Fragment,null,
        group(tr(['EngineCore 组织本轮工作','EngineCore organizes each step']),['core','scheduler','executor']),handoff(tr(['经所选执行器调用 Worker','Invoke workers through the selected executor'])),group(tr(['每个 rank 的执行侧','Execution within each rank']),['worker','runner'])),
      handoff(tr(['调用模型与设备算子','Invoke the model and device operators'])),
      group(tr(['模型计算结构与实现','Model structure and operator implementation']),['model','attention','kernel']),
      side==='sg'?h(React.Fragment,null,handoff(tr(['采样 token IDs 回程','Return sampled token IDs'])),render('return')):h('p',{className:'ir-return'},tr(['结果回程：Core 更新请求 → AsyncLLM 输出处理与反分词 → 流式响应。','Return: Core updates requests → AsyncLLM output processing/detokenization → streamed response.'])));
  }
  return {Body};
})();
