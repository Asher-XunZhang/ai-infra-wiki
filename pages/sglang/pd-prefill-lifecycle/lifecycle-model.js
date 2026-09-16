/* A causal teaching sequence, not a simulation of wall time or PP loop indices. */
(function (root, factory) {
  const model = factory();
  if (typeof module === 'object' && module.exports) module.exports = model;
  else root.PrefillLifecycle = model;
})(typeof globalThis !== 'undefined' ? globalThis : this, () => {
  'use strict';
  const nodes = { frontend: { x:20,y:430,w:145,h:140, title:'HTTP / Tokenizer', role:'处理输入，保留 PD 配对信息，并接收 Prefill 服务的完成输出。' } };
  const roles = {
    bootstrap:'建立本地 Req 与 sender，等待握手并处理 bootstrap 共识。',
    schedule:'等待调度预算，匹配前缀，组建 EXTEND batch 并分配本地 KV 位置。',
    compute:'执行本级模型层；写本级 prompt KV，并产生下一段计算所需的激活或首 token。',
    transfer:'接收末级回流结果，维护本地 Req；最终块加入 inflight，各块按已完成范围提交本级 KV。',
    release:'汇合传输终态，接收 release 名单，复查本地状态后清理请求资源。',
  };
  for (let r=0;r<3;r++) {
    Object.keys(roles).forEach((kind,i) => {
      const geometry=[{x:250+r*370,y:180,w:235,h:76},{x:270+r*370,y:305,w:195,h:82},{x:250+r*370,y:430,w:260,h:155},{x:335+r*370,y:695,w:210,h:150},{x:235+r*370,y:730,w:76,h:76}][i];
      nodes[`p${r}-${kind}`] = { ...geometry,rank:r,kind,
        title:`PP${r} · ${['Bootstrap','Scheduler / Batch','Model forward','Result / KV sender','Release'][i]}`,role:roles[kind] };
    });
  }
  nodes['decode-bootstrap']={x:1400,y:180,w:245,h:76,title:'Decode · 接收准备',role:'准备目的内存与接收索引，完成与 Prefill 的配对；内部调度在本图边界之外。'};
  nodes['decode-kv']={x:1400,y:695,w:245,h:150,title:'Decode · KV 接收',role:'按层接收三个 Prefill 级各自持有的 prompt KV，以及最终块的 metadata。'};
  nodes['decode-run']={x:1400,y:930,w:245,h:68,title:'Decode · 后续生成',role:'使用收到的 prompt KV 与首 token 继续生成；不需要等 Prefill 的全部本地清理完成才开始。'};
  const events=[];
  const wire=(from,to,kind='control',route='direct')=>({from,to,kind,route});
  function add(id,phase,active,title,description,gate,options={}) {
    events.push({id,phase,active,title,description,gate,kind:'control',routes:[],updates:[],...options});
  }
  const update=(rank,values)=>({rank,...values});
  add('intake',0,['frontend'],'接入请求 R','TokenizerManager 将输入整理成 token IDs，携带 rid 与 bootstrap_room 等配对信息。','请求已送达 PP0 的调度入口。');
  add('req-0',1,['p0-bootstrap'],'PP0 建立本地请求','PP0 收到 tokenized request，创建 Req R 与 sender₀，放入 bootstrap_queue。','请求对象沿 PP 通道继续转发。',{routes:[wire('frontend','p0-bootstrap','control','entry')],updates:[update(0,{queue:'bootstrap',sender:'bootstrapping'})]});
  add('req-1',1,['p1-bootstrap'],'请求对象传到 PP1','PP1 从 PP0 接收请求信息，创建自己进程里的 Req R 和 sender₁；它与 PP0 使用相同 rid。','PP1 将请求信息转发给 PP2。',{routes:[wire('p0-bootstrap','p1-bootstrap')],updates:[update(1,{queue:'bootstrap',sender:'bootstrapping'})]});
  add('req-2',1,['p2-bootstrap'],'请求对象传到 PP2','PP2 建立第三份本地 Req 与 sender₂。三个级都要握手，各自拥有自己的资源。','等待接收端的目的索引等信息。',{routes:[wire('p1-bootstrap','p2-bootstrap')],updates:[update(2,{queue:'bootstrap',sender:'bootstrapping'})]});
  add('destination',2,['decode-bootstrap','p0-bootstrap','p1-bootstrap','p2-bootstrap'],'Decode 接收准备就绪','这里将接收端准备和握手信息交换合并为一步：三个 sender 具备 WaitingForInput 条件，但 R 尚未进入 waiting_queue。','各级轮询并形成共同准入名单。',{routes:[0,1,2].map(r=>wire('decode-bootstrap',`p${r}-bootstrap`,'control','handshake')),updates:[0,1,2].map(r=>update(r,{sender:'ready'}))});
  add('boot-0',2,['p0-bootstrap'],'PP0 提出准入候选 R','PP0 轮询本地 sender，将 WaitingForInput 的 R 加入 good 候选；主线没有 bad 请求。','候选还要经过 PP1 和 PP2。');
  add('boot-1',2,['p1-bootstrap'],'PP1 合并准入候选','PP1 对 good 名单求交，对 bad 名单求并。R 在本级也已就绪，因此保留在 good 中。','PP2 还要确认自己的 sender。',{routes:[wire('p0-bootstrap','p1-bootstrap')]});
  add('boot-2',2,['p2-bootstrap'],'PP2 形成 bootstrap 共识','末级完成最后一次合并，R 属于三级共同认可的 good 名单。','结论必须回流，各级才能完成准入。',{routes:[wire('p1-bootstrap','p2-bootstrap')]});
  add('admit-0',2,['p0-bootstrap','p0-schedule'],'准入结论回到 PP0','PP2 → PP0 返回 good 名单；PP0 分配 metadata 槽、初始化 sender，再将 R 放进 waiting_queue。','本轮选批已发生；R 由后续选批使用。',{routes:[wire('p2-bootstrap','p0-bootstrap','control','return')],updates:[update(0,{queue:'waiting',metadata:true})]});
  add('admit-1',2,['p1-bootstrap','p1-schedule'],'PP1 收到准入结论','PP0 → PP1 转发结论；PP1 也完成本地 finalize_bootstrap，进入 waiting_queue。','结论继续到 PP2。',{routes:[wire('p0-bootstrap','p1-bootstrap')],updates:[update(1,{queue:'waiting',metadata:true})]});
  add('admit-2',2,['p2-bootstrap','p2-schedule'],'PP2 完成本地准入','结论绕回 PP2，第三个级完成本地准入。本演示接下来展开三级的选批与计算。','调度预算和请求槽 / KV 空间允许。',{routes:[wire('p1-bootstrap','p2-bootstrap')],updates:[update(2,{queue:'waiting',metadata:true})]});
  add('batch-0',3,['p0-schedule'],'PP0 为 R 组批、分配 KV','R 从 waiting_queue 进入单请求 batch B。prepare_for_extend 准备输入及本级 KV 写入位置。','输入和缓存位置准备完成。',{updates:[update(0,{queue:'batch',kv:'reserved',holdsKv:true})]});
  add('forward-0',4,['p0-compute'],'PP0 完成第一段前向','从 embedding 开始计算 L0–L9，写入这些层的 prompt KV，产生 hidden_states / residual。图中的状态表示本小步结束后的状态。','launch_event 的流依赖满足后才可发送激活。',{kind:'proxy',routes:[wire('p0-schedule','p0-compute','proxy')],updates:[update(0,{kv:'written',computed:true})]});
  add('batch-1',3,['p1-schedule'],'PP1 准备自己的 batch','PP1 也要完成本地选批与 KV 分配。实际程序允许这部分准备与 PP0 的工作交错；此处按依赖展开。','本地 batch 准备好，并接收 B 对应的上游激活。',{updates:[update(1,{queue:'batch',kv:'reserved',holdsKv:true})]});
  add('proxy-01',5,['p0-compute','p1-compute'],'激活从 PP0 交给 PP1','跨级传递的是中间激活。PP0 的 L0–L9 KV 继续留在 PP0，不随 proxy 搬到 PP1。','PP1 收到正确 batch 的激活。',{kind:'proxy',routes:[wire('p0-compute','p1-compute','proxy')],updates:[update(1,{proxy:true})]});
  add('forward-1',5,['p1-compute'],'PP1 完成中间段前向','使用上游激活计算 L10–L19，写入本级 KV；继续产生给 PP2 的激活。','本级前向事件依赖满足，PP2 的接收已准备。',{kind:'proxy',updates:[update(1,{kv:'written',computed:true})]});
  add('batch-2',3,['p2-schedule'],'PP2 准备末级 batch','末级同样准备本地 batch 和 KV 写入位置；它不会跳过调度与显存分配。','接收 PP1 对应的中间激活。',{updates:[update(2,{queue:'batch',kv:'reserved',holdsKv:true})]});
  add('proxy-12',6,['p1-compute','p2-compute'],'激活从 PP1 交给 PP2','hidden_states / residual 沿模型层方向传到末级。此时 PP0、PP1 分别保留已生成的本级 KV。','末级输入满足前向条件。',{kind:'proxy',routes:[wire('p1-compute','p2-compute','proxy')],updates:[update(2,{proxy:true})]});
  add('forward-2',6,['p2-compute'],'PP2 完成前向并采样 t₀','计算 L20–L29、末级归一化与输出头，采样首 token t₀；输出与 event 进入 last_rank_comm_queue。','末级输出满足事件依赖，进入 output 回流通道。',{kind:'output',updates:[update(2,{kv:'written',computed:true,sampled:true})]});
  add('output-20',7,['p2-compute','p0-transfer'],'首 token 的结果回到 PP0','末级发送 output 给 PP0。接收端匹配旧 batch 槽，准备结果并等待必要的 D2H 事件。','对应旧 B 的结果可在 CPU 上处理。',{kind:'output',routes:[wire('p2-compute','p0-transfer','output','output-return')],updates:[update(0,{output:true})]});
  add('result-0',8,['p0-transfer'],'PP0 处理结果，进入 inflight','将 t₀ 追加到本地 Req，维护缓存引用，再把 R 加入 inflight_queue。只完成 PP0 的本地结果处理即可继续。','bootstrap 已完成，可以提交本级 KV。',{kind:'output',updates:[update(0,{queue:'inflight',token:true})]});
  add('kv-0',8,['p0-transfer','decode-kv'],'PP0 提交本级 KV 发送','准备最终块 metadata 与传输页索引，向 Decode 提交 L0–L9 的 KV。send 返回仍需保留源资源。','后台传输推进；其他级仍可处理回流结果。',{kind:'kv',routes:[wire('p0-transfer','decode-kv','kv','kv')],updates:[update(0,{sent:true,sender:'transferring'})]});
  add('output-01',7,['p0-transfer','p1-transfer'],'结果沿环传给 PP1','PP0 转发保存的 pp_outputs；PP1 收到首 token 的结果并等待 D2H。此时 PP0 的 KV 发送可以仍在进行。','PP1 的结果可供本地处理。',{kind:'output',routes:[wire('p0-transfer','p1-transfer','output')],updates:[update(1,{output:true})]});
  add('result-1',8,['p1-transfer'],'PP1 处理自己的最终结果','PP1 更新 Req.output_ids，加入自己的 inflight_queue。不同级的 batch、请求对象与传输状态相互独立。','本地结果处理完毕，发送条件满足。',{kind:'output',updates:[update(1,{queue:'inflight',token:true})]});
  add('kv-1',8,['p1-transfer','decode-kv'],'PP1 提交中间层 KV','PP1 直接向 Decode 的对应层内存发送 L10–L19 KV。无需先把 KV 汇总到末级。','等待本级后端终态，继续推进 output 环。',{kind:'kv',routes:[wire('p1-transfer','decode-kv','kv','kv')],updates:[update(1,{sent:true,sender:'transferring'})]});
  add('output-12',7,['p1-transfer','p2-transfer'],'结果绕回 PP2','PP1 转发结果给 PP2。末级也通过回流路径处理本地旧 batch，而不是采样完就跳过收尾。','PP2 的 D2H 事件已完成。',{kind:'output',routes:[wire('p1-transfer','p2-transfer','output')],updates:[update(2,{output:true})]});
  add('result-2',8,['p2-transfer'],'PP2 进入自己的 inflight','PP2 对本地 Req 追加 t₀，进入 inflight_queue；三个级最终都进入传输跟踪阶段。','提交最后一段层的 KV。',{kind:'output',updates:[update(2,{queue:'inflight',token:true})]});
  add('kv-2',8,['p2-transfer','decode-kv'],'PP2 提交最后一段层 KV','末级向 Decode 发送 L20–L29 的 KV 与最终块信息。已经有 t₀ 不代表 t₀ 自己的 KV 也由这次 prompt 前向计算好了。','继续轮询三个 sender。',{kind:'kv',routes:[wire('p2-transfer','decode-kv','kv','kv')],updates:[update(2,{sent:true,sender:'transferring'})]});
  add('poll-0',9,['p0-transfer','p0-release'],'PP0 观察到本级传输成功','sender₀ 已到 Success，但 R 仍留在 PP0 的 inflight_queue，KV 与 metadata 暂不释放。','等待其他级也到终态。',{updates:[update(0,{sender:'success',terminal:true})]});
  add('poll-1-wait',9,['p1-transfer'],'PP1 暂时仍在传输','为展示等待门槛，本例安排 sender₁ 稍后才报告完成。它当前不是终态，不能贡献释放候选 R。','继续后续轮询，不能提前释放。');
  add('poll-2',9,['p2-transfer','p2-release'],'PP2 观察到本级传输成功','sender₂ 到 Success。PP0 和 PP2 都成功，也不足以证明整条请求可释放。','终态名单需要三级求交。',{updates:[update(2,{sender:'success',terminal:true})]});
  add('terminal-0-wait',9,['p0-release'],'PP0 提出终态候选 {R}','本级 Success / Failed 都属于终态候选。本演示仅走 Success 分支。','候选送到 PP1 求交。');
  add('terminal-1-wait',9,['p1-release','p2-release'],'PP1 尚未完成，交集不含 R','{R} 与 PP1 的空终态集合求交得到空集；再传到 PP2，R 仍不在释放名单中。这里只合并展示空名单的继续传播。','等 PP1 完成后，在后续轮次重新汇合。',{routes:[wire('p0-release','p1-release'),wire('p1-release','p2-release')]});
  add('poll-1',9,['p1-transfer','p1-release'],'后续轮询：PP1 也到 Success','现在三个 sender 都到终态；Decode 已具备这次交接的数据，P 侧仍需走自己的 release 共识。','后续 PP 轮次再次汇合终态名单。',{updates:[update(1,{sender:'success',terminal:true})]});
  add('terminal-0',9,['p0-release'],'PP0 再次提出 {R}','PP0 将终态请求 R 作为候选向下一级传播。','PP1、PP2 依次确认。');
  add('terminal-1',9,['p1-release'],'PP1 的交集保留 R','本级也到 Success，{R} ∩ {R} 仍为 {R}。','PP2 完成最后一次求交。',{routes:[wire('p0-release','p1-release')]});
  add('terminal-2',9,['p2-release'],'PP2 形成全 PP 终态交集','R 进入最终交集。这个结论允许各级开始一致收尾，但仍要经过名单回流与本地复查。','末级发出 release_rids。',{routes:[wire('p1-release','p2-release')],consensus:true});
  add('release-0',10,['p0-release'],'release 名单从 PP2 回到 PP0','PP0 收到名单，再轮询自己的 sender；名单包含 R 且本地仍为终态，才能执行清理。','本地 Success 分支可释放请求占用。',{routes:[wire('p2-release','p0-release','control','return')],updates:[update(0,{permit:true})]});
  add('finish-0',11,['p0-release','frontend'],'PP0 清理，并输出 P 侧完成结果','释放请求 KV 引用，清理 sender，发送完成输出并归还 metadata 槽。只有有效 IPC 输出 rank 对外应答，其他级仍继续各自清理。','release 结论还需传到 PP1、PP2。',{routes:[wire('p0-release','frontend','control','response')],updates:[update(0,{queue:'done',kv:'cache',holdsKv:false,metadata:false,sender:'cleared',released:true})]});
  add('release-1',10,['p1-release'],'PP1 收到 release 并复查','PP0 → PP1 转发 release 名单。PP1 按名单过滤 inflight 请求，并再次检查本地终态。','本地 sender 已确认 Success。',{routes:[wire('p0-release','p1-release')],updates:[update(1,{permit:true})]});
  add('finish-1',11,['p1-release'],'PP1 释放本地请求资源','PP1 归还请求占用与 metadata 槽、清理 sender，并将 R 移出本地 inflight。前缀缓存可继续管理可复用 KV。','名单继续到 PP2。',{updates:[update(1,{queue:'done',kv:'cache',holdsKv:false,metadata:false,sender:'cleared',released:true})]});
  add('release-2',10,['p2-release'],'release 结论绕回 PP2','PP1 → PP2 返回名单；末级也按同样规则复查本地终态。','完成最后一份本地请求资源清理。',{routes:[wire('p1-release','p2-release')],updates:[update(2,{permit:true})]});
  add('finish-2',11,['p2-release'],'PP2 完成本地清理','第三份 Req 退出 inflight，三个级均不再持有 R 的活跃 KV 引用与 metadata 槽。','Prefill 侧的生命周期已经结束。',{updates:[update(2,{queue:'done',kv:'cache',holdsKv:false,metadata:false,sender:'cleared',released:true})]});
  add('handoff',11,['decode-run'],'P 侧收尾完成，D 侧继续生成','Decode 使用收到的 prompt KV 与 t₀ 继续生成。它可能早已开始：这里是阅读终点，不是“等 P 全部清理才允许 D 启动”的同步屏障。','后续生成属于 Decode 的生命周期；可重播或回看任一动作。',{kind:'kv',routes:[wire('decode-kv','decode-run','kv')]});

  const queueLabels={none:'尚未接入',bootstrap:'bootstrap_queue',waiting:'waiting_queue',batch:'EXTEND batch B',inflight:'inflight_queue',done:'已移出 · P 侧完成'};
  const kvLabels={none:'KV · 尚未分配',reserved:'KV · 已分配，待写入',written:'KV · 已写入，引用保留',cache:'KV · 请求引用已释放'};
  const senderLabels={none:'未创建',bootstrapping:'握手中',ready:'WaitingForInput',transferring:'等待传输终态',success:'Success',cleared:'已清理'};
  function snapshot(index, sequence=events) {
    const state={ranks:Array.from({length:3},()=>({queue:'none',kv:'none',sender:'none',metadata:false,holdsKv:false,computed:false,proxy:false,output:false,token:false,sampled:false,sent:false,terminal:false,permit:false,released:false})),visited:[],consensus:false};
    const visited=new Set();
    for(const event of sequence.slice(0,index+1)) {
      event.active.forEach(n=>visited.add(n));
      for(const patch of event.updates) { const {rank,...values}=patch; Object.assign(state.ranks[rank],values); }
      if(event.consensus) state.consensus=true;
    }
    state.visited=[...visited];
    return state;
  }
  const tokenIds=[101,872,196,305,418,527,639,741,853,964,1076,1187];
  function scenario(mode='single') {
    const chunked=mode==='chunked', size=chunked?4:12;
    const makeChunk=i=>({index:i,start:(i-1)*size,end:Math.min(i*size,12),size,total:12,last:i*size>=12});
    const sequence=[];
    const batchStart=events.findIndex(e=>e.id==='batch-0');
    sequence.push(...events.slice(0,batchStart).map(e=>({...e,chunk:null})));
    function cut(chunk){return {id:`chunk-${chunk.index}-cut`,phase:3,active:['p0-schedule'],kind:'control',routes:[],updates:[],chunk,
      title:`切出第 ${chunk.index} 块：token [${chunk.start}, ${chunk.end})`,
      description:`请求 R 的完整 12-token 序列不变。已有前缀 [0, ${chunk.start})，本轮预算 ${size}，将 extend_range 设为 [${chunk.start}, ${chunk.end})；只对这 ${chunk.end-chunk.start} 个新 token 做前向。`,
      gate:chunk.last?'这是最终块，随后才会追加有效首 token、发送最终 metadata。':'后面仍有 token；本轮完成后保留同一条 chunked_req，继续选下一块。'};}
    if(chunked){
      for(let n=1;n<=2;n++){
        const chunk=makeChunk(n);
        sequence.push(cut(chunk));
        for(const original of events.slice(batchStart,events.findIndex(e=>e.id==='output-20'))){
          const event={...original,id:`chunk-${n}-${original.id}`,chunk,middle:true,title:`块 ${n} · ${original.title}`,updates:original.updates.map(u=>({...u}))};
          if(original.id==='forward-2'){
            event.title=`块 ${n} · PP2 完成中间块前向`;
            event.description='末级完成本块模型计算。中间块的结果仍需沿 PP 路径处理，但不会作为这条请求的有效首 token 追加到 output_ids。';
            event.gate='本块结果就绪；分别处理各级的中间块状态与发送范围。';
            event.updates=event.updates.map(u=>({...u,sampled:false}));
          }
          if(original.id.startsWith('batch-'))event.description=`在本级准备同一 R 的第 ${n} 块：前缀 [0, ${chunk.start})，本次新增 [${chunk.start}, ${chunk.end})，分配本级新增 KV 位置。`;
          sequence.push(event);
        }
        for(let r=0;r<3;r++){
          sequence.push({id:`chunk-${n}-result-${r}`,phase:7,active:[`p${r}-transfer`],kind:'output',middle:true,chunk,
            title:`块 ${n} · PP${r} 处理中间块结果`,description:'匹配本块结果，维护尚未结束的请求与缓存。此时不追加有效 t₀，也不把 R 当成最终块加入完成交接的 inflight。',gate:'本级已计算的 KV 范围可供发送；请求仍需下一块。',
            routes:[wire(r===0?'p2-compute':`p${r-1}-transfer`,`p${r}-transfer`,'output',r===0?'output-return':'direct')],updates:[]});
          sequence.push({id:`chunk-${n}-kv-${r}`,phase:8,active:[`p${r}-transfer`,'decode-kv'],kind:'kv',middle:true,chunk,
            title:`块 ${n} · PP${r} 提交 KV [${chunk.start}, ${chunk.end})`,
            description:`本例 page_size=2，end_idx=${chunk.end} 恰好页对齐。提交本级这些 token 的 KV，推进 start_send_idx；保留请求引用。非 overlap 路径由后续 process_prefill_chunk 提交，overlap 路径在中间块结果处理时提交。`,
            gate:'提交不等于传输成功。中间块不发送最终 metadata，也不执行请求释放。',routes:[wire(`p${r}-transfer`,'decode-kv','kv','kv')],updates:[]});
        }
      }
      sequence.push(cut(makeChunk(3)));
    }
    sequence.push(...events.slice(batchStart).map(e=>({...e,chunk:makeChunk(chunked?3:1)})));
    const selected=chunked?sequence:events.map((e,i)=>({...e,chunk:i<batchStart?null:makeChunk(1)}));
    function selectedSnapshot(index){
      const result=snapshot(index,selected);
      result.ranks.forEach(r=>Object.assign(r,{cacheEnd:0,sendEnd:0,selectedEnd:0}));
      for(const e of selected.slice(0,index+1)){
        if(!e.chunk)continue;
        for(let r=0;r<3;r++){
          if(e.id.endsWith(`batch-${r}`))result.ranks[r].selectedEnd=e.chunk.end;
          if(e.id.endsWith(`forward-${r}`))result.ranks[r].cacheEnd=e.chunk.end;
          if(e.id.endsWith(`kv-${r}`))result.ranks[r].sendEnd=e.chunk.end;
        }
      }
      return result;
    }
    return {nodes,events:selected,snapshot:selectedSnapshot,queueLabels,kvLabels,senderLabels,tokenIds,mode,size};
  }
  return {nodes,events,snapshot,queueLabels,kvLabels,senderLabels,tokenIds,scenario};
});
