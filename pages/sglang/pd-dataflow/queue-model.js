/* Local Req membership snapshots. Containers and aliases are not exclusive states. */
(function(root,factory){
  const api=factory(typeof module==='object'&&module.exports?require('./model.js'):root.PDDataflow);
  if(typeof module==='object'&&module.exports)module.exports=api;else root.PDQueues=api;
})(globalThis,M=>{
  'use strict';
  const sources={...M.sources,
    enqueue:['disaggregation/prefill.py',427,'self.queue.append(req)'],
    dequeue:['disaggregation/prefill.py',536,'self.queue = ['],
    waiting:['managers/scheduler_pp_mixin.py',621,'self.waiting_queue.extend(good_reqs)'],
    select:['managers/scheduler.py',4034,'self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]'],
    chunkRef:['managers/scheduler.py',4042,'self.chunked_req = adder.new_chunked_req'],
    batch:['managers/scheduler.py',4050,'new_batch = ScheduleBatch.init_new('],
    continueChunk:['managers/schedule_policy.py',950,'def add_chunked_req'],
    inflight:['disaggregation/prefill.py',860,'self.disagg_prefill_inflight_queue.append(req)'],
    pending:['disaggregation/prefill.py',1513,'self.disagg_prefill_pending_chunk_rids.add(req.rid)'],
    rebuild:['disaggregation/prefill.py',1078,'self.disagg_prefill_inflight_queue = undone_reqs'],
    filter:['managers/schedule_batch.py',3571,'def filter_batch'],
    requeue:['disaggregation/prefill.py',1515,'def optimistic_release_and_requeue'],
  };
  const nodes=[
    {id:'bootstrap',name:'disagg_prefill_bootstrap_queue.queue',kind:'请求列表',note:'尚未完成本地握手准入；good 后仍可能卡 metadata。',source:'admission'},
    {id:'waiting',name:'waiting_queue',kind:'请求列表',note:'待选批的 Req；预算不够就留在这里，并非严格 FIFO。',source:'select'},
    {id:'inflight',name:'disagg_prefill_inflight_queue',kind:'请求列表',note:'最终块已处理，等待传输收尾；不在 release 名单或复查非终态就保留。',source:'release'},
    {id:'batch',name:'ScheduleBatch.reqs',kind:'执行容器',note:'本次执行或待清理旧 batch 的 Req 引用；也会由 mbs / last_mbs 持有。不是 running_batch 必然含 R0。',source:'batch'},
    {id:'chunk',name:'scheduler.chunked_req',kind:'单个引用',note:'未算完输入的续算指针；可以与 batch.reqs 同时指向同一个 Req。',source:'continueChunk'},
    {id:'exit',name:'完成 / 中止',kind:'教学出口 · 非源码队列',note:'请求业务收尾不等于所有 Python 引用立即消失；旧 batch 可随后过滤。',source:'release'},
  ];
  const clone=x=>JSON.parse(JSON.stringify(x));
  function create(id,rank=0){
    id=M.scenarios.some(x=>x.id===id)?id:'normal';rank=Math.max(0,Math.min(2,Math.floor(Number(rank)||0)));
    const state={bootstrap:false,waiting:false,inflight:false,batch:false,chunk:false,exit:false,pendingIds:false,poll:'Bootstrapping',pending:true,metadata:false,kv:false,finished:'未完成',batchName:'—'};
    const frames=[];
    function emit(key,title,detail,code,source,patch={},from=null,to=null,kind='membership'){
      const before=clone(state);Object.assign(state,patch);frames.push({key,title,detail,code,source,before,after:clone(state),from,to,kind});
    }
    emit('intake','本级创建 Req，加入握手列表','三个 PP 级各自持有本地 Req。此面板只展开所选一级的引用变化，R0 是请求标识；跨级同名不等于同一 Python 对象。','self.queue.append(req)','enqueue',{bootstrap:true});
    if(id==='bootstrap-wait')emit('handshake-wait','共同 good 尚不包含 R0','即使本级已经 WaitingForInput，也要等待 PP1 握手及有效共识。列表保持不变。','poll_and_all_reduce_pp(...) → None','admission',{poll:rank===1?'Bootstrapping':'WaitingForInput'},'bootstrap','bootstrap','wait');
    if(['bootstrap-fail','abort'].includes(id)){
      emit('bad','bad 名单决定中止','握手失败或 FINISH_ABORT 进入 bad 并集。处理失败后，通过列表过滤移除；没有进入 waiting_queue，也没有本次计算 KV。','handle_bootstrap_failure(req); indices_to_remove.add(i)','bootstrapFailure',{poll:id==='bootstrap-fail'&&rank===1?'Failed':'WaitingForInput',finished:'中止'});
      emit('abort-remove','从 bootstrap 列表移除','这是失败出口，不是调度入队。此后本例结束。','self.queue = [entry ... if i not in indices_to_remove]','dequeue',{bootstrap:false,exit:true},'bootstrap','exit');
      return {id,rank,frames};
    }
    emit('good','收到有效握手名单','good 仅允许尝试本地 finalize；队列归属、poll 和资源状态是不同维度。','pop_bootstrapped(pp_good_rids, pp_bad_rids)','admission',{poll:'WaitingForInput'});
    if(id==='metadata')emit('metadata-wait','metadata 不足，请求仍在 bootstrap','本例 PP0 分配失败；下游也不能凭旧的原始 good 越过实际准入名单。假定后续槽位可用再继续。','if not finalize_bootstrap(req): continue','metadata',{},'bootstrap','bootstrap','wait');
    emit('finalize','初始化 sender，准备出队','metadata 已分配、pending_bootstrap 已清除，此时 pop_bootstrapped 还没执行最后的列表重建。相同 queue 内，字段已经改变。','finalize_bootstrap(req) → True','metadata',{pending:false,metadata:true});
    emit('admit','bootstrap 出队 → waiting 入队','合并展示相邻两个操作：pop_bootstrapped 过滤原列表、返回 good_reqs；调用方 extend 到 waiting_queue。移动的是引用，Req 没有复制。','self.queue = [entry ...]; self.waiting_queue.extend(good_reqs)','waiting',{bootstrap:false,waiting:true},'bootstrap','waiting');
    if(id==='budget')emit('budget-wait','本轮不选 R0，继续留在 waiting','NO_TOKEN 使本轮不能选入；已经分配的 metadata 保留。后续预算充足再进入下一步。','_select_prefill_admission(...) → NO_TOKEN','budget',{},'waiting','waiting','wait');
    const chunked=['chunks','skip-chunk-output'].includes(id),ends=chunked?[4,8,12]:[12];
    ends.forEach((end,i)=>{
      const last=end===12,b=`B${i+1}`,start=i*4;
      if(i===0){
        emit(`${b}-select`,'waiting 出队，选入本轮执行集合','只移除 can_run_set 中的请求；本例把过滤和 ScheduleBatch.init_new 合为一次选批展示。不把 waiting 当作 pop(0) 队列。','waiting_queue = [x ... if x not in can_run_set]; ScheduleBatch.init_new(can_run_list, ...)','select',{waiting:false,batch:true,chunk:!last,batchName:b},'waiting','batch');
        if(!last)emit(`${b}-alias`,'同一个 Req 同时被 batch 和续算指针引用','上一选批步骤已在创建 batch 前设置 chunked_req。本帧展开观察两处同时存在的引用：虚线不是请求在两个队列之间搬家，也没有创建第二个 Req。','self.chunked_req = adder.new_chunked_req','chunkRef',{chunk:true},'batch','chunk','reference');
      }else{
        emit(`${b}-continue`,last?'最后一块入 batch，清空续算指针':'续算引用直接选入下一批','add_chunked_req 将同一 Req 加入 can_run_list；仍有输入则返回 req，最后一块返回 None。没有退回 waiting_queue。本例预算足够。','self.chunked_req = adder.add_chunked_req(self.chunked_req)','continueChunk',{batch:true,chunk:!last,batchName:b},'chunk','batch','reference');
      }
      emit(`${b}-forward`,`${b} 计算输入 [${chunked?start:0}, ${end})`,'准备 batch 的 KV 映射并执行前向；这里按因果步骤归并，设备完成由事件约束。batch 引用与 KV 所有权分别显示。','ScheduleBatch.init_new(...); _pp_launch_batch(...)','forward',{kv:true});
      if(id==='output-wait')emit('d2h','结果尚不可用，batch 引用保持','先等待 D2H，才处理最终块结果；尚未 append 到 inflight。','d2h_event.synchronize()','d2h',{},'batch','batch','wait');
      if(!last){
        emit(`${b}-middle`,'中间块结果：不加入 inflight',id==='skip-chunk-output'?'本例开启纯中间块跳过结果通信；本地占位不产生有效首 token，队列路径不变。':'普通中间块尚未产生最终首 token；保留 chunked_req，不当作整个请求已完成。',id==='skip-chunk-output'?'next_pp_outputs = None; 本地占位结果':'process_batch_result_disagg_prefill(...)',id==='skip-chunk-output'?'skip':'result');
        emit(`${b}-send`,'提交中间块 KV，记录 rid 集合','disagg_prefill_pending_chunk_rids 只存已发送但尚未结束分块的 rid，不是 Req 队列，也不是 Decode 已接收证明。','send_kv_chunk(req); pending_chunk_rids.add(req.rid)','pending',{pendingIds:true});
        emit(`${b}-filter`,'旧 batch 去掉 R0，续算指针仍保留','process_prefill_chunk 排除 chunked_req 并 filter_batch。移除的是旧执行容器引用，不释放请求 KV。','last_batch.filter_batch(chunked_req_to_exclude=...)','chunk',{batch:false},'batch','chunk','reference');
      }else{
        emit('append-inflight','最终块结果：加入 inflight，旧 batch 仍可引用 R0','追加有效首 token 后 append(req)。这不是把 Req 从 batch 对象里立刻搬走；两处引用可以共存。实线强调请求加入在途列表。','self.disagg_prefill_inflight_queue.append(req)','inflight',{inflight:true},'batch','inflight');
        emit('send-final','提交最终 KV，并移除 pending rid','pending_chunk_rids.discard 只清理跟踪集合；请求仍在 inflight，KV 和 metadata 仍被持有。','send_kv_chunk(req, last_chunk=True); pending_chunk_rids.discard(req.rid)','send',{pendingIds:false});
      }
    });
    emit('poll','在同一个 inflight 中观察传输进度','poll 改变不要求换队列；WaitingForInput、Transferring，甚至等待共同 release 的 Success 都可处于这个列表。','poll_and_all_reduce_attn_cp_tp_group(...)','release',{poll:'Transferring'});
    if(['transfer-wait','transfer-fail'].includes(id))emit('release-wait','共同终态未齐：保留到 undone_reqs','PP2 仍在传输；本地成功或失败也不能越过本例的 release 筛选。稍后 PP2 到终态才继续。','if req.rid not in rids_to_check: undone_reqs.append(req)','release',{poll:rank===2?'Transferring':id==='transfer-fail'&&rank===1?'Failed':'Success'},'inflight','inflight','wait');
    if(id==='repoll'&&rank===1)emit('repoll','名单允许，但本地复查非终态：仍保留','这是源码防御的采样差异分支，不断言后端必然发生状态倒退。后续有效名单与终态采样到来后才能清理。','if poll not in (Success, Failed): undone_reqs.append(req)','release',{},'inflight','inflight','wait');
    const failed=id==='transfer-fail'&&rank===1;
    emit('terminal','名单允许，且本地复查到终态','Success 与 Failed 都可进入收尾，但调用不同处理分支。各级独立清理，不把这一步画成三级同步释放。','rid in rids_to_check and poll in (Success, Failed)','release',{poll:failed?'Failed':'Success'});
    emit('retire','先处理结果与资源，再重建 inflight 列表','归还本地请求 KV 所有权和 metadata。此帧故意保留列表成员，展示 finished 与队列归属并非同一个变量；不代表缓存物理清空。',failed?'handle_inflight_transfer_failure(req); maybe_release_metadata_buffer(...)':'release_kv_cache(req, ...); sender.clear(); maybe_release_metadata_buffer(...)',failed?'failure':'release',{kv:false,metadata:false,finished:failed?'中止':'完成'});
    emit('remove-inflight','从 inflight 出队','未完成者保留在 undone_reqs；已完成者不进入重建列表。出口不是源码中的 done_queue；旧 batch 引用仍可稍后清理。','self.disagg_prefill_inflight_queue = undone_reqs','rebuild',{inflight:false,exit:true},'inflight','exit');
    emit('filter-finished','后续旧 batch 过滤已完成请求','filter_batch 按 finished() / chunked_req_to_exclude 去掉引用。本图只展示相关容器，不承诺 Python 对象此刻被 GC；PP 微批槽仍可能持有空 batch 对象。','keep_indices = [i ... if not self.reqs[i].finished() ...]','filter',{batch:false},'batch','exit','reference');
    return {id,rank,frames};
  }
  function sourceUrl(key){const [file,line]=sources[key];return `https://github.com/sgl-project/sglang/blob/${M.commit}/python/sglang/srt/${file}#L${line}`;}
  return {create,nodes,sources,sourceUrl,scenarios:M.scenarios,commit:M.commit};
});
