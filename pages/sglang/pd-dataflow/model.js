/* A source-grounded causal walkthrough, not a scheduler or transport simulator. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.PDDataflow = api;
})(globalThis, () => {
  'use strict';
  const commit = '882577451e764a515df2a386a055012e8f075a16';
  const sources = {
    bootstrap: ['managers/scheduler_pp_mixin.py', 625, 'def _pp_pd_get_bootstrapped_ids'],
    admission: ['disaggregation/prefill.py', 450, 'def pop_bootstrapped'],
    metadata: ['disaggregation/prefill.py', 395, 'def finalize_bootstrap'],
    budget: ['managers/schedule_policy.py', 1267, 'def _select_prefill_admission'],
    commit: ['managers/schedule_policy.py', 1347, 'def _commit_prefill_admission'],
    forward: ['managers/scheduler_pp_mixin.py', 1601, 'def _pp_launch_batch'],
    proxy: ['managers/scheduler_pp_mixin.py', 913, 'def _pp_recv_proxy_tensors'],
    output: ['managers/scheduler_pp_mixin.py', 742, 'def _pp_commit_send_output_work_and_preprocess_output_tensors'],
    d2h: ['managers/scheduler_pp_mixin.py', 344, 'd2h_event.synchronize()'],
    result: ['disaggregation/prefill.py', 800, 'for i, (req, next_token_id) in enumerate('],
    chunk: ['disaggregation/prefill.py', 1210, 'def process_prefill_chunk'],
    send: ['disaggregation/prefill.py', 1311, 'def _send_kv_chunk'],
    terminal: ['managers/scheduler_pp_mixin.py', 667, 'def _pp_pd_get_prefill_transferred_ids'],
    release: ['disaggregation/prefill.py', 972, 'def process_disagg_prefill_inflight_queue'],
    failure: ['disaggregation/prefill.py', 1082, 'def handle_inflight_transfer_failure'],
    bootstrapFailure: ['disaggregation/prefill.py', 1150, 'def handle_bootstrap_failure'],
    verdict: ['disaggregation/utils.py', 52, 'def poll_and_all_reduce_pp'],
    skip: ['managers/scheduler_pp_mixin.py', 44, 'def _pp_can_skip_output_comm'],
  };
  const scenarios = [
    ['normal', '正常交接', '从握手到资源归还，分清激活、结果与 KV 三条数据通路。'],
    ['bootstrap-wait', '握手尚未就绪', 'PP1 仍在 Bootstrapping，R0 不在共同 good 中；随后条件改变，再求共识。'],
    ['bootstrap-fail', '握手局部失败', '仅 PP1 观察到 Failed，也会进入 bad 并集，使各级清理同一请求。'],
    ['abort', '握手期间取消', '即使 sender.poll 未变为 Failed，本地 FINISH_ABORT 也进入 bad 并集。'],
    ['metadata', 'metadata 槽不足', '所有级握手已好，PP0 仍无法 finalize；实际准入名单暂不向后传递 R0。'],
    ['budget', '选批预算不足', '已进入 waiting_queue，仍可能因 token 预算不能进入本轮 batch。'],
    ['chunks', '分块 Prefill', '12 个 token 分为三块；中间块发送整页 KV，保留请求，最终块才追加有效首 token。'],
    ['output-wait', '结果 D2H 尚未完成', '模型算完不等于 CPU 可以处理结果；先等待对应 D2H 事件。'],
    ['transfer-wait', '某一级 KV 仍在传输', 'PP0 与 PP1 已 Success，但 PP2 仍 Transferring，共同终态集合不包含 R0。'],
    ['transfer-fail', '传输局部失败', 'Failed 也属于终态；仍需等待另一在途级，最终各级按本地成功或失败分支收尾。'],
    ['repoll', 'release 后复查仍在途', '名单是先前采样结果；本地复查不是终态就保留资源，不能只凭名单释放。'],
    ['skip-chunk-output', '中间块跳过结果通信', '开启源码开关后，单请求纯中间块可使用占位结果；激活和 KV 通路仍存在。'],
  ].map(([id, label, description]) => ({id, label, description}));
  const queues = {bootstrap:'握手队列', waiting:'等待队列', batch:'本地 batch', chunked:'续算中的 chunked_req', inflight:'KV 在途队列', done:'已正常收尾', aborted:'已中止收尾'};
  const copy = x => JSON.parse(JSON.stringify(x));
  const terminal = x => ['Success', 'Failed'].includes(x);
  function consensus(polls, aborted = []) {
    return {good: polls.every(p => p === 'WaitingForInput') && !aborted.length ? ['R0'] : [], bad: polls.includes('Failed') || aborted.length ? ['R0'] : []};
  }
  function terminalSet(polls) { return polls.every(terminal) ? ['R0'] : []; }
  function create(id) {
    if (!scenarios.some(s => s.id === id)) id = 'normal';
    const state = {ranks: Array.from({length:3}, () => ({queue:'bootstrap', poll:'Bootstrapping', metadata:false, kv:false, computed:0, sent:0, token:false, pending:true})), good:[], bad:[], release:[], batch:'—'};
    const frames = [];
    function emit(key, title, detail, source, mutate = () => {}, flow = null, gate = '') {
      const before = copy(state); mutate();
      frames.push({key, title, detail, source, before, after:copy(state), flow, gate});
    }
    function all(patch) { state.ranks.forEach(q => Object.assign(q, patch)); }
    function collect(round) {
      const polls = state.ranks.map(q => q.poll), verdict = consensus(polls, id === 'abort' ? [1] : []);
      let good = true, bad = false;
      for (let r=0; r<3; r++) {
        good = good && polls[r] === 'WaitingForInput';
        bad = bad || polls[r] === 'Failed' || (id === 'abort' && r === 1);
        if (bad) good = false;
        const g = good, b = bad;
        emit(`bootstrap-${round}-${r}`, `PP${r} 合并握手候选`, 'good 逐级求交，bad 逐级求并；FINISH_ABORT 从 good 移除并加入 bad。面板中的名单是截至本级的结果。', 'bootstrap', () => {state.good=g?['R0']:[]; state.bad=b?['R0']:[];}, {kind:'control',from:r?`PP${r-1}`:'本地 poll',to:`PP${r}`,payload:`good ${g?'{R0}':'∅'} · bad ${b?'{R0}':'∅'}`});
      }
      return verdict;
    }
    emit('start','R0 在三个 Prefill 级建立本地请求','本例输入 12 个 token。三个 rank 各有本地 Req / sender；Decode 是 KV 接收方，不是 PP3。关闭乐观 Prefill、HiCache、staging、投机与前缀复用。','bootstrap');
    emit('poll-bootstrap','采样握手状态',id==='abort'?'PP1 已收到取消并标记 FINISH_ABORT；本例 sender 的 poll 仍为 WaitingForInput。':'poll 是最近一次观察，不是每一条通信完成后都自动更新的状态。','bootstrap', () => {all({poll:'WaitingForInput'}); if(id==='bootstrap-wait')state.ranks[1].poll='Bootstrapping'; if(id==='bootstrap-fail')state.ranks[1].poll='Failed';});
    if (id==='bootstrap-wait') emit('handshake-delay','PP1 还在等待握手','未就绪不会自动变成失败；请求留在 bootstrap_queue。','bootstrap',()=>{state.ranks[1].poll='Bootstrapping';},null,'PP1 握手尚未就绪');
    if (id==='bootstrap-fail') emit('handshake-failure','PP1 观察到 Failed','这次失败将被传播到各级的 bad 名单。','bootstrap',()=>{state.ranks[1].poll='Failed';});
    let verdict = collect(1);
    if (verdict.bad.length) {
      for(let r=0;r<3;r++) emit(`abort-${r}`,`PP${r} 按 bad 名单清理 R0`,'pop_bootstrapped 按共识映射 Failed 并调用 handle_bootstrap_failure。这里尚未分配计算 KV；其它级的最近 poll 可以仍为 WaitingForInput。','bootstrapFailure',()=>{Object.assign(state.ranks[r],{queue:'aborted',pending:false});},{kind:'control',from:r?`PP${r-1}`:'PP2',to:`PP${r}`,payload:'bad {R0}'});
      return {id,frames};
    }
    if (id==='bootstrap-wait') {
      emit('bootstrap-blocked','共同 good 为空：保持等待','没有可准入的 R0，本例也没有其它请求可选。真实服务可继续推进其它请求；这里不制造全服务屏障。','verdict',()=>{},null,'等待外部握手完成');
      emit('bootstrap-ready','后续握手完成，重新采样','示例显式改变外部条件，再开始一次候选传播。','bootstrap',()=>{state.ranks[1].poll='WaitingForInput';}); collect(2);
    }
    if(id==='metadata') {
      emit('metadata-blocked','good {R0}，但 PP0 没有 metadata 槽','finalize_bootstrap 返回 False；R0 仍在 bootstrap_queue。process_bootstrapped_queue 返回实际出队的 good_reqs，因此本轮向下游传递的准入名单不含 R0。','metadata',()=>{state.good=[];},{kind:'control',from:'PP2',to:'PP0',payload:'共同 good {R0} → 实际准入 ∅'},'metadata allocator 无空闲槽');
      emit('metadata-ready','后续 metadata 槽可用，再求共同名单','这是外部资源条件变化的示例，不是等待若干毫秒就一定成功。','metadata'); collect(2);
    }
    for(let r=0;r<3;r++) emit(`admit-${r}`,`PP${r} 完成握手准入`,'分配 metadata 槽，读取 Decode 已有前缀长度（本例为 0），初始化 sender，pending_bootstrap=False，加入 waiting_queue。准入尚不代表进入 batch。','admission',()=>{Object.assign(state.ranks[r],{queue:'waiting',metadata:true,pending:false});},{kind:'control',from:r?`PP${r-1}`:'PP2',to:`PP${r}`,payload:'good {R0} · bad ∅'});
    if(id==='budget') {
      emit('budget-blocked','R0 已就绪，本轮仍不能选入 batch','_select_prefill_admission 返回 NO_TOKEN；已拥有的 metadata 保留，尚未为本次计算持有 KV。','budget',()=>{},null,'本例剩余 token 容量不足');
      emit('budget-ready','后续预算充足，重新选批','示意其它请求退出后容量可用。预算判断、准入提交和实际分配不是同一步。','budget');
    }
    const chunked=['chunks','skip-chunk-output'].includes(id), ends=chunked?[4,8,12]:[12];
    let start=0;
    for(let b=0;b<ends.length;b++) {
      const end=ends[b], last=end===12, batch=`B${b+1}`, skip=id==='skip-chunk-output'&&!last;
      emit(`${batch}-select`,`${batch} 选定 R0 [${start}, ${end})`,'以一个请求展示状态：batch 是某次执行的容器，chunked_req 是继续计算的引用，都不等同于独立的请求队列。页大小为 4，区间均按整页对齐。','commit',()=>{state.batch=batch;});
      for(let r=0;r<3;r++) {
        emit(`${batch}-allocate-${r}`,`PP${r} 准备本地 batch / KV 映射`,'保留此前计算的前缀，仅准备本轮新增 token 的本级 KV 写入位置。','commit',()=>{Object.assign(state.ranks[r],{queue:'batch',kv:true});});
        if(r) emit(`${batch}-proxy-${r}`,`激活 PP${r-1} → PP${r}`,'hidden_states / residual 表示本轮 token 的中间计算结果。它们沿 PP 前向传递；KV 留在生产它的级，走另一条 PD 通路。','proxy',()=>{},{kind:'activation',from:`PP${r-1}`,to:`PP${r}`,payload:`${batch} · R0 [${start}, ${end})`});
        emit(`${batch}-compute-${r}`,`PP${r} 的本块计算完成`,'先提交前向，再由设备事件约束依赖。本帧代表本块计算完成后的教学快照，不把 CPU launch 当作 GPU 已完成。','forward',()=>{state.ranks[r].computed=end;});
      }
      for(let r=0;r<3;r++) {
        emit(`${batch}-output-${r}`,skip?`PP${r} 为纯中间块生成占位结果`:`${batch} 结果回流到 PP${r}`,skip?'SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM 开启、单请求 EXTEND、非最终块且无 logprob 时，next_pp_outputs=None；本地占位不是有效生成 token。':'结果按 batch 槽对应，沿 PP2 → PP0 → PP1 → PP2 传递；结果回流与激活前向是不同数据。',skip?'skip':'output',()=>{},skip?null:{kind:'output',from:r?`PP${r-1}`:'PP2',to:`PP${r}`,payload:`${batch} · ${last?'最终块结果':'中间块结果'}`});
        if(id==='output-wait'&&r===0) emit('d2h-wait','PP0 等待结果 D2H 完成','即使本块 GPU 已计算完成，d2h_event.synchronize() 尚未返回时，也不能调用这份旧 batch 的结果处理。KV 和 metadata 继续保留。','d2h',()=>{},null,'CPU 等待对应 d2h_event');
        emit(`${batch}-result-${r}`,`PP${r} 处理${last?'最终块':'中间块'}结果`,last?'D2H 已就绪，追加有效首 token t₀，加入 inflight_queue。接着在本级提交 KV，不要求所有级先处理完结果。':'中间块不追加有效首 token。保留请求及前缀，后续 process_prefill_chunk 维护续算与发送（本例关闭普通 overlap）。','result',()=>{Object.assign(state.ranks[r],{queue:last?'inflight':'chunked',token:last});});
        emit(`${batch}-send-${r}`,`PP${r} → Decode：提交本级 KV [${start}, ${end})`,last?'最终块写入 metadata 并提交剩余 KV。send 返回和 start_send_idx 前移仅代表已提交，不证明 Decode 已接收或本地可释放。':'在 process_prefill_chunk 的非 overlap 路径提交中间块 KV；非最终块向下对齐整页。源端引用保留，供后续块继续使用。',last?'send':'chunk',()=>{state.ranks[r].sent=end;},{kind:'kv',from:`PP${r}`,to:'Decode',payload:`R0 [${start}, ${end})${last?' + metadata':''}`});
      }
      start=end;
    }
    emit('transfer-poll','再次采样：观察 KV 传输进度','提交边界和传输终态分开记录。此处给出各后端 poll 的示例观察值，不假定 send 会同步设置 Transferring。','terminal',()=>all({poll:'Transferring'}));
    const wait=['transfer-wait','transfer-fail'].includes(id);
    emit('terminal-poll','采样本地 Success / Failed / Transferring','本地 Success 不等于整个 PP 请求成功；Failed 也计入“终态”，但不会变成成功。','terminal',()=>{state.ranks.forEach((q,r)=>{q.poll=id==='transfer-fail'&&r===1?'Failed':wait&&r===2?'Transferring':'Success';});});
    function collectTerminal(round) {
      let allowed=true;
      for(let r=0;r<3;r++) {
        allowed=allowed&&terminal(state.ranks[r].poll); const take=allowed;
        emit(`terminal-${round}-${r}`,`PP${r} 合并终态集合`,'各级的本地终态集合 = Success ∪ Failed；沿 PP 求交。只有最后一级得出的完整交集才可回流，前缀交集不是释放许可。','terminal',()=>{state.release=take?['R0']:[];},{kind:'control',from:r?`PP${r-1}`:'本地 poll',to:`PP${r}`,payload:`截至 PP${r} 的终态交集 ${take?'{R0}':'∅'}`});
      }
    }
    collectTerminal(1);
    if(wait) {
      emit('transfer-blocked','共同终态交集为空：三个级均保留请求','PP2 仍在途，即使 PP0 已成功或 PP1 已失败，也不能凭自己的局部状态绕过本次 PP release 筛选。','release',()=>{},null,'PP2 尚未到终态');
      emit('transfer-ready','后续 PP2 到达 Success，重新求交','本例后续观察发生改变；PP1 若 Failed 则仍保留 Failed，不能将它改写为成功。','terminal',()=>{state.ranks[2].poll='Success';}); collectTerminal(2);
    }
    function cleanup(r, suffix='') {
      emit(`release-${r}${suffix}`,`PP${r} 收到名单，复查本地状态`,'只有 rid 在名单中且本地 poll 仍为 Success / Failed 才继续。名单基于此前采样；复查得到瞬态必须放回 undone。','release',()=>{if(id==='repoll'&&r===1&&!suffix)state.ranks[r].poll='Transferring';},{kind:'control',from:r?`PP${r-1}`:'PP2',to:`PP${r}`,payload:'release {R0}'});
      if(!terminal(state.ranks[r].poll)) {
        emit('repoll-blocked','PP1 保留 KV、metadata 与 inflight 请求','这是源码明确防御的采样差异分支。面板没有假设后端一定会出现 Success → Transferring 的真实逆转。','release',()=>{},null,'本地复查不是终态'); return;
      }
      const failed=state.ranks[r].poll==='Failed';
      emit(`cleanup-${r}${suffix}`,`PP${r} ${failed?'失败清理':'成功清理'}`,failed?'handle_inflight_transfer_failure 释放本地请求 KV 所有权并处理中止，随后归还 metadata、移出 inflight。其它级可能走成功分支，不能据此宣称端到端成功。':'release_kv_cache → finish(SUCCESS) → sender.clear，随后输出、归还 metadata 并移出 inflight。归还请求所有权不等于物理缓存全被清空，也不是 Decode 启动屏障。',failed?'failure':'release',()=>{Object.assign(state.ranks[r],{queue:failed?'aborted':'done',kv:false,metadata:false});});
    }
    for(let r=0;r<3;r++) cleanup(r);
    if(id==='repoll') {
      emit('repoll-ready','后续 PP1 复查已到终态','假设后续处理再次携带有效 release 信息，且本地 poll 观察为 Success，再进行清理。其它级已经完成，不必等 PP1 才归还本地资源。','release',()=>{state.ranks[1].poll='Success';}); cleanup(1,'-retry');
    }
    return {id,frames};
  }
  function sourceUrl(key) {const [file,line]=sources[key];return `https://github.com/sgl-project/sglang/blob/${commit}/python/sglang/srt/${file}#L${line}`;}
  return {commit,sources,scenarios,queues,create,consensus,terminalSet,sourceUrl};
});
