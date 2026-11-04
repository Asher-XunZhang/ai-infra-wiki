/* Fixed-source teaching timelines; seconds are reading time, not measured GPU latency. */
(function(root){
'use strict';
const M=typeof module==='object'&&module.exports?require('./model.js'):root.SchedulingModel;
const sources={...M.sources,...{"extend": ["python/sglang/srt/managers/schedule_batch.py", 2678, "def prepare_for_extend("], "decodePrepare": ["python/sglang/srt/managers/schedule_batch.py", 3470, "def prepare_for_decode("], "stageInput": ["python/sglang/srt/managers/schedule_batch.py", 2874, "self.prefill_input_ids_cpu = pinned_input_ids"], "loop": ["python/sglang/srt/managers/scheduler.py", 1907, "def event_loop_normal("], "choose": ["python/sglang/srt/managers/scheduler.py", 3602, "def get_next_batch_to_run("], "prefill": ["python/sglang/srt/managers/scheduler.py", 3797, "def _get_new_batch_prefill_raw("], "update": ["python/sglang/srt/managers/scheduler.py", 4141, "def update_running_batch("], "relay": ["python/sglang/srt/managers/scheduler.py", 4543, "def _relay_forward_payload("], "run": ["python/sglang/srt/managers/scheduler.py", 4285, "def run_batch("], "stash": ["python/sglang/srt/managers/scheduler.py", 3496, "def stash_chunked_request("], "filter": ["python/sglang/srt/managers/schedule_batch.py", 3564, "def filter_batch("], "resolve": ["python/sglang/srt/managers/overlap_utils.py", 87, "def resolve_forward_inputs"], "result": ["python/sglang/srt/managers/scheduler_components/batch_result_processor.py", 257, "def process_batch_result_prefill"], "cacheFinish": ["python/sglang/srt/mem_cache/radix_cache.py", 479, "def cache_finished_req"]}};
sources.evict=['python/sglang/srt/mem_cache/radix_cache.py',638,'def evict('];
function build(view='batch',options={}){
 if(!['batch','loop','admission','chunks','retract'].includes(view))throw Error('Unknown scheduling view');
 const opts={hit:3,policy:'fcfs',scenario:'compute',cap:4,recovery:'resume',...options},phases=[],marks=[];let duration=0,data;
 const initial={cut:false,packed:false,allocated:false,executed:false,relayed:false,decodePrepared:false,gathered:false,decoded:false,step:0,event:0,trimmed:false,reserved:false,written:false,done:false};
 function phase(id,title,seconds,source,summary,detail,patch={},u=.94){const p={id,title,start:duration,end:duration+seconds,duration:seconds,index:phases.length,source,summary,detail};phases.push(p);duration=p.end;marks.push({time:p.start+seconds*u,patch});return p;}
 if(view==='batch'){
  if(![0,3].includes(Number(opts.hit)))throw Error('Unsupported prefix length');
  const requests=[{id:'R1',tokens:[...'ABCDEF'],hit:2,prefix:[4,5],fresh:[30,31,32,33],output:'a',next:'c',nextSlot:34},{id:'R2',tokens:[...'UVWXY'],hit:Number(opts.hit),prefix:[20,21,22].slice(0,opts.hit),fresh:[40,41,42,43,44].slice(0,5-opts.hit),output:'b',next:'d',nextSlot:45}];
  data={requests,flat:requests.flatMap(r=>r.tokens.slice(r.hit).map((token,i)=>({token,request:r.id,position:r.hit+i,slot:r.fresh[i]})))};
  phase('requests','两条 Req，完整历史与本轮范围不同',2.2,'extend','R1 已有 2 个前缀位置；R2 的命中长度由上方条件决定','这里只读普通文本、page_size=1 的非投机路径；假设准入已完成，资源足够。token 字母与槽号是教学设定。');
  phase('slice','按 prefix_indices 截取本轮剩余输入',3,'extend','get_fill_ids()[len(prefix_indices):]：只选需要扩展的位置','这里不重复 Radix Tree 查找。命中的历史仍属于请求，只是不再进入本轮 input token 列表。',{cut:true});
  phase('pack','把不等长片段展平，并保留长度边界',3.6,'stageInput',`两条请求 → ${data.flat.length} 个待算 token，而不是 ${data.flat.length} 条请求`,'prepare_for_extend 生成 pinned CPU prefill_input_ids_cpu；prefix_lens、extend_lens 和 seq_lens 保留各请求的边界。此时 input_ids=None，实际 GPU 输入由执行前的 resolve 准备。',{packed:true});
  phase('allocate','为新位置分配 out_cache_loc',3,'extend','每个待算 token 都对应一个本轮写地址','alloc_for_extend 返回新写地址，并写入请求映射。分配地址尚未生成新 KV；这里只显示对齐关系，地址分配细节见第 4 节。',{allocated:true});
  phase('forward','执行 Extend：本轮输入产生 KV 与首个采样 token',3.4,'run','resolve 输入 → 模型 forward → R1 得到 a，R2 得到 b','只概括执行接口，不重复前一模块的 H2D 和算子演示。a、b 是新采样结果，此时还没有它们自己的 KV。',{executed:true});
  phase('relay','把采样结果交给下一轮，记录请求输出',2.8,'relay','GPU future_map 保存 a、b；CPU Req.output_ids 追加结果','普通非 Overlap、非投机路径也调用 _relay_forward_payload。run_batch 把 batch.input_ids 置空；结果处理器维护 CPU 请求历史，不能画成 CPU 列表直接成为下一轮 GPU tensor。',{relayed:true});
  phase('decode-prepare','Decode 准备：每条请求只新增一个位置',3.2,'decodePrepare','seq_lens 各加 1；为 a、b 分配两个写地址','prepare_for_decode 设 DECODE 并分配 token_per_req=1 的 out_cache_loc；input_ids 仍待 resolve。这里两条请求均未结束，且 next decode 容量充足。',{decodePrepared:true});
  phase('gather','从 future_map 取回两条请求的末尾 token',2.8,'resolve','本轮 GPU input_ids = [a, b]，不再包含整段 prompt','resolve_forward_inputs 根据请求行取得上一轮 GPU 采样结果。输入长度现在等于本轮 Decode 请求数；不是把上一次 Extend 展平缓冲整段再执行一遍。',{gathered:true});
  phase('decode-run','执行下一轮：a、b 得到 KV，再采样 c、d',3,'run','已有历史保持；本轮只新增两个 KV 位置','c、d 是新的采样输出，仍需下一轮才有对应 KV。本例在这一轮结果产生后停下，不表示请求已经达到停止条件。',{decoded:true,done:true});
 }else if(view==='loop'){
  data={cap:4};
  phase('ingest','接收请求：D 正在续写，L、S 等待 Prefill',2.3,'loop','L 有 6 个输入位置，S 有 2 个；每轮 chunk 额度为 4','选择 event_loop_normal，关闭 Overlap、混合 chunk、Prefill delayer、PP 和投机；内存与请求行充足。执行回合只表示决策顺序，不表示时延。');
  phase('choose-1','先尝试新 Prefill：L 本轮只能处理 4 个位置',3,'prefill','L[0:4] 进入本轮；S 等待；D 留在 running_batch','在本例条件下，get_next_batch_to_run 有可运行 Prefill 时先选它。is_mixed_chunk=false，D 不混入这个 Extend batch。',{step:1});
  phase('run-1','执行中间 chunk：累计 KV，但不发布输出 token',2.8,'result','L 已完成 4 / 6 个输入；中间 chunk 结果不追加到 output_ids','结果处理遇到 inflight_middle_chunks>0 时减少计数并跳过该请求的流式输出。不能把每个 chunk 都画成一次新 token 输出。',{step:2});
  phase('stash','下一轮先保存 chunk，并从待合并集合排除 L',3,'choose','缓存上块状态 → 排除 chunked_req；L 尚未进入 Decode','get_next_batch_to_run 对需要保存的 chunk 调用 stash_chunked_request；从 last_batch 过滤尚未完成 Prefill 的请求后，才允许合并到 running_batch。',{step:3});
  phase('choose-2','先续算 L，再用剩余 2 个额度接纳 S',3.5,'chunkFirst','本轮 Extend = L[4:6] + S[0:2]，正好 4 个输入位置','adder.add_chunked_req 先处理既有 L，随后才遍历 waiting_queue。只有真正存在剩余额度时 S 才能加入。',{step:4});
  phase('run-2','最后一块完成：L、S 各得到首个输出',2.8,'result','L 还需续写；S 的 max_new_tokens=1，因此已结束','末块请求才正常追加输出并检查结束。S 的请求资源按结束路径处理；L 的完整 Prefill 结果下一轮可进入 running_batch。',{step:5});
  phase('merge','过滤已结束 S，把完成 Prefill 的 L 合入运行批次',3.4,'filter','last_batch 过滤后只剩 L；running_batch = [D, L]','“Prefill 完成”和“请求结束”是两种状态。L 完成了 Prefill 但请求未结束；S 已满足停止条件，不能再合并去 Decode。',{step:6});
  phase('choose-decode','没有新 Prefill，转入 update_running_batch',2.7,'update','过滤完成成员 → 检查 Decode 容量 → 准备每条请求一个位置','本例内存检查通过，选出 D、L。连续批处理不等于 Prefill/Decode 固定轮流一次；决策来自本轮状态与分支条件。',{step:7});
  phase('run-decode','D、L 推进一次 Decode，并达到停止条件',2.6,'loop','两条请求各处理末尾 token，再得到各自的新输出','为了展示结束过滤，本例 D、L 的输出上限均为 2。结果处理标记结束并释放资源；running_batch 的成员列表在随后过滤时移除它们。',{step:8});
  phase('idle','下一次过滤为空，没有可运行 batch',1.8,'update','等待队列、chunked_req 与运行批次都为空','普通 event loop 进入 on_idle。图中历史执行卡片是已完成回合的记录，不表示还持有这些请求的 KV。',{step:9,done:true});
 }else if(view==='admission'){
  data={events:M.admission(opts.policy,opts.scenario)};
  data.events.forEach((e,i)=>{
   const r=M.requests.find(r=>r.id===e.current),need=r?r.input-r.hit+r.maxNew+1:0;
   const titles={waiting:'等待队列：先看输入与命中',sorted:'排序：决定先尝试谁',check:`临时保护 ${e.current} 的前缀，再检查预算`,commit:`把 ${e.current} 加入 can_run_list 并记账`,blocked:`${e.current} 本轮未被接纳`,stop:'返回 OTHER：停止追加，已接纳请求保留',ready:'形成计划：接下来才准备输入与执行'};
   let summary=e.kind==='check'?`${e.current}：待算 ${r.input-r.hit}；KV 检查量 ${need}；检查时 KV 余额 ${e.available}`:e.kind==='blocked'?e.reason==='kv'?`要求 ${need} < ${e.checkAvailable} 不成立；临时保护退出`:e.reason==='rows'?'请求行额度耗尽；停止尝试后续候选':'后续候选的输入不少于剩余输入额度；本轮停止追加':e.kind==='commit'?`已接纳 [${e.admitted.join(', ')}]；输入记账 ${e.usedInput}`:e.kind==='sorted'?(opts.policy==='lpm'?'LPM 把命中 4 个位置的 R2 排在前面':'FCFS 保持 R1 → R2 → R3 的等待顺序'):e.kind==='stop'?'OTHER 控制下一名候选，不撤销刚才的接纳':e.kind==='ready'?`本轮 [${e.admitted.join(', ')||'空'}]；继续等待 [${e.waiting.join(', ')||'空'}]`:'R1 待算 6；R2 命中 4、待算 2；R3 待算 2';
   const p=phase('admit-'+i,titles[e.kind],({waiting:1.6,sorted:2.7,check:3.3,commit:2.6,blocked:2.8,stop:1.8,ready:1.4})[e.kind],({waiting:'policy',sorted:opts.policy==='lpm'?'lpm':'policy',check:'pin',commit:'commit',blocked:e.reason==='kv'?'budget':e.reason==='rows'?'rows':'input',stop:'stop',ready:'commit'})[e.kind],summary,'普通 RadixCache、page_size=1、无 chunk 与特殊模型。KV 账本 = 可用 + 可驱逐 − 已记账 − 新保护；检查量 = 未命中输入 + 剩余输出上限 2 + 页余量 1，必须严格小于余额。无 chunk 时首条可超过输入额度；后续新输入 >= 剩余额度即停止。这里不是物理 KV 分配图。',{event:i,done:i===data.events.length-1},e.kind==='check'?.24:.94);p.event=e;p.previous=data.events[Math.max(0,i-1)];
  });
 }else if(view==='chunks'){
  data={events:M.chunks(Number(opts.cap))};
  data.events.forEach((e,i)=>{const plan=e.plan,p=phase('chunk-'+i,e.kind==='waiting'?'长请求 L 与短请求 S 同时等待':e.kind==='plan'?`第 ${e.round} 轮：先续算 L，再检查 S`:`第 ${e.round} 轮：执行选中的输入片段`,e.kind==='waiting'?1.8:e.kind==='plan'?(e.round===1?3.4:2.8):2.7,e.kind==='plan'?'chunk':'result',e.kind==='waiting'?'L 有 10 个位置；S 有 2 个；内存与请求行足够':e.kind==='plan'?`L 使用 ${plan.l}，剩余 ${opts.cap-plan.l}；S ${plan.s?'进入本轮':'继续等待'}`:`本轮完成 L 的 ${plan.l} 个、S 的 ${plan.s} 个位置`,'选择普通单池、page_size=1；候选 FCFS，L 在前。已有 chunked_req 先占用额度，随后尝试 S。输入位置只有在执行阶段完成后才变成已计算；中间 chunk 不发布新输出。额度改变的是分块与接纳边界，不是性能预测。',{event:i,done:i===data.events.length-1});p.event=e;p.previous=data.events[Math.max(0,i-1)];});
 }else{
  data={scenario:opts.recovery};M.retraction(opts.recovery);
  phase('pressure','下一次 Decode 需要两个新位置',2.2,'decode','R1 已有 8 个 KV，R2 已有 4 个；两条请求各再需要 1 个','普通文本，page_size=1，无 Overlap、投机、共享前缀或回撤 KV offload。R1 输入 7、已有输出 2；R2 输入 4、已有输出 a。末尾采样 token 还没有自己的 KV。');
  phase('evict-check','先驱逐可回收缓存，再判断容量',2.8,'check',opts.recovery==='cache'?'有 2 个可驱逐缓存位置，归还后正好满足需求':'没有可驱逐缓存，仍缺少位置；进入回撤路径','check_decode_capacity 先 evict_to_free_tokens，然后检查 available_size >= demand。容量检查失败不等同于已经发生 CUDA OOM。',{step:1});
  if(opts.recovery==='cache'){
   phase('keep','容量足够，保留两条运行请求',1.8,'check','R1、R2 都不需要回撤','这个分支在容量检查阶段解决压力，不能仍然播放释放 R2 的动作。',{step:2});
   phase('continue','两条请求各推进一次 Decode',2.6,'decodePrepare','新 KV 各增加 1；R2 的输出追加 b','本例在结果生成处停止，后续结束判断与资源释放另行处理。',{step:3,done:true});
  }else{
   phase('release','按 length 回撤 R2，释放其 KV 和请求行',3.5,'retract','R2 的 4 个位置归还；输出历史 a 保留','默认 length 顺序优先保留已有输出更多的 R1。release_req 释放执行资源并 reset_for_retract；普通文本历史保留，input_embeds 等分支有不同语义。',{step:2});
   phase('requeue','R2 尝试回到等待队列',2.8,opts.recovery==='full'?'full':'queue',opts.recovery==='full'?'等待队列已满 → R2 被终止':'回队成功 → R2 等待重新准入','本例无优先级抢占。队列上限可以使回队失败；另一条终止分支是回撤后最后一个请求仍放不下。',{step:3,done:opts.recovery==='full'});
   if(opts.recovery==='resume'){
    phase('remaining','R1 继续 Decode，使用一个空闲位置',2.4,'decodePrepare','R1 的 KV 从 8 增为 9；R2 等待','R1 处理之前的末尾 token 并采样第三个输出。本例设其输出上限为 3，于是达到结束条件。',{step:4});
    phase('finish','R1 结束：缓存保留 prompt 7 + 输出后缀 2',3.3,'cacheFinish','9 个位置成为可驱逐缓存；池内仍只有 3 个空闲位置','cache_finished_req 会在 prompt 边界再插入一次前缀，形成 7-token prompt 节点与 2-token 输出后缀；新采样的第三个输出尚无 KV，不在缓存中。',{step:5});
    phase('reclaim','R2 恢复需要 5 个位置，只需再回收 2 个',3,'evict','驱逐 R1 的输出后缀 2；prompt 缓存 7 保留','R2 历史为原输入 4 + 已有输出 a，共 5 个待重算位置。已有空闲 3，回收输出叶子的 2 个即可满足本轮分配；不会无条件清空 R1 全部缓存。',{trimmed:true});
    phase('reserve-recovery','重新准入并预留 5 个写位置',2.5,'extend','input = [u,v,w,x,a]；已分配地址，新 KV 尚未写好','本例后续准入额度充足、R2 未匹配缓存，重新经过 Extend。原来的 a 作为已有历史输入，而非再次新生成的输出。',{step:6,reserved:true});
    phase('recompute','Extend 重建五个历史位置的 KV',3,'run','a 保持原来的输出身份；重算恢复的是执行状态','这里显示无 KV offload 的普通文本恢复。历史 token 重算不应再次计入新增输出数。',{written:true});
    phase('result','追加新的输出 b，R2 可以继续生成',2.2,'result','R2 输出历史变为 [a,b]，a 没有重复发布','本例 R2 的输出上限为 3，因此得到第二个输出 b 后尚未结束；b 仍要等下一轮才有 KV。',{step:7,done:true});
   }
  }
 }
 return {view,options:opts,data,phases,marks,initial,duration};
}
function at(t,time){const now=Math.max(0,Math.min(t.duration,Number(time)||0)),p=t.phases.find(p=>now<p.end)||t.phases.at(-1),state={...t.initial};for(const m of t.marks)if(m.time<=now)Object.assign(state,m.patch);return {time:now,phase:p,u:(now-p.start)/p.duration,state,ended:now===t.duration};}
function recovery(t,state){const d={...M.retraction(t.options.recovery,state.step)};if(state.trimmed&&state.step===5){d.cache=7;d.free=5;}d.reserved=state.step===6&&!state.written?5:0;d.writtenR2=d.r2-d.reserved;return d;}
function loopState(step){return [
 {waiting:['L','S'],chunk:null,running:['D'],current:[],round:0,history:0,finished:[]},
 {waiting:['S'],chunk:'L',running:['D'],current:['L'],round:1,history:0,finished:[]},
 {waiting:['S'],chunk:'L',running:['D'],current:['L'],round:1,history:1,finished:[]},
 {waiting:['S'],chunk:'L',running:['D'],current:[],round:1,history:1,finished:[]},
 {waiting:[],chunk:null,running:['D'],current:['L','S'],round:2,history:1,finished:[]},
 {waiting:[],chunk:null,running:['D'],current:['L','S'],round:2,history:2,finished:['S']},
 {waiting:[],chunk:null,running:['D','L'],current:[],round:2,history:2,finished:['S']},
 {waiting:[],chunk:null,running:['D','L'],current:['D','L'],round:3,history:2,finished:['S']},
 {waiting:[],chunk:null,running:['D','L'],current:['D','L'],round:3,history:3,finished:['S','D','L']},
 {waiting:[],chunk:null,running:[],current:[],round:3,history:3,finished:['S','D','L']}
][step];}
const api={revision:M.revision,sources,build,at,recovery,loopState,sourceURL:key=>{const[f,l]=sources[key];return `https://github.com/sgl-project/sglang/blob/${M.revision}/${f}#L${l}`;}};
if(typeof module==='object'&&module.exports)module.exports=api;root.SchedulingFlow=api;
})(typeof globalThis==='object'?globalThis:this);
