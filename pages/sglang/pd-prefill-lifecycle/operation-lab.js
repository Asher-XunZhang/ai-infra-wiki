/* Component close-ups share the request timeline's immutable snapshot. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  let context=null,frame=0,timer=null,running=false,pinned=null,interval=1400;
  const sourceRoot='https://github.com/sgl-project/sglang/blob/279339f113b79af84f27fd3ac92d0a13bd3f4cbd/python/sglang/srt/';
  const escape=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function recipe(id){
    const node=context.model.nodes[id],r=node.rank??0,c=context.event.chunk??{index:1,start:0,end:context.model.size,size:context.model.size,total:12,last:context.model.size===12};
    const recipes={
      frontend:{source:'managers/tokenizer_manager.py',steps:[['接收输入','请求文本、rid、生成参数和 PD 配对信息进入 TokenizerManager。'],['得到 token IDs','调用所选模型的 tokenizer，得到完整序列。这里使用预设的 12 个 ID 演示，不声称它们是示例文本的真实编码。'],['携带配对信息投递','TokenizedGenerateReqInput 携带完整 input_ids 和 bootstrap_room 进入 PP0；分块发生在调度阶段。']]},
      bootstrap:{source:'disaggregation/prefill.py#L395',steps:[['建本地 Req','每一级为同一个 rid 建立独立 Req 与 sender，先进入 bootstrap_queue。'],['等待接收端','Decode 提供目的索引等握手信息；本级 sender 到 WaitingForInput 才具备准入候选资格。'],['求交并回流','good 名单逐级求交，bad 名单求并。末级结论回流，不能只凭本级就绪就准入。'],['完成准入','本级收到 good 共识后分配 metadata 槽、初始化 sender，将 R 放进 waiting_queue。']]},
      schedule:{source:'managers/schedule_policy.py#L950',steps:[['看完整序列',`R 始终保留 12 个 token。已计算前缀为 [0, ${c.start})；未计算部分从 token ${c.start} 开始。`],['按预算划界',`新增长度 = min(剩余 ${12-c.start}, 本轮预算 ${c.size}) = ${c.end-c.start}。设置 extend_range=[${c.start}, ${c.end})，后面的 token 暂不进入本轮。`],['装入本轮 batch',`本轮 input_ids 只装入 [${c.start}, ${c.end}) 的新 token。前缀 [0, ${c.start}) 通过 prefix_indices 和已有 KV 参与注意力，不重新作为新 token 计算。`],['映射 KV 写入位置',`为 PP${r} 的新增 token 准备 req_to_token 映射和 out_cache_loc。${c.last?'这是最后一块，后续处理有效首 token。':'保留同一个 chunked_req，后续选批继续切剩余序列。'}`]]},
      compute:{source:'models/llama.py',steps:[['准备本级输入',r===0?`取本块 ${c.end-c.start} 个 token IDs，通过 embedding 得到隐状态。`:`接收 PP${r-1} 的 hidden_states / residual，形状示意 [${c.end-c.start}, hidden_size]；不是上一级的 KV。`],['经过本级层',`顺序执行 L${r*10}–L${r*10+9}。本块新 token 可以读取同请求已有前缀 KV；本例从冷缓存开始，后续块复用的是前面自己算出的前缀。`],['写入本级 KV',`为本级每层的 token [${c.start}, ${c.end}) 写入 K/V；此后本级拥有 prompt 前缀 [0, ${c.end}) 的 KV。`],['产出下一段输入',r<2?`把本块 hidden_states / residual 交给 PP${r+1}，本级 KV 留在本级。`:c.last?'末级输出头得到 logits，并采样有效首 token t₀。prompt KV 不含 t₀ 自己的 KV。':'完成中间块计算，返回块结果；不向 Req.output_ids 追加有效首 token t₀。']]},
      transfer:{source:'disaggregation/prefill.py#L1311',steps:[['处理回流结果',c.last?'最终块：将 t₀ 追加到本地 Req.output_ids，保留缓存引用，并将 R 加入 inflight_queue。':'中间块：维护 chunked_req / inflight_middle_chunks；不追加有效 t₀，不当作最终交接完成。'],['圈定发送页',`取 start_send_idx 到 end_idx。非最终块向下对齐页边界：floor(${c.end}/2)×2=${c.end}；本例每块恰好包含两个完整页。`],['提交本级 KV',`向 Decode 的对应层位置发送 [${c.start}, ${c.end})。${c.last?'最终块附带首 token 等 metadata。':'本中间块不附带最终 metadata。'}发送区间按 token 定位，层范围仍是 PP${r} 的 L${r*10}–L${r*10+9}。`],['保留并轮询','send 返回只表示提交。源 KV 引用和 metadata 继续保留，直到传输终态、跨级共识和本地复查都满足。']]},
      release:{source:'disaggregation/prefill.py#L1025',steps:[['查本级 sender','轮询得到 Success / Failed 才能进入终态候选；本演示走成功路径。'],['汇合三级交集','三级候选取交集。任一级仍在传输，R 就不能进入共同释放名单。'],['回流后再复查','release_rids 回到本级，还必须再次检查本地 sender 的终态，才能清理。'],['归还请求占用','满足门槛后归还 metadata 槽、移出 inflight、清理 sender、释放请求的 KV 引用；可复用 KV 仍可由前缀缓存管理。']]},
      'decode-bootstrap':{source:'disaggregation/decode.py',steps:[['准备接收内存','接收端为配对请求准备对应层的 KV 接收位置。'],['交换位置索引','通过 bootstrap_room 关联这次传输，向 sender 提供目的位置与前缀信息。'],['等待各级写入','接收准备就绪不表示数据已到齐；后续仍需各级 KV 和最终交接信息。']]},
      'decode-kv':{source:'disaggregation/decode.py',steps:[['按层接收','PP0、PP1、PP2 分别写入自己负责层的目的位置。中间块可以逐步传输，提交不能直接当作接收完成。'],['收最终交接信息','最终块还携带首 token t₀ 等 metadata。完整 prompt KV 与生成起点属于不同数据。'],['满足继续生成条件','本图以各 sender 的 Success 表示成功观察点。Decode 的内部调度不在本页展开，它不需要等待 P 侧全部资源清理才继续生成。']]},
      'decode-run':{source:'disaggregation/decode.py',steps:[['已有 prompt KV','接收完成的 prompt KV 提供已有上下文，t₀ 是继续生成的起点。'],['处理 t₀','Decode 对 t₀ 做新的前向，产生 t₀ 自己的 KV。这个 KV 不是 Prefill 的 prompt 前向产生的。'],['得到下一 token','采样后续 token t₁，继续逐 token 生成。这一分镜只说明交接去向，不模拟 Decode 调度。']]},
    };
    const result={...recipes[node.kind??id],node,r,c};
    if(pinned)return result;
    const action=context.event.id.replace(/^chunk-\d+-/,''),middle=context.event.middle;
    const set=(steps,flow,type='flow')=>Object.assign(result,{steps,flow,visualType:type});
    if(/^req-/.test(action))set([['接收完整请求','接收同一 rid 的完整 token IDs 和 PD 配对信息。'],['建立本地对象',`在 PP${r} 建立自己的 Req 与 sender；不是把另一级的 Python 对象搬过来。`],['进入握手队列','加入 bootstrap_queue，等待接收端准备与后续共识。']],['完整 input_ids + rid R',`Req@PP${r} + sender@PP${r}`,'bootstrap_queue']);
    else if(/^boot-/.test(action))set([['读本级状态','轮询本地 sender，得到 WaitingForInput 的 good 候选和 bad 请求。'],['合并名单',r===0?'首级提出本轮候选名单。':'本级 good 与上游 good 求交；bad 名单求并。'],['送出候选',r===2?'末级得出共识，后续仍要回流才能本地准入。':'把合并后的名单传给下一级。']],[r===0?'本级 good={R}':'上游 good={R}',r===0?'提出候选':'∩ 本级 good={R}','good={R} / bad=∅']);
    else if(/^admit-/.test(action))set([['收到共识','接收沿 PP 环回流的 good / bad 名单。'],['占用 metadata 槽','good 名单包含 R，尝试本地 finalize_bootstrap，准备 metadata 与 sender。'],['准入 waiting_queue','资源足够且初始化完成，pending_bootstrap=False，将 R 移入 waiting_queue。']],['good 共识含 R','metadata 槽 + sender.init','waiting_queue']);
    else if(action==='cut')result.steps=result.steps.slice(0,3).map((v,i)=>i===2?['保留未处理尾部',`只设置本轮区间 [${c.start}, ${c.end})；完整序列及尾部 [${c.end}, 12) 仍属于同一个 R。下一个主线动作才进入各级 batch 准备。`]:v);
    else if(/^proxy-/.test(action))set([['等待本级依赖','上游前向已产出 hidden_states / residual；发送还需要满足 launch_event 的流依赖。'],['移动本块激活',`传递本块 ${c.end-c.start} 个 token 的中间激活，保持 batch 对应关系。`],['下一级得到输入','接收方继续自己的模型层；发送方 KV 不随着激活移动。']],[`PP${r} hidden_states / residual`,`形状 [${c.end-c.start}, hidden_size]`,`PP${r+1} 对应 batch`]);
    else if(/^output-/.test(action))set([['末级结果沿环到达','接收返回的 pp_outputs，仍需找到它对应的旧 batch。'],['匹配旧 batch 槽','用环形槽位中的 batch/result 对应关系取回这次计算的结果。'],['等待 CPU 可处理','等待需要的 D2H 事件；本地 Req 的结果更新发生在后续 result 动作。']],[middle?'中间块 output':'含 t₀ 的 pp_outputs','旧 batch B / slot','D2H 完成 → CPU 结果']);
    else if(/^result-/.test(action))set(middle?[['接收本块结果','取得本级对应中间块的计算结果。'],['维护中间块状态','推进 inflight_middle_chunks 等请求状态，不追加有效首 token。'],['继续保留请求','保留 chunked_req 与本级 KV，后续继续计算或提交已完成区间。']]:[['取得有效 t₀','使用末级回流的采样结果。'],['追加本地 output_ids','把 t₀ 追加到本级 Req 的输出，维护缓存引用。'],['进入 inflight','把 R 加入本级 inflight_queue，后续动作才提交本级最终 KV。']],middle?['中间块结果','不追加有效 t₀','保留 R / chunked_req']:['回流 t₀','Req.output_ids.append(t₀)','inflight_queue']);
    else if(/^kv-/.test(action))result.steps=result.steps.slice(1);
    else if(/^poll-/.test(action))set([['读取后端进度','轮询这个 rank 的 sender，读取当前传输状态。'],['判断是否到终态',context.state.ranks[r].terminal?'本级观察到 Success，可贡献终态候选 R。':'本级仍未到终态，不能贡献释放候选 R。'],['保留请求占用','单次 poll 不是释放操作。资源仍需跨级共识、名单回流与本地复查。']],['sender.poll()',context.state.ranks[r].terminal?'Success → {R}':'仍在传输 → ∅','KV / metadata 继续保留'],'poll');
    else if(/^terminal-/.test(action))result.steps=[['读取候选集合','读取上游名单和本级 sender 的终态集合。'],['逐级求交','Success / Failed 都属于终态；本成功示例只包含 Success。任何空集都会从共同名单中去掉 R。'],['送出交集',context.state.consensus?'末级已经形成最终交集，随后才发出 release 名单。':'将候选交给下一阶段；当前动作不执行资源清理。']];
    else if(/^release-/.test(action))result.steps=[['接收 release_rids','收到回流的跨级共同终态名单。'],['本地再 poll','确认 R 在名单中，且本级 sender 此刻仍是终态。'],['获得清理条件','后续 finish 动作才执行资源归还；收到名单不能跳过本地复查。']];
    else if(/^finish-/.test(action)){result.steps=[['持有清理许可','R 已在 release 名单中，本地终态复查通过。'],['归还本地占用','释放 KV 请求引用、归还 metadata 槽并清理 sender；缓存内容不一定被物理擦除。'],['移出 inflight',r===0?'R 退出本级队列，有效 IPC 输出 rank 发出 P 侧完成结果。':'R 退出本级队列；其他级的清理由它们自己完成。']];result.visualType='release';}
    return result;
  }
  function currentState(){const info=recipe(pinned??context.event.active[0]);return !pinned&&frame<info.steps.length-1?(context.before??context.state):context.state;}
  function payload(){
    const id=pinned??context.event.active[0],node=context.model.nodes[id],c=context.event.chunk;
    if(id==='frontend')return ['文本 / 参数','IDs × 12','input_ids + room'][Math.min(frame,2)];
    if(node.kind==='schedule')return `${c?.end-c?.start||context.model.size} 个新 token`;
    if(node.kind==='compute')return node.rank===2&&frame>=3?(c?.last?'t₀':'块结果 B'):node.rank===0&&frame===0?`IDs × ${c?.end-c?.start||context.model.size}`:`h[${c?.end-c?.start||context.model.size}, d]`;
    if(node.kind==='release')return context.event.id==='terminal-1-wait'&&frame>0?'∅':context.state.ranks[node.rank].released&&frame>0?'归还资源':'{R}';
    if(node.kind==='transfer')return /(?:^|-)kv-\d$/.test(context.event.id)?`KV [${c.start}, ${c.end})`:context.event.middle?'块结果 B':context.event.kind==='output'?'t₀':'rid R';
    if(id==='decode-kv')return '各层 KV / metadata';
    if(id==='decode-run')return frame===0?'t₀':frame===1?'KV(t₀)':'t₁';
    return /^boot-|^admit-/.test(context.event.id)?'good {R}':'rid R / room';
  }
  function chain(items){return '<div class="lab-chain">'+items.map((s,i)=>`<div class="lab-flow-box ${i===Math.min(frame,items.length-1)?'is-current':i>frame?'is-next':''}"><span>${s}</span><div class="lab-data-slot">${i===Math.min(frame,items.length-1)?`<code class="lab-data-chip">${escape(payload())}</code>`:''}</div></div>`).join('<span class="lab-arrow" aria-hidden="true">→</span>')+'</div>';}
  function strip(c,kind='full',rank=0){
    const state=currentState().ranks[rank];
    return `<div class="token-scroll"><div class="token-strip ${kind==='batch'?'is-batch':''}">`+context.model.tokenIds.map((id,i)=>{
      let style=i<c.start?'is-prefix':i<c.end?'is-selected':'is-future';
      if(kind==='full'&&frame===0)style=i<c.start?'is-prefix':'is-future';
      if(kind==='cache')style=i<state.cacheEnd?'is-stored':'is-future';
      if(kind==='send')style=i<state.sendEnd?'is-submitted':'is-future';
      if(kind==='batch'&&(i<c.start||i>=c.end))style+=' is-outside';
      return `<div class="token-cell ${style} ${i===c.start&&frame>0?'cut-start':''} ${i===c.end-1&&frame>0?'cut-end':''}" style="--token-order:${i}"><small>${i}</small><strong>${id}</strong></div>`;
    }).join('')+'</div></div>';
  }
  function memory(r,c){const s=currentState().ranks[r];return `<div class="lab-row-label">PP${r} 本级 KV · ${frame===recipe(pinned??context.event.active[0]).steps.length-1?'处理后':'输入时'}：已计算 ${s.cacheEnd}/12，已提交 ${s.sendEnd}/12</div>${strip(c,'cache',r)}<div class="lab-row-label">发送游标 start_send_idx = ${s.sendEnd}（已提交 ≠ 已成功）</div>${strip(c,'send',r)}`;}
  function visual(info){
    const {node,r,c}=info,kind=node.kind??(pinned??context.event.active[0]),snapshot=currentState(),state=snapshot.ranks[r];
    if(info.visualType==='flow'||info.visualType==='poll')return chain(info.flow)+(info.visualType==='poll'?memory(r,c):'');
    if(kind==='schedule')return `<div class="lab-token-key"><span>灰：尚未计算</span><span>绿：已有前缀</span><span>蓝：本轮新 token</span><b>第 ${c.index} 块 · [${c.start}, ${c.end})</b></div><div class="lab-row-label">完整 origin_input_ids（切块不删除后面的 token）</div>${strip(c)}${frame>=2&&(!context.event.id.endsWith('-cut')||pinned)?`<div class="lab-row-label">本轮 batch 的 input_ids ↓</div>${strip(c,'batch')}`:''}${frame===3?chain([`token ${c.start}…${c.end-1}`,`req_to_token[R, ${c.start}:${c.end}]`,'out_cache_loc → 本级 KV 槽']):''}`;
    if(kind==='frontend')return frame===0?chain(['输入文本 + 参数','TokenizerManager', 'rid / bootstrap_room']):`${strip({...c,start:0,end:12})}${chain(['完整 input_ids[0:12]','TokenizedGenerateReqInput','投递到 PP0'])}`;
    if(kind==='bootstrap')return chain(frame===0?[`rid = R`,`Req@PP${r}`,`sender@PP${r}`]:frame===1?['Decode 接收索引','bootstrap_room 配对','WaitingForInput']:frame===2?['PP0 good {R}','∩ PP1 good','∩ PP2 good → 回流']:['bootstrap_queue','metadata 槽 + sender.init','waiting_queue'])+`<p class="lab-snapshot">当前 PP${r}：${escape(context.model.queueLabels[state.queue])} · sender ${escape(context.model.senderLabels[state.sender])} · metadata ${state.metadata?'占用':'未占用'}</p>`;
    if(kind==='compute')return (frame<2?chain([r===0?'本块 token IDs':`PP${r-1} 激活`,`[${c.end-c.start}, hidden_size]`,`PP${r} 本级层`]):memory(r,c))+`<div class="layer-stack">${Array.from({length:10},(_,i)=>`<span class="${frame>=1?'layer-working':''}" style="--token-order:${i}">L${r*10+i}<small>K / V</small></span>`).join('')}</div>`+(frame===3?chain([r<2?'本块新激活':c.last?'logits → t₀':'中间块结果',r<2?`发给 PP${r+1}`:'结果回流',`本级 KV 留在 PP${r}`]):'');
    if(kind==='transfer'){
      const stage=frame+(!pinned&&/(?:^|-)kv-\d$/.test(context.event.id)?1:0);
      const pages=stage>0?`<div class="lab-row-label">按页组织本轮 token 的 K/V（逻辑页号示意，真实物理索引由映射得到）</div><div class="transfer-pages">${Array.from({length:(c.end-c.start)/2},(_,i)=>`<div class="page-packet"><strong>逻辑页 ${c.start/2+i}</strong><span>token ${c.start+i*2} · K / V</span><span>token ${c.start+i*2+1} · K / V</span></div>`).join('')}</div>`:'';
      return (stage===0?chain([c.last?'有效 t₀':'中间块结果',`本地 Req@PP${r}`,c.last?'inflight_queue':'保留 chunked_req']):stage===1?chain([`[${c.start}, ${c.end})`,`页 ${c.start/2}…${c.end/2-1}`,c.last?'最终 metadata':'非最终块 / 整页发送']):chain([`PP${r} 源 KV`,`提交 [${c.start}, ${c.end})`,'Decode 对应层']))+pages+memory(r,c);
    }
    if(kind==='release'){
      const all=snapshot.ranks.every(s=>s.terminal),permit=state.permit;
      return `<div class="consensus-gates">${snapshot.ranks.map((s,i)=>`<div class="${s.terminal?'is-ready':''}"><strong>PP${i}</strong><span>${s.terminal?'Success → {R}':'未到终态 → ∅'}</span></div>`).join('')}</div>`+chain([`交集 ${all?'{R}':'∅'}`,`本级 release 名单：${permit?'含 R':'尚未收到 R'}`,state.released?'请求占用已释放':permit?'复查终态后可清理':'继续保留请求占用'])+`<div class="resource-slots"><span class="${state.metadata?'is-held':''}">metadata 槽：${state.metadata?'占用':'未占用'}</span><span class="${state.holdsKv?'is-held':''}">KV 请求引用：${state.holdsKv?'保留':'未持有'}</span><span>inflight：${state.queue==='inflight'?'在队列中':'不在队列中'}</span></div>`;
    }
    if(kind==='decode-bootstrap')return chain(frame===0?['接收请求 R','分配目的 KV 位置','按层建立映射']:frame===1?['bootstrap_room','目的页索引','各 PP sender']:['接收准备完成','数据仍需传输','等待最终交接']);
    if(kind==='decode-kv')return `<div class="receive-progress">${context.state.ranks.map((s,i)=>`<div><strong>PP${i} · L${i*10}–L${i*10+9}</strong><progress max="12" value="${s.sendEnd}"></progress><span>提交 ${s.sendEnd}/12 · ${s.terminal?'已观察到 Success':'仍待终态确认'}</span></div>`).join('')}</div>`+chain(['各层 prompt KV',context.state.ranks.some(s=>s.token)?'最终 t₀ 已在 P 侧处理':'最终 t₀ 尚未处理',context.state.ranks.every(s=>s.terminal)?'本例已满足交接数据条件':'等待剩余交接数据']);
    return chain(frame===0?['prompt KV','首 token t₀','Decode 输入']:frame===1?['t₀ 前向','写入 t₀ 的 KV','输出 logits']:['采样 t₁','追加到输出','继续 Decode']);
  }
  function stop(){clearTimeout(timer);timer=null;running=false;$('lab-play').textContent='▶ 演示本组件';$('lab-play').setAttribute('aria-pressed','false');$('operation-lab').classList.remove('is-demonstrating');}
  function render(){
    if(!context)return;const id=pinned??context.event.active[0],info=recipe(id);frame=Math.min(frame,info.steps.length-1);
    $('lab-title').textContent=info.node.title;
    $('lab-context').textContent=`${pinned?'已选设备':'跟随主线'} · ${context.event.title}。以下分镜拆解组件机制；标注“主线快照”的数值取自当前动作。`;
    $('lab-stages').replaceChildren();info.steps.forEach(([name],i)=>{const b=document.createElement('button');b.type='button';b.textContent=`${i+1}. ${name}`;b.setAttribute('aria-pressed',String(i===frame));b.addEventListener('click',()=>{stop();frame=i;render();});$('lab-stages').append(b);});
    $('lab-visual').innerHTML=visual(info);$('lab-explanation').textContent=info.steps[frame][1];
    $('lab-source').href=sourceRoot+info.source;
    $('lab-frame-status').textContent=`内部 ${frame+1} / ${info.steps.length} 帧`;
    $('lab-prev').disabled=frame===0;$('lab-next').disabled=frame===info.steps.length-1;
  }
  function tick(){timer=setTimeout(()=>{if(!running)return;const info=recipe(pinned??context.event.active[0]);if(frame===info.steps.length-1){stop();return;}frame++;render();tick();},interval);}
  function play(duration){if(running){stop();return;}interval=duration?duration/(recipe(pinned??context.event.active[0]).steps.length+.5):1400;frame=0;running=true;$('operation-lab').classList.add('is-demonstrating');$('lab-play').textContent='Ⅱ 暂停内部演示';$('lab-play').setAttribute('aria-pressed','true');render();tick();}
  $('lab-play').addEventListener('click',()=>{if(running){stop();return;}window.dispatchEvent(new Event('prefill-lab-play'));play();});
  $('lab-prev').addEventListener('click',()=>{stop();frame=Math.max(0,frame-1);render();});
  $('lab-next').addEventListener('click',()=>{stop();frame++;render();});
  $('lab-follow').addEventListener('click',()=>{stop();pinned=null;frame=0;render();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});window.addEventListener('pagehide',stop);
  window.PrefillOperationLab={update(value){stop();context=value;frame=0;render();},inspect(id){stop();pinned=id;frame=0;render();$('operation-lab').scrollIntoView({block:'start',behavior:'smooth'});},follow(){pinned=null;},stop,play};
})();
