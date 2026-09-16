/* Fixed-source causal teaching model, not a GPU/network timing simulation. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.PrefillLifecycle=api;})(globalThis,()=>{
  'use strict';
  const copy=x=>JSON.parse(JSON.stringify(x)),unique=a=>[...new Set(a)],intersection=(a,b)=>a.filter(x=>b.includes(x));
  const terminal=s=>s==='Success'||s==='Failed';
  const sources={intake:'managers/tokenizer_manager.py#L1414',request:'managers/scheduler.py#L2740',bootstrap:'managers/scheduler_pp_mixin.py#L593',admission:'disaggregation/prefill.py#L449',budget:'managers/scheduler.py#L3857',cut:'managers/schedule_policy.py#L1278',continuation:'managers/schedule_policy.py#L950',pack:'managers/schedule_batch.py#L2678',forward:'models/llama.py#L426',proxy:'managers/scheduler_pp_mixin.py#L257',output:'managers/scheduler_pp_mixin.py#L276',result:'disaggregation/prefill.py#L797',send:'disaggregation/prefill.py#L1311',consensus:'managers/scheduler_pp_mixin.py#L635',release:'disaggregation/prefill.py#L991',failure:'disaggregation/prefill.py#L1082',abort:'disaggregation/prefill.py#L1150',decode:'disaggregation/decode.py'};
  const nodes={frontend:{x:15,y:166,w:125,h:58,title:'Tokenizer',role:'准备完整 token IDs，保留 rid 和 PD 配对信息。'}};
  const kinds=['bootstrap','schedule','compute','transfer','release'],titles=['请求 / 握手','切块 / 组 batch','模型前向','结果 / KV 发送','释放'];
  const roles=['本地 Req / sender；good 求交、bad 求并，再处理返回结论。','先续算 chunked_req，再按共享 token 预算和请求容量选批、分配 KV。','执行本级模型层，读取前缀、写新增 KV，传递激活。','中间块保留请求；最终块追加 t₀ 并进入 inflight；各级独立提交 KV。','Success 或 Failed 都属于终态，求交后还需本地复查和分支清理。'];
  for(let r=0;r<3;r++){const x=180+r*280,g=[{x,y:45,w:220,h:42},{x,y:119,w:220,h:42},{x,y:192,w:220,h:62},{x,y:312,w:135,h:56},{x:x+150,y:312,w:70,h:56}];kinds.forEach((kind,i)=>nodes[`p${r}-${kind}`]={...g[i],rank:r,kind,title:`PP${r} · ${titles[i]}`,role:roles[i]});}
  nodes['decode-bootstrap']={x:1050,y:45,w:210,h:42,title:'Decode · 接收准备',role:'准备目的内存与索引，参与握手。'};
  nodes['decode-kv']={x:1050,y:312,w:210,h:56,title:'Decode · KV 接收',role:'按层接收各 PP 的 KV 和最终 metadata；提交不代表成功。'};
  nodes['decode-run']={x:1050,y:403,w:210,h:42,title:'Decode · 继续生成',role:'成功交接后继续生成；P 的全部清理不是 D 启动屏障。'};
  const defaults={requests:1,inputLength:12,chunkSize:4,batchSize:2,pageSize:2,fault:'none',failed:[[],['R0'],[]],waitRank:2,waitRid:'R0',detail:'first'};
  const presets={chunks:{...defaults,label:'一条请求 · 分三块'},single:{...defaults,chunkSize:0,label:'一条请求 · 一次完成'},batching:{...defaults,requests:3,inputLength:6,chunkSize:8,label:'多请求 · 共享预算组批'},bootstrap:{...defaults,requests:3,inputLength:4,chunkSize:8,fault:'bootstrap_fail',failed:[['R0'],['R1'],[]],label:'不同 PP 的部分握手失败'},transfer:{...defaults,requests:3,inputLength:4,chunkSize:8,fault:'transfer_fail',failed:[[],['R0'],['R1']],label:'Success / Failed 混合终态'},waiting:{...defaults,requests:2,fault:'bootstrap_wait',waitRank:1,label:'握手未就绪 · 继续等待'},repoll:{...defaults,fault:'repoll_wait',waitRank:1,label:'release 已到 · 本地仍未到终态'},abort:{...defaults,requests:2,fault:'bootstrap_abort',label:'本地取消 · 加入 bad 并集'}};
  function normalize(raw={}){
    const c={...copy(defaults),...raw};
    for(const [k,min,max] of [['requests',1,4],['inputLength',1,48],['chunkSize',0,96],['batchSize',1,4],['pageSize',1,4],['waitRank',0,2]]){c[k]=Number(c[k]);if(!Number.isInteger(c[k])||c[k]<min||c[k]>max)throw new Error(`${k} 必须是 ${min}–${max} 的整数。`);}
    if(![1,2,4].includes(c.pageSize))throw new Error('页大小请选择 1、2 或 4。');
    if(c.chunkSize>0&&c.chunkSize<c.pageSize)throw new Error('chunk 预算至少要容纳一个 KV 页，或设为 0 关闭切块。');
    if(c.chunkSize%c.pageSize!==0)throw new Error('chunk-size 必须能被 KV 页大小整除（源码配置约束）；设为 0 可关闭切块。');
    if(!['none','bootstrap_fail','bootstrap_abort','bootstrap_wait','transfer_fail','repoll_wait'].includes(c.fault))throw new Error('请选择支持的异常场景。');
    if(!['first','all'].includes(c.detail))throw new Error('请选择有效的展开方式。');
    const ids=Array.from({length:c.requests},(_,i)=>`R${i}`);c.failed=Array.from({length:3},(_,r)=>unique((c.failed?.[r]??[]).filter(id=>ids.includes(id))));c.waitRid=ids.includes(c.waitRid)?c.waitRid:'R0';
    if(['bootstrap_fail','bootstrap_abort','transfer_fail'].includes(c.fault)&&!c.failed.some(a=>a.length))throw new Error('请在 PP 矩阵中至少选择一条失败或取消的请求。');return c;
  }
  function bootstrapConsensus(good,bad,aborted=[[],[],[]]){const localGood=good.map((a,r)=>a.filter(id=>!aborted[r].includes(id))),localBad=bad.map((a,r)=>unique([...a,...aborted[r]]));return {good:localGood.reduce(intersection),bad:unique(localBad.flat()),localGood,localBad};}
  function terminalConsensus(rows){const local=rows.map(row=>Object.keys(row).filter(id=>terminal(row[id])));return {local,intersection:local.reduce(intersection)};}
  function planBatch(c,requests,progress){
    const candidates=requests.filter(q=>q.outcome==='active'&&progress[q.rid]<q.length).sort((a,b)=>Number(progress[b.rid]>0)-Number(progress[a.rid]>0)||a.index-b.index);
    let remaining=c.chunkSize||Infinity;const segments=[];
    for(const q of candidates){if(segments.length>=c.batchSize)break;const start=progress[q.rid],left=q.length-start,capacity=Number.isFinite(remaining)?Math.floor(remaining/c.pageSize)*c.pageSize:left,length=Math.min(left,capacity);if(length<=0)break;const charge=Math.ceil(length/c.pageSize)*c.pageSize;segments.push({rid:q.rid,start,end:start+length,last:start+length===q.length,charge});remaining-=charge;if(start+length<q.length)break;}
    return {segments,charge:segments.reduce((s,x)=>s+x.charge,0),tokenCount:segments.reduce((s,x)=>s+x.end-x.start,0),remaining:Number.isFinite(remaining)?remaining:null};
  }
  function createScenario(raw={}){
    const config=normalize(raw),requests=Array.from({length:config.requests},(_,index)=>({rid:`R${index}`,index,length:config.inputLength,tokenIds:Array.from({length:config.inputLength},(_,i)=>100+index*1000+i),outcome:'active'}));
    const state={requests:copy(requests),ranks:Array.from({length:3},()=>({requests:Object.fromEntries(requests.map(q=>[q.rid,{queue:'none',sender:'None',metadata:false,holdsKv:false,selectedEnd:0,cacheEnd:0,sendEnd:0,token:false,permit:false,released:false,reason:null,chunkNo:0}]))})),batch:null,bootstrap:null,terminal:null,release:[],visited:[]};
    const events=[],frames=[];let previous=copy(state),group=null;
    const route=(from,to,kind='control',path='direct')=>({from,to,kind,path});
    function emit(key,title,type,active,description,mutate=()=>{},extra={}){
      mutate();
      for(const q of state.requests)if(q.outcome==='active'&&state.ranks.every(r=>r.requests[q.rid].released))q.outcome=state.ranks.some(r=>r.requests[q.rid].reason==='FINISH_ABORT')?(state.ranks[0].requests[q.rid].cacheEnd?'transfer-failed':'bootstrap-failed'):'success';
      state.visited=unique([...state.visited,...active]);const after=copy(state),f={key,title,type,active,description,phase:extra.phase??3,kind:extra.kind??'control',routes:extra.routes??[],source:sources[extra.source??type]??sources.budget,work:copy(extra.work??{}),before:previous,after};previous=after;frames.push(f);if(group)group.frames.push(f);else events.push({key,title,frames:[f],compressed:false});return f;
    }
    function bootstrap(round){
      const rows=state.ranks.map(s=>s.requests),eligible=state.requests.filter(q=>q.outcome==='active'&&rows[0][q.rid].queue==='bootstrap').map(q=>q.rid);
      const good=rows.map(row=>eligible.filter(id=>row[id].sender==='WaitingForInput')),bad=rows.map(row=>eligible.filter(id=>row[id].sender==='Failed')),aborted=rows.map((row,r)=>eligible.filter(id=>config.fault==='bootstrap_abort'&&config.failed[r].includes(id)));
      const result=bootstrapConsensus(good,bad,aborted);let g=[],b=[];
      for(let r=0;r<3;r++){const upstream={good:[...g],bad:[...b]};g=r?intersection(g,result.localGood[r]):result.localGood[r];b=unique([...b,...result.localBad[r]]);emit(`bootstrap-${round}-${r}`,`PP${r}：good ${r?'求交':'候选'} / bad ${r?'求并':'候选'}`,'bootstrap',[`p${r}-bootstrap`],'good 是共同就绪集合；bad 是任一级失败或取消的并集。未出现在两者中的请求继续等待。',()=>{state.bootstrap={...result,upstream,through:r,good:[...g],bad:[...b]};},{phase:2,work:{rank:r},routes:r?[route(`p${r-1}-bootstrap`,`p${r}-bootstrap`)]:[]});}
      for(let r=0;r<3;r++)emit(`admit-${round}-${r}`,`PP${r} 处理返回的准入 / 失败名单`,'admission',[`p${r}-bootstrap`,`p${r}-schedule`],'共同 good 完成 metadata 分配和 sender 初始化；bad 走 bootstrap 失败清理；其余请求留在 bootstrap_queue。',()=>{for(const id of eligible){const q=state.ranks[r].requests[id];if(result.bad.includes(id))Object.assign(q,{queue:'done',released:true,reason:'FINISH_ABORT'});else if(result.good.includes(id))Object.assign(q,{queue:'waiting',metadata:true});}if(r===2)for(const q of state.requests)if(result.bad.includes(q.rid))q.outcome='bootstrap-failed';},{phase:2,work:{rank:r},routes:[route(r===0?'p2-bootstrap':`p${r-1}-bootstrap`,`p${r}-bootstrap`,'control',r===0?'return':'direct')]});
    }
    emit('intake','文本输入 → 完整 token IDs','intake',['frontend'],'每条请求先得到完整 token 序列。ID 为教学预设；切块不会再次 tokenize，也不会删除原序列。',()=>{},{phase:0});
    for(let r=0;r<3;r++)emit(`request-${r}`,`PP${r} 创建本地 Req / sender`,'intake',[`p${r}-bootstrap`],'同一个 rid 在三个 PP 进程各有本地对象，请求信息逐级转发。',()=>{for(const q of Object.values(state.ranks[r].requests))Object.assign(q,{queue:'bootstrap',sender:'Bootstrapping'});},{phase:1,source:'request',work:{rank:r},routes:[route(r===0?'frontend':`p${r-1}-bootstrap`,`p${r}-bootstrap`,'control',r===0?'entry':'direct')]});
    emit('bootstrap-poll','观察三个 PP 的握手状态','bootstrap',['decode-bootstrap','p0-bootstrap','p1-bootstrap','p2-bootstrap'],'比较各 PP 的本地状态。取消可加入 bad 并集，即使 sender.abort 没有将 poll 变成 Failed。',()=>{for(let r=0;r<3;r++)for(const req of requests)state.ranks[r].requests[req.rid].sender=config.fault==='bootstrap_fail'&&config.failed[r].includes(req.rid)?'Failed':config.fault==='bootstrap_wait'&&r===config.waitRank&&req.rid===config.waitRid?'Bootstrapping':'WaitingForInput';},{phase:2,work:{poll:true},routes:[0,1,2].map(r=>route('decode-bootstrap',`p${r}-bootstrap`,'control','handshake'))});
    bootstrap(1);
    if(config.fault==='bootstrap_wait'){
      emit('bootstrap-wait','未就绪的请求继续等待','bootstrap',[`p${config.waitRank}-bootstrap`],'共同 good 和 bad 均不含该请求：未就绪不是失败。实际可以先运行其他已准入请求；这里先展开后续握手。',()=>{},{phase:2,work:{wait:true}});
      emit('bootstrap-ready','后续轮询：握手就绪后重新求共识','bootstrap',[`p${config.waitRank}-bootstrap`],'外部握手条件已改变，再次计算共同名单。',()=>{state.ranks[config.waitRank].requests[config.waitRid].sender='WaitingForInput';},{phase:2});bootstrap(2);
    }
    function release(ids,tag){
      if(!ids.length)return;
      for(let r=0;r<3;r++){
        const delayed=config.fault==='repoll_wait'&&r===config.waitRank&&ids.includes(config.waitRid);
        emit(`${tag}-release-${r}`,`PP${r} 收到 release 名单并再次 poll`,'release',[`p${r}-release`],delayed?'名单来自先前采样，本级当前 poll 仍是瞬态：保留请求，不能直接释放。':'R 在名单中且本地仍是 Success / Failed，才允许对应清理。',()=>{state.release=[...ids];for(const id of ids){const q=state.ranks[r].requests[id];if(delayed&&id===config.waitRid)q.sender='Transferring';q.permit=terminal(q.sender);}},{phase:10,work:{rank:r,ids},routes:[route(r===0?'p2-release':`p${r-1}-release`,`p${r}-release`,'control',r===0?'return':'direct')]});
        const finish=()=>{for(const id of ids){const q=state.ranks[r].requests[id];if(!q.permit||!terminal(q.sender))continue;const failed=q.sender==='Failed';Object.assign(q,{queue:'done',sender:failed?'Failed':'Cleared',metadata:false,holdsKv:false,released:true,reason:failed?'FINISH_ABORT':'FINISH_LENGTH(0)'});}};
        emit(`${tag}-finish-${r}`,`PP${r}：按本地状态归还资源`,'cleanup',[`p${r}-release`],'Success 正常完成；Failed 调用 handle_inflight_transfer_failure 并中止。瞬态请求留在 undone，不能一起清理。',finish,{phase:11,source:'release',work:{rank:r,ids}});
        if(delayed)emit(`${tag}-retry-${r}`,`PP${r} 后续复查到终态，再清理`,'cleanup',[`p${r}-release`],'示意下一次携带有效 release 信息的处理调用：本地 poll 终态确认后，清理先前保留的请求。',()=>{const q=state.ranks[r].requests[config.waitRid];q.sender='Success';q.permit=true;finish();},{phase:11,source:'release',work:{rank:r,ids}});
      }
    }
    function terminalRound(ids,tag){
      let acc=[];const rows=state.ranks.map(rank=>Object.fromEntries(ids.map(id=>[id,rank.requests[id].sender]))),sets=terminalConsensus(rows);
      for(let r=0;r<3;r++){const upstream=[...acc];acc=r?intersection(acc,sets.local[r]):sets.local[r];emit(`${tag}-terminal-${r}`,`PP${r} 合并终态集合：Success ∪ Failed`,'consensus',[`p${r}-release`],'求交的是“已到终态”的请求，不是成功请求。Failed 也要统一收尾；任何仍在传输的级会阻止该请求进入共同名单。',()=>{state.terminal={rows,local:sets.local,upstream,through:r,intersection:[...acc]};},{phase:9,work:{rank:r,ids},routes:r?[route(`p${r-1}-release`,`p${r}-release`)]:[]});}return sets.intersection;
    }
    let batchNumber=0;
    while(state.requests.some(q=>q.outcome==='active'&&state.ranks[0].requests[q.rid].cacheEnd<q.length)){
      if(++batchNumber>192)throw new Error('教学场景超过批次数上限。');
      const progress=Object.fromEntries(requests.map(q=>[q.rid,state.ranks[0].requests[q.rid].cacheEnd])),plan=planBatch(config,state.requests,progress);if(!plan.segments.length)throw new Error('预算无法容纳请求，请增加 chunk 预算。');
      const batch={...plan,id:`B${batchNumber}`,number:batchNumber,capacity:config.batchSize,budget:config.chunkSize||null};batch.segments.forEach(s=>s.chunkIndex=state.ranks[0].requests[s.rid].chunkNo+1);
      if(batchNumber>1&&config.detail==='first'){group={key:`batch-${batchNumber}-summary`,title:`${batch.id} · ${batch.segments.map(s=>`${s.rid} 第 ${s.chunkIndex} 块`).join(' + ')} · 完整流程`,compressed:true,frames:[]};events.push(group);}
      const extra={work:{batch},phase:3};
      emit(`${batch.id}-budget`,`${batch.id}：共享 token 预算与 batch 容量`,'budget',['p0-schedule'],`本轮最多 ${config.batchSize} 条请求，共享 ${config.chunkSize||'不限制'} token 预算。先续算 chunked_req，再选等待请求；CUDA 普通路径按页向上计费。`,()=>{state.batch=copy(batch);},extra);
      emit(`${batch.id}-cut`,`${batch.id}：${config.chunkSize?'切出各请求本轮区间':'关闭切块，选择完整剩余序列'}`,'cut',['p0-schedule'],batch.segments.map(s=>`${s.rid}：前缀 [0, ${s.start})，本轮 [${s.start}, ${s.end})，${s.last?'最终块':'保留尾部'}`).join('；')+'。完整序列不变。',()=>{},{...extra,source:batch.segments.some(s=>s.start>0)?'continuation':'cut'});
      emit(`${batch.id}-pack`,`${batch.id}：将 ${batch.segments.length} 条请求装入同一个 batch`,'pack',['p0-schedule'],`按请求顺序拼入 ${batch.tokenCount} 个新 token，保留请求边界、prefix_len、extend_len；组批不是合并请求身份。`,()=>{},extra);
      for(let r=0;r<3;r++){
        emit(`${batch.id}-allocate-${r}`,`PP${r} 准备 ${batch.id} 的本地 batch / KV 槽`,'pack',[`p${r}-schedule`],'每一级都准备本地 batch 和 KV 写入映射；已有前缀保留，只为新增区间准备写入位置。',()=>{for(const s of batch.segments)Object.assign(state.ranks[r].requests[s.rid],{queue:'batch',selectedEnd:s.end,holdsKv:true,chunkNo:s.chunkIndex});},{...extra,work:{batch,rank:r,allocate:true}});
        if(r)emit(`${batch.id}-proxy-${r-1}-${r}`,`激活：PP${r-1} → PP${r}`,'proxy',[`p${r-1}-compute`,`p${r}-compute`],`传递 ${batch.tokenCount} 个新 token 的 hidden_states / residual，保持 batch 对应关系。KV 不沿这个通道移动。`,()=>{},{phase:4+r,kind:'proxy',work:{batch,rank:r},routes:[route(`p${r-1}-compute`,`p${r}-compute`,'proxy')]});
        emit(`${batch.id}-forward-${r}`,`PP${r} 执行 ${batch.id}：L${r*10}–L${r*10+9}`,'forward',[`p${r}-compute`],r===2?'末级完成本块计算；每条请求仅在最终块结果处理中追加有效首 token，中间块不能假装请求已完成。':'读取输入及前缀 KV，写入本级新 K/V，再产出给下一级的激活。',()=>{for(const s of batch.segments)state.ranks[r].requests[s.rid].cacheEnd=s.end;},{phase:4+r,kind:r===2?'output':'proxy',work:{batch,rank:r}});
      }
      for(let r=0;r<3;r++){
        emit(`${batch.id}-output-${r}`,`${batch.id} 结果沿 PP 环回到 PP${r}`,'output',[`p${r}-transfer`],'匹配旧 batch 槽并等待必要的 D2H 事件。接收结果与更新本地 Req 是相邻但不同的操作。',()=>{},{phase:7,kind:'output',work:{batch,rank:r},routes:[route(r===0?'p2-compute':`p${r-1}-transfer`,`p${r}-transfer`,'output',r===0?'output-return':'direct')]});
        emit(`${batch.id}-result-${r}`,`PP${r} 分别处理最终块 / 中间块`,'result',[`p${r}-transfer`],'最终块：追加 t₀，进入 inflight。中间块：维护未完成请求，不追加有效 t₀，保留前缀供下一轮使用。',()=>{for(const s of batch.segments){const q=state.ranks[r].requests[s.rid];q.queue=s.last?'inflight':'chunked';q.token=s.last;}},{phase:8,kind:'output',work:{batch,rank:r}});
        const sends=batch.segments.map(s=>({rid:s.rid,start:state.ranks[r].requests[s.rid].sendEnd,end:s.last?s.end:Math.floor(s.end/config.pageSize)*config.pageSize,last:s.last}));
        emit(`${batch.id}-send-${r}`,`PP${r} 提交 ${batch.id} 的本级 KV`,'send',[`p${r}-transfer`,'decode-kv'],'从 start_send_idx 发到已完成边界；非最终块只发整页，最终块附带 metadata。提交后源引用仍保留；不等待三级都处理完结果。',()=>{for(const s of sends){const q=state.ranks[r].requests[s.rid];q.sendEnd=s.end;if(s.last)q.sender='Transferring';}},{phase:8,kind:'kv',work:{batch,rank:r,sends},routes:[route(`p${r}-transfer`,'decode-kv','kv','kv')]});
      }
      const completed=batch.segments.filter(s=>s.last).map(s=>s.rid);
      if(completed.length){
        const wait=config.fault==='transfer_fail'&&completed.includes(config.waitRid);
        emit(`${batch.id}-poll`,'逐级 poll：成功、失败和仍在传输','consensus',['p0-transfer','p1-transfer','p2-transfer'],'同一 rid 在不同 PP 的状态可以不同。先按本地状态分类，再求跨 PP 终态交集。',()=>{for(let r=0;r<3;r++)for(const id of completed)state.ranks[r].requests[id].sender=config.fault==='transfer_fail'&&config.failed[r].includes(id)?'Failed':wait&&r===config.waitRank&&id===config.waitRid?'Transferring':'Success';},{phase:9,work:{ids:completed,poll:true}});
        const allowed=terminalRound(completed,batch.id);release(allowed,batch.id);const pending=completed.filter(id=>!allowed.includes(id));
        if(pending.length){
          emit(`${batch.id}-wait`,'交集未包含的请求继续保留资源','consensus',['p0-release','p1-release','p2-release'],'部分请求已允许收尾，其余不能一起释放。Failed 不会永远阻塞交集；阻塞的是某一级尚未到终态。',()=>{},{phase:9,work:{ids:pending,wait:true}});
          emit(`${batch.id}-settled`,'后续轮询：剩余级也到终态','consensus',['p0-transfer','p1-transfer','p2-transfer'],'状态变化后重新求交，剩余请求才可能进入 release 名单。',()=>{for(let r=0;r<3;r++)for(const id of pending){const q=state.ranks[r].requests[id];if(!terminal(q.sender))q.sender=config.failed[r].includes(id)?'Failed':'Success';}},{phase:9,work:{ids:pending,poll:true}});
          release(terminalRound(pending,`${batch.id}-retry`),`${batch.id}-retry`);
        }
      }
      group=null;
    }
    emit('finish','Prefill 请求结果总览','decode',['frontend',...(state.requests.some(q=>q.outcome==='success')?['decode-run']:[])],'终态交集只授权收尾；只将成功交接的请求标为可继续 Decode。P 的清理不是 D 的启动屏障；D 内部失败传播和调度不在本页模拟。',()=>{},{phase:11,kind:'kv'});
    return {config,requests,nodes,events,frames,sources,batches:batchNumber,snapshot(index){return copy(index<0?frames[0].before:frames[Math.min(index,frames.length-1)].after);}};
  }
  return {nodes,sources,defaults,presets,normalize,bootstrapConsensus,terminalConsensus,planBatch,createScenario};
});
