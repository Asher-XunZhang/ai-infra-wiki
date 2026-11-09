/* Fixed-source teaching schedules. Durations are narrative units, not traces. */
(function(root,factory){const api=factory();if(typeof module==='object')module.exports=api;else root.PPConsensus=api;})(typeof window==='undefined'?globalThis:window,()=>{
  'use strict';
  const revision='279339f113b79af84f27fd3ac92d0a13bd3f4cbd';
  const files={pp:'managers/scheduler_pp_mixin.py',prefill:'disaggregation/prefill.py',cache:'mem_cache/unified_radix_cache.py',utils:'disaggregation/utils.py'};
  const sources={bootstrap:['pp',593],abort:['pp',1340],terminal:['pp',635],bootReturn:['pp',660],releaseReturn:['pp',683],transport:['pp',735],finalize:['prefill',395],pop:['prefill',450],localRelease:['prefill',972],loop:['pp',235],counts:['cache',3073],prefix:['cache',3065],sync:['cache',346],reduce:['cache',333],load:['cache',3187],write:['cache',3130],events:['cache',3290],poll:['utils',252],commit:['pp',705]};
  const sourceURL=key=>{const [f,line]=sources[key];return `https://github.com/sgl-project/sglang/blob/${revision}/python/sglang/srt/${files[f]}#L${line}`;};
  const modes={
    bootstrap:{name:'Bootstrap · 请求准入',question:'一个级已握手，为什么请求还不能进入 waiting_queue？',cases:{normal:'全部就绪',late:'PP1 晚就绪 · 两轮采样',abort:'PP1 取消 R2',metadata:'PP0 metadata 不足'},scope:'PD Prefill · PP=3，Attention TP=CP=1；关闭乐观 Prefill。R0–R2 是请求 ID，三个进程各有本地 Req。',rule:'逐级合并：good 取交集，bad 取并集。回流经过本地 finalize 后，转发实际处理成功 / 失败的名单。'},
    release:{name:'Release · 本地收尾',question:'自己的 KV 已经发完，为什么还不能解除请求持有？',cases:{normal:'各级完成时间不同',slow:'PP1 传输慢 · 两轮采样',failure:'R1 在 PP1 失败',repoll:'名单回来，本地复查暂态'},scope:'PD Prefill · PP=3，Attention TP=CP=1；pending_bootstrap=False。金色条是独立 KV worker，终态不等于端到端成功。',rule:'终态集合 = Success ∪ Failed；沿 PP 求交集。收到 release 名单后，本地重新 poll，按成功 / 失败分别收尾。'},
    hicache:{name:'HiCache ACK · 缓存进度',question:'收到完成计数，为什么还要等自己的 CUDA event？',cases:{normal:'PP1 回载较慢',prefix:'本轮只公布一条 load ACK',zero:'PP0 未就绪 · load count=0'},scope:'UnifiedRadixCache + HiCache cache 模式 · PP=3，TP=CP=1；关闭 external linker、L3 与 buffer pipeline。W0 是写入 ACK，L0–L2 是回载 ACK，不是请求 ID。',rule:'PP0 统计连续就绪前缀，级内 MIN 后逐级发布计数。后级先 isend 转发，再按计数 pop ACK、等待本地 event、解除加载保护。'}
  };
  const ids=['R0','R1','R2'],clone=o=>JSON.parse(JSON.stringify(o));
  const intersection=(a,b)=>a.filter(x=>b.includes(x)),union=(a,b)=>[...new Set([...a,...b])].sort();
  function bootstrapMerge(up,local,aborts=[]){const good=up?intersection(up.good,local.good):[...local.good],bad=up?union(up.bad,local.bad):[...local.bad];return {good:good.filter(x=>!aborts.includes(x)),bad:union(bad,aborts)};}
  const readyPrefix=q=>{let n=0;for(const ready of q){if(!ready)break;n++;}return n;};
  function build(mode='bootstrap',condition='normal'){
    if(!modes[mode])mode='bootstrap';if(!modes[mode].cases[condition])condition='normal';
    const events=[],changes=[];let seq=0;
    const lanes=Array.from({length:3},(_,r)=>({id:'cpu'+r,name:`PP${r} · Scheduler / CPU`}));
    lanes.push({id:'wire',name:'跨 PP · 控制消息'});
    for(let r=0;r<3;r++)lanes.push({id:'device'+r,name:`PP${r} · ${mode==='release'?'KV worker':mode==='hicache'?'GPU · H2D 流':'GPU · 已提交前向'}`});
    const initial={mode,stages:Array.from({length:3},()=>({observed:null,aggregate:null,incoming:null,returned:null,forwarded:null,queue:Object.fromEntries(ids.map(id=>[id,mode==='bootstrap'?'bootstrap':'inflight'])),outcome:{},counts:null,acks:{W0:'queued',L0:'queued',L1:'queued',L2:'queued'},released:[],worker:false})),round:1};
    function add(id,lane,start,duration,label,detail,source,extra={}){const e={id,lane,start,end:start+duration,label,detail,source,kind:'control',deps:[],order:seq++,...extra};events.push(e);return e;}
    const change=(e,rank,patch,when='end')=>changes.push({time:e[when],rank,patch:clone(patch),order:seq++});
    const message=(id,from,to,start,payload,source,dep,back=false)=>{
      const e=add(id,'wire',start,1.2,`${back?'回流':'汇总'} PP${from} → PP${to}`,`${back?'异步发送回流结果':'异步发送本轮候选'}。载荷是 CPU 控制名单；发送返回与接收完成分开，接收端随后处理本地状态。`,source,{kind:'message',from,to,payload:clone(payload),back,deps:[dep.id]});
      change(e,to,{[back?'returned':'incoming']:payload});return e;
    };
    if(mode!=='hicache'){
      const late=condition==='late'||condition==='slow';
      if(mode==='bootstrap'){
        for(let r=0;r<3;r++)add('gpu'+r,'device'+r,0,5+r*1.3,`已有 B${r} 前向`,`t=0 之前已提交的不同 batch B${r}。GPU 可继续执行；本图没有为等待准入的 R0–R2 提交前向。`,'loop',{kind:'device'});
        if(late){lanes.push({id:'handshake',name:'PP1 · 握手后端'});add('handshake','handshake',7,1.3,'R1 握手随后就绪','外部握手完成，本轮 poll 快照已取得，必须等下一轮重新采样；这不是 GPU 操作。','bootstrap',{kind:'notice'});}
      }else{
        for(let r=0;r<3;r++){
          const end=late&&r===1?19.5:[1,3,5][r];
          const e=add('worker'+r,'device'+r,0,end,`本级 R0–R2 KV 传输`,r===1&&condition==='failure'?'R1 传输失败；本级另两条请求成功。其他级独立推进。':'每级 worker 独立推进本级层的 KV；CPU 通过 poll 采样其完成状态。','terminal',{kind:'device'});change(e,r,{worker:true});
        }
      }
      let previousRound=[];
      for(let round=0;round<(late?2:1);round++){
        const offset=round*17,local=[],aggs=[],polls=[],returns=[];let up=null,prior=null;
        for(let r=0;r<3;r++){
          const start=prior?prior.end:offset+.4;
          const statuses=ids.map(id=>mode==='bootstrap'?(late&&r===1&&id==='R1'&&round===0?'Bootstrapping':'WaitingForInput'):(late&&r===1&&round===0?'Transferring':r===1&&condition==='failure'&&id==='R1'?'Failed':'Success'));
          const pending=round?['R1']:ids;
          // After first-round successes leave the queue, a later round samples only the remaining requests.
          const eligible=mode==='bootstrap'&&round?pending:mode==='release'&&round?ids:ids;
          const observed=Object.fromEntries(eligible.map(id=>[id,statuses[ids.indexOf(id)]]));
          if(mode==='bootstrap'){
            local[r]={good:eligible.filter(id=>observed[id]==='WaitingForInput'),bad:[]};
            const aborts=condition==='abort'&&r===1?['R2']:[];
            aggs[r]=bootstrapMerge(up,local[r],aborts);
          }else{local[r]=eligible.filter(id=>['Success','Failed'].includes(observed[id]));aggs[r]=up===null?local[r]:intersection(up,local[r]);}
          const formula=mode==='bootstrap'?(r?`G = ${JSON.stringify(up.good)} ∩ ${JSON.stringify(local[r].good)}；B = ${JSON.stringify(up.bad)} ∪ ${JSON.stringify(local[r].bad)}`:'读取本地 WaitingForInput / Failed 集合'):(r?`${JSON.stringify(up)} ∩ ${JSON.stringify(local[r])}`:'读取本地 Success ∪ Failed 集合');
          const e=add(`q${round}-poll${r}`,'cpu'+r,start,1.1,`第 ${round+1} 轮 · ${r?'接收后采样 / 合并':'采样候选'}`,formula+(condition==='abort'&&r===1?'。本地 R2 已取消：从 good 移除，加入 bad；即使 poll 仍为 WaitingForInput。':''),mode==='bootstrap'?'bootstrap':'terminal',{rank:r,kind:'merge',formula,round:round+1,deps:prior?[prior.id]:previousRound});
          change(e,r,{observed,aggregate:aggs[r]});polls.push(e);
          if(r<2){const send=add(`q${round}-send${r}`,'cpu'+r,e.end+.4,.4,'提交候选发送','本轮候选在 loop 尾部异步转发；中间未展开的选批 / forward / 结果步骤没有合并成全局屏障。','transport',{rank:r,deps:[e.id]});prior=message(`q${round}-hop${r}`,r,r+1,send.end,aggs[r],mode==='bootstrap'?'bootstrap':'terminal',send);}
          up=aggs[r];
        }
        let payload=clone(up),last=polls[2];
        const send=add(`q${round}-back2`,'cpu2',last.end+.4,.4,'末级提交回流','对应 bmbs/tmbs 槽位已建立时，末级把本轮结果异步送回 PP0。这里只展开一条名单旅程；slot 的空迭代见 PP Event Loop。',mode==='bootstrap'?'bootReturn':'releaseReturn',{rank:2,deps:[last.id]});
        let msg=message(`q${round}-ring2`,2,0,send.end,payload,mode==='bootstrap'?'bootReturn':'releaseReturn',send,true);
        for(let r=0;r<3;r++){
          const input=clone(payload),queue={},outcome={},forward=mode==='bootstrap'?{good:[],bad:[]}:payload;
          for(const id of ids){
            if(mode==='bootstrap'){
              if(input.bad.includes(id)){queue[id]='failed';outcome[id]='失败清理';forward.bad.push(id);}
              else if(input.good.includes(id)){
                if(condition==='metadata'&&r===0&&id==='R1'){queue[id]='bootstrap';outcome[id]='metadata 不足';}
                else{queue[id]='waiting';outcome[id]='本地准入';forward.good.push(id);}
              }else if(!round||id==='R1'){queue[id]='bootstrap';outcome[id]='不在本轮 good';}
            }else if(input.includes(id)){
              if(condition==='repoll'&&r===1&&id==='R0'){queue[id]='inflight';outcome[id]='复查 Transferring';}
              else{queue[id]='done';outcome[id]=condition==='failure'&&r===1&&id==='R1'?'本级失败清理':'本级成功收尾';}
            }else{queue[id]='inflight';outcome[id]='未获收尾名单';}
          }
          const e=add(`q${round}-apply${r}`,'cpu'+r,msg.end,1.1,mode==='bootstrap'?'本地 finalize / 移队列':'本地重新 poll / 收尾',mode==='bootstrap'?'good 仍需 metadata 槽和 sender.init；process_bootstrapped_queue 返回实际移队列的 good/failed，供后续迭代转发。当前选批位置已过去，waiting 不等于立即 GPU 执行。':'名单仅限定检查范围。本地再次观察到 Success 才走成功清理；Failed 走失败处理；非终态继续留在 undone_reqs。',''+(mode==='bootstrap'?'pop':'localRelease'),{rank:r,kind:'apply',deps:[msg.id],input,output:clone(forward)});
          // Preserve requests removed in earlier rounds; patches only affect this round's members.
          change(e,r,{queue,outcome,forwarded:forward});returns.push(e);
          if(r<2){const f=add(`q${round}-forward${r}`,'cpu'+r,e.end+.4,.4,'后续迭代转发保存结果',mode==='bootstrap'?'这里转发上次本地实际处理结果，未必等于末级最初 good 名单。':'转发上次保存的 release_rids；本地暂态请求仍由本地防御分支保留。',mode==='bootstrap'?'bootReturn':'releaseReturn',{rank:r,deps:[e.id]});payload=clone(forward);msg=message(`q${round}-ring${r}`,r,r+1,f.end,payload,mode==='bootstrap'?'bootReturn':'releaseReturn',f,true);}
        }
        previousRound=returns.map(e=>e.id);
      }
    }else{
      const count=condition==='zero'?0:condition==='prefix'?1:2;
      const finishes=[[condition==='zero'?10:.5,condition==='normal'?1:11,16],[5,9,16],[3,5.5,16]];
      for(let r=0;r<3;r++){
        for(let k=0;k<3;k++)add(`copy${r}-${k}`,'device'+r,k?finishes[r][k-1]:0,finishes[r][k]-(k?finishes[r][k-1]:0),`L${k} 回载`,`PP${r} 的 L${k} 对应本地 CUDA finish_event；各级层的 KV 不同，ACK 顺序相同。`,'load',{kind:'device',rank:r,ack:'L'+k});
      }
      const rootCount=add('count','cpu0',1.2,.8,'连续前缀 → 级内 MIN',`PP0 当前 write=1，load=${count}。本例 TP=CP=1，MIN 是恒等；多 rank 时在 PP0 的相关组内规约，不收集 PP1/PP2 的 ready 数。`,'counts',{rank:0,kind:'merge',formula:`[write, load, digest, −digest] = [1, ${count}, 0, 0]`});
      change(rootCount,0,{counts:[1,count,0,0]});
      let input=rootCount;
      for(let r=0;r<3;r++){
        let at=r?input.end:rootCount.end,dep=input;
        if(r<2){
          const send=add('count-send'+r,'cpu'+r,at,.4,'isend 计数副本','先 clone 计数张量并 isend，保存 work；没有先等待本地 load ACK。下一轮开始会回收历史 work。','sync',{rank:r,deps:[dep.id]});
          const m=add('count-hop'+r,'wire',send.end,1.2,`计数 PP${r} → PP${r+1}`,`CPU int64 控制张量 [1, ${count}, 0, 0]；不携带 KV。接收端 recv 写入同长度缓冲区。`,'sync',{kind:'message',from:r,to:r+1,payload:[1,count,0,0],deps:[send.id]});change(m,r+1,{counts:[1,count,0,0]});input=m;at=send.end;dep=send;
        }
        const write=add('write'+r,'cpu'+r,at,.4,'消费 W0','先 writing_check，再 loading_check。W0 的 D2H 写入已在本例起点前完成，pop 后同步本地事件再收尾。','write',{rank:r,ack:'W0',deps:[dep.id]});change(write,r,{acks:{W0:'popped'}},'start');change(write,r,{acks:{W0:'finished'}});at=write.end;dep=write;
        for(let k=0;k<count;k++){
          const ack='L'+k,pop=add(`pop${r}-${k}`,'cpu'+r,at,.3,`pop ${ack}`,`从队首取出 ${ack}。pop 不表示加载保护已解除；还要执行本地 finish_event.synchronize()。`,'load',{rank:r,ack,deps:[dep.id]});change(pop,r,{acks:{[ack]:'popped'}},'start');at=pop.end;dep=pop;
          if(finishes[r][k]>at){const wait=add(`wait${r}-${k}`,'cpu'+r,at,finishes[r][k]-at,`等待 ${ack} event`,'本 Scheduler 线程等待本地事件；本级 H2D 流以及其他 PP 级仍可独立推进。','load',{kind:'wait',rank:r,ack,deps:[pop.id]});dep=wait;at=wait.end;}
          const finish=add(`finish${r}-${k}`,'cpu'+r,at,.4,`解除 ${ack} 加载保护`,'本地 finish_event 已完成，移出 ongoing_load_back，解除 device / host lock 并 finish_load_back；host 副本仍可保留。','load',{rank:r,ack,deps:[dep.id,`copy${r}-${k}`]});change(finish,r,{acks:{[ack]:'finished'}});at=finish.end;dep=finish;
        }
        add('continue'+r,'cpu'+r,at,.6,'本级继续选批','本级本轮缓存事件已处理，可进入后续选批；未纳入计数的 ACK 留待后轮。本图不由计数推断一个新 batch 已可计算。','events',{rank:r,deps:[dep.id]});
      }
    }
    const duration=mode==='hicache'?12:Math.max(...events.map(e=>e.end))+1;
    const sorted=changes.sort((a,b)=>a.time-b.time||a.order-b.order);
    function sample(time){
      const t=Math.max(0,Math.min(duration,Number(time)||0)),s=clone(initial);
      for(const c of sorted){if(c.time>t+1e-8)break;const stage=s.stages[c.rank];for(const [key,value]of Object.entries(c.patch)){if(['queue','outcome','acks'].includes(key))Object.assign(stage[key],clone(value));else stage[key]=clone(value);}}
      s.t=t;s.active=events.filter(e=>e.start<=t&&t<e.end);s.focus=s.active.filter(e=>e.kind!=='device').sort((a,b)=>b.start-a.start)[0]||events.filter(e=>e.start<=t&&e.kind!=='device').sort((a,b)=>b.start-a.start)[0]||events.find(e=>e.kind!=='device');
      s.progress=e=>Math.max(0,Math.min(1,(t-e.start)/(e.end-e.start)));return s;
    }
    return {mode,condition,events,lanes,duration,sample,initial,changes:sorted,steps:[...new Set([0,...events.filter(e=>e.kind!=='device'&&e.start<duration).map(e=>e.start),duration])].sort((a,b)=>a-b)};
  }
  return {revision,files,sources,sourceURL,modes,ids,intersection,union,bootstrapMerge,readyPrefix,build};
});
