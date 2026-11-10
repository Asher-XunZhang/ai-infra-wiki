/* Dependency-preserving teaching schedule. Durations are narration units, not traces. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.PDExecution=api;})(globalThis,()=>{
  'use strict';
  const clone=x=>JSON.parse(JSON.stringify(x));
  const revisions={lifecycle:'279339f113b79af84f27fd3ac92d0a13bd3f4cbd',dataflow:'279339f113b79af84f27fd3ac92d0a13bd3f4cbd'};
  const lanes=[...Array.from({length:3},(_,r)=>[{id:`cpu${r}`,label:`PP${r} · Scheduler / CPU`},{id:`gpu${r}`,label:`PP${r} · ModelRunner / GPU`},{id:`worker${r}`,label:`PP${r} · Mooncake worker`}]).flat(),{id:'link',label:'PP 消息 / 共识'}];
  function differences(a,b,path=[]){
    if(JSON.stringify(a)===JSON.stringify(b))return [];
    if(a&&b&&typeof a==='object'&&typeof b==='object'&&Array.isArray(a)===Array.isArray(b)&&(!Array.isArray(a)||a.length===b.length))return Object.keys(b).flatMap(k=>differences(a[k],b[k],[...path,k]));
    return [{path,value:clone(b)}];
  }
  function apply(state,changes){for(const {path,value}of changes){let target=state;for(const p of path.slice(0,-1))target=target[p];target[path.at(-1)]=clone(value);}}
  function build(model,kind){
    const events=[],copies=[],ends=[0,0,0];let cursor=0,lastMain=null;
    function rankOf(f){return f.work?.rank??Number(f.key.match(/(?:allocate|compute|forward|result|send|output|cleanup|release|finish|retry)-(\d)(?:-|$)/)?.[1]??f.title.match(/PP(\d)/)?.[1]??f.key.match(/-(\d)$/)?.[1]??0);}
    function add(e){events.push(e);return e;}
    for(let i=0;i<model.frames.length;i++){
      const f=model.frames[i],r=rankOf(f),k=f.key;
      const compute=/^B\d+-(forward|compute)-/.test(k),send=/^B\d+-send-/.test(k),route=f.flow||f.routes?.[0];
      const wait=Boolean(f.gate||f.work?.wait||/blocked|d2h-wait/.test(k));
      const poll=/^(terminal-poll|B\d+-poll|.*-settled|transfer-ready)$/.test(k);
      if(poll){
        // A sampled terminal cannot precede the corresponding worker completion.
        // The explicit delayed-rank scenario leaves one worker running at its first sample.
        const relevant=copies.filter(e=>e.end>cursor&&e.requests.some(q=>{
          const row=kind==='lifecycle'?f.after.ranks[e.rank].requests[q.rid]:f.after.ranks[e.rank];
          return ['Success','Failed'].includes(row.sender??row.poll);
        }));
        const until=Math.max(cursor,...relevant.map(e=>e.end));
        if(until>cursor)add({id:`wait-${k}`,start:cursor,end:until,lane:'link',rank:r,type:'wait',title:'等待传输终态采样',detail:'Scheduler 在后续迭代中再次检查；本请求仍保留源 KV。此条表示 R 的依赖等待，不表示 CPU 持续阻塞或整个服务停顿。',deps:[],source:f.source});
        cursor=until;
      }
      let type=compute?'compute':send?'submit':wait?'wait':route?(route.kind==='proxy'?'activation':route.kind):'control';
      if(type==='kv')type='submit';
      const duration=compute?1.8:wait?1.4:/result/.test(k)?0.85:route?0.75:0.5;
      const lane=compute?`gpu${r}`:route&&!send?'link':`cpu${r}`;
      const allocation=k.match(/^(B\d+)-allocate-(\d)$/);
      // Local input/mapping preparation does not consume the upstream activation.
      // The receive/forward remains after that activation is available.
      const priorAllocation=allocation&&r>0?events.find(e=>e.id===`${allocation[1]}-allocate-${r-1}`):null;
      const begin=priorAllocation?priorAllocation.end:cursor;
      const e=add({id:k,start:begin,end:begin+duration,lane,rank:r,type,title:f.title,detail:f.description??f.detail,source:f.source,frame:f,index:i,changes:differences(f.before,f.after),deps:priorAllocation?[priorAllocation.id]:lastMain?[lastMain.id]:[]});
      if(e.end>=cursor){cursor=e.end;lastMain=e;}
      if(send){
        const requests=kind==='lifecycle'?f.work.sends.map(s=>({...s,failed:model.config.fault==='transfer_fail'&&s.last&&model.config.failed[r].includes(s.rid)})):[{rid:'R0',start:f.before.ranks[r].sent,end:f.after.ranks[r].sent,last:f.after.ranks[r].token,failed:model.id==='transfer-fail'&&r===1}];
        const start=Math.max(e.end,ends[r]);
        const delayed=(model.id==='transfer-wait'||model.id==='transfer-fail')?r===2:model.config?.fault==='transfer_fail'&&r===model.config.waitRank;
        const c=add({id:`copy-${k}`,start,end:start+(delayed?15:9),lane:`worker${r}`,rank:r,type:'kv',title:`${k.split('-')[0]} · 本级 KV${requests.some(s=>s.last)?' + aux':''}`,detail:'sender.send 将任务加入传输队列后返回。独立 worker 读取源 KV 并搬运；最终块接着传 aux，再报告终态。图中进度是教学动作进度，不是已收到的字节数。',source:'worker',requests,deps:[e.id,...(copies.filter(c=>c.rank===r).slice(-1).map(c=>c.id))]});
        ends[r]=c.end;copies.push(c);
      }
    }
    const duration=Number(Math.max(cursor,...ends).toFixed(3)),revision=revisions[kind];
    // Separate state commits from activity spans. Reverse seek never replays mutations.
    const commits=events.filter(e=>e.frame).sort((a,b)=>a.end-b.end);
    function sample(t){
      t=Math.max(0,Math.min(duration,Number(t)||0));
      const done=commits.filter(e=>e.end<=t+1e-8);
      const state=clone(model.frames[0].before);for(const e of done)apply(state,e.changes);
      const active=events.filter(e=>e.start<=t&&t<e.end),focus=active.find(e=>e.frame)??active.find(e=>e.type==='wait')??active[0]??done.at(-1)??commits[0];
      const complete=copies.filter(e=>e.end<=t+1e-8);
      return {t,state,active,focus,done,complete,progress:e=>Math.max(0,Math.min(1,(t-e.start)/(e.end-e.start)))};
    }
    function sourceUrl(e){
      if(e.source==='worker')return `https://github.com/sgl-project/sglang/blob/${revision}/python/sglang/srt/disaggregation/mooncake/conn.py#L1847`;
      const source=kind==='lifecycle'?model.sources[e.source]??e.source:model.sources?.[e.source];
      const anchor=Array.isArray(source)?`${source[0]}#L${source[1]}`:source;
      return `https://github.com/sgl-project/sglang/blob/${revision}/python/sglang/srt/${anchor}`;
    }
    return {events,copies,commits,duration,kind,model,revision,sample,sourceUrl};
  }
  function rows(tl,s,rid='R0'){
    return s.state.ranks.map((row,r)=>{
      const q=tl.kind==='lifecycle'?row.requests[rid]:row;
      const completed=s.complete.flatMap(c=>c.rank===r?c.requests.filter(x=>x.rid===rid&&!x.failed):[]);
      return {queue:q.queue,poll:q.sender??q.poll,kv:q.holdsKv??q.kv,metadata:q.metadata,computed:q.cacheEnd??q.computed??0,submitted:q.sendEnd??q.sent??0,completed:Math.max(0,...completed.map(x=>x.end)),token:q.token,reason:q.reason};
    });
  }
  // Container references are independent of phase labels. Batch may coexist with inflight.
  function references(tl,s,rank,rid='R0'){
    const row=rows(tl,s,rid)[rank],refs={bootstrap:row.queue==='bootstrap',waiting:row.queue==='waiting',batch:false,chunk:false,inflight:row.queue==='inflight'};
    let latestAllocation=null;
    for(const e of s.done){
      if(e.rank!==rank)continue;
      const f=e.frame,k=e.id;
      if(/-allocate-/.test(k)){
        const seg=f.work?.batch?.segments.find(x=>x.rid===rid);
        if(tl.kind==='lifecycle'&&!seg)continue;
        latestAllocation=e;refs.batch=true;
        refs.chunk=tl.kind==='lifecycle'?!seg.last:/chunks|skip-chunk-output/.test(tl.model.id)&&!k.startsWith('B3-');
      }
      if(/-send-/.test(k)&&latestAllocation&&row.queue==='chunked')refs.batch=false;
    }
    // A terminal Req can remain in an old batch until a later filter pass.
    return refs;
  }
  return {build,rows,references,lanes,revisions};
});
