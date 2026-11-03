/* Approved PD architecture teaching animation; SGLang e00930c5489053f26d86b179cee0d087f846acbb. */
(()=>{
  'use strict';
  const root=document.getElementById('sglang-opening'),svg=root.querySelector('#op-map'),slider=root.querySelector('#op-range'),button=root.querySelector('#op-toggle'),restart=root.querySelector('#op-restart'),speedControl=root.querySelector('#op-speed');
  const NS='http://www.w3.org/2000/svg',schema='sglang-opening.v2';
  const colors={req:'#f0bf69',act:'#82c8f4',kv:'#69e0c2',meta:'#cbb7ff',out:'#f0bf69',neutral:'#466555'};
  const bounds=[0,.07,.16,.27,.52,.63,.72,.81,.90,1.0001];
  const names=['请求接入','D 预分配','P 准备 EXTEND','P · PP4 计算','KV / metadata 交接','两端独立收尾','D · 第 1 轮','D · 第 2 轮','响应完成'];
  // One monotone time map retimes the entire trace together. Concurrent flows
  // stay concurrent; the established source sequence and saved positions stay intact.
  // Each entry is [semantic end position, teaching seconds since previous entry].
  const beats=[
    [.010,1],[.027,1.4],[.038,1],[.070,1.8],
    [.093,1.4],[.110,1.2],[.128,1.3],[.160,1.6],
    [.182,1.2],[.200,1.1],[.211,.7],[.222,.8],[.235,.9],[.250,1.3],[.270,1.5],
    [.315,1.8],[.320,.2],[.3325,1.6],
    [.3775,1.2],[.3825,.15],[.395,1.3],
    [.440,1.2],[.445,.15],[.4575,1.3],
    [.502,1.8],[.510,1.3],[.520,1.3],
    [.536,1.1],[.549,.5],[.599,1.8],[.615,1],[.630,1],
    [.635,.15],[.675,1.65],[.678,.25],[.680,.2],[.689,1.4],
    [.699,1.3],[.709,2],[.719,.9],[.720,.1],
    [.729,1.5],[.734,.9],[.739,.7],[.742,.6],[.750,1.3],
    [.761,1.5],[.769,1.3],[.779,1.5],[.786,1.4],
    [.794,1.2],[.802,1.8],[.810,.9],
    [.818,1.3],[.824,.7],[.826,.2],[.832,.8],[.840,1.1],
    [.851,1.2],[.859,1.1],[.869,1.2],[.876,1.2],
    [.884,1],[.892,1.5],[.900,.8],
    [.908,1.3],[.916,.8],[.930,.35],[.965,.9],[1,.6]
  ];
  const pacing=[[0,0]];
  for(const [p,seconds]of beats)pacing.push([p,pacing[pacing.length-1][1]+seconds]);
  const duration=pacing[pacing.length-1][1],rates=[.5,.75,1,1.25,1.5,2];
  let progress=0,rate=1,playing=false,raf=0,last=0,lastWidth=0,scene={},saved='';
  const clamp=v=>Math.max(0,Math.min(1,v));
  function mapTime(value,axis){
    const other=1-axis,v=Math.max(0,Math.min(pacing[pacing.length-1][axis],value));
    for(let i=1;i<pacing.length;i++){const a=pacing[i-1],b=pacing[i];if(v<=b[axis])return a[other]+(v-a[axis])/(b[axis]-a[axis])*(b[other]-a[other]);}
    return pacing[pacing.length-1][other];
  }
  const timeAt=p=>mapTime(p,0),progressAt=t=>mapTime(t,1);
  const formatTime=t=>Math.floor(t/60)+':'+String(Math.floor(t%60)).padStart(2,'0');
  const snapshotKey=(p,r)=>String(Math.round(p*1000)/1000)+'|'+r;
  const text=(id,value)=>{const e=root.querySelector('#'+id);if(e&&e.textContent!==value)e.textContent=value;};
  const add=(parent,tag,attrs={},value)=>{const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(value!==undefined)e.textContent=value;parent.appendChild(e);return e;};
  function label(parent,x,y,value,cls='flow-label',anchor='start'){return add(parent,'text',{x,y,class:cls,'text-anchor':anchor},value);}
  function node(id,x,y,w,h,title,sub,accent='act'){
    const g=add(svg,'g',{class:'module','data-module':id,style:'--accent:'+colors[accent]});
    const fullTitle=title;
    if(scene.narrow){title=({'Scheduler · PrefillBootstrapQueue':'Scheduler · BootstrapQueue','Scheduler · DecodePreallocQueue':'Scheduler · PreallocQueue','ReqToTokenPool · KV allocator / Pool':'ReqToTokenPool · KV Pool','process_batch_result → inflight · Sender':'结果 → inflight · KV Sender','KV Receiver · DecodeTransferQueue':'KV Receiver · TransferQueue','Detokenizer → TokenizerManager / HTTP':'Detokenizer → HTTP 返回'})[title]||title;}
    g.setAttribute('aria-label',fullTitle);add(g,'rect',{x,y,width:w,height:h});label(g,x+8,y+(sub?14:h<30?17:20),title,'node-title');
    let subEl=null;if(sub)subEl=label(g,x+8,y+29,sub,'sub');
    scene.nodes[id]={g,x,y,w,h,sub:subEl};return scene.nodes[id];
  }
  function path(id,coords,type='neutral',dashed=false){
    const p=add(scene.lines,'path',{d:coords,class:'flow','data-edge':id,style:'--edge:'+colors[type],'marker-end':'url(#op-arrow-'+type+')'});
    if(dashed)p.setAttribute('stroke-dasharray','4 3');scene.edges[id]={p,type};return p;
  }
  function line(id,a,b,type='neutral',dashed=false){return path(id,`M${a[0]},${a[1]} L${b[0]},${b[1]}`,type,dashed);}
  function build(){
    const W=Math.round(svg.getBoundingClientRect().width);if(!W||W===lastWidth)return;lastWidth=W;
    const narrow=W<580,cw=narrow?W-24:(W-54)/2,px=12,dx=narrow?12:W-cw-12,py=70,dy=narrow?562:70;
    const H=narrow?1090:540;
    svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.style.height=H+'px';
    scene={W,H,narrow,nodes:{},edges:{},stages:[],dots:{},scopes:{},px,dx,cw,py,dy};
    add(svg,'title',{id:'op-title'},'SGLang SRT：真实源码对象的包含关系、调用链与 PD 数据传输');
    add(svg,'desc',{id:'op-desc'},'Client 与 Router 位于 SRT 运行时之外。每侧 HTTP Server 调用 TokenizerManager，后者经 IPC 向 Scheduler 发请求。Scheduler 管理请求队列、ScheduleBatch 和 tree_cache，并创建 TpModelWorker。Worker 持有 ModelRunner；Runner 组织模型层、Attention、Sampler 以及 KV 映射、分配和存储。缓存是执行链引用的资源，不是请求串行经过的服务。P 的每个 PP/CP/TP rank 都有相应执行链。请求级 MooncakeKVSender 和 Receiver 属于 SRT disaggregation 适配层，跨 P/D 发送各层 KV 与 metadata。DetokenizerManager 把结果 ID 转为文字，经 TokenizerManager 和 HTTP 返回。');
    const defs=add(svg,'defs');for(const[k,c]of Object.entries(colors)){const m=add(defs,'marker',{id:'op-arrow-'+k,viewBox:'0 0 8 8',refX:7,refY:4,markerWidth:5,markerHeight:5,orient:'auto'});add(m,'path',{d:'M1 1 L7 4 L1 7',fill:'none',stroke:c,'stroke-width':1.4});}
    const under=add(svg,'g');scene.lines=add(svg,'g');const captions=add(svg,'g');
    function scope(id,x,y,w,h,title,source,kind='object'){
      const r=add(under,'rect',{x,y,width:w,height:h,rx:4,class:'code-scope '+kind,'data-scope':id});scene.scopes[id]={r,x,y,w,h};
      label(captions,x+8,y+17,title,'scope-title');if(source)label(captions,x+w-8,y+17,source,'scope-source','end');
    }
    function child(id,owner,x,y,w,h,title,sub,color='act'){
      const n=node(id,x,y,w,h,title,sub,color);n.g.dataset.owner=owner;return n;
    }
    function below(edge,a,b,type='neutral',offset=0){const n=scene.nodes[a],m=scene.nodes[b];path(edge,`M${n.x+n.w/2+offset},${n.y+n.h} V${m.y-5} H${m.x+m.w/2} V${m.y}`,type);}
    node('client',px,2,65,25,'Client',null,'req');node('router',px+88,2,70,25,'Router',null,'req');
    line('client-router',[px+65,14],[px+88,14],'req');line('router-client',[px+88,23],[px+65,23],'req',true);
    label(svg,px+174,33,narrow?'外部调用':'外部调用 · Client / Router 不属于 SRT 运行时','flow-label');
    scope('srt',2,39,W-4,H-43,'SGLang SRT','python/sglang/srt','runtime');
    for(const[role,x,y]of [['p',px,py],['d',dx,dy]]){
      const isP=role==='p',ix=x+7,iw=cw-14;
      label(svg,x+2,y+2,isP?'P · Prefill 实例':'D · Decode 实例','panel-title');
      label(svg,x+cw-2,y+2,isP?'PP4 × CP4 × TP4':'逻辑端，rank 未展开','panel-tag','end');
      const serverW=narrow?84:92;
      child(role+'Http','srt',ix,y+12,serverW,26,'HTTP Server',null,'req');
      child(role+'Tok','srt',ix+serverW+13,y+12,iw-serverW-13,26,'TokenizerManager',null,'req');
      line(role+'-http-tok',[ix+serverW,y+25],[ix+serverW+13,y+25],'req');
      if(!isP)line('d-tok-http',[ix+serverW+13,y+34],[ix+serverW,y+34],'out');
      scope(role+'Scheduler',x,y+65,cw,288,'Scheduler','scheduler.py','process');
      child(role+'Queue',role+'Scheduler',ix,y+90,iw,24,isP?'PrefillBootstrapQueue':'DecodePreallocQueue',null,'meta');
      child(role+'Queue2',role+'Scheduler',ix,y+120,iw,24,isP?'waiting_queue':'DecodeTransferQueue',null,'meta');
      const bw=(iw-10)*.52;
      child(role+'Batch',role+'Scheduler',ix,y+158,bw,24,'ScheduleBatch',null);
      child(role+'Cache',role+'Scheduler',ix+bw+10,y+158,iw-bw-10,24,'tree_cache',null,'kv');
      scope(role+'Worker',ix,y+194,iw,113,'TpModelWorker',narrow?'':'tp_worker.py');
      scope(role+'Runner',ix+6,y+218,iw-12,87,'ModelRunner',narrow?'':'model_runner.py');
      const modelW=Math.max(95,(iw-36)*.51),mx=ix+12,my=y+243;
      child(role+'Model',role+'Runner',mx,my,modelW,24,narrow?'Model / Attention':'Model · Attention',null);
      child(role+'Sampler',role+'Runner',mx,my+30,modelW,22,'Sampler ← logits',null,'req');
      const poolX=mx+modelW+12,poolW=ix+iw-12-poolX;
      const pool=add(svg,'g',{class:'module pools','data-module':role+'Mem','data-owner':role+'Runner',style:'--accent:'+colors.kv});
      add(pool,'rect',{x:poolX,y:my,width:poolW,height:60});label(pool,poolX+5,my+13,'ReqToTokenPool','pool-label');label(pool,poolX+5,my+27,'token_to_kv_','pool-label');label(pool,poolX+5,my+41,'pool_allocator','pool-label');label(pool,poolX+5,my+55,'KVCache','pool-label');
      scene.nodes[role+'Mem']={g:pool,x:poolX,y:my,w:poolW,h:60};
      line(role+'-kv-ref',[mx+modelW,my+12],[poolX,my+12],'kv',true);
      line(role+'-logits',[mx+modelW/2,my+24],[mx+modelW/2,my+30],'req');
      child(role+'Result',role+'Scheduler',ix,y+318,iw,34,'process_batch_result()',isP?'disagg_prefill_inflight_queue':'running_batch / waiting_queue','req');
      const tok=scene.nodes[role+'Tok'];path(role+'-ipc',`M${tok.x+tok.w/2},${tok.y+tok.h} V${y+52} H${ix+iw/2} V${y+90}`,'req');
      label(svg,ix+8,y+55,'请求 IPC','flow-label');
      below(role+'-queue-step',role+'Queue',role+'Queue2');
      below(role+'-queue-batch',role+'Queue2',role+'Batch');
      const batch=scene.nodes[role+'Batch'];path(role+'-dispatch',`M${batch.x+batch.w/2},${batch.y+batch.h} V${y+188} H${ix+3} V${y+194}`,'act');
      path(role+'-forward',`M${ix+3},${y+194} V${my+12} H${mx}`,'act');
      line(role+'-batch-cache',[batch.x+batch.w,batch.y+12],[ix+bw+10,batch.y+12],'meta',true);
      path(role+'-alloc',`M${batch.x+batch.w-5},${batch.y+batch.h} V${y+188} H${x+cw-3} V${my+46} H${poolX+poolW}`,'meta',true);
      const result=scene.nodes[role+'Result'];path(role+'-result',`M${mx+modelW},${my+41} H${mx+modelW+5} V${result.y-5} H${result.x+result.w*.3} V${result.y}`,'req');
      const adapter=isP?'MooncakeKVSender':'MooncakeKVReceiver';
      child(role+'Adapter','srt',ix,y+365,iw,26,adapter,null,'kv');
      if(isP)label(svg,ix+iw-7,y+383,'KV / metadata','flow-label','end');
      const cache=scene.nodes[role+'Cache'];path(role+'-cache-ref',`M${cache.x+cache.w},${cache.y+12} H${x+cw-3} V${my+28} H${poolX+poolW}`,'kv',true);
      if(isP){
        label(svg,x+4,y+408,narrow?'每格一 rank，对应上方执行链':'每格一个 rank：各自运行上方 Scheduler / Worker','topology-caption');
        const sw=(cw-24)/4;
        for(let i=0;i<4;i++){
          const sx=x+3+i*(sw+6),sy=y+417,g=add(svg,'g',{class:'stage','data-stage':i});
          add(g,'rect',{x:sx,y:sy,width:sw,height:36,class:'stage-box'});label(g,sx+5,sy+14,'PP'+i,'stage-title');
          for(let r=0;r<4;r++)for(let c=0;c<4;c++)add(g,'rect',{x:sx+sw-25+c*5,y:sy+7+r*5,width:4,height:4,class:'tile'});
          const st=label(g,sx+5,sy+29,'等待','stage-state');scene.stages.push({g,state:st,x:sx,y:sy,w:sw});
          if(i){const prev=scene.stages[i-1];path('pp'+i,`M${prev.x+prev.w/2},${sy+36} V${sy+43} H${sx+sw/2} V${sy+36}`,'act');}
        }
        below('p-to-sender','pResult','pAdapter','meta',iw/2-12);
        path('p-kv-sender',`M${poolX+poolW},${my+50} H${x+cw-3} V${y+374} H${ix+iw}`,'kv');
      }else{
        child('dDetok','srt',ix,y+417,iw,26,'DetokenizerManager',null,'req');
        label(svg,ix+4,y+456,'token IDs → text → TokenizerManager / HTTP','topology-caption');
        const result=scene.nodes.dResult,out=scene.nodes.dDetok;
        path('d-to-detok',`M${result.x+result.w},${result.y+13} H${x+cw-1} V${out.y+13} H${out.x+out.w}`,'req');
        path('d-return',`M${out.x},${out.y+13} H${x-5} V${y+41} H${tok.x+tok.w/2} V${tok.y+tok.h}`,'req',true);
        const q=scene.nodes.dQueue,mem=scene.nodes.dMem,adapterNode=scene.nodes.dAdapter,q2=scene.nodes.dQueue2;
        path('d-prealloc',`M${q.x+q.w},${q.y+12} H${x+cw-3} V${mem.y+46} H${mem.x+mem.w}`,'meta',true);
        path('d-prepare-receiver',`M${mem.x+mem.w},${mem.y+50} H${x+cw-3} V${adapterNode.y+6} H${adapterNode.x+adapterNode.w}`,'meta',true);
        path('d-write-kv',`M${adapterNode.x+adapterNode.w},${adapterNode.y+19} H${x+cw-3} V${mem.y+54} H${mem.x+mem.w}`,'kv');
        path('d-first-token',`M${q2.x},${q2.y+12} H${x+2} V${result.y+13} H${result.x}`,'req');
      }
    }
    const pQ=scene.nodes.pQueue,dA=scene.nodes.dAdapter,pA=scene.nodes.pAdapter,dQ=scene.nodes.dQueue2;
    if(!narrow){
      path('receive-info',`M${dA.x},${dA.y+6} H${W/2+8} V${pQ.y+12} H${pQ.x+pQ.w}`,'meta',true);
      path('kv',`M${pA.x+pA.w},${pA.y+9} H${dA.x}`,'kv');path('metadata',`M${pA.x+pA.w},${pA.y+21} H${dA.x}`,'meta',true);
      path('d-receiver-queue',`M${dA.x+dA.w},${dA.y+13} H${dx+cw-2} V${dQ.y+12} H${dQ.x+dQ.w}`,'meta',true);
    }else{
      path('receive-info',`M${dA.x},${dA.y+6} H5 V${pQ.y+12} H${pQ.x}`,'meta',true);
      path('kv',`M${pA.x+pA.w},${pA.y+7} H${W-4} V${dA.y+7} H${dA.x+dA.w}`,'kv');
      path('metadata',`M${pA.x},${pA.y+21} H5 V${dA.y+21} H${dA.x}`,'meta',true);
      label(svg,px+3,py+479,'P → D：各层 KV + metadata（首个 ID / 身份）','flow-label');
      path('d-receiver-queue',`M${dA.x+dA.w},${dA.y+13} H${dx+cw-2} V${dQ.y+12} H${dQ.x+dQ.w}`,'meta',true);
    }
    const pHttp=scene.nodes.pHttp,dHttp=scene.nodes.dHttp;
    path('request-p',`M${px+123},27 V33 H6 V${pHttp.y+13} H${pHttp.x}`,'req');
    path('request-d',narrow?`M${px+145},27 V36 H2 V${dHttp.y+13} H${dHttp.x}`:`M${px+158},14 H${px+168} V40 H${W/2} V${dHttp.y+13} H${dHttp.x}`,'req');
    path('d-http-router',narrow?`M${dHttp.x},${dHttp.y+21} H4 V3 H${px+168} V23 H${px+158}`:`M${dHttp.x},${dHttp.y+21} H${W/2-6} V40 H${px+168} V23 H${px+158}`,'out');
    scene.motionLayer=add(svg,'g');scene.dotsLayer=add(svg,'g');
    for(const e of events){
      const edge=scene.edges[e.edge];if(!edge)continue;
      const trail=add(scene.motionLayer,'path',{d:edge.p.getAttribute('d'),class:e.mode==='reference'?'reference-pulse':'flow-trail',stroke:colors[e.type],style:'display:none','data-event':e.id,'data-start':timeAt(e.a),'data-end':timeAt(e.b),'data-semantic-start':e.a,'data-semantic-end':e.b});
      const g=add(scene.dotsLayer,'g',{class:'payload',style:'display:none','aria-label':e.payload,'data-packet':e.id,'data-edge':e.edge,'data-type':e.type});
      if(e.type==='kv'){for(let i=0;i<3;i++)add(g,'rect',{x:-5,y:-4+i*3,width:10,height:2,rx:1,fill:colors.kv});}
      else if(e.type==='meta')add(g,'path',{d:'M0,-5 L5,0 L0,5 L-5,0 Z',fill:colors.meta});
      else if(e.type==='act')add(g,'rect',{x:-6,y:-3,width:12,height:6,rx:3,fill:colors.act});
      else if(e.type==='out')add(g,'circle',{r:4,fill:colors.out});
      else add(g,'rect',{x:-4,y:-4,width:8,height:8,rx:1,fill:colors.req});
      scene.dots[e.id]={g,trail};
    }
    root.dataset.tileCount='64';render();
  }
  // One deterministic trace drives packets, local references, seek and playback.
  // Intervals are teaching time, not measured service/communication latency.
  const events=[];
  function flow(edge,a,b,type,payload,options={}){events.push({id:edge+'-'+events.length,edge,a,b,type,payload,...options});}
  flow('client-router',0,.010,'req','Client → Router：R1 请求文本 / 采样参数');
  for(const r of ['p','d']){
    const side=r==='p'?'P':'D';
    flow('request-'+r,.010,.027,'req','Router → '+side+' HTTP：R1 / 配对信息');
    flow(r+'-http-tok',.027,.038,'req',side+' HTTP → TokenizerManager：请求文本');
    flow(r+'-ipc',.038,.070,'req',side+' TokenizerManager → Scheduler：输入 IDs / 参数');
  }
  flow('d-prealloc',.070,.093,'meta','D PreallocQueue → 内存池：预留接收位置',{mode:'reference'});
  flow('d-prepare-receiver',.093,.110,'meta','D 内存池 → Receiver：接收位置 / metadata 地址');
  flow('d-queue-step',.110,.128,'meta','D PreallocQueue → TransferQueue：进入传输等待');
  flow('receive-info',.110,.160,'meta','D Receiver → P BootstrapQueue：接收位置 / 配对身份');
  flow('p-queue-step',.160,.182,'req','P BootstrapQueue → waiting_queue：R1 就绪');
  flow('p-batch-cache',.182,.200,'meta','P 选批：查询前缀匹配',{mode:'reference'});
  flow('p-cache-ref',.182,.211,'meta','P tree_cache：引用 KV 映射与缓存池',{mode:'reference'});
  flow('p-queue-batch',.202,.222,'req','P waiting_queue → ScheduleBatch：选入 R1');
  flow('p-alloc',.211,.235,'meta','P 本地分配：请求映射 / KV 写入位置',{mode:'reference'});
  flow('p-dispatch',.235,.250,'act','P ScheduleBatch → TpModelWorker：IDs / 长度 / 写入位置');
  flow('p-forward',.250,.270,'act','P Worker → ModelRunner / Model：ForwardBatch');
  for(let i=0;i<4;i++){
    const a=.27+i*.0625;
    flow('p-kv-ref',a,a+.045,'kv','PP'+i+' Attention → 本地 KVCache：写本段各层 prompt KV');
    if(i<3)flow('pp'+(i+1),a+.050,a+.0625,'act','PP'+i+' → PP'+(i+1)+'：hidden_states / residual');
  }
  flow('p-logits',.502,.510,'act','PP3 Model → Sampler：logits');
  flow('p-result',.510,.520,'req','PP3 Sampler → Scheduler：首个生成 ID y1');
  flow('p-to-sender',.520,.536,'meta','P Scheduler → Sender：y1 / 请求身份 metadata');
  flow('p-kv-sender',.520,.536,'kv','P 各层 KVCache → Sender：prompt KV（逻辑视图）');
  flow('kv',.536,.599,'kv','P Sender → D Receiver：各层 prompt KV');
  flow('metadata',.549,.615,'meta','P Sender → D Receiver：y1 / 请求身份 metadata');
  flow('d-write-kv',.599,.630,'kv','D Receiver → 已预留 KV 位置：写入 prompt KV');
  flow('d-receiver-queue',.630,.678,'meta','D Receiver → TransferQueue：传输终态 / metadata 核验');
  flow('p-cache-ref',.635,.675,'meta','P 本地收尾：传输终态与 PP 协作后释放源资源',{mode:'reference'});
  flow('d-queue-batch',.678,.702,'req','D TransferQueue → waiting_queue / ScheduleBatch：R1 准入');
  flow('d-first-token',.680,.689,'req','D 请求结果：metadata 中的首个 ID y1');
  function stream(a,b,id){
    const step=(b-a)/5;
    flow('d-to-detok',a,a+step,'req','Scheduler → DetokenizerManager：输出 ID '+id);
    flow('d-return',a+step,a+step*2,'out','DetokenizerManager → TokenizerManager：'+id+' 对应文字');
    flow('d-tok-http',a+step*2,a+step*3,'out','TokenizerManager → HTTP：增量文字');
    flow('d-http-router',a+step*3,a+step*4,'out','D HTTP → Router：流式响应');
    flow('router-client',a+step*4,b,'out','Router → Client：'+id+' 对应文字');
  }
  stream(.689,.739,'y1');
  for(let i=0;i<2;i++){
    const a=.72+i*.09,n=i+1;
    flow('d-batch-cache',a,a+.009,'meta','D 第 '+n+' 轮：复用请求与缓存引用',{mode:'reference'});
    flow('d-cache-ref',a,a+.014,'meta','D tree_cache：保留已提交的历史 KV 引用',{mode:'reference'});
    flow('d-alloc',a,a+.014,'meta','D 第 '+n+' 轮：为输入 y'+n+' 分配一个新 KV 位置',{mode:'reference'});
    flow('d-dispatch',a+.014,a+.022,'act','D ScheduleBatch → Worker：本轮输入 y'+n+' / 写入位置');
    flow('d-forward',a+.022,a+.030,'act','D Worker → ModelRunner / Model：ForwardBatch');
    flow('d-kv-ref',a+.030,a+.041,'kv','D KVCache → Attention：读取历史 KV（本地）',{reverse:true});
    flow('d-kv-ref',a+.041,a+.049,'kv','D Attention → KVCache：写入 KV(y'+n+')（本地）');
    flow('d-logits',a+.049,a+.059,'act','D Model → Sampler：本轮 logits');
    flow('d-result',a+.059,a+.066,'req','D Sampler → Scheduler：采样得到 y'+(n+1));
    stream(a+.066,a+.106,'y'+(n+1));
  }
  flow('d-alloc',.930,.965,'meta','D Scheduler：结束请求并释放本请求槽位',{mode:'reference',reverse:true});
  function phase(){let i=0;while(i<8&&progress>=bounds[i+1])i++;return i;}
  function drawFlows(){
    const active=[];
    for(const e of events){
      const visual=scene.dots[e.id],edge=scene.edges[e.edge];if(!visual||!edge||progress<e.a||progress>=e.b)continue;
      const u=clamp((progress-e.a)/(e.b-e.a)),p=edge.p,len=p.getTotalLength(),distance=(e.reverse?1-u:u)*len;
      const q=p.getPointAtLength(distance),tail=Math.min(18,len);
      p.classList.add('active');visual.trail.style.display='';visual.trail.setAttribute('stroke',colors[e.type]);
      visual.trail.setAttribute('data-fraction',u.toFixed(4));
      if(e.mode==='reference'){
        visual.trail.setAttribute('stroke-dasharray','3 4');visual.trail.setAttribute('stroke-dashoffset',String(-u*14));
        visual.trail.setAttribute('opacity',String(.45+.5*Math.sin(Math.PI*u)));
      }else{
        visual.trail.setAttribute('stroke-dasharray',tail+' '+(len+tail));visual.trail.setAttribute('stroke-dashoffset',String(e.reverse?-distance:tail-distance));
        visual.g.style.display='';visual.g.setAttribute('transform',`translate(${q.x} ${q.y})`);
        visual.g.dataset.fraction=u.toFixed(4);
      }
      active.push(e);
    }
    root.dataset.activeFlows=active.map(e=>e.id).join(',');
    return active;
  }
  function render(){
    if(!scene.nodes?.pHttp)return;const pi=phase(),u=clamp((progress-bounds[pi])/(bounds[pi+1]-bounds[pi]));
    for(const n of Object.values(scene.nodes))n.g.classList.remove('active','done');for(const v of Object.values(scene.scopes))v.r.classList.remove('active');for(const e of Object.values(scene.edges))e.p.classList.remove('active');for(const d of Object.values(scene.dots)){d.g.style.display='none';d.trail.style.display='none';}
    const active=[];function on(...ids){for(const id of ids){scene.nodes[id]?.g.classList.add('active');scene.scopes[id]?.r.classList.add('active');active.push(id);}}
    const pCalls=[
      'HTTP Server → TokenizerManager → Scheduler',
      'DecodePreallocQueue → MooncakeKVReceiver.send_metadata()',
      'Scheduler → ScheduleBatch → TpModelWorker',
      'TpModelWorker.forward_batch_generation() → ModelRunner.forward()',
      'Scheduler.process_batch_result() → MooncakeKVSender',
      'DecodeTransferQueue → waiting_queue → ScheduleBatch',
      'ScheduleBatch.prepare_for_decode() → ModelRunner.forward()',
      'ScheduleBatch.prepare_for_decode() → ModelRunner.forward()',
      'Scheduler → DetokenizerManager → TokenizerManager → HTTP'
    ];
    const payloads=[
      '请求 IPC：输入 token IDs、采样参数、请求身份；返回 IPC：输出 IDs / 文字。',
      'D → P：接收 KV 位置、metadata 位置和配对身份。P 的 PrefillBootstrapQueue 等待该信息。',
      'ScheduleBatch → ForwardBatch：输入 IDs、长度、KV 写入位置。tree_cache 管前缀，内存池供引用。',
      'PP 级间：hidden_states / residual；各 rank 的 Attention 写本段各层 KV。末级 Sampler 选首个 ID。',
      'P → D：各层 prompt KV + metadata（首个生成 ID、请求身份）。Sender / Receiver 是 SRT 后端适配类。',
      'D：KV / metadata 核验后提交请求；P：传输终态与 PP 协作后回收源资源。两个本地条件独立。',
      '输入 y1 → Attention 读 prompt KV、写 KV(y1) → logits → Sampler 选 y2。',
      '输入 y2 → Attention 读历史 KV、写 KV(y2) → logits → Sampler 选 y3；随后停止。',
      '输出 token IDs 经 DetokenizerManager 变成文字，逐轮返回；结束时由 Scheduler 回收请求资源。'
    ];
    if(pi===0){on('pHttp','pTok','dHttp','dTok');}
    if(pi===1){on('dScheduler','dQueue','dMem','dAdapter','pQueue');}
    if(pi===2){on('pScheduler','pQueue2','pBatch','pCache','pMem','pWorker');}
    let stage=-1;scene.stages.forEach((q,i)=>{const a=.27+i*.0625,b=a+.0625,now=progress>=a&&progress<b,receiving=i>0&&progress>=a-.0125&&progress<a,done=progress>=b;q.g.classList.toggle('active',now||receiving);q.g.classList.toggle('done',done);q.state.textContent=receiving?'接收':now?(i<3&&progress>=a+.050?'传出':i===3&&progress>=.502?'采样':'计算'):done?'完成':'等待';if(now)stage=i;});
    if(pi===3){on('pWorker','pRunner','pModel','pMem');if(stage===3)on('pSampler');pCalls[3]='PP'+stage+' · '+pCalls[3];}
    if(pi===4){on('pScheduler','pResult','pAdapter','dAdapter','dMem','dQueue2');}
    if(pi===5){on('dScheduler','dQueue2');if(progress>=.678){on('dBatch','dDetok');}else{pCalls[5]='DecodeTransferQueue · 等待 KV / metadata 核验';}on('pResult');}
    if(pi===6||pi===7){const n=pi-5;on('dScheduler');if(u<.16){on('dBatch','dMem');pCalls[pi]='ScheduleBatch.prepare_for_decode() → alloc_for_decode(1)';}else if(u<.545){on('dWorker','dRunner','dModel','dMem');pCalls[pi]='TpModelWorker → ModelRunner.forward() → Model / Attention';}else if(u<.734){on('dSampler','dResult');pCalls[pi]='ModelRunner.sample() → Scheduler.process_batch_result()';}else{on('dResult','dDetok','dTok','dHttp');pCalls[pi]='Scheduler → DetokenizerManager → TokenizerManager';}}
    if(pi===8)on('dResult','dDetok','dTok','dHttp','router','client');
    const flows=drawFlows();
    text('op-status',pCalls[pi]);text('op-wire',flows.length?flows.map(e=>e.payload).join('；'):(progress>=.965?'R1 完成：y1、y2、y3 已依次返回；P/D 本请求资源已释放。':payloads[pi]));text('op-step',String(pi+1).padStart(2,'0')+' / 09 · '+names[pi]);
    const elapsed=timeAt(progress);text('op-clock',formatTime(progress>=1?Math.ceil(duration):elapsed)+' / '+formatTime(Math.ceil(duration)));
    slider.value=String(Math.round(elapsed/duration*1000));slider.setAttribute('aria-valuetext',names[pi]+'，'+formatTime(elapsed)+'，'+rate+' 倍速');speedControl.value=String(rate);button.textContent=playing?'暂停':progress>=1?'重播':progress>0?'继续':'播放';
    root.dataset.progress=progress.toFixed(4);root.dataset.elapsed=elapsed.toFixed(4);root.dataset.duration=duration.toFixed(4);root.dataset.rate=String(rate);root.dataset.playing=String(playing);root.dataset.phase=['request','reserve','prepare','prefill','handoff','commit','decode1','decode2','output'][pi];root.dataset.activeModules=active.join(',');root.dataset.stage=stage>=0?'PP'+stage:'';root.dataset.ready='true';
  }
  function save(){const p=Math.round(progress*1000)/1000,key=snapshotKey(p,rate);if(saved===key)return;saved=key;try{localStorage.setItem('ai-infra:sglang-pd-opening:v1',JSON.stringify({modelContent:{schema,progress:p,speed:rate}}));}catch(e){}}
  function pause(persist=true){playing=false;cancelAnimationFrame(raf);last=0;render();if(persist)save();}
  function tick(t){if(!playing)return;if(!last)last=t;progress=progressAt(timeAt(progress)+Math.max(0,(t-last)/1000)*rate);last=t;render();if(progress>=1){pause();return;}raf=requestAnimationFrame(tick);}
  function play(fromStart=false){cancelAnimationFrame(raf);if(fromStart||progress>=1)progress=0;playing=true;last=0;render();raf=requestAnimationFrame(tick);}
  button.addEventListener('click',()=>{if(playing){pause();return;}play();});
  restart.addEventListener('click',()=>play(true));
  slider.addEventListener('input',()=>{const next=Number(slider.value)/1000;pause(false);progress=progressAt(clamp(next)*duration);render();});slider.addEventListener('change',save);
  speedControl.addEventListener('change',()=>{const next=Number(speedControl.value);if(!rates.includes(next))return;rate=next;last=0;render();save();});
  function restore(state){const s=state?.modelContent;if(s?.schema!==schema||typeof s.progress!=='number'||!Number.isFinite(s.progress))return;playing=false;cancelAnimationFrame(raf);last=0;progress=clamp(s.progress);rate=rates.includes(s.speed)?s.speed:1;saved=snapshotKey(progress,rate);render();}
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&playing)pause();});
  const ro=new ResizeObserver(()=>build());ro.observe(svg);build();try{restore(JSON.parse(localStorage.getItem('ai-infra:sglang-pd-opening:v1')||'null'));}catch(e){}
})();
