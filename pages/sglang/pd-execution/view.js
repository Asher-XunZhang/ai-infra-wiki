/* Shared controls, separate lesson views. All drawing is a pure function of the clock. */
(() => {
  'use strict';
  const M=window.PDExecution,esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const names={none:'尚未接入',bootstrap:'bootstrap_queue',waiting:'waiting_queue',batch:'batch.reqs',chunked:'chunked_req',inflight:'inflight_queue',done:'本级已收尾',aborted:'本级已中止'};
  const colours={control:'var(--px-control)',compute:'var(--px-compute)',activation:'var(--px-compute)',proxy:'var(--px-compute)',output:'var(--px-output)',submit:'var(--px-kv)',kv:'var(--px-kv)',wait:'var(--muted)'};
  window.PDExecutionView={mount(host,options){
    let tl,t=0,playing=false,raf=0,last=0,rank=0,rid='R0';
    const reduced=matchMedia('(prefers-reduced-motion: reduce)');
    host.classList.add('px');
    host.innerHTML=`<div class="px-controls"><button data-action="reset">重置</button><button data-action="prev">上一步</button><button data-action="play" aria-pressed="false">播放全程</button><button data-action="next">下一步</button><label class="animation-speed">速度 <select data-role="speed"><option value="0.5">0.5×</option><option selected value="1">1×</option><option value="2">2×</option><option value="3">3×</option></select></label><label>关键动作 <select data-role="steps"></select></label></div>
    <label class="px-clock">讲解时间 <output data-role="time"></output><input data-role="seek" aria-label="同一时钟下的讲解时间" type="range" min="0" step="0.01" value="0"></label>
    <div class="px-current"><strong data-role="title"></strong><a data-role="source" target="_blank" rel="noreferrer">固定源码 ↗</a><p data-role="detail"></p></div>
    <div class="px-view-controls"><label>观察 Req <select data-role="request"></select></label><label>本地对象 <select data-role="rank"><option value="0">PP0</option><option value="1">PP1</option><option value="2">PP2</option></select></label><span>每级一个进程 / GPU；物理主机放置未指定</span></div>
    <div class="px-board" data-role="board"></div>
    <div class="px-scroll" tabindex="0" aria-label="载荷通路，窄屏可横向查看"><svg data-role="routes" viewBox="0 0 960 158" role="img" aria-label="PP 消息与各级 KV 向 Decode 的交接"></svg></div>
    <div data-role="references"></div>
    <div class="px-track-heading"><h3>同一时钟下的执行轨道</h3><label>时间窗口 <select data-role="window"><option value="follow">跟随当前 · 18 单位</option><option value="all">完整流程</option></select></label></div>
    <div class="px-scroll" tabindex="0" aria-label="执行轨道，窄屏可横向查看"><svg data-role="tracks" viewBox="0 0 960 268" role="img" aria-label="CPU、GPU 与传输 worker 的共同时间轴"></svg></div>
    <p class="px-key"><span>蓝：GPU / 激活</span><span>紫：结果回流</span><span>绿：控制 / 本地操作</span><span>金：KV / aux（首 token、room 等元数据）</span><span>斜线：等待条件；空白：未安排示例工作</span></p>
    <details class="px-evidence"><summary>本例的时序依据与简化边界</summary><p>固定 PP=3、pp_async_batch_depth=0，普通 CUDA / Mooncake 路径，关闭 staging 与乐观 Prefill。B1/B2/B3 是本次选批编号；同一请求的后续块依赖本地前块结果处理。轨道给出一种合法教学交错，不复刻真实 loop 空槽或硬件耗时。</p><p>本地选批与 KV 映射准备可先于上游激活到达，GPU 前向仍须等待对应激活。本图重点展开异步 KV worker：send 只提交任务，三个级的传输可以并行，前块搬运可与后块计算重叠。单个 worker 的任务串行；最终块依次传 KV、aux 并报告状态。普通 PP 的多请求 GPU 流水及 CPU 处理其他 batch 的重叠详见 <a href="../parallelism/?mode=pp#parallel-title">并行专题的 PP 执行轨道</a>，不由本页这条请求推算利用率。</p><p>已计算、已提交和 worker 完成分别记账。最终终态仍需采样、跨 PP 求交与本地复查；解除本请求的资源持有不表示清空物理缓存。Decode 接收匹配与准入继续见 <a href="../inference-overview/deployment.html#pd-handoff-title">P/D 交接入口</a>，此处不根据 P 清理时间推断 D 开算。</p><a data-role="worker-source" target="_blank" rel="noreferrer">transfer_worker 与任务队列源码 ↗</a></details>
    <span class="sr-only" role="status" aria-live="polite" data-role="status"></span>`;
    const $=name=>host.querySelector(`[data-role="${name}"]`),button=name=>host.querySelector(`[data-action="${name}"]`);
    if(options.kind==='dataflow'){
      const details=document.createElement('details');details.className='px-resource-detail';
      const summary=document.createElement('summary');summary.textContent='查看三个 PP 的 KV 边界、队列与资源（与主图同步）';details.append(summary);
      const board=$('board');board.before(details);details.append(board);
    }
    function pause(){playing=false;cancelAnimationFrame(raf);button('play').textContent=t>=tl.duration?'重播':'播放全程';button('play').setAttribute('aria-pressed','false');}
    function write(){options.onPosition?.({t,rank,rid,event:tl.sample(t).focus.id});}
    function seek(value,persist=true){t=Math.max(0,Math.min(tl.duration,Number(value)||0));pause();render();if(persist)write();}
    function advance(now){if(!playing)return;const delta=Math.min(0.1,(now-last)/1000);last=now;t=Math.min(tl.duration,t+delta*Number($('speed').value));render();if(t>=tl.duration){pause();write();return;}raf=requestAnimationFrame(advance);}
    function play(){if(playing){pause();write();return;}if(t>=tl.duration)t=0;playing=true;last=performance.now();button('play').textContent='暂停';button('play').setAttribute('aria-pressed','true');raf=requestAnimationFrame(advance);}
    function stateCards(s){
      const rows=M.rows(tl,s,rid),length=tl.kind==='lifecycle'?tl.model.config.inputLength:12;
      $('board').innerHTML=rows.map((q,r)=>{
        const active=s.active.filter(e=>e.rank===r&&!['link'].includes(e.lane));
        const pending=s.active.filter(e=>e.rank===r&&e.type==='kv');
        const cells=(count,type)=>`<span class="px-cells" aria-label="${type} ${count} / ${length}">${Array.from({length},(_,i)=>`<i class="${i<count?'filled':''}" title="位置 ${i}"></i>`).join('')}</span>`;
        return `<article class="px-stage ${r===rank?'selected':''}"><h3>PP${r} <small>本级模型层 / KV</small></h3><p class="px-queue">${esc(names[q.queue]??q.queue)}</p><dl><div><dt>GPU 已计算</dt><dd>[0, ${q.computed}) ${cells(q.computed,'已计算')}</dd></div><div><dt>CPU 已提交</dt><dd>[0, ${q.submitted}) ${cells(q.submitted,'已提交')}</dd></div><div><dt>worker 已完成</dt><dd>[0, ${q.completed}) ${cells(q.completed,'搬运完成')}</dd></div></dl><p class="px-ownership">Req KV：<b>${q.kv?'持有':'未持有'}</b> · metadata：${q.metadata?'有':'无'}</p><p class="px-poll">最近 poll：${esc(q.poll)}${q.token?' · 首 token 已追加':''}</p><div class="px-active">${active.length?active.map(e=>`<span style="--event-color:${colours[e.type]}">${esc(e.type==='kv'?'worker 搬运中':e.title)} <b>${Math.floor(s.progress(e)*100)}%</b></span>`).join(''):'本请求暂无新操作'}${pending.length?'<small>源 KV 保留；CPU 可继续其他工作</small>':''}</div></article>`;
      }).join('');
    }
    const line=(x1,y1,x2,y2,colour,dash='')=>`<path d="M${x1} ${y1} H${x2} V${y2}" fill="none" stroke="${colour}" stroke-width="2" ${dash?'stroke-dasharray="5 4"':''}/>`;
    function packet(x1,x2,y,e,s,label){const p=reduced.matches?.5:s.progress(e),x=x1+(x2-x1)*p;return `<g data-motion="${esc(e.id)}" data-x="${x.toFixed(2)}"><rect x="${x-5}" y="${y-5}" width="10" height="10" rx="2" fill="${colours[e.type]??colours.control}"/><title>${esc(label)}</title></g>`;}
    function routes(s){
      const x=[185,420,655],flow=s.active.find(e=>e.frame&&(e.frame.flow||e.frame.routes?.length)&&e.type!=='submit');
      let out='<text x="12" y="17">跨 PP：控制名单 / 激活 / 结果，按标签分辨</text><text x="860" y="17">Decode 接收</text>';
      if(flow){let from,to,label;
        if(tl.kind==='dataflow'){const f=flow.frame.flow;from=Number(f.from.match(/PP(\d)/)?.[1]??flow.rank);to=Number(f.to.match(/PP(\d)/)?.[1]??flow.rank);label=f.payload;}
        else{const f=flow.frame.routes[0];from=Number(f.from.match(/p(\d)/)?.[1]??flow.rank);to=Number(f.to.match(/p(\d)/)?.[1]??flow.rank);label={activation:'hidden_states / residual',proxy:'hidden_states / residual',output:'batch 对应结果',control:'rid 名单 / 请求信息'}[flow.type]??'控制消息';
          if(f.from==='decode-bootstrap')label='各 PP 观察 Decode 目的地址握手；WaitingForInput';
          else if(f.from==='frontend')label='TokenizerManager → PP0 · token IDs / rid';
        }
        if(tl.kind==='lifecycle'&&['frontend','decode-bootstrap'].includes(flow.frame.routes[0].from)){out+=`<text x="185" y="43">${esc(label)}</text>`;}
        else
        out+=line(x[from],43,x[to],43,colours[flow.type]??colours.control,flow.type==='control')+packet(x[from],x[to],43,flow,s,label)+`<text x="12" y="46">${esc(flow.type==='output'?'结果回流':flow.type==='activation'||flow.type==='proxy'?'激活前向':'控制流')}</text><text x="${Math.min(x[from],x[to])}" y="32">PP${from} → PP${to} · ${esc(label)}</text>`;
      }else out+='<text x="185" y="43" class="px-muted">本时刻无跨 PP 消息；本地引用变化不会走这条通路</text>';
      for(let r=0;r<3;r++){
        const y=76+r*33,e=s.active.find(e=>e.type==='kv'&&e.rank===r),queued=tl.copies.filter(c=>c.rank===r&&c.start>s.t&&tl.events.find(e=>e.id===c.deps[0]).end<=s.t);
        out+=line(185,y,895,y,'var(--border)')+`<text x="12" y="${y+4}">PP${r} · 层 KV</text><rect x="895" y="${y-8}" width="50" height="16" rx="3" fill="var(--soft)" stroke="var(--border)"/><text x="900" y="${y+4}">本级层</text>`;
        if(e){const payload=e.requests.map(q=>`${q.rid} [${q.start},${q.end})${q.last?' + aux':''}`).join(' · ');out+=line(185,y,895,y,colours.kv)+packet(185,895,y,e,s,payload)+`<text x="210" y="${y-8}">${esc(payload)} · worker 执行${queued.length?' / 后续任务已排队':''}</text>`;}
      }
      $('routes').innerHTML=out;
    }
    function references(s){
      if(tl.kind!=='dataflow'){$('references').innerHTML='';return;}
      const refs=M.references(tl,s,rank,rid),labels={bootstrap:'bootstrap_queue',waiting:'waiting_queue',batch:'ScheduleBatch.reqs',chunk:'chunked_req',inflight:'inflight_queue'};
      const boxes=Object.entries(refs).map(([key,on],i)=>{const x=10+i*190;return `<g data-ref="${key}" data-held="${on}"><rect x="${x}" y="5" width="180" height="27" rx="3"/><text x="${x+90}" y="23" text-anchor="middle">${labels[key]}</text>${on?`<path d="M${x+90} 32V50H480V68" fill="none" stroke="var(--accent)" stroke-width="1.5"/>`:''}</g>`;}).join('');
      $('references').innerHTML=`<section id="queue-lab" class="px-refs"><h3 id="queue-heading">PP${rank}：谁还引用同一个 Req？</h3><div class="px-scroll"><svg class="px-ref-svg" viewBox="0 0 960 103" role="img" aria-label="PP${rank} 本地容器到 Req ${esc(rid)} 的引用">${boxes}<rect x="345" y="68" width="270" height="29" rx="5"/><text x="480" y="87" text-anchor="middle">Req ${esc(rid)} · PP${rank} 进程本地对象</text></svg></div><p>实线表示本地引用，不传 KV；batch 与 inflight 可以同时引用 Req。旧 batch 后续过滤的具体时刻不在此时钟中，收尾不等于 Python 对象已回收。</p></section>`;
    }
    function tracks(s){
      let start=0,end=tl.duration;
      if($('window').value==='follow'&&end>18){start=Math.max(0,Math.min(end-18,s.t-5));end=start+18;}
      const x=time=>155+(time-start)/(end-start)*790;
      let out=`<defs><pattern id="px-hatch" width="6" height="6" patternUnits="userSpaceOnUse"><path d="M0 6L6 0" stroke="var(--muted)" opacity=".4"/></pattern><clipPath id="px-time-clip"><rect x="155" y="0" width="790" height="268"/></clipPath></defs>`;
      for(let i=0;i<=6;i++){const at=start+(end-start)*i/6;out+=`<text x="${x(at)}" y="12" text-anchor="${i===6?'end':'start'}">${at.toFixed(1)}</text><path d="M${x(at)} 20 V255" stroke="var(--border)"/>`;}
      M.lanes.forEach((lane,i)=>{const y=24+i*23;out+=`<text x="4" y="${y+13}">${lane.label}</text><path d="M155 ${y+20}H945" stroke="var(--border)"/>`;});
      out+='<g clip-path="url(#px-time-clip)">';
      for(const e of tl.events){if(e.end<start||e.start>end)continue;const i=M.lanes.findIndex(l=>l.id===e.lane),y=24+i*23,w=x(e.end)-x(e.start),active=s.t>=e.start&&s.t<e.end;out+=`<g data-event="${esc(e.id)}" data-active="${active}" role="button" tabindex="0" aria-label="${esc(e.title)}，${e.start.toFixed(1)} 至 ${e.end.toFixed(1)}，定位此操作"><title>${esc(e.title)} · ${e.start.toFixed(1)}–${e.end.toFixed(1)}</title><rect x="${x(e.start)}" y="${y+1}" width="${w}" height="17" rx="2" fill="${e.type==='wait'?'url(#px-hatch)':colours[e.type]??colours.control}" fill-opacity="${active?1:.48}" stroke="${active?'var(--fg)':'none'}"/>${w>60?`<text x="${Math.max(157,x(e.start)+4)}" y="${y+13}" class="px-bar-label">${esc(e.type==='kv'?e.id.match(/B\d+/)?.[0]+' KV':e.type==='compute'?e.id.match(/B\d+/)?.[0]+' GPU':e.type==='wait'?'等终态':'')}</text>`:''}</g>`;}
      out+=`<path d="M${x(s.t)} 18V257" stroke="var(--fg)" stroke-width="1.5"/></g>`;$('tracks').innerHTML=out;$('tracks').dataset.clock=t.toFixed(3);
    }
    function render(){const s=tl.sample(t);host.dataset.time=t.toFixed(3);$('seek').value=t;$('time').textContent=`${t.toFixed(1)} / ${tl.duration.toFixed(1)} · 教学单位，非实际耗时`;
      $('title').textContent=(s.active.length?'进行中 · ':'已完成 · ')+(s.focus.type==='compute'?`PP${s.focus.rank} 执行 ${s.focus.id.split('-')[0]} 本级前向`:s.focus.title);$('detail').textContent=s.focus.type==='compute'?'GPU 执行本级模型层，读取已有前缀并写入新 KV。下方已计算边界在该动作完成后更新；后一级前向还要等待这批激活。':s.focus.detail;$('source').href=tl.sourceUrl(s.focus);$('steps').value=s.focus.frame?s.focus.id:s.done.at(-1)?.id??tl.commits[0].id;
      stateCards(s);routes(s);references(s);tracks(s);button('prev').disabled=t===0;button('next').disabled=t>=tl.duration;
    }
    $('seek').oninput=e=>seek(e.target.value);button('play').onclick=play;button('reset').onclick=()=>seek(0);
    button('prev').onclick=()=>seek([...tl.commits].reverse().find(e=>e.end<t-.01)?.end??0);
    button('next').onclick=()=>seek(tl.commits.find(e=>e.end>t+.01)?.end??tl.duration);
    $('steps').onchange=e=>seek(tl.events.find(x=>x.id===e.target.value).start);
    $('rank').onchange=e=>{rank=Number(e.target.value);render();write();};$('request').onchange=e=>{rid=e.target.value;render();write();};$('window').onchange=render;
    $('tracks').onclick=e=>{const g=e.target.closest('[data-event]');if(g)seek(tl.events.find(x=>x.id===g.dataset.event).start);};
    $('tracks').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.target.closest('[data-event]')?.dispatchEvent(new MouseEvent('click',{bubbles:true}));}};
    document.addEventListener('visibilitychange',()=>{if(document.hidden){pause();write();}});window.addEventListener('pagehide',pause);reduced.addEventListener('change',render);
    function load(model,position={}){if(tl)pause();tl=M.build(model,options.kind);rid=position.rid??'R0';rank=Number(position.rank)||0;t=0;
      $('steps').innerHTML=tl.commits.map(e=>`<option value="${esc(e.id)}">${esc(e.title)}</option>`).join('');$('request').innerHTML=(model.requests??[{rid:'R0'}]).map(r=>`<option>${r.rid}</option>`).join('');if(!$('request').querySelector(`option[value="${rid}"]`)&&!(model.requests??[{rid:'R0'}]).some(r=>r.rid===rid))rid='R0';$('request').value=rid;$('rank').value=rank;
      $('seek').max=tl.duration;$('worker-source').href=tl.sourceUrl({source:'worker'});seek(position.t??tl.events.find(e=>e.id===position.event)?.end??0,false);host.dataset.ready='true';
    }
    return {load,pause,seek,get timeline(){return tl;}};
  }};
})();
