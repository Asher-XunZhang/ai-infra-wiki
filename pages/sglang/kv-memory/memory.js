/* A single local clock drives address transfers, KV writes and release states. */
(function () {
  'use strict';
  const M = window.KVMemoryModel, lab = document.querySelector('[data-memory-lab]');
  if (!M || !lab) return;
  const $ = selector => lab.querySelector(selector);
  const seek = $('#memory-seek'), speed = $('#memory-speed');
  const namespace = 'http://www.w3.org/2000/svg';
  const reducedQuery = matchMedia('(prefers-reduced-motion: reduce)');
  let mode = 'extend', time = 0, playing = false, raf = null, previousTick = null;
  let timeline, frame, ports = {}, lastScene = '', reduced = reducedQuery.matches;
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const rect = (x,y,w,h,cls='') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" class="${cls}"/>`;
  const text = (x,y,value,cls='') => `<text x="${x}" y="${y}" class="${cls}">${esc(value)}</text>`;
  const options = () => ({target: Number($('#extend-length').value), hit: Number($('#reuse-length').value)});
  function port(id,x,y,w,h,kind) { ports[id] = {x,y,w,h,kind}; }
  function row(id, x, y, width, length, state) {
    const data = state.rows[id], mapping = data?.mapping || [], released = data?.released;
    port('row-'+id,x,y,width,96,'row');
    const status = released ? '请求行已归还' : data?.releasing ? '结束中，等待归还行' : mapping.length ? '映射已建立' : mode==='reuse'?'仅放大前缀区段':'等待建立映射';
    let out = rect(x,y,width,96,'memory-row') + text(x+10,y+18,`${id} · ReqToTokenPool`, 'memory-heading') + text(x+width-145,y+18,status,'memory-small');
    const cw=Math.min(76,(width-20)/length);
    for(let p=0;p<length;p++) {
      const px=x+10+p*cw,slot=mapping[p],focus=state.lookup?.request===id&&state.lookup.p===p;
      port('map-'+id+'-'+p,px,y+43,cw-4,29,'mapping');
      out+=text(px+6,y+37,'p'+p,'memory-small');
      out+=`<g data-map="${id}:${p}" data-slot="${slot??''}" data-released="${!!released}">`+
        rect(px,y+43,cw-4,29,`memory-map ${slot!==undefined?'is-mapped':''} ${focus?'is-focus':''} ${released||data?.releasing?'is-inactive':''}`)+
        text(px+6,y+63,slot===undefined?'—':slot,'memory-number')+'</g>';
    }
    return out+text(x+10,y+87,mode==='reuse'?'格内是槽号；未显示的其余写地址由同一次分配准备':'p 是逻辑位置；格内数字是 KV 槽号，不是 K/V 数值','memory-tiny');
  }
  function sourceBox(state) {
    let out=rect(282,144,600,112,'memory-operation');port('operation',282,144,600,112,'operation');
    if(state.result) {
      out+=text(296,169,mode==='reuse'?'地址复用结果':'Attention backend · 本层读到的结果','memory-heading');
      out+=text(296,199,state.result,'memory-result');
      out+=text(296,224,'同一槽号在不同模型层对应各层自己的 K / V。','memory-small');
    }else if((mode==='extend'&&frame.phase.index===3)||(mode==='reuse'&&frame.phase.index===2&&state.hit)) {
      out+=text(296,169,'Attention backend · 根据请求映射读取','memory-heading');
      const lookup=state.lookup,reached=frame.local>=(mode==='reuse'?1.15:1.2);
      out+=text(296,197,lookup?`req_to_token[${lookup.request}][p${lookup.p}] → 槽 ${lookup.slot}`:'先查请求的逻辑位置','memory-number');
      out+=text(296,222,reached&&lookup?`第 ℓ 层：Kℓ[${lookup.slot}] / Vℓ[${lookup.slot}]`:'沿蓝色地址线定位，随后读取橙色 K/V','memory-small');
      out+=text(296,244,'这里只放大一个读取位置；完整 Attention 会访问所需的历史 KV。','memory-tiny');
    }else if(mode==='reuse') {
      out+=text(296,167,'RadixCache 已返回 prefix_indices（搜索过程见上一节）','memory-heading');
      out+=text(296,196,'['+[8,9,10,11,20,21,22,23].slice(0,state.hit).join(', ')+']','memory-number');
      out+=text(296,222,`已有 ${state.hit} 个位置可复用 · 其余 ${9-state.hit} 个位置需新地址`,'memory-small');
      out+=text(296,242,'移动的是索引；现有 K/V 不复制到请求行。','memory-small');
    }else if(mode==='sharing'&&frame.phase.index<7) {
      out+=text(296,167,'缓存层与资源释放的边界','memory-heading');
      const active=Object.entries(state.rows).filter(([,r])=>!r.released&&!r.releasing).map(([id])=>id);
      out+=text(296,193,state.pages[0].owner ? `页 2 由缓存保留 · ${active.length?active.join('、')+' 正在引用':'当前无活跃请求引用'}`:'缓存层已通知释放页 2','memory-number');
      out+=text(296,220,'请求结束归还私有尾部；共享前缀由缓存层决定何时释放。','memory-small');
      out+=text(296,242,'分配器管理可用页号，不自行做前缀搜索或节点锁判断。','memory-small');
    }else {
      out+=text(296,165,'第 ℓ 层 · set_kv_buffer(layer, loc, cache_k, cache_v)','memory-heading');
      const addresses = mode==='extend'?M.extend(state.target,1).added:[8];
      out+=text(296,184,'loc = '+(state.outLoc.length?'['+state.outLoc.join(', ')+']':'尚未返回'),'memory-small');
      const ordered=addresses.slice().sort((a,b)=>a-b),w=Math.min(104,560/ordered.length);
      ordered.forEach((slot,i)=>{
        const x=296+i*w,label=mode==='sharing'?'Z':String(addresses.indexOf(slot)+6);
        port('payload-'+slot,x,196,w-8,43,'payload');
        out+=`<g data-payload="${slot}" class="${state.outLoc.length?'':'is-inactive'}">`+rect(x,196,w-8,43,'memory-payload')+
          text(x+8,213,'loc '+slot,'memory-tiny')+text(x+8,231,'K'+label+' / V'+label,'memory-number')+'</g>';
      });
    }
    return out;
  }
  function drawScene() {
    const s=frame.state; ports={};
    let out='';
    if(mode==='sharing'&&frame.phase.index<7) out+=row('R1',18,8,425,6,s)+row('R2',457,8,425,6,s);
    else out+=row(mode==='extend'?'R1':mode==='reuse'?'R2':'R3',18,8,864,mode==='extend'?s.target:mode==='reuse'?8:1,s);
    port('allocator',18,144,244,112,'allocator');
    out+=rect(18,144,244,112,'memory-allocator');
    out+=text(30,165,mode==='reuse'?'复用前后的存储':'PagedTokenToKVPoolAllocator','memory-heading');
    if(mode==='reuse') {
      out+=text(30,196,'历史 K/V：仍为 8 个位置','memory-number');
      out+=text(30,222,'新增历史 KV 副本：0','memory-small');
      out+=text(30,244,'缓存页保持固定','memory-small');
    }else {
      out+=text(30,189,'可分配页号','memory-small');
      out+=text(30,212,'['+s.freePages.join(', ')+']','memory-number');
      out+=text(30,233,`图示可分配：${s.freePages.length*4} 个位置`,'memory-small');
      out+=text(30,249,mode==='sharing'?`可用请求行：${s.freeRows}（与 KV 页分开归还）`:'其他页省略，本例无额外可用页','memory-tiny');
    }
    out+=sourceBox(s);
    out+=text(18,424,'GPU 静态 KV buffer · 仅放大第 ℓ 层的这些物理页','memory-heading');
    const pageWidth=864/s.pages.length-12;
    s.pages.forEach((page,i)=>{
      const x=18+i*(864/s.pages.length),y=300;
      port('page-'+page.id,x,y,pageWidth,112,'page');
      const owner=page.owner===null?'可重新分配':page.owner==='cache'?'缓存持有':page.owner+' 持有';
      out+=`<g data-page="${page.id}" data-owner="${page.owner??'free'}">`+rect(x,y,pageWidth,112,'memory-page '+(!page.owner?'is-free':''))+
        text(x+10,y+20,'页 '+page.id,'memory-heading')+text(x+pageWidth-106,y+20,owner,'memory-small');
      const cw=(pageWidth-20)/4;
      page.slots.forEach((cell,j)=>{
        const px=x+10+j*cw,py=y+31,lookup=s.lookup?.slot===cell.slot;
        port('slot-'+cell.slot,px,py,cw-5,65,'slot');
        const type=cell.valid?'written':cell.reserved?'reserved':page.owner?'held':'free';
        const payload=cell.valid?'K'+cell.value+' / V'+cell.value:cell.reserved?'待写入':cell.value!==null?'旧数据':'未使用';
        out+=`<g data-slot-id="${cell.slot}" data-state="${type}" data-value="${cell.value??''}">`+
          rect(px,py,cw-5,65,`memory-slot ${type} ${lookup?'is-focus':''}`)+text(px+6,py+17,'槽 '+cell.slot,'memory-tiny')+
          text(px+6,py+39,payload,'memory-kv')+text(px+6,py+55,cell.valid?'有效 KV':cell.reserved?'地址已预留':page.owner?'页内空位':'不可读取','memory-tiny')+'</g>';
      });
      out+='</g>';
    });
    $('#memory-scene-layer').innerHTML=out;
    $('#memory-caption').textContent=frame.event.label;
    const occupied=s.pages.filter(p=>p.owner).length;
    $('#memory-ledger').textContent=`图示物理页 ${s.pages.length}（固定） · 持有 ${occupied} · 可分配 ${s.freePages.length} · 持有页内未用位置 ${s.heldEmpty}`;
    const source=M.sources[frame.event.source];
    $('#memory-source').href=`https://github.com/sgl-project/sglang/blob/${M.revision}/${source[0]}#L${source[1]}`;
    $('#memory-source').textContent=`${source[0].split('/').pop()}:${source[1]} · ${source[2]} ↗`;
    $('#memory-why').textContent=mode==='extend'
      ?`本例是同一请求已有 6 个 KV 位置的续写。新增整页数 = ceil(${s.target}/4) − ceil(6/4) = ${s.target>8?1:0}。alloc_extend 返回 out_cache_loc，write_cache_indices 再写请求映射；set_kv_buffer 写入本层 K/V。尾页和新页的写入属于同一批散写，不表示两次顺序 kernel。`
      :mode==='reuse'?'匹配结果来自上一节的 RadixCache；这里仅放大请求映射的前缀区段。普通路径的 write_cache_indices 在同一次分配中写入前缀及其余新地址，后者在本图省略。索引可复制，已有 K/V 不因此复制。'
      :'本例每条请求已计算 6 个位置，其中 4 个为缓存前缀、2 个为私有尾部。cache_finished_req 处理尾部并解除保护，release_kv_cache 随后归还请求行。缓存何时允许驱逐由缓存层决定。图示分配器采用直接回收页号的路径；归还后旧字节可保留，但不能继续作为请求的有效 KV。';
  }
  function route(from,to) {
    const a=ports[from],b=ports[to];if(!a||!b)return null;
    const ac=a.x+a.w/2,bc=b.x+b.w/2;
    if((a.kind==='mapping'||a.kind==='row')&&(b.kind==='slot'||b.kind==='page'))return [[ac,a.y+a.h],[ac,122],[890,122],[890,276],[bc,276],[bc,b.y]];
    if(b.kind==='slot')return [[ac,a.y+a.h],[ac,273],[bc,273],[bc,b.y]];
    if(a.kind==='slot'||a.kind==='page')return [[ac,a.y],[ac,273],[bc,273],[bc,b.y+b.h]];
    if(a.kind==='row')return [[ac,a.y+a.h],[ac,123],[bc,123],[bc,b.y]];
    if(b.kind==='row')return [[ac,a.y],[ac,123],[bc,123],[bc,b.y+b.h]];
    return [[ac,a.y],[ac,123],[bc,123],[bc,b.y]];
  }
  function pointAt(points,u) {
    const lengths=points.slice(1).map((p,i)=>Math.hypot(p[0]-points[i][0],p[1]-points[i][1]));
    let distance=lengths.reduce((a,b)=>a+b,0)*u;
    for(let i=0;i<lengths.length;i++) {
      if(distance<=lengths[i]) {const f=lengths[i]?distance/lengths[i]:0;return [points[i][0]+(points[i+1][0]-points[i][0])*f,points[i][1]+(points[i+1][1]-points[i][1])*f];}
      distance-=lengths[i];
    }
    return points.at(-1);
  }
  function drawFlows() {
    const layer=$('#memory-flow-layer');layer.replaceChildren();
    const paths=document.createElementNS(namespace,'g'),packets=document.createElementNS(namespace,'g');
    layer.append(paths,packets);
    for(const flow of frame.flows) {
      const points=route(flow.from,flow.to);if(!points)continue;
      const path=document.createElementNS(namespace,'path');
      path.setAttribute('d',points.map((p,i)=>(i?'L':'M')+p.join(',')).join(' '));
      path.setAttribute('class','memory-flow-line '+flow.kind);path.setAttribute('marker-end','url(#memory-arrow-'+flow.kind+')');path.dataset.from=flow.from;path.dataset.to=flow.to;paths.append(path);
      const labelWidth=Array.from(flow.label).reduce((n,c)=>n+(c.charCodeAt(0)<128?6:10),14);
      const [x,y]=pointAt(points,reduced?.5:flow.progress),width=Math.min(220,Math.max(56,labelWidth));
      const packet=document.createElementNS(namespace,'g');packet.dataset.packet=flow.label;
      // Keep the packet's label in the canvas even on the outer lookup rail.
      packet.setAttribute('transform',`translate(${Math.max(width/2+3,Math.min(917-width/2,x))},${y})`);packet.setAttribute('class','memory-packet '+flow.kind);
      packet.setAttribute('opacity',reduced?1:Math.min(1,flow.progress*8,(1-flow.progress)*8));
      packet.innerHTML=rect(-width/2,-11,width,22)+text(-width/2+7,4,flow.label,'memory-packet-label');packets.append(packet);
    }
  }
  function render(force=false) {
    reduced=reducedQuery.matches;
    frame=M.sample(timeline,time);lab.dataset.time=time.toFixed(3);lab.dataset.mode=mode;lab.dataset.phase=frame.phase.index;lab.dataset.event=frame.event.at;lab.dataset.playing=String(playing);
    const key=[mode,timeline.target,timeline.hit,frame.phase.index,frame.event.at].join('/');
    if(force||key!==lastScene){lastScene=key;drawScene();}
    drawFlows();
    seek.max=timeline.duration;seek.value=time;seek.setAttribute('aria-valuetext',`${time.toFixed(1)} 秒，${frame.event.label}`);
    $('#memory-time').textContent=`${time.toFixed(1)} / ${timeline.duration.toFixed(1)} s`;
    const status=`${frame.phase.index+1} / ${timeline.phases.length}`;
    if($('#memory-status').textContent!==status)$('#memory-status').textContent=status;
    $('#memory-play').textContent=playing?'暂停':time>=timeline.duration?'重新播放':'播放全程';$('#memory-play').setAttribute('aria-pressed',String(playing));
    $('#memory-previous').disabled=time===0;$('#memory-next').disabled=time>=timeline.duration;
    $('#memory-steps').querySelectorAll('button').forEach(b=>b.setAttribute('aria-current',Number(b.dataset.step)===frame.phase.index?'step':'false'));
  }
  function save(){const u=new URL(location.href);u.searchParams.set('mode',mode);u.searchParams.set('t',time.toFixed(3));u.searchParams.set('target',timeline.target);u.searchParams.set('hit',timeline.hit);for(const k of ['mv','ms','fv','fs'])u.searchParams.delete(k);history.replaceState(null,'',u);}
  function stop(){playing=false;previousTick=null;if(raf!==null)cancelAnimationFrame(raf);raf=null;render();save();}
  function jump(t){time=Math.max(0,Math.min(timeline.duration,t));stop();}
  function tick(now){if(!playing)return;if(previousTick!==null)time=Math.min(timeline.duration,time+(now-previousTick)/1000*Number(speed.value));previousTick=now;if(time>=timeline.duration)playing=false;render();if(playing)raf=requestAnimationFrame(tick);else{raf=null;previousTick=null;save();}}
  function play(){if(playing)return stop();if(time>=timeline.duration)time=0;playing=true;previousTick=null;render();raf=requestAnimationFrame(tick);}
  function rebuild(){timeline=M.timeline(mode,options());lab.querySelectorAll('button[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode)));lab.querySelectorAll('[data-settings]').forEach(e=>e.hidden=e.dataset.settings!==mode);$('#memory-steps').innerHTML=timeline.phases.map(p=>`<button type="button" data-step="${p.index}">${p.index+1} · ${p.name}</button>`).join('');$('#memory-steps').querySelectorAll('button').forEach(b=>b.onclick=()=>jump(timeline.phases[Number(b.dataset.step)].end));render(true);}
  function reset(){stop();time=0;rebuild();save();}
  $('#memory-play').onclick=play;$('#memory-reset').onclick=reset;seek.oninput=()=>jump(Number(seek.value));speed.onchange=()=>{previousTick=null;};
  $('#memory-previous').onclick=()=>jump(frame.phase.index?timeline.phases[frame.phase.index-1].end:0);
  $('#memory-next').onclick=()=>jump(timeline.phases[Math.min(timeline.phases.length-1,frame.phase.index+1)].end);
  lab.querySelectorAll('button[data-mode]').forEach(b=>b.onclick=()=>{if(b.dataset.mode===mode)return;stop();mode=b.dataset.mode;time=0;rebuild();save();});
  lab.querySelectorAll('[data-settings] select').forEach(e=>e.onchange=reset);
  lab.addEventListener('keydown',e=>{if(/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)||e.metaKey||e.ctrlKey||e.altKey)return;if(e.key==='ArrowRight'){e.preventDefault();$('#memory-next').click();}else if(e.key==='ArrowLeft'){e.preventDefault();$('#memory-previous').click();}else if(e.key==='Home'){e.preventDefault();jump(0);}else if(e.key==='End'){e.preventDefault();jump(timeline.duration);}});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});window.addEventListener('pagehide',stop);
  reducedQuery.addEventListener('change',e=>{reduced=e.matches;render();});
  function load(){const p=new URL(location.href).searchParams;playing=false;previousTick=null;if(raf!==null)cancelAnimationFrame(raf);mode=['extend','reuse','sharing'].includes(p.get('mode'))?p.get('mode'):'extend';if([8,9,11].includes(Number(p.get('target'))))$('#extend-length').value=p.get('target');if([0,4,8].includes(Number(p.get('hit')))&&p.has('hit'))$('#reuse-length').value=p.get('hit');time=0;rebuild();time=Math.max(0,Math.min(timeline.duration,Number(p.get('t'))||0));render(true);}
  window.addEventListener('popstate',load);lab.hidden=false;load();
})();
