(() => {
  'use strict';
  const data = window.PP_QUICK_DATA;
  const $ = s => document.querySelector(s);
  const names = ['检查状态', '缓存与选批', '输入 / 提交前向', '结果与共识', '转发保存'];
  const lifeNames = ['准入与排队', '准备缓存 / 输入', '本级 GPU 前向', '等结果 / 处理', 'KV 传输', '等共识 / 释放'];
  const total = Math.ceil(data.end / 10) * 10;
  let mode = 'loop', rank = 0, loop = 5, batch = 1, lifePhase = 2;
  let start = 0, span = total;
  const svg = $('#chart'), NS = 'http://www.w3.org/2000/svg';
  const clamp = (n, a, b) => Math.max(a, Math.min(b, n));
  const fmt = n => n.toFixed(2);
  const currentLoop = () => data.loops.find(l => l.r === rank && l.n === loop);
  const currentBatch = () => data.batches.find(b => b.batch === batch);
  const batchName = n => n ? 'M' + n : '无';
  function html(parent, tag, text, className) {
    const e = document.createElement(tag);
    if (text !== undefined) e.textContent = text;
    if (className) e.className = className;
    parent.append(e); return e;
  }
  function button(parent, text, action) {
    const b = html(parent, 'button', text); b.type = 'button'; b.addEventListener('click', action); return b;
  }
  function el(parent, tag, attrs, text) {
    const e = document.createElementNS(NS, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => e.setAttribute(k, v));
    if (text !== undefined) e.textContent = text;
    parent.append(e); return e;
  }
  function label(parent, x, y, text, maxWidth, anchor = 'middle') {
    const t = el(parent, 'text', {x, y, 'text-anchor': anchor}, text);
    if (maxWidth !== undefined && t.getComputedTextLength() > maxWidth) t.remove();
    return t;
  }
  function selectable(g, action, name) {
    g.setAttribute('class', 'block'); g.setAttribute('role', 'button');
    g.setAttribute('tabindex', '0'); g.setAttribute('aria-label', name);
    g.addEventListener('click', action);
    g.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); action(); } });
    el(g, 'title', {}, name);
  }
  function setView(a, b) {
    span = clamp(b, 2, total); start = clamp(a, 0, total - span); draw();
  }
  function fitRange(a, b) { const pad = Math.max(1, (b-a) * .09), lo = Math.max(0,a-pad), hi = Math.min(total,b+pad); setView(lo,hi-lo); }
  function focus() {
    if (mode === 'loop') { const l = currentLoop(); fitRange(l.start, l.end); }
    else { const phases = currentBatch().ranks.map(r => r.phases[lifePhase]); fitRange(Math.min(...phases.map(p => p.start)), Math.max(...phases.map(p => p.end))); }
  }
  function fitBatch() {
    const rs = currentBatch().ranks;
    fitRange(Math.min(...rs.map(r => r.phases[0].start)), Math.max(...rs.map(r => r.phases[5].end)));
  }
  function goLoop(r, n) { rank = r; loop = n; mode = 'loop'; render(); focus(); }
  function goBatch(b) { batch = b; mode = 'batch'; render(); fitBatch(); }
  function selectLife(i) {
    lifePhase = i; render();
    const ps = currentBatch().ranks.map(r => r.phases[i]);
    if (Math.min(...ps.map(p=>p.start)) < start || Math.max(...ps.map(p=>p.end)) > start+span) focus();
  }
  function macroTexts(l) {
    return [
      '检查候选请求是否准备好，轮询历史 KV 传输状态。对象是候选请求集合和历史传输中的请求，不专属于某一份当前 batch。',
      '处理共享 L2 缓存完成通知（ACK）；' + (l.current ? '为 M' + l.current + ' 选批、准备缓存。' + ([2,4].includes(l.current) ? '本例有 host hit，需要安排 host → GPU 回载。' : '本例该 batch 无 host hit。') : '检查队列，本轮没有选到新的 batch。'),
      l.current ? '准备 M' + l.current + ' 的输入，必要时等上级激活及历史发送，再提交本级 forward。这里只是 CPU 提交，GPU 起止时间请看生命周期视角。' : '本轮不提交 forward；若仍有历史激活发送 work，会在这里处理。',
      '推进 output 回流、bootstrap / release 共识。' + (l.old ? '接收并处理旧 M' + l.old + ' 的结果，提交它的本级 KV 到 Decode。' : '本轮没有旧 batch 要做最终结果处理。') + (l.r === 2 && l.current ? '末级还会推进当前 M' + l.current + ' 的 output 发送。' : '') + (l.released.length ? '本轮实际清理：' + l.released.map(batchName).join('、') + '，与旧结果对象分开看。' : '本轮没有实际释放的 batch。'),
      (l.r < 2 ? '向后转发请求和状态；' + (l.current ? '安排 M' + l.current + ' 激活发送；' : '') : '末级不再向后发送激活；') + '保存本轮状态，进入下一次 loop。'
    ];
  }
  function renderLoop() {
    const l = currentLoop();
    $('#selection-title').textContent = 'PP' + rank + ' · L' + (loop+1) + '：提交 ' + batchName(l.current) + '，处理旧结果 ' + batchName(l.old);
    $('#selection-text').textContent = '槽位 s' + l.slot + ' · ' + fmt(l.start) + '–' + fmt(l.end) + ' u。本轮实际释放：' + (l.released.length ? l.released.map(batchName).join('、') : '无') + '。状态与缓存队列也在这一轮继续推进。';
    const links = $('#cross-links'); links.replaceChildren();
    [...new Set([l.current, l.old, ...l.released].filter(Boolean))].forEach(b => button(links, '跟踪 M' + b + ' 的生命周期', () => goBatch(b)));
    $('#chart-hint').textContent = '每个外框 = 本级的一次 CPU loop；里面横向拼接五类大步骤。点击外框查看对象与说明。宽度仍按示意时间占位，放大后可读短条块。';
    $('#detail-title').textContent = '这一轮的五个大步骤';
    $('#detail-intro').textContent = '按 CPU 执行顺序阅读。跨组的对象可以不同；GPU 异步计算并未被塞进 CPU 提交条块。';
    const detail = $('#detail'); detail.replaceChildren(); detail.className = 'macro-steps';
    const descriptions = macroTexts(l);
    l.groups.forEach((g, i) => {
      const card = html(detail, 'article', undefined, 'macro-step'); card.style.setProperty('--phase', 'var(--p' + i + ')');
      html(card, 'h3', (i+1) + ' · ' + names[i]);
      html(card, 'div', g ? fmt(g.start) + '–' + fmt(g.end) + ' u' : '本轮跳过', 'time');
      html(card, 'p', descriptions[i]);
      if (g && g.wait > 0) html(card, 'p', '含模型中的依赖等待 ' + fmt(g.wait) + ' u。', 'time');
    });
  }
  function lifecycleText(r) {
    const p = r.phases[lifePhase];
    if (lifePhase === 0) return 'M1–M5 共同推进 bootstrap；本级在 ' + fmt(r.admitted) + ' u 将这批请求准入 waiting queue。M' + batch + ' 随后排队，到 ' + fmt(p.end) + ' u 开始选批。这不是它独占调度线程。';
    if (lifePhase === 1) return 'L' + (r.currentLoop+1) + ' 准备 M' + batch + '，' + ([2,4].includes(batch) ? '本例包含 host hit 回载，' : '') + (r.r ? '等待上级激活及输入就绪，' : '准备首级输入，') + '直至 GPU 能开始本级前向。区间也包含事件 / 资源等待。';
    if (lifePhase === 2) return 'M' + batch + ' 在 PP' + r.r + ' 的 GPU 上执行本级模型层。对应提交来自 L' + (r.currentLoop+1) + '；实际计算可跨 CPU loop 边界。' + (r.r < 2 ? '后续激活发送 / 接收完成，才能接上下一流水级的前向。' : '末级前向完成后，相关 output 沿结果路径回流。');
    if (lifePhase === 3) return '本级前向已完成，等待流水线结果与后续调度推进；到 L' + (r.resultLoop+1) + '，M' + batch + ' 作为旧 batch 被处理并提交 KV。这一整段包含等待，不是持续进行 CPU 后处理。';
    if (lifePhase === 4) return '本级 M' + batch + ' 的 KV 已提交给传输后端，处于发往图外 Decode 的在途区间。KV 在途结束与本级引用清理仍是两件事。';
    return '本级 KV 传输完成后，继续等待终态名单和 release 共识推进；在 L' + (r.releaseLoop+1) + ' 的 ' + fmt(r.releaseStart) + '–' + fmt(p.end) + ' u 执行本地清理。这里只标本级请求引用 / 发送端清理，不宣称回收全部缓存物理页。';
  }
  function renderBatch() {
    const b = currentBatch();
    $('#selection-title').textContent = 'M' + batch + '：从请求准入到三个 Prefill 流水级完成清理';
    $('#selection-text').textContent = '共同准入 / 排队 → 准备输入 → PP0、PP1、PP2 逐级前向 → 各级等结果并处理 → KV 发送 → 各级等待共识并释放。选择一种阶段，比较三个流水级。';
    $('#cross-links').replaceChildren();
    $('#chart-hint').textContent = '每行只跟踪同一份 M' + batch + '。色块表示生命周期区间（含等待），不是 CPU 独占；③ 才是本级 GPU 前向。虚线概括跨级激活接续，Decode 内部不展开。';
    $('#detail-title').textContent = (lifePhase+1) + ' · ' + lifeNames[lifePhase] + '：三个流水级分别在哪里？';
    $('#detail-intro').textContent = '点击色块或选择阶段可切换；也可以跳回对应的本地 loop，看看同一轮还在处理谁。';
    const detail = $('#detail'); detail.replaceChildren(); detail.className = 'rank-details';
    b.ranks.forEach(r => {
      const p = r.phases[lifePhase], card = html(detail, 'article', undefined, 'rank-detail');
      card.style.borderColor = 'var(--p' + lifePhase + ')';
      html(card, 'h3', 'PP' + r.r + ' · M' + batch);
      html(card, 'div', fmt(p.start) + '–' + fmt(p.end) + ' u', 'time');
      html(card, 'p', lifecycleText(r));
      button(card, '查看 PP' + r.r + ' 的 L' + (p.loop+1), () => goLoop(r.r, p.loop));
    });
  }
  function render() {
    document.querySelectorAll('[data-mode]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
    $('#loop-controls').hidden = mode !== 'loop'; $('#batch-controls').hidden = mode !== 'batch';
    $('#rank').value = rank; $('#loop').value = loop; $('#batch').value = batch; $('#life-phase').value = lifePhase;
    mode === 'loop' ? renderLoop() : renderBatch();
    const legend = $('#legend'); legend.replaceChildren();
    (mode === 'loop' ? names : lifeNames).forEach((name, i) => {
      const item = html(legend, 'span'); const swatch = html(item, 'i', undefined, 'swatch'); swatch.style.background = 'var(--p' + i + ')'; swatch.setAttribute('aria-hidden', 'true');
      item.append(document.createTextNode((i+1) + ' ' + name));
    });
    draw();
  }
  function draw() {
    const w = Math.floor(svg.getBoundingClientRect().width), left = 58, right = 12, top = 48, row = 116, h = top + 3*row + 24;
    if (w <= left + right) return;
    const end = start+span, x = t => left+(t-start)/span*(w-left-right);
    svg.replaceChildren(); svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h); svg.setAttribute('height', h);
    svg.dataset.start = start; svg.dataset.end = end; svg.dataset.mode = mode;
    label(svg, left, 15, '示意时间 u →', undefined, 'start');
    const raw = span/(w<500?3:8), power = 10**Math.floor(Math.log10(raw)), step = [1,2,5,10].find(n => n*power>=raw)*power;
    for (let t = Math.ceil(start/step)*step; t<=end+.0001; t+=step) {
      const px = x(t); el(svg, 'line', {x1:px,x2:px,y1:top-6,y2:h-20,stroke:'var(--border)','stroke-width':.7});
      label(svg,px,top-15,String(Number(t.toFixed(2))),undefined,px>w-28?'end':px<left+10?'start':'middle');
    }
    for (let r=0;r<3;r++) {
      const y = top+r*row;
      label(svg,0,y+37,'PP'+r,undefined,'start');
      el(svg,'line',{x1:left,x2:w-right,y1:y+100,y2:y+100,stroke:'var(--border)'});
      if (mode === 'loop') {
        data.loops.filter(l => l.r===r && l.end>start && l.start<end).forEach(l => {
          const xx=x(Math.max(start,l.start)), ww=x(Math.min(end,l.end))-xx, selected=l.r===rank&&l.n===loop;
          const g=el(svg,'g',{'data-loop':r+':'+l.n});
          selectable(g,()=>{rank=l.r;loop=l.n;render();},'PP'+r+' L'+(l.n+1)+'，提交'+batchName(l.current)+'，处理旧结果'+batchName(l.old)+'，'+fmt(l.start)+'至'+fmt(l.end)+' u');
          el(g,'rect',{x:xx,y:y+3,width:ww,height:83,fill:'var(--surface)',stroke:selected?'var(--fg)':'var(--border)','stroke-width':selected?2:1});
          const cx=xx+ww/2;
          label(g,cx,y+22,'L'+(l.n+1),ww-4);
          label(g,cx,y+43,'提交 '+batchName(l.current)+' / 收 '+batchName(l.old),ww-8);
          l.groups.forEach((p,i)=>{
            if(!p||p.end<=start||p.start>=end)return;
            const px=x(Math.max(start,p.start)),pw=x(Math.min(end,p.end))-px;
            el(g,'rect',{x:px,y:y+54,width:pw,height:29,fill:'var(--p'+i+')','fill-opacity':selected?.4:.2,'data-group':i});
            const words=['1 状态','2 缓存选批',l.current?'3 提交前向':'3 回收发送','4 结果 / 共识','5 转发'];
            const t=label(g,px+pw/2,y+73,words[i],pw-6);
            if(!t.isConnected)label(g,px+pw/2,y+73,String(i+1),pw-4);
          });
        });
      } else {
        const rdata=currentBatch().ranks[r];
        rdata.phases.forEach((p,i)=>{
          if(p.end<=start||p.start>=end)return;
          const xx=x(Math.max(start,p.start)),ww=x(Math.min(end,p.end))-xx;
          const g=el(svg,'g',{'data-life-rank':r,'data-life-phase':i});
          selectable(g,()=>selectLife(i),'PP'+r+' M'+batch+' '+lifeNames[i]+' '+fmt(p.start)+'至'+fmt(p.end)+' u');
          el(g,'rect',{x:xx,y:y+24,width:ww,height:44,fill:'var(--p'+i+')','fill-opacity':i===lifePhase?.42:.2,stroke:i===lifePhase?'var(--fg)':'none','stroke-width':1.5});
          const full=(i+1)+' '+lifeNames[i],t=label(g,xx+ww/2,y+50,full,ww-8);
          if(!t.isConnected)label(g,xx+ww/2,y+50,String(i+1),ww-4);
        });
      }
    }
    if(mode==='batch'){
      const defs=el(svg,'defs'),marker=el(defs,'marker',{id:'quick-arrow',viewBox:'0 0 8 8',refX:8,refY:4,markerWidth:6,markerHeight:6,orient:'auto'});
      el(marker,'path',{d:'M0 0 L8 4 L0 8 Z',fill:'var(--p2)'});
      const rs=currentBatch().ranks;
      for(let r=0;r<2;r++){
        const a=rs[r].phases[2].end,b=rs[r+1].phases[2].start;
        if(a<start||a>end||b<start||b>end)continue;
        const sy=top+r*row+68,ty=top+(r+1)*row+24,mid=(sy+ty)/2;
        el(svg,'path',{d:'M'+x(a)+','+sy+' V'+mid+' H'+x(b)+' V'+ty,fill:'none',stroke:'var(--p2)','stroke-width':1.5,'stroke-dasharray':'4 3','marker-end':'url(#quick-arrow)','data-activation':r});
      }
    }
    $('#window').textContent=start.toFixed(1)+'–'+end.toFixed(1)+' u';
    $('#pan').max=Math.max(0,total-span);$('#pan').value=start;$('#pan').disabled=span===total;
  }
  data.loops.filter(l=>l.r===0).forEach(l=>{const o=html($('#loop'),'option','L'+(l.n+1)+' · 提交 '+batchName(l.current)+' / 收 '+batchName(l.old));o.value=l.n;});
  lifeNames.forEach((name,i)=>{const o=html($('#life-phase'),'option',(i+1)+' · '+name);o.value=i;});
  document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.mode;render();mode==='loop'?focus():fitBatch();}));
  $('#rank').addEventListener('change',e=>goLoop(Number(e.target.value),loop));
  $('#loop').addEventListener('change',e=>goLoop(rank,Number(e.target.value)));
  $('#batch').addEventListener('change',e=>goBatch(Number(e.target.value)));
  $('#life-phase').addEventListener('change',e=>selectLife(Number(e.target.value)));
  $('#zoom-in').addEventListener('click',()=>{const next=span/1.7;setView(start+(span-next)/2,next);});
  $('#zoom-out').addEventListener('click',()=>{const next=span*1.7;setView(start+(span-next)/2,next);});
  $('#focus').addEventListener('click',focus);
  $('#all').addEventListener('click',()=>mode==='loop'?setView(0,total):fitBatch());
  $('#pan').addEventListener('input',e=>setView(Number(e.target.value),span));
  const params=new URLSearchParams(location.search);
  if(params.get('view')==='batch'){mode='batch';batch=clamp(Math.round(Number(params.get('batch')))||1,1,5);}
  new ResizeObserver(draw).observe(svg);
  render();mode==='loop'?focus():fitBatch();
})();
