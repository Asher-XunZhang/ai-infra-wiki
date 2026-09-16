(() => {
  'use strict';
  let data = window.PP_QUICK_DATA;
  const $ = s => document.querySelector(s);
  const names = ['检查状态', '缓存与选批', '输入 / 提交前向', '结果与共识', '转发保存'];
  const lifeNames = ['准入与排队', '准备缓存 / 输入', '本级 GPU 前向', '等结果 / 处理', 'KV 传输', '等共识 / 释放'];
  let total = Math.ceil(data.end / 10) * 10, comparisonEnd = 0;
  let rank = 0, loop = 5, batch = 0, lifePhase = -1;
  let start = 0, span = total;
  let drag = null, suppressClick = false;
  const actions = new WeakMap();
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
    actions.set(g, action);
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
    const l = currentLoop(); fitRange(l.start, l.end);
  }
  function fitBatch() {
    const rs = currentBatch().ranks;
    fitRange(Math.min(...rs.map(r => r.phases[0].start)), Math.max(...rs.map(r => r.phases[5].end)));
  }
  function goLoop(r, n) { rank = r; loop = n; render(); focus(); }
  function goBatch(b) { batch = b; lifePhase = -1; render(); batch ? fitBatch() : focus(); }
  function selectLife(i, fit = true) {
    lifePhase = i; render();
    if (!fit) return;
    if (i < 0) { fitBatch(); return; }
    const ps = currentBatch().ranks.map(r => r.phases[i]);
    fitRange(Math.min(...ps.map(p=>p.start)), Math.max(...ps.map(p=>p.end)));
  }
  function macroTexts(l) {
    return [
      '检查候选请求是否准备好，轮询历史 KV 传输状态。对象是候选请求集合和历史传输中的请求，不专属于某一份当前 batch。',
      '选批前先推进共享 L2 缓存完成通知（ACK）；' + (l.current ? '为 M' + l.current + ' 检查预算和准入条件。' + ([2,4].includes(l.current) ? '本例有 host hit，准备回载后提交准入，再构造 batch 并发起 host → GPU 复制。' : '本例无 host hit，通过检查后提交准入并构造 batch。') : '检查队列，本轮没有选到新的 batch。'),
      l.current ? '准备 M' + l.current + ' 的输入，必要时等上级激活及历史发送，再提交本级 forward。这里只是 CPU 提交；高亮 M' + l.current + ' 可在下沿进度带看 GPU 起止时间。' : '本轮不提交 forward；若仍有历史激活发送 work，会在这里处理。',
      '推进 output 回流、bootstrap / release 共识。' + (l.old ? '接收并处理旧 M' + l.old + ' 的结果，提交它的本级 KV 到 Decode。' : '本轮没有旧 batch 要做最终结果处理。') + (l.r === 2 && l.current ? '末级还会推进当前 M' + l.current + ' 的 output 发送。' : '') + (l.released.length ? '本轮实际清理：' + l.released.map(batchName).join('、') + '，与旧结果对象分开看。' : '本轮没有实际释放的 batch。'),
      (l.r < 2 ? '向后转发请求和状态；' + (l.current ? '安排 M' + l.current + ' 激活发送；' : '') : '末级不再向后发送激活；') + '保存本轮状态，进入下一次 loop。'
    ];
  }
  function renderLoop() {
    const l = currentLoop();
    $('#selection-title').textContent = 'PP' + rank + ' · L' + (loop+1) + '：提交 ' + batchName(l.current) + '，处理旧结果 ' + batchName(l.old);
    $('#selection-text').textContent = '槽位 s' + l.slot + ' · ' + fmt(l.start) + '–' + fmt(l.end) + ' u。本轮实际释放：' + (l.released.length ? l.released.map(batchName).join('、') : '无') + '。状态与缓存队列也在这一轮继续推进。';
    const links = $('#cross-links'); links.replaceChildren();
    [...new Set([l.current, l.old, ...l.released].filter(Boolean))].forEach(b => button(links, '高亮 M' + b, () => goBatch(b)));
    $('#chart-hint').textContent = '每个外框 = 本级的一次 CPU loop，五类大步骤横向拼接。' + (batch ? '已高亮 M' + batch + '：下沿进度带显示它如何跨过这些 loop；其他 batch 的操作仍在图中。' : '选择“高亮 batch”即可叠加生命周期。') + '点击外框查看这一轮在处理谁。';
    $('#detail-title').textContent = '这一轮的五个大步骤';
    $('#detail-intro').textContent = '按 CPU 执行顺序阅读。跨组的对象可以不同；GPU 异步计算并未被塞进 CPU 提交条块。';
    const detail = $('#detail'); detail.replaceChildren(); detail.className = 'macro-steps';
    const descriptions = macroTexts(l);
    l.groups.forEach((g, i) => {
      const card = html(detail, 'article', undefined, 'macro-step'); card.style.setProperty('--phase', 'var(--p' + i + ')');
      html(card, 'h3', (i+1) + ' · ' + names[i]);
      html(card, 'div', g ? fmt(g.start) + '–' + fmt(g.end) + ' u' : '本轮跳过', 'time');
      html(card, 'p', descriptions[i]);
      if (batch && g && g.links[batch]) {
        const link = g.links[batch];
        card.dataset.batchRelated = 'true';
        const association = html(card, 'details', undefined, 'association');
        html(association, 'summary', 'M' + batch + (link.shared ? ' · 多请求共享' : ' · 包含相关操作'));
        html(association, 'p', link.actions.join('；') + '。');
      }
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
    return '本级 KV 传输完成后，继续等待终态名单和 release 共识推进；在 L' + (r.releaseLoop+1) + ' 的 ' + fmt(r.releaseStart) + '–' + fmt(p.end) + ' u 执行本地清理。本级依次释放请求 KV 引用、调用 finish(SUCCESS)、清理发送端与 metadata；不宣称回收全部缓存物理页。';
  }
  function renderBatch() {
    const detail = $('#batch-detail'); detail.replaceChildren();
    $('#batch-detail-section').hidden = !batch;
    if (!batch) return;
    $('#batch-detail-title').textContent = 'M' + batch + ' 与 loop 的关系' + (lifePhase < 0 ? '' : ' · ' + lifeNames[lifePhase]);
    $('#batch-detail-intro').textContent = '点下面的 loop 可定位原图；M' + batch + ' 的高亮保持不变。进度带包含计算、传输和等待，不能理解为持续占用调度线程。';
    currentBatch().ranks.forEach(r => {
      const card = html(detail, 'article', undefined, 'rank-detail');
      html(card, 'h3', 'PP' + r.r + ' · M' + batch);
      if (lifePhase < 0) {
        const milestones = [['共同准入',2],['准备 / 提交前向',r.currentLoop],['旧结果 / KV 提交',r.resultLoop],['本地清理',r.releaseLoop]];
        for (const [name,n] of milestones) {
          const line=html(card,'div',undefined,'milestone');
          html(line,'span',name);button(line,'L'+(n+1),()=>goLoop(r.r,n));
        }
        html(card,'p','GPU '+fmt(r.phases[2].start)+'–'+fmt(r.phases[2].end)+' u；KV 在途 '+fmt(r.phases[4].start)+'–'+fmt(r.phases[4].end)+' u。','time');
      } else {
        const p = r.phases[lifePhase];
        html(card, 'div', fmt(p.start) + '–' + fmt(p.end) + ' u', 'time');
        html(card, 'p', lifecycleText(r));
        button(card, '查看 PP' + r.r + ' 的 L' + (p.loop+1), () => goLoop(r.r, p.loop));
      }
    });
  }
  function render() {
    $('#rank').value = rank; $('#loop').value = loop; $('#batch').value = batch; $('#life-phase').value = lifePhase;
    $('#phase-control').hidden = !batch; $('#fit-batch').hidden = !batch; $('#tracking-legend').hidden = !batch;
    renderLoop(); renderBatch();
    const legend = $('#legend'); legend.replaceChildren();
    names.forEach((name, i) => {
      const item = html(legend, 'span'); const swatch = html(item, 'i', undefined, 'swatch'); swatch.style.background = 'var(--p' + i + ')'; swatch.setAttribute('aria-hidden', 'true');
      item.append(document.createTextNode((i+1) + ' ' + name));
    });
    draw();
  }
  function draw() {
    const w = Math.floor(svg.getBoundingClientRect().width), left = 58, right = 12, top = 48, row = batch ? 154 : 116, h = top + 3*row + 24;
    if (w <= left + right) return;
    const end = start+span, x = t => left+(t-start)/span*(w-left-right);
    svg.replaceChildren(); svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h); svg.setAttribute('height', h);
    svg.dataset.start = start; svg.dataset.end = end; svg.dataset.batch = batch;
    svg.classList.toggle('pannable', span < total);
    label(svg, left, 15, '示意时间 u →', undefined, 'start');
    const raw = span/(w<500?3:8), power = 10**Math.floor(Math.log10(raw)), step = [1,2,5,10].find(n => n*power>=raw)*power;
    for (let t = Math.ceil(start/step)*step; t<=end+.0001; t+=step) {
      const px = x(t); el(svg, 'line', {x1:px,x2:px,y1:top-6,y2:h-20,stroke:'var(--border)','stroke-width':.7});
      label(svg,px,top-15,String(Number(t.toFixed(2))),undefined,px>w-28?'end':px<left+10?'start':'middle');
    }
    for (let r=0;r<3;r++) {
      const y = top+r*row;
      label(svg,0,y+37,'PP'+r,undefined,'start');
      el(svg,'line',{x1:left,x2:w-right,y1:y+row-12,y2:y+row-12,stroke:'var(--border)'});
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
          const px=x(Math.max(start,p.start)),pw=x(Math.min(end,p.end))-px, link=batch&&p.links[batch];
          const block=el(g,'g',{'data-group':i,'data-related':link?(link.shared?'shared':'individual'):'none'});
          el(block,'rect',{x:px,y:y+54,width:pw,height:29,fill:'var(--p'+i+')','fill-opacity':link?.38:batch?.1:selected?.4:.2});
          if(link) el(block,'rect',{x:px+.5,y:y+54.5,width:Math.max(0,pw-1),height:28,fill:'none',stroke:'var(--p2)','stroke-width':1.5,'stroke-dasharray':link.shared?'4 3':'none'});
          const words=['1 状态','2 缓存选批',l.current?'3 提交前向':'3 回收发送','4 结果 / 共识','5 转发'];
          const t=label(block,px+pw/2,y+73,words[i],pw-6);
          if(!t.isConnected)label(block,px+pw/2,y+73,String(i+1),pw-4);
          if(link)el(block,'title',{},'M'+batch+(link.shared?' · 多请求共享：':' · 包含相关操作：')+link.actions.join('；'));
        });
      });
      if(batch){
        label(svg,0,y+113,'M'+batch,undefined,'start');
        label(svg,0,y+130,'进度',undefined,'start');
        currentBatch().ranks[r].phases.forEach((p,i)=>{
          if(p.end<=start||p.start>=end)return;
          const xx=x(Math.max(start,p.start)),ww=x(Math.min(end,p.end))-xx, active=lifePhase===i;
          const g=el(svg,'g',{'data-life-rank':r,'data-life-phase':i});
          selectable(g,()=>selectLife(i,false),'PP'+r+' M'+batch+' '+lifeNames[i]+' '+fmt(p.start)+'至'+fmt(p.end)+' u');
          el(g,'rect',{x:xx,y:y+101,width:ww,height:31,fill:'var(--p2)','fill-opacity':active?.45:i===2?.3:.12,stroke:active?'var(--p2)':'var(--border)','stroke-width':active?2:1});
          const letter=String.fromCharCode(97+i),t=label(g,xx+ww/2,y+121,letter+' '+lifeNames[i],ww-8);
          if(!t.isConnected)label(g,xx+ww/2,y+121,letter,ww-4);
        });
      }
    }
    $('#window').textContent=start.toFixed(1)+'–'+end.toFixed(1)+' u';
    $('#pan-hint').textContent=(span===total?'全程已显示，放大后可左右拖动。':'按住图中任意位置左右拖动平移；Ctrl / ⌘ + 滚轮缩放。')+' 点击 loop 查看大步骤'+(batch?'，点击进度带查看对应生命周期。':'。');
    $('#pan').max=Math.max(0,total-span);$('#pan').value=start;$('#pan').disabled=span===total;
  }
  svg.addEventListener('click',e=>{if(suppressClick&&e.detail>0){e.preventDefault();e.stopImmediatePropagation();}},true);
  svg.addEventListener('pointerdown',e=>{
    if(!e.isPrimary||e.button!==0)return;
    suppressClick=false;
    drag={id:e.pointerId,x:e.clientX,y:e.clientY,start,span,width:svg.getBoundingClientRect().width-70,moved:false,target:e.target.closest('.block')};
    svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener('pointermove',e=>{
    if(!drag||e.pointerId!==drag.id)return;
    const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
    if(!drag.moved){
      if(e.pointerType==='touch'&&Math.abs(dy)>6&&Math.abs(dy)>Math.abs(dx)){endDrag(e);return;}
      if(Math.abs(dx)<6)return;
      drag.moved=true;suppressClick=true;svg.classList.add('dragging');
    }
    e.preventDefault();setView(drag.start-dx/drag.width*drag.span,drag.span);
  });
  function endDrag(e){
    if(!drag||e.pointerId!==drag.id)return;
    const target=drag.target,activate=e.type==='pointerup'&&!drag.moved&&target&&document.elementFromPoint(e.clientX,e.clientY)?.closest('.block')===target;
    drag=null;svg.classList.remove('dragging');
    if(svg.hasPointerCapture(e.pointerId))svg.releasePointerCapture(e.pointerId);
    suppressClick=true;
    if(activate)actions.get(target)?.();
  }
  window.addEventListener('pointerup',endDrag);
  svg.addEventListener('pointercancel',endDrag);
  svg.addEventListener('lostpointercapture',endDrag);
  svg.addEventListener('wheel',e=>{
    if(!e.ctrlKey&&!e.metaKey)return;
    e.preventDefault();const rect=svg.getBoundingClientRect(),fraction=clamp((e.clientX-rect.left-58)/(rect.width-70),0,1),next=clamp(span*Math.exp(e.deltaY*.005),2,total);
    setView(start+fraction*(span-next),next);
  },{passive:false});
  function refillLoops(){$('#loop').replaceChildren();data.loops.filter(l=>l.r===0).forEach(l=>{const o=html($('#loop'),'option','L'+(l.n+1)+' · 提交 '+batchName(l.current)+' / 收 '+batchName(l.old));o.value=l.n;});}
  refillLoops();
  lifeNames.forEach((name,i)=>{const o=html($('#life-phase'),'option',String.fromCharCode(97+i)+' · '+name);o.value=i;});
  $('#rank').addEventListener('change',e=>goLoop(Number(e.target.value),loop));
  $('#loop').addEventListener('change',e=>goLoop(rank,Number(e.target.value)));
  $('#batch').addEventListener('change',e=>goBatch(Number(e.target.value)));
  $('#life-phase').addEventListener('change',e=>selectLife(Number(e.target.value)));
  $('#zoom-in').addEventListener('click',()=>{const next=span/1.7;setView(start+(span-next)/2,next);});
  $('#zoom-out').addEventListener('click',()=>{const next=span*1.7;setView(start+(span-next)/2,next);});
  $('#focus').addEventListener('click',focus);
  $('#fit-batch').addEventListener('click',fitBatch);
  $('#all').addEventListener('click',()=>setView(0,total));
  $('#pan').addEventListener('input',e=>setView(Number(e.target.value),span));
  const params=new URLSearchParams(location.search);
  if(params.has('batch')||params.get('view')==='batch')batch=clamp(Math.round(Number(params.get('batch')))||1,1,5);
  new ResizeObserver(draw).observe(svg);
  window.addEventListener('pp-timing-change',event=>{
    data=event.detail.packet.quick;total=comparisonEnd||Math.ceil(data.end/10)*10;
    loop=Math.min(loop,Math.max(...data.loops.filter(l=>l.r===rank).map(l=>l.n)));
    drag=null;start=0;span=total;refillLoops();render();
    svg.dataset.scenario=event.detail.id;
    comparisonEnd?setView(0,total):batch?fitBatch():focus();
  });
  window.addEventListener('pp-timing-scale',event=>{
    comparisonEnd=event.detail.end;total=comparisonEnd||Math.ceil(data.end/10)*10;
    comparisonEnd?setView(0,total):batch?fitBatch():focus();
  });
  svg.dataset.scenario='baseline';
  render();batch?fitBatch():focus();
})();
