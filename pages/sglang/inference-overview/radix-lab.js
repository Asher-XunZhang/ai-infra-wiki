/* Renders the fixed-commit RadixCache lifecycle model as a synchronized
 * tree / query / eviction-candidate lesson. No network or engine calls. */
(function () {
  'use strict';

  const M = window.RadixLifecycleModel;
  const lab = document.querySelector('[data-radix-lab]');
  if (!M || !lab) return;

  const DEMOS = Array.isArray(M.demos) && M.demos.length
    ? M.demos
    : [{ id: 'journey', label: '完整链路', hint: '', states: M.states, steps: M.steps, interval: 2600 }];
  let activeDemo = DEMOS[0];
  let demoSteps = activeDemo.steps;
  let demoStates = activeDemo.states;
  let LAST = demoStates.length - 1;
  const reducedQuery = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
  let reduced = Boolean(reducedQuery && reducedQuery.matches);
  let index = 0;
  let view = null;
  let playing = false;
  let previousLocks = new Map();

  const $ = (selector) => lab.querySelector(selector);

  const els = {
    treeScroll: $('#radix-tree-scroll'),
    treeCanvas: $('#radix-tree-canvas'),
    treeNodes: $('#radix-tree-nodes'),
    treeEdges: $('#radix-tree-edges'),
    requests: $('#radix-requests'),
    flow: $('#radix-flow'),
    modeHint: $('#radix-mode-hint'),
    timeline: $('#radix-timeline'),
    stepno: $('#radix-stepno'),
    phase: $('#radix-phase'),
    title: $('#radix-step-title'),
    caption: $('#radix-step-caption'),
    note: $('#radix-step-note'),
    source: $('#radix-source'),
    metrics: $('#radix-metrics'),
    evictable: $('#radix-evictable'),
    pressure: $('#radix-pressure'),
    progress: $('#radix-progress'),
    prev: $('#radix-prev'),
    next: $('#radix-next'),
    play: $('#radix-play'),
    reset: $('#radix-reset')
  };

  const nodeEls = new Map();

  function clamp(value) {
    return Math.max(0, Math.min(LAST, value));
  }

  function nodeName(id) {
    return id === 'root' ? 'root' : id.toUpperCase();
  }

  function isFocus(id) {
    return Boolean(view && view.meta.focus && view.meta.focus.nodes.includes(id));
  }


  function splitRole(id) {
    if (!view || !view.meta.split) return '';
    if (id === 'n2') return '分裂·父';
    if (id === 'n1') return '分裂·后缀';
    return '';
  }

  function nodeState(node) {
    if (node.id === 'root') return '固定根';
    if (view.evictable.includes(node.id)) return '可驱逐';
    if (node.lock > 0) return '保护中';
    return '缓存';
  }

  function nodeLastOwners(id) {
    return Object.values(view.requests)
      .filter((req) => req.lastNode === id)
      .map((req) => req.id);
  }

  function tokenLines(tokens, className) {
    const lines = [];
    for (let i = 0; i < tokens.length; i += 4) {
      const chips = tokens
        .slice(i, i + 4)
        .map((token, offset) => `<span class="${className}" data-slot="${i + offset}" style="--i:${i + offset}">${token}</span>`)
        .join('');
      lines.push(`<div class="radix-token-line">${chips}</div>`);
    }
    return lines.join('');
  }

  function valueGroups(node) {
    return `<span class="radix-index-value">value [${node.slots.join(', ')}]</span>`;
  }

  function ensureNode(id) {
    let el = nodeEls.get(id);
    if (!el) {
      el = document.createElement('article');
      el.className = 'radix-node';
      el.dataset.node = id;
      el.setAttribute('role', 'listitem');
      els.treeNodes.append(el);
      nodeEls.set(id, el);
    }
    return el;
  }

  function updateNode(node) {
    const el = ensureNode(node.id);
    const root = node.id === 'root';
    const evictable = view.evictable.includes(node.id);
    const lastOwners = nodeLastOwners(node.id);
    const classes = ['radix-node'];
    if (root) classes.push('is-root');
    if (isFocus(node.id)) classes.push('is-focus');
    if (evictable) classes.push('is-evictable');
    if (view.meta.split && (node.id === 'n1' || node.id === 'n2')) classes.push('is-split');
    el.className = classes.join(' ');
    el.dataset.node = node.id;
    el.dataset.lock = String(node.lock);
    el.dataset.evictable = String(evictable);

    if (root) {
      el.setAttribute(
        'aria-label',
        'Radix 树根节点，lock_ref 固定为 1，不参与驱逐'
      );
      el.innerHTML = '<b class="radix-root-label">root</b>';
      previousLocks.set(node.id, node.lock);
      return;
    }

    const split = splitRole(node.id);
    const matchedRequest = view.meta.match ? view.requests[view.meta.match.request] : null;
    const isMatchNode = Boolean(matchedRequest && matchedRequest.lastNode === node.id);
    const matchText = isMatchNode ? `命中 ${view.meta.match.hit}/${view.meta.match.total}` : '';
    if (isMatchNode) classes.push('is-matching');
    const lockClass = node.lock > 0 ? 'radix-lock is-locked' : 'radix-lock';
    const lockDiff = '';
    const lastText = lastOwners.length ? `last ${lastOwners.join('/')}` : '';
    el.setAttribute(
      'aria-label',
      `节点 ${nodeName(node.id)}，key ${node.tokens.join('')}，value 槽号 ${node.slots.join('、')}，lock_ref ${node.lock}，${nodeState(node)}`
    );
    el.innerHTML =
      `<div class="radix-node-head"><span class="radix-node-id">${nodeName(node.id)}</span>` +
      `${split ? `<span class="radix-node-split">${split}</span>` : ''}` +
      `<span class="radix-node-state">${nodeState(node)}</span>` +
      `${matchText ? `<span class="radix-node-match">${matchText}</span>` : ''}</div>` +
      `<div class="radix-node-tokens">${tokenLines(node.tokens, 'radix-chip')}</div>` +
      `<div class="radix-node-value">${valueGroups(node)}</div>` +
      `<div class="radix-node-foot"><span class="${lockClass}${lockDiff}">锁 ${node.lock}</span>` +
      `<span class="radix-node-last${lastText ? ' is-on' : ''}" title="${lastText || '没有请求以它为 last_node'}">${lastText || '—'}</span></div>`;

    previousLocks.set(node.id, node.lock);
  }

  function requestStatus(req) {
    if (req.rowReleased) return { text: '已结束', className: 'is-finished' };
    if (req.status === 'queued') return { text: req.prefixLen ? '已匹配 · 待准入' : '等待', className: '' };
    if (req.status === 'prefill') return { text: '已准入 · 前缀受保护', className: 'is-active' };
    if (req.status === 'decoding') return { text: '使用中 · 已入树', className: 'is-active' };
    return { text: req.status, className: '' };
  }

  function renderRequests() {
    const visible=view.meta.visibleRequests;
    els.requests.innerHTML=Object.values(view.requests).filter(req=>!visible||visible.includes(req.id)).map(req=>{
      const status=requestStatus(req);
      const tokens=req.tokens.map((token,i)=>`<span class="radix-input-token ${i<req.prefixLen?'is-hit':''}">${token}</span>`).join('');
      const prepared=req.row.filter(Boolean).map(entry=>entry.slot);
      const pendingInsert=prepared.length>req.prefixLen&&!req.rowReleased;
      return `<article class="radix-request${req.rowReleased?' is-finished':''}" data-request="${req.id}" aria-label="请求 ${req.id}，${status.text}">`+
        `<div class="radix-request-head"><strong>${req.id}</strong><span class="radix-status">${status.text}</span></div>`+
        `<div class="radix-input-tokens" aria-label="输入 token">${tokens}</div>`+
        `<dl class="radix-request-fields"><div><dt>prefix_indices</dt><dd>${req.prefixIndices.length?'['+req.prefixIndices.join(', ')+']':'[]'}</dd></div>`+
        `<div><dt>last_node</dt><dd>${req.lastNode?nodeName(req.lastNode):'—'} <span>· 索引长度 ${req.prefixLen}</span></dd></div></dl>`+
        (pendingInsert?`<p class="radix-insert-argument">待插入 value [${prepared.join(', ')}]</p>`:'')+
        (req.rowReleased?'<p class="radix-request-ended">请求已结束，不再持有路径保护</p>':'')+'</article>';
    }).join('');
  }

  function renderMetrics() {
    const active = ['R1', 'R2']
      .map((id) => view.requests[id])
      .filter((req) => !req.rowReleased && req.prefixLen > 0);
    const prefix = active.length
      ? active.map((req) => `${req.id} ${req.prefixLen}/${req.tokens.length}`).join(' · ')
      : '—';
    const evictable = view.evictable.length
      ? view.evictable.map((id) => `<b>${nodeName(id)}</b>`).join('')
      : '<em>∅ 无可驱逐叶</em>';
    els.metrics.innerHTML =
      `<div class="radix-metric"><span>索引长度 / 输入</span><strong>${prefix}</strong><small>命中索引长度 / 输入长度</small></div>` +
      `<div class="radix-metric"><span>已索引 token</span><strong>${view.treeSlots}</strong><small>索引，不是 K/V 字节</small></div>` +
      `<div class="radix-metric"><span>受保护位置</span><strong>${view.lockedSlots}</strong><small>lock_ref &gt; 0</small></div>` +
      `<div class="radix-metric"><span>索引节点数</span><strong>${Object.keys(view.nodes).length-1}</strong><small>不含 root</small></div>`;
    els.evictable.innerHTML = `<span>可驱逐叶（evictable_leaves）</span>${evictable}`;
    if (view.meta.pressure) {
      els.pressure.hidden = false;
      els.pressure.innerHTML =
        `<b>evict</b><span>候选集合为空，本次不移除节点</span>`;
    } else if (view.meta.evicted) {
      const evicted = view.meta.evicted.map((id) => nodeName(id)).join('、');
      els.pressure.hidden = false;
      els.pressure.innerHTML = `<b>evict</b><span>已移除 ${evicted}，更新父节点资格</span>`;
    } else {
      els.pressure.hidden = true;
      els.pressure.innerHTML = '';
    }
  }

  function renderCaption() {
    const step = view.meta;
    els.stepno.textContent = `步骤 ${index + 1} / ${LAST + 1}`;
    els.phase.textContent = step.phase;
    els.title.textContent = step.title;
    els.caption.textContent = step.caption;
    els.note.textContent = step.note;
    const source = M.sources[step.source];
    const file = source[0].split('/').pop();
    els.source.innerHTML =
      `源码：<a href="${M.sourceUrl(step.source)}" target="_blank" rel="noreferrer"><code>${file}:${source[1]}</code> ${source[2]} ↗</a>` +
      `<span>固定提交 ${M.revision.slice(0, 10)}</span>`;
  }

  function timelineColumns() {
    return window.innerWidth <= 600 ? Math.min(demoSteps.length, 4) : demoSteps.length;
  }

  function buildTimeline() {
    els.timeline.style.setProperty('--timeline-columns', String(timelineColumns()));
    els.timeline.setAttribute('aria-label', `${activeDemo.label}的机制步骤`);
    els.timeline.innerHTML = demoSteps
      .map(
        (step) =>
          `<li data-phase="${step.phase}"><button type="button" data-step="${step.index}" aria-label="第 ${step.index + 1} 步：${step.title}">` +
          `<b>${step.index + 1}</b><small>${step.short}</small></button></li>`
      )
      .join('');
    els.timeline.querySelectorAll('button[data-step]').forEach((button) => {
      button.addEventListener('click', () => go(Number(button.dataset.step), { user: true }));
    });
  }

  function updateModeUI() {
    lab.dataset.mode = activeDemo.id;
    if (els.modeHint) els.modeHint.textContent = activeDemo.hint;
    lab.querySelectorAll('[data-mode]').forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.mode === activeDemo.id));
    });
  }

  let clock=M.timeline(activeDemo.id),time=0,sample,lastTick=null,raf=null,sceneKey='',positions={},priorPositions={};
  const seek=$('#radix-seek'),speed=$('#radix-speed'),timeLabel=$('#radix-time');
  lab.dataset.clock='timeline';
  function layout(nodes) {
    const widths={root:46,n1:134,n2:134,n3:134,n4:134},height=id=>id==='root'?42:(nodes[id].tokens.length>4?160:126),sub={};
    function span(id){const n=nodes[id],children=n.children.filter(k=>nodes[k]);return sub[id]=Math.max(height(id),children.reduce((sum,k)=>sum+span(k),0)+Math.max(0,children.length-1)*22);}
    span('root');const out={},total=Math.max(258,sub.root);
    function place(id,top,depth){const n=nodes[id],children=n.children.filter(k=>nodes[k]),x=depth?64+(depth-1)*152:4,y=58+top+(sub[id]-height(id))/2;out[id]={x,y,w:widths[id],h:height(id)};let next=top;for(const k of children){place(k,next,depth+1);next+=sub[k]+22;}}
    place('root',(total-sub.root)/2,0);return {positions:out,width:Math.max(490,...Object.values(out).map(p=>p.x+p.w+8)),height:total+68};
  }
  function treeScene(){
    const current=layout(view.nodes),prior=layout(sample.prior.nodes);positions=current.positions;priorPositions=prior.positions;
    const all=new Set([...Object.keys(view.nodes),...(sample.morph<1?Object.keys(sample.prior.nodes):[])]);
    for(const id of [...nodeEls.keys()])if(!all.has(id)){nodeEls.get(id).remove();nodeEls.delete(id);}
    for(const id of all)updateNode(view.nodes[id]||sample.prior.nodes[id]);
    const width=Math.max(current.width,prior.width),height=Math.max(current.height,prior.height);
    els.treeCanvas.style.width=width+'px';els.treeCanvas.style.height=height+'px';els.treeNodes.style.width=width+'px';els.treeNodes.style.height=height+'px';
    els.treeEdges.style.width=width+'px';els.treeEdges.style.height=height+'px';
    els.treeEdges.setAttribute('viewBox',`0 0 ${width} ${height}`);els.treeEdges.setAttribute('width',width);els.treeEdges.setAttribute('height',height);
  }
  function treeMotion(){
    const u=reduced?1:1-Math.pow(1-sample.morph,3),now={};
    for(const[id,el]of nodeEls){const target=positions[id],old=priorPositions[id],fallback=old||priorPositions.n1||target;
      if(target){const start=old||fallback;now[id]={...target,x:start.x+(target.x-start.x)*u,y:start.y+(target.y-start.y)*u};el.style.height=target.h+'px';el.style.transform=`translate(${now[id].x}px,${now[id].y}px)`;el.style.opacity=String(old?1:u);el.style.visibility='visible';}
      else {el.style.transform=`translate(${old.x}px,${old.y}px)`;el.style.opacity=String(1-u);el.style.visibility=u===1?'hidden':'visible';now[id]=old;}
    }
    const paths=[];for(const[id,n]of Object.entries(view.nodes)){if(!n.parent)continue;const a=now[n.parent],b=now[id],x1=a.x+a.w,y1=a.y+a.h/2,x2=b.x,y2=b.y+b.h/2,m=(x1+x2)/2;paths.push(`<path d="M${x1},${y1} C${m},${y1} ${m},${y2} ${x2},${y2}" class="${n.lock?'is-focus':''}" marker-end="url(#radix-edge-arrow)"></path>`);}
    els.treeEdges.innerHTML='<defs><marker id="radix-edge-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L8 4L0 8Z" class="radix-arrow-head"/></marker></defs>'+paths.join('');
  }
  function anchor(selector) {
    const element=lab.querySelector(selector);
    if(!element)return null;
    const figure=lab.querySelector('.radix-figure').getBoundingClientRect();
    const rect=element.getBoundingClientRect(),panel=element.closest('.radix-panel');
    const panelRect=panel.getBoundingClientRect();
    return {x:rect.left-figure.left,y:rect.top-figure.top,w:rect.width,h:rect.height,
      kind:element.dataset.node?'node':'request',
      panel:{x:panelRect.left-figure.left,y:panelRect.top-figure.top,w:panelRect.width,
        header:panel.querySelector('.radix-panel-head').getBoundingClientRect().height}};
  }
  function route(a,b) {
    if(a.kind==='node'&&b.kind==='node'&&a.x+a.w<=b.x) {
      const middle=(a.x+a.w+b.x)/2;
      return [[a.x+a.w,a.y+a.h/2],[middle,a.y+a.h/2],[middle,b.y+b.h/2],[b.x,b.y+b.h/2]];
    }
    const figure=lab.querySelector('.radix-figure').getBoundingClientRect();
    const requests=lab.querySelector('.radix-req-panel').getBoundingClientRect();
    const rail=requests.left-figure.left>100?requests.left-figure.left-4:2;
    function port(r) {
      if(r.kind==='node') {
        const lane=r.panel.y+r.panel.header+10;
        return [[r.x+r.w/2,r.y],[r.x+r.w/2,lane],[rail,lane]];
      }
      if(r.kind==='request')return [[r.x,r.y+r.h/2],[rail,r.y+r.h/2]];

    }
    return [...port(a),...port(b).reverse()];
  }
  function along(points,u){const lengths=points.slice(1).map((p,i)=>Math.hypot(p[0]-points[i][0],p[1]-points[i][1])),total=lengths.reduce((a,b)=>a+b,0);let remaining=total*u;for(let i=0;i<lengths.length;i++){if(remaining<=lengths[i]){const f=lengths[i]?remaining/lengths[i]:0;return[points[i][0]+(points[i+1][0]-points[i][0])*f,points[i][1]+(points[i+1][1]-points[i][1])*f];}remaining-=lengths[i];}return points.at(-1);}
  function flowMotion(){
    els.flow.replaceChildren();if(reduced)return;
    const box=lab.querySelector('.radix-figure').getBoundingClientRect(),svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox',`0 0 ${box.width} ${box.height}`);svg.setAttribute('width',box.width);svg.setAttribute('height',box.height);svg.classList.add('radix-transfer-lines');els.flow.append(svg);
    for(const f of sample.flows){const anchors=[f.from,...f.via,f.to].map(anchor);if(anchors.some(a=>!a))continue;let pts=[];for(let i=1;i<anchors.length;i++)pts.push(...route(anchors[i-1],anchors[i]));const p=along(pts,f.u),path=document.createElementNS(svg.namespaceURI,'path');path.setAttribute('d',pts.map((p,i)=>(i?'L':'M')+p.join(' ')).join(' '));svg.append(path);const el=document.createElement('span');el.className='radix-packet';el.textContent=f.label;el.dataset.transfer=f.label;el.style.transform=`translate(${p[0]}px,${p[1]}px) translate(-50%,-50%)`;el.style.opacity=String(Math.min(1,f.u*8,(1-f.u)*8));els.flow.append(el);}
  }
  function updateControls(){
    els.prev.disabled=time===0;els.next.disabled=time>=clock.duration;
    const progress=`第 ${index+1} / ${LAST+1} 步`;if(els.progress.textContent!==progress)els.progress.textContent=progress;els.play.textContent=playing?'暂停':time===clock.duration?'重新播放':'播放全程';els.play.setAttribute('aria-pressed',String(playing));
    els.timeline.querySelectorAll('[data-step]').forEach(b=>{if(Number(b.dataset.step)===index)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});
    seek.max=clock.duration;seek.value=time;seek.setAttribute('aria-valuetext',`${time.toFixed(1)} 秒，${sample.label}`);timeLabel.textContent=`${time.toFixed(1)} / ${clock.duration.toFixed(1)} s`;
  }
  function render(force=false){
    sample=M.atTime(clock,time);index=sample.phase.index;view=M.derive(sample.raw);lab.dataset.time=time.toFixed(3);lab.dataset.phase=index;lab.dataset.event=sample.event.at;lab.dataset.playing=String(playing);
    const key=`${activeDemo.id}/${index}/${sample.event.at}/${innerWidth}/${reduced}/${sample.morph===1}`;
    if(force||key!==sceneKey){sceneKey=key;renderRequests();renderMetrics();renderCaption();els.title.textContent=sample.label;treeScene();}
    treeMotion();flowMotion();updateControls();
    depth.update({time,event:sample.label,explanation:view.meta.note,rows:Object.values(view.nodes).map(n=>[n.id,`key=[${(n.tokens||[]).join(', ')}] · value=[${n.slots.join(', ')}] · lock_ref=${n.lock}`]),href:M.sourceUrl(view.meta.source)});
  }
  function updateURL(){const url=new URL(location.href);if(activeDemo.id==='journey')url.searchParams.delete('mode');else url.searchParams.set('mode',activeDemo.id);if(index)url.searchParams.set('step',index);else url.searchParams.delete('step');if(time)url.searchParams.set('t',time.toFixed(3));else url.searchParams.delete('t');history.replaceState(null,'',url);}
  function pause(){playing=false;lastTick=null;if(raf!==null)cancelAnimationFrame(raf);raf=null;render();updateURL();}
  function jump(t){time=Math.max(0,Math.min(clock.duration,t));pause();}
  function go(next){jump(clock.phases[clamp(next)].end);}
  function tick(now){if(!playing)return;if(lastTick!==null)time=Math.min(clock.duration,time+(now-lastTick)/1000*Number(speed.value));lastTick=now;if(time===clock.duration)playing=false;render();if(playing)raf=requestAnimationFrame(tick);else {raf=null;lastTick=null;updateURL();}}
  function play(){if(playing)return pause();if(time===clock.duration)time=0;playing=true;lastTick=null;render();raf=requestAnimationFrame(tick);}
  function setMode(id){pause();activeDemo=DEMOS.find(d=>d.id===id)||DEMOS[0];demoSteps=activeDemo.steps;demoStates=activeDemo.states;LAST=demoStates.length-1;clock=M.timeline(activeDemo.id);time=0;sceneKey='';updateModeUI();buildTimeline();render(true);updateURL();}
  function reset(){jump(0);}
  function bind(){els.play.onclick=play;els.prev.onclick=()=>index>0?go(index-1):jump(0);els.next.onclick=()=>go(index+1);els.reset.onclick=reset;seek.oninput=()=>jump(Number(seek.value));speed.onchange=()=>{lastTick=null;};lab.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{if(b.dataset.mode!==activeDemo.id)setMode(b.dataset.mode);});lab.addEventListener('keydown',e=>{if(/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)||e.altKey||e.ctrlKey||e.metaKey)return;if(e.key==='ArrowRight'){e.preventDefault();els.next.click();}if(e.key==='ArrowLeft'){e.preventDefault();els.prev.click();}if(e.key==='Home'){e.preventDefault();reset();}if(e.key==='End'){e.preventDefault();jump(clock.duration);}});document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});window.addEventListener('pagehide',pause);window.addEventListener('resize',()=>{buildTimeline();render(true);});reducedQuery?.addEventListener('change',e=>{reduced=e.matches;render(true);});window.addEventListener('popstate',loadLocation);}
  function loadLocation(){const params=new URL(location.href).searchParams,demo=DEMOS.find(d=>d.id===params.get('mode'))||DEMOS[0];playing=false;lastTick=null;if(raf!==null)cancelAnimationFrame(raf);activeDemo=demo;demoSteps=demo.steps;demoStates=demo.states;LAST=demoStates.length-1;clock=M.timeline(demo.id);const step=Number(params.get('step'));time=params.has('t')?Math.max(0,Math.min(clock.duration,Number(params.get('t'))||0)):step>0?clock.phases[clamp(step)].end:0;updateModeUI();buildTimeline();render(true);}
  const depth=LearningDepth.mount({host:lab,before:$('.radix-figure'),compact:true,onChange:()=>render(true),persist:()=>updateURL()});
  lab.hidden=false;bind();loadLocation();
})();
