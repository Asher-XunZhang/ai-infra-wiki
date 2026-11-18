(() => {
  'use strict';
  const root = document.querySelector('[data-layer-atlas]');
  if (!root) return;

  const board = root.querySelector('[data-la-board]');
  const canvas = root.querySelector('[data-la-canvas]');
  const svg = root.querySelector('[data-la-wires]');
  const NS = 'http://www.w3.org/2000/svg';
  const STACK_LIMIT = 1000;
  const STEP_HOLD = 2300;

  const STEPS = [
    {
      title: '输入与嵌入',
      nodes: ['tokens', 'embed', 'x'],
      edges: ['tokens-embed', 'embed-x'],
      do: '按 token ID 查出各位置的初始向量。图中跟踪末位「？」的表示。',
      handoff: '各位置的初始向量一起进入第 1 层；图里选「？」的 x 来追踪。',
      load: 'token id → x'
    },
    {
      title: 'RMSNorm · 第一子层',
      nodes: ['norm1'],
      edges: ['x-norm1'],
      do: '按向量自身的幅度调整数值尺度，再用学到的系数调节各维。',
      handoff: 'h 进入 Attention 的 Q/K/V 投影；旁路上的 x 原样保留，等待后面相加。',
      load: 'x → h'
    },
    {
      title: 'Attention · 汇聚上下文',
      nodes: ['attn'],
      edges: ['norm1-attn'],
      do: '按相关程度汇聚可见 token 的信息。多头用多组参数分别计算联系，再合并结果；RoPE 让匹配同时考虑位置关系。',
      handoff: '每个头用旋转后的 Q/K 打分，再按位置权重汇聚 V。各头合并成 ΔA，送往残差加法。',
      load: 'h → Q/K/V → ΔA'
    },
    {
      title: '残差 · 得到 x₁',
      nodes: ['add1', 'x1'],
      edges: ['attn-add1', 'bypass1', 'add1-x1'],
      do: '把 Attention 的更新 ΔA 与输入 x 逐元素相加，得到更新后的表示 x₁。',
      handoff: 'x₁ 交给第二个子层；第二旁路也从这里取，不是从最初的 x 取。',
      load: 'x + ΔA → x₁'
    },
    {
      title: 'RMSNorm · 第二子层',
      nodes: ['x1', 'norm2'],
      edges: ['x1-norm2'],
      do: '调整 x₁ 的数值尺度，得到送入 FFN 的 h₂。原来的 x₁ 沿残差旁路保留。',
      handoff: 'h₂ 进入 FFN；x₁ 旁路保持不动。',
      load: 'x₁ → h₂'
    },
    {
      title: 'FFN · 提取与组合特征',
      nodes: ['ffn'],
      edges: ['norm2-ffn'],
      do: '上投影展开更多特征，SwiGLU 按输入调节各通道响应，下投影再组合回层间宽度，产生更新 ΔM。展开下方原理，可查看每一步的作用。',
      handoff: '同一层的 FFN 对各位置独立使用同一套参数；ΔM 与旁路保留的 x₁ 相加。',
      load: 'h₂ → ΔM'
    },
    {
      title: '残差 · 得到 y',
      nodes: ['add2', 'y'],
      edges: ['ffn-add2', 'bypass2', 'add2-y'],
      do: '把 FFN 的更新 ΔM 与输入 x₁ 逐元素相加，得到当前层的输出 y。',
      handoff: 'y 是这一层的输出，不是 token 概率；下一层会把它当作新的 x。',
      load: 'x₁ + ΔM → y'
    },
    {
      title: '其余模型层 · Final Norm',
      nodes: ['repeat', 'finalnorm'],
      edges: ['y-repeat', 'repeat-finalnorm'],
      do: '各层在上一层的表示上重新汇聚上下文、加工特征。最后用 Final Norm 调整表示的尺度。',
      handoff: '全模型共 N 层；最后一层的输出经 Final Norm 后交给输出头。',
      load: 'y → 其余模型层 → 最终归一化表示'
    },
    {
      title: 'LM Head · 下一 token',
      nodes: ['head', 'next'],
      edges: ['finalnorm-head', 'head-next'],
      do: 'LM Head 给词表候选打分。Softmax 将分数转成概率，供采样等选择策略使用。',
      handoff: '按贪心或采样等策略选出下一个 token；若生成尚未结束，将它追加到上下文，继续下一轮预测。',
      load: '最终归一化表示 → 词表分数 → 下一 token'
    }
  ];

  const EDGES = [
    { id: 'tokens-embed', from: 'tokens', to: 'embed', label: 'token ID' },
    { id: 'embed-x', from: 'embed', to: 'x', special: 'entry' },
    { id: 'x-norm1', from: 'x', to: 'norm1' },
    { id: 'norm1-attn', from: 'norm1', to: 'attn' },
    { id: 'attn-add1', from: 'attn', to: 'add1', label: 'ΔA' },
    { id: 'bypass1', from: 'x', to: 'add1', kind: 'residual', desktop: 'top', mobile: 'left', label: '保留 x' },
    { id: 'add1-x1', from: 'add1', to: 'x1' },
    { id: 'x1-norm2', from: 'x1', to: 'norm2' },
    { id: 'norm2-ffn', from: 'norm2', to: 'ffn' },
    { id: 'ffn-add2', from: 'ffn', to: 'add2', label: 'ΔM' },
    { id: 'bypass2', from: 'x1', to: 'add2', kind: 'residual', desktop: 'bottom', mobile: 'right', label: '保留 x₁', port: 'right' },
    { id: 'add2-y', from: 'add2', to: 'y' },
    { id: 'y-repeat', from: 'y', to: 'repeat' },
    { id: 'repeat-finalnorm', from: 'repeat', to: 'finalnorm' },
    { id: 'finalnorm-head', from: 'finalnorm', to: 'head' },
    { id: 'head-next', from: 'head', to: 'next' }
  ];

  const nodeElements = new Map();
  const edgeElements = new Map();
  const firstStep = new Map();
  const controls = [...root.querySelectorAll('[data-la-controls]')];
  const prevButton = root.querySelector('[data-la-prev]');
  const nextButton = root.querySelector('[data-la-next]');
  const toggleButton = root.querySelector('[data-la-toggle]');
  const fullButton = root.querySelector('[data-la-full]');
  const runHandoff = root.querySelector('[data-la-run-handoff]');
  const runRoute = root.querySelector('[data-la-run-route]');
  const runValue = root.querySelector('[data-la-run-value]');
  const runBranch = root.querySelector('[data-la-run-branch]');
  const resetButton = root.querySelector('[data-la-reset]');
  const range = root.querySelector('[data-la-range]');
  const stepButtons = [...root.querySelectorAll('[data-la-step-button]')];
  const count = root.querySelector('[data-la-step-count]');
  const title = root.querySelector('[data-la-step-title]');
  const doText = root.querySelector('[data-la-do]');
  const nextText = root.querySelector('[data-la-handoff-next]');
  const load = root.querySelector('[data-la-load]');
  const status = root.querySelector('[data-la-status]');
  const rangeReadout = root.querySelector('[data-la-range-readout]');
  const detail = root.querySelector('[data-la-detail]');
  const detailTitle = root.querySelector('[data-la-detail-title]');
  const explainers = [...root.querySelectorAll('[data-la-explainer]')];
  const trace = root.querySelector('[data-la-trace]');

  root.querySelectorAll('[data-la-node]').forEach(node => nodeElements.set(node.dataset.laNode, node));
  STEPS.forEach((step, index) => step.nodes.forEach(id => {
    if (!firstStep.has(id)) firstStep.set(id, index);
  }));

  let stepIndex = 0;
  let state = 'idle';
  let timer = 0;
  let drawFrame = 0;
  let focusController;
  const model = window.LayerAtlasFocusModel;
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const fullRun = model.fullRun();
  let run = null;
  const pulses = [];
  const write = (element, value) => { if (element.textContent !== value) element.textContent = value; };
  const formatVector = values => '[' + values.map(v=>Number(v.toFixed(2))).join(', ') + ']';

  function createSvg() {
    const defs = document.createElementNS(NS, 'defs');
    const marker = (id, color) => {
      const m = document.createElementNS(NS, 'marker');
      m.setAttribute('id', id);
      m.setAttribute('viewBox', '0 0 7 7');
      m.setAttribute('refX', '6');
      m.setAttribute('refY', '3.5');
      m.setAttribute('markerWidth', '7');
      m.setAttribute('markerHeight', '7');
      m.setAttribute('orient', 'auto');
      const p = document.createElementNS(NS, 'path');
      p.setAttribute('d', 'M0 0L7 3.5L0 7Z');
      p.setAttribute('fill', color);
      m.append(p);
      return m;
    };
    defs.append(
      marker('la-arrow', 'var(--la-line)'),
      marker('la-arrow-active', 'var(--la-signal)'),
      marker('la-arrow-residual', 'var(--la-residual)'),
      marker('la-arrow-residual-active', 'var(--la-residual)')
    );
    svg.replaceChildren(defs);

    for (const edge of EDGES) {
      const path = document.createElementNS(NS, 'path');
      path.setAttribute('class', 'la-edge');
      path.dataset.laEdge = edge.id;
      path.dataset.laFrom = edge.from;
      path.dataset.laTo = edge.to;
      path.dataset.laKind = edge.kind || 'flow';
      svg.append(path);
      const text = document.createElementNS(NS, 'text');
      text.setAttribute('class', 'la-edge-label' + (edge.kind === 'residual' ? ' is-residual' : ''));
      text.setAttribute('text-anchor', 'middle');
      text.textContent = edge.label || '';
      svg.append(text);
      edgeElements.set(edge.id, { path, text });
    }
    for (const residual of [false,true]) {
      const pulse=document.createElementNS(NS,'g');
      pulse.setAttribute('class','la-run-pulse'+(residual?' la-run-pulse--residual':''));
      const ring=document.createElementNS(NS,'circle');
      ring.setAttribute('r','8');
      pulse.append(ring);
      for(let i=0;i<4;i++){
        const bar=document.createElementNS(NS,'rect');
        bar.setAttribute('width','2');bar.setAttribute('height','7');bar.setAttribute('y','-3.5');
        pulse.append(bar);
      }
      pulse.style.display='none';svg.append(pulse);pulses.push(pulse);
    }
  }

  function rect(id) {
    const node = nodeElements.get(id);
    const nr = node.getBoundingClientRect();
    const cr = canvas.getBoundingClientRect();
    return {
      l: nr.left - cr.left, t: nr.top - cr.top,
      r: nr.right - cr.left, b: nr.bottom - cr.top,
      w: nr.width, h: nr.height,
      cx: nr.left - cr.left + nr.width / 2,
      cy: nr.top - cr.top + nr.height / 2
    };
  }

  function horizontal(source, target) {
    const leftToRight = target.cx >= source.cx;
    const sx = leftToRight ? source.r : source.l;
    const tx = leftToRight ? target.l : target.r;
    if (Math.abs(source.cy - target.cy) < 1.5) {
      return { d: `M${sx} ${source.cy}H${tx}`, lx: (sx + tx) / 2, ly: source.cy - 6 };
    }
    const midX = (sx + tx) / 2;
    return {
      d: `M${sx} ${source.cy}H${midX}V${target.cy}H${tx}`,
      lx: midX, ly: Math.min(source.cy, target.cy) - 6
    };
  }

  function vertical(source, target) {
    if (Math.abs(source.cx - target.cx) < 1.5) {
      return { d: `M${source.cx} ${source.b}V${target.t}`, lx: source.cx + 8, ly: (source.b + target.t) / 2 };
    }
    const midY = (source.b + target.t) / 2;
    return {
      d: `M${source.cx} ${source.b}V${midY}H${target.cx}V${target.t}`,
      lx: (source.cx + target.cx) / 2, ly: midY - 5
    };
  }

  function route(edge) {
    const s = rect(edge.from);
    const t = rect(edge.to);
    const stacked = board.dataset.laView !== 'focus' && root.getBoundingClientRect().width <= STACK_LIMIT;

    if (stacked) {
      if (edge.special === 'entry') {
        const layerRect = root.querySelector('.la-layer').getBoundingClientRect();
        const canvasRect = canvas.getBoundingClientRect();
        const rail = layerRect.right - canvasRect.left - 10;
        const top = (s.b + layerRect.top - canvasRect.top) / 2;
        return { d: `M${s.cx} ${s.b}V${top}H${rail}V${t.t - 22}H${t.cx}V${t.t}`, lx: rail, ly: top - 8, hideLabel: true };
      }
      if (edge.desktop === 'top' || edge.desktop === 'bottom') {
        const side = edge.mobile;
        const layerRect = root.querySelector('.la-layer').getBoundingClientRect();
        const canvasRect = canvas.getBoundingClientRect();
        const rail = side === 'left'
          ? layerRect.left - canvasRect.left + 10
          : layerRect.right - canvasRect.left - 10;
        const fromX = side === 'left' ? s.l : s.r;
        const toX = side === 'left' ? t.l : t.r;
        return {
          d: `M${fromX} ${s.cy}H${rail}V${t.cy}H${toX}`,
          lx: rail, ly: (s.cy + t.cy) / 2, hideLabel: true
        };
      }
      return vertical(s, t);
    }

    if (edge.special === 'entry') {
      const layer = root.querySelector('.la-layer').getBoundingClientRect();
      const c = canvas.getBoundingClientRect();
      const left = layer.left - c.left;
      const rail = Math.max(8, left - 18);
      const downY = s.b + 38;
      return {
        d: `M${s.cx} ${s.b}V${downY}H${rail}V${t.cy}H${t.l}`,
        lx: rail + 7, ly: t.cy - 8
      };
    }

    if (edge.desktop === 'top') {
      const laneRect = root.querySelector('.la-lane--a').getBoundingClientRect();
      const canvasRect = canvas.getBoundingClientRect();
      const lane = laneRect.top - canvasRect.top - 28;
      return {
        d: `M${s.cx} ${s.t}V${lane}H${t.cx}V${t.t}`,
        lx: (s.cx + t.cx) / 2, ly: lane - 5
      };
    }

    if (edge.desktop === 'bottom') {
      const laneRect = root.querySelector('.la-lane--b').getBoundingClientRect();
      const canvasRect = canvas.getBoundingClientRect();
      const lane = laneRect.bottom - canvasRect.top + 34;
      const rail = Math.min(canvas.clientWidth - 10, s.r + 24);
      return {
        d: `M${s.r} ${s.cy}H${rail}V${lane}H${t.cx}V${t.b}`,
        lx: (rail + t.cx) / 2, ly: lane + 18
      };
    }

    if (edge.id === 'x1-norm2' || edge.id === 'y-repeat') return vertical(s, t);
    return horizontal(s, t);
  }

  function draw() {
    drawFrame = 0;
    if (!canvas.clientWidth || !canvas.clientHeight) return;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    svg.setAttribute('width', w);
    svg.setAttribute('height', h);

    for (const edge of EDGES) {
      const item = edgeElements.get(edge.id);
      const layout = route(edge);
      item.path.setAttribute('d', layout.d);
      item.text.setAttribute('x', layout.lx);
      item.text.setAttribute('y', layout.ly);
      item.text.style.display = layout.hideLabel || !edge.label ? 'none' : '';
    }
    board.dataset.laReady = 'true';
    drawRunPulses();
  }

  function scheduleDraw() {
    if (drawFrame) cancelAnimationFrame(drawFrame);
    drawFrame = requestAnimationFrame(draw);
  }

  function setState(next) {
    state = next;
    board.dataset.laState = next;
    board.dataset.laPlaying = String(next === 'playing');
    toggleButton.setAttribute('aria-pressed', String(next === 'playing'));
    toggleButton.textContent = next === 'playing' ? '暂停' : next === 'paused' ? '继续' : next === 'ended' ? '重播' : '播放';
  }

  function updateStep(index, announce = true) {
    stepIndex = Math.max(0, Math.min(STEPS.length - 1, index));
    const current = STEPS[stepIndex];
    board.dataset.laStep = String(stepIndex);
    board.dataset.laActiveEdges = current.edges.join(' ');
    count.textContent = `${String(stepIndex + 1).padStart(2, '0')} / ${String(STEPS.length).padStart(2, '0')}`;
    title.textContent = current.title;
    doText.textContent = current.do;
    const [route, values] = window.LayerAtlasFocusModel.handoffs[stepIndex];
    trace.textContent = route + '  [' + values.map(v => Number(v.toFixed(2))).join(', ') + ']';
    trace.hidden = false;
    nextText.textContent = current.handoff;
    load.textContent = current.load;
    range.max = String(STEPS.length - 1);
    range.step = '1';
    range.setAttribute('aria-label', '回看：选择演示步骤');
    range.removeAttribute('aria-valuetext');
    range.value = String(stepIndex);
    rangeReadout.textContent = `第 ${stepIndex + 1} 步`;
    status.textContent = `步骤 ${stepIndex + 1} / ${STEPS.length}：${current.title}。`;
    explainers.forEach(explainer => {
      const selected = Number(explainer.dataset.laExplainer) === stepIndex;
      explainer.hidden = !selected;
      if (selected) detailTitle.textContent = explainer.dataset.laDetailHeading;
    });

    for (const [id, node] of nodeElements) {
      const active = current.nodes.includes(id);
      const first = firstStep.has(id) ? firstStep.get(id) : stepIndex;
      node.dataset.laState = active ? 'active' : first < stepIndex ? 'done' : 'pending';
      if (node.dataset.laPickable) node.setAttribute('aria-pressed', String(active));
    }
    for (const [id, item] of edgeElements) {
      item.path.classList.toggle('is-active', current.edges.includes(id));
      item.text.classList.toggle('is-active', current.edges.includes(id));
    }
    stepButtons.forEach((button, i) => {
      button.classList.toggle('is-done', i < stepIndex);
      if (i === stepIndex) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
    prevButton.disabled = stepIndex === 0;
    nextButton.disabled = stepIndex === STEPS.length - 1;
    if (!announce) status.textContent = '';
  }

  function clearTimer() {
    if (timer) clearTimeout(timer);
    timer = 0;
  }

  function pause(message) {
    clearTimer();
    setState('paused');
    status.textContent = message || `已暂停在第 ${stepIndex + 1} 步：${STEPS[stepIndex].title}。`;
  }

  function tick() {
    clearTimer();
    if (state !== 'playing') return;
    if (stepIndex >= STEPS.length - 1) {
      setState('ended');
      status.textContent = '演示结束。可回看任一步，或点“重播”从起点再走一遍。';
      return;
    }
    updateStep(stepIndex + 1);
    timer = setTimeout(tick, STEP_HOLD);
  }

  function play() {
    clearTimer();
    if (state === 'playing') {
      pause();
      return;
    }
    if (state === 'ended' || stepIndex >= STEPS.length - 1) updateStep(0, false);
    setState('playing');
    status.textContent = `播放中：第 ${stepIndex + 1} 步，${STEPS[stepIndex].title}。`;
    timer = setTimeout(tick, STEP_HOLD);
  }

  function stepTo(index, message) {
    clearTimer();
    setState('idle');
    updateStep(index);
    if (message) status.textContent = message;
  }

  function placePulse(pulse, edge, progress, dimension) {
    const path=edgeElements.get(edge).path;
    const point=path.getPointAtLength(path.getTotalLength()*progress);
    pulse.style.display='';
    pulse.setAttribute('transform',`translate(${point.x} ${point.y})`);
    pulse.dataset.edge=edge;
    pulse.dataset.dimension=String(dimension);
    [...pulse.children].slice(1).forEach((bar,i)=>{
      bar.style.display=i<dimension?'':'none';
      bar.setAttribute('x',String(3*i-(3*dimension-1)/2));
    });
  }

  function drawRunPulses() {
    if(!pulses.length)return;
    const frame=run?.frame;
    if(!frame||frame.type!=='transfer'){
      pulses.forEach(pulse=>{pulse.style.display='none';});
      return;
    }
    const {chapter,progress}=frame;
    const travel=Math.min(1,progress/.82);
    const stationary=reducedMotion.matches;
    const split=.48;
    const hasJunction=chapter.edges.length===2;
    const reachedJunction=hasJunction&&travel>=split;
    const along=hasJunction?(reachedJunction?(travel-split)/(1-split):travel/split):travel;
    placePulse(pulses[0],chapter.edges[Number(reachedJunction)],stationary?Number(along===1):along,chapter.value.length);
    if(chapter.branch&&reachedJunction){
      placePulse(pulses[1],chapter.branch,stationary?Number(along===1):along,chapter.value.length);
    }else{
      pulses[1].style.display='none';
    }
  }

  function renderRun() {
    const frame=model.sampleRun(fullRun,run.time);
    const {chapter,target,type,progress,ended}=frame;
    const previous=run.frame;
    run.frame=frame;
    if(previous?.index!==frame.index)updateStep(chapter.overview,false);
    trace.hidden=true;
    const computing=type==='compute';
    let seconds=computing?frame.from+(frame.until-frame.from)*progress:frame.until;
    if(computing&&reducedMotion.matches&&!ended){
      const scene=focusController.currentId===chapter.id&&focusController.scene?.kind===frame.kind
        ? focusController.scene : model.make(frame.kind,0,chapter.id);
      seconds=scene.ends.filter(end=>end<=seconds).at(-1)??frame.from;
    }
    const enter=frame.enters&&!reducedMotion.matches?Math.min(1,progress*6):1;
    focusController.present(chapter.id,frame.kind,seconds,enter,run.playing&&computing);
    if(previous?.type!==type||previous?.index!==frame.index){
      const routes=computing?[]:[...chapter.edges,...(chapter.branch?[chapter.branch]:[])];
      for(const [id,item] of edgeElements){
        item.path.classList.toggle('is-active',routes.includes(id));
        item.text.classList.toggle('is-active',routes.includes(id));
      }
      board.dataset.laActiveEdges=routes.join(' ');
    }
    const arrived=!computing&&progress>=.82;
    const activeId=arrived?target.id:chapter.id;
    if(previous?.index!==frame.index||run.highlight!==activeId){
      run.highlight=activeId;
      for(const [id,node] of nodeElements){
        node.dataset.laState=id===activeId?'active':fullRun.chapters.slice(0,frame.index).some(c=>c.id===id)?'done':'pending';
        if(node.dataset.laPickable){
          node.setAttribute('aria-pressed',String(id===activeId));
          node.setAttribute('aria-expanded',String(id===chapter.id));
        }
      }
    }
    drawRunPulses();
    board.dataset.laRun=ended?'ended':run.playing?'playing':'paused';
    board.dataset.laRunPhase=type;
    board.dataset.laRunModule=chapter.id;
    write(count,`${String(frame.index+1).padStart(2,'0')} / ${fullRun.chapters.length}`);
    write(title,ended?'已生成「'+fullRun.selected+'」':chapter.label);
    write(runRoute,ended?'生成结果':computing?'正在计算 · '+chapter.label:`${chapter.label} → ${target.label}`);
    write(runValue,ended?fullRun.selected:computing?'':`${chapter.symbol} ${formatVector(chapter.value)}${arrived?' · 已送达':''}`);
    write(runBranch,!computing&&chapter.branch?'旁路保留 '+chapter.symbol:'');
    const local=focusController.scene;
    const subStep=model.sample(local,focusController.time);
    write(doText,ended?'已把「'+fullRun.selected+'」追加到上下文。':computing
      ? (frame.kind==='heads'?'两组头的结果已就绪。':'')+local.steps[subStep.phase].why
      : '沿高亮连线传递上一步的输出。');
    const percent=Math.round(run.time/fullRun.duration*100);
    range.max='1000';range.step='1';range.value=String(Math.round(run.time/fullRun.duration*1000));
    range.setAttribute('aria-label','完整流程播放进度');
    range.setAttribute('aria-valuetext',`${chapter.label}，${percent}%`);
    write(rangeReadout,percent+'%');
    write(toggleButton,ended?'重播完整流程':run.playing?'暂停':'继续');
    toggleButton.setAttribute('aria-pressed',String(run.playing));
    prevButton.disabled=run.time===0;
    nextButton.disabled=frame.index===fullRun.chapters.length-1;
    write(status,ended?'完整播放结束，已生成「'+fullRun.selected+'」。':`${chapter.label}，${computing?'计算中':'传递输出'}。`);
  }

  function runTick(now) {
    if(!run?.playing)return;
    if(run.last)run.time=Math.min(fullRun.duration,run.time+(now-run.last)/1000*focusController.speed);
    run.last=now;
    if(run.time>=fullRun.duration)run.playing=false;
    renderRun();
    run.raf=run.playing?requestAnimationFrame(runTick):0;
  }

  function setRunPlaying(playing) {
    cancelAnimationFrame(run.raf);
    run.playing=playing;
    run.last=0;
    renderRun();
    run.raf=playing?requestAnimationFrame(runTick):0;
  }

  function startRun() {
    clearTimer();
    setState('idle');
    detail.open=false;
    run={ time:0, raf:0 };
    runHandoff.hidden=false;
    fullButton.setAttribute('aria-pressed','true');
    write(fullButton,'退出完整播放');
    write(prevButton,'← 上一模块');prevButton.setAttribute('aria-label','上一模块');
    write(nextButton,'下一模块 →');nextButton.setAttribute('aria-label','下一模块');
    setRunPlaying(true);
  }

  function stopRun() {
    if(!run)return;
    cancelAnimationFrame(run.raf);
    run=null;
    delete board.dataset.laRun;
    delete board.dataset.laRunPhase;
    delete board.dataset.laRunModule;
    runHandoff.hidden=true;
    fullButton.setAttribute('aria-pressed','false');
    write(fullButton,'完整播放');
    write(prevButton,'← 上一步');prevButton.setAttribute('aria-label','上一步');
    write(nextButton,'下一步 →');nextButton.setAttribute('aria-label','下一步');
    drawRunPulses();
  }

  function seekRun(time) {
    run.time=Math.max(0,Math.min(fullRun.duration,time));
    setRunPlaying(false);
  }

  function goToRunModule(offset) {
    const current=run.frame.index;
    const previous=offset<0&&run.time>fullRun.chapters[current].start+.02?current:current+offset;
    seekRun(fullRun.chapters[Math.max(0,Math.min(fullRun.chapters.length-1,previous))].start);
  }

  function bind() {
    controls.forEach(item => { item.hidden = false; });
    fullButton.addEventListener('click',()=>{
      if(!run){startRun();return;}
      stopRun();focusController.close();
    });
    prevButton.addEventListener('click', () => run ? goToRunModule(-1) : focusController.active ? focusController.prev() : stepTo(stepIndex - 1));
    nextButton.addEventListener('click', () => run ? goToRunModule(1) : focusController.active ? focusController.next() : stepTo(stepIndex + 1));
    resetButton.addEventListener('click', () => run ? seekRun(0) : focusController.active ? focusController.reset() : stepTo(0, '已回到起点。'));
    toggleButton.addEventListener('click', () => {
      if(run){
        if(run.time>=fullRun.duration)startRun();
        else setRunPlaying(!run.playing);
      }else if(focusController.active)focusController.toggle();
      else play();
    });
    range.addEventListener('input', () => {
      const value=Number(range.value);
      if(run)seekRun(value/1000*fullRun.duration);
      else if(focusController.active)focusController.seek(value/1000);
      else stepTo(value);
    });
    stepButtons.forEach((button, index) => button.addEventListener('click', () => stepTo(index)));
    for (const [id, node] of nodeElements) {
      if (node.classList.contains('la-node--junction')) continue;
      node.dataset.laPickable = 'true';
      node.setAttribute('role', 'button');
      node.setAttribute('tabindex', '0');
      node.setAttribute('aria-controls', 'la-inline-focus');
      const inspect = () => {
        if (focusController.currentId === id) return;
        stopRun();
        stepTo(firstStep.get(id));
        detail.open = false;
        focusController.open(id, node);
      };
      node.addEventListener('click', inspect);
      node.addEventListener('keydown', event => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        inspect();
      });
    }
    detail.addEventListener('toggle', () => {
      if (!detail.open) return;
      if (run) setRunPlaying(false);
      else if (focusController.active) focusController.seek(focusController.time / focusController.scene.duration);
      else if (state === 'playing') pause('已展开计算原理，演示暂停在当前步骤。');
    });
    root.addEventListener('keydown', event => {
      if (event.key === 'Escape' && focusController.active) {
        event.preventDefault();stopRun();focusController.close();
      }
    });
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && run?.playing) setRunPlaying(false);
      else if (document.hidden && state === 'playing') pause('页面切到后台，演示已暂停。');
    });
    window.addEventListener('pagehide',()=>{
      if(run?.playing)setRunPlaying(false);
      clearTimer();
    });
    new ResizeObserver(scheduleDraw).observe(canvas);
    window.addEventListener('resize', scheduleDraw, { passive: true });
  }

  focusController = window.LayerAtlasFocus.create(root, {
    frame(frame) {
      if(run)return;
      const put = (element, value) => { if (element.textContent !== value) element.textContent = value; };
      put(count, frame.count); put(title, frame.title); put(doText, frame.description);
      trace.hidden = true;
      const percent = Math.round(frame.progress * 100);
      range.max = '1000'; range.step = '1'; range.value = String(Math.round(frame.progress * 1000));
      range.setAttribute('aria-label', '选中模块内部计算的连续进度');
      range.setAttribute('aria-valuetext', frame.title + '，' + percent + '%');
      put(rangeReadout, percent + '%');
      put(toggleButton, frame.reduced ? '下一计算步' : frame.playing ? '暂停' : frame.ended ? '重播计算' : '播放计算');
      toggleButton.setAttribute('aria-pressed', String(frame.playing));
      prevButton.disabled = !frame.previous; nextButton.disabled = !frame.next;
      board.dataset.laPlaying = 'false';
      put(status, '模块内部：' + frame.title + '。');
    },
    view() {
      if (!run && focusController && focusController.active) for (const [id, node] of nodeElements) {
        const selected = id === focusController.currentId;
        node.dataset.laState = selected ? 'active' : 'pending';
        if (node.dataset.laPickable) { node.setAttribute('aria-pressed', String(selected)); node.setAttribute('aria-expanded', String(selected)); }
      }
      draw();
    },
    close() {
      if(run)return;
      setState('idle'); updateStep(stepIndex);
      for (const node of nodeElements.values()) if (node.dataset.laPickable) node.setAttribute('aria-expanded','false');
    },
    beforeClose:stopRun
  });
  createSvg();
  bind();
  updateStep(0, false);
  function revealScaling() {
    if (location.hash !== '#attention-scaling') return;
    stopRun();
    if (focusController.active) focusController.close();
    stepTo(2);
    detail.open = true;
    const section = root.querySelector('#attention-scaling');
    section.open = true;
    requestAnimationFrame(() => section.scrollIntoView({ block: 'start' }));
  }
  revealScaling();
  window.addEventListener('hashchange', revealScaling);
  requestAnimationFrame(draw);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(scheduleDraw).catch(() => {});
})();
