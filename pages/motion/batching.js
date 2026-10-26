/* Batching is a sequence-growing mechanism, not a three-column card workflow.
 * Consume the existing reversible snapshots; keep request and token nodes alive.
 * Coordinates are diagram positions, never physical GPU cores or measured time. */
(() => {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs = {}, text) {
    const node = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  }
  const clamp = n => Math.max(0, Math.min(1, n));
  window.BatchingVisual = function (svg, viewport, scenario) {
    viewport.classList.add('batch-viewport');
    viewport.setAttribute('aria-label', '等待队列与 GPU 执行位置；方块表示 token');
    const backdrop = el('g'), actorsLayer = el('g');
    svg.append(backdrop, actorsLayer);
    // Switch the camera to the batch type scale: this board is much taller
    // than a 360x520 foundation board, so the generic scale is too small.
    svg.dataset.teachingKind = 'batching';
    const totals = [1, 3, 1], letters = ['A', 'B', 'C'];
    // The same requests and arrival condition are shown for both policies, so
    // the reader does not have to remember one schedule while watching the other.
    const CONTRAST = {
      fixed: { label: '固定批次', rounds: ['A + B', 'B', 'B', 'C'], entry: 'C 第 4 轮' },
      continuous: { label: '连续批处理', rounds: ['A + B', 'B + C', 'B', '已完成'], entry: 'C 第 2 轮' }
    };
    const currentMode = scenario.name === '固定批次' ? 'fixed' : 'continuous';
    function contrastPanel() {
      const box = document.createElement('section'), head = document.createElement('div');
      box.className = 'batch-contrast';
      box.setAttribute('aria-label', '同一输入的固定批次与连续批处理逐轮对照');
      head.className = 'batch-contrast-head';
      head.append(
        Object.assign(document.createElement('strong'), { textContent: '同一输入 · 两种批次成员表' }),
        Object.assign(document.createElement('span'), { textContent: 'A 输出 1 · B 输出 3 · C 输出 1；C 在第 1 轮期间到达；最多同时 2 条' })
      );
      const table = document.createElement('table');
      table.setAttribute('aria-label', '逐轮成员对照；两种策略使用同一组请求');
      const thead = document.createElement('thead'), headRow = document.createElement('tr');
      for (const text of ['策略', '第 1 轮', '第 2 轮', '第 3 轮', '第 4 轮']) {
        const th = document.createElement('th'); th.scope = 'col'; th.textContent = text; headRow.append(th);
      }
      thead.append(headRow);
      const tbody = document.createElement('tbody'), rows = {};
      for (const key of ['fixed', 'continuous']) {
        const data = CONTRAST[key], tr = document.createElement('tr'), th = document.createElement('th');
        tr.dataset.batchRow = key; th.scope = 'row';
        th.append(Object.assign(document.createElement('strong'), { textContent: data.label }),
          Object.assign(document.createElement('small'), { textContent: data.entry }));
        const cells = data.rounds.map((text, i) => {
          const td = document.createElement('td');
          td.textContent = text; td.dataset.round = String(i + 1);
          if (text.includes('C')) td.classList.add('is-c');
          return td;
        });
        tr.append(th, ...cells); tbody.append(tr); rows[key] = { tr, cells };
      }
      table.append(thead, tbody);
      const scale = document.createElement('p');
      scale.className = 'batch-contrast-scale'; scale.textContent = '生成轮次，不代表实际耗时';
      const note = document.createElement('p');
      note.className = 'batch-contrast-note'; note.setAttribute('aria-live', 'polite');
      box.append(head, scale, table, note);
      let shown = '';
      return {
        box,
        update(round) {
          const current = Math.max(1, Math.min(4, round)), key = current + ':' + currentMode;
          if (key === shown) return;
          shown = key;
          for (const [rowKey, row] of Object.entries(rows)) {
            row.tr.classList.toggle('is-active', rowKey === currentMode);
            row.cells.forEach((cell, i) => cell.classList.toggle('is-current', i === current - 1));
          }
          const otherKey = currentMode === 'fixed' ? 'continuous' : 'fixed';
          note.textContent = `当前 ${CONTRAST[currentMode].label} · 第 ${current} 轮：${CONTRAST[currentMode].rounds[current - 1] || '已完成'}；同轮 ${CONTRAST[otherKey].label}：${CONTRAST[otherKey].rounds[current - 1] || '已完成'}。`;
        }
      };
    }
    const contrast = contrastPanel();
    (viewport.closest('.motion-theater') || viewport).before(contrast.box);
    const actors = letters.map((id, i) => {
      const g = el('g', { 'data-motion-id': id, class: `batch-request batch-request-${id}` });
      g.append(el('text', { x: 0, y: 22, class: 'batch-request-name' }, id));
      for (let j = 0; j < 2; j++) {
        g.append(el('rect', { x: 32 + j * 35, y: 0, width: 29, height: 32, rx: 4, class: 'batch-context-token' }),
          el('text', { x: 46.5 + j * 35, y: 21, class: 'batch-token-label' }, '·'));
      }
      const tokens = Array.from({ length: totals[i] }, (_, j) => {
        const cell = el('g', { 'data-output-token': j + 1 });
        const block = el('rect', { x: 102 + j * 35, y: 0, width: 0, height: 32, rx: 4, class: 'batch-output-token' });
        const label = el('text', { x: 116.5 + j * 35, y: 21, class: 'batch-output-label' }, String(j + 1));
        // The outline marks how many outputs this teaching request still needs.
        g.append(el('rect', { x: 102 + j * 35, y: 0, width: 29, height: 32, rx: 4, class: 'batch-needed-token' }));
        cell.append(block, label); g.append(cell);
        return { cell, block, label };
      });
      const note = el('text', { x: 32, y: 52, class: 'batch-request-note' });
      g.append(note); actorsLayer.append(g);
      return { g, tokens, note };
    });
    let narrow, slotLabels, iterationLabel, ledger, gate, gateNote, last;
    const rounds = [];
    // A generation round is exactly a snapshot transition that adds output.
    scenario.frames.forEach((frame, i) => {
      if (!i) return;
      const members = frame.actors.flatMap((actor, j) => actor.count > scenario.frames[i - 1].actors[j].count ? [j] : []);
      if (members.length) rounds.push({ endpoint: i, members });
    });
    function layout() {
      const next = viewport.clientWidth < 600;
      if (next === narrow) return;
      narrow = next;
      svg.setAttribute('viewBox', narrow ? '0 0 360 1060' : '0 0 900 630');
      backdrop.replaceChildren();
      function text(x, y, value, cls = 'batch-region-title') {
        const t = el('text', { x, y, class: cls }, value); backdrop.append(t); return t;
      }
      function region(x, y, w, h, title, note, tx, titleOffset = 27, noteOffset = 57) {
        backdrop.append(el('rect', { x, y, width: w, height: h, rx: 10, class: 'batch-region' }));
        text((tx ?? x + 16), y + titleOffset, title); text(x + 16, y + noteOffset, note, 'batch-small');
      }
      text(16, narrow ? 26 : 22, narrow ? '② 生成中 · 小方块是 token，不是请求' : '② 生成中 · 一行是一条请求；浅色已有，实色新增，虚框还需生成', 'batch-key');
      region(12, 40, narrow ? 336 : 238, narrow ? 220 : 266, '等待队列', 'C 在第一轮期间进入等待队列');
      // On the narrow board C descends from the queue through this region, so
      // its caption sits below the slots instead of in the path.
      region(narrow ? 12 : 274, narrow ? 282 : 40, narrow ? 336 : 614, 266, 'GPU · 一起执行这一轮', '每条参与请求，各新增一个 token', undefined, narrow ? 224 : 27, narrow ? 250 : 57);
      slotLabels = [0, 1].map(i => {
        const y = (narrow ? 336 : 108) + i * (narrow ? 74 : 96);
        backdrop.append(el('rect', { x: narrow ? 24 : 290, y, width: narrow ? 312 : 582, height: narrow ? 68 : 80, rx: 5, class: 'batch-slot', 'data-batch-slot': i }));
        return text(narrow ? 40 : 308, y + (narrow ? 44 : 48), '空位', 'batch-empty');
      });
      iterationLabel = text(narrow ? 28 : 290, narrow ? 574 : 334, '', 'batch-iteration');
      ledger = rounds.map((round, i) => {
        // Leave a right-hand aisle for requests leaving the GPU row: the round
        // boxes must not sit under that exit path on either width.
        const x = (narrow ? 16 : 290) + i * (narrow ? 72 : 124), y = narrow ? 594 : 354;
        const cell = el('g');
        cell.append(el('rect', { x, y, width: narrow ? 64 : 104, height: narrow ? 56 : 52, rx: 4, class: 'batch-round-box' }),
          el('text', { x: x + 8, y: y + 19, class: 'batch-small' }, `第 ${i + 1} 轮`));
        const members = el('text', { x: x + 8, y: y + (narrow ? 45 : 42), class: 'batch-round-members' }); cell.append(members);backdrop.append(cell);
        return { cell, members };
      });
      gate = text(20, narrow ? 680 : 440, '', 'batch-region-title');
      gateNote = text(20, narrow ? 706 : 468, '', 'batch-small');
      text(16, narrow ? 764 : 516, '已完成 · 不再占用执行位置');
      backdrop.append(el('line', { x1: 16, x2: narrow ? 344 : 884, y1: narrow ? 776 : 528, y2: narrow ? 776 : 528, class: 'batch-divider' }));
      if (narrow) text(16, 1040, 'A：1 个输出 · B：3 个输出 · C：1 个输出', 'batch-small');
    }
    function position(a, index) {
      if (a.x === 48) return { x: narrow ? 40 : 34, y: (narrow ? 112 : 116) + (index === 1 ? 86 : index === 2 ? 42 : 0), scale: 1 };
      if (a.x === 348) return { x: narrow ? 48 : 326, y: (narrow ? 344 : 122) + (a.y === 108 ? 0 : narrow ? 74 : 96), scale: 1 };
      return narrow ? { x: 48, y: 798 + index * 76, scale: 1 } : { x: 34 + index * 288, y: 550, scale: 1 };
    }
    function paint(frame, target, u, p) {
      last = [frame, target, u, p];
      actors.forEach((actor, i) => {
        const from = frame.actors[i], to = target.actors[i], a = position(from, i), b = position(to, i);
        // Travel horizontally out of the queue, then into the selected row. Keep
        // the unfinished request stationary so the refill cannot look like a reset.
        const mix = (x, y) => x + (y - x) * u;
        let x = mix(a.x, b.x), y = mix(a.y, b.y), scale = mix(a.scale, b.scale);
        if (from.x === 348 && to.x === 648) {
          // Exit along the right-hand aisle, never through ledger, gate or the
          // unfinished B sequence. The final drop uses a column clear of the
          // completion caption before sliding onto the shelf.
          const edge = narrow ? 326 : 780, small = narrow ? .18 : .52;
          if (narrow) {
            if (u < .12) { const t = u / .12; x = a.x + (edge - a.x) * t; y = a.y; scale = 1 + (small - 1) * t; }
            else if (u < .72) { x = edge; y = a.y + (b.y - a.y) * (u - .12) / .6; scale = small; }
            else { const t = (u - .72) / .28; x = edge + (b.x - edge) * t; y = b.y; scale = small + (1 - small) * t; }
          } else {
            const lane = Math.max(b.x, 200);
            if (u < .12) { const t = u / .12; x = a.x + (edge - a.x) * t; y = a.y; scale = 1 + (small - 1) * t; }
            else if (u < .58) { x = edge; y = a.y + (474 - a.y) * (u - .12) / .46; scale = small; }
            else if (u < .7) { x = edge; y = 474; scale = small; }
            else if (u < .8) { x = edge + (lane - edge) * (u - .7) / .1; y = 474; scale = small; }
            else if (u < .92) { x = lane; y = 474 + (550 - 474) * (u - .8) / .12; scale = small; }
            else { const t = (u - .92) / .08; x = lane + (b.x - lane) * t; y = 550; scale = small + (1 - small) * t; }
          }
        }
        const count = mix(from.count, to.count), opacity = mix(from.opacity, to.opacity);
        actor.g.setAttribute('transform', `translate(${x.toFixed(2)} ${y.toFixed(2)}) scale(${scale.toFixed(3)})`);
        actor.g.setAttribute('opacity', opacity.toFixed(3));
        Object.assign(actor.g.dataset, { x: x.toFixed(3), y: y.toFixed(3), count: count.toFixed(3) });
        actor.g.setAttribute('aria-label', `请求 ${letters[i]}，本例已新增 ${Math.floor(count + 1e-6)} 个 token，共需 ${totals[i]} 个`);
        actor.tokens.forEach(({ cell, block, label }, j) => {
          const fill = clamp(count - j);
          block.setAttribute('width', (29 * fill).toFixed(3));
          label.setAttribute('opacity', fill === 1 ? 1 : 0);
          cell.dataset.fill = fill.toFixed(3);
        });
        const completed = from.x === 648 && to.x === 648;
        actor.note.textContent = completed ? '已完成' : `${Math.floor(count + 1e-6)} / ${totals[i]} 个新增输出`;
      });
      slotLabels.forEach((label, slot) => {
        const row = slot === 0 ? 108 : 184;
        const occupied = frame.actors.some(a => a.x === 348 && a.y === row && a.opacity) || (u > 0 && target.actors.some(a => a.x === 348 && a.y === row && a.opacity));
        label.setAttribute('opacity', occupied ? 0 : 1);
        label.textContent = slot === 0 && scenario.name === '固定批次' && p >= 3 && frame.actors[1].x !== 648 ? '空位留着，等 B 结束' : '空位 · 下一次准入时可补入';
      });
      const fixed = scenario.name === '固定批次';
      gate.textContent = fixed ? '换成员条件：原批成员全部结束' : '换成员条件：轮次边界 + 空位 + 准入';
      const cEntered = frame.actors[2].x !== 48;
      gateNote.textContent = p < 2 ? '先让 A、B 执行；C 还没就绪。' : p < 3 ? 'A 正在退出；B 继续保留自己的历史。' : !cEntered ? (fixed ? (frame.actors[1].x === 648 ? 'A、B 均已退出 → C 现在可加入下一批。' : 'B 还未退出 → 即使有空位，C 也继续等。') : '本例资源满足 → C 可在下一轮加入。') : frame.actors[2].count < 1 ? (fixed ? 'C 已入批，将在第 4 轮参与生成。' : 'C 已入批，将在第 2 轮与 B 一起生成。') : fixed ? '结果：C 等 A、B 全部退出，第 4 轮才参与。' : '结果：C 第 2 轮参与；B 没有重新开始。';
      const completedRounds = rounds.filter(r => p >= r.endpoint - 1e-6).length;
      const growing = rounds.findIndex(r => p > r.endpoint - 1 && p < r.endpoint);
      const roundNow = growing >= 0 ? growing + 1 : Math.max(1, completedRounds);
      contrast.update(roundNow);
      iterationLabel.textContent = growing >= 0 ? `第 ${growing + 1} 轮正在生成 · 新方块从右侧长出` : innerWidth <= 340 ? `已执行 ${completedRounds} 轮 · Decode 轮次，不是等待时间` : `已执行 ${completedRounds} 轮 · 这是 Decode 轮次，不是等待时间`;
      ledger.forEach(({ cell, members }, i) => {
        const visible = p > rounds[i].endpoint - 1;
        cell.classList.toggle('is-generated', p >= rounds[i].endpoint);
        cell.classList.toggle('is-generating', i === growing);
        members.textContent = visible ? rounds[i].members.map(n => letters[n]).join(' + ') : '待执行';
      });
    }
    const observer = new ResizeObserver(() => { layout(); if (last) paint(...last); });
    layout();observer.observe(viewport);
    return { paint, destroy() { observer.disconnect(); contrast.box.remove(); } };
  };
})();
