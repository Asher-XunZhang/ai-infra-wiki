/* Browser/Node solver for the dependency template exported by the Python model. */
((scope) => {
  'use strict';
  const current = n => n >= 3 && n <= 7 ? n - 2 : null;
  const old = n => current(n - 2);
  const key = (r, n, name) => `${r}:${n}:${name}`;
  const round = value => Math.round((value + Number.EPSILON) * 1000) / 1000;
  const ensure = (condition, message) => { if (!condition) throw new Error(message); };
  const phases = 'ABCDEFGHI';
  const LIMIT = 512;

  function validate(timing) {
    ensure(timing && typeof timing === 'object', '自定义参数缺失。');
    const number = (n, min, max) => typeof n === 'number' && Number.isFinite(n) && n >= min && n <= max;
    for (const [name, columns] of [['gpu', 5], ['h2d', 2], ['kv', 5]]) {
      ensure(Array.isArray(timing[name]) && timing[name].length === 3 && timing[name].every(row =>
        Array.isArray(row) && row.length === columns && row.every(n => number(n, 0, 100))),
      'GPU、回载和 KV 时间需为 0–100 u 的完整表格。');
    }
    ensure(timing.cpu && [...phases].every(p => number(timing.cpu[p], .1, 10)), 'CPU 阶段倍率需为 0.1–10。');
    ensure(['proxy', 'output', 'control', 'd2h'].every(p => number(timing[p], 0, 100)), '通信和复制时间需为 0–100 u。');
    return timing;
  }

  function instantiate(template, timing, horizon) {
    const nodes = {};
    const source = Array.from({length:3}, () => Array.from({length:template.repeatFrom+1}, () => []));
    for (const node of template.nodes) source[node.r][node.n].push(node);
    const shift = (id, delta) => {
      if (!id) return id;
      const parts = id.split(':'); parts[1] = Number(parts[1]) + delta; return parts.join(':');
    };
    for (let r = 0; r < 3; r++) for (let n = 0; n < horizon; n++) {
      const from = Math.min(n, template.repeatFrom), delta = n - from;
      for (const item of source[r][from]) {
        const v = {...item, id:shift(item.id, delta), n, deps:item.deps.map(d => shift(d, delta))};
        for (const attr of ['prev','launch','previous_gpu']) if (attr in v) v[attr] = shift(v[attr], delta);
        if (v.kind === 'gpu') v.duration = timing.gpu[r][current(n)-1];
        else if (v.kind === 'h2d') v.duration = timing.h2d[r][Number(current(n) === 4)];
        else if (v.kind === 'kv') v.duration = timing.kv[r][old(n)-1];
        else if (v.kind === 'cpu') v.duration *= timing.cpu[v.phase];
        else if (v.kind === 'copy') v.duration = timing.d2h;
        else if (v.kind === 'message') v.duration = timing[v.id.endsWith('proxy_message') ? 'proxy' : v.id.endsWith('out_message') ? 'output' : 'control'];
        nodes[v.id] = v;
      }
    }
    return nodes;
  }

  function solve(nodes) {
    const degree = {}, children = {}, ready = [], resolved = [];
    for (const [id, v] of Object.entries(nodes)) {
      delete v.start; delete v.end; degree[id] = v.deps.length;
      if (!degree[id]) ready.push(id);
      for (const d of v.deps) (children[d] ||= []).push(id);
    }
    for (let i = 0; i < ready.length; i++) {
      const id = ready[i], v = nodes[id];
      v.start = Math.max(0, ...v.deps.map(d => nodes[d].end)); v.end = v.start + v.duration;
      resolved.push(id);
      for (const child of children[id] || []) if (--degree[child] === 0) ready.push(child);
    }
    return resolved;
  }

  function compute(template, timing, horizon) {
    const nodes = instantiate(template, timing, horizon), ackBase = {};
    for (const [id, v] of Object.entries(nodes)) if (id.endsWith(':ack_events')) ackBase[id] = [...v.deps];
    let last = '', stable = false;
    for (let attempt = 0; attempt < 32; attempt++) {
      solve(nodes);
      const remaining = [4,6], rounds = [];
      for (let n = 0; n < horizon; n++) {
        const poll = nodes[key(0,n,'l2_counts')], loads = [];
        while (remaining.length && remaining[0] < n && Number.isFinite(poll.start) && nodes[key(0,remaining[0],'h2d')].end <= poll.start) loads.push(remaining.shift());
        rounds.push(loads);
      }
      const signature = JSON.stringify(rounds);
      if (signature === last) { stable = true; break; }
      last = signature;
      for (let r = 0; r < 3; r++) rounds.forEach((loads,n) => {
        const id = key(r,n,'ack_events');
        nodes[id].deps = [...ackBase[id], ...loads.map(j => key(r,j,'h2d'))];
        nodes[id].owner = '已公布 ACK：' + (loads.length ? loads.map(j => `M${current(j)} 本地 H2D event`).join(', ') : '本轮计数为 0');
      });
    }
    ensure(stable, '这组参数的 ACK 轮次未收敛，请调整回载或 CPU 时间。');
    const resolved = solve(nodes), terminal = {}, released = [new Set(),new Set(),new Set()], releases = [];
    for (let n = 0; n < horizon - 3; n++) {
      for (let r = 0; r < 3; r++) {
        const poll = nodes[key(r,n,'term_poll')];
        if (!Number.isFinite(poll.end)) continue;
        terminal[`${r}:${n}`] = [1,2,3,4,5].filter(m => nodes[key(r,m+4,'kv')].end <= poll.start && !released[r].has(m) &&
          (r === 0 || (terminal[`${r-1}:${n}`] || []).includes(m)));
      }
      if (n < 2) continue;
      for (let r = 0; r < 3; r++) {
        const p = nodes[key(r,n,'release')];
        const actual = (terminal[`2:${n-2}`] || []).filter(m => nodes[key(r,m+4,'kv')].end <= p.start && !released[r].has(m));
        if (actual.length) {
          actual.forEach(m => released[r].add(m)); p.owner = '释放 ' + actual.map(m => 'M'+m).join(', ');
          releases.push({r,n,batches:actual,time:p.start});
        }
      }
    }
    if (!released.every(set => set.size === 5)) return null;
    const displayN = Math.max(...releases.map(v => v.n)) + 1, events = [], loops = [], values = Object.values(nodes);
    const sequences = Array.from({length:3}, () => Array.from({length:displayN}, () => []));
    for (const v of values) if (v.kind === 'cpu' && v.n < displayN) sequences[v.r][v.n].push(v);
    const event = v => Object.fromEntries(['id','r','n','kind','phase','start','end','label','owner','ref'].map(k => [k,v[k]]));
    for (let r = 0; r < 3; r++) for (let n = 0; n < displayN; n++) {
      const seq = sequences[r][n];
      ensure(seq.every(v => Number.isFinite(v.end)), '依赖未能闭合，请调整参数。');
      loops.push({r,n,slot:n%3,start:seq[0].prev ? nodes[seq[0].prev].end : 0,end:seq.at(-1).end,current:current(n),old:old(n)});
      for (const e of seq) {
        const prevEnd = e.prev ? nodes[e.prev].end : 0;
        if (e.start > prevEnd + 1e-7) events.push({id:e.id+':wait',target:e.id,r,n,kind:'wait',phase:e.phase,start:prevEnd,end:e.start,
          label:'等：'+e.label,owner:e.owner,ref:e.ref,deps:e.deps.filter(d => d !== e.prev && nodes[d].end > prevEnd + 1e-7)});
        events.push(event(e));
      }
    }
    for (const e of values) if (['gpu','h2d','kv'].includes(e.kind) && Number.isFinite(e.end) && e.n < displayN) events.push(event(e));
    for (let r = 0; r < 3; r++) {
      let prevEnd = 0;
      for (let n = 3; n < 8; n++) {
        const g = nodes[key(r,n,'gpu')], launch = nodes[key(r,n,'launch')], a = prevEnd, b = Math.max(prevEnd,launch.end);
        if (b > a + .001) {
          const causes = events.filter(e => e.r === r && e.kind === 'wait' && e.start < b && e.end > a)
            .sort((x,y) => (Math.min(b,y.end)-Math.max(a,y.start)) - (Math.min(b,x.end)-Math.max(a,x.start)));
          events.push({id:g.id+':idle',target:g.id,r,n,kind:'idle',phase:'W',start:a,end:b,
            label:n === 3 ? '填充等待' : '空泡：'+(causes[0]?.label || 'CPU 调度 / 准备 / 提交'),owner:'下一份 '+g.owner,ref:'pp:223'});
        }
        if (g.start > b + .001) events.push({id:g.id+':layer-wait',target:g.id,r,n,kind:'idle',phase:'W',start:b,end:g.start,
          label:'GPU 等全部 H2D（模型粗化）',owner:g.owner,ref:'controller:64'});
        prevEnd = g.end;
      }
    }
    const graph = {}, defaultRefs = {out_message:'pp:983',proxy_message:'pp:340',req_message:'pp:327',boot_message:'pp:331',term_message:'pp:334',copy:'pp:1084'};
    for (const id of resolved) {
      const e = nodes[id], {r,n} = e, name = id.split(':')[2], inputs = [];
      const g = Object.fromEntries(['r','n','kind','phase','label','owner','deps'].map(k => [k,e[k]]));
      Object.assign(g,{start:round(e.start),end:round(e.end),ref:e.ref || defaultRefs[name] || 'pp:221'});
      if (e.prev) g.prev = e.prev;
      if (e.external) g.external = e.external;
      const input = (id,note) => { if (Number.isFinite(nodes[id]?.end)) inputs.push({id,note}); };
      if (name === 'select' && current(n)) input(key(r,2,'recv_bc'),'本例请求已通过 bootstrap 准入');
      if (name === 'launch' && r && current(n)) input(key(r,n,'recv_proxy'),'当前 batch 的上级激活已接收');
      if (name === 'release' && n >= 2) {
        input(key(r,n,'recv_rc'),'本轮 release 名单');
        for (const row of releases) if (row.r === r && row.n === n) for (const m of row.batches) input(key(r,m+4,'kv'),`M${m} 本地 KV 传输终态`);
      }
      if (name === 'send_bc') input(key(r,r === 2 ? n : n-1,r === 2 ? 'boot_poll' : 'recv_bc'),'候选集合 / 上轮回流共识');
      if (name === 'send_rc') input(key(r,r === 2 ? n : n-1,r === 2 ? 'term_poll' : 'recv_rc'),'终态集合 / 上轮 release 回流');
      if (name === 'term_poll') for (const m of terminal[`${r}:${n}`] || []) input(key(r,m+4,'kv'),`本例候选 M${m} 的本地终态`);
      ensure(inputs.every(d => nodes[d.id].end <= e.start + 1e-7), '状态输入顺序校验失败。');
      if (inputs.length) g.inputs = inputs;
      graph[id] = g;
    }
    if (horizon > 22) {
      const needed = new Set(Object.keys(graph).filter(k => graph[k].n < displayN)), todo = [...needed];
      for (let i = 0; i < todo.length; i++) for (const d of [...graph[todo[i]].deps, ...(graph[todo[i]].inputs || []).map(v => v.id)]) {
        if (!needed.has(d)) { needed.add(d); todo.push(d); }
      }
      for (const id of Object.keys(graph)) if (!needed.has(id)) delete graph[id];
    }
    for (const e of [...events,...loops]) { e.start = round(e.start); e.end = round(e.end); }
    return {baseline:template.baseline,sourceCommit:template.sourceCommit,units:template.units,pp:3,depth:0,graph,
      assumptions:template.assumptions,loops,events,releases,end:Math.max(...loops.map(v => v.end)),
      checks:{resolved_nodes:resolved.length,batches:5,forward_blocks:15,released_all:true}};
  }

  function ownerBatches(owner) {
    const ids = new Set([...owner.matchAll(/M(\d+)/g)].map(m => Number(m[1])));
    for (const [,a,b] of owner.matchAll(/M(\d+)[–-]M(\d+)/g)) for (let m = +a; m <= +b; m++) ids.add(m);
    return [...ids].filter(m => m >= 1 && m <= 5).sort((a,b) => a-b);
  }

  function deriveQuick(data) {
    const {graph} = data, phaseGroups = ['A','BC','DE','FGH','I'];
    const loops = data.loops.map(original => {
      const {r,n} = original, released = data.releases.filter(v => v.r === r && v.n === n).flatMap(v => v.batches);
      const events = data.events.filter(e => e.r === r && e.n === n && ['cpu','wait'].includes(e.kind)).sort((a,b) => a.start-b.start || a.end-b.end);
      const groups = phaseGroups.map(phases => {
        const subset = events.filter(e => phases.includes(e.phase)), links = {};
        if (!subset.length) return null;
        for (const e of subset) {
          const owners = e.id === key(r,n,'release') ? released : ownerBatches(e.owner);
          for (const m of owners) {
            const link = links[m] ||= {shared:true,actions:[]}; link.shared &&= owners.length > 1;
            const action = e.label.replace(/^等：/,''); if (!link.actions.includes(action)) link.actions.push(action);
          }
        }
        return {start:subset[0].start,end:subset.at(-1).end,refs:[...new Set(subset.map(e => e.ref))].sort(),
          wait:round(subset.filter(e => e.kind === 'wait').reduce((sum,e) => sum+e.end-e.start,0)),links};
      });
      return {...original,groups,released};
    });
    const batches = [1,2,3,4,5].map(batch => ({batch,ranks:[0,1,2].map(r => {
      const active = loops.find(l => l.r === r && l.current === batch), result = loops.find(l => l.r === r && l.old === batch);
      const release = data.releases.find(l => l.r === r && l.batches.includes(batch));
      const gpu = graph[key(r,active.n,'gpu')], kv = graph[key(r,result.n,'kv')], cleanup = graph[key(r,release.n,'release')];
      const points = [graph[key(r,0,'recv_req')].start,graph[key(r,active.n,'select')].start,gpu.start,gpu.end,kv.start,kv.end,cleanup.end];
      ensure(points.every((p,i) => !i || p >= points[i-1]), '生命周期顺序校验失败。');
      return {r,phases:points.slice(0,-1).map((p,i) => ({start:p,end:points[i+1],loop:[2,active.n,active.n,result.n,result.n,release.n][i]})),
        admitted:graph[key(r,2,'recv_bc')].end,releaseStart:cleanup.start,currentLoop:active.n,resultLoop:result.n,releaseLoop:release.n};
    })}));
    return {sourceCommit:data.sourceCommit,end:data.end,loops,batches};
  }

  function buildPacket(template, timing, fixedHorizon) {
    validate(timing);
    let model;
    for (const n of fixedHorizon ? [fixedHorizon] : [22,64,128,256,LIMIT]) {
      ensure(n <= LIMIT, '自定义计算最多展开 512 轮。'); model = compute(template,timing,n); if (model) break;
    }
    ensure(model, '这组时间需要超过 512 轮才能释放全部请求。请缩短 KV 时间或提高 CPU 阶段倍率；图仍保留上次结果。');
    const summary = {end:round(model.end),gpu:timing.gpu.map(row => row.map(round)),
      gpuEnd:Math.max(...model.events.filter(e => e.kind === 'gpu').map(e => e.end)),
      releaseEnd:Math.max(...model.releases.map(v => model.graph[key(v.r,v.n,'release')].end)),loops:model.loops.length/3};
    return {model,quick:deriveQuick(model),summary};
  }

  const api = {validate,buildPacket};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else if (typeof document === 'undefined') scope.onmessage = event => {
    const {id,template,timing} = event.data;
    try { scope.postMessage({id,packet:buildPacket(template,timing)}); }
    catch (error) { scope.postMessage({id,error:error.message}); }
  };
  else scope.PP_TIMING_ENGINE = api;
})(typeof self !== 'undefined' ? self : globalThis);
