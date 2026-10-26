/* Deployment before/after: one shared service group vs independent P/D groups.
 * The existing f-deployment animation still owns the KV-copy and readiness
 * sequence. This panel only compares where the same A/B work is placed, so its
 * objects and arrows stay static while the player seeks below. */
(function(root){
'use strict';
const el=(tag,cls,text)=>{const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=text;return node;};

const T=(req,op,n)=>({req,op,n});
const requests={
 A:{id:'A',name:'A',stage:'Decode',detail:'已生成若干 token，正在续写',color:'a'},
 B:{id:'B',name:'B',stage:'Prefill',detail:'带长输入到达，先 Prefill 后 Decode',color:'b'}
};
function taskLabel(t){
 if(t.op==='D')return t.req==='B'?'B 的首次 Decode':'A 的第 '+t.n+' 次 Decode';
 if(t.op==='P')return 'B 的第 '+t.n+' 段 Prefill';
 if(t.op==='S')return 'P 发送 B 的 KV 副本、y1 与配对信息';
 return '';
}
function shortCode(t){
 if(t.op==='S')return '发送';
 return (t.op==='P'?'P':'D')+t.n;
}
function taskKey(t){return t.req+':'+t.op+':'+t.n;}

const baselineSlots=[
 {n:1,tasks:[T('A','D',1)],wait:{req:'B',why:'B 刚到，等这一轮结束'}},
 {n:2,tasks:[T('B','P',1)],wait:{req:'A',why:'A 的下一轮 Decode 等 B 的整段'}},
 {n:3,tasks:[T('A','D',2),T('B','P',2)],note:'同批 mixed/chunk'},
 {n:4,tasks:[T('B','P',3)],wait:{req:'A',why:'这一段 B 又单独占轮'}},
 {n:5,tasks:[T('A','D',3),T('B','P',4)],note:'同批 mixed/chunk'},
 {n:6,tasks:[T('A','D',4),T('B','D',1)],note:'B 的 KV 已在同组'}
];
const pSlots=[
 {n:1,tasks:[T('B','P',1)]},
 {n:2,tasks:[T('B','P',2)]},
 {n:3,tasks:[T('B','P',3)]},
 {n:4,tasks:[T('B','P',4)]},
 {n:5,tasks:[T('B','S',1)],note:'交给 D 的接收区'},
 {n:6,tasks:[],idle:'空闲 · 可接新 Prefill'}
];
const dSlots=[
 {n:1,tasks:[T('A','D',1)]},
 {n:2,tasks:[T('A','D',2)]},
 {n:3,tasks:[T('A','D',3)]},
 {n:4,tasks:[T('A','D',4)]},
 {n:5,tasks:[],idle:'接收区等 KV',gate:'wait',gateText:'准入未开',note:'A 的 4 次 Decode 已完成；接收区等 B 的 KV 副本、y1 与配对信息'},
 {n:6,tasks:[T('B','D',1)],gate:'open',gateText:'就绪'}
];
const transfer={
 from:4,to:5,
 title:'t4 完成 → t5 传输',
	payload:'历史 KV 副本 + y1 + 配对信息',
	wait:'D 接收区先等载荷齐备，再通过准入'
};

function countSlots(slots){
 const out={A:{decode:0},B:{prefill:0,decode:0},sends:0,idle:0,waits:{A:0,B:0},gates:{wait:0,open:0}};
 for(const slot of slots){
  for(const t of slot.tasks||[]){
   if(t.req==='A'&&t.op==='D')out.A.decode++;
   if(t.req==='B'&&t.op==='P')out.B.prefill++;
   if(t.req==='B'&&t.op==='D')out.B.decode++;
   if(t.op==='S')out.sends++;
  }
  if(slot.idle)out.idle++;
  if(slot.wait)out.waits[slot.wait.req]++;
  if(slot.gate)out.gates[slot.gate]++;
 }
 return out;
}

const model={
 key:'deployment-contrast',
 title:'同一个 A、B，两种服务组安排',
 premise:'A 已生成若干 token，正在 Decode；B 带长输入到达，先做 Prefill，再做首次 Decode。两侧使用同一组请求、同一组阶段依赖和同一组工作量：A 完成 4 次 Decode；B 完成 4 段 Prefill 与首次 Decode。',
 boundary:{
  baseline:'1 组共享执行容量；Prefill 与 Decode 在同一组排队，可同批 mixed/chunk。',
  contrast:'2 组独立执行容量；P、D 各加载完整模型，按生成阶段分工。',
  not:'不是把同一组资源变成两倍速，也不是 P 跑前几层、D 跑后几层。'
 },
 baseline:{
  id:'shared',title:'基线 · 一个共享服务组',badge:'1 组执行容量',groupCount:1,capacity:1,fullModel:true,
  group:{name:'共享服务组',role:'完整模型 · 同组完成 Prefill / Decode',queue:'共享队列'},
  arrivals:[{req:'A',text:'A 正在续写'},{req:'B',text:'B 长输入到达'}],
  queue:'A 的下一轮 Decode 与 B 的下一段 Prefill 都等这一组容量',
  slots:baselineSlots,
  schedule:'左图选用一种教学调度：B 的 4 段中，第 1、3 段单独占轮，第 2、4 段与 A 的 Decode 同批。换一种 chunk 或优先级，等待位置会变；不表示合并只能串行。',
  caption:'同一组容量里，任务只能排进不同轮次或同一批 mixed。A 在 t2、t4 等 B 单独占的轮次，B 在 t1 等 A 先结束；B 的 KV 留在本组，不需要跨组交接。'
 },
 contrast:{
  id:'separated',title:'对照 · 独立 P / D 服务组',badge:'2 组执行容量',groupCount:2,capacity:2,fullModel:true,
  p:{id:'p',name:'P 服务组',role:'完整模型 · 接 B 的 Prefill',capacity:1,queue:'P 队列',slots:pSlots,
     caption:'B 的 4 段 Prefill 在 P 组连续推进；t4 完成后由 P 发出 KV、y1 与配对信息。'},
  transfer,
  d:{id:'d',name:'D 服务组',role:'完整模型 · A 的 Decode + B 的接收/续写',capacity:1,queue:'D 队列',slots:dSlots,
     caption:'A 在 D 组完成 4 次 Decode；t5 接收区等 B 的 KV、y1 与配对信息，t6 准入通过后才执行 B 的首次 Decode。'},
  caption:'P、D 各自有执行容量，所以 t1–t4 两条时间线并行：P 执行 B 的 4 段 Prefill，D 执行 A 的 4 次 Decode。B 跨组要交 KV 与首 token；D 的接收区在 t5 等待，t6 就绪后才执行 B 的首次 Decode，A 不会在 t5/t6 继续多做 Decode。'
 },
 counts:{baseline:countSlots(baselineSlots),contrast:countSlots(pSlots),d:countSlots(dSlots)}
};

function ariaSummary(){
 return '部署前后对照：同一个正在 Decode 的 A 与同一个长输入 B，两侧完成同一组工作：A 4 次 Decode，B 4 段 Prefill 加首次 Decode。基线是一组共享执行容量，A 与 B 的 Prefill、Decode 在同一组排队，所选教学调度中 A 等待 2 轮、B 等待 1 轮，部分轮次同批 mixed/chunk；B 的 KV 留在同组。对照是两组独立执行容量，P、D 各加载完整模型：P 连续执行 B 的 4 段 Prefill，D 在 t1–t4 执行 A 的 4 次 Decode；B 的 KV 副本、y1 与配对信息在 t5 交给 D，D 的接收区等待，t6 就绪后才开始 B 的首次 Decode。右图多一组资源，不是同资源倍速，也不是 P 前层、D 后层的模型切分。';
}

function chip(t){
 const cls='dc-chip dc-req-'+t.req.toLowerCase()+(t.op==='S'?' dc-send':'');
 const node=el('span',cls);
 node.dataset.req=t.req;node.dataset.op=t.op;node.dataset.index=String(t.n);
 node.setAttribute('aria-label',taskLabel(t));
 node.setAttribute('title',taskLabel(t));
 node.append(el('strong','',t.req),el('span','',shortCode(t)));
 return node;
}
function slot(slot,side){
 const li=el('li','dc-slot');
 li.dataset.slot=String(slot.n);
 li.dataset.side=side;
 li.append(el('span','dc-slot-n','t'+slot.n));
 const chips=el('span','dc-chips');
 if(slot.tasks?.length)slot.tasks.forEach(t=>chips.append(chip(t)));
 if(slot.idle)chips.append(el('span','dc-idle',slot.idle));
 li.append(chips);
 if(slot.wait){
  const wait=el('span','dc-wait',slot.wait.req+' 等');
  wait.dataset.wait=slot.wait.req;wait.setAttribute('title',slot.wait.why||'');
  li.append(wait);
 }
 if(slot.gate){
  const gate=el('span','dc-gate '+(slot.gate==='open'?'is-open':'is-wait'),slot.gateText);
  gate.dataset.gate=slot.gate;
  li.append(gate);
 }
 if(slot.note)li.title=slot.note;
 return li;
}
function track(label,slots,side){
 const wrap=el('div','dc-track-wrap');
 const ol=el('ol','dc-track');ol.setAttribute('aria-label',label);
 slots.forEach(s=>ol.append(slot(s,side)));
 wrap.append(ol);
 return wrap;
}
function laneHead(name,role,capacity){
 const head=el('div','dc-lane-head');
 const title=el('div','dc-lane-title');
 title.append(el('strong','',name),el('span','',role));
 head.append(title,el('span','dc-cap',capacity));
 return head;
}
function arrivals(items,note){
 const row=el('div','dc-arrivals');
 items.forEach(item=>{
  const tag=el('span','dc-arrival dc-arrival-'+item.req.toLowerCase());
  tag.append(el('strong','',item.req),el('span','',item.text));
  row.append(tag);
 });
 row.append(el('span','dc-inflow','→'),el('span','dc-queue-note',note));
 return row;
}
function sideHead(title,badge){
 const head=el('header','dc-side-head');
 head.append(el('h3','',title),el('span','dc-side-badge',badge));
 return head;
}

function buildPanel(){
 const section=el('section','deployment-contrast');
 section.dataset.adoption='deployment';
 section.dataset.baselineCapacity='1';
 section.dataset.contrastCapacity='2';
 section.dataset.baselineGroups='1';
 section.dataset.contrastGroups='2';
 section.setAttribute('aria-label',ariaSummary());

 const head=el('header','dc-head');
 const heading=el('div','dc-heading');
 heading.append(
  el('strong','',model.title),
  el('span','','同一个 A、B：左边一组共享执行容量，A/B 同组排队；右边 P、D 两组独立容量，各自都能执行完整模型。')
 );
 head.append(heading,el('span','dc-badge','1 组 vs 2 组'));
 section.append(head);

 const premise=el('p','dc-premise');
 premise.append(el('strong','','共同任务：'),el('span','',model.premise));
 section.append(premise);

 const grid=el('div','dc-grid');

 /* Baseline: one group, one queue, one execution track. */
 const shared=el('article','dc-side dc-shared');
 shared.dataset.side='shared';shared.dataset.capacity='1';shared.dataset.fullModel='true';
 shared.append(sideHead(model.baseline.title,model.baseline.badge));
 const sharedGroup=el('div','dc-group dc-group-shared');
 sharedGroup.append(laneHead(model.baseline.group.name,model.baseline.group.role,'容量 1 组'));
 sharedGroup.append(arrivals(model.baseline.arrivals,model.baseline.queue));
 sharedGroup.append(track('共享执行容量：t1 到 t6 的任务安排',model.baseline.slots,'shared'));
 const schedule=el('p','dc-schedule');
 schedule.append(el('strong','','调度假设：'),el('span','',model.baseline.schedule));
 sharedGroup.append(schedule);
 shared.append(sharedGroup);
 const sharedCaption=el('p','dc-caption');sharedCaption.textContent=model.baseline.caption;shared.append(sharedCaption);
 const sharedBound=el('p','dc-bound');sharedBound.append(el('strong','','基线：'),el('span','',model.boundary.baseline));shared.append(sharedBound);

 /* Contrast: two independent groups, one transfer band, one readiness gate. */
 const separated=el('article','dc-side dc-separated');
 separated.dataset.side='separated';separated.dataset.capacity='2';separated.dataset.groups='2';separated.dataset.fullModel='true';
 separated.append(sideHead(model.contrast.title,model.contrast.badge));

 const p=el('div','dc-group dc-group-p');p.dataset.group='p';
 p.append(laneHead(model.contrast.p.name,model.contrast.p.role,'容量 1 组'));
 p.append(arrivals([{req:'B',text:'B 长输入 → P 队列'}],'B 的 Prefill 在本组推进'));
 p.append(track('P 服务组：B 的 Prefill 与发送',model.contrast.p.slots,'p'));
 const pCap=el('p','dc-lane-note',model.contrast.p.caption);p.append(pCap);
 separated.append(p);

 const band=el('div','dc-transfer');
 band.dataset.transfer='kv+y1+meta';band.dataset.fromSlot='4';band.dataset.toSlot='5';
 const bandCopy=el('div','dc-transfer-copy');
 bandCopy.append(el('strong','',model.contrast.transfer.title),el('span','',model.contrast.transfer.payload));
 const bandWait=el('span','dc-transfer-wait',model.contrast.transfer.wait);
 band.append(bandCopy,el('span','dc-transfer-arrow','↓'),bandWait);
 separated.append(band);

 const d=el('div','dc-group dc-group-d');d.dataset.group='d';
 d.append(laneHead(model.contrast.d.name,model.contrast.d.role,'容量 1 组'));
 d.append(arrivals([{req:'A',text:'A 续写 → D 队列'},{req:'B',text:'B 的交接项 → 接收区'}],'接收区与 A 的执行分开等'));
 d.append(track('D 服务组：A 的 Decode 与 B 的接收/续写',model.contrast.d.slots,'d'));
 const dCap=el('p','dc-lane-note',model.contrast.d.caption);d.append(dCap);
 separated.append(d);

 const separatedCaption=el('p','dc-caption');separatedCaption.textContent=model.contrast.caption;separated.append(separatedCaption);
 const separatedBound=el('p','dc-bound');separatedBound.append(el('strong','','对照：'),el('span','',model.boundary.contrast));separated.append(separatedBound);

 grid.append(shared,separated);
 section.append(grid);

 const note=el('p','dc-note');
 note.append(
  el('strong','','读图边界：'),
  el('span','','右图确实多了一组可执行容量，不是把同一组资源加速两倍；P、D 按阶段分工，不是 P 跑前几层、D 跑后几层。时隙是教学顺序，不是实测耗时。'),
  el('span','','角色与成本的独立说明见 '),
  (()=>{const a=el('a','','角色与成本说明');a.href='#deployment-comparison';return a;})(),
  el('span','','；交接过程见 '),
  (()=>{const a=el('a','','下方主动画');a.href='#lesson-animation';return a;})(),
  el('span','','。')
 );
 section.append(note);
 return section;
}

function mount(kind,viewport){
 if(kind!=='contrast')return null;
 const section=buildPanel();
 const film=viewport.closest('.mechanism-film');
 film?.querySelectorAll(':scope > .deployment-contrast').forEach(node=>node.remove());
 const anchor=viewport.closest('.motion-theater')||viewport;
 anchor.before(section);
 return {element:section,destroy(){section.remove();}};
}

root.DeploymentTeaching=model;
root.DeploymentAdoption={mount,kinds:['contrast'],model};
if(typeof module==='object'&&module.exports)module.exports=model;
})(typeof globalThis==='object'?globalThis:this);
