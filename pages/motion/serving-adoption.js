/* Serving placement comparison: one shared execution slot vs two complete
 * instances. It is a DOM diagram, not a second route animation: the player's
 * serving-route visual still owns R1's queue, execution and return states. */
(function(root){
'use strict';
const el=(tag,cls,text)=>{const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=text;return node;};

function requestChip(req,note,state='accepted',compact=false){
 const chip=el('span','sa-request sa-'+(req==='R2'?'r2':'r1')+' is-'+state+(compact?' is-compact':''));
 chip.dataset.request=req;chip.dataset.state=state;
 chip.append(el('b','',req));
 if(note)chip.append(el('small','',note));
 if(state==='executing')chip.append(el('span','sa-chip-state','执行中'));
 else if(state==='waiting')chip.append(el('span','sa-chip-state','等待'));
 return chip;
}

function modelStack(){
 const stack=el('div','sa-model-stack');
 stack.setAttribute('aria-hidden','true');
 for(let i=0;i<4;i++)stack.append(el('i',''));
 return stack;
}

function modelBlock(copies,compact=false){
 const block=el('div','sa-model-block'+(compact?' is-compact':''));
 const head=el('div','sa-model-head');
 head.append(el('span','','完整模型'),el('b','','×'+copies));
 block.append(head,modelStack(),el('p','sa-model-foot','参数副本 '+copies+' 份'));
 return block;
}

function metricStrip(counts){
 const strip=el('div','sa-metrics');
 for(const [value,label] of [[counts.accepted,'接纳'],[counts.executing,'执行'],[counts.waiting,'等待']]){
  const item=el('span','sa-metric');
  item.append(el('b','',String(value)),el('small','',label));
  strip.append(item);
 }
 return strip;
}

function cardHead(data,current){
 const head=el('header','sa-side-head');
 const copy=el('div','sa-side-copy');
 copy.append(el('h3','',data.title),el('p','',data.subtitle));
 const tag=el('span',current?'sa-current-tag':'sa-compare-tag',current?'当前场景':'对照');
 head.append(copy,tag);
 return head;
}

function sharedSide(data,current){
 const side=el('article','sa-side sa-shared'+(current?' is-current':''));
 side.dataset.side='shared';
 side.dataset.current=String(current);
 side.dataset.instances=String(data.instances);
 side.dataset.modelCopies=String(data.modelCopies);
 side.dataset.executing=String(data.counts.executing);
 side.dataset.waiting=String(data.counts.waiting);
 side.setAttribute('aria-label','一个实例共享容量：R1 与 R2 都被接纳，同一时刻只有一个执行槽。R1 执行，R2 等待。完整模型副本一份。');
 side.append(cardHead(data,current));

 const ingress=el('div','sa-ingress');
 const copy=el('div','sa-ingress-copy');
 copy.append(el('span','sa-kicker','请求到达'));
 const reqs=el('div','sa-requests');
 reqs.append(requestChip('R1','先到','accepted'),requestChip('R2','随后','accepted'));
 copy.append(reqs);
 const gate=el('div','sa-gate');
 gate.append(el('strong','','health ✓'),el('span','','只证明在线，不等于空闲'));
 ingress.append(copy,gate);
 side.append(ingress);

 const body=el('div','sa-shared-body');
 const queue=el('section','sa-queue');
 const qhead=el('div','sa-queue-head');
 qhead.append(el('strong','','共享队列'),el('span','','2 条已接纳 · 1 个执行槽'));
 queue.append(qhead);
 const slots=el('div','sa-slots');
 const running=el('div','sa-slot is-running');
 running.dataset.request='R1';running.dataset.state='executing';
 const runCopy=el('div','sa-slot-copy');
 runCopy.append(el('span','sa-slot-kicker','执行槽 1 / 1'),requestChip('R1','占本轮槽位','executing',true));
 running.append(runCopy,el('span','sa-track',''));
 const waiting=el('div','sa-slot is-waiting');
 waiting.dataset.request='R2';waiting.dataset.state='waiting';
 const waitCopy=el('div','sa-slot-copy');
 waitCopy.append(el('span','sa-slot-kicker','共享队列'),requestChip('R2','等 R1 让出槽位','waiting',true));
 waiting.append(waitCopy,el('span','sa-wait-line',''));
 slots.append(running,waiting);
 queue.append(slots,metricStrip(data.counts));
 body.append(queue,modelBlock(data.modelCopies));
 side.append(body);

 const caption=el('p','sa-caption');
 caption.append(el('strong','','同一时刻：'),el('span','','R1 占唯一执行槽，R2 只能等待；R1 完成后，R2 才能进入同一槽位。'));
 side.append(caption);
 const cost=el('p','sa-cost');
 cost.append(el('strong','','成本：'),el('span','','1 份完整模型；代价是两条请求争同一份执行容量。'));
 side.append(cost);
 return side;
}

function replicaPanel(replica){
 const panel=el('section','sa-replica');
 panel.dataset.instance=replica.id;
 const head=el('header','sa-replica-head');
 const name=el('div','sa-replica-name');
 name.append(el('b','','实例 '+replica.id));
 name.append(el('span','sa-health','health ✓'));
 const assign=el('span','sa-assign','接 '+replica.executionSlot);
 head.append(name,assign);
 panel.append(head);
 const slot=el('div','sa-slot is-running');
 slot.dataset.request=replica.executionSlot;slot.dataset.state='executing';
 const copy=el('div','sa-slot-copy');
 copy.append(el('span','sa-slot-kicker','本实例队列 · 执行槽 1 / 1'),requestChip(replica.executionSlot,'已接纳','executing',true));
 slot.append(copy,el('span','sa-track',''));
 panel.append(slot);
 const foot=el('div','sa-replica-foot');
 foot.append(el('span','','等待 0'),el('span','','队列 1'));
 panel.append(foot,modelBlock(replica.modelCopies,true));
 return panel;
}

function splitSide(data,current){
 const side=el('article','sa-side sa-split'+(current?' is-current':''));
 side.dataset.side='split';
 side.dataset.current=String(current);
 side.dataset.instances=String(data.instances);
 side.dataset.modelCopies=String(data.modelCopies);
 side.dataset.executing=String(data.counts.executing);
 side.dataset.waiting=String(data.counts.waiting);
 side.setAttribute('aria-label','两个实例各接一条：R1 进入实例 B，R2 进入实例 A；两条请求各自获得执行槽并同时执行。完整模型副本两份。');
 side.append(cardHead(data,current));

 const ingress=el('div','sa-ingress');
 const copy=el('div','sa-ingress-copy');
 copy.append(el('span','sa-kicker','请求到达'));
 const reqs=el('div','sa-requests');
 reqs.append(requestChip('R1','→ 实例 B','accepted'),requestChip('R2','→ 实例 A','accepted'));
 copy.append(reqs);
 const gate=el('div','sa-gate');
 gate.append(el('strong','','服务层选择'),el('span','','接纳后进入目标实例队列'));
 ingress.append(copy,gate);
 side.append(ingress);

 const replicas=el('div','sa-replicas');
 data.replicas.forEach(replica=>replicas.append(replicaPanel(replica)));
 side.append(replicas,metricStrip(data.counts));

 const caption=el('p','sa-caption');
 caption.append(el('strong','','同一时刻：'),el('span','','R1 在 B、R2 在 A，各自占一个执行槽，可以同时推进。'));
 side.append(caption);
 const cost=el('p','sa-cost');
 cost.append(el('strong','','成本：'),el('span','','2 份完整模型；多一份权重驻留、健康检查与路由协调。'));
 side.append(cost);
 return side;
}

function costStrip(comparison){
 const strip=el('div','sa-cost-strip');
 strip.append(el('span','sa-cost-label','完整模型副本'));
 const one=el('span','sa-copy is-one','×1');
 const arrow=el('span','sa-cost-arrow','→');
 const two=el('span','sa-copy is-two','×2');
 strip.append(one,arrow,two,el('span','sa-cost-note','1→2 指权重副本；KV、激活、通信另计'));
 return strip;
}

function buildPanel(comparison,variant){
 const split=variant===1;
 const section=el('section','serving-adoption');
 section.dataset.adoption='serving';
 section.dataset.variant=String(variant);
 section.dataset.currentSide=split?'split':'shared';
 section.setAttribute('aria-label','服务多实例对照：同一个 R1 与 R2，左边一个实例共享队列，右边两个实例各接一条；当前高亮'+(split?'两个实例':'一个实例')+'。');

 const head=el('header','sa-head');
 const heading=el('div','sa-heading');
 heading.append(
  el('strong','','同一个 R1、R2：一个实例排队，还是两个实例各接一条？'),
  el('span','',comparison.moment)
 );
 const badge=el('span','sa-badge',split?'当前：2 个实例':'当前：1 个实例');
 head.append(heading,badge);
 section.append(head);

 const premise=el('div','sa-premise');
 premise.append(el('span','sa-premise-label','共同条件'));
 premise.append(requestChip('R1','先到','accepted',true),requestChip('R2','随后到达','accepted',true));
 premise.append(el('span','sa-premise-copy','同一完整模型；每条请求先被接纳，再等执行槽。'));
 section.append(premise);

 const grid=el('div','sa-grid');
 grid.append(sharedSide(comparison.shared,!split),splitSide(comparison.split,split));
 section.append(grid,costStrip(comparison));

 const notes=el('div','sa-notes');
 const assumptions=[
  ['容量假设',comparison.capacity],
  ['健康 ≠ 空闲 ≠ 接纳',comparison.health],
  ['成本边界',comparison.cost]
 ];
 for(const [label,text] of assumptions){
  const note=el('p','sa-note');
  note.append(el('strong','',label+'：'),el('span','',text));
  notes.append(note);
 }
 section.append(notes);
 return section;
}

function comparisonFrom(data){
 return data?.comparison||root.FoundationModel?.servingComparison?.()||null;
}

function clear(viewport){
 const film=viewport?.closest?.('.mechanism-film');
 film?.querySelectorAll(':scope > .serving-adoption').forEach(node=>node.remove());
}

function mount(kind,viewport,data){
 if(kind!=='contrast')return null;
 const comparison=comparisonFrom(data);
 if(!comparison)return null;
 const variant=Number(data?.variant)===1?1:0;
 const section=buildPanel(comparison,variant);
 clear(viewport);
 const anchor=viewport.closest('.motion-theater')||viewport;
 anchor.before(section);
 return {element:section,destroy(){section.remove();}};
}

root.ServingAdoption={mount,clear};
})(typeof globalThis==='object'?globalThis:this);
