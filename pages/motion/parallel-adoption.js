/* Adoption comparison: same input and work, two side-by-side layouts.
 * This module is intentionally independent of the main GSAP canvas. It explains
 * what changes when a second GPU/replica is adopted, without drawing over the
 * existing mechanism animation. Extend `builders` for the later parallel
 * mechanisms (EP/SP/PCP/DCP); the shell, responsive layout and cleanup stay
 * shared. */
(function(root){
'use strict';
const el=(tag,cls,text)=>{const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=text;return node;};
const matrix=(label,rows,role,options={})=>{
 const unit=el('div','pa-matrix');if(options.compact)unit.classList.add('is-compact');
 if(label)unit.append(el('span','pa-matrix-label',label));
 const grid=el('div','pa-cells');grid.style.setProperty('--cols',String(rows[0].length));
 rows.forEach(row=>row.forEach(value=>{
  const cell=el('span','pa-cell pa-'+role,String(value));
  if(value===0)cell.classList.add('is-zero');
  cell.dataset.value=String(value);
  grid.append(cell);
 }));
 unit.append(grid);return unit;
};
const expression=parts=>{const row=el('div','pa-expression');for(const part of parts){row.append(typeof part==='string'?el('span','pa-op',part):part);}return row;};
const badge=(text,cls='')=>el('span','pa-stat '+cls,text);
const stats=(items,cls='')=>{const list=el('ul','pa-stats '+cls);items.forEach(item=>list.append(el('li','pa-stat',item)));return list;};
const side=(id,title,badgeText,children,attrs={})=>{
 const article=el('article','pa-side');article.dataset.side=id;
 for(const [key,value] of Object.entries(attrs))article.dataset[key]=String(value);
 const head=el('header','pa-side-head');head.append(el('h3','',title),el('span','pa-side-badge',badgeText));
 article.append(head,...children);
 return article;
};
const gpu=(rank,extra='')=>{
 const card=el('div','pa-gpu');card.dataset.rank=String(rank);
 const head=el('div','pa-gpu-head');head.append(el('strong','', 'GPU '+rank),el('span','',extra));
 card.append(head);return card;
};
const rail=(products,adds,label)=>{
 const wrap=el('div','pa-products');
 const labelNode=el('span','pa-products-label',label);
 const marks=el('span','pa-marks');marks.setAttribute('aria-hidden','true');
 for(let i=0;i<products;i++)marks.append(el('i','pa-product','×'));
 for(let i=0;i<adds;i++)marks.append(el('i','pa-plus','+'));
 wrap.append(marks,labelNode);
 wrap.setAttribute('aria-label',label);
 return wrap;
};

function buildTp(scenario){
 const model=root.ParallelTeachingModel;
 const kind=scenario?.kind==='parallel-tp'&&scenario.data?.kind==='column'?'column':'row';
 const data=model&&typeof model.tensor==='function'?model.tensor(kind):null;
 if(!data)return null;
 const row=kind==='row';
 const label=row?'切输入维':'切输出维';
 const shape=matrix=>matrix.length+'×'+matrix[0].length;
 const mergeText=row?'逐元素相加（'+data.mergeAdds+' 次）':'按输出列并排（0 次跨卡加法）';
 const baseline=[
  expression([
   matrix('X · 2×2',data.x,'x'),
   '×',
   matrix('W · 2×2',data.w,'w'),
   '=',
   matrix('Y · 2×2',data.result,'y')
  ]),
  stats(['4 个权重格','8 个乘积项','4 次加法（每格 2 项）']),
  rail(8,4,'8 个乘积项（8 乘、4 加）'),
  el('p','pa-caption','Y 的每个格子由 2 个乘积项相加；整层仍未拆分。')
 ];
 const ranks=el('div','pa-ranks');
 data.shards.forEach((shard,rank)=>{
  const ops=data.ops[rank],first=data.firstCell[rank];
  const card=gpu(rank,row?'持有权重行':'持有权重列');
  card.dataset.weightCells=String(shard.w.flat().length);
  card.dataset.productCount=String(ops.mul);
  card.dataset.localAdds=String(ops.add);
  card.dataset.innerDim=String(ops.k);
  const equation=first.terms.map(term=>`${term.left}×${term.right}`).join(' + ');
  card.append(
   expression([
    matrix((row?'X 列 '+(rank+1):'X 完整')+' · '+shape(shard.x),shard.x,'x',{compact:true}),
    matrix((row?'W 行 '+(rank+1):'W 列 '+(rank+1))+' · '+shape(shard.w),shard.w,'w',{compact:true})
   ]),
   matrix((row?'局部贡献 C'+rank:'输出列 C'+rank)+' · '+shape(data.local[rank]),data.local[rank],'c',{compact:true}),
   el('p','pa-local-equation',`C${rank}[0,0] = ${equation} = ${first.value}`),
   rail(ops.mul,ops.add,`${ops.mul} 个乘积项（${ops.mul} 乘、${ops.add} 次局部加法）`)
  );
  ranks.append(card);
 });
 const mergeParts=[];
 data.local.forEach((local,i)=>{
  if(i){
   if(row)mergeParts.push('+');
   else{
    const concat=el('span','pa-concat-op','∥');
    concat.title='按输出列并排；不把两卡的值相加';
    concat.setAttribute('aria-label','按列拼接');
    mergeParts.push(concat);
   }
  }
  mergeParts.push(matrix('C'+i,local,'c',{compact:true}));
 });
 mergeParts.push('=',matrix('Y · 2×2',data.result,'y',{compact:true}));
 const merge=el('div','pa-merge');
 merge.dataset.merge=row?'sum':'concat';
 merge.append(el('span','pa-merge-label','合并：'+mergeText),expression(mergeParts));
 const perCardAdds=data.ops[0].add;
 const adoptedStats=['2 张 GPU','每卡 2 个权重格','每卡 '+data.ops[0].mul+' 个乘积项',row?'每卡 0 次局部加法（每格 1 个乘积）':'每卡 '+perCardAdds+' 次局部加法（每格 '+data.ops[0].k+' 个乘积）',row?'合并再加 '+data.mergeAdds+' 次':'合并只拼接（0 次跨卡加法）',row?'X 按列切分':'X 完整复制到两卡'];
 const adopted=[
  ranks,
  stats(adoptedStats),
  merge,
  el('p','pa-caption',row?'每张卡只算自己那一列输入对应的 4 个乘积；每个输出格只有 1 个乘积，本地没有数学加法。两卡把同坐标的局部结果相加 4 次得到 Y；图中未计通信时间。':'每张卡沿完整输入维做内积：每个输出格 2 乘 1 加，共 4 乘 2 加；合并只把两条输出列并排，不做跨卡加法。图中未计通信时间。')
 ];
 return {
  attrs:{tpKind:kind,localAddsPerCard:perCardAdds,mergeAdds:data.mergeAdds},
  aria:'TP 采用前后对照：同一 X、W 和 Y；单卡 8 个乘积项、4 次加法；两卡各 4 个乘积项、不重复计算同一乘积。切输入维时每卡 0 次局部加法（每格 1 个乘积），合并再做 4 次相加；切输出维时每卡 2 次局部加法（每格 2 乘 1 加），合并只拼接、没有跨卡加法。系统两侧都仍是 8 乘、4 加；数字是乘积项与加法计数，不是 FLOPs 或实测加速。',
  title:'TP · 同一 Y=XW，单卡与两卡',
  subtitle:row?'左：完整单卡计算；右：按输入维切成两片，每卡 4 乘、0 次本地加，最后 4 次同坐标相加。':'左：完整单卡计算；右：按输出维切成两片，每卡 4 乘、2 次本地加，最后按列拼接。',
  extra:'两卡（额外 1 张 GPU）',
  sides:[
   side('baseline','未采用 · 单卡','1 GPU',baseline,{gpus:1,weightCells:4,productCount:8,adds:4,merge:'none'}),
   side('adopted','采用后 · '+label,'2 GPU',adopted,{gpus:2,weightCells:2,productCountPerCard:data.ops[0].mul,localAdds:perCardAdds,mergeAdds:data.mergeAdds,merge:row?'sum':'concat'})
  ],
  note:row?'总乘积项仍是 8，局部权重仍来自同一份 W；两卡各 4 个并不代表总计算消失。切输入维时每个输出格只收到 1 个乘积，本地没有数学加法；两卡合并时再做 4 次同坐标相加，系统仍是 8 乘、4 加。合并的通信是新增成本，图中没有把它折算成实测加速。':'总乘积项仍是 8，局部权重仍来自同一份 W；两卡各 4 个并不代表总计算消失。切输出维时每个输出格要做 2 乘 1 加，每卡 4 乘 2 加；合并只拼接，没有跨卡加法，系统仍是 8 乘、4 加。合并的通信是新增成本，图中没有把它折算成实测加速。'
 };
}

function buildPp(scenario){
 const n=Number(scenario?.data?.n)||2;
 const count=n===2?2:1;
 const baselineSlots=[];const adopted0=[];const adopted1=[];
 for(let mb=1;mb<=count;mb++)for(let layer=1;layer<=4;layer++)baselineSlots.push({mb,layer});
 const span=2*count+2;
 const scale=Math.max(baselineSlots.length,span);
 for(let time=0;time<span;time++){
  const s0=time<2*count?{mb:Math.floor(time/2)+1,layer:(time%2)+1}:null;
  const s1=time>=2&&time<2*count+2?{mb:Math.floor((time-2)/2)+1,layer:(time-2)%2+3}:null;
  adopted0.push(s0);adopted1.push(s1);
 }
 const slot=(entry,lane,time,state)=>{
  const node=el('span','pa-slot');
  node.dataset.state=state;node.dataset.lane=String(lane);node.dataset.time=String(time);
  if(entry){
   node.classList.add('is-work','pa-mb'+entry.mb);
   node.dataset.work='MB'+entry.mb+':L'+entry.layer;
   node.dataset.mb=String(entry.mb);node.dataset.layer=String(entry.layer);
   node.setAttribute('aria-label',`时隙 ${time+1}，GPU ${lane}，MB${entry.mb} 第 ${entry.layer} 层`);
   node.append(el('span','pa-slot-mb','MB'+entry.mb),el('span','pa-slot-layer','L'+entry.layer));
  }else{
   const fill=lane===1&&time<2;
   node.classList.add('is-bubble');node.dataset.bubble=fill?'fill':'drain';
   node.setAttribute('aria-label',`时隙 ${time+1}，GPU ${lane} 空闲：${fill?'等待激活':'没有后续微批次'}`);
   node.append(el('span','pa-slot-mb',fill?'等':'空'),el('span','pa-slot-layer',fill?'激活':'排空'));
  }
  return node;
 };
 const lane=(entries,rank,layerNote)=>{
  const row=el('div','pa-lane');
  const label=el('div','pa-lane-label');label.append(el('strong','','GPU '+rank),el('span','',layerNote));
  const cells=el('div','pa-timeline');cells.style.setProperty('--slots',String(scale));
  entries.forEach((entry,time)=>cells.append(slot(entry,rank,time,entry?'work':'bubble')));
  row.append(label,cells);return row;
 };
 const axis=()=>{
  const row=el('div','pa-axis');
  const ticks=el('div','pa-ticks');ticks.style.setProperty('--slots',String(scale));
  for(let time=1;time<=scale;time++)ticks.append(el('span','pa-tick',String(time)));
  row.append(el('span','pa-axis-label','时隙'),ticks);return row;
 };
 const scaleNote=`同一横向刻度：1 格 = 1 个教学时隙 · 共 ${scale} 格`;
 const baselineLane=lane(baselineSlots,0,'持有 L1–L4');
 const adoptedLanes=el('div','pa-lanes');
 adoptedLanes.append(lane(adopted0,0,'持有 L1–L2'),lane(adopted1,1,'持有 L3–L4'));
 const handoffs=el('div','pa-handoffs');
 handoffs.append(el('span','pa-handoff-label','stage 间激活'));
 for(let mb=1;mb<=count;mb++)handoffs.append(el('span','pa-handoff','MB'+mb+'：L2 → L3'));
 const baseline=[
  el('p','pa-scale-note',scaleNote),
  baselineLane,
  axis(),
  stats(['1 张 GPU','每卡保存 4 层',(4*count)+' 段层计算',baselineSlots.length+' 个 GPU·时隙','0 个气泡','完成于第 '+baselineSlots.length+' 格']),
  el('p','pa-caption','同一张卡按 L1→L4 依次执行；不同微批次也在同一条执行路径上排队。')
 ];
 const adopted=[
  el('p','pa-scale-note',scaleNote),
  adoptedLanes,
  axis(),
  handoffs,
  stats(['2 张 GPU','每卡保存 2 层',(4*count)+' 段层计算','2 × '+span+' = '+(2*span)+' 个 GPU·时隙','4 个气泡（填充/排空）','完成于第 '+span+' 格']),
  el('p','pa-caption',count===2?'两侧同一刻度：左 8 格、右 6 格；同一 MB 仍然 L1→L2→L3→L4。重叠只发生在不同 MB 之间；填充、排空与激活交接是新增成本。':'两侧同一刻度 4 格；只有一个 MB 时没有第二个微批次补位，两 stage 仍有 2 格填充 + 2 格排空。')
 ];
 return {
  attrs:{mb:count,slots:baselineSlots.length,adoptedSlots:span,scaleSlots:scale},
  aria:`PP 采用前后对照：同样 ${count} 个微批次、四层模型，两侧使用同一时隙刻度；单卡保存 4 层共 ${baselineSlots.length} 格，两 stage 各保存 2 层并在第 ${span} 格完成，显示重叠、激活交接与填充排空。`,
  title:'PP · 同样 '+count+' 个 MB，单卡与两 stage',
  subtitle:count===2?`同一横向刻度（1 格 = 1 个教学时隙）：单卡 ${baselineSlots.length} 格，两 stage 完成于第 ${span} 格；MB1 后半与 MB2 前半重叠。`:`同一横向刻度（1 格 = 1 个教学时隙）：单卡与两 stage 都是 ${scale} 格；只有一个 MB，没有第二个微批次可补位。`,
  extra:'两卡（额外 1 张 GPU）',
  sides:[
   side('baseline','未采用 · 单卡','1 GPU',baseline,{gpus:1,layersPerGpu:4,busy:4*count,bubbles:0,makespan:4*count,gpuSlots:baselineSlots.length}),
   side('adopted','采用后 · 两 stage','2 GPU',adopted,{gpus:2,layersPerGpu:2,busy:4*count,bubbles:4,makespan:span,gpuSlots:2*scale,overlap:2*(count-1)})
  ],
  note:count===2?'两侧横轴使用同一时隙宽度；采用后完成于第 6 格，但用 2 张 GPU、共 12 个 GPU·时隙，其中 4 格是填充/排空。时隙等长只表示依赖和重叠位置，不是实测耗时。':'保留单 MB 分支：两侧同 4 格，两 stage 用 2 张 GPU、共 8 个 GPU·时隙，其中 4 格是气泡；没有用“1 个 MB 对 2 个 MB”制造不等工作量比较。'
 };
}

function dpRequest(name,value){
 const chip=el('span','pa-request pa-req-'+name);
 chip.dataset.request=name;
 chip.append(el('strong','',name+' · x='+value));
 const work=el('span','pa-request-work');
 work.append(el('span','','L1'),el('span','','→'),el('span','','L2'));
 chip.append(work);return chip;
}
function dpModel(title,layers,note){
 const model=el('div','pa-model');
 model.append(el('strong','pa-model-name',title));
 const list=el('div','pa-layers');layers.forEach(layer=>list.append(el('span','pa-layer',layer)));
 model.append(list,el('span','pa-model-note',note));return model;
}
function buildDp(){
 const baselineRequest=el('div','pa-requests');
 baselineRequest.append(dpRequest('R1',1),dpRequest('R2',3));
 const baselineBatch=el('div','pa-batch');
 baselineBatch.append(el('span','pa-batch-label','同一 batch / 队列'),baselineRequest);
 const baselineModel=dpModel('完整模型 · 1 份参数',['L1 ×2','L2 +1'],'1 条共享执行容量；两请求可合批，也可先后进入，由调度决定');
 const baselineOut=el('div','pa-outputs');
 baselineOut.append(el('span','pa-output pa-out-0','R1 → 3'),el('span','pa-output pa-out-1','R2 → 7'));
 const baselineStorage=el('div','pa-storage');
 baselineStorage.append(el('span','pa-storage-title','参数存储'),el('span','pa-block','L1'),el('span','pa-block','L2'));
 const baseline=[
  baselineBatch,
  el('div','pa-down','↓ 进入同一完整模型（可同批或先后）'),
  baselineModel,
  baselineOut,
  baselineStorage,
  stats(['完整副本 1','执行路径 1（可合批）','每请求 2 层计算','参数块 2'])
 ];
 const replicas=el('div','pa-replicas');
 [['A','R1',1,'3'],['B','R2',3,'7']].forEach(([copy,name,value,out])=>{
  const card=el('div','pa-replica');card.dataset.replica=copy;card.dataset.request=name;
  card.append(dpRequest(name,value),el('div','pa-down','↓'));
  card.append(dpModel('完整副本 '+copy,['L1 ×2','L2 +1'],'独立执行路径'));
  const result=el('span','pa-output','R1 → 3');result.dataset.output=out;
  result.textContent=name+' → '+out;card.append(result);replicas.append(card);
 });
 const adoptedStorage=el('div','pa-storage');
 adoptedStorage.append(el('span','pa-storage-title','参数存储'));
 [['A','L1 L2'],['B','L1 L2']].forEach(([copy,layers])=>{
  const group=el('span','pa-storage-copy');group.append(el('em','','副本 '+copy));
  layers.split(' ').forEach(layer=>group.append(el('span','pa-block',layer)));
  adoptedStorage.append(group);
 });
 const adopted=[
  replicas,
  adoptedStorage,
  stats(['完整副本 2','执行路径 2','每请求仍 2 层计算','参数块 4','单请求计算不减少'])
 ];
 return {
  attrs:{requests:2},
  aria:'DP 采用前后对照：同样 R1、R2 和完整模型；一个副本让两请求共享容量（可合批或先后），两个完整副本各服务一条请求。单请求计算不减少，参数存储随副本增加。',
  title:'DP · 同样两条请求，一份与两份完整模型',
  subtitle:'左：两请求共享一份完整模型（可同批）；右：两个完整副本各自推进一条请求。',
  extra:'两份完整模型（额外 1 份参数副本）',
  sides:[
   side('baseline','未采用 · 一个完整副本','1 副本',baseline,{replicas:1,lanes:1,workPerRequest:2,layerCopies:2,batch:'available'}),
   side('adopted','采用后 · 两个完整副本','2 副本',adopted,{replicas:2,lanes:2,workPerRequest:2,layerCopies:4,batch:'per-replica'})
  ],
  note:'单请求计算没有减少：R1、R2 各自仍执行 L1→L2，输出仍是 3 和 7。变化是并发服务容量从 1 条共享执行路径变为 2 条独立路径；参数存储从 2 个层块变为 4 个。是否有吞吐收益取决于请求量、合批与调度，单实例也可 batch，本图不把单实例画成只能串行。'
 };
}

function buildPcp(scenario){
 const model=root.ParallelTeachingModel;
 const fallback=zigzag=>({ranks:zigzag?[[0,1,6,7],[2,3,4,5]]:[[0,1,2,3],[4,5,6,7]],work:zigzag?[18,18]:[10,26],single:36});
 const context=zigzag=>{
  if(model&&typeof model.context==='function'){try{return model.context(zigzag);}catch{}}
  return fallback(zigzag);
 };
 const continuous=context(false),zigzag=context(true);
 const current=Array.isArray(scenario?.data?.ranks?.[0])&&scenario.data.ranks[0].includes(6)?'zigzag':'continuous';
 const cases=[
  {key:'single',label:'单卡基线',badge:'未采用 · 1 GPU',ranks:[[0,1,2,3,4,5,6,7],[]],work:[continuous.single??36,0]},
  {key:'continuous',label:'两卡 · 连续切分',badge:'采用 · 2 GPU',ranks:continuous.ranks,work:continuous.work},
  {key:'zigzag',label:'两卡 · 前后配对',badge:'采用 · 2 GPU',ranks:zigzag.ranks,work:zigzag.work}
 ];
 const box=el('section','pcp-contrast pa-pcp');
 const head=el('div','pcp-contrast-head');
 head.append(el('strong','','同一请求 · 同样 8 个 Query · 三种分工'),el('span','','数字 = 允许的 Q–K 配对数；当前分支用边框标出。'));
 const grid=el('div','pcp-contrast-grid pa-pcp-grid');
 for(const split of cases){
  const owners=Array(8).fill(0);
  split.ranks.forEach((rows,r)=>rows.forEach(i=>owners[i]=r));
  const total=split.work[0]+split.work[1];
  const card=el('article','pcp-split'+(split.key===current?' is-current':'')+(split.key==='single'?' is-baseline':''));
  card.dataset.pcpSplit=split.key;card.dataset.work=split.work.join('/');card.dataset.total=String(total);
  if(split.key===current)card.setAttribute('aria-current','true');
  const cardHead=el('header','pcp-split-head');
  cardHead.append(el('h3','',split.label));
  cardHead.append(el('span','pa-pcp-badge',split.key===current?'当前场景':split.badge));
  const list=el('ol','pcp-query-row');
  list.setAttribute('aria-label',split.label+'：Q1 到 Q8 的归属和各自允许的配对数');
  owners.forEach((owner,i)=>{
   const item=el('li','pcp-query pcp-rank-'+owner);
   item.dataset.owner=String(owner);
   item.setAttribute('aria-label','Q'+(i+1)+'，Rank '+owner+'，允许 '+(i+1)+' 个配对');
   item.append(el('strong','','Q'+(i+1)),el('small','',String(i+1)));
   list.append(item);
  });
  const counts=el('dl','pcp-counts');
  [0,1].forEach(r=>{
   const idle=split.key==='single'&&r===1;
   const cell=el('div','pcp-count pcp-rank-'+r+(idle?' is-idle':''));
   cell.append(el('dt','','Rank '+r));
   if(idle){
    cell.append(el('dd','is-idle','未参与'));
    cell.title='单卡基线只有 Rank 0 参与；Rank 1 未投入';
   }else{
    cell.append(el('dd','',String(split.work[r])));
    cell.lastChild.append(el('em','','对'));
    const terms=split.ranks[r]||[];
    cell.title=terms.length?terms.map(i=>i+1).join('+')+' = '+split.work[r]:'未参与';
   }
   counts.append(cell);
  });
  const sum=el('div','pcp-count pcp-total');
  sum.append(el('dt','','总计'),el('dd','',String(total)));
  sum.lastChild.append(el('em','','对'));
  counts.append(sum);
  const formulaParts=split.ranks.map(rows=>rows.length?rows.map(i=>i+1).join('+')+' = '+rows.reduce((a,i)=>a+i+1,0):(split.key==='single'?'未参与':'0'));
  const formula=el('p','pcp-formula',formulaParts.join(' · '));
  card.append(cardHead,list,counts,formula);
  if(split.key==='single')card.append(el('p','pa-pcp-cost','36 对都在 Rank 0 这一张卡上算；Rank 1 未参与，不是第二张卡。'));
  else card.append(el('p','pa-pcp-cost','每行仍需所需历史 K/V；本例 AllGather 副本。'));
  grid.append(card);
 }
 box.append(head,grid);
 return {
  attrs:{singlePairs:continuous.single??36,work:continuous.work.join('/')+'|'+zigzag.work.join('/'),currentSplit:current,allgather:'true'},
  aria:'PCP 采用前后对照：同一请求、同一组 Q1…Q8；三种分工方案，不是三张卡。Qi 只允许读取 K1…Ki，所以 Qi 有 i 个配对，单卡共 36 对，只有 Rank 0 参与，Rank 1 未参与。两卡连续切分分为 10 / 26，前后配对各 18 / 18，总数仍是 36；Query 行归属改变，但每行仍要取得所需历史 K/V。本图 AllGather K/V 副本，不能据此说每卡 KV 驻留减半。配对数不是耗时或加速比。',
  title:'PCP · 同样 Q1…Q8，单卡与两种两卡分法',
  subtitle:'三种分工方案固定同一请求、同一组 Q1…Q8；单卡 36 对，两卡连续 10/26 或前后配对 18/18，总数不变。',
  extra:'两卡（额外 1 张 GPU）；AllGather 交换 K/V',
  body:box,
  note:'单卡与两卡都计算同样的 36 个允许配对，区别是 Query 行由谁负责。连续切分让后半段 Query 承担更多因果配对；前后配对把首尾行放在一起，使两卡各 18 对。每行仍要读取所需历史 K/V，本图的 AllGather 会保留 K/V 副本，不能写成每卡 KV 必减半。配对数只表示工作量分布，不是耗时或加速比。'
 };
}

function buildDcp(scenario){
 const model=root.ParallelTeachingModel;
 const equal=Array.isArray(scenario?.data?.sums)&&scenario.data.sums[0]===scenario.data.sums[1];
 const d=model&&typeof model.decode==='function'?model.decode(equal):null;
 if(!d)return null;
 const kLabel=i=>d.exp[i]===1?'0':d.exp[i]===3?'ln3':String(d.exp[i]);
 const positionChip=(i,rank)=>{
  const chip=el('div','pa-dcp-kv');
  chip.dataset.position='p'+(i+1);chip.dataset.rank=String(rank);
  chip.append(el('strong','','p'+(i+1)),el('span','','K='+kLabel(i)+' · V='+d.values[i]));
  return chip;
 };
 const resultBox=(label,value,cls,note)=>{
  const box=el('div','pa-dcp-result '+cls);
  box.append(el('span','pa-dcp-result-label',label),el('strong','',String(value)),el('small','',note));
  return box;
 };
 const baselineKvs=el('div','pa-dcp-kvs');
 [0,1,2,3].forEach(i=>baselineKvs.append(positionChip(i,0)));
 const baselineEquation=el('p','pa-dcp-equation',`(${d.exp.map((e,i)=>e+'×'+d.values[i]).join(' + ')}) / (${d.exp.join(' + ')}) = ${d.merge.numerator} / ${d.merge.denominator} = ${d.result}`);
 const baseline=[
  (()=>{const q=el('div','pa-dcp-query','Q = 1');q.dataset.query='1';return q;})(),
  baselineKvs,
  baselineEquation,
  (()=>{const row=el('div','pa-dcp-results is-single');row.append(resultBox('完整读取 4 个位置',d.result,'is-correct','一处保存、一处读取全部历史'));return row;})(),
  stats(['1 张 GPU','保存/读取 4 个历史位置','p1…p4 都参与同一次 Attention','一次全局归一化']),
  el('p','pa-caption','同一个 Q 读取全部 4 个位置，分子和分母都在一处得到。')
 ];
 const ranks=el('div','pa-dcp-ranks');
 [0,1].forEach(r=>{
  const card=gpu(r,'读取 2 个位置');
  card.dataset.positions=d.values.slice(r*2,r*2+2).map((_,i)=>'p'+(r*2+i+1)).join(' ');
  const kvs=el('div','pa-dcp-kvs');
  d.values.slice(r*2,r*2+2).forEach((_,i)=>kvs.append(positionChip(r*2+i,r)));
  const local=el('div','pa-dcp-local');
  local.append(
   el('span','','exp：'+d.exp.slice(r*2,r*2+2).join(' + ')+' = '+d.sums[r]),
   el('span','','分子：'+d.exp.slice(r*2,r*2+2).map((e,k)=>e+'×'+d.values[r*2+k]).join(' + ')+' = '+d.numerators[r]),
   el('strong','','局部 O = '+d.local[r]+'，Z = '+d.sums[r])
  );
  card.append(kvs,local);ranks.append(card);
 });
 const exchange=el('div','pa-dcp-exchange');
 exchange.append(
  el('strong','','本轮 Decode 通信：交换局部 O / Z'),
  el('span','pa-dcp-stat-chip','Rank 0 → O='+d.local[0]+', Z='+d.sums[0]),
  el('span','pa-dcp-stat-chip','Rank 1 → O='+d.local[1]+', Z='+d.sums[1]),
  el('span','pa-dcp-stat-note','各 1 组统计量；合并后得到新的全局输出')
 );
 const merge=el('p','pa-dcp-equation',`全局 O = (Z0×O0 + Z1×O1) / (Z0+Z1) = (${d.sums[0]}×${d.local[0]} + ${d.sums[1]}×${d.local[1]}) / ${d.merge.denominator} = ${d.result}`);
 const results=el('div','pa-dcp-results');
 results.append(
  resultBox('完整读取（对照）',d.result,'is-correct','同一 Q、同一 4 个位置'),
  resultBox('局部 O / Z 加权',d.result,'is-correct','按 Z 归一化后合并'),
  resultBox('局部平均',d.mean,equal?'is-coincident':'is-wrong',equal?'Z 相等时碰巧相同':'不等权时不是正确答案')
 );
 const adopted=[
  (()=>{const q=el('div','pa-dcp-query','Q = 1');q.dataset.query='1';q.append(el('span','pa-dcp-query-note','Q 参与两个分片'));return q;})(),
  ranks,
  exchange,
  merge,
  results,
  stats(['2 张 GPU','每卡保存/读取 2 个位置（4→2）','局部 O '+d.local.join(' / ')+'，Z '+d.sums.join(' / '),'每步 Decode 1 轮 O / Z 交换','合并结果 '+d.result]),
  el('p','pa-caption',equal?'等权场景两片 Z 都是 2，局部平均与加权合并都为 4；仍需先确认 Z 才能安全合并。':'不等权场景局部平均为 4，但按 Z 加权合并才是正确的 5。')
 ];
 return {
  attrs:{equal:equal?'true':'false',result:d.result,localMean:d.mean,reads:4,readsPerRank:2,rounds:1,weights:d.weights.join('/'),local:d.local.join('/'),sums:d.sums.join('/')},
  aria:`DCP 采用前后对照：同一 Q、同一组 4 个历史位置。单卡读取 4 个位置，完整结果为 ${d.result}；两卡各保存并读取 2 个位置，局部 O 为 ${d.local.join(' / ')}、Z 为 ${d.sums.join(' / ')}，每步 Decode 交换一轮局部结果与统计量，按 Z 加权合并仍得 ${d.result}。${equal?'等权时局部平均碰巧相同，但不能当作通用规则。':'不等权时局部平均为 '+d.mean+'，不是正确答案。'}`,
  title:'DCP · 同样 Q 与 4 个历史位置，完整读取与两片',
  subtitle:'左：一处读取 p1…p4；右：两张卡各读 2 个位置，交换局部 O / Z 后按全局归一化合并。',
  extra:'两卡（额外 1 张 GPU）；每步交换 O / Z',
  sides:[
   side('baseline','未采用 · 完整历史在一处','1 GPU',baseline,{gpus:1,reads:4,localMean:d.mean,result:d.result}),
   side('adopted','采用后 · 历史位置分两片','2 GPU',adopted,{gpus:2,reads:2,localMean:d.mean,result:d.result,rounds:1})
  ],
  note:'4→2 的是本例每卡保存/读取的历史位置数；新增的是每轮 Decode 合并所需的局部结果与统计量通信。局部平均只有在 Z0=Z1 时才等于加权合并；不等权场景 (2+6)/2=4 不是正确答案 5。这里只展示历史位置分片和 O/Z 合并，不推导其他 TP/DCP 布局下每卡 KV 总量。'
 };
}

/* SP's TP/TP+SP comparison is drawn inside the mechanism canvas itself; it no
   longer mounts a second panel above the player. */
const builders={tp:buildTp,pp:buildPp,dp:buildDp,pcp:buildPcp,dcp:buildDcp};

function mount(kind,viewport,scenario){
 const builder=builders[kind];if(!builder)return null;
 const spec=builder(scenario);if(!spec)return null;
 const section=el('section','parallel-adoption');
 section.dataset.adoption=kind;
 for(const [key,value] of Object.entries(spec.attrs||{}))section.dataset[key]=String(value);
 section.setAttribute('aria-label',spec.aria);
 const head=el('header','pa-head');
 const heading=el('div','pa-heading');
 heading.append(el('strong','','采用前后 · 同一输入与工作量'),el('span','',spec.title+'。'+spec.subtitle));
 head.append(heading,el('span','pa-resource',spec.extra));
 const grid=el('div','pa-grid');
 (spec.sides||[]).forEach(article=>grid.append(article));
 const content=spec.body||grid;
 const note=el('p','pa-note',spec.note);
 section.append(head,content,note);
 const film=viewport.closest('.mechanism-film');
 film?.querySelectorAll(':scope > .parallel-adoption').forEach(node=>node.remove());
 const anchor=viewport.closest('.motion-theater')||viewport;
 anchor.before(section);
 return {element:section,destroy(){section.remove();}};
}

root.ParallelAdoption={mount,kinds:Object.keys(builders)};
})(typeof globalThis==='object'?globalThis:this);
