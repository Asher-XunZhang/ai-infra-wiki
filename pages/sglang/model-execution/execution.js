/* One deterministic clock drives geometry, payloads, states and explanations. */
(()=>{'use strict';const host=document.querySelector('[data-model-flow]'),F=window.ModelExecutionFlow,M=window.ExecutionModel;if(!host||!F||!M)return;
const $=id=>host.querySelector('#'+id),ns='http://www.w3.org/2000/svg',reduced=matchMedia('(prefers-reduced-motion: reduce)');
const make=(tag,attrs={},text)=>{const el=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))el.setAttribute(k,v);if(text!==undefined)el.textContent=text;return el;};
const txt=(el,s)=>{if(el.textContent!==String(s))el.textContent=s;};
let view=host.dataset.modelFlow,timeline,time=0,playing=false,raf=0,last=0,previousPhase=-1,workbench=null,nodeEls={},edgeEls={},packets=[],bars=[];
function options(){return {mode:$('mf-mode')?.value||'prefill',origin:$('execution-input')?.value||'prefill',count:Number($('graph-count')?.value||3),condition:$('graph-condition')?.value||'ready',rows:Number($('matrix-rows')?.value||1),upgrade:$('hardware-upgrade')?.value||'base'};}
function draw(){
 workbench=null;if(['chain','graph'].includes(view)){workbench=window.ExecutionWorkbench.create($('mf-svg'),timeline);return;}
 const svg=$('mf-svg'),g=timeline.geometry;svg.setAttribute('viewBox',`0 0 1040 ${g.height}`);svg.replaceChildren(make('title',{id:'mf-svg-title'},view==='model'?'DeepSeek-V3 的模型计算与 PP 交接':'模型执行、图重放与计算访存机制'));nodeEls={};edgeEls={};packets=[];bars=[];
 const defs=make('defs');svg.append(defs);for(const kind of ['token','activation','kv','logits','reference']){const marker=make('marker',{id:'mf-arrow-'+kind,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto',markerUnits:'userSpaceOnUse'});marker.append(make('path',{d:'M0 0 L10 5 L0 10 Z',class:'mf-'+kind}));defs.append(marker);}
 if(view==='model'){
  svg.append(make('text',{x:28,y:186,class:'mf-caption'},'PP 通道只画本次激活交接；各层 MLA 缓存由对应阶段保存。四个阶段为教学拓扑，层数与物理主机未展开。'));
  svg.append(make('rect',{x:16,y:203,width:1008,height:303,rx:12,class:'mf-boundary'}),make('text',{x:30,y:231,id:'mf-layer-title',class:'mf-group-title'},'当前阶段 · 一层的接口放大'));
 }else if(view==='chain'){
  svg.append(make('text',{x:28,y:25,class:'mf-caption'},'单 rank · 同一 Scheduler 进程内的对象与接口；不是多个网络服务。'));
  svg.append(make('rect',{x:16,y:214,width:480,height:170,rx:12,class:'mf-boundary'}),make('text',{x:28,y:242,class:'mf-group-title'},'设备侧放大：GPU 上的缓存与 Attention 算子'));
 }
 for(const e of Object.values(g.edges)){const group=make('g',{class:'mf-wire mf-'+e.kind,'data-edge':e.id}),path=make('path',{d:e.points.map((p,i)=>(i?'L':'M')+p.join(' ')).join(' '),'marker-end':'url(#mf-arrow-'+e.kind+')'});group.append(path,make('text',{x:e.labelPosition[0],y:e.labelPosition[1],'text-anchor':'middle'},e.label));svg.append(group);edgeEls[e.id]={group,path,length:path.getTotalLength()};}
 for(const [id,n]of Object.entries(g.nodes)){
  const group=make('g',{class:'mf-node',transform:`translate(${n.x} ${n.y})`,'data-node':id}),body=make('text',{x:12,y:66,class:'mf-state'}),body2=make('text',{x:12,y:85,class:'mf-state'}),cellGroup=make('g'),bar=make('rect',{x:12,y:n.h-2,width:0,height:2,class:'mf-workbar'});
  group.append(make('rect',{width:n.w,height:n.h,rx:7,class:'mf-node-bg'}),make('text',{x:12,y:24,class:'mf-node-title'},n.title),make('text',{x:12,y:44,class:'mf-node-sub'},n.sub),body,body2,cellGroup,bar);svg.append(group);
  const cells=Array.from({length:8},(_,i)=>{const cell=make('g',{class:'mf-cell',visibility:'hidden'});cell.append(make('rect',{height:21,rx:3}),make('text',{'text-anchor':'middle',y:15}));cellGroup.append(cell);return cell;});nodeEls[id]={group,body,body2,bar,cells,n};
 }
 if(view==='hardware'){
  for(const [id,x]of [['compute',52],['memory',572]]){const outline=make('rect',{x,y:400,width:0,height:18,class:'mf-cost-base'}),fill=make('rect',{x,y:400,width:0,height:18,class:'mf-cost-fill'});svg.append(outline,fill);bars.push({id,outline,fill});}
 }
 for(let i=0;i<5;i++){const group=make('g',{class:'mf-packet',visibility:'hidden'}),shape=make('path'),label=make('text',{'text-anchor':'middle',y:4});group.append(shape,label);svg.append(group);packets.push({group,shape,label});}
}
function field(id,a='',b=''){const n=nodeEls[id];if(!n)return;txt(n.body,a);txt(n.body2,b);}
function cells(id,labels,filled=labels.length,pad=false){const n=nodeEls[id],gap=Math.min(38,(n.n.w-24)/Math.max(1,labels.length)),width=Math.min(32,gap-4);n.cells.forEach((el,i)=>{el.setAttribute('visibility',i<labels.length?'visible':'hidden');if(i>=labels.length)return;el.setAttribute('transform',`translate(${12+i*gap} ${n.n.h-29})`);el.querySelector('rect').setAttribute('width',width);el.querySelector('text').setAttribute('x',width/2);txt(el.querySelector('text'),labels[i]===null?'补':labels[i]);el.classList.toggle('filled',i<filled&&labels[i]!==null);el.classList.toggle('pad',pad&&labels[i]===null);});}
function renderModel(v){const s=v.state,p=v.phase,stage=Math.max(0,p.stage),count=timeline.options.mode==='decode'?5:4;
 const compact=p.id.startsWith('local-');
 for(let k=0;k<4;k++){const id='stage'+k;field(id,k<=s.completed?'本级层区间已完成':k===s.arrived&&s.embedded?(k!==stage?'输入已到达':compact?(v.u<.55?'执行本级层 · 读写各层历史':'继续完成本级层区间'):'正在处理本次输入'):'等待本次输入',`代表层缓存：${s.kv[k]} 个位置`);nodeEls[id].group.classList.toggle('is-current-stage',k===stage);cells(id,k===0&&!s.embedded?(timeline.options.mode==='decode'?['y1']:['p1','p2','p3','p4']):k<=s.arrived?['h',s.residualReady?'r':'∅']:[]);}
 txt($('mf-layer-title'),stage===0?'Stage 0 代表层 · 展开 Attention、残差与前馈，其他本级层继续执行':`Stage ${stage} 层区间概览 · 本级独立计算并更新缓存，层内沿用下方接口`);
 field('norm1','hidden_states / residual');field('mla',v.active.some(e=>e.nodes.includes('mla'))?'处理当前可读上下文':'查询与本层历史对应');field('norm2','保留主干，加入更新');field('mlp','按层选择一种实现');field('layerout','hidden_states + residual');
 field('layerkv','');cells('layerkv',Array.from({length:count},(_,i)=>i<4?'p'+(i+1):'y1'),s.kv[stage]);
 nodeEls.layerkv.group.classList.toggle('is-writing',v.active.some(e=>e.edge==='write')||compact&&v.u>=.3&&v.u<.55);
 field('norm',p.id==='norm'?'末级合并 residual':'等待末级全部层');field('logits',s.logits?'词表分数已就绪':'等待最终隐藏状态');field('sample',s.sampled?'已选出下一 ID':'等待候选分数');field('token',s.delivered?(timeline.options.mode==='decode'?'y2 · 尚无它的 KV':'y1 · 尚无它的 KV'):'尚未返回');
 $('mf-ledger').textContent=`输入：${timeline.options.mode==='decode'?'y1（已有 p1…p4 历史）':'p1…p4'} · 本次交接已到 Stage ${s.arrived} · ${s.delivered?'下一 token 已返回':'尚未完成采样回程'}`;
}
function renderHardware(v){const d=M.matrix(timeline.options.rows,timeline.options.upgrade),base=M.matrix(timeline.options.rows),max=Math.max(base.mathMs,base.memoryMs),cost=v.phase.index>=2,done=v.phase.index>=3||v.phase.id==='cost'&&v.u>=.94;
 field('x',`M = ${d.m} 个新位置`,d.m>4?'格子只画前 4 行代表':'');field('w','权重大小固定：32 MiB');field('y',v.state.calculated?`Y 已产生 ${d.m} 行`:'尚未得到结果');
 cells('x',Array.from({length:Math.min(d.m,4)},(_,i)=>'行'+(i+1)));cells('w',['w₁','w₂','w₃','…']);cells('y',Array.from({length:Math.min(d.m,4)},(_,i)=>'行'+(i+1)),v.state.calculated?Math.min(d.m,4):0);
 field('compute',done?d.mathMs.toFixed(3)+' ms':'等待公式核算');field('memory',done?d.memoryMs.toFixed(3)+' ms':'等待公式核算');field('bound',v.phase.id==='bound'?`max = ${d.lowerMs.toFixed(3)} ms · ${d.limit==='memory'?'访存项更长':'计算项更长'}`:'先分别核对两项成本');
 const reveal=!cost?0:done?1:Math.min(1,v.u/.94);
 for(const b of bars){const key=b.id==='compute'?'mathMs':'memoryMs';b.outline.setAttribute('width',cost?416*base[key]/max:0);b.fill.setAttribute('width',416*d[key]/max*reveal);}
 $('mf-ledger').textContent=`假设算力 ${timeline.options.upgrade==='compute'?16:8} TFLOP/s · 带宽 ${timeline.options.upgrade==='memory'?200:100} GB/s · 两条柱共用 0–${max.toFixed(3)} ms 刻度 · 虚线框 = 原配`;
}
const shape={token:'M-10 -10H10V10H-10Z',activation:'M-14 -7H14V7H-14Z',kv:'M-11 -10H11V-2H-11Z M-11 2H11V10H-11Z',logits:'M0 -12L12 0L0 12L-12 0Z',reference:'M6 0A6 6 0 1 1-6 0A6 6 0 1 1 6 0'};
const format=t=>Math.floor(t/60)+':'+String(Math.floor(t%60)).padStart(2,'0');
function render(){const v=F.at(timeline,time),p=v.phase;host.dataset.view=view;host.dataset.time=time.toFixed(3);host.dataset.phase=p.id;host.dataset.playing=String(playing);
 $('mf-time').textContent=format(time)+' / '+format(timeline.duration);$('mf-seek').value=time;$('mf-seek').setAttribute('aria-valuetext',`${format(time)}，${p.title}`);$('mf-prev').disabled=time===0;$('mf-next').disabled=v.ended;$('mf-toggle').textContent=playing?'暂停':v.ended?'重新播放':'播放全程';$('mf-toggle').setAttribute('aria-pressed',String(playing));
 if(previousPhase!==p.index){previousPhase=p.index;txt($('mf-phase'),`${String(p.index+1).padStart(2,'0')} / ${timeline.phases.length} · ${p.title}`);txt($('mf-explanation'),p.explanation);$('mf-source').href=p.source?F.sourceURL(p.source):'https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#understanding-performance';txt($('mf-source'),p.source?'固定源码 ↗':'计算与访存依据 ↗');}
 txt($('mf-payload'),v.ended?'本次过程结束，可拖动时间回看任意交接。':p.summary||v.active.map(e=>e.payload).join('；')||p.title);
 if(workbench){$('mf-ledger').textContent=workbench.render(v,reduced.matches);return;}
 const activeNodes=new Set(v.active.flatMap(e=>e.nodes));for(const [id,n]of Object.entries(nodeEls)){n.group.classList.toggle('is-active',activeNodes.has(id));n.bar.setAttribute('width',0);}
 for(const [id,e]of Object.entries(edgeEls))e.group.classList.toggle('is-active',v.active.some(a=>a.edge===id));
 ({model:renderModel,hardware:renderHardware})[view](v);
 for(const e of v.active)if(!e.edge)for(const id of e.nodes)nodeEls[id].bar.setAttribute('width',(nodeEls[id].n.w-24)*e.progress);
 const active=v.active.filter(e=>e.edge);packets.forEach((item,i)=>{const e=active[i];if(!e||reduced.matches){item.group.setAttribute('visibility','hidden');return;}const path=edgeEls[e.edge],point=path.path.getPointAtLength(path.length*e.progress);item.group.setAttribute('visibility','visible');item.group.setAttribute('class','mf-packet mf-'+e.kind);item.group.setAttribute('transform',`translate(${point.x} ${point.y})`);item.group.dataset.edge=e.edge;item.shape.setAttribute('d',shape[e.kind]);txt(item.label,e.kind==='token'?'ID':e.edge.startsWith('pp')?'h/r':'');});
}
function pause(){playing=false;cancelAnimationFrame(raf);last=0;render();}
function rebuild(){playing=false;cancelAnimationFrame(raf);time=0;last=0;previousPhase=-1;timeline=F.build(view,options());draw();$('mf-seek').max=timeline.duration;host.querySelectorAll('[data-view-settings]').forEach(el=>el.hidden=el.dataset.viewSettings!==view);host.querySelectorAll('[data-exec-view]').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.execView===view)));
 const legends={model:{token:'■ token ID',activation:'▰ hidden_states / residual',kv:'▤ MLA 历史',logits:'◆ 词表分数',reference:'┄ 调用 / 引用'},chain:{token:'■ R1 的数据',activation:'■ R2 的数据',kv:'实线：数据拷贝',reference:'虚线：引用 / 索引 / 提交'},graph:{kv:'实线：显存内更新',reference:'虚线：提交执行；空框：尚未就绪'},hardware:{activation:'▰ 矩阵数据',reference:'┄ 原配下界（虚线框）'}};
 host.querySelectorAll('.mf-legend [data-kind]').forEach(el=>{const label=legends[view][el.dataset.kind];el.hidden=!label;if(label)txt(el,label);});render();}
function tick(now){if(!playing)return;if(last)time=Math.min(timeline.duration,time+(now-last)/1000*Number($('mf-speed').value));last=now;if(time===timeline.duration)playing=false;render();if(playing)raf=requestAnimationFrame(tick);}
$('mf-toggle').addEventListener('click',()=>{if(playing)return pause();if(time===timeline.duration)time=0;playing=true;last=0;render();raf=requestAnimationFrame(tick);});$('mf-seek').addEventListener('input',()=>{time=Number($('mf-seek').value);pause();});$('mf-speed').addEventListener('change',()=>{last=0;});$('mf-reset').addEventListener('click',rebuild);
$('mf-next').addEventListener('click',()=>{const p=F.at(timeline,time).phase;time=timeline.phases[p.index+1]?.start??timeline.duration;pause();});$('mf-prev').addEventListener('click',()=>{const p=F.at(timeline,time).phase;time=time>p.start+.3?p.start:timeline.phases[Math.max(0,p.index-1)].start;pause();});
host.querySelectorAll('[data-exec-view]').forEach(el=>el.addEventListener('click',()=>{if(view===el.dataset.execView)return;view=el.dataset.execView;rebuild();}));host.querySelectorAll('.mf-settings select').forEach(el=>el.addEventListener('change',rebuild));
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});window.addEventListener('pagehide',pause);reduced.addEventListener('change',render);host.hidden=false;rebuild();
})();
