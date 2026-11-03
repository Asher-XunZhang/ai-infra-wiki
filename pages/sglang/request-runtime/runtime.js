(function(){
'use strict';
const host=document.querySelector('[data-runtime-lab]'),F=window.RequestRuntimeFlow,M=window.RequestRuntimeModel;
if(!host||!F||!M)return;
const get=id=>host.querySelector('#'+id),svg=get('runtime-graph'),ns='http://www.w3.org/2000/svg';
const make=(name,attrs={},text)=>{const e=document.createElementNS(ns,name);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;};
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
let timeline,time=0,playing=false,raf=0,last=0,phase=-1;
const nodeEls={},edgeEls={},packets=[];
function groupBox(x,y,w,h,title,cls){const g=make('g',{class:'rf-boundary '+cls});g.append(make('rect',{x,y,width:w,height:h,rx:12}),make('text',{x:x+14,y:y+25},title));svg.append(g);}
groupBox(18,28,168,314,'客户端 · 连接正常','client');
groupBox(204,28,244,314,'进程 A · HTTP / 前端','front');
groupBox(480,28,538,514,'进程 B · Scheduler / 执行 / 内存管理','scheduler');
groupBox(204,392,244,150,'进程 C · 反分词','detok');
const defs=make('defs');svg.prepend(defs);
for(const kind of ['request','tensor','kv','text','control','local']){
 const marker=make('marker',{id:'rf-arrow-'+kind,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto',markerUnits:'userSpaceOnUse'});
 marker.append(make('path',{d:'M0 0 L10 5 L0 10 Z',class:kind}));defs.append(marker);
}
for(const e of Object.values(F.edges)){
 const g=make('g',{class:'rf-wire '+e.kind,'data-edge':e.id}),d=e.points.map((p,i)=>(i?'L':'M')+p.join(' ')).join(' ');
 const path=make('path',{d,'marker-end':'url(#rf-arrow-'+e.kind+')'});
 g.append(path,make('text',{x:e.lx,y:e.ly,'text-anchor':'middle'},e.label));svg.append(g);edgeEls[e.id]={g,path,length:path.getTotalLength()};
}
for(const [id,n]of Object.entries(F.nodes)){
 const g=make('g',{class:'rf-node','data-node':id,transform:`translate(${n.x} ${n.y})`});
 const body=make('text',{x:12,y:69,class:'rf-node-state'}),sub=make('text',{x:12,y:44,class:'rf-node-sub'},n.sub);
 g.append(make('rect',{width:n.w,height:n.h,rx:7,class:'rf-node-bg'}),make('text',{x:12,y:24,class:'rf-node-title'},n.title),sub,body);
 const cells=make('g',{class:'rf-cells'});g.append(cells);
 const bar=make('rect',{x:12,y:n.h-6,height:3,width:0,rx:1,class:'rf-progress'});g.append(bar);svg.append(g);nodeEls[id]={g,body,cells,bar,sub};
}
function cells(id,count,x,y,gap,width,labels){
 const list=[];for(let i=0;i<count;i++){const g=make('g',{transform:`translate(${x+i*gap} ${y})`,class:'rf-cell'});g.append(make('rect',{width,height:22,rx:3}),make('text',{x:width/2,y:15,'text-anchor':'middle'},labels(i)));nodeEls[id].cells.append(g);list.push(g);}return list;
}
const kvCells=cells('memory',8,12,60,26,22,i=>i<4?'p'+(i+1):'y'+(i-3));
const inputCells=cells('tokenizer',4,12,91,38,32,i=>'p'+(i+1));
const outCells=cells('response',5,9,48,24,20,i=>'y'+(i+1));
const sampleCells=cells('result',5,12,75,29,24,i=>'y'+(i+1));
const batchCells=cells('batch',4,12,83,38,32,i=>'p'+(i+1));
for(let i=0;i<4;i++){
 const g=make('g',{class:'rf-packet',visibility:'hidden'}),shape=make('path'),label=make('text',{'text-anchor':'middle',y:4});g.append(shape,label);svg.append(g);packets.push({g,shape,label});
}
const palette={request:'M-10 -10 H10 V10 H-10 Z',tensor:'M-15 -7 H15 V7 H-15 Z',kv:'M-12 -10 H12 V-2 H-12 Z M-12 2 H12 V10 H-12 Z',text:'M10 0 A10 10 0 1 1 -10 0 A10 10 0 1 1 10 0',control:'M0 -12 L12 0 L0 12 L-12 0 Z',local:'M7 0 A7 7 0 1 1 -7 0 A7 7 0 1 1 7 0'};
const format=t=>Math.floor(t/60)+':'+String(Math.floor(t%60)).padStart(2,'0');
function options(){return {scenario:get('scenario').value,outputs:Number(get('outputs').value),cache:get('cache').checked};}
function render(){
 const view=F.at(timeline,time),f=view.phase.frame,s=view.state;
 host.dataset.playing=String(playing);get('runtime-snapshot').dataset.frame=f.id;host.dataset.time=time.toFixed(3);
 get('step').value=time;get('step').setAttribute('aria-valuetext',`${format(time)}，${f.title}`);
 get('runtime-time').textContent=format(time)+' / '+format(timeline.duration);
 get('previous').disabled=time===0;get('next').disabled=view.ended;
 get('runtime-toggle').textContent=playing?'暂停':view.ended?'重新播放':'播放全程';get('runtime-toggle').setAttribute('aria-pressed',String(playing));
 if(phase!==view.phase.index){
  phase=view.phase.index;get('phase-count').textContent=String(phase+1).padStart(2,'0')+' / '+timeline.phases.length;
  get('runtime-status').textContent=f.title;get('step-explanation').textContent=f.explanation;
  get('step-source').href=M.sourceURL(f.source);get('step-source').textContent=M.sources[f.source][2]+' ↗';
 }
 get('step-caption').textContent=view.active.map(e=>e.payload).join('；')||(view.ended?'本次路径结束；可拖动时间回看任意交接。':f.payload);
 get('front-state').textContent=s.frontFinished?'已清理等待状态':s.front;
 get('backend-state').textContent=s.scheduler;
 get('resource-state').textContent=`请求行：${s.slot?'R1 占用':'空闲'} · 请求 KV ${s.kv} · 缓存 KV ${s.cached}`;
 get('step-fields').textContent=`${s.batch}；dispatched=${s.dispatched}，abort_sent=${s.abortSent}，to_finish=${s.toFinish?'abort':'无'}，finished=${s.finished}。`;
 Object.assign(get('runtime-snapshot').dataset,{kv:s.kv,cached:s.cached,sampled:s.sampled,visible:s.visible});
 const activeNodes=new Set(view.active.flatMap(e=>e.nodes));
 for(const [id,n]of Object.entries(nodeEls)){n.g.classList.toggle('is-active',activeNodes.has(id));n.bar.setAttribute('width',0);}
 for(const [id,e]of Object.entries(edgeEls)){e.g.classList.toggle('is-active',view.active.some(a=>a.edge===id));e.g.style.display=id==='abort-reply'&&timeline.options.scenario!=='queued-abort'?'none':'';e.g.classList.toggle('is-cancel',id==='dispatch'&&f.id==='cancel-sent');e.path.setAttribute('marker-end','url(#rf-arrow-'+(id==='dispatch'&&f.id==='cancel-sent'?'control':F.edges[id].kind)+')');}
 nodeEls.tokenizer.body.textContent=s.frontFinished?'ReqState · 已清理':s.front==='尚未登记'?'ReqState · 尚未建立':s.dispatched?'ReqState(R1) · 等待输出':'ReqState(R1) · 已登记';
 nodeEls.req.body.textContent=s.finished?'R1 · finished_reason 已记录':s.toFinish?'R1 · to_finish=abort':s.waiting?'waiting_queue：[ R1 ]':s.slot?'running_batch：[ R1 ]':s.scheduler==='尚无 Req'?'Req · 尚未创建':s.scheduler;
 nodeEls.batch.body.textContent=s.batch==='无'?'尚未组批':s.batch==='无后续执行'?'已移出本轮执行':s.batch.split(' · ').slice(0,2).join(' · ');
 nodeEls.worker.body.textContent=s.batch.startsWith('B')?(f.id==='prefill'?'p1…p4 → logits → y1':s.forwards?'第 '+(f.actor==='worker'?f.forwards:s.forwards)+' 轮 · 模型与采样':'准备前向输入'):'等待本轮输入';
 nodeEls.result.body.textContent=s.finished?'结束原因：'+(s.abortSent?'abort':'length'):s.sampled?'已采样（尚需回程）':'等待执行结果';
 nodeEls.cache.body.textContent=s.cached?`${s.cached} 个位置 · 缓存持有`:f.id==='release'&&view.u<.8?'处理有效前缀与请求行':s.forwards&&!s.slot?'已归还 KV / 请求行':'等待请求结束';
 nodeEls.detokenizer.body.textContent=activeNodes.has('detokenizer')?'按 rid=R1 转换并回传':s.visible?'已有 '+s.visible+' 段回程':'等待输出 ID';
 nodeEls.memory.sub.textContent='请求行：'+(s.slot?'R1':'空闲')+' · KV '+(view.active.some(e=>e.edge==='kv')?'写入中':s.cached?'缓存持有':s.kv?'请求使用':'未占用');
 const tokenizing=f.id==='tokenize',prepared=phase>1||tokenizing;
 inputCells.forEach((el,i)=>el.classList.toggle('filled',prepared&&(!tokenizing||view.u>=(i+1)*.19)));
 kvCells.forEach((el,i)=>{el.classList.toggle('filled',i<s.kv+s.cached);el.classList.toggle('cached',i<s.cached);el.classList.toggle('writing',view.active.some(e=>e.edge==='kv')&&i<(f.kv||0)&&i>=view.phase.previous.kv);});
 sampleCells.forEach((el,i)=>el.classList.toggle('filled',i<s.sampled));outCells.forEach((el,i)=>el.classList.toggle('filled',i<s.visible));
 batchCells.forEach((el,i)=>{const prefill=s.batch.includes('EXTEND');el.style.display=prefill||i===0?'':'none';el.classList.toggle('filled',s.slot);el.querySelector('text').textContent=prefill?'p'+(i+1):(s.batch.match(/y\d+/)||['—'])[0];});
 for(const e of view.active)if(!e.edge)for(const id of e.nodes)nodeEls[id].bar.setAttribute('width',String((F.nodes[id].w-24)*e.progress));
 packets.forEach((p,i)=>{
  const e=view.active.filter(e=>e.edge)[i];if(!e||reduced.matches){p.g.setAttribute('visibility','hidden');return;}
  const path=edgeEls[e.edge],point=path.path.getPointAtLength(path.length*e.progress);
  p.g.setAttribute('visibility','visible');p.g.setAttribute('transform',`translate(${point.x} ${point.y})`);p.g.setAttribute('class','rf-packet '+e.kind);p.g.dataset.edge=e.edge;
  p.shape.setAttribute('d',palette[e.kind]);p.label.textContent=e.kind==='request'?(e.edge==='http'?'R1':'ID'):e.kind==='text'?'文':'';
 });
}
function stop(){playing=false;cancelAnimationFrame(raf);raf=0;last=0;render();}
function tick(now){if(!playing)return;if(last)time=Math.min(timeline.duration,time+(now-last)/1000*Number(get('runtime-speed').value));last=now;if(time===timeline.duration)playing=false;render();if(playing)raf=requestAnimationFrame(tick);}
function rebuild(){playing=false;cancelAnimationFrame(raf);last=0;time=0;phase=-1;timeline=F.build(options());get('outputs').disabled=get('scenario').value!=='normal';get('step').max=timeline.duration;render();}
get('runtime-toggle').addEventListener('click',()=>{if(playing){stop();return;}if(time===timeline.duration)time=0;playing=true;last=0;render();raf=requestAnimationFrame(tick);});
get('step').addEventListener('input',()=>{time=Number(get('step').value);stop();});
get('runtime-speed').addEventListener('change',()=>{last=0;});
get('previous').addEventListener('click',()=>{const p=F.at(timeline,time).phase;time=time>p.start+.3?p.start:timeline.phases[Math.max(0,p.index-1)].start;stop();});
get('next').addEventListener('click',()=>{const p=F.at(timeline,time).phase;time=timeline.phases[p.index+1]?.start??timeline.duration;stop();});
get('reset').addEventListener('click',rebuild);['scenario','outputs','cache'].forEach(id=>get(id).addEventListener('change',rebuild));
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});window.addEventListener('pagehide',stop);reduced.addEventListener('change',render);
host.hidden=false;rebuild();
})();
