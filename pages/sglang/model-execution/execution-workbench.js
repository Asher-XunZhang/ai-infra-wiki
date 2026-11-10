/* Data stays identifiable across host copies, device copies and execution.
   All positions and contents are derived from the shared teaching clock. */
(()=>{'use strict';
const NS='http://www.w3.org/2000/svg',clamp=x=>Math.max(0,Math.min(1,x));
function batch(origin){return origin==='prefill'?[
 {r:1,id:11,pos:0,slot:2},{r:1,id:21,pos:1,slot:7},{r:1,id:41,pos:2,slot:11},
 {r:2,id:32,pos:0,slot:4},{r:2,id:52,pos:1,slot:13}
]:[{r:1,id:41,pos:3,slot:11},{r:2,id:52,pos:2,slot:13}];}
function create(svg,timeline){
 const make=(tag,attrs={},value)=>{const e=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(value!==undefined)e.textContent=value;return e;};
 let root,v,reduced;
 const add=(tag,a,t)=>{const e=make(tag,a,t);root.append(e);return e;};
 const text=(x,y,s,cls='ex-text',a={})=>add('text',{x,y,class:cls,...a},s);
 const rect=(x,y,width,height,cls='ex-box',a={})=>add('rect',{x,y,width,height,rx:7,class:cls,...a});
 const line=(points,cls='ex-copy',a={})=>add('path',{d:points.map((p,i)=>(i?'L':'M')+p.join(' ')).join(' '),class:cls,...a});
 function chip(x,y,label,r=0,a={}){const g=add('g',{transform:`translate(${x} ${y})`,class:`ex-chip ex-r${r}`,...a});g.append(make('rect',{width:32,height:24,rx:4}),make('text',{x:16,y:17,'text-anchor':'middle'},label));return g;}
 function vector(x,y,r,kind='h',a={}){const g=add('g',{transform:`translate(${x} ${y})`,class:`ex-vector ex-r${r}`,...a});for(let i=0;i<4;i++)g.append(make('rect',{x:i*6,y:0,width:4,height:18,rx:1,opacity:[.45,.85,.6,1][i]}));g.append(make('text',{x:29,y:14},kind));return g;}
 function fly(label,a,b,u,r=0,attrs={}){const t=reduced?(u>=1?1:0):clamp(u);return chip(a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,label,r,{'data-moving':'true',...attrs});}
 function travel(label,points,u,r=0,attrs={}){
  const path=line(points,'ex-copy',{'marker-end':'url(#ex-arrow-copy)','data-payload':label,...attrs});
  if(u>0&&u<1){path.style.strokeDasharray='4 7';path.style.strokeDashoffset=String(reduced?0:-v.time*22/.9);path.dataset.flowActive='true';}return path;
 }
 function progress(id,a=0,b=1){const p=timeline.phases.find(p=>p.id===id);return p?clamp((v.time-p.start-a*p.duration)/((b-a)*p.duration)):0;}
 const running=id=>v.phase.id===id;
 const done=id=>progress(id)===1;
 const alpha=(x,y,w,p)=>rect(x,y,w*clamp(p),3,'ex-progress',{rx:0});
 function panel(x,y,w,h,title,sub,cls=''){rect(x,y,w,h,'ex-panel '+cls);text(x+16,y+26,title,'ex-title');if(sub)text(x+16,y+47,sub,'ex-muted');}
 function begin(title){svg.setAttribute('viewBox','0 0 1040 610');svg.dataset.surface=timeline.view;svg.replaceChildren(make('title',{id:'mf-svg-title'},title));root=svg;
  const defs=add('defs',{});for(const kind of ['copy','control','index']){const m=make('marker',{id:'ex-arrow-'+kind,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto'});m.append(make('path',{d:'M0 0L10 5L0 10Z',class:'ex-arrow-'+kind}));defs.append(m);}
 }
 function chain(){
  const pre=timeline.options.origin==='prefill',data=batch(timeline.options.origin),s=v.state;
  begin('CPU 与 GPU：数据拷贝、张量引用、任务提交和结果返回');
  panel(16,16,286,574,'CPU · 调度与调用','Scheduler → Worker → ModelRunner');
  panel(366,16,658,574,'GPU · 显存与计算','输入、权重与 KV 在显存；算子在 GPU 执行','ex-device');
  text(314,79,'边界','ex-muted');
  // Original request information stays on the CPU. Moving chips are copies.
  for(const r of [1,2]){const y=90+(r-1)*82,items=data.filter(d=>d.r===r);rect(32,y,254,72,'ex-request ex-r'+r);text(44,y+20,'R'+r+(pre?' · 本次新输入':' · 再生成一个位置'),'ex-label');
   items.forEach((d,j)=>chip(90+j*46,y+32,d.id,r,{'data-origin-token':d.id}));
  }
  text(32,278,pre?'按请求顺序打包 · CPU pinned staging':'本轮输入来自显存中的上一轮输出','ex-muted');
  if(pre)data.forEach((d,i)=>{const source=[90+(i-(d.r===2?3:0))*46,122+(d.r-1)*82],target=[38+i*48,290];const u=progress('pack',.12,.90);if(u>0)fly(d.id,source,target,u,d.r,{'data-staging':i});});
  else text(40,308,'不把这些 ID 再上传一次','ex-text');
  rect(32,336,254,88,'ex-descriptor',{'data-object':'forward-batch'});text(44,356,'ForwardBatch · 字段与引用','ex-label');
  text(44,377,'input_ids / positions','ex-code');text(44,394,'seq_lens / out_cache_loc','ex-code');text(44,413,s.bound?'指向设备张量 · 不是 H2D 拷贝':'等待输入就绪后建立执行视图','ex-muted');
  if(progress('bind')>0)line([[286,367],[334,367],[334,232],[498,232],[498,128],[504,128]],'ex-reference',{'marker-end':'url(#ex-arrow-control)','data-transfer':'reference'});
  text(32,439,'CPU 提交算子调用','ex-label');text(32,460,'携带地址与参数，触发设备工作','ex-muted');
  text(32,526,'Scheduler 更新请求','ex-label');
  if(s.delivered){chip(48,539,'y₁',1,{'data-cpu-result':1});chip(150,539,'y₂',2,{'data-cpu-result':2});text(87,556,'R1','ex-muted');text(189,556,'R2','ex-muted');}else text(32,554,'等待下一 token ID','ex-muted');
  const left=504,right=820,step=35;
  text(390,84,'当前批次张量','ex-label');text(708,84,'执行缓冲区 · 地址复用','ex-label');
  for(const [key,label,y]of [['id','input_ids',116],['pos','positions',157],['slot','写槽索引',198]]){
   text(390,y+17,label,'ex-code');
   for(let i=0;i<data.length;i++){
    rect(left+i*step,y,32,24,'ex-empty');rect(right+i*step,y,32,24,'ex-empty');
    const ready=key==='id'?(s.inputReady||pre&&progress('resolve',.04+i*.06,.70+i*.06)===1):s.bound;
    if(ready)chip(left+i*step,y,data[i][key],data[i].r,{'data-device-live':key,'data-row':i});
    if(s.loaded)chip(right+i*step,y,data[i][key],data[i].r,{'data-device-buffer':key,'data-row':i});
    if(running('load')){const u=progress('load',.08,.94);if(u>0&&u<1)travel(data[i][key],[[left+i*step+16,y+12],[right+i*step+16,y+12]],u,data[i].r,{'data-transfer':'d2d'});}
   }
  }
  if(pre&&running('resolve')){
   text(307,285,'H2D','ex-label');
   data.forEach((d,i)=>{const u=progress('resolve',.04+i*.06,.70+i*.06);if(u>0&&u<1)travel(d.id,[[54+i*48,302],[334,288],[690,288],[690,128],[left+i*step+16,128]],u,d.r,{'data-transfer':'h2d'});});
  }else if(!pre&&!s.inputReady){
   text(405,250,'上轮输出 · 已在显存','ex-muted');data.forEach((d,i)=>{const u=progress('resolve',.05,.94);travel(d.id,[[596+i*step,244],[690,244],[690,128],[left+i*step+16,128]],u,d.r,{'data-transfer':'gpu-reuse'});});
  }
  if(running('load'))text(629,251,'D2D · 显存内更新','ex-muted');
  if(s.bound&&!running('load'))text(390,251,pre?'positions = [0,1,2,0,1]；R1 长度 3，R2 长度 2':'positions = [3,2]；R1 长度 4，R2 长度 3','ex-muted');
  text(390,312,'本层 MLA 缓存 · 物理槽位置固定','ex-label');
  const history=pre?[]:[{r:1,slot:2,pos:0},{r:1,slot:7,pos:1},{r:1,slot:9,pos:2},{r:2,slot:4,pos:0},{r:2,slot:8,pos:1}];
  for(let i=0;i<14;i++){
   const x=390+i*43,old=history.find(h=>h.slot===i),fresh=data.find(d=>d.slot===i),entry=old||(s.stored?fresh:null);
   text(x+16,336,String(i),'ex-address',{'text-anchor':'middle'});rect(x,344,34,38,'ex-slot'+(entry?' ex-r'+entry.r+' ex-filled':''),{'data-slot':i,'data-filled':String(!!entry)});
   if(entry){text(x+17,359,'R'+entry.r,'ex-cell-label',{'text-anchor':'middle'});text(x+17,375,'p'+entry.pos,'ex-cell-label',{'text-anchor':'middle'});}
   if(fresh&&s.metadata)rect(x-2,342,38,42,'ex-write-target',{'data-write-target':i});
  }
  if(running('metadata')){data.forEach((d,i)=>line([[right+i*step+16,222],[right+i*step+16,319],[407+d.slot*43,319],[407+d.slot*43,330]],'ex-index',{'data-mapping':d.slot}));}
  if(!['compute','project','launch'].includes(v.phase.id))text(390,407,prefillCaption(pre,s),'ex-muted');
  if(running('launch')){const u=progress('launch',.05,.94);line([[270,437],[335,437],[335,395],[676,395],[676,420]],'ex-control',{'marker-end':'url(#ex-arrow-control)','data-transfer':'submit'});if(u>0&&u<1){rect(338+290*u,387,28,15,'ex-command',{'data-command':'launch'});}text(390,408,'提交计算 ≠ 搬运权重 / KV','ex-muted');}
  // Resident weights do not travel across the CPU/GPU boundary.
  rect(832,424,168,68,'ex-resident',{'data-resident':'weights'});text(844,445,'常驻模型权重 W','ex-label');for(let i=0;i<5;i++)rect(845+i*25,457,19,17,'ex-weight');
  rect(590,424,216,68,'ex-compute'+(['compute','project'].includes(v.phase.id)?' is-working':''),{'data-compute':'attention'});text(606,448,'GPU 算子执行','ex-label');text(606,471,running('project')?'Embedding / 层内投影':'当前层 Attention','ex-muted');
  if(s.projected){vector(396,433,1,pre?'Q末':'Q₁');vector(478,433,2,pre?'Q末':'Q₂');}
  if(running('project')){
   data.forEach((d,i)=>{const u=progress('project',.04+i*.07,.66+i*.07);if(u>0&&u<1)travel(d.id,[[right+i*step+16,128],[1006,128],[1006,401],[690,401],[690,424]],u,d.r,{'data-transfer':'device-read'});});
   line([[832,454],[806,454]],'ex-copy',{'marker-end':'url(#ex-arrow-copy)'});alpha(606,483,182,progress('project'));
  }
  if(running('compute')){
   const write=progress('compute',.04,.38),read=progress('compute',.42,.90);
   if(write>0&&write<1)data.forEach((d,i)=>{const a=[610+i*31,463],b=[391+d.slot*43,351],u=reduced?0:write;vector(a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u,d.r,'',{'data-transfer':'kv-write','data-row':i});});
   if(read>0&&read<1)[...history,...data].forEach(d=>{const a=[391+d.slot*43,351],b=[607+(d.r-1)*90,458],u=reduced?0:read;vector(a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u,d.r,'',{'data-transfer':'kv-read','data-request':d.r});});
   alpha(606,483,182,progress('compute'));
  }
  text(390,518,s.logits?'logits · 每请求一行词表分数':'模型其余层与输出头继续执行','ex-muted');
  if(s.logits){vector(400,532,1,'分数');vector(490,532,2,'分数');}
  if(running('result')){line([[560,541],[814,541]],'ex-copy',{'marker-end':'url(#ex-arrow-copy)'});text(640,528,'采样','ex-muted');}
  if(s.sampled){chip(830,530,'y₁',1,{'data-gpu-result':1});chip(930,530,'y₂',2,{'data-gpu-result':2});}
  if(running('return')||s.delivered){text(432,583,'D2H · 只返回本例需要的下一 token ID','ex-muted');if(!s.delivered)for(const r of [1,2])travel(r===1?'y₁':'y₂',[[846+(r-1)*100,542],[846+(r-1)*100,570],[64+(r-1)*102,570],[64+(r-1)*102,551]],progress('return',.05,.94),r,{'data-transfer':'d2h'});}
  return `R1 / R2 身份始终保留 · ${pre?'Prefill：5 个新位置':'Decode：2 个新位置'} · ${s.delivered?'结果已返回 CPU':'权重和历史 KV 留在 GPU'}`;
 }
 function prefillCaption(pre,s){return pre?'Prefill：图中聚焦每请求最后位置的查询；其他位置仍按因果范围读取':'Decode：R1 读 [2,7,9,11]；R2 读 [4,8,13]'+(s.stored?' · 本轮已写入':' · 新槽待写入');}
 function graph(){
  const s=v.state,d=window.ExecutionModel.graph(timeline.options.count,timeline.options.condition,4),count=d.real.length,capacity=d.eligible?d.bucket:count;
  begin('CUDA Graph：固定显存缓冲区、更新输入、提交重放、裁剪补齐行');
  panel(16,16,280,574,'CPU · 选择执行方式','ModelRunner / can_run_graph');panel(330,16,694,574,'GPU · 本轮数据仍要重新计算','Decode：每请求一个新位置；当前输入仍需重新计算','ex-device');
  text(34,97,`本轮 ${count} 条真实请求`,'ex-label');for(let i=0;i<count;i++)chip(36+i*45,111,'R'+(i+1),i===0?1:i===1?2:0);
  text(34,177,'已捕获容量','ex-muted');[1,2,4].forEach((b,i)=>{rect(34+i*78,192,64,42,'ex-bucket'+(d.eligible&&b===d.bucket&&progress('gate')>.5?' is-selected':''));text(66+i*78,220,String(b),'ex-title',{'text-anchor':'middle'});});
  text(34,267,d.eligible?`${count} 条 → 容量 ${capacity}`:'本轮使用 Eager','ex-label');text(34,290,d.reason,'ex-muted');
  text(34,352,d.eligible?'提交 graph replay':'逐次发起普通算子调用','ex-label');
  const opNames=['Embedding','Attention','MLP / MoE','输出头'];
  if(d.eligible){rect(34,369,238,45,'ex-command',{'data-submit-mode':'replay'});text(48,397,'replay(选中的图桶)','ex-code');}
  else opNames.forEach((name,i)=>{rect(34,366+i*36,238,28,'ex-command',{'data-submit-mode':'eager'});text(48,386+i*36,name,'ex-code');});
  text(34,548,'只画代表操作与提交关系','ex-muted');text(34,570,'时长、数量不是性能测量','ex-muted');
  text(350,97,'当前输入 · 已在显存','ex-label');text(688,97,d.eligible?'固定输入地址 · 内容每轮更新':'执行缓冲区 · 本轮真实行','ex-label');
  const width=43;
  for(let i=0;i<count;i++)chip(350+i*width,121,41+i*11,i===0?1:i===1?2:0,{'data-graph-input':i});
  for(let i=0;i<capacity;i++){
   rect(688+i*width,121,32,34,i>=count?'ex-pad':'ex-empty',{'data-buffer-slot':i});text(704+i*width,176,'S'+i,'ex-address',{'text-anchor':'middle'});
   if(s.padded&&i<count)chip(688+i*width,126,41+i*11,i===0?1:i===1?2:0,{'data-graph-loaded':i});
   if(s.padded&&i>=count)text(704+i*width,143,'补','ex-muted',{'text-anchor':'middle','data-pad':i});
  }
  line([[584,138],[678,138]],'ex-copy',{'marker-end':'url(#ex-arrow-copy)'});text(591,124,'D2D','ex-muted');
  if(running('load'))for(let i=0;i<count;i++){const u=progress('load',.03,.75);if(u<1)fly(41+i*11,[350+i*width,121],[688+i*width,126],u,i===0?1:i===1?2:0,{'data-transfer':'graph-input'});}
  text(350,214,d.eligible?`地址 S0…S${capacity-1} 保持不变；本轮更新真实输入，${capacity-count} 行仅用于补齐`:'不满足图条件也会执行 GPU 计算；保持全部真实输入','ex-muted');
  rect(350,242,652,137,'ex-program',{'data-program':d.path});text(366,268,d.eligible?'已捕获的设备工作 · 本轮重放':'普通前向 · 本轮执行','ex-label');
  opNames.forEach((name,i)=>{const u=progress('execute',.12,.94),active=running('execute')&&Math.floor(u*4)===i;rect(368+i*157,289,141,50,'ex-operation'+(active?' is-working':''));text(438+i*157,319,name,'ex-label',{'text-anchor':'middle'});if(active)alpha(375+i*157,343,127,(u*4)%1);});
  text(366,368,'代表性计算；同一份权重与历史 KV 仍在显存','ex-muted');
  if(running('execute')){line([[275,390],[310,390],[310,275],[350,275]],'ex-control',{'marker-end':'url(#ex-arrow-control)','data-transfer':'submit'});const u=progress('execute',.03,.35);if(u<1)rect(300,380-u*105,18,12,'ex-command',{'data-command':d.path});}
  text(350,416,'原始输出 · GPU logits','ex-label');text(688,416,'交给采样 · 只保留真实行','ex-label');
  for(let i=0;i<capacity;i++){
   const y=434+i*27;rect(350,y,235,22,i>=count?'ex-pad':'ex-output-row',{'data-raw-row':i,'data-visible':String(s.raw)});
   if(s.raw){text(361,y+16,i<count?'R'+(i+1)+' · 本轮词表分数':'补齐行 · 不是请求','ex-code');if(i>=count){line([[350,y],[585,y+22]],'ex-discard');}}
   if(i<count){rect(688,y,285,22,'ex-empty');if(s.trimmed)text(699,y+16,'R'+(i+1)+' · 本轮词表分数','ex-code',{'data-valid-output':i});
    if(running('trim')){const u=progress('trim',.04,.93);if(u<1){const x=350+(688-350)*(reduced?0:u);rect(x,y,235,22,'ex-output-row',{'data-moving-row':i});text(x+11,y+16,'R'+(i+1)+' · 本轮词表分数','ex-code');}}
   }
  }
  if(running('trim')||s.trimmed){text(600,452,'裁回','ex-muted');text(600,474,String(count)+' 行','ex-label');}
  return `真实请求 ${count} · ${d.eligible?'图桶 '+capacity+'，补齐 '+(capacity-count)+' 行':'Eager，无补齐'} · ${s.trimmed?'仅 '+count+' 行 logits 交给采样':'GPU 仍需计算当前输入'}；此处尚未回传 CPU`;
 }
 return {render(snapshot,isReduced){v=snapshot;reduced=isReduced;return timeline.view==='chain'?chain():graph();}};
}
window.ExecutionWorkbench={create,batch};
})();
