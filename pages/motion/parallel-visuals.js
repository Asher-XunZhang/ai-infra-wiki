/* Each parallel scheme visualizes the object it partitions. */
(()=>{'use strict';
const {board,phase,lerp}=window.TeachingDrawing,V=window.MechanismVisuals,pp=V['f-parallel'];
// PCP has its own integrated canvas (parallel-pcp.js): the single-card
// baseline, both two-card splits, the K/V gate and output ownership share one
// board, so there is no second adoption panel above it.
/* PP: one integrated board. The upper lane is the single-GPU baseline, the
   lower two lanes are the two-stage schedule on the same layer-slot ruler. */
function pipelineVisual(svg,viewport,s){
 const film=viewport.closest('.mechanism-film');if(film)film.classList.add('is-pp');
 const inner=board(svg,viewport,'pp-pipeline',()=>560,H=>{
  const {txt,text,rect,path,box,show,token,move,set,node}=H;
  const P=window.ParallelTeachingModel,d=P&&P.pipeline?P.pipeline(Number(s.data?.n)||2):null;
  const n=d?d.n:2;
  const W=720,startX=128,cellW=68,gap=4,cellH=56;
  const cellX=i=>startX+i*(cellW+gap);
  const panel=(x,y,w,h,cls='teach-panel')=>{const r=rect(cls);box(r,x,y,w,h);return r;};
  const label=(x,y,a,b)=>{const t1=txt(a,'pp-lane-strong'),t2=txt(b,'pp-lane-small');text(t1,x,y);text(t2,x,y+15);return[t1,t2];};
  const title=txt('PP · 同一四层模型：单卡与两 stage 对照','teach-heading');
  const sub=txt(n===2?'输入 MB1={R1,R2}、MB2={R3,R4}；1 格 = 1 层计算，两侧同一刻度':'输入 MB1={R1,R2}；只有 1 个 MB，1 格 = 1 层计算', 'pp-sub');
  const unit=txt('等长层为教学假设 · 未计传输延迟','pp-note');
  const tabs=[1,2].map(mb=>{
   const g=node('g',{class:'pp-tab','data-pp-tab':String(mb),role:'button',tabindex:'0','aria-label':mb===1?'切换到一个微批次':'切换到两个微批次','aria-pressed':'false'});
   const r=node('rect',{rx:7}),t=node('text',{'text-anchor':'middle'});
   t.textContent=mb===1?'1 个 MB':'2 个 MB';g.append(r,t);
   const activate=()=>svg.dispatchEvent(new CustomEvent('parallel-pp-mb',{bubbles:true,detail:{mb}}));
   g.addEventListener('click',activate);
   g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.stopPropagation();activate();}});
   return {g,r,t,mb};
  });
  const singlePanel=panel(12,58,696,150),stagePanel=panel(12,220,696,248);
  const singleTitle=txt('未采用 · 单卡 1 GPU','pp-panel-title');
  const singleMeta=txt('','pp-panel-meta');
  const stageTitle=txt('采用后 · 两个 stage / 2 GPU','pp-panel-title');
  const stageMeta=txt('','pp-panel-meta');
  const singleLabel=label(22,120,'GPU 0','权重 L1–L4');
  const stageLabel0=label(22,274,'GPU 0 · P0','权重 L1–L2');
  const stageLabel1=label(22,374,'GPU 1 · P1','权重 L3–L4');
  const overlapBand=rect('pp-overlap');
  const overlapNote=txt('','pp-overlap-note');
  const handoffs=[],payloads=[],arrivals=[];
  const handoffLine=(from,to)=>{const line=path('pp-handoff-line');const x1=cellX(from)+cellW/2,x2=cellX(to)+cellW/2;set(line,{d:`M${x1} 308 C${x1} 322 ${x2} 338 ${x2} 352`});show(line,0);return {line,x1,x2};};
  if(d)for(const a of d.activation){const h=handoffLine(a.from,a.to);handoffs.push(h);const p=token('pp-payload-mb'+a.mb,'MB'+a.mb+' 激活',84,'teach-token pp-payload');set(p.r,{height:24});set(p.t,{y:17});p.g.dataset.ppPayload=String(a.mb);payloads.push({...p,...h,mb:a.mb});const ar=txt('','pp-arrival');ar.setAttribute('text-anchor','middle');ar.dataset.ppArrival=String(a.mb);arrivals.push({t:ar,mb:a.mb,x:cellX(a.to)+cellW/2,y:348});}
  const singleCells=(d?d.single:[]).map((e,i)=>{const c=cellNode('pp-single-'+i);c.entry=e;return c;});
  const stage0Cells=(d?d.stage0:[]).map((e,i)=>{const c=cellNode('pp-stage0-'+i);c.entry=e;return c;});
  const stage1Cells=(d?d.stage1:[]).map((e,i)=>{const c=cellNode('pp-stage1-'+i);c.entry=e;return c;});
  function cellNode(id){
   const g=node('g',{class:'pp-cell','data-motion-id':id});const r=node('rect',{rx:5});const num=node('text',{class:'pp-cell-num'}),main=node('text',{class:'pp-cell-main','text-anchor':'middle'}),sub=node('text',{class:'pp-cell-sub','text-anchor':'middle'});
   g.append(r,num,main,sub);return {g,r,num,main,sub};
  }
  const gate=txt('','pp-gate');gate.dataset.ppGate='mb1';
  const boundary=txt('','pp-boundary');boundary.setAttribute('text-anchor','end');
  const singleStats=txt('','pp-stats'),stageStats=txt('','pp-stats');
  const take1=txt('','pp-take'),take2=txt('','pp-take'),take3=txt('','pp-take');
  const narrowCallout=txt('','pp-callout');narrowCallout.setAttribute('text-anchor','end');
  const place=(c,i,y)=>{const x=cellX(i);box(c.r,x,y,cellW,cellH);text(c.num,x+6,y+14);text(c.main,x+cellW/2,y+24);text(c.sub,x+cellW/2,y+42);};
  function setCell(c,index,visible,focus,kindTag){
   if(!c.entry||index>=visible){show(c.g,0);c.g.dataset.ppState='hidden';c.g.dataset.ppMb='';c.g.removeAttribute('data-pp-layer');return;}
   const e=c.entry,active=index===focus;
   const state=e.kind==='work'?(active?'active':'complete'):e.kind==='wait'?(active?'waiting':'waited'):(active?'draining':'drained');
   const cls=['pp-cell'];if(e.kind==='work')cls.push('is-work','is-mb'+e.mb);else cls.push(e.kind==='wait'?'is-wait':'is-drain');if(active)cls.push('is-active');
   place(c,index,kindTag==='single'?96:kindTag==='stage0'?252:352);
   set(c.g,{class:cls.join(' '),'aria-label':ppCellLabel(e,index,state)});
   c.g.dataset.ppState=state;c.g.dataset.ppSlot=kindTag+'-'+index;
   if(e.kind==='work'){c.g.dataset.ppMb=String(e.mb);c.g.dataset.ppLayer=String(e.layer);c.num.textContent=String(index+1);c.main.textContent='MB'+e.mb;c.sub.textContent='L'+e.layer;}
   else if(e.kind==='wait'){c.g.dataset.ppMb='';c.num.textContent=String(index+1);c.main.textContent='等';c.sub.textContent=active?'激活':'已等待';}
   else {c.g.dataset.ppMb='';c.num.textContent=String(index+1);c.main.textContent='排空';c.sub.textContent=active?'无新 MB':'空闲';}
   show(c.g,1);
  }
  function ppCellLabel(e,index,state){
   const now=state==='active'||state==='waiting'||state==='draining'?'（进行中）':'（已完成）';
   if(e.kind==='work')return `第 ${index+1} 格，MB${e.mb} 第 ${e.layer} 层${now}`;
   if(e.kind==='wait')return `第 ${index+1} 格，等待激活${now}`;
   return `第 ${index+1} 格，排空${now}`;
  }
  function viewAt(p){
   if(!d)return {vs:0,fs:-1,va:0,f0:-1,f1:-1,h1:0,h2:0};
   if(n===1){
    if(p<1)return {vs:0,fs:-1,va:0,f0:-1,f1:-1,h1:0,h2:0};
    if(p<2)return {vs:1,fs:0,va:1,f0:0,f1:0,h1:0,h2:0};
    if(p<3)return {vs:2,fs:1,va:2,f0:1,f1:1,h1:0,h2:0};
    if(p<4)return {vs:2,fs:-1,va:2,f0:-1,f1:-1,h1:phase(p,3,4),h2:0};
    if(p<5)return {vs:3,fs:2,va:3,f0:2,f1:2,h1:1,h2:0};
    if(p<6)return {vs:4,fs:3,va:4,f0:3,f1:3,h1:1,h2:0};
    return {vs:4,fs:-1,va:4,f0:-1,f1:-1,h1:1,h2:0};
   }
   if(p<1)return {vs:0,fs:-1,va:0,f0:-1,f1:-1,h1:0,h2:0};
   if(p<2)return {vs:1,fs:0,va:1,f0:0,f1:0,h1:0,h2:0};
   if(p<3)return {vs:2,fs:1,va:2,f0:1,f1:1,h1:0,h2:0};
   if(p<4)return {vs:2,fs:-1,va:2,f0:-1,f1:-1,h1:phase(p,3,4),h2:0};
   if(p<5)return {vs:3,fs:2,va:3,f0:2,f1:2,h1:1,h2:0};
   if(p<6){const u=phase(p,5.35,6);return {vs:4,fs:3,va:4,f0:u>0?-1:3,f1:3,h1:1,h2:u};}
   if(p<7)return {vs:5,fs:4,va:5,f0:4,f1:4,h1:1,h2:1};
   if(p<8)return {vs:6,fs:5,va:6,f0:5,f1:5,h1:1,h2:1};
   if(p<9)return {vs:7,fs:6,va:6,f0:-1,f1:-1,h1:1,h2:1};
   if(p<10)return {vs:8,fs:7,va:6,f0:-1,f1:-1,h1:1,h2:1};
   return {vs:8,fs:-1,va:6,f0:-1,f1:-1,h1:1,h2:1};
  }
  const statusFor=p=>{
   if(n===1){
    if(p<1)return '准备：只有一个 MB；单卡与两 stage 使用同一四层模型和同一格刻度。';
    if(p<2)return '第 1 格：GPU0 算 MB1 L1；GPU1 没有激活，先等待。单卡也开始 MB1 L1。';
    if(p<3)return '第 2 格：MB1 L2 在 GPU0 计算；尚未完成，不能发出激活。';
    if(p<4)return '交接：MB1 L2 已完成，激活在途；传输不另计格，GPU1 的 L3 仍不能执行。';
    if(p<5)return '第 3 格：MB1 激活到达后 GPU1 算 L3；GPU0 没有第二个 MB，开始排空。';
    if(p<6)return '第 4 格：GPU1 算 L4；GPU0 继续排空。两侧都在第 4 格结束。';
    return '总账：4 层格工作；两侧同 4 格；8 GPU·格中 4 格气泡，没有第二 MB 可补位，故无流水收益。';
   }
   if(p<1)return '准备：同一四层模型；单卡顺序执行，两 stage 各驻留两层权重。';
   if(p<2)return '第 1 格：GPU0 算 MB1 L1；GPU1 没有激活，先等待。单卡也开始 MB1 L1。';
   if(p<3)return '第 2 格：MB1 L2 在 GPU0 计算；尚未完成，不能发出激活。';
   if(p<4)return '交接：MB1 L2 已完成，激活在途；传输不另计格，GPU1 的 L3 仍不能执行。';
   if(p<5)return '第 3 格：MB1 激活到达后 GPU1 才算 L3；GPU0 同期算 MB2 L1，这是不同 MB 的重叠。';
   if(p<6)return '第 4 格：MB1 L4 与 MB2 L2 在不同 MB 上重叠；MB2 L2 完成后才发激活。';
   if(p<7)return '第 5 格：MB2 激活到达后 GPU1 算 L3；GPU0 没有新 MB，进入排空。';
   if(p<8)return '第 6 格：GPU1 算 MB2 L4；这是两 stage 的最后一格，单卡当前只到 6/8 格。';
   if(p<9)return '第 7 格：两 stage 已在第 6 格完成；单卡继续 MB2 L3，当前 7/8 格。';
   if(p<10)return '第 8 格：单卡算 MB2 L4；两 stage 保持完成，不再增加 GPU 格。';
   return '总账：8 层格工作；单卡 8 格、两 stage 6 格；12 GPU·格中 4 格气泡。';
  };
  const gateFor=(p,v)=>{
   if(n===1){
    if(p<1)return '门：等待 MB1 进入 L1';
    if(p<2)return '门：MB1 L1 执行 · L3 等待';
    if(p<3)return '门：MB1 L2 计算中 · L3 等待';
    if(p<4)return v.h1>0?'门：MB1 激活在途 · L3 等待':'门：MB1 L2 完成 · 准备发激活';
    if(p<5)return '门：MB1 到达 · L3 执行';
    if(p<6)return '门：MB1 到达 · L4 执行';
    return '门：MB1 完成 · 无第二 MB';
   }
   if(p<1)return '门：等待 MB1 进入 L1';
   if(p<2)return '门：MB1 L1 执行 · L3 等待';
   if(p<3)return '门：MB1 L2 计算中 · L3 等待';
   if(p<4)return v.h1>0?'门：MB1 激活在途 · L3 等待':'门：MB1 L2 完成 · 准备发激活';
   if(p<5)return '门：MB1 到达 · L3 执行';
   if(p<6)return v.h2>0?'门：MB2 激活在途 · L3 等待':'门：MB2 L2 计算中 · 等待';
   if(p<7)return '门：MB2 到达 · L3 执行';
   if(p<8)return '门：MB2 到达 · L4 执行';
   return '门：MB2 已到 · 两 stage 完成';
  };
  return (p,m)=>{
   svg.setAttribute('viewBox','0 0 720 560');svg.dataset.ppMb=String(n);svg.dataset.ppPlacement=svg.dataset.placement||'same-host';
   const v=viewAt(p),placement=svg.dataset.placement==='cross-host'?'cross-host':'same-host';
   text(title,18,22);text(sub,18,42);text(unit,18,526);
   tabs.forEach(t=>{const active=t.mb===n;set(t.g,{class:'pp-tab'+(active?' is-active':''),'aria-pressed':String(active)});box(t.r,0,0,72,24);text(t.t,36,16);t.g.setAttribute('transform','translate('+(500+(t.mb-1)*86)+' 10)');});
   text(singleTitle,22,80);text(stageTitle,22,242);
   const span=d?d.span:0,work=d?d.work:0,gpu=d?d.adoptedGpuSlots:0,bub=d?d.bubbles:0;
   const singleDone=n===1?p>=6:p>=10,stageDone=n===1?p>=6:p>=8,handoff1=p>=3&&p<4;
   const singleText=p<1?'等待开始':handoff1?'L2 完成 · 对照暂停':singleDone?`已完成 · ${4*n} / ${4*n} 格`:`进行中 · 第 ${v.vs} / ${4*n} 格`;
   const stageText=p<1?'等待开始':handoff1?'交接 · L2 完成':stageDone?(n===2&&p<10?'已完成 · 第 6 格 · 单卡继续':`已完成 · ${span} / ${span} 格`):`进行中 · 第 ${v.va} / ${span} 格`;
   text(singleMeta,700,80,singleText);singleMeta.setAttribute('text-anchor','end');
   text(stageMeta,700,242,stageText);stageMeta.setAttribute('text-anchor','end');
   set(singleMeta,{class:'pp-panel-meta'+(singleDone?' is-done':'')});set(stageMeta,{class:'pp-panel-meta'+(stageDone?' is-done':'')});
   singleLabel.forEach((t,i)=>text(t,22,120+i*15));
   stageLabel0.forEach((t,i)=>text(t,22,274+i*15));stageLabel1.forEach((t,i)=>text(t,22,374+i*15));
   if(n===2){const overlap=p>=4&&p<8;box(overlapBand,cellX(2)-2,252,cellW*2+gap+4,156);show(overlapBand,overlap?.55:0);text(overlapNote,cellX(2)+4,214,overlap?'重叠区：第 3–4 格 · GPU0 MB2 + GPU1 MB1（不同 MB）':'')}else{show(overlapBand,0);text(overlapNote,cellX(2)+4,214,'只有一个 MB：第 3–4 格没有另一份工作可重叠');}
   singleCells.forEach((c,i)=>setCell(c,i,v.vs,v.fs,'single'));
   stage0Cells.forEach((c,i)=>setCell(c,i,v.va,v.f0,'stage0'));
   stage1Cells.forEach((c,i)=>setCell(c,i,v.va,v.f1,'stage1'));
   handoffs.forEach((h,i)=>{const start=i===0?3:5.35;show(h.line,phase(p,start,start+.2));});
   payloads.forEach(pl=>{const u=pl.mb===1?v.h1:v.h2;const inFlight=u>0&&u<1;move(pl,u>0?lerp(pl.x1-42,pl.x2-42,u):pl.x1-42,u>0?lerp(310,324,u):310,inFlight?1:0);pl.g.dataset.ppTransfer=u.toFixed(3);});
   arrivals.forEach(a=>{text(a.t,a.x,a.y,'激活到');show(a.t,p>=(a.mb===1?4:6)?1:0);});
   text(gate,22,334,gateFor(p,v));
   gate.dataset.ppGate=n===1?'1':(p<5?'1':p<7?'2':'done');
   const ready=n===1?p>=4:(p<5?v.h1>=1:p<7?v.h2>=1:true);
   const inFlight=n===1?(v.h1>0&&v.h1<1):(p<5?(v.h1>0&&v.h1<1):p<7?(v.h2>0&&v.h2<1):false);
   gate.dataset.ppReady=String(ready);gate.dataset.ppInflight=String(inFlight);
   text(boundary,700,334,placement==='cross-host'?'跨主机：主机 A → 主机 B · 跨进程/GPU':'同机：进程 P0 → P1 · GPU0 → GPU1');boundary.dataset.ppBoundary=placement;
   text(singleStats,22,178,`${4*n} 层格 · 1 GPU · 0 气泡`);
   text(stageStats,22,446,`2 GPU × ${span} 格 = ${gpu} GPU·格 · 工作 ${work} · 气泡 ${bub}`);
   text(narrowCallout,700,178,n===2?'':'没有第二份工作可补位');
   text(take1,18,486,statusFor(p));
   text(take2,18,506,'传的是同一 MB 经 L2 后的 activation；权重留在各 stage，KV 与完整请求不搬。');
   text(take3,18,548,'完成格数不是加速比；四层等长、时隙等长只是教学假设，传输延迟未另计。');
   tabs.forEach(t=>show(t.g,1));
  };
 });
 const destroy=inner.destroy;inner.destroy=()=>{film?.classList.remove('is-pp');destroy();};
 return inner;
}
window.PipelineParallelVisual=pipelineVisual;
V['f-parallel']=(svg,viewport,s)=>{
 if(s.kind==='pipeline')return pipelineVisual(svg,viewport,s);
 if(s.kind==='parallel-dp')return window.DPMechanismVisual(svg,viewport,s);
 // EP is one integrated playable canvas: the old separate adoption panel
 // repeated the same single-card/two-card comparison above the animation.
 if(s.kind==='parallel-ep')return window.ExpertParallelVisual(svg,viewport,s);
 if(s.kind==='parallel-tp')return window.TensorParallelVisual(svg,viewport,s);
 if(s.kind==='parallel-pcp')return window.PCPMechanismVisual(svg,viewport,s);
 const visual=board(svg,viewport,s.kind,m=>s.kind==='parallel-dcp'?(m?792:636):s.kind==='parallel-sp'?(m?580:620):790,H=>{
 const {txt,rect,path,text,box,show,token,move,set,wire}=H,d=s.data,kind=s.kind.slice(9);
 const header=txt('','teach-heading'),hint=txt('','teach-small'),panels=[rect(),rect()],names=[0,1].map(r=>txt('','teach-small'));
 const note=txt('','teach-small'),result=txt('','teach-heading'),line2=txt('','teach-small');
 const tile=(id,label,color=0,w=64)=>token(id,label,w,'teach-token teach-color-'+color);
 let paint;
 if(kind==='sp'){
  // SP is one integrated board: the comparison and the animated lane share
  // the same two-GPU TP group. AllGather copies rows into empty row slots and
  // leaves the owner's local copy in place; ReduceScatter carries each partial
  // along a side rail so it never crosses another position's payload.
  const film=viewport.closest('.mechanism-film');if(film)film.classList.add('is-sp');
  const NS='http://www.w3.org/2000/svg';
  const raw=(tag,attrs={},value)=>{const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(value!==undefined)e.textContent=value;return e;};
  const model=window.ParallelTeachingModel;
  const seq=model&&typeof model.sequence==='function'?model.sequence(4):{tokens:['p1','p2','p3','p4'],shards:[['p1','p2'],['p3','p4']],rowsPerRank:2,replicatedPerRank:4,replicasBefore:8,replicasAfter:4};
  const tokens=seq.tokens;
  const ownerOf=i=>{for(let r=0;r<seq.shards.length;r++)if(seq.shards[r].includes(tokens[i]))return r;return i<seq.rowsPerRank?0:1;};
  let mode='sp',lastP=0,lastM=false,draw=null;
  try{if(new URL(location.href).searchParams.get('sp')==='tp')mode='tp';}catch(error){}
  const layout=m=>m?{
   W:360,H:580,
   cards:{y:50,h:94,x:[8,184],w:168,titleY:16,badgeY:16,rowY:[38,62],footY:84,chipW:24,chipH:16,gap:2,groupGap:5,labelW:12},
   src:{x:8,y:152,w:344,h:62,labelY:14,tokenY:26,tokenW:80,tokenH:26,gap:4},
   colX:[8,100,258],colW:[88,152,94],colHeadY:224,colSubY:238,
   laneY:[246,368],laneH:116,laneHeadY:14,panelTop:20,panelH:92,rowH:18,rowGap:2,rowPad:4,
   trackW:24,dockGap:6,
   footY:[500,520,540,560],rowSize:10.5,noteSize:9.5,eqSize:8.5,headSize:10.5,chipSize:8.5
  }:{
   W:720,H:620,
   cards:{y:56,h:78,x:[18,362],w:340,titleY:18,badgeY:18,rowY:[38],footY:64,chipW:29,chipH:17,gap:2,groupGap:8,labelW:18},
   src:{x:18,y:146,w:684,h:70,labelY:18,tokenY:30,tokenW:156,tokenH:30,gap:12},
   colX:[18,214,522],colW:[180,292,180],colHeadY:230,colSubY:244,
   laneY:[252,396],laneH:136,laneHeadY:16,panelTop:22,panelH:110,rowH:22,rowGap:2,rowPad:4,
   trackW:36,dockGap:8,
   footY:[548,570,592,612],rowSize:12.5,noteSize:10.5,eqSize:9.5,headSize:12,chipSize:9.5
  };
  const pLabel=i=>tokens[i]+' [a'+(i+1)+',b'+(i+1)+']';
  const nLabel=i=>'n'+(i+1)+' [a′,b′]';
  const cLabel=(i,r)=>'c'+(i+1)+(r?'¹':'⁰')+' [u'+(r?'¹':'⁰')+',v'+(r?'¹':'⁰')+']';
  const cShort=(i,r)=>'c'+(i+1)+(r?'¹':'⁰');
  const yLabel=i=>'y'+(i+1)+' [u'+(i+1)+',v'+(i+1)+']';
  const cards=[];
  for(const key of ['tp','sp']){
   const g=raw('g',{class:'sp-mode',role:'button',tabindex:'0','aria-pressed':'false','data-sp-mode-card':key});
   const bg=raw('rect',{class:'teach-panel sp-mode-panel',rx:7});
   const title=raw('text',{class:'teach-heading sp-mode-title'});
   const badge=raw('text',{class:'teach-small sp-mode-badge','text-anchor':'end'});
   const foot=raw('text',{class:'teach-small sp-mode-foot'});
   g.append(bg,title,badge,foot);
   const labels=[0,1].map(r=>{const t=raw('text',{class:'teach-small sp-mode-glabel'});g.append(t);return t;});
   const chips=[];
   for(let r=0;r<2;r++)for(let i=0;i<4;i++){
    const cg=raw('g',{class:'teach-token sp-mode-chip'});
    const cr=raw('rect',{rx:3}),ct=raw('text',{'text-anchor':'middle'});
    cg.append(cr,ct);g.append(cg);chips.push({rank:r,row:i,g:cg,rect:cr,label:ct});
   }
   svg.append(g);cards.push({key,g,bg,title,badge,foot,labels,chips});
  }
  const setMode=next=>{if(next===mode)return;mode=next;if(draw)draw(lastP,lastM);svg.dispatchEvent(new CustomEvent('parallel-sp-mode',{bubbles:true,detail:{mode}}));};
  for(const card of cards){
   card.g.addEventListener('click',()=>setMode(card.key));
   card.g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.stopPropagation();setMode(card.key);}});
  }
  const srcPanel=rect('teach-panel sp-source'),srcLabel=txt('','teach-small sp-sub'),srcNote=txt('','teach-small sp-sub');
  const regionPanels=Array.from({length:6},()=>rect('teach-panel sp-region'));
  const colHeads=[0,1,2].map(()=>txt('','sp-panel-head')),colSubs=[0,1,2].map(()=>txt('','sp-sub'));
  const laneNames=[0,1].map(()=>txt('','sp-lane')),laneNotes=[0,1].map(()=>txt('','sp-lane-note'));
  const foot=[0,1,2,3].map(()=>txt('','sp-foot'));
  // One narrow rail at the right edge of the TP region is the only cross-lane
  // path. Payloads shrink to a labeled chip there, so they never sweep across
  // another row's full-width token. The rail itself is behind every token.
  const rail=path('sp-rail');
  const dockSlots=[];
  for(let r=0;r<2;r++)for(let i=0;i<4;i++)dockSlots.push({r,i,s:rect('sp-slot sp-dock-slot')});
  const slots=[];
  for(let r=0;r<2;r++)for(let c=0;c<3;c++)for(let i=0;i<4;i++)slots.push({r,c,i,s:rect('sp-slot')});
  const source=Array.from({length:4},(_,i)=>token('sp-source-'+i,'',156,'teach-token sp-token teach-color-'+i));
  const rowTokens=[0,1].map(r=>Array.from({length:4},(_,i)=>token('sp-row-r'+r+'-'+i,'',168,'teach-token sp-token teach-color-'+i)));
  const midTokens=[0,1].map(r=>Array.from({length:4},(_,i)=>token('sp-mid-r'+r+'-'+i,'',240,'teach-token sp-token teach-color-'+i)));
  const outTokens=[0,1].map(r=>Array.from({length:4},(_,i)=>token('sp-out-r'+r+'-'+i,'',168,'teach-token sp-token sp-output teach-color-'+i)));
  const ghostLabels=[0,1].map(r=>Array.from({length:4},(_,i)=>txt('','sp-ghost-label')));
  const routeHints=[0,1].map(r=>Array.from({length:4},(_,i)=>txt('','sp-route-hint')));
  const plusSigns=[0,1].map(r=>Array.from({length:4},(_,i)=>txt('+','sp-plus')));
  const eqLines=[0,1].map(()=>txt('','sp-eq-row'));
  // AG never starts before the local p→n transform is readable at p=2.
  const agStart=[2.02,2.20,2.44,2.68],agEnd=[2.20,2.44,2.68,2.92];
  const rsStart=[4.00,4.16,4.46,4.64],rsEnd=[4.16,4.34,4.64,4.82];
  const mStart=[4.36,4.38,4.84,4.86],mEnd=[4.42,4.44,4.90,4.92];
  draw=(p,m)=>{
   lastP=p;lastM=m;
   const L=layout(m),tp=mode==='tp',splitIn=phase(p,.12,1);
   const midX=L.colX[1]+6,midW=L.colW[1]-12;
   const trackW=tp?0:L.trackW,mainW=tp?midW:midW-trackW-4;
   const trackX=midX+mainW+4,chipW=tp?mainW:Math.max(18,L.trackW-4),chipH=L.rowH;
   const halfW=tp?mainW:(mainW-L.dockGap)/2,dockX=midX+halfW+L.dockGap;
   const trackLeft=trackX+(trackW-chipW)/2;
   const rowTop=(r,i)=>L.laneY[r]+L.panelTop+L.rowPad+i*(L.rowH+L.rowGap);
   const cellX=c=>L.colX[c]+6,cellW=c=>L.colW[c]-12;
   const localIds=r=>tp?tokens:seq.shards[r];
   const localIn=tp?phase(p,2.2,2.8):phase(p,1.92,2.08);
   const agU=i=>tp?1:phase(p,agStart[i],agEnd[i]);
   const agRows=tp?4:[0,1,2,3].filter(i=>agU(i)>=1-1e-9).length;
   const gateOpen=tp||agRows===4;
   const contribU=i=>tp?phase(p,3.0,3.3):phase(p,3.26+.04*i,3.66+.04*i);
   const rsU=i=>tp?0:phase(p,rsStart[i],rsEnd[i]);
   const mergeU=i=>tp?0:phase(p,mStart[i],mEnd[i]);
   const outU=phase(p,4.94,5.0);
   const allMerged=r=>[0,1,2,3].filter(i=>ownerOf(i)===r).every(i=>mergeU(i)>=1-1e-9);
   const copiesDone=r=>tp?4:[0,1,2,3].filter(i=>ownerOf(i)!==r&&agU(i)>=1-1e-9).length;
   text(header,m?12:18,m?22:24,m?'SP · 同一 2 GPU 的 TP / TP+SP':'SP · 同一 2 GPU：TP 基线与 TP+SP 的行副本');
   set(header,{style:'font-size:'+(m?14:17)+'px'});
   text(hint,m?12:18,m?42:46,m?'同一 4 行；切换只改逐 token 区域副本':'同一序列 p1…p4；每行两维保持完整；切换条件只改逐 token 区域的激活行副本');
   set(hint,{style:'font-size:'+(m?10.5:12)+'px'});
   for(const card of cards){
    const on=card.key===mode,isTp=card.key==='tp',C=L.cards,i=cards.indexOf(card);
    set(card.g,{transform:`translate(${C.x[i]} ${C.y})`,class:'sp-mode'+(on?' is-active':''),'aria-pressed':String(on),'aria-label':(isTp?'未采用：TP 已启用，逐 token 区域每卡复制 4 行，共 8 行副本；TP 区域用 AllReduce':'采用后：同一 2 GPU，TP+SP 逐 token 区域每卡 2 行，共 4 行副本；AllGather 收齐 4 行，ReduceScatter 归位')+'。按 Enter 或空格切换。'});
    box(card.bg,0,0,C.w,C.h);
    text(card.title,10,C.titleY,isTp?(m?'TP 基线':'TP 基线 · 无 SP'):(m?'TP+SP':'TP + SP'));
    set(card.title,{style:'font-size:'+(m?13:15)+'px'});
    text(card.badge,C.w-10,C.badgeY,isTp?'8 行副本':'4 行副本');
    set(card.badge,{'text-anchor':'end',style:'font-size:'+(m?9.5:10.5)+'px'});
    text(card.foot,10,C.footY,isTp?(m?'4 行/卡 · AllReduce':'每卡 4 行 · TP 区域 AllReduce'):(m?'2 行/卡 · AG→TP→RS':'每卡 2 行 · AG → TP → RS'));
    set(card.foot,{style:'font-size:'+(m?9:10)+'px'});
    card.chips.forEach(c=>show(c.g,0));
    const groups=isTp?[[0,1,2,3],[0,1,2,3]]:[[0,1],[2,3]];
    const groupW=n=>C.labelW+4+n*C.chipW+(n-1)*C.gap;
    const place=(r,x,y)=>{
     text(card.labels[r],x,y+12,'G'+r);set(card.labels[r],{style:'font-size:'+(m?9:10)+'px'});
     let qx=x+C.labelW+4;
     groups[r].forEach(id=>{
      const c=card.chips.find(ch=>ch.rank===r&&ch.row===id);
      set(c.rect,{width:C.chipW,height:C.chipH,rx:3});
      set(c.label,{x:C.chipW/2,y:Math.round(C.chipH*.72),style:'font-size:'+(m?8.5:9.5)+'px'});
      c.label.textContent=tokens[id];set(c.g,{transform:`translate(${qx} ${y})`});show(c.g,1);qx+=C.chipW+C.gap;
     });
     return x+groupW(groups[r].length);
    };
    if(m){place(0,10,C.rowY[0]);place(1,10,C.rowY[1]);}
    else{
     const total=groupW(groups[0].length)+C.groupGap+groupW(groups[1].length),start=(C.w-total)/2;
     place(1,place(0,start,C.rowY[0])+C.groupGap,C.rowY[0]);
    }
   }
   box(srcPanel,L.src.x,L.src.y,L.src.w,L.src.h);
   text(srcLabel,L.src.x+12,L.src.y+L.src.labelY,m?'同一输入 · 4 行完整特征':'同一输入 · 4 个位置 × 2 个特征');
   set(srcLabel,{style:'font-size:'+(m?9.5:11)+'px'});
   text(srcNote,L.src.x+L.src.w-12,L.src.y+L.src.labelY,m?'切行不切特征':'两条件共享；切行不切特征');
   set(srcNote,{'text-anchor':'end',style:'font-size:'+(m?9.5:11)+'px'});
   source.forEach((t,i)=>{const sw=L.src.tokenW;set(t.r,{width:sw,height:L.src.tokenH});set(t.t,{x:sw/2,y:Math.round(L.src.tokenH*.66),style:'font-size:'+(L.rowSize)+'px'});t.t.textContent=pLabel(i);move(t,L.src.x+12+i*(sw+L.src.gap),L.src.y+L.src.tokenY);set(t.g,{class:'teach-token sp-token teach-color-'+i});t.g.dataset.spState='p';t.g.dataset.spRow=String(i);});
   const headTexts=m?['① 逐 token','② TP 区域','③ 输出']:['① 逐 token 区域','② TP 区域（需完整序列）','③ 逐 token 区域'];
   colHeads.forEach((t,i)=>{text(t,L.colX[i],L.colHeadY,headTexts[i]);set(t,{style:'font-size:'+(L.headSize)+'px'});});
   text(colSubs[0],L.colX[0],L.colSubY,tp?(m?'4/卡':'4 行/卡'):(m?'2/卡':'2 行/卡'));
   let midSub;
   if(tp)midSub=p<2?(m?'本地 4/4':'本地 4/4 · 无需 AG'):p<4?(m?'TP 计算':'TP 区域 · 部分贡献计算'):p<5?(m?'cᵢʳ 未合并':'部分贡献 c_i^r · 尚未合并'):(m?'Σ 同位置':'Σ 同位置求和 · 跨 rank');
   else if(p<1.92)midSub=m?'等待本地行':'等待本地 2 行';
   else if(p<2.06)midSub=m?'本地 2/4':'本地 2/4 · 等待收行';
   else if(!gateOpen)midSub=m?'AG '+agRows+'/4':'AG '+agRows+'/4 · 等待收齐';
   else if(p<3.2)midSub='AG 4/4 · 可进 TP';
   else if(p<4)midSub=m?'TP 计算 cᵢʳ':'TP 区域 · 部分贡献 c_i^r';
   else if(p<5)midSub=m?'RS 归约中':'c_i^r · RS 同位置归约中';
   else midSub=m?'Σ 同位置':'Σ 同位置求和 · 跨 rank';
   text(colSubs[1],L.colX[1],L.colSubY,midSub);
   text(colSubs[2],L.colX[2],L.colSubY,tp?(p>=5?(m?'4 行':'AllReduce · 4 行'):(m?'AllReduce':'AllReduce · '+(p<3.2?'待 TP':p<4?'计算中':'待归约'))):(p>=5?(m?'RS · 2 行':'ReduceScatter · 2 行'):(p<3.2?(m?'RS · 待 TP':'ReduceScatter · 待 TP'):p<4?(m?'RS · 待归约':'ReduceScatter · 待归约'):(m?'RS · 同位置求和':'ReduceScatter · 同位置求和'))));
   colSubs.forEach(t=>set(t,{style:'font-size:'+(m?9.5:10.5)+'px'}));
   for(let r=0;r<2;r++){
    text(laneNames[r],L.colX[0],L.laneY[r]+L.laneHeadY,'GPU '+r+' · '+(m?'r'+r:'rank '+r));
    set(laneNames[r],{style:'font-size:'+(m?10:11.5)+'px'});
    let note;
    if(p<1)note='等待分工';
    else if(p<2)note='本地 '+localIds(r).join('/');
    else if(p<3)note=tp?(m?'本地 n1…n4':'本地 n1…n4'):('本地 '+localIds(r).map(t=>'n'+t.slice(1)).join('/')+' · 副本 '+copiesDone(r)+'/2');
    else if(p<4)note=tp?(m?'无需 AG · 4/4':'无需 AG · 4 行齐'):'AG 4/4 · 进入 TP';
    else if(p<5)note=tp?(m?'c1⁰…c4⁰':'c1⁰…c4⁰ 完成'):(allMerged(r)?('RS 同位置 · 本卡 y'+[0,1,2,3].filter(i=>ownerOf(i)===r).map(i=>i+1).join('/')):'RS 同位置 · 收对端');
    else note=tp?(m?'AllReduce':'AllReduce · y1…y4'):('本卡 y'+[0,1,2,3].filter(i=>ownerOf(i)===r).map(i=>i+1).join('/'));
    text(laneNotes[r],L.W-18,L.laneY[r]+L.laneHeadY,note);
    set(laneNotes[r],{'text-anchor':'end',style:'font-size:'+(L.noteSize)+'px'});
    for(let c=0;c<3;c++)box(regionPanels[r*3+c],L.colX[c],L.laneY[r]+L.panelTop,L.colW[c],L.panelH);
   }
   for(const s of slots){
    box(s.s,cellX(s.c),rowTop(s.r,s.i),cellW(s.c),L.rowH);show(s.s,1);
   }
   if(tp||p<1.9||p>=5)show(rail,0);
   else wire(rail,`M${(trackLeft+chipW/2).toFixed(2)} ${(rowTop(0,0)-4).toFixed(2)} L${(trackLeft+chipW/2).toFixed(2)} ${(rowTop(1,3)+L.rowH+4).toFixed(2)}`,.35);
   // Local rows stay in the tokenwise column. The TP region starts with only
   // the two owned rows; the other two are filled by labeled AllGather copies.
   rowTokens.forEach((row,r)=>row.forEach((t,i)=>{
    const owner=ownerOf(i),visible=tp||owner===r,w=cellW(0),x=cellX(0),y=rowTop(r,i);
    set(t.r,{width:w,height:L.rowH});set(t.t,{x:w/2,y:Math.round(L.rowH*.68),style:'font-size:'+(L.rowSize)+'px'});
    t.t.textContent=p>=2?nLabel(i):pLabel(i);
    move(t,x,y,visible?splitIn:0);
    set(t.g,{class:'teach-token sp-token teach-color-'+i});
    t.g.dataset.spRank=String(r);t.g.dataset.spRow=String(i);t.g.dataset.spOwner=String(owner);t.g.dataset.spState=p>=2?'n':'p';
   }));
   midTokens.forEach((row,r)=>row.forEach((t,i)=>{
    const owner=ownerOf(i),isLocal=tp||owner===r,u=agU(i),c=contribU(i),rsP=rsU(i),merge=mergeU(i);
    let x=midX,y=rowTop(r,i),w=mainW,op=tp?localIn:(isLocal?localIn:0),label=nLabel(i),state='pending',origin=tp?'tp':(isLocal?'local':'ag');
    if(!tp&&!isLocal){
     if(u<=0)op=0;
     else if(u<.18){x=trackLeft;y=rowTop(owner,i);w=chipW;op=u/.18;label='n'+(i+1);state='copy';}
     else if(u<.62){x=trackLeft;y=lerp(rowTop(owner,i),rowTop(r,i),phase(u,.18,.62));w=chipW;op=1;label='n'+(i+1);state='copy';}
     else if(u<.72){x=trackLeft;y=rowTop(r,i);w=chipW;op=1;label='n'+(i+1);state='copy';}
     else{const h=phase(u,.72,1);x=lerp(trackLeft,midX,h);y=rowTop(r,i);w=lerp(chipW,mainW,h);op=1;label=h>=.55?nLabel(i):'n'+(i+1);state=u>=1?'n':'copy';}
    }
    if(tp||isLocal)state='n';
    if(c>0){
     if(!tp)w=lerp(w,halfW,c);
     if(c>=.35)label=(!tp&&halfW<100)?cShort(i,r):cLabel(i,r);
     state='c';origin=tp?'tp':(isLocal?'local':'ag');
    }
    if(!tp&&!isLocal&&rsP>0){
     if(rsP<.14){const h=phase(rsP,0,.14);x=lerp(midX,trackLeft,h);y=rowTop(r,i);w=lerp(halfW,chipW,h);}
     else if(rsP<.60){x=trackLeft;y=lerp(rowTop(r,i),rowTop(owner,i),phase(rsP,.14,.60));w=chipW;}
     else if(rsP<.72){x=trackLeft;y=rowTop(owner,i);w=chipW;}
     else{const h=phase(rsP,.72,1);x=lerp(trackLeft,dockX,h);y=rowTop(owner,i);w=lerp(chipW,halfW,h);}
     op=1;label=rsP>=.85?((!tp&&halfW<100)?cShort(i,r):cLabel(i,r)):cShort(i,r);state=rsP>=1?'dock':'rs';origin='rs';
    }
    if(!tp&&merge>0){
     x=isLocal?midX+10*merge:dockX-10*merge;
     w=halfW;
     // The two contribution labels stay readable through most of the merge and
     // are gone exactly when the y result takes the slot at merge=1.
     op=1-phase(merge,.72,1);
     label=(halfW<100)?cShort(i,r):cLabel(i,r);
     state='merge';
    }
    set(t.r,{width:w,height:L.rowH});set(t.t,{x:Math.round(w/2),y:Math.round(L.rowH*.68),style:'font-size:'+((!tp&&w<=chipW+2)?L.chipSize:L.rowSize)+'px'});
    t.t.textContent=label;
    move(t,x,y,op);
    set(t.g,{class:'teach-token sp-token teach-color-'+i+(c>0?' sp-contrib':'')+(!tp&&!isLocal?' sp-remote':'')});
    t.g.dataset.spRank=String(r);t.g.dataset.spRow=String(i);t.g.dataset.spOwner=String(owner);
    t.g.dataset.spSourceRank=String(origin==='ag'?owner:r);
    t.g.dataset.spOrigin=origin;
    t.g.dataset.spState=merge>0?'merge':(tp||isLocal?(c>0?'c':'n'):(rsP>0?(rsP>=1?'dock':'rs'):(c>0?'c':(u>=1?'n':'copy'))));
    t.g.dataset.spAgProgress=u.toFixed(4);t.g.dataset.spMerge=merge.toFixed(4);
    t.g.dataset.spDockRank=String(owner);
    t.g.setAttribute('aria-label',(state==='copy'?'AllGather 行副本':state==='rs'?'ReduceScatter 在途贡献':state==='merge'?'同位置归约':state==='c'?'部分贡献':'完整行')+' '+label+'；GPU '+r+'，第 '+(i+1)+' 行');
    // Only the owner lane has a wait slot for the peer contribution. The other
    // lane shows a short direction hint instead, since its partial leaves.
    const ghost=ghostLabels[r][i],plus=plusSigns[r][i],hint=routeHints[r][i];
    if(!tp&&c>=.9&&(owner===r)&&merge<1){
     const peer=rsU(i);
     // Placeholder text exists only while the peer chip is still waiting on the
     // rail. Once it starts expanding into the dock, the moving c_i label is the
     // only label: never draw the same c_i twice in the same slot.
     text(ghost,dockX+halfW/2,rowTop(r,i)+Math.round(L.rowH*.68),cShort(i,1-r));
     set(ghost,{'text-anchor':'middle',style:'font-size:'+(halfW<100?L.chipSize:L.rowSize)+'px'});
     show(ghost,peer<.72?.5:0);
     // The plus sits between the two contributions from the moment the peer
     // leaves the rail until the merged slot becomes y.
     text(plus,midX+halfW+L.dockGap/2,rowTop(r,i)+Math.round(L.rowH*.72),'+');
     set(plus,{'text-anchor':'middle',style:'font-size:'+(L.eqSize+2)+'px'});
     show(plus,peer>=.72?1:.35);
    }else{show(ghost,0);show(plus,0);}
    if(!tp&&c>=.9&&owner!==r&&rsP<.02){
     text(hint,midX+halfW+L.dockGap+halfW/2,rowTop(r,i)+Math.round(L.rowH*.7),'→G'+owner);
     set(hint,{'text-anchor':'middle',style:'font-size:'+(L.noteSize)+'px'});
     show(hint,.7);
    }else show(hint,0);
   }));
   const eqY=r=>L.laneY[r]+L.laneH-7;
   outTokens.forEach((row,r)=>row.forEach((t,i)=>{
    const owner=ownerOf(i),owned=tp||owner===r,merge=tp?0:mergeU(i),shown=tp?phase(p,4.9,5):(merge>=1?1:0);
    const x=tp?cellX(2):lerp(midX,cellX(2),outU),w=tp?cellW(2):lerp(mainW,cellW(2),outU),y=rowTop(r,i);
    set(t.r,{width:w,height:L.rowH});set(t.t,{x:Math.round(w/2),y:Math.round(L.rowH*.68),style:'font-size:'+(L.rowSize)+'px'});
    t.t.textContent=yLabel(i);move(t,x,y,owned?shown:0);
    set(t.g,{class:'teach-token sp-token sp-output teach-color-'+i});
    t.g.dataset.spRank=String(r);t.g.dataset.spRow=String(i);t.g.dataset.spOwner=tp?'both':String(owner);
    t.g.dataset.spState=owned&&shown>0?'y':'pending';t.g.dataset.spMerge=merge.toFixed(4);
    t.g.setAttribute('aria-label','归约输出 '+yLabel(i)+'；GPU '+r);
   }));
   eqLines.forEach((t,r)=>{
    const owned=[0,1,2,3].filter(i=>ownerOf(i)===r);
    if(tp){
     text(t,midX+mainW/2,eqY(r),'y_i = c_i⁰ + c_i¹ · AllReduce，两卡各 4 行');
     show(t,phase(p,4.9,5));
    }else{
     text(t,midX+mainW/2,eqY(r),m?owned.map(i=>'y'+(i+1)+'=c'+(i+1)+'⁰+c'+(i+1)+'¹').join(' · '):owned.map(i=>'y'+(i+1)+' = c'+(i+1)+'⁰+c'+(i+1)+'¹').join(' · '));
     show(t,allMerged(r)?1:0);
    }
    set(t,{'text-anchor':'middle',style:'font-size:'+(L.eqSize)+'px'});
   });
   text(foot[0],18,L.footY[0],tp?(m?'TP：4→4→4 行/卡':'TP 基线：逐 token 4 行/卡 → TP 区域 4 行/卡 → AllReduce 后 4 行/卡'):(m?'SP：2→4→2 行/卡':'TP+SP：逐 token 2 行/卡 → AllGather 后 4 行/卡 → ReduceScatter 后 2 行/卡'));
   text(foot[1],18,L.footY[1],tp
    ?(m?'y 同位置求和；AllReduce 每卡 4 行':'y_i = c_i⁰ + c_i¹；AllReduce 对同位置求和，两张卡各自保留完整 4 行 y；与 SP 的 y 值相同')
    :(m?'y 同位置求和；RS 再按行归属':'y_i = c_i⁰ + c_i¹；ReduceScatter 先对同位置求和，再按行归属：GPU0 得 y1/y2，GPU1 得 y3/y4；与 TP 的 y 值相同'));
   show(foot[1],phase(p,3.95,4.15));
   text(foot[2],18,L.footY[2],tp?(m?'TP 也用 AllReduce；逐 token 仍 4 行/卡':'基线不是无通信：TP 区域用 AllReduce；区别是逐 token 区域每卡仍复制 4 行'):(m?'本地行不搬空；错开只为看清':'AllGather 补的是行副本，本地行不搬空；AG 先收 n1/n2、RS 先并 y1/y2 只是错开演示，不代表真实串行'));
   text(foot[3],18,L.footY[3],m?'颜色=位置；虚线=副本/对端；切行不切特征':'颜色 = 序列位置；虚线 = AG 行副本 / 对端贡献；切的是序列行，不是每行特征列；不称加速');
   foot.forEach(t=>set(t,{style:'font-size:'+(m?9.5:10.5)+'px'}));
   svg.dataset.parallelKind='sp';svg.dataset.spMode=mode;
   svg.dataset.spRowsPerRank=String(tp?4:2);svg.dataset.spReplicas=String(tp?8:4);
   svg.dataset.spGate=gateOpen?'open':'closed';svg.dataset.spAgRows=String(agRows);
   svg.dataset.spPhase=p<1?'input':p<2?'split':p<2.06?'local':p<3?'gather':p<4?'contribution':p<5?'reduce':'output';
   const plan=tp?'r0:y1,y2,y3,y4;r1:y1,y2,y3,y4':('r0:'+localIds(0).map(t=>'y'+(tokens.indexOf(t)+1)).join(',')+';r1:'+localIds(1).map(t=>'y'+(tokens.indexOf(t)+1)).join(','));
   svg.dataset.spOutputPlan=plan;svg.dataset.spOutputs=p>=5?plan:'pending';
  };
  paint=(p,m)=>{draw(p,m);};
 
  }else if(kind==='dcp'){
  const equal=Math.abs(d.sums[0]-d.sums[1])<1e-9;
  const fmt=v=>Number.isInteger(v)?String(v):String(Number(v.toFixed(2)));
  const kLabel=i=>d.exp[i]===1?'0':d.exp[i]===3?'ln3':fmt(d.exp[i]);
  const kArr=d.exp.map((_,i)=>kLabel(i));
  const vals=d.values.map(fmt),zArr=d.sums.map(fmt),nArr=d.numerators.map(fmt),oArr=d.local.map(fmt);
  const weights=d.weights.map(v=>Math.round(v*100));
  const totalZ=fmt(d.merge.denominator),totalN=fmt(d.merge.numerator),answer=fmt(d.result);
  const lineTexts=[0,1].map(r=>{
   const e=d.exp.slice(r*2,r*2+2),v=d.values.slice(r*2,r*2+2);
   return [`exp(QK)：${e.join(', ')}`,`Z = ${e[0]}+${e[1]} = ${zArr[r]}`,`分子 = ${e[0]}×${v[0]}+${e[1]}×${v[1]} = ${nArr[r]}`,`O = ${nArr[r]}/${zArr[r]} = ${oArr[r]}`];
  });
  const srcLabel=txt('','teach-small dcp-note'),srcSummary=txt('','teach-small dcp-note'),qBanner=txt('','teach-small dcp-note'),derive=txt('','teach-small dcp-note');
  const srcChips=d.values.map((v,i)=>tile('dcp-source-'+i,`p${i+1} · K=${kArr[i]} · V=${v}`,i<2?0:1,220));
  const ownerLabels=[txt('','teach-small dcp-owner'),txt('','teach-small dcp-owner')];
  const qSource=tile('dcp-query','Q = 1',4,100),qMoves=[tile('dcp-qmove-0','Q',4,34),tile('dcp-qmove-1','Q',4,34)],qWires=[path('dcp-wire'),path('dcp-wire')];
  const rankSubs=[txt('','teach-small dcp-note'),txt('','teach-small dcp-note')];
  const qChips=[0,1].map(r=>tile('dcp-q-'+r,'Q=1（副本）',4,110));
  const kvLabels=[0,1].map(()=>txt('','teach-small dcp-note'));
  const kvChips=[0,1].map(r=>[0,1].map(j=>tile(`dcp-kv-${r}-${j}`,`p${r*2+j+1} K=${kArr[r*2+j]} V=${vals[r*2+j]}`,r,168)));
  const formulaTitles=[0,1].map(()=>txt('','teach-small dcp-note'));
  const formulaLines=[0,1].map(()=>Array.from({length:4},()=>txt('','teach-small dcp-formula')));
  const localBadges=[0,1].map(()=>txt('','teach-small dcp-badge'));
  const gateBg=rect('teach-gate dcp-gate'),gateText=txt('','teach-text dcp-gate-text');
  const laneBgs=[rect('dcp-lane'),rect('dcp-lane')],laneLabels=[txt('','teach-small dcp-lane-label'),txt('','teach-small dcp-lane-label')];
  const payloads=[tile('dcp-payload-0',`O₀=${oArr[0]} · Z₀=${zArr[0]}`,0,150),tile('dcp-payload-1',`O₁=${oArr[1]} · Z₁=${zArr[1]}`,1,150)];
  const mergePanel=rect('teach-panel dcp-merge'),mergeTitle=txt('','teach-text dcp-merge-title'),mergeWait=txt('','teach-small dcp-note');
  const denom=txt('','teach-small dcp-formula'),numer=txt('','teach-small dcp-formula'),weightLine=txt('','teach-small dcp-formula');
  const baseTitle=txt('','teach-text dcp-base-title'),baseEq=txt('','teach-small dcp-formula'),baseNote=txt('','teach-small dcp-note');
  const meanLine=txt('','teach-small dcp-formula'),meanNote=txt('','teach-small dcp-note');
  // The merge panel is appended before its own labels. These nodes were created
  // earlier with the shared chrome, so move them to the top of the SVG once, in
  // reading order. Otherwise the opaque panel rect covers the final global O.
  svg.append(mergePanel,mergeTitle,mergeWait,denom,numer,result,baseTitle,baseEq,meanLine,meanNote,weightLine);
  paint=(p,m)=>{
   const desktop=!m;
   svg.setAttribute('viewBox',desktop?'0 0 960 636':'0 0 360 792');
   svg.dataset.dcpLayout=desktop?'wide':'narrow';
   const arrive0=phase(p,3.0,3.5),arrive1=phase(p,3.15,3.95);
   const arrived=(arrive0>=1?1:0)+(arrive1>=1?1:0),merged=arrived===2&&p>=4;
   const status=p<2?'Q 已到两卡；各卡只读本地 K/V，尚未开始算局部 O/Z。':p<2.9?'正在逐项计算局部 O/Z；尚未交换。':p<3?'局部 O/Z 已算出；交换门打开，尚未开始传输。':arrived===0?'0/2 到达：两份 O/Z 都在途，不能合并。':arrived===1?`1/2 到达：${arrive0>=1?'R1 已收到 R0 的 O/Z':'R0 尚未收到 R1 的 O/Z'}；只表示对应目的卡收到，不能合并。`:merged?`2/2 到达：两片 O/Z 已合并；全局 O=${answer}。`:'2/2 到达：两片都已收到对端 O/Z；准备按全局 Z 加权合并。';
   const statusN=p<2?'Q 已到两卡；只读本地 K/V，未算 O/Z。':p<3?'逐项算局部 O/Z；尚未交换。':arrived===0?'0/2 到：都在途，不能合并':arrived===1?`1/2 到：${arrive0>=1?'R1 已收 R0':'R0 未收 R1'}；不能合并`:merged?`2/2 到并已合并；O=${answer}`:'2/2 到：两片到齐，准备合并';
   text(header,18,24,'DCP：历史 KV 分片，两张卡共同算一个 Query');
   text(hint,18,44,desktop?'同一新 Query Q=1 分发到两卡；历史 KV 分 2/2 留在本卡（Q 分发不搬历史 KV），其后只交换 O/Z；新位置 KV 写入与其余层省略':'Q 分发到两卡；历史 KV 留本卡（不随 Q 搬），只交换 O/Z');
   if(desktop){
    text(srcLabel,28,64,'完整历史：同一个请求的 4 个位置（总历史仍为 4）');
    srcChips.forEach((t,i)=>move(t,28+i*228,74,1-phase(p,.3,.7)*.45));
    text(ownerLabels[0],28,122,'→ Rank 0 持有 p1/p2');text(ownerLabels[1],492,122,'→ Rank 1 持有 p3/p4');
    text(derive,28,140,equal?'K 全为 0 → exp(QK)=1；V=[1,3,5,7]':'exp 由 Q=1 与 K 得出：exp(0)=1；exp(ln3)=3');
    show(srcSummary,0);show(qBanner,0);
    move(qSource,430,154,1);
    set(qWires[0],{d:'M452 188 L452 217 L91 217'});set(qWires[1],{d:'M508 188 L508 217 L631 217'});
    show(qWires[0],phase(p,.25,.45));show(qWires[1],phase(p,.35,.55));
    // The same Q is copied twice. The two copies leave from the two bottom
    // corners of the source card at the same time, then follow separate rails;
    // they never share a card-sized lane over the source's "Q=1" label.
    const q0=phase(p,.3,.9),q1=phase(p,.3,.9);
    const qpos=(u,startX,endX)=>{if(u<=0)return null;if(u<.2)return{x:startX-17,y:lerp(188,200,u/.2)};return{x:lerp(startX-17,endX-17,phase(u,.2,1)),y:200};};
    [[0,q0,452,91],[1,q1,508,631]].forEach(([i,u,startX,endX])=>{const pos=p<1?qpos(u,startX,endX):null;if(pos)move(qMoves[i],pos.x,pos.y,1);else show(qMoves[i].g,0);});
   }else{
    show(srcLabel,1);text(srcLabel,14,64,'完整历史：p1…p4（4 个位置，总历史不变）');
    show(srcSummary,1);text(srcSummary,14,80,`K=[${kArr.join(', ')}] · V=[${vals.join(', ')}]`);
    text(qBanner,180,94,'Q 分发到两卡；只交换 O/Z');qBanner.setAttribute('text-anchor','middle');show(qBanner,1);
    srcChips.forEach(t=>show(t.g,0));ownerLabels.forEach(t=>show(t,0));show(derive,0);show(qSource.g,0);qWires.forEach(t=>show(t,0));qMoves.forEach(t=>show(t.g,0));
   }
   const px=desktop?[20,560]:[8,8],py=desktop?[154,154]:[98,326],pw=desktop?380:344,ph=desktop?242:220;
   [0,1].forEach(r=>{
    box(panels[r],px[r],py[r],pw,ph);show(panels[r],1);
    set(names[r],{class:'teach-heading dcp-rank'});text(names[r],px[r]+16,py[r]+20,`Rank ${r} · P${r} / GPU ${r}`);
    text(rankSubs[r],px[r]+16,py[r]+38,'本地历史 K/V：2 个位置 · 原 KV 留在本卡');
    move(qChips[r],px[r]+16,py[r]+46,p>=1?1:0);
    text(kvLabels[r],px[r]+16,py[r]+98,'本地历史 K/V（不交换）');
    kvChips[r].forEach((t,j)=>{const kw=desktop?168:160;set(t.r,{width:kw});set(t.t,{x:kw/2});move(t,px[r]+16+j*(desktop?176:170),py[r]+106,1);});
    const formulaOn=p>=2;
    text(formulaTitles[r],px[r]+16,py[r]+152,'局部 Attention：Q × 本地 K/V');
    show(formulaTitles[r],formulaOn?1:0);
    formulaLines[r].forEach((t,i)=>{text(t,px[r]+16,py[r]+170+i*(desktop?20:15),lineTexts[r][i]);show(t,formulaOn?phase(p,2.02+i*.18,2.02+i*.18+.24):0);});
    if(desktop){
     const badge=p>=4?'局部 O/Z · 已合并':p>=3?'局部 O/Z · 交换中':'局部 O/Z · 已就绪';
     text(localBadges[r],px[r]+364,py[r]+20,badge);
     localBadges[r].setAttribute('text-anchor','end');
     show(localBadges[r],p>=2.9?1:0);
    }else show(localBadges[r],0);
   });
   const gateStr=desktop?(p<3?(p<2.9?'交换门：等待局部 O/Z':'交换门：局部 O/Z 已就绪 · 尚未开始'):arrived<2?`交换门：${arrived}/2 到达 · 未合并`:(merged?'交换门：2/2 到达 · 已合并':'交换门：2/2 到达 · 下一步合并')):(p<3?(p<2.9?'交换门：等待 O/Z':'交换门：局部 O/Z 已就绪'):arrived<2?`交换门：${arrived}/2 · 未合并`:(merged?'交换门：已合并':'交换门：2/2 · 待合并'));
   box(gateBg,desktop?330:70,desktop?404:588,desktop?300:220,26);text(gateText,desktop?480:180,desktop?421:605,gateStr);gateText.setAttribute('text-anchor','middle');
   show(gateBg,p>=2.9?1:0);show(gateText,p>=2.9?1:0);
   if(desktop){
    box(laneBgs[0],20,436,920,30);box(laneBgs[1],20,470,920,30);
    show(laneBgs[0],p>=3?1:0);show(laneBgs[1],p>=3?1:0);
    text(laneLabels[0],30,456,`R0 → R1 · O₀/Z₀${arrive0>=1?' · R1 已收到':''}`);
    text(laneLabels[1],30,490,`R1 → R0 · O₁/Z₁${arrive1>=1?' · R0 已收到':''}`);
    show(laneLabels[0],p>=3?1:0);show(laneLabels[1],p>=3?1:0);
    move(payloads[0],lerp(390,560,arrive0),433,p>=3?1:0);
    move(payloads[1],lerp(560,390,arrive1),468,p>=3?1:0);
    text(note,36,516,status);show(note,p>=2.9?1:0);
   }else{
    show(laneBgs[0],0);show(laneBgs[1],0);show(payloads[0].g,0);show(payloads[1].g,0);
    text(laneLabels[0],14,562,`R0 O=${oArr[0]} Z=${zArr[0]} → R1${arrive0>=1?' · 已收到':''}`);
    text(laneLabels[1],14,580,`R1 O=${oArr[1]} Z=${zArr[1]} → R0${arrive1>=1?' · 已收到':''}`);
    show(laneLabels[0],p>=3?1:0);show(laneLabels[1],p>=3?1:0);
    text(note,14,624,statusN);show(note,p>=2.9?1:0);
   }
   box(mergePanel,desktop?20:8,desktop?528:634,desktop?920:344,desktop?92:140);show(mergePanel,p>=2.9?1:0);
   text(mergeTitle,desktop?36:18,desktop?544:650,'到齐后合并（本图简化：全组屏障）');show(mergeTitle,p>=2.9?1:0);
   text(mergeWait,desktop?480:180,desktop?570:674,arrived<2?`等待两片 O/Z 到齐（${arrived}/2）`:'两片 O/Z 已到齐 · 准备合并');mergeWait.setAttribute('text-anchor','middle');show(mergeWait,p>=2.9&&p<4?1:0);
   const showMerge=p>=4;
   text(denom,desktop?36:18,desktop?564:670,`全局分母 Z = Z₀+Z₁ = ${zArr[0]}+${zArr[1]} = ${totalZ}`);show(denom,showMerge?1:0);
   text(numer,desktop?36:18,desktop?584:690,`全局分子 = Z₀×O₀+Z₁×O₁ = ${zArr[0]}×${oArr[0]}+${zArr[1]}×${oArr[1]} = ${totalN}`);show(numer,showMerge?1:0);
   text(result,desktop?36:18,desktop?606:714,`全局 O = ${totalN} / ${totalZ} = ${answer}`);set(result,{class:'teach-heading dcp-result'});show(result,showMerge?1:0);
   text(baseTitle,desktop?600:18,desktop?542:732,'单卡参照：1 GPU 保存/读取 4 个位置');show(baseTitle,desktop&&showMerge?1:0);
   text(baseEq,desktop?600:18,desktop?562:732,desktop?`${totalN} / ${totalZ} = ${answer} · 与两卡合并结果相同`:`单卡参照 ${totalN}/${totalZ}=${answer}`);show(baseEq,showMerge?1:0);
   text(baseNote,0,0,'');show(baseNote,0);
   text(meanLine,desktop?600:18,desktop?582:750,desktop?(equal?'简单平均 (2+6)/2 = 4 = 4':'简单平均 (2+6)/2 = 4 ≠ 5'):(equal?'平均 4=4（巧合）':'平均 4≠5（错反例）'));show(meanLine,showMerge?1:0);
   text(meanNote,desktop?600:18,desktop?600:750,equal?'（仅 Z 相等时巧合，不能当规则）':'（不等权下的错误反例）');show(meanNote,desktop&&showMerge?1:0);
   text(weightLine,desktop?600:18,desktop?615:766,desktop?`权重 ${weights[0]}% / ${weights[1]}% · 两卡各 2 位 · 总 4 位`:`每卡 2 位 · 总 4 位 · 2 GPU`);show(weightLine,showMerge?1:0);
   text(line2,desktop?36:14,desktop?630:786,desktop?'多 GPU：Q 分发到两卡，合并统计 O/Z；历史 KV 不搬；省略新位置 KV 与其余层；实际实现通常交换 LSE（先减 max 再 exp）；不承诺加速。':'2 卡：Q 分发 + O/Z 统计；KV 不搬；省略新 KV/层');show(line2,p>=3.2?1:0);
   svg.dataset.dcpEqual=String(equal);svg.dataset.dcpReads='4';svg.dataset.dcpReadsPerRank='2';svg.dataset.dcpGpus='2';svg.dataset.dcpExchange='1';
   svg.dataset.dcpBaseline=answer;svg.dataset.dcpLocal=oArr.join('/');svg.dataset.dcpSums=zArr.join('/');
   svg.dataset.dcpMean=fmt(d.mean);svg.dataset.dcpMeanStatus=equal?'coincident':'wrong';
   svg.dataset.dcpArrived=String(arrived);svg.dataset.dcpReady=String(arrived===2);
   svg.dataset.dcpMerged=merged?answer:'pending';svg.dataset.dcpResult=merged?answer:'pending';
   svg.dataset.dcpPhase=p<1?'input':p<2?'dispatch':p<3?'local':p<4?'exchange':'merge';
   svg.dataset.dcpGate=p<3?'waiting':arrived<2?'partial':merged?'merged':'ready';
   svg.dataset.dcpBarrier='group';svg.dataset.dcpLse='stable';svg.dataset.dcpQ='1';
   svg.dataset.dcpOmitted='new-token-kv,remaining-layers';
  };
 }

return(p,m)=>{svg.dataset.parallelKind=kind;svg.dataset.step=Math.floor(p);paint(p,m);if(m&&kind!=='dcp'){header.setAttribute('font-size','16');result.setAttribute('font-size','15');hint.setAttribute('font-size','12');}};
 });
 if(s.kind==='parallel-sp'){
  const film=viewport.closest('.mechanism-film');
  const destroy=visual.destroy;
  visual.destroy=()=>{film?.classList.remove('is-sp');destroy();};
  return visual;
 }
 if(s.kind==='parallel-dcp'){
  const film=viewport.closest('.mechanism-film');if(film)film.classList.add('is-dcp');
  const destroy=visual.destroy;visual.destroy=()=>{film?.classList.remove('is-dcp');destroy();};
  return visual;
 }
 return visual;
};
})();
