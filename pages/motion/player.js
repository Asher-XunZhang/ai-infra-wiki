/* A seekable teaching timeline. Geometry and explanations are pure functions of time.
 * No scheduler state is mutated by animation callbacks. Actor DOM nodes live for
 * the entire scenario, including moves between zones and backwards seeking. */
(function () {
'use strict';
const catalogue=window.MechanismScenes, gsap=window.gsap;
if(!catalogue||!gsap)return;
const ns='http://www.w3.org/2000/svg';
const svgNode=(name,attrs={},text)=>{const e=document.createElementNS(ns,name);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;};
const node=(name,cls,text)=>{const e=document.createElement(name);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;};
function travel(a,b,u){const turns=Math.abs(a.x-b.x)>1&&Math.abs(a.y-b.y)>1;return turns?{x:a.x+(b.x-a.x)*Math.max(0,(u-.35)/.65),y:a.y+(b.y-a.y)*Math.min(1,u/.35)}:{x:a.x+(b.x-a.x)*u,y:a.y+(b.y-a.y)*u};}
function fitText(e,text,max){if(e.textContent===text)return;e.textContent=text;e.removeAttribute('textLength');e.removeAttribute('lengthAdjust');if(e.getComputedTextLength()>max){e.setAttribute('textLength',max);e.setAttribute('lengthAdjust','spacingAndGlyphs');}}
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
const active=new Set();
// Choose the main representation by the question the learner needs to answer.
// The continuous card walkthrough stays available as an optional second view.
const primaryViews={
 'f-tokens':['片段与编号的对应关系','沿同一列看文字片段、词表编号和向量，先看清它们怎样对应。'],
 'f-attention':['权重与加权结果','按位置对照分数、权重和 V，观察未来位置为什么被遮住。'],
 'f-cache':['缓存格与逐轮计算量','固定每个历史位置，对照哪些位置复用、哪些位置需要新算。'],
 'f-parallel':['计算分片与流水线时间格','切换 TP、PP、DP；PP 的横轴表示教学时隙，空格表示依赖造成的等待。'],
 'f-communication':['各卡输入与输出','对照相加、拼接和传输后的元素，先判断结果应该是什么。'],
 'f-speculation':['候选前缀与提交位置','沿候选序列找第一个分歧，再对齐正式输出和有效 KV。'],
 'f-metrics':['到达时刻与样本分布','先确认计时起止点，再读延迟、吞吐与分位数。'],
 'model-execution':['执行交接、图桶与矩阵','按问题切换调用交接、图桶补齐或矩阵成本，分别观察相应结构。'],
 'kv-memory':['请求位置与物理页','物理页保持固定，观察位置映射、页内空位与共享保护。'],
 'parallelism':['分片、时间格与拓扑','选择一种并行方式，观察切开的对象和各 rank 之间的依赖。'],
 'communication':['参与方与数据对应关系','观察发送方、接收方和元素去向；KV 传输进一步对齐源页与目标页。'],
 'advanced-generation':['输出序列与有效状态','对齐候选、正式输出与 KV，观察接受、回退和提交的边界。'],
 'performance':['请求时间轴与执行条带','以共同时间轴对照等待、执行和到达，再计算相应指标。']
};
function mount(host,key){
 const stepDuration=key==='f-communication'?2:1.3;
 const integration=window.LessonIntegration;
 if(integration?.originalOnly.includes(key)){if(key==='system-overview')window.HandoffBoundaries?.mount(host,key);return;}
 const noLegacy=key.startsWith('f-')&&integration?.foundation[key.slice(2)]?.length===0;
 const choices=window.TeachingLessons?.[key]||catalogue[key];if(!choices)return;
 const dedicated=window.MechanismVisuals?.[key];
 const section=node('section','mechanism-film');section.dataset.motion=key;section.id='mechanism-process';section.setAttribute('aria-label','连续机制演示');
 const head=node('div','motion-heading'),kicker=node('p','motion-kicker','观察过程 · 可以在途中暂停'),title=node('h2'),intro=node('p','motion-intro');
 head.append(kicker,title,intro);section.append(head);
 const controls=node('div','motion-controls');
 const button=(text,action)=>{const e=node('button','',text);e.type='button';e.dataset.motionAction=action;controls.append(e);return e;};
 const prev=button('上一步','previous'),play=button('播放过程','play'),next=button('下一步','next'),reset=button('重置','reset');play.setAttribute('aria-pressed','false');
 const speedLabel=node('label','animation-speed','速度 '),speed=node('input'),speedValue=node('output','','1×');speed.type='range';speed.min=.25;speed.max=3;speed.step=.25;speed.value=1;speed.setAttribute('aria-label','演示速度');speed.setAttribute('aria-valuetext','1 倍');speedLabel.append(speed,speedValue);controls.append(speedLabel);
 const selectLabel=node('label','','条件 '),select=node('select');select.setAttribute('aria-label','连续演示条件');const groups=new Map();choices.forEach((c,i)=>{const o=node('option','',c.name);o.value=i;if(c.group){if(!groups.has(c.group)){const g=node('optgroup');g.label=c.group;groups.set(c.group,g);select.append(g);}groups.get(c.group).append(o);}else select.append(o);});selectLabel.append(select);if(choices.length>1)controls.append(selectLabel);
 const familyConfig={
  'f-communication':{label:'通信操作',names:{...Object.fromEntries((window.CommunicationTeaching?.definitions||[]).map(d=>[d.id,d.id==='Completion'?'完成与可用':d.id==='Neighbor'?'邻域交换':d.id==='Atomic'?'原子更新':d.id==='SendRecv'?'Send / Recv':d.id])),matrix:'矩阵对照'},family:c=>c.matrix?'matrix':c.op},
  'f-parallel':{label:'并行策略',names:{TP:'张量',PP:'流水线',DP:'数据',EP:'专家',SP:'序列',PCP:'Prefill 上下文',DCP:'Decode 上下文'},family:c=>c.kind==='pipeline'?'PP':c.kind.replace('parallel-','').toUpperCase()},
  'f-metrics':{label:'测量问题',names:{latency:'首个输出与后续间隔',distribution:'P95 与成功率',throughput:'统计窗口与吞吐'},family:c=>c.family},
  'f-hardware':{label:'学习重点',names:{dot:'矩阵乘加',roofline:'算力与带宽',graph:'CUDA Graph'},family:c=>c.kind.replace('hardware-','')},
  'f-batching':{label:'批处理机制',names:{dynamic:'① 动态组批',fixed:'② 固定批次',continuous:'② 连续批处理',chunk:'③ Chunked Prefill'},family:c=>c.kind==='chunk-batching'?'chunk':c.kind==='dynamic-batching'?'dynamic':choices.indexOf(c)===1?'fixed':'continuous'},
  scheduling:{label:'调度机制',names:{admission:'准入检查',chunk:'分块续算'},family:c=>c.kind==='chunk'?'chunk':'admission'}
 }[key];
 const families=familyConfig?Object.keys(familyConfig.names):null,familyOf=c=>familyConfig.family(c),familyButtons=[];
 const policyLabel=node('label','','排序策略 '),policy=node('select');policy.setAttribute('aria-label','准入排序策略');
 if(key==='scheduling'){for(const [value,label]of [['fcfs','FCFS · 先来先检查'],['lpm','LPM · 长匹配前缀优先']]){const o=node('option','',label);o.value=value;policy.append(o);}policyLabel.append(policy);controls.insertBefore(policyLabel,selectLabel);policy.addEventListener('change',()=>{const old=choices[Number(select.value)];load(choices.findIndex(c=>c.kind==='admission'&&c.policy===policy.value&&c.scenario===old.scenario));save();});}
 if(families){const nav=node('nav','parallel-strategies');nav.setAttribute('aria-label',familyConfig.label);for(const family of families){const b=node('button','',key==='f-parallel'?family:familyConfig.names[family]);b.type='button';b.dataset.lessonFamily=family;if(key==='f-parallel'){b.dataset.parallelFamily=family;b.append(node('small','',familyConfig.names[family]));}b.addEventListener('click',()=>{load(choices.findIndex(c=>familyOf(c)===family));save();});if(key==='f-communication'){nav.classList.add('comm-strategies');const def=window.CommunicationTeaching.definitions.find(d=>d.id===family);let group=nav.querySelector('[data-comm-group="'+(def?def.group:'matrix')+'"]');if(!group){group=node('div','comm-operation-group');group.dataset.commGroup=def?def.group:'matrix';group.append(node('strong','',def?window.CommunicationTeaching.groups[def.group]:'矩阵对照'));nav.append(group);}group.append(b);}else nav.append(b);familyButtons.push(b);}head.insertBefore(nav,title);selectLabel.firstChild.textContent='场景 ';}
 function syncFamily(index){if(!families)return;const current=choices[index],family=familyOf(current);select.replaceChildren();choices.forEach((c,i)=>{if(familyOf(c)!==family||(current.policy&&c.policy!==current.policy))return;const option=node('option','',c.scenarioLabel||c.name.split(' · ').slice(1).join(' · ')||c.name);option.value=i;select.append(option);});select.value=String(index);familyButtons.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.lessonFamily===family)));selectLabel.hidden=select.options.length===1;if(key==='scheduling'){policyLabel.hidden=family!=='admission';if(current.policy)policy.value=current.policy;}}
 if(key==='f-batching'){const guide=node('p','batch-comparison','先看 ① 执行前怎样凑批，再比较 ② 生成轮次之间是否换成员。再看 ③ 一条长输入怎样分块、与 Decode 交错。等待窗口与生成轮次使用不同刻度。');head.append(guide);}
 const readout=node('div','motion-readout');section.append(controls,readout);
 const viewport=node('div','motion-viewport');viewport.tabIndex=0;viewport.setAttribute('aria-label','可回看的机制画布，窄屏自动排列');
 const svg=svgNode('svg',{viewBox:'0 0 900 432',role:'img'});viewport.append(svg);section.append(viewport);
 let handoff;
 const scrubLabel=node('label','motion-scrub'),range=node('input');range.type='range';range.min=0;range.step=.001;range.value=0;range.setAttribute('aria-label','教学进度，可回看中间过程');scrubLabel.append(node('span','','教学进度（不是实测耗时）'),range);section.append(scrubLabel);
 const status=node('output','motion-status');status.setAttribute('aria-live','polite');readout.append(status);
 const action=node('p','motion-action');readout.append(action);
 const scrollHint=node('p','motion-scroll-hint','画布较长；可在框内上下滚动查看完整结构。');scrollHint.hidden=true;readout.append(scrollHint);
 function updateScrollHint(){scrollHint.hidden=!(viewport.scrollHeight>viewport.clientHeight+4);}
 const continuation=node('section','motion-continuation');section.append(continuation);
 const boundary=node('p','motion-boundary');if(key==='f-parallel'){const assumption=node('details','parallel-assumptions');assumption.append(node('summary','','教学假设与简化范围'),boundary);section.append(node('p','motion-boundary','两 rank / GPU 的教学示例；手算数值与教学时隙不代表实测性能。'),assumption);}else section.append(boundary);
 const evidence=node('a','motion-evidence','查看本课来源与适用范围 ↓');evidence.href=key.startsWith('f-')?'#evidence':'#source-baseline';section.append(evidence);
 host.before(section);
 const experimentLabel=node('p','motion-experiment-label','独立参数实验 · 下方的条件与进度单独控制，可逐步核对状态、数值和源码 ↓');section.after(experimentLabel);
 let supplement=null;
 if(primaryViews[key]&&!dedicated){
  const [subject,focus]=primaryViews[key];supplement=node('details','motion-supplement');
  supplement.append(node('summary','',`补充过程：${choices[0].title}`));host.after(supplement);supplement.append(section);experimentLabel.remove();
  host.before(node('p','motion-reading-focus',`观察重点 · ${subject}。${focus}`));
 }
 if(!noLegacy&&(key==='f-batching'||(dedicated&&!['dataflow','lifecycle'].includes(key)))){
  const extra=node('details','motion-supplement batch-extra');extra.append(node('summary','',({'f-batching':'进一步实验：Chunked Prefill 与逐轮 Batch 成员','f-hardware':'核对算例：乘加与资源下界','f-parallel':'继续比较：TP 合算、PP 接力、DP 分请求','f-communication':'继续实验：通知与载荷的就绪边界','f-speculation':'核对算例：接受前缀与有效历史','f-serving':'继续实验：正常回程与双实例路由','request-runtime':'完整请求路径：正常、验证失败与两种取消','communication':'深入实验：collective、完成通知与失败回收','advanced-generation':'深入实验：递推状态、检查点与语法约束','serving-operations':'完整服务实验：探活、准入、超时与退出'})[key]||'进一步实验：改变条件、核对数值与其他分支'));
  const extraTitle=integration?.extraTitles[key==='communication'?'communicationEngine':key.replace(/^f-/, '')];if(extraTitle)extra.querySelector('summary').textContent=extraTitle;
  extra.open=new URL(location.href).searchParams.has('fv')||new URL(location.href).searchParams.has('fs');
  host.before(extra);extra.append(host);experimentLabel.remove();
  extra.addEventListener('toggle',()=>{if(!extra.open)host.querySelectorAll('button[aria-pressed="true"]').forEach(b=>{if(b.textContent.includes('暂停'))b.click();});});
 }
 let scenario,clock,tl,stepTimes=[],driver=null,actors=[],trails=[],links=[],lastPhase=-1,playing=false,visual=null;
 function controlsState(){play.textContent=playing?'暂停':'播放过程';play.setAttribute('aria-pressed',String(playing));section.dataset.playing=String(playing);}
 function save(){const url=new URL(location.href);url.searchParams.set('mv',select.value);url.searchParams.set('ms',clock.p.toFixed(3));history.replaceState(null,'',url);}
 function pause(){driver?.kill();driver=null;tl?.pause();playing=false;controlsState();}
 function paint(){
  const p=Math.max(0,Math.min(scenario.frames.length-1,clock.p)),a=Math.floor(p+1e-8),b=Math.min(a+1,scenario.frames.length-1),raw=p-a,u=raw*raw*(3-2*raw),frame=scenario.frames[a],target=scenario.frames[b];
  section.dataset.position=p.toFixed(4);range.value=p;prev.disabled=p<.0001;next.disabled=p>=scenario.frames.length-1-.0001;
  const transitioning=raw>.0001,captionIndex=transitioning?b:a,captionKey=captionIndex+':'+transitioning;
  if(lastPhase!==captionKey){lastPhase=captionKey;status.textContent=`${transitioning?`${a+1} → ${b+1}`:captionIndex+1} / ${scenario.frames.length} · ${scenario.frames[captionIndex].title}${transitioning?'（进行中）':''}`;evidence.href=scenario.frames[captionIndex].evidence||scenario.evidence||(key.startsWith('f-')?'#evidence':'#source-baseline');action.textContent=(transitioning?'完成这一步后：':'')+scenario.frames[captionIndex].caption;}
  svg.setAttribute('aria-label',`${scenario.title}。${action.textContent}`);
  if(visual){visual.paint(frame,target,u,p);return;}
  links.forEach(({path,from,to})=>{const a=frame.actors[from],b=target.actors[from],c=frame.actors[to],d=target.actors[to];const mix=(x,y)=>x+(y-x)*u;const sx=mix(a.x,b.x)+102,sy=mix(a.y,b.y)+33,end=d.opacity>c.opacity?d:c,amount=mix(c.opacity,d.opacity);path.setAttribute('d',`M${sx},${sy} L${sx+(end.x+102-sx)*amount},${sy+(end.y+33-sy)*amount}`);path.setAttribute('opacity',amount);});
  actors.forEach((actor,i)=>{
   const from=frame.actors[i],to=target.actors[i],mix=k=>from[k]+(to[k]-from[k])*u;
   const linked=(scenario.links||[]).some(link=>link.to===scenario.actors[i].id);const {x,y}=linked?(to.opacity>from.opacity?to:from):travel(from,to,u),opacity=mix('opacity'),count=mix('count');
   actor.g.setAttribute('transform',`translate(${x.toFixed(2)} ${y.toFixed(2)})`);actor.g.setAttribute('opacity',opacity.toFixed(3));actor.g.dataset.x=x.toFixed(3);actor.g.dataset.y=y.toFixed(3);actor.g.dataset.count=count.toFixed(3);
   // A value is committed at the endpoint; during work the action explains what is being calculated.
   const text=from.opacity===0?to:from;fitText(actor.label,text.label,180);fitText(actor.note,text.note,180);
   actor.g.setAttribute('aria-label',`${text.label}，${text.note}`);
   actor.bars.forEach((bar,j)=>{bar.setAttribute('width',Math.max(0,Math.min(1,count-j))*18);});
   const moving=!linked&&(Math.abs(to.x-from.x)+Math.abs(to.y-from.y)>2)&&from.opacity>0&&to.opacity>0&&raw>.001;
   trails[i].setAttribute('d',moving?`M${from.x+100},${from.y+29} L${from.x+100},${to.y+29} L${to.x+100},${to.y+29}`:'');
  });
 }
 function load(index){
  syncFamily(Number(index));
  const oldPlacement=viewport.querySelector('[aria-label="PP 部署位置"]')?.value;const oldBoundary=scenario?.boundaryKey;
  pause();tl?.kill();visual?.destroy();visual=null;scenario=choices[index];clock={p:0};lastPhase=-1;title.textContent=scenario.title;intro.textContent=scenario.intro;boundary.textContent=scenario.boundary;range.max=scenario.frames.length-1;
  continuation.replaceChildren();continuation.hidden=true;
  if(key==='f-tokens'){continuation.hidden=false;continuation.append(node('h3','','Detokenization：同一串 ID 也能还原文字'),node('p','',`本例把 [${scenario.data.ids.join(', ')}] 交给同一个 tokenizer 解码，会还原为“${scenario.data.text}”。这里还原的是输入；生成新回答还需要下一课的模型计算与输出选择。`));}
  if(key==='f-attention'){const d=scenario.data,fmt=v=>'['+v.map(n=>n.toFixed(2)).join(', ')+']';continuation.hidden=false;const mlp=node('details');mlp.append(node('summary','','接下来：MLP 怎样使用 Attention 输出'),node('p','',`本例用 h=[a,b,max(0,a−b)]，再输出 [h₀+h₂,h₁+h₂]：${fmt(d.mixed)} → ${fmt(d.hidden)} → ${fmt(d.output)}。这是 2→3→2 维的手算例子，省略残差与归一化；切换查询位置会同步更新这些数值。`));continuation.append(mlp);}
  if(key==='f-parallel'&&scenario.kind==='parallel-tp'){continuation.hidden=false;const nav=node('nav','concept-links');nav.setAttribute('aria-label','单独查看通信操作');for(const [label,index]of [['Broadcast：复制输入',29],['Scatter：分发输入列',30],['AllGather：收齐输出列',31],['AllReduce：贡献求和',32]]){const a=node('a','',label+' ↗');a.href='communication.html?mv='+index+'#mechanism-process';nav.append(a);}continuation.append(nav,node('p','','图中的输入复制与切分说明数据布局；实际 TP 输入可能已在各 rank，或沿用前一层的分片，无需每层都重新广播、分发。'));}
  viewport.querySelector('.motion-handoff')?.remove();svg.dataset.placement='same-host';
  handoff=window.HandoffBoundaries?.mount(viewport,scenario.boundaryKey||key,placement=>{svg.dataset.placement=placement;paint();});
  svg.replaceChildren();viewport.classList.remove('batch-viewport','teaching-viewport');delete svg.dataset.teachingKind;actors=[];trails=[];links=[];evidence.href=scenario.evidence||(key.startsWith('f-')?'#evidence':'#source-baseline');
  if(scenario.kind==='dynamic-batching'){visual=window.DynamicBatchingVisual(svg,viewport,scenario);}
  else if(scenario.kind==='chunk-batching'){visual=window.ChunkBatchingVisual(svg,viewport,scenario);}
  else if(dedicated){visual=dedicated(svg,viewport,scenario);}
  else if(key==='f-batching'&&window.BatchingVisual){visual=window.BatchingVisual(svg,viewport,scenario);}
  else {
  svg.append(svgNode('text',{x:24,y:25,class:'motion-legend'},scenario.legend||'同一编号 = 同一对象 · 虚线 = 本次移动路径 · 小格 = 已生成 / 已处理的位置'));
  scenario.zones.forEach(z=>{const g=svgNode('g',{class:'motion-zone'});g.append(svgNode('rect',{x:z.x,y:z.y||42,width:z.w||270,height:z.h||366,rx:14}),svgNode('text',{x:z.x+16,y:(z.y||42)+27,class:'motion-zone-title'},z.title));if(z.note)g.append(svgNode('text',{x:z.x+16,y:(z.y||42)+49,class:'motion-zone-note'},z.note));svg.append(g);});
  (scenario.links||[]).forEach(link=>{const path=svgNode('path',{class:'motion-reference'});svg.append(path);links.push({path,from:scenario.actors.findIndex(a=>a.id===link.from),to:scenario.actors.findIndex(a=>a.id===link.to)});});
  scenario.actors.forEach(()=>{const path=svgNode('path',{class:'motion-trail'});svg.append(path);trails.push(path);});
  scenario.actors.forEach((def,i)=>{const g=svgNode('g',{'data-motion-id':def.id,class:`motion-object motion-color-${def.color??i%5}`}),label=svgNode('text',{x:12,y:23,class:'motion-object-title'}),note=svgNode('text',{x:12,y:43,class:'motion-object-note'}),bars=[];
   g.append(svgNode('rect',{width:204,height:66,rx:10}),label,note);for(let j=0;j<9;j++){const r=svgNode('rect',{x:12+j*20,y:53,width:0,height:5,rx:1,class:'motion-growth'});g.append(r);bars.push(r);}svg.append(g);actors.push({g,label,note,bars});
  });
  }
  tl=gsap.timeline({paused:true,onUpdate:paint,onComplete(){playing=false;controlsState();save();},defaults:{duration:stepDuration,ease:'none'}});
  stepTimes=[0];
  for(let i=1;i<scenario.frames.length;i++){const requested=scenario.stepDurations?.[i-1],duration=Number.isFinite(requested)&&requested>0?requested:stepDuration;stepTimes.push(stepTimes[i-1]+duration);tl.to(clock,{p:i,duration}).addLabel(`step-${i}`);}
  section.dataset.stepDurations=JSON.stringify(stepTimes.slice(1).map((t,i)=>Number((t-stepTimes[i]).toFixed(3))));
  if(oldBoundary===scenario.boundaryKey&&oldPlacement){const placement=viewport.querySelector('[aria-label="PP 部署位置"]');if(placement){placement.value=oldPlacement;placement.dispatchEvent(new Event("change"));}}
  paint();requestAnimationFrame(updateScrollHint);
 }
 // URL and scrubber positions remain logical steps; playback time may differ
 // by concept. A seek into a step must invert the exact same duration mapping.
 function timeAt(position){
  const p=Math.max(0,Math.min(scenario.frames.length-1,position)),i=Math.floor(p);
  return stepTimes[i]+(i<stepTimes.length-1?(stepTimes[i+1]-stepTimes[i])*(p-i):0);
 }
 function go(target){pause();const dest=Math.max(0,Math.min(scenario.frames.length-1,target));if(reduced.matches){tl.time(timeAt(dest));paint();save();return;}
  if(Math.abs(dest-clock.p)<.0001)return;playing=true;controlsState();driver=tl.tweenTo(timeAt(dest),{duration:Math.abs(timeAt(dest)-timeAt(clock.p)),ease:'none',onComplete(){driver=null;playing=false;controlsState();paint();save();}}).timeScale(Number(speed.value));
 }
 play.addEventListener('click',()=>{if(playing){pause();save();return;}if(reduced.matches){go(Math.floor(clock.p+.0001)+1);return;}if(clock.p>=scenario.frames.length-1-.0001)tl.time(0);playing=true;controlsState();tl.timeScale(Number(speed.value)).play();});
 prev.addEventListener('click',()=>go(Math.ceil(clock.p-.0001)-1));next.addEventListener('click',()=>go(Math.floor(clock.p+.0001)+1));reset.addEventListener('click',()=>{pause();tl.time(0);paint();save();});
 range.addEventListener('input',()=>{pause();const value=Number(range.value);tl.time(timeAt(reduced.matches?Math.round(value):value));paint();save();});
 select.addEventListener('change',()=>{load(Number(select.value));save();});speed.addEventListener('input',()=>{const rate=Number(speed.value);speedValue.textContent=rate+'×';speed.setAttribute('aria-valuetext',rate+' 倍');if(driver)driver.timeScale(rate);else tl.timeScale(rate);});
 reduced.addEventListener('change',()=>{pause();if(reduced.matches){tl.time(timeAt(Math.round(clock.p)));paint();}});
 section.addEventListener('keydown',e=>{if(e.target.matches('input,select,button,a'))return;if(e.key===' '){e.preventDefault();play.click();}if(e.key==='ArrowRight'){e.preventDefault();next.click();}if(e.key==='ArrowLeft'){e.preventDefault();prev.click();}});
 const observer=new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)pause();},{threshold:0});observer.observe(section);
 supplement?.addEventListener('toggle',()=>{if(!supplement.open)pause();});
 active.add({pause});window.addEventListener('resize',()=>requestAnimationFrame(updateScrollHint),{passive:true});if('ResizeObserver'in window)new ResizeObserver(()=>requestAnimationFrame(updateScrollHint)).observe(viewport);
 const params=new URL(location.href).searchParams,requested=Number(params.get('mv'));const legacyMetric=key==='f-metrics'&&!params.has('mv')?Number(params.get('fv')):NaN;const initial=Number.isInteger(legacyMetric)&&legacyMetric>=0&&legacyMetric<choices.length?legacyMetric:params.has('mv')&&Number.isInteger(requested)&&requested>=0&&requested<choices.length?requested:Math.max(0,choices.findIndex(s=>s.default));select.value=String(initial);load(initial);const saved=Number(params.get('ms')??(key==='f-metrics'?params.get('fs'):null));if(Number.isFinite(saved)&&saved>0){tl.time(timeAt(Math.min(saved,scenario.frames.length-1)));paint();}
 if(noLegacy){host.remove();experimentLabel.remove();}

}
const lesson=document.body.dataset.lesson;
if(lesson)mount(document.querySelector('[data-foundation-lab]'),'f-'+lesson);
else{const key=document.body.dataset.page,host=document.querySelector('[data-execution-lab],[data-schedule-lab],[data-memory-lab],[data-system-lab],[data-runtime-lab],[data-parallel-lab],[data-comm-lab],[data-generation-lab],[data-serving-lab],[data-performance-lab],#pipeline-player,#data-routes-overview');if(host&&catalogue[key])mount(host,key);}
document.addEventListener('visibilitychange',()=>{if(document.hidden)active.forEach(p=>p.pause());});window.addEventListener('pagehide',()=>active.forEach(p=>p.pause()));
})();
