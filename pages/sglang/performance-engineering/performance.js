(()=>{'use strict';
const M=window.PerformancePD,root=document.querySelector('[data-performance-pd]');if(!M||!root)return;
const $=r=>root.querySelector(`[data-role="${r}"]`),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const text=(x,y,v,c='',anchor='start')=>`<text x="${x}" y="${y}" class="${c}" text-anchor="${anchor}">${esc(v)}</text>`,rect=(x,y,w,h,c='',attrs='')=>`<rect x="${x}" y="${y}" width="${Math.max(0,w)}" height="${h}" rx="5" class="${c}" ${attrs}/>`;
const reduced=matchMedia('(prefers-reduced-motion: reduce)');let tl,time=0,playing=false,last=0,raf,selected,inspected='',view='overview';
const fmt=n=>Number(n.toFixed(1)).toString();
const compact={permit:'并发许可', 'client-send':'HTTP',dispatch:'派发','p-input':'登记','d-input':'登记','p-prepare':'入队','p-bootstrap':'等 D 地址','d-prepare':'入队','d-bootstrap':'分配 / 通知','p-queue':'排队','p-submit':'提交','p-gpu1':'块 1','p-gpu2':'块 2','p-chunk1-result':'结果','kv1':'块 1 KV','p-result':'首 token','p-send':'send','kv-last':'KV + 结果','p-poll':'退役','p-observe':'采集','p-out':'完成','p-gate':'P 结果','d-receive':'等 KV / 轮询提交','d-wait':'排队','d-prebuilt':'首 token','d-gpu2':'t2','d-gpu3':'t3','d-gpu4':'t4',tail:'DONE'};
function brief(e,w){const label=compact[e.id]||(/^(tm|client)-/.test(e.id)?e.payload.match(/n=(\d+)/)?.[1]:e.label)||e.label;return /^\d+$/.test(label)?label:w<24?'':short(label,w);}

function short(v,width){let out='',n=0;for(const c of v){n+=/[\x00-\xff]/.test(c)?5.8:11;if(n>width-16)return out+'…';out+=c;}return out;}
const simpleTitles={clock:'客户端等待首 token，时间花在哪里？',kv:'等 KV 就绪，就是一直在搬运吗？',stream:'输出一批 token，时间怎样被统计？',sample:'慢请求失败后，P95 为什么反而下降？'};
const metricNames={'d-transfer':'D 等待接收','p-transfer':'P 发送后的等待','kv-latency':'KV 传输统计','p-forward':'P 计算阶段','p-bootstrap':'P 等待接收地址','d-bootstrap':'D 接收准备','d-queue':'D 准入等待',fake:'D 首 token 回告','client-e2e':'客户端总时长','d-e2e':'D 本地总时长','p-e2e':'P 本地总时长','d-ttft':'D 前端 TTFT','p-ttft':'P 的 TTFT 样本','d-itl':'D 输出间隔 ITL'};
const metricHelp={
 'client-ttft':'从客户端发出请求，到收到首个非空内容。等待许可在这段计时之前。',
 'd-ttft':'从 D 前端登记请求，到收到首批输出。D 已收到结果时，客户端可能仍在等待。',
 'p-ttft':'这个版本的 P 端不记录这项 TTFT 样本；缺失不是零耗时。',
 'client-e2e':'从客户端发出请求，到收到流尾。最后一个 token 到达后，计时仍可能继续。',
 'd-e2e':'D 前端从登记到收到完成结果。它与 P 的本地时段有交叠。',
 'p-e2e':'P 前端从登记到收到完成结果。P 结束时，D 可能仍在生成。',
 'd-transfer':'接收位置准备好，到 D 确认 KV 可用。这段可能包含等 P 计算、真正搬运以及本地检查。',
 'p-transfer':'最后一块发送后，到 P 确认传输完成。前面块的完整搬运不在这个区间内。',
 'kv-latency':'本例统计最后一块发送后的收尾区间，不覆盖整条请求的全部 KV 搬运。',
 'p-forward':'P 开始处理这一轮，到取得 Prefill 结果。展开模块过程可分辨 CPU 和 GPU 的活动。',
 'p-bootstrap':'P 等待 D 的接收地址及相关准备条件；这时 D 在独立推进。',
 'd-bootstrap':'D 分配接收位置，再将信息通知给 P。此时尚未收到 KV 数据。',
 'd-queue':'D 提交接收结果后，还要等待进入处理批次。前面的接收等待没有计在这里。',
 fake:'D 回告 P 已经产生的首 token；这里没有再做一次 Prefill GPU 计算。',
 'd-itl':'两次回告的间隔，除以本次新增 token 数。收到一批输出，不代表它们在 GPU 上匀速产生。',
 'client-itl':'客户端按两批内容的间隔和新增 token 数记录 ITL。它的采样位置与 D 前端不同。',
 'client-tpot':'首个内容到流结束的时间，分摊给后续 token；结束后再汇总。只有一个 token 时没有这项样本。',
 p95:'只对成功请求统计。慢请求失败退出集合，P95 也可能下降。',throughput:'成功输出的 token 数，除以整批请求的观察窗口。失败前见到的内容不计入成功输出。'};
const choices={clock:['client-ttft','d-ttft','client-e2e'],kv:['d-transfer','p-transfer','kv-latency','p-forward'],stream:['d-itl','client-itl','client-tpot'],sample:['p95','throughput']};
const metricLabel=m=>view==='source'?m.name:metricNames[m.id]||m.name;
const plot=()=>view==='overview'?M.overview(tl):{lanes:tl.mode==='sample'?{R1:'R1 · 客户端请求',R2:'R2 · 客户端请求',R3:'R3 · 客户端请求',R4:'R4 · 客户端请求',records:'原始 outputs[]',aggregate:'汇总 CPU'}:M.lanes,events:tl.events};
function controls(){
 const metrics=view==='source'?tl.metrics:tl.metrics.filter(m=>choices[tl.mode].includes(m.id)||m.id===selected);
 $('metric').innerHTML=metrics.map(m=>`<option value="${m.id}">${esc(metricLabel(m)+(view==='source'?' · '+m.owner:''))}</option>`).join('');$('metric').value=selected;
 $('events').innerHTML=plot().events.slice().sort((a,b)=>a.start-b.start).map(e=>`<option value="${e.id}">${esc(e.label)}</option>`).join('');
}
function setView(value,remember=true){view=['overview','modules','source'].includes(value)?value:'overview';inspected='';$('options').open=view!=='overview';controls();render();if(remember)save();}
function chart(s){
 const sample=tl.mode==='sample',projection=plot(),keys=Object.keys(projection.lanes),names=projection.lanes,spacing=view==='overview'?44:36;
 const x=t=>196+t/tl.duration*878,ys=Object.fromEntries(keys.map((k,i)=>[k,38+i*spacing])),bottom=keys.length*spacing+62,h=bottom+92;
 $('diagram').setAttribute('viewBox',`0 0 1100 ${h}`);$('diagram').dataset.clock=time.toFixed(2);
 let svg='<title>P/D 模块的活动与采样区间共用教学时钟</title><defs><marker id="pm-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="context-stroke"/></marker></defs>';
 for(let t=0;t<=tl.duration;t+=100)svg+=text(x(t),18,t,'tick','middle')+`<path d="M${x(t)} 26V${bottom-10}" class="grid"/>`;
 for(const k of keys){const y=ys[k];svg+=rect(8,y-2,1082,32,k==='P'||k==='ptm'||k==='ps'||k==='pg'?'lane p':(k==='D'||k.startsWith('d'))?'lane d':'lane')+text(14,y+18,names[k],'lane-name');}
 for(const e of projection.events){const y=ys[e.lane],left=x(e.start),w=x(e.end)-left,active=e.start<=time&&time<e.end,p=s.progress(e);svg+=`<g class="activity ${e.kind} ${active?'active':''}" data-event="${e.id}" role="button" tabindex="0" aria-label="${esc(e.label+'；'+e.start+' 至 '+e.end+' 示例毫秒；查看依赖')}"><title>${esc(e.label)}</title>`+rect(left,y,w,28,'outline')+rect(left,y,w*p,28,'fill')+(w>=8?text(left+w/2,y+16,(view==='overview'?short(e.label,w):brief(e,w)),'bar-text','middle'):'');
  if(e.kind==='data'&&active&&w>8){const off=reduced.matches?0:-(time-e.start)*.3;svg+=`<path d="M${left+2} ${y+24}H${left+w-3}" class="pulse" stroke-dashoffset="${off.toFixed(2)}" marker-end="url(#pm-arrow)" data-pulse="${e.id}"/>`;}
  svg+='</g>';
 }
 if(sample){for(const r of tl.facts.rows)if(time>=r.ttft){const y=ys[r.id];svg+=`<circle cx="${x(r.ttft)}" cy="${y+25}" r="4" class="stamp"/>`;}}
 const m=s.observations.find(m=>m.id===selected),closed=time>=m.end,started=time>=m.start,painted=Math.min(Math.max(time,m.start),m.end),y=bottom+6;
 let ownerLane=tl.mode==='sample'?null:(m.owner==='native adapter'||m.owner.startsWith('客户端'))?'client':m.owner.startsWith('D Tokenizer')?'dtm':m.owner.startsWith('P Tokenizer')?'ptm':m.owner.startsWith('D Scheduler')?'ds':'ps';
 if(view==='overview'&&ownerLane)ownerLane=ownerLane==='client'?'client':ownerLane.startsWith('d')?'D':'P';
 if(ownerLane&&!m.absent)for(const [t,edge]of [[m.start,'start'],[m.end,'end']])svg+=`<path d="M${x(t)} ${ys[ownerLane]+28}V${y}" class="endpoint-guide"/><circle cx="${x(t)}" cy="${ys[ownerLane]+28}" r="3.5" class="stamp" data-endpoint="${edge}"/>`;
 svg+=text(14,y+14,'选中指标的统计区间','lane-name')+rect(x(m.start),y,x(m.end)-x(m.start),25,m.absent?'measure absent':'measure',`data-measure="${m.id}"`);
 if(!m.absent&&started)svg+=rect(x(m.start),y,x(painted)-x(m.start),25,'measure-fill');
 svg+=m.end>m.start?text(x(m.start),y+44,String(m.start),'tick')+text(x(m.end),y+44,String(m.end),'tick','end'):text(x(m.start),y+44,'无后续区间','tick','middle');
 svg+=`<path d="M${x(time)} 25V${y+30}" class="cursor"/>`;
 svg+=text(196,y+70,m.absent?'本条件没有这项样本；缺失不是零耗时。':`${metricLabel(m)}：${m.status==='已采集'?fmt(m.value)+' '+(m.unit||'ms'):tl.mode==='sample'?'等待汇总':'区间 '+fmt(m.elapsed)+' ms · '+m.status}${view==='source'&&m.observe!==m.end?'（采集点 '+m.observe+'）':''}`,'tick');
 $('diagram').innerHTML=svg;
 const active=projection.events.filter(e=>e.start<=time&&time<e.end),guide=active.find(e=>e.lane==='kv')||active.find(e=>e.id==='o-p-work')||active.find(e=>e.id==='o-d-work')||active.find(e=>e.lane!=='client')||active[0];
 let e=projection.events.find(e=>e.id===inspected)||(view==='overview'?(guide||projection.events.filter(e=>e.end<=time).at(-1)||projection.events[0]):s.focus);if(view==='overview'&&sample&&time>=600)e={...tl.events.find(e=>e.id==='aggregate'),label:time>=640?'统计完成':'汇总成功请求',detail:'原始记录都保留；只有成功请求进入本次延迟和输出统计。'};$('event-title').textContent=e.label;$('detail').textContent=e.detail;$('event-source').href=M.url(e.source);$('event-source').textContent=M.sources[e.source][0].split('/').at(-1)+':'+M.sources[e.source][1]+' ↗';$('events').value=e.id;
 $('dependency').textContent=e.label+' · 开始前提：'+((e.depends||[]).map(id=>tl.events.find(e=>e.id===id).label).join('；')||'本例给定条件 / 所在执行主体的独立推进')+(e.payload?'；载荷：'+e.payload:'');
}
function readings(s){
 const m=s.observations.find(m=>m.id===selected),done=m.status==='已采集';$('measure-name').textContent=metricLabel(m);$('measure-owner').textContent=m.owner;$('measure-value').textContent=m.absent?'无本次样本':done?`${fmt(m.value)} ${m.unit||'ms'}`:m.value!==undefined?(tl.mode==='sample'?'待汇总':`区间 ${fmt(m.elapsed)} ms · ${m.status}`):`${fmt(m.elapsed)} ms · ${m.status}`;
 $('formula').textContent=m.formula;$('measure-detail').textContent=view==='source'?m.detail:m.absent?'本条件没有这项统计样本。展开源码细节可查看跳过条件。':metricHelp[m.id]||m.detail;$('measure-wire').textContent=m.wire||'客户端报告字段';$('measure-source').href=M.url(m.source);$('measure-source').textContent=M.sources[m.source][0].split('/').at(-1)+':'+M.sources[m.source][1]+' ↗';
 let cards='';const card=(title,body)=>`<div><strong>${esc(title)}</strong><p>${esc(body)}</p></div>`;
 if(tl.mode==='sample'){
  const result=tl.facts.result,records=tl.facts.rows.filter(r=>time>=r.end+16),finished=time>=640;
  cards+=card('原始记录',records.map(r=>`${r.id} TTFT=${r.ttft}ms ${tl.condition==='failure'&&r.id==='R4'?'失败 ×（排除）':'成功 ✓'}`).join('；')||'仍在等待请求结果');
  cards+=card('成功集合',finished?`${result.completed}/4 成功；${result.failed} 失败；P95=${fmt(result.p95)}ms；有效输出 ${result.outputTokens} token`:'先保留原始结果，到 calculate_metrics 时才形成最终分布。');
  cards+=card('P/D 聚合边界','R1 正常输出 4 token：P 本地计 1、D 本地计 4，不能相加当客户端输出 5。保留 engine_type 与实例来源。');
 }else{
  const f=tl.facts,tokens=time<f.pFinished?0:1,serverN=s.tmOut.at(-1)?.n||0,clientN=s.clientOut.at(-1)?.n||0;
  cards+=card('同一 R1 的输出进度',`P 已采样 ${tokens}；D TM 已收到累计 ${serverN}；客户端已收到累计 ${clientN}。计数位置不同，不相加。`);
  if(tl.mode==='stream'){
   cards+=card('D TM / 客户端 ITL',`D TM：[${s.tmItl.map(fmt).join(', ')}] ms；客户端：[${s.clientItl.map(fmt).join(', ')}] ms。Δt/Δn 的每个值记 Δn 次。`);
   cards+=card('客户端 TPOT',time<f.end?'等 latency 与最终输出数确定后计算。':f.one?'输出仅 1 token：本请求不进入 TPOT 样本。':`(${fmt(f.end-f.st)} − ${fmt(f.first-f.st)}) / 3 = ${fmt(s.tpot)} ms；[DONE] 尾部不产生 ITL。`);
  }else if(tl.mode==='kv'){
   cards+=card('P 端专用 KV 指标',time<f.pCompletion?'成功退役后才读取 sender 指标。':f.kvBytes===null?'total_bytes=None → 本次不上报；不是 0 ms。':`字节数 64 MiB；回退 latency=${f.pCompletion-f.pTransfer}ms；分母只覆盖最后块尾部，不能用来还原全部通信。`);
   cards+=card('设备工作与统计窗口',f.chunked?'块 1 KV 与块 2 GPU 同时推进；P 的末块统计不覆盖块 1 完整搬运。':'D transfer 计时从接收地址发布后开始，P 尚有计算要做；它不是 NIC 的专用计时器。');
  }else{
   cards+=card('P 与 D 的时间不能相加',`P 本地 E2E：${time>=f.pTM?fmt(f.pTM-f.pBorn)+'ms':'未闭合'}；D 本地 E2E：${time>=f.dFinished?fmt(f.dFinished-f.dBorn)+'ms':'未闭合'}。两个区间在图上交叠。`);
   cards+=card('秒表边界',`许可等待 ${f.st}ms 在客户端 st 前。D TTFT ${time>=f.dFirst?fmt(f.dFirst-f.dBorn)+'ms':'待采样'}；客户端 TTFT ${time>=f.first?fmt(f.first-f.st)+'ms':'待采样'}。差值不只代表网络。`);
  }
 }
 $('readings').innerHTML=cards;$('accessible').textContent=JSON.stringify({time,observations:s.observations.map(m=>({id:m.id,status:m.status,value:m.status==='已采集'?m.value:null})),tmOut:s.tmOut,clientOut:s.clientOut,tmItl:s.tmItl,clientItl:s.clientItl,tpot:s.tpot});
}
function render(){const s=M.snapshot(tl,time);root.dataset.mode=tl.mode;root.dataset.view=view;root.dataset.condition=tl.condition;root.dataset.time=time.toFixed(2);root.dataset.playing=String(playing);$('title').textContent=view==='overview'?simpleTitles[tl.mode]:tl.question;$('depth-hint').textContent=view==='overview'?'关键阶段与一个指标':view==='modules'?'展开模块，查看谁在处理、谁在等待':'查看打点、公式与固定源码';$('options-summary').textContent=tl.cases[tl.condition]+' · '+metricLabel(tl.metrics.find(m=>m.id===selected))+' · 更换';$('legend').textContent=view==='overview'?'横向看先后，纵向看并行；金色量尺是指标区间。阶段条含内部处理与等待，时间为教学示例。':'蓝色：GPU 活动；紫色：数据交接；虚框：等待。金色量尺连接采集端点，时间为教学示例。';root.querySelectorAll('button[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===view)));$('scope').textContent=tl.scope;$('clock').max=tl.duration;$('clock').value=time;$('time').textContent=`${fmt(time)} / ${tl.duration} ms（示例）`;$('play').textContent=playing?'暂停':time>=tl.duration?'重播':'播放';$('prev').disabled=time<=0;$('next').disabled=time>=tl.duration;root.querySelectorAll('button[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===tl.mode)));chart(s);readings(s);}
function stop(){playing=false;cancelAnimationFrame(raf);last=0;}
function save(push=false){const u=new URL(location.href);for(const [k,v]of Object.entries({view,mode:tl.mode,condition:tl.condition,metric:selected,t:time.toFixed(2)}))u.searchParams.set(k,v);history[push?'pushState':'replaceState']({},'',u);}
function seek(t){stop();time=Math.max(0,Math.min(tl.duration,Number(t)||0));inspected='';render();save();}
function locate(id){const e=plot().events.find(e=>e.id===id);if(!e)return;stop();time=e.start;inspected=id;render();save();}
function configure(mode,condition,t=0,metric,fromURL=false){stop();const aliases={packet:'stream',experiment:'clock'};tl=M.timeline(aliases[mode]||mode,condition);time=Math.max(0,Math.min(tl.duration,Number(t)||0));selected=tl.metrics.some(m=>m.id===metric)?metric:tl.defaultMetric;inspected='';$('condition').innerHTML=Object.entries(tl.cases).map(([k,v])=>`<option value="${k}">${v}</option>`).join('');$('condition').value=tl.condition;if(fromURL)$('options').open=view!=='overview';controls();render();if(!fromURL)save(true);}
function read(){const u=new URL(location.href);view=['overview','modules','source'].includes(u.searchParams.get('view'))?u.searchParams.get('view'):'overview';configure(u.searchParams.get('mode')||'clock',u.searchParams.get('condition'),u.searchParams.get('t'),u.searchParams.get('metric'),true);}
function tick(ts){if(!playing)return;if(last)time=Math.min(tl.duration,time+(ts-last)*.04*Number($('speed').value));last=ts;if(time>=tl.duration)stop();render();if(playing)raf=requestAnimationFrame(tick);else save();}
root.addEventListener('click',e=>{const level=e.target.closest('button[data-view]');if(level){setView(level.dataset.view);return;}const b=e.target.closest('button[data-mode]');if(b){if(b.dataset.mode!==tl.mode)configure(b.dataset.mode);return;}const bar=e.target.closest('[data-event]');if(bar)locate(bar.dataset.event);});
root.addEventListener('keydown',e=>{const bar=e.target.closest('[data-event]');if(bar&&['Enter',' '].includes(e.key)){e.preventDefault();const id=bar.dataset.event;locate(id);root.querySelector(`[data-event="${id}"]`)?.focus();}});
$('condition').addEventListener('change',()=>configure(tl.mode,$('condition').value));$('metric').addEventListener('change',()=>{selected=$('metric').value;render();save();});$('events').addEventListener('change',()=>locate($('events').value));$('clock').addEventListener('input',()=>seek($('clock').value));$('reset').addEventListener('click',()=>seek(0));
$('prev').addEventListener('click',()=>seek([0,...plot().events.map(e=>e.end)].sort((a,b)=>b-a).find(t=>t<time-.1)??0));$('next').addEventListener('click',()=>seek(plot().events.map(e=>e.end).sort((a,b)=>a-b).find(t=>t>time+.1)??tl.duration));
$('play').addEventListener('click',()=>{if(playing){stop();render();save();}else{if(time>=tl.duration)time=0;playing=true;inspected='';last=0;render();raf=requestAnimationFrame(tick);}});
window.addEventListener('popstate',read);window.addEventListener('pagehide',stop);document.addEventListener('visibilitychange',()=>{if(document.hidden){stop();render();save();}});reduced.addEventListener('change',render);read();
})();
