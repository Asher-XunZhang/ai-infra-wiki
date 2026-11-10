(()=>{
'use strict';
const M=window.RequestState,host=document.querySelector('[data-request-state]');if(!M||!host)return;
const $=x=>host.querySelector(`[data-role="${x}"]`),esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
const text=(x,y,v,c='',extra='')=>`<text x="${x}" y="${y}" class="${c}" ${extra}>${esc(v)}</text>`;
const rect=(x,y,w,h,c='box')=>`<rect x="${x}" y="${y}" width="${Math.max(0,w)}" height="${h}" rx="6" class="${c}"/>`;
let tl,time=0,playing=false,last=0,raf=0,inspected='',lastFocus='';
function pointers(s){return ['current','running','last'].map(k=>`${k==='current'?'本轮 batch':k+'_batch'} → ${s[k]?s[k]+' {'+(s.members[s[k]]||[]).join(',')+'}':'—'}`);}
const G={consumer:{x:15,y:55,w:310,h:116},consumerResult:{x:245,y:134,w:66,h:28},queue:{x:15,y:398,w:310,h:65},request:{x:380,y:55,w:350,h:408},output:{x:398,y:115,w:310,h:66},device:{x:794,y:398,w:290,h:65}};
const box=(g,c='box',attrs='')=>`<rect x="${g.x}" y="${g.y}" width="${g.w}" height="${g.h}" rx="6" class="${c}" ${attrs}/>`;
const port=(g,side)=>side==='left'?[g.x,g.y+g.h/2]:side==='right'?[g.x+g.w,g.y+g.h/2]:[g.x+g.w/2,g.y+g.h];
function has(s,key){return key==='batch'?Object.values(s.members).some(a=>a.includes('R1')):!!s[key];}
function reference(y,from){const x1=G.consumer.x+G.consumer.w,x2=G.request.x;return `<g data-reference-from="${from}"><path d="M${x1} ${y}H${x2}" class="reference" marker-end="url(#rq-ref)"/>${text((x1+x2)/2,y-5,'引用','reference-label','text-anchor="middle"')}</g>`;}
function flow(points,event,snap,kind,label,source,target){
 const d='M'+points.map(p=>p.join(' ')).join('L');
 // One teaching clock drives the dash phase, including pause, seek and speed.
 // Repeating marks show an active operation, never a packet/token count.
 // Match chapter 2's ia-wireflow: 4/7 dashes, 22 units per .9 seconds.
 const offset=reduced.matches?0:-(snap.t-event.start)*22/.9;
 return `<g data-flow-event="${event.id}" data-flow-kind="${kind}" data-flow-source="${source}" data-flow-target="${target}"><title>${esc(label)}</title><path d="${d}" class="flow-path flow-pulse" marker-end="url(#rq-ref)" stroke-dashoffset="${offset.toFixed(3)}"/></g>`;
}
function main(snap){
 const s=snap.state,gpu=snap.active.find(e=>e.lane==='gpu'&&e.kvTo),gp=gpu?snap.p(gpu.id):0;
 let out='<title>同一个 Req 的容器引用、状态字段与资源归属</title><defs><marker id="rq-ref" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="context-stroke"/></marker></defs>';
 out+=text(16,25,'Scheduler 进程 · CPU 对象与本地引用','heading')+text(798,25,'资源账本 · CPU 管理 / GPU 存储','heading');
 out+=`<path d="M767 42V465" class="boundary"/>`;
 const copy=snap.active.find(e=>e.lane==='copy'),commit=snap.active.find(e=>e.token);
 const cpu=tl.mode==='overlap'?s.consuming:(s.resultReady||copy?{id:'本轮结果',batch:s.current||s.last,ready:s.resultReady,token:s.resultReady?s.deviceToken:null}:null);
 out+=`<g data-region="consumer" data-result-id="${cpu?.id||''}">`+box(G.consumer,cpu?'box consumer-active':'box')+text(27,77,'CPU 当前消费 · 局部 batch / result','code');
 out+=text(27,101,cpu?(tl.mode==='overlap'?`${cpu.id} · ${cpu.forward} · 已从队列取出`:`本轮结果 · ${cpu.batch||'batch'}`):'当前没有待处理结果','value');
 out+=text(27,122,cpu?(cpu.ready?(cpu.token?'结果可读；按请求状态处理':'中间块不提交有效输出'):'等待输出可读 / copy_done'):'等待本轮处理入口','small');
 out+=text(27,153,'CPU 可读结果','small')+box(G.consumerResult,cpu?.ready?'result-token':'pending','data-consumer-result="true"')+text(278,153,cpu?.ready?(cpu.token||'—'):'—','value','text-anchor="middle"')+'</g>';
 if(cpu)out+=reference(98,'consumer');
 const rows=[['waiting',187,43,'waiting_queue',s.waiting?'[ ref → R1 ]':'[ ]'],['chunk',242,43,'chunked_req',s.chunk?'ref → R1':'None'],['batch',297,89,'ScheduleBatch 与变量引用','']];
 rows.forEach(([key,y,h,label,value])=>{
  const on=has(s,key);
  out+=rect(15,y,310,h,has(s,key)?'box owned':'box')+text(27,y+20,label,'code');
  if(key==='batch')pointers(s).forEach((v,j)=>{out+=text(27,y+42+j*19,v,'small code');});
  else out+=text(27,y+37,value,'small');
  if(on)out+=reference(y+h/2,key);
 });
 const queuedCards=s.results.map((q,i)=>({q,geometry:{x:27+i*142,y:432,w:132,h:23}}));
 out+=`<g data-region="result-queue">`+box(G.queue,s.results.length?'box owned':'box')+text(27,420,'result_queue · 尚未取出的结果项','code');
 queuedCards.forEach(({q,geometry:g})=>{out+=`<g data-queued-result="${q.id}">`+box(g,'pending')+text(g.x+7,448,`${q.id} · ${q.forward} · ${q.ready?'可读':'在途'}`,'small')+'</g>';});
 if(!queuedCards.length)out+=text(27,446,tl.mode==='overlap'?'[ ] · 队列为空':'Normal loop 不使用 result_queue','small');
 out+='</g>';if(queuedCards.length)out+=reference(430,'result-queue');
 out+=box(G.request,'request')+text(398,81,'R1 · 同一个 Req 对象','heading');
 out+=text(398,104,'rid / input / output / finish / KV 账本','small');
 out+=box(G.output,commit?'output-field is-committing':'output-field','data-output-field="true"')+text(414,132,'output_ids','code');
 s.output.forEach((v,i)=>{out+=rect(414+i*64,144,56,31,s.finished==='ABORT'&&i===s.output.length-1?'token abort':'token')+text(442+i*64,165,v,'value','text-anchor="middle"');});
 if(!s.output.length)out+=text(414,165,'[ ] · 尚未提交有效输出','small');
 const fields=[['finished_reason',s.finished||'None'],['to_finish',s.pending||'None'],['is_retracted',String(s.retracted)]];
 fields.forEach(([k,v],i)=>{out+=text(398,204+i*28,k,'code')+text(610,204+i*28,v,v==='None'||v==='false'?'small':'value');});
 out+=text(398,300,'原始输入进度','small')+text(635,300,`${s.progress} / ${s.prompt}`,'small');
 out+=rect(398,312,310,9,'pending');const progress=gpu?Math.min(s.prompt,s.progress+(Math.min(gpu.kvTo,s.prompt)-s.progress)*gp):s.progress;
 out+=`<rect x="398" y="312" width="${310*progress/s.prompt}" height="9" rx="4" class="progress" data-input-progress="${progress.toFixed(2)}"/>`;
 out+=text(398,349,s.finished?'终态与引用清理分开观察':s.chunk?'续算指针与 batch 可同时引用 R1':'对象身份不随入队、选批或回撤改变','small');
 const owners=[...(s.waiting?['waiting_queue']:[]),...(s.chunk?['chunked_req']:[]),...Object.keys(s.members).filter(k=>s.members[k].includes('R1')),...s.results.map(q=>q.id+'（队列）'),...(s.consuming?[s.consuming.id+'（CPU）']:[])];
 out+=text(398,391,'观察到的持有者（教学汇总）','small');
 out+=text(398,414,owners.slice(0,3).join(' / ')||'这些容器已无 R1 引用','small');
 out+=text(398,436,owners.slice(3).join(' / '),'small');
 out+=rect(794,55,290,66,'box')+text(808,78,'ReqToTokenPool · 请求行','code')+text(808,104,s.row?`${s.row} → R1 的位置索引`:'未持有请求行','value');
 if(s.row)out+=`<path d="M730 91H794" class="resource-link" marker-end="url(#rq-ref)"/>`;
 out+=rect(794,136,290,193,'box')+text(808,160,'GPU KV · 每格是一个位置','heading');
 for(let i=0;i<8;i++){
  const x=808+i%4*66,y=175+Math.floor(i/4)*57,filled=i<s.written||i<s.cached,allocated=i<s.allocated;
  out+=rect(x,y,57,33,filled?(i<s.cached?'kv cache':'kv owned'):allocated?'pending allocated':'pending');
  if(gpu&&i>=s.written&&i<gpu.kvTo){const f=reduced.matches?0:Math.max(0,Math.min(1,(gpu.kvTo-s.written)*gp-(i-s.written)));out+=`<rect x="${x+2}" y="${y+31-29*f}" width="53" height="${29*f}" rx="4" class="progress" data-kv-write="${i}"/>`;}
  const inputToken=i<s.prompt?'p'+i:(s.output[i-s.prompt]||(tl.mode==='overlap'&&i===5?'EOS':'t'+(i-s.prompt+1)));
  out+=text(x+28,y+23,filled?inputToken:allocated?'待写':'空','small','text-anchor="middle"');
  out+=text(x+28,y+48,'位置 '+i,'tiny','text-anchor="middle"');
 }
 out+=text(808,308,s.protected?'缓存前缀：R1 仍持有保护':s.cached?'缓存前缀：已解锁，可驱逐':'本例没有保留缓存前缀','small');
 out+=box(G.device)+text(808,420,'GPU 设备输出','heading')+text(808,446,s.deviceToken||'尚未产生','value');
 if(copy&&s.deviceToken!==null){
  const queued=queuedCards.find(c=>c.q.id===copy.resultId),start=port(G.device,'bottom');
  const points=queued?[start,[start[0],475],[port(queued.geometry,'bottom')[0],475],port(queued.geometry,'bottom')]:[start,[start[0],475],[4,475],[4,148],[G.consumer.x,148]];
  const target=queued?queued.q.id:'consumer';
  out+=flow(points,copy,snap,'copy',`设备输出 → ${queued?queued.q.id+' 队列结果':'CPU 当前消费的结果'}；可读后才允许提交`,'device',target);
 }
 if(commit){out+=flow([port(G.consumerResult,'right'),port(G.output,'left')],commit,snap,'commit',`${cpu?.id||'本轮结果'} → Req.output_ids；本阶段结束时提交`,'consumer-result','output-ids');out+=text(353,140,'提交','flow-label','text-anchor="middle"');}
 out+=text(16,488,'S0 / S1：batch 对象；B0 / B1：执行轮次；Q0 / Q1：结果项。灰线只表示引用，橙色脉冲表示交接或提交。','small');
 $('diagram').innerHTML=out;
 const resource=s.row?`请求持有 ${s.row}；已预留 ${s.allocated} 个位置，完成写入 ${s.written} 个。${s.protected?'缓存前缀仍受 R1 保护。':''}`:s.resourceNote;
 $('resource').textContent=resource;
 $('state').textContent=s.terminal+(s.other?' · '+s.other:'');
 $('accessible-state').textContent=`R1：${s.terminal}；waiting=${s.waiting}；chunked_req=${s.chunk}；${pointers(s).join('；')}；队列=[${s.results.map(q=>q.id)}]；CPU 当前消费=${cpu?.id||'无'}；CPU 可读结果=${cpu?.token||'无'}；output_ids=[${s.output}]；finished_reason=${s.finished||'None'}；to_finish=${s.pending||'None'}；${resource}`;
}
function tracks(snap){
 const labels={cpu:'Scheduler · CPU',gpu:'GPU · forward',copy:tl.mode==='overlap'?'输出拷贝 · copy stream':'结果可读过程'};
 const x=t=>168+t/tl.duration*908,ys={cpu:46,gpu:91,copy:136};let out='<title>同一教学时钟下的执行轨道</title>';
 for(let t=0;t<=tl.duration;t+=5)out+=text(x(t),22,t+'u','tiny','text-anchor="middle"')+`<path d="M${x(t)} 30V179" class="grid"/>`;
 Object.entries(labels).forEach(([k,v])=>{out+=text(8,ys[k]+21,v,'small')+rect(168,ys[k],908,31,'track-bg');});
 const selected=tl.events.find(e=>e.id===inspected)||snap.focus;
 for(const e of tl.events){const a=x(e.start),w=x(e.end)-a,y=ys[e.lane],p=snap.p(e.id);let short=e.batch|| (e.kind==='wait'?'等结果':e.lane==='copy'?'回传':e.token?'提交':e.title.slice(0,4));if(w<25)short='';else if(w<48)short=short.slice(0,2);
  out+=`<g role="button" tabindex="0" data-event="${e.id}" class="track-event ${e.lane} ${e.kind||''} ${selected.id===e.id?'selected':''}" aria-label="${esc(e.title+'；'+e.start.toFixed(1)+' 至 '+e.end.toFixed(1)+' 教学单位，查看依赖')}"><title>${esc(e.title)}</title>`+rect(a,y,w,31,'track-outline')+rect(a,y,w*p,31,'track-fill')+text(a+w/2,y+21,short,'tiny','text-anchor="middle"')+'</g>';
 }
 out+=`<path d="M${x(time)} 31V181" class="cursor"/>`;
 $('tracks').innerHTML=out;$('tracks').dataset.clock=time.toFixed(3);
 $('dependency').textContent=`${selected.title}；前提：${(selected.depends||[]).map(id=>tl.events.find(e=>e.id===id)?.title).filter(Boolean).join('、')||'本场景初始条件'}。`;
}
function render(){const s=M.snapshot(tl,time);host.dataset.mode=tl.mode;host.dataset.condition=tl.condition;host.dataset.time=time.toFixed(3);host.dataset.playing=String(playing);
 $('clock').value=time;$('clock').max=tl.duration;$('clock-value').textContent=`${time.toFixed(1)} / ${tl.duration.toFixed(1)} u`;
 $('play').textContent=playing?'暂停':time>=tl.duration?'重播':'从这里播放';$('prev').disabled=time<=0;$('next').disabled=time>=tl.duration;
 $('question').textContent=tl.question;$('scope').textContent=tl.scope;$('cancel-choice').hidden=tl.mode!=='cancel';
 host.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===tl.mode)));
 main(s);tracks(s);
 depth.update({time,body:DepthProjections.req(tl,s),event:s.focus.title,explanation:s.focus.detail,href:M.url(s.focus.source),rows:[['finished_reason',s.state.finished||'None'],['to_finish',s.state.pending||'None'],['is_retracted',s.state.retracted],['batch 引用',pointers(s.state).join('；')],['CPU 消费结果',s.state.consuming?.id||'无'],['result_queue',s.state.results.map(q=>q.id).join(', ')||'空']]});
 $('event-title').textContent=s.focus.title;$('event-detail').textContent=s.focus.detail;$('source').href=M.url(s.focus.source);$('source').textContent=M.sources[s.focus.source][0].split('/').at(-1)+':'+M.sources[s.focus.source][1]+' ↗';
 if(lastFocus!==s.focus.id){lastFocus=s.focus.id;$('announce').textContent=s.focus.title;}
 const nearest=tl.events.filter(e=>e.lane==='cpu'&&e.end<=time).at(-1);$('steps').value=nearest?.id||'start';
}
function save(push=false){const u=new URL(location.href);u.searchParams.set('mode',tl.mode);u.searchParams.set('case',tl.condition);u.searchParams.set('t',time.toFixed(3));history[push?'pushState':'replaceState']({},'',u);}
function pause(){playing=false;cancelAnimationFrame(raf);last=0;}
function seek(t){pause();time=Math.max(0,Math.min(tl.duration,Number(t)||0));render();save();}
function configure(mode,condition,fromURL=false,t=0){pause();tl=M.timeline(mode,condition);inspected='';lastFocus='';time=Math.max(0,Math.min(tl.duration,Number(t)||0));$('condition').value=tl.condition;$('steps').innerHTML='<option value="start">场景起点</option>'+tl.events.filter(e=>e.lane==='cpu').map(e=>`<option value="${e.id}">${esc(e.title)}</option>`).join('');render();if(!fromURL)save(true);}
function tick(ts){if(!playing)return;if(last)time=Math.min(tl.duration,time+(ts-last)/1000*Number($('speed').value));last=ts;if(time>=tl.duration)pause();render();if(playing)raf=requestAnimationFrame(tick);else save();}
function readURL(){const u=new URL(location.href);configure(u.searchParams.get('mode')||'normal',u.searchParams.get('case')||'waiting',true,u.searchParams.get('t'));}
host.addEventListener('click',e=>{const b=e.target.closest('button[data-mode]');if(b){if(b.dataset.mode!==tl.mode)configure(b.dataset.mode,tl.condition);return;}const bar=e.target.closest('[data-event]');if(bar){inspected=bar.dataset.event;tracks(M.snapshot(tl,time));}});
host.addEventListener('keydown',e=>{const bar=e.target.closest('[data-event]');if(bar&&['Enter',' '].includes(e.key)){e.preventDefault();inspected=bar.dataset.event;tracks(M.snapshot(tl,time));host.querySelector(`[data-event="${inspected}"]`)?.focus();}});
$('condition').addEventListener('change',()=>configure(tl.mode,$('condition').value));
$('play').addEventListener('click',()=>{if(playing){pause();render();save();}else{if(time>=tl.duration)time=0;playing=true;last=0;render();raf=requestAnimationFrame(tick);}});
$('prev').addEventListener('click',()=>seek([...new Set([0,...tl.events.map(e=>e.end)])].sort((a,b)=>b-a).find(t=>t<time-.01)??0));
$('next').addEventListener('click',()=>seek([...new Set(tl.events.map(e=>e.end))].sort((a,b)=>a-b).find(t=>t>time+.01)??tl.duration));
$('reset').addEventListener('click',()=>seek(0));$('clock').addEventListener('input',()=>seek($('clock').value));$('steps').addEventListener('change',()=>seek(tl.events.find(e=>e.id===$('steps').value)?.end||0));
const depth=LearningDepth.mount({host,before:$('diagram').parentElement,module:[$('diagram').parentElement],onChange:()=>render(),persist:()=>save()});
window.addEventListener('popstate',readURL);window.addEventListener('pagehide',pause);document.addEventListener('visibilitychange',()=>{if(document.hidden){pause();render();save();}});reduced.addEventListener('change',render);readURL();
})();
