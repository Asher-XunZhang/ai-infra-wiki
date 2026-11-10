(()=>{
'use strict';
const M=window.GenerationExecution,host=document.querySelector('#generation-execution');if(!M||!host)return;
const $=r=>host.querySelector(`[data-role="${r}"]`),button=n=>host.querySelector(`[data-action="${n}"]`),reduced=matchMedia('(prefers-reduced-motion: reduce)');
let tl,t=0,playing=false,raf,last=0,focusId='',inspectId='';
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const txt=(x,y,v,c='',attrs='')=>`<text x="${x}" y="${y}" class="${c}" ${attrs}>${esc(v)}</text>`;
const box=(x,y,w,h,c='box')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" class="${c}"/>`;
const chip=(x,y,v,c='chip',w=66,attrs='')=>`<g ${attrs}>${box(x,y,w,30,c)}${txt(x+w/2,y+20,v,'token','text-anchor="middle"')}</g>`;
const label=(x,y,a,b)=>txt(x,y,a,'title')+txt(x,y+21,b,'small');
const route=(points,active=false)=>`<path d="M${points.map(p=>p.join(' ')).join('L')}" class="route" data-selected="${active}" marker-end="url(#gx-arrow)"/>`;
function motion(points,p,label=''){return `<g data-payload="${esc(label)}"><title>${esc(label)}</title>${LearningDepth.pulse(`<path d="M${points.map(p=>p.join(' ')).join('L')}" fill="none" stroke="var(--gx-purple)" stroke-width="2"/>`,t,p>0&&p<1)}</g>`;}
const defs='<defs><marker id="gx-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="context-stroke"/></marker></defs>';
function speculative(s){
 const d=tl.data,x=i=>190+i*95,progress=id=>reduced.matches?Number(s.done(id)):s.p(id);let out=label(20,25,'EAGLEWorkerV2 · 同一请求 R1','上轮输出末尾 r；旧目标 KV 长度 8。node 0–4 是验证行索引。');
 const modules=[['draft','Draft worker'],['verify','Target worker'],['select','eagle_sample'],['commit','CPU result processor']];
 modules.forEach(([id,name],i)=>{out+=box(20+i*265,58,250,39,s.active.some(e=>e.id===id)?'trial':'box')+txt(32+i*265,83,name,'code');if(i<3)out+=route([[272+i*265,78],[281+i*265,78]]);});
 out+=txt(20,140,'验证输入','title')+txt(20,159,'draft_token','code')+txt(20,211,'目标 KV 窗口','title')+txt(20,230,'已分配的 5 个槽','small')+txt(20,283,'GPU 接受结果','title')+txt(20,303,'predict 有效前部','small')+txt(20,352,'CPU 输出','title')+txt(20,371,'Req.output_ids','code');
 d.tokens.forEach((v,i)=>{
  out+=txt(x(i)+32,120,`node ${i}`,'small','text-anchor="middle"');
  if(i===0||s.done('draft'))out+=chip(x(i),130,v,'chip');
  else if(s.p('draft')>0)out+=chip(x(i),74+56*progress('draft'),v,'trial',66,`data-moving-token="${i}"`);
  else out+=chip(x(i),130,'·','pending');
  out+=box(x(i),200,66,30,'pending');
  if(s.p('verify')>0){out+=`<rect x="${x(i)+2}" y="${228-26*progress('verify')}" width="62" height="${26*progress('verify')}" rx="4" fill="var(--accent-soft)"/>`;out+=txt(x(i)+33,220,v,s.done('select')?'token muted':'token','text-anchor="middle"');}
  out+=txt(x(i)+32,248,`slot ${i}`,'small','text-anchor="middle"');
 });
 if(s.done('select')){
  // Accepted nodes retain identity while the tree path is compacted to the front.
  d.path.forEach((node,i)=>{const px=x(node)+(x(i)-x(node))*progress('compact');out+=chip(px,200,d.tokens[node],'good',66,`data-kv-node="${node}"`);});
  if(tl.condition==='tree'&&s.p('compact')>0&&!s.done('compact'))out+=txt(405,188,'按 accept_index 搬移 KV / predict / hidden','small');
  d.output.forEach((v,i)=>{const node=d.path[i],px=x(node)+(x(i)-x(node))*progress('compact');out+=chip(px,272,v,'good',66,'data-predict-node="'+node+'"');});
 }
 out+=chip(x(0),342,'r','chip');
 if(s.p('commit')>0)d.output.forEach((v,i)=>{const p=progress('commit');out+=chip(x(i)+95*p,272+70*p,v,'good',66,`data-output="${v}"`);});
 out+=box(755,112,325,266)+txt(773,138,'三份状态分别看','title');
 out+=txt(773,168,'GPU new_seq_lens','code')+txt(1010,168,s.gpuLen,'token');
 out+=txt(773,198,'CPU kv_committed_len','code')+txt(1010,198,s.cpuLen,'token');
 out+=txt(773,228,'accept_index','code')+txt(773,251,s.acceptIndex.length?'['+s.acceptIndex.join(', ')+']':'尚未验证','code');
 out+=txt(773,281,s.done('select')?`accept_lens = ${d.n+1} = ${d.n} 草稿 + bonus`:'accept_lens 尚未产生','small');
 out+=txt(773,310,s.done('extend')?'下一轮种子：bonus z + 刷新后 hidden/topk':'下一轮种子：等待 Draft Extend','small');
 out+=txt(773,339,!s.done('select')?'验证后区分：有效前缀与预留':d.n===4?'本轮窗口：全部进入有效前缀':'灰色尾部：不属于有效前缀','small')+txt(773,359,'预留空间不因本轮拒绝自动 free','small');
 if(s.active.some(e=>e.id==='copy'))out+=motion([[728,281],[741,281],[741,357],[728,357]],s.p('copy'));
 return out;
}
function grammar(s){
 const d=tl.data,phase=s.focus.id;let out=label(20,25,'CPU · GrammarManager / XGrammar','R1 为当前请求；R2 后到，使用同一规则。')+label(635,25,'GPU · ModelRunner / Sampler','真实 matcher 留在 CPU；这里只搬词表 mask。');
 out+=box(20,68,270,290)+txt(35,93,'规则 key：regex / '+(d.bad?'( （无效）':'yes|no'),'code')+txt(35,125,'R1：'+s.queue,'title');
 const compiling=d.cold&&!s.done('ready');out+=txt(35,155,d.bad&&s.done('ready')?'InvalidGrammarObject → abort':compiling?'req.grammar = Future':s.done('ready')?'req.grammar = XGrammarGrammar':'req.grammar 尚未取得','small');
 out+=box(37,190,235,75,s.done('ready')&&!d.bad?'good':'pending')+txt(49,216,'CompiledGrammar · ctx','code')+txt(49,244,s.done('ready')?d.bad?'编译失败，不进入采样':'编译结果可复用':'等待缓存查询 / 编译','small');
 if(d.cold&&s.p('compile')>0)out+=`<rect x="37" y="274" width="${235*s.p('compile')}" height="5" fill="var(--gx-gold)"/>`;
 out+=txt(35,308,d.cold?'线程池负责编译；CPU 短暂轮询':'命中缓存，跳过编译 Future','small')+txt(35,334,'ctx 相同 ≠ matcher 进度相同','small');
 for(let i=0;i<2;i++){
  const y=78+i*148,ready=s.done(i?'r2':'ready')&&!d.bad;
  out+=box(345,y,240,120,ready?'box':'pending')+txt(360,y+26,`R${i+1} · 独立 matcher`,'title')+txt(360,y+52,'共享 ctx；各自 accepted_tokens','small')+txt(360,y+80,ready?'前缀：'+((i?s.r2:s.r1)||'空'):'尚未创建 matcher','code')+txt(360,y+103,ready?'下次允许：'+(!i&&s.done('advance')?(d.chosen==='y'?'es':'EOS'):'y / no'):'等待有效规则对象','small');
  const points=[[274,226],[315,226],[315,y+60],[345,y+60]];out+=route(points,ready);
  if(s.active.some(e=>e.id===(i?'r2':'ready')))out+=motion(points,s.p(i?'r2':'ready'),'copy');
 }
 out+=box(635,68,445,290)+txt(650,96,'R1 的同一批次行','title');
 const xx=i=>690+i*74;
 d.vocab.forEach((v,i)=>{out+=txt(xx(i),130,v,'code','text-anchor="middle"');out+=txt(xx(i),168,s.done('forward')?d.scores[i]:'—','token','text-anchor="middle"');out+=txt(xx(i),211,s.done('mask')?(d.legal[i]?'允许':'禁止'):'—','small','text-anchor="middle"');out+=txt(xx(i),251,s.masked?(d.legal[i]?d.scores[i]:'−∞'):'—','token','text-anchor="middle"');if(s.masked&&!d.legal[i])out+=`<path d="M${xx(i)-23} 239l46 17" stroke="var(--gx-red)"/>`;});
 out+=txt(649,151,'logits','small')+txt(649,195,'mask','small')+txt(649,235,'筛选后','small');
 if(s.chosen)out+=chip(803,277,s.chosen,'good',80);
 out+=txt(653,337,s.chosen?'结果回 CPU 后，只更新 R1':'不合法的高分 ! 不参与选择','small');
 const points=[[585,139],[609,139],[609,198],[635,198]];out+=route(points,s.p('h2d')>0);if(s.active.some(e=>e.id==='h2d'))out+=motion(points,s.p('h2d'),'bitmask');
 if(s.p('advance')>0&&!s.done('advance')){const p=reduced.matches?1:s.p('advance');out+=chip(803+(370-803)*p,277+(138-277)*p,d.chosen,'good',80,`data-moving-token="sample"`);}
 return out;
}
const positions=[[80,193],[250,110],[250,290],[420,110],[420,290]],edges=[[0,1],[1,3],[0,2],[2,4]],travel={enter1:[0,1],enter3:[1,3],back3:[3,1],back1:[1,0],enter2:[0,2],prune4:[2,4],restore:[2,0]};
function combined(s){
 let out=label(20,25,'CPU · 候选树 DFS','node 编号对应 verify 行；r = {，来自上一轮。')+label(530,25,'每节点的后继 mask','只有试走合法路径才写这一行。')+label(850,25,'R1 · 规则与提交','临时试走与真实进度分开。');
 const move=travel[s.focus.id],p=s.p(s.focus.id);
 const port=(a,b)=>{const u=positions[a],v=positions[b],dx=v[0]-u[0],dy=v[1]-u[1],len=Math.hypot(dx,dy);return [[u[0]+dx/len*36,u[1]+dy/len*25],[v[0]-dx/len*36,v[1]-dy/len*25]];};
 edges.forEach(([a,b])=>out+=route(port(a,b),!!move&&move.includes(a)&&move.includes(b)));
 positions.forEach(([x,y],i)=>{out+=chip(x-36,y-16,tl.data.tokens[i],i===4&&s.done('prune4')?'bad':s.node===i?'trial':s.rows.includes(i)?'good':'chip',72,`data-node="${i}"`)+txt(x,y+34,`node ${i}${i===0?' · root':''}`,'small','text-anchor="middle"');});
 if(move&&p>0&&p<1){const [a,b]=port(...move),u=reduced.matches?1:p;out+=`<circle cx="${a[0]+(b[0]-a[0])*u}" cy="${a[1]+(b[1]-a[1])*u}" r="5" fill="var(--gx-purple)"><title>CPU 遍历游标；无网络传输</title></circle>`;}
 out+=txt(20,360,'连线：同进程索引关系，不是网络链路','small')+txt(20,380,'临时 accept / rollback 操作同一个 R1 matcher','small');
 const allowed=['"ok" / "err"',':',':','true','未生成 · 不可达'];
 for(let i=0;i<5;i++){const y=75+i*53;out+=box(530,y,280,43,s.rows.includes(i)?'good':'pending')+txt(543,y+27,`行 ${i}`,'code')+txt(605,y+27,s.rows.includes(i)?allowed[i]:i===4&&s.done('prune4')?allowed[i]:'待生成','code');}
 out+=box(850,68,230,132)+txt(865,95,s.done('advance')?'本轮已正式推进':s.restored?'DFS 已还原':'matcher 当前前缀','title')+txt(865,130,s.matcher||'空（等上一轮）','code')+txt(865,176,s.restored&&!s.done('advance')?'候选未提交；暂时仍是 {':'随当前 CPU 动作变化','small');
 out+=box(850,223,230,133)+txt(865,249,'本轮接受结果','title')+txt(865,278,s.done('sample')?'index = [0, 1, 3]':'等待 logits + mask','code')+txt(865,305,s.done('commit')?'输出："ok" : true':'CPU output 尚未追加','code')+txt(865,333,'grammar_advanced = '+s.advanced,'small');
 return out;
}
function draw(s){$('scene').innerHTML=defs+(tl.mode==='speculative'?speculative(s):tl.mode==='grammar'?grammar(s):combined(s));$('scene').dataset.clock=t.toFixed(3);}
function tracks(s){
 const H=tl.lanes.length*29+40,x=v=>205+v/tl.duration*875,colors={cpu:'var(--gx-green)',compute:'var(--gx-gold)',data:'var(--gx-purple)',wait:'url(#gx-hatch)'};let out='<defs><pattern id="gx-hatch" width="6" height="6" patternUnits="userSpaceOnUse"><path d="M0 6L6 0" stroke="var(--gx-gray)"/></pattern></defs>';
 for(let i=0;i<=5;i++){const n=i*tl.duration/5;out+=txt(x(n),15,n.toFixed(1),'axis',`text-anchor="${i===5?'end':'start'}"`)+`<path d="M${x(n)} 22V${H-7}" class="grid"/>`;}
 tl.lanes.forEach((lane,i)=>{const y=28+i*29;out+=txt(194,y+16,lane.name,'lane','text-anchor="end"')+`<path d="M205 ${y+25}H1080" class="grid"/>`;
  tl.events.filter(e=>e.lane===lane.id).forEach(e=>{const w=x(e.end)-x(e.start),active=s.active.includes(e),short=e.id==='next'?'下一轮 Draft':e.id==='verify'||e.id==='target'?'Target Verify':e.id==='extend'?'Draft Extend':e.id==='compile'?'异步编译':e.id==='draft'?'Draft':e.kind==='wait'?'等待':e.id==='previous'?'上一轮 D2H':e.id==='other'?'B0 前向':({select:'验证采样',sample:'采样',apply:'应用 mask',compact:'路径整理',publish:'长度',lookup:'查缓存',r2:'R2 copy',ready:'准入',commit:'CPU 提交',advance:'FSM 提交',barrier:'补齐 FSM',root:'根行',enter1:'试走',enter3:'试走',enter2:'试走',back3:'回退',back1:'回退',restore:'还原',prune4:'跳过'}[e.id]||(e.kind==='data'?'传递':e.kind==='compute'?'计算':'CPU'));out+=`<g data-event="${e.id}" data-active="${active}" class="gx-event" role="button" tabindex="0" aria-label="${esc(e.title+'；依赖 '+(e.deps.join('、')||'见说明'))}"><rect x="${x(e.start)}" y="${y}" width="${w}" height="23" rx="3" fill="${colors[e.kind]}" opacity="${active?1:t>=e.end?.65:.25}" stroke="${active?'var(--fg)':'none'}"/><title>${esc(e.title+'：'+e.detail)}</title>${w>[...short].reduce((n,c)=>n+(c.charCodeAt(0)>255?11:6.2),0)+8?txt(x(e.start)+4,y+16,short,'bar-label'):''}</g>`;});
 });out+=`<path d="M${x(t)} 21V${H-7}" stroke="var(--fg)" stroke-width="1.5"/><circle cx="${x(t)}" cy="21" r="3" fill="var(--fg)"/>`;$('tracks').setAttribute('viewBox',`0 0 1100 ${H}`);$('tracks').innerHTML=out;$('tracks').dataset.clock=t.toFixed(3);
}
function render(){const s=tl.sample(t),focus=tl.events.find(e=>e.id===inspectId)||s.focus;host.dataset.time=t.toFixed(3);host.dataset.mode=tl.mode;host.dataset.condition=tl.condition;$('seek').value=t;$('time').textContent=`${t.toFixed(1)} / ${tl.duration.toFixed(1)} u · 非实测`;draw(s);tracks(s);host.dataset.playing=String(playing);depth.update({time:t,body:DepthProjections.generation(tl,s),event:focus.title,explanation:focus.detail,href:M.url(focus.source),rows:tl.mode==='speculative'?[['GPU new_seq_lens',s.gpuLen],['CPU kv_committed_len',s.cpuLen],['accept_index',DepthProjections.value(s.acceptIndex)],['output_ids',DepthProjections.value(s.output)]]:[['请求队列',s.queue],['R1 matcher',s.r1??s.matcher],['R2 matcher',s.r2],['正式输出',DepthProjections.value(s.output)]]});if(focusId!==focus.id){focusId=focus.id;$('phase').textContent=focus.title;$('detail').textContent=focus.detail;$('source').href=M.url(focus.source);$('steps').value=focus.id;}$('result').textContent=tl.mode==='speculative'?`设备长度 ${s.gpuLen}；CPU KV 账本 ${s.cpuLen}；本轮 CPU 输出 [${s.output.join(', ')}]。`:tl.mode==='grammar'?`R1 ${s.queue}，前缀 ${s.r1||'空'}；R2 前缀 ${s.r2||'空'}。`:`真实 / 临时 matcher：${s.matcher||'空'}；本轮 CPU 输出 [${s.output.join(', ')}]。`;button('prev').disabled=t<=0;button('next').disabled=t>=tl.duration;}
function persist(){const u=new URL(location.href);u.searchParams.set('mode',tl.mode);u.searchParams.set('case',tl.condition);u.searchParams.set('t',t.toFixed(3));u.searchParams.delete('mv');u.searchParams.delete('ms');history.replaceState(null,'',u);}
function pause(){playing=false;cancelAnimationFrame(raf);button('play').textContent=t>=tl?.duration?'重播':'播放全程';button('play').setAttribute('aria-pressed','false');}
function seek(v,save=true,inspect=''){inspectId=inspect;pause();t=Math.max(0,Math.min(tl.duration,Number(v)||0));render();pause();if(save)persist();}
function load(mode,condition,time=0,save=true){pause();tl=M.build(mode,condition);focusId='';document.getElementById('gen-title').textContent=tl.question;$('scope').textContent=tl.scope;$('rule').textContent=tl.rule;host.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===tl.mode)));$('condition').innerHTML=Object.entries(tl.cases).map(([k,v])=>`<option value="${k}">${esc(v)}</option>`).join('');$('condition').value=tl.condition;$('steps').innerHTML=[...tl.events].sort((a,b)=>a.start-b.start).map(e=>`<option value="${e.id}">${e.start.toFixed(1)} · ${esc(e.title)}</option>`).join('');$('seek').max=tl.duration;seek(time,save);host.dataset.ready='true';}
function tick(now){if(!playing)return;t=Math.min(tl.duration,t+Math.min(.1,(now-last)/1000)*Number($('speed').value));last=now;render();if(t>=tl.duration){pause();persist();}else raf=requestAnimationFrame(tick);}
button('play').onclick=()=>{if(playing){pause();persist();return;}if(t>=tl.duration)t=0;inspectId='';playing=true;last=performance.now();button('play').textContent='暂停';button('play').setAttribute('aria-pressed','true');raf=requestAnimationFrame(tick);};button('reset').onclick=()=>seek(0);button('prev').onclick=()=>seek([...tl.steps].reverse().find(v=>v<t-.01)||0);button('next').onclick=()=>seek(tl.steps.find(v=>v>t+.01)??tl.duration);
host.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>load(b.dataset.mode));$('condition').onchange=()=>load(tl.mode,$('condition').value);$('seek').oninput=()=>seek($('seek').value);$('steps').onchange=()=>seek(tl.events.find(e=>e.id===$('steps').value).start,true,$('steps').value);
const jump=e=>{const n=e.target.closest('[data-event]');if(n){const id=n.dataset.event;seek(tl.events.find(v=>v.id===id).start,true,id);$('tracks').querySelector('[data-event="'+id+'"]').focus({preventScroll:true});}};$('tracks').onclick=jump;$('tracks').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();jump(e);}};
const depth=LearningDepth.mount({host,before:$('scene').parentElement,module:[$('scene').parentElement],onChange:()=>render(),persist});
function restore(){const q=new URL(location.href).searchParams;load(q.get('mode'),q.get('case'),q.get('t'),false);}window.addEventListener('popstate',restore);document.addEventListener('visibilitychange',()=>{if(document.hidden){pause();persist();}});window.addEventListener('pagehide',pause);reduced.addEventListener('change',render);restore();const resume=window.SiteUpdate?.take('generation-execution');if(resume){load(resume.mode,resume.condition,resume.time,false);if([...$('speed').options].some(o=>o.value===resume.speed))$('speed').value=resume.speed;}window.SiteUpdate?.register('generation-execution',()=>{pause();persist();return {mode:tl.mode,condition:tl.condition,time:t,speed:$('speed').value};});
})();
