/* One continuous teaching clock drives tracks and causal state checkpoints. */
(()=>{'use strict';const M=window.LoopJourney,host=document.querySelector('[data-loop-journey]');if(!host||!M)return;
const $=s=>host.querySelector(s),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let mode='normal',step=0,hold=false,released=false,view='all',focus='M1',colorBy='batch';
let inspected='',lastTrackWindow='',lastOwnerLayout='',lastPalette='',time=0,playing=false,speed=1,raf=0,lastFrame=0,lastContent='',lastTracks='',current;
const setText=(selector,value)=>{const node=$(selector);if(node.textContent!==value)node.textContent=value;};
const scenes=Object.fromEntries(Object.keys(M.configs).map(key=>[key,M.execution(key)]));
const loopLabel=s=>s.replace('P0:','PP0 · ').replace('P1:','PP1 · ');
const kinds={control:'CPU / 控制',compute:'模型计算',tensor:'数据准备 / 交接',wait:'等待前提'};
const textMeasure=document.createElement('canvas').getContext('2d');textMeasure.font='11px system-ui,sans-serif';
function alignTrackOwners(){const svg=$('#lj-tracks-svg'),viewport=$('.lj-tracks-scroll'),ex=current.execution;if(!svg.clientWidth)return;const scale=svg.getBoundingClientRect().width/1120;const layoutKey=mode+':'+scale;if(layoutKey!==lastOwnerLayout){lastOwnerLayout=layoutKey;$('#lj-track-owners').style.width=(160*scale)+'px';$('#lj-track-owners').innerHTML='<div style="top:0;height:'+48*scale+'px">执行实体</div>'+ex.lanes.map((l,i)=>'<div style="top:'+((48+i*48)*scale)+'px;height:'+48*scale+'px">'+esc(l.label)+'</div>').join('');}const key=mode+':'+ex.time+':'+viewport.clientWidth;if(key!==lastTrackWindow){lastTrackWindow=key;viewport.scrollLeft=Math.max(0,(160+ex.time/ex.duration*940)*scale-viewport.clientWidth*.65);}}
function renderTracks(){
 const ex=current.execution;
 const byId=Object.fromEntries(ex.events.map(e=>[e.id,e]));
 const groupKey=e=>colorBy==='kind'?(e.kind==='local'?'tensor':e.kind):(e[colorBy]||'none');
 const groups=[...new Set(ex.events.map(groupKey))].sort((a,b)=>a==='none'?1:b==='none'?-1:a.localeCompare(b,undefined,{numeric:true}));
 const groupColor=key=>key==='none'?'--muted':colorBy==='kind'?`--lj-kind-${key}`:`--lj-group-${groups.indexOf(key)}`;
 const groupLabel=key=>key==='none'?'未归入批次':colorBy==='kind'?kinds[key]:colorBy==='loop'?loopLabel(key):key;
 host.dataset.colorBy=colorBy;$('#lj-color').value=colorBy;
 $('#lj-color option[value=batch]').textContent=mode==='pp'?'Microbatch（M）':'执行批次（B）';
 if(lastPalette!==mode+':'+colorBy){lastPalette=mode+':'+colorBy;$('#lj-track-legend').innerHTML=groups.map(key=>`<span style="--track:var(${groupColor(key)})" data-color-group="${key}">${esc(groupLabel(key))}</span>`).join('');}
 setText('#lj-color-note',colorBy==='batch'?'同一批工作全程同色；等待按正在推进的批次归组，条带文字说明它在等谁。':colorBy==='loop'?(mode==='pp'?'各 stage 分别计轮；异步工作按组织它的轮次归组，消息按发送轮次着色。':'按组织活动的本地轮次归组；同一轮可能计算新批、处理旧批。'):'颜色区分控制、计算、数据交接与等待。');
 if(!byId[inspected])inspected='';
 const selected=byId[inspected]||ex.active.find(e=>e.kind==='compute')||ex.active[0];
 const x0=160,width=940,row=48,y0=48,height=y0+ex.lanes.length*row+28;
 const xpos=t=>x0+t/ex.duration*width,ypos=e=>y0+ex.lanes.findIndex(l=>l.id===e.lane)*row;
 const related=e=>mode!=='pp'||view==='all'||(view==='loop'?e.loop===focus||e.targetLoop===focus:e.batch===focus);
 const identity=e=>`${e.batch?'所属批次 '+e.batch:'未归入执行批次'}；${e.targetLoop?'发送 '+loopLabel(e.loop)+' → 接收 '+loopLabel(e.targetLoop):'本地轮次 '+loopLabel(e.loop)}`;
 const rect=(x,y,w,h,cls)=>`<rect x="${x}" y="${y}" width="${Math.max(0,w)}" height="${h}" rx="4" class="${cls}"/>`;
 const trackKey=[mode,colorBy,view,focus,selected?.id,ex.blocked].join('|');
 if(trackKey!==lastTracks){
 lastTracks=trackKey;
 let svg=`<title id="lj-track-title">${esc(M.configs[mode].name)}：同一教学时钟的执行轨道</title><desc id="lj-track-desc">横向位置表示同一场景的假设时刻，条带表示 CPU 工作、GPU 计算、数据交接和等待。点击条带查看依赖；u 不是实测秒数。</desc><defs><marker id="lj-track-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="var(--accent)"/></marker></defs>`;
 for(let t=0;t<=ex.duration;t+=2){svg+=`<text x="${xpos(t)+(t===0?5:0)}" y="23" class="lj-tick" text-anchor="${t===0?'start':'middle'}">${t}u</text><path d="M${xpos(t)},32V${height-20}" class="lj-track-grid"/>`;}
 for(const [i,l]of ex.lanes.entries()){
  const y=y0+i*row;svg+=`<text x="9" y="${y+20}" class="lj-track-owner">${esc(l.label)}</text>`+rect(x0,y,width,32,'lj-track-background');
 }
 // Draw only the inspected activity's incoming edges, using the same event
 // boundaries and lane geometry as the bars. Edges run in the inter-row gutter.
 if(selected)for(const id of selected.depends){const p=byId[id];if(!p)continue;const ax=xpos(p.end),ay=ypos(p)+16,bx=xpos(selected.start),by=ypos(selected)+16,gutter=ypos(selected)-8;
  if(p.lane===selected.lane&&Math.abs(ax-bx)<1)continue;
  svg+=`<path d="${p.lane===selected.lane?`M${ax},${ay}H${bx}`:`M${ax},${ay}H${ax+4}V${gutter}H${bx-8}V${by}H${bx}`}" class="lj-track-dependency" marker-end="url(#lj-track-arrow)"/>`;
 }
 for(const e of ex.events){const x=xpos(e.start),y=ypos(e),w=xpos(e.end)-x,progress=ex.progress[e.id],active=ex.active.some(a=>a.id===e.id),isSelected=e.id===selected?.id;
  const key=groupKey(e),short=colorBy==='batch'?e.batch:colorBy==='loop'?(e.targetLoop?e.loop.replace(':',''):e.loop.split(':').pop()):e.kind==='wait'?'等待':e.batch||'CPU';
  const label=[e.label,e.kind==='wait'&&short?short+' 等待':'',short||''].find(t=>t&&textMeasure.measureText(t).width<=w-8)||'';
  svg+=`<g tabindex="0" role="button" aria-label="${esc(e.label+'；'+identity(e)+'；'+e.start+' 至 '+e.end+' 教学单位；点击查看依赖')}" class="lj-track-event ${e.kind}${related(e)?'':' is-muted'}${isSelected?' is-selected':''}" style="--track:var(${groupColor(key)})" data-color-group="${key}" data-track-event="${e.id}" data-start="${e.start}" data-end="${e.end}" data-owner="${e.lane}" data-active="${active}"><title>${esc(e.label)} · ${esc(identity(e))} · ${e.start}–${e.end} u（教学设定）</title>`+rect(x,y,w,32,'lj-track-slot')+rect(x,y,w*progress,32,'lj-track-fill')+(label?`<text x="${x+w/2}" y="${y+21}" text-anchor="middle" class="lj-track-label">${esc(label)}</text>`:'')+'</g>';
 }
 svg+=`<path d="M${xpos(ex.time)},31V${height-20}" class="lj-track-cursor"/>`;
 $('#lj-tracks-svg').setAttribute('viewBox',`0 0 1120 ${height}`);$('#lj-tracks-svg').innerHTML=svg;
 }
 for(const node of host.querySelectorAll('[data-track-event]')){const id=node.dataset.trackEvent;node.querySelector('.lj-track-fill').setAttribute('width',Number(node.querySelector('.lj-track-slot').getAttribute('width'))*ex.progress[id]);node.dataset.active=String(ex.active.some(e=>e.id===id));}
 $('#lj-tracks-svg .lj-track-cursor').setAttribute('d',`M${xpos(ex.time)},31V${height-20}`);$('#lj-tracks-svg').dataset.clock=ex.time;
 setText('#lj-clock',ex.blocked?`在 ${ex.time.toFixed(2)}u 暂停演示，观察未就绪的依赖；不表示所有设备停机。后续轮廓保留原假设排程，迟到后的完成时刻未知。`:`当前教学时刻 ${ex.time.toFixed(2)}u / ${ex.duration.toFixed(1)}u · 播放、拖动与节点步进共用此游标`);
 alignTrackOwners();
 setText('#lj-track-detail',selected?`${selected.label} · ${identity(selected)}；前提：${selected.depends.map(id=>byId[id]?.label||id).join('、')||'本场景中已具备的输入 / 本地控制顺序'}。${ex.blocked?'暂缓的条件未解除，后续状态尚未提交。':''}`:'当前是初始或空闲状态；点选任一条带，核对它等待的具体活动。');
}

function pause(){playing=false;cancelAnimationFrame(raf);raf=0;}
function readURL(){pause();const u=new URL(location.href);mode=M.configs[u.hash.slice(1)]?u.hash.slice(1):'normal';hold=u.searchParams.get('hold')==='1';released=false;view='all';focus='M1';inspected='';colorBy=['kind','batch','loop'].includes(u.searchParams.get('color'))?u.searchParams.get('color'):'batch';const legacy=M.snapshot(mode,Number(u.searchParams.get('step'))||0);time=u.searchParams.has('t')?Number(u.searchParams.get('t')):scenes[mode].times[legacy.index];render(true,Boolean(M.configs[u.hash.slice(1)]));}
function url(push=false){const u=new URL(location.href);u.hash=mode;if(time>0)u.searchParams.set('t',time.toFixed(3));else u.searchParams.delete('t');if(step)u.searchParams.set('step',step);else u.searchParams.delete('step');if(hold)u.searchParams.set('hold','1');else u.searchParams.delete('hold');if(colorBy!=='batch')u.searchParams.set('color',colorBy);else u.searchParams.delete('color');history[push?'pushState':'replaceState']({},'',u);}
function seek(value,scroll=false){pause();time=value;render(scroll);}
function jump(index){inspected='';seek(scenes[mode].times[Math.max(0,Math.min(scenes[mode].times.length-1,index))],true);}
function tick(now){if(!playing)return;time+=Math.min(.1,(now-lastFrame)/1000)*speed;lastFrame=now;render(false,false);if(playing)raf=requestAnimationFrame(tick);else url();}
function renderPlayback(s){
 const ex=s.execution;host.dataset.time=time.toFixed(3);host.dataset.playing=String(playing);
 $('#lj-seek').max=ex.duration;$('#lj-seek').value=time;$('#lj-seek').setAttribute('aria-valuetext',`${time.toFixed(2)} / ${ex.duration.toFixed(1)} 教学单位${s.blocked?'，依赖未就绪，演示已暂停':''}`);
 $('#lj-time').textContent=`${time.toFixed(2)} / ${ex.duration.toFixed(1)} u`;
 $('#lj-speed').value=String(speed);$('#lj-play').textContent=playing?'暂停':time>=ex.duration?'重播全程':'播放全程';$('#lj-play').setAttribute('aria-pressed',String(playing));$('#lj-play').disabled=s.blocked;
 $('#lj-prev').disabled=time<=0;$('#lj-next').disabled=time>=ex.duration||s.blocked;$('#lj-release').hidden=!s.blocked;
 setText('#lj-status',s.blocked?'等待依赖 · 演示已暂停':time>=ex.duration?'旅程结束 · 可拖动回看':playing?'播放中':'已暂停 · 可播放或拖动');
 const active=ex.active.map(e=>`${ex.lanes.find(l=>l.id===e.lane).label}：${e.label}`);
 setText('#lj-activity',active.length?'此刻轨道：'+active.join('；'):'本轮旅程已收尾，后续循环可继续接纳请求。');
}
function relevant(a){return mode!=='pp'||view==='all'||(view==='batch'?a[2].split('/').includes(focus):a[3].includes(focus));}
function render(scroll=false,writeURL=true){const c=M.configs[mode],s=M.sample(mode,time,{hold,released});current=s;time=s.time;step=s.index;if(s.blocked||time>=s.execution.duration)pause();host.dataset.mode=mode;host.dataset.step=step;host.dataset.blocked=String(s.blocked);
 renderPlayback(s);
 const contentKey=[mode,step,s.blocked,view,focus].join('|');
 if(contentKey!==lastContent){lastContent=contentKey;
 document.querySelectorAll('[data-loop-reading]').forEach(e=>e.hidden=e.dataset.loopReading!==mode);
 for(const b of host.querySelectorAll('[data-mode]'))b.setAttribute('aria-pressed',String(b.dataset.mode===mode));
 $('#lj-question').textContent=c.question;$('#lj-scope').textContent=c.scope;$('#lj-hold-text').textContent=c.barrierLabel;$('#lj-hold').checked=hold;
 $('#lj-pp-controls').hidden=mode!=='pp';$('#lj-view').value=view;
 const options=view==='loop'?['P0:L0','P0:L1','P0:L2','P0:L3','P1:L0','P1:L1','P1:L2']:['M1','M2'];if(!options.includes(focus))focus=options[0];
 $('#lj-focus').innerHTML=options.map(v=>`<option value="${v}"${focus===v?' selected':''}>${v.replace('P0:','PP0 · ').replace('P1:','PP1 · ')}</option>`).join('');$('#lj-focus-label').hidden=view==='all';
 $('#lj-view-note').textContent=view==='loop'?`${loopLabel(focus)} 高亮其本地行为与收发消息；其余活动保留在图中。` : view==='batch'?`高亮 ${focus} 的请求、激活、输出及两级结果处理；同一 microbatch 可以跨过多个本地 loop。`:'同一竖线对应同一教学时刻；实色填充由同一游标驱动。CPU 提交、GPU 开始计算与 CPU 结果提交分别追踪。';
 $('#lj-count').textContent=`${step+1} / ${c.steps.length}`;$('#lj-title').textContent=s.current.title;$('#lj-summary').textContent=s.blocked?`等待：${c.barrierLabel}。前提尚未齐备，当前事件没有完成，后续消费者不能越过这个门槛。`:s.current.summary;
 const oldScroll=$('#lj-scroll').scrollLeft;
 $('#lj-timeline').style.setProperty('--steps',c.steps.length);
 $('#lj-timeline').innerHTML=`<div class="lj-corner">执行实体 ↓<br>事件顺序 →</div>`+c.steps.map((e,i)=>`<button class="lj-column ${i===step?'is-current':''}" data-step="${i}" aria-label="第 ${i+1} 步：${esc(e.title)}"${i===step?' aria-current="step"':''}><b>${String(i+1).padStart(2,'0')}</b><span>${esc(e.title)}</span></button>`).join('')+c.lanes.map(([id,label,description])=>`<div class="lj-lane-label"><strong>${esc(label)}</strong><small>${esc(description)}</small></div>`+c.steps.map((e,i)=>{const actions=e.a.filter(a=>a[0]===id);return `<div class="lj-cell ${i===step?'is-current':''} ${i>step?'is-future':''} ${s.blocked&&i===step?'is-blocked':''}">${actions.map(a=>`<div class="lj-action ${relevant(a)?'':'is-muted'}" data-batch="${esc(a[2])}"><small>${esc(a[3])} · ${esc(a[2])}</small><span>${esc(a[1])}</span>${i===step?`<em>${s.blocked?'等待条件':i===0?'初始状态':'当前事件'}</em>`:''}</div>`).join('')||'<span class="lj-no-event">—</span>'}</div>`;}).join('')).join('');
 $('#lj-scroll').scrollLeft=oldScroll;
 $('#lj-deps').innerHTML=(s.dependencies.length?s.dependencies.map(d=>`<li><span class="lj-check">${d.ready?'✓ 已发生':'○ 未发生'}</span><button data-step="${c.steps.findIndex(e=>e.id===d.id)}">${esc(d.title)} ↗</button></li>`).join(''):'<li>初始场景；尚无上游事件。</li>')+(s.blocked?`<li class="lj-wait">○ ${esc(c.barrierLabel)}：条件未满足</li>`:'');
 $('#lj-dependency-note').textContent=s.current.why||'箭头表达完成前提；事件编号用于讲解顺序，没有把所有独立动作规定为全局串行。';
 $('#lj-causal-current').textContent=s.blocked?'等待依赖':s.current.title;
 $('#lj-source').href=M.sourceURL(s.current.source);$('#lj-source').textContent=M.sources[s.current.source][2]+' · 固定源码 ↗';
 const labels={request:'请求',batch:'本轮执行 / 槽位',device:'设备数据',host:'CPU 已提交状态',queue:'待处理结果',resource:'资源与复用'};
 $('#lj-ledger').innerHTML=Object.entries(labels).map(([key,label])=>`<div><dt>${label}</dt><dd>${esc(s.data[key])}</dd></div>`).join('');
 const path=$('#lj-path');if(path.dataset.mode!==mode){path.dataset.mode=mode;path.innerHTML=c.positions.map((p,i)=>`<div class="lj-stop" data-position="${i}"><span>${i+1}</span><strong>${esc(p)}</strong></div>`).join('');}
 path.querySelectorAll('.lj-stop').forEach((e,i)=>{e.classList.toggle('is-active',i===s.data.place);e.classList.toggle('is-passed',i<s.data.place);});
 $('#lj-payload').textContent=s.data.payload;$('#lj-payload').classList.toggle('is-waiting',s.blocked);
 }
 if(scroll){const viewport=$('#lj-scroll'),cell=host.querySelector(`.lj-column[data-step="${step}"]`);viewport.scrollLeft=Math.max(0,cell.getBoundingClientRect().left-viewport.getBoundingClientRect().left+viewport.scrollLeft-host.querySelector('.lj-corner').offsetWidth);}
 renderTracks();
 depth.update({time,body:DepthProjections.loop(s,mode,colorBy),event:s.current.title,explanation:s.current.why,rows:Object.entries(s.data),href:M.sourceURL(s.current.source)});
 if(writeURL)url();
}
host.addEventListener('click',e=>{const track=e.target.closest('[data-track-event]');if(track){pause();inspected=track.dataset.trackEvent;render();return;}const modeButton=e.target.closest('button[data-mode]');if(modeButton){if(mode===modeButton.dataset.mode)return;pause();mode=modeButton.dataset.mode;inspected='';time=0;step=0;hold=false;released=false;view='all';focus='M1';colorBy='batch';url(true);render(true);return;}const b=e.target.closest('button[data-step]');if(b)jump(Number(b.dataset.step));});
$('#lj-prev').onclick=()=>{inspected='';seek([...scenes[mode].times].reverse().find(t=>t<time-.001)??0,true);};
$('#lj-next').onclick=()=>{inspected='';seek(scenes[mode].times.find(t=>t>time+.001)??scenes[mode].duration,true);};
$('#lj-reset').onclick=()=>{released=false;inspected='';seek(0,true);};
$('#lj-release').onclick=()=>{released=true;seek(time);};
$('#lj-play').onclick=()=>{if(playing){pause();render();return;}if(time>=scenes[mode].duration){time=0;released=false;}inspected='';playing=true;lastFrame=performance.now();render(false,false);if(playing)raf=requestAnimationFrame(tick);};
$('#lj-seek').oninput=e=>{inspected='';seek(Number(e.target.value));};$('#lj-speed').onchange=e=>{speed=Number(e.target.value);lastFrame=performance.now();};
$('#lj-hold').onchange=e=>{hold=e.target.checked;released=false;inspected='';seek(0,true);};$('#lj-view').onchange=e=>{view=e.target.value;if(view!=='all')colorBy=view;render(false);};$('#lj-focus').onchange=e=>{focus=e.target.value;render(false);};
$('#lj-color').onchange=e=>{colorBy=e.target.value;render(false,false);url();};
window.addEventListener('popstate',()=>{if(!location.hash||M.configs[location.hash.slice(1)])readURL();});window.addEventListener('hashchange',()=>{if(M.configs[location.hash.slice(1)])readURL();});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&playing){pause();render();}});window.addEventListener('pagehide',()=>{if(playing){pause();render();}});
document.querySelectorAll('[data-answer]').forEach(b=>b.onclick=()=>{const out=b.closest('.lj-quiz').querySelector('[role=status]');out.textContent=b.dataset.answer;});
host.addEventListener('keydown',e=>{const el=e.target.closest('[data-track-event]');if(el&&['Enter',' '].includes(e.key)){e.preventDefault();pause();inspected=el.dataset.trackEvent;render();host.querySelector(`[data-track-event="${inspected}"]`)?.focus();}});
const depth=LearningDepth.mount({host,before:$('.lj-tracks'),module:[$('.lj-event-detail'),$('#lj-ledger')],source:[$('.lj-event-detail')],onChange:()=>render(false,false),persist:()=>url()});
const resume=window.SiteUpdate?.take('loop-journey');
window.SiteUpdate?.register('loop-journey',()=>{pause();render(false,false);return {mode,step,time,speed,hold,released,view,focus,colorBy,inspected};});
host.hidden=false;
if(resume&&M.configs[resume.mode]){({mode,step,hold,released,view,focus,colorBy,inspected}=resume);time=resume.time??scenes[mode].times[M.snapshot(mode,step).index];speed=[.5,1,1.5,2].includes(resume.speed)?resume.speed:1;render(false,false);}else readURL();
new ResizeObserver(alignTrackOwners).observe($('#lj-tracks-svg'));
})();
