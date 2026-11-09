(function(){
'use strict';
const M=window.PDHandoff,host=document.querySelector('[data-pd-handoff]');if(!M||!host)return;
const $=id=>host.querySelector('#'+id),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let tl,time=0,playing=false,rate=1,last=0,raf=0,phaseKey='',S;
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
const text=(x,y,s,cls='',anchor='start')=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`;
const rect=(x,y,w,h,cls='',r=7)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`;
const progress=id=>S.progress[id]||0,done=id=>progress(id)===1,doing=id=>S.active.some(e=>e.id===id);
const geometry={
 p:{x:20,w:330},d:{x:610,w:330},
 destination:[[626,128],[334,128]],aux:[[334,204],[626,204]],notify:[[334,264],[626,264]],
 kv:[[334,381],[397,381],[397,335],[563,335],[563,381],[626,381]],
 prebuilt:[[777,270],[777,306]]
};
function wire(points,kind){return `<path d="${points.map((p,i)=>(i?'L':'M')+p.join(',')).join(' ')}" class="pd-wire ${kind}" marker-end="url(#pd-arrow-${kind})"/>`;}
function chip(x,y,label,kind='data',w=48){return `<g class="pd-chip ${kind}">${rect(x,y,w,25)}${text(x+w/2,y+17,label,'pd-chip-label','middle')}</g>`;}
function flow(id,points,label,kind,w=74,drawWire=true){
 const p=progress(id);let out=drawWire?wire(points,kind):'';if(p<=0||p>=1||reduced.matches)return out;
 const segments=points.slice(1).map((v,i)=>Math.hypot(v[0]-points[i][0],v[1]-points[i][1]));
 let remaining=segments.reduce((a,b)=>a+b,0)*p,k=0;
 while(k<segments.length-1&&remaining>segments[k])remaining-=segments[k++];
 const a=points[k],b=points[k+1],q=remaining/segments[k],x=a[0]+(b[0]-a[0])*q,y=a[1]+(b[1]-a[1])*q;
 return out+`<g data-motion="${id}" data-x="${x.toFixed(3)}" data-y="${y.toFixed(3)}">${chip(x-w/2,y-12.5,label,kind,w)}</g>`;
}
function bar(x,y,w,id){const p=progress(id),value=reduced.matches?(p===1?1:0):p;return rect(x,y,w,4,'pd-bar-base',2)+rect(x,y,w*value,4,'pd-bar-fill',2);}
function kvCells(side,x){
 let out='';const isP=side==='p',complete=isP?S.pKV:S.dKV;
 for(let i=0;i<(isP?4:5);i++){
  const id=i===4?'decode':(isP?'compute':'copy')+(i<2?1:2),value=progress(id),exists=i===4?done('prebuilt'):isP?done(i<2?'admit':'enqueue1'):S.reserved,valid=i===4?S.decodeKV:complete>i;
  const state=isP&&S.released?'已解除请求持有':valid?'已写入':doing(id)?'写入中':exists?'已预留':'未分配';
  out+=`<g data-kv="${side}${i}" data-state="${state}" class="pd-kv-cell ${valid?'valid':''} ${S.released&&isP?'released':''}">`;
  out+=rect(x+i*54,366,48,36,exists?'reserved':'unallocated');
  if(value>0&&!valid&&!reduced.matches)out+=rect(x+i*54,366,48*value,36,'writing');
  out+=text(x+i*54+24,382,i===4?'y1':`p${i}`,'pd-chip-label','middle');
  out+=text(x+i*54+24,396,exists?'@'+(isP?[2,3,6,7][i]:[8,9,12,13,14][i]):'待分配','pd-tick','middle')+'</g>';
 }
 return out;
}
function side(which){
 const isP=which==='p',x=geometry[which].x,inner=x+16,gpuID=isP?(done('compute1')?'compute2':'compute1'):'decode';
 const activeCPU=S.active.find(e=>e.lane===(isP?'pcpu':'dcpu')&&e.kind!=='wait');
 let out=rect(x,50,330,404,'pd-host')+text(x+16,75,isP?'主机 P · Prefill 服务':'主机 D · Decode 服务','pd-host-title');
 out+=rect(inner,92,298,188,'pd-cpu')+text(inner+12,113,'CPU · Scheduler / 本地 Req','pd-module');
 out+=text(inner+12,142,isP?S.pQueue:S.dQueue,'pd-queue')+text(inner+12,163,activeCPU?activeCPU.label:(isP&&doing('pwait')?'R1 等接收位置':!isP&&doing('metaWait')?'room=0，接收门槛未满足':'本请求状态由调度循环检查'),'pd-sub');
 out+=rect(inner+10,176,278,52,'pd-buffer')+text(inner+20,194,'MetadataBuffers · CPU aux','pd-sub');
 const auxValue=isP?(done('enqueue2')&&!S.released?'y1 · room 42':S.released?'metadata 槽已归还':'待末块写入'):(S.committed?'y1 已提交；aux 槽归还':S.metadataVisible?'y1 · room 42':S.reserved?'预留 aux_index · room 0':'尚未预留');
 out+=text(inner+20,216,auxValue,'pd-value');
 out+=text(inner+12,249,isP?'MooncakeKVSender / 传输 worker':'MooncakeKVReceiver / 接收状态','pd-sub');
 out+=text(inner+12,269,isP?(S.released?'sender 已清理':done('conclude')?'本地 Success':done('enqueue1')?'chunk 已入队 / 等待终态':'等待可发送数据'):(S.rawSuccess?(S.metadataVisible?'Success · metadata 可见':'raw Success · gate 仍未过'):'等待来源终态通知'),'pd-value');
 const gpuActive=doing(gpuID),gpuLabel=isP?(S.released?'本请求不再持有 KV':done('compute2')?'prompt KV 完成 · 已采样 y1':gpuActive?`计算 ${gpuID==='compute1'?'C1: p0 p1':'C2: p2 p3'}`:'等待本请求计算'):(done('decode')?'本轮采样 y2；等待后续轮次':gpuActive?'输入 y1 · 读 prompt KV / 写 y1 KV':'本请求尚未执行 Decode');
 out+=`<g data-device="${which}" data-running="${gpuActive?gpuID:''}">`+rect(inner,306,298,134,'pd-gpu '+(gpuActive?'active':''))+text(inner+12,328,'GPU · ModelRunner / 模型与 KV','pd-module')+text(inner+12,351,gpuLabel,'pd-sub')+kvCells(which,inner+18);
 out+=text(inner+12,424,isP?(S.released?'解除请求保护；缓存可保留':'KV 格 = 对应 token 的各层缓存'):(S.decodeKV?'追加 KV(y1)；y2 尚无 KV':S.dKV===4?'4 个 prompt 位置已写入':S.reserved?'虚线：预留地址，尚未写入':'R1 尚未分配 KV 地址'),'pd-sub');
 out+=bar(inner,436,298,gpuID)+'</g>';
 return out;
}
function tracks(){
 const x=192,w=724,y=518,row=25,map=t=>x+w*t/tl.duration;
 let out=text(28,488,'同一时钟下的执行轨道','pd-module')+text(932,488,'讲解时间 / 非实测；同列为同刻','pd-sub','end');
 for(let n=0;n<=tl.duration;n+=5)out+=text(map(n),510,n+' s','pd-tick','middle');
 for(let i=0;i<M.lanes.length;i++){
  const [id,label]=M.lanes[i],top=y+i*row;
  out+=text(28,top+16,label,'pd-lane-label')+rect(x,top,w,23,'pd-lane',3);
  for(const e of tl.events.filter(e=>e.lane===id)){
   const left=map(e.start),width=map(e.end)-left,on=S.t>=e.start&&S.t<e.end;
   out+=`<g data-event="${e.id}" data-active="${on}" class="pd-event ${e.kind} ${on?'active':''}"><title>${esc(e.label)} · ${e.start}–${e.end} s</title>`+rect(left,top,width,23,'',3);
   const max=Math.floor(width/6.6),label=e.label.length>max?(max>=5?e.label.slice(0,max-1)+'…':''):e.label;
   if(label)out+=text(left+width/2,top+15,label,'pd-event-label','middle');
   out+='</g>';
  }
 }
 out+=`<path d="M${map(S.t)},514 V668" class="pd-cursor"/><circle cx="${map(S.t)}" cy="514" r="3" class="pd-cursor-head"/>`;
 out+=text(28,688,'虚线条 = R1 等待条件，CPU 可继续循环；空白 = 本图未安排活动。','pd-sub');
 return `<g class="pd-tracks" data-clock="${S.t.toFixed(3)}">${out}</g>`;
}
function draw(){
 let out='<defs>'+['control','data','kv'].map(k=>`<marker id="pd-arrow-${k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="${k}"/></marker>`).join('')+'</defs>';
 out+=text(28,28,'R1 · prompt [p0 p1 p2 p3] → 首 token y1 → 下一 token y2','pd-board-title');
 out+=side('p')+side('d');
 out+=text(480,105,'D → P · 目标索引 / aux_index','pd-link-label','middle')+flow('destination',geometry.destination,'8,9,12,13','control',92);
 out+=text(480,183,'P → D · y1 / room 等 aux','pd-link-label','middle')+flow('aux',geometry.aux,'y1 + 42','data',78);
 out+=text(480,244,'P → D · 终态控制消息','pd-link-label','middle')+flow('notify',geometry.notify,'Success','control',76);
 out+=text(480,298,'KV 数据路径 · 跨主机 RDMA','pd-link-label','middle');
 out+=wire(geometry.kv,'kv');
 for(const [x,w,label] of [[372,50,'NIC P'],[448,64,'交换机'],[538,50,'NIC D']])out+=rect(x,320,w,30,'pd-network')+text(x+w/2,315,label,'pd-tick','middle');
 if(doing('copy1'))out+=flow('copy1',geometry.kv,'KV(p0,p1)','kv',94,false);
 if(doing('copy2'))out+=flow('copy2',geometry.kv,'KV(p2,p3)','kv',94,false);
 out+=text(480,411,'C1 / C2 = 两个输入 chunk','pd-link-label','middle')+text(480,431,'GPU KV 不经过 CPU 数据缓存','pd-link-label','middle');
 if(doing('prebuilt')){const points=[[777,280],[777,306]];out+=wire(points,'data')+text(790,300,'y1 → FutureMap','pd-sub');if(!reduced.matches)out+=`<circle cx="777" cy="${280+26*progress('prebuilt')}" r="3" class="pd-cursor-head" data-motion="prebuilt"/>`;}
 return out+tracks();
}
function render(){
 S=M.state(tl,time);time=S.t;
 const scene=$('pd-handoff-scene');scene.innerHTML=draw();scene.dataset.time=time.toFixed(3);scene.dataset.case=tl.option;
 $('pd-handoff-seek').value=time;$('pd-handoff-seek').max=tl.duration;
 $('pd-handoff-clock').textContent=`${time.toFixed(1)} / ${tl.duration.toFixed(1)} s`;
 $('pd-handoff-toggle').textContent=playing?'暂停':time>=tl.duration?'重播':'播放';$('pd-handoff-toggle').setAttribute('aria-pressed',String(playing));
 $('pd-handoff-prev').disabled=time<=0;$('pd-handoff-next').disabled=time>=tl.duration;
 const key=tl.option+S.phase.start;
 if(key!==phaseKey){phaseKey=key;$('pd-handoff-caption-title').textContent=S.phase.title;$('pd-handoff-caption').textContent=S.phase.text;
  $('pd-handoff-source').innerHTML=S.phase.refs.map(k=>`<a href="${M.sourceUrl(k)}" target="_blank" rel="noopener">${esc(M.sources[k][0])}:${M.sources[k][1]} ↗</a>`).join('');}
 const active=S.active.filter(e=>e.kind!=='wait').map(e=>`${M.lanes.find(l=>l[0]===e.lane)[1]}：${e.label}`);
 $('pd-handoff-now').textContent=active.join('；')||(tl.option==='capacity'?'R1 等待 D 接收预算；窗口结束后仍未获准。':'本窗口内的处理已完成，可回拖查看条件。');
 $('pd-handoff-proof').innerHTML=`<span>目标已预留 <b>${S.reserved?'是':'否'}</b></span><span>P / D prompt KV <b>${S.pKV} / ${S.dKV}</b></span><span>D 已提交 y1 <b>${S.committed?'是':'否'}</b></span><span>P 本侧收尾 <b>${S.released?'是':'否'}</b></span><span>D 输出 <b>${S.dOutput.join(' · ')||'尚无'}</b></span>`;
}
function stop(){playing=false;cancelAnimationFrame(raf);last=0;}
function frame(stamp){if(!playing)return;if(last)time+=Math.min(.1,(stamp-last)/1000)*rate;last=stamp;if(time>=tl.duration){time=tl.duration;stop();}render();if(playing)raf=requestAnimationFrame(frame);}
function seek(t){stop();time=t;render();}
function select(option,updateURL=true){stop();tl=M.build(option);time=0;phaseKey='';$('pd-handoff-case').value=tl.option;$('pd-handoff-case-note').textContent=M.cases[tl.option][1];if(updateURL){const u=new URL(location.href);u.searchParams.set('pdcase',tl.option);history.replaceState(null,'',u);}render();}
host.hidden=false;
select(new URL(location.href).searchParams.get('pdcase')||'normal',false);
const resume=window.SiteUpdate?.take('pd-handoff');
if(resume){select(resume.option,false);seek(Math.max(0,Math.min(tl.duration,Number(resume.time)||0)));}
window.SiteUpdate?.register('pd-handoff',()=>{stop();render();return {option:tl.option,time};});
$('pd-handoff-case').addEventListener('change',e=>select(e.target.value));
$('pd-handoff-toggle').addEventListener('click',()=>{if(playing)stop();else{if(time>=tl.duration)time=0;playing=true;last=0;raf=requestAnimationFrame(frame);}render();});
$('pd-handoff-reset').addEventListener('click',()=>seek(0));
$('pd-handoff-seek').addEventListener('input',e=>seek(Number(e.target.value)));
$('pd-handoff-prev').addEventListener('click',()=>seek([...tl.phases].reverse().find(p=>p.start<time-.05)?.start||0));
$('pd-handoff-next').addEventListener('click',()=>seek(tl.phases.find(p=>p.start>time+.05)?.start??tl.duration));
$('pd-handoff-rate').addEventListener('change',e=>{rate=Number(e.target.value);});
addEventListener('popstate',()=>select(new URL(location.href).searchParams.get('pdcase')||'normal',false));
addEventListener('pagehide',stop);reduced.addEventListener('change',render);
})();
