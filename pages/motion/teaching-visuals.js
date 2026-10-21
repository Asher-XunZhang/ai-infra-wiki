/* Mechanism-specific pictures. Object nodes persist across playback and seeking.
 * Only the drawing changes continuously; lesson state is a pure function of p. */
(() => {
'use strict';
let boardSerial=0;
const NS='http://www.w3.org/2000/svg',clamp=n=>Math.max(0,Math.min(1,n));
const phase=(p,start,end=start+1)=>{const t=clamp((p-start)/(end-start));return t*t*(3-2*t);};
const lerp=(a,b,t)=>a+(b-a)*t;
function board(svg,viewport,kind,height,build){
 viewport.classList.add('teaching-viewport');viewport.setAttribute('aria-label','可回看的机制画布，窄屏自动排列');
 const node=(tag,attrs={},value)=>{const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(value!==undefined)e.textContent=value;svg.append(e);return e;};
 const markerId='teach-arrow-'+(++boardSerial),defs=node('defs'),marker=document.createElementNS(NS,'marker'),arrow=document.createElementNS(NS,'path');
 for(const[k,v]of Object.entries({id:markerId,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'}))marker.setAttribute(k,v);arrow.setAttribute('d','M0 0L10 5L0 10Z');arrow.setAttribute('fill','#7893a4');marker.append(arrow);defs.append(marker);
 const txt=(value,cls='teach-text')=>node('text',{class:cls},value);
 const rect=(cls='teach-panel')=>node('rect',{class:cls,rx:7});
 const path=(cls='teach-link')=>node('path',{class:cls,fill:'none',pathLength:1});
 const set=(e,a)=>{for(const[k,v]of Object.entries(a))e.setAttribute(k,v);};
 const show=(e,n)=>set(e,{opacity:clamp(n).toFixed(3),'aria-hidden':n<=0?'true':'false'});
 const text=(e,x,y,value)=>{set(e,{x,y});if(value!==undefined)e.textContent=value;};
 const box=(e,x,y,w,h)=>set(e,{x,y,width:w,height:h});
 const wire=(e,d,n=1)=>{set(e,{d,'stroke-dasharray':'1','stroke-dashoffset':1-clamp(n),'marker-end':n>=1?'url(#'+markerId+')':'none'});show(e,n>0?1:0);};
 function token(id,label,width=72,cls='teach-token'){
  const g=node('g',{'data-motion-id':id,class:cls}),r=document.createElementNS(NS,'rect'),t=document.createElementNS(NS,'text');
  set(r,{width,height:34,rx:5});set(t,{x:width/2,y:22,'text-anchor':'middle'});t.textContent=label;g.append(r,t);return {g,r,t,width};
 }
 const move=(a,x,y,opacity=1,scale=1)=>{set(a.g,{transform:`translate(${x.toFixed(2)} ${y.toFixed(2)}) scale(${scale})`});Object.assign(a.g.dataset,{x:x.toFixed(3),y:y.toFixed(3)});show(a.g,opacity);};
 let narrow=viewport.clientWidth<600,last=0,dead=false,resizeFrame=0,resizeTimeout=0;
 const paint=build({node,txt,rect,path,set,show,text,box,wire,token,move});
 function render(p){last=p;narrow=viewport.clientWidth<600;svg.setAttribute('viewBox',`0 0 ${narrow?360:720} ${height(narrow)}`);svg.dataset.teachingKind=kind;paint(p,narrow);}
 const observer=new ResizeObserver(()=>{if(!dead){narrow=viewport.clientWidth<600;render(last);}});observer.observe(viewport);
 // A resized-away board must not repaint a later renderer's SVG. Cancel both
 // the animation frame and the debounce timer when the player replaces it.
 const repaintOnResize=()=>{resizeFrame=0;resizeTimeout=0;if(dead)return;narrow=viewport.clientWidth<600;render(last);};
 const schedule=()=>{if(dead)return;if(resizeFrame)cancelAnimationFrame(resizeFrame);if(resizeTimeout)clearTimeout(resizeTimeout);resizeFrame=requestAnimationFrame(repaintOnResize);resizeTimeout=setTimeout(repaintOnResize,50);};
 const mq=window.matchMedia('(max-width:650px)');
 window.addEventListener('resize',schedule,{passive:true});mq.addEventListener?.('change',schedule);
 return {paint(_a,_b,_u,p){render(p);},destroy(){dead=true;observer.disconnect();cancelAnimationFrame(resizeFrame);clearTimeout(resizeTimeout);window.removeEventListener('resize',schedule);mq.removeEventListener?.('change',schedule);}};
}
window.TeachingDrawing={board,phase,lerp,clamp};
const visuals=window.MechanismVisuals||(window.MechanismVisuals={});
visuals['f-deployment']=(svg,viewport,s)=>board(svg,viewport,'deployment',m=>m?440:420,H=>{
 const {txt,rect,text,box,show,token,move,set}=H;
 // Two complete roles stay side by side at every width. The narrow layout keeps
 // the same reading order and puts the copied KV in explicit destination slots.
 const panels=[rect(),rect()],heads=[txt('Prefill 角色','teach-heading'),txt('Decode 角色','teach-heading')],models=[txt('可执行完整模型','teach-small'),txt('可执行完整模型','teach-small')];
 const sourceLabel=txt('源 KV · 留在 P','teach-small'),copyLabel=txt('到达的 KV 副本','teach-small');
 const firstLabel=txt('首个输出（尚未计算 KV）','teach-small'),inputLabel=txt('新输入（同一 y1）','teach-small');
 const addLabel=txt('本地计算新增','teach-small');
 const src=[0,1].map(i=>token('source-kv-'+i,'位置 '+(i+1),68,'teach-token teach-color-'+i));
 const dest=[0,1].map(i=>token('target-kv-'+i,'等待副本',68,'teach-token teach-empty'));
 const copies=[0,1].map(i=>token('transfer-kv-'+i,'位置 '+(i+1),68,'teach-token teach-color-'+i));
 const first=token('first-token','y1',52,'teach-token teach-color-2'),input=token('decode-input','y1',52,'teach-token teach-color-2'),added=token('decode-kv-y1','y1 的 KV',100,'teach-token teach-color-2'),second=token('second-token','y2',52,'teach-token teach-color-4');
 const pNote=txt('','teach-small'),pDetail=txt('','teach-small'),dNote=txt('','teach-small'),requirements=txt('','teach-small'),total=txt('','teach-small'),result=txt('','teach-heading');
 const secondLabel=txt('下一 token','teach-small'),handoff=txt('交接：历史 KV 副本 + 下一轮输入 ID','teach-small'),pInputCaption=txt('待作为 D 的输入','teach-small'),dInputCaption=txt('到达后作为新输入','teach-small');
 return(p,m)=>{
  const W = m ? 360 : 720;
  const a = {x: m ? 8 : 16, y: 20};
  const b = {x: m ? 188 : 382, y: 20};
  const pw = m ? 164 : 322;
  const ph = 354;
  panels.forEach((v,r)=>{const q=r?b:a;box(v,q.x,q.y,pw,ph);text(heads[r],q.x+10,q.y+28);text(models[r],q.x+10,q.y+48);});
  text(handoff,W/2,12);handoff.setAttribute('text-anchor','middle');
  // Source KV and copied KV are separate objects: the sources never move away.
  src.forEach((t,i)=>move(t,a.x+10+i*74,a.y+82,phase(p,0)));
  text(sourceLabel,a.x+10,a.y+74);
  text(firstLabel,a.x+10,a.y+150);move(first,a.x+10,a.y+160,phase(p,1));text(pInputCaption,a.x+68,a.y+184);
  text(pNote,a.x+10,a.y+244,p<1?'输入处理中':p<4?'源 KV 2/2 · 留在 P':'源 KV 2/2 · 仍保留');
  text(pDetail,a.x+10,a.y+286,'两个位置已经过模型');show(pDetail,phase(p,1));
  dest.forEach((t,i)=>{move(t,b.x+10+i*74,b.y+82);show(t.g,1-phase(p,2));});
  text(copyLabel,b.x+10,b.y+74);
  text(inputLabel,b.x+10,b.y+150);text(dInputCaption,b.x+68,b.y+184);
  text(addLabel,b.x+10,b.y+214);
  const u=phase(p,2),y2u=phase(p,4);
  copies.forEach((t,i)=>{move(t,lerp(a.x+10+i*74,b.x+10+i*74,u),lerp(a.y+82,b.y+82,u),p>2?1:0);t.g.dataset.received=u.toFixed(3);});
  move(input,lerp(a.x+10,b.x+10,u),lerp(a.y+160,b.y+160,u),p>2?1:0);
  set(added.r,{width:m?78:100});set(added.t,{x:m?39:50});move(added,b.x+10,b.y+222,phase(p,3));
  text(secondLabel,b.x+(m?94:122),b.y+216);show(secondLabel,y2u);
  move(second,b.x+(m?94:122),lerp(b.y+246,b.y+222,y2u),y2u);
  // Counts follow what is already visible: arrival, computation, then KV written.
  const arrived=p>=3,computing=arrived&&p<4,kvReady=p>=4,y2Ready=p>=5;
  text(requirements,b.x+10,b.y+284,!arrived?(u>0?'副本在途 · y1 在途':'副本 0/2 · 等待'):'副本 2/2 · y1 已到');
  text(total,b.x+10,b.y+304,!arrived?'可用 0/3':computing?'可用 2/3 · 计算 y1 中':kvReady?'可用 3/3 · y1 KV 完成':'可用 2/3');
  text(dNote,b.x+10,b.y+326,!arrived?'输入未齐 · 等待':computing?'输入已齐 · 正在计算 y1':kvReady&&!y2Ready?'已追加 y1 KV · 生成 y2 中':'已追加 y1 KV · 选出 y2');
  total.dataset.ready=String(kvReady);requirements.dataset.received=String(arrived);
  text(result,8,m?396:402,p<3?'历史 KV 与 y1 齐备后，D 才能 Decode':p<4?'D 已收到输入，y1 的 KV 尚未写完':p<5?'D 正在用同一请求的历史生成 y2':'D 选出 y2，继续自回归生成');
  if(m)result.setAttribute('font-size','14');
 };
});
})();
