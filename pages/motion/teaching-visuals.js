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
 // Shared-canvas lessons use the wide drawing as soon as the PC stage has
 // room for it. Sampling/streaming keep their accepted 600-unit boundary.
 const sharedWide=()=>{const film=viewport.closest('.mechanism-film');return !!film?.classList.contains('is-shared-canvas')&&!film.classList.contains('is-generation-sampling');};
 const narrowAt=()=>viewport.clientWidth<(sharedWide()?520:600);
 let narrow=narrowAt(),last=0,dead=false,resizeFrame=0,resizeTimeout=0;
 const paint=build({node,txt,rect,path,set,show,text,box,wire,token,move});
 function render(p){last=p;narrow=narrowAt();svg.setAttribute('viewBox',`0 0 ${narrow?360:720} ${height(narrow)}`);svg.dataset.teachingKind=kind;paint(p,narrow);}
 const observer=new ResizeObserver(()=>{if(!dead)render(last);});observer.observe(viewport);
 // A resized-away board must not repaint a later renderer's SVG. Cancel both
 // the animation frame and the debounce timer when the player replaces it.
 const repaintOnResize=()=>{resizeFrame=0;resizeTimeout=0;if(dead)return;render(last);};
 const schedule=()=>{if(dead)return;if(resizeFrame)cancelAnimationFrame(resizeFrame);if(resizeTimeout)clearTimeout(resizeTimeout);resizeFrame=requestAnimationFrame(repaintOnResize);resizeTimeout=setTimeout(repaintOnResize,50);};
 const mq=window.matchMedia('(max-width:650px)');
 window.addEventListener('resize',schedule,{passive:true});mq.addEventListener?.('change',schedule);
 return {paint(_a,_b,_u,p){render(p);},destroy(){dead=true;observer.disconnect();cancelAnimationFrame(resizeFrame);clearTimeout(resizeTimeout);window.removeEventListener('resize',schedule);mq.removeEventListener?.('change',schedule);}};
}
window.TeachingDrawing={board,phase,lerp,clamp};
const visuals=window.MechanismVisuals||(window.MechanismVisuals={});
visuals['f-deployment']=(svg,viewport,s)=>{
 const inner=board(svg,viewport,'deployment',m=>m?440:420,H=>{
 const {txt,rect,text,box,show,token,move,set}=H;
 // Two complete roles stay side by side at every width. Source KV stays in P;
 // copies are dashed, labelled payloads that travel below the KV labels so the
 // handoff never runs through either role's text.
 const panels=[rect(),rect()],heads=[txt('Prefill 角色','teach-heading'),txt('Decode 角色','teach-heading')],models=[txt('可执行完整模型','teach-small'),txt('可执行完整模型','teach-small')];
 const sourceLabel=txt('源 KV · 留在 P','teach-small'),copyLabel=txt('到达的 KV 副本','teach-small');
 const firstLabel=txt('首个输出（尚未计算 KV）','teach-small'),inputLabel=txt('新输入（同一 y1）','teach-small');
 const addLabel=txt('本地计算新增','teach-small');
 const src=[0,1].map(i=>token('source-kv-'+i,'位置 '+(i+1),68,'teach-token teach-color-'+i));
 const dest=[0,1].map(i=>token('target-kv-'+i,'等待副本',68,'teach-token teach-empty'));
 const copies=[0,1].map(i=>token('transfer-kv-'+i,'副本 '+(i+1),60,'teach-token teach-empty teach-color-'+i));
 const first=token('first-token','y1',52,'teach-token teach-color-2'),input=token('decode-input','y1 副本',60,'teach-token teach-empty teach-color-2'),inputGhost=token('decode-input-wait','等待 y1',64,'teach-token teach-empty'),added=token('decode-kv-y1','y1 的 KV',100,'teach-token teach-color-2'),second=token('second-token','y2',52,'teach-token teach-color-4');
 const pNote=txt('','teach-small'),pDetail=txt('','teach-small'),dNote=txt('','teach-small'),requirements=txt('','teach-small'),total=txt('','teach-small'),result=txt('','teach-heading');
 const secondLabel=txt('下一 token','teach-small'),handoff=txt('交接：历史 KV 副本 + 下一轮输入 ID','teach-small'),pInputCaption=txt('待作为 D 的输入','teach-small'),dInputCaption=txt('到达后作为新输入','teach-small');
 return(p,m)=>{
  const W = m ? 360 : 720;
  const a = {x: m ? 2 : 16, y: 20};
  const b = {x: m ? 210 : 382, y: 20};
  const pw = m ? 148 : 322, ph = 354, step = m ? 72 : 74, tw = m ? 64 : 68;
  panels.forEach((v,r)=>{const q=r?b:a;box(v,q.x,q.y,pw,ph);text(heads[r],q.x+8,q.y+28);text(models[r],q.x+8,q.y+48);});
  text(handoff,W/2,12);handoff.setAttribute('text-anchor','middle');
  const u=phase(p,2,2.6),u2=phase(p,2.55,3),y2u=phase(p,4);
  const copiesArrived=u>=1,y1Arrived=u2>=1,ready=copiesArrived&&y1Arrived;
  // Source KV and target slots are separate, stationary objects. The source
  // appears only as the first Prefill step completes; its "stays in P" wording
  // begins with that output, not while the input is still being processed.
  src.forEach((t,i)=>{set(t.r,{width:tw});set(t.t,{x:tw/2});move(t,a.x+8+i*step,a.y+82,phase(p,0));});
  text(sourceLabel,a.x+8,a.y+72);
  text(firstLabel,a.x+8,a.y+160);move(first,a.x+8,a.y+168,phase(p,1));text(pInputCaption,a.x+8,a.y+226);
  text(pNote,a.x+8,a.y+246,p<1?'输入处理中':p<4?'源 KV 2/2 · 留在 P':'源 KV 2/2 · 仍保留');
  text(pDetail,a.x+8,a.y+286,'两个位置已经过模型');show(pDetail,phase(p,1));
  // The destination slot stays in place. Its "waiting" caption fades before
  // the payload arrives, then the received copy is written into the slot at
  // the same coordinates, so no moving card has to cross the KV row text.
  dest.forEach((t,i)=>{
   set(t.r,{width:tw});set(t.t,{x:tw/2});
   const received=copiesArrived,labelOpacity=received?1:1-phase(p,2.05,2.3);
   t.t.textContent=received?'副本 '+(i+1):'等待副本';
   set(t.g,{class:'teach-token '+(received?'teach-color-'+i:'teach-empty')});
   show(t.t,labelOpacity);
   move(t,b.x+8+i*step,b.y+82,1);
  });
  text(copyLabel,b.x+8,b.y+72);
  text(inputLabel,b.x+8,b.y+160);move(inputGhost,b.x+8,b.y+168);show(inputGhost.g,p>=2&&p<2.55?1:0);
  text(dInputCaption,b.x+8,b.y+226);
  text(addLabel,b.x+8,b.y+246,m?'本地新增':'本地计算新增');
  // Copies start below each source KV, then move diagonally through the empty
  // band between the KV row and the input labels. They are dashed and labelled
  // so an in-flight payload cannot be mistaken for a source or an existing target.
  copies.forEach((t,i)=>{
   const cw=m?56:60,sx=a.x+8+i*step+(tw-cw)/2,dx=b.x+8+i*step+(tw-cw)/2;
   set(t.r,{width:cw,height:26});set(t.t,{x:cw/2,y:18});
   // Travel and stop in the empty band below both KV rows. The payload never
   // enters either text row; the received copy is written into the slot above.
   move(t,lerp(sx,dx,u),a.y+118,u>0?1-phase(p,2.65,2.95):0);
   t.g.dataset.role='copy';t.g.dataset.received=u.toFixed(3);t.g.dataset.copyIndex=String(i+1);
  });
  const iw=60,ix0=a.x+8+tw+8,ix1=b.x+8+(m?0:0);
  set(input.r,{width:iw,height:26});set(input.t,{x:iw/2,y:18});
  move(input,lerp(ix0,ix1,u2),a.y+168,u2>0?1:0);input.g.dataset.role='copy';input.g.dataset.received=u2.toFixed(3);
  src.forEach(t=>t.g.dataset.role='source');dest.forEach(t=>t.g.dataset.role='target');
  set(added.r,{width:m?78:100});set(added.t,{x:m?39:50});move(added,b.x+8,b.y+252,phase(p,3));
  text(secondLabel,m?b.x+pw-8:b.x+96,b.y+246);set(secondLabel,{'text-anchor':m?'end':'start'});show(secondLabel,y2u);
  set(second.r,{width:m?46:52});set(second.t,{x:m?23:26});
  move(second,b.x+96,b.y+252,y2u);
  // Counts follow what is already visible: the two historical KV copies land
  // at p2.6, the input y1 lands at p3, and D may start only when both are in.
  const computing=ready&&p<4,kvReady=p>=4,y2Ready=p>=5;
  text(requirements,b.x+8,b.y+296,!copiesArrived?(u>0?(u2>0?'副本在途 · y1 在途':'副本在途 · y1 待发'):'副本 0/2 · 等待'):!y1Arrived?'副本 2/2 · y1 在途':'副本 2/2 · y1 已到');
  text(total,b.x+8,b.y+316,!copiesArrived?(u>0?'可用 0/3 · 副本在途':'可用 0/3 · 等待'):!y1Arrived?'可用 2/3 · 等 y1':computing?'可用 2/3 · 计算 y1 中':kvReady?'可用 3/3 · y1 KV 完成':'可用 2/3');
  text(dNote,b.x+8,b.y+336,!copiesArrived?(u>0?'副本在途 · 等 y1':'输入未齐 · 等待'):!y1Arrived?'历史 KV 已到 · 等 y1':computing?'输入已齐 · 计算 y1':kvReady&&!y2Ready?'y1 KV 已写入 · 生成 y2':'y1 KV 已写入 · 选出 y2');
  total.dataset.ready=String(ready);total.dataset.available=String(copiesArrived?(kvReady?3:2):0);
  requirements.dataset.copies=String(copiesArrived?2:0);requirements.dataset.y1=String(y1Arrived?1:0);
  requirements.dataset.gate=ready?'open':'closed';requirements.dataset.copyProgress=u.toFixed(3);requirements.dataset.y1Progress=u2.toFixed(3);
  text(result,8,m?396:402,p<3?'历史 KV 与 y1 齐备后，D 才能 Decode':p<4?'D 已收到输入，y1 的 KV 尚未写完':p<5?'D 正在用同一请求的历史生成 y2':'D 选出 y2，继续自回归生成');
  if(m)result.setAttribute('font-size','14');
 };
 });
 const panel=window.DeploymentAdoption?.mount('contrast',viewport)||null;
 return {paint(...args){inner.paint(...args);},destroy(){panel?.destroy();inner.destroy();}};
};
})();
