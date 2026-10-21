/* Foundation-chapter camera. It transforms only the teaching drawing, never the
 * page, and keeps user zoom/pan independent from the animation timeline. */
(function(root){
'use strict';
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
class MotionCamera{
 constructor({camera,plane,svg,zoomOut,zoomIn,zoomFit,zoomValue}){
  this.camera=camera;this.plane=plane;this.svg=svg;this.zoomOut=zoomOut;this.zoomIn=zoomIn;this.zoomFit=zoomFit;this.zoomValue=zoomValue;
  this.zoom=1;this.x=0;this.y=0;this.min=1;this.max=2.2;this.drag=null;this.frame=0;
  this.onDown=this.onDown.bind(this);this.onMove=this.onMove.bind(this);this.onUp=this.onUp.bind(this);this.onWheel=this.onWheel.bind(this);this.onKey=this.onKey.bind(this);this.refresh=this.refresh.bind(this);
  this.camera.addEventListener('pointerdown',this.onDown);this.camera.addEventListener('wheel',this.onWheel,{passive:false});this.camera.addEventListener('keydown',this.onKey);
  root.addEventListener('pointermove',this.onMove);root.addEventListener('pointerup',this.onUp);root.addEventListener('pointercancel',this.onUp);
  this.zoomOut?.addEventListener('click',()=>this.setZoom(this.zoom-.25));this.zoomIn?.addEventListener('click',()=>this.setZoom(this.zoom+.25));this.zoomFit?.addEventListener('click',()=>this.fit());
  if('ResizeObserver'in root){this.ro=new ResizeObserver(()=>this.refresh());this.ro.observe(this.camera);this.ro.observe(this.svg);}else root.addEventListener('resize',this.refresh,{passive:true});
  if('MutationObserver'in root){this.mo=new MutationObserver(()=>this.refresh());this.mo.observe(this.svg,{attributes:true,attributeFilter:['viewBox']});}
  this.refresh();
 }
 measure(){
  const r=this.camera.getBoundingClientRect(),vb=this.svg.viewBox?.baseVal;
  if(!r.width||!r.height||!vb||!vb.width||!vb.height)return null;
  const scale=Math.min(r.width/vb.width,r.height/vb.height);
  return {r,vb,scale};
 }
 refresh(){
  const m=this.measure();if(!m)return;
  // Keep the useful text sizes in screen pixels when a tall mechanism is fit
  // into the stage. Positions are unchanged; only the SVG type is compensated.
  // The parallel boards now use a shorter viewBox; a slightly higher ceiling
  // there keeps their 10.5px small type readable at tablet widths without
  // inflating labels in lessons that still use the taller boards.
  const key=this.camera.closest('.mechanism-film')?.dataset.motion;
  const boost=Math.min(key==='f-parallel'?2:1.7,1/m.scale);
  const f=(target)=>`${(target*boost).toFixed(3)}px`;
  const set=(name,value)=>{if(this.camera.style.getPropertyValue(name)!==value)this.camera.style.setProperty(name,value);};
  const narrow=innerWidth<=760,veryNarrow=innerWidth<=340;const batching=key==='f-batching'&&!!this.svg.dataset.teachingKind;
  const basics=key==='f-tokens'||key==='f-generation'||key==='f-attention'||key==='f-cache';
  const attention=key==='f-attention';
  const compact=narrow&&(key==='f-parallel'||key==='f-communication');
  const tp=key==='f-parallel'&&this.svg.dataset.parallelKind==='tp';
  const hardware=key==='f-hardware';
  // Hardware keeps its own type scale: the 390px stage is height-limited, and the
  // old dense-graph reduction made axis and formula labels unreadable.
  const heading=hardware?(narrow?(veryNarrow?15:16.5):16):attention?(narrow?(veryNarrow?15.2:15.6):15.8):basics?(veryNarrow?13.6:narrow?16:15):batching?(narrow?(veryNarrow?14.5:15.5):16):tp?(narrow?13.2:15.2):compact?13.2:(narrow?12:15);
  const text=hardware?(narrow?(veryNarrow?12.8:13.8):13):attention?(narrow?(veryNarrow?13.2:13.4):13.2):basics?(veryNarrow?11.2:narrow?12.8:12.2):batching?(narrow?(veryNarrow?12.8:13.8):13.5):tp?(narrow?12.2:12.8):compact?12.2:12.2;
  const small=hardware?(narrow?(veryNarrow?11.8:12.8):12):attention?(narrow?(veryNarrow?12.4:12.6):11.8):basics?(veryNarrow?10.8:narrow?11.2:10.5):batching?(narrow?(veryNarrow?11.6:12.4):12.2):tp?(narrow?10.8:11.4):compact?10.6:10.5;
  // Align tall mobile drawings to the left so labels can use the stage margin
  // instead of being clipped by the centered viewBox. Desktop remains centered.
  this.svg.setAttribute('preserveAspectRatio',narrow?'xMinYMid meet':'xMidYMid meet');
  set('--motion-small-size',f(small));set('--motion-text-size',f(text));set('--motion-heading-size',f(heading));
  this.apply();
 }
 clamp(){
  const r=this.camera.getBoundingClientRect(),ww=r.width,hh=r.height;
  if(this.zoom<=1.001){this.zoom=1;this.x=0;this.y=0;return;}
  this.x=clamp(this.x,ww-ww*this.zoom,0);this.y=clamp(this.y,hh-hh*this.zoom,0);
 }
 apply(){
  this.clamp();
  this.plane.style.transform=`translate(${this.x.toFixed(2)}px,${this.y.toFixed(2)}px) scale(${this.zoom.toFixed(3)})`;
  this.camera.dataset.zoom=this.zoom.toFixed(2);this.camera.classList.toggle('is-zoomed',this.zoom>1.001);
  if(this.zoomValue)this.zoomValue.textContent=`${Math.round(this.zoom*100)}%`;
  if(this.zoomIn)this.zoomIn.disabled=this.zoom>=this.max-.001;if(this.zoomOut)this.zoomOut.disabled=this.zoom<=this.min+.001;
 }
 setZoom(next,center){
  const old=this.zoom;this.zoom=clamp(next,this.min,this.max);
  if(center){const r=this.camera.getBoundingClientRect(),cx=center.x-r.left,cy=center.y-r.top;this.x=cx-(cx-this.x)*this.zoom/old;this.y=cy-(cy-this.y)*this.zoom/old;}
  this.apply();
 }
 fit(){this.zoom=1;this.x=0;this.y=0;this.apply();}
 onDown(e){
  if(e.button!==0||this.zoom<=1.001||e.target.closest('button,a,input,select,[role="button"],.motion-camera-tools'))return;
  this.drag={x:e.clientX,y:e.clientY,ox:this.x,oy:this.y};this.camera.setPointerCapture?.(e.pointerId);this.camera.classList.add('is-panning');e.preventDefault();
 }
 onMove(e){if(!this.drag)return;this.x=this.drag.ox+e.clientX-this.drag.x;this.y=this.drag.oy+e.clientY-this.drag.y;this.apply();}
 onUp(e){if(!this.drag)return;this.drag=null;this.camera.classList.remove('is-panning');try{this.camera.releasePointerCapture?.(e.pointerId)}catch{}}
 onWheel(e){if(!(e.ctrlKey||e.metaKey))return;e.preventDefault();this.setZoom(this.zoom+(e.deltaY<0?.15:-.15),{x:e.clientX,y:e.clientY});}
 onKey(e){if(e.target.closest('input,select,button,a,[role="button"]'))return;if(e.key==='+'||e.key==='='){e.preventDefault();this.setZoom(this.zoom+.25);}else if(e.key==='-'){e.preventDefault();this.setZoom(this.zoom-.25);}else if(e.key==='0'){e.preventDefault();this.fit();}}
 destroy(){this.ro?.disconnect();this.mo?.disconnect();root.removeEventListener('pointermove',this.onMove);root.removeEventListener('pointerup',this.onUp);root.removeEventListener('pointercancel',this.onUp);this.camera.removeEventListener('pointerdown',this.onDown);this.camera.removeEventListener('wheel',this.onWheel);this.camera.removeEventListener('keydown',this.onKey);}
}
root.MotionCamera=MotionCamera;
})(typeof globalThis==='object'?globalThis:this);
