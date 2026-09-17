/* A bounded, zoomable scene with one set of original experiment controls. */
(() => {
  'use strict';
  const clamp=(n,low,high)=>Math.max(low,Math.min(high,n));
  for(const [index,lab] of [...document.querySelectorAll('.lab')].entries()){
    const dock=lab.querySelector(':scope > .lab-controls');
    if(!dock)continue;
    const scene=document.createElement('div');scene.className='lab-scene';
    // The status explanation remains readable outside the transformed diagram.
    let next=dock.nextElementSibling;
    while(next&&!next.classList.contains('lab-output')&&next.tagName!=='NOSCRIPT'){
      const following=next.nextElementSibling;scene.append(next);next=following;
    }
    const handoff=scene.querySelector('#handoff-controls');
    if(handoff){
      const settings=document.createElement('details');settings.id=handoff.id;
      settings.className='dock-settings';settings.hidden=handoff.hidden;
      const summary=document.createElement('summary');summary.textContent='交接条件';
      const content=document.createElement('div');content.className='dock-settings-content';
      content.append(...handoff.childNodes);settings.append(summary,content);
      handoff.replaceWith(settings);dock.append(settings);
      settings.addEventListener('keydown',event=>{
        if(event.key==='Escape'&&settings.open){settings.open=false;summary.focus();event.stopPropagation();}
      });
    }
    const frame=document.createElement('div');frame.className='lab-workbench';
    dock.before(frame);frame.append(dock);dock.classList.add('lab-dock');
    dock.setAttribute('role','group');dock.setAttribute('aria-label','当前实验操作');
    const camera=document.createElement('div');camera.className='camera-controls';
    camera.setAttribute('role','group');camera.setAttribute('aria-label','画布视图操作');
    camera.innerHTML='<button type="button" data-camera="out" aria-label="缩小画布">−</button><output class="camera-zoom" aria-label="画布缩放比例"></output><button type="button" data-camera="in" aria-label="放大画布">＋</button><button type="button" data-camera="fit">适应全图</button><button type="button" data-camera="reset">重置</button><span class="camera-hint">拖动平移 · 双指或 Ctrl/⌘ + 滚轮缩放</span>';
    const viewport=document.createElement('div');viewport.className='lab-viewport';viewport.tabIndex=0;
    viewport.id=`lab-viewport-${index}`;viewport.setAttribute('role','region');
    viewport.setAttribute('aria-label','可缩放拖动的动画画布。方向键平移，加减键缩放，0 适应全图，1 重置到原尺寸。');
    for(const button of camera.querySelectorAll('button'))button.setAttribute('aria-controls',viewport.id);
    viewport.append(scene);frame.append(camera,viewport);
    let scale=1,x=0,y=0,autoFit=true;
    const pointers=new Map();
    function bounds(){return {w:viewport.clientWidth,h:viewport.clientHeight,sw:scene.offsetWidth,sh:scene.offsetHeight};}
    function render(){
      const b=bounds();
      const visibleX=Math.min(80,b.sw*scale/2,b.w/2),visibleY=Math.min(80,b.sh*scale/2,b.h/2);
      x=clamp(x,visibleX-b.sw*scale,b.w-visibleX);
      y=clamp(y,visibleY-b.sh*scale,b.h-visibleY);
      scene.style.transform=`translate(${x}px,${y}px) scale(${scale})`;
      viewport.dataset.scale=String(scale);viewport.dataset.x=String(x);viewport.dataset.y=String(y);
      camera.querySelector('output').textContent=Math.round(scale*100)+'%';
      camera.querySelector('[data-camera=out]').disabled=scale<=.1;
      camera.querySelector('[data-camera=in]').disabled=scale>=2.5;
    }
    function fit(){
      autoFit=true;const b=bounds();
      if(!b.w||!b.h||!b.sw||!b.sh)return;
      scale=clamp(Math.min((b.w-32)/b.sw,(b.h-32)/b.sh,1),.1,2.5);
      x=(b.w-b.sw*scale)/2;y=(b.h-b.sh*scale)/2;render();
    }
    function zoom(value,cx=viewport.clientWidth/2,cy=viewport.clientHeight/2){
      autoFit=false;const next=clamp(value,.1,2.5),ratio=next/scale;
      x=cx-(cx-x)*ratio;y=cy-(cy-y)*ratio;scale=next;render();
    }
    function reset(){autoFit=false;scale=1;x=20;y=20;render();}
    camera.addEventListener('click',event=>{
      const action=event.target.closest('button')?.dataset.camera;
      if(action==='in')zoom(scale*1.25);if(action==='out')zoom(scale/1.25);
      if(action==='fit')fit();if(action==='reset')reset();
    });
    const point=event=>{const r=viewport.getBoundingClientRect();return {x:event.clientX-r.left,y:event.clientY-r.top};};
    const gesture=()=>{
      const a=[...pointers.values()];if(a.length<2)return null;
      return {x:(a[0].x+a[1].x)/2,y:(a[0].y+a[1].y)/2,d:Math.hypot(a[0].x-a[1].x,a[0].y-a[1].y)};
    };
    viewport.addEventListener('pointerdown',event=>{
      if(event.button!==0||event.target.closest('a,button,input,select,textarea,summary'))return;
      event.preventDefault();viewport.focus({preventScroll:true});
      pointers.set(event.pointerId,point(event));viewport.setPointerCapture(event.pointerId);
      viewport.classList.add('is-panning');
    });
    viewport.addEventListener('pointermove',event=>{
      if(!pointers.has(event.pointerId))return;
      const previous=pointers.get(event.pointerId),before=gesture();
      pointers.set(event.pointerId,point(event));const after=gesture();autoFit=false;
      if(before&&after&&before.d>0){
        zoom(scale*after.d/before.d,before.x,before.y);x+=after.x-before.x;y+=after.y-before.y;
      }else{x+=point(event).x-previous.x;y+=point(event).y-previous.y;}
      render();
    });
    for(const type of ['pointerup','pointercancel','lostpointercapture'])viewport.addEventListener(type,event=>{
      pointers.delete(event.pointerId);if(!pointers.size)viewport.classList.remove('is-panning');
    });
    viewport.addEventListener('wheel',event=>{
      if(!event.ctrlKey&&!event.metaKey)return;
      event.preventDefault();const p=point(event);zoom(scale*Math.exp(-event.deltaY*.008),p.x,p.y);
    },{passive:false});
    viewport.addEventListener('keydown',event=>{
      if(event.target!==viewport)return;
      if(['+','=','-','0','1','ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key))event.preventDefault();
      if(event.key==='+'||event.key==='=')zoom(scale*1.25);
      else if(event.key==='-')zoom(scale/1.25);
      else if(event.key==='0')fit();else if(event.key==='1')reset();
      else if(event.key.startsWith('Arrow')){autoFit=false;const distance=event.shiftKey?100:30;
        if(event.key==='ArrowLeft')x+=distance;if(event.key==='ArrowRight')x-=distance;
        if(event.key==='ArrowUp')y+=distance;if(event.key==='ArrowDown')y-=distance;render();}
    });
    // Keep zoom/pan through model updates. In fit mode, include resized diagrams.
    new ResizeObserver(()=>autoFit?fit():render()).observe(scene);
    new ResizeObserver(()=>autoFit?fit():render()).observe(viewport);
    // Keep text readable on a phone; full overview is one tap away via Fit.
    if(matchMedia('(max-width:700px)').matches)reset();else fit();
  }
})();
