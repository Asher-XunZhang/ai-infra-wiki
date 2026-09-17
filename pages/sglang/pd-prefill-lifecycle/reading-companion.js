/* A reading cursor over the existing scenario, never a second simulation. */
(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.PrefillReading=api;
})(typeof globalThis!=='undefined'?globalThis:this,()=>{
  'use strict';
  const batchOf=f=>f?.work.batch??f?.after.batch;
  function occurrences(model,phase){
    const result=[];let last=-2;
    model.frames.forEach((frame,index)=>{
      if(frame.phase!==phase)return;
      const batch=batchOf(frame)?.id??null,previous=result.at(-1);
      if(previous&&last===index-1&&previous.batch===batch)previous.frames.push(frame);
      else result.push({batch,start:index,frames:[frame]});
      last=index;
    });
    return result;
  }
  function chooseOccurrence(items,context){
    if(!items.length)return -1;
    const exact=items.findIndex(item=>item.frames.some(f=>f.key===context?.key));
    if(exact>=0)return exact;
    const batch=batchOf(context)?.id,rank=context?.work.rank;
    const ranked=items.findIndex(item=>item.batch===batch&&item.frames.some(f=>f.work.rank===rank));
    if(ranked>=0)return ranked;
    const match=items.findIndex(item=>item.batch===batch);
    return match>=0?match:0;
  }
  function mount({articles,links,onInteract,onFrame,onFull}){
    const $=id=>document.getElementById('reader-'+id),panel=document.getElementById('reading-companion');
    const faults={none:'正常流程',bootstrap_fail:'部分握手失败',bootstrap_abort:'本地取消',bootstrap_wait:'握手等待',transfer_fail:'混合传输终态',repoll_wait:'本地终态复查等待'};
    let model,phase=0,items=[],occurrence=-1,cursor=0,timer=null,playing=false;
    const current=()=>items[occurrence]?.frames[cursor];
    function pause(){clearTimeout(timer);timer=null;playing=false;panel.classList.remove('is-playing');$('play').textContent='▶ 播放此片段';$('play').setAttribute('aria-pressed','false');}
    function render(write=true){
      const frame=current();
      $('empty').hidden=!!frame;$('active').hidden=!frame;$('occurrence').disabled=!frame;
      if(!frame)return;
      const batch=batchOf(frame),frames=items[occurrence].frames;
      $('context').textContent=[batch?.id,batch?.segments.map(s=>`${s.rid} [${s.start}, ${s.end})`).join(' · '),frame.work.rank===undefined?'':`PP${frame.work.rank}`].filter(Boolean).join(' · ')||`${model.requests.length} 条请求 · 接入与准入`;
      panel.style.setProperty('--flow-current',`var(--flow-${frame.kind})`);
      panel.dataset.frameKey=frame.key;
      $('prev').disabled=cursor===0;$('next').disabled=cursor===frames.length-1;
      $('progress').textContent=`动作 ${cursor+1} / ${frames.length}${cursor===frames.length-1?' · 片段结束':''}`;
      window.PrefillOperationLab.render(frame,model,'reader-');
      if(write)onFrame(frame);
    }
    function select(nextPhase,context=current(),write=true){
      const scroll={top:window.scrollY,left:window.scrollX,behavior:'instant'};
      pause();phase=nextPhase;items=occurrences(model,phase);occurrence=chooseOccurrence(items,context);cursor=0;
      if(occurrence>=0){const exact=items[occurrence].frames.findIndex(f=>f.key===context?.key);if(exact>=0)cursor=exact;}
      articles.forEach((article,i)=>article.hidden=i!==phase);
      links.forEach((link,i)=>{if(i===phase)link.setAttribute('aria-current','step');else link.removeAttribute('aria-current');});
      $('occurrence').replaceChildren();
      items.forEach((item,i)=>{const option=document.createElement('option'),ranks=[...new Set(item.frames.map(f=>f.work.rank).filter(r=>r!==undefined))];option.value=i;option.textContent=`${i+1} / ${items.length} · ${item.batch??'请求接入'}${ranks.length?' · '+ranks.map(r=>'PP'+r).join(' → '):''} · ${item.frames.length} 个动作`;$('occurrence').append(option);});
      if(!items.length){const option=document.createElement('option');option.textContent='本用例未执行';$('occurrence').append(option);delete panel.dataset.frameKey;}
      else $('occurrence').value=occurrence;
      $('body').scrollTop=0;
      render(write);
      window.scrollTo(scroll);
    }
    function setModel(next){
      pause();model=next;
      const c=model.config;
      $('scenario').textContent=`${c.requests} 条请求 · 每条 ${c.inputLength} token · ${c.chunkSize?'共享预算 '+c.chunkSize:'不切块'} · batch 上限 ${c.batchSize} · 页大小 ${c.pageSize} · ${faults[c.fault]}`;
      select(phase,null,false);panel.hidden=false;
    }
    function schedule(){
      clearTimeout(timer);if(!playing)return;
      timer=setTimeout(()=>{if(cursor+1===items[occurrence].frames.length){pause();return;}cursor++;render();schedule();},1800);
    }
    function play(restart=false){
      if(!current())return;
      onInteract();if(playing&&!restart){pause();return;}
      if(restart||cursor===items[occurrence].frames.length-1)cursor=0;
      playing=true;panel.classList.add('is-playing');$('play').textContent='Ⅱ 暂停';$('play').setAttribute('aria-pressed','true');render();schedule();
    }
    links.forEach((link,i)=>link.addEventListener('click',event=>{
      if(event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;
      event.preventDefault();onInteract();select(i);
      // A missing phase must still have a shareable chapter, never a stale normal frame.
      if(!current())onFrame(null,articles[i].id);
      else if($('panel').open&&!window.matchMedia('(prefers-reduced-motion: reduce)').matches)play(true);
    }));
    $('occurrence').addEventListener('change',()=>{onInteract();pause();occurrence=Number($('occurrence').value);cursor=0;$('body').scrollTop=0;render();if($('panel').open&&!window.matchMedia('(prefers-reduced-motion: reduce)').matches)play(true);});
    for(const [id,delta] of [['prev',-1],['next',1]])$(id).addEventListener('click',()=>{onInteract();pause();cursor=Math.max(0,Math.min(items[occurrence].frames.length-1,cursor+delta));render();});
    $('play').addEventListener('click',()=>play());$('replay').addEventListener('click',()=>play(true));
    $('full').addEventListener('click',()=>{pause();onFull(current());});
    $('panel').addEventListener('toggle',()=>{if(!$('panel').open)pause();});
    document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});window.addEventListener('pagehide',pause);
    return {setModel,pause,select,current,writePosition:()=>onFrame(current(),current()?null:articles[phase].id)};
  }
  return {occurrences,chooseOccurrence,batchOf,mount};
});
