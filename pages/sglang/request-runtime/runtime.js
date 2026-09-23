(function(){
 'use strict';
 const model=window.RequestRuntimeModel,diagram=window.RuntimeSequence,host=document.querySelector('[data-runtime-lab]');
 if(!host||!model||!diagram)return;
 const get=id=>host.querySelector('#'+id);
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 let frames,index=0,timer=null,overview=false;
 const column=142,top=56,pitch=64;
 const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 function stop(pauseMotion=true){clearInterval(timer);timer=null;get('play').textContent='播放';get('play').setAttribute('aria-pressed','false');if(pauseMotion)get('sequence-svg')?.pauseAnimations();}
 function arrow(x1,y1,x2,y2,label,kind,active,pending,hop=0){
  const self=y1===y2;
  const d=self?`M ${x1} ${y1} h 58 v 22 h -58`:`M ${x1} ${y1} L ${x2} ${y2}`;
  const labelY=self?y1-10:(y1+y2)/2;
  const labelX=self?x1+29:x1+8;
  return `<path class="seq-arrow ${kind} ${active?'is-current':''} ${pending?'is-pending':''}" d="${d}" marker-end="url(#arrow-${kind})"/>
   <text class="seq-message" x="${labelX}" y="${labelY}"${self?' text-anchor="middle"':''}>${esc(label)}</text>
   ${active?`<circle class="seq-packet ${kind}" visibility="${reduced.matches?'visible':'hidden'}" r="5" cx="${reduced.matches?x1:0}" cy="${reduced.matches?y1:0}">${!reduced.matches?`<set attributeName="visibility" to="visible" begin="${hop*.55}s"/><animateMotion begin="${hop*.55}s" dur=".55s" path="${d}" fill="freeze"/>`:''}</circle>`:''}`;
 }
 function draw(){
  const viewport=Math.max(180,get('sequence-scroll').clientWidth),width=overview?frames.length*column+24:viewport;
  const activeLanes=new Set(diagram.event(frames[index]).edges.flatMap(([a,b])=>[a,b]));
  const shown=diagram.lanes.map((_,i)=>i).filter(i=>overview||activeLanes.has(i));
  const sy=i=>top+shown.indexOf(i)*pitch,height=shown.length*pitch+28;
  get('sequence-scroll').classList.toggle('sequence-focus',!overview);
  host.querySelector('.sequence-labels').style.height=height+'px';
  host.querySelectorAll('.sequence-label').forEach((el,i)=>el.style.display=shown.includes(i)?'':'none');
  let svg=`<svg id="sequence-svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="group" aria-label="${overview?'完整请求流程，步骤从左到右，可点击跳转':'当前步骤的交接图'}"><defs>`;
  for(const k of ['message','compute','wait','abort','finish','memory'])svg+=`<marker id="arrow-${k}" class="seq-marker ${k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z"/></marker>`;
  svg+='</defs>';
  shown.forEach(i=>svg+=`<rect class="seq-lane lane-${i}" x="0" y="${sy(i)-25}" width="${width}" height="55"/><line class="seq-lifeline" x1="0" x2="${width}" y1="${sy(i)}" y2="${sy(i)}"/>`);
  frames.forEach((f,i)=>{
   if(!overview&&i!==index)return;
   const e=diagram.event(f),active=i===index,gap=overview?(e.edges.length===3?42:52):(e.edges.length===3?Math.min(90,(viewport-70)/3):52),x=overview?i*column+12:viewport/2-(e.edges.length===1&&e.edges[0][0]===e.edges[0][1]?46:17)-(e.edges.length-1)*gap/2;
   svg+=`<g data-step="${i}" class="seq-event ${i<index?'past':active?'current':'future'}" tabindex="0" role="button" aria-label="步骤 ${i+1}：${esc(f.title)}" aria-pressed="${active}"><title>${esc(f.title)}</title><rect class="seq-column" x="${x-4}" y="0" width="${column-5}" height="${height-5}" rx="8"/><text class="seq-step-label" x="${x+8}" y="20">${String(i+1).padStart(2,'0')} ${e.label}</text>`;
   if(e.sample)svg+=`<g class="seq-computation"><rect x="${x+3}" y="${sy(3)-22}" width="128" height="48" rx="6"/><text x="${x+10}" y="${sy(3)-8}">GPU 前向</text>${[0,1,2].map(k=>`<rect class="compute-bar bar-${k}" x="${x+10+k*22}" y="${sy(3)+10}" width="17" height="7" rx="2"/>`).join('')}</g>`;
   e.edges.forEach(([from,to,label],j)=>svg+=arrow(x+17+j*gap,sy(from),x+17+j*gap,sy(to),label,e.kind,active,e.pending,j));
   if(e.sample)svg+=`<g class="seq-sample" transform="translate(${x+87},${sy(3)-15})"><rect width="36" height="28" rx="5"/><text x="18" y="19" text-anchor="middle">${e.sample}</text></g>`;
   if(overview&&f.waiting&&f.id==='queued')svg+=`<g class="seq-queue" transform="translate(${x+20},${sy(2)+32})"><rect width="82" height="18" rx="3"/><text x="41" y="13" text-anchor="middle">▣ R1  ·  ·</text></g>`;
   if(f.id==='abort-pending')svg+=`<text class="seq-stop" x="${x+103}" y="${sy(2)+8}">!</text>`;
   if(f.id==='invalid'||f.id==='abort-queued')svg+=`<text class="seq-stop" x="${x+91}" y="${sy(f.id==='invalid'?1:2)+8}">×</text>`;
   svg+='</g>';
  });
  svg+='</svg>';get('sequence-scroll').innerHTML=svg;
  if(overview){const sc=get('sequence-scroll');sc.scrollTo({left:Math.max(0,index*column-sc.clientWidth/2+column/2),behavior:reduced.matches?'instant':'smooth'});}else get('sequence-scroll').scrollLeft=0;
 }
 function paintCells(container,count,prefix,kind){
  for(const [i,cell] of [...container.children].entries()){
   cell.classList.toggle('filled',i<count);cell.classList.toggle('cached',kind==='cache'&&i<count);
   cell.textContent=prefix==='kv'?(i<4?`p${i+1}`:`y${i-3}`):`y${i+1}`;
  }
 }
 function render(){
  const f=frames[index],focusStep=document.activeElement?.dataset.step;
  get('step').value=index;get('step-count').textContent=`${index+1} / ${frames.length}`;
  get('previous').disabled=index===0;get('next').disabled=index===frames.length-1;
  get('step').setAttribute('aria-valuetext',`${index+1}：${f.title}`);
  get('runtime-status').textContent=`${String(index+1).padStart(2,'0')} · ${f.title}`;
  get('runtime-snapshot').dataset.frame=f.id;
  draw();
  if(focusStep!==undefined)get('sequence-scroll').querySelector(`[data-step="${index}"]`)?.focus({preventScroll:true});
  get('request-slot').classList.toggle('occupied',f.slot);get('request-slot').textContent=f.slot?'R1':'空';
  get('queue-chip').classList.toggle('occupied',f.waiting);get('queue-chip').textContent=f.waiting?'R1':'空';
  get('kv-owner').textContent=f.cached?'前缀缓存接管':f.kv?'R1 使用中':f.forwards?'已归还':'尚未写入';
  get('kv-owner').classList.toggle('cached',f.cached>0);
  paintCells(get('kv-cells'),f.kv+f.cached,'kv',f.cached?'cache':'request');
  paintCells(get('sample-cells'),f.sampled,'y','sample');paintCells(get('visible-cells'),f.visible,'y','visible');
  get('memory-board').dataset.kv=f.kv;get('memory-board').dataset.cached=f.cached;
  get('output-board').dataset.sampled=f.sampled;get('output-board').dataset.visible=f.visible;
  get('step-caption').textContent=caption(f);
  get('step-explanation').textContent=f.explanation;
  get('step-fields').textContent=`前端：${f.front}；调度：${f.scheduler}；本轮：${f.batch}。dispatched=${f.dispatched}，abort_sent=${f.abortSent}，to_finish=${f.toFinish?'abort':'无'}，finished=${f.finished}。`;
  get('step-source').href=model.sourceURL(f.source);get('step-source').textContent=`源码 · ${model.sources[f.source][2]} ↗`;
  if(index===frames.length-1)stop(false);
 }
 function caption(f){
  const id=f.id;
  if(id==='prefill'||id.startsWith('decode-'))return `${f.id==='prefill'?'输入 p1…p4':`输入 y${f.sampled-1}`} → 计算 → 写入 KV → 采样 y${f.sampled}；新采样的 token 尚未回传。`;
  if(id==='release')return f.cached?'请求槽位变空；KV 留在原位置，由前缀缓存接管。':'请求槽位和本例 KV 一起归还，槽位可再分配。';
  if(id==='dispatch'||id==='cancel-sent')return '虚线箭头表示消息已派发；后端接收与处理发生在后续步骤。';
  if(id==='abort-pending')return '待结束已登记；已有执行还要收尾，KV 槽位仍被持有。';
  if(id==='abort-queued')return 'R1 从等待队列移出；取消回告仍在回程。';
  if(id==='invalid')return '请求止于前端，没有进入调度和 GPU 泳道。';
  if(id==='queued')return 'R1 到达调度队列；尚未提交 GPU 执行。';
  if(id==='tokenize')return '文本在前端转换为 p1…p4；这些是输入位置标签。';
  if(id==='input')return '从客户端发出 R1，先到前端登记与校验。';
  if(id==='finish')return '调度侧确认结束；资源与输出各自还需要收尾。';
  if(id==='abort-result')return '在途执行经过结果处理边界；后续才释放请求资源。';
  if(id==='abort-front')return '前端收到取消回告，结束等待。';
  if(id==='final'&&f.abortSent)return '取消结束回到前端；已发出的首段文字保留，额外文本不在此图模拟。';
  return '沿 ID → 反分词 → 前端 → 客户端的箭头，看输出走完回程。';
 }
 function rebuild(){stop();index=0;get('outputs').disabled=get('scenario').value!=='normal';frames=model.build({scenario:get('scenario').value,outputs:Number(get('outputs').value),cache:get('cache').checked});get('step').max=frames.length-1;render();}
 ['scenario','outputs','cache'].forEach(id=>get(id).addEventListener('change',rebuild));
 get('previous').addEventListener('click',()=>{stop();index=Math.max(0,index-1);render();});
 get('next').addEventListener('click',()=>{stop();index=Math.min(frames.length-1,index+1);render();});
 get('reset').addEventListener('click',rebuild);
 get('step').addEventListener('input',()=>{stop();index=Number(get('step').value);render();});
 get('show-sequence').addEventListener('change',()=>{overview=get('show-sequence').checked;draw();});
 let observedWidth=0;new ResizeObserver(entries=>{const w=entries[0].contentRect.width;if(w!==observedWidth){observedWidth=w;if(frames)draw();}}).observe(get('sequence-scroll'));
 function selectEvent(e){const target=e.target.closest('[data-step]');if(!target)return;if(e.type==='keydown'&&!['Enter',' '].includes(e.key))return;e.preventDefault();stop();index=Number(target.dataset.step);render();}
 get('sequence-scroll').addEventListener('click',selectEvent);get('sequence-scroll').addEventListener('keydown',selectEvent);
 get('play').addEventListener('click',()=>{if(timer){stop();return;}if(index===frames.length-1){index=0;render();}get('sequence-svg')?.unpauseAnimations();get('play').textContent='暂停';get('play').setAttribute('aria-pressed','true');timer=setInterval(()=>{index++;render();},2200);});
 document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});window.addEventListener('pagehide',stop);
 host.hidden=false;rebuild();
})();
