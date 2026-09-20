/* Focused views of the same versioned scenario packets used by the full timeline. */
(() => {
  'use strict';
  const root=document.querySelector('[data-pp-reader]');if(!root)return;
  const a=LearningDiagram.create(root,{rank:0,loop:5,batch:3,mode:'loop',beat:0,dep:'0:5:recv_out'});
  const {s,$,esc,fmt,clamp,button,select,bind,controls,change,status,evidence,animationControls,animationBind,railChart,mark,phaseColor,ppLegend}=a;
  const catalog=window.PP_SCENARIOS,base=new URL('.',document.currentScript.src),cache=new Map();
  let packet=null,original=null,comparison=null,compareId='slow-link',compareRequest=0;
  const graph=p=>Object.fromEntries(Object.entries(p.model.graph).map(([id,n])=>[id,{id,...n}]));
  const name=id=>catalog.find(c=>c.id===id)?.name||'自定义时间';
  let activeId='baseline';
  const params=new URLSearchParams(location.search);
  if(root.dataset.ppReader==='quick'){
    s.rank=clamp(Math.round(Number(params.get('rank')))||0,0,2);
    if(params.has('loop'))s.loop=Math.max(0,Number(params.get('loop'))||0);
    if(params.has('batch')||params.get('view')==='batch'){s.batch=clamp(Math.round(Number(params.get('batch')))||1,1,5);s.mode='batch';}
  }
  function get(id){const item=catalog.find(c=>c.id===id);if(!item)return Promise.reject(new Error('未知场景'));if(!cache.has(id)){const request=fetch(new URL(item.file,base)).then(r=>{if(!r.ok)throw new Error('无法读取场景');return r.json();});cache.set(id,request);request.catch(()=>cache.delete(id));}return cache.get(id);}
  function rowsForLoop(){
    const g=graph(packet),l=packet.quick.loops.find(l=>l.r===s.rank&&l.n===s.loop),prefix=`${s.rank}:${s.loop}:`,n=k=>g[prefix+k];
    const events=packet.model.events.filter(e=>e.r===s.rank&&e.n===s.loop&&['cpu','wait'].includes(e.kind));
    const cpu=events.map(e=>({...mark(e,e.kind==='wait'?'等待':''),fill:e.kind==='wait'?phaseColor.wait:phaseColor.cpu}));
    const prep=n('prepare_extend'),gpu=n('gpu'),recv=n('recv_out'),kv=n('kv'),proxy=n('proxy_message'),proxyWork=n('proxy_work');
    const waits=events.filter(e=>e.kind==='wait').sort((x,y)=>(y.end-y.start)-(x.end-x.start));
    const waitNote=waits.length?waits.slice(0,2).map(e=>`${e.owner} ${fmt(e.start)}–${fmt(e.end)}`).join('；'):'本轮无依赖等待';
    const rows=[{label:'CPU PP'+s.rank,bars:cpu,gate:recv?.start,note:`主要等待：${waitNote}`}];
    rows.push({label:'GPU PP'+s.rank,bars:gpu?[mark(gpu,`计算 M${l.current}`)]:[],note:gpu?`M${l.current} ${fmt(gpu.start)}–${fmt(gpu.end)}；空白未展开其他 GPU 工作`:'本轮没有提交新的前向'});
    const oldIds=recv?.deps.map(id=>g[id]).filter(v=>v&&v.kind==='message')||[];
    const oldBars=[...oldIds,...['copy','result','send_kv','kv'].map(n).filter(Boolean)].map(v=>mark(v,v.kind==='kv'?'KV → D':v.kind==='message'?'结果':v.kind==='copy'?'D2H':''));
    rows.push({label:l.old?`M${l.old} 交接`:'旧结果',bars:oldBars,gate:kv?.start,note:kv?`接收起点 ${fmt(recv.start)}；KV ${fmt(kv.start)}–${fmt(kv.end)} 在途`:'本轮没有旧结果 / KV 交接'});
    if(gpu){const bars=[];if(prep&&gpu.start>prep.end)bars.push({start:prep.end,end:gpu.start,fill:phaseColor.wait,label:'等提交'});bars.push(mark(gpu,'计算'));if(proxy){if(proxy.start>gpu.end)bars.push({start:gpu.end,end:proxy.start,fill:phaseColor.wait,label:'等发送'});bars.push(mark(proxy,'激活'));}
      rows.push({label:`M${l.current} 状态`,bars,gate:proxy?.start,note:proxy?`GPU ${fmt(gpu.end)} 算完；激活 ${fmt(proxy.start)} 发往 PP${s.rank+1}`:`末级 GPU ${fmt(gpu.end)} 算完；输出经结果通路回流`});}
    const beats=[{time:l.start,title:'开始本轮',detail:'先检查请求、握手与传输终态；共享队列操作不专属于当前 batch。'}];
    if(prep)beats.push({time:prep.end,title:`准备 M${l.current}`,detail:'完成本地 batch 准备后，仍需满足历史发送及 GPU 执行顺序约束。'});
    if(proxyWork&&proxyWork.start>(prep?.end??proxyWork.start))beats.push({time:proxyWork.start,title:'历史发送依赖解除',detail:`${proxyWork.owner} 的历史发送不等于当前 batch；此处等待至 ${fmt(proxyWork.start)} u。`});
    if(gpu)beats.push({time:gpu.start,title:`GPU 执行 M${l.current}`,detail:'提交后 GPU 按设备依赖执行；CPU 的后续工作可能处理另一份 batch 的结果。'});
    if(recv)beats.push({time:recv.start,title:`接收 M${l.old} 旧结果`,detail:'结果到达后还有 D2H、结果处理和本级 KV 提交；它们不是同一个完成点。'});
    beats.push({time:l.end,title:'本轮结束',detail:`本轮释放：${l.released.map(b=>'M'+b).join('、')||'无'}。已提交的计算或传输仍可能跨越 loop 边界。`});
    beats.sort((a,b)=>a.time-b.time);s.beat=clamp(s.beat,0,beats.length-1);
    return {l,rows,beats,min:Math.min(l.start,...rows.flatMap(r=>r.bars.map(b=>b.start))),max:Math.max(l.end,...rows.flatMap(r=>r.bars.map(b=>b.end)))};
  }
  function renderQuick(){
    const loops=packet.quick.loops.filter(l=>l.r===s.rank);if(!loops.some(l=>l.n===s.loop))s.loop=loops[0].n;
    const view=rowsForLoop();
    controls(button('pp-loop','看一轮',s.mode==='loop')+button('pp-batch','追一批',s.mode==='batch')+(s.mode==='loop'?select('pp-rank','流水级',[[0,'PP0'],[1,'PP1'],[2,'PP2']],s.rank)+select('pp-loop-index','本地轮次',loops.map(l=>[l.n,'L'+(l.n+1)]),s.loop):select('pp-batch-index','观察 batch',packet.quick.batches.map(b=>[b.batch,'M'+b.batch]),s.batch))+(s.mode==='loop'?animationControls('beat',view.beats.length,'pp-story'):''));
    bind('pp-loop','click',()=>change(()=>s.mode='loop'));bind('pp-batch','click',()=>change(()=>s.mode='batch'));
    if(s.mode==='loop'){
      bind('pp-rank','change',e=>change(()=>{s.rank=+e.target.value;s.beat=0;}));bind('pp-loop-index','change',e=>change(()=>{s.loop=+e.target.value;s.beat=0;}));animationBind('beat',view.beats.length,'pp-story');
      const {l,rows,beats,min,max}=view,b=beats[s.beat];
      $('wr-scene').innerHTML=`<h3>PP${s.rank} L${s.loop+1} · ${esc(b.title)} · ${fmt(b.time)} u</h3>`+railChart(rows,min,max,`PP${s.rank} L${s.loop+1} 的控制、计算与交接`,b.time)+ppLegend()+`<div class="wr-change">${esc(name(activeId))} · loop ${fmt(l.start)}–${fmt(l.end)} u · 槽位 s${l.slot} · 释放 ${l.released.map(b=>'M'+b).join('、')||'无'}</div><p class="wr-note">状态轨道表示生命周期，不表示持续占用 GPU。图中小三角表示对应行的关键就绪点。</p>`;
      status(b.detail);
    }else{
      bind('pp-batch-index','change',e=>change(()=>s.batch=+e.target.value));const b=packet.quick.batches.find(b=>b.batch===s.batch),colors=[phaseColor.ready,phaseColor.cpu,phaseColor.gpu,phaseColor.wait,phaseColor.kv,phaseColor.wait],names=['准入','准备','前向','等结果 / 处理','KV 在途','等共识 / 清理'];
      const rows=b.ranks.map(r=>({label:'PP'+r.r,bars:r.phases.map((p,i)=>({...p,fill:colors[i],label:names[i]})),gate:r.phases[4].end,note:`前向 L${r.currentLoop+1} · 结果 L${r.resultLoop+1} · 清理 L${r.releaseLoop+1}；KV 结束 ${fmt(r.phases[4].end)}`}));
      $('wr-scene').innerHTML=`<h3>M${s.batch} · ${esc(name(activeId))}</h3>`+railChart(rows,Math.min(...b.ranks.map(r=>r.phases[0].start)),Math.max(...b.ranks.map(r=>r.phases[5].end)),`M${s.batch} 的三级生命周期`)+ppLegend()+`<p class="wr-note">准入 → 准备 → 前向 → 等结果 / 处理 → KV 在途 → 等共识 / 清理。</p>`;
      status('激活沿 PP 传，KV 发给 Decode；各级 L# 是本地轮次，不是同步屏障。');
    }
  }
  async function chooseCompare(id){const request=++compareRequest;a.stop();root.setAttribute('aria-busy','true');try{const p=await get(id);if(request!==compareRequest)return;compareId=id;comparison=p;a.render();}catch(e){if(request===compareRequest)status('对照场景加载失败，请重新选择。');}finally{if(request===compareRequest)root.removeAttribute('aria-busy');}}
  function dependencyRows(p){
    const g=graph(p),n=g[s.dep];if(!n)return null;const deps=n.deps.map(id=>g[id]).filter(Boolean),prev=g[n.prev]||deps[0],other=deps.filter(d=>d!==prev),rows=[];
    if(prev)rows.push({label:'本地前序',bars:[mark(prev,prev.owner)],note:`PP${prev.r} ${prev.label} · ${fmt(prev.start)}–${fmt(prev.end)}`});
    other.forEach(d=>{const gpu=d.kind==='message'?d.deps.map(id=>g[id]).find(v=>v?.kind==='gpu'):null;rows.push({label:'所需前驱',bars:[...(gpu?[mark(gpu,'计算 '+gpu.owner)]:[]),mark(d,d.owner)],gate:d.end,note:`PP${d.r} ${d.owner} · ${d.label}，${fmt(d.end)} 完成`});});
    rows.push({label:(n.kind==='gpu'?'GPU':'CPU')+' PP'+n.r,bars:[...(prev&&n.start>prev.end?[{start:prev.end,end:n.start,fill:phaseColor.wait,label:'等前驱'}]:[]),mark(n,n.kind==='gpu'?n.owner:'执行')],gate:n.start,note:`${n.label} · ${fmt(n.start)}–${fmt(n.end)}；等待 ${fmt(Math.max(0,n.start-(prev?.end??n.start)))} u`});
    const next=({'0:5:recv_out':'0:5:kv','0:5:proxy_work':'0:5:gpu','0:5:gpu':'0:5:proxy_message'})[s.dep];if(next&&g[next]){const z=g[next];rows.push({label:'后续动作',bars:[mark(z,z.kind==='kv'?'M1 KV':z.kind==='gpu'?'M3 前向':'M3 激活')],note:`${z.label} · ${fmt(z.start)}–${fmt(z.end)}`});}
    return {n,deps,rows};
  }
  const history=[];
  function renderComparison(){
    const first=dependencyRows(original),second=dependencyRows(comparison);if(!first||!second){status('该事件不在两组共同模型中，请选择主要观察事件。');return;}
    const options=[['0:5:recv_out','M1 旧结果'],['0:5:proxy_work','M2 历史发送'],['0:5:gpu','M3 GPU 启动']];if(!options.some(x=>x[0]===s.dep))options.push([s.dep,'当前前驱 · '+second.n.label]);
    controls(select('dependency-event','观察',options,s.dep)+select('dependency-compare','独立对照场景',catalog.map(c=>[c.id,c.name]),compareId)+(history.length?button('dependency-back','← 返回'):''));
    bind('dependency-event','change',e=>change(()=>{s.dep=e.target.value;history.length=0;}));bind('dependency-compare','change',e=>chooseCompare(e.target.value));bind('dependency-back','click',()=>change(()=>s.dep=history.pop()));
    const bars=[first,second].flatMap(v=>v.rows.flatMap(r=>r.bars)),min=Math.min(...bars.map(b=>b.start)),max=Math.max(...bars.map(b=>b.end));
    let html=ppLegend();for(const [label,v] of [['原示例',first],[name(compareId),second]])html+=`<h3 class="wr-compare-title">${esc(label)} · 所选操作 ${fmt(v.n.start)} u 开始</h3>`+railChart(v.rows,min,max,label+'依赖时序');
    html+=`<div class="wr-change">起点变化：${fmt(first.n.start)} → ${fmt(second.n.start)} u（${fmt(second.n.start-first.n.start)} u）。</div><details><summary>继续追溯直接前驱 · ${esc(name(compareId))}</summary><div class="wr-grid">${second.deps.map((n,i)=>button('dependency-'+i,`PP${n.r} · ${esc(n.owner)}<br><small>${esc(n.label)} · ${fmt(n.end)} u</small>`)).join('')||'已到模型依赖起点'}</div></details>`;
    $('wr-scene').innerHTML=html;second.deps.forEach((n,i)=>bind('dependency-'+i,'click',()=>change(()=>{history.push(s.dep);s.dep=n.id;})));
    status('所需前驱都满足后，本操作才能推进；本面板独立对照预设场景，下方完整图继续使用页面顶部的场景。');
  }
  a.start(()=>{evidence('PP=3、depth=0；源码 279339f113。使用原完整依赖模型的数据，单位 u 为教学假设，不是实测或优化加速比。Work.wait 的实际影响依赖通信后端。');if(root.dataset.ppReader==='quick'){if(packet)renderQuick();else status('正在读取当前时间模型…');}else if(original&&comparison)renderComparison();else status('正在读取两组对照模型…');});
  if(root.dataset.ppReader==='quick'){
    window.addEventListener('pp-timing-change',e=>{a.stop();packet=e.detail.packet;activeId=e.detail.id;s.beat=0;a.render();});
    if(window.PP_SCENARIO_STATE){packet=window.PP_SCENARIO_STATE.packet;activeId=window.PP_SCENARIO_STATE.id;a.render();}
    else get('baseline').then(p=>{if(!packet){packet=p;activeId='baseline';a.render();}}).catch(()=>status('时间模型加载失败，请刷新页面重试。'));
  }else{root.setAttribute('aria-busy','true');Promise.all([get('baseline'),get(compareId)]).then(([p,c])=>{original=p;comparison=c;a.render();}).catch(()=>status('对照模型加载失败，请刷新页面重试。')).finally(()=>root.removeAttribute('aria-busy'));}
})();
