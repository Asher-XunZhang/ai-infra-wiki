/* Read-only snapshot of the shared clock. This page never starts a player. */
(() => {
  'use strict';
  const host=document.querySelector('[data-pp-reading-snapshot]');if(!host)return;
  const J=window.PP_JOURNEY,M=window.PP_JOURNEY_MODEL,base=new URL('.',document.currentScript.src);
  const params=new URLSearchParams(location.search),s=J.state;
  const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt=v=>Number(v).toFixed(2);
  // The dependency page can compare an independent B; name that model explicitly.
  const independent=s.compare!=='current',id=independent?s.compare:J.scenario;
  const scene=window.PP_SCENARIOS.find(x=>x.id===id);
  let worker;
  window.addEventListener('pagehide',()=>worker?.terminate(),{once:true});
  async function packet(){
    if(id!=='custom'){
      if(!scene)throw Error('未知场景');
      const response=await fetch(new URL(scene.file,base));if(!response.ok)throw Error('场景读取失败');
      return response.json();
    }
    const v=JSON.parse(params.get('timing'));
    if(!Array.isArray(v)||v.length!==6||v[0]!==1||![15,6,15,9,4].every((len,i)=>Array.isArray(v[i+1])&&v[i+1].length===len))throw Error('自定义参数无效');
    const matrix=(a,width)=>[0,1,2].map(r=>a.slice(r*width,(r+1)*width));
    const timing=window.PP_TIMING_ENGINE.validate({gpu:matrix(v[1],5),h2d:matrix(v[2],2),kv:matrix(v[3],5),cpu:Object.fromEntries([...'ABCDEFGHI'].map((p,i)=>[p,v[4][i]])),...Object.fromEntries(['proxy','output','control','d2h'].map((p,i)=>[p,v[5][i]]))});
    return new Promise((resolve,reject)=>{
      worker=new Worker(document.querySelector('#timing-engine').src);
      const timeout=setTimeout(()=>{worker.terminate();reject(Error('自定义场景计算超时'));},10000);
      const finish=(error,value)=>{clearTimeout(timeout);worker.terminate();error?reject(error):resolve(value);};
      worker.onmessage=e=>finish(e.data.error?Error(e.data.error):null,e.data.packet);
      worker.onerror=()=>finish(Error('自定义场景计算失败'));
      worker.postMessage({id:1,template:window.PP_TIMING_TEMPLATE,timing});
    });
  }
  packet().then(p=>{
    const model=M.compile(p),time=s.time;
    const observed=model.graph[s.inspect]||model.graph[model.events.find(e=>e.id===s.inspect)?.target]||model.graph[s.event];
    const d=observed&&M.dependencies(model,observed.id,time);
    const rows=[0,1,2].map(rank=>{
      const v=M.rankState(model,s.batch,rank,time),cpu=v.cpu;
      const work=model.nodes.filter(n=>n.r===rank&&['h2d','kv','copy','message'].includes(n.kind)&&M.active(n,time));
      const names={h2d:'KV 回载',kv:'本级 KV → Decode',copy:'结果就绪事件',message:'消息在途'};
      return `<tr data-snapshot-rank="${rank}"><th scope="row">PP${rank}</th><td>${cpu?`L${cpu.n+1} · ${esc(cpu.label)}<small>${esc(cpu.owner)}</small>`:'无图中 CPU 活动'}</td><td>${v.gpu?`${esc(v.gpu.owner)} 前向计算`:'无图中前向执行'}</td><td>${work.length?work.map(n=>`${esc(n.owner)} · ${names[n.kind]}${n.kind==='message'?' · '+esc(n.label):''}`).join('<br>'):'无图中异步活动'}</td></tr>`;
    }).join('');
    host.dataset.snapshotTime=time.toFixed(3);host.dataset.snapshotScenario=id;
    host.innerHTML=`<details class="pp-reading-snapshot" open><summary>同一时刻的三个 Stage · ${fmt(time)} u</summary><p>${independent?'独立对照 B':'当前场景'}：${esc(scene?.name||'自定义时间')}。这是进入源码页时的只读快照，时间保持不动；L 是各级本地迭代。${independent?'返回依赖页继续观察 B；执行旅程仍对应原页面场景。':''}</p><div class="table-scroll" tabindex="0" role="region" aria-label="三个 Stage 的同时刻状态"><table><thead><tr><th>Stage</th><th>CPU 正在做什么</th><th>GPU</th><th>异步活动</th></tr></thead><tbody>${rows}</tbody></table></div>${observed?`<p data-reading-operation><b>所选操作：</b>PP${observed.r} · L${observed.n+1} · ${esc(observed.label)}（${fmt(observed.start)}–${fmt(observed.end)} u）。${time>=observed.end?'已完成':time>=observed.start?'执行中':d.ready?'前置条件已齐，尚未开始':'仍待 '+d.deps.filter(p=>!p.ready).map(p=>`PP${p.r} · ${esc(p.label)}`).join('；')}。</p>`:''}<p><a href="quick.html#quick-timeline">返回执行旅程，继续此时刻 →</a> · <a href="index.html#dependency-comparison">返回等待与依赖 →</a> · <a href="#section-3">对照源码顺序表 ↓</a></p></details>`;
    J.links();
  }).catch(()=>{
    host.innerHTML='<p role="status">本次未能读取对应场景，未用其他场景替代。观察参数已保留，可返回动画重试；下方固定版本源码仍可阅读。</p>';
  });
})();
