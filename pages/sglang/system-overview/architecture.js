/* The architecture stays visible; views focus dependencies, not elapsed time. */
(function(){
'use strict';
const root=document.querySelector('.architecture');if(!root)return;
const views=root.querySelector('.arch-views'), note=root.querySelector('.arch-view-note');views.hidden=false;note.hidden=false;
const notes={all:'完整架构：沿请求与返回闭环阅读，KV 管理与执行在同一调度进程内协作。',input:'请求进入：HTTP 接入 → TokenizerManager → Scheduler。跨进程传递 token ID、生成配置及 rid；进入调度侧还要等待本轮选批。',compute:'一轮计算：Scheduler 结合缓存与空间选批 → Worker / Runner 组织模型计算 → 返回本轮结果。未结束的请求继续调度；KV 管理与模型执行不是两个网络服务。',output:'结果返回：Scheduler 处理本轮结果 → DetokenizerManager 转成文字 → TokenizerManager 找到对应请求 → HTTP 响应。普通生成可能多次经过回程。'};
views.addEventListener('click',e=>{const button=e.target.closest('[data-arch-view]');if(!button)return;const key=button.dataset.archView;for(const b of views.querySelectorAll('button'))b.setAttribute('aria-pressed',String(b===button));for(const el of root.querySelectorAll('[data-phase]'))el.classList.toggle('arch-muted',key!=='all'&&!el.dataset.phase.split(' ').includes(key));note.textContent=notes[key];});
function select(id,scroll=false){const detail=document.getElementById('arch-'+id);if(!detail?.classList.contains('arch-detail'))return;for(const d of root.querySelectorAll('.arch-detail'))d.open=d===detail;for(const n of root.querySelectorAll('[data-arch-node]'))n.classList.toggle('arch-selected',n.dataset.archNode===id);for(const a of root.querySelectorAll('[data-arch-select]'))a.setAttribute('aria-current',a.dataset.archSelect===id?'true':'false');if(scroll)detail.scrollIntoView({block:'nearest',behavior:'instant'});}
root.addEventListener('click',e=>{const a=e.target.closest('[data-arch-node],[data-arch-select]');if(!a)return;e.preventDefault();const id=a.dataset.archNode||a.dataset.archSelect;history.replaceState(null,'','#arch-'+id);select(id,true);});
function restore(){if(location.hash.startsWith('#arch-'))select(location.hash.slice(6),true);}
window.addEventListener('hashchange',restore);restore();
})();
