/* One teaching entry per mechanism. Models stay reusable; duplicate players do not mount. */
(function(root){
'use strict';
const foundation={tokens:[],generation:[],attention:[],cache:[2,3],batching:[],hardware:[],parallel:[],communication:[],deployment:[],speculation:[],serving:[0,1],metrics:[]};
const extraTitles={cache:'继续实验：物理页分配与共享前缀',communication:'继续实验：通知到达与载荷可用',serving:'继续实验：正常响应与双实例路由',
 'model-execution':'继续实验：CUDA Graph 资格与计算／访存成本','kv-memory':'继续实验：整页命中与共享保护',scheduling:'继续实验：Retract 容量回撤与恢复',communicationEngine:'继续实验：IPC、Collective 与就绪条件','advanced-generation':'继续实验：递推状态与语法约束','serving-operations':'继续实验：健康检查、路由与退出'};
const removedModes={'model-execution':['chain'],'kv-memory':['extend'],scheduling:['admission','chunk'],'advanced-generation':['speculative'],communication:['kv'],'serving-operations':['cancel']};
function prepare(lab,key,initial){
 const removed=removedModes[key]||[];
 for(const mode of removed)lab.querySelectorAll(`[data-mode="${mode}"],[data-settings="${mode}"]`).forEach(e=>e.remove());
 const mode=removed.includes(initial)?lab.querySelector('[data-mode]').dataset.mode:initial;
 lab.querySelectorAll('[data-mode]').forEach(e=>e.setAttribute('aria-pressed',String(e.dataset.mode===mode)));
 lab.querySelectorAll('[data-settings]').forEach(e=>e.hidden=e.dataset.settings!==mode);
 return mode;
}
const api={foundation,extraTitles,removedModes,prepare,originalOnly:['parallelism','performance','system-overview']};
if(typeof module==='object'&&module.exports)module.exports=api;root.LessonIntegration=api;
})(typeof globalThis==='object'?globalThis:this);
