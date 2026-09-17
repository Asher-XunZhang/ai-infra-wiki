/* Spatial teaching view. Instance boundaries are service boundaries, not hosts. */
window.renderDeploymentScene = function(s, options) {
  const {split,step,ready,blocked}=s;
  const {gpus,kv,metadata}=options;
  const active=blocked?3:step;
  const stage=(role,label,isActive,done)=>`<div class="execution-stage ${isActive?'is-current':''} ${done?'is-complete':''}" data-role="${role}"><b>${role}</b><span>${label}</span><small>${isActive?'正在处理':done?'本步已完成':'等待轮到'}</small></div>`;
  const prompt='<span class="kv-piece">p1<small>K · V</small></span><span class="kv-piece">p2<small>K · V</small></span><span class="kv-piece">p3<small>K · V</small></span><span class="kv-piece">p4<small>K · V</small></span>';
  const gpuGroup=()=>`<div class="device-group" aria-label="本实例使用 ${gpus} 张 GPU">${Array.from({length:gpus},(_,i)=>`<span class="gpu-chip"><span aria-hidden="true">▦</span> GPU ${i}</span>`).join('')}</div>`;
  const weights='<div class="model-residency"><strong>已加载完整模型 · 层 1 → 层 L</strong><span class="model-layers" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i></span><small>由本实例的 GPU 共同执行</small></div>';
  function instance(id,label,role) {
    const isD=role==='D',unified=role==='PD';
    const hasKV=isD?step>=3&&kv:step>=1;
    const decoding=step===4&&ready&&(isD||unified);
    const hasOutput=isD?step>=3&&kv&&metadata:step>=1;
    const isActive=unified?step>0:isD?step>=3:step===1;
    const status=unified?['等待请求','正在 Prefill','保留本地 KV','等待下一轮调度','正在 Decode'][active]:isD?(blocked?'等待交接条件':step===4?'正在 Decode':step>=3?'检查可执行条件':'等待接力'):(step===1?'正在 Prefill':step>=2?'本次 Prefill 已完成':'等待请求');
    return `<section class="service-instance ${isActive?'instance-active':''}" data-instance="${id}" aria-label="${label}">
      <header class="instance-heading"><span class="instance-symbol" aria-hidden="true">▤</span><div><h3>${label}</h3><p>${unified?'同一实例承担 P 和 D':isD?'独立的 Decode 服务':'独立的 Prefill 服务'}</p></div><span class="instance-state">${status}</span></header>
      ${gpuGroup()}${weights}
      <div class="instance-execution ${unified?'shared-execution':''}">
        ${!isD?stage('P','处理 prompt',step===1,step>1):''}
        ${unified?'<span class="inside-arrow" aria-label="同一实例内切换阶段">→</span>':''}
        ${isD||unified?stage('D','逐 token 续写',decoding,false):''}
      </div>
      <div class="kv-store ${hasKV?'has-kv':''}" data-kv-owner="${id}"><div class="store-heading"><b>▰ ${unified?'同一份本地':'本实例的'} KV</b><small>${hasKV?'已写入 / 可用':isD&&step===2?'接收中 · 尚未确认可用':'空 · 尚无本请求状态'}</small></div><div class="kv-pieces">${hasKV?prompt:'<span class="empty-kv">· · · ·</span>'}${decoding?'<span class="kv-piece latest">y1<small>K · V</small></span>':''}</div><small>${unified?'P 写入 ↓ 保留在这里 ↑ D 读取':isD?'接收 P 的历史 KV，D 再继续追加':'P 写入的历史 KV，用于向 D 交接'}</small></div>
      <div class="instance-result">${hasOutput?`<span class="token new">${decoding?'y2':'y1'}</span> ${decoding?'D 刚采样的输出':isD?'从 P 接过的首 token':'P 采样的首 token'}`:'<span class="token pending">输出待产生</span>'}</div>
      ${isD?`<div class="instance-admission ${blocked?'is-waiting':''}" data-admission="${blocked?'waiting':step>=3?'ready':'pending'}">${step<3?'○ 接收后检查可执行条件':blocked?'⏸ 检查可执行条件 · 仍需等待':'✓ 交接条件已满足'}<small>${blocked?'Decode 尚未执行':step===4?'输入 y1 → 读取 KV → 采样 y2':'请求还需要经过本地调度'}</small></div>`:''}
    </section>`;
  }
  const bridge=split?`<div class="instance-bridge ${step===2?'is-transferring':''}" aria-label="跨实例交接">
      <strong>跨实例交接</strong><span class="transfer-track" aria-hidden="true"><span class="transfer-packet">KV</span></span>
      <span class="bridge-direction">P → D</span><div class="bridge-payload"><b>传递</b><span>各层 KV</span><span>首 token y1</span><span>请求元数据</span></div><small>${step<2?'等待 P 产出状态':step===2?'正在交接 →':blocked?'D 侧仍有条件未满足':'D 侧接力'}</small>
    </div>`:'';
  return `<div class="deployment-summary"><strong>${split?'2 个独立服务实例':'1 个服务实例'}</strong><span>${split?'P 与 D 分属两个实例，通过交接接力执行':'P 和 D 在同一边界内，共用模型与本地 KV'}</span></div>
    <div class="request-entry ${step===0?'is-current':''}"><span>用户请求 R</span><span aria-hidden="true">↓</span><span>${split?'路由协调 P 实例与 D 实例':'路由到实例 A'}</span></div>
    <div class="deployment-topology ${split?'is-split':'is-unified'}">${instance(split?'P':'A',split?'实例 P':'实例 A',split?'P':'PD')}${bridge}${split?instance('D','实例 D','D'):''}</div>
    <p class="topology-caption">粗线外框 = 一个服务实例；芯片 = 该实例使用的 GPU。${split?'两侧分别加载模型，箭头只搬请求状态。实例分离不要求一定在两台物理服务器上。':'把一个实例改成多张 GPU，P/D 仍在同一个服务边界内。'} GPU 数量仅用于理解边界，不表示等成本性能比较。</p>`;
};
