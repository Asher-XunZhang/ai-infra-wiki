(()=>{'use strict';const M=window.ExecutionModel,$=s=>document.querySelector(s),lab=$('[data-execution-lab]');if(!M||!lab)return;
let mode='chain',step=0,timer=null;const scene=$('#exec-scene'),media=matchMedia('(prefers-reduced-motion: reduce)'),val=id=>$(id).value;
const sourceUrl=key=>{const [file,line]=M.sources[key];return `https://github.com/sgl-project/sglang/blob/${M.revision}/${file}#L${line}`;};
const chips=(xs,cls='')=>`<div class="exec-chips">${xs.map(x=>`<span class="exec-chip ${x===null?'pad':cls}">${x===null?'补齐':x}</span>`).join('')}</div>`;
const wire=()=>`<svg class="exec-wire" viewBox="0 0 40 110" aria-hidden="true"><path d="M20 4V104m-5-8 5 8 5-8"/><circle cx="20" cy="${media.matches?54:0}" r="4">${media.matches?'':'<animateMotion dur="1.4s" repeatCount="indefinite" path="M0 8V93"/>'}</circle></svg>`;
function last(){return mode==='chain'?7:mode==='graph'?4:3;}
function stop(){clearInterval(timer);timer=null;$('#exec-play').textContent='播放';$('#exec-play').setAttribute('aria-pressed','false');}
function payload(type){
 if(type==='requests')return chips(['R1','R2'])+'<small>各 1 个新位置 · Decode</small>';
 if(type==='forward')return '<div class="exec-rows"><div class="exec-row"><b>R1</b>'+chips(['41'])+'<span>p3 → 槽 11</span></div><div class="exec-row"><b>R2</b>'+chips(['52'],'blue')+'<span>p2 → 槽 13</span></div></div><small>token ID ≠ 位置 ≠ KV 槽位</small>';
 if(type==='indices')return '<div class="exec-rows"><div class="exec-row"><b>R1</b><span class="exec-index">2 · 7 · 9 · 11</span></div><div class="exec-row"><b>R2</b><span class="exec-index">4 · 8 · 13</span></div></div><small>分界 [0, 4, 7] · 先备地址，后写本轮值</small>';
 if(type==='hidden')return chips(['41','52'])+'<small>进入模型后：Embedding → hidden states</small>';
 if(type==='qkv')return chips(['Q','K','V'])+'<small>每层：新 K/V → 槽 11、13</small>';
 if(type==='kernel')return '<div class="exec-route"><span>Q + KV 索引</span><strong>→</strong><span>读取 K/V</span></div>'+chips(['Q','K buffer','V buffer'])+'<small>算子随后写入 Attention 输出</small>';
 if(type==='logits')return '<div class="exec-logits">'+[['a',1],['b',4],['c',2]].map(([name,n])=>`<div class="exec-score ${n===4?'best':''}"><span>${name}</span><i style="width:${n*25}%"></i><span>${n}</span></div>`).join('')+'</div><small>只展示 R1 的 3 个候选分数 · 教学数值</small>';
 return chips(['R1 → b','R2 → a'])+'<small>贪心示例 · R2 的分数省略</small>';
}
function render(){let html='',caption='',why='',source='worker';
 if(mode==='chain'){
 const f=M.trace[step];html=`<h3>${f.action}</h3><div class="exec-focus"><div class="exec-actor">${f.from}</div><div class="exec-handoff">${wire()}<div class="exec-payload">${payload(f.data)}</div></div><div class="exec-actor destination">${f.to}</div></div><p class="exec-note">当前交接 / 共 8 步 · 箭头表示调用或数据依赖</p>`;caption=f.caption;source=f.source;
 why='普通单 rank、无 overlap、无投机、非 prefill-only 的 Decode 主线，选 Triton 后端讲解。Worker、Runner、模型和 backend 是职责/对象，不是四个网络服务。元数据按本轮条件准备；Attention kernel 返回中间张量，模型仍需完成其余计算。图仅画当前交接，R1/R2、槽位和分数为教学值；不声称执行了实际 GPU 运算。';
 }else if(mode==='graph'){
 const d=M.graph(Number(val('#graph-count')),val('#graph-condition'),step),ready=step>=1;
 html=`<h3>本轮 ${d.real.length} 条真实请求</h3>${chips(d.real)}<p class="exec-gate">${ready?d.reason:'先由 Scheduler 选定本轮请求'}</p><div class="exec-branch"><div class="${ready&&d.eligible?'chosen':''}">CUDA Graph<small>复用已捕获的执行</small></div><div class="${ready&&!d.eligible?'chosen':''}">Eager<small>按本轮路径执行</small></div></div><div class="exec-buffer ${step<2?'exec-muted':''}"><small>${step<2?'等待路径选择':d.eligible?'静态输入 / 图桶 '+d.bucket:'本例真实输入 · 无图桶补齐'}</small>${chips(d.slots)}${step===3?'<p class="exec-equation">'+(d.eligible?'Replay':'Eager forward')+' → 模型计算 → logits</p>':''}</div><div class="exec-output ${step<4?'exec-muted':''}"><small>${step===4?(d.eligible?'裁回真实请求的输出行':'返回真实请求的输出行'):'等待模型输出'}</small>${chips(step===4?d.outputs:[])}</div>`;
 caption=['图重放不会扩充请求集合；先固定本轮真实成员。',d.eligible?'本例已捕获 1、2、4 三种容量，允许补齐，其余条件满足。':'本轮不满足图路径条件，转入普通执行；请求并没有被丢弃。',d.eligible&&d.bucket>d.real.length?'为匹配图桶补齐输入。虚线格不是新请求，不能生成对外结果。':'把真实输入交给所选执行路径。',d.eligible?'重放已捕获的设备工作；当前输入、KV 内容与必要元数据仍要更新。':'Eager 仍调用模型和算子；没有重放不等于不在 GPU 上执行。','返回真实请求对应的 logits 行，然后由后续采样与调度接手。'][step];
 source=step<2?'graphGate':step===2?'graphLoad':step===3?d.eligible?'replay':'runner':'trim';
 why='普通 Decode、每请求一个输入位置；图桶 [1, 2, 4] 是教学配置。固定代码还检查动态 embedding override、spec width、并行、encoder、TBO 等条件，页面不是完整资格判定器。示例中的 override 表示有效的动态 embedding 覆盖输入触发图资格拒绝，不模拟覆盖向量本身。Eager 也可能使用可复用缓冲区；这里的“无图桶补齐”只指单 rank 示例。采样与 PP 非末 stage、投机验证、Prefill 图另有路径。';
 }else{
 const m=Number(val('#matrix-rows')),upgrade=val('#hardware-upgrade'),d=M.matrix(m,upgrade),base=M.matrix(m),max=Math.max(base.mathMs,base.memoryMs);
 const grid=(rows,weights=false)=>`<div class="exec-grid ${weights?'weights':''}">${Array.from({length:weights?16:Math.min(rows,4)*4},()=>'<i></i>').join('')}</div>`;
 const bar=(name,time,original)=>`<div class="exec-bar-row"><div><span>${name}</span><span>${time.toFixed(3)} ms</span></div><div class="exec-bar-track"><b style="width:${original/max*100}%"></b><i style="width:${time/max*100}%"></i></div></div>`;
 html=`<h3>一层线性变换：X × W → Y</h3><div class="exec-schematic"><div class="exec-matrix">${grid(m)}<small>X · ${m} 个新位置</small></div><b>×</b><div class="exec-matrix">${grid(4,true)}<small>同一份 W</small></div><b>→</b><div class="exec-matrix ${step<1?'exec-muted':''}">${grid(m)}<small>Y · ${m} 行结果</small></div></div><p class="exec-note">色条示意行数分组 · 不是完整矩阵尺寸</p><div class="exec-bars ${step<2?'exec-muted':''}">${bar('计算下界 · FLOPs / 算力',d.mathMs,base.mathMs)}${bar('访存下界 · bytes / 带宽',d.memoryMs,base.memoryMs)}</div><p class="exec-note">虚线为原配 · 两条使用相同时间比例</p><p class="exec-equation ${step<3?'exec-muted':''}">理想下界 ${d.lowerMs.toFixed(3)} ms · ${d.limit==='memory'?'访存项更长':'计算项更长'}</p>`;
 caption=['多行输入复用同一份权重。M 是本轮新位置总数，可以来自多个请求，也可以来自一段 Prefill。','新增位置越多，结果行与乘加工作越多；权重矩阵的大小不随 M 增长。','分别计算两项理想成本，再比较较长的一项。切换算力或带宽，看哪条真正缩短。',d.limit==='memory'?'本例访存项更长：只提高算力不会改变这个理想下界。真实耗时还可能受启动、并行度和重复读取限制。':'本例计算项更长：只提高带宽不会改变这个理想下界。这个单线性层例子不代表完整 Prefill/Decode 的性能。'][step];
 why='整理者构造的算术模型：K=N=4096，输入/权重/输出均按 2 bytes 计，每个矩阵元素理想读或写一次；FLOPs=2MKN，bytes=2(MK+KN+MN)。假定算力 8 TFLOP/s、带宽 100 GB/s，均非实测设备参数。取 max(计算下界,访存下界)，不声称两者在一个请求上必然完全重叠；忽略启动、缓存重读、精度、Attention KV 与通信。M=1 时尤其不能据此预测实际利用率。';source=null;
 }
 scene.innerHTML=html;scene.dataset.step=step;scene.dataset.mode=mode;$('#exec-caption').textContent=caption;$('#exec-why').textContent=why;$('#exec-source').href=source?sourceUrl(source):'https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#understanding-performance';$('#exec-source').textContent=source?'查看固定源码 ↗':'NVIDIA：计算与访存边界 ↗';$('#exec-status').textContent=`${step+1} / ${last()+1}`;$('#exec-previous').disabled=step===0;$('#exec-next').disabled=step===last();if(step===last())stop();
}
function reset(){stop();step=0;render();}
lab.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.mode;lab.querySelectorAll('[data-mode]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));lab.querySelectorAll('[data-settings]').forEach(x=>x.hidden=x.dataset.settings!==mode);$('.parallel-details').open=false;reset();}));lab.querySelectorAll('select').forEach(s=>s.addEventListener('change',reset));$('#exec-next').addEventListener('click',()=>{stop();step=Math.min(last(),step+1);render();});$('#exec-previous').addEventListener('click',()=>{stop();step=Math.max(0,step-1);render();});$('#exec-reset').addEventListener('click',reset);$('#exec-play').addEventListener('click',()=>{if(timer)return stop();if(step===last())step=0;$('#exec-play').textContent='暂停';$('#exec-play').setAttribute('aria-pressed','true');render();timer=setInterval(()=>{step++;render();},2400);});document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});media.addEventListener('change',render);lab.hidden=false;render();
})();
