(function(root){
'use strict';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const f=n=>Number.isFinite(n)?Number(n.toFixed(4)).toLocaleString('en-US'):'—';
const pct=n=>f(n*100)+'%';
const table=(headers,rows)=>'<div class="plab-table"><table><thead><tr>'+headers.map(h=>'<th scope="col">'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map((v,i)=>(i===0?'<th scope="row">':'<td>')+v+(i===0?'</th>':'</td>')).join('')+'</tr>').join('')+'</tbody></table></div>';
const facts=rows=>'<dl class="plab-facts">'+rows.map(([k,v])=>'<div><dt>'+esc(k)+'</dt><dd>'+esc(v)+'</dd></div>').join('')+'</dl>';
const text=(x,y,s,cls='')=>'<text x="'+x+'" y="'+y+'" class="'+cls+'">'+esc(s)+'</text>';
const rect=(x,y,w,h,color='var(--soft)',extra='')=>'<rect x="'+x+'" y="'+y+'" width="'+Math.max(0,w)+'" height="'+h+'" rx="5" fill="'+color+'" '+extra+'/>';
const colors={compute:'var(--accent)',transfer:'var(--p-blue)',control:'var(--p-gold)',wait:'var(--muted)',error:'var(--p-red)'};
function tracks(d,t,lanes,top=270){
 const x=n=>205+n/d.duration*750;let s='';
 for(let i=0;i<=4;i++)s+=text(x(d.duration*i/4),top-15,f(d.duration*i/4)+' u','p-small');
 lanes.forEach(([id,label],i)=>{const y=top+i*47;s+=text(18,y+19,label,'p-small')+rect(205,y,750,29,'var(--soft)');
 for(const e of d.events.filter(e=>e.lane===id)){
  const color=colors[e.kind]||colors.compute,w=x(e.end)-x(e.start),active=t>=e.start&&t<e.end;
  s+='<g data-priority-event="'+esc(e.id)+'"><title>'+esc(e.title+' ['+e.start+', '+e.end+'] u')+'</title>'+rect(x(e.start),y,w,29,'none','stroke="'+color+'" '+(e.kind==='wait'?'stroke-dasharray="4 3"':''))+rect(x(e.start),y+23,w*Math.max(0,Math.min(1,(t-e.start)/(e.end-e.start))),6,color)+(active?rect(x(e.start),y,w,29,'none','stroke="var(--fg)" stroke-width="2"'):'')+(w>100?text(x(e.start)+5,y+18,e.title.slice(0,9),'p-small'):'')+'</g>';
 }});
 s+='<path d="M'+x(t)+' '+(top-6)+'V'+(top+lanes.length*47)+'" stroke="var(--p-red)" stroke-width="2"/>';
 return s;
}
function wire(id,points,event,t,reduced,label,kind='transfer'){
 const active=event&&t>=event.start&&t<event.end,d=points.map((p,i)=>(i?'L':'M')+p.join(' ')).join(' '),color=kind==='transfer'?'var(--p-blue)':kind==='reference'?'var(--muted)':'var(--p-gold)';
 return '<g data-priority-flow="'+esc(id)+'" data-active="'+!!active+'" data-flow-kind="'+kind+'"><title>'+esc(label)+'</title><path d="'+d+'" fill="none" stroke="'+color+'" stroke-width="'+(active?3:1.5)+'" opacity="'+(active?1:.5)+'" marker-end="url(#p-arrow-'+id+')"'+(active&&!reduced?' stroke-dasharray="8 6" stroke-dashoffset="'+(-(t-event.start)*25).toFixed(3)+'"':'')+'/><defs><marker id="p-arrow-'+id+'" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="'+color+'"/></marker></defs></g>';
}
const svg=(body,h,title)=>'<div class="plab-svg-scroll"><svg viewBox="0 0 1000 '+h+'" role="img" aria-label="'+esc(title)+'">'+body+'</svg></div>';
function cdf(d,t){
 const show=t>=5,phase=t<3?'过滤前概率 p':t<4?'候选权重正在置零':t<5?'保留权重正在归一化':'实际选择概率 q';
 const amount=Math.max(0,Math.min(1,t-3)),renorm=Math.max(0,Math.min(1,t-4));
 const rows=d.ranked.map(r=>{const p=d.probabilities[r.id],kept=r.keepK&&r.keepP;let weight=t<3?p:t<4?p*(kept?1:1-amount):t<5?(kept?p*(1+renorm*(1/d.mass-1)):0):d.q[r.id];return {...r,display:weight};});
 return '<p>条带当前：'+phase+'。完整推导表保留最终值；置零与缩放之间的视觉过渡不作为实际采样分布。</p><div class="plab-cdf">'+rows.filter(r=>r.display>0).map(r=>'<span style="width:'+(r.display*100)+'%;background:color-mix(in srgb,var(--accent) '+(18+r.id*9)+'%,var(--bg))" class="'+(show&&d.selected===r.id?'p-picked':'')+'" title="'+esc(d.words[r.id]+' '+pct(r.display))+'">'+esc(d.words[r.id])+' '+pct(r.display)+'</span>').join('')+(show&&!d.greedy?'<i style="left:'+(d.options.u*100)+'%" title="固定 u='+d.options.u+'"></i>':'')+'</div><p><b>'+(!show?'尚未执行选择。':d.greedy?'贪心：选择 '+esc(d.words[d.selected]):'固定 u='+d.options.u+' 落入 '+esc(d.words[d.selected])+' 的区间')+'</b></p>';
}
function sampling(d,t){
 const names=['模型原始分数','按输出历史施加重复惩罚','温度改变分数间距','top-k 与 top-p 限定候选','用保留质量重新归一化','固定 u 核对选择结果','核对停止与输出边界'],phase=Math.min(6,Math.floor(t));
 const overview=table(['候选 token','温度后概率 p','被保留？','实际选择概率 q'],d.ranked.map(r=>[esc(d.words[r.id]),pct(d.probabilities[r.id]),r.keepK&&r.keepP?'保留':r.keepK?'top-p 删除':'top-k 删除',pct(d.q[r.id])]))+cdf(d,t);
 const detail=table(['候选','原分数 z','生成历史次数','penalty 后','温度后','p','前缀累计','top-k','top-p','q'],d.ranked.map(r=>[esc(d.words[r.id]),f(d.logits[r.id]),String(d.history[r.id]),f(d.penalized[r.id]),f(d.scaled[r.id]),pct(d.probabilities[r.id]),pct(r.before),r.keepK?'✓':'×',r.keepP?'✓':'×',pct(d.q[r.id])]))+cdf(d,t);
 return {overview,detail,title:names[phase],description:d.greedy?'T=0 在本例走贪心分支，不计算 z/0；本轮其他候选全部被排除。':'本例先按符号处理重复分数，再除以温度并归一化；top-k/top-p 使用同一张排序前概率表，过滤之后再计算 q。',rows:[['当前讲解段',phase+1+' / 7'],['保留质量',f(d.mass)],['同分顺序','教学例子按候选 ID；实际后端另核对'],['top-p 边界','固定 PyTorch 分支删除 before > top_p；等号保留'],['输出 logprob','不能把过滤前的 p 当作过滤后的 q']],source:phase<2?0:1};
}
function stopping(d,t){
 const current=d.events.filter(e=>t>=e.start&&t<e.end),last=current.at(-1)||d.events.filter(e=>e.end<=t).at(-1)||d.events[0];
 const rows=d.events.map(e=>[e.title,t>=e.end?'已核对':t>=e.start?'当前':'尚未到达',e.detail]);
 const cancelTrack=d.options.scenario==='cancel'?svg(tracks(d,t,[['device','设备工作 / 收尾'],['client','取消通知']],70),190,'通知返回与设备收尾共用时钟'):'';
 return {overview:cancelTrack+table(['检查点','当前进度','含义'],rows),detail:svg(tracks(d,t,[['rank0','设备 / 后台'],['client','客户端通知']],100),250,'停止与收尾的独立条件')+table(['观察区间','内容'],rows.map(r=>[r[0],r[2]])),title:last.title,description:last.detail,rows:[['场景',d.options.scenario],['时钟用途','给定事件的教学顺序，不是实测'],['文本分片','与输出 token 数分别计数'],['资源退役','仍需设备依赖与服务端证据']],source:d.options.scenario==='cancel'?2:0};
}
function quantization(d,t){
 let s=text(24,30,'同一组原值 → 有限整数刻度 → 还原数值 → 误差','p-strong');
 const x=v=>180+(v+2)/4*760;
 for(let i=-2;i<=2;i++)s+='<path d="M'+x(i)+' 55V245" stroke="var(--border)"/>'+text(x(i)-6,260,String(i),'p-small');
 d.rows.forEach((r,i)=>{const y=72+i*35;s+=text(16,y+4,'x'+i+' / g'+r.group,'p-small')+'<path d="M'+x(r.x)+' '+y+'H'+x(r.restored)+'" stroke="var(--p-red)" stroke-width="3"/><circle cx="'+x(r.x)+'" cy="'+y+'" r="5" fill="var(--p-blue)"/><path d="M'+(x(r.restored)-5)+' '+(y-5)+'l10 10m-10 0l10 -10" stroke="var(--accent)" stroke-width="2"/>';});
 s+=text(24,300,'圆点为原值，叉号为还原值；连线长度是数值误差，不是执行时间。','p-small');
 const factsHtml=facts([['原 BF16 载荷',d.baseline+' bytes'],['整数载荷',d.payload+' bytes'],['scale 元数据',d.metadata+' bytes'],['本例合计',d.total+' bytes'],['最大绝对误差',f(d.maxError)],['被截断值',d.clipped+' 个']]);
 return {overview:svg(s,325,'均匀整数刻度与还原误差')+factsHtml,detail:table(['原值','分组','scale','整数 q','还原 q×scale','误差','截断'],d.rows.map(r=>[f(r.x),String(r.group),f(r.scale),String(r.q),f(r.restored),f(r.error),r.clipped?'是':'否']))+factsHtml,title:['分清量化对象','确定刻度与 scale','舍入和截断','连同元数据核算空间','回到 kernel 与质量验证'][Math.min(4,Math.floor(t))],description:'这是对称均匀整数教学例子，不把它当作 FP8/FP4 的位布局或实际 kernel。'+(d.total>d.baseline?'当前小数组的 scale 开销已超过节省的载荷。':'更少位宽缩小载荷；执行路径和模型质量仍要分别验证。'),rows:[['量化对象',d.options.object],['符号范围','−'+d.qmax+'…'+d.qmax],['公式','q=clip(round(x/s)); x̂=q×s'],['容量边界','未计打包对齐、临时激活、KV 或额外 buffers'],['运行验收','NOT_EXECUTED：未运行 kernel、模型质量或性能实验']],source:t<2?0:1};
}
function quantizationPath(d,t){
 const stage=Math.min(5,Math.floor(t)),steps=['读取实际配置','匹配模型层与量化方法','核对设备、shape 与 backend 条件','创建并加载编码与 scale','后处理布局与本轮激活','执行前核对路径并准备质量实验'];
 return {overview:table(['检查点','当前选择 / 需要的证据'],[['格式',esc(d.name)],['checkpoint / 池对象',esc(d.stored)],['实际方法',esc(d.method)],['执行含义',esc(d.compute)],['前提',d.allowed?'本例假定静态条件成立；仍未运行':'本例缺少匹配 kernel，不能把配置可选当作可执行']]),detail:table(['顺序','准备动作','本例状态'],steps.map((x,i)=>[String(i+1),x,!d.allowed&&i>=3?'被兼容性检查阻塞':t>=i+1?'已讲解':t>=i?'当前':'尚未讲解'])),title:!d.allowed&&stage>=2?'兼容性前提未齐，后续执行被阻塞':steps[stage],description:'参数名、权重编码、scale 布局与实际 kernel 形成一份接口契约。这个播放器只推进阅读检查点，没有运行设备探测、转换、kernel 或质量测试。',rows:[['实际方法',d.method],['表示',d.stored],['执行前提',d.allowed?'教学假设成立':'教学假设不成立'],['质量证据','NOT_EXECUTED：同输入与配置下的业务正确性、指标与误差均需另测']],source:d.options.format==='fp8'?0:d.options.format==='awq'?1:2};
}
function moe(d,t,ctx){
 const by=id=>d.events.find(e=>e.id===id),values=table(['token 行 / 来源','逻辑专家','目标 rank','专家输出','加权合并'],d.rows.map(r=>[r.id+' / Rank '+r.origin,r.experts.map(e=>'E'+e).join('、'),d.jobs.filter(j=>j.token===r.id).map(j=>'E'+j.expert+'→Rank '+j.rank).join('、'),d.jobs.filter(j=>j.token===r.id).map(j=>f(j.result)+'×'+j.weight).join(' + '),t>=d.join+1?f(d.outputs.find(x=>x.id===r.id).value):'等待全部分支']));
 let s=text(24,27,'4 个 token 行 × top-2 = 8 份专家任务；逻辑专家与物理副本分开','p-strong');
 s+=rect(40,65,240,135)+text(58,94,'来源 token 与原行位置','p-strong')+text(58,127,'R1：T0、T1；R2：T2、T3','p-small')+text(58,160,'Dispatch 后仍携带 token 身份','p-small');
 [0,1].forEach(rank=>{const y=60+rank*96;s+=rect(550,y,400,80)+text(568,y+25,'Rank '+rank+'：'+d.loads[rank]+' 份任务','p-strong')+text(568,y+54,rank===0?'逻辑 E0 / E1':d.replica?'E2 / E3 + E0 冗余副本':'逻辑 E2 / E3','p-small');if(d.jobs.some(j=>j.rank===rank&&j.origin!==rank))s+=wire('moe-dispatch-'+rank,[[280,100+rank*63],[420,100+rank*63],[420,y+40],[550,y+40]],by('dispatch'),t,ctx.reduced,'远程行参与跨 rank 分发；同 rank 只做本地重排');});
 s+=rect(40,245,240,55)+text(56,278,t>=d.join+1?'原行输出：'+d.outputs.map(r=>f(r.value)).join(' / '):'Combine：等待各分支按原行返回','p-small');
 [0,1].forEach(rank=>{const y=60+rank*96,e=by('combine'),remote=d.jobs.some(j=>j.rank===rank&&j.origin!==rank);if(remote)s+=wire('moe-return-'+rank,[[550,y+65],[485,y+65],[485,272],[280,272]],e,t,ctx.reduced,'专家结果回到来源行，随后按权重合并');});
 s+=text(300,80,'token 行 + 原行 ID','p-small')+text(300,300,'专家结果回到来源行','p-small');
 s+=tracks(d,t,[['gates','来源 GPU：Gate / 合并'],['network','跨 rank 载荷'],['gpu0','GPU Rank 0'],['gpu1','GPU Rank 1']],355);
 return {overview:values+facts([['逻辑专家任务数',d.logical.join(' / ')],['rank 工作量',d.loads.join(' / ')],['跨 rank 任务',d.remote+' / 8'],['权重副本',d.replica?'E0 额外保留一份；逻辑 E0 未变':'没有额外副本']]),detail:svg(s,575,'MoE 分发、并行专家执行和按原行合并'),title:t<1?'先决定逻辑专家与混合权重':t<3?'分发的是 token 行和对应关系':t<by('combine').start?'各 rank 按自己的任务推进；先完成者等待':t<d.join?'Combine：专家结果正在返回原行':t<d.join+1?'同一 token 的结果按混合权重相加':'原行输出已核对，继续检查本轮收尾',description:'本例每份任务耗时 1 u、每个 rank 串行计算，用于揭示热点与依赖；副本不增加 top-k，也不改变同一逻辑专家的计算结果。实际执行后端与流重叠另查固定源码。',rows:[['副本代价','额外专家权重与迁移成本'],['合并放行',t>=d.join?'相关 rank 的结果已返回':'仍有分支未返回'],['加权规则','y(T)=Σ w(T,E)·fE(xT)'],['引用 / 搬运','本地索引重排不冒充跨 rank 传输']],source:t<3?0:1};
}
function multimodal(d,t,ctx){
 const by=id=>d.events.find(e=>e.id===id);let s=text(24,28,'媒体位置展开、编码特征与 LM KV：先确认对象，再检查放行','p-strong');
 const nodes=[[35,220,'媒体与预处理','本例 2×2 图块 / 身份'],[300,300,'E：编码与特征适配','本例特征 4×4，尚未是 LM KV'],[655,310,d.pd?'P：语言模型 Prefill':'PD：语言模型处理与生成','计划 6 个 LM 输入位置']];
 nodes.forEach(([x,w,a,b])=>{s+=rect(x,65,w,135)+text(x+14,94,a,'p-strong')+text(x+14,124,b,'p-small');});
 for(let r=0;r<2;r++)for(let c=0;c<2;c++)s+=rect(52+c*27,143+r*23,21,18,'color-mix(in srgb,var(--p-gold) 30%,var(--bg))');
 for(let r=0;r<4;r++)for(let c=0;c<4;c++)s+=rect(320+c*18,140+r*13,13,10,t>=by(d.featureHit?'fetch':'encode').end?'var(--accent)':'var(--border)');
 for(let i=0;i<6;i++)s+=rect(674+i*44,151,38,27,t>=d.pEnd?'color-mix(in srgb,var(--accent) 30%,var(--bg))':'var(--soft)','stroke="var(--border)"')+text(680+i*44,170,'p'+i,'p-small');
 s+=wire('mm-input',[[255,165],[300,165]],null,t,ctx.reduced,'预处理输入的概念交接，不指定复制协议','reference');
 s+=wire('mm-feature',[[600,140],[655,140]],d.separate?by('feature'):null,t,ctx.reduced,d.separate?'跨实例编码特征 + 位置元数据':'本地特征引用；不是网络传输',d.separate?'transfer':'reference');
 if(d.pd){s+=rect(655,245,310,70)+text(674,274,'D：消费可读 KV 后接续','p-strong')+text(674,301,t>=d.dStart?'本例接收条件已满足':'仍等待 P 与 KV 交接','p-small');s+=wire('mm-kv',[[810,200],[810,245]],by('kv'),t,ctx.reduced,'LM KV 与元数据；不是图像特征');s+=text(824,232,'LM KV','p-small');}
 s+=text(35,239,'占位展开示例：文本 p0 | f0 f1 f2 f3 | 文本 p5','p-small');
 s+=text(35,338,d.separate?'E 与 P 跨实例：传编码特征及元数据。':'E 与 PD 同实例：概念分工不等于多个进程。','p-small');
 s+=text(35,362,d.pd?'P 与 D 跨实例：另传语言模型 KV；就绪条件独立。':'P/D 在本例共实例：不画 KV 网络传输。','p-small');
 s+=tracks(d,t,[['cpu','预处理 / 身份'],['encoder','Encode / 特征读入'],['network','跨实例载荷'],['prefill','LM Prefill'],['decode','LM Decode']],410);
 const state=t>=d.pEnd?'本例 Prefill 完成，6 个输入位置的 LM KV 已建立':t>=d.pStart?'LM 正在处理输入；分配或拿到特征均不能替代计算完成':'编码特征就绪后才能进入本例 Prefill';
 return {overview:table(['对象','形状 / 身份','下一步怎样使用'],[['原媒体',esc(d.mediaKey),'按模型 processor 转成可编码输入'],['编码特征','4×4（教学形状）','与文本位置共同组成 LM 输入'],['LM 位置','文本 2 + 特征 4 = 6','经过各层 Attention 计算并写 K/V'],['语言模型 KV',t>=d.pEnd?'本例 6 个位置已完成':'尚未完成','D 按 KV 兼容性、就绪与引用条件接续']])+facts([['部署选择',d.options.roles],['特征缓存',d.featureHit?'身份匹配，可读入':d.options.cache==='hit'?'媒体变化，旧命中作废':'本例重新编码'],['读入成本假设',d.options.readCost+' u（可高于编码）']]),detail:svg(s,675,'多模态输入位置及 E/P/D 的交接边界'),title:state,description:'一个媒体占位符可以展开成多个模型位置。图块、向量与位置只是形状示例；编码特征和 LM KV 有不同的产生与就绪条件。',rows:[['模型位置','示意 p0…p5，不是实际 tokenizer ID'],['特征命中条件','媒体、预处理和编码模型身份一致'],['P→D 载荷',d.pd?'语言模型 KV 与元数据；需单独确认接收完成':'本例 P/D 共实例，不画 KV 网络传输'],['放行判断',state]],source:t<d.pStart?0:1};
}
function routing(d,t,ctx){
 let s=text(24,28,'Router 的提示不等于 Worker 当前持有的 KV；恢复还要有 Agent 状态','p-strong');
 s+=rect(35,65,260,140)+text(52,95,'Router：估计 / 策略','p-strong')+text(52,127,'选择 '+d.w.id+'，只转发请求','p-small')+text(52,160,'本例不会自动搬迁 KV','p-small');
 d.workers.forEach((w,i)=>{const y=55+i*100;s+=rect(590,y,365,85,'var(--soft)',d.choice===i?'stroke="var(--accent)" stroke-width="2"':'')+text(608,y+26,w.id+'：提示 '+w.hint+' / 实际 '+w.resident,'p-strong')+text(608,y+56,'排队指标 '+w.backlog+'；Worker 再核对键','p-small');});
 const e=d.events.find(e=>e.id==='request');s+=wire('route-request',[[295,130],[450,130],[450,95+d.choice*100],[590,95+d.choice*100]],e,t,ctx.reduced,'请求与关联元数据，不是 KV 载荷');
 s+=tracks(d,t,[['router','Router 决策'],['network','请求传输'],['worker','Worker 校验 / 计算'],['agent','Agent 恢复条件']],295);
 return {overview:table(['Worker','索引提示前缀','实际可复用','负载指标','本轮选择'],d.workers.map((w,i)=>[w.id,String(w.hint),String(w.resident),String(w.backlog),i===d.choice?'选中':'未选']))+facts([['缺少的输入位置',d.missing+' / 12'],['完整恢复状态',d.canResume?'消息、工具结果与引用均齐':'缺少工具结果'],['可接续生成',d.canResume?'本例可重建或复用后继续':'恢复条件未齐']]),detail:svg(s,505,'缓存局部性、负载与恢复状态'),title:!d.canResume&&t>=3?'KV 可能仍在，但 Agent 恢复条件未齐':t<2?'先用提示与负载选择目标':t<3?'Worker 核对实际缓存键与驻留':'按真实缺口重算或继续生成',description:'本例均衡规则是教学阈值，不等同于 Router 的完整策略。索引可能过期，Worker 匹配才决定实际复用；Agent 的消息、工具结果和外部引用须独立恢复。',rows:[['本例规则','均衡模式下，W0 比 W1 多 4 以上负载则选 W1'],['缓存键',d.cacheKey.join(' / ')],['Worker 的实际缺口',d.missing+' 个位置'],['未知事项','未测真实等待时间、网络搬运或缓存收益']],source:t<2?0:1};
}
function faults(d,t){
 const logs=table(['身份','给定的证据'],d.logs.map(([rank,line])=>[esc(rank),esc(line)]));
 const lanes=[['rank0','Rank 0'],['rank1','Rank 1'],['rank2','Rank 2'],['client','客户端']].filter(([id])=>d.events.some(e=>e.lane===id));
 return {overview:logs+facts([['记录身份','全部为 synthetic 教学样本'],['当前证据',d.conclusive?'本例关键参与者齐全':'缺少 rank0 证据'],['可支持的判断',d.conclusive?d.answer:'仍需补证，不能定根因']]),detail:svg(text(24,28,'按同一教学时间轴对齐 rank、请求 / microbatch 与等待','p-strong')+tracks(d,t,lanes,95),130+lanes.length*47,'分布式故障的首错与后续等待')+logs,title:d.conclusive?d.answer:'证据缺口：先补关键参与者',description:d.conclusive?d.next:'当前只看到等待方。先采集同一请求/序号的上游日志、栈和状态快照；缺少证据时保留多个候选原因。',rows:[['时间口径','单机统一 u；不直接相减跨机 monotonic 时刻'],['故障起点','最早异常与产生等待的前驱分别核对'],['等待状态','栈停在 recv/collective 只说明此刻在等'],['操作边界','仅查看 synthetic 样本，没有连接集群或执行故障注入']],source:d.options.scenario==='oom'?0:1};
}
function lora(d,t,ctx){
 const loaded=t>=4,ready=t>=5,finished=t>=9;
 const rows=d.rows.map(r=>[r.id,esc(r.name),esc(r.identity),r.admitted?String(r.slot):'等待身份额度',r.admitted&&loaded?'已装载':'未进入本轮',r.admitted&&t>=8?'['+r.output.join(', ')+']':'尚未产出']);
 let s=text(24,28,'公共 W 不合并某个 adapter；每一行按自己的身份选择 A/B','p-strong');
 s+=rect(30,65,260,130)+text(47,94,'CPU 登记表 / A、B 权重','p-strong')+text(47,127,'名字 ≠ 内部 ID ≠ GPU 槽号','p-small')+text(47,160,'原始请求顺序始终保留','p-small');
 s+=rect(605,65,350,130)+text(623,95,'单 GPU 槽位与本轮 metadata','p-strong')+text(623,129,d.admitted.map(r=>r.identity+'→槽'+r.slot).join('；'),'p-small')+text(623,165,ready?'本轮映射已准备':'装载与映射分别确认','p-small');
 s+=wire('lora-load',[[290,130],[605,130]],d.events.find(e=>e.id==='load'),t,ctx.reduced,'A/B 权重 H2D；本轮计算再按本地槽位引用');
 s+=tracks(d,t,[['cpu','登记 / 本轮映射'],['h2d','A/B 权重复制'],['gpu','基础层 + LoRA 增量']],285);
 return {overview:table(['请求','adapter 名','内部身份','槽号示例','本轮就绪','输出'],rows)+facts([['身份预算',d.options.capacity+' / 需要 '+d.identities],['等待请求',d.waiting.map(r=>r.id).join('、')||'无'],['beta 的旧前缀',d.kvHit?'ID 相同；仍需键与驻留匹配':'新 ID，旧 beta 前缀不能直接命中'],['卸载条件',finished?'本轮引用已退役，仍核对其他请求':'本轮仍可能持有引用']]),detail:svg(s,455,'Multi-LoRA 身份、装载、混合组批与引用')+table(['请求','X Aᵀ','缩放增量','Y = X + 增量'],d.rows.map(r=>[r.id,f(r.low),'['+r.delta.join(', ')+']','['+r.output.join(', ')+']'])),title:t<4?'登记、CPU 装载与 GPU 驻留是不同状态':t<8?'按每行的 adapter 身份计算增量':'原行回填与资源收尾分开核对',description:'本例 W 为单位矩阵、X=[1,2]，所有结果是教学算术。容量按不同 adapter 身份计数，包含基础模型 None；槽号只是演示映射，KV 键使用身份而非槽号。',rows:[['缩放公式','Y=XWᵀ+(alpha/r)(XAᵀ)Bᵀ'],['低秩 r','矩阵维度，不是 GPU rank'],['旧 KV 键',d.oldKey],['本轮 beta 键',d.newKey],['公平与释放','drain、权重驱逐、动态卸载各有不同条件']],source:t<5?0:1};
}
function vllm(d,t){
 const {computed,selected,received,retired,allocated,covered}=root.PriorityLessonModels.vllmState(d,t);
 let s=text(24,28,'VR1：位置账本、每轮 token 预算和 KV block 映射用同一时钟','p-strong');
 for(let i=0;i<8;i++){const x=50+i*110;s+=rect(x,68,95,70,i<computed?'color-mix(in srgb,var(--accent) 24%,var(--bg))':'var(--soft)','stroke="'+(i<allocated*d.block?'var(--accent)':'var(--border)')+'" '+(i>=computed?'stroke-dasharray="4 3"':''))+text(x+12,94,i<6?'prompt p'+i:'输出 y'+(i-5),'p-small')+text(x+12,121,i<computed?'KV 已计算':i<6?'待计算':i<6+selected?'ID 已选':'尚未选择','p-small')+(i>=6&&i>=computed&&i<6+selected?text(x+12,136,'KV 未写','p-small'):'');}
 s+=text(50,173,'block 0：位置 0–3；block 1：位置 4–7。绿色边框为本请求分配，填色为已计算记录。','p-small')+tracks(d,t,[['frontend','输入 / 输出处理'],['scheduler','Scheduler：预算 / 块'],['runner','Worker / ModelRunner']],245);
 return {overview:facts([['已计算位置',computed+' / prompt '+d.prompt],['设备已选输出 ID',selected+' / 目标 2'],['前端已收输出 ID',received+' / 目标 2'],['逻辑已知 token',d.prompt+selected],['本请求已分配 block',allocated],['计算记录覆盖 block',covered]])+table(['轮次','输入位置','本轮 token 数','分配 block','计算后位置','是否采样'],d.steps.map(r=>[String(r.index+1),r.from+'…'+(r.from+r.count-1),String(r.count),String(r.blocks),String(r.computed),r.sampled?'本轮已到预测端点':'仍在分块 Prefill'])),detail:svg(s,415,'vLLM V1 请求、Scheduler、KV 与 Runner 的同窗账本'),title:t<1?'前端校验后，请求成为 EngineCore 的状态':t>=d.duration-2?'结束后归还引用；尾部新选 ID 尚未写 KV':'本轮只处理 Scheduler 分配的 token 位置',description:'V1 用 token 预算组织 Prefill 与 Decode，Runner 执行并选出 ID，前端稍后收到结果。表中完整排程用于核算；'+(retired?'本请求引用已退役，填色保留曾计算的位置，不代表仍持有这些块。':'绿色边框是已分配空间；填色是已计算位置，选出 ID 不等于前端已收到。'),rows:[['固定阅读路径','Processor → EngineCore → Scheduler / KV manager → ModelRunner V2'],['本轮 token 预算',String(d.options.budget)],['终点账本',d.computed+' 个已计算位置，'+d.totalIds+' 个已知 ID'],['页与 ID','页空间已分配不等于该位置 KV 已可读'],['版本边界','vLLM 独立 commit；不与 SGLang 算例拼成同一次运行']],source:t<1?0:t<5?1:2};
}
root.PriorityLessonViews={sampling,stopping,quantization,quantizationPath,moe,multimodal,routing,faults,lora,vllm,esc,f,table,facts};
})(window);
