/* Narratives consume the existing source-pinned models; no new scheduler policy. */
(function(root){
'use strict';
const make=(name,title,intro,boundary,frames,extra={})=>({name,title,intro,boundary,frames,...extra});
const source=(m,key)=>{const [file,line]=m.sources[key];return `https://github.com/sgl-project/sglang/blob/${m.revision}/${file}#L${line}`;};
function create(E,K,S){const out={};
 if(E){
 out['model-execution']=[make('DeepSeek-V3 · Eager MLA','DeepSeek-V3 与 MLA：一次模型调用怎样执行？','保留完整调用地图，只展开当前计算位置。请求编号、向量、Attention 结果和词表分数分别标注，跟随当前计算显现。','沿用本课 '+E.revision.slice(0,10)+'：单 rank、普通 Decode、Eager / Triton MLA，无 overlap、投机或 TBO。模型层画成概念展开，省略重复层与完整张量维度。MLA 历史用压缩状态示意，不套用通用 MHA 逐头缓存布局。41、52、8、7 是教学编号，不是真实模型预测。',E.trace.map(e=>({title:e.action,caption:e.caption,evidence:source(E,e.source)})),{kind:'execution',model:E}),
 make('CUDA Graph · 3 条请求装入容量 4','CUDA Graph 容量中的补齐行是什么？','图桶（graph bucket）指按固定输入形状准备的 CUDA Graph 执行规格。本例容量为 4，只有 R1、R2、R3 三条真实请求；PAD 是 padding（补齐行），用于匹配形状，不是第四条请求。','沿用本课 graph(3, ready) 的固定模型；图已经捕获且其余资格满足。不解释捕获过程，不表示真实请求数增加，也不将图重放结果缓存理解为新一轮直接复用旧答案。',[
 {title:'只有三条真实请求',caption:'本轮每条普通 Decode 请求各输入一个新位置；先固定真实成员。'},
 {title:'选择可容纳的图桶',caption:'本例可用容量为 1、2、4；三条请求选容量 4。容量不等于新增一条用户请求。'},
 {title:'装入当前输入，补齐形状',caption:'R1、R2、R3 写入前三行，第四行只用于补齐。当前输入与必要元数据仍要更新。'},
 {title:'重放设备工作',caption:'按已捕获的执行结构运行本轮计算，产生与桶形状对应的结果；不是直接拿上一轮答案。'},
 {title:'只取真实结果',caption:'只保留三条真实请求对应的 logits 行，再交给后续采样。PAD 不产生对外请求结果。'}],{kind:'graph',model:E})];
 }
 if(E)out['model-execution']=out['model-execution'].filter(s=>s.kind==='execution');
 if(K)out['kv-memory']=[8,9,11].map(target=>make('扩展到 '+target+' 个位置','KV Memory Pool：逻辑位置怎样映射到物理 KV？','上方按请求顺序排列，下面的物理页固定不动。先领地址，再填数据；尾页能写下时，不必另外领一整页。','沿用本课 '+K.revision.slice(0,10)+'，普通分页分配器、RadixCache 与静态 MHA KV 池。每页 4 个位置，页号与目标长度为教学值；不代表 DeepSeek-V3 MLA 的实际布局。映射线表示引用地址，不是一份额外 KV 副本。新增整页数为 ceil(目标长度/4)−ceil(6/4)；同一位置的地址用于定位各层相应的 KV，图只放大一层的槽位。',[
 {title:'已有六个位置',caption:'逻辑 p0–p5 连续，但分布在物理页 1 和页 7。页 7 的两个空位仍属于 R1。'},
 {title:'预留新增位置的地址',caption:'先使用尾页空位；需要时才领取页 3。映射建立与数据写入是两件事。'},
 {title:'先写满尾页',caption:'p6、p7 写进页 7 的槽 30、31。地址并未随逻辑顺序重新排列。'},
 {title:target===8?'尾页已经足够':'继续写新增页',caption:target===8?'本例写到 p7 已完成，不需要新页。':`其余位置写进页 3；目标共 ${target} 个位置，页内剩余空位仍由这个请求持有。`},
 {title:'沿映射核对最后一个位置',caption:'高亮最后一个逻辑位置及其真实槽地址。空闲整页与持有页里的空位要分开计数。'}],{kind:'memory',model:K,target}));
 if(S){
 const configs=[['fcfs','compute','按到达顺序 · 输入额度'],['lpm','compute','长前缀优先 · 输入额度'],['fcfs','exact','KV 恰好等于检查量'],['lpm','memory','保护前缀后检查 KV'],['fcfs','rows','请求行只剩一个']];
 for(const policy of ['fcfs','lpm'])for(const scenario of Object.keys(S.cases))if(!configs.some(c=>c[0]===policy&&c[1]===scenario))configs.push([policy,scenario,`${policy.toUpperCase()} · ${{compute:'输入额度 6',first:'首条超过额度 3',memory:'KV 预算 12',exact:'KV 等号边界',rows:'请求行 1',room:'额度充足'}[scenario]}`]);
 out.scheduling=configs.map(([policy,scenario,name])=>{const events=S.admission(policy,scenario);return make(name,'Admission Control：请求怎样通过准入检查？','把每条请求的“已命中历史”和“还要计算的输入”画出来，再分别核对输入额度、KV 检查量与请求行（ReqToTokenPool 中供一个请求记录 token→KV 地址映射的槽位）。队列排序只决定检查顺序。','沿用本课准入算例与 '+S.revision.slice(0,10)+' 固定基线。普通文本、page_size=1、无 chunk / HiCache / 特殊模型。KV 检查量是未命中输入 + 输出上限 2 + 1，并使用严格 <；图中账本是计划预留，不是物理 KV 已写入。',events.map(d=>{
 const r=S.requests.find(r=>r.id===d.current),need=r?r.input-r.hit+r.maxNew+1:0;
 const title={waiting:'请求还在等待',sorted:'只调整检查顺序',check:'拿一个候选来检查',commit:'通过检查，加入计划',blocked:'这一项条件未通过',stop:'停止继续追加',ready:'本轮计划已确定'}[d.kind];
 const caption=d.kind==='check'?`${d.current} 的新输入为 ${r.input-r.hit}，KV 检查量为 ${need}。先临时保护命中的前缀，再看剩余可用量。`:d.kind==='blocked'?d.reason==='kv'?`KV 检查要求 ${need} < ${d.checkAvailable}，本例不成立；候选未提交，临时保护退回。`:d.reason==='rows'?'请求行额度已经用完；即使别的额度充足，也不能继续加入。':'本例剩余输入额度不够；候选未提交，停止继续追加。':d.kind==='commit'?`${d.current} 加入计划；现在才提交相应的输入和 KV 预算。尚未执行模型，也没有把全部 KV 写完。`:d.kind==='stop'?'停止追加不等于把刚加入的请求取消。已经在计划里的成员仍会执行。':d.kind==='ready'?`本轮计划包含 ${d.admitted.join('、')||'无请求'}。其余候选继续等待；下一步的分配和执行不在本图假装完成。`:d.kind==='sorted'?'排序可以改变谁先检查，不能代替资源准入条件。':'每格是一份输入位置：深色命中历史，浅色为尚需计算的部分。';
 return {title,caption,state:d,evidence:source(S,d.kind==='sorted'?(policy==='lpm'?'lpm':'policy'):d.kind==='commit'?'commit':d.kind==='stop'?'stop':d.kind==='check'||d.kind==='blocked'?(d.config.resource==='kv'?'budget':d.config.resource==='rows'?'rows':'input'):'policy')};}),{kind:'admission',model:S,events,policy,scenario,scenarioLabel:{compute:'输入额度 6',first:'首条超过输入额度 3',memory:'KV 预算 12',exact:'KV 预算恰好为 9',rows:'仅 1 个请求行',room:'资源充足'}[scenario]});});
 out.scheduling.push(...[4,6,8].map(cap=>{const events=S.chunks(cap);return make('每轮 '+cap+' 个新位置','Chunked Prefill：已有长请求怎样分轮续算？','L 是已开始 Prefill 的长请求，还剩 L1–L10 十个位置；S 是等待中的短请求，需要 S1、S2。上方保留完整输入，下方逐轮列出实际选择的位置。','固定基线 '+S.revision.slice(0,10)+'，page_size=1，输入额度等于 chunk 额度，KV 与请求行充足；没有 Decode / mixed。已有 chunked_req 优先续算，位置格不表示实测耗时。',events.map(d=>({title:d.kind==='waiting'?'先看完整的待算位置':d.kind==='plan'?`第 ${d.round} 轮：先选择位置`:`第 ${d.round} 轮：完成所选位置`,caption:d.kind==='waiting'?'L 已在续算路径；S 还在等待队列。数字标记输入位置，不是请求数量。':d.kind==='plan'?(d.plan.s?'L 的末块留下额度，S 的两个位置也可进入本轮。':'本轮额度被 L 用满，S 仍等待；减小 chunk 不代表 S 自动抢占。'):'所选位置现在才计入完成历史。保留本轮结果，再观察下一轮从哪里继续。',evidence:source(S,d.kind==='waiting'?'chunkFirst':'chunk'),state:d})),{kind:'chunk',boundaryKey:'scheduling-chunk',model:S,events,cap,scenarioLabel:`每轮 ${cap} 个新位置`});}));

 }
 return out;
}
if(typeof module==='object'&&module.exports)module.exports={create};root.TeachingLessons={...(root.TeachingLessons||{}),...create(root.ExecutionModel,root.KVMemoryModel,root.SchedulingModel)};
})(typeof globalThis==='object'?globalThis:this);
