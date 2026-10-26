/* Deterministic teaching states, independent from SGLang and GPU execution. */
(function(root){
'use strict';
const tokenSource={model:'Qwen/Qwen2.5-0.5B-Instruct',revision:'7ae557604adf67be50417f59c2c2f167def9a775',sha256:'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',addSpecialTokens:false};
const examples=[{text:'我想学习人工智能。',pieces:['我想','学习','人工智能','。'],ids:[104100,100134,104455,1773]},{text:'我想学习机器学习。',pieces:['我想','学习','机器','学习','。'],ids:[104100,100134,102182,100134,1773]}];
const definitions={
 tokens:{label:'换一句话',options:['我想学习人工智能。','我想学习机器学习。'],steps:['你输入的原句','分成可复用的片段','查词表，取得编号','按原顺序排列输入','按编号查出向量','编号也能还原为文字']},
 generation:{label:'下一 token 怎样选择',options:['取最高分：贪心示例','按分布选择：固定的一次示例'],steps:['输入已经准备好','处理整段输入，得到候选分数（Prefill）','选择第一个输出','反分词，客户端看到第一段','把刚选出的 token 送回模型','根据新输入获得下一组候选（Decode）','选择并返回下一段','停止生成，进入收尾']},
 attention:{label:'Query 位置',options:['第 2 个位置','第 4 个位置','第 3 个位置'],steps:['四个位置，各有一行向量','从向量得到查询、键和值','比较查询与允许读取的键','把分数变成合计为 1 的权重','按权重混合各位置的值','遮罩回看：被遮项不进分母']},
 cache:{label:'观察哪件事',options:['每轮重算历史 K/V','保存历史 K/V','从空池分配页面','两条请求共享前缀'],steps:['先看输入与空缓存','第一次处理输入','开始下一轮工作','继续写入或复用','请求完成后的状态']},
 batching:{label:'调度方式',options:['固定批次对照','连续批处理','长输入分块'],steps:['A、B 准备开始；C 尚未到达','第一轮：A 完成，C 到达','第二轮：谁获得空位？','第三轮：B 完成','剩余请求继续']},
 hardware:{label:'只改变一项资源',options:['原配：搬运 8 u，计算 3 u','带宽翻倍','算力翻倍'],steps:['权重和输入先在存储里','把需要的数据送给计算单元','按两行权重做乘加','得到中间结果','比较搬运与计算的理想下界']},
 parallel:{label:'两张卡怎么分工',options:['TP：合算同一个结果','PP：前后层接力','DP：两个副本分请求'],steps:['先看未拆分的任务','确定各卡负责的部分','执行第一段工作','交接或继续另一份工作','取得完整结果']},
 communication:{label:'数据应该怎么合并',options:['求和：局部贡献','拼接：不同分片','通知先到，载荷后到'],steps:['各自持有自己的数据','提交交换，接收方仍等待','数据正在交接','接收并完成对应操作','结果可供后续计算']},
 deployment:{label:'服务角色',options:['合并部署','P/D 分离'],steps:['一个长输入，另一个请求在续写','P 处理输入并写入 K/V','产生首输出，准备继续','本地读取，或跨角色交接','D 获得可用状态','继续生成并分别收尾']},
 speculation:{label:'草稿在哪里分歧',options:['第 3 个不同','第 1 个不同','4 个全部一致'],steps:['普通生成：已有上一个输出','提出尚未确认的候选','目标模型验证候选前缀','保留连续接受部分，丢弃其后','提交输出与对应有效状态']},
 serving:{label:'跟踪场景',options:['正常返回','两个实例选择一个','客户端中途取消'],steps:['客户端准备发送 R1','服务接收，尚未执行','R1 等待执行机会','获得资源，开始计算','结果或取消通知正在回程','客户端与资源分别结束']},
 metrics:{label:'测量场景',options:['三个 token 分别到达','后两个 token 同包到达','20 个成功请求的 P95','最慢两个失败并被排除','10 秒窗口：请求数与 token 数'],steps:['先确定测量起点','记录首个可见输出','记录后续到达','统计有定义的时间区间','同时检查样本范围']}
};
// Different scenarios have different dependencies, so their step names must differ too.
const variantSteps={
 cache:{2:['空池：还没有请求占用','先分配一页，尚未写入','两个输入位置写满这一页','下一位置分配并写入另一页','保留快照，核对逻辑与物理位置'],3:['尚未建立共享历史','R1 建立并持有前缀','R2 也使用同一份前缀','R1 结束，R2 仍持有','请求结束，缓存可以继续保留']},
 batching:{2:['长输入等待，短请求正在续写','第一轮：续写一次，处理前两位置','第二轮：续写继续，长输入推进','第三轮：长输入最后一块完成','对照完整输入与输出进度']},
 parallel:{1:['两个微批次都要走完两个 PP stage','Stage 0 处理 MB1，Stage 1 等待','Stage 0 处理 MB2，Stage 1 处理 MB1','MB1 完成，Stage 1 继续 MB2','两个微批次都走完两段'],2:['两个副本，各有完整模型','R1 交给副本 A','R2 交给副本 B','两份请求可以独立推进','回顾副本分工与各自状态']},
 deployment:{0:['长输入与续写共享一个服务组','处理长输入，在本地留下 K/V','产生首输出，准备继续','后续计算读取本地 KV','得到执行机会，继续续写','回顾共享的计算资源']},
 serving:{2:['客户端准备发送 R1','服务接收，尚未执行','R1 等待执行机会','客户端断开，后端仍在执行','后端收到取消，等待安全收尾','在途操作已终止，归还资源']},
 metrics:{4:['打开一个 10 秒统计窗口','2 秒：R1 完成，输出 2 个 token','4 秒：R2 完成，累计 5 个 token','7 秒：R3 完成，累计 6 个 token','窗口结束：4 条请求，共 10 个 token'],2:['先看要统计的 20 个请求','每项都是一个成功请求的延迟','确定纳入统计的样本','排序后找到第 19 项','同时检查成功率与统计方法'],3:['先看原来的 20 个请求','保留成功与失败两类记录','最慢两项失败，不进入成功延迟分布','18 个成功样本中找到第 18 项','延迟降低，但成功率也降低']}
};
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const softmax=xs=>{const max=Math.max(...xs);const es=xs.map(x=>Math.exp(x-max));const sum=es.reduce((a,b)=>a+b,0);return es.map(x=>x/sum);};
// A reproducible draw makes the selection rule inspectable, not a mysterious
// hand-picked answer. The half-open intervals partition [0, 1).
function chooseToken(scores,draw=null){
 let total=0;const intervals=scores.map(({word,p})=>{const start=total;total+=p;return {word,p,start,end:total};});
 intervals[intervals.length-1].end=1; // Anchor the exclusive upper bound despite FP rounding.
 const selected=draw===null?intervals.reduce((a,b)=>b.p>a.p?b:a):intervals.find(x=>draw>=x.start&&draw<x.end);
 return {intervals,draw,selected:selected.word};
}
function attention(variant){
 // Variant 0 and 1 keep their original URL meaning (p2, p4); variant 2 adds p3.
 const queryByVariant=[1,3,2],query=queryByVariant[variant];
 const inputs=[[1,0],[0,1],[1,1],[2,0]],Q=inputs[query],keys=inputs,values=inputs.map(v=>v.map(x=>2*x));
 const scores=keys.map((k,i)=>i<=query?k.reduce((s,x,j)=>s+x*Q[j],0)/Math.sqrt(2):-Infinity);
 const exps=scores.map(s=>Number.isFinite(s)?Math.exp(s):0),denominator=exps.reduce((a,b)=>a+b,0);
 const weights=exps.map(e=>e/denominator),contributions=values.map((v,i)=>v.map(x=>weights[i]*x));
 const mixed=[0,1].map(j=>contributions.reduce((s,c)=>s+c[j],0));
 const hidden=[mixed[0],mixed[1],Math.max(0,mixed[0]-mixed[1])],output=[hidden[0]+hidden[2],hidden[1]+hidden[2]];
 return {inputs,query,queryPosition:query+1,allowed:query+1,Q,keys,values,scores,exps,denominator,weights,contributions,mixed,hidden,output,weightSum:weights.reduce((a,b)=>a+b,0)};
}
// Serving comparison is intentionally independent from the older R1 snapshot
// fields. The same pair of requests and the same complete model are placed in
// one shared instance or two independent instances; only the placement changes.
function servingComparison(){
 const requests=[{id:'R1',arrival:'先到'},{id:'R2',arrival:'随后到达'}];
 const shared={
  id:'shared',title:'1 个实例 · 共享容量',subtitle:'一个共享队列，一个执行槽',
  instances:1,modelCopies:1,executionSlots:1,executionCapacity:1,
  accepted:['R1','R2'],running:['R1'],waiting:['R2'],
  replicas:[{id:'shared',name:'共享实例',health:true,queue:['R1','R2'],executionSlot:'R1',modelCopies:1}],
  counts:{accepted:2,executing:1,waiting:1,modelCopies:1}
 };
 const split={
  id:'split',title:'2 个实例 · 各接一条',subtitle:'两条独立队列，各有一个执行槽',
  instances:2,modelCopies:2,executionSlots:2,executionCapacity:1,
  accepted:['R1','R2'],running:['R1','R2'],waiting:[],
  // The main trace follows R1 into B. R2 goes to A in this two-request
  // comparison, which explains why A is not doing R1 in the main animation.
  replicas:[
   {id:'A',name:'实例 A',health:true,queue:['R2'],executionSlot:'R2',modelCopies:1},
   {id:'B',name:'实例 B',health:true,queue:['R1'],executionSlot:'R1',modelCopies:1}
  ],
  counts:{accepted:2,executing:2,waiting:0,modelCopies:2}
 };
 return {
  key:'serving-contrast',requests,shared,split,
  capacity:'本例每个实例每轮只放 1 条请求，R1 先到先执行。真实推理实例可以把多条请求 batch（合批）在同一轮，单个实例不等于只能串行。',
  health:'health ✓ 只表示探活通过；它不等于实例空闲，也不等于请求已经进入执行槽。',
  cost:'完整模型权重副本 1→2；KV Cache、激活、通信与额外调度开销另计。',
  moment:'固定对照时刻：上方是另加入 R2 的固定对照快照；下方动画只包含 R1，追踪其全过程。'
 };
}
function servingRouteState(variant=0,step=0){
 const dual=variant===1,accepted=step>=1,routed=accepted&&(!dual||step>=2);
 const waiting=accepted&&routed&&step<3,executing=routed&&step>=3&&step<4,returning=routed&&step>=4&&step<5,visible=step>=5;
 return {
  dual,accepted,routed,selected:accepted?(dual?(routed?'实例 B':null):'实例 A'):null,target:dual?'B':'A',
  routePending:dual&&step===1,waiting,queueCount:waiting?1:0,executing,returning,visible,
  clientWaiting:!visible,
  stage:step<1?'arriving':step<2?'accepted':step<3?'waiting':step<4?'executing':step<5?'returning':'visible'
 };
}
function schedule(variant,step){
 if(variant===2){const round=Math.min(step,3),done=round*2;return {round,chunk:2,longDone:done,longTotal:6,shortOutputs:round,active:step<3?['短请求：续写一个位置','长请求：下一块 2 个位置']:[],waiting:[],done:step>=3?['长请求输入处理完成']:[]};}
 let remaining={A:1,B:3,C:1},waiting=['A','B'],active=[],done=[],rounds=[];
 for(let tick=0;tick<step;tick++){
  if(variant===1||active.length===0){while(active.length<2&&waiting.length)active.push(waiting.shift());}
  const running=active.slice();running.forEach(id=>remaining[id]--);const finished=active.filter(id=>remaining[id]===0);done.push(...finished);active=active.filter(id=>remaining[id]>0);
  if(tick===0)waiting.push('C');
  rounds.push(running);
 }
 return {remaining,waiting,active,done,rounds,round:step};
}
function snapshot(key,variant=0,step=0){
 const def=definitions[key];if(!def)throw Error('Unknown foundation: '+key);
 if(!Number.isInteger(variant)||variant<0||variant>=def.options.length||!Number.isInteger(step)||step<0||step>=def.steps.length)throw Error('Invalid teaching state');
 const d={key,variant,step,title:(variantSteps[key]?.[variant]||def.steps)[step],last:def.steps.length-1};
 if(key==='tokens')return {...d,...examples[variant],count:examples[variant].ids.length};
 if(key==='generation'){
  // The published candidate probabilities are exact for these logits:
  // [ln 6, ln 3, 0] -> [6, 3, 1] / 10 -> [.6, .3, .1]; the next round uses
  // [ln 4, 0] -> [4, 1] / 5 -> [.8, .2]. This is an equivalent hand example,
  // not a score from a trained model.
  const scores=[{word:'学习',p:.6,z:Math.log(6)},{word:'休息',p:.3,z:Math.log(3)},{word:'。',p:.1,z:0}],choice=chooseToken(scores,variant===0?null:.75),first=choice.selected;
  const nextScores=[{word:first==='学习'?'。':'一下',p:.8,z:Math.log(4)},{word:'其他',p:.2,z:0}],nextChoice=chooseToken(nextScores,variant===0?null:.25),second=nextChoice.selected;
  const expScores=scores.map(x=>Math.exp(x.z)),nextExpScores=nextScores.map(x=>Math.exp(x.z));
  return {...d,first,second,scores,nextScores,choice,nextChoice,selected:step>=2?first:null,visible:step>=6?first+second:step>=3?first:'',kv:step>=5?['输入1','输入2',first]:step>=1?['输入1','输入2']:[],newInput:step>=4?first:null,logits:scores.map(x=>x.z),expScores,expSum:expScores.reduce((a,b)=>a+b,0),nextLogits:nextScores.map(x=>x.z),nextExpScores,nextExpSum:nextExpScores.reduce((a,b)=>a+b,0)};
 }
 if(key==='attention')return {...d,...attention(variant)};
 if(key==='cache'){
  const round=clamp(step-1,0,2),history=4+round;
  return {...d,round,history,computed:step===0?[]:Array.from({length:round+1},(_,i)=>variant===0?4+i:i?1:4),kv:step===0||variant===0?0:history,ownedPages:variant===2?(step<1?0:step<3?1:2):0,written:variant===2?(step<2?0:step<3?2:3):0,holders:variant===3?(step===0?[]:step===1?['R1']:step===2?['R1','R2']:step===3?['R2']:[]):[],retained:variant===3&&step>0};
 }
 if(key==='batching')return {...d,...schedule(variant,step)};
 if(key==='hardware'){const move=variant===1?4:8,compute=variant===2?1.5:3;return {...d,input:[2,3],weights:[[1,2],[3,4]],output:[8,18],move,compute,bound:Math.max(move,compute)};}
 if(key==='parallel'){
  const lanes=step===0?[null,null]:step===1?['MB1',null]:step===2?['MB2','MB1']:step===3?[null,'MB2']:[null,null];
  return {...d,partial:[2,6],total:8,lanes,completed:step>=4?['MB1','MB2']:step>=3?['MB1']:[],requests:step===0?[[],[]]:step===1?[['R1'],[]]:[['R1'],['R2']]};
 }
 if(key==='communication')return {...d,input:[[1,2],[10,20]],result:variant===0?[11,22]:[1,2,10,20],metadata:variant===2&&step>=1,arrived:step>=3,usable:step>=4};
 if(key==='deployment')return {...d,kv:step>=1,first:step>=2,transit:variant===1&&step===3,received:step>=4,decode:step>=4,weights:['完整模型','完整模型']};
 if(key==='speculation'){
  const accepted=[2,0,4][variant],draft=['我','想','喝','茶'],replacement=variant===2?'。':variant===1?'你':'吃';
  // One fixture drives both the foundation visual and the older step lab.
  // The submitted sequence is the reference for ordinary per-token calls;
  // the verification is one call over five input positions, not five calls.
  const submitted=[...draft.slice(0,accepted),replacement],verificationPositions=5;
  return {...d,draft,accepted,replacement,submitted,
   output:step>=4?submitted:[],validKV:step>=4?['根',...draft.slice(0,accepted)]:[],
   draftCalls:4,verificationCalls:1,verificationPositions,
   targetCallsSequential:submitted.length,discardedCount:draft.length-accepted,
   bonusToken:submitted[submitted.length-1],bonusKVComputed:false};
 }
 if(key==='serving')return {...d,location:step,clientDone:variant===2?step>=3:step>=5,cancelReceived:variant===2&&step>=4,resourceHeld:step>=3&&step<5,visible:variant!==2&&step>=5,selected:variant===1?'实例 B':'实例 A',route:servingRouteState(variant,step),comparison:servingComparison()};
 if(key==='metrics'){const samples=Array.from({length:20},(_,i)=>(i+1)*10),success=variant===3?samples.slice(0,18):samples;return {...d,arrivals:variant===1?[100,180,180]:[100,140,180],ttft:100,tpot:40,samples,success,p95:success[Math.ceil(.95*success.length)-1],failed:20-success.length,window:10,completions:[2,4,7,9],outputCounts:[2,3,1,4],completed:Math.min(step,4),outputTotal:[0,2,5,6,10][step],qps:.4,outputTPS:1};}
 return d;
}

// Pure comparison fixtures for the cache lesson. They are deliberately separate
// from the older snapshot variants so the existing scene ids keep their meaning.
function cacheHistory(){
 const calls=[4,5,6], saved=[4,5,6], readCounts=[0,4,5];
 return {calls, saved, readCounts, computedNoCache:15, computedCache:6, readTotal:9,
  note:'15/6 是 K/V 投影位置数，不是总工作量或速度；读取历史 4+5=9 仍要访问 K/V。'};
}
function cachePages(){
 const pageSize=2, pageCount=8;
 const occupiedPages=[0,2,3,5,6,7], freePages=[1,4];
 const pages=Array.from({length:pageCount},(_,page)=>({page,slots:[null,null]}));
 // New teaching pool (not the empty-pool page 7/2 animation): six full pages
 // leave two separate two-slot holes, so the same 3-position demand fits only
 // when it may use slots from both holes.
 for(const page of occupiedPages) pages[page].slots=['已有占用','已有占用'];
 const before=pages.map(p=>p.slots.slice());
 const mapping=[
  {position:'p1',page:4,slot:0},
  {position:'p2',page:4,slot:1},
  {position:'p3',page:1,slot:0}
 ];
 for(const m of mapping) pages[m.page].slots[m.slot]=m.position;
 const freeRuns=freePages.map(page=>({page,slots:[0,1],length:2}));
 const after=pages.map(p=>p.slots.slice());
 const count=rows=>rows.reduce((n,slots)=>n+slots.filter(Boolean).length,0);
 const beforeOccupied=count(before), afterOccupied=count(after);
 const usedPages=[...new Set(mapping.map(m=>m.page))];
 // Two logical pages, two block-table entries. The three per-position rows are
 // a separate address derivation, not three page-table entries.
 const pageTable=[
  {entry:1,logicPage:0,positions:['p1','p2'],page:4,offset:0},
  {entry:2,logicPage:1,positions:['p3'],page:1,offset:0}
 ];
 const positionRows=mapping.map((m,i)=>({position:m.position,logicPage:Math.floor(i/pageSize),page:m.page,slot:m.slot,offset:m.slot}));
 const tailPage=pageTable[pageTable.length-1].page;
 const tailFreeSlots=after[tailPage].map((value,slot)=>value?null:slot).filter(slot=>slot!==null);
 const totalFree=pageSize*pageCount-beforeOccupied, maxContiguous=Math.max(...freeRuns.map(run=>run.length));
 const beforeAllocatedPages=occupiedPages.length, allocatedPages=pageCount;
 const beforeCapacity=beforeAllocatedPages*pageSize, allocatedCapacity=allocatedPages*pageSize;
 const validPositions=afterOccupied, allocatablePages=freePages.length, tailRemainder=tailFreeSlots.length;
 const occupiedUnchanged=before.every((slots,page)=>slots.every((value,slot)=>value===null||after[page][slot]===value));
 return {pageSize,pageCount,occupiedPages,freePages,pages,before,after,mapping,pageTable,positionRows,
  freeRuns,totalFree,maxContiguous,usedPages,
  continuous:{request:3,fits:false,maxRun:maxContiguous,metadataEntries:1,
   reason:'最大连续空槽 2 < 3；起始只有 6 个旧页的 12 个有效位置，页 1、4 是 2 个可分配页。'},
  paged:{fits:true,usedPages,pageTableEntries:pageTable.length,metadataEntries:pageTable.length,
   positionRows:positionRows.length,tailFree:{page:tailPage,slots:tailFreeSlots},tailSlots:tailRemainder,tailRemainder,
   beforeOccupied,afterOccupied,validPositions,beforeAllocatedPages,allocatedPages,
   beforeCapacity,allocatedCapacity,allocatablePages:0,unusedCapacity:tailRemainder},
  occupancyOnly:{beforeOccupied,beforeFree:totalFree,beforeAllocatedPages,beforeCapacity,
   allocatablePages,afterOccupied,afterAllocatedPages:allocatedPages,
   afterCapacity:allocatedCapacity,afterAllocatablePages:0,tailRemainder},
  occupiedUnchanged,
  note:'新教学池例：同一个 8 页池先有 6 个已分配旧页、12 个有效位置，页 1、页 4 是 2 个可分配页。连续预留需要 3 个相邻槽，但最大只有 2；分页把逻辑页 0（p1、p2）映射到物理页 4，逻辑页 1（p3）映射到物理页 1，是 2 条页表映射；另列 3 行逐 token 位置演算。分页后池内 8 页都有归属：6 个旧页保持不变、仍归已有请求，页 4、页 1 这 2 个原可分配页归新请求。新请求 3 个位置沿用页 4 的两槽和页 1 的一槽；有效位置 15、已分配容量 16、可分配页 0、末页余量 1。页 1 的 1 个空槽属于新请求的末页，不是池里可分配的空槽；原有 6 页的所有者没有被改写。'};
}
function cachePrefix(){
 const prefix=['p1','p2','p3'], r1Suffix=['a1','a2'], r2Suffix=['b1','b2'];
 const suffixCopies=r1Suffix.length+r2Suffix.length;
 const independentPrefix=prefix.length*2, sharedPrefix=prefix.length;
 const compatibility=[
  {key:'tokens',label:'token 前缀 p1 p2 p3',ok:true},
  {key:'model',label:'模型与权重',ok:true},
  {key:'config',label:'影响计算的配置/状态',ok:true}
 ];
 return {prefix,r1Suffix,r2Suffix,
  independent:{prefixCopies:independentPrefix,suffixCopies,total:independentPrefix+suffixCopies,refs:0,
   rows:[prefix.slice(),prefix.slice()],suffixRows:[r1Suffix.slice(),r2Suffix.slice()]},
  shared:{prefixCopies:sharedPrefix,suffixCopies,total:sharedPrefix+suffixCopies,refs:2,
   rows:[prefix.slice()],suffixRows:[r1Suffix.slice(),r2Suffix.slice()]},
  compatible:{tokens:true,model:true,config:true},compatibility,mismatchNoReuse:true,
  refTimeline:[{phase:'R1 持有',refs:1},{phase:'R2 接入',refs:2},{phase:'R1 结束',refs:1},{phase:'R2 结束',refs:0}],
  retainedAtZero:true,evictionSeparate:true,
  savings:{prefixPositions:independentPrefix-sharedPrefix,
   totalPositions:(independentPrefix+suffixCopies)-(sharedPrefix+suffixCopies)},
  note:'共同前缀 2×3=6 → 1×3=3，省 3 个前缀位置；各自后缀 a1/a2 与 b1/b2 仍分开保存，不做合并。引用 0 只表示没有活跃请求，缓存仍可保留；驱逐是另一个条件。'};
}

const api={definitions,snapshot,attention,schedule,chooseToken,examples,tokenSource,cacheHistory,cachePages,cachePrefix,servingComparison,servingRouteState};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.FoundationModel=api;
})(typeof globalThis!=='undefined'?globalThis:this);
