/* Pure teaching models. No requests, kernels, random sampling or performance prediction. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.PriorityLessonModels = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const clamp = (n, a, b) => Math.min(b, Math.max(a, n));
  const progress = (t, a, b) => clamp((t - a) / (b - a), 0, 1);
  const softmax = scores => {
    const m = Math.max(...scores), exps = scores.map(z => Math.exp(z - m)), sum = exps.reduce((a,b)=>a+b,0);
    return exps.map(x=>x/sum);
  };
  function sampling(options = {}) {
    const o = Object.assign({temperature:2, topK:3, topP:.6, repetition:1, u:.75, scores:'positive', rule:'sglang'}, options);
    if (![0,.5,1,2].includes(+o.temperature) || ![1,2,3,4].includes(+o.topK) || !(o.topP>0&&o.topP<=1) || !(o.repetition>=1) || !(o.u>=0&&o.u<1)) throw new Error('无效采样参数');
    const logits = o.scores==='signed' ? [2,1,-1,0] : [Math.log(25),Math.log(9),Math.log(4),0];
    const history = [2,0,1,0];
    const penalized = logits.map((z,i)=>history[i] ? z<0 ? z*o.repetition : z/o.repetition : z);
    const temperature = +o.temperature;
    const scaled = penalized.map(z=>temperature===0 ? z : z/temperature);
    const probabilities = softmax(scaled), order = [0,1,2,3].sort((a,b)=>probabilities[b]-probabilities[a]||a-b);
    let before = 0;
    const ranked = order.map((id,rank)=> {
      const previous = before; before += probabilities[id];
      const keepK = temperature===0 ? rank===0 : rank<o.topK;
      // Exact representative SGLang PyTorch predicate: remove when previous > top_p.
      const keepP = temperature===0 || (o.rule==='concept' ? previous<o.topP : previous<=o.topP);
      return {id,rank,before:previous,keepK,keepP,weight:keepK&&keepP ? probabilities[id] : 0};
    });
    const mass = ranked.reduce((a,r)=>a+r.weight,0);
    const q = [0,0,0,0];ranked.forEach(r=>q[r.id]=r.weight/mass);
    let cumulative = 0;
    const intervals = ranked.map(r=> {const start=cumulative;cumulative+=q[r.id];return {id:r.id,start,end:cumulative};});
    const selected = temperature===0 ? order[0] : intervals.find(r=>r.end>o.u&&r.end>r.start)?.id;
    return {options:o,logits,history,penalized,scaled,probabilities,ranked,q,mass,intervals,selected,greedy:temperature===0,words:['学习','休息','。','<eos>']};
  }
  function stopping(options = {}) {
    const o=Object.assign({scenario:'eos'},options);
    const event=(id,start,end,title,detail)=>({id,start,end,title,detail});
    const cases={
      eos:[event('token',0,2,'输出普通 token','生成 ID 后经过反分词，文本“你好”可见。'),event('eos',2,3,'生成 EOS ID','EOS 是结束标记；本例保留原始 ID 计数，展示文字时不输出该标记。'),event('finish',3,4,'核对 stop 结束','原始生成 ID 为 2 项，文本分片与 ID 数分别记账。')],
      length:[event('one',0,1,'输出第 1 项','上限为 3 个新 token，输入 prompt 不计入该上限。'),event('two',1,2,'输出第 2 项','还未达到本例长度上限。'),event('three',2,3,'输出第 3 项','已达到 max_new_tokens=3，不要求再选中 EOS。'),event('finish',3,4,'核对 length 结束','结果是否截在句子中间需要业务判断；长度收尾不保证内容完整。')],
      min:[event('masked',0,1,'EOS 暂时不可选','min_new_tokens=2；当前只输出了 1 项，EOS 候选被屏蔽，尚未输出第 2 项。'),event('two',1,2,'选择另一个普通 token','本例累计输出达到 2 项。'),event('allowed',2,3,'恢复 EOS 选择机会','达到最低长度只解除限制，不强制下一项是 EOS。'),event('finish',3,4,'本例随后选到 EOS','实际停止仍由选出的 ID、停止匹配或长度上限决定。')],
      string:[event('a',0,1,'文本片段 1：回答：','本例给定分片只练习字符串匹配，未记录逐 token 到达。'),event('b',1,2,'文本片段 2：EN','累计文本为“回答：EN”，尚未匹配 END。'),event('c',2,3,'文本片段 3：D尾部','END 横跨两个分片；累计缓冲中能定位完整匹配，展示策略再决定是否保留停止串。'),event('finish',3,4,'核对匹配与可见范围','本例展示“回答：”；分片数不能当作输出 token 数，也不能据此证明后台资源已退役。')],
      cancel:[event('decode',0,3,'设备工作在途','教学依赖：已经提交的设备工作有独立完成点。'),event('notify',1,2,'取消通知返回','返回只确认通知路径；此时设备工作仍未完成。'),event('finish',3,4,'核对流尾与后台收尾','停止等待、结束原因和资源退役分别核对；完整流程见最小部署课。')]
    };
    if(!cases[o.scenario])throw new Error('未知停止场景');
    return {options:o,events:cases[o.scenario].map(e=>({...e,lane:e.id==='notify'?'client':'device',kind:e.id==='notify'?'control':'compute'})),duration:4};
  }
  function quantization(options = {}) {
    const o=Object.assign({bits:4,group:5,scale:'auto',object:'weight'},options);
    if(![4,8].includes(+o.bits)||![1,2,5].includes(+o.group))throw new Error('无效量化参数');
    const values=[-1,-.3,.1,.8,1.7], qmax=2**(o.bits-1)-1;
    const rows=values.map((x,i)=> {
      const g=Math.floor(i/o.group),part=values.slice(g*o.group,(g+1)*o.group);
      const scale=o.scale==='small' ? .1 : Math.max(...part.map(Math.abs))/qmax;
      const raw=Math.round(x/scale),q=clamp(raw,-qmax,qmax),restored=q*scale;
      return {x,group:g,scale,q,restored,error:restored-x,clipped:raw!==q};
    });
    const payload=Math.ceil(values.length*o.bits/8),metadata=Math.ceil(values.length/o.group)*2;
    return {options:o,values,rows,qmax,payload,metadata,total:payload+metadata,baseline:values.length*2,
      maxError:Math.max(...rows.map(r=>Math.abs(r.error))),clipped:rows.filter(r=>r.clipped).length};
  }
  const ev = (id,lane,start,end,title,kind='compute') => ({id,lane,start,end,title,kind});
  function quantizationPath(options={}) {
    const o=Object.assign({format:'fp8',supported:'yes'},options);
    const formats={fp8:{name:'已量化 FP8 权重 + 动态激活',stored:'FP8 weight + weight scale；激活本轮产生',method:'Fp8Config → Fp8LinearMethod.apply',compute:'量化当前激活并交给满足条件的 FP8 GEMM'},awq:{name:'普通 AWQ 的 W4A16 路径',stored:'打包的 qweight / qzeros / scales',method:'AWQConfig → AWQLinearMethod.apply',compute:'按所选 shape 分支做反量化或 AWQ GEMM；不由 INT4 标签推断执行 dtype'},kv:{name:'KV 数据格式的独立选择',stored:'模型权重配置与 KV dtype 分开',method:'ModelRunner / KV pool → Attention backend',compute:'按实际 K/V scale 和 attention backend 读写缓存'}};
    if(!formats[o.format])throw new Error('未知量化格式');
    return {options:o,...formats[o.format],duration:6,allowed:o.supported==='yes'};
  }
  function moe(options={}) {
    const o=Object.assign({distribution:'hot',replica:'off'},options), replica=o.replica==='on';
    const pairs=o.distribution==='spread' ? [[0,1],[2,3],[0,2],[1,3]] : [[0,1],[0,1],[0,1],[0,1]];
    const rows=pairs.map((experts,i)=>({id:'T'+i,request:i<2?'R1':'R2',origin:i<2?0:1,x:i+1,experts,weights:[.75,.25]}));
    const jobs=rows.flatMap(r=>r.experts.map((expert,j)=>({token:r.id,origin:r.origin,expert,rank:replica&&expert===0 ? +r.id.slice(1)%2 : Math.floor(expert/2),weight:r.weights[j],x:r.x,result:r.x*(expert+1)})));
    const loads=[0,1].map(rank=>jobs.filter(j=>j.rank===rank).length),logical=[0,1,2,3].map(e=>jobs.filter(j=>j.expert===e).length);
    const events=[ev('route','gates',0,1,'保留逻辑专家与权重'),ev('dispatch','network',1,3,'分发 token 行 / 本地重排','transfer')];
    loads.forEach((n,i)=>{if(n)events.push(ev('expert'+i,'gpu'+i,3,3+n,'Rank '+i+' 的 '+n+' 份专家任务'));if(n<Math.max(...loads))events.push(ev('wait'+i,'gpu'+i,3+n,3+Math.max(...loads),'等待本轮专家分支','wait'));});
    events.push(ev('combine','network',3+Math.max(...loads),5+Math.max(...loads),'整批 AllToAll Combine','transfer'));
    const join=5+Math.max(...loads);events.push(ev('sum','gates',join,join+1,'按原 token 行加权合并'));
    return {options:o,rows,jobs,loads,logical,events,duration:join+2,join,replica,remote:jobs.filter(j=>j.origin!==j.rank).length,
      outputs:rows.map(r=>({id:r.id,value:jobs.filter(j=>j.token===r.id).reduce((a,j)=>a+j.weight*j.result,0)}))};
  }
  function multimodal(options={}) {
    const o=Object.assign({roles:'e-p-d',media:'same',cache:'miss',readCost:3},options);
    const featureHit=o.cache==='hit'&&o.media==='same', separate=o.roles!=='single', pd=o.roles==='e-p-d';
    const featureEnd=featureHit?2+Number(o.readCost):5, pStart=featureEnd+(separate?2:0),pEnd=pStart+3,dStart=pEnd+(pd?2:0);
    const events=[ev('prepare','cpu',0,2,'核对媒体与展开位置'),ev(featureHit?'fetch':'encode','encoder',2,featureEnd,featureHit?'读取匹配的编码特征':'视觉编码与特征适配')];
    if(separate)events.push(ev('feature','network',featureEnd,pStart,'编码特征 + 位置元数据','transfer'));
    events.push(ev('prefill','prefill',pStart,pEnd,'语言模型 Prefill 写 K/V'));
    if(pd)events.push(ev('kv','network',pEnd,dStart,'语言模型 KV + 请求元数据','transfer'));
    events.push(ev('decode','decode',dStart,dStart+2,'当前生成 token 进入 Decode'));
    return {options:o,featureHit,separate,pd,pStart,pEnd,dStart,events,duration:dStart+3,
      positions:['文本 p0','图像 f0','图像 f1','图像 f2','图像 f3','文本 p5'],featureShape:[4,4],kvPositions:6,
      mediaKey:o.media==='same'?'img-A / processor-v1 / encoder-v1':'img-B / processor-v1 / encoder-v1'};
  }
  function routing(options={}) {
    const o=Object.assign({policy:'balanced',backlog:8,residency:'resident',checkpoint:'complete'},options);
    const workers=[{id:'W0',hint:8,resident:o.residency==='resident'?8:0,backlog:+o.backlog},{id:'W1',hint:0,resident:0,backlog:1}];
    const imbalance=workers[0].backlog-workers[1].backlog>4,choice=o.policy==='balanced'&&imbalance?1:0,w=workers[choice];
    const canResume=o.checkpoint==='complete',missing=12-w.resident,first=3+w.backlog+Math.ceil(missing/4),events=[ev('select','router',0,1,'前缀提示与负载共同选点'),ev('request','network',1,2,'转发请求；不搬运 KV','transfer'),ev('verify','worker',2,3,'Worker 校验实际可复用前缀')];
    if(canResume){events.push(ev('queue','worker',3,3+w.backlog,'给定的排队等待','wait'),ev('prefill','worker',3+w.backlog,first,'重算剩余 '+missing+' 个输入位置'),ev('decode','worker',first,first+2,'继续生成'));}
    else events.push(ev('blocked','agent',3,6,'工具结果缺失，恢复条件未齐','wait'));
    return {options:o,workers,choice,w,imbalance,canResume,missing,events,duration:canResume?first+3:7,
      cacheKey:['model revision','tokenizer / token IDs','adapter ID','media identity','cache salt']};
  }
  function faults(options={}) {
    const o=Object.assign({scenario:'oom',evidence:'complete'},options);
    const cases={
      oom:{root:0,answer:'OOM',events:[ev('compute','rank0',0,3,'执行到设备分配'),ev('oom','rank0',3,4,'首次分配失败','error'),ev('recv','rank1',3,8,'等待上游数据','wait'),ev('timeout','rank1',8,9,'后续等待超时','error')],logs:[['rank0','t=3 OOM: allocation failed'],['rank1','t=8 waiting for upstream tensor timed out']],next:'先核对 rank0 最早异常与显存账本；rank1 的后续超时可能是传播结果。'},
      collective:{root:0,answer:'collective 次序不一致',events:[ev('ar','rank0',1,8,'seq17 / all_reduce / count8','wait'),ev('ag','rank1',1,8,'seq17 / all_gather / count8','wait'),ev('timeout','rank1',8,9,'等待超时','error')],logs:[['rank0','seq=17 op=all_reduce count=8'],['rank1','seq=17 op=all_gather count=8']],next:'同组相同 seq 对应的操作不同。先核对控制流与参与者；不能仅凭 timeout 认定网卡故障。'},
      pp:{root:0,answer:'上游尚未产出',events:[ev('kvwait','rank0',1,6,'等待本地 KV 就绪','wait'),ev('recv','rank1',2,8,'recv MB2：尚未有对应 send','wait'),ev('idle','rank2',2,8,'等待后续阶段','wait')],logs:[['rank0','MB2 awaiting local KV-ready event; send not submitted'],['rank1','MB2 recv is pending'],['rank2','no MB2 input yet']],next:'沿 MB2 的生产者向前追。本例上游尚未提交发送；recv 栈本身不能证明通信层出错。'},
      cancel:{root:0,answer:'仍在等待安全退役',events:[ev('kernel','rank0',0,5,'设备工作在途'),ev('ack','client',1,2,'取消通知返回','control'),ev('cleanup','rank0',5,6,'设备完成后处理引用'),ev('retired','rank0',6,7,'本例资源退役','control')],logs:[['client','t=2 cancel notification returned'],['rank0','t=5 device event completed'],['rank0','t=6 request refs retired']],next:'t=2 的通知响应与 t=6 的资源退役是不同证据。设备依赖未满足时不能把引用记成已归还。'}
    };
    const c=cases[o.scenario];if(!c)throw new Error('未知故障场景');
    return {options:o,...c,events:c.events.filter(e=>o.evidence!=='missing'||e.lane!=='rank0'),duration:10,logs:c.logs.filter(([rank])=>o.evidence!=='missing'||rank!=='rank0'),conclusive:o.evidence!=='missing'};
  }
  function lora(options={}) {
    const o=Object.assign({capacity:3,revision:'original'},options),capacity=+o.capacity;
    const rows=[{id:'R1',name:'alpha',identity:'uA',slot:1,A:[1,0],B:[1,1],scale:2},{id:'R2',name:'beta',identity:o.revision==='reload'?'uB2':'uB',slot:2,A:[0,1],B:o.revision==='reload'?[2,-1]:[1,-1],scale:1},{id:'R3',name:'基础模型',identity:'None',slot:0,A:[0,0],B:[0,0],scale:0}];
    rows.forEach((r,i)=>{r.admitted=i<capacity;r.slot=r.admitted?(capacity===3?[1,2,0][i]:i):null;r.x=[1,2];r.low=r.x.reduce((a,x,j)=>a+x*r.A[j],0);r.delta=r.B.map(b=>r.scale*b*r.low);r.output=r.x.map((x,j)=>x+r.delta[j]);});
    const events=[ev('register','cpu',0,1,'登记名字与 adapter 身份'),ev('cpu','cpu',1,2,'CPU 权重与配置已准备'),ev('load','h2d',2,4,'A/B 权重读入 GPU 槽位','transfer'),ev('metadata','cpu',4,5,'按原请求行建立槽位映射'),ev('base','gpu',5,6,'共享基础层输出'),ev('delta','gpu',6,8,'按各自行选择 A/B 并计算增量'),ev('finish','cpu',8,9,'回填原请求输出并退役本轮引用')];
    return {options:o,rows,admitted:rows.filter(r=>r.admitted),waiting:rows.filter(r=>!r.admitted),events,duration:10,identities:3,
      kvHit:rows[1].identity==='uB',oldKey:'base-v1 / uB / prompt-P',newKey:'base-v1 / '+rows[1].identity+' / prompt-P'};
  }
  function vllm(options={}) {
    const o=Object.assign({budget:2,prefix:'hit'},options),budget=+o.budget,prompt=6,block=4;
    let computed=o.prefix==='hit'?4:0,emitted=0,time=1;const initialComputed=computed,steps=[],events=[ev('input','frontend',0,1,'输入校验与请求入队')];
    while(emitted<2){
      const target=prompt+emitted,count=Math.min(budget,target-computed),from=computed;
      const schedule=time;events.push(ev('schedule'+steps.length,'scheduler',time,time+1,'本轮 token 预算与 block 分配'));time++;
      events.push(ev('execute'+steps.length,'runner',time,time+2,'处理位置 '+from+'…'+(from+count-1)));time+=2;computed+=count;
      const computeEnd=time;
      const sampled=computed>=prompt+emitted;
      if(sampled){emitted++;events.push(ev('output'+steps.length,'frontend',time,time+1,'返回 y'+emitted));time++;}
      steps.push({index:steps.length,schedule,start:schedule+1,computeEnd,end:time,from,count,computed,emitted,sampled,blocks:Math.ceil((from+count)/block)});
    }
    events.push(ev('finish','scheduler',time,time+1,'结束并归还请求引用'));
    return {options:o,prompt,block,initialComputed,steps,events,duration:time+2,computed,emitted,totalIds:prompt+emitted};
  }
  function vllmState(d,t){
    const computed=d.steps.filter(s=>t>=s.computeEnd).at(-1)?.computed??d.initialComputed;
    const selected=d.steps.filter(s=>s.sampled&&t>=s.computeEnd).length;
    const received=d.steps.filter(s=>s.sampled&&t>=s.end).length;
    const retired=t>=d.events.find(e=>e.id==='finish').end;
    const planned=d.steps.filter(s=>t>=s.start).at(-1)?.blocks??Math.ceil(d.initialComputed/d.block);
    return {computed,selected,received,retired,allocated:retired?0:planned,covered:Math.ceil(computed/d.block)};
  }
  return {clamp,progress,softmax,sampling,stopping,quantization,quantizationPath,moe,multimodal,routing,faults,lora,vllm,vllmState};
});
