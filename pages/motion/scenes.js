/* Declarative, reversible teaching events. Positions describe a diagram, not elapsed
 * device time. Each frame is a complete snapshot; stable actor ids never change. */
(function(root){
'use strict';
const scenes={};
const pos=(column,row=0)=>({x:48+300*column,y:108+76*row});
const zones=(a,b,c)=>[a,b,c].map((v,i)=>({x:24+300*i,title:v[0],note:v[1]}));
function make(key,name,title,intro,zoneNames,boundary){
 const s={name,title,intro,zones:zones(...zoneNames),boundary,actors:[],frames:[]},state={};
 const api={
  actor(id,label,column,row=0,note='',color){s.actors.push({id,color});state[id]={...pos(column,row),label,note,opacity:1,count:0};return api;},
  frame(title,caption,changes={}){for(const[id,patch]of Object.entries(changes)){if(!state[id])throw Error('Unknown actor '+id);Object.assign(state[id],patch);}s.frames.push({title,caption,actors:s.actors.map(a=>({...state[a.id]}))});return api;},
  done(){(scenes[key]??=[]).push(s);return s;}
 };return api;
}
const base='教学算例：位置、速度和数值只解释机制，不是设备运行轨迹。可在下方实验改变条件，再对照状态和说明。';
const impl='沿用本课固定源码基线与所选执行路径。图中的位移表示调用、消息或资源归属；不是测得的网络时间。细节和其他分支见下方实验。';
{
 const s=make('f-tokens','分词与查表','Tokenization 与 Embedding：一句话怎样进入模型？','模型做的是数值运算。先把文字切成词表里的片段，再用编号查出向量；编号大小不表示词的含义。',[['文字片段','我想学习人工智能。'],['词表编号','用 tokenizer 的词表查编号'],['模型输入','用模型的 embedding 表查向量']],'前四个 ID 沿用本课固定 Qwen2.5 分词样例；右侧向量是示意，不是模型权重。不同 tokenizer 可能切出不同片段。');
 const words=['我想','学习','人工智能','。'],ids=[104100,100134,104455,1773];
 s.actor('sentence','我想学习人工智能。',0,0,'原始文本');
 words.forEach((w,i)=>s.actor('piece-'+i,w,0,i,'片段 '+(i+1),i));ids.forEach((id,i)=>s.actor('id-'+i,String(id),0,i,'词表编号',i));words.forEach((w,i)=>s.actor('vector-'+i,'[ … 数值 … ]',1,i,'片段 '+(i+1)+' 的向量',i));
 s.frame('先看原句','这是一段文字，不是向量，也不是 token ID。分词器要先按它的规则找出片段。',Object.fromEntries([['sentence',{opacity:1}],...words.map((_,i)=>['piece-'+i,{opacity:0,...pos(0,0)}]),...ids.map((_,i)=>['id-'+i,{opacity:0}]),...words.map((_,i)=>['vector-'+i,{opacity:0}])]));
 s.frame('切出词表片段','这组固定 tokenizer 把原句切成四个片段。一个 token 不一定是一个字，也不一定是一个完整词。',Object.fromEntries([['sentence',{opacity:0}],...words.map((_,i)=>['piece-'+i,{opacity:1,...pos(0,i)}])]));
 s.frame('查到编号并排成序列','为每个片段查到词表编号，保持原顺序进入输入序列。编号大小不表示词的含义；文字留在左边供对照。',Object.fromEntries(ids.map((_,i)=>['id-'+i,{opacity:1,...pos(1,i)}])));
 s.frame('按 ID 查 embedding','模型用编号查自己的 embedding 表，得到能参与数值运算的向量。右移的是查出的向量，不是把汉字外形变成矩阵。',Object.fromEntries(words.map((_,i)=>['vector-'+i,{opacity:1,...pos(2,i)}])));
 s.frame('向量才进入后续计算','后续层才能对这些位置的向量做矩阵运算、混合信息。分词完成不等于已经理解或生成回答。').done();
}
{
 const s=make('f-generation','生成两次','自回归生成：输出 token 怎样成为下一轮输入？','跟着 y1 看一次完整的生成循环：预测出它、交给读者、再把它作为下一轮输入。',[['本轮输入','已有上下文与新位置'],['模型计算','先读输入，再预测下一个'],['输出与历史','输出 token 和 KV 分别记录']],base);
 s.actor('prompt','输入：我想',0,0,'已有的两个输入位置',0).actor('y1','y1：学习',1,0,'新预测的 token',1).actor('kv','历史 KV',1,2,'保存已经计算过的位置',2).actor('copy-y1','y1 的输入副本',2,0,'同一 token ID 的再次输入',1).actor('y2','y2：。',1,1,'下一轮的预测',3);
 s.frame('只有输入，没有回答','先处理“我想”的输入位置。此时 y1 还不存在。',{y1:{opacity:0},'copy-y1':{opacity:0},y2:{opacity:0},kv:{count:0}})
 .frame('读完输入，留下历史','输入进入模型；本轮写入两个位置的 KV。随后才得到下一个位置的预测。',{prompt:pos(1,0),kv:{count:2}})
 .frame('预测 y1','模型得到分数，再按采样规则选出 y1。新输出的 y1 还没有自己的 KV。',{prompt:{opacity:0},y1:{opacity:1}})
 .frame('把 y1 交给读者','y1 移到输出区。读者已经看到“学习”，但模型还没有把“学习”作为输入算过。',{y1:pos(2,0)})
 .frame('准备下一轮输入','保留输出区里的 y1，同时准备相同 ID 的输入副本；虚线不是把客户端文字收回。',{'copy-y1':{opacity:1,...pos(0,1)}})
 .frame('输入 y1，更新历史','这一轮只输入新位置 y1，并读取已有 KV；历史多一个位置。',{'copy-y1':pos(1,1),kv:{count:3}})
 .frame('再预测 y2','本轮得到 y2。停止时最后一个输出不一定已经作为输入写进 KV。',{'copy-y1':{opacity:0},y2:{opacity:1,...pos(2,1)}}).done();
}
{
 const s=make('f-attention','只看当前及过去','Attention 怎样把历史信息带过来？','用两个历史位置观察：当前问题 Q 与各位置 K 比较，得到权重，再用权重混合 V。',[['历史位置','每个位置保留 K 与 V'],['当前查询','Q 与 K 比较，再归一化'],['混合结果','按权重取各位置的 V']],'二维教学算例：Q=[1,0]，分数省略缩放项，softmax([1,0])≈[0.73,0.27]。这不是 DeepSeek-V3 的真实参数或完整 Attention 实现。');
 s.actor('k1','K1=[1,0]',0,0,'位置 1',0).actor('k2','K2=[0,1]',0,1,'位置 2',1).actor('q','Q=[1,0]',1,2,'当前查询；不读未来',2).actor('v1','V1=[2,0]',0,2,'位置 1 的内容',0).actor('v2','V2=[0,2]',0,3,'位置 2 的内容',1).actor('part1','0.73 × V1',1,0,'贡献 [1.46,0]',0).actor('part2','0.27 × V2',1,1,'贡献 [0,0.54]',1).actor('sum','结果 [1.46,0.54]',2,2,'两份贡献相加',3);
 s.frame('先分清 Q、K、V','K 用来比较，V 是要混合的内容；Q 表示当前要向历史查询什么。',{part1:{opacity:0},part2:{opacity:0},sum:{opacity:0}})
 .frame('比较位置 1','Q 访问第一个 K，点积得到分数 1。K 和 V 留在历史中，不被消耗。',{q:pos(1,0)})
 .frame('比较位置 2','Q 访问第二个 K，得到分数 0；因果约束不允许再向未来的位置查询。',{q:pos(1,1)})
 .frame('把分数变成权重','两个分数一起归一化，得到约 0.73 与 0.27。权重之和为 1。',{q:{...pos(1,2),label:'权重 [0.73,0.27]',note:'softmax 把分数归一化'}})
 .frame('形成两份加权贡献','从各 V 计算出的贡献进入右侧；这表示数学依赖，不表示两个历史缓存搬走。',{part1:{opacity:1,...pos(2,0)},part2:{opacity:1,...pos(2,1)}})
 .frame('相加得到当前表示','两份贡献相加，当前位置拿到混合了历史信息的新表示。',{sum:{opacity:1,count:2}}).done();
}
{
 const s=make('f-cache','保留历史','KV Cache：为什么不用每次重算历史 K/V？','让旧位置留在缓存里，新位置进入计算区；观察第二轮到底新增了什么。',[['输入位置','先处理已有上下文'],['计算区','本轮新增的工作'],['KV 缓存','旧位置保留，新位置追加']],base+' 本图采用通用逐位置 KV 示意，不代表 MLA 的具体存储布局。');
 s.actor('old','p1、p2',0,0,'已有上下文',0).actor('kv','p1、p2 的 KV',1,2,'本轮计算的历史表示',0).actor('new','新位置 y1',0,1,'下一轮的输入',1).actor('newkv','y1 的 KV',1,1,'只新增这一位置',1);
 s.frame('第一轮开始','没有缓存时，先计算输入位置。',{new:{opacity:0},kv:{opacity:0},newkv:{opacity:0}})
 .frame('计算已有位置','p1、p2 一起进入计算，生成对应历史表示。',{old:pos(1,0),kv:{opacity:1,count:2}})
 .frame('保留下来','历史 KV 写进缓存；输入卡退出计算区，缓存中的两个位置仍保留。',{old:{opacity:0},kv:pos(2,0)})
 .frame('下一轮只送新位置','y1 进入计算区。Attention 还要读取缓存中的历史，缓存减少的是历史 K/V 的重复计算。',{new:{opacity:1,...pos(1,1)}})
 .frame('计算新位置的 KV','y1 经过这一轮计算后，才得到自己的 KV。旧的 KV 不用再做一遍。',{newkv:{opacity:1,count:1},new:{opacity:0}})
 .frame('追加，而不是覆盖','新增位置写进缓存；p1、p2 的记录仍在。生成更长时，历史占用也会继续增长。',{newkv:pos(2,1)}).done();
}
for(const continuous of [true,false]){
 const s=make('f-batching',continuous?'连续补位':'固定批次',continuous?'Continuous Batching（连续批处理）：轮次间换成员':'Static Batching（固定批次）：等原批成员全部结束','A 还要生成 1 个 token，B 还要 3 个，C 还要 1 个。与图片组批例子分开：这里是多轮文字生成，最多同时 2 条。先预测 A 完成后 C 必须等 B 吗？两种生成策略使用相同请求、到达条件与输出长度。',[['等待队列','C 在第一轮期间准备好参与 Decode'],['本轮执行','最多同时容纳两条请求'],['已完成','退出后不再占执行位置']],'方块是 token，不是 KV 页。浅色两格表示进入本例前已有的文本，实色编号表示本例新增输出。所有请求均已准备好参与普通 Decode，省略 Prefill；“两个位置”是教学容量，不是 GPU 的两个物理核心。补位发生在轮次之间，且假设资源与准入条件都满足。轮数不能直接换算真实吞吐。');
 s.actor('A','请求 A',0,0,'还需 1 个输出',0).actor('B','请求 B',0,1,'还需 3 个输出',1).actor('C','请求 C',0,2,'还需 1 个输出',2);
 s.frame('两条请求等待','先接纳 A 和 B；C 稍后到达。',{C:{opacity:0}})
 .frame('组成第一批','A 和 B 从等待区进入两个执行位置。',{A:pos(1,0),B:pos(1,1)})
 .frame('各生成一个位置','A、B 同时长出新 token；这一轮结束时各新增一个，C 也进入等待队列。',{A:{count:1,note:'已达到本例结束条件'},B:{count:1,note:'还需 2 个输出'},C:{opacity:1}})
 .frame('A 完成退出','A 移出执行区，留下空位。B 仍留在原位置，它的历史与身份没有重置。',{A:pos(2,0)});
 if(continuous)s.frame('C 补进空位','下一轮接纳 C，同时保留 B。不是等到整批都结束才换一批。',{C:pos(1,0)});
 else s.frame('固定批次暂不补位','B 尚未结束，这个策略让 C 继续等待；空位本身不会带来新输出。');
 s.frame('再执行一轮',continuous?'B、C 同时长出新 token。本轮结束后，C 达到结束条件，B 还需一个。':'只有 B 新增输出；C 还没进入执行。',{B:{count:2,note:'还需 1 个输出'},...(continuous?{C:{count:1,note:'已达到结束条件'}}:{})});
 if(continuous)s.frame('C 完成退出','把 C 的执行位置归还；B 的第三轮不受影响。',{C:pos(2,2)});
 s.frame('B 生成最后一个位置','B 第三次参与 Decode；本轮结束后，累计新增三个输出。',{B:{count:3,note:'已达到结束条件'}})
 .frame('B 退出','B 也离开执行区。',{B:pos(2,1)});
 if(!continuous)s.frame('现在才接纳 C','旧批次全部结束，C 才能进入空闲位置。',{C:pos(1,0)}).frame('C 生成输出','C 新增一个输出并达到结束条件。',{C:{count:1,note:'已达到结束条件'}}).frame('C 退出','相同请求全部完成；这里只数教学轮次，不据此推算真实吞吐。',{C:pos(2,2)});
 const scene=s.done();
 scene.stepDurations=scene.frames.slice(1).map((f,i)=>i===0?1.3:f.title.includes('退出')?.75:f.title.includes('生成')||f.title.includes('执行')?1.4:1.05);
}
{
 const s=make('f-hardware','搬运与乘加','GPU 访存与乘加：数据到齐后怎样计算？','一次很小的矩阵乘法：x=[2,3]，W 的两行为 [1,2]、[3,4]。',[['显存','保存输入与权重'],['计算单元','就绪后执行乘加'],['结果','写回结果供下一步使用']],base+' 这是数据依赖示意，省略缓存层级、访存合并与计算搬运重叠。');
 s.actor('x','x=[2,3]',0,0,'输入',0).actor('w','W：两行权重',0,1,'[1,2] 与 [3,4]',1).actor('y0','y[0] = 8',1,2,'2×1 + 3×2',2).actor('y1','y[1] = 18',1,3,'2×3 + 3×4',3);
 s.frame('数据仍在显存','结果尚不存在。只画一个“GPU”方框会把搬运与计算混在一起。',{y0:{opacity:0},y1:{opacity:0}})
 .frame('把输入搬到计算侧','输入先到达；还需要相应权重。',{x:pos(1,0)})
 .frame('把所需权重搬过来','两个操作数就绪后，才能做这次乘加。',{w:pos(1,1)})
 .frame('执行乘加','每行权重与同一输入相乘相加，得到两个输出。',{y0:{opacity:1,count:2},y1:{opacity:1,count:2}})
 .frame('写回结果','输出交给后续算子。增加算力不会自动缩短所有搬运等待。',{y0:pos(2,0),y1:pos(2,1)}).done();
}
{
 const s=make('f-parallel','流水线 PP','PP Stage：后一级为什么要等待激活？','让 MB1、MB2 两个微批次经过两级。每张卡表示同一个批次的中间结果，不表示整套模型随它移动。',[['等待输入','两个微批次'],['PP Stage 0','Stage 0'],['PP Stage 1','Stage 1；结束后移到下方']],base+' 两级各用一个教学时隙；时间相等是为了说明依赖，不是硬件测量。');
 s.actor('MB1','微批次 MB1',0,0,'尚未执行 PP Stage 0',0).actor('MB2','微批次 MB2',0,1,'尚未执行 PP Stage 0',1);
 s.frame('前后级有依赖','PP Stage 1 需要 PP Stage 0 产生的中间结果。')
 .frame('MB1 进入前一级','MB2 等待；后一级暂时没有可用输入。',{MB1:pos(1,0)})
 .frame('MB1 做完 Stage 0','MB1 的第一段工作完成，得到中间结果。',{MB1:{count:1,note:'PP Stage 0 的中间结果'}})
 .frame('MB1 交给后一级，MB2 补位','MB1 的中间结果移动到 Stage 1；MB2 同时进入 Stage 0。',{MB1:pos(2,0),MB2:pos(1,0)})
 .frame('两级同时工作','MB1 完成 Stage 1；MB2 完成 Stage 0。两者不是同一请求的两个副本。',{MB1:{count:2,note:'两段都完成'},MB2:{count:1,note:'PP Stage 0 的中间结果'}})
 .frame('释放位置后继续','MB1 离开执行位置；MB2 的中间结果再交给后一级。',{MB1:pos(2,2),MB2:pos(2,0)})
 .frame('MB2 完成','流水线开始和结束时仍会有空闲，批次多也不能取消前后依赖。',{MB2:{count:2,note:'两段都完成',...pos(2,3)}}).done();
}
{
 const s=make('f-communication','AllReduce 求和','AllReduce：各 rank 的贡献怎样求和？','两张卡对相同输出算出不同贡献。求和后，两边都拿到相同的完整结果。',[['Rank 0','本地贡献 [1,2]'],['逻辑求和','这里省略实际通信算法'],['Rank 1','本地贡献 [10,20]']],base+' AllReduce 不要求真实硬件存在中央求和节点；中间区域只表示逻辑运算。');
 s.actor('a','Rank 0 的贡献',0,0,'[1,2]',0).actor('b','Rank 1 的贡献',2,0,'[10,20]',1).actor('a-copy','贡献 0 的副本',0,0,'[1,2]',0).actor('b-copy','贡献 1 的副本',2,0,'[10,20]',1).actor('result0','完整和 [11,22]',1,2,'返回 Rank 0',2).actor('result1','完整和 [11,22]',1,3,'返回 Rank 1',2);
 s.frame('两个局部贡献','只拿自己这一份，还不是完整输出。',{'a-copy':{opacity:0},'b-copy':{opacity:0},result0:{opacity:0},result1:{opacity:0}})
 .frame('交换贡献','送出贡献的副本；原本的局部值留在两端供比较。',{'a-copy':{opacity:1,...pos(1,0)},'b-copy':{opacity:1,...pos(1,1)}})
 .frame('逐元素相加','[1+10,2+20] 得到 [11,22]。不能把它理解成拼接成四个数。',{result0:{opacity:1},result1:{opacity:1}})
 .frame('两边都拿到和','AllReduce 的结果在两个 rank 上相同；接下来的计算才有完整输入。',{result0:pos(0,2),result1:pos(2,2)}).done();
}
{
 const s=make('f-deployment','Prefill / Decode 分离','PD Disaggregation：怎样交接 KV 并继续生成？','Prefill 先读输入并算出第一个输出。Decode 接过可用的历史状态，才继续预测。',[['Prefill 侧','计算输入的历史表示'],['传输中','KV 副本与控制信息分别交接'],['Decode 侧','收到且可用后才能继续']],base+' 分离带来传输与调度开销；不保证一定更快。首 token 的返回位置取决于具体实现。');
 s.actor('kv','输入的 KV',0,1,'保留在发送侧',0).actor('copy','KV 的接收副本',0,1,'不是把源缓存瞬间删除',0).actor('first','首个输出 y1',0,0,'作为 Decode 的新输入',1).actor('next','下个输出 y2',2,2,'Decode 计算结果',2);
 s.frame('先完成 Prefill','Prefill 处理输入，生成历史表示并预测 y1。',{kv:{count:4},copy:{opacity:0},next:{opacity:0}})
 .frame('发起历史状态传输','源端 KV 仍在。传输中的副本还不能当成 Decode 已经可读的历史。',{copy:{opacity:1,...pos(1,1)}})
 .frame('接收侧拿到状态','本例先表示数据到达；实际还要核对完成状态、元数据和资源准备。',{copy:pos(2,1)})
 .frame('交接新输入','在所需条件满足后，Decode 使用 y1 与已有历史继续。',{first:pos(2,0)})
 .frame('生成下一位置','Decode 只处理新增输入位置并追加历史，得到 y2。',{copy:{count:5},next:{opacity:1}}).done();
}
{
 const s=make('f-speculation','接受前两个','Speculative Decoding：候选怎样验证和接受？','草稿模型先提议 a、b、c、d，目标模型验证。这里设定只接受连续前缀 a、b。',[['草稿候选','候选尚未成为正式输出'],['目标模型验证','按原算法检查连续前缀'],['正式输出','只提交被接受的前缀与新选项']],base+' 用确定性线性候选说明边界；不代表所有采样算法都只做相等比较，也不等于真实加速收益。');
 ['a','b','c','d'].forEach((v,i)=>s.actor(v,'候选 '+v,0,i,'第 '+(i+1)+' 个候选',i));s.actor('z','修正项 z',1,2,'来自目标模型',4);
 s.frame('只有候选','候选尚未经过目标模型验证，不能直接当作回答发给用户。',{z:{opacity:0}})
 .frame('候选一起进入验证','验证不是只检查最后一个候选；需要找到从头开始能接受多少。',Object.fromEntries(['a','b','c','d'].map((v,i)=>[v,pos(1,i)])))
 .frame('确认前两个','a、b 被接受；c 开始失败，即使 d 看起来合适，也不能越过断点提交。',{a:{note:'接受',count:1},b:{note:'接受',count:1},c:{note:'拒绝：前缀在这里断开'},d:{note:'不提交：位于断点之后'}})
 .frame('提交连续前缀','a、b 进入正式输出；c、d 留在未提交区域。',{a:pos(2,0),b:pos(2,1)})
 .frame('使用目标模型的修正项','目标模型产生 z，接到已接受前缀之后；不是把草稿 c 改名冒充验证通过。',{z:{opacity:1,...pos(2,2)}})
 .frame('回收未提交候选','c、d 退出，后续从合法上下文继续。',{c:{opacity:0},d:{opacity:0}}).done();
}
{
 const s=make('f-serving','取消请求','Cancellation：断连后请求怎样取消？','连接状态、后端请求和已分配资源是三件事。取消需要送达，再经过安全的收尾边界。',[['客户端 / 接入层','可以先断开连接'],['后端','请求仍有自己的生命周期'],['资源池','分配与归还分别发生']],base+' 这是通用生命周期；具体安全边界、取消回告和缓存策略由实现决定。');
 s.actor('request','请求 R1',1,0,'执行中',0).actor('connection','R1 的连接',0,0,'连接正常',0).actor('resource','R1 的资源',2,0,'仍由请求持有',2).actor('cancel','取消消息 R1',0,1,'请求停止，不是直接删显存',3);
 s.frame('正在执行','服务器已经持有 R1 的资源。',{cancel:{opacity:0},resource:{count:3}})
 .frame('客户端断开','连接结束了，但 R1 和资源仍在后端。',{connection:{note:'已断开',opacity:.5}})
 .frame('发出取消消息','接入层把取消送给后端；在途中暂停时，资源仍然被持有。',{cancel:{opacity:1,...pos(1,1)}})
 .frame('等待安全收尾','后端收到取消，登记待结束状态；这不表示已经抢占了 GPU。',{request:{note:'待安全边界收尾'}})
 .frame('完成请求收尾','到达可释放的边界后，R1 退出执行。',{request:{...pos(1,2),note:'已结束'}})
 .frame('归还资源','这时资源才成为可再分配状态。缓存复用等策略可能继续持有部分 KV。',{resource:{...pos(2,2),note:'已归还分配器',count:0}}).done();
}
{
 const s=make('f-metrics','首包与后续包','TTFT 与 ITL：首字等待和后续间隔有何不同？','跟着请求和两次回程，把发送到首包、首包到后续包分开看。',[['客户端','计时的观察位置'],['服务器','等待与计算都可能在这里'],['返回路径','这里的每个包恰好一个 token']],'本动画的播放秒数不是 TTFT、ITL 或服务耗时。包只用于说明观测点；实际一个网络包可能含多个 token。数值计算见下方实验。');
 s.actor('R','请求 R1',0,0,'发送事件：开始计时',0).actor('p1','第一个数据包',1,1,'含 y1',1).actor('p2','第二个数据包',1,2,'含 y2',2).actor('clock','客户端秒表',0,3,'尚未收到首包',3);
 s.frame('发送请求','TTFT 从本例的发送事件开始计算。服务内部耗时不等于完整客户端等待。',{p1:{opacity:0},p2:{opacity:0}})
 .frame('请求进入服务器','排队、Prefill 等发生在首包到达之前。',{R:pos(1,0)})
 .frame('首包发出','生成 y1 后，数据包开始返回。仅服务器采样结束还不能停止客户端 TTFT。',{p1:{opacity:1,...pos(2,1)}})
 .frame('客户端收到首包','在这个边界记录 TTFT；接下来计后续到达间隔。',{p1:pos(0,1),clock:{note:'首包到达：记录 TTFT'}})
 .frame('第二包返回','后续生成与网络传输共同影响下一次到达。',{p2:{opacity:1,...pos(2,2)}})
 .frame('记录到达间隔','两次收到数据包的时间差是到达间隔；合包时不能当作单 token 的纯计算耗时。',{p2:pos(0,2),clock:{note:'后续包到达：记录间隔'}}).done();
}
{
 const s=make('system-overview','普通请求回程','DetokenizerManager 与 IPC：token ID 怎样返回为文字？','同一个 rid=R1 关联多个进程里的对象；跨进程传的是消息，不是把整个 Python 对象搬过去。',[['TokenizerManager','前端状态与响应连接'],['Scheduler / Worker','调度、执行与采样'],['DetokenizerManager','ID 转成文字，再交回前端']],impl+' 选择普通 Python HTTP、单 tokenizer、合并部署；省略输出与后续计算的重叠。');
 s.actor('front','ReqState · R1',0,0,'保存返回连接',0).actor('input','Tokenized 请求',0,1,'rid=R1；输入 ID',1).actor('req','Req · R1',1,0,'调度侧的另一个对象',0).actor('out','BatchTokenIDOutput',1,2,'R1 → y1 的 ID',2).actor('str','BatchStrOutput',2,2,'R1 → 文字片段',2);
 s.frame('前端登记请求','ReqState 留在前端；尚未把响应连接交给 GPU。',{req:{opacity:0},out:{opacity:0},str:{opacity:0}})
 .frame('派发输入消息','分词后的消息跨 IPC 到调度侧；前端状态仍在等待回程。',{input:pos(1,1)})
 .frame('建立调度对象','Scheduler 接收消息后建立 Req。相同 rid 不代表它和 ReqState 是同一内存对象。',{req:{opacity:1},input:{opacity:.35}})
 .frame('模型执行并采样','本例执行后得到 y1 的 token ID。这个 ID 本身不是已经送到客户端的文字。',{out:{opacity:1,count:1}})
 .frame('ID 消息进入反分词','带 rid 的输出消息进入 DetokenizerManager，转换成文字片段。',{out:pos(2,1)})
 .frame('准备文字输出','产生新的文本输出消息；原 ID 消息与文本消息分开表示。',{str:{opacity:1}})
 .frame('前端按 rid 找到连接','文本消息返回 TokenizerManager，关联 ReqState 后才可返回给请求方。',{str:pos(0,2),front:{note:'收到 R1 的输出，可返回响应'}}).done();
}
{
 const s=make('request-runtime','完成后缓存保留','Req 与 RadixCache：请求结束后怎样管理 KV？','沿一次普通请求的收尾看：请求行归还和有效 KV 转入前缀缓存是两个动作。',[['Scheduler','Req 与请求槽位'],['模型执行使用','本请求有效的 KV'],['Radix Cache / 分配器','缓存保留与请求行回收分开']],impl+' 采用原课 cache=true、page_size=1、无其他持有者的普通 Radix Cache 示例。');
 s.actor('req','Req · R1',0,0,'已有 3 个采样输出',0).actor('row','请求行 R1',0,1,'仍被请求占用',1).actor('kv','有效 KV：6 个位置',1,0,'4 个输入 + y1、y2',2).actor('last','最后输出 y3',1,2,'尚未作为下轮输入',3);
 s.frame('最后一轮计算完成','本例输入长 4、输出 3：只有前两个输出已经再次输入，因此有效 KV 是 6 个位置。',{kv:{count:6},last:{count:1}})
 .frame('记录结束原因','Req 记录长度结束。结束原因本身不会让所有缓存内容立刻消失。',{req:{note:'finished_reason = length'}})
 .frame('有效 KV 交给缓存管理','图中位移表示管理归属变化，不表示这 6 份 KV 被物理复制一遍。',{kv:{...pos(2,0),note:'缓存仍占显存，可被后续请求复用'}})
 .frame('请求行归还','请求槽位可以重新使用；这和缓存是否继续保留是独立资源边界。',{row:{...pos(2,2),note:'请求行已回到空闲池'}})
 .frame('请求对象收尾','请求结束，但缓存中的 6 个位置仍在。是否后续驱逐，需要另看缓存策略。',{req:{opacity:.35,note:'已完成资源收尾'}}).done();
}
{
 const s=make('model-execution','DeepSeek-V3 · Eager MLA','DeepSeek-V3 MLA：从执行输入到历史读取','沿选定的 DeepSeek-V3 Decode 路径看数据身份变化：batch 输入、查询与压缩状态、Attention 结果和采样结果不是同一种对象。',[['Worker / Runner','把 batch 整理成执行输入'],['DeepSeek-V3 模型层','Embedding、投影与其他模型计算'],['MLA backend / kernel','按地址写状态、读取历史、计算']],impl+' 使用本课 Eager / Triton MLA 路径。deepseek_v2.py 同时承载 V3；省略层间重复与 MoE 展开，不把单个 Attention 输出当最终回答。');
 s.actor('batch','执行输入 B0',0,0,'token ID、位置、KV 槽位',0).actor('meta','Attention 元数据',0,2,'历史长度与地址索引',1).actor('hidden','当前层 hidden',1,0,'Embedding / 前序层的输出',2).actor('q','查询 Q',1,1,'由当前 hidden 投影得到',3).actor('latent','压缩 KV + 位置分量',1,2,'本轮 MLA 状态',4).actor('attn','Attention 结果',2,0,'中间张量，不是回答',2).actor('out','logits → 新 token',1,3,'模型输出头后再采样',0);
 s.frame('准备 ForwardBatch','Worker / Runner 整理执行输入。还需要为这轮 Attention 准备索引等元数据。',{hidden:{opacity:0},q:{opacity:0},latent:{opacity:0},attn:{opacity:0},out:{opacity:0}})
 .frame('先准备历史地址','Triton backend 根据本轮信息准备历史地址和长度。元数据不是 KV 内容本身。',{meta:pos(2,3)})
 .frame('输入交给模型','token ID 与位置进入模型入口；模型把输入变为 hidden 表示。',{batch:pos(1,0)})
 .frame('产生当前层表示','这张卡是新计算的 hidden，不能继续把它叫作请求消息。',{batch:{opacity:0},hidden:{opacity:1}})
 .frame('投影得到 MLA 输入','当前 hidden 产生查询 Q、压缩 KV 及位置相关分量。三者承担不同职责。',{q:{opacity:1},latent:{opacity:1}})
 .frame('交给 Attention 实现','Q 用于当前查询；新压缩状态按本轮槽位写入缓存，结合历史地址做 Attention。',{q:pos(2,1),latent:pos(2,2)})
 .frame('kernel 返回中间结果','Attention kernel 使用历史与当前状态，返回中间张量。它不会直接生成整段回答。',{attn:{opacity:1}})
 .frame('模型继续计算','Attention 结果回到模型层，继续后续网络计算；图只展开其中一层的交接。',{attn:pos(1,0),hidden:{opacity:0}})
 .frame('输出头之后再采样','模型产生 logits 后，Worker / Runner 的采样路径得到 next_token_ids 并返回调度侧。',{out:{opacity:1,...pos(0,3)}}).done();
 const g=make('model-execution','CUDA Graph · 3 补到 4','CUDA Graph Padding：图桶里的补齐行是什么？','图重放使用固定形状。这里把 3 条真实请求装进容量 4 的桶，再把补齐结果裁掉。',[['真实输入','只有 R1、R2、R3'],['容量 4 的图桶','第四行仅为形状补齐'],['真实输出','只保留三个真实位置']],impl+' 采用本课 graph(3, ready)；图已捕获且其他资格满足，不展开捕获过程。');
 for(let i=1;i<=3;i++)g.actor('R'+i,'真实请求 R'+i,0,i-1,'一个 Decode 输入位置',i-1);
 g.actor('pad','补齐行 PAD',1,3,'不属于任何真实请求',4);
 g.frame('先数真实请求','真实 batch size 是 3。',{pad:{opacity:0}})
 .frame('填入已捕获的桶','真实输入填入图桶；保持 R1、R2、R3 的映射顺序。',{R1:pos(1,0),R2:pos(1,1),R3:pos(1,2)})
 .frame('补齐到固定形状','添加 PAD 行，使本例形状满足容量 4；没有新用户请求被创建。',{pad:{opacity:1}})
 .frame('重放已捕获计算','图按固定形状执行。这里只用小格表示已经执行，不拿播放时长推算重放耗时。',{R1:{count:1},R2:{count:1},R3:{count:1},pad:{count:1}})
 .frame('裁回真实输出','仅三个真实位置的 logits 交给后续处理，PAD 留在图桶侧。',{R1:{...pos(2,0),note:'R1 的 logits'},R2:{...pos(2,1),note:'R2 的 logits'},R3:{...pos(2,2),note:'R3 的 logits'},pad:{note:'补齐输出被裁掉',opacity:.35}}).done();
}
{
 const s=make('kv-memory','从 6 个位置扩到 11','Paged KV Allocation：先补尾页，再申请新页','页大小为 4。已有地址 [4,5,6,7,28,29]，新增位置先落到 30、31，再使用 12、13、14。',[['本轮新 KV','逻辑位置 p6…p10'],['分配与写入','保留地址不等于数据已写入'],['物理页','页编号固定，不因逻辑顺序搬家']],impl+' 采用本课普通 paged allocator 与 MHA KV pool 示例；该存储图不是 DeepSeek-V3 MLA 物理布局。');
 s.actor('page1','页 1 · 地址 4…7',2,0,'4 个位置已写入',0).actor('page7','页 7 · 地址 28…31',2,1,'前 2 个已写入，尾部有空位',1).actor('page3','页 3 · 地址 12…15',2,2,'空闲页',2).actor('tail','新位置 p6、p7',0,1,'将写到地址 30、31',1).actor('rest','新位置 p8…p10',0,2,'将写到地址 12、13、14',2);
 s.frame('已有 6 个位置','页 1 满了，页 7 还剩两个位置。逻辑相邻不要求物理页编号连续。',{page1:{count:4},page7:{count:2}})
 .frame('分配新增地址','先预留页 7 尾部，再拿到页 3。这里尚未执行 KV 写入。',{page7:{note:'28、29 已写；30、31 预留'},page3:{note:'12…14 预留；还没有写入'},tail:pos(1,1),rest:pos(1,2)})
 .frame('先写尾页空位','本轮 p6、p7 的 KV 写进 30、31；页 7 的占用从 2 增到 4。',{tail:{...pos(2,1),opacity:0},page7:{count:4,note:'四个位置都已写入'}})
 .frame('尾页内容留在原地址','将已写入的数据卡收进页内，继续看剩余三个新位置。',{tail:{opacity:0}})
 .frame('再写新页','p8…p10 写进页 3 的前三格。页 1、页 7 没有因此重排或复制。',{rest:{...pos(2,2),opacity:0},page3:{count:3,note:'12…14 已写；15 仍空'}})
 .frame('得到完整逻辑映射','最终 11 个位置分别落到三页。读取时沿映射找地址，不能直接拿位置号当物理地址。',{rest:{opacity:0}}).done();
}
// Admission frames are driven by the existing source-grounded model, not a second policy implementation.
if(root.SchedulingModel){for(const [policy,scenario,name]of [['lpm','compute','前缀优先 · 输入预算'],['fcfs','exact','KV 严格边界']]){
 const events=root.SchedulingModel.admission(policy,scenario),s=make('scheduling',name,'Admission Control：排序之后怎样检查准入？','把候选从等待队列拿出来检查预算；通过后才加入 can_run_list。检查失败的请求仍留待后续处理。',[['waiting_queue','排序只改变检查顺序'],['临时检查','缓存保护与剩余预算'],['can_run_list','只有提交成功的请求']],impl+' 动作和数值直接取自本课 SchedulingModel.admission；卡片位移是调度集合变化。');
 root.SchedulingModel.requests.forEach((r,i)=>s.actor(r.id,r.id+' · 输入 '+r.input,0,i,'前缀命中 '+r.hit+' 个位置',i));
 for(const e of events){const patches={};for(const id of e.order){const row=root.SchedulingModel.requests.findIndex(r=>r.id===id),admitted=e.admitted.indexOf(id);patches[id]={...pos(admitted>=0?2:0,admitted>=0?admitted:row),note:admitted>=0?'已提交到本轮':`检查次序 ${e.order.indexOf(id)+1} · 等待`};}if(e.kind==='check')patches[e.current]={...pos(1,1),note:'临时检查，不算正式接纳'};
  const titles={waiting:'请求在等待',sorted:'按策略排序',check:'检查 '+e.current,commit:'接纳 '+e.current,blocked:'检查未通过',stop:'本轮停止继续接纳',ready:'交出本轮可运行集合'};
  let caption=e.kind==='sorted'?'策略改变检查次序（标在卡片上）；卡片保留原位方便对照，can_run_list 仍未增加请求。':e.kind==='check'?`${e.current} 进入临时检查；本例可用 KV 预算为 ${e.checkAvailable}，检查期间的前缀保护不是最终接纳。`:e.kind==='commit'?`${e.current} 通过预算检查后移入 can_run_list；累计新增输入 ${e.usedInput}。`:e.kind==='blocked'?`因 ${e.reason==='kv'?'KV 预算严格边界':e.reason==='rows'?'请求行容量':'输入预算'} 未通过而停止；检查对象回到等待侧，不能画成正在 GPU 执行。`:e.kind==='ready'?`本轮接纳 ${e.admitted.join('、')||'零条请求'}；其余仍待后续调度。`:'每条请求只属于一个位置；下一步会改变集合或资源约束。';
  s.frame(titles[e.kind],caption,patches);
 }s.done();
}}
{
 const s=make('parallelism','RowParallelLinear','RowParallelLinear 与 AllReduce：局部贡献怎样合并？','同一输入按维度分开。Rank 0 算出 [3,5]，Rank 1 算出 [7,25]；输出需要逐元素求和。',[['Rank 0','输入分片 [1,2]'],['通信接口','逻辑 AllReduce；不代表中转设备'],['Rank 1','输入分片 [3,4]']],impl+' 复用本课 tensor(row) 算例；忽略 bias。权重维度切分与图上的左右位置无关。');
 s.actor('a','局部结果 [3,5]',0,0,'RowParallelLinear 的贡献',0).actor('b','局部结果 [7,25]',2,0,'RowParallelLinear 的贡献',1).actor('ac','Rank 0 贡献副本',0,0,'[3,5]',0).actor('bc','Rank 1 贡献副本',2,0,'[7,25]',1).actor('ra','归约结果 [10,30]',1,2,'Rank 0 继续后续计算',2).actor('rb','归约结果 [10,30]',1,3,'Rank 1 继续后续计算',2);
 s.frame('各算局部贡献','这两个向量是同一输出的不同贡献，不应把它们拼接为四个数。',{ac:{opacity:0},bc:{opacity:0},ra:{opacity:0},rb:{opacity:0}})
 .frame('交给归约','当该路径需要归约时，各 rank 的贡献交给通信接口。',{ac:{opacity:1,...pos(1,0)},bc:{opacity:1,...pos(1,1)}})
 .frame('求出完整输出','3+7=10，5+25=30。示意的是接口语义，实际 collective 可以有不同算法。',{ra:{opacity:1},rb:{opacity:1}})
 .frame('两边拿到同样结果','归约结果供各 rank 继续；这不是 DP 把不同请求分给不同副本。',{ra:pos(0,2),rb:pos(2,2)}).done();
}
{
 const s=make('communication','KV 页地址映射','KV Transfer：源页怎样映射到目标页？','源页 7、8、12 对应目标页 3、4、9。传输的是内容；不能把源端页号直接当作目标地址。',[['源端页','页 7=A，8=B，12=C'],['传输任务','源地址与目标地址成对出现'],['目标端页','页 3、4、9 接收副本']],impl+' 采用本课 pages(false)：单源、单目标、一个 K buffer；此图完成接收不代表所有 ranks 和元数据条件均满足。');
 ['A','B','C'].forEach((v,i)=>s.actor('src-'+v,'源页 '+[7,8,12][i],0,i,'内容 '+v,i));['A','B','C'].forEach((v,i)=>s.actor('dst-'+v,'内容 '+v+' 的副本',0,i,'目标页 '+[3,4,9][i],i));
 s.frame('先核对地址对应','源端内容保留；目标端尚未收到副本。',Object.fromEntries(['A','B','C'].map(v=>['dst-'+v,{opacity:0}])))
 .frame('建立传输段','7→3 与 8→4 两端都连续，可合并为一个段；12→9 另成一段。',{ 'dst-A':{opacity:1,...pos(1,0)},'dst-B':{opacity:1,...pos(1,1)},'dst-C':{opacity:1,...pos(1,2)}})
 .frame('第一段写到目标','A、B 分别写到页 3、4。源页 7、8 仍在源端，不在目标端被“重命名”。',{'dst-A':{...pos(2,0),count:1},'dst-B':{...pos(2,1),count:1}})
 .frame('第二段独立写入','C 写到页 9。页内容到达和请求可进入 Decode 是不同条件。',{'dst-C':{...pos(2,2),count:1}}).done();
}
{
 const s=make('advanced-generation','输出与 KV 提交','EAGLE：接受候选后怎样提交输出与 KV？','沿本课 EAGLE 线性验证算例：root=r，候选 a、b、c、d；接受两个后，输出 [a,b,z]，有效 KV 是 [r,a,b]。',[['验证临时区','保留 root 与所有候选的状态'],['接受边界','只允许连续前缀提交'],['正式状态 / 输出','两者相差最后的新采样项']],impl+' 直接对应本课 speculate(2)；线性候选为教学简化，不展示完整 EAGLE tree 或概率采样。');
 s.actor('kv','验证 KV',0,0,'r、a、b、c、d',0).actor('prefix','接受的前缀 a、b',0,1,'尚未提交',1).actor('z','新采样项 z',1,2,'目标模型在断点选择',2).actor('valid','有效 KV',1,0,'r、a、b',0).actor('out','正式输出',1,1,'a、b',1);
 s.frame('验证产生临时状态','一次验证计算涉及 root 与候选，但临时存在不等于全部都可提交。',{kv:{count:5},prefix:{count:2},z:{opacity:0},valid:{opacity:0},out:{opacity:0}})
 .frame('确定接受边界','仅 a、b 被接受；c、d 后的状态不属于新的合法历史。',{prefix:pos(1,1)})
 .frame('提交有效 KV','从验证区提交 r、a、b，未接受后缀不进入有效 KV。',{valid:{opacity:1,...pos(2,0),count:3},kv:{note:'c、d 未提交，等待回收'}})
 .frame('提交已接受输出','输出前缀为 a、b；root=r 已是旧输入，不再当新输出发一次。',{out:{opacity:1,...pos(2,1),count:2},prefix:{opacity:0}})
 .frame('选出额外的新项 z','z 是新的输出，还没有作为下一轮输入计算，所以没有 z 的 KV。',{z:{opacity:1}})
 .frame('追加 z 到输出','现在输出是 a、b、z，KV 仍为 r、a、b。不要把两边小格都机械增加到同一内容。',{z:pos(2,2),out:{label:'输出前缀 a、b',note:'下面的 z 接在后面'},kv:{opacity:.35,note:'未提交后缀回收'}}).done();
}
{
 const s=make('serving-operations','普通等待请求断连','AbortReq 发出后，哪一侧还没结束？','使用本课普通等待请求路径：尚未执行，没有预分配 GPU KV。前端与 Scheduler 各自收尾。',[['TokenizerManager','ReqState 与连接'],['IPC 消息','取消去程与处理回告'],['Scheduler','waiting_queue 中的 Req']],impl+' 对应 cancellation(disconnect)。不可套用到 PD Decode 已预分配资源的等待队列，也不表示断开的客户端会收到回告。');
 s.actor('front','ReqState · R1',0,0,'等待结果',0).actor('req','Req · R1',2,0,'在 waiting_queue',0).actor('abort','AbortReq · R1',0,1,'取消控制消息',1).actor('echo','取消回告 · R1',2,2,'回告前端，不是客户端 ACK',2);
 s.frame('前后端各有状态','相同 rid 关联两个对象。此例尚未执行，因此没有 GPU KV 需要释放。',{abort:{opacity:0},echo:{opacity:0}})
 .frame('客户端连接断开','连接断开不等于 waiting_queue 已移除 R1。',{front:{note:'连接已断；后端状态未清理'}})
 .frame('取消经过 IPC','abort_sent 表示派发，消息在途中时 Scheduler 仍保留请求。',{abort:{opacity:1,...pos(1,1)}})
 .frame('Scheduler 收到取消','后端处理 AbortReq；这才开始移除等待中的 R1。',{abort:pos(2,1)})
 .frame('移出等待队列','Req 从等待队列退出，再回告前端。本例没有抢占 GPU 的动作。',{req:{...pos(2,3),note:'已移出 waiting_queue'},echo:{opacity:1}})
 .frame('回告返回前端','消息返回 TokenizerManager，连接断开不妨碍服务内部收尾。',{echo:pos(0,2)})
 .frame('前端清理状态','前端结束自己的等待与状态；后端移除和前端收尾是两个时间边界。',{front:{note:'ReqState 已清理',opacity:.4}}).done();
}
{
 const s=make('performance','客户端许可与 TTFT','TTFT 计时边界：并发许可等待是否计入？','按本课 native /generate benchmark 路径：并发许可外的等待与单请求计时不是同一个区间。',[['等客户端许可','limited_request_func'],['请求计时区间','async_request_sglang_generate'],['首包与结束','按 benchmark 的观测事件记账']],'沿用本课 PerformanceModel.timeline(base) 的教学数值。80、20、80、100、20 ms 仅用于解释统计边界；动画进度与毫秒无比例关系。');
 s.actor('R','请求 R1',0,0,'许可外等待 80 ms',0).actor('watch','单请求秒表',1,2,'尚未开始',1).actor('first','首包事件',2,0,'首包收到才记录 TTFT',2);
 s.frame('还没有获得许可','总实验已经开始，但这个请求仍在客户端等待并发许可。单请求秒表尚未开始。',{first:{opacity:0}})
 .frame('拿到许可，开始请求计时','请求离开许可队列进入 adapter；这里记录 st。许可外等待没有被算进本例 TTFT。',{R:pos(1,0),watch:{note:'开始：st = perf_counter()'}})
 .frame('传输与服务器等待','本例传入 20 ms、Scheduler 排队 80 ms；这些在请求计时区间内。',{R:{note:'传入 20 + 排队 80 ms',count:2}})
 .frame('Prefill 与首包回传','再经历教学设定的 Prefill 100 ms、首包回传 20 ms；直到客户端收到首包。',{R:{note:'Prefill 100 + 回传 20 ms',count:4},first:{opacity:1}})
 .frame('记下 TTFT=220 ms','20+80+100+20=220 ms。若只把许可外等待多加 120 ms，本例 TTFT 不变，但总实验时间可能增加。',{watch:{...pos(2,2),note:'TTFT = 220 ms'}}).done();
}
if(root.PDQueues){for(const [kind,name]of [['normal','普通最终块'],['chunks','分块与共享引用']]){
 const s=make('dataflow',name,'Req 留在本级，哪些容器在引用它？','中间下方始终是同一个 PP0 本地 Req。小牌与生长、收回的连线表示引用的建立和解除；没有复制出多个请求。',[['准入与等待','bootstrap / waiting'],['选批与续算','batch / chunked_req'],['传输与退出','inflight / 教学出口']],'保留本课 882577451e 固定基线。快照直接来自 PDQueues.create；只观察 PP0 本地引用，不把各 PP 级的 Req 当成同一个对象。');
 const refs={bootstrap:[0,0,'bootstrap 引用'],waiting:[0,1,'waiting 引用'],batch:[1,0,'batch 引用'],chunk:[1,1,'chunked_req 引用'],inflight:[2,0,'inflight 引用'],exit:[2,2,'教学出口标记']};
 s.actor('Req','PP0 本地 Req R0',1,3,'只有这一个请求对象',0);
 for(const[id,[c,r,label]]of Object.entries(refs))s.actor(id,label,1,3,'指向同一个 Req R0',id==='chunk'?2:1);
 for(const f of root.PDQueues.create(kind,0).frames){const changes={Req:{note:f.after.finished+' · KV '+(f.after.kv?'仍有本地持有':'无本地持有')}};for(const[id,[c,r]]of Object.entries(refs))changes[id]={...(f.after[id]?pos(c,r):pos(1,3)),opacity:f.after[id]?1:0};s.frame(f.title,f.detail,changes);}
 const definition=s.done();definition.evidence='#communication-bridge';definition.legend='固定 Req = 本地对象 · 小牌 = 容器引用 · 实线 = 指向同一 Req';definition.links=Object.keys(refs).filter(id=>id!=='exit').map(id=>({from:'Req',to:id}));
}}
for(const wait of [true,false]){
 const s=make('lifecycle',wait?'release 后本地仍在途':'release 后本地已完成','名单到了，为什么本级还不能释放？','本例只放大 Prefill 的一个流水级：传输收尾名单到达后，还要复查本级 sender 的状态。',[['控制名单','来自跨级的终态判断'],['本级 sender 与资源','PP0 的本地复查'],['本级收尾','按 Success / Failed 分支清理']],'保留本课 279339f113 固定基线。省略跨级名单汇总细节；本地资源收尾不表示 Decode 已可运行，也不表示物理缓存全部清零。');
 s.actor('release','release 名单：R0',0,0,'是先前采样的终态判断',0).actor('resource','R0 的本地持有',1,1,'KV / metadata 待收尾',1).actor('sender','本级 sender',1,0,'等待复查',2);
 s.frame('收到候选名单','不要只看到名单里的 R0，就立刻把本级资源画成空闲。',{resource:{count:3}})
 .frame('把名单交给本地处理','本级进入 process_disagg_prefill_inflight_queue，再检查 sender 当前状态。',{release:pos(1,2)})
 .frame('复查本地 poll',wait?'本地仍为 Transferring，资源保持；R0 继续留在 inflight。':'本地已为 Success，可以进入成功收尾分支。',{sender:{note:wait?'Transferring：仍在途':'Success：本地成功'}});
 if(wait)s.frame('等待条件改变','没有释放、没有重新入队；本地传输结束后，后续调度轮次再次检查。').frame('后续复查达到终态','现在本例本地为 Success，才继续收尾。',{sender:{note:'Success：本地成功'}});
 s.frame('本级执行收尾','本地持有关系解除并回收相应资源；图中标记的是持有状态，不表示物理缓存全部清零。',{resource:{...pos(2,1),note:'本级资源已完成交接',count:0},sender:{note:'本级处理完成'}});
 const def=s.done();def.evidence='#baseline';
}
// Reject broken snapshots early; this also makes standalone model checks useful.
for(const variants of Object.values(scenes))for(const s of variants){
 const ids=s.actors.map(a=>a.id);if(new Set(ids).size!==ids.length)throw Error('Duplicate actor identity');
 for(const f of s.frames)if(f.actors.length!==ids.length||f.actors.some(a=>!Number.isFinite(a.x+a.y+a.opacity+a.count)))throw Error('Invalid mechanism frame');
}
if(typeof module==='object'&&module.exports)module.exports=scenes;root.MechanismScenes=scenes;
})(typeof globalThis==='object'?globalThis:this);
