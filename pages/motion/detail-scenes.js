/* Small mechanisms derived from the course models, with explicit teaching bounds. */
(function(root){
'use strict';
function create(F,C,G,R,O){
 const out={},make=(name,title,intro,boundary,steps,data,kind)=>({name,title,intro,boundary,data,kind,frames:steps.map(([title,caption])=>({title,caption}))});
 const scope='数值与动画时间是教学设定，不是模型预测、GPU 或网络测量。';
 if(F){
 out['f-hardware']=[0,1,2].map(v=>make(['原配','带宽翻倍','算力翻倍'][v],'矩阵乘加与 GPU 访存：一行结果怎样算出来？','输入 [2,3] 和两行权重保持可见。看参与计算的数如何配对、乘起来、再相加；最后只改变一个资源条件。',scope+' 只画一个小矩阵乘法；数据读取画成副本，权重不会从显存中消失。max(搬运,计算) 是可充分重叠时的理想下界，不是本动画的串行时长。',[
 ['先看原始数','权重第一行 [1,2]、第二行 [3,4]，都要与同一个输入 [2,3] 做点积。'],
 ['读取输入与权重','存储里的原数据保留；计算侧取得本次所需的数。这里省略实际缓存层级。'],
 ['对应位置分别相乘','第一行得到 2×1=2 和 3×2=6；第二行得到 2×3=6 和 3×4=12。'],
 ['每行把两项加起来','第一行 2+6=8，第二行 6+12=18。这才形成输出向量 [8,18]。'],
 ['结果可交给后续算子','不是把输入编号当答案，也不是搬来数据后无需计算。'],
 ['只改变一项资源','原配搬运 8 u、计算 3 u。带宽翻倍变成 4 与 3；算力翻倍变成 8 与 1.5。理想下界取较大者，不能把算力翻倍直接说成整体快一倍。']
 ],F.snapshot('hardware',v,4),'hardware'));
 out['f-communication']=['sum','gather'].map((kind,v)=>make(v?'拼接不同片段':'逐元素求和','AllReduce / AllGather：求和与拼接有何不同？','Rank 是进程在通信组内的编号，本例每个进程使用一张 GPU。两方各自保留 [1,2] 和 [10,20]。先确定这些数是“同一输出的贡献”还是“不同位置的片段”，再选择求和或拼接。',scope+' 图中“逻辑运算”不是中央服务器，也没有指定 ring、tree 等真实通信算法。返回值是结果副本，不把原输入搬空。',[
 ['两方各持有一份输入','Rank 是通信进程组内参与进程的编号；本例每个进程使用一张 GPU。编号需结合所属进程组理解，不是物理 GPU 的固有编号。'],
 ['取出本次参与合并的数据','原数据仍在各自一侧；副本进入逻辑运算示意区。'],
 [v?'按顺序排列片段':'对齐相同下标',v?'先放 [1,2]，再放 [10,20]；保留四个元素，没有相加。':'下标 0 对下标 0：1+10；下标 1 对下标 1：2+20。不能跨下标随意相加。'],
 ['得到完整结果',v?'结果长度从每方 2 个变为 4 个：[1,2,10,20]。':'结果仍为 2 个元素：[11,22]。每项都是双方在对应位置的贡献之和。'],
 ['两方各取得结果','此例分别对应 AllGather / AllReduce 的结果语义。看到两份相同结果，不意味着又多计算了两次合并。']
 ],F.snapshot('communication',v,4),kind));
 out['f-parallel']=[2,1].map(n=>make(n===2?'两个微批次（microbatch）':'只有一个微批次','PP 与 Microbatch：流水线气泡在等谁？','PP Stage 0 / Stage 1 是两个依次执行的流水线阶段，各负责不同模型层。MB1 / MB2 表示 microbatch（微批次）；每个微批次都必须依次经过两个 stage。',scope+' 本例 Stage 0 负责 L1–L2，Stage 1 负责 L3–L4，各占一张 GPU；stage 是层的分工，不等同于 GPU，也不是 Prefill / Decode。微批次是一次送入流水线的工作单元，可包含多条请求；这里用 MB1={R1,R2}、MB2={R3,R4} 示意，不规定真实引擎如何切分或组批。单微批次条件只送入 MB1，不作等工作量性能对比。时隙等长；交接动画不计真实通信耗时。',[
 ['先看阶段与微批次','MB1 必须完成 Stage 0 的 L1–L2，才有激活（activation，中间特征张量）供 Stage 1 的 L3–L4 使用。模型权重保留在各 stage。'],
 ['时隙 1：Stage 1 等激活','Stage 0 处理 MB1；Stage 1 还没有 MB1 的激活，不能开始计算。这段空闲是流水线气泡（pipeline bubble）。'],
 [n===2?'时隙 2：两个 stage 重叠执行':'时隙 2：Stage 1 接续计算',n===2?'MB1 的激活先到达 Stage 1，它才能计算；Stage 0 同期处理 MB2。两个 stage 处理不同微批次。':'MB1 的激活传到 Stage 1 后继续计算；Stage 0 没有下一个微批次可处理。'],
 ['时隙 3：流水线排空',n===2?'MB1 已完成两个 stage；Stage 1 处理 MB2，Stage 0 没有新微批次。':'MB1 已完成；本条件没有 MB2，后续时隙均无任务。'],
 ['回顾填充、重叠与排空','MB 是工作单元，Stage 是负责部分模型层的执行阶段。增加微批次可重叠不同阶段的工作，但不能消除同一微批次的依赖，也不保证等比例提速。']
 ],{n},'pipeline'));
 out['f-speculation']=[0,1,2].map(v=>{const d=F.snapshot('speculation',v,4);return speculation(make,['第 3 个分歧','第 1 个分歧','全部通过'][v],{...d,root:'根',predictions:Array.from({length:5},(_,i)=>i<d.accepted?d.draft[i]:i===d.accepted?d.replacement:d.draft[i]||'其他')},false,scope);});
 out['f-serving']=[make('运行中取消','Cancellation（请求取消）：断连后怎样收尾？','请求 R1 已经在执行。把客户端、取消消息、后端操作和资源占用分开看，先预测哪个能最早结束。',scope+' 通用安全生命周期示例，不对应某个引擎的抢占 API。完成在途使用后才解除资源占用；释放不是擦除显存内容。',[
 ['R1 已在执行','客户端仍在等；一项后端操作已经开始，资源被占用。'],
 ['客户端断开','客户端停止等待，但后端还未收到取消，也没有自动腾出资源。'],
 ['取消消息送达','控制消息到达后端，表示不要继续这条请求；它不是远程删除 GPU 数据的指令。'],
 ['后端登记取消','不再安排本例的后续工作；已开始的操作仍须经过安全边界。'],
 ['在途操作结束','本例已没有相关在途使用，才满足释放条件。客户端断开与这个事件之间没有固定时间差。'],
 ['归还资源使用权','资源可被后续请求重新分配。完成记录留在图上；不把释放画成物理显存清零。']
 ],{},'cancel')];
 }
 if(C)out.communication=[false,true].map(fragmented=>make(fragmented?'目标页不连续':'相邻页可合并','KV Transfer：源页怎样映射到目标页？','源页 7、8、12 固定不动。沿逐对映射把 A、B、C 的副本写到目标；改变目标页排列，观察传输块怎样变化。',scope+' 沿用本课页映射算例；单源、单目标、一个 K buffer。源端原数据保留，目标逐页写完才可读；此图未满足整个 PD 请求的全部就绪条件。',[
 ['源、目标页号不是同一套地址','目标只是预留位置。源页号 7 不能被直接当作目标页号。'],
 ['建立成对映射','两端 buffer 已按后端要求注册；目标通过控制路径提供可写地址与页映射。映射不是载荷，提交传输也不代表传输已完成。'],
 ['复制第一份载荷','A 的内容从源页 7 复制到指定目标页，源页仍保留。'],
 ['复制第二份载荷','B 按自己的映射写入；不是为了连续显示而移动物理页。'],
 ['复制第三份载荷','C 也写到对应目标。三个位置都写完，不等于跨 ranks、元数据与调度条件也都满足。'],
 ['检查哪些传输可合并','只有源页与目标页都连续，才把相邻页合成一个连续块。改变目标排列会改变块数，不改变总载荷 48 字节。']
 ],C.pages(fragmented),'pages'));
 if(G)out['advanced-generation']=[2,0,4].map(n=>speculation(make,`接受 ${n} 个候选`,G.speculate(n),true,scope));
 if(R)out['request-runtime']=[true,false].map(cache=>{const events=R.build({outputs:3,cache}),last=events.find(e=>e.id==='finish'),released=events.find(e=>e.id==='release');return make(cache?'保留前缀缓存':'关闭前缀缓存','Req 与 RadixCache：请求结束后谁持有 KV？','把 6 个有效 KV 位置固定下来。请求行（ReqToTokenPool 中记录 token 到 KV 地址映射的一行）、有效 KV 和最后输出 y3 分开观察。p1…p4 是输入位置，y1…y3 是输出次序，不是 token ID。',scope+' 沿用本课普通路径：输入 4、输出 3、page_size=1、无其他持有者。图中连接表示持有关系，取消连接不是搬动或擦除物理内容。',[
 ['最后一轮执行结束','p1…p4、y1、y2 已经过模型；新采样的 y3 还未再次输入，不能给它多记一份 KV。'],
 ['记录结束原因','finished_reason=length 表示后端判断已达到输出上限；请求行与 KV 仍未自动释放。'],
 [cache?'缓存管理继续保留有效 KV':'不保留可复用前缀',cache?'缓存管理接管有效 KV 的生命周期；物理位置不因为归属变化而移动。':'本例没有缓存保留或其他持有者，KV 将归还分配器。'],
 ['完成资源收尾',cache?'请求行归还，KV 由缓存继续保留，仍占显存。':'请求行与 KV 使用权都归还；原字节是否仍在不影响这些位置可重新分配。'],
 ['比较两种归属','是否保留缓存决定后续能否复用这份内容；不能从“请求结束”推导“显存全部空了”。']
 ],{cache,kv:last.kv,released},'ownership');});
 if(O)out['serving-operations']=['disconnect','waiting'].map(kind=>make(kind==='disconnect'?'等待中断连':'等待超时','AbortReq 与 IPC：取消去程和回告为何分开？','固定前端 ReqState 和 Scheduler 等待队列里的 Req。它们通过 rid=R1 对应，却是两个不同的本地对象。',scope+' 沿用本课普通等待请求取消路径；普通、从未执行的等待请求，没有分配 GPU KV。不能套用到 PD Decode 的预分配队列；断开的客户端不会收到回告。',[
 ['R1 在普通等待队列','前端与 Scheduler 各自持有状态；本例还没有执行 forward 或申请 GPU KV。'],
 [kind==='disconnect'?'客户端断开':'等待达到本例超时条件',kind==='disconnect'?'前端开始处理断连；后端队列中的 Req 此刻仍存在。':'Scheduler 检查等待时限；本地标记与移出队列分步展示。'],
 [kind==='disconnect'?'AbortReq 发出':'Scheduler 标记等待超时',kind==='disconnect'?'前端通过 IPC 派发取消；后端尚未处理，队列里的 R1 还在。':'等待超时由 Scheduler 本地发现，不需要前端先发送 AbortReq。随后移出请求并回告。'],
 ['后端移除并回告','Scheduler 处理取消，移出 Req，并产生回告。回告还需要送到前端。'],
 ['前端收到回告后收尾',kind==='disconnect'?'ReqState 清理完成；这不是向已断开客户端送达数据的证明。':'前端清理状态，并按本例等待超时路径返回 503。']
 ],{kind,states:Array.from({length:5},(_,i)=>O.cancellation(kind,i))},'queued-cancel')).map(s=>({...s,boundaryKey:s.data.kind==='waiting'?'queued-timeout':'serving-operations'}));
 return out;
}
function speculation(make,name,d,engine,scope){return make(name,engine?'EAGLE 验证与提交：输出和有效 KV 为何错位？':'Speculative Decoding：为什么只接受连续前缀？','固定验证输入与对应预测。逐列核对候选来自哪个前缀，再将接受的连续前缀与补出的 token 分开提交。'+(engine?' r 表示根 token，a–d 是四个草稿候选，z 是本例补出的 token；这些字母是教学标识。':'“根”表示上一轮已输出的 token，用作本轮验证输入起点；“我、想、喝、茶”是草稿候选。'),scope+(engine?' 沿用本课 EAGLE 线性教学算例；真实 tree 与概率接受算法未展开。':' 使用确定性贪心教学规则，不能套用到所有随机验证算法。')+' 每列预测依赖该列及之前的验证输入；补出的 token 尚未作为新输入。',[
 ['根位置已经输出','根 token（root token）是上一轮已生成、用于启动本轮验证的 token；不是一句话的第一个词。草稿候选尚未提交为正式输出。'],
 ['形成四个候选','候选的输入顺序为 [根, 候选1, 候选2, 候选3, 候选4]。后一个预测依赖前面的输入。'],
 ['目标模型给出逐位置预测','一轮验证可对多个位置计算预测；图按顺序揭示接受判断，不表示逐个发起五次模型调用。'],
 ['核对连续前缀',d.accepted===4?'四个都符合本例规则；最后验证位置还给出一个补出 token。':`前 ${d.accepted} 个通过，第 ${d.accepted+1} 个分歧。即使后面碰巧相同，也不能沿已被拒绝的前缀继续接受。`],
 ['提交正式输出','接受的候选前缀加上补出 token 成为输出；候选没有通过时不能直接发给用户。'],
 ['只保留有效输入对应的 KV','有效 KV 是根位置加接受前缀；输出是接受前缀加补出 token。两行长度相同，成员却不同。拒绝分支的临时状态不能当作有效历史。']
 ],d,engine?'engine-speculation':'speculation');}
if(typeof module==='object'&&module.exports)module.exports={create};
root.TeachingLessons={...(root.TeachingLessons||{}),...create(root.FoundationModel,root.CommunicationModel,root.GenerationModel,root.RequestRuntimeModel,root.ServingModel)};
})(typeof globalThis==='object'?globalThis:this);
