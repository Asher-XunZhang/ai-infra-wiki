/* Shared visual vocabulary; each mechanism has its own spatial representation. */
(()=>{'use strict';
const M=window.FoundationModel;if(!M)return;
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const token=(text,sub='',kind='',key='')=>`<span class="fd-token ${kind}"${key?` data-move="${esc(key)}"`:''}>${esc(text)}${sub?`<small>${esc(sub)}</small>`:''}</span>`;
const row=(items)=>`<div class="fd-row">${items.join('')}</div>`;
const zone=(title,body,active=false)=>`<div class="fd-zone${active?' active':''}"><strong>${esc(title)}</strong>${body}</div>`;
const columns=(a,b)=>`<div class="fd-columns">${a}${b}</div>`;
const label=x=>`<div class="fd-label">${esc(x)}</div>`;
const arrow=x=>`<div class="fd-arrow">↓ ${esc(x)}</div>`;
const equation=x=>`<div class="fd-equation">${esc(x)}</div>`;
const trace=(labels,active)=>`<div class="fd-trace">${labels.map((x,i)=>`<span class="${i===active?'active':''}">${esc(x)}</span>${i<labels.length-1?'<b aria-hidden="true">→</b>':''}`).join('')}</div>`;
const vec=v=>'['+v.map(x=>Number.isInteger(x)?x:x.toFixed(2)).join(', ')+']';
const score=(name,n,max,chosen=false,unit='')=>`<div class="fd-score${chosen?' chosen':''}"><span>${esc(name)}</span><div class="fd-track"><div class="fd-fill" style="width:${Math.max(0,n/max*100)}%"></div></div><span>${esc(n+unit)}</span></div>`;
function draw(d){let html='',why='';const {key:k,variant:v,step:s}=d;
 if(k==='tokens'){
  html=label(s<2?'原始文本 · 未加聊天模板':s===2?'片段 → 词表编号':s===3?'相同编号可以出现在不同位置':'片段与编号保持对应')+`<div class="fd-token-strip ${s===0?'joined':''}">`+d.pieces.map((p,i)=>`<div class="fd-word" data-move="piece-${i}">${s>=3?`<small>位置 ${i+1}</small>`:''}${token(p)}${s>=2?`<span aria-hidden="true">↓</span><strong class="fd-number">${d.ids[i]}</strong>`:''}${s===4?'<span class="fd-label">↓ Embedding 查表</span><span class="fd-vector" aria-label="一行向量的形状，非实际数值">[ <i></i><i></i><i></i> … ]</span>':''}</div>`).join('')+'</div>';
  if(s===3)html+=arrow('保持顺序，组成输入序列')+equation(vec(d.ids));
  if(s===5)html+=arrow('对整串编号执行反分词')+equation(d.text);
  why=[`你输入了“${d.text}”。先不引入模型层：文字还需要被组织成数值输入。`,`这一个 tokenizer 得到 ${d.count} 个 token。token 是能重复使用的文字片段；有限词表就能组合出许多新句子。它不一定是一个汉字，“学习”在这里是一块。`,'词表像一份编号目录。查到的 100134 表示“学习”这一项，不表示学习的难度或重要程度。','编号表示词表项，位置表示出现顺序。重复的“学习”可以共用编号，但不能共用位置。','模型不是拿 100134 这个整数当作意思直接计算；Embedding 用它选择一行数。这里的色格只表示向量形状。','还原后仍是原句，没有生成新回答。聊天包装、模型计算、输出选择，是后续另外的步骤。'][s];
 }else if(k==='generation'){
  const input=token('我',s>=4?'历史，读取记录':'教学 token',s>=4?'old':'')+token('想',s>=4?'历史，读取记录':'教学 token',s>=4?'old':'')+(s>=4?token(d.first,'上一轮输出送回','new','chosen-first'):'');
  const candidates=s>=5?[{word:d.second,p:.8},{word:'其他',p:.2}]:d.scores;
  html=columns(zone(s>=4?'本轮新输入与历史':'准备好的输入',row([input]),s===0||s===4),zone('客户端已经看见',row(d.visible?[token(d.visible,'可见文字','new','visible')]:[token('尚无文字','','future')]),s===3||s===6));
  if(s>=1&&s<7&&s!==4)html+=label(s>=5?'在选定的前缀之后，重新计算候选':'模型候选分布 · 手工教学值')+candidates.map(x=>score(x.word,Math.round(x.p*100),100,(s===2&&x.word===d.first)||(s===6&&x.word===d.second),'%')).join('');
  if(s===2)html+=row([token(d.first,'已选择，尚未返回','new','chosen-first')]);
  if(s>=1)html+=label('模型已经处理过的位置 · 留下计算记录')+row(d.kv.map(x=>token(x,'已处理','old')));
  if(s===7)html+=equation('达到本例输出上限 2 → 停止增加输出 → 进入资源收尾');
  why=['这里用手工词表的“我”“想”表示输入，不是上一课 Qwen 的真实切分。模型和推理引擎也不是同一件事。','输入经过模型，得到下一 token 的分数。图把分数转换后的候选占比画出来；不是答案正确率。',v===0?'贪心策略选择最高分“学习”。它是刚选出的结果，还没有作为下一轮输入经过模型。':'本次固定抽样选择“休息”。它不是最高分，但在教学分布中有非零概率。','所选编号经反分词和响应链成为可见文字。这里简化为一个片段一次返回。',`现在把“${d.first}”作为新位置送回完整模型，并读取保存的历史计算记录；这一轮才处理这个新输入。`,'上下文已经改变，因此要重新计算下一候选；不能反复沿用第一轮的分数。',`第二段“${d.second}”被选择并返回。它刚被选出，仍未作为下一轮输入交给模型。`,'本例用长度上限停止；实际也可能遇到结束标记或取消。停止生成和安全释放是不同动作。'][s];
 }else if(k==='attention'){
  html=label(`聚焦位置 ${d.query+1} · 输入向量是教学数值`)+row(d.inputs.map((x,i)=>token(vec(x),`位置 ${i+1}`,i===d.query?'new':'old','input-'+i)));
  if(s>=1)html+=equation(`Q = ${vec(d.Q)}；本例 K = 输入向量，V = 2 × 输入向量`);
  if(s>=2)html+=equation(`第 1 项分数：(${d.Q[0]}×${d.keys[0][0]} + ${d.Q[1]}×${d.keys[0][1]}) ÷ √2 = ${d.scores[0].toFixed(2)}`);
  if(s>=2)html+='<table class="fd-table"><thead><tr><th>位置 / K</th><th>Q·K / √2</th><th>权重</th><th>V</th></tr></thead><tbody>'+d.keys.map((x,i)=>`<tr class="${i>d.query?'masked':''}"><td>${i+1} · ${vec(x)}</td><td>${i>d.query?'遮住未来':d.scores[i].toFixed(2)}</td><td>${s>=3?(d.weights[i]*100).toFixed(1)+'%':'待归一化'}</td><td>${vec(d.values[i])}</td></tr>`).join('')+'</tbody></table>';
  if(s>=4)html+=arrow('每个 V 乘自己的权重，然后逐项相加')+equation('第一项 = '+d.weights.slice(0,d.query+1).map((w,i)=>`${w.toFixed(2)}×${d.values[i][0]}`).join(' + ')+' ≈ '+d.mixed[0].toFixed(2))+equation('两项分别计算，得到 '+vec(d.mixed));
  if(s===5)html+=trace([vec(d.mixed),vec(d.hidden),vec(d.output)],2)+label('MLP 小算例：2 个数 → 3 个数 → 2 个数；未显示完整残差与归一化');
  why=['每个输入位置是一行数。先观察哪一行，而不是先记 Q/K/V 三个缩写。','Q 用来提问，K 用来比较，V 携带要混合的信息。真实模型通过训练参数计算这些投影。','先计算允许位置的相关分数。即使输入已全部准备好，当前查询也不能使用后面的位置。','softmax 把允许位置的分数转成非负、合计为 1 的权重；被遮住的位置为 0。','取的是加权混合后的数值，不是从原文里直接复制一个词作为答案。','这一行表示还要经过后续变换。Prefill 和 Decode 都需要完整模型层；层数不是生成轮数。'][s];
 }else if(k==='cache'){
  if(v<2){
   html=label('每格只数一个位置的 K/V 投影工作，不是完整模型成本');
   html+=Array.from({length:3},(_,r)=>{const n=r+4,revealed=s>0&&r<=d.round;return `<div class="fd-label">第 ${r+1} 次模型调用</div>`+row(Array.from({length:n},(_,i)=>token(i<4?'输入'+(i+1):'输出'+(i-3),!revealed?'未发生':v===1&&r>0&&i<n-1?'复用':'计算',!revealed?'future':v===1&&r>0&&i<n-1?'old':'new')));}).join('');
   html+=`<div class="fd-summary">累计新算 K/V 的位置数：${d.computed.join(' + ')||'0'} = ${d.computed.reduce((a,b)=>a+b,0)}</div>`;
   why=['四个输入位置尚未经过模型，因此没有可复用的历史。','第一次都要计算四个位置。启用缓存时，把结果留下供下一次使用。',v===0?'不跨轮保留历史 K/V，第二次重新算五个位置。':'第二次只新增一个位置的 K/V；注意力仍需读取历史四个位置。',v===0?'第三次又算六个位置；这只是重复 K/V 投影的计数。':'第三次再新增一个位置。减少 K/V 投影不等于跳过其他模型计算。','请求停止后，缓存可被释放或由缓存系统继续持有。此处不把计数比值当作性能加速比。'][s];
  }else if(v===2){
   html=label('教学池：每页 2 个位置，物理页号不要求连续')+[7,2].map((page,i)=>`<div class="fd-page"><span>页 ${page}</span>`+[0,1].map(j=>{const n=i*2+j,held=i<d.ownedPages,written=n<d.written;return `<div class="fd-slot ${written?'written':held?'held':''}">${written?'位置 '+(n+1)+' 的 K/V':held?'已分配，待写':'空闲'}</div>`;}).join('')+'</div>').join('');
   if(s>=2)html+=equation(s>=3?'请求位置 1→页7/槽0，2→页7/槽1，3→页2/槽0':'请求位置 1→页7/槽0，2→页7/槽1');
   why=['先有空池，尚未给请求分配位置。','先领页 7。领到地址不等于已经算出或写入 K/V。','两个位置写满页 7。映射记录它们的位置，映射表不是第二份 K/V。','第三个位置需要另一页，领到页 2。逻辑顺序连续，物理页号可以跳跃。','这里保持最后快照以便核对映射。资源释放需要另外的生命周期条件，不能直接擦掉仍在使用的页。'][s];
  }else{
   html=columns(zone('R1',row([token(s===0?'尚未使用':s>=3?'已结束':'持有共享页',s>=1&&s<3?'↓ 指向同一份 K/V':'',s>=1&&s<3?'new':'old')])),zone('R2',row([token(s<2?'尚未使用':s===4?'已结束':'持有共享页',s>=2&&s<4?'↓ 指向同一份 K/V':'',s>=2&&s<4?'new':'old')])));
   html+=zone('同一个前缀的物理 K/V',row([token(d.retained?'已有一份历史':'尚未建立缓存',d.holders.length?'仍被 '+d.holders.join('、')+' 使用':'没有活跃请求持有',d.holders.length?'new':'old')]),true);
   why=['先建立一份兼容的前缀计算，才能谈共享；本例不以语义相似作为命中条件。','R1 使用这份历史，缓存只保存一份物理数据。','R2 的 token 前缀和状态兼容，因此也指向同一份历史，没有再复制一份。','R1 结束了，但 R2 仍在使用。不能把这份共享数据当作空闲空间覆盖。','两条请求都结束，缓存仍可保留历史供未来请求复用；驱逐才会归还池内位置。'][s];
  }
 }else if(k==='batching'){
  if(v<2){html=`<div class="fd-batch-board">${zone('等待下一次准入',row(d.waiting.length?d.waiting.map(x=>token(x,'等待','','req-'+x)):[token('空','','future')]))}${zone('仍需继续计算',row(d.active.length?d.active.map(x=>token(x,'还需 '+d.remaining[x]+' 轮','new','req-'+x)):[token('空','','future')]),true)}${zone('已完成',row(d.done.length?d.done.map(x=>token(x,'离开运行集合','old','req-'+x)):[token('空','','future')]))}</div>`;
   html+=label('已执行的轮次 · 每轮至多两个成员')+d.rounds.map((r,i)=>row([token('轮 '+(i+1),'','old'),...r.map(x=>token(x))])).join('');
   why=['A 需 1 轮，B 需 3 轮；C 会在第一轮结束后到达。先预测第二轮谁能进入。','A 完成离开，B 还需两轮，C 刚到等待区。成员的改变发生在轮次边界。',v===0?'固定批次对照继续只处理 B；即使有空位，C 仍等整批结束。':'连续批处理让 C 补进空位；本例所有额外准入条件都满足。','B 完成。连续模式下 C 已完成；固定模式下 C 还要下一轮才运行。','这里比较的是成员与依赖，不是等长 GPU 时间，也不能直接换算吞吐。'][s];
  }else{html=columns(zone('短请求：每轮续写一个位置',row(Array.from({length:d.shortOutputs},(_,i)=>token('输出'+(i+1),'已选出','new')))),zone('长请求：6 个输入位置',row(Array.from({length:6},(_,i)=>token(i+1,i<d.longDone?'已处理':'等待',i<d.longDone?'old':'future'))),true))+equation('本例每轮：1 个短请求位置 + 至多 2 个长输入位置');why=['先看一条正在续写的短请求，和一条有六个输入位置的长请求。','按本例策略先给短请求一个位置，再处理长输入的前两个位置。','第二轮续写继续，长输入推进到第四个位置；中间块还不是完整回答。','长输入最后一块完成，才具备产生其首输出的计算结果。','小块创造交错机会，也增加轮数。真正调度顺序、预算和耗时进入实现课再核对。'][s];}
 }else if(k==='hardware'){
  html=columns(zone('存储里的输入与参数',equation('输入 [2,3]；权重 [[1,2],[3,4]]'),s===0),zone('计算单元',s===0?row([token('等待数据','','future')]):s===1?row([token('[2,3]','输入已到','new','matrix')]):equation('第一行：2×1 + 3×2 = 8\n第二行：2×3 + 3×4 = 18'),s===1||s===2));
  if(s>=3)html+=arrow('乘加完成，得到中间结果')+row([token('[8,18]','不是整篇回答','new','matrix')]);
  if(s===4)html+=label('仅比较资源理想下界 · u 为教学单位')+score('搬运',d.move,8)+score('计算',d.compute,8)+equation('理想下界 ≥ max(搬运, 计算) = '+d.bound+' u');
  why=['权重是计算参数，输入是本轮数据。二者都不是事先写好的答案。','计算需要数据。数据从存储到计算单元的可用速度受带宽等条件约束。','把抽象矩阵拆开，就是多次乘法与加法。更大的模型重复更大规模的此类运算。','一次小算子输出中间数值，后面还有模型层、输出头和选择步骤。',v===1?'带宽翻倍使搬运下界从 8 降为 4 u；计算仍为 3 u。真实耗时可能高于这个下界。':v===2?'算力翻倍把计算下界降到 1.5 u，但搬运仍为 8 u，最大项没有改变。':'原配搬运下界比计算下界长。二者是可重叠资源的理想下界，不把 max 当作实测总时间。'][s];
 }else if(k==='parallel'){
  if(v===0){html=equation('单卡完整计算：2×1 + 3×2 = 8')+columns(zone('卡 0 的部分',row([token(s<2?'2×1':'2',s<2?'准备计算':'局部贡献','new','partial-a')])),zone('卡 1 的部分',row([token(s<2?'3×2':'6',s<2?'准备计算':'局部贡献','new','partial-b')])));if(s>=3)html+=arrow('相同结果的局部贡献求和')+equation('2 + 6 = 8');why=['先知道完整结果是什么，再拆分。','两卡分别持有这一次计算的一部分输入与权重。','得到 2 和 6，但单独任何一个都不是完整输出。','合并局部贡献，才恢复完整结果 8。','真实 TP 还有列切分、行切分等不同语义；不保证每种分片都用求和。'][s];}
  else if(v===1){html=columns(zone('GPU 0 · PP Stage 0',row([token(d.lanes[0]||'空闲',d.lanes[0]?'计算 Stage 0':'尚无这一轮工作',d.lanes[0]?'new':'future','lane0')])),zone('GPU 1 · PP Stage 1',row([token(d.lanes[1]||'等待',d.lanes[1]?'读取 Stage 0 结果':'没有可用 Stage 0 结果',d.lanes[1]?'new':'future','lane1')])));html+=label('完整依赖时间格 · 等长教学时隙')+'<div class="fd-grid"><span></span><span>时隙 1</span><span>时隙 2</span><span>时隙 3</span><span>结束</span><span>Stage 0</span>'+['MB1','MB2','空闲','完成'].map((x,i)=>`<span class="${i<2?'work':''} ${s===i+1?'current':''}">${x}</span>`).join('')+'<span>Stage 1</span>'+['等待','MB1','MB2','完成'].map((x,i)=>`<span class="${i===1||i===2?'work':''} ${s===i+1?'current':''}">${x}</span>`).join('')+'</div>';why=['MB1、MB2 是两个 microbatch（微批次）。每份都要走完 Stage 0 和 Stage 1。','只有 Stage 0 在做 MB1；Stage 1 必须等到它的中间结果。','Stage 0 处理 MB2，同时 Stage 1 处理 MB1；两张卡算的是不同批次。','MB1 已完成，Stage 1 继续 MB2。Stage 0 此时没有新任务。','两份批次都完成。填充与排空仍存在，流水线不会取消数据依赖。'][s];}
  else{html=columns(zone('副本 A · 完整模型',row(d.requests[0].length?d.requests[0].map(x=>token(x,'独立请求','new','dp-'+x)):[token('等待请求','','future')])),zone('副本 B · 完整模型',row(d.requests[1].length?d.requests[1].map(x=>token(x,'独立请求','new','dp-'+x)):[token('等待请求','','future')])));why=['两个副本各能独立完成请求，不是前后半个模型。','R1 选择副本 A；不会为了 DP 同时在两个副本重复执行。','R2 选择副本 B，两份请求有各自上下文与缓存。','两副本可以独立推进；它们不需要像 PP 那样逐层接力。','本例每副本一张卡；真实副本还可能内部组合 TP/PP。'][s];}
 }else if(k==='communication'){
  if(v<2){html=columns(zone('参与者 A',row((s<3?d.input[0]:d.result).map((x,i)=>token(x,s<3?'原有数据':'操作结果','new','comm-a-'+i)))),zone('参与者 B',row((s<3?d.input[1]:d.result).map((x,i)=>token(x,s<3?'原有数据':'操作结果','new','comm-b-'+i)))));if(s>=1)html+=trace(['提交',v===0?'对应元素相加':'按分片顺序收齐','双方获得结果'],Math.min(2,s-1));why=['先确认两组数据各自表示什么。',v===0?'这些是同一结果的局部贡献，应逐项求和。':'这些是不同片段，需要收齐，不能逐元素相加。','中间传输过程在本图中合并表示；结果还不能当作可用。',v===0?'1+10=11，2+20=22，参与者都得到 [11,22]。':'保留各片段，参与者都得到 [1,2,10,20]。','这才是后续计算需要的结果。这里没有推断网络带宽或硬件路径。'][s];}
  else{html=columns(zone('发送方',row([token('R1 元数据',s>=1?'已提交':'待发送','old'),token('R1 的 KV',s>=3?'已交接':s===2?'在途':'仍在本地','new','payload')])),zone('接收方',row([token(d.metadata?'已知 R1 身份':'尚无身份','','old'),token(d.arrived?'KV 已到达':'KV 尚不可读','','future')]),d.usable));why=['元数据描述请求身份，载荷才是计算需要的数值。','通知先到，接收方知道是谁，但没有完整载荷。','载荷还在途中，不能提前把它用于计算。','载荷到达，继续检查身份与其他就绪条件。','本教学场景的条件满足，才标为可使用。不同真实接口的完成语义要另外核对。'][s];}
 }else if(k==='deployment'){
  html=v===0?zone('同一服务组 · 完整模型',columns(zone('B：长输入',row([token(s>=1?'输入已处理':'等待 Prefill',s>=1?'历史 KV 在本地':'','new')])),zone('A / B：逐步续写',row([token(s>=4?'B 继续生成':'A 在续写',s>=4?'读取本地 KV':'与输入工作共享资源','old')]))),true):columns(zone('P 角色 · 完整模型',row([token(s>=1?'B 的 KV':'B 的输入',s>=1?'已算出的历史':'待处理','new','pd-kv')])+label(s>=2?'已得到首 token':'尚未生成首 token'),s<3),zone('D 角色 · 完整模型',row([token(s>=4?'B 的 KV 已可用':'等待 B 的状态',s>=4?'准入后续写':'可同时服务其他请求','old')]),s>=4));
  if(v===1&&s>=2)html+=trace(['首 token 与身份','KV 在途','接收并配对完成','D 准入继续'],s===2?0:s===3?1:3);
  why=['先看两种工作形态：长输入与续写。单个角色的服务组可以包含多张卡。',v===0?'同一服务组处理 B 的长输入，留下历史 K/V；A 的续写也需要这个组的资源。':'P 处理输入并在各层留下 K/V。这里没有把模型按前后层切开。','首 token 已产生。继续生成需要对应历史状态与请求信息。',v===0?'后续在同一实例读取本地 KV，不需要本例的跨角色交接。':'KV 与配对信息交给 D。传输增加了成本，此时 D 仍在等待。','有可用 KV、身份匹配且获得资源，才继续 Decode；不是数据一到就绕过调度。',v===0?'输入处理与续写共用资源，避免本例的跨角色 KV 搬运；两种工作也可能争用资源。':'P/D 可以分别配置资源，但也增加传输与协调。是否划算需要真实负载测量。'][s];
 }else if(k==='speculation'){
  html=label('上一轮已经输出的根 token')+row([token('根','本轮验证输入起点','old')]);
  if(s>=1)html+=label(s===4?'回顾候选：接受部分已提交，其余已丢弃':'草稿候选 · 尚未提交')+row(d.draft.map((x,i)=>token(x,s<3?'待验证':i<d.accepted?'接受':'丢弃',s<3?'future':i<d.accepted?'new':'blocked','draft-'+i)));
  if(s>=2)html+=equation(`目标模型：连续前 ${d.accepted} 个符合本例贪心规则；补出“${d.replacement}”`);
  if(s===4)html+=columns(zone('正式输出',row(d.output.map(x=>token(x)))),zone('有效 KV 对应输入',row(d.validKV.map(x=>token(x,'已作为输入','old')))));
  why=['普通生成中，上一输出会成为下一轮输入。先保留这个根位置。','先提出多个猜测，此时还不能当作正式输出。','本例逐位置检查固定的贪心结果；真实随机验证算法不能直接照搬。','第一个分歧之后都建立在不同前缀上，不能越过分歧保留后面的偶然相同项。','正式输出末尾的补出 token 尚未作为输入；根 token 却已写入 KV。两行长度相同也不等于成员相同。'][s];
 }else if(k==='serving'){
  const stages=v===2?['用户','接收','等待','断开 / 仍执行','取消已到 / 收尾','安全结束']:['用户','接收','等待','执行','回程','结束'];html=trace(stages,s)+columns(zone('客户端',row([token(d.clientDone?(v===2?'已停止等待':'看到回答'):'还在等',v===2&&s>=3?'连接已经断开':'','new','client')])),zone(d.selected,row([token(d.resourceHeld?'R1 正在使用执行资源':s===2?'R0 正在使用执行资源':s>=5?'资源已安全收尾':'R1 尚未占用执行资源',v===2&&s>=3?(d.cancelReceived?'取消已到达':'取消尚未到达'):'按请求推进','old','server')])));
  if(s===2)html+=equation('等待条件：R0 尚未归还执行资源 → R1 继续排队');
  if(s===3&&v!==2)html+=equation('R0 完成并归还 → R1 获得执行资源');
  if(v===1)html+=label('本例选择实例 B；实例 A 有完整模型，但没有接到 R1。');
  why=['用户准备好输入；这还不是后端已接到请求的证据。','入口接收了 R1，接收不等于 GPU 已开始。','本例把等待原因固定为：前一条请求 R0 正在使用执行资源，R1 留在等待区。',v===2?'用户此刻断开连接，但取消通知尚未到后端，执行资源仍被持有。':'R0 完成并归还执行资源，R1 才开始计算；客户端尚未收到结果。',v===2?'取消已被后端处理，仍要让相关在途操作安全结束后再释放。':'算出的结果经过反分词与响应链，正在发回客户端。',v===2?'本例在途操作终止后才收尾。不同实现的取消与回收路径有各自约束。':'客户端看到结果，同时本例资源收尾完成；真实系统这两个时刻也可能不同。'][s];
 }else if(k==='metrics'){
  if(v===4){
   html=label('固定窗口 0–10 秒 · 横线长度表示完成时刻，只统计已完成项')+'<div class="fd-throughput">'+d.completions.map((t,i)=>`<div class="fd-timing"><span>R${i+1}</span><div class="fd-completion ${i<d.completed?'arrived':'future'}" style="--at:${t*10}%"><span class="${i<d.completed?'arrived':'future'}">${t}s · ${d.outputCounts[i]} token</span></div></div>`).join('')+'</div>';
   html+=equation(`目前完成 ${d.completed} 条请求，共输出 ${d.outputTotal} 个 token`);
   if(s===4)html+=equation('请求吞吐 = 4 条 ÷ 10 秒 = 0.4 请求/秒\n输出吞吐 = 10 token ÷ 10 秒 = 1 token/秒');
   why=['先声明分母：整个 10 秒窗口。输入 token 不计入本例输出吞吐。','R1 在窗口内完成，记 1 条成功请求；它产生 2 个输出 token。','R2 完成后，累计 2 条请求、5 个输出 token。请求数和输出数的单位不同。','R3 完成。窗口还没有结束，不能拿累计数随意换一个分母。','R4 在 9 秒完成，10 秒关闭统计。总工作量除以共同窗口，不是把每条请求的 token/s 直接平均。'][s];
  }else if(v<2){html=label('客户端观测时间 · ms · 教学设定')+trace(['0 发出','100 首段',v===0?'140 第二段':'180 第二、三段',v===0?'180 第三段':'同包，无各自时刻'],Math.min(s,3));
   html+=label('已看见的输出')+row(Array.from({length:s===0?0:s===1?1:s===2&&v===0?2:3},(_,i)=>token('token '+(i+1),d.arrivals[i]+' ms','new')));
   if(s>=3)html+=equation(v===0?'TTFT = 100 ms；间隔 = 40、40 ms；TPOT = 40 ms':'首包 100 ms；下一个包 180 ms。无法从包内顺序恢复每个 token 的生成时刻。');
   why=['秒表从本例发出请求时开始，之前的客户端等待不计入。','100 ms 时第一次看到输出，TTFT 为 100 ms。',v===0?'140 ms 看到第二段；间隔是 40 ms。':'180 ms 的同一包带来第二和第三个 token；不能声称各自计算了 40 ms。','TPOT 在多于一个输出时才按后续时间除以后续数量。本例是平均口径。','所有指标必须交代起止点；吞吐还需要统计窗口与完成工作量，不由单个间隔代替。'][s];
  }else{html=label('延迟样本从小到大排列 · ms')+'<div class="fd-samples">'+d.samples.map((x,i)=>`<span class="fd-sample ${v===3&&i>=18?'failed':''} ${s>=3&&x===d.p95?'target':''}">${x}${v===3&&i>=18?' ×':''}</span>`).join('')+'</div>';if(s>=3)html+=equation(`成功 ${d.success.length}/20；nearest-rank P95 = 第 ${Math.ceil(.95*d.success.length)} 项 = ${d.p95} ms`);why=['固定 20 个样本，先看样本包含哪些请求。','本例用明确数列避免把图当作真实压测。',v===3?'把最慢两项改为失败，成功延迟分布只剩 18 项。':'全部 20 项都是成功请求。','对成功样本排序，再按声明的方法找百分位。',v===3?'P95 从 190 降到 180 ms，但成功率降到 90%。不能只看变小的延迟。':'P95 是分布位置，不是平均值，也不是吞吐。'][s];}
 }
 return {html:`<h3>${esc(d.title)}</h3>${html}`,why};
}
for(const lab of document.querySelectorAll('[data-foundation-lab]')){
 const key=lab.dataset.foundationLab,def=M.definitions[key];if(!def)continue;
 const allowed=window.LessonIntegration?.foundation[key]??def.options.map((_,i)=>i);if(!allowed.length)continue;
 let variant=Number(lab.dataset.initialVariant)||0,step=0,timer=null;const full=lab.dataset.full==='true',url=new URL(location.href);
 if(full){variant=Math.max(0,Math.min(def.options.length-1,Number(url.searchParams.get('fv'))||0));step=Math.max(0,Math.min(def.steps.length-1,Number(url.searchParams.get('fs'))||0));if(!Number.isInteger(variant))variant=0;if(!Number.isInteger(step))step=0;}
 if(!allowed.includes(variant)){variant=allowed[0];step=0;}
 const prefix='fd-'+key+'-'+[...document.querySelectorAll('[data-foundation-lab]')].indexOf(lab);
 lab.innerHTML=`<h2>${esc(window.LessonIntegration?.extraTitles[key]||lab.dataset.title||'动手走一遍')}</h2><div class="fd-settings"><label for="${prefix}-variant">${esc(def.label)}<select id="${prefix}-variant">${allowed.map(i=>`<option value="${i}">${esc(def.options[i])}</option>`).join('')}</select></label></div><div class="fd-controls"><button type="button" data-action="prev">上一步</button><button type="button" data-action="play" aria-pressed="false">播放</button><button type="button" data-action="next">下一步</button><button type="button" data-action="reset">重置</button><output aria-label="当前教学步骤"></output></div><div class="fd-scene"></div><p class="fd-caption" aria-live="polite"></p><p class="fd-boundary">教学步骤不代表执行耗时。可随时暂停、回看；改变选项会从头开始。</p>`;
 const q=s=>lab.querySelector(s),scene=q('.fd-scene'),media=matchMedia('(prefers-reduced-motion: reduce)');
 function stop(){window.LearningPlayback.stop(timer);timer=null;q('[data-action=play]').textContent='播放';q('[data-action=play]').setAttribute('aria-pressed','false');}
 function render(animate=false){const before=new Map([...scene.querySelectorAll('[data-move]')].map(e=>[e.dataset.move,e.getBoundingClientRect()]));const d=M.snapshot(key,variant,step),r=draw(d);scene.innerHTML=r.html;q('.fd-caption').textContent=r.why;q('output').textContent=`${step+1} / ${def.steps.length}`;q('select').value=String(variant);q('[data-action=prev]').disabled=step===0;q('[data-action=next]').disabled=step===d.last;lab.dataset.step=String(step);lab.dataset.variant=String(variant);
  if(animate&&!media.matches){const moves=[...scene.querySelectorAll('[data-move]')].map(e=>({e,a:before.get(e.dataset.move),b:e.getBoundingClientRect()}));for(const {e,a,b} of moves)if(a)e.animate([{transform:`translate(${a.x-b.x}px,${a.y-b.y}px)`},{transform:'translate(0,0)'}],{duration:225/window.LearningPlayback.rate,easing:'ease-out'});}
 }
 function remember(){if(full){const u=new URL(location.href);u.searchParams.set('fv',variant);u.searchParams.set('fs',step);history.replaceState(null,'',u);}}
 q('[data-action=prev]').onclick=()=>{stop();step=Math.max(0,step-1);render(true);remember();};q('[data-action=next]').onclick=()=>{stop();step=Math.min(def.steps.length-1,step+1);render(true);remember();};q('[data-action=reset]').onclick=()=>{stop();step=0;render();remember();};q('select').onchange=e=>{stop();variant=Number(e.target.value);step=0;render();remember();};
 q('[data-action=play]').onclick=()=>{if(timer){stop();return;}if(step===def.steps.length-1)step=0;q('[data-action=play]').textContent='暂停';q('[data-action=play]').setAttribute('aria-pressed','true');render();timer=window.LearningPlayback.repeat(()=>{if(document.hidden||(lab.closest('details')&&!lab.closest('details').open)){stop();return;}step++;render(true);remember();if(step===def.steps.length-1)stop();},3000);};
 document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});window.addEventListener('pagehide',stop);lab.closest('details')?.addEventListener('toggle',e=>{if(!e.target.open)stop();});render();
}
for(const quiz of document.querySelectorAll('.fd-quiz')){quiz.hidden=false;quiz.addEventListener('submit',e=>{e.preventDefault();const selected=quiz.querySelector('input:checked'),out=quiz.querySelector('.fd-feedback');if(!selected){out.textContent='先选一个判断，再对照演示。';return;}out.textContent=(selected.value===quiz.dataset.answer?'判断正确。':'再看一下图中的变化。')+quiz.dataset.feedback;});}
})();
