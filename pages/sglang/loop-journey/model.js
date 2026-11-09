/* Illustrative execution time, not wall time. State is rebuilt from milestones. */
(function(root,factory){const api=factory(typeof module==='object'&&module.exports?require('../parallelism/flow-model.js'):root.ParallelismFlow);if(typeof module==='object'&&module.exports)module.exports=api;else root.LoopJourney=api;})(typeof globalThis!=='undefined'?globalThis:this,P=>{
'use strict';
const revision='279339f113b79af84f27fd3ac92d0a13bd3f4cbd';
const sources={
 normal:['managers/scheduler.py',1907,'event_loop_normal'],
 overlap:['managers/scheduler.py',1942,'event_loop_overlap'],
 war:['managers/scheduler.py',1893,'_apply_war_barrier'],
 run:['managers/scheduler.py',4285,'run_batch'],
 resolve:['managers/overlap_utils.py',87,'resolve_forward_inputs'],
 relay:['managers/scheduler.py',4543,'_relay_forward_payload'],
 prefillResult:['managers/scheduler_components/batch_result_processor.py',257,'process_batch_result_prefill'],
 result:['managers/scheduler_components/batch_result_processor.py',920,'process_batch_result_decode'],
 pp:['managers/scheduler_pp_mixin.py',63,'event_loop_pp'],
 slots:['managers/scheduler_pp_mixin.py',553,'init_pp_loop_state'],
 output:['managers/scheduler_pp_mixin.py',983,'_pp_send_output_to_next_stage'],
 prep:['managers/scheduler_pp_mixin.py',1018,'_pp_send_recv_and_preprocess_output_tensors'],
 launch:['managers/scheduler_pp_mixin.py',1221,'_pp_launch_batch'],
 gate:['arg_groups/overrides.py',1662,'_pipeline_parallel_overlap_disable']
};
const sourceURL=key=>`https://github.com/sgl-project/sglang/blob/${revision}/python/sglang/srt/${sources[key][0]}#L${sources[key][1]}`;
// a: [lane, visible action, batch identity, local loop]. State patches contain
// learner-visible data, not timings; dependencies name completed event IDs.
const event=(id,title,summary,a,deps,source,patch={},why='')=>({id,title,summary,a,deps,source,patch,why});
const nonppLanes=[['cpu','CPU · Scheduler','一个进程内的选批、提交与结果处理'],['schedule','GPU · 输入准备','schedule stream；地址、长度等元数据'],['gpu','GPU · 模型与采样','Normal 为当前执行流；Overlap 为 forward stream'],['copy','GPU → CPU','结果可读性；Overlap 使用 copy stream']];
const initial={request:'R1 等待接纳；输入 [p0,p1,p2]',batch:'尚未组批',device:'输入未上卡；未生成 token',host:'output_ids = []',queue:'无待处理结果',resource:'尚未分配请求槽与本轮 KV 写入位置',place:0,payload:'R1 · 3 个输入位置'};
const normal=[
 event('start','请求进入循环','R1 是一次生成请求；本例先算 3 个 prompt 位置，首个输出为 t1，下一轮输出 EOS（结束符）。',[['cpu','接收 R1 → waiting','R1','L0']],[],'normal'),
 event('plan','选出 B0：一次 Extend','CPU 决定本轮成员和输入范围，分配请求槽与 KV 写入地址。地址已预留，不代表 KV 内容已算好。',[['cpu','选批 / 准备 B0','B0','L0'],['schedule','长度、位置与写入地址','B0','L0']],['start'],'normal',{request:'R1 已接纳；在设备执行与生成中',batch:'B0 = R1 的 Extend；3 个新位置',resource:'R1 槽已占用；3 个 KV 写入位置已预留',place:1,payload:'B0 · prompt + 地址'}),
 event('launch','提交 B0，输入到设备','run_batch 经 Worker / Runner 提交设备工作；resolve_forward_inputs 将 pinned CPU prompt 搬到 GPU。Python 返回不自动等于 GPU 全部完成。',[['cpu','run_batch 提交','B0','L0'],['gpu','H2D → 模型层开始','B0','L0']],['plan'],'resolve',{device:'B0：写各层 KV、计算 hidden states',place:2,payload:'B0 · GPU 执行中'}),
 event('sample','B0 生成首 token t1','模型层完成后经输出头与采样产生 t1；结果写入 FutureMap，供下一轮在设备上取输入。Normal 也使用这条设备接力。',[['gpu','模型 → logits → t1','B0','L0']],['launch'],'relay',{device:'FutureMap[R1] = t1；prompt KV 已写入',payload:'B0 → t1'}),
 event('ready','让 B0 结果在 CPU 可读','结果处理需要实际 token 值。普通路径读取结果时可能等待设备／拷贝；图用一个可读门槛概括，不虚构独立 copy stream。',[['copy','t1 → CPU 可读','B0','L0']],['sample'],'run',{queue:'B0 结果可供 CPU 消费',place:3,payload:'t1 · 结果可读'}),
 event('consume','处理 B0，提交一次输出','process_batch_result 更新 output_ids、完成条件与输出；R1 尚未结束，继续保留 KV。此处之后才进入下一轮选批。',[['cpu','追加 t1 / 继续 R1','B0','L0']],['ready'],'normal',{host:'output_ids = [t1]',request:'R1 继续生成',queue:'B0 已处理',place:4,payload:'R1 · 已提交 t1'}),
 event('decode','L1 选 B1：一次 Decode','R1 身份不变，B1 是新的执行批次；只新增 1 个位置。输入从 FutureMap 取 t1，并使用已有 KV。',[['cpu','更新成员 / 准备 B1','B1','L1'],['schedule','新增 1 个写入位置','B1','L1']],['consume'],'resolve',{batch:'B1 = R1 的 Decode；1 个新位置',resource:'保留 prompt KV；预留下一个 KV 位置',place:1,payload:'B1 · 输入 t1'}),
 event('decode-run','提交并执行 B1','先满足输入与元数据依赖，再执行模型层和采样；新 KV 对应输入 t1，采出的 EOS 不是刚写入的 KV。',[['cpu','提交 B1','B1','L1'],['gpu','读 t1 / 历史 KV → EOS','B1','L1']],['decode'],'run',{device:'FutureMap[R1] = EOS；t1 的 KV 已写入',place:2,payload:'B1 → EOS'}),
 event('end-ready','B1 的 EOS 可供 CPU 读取','输出已在 GPU 生成与 CPU 已能判断完成，是两个事件。',[['copy','EOS → CPU 可读','B1','L1']],['decode-run'],'result',{queue:'B1 结果待处理',place:3,payload:'EOS · 结果可读'}),
 event('finish','R1 完成，进入收尾','CPU 消费 EOS 后标记完成，停止向读者追加后续 token，按缓存策略处理请求与 KV。共享前缀可能保留，不等于清空显存。',[['cpu','判定完成 / 释放或缓存','B1','L1']],['end-ready'],'result',{host:'output_ids = [t1, EOS]；流式正文可过滤 EOS',request:'R1 已完成',queue:'结果全部处理',resource:'请求所有权已结束；KV 按引用与缓存策略处理',place:4,payload:'R1 · 完成'}),
 event('idle','L2 没有下一批，进入 idle','下一轮过滤完成成员；无 waiting、running 或待处理结果，执行 on_idle。一次 loop 和一次请求的生命周期并不相等。',[['cpu','无 batch → on_idle','—','L2']],['finish'],'normal',{batch:'无可执行 batch',queue:'空；无需排空',resource:'本例没有在途设备工作；可按分配器规则复用'})
];
const overlap=[
 event('start','同一条 R1，换成 Overlap','仍从 prompt 到 t1 再到 EOS。区别在于 CPU 可以先准备并提交下一批，再消费上一批结果；没有把相关 GPU forward 变成并行。',[['cpu','接收 R1 → waiting','R1','L0']],[],'overlap'),
 event('plan','准备 B0 的元数据','选批、地址分配和 schedule stream 上的元数据准备先发生。',[['cpu','选择 B0 / 准备输入','B0','L0'],['schedule','B0 地址与长度','B0','L0']],['start'],'overlap',{request:'R1 已接纳；在设备执行与生成中',batch:'B0 = R1 的 Extend',resource:'R1 槽与 prompt KV 地址已预留',place:1,payload:'B0 · prompt + 地址'}),
 event('launch','B0 提交后，结果进入队列','forward_stream 等待已提交的 schedule_stream 工作，然后 H2D、执行模型。CPU 保存 batch 元数据快照与结果引用；入队不表示计算已结束。',[['cpu','提交 / result_queue 入队','B0','L0'],['gpu','等待输入 → B0 模型','B0','L0']],['plan'],'run',{device:'B0 设备工作在途',queue:'[B0]；保存快照与结果引用',place:2,payload:'B0 · GPU 执行中'}),
 event('prepare-next','L1 提前准备 B1','CPU 不必先读回 t1 才能整理下一批元数据。schedule stream 上可能覆盖共享缓冲的写入必须等前一 forward 的共享读完成（WAR）。',[['cpu','规划下一批 B1','B1','L1'],['schedule','共享读完成 → 才能覆写','B1','L1'],['gpu','B0 仍可执行','B0','L0']],['launch'],'war',{batch:'B1 的元数据准备；t1 尚未供 CPU 消费',resource:'CPU 可规划；设备写共享缓冲仍受 WAR 栅栏约束',place:2,payload:'B0 计算 ∥ B1 准备'},'wait_event / wait_stream 为 GPU 排序；不是让 CPU 在这里等待整个 GPU。'),
 event('launch-next','先提交 B1，再等旧结果','CPU 可以在 t1 尚未生成时提交 B1；设备将 B1 排在 B0 后，实际读取必须等 FutureMap 写入。随后 CPU 等旧 B0 结果，不必把 t1 先读回再传回 GPU。',[['cpu','提交 B1 / 保存快照','B1','L1'],['gpu','B1 已排队；等待 t1','B1','L1']],['prepare-next'],'overlap',{batch:'B1 = R1 的 Decode，已提交',queue:'[B0, B1]；队首仍是 B0',device:'B1 已提交，设备消费仍等待 B0 的 t1',payload:'B1 · GPU 执行中'}),
 event('sample','B0 写入 t1，B1 才能消费','同一 forward stream 中，B0 写 FutureMap 先于 B1 读取它。copy stream 等待 B0 输出，再准备 CPU 副本。',[['gpu','采样 t1 → FutureMap','B0','L0'],['copy','等待 B0 → 拷贝 t1','B0','L0']],['launch-next'],'relay',{device:'FutureMap[R1] = t1；B1 现可继续设备执行',queue:'[B0, B1]；B0 的 CPU 副本尚未确认可读',payload:'t1 · 设备接力'}),
 event('ready','B0 的 copy_done 才允许 CPU 消费','process_batch_result 在需要读取结果时 synchronize copy_done；如果 t1 副本没准备好，阻塞的是 CPU 的结果处理。B1 不因此变成未提交。',[['copy','B0 copy_done','B0','L0'],['cpu','等待后可读 t1','B0','L1']],['sample'],'prefillResult',{queue:'B0 已可读；B1 仍待处理',place:3,payload:'B0 · t1 可读'},'CPU 等旧 B0 的结果；已提交的 B1 可以继续执行。'),
 event('consume','CPU 提交 t1，GPU 可执行 B1','从 result_queue 弹出 B0，更新 R1。这里的并行窗口是 CPU 处理旧结果与设备执行新批；实际是否重叠、重叠多少由运行时决定。',[['cpu','处理旧 B0 / 追加 t1','B0','L1'],['gpu','B1 模型继续','B1','L1']],['ready'],'overlap',{host:'output_ids = [t1]',request:'R1 继续生成',queue:'[B1]',place:4,payload:'已提交 t1；B1 在途'}),
 event('ahead','L2 先提交 B2；B1 产生 EOS','在 CPU 尚未消费 B1 的 EOS 时，R1 看起来仍未完成，B2 可能已被提交。B2 的设备执行仍要等 B1 的 EOS 写入 FutureMap。',[['cpu','准备并提交 B2','B2','L2'],['schedule','B2 写入受 WAR 约束','B2','L2'],['gpu','B1 → EOS；B2 已排在后面','B1','L1'],['copy','B1 EOS → CPU 副本','B1','L1']],['consume'],'run',{batch:'B2 已提交，但不保证会形成有效输出',device:'B1 生成 EOS；B2 在其后使用设备输入',queue:'[B1, B2]',resource:'已有在途 B2；不能仅因逻辑结束就立即覆盖它使用的缓冲',place:2,payload:'B2 · 可能多算的一批'}),
 event('finish','CPU 读到 EOS，逻辑上结束 R1','等待 B1 的结果可读后，提交 EOS 并标记完成。已经提交的 B2 不能被这一步当成从未发生；释放路径仍受设备依赖与引用保活约束。',[['copy','B1 结果可读','B1','L1'],['cpu','处理 B1 / R1 完成','B1','L2']],['ahead'],'result',{host:'output_ids = [t1, EOS]；后续结果不得追加',request:'R1 已完成；B2 仍需收尾',queue:'[B2]',place:4,payload:'R1 逻辑完成；B2 未排空'}),
 event('drain','L3 无新 batch，仍需排空 B2','过滤完成请求后选不到新批；last_batch 仍存在，继续 pop_and_process。等 B2 结果就绪，结果处理跳过已完成的 R1，不产生额外有效输出。',[['gpu','B2 已提交工作结束','B2','L2'],['copy','B2 结果就绪','B2','L2'],['cpu','无新批 / 排空旧 B2','B2','L3']],['finish'],'overlap',{batch:'无新 batch',queue:'空；B2 已收尾',device:'B2 不再作为有效 token 输出',resource:'在途工作已收尾；复用仍按缓存／分配器规则',payload:'有效输出仍只有 [t1, EOS]'}),
 event('idle','L4 才是空闲闭环','last_batch 已为空，当前也没有新批，才执行 on_idle。队列空、逻辑完成和设备工作结束分别检查。',[['cpu','无新批 / 无旧结果 → idle','—','L4']],['drain'],'overlap',{request:'R1 已完成并收尾',resource:'本例没有在途设备工作'})
];
const ppInitial={request:'R1、R2 是两条独立请求；已到各自最后一次 Decode',batch:'M1 → R1，M2 → R2；两级本地对象分别维护',device:'各 stage 保有自己模型层的权重与 KV',host:'两级尚未处理本轮输出',queue:'slot 0、slot 1 初始可用',resource:'2 个可复用 slot；pp_size 2 + async_depth 0',place:0,payload:'R1 / R2 · 最后一次 Decode'};
const pp=[
 event('start','两个 stage，各有自己的循环','教学设定：同一主机两张 GPU，PP=2、TP=1、CUDA、无 PD / HiCache；async depth=0。为看清排空，R1、R2 都在本轮结束。',[['cpu0','slot 0：接入 R1','M1','P0:L0'],['cpu1','等待 R1 请求信息','M1','P1:L0']],[],'slots'),
 event('plan','PP0 选 M1，向 PP1 传请求信息','转发的是请求控制信息；各 stage 建立本地 batch 和地址。M1 是本次执行身份，slot 0 是将来还会复用的位置。',[['cpu0','slot 0：选 M1 / 传请求','M1','P0:L0'],['cpu1','slot 0：接收 / 本地选 M1','M1','P1:L0']],['start'],'pp',{batch:'slot 0 = M1(R1)；slot 1 暂空',queue:'各 stage 的 mbs[0] = 本地 M1',place:1,payload:'M1 · slot 0'}),
 event('launch0','GPU0 只算前半模型层','forward stream 等本级输入准备，再启动 M1。记录 launch_event；事件记录动作已提交，不等于 CPU 等待计算结束。',[['cpu0','提交 M1 / 记录事件','M1','P0:L0'],['gpu0','M1：前段模型层','M1','P0:L0']],['plan'],'launch',{device:'M1 的本级 KV 留在 GPU0；hidden states 在生成',place:2,payload:'M1 · 前段激活生成中'}),
 event('proxy','激活就绪才可送到 GPU1','发送流等待 launch_event，传 PPProxyTensors 中的 hidden states 等激活；接收端建立设备依赖后才能让后段层消费。这里不传整套 KV。',[['wire','GPU0 → GPU1：M1 激活','M1','P0:L0 → P1:L0'],['cpu1','接收 proxy / 建立依赖','M1','P1:L0']],['launch0'],'pp',{batch:'slot 0=M1；slot 1=M2 已提交',queue:'PP0 已进入 L1，接下来处理旧 M1',device:'M1 激活在 GPU1 可用；M2 已在 GPU0 排队；各级 KV 留本地',place:3,payload:'M1 · 激活已到 GPU1'},'若激活未到，GPU1 的模型消费者不能开始；CPU 接收路径也可能等待匹配数据。'),
 event('parallel','GPU1 算 M1；PP0 推进 slot 1','PP1 已提交的 M1 现在能消费到达的激活；PP0 也已在下一本地轮次选 M2 并提交前段。不同 GPU、不同 microbatch 具备重叠条件；列宽不是耗时。',[['cpu0','slot 1：M2 已提交 / 等 M1 输出','M2','P0:L1'],['gpu0','M2：前段模型层','M2','P0:L1'],['cpu1','slot 0：M1 已提交 / 消费者就绪','M1','P1:L0'],['gpu1','M1：后段层 → 采样','M1','P1:L0']],['proxy'],'launch',{batch:'slot 0=M1；slot 1=M2',queue:'PP0 当前算 M2，接下来要处理旧 M1',device:'GPU0 可算 M2；GPU1 可算 M1',payload:'M1 后段 ∥ M2 前段'}),
 event('output1','末级把 M1 输出回传给 PP0','GPU1 采样完成后，output 消息从末级回到首级。GPU 激活前传和 token 结果回传属于不同载荷、不同完成条件。',[['gpu1','M1 采样完成','M1','P1:L0'],['wire','GPU1 → GPU0：M1 输出','M1','P1:L0 → P0:L1']],['parallel'],'output',{queue:'M1 输出回传；M2 尚未发给 GPU1',device:'GPU1 已产生 M1 输出；GPU0 的 M2 可继续',place:4,payload:'M1 · 采样输出回到 PP0'}),
 event('consume0','PP0 在 L1 处理旧 slot 0','depth=0 时先提交当前 M2，再接收旧 M1 输出；copy stream 准备本地结果，CPU synchronize d2h_event 后才更新 R1。',[['cpu0','等 d2h_event / 处理 M1','M1','P0:L1'],['gpu0','M2 已提交，可继续','M2','P0:L1']],['output1'],'prep',{host:'PP0：R1 完成；PP1 尚未处理 M1',queue:'PP0 last_mbs[0] = M1；slot 1 仍是 M2',payload:'M1 · PP0 已提交完成'},'当前槽 mb_id=1；处理的是 next_mb_id=(1+1)%2=0 的旧批。'),
 event('proxy2','M2 激活现在才向后传','本例普通 PP depth=0 的循环在旧结果处理之后发送当前 proxy；还必须满足 M2 的 launch_event。上一批结果迟到可拖住下一批的前传。',[['wire','GPU0 → GPU1：M2 激活','M2','P0:L1 → P1:L1'],['cpu1','slot 1：接收 / 提交 M2','M2','P1:L1'],['gpu1','M2 后段层 → 采样','M2','P1:L1']],['consume0'],'pp',{device:'M2 后段执行；M1 的结果仍需在 PP1 提交',place:3,payload:'M2 · 后段执行'}),
 event('return2','PP1 发 M2 输出，再等旧 M1','PP1 L1 的当前槽是 1；末级发送 M2 输出，并等待 PP0 转发上一轮已收到的 M1 结果。GPU1 算过 M1，不等于本地 CPU 已处理它。',[['wire','GPU1 → GPU0：M2 输出','M2','P1:L1 → P0:L2'],['cpu1','等待旧 M1 结果回到本级','M1','P1:L1']],['proxy2'],'output',{queue:'M2 输出就绪；PP1 等 M1 回环',payload:'M2 输出；M1 待本地提交'}),
 event('relay1','PP0 L2 过滤完成 R1，转发 M1','轮转回 slot 0 后，完成 R1 被过滤，当前无新批；仍转发上轮保存的 M1 输出给 PP1，同时接收旧 slot 1 的 M2 输出。',[['cpu0','slot 0：无新批 / 接收 M2','M2','P0:L2'],['wire','GPU0 → GPU1：转发 M1 输出','M1','P0:L2 → P1:L1']],['return2'],'pp',{batch:'PP0 slot 0 无新批；slot 1 等 M2 处理',place:4,payload:'M1 输出环回 PP1；M2 回到 PP0'}),
 event('commit-both','各自等待结果，再更新本地请求','PP1 L1 的 CPU 消费 M1，PP0 L2 的 CPU 消费 M2；它们属于不同本地轮次，不是一只全局同步 loop。各自先等 d2h_event。',[['cpu0','处理 M2 / R2 完成','M2','P0:L2'],['cpu1','处理 M1 / R1 完成','M1','P1:L1']],['relay1'],'prep',{host:'PP0：R1、R2 完成；PP1：R1 完成，R2 待处理',queue:'PP0 保存 M2 输出，待下一轮转发',payload:'各级本地状态逐步收尾'}),
 event('drain','无新计算，继续完成 M2 回环','PP0 L3（slot 1）过滤完成 R2，并把 M2 输出转发 PP1；PP1 L2（slot 0）无新批，但仍等待并处理旧 slot 1 的 M2。',[['wire','GPU0 → GPU1：转发 M2 输出','M2','P0:L3 → P1:L2'],['cpu0','slot 1：过滤完成 R2','M2','P0:L3'],['cpu1','等结果 / 处理 M2','M2','P1:L2']],['commit-both'],'pp',{host:'PP0、PP1 均已处理 R1、R2 完成',queue:'两个 microbatch 的输出回环都已闭合',place:5,payload:'M1 / M2 · 各级结果处理完成'}),
 event('idle','槽位可以复用，循环继续','各 stage 在后续轮转中过滤完成成员；都没有新 batch 时进入空闲检查。slot 0/1 是循环状态容器，不是永不变化的 M1/M2，也不是显存页。',[['cpu0','空槽 / 后续可接新请求','—','P0:后续'],['cpu1','过滤 slot 1 / 空闲检查','—','P1:L3']],['drain'],'slots',{request:'R1、R2 在两级均完成',batch:'mbs 中不再有可执行请求',queue:'本例在途输出已处理',resource:'slot 0/1 可接下一代 batch；KV 按各级缓存策略回收或保留'})
];
const configs={
 normal:{name:'普通循环',question:'一轮如何从选批走到结果提交，再进入下一轮？',scope:'PP=1、TP=1 · CUDA · 普通生成 · 无投机、无约束语法、无 PD / HiCache',lanes:nonppLanes,initial,steps:normal,barrier:'ready',barrierLabel:'暂缓 B0 的 CPU 结果可读',positions:['等待接纳','CPU 输入准备','GPU 模型 / 采样','结果可读','CPU 提交 / 收尾']},
 overlap:{name:'Overlap',question:'CPU 可以提前到哪一步，又必须在哪些地方等待？',scope:'PP=1、TP=1 · CUDA · Overlap 开启 · 普通生成、无延迟采样 · EOS 可能晚于下一批提交被 CPU 发现',lanes:nonppLanes,initial,steps:overlap,barrier:'ready',barrierLabel:'暂缓 B0 的 copy_done',positions:['等待接纳','CPU 输入准备','GPU 模型 / 采样','结果可读','CPU 提交 / 收尾']},
 pp:{name:'PP 循环',question:'当前槽位算哪一批，旧结果归谁，何时才能进入下一级？',scope:'PP=2、TP=1 · 同机两 GPU / 两 Scheduler 进程 · CUDA · async depth=0 · 无 PD / HiCache · R1/R2 本轮完成',lanes:[['cpu0','CPU · PP0','首级 Scheduler 进程 / 本地 loop'],['gpu0','GPU0 · PP0','前段模型层 / 本地 KV'],['wire','GPU 间交接','激活前传、输出回环；不搬整套 KV'],['gpu1','GPU1 · PP1','后段层、输出头与采样 / 本地 KV'],['cpu1','CPU · PP1','末级 Scheduler 进程 / 本地 loop']],initial:ppInitial,steps:pp,barrier:'proxy',barrierLabel:'暂缓 M1 激活就绪',positions:['请求 / 旧 KV','CPU0 输入准备','GPU0 前段层','GPU1 后段层','PP0 处理输出','PP1 处理输出']}
};
function snapshot(mode,step=0,{hold=false,released=false}={}){
 const c=configs[mode];if(!c)throw Error('Unknown mode');
 const barrier=c.steps.findIndex(s=>s.id===c.barrier);
 const index=Math.max(0,Math.min(c.steps.length-1,Math.floor(Number(step)||0),hold&&!released?barrier:Infinity));
 const blocked=hold&&!released&&index===barrier;
 const completed=c.steps.slice(0,index+(blocked?0:1));
 const data=Object.assign({},c.initial,...completed.map(s=>s.patch));
 if(blocked&&mode==='pp')Object.assign(data,{batch:'slot 0=M1；slot 1=M2 已提交',queue:'PP0 已进入 L1；等待 M1 输出',device:'M1 激活尚未就绪；M2 已提交，但设备仍需等待前序 proxy 发送'});
 const current=c.steps[index],done=new Set(completed.map(s=>s.id));
 return {mode,index,blocked,data,current,completed:[...done],dependencies:current.deps.map(id=>({id,title:c.steps.find(s=>s.id===id).title,ready:done.has(id)})),last:index===c.steps.length-1};
}
// A single teaching clock for every lane in each scenario. Durations are
// illustrative units, not measurements; PP reuses the parallel lesson's model.
function execution(mode){
 const events=[],lanes=[];let duration,times,barrierEvent;
 const lane=(id,label)=>{lanes.push({id,label});};
 const add=(lane,id,label,start,end,kind,deps=[],loop='',batch='')=>events.push({lane,id,label,start,end,kind,depends:deps,loop,batch:batch||(label.match(/[BM]\d/)||[''])[0]});
 if(mode==='pp'){
  const base=P.build('pp');duration=base.duration;
  const loops={postReq1:'P0:L0',submit01:'P0:L0',postProxy1:'P0:L0',postReq2:'P0:L1',submit02:'P0:L1',waitOut01:'P0:L1',d2h01:'P0:L1',process01:'P0:L1',postProxy2:'P0:L1',postRelay1:'P0:L2',waitOut02:'P0:L2',d2h02:'P0:L2',process02:'P0:L2',postRelay2:'P0:L3',gpu01:'P0:L0',gpu02:'P0:L1',waitProxySend1:'P0:L1',recvProxy1:'P1:L0',submit11:'P1:L0',postOut1:'P1:L0',recvReq2:'P1:L1',waitProxyMeta2:'P1:L1',recvProxy2:'P1:L1',submit12:'P1:L1',postOut2:'P1:L1',waitRelay1:'P1:L1',d2h11:'P1:L1',process11:'P1:L1',waitRelay2:'P1:L2',d2h12:'P1:L2',process12:'P1:L2',gpu11:'P1:L0',gpu12:'P1:L1',waitInput1:'P1:L0',waitInput2:'P1:L1'};
  // Local loop identities are stage-scoped. Wire activities belong to the
  // sending loop, with the receiving loop retained for inspection/highlighting.
  const messageLoops={req1:['P0:L0','P1:L0'],req2:['P0:L1','P1:L1'],proxy1:['P0:L0','P1:L0'],proxy2:['P0:L1','P1:L1'],out1:['P1:L0','P0:L1'],out2:['P1:L1','P0:L2'],relay1:['P0:L2','P1:L1'],relay2:['P0:L3','P1:L2']};
  for(const l of base.execution.lanes){lane(l.id,l.label.replace('Stage','PP'));for(const e of l.events){
   const message=messageLoops[e.id];
   add(l.id,e.id,e.label.replaceAll('MB','M'),e.start,e.end,e.kind,e.depends,message?.[0]||loops[e.id]||'',e.id==='waitProxySend1'?'M2':e.id.match(/[12]$/)?'M'+e.id.slice(-1):'');
   if(message)events[events.length-1].targetLoop=message[1];
  }}
  events.find(e=>e.id==='waitProxySend1').label='M2 等 M1 proxy 发送';
  // Existing PP lane durations for D2H describe CPU-side preparation/wait, not
  // another model computation. Preserve that ownership and make it explicit.
  for(const e of events)if(e.id.startsWith('d2h')){e.kind='wait';e.label=e.label.replace('D2H','等结果可读');}
  times=[0,2,2.4,6.7,8,10.2,11.2,12.8,17.2,17.8,18.7,20,21];barrierEvent='proxy1';
 }else{
  lane('cpu','CPU · Scheduler');lane('schedule','GPU · 元数据流');lane('gpu','GPU · 模型执行流');lane('copy',mode==='overlap'?'GPU · copy stream':'GPU → CPU · 结果可读');
  add('cpu','ingest','接收 R1',0,.6,'control',[],'L0');add('cpu','plan0','准备 B0',.6,1.6,'control',['ingest'],'L0');
  add('schedule','input0','B0 输入 / 地址',.8,1.6,'local',[],'L0');add('cpu','submit0','提交 B0',1.6,2,'control',['plan0'],'L0');
  add('gpu','forward0','B0 模型层',2,6.6,'compute',['input0','submit0'],'L0');add('gpu','sample0','B0 采样 / 接力 t1',6.6,7.2,'compute',['forward0'],'L0');
  if(mode==='normal'){
   duration=15;times=[0,1.6,2.1,7.2,7.6,8.4,9.4,12.8,13.2,14,15];
   add('copy','copy0','B0 结果可读',7.2,7.6,'tensor',['sample0'],'L0');add('cpu','wait0','等 B0 结果',2,7.6,'wait',['submit0'],'L0');add('cpu','process0','处理 B0',7.6,8.4,'control',['copy0','wait0'],'L0');
   add('cpu','plan1','准备 B1',8.4,9.4,'control',['process0'],'L1');add('schedule','input1','B1 输入 / 地址',8.7,9.4,'local',[],'L1');add('cpu','submit1','提交 B1',9.4,9.8,'control',['plan1'],'L1');
   add('gpu','forward1','B1 模型层',9.8,12.4,'compute',['submit1','input1','sample0'],'L1');add('gpu','sample1','B1 → EOS',12.4,12.8,'compute',['forward1'],'L1');
   add('copy','copy1','B1 结果可读',12.8,13.2,'tensor',['sample1'],'L1');add('cpu','wait1','等 B1 结果',9.8,13.2,'wait',['submit1'],'L1');add('cpu','process1','处理 EOS',13.2,14,'control',['copy1','wait1'],'L1','B1');add('cpu','idle','空闲检查',14,15,'control',['process1'],'L2');
  }else{
   duration=14.8;times=[0,1.6,2.1,3,3.8,7.2,7.8,8.4,10.3,11.6,14.2,14.8];
   add('cpu','plan1','提前准备 B1',2.1,3.4,'control',['submit0'],'L1');add('cpu','submit1','提交 B1',3.4,3.8,'control',['plan1'],'L1');
   add('schedule','war0','B1 等 B0 共享读',2.2,6.2,'wait',['submit0'],'L1','B1');add('schedule','input1','B1 元数据写入',6.2,6.6,'local',['war0'],'L1');
   add('gpu','forward1','B1 模型层',7.2,9.8,'compute',['submit1','input1','sample0'],'L1');add('gpu','sample1','B1 → EOS',9.8,10.2,'compute',['forward1'],'L1');
   add('copy','copy0','B0 copy_done',7.2,7.8,'tensor',['sample0'],'L0');add('cpu','wait0','等 B0 copy_done',3.8,7.8,'wait',['submit1'],'L1');add('cpu','process0','处理旧 B0',7.8,8.4,'control',['copy0','wait0'],'L1');
   add('cpu','plan2','提前准备 B2',8.4,9.4,'control',['process0'],'L2');add('cpu','submit2','提交 B2',9.4,9.8,'control',['plan2'],'L2');
   add('schedule','war1','B2 等 B1 共享读',8.5,9.2,'wait',['submit1'],'L2','B2');add('schedule','input2','B2 元数据写入',9.2,9.6,'local',['war1'],'L2');
   add('gpu','forward2','B2 在途工作',10.2,12.8,'compute',['submit2','input2','sample1'],'L2');add('gpu','sample2','B2 结果',12.8,13.2,'compute',['forward2'],'L2');
   add('copy','copy1','B1 EOS 可读',10.2,10.8,'tensor',['sample1'],'L1');add('cpu','wait1','等 B1 copy_done',9.8,10.8,'wait',['submit2'],'L2');add('cpu','process1','处理 EOS',10.8,11.6,'control',['copy1','wait1'],'L2','B1');
   add('cpu','empty','过滤完成 R1',11.6,12.2,'control',['process1'],'L3');add('copy','copy2','B2 结果可读',13.2,13.6,'tensor',['sample2'],'L2');add('cpu','wait2','等在途 B2',12.2,13.6,'wait',['empty'],'L3');add('cpu','process2','B2 排空 / 跳过 R1',13.6,14.2,'control',['wait2','copy2'],'L3');add('cpu','idle','空闲检查',14.2,14.8,'control',['process2'],'L4');
  }
  barrierEvent='copy0';
 }
 return {lanes,events,duration,times,barrierEvent};
}
function sample(mode,value=0,options={}){
 const scene=execution(mode);
 const barrier=scene.events.find(e=>e.id===scene.barrierEvent);
 const requested=Math.max(0,Math.min(scene.duration,Number(value)||0));
 // A teaching pause at the withheld dependency, not a simulated global stall.
 // Independent work is still visible at this instant; no delay is fabricated.
 const blocked=Boolean(options.hold&&!options.released&&requested>=barrier.start);
 const time=blocked?barrier.start:requested;
 const index=blocked?configs[mode].steps.findIndex(s=>s.id===configs[mode].barrier):scene.times.findLastIndex(t=>t<=time);
 const state=snapshot(mode,index,options);
 const ex={...scene,time,blocked,active:scene.events.filter(e=>time>=e.start&&time<e.end&&!(blocked&&e.id===barrier.id)),progress:Object.fromEntries(scene.events.map(e=>[e.id,Math.max(0,Math.min(1,(time-e.start)/(e.end-e.start)))]))};
 return {...state,time,execution:ex};
}
function executionSnapshot(mode,step,options={}){
 const scene=execution(mode),index=Math.max(0,Math.min(scene.times.length-1,Math.floor(Number(step)||0)));
 return sample(mode,scene.times[index],options).execution;
}

function validate(){for(const c of Object.values(configs)){const seen=new Set();for(const s of c.steps){if(seen.has(s.id))throw Error('Duplicate event');for(const d of s.deps)if(!seen.has(d))throw Error(`Invalid causal order ${s.id} <- ${d}`);for(const [lane]of s.a)if(!c.lanes.some(l=>l[0]===lane))throw Error('Unknown lane');if(!sources[s.source])throw Error('Unknown source');seen.add(s.id);}}return true;}
validate();return {revision,sources,sourceURL,configs,snapshot,execution,executionSnapshot,sample,validate};
});
