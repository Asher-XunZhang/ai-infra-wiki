/* Source-reviewed explanations; embedded into index.html by build_pp_step_guide.py. */
(function (scope) {
  'use strict';
  const specs = {};
  const define = (key, title, what, why, next, boundary, refs) => {
    specs[key] = {title, what, why, next, boundary, refs};
  };
  const batch = value => value ? `M${value}` : '无';
  const typeOf = id => id.split(':')[2];
  const list = nodes => [...new Set(nodes.map(n => n.owner))].join('、');
  const workBoundary = '源码调用 Work.wait() 并清空句柄列表；其对 CPU 或设备流的具体影响取决于通信后端。图中的等待宽度是依赖模型的示意值。';
  const layerBoundary = '源码按层等待 KV 就绪，允许回载与前向重叠。本图把它粗化成整份 H2D 完成后再画 GPU 前向，不能据此断言真实程序有整批屏障。';

  define('recv_req', '接收请求，建立本级调度状态',
    c => c.n > 0
      ? `${c.pp} 检查本轮接收的请求列表；固定示例的 M1–M5 已在开头到达，本轮列表为空，因此不会在这里新增请求状态。`
      : c.r === 0
      ? 'PP0 从请求接收器取得本轮新请求，处理输入并建立本级请求状态；PD Prefill 的新生成请求随后需要经过 bootstrap 握手准备。'
      : `${c.pp} 接收 PP${c.r - 1} 转发的请求，并在本级处理相同请求，建立供本级调度、缓存和 KV 发送使用的状态。`,
    '每个流水级只计算自己的模型层，却都需要知道在处理哪些请求、使用哪些缓存，以及把 KV 交给哪个 Decode 接收方。',
    c => `${c.last ? '本级继续推进请求状态' : '本轮收到的列表保留到本轮末尾，再向下一流水级转发'}；新请求能否成为可运行 batch，要继续经过 bootstrap 共识和选批。`,
    '收到请求不等于 GPU 已可运行。本例假设 M1–M5 在开头到达，后续很多轮收到的是空列表。', ['scheduler:2064', 'pp:327']);
  define('boot_poll', '检查与 Decode 的准备状态，汇总 bootstrap 候选',
    c => c.r === 0
      ? 'PP0 轮询 bootstrap 队列中各请求的 KV sender 状态，得到已准备好传输的 good 名单和失败的 bad 名单。'
      : `${c.pp} 接收上一级的候选名单，再结合本级轮询结果：good 取交集，bad 取并集；中止的请求也进入失败处理。`,
    '同一请求在各级都要准备好 KV 传输相关状态，才能让它一致地进入后续调度；某一级失败也必须让其他级知道。',
    '候选沿 PP0→PP1→PP2 汇总，之后还要通过 bootstrap 共识回流，成功请求才会移入 waiting_queue。',
    '这是读取当前状态的一次轮询，不是原地等所有请求 ready；bootstrap 也不是 L2 缓存加载完成。', ['pp:593', 'pp:305']);
  define('term_poll', '检查 KV 传输是否进入终态',
    c => `${c.pp} 轮询已经处于 inflight 队列的请求，挑出 sender 报告 Success 或 Failed 的请求${c.r ? '，并与上一级传来的终态名单取交集' : ''}。`,
    '本级发完 KV 不代表其他流水级也发完；释放 Prefill 侧资源前，需要逐级汇总传输终态，避免仍在发送的级过早清理。',
    '终态候选继续向末级汇总，随后形成 release 回流名单；真正清理由后面的 release 步骤完成。',
    '终态包含成功和失败，源码分别处理。本图只建模成功路径；传输终态不代表 Decode 已生成完全部回答。', ['pp:635', 'prefill:972']);
  define('chunk', '整理上一批和未完成的 Prefill 分块',
    '检查是否有未完成的 chunked request，必要时缓存已完成的前缀、推进该分块的 KV 发送，并从上一批中滤掉不应继续保留的请求。',
    '长 prompt 可以分多轮计算；选下一批之前必须保存旧进度并整理 batch，避免把已经处理过或应暂时移出的请求再次计算。',
    '调度器获得整理后的上一批与分块状态，再处理缓存事件并选择本轮的新 batch。',
    '本图采用完整 Prefill、没有中间 chunk，因此这里只表示检查和整理路径，不能把这段看成又执行了一次 GPU 前向。', ['prefill:1210']);
  define('l2_counts', '统一这一轮可以处理多少条缓存 ACK',
    c => c.r === 0
      ? 'PP0 统计缓存 write/load 完成队列中从队首开始连续就绪的 ACK 数，将计数合并同步；级内参与进程取最小值，然后把结果向后续 PP 级传播。'
      : `${c.pp} 接收由 PP0 发布、上一级转来的缓存完成计数；如果还有下一流水级，继续异步转发相同计数。`,
    '各级的拷贝进度可以不同，但缓存完成队列必须按一致的数量和顺序消费，否则缓存节点的生命周期可能分叉。',
    'writing_check / loading_check 按公布的数量处理 ACK；每一级仍要确认对应的本地完成事件。',
    '这里不是对所有 PP 的完成时间取最小值。PP0 发布计数，其他级可能随后等待本地 event。本例没有淘汰写回，write=0；空 batch 轮也要参加。', ['cache:3073', 'cache:333', 'cache:3290']);
  define('ack_events', '确认本地回载完成，收尾缓存 ACK',
    c => c.acks.length
      ? `${c.pp} 按本轮公布的数量处理 ${list(c.acks)} 的 load ACK，逐个同步本地 finish_event，移除 ongoing_load_back 记录并解除这次回载的临时保护。`
      : `${c.pp} 检查本轮缓存完成计数；当前 load ACK 数为 0，没有需要同步并收尾的回载条目。`,
    '收到统一计数只说明该处理这些队首条目了，本级数据未必已经复制完；先等本地事件，再解除临时保护，才能安全管理缓存。',
    '已确认的条目退出回载中的状态，缓存节点可以按正常引用和淘汰规则管理，调度继续进入选批。',
    '解除的是回载期间的临时引用，不是把正在服务请求的 KV 全部释放。当前请求对前缀的使用引用是另一层生命周期。', ['cache:3187', 'cache:3210']);
  define('select', '匹配前缀并判断本轮是否可以选批',
    c => c.current
      ? `本例本轮选择 ${batch(c.current)}：先匹配可复用的 KV 前缀，再检查 Prefill token、显存等预算，决定本轮接纳的计算范围，并通过延迟准入检查。`
      : '本例这一轮没有选出可计算的请求，get_new_batch_prefill 返回 None；调度仍继续推进已有请求的状态、通信与清理。',
    '请求已经到达、完成 bootstrap，也可能因预算或调度条件不能立刻运行。先判断可接纳范围，才能避免超额分配资源。',
    c => c.current ? '通过预检查后，才准备必要的 host 缓存回载，并正式提交准入。' : '跳过本轮新 batch 的准备和前向；后续旧结果处理或控制通信仍可能有工作。',
    '本图固定 L4–L8 每轮选 M1–M5，假设容量充足；改时间不会重新模拟真实预算竞争、选批策略或请求到达。', ['scheduler:3959', 'policy:1185']);
  define('init_load', '为 host 命中的前缀准备回载',
    c => c.hasLoad
      ? `${c.owner} 的一部分前缀 KV 只在主机内存中。调度器调用 init_load_back，为回载准备 GPU 位置和缓存节点保护，取得可接入请求前缀的设备索引。`
      : `${c.owner} 在本例没有需要回载的 host 命中，因此跳过 init_load_back；这段表示分支检查，不会产生 KV 拷贝。`,
    '命中主机缓存可以少算一段前缀，但 GPU 做 attention 前还需要对应 KV 和有效的设备地址。',
    '回载准备成功后，把设备索引纳入请求的前缀，再提交准入；实际异步 H2D 由后面的 start_loading 发起。',
    '“拿到设备索引”不代表数据已复制完成。本图只有 M2、M4 有 host hit，其他 batch 不应被讲成也在回载。', ['policy:1208', 'cache:3243']);
  define('commit_admission', '正式接纳请求并保护它要使用的前缀',
    c => `把 ${c.owner} 的本轮计算范围写入请求，增加前缀使用引用，将请求加入 can_run_list，并扣减这一轮的调度预算。`,
    '前面的预算判断和回载准备通过后，才可以正式占用预算并保护前缀，防止后续选批或缓存回收破坏这次计算。',
    '调度器随后将已接纳的请求构造成 ScheduleBatch，进入缓存提交和输入准备。',
    '这是准入状态提交，不是 GPU 提交；它在回载准备成功之后、实际 H2D 提交之前。', ['policy:1347', 'scheduler:4037']);
  define('start_load', '构造 batch，提交已准备的缓存回载',
    c => c.hasLoad
      ? `调度器将已接纳的 ${c.owner} 构造成 batch。start_loading 合并待回载任务，记录 start_event，提交 H2D，并把完成事件放进 ACK 队列；batch 保存 consumer index。`
      : `调度器构造 ${c.owner} 的 batch 并检查回载队列；本例该队列为空，start_loading 返回 -1，不提交 H2D。`,
    '把准备好的缓存任务交给异步拷贝流，同时保留批次与逐层完成事件的对应关系，计算时才能等待正确的数据。',
    'CPU 继续准备输入；若确有回载，H2D 独立推进，完成后还需要在后续缓存事件轮消费 ACK。',
    '提交异步拷贝不等于拷贝完成；consumer index 用来定位这次回载的事件组，不是 micro-batch 编号。', ['scheduler:4054', 'controller:941', 'controller:960']);
  define('prepare_extend', '把请求整理成可执行的 Prefill 输入',
    c => `为 ${c.owner} 准备本次需要计算的 token、序列长度、已命中前缀长度以及新 KV 的写入位置等 batch 字段，并保存到当前 PP 槽位。`,
    '模型执行需要连续的输入张量和明确的缓存位置，不能直接拿调度队列里的请求对象进行前向。',
    c => c.r ? '接下来接收上一级传来的中间激活，再提交本级计算。' : 'PP0 不需要接收上级激活，准备好后可进入本级前向提交。',
    '这里准备的是本轮未被缓存覆盖的计算输入；部分输入搬运会延迟到 forward stream，不能把准备阶段结束等同于所有设备工作结束。', ['scheduler:4058', 'pp:256']);
  define('recv_proxy', '接收上一级算好的中间激活',
    c => `${c.pp} 从 PP${c.r - 1} 接收 ${c.owner} 的 proxy 张量（hidden states，即中间特征），并包装成供本级模型继续计算的输入。`,
    'PP 把模型层拆在不同级上；后一级没有前面层的输出，就无法继续执行自己持有的后续层。',
    '当前 batch 的上级输入就绪，随后在本地提交前向；同一通信通道中的 output 消息由类型标记区分。',
    'proxy 是 Prefill 流水级之间的中间激活，既不是发给 Decode 的 KV，也不是最终 token 结果。PP0 没有这一步。', ['pp:858', 'pp:825']);
  define('launch', '向 GPU 流提交本级前向',
    c => `${c.pp} 为 ${c.owner} 在 forward stream 上安排 run_batch，保存本批结果所需的元数据，并在流上记录 launch_event${c.last ? '；末级还把事件与 output 放入待发送队列' : ''}。`,
    'CPU 调度与 GPU 计算异步推进；后续发送激活或结果时，需要一个事件来保证不会读到尚未计算完成的数据。',
    'CPU 可以继续这一轮的结果、共识或转发工作；GPU 在自己的流依赖满足后实际执行。',
    '记录 event 只是安排完成标记，不能理解为 CPU 此刻已经等到 GPU 算完；图中 launch 和 GPU 前向是两段不同工作。', ['pp:1221', 'pp:1247']);
  define('gpu', '执行当前 batch 在本级的模型层',
    c => `${c.pp} 的 GPU 实际计算 ${c.owner} 在本级持有的层，读取所需前缀 KV，并产生本级新增 KV${c.last ? '以及末级输出，供后续得到 token 结果' : '和传给下一级的中间激活'}。`,
    '同一 batch 必须依次通过 PP0、PP1、PP2 的模型层，才能得到完整 Prefill 结果；各级也因此拥有需要交给 Decode 的本地 KV。',
    c => c.last ? '对应完成事件可以放行 output 发送；各级拿到旧 batch 的结果后再推进本地 KV 发送。' : `前向完成后，${c.owner} 的中间激活可以传给 PP${c.r + 1}，让后续层继续计算。`,
    c => c.hasLoad ? layerBoundary : '条宽是本级这一批的示意服务时间，不包含此前所有 CPU 准备或输入等待；也不是该请求的端到端耗时。', ['pp:1229', 'controller:64']);
  define('h2d', '把命中的前缀 KV 从主机搬回 GPU',
    c => `${c.pp} 的拷贝流将 ${c.owner} 命中的主机缓存 KV 从 L2 搬到 GPU HBM，按层记录完成事件。`,
    '复用已有前缀能够避免重复计算，但 attention 执行时需要相应 KV 在设备上可用。',
    '逐层事件让计算知道哪些层已可使用；最终完成事件则由后面的 load ACK 检查消费，用于缓存状态收尾。',
    `${layerBoundary} 图中同一拷贝资源的串行安排也是模型简化。`, ['controller:960', 'controller:64', 'cache:3187']);
  define('send_proxy', '安排把本级激活发给下一流水级',
    c => `${c.pp} 先在发送所在流上等待 ${c.owner} 的 launch_event，再异步提交带 proxy 类型的张量发送到 PP${c.r + 1}，保存 send_proxy_work。`,
    '发送必须排在本级前向产出激活之后；通过流事件建立先后关系，可以让 CPU 继续调度，而不必在此同步等待 GPU 完成。',
    '本级 GPU 输出就绪后，激活传输才能真正推进；发送句柄留到后续轮次回收。',
    'wait_event 是设备流依赖，不是 event.synchronize()。本步骤结束只能说明发送已安排，不能说明对方已收到。', ['pp:336', 'pp:340']);
  define('proxy_message', '中间激活传到下一流水级',
    c => `${c.pp} 正在把 ${c.owner} 在本级算出的 hidden states（中间特征张量）传给 PP${c.r + 1}。模型要求发送已提交，且本级 GPU 前向已经产出这些数据。`,
    c => `PP${c.r + 1} 只持有后续模型层，必须拿到这份输入，才能继续同一个 ${c.owner} 的计算。`,
    c => `PP${c.r + 1} 的 recv_proxy 可以接收这份激活，再推进本级前向。`,
    '这是 Prefill 级间激活传输，不是 PD KV 传输，也不是 output 回流。条宽代表模型中的在途时长，不是 CPU 调用 send 的耗时。', ['pp:340', 'pp:858']);
  define('send_out', '提交或转发末级产出的 output',
    c => c.last
      ? `PP2 取出 ${c.owner} 的前向完成事件和输出，在发送流上等待该事件，再异步发往 PP0。`
      : `${c.pp} 把之前收到并保存的 ${c.owner} output 转发给 PP${c.r + 1}；这里使用的是保存的 pp_outputs，并不是重新做一次末级计算。`,
    '只有末级拥有完整模型输出，但每一级都要用同一结果更新本地请求并推进 KV 发送，所以结果沿 PP2→PP0→PP1→PP2 回流。',
    '数据在依赖满足后传输，接收级把它关联到相应的旧 batch，发送 work 留待后续回收。',
    'output 主要携带 next_token_ids 等结果字段；它和顺着 PP0→PP1→PP2 流动的 proxy 激活是两类消息。', ['pp:983', 'pp:769']);
  define('out_message', 'output 沿结果环路传输',
    c => `${c.owner} 的 output 正从 ${c.pp} 传往 PP${(c.r + 1) % c.data.pp}。${c.last ? '这份数据来自末级前向，必须先等其输出就绪。' : '这是对已收到结果的接力转发。'}`,
    '让持有不同模型层和本地 KV 的各级，对同一请求使用一致的 token 结果进行后处理。',
    '接收级的 recv_out 得到旧 batch 的输出，准备结果对象和就绪事件，再进入请求后处理。',
    '这里传的是结果张量，不是把 Prefill 的全部 KV 绕 PP 环传一遍；KV 另由各级发送给 Decode。', ['pp:983', 'pp:1072']);
  define('recv_out', '接收要在本轮处理的旧 batch 结果',
    c => `${c.pp} 从 PP${(c.r + c.data.pp - 1) % c.data.pp} 接收 ${c.owner} 的 output，使用该旧 batch 保存的元数据解释结果，得到 next_token_ids 等字段。`,
    '本轮提交的可能是另一份新 batch；旧 batch 的结果稍后才沿环路返回，需要根据槽位和元数据找到正确的请求。',
    '接着在 copy stream 上预处理结果并记录就绪事件；等到后处理位置再同步事件、更新旧请求。',
    '本例 depth=0，结果接收位于当前前向提交之后；不能把这里的旧结果默认当作本轮刚 launch 的 batch。', ['pp:1072', 'pp:1084']);
  define('copy', '预处理旧结果并记录就绪事件（图中 D2H）',
    c => `为 ${c.owner} 构造 GenerationBatchResult，处理结果字段，并在 copy stream 上记录 d2h_event；该流先等待 schedule_stream 的先行工作。`,
    '后续 CPU 请求处理要使用这份结果，需要一个明确的完成点，确保预处理及按配置发起的异步拷贝已达到可使用的状态。',
    '后面的 wait_copy 同步这个事件，然后进入 process_batch_result_disagg_prefill。',
    c => `图中 D2H 是结果预处理／就绪区间的简写。源码根据输出配置决定拷贝内容；本例关闭 sampling mask，相关分支只记录 event，不能理解为这里必然完整复制 token 张量，更不是 KV 回载。${c.last && c.current ? '末级 copy stream 还会继承等待当前前向完成事件的流依赖。' : ''}`,
    ['pp:1084', 'pp:1091', 'pp:1100']);
  define('wait_copy', '在使用旧结果前同步完成事件',
    c => `${c.pp} 调用 d2h_event.synchronize()，确认 ${c.owner} 的结果就绪事件已经完成，才允许 CPU 进入后处理。`,
    '异步流已经提交任务并不保证数据可读；这里是从异步结果准备跨到 CPU 使用结果的明确同步点。',
    '读取旧结果、更新请求并推进其 Prefill KV 发送。',
    '若事件早已完成，这个调用可以很快返回；若尚未完成，图中此前的斜线等待段表示依赖尚未满足的区间，勿与此调用的服务段重复计算。', ['pp:317', 'prefill:800']);
  define('result', '把旧结果写回请求，转入等待 KV 传输阶段',
    c => `处理 ${c.owner} 的 Prefill 结果：把 next_token_ids 转成 CPU 可用的值，更新请求的 output_ids，按缓存策略保存未结束请求的前缀，并加入 disagg_prefill_inflight_queue。`,
    'GPU 返回的是 batch 结果，调度器需要把它落到逐个请求；这些请求的 KV 还要交给 Decode，因此此时不能直接丢弃本地状态。',
    '正常完整 Prefill 路径继续提交本级 KV；请求留在 inflight 队列中，等待后续终态轮询与清理。',
    '源码在同一个结果处理函数内推进这些动作；图为便于阅读把后面的 send_kv 单独画出。加入 inflight 不代表传输已经完成。', ['prefill:781', 'prefill:800', 'prefill:859']);
  define('send_kv', '把本级 KV 交给传输后端发往 Decode',
    c => `为 ${c.owner} 找出本级已计算好的 KV 范围，转换为传输需要的物理页索引，交给 disagg_kv_sender.send；完整 Prefill 在这里提交最后一块及所需附带状态。`,
    'Decode 要接着生成 token，需要继承 Prefill 已计算的上下文 KV。每个 PP 级负责自己持有层的 KV。',
    '传输后端异步推进 KV 发送，后续通过 sender.poll() 观察是否完成。',
    '目的地是图外的 Decode 接收端，不是下一 PP 级；本步骤结束表示发送已提交，资源释放还要等终态和共识。', ['prefill:893', 'prefill:1502']);
  define('kv', '本级 KV 正在传向 Decode',
    c => `${c.pp} 已为 ${c.owner} 提交 KV，传输后端正在把这些缓存页送往 Decode。图把它表示为一段异步在途区间。`,
    '计算与传输可以分开推进；只看 GPU 前向结束，无法判断 Prefill 请求什么时候可以安全清理。',
    '本地传输终态会在后续轮询中被观察，经过 PP 共识回流和本地复查后才允许清理。',
    '本图假设每级的 KV 在途段串行，真实后端的传输并发与链路争用未建模。条段结束既不等于马上释放，也不等于 Decode 生成结束。', ['prefill:1502', 'pp:635', 'prefill:972']);
  define('send_bc', '将 bootstrap 共识沿环路传回各级',
    c => c.last
      ? 'PP2 将逐级汇总后的 bootstrap good/bad 名单发回 PP0，开始共识回流。'
      : `${c.pp} 将上一轮已经收到并保存的 bootstrap 共识转发给 PP${c.r + 1}。`,
    '各级需要看到同一份成功和失败名单，才能一致地把成功请求放入可调度队列，并处理失败请求。',
    '对应级收到名单后调用 process_bootstrapped_queue；发送 work 还需要在本轮后面的 commit_bc 回收。',
    '发送的是请求 ID 和状态名单，不传激活或 KV；名单来自队列快照，不能按“本轮当前 batch”理解其成员。', ['pp:660', 'pp:305']);
  define('recv_bc', '按 bootstrap 共识把请求移入等待队列',
    c => `${c.pp} 接收共识 good/bad 名单，调用 process_bootstrapped_queue 处理 bootstrap 队列：成功请求进入 waiting_queue，失败请求走对应处理。`,
    '把“各级已同意准备就绪”转换成真正可参与选批的调度状态，避免某一级先计算其他级还没准备好的请求。',
    '成功请求可以在后续选批中被接纳；因为本轮的选批位置已经过去，所以不是收到名单就立即补做本轮前向。',
    c => c.n === 2 ? '固定示例在 L3 让 M1–M5 完成这次准入，L4 才开始选 M1；真实服务的名单和时机随请求状态变化。' : '固定示例后续没有新准入，仍可传递和处理空名单；本步骤不表示每轮都增加新请求。', ['pp:305', 'pp:589']);
  define('send_rc', '将允许复查清理的终态名单传回各级',
    c => c.last
      ? 'PP2 把逐级取交集后的传输终态名单发回 PP0，启动 release 共识回流。'
      : `${c.pp} 将之前收到并保存的 release 名单转发给 PP${c.r + 1}。`,
    '各级需要协调请求的传输生命周期；仅凭某一级的发送完成就回收资源，可能早于其他级的完成时刻。',
    '各级接收名单后，还会在本地 inflight 队列中复查状态，确定本轮实际能清理的请求。',
    '名单中的对象可能是更早的请求，与本轮当前 batch 或正在处理的旧 batch 没有固定一一对应关系。', ['pp:683', 'prefill:972']);
  define('recv_rc', '接收 release 名单，限制本轮清理范围',
    c => `${c.pp} 接收对应槽位回流的请求 ID 名单，保存为 next_release_rids，供后面的 inflight 队列清理使用。`,
    '将各级已经汇总的终态信息带回本地，限定哪些请求可以进入本轮的清理检查。',
    '先完成本轮需要的旧结果处理，再用这份名单调用 process_disagg_prefill_inflight_queue。',
    '收到名单尚未释放资源；名单也可能为空。只有本地复查仍满足条件的请求才会被清理。', ['pp:312', 'prefill:972']);
  define('release', '复查传输状态，清理 Prefill 请求资源',
    c => c.released.length
      ? `${c.pp} 本轮实际清理 ${c.released.map(batch).join('、')}：在 release 名单限定的范围内复查本地 sender，成功后依次 release_kv_cache、finish(SUCCESS)、sender.clear，归还元数据并移出 inflight 队列。`
      : `${c.pp} 检查 release 名单和本地 inflight 状态；当前场景这一轮没有实际清理的请求，未满足条件的请求继续保留。`,
    '避免 KV 仍在被传输后端使用时回收请求资源，同时及时归还已经完成 Prefill 交接的请求占用。',
    '清理完成的请求退出本地传输生命周期；未完成的请求留待后续轮询，Decode 端继续自己的生成流程。',
    '释放请求引用不等于所有前缀缓存都立即从显存消失。当前默认缓存的 finish(SUCCESS) 不取消异步缓存工作；图只表示成功路径。', ['prefill:972', 'prefill:1025', 'prefill:1074']);
  define('end', '保存回流状态，结束这一轮槽位迭代',
    '把本轮收到的 output、bootstrap 共识和 release 名单保存到调度器状态，更新 batch_is_full 等标记，然后进入下一次槽位迭代。',
    '结果和共识沿流水线分轮传播，下次转发需要使用这次保存的状态；环形槽位也要继续被复用。',
    'CPU 推进下一轮接收和状态检查，已经提交的 GPU、拷贝与通信任务仍按各自依赖异步运行。',
    'loop 结束不等于当前 batch 结束，更不等于服务器所有任务结束。L# 是本级迭代编号，不是全局同步屏障。', ['pp:346', 'pp:223']);

  const controls = {
    req: {name:'请求', payload:'本轮收到的请求列表', target:'recv_req', ref:'pp:327', reason:'使后续流水级也建立相同请求的本地调度状态'},
    boot: {name:'bootstrap 候选', payload:'本级汇总的 bootstrap good/bad 候选名单', target:'boot_poll', ref:'pp:331', reason:'让下一级结合自己的准备状态继续求 good 交集、bad 并集'},
    term: {name:'transfer 终态候选', payload:'本级汇总的 KV 传输终态候选名单', target:'term_poll', ref:'pp:334', reason:'让下一级结合本地传输状态继续求交集，最终形成 release 名单'},
  };
  for (const [key, control] of Object.entries(controls)) {
    define('tail_'+key, `提交${control.name}的逐级转发`,
      c => `${c.pp} 把${control.payload}异步发送给 PP${c.r + 1}，并保存发送 work，供之后确认完成。`,
      control.reason+'。',
      '消息在通信路径上推进，CPU 可以继续本轮剩余工作；后续轮次再回收发送句柄。',
      '这是队列级控制消息，名单可能为空，也可能涉及多份请求；其成员不由当前前向 batch 决定。', [control.ref, 'pp:735']);
    define(key+'_message', `${control.name}消息正在传输`,
      c => `${control.payload}正从 ${c.pp} 传往 PP${c.r + 1}；这是前面异步发送提交后的在途阶段。`,
      control.reason+'。',
      c => `PP${c.r + 1} 的 ${control.target} 可以消费消息，继续本级请求或状态处理。`,
      '图中的在途段是教学模型单独拆出的通信时长，不是源码中又调用了一个同名 CPU 函数；控制消息不携带完整激活或 KV。', [control.ref]);
  }
  const works = {
    req_work: ['上一轮请求发送', '避免未完成的请求发送句柄无限累积，再保存新一轮发送任务', 'pp:238'],
    boot_work: ['上一轮 bootstrap 候选发送', '维持候选消息的有序推进，再保存新一轮发送任务', 'pp:242'],
    term_work: ['上一轮 transfer 终态候选发送', '有序推进传输状态消息并回收通信句柄', 'pp:245'],
    proxy_work: ['上一轮激活发送', '确认此前提交的激活发送达到后端的完成条件，再推进当前 batch 的提交', 'pp:272'],
    out_work: ['上一轮 output 发送', '回收旧结果发送句柄，再启动这一轮的结果收发', 'pp:710'],
    commit_bc: ['本轮 bootstrap 共识发送', '确认共识消息的发送工作得到处理，避免留下未回收的通信句柄', 'pp:311'],
    commit_rc: ['本轮 release 共识发送', '确认终态回流消息的发送工作得到处理，再继续本地后处理', 'pp:314'],
  };
  for (const [key, [name, reason, ref]] of Object.entries(works)) {
    define(key, `确认并回收${name}`,
      c => `${c.pp} 对${name}保留的 work 调用 wait()，随后清空列表。这里处理的是已提交的通信，不重新发送一份数据。`,
      reason+'。',
      '调度继续执行后续步骤；下方跨 PP 的前置关系可以追到对方对应的接收操作。', workBoundary, [ref, 'pp:705']);
  }
  define('l2_drain', '回收上一轮缓存计数的异步发送',
    c => `${c.pp} 在开始新的缓存事件轮之前，等待 work_list 中上一轮缓存计数发送的 work，并清空这个列表。`,
    '限制未完成发送的积累；当下游流水级落后时，通过这里形成反压，使缓存控制消息按轮推进。',
    '然后发布或接收本轮 write/load 完成计数，再处理本地缓存 ACK。',
    '这里回收的是缓存计数通信，不是等待所有 KV 回载，也不是请求的 PD 传输释放。', ['cache:321', 'cache:3290']);

  function describe(data, node, event = node) {
    const op = typeOf(node.id), spec = specs[op];
    if (!spec) throw new Error(`Missing step explanation: ${op}`);
    const graph = data.graph;
    const loop = data.loops.find(l => l.r === node.r && l.n === node.n);
    const related = (node.deps || []).map(id => graph[id]).filter(Boolean);
    const c = {data, node, r:node.r, n:node.n, pp:`PP${node.r}`, last:node.r === data.pp - 1,
      owner:node.owner, current:loop?.current, old:loop?.old,
      hasLoad:!!graph[`${node.r}:${node.n}:h2d`],
      acks:related.filter(n => n.kind === 'h2d'),
      released:data.releases.filter(x => x.r === node.r && x.n === node.n).flatMap(x => x.batches)};
    const read = value => typeof value === 'function' ? value(c) : value;
    const result = {op, title:read(spec.title), what:read(spec.what), why:read(spec.why), next:read(spec.next), boundary:read(spec.boundary), refs:[...spec.refs]};
    result.context = `${c.pp} · L${node.n + 1}：本轮前向 ${batch(c.current)}；本轮旧结果 ${batch(c.old)}。所选步骤的对象：${node.owner}。`;
    if (op === 'term_poll') {
      const candidates = (node.inputs || []).map(x => graph[x.id]).filter(n => n?.kind === 'kv');
      result.context += candidates.length ? `当前模型候选包含 ${list(candidates)}，只说明该轮已观察到相应终态。` : '当前模型这一轮没有进入终态候选的请求。';
    }
    if (op === 'l2_counts') {
      const ack = graph[`${node.r}:${node.n}:ack_events`];
      const loads = (ack?.deps || []).map(id => graph[id]).filter(n => n?.kind === 'h2d');
      result.context += `本轮公布 write=0、load=${loads.length}${loads.length ? `（${list(loads)}）` : ''}；这些条目的本地完成情况在后续 ACK 步骤确认。`;
    }
    if (['send_bc','send_rc'].includes(op)) {
      const input = node.inputs?.[0];
      if (input) { const p = graph[input.id]; result.context += `这次发送使用 PP${p.r} · L${p.n + 1} 的${c.last ? '本轮候选' : '已收共识'}。`; }
    }
    if (event.kind === 'wait') {
      const blockers = (event.deps || []).map(id => graph[id]).filter(Boolean);
      result.title = '等待前置条件：'+result.title;
      result.wait = `本地前一步已经走完，但所选操作尚不能开始。当前等待：${blockers.map(p => `PP${p.r} · L${p.n + 1} 的 ${p.label}（${p.owner}，${p.end.toFixed(2)} u 完成）`).join('；')}。`;
      result.next = `这些前置条件满足后，才从 ${node.start.toFixed(2)} u 开始所选操作。${result.next}`;
      result.boundary += ' 斜线区间是模型推导出的依赖等待，不是额外执行的一遍操作，也不能直接当作实测 CPU 阻塞。';
    } else if (event.kind === 'idle') {
      const layer = event.id.endsWith(':layer-wait');
      result.title = layer ? `GPU 等待 ${node.owner} 的缓存数据` : `GPU 尚未开始 ${node.owner} 的前向`;
      result.wait = layer
        ? `当前 batch 的 launch 已完成，但本图仍在等本级 H2D 完成，才允许 ${node.owner} 的 GPU 前向开始。`
        : `本级当前没有执行图中的 GPU 前向，${node.owner} 还没有完成提交所需的调度和准备。此区间 CPU 可能在接收消息、检查缓存、选批或等待依赖；不能用一个原因概括整段空档。`;
      result.boundary = layer ? layerBoundary : '这是本图两段 GPU 前向之间的未计算区间，不表示设备没有任何通信或拷贝活动，也不是源码中的单独“idle”函数。';
      result.refs = [...new Set([event.ref, ...result.refs])];
    }
    return result;
  }
  const api = {describe, operationTypes:Object.keys(specs)};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else scope.PP_STEP_GUIDE = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
