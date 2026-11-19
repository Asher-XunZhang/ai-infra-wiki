/* Reading notes follow the selected diagram step. Numeric examples are illustrative. */
window.AtlasLessonNotes = (() => {
  const h = React.createElement;
  const local = (en, zh, english) => en ? english : zh;
  const variable = (base, subscript) => subscript ? h('msub', null, h('mi', null, base), h('mtext', null, subscript)) : h('mi', null, base);
  const operator = value => h('mo', null, value);
  const number = value => h('mn', null, value);
  const row = (...parts) => h('mrow', null, ...parts);
  const fraction = (numerator, denominator) => h('mfrac', null, numerator, denominator);
  const Formula = ({ children }) => h('div', { className: 'ia-formula' }, h('math', { display: 'block' }, children));

  const notes = {
    process: [
      ['沿着同一个请求 ID 追踪。入口先把文本变成 token ID，并附上采样参数。调度器收到的是一笔待执行工作，有了请求 ID 就能把后续几轮的输出归回同一请求。', 'Follow one request ID. The frontend converts text to token IDs and attaches sampling settings. The scheduler receives a work item whose ID also identifies its output across later steps.'],
      ['区分“到达队列”“选入本轮”“已经算完”。选入时先确定本轮要算多少 token、占用哪些 KV 位置。切到「职责与 KV 归属」对照这时的地址和下一步才出现的 K/V。', 'Distinguish queue arrival, admission to this step, and completion. Admission assigns token work and KV locations. In Roles & KV ownership, compare the newly reserved addresses with K/V data written in the next step.'],
      ['同一次 forward 中，TP rank 协作处理同一批请求。每张卡执行各自那份模型并写入本地 KV。把右侧职责图从 TP=2 切到 TP=1，可以观察 vLLM 执行器的进程边界也随之改变。', 'In one forward pass, TP ranks cooperate on the same requests. Each GPU runs its model partition and writes local KV. Switching the ownership view from TP=2 to TP=1 also changes the vLLM executor’s process boundary.'],
      ['采样先得到 token ID，反分词随后把能显示的内容拼成文本。流式场景还得经过 HTTP 发送，客户端才读得到。记住这三个事件，最后一课会用它们定位 TTFT 的停表点。', 'Sampling produces a token ID. Detokenization then turns available content into text, and streaming still needs HTTP delivery before the client can read it. These three events locate the TTFT endpoint in the final topic.'],
    ],
    wire: [
      ['从图里同时找三件事：哪两个进程连上了、连接用了什么通道、通道传的是什么。这里同机 ipc:// 是连接端点，ZMQ socket 决定消息怎样交给对端，序列化才决定请求怎样编码。', 'For each arrow, find the two processes, the channel, and its payload. Local ipc:// identifies an endpoint, the ZMQ socket governs delivery, and serialization encodes the request.'],
      ['沿回程箭头看载荷在哪里变了。SGLang 先把 token ID 交给 Detokenizer，再把文本交回入口；vLLM 在前端处理 EngineCoreOutputs。两边都有反分词，所在的进程不同。', 'Follow the return arrows to see where the payload changes. SGLang sends token IDs to the detokenizer, which returns text to the frontend. vLLM processes EngineCoreOutputs on the frontend side. Both detokenize, in different processes.'],
      ['以 TP=2 为例，请求内部有两种交接：Executor 把本轮工作交给 Worker 属于调度到执行的控制交接，rank 间同步模型中间结果才属于 GPU 通信。共享内存队列与 NCCL 各对应一层。', 'At TP=2, the executor hands a work description to workers on the control path. Ranks synchronize model intermediates on the GPU path. The shared-memory queue and NCCL belong to these different layers.'],
      ['把同机请求转发、rank 间张量同步、跨实例 KV 搬运分别沿箭头看。取消请求传的是请求 ID 和控制意图。RDMA 即使参与 P/D 数据传输，接收方仍需要自己的控制消息来确认可用。', 'Trace local request delivery, rank-to-rank tensor synchronization, and cross-instance KV transfer separately. Cancellation carries an ID and control intent. Even if RDMA carries KV between P/D instances, readiness still needs control coordination.'],
    ],
    prefix: [
      ['先区分索引、地址与实存。相同前缀提供查找线索，命中后得到页或物理块编号，再用编号访问 GPU 缓冲区里的 K/V。索引命中因此可省去已复用 token 的前向工作。', 'Separate index, address, and storage. A matching prefix supplies a lookup key, which resolves to page or block IDs used to access K/V in GPU buffers. The hit avoids recomputing the reused tokens.'],
      ['本例连续相同的前缀有 L=10 个 token，每页或块有 B=4 个 token。两边都按完整单位复用，向下取整后只有两组。位置 9–10 和 11–12 在同一组，后两项不同使整组需要补算。', 'The example shares L=10 consecutive tokens and stores B=4 tokens per page or block. Both sides reuse complete units here, so rounding down gives two units. Positions 9–10 share a unit with 11–12, and the mismatch forces that full unit to be recomputed.'],
      ['两个 12-token 请求共享前两组地址，第三组各有自己的地址。这样新尾部可以写入自己的 K/V，也能继续读取共享的历史。两边的页号或块号无需相同。', 'The two 12-token requests share their first two address units and each has its own third unit. The new tail can write its own K/V while reading the shared history. Page or block IDs need not match across engines.'],
      ['Decode 追加一个位置时，先看尾页或尾块是否有空位，再决定是否需要新地址。正在使用的共享前缀仍受保护，请求结束之后是否可回收会在第 05 节说明。', 'When decode appends a position, check for room in the tail page or block before allocating a new address. Active shared prefixes remain protected. Topic 05 explains when they can be reclaimed after completion.'],
    ],
    batch: [
      ['每轮重新确定成员与本轮工作量。图里等待新请求要做 Prefill，已经在运行的请求可能只需推进一个 Decode 位置，因此同一轮可以给不同请求安排不同数量的 token。', 'Each step chooses its membership and assigns work. A waiting request needs prefill while an active request may only advance one decode position, so requests can use different token counts in the same step.'],
      ['图中 vLLM 把 Decode 和 Prefill 放在同一条预算带里。假设两个 Decode 请求各算 1 个 token，再安排 4 个 Prefill token，本步就用掉 6 个 token 的额度。还需要为这批申请可用的 KV 空间，所以 token 额度有余时也可能暂缓。', 'On the vLLM side, decode and prefill share one budget bar. Suppose two decode requests use one token each and prefill needs four tokens. The step uses six tokens of its budget. This batch also needs available KV space, so spare token budget alone does not guarantee admission.'],
      ['读两条轨道时先跟同一批的依赖：CPU 准备好输入，GPU 才能执行这批。执行期间，CPU 可以处理已返回的前一批，或者准备满足依赖的后续工作。PP 流水还要有多个 stage，本章默认 PP=1。', 'Follow the dependencies for one batch first: the CPU prepares its input before GPU execution. During that execution, the CPU can process earlier results or prepare independent work. A PP pipeline additionally needs multiple stages; this chapter defaults to PP=1.'],
      ['本轮结果到达后，调度器提交已生成的 token、完成状态等变化。某条序列继续 Decode 的后续轮次要满足它自己的输入依赖，其他请求仍可继续排入批次。', 'When the results return, the scheduler commits generated tokens and completion state. A later decode step must satisfy that sequence’s input dependencies while other requests can still be scheduled.'],
    ],
    crosscut: [
      ['分别看请求的运行状态和 KV 的归属。调度器确认请求结束后，会解除这个请求对 KV 的占用。其他请求仍在用的前缀必须保留。符合缓存条件的 KV 还可能留给下一次命中，直到需要腾出空间。', 'Track request state and KV ownership separately. The scheduler drops this request’s hold on KV when the request finishes. Other requests can keep using their shared prefix. Reusable entries may stay cached until the allocator needs the space.'],
      ['压力来自“下一轮需要更多 KV 地址，现有空位不足”。先看能否驱逐闲置缓存；需要暂缓活动请求时，它将来恢复可能要补算已失去的 KV。单独看队列里还有几条请求，解释不了这笔额外工作。', 'Pressure arises when the next step requires more KV slots than are available. Look first for evictable idle entries. An active request that has to yield may later recompute missing KV. Queue length alone misses that extra work.'],
      ['等待中的请求通常只需停止后续准入。运行中的请求要等调度器处理取消，已经提交的 GPU 工作还受执行和回收边界约束。因此收到取消与资源能够复用之间可能有间隔。', 'A waiting request typically needs to be kept out of future batches. An active request needs scheduler cleanup, and submitted GPU work still has execution and reclamation boundaries. Capacity can become reusable after cancellation is received.'],
      ['图捕获提前记录可重放的 GPU 执行序列。新一批要落入可用的形状桶才具备重放条件，较小批次可以填充到桶大小。否则走支持的分段或 eager 路径，仍计算该批输入的结果。', 'Graph capture records a replayable GPU launch sequence. The next batch needs a compatible captured shape, sometimes by padding up to a bucket. Otherwise the supported segmented or eager path computes the batch output.'],
    ],
    disagg: [
      ['选点时路由器可以参考缓存线索，但线索可能在请求入队前失效。要确定能省哪段 Prefill，得由接收实例查自己的缓存并由调度器判断本轮能否准入。', 'Routing can use cache hints, which may be out of date by admission time. The receiving instance looks up its own cache to determine which prefill work is reusable, and its scheduler decides admission.'],
      ['沿 Prefill → Decode 的数据箭头看消费条件。接收方需要知道 KV 对应哪个请求和地址，并确认本轮使用的 KV 已可读取。预留地址、开始传输与完成传输是三个事件，前两者发生后仍可能等待。', 'On the Prefill → Decode data path, find what allows consumption. The receiver needs the request-to-address mapping and readable KV for the step. Reservation, transfer initiation, and completion are three events; the first two can still leave a wait.'],
      ['查到前缀后再问副本在哪。GPU 上的副本可直接用，其他层上的副本要等恢复或加载。这个等待由缓存层次产生，和跨 P/D 实例交接对应不同的箭头。', 'After a prefix lookup, ask where the copy is. A GPU copy can be used directly, while a lower-tier copy needs restore or load. This cache-tier dependency is a different arrow from a cross-instance P/D handoff.'],
      ['Prefill 较慢会让 Decode 等输入 KV，Decode 接收慢也会让已算好的 KV 等消费。扩大其中一池之前，先定位排队发生在选点、传输就绪还是执行准入。', 'A slow prefill pool can leave Decode waiting for KV. A slow Decode pool can leave completed KV waiting to be consumed. Before changing pool size, locate the wait at routing, transfer readiness, or admission.'],
    ],
    metrics: [
      ['从图里的起点走到停表标记。请求到达、GPU 产生 token、引擎前端读到首批输出和客户端看到文字，各是不同事件。', 'Follow the start and stop markers. Arrival, GPU token production, the frontend receiving the first batch, and visible client text are distinct events.'],
      ['即使 GPU 已经生成最后一个 token，请求仍可能在引擎前端处理、发送响应或等待客户端读取。拿两组 E2E 数据比较之前，先对齐请求起点和终态定义。', 'After the GPU generates the last token, the frontend still processes it, delivers a response, and the client reads it. Before comparing E2E results, align the start event and what counts as completion.'],
      ['例如首次等 12 ms，运行后又等 30 ms。首次排队仍是 12 ms。若另一份账本确实累计了这两段等待，才会得到 42 ms。看公式时要问的是“记录哪一次等待”和“什么时候上报”。', 'Suppose a request waits 12 ms initially and another 30 ms later. Initial queueing is still 12 ms. A cumulative ledger includes 42 ms only if it records both waits. Ask which wait is measured and when it is reported.'],
      ['先看图里的观测间隔和新增 token 数，再切换 SGLang 的合并数。每个新 token 分到的是同一个平均值，而 Histogram 的样本数随新增 token 数增长。', 'Look at the observed interval and new-token count, then change the SGLang batch size. Each new token receives the same average, while the histogram observation count grows with the token count.'],
    ],
  };

  function StepNote({ topic, step, en, depth }) {
    let math;
    if (topic === 'prefix' && step === 1) {
      math = row(variable('L', 'reuse'), operator('='), variable('B'), operator('⌊'),
        fraction(variable('L'), variable('B')), operator('⌋'), operator('='),
        number(4), operator('×'), operator('⌊'), fraction(number(10), number(4)), operator('⌋'),
        operator('='), number(8));
    } else if (topic === 'batch' && step === 1) {
      math = row(variable('N', 'step'), operator('='), variable('N', 'decode'), operator('+'),
        variable('N', 'prefill'), operator('='), number(2), operator('×'), number(1),
        operator('+'), number(4), operator('='), number(6));
    }
    const showAddress = topic === 'prefix' && step >= 2 && depth !== 'overview';
    const position = step === 2 ? 8 : 12;
    const physicalBlock = step === 2 ? 31 : 32;
    if (showAddress) {
      math = row(variable('j', 'KV'), operator('='), variable('b'), operator('×'), variable('B'),
        operator('+'), variable('o'), operator('='), number(physicalBlock), operator('×'),
        number(4), operator('+'), number(0), operator('='), number(physicalBlock * 4));
    }
    return h('div', { className: 'ia-stepNote' },
      h('p', null, notes[topic][step][en ? 1 : 0]),
      showAddress ? h('p', null, local(en,
        `右图从 0 编号。位置 ${position} 除以块大小 B=4，商 ${position / 4} 用于查 block table，查到物理块 b=${physicalBlock}。余数 o=0 是块内偏移。物理块起点加上偏移，得到 KV 槽位；图中的请求槽位 s 用来找 block table 的那一行。`,
        `The right diagram uses zero-based positions. Dividing position ${position} by block size B=4 gives quotient ${position / 4}, which indexes the block table and yields physical block b=${physicalBlock}. The remainder o=0 is the in-block offset. Add it to the block’s start to get the KV slot. The diagram’s request slot s instead selects a row of the block table.`)) : null,
      math ? h(Formula, null, math) : null);
  }

  function Completion({ en }) {
    return h('div', { className: 'ib-reading' },
      h('p', null, local(en, 'SGLang 在结束时按缓存路径结算已计算部分，再撤销请求对路径的保护。仍受其他请求保护的页不可回收。vLLM 释放请求的块引用，引用降到零的完整缓存块可以同时在空闲队列里并保留前缀 hash，直到被重分配或驱逐。', 'At completion, SGLang settles the computed prefix and removes the request’s protection of that path. Pages still protected by other requests cannot be reclaimed. vLLM releases the request’s block references. A complete cached block with zero references can be on the free queue and retain its prefix hash until reassigned or evicted.')),
      h('p', null, h('a', { href: '?step=1#prefix', 'data-engine-topic': 'prefix', 'data-engine-step': '1' },
        local(en, '回看第 03 节的共享前缀与页／块地址', 'Review shared prefixes and page/block addresses in topic 03'))));
  }

  return { StepNote, Completion, Formula, variable, operator, number, row, fraction };
})();
