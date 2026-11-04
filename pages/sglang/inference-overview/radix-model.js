/* Teaching model for the ordinary SGLang RadixCache / paged KV path.
 * Fixed source: sgl-project/sglang 279339f113b79af84f27fd3ac92d0a13bd3f4cbd.
 * The example fixes page_size=4, one cache namespace, and one static MHA KV pool.
 * It is an index/lifecycle model, not a GPU or timing simulation. */
(function (root) {
  'use strict';

  const revision = '279339f113b79af84f27fd3ac92d0a13bd3f4cbd';
  const pageSize = 4;
  const totalPages = 4;
  const pageIds = Array.from({ length: totalPages }, (_, i) => i + 1);
  const sourceRoot = 'python/sglang/srt/';

  const sources = {
    pageAlign: [sourceRoot + 'mem_cache/radix_cache.py', 147, 'def page_aligned'],
    childKey: [sourceRoot + 'mem_cache/radix_cache.py', 222, 'def child_key'],
    matchAt: [sourceRoot + 'mem_cache/radix_cache.py', 182, 'def match_at'],
    treeNode: [sourceRoot + 'mem_cache/radix_cache.py', 256, 'class TreeNode'],
    match: [sourceRoot + 'mem_cache/radix_cache.py', 397, 'def match_prefix'],
    insert: [sourceRoot + 'mem_cache/radix_cache.py', 457, 'def insert'],
    unfinished: [sourceRoot + 'mem_cache/radix_cache.py', 561, 'def cache_unfinished_req'],
    finished: [sourceRoot + 'mem_cache/radix_cache.py', 479, 'def cache_finished_req'],
    evict: [sourceRoot + 'mem_cache/radix_cache.py', 638, 'def evict'],
    lock: [sourceRoot + 'mem_cache/radix_cache.py', 668, 'def inc_lock_ref'],
    unlock: [sourceRoot + 'mem_cache/radix_cache.py', 683, 'def dec_lock_ref'],
    split: [sourceRoot + 'mem_cache/radix_cache.py', 750, 'def _split_node'],
    insertHelper: [sourceRoot + 'mem_cache/radix_cache.py', 783, 'def _insert_helper'],
    deleteLeaf: [sourceRoot + 'mem_cache/radix_cache.py', 856, 'def _delete_leaf'],
    leaf: [sourceRoot + 'mem_cache/radix_cache.py', 866, 'def _update_leaf_status'],
    rows: [sourceRoot + 'mem_cache/memory_pool.py', 259, 'class ReqToTokenPool'],
    rowWrite: [sourceRoot + 'mem_cache/memory_pool.py', 292, 'def write'],
    rowAlloc: [sourceRoot + 'mem_cache/memory_pool.py', 298, 'def alloc'],
    pagedAlloc: [sourceRoot + 'mem_cache/allocator/paged.py', 183, 'def alloc_extend'],
    freeSegment: [sourceRoot + 'mem_cache/allocator/paged.py', 282, 'def free_segment'],
    allocation: [sourceRoot + 'mem_cache/allocation.py', 55, 'def write_cache_indices'],
    prepare: [sourceRoot + 'managers/schedule_batch.py', 2678, 'def prepare_for_extend'],
    matchCall: [sourceRoot + 'managers/schedule_batch.py', 1617, 'match_result = tree_cache.match_prefix'],
    attention: [sourceRoot + 'layers/radix_attention.py', 157, 'def forward']
  };

  const R1_TOKENS = 'ABCDEFGHIJKL'.split('');
  const R2_TOKENS = 'ABCDXYZW'.split('');

  const steps = [
    {
      index: 0,
      phase: '场景',
      short: '已有前缀',
      title: '先放好一段旧前缀',
      caption:
        '树里已有节点 N1：key 是 token 序列 ABCDEFGH，value 是对应的 KV 位置索引 [4,5,6,7,8,9,10,11]。两条请求尚未搜索。节点只保存索引，不存储 K/V 数值。',
      source: 'treeNode',
      focus: { nodes: ['n1'], pages: [1, 2] },
      note: '节点保存 key + value（槽号），不保存文本，也不复制 K/V。'
    },
    {
      index: 1,
      phase: '搜索',
      short: 'R1 命中 8',
      title: 'R1 搜索：8 个 token 完整命中 N1',
      caption:
        'R1 的输入是 ABCDEFGHIJKL。match_prefix 沿 root 找到 N1，比较后命中前 8 个 token，返回 N1.value。prefix_indices 长度为 8，IJKL 尚未命中。搜索本身不加锁；随后调度准入调用 inc_lock_ref，保护从 last_node 到 root 的路径。',
      source: 'match',
      focus: { nodes: ['n1'], pages: [1, 2] },
      match: { request: 'R1', hit: 8, total: 12 },
      note: '命中长度是 8；未命中的 IJKL 不在返回索引中。'
    },
    {
      index: 2,
      phase: '插入',
      short: 'R1 插入',
      title: 'R1 插入新后缀 IJKL',
      caption:
        '以本轮 K/V 计算已经完成为前提，R1 提供 token 与 KV 位置索引。insert 复用已有前缀，只为 IJKL 创建叶 N3，value 为 [12,13,14,15]。cache_unfinished_req 重新匹配后更新 prefix_indices 与 last_node，并将保护移到新的末端。',
      source: 'unfinished',
      focus: { nodes: ['n3'], pages: [3] },
      inserted: ['n3'],
      note: 'prefill 完成就可以插入缓存，不必等请求结束；这正是后续请求能立即命中的原因。'
    },
    {
      index: 3,
      phase: '搜索',
      short: 'R2 命中 4',
      title: 'R2 只命中 4 个：在 N1 的中间分裂',
      caption:
        'R2 的 ABCD 与 N1 的前 4 个 token 相同，后面的 X 与 E 不同。match_prefix 调用 _split_node：新建 N2 保存 ABCD 与 value 的前半段；原对象 N1 保留 EFGH 与后半段。只修改索引切片和父子关系。返回 R2 的匹配结果后，准入保护让 N2 的 lock_ref 从 1 变成 2。',
      source: 'split',
      focus: { nodes: ['n2', 'n1'], pages: [1, 2] },
      split: true,
      match: { request: 'R2', hit: 4, total: 8 },
      note: '匹配在节点内部结束时，搜索本身会改树结构；这不是 insert 才发生的动作。'
    },
    {
      index: 4,
      phase: '插入',
      short: 'R2 插入',
      title: 'R2 的 XYZW 成为另一条分支',
      caption:
        '以 XYZW 的计算已经完成为前提，R2 提供新后缀与对应索引 [16,17,18,19]。insert 在 N2 下创建第二个孩子 N4。重新匹配后，R2 的 prefix_indices 长度变成 8，last_node 变成 N4。两条请求共享 ABCD 这一个树节点。',
      source: 'unfinished',
      focus: { nodes: ['n4'], pages: [4] },
      inserted: ['n4'],
      note: '共同前缀是同一个索引节点，不因请求增加而复制。'
    },
    {
      index: 5,
      phase: '保护与移除',
      short: '无可驱逐叶',
      title: '两条请求都在解码：可驱逐叶是空集',
      caption:
        'evict 只从 evictable_leaves 中选择节点：必须没有孩子，并且 lock_ref=0。N3、N4 都被活跃请求锁住，N1、N2 又有孩子，所以集合为空，不能移除任何节点。',
      source: 'evict',
      focus: { nodes: ['n2', 'n1', 'n3', 'n4'], pages: [] },
      pressure: { requested: 8, reclaimed: 0 },
      note: '锁保护的是活跃请求仍会读取的路径，而不是整棵树。'
    },
    {
      index: 6,
      phase: '保护与移除',
      short: 'R1 结束',
      title: 'R1 结束：解锁路径，节点继续保留',
      caption:
        'R1 结束后，dec_lock_ref 沿 N3→N1→N2 减锁：N3 与 N1 变成 0，N2 从 2 变成 1，因为 R2 仍保护 ABCD。N3 成为未锁叶，进入可驱逐集合；节点仍留在树上，等待真正的驱逐。',
      source: 'finished',
      focus: { nodes: ['n3'], pages: [] },
      finished: 'R1',
      note: '请求结束解除它的路径保护，不代表树节点立即消失。'
    },
    {
      index: 7,
      phase: '保护与移除',
      short: '驱逐 IJKL',
      title: '驱逐 N3：父节点成为新的候选',
      caption:
        'evict 选中未锁叶 N3。删除它后，N1 不再有孩子，且 lock_ref=0，于是 N1 成为新的可驱逐叶。本图将同一个驱逐循环的相邻选择拆开，方便观察资格变化。',
      source: 'deleteLeaf',
      focus: { nodes: ['n3'], pages: [3] },
      evicted: ['n3'],
      note: '本图只展示树内驱逐；底层存储的释放机制在下一节展开。'
    },
    {
      index: 8,
      phase: '保护与移除',
      short: '驱逐 EFGH',
      title: '驱逐 N1：EFGH 走了，ABCD 仍被 R2 锁着',
      caption:
        'N1（EFGH）已经是未锁叶，可以被移除。它的父节点 N2 仍被 R2 保护，而且还有孩子 N4，所以 N2 不可驱逐。R2 只使用 ABCD，不会替 EFGH 加锁。',
      source: 'evict',
      focus: { nodes: ['n1'], pages: [2] },
      evicted: ['n1'],
      note: '祖先被锁住，不代表它的所有后代都受保护；反过来，后代释放后祖先才可能变成叶。'
    },
    {
      index: 9,
      phase: '保护与移除',
      short: 'R2 结束',
      title: 'R2 结束：最后一把锁释放',
      caption:
        'R2 结束后，dec_lock_ref 沿 N4→N2 减锁，两者都变成 0。N4 已经是叶，可以进入候选集合；N2 仍有孩子 N4，所以不能立即移除。',
      source: 'finished',
      focus: { nodes: ['n4', 'n2'], pages: [] },
      finished: 'R2',
      note: 'lock_ref=0 且无孩子，才是可驱逐候选。'
    },
    {
      index: 10,
      phase: '保护与移除',
      short: '驱逐 XYZW',
      title: '驱逐 N4：ABCD 第一次成为叶',
      caption:
        '删除 N4（XYZW）后，N2 的最后一个孩子消失。此时 N2 同时满足“无孩子”和“lock_ref=0”，成为下一次可选的驱逐对象。',
      source: 'evict',
      focus: { nodes: ['n4'], pages: [4] },
      evicted: ['n4'],
      note: '父节点成为叶是资格变化，还没有被移除。'
    },
    {
      index: 11,
      phase: '保护与移除',
      short: '驱逐 ABCD',
      title: '驱逐 N2：树中只剩 root',
      caption:
        '移除最后的 N2（ABCD）后，只剩固定根节点 root。这些前缀不再能从树里命中；未来有相同输入时，需要重新计算并建立索引。',
      source: 'evict',
      focus: { nodes: ['n2'], pages: [1] },
      evicted: ['n2'],
      note: '缓存命中不是永久属性：被逐出后，相同 token 也要重新计算。'
    }
  ];


  function makeNode(id, tokens, slots, parent, lock) {
    return {
      id,
      tokens: tokens.slice(),
      slots: slots.slice(),
      parent: parent || null,
      children: [],
      lock: lock || 0
    };
  }

  function initialState() {
    const root = makeNode('root', [], [], null, 1);
    const n1 = makeNode('n1', R1_TOKENS.slice(0, 8), [4, 5, 6, 7, 8, 9, 10, 11], 'root', 0);
    root.children = ['n1'];
    return {
      step: 0,
      nodes: { root, n1 },
      requests: {
        R1: request('R1', R1_TOKENS, 0),
        R2: request('R2', R2_TOKENS, 0)
      },
      meta: steps[0]
    };
  }

  function request(id, tokens, prefixLen) {
    return {
      id,
      tokens: tokens.slice(),
      status: 'queued',
      row: Array(tokens.length).fill(null),
      prefixLen,
      prefixIndices: [],
      lastNode: null,
      rowReleased: false
    };
  }

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }


  const transitions = {
    1(state) {
      const r1 = state.requests.R1;
      const n1 = state.nodes.n1;
      r1.status = 'prefill';
      r1.prefixLen = 8;
      r1.lastNode = 'n1';
      r1.prefixIndices = n1.slots.slice();
      n1.lock = 1;
    },
    2(state) {
      const r1 = state.requests.R1;
      const n1 = state.nodes.n1;
      const n3 = makeNode('n3', R1_TOKENS.slice(8), [12, 13, 14, 15], 'n1', 1);
      state.nodes.n3 = n3;
      n1.children.push('n3');
      r1.row = r1.tokens.map((token, i) => ({
        slot: i < 8 ? n1.slots[i] : n3.slots[i - 8],
        token,
        origin: i < 8 ? 'hit' : 'computed'
      }));
      r1.prefixLen = 12;
      r1.prefixIndices = r1.row.map(e => e.slot);
      r1.status = 'decoding';
      r1.lastNode = 'n3';
      n1.lock = 1;
    },
    3(state) {
      const r2 = state.requests.R2;
      const n1 = state.nodes.n1;
      const n2 = makeNode('n2', n1.tokens.slice(0, 4), n1.slots.slice(0, 4), 'root', n1.lock);
      n2.children = ['n1'];
      state.nodes.n2 = n2;
      n1.tokens = n1.tokens.slice(4);
      n1.slots = n1.slots.slice(4);
      n1.parent = 'n2';
      state.nodes.root.children = ['n2'];
      r2.status = 'prefill';
      r2.prefixLen = 4;
      r2.lastNode = 'n2';
      r2.prefixIndices = n2.slots.slice();
      n2.lock = 2;
    },
    4(state) {
      const r2 = state.requests.R2;
      const n2 = state.nodes.n2;
      const n4 = makeNode('n4', ['X', 'Y', 'Z', 'W'], [16, 17, 18, 19], 'n2', 1);
      state.nodes.n4 = n4;
      n2.children.push('n4');
      r2.row = r2.tokens.map((token, i) => ({
        slot: i < 4 ? n2.slots[i] : n4.slots[i - 4],
        token,
        origin: i < 4 ? 'hit' : 'computed'
      }));
      r2.prefixLen = 8;
      r2.prefixIndices = r2.row.map(e => e.slot);
      r2.status = 'decoding';
      r2.lastNode = 'n4';
      n2.lock = 2;
    },
    5() {
      // Capacity pressure: evict() is called, but no node is an unlocked leaf.
    },
    6(state) {
      const r1 = state.requests.R1;
      r1.status = 'finished';
      r1.row = Array(r1.tokens.length).fill(null);
      r1.rowReleased = true;
      r1.lastNode = null;
      state.nodes.n3.lock = 0;
      state.nodes.n1.lock = 0;
      state.nodes.n2.lock = 1;
    },
    7(state) {
      delete state.nodes.n3;
      state.nodes.n1.children = [];
    },
    8(state) {
      delete state.nodes.n1;
      state.nodes.n2.children = state.nodes.n2.children.filter((id) => id !== 'n1');
    },
    9(state) {
      const r2 = state.requests.R2;
      r2.status = 'finished';
      r2.row = Array(r2.tokens.length).fill(null);
      r2.rowReleased = true;
      r2.lastNode = null;
      state.nodes.n4.lock = 0;
      state.nodes.n2.lock = 0;
    },
    10(state) {
      delete state.nodes.n4;
      state.nodes.n2.children = [];
    },
    11(state) {
      delete state.nodes.n2;
      state.nodes.root.children = [];
    }
  };

  function buildStates() {
    const result = [];
    let current = initialState();
    result.push(current);
    for (let i = 1; i < steps.length; i += 1) {
      current = clone(current);
      transitions[i](current);
      current.step = i;
      current.meta = steps[i];
      result.push(current);
    }
    return result;
  }

  const states = buildStates();

  function cloneWithMeta(state, meta) {
    const copy = clone(state);
    copy.step = meta.index;
    copy.meta = meta;
    return copy;
  }

  function searchSplitState() {
    const state = clone(states[1]);
    const n1 = state.nodes.n1;
    const n2 = makeNode('n2', n1.tokens.slice(0, 4), n1.slots.slice(0, 4), 'root', n1.lock);
    n2.children = ['n1'];
    state.nodes.n2 = n2;
    n1.tokens = n1.tokens.slice(4);
    n1.slots = n1.slots.slice(4);
    n1.parent = 'n2';
    state.nodes.root.children = ['n2'];
    const r2 = state.requests.R2;
    r2.status = 'prefill';
    r2.prefixLen = 4;
    r2.lastNode = 'n2';
    r2.prefixIndices = n2.slots.slice();
    n2.lock = 2;
    return state;
  }

  function insertReadyState() {
    const state = clone(states[1]);
    const r1 = state.requests.R1;
    r1.row = r1.tokens.map((token, i) =>
      i < 8
        ? { slot: state.nodes.n1.slots[i], token, origin: 'hit' }
        : { slot: 12 + i - 8, token, origin: 'computed' }
    );
    // The row now maps all 12 positions, but the tree still owns only 8 tokens.
    r1.prefixLen = 8;
    r1.status = 'prefill';
    r1.lastNode = 'n1';
    return state;
  }

  const searchSteps = [
    {
      index: 0,
      phase: '搜索',
      short: '起点',
      title: '搜索前：树里已有 8 个 token',
      caption:
        'N1 的 key 是 ABCDEFGH，value 是 8 个 KV 位置索引。两条查询尚未进入。观察请求怎样路由到候选节点、比较 token，再取得匹配结果。',
      source: 'treeNode',
      focus: { nodes: ['n1'], pages: [1, 2] },
      note: '查询是 token 序列；value 是槽号，不是 K/V 数值。'
    },
    {
      index: 1,
      phase: '搜索',
      short: 'R1 命中 8',
      title: 'R1 的 12 个 token 命中前 8 个',
      caption:
        'R1 沿 root 找到 N1，命中前 8 个 token。match_prefix 返回位置索引并设置 last_node；它本身不加锁。调度准入才调用 inc_lock_ref 保护命中路径。',
      source: 'match',
      focus: { nodes: ['n1'], pages: [1, 2] },
      match: { request: 'R1', hit: 8, total: 12 },

      note: '完整命中一个 8-token 节点，不会因为查询更长而返回 12 个槽号。'
    },
    {
      index: 2,
      phase: '搜索',
      short: 'R2 分裂',
      title: 'R2 只命中 4 个：查询在节点内部结束',
      caption:
        'R2 在 ABCD 后与 N1 不再匹配，搜索本身触发 _split_node。新节点 N2 保存共同前缀，原 N1 保留 EFGH；value 仅切成对应的两段索引。准入后，共同前缀的 lock_ref 变为 2。',
      source: 'split',
      focus: { nodes: ['n2', 'n1'], pages: [1, 2] },
      split: true,
      match: { request: 'R2', hit: 4, total: 8 },

      note: '搜索本身会改树结构：只要匹配结束在节点中间，就需要分裂。'
    }
  ];
  const searchStates = [
    cloneWithMeta(states[0], searchSteps[0]),
    cloneWithMeta(states[1], searchSteps[1]),
    cloneWithMeta(searchSplitState(), searchSteps[2])
  ];

  const insertSteps = [
    {
      index: 0,
      phase: '插入',
      short: '起点',
      title: '插入前：请求命中一段旧前缀',
      caption:
        '树里已有 ABCDEFGH，R1 的输入为 ABCDEFGHIJKL。先找出可复用的前缀，再观察新后缀怎样成为树节点。',
      source: 'treeNode',
      focus: { nodes: ['n1'], pages: [1, 2] },
      note: '插入的是索引，不是把 K/V 从池里搬到树节点。'
    },
    {
      index: 1,
      phase: '插入',
      short: '命中 8',
      title: '命中后，本轮还剩 4 个位置',
      caption:
        'match_prefix 返回前 8 个 token 的 KV 位置索引。剩余 IJKL 尚未入树；后续插入以其计算已经完成、对应索引已经可用为前提。',
      source: 'match',
      focus: { nodes: ['n1'], pages: [1, 2] },
      match: { request: 'R1', hit: 8, total: 12 },

      note: 'match_prefix 返回索引；准入保护是后续独立动作。'
    },
    {
      index: 2,
      phase: '插入',
      short: '准备参数',
      title: '准备 insert 的 key 与 value',
      caption:
        '本轮计算已完成：R1 向插入接口提供完整 token key 与对应的 value 索引。这里仅展示 insert 的参数准备，树中仍只有旧前缀；计算与内存管理不在本图展开。',
      source: 'insert',
      focus: { nodes: ['n1'], pages: [] },
      insertReady: true,
      note: '插入接口接收已有 KV 的索引；不计算 K/V。'
    },
    {
      index: 3,
      phase: '插入',
      short: '挂上 N3',
      title: '挂接 N3，更新匹配结果与保护路径',
      caption:
        'insert 复用已有前缀，将 IJKL 与索引 [12,13,14,15] 挂成叶 N3。随后重新匹配，更新请求的 prefix_indices 与 last_node，并把路径保护移到新的末端。',
      source: 'unfinished',
      focus: { nodes: ['n3'], pages: [3] },
      inserted: ['n3'],
      note: 'prefill 完成后就可以插入，后续请求不必等这条请求结束。'
    }
  ];
  for (const step of insertSteps) step.visibleRequests = ['R1'];

  const insertStates = [
    cloneWithMeta(states[0], insertSteps[0]),
    cloneWithMeta(states[1], insertSteps[1]),
    cloneWithMeta(insertReadyState(), insertSteps[2]),
    cloneWithMeta(states[2], insertSteps[3])
  ];

  const removeSteps = states.slice(5).map((state, i) => {
    const base = steps[i + 5];
    return Object.assign({}, base, {
      index: i,
      phase: '移除',

    });
  });
  removeSteps[0] = Object.assign({}, removeSteps[0], {
    short: '无可驱逐叶',
    title: '移除前：两条请求都锁着路径',
    caption:
        '两条请求都持有路径保护。N3、N4 被锁，N1、N2 有孩子，evictable_leaves 为空，暂时没有可移除节点。',
    note: '移除的前提不是“有缓存”，而是“有未锁的叶”。'
  });
  const removeStates = states.slice(5).map((state, i) => cloneWithMeta(state, removeSteps[i]));
  for (const state of removeStates) state.meta = removeSteps[state.step];

  const demos = [
    {
      id: 'journey',
      label: '完整链路',
      hint: '从前缀搜索、后缀插入到保护与逐叶驱逐。',
      states,
      steps,
      interval: 2600
    },
    {
      id: 'search',
      label: '搜索',
      hint: '单独观察 match_prefix：完整命中 8 个，以及命中 4 个时搜索本身触发节点分裂。',
      states: searchStates,
      steps: searchSteps,
      interval: 2200
    },
    {
      id: 'insert',
      label: '插入',
      hint: '观察 insert 如何复用已有前缀、挂接新叶并更新路径保护。',
      states: insertStates,
      steps: insertSteps,
      interval: 2200
    },
    {
      id: 'remove',
      label: '移除',
      hint: '观察路径解锁、候选叶选择，以及删除子节点后父节点的资格变化。',
      states: removeStates,
      steps: removeSteps,
      interval: 2200
    }
  ];


  function pageOf(slot) {
    return Math.floor(slot / pageSize);
  }

  function slotOf(page, offset) {
    return page * pageSize + offset;
  }

  function derive(raw) {
    const nodes = Object.values(raw.nodes);
    const nodeBySlot = new Map();
    for (const node of nodes) {
      if (node.id === 'root') continue;
      node.slots.forEach((slot, index) => {
        nodeBySlot.set(slot, { node, index });
      });
    }
    const reqBySlot = new Map();
    for (const req of Object.values(raw.requests)) {
      if (req.rowReleased) continue;
      req.row.forEach((entry, index) => {
        if (entry) reqBySlot.set(entry.slot, { req, index });
      });
    }
    const pages = pageIds.map((id) => {
      const cells = Array.from({ length: pageSize }, (_, offset) => {
        const slot = slotOf(id, offset);
        const treeRef = nodeBySlot.get(slot);
        const reqRef = reqBySlot.get(slot);
        return {
          slot,
          reserved: !treeRef && !!(reqRef && reqRef.req.row[reqRef.index].reserved),
          token: treeRef ? treeRef.node.tokens[treeRef.index] : reqRef ? reqRef.req.tokens[reqRef.index] : null,
          node: treeRef ? treeRef.node.id : null,
          req: reqRef ? reqRef.req.id : null
        };
      });
      const nodeIds = [...new Set(cells.map((cell) => cell.node).filter(Boolean))];
      const reqIds = [...new Set(cells.map((cell) => cell.req).filter(Boolean))];
      let kind = 'free';
      let owner = null;
      if (nodeIds.length === 1 && cells.every((cell) => cell.node === nodeIds[0])) {
        kind = 'node';
        owner = nodeIds[0];
      } else if (reqIds.length === 1 && cells.every((cell) => cell.req)) {
        kind = 'private';
        owner = reqIds[0];
      } else if (cells.some((cell) => cell.node || cell.req)) {
        kind = 'mixed';
        owner = nodeIds[0] || reqIds[0] || null;
      }
      return {
        id,
        cells,
        kind,
        owner,
        tokens: cells.map((cell) => cell.token),
        refs: [...new Set([...nodeIds, ...reqIds])]
      };
    });
    const treeNodes = nodes.filter((node) => node.id !== 'root');
    const evictable = treeNodes
      .filter((node) => node.lock === 0 && node.children.length === 0)
      .map((node) => node.id);
    const usedPages = pages.filter((page) => page.kind !== 'free').length;
    const treeSlots = treeNodes.reduce((sum, node) => sum + node.tokens.length, 0);
    const lockedSlots = treeNodes
      .filter((node) => node.lock > 0)
      .reduce((sum, node) => sum + node.tokens.length, 0);
    return {
      step: raw.step,
      meta: raw.meta,
      nodes: raw.nodes,
      requests: raw.requests,
      pages,
      evictable,
      usedPages,
      freePages: totalPages - usedPages,
      treeSlots,
      lockedSlots,
      requestRows: Object.values(raw.requests).filter((req) => !req.rowReleased && req.row.some(Boolean)).length
    };
  }

  function sourceUrl(key) {
    const entry = sources[key];
    if (!entry) throw new Error('Unknown source ' + key);
    return 'https://github.com/sgl-project/sglang/blob/' + revision + '/' + entry[0] + '#L' + entry[1];
  }

  // One pure time domain for semantic events, packets and node geometry.
  // Durations are explanation time. No engine or GPU timing is inferred.
  sources.admit = [sourceRoot + 'managers/schedule_policy.py', 907, 'def _req_inc_lock_ref'];
  function timeline(mode = 'journey') {
    const demo = demos.find(d => d.id === mode) || demos[0];
    const phases = []; let cursor = 0;
    demo.steps.forEach((meta, index) => {
      const before = clone(demo.states[Math.max(0,index-1)]), after = clone(demo.states[index]);
      const operation = index === 0 ? (mode === 'remove' ? 'pressure' : 'intro') : meta.match ? 'search' : meta.finished ? 'finish' : meta.evicted ? 'evict' : meta.insertReady ? 'write' : meta.inserted ? (mode === 'insert' ? 'append' : 'insert') : 'pressure';
      const p = {index,meta,operation,before,after,start:cursor,events:[],flows:[],duration:0};
      let state = clone(before);
      const event = (at,label,source,change) => {if(change)change(state);p.events.push({at,label,source,state:clone(state)});};
      const flow = (label,from,to,start,end,via=[]) => p.flows.push({label,from,to,start,end,via});
      const reqSel = id => `[data-request="${id}"]`, nodeSel = id => `[data-node="${id}"]`;
      event(0,operation==='intro'?'观察已有 token 前缀与索引':operation==='pressure'?'检查叶子与保护状态':'开始本阶段',meta.source);
      if(operation === 'search') {
        const id=meta.match.request, req=after.requests[id], last=req.lastNode, splits=!!meta.split;
        event(.1,`${id} 提交查询，命中结果尚未返回`,'match');
        flow(id+' 查询',reqSel(id),nodeSel('n1'),.1,1.05,[nodeSel('root')]);
        event(1.05,'精确比较 token，确定可复用的前缀','matchAt');
        if(splits) event(1.75,'按匹配边界拆分 key 与 value','split',s=>{
          const inherited=s.nodes.n1.lock;
          s.nodes=clone(after.nodes);s.nodes.n2.lock=inherited;
        });
        flow(meta.match.hit+' 个槽号',nodeSel(last),reqSel(id),splits?2.1:1.75,2.85);
        event(2.85,'返回 prefix_indices 与 last_node','match',s=>{
          s.requests[id].prefixLen=req.prefixLen;s.requests[id].prefixIndices=clone(req.prefixIndices);s.requests[id].lastNode=last;
        });
        flow('准入保护 +1',reqSel(id),nodeSel(last),2.95,3.55);
        event(3.55,'准入后沿命中路径增加 lock_ref','admit',s=>Object.assign(s,clone(after)));
        p.duration=4.1;
      } else if(['write','insert','append'].includes(operation)) {
        const id=mode==='insert'?'R1':meta.index===2?'R1':'R2';
        const req=after.requests[id],newId=id==='R1'?'n3':'n4',parent=id==='R1'?'n1':'n2';
        event(.1,'已有新后缀的 KV 索引，准备 key 与 value','insert',s=>{
          s.requests[id].row=clone(req.row);
        });
        if(operation==='write') {
          event(.7,'insert 参数就绪，树结构尚未改变','insert',s=>Object.assign(s,clone(after)));
          p.duration=1.2;
        } else {
          flow('key + value',reqSel(id),nodeSel(parent),.25,1.1);
          event(1.1,'复用已有前缀，只为新后缀挂接叶节点','insertHelper',s=>{
            const locks=Object.fromEntries(Object.entries(s.nodes).map(([k,n])=>[k,n.lock]));
            s.nodes=clone(after.nodes);
            for(const[k,n]of Object.entries(s.nodes))n.lock=locks[k]??0;
          });
          flow('重新匹配的索引',nodeSel(newId),reqSel(id),1.4,2.1);
          event(2.1,'重新匹配，更新 prefix_indices','unfinished',s=>{
            s.requests[id].prefixLen=req.prefixLen;s.requests[id].prefixIndices=clone(req.prefixIndices);
          });
          flow('保护移到新末端',nodeSel(parent),nodeSel(newId),2.25,2.85);
          event(2.85,'last_node 更新；保护整条命中路径','unfinished',s=>Object.assign(s,clone(after)));
          p.duration=3.4;
        }
      } else if(operation==='finish') {
        const id=meta.finished,last=before.requests[id].lastNode;
        event(.1,'结束请求：先处理缓存与释放路径保护','finished');
        flow('路径锁 −1',reqSel(id),nodeSel(last),.1,.9);
        event(.9,'沿路径减锁，未锁叶成为驱逐候选','unlock',s=>{for(const[k,n]of Object.entries(s.nodes))n.lock=after.nodes[k].lock;});
        event(1.6,'请求结束；缓存索引继续留在树上','finished',s=>Object.assign(s,clone(after)));
        p.duration=2.3;
      } else if(operation==='evict') {
        const id=meta.evicted[0],parent=before.nodes[id].parent;
        event(.1,'从候选集合选中未锁叶','evict');
        flow('选中 '+id.toUpperCase(),'#radix-evictable',nodeSel(id),.15,1.1);
        event(1.1,'断开叶节点，重新检查父节点资格','deleteLeaf',s=>Object.assign(s,clone(after)));
        if(parent!=='root')flow('检查父节点',nodeSel(parent),'#radix-evictable',1.2,1.7);
        p.duration=1.9;
      } else {p.duration=operation==='intro'?1.2:1.8;event(.65,operation==='intro'?'树节点保存 token key 与位置索引 value':'没有未锁叶，本次不移除节点',meta.source,s=>Object.assign(s,clone(after)));}
      p.end=p.start+p.duration;cursor=p.end;phases.push(p);
    });
    return {mode:demo.id,phases,duration:cursor};
  }
  function atTime(t,time) {
    const now=Math.max(0,Math.min(t.duration,Number(time)||0)),p=t.phases.find(p=>now<=p.end+1e-8)||t.phases.at(-1),local=now-p.start;
    let e=p.events[0],prior=p.before;
    for(const next of p.events){if(next.at>local+1e-8)break;prior=e.state;e=next;}
    const raw=clone(e.state);raw.meta=clone(p.meta);raw.meta.source=e.source;
    if(p.operation==='evict'&&local<1.1)delete raw.meta.evicted;
    if(p.operation==='finish'&&local<1.6)delete raw.meta.finished;
    if(raw.meta.match && raw.requests[raw.meta.match.request].prefixLen!==raw.meta.match.hit)delete raw.meta.match;
    const changed=JSON.stringify(prior.nodes)!==JSON.stringify(raw.nodes);
    return {time:now,phase:p,local,event:e,raw,label:e.label,prior:changed?clone(prior):raw,morph:changed?Math.max(0,Math.min(1,(local-e.at)/.3)):1,ended:now>=t.duration,flows:p.flows.filter(f=>local>=f.start&&local<f.end).map(f=>({...f,u:(local-f.start)/(f.end-f.start)}))};

  }

  const api = {
    timeline,
    atTime,
    revision,
    pageSize,
    totalPages,
    pageIds,
    steps,
    states,
    demos,
    sources,
    derive,
    initialState,
    pageOf,
    slotOf,
    sourceUrl
  };

  if (typeof module === 'object' && module.exports) module.exports = api;
  root.RadixLifecycleModel = api;
})(typeof globalThis === 'object' ? globalThis : this);
