/* Both projections share the same prefix, event intervals and player clock. */
(function () {
  'use strict';
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  function summary(plan, s) {
    if (s.hostReleased) return ['R1 已取消，L3 最终 ACK 已消费', '未采用本次读取结果；后台交还目标所有权后，L2 目标才归还。本例没有提交 H2D。'];
    if (s.releaseQueued) return ['L3 最终 ACK 已消费，目标尚待实际归还', '后台已交回所有权，本次未采用的 L2 目标在释放队列中。排队释放不等于已经归还到主机池。'];
    if (s.canceled) return ['R1 已取消，后台仍持有 L2 目标', 'L3 读取可能仍在进行，或正在交接最终结果。取消请求不能立即归还后台还在使用的目标。'];
    if (plan.scenario === 'miss' && s.storageChecked) return ['L3 未命中，本次不走回载', '不分配本次 L2 目标，也没有 H2D；后续按剩余输入重算。图中只标重算阶段，不补画其新 KV。'];
    if (plan.scenario === 'device-shortage' && !s.loadPreparing && plan.events.some(e => e.id === 'device-alloc' && s.done.includes(e.id)))
      return ['L1 空间不足，本次回载未接纳', '本次临时保护已解除，H2D 没有开始；L2 副本仍可保留。原设备前缀不变，R1 后续仍要通过准入。'];
    if (s.ack) return ['H2D ACK 已消费，传输保护已解除', '这只结束本次回载的传输引用。L2 副本仍在，R1 仍持有 L1 前缀；图中没有结束 R1，也不会把整段 KV 自动释放。'];
    if (s.ackPending) return ['三层复制已完成，H2D ACK 仍待 CPU 消费', '此时没有 KV 正在传输，脉冲已停止。Forward 可以继续；本次主机与设备路径保护仍保留到 loading_check。'];
    if (s.readyLayers.length) return ['已就绪层先读，后续层可以继续回载', '可读层：' + s.readyLayers.join('、') + '；正在读取：' + (s.activeLayers.join('、') || '无') + '。读取只等本层复制 event，不等 CPU 消费整批 H2D ACK。'];
    if (s.deviceAllocated) return ['L1 地址已分配，内容尚未可读', '返回设备索引只是取得目标地址。复制正在推进时，本层读取仍须等待对应 event。'];
    if (s.loadPreparing) return ['准备 L1 空间，并先建立临时保护', '先保护主机源和已有设备路径，再尝试分配目标；分配失败时撤销本次保护，不伪造一次 H2D。'];
    if (s.hostValid) return ['L2 副本有效，GPU 仍不能直接使用', '已有主机副本还要经过 L1 分配、H2D 和逐层完成事件。L2 命中不等于 GPU 内容已经可读。'];
    if (s.storageReadDone) return ['KV 已读到 L2 目标，尚未被缓存侧采用', '本例等待 L3 读取交接 ACK。后台目标所有权尚未交回，不能提前复用，也尚未提交 L1 回载。'];
    if (s.ioOwns) return ['L3 命中后，正在读入 L2 目标', '本例选择整段读完再采用。写入进度不是可读承诺：后台仍持有目标，L2 缓存与 L1 均未获得本次有效前缀。'];
    if (s.storageHitKnown) return ['L3 命中只确认存在，还没有读 KV', '调度侧先取得 L2 目标空间，之后才发起 get。查询结果不能直接当作 Attention 可以读取的内容。'];
    return ['先查候选前缀在哪里', 'L3 查询尚未完成。完整静态箭头表示本例可能使用的数据路径；只有对应传输区间内才有持续脉冲。'];
  }
  function storageAckLabel(plan, s) {
    if (plan.scenario === 'host') return '本例无需';
    if (s.storageAck) return '已消费';
    if (!s.hostAllocated) return '未发起读取';
    return s.storageReadDone ? '读完，待交接' : '读入中，待产生';
  }
  function sourceKey(s) {
    return s.hostReleased || s.releaseQueued ? '_drain_storage_control_queues_impl' : s.canceled ? 'release_aborted_request' :
      s.ack || s.ackPending ? 'loading_check' : s.readyLayers.length ? 'wait_until' : s.deviceAllocated ? 'start_loading' :
      s.loadPreparing ? 'load_back' : s.hostValid ? 'init_load_back' : s.storageReadDone ? '_handle_prefetch_result' :
      s.ioOwns || s.storageHitKnown ? '_drain_storage_control_queues_impl' : 'prefetch_from_storage';
  }
  function gates(plan, s) {
    const host = plan.scenario === 'host';
    const rows = [
      ['L3 查询', host ? '本例不访问' : s.storageChecked ? s.storageHitKnown ? '已确认命中' : '未命中' : '尚未确认'],
      ['L2 采用', s.hostValid ? '已有有效副本' : s.hostReleased ? '未采用，目标已归还' : s.releaseQueued ? '未采用，等待实际归还' : s.storageReadDone ? '读完，尚未采用' : s.ioOwns ? '读入中，尚未采用' : '尚无本次有效副本'],
      ['L1 目标', s.deviceAllocated ? '已取得设备索引' : s.loadPreparing ? '正在尝试分配' : plan.scenario === 'device-shortage' && s.hostValid ? '本次分配未成功' : '尚未取得'],
      ['当前层可读', s.readyLayers.length + ' / 3 层'],
      ['H2D ACK', s.ack ? '已消费，解除传输保护' : s.ackPending ? '复制完，待 CPU 消费' : s.deviceAllocated ? '复制尚未全部完成' : '尚未提交 H2D']
    ];
    return '<dl class="hc-gates">' + rows.map(([k, v]) => '<div><dt>' + k + '</dt><dd>' + esc(v) + '</dd></div>').join('') + '</dl>';
  }
  function figure(c, detailed) {
    const { M, plan, t, duration, tx, rect, flow, tracks, palette: p } = c;
    const s = M.hicacheState(plan, t), e = id => plan.events.find(e => e.id === id);
    let out = tx(30, 28, 'R1 的同一段前缀 S：教学模型共 3 层；行号 0–2 是模型层，L1/L2/L3 是存储层', 'sl-strong');
    const tiers = [[36, 'L3 外部存储', 'purple'], [369, 'L2 主机 DRAM', 'blue'], [702, 'L1 GPU HBM', 'green']];
    const headers = [plan.scenario === 'host' ? '本例不访问 L3' : s.storageChecked ? s.storageHitKnown ? '查询已确认存在' : '查询未命中' : '查询尚未完成',
      s.hostReleased ? '本次目标已归还' : s.releaseQueued ? '目标在释放队列中' : s.hostValid ? '已采用有效副本' : s.storageReadDone ? '读完；待交接 ACK' : s.ioOwns ? '后台正在读入' : '尚无本次有效副本',
      '可读 ' + s.readyLayers.length + '/3；读完 ' + s.readLayers.length + '/3'];
    tiers.forEach(([x, title, color], idx) => {
      out += rect(x, 51, 260, 225, 'var(--soft)', 'stroke="var(--border)"') + tx(x + 14, 78, title, 'sl-strong') + tx(x + 14, 101, headers[idx], 'sl-small');
      for (let layer = 0; layer < 3; layer++) {
        const y = 117 + layer * 48, copy = e('copy-' + layer);
        const copying = copy && t >= copy.start && t < copy.end;
        const valid = idx === 0 ? !['miss', 'host'].includes(plan.scenario) : idx === 1 ? s.hostValid : s.readyLayers.includes(layer);
        const staged = idx === 1 && s.storageReadDone && !s.hostReleased;
        const partial = idx === 1 && s.ioOwns ? s.storageProgress : idx === 2 && copying ? M.progress(t, copy.start, copy.end) : 0;
        const reserved = idx === 1 ? s.hostAllocated && !s.hostReleased : idx === 2 && s.deviceAllocated;
        out += '<g data-hc-tier="' + idx + '" data-hc-layer="' + layer + '">' + rect(x + 16, y, 228, 36, valid || staged ? p.purple : 'var(--surface)',
          'class="hc-cell" opacity="' + (valid || staged ? '.25' : '1') + '" stroke="' + p[color] + '"' + (!valid && reserved ? ' stroke-dasharray="4 3"' : ''));
        if (partial > 0 && !staged) out += rect(x + 16, y, 228 * partial, 36, p.purple, 'opacity=".3"');
        const status = idx === 0 ? plan.scenario === 'host' ? '未访问' : plan.scenario === 'miss' ? '本例无副本' : 'S 的原始 KV' :
          idx === 1 ? s.hostReleased ? '已归还目标' : s.releaseQueued ? '待归还，不可采用' : s.hostValid ? 'S 的有效 KV' : staged ? '读完，尚未采用' : s.ioOwns ? '读入中，尚未采用' : reserved ? '已分配目标' : '尚无本次副本' :
            s.activeLayers.includes(layer) ? '可读，正在读取' : s.readLayers.includes(layer) ? '可读，本层已读完' : valid ? '可读，等待本层读取' : copying ? '复制中，尚不可读' : reserved ? '已分配，等待复制' : '尚无本次目标';
        out += tx(x + 24, y + 24, '层 ' + layer + ' · ' + status, 'sl-small');
        if (idx === 2 && s.activeLayers.includes(layer)) out += rect(x + 12, y - 4, 236, 44, 'none', 'stroke="var(--accent)" stroke-width="3"');
        out += '</g>';
      }
    });
    out += flow([[296, 187], [369, 187]], e('storage-read'), '前缀 S 的 KV：L3 → L2', 'purple') + tx(333, 173, '整段 KV', 'sl-small', 'text-anchor="middle"');
    for (let layer = 0; layer < 3; layer++) {
      const y = 135 + layer * 48;
      out += flow([[613, y], [718, y]], e('copy-' + layer), '前缀 S 层 ' + layer + ' KV：L2 → L1', 'blue') + tx(665, y - 10, '层 ' + layer + ' H2D', 'sl-small', 'text-anchor="middle"');
    }
    if (detailed) {
      const badges = [['L2 后台 IO 目标', s.ioOwns ? '仍在持有' : s.hostReleased ? '已归还目标' : s.releaseQueued ? '释放队列持有目标' : s.hostValid ? '交给缓存侧保留' : '尚未取得'],
        ['本次回载保护', s.loadPreparing ? '主机源与设备路径临时保护' : s.deviceTransferPinned ? '设备与主机仍受保护' : s.ack ? 'ACK 已解除本次保护' : '当前无本次保护'],
        ['R1 的设备前缀引用', s.requestPinned ? '继续持有 L1 前缀' : '本次尚未建立']];
      badges.forEach(([k, v], i) => { const x = 36 + 321 * i;
        out += rect(x, 304, 306, 70, 'var(--soft)', 'stroke="var(--border)"') + tx(x + 12, 330, k, 'sl-small') + tx(x + 12, 357, v, 'sl-small');
      });
    }
    const top = detailed ? 478 : 383;
    const ackY = detailed ? 405 : 312;
    out += tx(36, ackY, 'L3 读取交接 ACK：' + storageAckLabel(plan, s) +
      '；H2D 复制完成：' + (s.allLayersReady ? '是' : '否') + '；H2D ACK：' + (s.ack ? '已消费' : s.ackPending ? '待 CPU 消费' : s.deviceAllocated ? '尚不满足完成条件' : '尚未提交'), 'sl-small');
    out += tx(36, detailed ? 433 : 340, s.hostReleased ? 'R1 已取消；L3 最终 ACK 和释放队列处理后，目标已归还。' : s.releaseQueued ? 'R1 已取消；L3 最终 ACK 已消费，目标待实际归还。' : s.canceled ? 'R1 已取消；目标仍受 IO 所有权约束。' : s.requestPinned ?
      'R1 仍持有 L1 前缀；解除传输保护不等于释放请求数据。' : '地址、有效内容与可释放条件分别判断。', 'sl-small');
    out += tracks(plan.events, [['control', 'CPU 控制（含等待）', 'gold'], ['storage', 'L3 后台 IO', 'purple'],
      ['h2d', 'GPU H2D stream', 'blue'], ['forward', 'GPU Forward stream', 'green']], top, t, duration);
    out += tx(36, top + 200, '同一时钟；u 非实测。控制条含等待，不代表 CPU 一直忙；Forward 斜纹是等待本层 event。', 'sl-small');
    out += tx(36, top + 228, 'H2D 是同机主机到 GPU；L3 位置与协议未指定。本图只恢复前缀，不结束 R1 或自动驱逐副本。', 'sl-small');
    return { body: out, height: top + 252 };
  }
  window.ServingHiCacheView = { summary, gates, figure, storageAckLabel, sourceKey };
})();
