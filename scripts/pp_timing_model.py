"""Illustrative dependency model pinned to SGLang 279339f113.

Service-time experiments keep request admission/batch selection fixed. They do
not simulate a dynamic scheduler, backend contention, or measured performance.
Adapted from the original published timing-model generator.
"""
from collections import defaultdict, deque
import json

from pp_source_baseline import SOURCE_COMMIT, SOURCE_SHORT


def build_model(config=None):
    config = config or {}
    N = config.get("horizon", 22)  # extend beyond the displayed window so delayed ring messages exist
    nodes = {}
    cpu_sequences = defaultdict(list)

    def key(r, n, name): return f'{r}:{n}:{name}'
    def current(n): return n - 2 if 3 <= n <= 7 else None
    def old(n): return current(n - 2)

    def add(k, deps, duration, **attrs):
        kind = attrs.get('kind')
        if kind == 'gpu':
            duration = config.get('gpu_value', duration)
            duration *= config.get('gpu_scale', 1) * config.get('stage_scale', [1,1,1])[attrs['r']]
            duration *= config.get('batch_scale', [1,1,1,1,1])[current(attrs['n'])-1]
        elif kind == 'cpu': duration *= config.get('cpu_scale', 1)
        elif kind == 'h2d': duration *= config.get('h2d_scale', 1)
        elif kind == 'kv': duration *= config.get('kv_scale', 1)
        elif k.endswith(('proxy_message', 'out_message')): duration *= config.get('link_scale', 1)
        assert k not in nodes, k
        nodes[k] = dict(id=k, deps=[d for d in deps if d], duration=duration, **attrs)
        return k

    # Per-stage forward service times; deliberately unbalanced to expose backpressure.
    compute = [[4.0,3.2,6.0,3.8,4.5], [4.8,6.6,3.8,6.0,3.6], [9.5,4.4,7.6,3.5,5.4]]
    loads = [[2.2,2.8], [8.2,4.2], [3.0,7.8]]

    for r in range(3):
        prev = None
        prev_gpu = None
        prev_load = None
        prev_kv = None
        for n in range(N):
            m, o = current(n), old(n)
            def cpu(name, label, phase, dur=.12, deps=(), owner='队列级', ref='pp:221'):
                nonlocal prev
                k = key(r,n,name)
                add(k, [prev,*deps], dur, r=r, n=n, kind='cpu', phase=phase,
                    label=label, owner=owner, ref=ref, prev=prev)
                cpu_sequences[(r,n)].append(k)
                prev=k
                return k
            cpu('recv_req','接收/处理请求','A',.20,
                [key(r-1,n,'req_message')] if r else [],
                'M1–M5 请求集合' if n==0 else '后续请求集合（例中为空）','scheduler:2064')
            if r<2 and n:
                cpu('req_work','回收上轮 request send work','A',.04,[key(r+1,n-1,'recv_req')])
            cpu('boot_poll','bootstrap good∩ / bad∪','A',.18,
                [key(r-1,n,'boot_message')] if r else [],
                '请求候选集合；并非固定 M(i+1)','pp:593')
            if r<2 and n:
                cpu('boot_work','回收上轮 bootstrap send work','A',.03,[key(r+1,n-1,'boot_poll')])
            cpu('term_poll','轮询 PD transfer 终态 / 求交集','A',.15,
                [key(r-1,n,'term_message')] if r else [],'inflight 请求集合','pp:635')
            if r<2 and n:
                cpu('term_work','回收上轮 terminal send work','A',.03,[key(r+1,n-1,'term_poll')])
            cpu('chunk','处理旧 chunk / 过滤旧 batch','B',.10,owner='本例无中间 chunk',ref='prefill:1210')
            if r<2 and n:
                cpu('l2_drain','回收上轮 L2 计数发送','B',.03,[key(r+1,n-1,'l2_counts')],ref='cache:321')
            # The pinned default is UnifiedRadixCache: ONE combined count sync,
            # followed by local write/load completion handling (not two syncs).
            cpu('l2_counts','L2 write/load ACK：合并计数→逐级传播','B',.30,
                [key(r-1,n,'l2_counts')] if r else [],'缓存 ACK 队列；write=0，load 按完成队列统计','cache:3073')
            cpu('ack_events','已公布 ACK 的本地 event 检查/等待','B',.08,
                owner='历史 L2 操作；后级可等待本地完成',ref='cache:3187')
            cpu('select','前缀匹配 / 预算预检查' if m else '空队列检查 → 返回 None','C',.36 if m else .10,
                owner=f'M{m}' if m else '无可计算 batch',ref='scheduler:3959')
            if m:
                cpu('init_load','init_load_back：分配/合并 host hit' if m in (2,4) else '无 host hit：跳过 init_load_back','C',.12,
                    owner=f'M{m}',ref='policy:1208')
                # Current admission commits only after successful host materialization.
                # Split the former .48 u selection placeholder into .36 + .12 u.
                cpu('commit_admission','提交准入 / 锁定前缀 / 加入 can_run_list','C',.12,
                    owner=f'M{m}',ref='policy:1347')
                load_submit=cpu('start_load','构造 batch → start_loading / consumer index','C',.12,
                    owner=f'M{m}' if m in (2,4) else f'M{m}：load 队列为空',ref='controller:941')
            if m in (2,4):
                # start_loading records its start_event on schedule_stream.
                # The preceding loop queued wait_event(launch_event) for proxy
                # (PP0/1) or output (PP2). H2D cannot bypass that GPU completion.
                schedule_gate=key(r,n-1,'gpu') if current(n-1) else None
                prev_load=add(key(r,n,'h2d'),[load_submit,prev_load,schedule_gate],loads[r][(m==4)],
                    r=r,n=n,kind='h2d',phase='B',label=f'M{m} L2→HBM',owner=f'M{m}',ref='controller:960')
            if m:
                cpu('prepare_extend','prepare_for_extend / 保存当前 batch','C',.13,owner=f'M{m}',ref='scheduler:4058')
            if m and r:
                cpu('recv_proxy','等待/接收上级 hidden states','D',.14,[key(r-1,n,'proxy_message')],f'M{m}','pp:858')
            if r<2 and current(n-1):
                cpu('proxy_work','等待上轮 proxy send work','D',.04,[key(r+1,n-1,'recv_proxy')],f'M{current(n-1)}','pp:272')
            if m:
                launch=cpu('launch','提交本级 forward；记录 launch_event','E',.24,owner=f'M{m}',ref='pp:1221')
                # Deliberately coarsen per-layer waits to ALL H2D completion.
                # This is a model-only dependency, not a source batch barrier.
                gpu=add(key(r,n,'gpu'),[launch,prev_gpu,*([prev_load] if m in (2,4) else [])],compute[r][m-1],
                    r=r,n=n,kind='gpu',phase='E',label=f'M{m}',owner=f'M{m}',ref='pp:1229',launch=launch,previous_gpu=prev_gpu)
                prev_gpu=gpu
            # D=0: output helper is after current forward launch.
            if r==2 and current(n-1):
                cpu('out_work','等待上轮 output send work','F',.03,[key(0,n+1,'recv_out')],f'M{current(n-1)}','pp:710')
            elif r<2 and old(n-2):
                cpu('out_work','等待上轮 relay output send work','F',.03,[key(r+1,n-2,'recv_out')],f'M{old(n-2)}','pp:710')
            sent = m if r==2 else old(n-1)
            if sent:
                send=cpu('send_out','PP2 发 output' if r==2 else '转发上轮收到的 output','F',.10,owner=f'M{sent}',ref='pp:983')
                add(key(r,n,'out_message'),[send,key(r,n,'gpu')] if r==2 else [send,key(r,n-1,'recv_out')],.22,
                    r=r,n=n,kind='message',phase='F',label=f'output M{sent}',owner=f'M{sent}')
            if o:
                outdep=key(2,n-2,'out_message') if r==0 else key(r-1,n+1,'out_message')
                recv=cpu('recv_out','等待/接收旧 batch output','F',.12,[outdep],f'M{o}','pp:1072')
                # CUDA send-first: PP2 queues wait_event(current q_event) on
                # schedule_stream before old-output recv. copy_stream then
                # waits for that stream, even though the copied batch is older.
                copy_gate=key(r,n,'gpu') if r==2 and m else None
                add(key(r,n,'copy'),[recv,copy_gate],.24,r=r,n=n,kind='copy',phase='H',label='D2H',owner=f'M{o}',ref='pp:1084')
            # Control consensus is independent from the output tensor ring.
            if r==2 or n>=3:
                cpu('send_bc','发送 bootstrap 共识回流','G',.10,owner='请求候选集合',ref='pp:660')
                cpu('send_rc','发送 release 共识回流','G',.10,owner='终态请求集合',ref='pp:683')
            if n>=2:
                bc=key(2,n-2,'send_bc') if r==0 else key(r-1,n+1,'send_bc')
                rc=key(2,n-2,'send_rc') if r==0 else key(r-1,n+1,'send_rc')
                cpu('recv_bc','接收 bootstrap 共识 → waiting_queue','G',.18,[bc],
                    'M1–M5 准入' if n==2 else '候选集合（本例无新准入）','pp:305')
            if r==2 or n>=3:
                peer=key(0,n+2,'recv_bc') if r==2 else key(r+1,n-1,'recv_bc')
                cpu('commit_bc','回收 bootstrap 共识发送','G',.025,[peer],ref='pp:311')
            if n>=2:
                cpu('recv_rc','接收 release 共识名单','G',.10,[rc],owner='已终态的历史请求集合',ref='pp:312')
            if r==2 or n>=3:
                peer=key(0,n+2,'recv_rc') if r==2 else key(r+1,n-1,'recv_rc')
                cpu('commit_rc','回收 release 共识发送','G',.025,[peer],ref='pp:314')
            if o:
                cpu('wait_copy','等待 d2h_event','H',.03,[key(r,n,'copy')],f'M{o}','pp:317')
                cpu('result','处理旧结果 / cache / inflight','H',.48+.08*o,owner=f'M{o}',ref='prefill:800')
                sendkv=cpu('send_kv','提交本级 KV 到 Decode','H',.22,owner=f'M{o}',ref='prefill:893')
                prev_kv=add(key(r,n,'kv'),[sendkv,prev_kv],[3.0,5.0,3.5,6.0,4.0][o-1]+r*.4,
                    r=r,n=n,kind='kv',phase='H',label=f'M{o} KV 在途',owner=f'M{o}',ref='prefill:1502')
            if n>=2:
                cpu('release','复查终态 / release_kv_cache / finish(SUCCESS)','H',.18,owner='候选终态请求；不等于当前旧 batch',ref='prefill:972')
            if r<2:
                for name,lab in [('req','请求'),('boot','bootstrap 候选'),('term','transfer 终态候选')]:
                    snd=cpu('tail_'+name,'转发本轮'+lab,'I',.10,ref='pp:326')
                    add(key(r,n,name+'_message'),[snd],.16,r=r,n=n,kind='message',phase='I',label=lab,owner='本轮队列快照')
                if m:
                    proxy=cpu('send_proxy','wait_event → 异步发送 hidden states','I',.10,owner=f'M{m}',ref='pp:336')
                    add(key(r,n,'proxy_message'),[proxy,key(r,n,'gpu')],.26,
                        r=r,n=n,kind='message',phase='I',label=f'M{m} 激活',owner=f'M{m}')
            cpu('end','保存 output / 共识状态，结束本轮','I',.04,ref='pp:346')

    # Solve the DAG; references outside the finite horizon remain unresolved.
    def solve():
        children=defaultdict(list);degree={}
        for k,v in nodes.items():
            v.pop('start',None);v.pop('end',None)
            degree[k]=len(v['deps'])
            for d in v['deps']: children[d].append(k)
        ready=deque(k for k,v in degree.items() if v==0)
        resolved=[]
        while ready:
            k=ready.popleft();v=nodes[k]
            v['start']=max((nodes[d]['end'] for d in v['deps']),default=0.0)
            v['end']=v['start']+v['duration'];resolved.append(k)
            for child in children[k]:
                degree[child]-=1
                if degree[child]==0: ready.append(child)
        return resolved

    # UnifiedRadix: PP0 publishes combined ready counts, then every rank waits for the
    # matching LOCAL completion events. Derive counts from synthetic finish times.
    ack_base={key(r,n,'ack_events'):nodes[key(r,n,'ack_events')]['deps'][:] for r in range(3) for n in range(N)}
    last_ack_rounds=None
    for attempt in range(12):
        resolved=solve();remaining=[4,6];ack_rounds={}
        for n in range(N):
            poll=nodes[key(0,n,'l2_counts')]
            ready_loads=[]
            while remaining and remaining[0]<n and 'start' in poll and nodes[key(0,remaining[0],'h2d')].get('end',float('inf'))<=poll['start']:
                ready_loads.append(remaining.pop(0))
            ack_rounds[n]=ready_loads
        if ack_rounds==last_ack_rounds:break
        last_ack_rounds=ack_rounds
        for r in range(3):
            for n,ready_loads in ack_rounds.items():
                k=key(r,n,'ack_events')
                nodes[k]['deps']=ack_base[k]+[key(r,j,'h2d') for j in ready_loads]
                nodes[k]['owner']='已公布 ACK：'+(', '.join(f'M{current(j)} 本地 H2D event' for j in ready_loads) if ready_loads else '本轮计数为 0')
    else:raise AssertionError('ACK count schedule did not stabilize')
    resolved=solve()

    if not all('end' in nodes[key(r,9,'end')] for r in range(3)):
        blocked=[(k,[d for d in v['deps'] if d not in nodes or 'end' not in nodes[d]]) for k,v in nodes.items() if v['n']<=10 and 'end' not in v]
        raise RuntimeError(json.dumps(blocked[:35],ensure_ascii=False,indent=2))

    # Resolve actual terminal/release memberships from the synthetic KV finish times.
    terminal={}; consensus={}; released=[set(),set(),set()]; release_rows=[]
    for n in range(N-3):
        for r in range(3):
            p=nodes[key(r,n,'term_poll')]
            if 'end' not in p:continue
            local={m for m in range(1,6) if 'end' in nodes.get(key(r,m+4,'kv'),{}) and nodes[key(r,m+4,'kv')]['end']<=p['start'] and m not in released[r]}
            terminal[(r,n)]=local if r==0 else local & terminal.get((r-1,n),set())
        for r in range(3):
            if n<2:continue
            # Last-stage snapshot n-2 -> PP0 iteration n -> next rank same n.
            targets=terminal.get((2,n-2),set())
            p=nodes[key(r,n,'release')]
            actual={m for m in targets if nodes[key(r,m+4,'kv')].get('end',float('inf'))<=p.get('start',-1)} - released[r]
            released[r]|=actual
            if actual:
                p['owner']='释放 '+', '.join(f'M{m}' for m in sorted(actual))
                release_rows.append(dict(r=r,n=n,batches=sorted(actual),time=p['start']))

    assert all(x==set(range(1,6)) for x in released),released
    last_release=max(x['n'] for x in release_rows)
    display_n=last_release+1
    loops=[];events=[]
    for r in range(3):
        for n in range(display_n):
            seq=[nodes[k] for k in cpu_sequences[(r,n)]]
            if not all('end' in e for e in seq):continue
            start=nodes[seq[0]['prev']]['end'] if seq[0]['prev'] else 0.0
            loops.append(dict(r=r,n=n,slot=n%3,start=start,end=seq[-1]['end'],current=current(n),old=old(n)))
            for e in seq:
                prev_end=nodes[e['prev']]['end'] if e['prev'] else 0.0
                if e['start']>prev_end+1e-7:
                    waited_for=[d for d in e['deps'] if d!=e['prev'] and nodes[d]['end']>prev_end+1e-7]
                    events.append(dict(id=e['id']+':wait',target=e['id'],r=r,n=n,kind='wait',phase=e['phase'],start=prev_end,end=e['start'],label='等：'+e['label'],owner=e['owner'],ref=e['ref'],deps=waited_for))
                events.append({k:e[k] for k in ['id','r','n','kind','phase','start','end','label','owner','ref']})
    for e in nodes.values():
        if e['kind'] in ('gpu','h2d','kv') and 'end' in e and e['n']<display_n:
            events.append({k:e[k] for k in ['id','r','n','kind','phase','start','end','label','owner','ref']})

    for r in range(3):
        prev_end=0.0
        for n in range(3,8):
            g=nodes[key(r,n,'gpu')]; launch=nodes[key(r,n,'launch')]
            a,b=prev_end,max(prev_end,launch['end'])
            if b>a+.001:
                waits=[e for e in events if e['r']==r and e['kind']=='wait' and e['start']<b and e['end']>a]
                causes=sorted(waits,key=lambda e:min(b,e['end'])-max(a,e['start']),reverse=True)
                cause=causes[0]['label'] if causes else 'CPU 调度 / 准备 / 提交'
                events.append(dict(id=g['id']+':idle',target=g['id'],r=r,n=n,kind='idle',phase='W',start=a,end=b,label='填充等待' if n==3 else '空泡：'+cause,owner='下一份 '+g['owner'],ref='pp:223'))
            if g['start']>b+.001:
                events.append(dict(id=g['id']+':layer-wait',target=g['id'],r=r,n=n,kind='idle',phase='W',start=b,end=g['start'],label='GPU 等全部 H2D（模型粗化）',owner=g['owner'],ref='controller:64'))
            prev_end=g['end']

    # CPU order, data dependencies and every batch's producer / consumer relations.
    for k in resolved:
        v=nodes[k]
        assert all(v['start']+1e-7>=nodes[d]['end'] for d in v['deps'])
    for r in range(3):
        assert sum(e['kind']=='gpu' and e['r']==r for e in events)==5
        for m in range(1,6):
            assert nodes[key(r,m+4,'send_kv')]['start']>=nodes[key(r,m+2,'gpu')]['end']
            if r:assert nodes[key(r,m+2,'gpu')]['start']>=nodes[key(r-1,m+2,'proxy_message')]['end']

    # Export all resolved predecessors, including asynchronous messages not drawn as
    # CPU work. Keep simulation scheduling edges separate from source data inputs.
    graph={}
    for k in resolved:
        e=nodes[k]
        g={f:e[f] for f in ['r','n','kind','phase','label','owner','deps']}
        g.update(start=round(e['start'],3),end=round(e['end'],3),ref=e.get('ref','pp:221'))
        if e.get('prev'):g['prev']=e['prev']
        extra=[]
        r,n=e['r'],e['n'];name=k.split(':')[-1]
        if 'ref' not in e:g['ref']={'out_message':'pp:983','proxy_message':'pp:340','req_message':'pp:327','boot_message':'pp:331','term_message':'pp:334','copy':'pp:1084'}.get(name,'pp:221')
        if name=='boot_poll':g['external']='本级 KV sender 的 bootstrap poll 状态（含级内 TP/CP 聚合）；Decode/握手细节未展开。轮询读取状态，不代表等待所有请求 ready。'
        if name in ('term_poll','release'):g['external']='本级 sender.poll() 的 Success/Failed 状态及 TP/CP 聚合；Decode 接收端和传输后端未展开。本图只画成功路径。'
        if name=='select':g['external']='waiting_queue、前缀匹配、PrefillBudget 与 _select_prefill_admission；预算/延迟准入先通过才准备回载。本图固定选批且假设容量充足，不模拟预算失败反馈。'
        if name=='commit_admission':g['external']='_commit_prefill_admission 在无需回载或回载准备成功之后锁定前缀、加入 can_run_list 并更新预算；随后构造 batch 才由 start_loading 提交实际 H2D。'
        if name=='start_load':g['external']='先由 ScheduleBatch.init_new 构造已准入请求的 batch，再由 ready_to_load_host_cache → start_loading 合并提交 H2D；load 队列为空时返回 -1，不发起复制。'
        if name=='l2_counts':g['external']='PP loop 在 process_prefill_chunk 后、get_new_batch_prefill 前显式调用 _process_hicache_events；空 batch loop 也进入缓存事件轮。'
        if name=='release':g['external']+=' 成功路径依次 release_kv_cache → tree_cache.finish(handle, SUCCESS) → sender.clear，并归还 metadata；当前默认缓存继承的 finish(SUCCESS) 不取消异步缓存工作，也不新增等待。'
        if name=='h2d':g['external']='start_loading 的 start_event 在 schedule_stream 上记录；H2D 流等待该事件，因此承接上一轮 launch_event 的顺序约束。'
        if name=='copy' and r==2 and current(n):g['external']='末级先在 schedule_stream 等当前 forward 的 q_event 再发送 output；copy_stream.wait_stream(schedule_stream) 将此约束传给旧结果 D2H。不是 CPU 在 wait_event 调用处同步等待。'
        if name=='gpu' and current(n) in (2,4):g['external']='模型把逐层 KV 就绪粗化成整份 H2D 完成后再开始前向；源码按层 wait_event，可边回载边计算，没有这个整批 barrier。'
        if name=='select' and current(n):extra.append((key(r,2,'recv_bc'),'本例请求已通过 bootstrap 准入'))
        if name=='launch' and r and current(n):extra.append((key(r,n,'recv_proxy'),'当前 batch 的上级激活已接收'))
        if name=='release' and n>=2:
            extra.append((key(r,n,'recv_rc'),'本轮 release 名单'))
            for row in release_rows:
                if row['r']==r and row['n']==n:
                    extra.extend((key(r,m+4,'kv'),f'M{m} 本地 KV 传输终态') for m in row['batches'])
        if name=='send_bc':extra.append((key(r,n,'boot_poll') if r==2 else key(r,n-1,'recv_bc'),'候选集合 / 上轮回流共识'))
        if name=='send_rc':extra.append((key(r,n,'term_poll') if r==2 else key(r,n-1,'recv_rc'),'终态集合 / 上轮 release 回流'))
        if name=='term_poll':
            extra.extend((key(r,m+4,'kv'),f'本例候选 M{m} 的本地终态') for m in sorted(terminal.get((r,n),set())))
        if extra:g['inputs']=[{'id':d,'note':note} for d,note in extra if d in nodes and 'end' in nodes[d]]
        assert all(nodes[d['id']]['end']<=e['start']+1e-7 for d in g.get('inputs',[])),k
        graph[k]=g

    data=dict(baseline=SOURCE_SHORT,sourceCommit=SOURCE_COMMIT,units='u（假设时间单位，非实测）',pp=3,depth=0,graph=graph,
              assumptions=['五个单请求 micro-batch；每轮至多选一份；完整 Prefill；成功路径',
                           '先展开三轮 bootstrap；UnifiedRadixCache + HiCache cache 模式；M2、M4 有 host hit；write_back 且无淘汰写回',
                           '服务时长为演示值；只模拟代码级依赖，省略真实 CUDA/NCCL 和链路争用',
                           '模型等待全部 H2D 后再画 GPU 前向；源码逐层等待，可与回载重叠，无整批 barrier'],
              loops=loops,events=events,releases=release_rows,
              end=max(l['end'] for l in loops),checks={'resolved_nodes':len(resolved),'batches':5,'forward_blocks':15,'released_all':True})
    for e in data['events']:
        e['start']=round(e['start'],3);e['end']=round(e['end'],3)
    for l in data['loops']:
        l['start']=round(l['start'],3);l['end']=round(l['end'],3)
    if N > 22:
        needed={k for k,v in graph.items() if v['n']<display_n}
        todo=list(needed)
        while todo:
            k=todo.pop()
            for d in graph[k]['deps']+[x['id'] for x in graph[k].get('inputs',[])]:
                if d not in needed: needed.add(d);todo.append(d)
        data['graph']={k:v for k,v in graph.items() if k in needed}
    return data
