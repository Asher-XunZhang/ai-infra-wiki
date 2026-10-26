#!/usr/bin/env python3
"""Render the foundational teaching chapter from its reviewable content catalog."""
import argparse
import json
from html import escape
from pathlib import Path
from build_learning_navigation import ROOT, PAGES, FOUNDATIONS, FOUNDATION_BY_SLUG, sidebar

EXTENSIONS = json.loads((PAGES / 'foundations/extensions.json').read_text(encoding='utf-8'))
CONCEPTS = json.loads((PAGES / 'foundations/concepts.json').read_text(encoding='utf-8'))


def concept_table(table):
    head = ''.join(f'<th scope="col">{escape(x)}</th>' for x in table['headers'])
    rows = ''.join('<tr><th scope="row">' + escape(row[0]) + '</th>' + ''.join(f'<td>{escape(x)}</td>' for x in row[1:]) + '</tr>' for row in table['rows'])
    return f'<table class="fd-table concept-table"><caption>{escape(table["caption"])}</caption><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'


def concept_guide(c):
    """Explain the design problem independently of the animation's toy state."""
    topics = CONCEPTS[c['slug']]['topics']
    def paragraphs(items):
        return ''.join(f'<p>{escape(p)}</p>' for p in items)
    nav = '<nav class="concept-nav" aria-label="按概念问题阅读">' + ''.join(
        f'<a href="#why-{t["id"]}">{escape(t["title"])}</a>' for t in topics) + '</nav>'
    body = []
    for n, t in enumerate(topics, 1):
        content = paragraphs(t['paragraphs'])
        if table := t.get('table'):
            content += concept_table(table)
        if ex := t.get('example'):
            steps = ''.join(f'<li>{escape(s)}</li>' for s in ex['steps'])
            example_table = concept_table(ex['table']) if ex.get('table') else ''
            content += f'<aside class="concept-example" aria-labelledby="example-{t["id"]}"><h4 id="example-{t["id"]}">{escape(ex["title"])}</h4>{example_table}<ol>{steps}</ol><p class="concept-note">{escape(ex["note"])}</p></aside>'
        if detail := t.get('detail'):
            content += f'<details class="concept-background"><summary>{escape(detail["title"])}</summary>{paragraphs(detail["paragraphs"])}</details>'
        if t.get('links'):
            content += '<ul class="concept-links">' + ''.join(f'<li><a href="{escape(url, quote=True)}">{escape(label)} →</a></li>' for label, url in t['links']) + '</ul>'
        if t.get('refs'):
            content += '<p class="concept-refs">依据：' + ' · '.join(f'<a href="{escape(url, quote=True)}">{escape(label)}</a>' for label, url in t['refs']) + '</p>'
        body.append(f'<section id="why-{t["id"]}" class="concept-topic"><div class="concept-heading"><span aria-hidden="true">{n:02}</span><h3>{escape(t["title"])}</h3></div>{content}<a class="concept-back" href="#understand">回到问题导航 ↑</a></section>')
    check = CONCEPTS[c['slug']]['check']
    challenge = f'<aside class="concept-check"><h3>试着解释</h3><p>{escape(check["question"])}</p><details><summary>展开解释</summary><p>{escape(check["answer"])}</p></details></aside>'
    return '<section id="understand" class="concept-guide"><h2>理解设计：为什么需要这些机制？</h2>' + nav + ''.join(body) + challenge + '</section>'



def reading_guide(c):
    groups = c['reading_groups']
    parallel = c['slug'] == 'parallel'
    labels = ['策略', '分什么', '交接什么'] if parallel else ['决策', '处理对象', '观察重点']
    overview_title = '先看分工对象' if parallel else '先分清三个调度问题'
    overview_lead = '名字相近，切分的对象和需要交换的数据不同。选择下方策略可阅读收益、代价和例子。' if parallel else '执行前怎样凑批、生成中何时换成员、每轮做多少工作，使用不同的观察视角。'
    assert sorted(i for g in groups for i in g['sections']) == list(range(len(c['sections'])))
    nav = '<nav class="parallel-reading-nav" aria-label="按学习问题阅读">' + ''.join(f'<a href="#read-{g["id"]}">{escape(g.get("label", g["id"].upper() if g["id"] != "chunking" else "与切块的关系"))}</a>' for g in groups) + '</nav>'
    rows = ''.join(f'<tr><th scope="row"><a href="#read-{g["id"]}">{escape(g.get("label", g["id"].upper()))}</a></th><td>{escape(g["split"])}</td><td>{escape(g["exchange"])}</td></tr>' for g in groups if g['split'])
    overview = f'<section class="parallel-overview" id="parallel-overview"><h2>{overview_title}</h2><p>{overview_lead}</p><table class="fd-table"><thead><tr><th>{labels[0]}</th><th>{labels[1]}</th><th>{labels[2]}</th></tr></thead><tbody>' + rows + '</tbody></table>' + nav + '</section>'
    body = []
    for g in groups:
        points = ''.join(f'<div><dt>{escape(k)}</dt><dd>{escape(v)}</dd></div>' for k, v in g['points'])
        detail = ''.join(f'<h4>{escape(c["sections"][i][0])}</h4><p>{escape(c["sections"][i][1])}</p>' for i in g['sections'])
        body.append(f'<section id="read-{g["id"]}" class="parallel-reading-section"><div class="parallel-reading-title"><h3>{escape(g["title"])}</h3><a href="#parallel-overview">回到速览 ↑</a></div><p class="parallel-reading-lead">{escape(g["lead"])}</p><dl class="parallel-comparison">{points}</dl><details><summary>展开机制细节与完整说明</summary>{detail}</details></section>')
    return '<section id="parallel-reading"><h2>机制与取舍</h2>' + overview + ''.join(body) + '</section>'


def parallel_adoption_summary():
    """No-JS summary of the same fixed inputs used by the interactive panels."""
    return """<section class="parallel-adoption-static" aria-labelledby="parallel-adoption-static-title">
<h2 id="parallel-adoption-static-title">TP / PP / DP / EP / SP / PCP / DCP 采用前后速览</h2>
<p>固定同一输入、模型和工作量；下表只列核心差异，下面的动态图把对象、资源和时间关系展开。乘积项、层数、行数、配对数、位置数与教学时隙都是计数，不是实测性能。</p>
<table class="fd-table"><caption>同一算例的七项对照</caption><thead><tr><th scope="col">机制</th><th scope="col">未采用</th><th scope="col">采用后</th><th scope="col">代价与边界</th></tr></thead><tbody>
<tr><th scope="row">TP</th><td>1 张 GPU；W 有 4 个权重格；8 个乘积项、4 次加法，得到 Y=XW。</td><td>2 张 GPU；每卡 2 个权重格、4 个乘积项。切输入维时每格只有 1 个乘积，每卡 0 次局部加法，合并再加 4 次；切输出维时每格 2 乘 1 加，每卡 2 次局部加法，合并只按列拼接。</td><td>两侧系统都仍是 8 乘、4 加；权重来自同一份 W；合并与通信是新增成本，不折算成实测加速。</td></tr>
<tr><th scope="row">PP</th><td>1 张 GPU 保存 4 层；两个 MB 共 8 段层计算，用 8 个教学时隙。</td><td>2 张 GPU，每卡保存 2 层；同样 8 段层计算，第 6 格完成；2 张卡合计 12 个 GPU·时隙，其中 4 格是填充/排空。</td><td>同一 MB 仍须逐层经过两个 stage；两侧横轴使用同一时隙刻度；时隙等长只表示依赖和重叠，不是实测耗时。</td></tr>
<tr><th scope="row">DP</th><td>1 份完整模型；1 条共享执行路径；两请求可同批或先后进入；每请求仍做 2 层计算。</td><td>2 份完整模型；2 条独立执行路径；每请求仍做 2 层计算，输出仍分别对应。</td><td>参数存储从 2 个层块变为 4 个；单请求计算不减少；吞吐是否增加取决于请求量、合批与调度。</td></tr>
<tr><th scope="row">EP</th><td>1 张 GPU 保存 4 个专家；4 个 token 各选 2 个专家，8 个专家分支都在本卡。均衡路由 t1 选中 E0/E2，输出 3。</td><td>2 张 GPU，每卡 2 个专家；总专家、top-2 与每个路由条件内的 t1 输出不变。均衡路由 t1 仍为 3；热点路由改选 E0/E1，t1 输出 2.5。均衡时两 rank 各 4 次激活；热点时 6 / 2。</td><td>跨卡分支要 dispatch 激活并 combine 返回；两种路由本例都合计 4 次跨卡派发/返回，但来源不同。参数 2/2 均分不保证计算均衡；MoE 的稀疏激活收益不是 EP 独有。</td></tr>
<tr><th scope="row">SP</th><td>同一 TP 组 2 张 GPU；逐 token 区域每卡复制 p1…p4 共 4 行。</td><td>TP+SP 仍是同一 2 张 GPU；每卡 2 行（p1/p2 与 p3/p4），经 AllGather 进入 TP 区域，再由 ReduceScatter 回到分片。</td><td>不新增卡；总通信取决于实现和原 TP collective，不能笼统写成新增两次；不改变 KV 总量或完整 Attention 上下文。</td></tr>
<tr><th scope="row">PCP</th><td>1 张 GPU 负责 Q1…Q8；Qi 允许 i 个配对，1+…+8 = 36 对。</td><td>2 张 GPU；连续切分 10 / 26，前后配对 18 / 18，总数仍 36；Query 行归属改变。</td><td>每行仍需因果范围内的历史 K/V；本例 AllGather 副本，不保证每卡 KV 驻留减半；配对数不是耗时或加速比。</td></tr>
<tr><th scope="row">DCP</th><td>1 张 GPU 读取 p1…p4；不等权分支 exp=[1,1,3,3]、Z=8，结果为 5；等权分支 exp=[1,1,1,1]、Z=4，结果为 4。</td><td>2 张 GPU 各读 2 个位置；不等权分支 Z=2/6、局部 O=2/6，按 Z 加权合并为 5；等权分支 Z=2/2、局部 O=2/6，合并为 4。每步 Decode 交换一轮 O/Z。</td><td>局部平均只在 Z 相等时碰巧等于加权结果；不等权时局部平均为 4，不是正确答案 5。Z=2/6 只属于不等权分支，不能与等权的 Z=2/2 混用；每步新增局部结果与统计量通信，4→2 是读取位置数。</td></tr>
</tbody></table>
<p class="parallel-adoption-static-note">PP 还保留单 MB 分支：单 MB 示例两侧各 4 段层计算、同在第 4 格结束，两 stage 合计 8 GPU·时隙，其中 4 格为空闲；1 MB 与 2 MB 工作量不同，不能当等量性能比较，但单 MB 自身仍可对照采用前后的依赖与空闲。DP 的单副本也可 batch，图中不把单实例画成只能串行。EP 的 4/4 与 6/2 是 rank 激活负载，不是参数量；热点改变路由与 t1 结果，但同一路由条件内采用前后输出相同。SP 不新增 GPU；经典 SP 减轻 TP 组内逐 token 激活副本，但不改变总 KV，也不把激活分片等同于 KV 分片。PCP 的单卡基线并入同一面板，连续/前后配对仍可选。DCP 的 5 与 4 分属不等权与等权分支，Z=2/6 与 Z=2/2 也按分支区分，不混用结果。</p>
</section>"""

def parallel_space_time_entry():
    """V41: same eight positions, independent owner-rank and round choices."""
    def head(n):
        selected = ' is-selected' if n == 1 else ''
        pressed = 'true' if n == 1 else 'false'
        return (f'<th scope="col" data-col="{n}" class="pst-head-cell{selected}">'
                f'<button type="button" class="pst-pick" data-pst-pick="{n-1}" aria-pressed="{pressed}" aria-label="选择 p{n}，查看 Q{n} 的所属卡与轮次">'
                f'<span class="pst-pos">p{n}</span><small>Q{n}</small></button></th>')
    def pairs(n):
        selected = ' is-selected' if n == 1 else ''
        return f'<td data-col="{n}" class="pst-cell{selected}"><span class="pst-pairs">{n}</span></td>'
    def owners(n):
        selected = ' is-selected' if n == 1 else ''
        owner = 0 if n in (1, 2, 7, 8) else 1
        return f'<td data-col="{n}" class="pst-cell{selected}"><span class="pst-owner" data-pst-owner-cell="{n-1}" data-owner="{owner}">GPU {owner}</span></td>'
    def rounds(n):
        selected = ' is-selected' if n == 1 else ''
        rnd = 1 if n <= 4 else 2
        return f'<td data-col="{n}" class="pst-cell{selected}"><span class="pst-round" data-pst-round-cell="{n-1}" data-round="{rnd}">第 {rnd} 轮</span></td>'
    def state(n):
        return f'<td data-col="{n}" class="pst-cell"><span class="pst-state" data-pst-state-cell="{n-1}" data-state="waiting">未进入</span></td>'
    def kv(n):
        rnd = 1 if n <= 4 else 2
        return (f'<div class="pst-kv-col" data-pst-kv-col="{n-1}">'
                f'<div class="pst-kv-pos"><strong>p{n}</strong><small data-pst-kv-round="{n-1}" data-round="{rnd}">第 {rnd} 轮</small></div>'
                f'<div class="pst-kv-cell" data-pst-kv="K" data-pst-pos="{n-1}" data-state="waiting"><span>K{n}</span><small>未写入</small></div>'
                f'<div class="pst-kv-cell" data-pst-kv="V" data-pst-pos="{n-1}" data-state="waiting"><span>V{n}</span><small>未写入</small></div>'
                f'</div>')
    heads = ''.join(head(n) for n in range(1, 9))
    pair_cells = ''.join(pairs(n) for n in range(1, 9))
    owner_cells = ''.join(owners(n) for n in range(1, 9))
    round_cells = ''.join(rounds(n) for n in range(1, 9))
    state_cells = ''.join(state(n) for n in range(1, 9))
    kv_axis = '<div class="pst-kv-axis" aria-hidden="true"><span></span><span>K</span><span>V</span></div>'
    kv_cols = ''.join(kv(n) for n in range(1, 9))
    return f'''<section class="parallel-space-time" id="parallel-space-time" data-parallel-space-time data-owner="zigzag" data-chunk="chunk4" data-ready="false" aria-labelledby="pst-title">
<header class="pst-head">
 <div class="pst-head-copy">
  <p class="pst-kicker">同一条 Prefill 请求 · 两条轴</p>
  <h2 id="pst-title">Query 归哪张卡，和它在第几轮进入前向，是两件事</h2>
  <p class="pst-lead">同一组 p1…p8、同一模型与因果遮罩。先选“按卡分工”，再选“每轮处理范围”；改一条轴，另一条保持不动。播放后逐位置看 Q 行读取 K/V、写入缓存，以及下一轮怎样读取上轮保留的历史。</p>
 </div>
 <p class="pst-total"><span>计划总量 · 因果配对</span><strong>36</strong><small>不随切法或轮次改变；完成数另计</small></p>
</header>
<div class="pst-controls">
 <div class="pst-control" role="group" aria-labelledby="pst-owner-label">
  <p class="pst-control-label" id="pst-owner-label">按卡分工 <span>空间 · 谁算这一行</span></p>
  <div class="pst-segments">
   <button type="button" class="pst-segment" data-pst-owner="single" aria-pressed="false"><strong>单卡</strong><small>对照基线</small></button>
   <button type="button" class="pst-segment" data-pst-owner="contiguous" aria-pressed="false"><strong>连续切分</strong><small>p1–p4 / p5–p8</small></button>
   <button type="button" class="pst-segment" data-pst-owner="zigzag" aria-pressed="true"><strong>前后配对</strong><small>p1,2,7,8 / p3–p6</small></button>
  </div>
 </div>
 <div class="pst-control" role="group" aria-labelledby="pst-chunk-label">
  <p class="pst-control-label" id="pst-chunk-label">每轮处理范围 <span>时间 · 这一行何时进入前向</span></p>
  <div class="pst-segments">
   <button type="button" class="pst-segment" data-pst-chunk="whole" aria-pressed="false"><strong>整段 1 轮</strong><small>p1–p8 同轮</small></button>
   <button type="button" class="pst-segment" data-pst-chunk="chunk4" aria-pressed="true"><strong>chunk 4</strong><small>每轮 4 位置 · 2 轮</small></button>
  </div>
 </div>
 <p class="pst-unit">单位：因果配对数量<br>不是 GPU 耗时或加速比</p>
</div>
<p class="pst-status" data-pst-status role="status" aria-live="polite">前后配对 + chunk 4：已完成 0 / 36 对；第 1 轮计划 3 / 7（10 对），第 2 轮计划 15 / 11（26 对），全程计划 18 / 18（36 对）。尚未进入轮次。</p>
<p class="pst-sr" data-pst-live aria-live="polite" aria-atomic="true"></p>
<div class="pst-player" data-pst-player>
 <div class="pst-transport" role="group" aria-label="过程播放控制">
  <button type="button" data-pst-action="play" aria-pressed="false">播放</button>
  <button type="button" data-pst-action="prev">回看</button>
  <button type="button" data-pst-action="next">单步</button>
  <button type="button" data-pst-action="replay">重播</button>
  <button type="button" data-pst-action="reset">重置</button>
  <span class="pst-step-label" data-pst-step-label>步骤 1 / 12 · 计划</span>
 </div>
 <p class="pst-phase" data-pst-phase>计划：p1…p8 尚未进入前向，K/V1–8 尚未写入。每个 Qi 只读 K/V1–i；按播放或单步开始第 1 轮。</p>
</div>
<div class="pst-board">
 <p class="pst-scroll-hint">窄屏可左右滑动，查看全部 8 个位置。</p>
 <div class="pst-table-wrap">
  <table class="pst-table">
   <caption class="pst-sr">同一组八个 Query 位置的所属 rank 与进入轮次；选择位置可查看因果范围。</caption>
   <thead><tr><th scope="col">位置</th>{heads}</tr></thead>
   <tbody>
    <tr><th scope="row">因果配对</th>{pair_cells}</tr>
    <tr><th scope="row">所属 rank</th>{owner_cells}</tr>
    <tr><th scope="row">进入轮次</th>{round_cells}</tr>
    <tr><th scope="row">本轮状态</th>{state_cells}</tr>
   </tbody>
  </table>
 </div>
 <div class="pst-kv-panel">
  <div class="pst-kv-head"><strong>K/V 缓存 · 逐位置</strong><span>K 和 V 都写入；青色 = 当前读取，斜线 = 未来遮罩，琥珀 = 上轮保留</span></div>
  <div class="pst-kv-scroll"><div class="pst-kv-grid">{kv_axis}{kv_cols}</div></div>
  <div class="pst-kv-legend"><span><i></i>已写入</span><span><i class="history"></i>上轮保留</span><span><i class="future"></i>未来遮罩</span><span><i class="read"></i>当前读取</span></div>
 </div>
 <div class="pst-history" data-pst-history>
  <div><strong>第 1 轮（计划）</strong> 完成后写入并保留 K/V1–4</div>
  <span class="pst-history-arrow" aria-hidden="true">→</span>
  <div><strong>第 2 轮（计划）</strong> 只新增 p5–p8；Q5 读 1–5，Q6 读 1–6，Q7 读 1–7，Q8 读 1–8</div>
  <p class="pst-history-note">不是“Q5 读 K/V1–8”：每个 Qi 只读 K/V1..i；K/V1–4 是上轮历史，K 和 V 都保留。</p>
 </div>
</div>
<div class="pst-lower">
 <aside class="pst-detail" aria-label="当前 Query 的位置属性" aria-live="polite">
  <div class="pst-detail-head">
   <div class="pst-detail-id"><small>当前 Query</small><strong data-pst-detail-q>Q1</strong></div>
   <div class="pst-detail-tags"><span class="pst-detail-state" data-pst-detail-state data-state="planned">计划</span><span class="pst-round" data-pst-detail-round data-round="1">第 1 轮</span><span class="pst-owner" data-pst-detail-owner data-owner="0">GPU 0</span></div>
  </div>
  <p class="pst-detail-pairs"><strong data-pst-detail-pairs>1</strong> 个因果配对 <span class="pst-detail-causal" data-pst-detail-causal>因果范围：K/V1</span></p>
  <p class="pst-detail-note" data-pst-detail-note>计划进入第 1 轮，由 GPU 0 计算；尚未进入前向，未计算，K/V1 未写入。</p>
 </aside>
 <div class="pst-ledger">
  <div class="pst-ledger-head"><strong>计划账本</strong><span>表内是计划总数</span></div>
  <div class="pst-ledger-scroll"><table>
   <caption class="pst-sr">各组合的计划因果配对数量；完成数在表下单独统计。</caption>
   <thead><tr><th scope="col">轮次</th><th scope="col">GPU 0</th><th scope="col">GPU 1</th><th scope="col">本轮</th></tr></thead>
   <tbody>
    <tr data-pst-round-row="1"><th scope="row"><span data-pst-round-name="1">第 1 轮</span><small data-pst-round-range="1">p1–p4</small></th><td data-pst-ledger="r1-0">3</td><td data-pst-ledger="r1-1">7</td><td data-pst-ledger="r1-total">10</td></tr>
    <tr data-pst-round-row="2"><th scope="row"><span data-pst-round-name="2">第 2 轮</span><small data-pst-round-range="2">p5–p8</small></th><td data-pst-ledger="r2-0">15</td><td data-pst-ledger="r2-1">11</td><td data-pst-ledger="r2-total">26</td></tr>
    <tr class="is-total"><th scope="row">全程</th><td data-pst-ledger="all-0">18</td><td data-pst-ledger="all-1">18</td><td data-pst-ledger="all-total">36</td></tr>
   </tbody>
  </table></div>
  <div class="pst-done">
   <div class="pst-done-line"><span>已完成</span><strong data-pst-done-total>0</strong><span>/ 36 对</span></div>
   <div class="pst-bar" role="progressbar" aria-label="已完成因果配对" aria-valuemin="0" aria-valuemax="36" aria-valuenow="0" data-pst-progress><i data-pst-ledger-fill></i></div>
   <p class="pst-done-breakdown" data-pst-done-breakdown>第 1 轮 0 / 0 对 · 第 2 轮 0 / 0 对</p>
  </div>
  <p class="pst-ledger-note">表内是计划；已完成只统计已经读完并写入 K/V 的 Q 行。第 2 轮只新增 p5–p8，仍读上轮 K/V1–4；每个 Qi 只读 1..i。</p>
 </div>
</div>
<p class="pst-note"><strong>总 36 不变：</strong>Q1…Q8 各自的因果配对是 1…8，合计 36。每个 Qi 只读 K/V1..i，i 之后保持未来遮罩；第 2 轮的 Q5 只读 1–5，不能理解成 Q5 读 K/V1–8。</p>
<p class="pst-note"><strong>历史保留的是 K 和 V：</strong>chunk 4 的第 1 轮写入并保留 K/V1–4；第 2 轮只新增 p5–p8，Q5..Q8 仍按 1..i 读取上轮 K/V1–4 与本轮新增。整段 1 轮没有跨轮历史，但每个 Qi 仍只读 1..i。</p>
<p class="pst-note"><strong>计数不是耗时：</strong>全程 18/18 是总配对均衡，不代表每轮均衡：前后配对 + chunk 4 的第 1 轮是 3/7，第 2 轮是 15/11。配对数是因果读取次数，不是 GPU 耗时、吞吐或加速比。</p>
<p class="pst-note"><strong>逐行演示边界：</strong>为看清因果范围，过程按 Q 行逐个展示；真实内核会在一次前向中批量处理一个 chunk，不表示每行单独启动一次 kernel。PCP 的 owner 是“谁计算该 Q 行”，K/V 读取按因果范围（本例 AllGather 副本），不表示每卡只驻留一半。</p>
<aside class="pst-dcp" aria-labelledby="pst-dcp-title">
 <div><p class="pst-kicker">Decode 阶段 · 另一个例子</p><h3 id="pst-dcp-title">DCP 不把一个新 Q 切成 Prefill chunk</h3><p>Decode 时，同一个新 Q 读取按上下文位置分片的历史 K/V；各片得到局部 O 与 Z，再按 Z 合并。它不在这张 Prefill 的轮次轴上，也不把新 Q 逐块 Prefill。</p></div>
 <a href="?mv=10#lesson-animation">看本课 DCP 的局部 O / Z 合并 →</a>
</aside>
<noscript><div class="pst-noscript"><strong>未启用 JavaScript：</strong>下面是六种组合的计划配对账本；过程不能推进，已完成数保持 0。
<table><caption>六种组合的因果配对账本</caption><thead><tr><th scope="col">按卡分工</th><th scope="col">整段 1 轮</th><th scope="col">chunk 4 两轮</th></tr></thead><tbody>
<tr><th scope="row">单卡</th><td>GPU 0：36；Total 36</td><td>第 1 轮 10；第 2 轮 26；Total 36</td></tr>
<tr><th scope="row">连续切分</th><td>10 / 26；Total 36</td><td>第 1 轮 10 / 0；第 2 轮 0 / 26；Total 36</td></tr>
<tr><th scope="row">前后配对</th><td>18 / 18；Total 36</td><td>第 1 轮 3 / 7；第 2 轮 15 / 11；Total 36</td></tr>
</tbody></table>
<p>第 2 轮只新增 p5–p8，仍读第 1 轮保留的 K/V1–4；每个 Qi 只读 K/V1..i。表内是计划数量，配对数量不是耗时。</p></div></noscript>
</section>'''

def parallel_goal_selector():
    """V33: static goal/cost entry. JS only filters these complete panels."""
    return """<section class="parallel-goals" id="parallel-goals" data-parallel-goals aria-labelledby="parallel-goals-title">
<header class="pg-head">
<p class="pg-kicker">先问哪里不够</p>
<h2 id="parallel-goals-title">三种“卡不够”，先选最像的那个</h2>
<p class="pg-lead">装不下权重、接不住请求、放不下长上下文，切的是不同对象。先选一个目标，只展开相关策略；这是阅读入口，不是性能排名，也不表示只能选一种。</p>
</header>
<div class="pg-goals" role="group" aria-label="选择当前最接近的容量问题">
<button type="button" class="pg-goal" id="pg-goal-fit" data-goal="fit" data-label="权重 / 模型层装不下" aria-pressed="false" aria-controls="parallel-goal-fit"><small>容量 · 参数</small><strong>权重 / 模型层装不下</strong><span>一张卡放不下整份权重，或单个算子、整段层数在单卡上执行不了。</span><span class="pg-tags">先看：TP 算子内切分 · PP 按层分段 · EP 只对 MoE 专家</span></button>
<button type="button" class="pg-goal" id="pg-goal-requests" data-goal="requests" data-label="模型装得下，但请求多" aria-pressed="false" aria-controls="parallel-goal-requests"><small>容量 · 并发</small><strong>模型装得下，但请求多</strong><span>单条请求能跑，独立请求在排队；希望同时服务更多条。</span><span class="pg-tags">先看：DP 完整副本分请求</span></button>
<button type="button" class="pg-goal" id="pg-goal-context" data-goal="context" data-label="Prefill 行多 / Decode 历史长" aria-pressed="false" aria-controls="parallel-goal-context"><small>长上下文</small><strong>Prefill 行多 / Decode 历史长</strong><span>权重装得下，但长输入的 Query 行，或 Decode 要读的历史 KV，成为压力。</span><span class="pg-tags">先看：PCP 分 Prefill 行 · DCP 分历史 KV</span></button>
</div>
<p class="pg-rule">TP / PP / EP / DP / SP / PCP / DCP 可以组合；这里只决定先读哪组说明，不把策略当成互斥选项，也不声明任意组合在某引擎可运行。策略链接只进入对应现有图，默认停在第 1 步；播放、暂停和切步骤仍由你控制。</p>
<div class="pg-aside">
<div class="pg-aside-copy"><small>激活布局 · 可与上述策略组合</small><strong>逐 token 激活占用</strong><span>长序列会放大逐 token 区域的激活；经典 SP 在 TP 组内按序列分片，减少这部分激活的重复保存，但不改变 KV 总量。</span></div>
<button type="button" id="pg-goal-activation" data-goal="activation" data-label="逐 token 激活占用" aria-pressed="false" aria-controls="parallel-goal-activation">看 SP 布局</button>
</div>
<p class="pg-status" data-goal-status role="status" aria-live="polite">先选一个目标，只展开相关策略与代价；未选择时不会自动播放动画。</p>
<noscript><p class="pg-noscript">未启用 JavaScript：动画不能播放，但下面仍按目标分组列出全部策略说明与算例；启用 JavaScript 后可切换目标、单步查看对应动画。</p></noscript>
<div class="pg-panels">
<section class="pg-panel" id="parallel-goal-fit" data-goal-panel="fit" aria-labelledby="parallel-goal-fit-title">
<div class="pg-panel-head"><h3 id="parallel-goal-fit-title">权重 / 模型层装不下</h3><p>先分清整份权重、层间流水和 MoE 专家。它们分担的对象不同，不能互相替代。</p></div>
<div class="pg-strategies">
<article class="pg-strategy"><div class="pg-strategy-head"><h4>张量并行</h4><span class="pg-abbr">TP</span></div><dl>
<div><dt>分担</dt><dd>单个算子的权重矩阵与计算；输入维切列、W 切行，或 W 切列后按输出列拼接。</dd></div>
<div><dt>不消失</dt><dd>总参数量与总乘加数；本例仍是 8 乘、4 加，权重来自同一份 W。</dd></div>
<div><dt>代价</dt><dd>局部结果要相加或拼接，跨卡通信；每卡只持有权重分片，激活或临时缓冲还可能复制。</dd></div>
<div><dt>边界</dt><dd>相邻算子的布局决定是否需要通信；TP 组可以作为更大配置的一部分。</dd></div>
</dl><div class="pg-links"><a href="?mv=0#lesson-animation">看输入维切分 ↗</a><a href="?mv=1#lesson-animation">看输出维切分 ↗</a></div></article>
<article class="pg-strategy"><div class="pg-strategy-head"><h4>流水线并行</h4><span class="pg-abbr">PP</span></div><dl>
<div><dt>分担</dt><dd>连续模型层；Stage 0 执行 L1–L2，Stage 1 执行 L3–L4。</dd></div>
<div><dt>不消失</dt><dd>单个 MB 仍要按 L1→L4 逐层经过两个 stage；两个 MB 示例合计 8 段层计算，单 MB 示例两侧各 4 段、同在第 4 格结束。</dd></div>
<div><dt>代价</dt><dd>激活交接、填充/排空气泡与 stage 间负载不均；两个 MB 示例两 stage 合计 12 GPU·时隙，其中 4 格是气泡；单 MB 示例两 stage 合计 8 GPU·时隙，其中 4 格为空闲。</dd></div>
<div><dt>边界</dt><dd>PP 是层间流水，不是 Prefill/Decode 分离；1 MB 与 2 MB 工作量不同，不能当等量性能比较；单 MB 自身仍可对照采用前后的依赖与空闲。</dd></div>
</dl><div class="pg-links"><a href="?mv=2#lesson-animation">看两个 MB ↗</a><a href="?mv=3#lesson-animation">看单 MB ↗</a></div></article>
<article class="pg-strategy"><div class="pg-strategy-head"><h4>专家并行</h4><span class="pg-abbr">EP</span></div><dl>
<div><dt>分担</dt><dd>MoE 层的专家参数与专家计算；token 按路由被派发到持有专家的 rank。</dd></div>
<div><dt>不消失</dt><dd>总专家数与 top-2 分支数；每个 token 仍只算被选中的专家，稀疏激活不是 EP 新增的。</dd></div>
<div><dt>代价</dt><dd>dispatch/combine 激活交换；专家参数均分不保证负载均衡（本例 4/4 与 6/2）。</dd></div>
<div><dt>边界</dt><dd>只适用于带 MoE 专家的模型；不切 dense 权重，也不能当成所有模型的省显存开关。</dd></div>
</dl><div class="pg-links"><a href="?mv=5#lesson-animation">看均衡路由 ↗</a><a href="?mv=6#lesson-animation">看热点路由 ↗</a></div></article>
</div>
<p class="pg-note"><strong>DP 不在这一组：</strong>它复制完整模型，不能解决一份模型装不下。<strong>SP 也不在这一组：</strong>它不减少权重，但可减轻 TP 组内逐 token 激活的重复保存。TP/PP/EP 可组合，但组合后的通信、气泡、负载与支持范围要按具体引擎和拓扑核对。</p>
</section>
<section class="pg-panel" id="parallel-goal-requests" data-goal-panel="requests" aria-labelledby="parallel-goal-requests-title">
<div class="pg-panel-head"><h3 id="parallel-goal-requests-title">模型装得下，但请求多</h3><p>先把目标定在并发容量，而不是单条请求延迟；DP 复制的是执行容量，不是把一条请求的矩阵拆小。</p></div>
<div class="pg-strategies">
<article class="pg-strategy"><div class="pg-strategy-head"><h4>数据并行</h4><span class="pg-abbr">DP</span></div><dl>
<div><dt>分担</dt><dd>不同请求；每条请求仍经过一个完整模型，图中 R1、R2 分别到两个副本。</dd></div>
<div><dt>不消失</dt><dd>单条请求的矩阵计算、层数与 KV 读写；每个副本都保留完整参数。</dd></div>
<div><dt>代价</dt><dd>本例每副本一张 GPU：1→2 个副本就是 1→2 张 GPU 和 2 份完整权重；另有每副本 KV/状态、请求路由与负载均衡。</dd></div>
<div><dt>边界</dt><dd>只有请求足够多、能并行或合批并被调度时，容量才可能转成吞吐；不保证单请求延迟下降。一个副本内部还可以组合 TP/PP，“副本”不等于“一张 GPU”。</dd></div>
</dl><div class="pg-links"><a href="?mv=4#lesson-animation">看完整副本 ↗</a></div></article>
</div>
<p class="pg-note"><strong>单副本也可以 batch：</strong>图中把两条请求分到两个副本，是为了看清复制成本，不是说一个副本只能串行。<strong>两个反例：</strong>一份模型本身装不下时，DP 不能解决；单条请求算得慢时，DP 也不会拆它的矩阵。</p>
</section>
<section class="pg-panel" id="parallel-goal-context" data-goal-panel="context" aria-labelledby="parallel-goal-context-title">
<div class="pg-panel-head"><h3 id="parallel-goal-context-title">Prefill 行多 / Decode 历史长</h3><p>长上下文不是一个阶段。Prefill 的压力在 Query 行与计算，Decode 的压力在每轮读取历史 KV；下面是两个独立教学例。</p></div>
<div class="pg-strategies">
<article class="pg-strategy"><div class="pg-strategy-head"><h4>Prefill 上下文并行</h4><span class="pg-abbr">PCP</span></div><dl>
<div><dt>分担</dt><dd>同一条长 prompt 的 Prefill Query/token 行；连续切分或前后配对改变行的归属。</dd></div>
<div><dt>不消失</dt><dd>每行仍需因果范围内的历史 K/V；总因果配对数仍是 36。</dd></div>
<div><dt>代价</dt><dd>远端 K/V 的交换或暂存；因果遮罩造成负载不均（连续 10/26，前后配对 18/18）。</dd></div>
<div><dt>边界</dt><dd>本例用 AllGather 副本，不保证每卡 KV 驻留减半；配对数不是耗时或加速比。</dd></div>
</dl><div class="pg-links"><a href="?mv=8#lesson-animation">看连续分片 ↗</a><a href="?mv=9#lesson-animation">看前后配对 ↗</a></div></article>
<article class="pg-strategy"><div class="pg-strategy-head"><h4>Decode 上下文并行</h4><span class="pg-abbr">DCP</span></div><dl>
<div><dt>分担</dt><dd>Decode 时按上下文位置分散的历史 KV；每卡读一部分历史，再合并当前 Query 的局部 Attention。</dd></div>
<div><dt>不消失</dt><dd>总历史位置仍是 4；每卡读 2 个不等于总读取或总算力减半，当前 Query 仍要看全部允许历史。</dd></div>
<div><dt>代价</dt><dd>每步 Decode 交换局部 O 与归一化量 Z/LSE；短上下文时通信可能占主导，ITL 可能变差。</dd></div>
<div><dt>边界</dt><dd>必须按 Z 加权：两片局部 O 分别为 2、6，Z 分别为 2、6 时，加权结果为 5；两片 Z 分别为 2、2（局部 O 仍为 2、6）时，结果为 4。局部平均只在 Z 相等时碰巧正确。</dd></div>
</dl><div class="pg-links"><a href="?mv=10#lesson-animation">看不等权 ↗</a><a href="?mv=11#lesson-animation">看等权 ↗</a></div></article>
</div>
<p class="pg-note"><strong>不是二选一：</strong>PCP 与 DCP 是不同阶段的分工；8 个 Query 位置与 4 个历史位置是两个独立例子，不拿两图数值做速度比较。它们也不等于 TP/PP：TP 切算子，PP 切层。<strong>与切块的关系：</strong>Chunked Prefill 是按轮分段时间切块，不是 PCP/DCP 的卡间分工；可以组合，但支持范围要另核。</p><p class="pg-note"><a href="#parallel-space-time">用同一组 p1…p8 对齐“按卡分工”与“每轮范围” →</a></p>
</section>
<section class="pg-panel" id="parallel-goal-activation" data-goal-panel="activation" aria-labelledby="parallel-goal-activation-title">
<div class="pg-panel-head"><h3 id="parallel-goal-activation-title">逐 token 激活占用（SP）</h3><p>长上下文除了历史 KV，逐 token 区域的激活也会随序列长度增加。经典 SP 在 TP 组内沿序列切分这部分激活，减少每张卡的重复副本；它不改变历史 KV 的分片方式，也不等于全程显存减半。</p></div>
<div class="pg-strategies">
<article class="pg-strategy"><div class="pg-strategy-head"><h4>经典序列并行</h4><span class="pg-abbr">SP</span></div><dl>
<div><dt>分担</dt><dd>同一 TP 组内、逐 token 区域的序列行；长序列时每卡只保留本地行，减少这部分激活的重复副本。本例每卡完整行由 4 变为 2。</dd></div>
<div><dt>不消失</dt><dd>不新增 GPU；不改变总 KV、完整 Attention 上下文、模型权重与总计算。激活分片本身仍占空间，只是不再每卡复制整段。</dd></div>
<div><dt>代价</dt><dd>进入 TP 区域前 AllGather，离开时 ReduceScatter；总通信取决于实现与原 TP collective，不能笼统写成固定新增两次。</dd></div>
<div><dt>边界</dt><dd>这是 Megatron 经典 SP 的教学布局，不声称是某引擎的通用开关；它不替代 PCP/DCP 对历史 KV 与 Attention 上下文的分工，也不把激活分片等同于 KV 分片。</dd></div>
</dl><div class="pg-links"><a href="?mv=7#lesson-animation">看 TP+SP 布局 ↗</a></div></article>
</div>
<p class="pg-note">SP 通常与 TP 一起出现，也可放进更大的 PP/DP 配置；实际支持范围按引擎与拓扑核对。<strong>它减轻的是 TP 组内逐 token 激活的重复保存，</strong>不减总 KV，也不代表全程显存减半。</p>
</section>
</div>
</section>"""


def deployment_contrast_summary():
    """No-JS summary of the same A/B placement; the live panel draws the tracks."""
    return """<section class="deployment-contrast-static" id="deployment-contrast" aria-labelledby="deployment-contrast-title">
<h2 id="deployment-contrast-title">部署对照速览：同一个 A、B 落在哪</h2>
<p>同一条正在 Decode 的 A 与同一条长输入 B。两侧完成同一组工作：A 4 次 Decode；B 4 段 Prefill 加首次 Decode。基线是一组共享执行容量；对照是 P、D 两组独立容量，各能执行完整模型。右边确实多一组资源，不是把同一组资源变成两倍速，也不是把模型按层切成前后半段。</p>
<table class="fd-table"><caption>资源与任务安排（时隙是教学顺序，不是实测耗时）</caption><thead><tr><th scope="col">观察点</th><th scope="col">一个共享服务组</th><th scope="col">独立 P / D 服务组</th></tr></thead><tbody>
<tr><th scope="row">资源</th><td>1 组共享执行容量；A、B 在同一组完成两个阶段。</td><td>2 组独立容量；P、D 各自加载完整模型，按阶段分工。</td></tr>
<tr><th scope="row">完成的工作</th><td>A 4 次 Decode；B 4 段 Prefill + 首次 Decode（t6）。</td><td>同样 A 4 次 Decode（t1–t4）；B 4 段 Prefill（t1–t4）+ 首次 Decode（t6）。</td></tr>
<tr><th scope="row">A 的 Decode</th><td>与 B 的 Prefill 争同一组容量；所选教学调度中，B 单独占轮时 A 等待，但 A 仍完成 4 次 Decode。</td><td>在 D 组完成同样 4 次 Decode（t1–t4）；P 组执行 B 的 Prefill 时不占 D 组的容量。</td></tr>
<tr><th scope="row">B 的 Prefill / 首个 Decode</th><td>Prefill 可分 4 段并与 A 同批 mixed；KV 留在本组，B 的首次 Decode 也在同组排队（t6）。</td><td>Prefill 在 P 组；t5 把 KV 副本、y1 与配对信息交到 D，t6 就绪门通过后 B 才首次 Decode。</td></tr>
<tr><th scope="row">成本与代价</th><td>省去跨组交接；代价是同一组执行容量会互相等待。</td><td>多一组部署与协调，增加传输、接收容量和就绪等待。</td></tr>
</tbody></table>
<p>合并部署也可以换 chunk、优先级或 mixed 方式；图中的 4 段 Prefill 和等待位置只是所选教学调度，不表示合并只能串行。角色与成本的完整对照见 <a href="#deployment-comparison">角色与成本说明：排队、状态与差异</a>；交接过程见下方主动画。</p>
</section>"""


def serving_adoption_summary():
    """No-JS summary of the same R1/R2 placement; the live panel draws the slots."""
    return """<section class="serving-adoption-static" id="serving-adoption" aria-labelledby="serving-adoption-title">
<h2 id="serving-adoption-title">服务多实例速览：同一个 R1、R2 落在哪</h2>
<p>同一个 R1、R2 与同一个完整模型。一个实例让两条请求共享队列和一个执行槽；两个实例各接一条，各自拥有队列、执行槽和一份完整模型。两边的“执行”都必须先接纳并拿到本轮槽位。</p>
<div class="sa-static-wrap"><table class="fd-table"><caption>接纳、等待/执行与副本成本（本例每个实例每轮 1 条请求）</caption><thead><tr><th scope="col">观察点</th><th scope="col">1 个实例 · 共享容量</th><th scope="col">2 个实例 · 各接一条</th></tr></thead><tbody>
<tr><th scope="row">接纳与队列</th><td>R1、R2 都进入同一队列；两条都已接纳，但只有一个执行槽。</td><td>R1 进入实例 B，R2 进入实例 A；两条各有自己的队列和执行槽。</td></tr>
<tr><th scope="row">等待 / 执行</th><td>R1 占用执行槽，R2 等待；R1 完成后 R2 才能进入同一槽位。</td><td>R1 在 B、R2 在 A 同时执行；各实例等待 0 条，但并行不是单条请求变快。</td></tr>
<tr><th scope="row">参数副本</th><td>完整模型副本 1 份。</td><td>完整模型副本 2 份（+1 份权重驻留）。</td></tr>
<tr><th scope="row">代价与边界</th><td>省去第二份模型；两条请求争同一份执行容量。</td><td>多一份完整模型、健康检查与路由协调；KV、激活和通信开销另计。</td></tr>
</tbody></table></div>
<p class="sa-static-note"><strong>容量假设：</strong>本例每个实例每轮只放 1 条请求，R1 先到先执行；真实推理实例可以把多条请求 batch（合批）在同一轮，单个实例不等于只能串行。<strong>health ✓ 只表示探活通过：</strong>它不等于实例空闲，也不等于请求已进入执行槽。主图只追踪 R1，所以双实例主图里的 A 不接 R1；对照补上的 R2→A 是同时到达的另一条请求。</p>
</section>"""


def speculation_adoption_summary():
    """No-JS summary of the same three accepted-prefix cases; the live panel draws the calls."""
    return """<section class="speculation-adoption-static" id="speculation-adoption" aria-labelledby="speculation-adoption-title">
<h2 id="speculation-adoption-title">草稿与验证速览：同一最终提交序列，调用组合不同</h2>
<p>三种条件使用同一组草稿候选和同一条贪心验证规则。下表把“普通目标逐 token 调用”与“4 次草稿 + 1 次目标验证”并排；两种方式最终提交的序列相同，但目标调用、草稿调用和验证调用的成本并不相等，表中次数不是速度比。</p>
<table class="fd-table"><caption>接受 2 / 0 / 4 个候选时的调用与有效状态</caption><thead><tr><th scope="col">条件</th><th scope="col">最终提交序列</th><th scope="col">调用次数</th><th scope="col">有效 KV（根 + 接受前缀）</th></tr></thead><tbody>
<tr><th scope="row">接受 2 个（第 3 个分歧）</th><td>我 想 吃</td><td>普通目标 3 次；采用后 4 次草稿 + 1 次验证，1 次验证覆盖 5 个输入位置</td><td>根 我 想</td></tr>
<tr><th scope="row">接受 0 个（第 1 个分歧）</th><td>你</td><td>普通目标 1 次；采用后 4 次草稿 + 1 次验证，目标调用没有减少，另多 4 次草稿</td><td>根</td></tr>
<tr><th scope="row">接受 4 个（全部一致）</th><td>我 想 喝 茶 。</td><td>普通目标 5 次；采用后 4 次草稿 + 1 次验证，1 次验证覆盖 5 个输入位置</td><td>根 我 想 喝 茶</td></tr>
</tbody></table>
<p class="parallel-adoption-static-note">有效 KV 只含根 token 与接受的连续前缀；替代或补出的 token 尚未作为输入，不计入。1 次验证覆盖 5 个输入位置，不是 5 次单 token 调用；普通目标调用与草稿/验证调用的成本不相等，不能把次数差折算成速度比。</p>
</section>"""


def cache_static_summary():
    """No-JS C02/C03 summary; the live cache panels draw the same objects."""
    return """<section class="parallel-overview cache-static" id="cache-static" aria-labelledby="cache-static-title">
<h2 id="cache-static-title">核心算例：分页分配与前缀共享</h2>
<p>下面的数值与主动画、动态对照使用同一组教学假设；不展开脚本也能读完。先看 3 个新位置怎样进入一个已有旧页的池，再看两个请求的共同前缀怎样变成一份对象加两个引用。</p>
<section class="cache-static-part" aria-labelledby="cache-static-pages-title">
<h3 id="cache-static-pages-title">同一个 8 页池：3 个新位置怎样落位</h3>
<p>池子固定为 8 个物理页 × 每页 2 个位置 = 16 个位置。起始时 6 个旧页各有 2 个有效位置，共 12 个；这 6 页仍归已有请求。页 1、页 4 尚未分配，是仅有的 2 个可分配页。新请求需要 p1、p2、p3 三个位置。</p>
<table class="fd-table"><caption>连续预留与按需分页使用同一个池、同一组需求</caption><thead><tr><th scope="col">观察点</th><th scope="col">连续 3 槽</th><th scope="col">按需分页</th></tr></thead><tbody>
<tr><th scope="row">起始状态</th><td colspan="2">有效位置 12；6 个旧页归已有请求；页 1、页 4 可分配；总空闲 4 槽，但两个空闲页各只有 2 槽，没有跨页的 3 个连续槽。</td></tr>
<tr><th scope="row">放置规则</th><td>要求 3 个相邻空槽；最大连续空槽只有 2，因此放不下，3 个新位置都没有写入。</td><td>逻辑页 0 的 p1、p2 映射到物理页 4 的槽 0、1；逻辑页 1 的 p3 映射到物理页 1 的槽 0。2 个逻辑页映射到 2 个物理页。</td></tr>
<tr><th scope="row">地址记录</th><td>没有合法的连续区间。</td><td>2 条页表映射；另有 3 行逐 token 地址演算，不是 3 条页表项。</td></tr>
<tr><th scope="row">结果</th><td>仍为 12 个有效位置、已分配容量 12、可分配页 2；新请求未放入。</td><td>有效位置 15（12 旧 + 3 新）；已分配容量 16；可分配页 0；末页余量 1。</td></tr>
</tbody></table>
<p>3 行地址演算：p1 → 逻辑页 0 → 物理页 4 槽 0；p2 → 逻辑页 0 → 物理页 4 槽 1；p3 → 逻辑页 1 → 物理页 1 槽 0。末页余量 1 指页 1 的槽 1 仍空着；它属于新请求的末页，不是池里还可分配的页。1 个余量不等于 1 个可分配页。</p>
<p><strong>所有权不能合并成一句话：</strong>分页后 8 个物理页都有归属，但不能说全部 8 页都属于新请求。6 个旧页仍归已有请求，只有原来的 2 个空闲页（页 4、页 1）被新请求使用。</p>
</section>
<section class="cache-static-part" aria-labelledby="cache-static-prefix-title">
<h3 id="cache-static-prefix-title">两个请求共享共同前缀：6 → 3，总 10 → 7</h3>
<p>R1、R2 都从共同前缀 p1 p2 p3 开始；之后分别接自己的后缀 a1 a2 与 b1 b2。未共享时两个请求各保存一份前缀；共享后前缀只保存一份，由两个请求引用，后缀仍各自独立。</p>
<table class="fd-table"><caption>共同前缀与独立后缀的位置数</caption><thead><tr><th scope="col">对象</th><th scope="col">未共享 · 各存一份</th><th scope="col">共享 · 1 份前缀 + 2 引用</th></tr></thead><tbody>
<tr><th scope="row">共同前缀 p1 p2 p3</th><td>2 × 3 = 6 个位置</td><td>1 × 3 = 3 个位置，refs 2</td></tr>
<tr><th scope="row">独立后缀</th><td>a1 a2 与 b1 b2，共 4 个位置</td><td>仍是 a1 a2 与 b1 b2，共 4 个位置，不合并</td></tr>
<tr><th scope="row">总位置</th><td>6 + 4 = 10</td><td>3 + 4 = 7；省下 3 个前缀位置</td></tr>
</tbody></table>
<p><strong>复用条件：</strong>token 前缀、模型与权重、影响计算的配置/状态都要匹配；任一项不匹配就不共享，回到各自保存。后缀相同或语义相近，不能代替前缀匹配。</p>
<p><strong>引用与驱逐：</strong>R1 持有 → R2 接入，refs 从 1 到 2 → R1 结束，refs 回到 1 → R2 结束，refs 为 0。refs 0 只表示没有活跃请求，不等于立即驱逐；缓存仍可按策略保留，驱逐是另一个条件。共享的是前缀对象，不是后缀；R1 先结束也不会提前释放 R2 仍在用的前缀。</p>
</section>
<p>这两组数字是容量与位置计数，不是速度、吞吐或显存占用实测；分页的代价是额外页表与不连续访问，共享的代价是引用管理与兼容检查。</p>
</section>"""


def render(c, i):
    page = PAGES / 'foundations' / (c['slug'] + '.html')
    title = escape(c['title'])
    prerequisites = '从这一课开始，不要求推理知识。' if not c['prerequisites'] else '先看：' + ' · '.join(f'<a href="{slug}.html">{escape(FOUNDATION_BY_SLUG[slug]["title"])}</a>' for slug in c['prerequisites'])
    terms = ''.join(f'<div><dt>{escape(a)}</dt><dd>{escape(b)}</dd></div>' for a, b in c['terms'])
    sections = ''.join(f'<section id="concept-{n}"><h2>{escape(a)}</h2><p>{escape(b)}</p></section>' for n, (a, b) in enumerate(c['sections']))
    terms_html = f'<dl class="foundation-terms" aria-label="这课用到的词">{terms}</dl>'
    if c['slug'] == 'communication':
        terms_html = f'<details class="parallel-glossary"><summary>展开：Rank、Root、Count、RMA 与 Stream</summary>{terms_html}</details>'
    reading_entry = ''
    if len(c['sections']) >= 4 and not c.get('reading_groups'):
        reading_entry = '<nav class="parallel-reading-nav" aria-label="本课阅读入口">' + ''.join(f'<a href="#concept-{n}">{escape(a.split("：")[0])}</a>' for n, (a, b) in enumerate(c['sections'])) + '</nav>'
    if c.get('reading_groups'):
        sections = reading_guide(c)
        terms_html = f'<details id="parallel-glossary" class="parallel-glossary"><summary>术语速查 · 英文全称与含义</summary>{terms_html}</details>'
        reading_entry = '<nav class="parallel-reading-nav" aria-label="本课阅读入口"><a href="#parallel-animation">操作动画</a><a href="#parallel-overview">机制速览</a><a href="#parallel-glossary">术语速查</a><a href="#evidence">来源与边界</a></nav><div id="parallel-animation"></div>'
    # Keep existing per-mechanism navigation beside its original content.
    reading_entry = reading_entry.replace('aria-label="本课阅读入口"', 'aria-label="算例与机制入口"')
    reading_entry = reading_entry.replace('<div id="parallel-animation"></div>', '')
    animation_anchor = '<div id="parallel-animation"></div>' if c.get('reading_groups') else ''
    learning_entry = '<nav class="parallel-reading-nav" aria-label="本课阅读入口"><a href="#understand">理解设计</a><a href="#lesson-animation">观察过程</a><a href="#lesson-glossary">术语速查</a><a href="#lesson-examples">算例与取舍</a><a href="#evidence">来源</a></nav>'
    if c['slug'] == 'parallel':
        learning_entry = learning_entry.replace('<a href="#understand">理解设计</a>', '<a href="#parallel-goals">先问哪里不够</a><a href="#understand">理解设计</a>')
        learning_entry = learning_entry.replace('<a href="#lesson-animation">观察过程</a>', '<a href="#parallel-space-time">两轴对照</a><a href="#lesson-animation">观察过程</a>')
    if c['slug'] == 'deployment':
        learning_entry = learning_entry.replace('<a href="#lesson-animation">观察过程</a>', '<a href="#deployment-contrast">部署对照</a><a href="#lesson-animation">观察过程</a>')
    if c['slug'] == 'serving':
        learning_entry = learning_entry.replace('<a href="#lesson-animation">观察过程</a>', '<a href="#serving-adoption">服务对照</a><a href="#lesson-animation">观察过程</a>')
    if c['slug'] == 'speculation':
        learning_entry = learning_entry.replace('<a href="#lesson-animation">观察过程</a>', '<a href="#speculation-adoption">调用对照</a><a href="#lesson-animation">观察过程</a>')
    if c['slug'] == 'cache':
        learning_entry = learning_entry.replace('<a href="#lesson-animation">观察过程</a>', '<a href="#cache-static">核心摘要</a><a href="#lesson-animation">观察过程</a>')
    if c['slug'] == 'attention':
        learning_entry = learning_entry.replace('<a href="#lesson-animation">观察过程</a>', '<a href="#attention-motive">先看动机</a><a href="#lesson-animation">观察过程</a><a href="#multihead">多头算例</a>')
    sources = ''.join(f'<li><a href="{escape(url, quote=True)}">{escape(name)}</a></li>' for name, url in c['sources'])
    visual_refs = ''
    if c.get('visual_references'):
        visual_refs = '<details class="concept-background"><summary>画法参考与演示范围</summary><ul>' + ''.join(f'<li><a href="{escape(url, quote=True)}">{escape(name)}</a>：{escape(note)}</li>' for name, url, note in c['visual_references']) + '</ul><p>本课数值、交互模型与图形独立实现；输入输出和完成条件按上方官方文档复核。参考资料读取：2026-10-19。</p></details>'
    choices = ''.join(f'<label><input type="radio" name="prediction" value="{n}"><span>{escape(value)}</span></label>' for n, value in enumerate(c['choices']))
    previous = '<a href="../#chapter-foundations"><small>第一章目录</small><strong>回到推理基础</strong></a>' if i == 0 else f'<a href="{FOUNDATIONS[i-1]["slug"]}.html"><small>上一课</small><strong>{escape(FOUNDATIONS[i-1]["title"])}</strong></a>'
    following = '<a class="next" href="../#chapter-sglang"><small>进入第二章</small><strong>SGLang 实现剖析</strong></a>' if i == len(FOUNDATIONS)-1 else f'<a class="next" href="{FOUNDATIONS[i+1]["slug"]}.html"><small>下一课</small><strong>{escape(FOUNDATIONS[i+1]["title"])}</strong></a>'
    if c['slug'] == 'parallel':
        static_summary = parallel_adoption_summary()
    elif c['slug'] == 'deployment':
        static_summary = deployment_contrast_summary()
    elif c['slug'] == 'serving':
        static_summary = serving_adoption_summary()
    elif c['slug'] == 'speculation':
        static_summary = speculation_adoption_summary()
    elif c['slug'] == 'cache':
        static_summary = cache_static_summary()
    else:
        static_summary = ''
    extension = EXTENSIONS.get(c['slug'])
    extension_assets = extension_html = ''
    mechanism_assets = '<script src="../motion/batching.js" defer></script><link rel="stylesheet" href="../motion/teaching.css"><script src="../motion/teaching-visuals.js" defer></script><script src="../motion/dynamic-batching.js" defer></script><script src="../motion/batching-chunk.js" defer></script>' if c['slug'] == 'batching' else ''
    if c['slug'] in {'tokens', 'generation', 'attention', 'cache', 'deployment'}:
        mechanism_assets += '<link rel="stylesheet" href="../motion/teaching.css"><script src="../motion/teaching-scenes.js" defer></script><script src="../motion/teaching-visuals.js" defer></script>'
        if c['slug'] == 'attention':
            mechanism_assets += '<script src="../motion/attention-visual.js" defer></script><link rel="stylesheet" href="../motion/components.css"><script src="components-model.js" defer></script><script src="../motion/attention-motive.js" defer></script><link rel="stylesheet" href="../motion/attention-layer.css"><script src="../motion/attention-layer.js" defer></script><link rel="stylesheet" href="../motion/attention-multihead.css"><script src="../motion/attention-multihead.js" defer></script>'
        elif c['slug'] != 'deployment':
            mechanism_assets += '<script src="../motion/basics-visuals.js" defer></script>'
        if c['slug'] == 'generation':
            mechanism_assets += '<link rel="stylesheet" href="../motion/generation-sampling.css"><script src="../motion/generation-sampling.js" defer></script><link rel="stylesheet" href="../motion/generation-streaming.css"><script src="../motion/generation-streaming.js" defer></script>'
        if c['slug'] == 'cache':
            mechanism_assets += '<link rel="stylesheet" href="../motion/cache-adoption.css"><script src="../motion/cache-adoption.js" defer></script>'
        if c['slug'] == 'deployment':
            mechanism_assets += '<link rel="stylesheet" href="../motion/deployment-adoption.css"><script src="../motion/deployment-adoption.js" defer></script>'
    if c['slug'] == 'tokens':
        mechanism_assets += '<link rel="stylesheet" href="../motion/tokens.css"><script src="../motion/tokens.js" defer></script>'
    if c['slug'] == 'components':
        mechanism_assets += '<link rel="stylesheet" href="../motion/teaching.css"><link rel="stylesheet" href="../motion/components.css"><script src="../motion/teaching-visuals.js" defer></script><script src="components-model.js" defer></script><script src="../motion/components.js" defer></script>'
    if c['slug'] in {'hardware', 'parallel', 'communication', 'speculation', 'serving'}:
        mechanism_assets += '<link rel="stylesheet" href="../motion/teaching.css"><script src="../motion/teaching-visuals.js" defer></script><script src="../motion/detail-scenes.js" defer></script>'
        if c['slug'] == 'serving':
            mechanism_assets += '<link rel="stylesheet" href="../motion/serving-adoption.css"><script src="../motion/serving-adoption.js" defer></script>'
        mechanism_assets += '<script src="../motion/detail-visuals.js" defer></script>'
    if c['slug'] == 'metrics':
        mechanism_assets += '<link rel="stylesheet" href="../motion/teaching.css"><script src="../motion/teaching-visuals.js" defer></script><script src="../motion/metrics.js" defer></script>'
    if c['slug'] == 'communication':
        mechanism_assets += '<link rel="stylesheet" href="../motion/communication.css"><script src="communication-model.js" defer></script><script src="../motion/communication-visuals.js" defer></script>'
    if c['slug'] == 'parallel':
        mechanism_assets += '<link rel="stylesheet" href="../motion/parallel-adoption.css"><link rel="stylesheet" href="../motion/parallel-pcp.css"><link rel="stylesheet" href="../motion/parallel-dcp.css"><script src="parallel-model.js" defer></script><script src="../motion/parallel-scenes.js" defer></script><script src="../motion/parallel-tensor.js" defer></script><script src="../motion/parallel-routing.js" defer></script><script src="../motion/parallel-dp.js" defer></script><script src="../motion/parallel-adoption.js" defer></script><script src="../motion/parallel-pcp.js" defer></script><script src="../motion/parallel-visuals.js" defer></script><link rel="stylesheet" href="../motion/parallel-goals.css"><script src="../motion/parallel-goals.js" defer></script><link rel="stylesheet" href="../motion/parallel-space-time.css"><script src="../motion/parallel-space-time.js" defer></script>'
    if extension:
        extension_assets = '<link rel="stylesheet" href="../sglang/inference-overview/overview.css"><link rel="stylesheet" href="../diagrams.css"><script src="../sglang/inference-overview/model.js" defer></script><script src="../diagrams.js" defer></script><script src="../sglang/inference-overview/overview.js" defer></script>'
        if c['slug'] == 'generation':
            extension_assets += '<link rel="stylesheet" href="../motion/journey-layer.css"><script src="../motion/journey-layer.js" defer></script>'
        extension_html = (f'<details class="foundation-extension" id="mechanism-extension"><summary>选读 · {escape(extension["title"])}</summary>'
            '<p>主线之外的完整机制对照图。沿用原课程的教学假设；不代表 SGLang 的运行轨迹。</p>'
            + extension['html'] + f'<p><a href="../sglang/inference-overview/{extension["source_lesson"]}">转到 SGLang：核对实现接口与固定源码 →</a></p></details>')
    motive_html = ''
    if c['slug'] == 'attention':
        motive_html = '''<section id="attention-motive" class="attention-motive" aria-labelledby="attention-motive-title">
<h2 id="attention-motive-title">先看动机：同一组 Q/K/V，只读自己与读取上下文</h2>
<p>同一个 Query 与同一组 K/V，只改变允许读取的范围。先看只读自己的结果，再看 p1 的历史贡献怎样进入表示。</p>
<div id="attention-motive-app" class="motive-card">
 <p class="motive-boundary">交互版本需要 JavaScript。固定例：Q=[0,1]，K1=[1,0]，K2=[0,1]，V1=[2,0]，V2=[0,2]。</p>
 <table class="fd-table"><caption>同一输入的两种读取范围</caption><thead><tr><th>读取范围</th><th>权重</th><th>结果 m</th></tr></thead><tbody>
 <tr><th scope="row">只读自己 p2</th><td>[0, 1]</td><td>[0, 2]</td></tr>
 <tr><th scope="row">允许读 p1、p2</th><td>[0.33023845, 0.66976155]</td><td>[0.6604769, 1.3395231]</td></tr>
 </tbody></table>
 <p class="motive-boundary">这是解释性对照，不是性能优化或语义正确率。</p>
</div>
</section>'''
    multihead_html = ''
    if c['slug'] == 'attention':
        multihead_html = '''<section id="multihead" class="mh-section" aria-labelledby="multihead-title">
<p class="eyebrow">独立算例 · 2 头 × 每头 2 维 · 与上方单头 Query 分开</p>
<h2 id="multihead-title">多头：同一输入 x 的不同投影，最后怎样合并？</h2>
<p class="mh-lead">上方单头演示回答“一个 Query 怎样汇合上下文”。这里换成明确标注的 2 头小例：x=[1,1] 是投影前输入，分别乘 Wq0、Wq1，得到两个头各自的 Query q0、q1；两头再各自形成 w×V 结果，按头序 concat 成 4 维，最后只由 Wo 投回 2 维。上方单头演示的位置、遮罩与观察方式保持不变。</p>
<div class="mh-app" data-multihead>
<noscript><p>本例取 x=[1,1]，两个位置 K=[[1,0],[0,1]]、V=[[2,0],[0,2]]；Wq0=[[1,0],[0,0]]、Wq1=[[0,0],[0,1]]。两头权重分别为 [66.98%, 33.02%] 与 [33.02%, 66.98%]，结果为 [1.3395, 0.6605] 与 [0.6605, 1.3395]；concat 为 4 维 [1.3395, 0.6605, 0.6605, 1.3395]，教学 Wo 再投回 [1,1]。</p></noscript>
</div>
</section>'''
    lab_html = '<div data-foundation-lab="components" hidden></div>' if c['slug'] == 'components' else f'<section class="foundation-lab" data-foundation-lab="{c["slug"]}" data-full="true" data-title="{title}"><h2>走一遍具体过程</h2><p>{escape(c["goal"])}</p><p>交互需要 JavaScript；下面保留完整解释和自测答案。</p></section>'
    goal_selector = parallel_goal_selector() if c['slug'] == 'parallel' else ''
    space_time_entry = parallel_space_time_entry() if c['slug'] == 'parallel' else ''
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title} · 推理基础</title><meta name="description" content="{escape(c['lead'], quote=True)}"><link rel="icon" href="data:,"><link rel="stylesheet" href="../site.css"><link rel="stylesheet" href="foundations.css"><script src="../site.js" defer></script><script src="../motion/integration.js" defer></script><script src="model.js" defer></script><script src="player.js" defer></script><link rel="stylesheet" href="../motion/motion.css"><script src="../vendor/gsap/gsap.min.js" defer></script><script src="../motion/scenes.js" defer></script>{mechanism_assets}<script src="../motion/camera.js" defer></script><script src="../motion/boundaries.js" defer></script><script src="../motion/player.js" defer></script>{extension_assets}</head>
<body data-page="foundations" data-lesson="{c['slug']}"><a class="skip-link" href="#main-content">跳到正文</a>
<header class="site-header"><a class="site-brand" href="../"><span class="brand-mark" aria-hidden="true">ai<span>·</span></span><span>AI Infra Wiki<small>推理系统学习手册</small></span></a><div class="site-actions"><a class="repo-link" href="https://github.com/Asher-XunZhang/ai-infra-wiki">GitHub ↗</a><button class="theme-toggle" type="button" aria-label="切换配色">配色：系统</button></div></header>
<div class="site-layout">{sidebar(page, None)}<div class="site-content"><main id="main-content">
<nav class="learning-breadcrumb" aria-label="当前位置"><a href="../">学习首页</a><span aria-hidden="true">/</span><a href="../#chapter-foundations">第一章 · 推理基础</a><span aria-hidden="true">/</span><span aria-current="page">{title}</span></nav>
<header class="lesson-header"><div class="lesson-kicker"><a class="eyebrow" href="../#chapter-foundations">推理基础 · {i+1:02} / {len(FOUNDATIONS):02}</a><span class="lesson-count">通用原理</span></div><h1>{title}</h1><p>{escape(c['lead'])}</p><div class="meta-tags"><span>可单步操作</span><span>原理与实现分开</span><a href="#evidence">来源与教学边界</a></div></header>
<p class="foundation-prereqs">{prerequisites}</p>
<p class="foundation-intro">{escape(c['intro'])}</p>
<section class="concept-problem" aria-labelledby="concept-problem-title"><h2 id="concept-problem-title">先想清楚：这一课解决什么问题？</h2><p>{escape(CONCEPTS[c['slug']]['problem'])}</p></section>
{motive_html}
{learning_entry}{goal_selector}{space_time_entry}{static_summary}
<div id="lesson-animation"></div>
{animation_anchor}
{lab_html}
{multihead_html}
{concept_guide(c)}
<section id="lesson-glossary"><h2>术语速查</h2>{terms_html}</section>
<section id="lesson-examples"><h2>对照算例，回看机制与取舍</h2>{reading_entry}<div class="foundation-copy">{sections}</div></section>
<form class="fd-quiz" data-answer="{c['answer']}" data-feedback="{escape(c['feedback'], quote=True)}" hidden><fieldset><legend>{escape(c['question'])}</legend>{choices}</fieldset><button type="submit">检查判断</button><p class="fd-feedback" aria-live="polite"></p></form>
<details class="foundation-fallback"><summary>自测与解释：{escape(c['question'])}</summary><p>{escape(c['feedback'])}</p></details>
{extension_html}
<a class="foundation-to-engine" href="../{c['implementation']}"><small>对应的 SGLang 实现 · 保留独立源码基线</small>带着这一课的问题，进入实现剖析 →</a>
<section id="evidence" class="foundation-scope"><h2>来源与教学边界</h2><p>资料类型：基础原理与教学算例。读取 / 整理日期：{escape(c.get("reviewed", "2026-10-15"))}。{escape(c['scope'])}</p><p>概念动机、取舍与背景整理：{escape(c.get("reviewed", "2026-10-16"))}。原论文、官方说明及相关课程就近列在对应解释下；其中数值小例子为独立教学假设，不替代原动画的分词器版本或源码基线。</p><ul class="foundation-sources">{sources}</ul>{visual_refs}<p>图与文字由整理者设计；没有启动模型服务、GPU 或网络性能实验。通用原理不规定任何特定版本 SGLang 的完整执行顺序。</p></section>
<nav class="lesson-pager" aria-label="推理基础内继续学习">{previous}{following}</nav>
</main><footer class="site-footer"><span>AI Infra Wiki · 从原理走向实现</span><a href="../">两章完整目录</a></footer></div></div></body></html>
'''


def build(check=False):
    stale = []
    for i, c in enumerate(FOUNDATIONS):
        path = PAGES / 'foundations' / (c['slug'] + '.html')
        result = render(c, i)
        if not path.exists() or path.read_text(encoding='utf-8') != result:
            if check:
                stale.append(str(path.relative_to(ROOT)))
            else:
                path.write_text(result, encoding='utf-8')
    if stale:
        raise SystemExit('Foundation pages are stale: ' + ', '.join(stale))
    print(('Checked' if check else 'Generated') + f' {len(FOUNDATIONS)} foundational lessons.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    build(parser.parse_args().check)
