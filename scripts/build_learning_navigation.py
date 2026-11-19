#!/usr/bin/env python3
"""Render the shared learning framework into source HTML; --check detects drift."""
import argparse
import json
import os
import re
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / 'pages'
FOUNDATIONS = json.loads((PAGES / 'foundations/courses.json').read_text(encoding='utf-8'))
FOUNDATION_BY_SLUG = {c['slug']: c for c in FOUNDATIONS}
DESIGN_NOTES = json.loads((PAGES / 'sglang/design-notes.json').read_text(encoding='utf-8'))
LEARNING_PATHS = json.loads((PAGES / 'learning-paths.json').read_text(encoding='utf-8'))
from system_architecture import render_architecture

ENGINE_COMPARISON = 'engine-comparison/index.html'
ENGINE_PAGE = PAGES / ENGINE_COMPARISON
ENGINE_TOPICS = [
    ('process', '进程与请求路径', 'Processes & request path', '从 API 入口追到调度、模型执行与文本回传。'),
    ('wire', '进程间通信', 'Communication', '沿进程边界辨认通道、载荷和控制消息。'),
    ('prefix', '前缀缓存', 'Prefix cache', '看相同前缀怎样命中共享 KV 地址，尾部从哪里补算。'),
    ('batch', '调度与批处理', 'Scheduling & batching', '看一轮给哪些请求分配 token 预算，执行结果何时生效。'),
    ('crosscut', '生命周期与执行', 'Lifecycle & execution', '看正常结束、内存压力和取消怎样改变资源归属，以及执行怎样选择 CUDA Graph。'),
    ('disagg', 'P/D 分离与扩展', 'P/D disaggregation', '看实例选择、Prefill → Decode 的 KV 交接和消费条件。'),
    ('metrics', '延迟口径', 'Metric boundaries', '找到 TTFT、E2E、排队时间和 ITL 的计时点与样本单位。'),
]

DESIGN_REVISION = '279339f113b79af84f27fd3ac92d0a13bd3f4cbd'

STAGES = [
    ('foundations', '第一层 · 建立全局', '从请求认识系统'),
    ('instance', '第二层 · 理解单实例', '循环、执行、缓存与策略'),
    ('distributed', '第三层 · 扩展到多卡多机', '并行、通信与分离部署'),
    ('serving', '第四层 · 理解功能与服务', '模型特性与服务运行'),
    ('practice', '第五层 · 综合验证', '性能分析与源码实践'),
]
# A lesson has one primary module. Other modules may link to it as a related case.
MODULES = [
    ('system-overview', '推理系统全景', 'foundations', False, [
        ('sglang/system-overview/index.html', 'SGLang 代码架构：模块划分与协作全景', 'SGLang · 架构入口')]),
    ('request-runtime', '请求生命周期与运行时架构', 'foundations', False, [
        ('sglang/request-runtime/index.html', '请求生命周期：Req、TokenizerManager 与 Scheduler', 'SGLang · 入门')]),
    ('loop-journey', '调度循环与批次执行旅程', 'instance', False, [
        ('sglang/inference-overview/journey.html', 'ScheduleBatch：Extend 与 Decode 怎样准备输入？', 'SGLang · 源码导读'),
        ('sglang/request-state/index.html', 'Req 状态与队列流转：请求为什么还没继续？', 'SGLang · 状态与引用'),
        ('sglang/loop-journey/index.html', '调度循环：Normal、Overlap 与 PP 的完整旅程', 'SGLang · 时间与依赖'),
        ('sglang/inference-overview/scheduling.html', 'Scheduler：选批、Chunked Prefill 与 Decode 更新', 'SGLang · 源码导读')]),
    ('model-execution', '模型执行与设备协作', 'instance', False, [
        ('sglang/model-execution/index.html', 'ModelRunner 与 CUDA Graph：输入怎样交给 GPU？', 'SGLang · 执行机制'),
        ('sglang/inference-overview/transformer.html', 'DeepSeek-V3：Embedding、Decoder Layer 与输出头', 'SGLang · 源码导读')]),
    ('kv-memory', 'KV Cache 与资源生命周期', 'instance', False, [
        ('sglang/kv-memory/index.html', 'KV Memory Pool：地址映射、分页与缓存回收', 'SGLang · 缓存机制'),
        ('sglang/inference-overview/kv-cache.html', 'RadixCache 与 Attention：前缀索引怎样连接 KV？', 'SGLang · 源码导读')]),
    ('scheduling', '调度策略、批处理与资源约束', 'instance', False, [
        ('sglang/scheduling/index.html', 'Batch 调度：准入预算、Chunked Prefill 与 Retract', 'SGLang · 调度机制')]),
    ('parallelism', '并行与执行拓扑', 'distributed', False, [
        ('sglang/parallelism/index.html', 'TP / PP / DP / EP / CP：SGLang 并行分工', 'SGLang · 并行分工')]),
    ('communication', '通信与传输', 'distributed', False, [('sglang/communication/index.html', 'IPC、Collective 与 KV Transfer：数据何时可用？', 'SGLang · 通信机制')]),
    ('disaggregation', '分离部署与分布式状态交接', 'distributed', False, [
        ('sglang/inference-overview/deployment.html', 'PD Disaggregation：Bootstrap、KV 接收与准入', 'SGLang · 源码导读'),
        ('sglang/pd-prefill-lifecycle/index.html', 'PD + PP：Prefill 请求生命周期与 KV 释放', 'SGLang · PD＋PP 案例'),
        ('sglang/pd-dataflow/index.html', 'PD 数据流：激活、KV Transfer 与 Req 队列引用', 'SGLang · PD＋PP 案例'),
        ('sglang/pp-consensus/index.html', 'PP 共识：Bootstrap、Release 与 HiCache ACK', 'SGLang · 状态协调')]),
    ('advanced-generation', '模型结构与高级生成', 'serving', False, [('sglang/advanced-generation/index.html', 'Speculative Decoding 与约束生成：状态怎样提交？', 'SGLang · 高级生成')]),
    ('serving-operations', '服务部署与运行治理', 'serving', False, [('sglang/serving-operations/index.html', '服务治理：Health Check、Timeout 与 Cancellation', 'SGLang · 服务治理')]),
    ('performance', '性能分析与优化方法', 'practice', False, [('sglang/performance-engineering/index.html', 'TTFT、ITL、TPOT 与 P95：SGLang 性能观测', 'SGLang · 性能分析')]),
    ('case-studies', 'PD＋PP 综合实践', 'practice', False, [
        ('sglang/pd-prefill-pp-loop/quick.html', 'Prefill Batch 的执行旅程：跨级计算与异步收尾', 'SGLang · 综合实践'),
        ('sglang/pd-prefill-pp-loop/index.html', 'Prefill 流水线的等待与依赖：空档从哪里来', 'SGLang · PD＋PP 案例'),
        ('sglang/pd-prefill-pp-loop/notes.html', 'Prefill PP 主循环源码：槽位复用与异步收尾', 'SGLang · PD＋PP 案例')]),
]

# Reading order is independent of the stable module taxonomy. Links are rendered
# into HTML so the Pages build can version them and no-JS readers can use them.
READING_ROUTES = [
    ('first', '从请求走到一轮计算', '先认识对象和本轮输入，再读执行与资源约束。', [
        ('sglang/system-overview/index.html', '系统与进程地图', '先沿单实例的请求和返回闭环认模块。'),
        ('sglang/request-runtime/index.html', '同一请求的不同对象', '从前端状态追到调度请求。'),
        ('sglang/inference-overview/journey.html', '本轮输入怎样组成', '分清完整历史、本轮位置和写入地址。'),
        ('sglang/request-state/index.html', '请求由谁继续持有', '区分字段、batch 引用与资源。'),
        ('sglang/loop-journey/index.html', '本轮与后续轮次怎样接续', '再看 Normal、Overlap 与 PP 的依赖。'),
        ('sglang/model-execution/index.html', '选好的工作怎样交给设备', '进入 Worker、Runner 和 Graph 资格。')]),
    ('implementation', '沿资源与调度深入', '在单实例主线上核对内存、索引、准入和具体模型。', [
        ('sglang/kv-memory/index.html', '先读 KV 地址与分配', '逻辑位置映射到物理位置。'),
        ('sglang/inference-overview/kv-cache.html', '再读前缀树索引', '解释匹配、保护与逐叶驱逐。'),
        ('sglang/inference-overview/scheduling.html', '定位每轮选批入口', '在循环中找到续算与成员更新。'),
        ('sglang/scheduling/index.html', '改变预算观察准入', '区分候选顺序、额度与回撤。'),
        ('sglang/inference-overview/transformer.html', '跟进具体模型调用', '用 DeepSeek-V3 核对 PP 层段和输出责任。')]),
    ('questions', '带着现象找原因', '选择与你当前现象有关的一课，按需回补先修知识。', [
        ('sglang/performance-engineering/index.html', '首字为什么慢？', '先确定计时边界和等待发生在哪。'),
        ('sglang/kv-memory/index.html', '显存为什么不够？', '检查空间占用与仍被引用的状态。'),
        ('sglang/parallelism/index.html', '多卡为什么还在等？', '区分工作切分与前后级依赖。'),
        ('sglang/serving-operations/index.html', '服务在线却没回答？', '分清健康、接纳和实际完成。')]),
]


LESSON_TITLES = {lesson[0]: lesson[1] for module in MODULES for lesson in module[4]}
LESSON_TITLES.update({'foundations/' + c['slug'] + '.html': c['title'] for c in FOUNDATIONS})
LESSON_TITLES.update({ENGINE_COMPARISON + '#' + key: '引擎对照：' + label for key, label, _, _ in ENGINE_TOPICS})
READING_ROUTES.insert(0, ('beginner', LEARNING_PATHS['beginner']['title'], LEARNING_PATHS['beginner']['intro'],
    [(step['path'], step['path'], step['hint']) for step in LEARNING_PATHS['beginner']['steps']]))
READING_ROUTES = [(key, title, intro, [(path, LESSON_TITLES.get(path, label), hint) for path, label, hint in lessons])
                  for key, title, intro, lessons in READING_ROUTES]

def route_href(page, target, route):
    parts = urlsplit(target)
    return href(page, parts.path) + '?learn=' + route + ('#' + parts.fragment if parts.fragment else '')


def route_outline(page, route):
    rid, title, intro, lessons = route
    items = ''.join(f'<li><a href="{route_href(page, path, rid)}">{escape(label)}</a><small>{escape(hint)}</small></li>' for path, label, hint in lessons)
    return f'<p>{escape(intro)}</p><ol class="route-steps">{items}</ol>'


def topic_matrix(page):
    rows = []
    for topic in LEARNING_PATHS['topics']:
        basic = ' · '.join(f'<a href="{href(page, "foundations/"+slug+".html")}">{escape(FOUNDATION_BY_SLUG[slug]["title"].split("：")[0])}</a>' for slug in topic['foundations'])
        comparison = ' · '.join(f'<a href="{href(page, ENGINE_COMPARISON, key)}">{escape(next(t[1] for t in ENGINE_TOPICS if t[0]==key))}</a>' for key in topic['comparison']) or '本章无独立专题，可直接进入实现'
        implementation = ' · '.join(f'<a href="{href(page, path)}">{escape(LESSON_TITLES[path].split("：")[0])}</a>' for path in topic['implementation'])
        rows.append(f'<tr><th scope="row">{escape(topic["title"])}</th><td>{basic}</td><td>{comparison}</td><td>{implementation}</td><td>{escape(topic["delta"])}</td></tr>')
    return '<details class="cross-chapter-index" id="topic-paths"><summary>按同一个问题查三章：已经会的部分可以跳过</summary><p class="muted">窄屏可在表格内横向滚动，逐列查看同一问题的入口。</p><div class="table-scroll" tabindex="0" aria-label="三章问题索引，可横向滚动"><table><caption>原理、引擎对照与固定源码各自回答什么</caption><thead><tr><th>学习问题</th><th>第一章 · 原理</th><th>第二章 · 引擎对照</th><th>第三章 · 实现</th><th>怎样接着读</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></details>'


def beginner_entry(page):
    main = READING_ROUTES[0]
    other = ''.join(f'<section><h3>{escape(route[1])}</h3><a href="{route_href(page, route[3][0][0], route[0])}">从这条路线开始 →</a>{route_outline(page, route)}</section>' for route in READING_ROUTES[1:])
    return ('<div class="beginner-path"><p class="eyebrow">第一次来 · 单实例主线</p><h2>先沿一条请求走通三章</h2>'
        '<p>第一章解释计算与状态，第二章比较职责分工，第三章回查对象、字段与依赖。三章里的数字是各自的教学算例，衔接的是机制和问题。</p>'
        f'<a class="button-primary" data-beginner-start href="{route_href(page, main[3][0][0], main[0])}">从文字与 token 开始 <span aria-hidden="true">→</span></a>'
        '<ol class="route-checkpoints"><li><strong>基础 · 8 课</strong><span>输入、前向、生成、缓存、批次、设备与服务</span></li><li><strong>对照 · 3 主题</strong><span>请求路径、前缀缓存与调度</span></li><li><strong>实现 · 6 课</strong><span>单实例地图、对象、本轮输入与执行</span></li></ol>'
        f'<details id="route-beginner"><summary>查看 17 步及每步阅读范围</summary>{route_outline(page, main)}</details>'
        f'<details class="route-alternatives"><summary>已有基础：选择实现路线或按现象定位</summary>{other}</details></div>' + topic_matrix(page))


def foundation_connection(page, course):
    key = LEARNING_PATHS['foundation_topic'][course['slug']]
    labels = {'tokens':'输入 ID 在哪个进程准备？', 'components':'在引擎中定位模型执行职责', 'attention':'在引擎中定位模型执行职责', 'generation':'输入、执行和返回由谁接续？', 'cache':'两套索引怎样找到相同前缀的地址？', 'batching':'两套引擎每轮怎样选批？', 'hardware':'图重放在执行器中怎样判断资格？', 'serving':'引擎收到取消后怎样收尾？'}
    step = {'hardware':3, 'serving':2}.get(course['slug'])
    url = href(page, ENGINE_COMPARISON, key) if step is None else href(page, ENGINE_COMPARISON) + f'?step={step}#{key}'
    compare = (f'<a href="{url}">第二章：{escape(labels.get(course["slug"], "比较两套引擎的职责与交接"))} →</a>' if key else '<p>第二章没有独立的候选验证专题，可直接进入实现。</p>')
    return ('<aside class="chapter-handoff"><strong>把这个问题接到引擎上</strong>' + compare +
        f'<a class="foundation-to-engine" href="{href(page, course["implementation"])}"><small>已有基础可直接查阅 · 独立源码基线</small>{escape(LESSON_TITLES[course["implementation"]])} →</a></aside>')


def engine_connections(page):
    data = {}
    english = {'generation':'Autoregressive generation', 'serving':'Serving and cancellation', 'cache':'KV cache', 'batching':'Batching', 'hardware':'GPU and CUDA Graph', 'parallel':'Parallelism', 'communication':'Communication', 'deployment':'P/D disaggregation', 'metrics':'Metric definitions'}
    for key, entry in LEARNING_PATHS['comparison'].items():
        title, title_en = next((t[1], t[2]) for t in ENGINE_TOPICS if t[0] == key)
        data[key] = {**entry, 'title':title, 'english':title_en, 'foundations': [{'href':href(page, 'foundations/'+slug+'.html'), 'label':FOUNDATION_BY_SLUG[slug]['title'], 'english':english[slug]} for slug in entry['foundations']]}
    return '<!-- chapter-connections:start --><script type="application/json" id="chapter-connections">' + json.dumps(data, ensure_ascii=False).replace('<','\\u003c') + '</script><!-- chapter-connections:end -->'


def learning_start(page):
    parts = ['<!-- learning-start:start -->', '<section id="reading-routes" aria-label="三章学习入口">',
        beginner_entry(page), '<div class="chapter-choice"><a href="#chapter-foundations"><span>第一章</span><strong>推理基础</strong><p>从一句话怎样变成数字开始，逐步认识生成、缓存、调度与多卡协作。</p><span>先建立直觉，再进入实现</span></a>',
        '<a href="#chapter-engine"><span>第二章</span><strong>SGLang 与 vLLM 架构对照</strong><p>按固定 release 对照两套引擎的单实例请求路径、P/D 交接和延迟口径。</p><span>七个主题 · 保留各自版本边界</span></a>',
        '<a href="#chapter-sglang"><span>第三章</span><strong>SGLang 实现剖析</strong><p>沿请求追踪对象、队列、资源与源码，保留各课固定版本和验证边界。</p><span>十三模块 · 二十三节实现课程</span></a></div></section>',
        '<section id="chapter-foundations" class="foundation-directory" aria-labelledby="foundation-directory-title"><h2 id="foundation-directory-title">第一章 · 推理基础</h2>',
        '<p>先把文字变成向量，再沿一次 Transformer 前向追到下一 token。接着拆开 Attention，看看生成怎样一轮轮接续，最后理解历史状态为什么值得缓存。</p><ol class="foundation-lessons">']
    for i, c in enumerate(FOUNDATIONS, 1):
        parts.append(f'<li><a href="{href(page, "foundations/"+c["slug"]+".html")}"><span class="lesson-no">{i:02}</span><span><strong>{escape(c["title"])}</strong><small>{escape(c["goal"])}</small></span></a></li>')
    parts.append('</ol></section><!-- learning-start:end -->')
    return '\n'.join(parts)


def foundation_bridge(page, module):
    relative = page.relative_to(PAGES).as_posix()
    topic = next(t for t in LEARNING_PATHS['topics'] if relative in t['implementation'])
    overrides = {
        'sglang/system-overview/index.html': ['generation', 'serving'],
        'sglang/request-runtime/index.html': ['generation', 'serving'],
        'sglang/request-state/index.html': ['batching', 'cache'],
        'sglang/inference-overview/journey.html': ['generation', 'cache'],
        'sglang/loop-journey/index.html': ['batching', 'hardware'],
        'sglang/model-execution/index.html': ['hardware', 'components'],
        'sglang/inference-overview/transformer.html': ['components', 'parallel'],
        'sglang/inference-overview/scheduling.html': ['batching', 'cache'],
        'sglang/scheduling/index.html': ['batching', 'cache'],
        'sglang/serving-operations/index.html': ['serving', 'deployment'],
    }
    slugs = overrides.get(relative, topic['foundations'])
    links = ''.join(f'<a href="{href(page, "foundations/"+slug+".html")}">原理：{escape(FOUNDATION_BY_SLUG[slug]["title"].split("：")[0])}</a>' for slug in slugs)
    for key in topic['comparison']:
        label = next(t[1] for t in ENGINE_TOPICS if t[0] == key)
        links += f'<a href="{href(page, ENGINE_COMPARISON, key)}">引擎对照：{label}（独立版本）</a>'
    note = ''
    if relative == 'sglang/system-overview/index.html':
        note = '<p>第二章以 SGLang 297e54f 对照 vLLM；本章沿 279339f113 回查实现。两章连接同一类职责，类名、分支和配置按本课源码核对。</p>'
    elif relative == 'sglang/loop-journey/index.html':
        note = f'<p>初读先走 Normal。进入 PP 分支前，先看 <a href="{href(page, "foundations/parallel.html")}">PP 阶段与微批次</a>；切换机制后按各自的教学条件读图。</p>'
    elif relative.startswith('sglang/pd-prefill-pp-loop/'):
        note = f'<p>综合案例先修：<a href="{href(page, "sglang/pd-prefill-lifecycle/index.html")}">P 侧生命周期与资源释放</a>。先认清交接条件，再追槽位与循环依赖。</p>'
    return '<!-- foundation-bridge:start --><section class="implementation-bridge" aria-label="先修与跨章回看"><strong>先修与跨章回看</strong><nav>' + links + '</nav>' + note + '</section><!-- foundation-bridge:end -->'


def reading_pagers(page):
    parts = ['<!-- reading-pagers:start -->']
    relative = page.relative_to(PAGES).as_posix()
    for rid, title, _, lessons in READING_ROUTES:
        for i, (target, _, _) in enumerate(lessons):
            parsed = urlsplit(target)
            if parsed.path != relative:
                continue
            previous = (f'<a href="{route_href(page, lessons[i-1][0], rid)}"><small>这条路线 · 上一节</small><strong>← {escape(lessons[i-1][1])}</strong></a>' if i else
                        f'<a href="{href(page, anchor="reading-routes")}"><small>选择学习起点</small><strong>返回路线</strong></a>')
            following = (f'<a class="next" href="{route_href(page, lessons[i+1][0], rid)}"><small>这条路线 · 下一节</small><strong>{escape(lessons[i+1][1])} →</strong></a>' if i+1 < len(lessons) else
                         f'<a class="next" href="{href(page, anchor="reading-routes")}"><small>这条路线已完成</small><strong>按问题选择下一步 →</strong></a>')
            hint = lessons[i][2]
            parts.append(f'<nav class="reading-route-pager" data-reading-route="{rid}" data-route-topic="{parsed.fragment}" data-route-title="{escape(title)}" data-route-hint="{escape(hint, quote=True)}" data-route-position="{i+1} / {len(lessons)}" data-route-home="{href(page, anchor="reading-routes")}" aria-label="{escape(title)} · 继续学习" hidden>{previous}{following}</nav>')
    parts.append('<!-- reading-pagers:end -->')
    return '\n'.join(parts)


def href(page, target='index.html', anchor=''):
    if target == 'index.html':
        value = './' if page.parent == PAGES else os.path.relpath(PAGES, page.parent) + '/'
    else:
        value = os.path.relpath(PAGES / target, page.parent)
    return escape(value + ('#' + anchor if anchor else ''), quote=True)


def sidebar(page, active):
    home = page == PAGES / 'index.html'
    engine = page == ENGINE_PAGE
    lines = ['<aside class="site-sidebar" aria-label="学习导航">',
             '<details class="sidebar-menu" open><summary>学习导航 <span aria-hidden="true">⌄</span></summary>',
             '<div class="sidebar-content">',
             f'<a class="overview-link" href="{href(page)}"' + (' aria-current="page"' if home else '') + '>学习首页 <span aria-hidden="true">↗</span></a>']
    lines.append(f'<a class="route-index-link" href="{href(page, anchor="reading-routes")}">推荐路线与跨章索引</a>')
    in_foundations = page.parent == PAGES / 'foundations'
    lines.append('<details class="chapter-nav" data-chapter="foundations"' + (' open' if in_foundations else '') + '><summary>第一章 · 推理基础</summary><nav class="foundation-nav" aria-label="推理基础课程">')
    for i, c in enumerate(FOUNDATIONS, 1):
        target = 'foundations/' + c['slug'] + '.html'
        selected = page == PAGES / target
        lines.append(f'<a href="{href(page, target)}"' + (' aria-current="page"' if selected else '') + f'>{i:02} {escape(c["title"])}</a>')
    lines.append('</nav></details>')
    lines.append('<details class="chapter-nav" data-chapter="engine-comparison"' + (' open' if engine else '') + '><summary>第二章 · SGLang 与 vLLM 架构对照</summary><div class="module-lessons-nav engine-topic-nav">')
    for number, (key, label, english, _) in enumerate(ENGINE_TOPICS, 1):
        lines.append(f'<a class="lesson-link" data-engine-topic="{key}" href="{href(page, ENGINE_COMPARISON, key)}">'
                     f'<span class="engine-topic-number" aria-hidden="true">{number:02}</span>'
                     f'<span class="engine-topic-zh" lang="zh-CN">{escape(label)}</span>'
                     f'<span class="engine-topic-en" lang="en">{escape(english)}</span></a>')
    lines.append('</div></details>')
    lines.append('<details class="chapter-nav" data-chapter="sglang"' + (' open' if active else '') + '><summary>第三章 · SGLang 实现剖析</summary>')
    for stage, label, title in STAGES:
        opened = active and active[2] == stage
        lines.append(f'<details class="topic-group" data-topic="framework-{stage}"' + (' open' if opened else '') + '>')
        lines.append(f'<summary><span class="topic-heading"><span class="nav-label">{label}</span><strong class="sidebar-title">{title}</strong></span><span class="topic-chevron" aria-hidden="true">›</span></summary>')
        lines.append(f'<nav aria-label="{label}">')
        for n, module in enumerate(MODULES, 1):
            mid, name, group, planned, lessons = module
            if group != stage:
                continue
            heading = f'<span class="course-number">{n:02}</span><span><strong>{name}</strong>'
            if planned:
                heading += '<small>预留模块</small>'
            heading += '</span>'
            if lessons:
                current = active == module
                lines.append(f'<details class="module-nav" data-module="{mid}"' + (' open' if current else '') + '>')
                lines.append('<summary class="module-heading"' + (' aria-current="location"' if current else '') + f'>{heading}<span class="module-chevron" aria-hidden="true">›</span></summary>')
                lines.append(f'<div class="module-lessons-nav"><a class="module-overview-link" href="{href(page, anchor=mid)}">模块目标与学习路线 ↗</a>')
                for path, lesson_title, kind in lessons:
                    selected = PAGES / path == page
                    lines.append(f'<a class="lesson-link" href="{href(page, path)}"' + (' aria-current="page"' if selected else '') + f'>{escape(lesson_title)}' + (f'<small class="lesson-kind">{escape(kind.replace("SGLang · ", ""))}</small>' if len(lessons) > 1 else '') + '</a>')
                lines.append('</div></details>')
            else:
                lines.append(f'<a class="course-link" data-module="{mid}" href="{href(page, anchor=mid)}">{heading}</a>')
        lines.append('</nav></details>')
    lines.append('</details>')
    lines.extend(['<div class="sidebar-note"><span class="status-dot"></span> 从原理走向实现<p>第一章建立概念，第二章对照固定 release，第三章沿固定版本源码核对实现。</p></div>',
                  '<a class="sidebar-repo" href="https://github.com/Asher-XunZhang/ai-infra-wiki">浏览完整资料库 ↗</a>',
                  '</div></details></aside>'])
    return '\n'.join(lines)


def breadcrumb(page, module, lesson):
    stage = next(s for s in STAGES if s[0] == module[2])
    n = MODULES.index(module) + 1
    return ('<!-- learning-context:start -->\n<nav class="learning-breadcrumb" aria-label="当前位置">'
            f'<a href="{href(page, anchor="learning-path")}">第三章 · SGLang 实现剖析</a><span aria-hidden="true">/</span>'
            f'<a href="{href(page, anchor="stage-" + stage[0])}">{stage[1]}</a><span aria-hidden="true">/</span>'
            f'<a href="{href(page, anchor=module[0])}">{n:02} {module[1]}</a>'
            f'<span aria-hidden="true">/</span><span aria-current="page">{escape(lesson[1])}</span>'
            '</nav>\n<!-- learning-context:end -->\n')


def pager(page, module, lesson):
    n = MODULES.index(module)
    lessons = module[4]
    i = lessons.index(lesson)
    if i:
        previous = f'<a href="{href(page, lessons[i-1][0])}"><small>本模块 · 上一节</small><strong>← {escape(lessons[i-1][1])}</strong></a>'
    else:
        previous = f'<a href="{href(page, anchor=module[0])}"><small>回到所属模块</small><strong>← {n+1:02} {module[1]}</strong></a>'
    if i + 1 < len(lessons):
        following = f'<a class="next" href="{href(page, lessons[i+1][0])}"><small>本模块 · 下一节</small><strong>{escape(lessons[i+1][1])} →</strong></a>'
    elif n + 1 < len(MODULES):
        target = MODULES[n+1]
        url = href(page, target[4][0][0]) if target[4] else href(page, anchor=target[0])
        note = '预留模块 · 查看学习目标' if target[3] else '查看模块路线与已有内容'
        following = f'<a class="next" href="{url}"><small>下一模块 · {n+2:02}</small><strong>{target[1]} →</strong><span>{note if not target[4] else "从本模块入门课程开始"}</span></a>'
    else:
        following = f'<a class="next" href="{href(page, anchor="reading-routes")}"><small>选择下一条路线</small><strong>回到整体学习框架 →</strong></a>'
    return '<nav class="lesson-pager" aria-label="框架内继续学习">' + previous + following + '</nav>'


def implementation_design(page):
    entry = DESIGN_NOTES[str(page.relative_to(PAGES))]
    revision = entry.get('revision', DESIGN_REVISION)
    body = []
    background = entry['background']
    references = ' · '.join(f'<a href="{escape(url, quote=True)}">{escape(title)} ↗</a>' for title, url in background['references'])
    context = (f'<details class="design-background"><summary>{escape(background["title"])}</summary>'
        f'<p>{escape(background["text"])}</p><p class="design-source">概念与设计说明：{references}</p>'
        f'<p class="design-source">{escape(background["scope"])} 资料读取：{escape(background["read_date"])}。</p></details>')
    navigation = ('<nav class="design-nav" aria-label="实现问题导航">' + ''.join(f'<a href="#design-{t["id"]}">{escape(t["title"])}</a>' for t in entry['topics']) + '</nav>') if len(entry['topics']) > 1 else ''
    for t in entry['topics']:
        file, line, label = t['source']
        file = file if file.startswith(('python/', 'sgl-model-gateway/')) else 'python/sglang/srt/' + file
        source_url = f'https://github.com/sgl-project/sglang/blob/{revision}/{file}#L{line}'
        example = f'<div class="design-example"><strong>用一个例子理解</strong><p>{escape(t["example"])}</p></div>' if t.get('example') else ''
        formula = t.get('formula', '')
        links = ''.join(f'<li><a href="{href(page, target)}">{escape(title)} →</a></li>' for title, target in t.get('links', []))
        body.append(f'<section class="design-question" id="design-{t["id"]}"><h3>{escape(t["title"])}</h3><p>{escape(t["why"])}</p><details><summary>展开机制与具体例子</summary><p>{escape(t["mechanism"])}</p>{example}{formula}</details><p class="design-tradeoff"><strong>取舍与条件：</strong>{escape(t["tradeoff"])}</p><p class="design-source"><a href="{source_url}">{escape(label)} · 本课实现核对 ↗</a></p>' + (f'<ul class="design-links">{links}</ul>' if links else '') + '</section>')
    return ('<!-- implementation-design:start --><section class="implementation-design" id="implementation-why"><h2 id="implementation-why-title">从问题理解实现</h2>'
        f'<p class="design-problem">{escape(entry["problem"])}</p><a class="design-skip" href="#implementation-process">进入本课图解与源码过程 ↓</a>'
        + '<details class="design-reading"><summary>展开机制、算例与取舍</summary>' + context + navigation + ''.join(body) + f'<p class="design-baseline">本课实现核对：SGLang 固定提交 {revision[:10]}，{entry.get("reviewed", "2026-10-16")}。外部资料解释概念背景；对象职责与条件沿固定源码理解。教学算例不代表实测性能。</p></details></section><div id="implementation-process"></div><!-- implementation-design:end -->')


def lesson_reading_nav(source):
    """Keep existing specialized navigators; index substantial prose only."""
    source = re.sub(r'<!-- lesson-reading:start -->.*?<!-- lesson-reading:end -->', '', source, flags=re.S)
    if 'class="life-nav"' in source or 'class="article-toc"' in source:
        return source
    links = []
    seen = set()
    def heading(match):
        attrs, body = match.groups()
        title = re.sub(r'<[^>]+>', '', body).strip()
        if not title or title == '本课追踪什么实现':
            return match[0]
        existing = re.search(r'\bid="([^"]+)"', attrs)
        anchor = existing[1] if existing else 'reading-topic-' + str(len(links)+1)
        if anchor in seen:
            base = anchor
            suffix = 2
            while anchor in seen:
                anchor = f'{base}-{suffix}'
                suffix += 1
            attrs = re.sub(r'\bid="[^"]+"', f'id="{anchor}"', attrs)
        seen.add(anchor)
        links.append((anchor, title))
        return f'<h2{attrs if existing else attrs + chr(32) + "id=" + chr(34) + anchor + chr(34)}>{body}</h2>'
    updated = re.sub(r'<h2([^>]*)>(.*?)</h2>', heading, source, flags=re.S)
    if len(links) < 4:
        return source
    nav = '<!-- lesson-reading:start --><nav class="lesson-reading-nav" aria-label="本课阅读入口"><span>按问题阅读</span>' + ''.join(f'<a href="#{anchor}">{title}</a>' for anchor, title in links) + '</nav><!-- lesson-reading:end -->'
    if 'data-reading-outline="compact"' in source:
        nav = nav.replace('<!-- lesson-reading:start -->', '<!-- lesson-reading:start --><details class="lesson-reading-outline"><summary>本页内容与深入入口</summary>').replace('<!-- lesson-reading:end -->', '</details><!-- lesson-reading:end -->')
    return re.sub(r'(<header class="lesson-header".*?</header>)', lambda m: m[0] + nav, updated, count=1, flags=re.S)


def engine_breadcrumb(page):
    return ('<!-- learning-context:start -->\n<nav class="learning-breadcrumb" aria-label="当前位置">'
            f'<a href="{href(page)}">学习首页</a><span aria-hidden="true">/</span>'
            f'<a href="{href(page, anchor="chapter-engine")}">第二章 · SGLang 与 vLLM 架构对照</a><span aria-hidden="true">/</span>'
            '<span aria-current="page">架构与观测对照</span></nav>\n<!-- learning-context:end -->')


def engine_pager(page):
    return ('<nav class="lesson-pager" aria-label="跨章继续学习">'
            f'<a href="{href(page, anchor="chapter-foundations")}"><small>上一章 · 第一章</small><strong>← 推理基础</strong></a>'
            f'<a class="next" href="{href(page, "sglang/system-overview/index.html")}"><small>下一章 · 第三章</small><strong>进入 SGLang 实现剖析 →</strong><span>从推理系统全景开始</span></a>'
            '</nav>')


def rendered(page, module=None, lesson=None):
    source = page.read_text(encoding='utf-8')
    engine = page == ENGINE_PAGE
    if 'motion/integration.js' not in source and not engine:
        source = source.replace('<script ', f'<script src="{href(page, "motion/integration.js")}" defer></script><script ', 1)
    source, count = re.subn(r'<aside class="site-sidebar".*?</aside>', lambda _: sidebar(page, module), source, count=1, flags=re.S)
    if count != 1:
        raise ValueError(f'{page}: expected exactly one site sidebar')
    if page.parent == PAGES / 'foundations':
        return source
    if engine:
        source = re.sub(r'<!-- learning-context:start -->.*?<!-- learning-context:end -->', lambda _: engine_breadcrumb(page), source, count=1, flags=re.S)
        source, count = re.subn(r'<nav class="lesson-pager".*?</nav>', lambda _: engine_pager(page), source, count=1, flags=re.S)
        if count != 1:
            raise ValueError(f'{page}: missing cross-chapter pager')
        connections = engine_connections(page)
        source = re.sub(r'<!-- chapter-connections:start -->.*?<!-- chapter-connections:end -->', '', source, flags=re.S)
        source = source.replace('<div id="atlas-root">', connections + '<div id="atlas-root">', 1)
        source = re.sub(r'<!-- reading-pagers:start -->.*?<!-- reading-pagers:end -->\n?', '', source, flags=re.S)
        source = source.replace('<nav class="lesson-pager"', reading_pagers(page) + '\n<nav class="lesson-pager"', 1)
        return source
    if not module:
        topics = ['<ol class="foundation-lessons engine-topic-links">']
        for number, (key, label, _, description) in enumerate(ENGINE_TOPICS, 1):
            topics.append(f'  <li><a href="engine-comparison/#{key}"><span class="lesson-no">{number:02}</span>'
                          f'<span><strong>{escape(label)}</strong><small>{escape(description)}</small></span></a></li>')
        topics.append('</ol>')
        source, count = re.subn(r'<ol class="foundation-lessons engine-topic-links">.*?</ol>',
                                lambda _: '\n        '.join(topics), source, count=1, flags=re.S)
        if count != 1:
            raise ValueError(f'{page}: missing engine topic list')
        source, count = re.subn(r'<!-- learning-start:start -->.*?<!-- learning-start:end -->', lambda _: learning_start(page), source, count=1, flags=re.S)
        if count != 1:
            raise ValueError(f'{page}: missing learning-start marker')
    if module:
        source = re.sub(r'<h1[^>]*>.*?</h1>', lambda _: '<h1>' + escape(lesson[1]) + '</h1>', source, count=1, flags=re.S)
        source = re.sub(r'<title>.*?</title>', lambda _: '<title>' + escape(lesson[1]) + ' · AI Infra Wiki</title>', source, count=1, flags=re.S)
        context = breadcrumb(page, module, lesson)
        if '<!-- learning-context:start -->' in source:
            source = re.sub(r'<!-- learning-context:start -->.*?<!-- learning-context:end -->\n?', lambda _: context, source, count=1, flags=re.S)
        else:
            source, count = re.subn(r'<header class="lesson-header"', lambda m: context + m[0], source, count=1)
            if count != 1:
                raise ValueError(f'{page}: missing lesson header')
        kicker = (f'<div class="lesson-kicker"><a class="eyebrow" href="{href(page, anchor=module[0])}">'
                  f'模块 {MODULES.index(module)+1:02} / {module[1]}</a><span class="lesson-count">{lesson[2]}</span></div>')
        source, count = re.subn(r'<div class="lesson-kicker">.*?</div>', lambda _: kicker, source, count=1, flags=re.S)
        if count != 1:
            raise ValueError(f'{page}: missing lesson kicker')
        source, count = re.subn(r'<nav class="lesson-pager".*?</nav>', lambda _: pager(page, module, lesson), source, count=1, flags=re.S)
        if count != 1:
            raise ValueError(f'{page}: missing lesson pager')
        bridge = foundation_bridge(page, module)
        if '<!-- foundation-bridge:start -->' in source:
            source = re.sub(r'<!-- foundation-bridge:start -->.*?<!-- foundation-bridge:end -->', lambda _: bridge, source, count=1, flags=re.S)
        else:
            source = re.sub(r'(<header class="lesson-header".*?</header>)', lambda m: m[0] + bridge, source, count=1, flags=re.S)
        design = implementation_design(page)
        source = re.sub(r'<!-- implementation-design:start -->.*?<!-- implementation-design:end -->', '', source, flags=re.S)
        source = source.replace('<!-- foundation-bridge:end -->', '<!-- foundation-bridge:end -->' + design, 1)
        for asset in ['foundations/model.js', 'foundations/player.js']:
            source = source.replace(f'<script src="{href(page, asset)}" defer></script>', '')
        for asset, tag in [('foundations/foundations.css', 'style')]:
            url = href(page, asset)
            if url not in source:
                element = f'<link rel="stylesheet" href="{url}">' if tag == 'style' else f'<script src="{url}" defer></script>'
                source = source.replace('</head>', element + '</head>', 1)
        if page.relative_to(PAGES).as_posix() == 'sglang/system-overview/index.html':
            for asset, tag in [('sglang/system-overview/opening.css', 'style'), ('sglang/system-overview/opening.js', 'script')]:
                url = href(page, asset)
                if url not in source:
                    element = f'<link rel="stylesheet" href="{url}">' if tag == 'style' else f'<script src="{url}" defer></script>'
                    source = source.replace('</head>', element + '</head>', 1)
            source = re.sub(r'<!-- system-architecture:start -->.*?<!-- system-architecture:end -->', '', source, flags=re.S)
            source = source.replace('<!-- foundation-bridge:start -->', render_architecture() + '<!-- foundation-bridge:start -->', 1)
        source = lesson_reading_nav(source)
        extra = reading_pagers(page)
        if '<!-- reading-pagers:start -->' in source:
            source = re.sub(r'<!-- reading-pagers:start -->.*?<!-- reading-pagers:end -->', lambda _: extra, source, count=1, flags=re.S)
        else:
            source = source.replace('<nav class="lesson-pager"', extra + '\n<nav class="lesson-pager"', 1)
    return source


def build(check=False):
    # The problem index must cover the curriculum without silently orphaning a lesson.
    topics = LEARNING_PATHS['topics']
    assert {s for t in topics for s in t['foundations']} == set(FOUNDATION_BY_SLUG)
    assert {s for t in topics for s in t['comparison']} == {t[0] for t in ENGINE_TOPICS}
    implementation_paths = [p for t in topics for p in t['implementation']]
    assert len(implementation_paths) == len(set(implementation_paths))
    assert set(implementation_paths) == {lesson[0] for module in MODULES for lesson in module[4]}
    completed = set()
    for step in LEARNING_PATHS['beginner']['steps']:
        assert step['path'] in LESSON_TITLES
        if step['path'].startswith('foundations/'):
            slug = Path(step['path']).stem
            assert set(FOUNDATION_BY_SLUG[slug]['prerequisites']) <= completed, f'unmet prerequisite: {slug}'
            completed.add(slug)
    targets = [(PAGES / 'index.html', None, None), (ENGINE_PAGE, None, None)]
    targets += [(PAGES / ('foundations/' + c['slug'] + '.html'), None, None) for c in FOUNDATIONS]
    targets += [(PAGES / lesson[0], module, lesson) for module in MODULES for lesson in module[4]]
    assert len({p for p, _, _ in targets}) == len(targets), 'each course must have exactly one primary module'
    existing = {p for p in PAGES.rglob('*.html') if '<aside class="site-sidebar"' in p.read_text(encoding='utf-8')}
    assert existing == {p for p, _, _ in targets}, 'every content page must belong to the shared framework'
    stale = []
    for page, module, lesson in targets:
        result = rendered(page, module, lesson)
        if result != page.read_text(encoding='utf-8'):
            if check:
                stale.append(str(page.relative_to(ROOT)))
            else:
                page.write_text(result, encoding='utf-8')
    if stale:
        raise SystemExit('Navigation is stale; run scripts/build_learning_navigation.py: ' + ', '.join(stale))
    print(f'{"Checked" if check else "Generated"} shared framework navigation: {len(targets)} pages, 3 chapters, {len(FOUNDATIONS)} foundations, {len(ENGINE_TOPICS)} comparison topics, {len(MODULES)} SGLang modules, {sum(len(m[4]) for m in MODULES)} implementation courses.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    build(parser.parse_args().check)
