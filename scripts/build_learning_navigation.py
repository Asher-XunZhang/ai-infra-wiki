#!/usr/bin/env python3
"""Render the shared learning framework into source HTML; --check detects drift."""
import argparse
import json
import os
import re
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / 'pages'
FOUNDATIONS = json.loads((PAGES / 'foundations/courses.json').read_text(encoding='utf-8'))
FOUNDATION_BY_SLUG = {c['slug']: c for c in FOUNDATIONS}
DESIGN_NOTES = json.loads((PAGES / 'sglang/design-notes.json').read_text(encoding='utf-8'))
from system_architecture import render_architecture

DESIGN_REVISION = '279339f113b79af84f27fd3ac92d0a13bd3f4cbd'
BRIDGES = {'system-overview':'generation','request-runtime':'serving','model-execution':'hardware','kv-memory':'cache','scheduling':'batching','parallelism':'parallel','communication':'communication','disaggregation':'deployment','advanced-generation':'speculation','serving-operations':'serving','performance':'metrics','case-studies':'parallel'}

IMPLEMENTATION_FOCUS = {'system-overview': '分清进程、组件职责与部署边界，再把问题定位到对应实现。',
 'request-runtime': '跟踪同一请求在 TokenizerManager、Scheduler 和 DetokenizerManager 之间的交接，以及完成与取消的责任方。',
 'model-execution': '跟踪模型调用、Worker / Runner 交接、图重放资格与真实计算行。',
 'kv-memory': '核对请求索引如何映射到物理 KV 页，以及共享保护、缓存保留和驱逐条件。',
 'scheduling': '区分候选排序、资源准入、分块预算和回撤后的状态恢复。',
 'parallelism': '把切分方式对应到 rank、进程组和模型实现，检查各模式的输入输出及组合约束。',
 'communication': '按实际接口区分控制消息、张量通信与 KV 搬运，追踪可用和释放条件。',
 'disaggregation': '追踪 bootstrap、接收匹配、回载等待与请求准入；分别核对 P 和 D 的资源生命周期。',
 'advanced-generation': '追踪具体生成路径的状态提交、回退与约束筛选，并对照各自源码入口。',
 'serving-operations': '核对启动、探活、路由、队列准入、取消和退出时各组件的责任。',
 'performance': '把观测区间映射到实际执行阶段，核对流式响应、压测统计和失败样本口径。',
 'case-studies': '沿固定版本的 PP loop，核对 batch、slot、通信共识与本地状态复查的依赖。'}

STAGES = [
    ('foundations', '第一层 · 建立全局', '从请求认识系统'),
    ('instance', '第二层 · 理解单实例', '执行、缓存与调度'),
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
    ('model-execution', '模型执行、硬件与算子', 'instance', False, [
        ('sglang/inference-overview/transformer.html', 'DeepSeek-V3：Embedding、Decoder Layer 与输出头', 'SGLang · 源码导读'),
        ('sglang/model-execution/index.html', 'ModelRunner 与 CUDA Graph：输入怎样交给 GPU？', 'SGLang · 执行机制')]),
    ('kv-memory', 'KV Cache 与内存管理', 'instance', False, [
        ('sglang/inference-overview/kv-cache.html', 'RadixCache 与 Attention：前缀索引怎样连接 KV？', 'SGLang · 源码导读'),
        ('sglang/kv-memory/index.html', 'KV Memory Pool：地址映射、分页与缓存回收', 'SGLang · 缓存机制')]),
    ('scheduling', '调度与批处理', 'instance', False, [
        ('sglang/inference-overview/journey.html', 'ScheduleBatch：Extend 与 Decode 怎样准备输入？', 'SGLang · 源码导读'),
        ('sglang/inference-overview/scheduling.html', 'Scheduler：选批、Chunked Prefill 与 Decode 更新', 'SGLang · 源码导读'),
        ('sglang/scheduling/index.html', 'Batch 调度：准入预算、Chunked Prefill 与 Retract', 'SGLang · 调度机制')]),
    ('parallelism', '并行与执行拓扑', 'distributed', False, [
        ('sglang/parallelism/index.html', 'TP / PP / DP / EP / CP：SGLang 并行分工', 'SGLang · 并行分工')]),
    ('communication', '通信与传输', 'distributed', False, [('sglang/communication/index.html', 'IPC、Collective 与 KV Transfer：数据何时可用？', 'SGLang · 通信机制')]),
    ('disaggregation', '分离部署与分布式状态交接', 'distributed', False, [
        ('sglang/inference-overview/deployment.html', 'PD Disaggregation：Bootstrap、KV 接收与准入', 'SGLang · 源码导读'),
        ('sglang/pd-prefill-lifecycle/index.html', 'PD + PP：Prefill 请求生命周期与 KV 释放', 'SGLang · PD＋PP 案例'),
        ('sglang/pd-dataflow/index.html', 'PD 数据流：激活、KV Transfer 与 Req 队列引用', 'SGLang · PD＋PP 案例')]),
    ('advanced-generation', '模型结构与高级生成', 'serving', False, [('sglang/advanced-generation/index.html', 'Speculative Decoding 与约束生成：状态怎样提交？', 'SGLang · 高级生成')]),
    ('serving-operations', '服务部署与运行治理', 'serving', False, [('sglang/serving-operations/index.html', '服务治理：Health Check、Timeout 与 Cancellation', 'SGLang · 服务治理')]),
    ('performance', '性能分析与优化方法', 'practice', False, [('sglang/performance-engineering/index.html', 'TTFT、ITL、TPOT 与 P95：SGLang 性能观测', 'SGLang · 性能分析')]),
    ('case-studies', '综合案例与源码实践', 'practice', False, [
        ('sglang/pd-prefill-pp-loop/quick.html', 'PP Event Loop：追踪 Microbatch 的完整旅程', 'SGLang · 综合实践'),
        ('sglang/pd-prefill-pp-loop/index.html', 'PP 依赖时间轴：计算、通信与流水线气泡', 'SGLang · PD＋PP 案例'),
        ('sglang/pd-prefill-pp-loop/notes.html', 'PP Event Loop 源码：Batch、Slot 与通信共识', 'SGLang · PD＋PP 案例')]),
]

# Reading order is independent of the stable module taxonomy. Links are rendered
# into HTML so the Pages build can version them and no-JS readers can use them.
READING_ROUTES = [
    ('first', '从基础走向源码', '在已有原理之上，定位各机制的实现入口。', [
        ('sglang/system-overview/index.html', '先建立代码框架', '从完整架构找到各模块及其协作。'),
        ('sglang/inference-overview/journey.html', 'SGLang 怎样准备两种生成批次？', '对应固定版本的实现入口。'),
        ('sglang/inference-overview/transformer.html', 'SGLang 怎样调用 DeepSeek-V3 模型层？', '对应固定版本的实现入口。'),
        ('sglang/inference-overview/kv-cache.html', 'SGLang 如何连接前缀索引与 KV 计算？', '对应固定版本的实现入口。'),
        ('sglang/inference-overview/scheduling.html', 'SGLang 怎样选出并更新下一批？', '对应固定版本的实现入口。')]),
    ('implementation', '已有基础，追踪实现', '从请求流转追到资源、调度和设备执行；各课保留自己的固定基线。', [
        ('sglang/request-runtime/index.html', '请求经过哪些对象？', '追踪接入、执行、返回与释放。'),
        ('sglang/kv-memory/index.html', 'KV 放哪，何时回收？', '区分地址映射、共享保护与物理空间。'),
        ('sglang/scheduling/index.html', '谁能进入下一批？', '把候选顺序与准入条件分开。'),
        ('sglang/model-execution/index.html', '工作怎样交给设备？', '连接调度、执行封装与设备计算。')]),
    ('questions', '带着现象找原因', '选择与你当前现象有关的一课，按需回补先修知识。', [
        ('sglang/performance-engineering/index.html', '首字为什么慢？', '先确定计时边界和等待发生在哪。'),
        ('sglang/kv-memory/index.html', '显存为什么不够？', '检查空间占用与仍被引用的状态。'),
        ('sglang/parallelism/index.html', '多卡为什么还在等？', '区分工作切分与前后级依赖。'),
        ('sglang/serving-operations/index.html', '服务在线却没回答？', '分清健康、接纳和实际完成。')]),
]


LESSON_TITLES = {lesson[0]: lesson[1] for module in MODULES for lesson in module[4]}
READING_ROUTES = [(key, title, intro, [(path, LESSON_TITLES.get(path, label), hint) for path, label, hint in lessons])
                  for key, title, intro, lessons in READING_ROUTES]

def route_href(page, target, route):
    return href(page, target) + '?learn=' + route


def learning_start(page):
    parts = ['<!-- learning-start:start -->', '<section id="reading-routes" aria-label="两章学习入口">',
        '<div class="chapter-choice"><a href="#chapter-foundations"><span>第一章</span><strong>推理基础</strong><p>从一句话怎样变成数字开始，逐步认识生成、缓存、调度与多卡协作。</p><span>先建立直觉，再进入实现</span></a>',
        '<a href="#chapter-sglang"><span>第二章</span><strong>SGLang 实现剖析</strong><p>沿请求追踪对象、队列、资源与源码，保留各课固定版本和验证边界。</p><span>十二模块 · 二十节实现课程</span></a></div></section>',
        '<section id="chapter-foundations" class="foundation-directory" aria-labelledby="foundation-directory-title"><h2 id="foundation-directory-title">第一章 · 推理基础</h2>',
        '<p>第一章从文字、生成、Attention 与基础部件，走向缓存、调度、硬件、多卡、服务和性能。每课先看具体例子，再连接相应的 SGLang 实现。</p><ol class="foundation-lessons">']
    for i, c in enumerate(FOUNDATIONS, 1):
        parts.append(f'<li><a href="{href(page, "foundations/"+c["slug"]+".html")}"><span class="lesson-no">{i:02}</span><span><strong>{escape(c["title"])}</strong><small>{escape(c["goal"])}</small></span></a></li>')
    parts.append('</ol></section><!-- learning-start:end -->')
    return '\n'.join(parts)


def foundation_bridge(page, module):
    slug = BRIDGES[module[0]]
    if page.name == 'journey.html':
        slug = 'generation'
    elif page.name == 'transformer.html':
        slug = 'attention'
    c = FOUNDATION_BY_SLUG[slug]
    guide_focus = {
        'journey.html': '跟踪 ScheduleBatch 在 Extend 与 Decode 两条准备路径中怎样组织输入、切换模式和分配写入位置。',
        'transformer.html': '沿 DeepSeek-V3 的 Embedding、层循环和输出头，核对 PP 首级、非首级与末级的责任。',
        'kv-cache.html': '连接请求的前缀索引、本轮输入范围与 attention 调用，再进入物理页和回收实现。',
        'scheduling.html': '找到选批、Prefill 组批和运行批次更新的入口，核对分块与内存回撤发生在哪里。',
        'deployment.html': '分别沿 Prefill 与 Decode 的接口核对首 token、metadata、接收就绪和调度准入。',
    }
    focus = guide_focus.get(page.name, IMPLEMENTATION_FOCUS[module[0]])
    return (f'<!-- foundation-bridge:start --><section class="implementation-bridge" aria-label="本课实现目标">'
        f'<h2>本课追踪什么实现</h2><p>{escape(focus)}</p>'
        f'<a href="{href(page, "foundations/"+slug+".html")}">需要回顾原理：{escape(c["title"])}</a>'
        '</section><!-- foundation-bridge:end -->')


def reading_pagers(page):
    parts = ['<!-- reading-pagers:start -->']
    relative = str(page.relative_to(PAGES))
    for rid, title, _, lessons in READING_ROUTES:
        paths = [item[0] for item in lessons]
        if relative not in paths:
            continue
        i = paths.index(relative)
        previous = (f'<a href="{route_href(page, paths[i-1], rid)}"><small>这条路线 · 上一节</small><strong>← {lessons[i-1][1]}</strong></a>' if i else
                    f'<a href="{href(page, anchor="reading-routes")}"><small>选择学习起点</small><strong>返回路线</strong></a>')
        following = (f'<a class="next" href="{route_href(page, paths[i+1], rid)}"><small>这条路线 · 下一节</small><strong>{lessons[i+1][1]} →</strong></a>' if i+1 < len(paths) else
                     f'<a class="next" href="{href(page, anchor="reading-routes")}"><small>已到这条路线末尾</small><strong>选择下一条学习路线 →</strong></a>')
        parts.append(f'<nav class="reading-route-pager" data-reading-route="{rid}" data-route-title="{title}" data-route-position="{i+1} / {len(paths)}" aria-label="{title} · 继续学习" hidden>{previous}{following}</nav>')
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
    lines = ['<aside class="site-sidebar" aria-label="学习导航">',
             '<details class="sidebar-menu" open><summary>学习导航 <span aria-hidden="true">⌄</span></summary>',
             '<div class="sidebar-content">',
             f'<a class="overview-link" href="{href(page)}"' + (' aria-current="page"' if home else '') + '>学习首页 <span aria-hidden="true">↗</span></a>']
    in_foundations = page.parent == PAGES / 'foundations'
    lines.append('<details class="chapter-nav" data-chapter="foundations"' + (' open' if in_foundations or home else '') + '><summary>第一章 · 推理基础</summary><nav class="foundation-nav" aria-label="推理基础课程">')
    for i, c in enumerate(FOUNDATIONS, 1):
        target = 'foundations/' + c['slug'] + '.html'
        selected = page == PAGES / target
        lines.append(f'<a href="{href(page, target)}"' + (' aria-current="page"' if selected else '') + f'>{i:02} {escape(c["title"])}</a>')
    lines.append('</nav></details>')
    lines.append('<details class="chapter-nav" data-chapter="sglang"' + (' open' if active else '') + '><summary>第二章 · SGLang 实现剖析</summary>')
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
                    lines.append(f'<a class="lesson-link" href="{href(page, path)}"' + (' aria-current="page"' if selected else '') + f'>{escape(lesson_title)}</a>')
                lines.append('</div></details>')
            else:
                lines.append(f'<a class="course-link" data-module="{mid}" href="{href(page, anchor=mid)}">{heading}</a>')
        lines.append('</nav></details>')
    lines.append('</details>')
    lines.extend(['<div class="sidebar-note"><span class="status-dot"></span> 从原理走向实现<p>第一章建立概念，第二章沿固定版本源码核对实现。</p></div>',
                  '<a class="sidebar-repo" href="https://github.com/Asher-XunZhang/ai-infra-wiki">浏览完整资料库 ↗</a>',
                  '</div></details></aside>'])
    return '\n'.join(lines)


def breadcrumb(page, module, lesson):
    stage = next(s for s in STAGES if s[0] == module[2])
    n = MODULES.index(module) + 1
    return ('<!-- learning-context:start -->\n<nav class="learning-breadcrumb" aria-label="当前位置">'
            f'<a href="{href(page, anchor="learning-path")}">第二章 · SGLang 实现剖析</a><span aria-hidden="true">/</span>'
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
        file = file if file.startswith('python/') else 'python/sglang/srt/' + file
        source_url = f'https://github.com/sgl-project/sglang/blob/{revision}/{file}#L{line}'
        example = f'<div class="design-example"><strong>用一个例子理解</strong><p>{escape(t["example"])}</p></div>' if t.get('example') else ''
        links = ''.join(f'<li><a href="{href(page, target)}">{escape(title)} →</a></li>' for title, target in t.get('links', []))
        body.append(f'<section class="design-question" id="design-{t["id"]}"><h3>{escape(t["title"])}</h3><p>{escape(t["why"])}</p><details><summary>展开机制与具体例子</summary><p>{escape(t["mechanism"])}</p>{example}</details><p class="design-tradeoff"><strong>取舍与条件：</strong>{escape(t["tradeoff"])}</p><p class="design-source"><a href="{source_url}">{escape(label)} · 本课实现核对 ↗</a></p>' + (f'<ul class="design-links">{links}</ul>' if links else '') + '</section>')
    return ('<!-- implementation-design:start --><section class="implementation-design" id="implementation-why"><h2 id="implementation-why-title">从问题理解实现</h2>'
        f'<p class="design-problem">{escape(entry["problem"])}</p><a class="design-skip" href="#implementation-process">进入本课图解与源码过程 ↓</a>'
        + context + navigation + ''.join(body) + f'<p class="design-baseline">本课实现核对：SGLang 固定提交 {revision[:10]}，2026-10-16。外部资料解释概念背景；对象职责与条件沿固定源码理解。教学算例不代表实测性能。</p></section><div id="implementation-process"></div><!-- implementation-design:end -->')


def lesson_reading_nav(source):
    """Keep existing specialized navigators; index substantial prose only."""
    source = re.sub(r'<!-- lesson-reading:start -->.*?<!-- lesson-reading:end -->', '', source, flags=re.S)
    if 'class="life-nav"' in source or 'class="article-toc"' in source:
        return source
    links = []
    def heading(match):
        attrs, body = match.groups()
        title = re.sub(r'<[^>]+>', '', body).strip()
        if not title or title == '本课追踪什么实现':
            return match[0]
        existing = re.search(r'\bid="([^"]+)"', attrs)
        anchor = existing[1] if existing else 'reading-topic-' + str(len(links)+1)
        links.append((anchor, title))
        return f'<h2{attrs if existing else attrs + chr(32) + "id=" + chr(34) + anchor + chr(34)}>{body}</h2>'
    updated = re.sub(r'<h2([^>]*)>(.*?)</h2>', heading, source, flags=re.S)
    if len(links) < 4:
        return source
    nav = '<!-- lesson-reading:start --><nav class="lesson-reading-nav" aria-label="本课阅读入口"><span>按问题阅读</span>' + ''.join(f'<a href="#{anchor}">{title}</a>' for anchor, title in links) + '</nav><!-- lesson-reading:end -->'
    return re.sub(r'(<header class="lesson-header".*?</header>)', lambda m: m[0] + nav, updated, count=1, flags=re.S)


def rendered(page, module=None, lesson=None):
    source = page.read_text(encoding='utf-8')
    if 'motion/integration.js' not in source:
        source = source.replace('<script ', f'<script src="{href(page, "motion/integration.js")}" defer></script><script ', 1)
    source, count = re.subn(r'<aside class="site-sidebar".*?</aside>', lambda _: sidebar(page, module), source, count=1, flags=re.S)
    if count != 1:
        raise ValueError(f'{page}: expected exactly one site sidebar')
    if page.parent == PAGES / 'foundations':
        return source
    if not module:
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
    targets = [(PAGES / 'index.html', None, None)]
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
    print(f'{"Checked" if check else "Generated"} shared framework navigation: {len(targets)} pages, 2 chapters, {len(FOUNDATIONS)} foundations, 12 SGLang modules, 20 implementation courses.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    build(parser.parse_args().check)
