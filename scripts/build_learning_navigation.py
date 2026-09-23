#!/usr/bin/env python3
"""Render the shared learning framework into source HTML; --check detects drift."""
import argparse
import os
import re
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / 'pages'
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
        ('sglang/inference-overview/journey.html', '一条请求怎样变成回答', 'SGLang · 入门')]),
    ('request-runtime', '请求生命周期与运行时架构', 'foundations', False, [
        ('sglang/request-runtime/index.html', '一条请求穿过哪些进程与对象', 'SGLang · 入门')]),
    ('model-execution', '模型执行、硬件与算子', 'instance', False, [
        ('sglang/inference-overview/transformer.html', 'P / D 在 Transformer 里做什么', 'SGLang · 入门')]),
    ('kv-memory', 'KV Cache 与内存管理', 'instance', False, [
        ('sglang/inference-overview/kv-cache.html', 'KV：模型留给下一轮的记忆', 'SGLang · 入门')]),
    ('scheduling', '调度与批处理', 'instance', False, [
        ('sglang/inference-overview/scheduling.html', '多请求怎样共享一次执行', 'SGLang · 入门')]),
    ('parallelism', '并行与执行拓扑', 'distributed', False, [
        ('sglang/parallelism/index.html', '多张卡，究竟切开了什么？', 'SGLang · 并行分工')]),
    ('communication', '通信与传输', 'distributed', False, [('sglang/communication/index.html', '数据传过去，何时才算可用？', 'SGLang · 通信机制')]),
    ('disaggregation', '分离部署与分布式状态交接', 'distributed', False, [
        ('sglang/inference-overview/deployment.html', '合并部署与 PD 分离', 'SGLang · 入门'),
        ('sglang/pd-prefill-lifecycle/index.html', 'PD Prefill 请求生命周期', 'SGLang · PD＋PP 案例'),
        ('sglang/pd-dataflow/index.html', '数据流、状态与队列', 'SGLang · PD＋PP 案例')]),
    ('advanced-generation', '模型结构与高级生成', 'serving', False, [('sglang/advanced-generation/index.html', '生成一个 token，还会改变哪些状态？', 'SGLang · 高级生成')]),
    ('serving-operations', '服务部署与运行治理', 'serving', False, [('sglang/serving-operations/index.html', '服务在线，请求就一定能完成吗？', 'SGLang · 服务治理')]),
    ('performance', '性能分析与优化方法', 'practice', False, [('sglang/performance-engineering/index.html', '性能数字，究竟量到了哪一段？', 'SGLang · 性能分析')]),
    ('case-studies', '综合案例与源码实践', 'practice', False, [
        ('sglang/pd-prefill-pp-loop/quick.html', 'PP loop 快速入门', 'SGLang · PD＋PP 案例'),
        ('sglang/pd-prefill-pp-loop/index.html', '依赖分析与耗时场景', 'SGLang · PD＋PP 案例'),
        ('sglang/pd-prefill-pp-loop/notes.html', 'PP loop 源码阅读', 'SGLang · PD＋PP 案例')]),
]


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
    lines.extend(['<div class="sidebar-note"><span class="status-dot"></span> 一张框架，逐步深入<p>先理解单实例，再连接多卡多机；模型、服务与性能按目标选读。</p></div>',
                  '<a class="sidebar-repo" href="https://github.com/Asher-XunZhang/ai-infra-wiki">浏览完整资料库 ↗</a>',
                  '</div></details></aside>'])
    return '\n'.join(lines)


def breadcrumb(page, module, lesson):
    stage = next(s for s in STAGES if s[0] == module[2])
    n = MODULES.index(module) + 1
    return ('<!-- learning-context:start -->\n<nav class="learning-breadcrumb" aria-label="当前位置">'
            f'<a href="{href(page, anchor="learning-path")}">学习框架</a><span aria-hidden="true">/</span>'
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


def rendered(page, module=None, lesson=None):
    source = page.read_text(encoding='utf-8')
    source, count = re.subn(r'<aside class="site-sidebar".*?</aside>', lambda _: sidebar(page, module), source, count=1, flags=re.S)
    if count != 1:
        raise ValueError(f'{page}: expected exactly one site sidebar')
    if module:
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
    return source


def build(check=False):
    targets = [(PAGES / 'index.html', None, None)]
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
    print(f'{"Checked" if check else "Generated"} shared framework navigation: {len(targets)} pages, 5 stages, 12 modules, {len(targets) - 1} courses.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    build(parser.parse_args().check)
