#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render source-pinned follow-on courses and independent foundation experiments."""
import argparse
import json
import os
import re
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / 'pages'
DATA = json.loads((PAGES / 'interactive-lessons/lessons.json').read_text(encoding='utf-8'))
SOURCES = json.loads((PAGES / 'interactive-lessons/sources.json').read_text(encoding='utf-8'))


def href(page, target):
    if target.startswith('doc:'):
        return 'https://github.com/Asher-XunZhang/ai-infra-wiki/blob/main/' + target[4:]
    if target.startswith(('https://', 'http://')):
        return target
    path, _, anchor = target.partition('#')
    return os.path.relpath(PAGES / path, page.parent).replace(os.sep, '/') + ('#' + anchor if anchor else '')


def assets(page):
    style = ''.join(f'<link rel="stylesheet" href="{href(page, p)}">' for p in
                    ('learning-depth.css', 'interactive-lessons/lessons.css'))
    scripts = ''.join(f'<script src="{href(page, p)}" defer></script>' for p in
                      ('learning-depth.js', 'interactive-lessons/models.js',
                       'interactive-lessons/views.js', 'interactive-lessons/controller.js'))
    return style + scripts


def paragraphs(items):
    return ''.join('<p>' + escape(s) + '</p>' for s in items)


def experiment(key, page):
    data = DATA[key]
    config = {'unit': data['unit'], 'sources': SOURCES[data['sources']]}
    controls = []
    for c in data['controls']:
        options = ''.join('<option value="' + escape(str(v), quote=True) + '"' +
                          (' selected' if str(v) == str(c['value']) else '') + '>' + escape(str(label)) + '</option>'
                          for v, label in c['values'])
        controls.append('<label>' + escape(c['label']) + '<select data-param="' + c['name'] +
                        '" data-type="' + ('number' if c['numeric'] else 'string') + '">' + options + '</select></label>')
    unit = '讲解位置' if data['unit'] == 'step' else '教学进度'
    note = ('播放器只推进推导或检查顺序，不表示运行耗时。' if data['unit'] == 'step' else
            'u 是给定的教学时间单位；轨道与图共用时钟，不代表实测或速度预测。')
    legend = '' if data['unit'] == 'step' else '<ul class="plab-legend"><li>绿色：本地处理</li><li>蓝色：载荷传递</li><li>虚框：等待</li><li>黄色：控制完成点</li>' + ('<li>红色：给定异常</li>' if key == 'faults' else '') + '</ul>'
    return ('<section class="plab" data-priority-lab="' + data['kind'] + '" data-priority-key="' + key +
            '" data-priority-config="' + escape(json.dumps(config, ensure_ascii=False), quote=True) + '">' +
            '<h3>改变一个条件，再解释结果</h3><div class="plab-controls">' + ''.join(controls) + '</div>' +
            '<div class="plab-transport"><button type="button" data-priority-action="prev">上个检查点</button>' +
            '<button type="button" data-priority-action="play" aria-pressed="false">播放</button>' +
            '<button type="button" data-priority-action="next">下个检查点</button>' +
            '<button type="button" data-priority-action="reset">重置</button>' +
            '<label class="plab-speed animation-speed">回看速度 <select data-priority-speed aria-label="回看速度">' +
            '<option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select>' +
            '<output data-priority-speed-label>1×</output></label></div>' +
            '<div class="plab-seek"><label>' + unit + '</label><input data-priority-seek type="range" min="0" max="1" step="0.001" value="0" aria-label="' + unit + '"><output data-priority-clock></output></div>' +
            '<p class="plab-note">' + note + ' 窄屏可在图或表内横向查看，控件与说明留在图外。</p>' +
            legend + '<div class="plab-figure" tabindex="0" role="region" aria-label="可横向查看的机制过程"></div>' +
            '<div class="plab-panel plab-state" aria-live="polite"><h3>准备观察</h3><p>启用 JavaScript 后可操作；下方正文与算例可独立阅读。</p></div>' +
            '<p class="plab-note">条件切换回到起点并暂停；展开与收起保留当前参数、时刻和播放状态。' +
            '完整箭头说明载荷；只有明确的传递区间使用持续脉冲，本地引用与等待分别表示。</p>' +
            '<noscript><p>互动需要 JavaScript；本课正文、算例、自测和固定源码仍可阅读。</p></noscript></section>')


def body(key, page, fragment=False):
    data = DATA[key]
    sections = [paragraphs(data['intro']), experiment(key, page)]
    for item in data['sections']:
        text = '<section class="overview-section" id="' + item['id'] + '"><h2>' + escape(item['title']) + '</h2>'
        text += paragraphs(item.get('paragraphs', []))
        if item.get('table'):
            rows = item['table']
            text += '<div class="plab-table"><table><thead><tr>' + ''.join('<th scope="col">' + escape(s) + '</th>' for s in rows[0]) + '</tr></thead><tbody>'
            text += ''.join('<tr>' + ''.join('<td>' + escape(s) + '</td>' for s in row) + '</tr>' for row in rows[1:]) + '</tbody></table></div>'
        if item.get('code'):
            text += '<pre><code>' + escape(item['code']) + '</code></pre>'
        sections.append(text + '</section>')
    sections.append('<section class="overview-section" id="check-' + key + '"><h2>先预测，再解释</h2>' +
                    ''.join('<details class="self-check"><summary>' + escape(q) + '</summary><p>' + escape(a) + '</p></details>'
                            for q, a in data['checks']) + '</section>')
    if data.get('related'):
        sections.append('<section class="overview-section"><h2>回看与下一步</h2><ul>' + ''.join(
            '<li><a href="' + escape(href(page, target), quote=True) + '">' + escape(label) + '</a></li>'
            for label, target in data['related']) + '</ul></section>')
    evidence_id = 'evidence-' + key if fragment else 'evidence'
    sources = SOURCES[data['sources']]
    source = sources[0]
    origin = '公开固定 Git 文件的只读核对；vLLM 未检出、导入或运行。' if key == 'vllm' else '读取干净检出目录中的固定 Git 对象，未修改外部源码。'
    text = ('<section class="overview-section" id="' + evidence_id + '"><h2>来源、固定实现与验证边界</h2>' +
            '<p>概念与图示复核日期：<time>2026-11-23</time>。源码仓库 <a href="https://github.com/' + source['repo'] + '">' + source['repo'] + '</a>，' +
            ('固定 tag v0.33.0rc3 提交 ' if key == 'vllm' else '固定 main 快照 ') + '<code>' + source['commit'] + '</code>。' + origin + '</p>' +
            '<p>数值、候选、任务与时间轨道为教学模型；静态锚点、独立算术和浏览器验证不等于模型、GPU、网络、精度或性能实验。</p><ul>')
    text += ''.join('<li><a href="' + s['url'] + '"><code>' + escape(s['path'] + '::' + s['symbol']) + '</code></a></li>' for s in sources)
    text += '</ul>'
    if data['refs']:
        text += '<p>概念与成熟图示参考：' + '；'.join('<a href="' + url + '">' + escape(label) + '</a>' for label, url in data['refs']) + '。工具文档只用于概念与读图，不替代所选提交的实现边界。</p>'
    sections.append(text + '</section>')
    html = ''.join(sections)
    if fragment:
        return '<section id="' + data['anchor'] + '" class="priority-foundation"><h2>' + escape(data['title']) + '</h2><p>' + escape(data['lead']) + '</p>' + html + '</section>'
    return html


def foundation_fragment(slug):
    keys = ['sampling', 'stopping'] if slug == 'generation' else ['quantization-intuition'] if slug == 'hardware' else []
    page = PAGES / 'foundations' / (slug + '.html')
    return ''.join(body(key, page, True) for key in keys)


def render_page(key):
    from build_learning_navigation import rendered, MODULES
    data = DATA[key]
    page = PAGES / data['path']
    prefix = '../../'
    chapter = '第二章 · 引擎对照的独立源码导读' if key == 'vllm' else '第三章 · SGLang 实现剖析'
    heading = ('<nav class="learning-breadcrumb" aria-label="当前位置"><a href="../../#reading-routes">学习首页</a><span>/</span><span>' + chapter + '</span></nav>' if key == 'vllm' else '<!-- learning-context:start --><!-- learning-context:end -->')
    intro = ('<header class="lesson-header"><div class="lesson-kicker"><a class="eyebrow" href="../../#' + ('chapter-engine' if key == 'vllm' else data['module']) + '">' + chapter + '</a>' +
             '<span class="lesson-count">原理、实现与验证边界分别说明</span></div><h1>' + escape(data['title']) + '</h1><p>' + escape(data['lead']) + '</p></header>')
    toc = '<nav class="article-toc" aria-label="本课入口">' + ''.join('<a href="#' + item['id'] + '">' + escape(item['title'].split('：')[0]) + '</a>' for item in data['sections']) + '<a href="#evidence">固定源码</a></nav>'
    pager = '<nav class="lesson-pager" aria-label="继续学习"><a href="../../engine-comparison/#process">返回第二章请求对照</a><a class="next" href="../../#reading-routes">按问题选择下一步</a></nav>'
    main = heading + intro + toc + body(key, page) + pager
    head = ('<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + escape(data['title']) + ' · AI Infra Wiki</title>' +
            '<meta name="description" content="' + escape(data['lead'], quote=True) + '"><link rel="stylesheet" href="../../site.css"><link rel="icon" href="data:,">' +
            '<script src="../../motion/integration.js" defer></script><script src="../../site.js" defer></script>' + assets(page))
    template = (ROOT / 'scripts/templates/priority-page.html.in').read_text(encoding='utf-8')
    html = template.replace('$HEAD', head).replace('$MAIN', main).replace('$PAGE', key)
    if key == 'vllm':
        return rendered(page, source=html)
    module = next(m for m in MODULES if m[0] == data['module'])
    lesson = next(l for l in module[4] if l[0] == data['path'])
    return rendered(page, module, lesson, source=html)


def build(check=False):
    stale = []
    for key, data in DATA.items():
        if not data.get('path'):
            continue
        page = PAGES / data['path']
        html = render_page(key)
        if not page.is_file() or page.read_text(encoding='utf-8') != html:
            if check:
                stale.append(data['path'])
            else:
                page.parent.mkdir(parents=True, exist_ok=True)
                page.write_text(html, encoding='utf-8')
    if stale:
        raise SystemExit('Follow-on lessons are stale: ' + ', '.join(stale))
    print(('Checked' if check else 'Generated') + ' 7 implementation/read-through pages and 3 foundation experiments.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    build(parser.parse_args().check)
