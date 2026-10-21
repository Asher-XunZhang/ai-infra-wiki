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
    sources = ''.join(f'<li><a href="{escape(url, quote=True)}">{escape(name)}</a></li>' for name, url in c['sources'])
    visual_refs = ''
    if c.get('visual_references'):
        visual_refs = '<details class="concept-background"><summary>画法参考与演示范围</summary><ul>' + ''.join(f'<li><a href="{escape(url, quote=True)}">{escape(name)}</a>：{escape(note)}</li>' for name, url, note in c['visual_references']) + '</ul><p>本课数值、交互模型与图形独立实现；输入输出和完成条件按上方官方文档复核。参考资料读取：2026-10-19。</p></details>'
    choices = ''.join(f'<label><input type="radio" name="prediction" value="{n}"><span>{escape(value)}</span></label>' for n, value in enumerate(c['choices']))
    previous = '<a href="../#chapter-foundations"><small>第一章目录</small><strong>回到推理基础</strong></a>' if i == 0 else f'<a href="{FOUNDATIONS[i-1]["slug"]}.html"><small>上一课</small><strong>{escape(FOUNDATIONS[i-1]["title"])}</strong></a>'
    following = '<a class="next" href="../#chapter-sglang"><small>进入第二章</small><strong>SGLang 实现剖析</strong></a>' if i == len(FOUNDATIONS)-1 else f'<a class="next" href="{FOUNDATIONS[i+1]["slug"]}.html"><small>下一课</small><strong>{escape(FOUNDATIONS[i+1]["title"])}</strong></a>'
    extension = EXTENSIONS.get(c['slug'])
    extension_assets = extension_html = ''
    mechanism_assets = '<script src="../motion/batching.js" defer></script><link rel="stylesheet" href="../motion/teaching.css"><script src="../motion/teaching-visuals.js" defer></script><script src="../motion/dynamic-batching.js" defer></script><script src="../motion/batching-chunk.js" defer></script>' if c['slug'] == 'batching' else ''
    if c['slug'] in {'tokens', 'generation', 'attention', 'cache', 'deployment'}:
        mechanism_assets += '<link rel="stylesheet" href="../motion/teaching.css"><script src="../motion/teaching-scenes.js" defer></script><script src="../motion/teaching-visuals.js" defer></script>'
        if c['slug'] != 'deployment':
            mechanism_assets += '<script src="../motion/basics-visuals.js" defer></script>'
    if c['slug'] in {'hardware', 'parallel', 'communication', 'speculation', 'serving'}:
        mechanism_assets += '<link rel="stylesheet" href="../motion/teaching.css"><script src="../motion/teaching-visuals.js" defer></script><script src="../motion/detail-scenes.js" defer></script><script src="../motion/detail-visuals.js" defer></script>'
    if c['slug'] == 'metrics':
        mechanism_assets += '<link rel="stylesheet" href="../motion/teaching.css"><script src="../motion/teaching-visuals.js" defer></script><script src="../motion/metrics.js" defer></script>'
    if c['slug'] == 'communication':
        mechanism_assets += '<link rel="stylesheet" href="../motion/communication.css"><script src="communication-model.js" defer></script><script src="../motion/communication-visuals.js" defer></script>'
    if c['slug'] == 'parallel':
        mechanism_assets += '<script src="parallel-model.js" defer></script><script src="../motion/parallel-scenes.js" defer></script><script src="../motion/parallel-tensor.js" defer></script><script src="../motion/parallel-routing.js" defer></script><script src="../motion/parallel-visuals.js" defer></script>'
    if extension:
        extension_assets = '<link rel="stylesheet" href="../sglang/inference-overview/overview.css"><link rel="stylesheet" href="../diagrams.css"><script src="../sglang/inference-overview/model.js" defer></script><script src="../diagrams.js" defer></script><script src="../sglang/inference-overview/overview.js" defer></script>'
        if c['slug'] == 'generation':
            extension_assets += '<link rel="stylesheet" href="../motion/journey-layer.css"><script src="../motion/journey-layer.js" defer></script>'
        extension_html = (f'<details class="foundation-extension" id="mechanism-extension"><summary>选读 · {escape(extension["title"])}</summary>'
            '<p>主线之外的完整机制对照图。沿用原课程的教学假设；不代表 SGLang 的运行轨迹。</p>'
            + extension['html'] + f'<p><a href="../sglang/inference-overview/{extension["source_lesson"]}">转到 SGLang：核对实现接口与固定源码 →</a></p></details>')
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
{learning_entry}
<div id="lesson-animation"></div>
{animation_anchor}
<section class="foundation-lab" data-foundation-lab="{c['slug']}" data-full="true" data-title="{title}"><h2>走一遍具体过程</h2><p>{escape(c['goal'])}</p><p>交互需要 JavaScript；下面保留完整解释和自测答案。</p></section>
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
