#!/usr/bin/env python3
"""Render the source-pinned architecture map as readable, no-JS HTML/SVG."""
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def render_architecture():
    data = json.loads((ROOT / 'pages/sglang/system-overview/architecture.json').read_text())
    base = 'https://github.com/sgl-project/sglang/'
    rev = data['revision']
    def source_link(s):
        file, line, symbol = s
        return f'<a href="{base}blob/{rev}/{file}#L{line}"><code>{escape(file)}</code> · L{line}</a>'
    parts = ['<!-- system-architecture:start -->', '<section class="architecture" id="architecture" aria-labelledby="architecture-title">',
        '<h2 id="architecture-title">代码架构：先看全局，再进入模块</h2>',
        '<p>入口管理请求，Scheduler 组织本轮工作，执行层调用模型；KV 管理为两者提供可复用的历史和写入位置，反分词将输出送回入口。</p>',
        '<p class="arch-scope"><strong>图的范围</strong> 普通 Python 文本生成 · 单 tokenizer · 单卡合并部署 · 同一主机。用无 overlap（计算与调度重叠）的路径说明调用关系。</p>',
        '<div class="arch-views" role="group" aria-label="架构阅读视角" hidden><button type="button" data-arch-view="all" aria-pressed="true">完整架构</button><button type="button" data-arch-view="input" aria-pressed="false">请求进入</button><button type="button" data-arch-view="compute" aria-pressed="false">一轮计算</button><button type="button" data-arch-view="output" aria-pressed="false">结果返回</button></div>',
        '<p class="arch-hint">点选模块查看职责、固定源码与课程。窄屏可横向查看全图，或使用图下的模块索引。</p>',
        '<div class="arch-scroll" tabindex="0" role="region" aria-label="SGLang 代码架构图，可横向滚动"><svg class="arch-map" viewBox="0 0 1200 800" role="group" aria-labelledby="arch-svg-title arch-svg-desc">',
        '<title id="arch-svg-title">SGLang 普通 Python 文本生成的模块与协作</title>',
        '<desc id="arch-svg-desc">入口进程包含 HTTP 和 TokenizerManager；Scheduler 进程内组织调度、KV 管理、Worker、ModelRunner、模型与算子；独立 Detokenizer 进程把文字结果送回入口。GPU 是设备资源。跨进程箭头标有 IPC，其余进程内箭头表示调用或资源管理。</desc>',
        '<defs><marker id="arch-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto-start-reverse"><path d="M0 0L7 3.5L0 7Z"/></marker></defs>',
        '<g class="arch-boundary"><rect x="20" y="145" width="250" height="560" rx="12"/><text x="38" y="172">入口进程</text><rect x="395" y="145" width="510" height="560" rx="12"/><text x="415" y="172">Scheduler 进程 · 本例一张 GPU</text><rect x="1020" y="145" width="175" height="560" rx="12"/><text x="1033" y="172">反分词进程</text></g>',
        '<g class="arch-client"><rect x="40" y="25" width="210" height="55" rx="8"/><text x="145" y="59">客户端</text></g>',
        '<text class="arch-legend" x="420" y="49">SRT（SGLang Runtime，推理运行时）</text><text class="arch-legend" x="420" y="78">代码主线：python/sglang/srt/</text>',
        '<text class="arch-legend" x="420" y="110">进程外框内为组件分工；IPC = 进程间通信</text>']
    for e in data['edges']:
        phase = ' '.join(e['phase'])
        parts.append(f'<g class="arch-edge" data-edge="{e["id"]}" data-phase="{phase}"><path d="{e["path"]}" marker-end="url(#arch-arrow)"'+(' marker-start="url(#arch-arrow)"' if e.get('both') else '')+'/>')
        x,y=e['label']
        parts.append(f'<text x="{x}" y="{y}">'+''.join(f'<tspan x="{x}" dy="{0 if i==0 else 22}">{escape(line)}</tspan>' for i,line in enumerate(e['lines']))+'</text></g>')
    for n in data['nodes']:
        x,y,w,h=n['box']
        # Longer code identifiers are in the accessible detail below the map.
        directory=n['directory'] if n['id']!='tokenizer' else 'managers/'
        parts.append(f'<a class="arch-node" href="#arch-{n["id"]}" data-arch-node="{n["id"]}" data-phase="{" ".join(n["phase"])}" aria-label="{escape(n["title"])}：查看职责与源码"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>')
        parts.append(f'<text class="arch-name" x="{x+w/2}" y="{y+28}">{escape(n["title"])}</text><text class="arch-code" x="{x+w/2}" y="{y+52}">{escape(n["name"])}</text><text class="arch-directory" x="{x+w/2}" y="{y+74}">{escape(directory)}</text></a>')
    parts += ['<g class="arch-device"><rect x="675" y="570" width="205" height="100" rx="8"/><text class="arch-name" x="777" y="600">GPU 设备资源</text><text x="777" y="628">权重 · 激活 · 各层 KV</text><text class="arch-code" x="777" y="653">由当前进程调用与管理</text></g></svg></div>',
        '<p class="arch-caption">外框表示所选配置的进程，GPU 框表示设备。进程内箭头是函数调用、返回或资源管理；标有 IPC 的箭头传递进程间消息。本图概括依赖关系，不规定各阶段的实测耗时或重叠时序。<a href="../communication/">深入通信边界</a></p>',
        '<nav class="arch-index" aria-label="架构模块索引">'+''.join(f'<a href="#arch-{n["id"]}" data-arch-select="{n["id"]}">{n["title"]}</a>' for n in data['nodes'])+'</nav>',
        '<p class="arch-view-note" role="status" hidden>完整架构：沿请求与返回闭环阅读，KV 管理与执行在同一调度进程内协作。</p>',
        '<div class="arch-details">']
    for n in data['nodes']:
        parts += [f'<details id="arch-{n["id"]}" class="arch-detail"><summary>{n["title"]} · {n["name"]}</summary><p><strong>为什么需要：</strong>{n["why"]}</p><p>{n["action"]}</p><ul class="arch-source-list">'+''.join(f'<li>{source_link(s)}</li>' for s in n['sources'])+'</ul>', f'<a href="{n["course"][1]}">进入课程：{n["course"][0]}</a></details>']
    parts += ['</div><details class="arch-extension"><summary>其他目录怎样接到主线？</summary><p>这张图聚焦 SRT 在线文本服务。仓库还有 <code>python/sglang/lang/</code> 语言前端、<code>multimodal_gen/</code> 生成路径和测试、基准工具；它们不构成这条普通请求的串行步骤。</p><dl>']
    for item in data['extensions']:
        links=' · '.join(f'<a href="{base}tree/{rev}/{path}"><code>{path}</code></a>' for path in item['paths'])
        parts.append(f'<dt><a href="{item["link"]}">{item["title"]}</a></dt><dd>{item["text"]}<p>{links}</p></dd>')
    parts += ['</dl></details><details class="arch-provenance"><summary>源码基线与架构图参考</summary>',
        f'<p>本图的模块、目录和连接已逐项核对 <a href="{base}tree/{rev}">SGLang {rev[:10]}</a>，读取于 {data["reviewed"]}。普通进程装配见 {source_link(data["startup"])}。读取的是固定 Git 对象，当前检出目录干净；未修改 SGLang 源码，未运行模型、GPU 或网络实验。</p>',
        '<p>本章主体课程采用此提交；<a href="../pd-dataflow/">PD 数据流专题</a>独立固定在 <code>882577451e</code>，进入该课时以它自己的源码说明为准。</p>',
        '<p>表达参考：Ziyi Xu 的 <a href="https://www.lmsys.org/blog/2025-12-17-minisgl/">Mini-SGLang 作者文章（2025-12-17）</a>及其中系统架构图（读取于 2026-10-19）。借鉴请求闭环、调度与缓存相邻的组织方式；这里的组件与边界重新按完整 SGLang 固定源码绘制。</p></details></section><!-- system-architecture:end -->']
    return '\n'.join(parts)
