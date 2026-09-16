"""Embed the reviewed step guide and derive its readable Markdown companion.

The iframe uses srcdoc with an opaque origin. Inlining keeps it self-contained,
and avoids an unversioned external dependency inside escaped HTML.
"""
import json
import re
import subprocess
from html import escape

from build_pp_quick_data import EmbeddedModel, PAGE, ROOT
from pp_source_baseline import SOURCE_COMMIT, SOURCE_FILES

DOC = ROOT / 'sglang/PD Prefill PP loop 步骤详解.md'
MARKER = '<script data-step-guide>\n'
EXAMPLES = r"""
const fs = require('node:fs');
const guide = require(process.argv[1]);
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
const nodes = Object.entries(data.graph).map(([id, n]) => ({id, ...n}));
const examples = guide.operationTypes.map(op => {
  const candidates = nodes.filter(n => n.id.split(':')[2] === op);
  const score = n => (n.r === 1 ? 10 : 0) + (n.n === 4 ? 4 : 0)
    + (op === 'ack_events' && n.deps.some(id => data.graph[id].kind === 'h2d') ? 20 : 0)
    + (op === 'release' && data.releases.some(x => x.r === n.r && x.n === n.n) ? 20 : 0);
  const node = candidates.sort((a,b) => score(b)-score(a))[0];
  return guide.describe(data,node);
});
const branches = ['0:0:select','0:3:init_load','0:3:start_load','0:0:ack_events',
  '0:2:release','0:0:recv_req','0:0:l2_counts','2:3:send_out','2:5:copy',
  '2:3:send_bc','2:3:send_rc'].map(id => guide.describe(data,{id,...data.graph[id]}));
for(const kind of ['wait','idle']) {
  const e = data.events.find(e => e.kind === kind);
  branches.push(guide.describe(data,{id:e.target,...data.graph[e.target]},e));
}
const e = data.events.find(e => e.id.endsWith(':layer-wait'));
if (e) branches.push(guide.describe(data,{id:e.target,...data.graph[e.target]},e));
process.stdout.write(JSON.stringify({examples,branches}));
"""


def inline_guide(srcdoc):
    code = (PAGE / 'step-guide.js').read_text(encoding='utf-8').rstrip()
    assert '</script' not in code.lower(), 'Inline script closing tag in guide'
    script = MARKER + code + '\n</script>'
    if MARKER in srcdoc:
        return re.sub(r'<script data-step-guide>\n.*?</script>', lambda _: script,
                      srcdoc, count=1, flags=re.S)
    marker = '  <script>\n'
    assert marker in srcdoc
    return srcdoc.replace(marker, script + '\n' + marker, 1)


def markdown(model):
    result = subprocess.run(['node', '-e', EXAMPLES, str(PAGE / 'step-guide.js')],
                            input=json.dumps(model), capture_output=True,
                            encoding='utf-8', check=True, timeout=30)
    cases = json.loads(result.stdout)
    lines = [
        '# PD Prefill PP loop：逐步理解行为与目的', '',
        '本文是[依赖分析交互图说明](PD%20Prefill%20PP%20loop%20交互图.md)的逐项阅读版。'
        '每个示例说明“在做什么、为什么需要、完成后推进什么”。交互页面会按所选场景和步骤实时替换对象与等待条件；本文的数字和对象只对应原示例。', '',
        '## 1. 阅读基线', '',
        '| 项目 | 内容 |', '| --- | --- |',
        '| 源码仓库 | [sgl-project/sglang](https://github.com/sgl-project/sglang) |', '| 分支 | `main` |',
        f'| 固定提交 | `{SOURCE_COMMIT}` |', '| 读取时间 | 2026-09-17 |',
        '| 工作区状态 | 读取时干净，无本地改动或未跟踪文件 |',
        '| 操作边界 | 只读源码；验证教学模型与页面，未运行 SGLang 或 GPU 实验 |', '',
        '范围固定为 PP=3、depth=0、CUDA、UnifiedRadixCache + HiCache cache 模式，'
        '完整 Prefill 成功路径。M2/M4 有 host hit；无中间 chunk、L3、sampling mask 或推测解码。'
        '时间 u 是模型示意值；选批安排固定，资源串行和整批 H2D 屏障属于模型简化。', '',
        'PP0/PP1/PP2 都在 Prefill 侧。M# 是 batch；L# 是本级槽位迭代。'
        '同一轮的新前向、旧结果和队列级名单可能属于不同请求。', '',
        '| 对象 | 如何理解 |', '| --- | --- |',
        '| proxy / hidden states | 上一级模型层的中间特征，送往下一 PP 级继续计算 |',
        '| output | 末级产生的 token 等结果，沿 PP 环回流，供各级更新请求 |',
        '| KV | 各级本地层的注意力缓存，发送给图外 Decode |',
        '| ACK / event | ACK 是完成队列条目；event 是异步工作完成的标记，仍需本级确认 |',
        '| Work | 异步通信的句柄，回收它不等于执行新的发送 |', '',
        '## 2. 各类操作', '',
        '以下按调度、计算、结果、共识和通信整理；实际先后以交互图时间轴及前置依赖为准。', '',
    ]

    def add_case(item, number):
        lines.extend([f"### {number}. {item['title']}", '', f"**当前例子：** {item['context']}", ''])
        for key, label in [('wait','正在等待'),('what','等到后做什么' if item.get('wait') else '在做什么'),
                           ('why','为什么需要'),('next','完成后'),('boundary','读图边界')]:
            if item.get(key):
                lines.extend([f"**{label}：** {item[key]}", ''])
        links = []
        for ref in item['refs']:
            key, line = ref.split(':')
            links.append(f'[{ref}](https://github.com/sgl-project/sglang/blob/{SOURCE_COMMIT}/python/sglang/srt/{SOURCE_FILES[key]}#L{line})')
        lines.extend(['**源码：** ' + ' · '.join(links), ''])

    for i, item in enumerate(cases['examples'], 1):
        add_case(item, f'2.{i}')
    lines.extend(['## 3. 分支与等待：同名步骤不一定在做同样的工作', '',
                  '空轮、无 host hit、末级发出与中间级转发，必须结合当前上下文阅读。'
                  '斜线等待是前置条件未满足的区间，不是重复执行一次操作。', ''])
    for i, item in enumerate(cases['branches'], 1):
        add_case(item, f'3.{i}')
    lines.extend(['## 4. 维护与验证', '',
                  '文字维护在 `pages/sglang/pd-prefill-pp-loop/step-guide.js`。'
                  '运行 `python -B scripts/build_pp_step_guide.py` 同步 iframe 和本文；'
                  '`build_pp_scenarios.py` 重建场景时也会同步。', '',
                  '`test_pp_step_guide.py` 检查全部操作、等待、空轮、ACK、release、'
                  '发送者分支和自定义时间场景，防止说明沿用错误的 batch 或固定轮次。'
                  '`check_pp_source.py --source-root "$SGLANG_SOURCE_ROOT"` 检查固定基线与逐行源码锚点；`SGLANG_SOURCE_ROOT` 由读者设置为官方仓库的检出目录。', ''])
    return '\n'.join(lines)


def build():
    page = PAGE / 'index.html'
    text = page.read_text(encoding='utf-8')
    outer = EmbeddedModel(); outer.feed(text)
    updated = inline_guide(outer.srcdoc)
    old = escape(outer.srcdoc, quote=True)
    assert text.count(old) == 1
    if updated != outer.srcdoc:
        page.write_text(text.replace(old, escape(updated, quote=True), 1), encoding='utf-8')
    inner = EmbeddedModel(); inner.feed(updated)
    DOC.write_text(markdown(json.loads(''.join(inner.chunks))), encoding='utf-8')


if __name__ == '__main__':
    build()
