"""Check source distinctions and selection context across all timing scenarios."""
import json
import re
import subprocess

from build_pp_quick_data import EmbeddedModel, PAGE, ROOT
from build_pp_step_guide import DOC, inline_guide, markdown
from pp_source_baseline import SOURCE_ANCHORS, STEP_GUIDE_ANCHORS

NODE = r"""
const assert = require('node:assert/strict'), fs = require('node:fs'), vm = require('node:vm');
const page = './pages/sglang/pd-prefill-pp-loop/';
const guide = require(page+'step-guide.js');
const {embedded,baseline,knownRefs,renderer} = JSON.parse(fs.readFileSync(0,'utf8'));
const frame = vm.createContext({});vm.runInContext(embedded,frame);
const sourceTypes = new Set(guide.operationTypes), seen = new Set();
const models = fs.readdirSync(page+'scenarios').filter(f=>f.endsWith('.json')).map(f => JSON.parse(fs.readFileSync(page+'scenarios/'+f,'utf8')).model);
const sandbox={window:{}};
for(const name of ['scenario-catalog.js','timing-template.js'])vm.runInNewContext(fs.readFileSync(page+name,'utf8'),sandbox);
const engine=require(page+'timing-engine.js');
const custom=structuredClone(sandbox.window.PP_SCENARIOS[0].timing);
custom.h2d[1][0]=30;custom.kv[2][4]=100;custom.gpu[1][1]=14;custom.output=3;
models.push(engine.buildPacket(sandbox.window.PP_TIMING_TEMPLATE,custom).model);
const zero=structuredClone(custom);
for(const name of ['gpu','h2d','kv'])zero[name]=zero[name].map(row=>row.map(()=>0));
for(const name of ['proxy','output','control','d2h'])zero[name]=0;
models.push(engine.buildPacket(sandbox.window.PP_TIMING_TEMPLATE,zero).model);
const explain=(model,id,event)=>guide.describe(model,{id,...model.graph[id]},event);
let nodes=0, waits=0, acknowledgements=0, releases=0;
for(const model of models){
  for(const [id,node] of Object.entries(model.graph)){
    if(!model.loops.some(l=>l.r===node.r&&l.n===node.n))continue;
    nodes++;const d=explain(model,id);seen.add(d.op);
    assert.deepEqual(JSON.parse(JSON.stringify(frame.PP_STEP_GUIDE.describe(model,{id,...node}))),d,'embedded guide drift');
    for(const key of ['title','what','why','next','context','boundary']){
      assert(d[key]?.trim(),`${id}: missing ${key}`);
      assert(!/undefined|NaN|PP-1|PP3/.test(d[key]),`${id}: invalid context`);
    }
    assert(d.refs.every(ref=>knownRefs.includes(ref)),`${id}: unreviewed source reference`);
    const loop=model.loops.find(l=>l.r===node.r&&l.n===node.n);
    assert(d.context.includes(`本轮前向 ${loop.current?'M'+loop.current:'无'}`));
    assert(d.context.includes(`本轮旧结果 ${loop.old?'M'+loop.old:'无'}`));
    if(d.op==='ack_events'){
      const local=node.deps.map(id=>model.graph[id]).filter(p=>p.kind==='h2d');
      if(local.length){acknowledgements++;for(const p of local)assert(d.what.includes(p.owner));}
      else assert(d.what.includes('ACK 数为 0'));
    }
    if(d.op==='release'){
      const row=model.releases.find(x=>x.r===node.r&&x.n===node.n);
      if(row){releases++;for(const m of row.batches)assert(d.what.includes('M'+m));}
      else assert(d.what.includes('没有实际清理'));
    }
  }
  for(const event of model.events.filter(e=>['wait','idle'].includes(e.kind))){
    waits++;const d=explain(model,event.target,event);assert(d.wait);
    if(event.kind==='wait')for(const id of event.deps){
      const p=model.graph[id];assert(d.wait.includes(`PP${p.r} · L${p.n+1}`));assert(d.wait.includes(p.end.toFixed(2)+' u'));
    }
    if(event.id.endsWith(':layer-wait'))assert(d.boundary.includes('源码按层等待'));
  }
}
assert.deepEqual(seen,sourceTypes,'every graph operation needs an explanation');
assert(acknowledgements&&releases&&waits);
assert(models[10].loops.length>baseline.loops.length,'custom fixture must exercise extended empty loops');
// The screenshot case: PP1 sends current M2 activation to PP2, not KV/output.
const proxy=explain(baseline,'1:4:proxy_message');
assert(proxy.what.includes('PP1')&&proxy.what.includes('M2')&&proxy.what.includes('PP2'));
assert(proxy.why.includes('后续模型层'));assert(proxy.boundary.includes('不是 PD KV'));
// Same-looking operations take distinct source branches.
assert(explain(baseline,'0:3:init_load').what.includes('跳过'));
assert(explain(baseline,'0:4:init_load').what.includes('设备索引'));
assert(explain(baseline,'0:3:start_load').what.includes('返回 -1'));
assert(explain(baseline,'0:0:select').what.includes('None'));
assert(explain(baseline,'0:3:select').what.includes('M1'));
assert(explain(baseline,'2:3:send_out').what.includes('发往 PP0'));
assert(explain(baseline,'0:6:send_out').what.includes('转发给 PP1'));
assert(explain(baseline,'2:5:copy').boundary.includes('只记录 event'));
assert(explain(baseline,'2:5:copy').boundary.includes('当前前向'));
assert(explain(baseline,'0:5:copy').context.includes('本轮前向 M3；本轮旧结果 M1'));
assert(explain(baseline,'0:0:l2_counts').what.includes('PP0'));
assert(explain(baseline,'1:0:l2_counts').what.includes('PP0 发布'));
assert(explain(baseline,'1:3:send_bc').context.includes('PP1 · L3 的已收共识'));
// Exercise the actual DOM renderer: safe text, source links, changing selection,
// and clearing stale wait prose after following a dependency.
class Element{
  constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.attrs={};this.textContent='';}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(){this.children=[];}
  setAttribute(k,v){this.attrs[k]=v;}
}
const box=new Element('div'), doc={createElement:tag=>new Element(tag),createTextNode:text=>({textContent:text})};
const dom=vm.createContext({PP_STEP_GUIDE:guide,data:baseline,document:doc,root:{querySelector:()=>box},url:ref=>'https://source/'+ref});
vm.runInContext(renderer,dom);
const render=(id,event)=>{const node={id,...baseline.graph[id]};dom.renderOpDetails(event||node,node);};
const flatten=e=>[e,...(e.children||[]).flatMap(flatten)];
render('1:4:proxy_message');assert.equal(box.dataset.operation,'proxy_message');
assert.equal(flatten(box).filter(e=>e.tag==='dt').length,5);
assert.deepEqual(flatten(box).filter(e=>e.tag==='a').map(e=>e.href),proxy.refs.map(ref=>'https://source/'+ref));
const waiting=baseline.events.find(e=>e.kind==='wait');render(waiting.target,waiting);
assert(flatten(box).some(e=>e.dataset?.explanation==='wait'));
render('0:5:copy');assert(!flatten(box).some(e=>e.dataset?.explanation==='wait'));
assert.equal(box.dataset.selectedEvent,'0:5:copy');
console.log(`Step guide: ${sourceTypes.size} operation types, ${models.length} scenarios, ${nodes} selections and ${waits} waits/idle intervals passed.`);
"""


def check():
    outer = EmbeddedModel(); outer.feed((PAGE / 'index.html').read_text(encoding='utf-8'))
    assert inline_guide(outer.srcdoc) == outer.srcdoc, 'Run build_pp_step_guide.py'
    embedded = re.search(r'<script data-step-guide>\n(.*?)</script>', outer.srcdoc, re.S).group(1)
    inner = EmbeddedModel(); inner.feed(outer.srcdoc)
    baseline = json.loads(''.join(inner.chunks))
    assert DOC.read_text(encoding='utf-8') == markdown(baseline), 'Step Markdown is stale'
    start = outer.srcdoc.index('function renderOpDetails(')
    end = outer.srcdoc.index('function showOp(', start)
    result = subprocess.run(['node','-e',NODE], cwd=ROOT, encoding='utf-8', timeout=60,
                            input=json.dumps(dict(embedded=embedded,baseline=baseline,
                                renderer=outer.srcdoc[start:end],
                                knownRefs=list(SOURCE_ANCHORS | STEP_GUIDE_ANCHORS))))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    check()
