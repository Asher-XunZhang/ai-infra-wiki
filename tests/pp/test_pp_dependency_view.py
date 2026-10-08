"""Regression checks for the embedded renderer's actual wheel/overlay functions."""
import json
import subprocess

from build_pp_quick_data import EmbeddedModel, PAGE

NODE = r"""
const assert = require('node:assert/strict'), vm = require('node:vm'), fs = require('node:fs');
const {code,graph} = JSON.parse(fs.readFileSync(0,'utf8'));
let selected='0:5:gpu', elements=[], hint={hidden:true,textContent:''}, wheel;
const context = vm.createContext({
  graph,
  root:{querySelector:()=>hint},
  selectedNode:()=>selected?{id:selected,...graph[selected]}:null,
  el:(parent,tag,attrs={})=>{const e={tag,attrs};elements.push(e);return e;},
  text:(parent,x,y,value,attrs={})=>{const e={tag:'text',value,attrs:{x,y,...attrs}};elements.push(e);return e;},
  timeline:{getBoundingClientRect:()=>({width:1000,left:0}),addEventListener:(type,fn)=>{assert.equal(type,'wheel');wheel=fn;}},
});
context.draw=()=>{
  elements=[];
  vm.runInContext('dependencyOverlay({},t=>84+(t-viewStart)/viewSpan*896,52,194,viewStart,viewStart+viewSpan)',context);
};
vm.runInContext('let viewStart=0,viewSpan=110,total=110;const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));'+code,context);
const run=code=>vm.runInContext(code,context);
const lines=()=>elements.filter(e=>e.attrs['data-dependency']);
const view=(start,span)=>run(`setView(${start},${span})`);
const bounds=()=>{
  for(const e of lines()){
    const coordinates=e.tag==='path'?[...e.attrs.d.matchAll(/[ML]([\d.e+-]+),([\d.e+-]+)/g)].map(m=>+m[1]):[e.attrs.x1,e.attrs.x2];
    assert(coordinates.length,'dependency has visible geometry');
    for(const x of coordinates)assert(Number.isFinite(x)&&x>=84-1e-6&&x<=980+1e-6,`out-of-view dependency x=${x}`);
  }
};
// A pointer at t=22 keeps part of M3's 18–24 u forward visible while
// zooming its start off the left edge. This used to drop EVERY edge.
for(const modifier of ['ctrlKey','metaKey']){
  view(15,12);assert.equal(lines().length,2);
  let prevented=false;
  wheel({[modifier]:true,clientX:84+896*7/12,deltaY:-Math.log(1.8)/.005,preventDefault:()=>{prevented=true;}});
  assert(prevented);assert(run('viewStart')>18&&run('viewStart')<24);
  assert.equal(lines().length,2,'wheel zoom must retain dependencies for a partially visible selection');
  assert(lines().every(e=>e.attrs['data-target-offscreen']==='true'));assert(!hint.hidden);bounds();
}
// Plain scrolling must not zoom the chart.
const old=run('viewSpan');wheel({clientX:500,deltaY:-50,preventDefault:()=>assert.fail('plain scroll blocked')});assert.equal(run('viewSpan'),old);
// A target to the right still exposes the selected dependency at the edge.
view(15,2);assert.equal(lines().length,2);assert(!hint.hidden);bounds();
// Fully offscreen relations remain explicitly marked; restoring the window
// restores real endpoint positions and clears the offscreen hint.
view(30,2);assert.equal(lines().length,2);assert(!hint.hidden);bounds();
view(15,12);assert.equal(lines().length,2);assert(hint.hidden);assert(lines().every(e=>e.attrs['data-target-offscreen']==='false'));bounds();
// Cross-rank dependencies retain their dashed style when clipped.
selected='0:5:recv_out';view(28,2);
assert.equal(lines().length,2);assert(lines().some(e=>e.attrs['stroke-dasharray']==='5 3'));bounds();
// Only a real visible adjacent boundary may be represented as a shared edge.
selected='0:5:commit_admission';view(14,4);
assert(lines().some(e=>e.attrs['data-dependency-kind']==='boundary'));
view(30,2);assert(lines().every(e=>e.attrs['data-dependency-kind']==='arrow'));bounds();
// The selected start exactly on the right boundary must not disappear.
selected='0:5:gpu';view(16,2);assert.equal(lines().length,2);bounds();
selected=null;view(16,2);assert.equal(lines().length,0);assert(hint.hidden);
console.log('Dependency viewport: Ctrl/Meta wheel zoom, clipping, restore, boundaries and plain scrolling passed.');
"""


def check(srcdoc):
    model = EmbeddedModel()
    model.feed(srcdoc)
    graph = json.loads(''.join(model.chunks))['graph']
    pieces = []
    for name, following in [('setView', 'zoomBy'), ('zoomBy', 'focusDependencies'),
                            ('edges', 'populateSteps'), ('laneGeometry', 'dependencyOverlay'),
                            ('dependencyOverlay', 'mark')]:
        begin = srcdoc.index('function '+name+'(')
        end = srcdoc.index('function '+following+'(', begin)
        pieces.append(srcdoc[begin:end])
    wheel = next(line for line in srcdoc.splitlines() if "timeline.addEventListener('wheel'" in line)
    pieces.append(wheel)
    result = subprocess.run(['node','-e',NODE], input=json.dumps(dict(code='\n'.join(pieces),graph=graph)),
                            encoding='utf-8', timeout=30)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    outer = EmbeddedModel()
    outer.feed((PAGE/'index.html').read_text(encoding='utf-8'))
    check(outer.srcdoc)
