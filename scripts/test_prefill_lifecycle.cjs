/* Validate the teaching model's causal/ownership boundaries without SGLang or a GPU. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const model = require('../pages/sglang/pd-prefill-lifecycle/lifecycle-model.js');
const html = fs.readFileSync(path.join(__dirname, '../pages/sglang/pd-prefill-lifecycle/index.html'), 'utf8');
const {events, snapshot, nodes} = model;
assert.equal(new Set(events.map(e => e.id)).size, events.length);
assert.equal(new Set(events.map(e => e.phase)).size, 12);
assert.equal((html.match(/class="life-step"/g) || []).length, 12);
for (const id of Object.keys(nodes)) assert(html.includes(`data-node="${id}"`), id);
for (let i=0; i<events.length; i++) {
  const event=events[i], before=snapshot(i-1), after=snapshot(i);
  for (const node of event.active) assert(nodes[node], `${event.id}: missing node ${node}`);
  for (const route of event.routes) assert(nodes[route.from] && nodes[route.to], event.id);
  for (let rank=0; rank<3; rank++) {
    const a=before.ranks[rank], b=after.ranks[rank];
    if (a.queue!==b.queue && b.queue==='waiting') assert(b.metadata && b.sender==='ready', event.id);
    if (a.queue!==b.queue && b.queue==='batch') assert.equal(a.queue,'waiting',event.id);
    if (!a.computed && b.computed) {
      assert.equal(b.queue,'batch',event.id);
      if (rank>0) assert(a.proxy && before.ranks[rank-1].computed,event.id);
    }
    if (!a.output && b.output) assert(before.ranks[2].sampled,event.id);
    if (!a.sent && b.sent) assert(b.computed && b.token && b.queue==='inflight' && b.holdsKv && b.metadata,event.id);
    if (!a.terminal && b.terminal) assert(a.sent,event.id);
    if (!a.permit && b.permit) assert(before.consensus && before.ranks.every(r=>r.terminal),event.id);
    if (!a.released && b.released) assert(a.permit && a.terminal && a.holdsKv && a.metadata,event.id);
    if (b.released) assert(b.queue==='done' && !b.holdsKv && !b.metadata && b.sender==='cleared',event.id);
  }
  if (!before.consensus && after.consensus) assert(after.ranks.every(r=>r.terminal),event.id);
}
const index=id=>events.findIndex(e=>e.id===id);
assert(index('kv-0') < index('output-12'), 'KV0 must not wait for every stage to process output');
const waiting=snapshot(index('terminal-1-wait'));
assert(!waiting.consensus && !waiting.ranks[1].terminal);
assert(waiting.ranks.every(r=>r.holdsKv && r.metadata && !r.released), 'A partial terminal set must retain all request resources');
assert(snapshot(events.length-1).ranks.every(r=>r.released));
const first=snapshot(0);
snapshot(events.length-1).ranks[0].queue='corrupted';
assert.deepEqual(snapshot(0),first,'Seeking backwards must not retain future state');
assert(snapshot(index('kv-0')).ranks[0].holdsKv,'A successful send submission is not permission to free KV');
for (const id of ['output-20','output-01','output-12']) {
  const route=events[index(id)].routes[0];
  assert.equal(route.kind,'output');
}
console.log(`PASS: ${events.length} events, ${Object.keys(nodes).length} modules, causal dependencies, partial consensus, ownership, reverse seeking and HTML bindings.`);

// Exercise playback with a minimal DOM substitute and a deterministic clock.
// This checks controller state/timers; it does not claim browser rendering QA.
const vm = require('node:vm');
class Element {
  constructor(){this.children=[];this.listeners={};this.attrs={};this.dataset={};this.hidden=false;this.value='1';const classes=new Set();this.classList={toggle(c,on){if(on??!classes.has(c))classes.add(c);else classes.delete(c);},add(c){classes.add(c);},remove(c){classes.delete(c);},contains(c){return classes.has(c);}};this.style={setProperty(){}};}
  setAttribute(k,v){this.attrs[k]=String(v);}
  removeAttribute(k){delete this.attrs[k];}
  addEventListener(k,fn){this.listeners[k]=fn;}
  append(...elements){this.children.push(...elements);}
  replaceChildren(...elements){this.children=[...elements];}
  get firstElementChild(){return this.children[0];}
  scrollIntoView(){}
  focus(){}
  querySelector(){return new Element();}
  fire(type='click',event={}){this.listeners[type]?.({preventDefault(){},...event});}
}
const ids=Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>[m[1],new Element()]));
ids['player-scenario'].value='single';
const articleIds=[...html.matchAll(/class="life-step" id="([^"]+)"/g)].map(m=>m[1]);
const articles=articleIds.map(id=>Object.assign(ids[id],{id}));
const links=articles.map(()=>new Element());
const drawnNodes=Object.keys(nodes).map(id=>Object.assign(new Element(),{dataset:{node:id}}));
const singletonSelectors=Object.fromEntries(['.player-toolbar','.life-walkthrough','.life-controls','.module-inspector'].map(s=>[s,new Element()]));
ids['pipeline-svg'].querySelectorAll=()=>drawnNodes;
const svgBindings=new Map(drawnNodes.map(n=>[`[data-node="${n.dataset.node}"]`,n]));
ids['pipeline-svg'].querySelector=s=>{if(!svgBindings.has(s))svgBindings.set(s,new Element());return svgBindings.get(s);};
const documentEvents={},windowEvents={},timers=new Map();let timerId=0;
const location={hash:''};
const document={hidden:false,getElementById:id=>ids[id],createElement:()=>new Element(),createElementNS:()=>new Element(),querySelector:s=>singletonSelectors[s],querySelectorAll:s=>s==='.life-step'?articles:links,addEventListener:(k,fn)=>{documentEvents[k]=fn;}};
const context={document,window:{PrefillLifecycle:model,addEventListener:(k,fn)=>{windowEvents[k]=fn;}},location,history:{replaceState:(_a,_b,hash)=>{location.hash=hash;}},matchMedia:()=>({matches:false,addEventListener(){}}),setTimeout:fn=>{timers.set(++timerId,fn);return timerId;},clearTimeout:id=>timers.delete(id)};
vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../pages/sglang/pd-prefill-lifecycle/player.js'),'utf8'),context);
const tick=()=>{const [id,fn]=[...timers][0];timers.delete(id);fn();};
const seek=id=>ids['player-seek'].fire('input',{target:{value:String(index(id))}});
assert.equal(ids['player-counter'].textContent,'01 / 44');
assert.equal(ids['event-list'].children.length,44);
ids['player-play'].fire();assert.equal(timers.size,1);tick();
assert.equal(ids['player-counter'].textContent,'02 / 44');
ids['player-play'].fire();assert.equal(timers.size,0);
seek('kv-0');assert.equal(ids['event-title'].textContent,events[index('kv-0')].title);assert.equal(timers.size,0);
seek('handoff');assert.equal(ids['player-play'].textContent,'↺ 重播');assert(ids['player-next'].disabled);
ids['player-prev'].fire();assert.equal(ids['player-play'].textContent,'▶ 播放');
seek('handoff');ids['player-play'].fire();assert.equal(ids['player-counter'].textContent,'01 / 44');assert.equal(timers.size,1);
document.hidden=true;documentEvents.visibilitychange();assert.equal(timers.size,0);
document.hidden=false;ids['player-play'].fire();
ids['player-speed'].value='2';ids['player-speed'].fire('change');assert.equal(timers.size,1,'Speed changes must not multiply timers');
ids['player-reset'].fire();assert.equal(timers.size,0);assert.equal(ids['player-counter'].textContent,'01 / 44');
location.hash='#event-release-1';windowEvents.hashchange();assert.equal(ids['event-title'].textContent,events[index('release-1')].title);
location.hash='#step-pp1';windowEvents.hashchange();assert.equal(ids['event-title'].textContent,events.find(e=>e.phase===5).title);
ids['player-play'].fire();drawnNodes.find(n=>n.dataset.node==='p1-transfer').fire();assert.equal(timers.size,0);assert(singletonSelectors['.module-inspector'].open);
assert(ids['module-description'].textContent.includes('PP1'));
seek('finish-2');ids['player-play'].fire();tick();assert.equal(timers.size,0);assert.equal(ids['player-play'].textContent,'↺ 重播');
console.log('PASS: playback, pause, seek, reset, replay, speed, hidden-page pause, deep links, module inspection and end-of-sequence stop (DOM substitute).');

// The factory's stored blocks must track ownership when seeking in both directions.
const tank=svgBindings.get('[data-node="p0-transfer"]');
seek('batch-0');assert(tank.classList.contains('has-reservation'));assert(!tank.classList.contains('has-kv'));
seek('kv-0');assert(tank.classList.contains('has-kv'));assert(!tank.classList.contains('is-released'));
assert(!svgBindings.get('[data-received="0"]').classList.contains('is-received'));
seek('poll-0');assert(svgBindings.get('[data-received="0"]').classList.contains('is-received'));assert(tank.classList.contains('has-kv'));
seek('finish-0');assert(tank.classList.contains('is-released'));assert(!tank.classList.contains('has-kv'));
seek('intake');assert(!tank.classList.contains('has-kv'));assert(!tank.classList.contains('is-released'));assert(!svgBindings.get('[data-received="0"]').classList.contains('is-received'));
for(const event of events){
  seek(event.id);
  for(const el of ids['pipeline-routes'].children.filter(el=>el.attrs.class==='active-route')){
    assert(!/NaN|undefined/.test(el.attrs.d),event.id);
    const coords=[...el.attrs.d.matchAll(/[MHV]([\d.]+)(?: ([\d.]+))?/g)];
    for(const [,a,b] of coords){assert(Number(a)>=0&&Number(a)<=1680,event.id);if(b!==undefined)assert(Number(b)>=0&&Number(b)<=1040,event.id);}
  }
}
const finalPath=ids['pipeline-routes'].children.find(el=>el.attrs.class==='active-route').attrs.d;
assert.equal(finalPath,'M1522.5 845 V930','Decode continuation exits through the receiving depot bottom');
console.log('PASS: factory storage/receipt states, rewind, all route coordinates and Decode exit port.');

const chunked=model.scenario('chunked');
assert.equal(chunked.events.length,75);
assert.equal(new Set(chunked.events.map(e=>e.id)).size,75);
for(let i=0;i<chunked.events.length;i++){
  const e=chunked.events[i],before=chunked.snapshot(i-1),after=chunked.snapshot(i);
  if(e.middle){assert(after.ranks.every(r=>!r.token&&!r.released));assert(!after.ranks[2].sampled);}
  for(let r=0;r<3;r++){
    assert(after.ranks[r].sendEnd<=after.ranks[r].cacheEnd,e.id);
    assert(after.ranks[r].cacheEnd<=after.ranks[r].selectedEnd,e.id);
    if(e.id.endsWith(`forward-${r}`)&&r>0)assert(before.ranks[r-1].cacheEnd>=e.chunk.end,e.id);
    if(e.id.endsWith(`kv-${r}`))assert(before.ranks[r].cacheEnd>=e.chunk.end,e.id);
  }
}
for(const n of [1,2,3]){
  const i=chunked.events.findIndex(e=>e.id===`chunk-${n}-cut`),e=chunked.events[i];
  assert.equal(e.chunk.start,(n-1)*4);assert.equal(e.chunk.end,n*4);
  assert(chunked.snapshot(i).ranks.every(r=>r.cacheEnd===(n-1)*4));
}
assert.deepEqual(chunked.snapshot(-1).ranks.map(r=>r.cacheEnd),[0,0,0]);
ids['player-scenario'].fire('change',{target:{value:'chunked'}});
assert.equal(ids['event-list'].children.length,75);assert.equal(ids['player-counter'].textContent,'01 / 75');
location.hash='#chunked-event-chunk-2-cut';windowEvents.hashchange();assert(ids['event-title'].textContent.includes('[4, 8)'));
console.log('PASS: all three chunks, cumulative KV, partial sends, no early t0/release, scenario switching and chunk deep links.');
