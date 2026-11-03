const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process'),M=require('../../pages/sglang/model-execution/model.js');
for(const n of [1,3,4,5])for(const condition of ['ready','off','override'])for(let step=0;step<=4;step++){
 const d=M.graph(n,condition,step);assert.equal(d.real.length,n);assert.equal(d.eligible,condition==='ready'&&n<=4);assert.deepEqual(d.slots.filter(x=>x!==null),d.real);assert.equal(d.slots.length,step>=2&&d.eligible?d.bucket:n);assert.deepEqual(d.outputs,step>=4?d.real:[]);assert.ok(d.outputs.every(x=>x!==null));
}assert.deepEqual(M.graph(3,'ready',2).slots,['R1','R2','R3',null]);assert.equal(M.graph(5).path,'eager');assert.throws(()=>M.graph(0));
for(const m of [1,16,256]){
 const d=M.matrix(m),math=M.matrix(m,'compute'),memory=M.matrix(m,'memory');assert.equal(d.flops,2*m*4096*4096);assert.equal(d.bytes,2*(2*m*4096+4096*4096));assert.equal(math.memoryMs,d.memoryMs);assert.equal(memory.mathMs,d.mathMs);assert.equal(math.mathMs,d.mathMs/2);assert.equal(memory.memoryMs,d.memoryMs/2);assert.equal(d.weightBytes,M.matrix(1).weightBytes);assert.ok(d.lowerMs>=d.mathMs&&d.lowerMs>=d.memoryMs);
}assert.equal(M.matrix(1).limit,'memory');assert.equal(M.matrix(256).limit,'math');assert.equal(M.matrix(1,'compute').lowerMs,M.matrix(1).lowerMs);assert.equal(M.matrix(256,'memory').lowerMs,M.matrix(256).lowerMs);assert.throws(()=>M.matrix(0));assert.equal(M.trace.at(-1).to,'Scheduler');assert.equal(M.trace[5].data,'kernel');assert.equal(M.trace[6].data,'logits');
if(process.env.SGLANG_SOURCE_DIR)for(const [key,[file,line,symbol]] of Object.entries(M.sources)){const code=execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',`${M.revision}:${file}`],{encoding:'utf8'});assert.ok(code.split('\n')[line-1].includes(symbol),`${key}: line ${line}: ${code.split('\n')[line-1]}`);}
console.log(`PASS: graph eligibility, real-request conservation, padding isolation, matrix arithmetic and single-resource limits; ${process.env.SGLANG_SOURCE_DIR?Object.keys(M.sources).length:0} source anchors.`);
const F=require('../../pages/sglang/model-execution/flow-model.js');
const configs=[...['prefill','decode'].map(mode=>['model',{mode}]),...['prefill','decode'].map(origin=>['chain',{origin}]),...[1,3,4,5].flatMap(count=>['ready','off','override'].map(condition=>['graph',{count,condition}])),...[1,16,256].flatMap(rows=>['base','compute','memory'].map(upgrade=>['hardware',{rows,upgrade}]))];
for(const [view,opts]of configs){
 const t=F.build(view,opts);assert.equal(F.at(t,0).state.delivered,false);assert.equal(F.at(t,t.duration).state.delivered,true);
 for(const e of t.events){assert.ok(e.end>e.start);if(e.edge)assert.ok(t.geometry.edges[e.edge]);for(const n of e.nodes)assert.ok(t.geometry.nodes[n]);}
 for(const p of t.phases)for(const frac of [0,.3,.7,.99]){
  const v=F.at(t,p.start+p.duration*frac),s=v.state;
  assert.deepEqual(F.at(t,v.time),v,'state independent of play or reverse-seek history');
  if(view==='model'){
   assert.ok(!s.sampled||s.logits,'sampling requires logits');assert.ok(!s.logits||s.completed===3,'logits only after last stage');
   if(s.arrived>0)assert.ok(s.completed>=s.arrived-1,'receiver cannot advance before sender finishes');
   if(p.id.startsWith('pp-'))assert.ok(s.completed>=p.stage,'PP message must follow sender layer completion');
   for(let k=s.arrived+1;k<4;k++)assert.equal(s.kv[k],opts.mode==='decode'?4:0,'future stage has no new KV');
  }else if(view==='chain'){assert.ok(!s.bound||s.inputReady);assert.ok(!s.loaded||s.bound);assert.ok(!s.metadata||s.loaded);assert.ok(!s.launched||s.metadata);assert.ok(!s.projected||s.launched);assert.ok(!s.stored||s.projected);assert.ok(!s.stored||s.metadata);assert.ok(!s.kernelDone||s.stored);assert.ok(!s.sampled||s.logits);assert.ok(!s.delivered||s.sampled);}
  else if(view==='graph'){assert.ok(!s.raw||s.padded);assert.ok(!s.trimmed||s.raw);const d=M.graph(opts.count,opts.condition,4);assert.equal(s.eligible,d.eligible);}
 }
 // A condensed stage still owns real work: receive, update local history,
 // finish its layer range, then send. Check either side of every milestone.
 if(view==='model'||view==='chain')for(const m of t.milestones)for(const delta of [-1e-6,0,1e-6]){
  const s=F.at(t,m.time+delta).state;
  assert.ok(!s.sampled||s.logits,'compressed output path must not sample early');
  assert.ok(!s.delivered||s.sampled,'results must exist before delivery');
  if(view==='model'){
   for(let k=0;k<4;k++){
    if(k>s.arrived)assert.equal(s.kv[k],opts.mode==='decode'?4:0,'unreached stages retain their old history');
    if(k<=s.completed)assert.equal(s.kv[k],opts.mode==='decode'?5:4,'completed stages have their own new history');
   }
   assert.ok(s.arrived===0||s.completed>=s.arrived-1,'receiver waits for sender completion');
   assert.ok(!s.logits||s.completed===3);
  }
 }
 if(view==='model')for(const e of t.events.filter(e=>e.edge?.startsWith('pp'))){
  const sender=t.phases[e.phase].stage;
  assert.equal(F.at(t,e.end-1e-6).state.arrived,sender,'receiver cannot advance during transit');
  assert.equal(F.at(t,e.end).state.arrived,sender+1,'receiver advances on arrival');
 }
 for(const e of Object.values(t.geometry.edges))for(const [id,p]of [[e.from,e.points[0]],[e.to,e.points.at(-1)]]){
  const n=t.geometry.nodes[id];assert.ok(p[0]>=n.x-1e-6&&p[0]<=n.x+n.w+1e-6&&p[1]>=n.y-1e-6&&p[1]<=n.y+n.h+1e-6,'port lies on participant');
  assert.ok(p[0]===n.x||p[0]===n.x+n.w||p[1]===n.y||p[1]===n.y+n.h,'wire reaches boundary');
 }
}
if(process.env.SGLANG_SOURCE_DIR)for(const key of ['modelReturn','finalNorm','v3','resolve','eagerLoad','eagerExtend','returnCPU']){const [file,line,symbol]=F.sources[key],code=execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',F.revision+':'+file],{encoding:'utf8'});assert.ok(code.split('\n')[line-1].includes(symbol),key);}
console.log('PASS: 25 continuous configurations; PP causality, layer-local KV, logits-before-sampling, graph conservation and wire endpoints.');
