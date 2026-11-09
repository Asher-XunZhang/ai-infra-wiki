const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const M=require('../../pages/sglang/communication/model.js');
assert.deepEqual(M.collective('reduce').output,[[11,22],[11,22]]);
assert.deepEqual(M.collective('gather').output,[[1,2,10,20],[1,2,10,20]]);
const exchange=M.collective('exchange');
assert.deepEqual(exchange.input.flat().sort(),exchange.output.flat().sort(),'exchange conserves elements');
exchange.output.forEach((xs,i)=>xs.forEach(x=>assert.equal(x[0],i===0?'A':'B')));
assert.throws(()=>M.collective('bad'));
assert.deepEqual(M.blocks([],[]),[]);assert.throws(()=>M.blocks([1],[1,2]));
for(const fragmented of [false,true]){
 const d=M.pages(fragmented);assert.equal(d.blocks.length,fragmented?3:2);
 const pairs=d.blocks.flatMap(b=>Array.from({length:b.pages},(_,i)=>[b.src+i,b.dst+i]));
 assert.deepEqual(pairs,d.src.map((v,i)=>[v,d.dst[i]]));
 assert.equal(d.blocks.reduce((n,b)=>n+b.pages*d.bytesPerPage,0),48);
}
assert.equal(M.blocks([3,5],[7,8]).length,2,'contiguous destination alone cannot merge');
assert.equal(M.blocks([3,4],[8,7]).length,2,'adjacent but reversed cannot merge');
for(const sources of [[],[0],[0,0],[0,1]])for(const metadata of [0,42,99])for(const peers of [false,true])for(const restored of [false,true]){
 const d=M.readiness({sources,metadata,peers,restored});
 assert.equal(d.ready,new Set(sources).size===2&&metadata===42&&peers&&restored);
 assert.equal(d.corrupt,new Set(sources).size===2&&metadata===99&&peers&&restored);
}
assert.deepEqual(M.release({acks:[0,0]}),{held:true,drained:false,timedOut:false});
assert.deepEqual(M.release({acks:[0,1]}),{held:false,drained:true,timedOut:false});
assert.deepEqual(M.release({acks:[0],expired:true}),{held:false,drained:false,timedOut:true});
for(const kind of ['normal','cache','corrupt']){
 const last=kind==='cache'?6:5;
 for(let i=0;i<last;i++)assert.equal(M.gates(kind,i).ready,false);
 assert.equal(M.gates(kind,last).ready,kind!=='corrupt');
}
if(process.env.SGLANG_SOURCE_DIR){for(const [key,[file,line,symbol]] of Object.entries(M.sources)){
 const code=execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',`${M.revision}:python/sglang/srt/${file}`],{encoding:'utf8'});
 assert.match(code.split('\n')[line-1],new RegExp(`def ${symbol}\\(`),key);
}console.log(`PASS: ${Object.keys(M.sources).length} fixed source anchors.`);}
console.log('PASS: collective semantics, page-copy coverage and byte conservation, dependency gates, duplicate ACKs, timeout versus drain.');

// Continuous implementation model: causal boundaries, reversibility and variants.
const F=require('../../pages/sglang/communication/flow-model.js');
for(const mode of Object.keys(F.catalog))for(const [option] of F.options[mode]){
 const tl=F.build(mode,option);assert.ok(tl.duration>10&&tl.duration<25);
 let end=0;for(const p of tl.phases){assert.equal(p.start,end);end=p.end;assert.ok(p.refs.length);for(const ref of p.refs)assert.ok(F.sources[ref]);}
 assert.equal(F.sample(tl,Number(tl.duration.toFixed(2))).finished,true,mode+': slider precision can reach finish');assert.equal(F.sample(tl,-1).time,0);assert.equal(F.sample(tl,Infinity).time,tl.duration);
 const snapshots=Array.from({length:101},(_,i)=>[tl.duration*i/100,F.state(tl,tl.duration*i/100)]);
 for(const [t,s]of snapshots.reverse())assert.deepEqual(F.state(tl,t),s);
 const before=id=>F.state(tl,tl.phases.find(p=>p.id===id).end-.001),after=id=>F.state(tl,tl.phases.find(p=>p.id===id).end);
 if(mode==='ipc'){assert.equal(after('mark').reqConstructed,false);assert.equal(before('req').reqConstructed,false);assert.equal(after('req').reqConstructed,true);}
 if(mode==='tensor'){assert.equal(before('residual').tensorReady,false);assert.equal(after('residual').tensorReady,true);}
 if(mode==='collective'){assert.equal(after('submit').collectiveReady,false);assert.equal(before('execute').collectiveReady,false);assert.equal(after('execute').collectiveReady,true);}
 if(mode==='kv'){assert.ok(tl.phases.find(p=>p.id==='metadata').end<=tl.phases.find(p=>p.id==='compute').start);assert.ok(tl.phases.find(p=>p.id==='compute').end<=tl.phases.find(p=>p.id==='queue').start);assert.equal(before('copy').kvWritten,false);assert.equal(after('copy').kvWritten,true);assert.equal(after('copy').auxWritten,false);assert.equal(after('aux').auxWritten,true);assert.equal(after('aux').successNotified,false);assert.equal(before('notify').successNotified,false);assert.equal(after('notify').successNotified,true);}
 if(mode==='ready'){
  assert.equal(after('duplicate').successCount,1);assert.equal(after('second').successCount,2);assert.equal(after('second').metadata,0);assert.equal(after('metadata').peersReady,false);
  if(option==='cache'){assert.equal(after('metadata').restored,false);assert.equal(after('cache').peersReady,false);assert.equal(after('peers').restored,true);}
  assert.equal(before('commit').committed,false);assert.equal(after('commit').committed,option!=='corrupt');assert.equal(after('commit').rejected,option==='corrupt');
 }
 if(mode==='release'){assert.equal(after('ack').released,false);assert.equal(before('free').released,false);assert.equal(after('free').released,true);assert.equal(after('free').timedOut,option==='timeout');assert.equal(after('free').ackCount,option==='timeout'?1:2);}
}
for(const fragmented of[false,true]){const plan=F.transferPlan(fragmented);assert.equal(plan.blocks.length,fragmented?3:2);assert.equal(plan.blocks.reduce((n,b)=>n+b.bytes,0),48);assert.deepEqual(plan.blocks.flatMap(b=>Array.from({length:b.count},(_,i)=>[b.src+i,b.dst+i])),plan.src.map((s,i)=>[s,plan.dst[i]]));}
assert.throws(()=>F.build('bad'));assert.throws(()=>F.build('ipc','bad'));
if(process.env.SGLANG_SOURCE_DIR){for(const [key,[file,line,symbol]] of Object.entries(F.sources)){
 const code=execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',`${F.revision}:python/sglang/srt/${file}`],{encoding:'utf8'});
 assert.ok(code.split('\n')[line-1].includes(symbol),key);
}console.log(`PASS: ${Object.keys(F.sources).length} implementation source anchors.`);}
console.log('PASS: 6 continuous timelines / 12 conditions, receiver causality, source deduplication, GPU stream dependency, reversible state, timeout versus drain.');

// Explicit hardware placement: tensor values never route through a CPU buffer in these direct-path examples.
for(const placement of ['local','rdma']){
 const gpu=F.hardware.route('gpu',placement),control=F.hardware.route('control',placement),aux=F.hardware.route('aux',placement);
 assert.equal(gpu.plane,'data');assert.equal(gpu.memory,'device');assert.ok(!gpu.nodes.some(x=>x.startsWith('cpu')));
 assert.equal(control.plane,'control');assert.equal(control.memory,'host');assert.equal(aux.plane,'data');assert.equal(aux.memory,'host');
 assert.equal(gpu.nodes.includes('switch'),placement==='rdma');assert.equal(control.nodes.includes('switch'),placement==='rdma');
 assert.deepEqual(F.hardware.route('gpu',placement,true).nodes,gpu.nodes.slice().reverse());
 for(const r of [gpu,control,aux])for(const point of [r.points[0],r.points.at(-1)]){const id=point===r.points[0]?r.nodes[0]:r.nodes.at(-1),n=F.hardware.nodes[id];assert.ok(point[0]>=n.x&&point[0]<=n.x+n.w&&point[1]>=n.y&&point[1]<=n.y+n.h,'route port on endpoint device');}
}
console.log('PASS: direct GPU / CPU control / host aux route separation and explicit placement.');
// Regressions against turning independent execution into narration-order serialization.
{
 const tl=F.build('tensor'),early=F.state(tl,3.7),overlap=F.state(tl,6.9),gpuPending=F.state(tl,8);
 assert.equal(early.progress.sendPosts,1);assert.ok(early.progress.meta<1,'sender returns before receiver finishes metadata');
 assert.equal(overlap.progress.rxResidual,1);assert.ok(overlap.progress.hidden>0&&overlap.progress.hidden<1,'CPU posts residual while GPU still receives hidden');
 assert.equal(gpuPending.progress.forwardSubmit,1);assert.equal(gpuPending.tensorReady,false,'CPU forward submission is not GPU input readiness');
 const ipc=F.state(F.build('ipc'),5.2);assert.ok(ipc.progress.mark>0&&ipc.progress.mark<1&&ipc.progress.receive>0&&ipc.progress.receive<1);
 const ready=F.state(F.build('ready','cache'),3);for(const id of ['first','second','metadata','cache'])assert.ok(ready.progress[id]>0&&ready.progress[id]<1,id+' prepares independently');assert.equal(ready.committed,false);
 const kv=F.build('kv'),t=kv.phases.find(x=>x.id==='copy').start+1;
 const active=kv.execution.lanes.flatMap(l=>l.events.filter(e=>t>=e.start&&t<e.end).map(e=>e.id));for(const id of ['returned','copy','sync','poll'])assert.ok(active.includes(id));assert.equal(F.state(kv,t).auxWritten,false);
 const release=F.state(F.build('release'),2);assert.ok(release.progress.drain0>0&&release.progress.drain1>0);assert.equal(release.released,false);
 for(const mode of Object.keys(F.catalog))for(const [option] of F.options[mode])for(const lane of F.build(mode,option).execution.lanes){let end=0;for(const e of lane.events){assert.ok(e.start>=end,mode+'/'+lane.id+' does not overlap one owner');end=e.end;assert.ok(e.end>e.start);}}
}
console.log('PASS: asynchronous CPU submission, independent preparation, worker-local blocking and ordered device consumption.');
