const assert=require('node:assert/strict');
const M=require('../../pages/sglang/pp-consensus/model.js');
let frames=0;
for(const [mode,info]of Object.entries(M.modes))for(const condition of Object.keys(info.cases)){
 const tl=M.build(mode,condition),byId=new Map(tl.events.map(e=>[e.id,e]));
 assert.equal(byId.size,tl.events.length);
 for(const e of tl.events){
  assert.ok(e.end>e.start&&e.start>=0,`${mode}/${condition}/${e.id} time`);
  assert.ok(M.sources[e.source]);
  for(const id of e.deps){assert.ok(byId.has(id),id);assert.ok(byId.get(id).end<=e.start+1e-7,`${mode}/${condition}/${e.id} before ${id}`);}
 }
 for(const lane of tl.lanes){const es=tl.events.filter(e=>e.lane===lane.id).sort((a,b)=>a.start-b.start);for(let i=1;i<es.length;i++)assert.ok(es[i-1].end<=es[i].start+1e-7,`${mode}/${condition}/${lane.id} overlap`);}
 const end=tl.sample(tl.duration);
 for(let t=0;t<=tl.duration;t+=.17){const a=JSON.stringify(tl.sample(t));tl.sample(tl.duration);assert.equal(a,JSON.stringify(tl.sample(t)));frames++;
  if(mode==='hicache')for(let r=0;r<3;r++)for(const ack of ['L0','L1','L2'])if(tl.sample(t).stages[r].acks[ack]==='finished')assert.ok(t>=byId.get(`copy${r}-${ack[1]}`).end);
 }
 if(mode==='bootstrap'){
  for(let r=0;r<3;r++)for(const id of M.ids){const expected=condition==='abort'&&id==='R2'?'failed':condition==='metadata'&&id==='R1'?'bootstrap':'waiting';assert.equal(end.stages[r].queue[id],expected,`${condition}/PP${r}/${id}`);}
  if(condition==='metadata'){assert.deepEqual(byId.get('q0-ring2').payload.good,M.ids);assert.deepEqual(byId.get('q0-ring0').payload.good,['R0','R2']);assert.deepEqual(byId.get('q0-ring1').payload.good,['R0','R2']);}
  if(condition==='late'){assert.ok(!tl.sample(16.9).stages[2].returned.good.includes('R1'));assert.ok(end.stages[2].returned.good.includes('R1'));}
 }
 if(mode==='release'){
  for(let r=0;r<3;r++)for(const id of M.ids)assert.equal(end.stages[r].queue[id],condition==='repoll'&&r===1&&id==='R0'?'inflight':'done');
  if(condition==='failure'){assert.ok(byId.get('q0-ring2').payload.includes('R1'));assert.match(end.stages[1].outcome.R1,/失败/);assert.match(end.stages[0].outcome.R1,/本级成功/);}
  if(condition==='slow')for(const r of tl.sample(17).stages)assert.ok(Object.values(r.queue).every(q=>q==='inflight'));
 }
 if(mode==='hicache'){
  const count={normal:2,prefix:1,zero:0}[condition];for(const r of end.stages){assert.equal(Object.values(r.acks).filter(x=>x==='finished').length,count+1);assert.equal(r.acks.L2,'queued');assert.deepEqual(r.counts,[1,count,0,0]);}
  if(condition==='normal'){const wait=byId.get('wait1-1'),go=byId.get('continue2');assert.ok(go.end<wait.end,'PP2 must continue while PP1 still waits');assert.ok(byId.get('count-send1').end<wait.start);assert.equal(tl.sample((go.end+wait.end)/2).stages[1].acks.L1,'popped');}
 }
}
assert.deepEqual(M.bootstrapMerge({good:['R0','R1'],bad:['R2']},{good:['R0','R1'],bad:[]},['R1']),{good:['R0'],bad:['R1','R2']});
assert.equal(M.readyPrefix([true,false,true]),1,'must not count a later ready item across a gap');
assert.equal(M.readyPrefix([false,true]),0);
console.log(`PASS: 11 scenarios; dependency order, resource exclusivity, bootstrap return filtering, terminal vs success, ACK prefix/event gates and ${frames} reversible snapshots.`);
