const assert=require('node:assert/strict');
const schedule=require('../../pages/sglang/scheduling/model.js');
global.PDQueues=require('../../pages/sglang/pd-dataflow/queue-model.js');
const scenes=require('../../pages/motion/scenes.js');
assert.equal(Object.keys(scenes).length,24);
const actor=(s,n,id)=>s.frames[n].actors[s.actors.findIndex(a=>a.id===id)];
for(const [key,variants]of Object.entries(scenes))for(const s of variants){
 assert.ok(s.frames.length>=3,key);assert.equal(new Set(s.actors.map(a=>a.id)).size,s.actors.length);
 assert.ok(s.frames.some((f,i)=>i&&f.actors.some((a,j)=>a.x!==s.frames[i-1].actors[j].x||a.y!==s.frames[i-1].actors[j].y)),key+' must actually move an identified object');
 for(const f of s.frames)for(const a of f.actors){assert.ok(a.x>=0&&a.x+204<=900&&a.y>=0&&a.y+66<=432,key);assert.ok(a.count>=0&&a.count<=9);}
}
for(const s of scenes['f-batching']){
 for(const f of s.frames){const live=f.actors.filter(a=>a.opacity===1&&a.x===348);assert.ok(live.length<=2);assert.equal(new Set(live.map(a=>a.y)).size,live.length,'one request per running slot');}
 for(const id of ['A','B','C']){const last=actor(s,s.frames.length-1,id);assert.equal(last.x,648);assert.equal(last.count,id==='B'?3:1);}
 const admission=s.frames.findIndex((f,i)=>actor(s,i,'C').x===348);
 const finishedB=s.frames.findIndex((f,i)=>actor(s,i,'B').x===648);
 assert.equal(admission<finishedB,s.name==='连续补位','continuous admission must differ from a fixed batch');
}
const graph=scenes['model-execution'][1];assert.equal(graph.frames.at(-1).actors.filter(a=>a.x===648).length,3);assert.equal(actor(graph,4,'pad').x,348);
const memory=scenes['kv-memory'][0];for(const id of ['page1','page7','page3'])for(let n=0;n<memory.frames.length;n++)assert.equal(actor(memory,n,id).x,648,'physical pages must not move on write');assert.equal(memory.frames.at(-1).actors.slice(0,3).reduce((n,a)=>n+a.count,0),11);
for(const [i,[policy,kind]]of [['lpm','compute'],['fcfs','exact']].entries()){
 const events=schedule.admission(policy,kind),s=scenes.scheduling[i];assert.equal(events.length,s.frames.length);
 events.forEach((e,n)=>assert.deepEqual(s.actors.filter((a,i)=>s.frames[n].actors[i].x===648).map(a=>a.id).sort(),e.admitted.slice().sort()));
}
for(const [i,kind]of ['normal','chunks'].entries()){
 const events=global.PDQueues.create(kind,0).frames,s=scenes.dataflow[i];events.forEach((e,n)=>{for(const id of ['bootstrap','waiting','batch','chunk','inflight','exit'])assert.equal(actor(s,n,id).opacity,Number(e.after[id]));});
}
const tokens=scenes['f-tokens'][0];assert.equal(actor(tokens,0,'sentence').opacity,1);for(let i=0;i<4;i++){assert.equal(actor(tokens,0,'piece-'+i).opacity,0);assert.equal(actor(tokens,2,'id-'+i).label,String([104100,100134,104455,1773][i]));}
console.log(`PASS: ${Object.keys(scenes).length} lesson scenes; identity, bounds, real movement, batching capacity/order, graph trim, KV addresses, policy admission and PD reference parity.`);
