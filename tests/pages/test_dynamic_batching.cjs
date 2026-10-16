const assert=require('node:assert/strict');
const {simulate,scenarios}=require('../../pages/motion/dynamic-batching.js');
const batches=delay=>simulate(delay).at(-1).batches;
assert.deepEqual(batches(5),[{ids:['A','B','C'],time:4,reason:'凑满',waits:[4,2,0]}]);
assert.deepEqual(batches(3),[{ids:['A','B'],time:3,reason:'等待到期',waits:[3,1]},{ids:['C'],time:7,reason:'等待到期',waits:[3]}]);
assert.deepEqual(batches(0).map(b=>[b.ids,b.time]),[[['A'],0],[['B'],2],[['C'],4]]);
// No additional waiting still permits multiple already available requests.
assert.deepEqual(simulate(0,[{id:'A',at:0},{id:'B',at:0}]).at(-1).batches[0].ids,['A','B']);
// Deadline equality collects arrivals before dispatch; arrival does not reset deadline.
assert.deepEqual(simulate(3,[{id:'A',at:0},{id:'B',at:3}]).at(-1).batches[0].ids,['A','B']);
for(const delay of [0,1,3,5,8]){
 const arrivals=Array.from({length:9},(_,i)=>({id:String(i),at:Math.floor(i/2)}));
 const frames=simulate(delay,arrivals,3),sent=frames.at(-1).batches;
 assert.deepEqual(sent.flatMap(b=>b.ids),arrivals.map(r=>r.id),'FIFO, exactly once, no missing requests');
 for(const b of sent){assert.ok(b.ids.length<=3);assert.ok(b.waits.every(w=>w>=0&&w<=delay));}
 for(const f of frames){assert.equal(f.queue.length+f.batches.reduce((n,b)=>n+b.ids.length,0),f.arrived.length,'conservation');if(f.queue.length)assert.equal(f.deadline,arrivals.find(r=>r.id===f.queue[0]).at+delay);}
}
assert.deepEqual(simulate(2).map(f=>f.queue),simulate(2).map(f=>f.queue));
console.log('PASS dynamic batching: capacity, deadline, zero delay, FIFO, conservation; '+scenarios.map(s=>s.frames.length).join('/')+' frames');
