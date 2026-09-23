const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const M=require('../pages/sglang/communication/model.js');
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
