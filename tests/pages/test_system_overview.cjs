const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process'),M=require('../../pages/sglang/system-overview/model.js'),A=require('../../pages/sglang/system-overview/architecture.json');
assert.equal(A.revision,M.revision);
assert.equal(M.topology('tp').replicas,1);assert.deepEqual(M.topology('tp').fractions,[.5,.5]);assert.deepEqual(M.topology('tp').assignments,[['R1'],['R1']]);assert.equal(M.topology('dp').replicas,2);assert.deepEqual(M.topology('dp').assignments,[['R1'],['R2']]);assert.deepEqual(M.topology('pd').fractions,[1,1]);assert.deepEqual(M.topology('pd').assignments,[['R1 · P'],['R1 · D']]);assert.throws(()=>M.topology('bad'));
assert.equal(new Set(A.nodes.map(n=>n.id)).size,A.nodes.length);
for(const e of A.edges){const [from,to]=e.id.split('-');for(const id of [from,to])assert(A.nodes.some(n=>n.id===id)||['gpu','client'].includes(id),e.id);assert(e.phase.every(p=>['input','compute','output'].includes(p)));}
let count=0;
if(process.env.SGLANG_SOURCE_DIR){
 const source=(path)=>execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',`${A.revision}:${path}`],{encoding:'utf8'});
 for(const [file,line,symbol] of [A.startup,...A.nodes.flatMap(n=>n.sources),...Object.values(M.sources)]){assert.ok(source(file).split('\n')[line-1].includes(symbol),`${file}:${line}: ${symbol}`);count++;}
 for(const path of A.extensions.flatMap(e=>e.paths))execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'cat-file','-e',`${A.revision}:${path.replace(/\/$/,'')}`]);
 const engine=source('python/sglang/srt/entrypoints/engine.py');assert(engine.includes('HTTP server, Engine, and TokenizerManager all run in the main process'));assert(engine.includes('Scheduler (subprocess)'));assert(engine.includes('DetokenizerManager (subprocess)'));
}
console.log(`PASS: architecture graph references, three deployment scopes; ${count} fixed source anchors and extension directories checked.`);
