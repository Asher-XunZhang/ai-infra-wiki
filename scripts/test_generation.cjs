const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const M=require('../pages/sglang/advanced-generation/model.js');
const trace=[0,2,4,3,5.5];for(let i=0;i<5;i++){const d=M.state(i);assert.equal(d.value,trace[i]);assert.equal(d.kv.length,i);}
assert.equal(M.state(5).value,4);assert.equal(M.state(5).checkpoint,4);assert.deepEqual(M.state(5).kv,[2,3]);assert.notEqual(M.state(5).value,M.state(4).value);assert.throws(()=>M.state(-1));
for(let a=0;a<=4;a++){
 const d=M.speculate(a);assert.equal(d.accepted,a);assert.equal(d.output.length,a+1);assert.equal(d.validKV.length,a+1);
 assert.equal(d.output.at(-1),'z');assert.equal(d.validKV[0],'r');assert.ok(!d.validKV.includes('z'));
 assert.deepEqual(d.output.slice(0,-1),d.validKV.slice(1));assert.equal(d.uncommitted,4-a);
}
assert.deepEqual(M.verifyChain(['a','b','c','d'],['a','z','c','d','x']),{accepted:1,output:['a','z']},'cannot skip mismatch to retain later matches');
assert.deepEqual(M.verifyChain(['a'],['x','z']),{accepted:0,output:['x']});assert.throws(()=>M.verifyChain(['a'],['a']));
assert.deepEqual(M.vocab.filter(t=>M.allowed('',t)),['y','no']);assert.deepEqual(M.vocab.filter(t=>M.allowed('y',t)),['es']);assert.deepEqual(M.vocab.filter(t=>M.allowed('yes',t)),['EOS']);
for(const pref of ['yes','no']){
 const frames=M.grammar(pref);let built='';
 for(const f of frames){if(f.phase==='accept'){
  assert.equal(M.allowed(built,f.token),true);assert.notEqual(f.token,'!');
  if(f.token!=='EOS')built+=f.token;
  assert.equal(f.after,built);
  assert.ok(f.scores[M.vocab.indexOf(f.token)]>=Math.max(...f.scores.filter((v,i)=>f.valid[i])));
 }}assert.equal(built,pref);assert.equal(frames.at(-1).ended,true);
}
const first=M.grammar('yes');first[2].after='changed';assert.equal(M.grammar('yes')[0].prefix,'','separate teaching runs do not share progress');
if(process.env.SGLANG_SOURCE_DIR)for(const [key,[file,line,symbol,kind]] of Object.entries(M.sources)){
 const code=execFileSync('git',['-C',process.env.SGLANG_SOURCE_DIR,'show',`${M.revision}:python/sglang/srt/${file}`],{encoding:'utf8'});
 assert.match(code.split('\n')[line-1],new RegExp(`${kind||'def'} ${symbol}[( :]`),key);
}
console.log(`PASS: state/checkpoint consistency, greedy prefix rejection, output/KV offset, grammar language and logits; ${process.env.SGLANG_SOURCE_DIR?Object.keys(M.sources).length:0} fixed source anchors checked.`);
