/* Small deterministic teaching models; no model execution or performance prediction. */
(function(root){
'use strict';
const revision='279339f113b79af84f27fd3ac92d0a13bd3f4cbd';
const sources={
 kv:['mem_cache/memory_pool.py',2526,'set_kv_buffer'],
 state:['mem_cache/memory_pool.py',382,'MambaPool','class'],
 copy:['mem_cache/memory_pool.py',1020,'copy_from'],
 hybrid:['models/qwen3_5.py',1550,'Qwen3_5ForCausalLM','class'],
 stateCommit:['layers/attention/hybrid_linear_attn_backend.py',1340,'update_mamba_state_after_mtp_verify'],
 draft:['speculative/eagle_worker_v2.py',596,'draft'],
 build:['speculative/eagle_worker_common.py',316,'build_eagle_verify_input'],
 verify:['speculative/eagle_worker_common.py',461,'run_eagle_verify'],
 accept:['speculative/eagle_utils.py',724,'eagle_sample'],
 grammar:['constrained/xgrammar_backend.py',118,'fill_vocab_mask'],
 mask:['constrained/xgrammar_backend.py',125,'apply_vocab_mask'],
 advance:['constrained/xgrammar_backend.py',92,'accept_token'],
 grammarCopy:['constrained/xgrammar_backend.py',144,'copy'],
 sample:['layers/sampler.py',139,'forward']
};
const url=k=>{const [file,line]=sources[k];return `https://github.com/sgl-project/sglang/blob/${revision}/python/sglang/srt/${file}#L${line}`;};
const inputs=[2,3,1,4];
function state(step){
 if(!Number.isInteger(step)||step<0||step>5)throw new Error('invalid state frame');
 const history=inputs.reduce((a,x)=>[...a,a.at(-1)*.5+x],[0]);
 const n=step===5?2:step;
 return {n,history,value:history[n],checkpoint:step>=2?history[2]:null,kv:inputs.slice(0,n),restored:step===5};
}
function verifyChain(draft,predictions){
 if(predictions.length!==draft.length+1)throw new Error('one target row per root/candidate');
 let accepted=0;while(accepted<draft.length&&draft[accepted]===predictions[accepted])accepted++;
 return {accepted,output:[...draft.slice(0,accepted),predictions[accepted]]};
}
function speculate(accepted){
 if(!Number.isInteger(accepted)||accepted<0||accepted>4)throw new Error('invalid acceptance');
 const draft=['a','b','c','d'],root='r';
 const predictions=[...draft.slice(0,accepted),'z',...Array(4-accepted).fill('x')];
 const checked=verifyChain(draft,predictions);
 return {draft,root,verify:[root,...draft],predictions,accepted:checked.accepted,output:checked.output,validKV:[root,...draft.slice(0,accepted)],bonus:'z',uncommitted:4-accepted};
}
const vocab=['y','es','no','!','EOS'],words=['yes','no'];
function allowed(prefix,token){return token==='EOS'?words.includes(prefix):words.some(w=>w.startsWith(prefix+token));}
function grammar(prefer='yes'){
 if(!words.includes(prefer))throw new Error('invalid preference');
 const scores=[4,6,prefer==='no'?7:3,9,2],frames=[];let prefix='';
 while(true){
  const valid=vocab.map(t=>allowed(prefix,t));
  const chosen=vocab.reduce((best,t,i)=>valid[i]&&(best<0||scores[i]>scores[best])?i:best,-1);
  if(chosen<0)throw new Error('empty valid set');
  const token=vocab[chosen],after=token==='EOS'?prefix:prefix+token;
  frames.push({prefix,after:prefix,valid,scores,phase:'scores',token:null,ended:false});
  frames.push({prefix,after:prefix,valid,scores,phase:'mask',token:null,ended:false});
  frames.push({prefix,after,valid,scores,phase:'accept',token,ended:token==='EOS'});
  if(token==='EOS')break;prefix=after;
 }
 return frames;
}
const api={revision,sources,url,inputs,state,verifyChain,speculate,vocab,allowed,grammar};
if(typeof module!=='undefined')module.exports=api;else root.GenerationModel=api;
})(typeof globalThis!=='undefined'?globalThis:this);
