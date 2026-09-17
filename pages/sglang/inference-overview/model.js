/* Teaching arithmetic, not a runtime simulator or a performance predictor. */
(function(root){
  'use strict';
  const integer=(n,min,max)=>Math.max(min,Math.min(max,Math.trunc(Number(n)||0)));
  function generation(prompt,step){
    prompt=integer(prompt,1,12);step=integer(step,0,5);
    return {prompt,step,kv:step===0?0:prompt+step-1,outputs:step,input:step<=1?prompt:1,phase:step===0?'等待':step===1?'Prefill':'Decode'};
  }
  function cache(prompt,reuse,decode){
    prompt=integer(prompt,2,12);reuse=integer(reuse,0,prompt-1);decode=integer(decode,0,4);
    return {prompt,reuse,decode,computed:prompt-reuse,kv:prompt+decode,outputs:1+decode};
  }
  function handoff(split,step,kv,metadata,slot){
    step=integer(step,0,4);
    return {split,step,ready:!split||Boolean(kv&&metadata&&slot),blocked:split&&step>=3&&!(kv&&metadata&&slot)};
  }
  function schedule(chunk,round){
    chunk=integer(chunk,1,8);round=integer(round,0,Math.ceil(8/chunk));
    const rounds=Array.from({length:Math.ceil(8/chunk)},(_,i)=>({prefill:Math.min(chunk,8-i*chunk),decode:1}));
    return {chunk,round:Math.min(round,rounds.length),rounds,processed:Math.min(round*chunk,8),aTokens:Math.min(round,rounds.length)};
  }
  function latency(queue,prefill,handoff,decode){
    const values=[queue,prefill,handoff,decode].map(v=>integer(v,0,200));
    const [q,p,h,d]=values;
    return {values,ttft:q+p+h,total:q+p+h+3*d,tpot:d};
  }
  const api={generation,cache,handoff,schedule,latency};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  else root.InferenceOverview=api;
})(typeof window!=='undefined'?window:globalThis);
