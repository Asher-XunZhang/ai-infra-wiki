/* Independent teaching arithmetic; durations are illustrative, not benchmark results. */
(function(root){
'use strict';
const revision='279339f113b79af84f27fd3ac92d0a13bd3f4cbd';
const bench='python/sglang/benchmark/serving.py';
const sources={adapter:[bench,658,'async def async_request_sglang_generate'],start:[bench,701,'st = time.perf_counter()'],chunks:[bench,756,'num_new_tokens = output_len - last_output_len'],metrics:[bench,1096,'def calculate_metrics'],success:[bench,1124,'if outputs[i].success:'],tpot:[bench,1136,'tpots.append'],throughput:[bench,1236,'request_throughput='],permit:[bench,1389,'async def limited_request_func'],window:[bench,1499,'benchmark_start_time ='],end:[bench,1613,'benchmark_duration ='],server:['python/sglang/srt/managers/tokenizer_manager.py',2942,'def collect_metrics']};
function timeline(kind='base'){
 const durations=[80,20,80,100,20,120,20];
 if(kind==='client')durations[0]+=120;
 if(kind==='queue')durations[2]+=120;
 if(kind==='compute')durations[3]/=2;
 let cursor=0;const labels=['客户端等许可','请求传入','Scheduler 排队','Prefill + 首 token','首包回传','3 轮 Decode / 回传','流结束尾部'];
 const segments=durations.map((duration,i)=>{const s={start:cursor,end:cursor+duration,duration,label:labels[i],kind:i===0?'outside':i===2?'queue':i===3||i===5?'compute':'transport'};cursor+=duration;return s;});
 const send=segments[1].start,first=segments[4].end,end=cursor;
 return {segments,send,first,end,ttft:first-send,e2e:end-send,tpot:(end-first)/3,total:end};
}
function packets(kind='token'){
 const single=kind==='single';
 const chunks=single?[{t:100,n:1}]:kind==='bundle'?[{t:100,n:1},{t:220,n:4}]:[100,140,180,220].map((t,i)=>({t,n:i+1}));
 const itls=[];for(let i=1;i<chunks.length;i++){const count=chunks[i].n-chunks[i-1].n;for(let j=0;j<count;j++)itls.push((chunks[i].t-chunks[i-1].t)/count);}
 const end=chunks.at(-1).t+20,n=chunks.at(-1).n;
 return {chunks,itls,end,n,ttft:100,tpot:n>1?(end-100)/(n-1):null};
}
function schedule(concurrency=1,prefillSpeed=1){
 if(![1,3].includes(concurrency)||![1,2].includes(prefillSpeed))throw Error('Unsupported teaching parameter');
 const first=(concurrency===1?60:90)/prefillSpeed,decode=concurrency===1?60:90,duration=first+decode;
 const groups=Array.from({length:6/concurrency},(_,i)=>({start:i*duration,end:(i+1)*duration,ids:Array.from({length:concurrency},(_,j)=>i*concurrency+j+1)}));
 const wall=groups.at(-1).end;
 return {groups,wall,first,e2e:duration,tpot:decode/3,throughput:24/(wall/1000),readyMean:groups.reduce((s,g)=>s+g.end,0)/groups.length};
}
function experiment(kind){return kind==='prefill'?{a:schedule(3,1),b:schedule(3,2)}:{a:schedule(1,1),b:schedule(3,1)};}
function percentile(xs,p){if(!xs.length)return null;const s=[...xs].sort((a,b)=>a-b),i=(s.length-1)*p,l=Math.floor(i);return s[l]+(s[Math.ceil(i)]-s[l])*(i-l);}
function sample(kind='all'){
 const requests=[80,80,80,500].map((ttft,i)=>({id:i+1,ttft,success:kind!=='failure'||i!==3,tokens:4}));
 const ok=requests.filter(r=>r.success),ttfts=ok.map(r=>r.ttft);
 return {requests,completed:ok.length,failed:4-ok.length,mean:ttfts.reduce((a,b)=>a+b,0)/ok.length,p95:percentile(ttfts,.95),throughput:ok.reduce((a,r)=>a+r.tokens,0)};
}
const api={revision,sources,timeline,packets,schedule,experiment,percentile,sample};
if(typeof module==='object'&&module.exports)module.exports=api;root.PerformanceModel=api;
})(typeof globalThis==='object'?globalThis:this);
