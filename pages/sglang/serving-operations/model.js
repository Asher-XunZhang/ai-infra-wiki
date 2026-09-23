(function(root){
'use strict';
const revision='279339f113b79af84f27fd3ac92d0a13bd3f4cbd',srt='python/sglang/srt/';
const sources={
 warmup:[srt+'entrypoints/http_server.py',2203,'_execute_server_warmup'],
 health:[srt+'entrypoints/http_server.py',664,'health_generate'],
 route:['sgl-model-gateway/src/policies/round_robin.rs',31,'select_worker'],
 eligible:['sgl-model-gateway/src/policies/mod.rs',136,'get_healthy_worker_indices'],
 queue:[srt+'managers/scheduler.py',3298,'_abort_on_queued_limit'],
 timeout:[srt+'managers/scheduler.py',3344,'_poll_timeout_aborts'],
 disconnect:[srt+'managers/tokenizer_manager.py',1755,'_stream_one_response'],
 abort:[srt+'managers/tokenizer_manager.py',2024,'abort_request'],
 dequeue:[srt+'managers/scheduler.py',5257,'abort_request'],
 echo:[srt+'managers/tokenizer_manager.py',3281,'_handle_abort_req'],
 signal:[srt+'managers/tokenizer_manager.py',3749,'sigterm_handler'],
 drain:[srt+'managers/tokenizer_manager.py',3212,'sigterm_watchdog'],
 shutdown:[srt+'managers/scheduler.py',5693,'handle_shutdown'],
 release:[srt+'managers/scheduler.py',1842,'release_host_resources'],
 process:[srt+'managers/scheduler.py',5830,'run_scheduler_process']
};
const url=k=>{const [file,line]=sources[k];return `https://github.com/sgl-project/sglang/blob/${revision}/${file}#L${line}`;};
function health({starting=false,exiting=false,generation=true,bypass=false,last=99,tic=100,now=101,timeout=20}={}){
 if(starting||exiting)return 503;
 if(bypass||!generation||last>tic)return 200;
 return now>=tic+timeout?503:null;
}
function route(condition,step){
 const eligible=condition==='none'?[]:condition==='both'?[0,1]:[0];
 const assigned=[[],[]],rejected=[];
 for(let i=0;i<step;i++)if(eligible.length)assigned[eligible[i%eligible.length]].push('R'+(i+1));else rejected.push('R'+(i+1));
 return {eligible,assigned,rejected,selected:step&&eligible.length?eligible[(step-1)%eligible.length]:null};
}
function queue(capacity,step){
 if(![1,2].includes(capacity))throw new Error('invalid capacity');
 const waiting=[],rejected=[],done=[];let running='R0';
 const events=['R1','R2','R3','finish','R4'];
 for(const e of events.slice(0,step)){
  if(e==='finish'){done.push(running);running=waiting.shift()||null;}
  else if(waiting.length<capacity)waiting.push(e);else rejected.push(e);
 }
 return {waiting,rejected,done,running};
}
function waitingTimedOut(entry,now,timeout){return timeout>0&&entry>0&&entry<now-timeout;}
function cancellation(kind,step){
 return {client:kind==='disconnect'&&step>=1?'断连':'连接中',queued:step<3,frontend:step<4,abort:step>=2,echo:step>=3,code:kind==='waiting'&&step===4?503:null};
}
function drain(kind,step){
 const exiting=step>=1,ingressStopped=kind!=='arrivals'&&step>=2;
 const pending=step===0||step===1?['R1','R2']:step===2?['R2']:kind==='arrivals'?['R3']:[];
 const shutdownSent=exiting&&pending.length===0;
 return {exiting,ingressStopped,pending,shutdownSent,schedulerReleased:kind==='normal'&&step>=4,killFallback:kind==='stuck'&&step>=5,exitObserved:kind==='normal'&&step===5};
}
const api={revision,sources,url,health,route,queue,waitingTimedOut,cancellation,drain};if(typeof module!=='undefined')module.exports=api;else root.ServingModel=api;
})(typeof globalThis!=='undefined'?globalThis:this);
