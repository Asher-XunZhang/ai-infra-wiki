/* Selected ordinary text path; arithmetic examples are not a throughput simulator. */
(function(root){
'use strict';
const revision='279339f113b79af84f27fd3ac92d0a13bd3f4cbd',srt='python/sglang/srt/';
const sources={
 policy:[srt+'managers/schedule_policy.py',256,'def calc_priority'],
 lpm:[srt+'managers/schedule_policy.py',414,'def _sort_by_longest_prefix'],
 budget:[srt+'mem_cache/prefill_budget.py',98,'def check_prefill'],
 reserve:[srt+'mem_cache/prefill_budget.py',139,'def reserve'],
 input:[srt+'managers/schedule_policy.py',1267,'def _select_prefill_admission'],
 commit:[srt+'managers/schedule_policy.py',1347,'def _commit_prefill_admission'],
 stop:[srt+'managers/schedule_policy.py',763,'def budget_state'],
 pin:[srt+'managers/schedule_policy.py',1185,'with self._lock_node(req.last_node)'],
 chunk:[srt+'managers/schedule_policy.py',950,'def add_chunked_req'],
 chunkFirst:[srt+'managers/scheduler.py',3894,'adder.add_chunked_req'],
 rows:[srt+'managers/scheduler.py',3922,'if len(adder.can_run_list)'],
 check:[srt+'mem_cache/allocator/base.py',114,'def check_decode_capacity'],
 decode:[srt+'managers/schedule_batch.py',3166,'def new_tokens_required_next_decode'],
 retract:[srt+'managers/schedule_batch.py',3214,'def retract_decode'],
 order:[srt+'managers/schedule_batch.py',3301,'def _get_decode_retraction_order'],
 release:[srt+'managers/schedule_batch.py',2209,'def release_req'],
 reset:[srt+'managers/schedule_batch.py',1931,'def reset_for_retract'],
 queue:[srt+'managers/scheduler.py',3233,'def _add_request_to_queue'],
 full:[srt+'managers/scheduler.py',3298,'def _abort_on_queued_limit'],
 finish:[srt+'mem_cache/common.py',254,'def release_kv_cache']
};
const requests=[{id:'R1',input:6,hit:0,maxNew:2},{id:'R2',input:6,hit:4,maxNew:2},{id:'R3',input:2,hit:0,maxNew:2}];
const cases={compute:{input:6,kv:32,rows:3,resource:'input'},first:{input:3,kv:32,rows:3,resource:'input'},memory:{input:20,kv:12,rows:3,resource:'kv'},exact:{input:20,kv:9,rows:3,resource:'kv'},rows:{input:20,kv:32,rows:1,resource:'rows'},room:{input:20,kv:32,rows:3,resource:'input'}};
function admission(policy='fcfs',scenario='compute'){
 if(!['fcfs','lpm'].includes(policy)||!cases[scenario])throw Error('Unsupported admission scenario');
 const config=cases[scenario],order=[...requests],admitted=[],events=[];let usedInput=0,offset=0,protectedTokens=0;
 const emit=(kind,current=null,reason=null,trial=0,checkAvailable=null)=>events.push({kind,current,reason,order:order.map(r=>r.id),admitted:[...admitted],waiting:order.filter(r=>!admitted.includes(r.id)).map(r=>r.id),usedInput,offset,protectedTokens,trial,available:config.kv-offset-protectedTokens-trial,checkAvailable,config});
 emit('waiting');if(policy==='lpm')order.sort((a,b)=>b.hit-a.hit);emit('sorted');
 for(const r of order){
  if(admitted.length>=config.rows){emit('blocked',r.id,'rows');break;}
  const extend=r.input-r.hit,need=extend+r.maxNew+1,available=config.kv-offset-protectedTokens-r.hit;
  emit('check',r.id,null,r.hit,available);
  if(need>=available){emit('blocked',r.id,'kv',0,available);break;}
  if(admitted.length&&extend>=config.input-usedInput){emit('blocked',r.id,'input',0,available);break;}
  admitted.push(r.id);usedInput+=extend;offset+=need;protectedTokens+=r.hit;emit('commit',r.id);
  if(usedInput>=config.input){emit('stop',r.id,'input');break;}
 }
 emit('ready');return events;
}
function chunks(cap=4){
 if(![4,6,8].includes(cap))throw Error('Unsupported chunk cap');
 let left=10,shortLeft=2,round=0;const events=[],history=[];
 const emit=(kind,plan=null)=>events.push({kind,left,shortLeft,round,cap,plan,history:history.map(x=>({...x}))});
 emit('waiting');while(left||shortLeft){round++;const l=Math.min(left,cap),s=cap-l>=shortLeft?shortLeft:0,plan={round,l,s};emit('plan',plan);left-=l;shortLeft-=s;history.push(plan);emit('done',plan);}return events;
}
function retraction(scenario='resume',step=0){
 if(!['resume','full','cache'].includes(scenario))throw Error('Unsupported recovery scenario');
 const last=scenario==='cache'?3:scenario==='full'?5:7;if(step<0||step>last)throw Error('Unsupported recovery step');
 const capacity=scenario==='cache'?14:12;
 let r1=8,r2=4,cache=scenario==='cache'?2:0,location='running',output=['a'],row=true,recompute=0;
 if(scenario==='cache'){if(step>=1)cache=0;if(step===3){r1=9;r2=5;output.push('b');}}
 else {if(step>=2){r2=0;row=false;location='released';}if(step>=3)location=scenario==='full'?'aborted':'waiting';if(step>=4)r1=9;if(step>=5){cache=9;r1=0;}if(step>=6){cache=0;r2=5;row=true;location='prefill';recompute=5;}if(step===7){location='running';output.push('b');}}
 return {scenario,step,last,capacity,r1,r2,cache,free:capacity-r1-r2-cache,location,output,row,recompute,need:scenario==='cache'||step<2?2:step<5?1:0};
}
const api={revision,sources,requests,cases,admission,chunks,retraction};if(typeof module==='object'&&module.exports)module.exports=api;root.SchedulingModel=api;
})(typeof globalThis==='object'?globalThis:this);
