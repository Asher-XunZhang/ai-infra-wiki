/* Teaching models, not a transport or performance simulator. */
(function(root){
'use strict';
const revision='279339f113b79af84f27fd3ac92d0a13bd3f4cbd';
const sources={
 ipc:['managers/tokenizer_manager.py',601,'_dispatch_to_scheduler'],
 send:['distributed/parallel_state.py',1744,'send_tensor_dict'],
 receive:['distributed/parallel_state.py',1799,'recv_tensor_dict'],
 reduce:['distributed/parallel_state.py',655,'all_reduce'],
 gather:['distributed/parallel_state.py',1326,'all_gather'],
 exchange:['distributed/parallel_state.py',1171,'all_to_all_single'],
 register:['disaggregation/mooncake/conn.py',325,'register_buffer_to_engine'],
 metadata:['disaggregation/mooncake/conn.py',2706,'send_metadata'],
 queue:['disaggregation/mooncake/conn.py',2385,'add_transfer_request'],
 blocks:['disaggregation/common/utils.py',111,'group_concurrent_contiguous'],
 success:['disaggregation/common/conn.py',569,'apply_prefill_status'],
 gate:['disaggregation/utils.py',212,'_apply_metadata_gate'],
 commit:['disaggregation/decode.py',2158,'_commit_transfer_to_req'],
 ready:['disaggregation/decode.py',2364,'pop_transferred'],
 release:['disaggregation/decode.py',2525,'resolve_deferred_releases']
};
const url=key=>{const [file,line]=sources[key];return `https://github.com/sgl-project/sglang/blob/${revision}/python/sglang/srt/${file}#L${line}`;};
function collective(kind){
 if(kind==='reduce')return {input:[[1,2],[10,20]],output:[[11,22],[11,22]]};
 if(kind==='gather')return {input:[[1,2],[10,20]],output:[[1,2,10,20],[1,2,10,20]]};
 if(kind==='exchange')return {input:[['A0','B0'],['A1','B1']],output:[['A0','A1'],['B0','B1']]};
 throw new Error('unknown collective');
}
function blocks(src,dst){
 if(!src.length||!dst.length)return [];
 if(src.length!==dst.length)throw new Error('page counts differ');
 const out=[];
 src.forEach((p,i)=>{if(!i||p!==src[i-1]+1||dst[i]!==dst[i-1]+1)out.push({src:p,dst:dst[i],pages:1});else out.at(-1).pages++;});
 return out;
}
function pages(fragmented=false){
 const src=[7,8,12],dst=fragmented?[3,9,4]:[3,4,9];
 return {src,dst,bytesPerPage:16,blocks:blocks(src,dst),payload:['A','B','C']};
}
function readiness({sources:done=[],expected=2,metadata=0,room=42,peers=true,restored=true}){
 const kv=new Set(done).size>=expected;
 const gate=kv&&metadata!==0&&peers&&restored;
 return {kv,gate,ready:gate&&metadata===room,corrupt:gate&&metadata!==room};
}
function release({acks=[],expected=2,expired=false}){
 const drained=new Set(acks).size>=expected;
 return {held:!drained&&!expired,drained,timedOut:expired&&!drained};
}
function gates(kind,step){
 const failed=['abort','timeout'].includes(kind);
 if(failed){
  const acks=step>=3&&kind==='abort'?[0,1]:step>=2?[0]:[];
  return {...release({acks,expired:step===3&&kind==='timeout'}),acks,failed,step};
 }
 const stages=kind==='cache'?6:5, final=step===stages;
 const data={sources:step>=2?[0,1]:step>=1?[0]:[],metadata:step>=3?(kind==='corrupt'?99:42):0,peers:step>=4,restored:kind!=='cache'||step>=5};
 // Last frame commits the request after all gates, earlier frames only establish prerequisites.
 return {...data,...readiness(data),ready:final&&readiness(data).ready,failed:false,step};
}
const api={revision,sources,url,collective,blocks,pages,readiness,release,gates};
if(typeof module!=='undefined')module.exports=api;else root.CommunicationModel=api;
})(typeof globalThis!=='undefined'?globalThis:this);
