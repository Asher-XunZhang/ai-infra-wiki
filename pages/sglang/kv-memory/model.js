/* Selected path: ordinary RadixCache + paged allocator + static MHA KV pool. */
(function(root){
'use strict';
const revision='279339f113b79af84f27fd3ac92d0a13bd3f4cbd',base='python/sglang/srt/mem_cache/';
const sources={
 rows:[base+'memory_pool.py',259,'class ReqToTokenPool'],
 buffers:[base+'memory_pool.py',2255,'def _create_buffers_normal'],
 write:[base+'memory_pool.py',2526,'def set_kv_buffer'],
 extend:[base+'allocator/paged.py',183,'def alloc_extend'],
 capacity:[base+'allocator/paged.py',147,'def available_size'],
 free:[base+'allocator/paged.py',282,'def free_segment'],
 align:[base+'radix_cache.py',147,'def page_aligned'],
 keyMatch:[base+'radix_cache.py',178,'def match('],
 salt:[base+'radix_cache.py',245,'if self.cache_salt is not None'],
 match:[base+'radix_cache.py',397,'def match_prefix'],
 finish:[base+'radix_cache.py',479,'def cache_finished_req'],
 evict:[base+'radix_cache.py',638,'def evict'],
 lock:[base+'radix_cache.py',668,'def inc_lock_ref'],
 unlock:[base+'radix_cache.py',683,'def dec_lock_ref'],
 eligible:[base+'radix_cache.py',866,'def _update_leaf_status'],
 release:[base+'common.py',254,'def release_kv_cache']
};
const pageSize=4,initial=[4,5,6,7,28,29];
function extend(target=11,step=0){
 if(![8,9,11].includes(target))throw Error('Unsupported target');
 const added=Array.from({length:target-6},(_,i)=>i<2?30+i:12+i-2);
 const newPages=Math.ceil(target/pageSize)-Math.ceil(6/pageSize);
 const mapping=step>=1?[...initial,...added]:[...initial];
 const written=step>=3?target:step>=2?8:6;
 const focused=step===4?target-1:null;
 return {target,added,newPages,mapping,written,focused,ownedPages:2+(step>=1?newPages:0),pages:[1,7,3].map(id=>({id,free:id===3&&(step===0||newPages===0),cells:Array.from({length:4},(_,offset)=>{const slot=id*4+offset,p=mapping.indexOf(slot);return {slot,p,state:p<0?'empty':p<written?'written':'reserved',focus:p===focused};})}))};
}
function prefix(common=6,sameNamespace=true,step=0){
 if(![3,6,8].includes(common))throw Error('Unsupported common prefix');
 const cached='ABCDEFGH'.split(''),request=[...cached.map((t,i)=>i<common?t:'x'), 'Z'];
 const hit=sameNamespace?Math.floor(common/pageSize)*pageSize:0;
 const slots=[8,9,10,11,20,21,22,23].slice(0,hit);
 return {cached,request,common,hit,sameNamespace,reused:step>=3?slots:[],recompute:request.length-hit,newCopies:0};
}
function sharing(step=0){
 if(!Number.isInteger(step)||step<0||step>6)throw Error('Unsupported step');
 const r1=step>=1&&step<3,r2=step>=2&&step<5,locks=Number(r1)+Number(r2),cached=step<6;
 const pages=[{id:2,state:cached?(locks?'protected':'cached'):'free'},{id:3,state:r1?'private':'free'},{id:4,state:r2?'private':'free'}];
 return {r1,r2,locks,cached,evictable:cached&&locks===0,evictAttempt:step===4||step===6,pages,freePages:pages.filter(p=>p.state==='free').length,bufferPages:3};
}
const api={revision,sources,pageSize,extend,prefix,sharing};if(typeof module==='object'&&module.exports)module.exports=api;root.KVMemoryModel=api;
})(typeof globalThis==='object'?globalThis:this);
