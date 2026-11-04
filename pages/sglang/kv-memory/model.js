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
// All state, progress and moving payloads are sampled in explanation time.
// Physical slot contents persist after release; ownership and validity do not.
sources.mapWrite=[base+'allocation.py',55,'def write_cache_indices'];
sources.rowFree=[base+'memory_pool.py',341,'def free'];
const copy=x=>JSON.parse(JSON.stringify(x));
function timeline(mode='extend',options={}) {
 if(!['extend','reuse','sharing'].includes(mode))throw Error('Unknown memory mechanism');
 const target=[8,9,11].includes(Number(options.target))?Number(options.target):11;
 const hit=[0,4,8].includes(Number(options.hit))?Number(options.hit):4;
 const pageIds=mode==='extend'?[1,3,7]:mode==='reuse'?[2,5]:[2,3,4];
 let state={mode,target,hit,pages:pageIds.map(id=>({id,owner:null,slots:Array.from({length:4},(_,i)=>({slot:id*4+i,value:null,valid:false,reserved:false}))})),rows:{},outLoc:[],lookup:null,result:null,freeRows:mode==='sharing'?2:1};
 const getPage=id=>state.pages.find(p=>p.id===id);
 const slot=id=>getPage(Math.floor(id/4)).slots[id%4];
 const row=(id,values)=>{state.rows[id]={mapping:values.slice(),released:false};};
 const write=(indices,labels)=>indices.forEach((s,i)=>Object.assign(slot(s),{value:labels[i],valid:true,reserved:false}));
 if(mode==='extend') {
  getPage(1).owner=getPage(7).owner='R1';row('R1',initial);state.freeRows=0;
  write(initial,initial.map((_,i)=>String(i)));
 }else if(mode==='reuse') {
  getPage(2).owner=getPage(5).owner='cache';write([8,9,10,11,20,21,22,23],Array.from({length:8},(_,i)=>String(i)));
  row('R2',[]);state.freeRows=0;
 }else {getPage(2).owner='cache';write([8,9,10,11],['0','1','2','3']);}
 const phases=[];let cursor=0;
 function phase(name,duration,source,build) {
  const p={index:phases.length,name,duration,start:cursor,end:cursor+duration,events:[],flows:[]};
  const event=(at,label,change,key=source)=>{if(change)change();p.events.push({at,label,source:key,state:copy(state)});};
  const flow=(label,from,to,start,end,kind='address')=>p.flows.push({label,from,to,start,end,kind});
  event(0,name);build?.(event,flow);phases.push(p);cursor=p.end;
 }
 if(mode==='extend') {
  const addresses=extend(target,1).added,newPage=target>8;
  phase('已有 6 个位置',1.2,'rows');
  phase('领取地址',2.5,'extend',(event,flow)=>{
   event(.1,newPage?'尾页剩 2 个位置，再领一整页':'尾页足够，本轮无需新页');
   flow(newPage?'领页 3':'沿用尾页','allocator','operation',.15,1.1);
   event(1.1,'out_cache_loc 已返回；这些地址尚无新 K/V',()=>{
    if(newPage)getPage(3).owner='R1';state.outLoc=addresses.slice();addresses.forEach(s=>slot(s).reserved=true);
   });
   flow('新槽号','operation','row-R1',1.25,2.1);
   event(2.1,'write_cache_indices 将槽号写入请求映射',()=>row('R1',[...initial,...addresses]),'mapWrite');
  });
  phase('按地址写 K/V',3.1,'write',(event,flow)=>{
   event(.15,'set_kv_buffer：本层 K/V 按 loc 散写到目标槽');
   addresses.forEach((s,i)=>flow('K'+(i+6)+' / V'+(i+6),'payload-'+s,'slot-'+s,.3,2.35,'data'));
   event(2.35,'本轮 K/V 已写入；尾页和新页保持原位',()=>write(addresses,addresses.map((_,i)=>String(i+6))));
  });
  const p=target-1,s=[...initial,...addresses][p];
  phase('沿映射读取',2.8,'rows',(event,flow)=>{
   event(.1,'先查逻辑位置 p'+p+' 对应的槽号',()=>{state.lookup={request:'R1',p,slot:s};});
   flow('槽号 '+s,'map-R1-'+p,'slot-'+s,.2,1.2);
   event(1.2,'槽 '+s+' 定位到第 ℓ 层的 K / V');
   flow('K'+p+' / V'+p,'slot-'+s,'operation',1.35,2.3,'data');
   event(2.3,'读取结果：Kℓ['+s+'] / Vℓ['+s+']',()=>{state.result='K'+p+' / V'+p;});
  });
 }else if(mode==='reuse') {
  const indices=[8,9,10,11,20,21,22,23].slice(0,hit);
  phase('接收前缀索引',1.2,'match');
  phase('写入映射前缀',2.7,'mapWrite',(event,flow)=>{
   event(.1,hit?'已命中 '+hit+' 个位置，只传递索引':'没有可复用索引；本轮全部需要新地址');
   if(hit)flow('索引 ['+indices.join(',')+']','operation','row-R2',.25,1.75);
   event(1.75,hit?'映射指向已有槽；缓存数值保持原位':'映射前缀为空，不能借用已有缓存',()=>row('R2',indices));
  });
  phase('核对复用结果',2.6,'rows',(event,flow)=>{
   if(hit){const p=hit-1,s=indices[p];
    event(.1,'同一槽号定位同一份历史 K/V',()=>{state.lookup={request:'R2',p,slot:s};});
    flow('槽号 '+s,'map-R2-'+p,'slot-'+s,.15,1.15);
    flow('已有 K/V','slot-'+s,'operation',1.25,2.1,'data');
    event(2.1,'只新增映射引用，历史 KV 的副本数不变',()=>{state.result='复用 '+hit+' 个已有位置';});
   }else event(.8,'0 命中：9 个位置均走新的分配与计算',()=>{state.result='无历史 KV 可复用';});
  });
 }else {
  const shared=[8,9,10,11];
  phase('缓存持有页 2',1.2,'buffers');
  for(const [id,pg,labels]of [['R1',3,['4','5']],['R2',4,['X','Y']]]) {
   phase(id+' 引用缓存',2.6,'mapWrite',(event,flow)=>{
    flow('共享槽号 + 私有槽号','operation','row-'+id,.15,1.25);
    event(1.25,id+' 建立映射；页 2 仍只保存一份 K/V',()=>{
     getPage(pg).owner=id;row(id,[...shared,pg*4,pg*4+1]);write([pg*4,pg*4+1],labels);state.freeRows--;
    });
   });
  }
  function finish(id,pg) {
   phase(id+' 结束',2.6,'finish',(event,flow)=>{
    event(.1,id+' 停止使用 KV，进入释放流程',()=>{state.rows[id].releasing=true;});
    flow('归还页 '+pg,'page-'+pg,'allocator',.15,1.15,'release');
    event(1.15,'未满私有尾页归还；共享缓存仍保留',()=>{
     getPage(pg).owner=null;getPage(pg).slots.forEach(s=>{s.valid=false;});

    });
    flow('归还请求行','row-'+id,'allocator',1.3,2.05,'release');
    event(2.05,id+' 的请求行也已归还',()=>{state.rows[id].released=true;state.rows[id].mapping=[];state.freeRows++;},'rowFree');
   });
  }
  finish('R1',3);
  phase('共享页暂不能归还',1.8,'eligible',(event,flow)=>{
   flow('仍被 R2 使用','row-R2','page-2',.15,1.15,'address');
   event(1.15,'缓存层未允许释放；分配器不回收页 2');
  });
  finish('R2',4);
  phase('缓存释放页 2',2.6,'evict',(event,flow)=>{
   event(.1,'请求均已结束；现在由缓存层触发驱逐');
   flow('释放槽号 [8…11]','operation','allocator',.15,1.1,'release');
   event(1.1,'页 2 回到可分配集合，旧字节不保证清零',()=>{
    getPage(2).owner=null;getPage(2).slots.forEach(s=>{s.valid=false;});
   });
  });
  phase('下一请求重用',3.4,'extend',(event,flow)=>{
   flow('重领页 2','allocator','row-R3',.15,.95);
   event(.95,'R3 取得槽 8；旧内容尚不是 R3 的有效 KV',()=>{
    getPage(2).owner='R3';row('R3',[8]);state.freeRows--;state.outLoc=[8];slot(8).reserved=true;
   });
   flow('KZ / VZ','payload-8','slot-8',1.15,2.55,'data');
   event(2.55,'R3 写入新 K/V；同一物理空间再次使用',()=>write([8],['Z']),'write');
  });
 }
 return {mode,target,hit,phases,duration:cursor};
}
function sample(timeline,time) {
 const t=Math.max(0,Math.min(timeline.duration,Number(time)||0));
 const phase=timeline.phases.find(p=>t<=p.end+1e-7)||timeline.phases.at(-1),local=t-phase.start;
 const event=phase.events.filter(e=>e.at<=local+1e-7).at(-1)||phase.events[0];
 const state=copy(event.state);state.freePages=state.pages.filter(p=>p.owner===null).map(p=>p.id);
 state.heldEmpty=state.pages.filter(p=>p.owner!==null).flatMap(p=>p.slots).filter(s=>!s.valid&&!s.reserved).length;
 return {time:t,phase,local,event,state,flows:phase.flows.filter(f=>local>=f.start&&local<f.end).map(f=>({...f,progress:(local-f.start)/(f.end-f.start)}))};
}

const api={revision,sources,pageSize,extend,prefix,sharing,timeline,sample};if(typeof module==='object'&&module.exports)module.exports=api;root.KVMemoryModel=api;
})(typeof globalThis==='object'?globalThis:this);
