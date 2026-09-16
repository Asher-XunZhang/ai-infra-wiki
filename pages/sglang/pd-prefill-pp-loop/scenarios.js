(() => {
  'use strict';
  const catalog=window.PP_SCENARIOS, picker=document.querySelector('#scenario-select');
  if(!picker||!catalog)return;
  const frame=document.querySelector('#timeline iframe'), base=new URL('.',document.currentScript.src);
  const engine=window.PP_TIMING_ENGINE, template=window.PP_TIMING_TEMPLATE;
  const workerURL=document.querySelector('#timing-engine').src;
  const cache=new Map(), baseExtent=Math.ceil(Math.max(...catalog.map(s=>s.summary.end))/10)*10;
  const scale=document.querySelector('#scenario-scale'), status=document.querySelector('#scenario-status');
  const params=new URLSearchParams(location.search);
  let active='baseline',packet=null,request=0,frameReady=false,extent=baseExtent;
  let worker=null,timer=null,seed=catalog[0],draft=null,savedTiming=null;
  const clone=value=>JSON.parse(JSON.stringify(value));
  const make=(tag,text,attrs={})=>Object.assign(document.createElement(tag),attrs,text==null?{}:{textContent:text});

  // A compact versioned URL keeps the same experiment across both views/reloads.
  function encode(t) {return JSON.stringify([1,t.gpu.flat(),t.h2d.flat(),t.kv.flat(),[...'ABCDEFGHI'].map(p=>t.cpu[p]),[t.proxy,t.output,t.control,t.d2h]]);}
  function decode(value) {
    const v=JSON.parse(value);
    if(!Array.isArray(v)||v.length!==6||v[0]!==1||!Array.isArray(v[1])||v[1].length!==15||!Array.isArray(v[2])||v[2].length!==6||!Array.isArray(v[3])||v[3].length!==15||!Array.isArray(v[4])||v[4].length!==9||!Array.isArray(v[5])||v[5].length!==4)throw new Error('自定义链接参数不完整或版本不支持。');
    const matrix=(a,width)=>[0,1,2].map(r=>a.slice(r*width,(r+1)*width));
    return engine.validate({gpu:matrix(v[1],5),h2d:matrix(v[2],2),kv:matrix(v[3],5),cpu:Object.fromEntries([...'ABCDEFGHI'].map((p,i)=>[p,v[4][i]])),...Object.fromEntries(['proxy','output','control','d2h'].map((p,i)=>[p,v[5][i]]))});
  }
  function setParams(url) {
    url.searchParams.set('scenario',active);
    scale.checked?url.searchParams.set('scale','all'):url.searchParams.delete('scale');
    if(active==='custom'&&savedTiming) {
      url.searchParams.set('timing',encode(savedTiming));url.searchParams.set('seed',seed.id);
    } else {url.searchParams.delete('timing');url.searchParams.delete('seed');}
  }
  function links() {
    document.querySelectorAll('a[href]').forEach(a=>{
      const url=new URL(a.href,location.href);
      if(url.origin===location.origin&&/\/pd-prefill-pp-loop\/(?:quick\.html|index\.html)?$/.test(url.pathname)){setParams(url);a.href=url.href;}
    });
  }
  function persist(){const url=new URL(location.href);setParams(url);history.replaceState(null,'',url);links();}
  function sendToFrame(){
    if(!frameReady)return;
    frame.contentWindow.postMessage({type:'pp-timing-scale',end:scale.checked?extent:0},'*');
    if(packet)frame.contentWindow.postMessage({type:'pp-timing-model',id:active,model:packet.model},'*');
  }
  function applyScale(save=true){
    document.querySelector('#scenario-scale-label').textContent='切换场景时统一全程时间轴（0–'+extent+' u），便于比较条块长度';
    window.dispatchEvent(new CustomEvent('pp-timing-scale',{detail:{end:scale.checked?extent:0}}));
    sendToFrame();if(save)persist();
  }
  function summary(scene){
    document.querySelector('#scenario-change').textContent=scene.change;
    document.querySelector('#scenario-observe').textContent='观察点：'+scene.observe;
    const metrics=document.querySelector('#scenario-metrics');metrics.replaceChildren();
    const values=[['GPU 全部完成',scene.summary.gpuEnd,'u'],['全部本地清理完成',scene.summary.releaseEnd,'u'],['每级展示的 loop',scene.summary.loops,'轮']];
    values.forEach(([name,value,unit],i)=>{
      const item=make('div'),label=make('span',name),number=make('strong',value+' '+unit),delta=make('small');
      const original=[catalog[0].summary.gpuEnd,catalog[0].summary.releaseEnd,catalog[0].summary.loops][i],difference=Number((value-original).toFixed(2));
      delta.textContent=scene.id==='baseline'?'原示例参考值':'较原示例 '+(difference>0?'+':'')+difference+' '+unit;
      item.append(label,number,delta);metrics.append(item);
    });
    const rows=document.querySelector('#scenario-parameters');rows.replaceChildren();
    scene.summary.gpu.forEach((values,r)=>{
      const row=make('tr'),heading=make('th','PP'+r,{scope:'row'});row.append(heading);
      values.forEach(value=>row.append(make('td',String(value))));rows.append(row);
    });
    document.body.dataset.scenario=scene.id;
    document.querySelectorAll('[data-original-scene-note]').forEach(e=>e.hidden=scene.id==='baseline');
  }
  function install(next,scene) {
    packet=next;active=scene.id;picker.value=active;
    extent=Math.max(baseExtent,Math.ceil(scene.summary.end/10)*10);
    summary(scene);applyScale(false);
    window.dispatchEvent(new CustomEvent('pp-timing-change',{detail:{id:active,packet}}));
    sendToFrame();persist();
  }
  function cancel(){clearTimeout(timer);worker?.terminate();worker=null;picker.removeAttribute('aria-busy');return ++request;}
  function state(message,error=false){status.textContent=message;status.classList.toggle('timing-error',error);}
  async function choose(id){
    const scene=catalog.find(s=>s.id===id)||catalog[0],token=cancel();
    editor.hidden=true;editButton.hidden=false;
    state('正在切换场景…');picker.setAttribute('aria-busy','true');
    try{
      if(!cache.has(scene.id)){
        const pending=fetch(new URL(scene.file,base)).then(response=>{if(!response.ok)throw new Error('场景资源加载失败');return response.json();});
        cache.set(scene.id,pending);pending.catch(()=>{if(cache.get(scene.id)===pending)cache.delete(scene.id);});
      }
      const next=await cache.get(scene.id);if(token!==request)return;
      seed=scene;install(next,scene);
      state('已切换：'+scene.name+'。两张图使用相同的时间模型。');
    }catch(error){if(token===request){picker.value=active;state('切换未完成，仍显示上次计算结果。请重试。',true);}}
    finally{if(token===request)picker.removeAttribute('aria-busy');}
  }

  const editButton=make('button','基于当前场景自定义',{type:'button',className:'timing-edit'});
  const editor=make('section',null,{className:'timing-editor',hidden:true});
  editor.setAttribute('aria-labelledby','timing-title');
  const heading=make('div',null,{className:'timing-heading'}), title=make('h3','自定义时间',{id:'timing-title'});
  const reset=make('button','恢复起始预设',{type:'button'}), seedLabel=make('span',null,{className:'muted'});
  heading.append(title,seedLabel,reset);editor.append(heading);
  editor.append(make('p','直接修改下表，停顿片刻即可更新下方图形。单位 u 为假设时间，输入范围 0–100；依赖等待、ACK 和释放轮次由模型重新计算。',{className:'muted'}));
  const fields=[];
  function field(parent,path,label,min=0,max=100){
    const input=make('input',null,{type:'number',min:String(min),max:String(max),step:'any',inputMode:'decimal'});
    input.setAttribute('aria-label',label);input.setAttribute('aria-describedby','timing-help');
    input.addEventListener('input',schedule);parent.append(input);fields.push({input,path});return input;
  }
  function table(parent,name,caption,columns){
    const wrapper=make('div',null,{className:'table-scroll'}),table=make('table',null,{className:'scenario-parameters timing-table'});
    const head=make('thead'),row=make('tr');row.append(make('th','流水级',{scope:'col'}));columns.forEach(m=>row.append(make('th','M'+m,{scope:'col'})));head.append(row);
    table.append(make('caption',caption),head);const body=make('tbody');
    for(let r=0;r<3;r++){
      const row=make('tr');row.append(make('th','PP'+r,{scope:'row'}));
      columns.forEach((m,i)=>{const cell=make('td');field(cell,[name,r,i],`PP${r} M${m} ${name==='gpu'?'GPU':name==='h2d'?'H2D 回载':'KV 传输'}时间（u）`);row.append(cell);});body.append(row);
    }
    table.append(body);wrapper.append(table);parent.append(wrapper);
  }
  table(editor,'gpu','GPU 前向服务时间（u）',[1,2,3,4,5]);
  const advanced=make('details',null,{className:'timing-advanced'});
  advanced.append(make('summary','更多阶段：回载、KV 传输、通信与 CPU 调度'));
  table(advanced,'h2d','L2 → HBM 回载时间（u）· 仅 M2、M4 有 host hit',[2,4]);
  table(advanced,'kv','本级 KV 传往 Decode 的服务时间（u）',[1,2,3,4,5]);
  function grid(parent,caption,entries,cpu=false){
    const group=make('fieldset'),legend=make('legend',caption),grid=make('div',null,{className:'timing-grid'});group.append(legend,grid);
    entries.forEach(([key,label])=>{const item=make('label',label);field(item,cpu?['cpu',key]:[key],label,cpu ? .1 : 0,cpu?10:100);grid.append(item);});parent.append(group);
  }
  grid(advanced,'通信与复制（u）· 每次对应操作共用此时间',[
    ['proxy','激活传输'],['output','output 回传'],['control','请求 / bootstrap / 终态候选传输'],['d2h','旧结果 D2H 复制']]);
  grid(advanced,'CPU 阶段服务时间倍率（0.1–10）· 相对原示例，1 = 原时长',[
    ['A','A · 请求与状态轮询'],['B','B · chunk / 缓存事件'],['C','C · 选批与回载准备'],['D','D · 激活接收 / 回收'],['E','E · 提交前向'],['F','F · output 收发'],['G','G · 控制共识'],['H','H · 结果处理与清理'],['I','I · 转发与保存']],true);
  advanced.append(make('p','CPU 倍率只缩放该阶段中各个操作的服务时长，不缩放依赖等待，也不直接缩放 GPU、回载、KV 或通信。控制传输参数只作用于请求、bootstrap 候选和终态候选消息；其余控制操作沿用模型依赖与 CPU 时间。',{className:'muted'}));
  editor.append(advanced,make('p','保留 PP=3、5 份 micro-batch、固定准入与选批顺序，以及“等全部 H2D 后前向”的模型粗化。这里用于比较依赖关系，不是 SGLang 性能预测。参数随链接保留，可刷新或切换两张图；下方图形始终对应最后一次成功计算。',{id:'timing-help',className:'muted'}));
  status.before(editButton,editor);

  function fill(timing){
    for(const {input,path} of fields){input.value=path.reduce((v,k)=>v[k],timing);input.removeAttribute('aria-invalid');}
    seedLabel.textContent='起始预设：'+seed.name;
  }
  function read(){
    const timing=clone(draft),bad=[];
    for(const {input,path} of fields){
      const value=input.valueAsNumber,valid=input.value!==''&&Number.isFinite(value)&&value>=Number(input.min)&&value<=Number(input.max);
      input.setAttribute('aria-invalid',String(!valid));
      if(!valid)bad.push(input);else path.slice(0,-1).reduce((v,k)=>v[k],timing)[path.at(-1)]=value;
    }
    if(bad.length)throw new Error(bad[0].getAttribute('aria-label')+'：请填写 '+bad[0].min+'–'+bad[0].max+' 内的数值。图仍显示上次计算结果。');
    return engine.validate(timing);
  }
  function schedule(){
    const token=cancel();let timing;
    try{timing=read();draft=timing;}catch(error){state(error.message,true);return;}
    state('正在重新计算依赖与等待…');picker.setAttribute('aria-busy','true');
    timer=setTimeout(()=>{
      try{
        worker=new Worker(workerURL);
        const fail=message=>{if(token!==request)return;worker?.terminate();worker=null;picker.removeAttribute('aria-busy');state(message+' 图仍显示上次计算结果。',true);};
        worker.onerror=()=>fail('自定义计算未完成，请重试或刷新页面。');
        worker.onmessage=event=>{
          if(token!==request||event.data.id!==token)return;
          if(event.data.error){fail(event.data.error);return;}
          worker.terminate();worker=null;picker.removeAttribute('aria-busy');savedTiming=clone(timing);
          const next=event.data.packet;
          install(next,{id:'custom',summary:next.summary,change:'自定义时间 · 基于「'+seed.name+'」修改。图形与指标已按当前参数重新计算。',observe:'比较 GPU 完成、KV 终态与实际清理之间的间隔；改变服务时间也可能改变等待原因和释放轮次。'});
          state('已实时更新：自定义时间。参数已保留在链接中，可切换到另一张图继续查看。');
        };
        worker.postMessage({id:token,template,timing});
      }catch(error){worker?.terminate();worker=null;picker.removeAttribute('aria-busy');state('无法启动自定义计算，请重试或刷新页面。',true);}
    },220);
  }
  function enter(timing){
    cancel();picker.value='custom';editor.hidden=false;editButton.hidden=true;
    draft=clone(timing||seed.timing);fill(draft);schedule();
  }
  editButton.addEventListener('click',()=>enter(active==='custom'&&savedTiming?savedTiming:seed.timing));
  reset.addEventListener('click',()=>enter(seed.timing));
  catalog.forEach(scene=>picker.append(make('option',scene.name,{value:scene.id})));
  picker.append(make('option','自定义时间',{value:'custom'}));
  picker.addEventListener('change',()=>picker.value==='custom'?enter(seed.timing):choose(picker.value));
  scale.checked=params.get('scale')==='all';scale.addEventListener('change',()=>applyScale());
  if(frame){
    window.addEventListener('message',event=>{if(event.source===frame.contentWindow&&event.data?.type==='pp-timing-ready'){frameReady=true;sendToFrame();}});
    frame.addEventListener('load',()=>{frameReady=true;sendToFrame();});
    frame.contentWindow?.postMessage({type:'pp-timing-ping'},'*');
  }
  window.addEventListener('pagehide',cancel);
  summary(catalog[0]);applyScale(false);
  const initial=params.get('scenario');
  if(initial==='custom'){
    try{seed=catalog.find(s=>s.id===params.get('seed'))||catalog[0];enter(decode(params.get('timing')));}
    catch(error){state('自定义链接无效：'+error.message+' 当前显示原示例。',true);persist();}
  } else if(initial&&initial!=='baseline')choose(initial);
  else persist();
})();
