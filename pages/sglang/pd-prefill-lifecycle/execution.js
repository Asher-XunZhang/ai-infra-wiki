(() => {
  const M=window.PrefillLifecycle,$=id=>document.getElementById(id),fields=['requests','inputLength','chunkSize','batchSize','pageSize','fault','waitRank','waitRid','detail'];
  let config=M.normalize(),draft=config.failed;
  const view=window.PDExecutionView.mount($('lifecycle-execution'),{kind:'lifecycle',onPosition:p=>{const u=new URL(location.href);for(const key of fields)u.searchParams.set(key,config[key]);u.searchParams.set('failed',JSON.stringify(config.failed));u.searchParams.set('t',p.t.toFixed(3));u.searchParams.set('rank',p.rank);u.searchParams.set('rid',p.rid);history.replaceState(null,'',u);}});
  function matrix(){const n=Math.max(1,Math.min(4,Number($('config-requests').value)||1)),fault=$('config-fault').value;
    $('failure-field').hidden=!['bootstrap_fail','bootstrap_abort','transfer_fail'].includes(fault);$('wait-field').hidden=!['bootstrap_wait','transfer_fail','repoll_wait'].includes(fault);
    $('failure-matrix').innerHTML=[0,1,2].map(r=>`<div><strong>PP${r}</strong>${Array.from({length:n},(_,i)=>`<label><input type="checkbox" data-rank="${r}" value="R${i}" ${draft[r].includes('R'+i)?'checked':''}>R${i}</label>`).join('')}</div>`).join('');
    const rid=$('config-waitRid').value;$('config-waitRid').innerHTML=Array.from({length:n},(_,i)=>`<option>R${i}</option>`).join('');if(Number(rid.slice(1))<n)$('config-waitRid').value=rid;
  }
  function failures(){return [0,1,2].map(r=>[...$('failure-matrix').querySelectorAll(`[data-rank="${r}"]:checked`)].map(e=>e.value));}
  function fill(){for(const key of fields)if(key!=='waitRid')$('config-'+key).value=config[key];draft=config.failed;matrix();$('config-waitRid').value=config.waitRid;}
  function load(raw,position){try{config=M.normalize(raw);const model=M.createScenario(config);view.load(model,position);$('scenario-summary').textContent=`${config.requests} 条请求 · ${model.batches} 个 batch · PP=3`;$('config-error').hidden=true;$('config-status').textContent='参数已应用，播放和拖动共用同一时钟。';return true;}catch(e){$('config-error').hidden=false;$('config-error').textContent=e.message;return false;}}
  $('config-preset').onchange=e=>{if(M.presets[e.target.value]){load(M.presets[e.target.value]);fill();view.seek(0);}};
  $('config-requests').onchange=()=>{draft=failures();matrix();};$('config-fault').onchange=matrix;
  $('failure-matrix').onchange=()=>draft=failures();
  $('scenario-form').onsubmit=e=>{e.preventDefault();if(load({...Object.fromEntries(fields.map(k=>[k,$('config-'+k).value])),failed:failures()})){$('config-preset').value='custom';view.seek(0);}};
  $('share-scenario').onclick=async()=>{try{await navigator.clipboard.writeText(location.href);$('config-status').textContent='已复制参数与时间位置。';}catch{$('config-status').textContent='可复制地址栏中的完整链接。';}};
  // Group compression belonged to the removed serial player; one clock now includes all chunks.
  $('config-detail').closest('label').hidden=true;
  const p=new URLSearchParams(location.search),raw={};for(const key of fields)if(p.has(key))raw[key]=p.get(key);
  try{if(p.has('failed'))raw.failed=JSON.parse(p.get('failed'));}catch{}
  if(!load(raw,{t:p.has('t')?Number(p.get('t')):undefined,event:location.hash.startsWith('#action=')?decodeURIComponent(location.hash.slice(8)):undefined,rank:Math.max(0,Math.min(2,Number(p.get('rank'))||0)),rid:p.get('rid')||'R0'}))load({});
  fill();$('config-preset').value=Object.keys(M.presets).find(k=>JSON.stringify(M.normalize(M.presets[k]))===JSON.stringify(config))??'custom';
})();
