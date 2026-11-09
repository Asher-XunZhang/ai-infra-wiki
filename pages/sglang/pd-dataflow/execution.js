(() => {
  const M=window.PDDataflow,select=document.getElementById('scenario');
  select.innerHTML=M.scenarios.map(s=>`<option value="${s.id}">${s.label}</option>`).join('');
  const view=window.PDExecutionView.mount(document.getElementById('dataflow-execution'),{kind:'dataflow',onPosition:position=>{const u=new URL(location.href);u.searchParams.set('scenario',select.value);u.searchParams.set('t',position.t.toFixed(3));u.searchParams.set('rank',position.rank);u.searchParams.delete('step');history.replaceState(null,'',u);}});
  function load(id,position){const model=M.create(id);model.sources=M.sources;select.value=model.id;document.getElementById('scenario-description').textContent=M.scenarios.find(s=>s.id===model.id).description;view.load(model,position);}
  select.onchange=()=>{load(select.value);view.seek(0);};
  function restore(){const p=new URLSearchParams(location.search);load(p.get('scenario')||'chunks',{t:p.has('t')?Number(p.get('t')):undefined,event:p.get('step'),rank:Math.max(0,Math.min(2,Number(p.get('rank'))||0))});}
  window.addEventListener('popstate',restore);restore();
})();
