/* V41: one Prefill request, eight positions, two independent axes.
 * The spatial choice changes Query ownership; the temporal choice changes the
 * round in which a position enters the forward pass. Pair counts are causal
 * pairs, not elapsed GPU time. This module keeps a small explicit state machine
 * so every visible KV cell is written only after its Q row has entered. */
(function(root){
'use strict';
const TOTAL=36;
const CAUSAL=Array.from({length:8},(_,i)=>i+1);
const SPATIAL={
  single:{label:'单卡',owners:[0,0,0,0,0,0,0,0],ranks:1},
  contiguous:{label:'连续切分',owners:[0,0,0,0,1,1,1,1],ranks:2},
  zigzag:{label:'前后配对',owners:[0,0,1,1,1,1,0,0],ranks:2}
};
const TEMPORAL={
  whole:{label:'整段 1 轮',groups:[[0,1,2,3,4,5,6,7]]},
  chunk4:{label:'chunk 4',groups:[[0,1,2,3],[4,5,6,7]]}
};
const TIMING={project:400,read:110,write:450,settle:250,dwell:600,enter:700,done:700};

function work(owners,positions){
  const byRank=[0,0];
  for(const index of positions)byRank[owners[index]]+=CAUSAL[index];
  return {byRank,total:byRank[0]+byRank[1]};
}

function roundsFor(ownerMode,chunkMode){
  const owners=(SPATIAL[ownerMode]||SPATIAL.zigzag).owners;
  return (TEMPORAL[chunkMode]||TEMPORAL.chunk4).groups.map((positions,index)=>{
    const counted=work(owners,positions);
    return {index,positions,byRank:counted.byRank,total:counted.total};
  });
}

function roundOf(index,chunkMode){
  const groups=(TEMPORAL[chunkMode]||TEMPORAL.chunk4).groups;
  for(let r=0;r<groups.length;r++)if(groups[r].includes(index))return r;
  return 0;
}

function stepsFor(chunkMode){
  const groups=(TEMPORAL[chunkMode]||TEMPORAL.chunk4).groups;
  const steps=[{kind:'plan',round:-1,position:-1,label:'计划'}];
  groups.forEach((positions,round)=>{
    steps.push({kind:'enter',round,position:-1,positions:[...positions],label:`第 ${round+1} 轮进入`});
    positions.forEach(position=>steps.push({kind:'compute',round,position,label:`计算 Q${position+1}`}));
  });
  steps.push({kind:'done',round:groups.length-1,position:-1,label:'完成'});
  return steps;
}

function timingFor(step){
  if(step.kind==='compute'){
    const i=step.position+1;
    const readEnd=TIMING.project+TIMING.read*i;
    const writeEnd=readEnd+TIMING.write;
    return {project:TIMING.project,readEnd,writeEnd,stable:writeEnd+TIMING.settle,dwell:TIMING.dwell};
  }
  if(step.kind==='enter')return {stable:TIMING.enter,dwell:650};
  if(step.kind==='done')return {stable:TIMING.done,dwell:0};
  return {stable:0,dwell:TIMING.dwell};
}

function snapshot(ownerMode='zigzag',chunkMode='chunk4',stepIndex=0,elapsed=0){
  const spatial=SPATIAL[ownerMode]||SPATIAL.zigzag;
  const temporal=TEMPORAL[chunkMode]||TEMPORAL.chunk4;
  const steps=stepsFor(chunkMode);
  const index=Math.max(0,Math.min(steps.length-1,Number(stepIndex)||0));
  const step=steps[index];
  const timing=timingFor(step);
  const completed=[];
  for(let s=0;s<index;s++)if(steps[s].kind==='compute')completed.push(steps[s].position);
  let enteredRounds=0;
  for(let s=0;s<=index;s++)if(steps[s].kind==='enter')enteredRounds=Math.max(enteredRounds,steps[s].round+1);
  if(step.kind==='done')enteredRounds=temporal.groups.length;
  let active=null,phase=step.kind,readCount=0;
  if(step.kind==='compute'){
    active=step.position;
    const i=active+1;
    if(elapsed>=timing.stable){phase='settled';readCount=i;}
    else if(elapsed>=timing.readEnd){phase='write';readCount=i;}
    else if(elapsed>=timing.project){
      phase='read';
      readCount=Math.max(1,Math.min(i,Math.ceil((elapsed-timing.project)/TIMING.read)));
    }else phase='project';
    if(phase==='settled')completed.push(active);
  }
  const completedSet=new Set(completed);
  const completedByRound=temporal.groups.map((positions,round)=>{
    const counted=work(spatial.owners,positions.filter(position=>completedSet.has(position)));
    return {index:round,positions,byRank:counted.byRank,total:counted.total};
  });
  const completedTotal=completedByRound.reduce((sum,round)=>sum+round.total,0);
  const readPositions=[];
  if(step.kind==='compute'){
    if(phase==='read')for(let p=0;p<readCount;p++)readPositions.push(p);
    if(phase==='write'||phase==='settled')for(let p=0;p<=active;p++)readPositions.push(p);
  }
  const currentRound=active!==null?roundOf(active,chunkMode):(step.kind==='enter'?step.round:-1);
  return {
    ownerMode,chunkMode,ownerLabel:spatial.label,chunkLabel:temporal.label,
    owners:[...spatial.owners],ranks:spatial.ranks,rounds:roundsFor(ownerMode,chunkMode),all:work(spatial.owners,[0,1,2,3,4,5,6,7]),
    total:36,steps,stepIndex:index,step,timing,enteredRounds,completed,completedSet,
    completedByRound,completedTotal,active,phase,readCount,readPositions,currentRound
  };
}

function detail(index,ownerMode,chunkMode,s){
  if(!s){const steps=stepsFor(chunkMode);const at=steps.findIndex(step=>step.kind==='compute'&&step.position===index);s=snapshot(ownerMode,chunkMode,at<0?0:at,999999);}
  const spatial=SPATIAL[ownerMode]||SPATIAL.zigzag;
  const round=roundOf(index,chunkMode);
  const owner=spatial.owners[index];
  const pairs=CAUSAL[index];
  const causal=pairs===1?'K/V1':`K/V1–${pairs}`;
  const done=s.completedSet.has(index);
  const active=s.active===index;
  let status='planned',note='';
  if(done){
    status='done';
    if(s.chunkMode==='chunk4'&&round===1){
      note=`已完成：读取 K/V1–${pairs}，${pairs} 对；K${pairs}/V${pairs} 已写入。K/V1–4 来自第 1 轮历史，仍保留。`;
    }else if(s.chunkMode==='chunk4'){
      note=`已完成：读取 K/V1–${pairs}，${pairs} 对；K${pairs}/V${pairs} 已写入并保留，供第 2 轮读取。`;
    }else{
      note=`已完成：读取 K/V1–${pairs}，${pairs} 对；K${pairs}/V${pairs} 已写入。整段 1 轮没有跨轮历史。`;
    }
  }else if(active){
    status='active';
    if(s.phase==='project'){
      note=`正在计算本步的 K${pairs}/V${pairs}（暂存）；随后只读 K/V1–${pairs}。未来位置保持遮罩。`;
    }else if(s.phase==='read'){
      const history=(s.chunkMode==='chunk4'&&round===1)?'其中 K/V1–4 来自第 1 轮保留的历史；':'';
      note=`正在读取 K/V1–${s.readCount}（目标 1–${pairs}）。${history}写入完成前，完成计数不增加。`;
    }else if(s.phase==='write'){
      note=`已读完 K/V1–${pairs}（${pairs} 对）；正在写入 K${pairs}/V${pairs}，写入完成后才计入已完成。`;
    }else{
      note=`Q${pairs} 已完成：读取 K/V1–${pairs}，${pairs} 对；K${pairs}/V${pairs} 已写入。`;
    }
  }else if(s.enteredRounds>round){
    status='entered';
    note=`已进入第 ${round+1} 轮前向，等待计算；K/V${pairs} 尚未写入。`;
  }else{
    status='planned';
    note=`计划进入第 ${round+1} 轮，由 GPU ${owner} 计算；尚未进入前向，未计算，K/V${pairs} 未写入。`;
  }
  return {index,q:`Q${index+1}`,p:`p${index+1}`,round,owner,pairs,causal,status,note};
}

/* Rebase the time axis without losing the Query being compared. The same Q row
 * keeps its semantic phase (project/read/write/settled); only the round labels
 * and the completed-count attribution are rebuilt from the new condition. */
function remapStep(ownerMode,fromChunk,toChunk,stepIndex,elapsed){
  const before=snapshot(ownerMode,fromChunk,stepIndex,elapsed);
  const steps=stepsFor(toChunk);
  const mapped={step:0,elapsed:0,before,note:''};
  if(before.step.kind==='plan')return mapped;
  if(before.step.kind==='done'){
    mapped.step=steps.length-1;
    mapped.elapsed=timingFor({kind:'done'}).stable;
    return mapped;
  }
  if(before.step.kind==='compute'){
    const at=steps.findIndex(step=>step.kind==='compute'&&step.position===before.step.position);
    mapped.step=at<0?0:at;
    mapped.elapsed=Number(elapsed)||0;
    return mapped;
  }
  const groups=(TEMPORAL[toChunk]||TEMPORAL.chunk4).groups;
  const round=Math.max(0,Math.min(before.enteredRounds,groups.length)-1);
  const at=steps.findIndex(step=>step.kind==='enter'&&step.round===round);
  mapped.step=at<0?0:at;
  mapped.elapsed=Number(elapsed)||0;
  const doneKey=step=>[...step.completedSet].sort((a,b)=>a-b).join(',');
  const beforeDone=doneKey(before);
  if(doneKey(snapshot(ownerMode,toChunk,mapped.step,mapped.elapsed))!==beforeDone){
    let fallback=-1;
    for(let i=mapped.step;i<steps.length;i++){
      if(steps[i].kind!=='compute')continue;
      if(doneKey(snapshot(ownerMode,toChunk,i,0))===beforeDone){fallback=i;break;}
    }
    if(fallback>=0){
      mapped.step=fallback;
      mapped.elapsed=0;
      const target=steps[fallback];
      mapped.note=`轮进入边界不唯一，按 Q${target.position+1} 的 K/V 暂存起步对齐。`;
      return mapped;
    }
  }
  const target=steps[mapped.step];
  if(target&&target.positions&&before.step.positions&&target.positions.join(',')!==before.step.positions.join(',')){
    mapped.note=`轮进入边界不唯一，按已进入的第 ${round+1} 轮对齐。`;
  }
  return mapped;
}

function init(){
  const root=document.querySelector('[data-parallel-space-time]');
  if(!root||root.dataset.ready==='true')return;
  const ownerButtons=[...root.querySelectorAll('[data-pst-owner]')];
  const chunkButtons=[...root.querySelectorAll('[data-pst-chunk]')];
  const pickButtons=[...root.querySelectorAll('[data-pst-pick]')];
  const ownerCells=[...root.querySelectorAll('[data-pst-owner-cell]')];
  const roundCells=[...root.querySelectorAll('[data-pst-round-cell]')];
  const stateCells=[...root.querySelectorAll('[data-pst-state-cell]')];
  const kvRoundLabels=[...root.querySelectorAll('[data-pst-kv-round]')];
  const kvCells=[...root.querySelectorAll('[data-pst-kv]')];
  const columns=Array.from({length:8},(_,i)=>[...root.querySelectorAll(`[data-col="${i+1}"]`)]);
  const round2Row=root.querySelector('[data-pst-round-row="2"]');
  const history=root.querySelector('[data-pst-history]');
  const status=root.querySelector('[data-pst-status]');
  const phase=root.querySelector('[data-pst-phase]');
  const live=root.querySelector('[data-pst-live]');
  const stepLabel=root.querySelector('[data-pst-step-label]');
  const playButton=root.querySelector('[data-pst-action="play"]');
  const prevButton=root.querySelector('[data-pst-action="prev"]');
  const nextButton=root.querySelector('[data-pst-action="next"]');
  const replayButton=root.querySelector('[data-pst-action="replay"]');
  const resetButton=root.querySelector('[data-pst-action="reset"]');
  const doneTotal=root.querySelector('[data-pst-done-total]');
  const doneBreakdown=root.querySelector('[data-pst-done-breakdown]');
  const ledgerNote=root.querySelector('.pst-ledger-note');
  const ledgerFill=root.querySelector('[data-pst-ledger-fill]');
  const detailQ=root.querySelector('[data-pst-detail-q]');
  const detailRound=root.querySelector('[data-pst-detail-round]');
  const detailOwner=root.querySelector('[data-pst-detail-owner]');
  const detailState=root.querySelector('[data-pst-detail-state]');
  const detailPairs=root.querySelector('[data-pst-detail-pairs]');
  const detailCausal=root.querySelector('[data-pst-detail-causal]');
  const detailNote=root.querySelector('[data-pst-detail-note]');
  const state={owner:'zigzag',chunk:'chunk4',selected:0,step:0,elapsed:0,playing:false,mode:'idle',timer:null,lastTs:0,notice:'',lastLive:'',lastPhaseKey:''};
  const rankText=(counts,ranks)=>counts.map((count,i)=>ranks===1&&i===1?'—':String(count)).join(' / ');
  const ledger=key=>root.querySelector(`[data-pst-ledger="${key}"]`);
  function put(key,value,empty=false){
    const cell=ledger(key);if(!cell)return;
    cell.textContent=empty?'—':String(value);
    cell.classList.toggle('is-empty',empty);
  }
  function timing(){const s=snapshot(state.owner,state.chunk,state.step,state.elapsed);return s.timing;}
  function stopTimer(){
    if(state.timer){clearInterval(state.timer);state.timer=null;}
    state.playing=false;
  }
  function phaseText(s){
    const i=s.active===null?0:s.active+1;
    const step=s.step;
    let text='';
    if(state.notice)text=state.notice+' ';
    if(step.kind==='plan'){
      text+=`计划：p1…p8 尚未进入前向，K/V1–8 尚未写入。第 1 轮 ${s.chunkMode==='chunk4'?'p1–p4':'p1–p8'}；每个 Qi 只读 K/V1–i。`;
    }else if(step.kind==='enter'){
      const positions=step.positions.map(p=>`p${p+1}`).join('、');
      const later=s.chunkMode==='chunk4'&&step.round===0?'p5–p8 仍等待第 2 轮。':'';
      text+=`第 ${step.round+1} 轮进入：${positions} 已进入前向；${later}本轮尚未计算，K/V 尚未写入。`;
    }else if(step.kind==='compute'){
      if(s.phase==='project')text+=`计算 Q${i}：先算本步的 K${i}/V${i}（暂存，尚未写入）；随后只读 K/V1–${i}，未来位置保持遮罩。`;
      else if(s.phase==='read'){
        const hist=(s.chunkMode==='chunk4'&&step.round===1)?'其中 K/V1–4 来自上轮保留的历史；':'';
        text+=`计算 Q${i}：正在读取 K/V1–${s.readCount}（目标 1–${i}）。${hist}已读 ${s.readCount}/${i} 对；写入完成前，完成计数不增加。`;
      }
      else if(s.phase==='write')text+=`Q${i} 已读完 K/V1–${i}（${i} 对）；正在写入 K${i}/V${i}，写入完成后才计入已完成。`;
      else text+=`Q${i} 完成：读取 K/V1–${i}，${i} 对；K${i}/V${i} 已写入并保留。已完成 ${s.completedTotal} / 36 对。`;
    }else{
      if(s.chunkMode==='chunk4'){
        const per=s.rounds.map(round=>`第 ${round.index+1} 轮 ${s.ranks===1?round.byRank[0].toString()+' / —':round.byRank.join(' / ')}`).join('、');
        text+=`全部完成：8 个 Q 行、K/V1–8 已写入；已完成 36 / 36 对。计划每轮 ${per}；全程是总配对，不代表每轮均衡，配对也不是耗时。`;
      }else text+='全部完成：整段 1 轮，8 个 Q 行、K/V1–8 已写入；没有跨轮历史；已完成 36 / 36 对。';
    }
    return text;
  }
  function conditionNotice(s,kind,boundary){
    const q=s.active!==null?`Q${s.active+1}`:'当前步骤';
    let phaseLabel='计划';
    if(s.active!==null){
      if(s.phase==='project')phaseLabel='投影中';
      else if(s.phase==='read')phaseLabel=`读取中（${s.readCount}/${s.active+1} 对）`;
      else if(s.phase==='write')phaseLabel='写入中';
      else phaseLabel='已完成本行';
    }else if(s.step.kind==='enter')phaseLabel=`第 ${s.step.round+1} 轮进入`;
    else if(s.step.kind==='done')phaseLabel='全部完成';
    const attribution=kind==='owner'?'按新 owner 重算分卡':'按新 chunk 重标轮次与上轮历史';
    const perRound=s.completedByRound.map(round=>`第 ${round.index+1} 轮 ${rankText(round.byRank,s.ranks)}`).join('，');
    const verb=boundary?'对齐到':(s.active!==null?'保持':'停在');
    return `已暂停；${q} ${verb}${phaseLabel}，${attribution}；已完成 ${s.completedTotal} / 36 对（${perRound}）。${boundary}`;
  }
  function renderKv(s){
    kvRoundLabels.forEach((label,index)=>{
      const round=roundOf(index,state.chunk)+1;
      label.textContent=`第 ${round} 轮`;
      label.dataset.round=String(round);
    });
    kvCells.forEach(cell=>{
      const index=Number(cell.dataset.pstPos);
      const round=roundOf(index,state.chunk);
      const written=s.completedSet.has(index);
      const active=s.active===index;
      const future=s.active!==null&&index>s.active;
      const entered=s.enteredRounds>round;
      let value='未写入',stateName='waiting';
      if(active){
        if(s.phase==='project'){value='计算中';stateName='projecting';}
        else if(s.phase==='read'){value=index<s.readCount?'已读':'暂存';stateName='staged';}
        else if(s.phase==='write'){value='写入中';stateName='writing';}
        else {value='已写入';stateName='written';}
      }else if(written){
        const history=s.currentRound>=0&&round<s.currentRound;
        value=history?'上轮保留':'已写入';
        stateName=history?'history':'written';
      }else if(future){
        value='未来遮罩';stateName='future';
      }else if(entered){
        value='未计算';stateName='empty';
      }
      cell.dataset.state=stateName;
      cell.classList.toggle('is-read',s.readPositions.includes(index));
      cell.classList.toggle('is-reading',s.phase==='read'&&index+1===s.readCount&&index<=s.active);
      cell.classList.toggle('is-mask-start',future&&(index===0||index-1===s.active));
      const small=cell.querySelector('small');if(small)small.textContent=value;
    });
  }
  function renderProcess(s){
    stateCells.forEach((cell,index)=>{
      const round=roundOf(index,state.chunk);
      const written=s.completedSet.has(index);
      const active=s.active===index;
      const entered=s.enteredRounds>round;
      let label='未进入',data='waiting';
      if(written){label='已完成';data='done';}
      else if(active){label=s.phase==='read'?`读取 ${s.readCount}/${index+1}`:s.phase==='write'?'写入中':'计算中';data='computing';}
      else if(entered){label='已进入·未计算';data='entered';}
      else if(round>0)label='等待第2轮';
      cell.textContent=label;
      cell.dataset.state=data;
    });
    columns.forEach((nodes,index)=>{
      const written=s.completedSet.has(index);
      const active=s.active===index;
      nodes.forEach(node=>{node.classList.toggle('is-done',written);node.classList.toggle('is-active',active);});
    });
    renderKv(s);
    phase.textContent=phaseText(s);
    const step=s.step;
    stepLabel.textContent=`步骤 ${s.stepIndex+1} / ${s.steps.length} · ${step.label}`;
    playButton.textContent=state.playing?'暂停':(s.stepIndex===s.steps.length-1?'重播':'播放');
    playButton.setAttribute('aria-pressed',String(state.playing));
    prevButton.disabled=s.stepIndex===0;
    nextButton.disabled=s.stepIndex===s.steps.length-1;
    doneTotal.textContent=String(s.completedTotal);
    const perRound=s.completedByRound.map(round=>`第 ${round.index+1} 轮 ${rankText(round.byRank,s.ranks)}（${round.total} 对）`).join(' · ');
    doneBreakdown.textContent=perRound;
    if(ledgerFill)ledgerFill.style.width=`${s.completedTotal/TOTAL*100}%`;
    const progress=root.querySelector('[data-pst-progress]');
    if(progress)progress.setAttribute('aria-valuenow',String(s.completedTotal));
    const d=detail(state.selected,state.owner,state.chunk,s);
    detailQ.textContent=d.q;
    detailRound.textContent=`第 ${d.round+1} 轮`;
    detailRound.dataset.round=String(d.round+1);
    detailOwner.textContent=`GPU ${d.owner}`;
    detailOwner.dataset.owner=String(d.owner);
    if(detailPairs)detailPairs.textContent=String(d.pairs);
    if(detailCausal)detailCausal.textContent=`因果范围：${d.causal}`;
    if(detailState){detailState.textContent=d.status==='done'?'已完成':d.status==='active'?'计算中':d.status==='entered'?'已进入':'计划';detailState.dataset.state=d.status;}
    detailNote.textContent=d.note;
    const phaseKey=`${s.stepIndex}:${s.phase}`;
    if(phaseKey!==state.lastPhaseKey){
      state.lastPhaseKey=phaseKey;
      const liveText=phaseText(s);
      if(live&&liveText!==state.lastLive){live.textContent=liveText;state.lastLive=liveText;}
    }
    const statusText=`${s.ownerLabel} + ${s.chunkLabel}：已完成 ${s.completedTotal} / ${TOTAL} 对；第 ${s.completedByRound.map(round=>`${round.index+1} 轮 ${rankText(round.byRank,s.ranks)} 对`).join('，')}。`;
    if(status.textContent!==statusText)status.textContent=statusText;
    root.dataset.owner=state.owner;root.dataset.chunk=state.chunk;root.dataset.step=String(s.stepIndex);root.dataset.phase=s.phase;
  }
  function renderPlan(){
    const s=snapshot(state.owner,state.chunk,state.step,state.elapsed);
    ownerButtons.forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.pstOwner===state.owner)));
    chunkButtons.forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.pstChunk===state.chunk)));
    pickButtons.forEach((button,index)=>{
      button.setAttribute('aria-pressed',String(index===state.selected));
      button.closest('th')?.classList.toggle('is-selected',index===state.selected);
    });
    for(let i=0;i<8;i++){
      const owner=s.owners[i],round=roundOf(i,state.chunk)+1;
      ownerCells[i].textContent=`GPU ${owner}`;ownerCells[i].dataset.owner=String(owner);
      roundCells[i].textContent=`第 ${round} 轮`;roundCells[i].dataset.round=String(round);
      columns[i].forEach(cell=>cell.classList.toggle('is-selected',i===state.selected));
    }
    const first=s.rounds[0],second=s.rounds[1];
    if(ledgerNote)ledgerNote.textContent=state.chunk==='chunk4'
      ?'表内是计划；已完成只统计已经读完并写入 K/V 的 Q 行。第 2 轮只新增 p5–p8，仍读上轮 K/V1–4；每个 Qi 只读 1..i。'
      :'表内是计划；已完成只统计已经读完并写入 K/V 的 Q 行。整段 1 轮，没有跨轮历史；每个 Qi 仍因果只读 1..i。';
    root.querySelector('[data-pst-round-name="1"]').textContent=s.rounds.length===1?'整段 · 第 1 轮':'第 1 轮';
    root.querySelector('[data-pst-round-range="1"]').textContent=s.rounds.length===1?'p1–p8':'p1–p4';
    put('r1-0',first.byRank[0]);put('r1-1',first.byRank[1],s.ranks===1);put('r1-total',first.total);
    if(second){
      root.querySelector('[data-pst-round-name="2"]').textContent='第 2 轮';
      root.querySelector('[data-pst-round-range="2"]').textContent='p5–p8';
      put('r2-0',second.byRank[0]);put('r2-1',second.byRank[1],s.ranks===1);put('r2-total',second.total);
    }
    round2Row.hidden=!second;
    history.hidden=!second;
    put('all-0',s.all.byRank[0]);put('all-1',s.all.byRank[1],s.ranks===1);put('all-total',s.total);
    renderProcess(s);
    root.dataset.ready='true';
  }
  function render(){renderPlan();}
  function clearNotice(){state.notice='';}
  function select(index,focus=false){
    state.selected=(index+8)%8;
    if(state.playing)stopTimer();
    render();
    if(focus)pickButtons[state.selected].focus();
  }
  function seek(step){
    stopTimer();
    state.step=Math.max(0,Math.min(stepsFor(state.chunk).length-1,step));
    state.elapsed=timing().stable;
    clearNotice();
    const s=snapshot(state.owner,state.chunk,state.step,state.elapsed);
    if(s.active!==null)state.selected=s.active;
    render();
  }
  function tick(){
    if(!state.playing)return;
    const now=Date.now();
    state.elapsed+=Math.min(80,now-state.lastTs);
    state.lastTs=now;
    const s=snapshot(state.owner,state.chunk,state.step,state.elapsed);
    const stopAt=state.mode==='step'?s.timing.stable:s.timing.stable+s.timing.dwell;
    if(state.elapsed>=stopAt){
      if(state.mode==='step'||s.stepIndex>=s.steps.length-1){
        state.elapsed=s.timing.stable;stopTimer();state.mode='idle';render();return;
      }
      state.step++;state.elapsed=0;
      const next=stepsFor(state.chunk)[state.step];
      if(next.kind==='compute')state.selected=next.position;
      state.lastTs=Date.now();render();return;
    }
    render();
  }
  function start(mode){
    clearNotice();
    if(state.playing)return;
    const steps=stepsFor(state.chunk);
    if(state.elapsed>=timing().stable){
      if(state.step>=steps.length-1){state.step=0;state.elapsed=0;}
      else {state.step++;state.elapsed=0;const next=steps[state.step];if(next.kind==='compute')state.selected=next.position;}
    }
    if(state.step===0&&steps[0].kind==='plan'){state.step=1;state.elapsed=0;}
    state.mode=mode;state.playing=true;state.lastTs=Date.now();
    clearInterval(state.timer);
    state.timer=setInterval(tick,80);
    render();
  }
  function play(){
    if(state.playing){stopTimer();render();return;}
    start('play');
  }
  function nextStep(){
    clearNotice();
    if(state.playing||state.elapsed<timing().stable){stopTimer();state.elapsed=timing().stable;state.mode='idle';render();return;}
    const steps=stepsFor(state.chunk);
    if(state.step>=steps.length-1)return;
    state.step++;state.elapsed=0;const next=steps[state.step];if(next.kind==='compute')state.selected=next.position;
    start('step');
  }
  function prevStep(){
    stopTimer();clearNotice();
    state.step=Math.max(0,state.step-1);state.elapsed=timing().stable;
    const s=snapshot(state.owner,state.chunk,state.step,state.elapsed);if(s.active!==null)state.selected=s.active;
    render();
  }
  function reset(){
    stopTimer();clearNotice();
    state.step=0;state.elapsed=0;state.mode='idle';state.selected=0;
    state.notice='已重置到计划起点。';
    state.lastPhaseKey='';
    render();
  }
  function replay(){
    stopTimer();clearNotice();
    state.step=0;state.elapsed=0;state.selected=0;state.mode='idle';
    start('play');
  }
  function retarget(kind,value){
    if((kind==='owner'&&value===state.owner)||(kind==='chunk'&&value===state.chunk))return;
    stopTimer();
    const before=snapshot(state.owner,state.chunk,state.step,state.elapsed);
    let boundary='';
    if(kind==='chunk'){
      const mapped=remapStep(state.owner,state.chunk,value,state.step,state.elapsed);
      state.chunk=value;
      state.step=mapped.step;
      state.elapsed=mapped.elapsed;
      boundary=mapped.note;
    }else state.owner=value;
    state.mode='idle';
    state.playing=false;
    state.lastPhaseKey='';
    const after=snapshot(state.owner,state.chunk,state.step,state.elapsed);
    if(before.active!==null)state.selected=before.active;
    else if(after.active!==null)state.selected=after.active;
    state.notice=conditionNotice(after,kind,boundary);
    render();
  }
  ownerButtons.forEach(button=>button.addEventListener('click',()=>retarget('owner',button.dataset.pstOwner)));
  chunkButtons.forEach(button=>button.addEventListener('click',()=>retarget('chunk',button.dataset.pstChunk)));
  pickButtons.forEach((button,index)=>button.addEventListener('click',()=>select(index)));
  playButton.addEventListener('click',play);
  prevButton.addEventListener('click',prevStep);
  nextButton.addEventListener('click',nextStep);
  replayButton.addEventListener('click',replay);
  resetButton.addEventListener('click',reset);
  root.addEventListener('keydown',event=>{
    if(!event.target.matches('[data-pst-pick]'))return;
    const key=event.key;
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(key))return;
    event.preventDefault();
    if(key==='Home')select(0,true);
    else if(key==='End')select(7,true);
    else select(state.selected+(key==='ArrowRight'?1:-1),true);
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&state.playing){stopTimer();render();}});
  window.addEventListener('pagehide',()=>{stopTimer();});
  render();
}

root.ParallelSpaceTime={model:{TOTAL,CAUSAL,SPATIAL,TEMPORAL,TIMING,roundsFor,roundOf,stepsFor,timingFor,snapshot,detail,remapStep},mount:init};
if(typeof document!=='undefined'){
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init,{once:true});
  else init();
}
})(typeof globalThis==='object'?globalThis:this);
