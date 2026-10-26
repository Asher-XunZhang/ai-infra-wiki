/* Chunking changes work admitted per round, not the input or its causal order. */
(function(root){
'use strict';
function plan(chunk){
 if(![1,2,4,8].includes(chunk))throw new Error('Invalid chunk');
 const prefillRounds=8/chunk;
 const fragments=Array.from({length:prefillRounds},(_,i)=>({from:i*chunk+1,to:(i+1)*chunk,positions:Array.from({length:chunk},(_,j)=>i*chunk+j+1)}));
 const rounds=Array.from({length:8},(_,i)=>({round:i+1,a:i+1,aLabel:'A'+(i+1),fragment:fragments[i]||null,afterB:!fragments[i]}));
 return {chunk,prefillRounds,rounds,fragments,aDecodes:8,bPositions:8,aBeforeBFinish:prefillRounds,aAfterBFinish:8-prefillRounds};
}
function compare(chunk){const selected=plan(chunk),whole=plan(8);return {whole,selected,same:chunk===8,aDecodes:8,bPositions:8};}
function create(chunk){
 const p=plan(chunk);
 const frames=[{title:'同一条长输入',caption:'B 有 8 个输入位置；A 固定需要 8 次 Decode。本动画只观察到 B 处理完，截止前的 A 次数不是 A 的总次数。'}];
 for(const f of p.fragments){
  const r=f.from===1?1:Math.floor((f.from-1)/chunk)+1;
  frames.push({title:`第 ${r} 轮：选择工作`,caption:`选入 A 的第 ${r} 个 Decode（A${r}），以及 B 的位置 ${f.from}–${f.to}。A 总计 8 次 Decode，本过程只显示到 B 完成。`});
  frames.push({title:`第 ${r} 轮：完成本轮`,caption:`第 ${r} 轮同时包含 A${r} 与 B 的位置 ${f.from}–${f.to}；B 累计处理 ${f.to} / 8 个输入位置。${r===p.prefillRounds?`B 的 Prefill 完成；${p.aAfterBFinish?`A 还有 ${p.aAfterBFinish} 次 Decode 不在本动画的观察范围内。`:'A 的 8 次 Decode 也恰好在截止前全部出现。'}`:'下一轮继续处理 B 的下一块。'}`});
 }
 return {name:`每轮 ${chunk} 个输入位置`,kind:'chunk-batching',boundaryKey:'f-chunk',chunk,rounds:p.prefillRounds,aDecodes:8,cutoff:'B-prefill-complete',title:'Chunked Prefill：长输入分几轮处理？',intro:'B 的 8 个输入位置保持原顺序，A 固定需要 8 次 Decode。上方对照同时展示“整段 8”和“所选 chunk”：改变的是 B 的 Prefill 分段与轮次，不是 A 的总调度次数。下方动画只观察到 B 处理完。',frames,stepDurations:frames.slice(1).map((_,i)=>i%2===0?.65:1.3),boundary:'合并部署、允许 mixed chunk 的教学策略：每轮选入 A 的一个 Decode 位置和 B 的一块 Prefill。条带同轮推进表示同一批工作，不规定 kernel 并发。轮次不是时间长度，也不能据此推导 TTFT 或 ITL；不模拟 B 的首 token 采样。本动画截止于 B 的 8 个位置处理完，不把截止前的 A 次数当作 A 的总输出数。'};
}
const scenes=[1,2,4,8].map(create);
if(typeof module==='object'&&module.exports){module.exports={plan,compare,create,scenes};return;}

function comparisonPanel(currentChunk){
 const data=compare(currentChunk);
 const section=document.createElement('section');section.className='chunk-compare';
 section.setAttribute('aria-label','整段 8 与所选 chunk 的调度顺序对照');
 const head=document.createElement('div');head.className='chunk-compare-head';
 const title=document.createElement('strong');title.textContent='同一段 B · A 固定 8 次 Decode';
 const sub=document.createElement('span');sub.textContent='顺序图：只表示轮次先后；格子等宽不代表真实耗时';
 head.append(title,sub);
 const grid=document.createElement('div');grid.className='chunk-compare-grid';
 function card(p,label,current){
  const box=document.createElement('article');box.className='chunk-compare-card';
  box.dataset.chunk=String(p.chunk);box.dataset.prefillRounds=String(p.prefillRounds);box.dataset.aDecodes='8';box.dataset.bPositions='8';
  if(current)box.classList.add('is-current');
  const header=document.createElement('header');
  const h=document.createElement('h3');h.textContent=label;header.append(h);
  const tag=document.createElement('span');tag.className='chunk-compare-tag';tag.textContent=data.same&&current?'所选 = 整段':`B Prefill ${p.prefillRounds} 轮`;header.append(tag);
  const subline=document.createElement('p');subline.className='chunk-compare-sub';subline.textContent=`每段 ${p.chunk} 个位置 · A 调度 8 次`;
  const headrow=document.createElement('div');headrow.className='chunk-rounds-head';
  for(const [cls,value] of [['chunk-round-no','轮'],['chunk-a-head','A'],['chunk-b-head','B Prefill（8 个位置）']]){const s=document.createElement('span');s.className=cls;s.textContent=value;headrow.append(s);}
  const rounds=document.createElement('div');rounds.className='chunk-rounds';
  for(const r of p.rounds){
   const row=document.createElement('div');row.className='chunk-round';row.dataset.round=String(r.round);row.dataset.hasB=String(!!r.fragment);
   const no=document.createElement('span');no.className='chunk-round-no';no.textContent=String(r.round);
   const a=document.createElement('span');a.className='chunk-a';a.textContent=r.aLabel;
   const b=document.createElement('span');b.className='chunk-b';
   if(r.fragment){
    const group=document.createElement('span');group.className='chunk-b-group';group.setAttribute('aria-label',`B 位置 ${r.fragment.from} 到 ${r.fragment.to}`);
    for(const pos of r.fragment.positions){const cell=document.createElement('i');cell.className='chunk-b-cell';cell.dataset.bPosition=String(pos);cell.textContent=String(pos);group.append(cell);}
    b.append(group);
   }else{b.classList.add('is-after');b.textContent='—';b.setAttribute('aria-label','本轮只有 A');}
   row.append(no,a,b);rounds.append(row);
  }
  const foot=document.createElement('p');foot.className='chunk-compare-foot';
  foot.textContent=`B 位置 8/8 · A1–A8 全保留 · B 在第 ${p.prefillRounds} 轮完成，该轮同批含 A${p.prefillRounds}`+(p.aAfterBFinish?`；其后 ${p.aAfterBFinish} 轮只有 A`:'，截止前 A 已全部出现');
  box.append(header,subline,headrow,rounds,foot);return box;
 }
 grid.append(card(data.whole,'整段 8',false),card(data.selected,`所选 chunk ${currentChunk}`,true));
 const note=document.createElement('p');note.className='chunk-compare-note';
 note.textContent=data.same
  ?'所选就是整段 8，两栏数据相同。两栏都保留 B1–B8 和 A1–A8；A 的总调度次数始终是 8。下方动画截止于 B 处理完，截止前的 A 次数不是 A 的总次数。同一轮只表示同批，不表示 A 先完成；不比较 A 输出总数或速度。'
  :`两栏都保留 B1–B8 和 A1–A8。所选 chunk 越小，相邻 A 之间插入的 B 段越短，B 的 Prefill 轮数越多；A 的总调度次数始终是 8。整段 8 为 1 轮，chunk ${currentChunk} 为 ${data.selected.prefillRounds} 轮。下方动画截止于 B 处理完，截止前的 A 次数不是 A 的总次数。同一轮只表示同批，不表示 A 先完成；不比较 A 输出总数或速度。`;
 section.append(head,grid,note);return section;
}

root.MechanismScenes['f-batching'].push(...scenes);
root.ChunkBatchingVisual=(svg,viewport,s)=>{
 const visual=root.TeachingDrawing.board(svg,viewport,'chunk-batching',()=>552,H=>{
  const {txt,rect,text,box,show,set,token,move}=H;
  const title=txt('B · 原输入的 8 个位置','teach-heading'),legend=txt('下划线 = 计算进度；外框 = 当前输入块','teach-small');
  const inputs=Array.from({length:8},(_,i)=>({cell:token('chunk-input-'+i,String(i+1),34,'teach-token teach-color-1'),fill:rect('teach-value')}));
  const selected=rect('teach-map'),roundText=txt('','teach-heading'),progress=txt('','teach-small'),history=txt('已处理位置保留 KV，后续块仍可读取。','teach-small');
  const aTitle=txt('A · 每轮产生一个新输出 token','teach-heading');
  const outputs=Array.from({length:8},(_,i)=>token('chunk-output-'+i,'y'+(i+1),34,'teach-token teach-color-0'));
  const ledgerTitle=txt('按轮次对照：同列在同一轮执行','teach-heading');
  const ledger=Array.from({length:s.rounds},(_,i)=>({label:txt('轮 '+(i+1),'teach-small'),panel:rect('teach-track'),b:txt('','teach-small'),a:txt('A +1','teach-small'),fill:rect('teach-value')}));
  const result=txt('','teach-text'),cutoff1=txt('本动画观察截止：B 处理完','teach-small'),cutoff2=txt('','teach-small');
  return(p,m)=>{
   const W=m?360:720,left=20,gap=(W-40)/8,r=Math.min(s.rounds-1,Math.floor(Math.max(0,p-.000001)/2)),local=Math.max(0,Math.min(2,p-r*2)),work=root.TeachingDrawing.phase(local,1,2),done=Math.floor(p/2),processed=Math.min(8,done*s.chunk),end=p>=s.frames.length-1;
   text(title,18,30);text(legend,18,57);
   inputs.forEach((e,i)=>{const fraction=i<r*s.chunk?1:i<(r+1)*s.chunk?work:0;move(e.cell,left+i*gap,86,1);set(e.cell.r,{width:gap-5});set(e.cell.t,{x:(gap-5)/2});box(e.fill,left+i*gap,128,(gap-5)*fraction,7);});
   box(selected,left+r*s.chunk*gap-3,80,s.chunk*gap+1,61);set(selected,{fill:'none','stroke-width':2});show(selected,p>0&&!end?1:0);
   text(roundText,18,175,end?'B 的 Prefill 完成':`第 ${r+1} / ${s.rounds} 轮`);text(progress,18,203,`B 已完成 ${processed} / 8 个位置 · 每轮上限 ${s.chunk}`);
   text(history,18,225);
   text(aTitle,18,251);outputs.forEach((e,i)=>{move(e,left+i*gap,269,i<done?1:i===r?work:0);set(e.r,{width:gap-5});set(e.t,{x:(gap-5)/2});e.g.dataset.complete=String(i<done);});
   text(ledgerTitle,18,340);const width=(W-40)/s.rounds;
   ledger.forEach((e,i)=>{const x=left+i*width,f=i<r?1:i===r?work:0;box(e.panel,x,370,width-5,69);text(e.label,x+2,362);text(e.b,x+2,392,s.chunk===1?`B ${i+1}`:`B ${i*s.chunk+1}–${(i+1)*s.chunk}`);text(e.a,x+2,416);box(e.fill,x,433,(width-5)*f,6);});
   text(result,18,479,end?`B 完成：${s.rounds} 轮 · A 截止前 ${s.rounds} 次`:'下一块要等下一轮才继续');
   text(cutoff1,18,505);text(cutoff2,18,523,end?`A 总计 8 次 Decode；余下 ${8-s.rounds} 次不在本图`:'A 总计 8 次 Decode；本图只显示到 B 完成');
   svg.dataset.processed=processed;svg.dataset.decodeOutputs=done;svg.dataset.aTotal='8';svg.dataset.aShown=String(done);svg.dataset.prefillRounds=String(s.rounds);svg.dataset.cutoff='B-prefill-complete';
  };
 });
 const panel=comparisonPanel(s.chunk);
 (viewport.closest('.motion-theater')||viewport).before(panel);
 const destroy=visual.destroy;
 visual.destroy=()=>{panel.remove();destroy();};
 return visual;
};
})(typeof window==='object'?window:globalThis);
