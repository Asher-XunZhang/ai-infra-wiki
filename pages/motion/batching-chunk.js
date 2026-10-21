/* Chunking changes work admitted per round, not the input or its causal order. */
(function(root){
'use strict';
function create(chunk){
 const rounds=8/chunk,frames=[{title:'同一条长输入',caption:'B 有 8 个输入位置；A 已在 Decode。每轮给 A 一个 Decode 位置，再处理 B 的一块输入。'}];
 for(let r=0;r<rounds;r++){
  frames.push({title:`第 ${r+1} 轮：选择工作`,caption:`选入 A 的一个 Decode 位置，以及 B 的位置 ${r*chunk+1}–${(r+1)*chunk}。已处理的 B 位置保持不变。`});
  frames.push({title:`第 ${r+1} 轮：完成本轮`,caption:`B 累计处理 ${(r+1)*chunk} / 8 个输入位置；A 又得到一个输出 token。${r+1===rounds?'B 的 Prefill 完成。':'下一轮继续处理 B 的下一块。'}`});
 }
 return {name:`每轮 ${chunk} 个输入位置`,kind:'chunk-batching',boundaryKey:'f-chunk',chunk,rounds,title:'Chunked Prefill：长输入分几轮处理？',intro:'B 的 8 个输入位置保持原顺序。改变每轮处理的数量，观察 B 需要多少轮，以及 A 获得几次交错生成的机会。',frames,stepDurations:frames.slice(1).map((_,i)=>i%2===0?.65:1.3),boundary:'合并部署、允许 mixed chunk 的教学策略：每轮选入 A 的一个 Decode 位置和 B 的一块 Prefill。条带同轮推进表示同一批工作，不规定 kernel 并发。轮次不是时间长度，也不能据此推导 TTFT 或 ITL；不模拟 B 的首 token 采样。'};
}
const scenes=[1,2,4,8].map(create);
if(typeof module==='object'&&module.exports){module.exports={create,scenes};return;}
root.MechanismScenes['f-batching'].push(...scenes);
root.ChunkBatchingVisual=(svg,viewport,s)=>root.TeachingDrawing.board(svg,viewport,'chunk-batching',()=>530,H=>{
 const {txt,rect,text,box,show,set,token,move}=H;
 const title=txt('B · 原输入的 8 个位置','teach-heading'),legend=txt('下划线 = 计算进度；外框 = 当前输入块','teach-small');
 const inputs=Array.from({length:8},(_,i)=>({cell:token('chunk-input-'+i,String(i+1),34,'teach-token teach-color-1'),fill:rect('teach-value')}));
 const selected=rect('teach-map'),roundText=txt('','teach-heading'),progress=txt('','teach-small'),history=txt('已处理位置保留 KV，后续块仍可读取。','teach-small');
 const aTitle=txt('A · 每轮产生一个新输出 token','teach-heading');
 const outputs=Array.from({length:8},(_,i)=>token('chunk-output-'+i,'y'+(i+1),34,'teach-token teach-color-0'));
 const ledgerTitle=txt('按轮次对照：同列在同一轮执行','teach-heading');
 const ledger=Array.from({length:s.rounds},(_,i)=>({label:txt('轮 '+(i+1),'teach-small'),panel:rect('teach-track'),b:txt('','teach-small'),a:txt('A +1','teach-small'),fill:rect('teach-value')}));
 const result=txt('','teach-heading');
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
  text(result,18,479,end?`B 用了 ${s.rounds} 轮；A 交错生成 ${s.rounds} 次`:'下一块要等下一轮才继续');svg.dataset.processed=processed;svg.dataset.decodeOutputs=done;
 };
});
})(typeof window==='object'?window:globalThis);
