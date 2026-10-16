/* Each position keeps its identity. A plan is not completed KV history. */
(() => {
'use strict';
const {board,phase,lerp}=window.TeachingDrawing;
window.ChunkPrefillVisual=(svg,viewport,s)=>board(svg,viewport,'chunk-prefill',m=>m?410+s.model.chunks(s.cap).at(-1).history.length*170:260+s.model.chunks(s.cap).at(-1).history.length*112,H=>{
 const {txt,rect,text,box,token,move,show}=H,history=s.events.at(-1).history;
 const head=txt('① 完整待算输入 · 原位置保留','teach-heading'),sub=txt('L = 已有长请求；S = 等待短请求','teach-small'),ledger=txt('② 每轮计划 → 执行完成','teach-heading'),status=txt('','teach-small');
 const positions=[...Array.from({length:10},(_,i)=>({id:'L'+(i+1),owner:'L',i})),...Array.from({length:2},(_,i)=>({id:'S'+(i+1),owner:'S',i}))];
 let li=0,si=0;
 history.forEach((r,j)=>{for(let k=0;k<r.l;k++)Object.assign(positions[li++],{round:j,slot:k});for(let k=0;k<r.s;k++)Object.assign(positions[10+si++],{round:j,slot:r.l+k});});
 const rows=history.map((r,i)=>({panel:rect('teach-panel'),label:txt('','teach-small'),slots:Array.from({length:s.cap},()=>rect('teach-slot')),r,i}));
 const actors=positions.map(d=>({...d,original:token('input-'+d.id,d.id,44,'teach-token teach-color-'+(d.owner==='L'?0:1)),copy:token('selected-'+d.id,d.id,44,'teach-token teach-color-'+(d.owner==='L'?0:1)),progress:rect('teach-progress')}));
 actors.forEach(a=>svg.append(a.copy.g));
 return(p,m)=>{
  const W=m?360:720,cols=m?4:10,stepX=m?62:60,ledgerY=m?390:240,rowHeight=m?170:112,slotCols=m?4:8;
  const state=s.events[Math.floor(p)];
  text(head,18,28);text(sub,18,54);text(ledger,18,ledgerY-22);text(status,18,ledgerY-48,`已完成：L ${10-state.left} / 10，S ${2-state.shortLeft} / 2`);
  rows.forEach(({panel,label,slots,r,i})=>{const y=ledgerY+i*rowHeight,planAt=2*i+1,doneAt=2*i+2;
   box(panel,18,y,W-36,rowHeight-12);text(label,30,y+24,`第 ${i+1} 轮 · ${p>=doneAt?'已完成':p>=planAt?'执行中':p>planAt-1?'选择中':'待选择'} · 额度 ${s.cap}`);
   panel.dataset.round=String(i+1);panel.dataset.state=p>=doneAt?'done':p>=planAt?'plan':'waiting';
   slots.forEach((cell,j)=>box(cell,30+(j%slotCols)*(m?78:80),y+38+Math.floor(j/slotCols)*48,44,34));
  });
  actors.forEach(a=>{
   const sx=30+(a.i%cols)*stepX,sy=a.owner==='L'?82+Math.floor(a.i/cols)*48:(m?242:144);
   const dx=30+(a.slot%slotCols)*(m?78:80),dy=ledgerY+a.round*rowHeight+38+Math.floor(a.slot/slotCols)*48;
   const planAt=2*a.round+1,doneAt=planAt+1,count=history[a.round].l+history[a.round].s,movement=phase(p,planAt-1+a.slot/count,planAt-1+(a.slot+1)/count),work=phase(p,planAt,doneAt);
   move(a.original,sx,sy);a.original.g.dataset.completed=String(p>=doneAt);a.original.r.setAttribute('opacity',p>=doneAt?.35:1);
   const lane=m?296:652,x=movement<.25?lerp(sx,lane,movement/.25):movement<.75?lane:lerp(lane,dx,(movement-.75)/.25),y=movement<.25?sy:movement<.75?lerp(sy,dy,(movement-.25)/.5):dy;move(a.copy,x,y,movement>0?1:0);a.copy.g.dataset.committed=String(p>=doneAt);
   box(a.progress,dx,dy+37,44*work,4);show(a.progress,movement===1?1:0);
  });
 };
});
})();
