/* PCP: one long Prefill request, eight Query positions, 36 allowed causal
 * Q–K pairs. This canvas replaces the former 870-high matrix plus the separate
 * adoption panel: ownership, missing remote K/V, the arrival gate, O_i
 * ownership and all three split totals live on one board.
 *
 * Teaching times are labels, not measured latency. The AllGather copy progress
 * is shown separately from per-row data readiness so a partial arrival is
 * never called a completed collective. */
(()=>{'use strict';
const NS='http://www.w3.org/2000/svg';
const P=window.ParallelTeachingModel;
const {board,phase}=window.TeachingDrawing;

function visual(svg,viewport,s){
 const film=viewport.closest('.mechanism-film');if(film)film.classList.add('is-pcp');
 const d=Array.isArray(s?.data?.queries)?s.data:(P&&typeof P.context==='function'?P.context(false):null);
 if(!d)return {paint(){},destroy(){film?.classList.remove('is-pcp');}};
 const zigzag=Array.isArray(d.ranks?.[0])&&d.ranks[0].includes(6);
 const splitKey=zigzag?'zigzag':'continuous';
 const other=zigzag?(P&&P.context?P.context(false):d):(P&&P.context?P.context(true):d);
 const cases={
  continuous:zigzag?other:d,
  zigzag:zigzag?d:other
 };
 const q1=d.queries[0],q7=d.queries[6];
 const q7Remote=q7.requiredRemote.map(i=>'K'+(i+1));
 const q7Range=q7Remote.length?q7Remote[0]+'–'+q7Remote[q7Remote.length-1]:'无';
 const copyByPos=[];d.copies.forEach(c=>copyByPos[c.position]=c);
 const required=d.requiredRemoteByRank;
 const pairSum=rows=>rows.reduce((sum,i)=>sum+i+1,0);
 const rowsText=rows=>rows.length?rows.map(i=>'Q'+(i+1)).join(' '):'—';
 const localText=rows=>rows.length?rows.map(i=>'p'+(i+1)).join(' '):'—';
 const rangeText=positions=>positions.length?(positions.length===1?'K'+(positions[0]+1):'K'+(positions[0]+1)+'–K'+(positions[positions.length-1]+1)):'无';
 const ownerText=r=>'Rank '+r+' · P'+r+' / GPU '+r;

 const inner=board(svg,viewport,'parallel-pcp',m=>m?744:622,H=>{
  const {set,show}=H;
  const mk=(tag,attrs={})=>{const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))if(v!==undefined&&v!==null)e.setAttribute(k,v);return e;};
  const add=(parent,child)=>{parent.append(child);return child;};
  const at=(e,x,y)=>set(e,{x,y});
  const dims=(e,x,y,w,h)=>set(e,{x,y,width:w,height:h});
  const place=(g,x,y)=>set(g,{transform:`translate(${x} ${y})`});
  const txt=(value,cls='pcpv-label',extra={})=>{const t=mk('text',{class:cls,...extra});t.textContent=value;svg.append(t);return t;};
  const panel=(cls='pcpv-panel')=>add(svg,mk('rect',{class:cls,rx:8}));
  const chip=(label,cls,w,h,dot=false)=>{
   const g=mk('g',{class:cls});
   const r=mk('rect',{rx:3});
   const t=mk('text',{'text-anchor':'middle'});
   t.textContent=label;
   g.append(r,t);
   let d=null;
   if(dot){d=mk('circle',{class:'pcpv-chip-dot'});g.append(d);}
   svg.append(g);
   return {g,r,t,dot:d,w,h};
  };
  const sizeChip=(c,w,h)=>{c.w=w;c.h=h;set(c.r,{width:w,height:h});set(c.t,{x:w/2,y:h-5});if(c.dot)set(c.dot,{cx:w-4,cy:4});};

  // Backgrounds first; every label and cell is created afterwards.
  const gridPanel=panel(),calloutPanel=panel(),rankPanels=[panel(),panel()],laneBgs=[panel('pcpv-lane'),panel('pcpv-lane')];
  const gridLayer=mk('g',{'aria-hidden':'true'});svg.append(gridLayer);
  const cells=[];
  // c>r is causally forbidden, not a pending pair. Mark it masked at birth;
  // paint only revisits c<=i, so these cells can never become computed.
  for(let r=0;r<8;r++)for(let c=0;c<8;c++){
   const masked=c>r;
   const rect=mk('rect',{class:'pcpv-cell'+(masked?' is-masked':''),rx:2});
   if(masked)rect.dataset.masked='true';
   gridLayer.append(rect);cells.push({r,c,rect,masked});
  }
  const focusRow=mk('rect',{class:'pcpv-focus-row',rx:4,fill:'none'});svg.append(focusRow);

  const title=txt('PCP · 一条请求，8 个 Query，36 个因果配对','pcpv-title');
  const sub1=txt('','pcpv-sub'),sub2=txt('','pcpv-sub');
  const gridTitle=txt('','pcpv-section-title');
  const colLabels=Array.from({length:8},(_,i)=>txt('K'+(i+1),'pcpv-col-label pcpv-owner-text pcpv-owner-'+d.owners[i],{'text-anchor':'middle'}));
  const rowLabels=Array.from({length:8},(_,i)=>txt('Q'+(i+1),'pcpv-row-label pcpv-owner-text pcpv-owner-'+d.owners[i]+(i===6?' is-focus':'')));
  const counts=Array.from({length:8},(_,i)=>txt((i+1)+' 对','pcpv-count'));
  const outs=Array.from({length:8},(_,i)=>chip('O'+(i+1),'pcpv-out pcpv-owner-'+d.owners[i]+(i===6?' is-focus':''),30,18));
  const legendItems=[
   {label:'R0 行',cls:'pcpv-legend-swatch pcpv-owner-0'},
   {label:'R1 行',cls:'pcpv-legend-swatch pcpv-owner-1'},
   {label:'远端未到',cls:'pcpv-legend-swatch is-remote'},
   {label:'已计入',cls:'pcpv-legend-swatch is-computed pcpv-owner-0'},
   {label:'禁止 c>r',cls:'pcpv-legend-swatch is-masked'}
  ];
  const legend=legendItems.map(item=>{
   const g=mk('g',{class:'pcpv-legend'}),sw=mk('rect',{class:item.cls,rx:2}),t=mk('text',{class:'pcpv-legend-label'});
   t.textContent=item.label;g.append(sw,t);svg.append(g);return {g,sw,t};
  });
  const formula=txt('','pcpv-formula');

  const sourceLabel=txt('','pcpv-note');
  const sourceChips=Array.from({length:8},(_,i)=>chip('p'+(i+1),'pcpv-source pcpv-owner-'+d.owners[i],28,20));
  const q1Card=mk('rect',{class:'pcpv-subcard',rx:7});svg.append(q1Card);
  const q1Title=txt('','pcpv-card-title'),q1Line1=txt('','pcpv-card-line'),q1Line2=txt('','pcpv-card-line');
  const q7Card=mk('rect',{class:'pcpv-subcard',rx:7});svg.append(q7Card);
  const q7Title=txt('','pcpv-card-title'),q7Line1=txt('','pcpv-card-line'),q7Line2=txt('','pcpv-card-line'),q7Line3=txt('','pcpv-card-line');
  q1Title.textContent='Q1 · R'+d.owners[0]+' · 1 对';
  q1Line1.textContent='依赖要求：只读本地 K1，无远端';
  q1Line2.textContent='当前：未开算';
  q7Title.textContent='Q7 · R'+q7.owner+' · 7 对';
  q7Line1.textContent='依赖要求：远端 '+q7Range+'（'+q7Remote.length+' 份）';
  q7Line2.textContent='要求：读 K1…K7；远端到齐后才可算';
  q7Line3.textContent='当前：远端 0/'+q7Remote.length+' 到齐';

  const gatherLabel=txt('','pcpv-gather-label');
  const gatherTrack=mk('rect',{class:'pcpv-gather-track',rx:3});svg.append(gatherTrack);
  const gatherFill=mk('rect',{class:'pcpv-gather-fill',rx:3});svg.append(gatherFill);
  const gatherTicks=Array.from({length:8},()=>{const t=mk('rect',{class:'pcpv-gather-tick'});svg.append(t);return t;});
  const gatherNotes=[0,1,2].map(()=>txt('','pcpv-note'));
  const noteTexts=['当前：部分副本 0/8，未完成；','本图按所需数据就绪开算；','本地齐备 ≠ 全量副本到齐。'];
  gatherNotes.forEach((t,i)=>t.textContent=noteTexts[i]);

  const rankTitles=[0,1].map(r=>txt('','pcpv-card-title'));
  const rankOwners=[0,1].map(r=>txt('','pcpv-card-line'));
  const rankLocals=[0,1].map(r=>txt('','pcpv-card-line'));
  const rankRemotes=[0,1].map(r=>txt('','pcpv-card-line'));
  for(let r=0;r<2;r++){
   rankTitles[r].textContent=ownerText(r);
   rankOwners[r].textContent='负责 '+rowsText(d.ranks[r])+' · '+d.work[r]+' 对';
   rankLocals[r].textContent='本地原 K/V '+localText(d.localKV[r])+'（保留）';
   const need=required[r];
   rankRemotes[r].textContent='依赖要求：'+(need.length?rangeText(need)+'（'+need.length+' 份）':'无远端'+(d.ranks[r].length?'（本行只读本地）':''));
   rankRemotes[r].classList.add(need.length?'is-waiting':'is-local');
  }

  const laneLabel=txt('','pcpv-note');
  const lanes=[0,1].map(from=>{
   const to=1-from;
   const label=txt('','pcpv-lane-label');
   label.textContent='R'+from+' → R'+to;
   const copies=d.copies.filter(c=>c.from===from).sort((a,b)=>a.position-b.position);
   const chips=copies.map(c=>{
    const needed=required[to].includes(c.position);
    const ch=chip('p'+(c.position+1),'pcpv-chip is-copy pcpv-owner-'+from+(needed?' is-needed':' is-extra'),34,18,true);
    ch.copy=c;ch.from=from;ch.to=to;ch.needed=needed;
    return ch;
   });
   const note=txt('','pcpv-lane-note');
   const status=txt('','pcpv-lane-status',{'text-anchor':'end'});
   const need=required[to];
   note.textContent='R'+to+' 依赖 '+need.length+' 份'+(need.length?'（'+rangeText(need)+'）':'（本切法不读）');
   return {from,to,label,chips,note,status};
  });

  const ledgerTitle=txt('','pcpv-section-title');
  const ledgerDefs=[
   {key:'single',data:null,ranks:[[0,1,2,3,4,5,6,7],[]],work:[36,0]},
   {key:'continuous',data:cases.continuous,ranks:cases.continuous.ranks,work:cases.continuous.work},
   {key:'zigzag',data:cases.zigzag,ranks:cases.zigzag.ranks,work:cases.zigzag.work}
  ];
  const ledger=ledgerDefs.map(def=>{
   const g=mk('g',{class:'pcpv-split-card'+(def.key===splitKey?' is-current':'')+(def.key==='single'?' is-reference':' is-interactive'),'data-split':def.key});
   const rect=mk('rect',{class:'pcpv-card',rx:7});
   const title=mk('text',{class:'pcpv-card-title'});
   const badge=mk('text',{class:'pcpv-badge','text-anchor':'end'});
   const owners=Array(8).fill(0);def.ranks.forEach((rows,r)=>rows.forEach(i=>owners[i]=r));
   const mini=owners.map((owner,i)=>chip(String(i+1),'pcpv-mini pcpv-owner-'+owner,19,14));
   const formula=mk('text',{class:'pcpv-card-formula'});
   g.append(rect,title,badge,...mini.map(c=>c.g),formula);
   svg.append(g);
   if(def.key!=='single'){
    g.setAttribute('role','button');g.setAttribute('tabindex','0');
    g.setAttribute('aria-pressed',String(def.key===splitKey));
    const activate=()=>svg.dispatchEvent(new CustomEvent('parallel-pcp-split',{bubbles:true,detail:{split:def.key}}));
    g.addEventListener('click',e=>{e.stopPropagation();activate();});
    g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.stopPropagation();activate();}});
   }
   const total=def.work[0]+def.work[1];
   const formulas=def.ranks.map((rows,r)=>rows.length?rows.map(i=>i+1).join('+')+' = '+pairSum(rows):'未参与');
   return {g,rect,title,badge,mini,formula,def,owners,total,formulas};
  });

  const status=txt('','pcpv-status');
  const note=txt('','pcpv-note');

  const wide={
   W:720,H:622,
   title:{x:18,y:26},sub1:{x:18,y:46},sub2:null,
   gridPanel:{x:12,y:56,w:424,h:282},gridTitle:{x:24,y:80},
   grid:{x:68,y:112,cell:23,pitch:24},rowLabelX:52,countX:270,outX:306,outW:30,outH:18,
   legend:{x:350,y:116,dy:18},formula:{x:24,y:328},
   callout:{x:448,y:56,w:260,h:282},sourceLabel:{x:456,y:78},sourceChips:{x:456,y:84,w:28,h:20,gap:3},
   q1Card:{x:456,y:114,w:244,h:50},q1Title:{x:464,y:132},q1Line1:{x:464,y:148},q1Line2:{x:464,y:162},
   q7Card:{x:456,y:172,w:244,h:74},q7Title:{x:464,y:190},q7Line1:{x:464,y:206},q7Line2:{x:464,y:220},q7Line3:{x:464,y:234},
   gather:{x:456,y:266,w:244,h:8},gatherLabel:{x:456,y:258},gatherNotes:[{x:456,y:286},{x:456,y:300},{x:456,y:314}],
   rankCards:[{x:12,y:348,w:340,h:76},{x:368,y:348,w:340,h:76}],rankText:{x:22,title:18,owner:36,local:54,remote:70},
   laneLabel:{x:12,y:436},lanes:[{x:12,y:444,w:696,h:24},{x:12,y:470,w:696,h:24}],
   lane:{labelX:20,chipX:96,chipW:34,chipH:18,gap:4,noteX:260,statusX:700},
   ledgerTitle:{x:12,y:510},ledgerCards:[{x:12,y:518,w:224,h:60},{x:248,y:518,w:224,h:60},{x:484,y:518,w:224,h:60}],
   ledger:{titleX:8,titleY:16,badgeX:216,titleAnchor:'start',chipsX:8,chipsY:26,chipW:19,chipH:14,gap:3,formulaX:8,formulaY:50},
   status:{x:12,y:594},note:{x:12,y:610}
  };
  const narrow={
   W:360,H:744,
   title:{x:12,y:22},sub1:{x:12,y:42},sub2:{x:12,y:56},
   gridPanel:{x:6,y:64,w:348,h:250},gridTitle:{x:16,y:84},
   grid:{x:52,y:112,cell:20,pitch:21},rowLabelX:42,countX:224,outX:264,outW:26,outH:16,
   legend:null,formula:{x:16,y:300},
   callout:{x:6,y:324,w:348,h:92},sourceLabel:null,sourceChips:null,
   q1Card:{x:12,y:330,w:166,h:60},q1Title:{x:20,y:344},q1Line1:{x:20,y:358},q1Line2:{x:20,y:372},
   q7Card:{x:182,y:330,w:166,h:60},q7Title:{x:190,y:344},q7Line1:{x:190,y:358},q7Line2:{x:190,y:372},q7Line3:{x:190,y:384},
   gather:{x:12,y:404,w:336,h:6},gatherLabel:{x:12,y:398},gatherNotes:null,
   rankCards:[{x:6,y:424,w:348,h:58},{x:6,y:488,w:348,h:58}],rankText:{x:16,title:16,owner:31,local:44,remote:56},
   laneLabel:{x:6,y:558},lanes:[{x:6,y:566,w:348,h:20},{x:6,y:588,w:348,h:20}],
   lane:{labelX:12,chipX:66,chipW:30,chipH:16,gap:2,noteX:200,statusX:348},
   ledgerTitle:{x:6,y:622},ledgerCards:[{x:6,y:630,w:348,h:24},{x:6,y:654,w:348,h:24},{x:6,y:678,w:348,h:24}],
   ledger:{titleX:8,titleY:16,badgeX:0,titleAnchor:'start',chipsX:78,chipsY:5,chipW:18,chipH:14,gap:2,formulaX:246,formulaY:16},
   status:{x:6,y:716},note:{x:6,y:730}
  };
  let L=null,lastM=null;
  function applyLayout(m){
   L=m?narrow:wide;
   svg.dataset.pcpLayout=m?'narrow':'wide';
   title.textContent=m?'PCP · 8 位置，36 个因果配对':'PCP · 一条请求，8 个 Query，36 个因果配对';
   sub1.textContent=m?'同一请求 · 8 个位置 · 36 个配对':'同一请求 p1…p8；Qi 只允许读 K1…Ki，所以 Q1=1 对、Q8=8 对；灰格 c>r 禁止读取。';
   sub2.textContent=m?'Q_i 只读 K1…Ki；灰格 = 禁止的未来位置':'';
   show(sub2,m?1:0);
   at(title,L.title.x,L.title.y);at(sub1,L.sub1.x,L.sub1.y);if(L.sub2)at(sub2,L.sub2.x,L.sub2.y);
   dims(gridPanel,L.gridPanel.x,L.gridPanel.y,L.gridPanel.w,L.gridPanel.h);dims(calloutPanel,L.callout.x,L.callout.y,L.callout.w,L.callout.h);
   at(gridTitle,L.gridTitle.x,L.gridTitle.y);
   gridTitle.textContent=m?'因果配对矩阵 · 36 格；灰格 = 禁止':'因果配对矩阵 · 下三角 36 格可算；上三角禁止';
   const gx=L.grid.x,gy=L.grid.y,cell=L.grid.cell,pitch=L.grid.pitch;
   cells.forEach(({r,c,rect})=>dims(rect,gx+c*pitch,gy+r*pitch,cell,cell));
   dims(focusRow,gx-2,gy+6*pitch-2,pitch*7+cell+4,cell+4);
   colLabels.forEach((t,i)=>{at(t,gx+i*pitch+cell/2,gy-8);});
   rowLabels.forEach((t,i)=>at(t,L.rowLabelX,gy+i*pitch+cell-6));
   counts.forEach((t,i)=>at(t,L.countX,gy+i*pitch+cell-6));
   outs.forEach((c,i)=>{place(c.g,L.outX,gy+i*pitch+(cell-L.outH)/2);sizeChip(c,L.outW,L.outH);});
   if(L.legend){
    legend.forEach((item,i)=>{place(item.g,L.legend.x,L.legend.y+i*L.legend.dy);dims(item.sw,0,0,10,10);at(item.t,14,9);show(item.g,1);});
   }else legend.forEach(item=>show(item.g,0));
   at(formula,L.formula.x,L.formula.y);
   formula.textContent=m?'1+…+8=36 对 · 灰格 = c>r 禁止':'Q1=1 · Q7=7 · Q8=8 · 1+…+8=36；灰格 = c>r 禁止读取';

   if(L.sourceLabel){
    at(sourceLabel,L.sourceLabel.x,L.sourceLabel.y);sourceLabel.textContent='同一请求 · 8 个位置（颜色 = 归属）';show(sourceLabel,1);
    sourceChips.forEach((c,i)=>{place(c.g,L.sourceChips.x+i*(L.sourceChips.w+L.sourceChips.gap),L.sourceChips.y);sizeChip(c,L.sourceChips.w,L.sourceChips.h);show(c.g,1);});
   }else{
    show(sourceLabel,0);sourceChips.forEach(c=>show(c.g,0));
   }
   dims(q1Card,L.q1Card.x,L.q1Card.y,L.q1Card.w,L.q1Card.h);
   at(q1Title,L.q1Title.x,L.q1Title.y);at(q1Line1,L.q1Line1.x,L.q1Line1.y);at(q1Line2,L.q1Line2.x,L.q1Line2.y);
   dims(q7Card,L.q7Card.x,L.q7Card.y,L.q7Card.w,L.q7Card.h);
   at(q7Title,L.q7Title.x,L.q7Title.y);at(q7Line1,L.q7Line1.x,L.q7Line1.y);at(q7Line2,L.q7Line2.x,L.q7Line2.y);at(q7Line3,L.q7Line3.x,L.q7Line3.y);
   q1Title.textContent='Q1 · R'+d.owners[0]+' · 1 对';
   q7Title.textContent='Q7 · R'+q7.owner+' · 7 对';
   q1Line1.textContent=m?'依赖：只读本地 K1':'依赖要求：只读本地 K1，无远端';
   q7Line1.textContent=m?'依赖：远端 '+q7Range:'依赖要求：远端 '+q7Range+'（'+q7Remote.length+' 份）';
   q7Line2.textContent=m?'读 K1…K7；到齐才可算':'要求：读 K1…K7；远端到齐后才可算';

   dims(gatherTrack,L.gather.x,L.gather.y,L.gather.w,L.gather.h);
   set(gatherFill,{x:L.gather.x,y:L.gather.y,height:L.gather.h});
   at(gatherLabel,L.gatherLabel.x,L.gatherLabel.y);
   gatherTicks.forEach((t,i)=>{dims(t,L.gather.x+i*L.gather.w/8,L.gather.y,1,L.gather.h);});
   if(L.gatherNotes){
    L.gatherNotes.forEach((p,i)=>{at(gatherNotes[i],p.x,p.y);show(gatherNotes[i],1);});
   }else gatherNotes.forEach(t=>show(t,0));

   L.rankCards.forEach((p,i)=>{
    dims(rankPanels[i],p.x,p.y,p.w,p.h);
    at(rankTitles[i],p.x+L.rankText.x,p.y+L.rankText.title);
    at(rankOwners[i],p.x+L.rankText.x,p.y+L.rankText.owner);
    at(rankLocals[i],p.x+L.rankText.x,p.y+L.rankText.local);
    at(rankRemotes[i],p.x+L.rankText.x,p.y+L.rankText.remote);
   });
   at(laneLabel,L.laneLabel.x,L.laneLabel.y);
   laneLabel.textContent=m?'远端副本 · 实线=原，虚线=副本':'远端副本 · AllGather 示例；原 K/V 留在本 rank（实线 = 原，虚线 = 副本）';
   lanes.forEach((lane,i)=>{
    const p=L.lanes[i];
    dims(laneBgs[i],p.x,p.y,p.w,p.h);
    at(lane.label,p.x+L.lane.labelX,p.y+15);
    lane.chips.forEach((c,j)=>{
     place(c.g,p.x+L.lane.chipX+j*(L.lane.chipW+L.lane.gap),p.y+3);
     sizeChip(c,L.lane.chipW,L.lane.chipH);
    });
    at(lane.note,p.x+L.lane.noteX,p.y+15);
    at(lane.status,p.x+L.lane.statusX,p.y+15);
    lane.note.textContent=m?'R'+lane.to+' 依赖 '+required[lane.to].length+' 份':'R'+lane.to+' 依赖 '+required[lane.to].length+' 份'+(required[lane.to].length?'（'+rangeText(required[lane.to])+'）':'（本切法不读）');
   });
   at(ledgerTitle,L.ledgerTitle.x,L.ledgerTitle.y);
   ledgerTitle.textContent=m?'三种分工 · 总数 36 对':'三种分工 · 同一组 Q1…Q8，允许配对总数都是 36';
   ledger.forEach((card,i)=>{
    const p=L.ledgerCards[i],q=L.ledger;
    place(card.g,p.x,p.y);dims(card.rect,0,0,p.w,p.h);
    at(card.title,q.titleX,q.titleY);set(card.title,{'text-anchor':q.titleAnchor});
    at(card.badge,q.badgeX,q.titleY);if(m)show(card.badge,0);else{show(card.badge,1);card.badge.setAttribute('text-anchor','end');}
    card.mini.forEach((c,j)=>{place(c.g,q.chipsX+j*(q.chipW+q.gap),q.chipsY);sizeChip(c,q.chipW,q.chipH);});
    at(card.formula,q.formulaX,q.formulaY);
    const titles={single:m?'单卡 36':'单卡 · 36 对',continuous:m?'连续 10/26':'连续 · 10 / 26',zigzag:m?'前后 18/18':'前后 · 18 / 18'};
    const formulas={
     single:m?'R0 36':'R0 1+…+8=36 · R1 未参与',
     continuous:m?'R0 10 · R1 26':'R0 1+2+3+4=10 · R1 5+6+7+8=26',
     zigzag:m?'R0 18 · R1 18':'R0 1+2+7+8=18 · R1 3+4+5+6=18'
    };
    card.title.textContent=titles[card.def.key];
    card.formula.textContent=formulas[card.def.key];
    card.badge.textContent=card.def.key==='single'?'参照':(card.def.key===splitKey?'当前':'切换');
    card.g.setAttribute('class','pcpv-split-card'+(card.def.key===splitKey?' is-current':'')+(card.def.key==='single'?' is-reference':' is-interactive'));
    if(card.def.key!=='single')card.g.setAttribute('aria-pressed',String(card.def.key===splitKey));
   });
   at(status,L.status.x,L.status.y);at(note,L.note.x,L.note.y);
   note.textContent=m?'配对数 ≠ 耗时；O_i 按 Query 归属，不跨 rank 求和，不承诺加速。':'配对数 = 允许的 Q–K 格，不是 GPU 等价工作量；O_i 按 Query 归属，不跨 rank 求和。本图不承诺加速。';
   lastM=m;
  }

  function paint(p,m){
   if(m!==lastM)applyLayout(m);
   const gx=L.grid.x,gy=L.grid.y,cell=L.grid.cell,pitch=L.grid.pitch;
   const arrivedAll=d.copies.filter(c=>p>=c.arrive).length;
   const allArrived=arrivedAll>=8;
   const rankArrived=required.map(need=>need.filter(pos=>p>=copyByPos[pos].arrive).length);
   let started=0,done=0;
   d.queries.forEach((q,i)=>{
    const rowProgress=phase(p,q.computeStart,q.computeEnd);
    const stepCount=Math.min(i+1,Math.max(0,Math.ceil(rowProgress*(i+1)-1e-9)));
    if(p>=q.computeStart)started++;
    if(p>=q.computeEnd)done++;
    for(let c=0;c<=i;c++){
     const remote=d.owners[c]!==q.owner;
     const copy=copyByPos[c];
     const copyReady=!remote||p>=copy.arrive;
     const computed=stepCount>=c+1&&p>=q.computeStart&&copyReady;
     const cls=['pcpv-cell','pcpv-owner-'+q.owner];
     if(computed)cls.push('is-computed');
     else if(remote&&!copyReady)cls.push('is-remote','is-waiting');
     else if(remote)cls.push('is-remote','is-ready');
     else cls.push('is-local');
     if(i===6)cls.push('is-focus-row');
     set(cells[i*8+c].rect,{class:cls.join(' ')});
     const node=cells[i*8+c].rect;
     node.dataset.query=String(i);node.dataset.key=String(c);node.dataset.owner=String(q.owner);
     node.dataset.remote=String(remote);node.dataset.arrived=String(copyReady);
     node.dataset.computed=String(computed);node.dataset.ready=String(p>=q.computeStart);
    }
    const out=outs[i],outReady=p>=q.computeEnd;
    out.g.classList.toggle('is-final',outReady);
    out.g.setAttribute('opacity',outReady?'1':'.22');
    out.g.dataset.owner=String(q.owner);out.g.dataset.ready=String(outReady);
    counts[i].classList.toggle('is-focus',i===6);
   });
   q1Line2.textContent='当前：'+(p>=q1.computeEnd?'已计入 O1':p>=q1.computeStart?'计算中':'未开算');
   const q7Need=q7.requiredRemote.length;
   const q7Arrived=q7.requiredRemote.filter(pos=>p>=copyByPos[pos].arrive).length;
   const q7State=p>=q7.computeEnd?(m?'已计入':'依赖已到齐 · 已计入')
    :p>=q7.computeStart?(m?'计算中':'依赖已到齐 · 计算中')
    :q7Arrived>=q7Need?(m?'待开算':'依赖已到齐 · 待开算')
    :'远端 '+q7Arrived+'/'+q7Need+' 到齐';
   q7Line3.textContent='当前：'+q7State;
   for(let r=0;r<2;r++){
    const need=required[r],got=rankArrived[r];
    rankRemotes[r].textContent=need.length
     ?'依赖要求：'+rangeText(need)+'（'+need.length+' 份）· '+(got>=need.length?'已到齐 ':'未到齐 ')+got+'/'+need.length
     :'依赖要求：无远端'+(d.ranks[r].length?'（本行只读本地）':'');
    rankRemotes[r].classList.toggle('is-waiting',need.length>0&&got<need.length);
    rankRemotes[r].classList.toggle('is-local',need.length===0&&d.ranks[r].length>0);
   }
   focusRow.style.opacity=p<.6?'.45':'1';
   lanes.forEach(lane=>{
    let arrived=0;const needed=required[lane.to].length;
    lane.chips.forEach((c,j)=>{
     const u=phase(p,1.35,c.copy.arrive);
     const base=L.lanes[lane.from].x+L.lane.chipX+j*(L.lane.chipW+L.lane.gap);
     const offset=(1-u)*(m?12:20);
     place(c.g,base-offset,L.lanes[lane.from].y+3);
     const isArrived=u>=1;if(isArrived)arrived++;
     c.g.classList.toggle('is-arrived',isArrived);
     c.g.setAttribute('opacity',String(phase(p,1.15,1.35)*(c.needed?1:.58)));
     c.g.dataset.arrived=String(isArrived);
     c.g.dataset.position=String(c.copy.position+1);
     c.g.dataset.from=String(c.from);c.g.dataset.to=String(c.to);
     c.g.dataset.needed=String(c.needed);
    });
    lane.status.textContent=(arrived>=4?'全量 4/4':'到 '+arrived+'/4');
    lane.status.dataset.arrived=String(arrived);lane.status.dataset.needed=String(needed);
   });
   const ratio=arrivedAll/8;
   set(gatherFill,{width:(L.gather.w*ratio).toFixed(2)});
   gatherLabel.textContent=m
    ?'全量 '+arrivedAll+'/8 · '+(allArrived?'本图全量已到齐':'部分到达≠完成')
    :'AllGather 全量副本 '+arrivedAll+'/8 · '+(allArrived?'本图全量已到齐':'部分到达 ≠ 完成');
   if(allArrived){
    gatherNotes[0].textContent='当前：本图全量 8/8 已到齐；';
    gatherNotes[1].textContent='AllGather 已完成，无等待副本；';
    gatherNotes[2].textContent='条满 = 全量到齐，不是部分到达。';
   }else{
    gatherNotes[0].textContent='当前：部分到达 '+arrivedAll+'/8，未完成；';
    gatherNotes[1].textContent='本图按所需数据就绪开算；';
    gatherNotes[2].textContent='本地齐备 ≠ 全量副本到齐。';
   }
   gatherLabel.dataset.arrived=String(arrivedAll);
   gatherFill.dataset.arrived=String(arrivedAll);

   const phaseName=p<.35?'positions':p<1.25?'ownership':p<2.6?'gather':p<3.7?'compute':'output';
   let msg='';
   if(phaseName==='positions')msg=m?'8 个位置：Q1=1 对，Q8=8 对，总计 36 对。':'先看 8 个位置：Qi 的允许集合是 K1…Ki，所以 Q1=1 对、Q8=8 对，总计 36 对。';
   else if(phaseName==='ownership')msg=m
    ?(zigzag?'前后：R0 18 对 / R1 18 对；O_i 随 Q_i 归属。':'连续：R0 10 对 / R1 26 对；O_i 随 Q_i 归属。')
    :(zigzag?'前后配对：R0 负责 Q1/Q2/Q7/Q8（18 对），R1 负责 Q3–Q6（18 对），总 36 对。O_i 归 Q_i 的 rank。':'连续切分：R0 负责 Q1–Q4（10 对），R1 负责 Q5–Q8（26 对），总 36 对。O_i 归 Q_i 的 rank。');
   else if(phaseName==='gather')msg=m
    ?'副本 '+arrivedAll+'/8；Q7 等 '+q7Range+' 的副本到齐；部分到达≠完成。'
    :'全量副本 '+arrivedAll+'/8；Q7 仍需 '+q7Range+' 的副本到齐。部分副本到达 ≠ AllGather 完成。';
   else if(phaseName==='compute')msg=done>=8
    ?(m?'全量 8/8 已到齐；8 行已计入，完成 8/8。':'全量副本 8/8 已到齐；Q1–Q8 均已计入，完成 8/8。开算先后是教学顺序，不是实测耗时。')
    :(m?'Q1 本地齐备即算；Q7 等 '+q7Range+' 齐备才算；已开算 '+started+'/8，完成 '+done+'/8。'
     :'所需数据齐备后开算：Q1 只读本地 K1，无需等副本；Q7 等远端 '+q7Range+' 的副本到齐。已开算 '+started+'/8，已完成 '+done+'/8；开算先后是教学顺序，不是实测耗时。');
   else msg=m
    ?'O_i 随 Q_i 的 rank；不求和。36 是配对数，不是耗时。'
    :'O1…O8 留在各自 Query 的 rank；不跨 rank 求和。36 是配对数，不是耗时或加速比。';
   status.textContent=msg;
   status.dataset.phase=phaseName;

   svg.dataset.parallelKind='pcp';
   svg.dataset.pcpSplit=splitKey;
   svg.dataset.pcpWork=d.work.join('/');
   svg.dataset.pcpPairs=String(d.total);
   svg.dataset.pcpAllowed=String(d.total);
   svg.dataset.pcpMasked=String(cells.filter(cell=>cell.masked).length);
   svg.dataset.pcpSingle=String(d.single);
   svg.dataset.pcpArrived=String(arrivedAll);
   svg.dataset.pcpGatherComplete=String(arrivedAll>=8);
   svg.dataset.pcpStarted=String(started);
   svg.dataset.pcpDone=String(done);
   svg.dataset.pcpPhase=phaseName;
   svg.dataset.pcpQ7Remote=q7Remote.join(',');
   svg.dataset.pcpQ7Ready=String(p>=q7.computeStart);
   svg.dataset.pcpStatus=phaseName;
  }
  applyLayout(false);
  return(p,m)=>paint(p,m);
 });
 const destroy=inner.destroy;inner.destroy=()=>{film?.classList.remove('is-pcp');destroy();};
 return inner;
}
window.PCPMechanismVisual=visual;
})();
