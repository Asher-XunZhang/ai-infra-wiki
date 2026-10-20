/* Mechanism-specific pictures. Object nodes persist across playback and seeking.
 * Only the drawing changes continuously; lesson state is a pure function of p. */
(() => {
'use strict';
let boardSerial=0;
const NS='http://www.w3.org/2000/svg',clamp=n=>Math.max(0,Math.min(1,n));
const phase=(p,start,end=start+1)=>{const t=clamp((p-start)/(end-start));return t*t*(3-2*t);};
const lerp=(a,b,t)=>a+(b-a)*t;
function board(svg,viewport,kind,height,build){
 viewport.classList.add('teaching-viewport');viewport.setAttribute('aria-label','可回看的机制画布，窄屏自动排列');
 const node=(tag,attrs={},value)=>{const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(value!==undefined)e.textContent=value;svg.append(e);return e;};
 const markerId='teach-arrow-'+(++boardSerial),defs=node('defs'),marker=document.createElementNS(NS,'marker'),arrow=document.createElementNS(NS,'path');
 for(const[k,v]of Object.entries({id:markerId,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'}))marker.setAttribute(k,v);arrow.setAttribute('d','M0 0L10 5L0 10Z');arrow.setAttribute('fill','#7893a4');marker.append(arrow);defs.append(marker);
 const txt=(value,cls='teach-text')=>node('text',{class:cls},value);
 const rect=(cls='teach-panel')=>node('rect',{class:cls,rx:7});
 const path=(cls='teach-link')=>node('path',{class:cls,fill:'none',pathLength:1});
 const set=(e,a)=>{for(const[k,v]of Object.entries(a))e.setAttribute(k,v);};
 const show=(e,n)=>set(e,{opacity:clamp(n).toFixed(3),'aria-hidden':n<=0?'true':'false'});
 const text=(e,x,y,value)=>{set(e,{x,y});if(value!==undefined)e.textContent=value;};
 const box=(e,x,y,w,h)=>set(e,{x,y,width:w,height:h});
 const wire=(e,d,n=1)=>{set(e,{d,'stroke-dasharray':'1','stroke-dashoffset':1-clamp(n),'marker-end':n>=1?'url(#'+markerId+')':'none'});show(e,n>0?1:0);};
 function token(id,label,width=72,cls='teach-token'){
  const g=node('g',{'data-motion-id':id,class:cls}),r=document.createElementNS(NS,'rect'),t=document.createElementNS(NS,'text');
  set(r,{width,height:34,rx:5});set(t,{x:width/2,y:22,'text-anchor':'middle'});t.textContent=label;g.append(r,t);return {g,r,t,width};
 }
 const move=(a,x,y,opacity=1,scale=1)=>{set(a.g,{transform:`translate(${x.toFixed(2)} ${y.toFixed(2)}) scale(${scale})`});Object.assign(a.g.dataset,{x:x.toFixed(3),y:y.toFixed(3)});show(a.g,opacity);};
 let narrow=viewport.clientWidth<600,last=0;
 const paint=build({node,txt,rect,path,set,show,text,box,wire,token,move});
 function render(p){last=p;narrow=viewport.clientWidth<600;svg.setAttribute('viewBox',`0 0 ${narrow?360:720} ${height(narrow)}`);svg.dataset.teachingKind=kind;paint(p,narrow);}
 const observer=new ResizeObserver(()=>{narrow=viewport.clientWidth<600;render(last);});observer.observe(viewport);
 const repaintOnResize=()=>{narrow=viewport.clientWidth<600;render(last);};
 const resizeTimer=()=>setTimeout(repaintOnResize,50);
 const onResize=()=>{requestAnimationFrame(repaintOnResize);resizeTimer();};
 const mq=window.matchMedia('(max-width:650px)');const onMedia=()=>{requestAnimationFrame(repaintOnResize);resizeTimer();};
 window.addEventListener('resize',onResize,{passive:true});mq.addEventListener?.('change',onMedia);
 return {paint(_a,_b,_u,p){render(p);},destroy(){observer.disconnect();window.removeEventListener('resize',onResize);mq.removeEventListener?.('change',onMedia);}};
}
window.TeachingDrawing={board,phase,lerp,clamp};
const visuals=window.MechanismVisuals||(window.MechanismVisuals={});
visuals['f-tokens']=(svg,viewport,s)=>board(svg,viewport,'tokens',n=>n?1050:650,H=>{
 const {txt,rect,path,text,box,show,wire,token,move,set}=H,d=s.data,n=d.pieces.length;
 const unique=[...new Set(d.ids)],rowOf=id=>unique.indexOf(id),color=id=>'teach-color-'+rowOf(id);
 const labels=['原句 → 按分词规则拆开','① 查词表：片段 → ID','② 查 Embedding 表：ID → 一行数','③ 保持原来的出现顺序'];
 const heads=labels.map(x=>txt(x,'teach-heading'));
 const info=txt(n===5?'两个“学习” → 同一表行 → 两个输入位置':'同色对应同一词条；位置编号表示出现顺序','teach-small');
 const sentence=txt(d.text,'teach-heading');
 const pieces=d.pieces.map((w,i)=>token('piece-'+i,w,w.length*18+14,'teach-token '+color(d.ids[i])));
 const ids=d.ids.map((id,i)=>token('id-'+i,String(id),56,'teach-token '+color(id)));
 const positions=unique.map(id=>d.pieces.map((w,i)=>d.ids[i]===id?'p'+(i+1):null).filter(Boolean).join(' · '));
 const lookup=unique.map((id,i)=>({bg:rect('teach-row '+color(id)),word:txt(d.pieces[d.ids.indexOf(id)]),id:txt(String(id),'teach-number'),positions:txt(positions[i],'teach-small'),path:path()}));
 // Equal ids deliberately share equal teaching vectors; the table is not real weights.
 const vectors=d.ids.map(id=>id===100134?[.2,.8,-.1]:id===104100?[.5,-.2,.4]:id===1773?[-.1,.4,.6]:[.7,.1,-.3]);
 const embedding=unique.map((id,i)=>({bg:rect('teach-row '+color(id)),id:txt(String(id),'teach-number'),positions:txt(positions[i],'teach-small'),value:txt('['+vectors[d.ids.indexOf(id)].join(', ')+']','teach-number')}));
 const outputs=vectors.map((v,i)=>({t:token('vector-'+i,'['+v.join(', ')+']',130,'teach-token '+color(d.ids[i])),position:txt('位置 '+(i+1),'teach-small'),link:path()}));
 ids.forEach(a=>svg.append(a.g));
 return(p,m)=>{
  const W=m?360:720,dictY=196,embedY=m?496:196,outY=m?780:470;
  text(heads[0],18,28);text(heads[1],18,dictY-20);text(heads[2],m?18:376,embedY-20);text(heads[3],18,outY-20);text(info,18,m?1030:630);
  text(sentence,20,82);show(sentence,1-phase(p,0));let cursor=20;pieces.forEach((a,i)=>{const gap=phase(p,0)*10;move(a,cursor+i*gap,58,phase(p,0));cursor+=a.width;set(a.r,{opacity:phase(p,0)*.85+.15});});
  lookup.forEach((a,i)=>{const y=dictY+i*48;box(a.bg,18,y,m?324:326,40);text(a.word,30,y+25);text(a.positions,m?158:158,y+25);text(a.id,m?252:254,y+25);show(a.bg,.18+.47*phase(p,1));show(a.word,.35+.65*phase(p,1));show(a.positions,.35+.65*phase(p,1));set(a.id,{'text-anchor':'middle',x:(m?248:250)+38});show(a.id,.35+.65*phase(p,2));
   wire(a.path,`M${m?286:288} ${y+42} ${m?'V'+(y+45):'H370'}`,phase(p,1));
  });
  // The occurrence IDs stay in a fixed rail; they never cover the table row
  // that they are querying. Repeated occurrences point at the same row by identity.
  ids.forEach((a,i)=>{const row=rowOf(d.ids[i]);move(a,18+i*60,dictY-76,phase(p,1));a.g.dataset.tableRow=String(row);});

  embedding.forEach((a,i)=>{const x=m?18:376,y=embedY+i*48;box(a.bg,x,y,m?324:326,40);text(a.id,x+12,y+25);text(a.positions,x+78,y+25);text(a.value,x+140,y+25);show(a.bg,.16+.54*phase(p,2));show(a.id,.22+.78*phase(p,2));show(a.positions,.25+.75*phase(p,2));show(a.value,.25+.75*phase(p,2));});
  outputs.forEach((a,i)=>{const x=m?94:28+i*137,y=m?outY+i*44:outY+18,sourceX=(m?18:376)+116,sourceY=embedY+rowOf(d.ids[i])*48+3,t=phase(p,3);
   move(a.t,lerp(sourceX,x,t),lerp(sourceY,y,t),t);text(a.position,m?18:x,m?y+22:outY+8);show(a.position,t);
   wire(a.link,`M${x+65} ${y+34} v6`,t);
  });
  // The first row of the embedding table is visible only when it is actually used.
  [heads[1],heads[2],heads[3]].forEach((e,i)=>show(e,.55+.45*phase(p,i)));
 };
});
visuals['f-generation']=(svg,viewport,s)=>board(svg,viewport,'generation',m=>m?940:810,H=>{
 const {txt,rect,path,wire,text,box,show,token,move,set}=H,d=s.data;
 const model=rect(),out=rect(),cache=rect(),mt=txt('同一个模型 · 权重不变','teach-heading'),ot=txt('输出序列','teach-heading'),ct=txt('历史 KV · 保留已计算的输入','teach-small');
 const mode=txt('','teach-small'),caption=txt('','teach-small'),dist=txt('下一 token 的候选','teach-small'),client=txt('','teach-small');
 const inputs=['我','想'].map((w,i)=>token('input-'+i,w,60,'teach-token teach-color-0'));
 const selectedColor=d.scores.findIndex(r=>r.word===d.first);
 const first=token('output-y1',d.first,76,'teach-token teach-color-'+selectedColor),copy=token('input-y1',d.first,76,'teach-token teach-color-'+selectedColor),second=token('output-y2',d.second,76,'teach-token teach-color-3');
 const cached=['我','想',d.first].map((w,i)=>token('kv-'+i,w+' 的 KV',94,'teach-token teach-color-'+(i===2?selectedColor:0)));
 const rows=Array.from({length:3},()=>({name:txt('','teach-small'),bar:rect('teach-value'),value:txt('','teach-number')}));
 const selection=txt('','teach-heading'),explanation=txt('','teach-small'),rangeLabels=Array.from({length:4},()=>txt('','teach-small'));
 const intervals=Array.from({length:3},(_,i)=>rect('teach-contribution teach-color-'+i)),intervalNames=intervals.map(()=>txt('','teach-small')),needle=path('teach-map');
 return(p,m)=>{const W=m?360:720,my=90,oy=m?690:540,ky=oy+130,cx=m?38:398,cy=m?310:166,choice=p<4?d.choice:d.nextChoice,round=p<4?0:1,distStart=round?4:1,selectStart=round?5:2;
  box(model,16,my,W-32,m?570:400);box(out,16,oy,W-32,106);box(cache,16,ky,W-32,110);text(mt,30,my+28);text(ot,30,oy+28);text(ct,30,ky+28);text(mode,30,my+58,p<1?'准备输入':p<4?'Prefill · 计算输入位置':'Decode · 计算新位置');
  inputs.forEach((t,i)=>move(t,lerp(32+i*78,44+i*78,phase(p,0)),lerp(16,my+100,phase(p,0)),1-phase(p,3)));
  text(dist,cx,cy-16);show(dist,.3+.7*phase(p,1));rows.forEach((r,i)=>{const y=cy+i*42,write=phase(p,4),prob=p<4?d.scores[i].p:i===0?.8:i===1?.2:0;const u=p<4?phase(p,1):write,rowVis=.3+.7*phase(p,1);
   text(r.name,cx,y+14,p<4?d.scores[i].word:i===0?d.second:i===1?'其他':'');box(r.bar,cx+54,y,148*prob*u,18);text(r.value,cx+210,y+14,u===1?Math.round(prob*100)+'%':p<1?'待计算':'计算中');[r.name,r.bar,r.value].forEach(e=>show(e,rowVis*(p>=4&&i===2?0:1)));
   set(r.bar,{class:'teach-value teach-color-'+(round?i+3:i)});});
  const sample=choice.draw!==null,bx=38,bw=W-76,by=m?490:348,available=phase(p,distStart+.65,distStart+1),chosenItem=choice.intervals.find(x=>x.word===choice.selected);
  text(selection,bx,by-30,sample?'按概率划分 0–1，再抽一个数':'贪心选择：取本轮概率最大的候选');show(selection,available);
  let edges=choice.intervals.map(x=>x.start).concat(1);
  intervals.forEach((bar,i)=>{const item=choice.intervals[i];if(!item){show(bar,0);show(intervalNames[i],0);return;}set(bar,{class:'teach-contribution teach-color-'+(round?i+3:i)});box(bar,bx+item.start*bw,by,item.p*bw,26);show(bar,sample?available:0);text(intervalNames[i],bx+item.start*bw+5,by+18,item.word);show(intervalNames[i],sample?available:0);});
  rangeLabels.forEach((t,i)=>{text(t,bx+(edges[i]??0)*bw-4,by+47,edges[i]===undefined?'':Number(edges[i].toFixed(2)).toString());show(t,sample?available:0);});
  // A draw is a given random value, not a scan across possible candidates.
  wire(needle,`M${bx+(choice.draw??0)*bw} ${by-20} V${by-3}`,sample?phase(p,selectStart,selectStart+.3):0);
  text(explanation,bx,by+75,sample?`抽样数 u=${choice.draw} 落在 ${chosenItem.start.toFixed(1)}–${chosenItem.end.toFixed(1)} → ${choice.selected}`:`${Math.round(chosenItem.p*100)}% 是最大值 → ${choice.selected}`);show(explanation,phase(p,selectStart,selectStart+.3));
  const selected=phase(p,2.5,2.9),feedback=phase(p,3),selected2=phase(p,5.5,5.9);
  // The selected output has a fixed landing tray. It fades in there rather than
  // travelling across the probability explanation and interval labels.
  move(first,40,oy+48,selected);
  move(copy,lerp(40,44,feedback),lerp(oy+48,my+100,feedback),p>3?1:0);
  move(second,140,oy+48,selected2);
  cached.forEach((t,i)=>{const u=i<2?phase(p,1):phase(p,4);move(t,26+i*(m?104:190),ky+48,u);t.g.dataset.written=u.toFixed(3);});
  text(client,30,oy+92,p>=3?'客户端已见：'+d.visible:'客户端尚未收到可见文字');show(client,phase(p,3));
  text(caption,30,my+(m?544:374),p<3?'概率来自本例给定分布；不是正确率。':p<5?'同一 token ID 反馈为新输入。':'输出选择后，KV 仍只覆盖已计算输入。');
  svg.dataset.selection=choice.selected;svg.dataset.selectionDraw=choice.draw??'greedy';
 };
});
visuals['f-attention']=(svg,viewport,s)=>board(svg,viewport,'attention',m=>m?1110:940,H=>{
 const {txt,rect,path,text,box,show,wire,token,move,set}=H,d=s.data;
 const title=txt('','teach-heading'),q=token('query','Q = ['+d.Q.join(', ')+']',144,'teach-token teach-color-4');
 const scoreExample=txt('','teach-small'),transform=txt('本例 Q=当前位置输入；K=输入；V=2×输入','teach-small');
 const normalization=txt('Softmax：先取指数，再除以总和','teach-heading'),denominator=txt('','teach-number'),worked=txt('','teach-number'),rounding=txt('exp(s) = e 的 s 次方；显示值四舍五入。','teach-small');
 const exps=d.scores.map((v,i)=>i<=d.query?Math.exp(v):0),total=exps.reduce((a,b)=>a+b,0);
 const legend=txt('灰色行：未来位置，不参与；同色段：同一位置的贡献','teach-small');
 const heads=['输入 / K','Q·K / √2','权重','V'].map(x=>txt(x,'teach-small'));
 const rows=d.keys.map((k,i)=>({bg:rect(i>d.query?'teach-mask':'teach-row teach-color-'+i),pos:txt('位置 '+(i+1),'teach-small'),key:token('key-'+i,'['+k.join(', ')+']',66,'teach-token teach-color-'+i),score:txt('','teach-number'),exp:txt('','teach-small'),track:rect('teach-track'),bar:rect('teach-value teach-color-'+i),weight:txt('','teach-number'),v:token('value-'+i,'['+d.values[i].join(', ')+']',66,'teach-token teach-color-'+i),weighted:txt('','teach-small')}));
 const resultTitle=txt('结果的两个维度，分别相加','teach-heading'),result=[txt('','teach-number'),txt('','teach-number')];
 const contributions=d.weights.map((w,i)=>[0,1].map(j=>({bar:rect('teach-contribution teach-color-'+i),number:txt('','teach-small teach-contribution-number')})));
 const sum=txt('','teach-heading'),equation=txt('','teach-small'),equation2=txt('','teach-small'),equation3=txt('','teach-small'),equation4=txt('','teach-small');
 return(p,m)=>{
  text(title,18,28,`观察位置 ${d.query+1}：只读取当前位置及之前`);move(q,m?18:32,48,phase(p,0));text(transform,18,106);text(legend,18,148);if(m)legend.setAttribute('font-size','11');
  text(scoreExample,18,128,`位置 ${d.query+1}：(${d.Q[0]}×${d.keys[d.query][0]} + ${d.Q[1]}×${d.keys[d.query][1]}) / √2 = ${d.scores[d.query].toFixed(2)}`);show(scoreExample,phase(p,1));
  heads.forEach((e,i)=>{text(e,m?[24,124,222,290][i]:[28,216,394,626][i],174);show(e,phase(p,0));});
  const rowH=m?126:90;
  rows.forEach((r,i)=>{const y=188+i*rowH,allowed=i<=d.query;box(r.bg,16,y,m?328:688,rowH-8);text(r.pos,26,y+21);move(r.key,26,y+30,phase(p,0)*(allowed?1:.22));
   text(r.score,m?124:218,y+52,allowed?(phase(p,1)===1?d.scores[i].toFixed(2):'计算中'):'遮住');show(r.score,phase(p,1));
   text(r.exp,m?108:218,y+(m?74:76),allowed?`exp(${d.scores[i].toFixed(2)})≈${exps[i].toFixed(2)}`:'不计入分母');show(r.exp,phase(p,2,2.35));
   const bx=m?108:358,by=m?y+83:y+35,bw=m?206:180;box(r.track,bx,by,bw,14);box(r.bar,bx,by,bw*d.weights[i]*phase(p,2.65,3),14);text(r.weight,m?222:550,m?y+52:y+49,phase(p,2.65,3)===1?(d.weights[i]*100).toFixed(1)+'%':'归一化中');[r.track,r.bar,r.weight].forEach(e=>show(e,phase(p,2.65,3)));
   move(r.v,m?274:626,y+30,phase(p,0)*(allowed?1:.22));
   text(r.weighted,m?108:358,y+(m?113:77),p>=3?`${(d.weights[i]*100).toFixed(1)}% × V ≈ [${d.values[i].map(x=>(x*d.weights[i]).toFixed(2)).join(', ')}]`:'本行权重 × V');show(r.weighted,allowed?phase(p,3,3.35):0);
  });
  const ny=m?716:574,outY=ny+132;text(normalization,18,ny);show(normalization,phase(p,2,2.35));
  text(denominator,18,ny+30,'总和 ≈ '+exps.slice(0,d.query+1).map(x=>x.toFixed(2)).join(' + ')+' ≈ '+total.toFixed(2));show(denominator,phase(p,2.35,2.65));
  text(worked,18,ny+58,p>=3?`位置 ${d.query+1}：${exps[d.query].toFixed(2)} ÷ ${total.toFixed(2)} ≈ ${(d.weights[d.query]*100).toFixed(1)}%`:'每行的指数值 ÷ 同一个总和');show(worked,phase(p,2.65,3));text(rounding,18,ny+82);show(rounding,phase(p,2));if(m)denominator.setAttribute('font-size','13');
  const sumLines=[];
  for(let j=0;j<2;j++){
   const terms=d.weights.slice(0,d.query+1).map((w,i)=>(w*d.values[i][j]).toFixed(2));
   const first=terms.slice(0,2),rest=terms.slice(2);
   sumLines.push(`维度${j+1}：${first.join(' + ')}`);
   sumLines.push(`${rest.length?'+ '+rest.join(' + ')+' ':''}≈ ${d.mixed[j].toFixed(2)}`);
  }
  text(resultTitle,18,outY);[equation,equation2,equation3,equation4].forEach((e,i)=>text(e,18,outY+16+i*16,p>=3?sumLines[i]:i===0?'每个 V × 自己的权重，再按维度相加':''));show(resultTitle,phase(p,3));[equation,equation2,equation3,equation4].forEach(e=>show(e,phase(p,3)));
  for(let j=0;j<2;j++){
   let start=90;const unit=m?72:150,y=outY+84+j*56;text(result[j],18,y+19,'维度 '+(j+1));show(result[j],phase(p,3));
   contributions.forEach((parts,i)=>{const part=parts[j],value=d.weights[i]*d.values[i][j],width=value*unit*phase(p,3.35,4);
    // Multiplication is visible in each source row first. Then the same-color
    // lengths accumulate directly in the sum: no invented transport or routing.
    box(part.bar,start,y,width,25);show(part.bar,value>0?1:0);
    text(part.number,start+3,y+18,width>30&&p>=4?value.toFixed(2):'');show(part.number,p>=4?1:0);start+=width;
   });
  }
  text(sum,18,outY+214,phase(p,3)===1?'汇合得到 ['+d.mixed.map(x=>x.toFixed(2)).join(', ')+']':'正在按维度汇合 · 条形为绘制进度');show(sum,phase(p,3));
 };
});

function cachePages(svg,viewport,s){return board(svg,viewport,'cache-pages',m=>m?560:430,H=>{
 const {txt,rect,text,box,show,token,move,wire}=H;
 const title=txt('逻辑位置 → 页表 → 物理页','teach-heading'),pool=txt('物理页池','teach-small'),logic=txt('R1 的逻辑位置','teach-small');
 const page7=rect(),page2=rect();
 const p7=[0,1].map(i=>token('page7-slot-'+i,'空',44,'teach-token teach-empty')),p2=[0,1].map(i=>token('page2-slot-'+i,'空',44,'teach-token teach-empty'));
 const positions=[1,2,3].map(i=>token('page-pos-'+i,'p'+i,44,'teach-token teach-color-0'));
 const p7Label=txt('页 7 · 2 槽','teach-small'),p2Label=txt('页 2 · 2 槽','teach-small'),note=txt('页号不要求连续；逻辑位置仍按原顺序。','teach-small'),note2=txt('页 7 / 页 2 是物理位置；p1…p3 是请求的位置序号。','teach-small');
 return(p,m)=>{const W=m?360:720,x=m?24:360;text(title,18,28);text(pool,28,64);text(logic,m?24:360,m?330:64);
  box(page7,24,84,m?312:198,116);text(p7Label,36,108);p7.forEach((t,i)=>move(t,44+i*54,132,1));
  box(page2,24,220,m?312:198,116);text(p2Label,36,244);p2.forEach((t,i)=>move(t,44+i*54,268,1));
  positions.forEach((t,i)=>move(t,m?44+i*70:400, m?360+i*54:110+i*54,1));
  if(p>=1){p7[0].t.textContent='p1';p7[0].g.classList.remove('teach-empty');p7[0].g.classList.add('teach-color-0');}
  if(p>=2){p7[1].t.textContent='p2';p7[1].g.classList.remove('teach-empty');p7[1].g.classList.add('teach-color-0');}
  if(p>=3){p2[0].t.textContent='p3';p2[0].g.classList.remove('teach-empty');p2[0].g.classList.add('teach-color-0');}
  // The page table is represented by the stable p1…p3 cards and page slots;
  // no extra transfer lane is needed for this local lookup.

  text(note,18,m?520:390);if(m)note.setAttribute('font-size','11');
  text(note2,18,m?492:366);
 };
});}
function cacheShared(svg,viewport,s){return board(svg,viewport,'cache-shared',m=>m?500:380,H=>{
 const {txt,rect,path,text,box,show,token,move,wire}=H;
 const r1Box=rect(),r2Box=rect(),sharedBox=rect();
 const title=txt('两条请求引用同一份物理 K/V','teach-heading'),r1=txt('R1','teach-heading'),r2=txt('R2','teach-heading');
 const r1State=txt('','teach-small'),r2State=txt('','teach-small'),shared=txt('共享历史：同一份 K/V','teach-small');
 const note1=txt('共享是引用，不是复制；R1 结束后 R2 仍持有。','teach-small'),note2=txt('','teach-small');
 const kv=token('shared-kv','p1 · p2 · p3',148,'teach-token teach-color-0');
 const link1=path('teach-link'),link2=path('teach-link');
 return(p,m)=>{const W=m?360:720;text(title,18,28);box(r1Box,20,70,m?W-40:210,82);box(r2Box,m?20:270,70,m?W-40:210,82);
  text(r1,m?32:32,96);text(r2,m?32:282,96);text(r1State,m?32:32,126,p>=3?'R1 已结束':p>=1?'持有共享页':'尚未建立');text(r2State,m?32:282,126,p>=2?(p>=4?'R2 已结束':'R2 持有共享页'):'尚未建立');
  box(sharedBox,m?20:270,m?220:220,m?W-40:210,82);text(shared,m?32:282,m?246:246);move(kv,m?42:300,m?260:260,phase(p,1));
  wire(link1,m?`M120 152 V220` :`M120 152 V220`,p>=1&&p<3?1:0);wire(link2,m?`M120 152 V220`:`M370 152 V220`,p>=2&&p<4?1:0);
  text(note1,18,m?360:330);text(note2,18,m?390:350,p>=4?'没有活跃请求，但缓存仍可按策略保留。':'覆盖前必须确认没有引用。');
 };
});}
visuals['f-cache']=(svg,viewport,s)=>s.kind==='cache-pages'?cachePages(svg,viewport,s):s.kind==='cache-shared'?cacheShared(svg,viewport,s):board(svg,viewport,'cache',m=>m?1100:680,H=>{
 const {txt,rect,text,box,show,token,move,set}=H;
 const panels=[rect(),rect()],titles=[txt('不跨轮保留 K/V','teach-heading'),txt('保存历史 K/V','teach-heading')];
 const rows=[0,1].map(side=>[0,1,2].map(r=>({heading:txt('第 '+(r+1)+' 次调用','teach-small'),cells:Array.from({length:r+4},(_,i)=>({a:token(`work-${side}-${r}-${i}`,i<4?'p'+(i+1):'y'+(i-3),42,'teach-token'),fill:rect('teach-compute'),note:txt('','teach-small')}))})));
 const totals=[txt('','teach-heading'),txt('','teach-heading')],reads=[txt('','teach-small'),txt('','teach-small')],countNotes=[txt('','teach-small'),txt('','teach-small')],poolTitle=txt('右侧保留的一份历史 · 没有每轮复制一套缓存','teach-small');
 const pool=Array.from({length:6},(_,i)=>token('stored-'+i,i<4?'p'+(i+1):'y'+(i-3),42,'teach-token teach-color-0'));
 const note=txt('','teach-small'),identity=txt('p1–p4：原输入；y1、y2：后续新输入','teach-small');
 return(p,m)=>{
  const ys=m?[24,494]:[24,24],xs=m?[16,16]:[16,370],pw=m?328:334;
  [0,1].forEach(side=>{box(panels[side],xs[side],ys[side],pw,438);text(titles[side],xs[side]+14,ys[side]+30);
   let computed=0;rows[side].forEach((r,j)=>{const y=ys[side]+74+j*104,t=phase(p,j),active=p>=j&&p<j+1;text(r.heading,xs[side]+14,y-12);
    r.cells.forEach(({a,fill,note},i)=>{const x=xs[side]+14+i*49,isReuse=side===1&&j>0&&i<j+3,count=side===0||j===0?j+4:1;
     move(a,x,y,p<j?.28:1);a.g.setAttribute('class','teach-token teach-color-'+i%5);set(a.g,{style:i===5?'--ink:#607d8b;--tint:#e7eef2':''});set(a.r,{opacity:p<=j?.18:1});box(fill,x,y+38,42*t,5);fill.setAttribute('class',isReuse?'teach-read':'teach-compute');text(note,x,y+65,p<=j?'待算':isReuse?'复用':'新算');show(note,p>j?1:.4);a.g.dataset.progress=t.toFixed(3);
     if(i===0)computed+=t===1?count:0;
    });
   });const terms=Array.from({length:Math.min(3,Math.floor(p))},(_,r)=>side===0?4+r:r===0?4:1);
   const readTotal=side===1?(p>=3?9:p>=2?4:0):0;
   text(reads[side],xs[side]+14,ys[side]+368,side===0?'读取历史：0 个位置':`读取历史：${p>=3?'4 + 5 = 9':p>=2?'4':'0'} 个位置`);
   text(totals[side],xs[side]+14,ys[side]+392,`累计新算：${terms.join(' + ')||'0'} = ${computed}`);
   text(countNotes[side],xs[side]+14,ys[side]+420,p%1>0&&p<3?'本轮尚未完成，暂不计入累计。':'单位：K/V 投影的位置数。');totals[side].dataset.computed=String(computed);
  });
  const py=m?982:534;text(poolTitle,18,py-15);pool.forEach((a,i)=>{a.g.setAttribute('class','teach-token teach-color-'+i%5);set(a.g,{style:i===5?'--ink:#607d8b;--tint:#e7eef2':''});move(a,24+i*49,py,phase(p,i<4?0:i-3));});text(identity,18,m?1050:620);text(note,18,m?1074:644,'下划线：橙 = 新算，绿 = 复用；格色对应输入位置');
 };
});
visuals['f-deployment']=(svg,viewport,s)=>board(svg,viewport,'deployment',m=>m?960:640,H=>{
 const {txt,rect,path,text,box,show,wire,token,move}=H;
 const panels=[rect(),rect()],heads=[txt('Prefill 角色','teach-heading'),txt('Decode 角色','teach-heading')],models=[txt('可执行完整模型','teach-small'),txt('可执行完整模型','teach-small')];
 const src=[0,1].map(i=>token('source-kv-'+i,'位置 '+(i+1)+' KV',100,'teach-token teach-color-'+i));
 const dest=[0,1].map(i=>token('target-kv-'+i,'等待 KV',100,'teach-token teach-empty'));
 const copies=[0,1].map(i=>token('transfer-kv-'+i,'位置 '+(i+1)+' KV',100,'teach-token teach-color-'+i));
 const first=token('first-token','y1',52,'teach-token teach-color-2'),input=token('decode-input','y1',52,'teach-token teach-color-2'),added=token('decode-kv-y1','y1 的 KV',100,'teach-token teach-color-2'),second=token('second-token','y2',52,'teach-token teach-color-4');
 const notes=[txt('','teach-small'),txt('','teach-small')],requirements=txt('','teach-small'),total=txt('','teach-small'),flow=path(),label=txt('交接：历史 KV + 下一轮输入 ID','teach-small'),result=txt('','teach-heading');
 return(p,m)=>{const W=m?360:720,pos=r=>({x:m?16:16+r*366,y:m?60+r*410:60}),a=pos(0),b=pos(1);
  panels.forEach((v,r)=>{const q=pos(r);box(v,q.x,q.y,m?328:322,350);text(heads[r],q.x+16,q.y+30);text(models[r],q.x+16,q.y+60);text(notes[r],q.x+16,q.y+326,r===0?(p<1?'计算 prompt':p<3?'保留已有 KV':'源 KV 不随复制而搬走'):(p<3?'所需输入未齐 → 等待':p<4?'输入已齐 → 处理 y1':'已追加 y1 的 KV'));});
  text(requirements,b.x+16,b.y+278,p<3?'收到的前缀 0/2；y1 未到':'收到的前缀 2/2；y1 已到');text(total,b.x+16,b.y+298,p>=3?'本地新增 y1 KV 1 · 当前可用 3/3':'当前可用 0/3');requirements.dataset.ready=String(p>=3);
  src.forEach((t,i)=>move(t,a.x+20,a.y+96+i*66,phase(p,0)));dest.forEach((t,i)=>{move(t,b.x+20,b.y+96+i*66);show(t.g,1-phase(p,2));});
  const u=phase(p,2);copies.forEach((t,i)=>{move(t,lerp(a.x+20,b.x+20,u),lerp(a.y+96+i*66,b.y+96+i*66,u),p>2?1:0);t.g.dataset.received=u.toFixed(3);});
  move(first,a.x+20,a.y+242,phase(p,1));move(input,lerp(a.x+20,b.x+20,u),lerp(a.y+242,b.y+242,u),p>2?1:0);
  move(added,b.x+170,b.y+162,phase(p,3));move(second,b.x+170,b.y+242,phase(p,4));
  wire(flow,m?`M${a.x+280} ${a.y+240} V${b.y+90}`:`M${a.x+300} ${a.y+190} H${b.x+12}`,p>2&&p<3?1:0);
  text(label,18,m?450:460);text(result,18,m?910:520,p<3?'历史 KV 与新输入齐备，才能 Decode':p<4?'D 使用已有历史计算新输入':'用同一请求的历史继续生成');if(m)result.setAttribute('font-size','16');
 };
});
})();
