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
 function render(p){last=p;svg.setAttribute('viewBox',`0 0 ${narrow?360:720} ${height(narrow)}`);svg.dataset.teachingKind=kind;paint(p,narrow);}
 const observer=new ResizeObserver(()=>{narrow=viewport.clientWidth<600;render(last);});observer.observe(viewport);
 return {paint(_a,_b,_u,p){render(p);},destroy(){observer.disconnect();}};
}
window.TeachingDrawing={board,phase,lerp,clamp};
const visuals=window.MechanismVisuals||(window.MechanismVisuals={});
visuals['f-tokens']=(svg,viewport,s)=>board(svg,viewport,'tokens',n=>n?1050:650,H=>{
 const {txt,rect,path,text,box,show,wire,token,move,set}=H,d=s.data,n=d.pieces.length;
 const labels=['原句 → 按分词规则拆开','① 查词表：片段 → ID','② 查 Embedding 表：ID → 一行数','③ 保持原来的出现顺序'];
 const heads=labels.map(x=>txt(x,'teach-heading'));
 const info=txt(n===5?'重复 ID 查同一行；图中按出现位置展开':'仅展示本例用到的词表 / Embedding 条目','teach-small');
 const sentence=txt(d.text,'teach-heading');
 const pieces=d.pieces.map((w,i)=>token('piece-'+i,w,w.length*18+14,'teach-token teach-color-'+i%5));
 const ids=d.ids.map((id,i)=>token('id-'+i,String(id),76,'teach-token teach-color-'+i%5));
 const lookup=d.pieces.map((w,i)=>({bg:rect('teach-row teach-color-'+i%5),word:txt(w),id:txt(String(d.ids[i]),'teach-number'),path:path()}));
 // Equal ids deliberately share equal teaching vectors; the table is not real weights.
 const vectors=d.ids.map(id=>id===100134?[.2,.8,-.1]:id===104100?[.5,-.2,.4]:id===1773?[-.1,.4,.6]:[.7,.1,-.3]);
 const embedding=vectors.map((v,i)=>({bg:rect('teach-row teach-color-'+i%5),id:txt(String(d.ids[i]),'teach-number'),value:txt('['+v.join(', ')+']','teach-number')}));
 const outputs=vectors.map((v,i)=>({t:token('vector-'+i,'['+v.join(', ')+']',130,'teach-token teach-color-'+i%5),position:txt('位置 '+(i+1),'teach-small'),link:path()}));
 ids.forEach(a=>svg.append(a.g));
 return(p,m)=>{
  const W=m?360:720,dictY=160,embedY=m?460:160,outY=m?780:470;
  text(heads[0],18,28);text(heads[1],18,dictY-20);text(heads[2],m?18:376,embedY-20);text(heads[3],18,outY-20);text(info,18,m?1030:630);
  text(sentence,20,82);show(sentence,1-phase(p,0));let cursor=20;pieces.forEach((a,i)=>{const gap=phase(p,0)*10;move(a,cursor+i*gap,58,phase(p,0));cursor+=a.width;set(a.r,{opacity:phase(p,0)*.85+.15});});
  lookup.forEach((a,i)=>{const y=dictY+i*48;box(a.bg,18,y,m?324:326,40);text(a.word,30,y+25);text(a.id,m?252:254,y+25);show(a.bg,phase(p,1)*.65);show(a.word,phase(p,1));set(a.id,{'text-anchor':'middle',x:(m?248:250)+38});show(a.id,phase(p,2));
   const destX=m?30:386,destY=embedY+i*48+3,fromX=m?248:250;
   move(ids[i],lerp(fromX,destX,phase(p,2)),lerp(y+3,destY,phase(p,2)),phase(p,1));
   wire(a.path,`M${m?286:288} ${y+42} ${m?'V'+(y+45):'H370'}`,phase(p,1));
  });
  embedding.forEach((a,i)=>{const x=m?18:376,y=embedY+i*48;box(a.bg,x,y,m?324:326,40);text(a.id,x+12,y+25);text(a.value,x+116,y+25);show(a.bg,phase(p,2)*.7);show(a.id,0);show(a.value,phase(p,2));});
  outputs.forEach((a,i)=>{const x=m?94:28+i*137,y=m?outY+i*44:outY+18,sourceX=(m?18:376)+116,sourceY=embedY+i*48+3,t=phase(p,3);
   move(a.t,lerp(sourceX,x,t),lerp(sourceY,y,t),t);text(a.position,m?18:x,m?y+22:outY+8);show(a.position,t);
   wire(a.link,`M${x+65} ${y+34} v6`,t);
  });
  // The first row of the embedding table is visible only when it is actually used.
  [heads[1],heads[2],heads[3]].forEach((e,i)=>show(e,phase(p,i)));
 };
});
visuals['f-generation']=(svg,viewport,s)=>board(svg,viewport,'generation',m=>m?820:670,H=>{
 const {txt,rect,path,text,box,show,wire,token,move,set}=H,d=s.data;
 const client=rect(),model=rect(),cache=rect(),clientTitle=txt('客户端 · 已看到的文字','teach-heading'),modelTitle=txt('同一个模型 · 权重保持不变','teach-heading'),cacheTitle=txt('历史记录 · 只保存已经计算过的输入位置','teach-heading');
 const promptLabel=txt('输入 · 已完成分词','teach-small');
 const mode=txt('','teach-badge'),inputNote=txt('本轮输入','teach-small'),distribution=txt('下一 token 的候选 · 教学分布','teach-small');
 const inputs=['我','想'].map((w,i)=>token('input-'+i,w,64,'teach-token teach-color-0'));
 const first=token('output-y1',d.first,76,'teach-token teach-color-1'),second=token('output-y2',d.second,76,'teach-token teach-color-3'),copy=token('input-y1',d.first,76,'teach-token teach-color-1');
 const cached=['我','想',d.first].map((w,i)=>({t:token('kv-'+i,w+' 的记录',94,'teach-token teach-color-'+(i===2?1:0)),cover:rect('teach-write-cover')}));
 const rows=Array.from({length:3},()=>({name:txt('','teach-small'),track:rect('teach-track'),bar:rect('teach-value'),value:txt('','teach-number')}));
 const feedback=path(),returnPath=path(),readPath=path();
 return(p,m)=>{
  const W=m?360:720,clientY=50,modelY=210,cacheY=m?650:500,candidateX=m?38:414,candidateY=m?400:268;
  box(client,16,clientY,W-32,128);box(model,16,modelY,W-32,m?402:250);box(cache,16,cacheY,W-32,130);
  text(clientTitle,30,clientY+28);text(modelTitle,130,modelY+28,m?'模型计算 · 权重不变':'同一个模型 · 权重保持不变');text(cacheTitle,30,cacheY+28);if(m)cacheTitle.setAttribute('font-size','14');
  text(mode,130,modelY+58,p<1?'输入就绪':p<4?'Prefill · 处理已有输入':p<5?'准备下一轮':'Decode · 处理新位置');text(inputNote,130,modelY+94);
  const enter=phase(p,0),secondRound=phase(p,4),write=phase(p,5);text(promptLabel,204,30);show(promptLabel,1-enter);
  inputs.forEach((a,i)=>move(a,lerp(36+i*80,44+i*80,enter),lerp(8,modelY+110,enter),1-secondRound));
  text(distribution,candidateX,candidateY-16);show(distribution,phase(p,1));
  rows.forEach((r,i)=>{const y=candidateY+i*43,prob=p<5?d.scores[i].p:i===0?.8:i===1?.2:0,label=p<5?d.scores[i].word:i===0?d.second:i===1?'其他':'';
   text(r.name,candidateX,y+13,label);box(r.track,candidateX+52,y,m?148:160,17);box(r.bar,candidateX+52,y,(m?148:160)*prob*(p<5?phase(p,1):write),17);text(r.value,candidateX+(m?210:224),y+14,(p<5?phase(p,1):write)===1?Math.round(prob*100)+'%':'计算中');[r.name,r.track,r.bar,r.value].forEach(e=>show(e,phase(p,1)*(i===2&&p>=5?0:1)));});
  const selectedX=candidateX+52,selectedY=candidateY+135,clientX=m?54:418,clientTokenY=clientY+60,toClient=phase(p,3);
  move(first,lerp(selectedX,clientX,toClient),lerp(selectedY,clientTokenY,toClient),phase(p,2));
  move(copy,lerp(clientX,44,clamp(secondRound*2)),lerp(clientTokenY,modelY+110,clamp(secondRound*2-1)),secondRound);
  move(second,lerp(selectedX,clientX+90,phase(p,7)),lerp(selectedY,clientTokenY,phase(p,7)),phase(p,6));
  cached.forEach((a,i)=>{const x=m?26+i*103:72+i*190,y=cacheY+56,t=i<2?phase(p,1):write;move(a.t,x,y,t);box(a.cover,x,y+38,94*t,5);a.cover.setAttribute('class','teach-compute');show(a.cover,t);a.t.g.dataset.written=t.toFixed(3);});
  wire(feedback,`M${clientX+38} ${clientTokenY+34} H82 V${modelY+110}`,secondRound);
  wire(returnPath,`M${selectedX+38} ${selectedY} H${W-25} V${clientTokenY+17} H${clientX+76}`,toClient);
  wire(readPath,`M${W-44} ${cacheY} V${modelY+(m?390:238)} H${m?130:200}`,write);
 };
});
visuals['f-attention']=(svg,viewport,s)=>board(svg,viewport,'attention',m=>m?930:760,H=>{
 const {txt,rect,path,text,box,show,wire,token,move,set}=H,d=s.data;
 const title=txt('','teach-heading'),q=token('query','Q = ['+d.Q.join(', ')+']',144,'teach-token teach-color-4');
 const scoreExample=txt('','teach-small'),normalization=txt('权重 = exp(分数) / 允许位置的 exp 总和','teach-small');
 const legend=txt('灰色行：未来位置，不参与；同色段：同一位置的贡献','teach-small');
 const heads=['位置 / K','Q·K / √2','权重','V'].map(x=>txt(x,'teach-small'));
 const rows=d.keys.map((k,i)=>({bg:rect(i>d.query?'teach-mask':'teach-row teach-color-'+i),pos:txt('位置 '+(i+1),'teach-small'),key:token('key-'+i,'['+k.join(', ')+']',66,'teach-token teach-color-'+i),score:txt('','teach-number'),track:rect('teach-track'),bar:rect('teach-value teach-color-'+i),weight:txt('','teach-number'),v:token('value-'+i,'['+d.values[i].join(', ')+']',66,'teach-token teach-color-'+i),link:path()}));
 const resultTitle=txt('结果的两个维度，分别相加','teach-heading'),result=[txt('','teach-number'),txt('','teach-number')];
 const contributions=d.weights.map((w,i)=>[0,1].map(j=>({bar:rect('teach-contribution teach-color-'+i),number:txt('','teach-small teach-contribution-number')})));
 const sum=txt('','teach-heading'),equation=txt('','teach-small');
 return(p,m)=>{
  text(title,18,28,`观察位置 ${d.query+1}：只读取当前位置及之前`);move(q,m?18:32,48,phase(p,0));text(legend,18,m?126:116);if(m)legend.setAttribute('font-size','11');
  text(scoreExample,18,102,`位置 ${d.query+1}：(${d.Q[0]}×${d.keys[d.query][0]} + ${d.Q[1]}×${d.keys[d.query][1]}) / √2 = ${d.scores[d.query].toFixed(2)}`);show(scoreExample,phase(p,1));
  heads.forEach((e,i)=>{text(e,m?[24,124,222,290][i]:[28,216,394,626][i],m?150:150);show(e,phase(p,0));});
  const rowH=m?126:86;
  rows.forEach((r,i)=>{const y=164+i*rowH,allowed=i<=d.query;box(r.bg,16,y,m?328:688,rowH-8);text(r.pos,26,y+21);move(r.key,26,y+30,phase(p,0));
   text(r.score,m?124:218,y+52,allowed?(phase(p,1)===1?d.scores[i].toFixed(2):'计算中'):'遮住');show(r.score,phase(p,1));
   const bx=m?108:358,by=m?y+83:y+35,bw=m?206:180;box(r.track,bx,by,bw,14);box(r.bar,bx,by,bw*d.weights[i]*phase(p,2),14);text(r.weight,m?222:550,m?y+52:y+49,phase(p,2)===1?(d.weights[i]*100).toFixed(1)+'%':'归一化中');[r.track,r.bar,r.weight].forEach(e=>show(e,phase(p,2)));
   move(r.v,m?274:626,y+30,phase(p,0));wire(r.link,`M${m?307:659} ${y+64} v10`,phase(p,3)*(allowed?1:0));
  });
  const outY=m?704:540;text(resultTitle,18,outY);text(equation,18,outY+28,'每个 V × 自己的权重，再按维度相加');show(resultTitle,phase(p,3));show(equation,phase(p,3));
  for(let j=0;j<2;j++){
   let start=90;const unit=m?72:150,y=outY+48+j*56;text(result[j],18,y+19,'维度 '+(j+1));show(result[j],phase(p,3));
   contributions.forEach((parts,i)=>{const part=parts[j],value=d.weights[i]*d.values[i][j],width=value*unit*phase(p,3);box(part.bar,start,y,width,25);text(part.number,start+3,y+18,width>30&&phase(p,3)===1?value.toFixed(2):'');show(part.number,phase(p,3));start+=width;});
  }
  text(normalization,18,outY+154);show(normalization,phase(p,2));
  text(sum,18,outY+184,phase(p,3)===1?'汇合得到 ['+d.mixed.map(x=>x.toFixed(2)).join(', ')+']':'正在按维度汇合 · 条形为绘制进度');show(sum,phase(p,3));
 };
});
visuals['f-cache']=(svg,viewport,s)=>board(svg,viewport,'cache',m=>m?1080:660,H=>{
 const {txt,rect,text,box,show,token,move,set}=H;
 const panels=[rect(),rect()],titles=[txt('不跨轮保留 K/V','teach-heading'),txt('保存历史 K/V','teach-heading')];
 const rows=[0,1].map(side=>[0,1,2].map(r=>({heading:txt('第 '+(r+1)+' 次调用','teach-small'),cells:Array.from({length:r+4},(_,i)=>({a:token(`work-${side}-${r}-${i}`,i<4?'p'+(i+1):'y'+(i-3),42,'teach-token'),fill:rect('teach-compute'),note:txt('','teach-small')}))})));
 const totals=[txt('','teach-heading'),txt('','teach-heading')],poolTitle=txt('右侧保留的一份历史 · 没有每轮复制一套缓存','teach-small');
 const pool=Array.from({length:6},(_,i)=>token('stored-'+i,i<4?'p'+(i+1):'y'+(i-3),42,'teach-token teach-color-0'));
 const note=txt('橙：本轮新算 K/V   绿：读取已经保存的 K/V','teach-small');
 return(p,m)=>{
  const ys=m?[24,494]:[24,24],xs=m?[16,16]:[16,370],pw=m?328:334;
  [0,1].forEach(side=>{box(panels[side],xs[side],ys[side],pw,438);text(titles[side],xs[side]+14,ys[side]+30);
   let computed=0;rows[side].forEach((r,j)=>{const y=ys[side]+74+j*104,t=phase(p,j),active=p>=j&&p<j+1;text(r.heading,xs[side]+14,y-12);
    r.cells.forEach(({a,fill,note},i)=>{const x=xs[side]+14+i*49,isReuse=side===1&&j>0&&i<j+3,count=side===0||j===0?j+4:1;
     move(a,x,y,1);a.g.setAttribute('class','teach-token '+(isReuse?'teach-color-0':'teach-color-2'));set(a.r,{opacity:p<=j?.18:1});box(fill,x,y+38,42*t,5);fill.setAttribute('class',isReuse?'teach-read':'teach-compute');text(note,x,y+65,p<=j?'待算':isReuse?'复用':'新算');show(note,p>j?1:.4);a.g.dataset.progress=t.toFixed(3);
     if(i===0)computed+=t===1?count:0;
    });
   });text(totals[side],xs[side]+14,ys[side]+412,`已算完 ${computed} 个位置${p%1>0&&p<3?' · 本轮计算中':''}`);
  });
  const py=m?982:534;text(poolTitle,18,py-15);pool.forEach((a,i)=>move(a,24+i*49,py,phase(p,i<4?0:i-3)));text(note,18,m?1060:630);
 };
});
visuals['f-deployment']=(svg,viewport,s)=>board(svg,viewport,'deployment',m=>m?1010:660,H=>{
 const {txt,rect,path,text,box,show,wire,token,move,set}=H;
 const P=rect(),D=rect(),pt=txt('Prefill · 先处理输入','teach-heading'),dt=txt('Decode · 接着生成','teach-heading'),client=txt('客户端输出','teach-small');
 const src=[0,1].map(i=>token('source-kv-'+i,'历史 '+(i+1),94,'teach-token teach-color-'+i)),dest=[0,1].map(i=>token('target-kv-'+i,'接收位置 '+(i+1),108,'teach-token teach-empty'));
 const copies=[0,1].map(i=>token('transfer-kv-'+i,'历史 '+(i+1),94,'teach-token teach-color-'+i));
 const meta=token('metadata','元数据通知',108,'teach-token teach-color-3'),first=token('first-token','y1',48,'teach-token teach-color-2'),firstInput=token('decode-input','输入 y1',76,'teach-token teach-color-2'),second=token('second-token','y2',48,'teach-token teach-color-4');
 const added=token('decode-kv-y1','y1 的历史',108,'teach-token teach-color-2');
 const gates=['历史载荷到齐','元数据已到','本例执行机会'].map(x=>({box:rect('teach-gate'),t:txt(x,'teach-small'),check:txt('','teach-number')}));
 const sourceNote=txt('','teach-small'),targetNote=txt('','teach-small'),transfer=path(),inputLink=path();
 return(p,m)=>{
  const px=16,py=70,dx=m?16:386,dy=m?486:70,pw=m?328:318;
  box(P,px,py,pw,300);box(D,dx,dy,pw,300);text(pt,px+14,py+30);text(dt,dx+14,dy+30);text(client,18,28);
  src.forEach((a,i)=>move(a,px+28,py+92+i*70,phase(p,0)*(1-phase(p,5))));
  dest.forEach((a,i)=>{move(a,dx+28,dy+92+i*70);show(a.t,1-phase(p,3+i));});
  copies.forEach((a,i)=>{const t=phase(p,3+i);move(a,lerp(px+28,dx+34,t),lerp(py+92+i*70,dy+92+i*70,t),t>0?1:0);a.g.dataset.received=t.toFixed(3);});
  const mt=phase(p,2);move(meta,lerp(px+160,dx+168,mt),lerp(py+52,dy+52,mt),mt);
  move(first,145,8,phase(p,1));move(second,205,8,phase(p,7));
  move(firstInput,lerp(145,dx+28,phase(p,6)),lerp(8,dy+250,phase(p,6)),phase(p,6));move(added,dx+164,dy+250,phase(p,7));
  text(sourceNote,px+16,py+276,p<1?'等待输入计算':p<6?'发送和本地使用结束前保留源数据':'本例源端已安全收尾');
  text(targetNote,dx+16,dy+218,p<5?'仍在等待完整历史':p<7?'历史到齐，等待执行条件':'可以使用历史继续');
  const gy=m?828:436;gates.forEach((g,i)=>{const x=m?18:18+i*235,y=m?gy+i*52:gy,threshold=[4,2,6][i],t=phase(p,threshold);box(g.box,x,y,m?324:218,40);text(g.t,x+12,y+25);text(g.check,x+(m?274:168),y+25,t===1?'满足':'等待');g.box.classList.toggle('is-ready',t===1);});
  wire(transfer,`M${px+122} ${py+109} ${m?'H350 V'+(dy+109)+' H'+(dx+128):'H'+(dx+34)}`,phase(p,3));
  wire(inputLink,`M169 42 ${m?'H350 V'+(dy+267)+' H'+(dx+104):'H'+(dx+300)+' V'+(dy+267)+' H'+(dx+104)}`,phase(p,6));
 };
});
})();
