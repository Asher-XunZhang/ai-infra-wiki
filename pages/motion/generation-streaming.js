/* Chapter 1, generation lesson. One independent concept after sampling:
 * the selected token ID is one object with two paths.
 *
 *   ID -> Detokenizer -> text fragment -> transport -> client
 *   ID -> next model input -> next candidates        (does not wait)
 *
 * The toy vocabulary is deliberately small. A real detokenizer can buffer
 * several tokens, and one token is not guaranteed to decode to a complete
 * Unicode character by itself; this board says that in its boundary text
 * instead of turning the teaching simplification into a general rule. */
(function(root){
'use strict';
const TABLE={101:'学习',102:'休息',103:'一下',104:'句号'};
const BRANCHES={
 learning:{key:'learning',id:101,word:'学习',tone:2,next:[{word:'句号',p:.8},{word:'其他',p:.2}]},
 rest:{key:'rest',id:102,word:'休息',tone:3,next:[{word:'一下',p:.8},{word:'其他',p:.2}]}
};
const clamp01=n=>Math.max(0,Math.min(1,n));
function state(key,p){
 const b=BRANCHES[key]||BRANCHES.learning;
 const selected=p>=1,detokenized=p>=2,received=p>=4;
 return {
  branch:b.key,id:b.id,word:b.word,tone:b.tone,
  stage:p<1?'ready':p<2?'selected':p<3?'detokenized':p<4?'in-flight':'received',
  selected,detokenized,
  inFlight:p>=2.55&&p<4,
  received,
  packetProgress:clamp01((p-2.55)/1.3),
  clientText:received?b.word:'',
  nextInputId:selected?b.id:null,
  candidates:p>=3?b.next:null,
  // One semantic gate for both the candidate panel and the loop caption.
  // The next candidates are ready exactly when they become non-null at p=3;
  // they are being computed only while the packet is in flight before that.
  candidatesState:p>=3?'ready':p>=2.55?'calculating':'waiting',
  historyKV:['我','想']
 };
}
const api={table:TABLE,branches:BRANCHES,state};
root.GenerationStreamingModel=api;
if(typeof module==='object'&&module.exports)module.exports=api;

const V=root.MechanismVisuals,D=root.TeachingDrawing;
if(!V||!D)return;
if(root.__generationStreaming)return;
root.__generationStreaming=true;
const {board,phase,lerp}=D;

function makeButton(H,label,aria,key){
 const g=H.node('g',{class:'gs-btn',role:'button',tabindex:'0','aria-pressed':'false','aria-label':aria,opacity:'0'});
 const r=H.node('rect',{rx:7,width:10,height:10});
 const t=H.node('text',{x:0,y:0,'text-anchor':'middle'},label);
 g.append(r,t);
 g.addEventListener('click',e=>{e.stopPropagation();g.streamPick?.(key);});
 g.addEventListener('keydown',e=>{
  if(e.key==='Enter'||e.key===' '||e.key==='Spacebar'){e.preventDefault();e.stopPropagation();g.streamPick?.(key);}
 });
 return {g,r,t};
}
function placeButton(H,btn,x,y,w,h,active){
 H.set(btn.g,{transform:`translate(${x} ${y})`});
 H.set(btn.r,{width:w,height:h});
 H.text(btn.t,w/2,h/2+4);
 H.show(btn.g,1);
 btn.g.setAttribute('aria-pressed',active?'true':'false');
 btn.g.classList.toggle('is-on',active);
}

function streamingBoard(svg,viewport,s){
 let branchKey='learning',lastP=0,lastM=false,repaint=()=>{};
 function pauseFilm(){
  const film=svg.closest('.mechanism-film');
  const b=film?.querySelector('[data-motion-action=play]');
  if(b&&b.getAttribute('aria-pressed')==='true')b.click();
 }
 function change(key){
  pauseFilm();
  branchKey=BRANCHES[key]?key:'learning';
  repaint();
 }

 return board(svg,viewport,'generation-streaming',m=>m?656:570,H=>{
  const {txt,rect,path,text,box,show,wire,token,move,set}=H;
  const panelModel=rect('teach-panel'),panelDetok=rect('teach-panel'),panelClient=rect('teach-panel'),panelLoop=rect('teach-panel');
  const lane=rect('stream-lane'),bubble=rect('stream-client-bubble');
  const ctxA=token('stream-context-a','我',42,'teach-token teach-color-0');
  const ctxB=token('stream-context-b','想',42,'teach-token teach-color-0');
  const selectedChip=token('stream-selected','等待选中',100,'teach-token teach-empty');
  const inChip=token('stream-detok-in','ID —',62,'teach-token teach-empty');
  const outChip=token('stream-fragment','片段',62,'teach-token teach-empty');
  const packet=token('stream-packet','',76,'teach-token teach-empty');
  const selectedOut=token('stream-next-source','ID —',82,'teach-token teach-empty');
  const nextChip=token('stream-next-input','ID —',82,'teach-token teach-empty');
  const kvA=token('stream-kv-a','我',42,'teach-token teach-color-0');
  const kvB=token('stream-kv-b','想',42,'teach-token teach-color-0');

  const title=txt('','teach-heading'),sub=txt('','teach-small');
  const modelHead=txt('模型侧（服务端）','teach-small'),contextLabel=txt('输入上下文','teach-small');
  const selectedLabel=txt('本轮选中','teach-small'),selectedNote=txt('','teach-small');
  const detokHead=txt('Detokenizer','teach-heading'),detokSub=txt('ID → 文字片段','teach-small');
  const detokTable=txt('教学词表（玩具 ID）','teach-small'),table1=txt('101 学习 · 102 休息','teach-small'),table2=txt('103 一下 · 104 句号','teach-small'),tableNote=txt('本例：一词一片段','teach-small');
  const transportLabel=txt('响应传输','teach-small'),transportSub=txt('步骤不是耗时','teach-small'),packetStatus=txt('尚未发出','teach-small');
  const clientHead=txt('客户端','teach-heading'),clientLabel=txt('可见文本','teach-small'),clientText=txt('','stream-client-empty'),clientStatus=txt('已收到 0 个片段','teach-small'),clientNote=txt('模型不等待这段文字','teach-small');
  const loopHead=txt('模型回送：不等待网络','teach-heading'),sourceLabel=txt('已选 ID','teach-small'),nextLabel=txt('下一轮输入','teach-small'),loopNote=txt('同一 ID 进入下一轮；不经过文字回读','teach-small');
  const kvLabel=txt('历史 KV','teach-small'),kvNote=txt('保持','teach-small'),loopStatus=txt('','teach-small');
  const candHead=txt('下一轮候选（教学）','teach-small'),candPending=txt('等待模型产出','teach-small'),candNote=txt('候选在片段到达前产生；分支会改变下一词。','teach-small');
  const candRows=[0,1].map(i=>({
   word:txt('','teach-number'),pct:txt('','teach-small'),track:rect('teach-track'),bar:rect('teach-value teach-color-'+(i?4:3))
  }));
  const narrowLoop=txt('','teach-small'),narrowKv=txt('','teach-small'),narrowCand=txt('','teach-small');
  const footer=txt('玩具 ID 与一词一片段；步骤只表示依赖，不代表真实网络时序或耗时。','teach-small');
  const idWire=path('teach-link'),detokWire=path('teach-link'),rail=path('teach-link'),loopWire=path('teach-link');
  const branchLearning=makeButton(H,'学习 · ID 101','选择教学分支：学习，ID 101','learning');
  const branchRest=makeButton(H,'休息 · ID 102','选择教学分支：休息，ID 102','rest');
  branchLearning.g.streamPick=change;branchRest.g.streamPick=change;

  function setSelected(chip,value,filled,tone,on){
   set(chip.g,{class:'teach-token '+(filled?'teach-color-'+tone:'teach-empty')});
   H.text(chip.t,chip.width/2,22,value);
   show(chip.g,on);
  }

  function draw(p,m){
   lastP=p;lastM=m;
   const st=state(branchKey,p),b=BRANCHES[st.branch],tone=b.tone;
   const color='teach-color-'+tone;

   if(m){
    text(title,14,24,'ID → 文字 → 网络 → 客户端');
    text(sub,14,43,'模型回送不等网络；步骤不是耗时');
    box(panelModel,12,88,336,198);box(panelDetok,12,296,336,144);box(panelClient,12,486,336,110);
    show(panelLoop,0);
    text(modelHead,24,108);text(contextLabel,24,130);
    move(ctxA,24,138,1);move(ctxB,72,138,1);
    text(selectedLabel,24,188);text(selectedNote,150,210,p<1?'尚未选出':p<2?'ID 已选，尚无文字':p<3?'片段已生成，尚未发送':p<4?'片段在途，客户端未收到':'客户端已收到');
    move(selectedChip,24,196,st.selected?phase(p,.9,1.2):1);
    setSelected(selectedChip,st.selected?'ID '+b.id:'等待选中',st.selected,tone,1);
    text(detokHead,24,316);text(detokSub,24,332);
    move(inChip,24,342,st.selected?phase(p,1.1,1.5):0);
    setSelected(inChip,st.selected?'ID '+b.id:'ID —',st.selected,tone,1);
    wire(detokWire,'M88 359 H108',phase(p,1.85,2.2));
    move(outChip,112,342,st.detokenized?phase(p,1.9,2.25):0);
    setSelected(outChip,st.detokenized?b.word:'片段',st.detokenized,tone,1);
    text(detokTable,24,394);text(table1,24,410);text(table2,24,426);show(tableNote,0);
    text(transportLabel,24,454);text(transportSub,0,0,'');show(transportSub,0);
    wire(rail,'M180 444 V482',st.inFlight?1:0);
    move(packet,144,448,st.inFlight?1:0);
    setSelected(packet,st.inFlight?b.word:'',st.inFlight,tone,st.inFlight?1:0);
    text(packetStatus,24,476,st.received?'已到达客户端':st.inFlight?'在途':'尚未发出');
    show(packetStatus,p>=1.5?1:0);
    box(panelClient,12,486,336,110);box(bubble,24,496,312,44);
    text(clientHead,24,510);text(clientLabel,24,556);
    text(clientText,180,522,st.received?b.word:'尚未收到');
    set(clientText,{'text-anchor':'middle',class:st.received?'stream-client-word':'stream-client-empty'});
    text(clientStatus,24,580,st.received?'已收到 1 个片段':'已收到 0 个片段');
    show(clientNote,0);show(narrowKv,1);
    text(narrowLoop,24,250,st.selected?'模型已回送：下一轮输入 ID '+b.id:'尚未选出 ID');
    text(narrowKv,24,270,'原有历史 KV：我、想（保留；本图省略新增写入）');
    text(narrowCand,24,618,st.candidatesState==='ready'?'下一轮候选：'+b.next.map(x=>x.word+' '+Math.round(x.p*100)+'%').join(' / '):st.candidatesState==='calculating'?'下一轮候选：模型计算中…':'下一轮候选：等待模型产出');
    show(narrowCand,p>=1?1:0);
    text(footer,14,642,'玩具 ID 与一词一片段；步骤只表示依赖，不代表网络时序。');
    // The narrow board has no separate bottom panel; the model loop and the
    // branch-dependent next candidates stay beside the two main paths.
    show(loopHead,0);show(sourceLabel,0);show(nextLabel,0);show(loopNote,0);show(kvLabel,0);show(kvNote,0);show(loopStatus,0);
    show(candHead,0);show(candPending,0);show(candNote,0);candRows.forEach(r=>{show(r.word,0);show(r.pct,0);show(r.track,0);show(r.bar,0);});
    show(selectedOut.g,0);show(nextChip.g,0);show(kvA.g,0);show(kvB.g,0);
    show(idWire,0);show(loopWire,0);show(lane,0);
    placeButton(H,branchLearning,14,54,160,28,branchKey==='learning');
    placeButton(H,branchRest,182,54,164,28,branchKey==='rest');
   }else{
    text(title,16,26,'ID → 文字 → 网络 → 客户端');
    text(sub,16,46,'同一个输出 ID 有两条去向；模型回送不等待网络。');
    box(panelModel,14,72,210,238);box(panelDetok,238,72,198,238);box(panelClient,548,72,158,238);box(panelLoop,14,322,692,210);
    text(modelHead,26,94);text(contextLabel,26,116);
    move(ctxA,26,124,1);move(ctxB,74,124,1);
    text(selectedLabel,26,180);
    move(selectedChip,26,188,st.selected?phase(p,.9,1.2):1);
    setSelected(selectedChip,st.selected?'ID '+b.id:'等待选中',st.selected,tone,1);
    wire(idWire,'M126 205 H246',st.selected?phase(p,1.05,1.45):0);
    text(selectedNote,26,250,'选中的是 ID；文字片段另走右侧路径。');
    text(detokHead,250,94);text(detokSub,250,116);
    move(inChip,250,188,st.selected?phase(p,1.1,1.5):0);
    setSelected(inChip,st.selected?'ID '+b.id:'ID —',st.selected,tone,1);
    wire(detokWire,'M316 205 H344',phase(p,1.85,2.2));
    move(outChip,348,188,st.detokenized?phase(p,1.9,2.25):0);
    setSelected(outChip,st.detokenized?b.word:'片段',st.detokenized,tone,1);
    text(detokTable,250,248);text(table1,250,266);text(table2,250,282);show(tableNote,1);text(tableNote,250,300,'本例：一词一片段（教学简化）');
    text(transportLabel,442,150);text(transportSub,442,164,'步骤不是耗时');show(transportSub,1);
    box(lane,440,182,102,46);show(lane,1);
    wire(rail,'M444 240 H544',st.inFlight?1:0);
    move(packet,lerp(412,466,st.packetProgress),188,st.inFlight?1:0);
    // The transport copy carries the real payload only while it is in flight.
    // Clear both its text and its opacity outside that window so an idle lane
    // cannot leak the selected word and the arrived copy does not linger.
    setSelected(packet,st.inFlight?b.word:'',st.inFlight,tone,st.inFlight?1:0);
    text(packetStatus,442,262,st.received?'已到达客户端':st.inFlight?'片段在途':'尚未发出');
    show(packetStatus,p>=1.5?1:0);
    box(panelClient,548,72,158,238);box(bubble,560,124,134,86);
    text(clientHead,560,94);text(clientLabel,560,228);
    text(clientText,627,174,st.received?b.word:'尚未收到');
    set(clientText,{'text-anchor':'middle',class:st.received?'stream-client-word':'stream-client-empty'});
    text(clientStatus,560,250,st.received?'已收到 1 个片段':'已收到 0 个片段');
    show(clientNote,1);text(clientNote,560,276,'模型不等待这段文字');

    text(loopHead,28,348);text(sourceLabel,28,368);text(nextLabel,152,368);
    move(selectedOut,28,376,st.selected?1:0);
    setSelected(selectedOut,st.selected?'ID '+b.id:'ID —',st.selected,tone,1);
    wire(loopWire,'M114 391 H142',st.selected?phase(p,1.05,1.45):0);
    move(nextChip,152,376,st.selected?phase(p,1.1,1.5):1);
    setSelected(nextChip,st.selected?'ID '+b.id:'ID —',st.selected,tone,1);
    text(loopNote,28,430);show(loopNote,p>=1?1:0);
    text(kvLabel,28,456,'原有历史 KV');move(kvA,102,446,1);move(kvB,150,446,1);
    text(kvNote,204,466,'保留；省略新输入 KV 写入');show(kvNote,1);
    // The loop caption follows the same selected / fragment / candidate state
    // as the objects above; it never announces a receipt that has not happened.
    const loopText=!st.selected?'尚未选择 ID；没有可回送的输入。':st.received?'客户端已收到；模型下一轮继续。':st.inFlight?(st.candidatesState==='ready'?'片段在途；模型已产出下一轮候选。':'片段在途；模型正在计算下一轮候选。'):st.detokenized?'片段尚未发送，模型已回送同一 ID。':'ID 已回送，模型先继续。';
    text(loopStatus,28,500,loopText);show(loopStatus,1);
    text(candHead,360,348);
    show(candPending,st.candidatesState==='ready'?0:1);
    text(candPending,360,386,st.candidatesState==='calculating'?'模型计算中…':'等待模型产出');
    // p=3 is the exact ready endpoint, so the rows and their explanation share
    // one gate. A fade that starts at 3 would leave the endpoint blank.
    const candOn=st.candidatesState==='ready'?1:0;
    candRows.forEach((row,i)=>{
     const c=b.next[i];
     text(row.word,368,386+i*32,c?c.word:'');
     text(row.pct,512,386+i*32,c?Math.round(c.p*100)+'%':'');
     set(row.pct,{'text-anchor':'end'});
     box(row.track,360,392+i*32,160,8);
     box(row.bar,360,392+i*32,160*(c?c.p:0),8);
     show(row.word,candOn);show(row.pct,candOn);show(row.track,candOn);
     show(row.bar,candOn);
    });
    show(candNote,candOn);text(candNote,360,466,'候选在片段到达前产生；分支会改变下一词。');
    show(narrowLoop,0);show(narrowKv,0);show(narrowCand,0);show(lane,0);
    text(footer,16,554,'玩具 ID 与一词一片段；步骤只表示依赖，不代表真实网络时序或耗时。');
    placeButton(H,branchLearning,446,14,126,28,branchKey==='learning');
    placeButton(H,branchRest,582,14,124,28,branchKey==='rest');
   }

   svg.dataset.streamingBranch=branchKey;
   svg.dataset.streamingStage=st.stage;
   svg.dataset.streamingSelectedId=st.selected?String(b.id):'';
   svg.dataset.streamingFragment=st.detokenized?b.word:'';
   svg.dataset.streamingClient=st.clientText;
   svg.dataset.streamingNextId=st.nextInputId?String(st.nextInputId):'';
   svg.dataset.streamingPacket=st.inFlight?'in-flight':st.received?'arrived':'waiting';
   svg.dataset.streamingKv='我,想';
   svg.dataset.streamingCandidates=st.candidates?b.next.map(x=>x.word+':'+x.p).join(','):'';
   svg.dataset.streamingCandidatesState=st.candidatesState;
  }
  repaint=()=>draw(lastP,lastM);
  return draw;
 });
}

const streamingScene={
 concept:'streaming',mv:3,
 name:'Detokenization 与流式可见',
 title:'Detokenization 与流式可见：ID 什么时候变成你看到的文字？',
 intro:'同一个输出 ID 有两条去向：回到模型作为下一轮输入，或经过反分词、传输成为客户端文字。改变“学习 / 休息”分支，观察 ID、片段、上下文和下一轮候选怎样一起变化。',
 boundary:'教学词表 101→学习、102→休息、103→一下、104→句号均为玩具 ID，不对应 Qwen 真实词表。本例假设一个 token 对应一个完整片段，便于核对先后；真实流式可能缓冲多个 token，单个 token 也未必能独立还原完整 Unicode 文字。步骤只表示依赖，不给出毫秒，也不声称真实网络时序。模型回送只依赖已选 ID；客户端是否显示还取决于具体响应与会话状态。本图只保留进入本轮前的原有历史 KV，省略本轮新输入 KV 写入；真实推理会为下一轮输入写入新 KV，这不影响本图的回送因果。',
 kind:'generation-streaming',
 frames:[
  {title:'先看模型侧的输入',caption:'上下文“我 想”已经过模型计算；此刻还没有选出新的 token ID，客户端也没有收到文字。'},
  {title:'选出 token ID：还不是文字',caption:'模型选出一个教学 ID（101 学习或 102 休息）。它已经是下一轮输入；反分词还没发生，客户端仍然没有文字。'},
  {title:'反分词：ID 变成文字片段',caption:'Detokenizer 按词表把 ID 映射成片段。片段现在只在服务端侧，尚未进入传输；模型侧已经用同一 ID 回送并继续。'},
  {title:'片段在传输，模型不等待',caption:'片段正在去客户端的路上，客户端还没有收到；模型侧不等待这段网络，已用同一个 ID 产出下一轮候选。'},
  {title:'客户端收到文字',caption:'客户端现在才显示“学习”或“休息”。同一 ID 仍在模型侧作为下一轮输入；两条路径各自完成，步骤不代表真实耗时。'}
 ],
 stepDurations:[1.0,1.5,1.5,2.2]
};

const current=root.TeachingLessons?.['f-generation']||[];
if(!current.some(x=>x.concept==='streaming')){
 const at=current.findIndex(x=>x.legacy);
 const next=current.slice();
 next.splice(at<0?next.length:at,0,streamingScene);
 root.TeachingLessons=root.TeachingLessons||{};
 root.TeachingLessons['f-generation']=next;
}

const previous=V['f-generation'];
V['f-generation']=(svg,viewport,s)=>{
 const film=svg.closest('.mechanism-film');
 if(film){
  film.classList.toggle('is-generation-streaming',s.concept==='streaming');
  film.classList.toggle('is-generation-sampling',s.concept==='sampling'||s.concept==='streaming');
 }
 if(s.concept==='streaming')return streamingBoard(svg,viewport,s);
 return previous?previous(svg,viewport,s):null;
};
})(typeof globalThis==='object'?globalThis:this);
