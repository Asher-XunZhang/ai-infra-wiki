/* Plain-language figures; the accounting model is shared with the model tests. */
(() => {
  'use strict';
  for(const root of document.querySelectorAll('[data-diagram]')) {
    const a=window.LearningDiagram.create(root,{j:0,phase:'prefill',query:3,reuse:3,decode:1,chunk:4,round:0,q:20,d:20});
    const {s,$,esc,clamp,fmt,chip,node,path,button,select,slider,stepControls,stepBind,status,evidence,bind,controls,change,animationControls,animationBind,svgStart,tx,line}=a;
function renderJourney(){
 const j=s.j,p=['p1','p2','p3','p4'],labels=['接入与分词','Prefill · 层 1','Prefill · 层 2','Prefill · 层 3','采样首字 y1','Decode · 只看一轮'];
 controls(animationControls('j',6,'j'));animationBind('j',6,'j');
 let html=`<h3>${labels[j]}</h3><div class="wr-row">${(j===5?['y1']:p).map(x=>chip(x,j===0?'token ID':j===5?'上一轮输出':'同一轮处理','new')).join('')}</div><div class="wr-down">↓ ${j===0?'分词后排队，获准才执行':j===1?'Embedding：token ID → 输入向量':j<4?'接续上一层的隐藏状态':j===4?'取最后一个 prompt 位置的最终隐藏状态':'复用历史 KV，仍遍历完整模型'}</div>`;
 html+=path([1,2,3].map(l=>node('层 '+l,j>=l?(j===5?'新增 y1 的 KV':'已写 p1…p4 的 KV'):'尚未计算',j===l||j===5?'active':'')));
 if(j>0&&j<4)html+=`<div class="wr-label">当前层 ${j} · 因果 Attention → MLP</div><div class="wr-node"><div>p1 读 p1；p2 读 p1…p2；p3 读 p1…p3；p4 读全部 4 个位置。</div><div class="wr-row" style="margin-top:10px">${p.map(x=>chip(x,'K'+j+' / V'+j,'new')).join('')}</div><small>本层保存自己的 K/V；隐藏状态送往下一层。这里按层分解一次 Prefill，并非 3 轮请求调度。</small></div>`;
 if(j===0)html+=node('待调度','此时还没有任何层的 KV，也没有输出 token。','wait');
 if(j===4)html+=path([node('输出头','最后位置 → 词表 logits','active'),node('采样','得到首 token y1','active'),node('返回首字','y1 此时尚无 KV')]);
 if(j===5)html+=`<div class="wr-label">每层：读历史 4 个位置，写入 y1 的 K/V</div><div class="wr-row">${p.map(x=>chip(x,'复用','read')).join('')}${chip('y1','本轮新写','new')}${chip('y2','新采样 · 无 KV','future')}</div><div class="wr-down">↶ 后续重复：上一输出 → 完整模型 → 下一输出；直到 EOS 或长度上限</div>`;
 $('wr-scene').innerHTML=html;status(j===0?'一条文本请求先变成 token IDs，再等待执行资源。':j<4?`Prefill 同时处理 4 个输入位置；当前完成第 ${j} 层，尚未采样首字。`:j===4?'Prefill 最终得到 y1；缓存中只有已处理的 p1…p4，尚无 y1。':'输入 y1 → 各层新增它的 KV → 采样 y2；相同机制不再逐轮重复展示。');
 evidence('3 层、4 个输入位置的普通因果 Transformer 教学示意；不表示真实模型只有 3 层。一次 Prefill 的层内多位置计算，与逐层依赖同时成立。','pages/sglang/inference-overview/model.js');
}

function renderTransformer(){
 const p=s.phase==='prefill';s.query=p?Math.min(3,s.query):4;const keys=p?['p1','p2','p3','p4']:['p1','p2','p3','p4','y1'];
 controls(button('t-p','Prefill · 4 个位置',p)+button('t-d','Decode · 1 个位置',!p));bind('t-p','click',()=>change(()=>{s.phase='prefill';s.query=3;}));bind('t-d','click',()=>change(()=>{s.phase='decode';s.query=4;}));
 $('wr-scene').innerHTML=`<div class="wr-label">本轮输入 · 点选观察位置</div><div class="wr-row">${(p?keys:['y1']).map((x,i)=>button('tok-'+i,x,s.query===(p?i:4))).join('')}</div><div class="wr-down">↓ 每层都执行 Attention 和 MLP</div><div class="wr-node"><div class="wr-row"><span class="wr-token new">Q(${keys[s.query]})</span><span class="wr-small">${p?'4 个位置一起前向；当前只聚焦一个':'只新增 y1 的 Q/K/V'}</span></div><div class="wr-label">↓ 读取同层 K/V</div><div class="wr-row">${keys.map((x,i)=>chip(x,(i<=s.query?'可读':'× 遮罩')+' · '+(p||i===4?'新写':'复用'),(i<=s.query?'read':'mask')+(p||i===4?' new':''))).join('')}</div><div class="wr-down">↓</div>${path([node('Attention','Q 读取 '+(s.query+1)+' 个位置','active'),node('MLP','仍执行本层计算')])}</div><div class="wr-down">↓ 后续层 → 输出头 → 采样</div><div class="wr-row">${chip(p?'y1':'y2','本轮输出','new')}${chip(p?'y1':'y2','尚无 KV','future')}</div>`;
 (p?keys:['y1']).forEach((_,i)=>bind('tok-'+i,'click',()=>change(()=>s.query=p?i:4)));
 status(p?`${keys[s.query]} 只能读取当前位置及之前的 K/V；选择焦点不改变批量前向。`:'历史 K/V 被复用，但 Decode 仍运行完整模型。');
}

function renderCache(){
 controls(slider('c-reuse','复用前缀',s.reuse,0,7)+slider('c-decode','Decode 轮数',s.decode,0,4));
 for(const [id,key] of [['c-reuse','reuse'],['c-decode','decode']])bind(id,'input',e=>change(()=>s[key]=+e.target.value));
 const accounting=window.InferenceOverview.cache(8,s.reuse,s.decode);
 const cells=Array.from({length:8},(_,i)=>chip('p'+(i+1),i<s.reuse?'前缀复用':'本次 Prefill 写',i<s.reuse?'read':'new read')).concat(Array.from({length:s.decode},(_,i)=>chip('y'+(i+1),i===s.decode-1?'本轮新写':'已写入',i===s.decode-1?'new read':'read')));
 $('wr-scene').innerHTML=`<div class="wr-label">各层覆盖相同的 token 位置</div><div class="wr-row">${cells.join('')}${chip('y'+(s.decode+1),'尚未写入','future')}</div><div class="wr-label">同一个 p8，在 3 层有 3 份独立的 K/V</div><div class="wr-three">${[1,2,3].map(l=>`<div class="wr-node"><h3>层 ${l}</h3><div class="wr-readonly">K[${l}, p8]<br>V[${l}, p8]</div><small>本层 Q 只读本层 KV</small></div>`).join('')}</div><div class="wr-note">K / V 是向量标识。位置数量相同，不表示各层向量数值相同或共享缓存。</div><div class="wr-output"><span>本次 Prefill 新算 <strong>${accounting.computed}</strong> 个位置</span><span>每层覆盖 <strong>${accounting.kv}</strong> 个位置</span></div>`;
 status(`复用 ${s.reuse} 个前缀位置；不同层的 K/V 独立，不能互换。输出 y${s.decode+1} 尚无 KV。`);
}

function renderDeployment(){
 controls('');
 $('wr-scene').innerHTML=`<div class="wr-grid"><div class="wr-node"><h3>合并部署 · 一个实例</h3>${path([node('Prefill','完整模型前向 → y1'),node('Decode','输入 y1 → y2 → …')])}<div class="wr-down">↓ 写 KV　　↑ 读 KV</div>${node('本地 KV','留在同一实例内接续','active')}</div><div class="wr-node"><h3>PD 分离 · 两个角色</h3>${node('实例 P','完整模型前向 → y1','active')}<div class="wr-down">↓ KV + 首 token / 配对信息</div>${node('实例 D','完整模型前向 → y2 → …','active')}<div class="wr-note">P 与 D 分工于阶段；各角色覆盖完整模型计算。</div></div></div><div class="wr-label">D 执行前，三个条件要同时满足</div><div class="wr-three">${node('KV 可用','数据已经可读')}${node('配对正确','请求与 metadata 对得上')}${node('可执行','准入与执行资源允许')}</div>`;
 status('KV 到达只是一个条件；角色分离会增加交接步骤，也让 P、D 能分别配置资源。');
 evidence('架构与准入条件概览；省略每个角色内部的 TP / PP 切分与具体传输协议。','pages/sglang/inference-overview/deployment-scene.js');
}

function renderScheduling(){
 const n=window.InferenceOverview.schedule(s.chunk,0).rounds.length;s.round=clamp(s.round,0,n-1);
 controls(select('s-chunk','B 每轮输入',[[1,'1 token'],[2,'2 tokens'],[4,'4 tokens'],[8,'8 tokens']],s.chunk)+stepControls(s.round,n,'s'));bind('s-chunk','change',e=>change(()=>{s.chunk=+e.target.value;s.round=0;}));stepBind('s','round',n);
 const narrow=$('wr-scene').clientWidth<480,visible=narrow?[s.round]:Array.from({length:n},(_,i)=>i);
 let html=`<div class="wr-round-grid" style="grid-template-columns:60px repeat(${visible.length},minmax(0,1fr))"><span></span>${visible.map(i=>`<span class="wr-small" style="text-align:center">轮 ${i+1}</span>`).join('')}<span>A 续写</span>${visible.map(i=>`<div class="wr-round-cell ${i===s.round?'now':''}">y${i+1}</div>`).join('')}<span>B 输入</span>${visible.map(i=>`<div class="wr-round-cell ${i===s.round?'now':''}">p${i*s.chunk+1}…p${Math.min(8,(i+1)*s.chunk)}</div>`).join('')}</div>`;
 html+=`<div class="wr-label">当前 batch · 轮 ${s.round+1}</div><div class="wr-node"><div class="wr-row">${chip('A','1 个 Decode 输入','read')}${Array.from({length:Math.min(s.chunk,8-s.round*s.chunk)},(_,i)=>chip('B:p'+(s.round*s.chunk+i+1),'Prefill','new')).join('')}</div></div><div class="wr-label">B 已处理 ${Math.min(8,(s.round+1)*s.chunk)} / 8 个位置</div><div class="wr-meter"><span style="width:${Math.min(8,(s.round+1)*s.chunk)/8*100}%"></span></div>`;
 $('wr-scene').innerHTML=html;status(`本轮包含 A 的一次续写和 B 的一块输入；${s.round===n-1?'B 的 Prefill 到此完成。':'B 的中间块不等于一个输出 token。'}`);
}

function renderLatency(){
 controls(slider('l-q','排队 ms',s.q,0,100,10)+slider('l-d','Decode 每步 ms',s.d,10,60,10));for(const [id,key] of [['l-q','q'],['l-d','d']])bind(id,'input',e=>change(()=>s[key]=+e.target.value));
 const w=$('wr-scene').clientWidth,left=48,right=w-14,x=t=>left+t/360*(right-left);let svg=svgStart(w,204,'基线与调整后共用 0 到 360 毫秒时间轴');
 [0,120,240,360].forEach(t=>{svg+=line(x(t),28,x(t),168)+tx(x(t),19,t,t===360?'text-anchor="end"':'text-anchor="middle"');});svg+=tx(right,195,'时间 ms →','text-anchor="end"');
 [[20,20,'基线',62],[s.q,s.d,'当前',129]].forEach(([q,d,label,y])=>{svg+=tx(0,y+6,label);const parts=[[0,q,'var(--line)'],[q,q+60,'var(--green)']];parts.forEach(([a,b,c])=>{if(b>a)svg+=`<rect x="${x(a)}" y="${y-12}" width="${x(b)-x(a)}" height="22" fill="${c}" opacity=".45"/>`;});const tt=q+60;svg+=line(x(tt),y,x(tt+3*d),y,'var(--blue)','stroke-width="3"');for(let i=0;i<4;i++)svg+=`<circle cx="${x(tt+i*d)}" cy="${y}" r="5" fill="${i===0?'var(--amber)':'var(--blue)'}"/>`;svg+=tx(x(tt),y+29,`首字 ${tt} ms`);});
 $('wr-scene').innerHTML=svg+'</svg>'+`<div class="wr-chips"><span class="wr-chip">灰：排队</span><span class="wr-chip">绿：Prefill 60 ms</span><span class="wr-chip">● 首字 / 后续输出</span></div><div class="wr-output"><span>首字等待 ${s.q+60} ms</span><span>后续间隔 ${s.d} ms</span><span>完成 ${s.q+60+3*s.d} ms</span></div>`;
 status(`排队改变首字出现位置；Decode 单步时间改变后续三个输出之间的距离。`);
}

    const renders={journey:renderJourney,transformer:renderTransformer,cache:renderCache,deployment:renderDeployment,scheduling:renderScheduling,latency:renderLatency};
    a.start(()=>{evidence(root.dataset.diagram==='scheduling'?'固定教学策略：A 每轮续写一次，B 共 8 个输入位置；真实顺序还取决于预算、准入与调度策略。':root.dataset.diagram==='latency'?'固定 Prefill=60 ms，4 个输出 token；忽略额外等待与重叠，不能用作性能预测。':'普通因果 Transformer 示意；省略多头、归一化、残差等细节。');renders[root.dataset.diagram]();});
  }
})();
