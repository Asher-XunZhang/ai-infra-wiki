/* Archify owns checked geometry. The lesson owns every state and pulse phase.
 * No Viewer animation, independent timer, iframe, or runtime layout solver. */
window.AtlasArchify = (() => {
  'use strict';
  const h = React.createElement;
  const choose = (zh, en, english) => english ? en : zh;
  const aliases = {input:'tokenizer',core:'engine',scheduler:'scheduler',executor:'engine',worker:'workers',runner:'workers',model:'workers',attention:'workers',kernel:'workers',return:'detok'};
  const text = {
    '整理输入与响应':'Prepare inputs and responses','选择本轮请求与 token':'Select requests and token work',
    '衔接本 rank 的执行':'Bridge rank-local execution','准备输入并组织前向':'Prepare inputs and run forward',
    '定义模型各层的计算':'Define model-layer computation','结合当前输入与历史 KV':'Use inputs and historical KV',
    '在设备上执行具体算子':'Execute device operators','将 token IDs 转为文本':'Convert token IDs to text',
    '调度与本 rank 执行对象':'Scheduling and rank-local objects','模型计算结构与算子实现':'Model structure and operators',
    '内部请求':'Internal request','批次':'Batch','调用':'Call','前向':'Forward','子层':'Sublayer','后端算子':'Backend operators','采样 IDs':'Sampled IDs','文本片段':'Text chunks',
    '整理请求与输出':'Prepare requests and output','组织调度、执行与更新':'Organize schedule, execution, update',
    '选择请求与 token 工作量':'Select requests and token work','派发调用并汇集结果':'Dispatch calls and collect results',
    '管理设备与执行环境':'Manage device and environment','准备输入并组织计算':'Prepare inputs and computation',
    '定义各层前向结构':'Define forward structure','读取当前输入与历史 KV':'Use inputs and historical KV',
    '执行设备上的具体算子':'Execute device operators','EngineCore 内的组织与决策对象':'Objects inside EngineCore',
    '每个 rank 的执行对象':'Execution objects per rank','调用调度':'Call schedule','本轮计划':'Step plan','执行调用':'Execution call','结果':'Result',
    '输入与响应':'Inputs and responses','本轮批次':'Step batch','同进程调用':'In-process call','组织本轮循环':'Organize the step loop',
    '决定任务与 KV 分配':'Choose work and allocate KV','uni 本地调用':'uni in-process calls','mp 派发到 Worker':'mp dispatch to workers',
    '本 rank 执行环境':'Rank-local environment','准备输入与前向':'Inputs and forward','本 rank 的设备驻留数据':'Rank-local device data',
    '请求消息':'Request message','输出消息':'Output message','调度':'Schedule','请求 · ZMQ':'Request · ZMQ','同步输入':'Synchronize inputs',
    '本地调用':'Local call','执行计划 · IPC':'Execution plan · IPC','前向写入 K/V':'Forward writes K/V','API / 主进程':'API / main process',
    'Detokenizer 进程':'Detokenizer process','EngineCore 进程':'EngineCore process',
    '入口与输出':'Input and output','rank 0 接收请求':'rank 0 receives requests','增量恢复文本':'Incremental text decoding',
    '同进程执行':'In-process execution','独立管理 socket':'Separate management socket','请求 IDs / 参数':'Request IDs / parameters',
    '采样 token IDs':'Sampled token IDs','进程内调用':'In-process call','输入 DEALER / 输出 PUSH':'Input DEALER / output PUSH',
    '本轮执行计划':'Step execution plan','本地 mp 执行':'Local mp execution','仅 DP > 1 的扩展':'Optional DP > 1 extension',
    '请求帧':'Request frames','执行调用 / MQ':'Calls / MQ','CPU 可读结果':'CPU-readable results','负载 / wave':'Load / wave',
    '输入与流式响应':'Inputs and streaming','rank 0 入口':'rank 0 input','文本回程':'Text return',
    'token IDs / 参数':'token IDs / parameters','Python 前端':'Python frontend','调度与更新':'Schedule and update',
    '请求帧 · ROUTER/DEALER':'Request · ROUTER/DEALER','执行计划 · Executor MQ':'Plan · Executor MQ',
    'CPU 可读结果 · response MQ':'CPU result · response MQ','管理调用':'Management call','执行管理方法':'Run management method',
    '前向与采样结果':'Forward and sampling results','RpcReqOutput · 状态':'RpcReqOutput · status',
    '各 DP 引擎':'DP engines','仅 DP > 1':'Only DP > 1','前端路由参考':'Frontend routing input',
    '负载 / 运行状态':'Load / execution state','负载 / request-wave':'Load / request-wave',
    '外部 Router':'External Router','选择实例 / 请求角色':'Choose instances / roles','Prefill 实例':'Prefill instance',
    '计算输入并产生 KV':'Compute inputs and produce KV','Decode 实例':'Decode instance','接收 KV 并继续生成':'Receive KV and continue generation',
    'P 侧传输接口':'P transfer interface','按所选后端发送':'Send through selected backend','D 侧传输接口':'D transfer interface',
    '写入目标 GPU KV':'Write target GPU KV','Decode 准入':'Decode admission','接收完成 + 元数据 + 预算':'Received + metadata + budget',
    '选择 P':'Select P','选择 D':'Select D','本地 KV':'Local KV','KV 载荷':'KV payload','接收状态':'Receive status','条件齐备':'All conditions met',
    'Prefill 角色':'Prefill role','Decode 角色':'Decode role','L3 存储':'L3 storage','后端存储':'Backend storage',
    '可选持久层':'Optional persistent tier','主机恢复':'Host restore','绑定 metadata / 发起加载':'Bind metadata / start load',
    '恢复的目标地址':'Target KV addresses','恢复':'Restore','加载结果':'Load results','恢复 KV':'Restore KV',
    '后端写入 KV':'Backend writes KV','加载控制':'Load control',
    '等待调度':'Waiting','尚未执行':'Not executing','执行中':'Running','本轮选中':'Selected for this step',
    '请求完成':'Finished','记录终态':'Terminal state recorded','回到等待':'Requeued','回撤 / 抢占后重建':'Rebuild after preemption',
    '请求取消':'Aborted','按状态分别收尾':'Clean up according to state','预算与 KV 允许':'Budget and KV allow',
    'EOS / 长度终止':'EOS / length limit','显存压力':'Memory pressure','重新准入':'Readmit',
    '取消等待请求':'Abort a waiting request','取消执行中请求':'Abort a running request',
    '本批条件?':'Batch conditions?','形状 / 模式 / 后端':'Shape / mode / backend',
    '满足 Decode 图条件':'Decode graph eligible','满足 Extend 图条件':'Extend graph eligible','普通执行回退':'Ordinary forward fallback',
    '填输入缓冲再 replay':'Fill buffers, then replay','编译分段 / breakable':'Compiled segments / breakable',
    '无兼容候选，普通前向':'No compatible capture; forward','本轮输出':'Step output','仍计算当前输入':'Compute the current inputs',
    '候选请求':'Candidate request','预算与 KV 允许?':'Budget and KV allow?',
    'token 工作量 + 空间':'Token work + capacity','进入本轮计划':'Admit to this step','继续等待':'Keep waiting',
    '本轮不交给 GPU':'Not dispatched this step','本轮判断与处理':'Step decisions and handling',
    'D 侧 GPU KV':'D-side GPU KV','P 侧 GPU KV':'P-side GPU KV','D 侧请求':'D-side request',
    'Scheduler 准入':'Scheduler admission','poll 接收状态':'Poll receive status','rid / 请求状态':'rid / request state',
    '准入后执行':'Execute after admission','所选后端的 KV 来源':'KV source for chosen backend','所需 KV 可读':'Required KV readable',
    '接收 + 元数据 + 预算':'Receipt + metadata + budget','接收与驻留':'Receipt and residency','接收完成状态':'Receive completion',
    '控制 / 执行调用':'Control / execution calls','本地引用 / 依赖（静态）':'Local reference / dependency (static)',
    '本地绑定 / 引用':'Local binding / reference','本轮准入后继续 Decode':'Continue Decode after admission',
    '来源 / 控制':'Source / control','模型执行':'Model execution','目标设备地址':'Target device addresses',
    '载荷与依赖':'Payload and dependencies','启动加载 / 汇集结果':'Start load / collect results',
    'KV 载荷、控制与可用条件':'KV payload, control and readiness','KV 经后端交接':'KV via backend',
    'P/D：先看三个角色':'P/D: three roles','职责组件':'Components','阅读条件':'Reading conditions',
    '判断与执行':'Decisions and execution','条件与结果':'Conditions and results','本轮能接纳这个请求吗？':'Can this request join the step?',
    '交接 / 条件':'Handoff / conditions','依赖与边界':'Dependencies and boundaries','当前主线':'Current path',
    '可选 DP 扩展：负载与 request-wave':'Optional DP: load and request-wave','独立管理通道的往返':'Separate management round trip',
    'token IDs → 文本':'token IDs → text','选择本轮任务':'Choose this step’s work','计划':'Plan',
    'P/D：实例、KV 载荷与准入':'P/D: instances, KV and admission','请求状态与资源收尾':'Request state and cleanup',
    '资源边界':'Resource boundaries','怎样读这张图':'How to read this diagram','模型服务组件':'Model-serving components',
    '分层恢复：控制与 KV 驻留':'Tier restore: control and KV residency','每轮选择一条图执行路径':'Choose one execution path per step',
    '本 rank 内的执行调用':'Rank-local execution calls','ScheduleBatch → 执行':'ScheduleBatch → execution',
    '图例':'Legend','检查':'Check','是':'Yes','否':'No'
  };
  function tr(s,en) {
    if(!en||!s)return s;
    if(text[s])return text[s];
    const numbered=s.match(/^(\d+ \/ )(.*)$/);if(numbered)return numbered[1]+tr(numbered[2],en);
    return s.replace('Scheduler 进程','Scheduler process').replace('Worker 进程','Worker process').replace('设备内存','Device memory').replace('通信拓扑：','Communication topology: ').replace('一次请求的消息交接：','Request message handoff: ').replace('：模块职责与调用关系',': roles and calls').replace(/：TP=(\d) 的进程与数据归属/,': TP=$1 processes and data');
  }
  function sourceUrl(asset,node) {
    const s=node.sources?.[0],r=asset.repository;
    return s?`${r.url}/blob/${r.revision}/${s.path}${s.line?'#L'+s.line+(s.end_line?'-L'+s.end_line:''):''}`:null;
  }
  function pickNode(asset,id,side,onSelect) {
    const role=window.AtlasRoles.items?.[side]?.find(r=>r.id===id);
    if(role)onSelect({side,id:role.node,roleId:role.id,label:role.name,role:role.summary,description:role.detail,source:role.path});
    else {
      const n=asset.nodes.find(v=>v.id===id);
      if(n)onSelect({side,id,label:n.label,role:[n.sublabel||n.label,tr(n.sublabel||n.label,true)],description:[n.explanation||n.sublabel||n.label,tr(n.explanation||n.sublabel||n.label,true)],source:sourceUrl(asset,n)});
    }
  }
  function Diagram({assetKey,side,en,selected,onSelect,activeNodes=[],activeEdges=[],staticEdges=false,overrides={},className=''}) {
    const asset=window.AtlasArchifyAssets[assetKey],ref=React.useRef(null);
    const uid=React.useId().replace(/[^a-zA-Z0-9_-]/g,'');
    const markup=React.useMemo(()=>asset.svg.replace(/(\s)id="([^"]+)"/g,(_,space,s)=>`${space}id="${uid}-${s}"`).replace(/url\(#([^)]*)\)/g,(_,s)=>`url(#${uid}-${s})`).replace(/aria-labelledby="([^"]+)"/g,(_,s)=>`aria-labelledby="${s.split(' ').map(v=>uid+'-'+v).join(' ')}"`),[asset,uid]);
    React.useLayoutEffect(()=>{
      const svg=ref.current.querySelector('svg');
      svg.classList.add('ac-native','ia-svg');svg.setAttribute('role','group');svg.setAttribute('aria-label',tr(asset.title,en));
      svg.querySelectorAll('.ac-pulse').forEach(n=>n.remove());
      svg.querySelectorAll('text,title,desc').forEach(n=>{
        if(!n.dataset.originalText)n.dataset.originalText=n.textContent;
        n.textContent=tr(n.dataset.originalText,en);
      });
      svg.querySelectorAll('[data-node-id]').forEach(g=>{
        const id=g.dataset.nodeId, isSelected=selected?.side===side&&(selected.roleId?selected.roleId===id:selected.id===id||selected.id===aliases[id]);
        g.classList.add('ia-node','is-clickable');g.classList.toggle('is-active',activeNodes.includes(id));g.classList.toggle('is-selected',!!isSelected);
        g.setAttribute('aria-pressed',!!isSelected);g.setAttribute('aria-label',tr(g.dataset.nodeLabel,en)+' · '+tr(g.dataset.nodeSublabel||'',en));
        if(overrides[id]){const t=g.querySelector('text[data-detail="context"]');if(t)t.textContent=overrides[id][en?1:0];}
        // Translation may need two short context lines. The node rectangle and
        // all routes retain Archify's coordinates; captions stay inside it.
        const rect=g.querySelector('rect.c-mask'),caption=g.querySelector('text[data-detail="context"]');
        if(rect&&caption){
          const box=rect.getBBox(),width=box.width-14,words=caption.textContent;
          const heading=g.querySelector('text[data-node-label]');
          if(heading){heading.setAttribute('y',box.y+24);heading.setAttribute('x',box.x+box.width/2);heading.style.fontSize='13px';const length=heading.getComputedTextLength();if(length>width)heading.style.fontSize=Math.max(10,13*width/length)+'px';}
          caption.setAttribute('y',box.y+48);caption.setAttribute('x',box.x+box.width/2);
          caption.style.fontSize='10.5px';
          if(caption.getComputedTextLength()>width){
            const tokens=en?words.split(/\s+/):words.match(/[A-Za-z0-9_]+|[^A-Za-z0-9_]/g)||[];let first='',second='';
            for(const word of tokens){const next=first+(en&&first?' ':'')+word;caption.textContent=next;if(!second&&caption.getComputedTextLength()<=width)first=next;else second+=(en&&second?' ':'')+word;}
            caption.textContent='';
            [first,second].forEach((line,index)=>{const span=document.createElementNS('http://www.w3.org/2000/svg','tspan');span.setAttribute('x',caption.getAttribute('x'));span.setAttribute('y',box.y+box.height-17+index*11);span.textContent=line;caption.append(span);});
          }
        }
      });
      svg.querySelectorAll('[data-edge-id]').forEach(g=>{
        const id=g.dataset.edgeId,active=activeEdges.includes(id),isPath=g.tagName.toLowerCase()==='path';
        g.classList.add('ac-edge');g.classList.toggle('is-active',active);
        const path=isPath?g:g.querySelector('path[data-composition-edge-id],path[data-animate="edge"]');
        if(path&&active&&!(staticEdges===true||Array.isArray(staticEdges)&&staticEdges.includes(id))){
          const pulse=path.cloneNode(false);pulse.removeAttribute('id');pulse.removeAttribute('data-edge-id');pulse.removeAttribute('marker-end');pulse.removeAttribute('marker-start');
          pulse.setAttribute('class','ac-pulse');pulse.removeAttribute('style');pulse.setAttribute('aria-hidden','true');path.parentNode.insertBefore(pulse,path.nextSibling);
        }
      });
      if(asset.type==='workflow'){
        // Native workflows reserve a full lane width. Fit that decorative
        // frame to the authored graph for the narrower lesson column, while
        // preserving every node, label, arrow and its route geometry.
        let right=0;svg.querySelectorAll('[data-node-id],[data-edge-id],text').forEach(n=>{const b=n.getBBox();right=Math.max(right,b.x+b.width);});
        const view=svg.viewBox.baseVal,w=Math.min(view.width,Math.ceil(right+28));
        svg.setAttribute('viewBox',`${view.x} ${view.y} ${w} ${view.height}`);
        svg.querySelectorAll('rect.c-lane').forEach(n=>n.setAttribute('width',Math.max(0,w-2*Number(n.getAttribute('x')))));
      }
    });
    const activate=e=>{const node=e.target.closest('[data-node-id]');if(node&&ref.current.contains(node)){e.preventDefault();pickNode(asset,node.dataset.nodeId,side,onSelect);}};
    return h('div',{className:'ac-diagram '+className,'data-archify':assetKey,'data-diagram-type':asset.type},
      h('div',{className:'ac-scroll',ref,tabIndex:0,role:'region','aria-label':choose('图表可左右滚动或放大查看','Scroll or enlarge the diagram',en),onClick:activate,onKeyDown:e=>{if(e.key==='Enter'||e.key===' '){if(e.target.closest('[data-node-id]'))activate(e);}},dangerouslySetInnerHTML:{__html:markup}}));
  }
  function Note({en,children,english}){return h('p',{className:'ac-note'},en?english:children);}
  function Roles({side,en,selected,onSelect,showSources,onLocate}) {
    const [narrow,setNarrow]=React.useState(()=>matchMedia('(max-width:900px)').matches);
    React.useEffect(()=>{const mq=matchMedia('(max-width:900px)'),change=()=>setNarrow(mq.matches);mq.addEventListener('change',change);return()=>mq.removeEventListener('change',change);},[]);
    const items=window.AtlasRoles.items[side],chosen=selected?.side===side?items.find(r=>selected.roleId===r.id):null;
    return h('div',{className:'ir-map ac-roles',id:'ia-module-roles-'+side,'data-role-diagram':side},
      h(Diagram,{assetKey:'roles-'+side,side,en,selected,onSelect}),
      h(Note,{en,english:'Frames group objects; arrows show logical calls or results. The model section illustrates attention; kernels also support MLP and normalization. These are not process or duration boundaries.'},'边框区分对象组，箭头表示逻辑调用或结果交接。模型区以 Attention 为例；Kernel 也支撑 MLP、Norm。对象分组不代表进程边界或耗时。'),
      narrow?h(Note,{en,english:'Scroll the structure horizontally; read the duties in order below.'},'结构图可左右滑动；下方按顺序阅读各项职责。'):null,
      chosen?h('div',{className:'ac-selection'},h('strong',null,chosen.name),h('p',null,chosen.detail[en?1:0]),h('a',{href:chosen.path,target:'_blank',rel:'noreferrer'},choose('固定源码 ↗','Pinned source ↗',en)),h('button',{type:'button','data-locate-role':chosen.id,onClick:onLocate},choose('在请求图定位','Locate in request diagram',en))):null,
      h('details',{className:'ac-reader',open:showSources||narrow||undefined},h('summary',null,choose('逐项阅读职责与源码','Read responsibilities and sources',en)),
        items.map(item=>h('div',{key:item.id,'data-module-role':item.id},h('button',{type:'button',className:'ir-role','aria-pressed':chosen?.id===item.id,onClick:()=>pickNode(window.AtlasArchifyAssets['roles-'+side],item.id,side,onSelect)},h('strong',null,item.name),h('span',null,item.summary[en?1:0])),showSources?h('a',{href:item.path,target:'_blank',rel:'noreferrer'},choose('固定源码 ↗','Pinned source ↗',en)):null))));
  }
  function Process({side,en,tp,step,ownership,selected,onSelect}) {
    const sg=side==='sg',assetKey=`process-${side}-tp${tp}`,asset=window.AtlasArchifyAssets[assetKey];
    const ids=asset.nodes.map(n=>n.id),rankNodes=ids.filter(id=>/^(worker|runner|gpu)/.test(id));
    const activeNodes=step===0?['input']:step===1?['scheduler','scheduler-1','core','executor']:step===2?rankNodes:sg?['scheduler','return','input']:['core','input'];
    const activeEdges=ownership&&step!==2?[]:step===0?[]:step===1?[sg?'input-scheduler':'input-core','core-schedule','schedule-execute','rank-input']:step===2?asset.edges.filter(e=>/^(scheduler-worker|worker-runner|write-kv)/.test(e.id)).map(e=>e.id):sg?['output-token','output-text']:['core-output'];
    const overrides={};for(const id of ids.filter(id=>id.startsWith('gpu')))overrides[id]=step>=2?['R1 的 K/V 已写入','R1 K/V written']:step>=1?['已分配地址；尚未写 K/V','Addresses allocated; no K/V yet']:['尚未分配 R1 地址','R1 addresses not allocated'];
    return h(React.Fragment,null,h(Diagram,{assetKey,side,en,selected,onSelect,activeNodes,activeEdges,overrides}),
      h('div',{className:'ac-process-status','data-process-count':sg?tp+2:tp===1?2:4},h('b',null,choose('本图服务进程：','Serving processes shown: ',en)+(sg?tp+2:tp===1?2:4)),
        h('span',null,choose(step>=2?'R1：地址已分配 → 本 rank 的 K/V 已写入':step===1?'R1：本轮准入 → 账本分配地址 → 等待前向写入':'R1：尚未进入本轮执行',step>=2?'R1: addresses allocated → rank-local K/V written':step===1?'R1: admitted → addresses allocated → waiting for forward writes':'R1: not executing yet',en))),
      h(Note,{en,english:sg?'Each Scheduler process contains its rank-local worker and runner. TP peers coordinate one instance; K/V resides on the respective GPUs.':'TP1 uses uni in-process execution; TP2 uses local mp worker processes. Allocation authority is distinct from device residency.'},sg?'Scheduler、TpModelWorker 和 ModelRunner 在各 rank 的同一进程内；TP 同伴协同推进一个实例，K/V 驻留对应 GPU。':'TP1 使用 uni 进程内执行；TP2 使用本机 mp Worker 进程。地址分配权与 GPU 数据驻留分开。'));
  }
  function Wire({side,en,step,wireView,selected,onSelect,sourceOpen}) {
    const sg=side==='sg',assetKey=wireView==='topology'?'wire-'+side:sg&&step===2?'execute-messages-sg':sg&&step===3?'rpc-messages-sg':!sg&&step===3?'coord-messages-vl':'messages-'+side;
    const activeEdges=sg?[['request'],['tokens','text'],['execute','result'],['rpc-request','rpc-result']][step]:[['request'],['output'],['dispatch','execute','result'],['load-report','load']][step];
    const asset=window.AtlasArchifyAssets[assetKey],activeNodes=[...new Set(asset.edges.filter(e=>activeEdges.includes(e.id)).flatMap(e=>[e.from,e.to]))];
    const protocol=sg?[
      ['PUSH / PULL','tokenized request · ipc:// / tcp://'],['PUSH / PULL','Scheduler → Detokenizer → TokenizerManager'],['local call','Scheduler → TpModelWorker · rank local'],['DEALER RPC','RpcReqInput → RpcReqOutput · separate socket']
    ]:[['ROUTER / DEALER','Frontend → EngineCore · request frames'],['PUSH / PULL','EngineCore → Frontend · EngineCoreOutputs'],['Executor MQ','shared-memory ring / socket branches; CPU-readable results'],['DP > 1','DPCoordinator → Frontend · load / request-wave']];
    return h(React.Fragment,null,h(Diagram,{assetKey,side,en,selected,onSelect,activeNodes,activeEdges}),
      h(Note,{en,english:wireView==='topology'?'Arrows identify endpoints, payloads and directions. In-process calls are distinct from IPC messages.':step===3?'Management extension only. These messages do not carry KV payloads.':step===2&&sg?'This is an in-process call. TP synchronization is omitted, not removed.':'Messages show event order, not durations. Frontend output handling can overlap later engine work.'},wireView==='topology'?'箭头标明端点、载荷与方向；进程内调用与跨进程消息分别阅读。':step===3?'本步展示管理或协调扩展；这条通道不搬运 GPU KV。':step===2&&sg?'本步是进程内调用；图中省略 TP 同步，不表示它不存在。':'消息按先后排列，不按耗时比例绘制。前端处理输出时，核心可推进后续工作。'),
      step===2?h('details',{className:'ac-execution-detail'},h('summary',null,choose('执行侧怎样准备本批输入？','How are this batch’s inputs prepared?',en)),
        sg?h('p',null,choose('TpModelWorker 从 ScheduleBatch 构造 ForwardBatch，ModelRunner 准备本轮模型输入及 attention metadata，再执行前向与采样。它们属于本 rank 的进程内调用；TP collective 另由通信组完成。','TpModelWorker builds ForwardBatch from ScheduleBatch. ModelRunner prepares inputs and attention metadata, then executes forward and sampling. These are rank-local calls; TP collectives use the communication group.',en)):h(window.AtlasV2.Diagram,{chapter:'wire',step,en})):null,
      h('details',{className:'ac-protocol',open:sourceOpen||undefined},h('summary',null,choose('本步协议与载荷','Protocol and payload for this step',en)),h('strong',null,protocol[step][0]),h('p',null,protocol[step][1])));
  }
  function Disagg({side,en,step,selected,onSelect,depth}) {
    if(step===3)return h('div',{className:'ac-config'},h('h4',null,choose('扩展条件按能力分别判断','Check each deployment capability separately',en)),
      (side==='sg'?[
        ['Router','选副本 / P-D 角色；不决定本轮 token','Select replicas / P-D roles, not step tokens'],
        ['HiCache','按配置恢复或保存分层 KV；不是 P/D 必经步骤','Configured tiered KV; not required by every P/D request'],
        ['DP / EP / EPLB','并行与专家负载策略，需核对对应模型和配置','Parallel and expert-load policies depend on model/configuration']
      ]:[['TP / PP / DP / EP','切分与副本维度不同；名称不证明任意组合受支持','Different partition/replica dimensions; names do not prove all combinations'],['KV Connector','Scheduler 管控制 metadata，Worker 后端管载荷','Scheduler metadata and worker payload handling are distinct'],['ModelRunner V2','此提交的 external_launcher + PP>1 触发自动选择回退','At this revision external_launcher + PP>1 triggers automatic fallback']]).map(([title,zh,enText])=>h('div',{key:title},h('b',null,title),h('p',null,en?enText:zh))));
    const assetKey=(step===2?'restore-':depth==='overview'?'pd-overview-':step===1?'kv-flow-':'pd-')+side,asset=window.AtlasArchifyAssets[assetKey];
    const activeEdges=step===0?['select-p','select-d']:step===1?depth==='overview'?['payload']:['payload','received','readable','bind','admit']:asset.edges.map(e=>e.id);
    const activeNodes=[...new Set(asset.edges.filter(e=>activeEdges.includes(e.id)).flatMap(e=>[e.from,e.to]))];
    return h(React.Fragment,null,h(Diagram,{assetKey,side,en,selected,onSelect,activeNodes,activeEdges,staticEdges:['bind','readable'],overrides:side==='vl'&&step===1&&depth!=='overview'?{metadata:['KV metadata','KV metadata']}:{}}),
      h(Note,{en,english:step===2?'Optional tiering scenario, separate from P/D handoff. Only the restore direction is active.':'Frames identify P/D roles. Host placement is unspecified. KV receipt, metadata readiness and scheduler admission remain separate conditions.'},step===2?'这里是独立的可选缓存场景；本步只播放恢复方向。':'边框区分 P/D 角色，未指定是否同机。KV 接收完成、元数据就绪和本轮准入分别判断。'),
      step===1&&depth!=='overview'?h('details',{className:'ac-local-detail'},h('summary',null,choose('展开接收与可用条件','Expand receipt and readiness',en)),
        side==='vl'?h(window.AtlasV2.Diagram,{chapter:'disagg',step,en}):h('div',{className:'ac-readiness'},h('p',null,choose('Decode 侧先确认传输及请求元数据，再等待本轮预算与资源允许。接收完成不等于已经计算。','Decode confirms transfer and request metadata, then waits for scheduling budget and resources. Receipt is not execution.',en)),h('ol',null,h('li',null,choose('KV 已到目标位置','KV at target addresses',en)),h('li',null,choose('请求元数据与接收状态就绪','Request metadata and receive state ready',en)),h('li',null,choose('Scheduler 检查准入，才交给模型执行','Scheduler admits work before model execution',en))))):null);
  }
  function Lifecycle({side,en,step,selected,onSelect}) {
    const sg=side==='sg';const activeNodes=step===0?['finished']:step===1?['retracted']:['aborted'];
    const activeEdges=step===0?['complete']:step===1?['preempt']:sg?['abort-wait']:['abort-run'];
    const ledger=step===0?[
      [choose('请求状态','Request state',en),choose('终态已记录','Terminal state recorded',en)],
      [choose('KV 引用与缓存','KV references and cache',en),choose('归还请求引用；可复用前缀仍可保留','Release request references; reusable prefixes may remain',en)],
      [choose('执行侧槽位','Worker request slot',en),sg?choose('退出本轮使用关系','Leave active batch ownership',en):choose('finished_req_ids 通知 V2 后续清理','finished_req_ids notifies later V2 cleanup',en)]
    ]:step===1?[[choose('请求去向','Request location',en),'waiting'],[choose('KV 与进度','KV and progress',en),choose('按回撤路径释放，恢复时重建','Release on retraction; rebuild on resumption',en)],['ModelRunner',choose('执行已经确定的计划，不重新调度预算','Executes the plan; does not reschedule budgets',en)]]:[['R1',sg?choose('示例：waiting，尚未分配 KV','Example: waiting with no allocated KV',en):choose('示例：已建立执行侧请求槽位','Example: worker request slot exists',en)],['KV',sg?choose('未分配 KV；只移出等待队列','No allocated KV; remove from waiting queue',en):choose('按 delay_free 条件归还 blocks','Block release obeys delay_free',en)],['GPU',choose('已提交 kernel 不会因 CPU 取消而立即抢占','CPU cancellation does not preempt issued kernels',en)]];
    return h(React.Fragment,null,h(Diagram,{assetKey:'lifecycle-'+side,side,en,selected,onSelect,activeNodes,activeEdges,staticEdges:true}),h('dl',{className:'ac-ledger'},ledger.map(([a,b])=>h('div',{key:a},h('dt',null,a),h('dd',null,b)))));
  }
  function Graph({side,en,selected,onSelect}) {
    const sg=side==='sg',[mode,setMode]=React.useState(sg?'decode':'FULL');
    const modes=sg?[['decode','Decode graph'],['piece','Prefill graph'],['eager','Eager']]:[['FULL','FULL'],['PIECEWISE','PIECEWISE'],['NONE','NONE']];
    return h(React.Fragment,null,h('fieldset',{className:'rv-graph-controls'},h('legend',null,choose('教学示例：本轮只选择一条路径','Teaching example: one path runs this step',en)),h('div',{className:'rv-graph-choices'},modes.map(([id,label])=>h('button',{key:id,type:'button',[sg?'data-sg-graph-choice':'data-graph-choice']:id,'aria-pressed':mode===id,onClick:()=>setMode(id)},label)))),
      h(Diagram,{assetKey:'graph-'+side,side,en,selected,onSelect,activeNodes:['shape',mode,'output'],activeEdges:['choose-'+mode,mode+'-output']}),
      h(Note,{en,english:'Capture happens in advance. Every selected path computes the current inputs; no compatible capture falls back to ordinary forward.'},'图在前期捕获；本轮更新输入后只执行选中的一条。没有兼容图时仍走普通前向。'));
  }
  function Admission({side,en,step,selected,onSelect}) {
    return h(React.Fragment,null,h(Diagram,{assetKey:'admission-'+side,side,en,selected,onSelect,activeNodes:step===0?['candidate']:['candidate','budget','batch'],activeEdges:step===0?[]:['check','accept'],staticEdges:true}),
      h(Note,{en,english:'This summarizes token and KV admission conditions, not their complete source order. The budget display keeps the concrete work quantities.'},'流程概括 token 与 KV 准入条件，不代替源码中的完整检查顺序；预算图继续显示具体工作量。'));
  }
  return {Diagram,Roles,Process,Wire,Disagg,Lifecycle,Graph,Admission};
})();
