/* Inline architecture expansion. One deterministic clock drives every payload. */
(() => {
  'use strict';
  const M = window.LayerAtlasFocusModel, NS = 'http://www.w3.org/2000/svg';
  const colors = ['var(--la-signal)', 'light-dark(#3977a3,#88bce2)', 'var(--la-residual)', 'light-dark(#8a6098,#d7a6e7)'];
  const fmt = v => Number(v.toFixed(2)).toString();
  const vector = a => '[' + a.map(fmt).join(', ') + ']';
  const ease = x => x * x * (3 - 2 * x);
  const between = (value, start, end) => Math.max(0, Math.min(1, (value - start) / (end - start)));
  const mix = (from, to, p) => from.map((v, i) => v + (to[i] - v) * ease(p));
  function el(tag, attrs = {}, parent) {
    const e = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, String(v)));
    if (parent) parent.append(e); return e;
  }
  function text(parent, x, y, value, size = 18, anchor = 'middle', className = '') {
    const e = el('text', { x, y, 'font-size': size, 'text-anchor': anchor, class: className }, parent); e.textContent = value; return e;
  }
  function card(parent, x, y, w, h) { return el('rect', { x: x-w/2, y:y-h/2, width:w, height:h, rx:10, class:'laf-box' }, parent); }
  function build(stage, scene, narrow) {
    const s = el('svg', { viewBox: `0 -22 ${narrow ? 460 : 900} ${(narrow ? (['attention','generation','head'].includes(scene.kind) ? 1000 : 880) : 500)+22}`, role:'img', 'aria-label':scene.title, 'data-laf-kind':scene.kind }, stage);
    const defs = el('defs', {}, s), marker = el('marker', {id:'laf-arrow',viewBox:'0 0 8 8',refX:7,refY:4,markerWidth:7,markerHeight:7,orient:'auto'}, defs);
    el('path',{d:'M0 0L8 4L0 8Z',fill:'var(--la-line)'},marker);
    const wires = el('g',{},s), nodes = el('g',{},s), packets = el('g',{},s), updates = [];
    function bank(x,y,label,values,opts={}) {
      const w=opts.width||160,h=values.length*30+62+(opts.footer?30:0),g=el('g',{'data-laf-bank':label},nodes);
      card(g,x,y,w,h);const heading=text(g,x,y-h/2+24,label,17);
      const rows=values.map((v,i)=>{
        const yy=y-h/2+54+i*30;
        text(g,x-w/2+12,yy,opts.names?opts.names[i]:String(i+1),15,'start','laf-muted');
        el('line',{x1:x-8,y1:yy-16,x2:x-8,y2:yy+2,stroke:'var(--la-border)'},g);
        const bar=el('rect',{y:yy-14,height:14,rx:3,fill:opts.color||colors[0]},g);
        const num=text(g,x+28,yy,'—',16,'start','laf-number');return {bar,num};
      });
      return {x,y,w,h, group:g, caption(value){ heading.textContent=value; }, update(a,amount=1,showNumbers=amount>=0.999){
        const max=opts.scale||Math.max(1.5,...values.map(Math.abs),...a.map(Math.abs));
        rows.forEach((r,i)=>{const width=Math.abs(a[i])/max*27*ease(amount);r.bar.setAttribute('x',a[i]<0?x-8-width:x-8);r.bar.setAttribute('width',width);r.bar.setAttribute('opacity',amount);r.num.textContent=showNumbers?fmt(a[i]):'—';});
      }};
    }
    function edge(points, label='', color=colors[0]) {
      const d=points.map((p,i)=>(i?'L':'M')+p.join(' ')).join(' '), path=el('path',{d,class:'laf-wire'},wires);
      const g=el('g',{class:'laf-packet','data-laf-packet':label},packets);el('circle',{r:21},g);
      const bars=Array.from({length:4},(_,i)=>el('rect',{x:-11+i*7,y:-6,width:5,height:12,rx:2,fill:color},g));
      const name=text(g,0,-24,label,16);name.setAttribute('paint-order','stroke');name.setAttribute('stroke','var(--la-canvas)');name.setAttribute('stroke-width','5');
      const length=path.getTotalLength();
      return {path, update(progress,active,values=[1,1]){
        const visible=active&&progress>0.01&&progress<0.99;g.style.display=visible?'':'none';
        path.style.stroke=active?color:'var(--la-line)';
        path.style.strokeWidth=active?'2.4':'1.7';
        bars.forEach((b,i)=>{b.style.display=i<values.length?'':'none';const h=8+Math.min(12,Math.abs(values[i]||0)*6);b.setAttribute('height',h);b.setAttribute('y',-h/2);});
        if(!visible){g.removeAttribute('transform');g.setAttribute('opacity','0');return;}
        const p=path.getPointAtLength(length*progress);g.setAttribute('transform',`translate(${p.x} ${p.y})`);
        g.setAttribute('opacity',Math.min(1,progress*9,(1-progress)*9));
      }};
    }
    function link(a,b,label='',color=colors[0],rail=null) {
      if(narrow){const sx=a.x,sy=a.y+a.h/2,tx=b.x,ty=b.y-b.h/2;return edge(rail===null?[[sx,sy],[sx,(sy+ty)/2],[tx,(sy+ty)/2],[tx,ty]]:[[a.x+a.w/2,a.y],[rail,a.y],[rail,b.y],[b.x+b.w/2,b.y]],label,color);}
      const sx=a.x+a.w/2,tx=b.x-b.w/2;return edge([[sx,a.y],[(sx+tx)/2,a.y],[(sx+tx)/2,b.y],[tx,b.y]],label,color);
    }
    function drive(e,phase,values,f,start=0,end=.62){e.update(between(f.progress(phase),start,end),f.phase===phase,values);}
    const d=scene.data;
    // Horizontal travel depicts delivery. Only after delivery do result bars fill.
    const result = (f,i,arrival=.62) => between(f.progress(i),arrival,1);
    if(scene.kind==='ffn') {
      const p=narrow?[[230,70],[115,285],[345,285],[230,535],[230,775]]:[[80,235],[290,130],[290,355],[550,235],[810,235]];
      const gateScale=Math.max(1.5,...d.g.map(Math.abs),...d.a.map(Math.abs));
      const i=bank(...p[0],'输入 h₂ · 2维',d.h),u=bank(...p[1],'上投影 u · 3维',d.u),a=bank(...p[2],'门控 · 3维',d.a,{color:colors[2],footer:true,scale:gateScale}),z=bank(...p[3],'逐维乘积 z · 3维',d.z),o=bank(...p[4],'更新 ΔM · 2维',d.out);
      const e1=link(i,u,'h',colors[0]),e2=link(i,a,'h',colors[2]);
      const e3=narrow?edge([[u.x,u.y+u.h/2],[u.x,z.y],[z.x-z.w/2,z.y]],'u'):edge([[u.x+u.w/2,u.y],[z.x,u.y],[z.x,z.y-z.h/2]],'u');
      const e4=narrow?edge([[a.x,a.y+a.h/2],[a.x,z.y],[z.x+z.w/2,z.y]],'a',colors[2]):edge([[a.x+a.w/2,a.y],[z.x,a.y],[z.x,z.y+z.h/2]],'a',colors[2]);
      const e5=link(z,o,'z · 3维');
      const gateSource=text(nodes,a.x,a.y+a.h/2-12,'',15);const downLabel=text(nodes,narrow?230:690,narrow?665:150,'W_down · 3→2',17);
      updates.push(f=>{
        i.update(d.h);u.update(d.u,result(f,0));
        const gateProgress=f.progress(1);
        a.caption(f.complete(1)?'激活 a · 3维':gateProgress?'SiLU：g → a':'门控投影 g · 3维');
        a.update(mix(d.g,d.a,gateProgress),result(f,0),f.complete(0));
        gateSource.textContent=f.complete(0)?'原 g '+vector(d.g):'两路投影并行';
        z.update(d.z,result(f,2));o.update(d.out,result(f,3));
        drive(e1,0,d.h,f);drive(e2,0,d.h,f);drive(e3,2,d.u,f);drive(e4,2,d.a,f);drive(e5,3,d.z,f);
        downLabel.setAttribute('fill',f.phase===3?colors[0]:'var(--la-muted)');
      });
    } else if(scene.kind==='norm') {
      const p=narrow?[[230,75],[230,285],[230,500],[230,745]]:[[85,235],[320,100],[550,235],[810,235]];
      const scale=Math.max(1.5,...d.input.map(Math.abs),...d.out.map(Math.abs));
      const inputName=scene.id==='norm2'?'x₁':scene.id==='finalnorm'?'第 3 层输出':'x';
      const i=bank(...p[0],inputName+' · 2维',d.input,{scale}),stat={x:p[1][0],y:p[1][1],w:170,h:88};card(nodes,stat.x,stat.y,170,88);text(nodes,stat.x,stat.y-20,'整体 RMS',18);
      const scalar=text(nodes,stat.x,stat.y+23,'—',24,'middle','laf-number');
      const n=bank(...p[2],'除以 RMS · 2维',d.scaled,{scale}),o=bank(...p[3],'乘 γ 后 · 2维',d.out,{scale});
      const e1=link(i,stat,'x'),e2=link(stat,n,'RMS',colors[2]),e3=narrow?edge([[i.x-i.w/2,i.y],[35,i.y],[35,n.y],[n.x-n.w/2,n.y]],'原 x'):edge([[i.x,i.y+i.h/2],[i.x,400],[n.x,400],[n.x,n.y+n.h/2]],'原 x');
      const e4=link(n,o,'n');text(nodes,o.x,o.y+o.h/2+32,'γ = '+vector(d.gamma),17);
      updates.push(f=>{i.update(d.input);scalar.textContent=f.progress(0)>=.62?fmt(d.r):'—';scalar.setAttribute('opacity',f.progress(0)<.62?1:.4+.6*result(f,0));n.update(d.scaled,result(f,1));o.update(d.out,result(f,2));drive(e1,0,d.input,f);drive(e2,1,[d.r],f);drive(e3,1,d.input,f);drive(e4,2,d.scaled,f);});
    } else if(scene.kind==='residual') {
      const p=narrow?[[110,140],[350,140],[230,435],[230,725]]:[[170,125],[170,365],[520,245],[805,245]];
      const scale=Math.max(...d.input.map(Math.abs),...d.delta.map(Math.abs),...d.out.map(Math.abs));
      const i=bank(...p[0],d.base+' · 2维',d.input,{scale}),delta=bank(...p[1],d.deltaName+' · 2维',d.delta,{color:colors[2],scale}),sum={x:p[2][0],y:p[2][1],w:66,h:66};
      el('circle',{cx:sum.x,cy:sum.y,r:33,fill:'var(--la-signal-soft)',stroke:'var(--la-border)'},nodes);text(nodes,sum.x,sum.y+11,'+',40);
      const o=bank(...p[3],'相加结果 · 2维',d.out,{scale});
      const e1=narrow?edge([[i.x,i.y+i.h/2],[i.x,sum.y],[sum.x-sum.w/2,sum.y]],d.base):edge([[i.x+i.w/2,i.y],[sum.x,i.y],[sum.x,sum.y-sum.h/2]],d.base);
      const e2=narrow?edge([[delta.x,delta.y+delta.h/2],[delta.x,sum.y],[sum.x+sum.w/2,sum.y]],d.deltaName,colors[2]):edge([[delta.x+delta.w/2,delta.y],[sum.x,delta.y],[sum.x,sum.y+sum.h/2]],d.deltaName,colors[2]);
      const e3=link(sum,o,'逐维和');
      const status=text(nodes,sum.x,sum.y-55,'',17);
      const equations=d.out.map((v,j)=>text(nodes,sum.x,(narrow?815:440)+j*30,`${fmt(d.input[j])} + (${fmt(d.delta[j])}) = ${fmt(v)}`,18));
      updates.push(f=>{
        const sumProgress=f.progress(1);
        i.update(d.input);delta.update(d.delta);o.update(d.out,result(f,1,.78));
        status.textContent=sumProgress>=.38?'逐维加法完成':f.complete(0)?'两个向量就绪':f.p>=.44?'原表示已到，等待更新':'';
        equations.forEach(eq=>eq.setAttribute('opacity',ease(between(sumProgress,0,.38))));
        drive(e1,0,d.input,f,0,.44);drive(e2,0,d.delta,f,.2,.94);
        drive(e3,1,d.out,f,.38,.78);
      });
    } else if(scene.kind==='embed') {
      const q=narrow?[230,65]:[80,235],t=narrow?[230,345]:[420,235],r=narrow?[230,740]:[810,235];
      const id={x:q[0],y:q[1],w:110,h:62};card(nodes,...q,110,62);text(nodes,q[0],q[1]+6,'ID 17',22);
      card(nodes,t[0],t[1],300,245);text(nodes,t[0],t[1]-146,'训练得到的 E 表（节选）',19);
      const hi=el('rect',{x:t[0]-142,y:t[1]-26,width:284,height:53,rx:5,fill:'var(--la-signal-soft)',opacity:0},nodes);
      d.rows.forEach((row,j)=>{const y=t[1]-60+j*60;text(nodes,t[0]-125,y,`${row[0]} · ${row[1]}`,18,'start');text(nodes,t[0]+27,y,vector(row[2]),18,'start','laf-number');});
      const row={x:t[0],y:t[1],w:300,h:50},o=bank(...r,'初始向量 x · 2维',d.out);
      const e1=narrow?edge([[id.x-id.w/2,id.y],[35,id.y],[35,row.y],[row.x-row.w/2,row.y]],'17'):link(id,row,'17');
      const e2=narrow?edge([[row.x+row.w/2,row.y],[425,row.y],[425,o.y],[o.x+o.w/2,o.y]],'x · 2维'):link(row,o,'x · 2维');
      updates.push(f=>{hi.setAttribute('opacity',result(f,0));o.update(d.out,result(f,1));drive(e1,0,[17],f);drive(e2,1,d.out,f);});
    } else if(scene.kind==='attention') {
      const q=bank(...(narrow?[230,65]:[80,240]),'头 1 · 当前 Q′',d.q),rows=[],es=[],outs=[];
      q.group.dataset.lafAnchor='q';
      d.labels.forEach((label,j)=>{
        const x=narrow?230:405,y=narrow?270+j*155:100+j*135,w=narrow?230:235,h=112;
        const g=el('g',{'data-laf-anchor':j===0?'k':''},nodes);card(g,x,y,w,h);
        text(g,x,y-32,label,20);text(g,x,y-9,'K′ '+vector(d.keys[j])+'  V '+vector(d.values[j]),15);
        const score=text(g,x-w/2+12,y+19,'分数 —',16,'start');
        const weight=text(g,x+w/2-12,y+19,'',16,'end');
        const rail=el('rect',{x:x-w/2+12,y:y+31,width:w-24,height:9,rx:4,fill:'var(--la-border)'},g);
        const bar=el('rect',{x:x-w/2+12,y:y+31,width:0,height:9,rx:4,fill:colors[j]},g);
        rows.push({x,y,w,h,score,weight,rail,bar});
        es.push(narrow?edge([[q.x-q.w/2,q.y],[35+j*17,q.y],[35+j*17,y],[x-w/2,y]],'Q',colors[j]):link(q,{x,y,w,h},'Q',colors[j]));
      });
      const o=bank(...(narrow?[230,850]:[815,240]),'加权结果 o · 2维',d.out),sum={x:narrow?230:630,y:narrow?710:240,r:24};
      el('circle',{cx:sum.x,cy:sum.y,r:sum.r,fill:'var(--la-signal-soft)',stroke:'var(--la-border)'},nodes);text(nodes,sum.x,sum.y+8,'Σ',26);
      rows.forEach((r,j)=>{const yy=sum.y+(j-1)*12,xx=sum.x+(narrow?1:-1)*Math.sqrt(sum.r**2-((j-1)*12)**2),rail=narrow?385+j*18:548+j*18;outs.push(edge([[r.x+r.w/2,r.y],[rail,r.y],[rail,yy],[xx,yy]],'αV',colors[j]));});
      const eSum=narrow?edge([[sum.x,sum.y+sum.r],[o.x,o.y-o.h/2]],'o'):edge([[sum.x+sum.r,sum.y],[o.x-o.w/2,o.y]],'o');
      text(nodes,narrow?230:445,narrow?982:490,'未来位置：−∞ → 权重 0，不参与汇聚',16,'middle','laf-muted');
      updates.push(f=>{
        q.update(d.q);o.update(d.out,result(f,2,.88));
        const alpha=f.progress(1);
        rows.forEach((r,j)=>{
          r.score.textContent=f.progress(0)>=.62?'分数 '+fmt(d.scores[j]):'分数 —';
          r.rail.setAttribute('opacity',alpha>0?1:0);
          r.bar.setAttribute('width',(r.w-24)*d.weights[j]*ease(alpha));
          r.weight.textContent=alpha>0?'权重 '+fmt(d.weights[j]):'';
          r.weight.setAttribute('opacity',ease(alpha));
          drive(es[j],0,d.q,f);drive(outs[j],2,d.values[j].map(v=>v*d.weights[j]),f,0,.60);
        });
        eSum.update(between(f.progress(2),.66,.88),f.phase===2,d.out);
      });
    } else if(scene.kind==='heads') {
      const input={x:narrow?230:75,y:narrow?65:240,w:140,h:66};card(nodes,input.x,input.y,input.w,input.h);text(nodes,input.x,input.y+5,'同一份上下文',18);
      const hs=d.heads.map((h,j)=>bank(...(narrow?[115+j*230,300]:[310,120+j*250]),'头 '+(j+1)+' · 2维',h.out,{width:185,color:colors[j],footer:true}));
      hs[0].group.dataset.lafAnchor='q';
      const weights=hs.map((h,j)=>text(nodes,h.x,h.y+h.h/2-12,'α '+vector(d.heads[j].weights),15));
      const joined=bank(...(narrow?[230,560]:[570,240]),'拼接 · 4维',d.joined),o=bank(...(narrow?[230,795]:[815,240]),'W_O 混合 · 2维',d.out);
      const incoming=hs.map((h,j)=>link(input,h,'输入',colors[j])),outgoing=hs.map((h,j)=>narrow?edge([[h.x,h.y+h.h/2],[h.x,joined.y],[joined.x+(j?joined.w/2:-joined.w/2),joined.y]],'头 '+(j+1),colors[j]):edge([[h.x+h.w/2,h.y],[joined.x,h.y],[joined.x,joined.y+(j?joined.h/2:-joined.h/2)]],'头 '+(j+1),colors[j])),last=link(joined,o,'4维');
      updates.push(f=>{hs.forEach((h,j)=>h.update(d.heads[j].out,result(f,0)));weights.forEach(w=>w.setAttribute('opacity',result(f,0)));joined.update(d.joined,result(f,1));o.update(d.out,result(f,2));incoming.forEach(e=>drive(e,0,d.input,f));outgoing.forEach((e,j)=>drive(e,1,d.heads[j].out,f));drive(last,2,d.joined,f);});
    } else if(scene.kind==='rope') {
      const centers=narrow?[[230,185],[230,510]]:[[240,220],[665,220]],radius=100;
      const arrows=centers.map((c,j)=>{
        el('circle',{cx:c[0],cy:c[1],r:radius,fill:'none',stroke:'var(--la-border)','stroke-dasharray':'4 5'},nodes);
        el('path',{d:`M${c[0]-125} ${c[1]}H${c[0]+125}M${c[0]} ${c[1]-125}V${c[1]+125}`,stroke:'var(--la-border)',fill:'none'},nodes);
        const g=el('g',{'data-laf-anchor':j?'k':'q'},nodes);
        text(g,c[0],c[1]-147,j?`头 1 · K（${d.key}，位置 ${d.r}）`:`头 1 · Q（？，位置 ${d.p}）`,21);
        const original=j?d.k:d.q;
        const unit=radius/Math.max(1,Math.hypot(...d.q),Math.hypot(...d.k));
        el('path',{d:`M${c[0]} ${c[1]}l${original[0]*unit} ${-original[1]*unit}`,stroke:'var(--la-line)','stroke-dasharray':'5 5',fill:'none'},g);
        const line=el('path',{stroke:colors[j], 'stroke-width':4,'marker-end':'url(#laf-arrow)',fill:'none'},g);
        const point=el('circle',{r:7,fill:colors[j]},g),value=text(g,c[0],c[1]+151,'',19,'middle','laf-number');return {c,line,point,value,original,unit};
      });
      const score=text(nodes,narrow?230:450,narrow?772:431,'',20);
      const scaled=text(nodes,narrow?230:450,narrow?812:469,'',20);
      updates.push(f=>{
        const vectors=arrows.map((a,j)=>{
          const v=M.rotate(a.original,(j?d.r:d.p)*d.theta*ease(f.progress(0)));
          const x=a.c[0]+v[0]*a.unit,y=a.c[1]-v[1]*a.unit;
          a.line.setAttribute('d',`M${a.c[0]} ${a.c[1]}L${x} ${y}`);
          a.point.setAttribute('cx',x);a.point.setAttribute('cy',y);
          a.value.textContent=vector(v);
          return v;
        });
        const angle=Math.atan2(Math.abs(vectors[0][0]*vectors[1][1]-vectors[0][1]*vectors[1][0]),M.dot(...vectors));
        score.textContent=`夹角 ${Math.round(angle*180/Math.PI)}° · 点积 ${fmt(M.dot(...vectors))}`;
        scaled.textContent=f.complete(0)?`÷ √2  →  ${d.key}的匹配分数 ${fmt(d.out[0]/Math.SQRT2)}`:'';
        scaled.setAttribute('opacity',ease(f.progress(1)));
      });
    } else if(scene.kind==='layers') {
      const scale=Math.max(1.5,...d.states.flat().map(Math.abs));
      const labels=['第 1 层输出 y','第 2 层输出','第 3 层输出'];
      const banks=d.states.map((v,j)=>bank(...(narrow?[230,95+j*260]:[100+j*345,240]),labels[j],v,{scale}));
      const edges=banks.slice(0,-1).map((b,j)=>{const e=link(b,banks[j+1],'2 维表示');text(nodes,narrow?315:(b.x+banks[j+1].x)/2,narrow?(b.y+banks[j+1].y)/2:192,'第 '+(j+2)+' 层',19);return e;});
      updates.push(f=>{banks.forEach((b,j)=>b.update(d.states[j],j?result(f,j-1):1));edges.forEach((e,j)=>drive(e,j,d.states[j],f));});
    } else if(scene.kind==='head') {
      const i=bank(...(narrow?[230,65]:[85,235]),'最终表示 h · 2维',d.input),rows=[],ins=[],outs=[];
      d.names.forEach((name,j)=>{const r={x:narrow?230:420,y:narrow?270+j*100:90+j*105,w:220,h:86};card(nodes,r.x,r.y,r.w,r.h);text(nodes,r.x,r.y-17,name,19);text(nodes,r.x,r.y+6,'w = '+vector(d.vectors[j]),16);r.score=text(nodes,r.x,r.y+30,'',16);rows.push(r);ins.push(narrow?edge([[i.x-i.w/2,i.y],[35+j*16,i.y],[35+j*16,r.y],[r.x-r.w/2,r.y]],'h',colors[j]):link(i,r,'h',colors[j]));});
      const o=bank(...(narrow?[230,835]:[805,235]),'词表分数 · 4维',d.scores,{names:d.names,width:185});
      rows.forEach((r,j)=>{const ty=o.y-o.h/2+47+j*30,tx=narrow?o.x+o.w/2:o.x-o.w/2,rail=narrow?385+j*16:575+j*24;outs.push(edge([[r.x+r.w/2,r.y],[rail,r.y],[rail,ty],[tx,ty]],'logit',colors[j]));});
      updates.push(f=>{
        i.update(d.input);o.update(d.scores,result(f,0,.88));
        rows.forEach((r,j)=>{r.score.textContent=f.p>=.42?'点积 '+fmt(d.scores[j]):'';r.score.setAttribute('opacity',ease(between(f.p,.42,.65)));});
        ins.forEach(e=>drive(e,0,d.input,f,0,.42));
        outs.forEach((e,j)=>drive(e,0,[d.scores[j]],f,.65,.88));
      });
    } else if(scene.kind==='generation') {
      const scores=bank(...(narrow?[230,145]:[100,235]),'候选分数',d.scores,{names:d.names,width:185}),p=bank(...(narrow?[230,455]:[405,235]),'概率 · 总和 1',d.probabilities,{names:d.names,width:185});
      const chosen={x:narrow?230:725,y:narrow?725:140,w:220,h:72};card(nodes,chosen.x,chosen.y,chosen.w,chosen.h);text(nodes,chosen.x,chosen.y-60,'本轮选择',18);const word=text(nodes,chosen.x,chosen.y+8,'—',26);
      const ctx={x:narrow?230:725,y:narrow?940:375,w:narrow?380:290,h:85};card(nodes,ctx.x,ctx.y,ctx.w,ctx.h);text(nodes,ctx.x,ctx.y-60,'追加后的上下文',18);const line=text(nodes,ctx.x,ctx.y+5,'法国…哪里？',19);
      const e1=link(scores,p,'分数 · 4维'),e2=link(p,chosen,'概率 · 4维'),e3=edge([[chosen.x,chosen.y+chosen.h/2],[ctx.x,ctx.y-ctx.h/2]],'token');
      const ctxText='法国 · 首都 · ？';
      const selection=text(nodes,p.x,p.y+p.h/2+27,'',16);
      updates.push(f=>{
        scores.update(d.scores);p.update(d.probabilities,result(f,0));
        selection.textContent=f.progress(1)>.62?scene.variant?'累计概率跨过 0.8：'+d.names[d.selected]:'最高分：'+d.names[d.selected]:'';
        word.textContent=f.complete(1)?d.names[d.selected]:'—';
        line.textContent=f.complete(2)?ctxText+' · '+d.names[d.selected]:ctxText;
        drive(e1,0,d.scores,f);drive(e2,1,d.probabilities,f);drive(e3,2,[d.selected],f,0,1);
      });
    }
    return frame => { updates.forEach(fn=>fn(frame)); s.dataset.phase=String(frame.phase);s.dataset.time=frame.time.toFixed(4); };
  }
  function create(root, hooks) {
    const board=root.querySelector('[data-la-board]'),viewport=root.querySelector('[data-la-viewport]'),canvas=root.querySelector('[data-la-canvas]'),focus=root.querySelector('[data-la-focus]'),stage=root.querySelector('[data-laf-stage]');
    const context=root.querySelector('[data-laf-context]');
    const name=root.querySelector('[data-laf-title]'),formula=root.querySelector('[data-laf-formula]'),options=root.querySelector('[data-laf-options]'),speedInput=root.querySelector('[data-laf-speed]'),position=root.querySelector('[data-laf-position]');
    let id='',kind='',variant=0,scene=null,render=null,time=0,playing=false,raf=0,last=0,speed=1,animations=[],ghost=null,zoomLabel=null,revision=0,closing=false,topicAnimation=null;
    const reduced=matchMedia('(prefers-reduced-motion: reduce)');
    const api={get active(){return !!id;},get currentId(){return id;},get scene(){return scene;},get time(){return time;},get speed(){return speed;},open,close,present,toggle,prev,next,seek,reset};
    function ui(){if(!scene)return;const f=M.sample(scene,time),step=scene.steps[f.phase];render(f);if(name.textContent!==scene.title)name.textContent=scene.title;
      if(formula.dataset.equation!==step.formula){
        const template=document.getElementById('equation-'+step.formula);
        if(template)formula.replaceChildren(template.content.cloneNode(true));
        else formula.textContent=step.formula;
        formula.dataset.equation=step.formula;
      }
      const readout=`${f.phase+1} / ${scene.steps.length} · ${step.title}`;if(position.textContent!==readout)position.textContent=readout;
      hooks.frame({count:`${String(f.phase+1).padStart(2,'0')} / ${String(scene.steps.length).padStart(2,'0')}`,title:step.title,description:step.why,progress:time/scene.duration,playing,ended:time>=scene.duration,previous:time>0,next:time<scene.duration,reduced:reduced.matches});
      focus.dataset.time=time.toFixed(4);focus.dataset.kind=kind;focus.dataset.playing=String(playing);
    }
    function stop(){playing=false;if(raf)cancelAnimationFrame(raf);raf=0;last=0;}
    function tick(now){if(!playing)return;if(last)time=Math.min(scene.duration,time+(now-last)/1000*speed);last=now;if(time>=scene.duration)stop();ui();if(playing)raf=requestAnimationFrame(tick);}
    function toggle(){if(reduced.matches){next();return;}if(playing){stop();ui();return;}if(time>=scene.duration)time=0;playing=true;last=0;ui();raf=requestAnimationFrame(tick);}
    function seek(value){stop();time=Math.max(0,Math.min(scene.duration,value*scene.duration));ui();}
    function reset(){seek(0);}
    function next(){stop();time=scene.ends.find(t=>t>time+0.001)||scene.duration;ui();}
    function prev(){stop();time=[0,...scene.ends].filter(t=>t<time-0.001).pop()||0;ui();}
    function load(nextKind=kind,nextVariant=variant, transition=false){
      stop();
      topicAnimation?.cancel();
      stage.querySelector('.laf-outgoing')?.remove();
      const old=stage.firstElementChild;
      const inAttention=id==='attn';
      const fromAttention=['attention','heads','rope'].includes(old?.dataset.lafKind);
      const q=context.querySelector('[data-laf-context-item="q"]');
      const heads=context.querySelector('[data-laf-context-item="heads"]');
      const narrow=stage.clientWidth<600;
      const pivot=nextKind==='heads'||kind==='heads'?heads:q;
      const origin=old?.querySelector('[data-laf-anchor="q"]')?.getBoundingClientRect();
      kind=nextKind;variant=nextVariant;scene=M.make(kind,variant,id);time=0;
      stage.replaceChildren();
      render=build(stage,scene,narrow);
      layoutNarrow=narrow;
      context.hidden=!inAttention;
      if(!context.hidden){
        const main=M.make('attention').data;
        q.textContent='头 1 · 当前 Q′ '+vector(main.q);
        heads.textContent='头 1 · 汇聚结果';
        q.dataset.current=String(kind!=='heads');
        heads.dataset.current=String(kind==='heads');
      }
      mountOptions();ui();
      if(!transition||!old||reduced.matches)return;
      // Keep the shared Q / head reference in place as the local view changes.
      const incoming=stage.firstElementChild, height=stage.clientHeight;
      const host=stage.getBoundingClientRect(), anchor=origin||pivot.getBoundingClientRect();
      const x=Math.max(0,Math.min(host.width,anchor.left+anchor.width/2-host.left));
      const y=Math.max(0,Math.min(height,anchor.top+anchor.height/2-host.top));
      old.classList.add('laf-outgoing');old.setAttribute('aria-hidden','true');
      old.removeAttribute('role');
      for(const marker of old.querySelectorAll('marker'))marker.removeAttribute('id');
      stage.prepend(old);
      const fade=old.animate([{opacity:1},{opacity:0}],{duration:210,fill:'both'});
      const frames=inAttention&&fromAttention?[
        {opacity:0,clipPath:`circle(0px at ${x}px ${y}px)`},
        {opacity:1,clipPath:`circle(${Math.hypot(host.width,height)}px at ${x}px ${y}px)`}
      ]:[{opacity:0},{opacity:1}];
      topicAnimation=incoming.animate(frames,{duration:300,easing:'ease-out'});
      fade.finished.catch(()=>{}).then(()=>old.remove());
    }
    function choice(label,selected,fn){
      const b=document.createElement('button');b.type='button';b.textContent=label;
      b.setAttribute('aria-pressed',String(selected));
      b.addEventListener('click',()=>{
        const hadFocus=document.activeElement===b;
        fn();
        if(hadFocus)[...options.children].find(button=>button.textContent===label)?.focus({preventScroll:true});
      });
      options.append(b);
    }
    function mountOptions(){
      options.replaceChildren();
      if(id==='attn')for(const [key,label] of [['attention','汇聚信息'],['rope','RoPE 旋转'],['heads','多头合并']]){
        choice(label,kind===key,()=>{if(kind!==key)load(key,0,true);});
      }
      if(kind==='norm')for(const [v,label] of [[0,'输入原幅度'],[1,'输入整体 ×10']])choice(label,variant===v,()=>{if(variant!==v)load(kind,v);});
      if(kind==='rope')for(const [v,label] of [[0,'看「法国」的位置'],[1,'看「首都」的位置']])choice(label,variant===v,()=>{if(variant!==v)load(kind,v);});
      if(kind==='generation')for(const [v,label] of [[0,'贪心'],[1,'采样 · 固定随机数 0.8']])choice(label,variant===v,()=>{if(variant!==v)load(kind,v);});
    }
    function cancelMorph(){
      revision++;animations.forEach(a=>a.cancel());animations=[];
      ghost?.remove();ghost=null;zoomLabel?.remove();zoomLabel=null;
      focus.style.position='';focus.style.left='';focus.style.top='';focus.style.width='';
      focus.hidden=!id;
      viewport.dataset.laMorphing='false';
      closing=false;
    }
    function frozenClone(source){const clone=source.cloneNode(true),from=[source,...source.querySelectorAll('*')],to=[clone,...clone.querySelectorAll('*')];
      from.forEach((e,i)=>{const c=getComputedStyle(e);to[i].style.cssText=[...c].map(k=>k+':'+c.getPropertyValue(k)).join(';');for(const a of [...to[i].attributes])if(a.name==='id'||a.name==='role'||a.name==='tabindex'||a.name.startsWith('data-la-')||a.name.startsWith('aria-'))to[i].removeAttribute(a.name);});
      clone.classList.add('la-zoom-ghost');clone.setAttribute('aria-hidden','true');clone.inert=true;return clone;
    }
    function relative(e){const r=e.getBoundingClientRect(),v=viewport.getBoundingClientRect();return {x:r.left-v.left,y:r.top-v.top,w:r.width,h:r.height};}
    const transform=(from,to)=>`translate(${from.x-to.x}px,${from.y-to.y}px) scale(${from.w/to.w},${from.h/to.h})`;
    function carryTitle(node,from,to,opts){
      const label=document.createElement('span');
      label.className='la-zoom-label';label.setAttribute('aria-hidden','true');
      label.textContent=node.querySelector('.la-node-tag')?.textContent||'残差相加';
      Object.assign(label.style,{left:from.x+'px',top:from.y+'px'});
      viewport.append(label);zoomLabel=label;
      animations.push(label.animate([
        {transform:'translate(-50%,-50%)',opacity:1},
        {opacity:1,offset:.75},
        {transform:`translate(calc(${to.x-from.x}px - 50%),calc(${to.y-from.y}px - 50%))`,opacity:0}
      ],opts));
    }
    const center=box=>({x:box.x+box.w/2,y:box.y+box.h/2});
    function revealIfOutside(){
      const header=focus.querySelector('.laf-header');
      const bounds=header.getBoundingClientRect();
      const chrome=root.ownerDocument.querySelector('.site-header')?.getBoundingClientRect().bottom||0;
      if(bounds.top<chrome+8||bounds.bottom>innerHeight)header.scrollIntoView({block:'center',behavior:'instant'});
    }
    async function morph(node,opening){
      const rev=revision,oldHeight=viewport.clientHeight,source=relative(canvas),beforeNode=relative(node);
      const oldPanel=opening?null:relative(focus);
      if(!reduced.matches){
        ghost=frozenClone(canvas);
        Object.assign(ghost.style,{position:'absolute',left:source.x+'px',top:source.y+'px',width:source.w+'px',height:source.h+'px',margin:'0',zIndex:'5',transformOrigin:'0 0',pointerEvents:'none'});
        viewport.append(ghost);
      }
      if(!opening&&oldPanel){
        Object.assign(focus.style,{position:'absolute',left:oldPanel.x+'px',top:oldPanel.y+'px',width:oldPanel.w+'px'});
      }
      board.dataset.laView=opening?'focus':'overview';
      if(opening){focus.hidden=false;load();}
      hooks.view();
      if(reduced.matches){cancelMorph();if(opening)revealIfOutside();return true;}
      const target=relative(canvas),panel=relative(focus),newHeight=viewport.clientHeight;
      const opts={duration:410,easing:'cubic-bezier(.24,.65,.28,1)',fill:'both'};
      viewport.dataset.laMorphing='true';
      animations.push(ghost.animate([
        {transform:'none',opacity:1},{transform:transform(target,source),opacity:0}
      ],opts));
      animations.push(canvas.animate([{opacity:0},{opacity:1}],opts));
      animations.push(viewport.animate([{height:oldHeight+'px'},{height:newHeight+'px'}],opts));
      const caption=relative(name);
      if(opening){
        animations.push(focus.animate([{transform:transform(beforeNode,panel),opacity:.35},{transform:'none',opacity:1}],opts));
        animations.push(stage.animate([{opacity:0},{opacity:0,offset:.3},{opacity:1}],opts));
        carryTitle(node,center(beforeNode),{x:caption.x+Math.min(caption.w/2,75),y:caption.y+10},opts);
      }else{
        const destination=relative(node);
        animations.push(focus.animate([{transform:'none',opacity:1},{transform:transform(destination,panel),opacity:0}],opts));
        carryTitle(node,{x:caption.x+Math.min(caption.w/2,75),y:caption.y+10},center(destination),opts);
      }
      await Promise.allSettled(animations.map(a=>a.finished));
      if(revision!==rev)return false;
      cancelMorph();hooks.view();
      if(opening)revealIfOutside();
      return true;
    }
    function open(nextId,node){
      if(nextId===id)return;
      const wasOpen=!!id;
      cancelMorph();stop();
      id=nextId;const nextKind=M.keyFor(id);
      if(wasOpen){
        load(nextKind,0,true);hooks.view();revealIfOutside();
      }else{
        kind=nextKind;variant=0;
        morph(node,true);
      }
    }
    function present(nextId,nextKind,seconds,progress,running){
      if(id!==nextId||kind!==nextKind||variant!==0||board.dataset.laView!=='focus'||viewport.dataset.laMorphing==='true'){
        cancelMorph();
        id=nextId;
        board.dataset.laView='focus';focus.hidden=false;
        load(nextKind,0);
        hooks.view();
      }else if(playing){
        stop();
      }
      if(time!==seconds){time=seconds;ui();}
      focus.dataset.playing=String(running);
      stage.style.setProperty('--laf-enter',String(progress));
    }
    async function close(){
      if(!id||closing)return;
      stop();cancelMorph();
      const node=canvas.querySelector(`[data-la-node="${id}"]`);
      id='';closing=true;
      if(await morph(node,false)){
        hooks.close();node.focus({preventScroll:true});
      }
    }
    root.querySelector('[data-laf-close]').addEventListener('click',()=>{
      if(hooks.beforeClose)hooks.beforeClose();
      close();
    });
    speedInput.addEventListener('input',()=>{speed=Number(speedInput.value);root.querySelector('[data-laf-speed-label]').textContent=speed+'×';});
    let layoutNarrow;
    new ResizeObserver(()=>{
      if(!id)return;
      const narrow=stage.clientWidth<600;
      if(narrow===layoutNarrow)return;
      layoutNarrow=narrow;
      topicAnimation?.cancel();
      stage.replaceChildren();render=build(stage,scene,narrow);ui();
    }).observe(stage);
    document.addEventListener('visibilitychange',()=>{if(document.hidden&&playing){stop();ui();}});
    window.addEventListener('pagehide',()=>{stop();cancelMorph();});
    return api;
  }
  window.LayerAtlasFocus={create};
})();
