/* Responsive teaching figures: shared drawing and playback, no model state. */
(() => {
  'use strict';
  function create(root, initial={}) {
    const s={...initial}, $=id=>root.querySelector(`[data-id="${id}"]`);
    root.classList.add('lesson-diagram');
    root.innerHTML='<div class="wr-controls" data-id="wr-controls"></div><div class="wr-scene" data-id="wr-scene"></div><div class="wr-status" data-id="wr-status" role="status" aria-live="polite"></div><details data-id="wr-notes"><summary>模型边界</summary><div data-id="wr-evidence"></div></details>';
    let render=()=>{},timer=null;
    const status=t=>$('wr-status').textContent=t;
    const evidence=t=>{$('wr-evidence').textContent=t;};
    function stop(){clearInterval(timer);timer=null;}
    function controls(html){const active=document.activeElement;if(active?.type==='range'&&$('wr-controls').contains(active)&&html.includes('data-id="'+active.dataset.id+'"')){const strong=active.closest('label')?.querySelector('strong');if(strong)strong.textContent=active.value;}else $('wr-controls').innerHTML=html;}
    function bind(id,event,fn){const el=$(id);if(el)el['on'+event]=fn;}
    function change(fn){stop();const focus=document.activeElement?.dataset.id;fn();render();if(focus&&$(focus)&&!$(focus).disabled)$(focus).focus({preventScroll:true});}
    function stepBind(prefix,key,total){bind(prefix+'-prev','click',()=>change(()=>s[key]=Math.max(0,s[key]-1)));bind(prefix+'-next','click',()=>change(()=>s[key]=Math.min(total-1,s[key]+1)));}
    function animationControls(key,total,prefix){return button(prefix+'-play',timer?'暂停':'▶ 播放')+stepControls(s[key],total,prefix);}
    function animationBind(key,total,prefix){stepBind(prefix,key,total);bind(prefix+'-play','click',()=>{if(timer){stop();render();return;}if(s[key]>=total-1)s[key]=0;timer=setInterval(()=>{s[key]=Math.min(total-1,s[key]+1);if(s[key]>=total-1)stop();render();},2200);render();});}
    document.addEventListener('visibilitychange',()=>{if(document.hidden){stop();render();}});window.addEventListener('pagehide',stop);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));
const fmt=n=>Number(n.toFixed(2));
const chip=(name,sub='',kind='')=>`<span class="wr-token ${kind}">${esc(name)}${sub?`<small>${esc(sub)}</small>`:''}</span>`;
const node=(name,sub='',kind='')=>`<div class="wr-node ${kind}"><strong>${esc(name)}</strong>${sub?`<small>${esc(sub)}</small>`:''}</div>`;
const arrow='<span class="wr-arrow" aria-hidden="true">→</span>';
const path=items=>`<div class="wr-path">${items.join(arrow)}</div>`;
function button(id,label,active=false,disabled=false){return `<button type="button" data-id="${id}"${active?' aria-pressed="true"':''}${disabled?' disabled':''}>${label}</button>`;}
function select(id,label,items,value){return `<label>${label}<select data-id="${id}">${items.map(([v,t])=>`<option value="${esc(v)}"${String(v)===String(value)?' selected':''}>${esc(t)}</option>`).join('')}</select></label>`;}
function slider(id,label,v,min,max,step=1){return `<label>${label} <strong class="wr-count">${v}</strong><input data-id="${id}" type="range" value="${v}" min="${min}" max="${max}" step="${step}" aria-label="${label}"></label>`;}
function stepControls(n,total,prefix='st'){return button(prefix+'-prev','← 上一步',false,n===0)+`<span class="wr-small">${n+1} / ${total}</span>`+button(prefix+'-next','下一步 →',false,n===total-1);}
function svgStart(w,h,label){return `<svg class="wr-chart" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${esc(label)}"><title>${esc(label)}</title>`;}
const tx=(x,y,t,attrs='')=>`<text x="${x}" y="${y}" ${attrs}>${esc(t)}</text>`;
const line=(x1,y1,x2,y2,col='var(--line)',extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" ${extra}/>`;
function wrapText(text,width){const lines=[];let line='',used=0;const words=String(text||'').match(/[A-Za-z0-9_]+(?:[.\-–][A-Za-z0-9_]+)*|[^\x00-\x7f]|\s|./g)||[];for(const word of words){const n=[...word].reduce((sum,c)=>sum+(/[\x00-\x7f]/.test(c)?7:12),0);if(n>width){for(const c of word){const cw=/[\x00-\x7f]/.test(c)?7:12;if(used+cw>width&&line){lines.push(line);line='';used=0;}line+=c;used+=cw;}continue;}if(used+n>width&&line){lines.push(line.trimEnd());line='';used=0;}if(!line&&/^\s+$/.test(word))continue;line+=word;used+=n;}if(line)lines.push(line);return lines;}
const phaseColor={gpu:'var(--compute)',cpu:'var(--prepare)',message:'var(--transfer)',copy:'var(--transfer)',kv:'var(--transfer)',wait:'var(--waiting)',ready:'var(--available)'};
function ppLegend(){return `<div class="wr-legend">${[['compute','GPU 计算'],['prepare','CPU 操作'],['transfer','传输'],['waiting','等待'],['available','就绪 / 可推进']].map(([k,v])=>`<span><i style="background:var(--${k})"></i>${v}</span>`).join('')}</div>`;}
function mark(n,label){return {start:n.start,end:n.end,fill:phaseColor[n.kind]||'var(--line)',label:label||n.owner};}
function railChart(rows,min,max,label,at=null){
 if(max<=min)max=min+1;
 const w=$('wr-scene').clientWidth,L=w<450?57:76,R=w-8,x=t=>L+(t-min)/(max-min)*(R-L);let cursor=38;
 const layout=rows.map(row=>{const y=cursor,notes=wrapText(row.note,R-L);cursor+=40+17*notes.length;return {row,y,notes};});const H=cursor+22;
 let svg=svgStart(w,H,label);
 for(let i=0;i<4;i++){const t=min+(max-min)*i/3;svg+=line(x(t),24,x(t),H-26)+tx(x(t),15,fmt(t),i===0?'':i===3?'text-anchor="end"':'text-anchor="middle"');}
 layout.forEach(({row,y,notes})=>{svg+=tx(0,y+15,row.label)+`<rect x="${L}" y="${y}" width="${R-L}" height="24" fill="var(--soft)"/>`;
 row.bars.forEach(b=>{const a=Math.max(min,b.start),z=Math.min(max,b.end);if(z<=a)return;const bw=Math.max(1,x(z)-x(a));svg+=`<rect x="${x(a)}" y="${y}" width="${bw}" height="24" fill="${b.fill}"/>`;if(b.label&&wrapText(b.label,bw-10).length===1)svg+=tx(x(a)+bw/2,y+16,b.label,'text-anchor="middle"');});
 if(row.gate!==undefined&&row.gate>=min&&row.gate<=max)svg+=`<path d="M ${x(row.gate)-4} ${y+26} L ${x(row.gate)+4} ${y+26} L ${x(row.gate)} ${y+32} Z" fill="var(--ink)"/>`;
 notes.forEach((t,i)=>svg+=tx(L,y+43+i*17,t));});
 if(at!==null)svg+=line(x(at),22,x(at),H-27,'var(--green)','stroke-width="2" stroke-dasharray="4 3"');
 return svg+tx(R,H-2,'教学时间 u →','text-anchor="end"')+'</svg>';
}

    let width=0;
    const api={s,$,esc,clamp,fmt,chip,node,path,button,select,slider,stepControls,stepBind,status,evidence,bind,controls,change,animationControls,animationBind,svgStart,tx,line,wrapText,phaseColor,ppLegend,mark,railChart,stop,
      start(fn){render=fn;render();new ResizeObserver(()=>{const w=root.clientWidth;if(w>0&&Math.abs(w-width)>1){width=w;render();}}).observe(root);},render(){render();}};
    return api;
  }
  window.LearningDiagram={create};
})();
