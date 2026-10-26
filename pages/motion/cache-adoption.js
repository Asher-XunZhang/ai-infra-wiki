/* Static cache comparisons: the same pool / prefix laid out as objects.
 * Mounted beside the existing cache animations, so the old playback and scene
 * URLs stay intact. Counts come from FoundationModel; this file only draws. */
(function(root){
'use strict';
const el=(tag,cls,text)=>{const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=text;return node;};
const chip=(text,cls='')=>el('span','ca-chip '+cls,text);
const stat=(text)=>el('span','ca-stat',text);
const fact=(text)=>el('li','ca-fact',text);

function head(title,subtitle,badge){
 const header=el('header','ca-head'), heading=el('div','ca-heading');
 heading.append(el('strong','',title),el('span','',subtitle));
 header.append(heading,el('span','ca-badge',badge));
 return header;
}

function side(id,title,badgeText,children,attrs={}){
 const article=el('article','ca-side');article.dataset.side=id;
 for(const [key,value] of Object.entries(attrs))article.dataset[key]=String(value);
 const header=el('header','ca-side-head');
 header.append(el('h3','',title),el('span','ca-side-badge',badgeText));
 article.append(header,...children);
 return article;
}

function meter(label,before,after,total){
 const row=el('div','ca-meter-row');
 row.append(el('span','ca-meter-label',label));
 const meter=el('div','ca-meter');
 meter.setAttribute('role','img');
 meter.setAttribute('aria-label',`${label}：${before}/${total} → ${after}/${total}`);
 const beforeBar=el('span','ca-meter-before');beforeBar.style.width=(before/total*100)+'%';
 const afterBar=el('span','ca-meter-after');afterBar.style.width=(after/total*100)+'%';
 meter.append(beforeBar,afterBar);
 row.append(meter,el('span','ca-meter-value',`${before}/${total} → ${after}/${total}`));
 return row;
}

function buildPages(){
 const model=root.FoundationModel;
 const d=model&&typeof model.cachePages==='function'?model.cachePages():null;
 if(!d)return null;
 const tailRemainder=d.paged.tailRemainder??d.paged.unusedCapacity;
 if(!Number.isFinite(tailRemainder))throw new Error('cachePages must expose tailRemainder');
 const section=el('section','cache-adoption');
 section.dataset.adoption='pages';
 section.dataset.beforeOccupied=String(d.occupancyOnly.beforeOccupied);
 section.dataset.afterOccupied=String(d.paged.afterOccupied);
 section.dataset.totalFree=String(d.totalFree);
 section.dataset.maxContiguous=String(d.maxContiguous);
 section.dataset.pageTable=String(d.paged.pageTableEntries);
 section.dataset.positionRows=String(d.paged.positionRows);
 section.dataset.validPositions=String(d.paged.validPositions);
 section.dataset.allocatedCapacity=String(d.paged.allocatedCapacity);
 section.dataset.allocatablePages=String(d.paged.allocatablePages);
 section.dataset.tailRemainder=String(tailRemainder);
 section.setAttribute('aria-label','分页对照：同一个物理池和同一组 3 个位置。起始 6 个旧页共 12 个有效位置，页 1、4 是 2 个可分配页；连续预留没有 3 个相邻空槽。按需分页把逻辑页 0（p1、p2）映射到物理页 4，逻辑页 1（p3）映射到物理页 1，是 2 条页表映射，另有 3 行逐 token 位置演算。分页后池内 8 页都有归属：6 个旧页仍归已有请求，页 4、页 1 这 2 页归新请求；有效位置 15、已分配容量 16、可分配页 0、末页余量 1。');
 section.append(head(
  '分页对照 · 同一池、同一需求',
  '新教学池例：8 页 × 2 槽；起始 6 个旧页共 12 个有效位置，页 1、4 是 2 个可分配页。两边看到同一个池、同一组 3 个位置；这不是下方空池分配动画的页 7 / 页 2 过程。',
  '新教学池例'
 ));

 const shared=el('p','ca-shared-fact');
 shared.append(el('strong','','起始物理状态完全相同：'),el('span','','有效位置 12、已分配容量 12、可分配页 2（页 1、4）；连续与分页的差别只在放置规则，不是换了池或缩小了需求。'));
 section.append(shared);

 const pool=el('div','ca-pool');
 pool.setAttribute('role','list');
 pool.setAttribute('aria-label','物理页池起始状态：6 个旧页已分配，页 1、4 未分配');
 for(let page=0;page<d.pageCount;page++){
  const card=el('div','ca-page');card.dataset.page=String(page);card.setAttribute('role','listitem');
  card.append(el('span','ca-page-name',`页 ${page}`));
  const slots=el('div','ca-slots');
  for(let slot=0;slot<d.pageSize;slot++){
   const occupied=Boolean(d.before[page][slot]);
   const target=d.mapping.find(m=>m.page===page&&m.slot===slot);
   const state=occupied?'is-occupied':target?'is-target':'is-free';
   const cell=el('span','ca-slot '+state,occupied?'占':target?target.position:'空');
   cell.dataset.slot=String(slot);
   if(target){cell.dataset.position=target.position;cell.dataset.target='true';}
   cell.setAttribute('aria-label',`页 ${page} 槽 ${slot}：${occupied?'已有占用':target?'起始空闲，分页目标 '+target.position:'起始空闲'}`);
   slots.append(cell);
  }
  card.append(slots);pool.append(card);
 }
 section.append(pool);
 const legend=el('p','ca-legend');
 legend.append(stat('灰底 = 已有占用'),stat('虚线 = 起始空闲'),stat('绿框 = 分页写入的 3 个空槽'));
 section.append(legend);

 const grid=el('div','ca-grid');
 const contiguous=side('contiguous','连续预留','放不下',[
  (()=>{
   const run=el('div','ca-run');
   run.append(el('span','ca-run-label','需求 3 个连续槽'));
   run.append(el('span','ca-run-slot is-free','空'),el('span','ca-run-slot is-free','空'),el('span','ca-run-missing','第 3 个相邻槽不存在'));
   return run;
  })(),
  el('ul','ca-facts',undefined),
  meter('有效位置 / 池容量',d.occupancyOnly.beforeOccupied,d.occupancyOnly.beforeOccupied,d.pageSize*d.pageCount)
 ],{fits:'false',maxRun:String(d.maxContiguous),metadataEntries:String(d.continuous.metadataEntries)});
 const contFacts=contiguous.querySelector('.ca-facts');
 contFacts.append(
  fact('起始：6 个旧页共 12 个有效位置；页 1、4 尚未分配，是 2 个可分配页。'),
  fact(`总空闲 ${d.totalFree} 槽，最大连续空闲 ${d.maxContiguous} 槽。`),
  fact('结果：没有 3 个相邻空槽，0 个位置写入；请求仍只有 12 个有效位置和 2 个可分配页。'),
  fact('连续方案本可只用 1 个“基址 + 长度”描述，但本例没有可用区间；总空闲不等于连续空间。')
 );

 const paged=side('paged','按需分页','2 条页表',[
  (()=>{
   const wrap=el('div','ca-maps');
   wrap.append(el('p','ca-map-label',`页表 · 逻辑页 → 物理页（${d.paged.pageTableEntries} 条）`));
   const rows=el('div','ca-map');
   for(const entry of d.pageTable){
    const row=el('div','ca-map-row is-page');
    row.dataset.logicPage=String(entry.logicPage);row.dataset.page=String(entry.page);row.dataset.offset=String(entry.offset);
    row.append(chip(`逻辑页 ${entry.logicPage}`,'is-logic'),el('span','ca-map-positions',entry.positions.join(' · ')),el('span','ca-arrow','→'),el('span','ca-map-target',`物理页 ${entry.page} · 偏移 ${entry.offset}`),el('span','ca-pte',`页表项 ${entry.entry}`));
    rows.append(row);
   }
   wrap.append(rows,el('p','ca-map-label',`位置演算 · 逐 token 查页 + 偏移（${d.paged.positionRows} 行，不是 ${d.paged.positionRows} 条页表）`));
   const posRows=el('div','ca-map');
   for(const entry of d.positionRows){
    const row=el('div','ca-map-row is-position');
    row.dataset.position=entry.position;row.dataset.page=String(entry.page);row.dataset.slot=String(entry.slot);
    row.append(chip(entry.position,'is-logic'),el('span','ca-arrow','→'),el('span','ca-map-target',`页 ${entry.page} · 槽 ${entry.slot}`),el('span','ca-pte',`偏移 ${entry.offset}`));
    posRows.append(row);
   }
   wrap.append(posRows);
   return wrap;
  })(),
  (()=>{
   const stats=el('div','ca-stats');
   stats.setAttribute('role','group');stats.setAttribute('aria-label','分页后的容量口径');
   stats.append(stat(`有效位置 ${d.paged.validPositions}`),stat(`已分配容量 ${d.paged.allocatedCapacity}`),stat(`可分配页 ${d.paged.allocatablePages}`),stat(`末页余量 ${tailRemainder}`));
   return stats;
  })(),
  el('ul','ca-facts',undefined),
  meter('有效位置 / 池容量',d.paged.beforeOccupied,d.paged.afterOccupied,d.pageSize*d.pageCount)
 ],{fits:'true',metadataEntries:String(d.paged.metadataEntries),usedPages:d.usedPages.join(',')});
 const pagedFacts=paged.querySelector('.ca-facts');
 pagedFacts.append(
  fact(`逻辑页 0 → 物理页 4；逻辑页 1 → 物理页 1：${d.paged.pageTableEntries} 条页表映射，物理页不连续。`),
  fact(`分页后池内 8 页都有归属：6 个旧页保持不变、仍归已有请求；页 4、页 1 这 2 个原可分配页归新请求。有效位置 ${d.paged.validPositions}、已分配容量 ${d.paged.allocatedCapacity}、可分配页 ${d.paged.allocatablePages}、末页余量 ${tailRemainder}（页 ${d.paged.tailFree.page} 槽 ${d.paged.tailFree.slots.join('、')}）。原有 6 页的所有者未被改写。`),
  fact(`${d.paged.positionRows} 行位置演算共用这 ${d.paged.pageTableEntries} 条页表映射；逐 token 查页和跨页管理是代价，不是另加 ${d.paged.positionRows} 条页表。`)
 );
 grid.append(contiguous,paged);
 section.append(grid);

 const note=el('p','ca-note');
 note.append(el('strong','','两边起始状态相同，收益有代价：'),el('span','','分页后池内 8 页都有归属：6 个旧页仍归已有请求，页 4、页 1 这 2 页归新请求；有效位置 15、已分配容量 16、可分配页 0、末页余量 1。1 个未用槽在新请求自己的末页内，不是池里可分配的空槽。收益来自用上分散空槽，代价是 2 条页表映射和不连续访问；这不是零碎片，也不是把池变大或把需求改小。'));
 section.append(note);
 return section;
}

function buildPrefix(){
 const model=root.FoundationModel;
 const d=model&&typeof model.cachePrefix==='function'?model.cachePrefix():null;
 if(!d)return null;
 const section=el('section','cache-adoption');
 section.dataset.adoption='prefix';
 section.dataset.independentTotal=String(d.independent.total);
 section.dataset.sharedTotal=String(d.shared.total);
 section.dataset.independentPrefix=String(d.independent.prefixCopies);
 section.dataset.sharedPrefix=String(d.shared.prefixCopies);
 section.dataset.suffixCopies=String(d.shared.suffixCopies);
 section.dataset.refs=String(d.shared.refs);
 section.dataset.retainedAtZero=String(d.retainedAtZero);
 section.setAttribute('aria-label','前缀共享对照：同一 R1/R2 与共同前缀 p1 p2 p3。未共享时前缀保存两份共 6 个位置，后缀 4 个位置，合计 10；共享后前缀 1 份 3 个位置加 2 个引用，后缀仍各自保存 4 个位置，合计 7。引用 0 不等于立即驱逐。');
 section.append(head(
  '前缀共享对照 · 同一 R1 / R2 与共同前缀',
  '共同前缀都是 p1 p2 p3；R1 的后缀 a1 a2、R2 的后缀 b1 b2 是各自独立的物理位置。左边未共享，右边把前缀收成一份并加两个引用。',
  '对象数量对照'
 ));

 const grid=el('div','ca-grid');
 const independent=side('independent','未共享 · 各存一份','10 个物理位置',[
  (()=>{
   const block=el('div','ca-requests');
   ['R1','R2'].forEach((name,i)=>{
    const request=el('div','ca-request');request.dataset.request=name;
    request.append(el('span','ca-request-name',`${name} · 前缀 + 后缀`));
    const prefixRow=el('div','ca-chip-row');
    d.independent.rows[i].forEach(token=>prefixRow.append(chip(token,'is-prefix')));
    const suffixRow=el('div','ca-chip-row');
    d.independent.suffixRows[i].forEach(token=>suffixRow.append(chip(token,'is-suffix-'+i)));
    request.append(prefixRow,el('span','ca-row-label','后缀'),suffixRow);
    block.append(request);
   });
   return block;
  })(),
  (()=>{
   const dup=el('p','ca-derived');
   dup.append(el('strong','','前缀被保存 2 次：'),el('span','','2 × 3 = 6 个前缀位置；后缀仍为 a1 a2 与 b1 b2，合计 4 个。'));
   return dup;
  })(),
  el('div','ca-total',`前缀 ${d.independent.prefixCopies} + 后缀 ${d.independent.suffixCopies} = ${d.independent.total}`)
 ],{prefixCopies:String(d.independent.prefixCopies),suffixCopies:String(d.independent.suffixCopies),refs:String(d.independent.refs)});

 const shared=side('shared','共享 · 1 份前缀 + 2 引用','7 个物理位置',[
  (()=>{
   const block=el('div','ca-shared-block');
   const prefixRow=el('div','ca-chip-row');
   d.shared.rows[0].forEach(token=>prefixRow.append(chip(token,'is-prefix is-shared')));
   const prefix=el('div','ca-shared-prefix');
   prefix.append(el('span','ca-shared-label','共享前缀'),prefixRow,el('span','ca-ref-count',`refs ${d.shared.refs}`));
   prefix.dataset.refs=String(d.shared.refs);
   const refs=el('div','ca-refs');
   refs.append(el('span','ca-ref-row','R1 → 同一份前缀'),el('span','ca-ref-row','R2 → 同一份前缀'));
   const suffixes=el('div','ca-suffixes');
   ['R1','R2'].forEach((name,i)=>{
    const row=el('div','ca-suffix-row');
    row.append(el('span','ca-request-name',`${name} 后缀`));
    d.shared.suffixRows[i].forEach(token=>row.append(chip(token,'is-suffix-'+i)));
    suffixes.append(row);
   });
   block.append(prefix,refs,suffixes);
   return block;
  })(),
  (()=>{
   const saved=el('p','ca-derived');
   saved.append(el('strong','','省下 3 个前缀位置：'),el('span','','1 × 3 = 3；后缀 a1 a2、b1 b2 各自保存，不合并。'));
   return saved;
  })(),
  el('div','ca-total',`前缀 ${d.shared.prefixCopies} + 后缀 ${d.shared.suffixCopies} = ${d.shared.total}`)
 ],{prefixCopies:String(d.shared.prefixCopies),suffixCopies:String(d.shared.suffixCopies),refs:String(d.shared.refs)});
 grid.append(independent,shared);
 section.append(grid);

 const condition=el('section','ca-condition');
 condition.append(el('h3','','复用条件：精确兼容才复用'));
 const checks=el('ul','ca-checks');
 for(const item of d.compatibility){
  const li=el('li','ca-check');
  li.dataset.check=item.key;
  li.append(el('span','ca-check-mark','✓'),el('span','',item.label));
  checks.append(li);
 }
 condition.append(checks);
 const mismatch=el('p','ca-condition-note');
 mismatch.append(el('strong','','任一不匹配 → 不共享：'),el('span','','回到各自保存。不能只凭后缀相同或语义相似复用；具体引擎还可能有隔离、权限和回载规则。'));
 condition.append(mismatch);
 section.append(condition);

 const life=el('section','ca-life');
 life.append(el('h3','','引用生命周期（下方原动画保留）'));
 const timeline=el('ol','ca-timeline');
 for(const step of d.refTimeline){
  const li=el('li','ca-step');
  li.dataset.refs=String(step.refs);
  li.append(el('span','ca-step-phase',step.phase),el('span','ca-step-refs',`refs ${step.refs}`));
  timeline.append(li);
 }
 life.append(timeline);
 const retain=el('p','ca-retain');
 retain.append(el('strong','','refs 0 ≠ 立即驱逐：'),el('span','','只表示没有活跃请求。缓存仍可按策略保留；驱逐、回载和隔离是另一个条件。'));
 life.append(retain);
 section.append(life);

 const note=el('p','ca-note');
 note.append(el('strong','','共享的是前缀对象，不是后缀：'),el('span','','左边 10 个物理位置，右边 7 个；引用数从 1 到 2 再到 1、0，R1 先结束也不会提前释放 R2 仍在用的前缀。'));
 section.append(note);
 return section;
}

const builders={pages:buildPages,prefix:buildPrefix};

function mount(kind,viewport){
 const builder=builders[kind];if(!builder)return null;
 const section=builder();if(!section)return null;
 const film=viewport.closest('.mechanism-film');
 film?.querySelectorAll(':scope > .cache-adoption').forEach(node=>node.remove());
 const anchor=viewport.closest('.motion-theater')||viewport;
 anchor.before(section);
 return {element:section,destroy(){section.remove();}};
}

root.CacheAdoption={mount,kinds:Object.keys(builders)};
})(typeof globalThis==='object'?globalThis:this);
