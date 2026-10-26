const assert=require('node:assert/strict');
const M=require('../../pages/foundations/model.js');
for(const [key,d] of Object.entries(M.definitions))for(let v=0;v<d.options.length;v++)for(let s=0;s<d.steps.length;s++){const state=M.snapshot(key,v,s);assert.equal(state.step,s);assert.equal(state.last,d.steps.length-1);}
assert.equal(M.examples[0].pieces.join(''),M.examples[0].text);assert.deepEqual(M.examples[0].ids,[104100,100134,104455,1773]);assert.equal(M.examples[1].ids[1],M.examples[1].ids[3]);
for(const v of [0,1]){const d=M.attention(v);assert(Math.abs(d.weights.reduce((a,b)=>a+b,0)-1)<1e-10);assert(d.weights.every((x,i)=>i<=d.query||x===0));assert.equal(d.output.length,2);}
assert.equal(M.snapshot('generation',0,2).kv.length,2);assert.equal(M.snapshot('generation',0,4).kv.length,2,'new token has not yet completed a forward');assert.equal(M.snapshot('generation',0,5).kv.length,3,'KV appears after computing the new input');assert.equal(M.snapshot('generation',0,6).kv.at(-1),'学习');assert.equal(M.snapshot('generation',1,6).visible,'休息一下');
const sampled=M.snapshot('generation',1,6);
for(const [u,word]of [[0,'学习'],[.5999,'学习'],[.6,'休息'],[.75,'休息'],[1-Number.EPSILON,'。']])assert.equal(M.chooseToken(sampled.scores,u).selected,word,'sample uses its probability interval');
assert.equal(sampled.choice.draw,.75);assert.equal(sampled.nextChoice.draw,.25);assert.equal(sampled.nextChoice.selected,'一下');assert.equal(M.chooseToken(sampled.scores).selected,'学习','greedy shares the same first distribution');
assert.equal(M.snapshot('cache',0,3).computed.reduce((a,b)=>a+b,0),15);assert.equal(M.snapshot('cache',1,3).computed.reduce((a,b)=>a+b,0),6);
assert.equal(M.snapshot('cache',2,1).ownedPages,1);assert.equal(M.snapshot('cache',2,1).written,0);assert.equal(M.snapshot('cache',2,3).ownedPages,2);assert.equal(M.snapshot('cache',2,3).written,3);assert.deepEqual(M.snapshot('cache',3,3).holders,['R2']);assert.equal(M.snapshot('cache',3,4).retained,true);
const pages=M.cachePages();
assert.equal(pages.pageSize*pages.pageCount,pages.occupancyOnly.beforeOccupied+pages.totalFree);
assert.equal(pages.before.flat().filter(Boolean).length,12);assert.equal(pages.totalFree,4);
assert.equal(pages.maxContiguous,2);assert.equal(pages.continuous.fits,false);assert.equal(pages.continuous.maxRun,2);assert.equal(pages.continuous.metadataEntries,1);
// Six old pages keep their owners. Only the two free pages are assigned, and
// three new positions use two page-table entries plus three address rows.
assert.equal(pages.occupiedPages.length,6);
assert.deepEqual(pages.occupiedPages.slice().sort((a,b)=>a-b),[0,2,3,5,6,7]);
for(const page of pages.occupiedPages)assert.deepEqual(pages.after[page],pages.before[page],'an allocated old page must not change owner or slots');
assert.equal(pages.mapping.length,3,'three new positions');
assert.deepEqual(pages.usedPages.slice().sort((a,b)=>a-b),[1,4],'two new pages');
assert.equal(pages.pageTable.length,2,'two logical pages -> two page-table entries');
assert.equal(pages.positionRows.length,3,'three per-token address rows, not three page-table entries');
assert.equal(pages.paged.pageTableEntries,2);assert.equal(pages.paged.metadataEntries,2);assert.equal(pages.paged.positionRows,3);
assert.deepEqual(pages.pageTable.map(e=>[e.logicPage,e.page,e.positions.length]),[[0,4,2],[1,1,1]]);
assert.deepEqual(pages.positionRows.map(r=>[r.position,r.page,r.slot]),[['p1',4,0],['p2',4,1],['p3',1,0]]);
assert.equal(new Set(pages.mapping.map(m=>m.page+':'+m.slot)).size,3);
for(const m of pages.mapping){assert.equal(pages.before[m.page][m.slot],null);assert.equal(pages.after[m.page][m.slot],m.position);}
for(const row of pages.positionRows)assert.equal(pages.after[row.page][row.slot],row.position);
assert.equal(pages.occupiedUnchanged,true);
assert.equal(pages.paged.afterOccupied,15);assert.equal(pages.paged.validPositions,15,'12 old + 3 new positions');
assert.equal(pages.paged.allocatedPages,8);assert.equal(pages.paged.allocatedCapacity,16);
assert.equal(pages.paged.allocatablePages,0);assert.equal(pages.occupancyOnly.afterAllocatablePages,0);
assert.equal(pages.paged.tailRemainder,1);assert.equal(pages.paged.unusedCapacity,1);assert.equal(pages.paged.tailSlots,1);
assert.equal(pages.paged.validPositions+pages.paged.unusedCapacity,pages.paged.allocatedCapacity,'capacity conservation');
assert.equal(pages.paged.afterOccupied-pages.occupancyOnly.beforeOccupied,pages.mapping.length);
assert.equal(pages.paged.allocatedCapacity-pages.occupancyOnly.beforeCapacity,pages.usedPages.length*pages.pageSize);
assert.equal(pages.paged.tailFree.page,1);assert.deepEqual(pages.paged.tailFree.slots,[1]);
assert.notDeepEqual(pages.usedPages,[7,2],'new pool fixture stays distinct from the old page 7/2 animation');
const prefix=M.cachePrefix();
assert.equal(prefix.independent.prefixCopies,prefix.prefix.length*2);assert.equal(prefix.independent.prefixCopies,6);
assert.equal(prefix.shared.prefixCopies,prefix.prefix.length);assert.equal(prefix.shared.prefixCopies,3);
assert.equal(prefix.independent.suffixCopies,prefix.r1Suffix.length+prefix.r2Suffix.length);
assert.equal(prefix.independent.suffixCopies,prefix.shared.suffixCopies);
assert.equal(prefix.independent.total,prefix.independent.prefixCopies+prefix.independent.suffixCopies);
assert.equal(prefix.shared.total,prefix.shared.prefixCopies+prefix.shared.suffixCopies);
assert.equal(prefix.independent.total,10);assert.equal(prefix.shared.total,7);assert.equal(prefix.savings.totalPositions,3);
assert.equal(prefix.shared.refs,2);assert.deepEqual(prefix.refTimeline.map(x=>x.refs),[1,2,1,0]);
assert.equal(prefix.retainedAtZero,true);assert.equal(prefix.evictionSeparate,true);assert.equal(prefix.mismatchNoReuse,true);
assert.equal(prefix.compatibility.length,3);assert(prefix.compatibility.every(x=>x.ok));
assert.deepEqual(M.schedule(0,2).rounds[1],['B']);assert.deepEqual(M.schedule(1,2).rounds[1],['B','C']);for(const v of [0,1])for(let s=0;s<5;s++){const d=M.schedule(v,s),ids=[...d.active,...d.waiting,...d.done];assert.equal(new Set(ids).size,ids.length);assert.equal(ids.length,s?3:2);assert(d.rounds.every(r=>r.length<=2));}
assert.equal(M.snapshot('hardware',2,4).bound,8);assert.equal(M.snapshot('hardware',1,4).bound,4);
assert.deepEqual(M.snapshot('parallel',1,1).lanes,['MB1',null]);assert.deepEqual(M.snapshot('parallel',1,2).lanes,['MB2','MB1']);
assert.deepEqual(M.snapshot('communication',0,4).result,[11,22]);assert.deepEqual(M.snapshot('communication',1,4).result,[1,2,10,20]);assert.equal(M.snapshot('communication',2,1).arrived,false);
assert.equal(M.snapshot('deployment',1,3).decode,false);assert.equal(M.snapshot('deployment',1,4).decode,true);
for(const [v,a] of [[0,2],[1,0],[2,4]]){const d=M.snapshot('speculation',v,4);assert.equal(d.output.length,a+1);assert.deepEqual(d.validKV,['根',...d.draft.slice(0,a)]);assert.notDeepEqual(d.validKV,d.output);}
assert.equal(M.snapshot('serving',2,3).clientDone,true);assert.equal(M.snapshot('serving',2,3).resourceHeld,true);assert.equal(M.snapshot('serving',2,3).cancelReceived,false);
assert.equal(M.snapshot('metrics',2,4).p95,190);assert.equal(M.snapshot('metrics',3,4).p95,180);assert.equal(M.snapshot('metrics',3,4).failed,2);
assert.throws(()=>M.snapshot('tokens',999,0));assert.throws(()=>M.snapshot('tokens',0,-1));


const throughput=M.snapshot('metrics',4,4);assert.equal(throughput.completed/throughput.window,throughput.qps);assert.equal(throughput.outputCounts.reduce((a,b)=>a+b,0)/throughput.window,throughput.outputTPS);

console.log('PASS: 12 teaching models; token fixtures, causal weights, KV ownership, batch conservation, data dependencies, statistics boundaries.');
