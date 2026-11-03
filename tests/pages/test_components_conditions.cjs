const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');

const near=(a,b,eps=1e-6,label='')=>assert.ok(Math.abs(a-b)<=eps,`${label?label+': ':''}${a} != ${b}`);

/* Focused renderer regression: changing an in-board condition must update the
 * board and ask the shared player to rewind to 0. Re-clicking the active option
 * or opening a detail panel must not rewind. No browser is needed. */

class FakeClassList{
 constructor(node){this.node=node;this.values=new Set();}
 add(...names){for(const name of names)this.values.add(name);this.sync();}
 remove(...names){for(const name of names)this.values.delete(name);this.sync();}
 contains(name){return this.values.has(name);}
 toggle(name,on){const next=on===undefined?!this.contains(name):!!on;next?this.add(name):this.remove(name);return next;}
 sync(){this.node.attrs.class=[...this.values].join(' ');}
}
class FakeNode{
 constructor(tag='div'){
  this.tagName=tag;this.attrs={};this.childNodes=[];this.parentNode=null;
  this.style={};this.dataset={};this.classList=new FakeClassList(this);
  this.listeners={};this.textContent='';this.clientWidth=720;
 }
 setAttribute(name,value){this.attrs[name]=String(value);if(name==='class'){this.classList.values=new Set(String(value).split(/\s+/).filter(Boolean));}}
 getAttribute(name){return Object.hasOwn(this.attrs,name)?this.attrs[name]:null;}
 removeAttribute(name){delete this.attrs[name];}
 append(...nodes){for(const node of nodes){node.parentNode=this;this.childNodes.push(node);}}
 insertBefore(node,reference){const index=this.childNodes.indexOf(reference);node.parentNode=this;this.childNodes.splice(index<0?this.childNodes.length:index,0,node);}
 addEventListener(type,listener){(this.listeners[type]||(this.listeners[type]=[])).push(listener);}
 dispatchEvent(event){
  event.target=this;event.currentTarget=this;
  for(const listener of [...(this.listeners[event.type]||[])])listener.call(this,event);
  if(event.bubbles&&!event.stopped&&this.parentNode)this.parentNode.dispatchEvent(event);
  return !event.defaultPrevented;
 }
 closest(){return null;}
 querySelector(){return null;}
 getTotalLength(){return 100;}
 getPointAtLength(length){return{x:length,y:0};}
}
class FakeEvent{
 constructor(type,init={}){this.type=type;this.bubbles=!!init.bubbles;this.detail=init.detail;this.defaultPrevented=false;this.stopped=false;}
 preventDefault(){this.defaultPrevented=true;}
 stopPropagation(){this.stopped=true;}
}
const document={createElementNS:(_ns,tag)=>new FakeNode(tag),createElement:tag=>new FakeNode(tag)};
const context={
 console,document,CustomEvent:FakeEvent,
 matchMedia:()=>({matches:false,addEventListener(){},removeEventListener(){}}),
 addEventListener(){},removeEventListener(){},
 requestAnimationFrame:fn=>{fn();return 1;},cancelAnimationFrame(){},
 setTimeout:()=>1,clearTimeout(){},
 ResizeObserver:class{observe(){}disconnect(){}},
};
context.window=context;
vm.createContext(context);
for(const file of ['pages/foundations/components-model.js','pages/motion/teaching-visuals.js','pages/motion/components.js'])
 vm.runInContext(fs.readFileSync(path.join(__dirname,'../..',file),'utf8'),context,{filename:file});

const rmsLesson=context.TeachingLessons['f-components'].find(s=>s.concept==='rmsnorm');
assert.deepEqual([...rmsLesson.data.gamma],[1,2],'RMSNorm lesson data matches the board default gamma');
assert.match(rmsLesson.intro,/默认 γ=\[1,2\]/,'RMSNorm intro states the default gamma');
const gammaFrame=rmsLesson.frames[3].caption;
assert.ok(gammaFrame.includes('γ=[1,1]')&&gammaFrame.includes('γ=[1,2]'),'gamma stage explains both gamma settings');

function walk(node,out=[]){out.push(node);for(const child of node.childNodes)walk(child,out);return out;}
function makeBoard(concept,width=720){
 const viewport=new FakeNode('div'),svg=new FakeNode('svg');viewport.clientWidth=width;
 let visual,position=0;const seeks=[];
 svg.addEventListener('component-seek',event=>{
  const step=Number(event.detail?.step);seeks.push(step);position=step;visual.paint(null,null,0,step);
 });
 visual=context.MechanismVisuals['f-components'](svg,viewport,{concept});
 return {
  svg,seeks,
  get position(){return position;},
  paint(p){position=p;visual.paint(null,null,0,p);},
  click(label){const node=walk(svg).find(n=>n.cmpLabelText===label||(n.attrs['aria-label']||'').startsWith(label));assert.ok(node,`missing control: ${label}`);node.dispatchEvent(new FakeEvent('click',{bubbles:true}));},
 };
}
function change(board,label,check){
 board.paint(5);
 const before=board.seeks.length;
 board.click(label);
 assert.equal(board.seeks.length,before+1,`${label} must seek once`);
 assert.equal(board.seeks[before],0,`${label} must rewind the shared player`);
 assert.equal(board.position,0,`${label} must leave the clock at 0`);
 check(board.svg);
}

const rope=makeBoard('rope');
change(rope,'只移 Q +1',svg=>{
 assert.equal(svg.dataset.condition,'q1');
 assert.equal(svg.dataset.qp,'3');
 assert.equal(svg.dataset.pks,'0,1,2');
 assert.equal(svg.dataset.progress,'0.000');
 assert.equal(svg.dataset.component,'rope');
 const header=walk(svg).find(n=>n.tagName==='text'&&n.attrs.class==='att-mode-sub');
 assert.ok(header&&header.textContent.includes('p3'),'header follows the moved Q');
 const legend=walk(svg).find(n=>n.tagName==='text'&&n.attrs.class==='cmp-legend');
 assert.ok(legend&&legend.textContent.includes('参照')&&legend.textContent.includes('点积'),'legend explains the reference direction and signed dot');
});
const ropeBefore=rope.svg.dataset.progress;
rope.paint(2.6);
const ropeSeeks=rope.seeks.length;
rope.click('只移 Q +1');
assert.equal(rope.seeks.length,ropeSeeks,'clicking the active condition must not reset');
assert.notEqual(rope.svg.dataset.progress,ropeBefore,'same-condition click keeps the current position');

// Opening the optional numeric layer is a view toggle, not a condition change.
const ropeDetailSeeks=rope.seeks.length;
rope.click('展开计算');
assert.equal(rope.seeks.length,ropeDetailSeeks,'detail toggle must not seek');
assert.equal(rope.position,2.6,'detail toggle keeps the paused position');
assert.equal(rope.svg.dataset.detail,'true');
const openDetail=walk(rope.svg).find(n=>n.cmpLabelText==='收起计算');
assert.ok(openDetail,'visible label switches to collapse');
assert.equal(openDetail.attrs['aria-label'],'收起计算');
assert.equal(openDetail.attrs['aria-expanded'],'true');
rope.click('收起计算');
assert.equal(rope.svg.dataset.detail,'false');
assert.equal(rope.position,2.6);
const closedDetail=walk(rope.svg).find(n=>n.cmpLabelText==='展开计算');
assert.equal(closedDetail.attrs['aria-label'],'展开计算');
assert.equal(closedDetail.attrs['aria-expanded'],'false');

const rms=makeBoard('rmsnorm');
rms.paint(0);
assert.equal(rms.svg.dataset.gamma,'[1, 2]','RMSNorm starts from the non-uniform default gamma');
change(rms,'γ=[1,1]',svg=>{
 assert.equal(svg.dataset.gamma,'[1, 1]');
 assert.equal(svg.dataset.direction,'true');
 assert.equal(svg.dataset.component,'rmsnorm');
 assert.equal(svg.dataset.output,'[0.848528, 1.131371]');
 near(Number(svg.dataset.outputLength),Math.SQRT2,1e-5,'uniform gamma keeps the normalized Euclidean length');
 near(Number(svg.dataset.outputRms),1,1e-5,'uniform gamma keeps RMS at 1');
});
change(rms,'γ=[1,2]',svg=>{
 assert.equal(svg.dataset.gamma,'[1, 2]');
 assert.equal(svg.dataset.direction,'false');
 assert.equal(svg.dataset.component,'rmsnorm');
 assert.equal(svg.dataset.output,'[0.848528, 2.262741]');
 near(Number(svg.dataset.outputLength),2.416609,1e-5,'non-uniform gamma lengthens the output');
 near(Number(svg.dataset.outputRms),1.708800,1e-5,'non-uniform gamma raises output RMS');
});
rms.paint(2.4);
const rmsSeeks=rms.seeks.length;
rms.click('展开计算');
assert.equal(rms.seeks.length,rmsSeeks,'detail toggle must not rewind');
assert.equal(rms.position,2.4,'detail toggle keeps paused progress');

const residual=makeBoard('residual');
residual.paint(0);
assert.equal(residual.svg.dataset.x,'[1, 2]','residual keeps the same x input');
assert.equal(residual.svg.dataset.fx,'[0.5, -1]','residual starts from the signed teaching F');
assert.equal(residual.svg.dataset.resultState,'pending');
assert.equal(residual.svg.dataset.y,'','y is not written before any merge');
assert.equal(residual.svg.dataset.merge0,'0.000');
change(residual,'F=[1,1]',svg=>{
 assert.equal(svg.dataset.sum,'[2, 3]');
 assert.equal(svg.dataset.onlyF,'[1, 1]');
 assert.equal(svg.dataset.component,'residual');
 assert.equal(svg.dataset.condition,'2');
 assert.equal(svg.dataset.resultState,'pending');
});

const swiglu=makeBoard('swiglu');
change(swiglu,'聚焦 p3',svg=>{
 assert.equal(svg.dataset.focus,'p3');
 assert.equal(svg.dataset.applyProgress,'0.000');
});

const heads=makeBoard('heads');
// All three topologies stay on screen; only history is a condition. The board
// proves readiness before reading and keeps four independent Q heads per mode.
heads.paint(0);
assert.equal(heads.svg.dataset.mode,'parallel');
assert.equal(heads.svg.dataset.layout,'parallel');
assert.equal(heads.svg.dataset.qHeads,'4');
assert.equal(heads.svg.dataset.kvHeads,'4,2,1');
assert.equal(heads.svg.dataset.mapping,'0,1,2,3;0,0,1,1;0,0,0,0');
assert.equal(heads.svg.dataset.qIndependent,'true');
assert.equal(heads.svg.dataset.qMerged,'false');
assert.equal(heads.svg.dataset.losslessMerge,'false');
assert.equal(heads.svg.dataset.mergedFromMHA,'false');
assert.equal(heads.svg.dataset.outputQualityEqual,'false');
assert.equal(heads.svg.dataset.scalars3,'48,24,12');
assert.equal(heads.svg.dataset.scalars6,'96,48,24');
assert.equal(heads.svg.dataset.cacheScale,'96');
assert.equal(heads.svg.dataset.historyReady,'false');
assert.equal(heads.svg.dataset.filledPositions,'0');
assert.equal(heads.svg.dataset.readPhase,'wait');
const qGroups=()=>walk(heads.svg).filter(n=>n.tagName==='g'&&(n.attrs.class||'')==='cmp-headq'&&n.style.display!=='none');
const kvGroups=()=>walk(heads.svg).filter(n=>n.tagName==='g'&&(n.attrs.class||'')==='cmp-headkv'&&n.style.display!=='none');
assert.equal(qGroups().length,12,'MHA/GQA/MQA each keep four independent Q heads');
assert.equal(kvGroups().length,7,'4 + 2 + 1 K/V groups are visible side by side');
heads.paint(.99);
assert.equal(heads.svg.dataset.historyReady,'true','history is full before any read starts');
assert.equal(heads.svg.dataset.readPhase,'wait');
assert.equal(heads.svg.dataset.readOut,'0.000');
heads.paint(1.2);
assert.equal(heads.svg.dataset.readPhase,'out');
assert.ok(Number(heads.svg.dataset.readOut)>0,'Q starts reading only after the history is ready');
assert.equal(heads.svg.dataset.readBack,'0.000');
const readPaths=walk(heads.svg).filter(n=>n.tagName==='path'&&(n.attrs.class||'').includes('cmp-read'));
assert.equal(readPaths.length,12,'every Q has its own read path');
assert.ok(readPaths.every(n=>n.style.display!=='none'));
heads.paint(2.2);
assert.equal(heads.svg.dataset.readPhase,'return');
assert.ok(Number(heads.svg.dataset.readBack)>0,'the read result returns to Q before completion');
heads.paint(2.6);
assert.equal(heads.svg.dataset.readPhase,'done');
assert.equal(heads.svg.dataset.historyReady,'true');
assert.equal(heads.svg.dataset.filledPositions,'3');
change(heads,'历史 6',svg=>{
 assert.equal(svg.dataset.history,'6');
 assert.equal(svg.dataset.scalars,'96,48,24');
 assert.equal(svg.dataset.readProgress,'0.000');
 assert.equal(svg.dataset.filledPositions,'0');
 assert.equal(svg.dataset.historyReady,'false');
 assert.equal(svg.dataset.readPhase,'wait');
});
const rmsShrink=walk(rms.svg).find(n=>n.tagName==='text'&&n.textContent==='两条箭头正沿原方向缩短…');
rms.paint(1.3);
assert.ok(rmsShrink,'the RMSNorm shrinking hint exists');
assert.ok(Number(rmsShrink.attrs.x)>600,'the hint sits in the fixed right note slot, away from the arrow tips');
assert.ok(Number(rmsShrink.attrs.y)>=330,'the hint sits below the arrow area');


// Real renderer geometry: non-integer progress must move the drawn vectors,
// normalization must finish before gamma starts, and gamma must change only
// the selected dimension. These read the SVG line endpoints, not just text.
// The uniform-gamma geometry selects γ=[1,1] explicitly: the board now starts
// from the non-uniform γ=[1,2], so the old implicit default no longer holds.
function lineByClass(svg,cls){
 const node=walk(svg).find(n=>n.tagName==='line'&&(n.attrs.class||'').split(/\s+/).includes(cls));
 assert.ok(node,`missing line: ${cls}`);return node;
}
function endOf(line){return{x:Number(line.attrs.x2),y:Number(line.attrs.y2)};}
function lengthOf(line){return Math.hypot(Number(line.attrs.x2)-Number(line.attrs.x1),Number(line.attrs.y2)-Number(line.attrs.y1));}
// Residual renderer: one signed ruler per dimension. The x lane is the base,
// the F lane is the sub-layer output alone, and the x+F lane grows its change
// only after that dimension's contribution has arrived. Geometry is read from
// the SVG attrs, not from status text.
function nodesByClass(svg,cls){return walk(svg).filter(n=>(n.attrs.class||'').split(/\s+/).includes(cls));}
function barRects(svg,cls){return nodesByClass(svg,cls).filter(n=>n.tagName==='rect'&&Number(n.attrs.y)>=200);}
function visibleText(svg,cls){return nodesByClass(svg,cls).filter(n=>n.style.display!=='none');}

const resFlow=makeBoard('residual');
resFlow.paint(1);
assert.equal(resFlow.svg.dataset.phase,'sub');
assert.equal(resFlow.svg.dataset.merge0,'0.000','F alone does not merge before arrival');
// F is fully drawn at the end of the sub stage; p=1.05 is before any merge.
resFlow.paint(1.05);
const baseBars=barRects(resFlow.svg,'cmp-xbar'),baseOutlines=barRects(resFlow.svg,'cmp-res-base');
assert.equal(baseBars.length,2);assert.equal(baseOutlines.length,2);
near(Number(baseBars[0].attrs.width),106.25,0.2,'dim0 x bar uses the shared ruler');
near(Number(baseBars[1].attrs.width),212.5,0.2,'dim1 x bar is twice as long');
near(Number(baseOutlines[0].attrs.width),Number(baseBars[0].attrs.width),0.2,'y base keeps x length');
near(Number(baseOutlines[1].attrs.width),Number(baseBars[1].attrs.width),0.2);
const fBars=barRects(resFlow.svg,'cmp-res-fonly');
assert.equal(fBars.length,2,'F-only bars are separate from the result');
assert.ok(fBars[1].attrs.class.includes('is-negative'),'negative F is visibly negative');
near(Number(fBars[1].attrs.width),106.25,0.2);
assert.ok(Number(fBars[1].attrs.x)<Number(baseBars[0].attrs.x),'negative F extends left of zero');

resFlow.paint(1.49);
near(Number(resFlow.svg.dataset.arrival0),1,0.002,'dim0 contribution has arrived');
assert.equal(resFlow.svg.dataset.merge0,'0.000','merge waits for arrival');
assert.equal(visibleText(resFlow.svg,'is-final').length,0,'no exact y before merge');
resFlow.paint(1.6);
assert.ok(Number(resFlow.svg.dataset.merge0)>0&&Number(resFlow.svg.dataset.merge0)<1);
assert.equal(visibleText(resFlow.svg,'is-final').length,0,'partial merge is not printed as a result');
const progress=visibleText(resFlow.svg,'cmp-res-progress');
assert.equal(progress.length,1);
assert.match(progress[0].textContent,/^\d+%$/,'partial merge shows progress, not an exact y');
resFlow.paint(1.9);
assert.equal(resFlow.svg.dataset.done0,'true');
assert.equal(resFlow.svg.dataset.done1,'false');
assert.equal(resFlow.svg.dataset.resultState,'merging');
assert.equal(resFlow.svg.dataset.y,'','y stays unwritten until both dimensions complete');
const firstFinal=visibleText(resFlow.svg,'is-final');
assert.equal(firstFinal.length,1);assert.equal(firstFinal[0].textContent,'1.5');

resFlow.paint(2.5);
assert.equal(resFlow.svg.dataset.complete,'true');
assert.equal(resFlow.svg.dataset.resultState,'complete');
assert.equal(resFlow.svg.dataset.y,'[1.5, 1]');
assert.deepEqual(visibleText(resFlow.svg,'is-final').map(n=>n.textContent),['1.5','1']);
const deltas=barRects(resFlow.svg,'cmp-res-delta');
assert.equal(deltas.length,2);
assert.equal(deltas[0].attrs.class.includes('is-negative'),false);
assert.ok(deltas[1].attrs.class.includes('is-negative'),'dim1 change is drawn as negative');
const dim1Base=Number(baseOutlines[1].attrs.x)+Number(baseOutlines[1].attrs.width);
near(Number(deltas[1].attrs.x)+Number(deltas[1].attrs.width),dim1Base,0.2,'negative change returns from x to y');
const dots=nodesByClass(resFlow.svg,'cmp-res-result').filter(n=>n.tagName==='circle'&&n.attrs.r==='5');
near(Number(dots[1].attrs.cx),425,0.2,'negative result marker lands left of the x=2 base end');

resFlow.paint(3.2);
assert.ok(Number(resFlow.svg.dataset.compare)>0);
const bands=barRects(resFlow.svg,'cmp-res-diff');
assert.equal(bands.length,2);
near(Number(bands[0].attrs.width),106.25,0.2,'difference is exactly x for dim0');
near(Number(bands[1].attrs.width),212.5,0.2,'difference is exactly x for dim1');
assert.ok(bands[1].attrs.class.includes('is-negative'));
const diffTexts=visibleText(resFlow.svg,'cmp-res-diff-label').map(n=>n.textContent);
assert.ok(diffTexts.includes('差 = x0 = 1')&&diffTexts.includes('差 = x1 = 2'),'difference labels name the bypass');

const resZero=makeBoard('residual');
change(resZero,'F=[0,0]',svg=>{
 assert.equal(svg.dataset.fx,'[0, 0]');
 assert.equal(svg.dataset.zero,'true');
 assert.equal(svg.dataset.merge0,'0.000');
});
resZero.paint(2.5);
assert.equal(resZero.svg.dataset.complete,'true');
assert.equal(resZero.svg.dataset.sum,'[1, 2]');
assert.deepEqual(visibleText(resZero.svg,'is-final').map(n=>n.textContent),['1','2'],'F=0 still gives y=x');
assert.equal(visibleText(resZero.svg,'cmp-res-zero-dot').length,2,'zero F is marked, not given a fake bar');
const zeroDeltas=barRects(resZero.svg,'cmp-res-delta');
assert.ok(zeroDeltas.every(n=>Number(n.attrs.width)<.5),'F=0 has no change segment');
resZero.paint(3.2);
const zeroBands=barRects(resZero.svg,'cmp-res-diff');
near(Number(zeroBands[0].attrs.width),Number(baseBars[0].attrs.width),0.2,'F=0 difference is still x0');
near(Number(zeroBands[1].attrs.width),Number(baseBars[1].attrs.width),0.2,'F=0 difference is still x1');

const resPos=makeBoard('residual');
change(resPos,'F=[1,1]',svg=>{assert.equal(svg.dataset.sum,'[2, 3]');});
resPos.paint(2.5);
assert.deepEqual(visibleText(resPos.svg,'is-final').map(n=>n.textContent),['2','3']);
const posDeltas=barRects(resPos.svg,'cmp-res-delta');
assert.ok(posDeltas.every(n=>!n.attrs.class.includes('is-negative')),'positive F has positive change segments');
near(Number(posDeltas[0].attrs.width),106.25,0.2);
resPos.paint(3.2);
const posBands=barRects(resPos.svg,'cmp-res-diff');
near(Number(posBands[0].attrs.width),106.25,0.2,'positive condition still differs by x0');
near(Number(posBands[1].attrs.width),212.5,0.2,'positive condition still differs by x1');

const motion=makeBoard('rmsnorm');
change(motion,'γ=[1,1]',svg=>{
 assert.equal(svg.dataset.gamma,'[1, 1]');
 assert.equal(svg.dataset.direction,'true');
 assert.equal(svg.dataset.output,'[0.848528, 1.131371]');
});
motion.paint(0);
const inLine=lineByClass(motion.svg,'cmp-input-x'),in2Line=lineByClass(motion.svg,'cmp-input-2x');
near(lengthOf(in2Line),2*lengthOf(inLine),1e-6);
assert.equal(motion.svg.dataset.normProgress,'0.000');
const outX=lineByClass(motion.svg,'cmp-output-norm'),out2=lineByClass(motion.svg,'cmp-output-2x');
const startXLen=lengthOf(outX),start2Len=lengthOf(out2),startNorm=endOf(outX);
near(startXLen,lengthOf(inLine),1e-6,'normalization starts at the input length on the same ruler');
near(start2Len,lengthOf(in2Line),1e-6,'2x also starts at its input length');
assert.ok(start2Len>startXLen&&endOf(out2).y<startNorm.y,'2x starts farther along the same ray');
motion.paint(1.5);
const midXLen=lengthOf(outX),mid2Len=lengthOf(out2),midNorm=endOf(outX);
assert.ok(midXLen<startXLen-1&&mid2Len<start2Len-1,'both arrows physically shrink toward the origin');
assert.ok(midXLen>0&&mid2Len>0,'shrinking arrows remain visible');
assert.ok(Number(motion.svg.dataset.normProgress)>0&&Number(motion.svg.dataset.normProgress)<1);
motion.paint(2);
const doneXLen=lengthOf(outX),done2Len=lengthOf(out2),doneNorm=endOf(outX),done2=endOf(out2);
assert.ok(doneXLen<midXLen-1&&done2Len<mid2Len-1,'normalization finishes by shrinking further');
assert.ok(doneNorm.y>midNorm.y,'normalization ends closer to the origin, not grown from it');
near(doneXLen,done2Len,1e-3,'both inputs reach the same length');
assert.ok(Math.hypot(doneNorm.x-done2.x,doneNorm.y-done2.y)<1e-3,'both tips land on the same point');
assert.equal(motion.svg.dataset.normProgress,'1.000');
assert.equal(motion.svg.dataset.gammaProgress,'0.000','gamma has not started at the normalization endpoint');
const normY=doneNorm.y;
change(motion,'γ=[1,2]',svg=>{
 assert.equal(svg.dataset.gamma,'[1, 2]');
 assert.equal(svg.dataset.normalized,'[0.848528, 1.131371]','gamma does not change the normalized direction');
});
motion.paint(2);
assert.equal(motion.svg.dataset.gammaProgress,'0.000');
near(endOf(lineByClass(motion.svg,'cmp-output-norm')).y,normY,1e-6);
motion.paint(3.5);
const midGamma=endOf(lineByClass(motion.svg,'cmp-output-norm'));
assert.ok(midGamma.y<normY,'gamma moves only the dim1 component upward');
assert.ok(Number(motion.svg.dataset.gammaProgress)>0&&Number(motion.svg.dataset.gammaProgress)<1);
motion.paint(4);
const doneGamma=endOf(lineByClass(motion.svg,'cmp-output-norm'));
assert.ok(doneGamma.y<midGamma.y,'gamma completes at the stretched endpoint');
assert.equal(motion.svg.dataset.gammaProgress,'1.000');
assert.equal(motion.svg.dataset.direction,'false');
assert.ok(Math.abs(Number(motion.svg.dataset.angleDegrees)-16.31)<.02);
assert.equal(motion.svg.dataset.phase,'compare');
assert.equal(motion.svg.dataset.output,'[0.848528, 2.262741]');
near(Number(motion.svg.dataset.outputLength),2.416609,1e-5,'non-uniform gamma stretches the drawn output length');
near(Number(motion.svg.dataset.outputRms),1.708800,1e-5,'non-uniform gamma raises the output RMS');
const uniform=makeBoard('rmsnorm');
change(uniform,'γ=[1,1]',svg=>{
 assert.equal(svg.dataset.gamma,'[1, 1]');
 assert.equal(svg.dataset.direction,'true');
});
uniform.paint(4.5);
const compareLine=lineByClass(uniform.svg,'cmp-compare');
assert.ok(lengthOf(compareLine)>0,'uniform gamma shows the dashed comparison');
assert.ok(endOf(compareLine).y<endOf(lineByClass(uniform.svg,'cmp-output-norm')).y);
assert.ok(Number(uniform.svg.dataset.compareProgress)>0&&Number(uniform.svg.dataset.compareProgress)<1);
assert.equal(uniform.svg.dataset.direction,'true');
assert.equal(uniform.svg.dataset.angleDegrees,'0.00','uniform gamma keeps the actual output angle at 0');
assert.ok(Math.abs(Number(uniform.svg.dataset.compareAngleDegrees)-16.31)<.02,'dashed comparison is the 16.31° case');
// RoPE renderer: the default picture is a fixed-axis unit circle with real
// Q/K rotation, a position track above it, and three same-source rows. Every
// endpoint and dot is derived from the same qVec/kVecs at the current t.
function vectorLines(svg){return walk(svg).filter(n=>n.tagName==='line'&&(n.attrs.class||'').split(/\s+/).includes('cmp-vector-line')&&n.attrs['data-key']);}
function vectorAngles(svg){const out={};for(const n of vectorLines(svg))out[n.attrs['data-key']]=Number(n.attrs['data-angle']);return out;}
function compareFills(svg){return walk(svg).filter(n=>n.tagName==='rect'&&(n.attrs.class||'').split(/\s+/).includes('cmp-compare-fill'));}
function compareWidths(svg){return compareFills(svg).map(n=>Number(n.attrs.width));}
function compareCenter(svg){const tracks=walk(svg).filter(n=>n.tagName==='rect'&&(n.attrs.class||'').split(/\s+/).includes('cmp-compare-track'));return tracks.map(t=>Number(t.attrs.x)+Number(t.attrs.width)/2);}
function projectionLine(svg){return walk(svg).find(n=>n.tagName==='line'&&(n.attrs.class||'').split(/\s+/).includes('cmp-projection'));}
const ropeBase=makeBoard('rope');
ropeBase.paint(0);
const startRows=JSON.parse(ropeBase.svg.dataset.rows);
assert.deepEqual(startRows.map(r=>r.currentAngle),[0,0,0]);
assert.deepEqual(startRows.map(r=>r.currentDot),[1,1,1]);
assert.deepEqual(startRows.map(r=>r.angle),[90,45,0]);
assert.deepEqual(vectorAngles(ropeBase.svg),{Q:0,K1:0,K2:0,K3:0});
assert.deepEqual(compareWidths(ropeBase.svg),compareCenter(ropeBase.svg).map(c=>c-(360+9)),'rest state bars already show the current dot +1');
assert.deepEqual(compareFills(ropeBase.svg).map(n=>Number(n.attrs['data-value'])),[1,1,1]);
assert.equal(ropeBase.svg.dataset.detail,'false');
assert.equal(ropeBase.svg.dataset.commonRotation,'0');
ropeBase.paint(3);
const endRows=JSON.parse(ropeBase.svg.dataset.rows);
near(endRows[0].dot,0,1e-9);near(endRows[1].dot,Math.SQRT1_2,1e-9);near(endRows[2].dot,1,1e-9);
assert.deepEqual(endRows.map(r=>r.angle),[90,45,0]);
assert.deepEqual(vectorAngles(ropeBase.svg),{Q:90,K1:0,K2:45,K3:90});
const radius=Number(ropeBase.svg.dataset.radius);
for(const line of vectorLines(ropeBase.svg))near(lengthOf(line),radius,0.01,'rotation keeps unit length '+line.attrs['data-key']);
const tips={};for(const line of vectorLines(ropeBase.svg))tips[line.attrs['data-key']]=endOf(line);
near(tips.Q.x,tips.K3.x,1e-9,'Q and K3 share the same position in base');near(tips.Q.y,tips.K3.y,1e-9);
const endWidths=compareWidths(ropeBase.svg),endCenters=compareCenter(ropeBase.svg);
assert.equal(endWidths[0],0);
near(endWidths[1],(endCenters[1]-360-9)*Math.SQRT1_2,0.02);
near(endWidths[2],(endCenters[2]-360-9),0.02);
const endProjection=Number(projectionLine(ropeBase.svg).attrs['data-value']);
near(endProjection,endRows[1].currentDot,1e-9,'K2 projection uses the row dot');
near(Number(compareFills(ropeBase.svg)[1].attrs['data-value']),endProjection,1e-9,'bar and projection share one number');
const joint=walk(ropeBase.svg).find(n=>n.tagName==='text'&&n.textContent==='Q 与 K3 同向');
assert.ok(joint,'coincident Q/K3 names both identities');
assert.ok(!walk(ropeBase.svg).some(n=>n.tagName==='text'&&n.textContent.includes('Q·K3')),'joint badge cannot be mistaken for a dot product');

// Small-angle readability: identity chips stay separated while Q/K3 coincide
// and K2 is only 9° away; angles live in the fixed key, not on the endpoint.
const crowded=makeBoard('rope');
crowded.paint(1.3);
const chips=walk(crowded.svg).filter(n=>n.tagName==='text'&&n.attrs['data-identity']&&n.attrs['data-visible']==='true');
assert.equal(chips.length,4,'all four identities stay queryable in the crowded frame');
for(let i=0;i<chips.length;i++)for(let j=i+1;j<chips.length;j++){
 const a=chips[i].attrs,b=chips[j].attrs;
 assert.ok(Math.hypot(Number(a['data-label-x'])-Number(b['data-label-x']),Number(a['data-label-y'])-Number(b['data-label-y']))>=13,
  `identity chips ${a['data-identity']}/${b['data-identity']} keep readable spacing`);
}
const baseQ=Number(crowded.svg.dataset.vectorAngles.split(',')[0]);
near(Number(crowded.svg.dataset.arcRotation),baseQ,0.05,'base arc span is the endpoint angle, not progress-squared');
const baseArcLabel=walk(crowded.svg).find(n=>n.attrs.class==='cmp-arc-label');
assert.equal(baseArcLabel.textContent,`Q 弧 +${Math.round(baseQ)}°`,'base arc label matches the Q endpoint');
const q1crowd=makeBoard('rope');
q1crowd.click('只移 Q +1');
q1crowd.paint(1.3);
for(let i=0;i<4;i++)for(let j=i+1;j<4;j++){
 const nodes=walk(q1crowd.svg).filter(n=>n.tagName==='text'&&n.attrs['data-identity']).map(n=>n.attrs);
 const a=nodes[i],b=nodes[j];
 assert.ok(Math.hypot(Number(a['data-label-x'])-Number(b['data-label-x']),Number(a['data-label-y'])-Number(b['data-label-y']))>=13,
  `q1 chips ${a['data-identity']}/${b['data-identity']} keep readable spacing`);
}

// q+1: only Q moves and turns; K vectors stay fixed. K1 becomes negative and
// must grow left of the zero line.
const q1=makeBoard('rope');
q1.paint(0);
q1.click('只移 Q +1');
assert.equal(q1.svg.dataset.condition,'q1');
assert.equal(q1.svg.dataset.qp,'3');
assert.equal(q1.svg.dataset.currentQp,'2.000');
assert.equal(q1.svg.dataset.pks,'0,1,2');
assert.equal(q1.svg.dataset.progress,'0.000');
assert.deepEqual(vectorAngles(q1.svg),{Q:90,K1:0,K2:45,K3:90});
const q1StartRows=JSON.parse(q1.svg.dataset.rows);
near(q1StartRows[0].currentDot,0,1e-9);
near(q1StartRows[1].currentDot,Math.SQRT1_2,1e-9);
near(q1StartRows[2].currentDot,1,1e-9);
q1.paint(3);
const q1Rows=JSON.parse(q1.svg.dataset.rows);
near(q1Rows[0].dot,-Math.SQRT1_2,1e-9,'q+1 K1 is negative');
near(q1Rows[1].dot,0,1e-9,'q+1 K2 is exactly zero');
near(q1Rows[2].dot,Math.SQRT1_2,1e-9,'q+1 K3 is positive');
assert.deepEqual(vectorAngles(q1.svg),{Q:135,K1:0,K2:45,K3:90});
const q1Fills=compareFills(q1.svg),q1Centers=compareCenter(q1.svg);
const q1TrackX=Number(walk(q1.svg).find(n=>n.tagName==='rect'&&(n.attrs.class||'').split(/\s+/).includes('cmp-compare-track')).attrs.x);
assert.equal(q1Fills[0].attrs['data-sign'],'negative');
assert.ok(Number(q1Fills[0].attrs.x)<q1Centers[0]-1,'negative fill extends left');
near(Number(q1Fills[0].attrs.width),(q1Centers[0]-q1TrackX)*Math.SQRT1_2,0.02);
assert.equal(q1Fills[1].attrs['data-sign'],'zero');
assert.equal(Number(q1Fills[1].attrs.width),0);
assert.equal(q1Fills[2].attrs['data-sign'],'positive');
assert.ok(Number(q1Fills[2].attrs.x)>=q1Centers[2]-0.01,'positive fill grows right');
assert.ok(walk(q1.svg).some(n=>(n.attrs.class||'')==='cmp-row-value'&&n.textContent.includes('负比较')),'negative result is named');
near(Number(projectionLine(q1.svg).attrs['data-value']),0,1e-9,'K2 projection matches the zero dot');

// Whole +3 is a common rotation from the base angles, not a jump. Track and
// vectors move together; the three rows stay constant throughout.
const shifted=makeBoard('rope');
shifted.paint(0);
shifted.click('整段 +3');
assert.equal(shifted.svg.dataset.condition,'shift3');
assert.equal(shifted.svg.dataset.qp,'5');
assert.equal(shifted.svg.dataset.currentQp,'2.000');
assert.equal(shifted.svg.dataset.pks,'3,4,5');
assert.equal(shifted.svg.dataset.currentPks,'0.000,1.000,2.000');
assert.equal(shifted.svg.dataset.commonRotation,'135');
shifted.paint(1.75);
const midShiftRows=JSON.parse(shifted.svg.dataset.rows);
for(const row of midShiftRows){
 near(row.qAngle-row.fromQAngle,row.kAngle-row.fromKAngle,0.02,'common rotation moves Q and every K by the same angle');
 near(row.currentDot,row.dot,1e-9,'rows stay constant during the common rotation');
}
shifted.paint(3);
const shiftedRows=JSON.parse(shifted.svg.dataset.rows);
assert.deepEqual(shiftedRows.map(r=>r.pos),[3,4,5]);
assert.deepEqual(shiftedRows.map(r=>r.currentPos),[3,4,5]);
shiftedRows.forEach((r,i)=>near(r.dot,endRows[i].dot,1e-12,'common +3 keeps dot '+i));
assert.deepEqual(vectorAngles(shifted.svg),{Q:225,K1:135,K2:180,K3:225});
const arc=walk(shifted.svg).find(n=>n.attrs.class==='cmp-arc-label');
assert.ok(arc&&arc.textContent.includes('共同 +135°'),'the common rotation is named at full angle');
const arcSpan=Number(shifted.svg.dataset.arcRotation);
near(arcSpan,135,0.05,'shift3 arc span directly matches the 135° endpoint travel');
const shiftRows=walk(shifted.svg).filter(n=>n.attrs.class==='cmp-row-label').map(n=>n.textContent);
assert.deepEqual(shiftRows,['K1 · p3','K2 · p4','K3 · p5'],'completed shift rows name the reached positions');
const midShift=makeBoard('rope');
midShift.click('整段 +3');
midShift.paint(1.75);
const midLabels=walk(midShift.svg).filter(n=>n.attrs.class==='cmp-row-label').map(n=>n.textContent);
assert.deepEqual(midLabels,['K1 · p0→p3','K2 · p1→p4','K3 · p2→p5'],'moving rows separate current from target');
const midArc=walk(midShift.svg).find(n=>n.attrs.class==='cmp-arc-label');
const midSpan=Number(midShift.svg.dataset.arcRotation);
near(midSpan,135*Number(midShift.svg.dataset.progress),0.1,'shift3 arc label is 135*rot, not 135*rot*rot');
assert.equal(midArc.textContent,`共同 +${Math.round(midSpan)}°`,'shift3 arc text equals its displayed span');

// The calculation layer reads the same current angle and dot as the main rows.
const detailBoard=makeBoard('rope');
detailBoard.paint(0);
detailBoard.click('只移 Q +1');
detailBoard.paint(1.75);
const midRows=JSON.parse(detailBoard.svg.dataset.rows);
const midQ=Math.round(midRows[0].qAngle),midK=Math.round(midRows[0].kAngle);
detailBoard.click('展开计算');
const detailLines=walk(detailBoard.svg).filter(n=>{const cls=n.attrs.class||'';return cls.includes('cmp-detail-line')||cls.includes('cmp-detail-sub');}).map(n=>n.textContent).join(' | ');
assert.ok(detailLines.includes('Q ')&&detailLines.includes('K ')&&detailLines.includes('q′·k′'),'calculation explains Q/K rotation and the dot');
assert.ok(detailLines.includes(`Q ${midQ}°→135°`),'calculation shows the current Q angle, not only the target');
assert.ok(detailLines.includes(`K ${midK}°→${midRows[0].targetKAngle}°`),'calculation shows the current K angle');
assert.ok(detailLines.includes('有向夹角')&&detailLines.includes('arg k−arg q')&&detailLines.includes('Δθ−φ'),'general formula defines φ as K relative to Q');
assert.ok(detailLines.includes('高维'),'general formula keeps the other frequency pairs');
assert.equal(detailBoard.position,1.75,'detail toggle keeps the paused position');
const detailSeeks=detailBoard.seeks.length;
detailBoard.click('收起计算');
assert.equal(detailBoard.seeks.length,detailSeeks,'detail toggle must not seek');
assert.equal(detailBoard.position,1.75);

// A phone board keeps the same data and a 360x720 drawing; opening the
// derivation must not reset the condition or the clock.
const ropeMobile=makeBoard('rope',360);
ropeMobile.paint(3);
assert.equal(ropeMobile.svg.attrs.viewBox,'0 0 360 720');
const mobileRows=JSON.parse(ropeMobile.svg.dataset.rows);
near(mobileRows[1].dot,Math.SQRT1_2,1e-9);
const mobileSeeks=ropeMobile.seeks.length;
ropeMobile.click('展开计算');
assert.equal(ropeMobile.seeks.length,mobileSeeks);
assert.equal(ropeMobile.position,3);
assert.ok(walk(ropeMobile.svg).filter(n=>(n.attrs.class||'').includes('cmp-detail-line')).map(n=>n.textContent).join(' ').includes('q′·k′'));

// SwiGLU: one shared-weight graph, two branches, a continuous coefficient and
// an elementwise product. Read the drawn u/z bar geometry, not only the text.
const gluFlow=makeBoard('swiglu');
gluFlow.paint(0);
assert.equal(gluFlow.svg.dataset.component,'swiglu');
assert.equal(gluFlow.svg.dataset.focus,'p1');
assert.equal(gluFlow.svg.dataset.shared,'Wg,Wu,Wd');
assert.equal(gluFlow.svg.dataset.branches,'g,u');
assert.equal(gluFlow.svg.dataset.positionCount,'3');
assert.equal(gluFlow.svg.dataset.crossPosition,'false');
assert.equal(gluFlow.svg.dataset.resultState,'multiplying');
assert.equal(gluFlow.svg.dataset.y,'','y is not written before Wd');
assert.ok(Number(gluFlow.svg.dataset.a0)>1,'default p1 has a coefficient above 1');
assert.ok(Number(gluFlow.svg.dataset.a1)<0,'default p1 has a negative coefficient');
assert.equal(gluFlow.svg.dataset.negative,'true');
assert.equal(gluFlow.svg.dataset.exceedsOne,'true');
assert.equal(gluFlow.svg.dataset.applyProgress,'0.000');
assert.equal(gluFlow.svg.dataset.downProgress,'0.000');
gluFlow.paint(1.1);
assert.equal(gluFlow.svg.dataset.gateProgress,'0.000','SiLU has not started during projection');
const earlyZ=nodesByClass(gluFlow.svg,'cmp-z-fill');
assert.ok(earlyZ.every(n=>n.style.display==='none'),'z bars wait for the multiply stage');
gluFlow.paint(3.2);
assert.ok(Number(gluFlow.svg.dataset.multiplyProgress)>0&&Number(gluFlow.svg.dataset.multiplyProgress)<1);
assert.equal(gluFlow.svg.dataset.resultState,'multiplying');
assert.equal(gluFlow.svg.dataset.y,'');
gluFlow.paint(5);
assert.equal(gluFlow.svg.dataset.resultState,'complete');
assert.equal(gluFlow.svg.dataset.downProgress,'1.000');
assert.equal(gluFlow.svg.dataset.y,'[5.715445, -0.806824]');
assert.equal(gluFlow.svg.dataset.z,'[5.715445, -0.806824]');
const zBarsFinal=nodesByClass(gluFlow.svg,'cmp-z-fill').filter(n=>n.tagName==='rect');
const uBarsFinal=nodesByClass(gluFlow.svg,'cmp-u-fill').filter(n=>n.tagName==='rect');
assert.equal(zBarsFinal.length,2);assert.equal(uBarsFinal.length,2);
assert.ok(Number(zBarsFinal[0].attrs.width)>Number(uBarsFinal[0].attrs.width),'a0>1 makes z0 longer than u0');
assert.ok(zBarsFinal[1].attrs.class.includes('is-negative'),'negative z1 is visibly negative');
const zeroB=Number(gluFlow.svg.dataset.zeroX);
assert.ok(zeroB>0,'the shared u/z ruler exposes its zero');
near(Number(zBarsFinal[1].attrs.x)+Number(zBarsFinal[1].attrs.width),zeroB,.2,'negative z1 grows left of zero');
function focusIdentity(board){
 const nodes=walk(board.svg);
 return {
  title:nodes.find(n=>n.attrs['data-role']==='graph-title'),
  label:nodes.find(n=>n.attrs['data-role']==='focus-label'),
  input:nodes.find(n=>n.attrs['data-role']==='focus-input')
 };
}

// Detail is a view toggle; changing focus rewinds the clock but keeps it open.
gluFlow.paint(3.2);
const gluSeeks=gluFlow.seeks.length;
gluFlow.click('显示手算核对');
assert.equal(gluFlow.seeks.length,gluSeeks,'detail toggle must not rewind');
assert.equal(gluFlow.position,3.2);
assert.equal(gluFlow.svg.dataset.detail,'shown');
assert.ok(walk(gluFlow.svg).some(n=>n.textContent.includes('Wg = [[1, 1], [1, -1]]')));
change(gluFlow,'聚焦 p3',svg=>{
 assert.equal(svg.dataset.focus,'p3');
 assert.equal(svg.dataset.detail,'shown','detail stays open after a condition change');
 assert.equal(svg.dataset.applyProgress,'0.000');
 assert.equal(svg.dataset.downProgress,'0.000');
 assert.equal(svg.dataset.resultState,'multiplying');
});
gluFlow.paint(5);
assert.equal(gluFlow.svg.dataset.y,'[3.523188, -0.476812]');
const closeSeeks=gluFlow.seeks.length;
gluFlow.click('收起手算核对');
assert.equal(gluFlow.seeks.length,closeSeeks,'closing detail must not rewind');
assert.equal(gluFlow.svg.dataset.detail,'hidden');
const p3Identity=focusIdentity(gluFlow);
assert.equal(p3Identity.title.textContent,'焦点 p3 的调制链','graph title follows the selected position');
assert.equal(p3Identity.label.textContent,'焦点 p3','input chip label follows the selected position');
assert.equal(p3Identity.input.textContent,'x = [0, 2]','input chip vector follows the selected position');
assert.ok(!walk(gluFlow.svg).some(n=>n.tagName==='text'&&n.style.display!=='none'&&n.textContent==='焦点 p1'),'p3 selection leaves no stale p1 focus label');

const gluP2=makeBoard('swiglu');
change(gluP2,'聚焦 p2',svg=>{
 assert.equal(svg.dataset.focus,'p2');
 assert.equal(svg.dataset.negative,'false');
 assert.equal(svg.dataset.exceedsOne,'true');
 assert.equal(svg.dataset.a1,'0.731059');
});
gluP2.paint(5);
const p2Identity=focusIdentity(gluP2);
assert.equal(p2Identity.title.textContent,'焦点 p2 的调制链','p2 graph title is not stale p1');
assert.equal(p2Identity.label.textContent,'焦点 p2');
assert.equal(p2Identity.input.textContent,'x = [2, 1]','p2 input chip shows the selected vector');
assert.ok(!walk(gluP2.svg).some(n=>n.tagName==='text'&&n.style.display!=='none'&&n.textContent==='焦点 p1'),'p2 selection leaves no stale p1 focus label');
const p2z=nodesByClass(gluP2.svg,'cmp-z-fill').filter(n=>n.tagName==='rect');
const p2u=nodesByClass(gluP2.svg,'cmp-u-fill').filter(n=>n.tagName==='rect');
assert.equal(p2z.length,2);assert.equal(p2u.length,2);
assert.ok(Number(p2z[1].attrs.width)<Number(p2u[1].attrs.width),'0<a1<1 makes z1 shorter than u1');
assert.ok(!p2z[1].attrs.class.includes('is-negative'),'p2 z1 stays on the same side as u1');
const gluMobile=makeBoard('swiglu',360);
gluMobile.paint(5);
assert.equal(gluMobile.svg.attrs.viewBox,'0 0 360 700');
assert.equal(gluMobile.svg.dataset.resultState,'complete');
assert.equal(gluMobile.svg.dataset.y,'[5.715445, -0.806824]');

// Transformer Block renderer: two explicit bypasses, two gated merges, and a
// per-position shared MLP. Read the live draw state at each boundary, then read
// the merge bars/endpoints rather than trusting only the dataset strings.
function blockMergeCard(svg,label){
 const card=walk(svg).find(n=>n.tagName==='g'&&n.dataset.merge===label);
 assert.ok(card,`missing merge card: ${label}`);return card;
}
function blockMergeRow(card,index){
 const base=nodesByClass(card,'cmp-block-base')[index];
 const delta=nodesByClass(card,'cmp-block-delta')[index];
 const dot=nodesByClass(card,'cmp-block-dot')[index];
 assert.ok(base&&delta&&dot,`missing merge row ${index}`);
 return {base,delta,dot};
}
function vectorFromDataset(text){return JSON.parse(text.replace(/−/g,'-'));}
function nearVector(text,expected,eps,label){
 const actual=vectorFromDataset(text);
 assert.equal(actual.length,expected.length,`${label} length`);
 actual.forEach((v,i)=>near(v,expected[i],eps,`${label}[${i}]`));
}

const block=makeBoard('block');
const blockModel=context.ComponentModel.blockPass();
const blockOff=context.ComponentModel.blockPass({residual:false});
const blockP3=blockModel.positions[2],blockOffP3=blockOff.positions[2];
block.paint(0);
assert.equal(block.svg.dataset.focus,'p3');
assert.equal(block.svg.dataset.firstBypassFrom,'x');
assert.equal(block.svg.dataset.secondBypassFrom,'x1');
assert.equal(block.svg.dataset.residual,'on');
assert.equal(block.svg.dataset.mlpInput,'x1-norm');
assert.equal(block.svg.dataset.mlpShared,'Wg,Wu,Wd');

change(block,'聚焦 p1',svg=>{
 assert.equal(svg.dataset.focus,'p1');
 assert.equal(svg.dataset.readRange,'p1');
 assert.equal(svg.dataset.readOut,'0.000');
 assert.equal(svg.dataset.y,'');
});
assert.equal(block.position,0);
block.paint(5);
assert.equal(block.svg.dataset.focus,'p1');
assert.equal(block.svg.dataset.x,'[0.000000, 2.000000]');
assert.ok(walk(block.svg).some(n=>n.tagName==='text'&&n.textContent==='p1 · x'),'input label follows focus');
assert.equal(walk(block.svg).filter(n=>n.tagName==='text'&&n.textContent==='焦点'&&n.style.display!=='none').length,1,
 'exactly one focused position tag');
assert.equal(blockMergeCard(block.svg,'x1 p1').dataset.result,'[0, 3.41]');
assert.equal(blockMergeCard(block.svg,'y p1').dataset.result,'[1.61, 3.02]');
assert.ok(walk(block.svg).some(n=>n.tagName==='text'&&n.textContent==='= [1.61, 3.02]'),'p1 result is visible');
change(block,'聚焦 p3',svg=>{
 assert.equal(svg.dataset.focus,'p3');
 assert.equal(svg.dataset.readRange,'p1-p3');
});
block.paint(5);
assert.equal(block.svg.dataset.x,'[2.000000, 0.000000]');
assert.equal(blockMergeCard(block.svg,'x1 p3').dataset.result,'[2.80, 0.80]');
assert.equal(blockMergeCard(block.svg,'y p3').dataset.result,'[3.39, 2.03]');

// Timing gates: no read before Norm(x) finishes, no x1 before the first merge,
// no ΔM before the MLP finishes, and no y before the second merge finishes.
const blockFlow=makeBoard('block');
blockFlow.paint(1.1);
assert.equal(blockFlow.svg.dataset.normDone,'false');
assert.equal(blockFlow.svg.dataset.readOut,'0.000');
assert.equal(nodesByClass(blockFlow.svg,'cmp-beam').filter(n=>n.style.display!=='none').length,0,
 'no Attention read beam before Norm(x) completes');
assert.ok(walk(blockFlow.svg).some(n=>n.tagName==='text'&&n.textContent.includes('先 Norm(x)')),
 'the board says it is waiting for Norm(x)');
blockFlow.paint(1.3);
assert.ok(Number(blockFlow.svg.dataset.readOut)>0,'Q reads after Norm completes');
assert.ok(nodesByClass(blockFlow.svg,'cmp-beam').some(n=>n.style.display!=='none'),'read beam appears');

blockFlow.paint(2.4);
assert.equal(blockFlow.svg.dataset.resultState,'x1-pending');
assert.equal(blockFlow.svg.dataset.x1,'','full x1 is not published during the first merge');
const x1Card=blockMergeCard(blockFlow.svg,'x1 p3');
assert.equal(x1Card.dataset.done,'false');
assert.equal(x1Card.dataset.result,'');
assert.ok(walk(x1Card).some(n=>n.tagName==='text'&&n.textContent==='合并中…'),'first merge shows progress, not a result');
const x1Partial=blockMergeRow(x1Card,0);
const x1BaseEnd=Number(x1Partial.base.attrs.x)+Number(x1Partial.base.attrs.width);
near(Number(x1Partial.delta.attrs.x),x1BaseEnd,.01,'first merge grows from the x base');
near(Number(x1Partial.dot.attrs.cx),x1BaseEnd+Number(x1Partial.delta.attrs.width),.01,
 'partial x1 endpoint is base plus grown delta');
assert.ok(Number(x1Partial.delta.attrs.width)>0);

for(const t of [3.4,3.6]){
 blockFlow.paint(t);
 assert.equal(blockFlow.svg.dataset.mlpDone,'false',`MLP is not done at ${t}`);
 assert.equal(blockFlow.svg.dataset.mlpDelta,'',`ΔM is not published at ${t}`);
 assert.ok(Number(blockFlow.svg.dataset.mlpProgress)>0&&Number(blockFlow.svg.dataset.mlpProgress)<1,
  `MLP is still in progress at ${t}`);
 assert.ok(blockFlow.svg.dataset.x1!=='','x1 is already published before the MLP');
 for(const node of nodesByClass(blockFlow.svg,'cmp-mlp-delta')){
  assert.equal(node.textContent,'ΔM …',`no ΔM text at ${t}`);
  assert.ok(!(node.attrs.class||'').includes('is-done'),`no done ΔM class at ${t}`);
 }
}

blockFlow.paint(4.4);
assert.equal(blockFlow.svg.dataset.mlpDone,'true');
assert.equal(blockFlow.svg.dataset.y,'','y is not published during the second merge');
assert.equal(blockFlow.svg.dataset.resultState,'merging');
nearVector(blockFlow.svg.dataset.mlpDelta,blockP3.dM,1e-6,'published ΔM at 4.4');
const yCard=blockMergeCard(blockFlow.svg,'y p3');
assert.equal(yCard.dataset.done,'false');
assert.equal(yCard.dataset.result,'');
const yPartial=blockMergeRow(yCard,0);
const yBaseEnd=Number(yPartial.base.attrs.x)+Number(yPartial.base.attrs.width);
const yPartialEnd=Number(yPartial.dot.attrs.cx);
assert.ok(yPartialEnd>yBaseEnd,'second merge has started from x1');
assert.ok(yPartialEnd<yBaseEnd+Number(yPartial.delta.attrs.width),'second merge is not complete at 4.4');
assert.ok(walk(yCard).some(n=>n.tagName==='text'&&n.textContent==='合并中…'));

blockFlow.paint(5);
assert.equal(blockFlow.svg.dataset.resultState,'complete');
nearVector(blockFlow.svg.dataset.attn,blockP3.attnOut,1e-6,'final ΔA');
nearVector(blockFlow.svg.dataset.x1,blockP3.x1,1e-6,'final x1');
nearVector(blockFlow.svg.dataset.mlpDelta,blockP3.dM,1e-6,'final ΔM');
nearVector(blockFlow.svg.dataset.y,blockP3.y,1e-6,'final y');
assert.equal(blockFlow.svg.dataset.x,'[2.000000, 0.000000]');
assert.ok(walk(blockFlow.svg).some(n=>n.tagName==='text'&&n.textContent==='= [3.39, 2.03]'),'final y is visible');
const yFinal=blockMergeRow(yCard,0);
const yFinalBaseEnd=Number(yFinal.base.attrs.x)+Number(yFinal.base.attrs.width);
near(Number(yFinal.dot.attrs.cx),yFinalBaseEnd+Number(yFinal.delta.attrs.width),.01,
 'final y endpoint is the x1 base plus the ΔM bar');
assert.ok(Number(yFinal.dot.attrs.cx)>yPartialEnd,'the result endpoint advances to completion');
near(Number(yFinal.delta.attrs.width)/Number(yFinal.base.attrs.width),blockP3.dM[0]/blockP3.x1[0],2e-3,
 'ΔM bar length is proportional to the model delta');
assert.notEqual(yFinal.base.style.display,'none','the x1 base is drawn under the second merge');
const skipPaths=nodesByClass(blockFlow.svg,'cmp-block-skip');
assert.ok(skipPaths.length>=2&&skipPaths.every(n=>n.attrs.d&&n.style.display!=='none'),'both bypasses are visible');
const skipTags=walk(blockFlow.svg).filter(n=>n.tagName==='text'&&n.style.display!=='none').map(n=>n.textContent);
assert.ok(skipTags.includes('x 旁路')&&skipTags.includes('x1 旁路'),'both bypass sources are labelled');

// Removing the bypasses is a real counterfactual: no base bars, deltas start
// at zero, x1 is ΔA alone and y is the recomputed ΔM alone.
change(block,'断开旁路（对照）',svg=>{
 assert.equal(svg.dataset.ablation,'true');
 assert.equal(svg.dataset.residual,'counterfactual-off');
 assert.equal(svg.dataset.firstBypassFrom,'none-counterfactual');
 assert.equal(svg.dataset.secondBypassFrom,'none-counterfactual');
 assert.ok(walk(svg).some(n=>n.tagName==='g'&&n.cmpLabelText==='恢复本例旁路'),'toggle switches to restore');
});
assert.equal(block.position,0);
block.paint(5);
assert.equal(block.svg.dataset.focus,'p3');
nearVector(block.svg.dataset.x1,blockOffP3.x1,1e-6,'no-bypass x1 is ΔA');
nearVector(block.svg.dataset.y,blockOffP3.y,1e-6,'no-bypass y is ΔM');
assert.equal(blockMergeCard(block.svg,'x1 p3').dataset.result,'[0.80, 0.80]');
assert.equal(blockMergeCard(block.svg,'y p3').dataset.result,'[1.76, 0]');
const offX1Card=blockMergeCard(block.svg,'x1 p3');
assert.ok(walk(offX1Card).some(n=>n.tagName==='text'&&n.textContent==='仅 ΔA（对照）'),'x1 card names the no-bypass counterfactual');
assert.ok(walk(blockMergeCard(block.svg,'y p3')).some(n=>n.tagName==='text'&&n.textContent==='仅 ΔM（对照）'),
 'y card names the no-bypass counterfactual');
for(const label of ['x1 p3','y p3']){
 const card=blockMergeCard(block.svg,label);
 for(const row of [0,1])assert.equal(blockMergeRow(card,row).base.style.display,'none',`${label} hides the base bar`);
}
const offRow=blockMergeRow(offX1Card,0);
near(Number(offRow.delta.attrs.x),Number(nodesByClass(offX1Card,'cmp-block-zero')[0].attrs.x1),.01,
 'no-bypass delta grows from zero');
const offY=vectorFromDataset(block.svg.dataset.y);
assert.ok(Math.abs(offY[0]-(blockP3.y[0]-blockP3.x[0]))>1e-3,'no-bypass render is not default y minus x');
const offSkips=nodesByClass(block.svg,'cmp-block-skip');
assert.ok(offSkips.length>=2&&offSkips.every(n=>n.style.display==='none'),'both bypass paths are removed');
assert.ok(!walk(block.svg).some(n=>n.tagName==='text'&&n.style.display!=='none'&&
 (n.textContent==='x 旁路'||n.textContent==='x1 旁路')),'removed bypass labels are hidden');

change(block,'恢复本例旁路',svg=>{
 assert.equal(svg.dataset.ablation,'false');
 assert.equal(svg.dataset.residual,'on');
 assert.equal(svg.dataset.firstBypassFrom,'x');
 assert.equal(svg.dataset.secondBypassFrom,'x1');
 assert.ok(walk(svg).some(n=>n.tagName==='g'&&n.cmpLabelText==='断开旁路（对照）'),'toggle switches back');
});
assert.equal(block.position,0);
block.paint(5);
nearVector(block.svg.dataset.y,blockP3.y,1e-6,'restored y matches the model default');
assert.equal(blockMergeCard(block.svg,'y p3').dataset.result,'[3.39, 2.03]');
assert.notEqual(blockMergeRow(blockMergeCard(block.svg,'y p3'),0).base.style.display,'none',
 'restored second merge keeps the x1 base');
const restoredSkips=nodesByClass(block.svg,'cmp-block-skip');
assert.ok(restoredSkips.length>=2&&restoredSkips.every(n=>n.style.display!=='none'),'both bypasses are drawn again');
const restoredTags=walk(block.svg).filter(n=>n.tagName==='text'&&n.style.display!=='none').map(n=>n.textContent);
assert.ok(restoredTags.includes('x 旁路')&&restoredTags.includes('x1 旁路'),'restored bypass labels return');

console.log('PASS: component conditions — changed options update immediately, rewind to 0, preserve progress, and Block gates both residual merges.');
