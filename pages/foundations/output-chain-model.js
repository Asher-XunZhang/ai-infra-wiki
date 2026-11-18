/* Deterministic one-layer teaching model for "法国的首都是哪里？".
 *
 * This is not a trained model or a tokenizer observation. The six prompt
 * pieces are pre-segmented teaching labels, the 2-D vectors are hand-chosen,
 * and the layer is treated as the last layer so the LM Head can be shown once.
 * A real decoder repeats Transformer layers before its output head.
 *
 * The chain is deliberately complete: attention scores -> softmax weights ->
 * weighted V -> residual -> RMSNorm -> SwiGLU -> residual -> LM Head logits ->
 * softmax -> greedy token. The context control removes only p1's K/V row,
 * keeps Q and every other teaching K/V fixed, and recomputes values downstream.
 * This is a local ablation, not a second tokenizer run or a real-model trace.
 */
(function(root){
'use strict';
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*(b[i]||0),0);
const sum=a=>a.reduce((s,x)=>s+x,0);
const silu=x=>x/(1+Math.exp(-x));
const sqrt2=Math.SQRT2;

const POSITIONS=[
 {id:'p1',token:'法国',role:'国家线索',key:[2,0],value:[2.0,0.2]},
 {id:'p2',token:'的',role:'结构词',key:[0,0],value:[0.1,0.0]},
 {id:'p3',token:'首都',role:'问题线索',key:[1,0],value:[1.0,0.8]},
 {id:'p4',token:'是',role:'结构词',key:[0,0],value:[0.0,0.2]},
 {id:'p5',token:'哪里',role:'疑问词',key:[1,0],value:[0.2,1.0]},
 {id:'p6',token:'？',role:'当前预测位置',key:[0,0],value:[0.2,0.2]}
];

// x is the p6 layer input; Q is its teaching query. K/V are fixed teaching
// projections, not derived from the displayed 2-D x.
const X6=[0.2,0.3];
const Q=[sqrt2,0];
const WG=[[1.2,-0.8,0.5],[0.4,0.2,0.4]];
const WU=[[0.6,0.2,-1.0],[0.4,0.1,0.2]];
const WD=[[1.0,0.2],[0.1,-0.5],[0.4,0.8]];
const CANDIDATES=[
 {token:'巴黎',vector:[0.9,-0.6]},
 {token:'里昂',vector:[0.1,1.0]},
 {token:'北京',vector:[-0.4,0.3]},
 {token:'马赛',vector:[0.2,-0.4]}
];

function matVec(x,matrix){
 return matrix[0].map((_,j)=>x.reduce((s,v,i)=>s+v*matrix[i][j],0));
}
function rmsnorm(x){
 const meanSquare=sum(x.map(v=>v*v))/x.length;
 const rms=Math.sqrt(meanSquare);
 return {x:meanSquare,rms,h:x.map(v=>v/rms)};
}
function softmax(values){
 const max=Math.max(...values);
 const exps=values.map(v=>Math.exp(v-max));
 const denominator=sum(exps);
 return {max,exps,denominator,probabilities:exps.map(v=>v/denominator)};
}

function chain(mode='with-france'){
 const removeFrance=mode==='without-france';
 const rows=POSITIONS.filter(p=>!(removeFrance&&p.id==='p1')).map(p=>({
  ...p,
  key:p.key.slice(),
  value:p.value.slice(),
  rawScore:dot(Q,p.key)/Math.sqrt(Q.length)
 }));
 const rawScores=rows.map(r=>r.rawScore);
 const sm=softmax(rawScores);
 rows.forEach((r,i)=>{
  r.weight=sm.probabilities[i];
  r.contribution=r.value.map(v=>r.weight*v);
 });
 const attention=rows[0].value.map((_,d)=>sum(rows.map(r=>r.contribution[d])));
 const x1=X6.map((v,i)=>v+attention[i]);
 const norm=rmsnorm(x1);
 const h=norm.h;
 const g=matVec(h,WG);
 const u=matVec(h,WU);
 const a=g.map(silu);
 const z=a.map((v,i)=>v*u[i]);
 const ffnContributions=z.map((value,j)=>WD[j].map(w=>value*w));
 const dM=matVec(z,WD);
 const y=x1.map((v,i)=>v+dM[i]);
 const head=CANDIDATES.map(c=>({token:c.token,vector:c.vector.slice(),logit:dot(y,c.vector)}));
 const logits=head.map(c=>c.logit);
 const logitSoftmax=softmax(logits);
 head.forEach((c,i)=>{
  c.probability=logitSoftmax.probabilities[i];
  c.contribution=y.map((v,d)=>v*c.vector[d]);
  c.selected=false;
 });
 let selected=0;
 for(let i=1;i<head.length;i++)if(head[i].logit>head[selected].logit)selected=i;
 head[selected].selected=true;

 return {
  mode:removeFrance?'without-france':'with-france',
  condition:removeFrance?'屏蔽「法国」的 K/V':'保留「法国」',
  positions:rows.map(r=>r.id),
  tokens:rows.map(r=>r.token),
  Q:Q.slice(),
  rows,
  rawScores,
  attentionMaxScore:sm.max,
  exps:sm.exps,
  denominator:sm.denominator,
  weights:rows.map(r=>r.weight),
  attention,
  x6:X6.slice(),
  x1,
  rms:norm.rms,
  meanSquare:norm.x,
  h,
  WG:WG.map(r=>r.slice()),WU:WU.map(r=>r.slice()),WD:WD.map(r=>r.slice()),
  g,u,a,z,
  ffnContributions,
  dM,
  y,
  head,
  headMaxLogit:logitSoftmax.max,
  headExps:logitSoftmax.exps,
  softmaxDenominator:logitSoftmax.denominator,
  probabilities:head.map(c=>c.probability),
  selected:head[selected].token,
  teachingOnly:true,
  realModelObservation:false,
  singleLayerSlice:true
 };
}

const api={chain,POSITIONS,Q,X6,WG,WU,WD,CANDIDATES,dot,softmax,rmsnorm,matVec};
if(typeof module!=='undefined'&&module.exports)module.exports=api;
else root.OutputChainModel=api;
})(typeof globalThis!=='undefined'?globalThis:this);
