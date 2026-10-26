/* Deterministic teaching model for the six Transformer components.
 * Every function is a pure calculation over the fixed teaching examples in
 * layer-facts.md and component-independent-examples.json. No model weights,
 * GPU memory or trained architecture is implied. */
(function(root){
'use strict';
const TAU=Math.PI*2;
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*(b[i]||0),0);
const sum=a=>a.reduce((s,x)=>s+x,0);
const rotate=(v,angle)=>{
 const c=Math.cos(angle),s=Math.sin(angle);
 return [c*v[0]-s*v[1],s*v[0]+c*v[1]];
};
const silu=x=>x/(1+Math.exp(-x));
const close=(a,b,eps=1e-9)=>Math.abs(a-b)<=eps;

function rope(pQ,pK,theta=Math.PI/4,q=[1,0],k=[1,0]){
 const qAngle=pQ*theta,kAngle=pK*theta;
 const qRotated=rotate(q,qAngle),kRotated=rotate(k,kAngle);
 return {
  pQ,pK,theta,q,k,qAngle,kAngle,qRotated,kRotated,
  delta:pQ-pK,relativeAngle:(pQ-pK)*theta,
  dot:dot(qRotated,kRotated),
  qLength:Math.hypot(...qRotated),kLength:Math.hypot(...kRotated)
 };
}
// The independent Codex example: q=[1,0], k=[1,1], positions 1/3, theta=pi/4.
// Shifting both positions by one leaves the same relative displacement.
function ropeIndependent(shift=0){
 return rope(1+shift,3+shift,Math.PI/4,[1,0],[1,1]);
}

function rmsnorm(x,gamma=[1,1],eps=1e-6){
 const squares=x.map(v=>v*v),meanSquare=sum(squares)/x.length;
 const rms=Math.sqrt(meanSquare+eps),normalized=x.map(v=>v/rms);
 const output=normalized.map((v,i)=>v*(gamma[i]===undefined?1:gamma[i]));
 const cos=dot(output,x)/(Math.hypot(...output)*Math.hypot(...x));
 return {x,gamma,eps,squares,mean:sum(x)/x.length,meanSquare,rms,normalized,output,
  cosine:cos,directionPreserved:close(cos,1)};
}

function residual(x,fx){
 const sumVector=x.map((v,i)=>v+fx[i]);
 const concatenated=[...x,...fx];
 return {x,fx,sum:sumVector,concatenated,shape:x.length,
  elementwise:true,isConcatenation:false};
}

function matVec(x,matrix){
 return matrix[0].map((_,j)=>x.reduce((s,v,i)=>s+v*matrix[i][j],0));
}
function swiglu(x,Wg=[[1,-1],[0,0]],Wu=[[0,1],[1,1]],Wd=[[1,0],[0,1]]){
 const g=matVec(x,Wg),u=matVec(x,Wu),activated=g.map(silu);
 const z=activated.map((v,i)=>v*u[i]),y=matVec(z,Wd);
 return {x,Wg,Wu,Wd,g,u,silu:activated,z,y,
  hasNegativeSilu:activated.some(v=>v<0),
  siluRangeIsUnit:activated.every(v=>v>=0&&v<=1),
  isExpertRouting:false};
}

function kvHeads(kv,{qHeads=4,positions=3,dim=2,outputWidth=2}={}){
 const groups=Array.from({length:kv},(_,i)=>i);
 const mapping=Array.from({length:qHeads},(_,q)=>({q,group:Math.floor(q*kv/qHeads)}));
 const scalars=2*positions*dim*kv;
 return {qHeads,kvHeads:kv,positions,dim,outputWidth,groups,mapping,scalars,
  qIndependent:true,concatDim:qHeads*dim,outputProjection:true,
  note:'K+V 标量 = 2 × 位置数 × 每头维度 × KV 组数；只是该层 KV 数。'};
}

function contextMotive(mode='self'){
 const Q=[0,1],K=[[1,0],[0,1]],V=[[2,0],[0,2]],scale=Math.SQRT2;
 const scores=K.map(k=>dot(Q,k)/scale);
 const exps=scores.map(x=>Math.exp(x)),denominator=sum(exps);
 const contextWeights=exps.map(x=>x/denominator);
 const weights=mode==='context'?contextWeights:[0,1];
 const contributions=V.map((v,i)=>v.map(x=>weights[i]*x));
 const result=[0,1].map(j=>sum(contributions.map(c=>c[j])));
 return {mode,Q,K,V,scores,exps,denominator,weights,contextWeights,contributions,result,
  readRange:mode==='context'?'p1、p2':'只读 p2（p1 人为置 0）',
  isPerformanceClaim:false,isAccuracyClaim:false};
}

const block={order:['x','RMSNorm','Attention','+x','RMSNorm','SwiGLU MLP','+x₁','y'],
 residuals:2,residualInputs:['x','x₁'],preNorm:true,attentionContains:['RoPE','MHA/GQA/MQA'],mlp:'SwiGLU',
 oneLayerIsOneToken:false,
 note:'一次前向经过全部层后由输出头给出下一 token 分数；生成本身还要选择。'};

const api={rope,ropeIndependent,rmsnorm,residual,swiglu,kvHeads,contextMotive,block,
 rotate,silu,dot,TAU};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ComponentModel=api;
})(typeof globalThis!=='undefined'?globalThis:this);
