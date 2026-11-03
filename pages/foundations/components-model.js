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
 const plainDot=dot(q,k),rotatedDot=dot(qRotated,kRotated);
 return {
  pQ,pK,theta,q,k,qAngle,kAngle,qRotated,kRotated,
  delta:pQ-pK,relativeAngle:(pQ-pK)*theta,
  plainDot,rotatedDot,
  dot:rotatedDot,
  qAngleDegrees:qAngle*180/Math.PI,
  kAngleDegrees:kAngle*180/Math.PI,
  relativeAngleDegrees:(pQ-pK)*theta*180/Math.PI,
  qLength:Math.hypot(...qRotated),kLength:Math.hypot(...kRotated),
  rotationPreservesLength:close(Math.hypot(...qRotated),Math.hypot(...q))
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
 const uniformGamma=gamma.every(v=>Math.abs(v-gamma[0])<=1e-12);
 return {x,gamma,eps,squares,mean:sum(x)/x.length,meanSquare,rms,normalized,output,
  cosine:cos,directionPreserved:close(cos,1),
  normalizedLength:Math.hypot(...normalized),
  outputLength:Math.hypot(...output),
  normalizedRms:Math.sqrt(sum(normalized.map(v=>v*v))/normalized.length),
  outputRms:Math.sqrt(sum(output.map(v=>v*v))/output.length),
  uniformGamma};
}

function residual(x,fx){
 const sumVector=x.map((v,i)=>v+fx[i]);
 const concatenated=[...x,...fx];
 const zeroUpdate=fx.every(v=>Math.abs(v)<=1e-12);
 return {x,fx,sum:sumVector,concatenated,shape:x.length,
  elementwise:true,isConcatenation:false,
  zeroUpdate,unchanged:zeroUpdate?x.slice():null,
  updateMagnitude:Math.hypot(...fx),outputMagnitude:Math.hypot(...sumVector)};
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

function swigluExample(x=[1,2]){
 // Explicit teaching matrices. The default swiglu(x) API keeps its original
 // weights so the accepted hand-check evidence stays valid.
 const Wg=[[1,1],[1,-1]],Wu=[[0,1],[1,1]],Wd=[[1,0],[0,1]];
 const r=swiglu(x,Wg,Wu,Wd);
 const combinations=[
  {name:'g0',expr:'x0 + x1',value:r.g[0]},
  {name:'g1',expr:'x0 − x1',value:r.g[1]},
  {name:'u0',expr:'x1',value:r.u[0]},
  {name:'u1',expr:'x0 + x1',value:r.u[1]}
 ];
 return {...r,Wg,Wu,Wd,combinations,
  gateMin:Math.min(...r.silu),gateMax:Math.max(...r.silu),
  gateCanExceedOne:r.silu.some(v=>v>1),
  gateCanBeNegative:r.silu.some(v=>v<0),
  gateIsProbability:false,gateIsSwitch:false,isExpertRouting:false,
  note:'显式教学矩阵；g/u 只是输入维度的组合，不赋予某个维度固定的人类语义。'};
}

function swigluPurpose(x=[1,2]){
 // The default board shows two readable feature lanes and the coefficient
 // that changes their contribution. Matrices stay in swigluExample() for the
 // optional hand-check panel; this view deliberately keeps no Wg/Wu/Wd keys.
 const r=swigluExample(x);
 return {
  x:r.x.slice(),
  features:[
   {id:'A',label:'特征 A',source:'x1',value:r.u[0]},
   {id:'B',label:'特征 B',source:'x0 + x1',value:r.u[1]}
  ],
  coefficients:[
   {id:'a0',value:r.silu[0],g:r.g[0]},
   {id:'a1',value:r.silu[1],g:r.g[1]}
  ],
  contributions:r.z.slice(),
  y:r.y.slice(),
  coefficientCanBeNegative:r.gateCanBeNegative,
  coefficientCanExceedOne:r.gateCanExceedOne,
  isProbability:false,isSwitch:false,isRouting:false,
  note:'两条特征只是本图的人工教学示意；不对应真实神经元里固定的人类语义。'
 };
}

function kvHeads(kv,{qHeads=4,positions=3,dim=2,outputWidth=2}={}){
 const groups=Array.from({length:kv},(_,i)=>i);
 const mapping=Array.from({length:qHeads},(_,q)=>({q,group:Math.floor(q*kv/qHeads)}));
 const perGroupPerPosition=2*dim;
 const perPosition=perGroupPerPosition*kv;
 const scalars=positions*perPosition;
 const qProjectionDimensions=qHeads*dim;
 const outputProjectionMacs=qProjectionDimensions*outputWidth;
 return {qHeads,kvHeads:kv,positions,dim,outputWidth,groups,mapping,scalars,
  perGroupPerPosition,perPosition,growthPerPosition:perPosition,
  qProjectionDimensions,outputProjectionMacs,qScoreEntries:qHeads*positions,
  cacheCopyRatio:kv/qHeads,
  qIndependent:true,concatDim:qHeads*dim,outputProjection:true,
  note:'K+V 标量 = 2 × 位置数 × 每头维度 × KV 组数；只是该层保存的 KV 元素数，不是带宽或总显存。',
  bandwidthNote:'实际读取次数与带宽取决于 kernel；按组计数只是教学比较，不表示实现只读一次。'};
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

function blockPass({x=[[0,2],[1,1],[2,0]],theta=Math.PI/4,gamma=[1,1],eps=1e-6,residual=true}={}){
 // One deterministic teaching layer: single-head causal self-attention with
 // Q/K/V equal to the normalized representation and W_O=I, followed by the
 // explicit SwiGLU teaching matrices. Default is Pre-Norm with both residual
 // additions; `residual:false` is the no-bypass counterfactual only.
 const add=(a,b)=>a.map((v,i)=>v+b[i]);
 const dim=x[0].length;
 const normalized=x.map(v=>rmsnorm(v,gamma,eps));
 const rotated=normalized.map((r,i)=>rotate(r.output,i*theta));
 const positions=x.map((xv,i)=>{
  const normedX=normalized[i].output;
  const q=rotated[i];
  const keys=rotated.slice(0,i+1);
  const rawScores=keys.map(k=>dot(q,k)/Math.sqrt(dim));
  const exps=rawScores.map(v=>Math.exp(v));
  const denom=sum(exps);
  const weights=exps.map(v=>v/denom);
  const values=normalized.slice(0,i+1).map(r=>r.output);
  const attnOut=Array.from({length:dim},(_,d)=>weights.reduce((s,w,j)=>s+w*values[j][d],0));
  const x1=residual?add(xv,attnOut):attnOut.slice();
  const rms1=rmsnorm(x1,gamma,eps);
  const mlp=swigluExample(rms1.output);
  const dM=mlp.y;
  const y=residual?add(x1,dM):dM.slice();
  return {index:i,label:'p'+(i+1),x:xv.slice(),rmsX:normalized[i].rms,normedX,q,
   keys,rawScores,weights,attnOut,x1,rmsX1:rms1.rms,normedX1:rms1.output,
   g:mlp.g,u:mlp.u,a:mlp.silu,z:mlp.z,dM,y,
   readIndices:Array.from({length:i+1},(_,j)=>j),
   readRange:i===0?'p1':'p1-p'+(i+1),
   qAngleDegrees:i*theta*180/Math.PI,
   kAngleDegrees:keys.map((_,j)=>j*theta*180/Math.PI)};
 });
 return {positions,x:x.map(v=>v.slice()),theta,gamma:gamma.slice(),eps,residual,dim,
  preNorm:true,residuals:2,residualInputs:['x','x1'],firstBypassFrom:'x',secondBypassFrom:'x1',
  attention:'causal-cross-position',mlp:'per-position-shared',rope:'q,k',rms:'scale',
  oneLayerIsOneToken:false,teaching:true,modelObservation:false,
  teachingMatrices:{Wg:[[1,1],[1,-1]],Wu:[[0,1],[1,1]],Wd:[[1,0],[0,1]]},
  note:'确定性教学层：二维表示、单头注意力、W_O=I、γ=[1,1]；数值不是模型观测。'};
}

const block={order:['x','RMSNorm','Attention','+x','RMSNorm','SwiGLU MLP','+x₁','y'],
 residuals:2,residualInputs:['x','x₁'],preNorm:true,preNormExample:true,generalizesAllTransformers:false,
 attentionContains:['RoPE','MHA/GQA/MQA'],mlp:'SwiGLU',
 positions:3,attentionRole:'跨位置：Q 只读因果可读的历史位置',mlpRole:'逐位置：列内加工，不混合其它位置',
 normRole:'Pre-Norm：每个子层前控制尺度',residualRole:'旁路保留主路；第二个旁路来自 x₁',
 oneLayerIsOneToken:false,
 nextToken:'全部层之后由输出头给分数，生成过程再做选择。',
 note:'一次前向经过全部层后由输出头给出下一 token 分数；生成本身还要选择。'};

const api={rope,ropeIndependent,rmsnorm,residual,swiglu,swigluExample,swigluPurpose,kvHeads,contextMotive,block,blockPass,
 rotate,silu,dot,TAU};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ComponentModel=api;
})(typeof globalThis!=='undefined'?globalThis:this);
