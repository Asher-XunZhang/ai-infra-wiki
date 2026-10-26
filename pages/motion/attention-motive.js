/* Attention V11: one fixed Q/K/V set, two read policies. This is an
 * explanatory contrast only: the historical contribution becomes visible in
 * the mixed vector; it is not a performance or accuracy measurement. */
(function(){
'use strict';
const host=document.getElementById('attention-motive');
const M=window.ComponentModel;
if(!host||!M)return;
const app=host.querySelector('#attention-motive-app');
if(!app)return;
const chip=(text,cls='')=>`<span class="motive-chip ${cls}">${text}</span>`;
const self=M.contextMotive('self'),context=M.contextMotive('context');
const fixed=context;
app.innerHTML=`
 <div class="motive-inputs" aria-label="固定输入">
  ${chip('Q = [0, 1]')}${chip('K1 = [1, 0]','p1')}${chip('K2 = [0, 1]','p2')}
  ${chip('V1 = [2, 0]','p1')}${chip('V2 = [0, 2]','p2')}
 </div>
 <div class="motive-controls" role="group" aria-label="允许读取的范围">
  <button type="button" data-motive-mode="self" aria-pressed="true">只读自己 p2</button>
  <button type="button" data-motive-mode="context" aria-pressed="false">允许读 p1、p2</button>
 </div>
 <div>
  <p class="motive-title">分数与权重 · 同一组 Q/K</p>
  <div class="motive-weights">
   <div class="motive-weight"><span>p1</span><div class="motive-track"><div class="motive-fill p1" data-w="0"></div></div><b data-weight="0">0.0%</b></div>
   <div class="motive-weight"><span>p2</span><div class="motive-track"><div class="motive-fill p2" data-w="1"></div></div><b data-weight="1">100.0%</b></div>
  </div>
  <p class="motive-title" style="margin-top:12px">同一输入的两个分数</p>
  <p class="motive-value" data-scores>s = [0.0000, 0.7071]；只读自己时把 p1 的权重人为置 0。</p>
 </div>
 <div>
  <p class="motive-title">结果 m = Σ w·V · 按维分解</p>
  <div class="motive-result">
   <div class="motive-result-row"><span>m₀</span><div class="motive-stack" data-stack="0"><div class="motive-seg p1" data-seg="0-0"></div><div class="motive-seg p2" data-seg="0-1"></div></div><span class="motive-value" data-m="0">0.0000</span></div>
   <div class="motive-result-row"><span>m₁</span><div class="motive-stack" data-stack="1"><div class="motive-seg p1" data-seg="1-0"></div><div class="motive-seg p2" data-seg="1-1"></div></div><span class="motive-value" data-m="1">2.0000</span></div>
  </div>
  <p class="motive-value" data-result aria-live="polite" style="margin-top:8px">m = [0.0000, 2.0000]</p>
 </div>
 <p class="motive-explain" data-explain>只保留 p2 自己的权重 1，p1 权重 0；结果只含自己的 V。</p>
 <p class="motive-boundary">同一组输入，只改变允许读取的范围。历史贡献进入表示是解释性对照，不称性能优化或语义正确率。</p>`;
const buttons=[...app.querySelectorAll('[data-motive-mode]')];
const fills=[...app.querySelectorAll('[data-w]')];
const weightLabels=[...app.querySelectorAll('[data-weight]')];
const segs=[...app.querySelectorAll('[data-seg]')];
const values=[...app.querySelectorAll('[data-m]')];
const result=app.querySelector('[data-result]');
const explain=app.querySelector('[data-explain]');
const scores=app.querySelector('[data-scores]');
const source={self,context};
function render(mode){
 const d=source[mode];
 buttons.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.motiveMode===mode)));
 fills.forEach((f,i)=>{f.style.width=(d.weights[i]*100).toFixed(2)+'%';});
 weightLabels.forEach((b,i)=>{b.textContent=(d.weights[i]*100).toFixed(1)+'%';});
 const max=2;
 for(let j=0;j<2;j++){
  const stack=app.querySelector(`[data-stack="${j}"]`);
  const total=Math.min(1,d.result[j]/max);
  stack.style.width=(total*100).toFixed(2)+'%';
  for(let i=0;i<2;i++){
   const seg=app.querySelector(`[data-seg="${j}-${i}"]`);
   const width=d.result[j]>0?d.contributions[i][j]/d.result[j]*100:0;
   seg.style.width=width.toFixed(2)+'%';
  }
  values[j].textContent=d.result[j].toFixed(4);
 }
 result.textContent=`m = [${d.result.map(x=>x.toFixed(4)).join(', ')}]`;
 scores.textContent=mode==='self'
  ?`s = [0.0000, 0.7071]；只读自己：p1 权重人为置 0，p2 权重 1。`
  :`s = [0.0000, 0.7071]；e^s 合计 ${d.denominator.toFixed(4)}，权重 [${d.weights.map(x=>x.toFixed(8)).join(', ')}]。`;
 explain.textContent=mode==='self'
  ?'只保留 p2 自己的权重 1，p1 权重 0；结果 [0,2] 只含自己的 V。'
  :`p1 的历史贡献 ${d.contributions[0][0].toFixed(4)} 进入第 1 维；p2 贡献 ${d.contributions[1][1].toFixed(4)} 进入第 2 维；结果 [${d.result.map(x=>x.toFixed(4)).join(', ')}]。`;
 host.dataset.mode=mode;
}
buttons.forEach(b=>b.addEventListener('click',()=>render(b.dataset.motiveMode)));
host.dataset.fixed=JSON.stringify(fixed.result);
render('self');
})();
