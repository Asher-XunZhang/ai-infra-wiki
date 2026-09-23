(function () {
  'use strict';
  const model = window.RequestRuntimeModel, host = document.querySelector('[data-runtime-lab]');
  if (!host || !model) return;
  const get = id => host.querySelector('#' + id);
  const actors = [['front','前端进程'],['ipc','IPC 通路'],['scheduler','调度职责'],['worker','模型执行'],['output','输出回程'],['memory','资源管理']];
  let frames, index = 0, timer = null;
  const tokens = (count, prefix) => Array.from({length: count}, (_,i) => `<span class="runtime-token">${prefix}${i+1}</span>`).join('');
  function stop() { clearInterval(timer); timer = null; get('play').textContent = '播放'; get('play').setAttribute('aria-pressed','false'); }
  function render() {
    const f = frames[index];
    get('step').value = index; get('step-count').textContent = `${index + 1} / ${frames.length}`;
    get('previous').disabled = index === 0; get('next').disabled = index === frames.length - 1;
    get('step').setAttribute('aria-valuetext', `${index + 1}：${f.title}`);
    get('runtime-status').textContent = `第 ${index + 1} 步：${f.title}`;
    get('runtime-snapshot').dataset.frame = f.id;
    get('runtime-snapshot').innerHTML = `<div class="runtime-actors">${actors.map(([a,n])=>`<span class="${a===f.actor?'active':''}">${a===f.actor?'当前 · ':''}${n}</span>`).join('')}</div>
      <div class="runtime-event"><h3>${f.title}</h3><div class="runtime-payload">${f.payload}</div><p>${f.explanation}</p></div>
      <div class="runtime-ledgers">
        <section class="runtime-ledger"><h4>① 请求身份 · 同一个 rid，不同对象</h4><strong>R1</strong><p>前端：${f.front}</p><p>调度侧：${f.scheduler}</p><p class="runtime-mini">dispatched=${f.dispatched} · abort_sent=${f.abortSent}</p><p class="runtime-mini">to_finish=${f.toFinish?'abort':'无'} · 后端 finished=${f.finished}</p></section>
        <section class="runtime-ledger"><h4>② 本轮工作单 · batch 会改变</h4><strong>${f.batch}</strong><p>等待队列：${f.waiting?'[R1]':'[]'}</p><p>模型前向累计：${f.forwards} 次</p><p class="runtime-mini">Worker / ModelRunner 在本例 Scheduler 进程内；这里按职责高亮。</p></section>
        <section class="runtime-ledger"><h4>③ 资源归属 · 请求行与 KV 分开</h4><p>请求槽位：<strong>${f.slot?'R1 持有':'本例未持有'}</strong></p><p>请求使用的 KV：${f.kv} 个位置</p><div class="runtime-tokens">${tokens(Math.min(f.kv,4),'p')}${tokens(Math.max(0,f.kv-4),'y')}</div><p>缓存接管的 KV：${f.cached} 个位置</p><div class="runtime-tokens">${tokens(Math.min(f.cached,4),'p')}${tokens(Math.max(0,f.cached-4),'y')}</div><p class="runtime-mini">按位置计数，非字节数；page_size=1 的教学模型。</p></section>
        <section class="runtime-ledger"><h4>④ 输出进度 · 采样与可见分开</h4><p>后端已采样：${f.sampled}</p><div class="runtime-tokens">${tokens(f.sampled,'y')}</div><p>示例中已回传：${f.visible}</p><div class="runtime-tokens">${tokens(f.visible,'y')}${f.sampled>f.visible?'<span class="runtime-token pending">其余未展示</span>':''}</div><p class="runtime-mini">取消分支不模拟额外结果的文本协议；网络断开时不保证可达。</p></section>
      </div><p class="runtime-evidence"><a href="${model.sourceURL(f.source)}">核对当前步骤源码 · ${model.sources[f.source][2]} ↗</a></p>`;
    if (index === frames.length - 1) stop();
  }
  function rebuild() {
    stop(); index = 0;
    const normal = get('scenario').value === 'normal';
    get('outputs').disabled = !normal;
    frames = model.build({scenario: get('scenario').value, outputs: Number(get('outputs').value), cache: get('cache').checked});
    get('step').max = frames.length - 1; render();
  }
  ['scenario','outputs','cache'].forEach(id=>get(id).addEventListener('change',rebuild));
  get('previous').addEventListener('click',()=>{stop();index=Math.max(0,index-1);render();});
  get('next').addEventListener('click',()=>{stop();index=Math.min(frames.length-1,index+1);render();});
  get('reset').addEventListener('click',rebuild);
  get('step').addEventListener('input',()=>{stop();index=Number(get('step').value);render();});
  get('play').addEventListener('click',()=>{
    if(timer){stop();return;}
    if(index===frames.length-1){index=0;render();}
    get('play').textContent='暂停';get('play').setAttribute('aria-pressed','true');
    timer=setInterval(()=>{index++;render();},2200);
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  window.addEventListener('pagehide',stop);
  host.hidden = false; rebuild();
})();
