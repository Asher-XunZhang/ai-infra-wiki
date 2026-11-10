/* Stable reading URLs, explicit updates, and one-use per-tab reading checkpoints. */
(() => {
  'use strict';
  const release = document.querySelector('meta[name="site-release"]')?.content;
  const root = new URL(document.querySelector('meta[name="site-root"]')?.content || './', location.href);
  const valid = value => typeof value === 'string' && /^[a-f0-9]{16}$/.test(value);
  if (!valid(release) || !/^https?:$/.test(location.protocol)) return;
  const key = 'ai-infra-update:' + root.pathname, url = new URL(location.href);
  const nonce = url.searchParams.get('_refresh');
  let resume = null;
  try {
    const saved = JSON.parse(sessionStorage.getItem(key) || 'null');
    if (saved?.schema === 1 && saved.nonce === nonce && nonce && saved.path === url.pathname && Date.now() - saved.at < 1800000) resume = saved;
    sessionStorage.removeItem(key);
  } catch {}
  function clean(address) {
    const result = new URL(address);
    result.searchParams.delete('v'); result.searchParams.delete('_refresh');
    return result;
  }
  const nativeScrollMode = history.scrollRestoration;
  if (resume) history.scrollRestoration = 'manual';
  const stable = clean(url);
  if (stable.href !== location.href) history.replaceState(history.state, '', stable);
  const checkpoints = new Map();
  window.SiteUpdate = {
    register(name, capture) { checkpoints.set(name, capture); return () => { if (checkpoints.get(name) === capture) checkpoints.delete(name); }; },
    take(name) { const value = resume?.state?.[name]; if (resume?.state) delete resume.state[name]; return value; }
  };
  let latest = null, checking = null, reportCheck = false, updating = false, checkedAt = 0, requestedPosition = null, toolbar, status, checkButton, updateButton;
  function render(message = '') {
    if (!toolbar) return;
    const available = latest && latest !== release;
    checkButton.disabled = updating;
    checkButton.setAttribute('aria-busy', String(Boolean(checking)));
    checkButton.textContent = checking ? '检查中…' : '检查更新';
    updateButton.hidden = !available;
    updateButton.disabled = updating;
    updateButton.textContent = updating ? '正在更新…' : '更新并继续';
    status.textContent = message ? (message.includes('最新') ? message : message.includes('同步') ? '新版同步中' : message.includes('无法保存') ? '无法保存进度' : available ? '暂未更新' : '检查失败') : available ? '发现新版' : '';
    status.setAttribute('aria-label', message || status.textContent);
    status.title = message;
    status.hidden = !status.textContent;
    toolbar.dataset.available = String(Boolean(available));
    toolbar.dataset.state = updating ? 'updating' : available ? 'available' : checking ? 'checking' : 'current';
  }
  async function fresh(address) {
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 10000);
    try {
      const response = await fetch(address, { cache: 'no-store', headers: { 'Cache-Control': 'no-cache' }, signal: controller.signal });
      if (!response.ok) throw Error('unavailable');
      return await response.text();
    } finally { clearTimeout(timer); }
  }
  async function check(manual = false) {
    if (updating) return;
    if (checking) { if (manual) reportCheck = true; return checking; }
    if (!manual && (document.hidden || Date.now() - checkedAt < 60000)) return;
    checkedAt = Date.now(); reportCheck = manual;
    checking = (async () => {
      try {
        const address = new URL('site-release.json', root);
        address.searchParams.set('_check', Date.now().toString(36));
        const data = JSON.parse(await fresh(address));
        if (data.schema !== 1 || !valid(data.release)) throw Error('invalid manifest');
        latest = data.release;
        return reportCheck && latest === release ? '已是最新内容' : '';
      } catch { return reportCheck ? '暂时无法检查更新，当前页面可继续阅读。' : ''; }
    })();
    render();
    const message = await checking; checking = null; render(message);
  }
  function detailKey(element) {
    return element.id || ['module', 'chapter', 'topic'].map(k => element.dataset[k] ? k + ':' + element.dataset[k] : '').find(Boolean) || element.querySelector(':scope > summary')?.textContent.trim();
  }
  function readingPosition() {
    const readingTop = (document.querySelector('.site-header')?.getBoundingClientRect().bottom || 76) + 24;
    const content = document.querySelector('main') || document.querySelector('.site-content');
    const anchors = [...content.querySelectorAll('section[id], article[id], h1[id], h2[id], h3[id], [role=tabpanel][id]')].filter(e => e.checkVisibility() && e.getBoundingClientRect().top <= readingTop + 1);
    const anchor = anchors.sort((a, b) => b.getBoundingClientRect().top - a.getBoundingClientRect().top)[0];
    return { x: scrollX, y: scrollY, anchor: anchor?.id, offset: anchor?.getBoundingClientRect().top };
  }
  async function update() {
    if (updating || !latest || latest === release) return;
    const position = requestedPosition && performance.now() - requestedPosition.at < 1500 ? requestedPosition.value : readingPosition();
    requestedPosition = null;
    // Preserve what was being read before focusing the sticky update button.
    // Explicit reading/scrolling during the fetch takes precedence over it.
    let readerMoved = false;
    const trackReader = event => { if (['wheel', 'touchmove'].includes(event.type) || !event.target.closest?.('.site-update-tools')) readerMoved = true; };
    const inputs = ['pointerdown', 'keydown', 'wheel', 'touchmove'];
    for (const type of inputs) document.addEventListener(type, trackReader, { passive: true });
    updating = true; render();
    try {
      const state = {};
      for (const [name, capture] of checkpoints) { const value = await capture(); if (value !== undefined) state[name] = value; }
      const address = clean(location.href), targetRelease = latest;
      const token = targetRelease + '-' + Date.now().toString(36);
      address.searchParams.set('_refresh', token);
      // Check the actual document, not only the manifest: CDN edges may publish
      // at different times. A mismatch stays on this page, with no reload loop.
      const html = await fresh(address), parsed = new DOMParser().parseFromString(html, 'text/html');
      if (parsed.querySelector('meta[name="site-release"]')?.content !== targetRelease) throw Error('syncing');
      // Capture the final URL and live controls again in case the reader moved
      // while the network request was in flight. The verified path is unchanged.
      for (const [name, capture] of checkpoints) { const value = await capture(); if (value !== undefined) state[name] = value; }
      const destination = clean(location.href);
      if (destination.pathname !== address.pathname) throw Error('changed page');
      destination.searchParams.set('_refresh', token);
      const saved = { schema: 1, nonce: token, to: targetRelease, at: Date.now(), path: destination.pathname, state, scroll: readerMoved ? readingPosition() : position, details: [...document.querySelectorAll('details')].map(e => [detailKey(e), e.open]).filter(([k]) => k) };
      try { sessionStorage.setItem(key, JSON.stringify(saved)); }
      catch { throw Error('storage'); }
      location.replace(destination.href);
    } catch (error) {
      updating = false;
      render(error.message === 'syncing' ? '新版仍在同步，请稍后重试。' : error.message === 'storage' ? '无法保存当前位置，更新尚未执行。请稍后重试。' : '更新暂未完成，当前位置已保留，可重试。');
    } finally { for (const type of inputs) document.removeEventListener(type, trackReader); }
  }
  function mount() {
    const container = document.querySelector('.site-header'), main = document.querySelector('main') || document.querySelector('.site-content');
    if (!container || !main) return;
    toolbar = document.createElement('div'); toolbar.className = 'site-update-tools';
    toolbar.setAttribute('role', 'group'); toolbar.setAttribute('aria-label', '页面更新');
    checkButton = document.createElement('button'); checkButton.type = 'button'; checkButton.textContent = '检查更新'; checkButton.dataset.siteCheck = '';
    status = document.createElement('span'); status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
    updateButton = document.createElement('button'); updateButton.type = 'button'; updateButton.dataset.siteUpdate = ''; updateButton.hidden = true;
    toolbar.append(checkButton, status, updateButton); container.insertBefore(toolbar, container.querySelector('.site-actions'));
    // The narrow header wraps in normal flow. Sticky lesson controls and anchor
    // clearance follow its real height, including zoom and wrapped feedback.
    const measureHeader = () => document.documentElement.style.setProperty('--site-header-height', container.getBoundingClientRect().height + 'px');
    measureHeader(); new ResizeObserver(measureHeader).observe(container);
    checkButton.onclick = () => check(true); updateButton.onclick = update;
    for (const type of ['pointerdown', 'keydown']) updateButton.addEventListener(type, event => {
      if (type === 'keydown' && !['Enter', ' '].includes(event.key)) return;
      requestedPosition = { at: performance.now(), value: readingPosition() };
    });
    // Page restoration is a one-use checkpoint, never a persistent auto-resume
    // preference. A normal mode switch/navigation still starts its own clock.
    if (resume) {
      let interacted = false;
      for (const name of ['pointerdown', 'keydown', 'wheel']) document.addEventListener(name, () => { interacted = true; }, { once: true, passive: true });
      const restore = () => requestAnimationFrame(() => requestAnimationFrame(() => {
        if (interacted) { history.scrollRestoration = nativeScrollMode; return; }
        const states = new Map(resume.details || []);
        for (const e of document.querySelectorAll('details')) if (states.has(detailKey(e))) e.open = states.get(detailKey(e));
        const s = resume.scroll, anchor = s?.anchor && document.getElementById(s.anchor);
        if (s) scrollTo({ left: s.x, top: anchor ? anchor.getBoundingClientRect().top + scrollY - s.offset : s.y, behavior: 'instant' });
        requestAnimationFrame(() => { history.scrollRestoration = nativeScrollMode; });
      }));
      if (document.readyState === 'complete') restore(); else window.addEventListener('load', restore, { once: true });
      if (resume.to !== release) render('新版仍在同步，当前位置已恢复，可重新检查更新。');
    }
    check();
    window.addEventListener('pageshow', event => { if (event.persisted) { checkedAt = 0; check(); } });
    window.addEventListener('focus', () => check());
    document.addEventListener('visibilitychange', () => { if (!document.hidden) check(); });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount, { once: true }); else mount();
})();
