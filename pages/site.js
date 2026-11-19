/* Legacy lesson playback shares a live speed slider. New 1× uses half the old
 * interval. Only teaching timers use this API; UI/debounce timers are unchanged. */
(()=>{'use strict';let rate=1;const timers=new Set(),sliders=new Set();
 function mount(container){for(const c of sliders)if(!c.input.isConnected)sliders.delete(c);if(!container||container.closest('.motion-controls')||container.querySelector('.animation-speed'))return;
  const label=document.createElement('label'),input=document.createElement('input'),out=document.createElement('output');label.className='animation-speed';label.append('速度 ');input.type='range';input.min=.25;input.max=3;input.step=.25;input.value=rate;input.setAttribute('aria-label','动画速度');out.textContent=rate+'×';label.append(input,out);container.append(label);sliders.add({input,out});
  input.addEventListener('input',()=>{rate=Number(input.value);for(const c of sliders){c.input.value=rate;c.out.textContent=rate+'×';c.input.setAttribute('aria-valuetext',rate+' 倍');}for(const t of timers)t.reschedule();});
 }
 function repeat(callback,oldMilliseconds){const t={id:null,stopped:false,reschedule(){clearInterval(t.id);t.id=setInterval(callback,oldMilliseconds/2/rate);}};t.reschedule();timers.add(t);return t;}
 function stop(t){if(!t)return;if(typeof t==='object'){clearInterval(t.id);t.stopped=true;timers.delete(t);}else clearInterval(t);}
 function scan(){document.querySelectorAll('button[id$="-play"],button#play,button[data-action="play"]').forEach(b=>mount(b.parentElement));}
 window.LearningPlayback={mount,repeat,stop,get rate(){return rate;}};
 if(document.readyState!=='complete')document.addEventListener('DOMContentLoaded',scan);else scan();
 window.addEventListener('pagehide',()=>{for(const t of timers)stop(t);});
})();
(() => {
  'use strict';
  const root = document.documentElement;
  const themes = ['system', 'light', 'dark'];
  const labels = {system:'系统',light:'浅色',dark:'深色'};
  let theme = 'system';
  try { const saved=localStorage.getItem('ai-infra-theme');if(themes.includes(saved))theme=saved; } catch {}
  function sendTheme(){
    document.querySelectorAll('iframe').forEach(frame=>frame.contentWindow?.postMessage({type:'learning-theme',theme},'*'));
  }
  function applyTheme(){
    root.dataset.theme=theme;
    document.querySelectorAll('.theme-toggle').forEach(button=>{
      button.textContent='配色：'+labels[theme];
      button.setAttribute('aria-label','切换配色，当前'+labels[theme]);
    });
    sendTheme();
  }
  document.querySelectorAll('.theme-toggle').forEach(button=>button.addEventListener('click',()=>{
    theme=themes[(themes.indexOf(theme)+1)%themes.length];applyTheme();
    try { localStorage.setItem('ai-infra-theme',theme); } catch {}
  }));
  document.querySelectorAll('iframe').forEach(frame=>frame.addEventListener('load',sendTheme));
  window.addEventListener('message',event=>{
    if(event.data?.type==='learning-theme-ready'&&[...document.querySelectorAll('iframe')].some(frame=>event.source===frame.contentWindow))sendTheme();
  });
  applyTheme();
  // Route links span static lessons and hash-addressed comparison topics.
  const routePagers = [...document.querySelectorAll('[data-reading-route]')];
  const canonicalPager = document.querySelector('.lesson-pager');
  const routeContext = document.createElement('aside');
  routeContext.className = 'active-reading-route';
  routeContext.hidden = true;
  routeContext.setAttribute('aria-label', '当前学习路线');
  document.querySelector('.learning-breadcrumb')?.after(routeContext);
  function updateReadingRoute() {
    const current = new URL(location.href), id = current.searchParams.get('learn');
    const candidates = routePagers.filter(nav => nav.dataset.readingRoute === id);
    const engine = document.querySelector('.ia-page');
    const hash = current.hash.slice(1);
    const topic = candidates.some(nav => nav.dataset.routeTopic === hash) ? hash : engine?.dataset.chapter || hash || 'process';
    const active = candidates.find(nav => !nav.dataset.routeTopic || nav.dataset.routeTopic === topic);
    routePagers.forEach(nav => { nav.hidden = nav !== active; });
    const topicPager = document.querySelector('.ia-topic-pager');
    if (canonicalPager) canonicalPager.hidden = Boolean(active || topicPager);
    if (topicPager) topicPager.hidden = Boolean(active);
    routeContext.replaceChildren();
    routeContext.hidden = !candidates.length;
    if (!candidates.length) return;
    const info = active || candidates[0];
    const title = document.createElement('strong');
    title.textContent = info.dataset.routeTitle + (active ? ' · ' + info.dataset.routePosition : ' · 当前主题为选读');
    const hint = document.createElement('p');
    hint.textContent = active ? info.dataset.routeHint : '可继续查阅本主题，或回路线目录接着读。';
    const homeLink = document.createElement('a');homeLink.href = info.dataset.routeHome;homeLink.textContent = '路线目录';
    const exitLink = document.createElement('a');exitLink.dataset.routeExit='true';current.searchParams.delete('learn');exitLink.href = current;exitLink.textContent = '按章节自由阅读';
    exitLink.addEventListener('click', event => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();const next = new URL(location.href);next.searchParams.delete('learn');history.replaceState(history.state, '', next);updateReadingRoute();
    });
    routeContext.append(title, hint, homeLink);
    if(active){const nextLink=document.createElement('a');nextLink.href=active.querySelector('.next').href;nextLink.textContent='下一步 →';nextLink.dataset.routeContinue='true';routeContext.append(nextLink);}
    routeContext.append(exitLink);
  }
  updateReadingRoute();
  for (const event of ['popstate','hashchange','learning:navigation']) window.addEventListener(event, updateReadingRoute);
  const small=matchMedia('(max-width:900px)'),menu=document.querySelector('.sidebar-menu');
  const sidebar=document.querySelector('.site-sidebar');
  const groups=sidebar?.querySelectorAll('.topic-group,.module-nav,.chapter-nav') ?? [];
  const home=Boolean(sidebar?.querySelector('.overview-link[aria-current="page"]'));
  function adaptMenu(){if(menu)menu.open=!small.matches;}
  small.addEventListener('change',adaptMenu);adaptMenu();
  if(home){
    // The homepage starts with its collapsed HTML navigation on every visit.
    window.addEventListener('pageshow',event=>{
      if(event.persisted){
        groups.forEach(group=>{group.open=false;});
        adaptMenu();
        sidebar.scrollTop=0;
      }
      if(!location.hash)requestAnimationFrame(()=>window.scrollTo({top:0,behavior:'instant'}));
    });
  } else {
    const saveGroups=[];
    const sitePath=new URL(document.querySelector('.site-brand')?.href||'./',location.href).pathname;
    const currentChapter=sidebar?.querySelector('.chapter-nav:has([aria-current="page"])') ||
      (document.querySelector('#atlas-root') ? sidebar?.querySelector('[data-chapter="engine-comparison"]') : null);
    const chapterId=currentChapter?.dataset.chapter || 'sglang';
    let crossedChapter=true;
    try {const key='ai-infra-current-chapter:'+sitePath;crossedChapter=sessionStorage.getItem(key)!==chapterId;sessionStorage.setItem(key,chapterId);} catch {}
    groups.forEach(group=>{
      const key=group.classList.contains('chapter-nav')?'ai-infra-chapter-'+group.dataset.chapter:group.classList.contains('module-nav')?'ai-infra-module-'+group.dataset.module:'ai-infra-topic-'+group.dataset.topic;
      const current=group===currentChapter || Boolean(group.querySelector('[aria-current="page"]'));
      if(current)group.open=true;
      else {
        try {const stored=sessionStorage.getItem(key);if(stored!==null)group.open=stored==='open';} catch {}
        if(crossedChapter&&group.classList.contains('chapter-nav'))group.open=false;
      }
      const saveGroup=()=>{
        try {sessionStorage.setItem(key,group.open?'open':'closed');} catch {}
      };
      // Also remember the current lesson's initially open ancestors.
      saveGroup();
      saveGroups.push(saveGroup);
      group.addEventListener('toggle',saveGroup);
    });
    const scrollKey='ai-infra-sidebar-scroll:'+sitePath+':'+chapterId;
    function restoreSidebar(){
      if(!sidebar||small.matches)return;
      try {
        const saved=sessionStorage.getItem(scrollKey),top=Number(saved);
        if(saved!==null&&Number.isFinite(top)&&top>=0)sidebar.scrollTop=top;
      } catch {}
    }
    function saveSidebar(){
      // On mobile the menu is part of the document, not a scrollable sidebar.
      if(!sidebar||small.matches)return;
      try {sessionStorage.setItem(scrollKey,String(sidebar.scrollTop));} catch {}
    }
    small.addEventListener('change',restoreSidebar);restoreSidebar();
    let focusChapter=crossedChapter;
    function revealCurrentChapter(){
      if(!focusChapter||!sidebar||small.matches)return;
      const link=sidebar.querySelector('[aria-current="page"],[aria-current="location"]');
      if(!link)return;
      const row=link.getBoundingClientRect(),bounds=sidebar.getBoundingClientRect();
      if(row.top<bounds.top+16)sidebar.scrollTop+=row.top-bounds.top-16;
      else if(row.bottom>bounds.bottom-16)sidebar.scrollTop+=row.bottom-bounds.bottom+16;
      focusChapter=false;
    }
    requestAnimationFrame(revealCurrentChapter);
    window.addEventListener('learning:navigation',revealCurrentChapter);
    sidebar?.addEventListener('scroll',saveSidebar,{passive:true});
    function saveNavigation(){saveGroups.forEach(save=>save());saveSidebar();}
    // Capture state before navigation, even before queued scroll/toggle events.
    sidebar?.addEventListener('click',saveNavigation,{capture:true});
    window.addEventListener('pagehide',saveNavigation);
  }
  const toc=document.querySelector('.article-toc details'),compact=matchMedia('(max-width:1220px)');
  function adaptToc(){if(toc)toc.open=!compact.matches;}
  compact.addEventListener('change',adaptToc);adaptToc();
  const links=[...document.querySelectorAll('.article-toc nav a')];
  if(links.length){
    let pending=false;
    const update=()=>{
      pending=false;
      let active=links[0];
      for(const link of links){const target=document.getElementById(link.hash.slice(1));if(target&&target.getBoundingClientRect().top<=150)active=link;}
      links.forEach(link=>{if(link===active)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');});
    };
    document.addEventListener('scroll',()=>{if(!pending){pending=true;requestAnimationFrame(update);}},{passive:true});
    update();
  }
  // A deep link must reveal the explanation even when it lives in a closed experiment.
  function revealTarget(hash, scroll=false){
    let target;try{target=document.getElementById(decodeURIComponent(hash.slice(1)));}catch{return;}
    if(!target)return;
    let opened=false;
    for(let e=target;e;e=e.parentElement){if(e.tagName==='DETAILS'&&!e.open){e.open=true;opened=true;}}
    if(scroll&&opened)requestAnimationFrame(()=>target.scrollIntoView({block:'start'}));
  }
  document.addEventListener('click',event=>{const a=event.target.closest('a[href]');if(!a)return;const dest=new URL(a.href,location.href);if(dest.origin===location.origin&&dest.pathname===location.pathname&&dest.search===location.search)revealTarget(dest.hash);});
  window.addEventListener('hashchange',()=>revealTarget(location.hash,true));
  window.addEventListener('load',()=>revealTarget(location.hash,true));
})();
