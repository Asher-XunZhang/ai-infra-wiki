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
  // Learning routes are an optional reading order; module navigation stays intact.
  const url = new URL(location.href);
  const routeChoices = [...document.querySelectorAll('[data-route-choice]')];
  const routePanels = [...document.querySelectorAll('[data-route-panel]')];
  function showRoute(id) {
    const selected = routePanels.find(panel => panel.dataset.routePanel === id) || routePanels[0];
    if (!selected) return;
    routePanels.forEach(panel => { panel.hidden = panel !== selected; });
    routeChoices.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.routeChoice === selected.dataset.routePanel)));
  }
  if (routePanels.length) {
    document.querySelector('.entry-picker').hidden = false;
    document.querySelector('.learning-start').classList.add('is-interactive');
    showRoute(url.searchParams.get('learn'));
    routeChoices.forEach(button => button.addEventListener('click', () => {
      showRoute(button.dataset.routeChoice);
      const next = new URL(location.href);
      next.searchParams.set('learn', button.dataset.routeChoice);
      history.replaceState(null, '', next);
    }));
    window.addEventListener('popstate', () => showRoute(new URL(location.href).searchParams.get('learn')));
  }
  const routePager = [...document.querySelectorAll('[data-reading-route]')].find(nav => nav.dataset.readingRoute === url.searchParams.get('learn'));
  if (routePager) {
    routePager.hidden = false;
    document.querySelector('.lesson-pager').hidden = true;
    const context = document.createElement('p');
    context.className = 'active-reading-route';
    context.textContent = routePager.dataset.routeTitle + ' · ' + routePager.dataset.routePosition;
    document.querySelector('.lesson-header').prepend(context);
  }
  const saveGroups=[];
  document.querySelectorAll('.topic-group,.module-nav,.chapter-nav').forEach(group=>{
    const key=group.classList.contains('chapter-nav')?'ai-infra-chapter-'+group.dataset.chapter:group.classList.contains('module-nav')?'ai-infra-module-'+group.dataset.module:'ai-infra-topic-'+group.dataset.topic;
    const current=Boolean(group.querySelector('[aria-current="page"]'));
    if(current)group.open=true;
    else {
      try {const stored=sessionStorage.getItem(key);if(stored!==null)group.open=stored==='open';} catch {}
    }
    const saveGroup=()=>{
      try {sessionStorage.setItem(key,group.open?'open':'closed');} catch {}
    };
    // Also remember the current lesson's initially open ancestors.
    saveGroup();
    saveGroups.push(saveGroup);
    group.addEventListener('toggle',saveGroup);
  });
  const small=matchMedia('(max-width:900px)'),menu=document.querySelector('.sidebar-menu');
  const sidebar=document.querySelector('.site-sidebar');
  const sitePath=new URL(document.querySelector('.site-brand')?.href||'./',location.href).pathname;
  const scrollKey='ai-infra-sidebar-scroll:'+sitePath;
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
  function adaptMenu(){if(menu)menu.open=!small.matches;restoreSidebar();}
  small.addEventListener('change',adaptMenu);adaptMenu();
  sidebar?.addEventListener('scroll',saveSidebar,{passive:true});
  function saveNavigation(){saveGroups.forEach(save=>save());saveSidebar();}
  // Capture state before navigation, even before queued scroll/toggle events.
  sidebar?.addEventListener('click',saveNavigation,{capture:true});
  window.addEventListener('pagehide',saveNavigation);
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
