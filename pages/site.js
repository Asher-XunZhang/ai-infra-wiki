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
  document.querySelectorAll('.topic-group').forEach(group=>{
    const key='ai-infra-topic-'+group.dataset.topic;
    const current=Boolean(group.querySelector('[aria-current="page"]'));
    if(current)group.open=true;
    else {
      try {group.open=sessionStorage.getItem(key)==='open';} catch {}
    }
    group.addEventListener('toggle',()=>{
      try {sessionStorage.setItem(key,group.open?'open':'closed');} catch {}
    });
  });
  const small=matchMedia('(max-width:900px)'),menu=document.querySelector('.sidebar-menu');
  function adaptMenu(){if(menu)menu.open=!small.matches;}
  small.addEventListener('change',adaptMenu);adaptMenu();
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
})();
