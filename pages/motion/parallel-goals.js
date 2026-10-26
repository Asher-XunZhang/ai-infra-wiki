/* V33: filter the static goal panels. No animation or autoplay is started here.
   The generated markup remains fully readable when this file does not run. */
(() => {
  'use strict';
  const root = document.querySelector('[data-parallel-goals]');
  if (!root) return;
  const buttons = [...root.querySelectorAll('[data-goal]')];
  const panels = [...root.querySelectorAll('[data-goal-panel]')];
  const empty = root.querySelector('[data-goal-empty]');
  const status = root.querySelector('[data-goal-status]');
  const valid = new Set(buttons.map(button => button.dataset.goal));
  const labels = new Map(buttons.map(button => {
    const title = button.querySelector('strong');
    return [button.dataset.goal, button.dataset.label || (title ? title.textContent.trim() : button.dataset.goal)];
  }));
  let active = '';

  function render(goal, options = {}) {
    const { announce = true, updateUrl = true } = options;
    active = valid.has(goal) ? goal : '';
    for (const button of buttons) {
      button.setAttribute('aria-pressed', String(button.dataset.goal === active));
    }
    for (const panel of panels) {
      panel.hidden = panel.dataset.goalPanel !== active;
    }
    if (empty) empty.hidden = Boolean(active);
    if (status) {
      status.textContent = active
        ? `已展开：${labels.get(active)}。下面只显示相关策略；其他目标仍可切换。`
        : '先选一个目标，只展开相关策略与代价；未选择时不会自动播放动画。';
    }
    root.dataset.activeGoal = active;
    if (updateUrl) {
      const url = new URL(location.href);
      if (active) url.searchParams.set('goal', active);
      else url.searchParams.delete('goal');
      history.replaceState(null, '', url);
    }
    if (announce && active) {
      /* Keep the user's scroll position; the status line is the announcement. */
      root.dispatchEvent(new CustomEvent('parallel-goal-change', { detail: { goal: active } }));
    }
  }

  for (const button of buttons) {
    button.addEventListener('click', () => render(button.dataset.goal));
  }
  root.addEventListener('keydown', event => {
    if (event.key !== 'Escape' || !active) return;
    render('');
    buttons[0]?.focus();
  });

  const requested = new URL(location.href).searchParams.get('goal');
  root.dataset.enhanced = 'true';
  render(valid.has(requested) ? requested : '', { announce: false, updateUrl: false });
})();
