// Lista de items, selección de item y reclasificación a nivel de item.

/* ===== Lista de items ===== */
async function fetchItems() {
  const p = new URLSearchParams(filters());
  return api((S.mode === 'raw' ? '/api/raw-items?' : '/api/items?') + p);
}

async function loadItems() {
  S.items = await fetchItems();
  S.idx = -1;
  renderList();
  showState('empty');
}

// Vuelve a pedir la lista conservando el item seleccionado (si sigue en la lista)
async function refreshList() {
  const current = S.idx >= 0 ? S.items[S.idx].item_id : null;
  S.items = await fetchItems();
  S.idx = current ? S.items.findIndex(it => it.item_id === current) : -1;
  renderList();
  syncActive();
  return S.idx >= 0;
}

function renderList() {
  const ul = $('item-list');
  $('list-badge').textContent = S.items.length;
  ul.innerHTML = '';
  if (!S.items.length) {
    ul.innerHTML = '<li class="no-items">Sin resultados</li>';
    return;
  }
  S.items.forEach((it, i) => {
    const li = document.createElement('li');
    li.className = `item-row${i === S.idx ? ' active' : ''}`;
    li.setAttribute('role', 'option');
    li.setAttribute('aria-selected', i === S.idx);
    li.dataset.i = i;

    if (S.mode === 'raw') {
      const durText = it.duration ? `${fmtSec(it.duration)} (${it.duration.toFixed(1)}s)` : '—';
      const doneBadge = it.clip_count > 0
        ? `<span class="chip cat-normal">✓ ${it.clip_count} clip${it.clip_count!==1?'s':''}</span>`
        : `<span class="split-badge">sin clips</span>`;
      li.innerHTML = `
        <div class="ir-top">
          <span class="ir-id">${it.item_id.slice(0,10)}…</span>
          <span class="split-badge">${H(it.country)}</span>
        </div>
        <div class="ir-title">${H(durText)}</div>
        <div class="ir-meta">${doneBadge}</div>`;
    } else {
      const classBadges = it.classes.map(c => `<span class="chip ${catCls(c)}">${H(c)}</span>`).join('');
      const maxDurHtml = it.max_clip_duration != null
        ? `<span class="ir-sub">clip más largo: ${fmtSec(it.max_clip_duration)} (${it.max_clip_duration.toFixed(1)}s)</span>`
        : '';
      li.innerHTML = `
        <div class="ir-top">
          <span class="ir-id">${it.item_id.slice(0,10)}…</span>
          <span class="split-badge">${H(it.country || '')}</span>
        </div>
        <div class="ir-title">${it.clip_count} clip${it.clip_count!==1?'s':''}</div>
        ${maxDurHtml ? `<div class="ir-meta">${maxDurHtml}</div>` : ''}
        <div class="ir-meta">${classBadges}</div>`;
    }
    li.addEventListener('click', () => selectItem(i));
    ul.appendChild(li);
  });
}

function syncActive() {
  document.querySelectorAll('.item-row').forEach((el, i) => {
    const on = i === S.idx;
    el.classList.toggle('active', on);
    el.setAttribute('aria-selected', on);
    if (on) el.scrollIntoView({ block:'nearest' });
  });
}

/* ===== Seleccionar item ===== */
async function selectItem(i) {
  if (i < 0 || i >= S.items.length) return;
  S.idx = i;
  syncActive();
  showState('loading');
  const it = S.items[i];

  $('d-title').textContent = it.item_id;
  const durChip = it.duration ? `<span class="chip">${fmtSec(it.duration)} (${it.duration.toFixed(1)}s)</span>` : '';
  $('d-chips').innerHTML = `<span class="split-badge">${H(it.country || '')}</span>${durChip}`;

  try {
    const info = await api(`/api/items/${encodeURIComponent(it.item_id)}/clips`);
    S.clips = info.clips;
    S.rawExists = info.raw_exists;
    S.editing = null;
    renderEditor();
    renderRecordedClips();
    showState('detail');
  } catch(e) {
    toast('Error al cargar el item: ' + e.message, 'err');
    showState('empty');
  }
}

// Recarga los clips del item seleccionado y refresca lista y estadísticas
async function refreshItem() {
  const it = S.items[S.idx];
  if (!it) return;
  const info = await api(`/api/items/${encodeURIComponent(it.item_id)}/clips`);
  S.clips = info.clips;
  S.rawExists = info.raw_exists;
  renderRecordedClips();
  loadStats();
  const stillListed = await refreshList();
  if (!stillListed) showState('empty');
}

/* ===== Reclasificar item ===== */
async function reclassifyItem() {
  if (S.idx < 0) return;
  const it = S.items[S.idx];
  const newClass = $('item-class').value;
  try {
    const res = await api(`/api/items/${encodeURIComponent(it.item_id)}/relabel`, {
      method: 'POST', body: JSON.stringify({ class: newClass }),
    });
    toast(`✅ ${res.updated} clip(s) reclasificados a ${newClass}`, 'ok');
    await refreshItem();
  } catch(e) {
    toast('Error al reclasificar: ' + e.message, 'err');
  }
}
