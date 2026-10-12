// Lista de items, selección de item, reclasificación por item y reclasificación en bloque de los items listados.

/* ===== Lista de items ===== */
async function fetchItems() {
  return api('/api/items?' + new URLSearchParams(filters()));
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

    const classBadges = it.classes.map(c => `<span class="chip ${catCls(c)}">${H(c)}</span>`).join('');
    const extra = [
      it.discarded ? `<span class="split-badge">🗑 ${it.discarded}</span>` : '',
      it.pending   ? `<span class="split-badge" title="Sin generar">⚠️ ${it.pending}</span>` : '',
    ].join('');
    const maxDurHtml = it.max_clip_duration != null
      ? `<span class="ir-sub">clip más largo: ${fmtSec(it.max_clip_duration)} (${it.max_clip_duration.toFixed(1)}s)</span>`
      : '';

    li.innerHTML = `
      <div class="ir-top">
        <span class="ir-id">${it.item_id.slice(0,10)}…</span>
        <span class="split-badge">${H(it.country || '')}</span>${extra}
      </div>
      <div class="ir-title">${it.clip_count} clip${it.clip_count!==1?'s':''}</div>
      ${maxDurHtml ? `<div class="ir-meta">${maxDurHtml}</div>` : ''}
      <div class="ir-meta">${classBadges}</div>`;
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

  const it = S.items[i];
  showState('loading');

  $('d-title').textContent = it.item_id;
  $('d-chips').innerHTML = `<span class="split-badge">${H(it.country || '')}</span>` +
    it.classes.map(c => `<span class="chip ${catCls(c)}">${H(c)}</span>`).join('');

  await reloadClips(it.item_id);
}

async function reloadClips(itemId) {
  try {
    S.clips = await api(`/api/items/${itemId}/clips`);
    renderClips();
    showState('detail');
  } catch(e) {
    toast('Error al cargar clips: ' + e.message, 'err');
    showState('empty');
  }
}

// Tras un cambio: recarga los clips del item abierto, las estadísticas y la lista
async function refreshAfterChange() {
  loadStats();
  const it = S.items[S.idx];
  if (it) await reloadClips(it.item_id);
  const stillListed = await refreshList();
  if (!stillListed) showState('empty');
}

/* ===== Reclasificar item ===== */
async function reclassifyItem() {
  if (S.idx < 0) return;
  const it = S.items[S.idx];
  const newClass = $('item-class').value;
  try {
    const res = await api(`/api/items/${it.item_id}/relabel`, {
      method: 'POST', body: JSON.stringify({ class: newClass }),
    });
    toast(`✅ ${res.updated} clip(s) reclasificados a ${newClass}`, 'ok');
    await refreshAfterChange();
  } catch(e) {
    toast('Error al reclasificar: ' + e.message, 'err');
  }
}

/* ===== Reclasificar en bloque todos los items listados (respeta los filtros activos) ===== */
async function bulkRelabelListed() {
  if (!S.items.length) { toast('No hay items listados para reclasificar', 'info'); return; }
  const newClass = $('bulk-class').value;
  const n = S.items.length;
  if (!confirm(`¿Reclasificar los clips anómalos de los ${n} items listados a "${newClass}"?\n\nEsto respeta los filtros activos (estado/clase/búsqueda/duración).`)) {
    return;
  }

  const btn = $('btn-bulk-relabel');
  btn.disabled = true;
  try {
    const res = await api('/api/items/bulk-relabel', {
      method: 'POST',
      body: JSON.stringify({ item_ids: S.items.map(it => it.item_id), class: newClass }),
    });
    let msg = `✅ ${res.clips_updated} clip(s) reclasificados en ${res.items_found}/${res.items_requested} items a ${newClass}`;
    if (res.items_missing.length) msg += ` — ${res.items_missing.length} no encontrados`;
    toast(msg, 'ok');
    loadStats();
    await loadItems();   // vuelve a filtrar: la mayoría de los items reetiquetados saldrán de una lista filtrada por clase
    showState('empty');
  } catch(e) {
    toast('Error al reclasificar en masa: ' + e.message, 'err');
  } finally {
    btn.disabled = false;
  }
}
