// Filtros de la barra lateral, cambio de modo Raw/Clips y estadísticas del encabezado.

/* ===== Filtros ===== */
function filters() {
  const f = { q: $('q').value.trim() };
  if (S.mode === 'raw') {
    f.country = $('f-country').value;
  } else {
    f.class = $('f-class').value;
  }
  const minDur = $('f-min-dur').value.trim();
  const maxDur = $('f-max-dur').value.trim();
  if (minDur !== '') f.min_duration = minDur;
  if (maxDur !== '') f.max_duration = maxDur;
  return f;
}

/* ===== Cambio de modo ===== */
function setMode(mode) {
  if (S.mode === mode) return;
  S.mode = mode;
  $('mode-raw').classList.toggle('active', mode === 'raw');
  $('mode-clips').classList.toggle('active', mode === 'clips');
  $('fg-country').style.display = mode === 'raw' ? '' : 'none';
  $('fg-class').style.display   = mode === 'raw' ? 'none' : '';
  $('q').value = '';
  $('f-min-dur').value = '';
  $('f-max-dur').value = '';
  S.idx = -1;
  loadStats();
  loadItems();
  showState('empty');
}

/* ===== Estadísticas ===== */
function buildCountryFilterOptions(countries) {
  const sel = $('f-country');
  const current = sel.value;
  const names = Object.keys(countries).sort();
  sel.innerHTML = '<option value="all">Todos</option>' +
    names.map(c => `<option value="${H(c)}">${H(c)} (${countries[c]})</option>`).join('');
  if ([...sel.options].some(o => o.value === current)) sel.value = current;
}

async function loadStats() {
  try {
    const [raw, st] = await Promise.all([api('/api/raw-stats'), api('/api/stats')]);
    buildCountryFilterOptions(raw.countries);
    $('mode-badge-raw').textContent   = raw.total_raw;
    $('mode-badge-clips').textContent = st.total_clips;
    const classesHtml = Object.entries(st.classes).sort()
      .map(([c,n]) => `<span class="chip ${catCls(c)}">${H(c)} ${n}</span>`).join('');
    $('hdr-stats').innerHTML =
      `<span class="stat-chip">Σ ${raw.total_raw} videos crudos · ${raw.items_with_clips} con clips · ${st.total_clips} clips registrados</span>
       ${classesHtml}`;
  } catch(e) { console.warn('stats:', e); }
}
