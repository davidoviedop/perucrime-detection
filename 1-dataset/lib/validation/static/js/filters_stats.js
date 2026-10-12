// Filtros de la barra lateral y estadísticas del encabezado.

/* ===== Filtros ===== */
function filters() {
  const f = { status: $('f-status').value, class: $('f-class').value, q: $('q').value.trim() };
  const minDur = $('f-min-dur').value.trim();
  const maxDur = $('f-max-dur').value.trim();
  if (minDur !== '') f.min_duration = minDur;
  if (maxDur !== '') f.max_duration = maxDur;
  return f;
}

/* ===== Estadísticas ===== */
async function loadStats() {
  try {
    const st = await api('/api/stats');
    const classesHtml = Object.entries(st.classes).sort()
      .map(([c,n]) => `<span class="chip ${catCls(c)}">${H(c)} ${n}</span>`).join('');
    const pendingHtml = st.pending
      ? `<span class="stat-chip" title="Ejecuta 5_render.py para generarlos">⚠️ ${st.pending} sin generar</span>` : '';
    const discHtml = st.discarded ? `<span class="stat-chip">🗑 ${st.discarded} descartados</span>` : '';
    $('hdr-stats').innerHTML = `
      <span class="stat-chip">Σ ${st.total_items} items · ${st.total_clips} clips activos</span>
      ${classesHtml}${discHtml}${pendingHtml}`;
  } catch(e) { console.warn('stats:', e); }
}
