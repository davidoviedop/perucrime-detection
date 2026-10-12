// Lista de los clips registrados del item seleccionado (filas de clips.csv):
// reclasificar, cargar en el editor para cambiar su recorte/crop, o eliminar.

/* ===== Clips registrados ===== */
function renderRecordedClips() {
  const g = $('clips-grid');
  g.innerHTML = '';
  $('reclass-bar').style.display = S.clips.length ? '' : 'none';
  if (!S.clips.length) {
    g.innerHTML = '<p style="color:var(--text-m);font-size:13px;">Aún no hay clips registrados de este video.</p>';
    return;
  }
  S.clips.forEach(c => g.appendChild(buildClipCard(c)));
}

function cropText(crop) {
  if (!crop) return 'sin crop';
  return `crop ${crop.right - crop.left}×${crop.bottom - crop.top} en (${crop.left}, ${crop.top})`;
}

function buildClipCard(c) {
  const div = document.createElement('div');
  div.className = 'clip-card rc-card' + (S.editing && S.editing.clip_id === c.clip_id ? ' editing' : '');
  div.dataset.id = c.clip_id;

  const classOptions = window.CLASSES.map(cls => `<option value="${cls}">${cls}</option>`).join('');
  const statusChip = c.discarded
    ? '<span class="chip">descartado</span>'
    : (c.rendered ? '<span class="chip cat-normal">generado</span>' : '<span class="chip">pendiente de generar</span>');

  div.innerHTML = `
    <div class="clip-foot">
      <div class="clip-meta">
        <span class="clip-id">${H(c.clip_id.slice(-8))}</span>
        <span class="chip ${catCls(c['class'])}">${H(c['class'])}</span>
        ${statusChip}
      </div>
      <div class="rc-line">${fmtSec(c.start_sec)} → ${fmtSec(c.end_sec)} (${c.duration.toFixed(1)}s) · ${H(cropText(c.crop))}</div>
      <div class="class-row">
        <select class="rc-class">${classOptions}</select>
        <button class="btn btn-secondary btn-sm rc-edit" ${S.rawExists ? '' : 'disabled'}>✏️ Editar</button>
        <button class="btn btn-secondary btn-sm rc-del" title="Eliminar clip">🗑</button>
      </div>
    </div>`;

  const sel = div.querySelector('.rc-class');
  sel.value = c['class'];
  sel.addEventListener('change', async () => {
    try {
      await api(`/api/clips/${encodeURIComponent(c.clip_id)}`, {
        method: 'PUT', body: JSON.stringify({ class: sel.value }),
      });
      toast(`✅ ${c.clip_id} → ${sel.value}`, 'ok');
      await refreshItem();
    } catch (e) {
      toast('Error al reclasificar: ' + e.message, 'err');
      sel.value = c['class'];
    }
  });

  div.querySelector('.rc-edit').addEventListener('click', () => {
    S.editing = c;
    renderEditor();
    renderRecordedClips();
    $('clip-editor').scrollIntoView({ block: 'nearest' });
  });

  div.querySelector('.rc-del').addEventListener('click', async () => {
    if (!confirm(`¿Eliminar el clip ${c.clip_id}? Se borra su fila de clips.csv (el video crudo no se toca).`)) return;
    try {
      await api(`/api/clips/${encodeURIComponent(c.clip_id)}`, { method: 'DELETE' });
      toast(`🗑 Clip eliminado: ${c.clip_id}`, 'ok');
      if (S.editing && S.editing.clip_id === c.clip_id) { S.editing = null; renderEditor(); }
      await refreshItem();
    } catch (e) {
      toast('Error al eliminar: ' + e.message, 'err');
    }
  });

  return div;
}
