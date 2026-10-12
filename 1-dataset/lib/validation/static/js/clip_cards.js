// Una tarjeta por clip del item seleccionado: se ve el video generado y se le asigna
// la clase correcta o se descarta. El recorte y el crop se deciden en 4_edit.py.

/* ===== Clips ===== */
function renderClips() {
  const g = $('clips-grid');
  g.innerHTML = '';
  if (!S.clips.length) {
    g.innerHTML = '<p style="color:var(--text-m);font-size:13px;">No hay clips.</p>';
    return;
  }
  S.clips.forEach(c => g.appendChild(buildCard(c)));
}

function cropText(crop) {
  return crop ? `crop ${crop.right - crop.left}×${crop.bottom - crop.top}` : 'sin crop';
}

function buildCard(c) {
  const div = document.createElement('div');
  div.className = 'clip-card' + (c.discarded ? ' discarded' : '');
  div.dataset.id = c.clip_id;
  if (c.discarded) div.style.opacity = '.55';

  const videoSrc = `/api/video/${encodeURIComponent(c.clip_id)}`;
  const vidInner = c.file_exists
    ? `<video preload="metadata" controls src="${videoSrc}"></video>`
    : `<div class="clip-missing"><span class="clip-missing-ico">⚠️</span><span>Sin generar: ejecuta 5_render.py</span></div>`;

  const classOptions = window.CLASSES.map(cls =>
    `<option value="${cls}"${cls === c['class'] ? ' selected' : ''}>${cls}</option>`).join('');

  div.innerHTML = `
    <div class="vid-wrap">${vidInner}</div>
    <div class="clip-foot">
      <div class="clip-meta">
        <span class="clip-id">${H(c.clip_id.slice(-8))}</span>
        <span class="clip-dur">${fmtSec(c.duration)} (${c.duration.toFixed(1)}s)</span>
        ${c.discarded ? '<span class="chip">descartado</span>' : ''}
      </div>
      <div class="rc-line">del crudo: ${c.start_sec.toFixed(1)}s → ${c.end_sec.toFixed(1)}s · ${H(cropText(c.crop))}</div>
      <div class="class-row">
        <select class="clip-class">${classOptions}</select>
        <button class="btn btn-secondary btn-sm clip-discard">${c.discarded ? '↩ Restaurar' : '🗑 Descartar'}</button>
      </div>
    </div>`;

  const sel = div.querySelector('.clip-class');
  sel.addEventListener('change', async () => {
    try {
      await api(`/api/clips/${encodeURIComponent(c.clip_id)}`, {
        method: 'PUT', body: JSON.stringify({ class: sel.value }),
      });
      toast(`✅ ${c.clip_id.slice(-8)} → ${sel.value}`, 'ok');
      await refreshAfterChange();
    } catch (e) {
      toast('Error al asignar la clase: ' + e.message, 'err');
      sel.value = c['class'];
    }
  });

  div.querySelector('.clip-discard').addEventListener('click', async () => {
    try {
      await api(`/api/clips/${encodeURIComponent(c.clip_id)}`, {
        method: 'PUT', body: JSON.stringify({ discarded: !c.discarded }),
      });
      toast(c.discarded ? `↩ Restaurado: ${c.clip_id.slice(-8)}` : `🗑 Descartado: ${c.clip_id.slice(-8)}`, 'ok');
      await refreshAfterChange();
    } catch (e) {
      toast('Error: ' + e.message, 'err');
    }
  });

  return div;
}
