// Editor de clip: video crudo + rango de recorte + caja de crop + clase. Sirve para
// registrar un clip nuevo o para editar uno ya registrado. No codifica nada:
// guarda los parámetros en clips.csv y el video lo genera después 5_render.py.

/* ===== Editor (un video crudo, recorte+crop siempre abiertos) ===== */
function renderEditor() {
  const host = $('clip-editor');
  host.innerHTML = '';
  const it = S.items[S.idx];
  $('editor-title').textContent = S.editing
    ? `Editando ${S.editing.clip_id}` : 'Nuevo clip de este video';
  if (!S.rawExists) {
    host.innerHTML = '<p style="color:var(--text-m);font-size:13px;">El video crudo de este item no está en la carpeta ' +
      '<code>--raw-dir</code>: se pueden ver y reclasificar sus clips, pero no editar su ventana ni su crop.</p>';
    return;
  }
  host.appendChild(buildClipEditor(it, S.editing));
}

function buildClipEditor(it, clip) {
  const div = document.createElement('div');
  div.className = 'clip-card raw-extractor-card';

  const videoSrc = `/api/raw-video/${encodeURIComponent(it.item_id)}`;
  const classOptions = window.CLASSES.map(cls => `<option value="${cls}">${cls}</option>`).join('');

  div.innerHTML = `
    <div class="vid-wrap">
      <video preload="metadata" src="${videoSrc}"></video>
      <canvas class="crop-canvas"></canvas>
    </div>
    <div class="vid-ctrl">
      <button class="vc-play" title="Reproducir/Pausar (clic en video)">▶</button>
      <span class="vc-time">0:00</span>
      <input class="vc-seek" type="range" min="0" value="0" step="0.01" />
      <span class="vc-dur">—:——</span>
    </div>
    <div class="clip-foot">
      <div class="edit-sec-lbl">Recorte (inicio / fin) — por defecto, el video completo</div>
      <div class="trim-track">
        <div class="trim-rail"></div>
        <div class="trim-range"></div>
        <input type="range" class="trim-start" min="0" max="1" step="0.05" value="0">
        <input type="range" class="trim-end"   min="0" max="1" step="0.05" value="1">
      </div>
      <div class="trim-readout"><span class="trim-start-val">0:00</span><span class="trim-end-val">0:00</span></div>
      <div class="edit-hint">Arrastra las esquinas sobre el video para recortar el encuadre (opcional — por defecto usa el cuadro completo). El video original nunca se modifica: "Guardar" solo registra los parámetros en clips.csv y el clip se genera después con 5_render.py. La reproducción se detiene al llegar al final del recorte.</div>
      <div class="class-row">
        <select class="extract-class">${classOptions}</select>
        <button class="btn btn-secondary btn-sm extract-reset">Restablecer</button>
        ${clip ? '<button class="btn btn-secondary btn-sm extract-cancel">Cancelar</button>' : ''}
        <button class="btn btn-primary btn-sm extract-apply">${clip ? '💾 Actualizar clip' : '💾 Guardar clip'}</button>
      </div>
    </div>`;

  const video  = div.querySelector('video');
  const canvas = div.querySelector('.crop-canvas');
  const wrap   = div.querySelector('.vid-wrap');

  attachVideoControls(div, video);

  const trimStartEl  = div.querySelector('.trim-start');
  const trimEndEl    = div.querySelector('.trim-end');
  const trimRangeEl  = div.querySelector('.trim-range');
  const trimStartVal = div.querySelector('.trim-start-val');
  const trimEndVal   = div.querySelector('.trim-end-val');
  const applyBtn     = div.querySelector('.extract-apply');
  const resetBtn     = div.querySelector('.extract-reset');
  const cancelBtn    = div.querySelector('.extract-cancel');
  const classSel     = div.querySelector('.extract-class');
  if (clip) classSel.value = clip['class'];

  function updateTrimReadout() {
    const s = +trimStartEl.value, e = +trimEndEl.value, dur = +trimEndEl.max || 1;
    trimRangeEl.style.left  = (s / dur * 100) + '%';
    trimRangeEl.style.width = Math.max(0, (e - s) / dur * 100) + '%';
    trimStartVal.textContent = `${fmtSec(s)} (${s.toFixed(1)}s)`;
    trimEndVal.textContent   = `${fmtSec(e)} (${e.toFixed(1)}s)`;
  }

  // Valores iniciales: el video completo, o los del clip que se edita
  function initTrimUI() {
    if (!video.duration || !isFinite(video.duration)) return;
    const dur = video.duration;
    trimStartEl.max = trimEndEl.max = dur.toFixed(3);
    trimStartEl.value = clip ? clip.start_sec : 0;
    trimEndEl.value   = clip ? Math.min(clip.end_sec, dur).toFixed(3) : dur.toFixed(3);
    updateTrimReadout();
    if (clip && video.seekable.length) video.currentTime = +trimStartEl.value;
  }

  trimStartEl.addEventListener('input', () => {
    if (+trimStartEl.value > +trimEndEl.value - 0.1) trimStartEl.value = Math.max(0, +trimEndEl.value - 0.1);
    updateTrimReadout();
    if (video.seekable.length) video.currentTime = +trimStartEl.value;
  });
  trimEndEl.addEventListener('input', () => {
    if (+trimEndEl.value < +trimStartEl.value + 0.1) trimEndEl.value = +trimStartEl.value + 0.1;
    updateTrimReadout();
    if (video.seekable.length) video.currentTime = +trimEndEl.value;
  });

  // Vista previa del clip: al llegar al final del recorte se detiene y vuelve al inicio
  video.addEventListener('timeupdate', () => {
    const end = +trimEndEl.value;
    if (!video.paused && end > 0 && video.currentTime >= end) {
      video.pause();
      video.currentTime = +trimStartEl.value;
    }
  });

  let cropDrag = null;
  function baseCrop() {
    return video.videoWidth ? { left: 0, top: 0, right: video.videoWidth, bottom: video.videoHeight } : null;
  }
  function initCropUI() {
    const base = baseCrop();
    if (!base) return;
    if (!cropDrag) cropDrag = attachCropDragLayer(wrap, video, canvas, () => {});
    cropDrag.setCrop(clip && clip.crop ? { ...clip.crop } : base);
  }

  if (video.readyState >= 1) { initTrimUI(); initCropUI(); }
  video.addEventListener('loadedmetadata', () => { initTrimUI(); initCropUI(); });

  resetBtn.addEventListener('click', () => { initTrimUI(); initCropUI(); });
  if (cancelBtn) cancelBtn.addEventListener('click', () => { S.editing = null; renderEditor(); });

  applyBtn.addEventListener('click', async () => {
    const payload = { class: classSel.value };
    const s = +trimStartEl.value, e = +trimEndEl.value;
    if (isFinite(s) && isFinite(e) && e > s) {
      payload.trim_in  = s;
      payload.trim_out = e;
    }
    const base = baseCrop();
    const crop = cropDrag ? cropDrag.getCrop() : null;
    const isFull = !crop || !base || (crop.left === base.left && crop.top === base.top &&
                                      crop.right === base.right && crop.bottom === base.bottom);
    if (!isFull) payload.crop = crop;
    else if (clip) payload.crop = null;          // al editar, el cuadro completo quita el crop guardado

    applyBtn.disabled = true;
    try {
      const saved = clip
        ? await api(`/api/clips/${encodeURIComponent(clip.clip_id)}`,
                    { method: 'PUT', body: JSON.stringify(payload) })
        : await api(`/api/raw-items/${encodeURIComponent(it.item_id)}/clips`,
                    { method: 'POST', body: JSON.stringify(payload) });
      toast(`✅ Clip ${clip ? 'actualizado' : 'guardado'}: ${saved.clip_id} (${saved['class']})`, 'ok');
      S.editing = null;
      await refreshItem();
      renderEditor();
    } catch (err) {
      toast('Error al guardar: ' + err.message, 'err');
    } finally {
      applyBtn.disabled = false;
    }
  });

  return div;
}
