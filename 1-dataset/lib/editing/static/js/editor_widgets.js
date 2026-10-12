// Capa arrastrable de la caja de crop y controles de video personalizados, usados por el extractor Raw y por las tarjetas de clips.

/* ===== Capa arrastrable de la caja de crop (compartida por ambos editores) =====
   Construye sobre `wrap` la superposición con asas para mover/redimensionar y
   llama a onChange(crop) en cada actualización del arrastre. Devuelve
   { setCrop(crop), destroy() }. */
function attachCropDragLayer(wrap, video, canvas, onChange) {
  const layer = document.createElement('div');
  layer.className = 'crop-edit-layer';
  layer.innerHTML = `
    <div class="crop-move"></div>
    <div class="crop-handle tl"></div>
    <div class="crop-handle tr"></div>
    <div class="crop-handle bl"></div>
    <div class="crop-handle br"></div>`;
  wrap.appendChild(layer);

  let crop = null;
  let dragging = null;

  function position() {
    if (!crop || !video.videoWidth) return;
    drawCropOverlay(canvas, video, crop);
    const { x, y, w, h } = getVideoRenderedRect(video, canvas);
    const sx = w / video.videoWidth, sy = h / video.videoHeight;
    const l = x + crop.left * sx,  t = y + crop.top    * sy;
    const r = x + crop.right * sx, b = y + crop.bottom * sy;

    const move = layer.querySelector('.crop-move');
    move.style.left = l + 'px'; move.style.top = t + 'px';
    move.style.width  = Math.max(0, r - l) + 'px';
    move.style.height = Math.max(0, b - t) + 'px';

    layer.querySelector('.tl').style.left = l + 'px'; layer.querySelector('.tl').style.top = t + 'px';
    layer.querySelector('.tr').style.left = r + 'px'; layer.querySelector('.tr').style.top = t + 'px';
    layer.querySelector('.bl').style.left = l + 'px'; layer.querySelector('.bl').style.top = b + 'px';
    layer.querySelector('.br').style.left = r + 'px'; layer.querySelector('.br').style.top = b + 'px';
  }

  function startDrag(e, mode) {
    if (!crop) return;
    e.preventDefault(); e.stopPropagation();
    dragging = { mode, startX: e.clientX, startY: e.clientY, orig: { ...crop } };
    document.addEventListener('mousemove', onDrag);
    document.addEventListener('mouseup', endDrag);
  }

  function onDrag(e) {
    if (!dragging) return;
    const { w, h } = getVideoRenderedRect(video, canvas);
    const sx = video.videoWidth / w, sy = video.videoHeight / h;
    const dx = (e.clientX - dragging.startX) * sx;
    const dy = (e.clientY - dragging.startY) * sy;
    const o  = dragging.orig;
    const vw = video.videoWidth, vh = video.videoHeight, minSize = 16;
    let { left, top, right, bottom } = o;

    if (dragging.mode === 'move') {
      const bw = o.right - o.left, bh = o.bottom - o.top;
      left  = Math.min(Math.max(0, o.left + dx), vw - bw);
      top   = Math.min(Math.max(0, o.top  + dy), vh - bh);
      right = left + bw; bottom = top + bh;
    } else {
      if (dragging.mode.includes('l')) left   = Math.min(Math.max(0, o.left + dx), o.right - minSize);
      if (dragging.mode.includes('r')) right  = Math.max(Math.min(vw, o.right + dx), o.left + minSize);
      if (dragging.mode.includes('t')) top    = Math.min(Math.max(0, o.top + dy), o.bottom - minSize);
      if (dragging.mode.includes('b')) bottom = Math.max(Math.min(vh, o.bottom + dy), o.top + minSize);
    }
    crop = {
      left: Math.round(left), top: Math.round(top),
      right: Math.round(right), bottom: Math.round(bottom),
    };
    position();
    onChange && onChange(crop);
  }

  function endDrag() {
    dragging = null;
    document.removeEventListener('mousemove', onDrag);
    document.removeEventListener('mouseup', endDrag);
  }

  layer.querySelector('.crop-move').addEventListener('mousedown', e => startDrag(e, 'move'));
  layer.querySelector('.tl').addEventListener('mousedown', e => startDrag(e, 'tl'));
  layer.querySelector('.tr').addEventListener('mousedown', e => startDrag(e, 'tr'));
  layer.querySelector('.bl').addEventListener('mousedown', e => startDrag(e, 'bl'));
  layer.querySelector('.br').addEventListener('mousedown', e => startDrag(e, 'br'));

  if (window.ResizeObserver) {
    new ResizeObserver(() => position()).observe(canvas);
  }

  return {
    setCrop(c) { crop = c; position(); },
    getCrop()  { return crop; },
    destroy()  { layer.remove(); },
  };
}

/* ===== Controles de video personalizados (compartidos por ambos editores) ===== */
function attachVideoControls(root, video) {
  const playBtn = root.querySelector('.vc-play');
  const timeEl  = root.querySelector('.vc-time');
  const durEl   = root.querySelector('.vc-dur');
  const seek    = root.querySelector('.vc-seek');
  let seeking = false;

  video.addEventListener('click',  () => video.paused ? video.play() : video.pause());
  video.addEventListener('play',   () => { playBtn.textContent = '⏸'; });
  video.addEventListener('pause',  () => { playBtn.textContent = '▶'; });
  video.addEventListener('ended',  () => { playBtn.textContent = '▶'; });
  video.addEventListener('loadedmetadata', () => {
    if (isFinite(video.duration)) {
      seek.max = video.duration;
      durEl.textContent = fmtSec(Math.round(video.duration));
    }
  });
  video.addEventListener('timeupdate', () => {
    if (!seeking) seek.value = video.currentTime;
    timeEl.textContent = fmtSec(Math.round(video.currentTime));
  });
  playBtn.addEventListener('click', () => video.paused ? video.play() : video.pause());
  seek.addEventListener('mousedown', () => { seeking = true; });
  seek.addEventListener('mouseup',   () => { seeking = false; video.currentTime = +seek.value; });
  seek.addEventListener('input',     () => { if (seeking) video.currentTime = +seek.value; });
}

