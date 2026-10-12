// Geometría y dibujo de la superposición de la caja de crop sobre un <video>. La usan ambos sitios.

/* ===== Auxiliares de la superposición de crop (compartidos por ambos sitios) ===== */
function getVideoRenderedRect(video, canvas) {
  const cW = canvas.width  / (devicePixelRatio || 1);
  const cH = canvas.height / (devicePixelRatio || 1);
  const vW = video.videoWidth;
  const vH = video.videoHeight;
  if (!vW || !vH) return { x: 0, y: 0, w: cW, h: cH };
  const videoAspect  = vW / vH;
  const canvasAspect = cW / cH;
  let renderW, renderH, offsetX, offsetY;
  if (videoAspect > canvasAspect) {
    renderW = cW;  renderH = cW / videoAspect;
    offsetX = 0;   offsetY = (cH - renderH) / 2;
  } else {
    renderH = cH;  renderW = cH * videoAspect;
    offsetX = (cW - renderW) / 2; offsetY = 0;
  }
  return { x: offsetX, y: offsetY, w: renderW, h: renderH };
}

function drawCropOverlay(canvas, video, crop) {
  if (!video.videoWidth) return;
  const dpr = devicePixelRatio || 1;
  canvas.width  = canvas.offsetWidth  * dpr;
  canvas.height = canvas.offsetHeight * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const { x, y, w, h } = getVideoRenderedRect(video, canvas);
  const scaleX = w / video.videoWidth;
  const scaleY = h / video.videoHeight;

  const l  = x + crop.left   * scaleX;
  const t  = y + crop.top    * scaleY;
  const bw = (crop.right  - crop.left) * scaleX;
  const bh = (crop.bottom - crop.top)  * scaleY;

  const cW = canvas.offsetWidth;
  const cH = canvas.offsetHeight;

  ctx.fillStyle = 'rgba(0,0,0,0.52)';
  ctx.fillRect(0, 0, cW, cH);
  ctx.clearRect(l, t, bw, bh);

  ctx.strokeStyle = 'rgba(99,102,241,0.9)';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(l, t, bw, bh);

  const label = `${crop.right - crop.left}×${crop.bottom - crop.top}`;
  ctx.font = 'bold 9px system-ui';
  const tw = ctx.measureText(label).width;
  ctx.fillStyle = 'rgba(99,102,241,0.85)';
  ctx.fillRect(l, t, tw + 6, 14);
  ctx.fillStyle = '#fff';
  ctx.fillText(label, l + 3, t + 10);
}

function clearCropOverlay(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

