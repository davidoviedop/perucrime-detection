// Utilidades compartidas por los sitios de edición y validación: auxiliares de DOM/escape,
// el wrapper de fetch, estado de paneles, toasts, cambio de tema y modal de atajos.
// Cargar antes de los scripts propios de cada sitio.

/* ===== Auxiliares de DOM ===== */
const $  = id  => document.getElementById(id);
const H  = s   => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');

function catCls(cat) {
  if (!cat) return '';
  return 'cat-' + cat.toLowerCase().replace(/[^a-z]/g, '_');
}

function fmtSec(s) {
  if (s == null) return null;
  s = Math.round(s);
  return `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;
}

/* ===== API ===== */
async function api(url, opts={}) {
  const r = await fetch(url, { headers:{'Content-Type':'application/json'}, ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
  return body;
}

/* ===== Estado de los paneles ===== */
function showState(which) {
  $('st-empty').style.display   = which==='empty'   ? '' : 'none';
  $('st-loading').style.display = which==='loading' ? '' : 'none';
  $('st-detail').style.display  = which==='detail'  ? '' : 'none';
}

/* ===== Toasts ===== */
function toast(msg, type='info') {
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  $('toasts').appendChild(t);
  setTimeout(() => t.remove(), 3600);
}

/* ===== Tema ===== */
function toggleTheme() {
  const html = document.documentElement;
  const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  $('btn-theme').textContent = next === 'dark' ? '☀️' : '🌙';
  localStorage.setItem('cctv-postedit-theme', next);
}

/* ===== Modal de atajos ===== */
function showSC(v) { $('sc-modal').style.display = v ? '' : 'none'; }

