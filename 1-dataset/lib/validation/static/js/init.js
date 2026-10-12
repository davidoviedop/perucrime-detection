// Atajos de teclado e inicialización de la página. Debe cargarse al final. window.CLASSES lo define la plantilla en línea.

/* ===== Teclado ===== */
document.addEventListener('keydown', e => {
  const tag = e.target.tagName;
  if (tag==='INPUT'||tag==='TEXTAREA'||tag==='SELECT') {
    if (e.key==='Escape') e.target.blur();
    return;
  }
  switch(e.key) {
    case 'ArrowDown': case 'j': e.preventDefault(); selectItem(S.idx+1); break;
    case 'ArrowUp':   case 'k': e.preventDefault(); selectItem(S.idx-1); break;
    case 't': case 'T': toggleTheme(); break;
    case '?': showSC($('sc-modal').style.display === 'none'); break;
    case 'Escape': showSC(false); break;
  }
});

/* ===== Inicio ===== */
(function init() {
  const saved = localStorage.getItem('cctv-postedit-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  $('btn-theme').textContent = saved === 'dark' ? '☀️' : '🌙';

  ['f-status','f-class'].forEach(id => $(id).addEventListener('change', loadItems));
  let debounce;
  $('q').addEventListener('input', () => { clearTimeout(debounce); debounce = setTimeout(loadItems, 280); });
  // Debounce más largo para los filtros de duración, para no consultar en cada tecla.
  let durDebounce;
  ['f-min-dur','f-max-dur'].forEach(id => $(id).addEventListener('input', () => {
    clearTimeout(durDebounce); durDebounce = setTimeout(loadItems, 500);
  }));

  $('btn-reclass').addEventListener('click', reclassifyItem);
  $('btn-bulk-relabel').addEventListener('click', bulkRelabelListed);
  $('btn-theme').addEventListener('click', toggleTheme);
  $('btn-shortcuts').addEventListener('click', () => showSC(true));
  $('close-sc').addEventListener('click', () => showSC(false));
  $('sc-modal').addEventListener('click', e => { if(e.target===$('sc-modal')) showSC(false); });

  (async () => { await loadStats(); loadItems(); })();
})();
