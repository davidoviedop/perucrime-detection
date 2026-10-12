// Estado global de la interfaz del sitio de edición.

/* ===== Estado ===== */
const S = {
  mode: 'raw',        // 'raw' = todos los videos crudos | 'clips' = solo los que ya tienen clips registrados
  items: [],           // lista de items del modo actual
  idx: -1,
  clips: [],            // clips registrados (filas de clips.csv) del item seleccionado
  rawExists: true,      // si el video crudo del item seleccionado está en --raw-dir
  editing: null,        // clip que se está editando en el editor (null = clip nuevo)
};
