"""
lib/common/schemas.py — columnas de los CSV del pipeline (los "metadatos" del dataset).

Todos los CSV son UTF-8 con cabecera y una fila por registro; los valores vacíos
son celdas vacías. Viven en data/metadata/.

items.csv — una fila por VIDEO FUENTE descubierto (pasos 1-3; ver lib/scraping/store.py)
    id                 sha256(url)[:16]; el nombre del video crudo es {id}.mp4
    url, source, country, raw_title, raw_description, channel, upload_date
    duration, view_count        segundos / vistas, al filtrar
    download_path              ruta (relativa a 1-dataset/) del video crudo
    provisional_label          categoría UCF-Crime asignada por palabras clave
    ucf_category, filter_reason
    status                     discovered | approved | rejected | downloaded | failed
    created_at, updated_at

clips.csv — una fila por CLIP decidido sobre un video crudo (pasos 4-6; ver lib/common/clips.py)
    clip_id            {item_id}_clip_NNN; es también el nombre del video generado
    item_id            video fuente de donde sale (la unidad de la división: sin fugas)
    country, raw_file  de dónde se corta ("PE/abc….mp4", relativo a la carpeta cruda)
    class, label       clase de entrenamiento y Normal/Anomaly
    start_sec, end_sec ventana de recorte en la línea de tiempo del video CRUDO
    crop_*             caja de crop en píxeles del video crudo (vacía = fotograma completo)
    status             pending (falta generar o cambió) | rendered (el video generado está al día)
    discarded          1 si se descartó en la validación (no se divide ni se exporta)
    render_key         huella de los parámetros con los que se generó el video
    n_frames, fps, width, height, duration     medidos en el video generado (5_render.py)
    source_url, title  copiados de items.csv para que el CSV se entienda solo
    notes, created_at, updated_at

split.csv — una fila por ITEM con su split (paso 7; ver lib/splitting/)
    source, item_id, split, n_clips, windows
"""

ITEM_COLUMNS = [
    "id", "url", "source", "country",
    "raw_title", "raw_description", "channel", "upload_date",
    "duration", "view_count",
    "download_path",
    "provisional_label", "ucf_category", "filter_reason",
    "status", "created_at", "updated_at",
]

CLIP_COLUMNS = [
    "clip_id", "item_id", "country", "raw_file", "class", "label",
    "start_sec", "end_sec", "crop_left", "crop_top", "crop_right", "crop_bottom",
    "status", "discarded", "render_key",
    "n_frames", "fps", "width", "height", "duration",
    "source_url", "title", "notes", "created_at", "updated_at",
]

SPLIT_COLUMNS = ["source", "item_id", "split", "n_clips", "windows"]
