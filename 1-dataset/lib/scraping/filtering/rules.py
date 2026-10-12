"""
Reglas de relevancia aplicadas a cada item descubierto una vez conocidos sus
metadatos: límites de duración, vistas mínimas, palabras de rechazo en el
título, una categoría por palabras clave (lib/common/label_rules.py) y un
enfoque opcional por clases (--classes).
"""
from lib.common.classes import target_class_of
from lib.common.label_rules import assign_label


# Rechazo directo: si CUALQUIERA aparece en el título → se rechaza de inmediato.
# Es propio del filtrado (no forma parte del vocabulario de categorías de
# lib/common/label_rules.py), por eso queda local a este módulo.
REJECT_TITLE_KEYWORDS = [
    "tráiler", "trailer", "teaser",
    "película", "pelicula", "telenovela", "novela", " serie ",
    "podcast", "karaoke",
    "videoclip", "clip oficial", "lyric video", "music video",
    "resumen semanal", "resumen mensual", "resumen anual",
    "best of", "top 10", "top10",
    "reacción", "reaccion",
    "capítulo", "capitulo", "episodio", "temporada",
    "entrevista exclusiva",
]


def evaluate(item: dict, meta: dict, cfg: dict) -> tuple[bool, str, str | None]:
    """Devuelve (aprobado, motivo, etiqueta). La etiqueta es None si se rechaza."""
    duration   = meta.get("duration")
    view_count = meta.get("view_count") or 0
    title      = (meta.get("raw_title") or item.get("raw_title") or "").strip()
    desc       = meta.get("raw_description") or ""

    min_dur  = cfg["min_duration_sec"]
    max_dur  = cfg["max_duration_sec"]
    min_view = cfg["min_view_count"]

    # 1. Duración
    if duration is not None:
        if duration < min_dur:
            return False, f"too_short:{duration}s", None
        if duration > max_dur:
            return False, f"too_long:{duration}s", None

    # 2. Vistas
    if min_view > 0 and view_count < min_view:
        return False, f"low_views:{view_count}", None

    # 3. Rechazo directo por título
    title_lc = title.lower()
    for kw in REJECT_TITLE_KEYWORDS:
        if kw in title_lc:
            return False, f"reject_kw:{kw.strip()}", None

    # 4. Etiqueta: se rechaza si no se pudo determinar
    label = assign_label(title, desc)
    if label is None:
        return False, "no_label", None

    # 5. Enfoque por clase: se rechazan los items fuera de las clases pedidas.
    # Quedan como 'rejected' pero con un motivo distinto, de modo que un
    # --reset posterior los re-encola para una pasada más amplia (o con otro
    # enfoque) sin volver a pedir los metadatos.
    target_classes = cfg.get("target_classes")
    if target_classes:
        tc = target_class_of(label)
        if tc not in target_classes:
            return False, f"class_filtered:{tc}", None

    return True, "ok", label
