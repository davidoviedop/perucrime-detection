"""
Generación de los videos de clip a partir de las filas de clips.csv.

Cada clip se corta del video CRUDO original con una sola recodificación (recorte
temporal + crop), así que un clip nunca es la recodificación de otro clip. El
resultado es data/2_clips/{clip_id}.mp4, un archivo derivado que se puede borrar y
regenerar cuando se quiera.
"""
from pathlib import Path

from lib.common.clips import RENDERED, ClipStore, clip_file, compute_render_key, crop_of
from lib.common.rawfiles import find_raw_file
from lib.webkit.media import extract_clip

from .probe import probe_video


def find_raw_for(row: dict, raw_dir: Path) -> Path | None:
    """El video crudo de una fila: primero por su raw_file, si no por su item_id."""
    direct = raw_dir / row["raw_file"]
    if row["raw_file"] and direct.is_file():
        return direct
    return find_raw_file(raw_dir, row["item_id"])


def render_reason(row: dict, clips_dir: Path) -> str | None:
    """Por qué hay que generar este clip, o None si su video está al día."""
    if row["status"] != RENDERED:
        return "pendiente"
    if row["render_key"] != compute_render_key(row):
        return "parámetros cambiados"
    if not clip_file(clips_dir, row["clip_id"]).is_file():
        return "archivo faltante"
    return None


def render_clip(row: dict, raw_dir: Path, clips_dir: Path, store: ClipStore) -> tuple[bool, str]:
    """Genera el video de un clip y anota sus medidas en clips.csv. Devuelve (éxito, detalle)."""
    src = find_raw_for(row, raw_dir)
    if src is None:
        return False, f"no se encuentra el video crudo de {row['item_id']}"
    key = compute_render_key(row)
    out = clip_file(clips_dir, row["clip_id"])
    clips_dir.mkdir(parents=True, exist_ok=True)
    if not extract_clip(src, out, float(row["start_sec"]), float(row["end_sec"]), crop_of(row)):
        return False, "ffmpeg falló"
    meta = probe_video(out)
    if not meta:
        return False, "no se pudo leer el video generado"
    if not store.mark_rendered(row["clip_id"], key, meta):
        return False, "los parámetros cambiaron durante la generación; queda pendiente"
    return True, f"{meta['width']}x{meta['height']}, {meta['n_frames']} fotogramas, {meta['duration']:.1f}s"
