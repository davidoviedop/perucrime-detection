"""
Arma el pool de items a dividir, cada uno con su cantidad de ventanas por clase.

Dos orígenes:
  scrapped             las filas ACTIVAS de clips.csv (no descartadas, clase distinta de
                       Others). Sus ventanas salen de n_frames, medido por 5_render.py al
                       generar cada video, así que no hace falta abrir los videos.
  ucf-crime / vioperu  carpetas sueltas data/external/{fuente}/{clase}/*.mp4, cuyos
                       fotogramas se cuentan con OpenCV.

Identidad del item, para que los clips de un mismo video fuente nunca caigan en
splits distintos:
  scrapped:            el item_id (varios clips pueden salir del mismo video fuente)
  ucf-crime / vioperu: el nombre de archivo sin extensión: un archivo ES un video
                       fuente independiente, sin sub-clips
"""
from collections import defaultdict
from pathlib import Path

import cv2

from lib.common.clips import ClipStore, clip_file

from .config import (EXCLUDE, FOLD, LOCAL_SOURCE, MAX_WINDOWS_PER_VIDEO, SPAN, STRIDE, VIDEO_EXTS)


class UnrenderedClipsError(RuntimeError):
    """Hay clips activos cuyo video aún no se generó (falta ejecutar 5_render.py)."""


def target_class(raw_class_name):
    """Clase canónica para el nombre de una carpeta/clase, o None para el cajón excluido."""
    if raw_class_name in EXCLUDE:
        return None
    return FOLD.get(raw_class_name, raw_class_name)


def item_key(source, path):
    """Clave de item de un archivo del dataset final o de una carpeta externa."""
    if source == LOCAL_SOURCE:
        return f"{source}:{path.stem.split('_clip_')[0]}"
    return f"{source}:{path.stem}"


def windows_from_frames(n_frames: int) -> int:
    """Cantidad de ventanas de entrenamiento de un clip con n_frames fotogramas (misma fórmula del data loader)."""
    if n_frames < SPAN:
        return 0
    return min((n_frames - SPAN) // STRIDE + 1, MAX_WINDOWS_PER_VIDEO)


def n_windows(path):
    """Ventanas de un video, contando sus fotogramas con OpenCV."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return windows_from_frames(n)


def active_clip_rows(clips: ClipStore) -> list[dict]:
    """Filas de clips.csv que entran al dataset: no descartadas y de una clase de entrenamiento."""
    return [r for r in clips.all() if r["discarded"] != "1" and target_class(r["class"]) is not None]


def scan_pool(clips: ClipStore, clips_dir: Path, external_dir: Path, sources):
    """item_key -> clase -> {'windows': int, 'files': [(fuente, clase_original, ruta), ...]}"""
    items = defaultdict(lambda: defaultdict(lambda: {"windows": 0, "files": []}))

    if LOCAL_SOURCE in sources:
        rows = active_clip_rows(clips)
        unrendered = [r["clip_id"] for r in rows if r["status"] != "rendered" or not r["n_frames"]]
        if unrendered:
            raise UnrenderedClipsError(
                f"{len(unrendered)} clips activos sin generar (p. ej. {unrendered[0]}): "
                "ejecuta 5_render.py antes de dividir")
        for r in rows:
            cls = target_class(r["class"])
            path = clip_file(clips_dir, r["clip_id"])
            entry = items[f"{LOCAL_SOURCE}:{r['item_id']}"][cls]
            entry["windows"] += windows_from_frames(int(r["n_frames"]))
            entry["files"].append((LOCAL_SOURCE, r["class"], path))

    for source in sources:
        if source == LOCAL_SOURCE:
            continue
        src_path = external_dir / source
        if not src_path.is_dir():
            continue
        for class_dir in sorted(src_path.iterdir()):
            if not class_dir.is_dir():
                continue
            tgt = target_class(class_dir.name)
            if tgt is None:
                continue
            for f in sorted(class_dir.iterdir()):
                if not (f.is_file() and f.suffix.lower() in VIDEO_EXTS):
                    continue
                entry = items[item_key(source, f)][tgt]
                entry["windows"] += n_windows(f)
                entry["files"].append((source, class_dir.name, f))
    return items
