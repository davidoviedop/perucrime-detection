"""
Duraciones de video memorizadas.

No es un manifiesto: las entradas se indexan por (ruta, mtime, tamaño), así que
cualquier edición que reescriba un archivo (recorte) invalida su propia entrada
en la siguiente consulta y nunca se sirve un dato desactualizado. Existe porque
consultar con ffprobe cada clip en cada listado (para el filtro de duración)
volvería a correr ffprobe sobre todo el dataset en cada tecla.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .media import video_duration

_duration_cache: dict[str, tuple[float, int, float]] = {}


def cached_durations(paths: list[Path]) -> dict[Path, float]:
    """Duraciones de muchas rutas a la vez; solo los fallos de caché llaman a ffprobe (en paralelo)."""
    result: dict[Path, float] = {}
    misses: list[tuple[Path, os.stat_result]] = []
    for p in paths:
        try:
            st = p.stat()
        except OSError:
            result[p] = 0.0
            continue
        cached = _duration_cache.get(str(p))
        if cached and cached[0] == st.st_mtime and cached[1] == st.st_size:
            result[p] = cached[2]
        else:
            misses.append((p, st))

    if misses:
        with ThreadPoolExecutor(max_workers=min(16, len(misses))) as ex:
            durations = list(ex.map(lambda ps: video_duration(ps[0]), misses))
        for (p, st), dur in zip(misses, durations):
            _duration_cache[str(p)] = (st.st_mtime, st.st_size, dur)
            result[p] = dur

    return result
