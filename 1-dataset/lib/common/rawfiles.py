"""
Escaneo de la carpeta de videos crudos. Nada se cachea, así que el listado nunca
queda desactualizado. Las decisiones de edición (clips) no viven en el disco sino
en data/metadata/clips.csv (ver lib/common/clips.py).

Estructura de la carpeta cruda:  {país}/{item_id}.{ext}   (lo que escribe 3_download.py)
"""
from pathlib import Path

from lib.webkit.media import VIDEO_EXTS


def iter_raw_files(raw_dir: Path):
    """Produce (país, item_id, ruta) por cada video crudo en raw_dir/{país}/{item_id}.{ext}."""
    if not raw_dir.is_dir():
        return
    for country_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for f in sorted(country_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
                yield country_dir.name, f.stem, f


def find_raw_file(raw_dir: Path, item_id: str) -> Path | None:
    """Ubica un video crudo solo por item_id (nombre sin extensión == item_id),
    buscando de forma recursiva para que quien llama no necesite saber el país."""
    for p in raw_dir.rglob(f"{item_id}.*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            return p
    return None
