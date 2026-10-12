"""
lib/common/paths.py — rutas por defecto del pipeline y conversión de rutas a un
formato portable (relativo a 1-dataset/), para que los CSV se puedan mover de
máquina sin romperse.

    data/
    ├── metadata/            items.csv, clips.csv, split.csv   (el estado del pipeline)
    ├── 1_raw/               videos descargados      {país}/{item_id}.mp4
    ├── 2_clips/             clips generados         {clip_id}.mp4
    ├── external/            ucf-crime/, vioperu/    {clase}/{archivo}
    ├── 3_dataset/           dataset final           {fuente}/{split}/{clase}/ + manifest.csv
    └── logs/
"""
from pathlib import Path

# Raíz de 1-dataset/ (este archivo vive en 1-dataset/lib/common/)
DATASET_DIR  = Path(__file__).resolve().parents[2]
DATA_DIR     = DATASET_DIR / "data"
METADATA_DIR = DATA_DIR / "metadata"

ITEMS_CSV = METADATA_DIR / "items.csv"
CLIPS_CSV = METADATA_DIR / "clips.csv"
SPLIT_CSV = METADATA_DIR / "split.csv"

RAW_DIR      = DATA_DIR / "1_raw"
CLIPS_DIR    = DATA_DIR / "2_clips"
EXTERNAL_DIR = DATA_DIR / "external"
DATASET_OUT  = DATA_DIR / "3_dataset"


def to_portable(path) -> str:
    """Ruta como texto: relativa a 1-dataset/ (con '/') si está dentro, absoluta si no."""
    p = Path(path).resolve()
    try:
        return p.relative_to(DATASET_DIR).as_posix()
    except ValueError:
        return str(p)


def from_portable(text: str) -> Path:
    """Inversa de to_portable()."""
    p = Path(text)
    return p if p.is_absolute() else DATASET_DIR / p
