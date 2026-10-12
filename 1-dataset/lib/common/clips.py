"""
lib/common/clips.py — almacén de clips (data/metadata/clips.csv).

El CSV es la fuente de verdad de las decisiones de edición: de qué video crudo
sale cada clip, en qué ventana de tiempo, con qué crop y en qué clase. El video
del clip es un archivo DERIVADO que 5_render.py genera a partir de esas filas.

Ciclo de vida de una fila
-------------------------
  4_edit.py       crea/edita/borra filas            status = pending
  5_render.py     genera el video y lo mide         status = rendered
  6_validate.py   cambia la clase o descarta        discarded = 1
  7_split.py / 8_export.py  leen las filas activas (no descartadas, clase != Others)

Si se cambian los parámetros de un clip (ventana o crop) su fila vuelve a
`pending` y 5_render.py lo regenera desde el crudo; cambiar solo la clase no
obliga a regenerar nada.
"""
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from lib.common.csvstore import CsvTable
from lib.common.paths import CLIPS_CSV, ITEMS_CSV
from lib.common.schemas import CLIP_COLUMNS, ITEM_COLUMNS

PENDING, RENDERED = "pending", "rendered"
CROP_KEYS = ("left", "top", "right", "bottom")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clip_file(clips_dir: Path, clip_id: str) -> Path:
    """Ruta del video generado de un clip: {clips_dir}/{clip_id}.mp4."""
    return Path(clips_dir) / f"{clip_id}.mp4"


def label_of(klass: str) -> str:
    return "Normal" if klass == "Normal" else "Anomaly"


def crop_of(row: dict) -> dict | None:
    """La caja de crop de una fila como dict, o None si no tiene (fotograma completo)."""
    vals = [row.get(f"crop_{k}", "") for k in CROP_KEYS]
    if any(v == "" for v in vals):
        return None
    return {k: int(float(v)) for k, v in zip(CROP_KEYS, vals)}


def compute_render_key(row: dict) -> str:
    """Huella de lo que determina el contenido del video generado."""
    crop = crop_of(row)
    text = "|".join([
        row["raw_file"], f"{float(row['start_sec']):.3f}", f"{float(row['end_sec']):.3f}",
        "" if crop is None else ",".join(str(crop[k]) for k in CROP_KEYS), "v1",
    ])
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def to_api(row: dict) -> dict:
    """Fila -> dict tipado para la interfaz web."""
    start, end = float(row["start_sec"]), float(row["end_sec"])
    return {
        "clip_id": row["clip_id"], "item_id": row["item_id"], "country": row["country"],
        "class": row["class"], "label": row["label"],
        "start_sec": start, "end_sec": end, "duration": round(end - start, 3),
        "crop": crop_of(row), "status": row["status"],
        "discarded": row["discarded"] == "1", "rendered": row["status"] == RENDERED,
        "notes": row["notes"],
    }


def lookup_item(item_id: str, items_csv: Path = ITEMS_CSV) -> dict:
    """Metadatos del video fuente en items.csv (o {} si no existe el CSV o el item)."""
    if not Path(items_csv).exists():
        return {}
    return CsvTable(items_csv, ITEM_COLUMNS, "id").get(item_id) or {}


class ClipStore:
    def __init__(self, path: Path = CLIPS_CSV):
        self.path = Path(path)
        self._t = CsvTable(self.path, CLIP_COLUMNS, key="clip_id")      # modo compartido

    # ── Lectura ─────────────────────────────────────────────────────────

    def all(self) -> list[dict]:
        return self._t.rows()

    def get(self, clip_id: str) -> dict | None:
        return self._t.get(clip_id)

    def by_item(self, item_id: str) -> list[dict]:
        return sorted((r for r in self._t.rows() if r["item_id"] == item_id), key=lambda r: r["clip_id"])

    # ── Escritura ───────────────────────────────────────────────────────

    def add(self, item_id: str, country: str, raw_file: str, klass: str,
            start: float, end: float, crop: dict | None, items_csv: Path = ITEMS_CSV) -> dict:
        """Crea el clip siguiente del item (clip_id = {item_id}_clip_NNN) con status pending."""
        meta = lookup_item(item_id, items_csv)
        with self._t.transaction():
            used = [int(r["clip_id"].rsplit("_clip_", 1)[1]) for r in self._t.rows()
                    if r["item_id"] == item_id and "_clip_" in r["clip_id"]]
            clip_id = f"{item_id}_clip_{(max(used) + 1 if used else 0):03d}"
            row = {
                "clip_id": clip_id, "item_id": item_id, "country": country, "raw_file": raw_file,
                "class": klass, "label": label_of(klass),
                "start_sec": round(start, 3), "end_sec": round(end, 3),
                "status": PENDING, "discarded": "",
                "source_url": meta.get("url", ""), "title": meta.get("raw_title", ""),
                "created_at": _now(), "updated_at": _now(),
            }
            row.update({f"crop_{k}": (crop or {}).get(k, "") for k in CROP_KEYS})
            self._t.insert(row)
            return self._t.get(clip_id)

    def update(self, clip_id: str, **fields) -> dict:
        """Actualiza campos de un clip. Si cambian la ventana o el crop vuelve a `pending`."""
        with self._t.transaction():
            row = self._t.get(clip_id)
            updates = {k: v for k, v in fields.items() if k != "crop"}
            if "crop" in fields:
                crop = fields["crop"]
                updates.update({f"crop_{k}": (crop or {}).get(k, "") for k in CROP_KEYS})
            if "class" in updates:
                updates["label"] = label_of(updates["class"])
            merged = {**row, **updates}
            if compute_render_key(merged) != compute_render_key(row):
                updates["status"] = PENDING
            updates["updated_at"] = _now()
            return self._t.update(clip_id, updates)

    def mark_rendered(self, clip_id: str, rendered_key: str, meta: dict) -> bool:
        """
        Anota que el video del clip se generó con los parámetros `rendered_key` y guarda
        sus medidas (n_frames, fps, width, height, duration). Si mientras tanto se
        editaron los parámetros del clip (otra huella), no lo marca como generado y
        devuelve False: el clip sigue `pending` y se regenerará.
        """
        with self._t.transaction():
            row = self._t.get(clip_id)
            if row is None or compute_render_key(row) != rendered_key:
                return False
            self._t.update(clip_id, {**meta, "status": RENDERED, "render_key": rendered_key,
                                     "updated_at": _now()})
            return True

    def delete(self, clip_id: str) -> bool:
        return self._t.delete(clip_id)

    def relabel_item(self, item_id: str, new_class: str, skip=("Normal",)) -> int:
        """Reclasifica los clips de un item que no estén en `skip` ni ya en `new_class`."""
        return self._t.update_where(
            lambda r: r["item_id"] == item_id and r["class"] not in skip and r["class"] != new_class,
            {"class": new_class, "label": label_of(new_class), "updated_at": _now()})
