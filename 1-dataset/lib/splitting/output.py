"""
Escritura y lectura de split.csv: el split asignado a cada item (una fila por item).

La clave de un item es la pareja (fuente, item_id), por eso se escribe con csv
directamente (con reemplazo atómico) y no con CsvTable, que tiene una sola columna clave.
"""
import csv
import os
from pathlib import Path

from lib.common.schemas import SPLIT_COLUMNS


def write_split_csv(path: Path, items, assignment) -> int:
    """Escribe split.csv (reemplazando el anterior): source, item_id, split, n_clips, windows."""
    rows = []
    for iid, classes_of_item in items.items():
        source, item_id = iid.split(":", 1)
        rows.append({
            "source": source, "item_id": item_id, "split": assignment[iid],
            "n_clips": sum(len(d["files"]) for d in classes_of_item.values()),
            "windows": sum(d["windows"] for d in classes_of_item.values()),
        })
    rows.sort(key=lambda r: (r["source"], r["item_id"]))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SPLIT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return len(rows)


def read_split_csv(path: Path) -> dict[tuple[str, str], str]:
    """{(fuente, item_id): split}. Vacío si el archivo no existe."""
    if not Path(path).is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["source"], r["item_id"]): r["split"] for r in csv.DictReader(f)}
