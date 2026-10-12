"""
lib/scraping/store.py — almacén de items del scraping, respaldado por un CSV.

Una fila por video fuente descubierto (URL), en `data/metadata/items.csv`. Es el
estado de los pasos 1 a 3. Cada insert es un no-op si el item ya existe, así que
los scripts de descubrir / filtrar / descargar se pueden volver a ejecutar sin riesgo.

Flujo de estados de un item (cada script solo avanza las filas que le corresponden):

    1_discover  ->  discovered
    2_filter    ->  approved | rejected
    3_download  ->  downloaded | failed

La tabla se abre en modo exclusivo (ver lib/common/csvstore.py): los scripts de
scraping hacen miles de actualizaciones seguidas, así que los guardados se
agrupan cada AUTOSAVE_EVERY cambios y siempre se vuelcan al terminar. Si un
script se interrumpe de golpe (kill -9) se pueden perder, como máximo, los
últimos AUTOSAVE_EVERY cambios; un Ctrl+C o una excepción sí guardan todo.
"""
from datetime import datetime, timezone
from pathlib import Path

from lib.common.csvstore import CsvTable
from lib.common.schemas import ITEM_COLUMNS


AUTOSAVE_EVERY = 25


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ItemStore:
    """
    Uso
    ---
        store = ItemStore(Path("data/metadata/items.csv"))
        ...
        store.close()      # también se vuelca solo al salir del proceso
    """

    def __init__(self, path: Path):
        self._table = CsvTable(path, ITEM_COLUMNS, key="id", exclusive=True,
                               autosave_every=AUTOSAVE_EVERY)
        self.path = Path(path)

    # ── Escritura ───────────────────────────────────────────────────────

    def insert_item(self, item: dict) -> bool:
        """Inserta un item nuevo. Devuelve True si se insertó, False si era duplicado."""
        item.setdefault("created_at", _now())
        item.setdefault("updated_at", _now())
        return self._table.insert(item)

    def update_item(self, item_id: str, updates: dict) -> None:
        """Mezcla `updates` en la fila del item. updated_at se actualiza solo."""
        updates["updated_at"] = _now()
        self._table.update(item_id, updates)

    def requeue_rejected(self) -> None:
        """Devuelve todo item rechazado a 'discovered' para volver a filtrarlo."""
        self._table.update_where(
            lambda r: r["status"] == "rejected",
            {"status": "discovered", "filter_reason": "", "provisional_label": "",
             "ucf_category": "", "updated_at": _now()})

    # ── Lectura ─────────────────────────────────────────────────────────

    def exists(self, item_id: str) -> bool:
        return self._table.exists(item_id)

    def get_by_status(self, status: str, limit: int = None) -> list[dict]:
        """Todos los items con ese estado (los más antiguos primero), con tope opcional."""
        rows = sorted((r for r in self._table.rows() if r["status"] == status),
                      key=lambda r: r["created_at"])
        return rows[:limit] if limit else rows

    def status_summary(self) -> dict[str, int]:
        """Asocia cada estado con su cantidad de items."""
        counts: dict[str, int] = {}
        for r in self._table.rows():
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return counts

    def label_counts(self) -> list[dict]:
        """Distribución de provisional_label, de la más común a la menos común."""
        return self._counts("provisional_label")

    def reject_reason_counts(self) -> list[dict]:
        """Distribución de filter_reason, de la más común a la menos común."""
        return self._counts("filter_reason")

    def _counts(self, column: str) -> list[dict]:
        counts: dict[str, int] = {}
        for r in self._table.rows():
            if r[column]:
                counts[r[column]] = counts.get(r[column], 0) + 1
        return [{column: k, "n": n} for k, n in sorted(counts.items(), key=lambda kv: -kv[1])]

    # ── Ciclo de vida ───────────────────────────────────────────────────

    def flush(self) -> None:
        self._table.flush()

    def close(self) -> None:
        self._table.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
