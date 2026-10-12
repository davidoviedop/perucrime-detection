"""
lib/common/csvstore.py — tabla respaldada por un archivo CSV.

Reemplaza a la base de datos: el estado del pipeline vive en archivos CSV
legibles (UTF-8, con cabecera, una fila por registro), igual que los metadatos
que acompañan a los datasets públicos. Se pueden abrir en cualquier editor de
hojas de cálculo, versionar, revisar y compartir tal cual.

Garantías
---------
* Escritura atómica: se escribe a un archivo temporal, se hace fsync y se
  reemplaza el original con os.replace; un corte a mitad de escritura nunca deja
  un CSV a medias.
* Bloqueo entre procesos (fcntl.flock sobre un archivo `.lock` al lado):
    - modo compartido (por defecto): cada operación toma el bloqueo, recarga el
      archivo si otro proceso lo cambió, aplica el cambio y guarda. Sirve para
      las apps web y los scripts que pueden convivir (p. ej. dejar abierto el
      sitio de edición mientras corre 5_render.py).
    - modo exclusivo (`exclusive=True`): el bloqueo se toma al abrir y se mantiene
      hasta cerrar; los guardados se agrupan (`autosave_every`). Sirve para los
      scripts de scraping, que hacen miles de actualizaciones seguidas. Si otro
      proceso ya tiene la tabla, falla con un mensaje claro.
* Todos los valores se guardan como texto; los módulos de dominio (items, clips)
  convierten a número donde hace falta. Una celda vacía es ''.
"""
import atexit
import csv
import os
import threading
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:          # Windows: sin bloqueo entre procesos (sí hay bloqueo entre hilos)
    fcntl = None


def _cell(value) -> str:
    """Convierte un valor a la celda de texto del CSV."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else ""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") if value == value else ""
    return str(value)


class TableLockedError(RuntimeError):
    """Otro proceso tiene la tabla en modo exclusivo."""


class CsvTable:
    def __init__(self, path, columns, key, exclusive=False, autosave_every=1):
        self.path = Path(path)
        self.columns = list(columns)
        self.key = key
        self.exclusive = exclusive
        self.autosave_every = max(1, int(autosave_every))

        self._rows: dict[str, dict] = {}
        self._dirty = 0
        self._stat = None
        self._tlock = threading.RLock()
        self._depth = 0
        self._lockfd = None
        self._closed = False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if exclusive:
            self._flock(blocking=False)
            self._load()
            atexit.register(self.close)
        else:
            with self._tx():
                pass

    # ── Bloqueo ─────────────────────────────────────────────────────────

    def _lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    def _flock(self, blocking: bool) -> None:
        if fcntl is None:
            return
        fd = os.open(self._lock_path(), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            os.close(fd)
            raise TableLockedError(
                f"{self.path.name} está en uso por otro proceso (modo exclusivo). "
                "Espera a que termine o ciérralo antes de volver a ejecutar.") from None
        self._lockfd = fd

    def _funlock(self) -> None:
        if self._lockfd is not None:
            if fcntl is not None:
                fcntl.flock(self._lockfd, fcntl.LOCK_UN)
            os.close(self._lockfd)
            self._lockfd = None

    def transaction(self):
        """Agrupa varias operaciones bajo un mismo bloqueo (lectura-modificación-escritura atómica)."""
        return self._tx()

    @contextmanager
    def _tx(self):
        """Transacción: bloquea, recarga si el archivo cambió y, al salir, guarda si hubo cambios."""
        with self._tlock:
            outer = self._depth == 0
            self._depth += 1
            took_lock = False
            try:
                if outer and not self.exclusive:
                    self._flock(blocking=True)
                    took_lock = True
                    self._load_if_changed()
                try:
                    yield
                except BaseException:
                    if outer and not self.exclusive:
                        # se descartan los cambios en memoria de la transacción fallida
                        self._dirty, self._stat = 0, None
                    raise
                if outer and self._dirty and (not self.exclusive or self._dirty >= self.autosave_every):
                    self._save()
            finally:
                self._depth -= 1
                if took_lock:
                    self._funlock()

    # ── Carga y guardado ────────────────────────────────────────────────

    def _file_stat(self):
        try:
            st = self.path.stat()
            return (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            return None

    def _load_if_changed(self) -> None:
        if self._file_stat() != self._stat:
            self._load()

    def _load(self) -> None:
        self._rows = {}
        self._stat = self._file_stat()
        if self._stat is None or self._stat[1] == 0:
            return
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for extra in (reader.fieldnames or []):
                if extra not in self.columns:
                    self.columns.append(extra)      # se conservan las columnas añadidas por fuera
            for raw in reader:
                row = {c: (raw.get(c) or "") for c in self.columns}
                self._rows[row[self.key]] = row

    def _save(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(self._rows.values())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        self._stat = self._file_stat()
        self._dirty = 0

    def flush(self) -> None:
        """Guarda los cambios pendientes (útil en modo exclusivo)."""
        with self._tlock:
            if self._dirty:
                self._save()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.flush()
        finally:
            if self.exclusive:
                self._funlock()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── Operaciones ─────────────────────────────────────────────────────

    def _normalize(self, row: dict) -> dict:
        unknown = set(row) - set(self.columns)
        if unknown:
            raise KeyError(f"columnas desconocidas para {self.path.name}: {sorted(unknown)}")
        return {c: _cell(row.get(c)) for c in self.columns}

    def exists(self, key: str) -> bool:
        with self._tx():
            return key in self._rows

    def get(self, key: str) -> dict | None:
        with self._tx():
            row = self._rows.get(key)
            return dict(row) if row is not None else None

    def rows(self) -> list[dict]:
        """Copia de todas las filas, en el orden del archivo."""
        with self._tx():
            return [dict(r) for r in self._rows.values()]

    def __len__(self) -> int:
        with self._tx():
            return len(self._rows)

    def insert(self, row: dict) -> bool:
        """Agrega una fila. Devuelve False (sin cambios) si la clave ya existe."""
        with self._tx():
            norm = self._normalize(row)
            if not norm[self.key]:
                raise ValueError(f"la fila no tiene valor para la clave '{self.key}'")
            if norm[self.key] in self._rows:
                return False
            self._rows[norm[self.key]] = norm
            self._dirty += 1
            return True

    def update(self, key: str, updates: dict) -> dict:
        """Mezcla `updates` en la fila `key` y devuelve la fila resultante."""
        with self._tx():
            row = self._rows[key]
            row.update(self._normalize({**row, **updates}))
            self._dirty += 1
            return dict(row)

    def delete(self, key: str) -> bool:
        with self._tx():
            if key not in self._rows:
                return False
            del self._rows[key]
            self._dirty += 1
            return True

    def update_where(self, predicate, updates: dict) -> int:
        """Aplica `updates` a todas las filas que cumplan `predicate(fila)`. Devuelve cuántas."""
        with self._tx():
            n = 0
            for row in self._rows.values():
                if predicate(row):
                    row.update(self._normalize({**row, **updates}))
                    n += 1
            if n:
                self._dirty += n
            return n
