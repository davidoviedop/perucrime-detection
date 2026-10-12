"""
5_render.py — PASO 5: genera los videos de los clips a partir de data/metadata/clips.csv.

Para cada clip registrado en el paso 4 corta el video CRUDO original con los
parámetros guardados (ventana de recorte + crop) y escribe
data/2_clips/{clip_id}.mp4. Luego mide el video generado (fotogramas, fps, tamaño,
duración) y lo anota en el mismo CSV; esas medidas son las que usa 7_split.py para
contar ventanas.

Es incremental: solo genera los clips pendientes, los que cambiaron de parámetros
desde la última vez y los que perdieron su archivo. Los clips descartados en la
validación se omiten. Cada clip se corta siempre del crudo (nunca de otro clip),
así que volver a generar no degrada la calidad.

Uso
---
    python 5_render.py                    # genera lo que falte
    python 5_render.py --dry-run          # solo lista qué se generaría
    python 5_render.py --force            # regenera todo
    python 5_render.py --item <item_id>   # solo los clips de un video
    python 5_render.py --prune            # además borra los videos huérfanos de data/2_clips
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 1-dataset/

from lib.common.clips import ClipStore
from lib.common.paths import CLIPS_CSV, CLIPS_DIR, RAW_DIR
from lib.rendering.render import render_clip, render_reason


def main():
    parser = argparse.ArgumentParser(description="Genera los videos de los clips desde clips.csv")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                        help="Raíz de los videos crudos, {país}/{item_id}.{ext} (por defecto: data/1_raw)")
    parser.add_argument("--clips-dir", type=Path, default=CLIPS_DIR,
                        help="Carpeta de salida de los videos de clip (por defecto: data/2_clips)")
    parser.add_argument("--clips-csv", type=Path, default=CLIPS_CSV,
                        help="CSV de clips (por defecto: data/metadata/clips.csv)")
    parser.add_argument("--item", default=None, help="Generar solo los clips de este item_id")
    parser.add_argument("--force", action="store_true", help="Regenerar también los clips que están al día")
    parser.add_argument("--workers", type=int, default=2,
                        help="Cortes de ffmpeg en paralelo (por defecto: 2)")
    parser.add_argument("--dry-run", action="store_true", help="Solo lista qué se generaría")
    parser.add_argument("--prune", action="store_true",
                        help="Borra de --clips-dir los videos que no corresponden a ningún clip activo")
    args = parser.parse_args()

    store = ClipStore(args.clips_csv)
    rows = [r for r in store.all() if r["discarded"] != "1"]
    if args.item:
        rows = [r for r in rows if r["item_id"] == args.item]

    todo = []
    for r in rows:
        reason = "forzado" if args.force else render_reason(r, args.clips_dir)
        if reason:
            todo.append((r, reason))

    print(f"Clips activos: {len(rows)}  |  por generar: {len(todo)}  |  al día: {len(rows) - len(todo)}")
    if args.dry_run:
        for r, reason in todo:
            print(f"  [dry-run] {r['clip_id']}  ({reason})")
    else:
        failed = 0

        def work(item):
            row, reason = item
            ok, detail = render_clip(row, args.raw_dir, args.clips_dir, store)
            print(f"  {'OK ' if ok else 'ERR'} {row['clip_id']}  ({reason}) {detail}")
            return ok

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            failed = sum(1 for ok in ex.map(work, todo) if not ok)
        print(f"Generados: {len(todo) - failed}  |  con error: {failed}")

    # Videos en --clips-dir que no corresponden a ningún clip activo
    active_ids = {r["clip_id"] for r in store.all() if r["discarded"] != "1"}
    orphans = sorted(p for p in args.clips_dir.glob("*.mp4") if p.stem not in active_ids) if args.clips_dir.is_dir() else []
    if orphans:
        print(f"Videos huérfanos en {args.clips_dir}: {len(orphans)}" + ("" if args.prune else "  (usa --prune para borrarlos)"))
        if args.prune and not args.dry_run:
            for p in orphans:
                p.unlink()
            print(f"  borrados: {len(orphans)}")


if __name__ == "__main__":
    main()
