"""
7_split.py — PASO 7: división train/val/test balanceada por cantidad de ventanas y
con control de diversidad, sobre todas las fuentes (scrapped, ucf-crime, vioperu).
Escribe el resultado en data/metadata/split.csv (una fila por item); no mueve ni copia
ningún video: de eso se encarga 8_export.py.

Por qué ventanas y no clips: la duración de los clips varía mucho entre fuentes
(los videos crudos de ucf-crime son mucho más largos que los clips de scrapped o los
segmentos uniformes de ~5 s de vioperu), así que una división 70/15/15 por cantidad
de CLIPS no da 70/15/15 en las ventanas deslizantes (seq_len=16) con las que se
entrenan los modelos. Este script balancea directamente la cantidad de ventanas, con
la fórmula exacta de la etapa de Modelado (ver lib/splitting/config.py).

Entradas
  scrapped              clips.csv: los clips activos (no descartados, clase != Others).
                        Sus ventanas salen de n_frames, medido por 5_render.py.
  ucf-crime, vioperu    carpetas sueltas data/external/{fuente}/{clase}/*.mp4

No se recorta nada: lo único que se decide es a qué split va cada ITEM (todos los clips
de un video fuente se quedan juntos, así que no hay fuga entre train y test). "Others"
queda fuera por completo.

Es seguro repetirlo (tras agregar clips o tras otra pasada de validación): reoptimiza
desde el pool completo y reemplaza split.csv.

Uso
---
    python 7_split.py --stats      # solo reporta el pool
    python 7_split.py --dry-run    # calcula y muestra la división sin escribir split.csv
    python 7_split.py              # calcula y escribe split.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 1-dataset/

from lib.common.clips import ClipStore
from lib.common.paths import CLIPS_CSV, CLIPS_DIR, EXTERNAL_DIR, SPLIT_CSV
from lib.splitting.assign import best_assignment
from lib.splitting.config import DEFAULT_RESTARTS, SOURCES
from lib.splitting.output import write_split_csv
from lib.splitting.pool import UnrenderedClipsError, scan_pool
from lib.splitting.report import print_diversity, print_pool_stats, print_report


def main():
    parser = argparse.ArgumentParser(description="División balanceada por ventanas y con control de diversidad")
    parser.add_argument("--clips-csv", type=Path, default=CLIPS_CSV,
                        help="CSV de clips (por defecto: data/metadata/clips.csv)")
    parser.add_argument("--clips-dir", type=Path, default=CLIPS_DIR,
                        help="Carpeta de los videos generados (por defecto: data/2_clips)")
    parser.add_argument("--external-dir", type=Path, default=EXTERNAL_DIR,
                        help="Carpeta de las fuentes externas, {fuente}/{clase}/*.mp4 (por defecto: data/external)")
    parser.add_argument("--split-csv", type=Path, default=SPLIT_CSV,
                        help="CSV de salida con el split de cada item (por defecto: data/metadata/split.csv)")
    parser.add_argument("--sources", nargs="+", default=SOURCES,
                        help=f"Fuentes a juntar (por defecto: {' '.join(SOURCES)})")
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS,
                        help=f"Reinicios aleatorios de la búsqueda de asignación (por defecto: {DEFAULT_RESTARTS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo calcula y muestra: no escribe split.csv")
    parser.add_argument("--stats", action="store_true",
                        help="Reporta ventanas/items por clase y termina, sin dividir")
    args = parser.parse_args()

    print(f"Pool: clips.csv ({args.clips_csv}) + {args.external_dir}  [{', '.join(args.sources)}; 'Others' excluded]")
    try:
        items = scan_pool(ClipStore(args.clips_csv), args.clips_dir, args.external_dir, args.sources)
    except UnrenderedClipsError as exc:
        sys.exit(str(exc))

    item_class_windows = {iid: {c: d["windows"] for c, d in classes_of_item.items()}
                           for iid, classes_of_item in items.items()}
    classes = sorted({c for cw in item_class_windows.values() for c in cw})
    print(f"Pool: {len(items)} items across classes {classes}")

    if args.stats:
        print_pool_stats(classes, item_class_windows)
        return

    sc, assignment, window_counts, item_counts = best_assignment(item_class_windows, classes, args.restarts)
    print(f'Best objective score: {sc:.4f} (window-count + item-diversity, normalized, lower=better)')
    print_report(classes, item_class_windows, window_counts, item_counts)
    print_diversity(item_class_windows, assignment, classes)

    if args.dry_run:
        print("\n[dry-run] split.csv no se escribió")
        return
    n = write_split_csv(args.split_csv, items, assignment)
    print(f"\nsplit.csv escrito: {args.split_csv} ({n} items)")
    print("Siguiente paso: python 8_export.py")


if __name__ == "__main__":
    main()
