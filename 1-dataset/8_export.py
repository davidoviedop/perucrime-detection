"""
8_export.py — PASO 8: exporta todos los videos a la carpeta final del dataset.

Se ejecuta después de validar (6) y dividir (7). Arma
  data/3_dataset/{fuente}/{split}/{clase}/{archivo}.mp4
  data/3_dataset/manifest.csv
a partir de los CSV y de los videos generados:
  scrapped              los clips activos de clips.csv (con la clase que quedó tras la
                        validación) y sus videos de data/2_clips
  ucf-crime, vioperu    los archivos de data/external/{fuente}/{clase}/
  el split de cada item lo da split.csv (paso 7)

Esa es la estructura que lee el cargador de datos de la etapa de Modelado. La carpeta
final es un derivado: se puede borrar y volver a generar. Al repetir la exportación se
borran los archivos de la anterior que ya no corresponden (según su manifest.csv) y
se omiten los que ya están.

manifest.csv: item_id, clip_id, source, class, label, split, filename, output_path
(relativo a la carpeta final), start_sec, end_sec (ventana en el video crudo) y
source_url (vacíos en las fuentes externas).

Uso
---
    python 8_export.py                    # copia los archivos
    python 8_export.py --mode hardlink    # enlaces duros (sin duplicar espacio; mismo disco)
    python 8_export.py --dry-run          # solo muestra qué se haría
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 1-dataset/

from lib.common.clips import ClipStore
from lib.common.paths import CLIPS_CSV, CLIPS_DIR, DATASET_OUT, EXTERNAL_DIR, SPLIT_CSV
from lib.exporting.export import MissingSplitError, apply_export, plan_export, verify_export
from lib.splitting.config import SOURCES
from lib.splitting.output import read_split_csv
from lib.splitting.pool import UnrenderedClipsError


def main():
    parser = argparse.ArgumentParser(description="Exporta el dataset final a {fuente}/{split}/{clase}/")
    parser.add_argument("--out-dir", type=Path, default=DATASET_OUT,
                        help="Carpeta final del dataset (por defecto: data/3_dataset)")
    parser.add_argument("--clips-csv", type=Path, default=CLIPS_CSV,
                        help="CSV de clips (por defecto: data/metadata/clips.csv)")
    parser.add_argument("--clips-dir", type=Path, default=CLIPS_DIR,
                        help="Carpeta de los videos generados (por defecto: data/2_clips)")
    parser.add_argument("--external-dir", type=Path, default=EXTERNAL_DIR,
                        help="Carpeta de las fuentes externas (por defecto: data/external)")
    parser.add_argument("--split-csv", type=Path, default=SPLIT_CSV,
                        help="CSV con el split de cada item (por defecto: data/metadata/split.csv)")
    parser.add_argument("--sources", nargs="+", default=SOURCES,
                        help=f"Fuentes a exportar (por defecto: {' '.join(SOURCES)})")
    parser.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy",
                        help="Cómo colocar los videos en la carpeta final (por defecto: copy)")
    parser.add_argument("--dry-run", action="store_true", help="Solo muestra qué se haría")
    args = parser.parse_args()

    splits = read_split_csv(args.split_csv)
    if not splits:
        sys.exit(f"{args.split_csv} no existe o está vacío: ejecuta 7_split.py primero")
    try:
        plan = plan_export(ClipStore(args.clips_csv), args.clips_dir, args.external_dir, args.sources, splits)
    except (MissingSplitError, UnrenderedClipsError) as exc:
        sys.exit(str(exc))

    by = {}
    for e in plan:
        by[(e["source"], e["split"])] = by.get((e["source"], e["split"]), 0) + 1
    print(f"Exportando {len(plan)} videos a {args.out_dir}  (modo {args.mode})")
    for (src, split), n in sorted(by.items()):
        print(f"  {src:<10} {split:<5} {n:>5}")

    placed, stale = apply_export(args.out_dir, plan, args.mode, args.dry_run)
    if args.dry_run:
        print(f"[dry-run] se colocarían {placed} videos y se borrarían {stale} obsoletos")
        return
    print(f"Colocados: {placed}  |  obsoletos borrados: {stale}  |  manifest.csv: {args.out_dir / 'manifest.csv'}")
    if not verify_export(args.out_dir, plan, args.sources):
        sys.exit(1)


if __name__ == "__main__":
    main()
