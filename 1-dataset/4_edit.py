"""
4_edit.py — PASO 4: sitio de edición. Se eligen, sobre los videos crudos, los
clips que formarán el dataset: ventana de recorte, crop opcional y clase. Sin
base de datos: las decisiones se guardan en data/metadata/clips.csv.

Aquí NO se genera ningún video. Cada clip es una fila de clips.csv
(item_id, clase, start_sec, end_sec, crop…) y el video se genera después, desde el
crudo, con 5_render.py. Por eso editar es siempre reversible y nunca recodifica
un clip ya recodificado.

Dos vistas, que se cambian desde la interfaz:

  Raw    todos los videos crudos en --raw-dir ({país}/{item_id}.{ext}, lo que
         escribe 3_download.py). Se ajusta una ventana de recorte (por defecto el
         video completo), una caja de crop opcional y una clase, y "Guardar clip"
         registra la fila. De un mismo video se pueden registrar varios clips
         (p. ej. el incidente y, aparte, un tramo tranquilo como Normal).

  Clips  solo los videos que ya tienen clips registrados, con filtro por clase.

En ambas vistas, un clip ya registrado se puede reclasificar, cargar en el editor
para cambiar su recorte/crop, o eliminar; y se puede reclasificar de una vez todos
los clips anómalos de un video. Si se cambia la ventana o el crop de un clip ya
generado, vuelve a "pendiente" y 5_render.py lo regenera.

Uso
---
    python 4_edit.py [--raw-dir data/1_raw] [--clips-csv data/metadata/clips.csv] [--port 5050]

Luego abre http://localhost:5050
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 1-dataset/

from lib.common.paths import CLIPS_CSV, ITEMS_CSV, RAW_DIR
from lib.editing.app import create_app
from lib.common.rawfiles import iter_raw_files


def main():
    parser = argparse.ArgumentParser(description="Sitio de edición: registrar clips sobre los videos crudos")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                        help="Raíz de los videos fuente, {país}/{item_id}.{ext} (por defecto: data/1_raw)")
    parser.add_argument("--clips-csv", type=Path, default=CLIPS_CSV,
                        help="CSV donde se guardan los clips decididos "
                             "(por defecto: data/metadata/clips.csv; se crea si no existe)")
    parser.add_argument("--items-csv", type=Path, default=ITEMS_CSV,
                        help="CSV de items del scraping, del que se copian título y URL de cada "
                             "video a su clip (opcional; por defecto: data/metadata/items.csv)")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    if not raw_dir.is_dir():
        sys.exit(f"--raw-dir {raw_dir} does not exist or is not a directory")

    app = create_app(raw_dir, args.clips_csv, args.items_csv)
    cfg = app.config["EDITING"]

    clips = cfg.clips.all()
    n_raw = sum(1 for _ in iter_raw_files(cfg.raw_dir))
    print(f"  Raw source : {cfg.raw_dir}  ({n_raw} videos)")
    print(f"  Clips CSV  : {cfg.clips.path}  ({len(clips)} clips, {len({c['item_id'] for c in clips})} items)")
    print(f"  Starting at  http://{args.host}:{args.port}")
    # threaded=True: cada petición es autocontenida (el estado vive en el CSV, con
    # bloqueo entre procesos), así que una consulta lenta no bloquea las demás.
    app.run(debug=True, port=args.port, host=args.host, threaded=True)


if __name__ == "__main__":
    main()
