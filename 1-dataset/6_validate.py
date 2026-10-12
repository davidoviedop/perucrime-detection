"""
6_validate.py — PASO 6: sitio de validación humana. Sin base de datos: lee y
escribe data/metadata/clips.csv.

Se revisan los clips ya generados por 5_render.py (data/2_clips/{clip_id}.mp4) y,
para cada uno, solo se puede hacer una de dos cosas:

  - asignarle la clase correcta     (un clip, los clips anómalos de un video, o todos
                                     los videos que estén listados en la barra lateral)
  - descartarlo                     (queda marcado en el CSV, no se borra, y se puede
                                     restaurar; los descartados no se dividen ni se exportan)

La ventana de recorte y el crop NO se tocan aquí: se deciden en 4_edit.py y, si hay
que cambiarlos, se vuelve allí y se ejecuta de nuevo 5_render.py. Esta etapa solo
cubre los clips de la fuente `scrapped` (los de clips.csv).

Terminada la validación, el siguiente paso es 7_split.py (asigna el split) y luego
8_export.py (arma la carpeta final).

Uso
---
    python 6_validate.py [--clips-csv data/metadata/clips.csv] [--clips-dir data/2_clips] [--port 5051]

Luego abre http://localhost:5051
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 1-dataset/

from lib.common.paths import CLIPS_CSV, CLIPS_DIR
from lib.validation.app import create_app


def main():
    parser = argparse.ArgumentParser(description="Sitio de validación humana: asignar clase o descartar clips")
    parser.add_argument("--clips-csv", type=Path, default=CLIPS_CSV,
                        help="CSV de clips (por defecto: data/metadata/clips.csv)")
    parser.add_argument("--clips-dir", type=Path, default=CLIPS_DIR,
                        help="Carpeta de los videos generados por 5_render.py (por defecto: data/2_clips)")
    parser.add_argument("--port", type=int, default=5051)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if not args.clips_csv.is_file():
        sys.exit(f"--clips-csv {args.clips_csv} does not exist: registra clips con 4_edit.py y genéralos con 5_render.py")

    app = create_app(args.clips_csv, args.clips_dir)
    cfg = app.config["VALIDATION"]
    rows = cfg.clips.all()
    print(f"  Clips CSV : {cfg.clips.path}  ({len(rows)} clips, {len({r['item_id'] for r in rows})} items)")
    print(f"  Videos    : {cfg.clips_dir}")
    print(f"  Starting at  http://{args.host}:{args.port}")
    app.run(debug=True, port=args.port, host=args.host, threaded=True)


if __name__ == "__main__":
    main()
